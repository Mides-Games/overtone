"""Beat This!: the model session, its activation, beats, judge and vote."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

from overtone_paths import app_version, data_root

from . import data_model, params, phrases, pipeline, red_lines


def _beat_this_frames(heard, sr: int, hop: int, frames: int) -> np.ndarray | None:
    """Beat This!'s beats as onset-envelope frames, or None without them."""
    if heard is None:
        return None
    beats = heard()
    if beats is None or len(beats) < 8:
        return None
    out = np.unique(np.round(np.asarray(beats, dtype=float) * sr / hop).astype(int))
    out = out[(out >= 0) & (out < frames)]
    return out if out.size >= 8 else None


def _beat_this_activation(y: np.ndarray, sr: int, session=None) -> np.ndarray | None:
    """Beat This!'s beat probability per 50 fps frame, or None without the model."""
    if session is None:
        if not params._BEAT_THIS_JUDGE:
            params._BEAT_THIS_JUDGE.append(_load_beat_this())
        session = params._BEAT_THIS_JUDGE[0]
    if session is None or y.size == 0:
        return None
    import soxr
    signal = soxr.resample(np.asarray(y, dtype=np.float64), sr, params.BEAT_THIS_SR) \
        if sr != params.BEAT_THIS_SR else np.asarray(y, dtype=np.float64)
    logits = _beat_this_logits(session, _beat_this_spectrogram(signal)).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def _beat_this_judge_beats(y: np.ndarray, sr: int):
    """Beat This!'s beats (s) by the judge's own session (the vote's switch
    aside); None without the model or the runtime."""
    if not params._BEAT_THIS_JUDGE:
        params._BEAT_THIS_JUDGE.append(_load_beat_this())
    if params._BEAT_THIS_JUDGE[0] is None:
        return None
    key = (id(y), int(y.size), int(sr))
    if params._BEAT_THIS_HEARD and params._BEAT_THIS_HEARD[0][0] == key and params._BEAT_THIS_HEARD[0][1] is y:
        return params._BEAT_THIS_HEARD[0][2]
    beats = _beat_this_beats(y, sr, session=params._BEAT_THIS_JUDGE[0])
    params._BEAT_THIS_HEARD[:] = [(key, y, beats)]
    return beats


def _song_octave_by_beat_this(analysis: Analysis, heard) -> Analysis:
    """Every red line at Beat This!'s octave when it hears the song near 2x
    or 1/2 of its own tempo (SONG_OCTAVE_BEAT_THIS); a halved line keeps the
    phase nearest Beat This!'s beats."""
    points = list(getattr(analysis, "points", None) or [])
    if not params.SONG_OCTAVE_BEAT_THIS or not points:
        return analysis
    duration_ms = float(analysis.duration) * 1000.0
    song = red_lines._song_tempo(points, duration_ms)
    found = heard() if song else None
    if found is None or len(found) < 16:
        return analysis
    beats = np.sort(np.asarray(found, dtype=float) * 1000.0)
    tempo = 60000.0 / float(np.median(np.diff(beats)))
    ratio = tempo / song
    factor = (2.0 if abs(ratio / 2.0 - 1.0) <= params.LOCAL_OCTAVE_SLACK
              else 0.5 if abs(ratio / 0.5 - 1.0) <= params.LOCAL_OCTAVE_SLACK else None)
    if factor is None:
        return analysis
    out = []
    for n, point in enumerate(points):
        end = points[n + 1].offset_ms if n + 1 < len(points) else duration_ms
        target = point.bpm * factor
        offset = point.offset_ms
        inside = beats[(beats >= point.offset_ms) & (beats <= end)]
        if factor < 1.0 and inside.size >= 2:
            period = 60000.0 / target

            def distance(start: float) -> float:
                grid = start + np.arange(max(1, int((end - start) / period) + 1)) * period
                i = np.clip(np.searchsorted(inside, grid), 1, inside.size - 1)
                return float(np.median(np.minimum(np.abs(inside[i - 1] - grid),
                                                  np.abs(inside[i] - grid))))

            offset = min((point.offset_ms, point.offset_ms + 60000.0 / point.bpm), key=distance)
        out.append(replace(point, bpm=target, offset_ms=offset))
    return replace(analysis, points=out)


def _song_triplet_by_beat_this(analysis: Analysis, heard) -> Analysis:
    """Every red line at 3/2 of its tempo when Beat This! hears the song there
    (SONG_TRIPLET_BEAT_THIS).

    A song in a shuffled or dotted rhythm has as many attacks on a grid at
    2/3 of its beat as on the beat itself, and the core settles on the slower
    one: 131.33 for a song mapped at 197. No octave rule can find it, the two
    are not octaves apart. When the network's tempo is within
    SONG_TRIPLET_SLACK of 3/2 of the song's own, every line moves to 3/2 of
    its BPM; its beat stays where it was or moves half a new beat, whichever
    puts the grid nearer the network's beats. Only this way round: where the
    network hears 2/3 or 3/4 of the engine's tempo the maps side with the
    engine (anime-single, 2026-11-11)."""
    points = list(getattr(analysis, "points", None) or [])
    if not params.SONG_TRIPLET_BEAT_THIS or not points or any(getattr(p, "manual", False) for p in points):
        return analysis
    duration_ms = float(analysis.duration) * 1000.0
    song = red_lines._song_tempo(points, duration_ms)
    found = heard() if song else None
    if found is None or len(found) < 16:
        return analysis
    beats = np.sort(np.asarray(found, dtype=float) * 1000.0)
    tempo = 60000.0 / float(np.median(np.diff(beats)))
    if abs(tempo / (1.5 * song) - 1.0) > params.SONG_TRIPLET_SLACK:
        return analysis
    out = []
    for n, point in enumerate(points):
        end = points[n + 1].offset_ms if n + 1 < len(points) else duration_ms
        target = point.bpm * 1.5
        period = 60000.0 / target
        inside = beats[(beats >= point.offset_ms) & (beats <= end)]
        offset = point.offset_ms
        if inside.size >= 2:
            def distance(start: float) -> float:
                miss = (inside - start) / period
                return float(np.median(np.abs(miss - np.round(miss)))) * period

            offset = min((point.offset_ms, point.offset_ms + 0.5 * period), key=distance)
        out.append(replace(point, bpm=target, offset_ms=offset))
    lines = [(float(p.offset_ms), float(p.bpm)) for p in out]
    return replace(analysis, points=out, beats=red_lines._beats_of(lines, float(analysis.duration)))


def _song_double_by_beat_this(analysis: Analysis, heard) -> Analysis:
    """Every red line at twice its tempo when Beat This! hears the song there
    and the song's own tempo is SONG_DOUBLE_FLOOR_BPM or more
    (SONG_DOUBLE_BEAT_THIS).

    The octave cap and the map range read a song mapped at 240-280 at half.
    The network hearing an octave apart is no evidence by itself: it counts
    most fast songs at half, and the maps side with the engine there (69 of
    75 songs). Hearing double is rare and, from 120 up, was the map's tempo
    every time (three songs of three, dev only; the two under 110 were
    mapped at the engine's tempo). Every beat of the line stays a beat."""
    points = list(getattr(analysis, "points", None) or [])
    if not params.SONG_DOUBLE_BEAT_THIS or not points or any(getattr(p, "manual", False) for p in points):
        return analysis
    duration_ms = float(analysis.duration) * 1000.0
    song = red_lines._song_tempo(points, duration_ms)
    if not song or song < params.SONG_DOUBLE_FLOOR_BPM:
        return analysis
    found = heard()
    if found is None or len(found) < 16:
        return analysis
    beats = np.sort(np.asarray(found, dtype=float) * 1000.0)
    tempo = 60000.0 / float(np.median(np.diff(beats)))
    if abs(tempo / (2.0 * song) - 1.0) > params.LOCAL_OCTAVE_SLACK:
        return analysis
    out = [replace(point, bpm=point.bpm * 2.0) for point in points]
    lines = [(float(p.offset_ms), float(p.bpm)) for p in out]
    return replace(analysis, points=out, beats=red_lines._beats_of(lines, float(analysis.duration)))


def _beat_grid_cost(lines: list[tuple[float, float]], beats_ms: np.ndarray, start: float, stop: float):
    """Median distance (ms) from each beat in [start, stop) to the grid of
    the line it falls in; None with fewer than 16 beats."""
    b = beats_ms[(beats_ms >= start) & (beats_ms < stop)]
    if b.size < 16:
        return None
    starts = np.asarray([o for o, _b in lines])
    idx = np.clip(np.searchsorted(starts, b, side="right") - 1, 0, len(lines) - 1)
    o = starts[idx]
    step = 60000.0 / np.asarray([bp for _o, bp in lines])[idx]
    return float(np.median(np.abs(b - o - np.round((b - o) / step) * step)))


def _heard_bpm(b: np.ndarray, offset: float, bpm: float):
    """The tempo of beats counted on a line's grid (a least-squares slope), or
    None under 16 beats or when they do not sit on a steady grid (spread over
    COLLAPSE_LINES_STEADY of a beat). Beat This!'s 20 ms frames average out
    over a line."""
    if b.size < 16:
        return None
    step = 60000.0 / bpm
    k = np.round((b - offset) / step)
    if np.ptp(k) < 8:
        return None
    fit = np.polyfit(k, b, 1)
    if float(np.std(b - np.polyval(fit, k))) > params.COLLAPSE_LINES_STEADY * step or fit[0] <= 0:
        return None
    return 60000.0 / float(fit[0])


def _collapse_by_beat_this(lines: list[tuple[float, float]], beats_ms: np.ndarray, end_ms: float,
                           meters: list[int] | None = None):
    """The lines with those at one tempo on one grid (COLLAPSE_LINES_BEAT_THIS),
    as (offset, bpm, index of the line each came from), or None.

    The engine sometimes writes a single-tempo song as several lines at
    nearly one BPM, each a few ms out of phase with the last; the map has
    one. When lines within COLLAPSE_LINES_AGREE_PCT of the longest line's BPM
    cover COLLAPSE_LINES_SHARE of the song, they all take that line's grid
    (a run of them becomes one line, on the grid's beat nearest where it
    starts) if Beat This!'s beats inside them sit on it within
    COLLAPSE_LINES_SLACK times their distance to their own grids. Lines at
    another tempo stay as they are, and a line where the meter changes stays
    a line, on the one grid: a signature change at one tempo is a red line."""
    spans = [max(0.0, (lines[n + 1][0] if n + 1 < len(lines) else end_ms) - o) for n, (o, _b) in enumerate(lines)]
    total = sum(spans)
    if total <= 0.0:
        return None
    o, bpm = lines[int(np.argmax(spans))]
    agree = [abs(b / bpm - 1.0) * 100.0 <= params.COLLAPSE_LINES_AGREE_PCT for _o, b in lines]
    if sum(agree) < 2 or sum(s for s, a in zip(spans, agree) if a) < params.COLLAPSE_LINES_SHARE * total:
        return None
    step = 60000.0 / bpm
    first = o - np.floor(o / step) * step
    meters = list(meters) if meters is not None else [4] * len(lines)
    out: list[tuple[float, float, int]] = []
    for n, (lo, lb) in enumerate(lines):
        if not agree[n]:
            out.append((lo, lb, n))
        elif n == 0:
            out.append((first, bpm, n))
        elif not agree[n - 1] or meters[n] != meters[n - 1]:
            out.append((first + np.round((lo - first) / step) * step, bpm, n))
    if len(out) == len(lines):
        return None
    # the beats inside the lines that agree, on their own grids and on the one
    inside = np.zeros(beats_ms.size, dtype=bool)
    for n, (lo, _lb) in enumerate(lines):
        if agree[n]:
            stop = lines[n + 1][0] if n + 1 < len(lines) else end_ms
            inside |= (beats_ms >= (lo if n else -np.inf)) & (beats_ms < stop)
    many = _beat_grid_cost(lines, beats_ms[inside], -np.inf, end_ms)
    single = _beat_grid_cost([(first, bpm)], beats_ms[inside], -np.inf, end_ms)
    if many is None or single is None or single > params.COLLAPSE_LINES_SLACK * many:
        return None
    # The song's median would hide a stretch the one grid misses. A long line
    # keeps a real phase reset (Noble: 131 s left 125 ms off), and any line
    # whose beats keep its own tempo keeps a real small change (200 -> 203.5
    # BPM); a line merely out of phase has beats at the one grid's tempo.
    for n, (lo, lb) in enumerate(lines):
        stop = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        if not agree[n]:
            continue
        heard = _heard_bpm(beats_ms[(beats_ms >= lo) & (beats_ms < stop)], lo, lb)
        if heard is not None and abs(heard / bpm - 1.0) * 100.0 > params.COLLAPSE_LINES_TEMPO_PCT \
                and abs(heard / lb - 1.0) < abs(heard / bpm - 1.0):
            return None
        if (stop - lo) / 1000.0 < params.COLLAPSE_LINES_LONG_S:
            continue
        own = _beat_grid_cost([(lo, lb)], beats_ms, lo, stop)
        new = _beat_grid_cost([(first, bpm)], beats_ms, lo, stop)
        if own is not None and new is not None \
                and new > max(params.COLLAPSE_LINES_SLACK * own, own + params.COLLAPSE_LINES_FLOOR_MS):
            return None
    return [(float(x), float(y), k) for x, y, k in out]


def _half_beat_by_beat_this(lines: list[tuple[float, float]], beats_ms: np.ndarray, end_ms: float,
                            meters: list[int] | None = None):
    """The lines, each moved half a beat where Beat This!'s beats inside it
    sit on its half beats (their median distance under PHASE_BEAT_THIS_GAIN
    of the distance to its beats), and whether any moved. A line in a meter
    other than 4 also needs PHASE_BEAT_THIS_SHARE of the beats nearer the
    half beat: in a signature region's six-beat bars the network hears a
    pulse twice the line's tempo and switches between its halves, and about
    60 % of its beats on one half says nothing about where the beat is."""
    meters = list(meters) if meters is not None else [4] * len(lines)
    out, moved = [], False
    for n, (offset, bpm) in enumerate(lines):
        stop = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        inside = beats_ms[(beats_ms >= offset) & (beats_ms < stop)]
        step = 60000.0 / bpm
        if inside.size >= params.PHASE_BEAT_THIS_MIN_BEATS:
            on = np.mod((inside - offset) / step, 1.0)
            on = np.minimum(on, 1.0 - on)
            half = 0.5 - on
            if float(np.median(half)) < params.PHASE_BEAT_THIS_GAIN * float(np.median(on)) \
                    and (meters[n] == 4 or float(np.mean(half < on)) >= params.PHASE_BEAT_THIS_SHARE):
                offset += 0.5 * step
                moved = True
        out.append((offset, bpm))
    return out, moved


def _phase_by_beat_this(analysis: Analysis, heard) -> Analysis:
    """Each red line moved half a beat where Beat This!'s beats sit on its
    half beats (PHASE_BEAT_THIS). Fast punk and metal put the loud snare
    between the map's beats, and the attack-weighted phase check
    (PHASE_CHECK) follows the snare; the network was trained on beats.
    Then several lines at one tempo collapse into one where the beats agree
    (COLLAPSE_LINES_BEAT_THIS)."""
    points = list(getattr(analysis, "points", None) or [])
    if not (params.PHASE_BEAT_THIS or params.COLLAPSE_LINES_BEAT_THIS) or not points \
            or any(getattr(p, "manual", False) for p in points):
        return analysis
    found = heard()
    if found is None or len(found) < params.PHASE_BEAT_THIS_MIN_BEATS:
        return analysis
    beats = np.sort(np.asarray(found, dtype=float)) * 1000.0
    end_ms = float(analysis.duration) * 1000.0
    lines = [(float(p.offset_ms), float(p.bpm)) for p in points]
    sources = list(range(len(points)))
    moved = False
    if params.PHASE_BEAT_THIS:
        lines, moved = _half_beat_by_beat_this(lines, beats, end_ms, [p.meter for p in points])
    if params.COLLAPSE_LINES_BEAT_THIS and len(lines) > 1:
        collapsed = _collapse_by_beat_this(lines, beats, end_ms, [p.meter for p in points])
        if collapsed is not None:
            lines, sources = [(o, b) for o, b, _k in collapsed], [k for _o, _b, k in collapsed]
            moved = True
    if not moved:
        return analysis
    out = phrases._with_lines(analysis, lines,
                              beats=red_lines._beats_of(lines, float(analysis.duration)))
    # each line keeps its own meter and kiai, even snapped a hair before it
    return replace(out, points=[replace(points[k], offset_ms=o, bpm=b) for (o, b), k in zip(lines, sources)])


def _foreign_lines_by_beat_this(analysis: Analysis, heard) -> Analysis:
    """A line at another tempo whose Beat This! beats sit on the song's main
    grid, and far off its own, goes back onto the main grid
    (FOREIGN_LINES_BEAT_THIS). The main line governs the longest span. A
    section read at 2/3 or 4/3 of the song's beat (Mili's Paper Bouquet: 212
    then 130.73 from 16 s) is still the song's beat to the network: its beats
    there sit within FOREIGN_LINES_ON of a main beat and within
    FOREIGN_LINES_RATIO of their distance to the line's own grid."""
    points = list(getattr(analysis, "points", None) or [])
    if not params.FOREIGN_LINES_BEAT_THIS or len(points) < 2 \
            or any(getattr(p, "manual", False) for p in points):
        return analysis
    found = heard()
    if found is None or len(found) < params.PHASE_BEAT_THIS_MIN_BEATS:
        return analysis
    beats = np.sort(np.asarray(found, dtype=float)) * 1000.0
    end_ms = float(analysis.duration) * 1000.0
    spans = [(0.0 if n == 0 else p.offset_ms, points[n + 1].offset_ms if n + 1 < len(points) else end_ms)
             for n, p in enumerate(points)]
    main = max(range(len(points)), key=lambda n: spans[n][1] - spans[n][0])
    m_offset, m_bpm = float(points[main].offset_ms), float(points[main].bpm)
    m_step = 60000.0 / m_bpm

    def off_grid(x, offset, bpm):
        step = 60000.0 / bpm
        k = (x - offset) / step
        return np.abs(k - np.round(k)) * step

    lines, moved = [], False
    for n, p in enumerate(points):
        lo, hi = spans[n]
        inside = beats[(beats >= lo) & (beats < hi)]
        if n != main and abs(p.bpm / m_bpm - 1.0) * 100.0 > params.FOREIGN_LINES_PCT \
                and inside.size >= params.FOREIGN_LINES_MIN_BEATS:
            own = float(np.median(off_grid(inside, p.offset_ms, p.bpm)))
            on = float(np.median(off_grid(inside, m_offset, m_bpm)))
            if on <= params.FOREIGN_LINES_ON * m_step and on <= params.FOREIGN_LINES_RATIO * own:
                k = np.round((p.offset_ms - m_offset) / m_step)
                lines.append((m_offset + k * m_step, m_bpm))
                moved = True
                continue
        lines.append((float(p.offset_ms), float(p.bpm)))
    if not moved:
        return analysis
    lines = red_lines._merge_lines(lines)
    return phrases._with_lines(analysis, lines, beats=red_lines._beats_of(lines, float(analysis.duration)))


def _off_grid(x: np.ndarray, offset: float, bpm: float) -> np.ndarray:
    """Distance (ms) from each time to the nearest beat of a grid."""
    step = 60000.0 / bpm
    k = (x - offset) / step
    return np.abs(k - np.round(k)) * step


def _same_tempo(a: float, b: float, pct: float) -> bool:
    """Whether two tempi are one, octaves apart included."""
    return any(abs(a / (b * f) - 1.0) * 100.0 <= pct for f in (1.0, 2.0, 0.5, 4.0, 0.25))


def _tempo_runs(sections: list) -> list[dict]:
    """Consecutive sections at one tempo (within TEMPO_RUNS_TOL) as runs of
    TEMPO_RUNS_MIN_S or more, each on its longest section's grid: start and
    end (s), bpm and that section's index. Shorter runs are left out, and the
    runs either side of one join when they are at one tempo."""
    runs: list[dict] = []
    for n, section in enumerate(sections):
        bpm, length = float(section.bpm), float(section.end_s - section.start_s)
        if runs and abs(bpm / runs[-1]["bpm"] - 1.0) <= params.TEMPO_RUNS_TOL:
            run = runs[-1]
            run["end"] = float(section.end_s)
            if length > run["length"]:
                run.update(length=length, bpm=bpm, best=n)
        else:
            runs.append({"start": float(section.start_s), "end": float(section.end_s), "bpm": bpm,
                         "length": length, "best": n})
    out: list[dict] = []
    for run in runs:
        if run["end"] - run["start"] < params.TEMPO_RUNS_MIN_S:
            continue
        if out and abs(run["bpm"] / out[-1]["bpm"] - 1.0) <= params.TEMPO_RUNS_TOL:
            last = out[-1]
            last["end"] = run["end"]
            if run["length"] > last["length"]:
                last.update(length=run["length"], bpm=run["bpm"], best=run["best"])
        else:
            out.append(dict(run))
    return out


def _cut_between_grids(times: np.ndarray, weights: np.ndarray, beats: np.ndarray, near: float,
                       left: tuple[float, float], right: tuple[float, float],
                       lo_limit: float, hi_limit: float) -> float:
    """The beat (ms) of the ``right`` grid, within TEMPO_RUNS_REACH_S of
    ``near``, where leaving the ``left`` grid for it explains the attacks and
    Beat This!'s beats best; grids are (offset ms, bpm).

    An attack counts by its distance to its grid's quarter beats, times its
    weight: a run of sixteenths says as much about a grid as its beats do,
    and counted against the beats alone it fits every grid equally badly. A
    Beat This! beat counts by its distance to the beats and half beats,
    TEMPO_RUNS_BEAT_WEIGHT median attacks' worth: the network keeps counting
    the old tempo through a pause. After a pause (TEMPO_RUNS_PAUSE_BEATS
    beats with no strong attack) the right grid starts on the first strong
    attack's beat, as a mapper puts it, when that beat explains the rest
    within TEMPO_RUNS_PAUSE_SLACK beats' worth of the best one."""
    r_offset, r_bpm = right
    step, l_step = 60000.0 / r_bpm, 60000.0 / left[1]
    reach = params.TEMPO_RUNS_REACH_S * 1000.0
    lo, hi = max(near - reach, lo_limit), min(near + reach, hi_limit)
    k0, k1 = int(np.ceil((lo - r_offset) / step)), int(np.floor((hi - r_offset) / step))
    inside = (times >= lo - step) & (times < hi + step)
    t, w = times[inside], weights[inside]
    if k1 <= k0 or t.size < 8:
        return float(r_offset + np.ceil((near - 0.25 * step - r_offset) / step - 1e-9) * step)
    b = beats[(beats >= lo - step) & (beats < hi + step)]
    unit = params.TEMPO_RUNS_BEAT_WEIGHT * float(np.median(w))
    x = np.concatenate([t, b])
    on_left = np.concatenate([
        np.minimum(_off_grid(t, left[0], 4.0 * left[1]) / (l_step / 8.0), 1.0) * w,
        np.minimum(_off_grid(b, left[0], 2.0 * left[1]) / (l_step / 4.0), 1.0) * unit])
    on_right = np.concatenate([
        np.minimum(_off_grid(t, r_offset, 4.0 * r_bpm) / (step / 8.0), 1.0) * w,
        np.minimum(_off_grid(b, r_offset, 2.0 * r_bpm) / (step / 4.0), 1.0) * unit])
    order = np.argsort(x, kind="stable")
    x, on_left, on_right = x[order], on_left[order], on_right[order]
    before = np.concatenate([[0.0], np.cumsum(on_left)])
    after = np.concatenate([[0.0], np.cumsum(on_right)])
    cuts = r_offset + np.arange(k0, k1 + 1) * step
    # an attack a quarter beat before the cut is already the right grid's
    costs = np.array([before[i] + (after[-1] - after[i])
                      for i in np.searchsorted(x, cuts - 0.25 * step)])
    best = int(np.argmin(costs))
    strong = t[w >= 1.5 * float(np.median(w))]
    if strong.size >= 2:
        ends = strong[1:][np.diff(strong) >= params.TEMPO_RUNS_PAUSE_BEATS * min(step, l_step)]
        picks = [int(np.argmin(np.abs(cuts - end))) for end in ends]
        picks = [i for i, end in zip(picks, ends) if abs(cuts[i] - end) <= 0.25 * step
                 and costs[i] <= costs[best] + params.TEMPO_RUNS_PAUSE_SLACK * unit]
        if picks:
            best = min(picks, key=lambda i: costs[i])
    return float(cuts[best])


def _tempo_runs_by_beat_this(analysis: Analysis, heard) -> Analysis:
    """A stretch at another tempo that the confidence floor dropped, back as
    a red line when Beat This! follows it (TEMPO_RUNS_BEAT_THIS).

    The precision engine keeps a line per section that reaches the minimum
    confidence, or the likeliest section's alone. A live band's sections
    never reach it, so a song at 140 for 38 s and 200 after came out at 140
    throughout, with both sections in the core's report. A run of sections at
    one tempo (_tempo_runs), at another tempo than the line over it, becomes a
    line when Beat This!'s beats inside it sit on the run's grid and not on
    that line's; the beats or the half beats, since a fast band's sections sit
    on the kick or the snare, and the run's line takes the half the network
    is on. Each change lands on the new grid's beat where the attacks and the
    beats leave the old one (_cut_between_grids), and the song's tempo
    resumes after the run on its own grid where the attacks still sit on it.
    The precision engine's lines only; the analysis as it was otherwise."""
    points = list(getattr(analysis, "points", None) or [])
    sections = list(getattr(analysis, "sections", None) or [])
    if not params.TEMPO_RUNS_BEAT_THIS or getattr(analysis, "engine", "") != "precision" \
            or len(sections) < 2 or not points or any(getattr(p, "manual", False) for p in points):
        return analysis
    # The sections keep the core's pulse and the lines have been read at the
    # map's octave since: tempi are compared with octaves apart as the same.
    runs = _tempo_runs(sections)
    if len(runs) < 2:
        return analysis
    found = heard()
    if found is None or len(found) < params.TEMPO_RUNS_MIN_BEATS:
        return analysis
    beats = np.sort(np.asarray(found, dtype=float)) * 1000.0
    times = np.asarray(analysis.attack_times, dtype=float) * 1000.0
    weights = np.asarray(analysis.attack_weights, dtype=float)
    if times.size != weights.size:
        return analysis
    lines = [(float(p.offset_ms), float(p.bpm)) for p in points]

    def governing(moment: float) -> tuple[float, float]:
        line = lines[0]
        for candidate in lines:
            if candidate[0] <= moment:
                line = candidate
        return line

    for run in runs:
        lo, hi = run["start"] * 1000.0, run["end"] * 1000.0
        offset, bpm = run["line"] = governing(0.5 * (lo + hi))
        run["same"] = _same_tempo(run["bpm"], bpm, params.TEMPO_RUNS_PCT)
        run["heard"] = beats[(beats >= lo) & (beats < hi)]
        run["taken"] = False
        if run["same"] or run["heard"].size < params.TEMPO_RUNS_MIN_BEATS:
            continue
        own_offset = sections[run["best"]].phase * 1000.0
        own = float(np.mean(_off_grid(run["heard"], own_offset, 2.0 * run["bpm"])
                            <= params.TEMPO_RUNS_ON * 60000.0 / run["bpm"]))
        other = float(np.mean(_off_grid(run["heard"], offset, 2.0 * bpm)
                              <= params.TEMPO_RUNS_ON * 60000.0 / bpm))
        run["taken"] = own >= params.TEMPO_RUNS_SHARE and other <= own - params.TEMPO_RUNS_GAP
    if not any(run["taken"] for run in runs):
        return analysis

    def grid(run: dict, bpm: float, first: float | None = None) -> float:
        """The run's grid at ``bpm`` (its offset, ms) on the phase its attacks
        keep: counted from ``first``, the song's own grid, where they sit near
        it, else from the run's longest section; on the half beat where Beat
        This!'s beats are (PHASE_BEAT_THIS's test)."""
        window = (times >= run["start"] * 1000.0) & (times <= run["end"] * 1000.0)
        t, w = times[window], weights[window]
        step = 60000.0 / bpm

        def settled(start: float):
            miss = t - (start + np.round((t - start) / step) * step)
            near = np.abs(miss) <= params.ONE_LINE_PHASE_TOL * step
            if int(near.sum()) < params.ONE_LINE_PHASE_MIN or float(w[near].sum()) <= 0:
                return None
            share = float(w[near].sum()) / max(float(w[np.abs(miss) <= 0.25 * step].sum()), 1e-12)
            return start + float(np.average(miss[near], weights=w[near])), share

        offset = sections[run["best"]].phase * 1000.0
        tried = settled(first) if first is not None else None
        if tried is not None and tried[1] >= 0.35:
            offset = tried[0]
        if run["heard"].size >= params.PHASE_BEAT_THIS_MIN_BEATS:
            on = np.mod((run["heard"] - offset) / step, 1.0)
            on = np.minimum(on, 1.0 - on)
            if float(np.median(0.5 - on)) < params.PHASE_BEAT_THIS_GAIN * float(np.median(on)):
                offset += 0.5 * step
        tried = settled(offset)
        return tried[0] if tried is not None else offset

    def octave(run: dict) -> float:
        """The run's tempo at the octave Beat This! counts it, under the cap."""
        tempo = 60000.0 / float(np.median(np.diff(run["heard"])))
        bpm = run["bpm"] * min((1.0, 0.5, 2.0), key=lambda f: abs(np.log2(tempo / (run["bpm"] * f))))
        while params.OCTAVE_CAP and bpm > params.OCTAVE_CAP_BPM:
            bpm /= 2.0
        return bpm

    out = list(lines)
    previous = None                       # the last run's line, when it took one of its own
    for n, run in enumerate(runs):
        lo, hi = run["start"] * 1000.0, run["end"] * 1000.0
        if run["taken"]:
            bpm = octave(run)
            offset, step = grid(run, bpm), 60000.0 / bpm
            if n == 0:
                start = offset + np.ceil((lo - 0.25 * step - offset) / step - 1e-9) * step
            else:
                left = previous if previous is not None else run["line"]
                start = _cut_between_grids(times, weights, beats, lo, left, (offset, bpm),
                                           runs[n - 1]["start"] * 1000.0 + 4.0 * 60000.0 / left[1], hi)
            out = [line for line in out if not min(start, lo) - 0.5 * step <= line[0] < hi]
            out.append((float(start), float(bpm)))
            previous = out[-1]
        elif previous is not None and run["same"]:
            # Back at the song's tempo after a run at another.
            main_offset, main_bpm = run["line"]
            keeps = any(abs(run["bpm"] * f / main_bpm - 1.0) <= params.TEMPO_RUNS_SAME
                        for f in (1.0, 2.0, 0.5))
            bpm = main_bpm if keeps else octave(run)
            offset, step = grid(run, bpm, main_offset if keeps else None), 60000.0 / bpm
            start = _cut_between_grids(times, weights, beats, lo, previous, (offset, bpm),
                                       previous[0] + 4.0 * 60000.0 / previous[1], hi)
            out = [line for line in out if not previous[0] < line[0] < start + 4.0 * step]
            out.append((float(start), float(bpm)))
            previous = None
        elif run["same"]:
            previous = None
    out.sort()
    # osu! runs the first line back to 0 s: a run's line ahead of the song's
    # own would take the start with it, so the song's line moves back first.
    first = min(lines)
    if out[0] != first and not runs[0]["taken"]:
        step = 60000.0 / first[1]
        back = first[0] - np.floor((first[0] - max(0.0, runs[0]["start"] * 1000.0)) / step) * step
        if back < out[0][0] - 4.0 * step:
            out.insert(0, (float(back), first[1]))
    out = red_lines._snap_whole_lines(out, times / 1000.0, weights, float(analysis.duration))
    return phrases._with_lines(analysis, out, beats=red_lines._beats_of(out, float(analysis.duration)))


def _load_beat_this():
    """A new ONNX Runtime session on the model; None without one or the other."""
    path = params.ROOT / "assets" / "models" / params.BEAT_THIS_MODEL
    try:
        import onnxruntime
        if not path.is_file():
            return None
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        # The memory pattern plans one buffer layout per input shape, and every
        # song's last chunk is a new shape. Off, the peak working set over three
        # songs (135, 257 and 477 s) fell 1904 -> 1278 MB with every beat
        # within 1 ms and no more CPU (2026-10-27, docs/17-stack-alternatives.md).
        options.enable_mem_pattern = False
        return onnxruntime.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
    except Exception:  # noqa: BLE001 -- no runtime or no model: nothing to ask
        return None


def _beat_this_session():
    """The vote's ONNX Runtime session, loaded once; None without the model
    or the runtime, or where the vote is turned off."""
    if not params._BEAT_THIS_SESSION:
        params._BEAT_THIS_SESSION.append(_load_beat_this())
    return params._BEAT_THIS_SESSION[0]


def _beat_this_mel_filters() -> np.ndarray:
    """Slaney mel filters without area normalisation, as torchaudio builds them."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0

    def to_mel(f):
        f = np.asarray(f, dtype=float)
        return np.where(f >= min_log_hz,
                        min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, f / f_sp)

    def to_hz(m):
        m = np.asarray(m, dtype=float)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    freqs = np.linspace(0, params.BEAT_THIS_SR // 2, params.BEAT_THIS_FFT // 2 + 1)
    lo, hi = params.BEAT_THIS_BAND_HZ
    edges = to_hz(np.linspace(to_mel(lo), to_mel(hi), params.BEAT_THIS_MELS + 2))
    gaps = np.diff(edges)
    slopes = edges[None, :] - freqs[:, None]
    return np.maximum(0.0, np.minimum(-slopes[:, :-2] / gaps[:-1], slopes[:, 2:] / gaps[1:]))


def _beat_this_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Beat This!'s input: log(1 + 1000 * mel magnitude), (frames, 128)."""
    pad = params.BEAT_THIS_FFT // 2
    x = np.pad(np.asarray(signal, dtype=np.float32), pad, mode="reflect")
    window = np.hanning(params.BEAT_THIS_FFT + 1)[:-1].astype(np.float32)     # periodic Hann
    count = 1 + (len(x) - params.BEAT_THIS_FFT) // params.BEAT_THIS_HOP
    if count < 1:
        return np.zeros((0, params.BEAT_THIS_MELS), dtype=np.float32)
    mel = _beat_this_mel_filters().astype(np.float32)
    out = np.empty((count, params.BEAT_THIS_MELS), dtype=np.float32)
    for lo in range(0, count, 4096):            # a few thousand frames at a time
        hi = min(count, lo + 4096)
        idx = np.arange(params.BEAT_THIS_FFT)[None, :] + params.BEAT_THIS_HOP * np.arange(lo, hi)[:, None]
        # torchaudio's normalized="frame_length" divides by sqrt(n_fft)
        magnitude = np.abs(np.fft.rfft(x[idx] * window, axis=1)) / np.sqrt(params.BEAT_THIS_FFT)
        out[lo:hi] = np.log1p(1000.0 * (magnitude @ mel))
    return out


def _beat_this_logits(session, spect: np.ndarray) -> np.ndarray:
    """Frame-wise beat logits over the whole song, chunk by chunk."""
    frames = len(spect)
    step = params.BEAT_THIS_CHUNK - 2 * params.BEAT_THIS_BORDER
    starts = np.arange(-params.BEAT_THIS_BORDER, frames - params.BEAT_THIS_BORDER, step)
    if frames > step:
        starts[-1] = frames - (params.BEAT_THIS_CHUNK - params.BEAT_THIS_BORDER)
    logits = np.full(frames, -1000.0, dtype=np.float32)
    for start in starts[::-1]:                  # earlier chunks win their overlap
        a, b = max(int(start), 0), min(int(start) + params.BEAT_THIS_CHUNK, frames)
        chunk = np.pad(spect[a:b], ((max(0, -int(start)),
                                     max(0, min(params.BEAT_THIS_BORDER, int(start) + params.BEAT_THIS_CHUNK - frames))),
                                    (0, 0)))
        data_model.checkpoint()
        out = session.run(None, {"spect": chunk[None]})[0][0][params.BEAT_THIS_BORDER:-params.BEAT_THIS_BORDER]
        lo = int(start) + params.BEAT_THIS_BORDER
        keep = max(0, min(len(out), frames - lo))
        logits[lo:lo + keep] = out[:keep]
    return logits


def _beat_this_beats(y: np.ndarray, sr: int, session=None) -> np.ndarray | None:
    """Beat times (s) by Beat This!, or None when the network is unavailable;
    ``session`` instead of the vote's."""
    session = session if session is not None else _beat_this_session()
    if session is None or y.size == 0:
        return None
    import soxr
    signal = soxr.resample(np.asarray(y, dtype=np.float64), sr, params.BEAT_THIS_SR) \
        if sr != params.BEAT_THIS_SR else np.asarray(y, dtype=np.float64)
    logits = _beat_this_logits(session, _beat_this_spectrogram(signal))
    padded = np.pad(logits, 3, constant_values=-1000.0)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 7).max(axis=1)
    frames = np.flatnonzero((logits == local_max) & (logits > 0))
    beats: list[float] = []
    mean, count = None, 0
    for f in frames:                            # adjacent peaks become their mean
        if mean is not None and f - mean <= 1:
            count += 1
            mean += (f - mean) / count
        else:
            if mean is not None:
                beats.append(mean)
            mean, count = float(f), 1
    if mean is not None:
        beats.append(mean)
    return np.asarray(beats, dtype=float) * params.BEAT_THIS_HOP / params.BEAT_THIS_SR


def _grid_fit_score(beats: np.ndarray, times: np.ndarray, weights: np.ndarray,
                    tol_ms: float) -> float:
    """How well a beat grid explains the attacks: F-measure of attack weight
    within ``tol_ms`` of a beat and beats within ``tol_ms`` of an attack."""
    if beats.size < 4 or times.size < 2:
        return 0.0
    i = np.clip(np.searchsorted(beats, times), 1, beats.size - 1)
    d = np.minimum(np.abs(beats[i - 1] - times), np.abs(beats[i] - times)) * 1000.0
    recall = float(np.sum(weights[d <= tol_ms]) / max(float(np.sum(weights)), 1e-12))
    j = np.clip(np.searchsorted(times, beats), 1, times.size - 1)
    e = np.minimum(np.abs(times[j - 1] - beats), np.abs(times[j] - beats)) * 1000.0
    precision = float(np.mean(e <= tol_ms))
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def _beat_this_vote(analysis: Analysis, y: np.ndarray, sr: int) -> Analysis:
    """The precision analysis, or Beat This!'s lines where they fit the attacks
    clearly better (BEAT_THIS_MARGIN). Never runs on a fallback analysis."""
    if analysis.engine != "precision" or not analysis.points:
        return analysis
    times = np.asarray(analysis.attack_times, dtype=float)
    weights = np.asarray(analysis.attack_weights, dtype=float)
    if times.size < 8:
        return analysis
    beats = _beat_this_beats(y, sr)
    if beats is not None:
        # the judge loads the same model: its later reads of this decode
        # (PHASE_BEAT_THIS, the octave checks) take these beats, not a second run
        params._BEAT_THIS_HEARD[:] = [((id(y), int(y.size), int(sr)), y, beats)]
    if beats is None or beats.size < 8:
        return analysis
    duration = float(analysis.duration)
    lines = red_lines._beats_to_lines(beats, params.LOCAL_LINE_TOL_MS)
    if not lines:
        return analysis
    ours = _grid_fit_score(red_lines._beats_of([(p.offset_ms, p.bpm) for p in analysis.points], duration),
                           times, weights, params.BEAT_THIS_TOL_MS)
    theirs = _grid_fit_score(red_lines._beats_of(lines, duration), times, weights, params.BEAT_THIS_TOL_MS)
    if theirs <= ours + params.BEAT_THIS_MARGIN:
        return analysis
    points = []
    for offset, bpm in lines:
        governing = analysis.points[0]
        for point in analysis.points:
            if point.offset_ms <= offset:
                governing = point
        points.append(replace(governing, offset_ms=float(offset), bpm=float(bpm)))
    bpms = np.array([p.bpm for p in points])
    spans = np.diff([p.offset_ms for p in points] + [duration * 1000.0])
    return replace(analysis, points=points, beats=beats, local_tempo=True,
                   global_bpm=pipeline._weighted_median(bpms, np.maximum(spans, 1e-6)))
