"""Prototype: red lines fitted to Beat This!'s beats, scored as ranked maps.

The hand-written detector misses short sections and refuses songs with no
single pulse. Beat This! hears beats through tempo changes; this fits the
fewest constant-BPM lines that keep every beat within ``--tol`` ms and scores
them with bench/rankable.py against the same ranked maps a stage run scored.

    python proto/beat_lines.py beats <report.json>... --out beats.json.gz   # slow: the model
    python proto/beat_lines.py score <report.json>... --beats beats.json.gz # fast: fit + score

A measurement, not part of the engine.
"""
from __future__ import annotations

import argparse
import gzip
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench"), str(ROOT / "proto")]


SUBFRAME = False


def _beats_of(path: str):
    import overtone as ta
    try:
        y, sr = ta._load_audio(path, lambda _m: None)
        if SUBFRAME or os.environ.get("BEAT_LINES_SUBFRAME"):
            beats = _subframe_beats(y, sr)
        else:
            beats = ta._beat_this_judge_beats(y, sr)
    except Exception as exc:  # noqa: BLE001 - a measurement: name it and go on
        return path, None, f"{type(exc).__name__}: {exc}"
    return path, None if beats is None else [round(float(b), 5) for b in beats], None


def _subframe_beats(y, sr):
    """Beat This!'s beats with each peak placed between frames (proto/beat_subframe.py)."""
    import soxr
    from beat_subframe import peaks
    from overtone import beat_this, params
    session = beat_this._load_beat_this()
    if session is None:
        return None
    sig = soxr.resample(np.asarray(y, dtype=np.float64), sr, params.BEAT_THIS_SR)
    logits = beat_this._beat_this_logits(session, beat_this._beat_this_spectrogram(sig)).astype(float)
    return peaks(logits, True) * params.BEAT_THIS_HOP / params.BEAT_THIS_SR


def _audios(reports: list[dict]) -> dict[str, str]:
    out = {}
    for r in reports:
        for m in r["maps"]:
            out.setdefault(m["audio_sha1"], str(Path(m["folder_path"]) / m["audio_filename"]))
    return out


def fit_lines(beats_s, tol_ms: float, min_beats: int, merge_pct: float, spans: bool = False):
    """The fewest lines (offset ms, BPM) keeping every beat within tol_ms.

    Beats are indexed along the grid (a missed beat leaves a gap, not a
    shift); a segment grows while its least-squares grid holds every beat.
    Validity is prefix-closed, so the farthest reach from each start gives
    the fewest segments."""
    t = np.sort(np.asarray(beats_s, dtype=float)) * 1000.0
    if t.size < 8:
        return []
    lines: list[list[float]] = []
    i = 0
    while i < t.size - 1:
        k = [0]
        best = None
        for j in range(i + 1, t.size):
            if len(k) > 1:
                period = (t[j - 1] - t[i]) / k[-1]
            else:
                period = t[j] - t[i]
            step = max(1, int(round((t[j] - t[j - 1]) / period))) if len(k) > 1 else 1
            k.append(k[-1] + step)
            kk = np.asarray(k, dtype=float)
            seg = t[i:j + 1]
            a, b = np.polyfit(kk, seg, 1)
            if np.max(np.abs(seg - (a * kk + b))) > tol_ms:
                k.pop()
                break
            best = (j, a, b)
        if best is None:
            i += 1
            continue
        j, a, b = best
        if j - i + 1 >= min_beats:
            lines.append([b, 60000.0 / a, t[i], t[j]])
        i = j
    merged: list[list[float]] = []
    for line in lines:
        if merged:
            prev = merged[-1]
            if abs(line[1] / prev[1] - 1.0) * 100.0 <= merge_pct:
                period = 60000.0 / prev[1]
                phase = (line[0] - prev[0]) / period
                if abs(phase - round(phase)) * period <= tol_ms:
                    prev[3] = line[3]
                    continue
        merged.append(line)
    if spans:
        return [(float(o), float(bpm), float(a), float(e)) for o, bpm, a, e in merged]
    return [(float(o), float(bpm)) for o, bpm, _s, _e in merged]


def _fold_ratio(a: float, b: float) -> float:
    """a/b folded by octaves into [1/sqrt2, sqrt2)."""
    r = a / b
    while r >= 2 ** 0.5:
        r /= 2.0
    while r < 2 ** -0.5:
        r *= 2.0
    return r


def insert_sections(lines, segs, duration_s: float, min_s: float, pct: float, octave_too: bool):
    """The engine's lines with Beat This!'s long sections of another tempo
    spliced in; after each, the engine's grid resumes on its nearest beat."""
    end_ms = duration_s * 1000.0
    lines = sorted((float(o), float(b)) for o, b in lines)

    def engine_at(t):
        cur = lines[0]
        for o, b in lines:
            if o <= t:
                cur = (o, b)
        return cur

    picked = []
    for o, bpm, a, e in segs:
        if (e - a) / 1000.0 < min_s:
            continue
        eo, eb = engine_at(0.5 * (a + e))
        r = bpm / eb
        folded = _fold_ratio(bpm, eb)
        if abs(r - 1.0) * 100.0 < pct:
            continue
        if not octave_too and abs(folded - 1.0) * 100.0 < pct:
            continue
        picked.append((o, bpm, a, e))
    if not picked:
        return None
    events = []  # (start ms, offset ms, bpm)
    def snap(o, b, t):
        step = 60000.0 / b
        return o + np.ceil((t - o) / step - 0.25) * step
    cursor = 0.0
    for o, bpm, a, e in picked:
        # engine pieces from cursor to a
        for n, (lo, lb) in enumerate(lines):
            nxt = lines[n + 1][0] if n + 1 < len(lines) else end_ms
            s0, s1 = max(cursor, lo if n else -1e12), min(a, nxt)
            if s1 > s0 + 1.0:
                events.append((snap(lo, lb, max(s0, 0.0)), lb))
        events.append((snap(o, bpm, a), bpm))
        cursor = e
    for n, (lo, lb) in enumerate(lines):
        nxt = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        s0 = max(cursor, lo if n else -1e12)
        if nxt > s0 + 1.0:
            events.append((snap(lo, lb, max(s0, 0.0)), lb))
    events.sort()
    out = []
    for o, b in events:
        if out and abs(o - out[-1][0]) < 1.0:
            out[-1] = (o, b)
        else:
            out.append((o, b))
    return out


def phase_by_beats(lines, beats_s, duration_s: float, shifts=(0.5,)):
    """Each engine line moved by the fraction of a beat (or kept) whose grid
    lies nearest Beat This!'s beats inside it: the engine's BPM, the
    network's idea of where the beat is."""
    t = np.sort(np.asarray(beats_s, dtype=float)) * 1000.0
    out = []
    for n, (o, bpm) in enumerate(lines):
        stop = lines[n + 1][0] if n + 1 < len(lines) else duration_s * 1000.0
        inside = t[(t >= o) & (t < stop)]
        step = 60000.0 / bpm
        if inside.size < 8:
            out.append((o, bpm))
            continue

        def cost(shift):
            ph = np.mod((inside - (o + shift * step)) / step, 1.0)
            return float(np.median(np.minimum(ph, 1.0 - ph)))

        best = min((0.0,) + tuple(shifts), key=cost)
        if best and cost(best) < 0.5 * cost(0.0):
            o = o + best * step
        out.append((o, bpm))
    return out


CLIP_MS = 25.0


def _centre(x):
    """A trimmed mean: Beat This!'s beats sit on a 20 ms grid, a median of them
    would too."""
    x = np.sort(np.asarray(x, dtype=float))
    med = np.median(x)
    keep = x[np.abs(x - med) <= CLIP_MS]
    return float(np.mean(keep)) if keep.size else float(med)


def _cost(x):
    return float(np.sum(np.minimum(np.abs(x - np.median(x)), CLIP_MS)))


def _split_points(r, min_beats: int, min_ms: float, out: list, lo: int, hi: int, depth=0):
    """Binary segmentation of residuals r[lo:hi] into stretches of constant
    centre, each deviation clipped at CLIP_MS (half-beat blips cost no more
    than a miss)."""
    n = hi - lo
    if n < 2 * min_beats or depth > 6:
        return
    seg = r[lo:hi]
    base = _cost(seg)
    best = None
    for k in range(min_beats, n - min_beats + 1, 2):
        a, b = seg[:k], seg[k:]
        if abs(np.median(a) - np.median(b)) < min_ms:
            continue
        cost = _cost(a) + _cost(b)
        if best is None or cost < best[0]:
            best = (cost, k)
    if best is None or base - best[0] < SPLIT_GAIN * base:
        return
    k = lo + best[1]
    out.append(k)
    _split_points(r, min_beats, min_ms, out, lo, k, depth + 1)
    _split_points(r, min_beats, min_ms, out, k, hi, depth + 1)


SPLIT_GAIN = 0.15


def resets_by_beats(lines, beats_s, duration_s: float, min_beats: int = 16, min_ms: float = 12.0):
    """Each engine line split where Beat This!'s beats move off its grid for a
    stretch and stay there (a metronome reset at the same BPM): the stretch
    gets a line at the engine's BPM, its offset moved by the stretch's median
    distance less that of the longest stretch (Beat This!'s own lag). The
    first line also covers the song before its offset."""
    t = np.sort(np.asarray(beats_s, dtype=float)) * 1000.0
    pieces = []
    for n, (o, bpm) in enumerate(lines):
        start = 0.0 if n == 0 else o
        stop = lines[n + 1][0] if n + 1 < len(lines) else duration_s * 1000.0
        inside = t[(t >= start) & (t < stop)]
        step = 60000.0 / bpm
        r = inside - o - np.round((inside - o) / step) * step
        cuts = []
        if inside.size >= 2 * min_beats:
            _split_points(r, min_beats, min_ms, cuts, 0, r.size)
        bounds = [0] + sorted(cuts) + [r.size]
        segs = [(a, b, _centre(r[a:b]) if b > a else 0.0) for a, b in zip(bounds, bounds[1:])]
        pieces.append((o, bpm, inside, segs))
    longest = max((b - a, med) for _o, _b, _i, segs in pieces for a, b, med in segs)
    if longest[0] < min_beats:
        return list(lines)
    bias = longest[1]
    out = []
    for n, (o, bpm, inside, segs) in enumerate(pieces):
        step = 60000.0 / bpm
        new = []
        for a, b, med in segs:
            shift = med - bias
            if abs(shift) < min_ms:
                shift = 0.0
            if a == 0 and (n > 0 or b == inside.size or inside.size == 0):
                new.append((o + shift, bpm))
                continue
            k = round((inside[a] - o) / step) if inside.size else 0
            new.append((o + k * step + shift, bpm))
        merged = [new[0]]
        for line in new[1:]:
            prev = merged[-1]
            ph = (line[0] - prev[0]) / step
            if abs(ph - round(ph)) * step < 3.0:
                continue
            merged.append(line)
        out.extend(merged)
    return out


def one_line(beats_s, hint_bpm: float, tol_ms: float = 25.0):
    """One line (offset ms, BPM) through Beat This!'s beats over the whole
    song, at the octave of hint_bpm: beats indexed on the grid, fitted by
    least squares, the ones farther than tol_ms dropped and the fit redone."""
    t = np.sort(np.asarray(beats_s, dtype=float)) * 1000.0
    if t.size < 16:
        return None
    period = float(np.median(np.diff(t)))
    want = 60000.0 / hint_bpm
    while period > want * 1.42:
        period /= 2.0
    while period < want / 1.42:
        period *= 2.0
    keep = np.ones(t.size, dtype=bool)
    t0 = float(np.median(t - np.round((t - t[0]) / period) * period))
    for _ in range(4):
        k = np.round((t - t0) / period)
        r = t - (t0 + k * period)
        keep = np.abs(r) <= max(tol_ms, 0.15 * period) if _ < 2 else np.abs(r) <= tol_ms
        if keep.sum() < 16:
            return None
        period, t0 = np.polyfit(k[keep], t[keep], 1)
    return float(t0 + np.ceil((t[0] - t0) / period) * period), 60000.0 / float(period)


def _fragmented(m: dict, at_least: int, short_s: float) -> bool:
    lines = [(float(o), float(b)) for o, b, *_ in m.get("lines") or []]
    if len(lines) < at_least:
        return False
    end = (m.get("duration_s") or 0.0) * 1000.0
    spans = [(lines[n + 1][0] if n + 1 < len(lines) else end) - o for n, (o, _b) in enumerate(lines)]
    return float(np.median(spans)) / 1000.0 < short_s


def _hint(m: dict) -> float:
    """The engine's BPM weighted by how long each line lasts."""
    lines = [(float(o), float(b)) for o, b, *_ in m.get("lines") or []]
    end = (m.get("duration_s") or 0.0) * 1000.0
    spans = np.asarray([max(1.0, (lines[n + 1][0] if n + 1 < len(lines) else end) - o)
                        for n, (o, _b) in enumerate(lines)])
    bpms = np.asarray([b for _o, b in lines])
    order = np.argsort(bpms)
    c = np.cumsum(spans[order])
    return float(bpms[order][np.searchsorted(c, 0.5 * c[-1])])


def tidy(lines, duration_s: float):
    """Whole or tidy BPMs as the engine writes them (phrases._tidy_bpm)."""
    import overtone as ta
    out = []
    for n, (o, bpm) in enumerate(lines):
        stop = lines[n + 1][0] if n + 1 < len(lines) else duration_s * 1000.0
        nb = max(0.0, stop - o) * bpm / 60000.0
        new = ta._tidy_bpm(bpm, nb)
        if new != bpm:
            was, now = 60000.0 / bpm, 60000.0 / new
            o = o + max(0, int((stop - o) / was) // 2) * (was - now)
        out.append((round(o), new))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["beats", "score"])
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--beats")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tol", type=float, default=25.0)
    ap.add_argument("--min-beats", type=int, default=8)
    ap.add_argument("--merge-pct", type=float, default=0.4)
    ap.add_argument("--no-tidy", action="store_true")
    ap.add_argument("--phase", action="store_true",
                    help="keep the run's lines, moved half a beat where Beat This! says so")
    ap.add_argument("--resets", action="store_true",
                    help="keep the run's lines, split where Beat This!'s beats leave the grid")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--insert", action="store_true",
                    help="score: Beat This!'s long sections of another tempo spliced into the engine's lines")
    ap.add_argument("--insert-s", type=float, default=15.0)
    ap.add_argument("--insert-pct", type=float, default=4.0)
    ap.add_argument("--insert-octave", action="store_true", help="--insert: octave sections count too")
    ap.add_argument("--insert-single", action="store_true", help="--insert: only where the engine wrote one line")
    ap.add_argument("--fragmented", type=int, default=0,
                    help="score: where the engine wrote at least this many short lines, one line through the beats")
    ap.add_argument("--frag-short", type=float, default=10.0, help="median line length (s) counted as short")
    ap.add_argument("--frag-fit", action="store_true", help="--fragmented: the fitted lines instead of one")
    ap.add_argument("--subframe", action="store_true", help="beats mode: peaks placed between frames")
    ap.add_argument("--changes", action="store_true",
                    help="fitted lines only where the run wrote one line and the fit holds distinct tempos")
    ap.add_argument("--change-pct", type=float, default=1.5)
    ap.add_argument("--split-gain", type=float, default=0.15)
    ap.add_argument("--reset-ms", type=float, default=12.0)
    ap.add_argument("--reset-beats", type=int, default=16)
    ap.add_argument("--multi", action="store_true",
                    help="fitted lines only where the run wrote more than one line or none")
    ap.add_argument("--only-refused", action="store_true",
                    help="use the fitted lines only where the run's Rust core refused")
    args = ap.parse_args(argv)
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    audios = _audios(reports)
    global SUBFRAME
    SUBFRAME = args.subframe
    if SUBFRAME:
        os.environ["BEAT_LINES_SUBFRAME"] = "1"
    if args.mode == "beats":
        done = {}
        if args.out and Path(args.out).exists():
            done = json.load(gzip.open(args.out, "rt", encoding="utf-8"))
        todo = [p for a, p in audios.items() if a not in done]
        by_path = {p: a for a, p in audios.items()}
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for n, (path, beats, why) in enumerate(pool.map(_beats_of, todo), 1):
                done[by_path[path]] = beats
                if why:
                    print("failed", path, why, flush=True)
                if n % 25 == 0 or n == len(todo):
                    print(f"{n}/{len(todo)}", flush=True)
                    with gzip.open(args.out, "wt", encoding="utf-8") as h:
                        json.dump(done, h)
        return 0

    global SPLIT_GAIN
    SPLIT_GAIN = args.split_gain
    import rankable
    beats = json.load(gzip.open(args.beats, "rt", encoding="utf-8"))
    maps = [m for r in reports for m in r["maps"]]
    fitted = {}
    segs = {}
    have = {m["audio_sha1"] for m in maps}
    for a, b in beats.items():
        if not b or a not in have:
            continue
        dur = next(m["duration_s"] for m in maps if m["audio_sha1"] == a)
        if not dur:
            continue
        lines = fit_lines(b, args.tol, args.min_beats, args.merge_pct)
        if args.insert:
            segs[a] = fit_lines(b, args.tol, args.min_beats, args.merge_pct, spans=True)
        fitted[a] = lines if args.no_tidy or not lines else tidy(lines, dur)
    dev = [m for m in maps if m.get("split") == "dev"]
    shift = rankable.beat_shift(dev)
    after = []
    for m in maps:
        m["rank"] = rankable.verdict(m, shift)
        use = fitted.get(m["audio_sha1"])
        if args.phase:
            b = beats.get(m["audio_sha1"])
            use = phase_by_beats([(o, bp) for o, bp, *_ in m["lines"]], b, m["duration_s"] or 0.0)                 if b and m.get("lines") else None
        elif args.resets:
            b = beats.get(m["audio_sha1"])
            use = resets_by_beats([(o, bp) for o, bp, *_ in m["lines"]], b, m["duration_s"],
                                  args.reset_beats, args.reset_ms) if b and m.get("lines") else None
        elif args.changes:
            f = fitted.get(m["audio_sha1"]) or []
            bp = [b for _o, b in f]
            distinct = len(f) >= 2 and (max(bp) / min(bp) - 1.0) * 100.0 >= args.change_pct
            use = f if distinct and len(m.get("lines") or []) == 1 else None
        elif args.insert:
            seg = segs.get(m["audio_sha1"])
            eng = [(float(o), float(bp)) for o, bp, *_ in m.get("lines") or []]
            use = None
            if seg and eng and (not args.insert_single or len(eng) == 1):
                use = insert_sections(eng, seg, m["duration_s"] or 0.0, args.insert_s, args.insert_pct,
                                      args.insert_octave)
                if use and not args.no_tidy:
                    use = tidy(use, m["duration_s"])
        elif args.fragmented:
            b = beats.get(m["audio_sha1"])
            use = None
            if b and _fragmented(m, args.fragmented, args.frag_short):
                if args.frag_fit:
                    use = fitted.get(m["audio_sha1"]) or None
                else:
                    one = one_line(b, _hint(m), args.tol)
                    use = tidy([one], m["duration_s"]) if one else None
        elif args.multi and len(m.get("lines") or []) == 1:
            use = None
        if use and (not args.only_refused or m.get("backend") != "rust"):
            n = dict(m, lines=[[o, b] for o, b in use])
        else:
            n = dict(m)
        after.append(n)
    shift_after = rankable.beat_shift([m for m in after if m.get("split") == "dev"])
    for m in after:
        m["rank"] = rankable.verdict(m, shift_after)
    print(f"beats for {len(fitted)} of {len(audios)} audios; lead {shift:+.1f} -> {shift_after:+.1f} ms;"
          f" tol {args.tol} ms, min {args.min_beats} beats, merge {args.merge_pct} %,"
          f" {'refused only' if args.only_refused else 'every song'}")
    for split in ("dev", "holdout"):
        b = [m for m in maps if m.get("split") == split]
        a = [m for m in after if m.get("split") == split]
        if not b:
            continue
        sb, sa = rankable.summarise(b), rankable.summarise(a)
        fixed = sum(x["rank"]["verdict"] == "miss" and y["rank"]["verdict"] != "miss" for x, y in zip(b, a))
        broken = sum(x["rank"]["verdict"] != "miss" and y["rank"]["verdict"] == "miss" for x, y in zip(b, a))
        if args.list:
            for x, y in zip(b, a):
                if (x["rank"]["verdict"] == "miss") != (y["rank"]["verdict"] == "miss"):
                    print("    ", "fixed " if x["rank"]["verdict"] == "miss" else "broken",
                          Path(x["folder_path"]).name[:40], len(x.get("lines") or []), "->", len(y.get("lines") or []),
                          round(x["rank"]["abs_ms"] or -1, 1), "->", round(y["rank"]["abs_ms"] or -1, 1))
        print(f"  {split:<8} songs {rankable.pct(sb['songs_rankable'], sb['songs'])} -> "
              f"{rankable.pct(sa['songs_rankable'], sa['songs'])} %, maps "
              f"{rankable.pct(sb['rankable'], sb['maps'])} -> {rankable.pct(sa['rankable'], sa['maps'])} %"
              f" (fixed {fixed}, broken {broken})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
