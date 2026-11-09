"""Prototype: several red lines at one tempo collapsed into one when Beat
This!'s beats sit on the one line as well as on the several.

The engine sometimes writes a single-tempo song as 2-14 lines at nearly the
same BPM, each a few ms out of phase with the last (bands-variable: every
such song a miss). Where the lines covering most of the song agree on the
BPM, one line through the longest of them is tried; it replaces the several
when the median distance of the network's beats to its grid is at most
--slack times theirs.

    python proto/collapse_lines.py <stage report.json>... --beats <beats.json.gz> [--phase-first] [--list]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench"), str(ROOT / "proto")]


def grid_cost(lines, beats_ms, start_ms, end_ms):
    """Median distance (ms) from each beat in [start, end) to the grid of the line it falls in."""
    b = beats_ms[(beats_ms >= start_ms) & (beats_ms < end_ms)]
    if b.size < 16:
        return None
    starts = np.asarray([o for o, _b in lines])
    idx = np.clip(np.searchsorted(starts, b, side="right") - 1, 0, len(lines) - 1)
    o = starts[idx]
    step = 60000.0 / np.asarray([bp for _o, bp in lines])[idx]
    r = b - o - np.round((b - o) / step) * step
    return float(np.median(np.abs(r)))


LONG_S = 0.0


def per_line_ok(lines, one, beats_ms, end_ms, slack, floor_ms):
    """Every line with 16+ beats: the beats' median distance to the one grid
    within slack times (or floor_ms more than) their distance to its own."""
    for n, line in enumerate(lines):
        stop = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        if (stop - line[0]) / 1000.0 < LONG_S:
            continue
        own = grid_cost([line], beats_ms, line[0], stop)
        new = grid_cost([one], beats_ms, line[0], stop)
        if own is None or new is None:
            continue
        if new > max(slack * own, own + floor_ms):
            return False
    return True


def one_of(lines, end_ms, agree_pct, share):
    """The longest line extended over the song, when the lines covering
    `share` of it agree with its BPM within agree_pct; else None."""
    spans = [((lines[n + 1][0] if n + 1 < len(lines) else end_ms) - o) for n, (o, _b) in enumerate(lines)]
    total = sum(max(0.0, s) for s in spans)
    if total <= 0:
        return None
    k = int(np.argmax(spans))
    o, bpm = lines[k]
    same = sum(max(0.0, s) for s, (_o, b) in zip(spans, lines) if abs(b / bpm - 1.0) * 100.0 <= agree_pct)
    if same / total < share:
        return None
    step = 60000.0 / bpm
    return o - np.floor(o / step) * step, bpm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--beats", required=True)
    ap.add_argument("--agree-pct", type=float, nargs="+", default=[0.5])
    ap.add_argument("--share", type=float, default=0.85)
    ap.add_argument("--slack", type=float, nargs="+", default=[1.0, 1.2])
    ap.add_argument("--phase-first", action="store_true")
    ap.add_argument("--engine", action="store_true",
                    help="score the engine's own _collapse_by_beat_this, its parameters from these flags")
    ap.add_argument("--long-s", type=float, default=0.0, help="--per-line: only lines at least this long (s)")
    ap.add_argument("--per-line", type=float, nargs="+", default=None,
                    help="also check each line on its own, these floors (ms)")
    ap.add_argument("--tempo-pct", type=float, nargs="+", default=[0.5],
                    help="--engine: COLLAPSE_LINES_TEMPO_PCT, a line's heard tempo off the one grid's")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    import rankable
    from beat_lines import phase_by_beats
    beats = json.load(gzip.open(args.beats, "rt", encoding="utf-8"))
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    if args.phase_first:
        for m in maps:
            b = beats.get(m["audio_sha1"])
            if b and m.get("lines"):
                moved = phase_by_beats([(float(o), float(bp)) for o, bp, *_ in m["lines"]], b, m["duration_s"] or 0.0)
                m["lines"] = [[o, bp, *rest] for (o, bp), (_o, _b, *rest) in zip(moved, m["lines"])]
    shift0 = rankable.beat_shift([m for m in maps if m.get("split") == "dev"])
    for m in maps:
        m["rank"] = rankable.verdict(m, shift0)
    import itertools
    global LONG_S
    LONG_S = args.long_s
    for agree, slack, floor, drift in itertools.product(args.agree_pct, args.slack, args.per_line or [None], args.tempo_pct):
        if True:
            after, tried = [], 0
            for m in maps:
                lines = [(float(o), float(bp)) for o, bp, *_ in m.get("lines") or []]
                b = beats.get(m["audio_sha1"])
                end = (m.get("duration_s") or 0.0) * 1000.0
                if len(lines) < 2 or not b or end <= 0:
                    after.append(dict(m))
                    continue
                bm = np.sort(np.asarray(b, dtype=float)) * 1000.0
                if args.engine:
                    from overtone import beat_this, params
                    params.COLLAPSE_LINES_AGREE_PCT, params.COLLAPSE_LINES_SHARE = agree, args.share
                    params.COLLAPSE_LINES_SLACK, params.COLLAPSE_LINES_LONG_S = slack, args.long_s
                    params.COLLAPSE_LINES_FLOOR_MS = floor if floor is not None else 1e9
                    params.COLLAPSE_LINES_TEMPO_PCT = drift
                    new = beat_this._collapse_by_beat_this(lines, bm, end)
                    if new is None:
                        after.append(dict(m))
                    else:
                        tried += 1
                        after.append(dict(m, lines=[[o, bp] for o, bp, *_k in new]))
                    continue
                one = one_of(lines, end, agree, args.share)
                if one is None:
                    after.append(dict(m))
                    continue
                c_many = grid_cost(lines, bm, lines[0][0], end)
                c_one = grid_cost([one], bm, lines[0][0], end)
                if c_many is None or c_one is None or c_one > slack * c_many or                         (floor is not None and not per_line_ok(lines, one, bm, end, slack, floor)):
                    after.append(dict(m))
                    continue
                tried += 1
                after.append(dict(m, lines=[[one[0], one[1]]]))
            shift = rankable.beat_shift([m for m in after if m.get("split") == "dev"])
            for m in after:
                m["rank"] = rankable.verdict(m, shift)
            print(f"agree {agree} % slack {slack} per-line floor {floor} tempo {drift} %: {tried} maps collapsed; lead {shift:+.1f} ms")
            for split in ("dev", "holdout"):
                b0 = [m for m in maps if m.get("split") == split]
                a0 = [m for m in after if m.get("split") == split]
                sb, sa = rankable.summarise(b0), rankable.summarise(a0)
                fixed = sum(x["rank"]["verdict"] == "miss" and y["rank"]["verdict"] != "miss" for x, y in zip(b0, a0))
                broken = sum(x["rank"]["verdict"] != "miss" and y["rank"]["verdict"] == "miss" for x, y in zip(b0, a0))
                print(f"  {split:<8} songs {rankable.pct(sb['songs_rankable'], sb['songs'])} -> "
                      f"{rankable.pct(sa['songs_rankable'], sa['songs'])} % (fixed {fixed}, broken {broken} maps)")
                if args.list:
                    seen = set()
                    for x, y in zip(b0, a0):
                        if (x["rank"]["verdict"] == "miss") != (y["rank"]["verdict"] == "miss") and x["audio_sha1"] not in seen:
                            seen.add(x["audio_sha1"])
                            print("     ", x["rank"]["verdict"][:4], "->", y["rank"]["verdict"][:4],
                                  len(x.get("lines") or []), Path(x["folder_path"]).name[:40])
    return 0


if __name__ == "__main__":
    sys.exit(main())
