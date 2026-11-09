"""Prototype: each red line's offset moved toward where Beat This!'s beats sit.

The engine's BPM is often exact while its offset sits 10-20 ms from where the
mapper put the beat (the stage report's "offset" misses). Beat This! marks
beats from the whole mix, not from attack peaks; over a few hundred beats its
mean distance to the line's grid is an independent read of the phase. This
moves each line by w * (that distance - k), k fitted on dev so the lead stays
put, and scores the songs with bench/rankable.py.

    python proto/offset_by_beats.py <stage report.json>... --beats <beats.json.gz> [--w 0.5 1]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench")]


def residuals(lines, beats_ms, duration_ms, clip_ms):
    """Per line: the beats inside it, each one's signed distance to the line's grid."""
    out = []
    for n, (o, bpm) in enumerate(lines):
        step = 60000.0 / bpm
        start = o if n else -1e12
        end = lines[n + 1][0] if n + 1 < len(lines) else duration_ms
        b = beats_ms[(beats_ms >= start) & (beats_ms < end)]
        r = b - o - np.round((b - o) / step) * step
        out.append(r[np.abs(r) <= clip_ms])
    return out


def centre(r, how):
    if r.size == 0:
        return None
    if how == "median":
        return float(np.median(r))
    if how == "trim":
        lo, hi = np.percentile(r, [20, 80])
        k = r[(r >= lo) & (r <= hi)]
        return float(np.mean(k)) if k.size else float(np.median(r))
    return float(np.mean(r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--beats", required=True)
    ap.add_argument("--w", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--clip", type=float, default=40.0)
    ap.add_argument("--min-beats", type=int, default=32)
    ap.add_argument("--how", choices=["median", "mean", "trim"], default="trim")
    ap.add_argument("--single-only", action="store_true", help="only songs the engine gave one line")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--k", type=float, help="the network's lead against the engine (ms); fitted on dev if left out")
    ap.add_argument("--mad", type=float, help="move only lines whose beats' median absolute deviation is under this (ms)")
    ap.add_argument("--cap", type=float, help="no move larger than this (ms)")
    ap.add_argument("--phase-first", action="store_true",
                    help="reports from before PHASE_BEAT_THIS: apply its half-beat check first")
    args = ap.parse_args(argv)
    import rankable
    beats = json.load(gzip.open(args.beats, "rt", encoding="utf-8"))
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    if args.phase_first:
        sys.path.insert(0, str(ROOT / "proto"))
        from beat_lines import phase_by_beats
        for m in maps:
            b = beats.get(m["audio_sha1"])
            if b and m.get("lines"):
                moved = phase_by_beats([(float(o), float(bp)) for o, bp, *_ in m["lines"]], b, m["duration_s"] or 0.0)
                m["lines"] = [[o, bp, *rest] for (o, bp), (_o, _b, *rest) in zip(moved, m["lines"])]
    dev = [m for m in maps if m.get("split") == "dev"]
    shift0 = rankable.beat_shift(dev)
    for m in maps:
        m["rank"] = rankable.verdict(m, shift0)
    # each map's per-line distance from its grid to Beat This!'s beats
    moves = {}
    for m in maps:
        b = beats.get(m["audio_sha1"])
        lines = [(float(o), float(bp)) for o, bp, *_ in m.get("lines") or []]
        if not b or not lines or (args.single_only and len(lines) != 1):
            continue
        rs = residuals(lines, np.asarray(b) * 1000.0, (m["duration_s"] or 0.0) * 1000.0, args.clip)
        moves[id(m)] = [centre(r, args.how) if r.size >= args.min_beats and
                        (args.mad is None or float(np.median(np.abs(r - np.median(r)))) <= args.mad) else None
                        for r in rs]
    # k: the typical distance on dev (the network's own lead against the engine's)
    ds = [d for m in dev if id(m) in moves for d in moves[id(m)] if d is not None]
    k = args.k if args.k is not None else (float(np.median(ds)) if ds else 0.0)
    print(f"lead {shift0:+.1f} ms, Beat This! vs engine median {k:+.1f} ms over {len(ds)} dev lines, "
          f"spread p10/p90 {np.percentile(ds, 10):+.1f}/{np.percentile(ds, 90):+.1f}")
    for w in args.w:
        after = []
        for m in maps:
            mv = moves.get(id(m))
            if not mv:
                after.append(dict(m))
                continue
            new = []
            for (o, bp, *rest), d in zip(m["lines"], mv):
                mv_ms = w * (d - k) if d is not None else 0.0
                if args.cap is not None:
                    mv_ms = float(np.clip(mv_ms, -args.cap, args.cap))
                new.append([o + mv_ms, bp, *rest])
            after.append(dict(m, lines=new))
        shift = rankable.beat_shift([m for m in after if m.get("split") == "dev"])
        for m in after:
            m["rank"] = rankable.verdict(m, shift)
        print(f"w {w:.2f}: lead {shift:+.1f} ms")
        for split in ("dev", "holdout"):
            b = [m for m in maps if m.get("split") == split]
            a = [m for m in after if m.get("split") == split]
            sb, sa = rankable.summarise(b), rankable.summarise(a)
            fixed = sum(x["rank"]["verdict"] == "miss" and y["rank"]["verdict"] != "miss" for x, y in zip(b, a))
            broken = sum(x["rank"]["verdict"] != "miss" and y["rank"]["verdict"] == "miss" for x, y in zip(b, a))
            print(f"  {split:<8} songs {rankable.pct(sb['songs_rankable'], sb['songs'])} -> "
                  f"{rankable.pct(sa['songs_rankable'], sa['songs'])} %, exact "
                  f"{rankable.pct(sb.get('songs_exact', 0), sb['songs'])} -> {rankable.pct(sa.get('songs_exact', 0), sa['songs'])} "
                  f"(fixed {fixed}, broken {broken} maps)")
            if args.list:
                seen = set()
                for x, y in zip(b, a):
                    if x["rank"]["verdict"] != y["rank"]["verdict"] and x["audio_sha1"] not in seen:
                        seen.add(x["audio_sha1"])
                        print("     ", x["rank"]["verdict"][:4], "->", y["rank"]["verdict"][:4],
                              f"{x['rank']['abs_ms'] or 0:.1f} -> {y['rank']['abs_ms'] or 0:.1f}",
                              Path(x["folder_path"]).name[:40])
    return 0


if __name__ == "__main__":
    sys.exit(main())
