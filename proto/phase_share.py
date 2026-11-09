"""Prototype: the engine's half-beat check by Beat This! (and the line
collapse after it) replayed on stage reports taken without either, at
several PHASE_BEAT_THIS_SHARE values.

The `signatures` gate's six-beat bars put a pulse at twice the line's tempo;
the network switched between its halves (59 % of its beats on one) and the
median alone moved those lines off their proven downbeat. This measures
what requiring a share of the beats on the half costs on real songs.

    python proto/phase_share.py <stage report.json>... --beats <beats.json.gz> --share 0 0.7 0.75 0.8 [--list]
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


def replay(maps, beats, share):
    from overtone import beat_this, params
    params.PHASE_BEAT_THIS_SHARE = share
    out, moved = [], 0
    for m in maps:
        lines = [(float(o), float(bp)) for o, bp, *_ in m.get("lines") or []]
        b = beats.get(m["audio_sha1"])
        end = (m.get("duration_s") or 0.0) * 1000.0
        if not lines or not b or len(b) < params.PHASE_BEAT_THIS_MIN_BEATS or end <= 0:
            out.append(dict(m))
            continue
        bm = np.sort(np.asarray(b, dtype=float)) * 1000.0
        new, changed = beat_this._half_beat_by_beat_this(lines, bm, end)
        if params.COLLAPSE_LINES_BEAT_THIS and len(new) > 1:
            collapsed = beat_this._collapse_by_beat_this(new, bm, end)
            if collapsed is not None:
                new, changed = [(o, bp) for o, bp, _k in collapsed], True
        moved += changed
        out.append(dict(m, lines=[[o, bp] for o, bp in new]))
    return out, moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--beats", required=True)
    ap.add_argument("--share", type=float, nargs="+", default=[0.0, 0.75])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    import rankable
    beats = json.load(gzip.open(args.beats, "rt", encoding="utf-8"))
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    runs = {}
    for share in args.share:
        after, moved = replay(maps, beats, share)
        shift = rankable.beat_shift([m for m in after if m.get("split") == "dev"])
        for m in after:
            m["rank"] = rankable.verdict(m, shift)
        runs[share] = after
        print(f"share {share}: {moved} maps changed; lead {shift:+.1f} ms")
    base = runs[args.share[0]]
    for share in args.share:
        print(f"== share {share} against share {args.share[0]}")
        for split in ("dev", "holdout"):
            b0 = [m for m in base if m.get("split") == split]
            a0 = [m for m in runs[share] if m.get("split") == split]
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
                              Path(x["folder_path"]).name[:40])
    return 0


if __name__ == "__main__":
    sys.exit(main())
