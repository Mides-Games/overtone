"""Prototype: a stretch of a red line where Beat This!'s beats sit steadily
off the line's phase (not half a beat: the half-beat check owns that) gets a
line of its own, at the same BPM, moved by the stretch's median offset.

Sleeping With Sirens' CTRL + ALT + DEL: the engine writes one 210 BPM line
anchored at 65 s; the network's beats sit 40 ms early from 9 s to 64 s and on
it after; the map has a line at 66 s.

    python proto/phase_split.py <stage report.json>... --beats <beats.json.gz> --shift 15 20 [--list]
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

WIN = 8


def split(lines, bm, end_ms, min_shift, min_windows, spread):
    out, changed = [], False
    for n, (o, bp) in enumerate(lines):
        lo = 0.0 if n == 0 else o
        hi = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        b = bm[(bm >= lo) & (bm < hi)]
        step = 60000.0 / bp
        if b.size < WIN * (min_windows + 2):
            out.append((o, bp))
            continue
        r = (b - o) / step
        r = (r - np.round(r)) * step
        nw = b.size // WIN
        med = np.array([np.median(r[i * WIN:(i + 1) * WIN]) for i in range(nw)])
        base = float(np.median(med))
        off = med - base
        # windows steadily off, not on the other half
        bad = (np.abs(off) >= min_shift) & (np.abs(off) < 0.3 * step)
        runs, i = [], 0
        while i < nw:
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < nw and bad[j + 1] and np.sign(off[j + 1]) == np.sign(off[i]):
                j += 1
            seg = off[i:j + 1]
            if j - i + 1 >= min_windows and np.ptp(seg) <= spread * step:
                runs.append((i, j, float(np.median(seg))))
            i = j + 1
        if not runs:
            out.append((o, bp))
            continue
        pieces = [(o, bp)]
        for i, j, s in runs:
            t0 = b[i * WIN]
            t1 = b[min((j + 1) * WIN, b.size - 1)]
            k0 = np.floor((t0 - o) / step)
            k1 = np.ceil((t1 - o) / step)
            pieces.append((o + k0 * step + s, bp))
            if (j + 1) * WIN < b.size - WIN:
                pieces.append((o + k1 * step, bp))
        pieces.sort()
        # the first line of the span keeps its own start when it moved before it
        first = pieces[0][0]
        pieces = [p for p in pieces if p[0] >= lo - step] or [(first, bp)]
        out.extend(pieces)
        changed = True
    out.sort()
    return out, changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--beats", required=True)
    ap.add_argument("--shift", type=float, nargs="+", default=[20.0])
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--spread", type=float, default=0.08)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    import rankable
    beats = json.load(gzip.open(args.beats, "rt", encoding="utf-8"))
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    shift = rankable.beat_shift([m for m in maps if m.get("split") == "dev"])
    for m in maps:
        m["rank"] = rankable.verdict(m, shift)
    for ms in args.shift:
        after, moved = [], 0
        for m in maps:
            lines = [(float(o), float(bp)) for o, bp, *_ in m.get("lines") or []]
            b = beats.get(m["audio_sha1"])
            end = (m.get("duration_s") or 0.0) * 1000.0
            if not lines or not b or end <= 0:
                after.append(m)
                continue
            bm = np.sort(np.asarray(b, dtype=float)) * 1000.0
            new, changed = split(lines, bm, end, ms, args.windows, args.spread)
            moved += changed
            after.append(dict(m, lines=[[o, bp] for o, bp in new]) if changed else m)
        sh = rankable.beat_shift([m for m in after if m.get("split") == "dev"])
        for m in after:
            m["rank2"] = rankable.verdict(m, sh)
        print(f"== shift {ms} ms: {moved} maps changed; lead {shift:+.1f} -> {sh:+.1f} ms")
        for split_name in ("dev", "holdout"):
            b0 = [m for m in maps if m.get("split") == split_name]
            a0 = [m for m in after if m.get("split") == split_name]
            sa = rankable.summarise([dict(m, rank=m["rank2"]) for m in a0])
            sb = rankable.summarise(b0)
            fixed = sum(x["rank"]["verdict"] == "miss" and y["rank2"]["verdict"] != "miss" for x, y in zip(b0, a0))
            broken = sum(x["rank"]["verdict"] != "miss" and y["rank2"]["verdict"] == "miss" for x, y in zip(b0, a0))
            print(f"  {split_name:<8} songs {rankable.pct(sb['songs_rankable'], sb['songs'])} -> "
                  f"{rankable.pct(sa['songs_rankable'], sa['songs'])} % (fixed {fixed}, broken {broken} maps)")
            if args.list:
                seen = set()
                for x, y in zip(b0, a0):
                    if x is not y and x["audio_sha1"] not in seen:
                        seen.add(x["audio_sha1"])
                        a, c = x["rank"].get("abs_ms"), y["rank2"].get("abs_ms")
                        print("     ", x["rank"]["verdict"][:4], "->", y["rank2"]["verdict"][:4],
                              f"{a if a is None else round(a, 1)} -> {c if c is None else round(c, 1)}",
                              len(x["lines"]), "->", len(y["lines"]), Path(x["folder_path"]).name[:40])
    return 0


if __name__ == "__main__":
    sys.exit(main())
