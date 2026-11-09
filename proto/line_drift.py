"""Prototype: a red line whose BPM drifts against Beat This!'s beats over the
whole span it governs (the first line back to 0 s, as osu! plays it) takes
the tempo those beats count, turned about its own offset.

The engine fits a line where its attacks are clearest and may round its BPM
to a whole number there; over the minutes the line then governs, a few
hundredths of a BPM drift tens of ms (Through the Fire and Flames: 200.00
against the map's 199.97, its one line anchored at 385 s).

    python proto/line_drift.py <stage report.json>... --beats <beats.json.gz> --drift 6 10 15 [--list]
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


def refit(lines, bm, end_ms, drift_ms, steady, decimals):
    out, changed = [], False
    for n, (o, bp) in enumerate(lines):
        lo = 0.0 if n == 0 else o
        hi = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        b = bm[(bm >= lo) & (bm < hi)]
        step = 60000.0 / bp
        if b.size < 32:
            out.append((o, bp))
            continue
        k = np.round((b - o) / step)
        if np.ptp(k) < 16:
            out.append((o, bp))
            continue
        res = b - (o + k * step)
        res = res - np.median(res)
        keep = np.abs(res) <= steady * step
        if keep.mean() < 0.8:
            out.append((o, bp))
            continue
        kk, bb = k[keep], b[keep]
        # Theil-Sen on pairs a fixed number of beats apart: robust to the
        # stretches where the network sits on the other half
        gap = max(8, int(np.ptp(kk) // 4))
        idx = np.searchsorted(kk, kk + gap)
        ok = idx < kk.size
        if ok.sum() < 16:
            out.append((o, bp))
            continue
        slopes = (bb[idx[ok]] - bb[ok]) / (kk[idx[ok]] - kk[ok])
        q1, q3 = np.percentile(slopes, [25, 75])
        fit = [float(np.median(slopes))]
        if (q3 - q1) > 0.5 * abs(fit[0] - step) or fit[0] <= 0:
            out.append((o, bp))
            continue
        # the drift the line's own BPM builds against the heard one, at the
        # far end of its span from its offset
        reach = max(abs(k.min()), abs(k.max()))
        if abs(fit[0] - step) * reach <= drift_ms:
            out.append((o, bp))
            continue
        new = round(60000.0 / float(fit[0]), decimals)
        out.append((o, new))
        changed = True
    return out, changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--beats", required=True)
    ap.add_argument("--drift", type=float, nargs="+", default=[10.0])
    ap.add_argument("--steady", type=float, default=0.1)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    import rankable
    beats = json.load(gzip.open(args.beats, "rt", encoding="utf-8"))
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    shift = rankable.beat_shift([m for m in maps if m.get("split") == "dev"])
    for m in maps:
        m["rank"] = rankable.verdict(m, shift)
    for drift in args.drift:
        after, moved = [], 0
        for m in maps:
            lines = [(float(o), float(bp)) for o, bp, *_ in m.get("lines") or []]
            b = beats.get(m["audio_sha1"])
            end = (m.get("duration_s") or 0.0) * 1000.0
            if not lines or not b or end <= 0:
                after.append(m)
                continue
            bm = np.sort(np.asarray(b, dtype=float)) * 1000.0
            new, changed = refit(lines, bm, end, drift, args.steady, args.decimals)
            moved += changed
            after.append(dict(m, lines=[[o, bp] for o, bp in new]) if changed else m)
        sh = rankable.beat_shift([m for m in after if m.get("split") == "dev"])
        for m in after:
            m["rank2"] = rankable.verdict(m, sh)
        print(f"== drift {drift} ms: {moved} maps changed; lead {shift:+.1f} -> {sh:+.1f} ms")
        for split in ("dev", "holdout"):
            b0 = [m for m in maps if m.get("split") == split]
            a0 = [m for m in after if m.get("split") == split]
            sa = rankable.summarise([dict(m, rank=m["rank2"]) for m in a0])
            sb = rankable.summarise(b0)
            fixed = sum(x["rank"]["verdict"] == "miss" and y["rank2"]["verdict"] != "miss" for x, y in zip(b0, a0))
            broken = sum(x["rank"]["verdict"] != "miss" and y["rank2"]["verdict"] == "miss" for x, y in zip(b0, a0))
            print(f"  {split:<8} songs {rankable.pct(sb['songs_rankable'], sb['songs'])} -> "
                  f"{rankable.pct(sa['songs_rankable'], sa['songs'])} % (fixed {fixed}, broken {broken} maps)")
            if args.list:
                seen = set()
                for x, y in zip(b0, a0):
                    if x is not y and x["audio_sha1"] not in seen:
                        seen.add(x["audio_sha1"])
                        print("     ", x["rank"]["verdict"][:4], "->", y["rank2"]["verdict"][:4],
                              f"{x['rank'].get('abs_ms')} -> {y['rank2'].get('abs_ms')}",
                              [round(l[1], 3) for l in x["lines"]][:3], "->",
                              [round(l[1], 3) for l in y["lines"]][:3], Path(x["folder_path"]).name[:40])
    return 0


if __name__ == "__main__":
    sys.exit(main())
