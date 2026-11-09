"""Where a stage report's songs miss: per song (first map), the cause, the
worst beat locked to its own offset and against the dev lead, the BPM gap and
the line counts.

    python proto/miss_causes.py <stage report.json>... [--list]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    import rankable
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    shift = rankable.beat_shift([m for m in maps if m.get("split") == "dev"])
    by_song = collections.defaultdict(list)
    for m in maps:
        m["rank"] = rankable.verdict(m, shift)
        by_song[rankable.song_key(m)].append(m)
    tally = collections.Counter()
    rows = []
    for key, ms in by_song.items():
        best = min(ms, key=lambda m: m["rank"]["abs_ms"] if m["rank"]["abs_ms"] is not None else 1e9)
        r = best["rank"]
        n_map = len(rankable.map_lines_of(best))
        n_eng = len(best.get("lines") or [])
        kind = "single" if n_map == 1 else "multi"
        if r["verdict"] != "miss":
            tally[(kind, "ok")] += 1
            continue
        tally[(kind, r["cause"])] += 1
        bpm_err = max((x["bpm_error"] or 0.0) for x in best["rows"]) if best["rows"] else None
        rows.append((kind, r["cause"], r["lock_ms"], r["abs_ms"], r["own_offset_ms"], bpm_err, n_map, n_eng,
                     Path(best["folder_path"]).name[:40], best.get("split")))
    for k in sorted(tally):
        print(k, tally[k])
    if args.list:
        for row in sorted(rows, key=lambda x: (x[0], x[1], x[3] or 0)):
            kind, cause, lock, ab, own, be, nm, ne, name, split = row
            f = lambda v: "-" if v is None else f"{v:.1f}"
            print(f"{kind:<6} {cause:<7} lock {f(lock):>6} abs {f(ab):>6} own {f(own):>6} bpm± {f(be):>5} "
                  f"lines {nm}/{ne} {split:<7} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
