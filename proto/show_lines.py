"""Map lines beside engine lines for missed songs with few map lines.

    python proto/show_lines.py <stage report.json>... [--max-map 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench")]


def fmt(lines, end_ms, cap=8):
    out = []
    for n, (o, b) in enumerate(lines[:cap]):
        nxt = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        out.append(f"{o / 1000:.1f}s@{b:.2f}x{(nxt - o) / 1000:.0f}s")
    return " ".join(out) + (" ..." if len(lines) > cap else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--max-map", type=int, default=6)
    ap.add_argument("--min-map", type=int, default=2)
    ap.add_argument("--min-eng", type=int, default=0)
    args = ap.parse_args(argv)
    import rankable
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    shift = rankable.beat_shift([m for m in maps if m.get("split") == "dev"])
    seen = set()
    for m in maps:
        key = rankable.song_key(m)
        if key in seen:
            continue
        seen.add(key)
        ml = rankable.map_lines_of(m)
        if not args.min_map <= len(ml) <= args.max_map or (args.min_eng and len(m.get("lines") or []) < args.min_eng):
            continue
        r = rankable.verdict(m, shift)
        el = [(float(o), float(b)) for o, b, *_ in m.get("lines") or []]
        end = (m.get("duration_s") or 0) * 1000.0
        print(f"{r['verdict'][:4]} {Path(m['folder_path']).name[:44]} ({m.get('split')})")
        print("   map ", fmt(ml, end))
        print("   eng ", fmt(el, end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
