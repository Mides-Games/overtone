"""Prototype: the Rust core's tempo sections as red lines, each change moved
to the beat where the attacks say it happens.

The core finds sections (start, end, period, phase) but writes one point per
song. Where its sections hold distinct tempos (after folding each into
MAP_OCTAVE_RANGE), this writes one line per tempo run: BPM and phase from the
run's longest section (tidied as the engine tidies), the change placed on
the new grid's beat whose composite grid best explains the attacks within
--window seconds of the section boundary.

    python proto/section_lines.py <replay.json.gz> <stage report.json>... [--list]

Scores with bench/rankable.py against the report's ranked maps. A measurement.
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


def fold(bpm: float, lo: float, hi: float) -> float:
    while bpm > hi:
        bpm /= 2.0
    while bpm < lo:
        bpm *= 2.0
    return bpm


def runs_of(sections, lo, hi, same_pct, min_s):
    """Adjacent sections at one folded tempo merged; runs shorter than min_s
    joined to their neighbour."""
    runs = []
    for s in sections:
        bpm = fold(float(s["bpm"]), lo, hi)
        if runs and abs(bpm / runs[-1]["bpm"] - 1.0) * 100.0 <= same_pct:
            r = runs[-1]
            r["end"] = float(s["end_s"])
            r["parts"].append((s, bpm))
            continue
        runs.append({"start": float(s["start_s"]), "end": float(s["end_s"]), "bpm": bpm, "parts": [(s, bpm)]})
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for n, r in enumerate(runs):
            if r["end"] - r["start"] < min_s:
                k = n - 1 if n > 0 else n + 1
                runs[k]["start"] = min(runs[k]["start"], r["start"])
                runs[k]["end"] = max(runs[k]["end"], r["end"])
                del runs[n]
                changed = True
                break
    return runs


def line_of(run, ta):
    """(offset ms, BPM) of a run: its longest section's grid, BPM tidied."""
    s, bpm = max(run["parts"], key=lambda p: float(p[0]["end_s"]) - float(p[0]["start_s"]))
    period = 60.0 / bpm
    beats = (run["end"] - run["start"]) / period
    new = ta._tidy_bpm(bpm, beats)
    phase = float(s["phase_s"]) % period
    mid = 0.5 * (float(s["start_s"]) + float(s["end_s"]))
    k = round((mid - phase) / period)
    anchor = (phase + k * period) * 1000.0  # a beat in the middle of the section
    # re-anchor the tidied grid on that beat; start the line at its run start
    step = 60000.0 / new
    o = anchor - np.floor((anchor - run["start"] * 1000.0) / step) * step
    return o, new


def best_change(prev, line, boundary_s, times, weights, window_s, fit):
    """The new line started on the beat of its grid near the boundary whose
    composite grid fits the attacks best."""
    from overtone import phrases
    po, pb = prev
    o, b = line
    step = 60000.0 / b
    lo, hi = (boundary_s - window_s) * 1000.0, (boundary_s + window_s) * 1000.0
    m = (times * 1000.0 >= lo) & (times * 1000.0 < hi)
    if m.sum() < 8:
        return line
    k0, k1 = int(np.ceil((lo - o) / step)), int(np.floor((hi - o) / step))
    best = None
    for k in range(k0, k1 + 1):
        at = o + k * step
        grid = np.concatenate([phrases._line_beats(po, pb, lo, at), phrases._line_beats(o, b, at, hi)])
        score = fit(grid, times[m], weights[m], 10.0)
        if best is None or score > best[0]:
            best = (score, at)
    return (best[1], b) if best else line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--same-pct", type=float, default=1.5)
    ap.add_argument("--min-s", type=float, default=8.0)
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--only-one", action="store_true", help="only where the run wrote one line")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    import overtone as ta
    import rankable
    from overtone import beat_this, params
    lo, hi = params.MAP_OCTAVE_RANGE
    bundle = json.load(gzip.open(args.replay, "rt", encoding="utf-8"))["audios"]
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    lines_for = {}
    for sha, v in bundle.items():
        rep = (v.get("replay") or {}).get("report") or {}
        secs = rep.get("sections") or []
        if len(secs) < 2:
            continue
        if args.only_one and len(v.get("lines") or []) != 1:
            continue
        runs = runs_of(secs, lo, hi, args.same_pct, args.min_s)
        if len(runs) < 2:
            continue
        ev = rep.get("evidence") or {}
        times = np.asarray(ev.get("attack_times") or [], dtype=float)
        weights = np.asarray(ev.get("attack_weights") or [], dtype=float)
        if weights.size != times.size:
            weights = np.ones(times.size)
        out = [line_of(runs[0], ta)]
        for r in runs[1:]:
            out.append(best_change(out[-1], line_of(r, ta), r["start"], times, weights, args.window,
                                   beat_this._grid_fit_score))
        lines_for[sha] = out
    dev = [m for m in maps if m.get("split") == "dev"]
    shift = rankable.beat_shift(dev) if dev else 26.3
    after = []
    for m in maps:
        m["rank"] = rankable.verdict(m, shift)
        use = lines_for.get(m["audio_sha1"])
        after.append(dict(m, lines=[[o, b] for o, b in use]) if use else dict(m))
    for m in after:
        m["rank"] = rankable.verdict(m, shift)
    print(f"section lines for {len(lines_for)} of {len(bundle)} audios; lead {shift:+.1f} ms")
    for split in ("dev", "holdout"):
        b = [m for m in maps if m.get("split") == split and m["audio_sha1"] in bundle]
        a = [m for m in after if m.get("split") == split and m["audio_sha1"] in bundle]
        if not b:
            continue
        if args.list:
            seen = set()
            for x, y in zip(b, a):
                if x["audio_sha1"] in lines_for and x["audio_sha1"] not in seen:
                    seen.add(x["audio_sha1"])
                    print("    ", x["rank"]["verdict"][:4], "->", y["rank"]["verdict"][:4],
                          Path(x["folder_path"]).name[:36], "map", [(round(o), round(bp, 1)) for o, bp in rankable.map_lines_of(x)][:5],
                          "new", [(round(o), round(bp, 2)) for o, bp in lines_for[x["audio_sha1"]]][:5])
        sb, sa = rankable.summarise(b), rankable.summarise(a)
        fixed = sum(x["rank"]["verdict"] == "miss" and y["rank"]["verdict"] != "miss" for x, y in zip(b, a))
        broken = sum(x["rank"]["verdict"] != "miss" and y["rank"]["verdict"] == "miss" for x, y in zip(b, a))
        print(f"  {split:<8} songs {rankable.pct(sb['songs_rankable'], sb['songs'])} -> "
              f"{rankable.pct(sa['songs_rankable'], sa['songs'])} % (fixed {fixed}, broken {broken} maps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
