"""Would this timing be ranked? One verdict per map, decided by its worst beat.

The stage figures (bench/stages.py) are shares and means: lines within 5 ms,
beat F-measure, tempo ok per red line. A mean hides the one section that
drifts 20 ms, and a player hears that section. This scores each map the way a
modder checks a timing: beat by beat, from the map's first red line to the end
of the song, and the map's verdict is its **worst** beat (owner's decision,
2026-11-08: timings that work almost 10/10, then the averages).

For every map beat the nearest engine beat, and for every engine beat the
nearest map beat (a grid read at double time has every other beat on nothing,
so both directions are needed). Two worst-beat readings:

- **lock**: after the map's own constant offset is taken off (the median beat
  distance within 50 ms). Says whether the grid holds the song's tempo, octave
  and changes, whatever the offset; the ranking convention cannot hide or
  cause a miss here.
- **absolute**: after one constant for every map, the dev part's median beat
  distance (the mappers' lead, about +25 ms on the raw grid). What the export
  would hand a mapper, mapper-to-mapper spread included (sd ~8 ms, 27.11).

Verdicts, on the absolute reading: **exact** within 5 ms, **rankable** within
10 ms, else a miss, with the miss's cause: refused, octave, tempo (a section's
BPM off by more than 0.05), lines (BPMs right, a change missed, added or
misplaced), offset (locked within 10 ms, only the constant is out).

Why 10 ms is the rankable bar: it is how far two ranked timings of the same
audio file stand apart. Every run prints that reference from the reports' own
maps: audios ranked in two or more sets, one map per set, copied timings left
out (2026-11-08, 50 independent pairs: worst beat within 5 ms 36 %, within 10 ms
74 %, within 15 ms 88 %; offsets apart by a median 5 ms, 12 ms at the 90th
percentile). Holding the engine to 5 ms against one map asks more of it than
ranked maps manage against each other; exact is kept beside it as the stretch.
Decimal BPMs never count against a grid: only where its beats land does.

A song counts once (its audio hash): it passes if the engine's grid passes
against at least one of its ranked maps (difficulties that disagree among
themselves are reference conflicts, 27.11), and the share against all of them
is printed beside it.

    .venv/Scripts/python.exe bench/rankable.py bench/.cache/remote/<id>.json
    .venv/Scripts/python.exe bench/rankable.py <dev report> <held-out report>  # one stage, two runs
    .venv/Scripts/python.exe bench/rankable.py <report> --worst 20   # the dev misses to look at

Reads a stage report (bench/stages.py writes one; bench/remote.py result
fetches the server's) and nothing else. A measurement, not a gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

EXACT_MS = 5.0
RANKABLE_MS = 10.0
#: A map's own offset is the median beat distance within this window.
OFFSET_WINDOW_MS = 50.0
#: How far past the scored span the other grid is drawn, so edge beats find their partner.
EDGE_MS = 2000.0
BPM_BAR = 0.05
CAUSES = ("refused", "octave", "tempo", "lines", "offset")


def grid_beats_ms(lines: list[tuple[float, float]], start_ms: float, end_ms: float) -> np.ndarray:
    """Beats of red lines ``(offset_ms, bpm)`` from ``start_ms`` to ``end_ms``,
    each line running to the next, the first one back to 0 as osu! runs it
    (or to ``start_ms`` when that is earlier)."""
    beats = []
    for n, (offset, bpm) in enumerate(lines):
        if bpm <= 0:
            continue
        step = 60000.0 / bpm
        stop = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        k = -math.floor((offset - min(0.0, start_ms)) / step) if n == 0 else 0
        while offset + k * step < stop:
            t = offset + k * step
            if t >= start_ms - 1e-6:
                beats.append(t)
            k += 1
    return np.asarray(beats, dtype=float)


def nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Signed distance from each ``src`` beat to the nearest ``dst`` beat."""
    i = np.clip(np.searchsorted(dst, src), 1, dst.size - 1) if dst.size > 1 else np.zeros(src.size, int)
    left, right = dst[np.maximum(i - 1, 0)], dst[i]
    return np.where(np.abs(left - src) <= np.abs(right - src), left, right) - src


def beat_errors(map_lines, engine_lines, duration_s: float) -> np.ndarray | None:
    """Every beat's signed error (ms, positive when the engine is late): the
    map's beats against the engine's and the engine's against the map's.
    None when the engine gave no grid."""
    if not engine_lines or not map_lines:
        return None
    start = map_lines[0][0]
    end = max(duration_s * 1000.0, start)
    # Each side is judged inside [start, end] against the other's whole grid,
    # past both edges: a grid whose first beat sits a few ms before the map's
    # first red line must still find it, not the beat after (a false ~300 ms).
    mb_all = grid_beats_ms(map_lines, start - EDGE_MS, end + EDGE_MS)
    eb_all = grid_beats_ms(engine_lines, start - EDGE_MS, end + EDGE_MS)
    mb = mb_all[(mb_all >= start) & (mb_all <= end)]
    eb = eb_all[(eb_all >= start) & (eb_all <= end)]
    if mb.size == 0 or eb.size == 0 or mb_all.size == 0 or eb_all.size == 0:
        return None
    return np.concatenate([nearest(mb, eb_all), -nearest(eb, mb_all)])


def own_offset(errors: np.ndarray) -> float:
    near = errors[np.abs(errors) <= OFFSET_WINDOW_MS]
    return float(np.median(near)) if near.size else 0.0


def verdict(m: dict, shift_ms: float) -> dict:
    """One map's worst beats and verdict. ``m`` is a stage report's map row."""
    map_lines = [(float(r["offset_ms"]), float(r["bpm"])) for r in m["rows"]]
    engine_lines = [(float(o), float(b)) for o, b, *_ in m.get("lines") or []]
    errors = beat_errors(map_lines, engine_lines, m.get("duration_s") or 0.0)
    out = {"beats": 0, "lock_ms": None, "abs_ms": None, "own_offset_ms": None,
           "beats_over_5": None, "verdict": "miss", "cause": "refused"}
    if errors is None:
        return out
    offset = own_offset(errors)
    lock = float(np.max(np.abs(errors - offset)))
    absolute = float(np.max(np.abs(errors - shift_ms)))
    out.update(beats=int(errors.size), lock_ms=lock, abs_ms=absolute, own_offset_ms=offset,
               beats_over_5=int(np.sum(np.abs(errors - shift_ms) > EXACT_MS)))
    if absolute <= EXACT_MS:
        out.update(verdict="exact", cause=None)
    elif absolute <= RANKABLE_MS:
        out.update(verdict="rankable", cause=None)
    else:
        rows = m["rows"]
        if any(r["octave"] is not None and r["octave"] != 1 for r in rows):
            out["cause"] = "octave"
        elif any(r["bpm_error"] is None or r["bpm_error"] > BPM_BAR for r in rows):
            out["cause"] = "tempo"
        elif lock > RANKABLE_MS:
            out["cause"] = "lines"
        else:
            out["cause"] = "offset"
    return out


def beat_shift(maps: list[dict]) -> float:
    """The mappers' lead: the dev maps' median beat distance within 50 ms."""
    pooled = []
    for m in maps:
        map_lines = [(float(r["offset_ms"]), float(r["bpm"])) for r in m["rows"]]
        engine_lines = [(float(o), float(b)) for o, b, *_ in m.get("lines") or []]
        e = beat_errors(map_lines, engine_lines, m.get("duration_s") or 0.0)
        if e is not None:
            pooled.append(e[np.abs(e) <= OFFSET_WINDOW_MS])
    pooled = np.concatenate(pooled) if pooled else np.zeros(0)
    return float(np.median(pooled)) if pooled.size else 0.0


def song_key(m: dict) -> str:
    return m.get("audio_sha1") or m["folder"]


def map_lines_of(m: dict) -> list[tuple[float, float]]:
    return [(float(r["offset_ms"]), float(r["bpm"])) for r in m["rows"]]


def ranked_pairs(maps: list[dict]) -> np.ndarray:
    """The human reference: two ranked timings of one audio file, from two
    sets (one map per set), scored against each other as the engine is.
    Timings within 0.5 ms everywhere are copies, not a second timer, and are
    left out. Rows: worst beat (ms, no shift: both follow the ranking
    convention) and worst beat with the pair's own offset off."""
    by_audio: dict[str, dict[str, dict]] = {}
    for m in maps:
        by_audio.setdefault(song_key(m), {}).setdefault(m["folder"], m)
    rows = []
    for sets in by_audio.values():
        chosen = list(sets.values())
        for i, a in enumerate(chosen):
            for b in chosen[i + 1:]:
                errors = beat_errors(map_lines_of(a), map_lines_of(b), a.get("duration_s") or 0.0)
                if errors is None:
                    continue
                worst = float(np.max(np.abs(errors)))
                if worst <= 0.5:
                    continue
                rows.append((worst, float(np.max(np.abs(errors - own_offset(errors))))))
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def summarise(maps: list[dict]) -> dict:
    n = len(maps)
    counts = Counter(m["rank"]["verdict"] for m in maps)
    causes = Counter(m["rank"]["cause"] for m in maps if m["rank"]["verdict"] == "miss")

    def locked(m: dict) -> bool:
        return m["rank"]["lock_ms"] is not None and m["rank"]["lock_ms"] <= RANKABLE_MS

    by_song: dict[str, list[dict]] = {}
    for m in maps:
        by_song.setdefault(song_key(m), []).append(m)
    songs = list(by_song.values())
    return {"maps": n, "songs": len(by_song),
            "exact": counts["exact"], "rankable": counts["exact"] + counts["rankable"],
            "miss": counts["miss"], "locked": sum(map(locked, maps)),
            "causes": {c: causes.get(c, 0) for c in CAUSES},
            "songs_exact": sum(any(m["rank"]["verdict"] == "exact" for m in ms) for ms in songs),
            "songs_rankable": sum(any(m["rank"]["verdict"] != "miss" for m in ms) for ms in songs),
            "songs_rankable_all": sum(all(m["rank"]["verdict"] != "miss" for m in ms) for ms in songs),
            "songs_locked": sum(any(map(locked, ms)) for ms in songs)}


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f}" if whole else "  -  "


def print_summary(name: str, s: dict) -> None:
    n, songs = s["maps"], s["songs"]
    print(f"  {name:<9} {n:>5} maps {songs:>4} songs")
    print(f"  {'':<9} songs : rankable {pct(s['songs_rankable'], songs)} %"
          f" (against every map {pct(s['songs_rankable_all'], songs)} %),"
          f" exact {pct(s['songs_exact'], songs)} %,"
          f" grid locked {pct(s['songs_locked'], songs)} %")
    print(f"  {'':<9} maps  : rankable {pct(s['rankable'], n)} %, exact {pct(s['exact'], n)} %,"
          f" grid locked {pct(s['locked'], n)} %")
    print(f"  {'':<9} why a miss ({s['miss']} maps): " + ", ".join(
        f"{c} {pct(s['causes'][c], s['miss']).strip()} %" for c in CAUSES))


def print_pairs(pairs: np.ndarray) -> None:
    if not len(pairs):
        print("  ranked against ranked: no audio is ranked in two sets with different timings")
        return
    print(f"  ranked against ranked ({len(pairs)} pairs of sets on one audio, copies left out):"
          f" rankable {pct(int(np.sum(pairs[:, 0] <= RANKABLE_MS)), len(pairs))} %,"
          f" exact {pct(int(np.sum(pairs[:, 0] <= EXACT_MS)), len(pairs))} %,"
          f" grid locked {pct(int(np.sum(pairs[:, 1] <= RANKABLE_MS)), len(pairs))} %")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+",
                        help="stage reports (bench/stages.py or bench/remote.py result) of one "
                             "stage and engine; a held-out run is read with its dev run's lead")
    parser.add_argument("--worst", type=int, default=0, metavar="N",
                        help="list the N dev maps nearest to rankable that still miss")
    parser.add_argument("--json", metavar="OUT", help="write every map's verdict here")
    args = parser.parse_args(argv)
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    report = reports[0]
    for other in reports[1:]:
        for key in ("engine", "fingerprint"):
            if other.get(key) != report.get(key):
                parser.error(f"the reports differ in {key}: {report.get(key)} and {other.get(key)}")
        if (other.get("stage") or {}).get("id") != (report.get("stage") or {}).get("id"):
            parser.error("the reports are of different stages")
    maps = [m for r in reports for m in r["maps"]]
    if len({(m.get("split"), m.get("audio_sha1") or m["folder"], m["file"]) for m in maps}) < len(maps):
        parser.error("a map is in two of the reports")
    dev = [m for m in maps if m.get("split") == "dev"]
    shift = beat_shift(dev)
    for m in maps:
        m["rank"] = verdict(m, shift)
    stage = report.get("stage") or {}
    print(f"{stage.get('id', '?')}, engine {report.get('engine')} {report.get('fingerprint')},"
          f" run {report.get('run')}; lead fitted on dev: {shift:+.1f} ms;"
          f" rankable <= {RANKABLE_MS:g} ms, exact <= {EXACT_MS:g} ms (worst beat)")
    pairs = ranked_pairs(maps)
    print_pairs(pairs)
    summary = {"lead_ms": shift, "ranked_pairs": len(pairs)}
    for name in ("dev", "holdout", "confirmation"):
        part = [m for m in maps if m.get("split") == name]
        if part:
            summary[name] = summarise(part)
            print_summary("held out" if name == "holdout" else name, summary[name])
    if args.worst:
        misses = sorted((m for m in dev if m["rank"]["verdict"] == "miss"
                         and m["rank"]["abs_ms"] is not None),
                        key=lambda m: m["rank"]["abs_ms"])
        print(f"\n  dev maps nearest to rankable that miss (first {args.worst}):")
        for m in misses[:args.worst]:
            r = m["rank"]
            print(f"    {r['abs_ms']:6.1f} ms abs, {r['lock_ms']:6.1f} lock, own offset"
                  f" {r['own_offset_ms']:+6.1f}, {r['beats_over_5']:>4}/{r['beats']} beats over 5 ms,"
                  f" {r['cause']:<6} {Path(m['folder']).name[:60]}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "maps": [
            {"file": m["file"], "folder": Path(m["folder"]).name, "split": m.get("split"),
             **m["rank"]} for m in maps]}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
