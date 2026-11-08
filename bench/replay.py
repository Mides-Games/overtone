"""Replay a server run's conventions locally: an experiment in minutes, not hours.

Most accuracy ideas are steps after the Rust core (octave rules, line merging,
tidying, phase checks): python/overtone_rust.conventions. A server run keeps
the core's own report per audio (bench/stages.py write_replay; fetched with
``bench/remote.py replay <id>``), so those steps can be re-run here under any
flags, without the audio, and scored by bench/rankable.py against the same
ranked maps (roadmap 29.1).

    .venv/Scripts/python.exe bench/replay.py --check <dev report> <held-out report> \\
        --bundle <dev bundle> --bundle <held-out bundle>
    .venv/Scripts/python.exe bench/replay.py <reports...> --bundle ... --experiment merge-steady
    .venv/Scripts/python.exe bench/replay.py <reports...> --bundle ... --set PHASE_CHECK=True

``--check`` replays at the run's own flags and counts audios whose lines differ
from the report's: the replay is only trusted where that count is near zero.
Audios v3 answered (the Rust core refused) and audios whose replay needs the
audio itself (the tempo judge, the Beat This! vote) keep the run's lines and
are counted. A measurement, not a gate.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Lines within this many ms and BPM of the report's count as reproduced
#: (exported offsets are whole milliseconds).
SAME_MS = 0.6
SAME_BPM = 1e-3


def load_bundle(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _set_flags(flags: dict) -> tuple[list, dict]:
    """Switch the engine as the server's runs do (the vote off) plus
    ``flags``; returns what was there, for _restore_flags."""
    import overtone as ta
    session = list(ta._BEAT_THIS_SESSION)
    before = {}
    for name in flags:
        if not hasattr(ta, name):
            raise SystemExit(f"no engine flag {name}")
        before[name] = getattr(ta, name)
    ta._BEAT_THIS_SESSION[:] = [None]
    for name, value in flags.items():
        setattr(ta, name, value)
    return session, before


def _restore_flags(saved: tuple[list, dict]) -> None:
    import overtone as ta
    session, before = saved
    ta._BEAT_THIS_SESSION[:] = session
    for name, value in before.items():
        setattr(ta, name, value)


def replay_one(job: tuple[str, dict, dict]) -> tuple[str, list | None, str | None]:
    """One audio's exported lines under ``flags``, or why it cannot be replayed."""
    audio, entry, flags = job
    saved = _set_flags(flags)
    try:
        return _replay(audio, entry)
    finally:
        # Called in-process (the tests), the engine is left as it was found.
        _restore_flags(saved)


def _replay(audio: str, entry: dict) -> tuple[str, list | None, str | None]:
    import corpus_b
    import overtone_rust
    capture = entry.get("replay") or {}
    report = capture.get("report")
    if entry.get("backend") != "rust" or capture.get("returncode") != 0 or report is None:
        return audio, None, "not rust" if entry.get("backend") != "rust" else "no report"
    try:
        analysis = overtone_rust.analysis_from_report(report)
        analysis = overtone_rust.conventions(analysis, "", replay=capture)
        return audio, corpus_b.exported_lines(analysis), None
    except overtone_rust.ReplayMissing as exc:
        return audio, None, f"needs audio: {exc}"


def same_lines(a: list, b: list) -> bool:
    return len(a) == len(b) and all(abs(x[0] - y[0]) <= SAME_MS and abs(x[1] - y[1]) <= SAME_BPM
                                    for x, y in zip(a, b))


def parse_set(items: list[str]) -> dict:
    flags = {}
    for item in items:
        name, _, value = item.partition("=")
        try:
            flags[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise SystemExit(f"--set {item}: the value must be a Python literal") from None
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="stage reports of one stage (dev, held out)")
    parser.add_argument("--bundle", action="append", required=True,
                        help="the runs' replay bundles (bench/remote.py replay <id>)")
    parser.add_argument("--experiment", help="flags of python/overtone_runs.py EXPERIMENTS")
    parser.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                        help="an engine flag, e.g. MERGE_STEADY_LINES=True")
    parser.add_argument("--check", action="store_true",
                        help="replay at the run's flags and count lines that differ")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--worst", type=int, default=0, metavar="N",
                        help="name the N maps the change made worse")
    args = parser.parse_args(argv)
    import rankable

    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    flags: dict = {}
    if args.experiment:
        import overtone_runs
        if args.experiment not in overtone_runs.EXPERIMENTS:
            parser.error(f"no experiment {args.experiment}")
        flags.update(overtone_runs.EXPERIMENTS[args.experiment])
    flags.update(parse_set(args.set))
    run_experiments = {(r.get("experiment") or "") for r in reports}
    if args.check and flags:
        parser.error("--check replays the run's own flags: drop --experiment / --set")

    audios: dict[str, dict] = {}
    for path in args.bundle:
        audios.update(load_bundle(Path(path))["audios"])
    jobs = [(audio, entry, flags) for audio, entry in audios.items()]
    replayed: dict[str, list] = {}
    skipped: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for audio, lines, why in pool.map(replay_one, jobs, chunksize=8):
            if lines is None:
                key = why.split(":")[0]
                skipped[key] = skipped.get(key, 0) + 1
            else:
                replayed[audio] = lines
    print(f"{len(audios)} audios: {len(replayed)} replayed"
          + "".join(f", {n} {why}" for why, n in sorted(skipped.items()))
          + (f"; flags {flags}" if flags else "; the run's own flags"))

    maps = [m for r in reports for m in r["maps"]]
    if args.check:
        differ = [a for a, lines in replayed.items()
                  if not same_lines(lines, audios[a].get("lines") or [])]
        print(f"check: {len(replayed) - len(differ)} of {len(replayed)} audios reproduce the"
              f" run's lines ({len(differ)} differ)")
        for audio in differ[:10]:
            print(f"  {audio[:12]}  run {audios[audio].get('lines')[:3]}  replay {replayed[audio][:3]}")
        return 0 if len(differ) <= 0.01 * max(1, len(replayed)) else 1
    if run_experiments - {""}:
        print(f"note: the reports' runs had experiments {sorted(run_experiments - {''})}")

    dev = [m for m in maps if m.get("split") == "dev"]
    shift = rankable.beat_shift(dev)
    after_maps = []
    for m in maps:
        m["rank"] = rankable.verdict(m, shift)
        lines = replayed.get(m.get("audio_sha1"))
        changed = dict(m, lines=[[o, b] for o, b, *_ in lines]) if lines is not None else dict(m)
        changed["rank"] = rankable.verdict(changed, shift)
        after_maps.append(changed)
    print(f"lead fitted on dev: {shift:+.1f} ms (the run's lines)")
    for name in ("dev", "holdout", "confirmation"):
        before = [m for m in maps if m.get("split") == name]
        after = [m for m in after_maps if m.get("split") == name]
        if not before:
            continue
        b, a = rankable.summarise(before), rankable.summarise(after)
        fixed = sum(x["rank"]["verdict"] == "miss" and y["rank"]["verdict"] != "miss"
                    for x, y in zip(before, after))
        broken = sum(x["rank"]["verdict"] != "miss" and y["rank"]["verdict"] == "miss"
                     for x, y in zip(before, after))
        label = "held out" if name == "holdout" else name
        print(f"  {label:<9} rankable songs {rankable.pct(b['songs_rankable'], b['songs'])} ->"
              f" {rankable.pct(a['songs_rankable'], a['songs'])} %, maps"
              f" {rankable.pct(b['rankable'], b['maps'])} -> {rankable.pct(a['rankable'], a['maps'])} %,"
              f" exact {rankable.pct(b['exact'], b['maps'])} -> {rankable.pct(a['exact'], a['maps'])} %"
              f" (fixed {fixed}, broken {broken})")
        if args.worst and broken:
            worse = [(x, y) for x, y in zip(before, after)
                     if x["rank"]["verdict"] != "miss" and y["rank"]["verdict"] == "miss"]
            for x, y in worse[:args.worst]:
                print(f"    broken: {Path(x['folder']).name[:60]}  {x['lines'][:2]} -> {y['lines'][:2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
