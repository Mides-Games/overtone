"""Audit timing agreement within a locked local accuracy cohort.

Compares difficulties sharing identical audio in both directions. This detects
conflicting annotations, not whether either annotation matches the sound.
Private pair-level output belongs under bench/.cache; stdout is aggregate only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench")]
import accuracy_manifest as manifest_tools
import corpus_b as cb
import overtone as ta
from phase_residual import signed_nearest


def compare_grids(left: list, right: list, duration_s: float) -> dict:
    """Bidirectional grid agreement; half/double-time is not treated as equal."""
    a = np.asarray(cb.grid_beats(left, duration_s)) * 1000
    b = np.asarray(cb.grid_beats(right, duration_s)) * 1000
    if not a.size or not b.size:
        return {"status": "unscorable"}
    ab, ba = signed_nearest(a, b), signed_nearest(b, a)
    p95 = max(float(np.quantile(np.abs(ab), .95)), float(np.quantile(np.abs(ba), .95)))
    offset = float(np.median(ab))
    return {"status": "agree" if p95 <= 5 else "disagree",
            "p95_grid_error_ms": p95, "signed_offset_ms": offset,
            "p95_after_constant_offset_ms": max(
                float(np.quantile(np.abs(ab - offset), .95)),
                float(np.quantile(np.abs(ba + offset), .95)))}


def audit(manifest: dict) -> dict:
    groups, folders = defaultdict(list), defaultdict(set)
    for entry in manifest["entries"]:
        groups[entry["audio_sha1"]].append(entry)
        folders[entry["folder"]].add(entry["split"])
    pairs = []
    for audio, entries in sorted(groups.items()):
        grids = {e["osu_path"]: ta.read_osu_red_lines(e["osu_path"]) for e in entries}
        for left, right in itertools.combinations(entries, 2):
            durations = [e.get("duration_s") for e in (left, right)]
            result = (compare_grids(grids[left["osu_path"]], grids[right["osu_path"]],
                                    min(durations))
                      if all(d is not None and np.isfinite(d) and d > 0 for d in durations)
                      else {"status": "unscorable"})
            pairs.append({"audio_sha1": audio, "left": left["osu_path"],
                          "right": right["osu_path"], **result})
    crossings = sorted(folder for folder, splits in folders.items() if len(splits) > 1)
    return {"maps": len(manifest["entries"]), "audios": len(groups),
            "pairs": len(pairs), "agree": sum(p["status"] == "agree" for p in pairs),
            "disagree": sum(p["status"] == "disagree" for p in pairs),
            "unscorable": sum(p["status"] == "unscorable" for p in pairs),
            "sets_crossing_splits": len(crossings), "crossing_folders": crossings,
            "pair_results": pairs,
            "limitation": "Agreement cannot certify audio alignment or an untouched confirmation split."}


SWEEP_MS = 80.0       # offsets tried around each grid, both ways
MARGIN_MS = 5.0       # one side must sit this much nearer the cohort's lead to win
REFERENCE_AUDIOS = 160  # agreeing audios read: half set the lead and spread, half check it
CONFIDENT = 0.9         # posterior a side needs to win
VALIDATION_SHIFTS_MS = (0, 2, 3, 5, 7, 10, 15)
MIN_SALIENCE = 1.15   # peak-to-median onset ratio below this: the sound cannot decide


ATTACK_FRAME_MS = 0.5  # the attack curve's resolution
ATTACK_WIDTH_MS = 1.5  # each attack's spread on it


def attack_curve(times_s: np.ndarray, weights: np.ndarray, duration_s: float,
                 frame_ms: float = ATTACK_FRAME_MS) -> np.ndarray:
    """A narrow peak at every attack start, scaled by its weight."""
    curve = np.zeros(int(duration_s * 1000 / frame_ms) + 1)
    if not times_s.size:
        return curve
    reach = int(np.ceil(3 * ATTACK_WIDTH_MS / frame_ms))
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) * frame_ms / ATTACK_WIDTH_MS) ** 2)
    for t, w in zip(times_s, weights):
        centre = int(round(t * 1000 / frame_ms))
        lo, hi = max(0, centre - reach), min(curve.size, centre + reach + 1)
        if lo < hi:
            curve[lo:hi] += w * kernel[lo - centre + reach:hi - centre + reach]
    return curve


def onset_curve(audio_path: str) -> tuple[np.ndarray, float]:
    """The audio's attack starts as a curve, and its frame period in ms.

    Attacks are the engine's own, re-timed on the waveform to where the energy
    rise begins (precision_grid._retime_onsets). The onset envelope's peak was
    tried first and sits about 10 ms after the beat a listener hears, which made
    the verdicts favour the late side (owner's listening check, 2026-11-07).
    """
    from overtone.audio_io import _load_audio
    from overtone.precision_grid import _detect_attacks
    y, sr = _load_audio(audio_path, lambda _msg: None)
    times, weights, _env = _detect_attacks(y, sr)
    return attack_curve(np.asarray(times, float), np.asarray(weights, float),
                        y.size / sr), ATTACK_FRAME_MS

def grid_alignment(env: np.ndarray, frame_ms: float, beats_ms: np.ndarray) -> dict:
    """Where a grid's beats sit best on the onsets, within +/- SWEEP_MS.

    The score at each offset is the mean onset strength under the shifted beats;
    salience is its peak over its median across the sweep, so a grid that fits
    the sound everywhere it is moved (sustained pads, dense noise) is not judged.
    """
    beats_ms = beats_ms[(beats_ms > SWEEP_MS) & (beats_ms < len(env) * frame_ms - SWEEP_MS)]
    if beats_ms.size < 8 or not env.size:
        return {"status": "unscorable"}
    offsets = np.arange(-SWEEP_MS, SWEEP_MS + 0.5, 0.5)
    frames = np.arange(env.size) * frame_ms
    scores = np.array([np.interp(beats_ms + d, frames, env).mean() for d in offsets])
    median = float(np.median(scores))
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return {"status": "unscorable"}
    salience = float(scores[best] / median) if median > 0 else float("inf")
    return {"status": "scored", "best_offset_ms": float(offsets[best]),
            "salience": salience, "decisive": salience >= MIN_SALIENCE}


def offset_density(deviations: np.ndarray):
    """The natural spread of (grid fit - cohort lead) on maps known to agree.

    A Gaussian kernel estimate (Silverman's bandwidth, at least 1 ms), mixed
    with 1 % of a uniform over the sweep so no offset is ever impossible and a
    posterior cannot reach 0 or 1 on one outlying fit.
    """
    dev = np.asarray(deviations, float)
    spread = min(dev.std(), (np.quantile(dev, .75) - np.quantile(dev, .25)) / 1.34) or dev.std()
    width = max(1.0, 0.9 * spread * dev.size ** -0.2)
    norm = 1 / (dev.size * width * np.sqrt(2 * np.pi))

    def pdf(x: float) -> float:
        kde = norm * np.exp(-0.5 * ((x - dev) / width) ** 2).sum()
        return 0.99 * kde + 0.01 / (4 * SWEEP_MS)
    return pdf


def pairing_posterior(left_ms: float, right_ms: float, lead_ms: float, pdf) -> float:
    """P(the left grid is the right one), equal priors, from the agreeing spread."""
    a, b = pdf(left_ms - lead_ms), pdf(right_ms - lead_ms)
    return float(a / (a + b))


def pairing_verdict(left: dict, right: dict, lead_ms: float, pdf=None) -> tuple[str, float | None]:
    """Which annotation of a disagreeing pair the sound supports, and how sure.

    Ranked maps do not sit at 0 ms on the decoded audio: decoder delay and the
    mappers' convention put them a cohort-wide lead away. A side wins only when
    its posterior reaches CONFIDENT under the spread of maps that agree.
    """
    if not (left.get("decisive") and right.get("decisive")):
        return "ambiguous", None
    if pdf is None:   # no reference spread: the nearer side by MARGIN_MS
        gap = [abs(side["best_offset_ms"] - lead_ms) for side in (left, right)]
        if abs(gap[0] - gap[1]) < MARGIN_MS:
            return "tie", None
        return ("left" if gap[0] < gap[1] else "right"), None
    p = pairing_posterior(left["best_offset_ms"], right["best_offset_ms"], lead_ms, pdf)
    if p >= CONFIDENT:
        return "left", p
    if p <= 1 - CONFIDENT:
        return "right", p
    return "tie", p


class _Reader:
    """Onset curves per audio and beat grids per map, each read once."""

    def __init__(self, curve):
        self.curve, self.curves, self.grids = curve, {}, {}

    def alignment(self, entry: dict, duration_s: float) -> dict:
        audio = entry["audio_sha1"]
        if audio not in self.curves:
            try:
                self.curves[audio] = self.curve(entry["audio_path"])
            except (OSError, RuntimeError, ValueError) as error:
                self.curves[audio] = error
        if isinstance(self.curves[audio], Exception):
            return {"status": "unscorable", "reason": str(self.curves[audio])}
        path = entry["osu_path"]
        if path not in self.grids:
            self.grids[path] = np.asarray(
                cb.grid_beats(ta.read_osu_red_lines(path), duration_s)) * 1000
        return grid_alignment(*self.curves[audio], self.grids[path])


def reference_fits(manifest: dict, pairs: list[dict], reader: _Reader,
                   limit: int = REFERENCE_AUDIOS) -> dict:
    """Grid fits on audios whose difficulties all agree, split in two by hash.

    The 'reference' half sets the lead and the natural spread; the 'check' half
    is never used for either, so validation runs on audios the model has not
    seen. The disputed audios set neither. One map per audio.
    """
    disputed = {p["audio_sha1"] for p in pairs if p["status"] != "agree"}
    first = {}
    for entry in sorted(manifest["entries"], key=lambda e: (e["audio_sha1"], e["osu_path"])):
        if entry["audio_sha1"] not in disputed and entry.get("duration_s"):
            first.setdefault(entry["audio_sha1"], entry)
    halves = {"reference": [], "check": []}
    for entry in list(first.values())[:limit]:
        side = reader.alignment(entry, entry["duration_s"])
        if side.get("decisive"):
            half = "reference" if int(entry["audio_sha1"][-1], 16) % 2 == 0 else "check"
            halves[half].append(side["best_offset_ms"])
    return halves


def validate_shifts(reference: list[float], check: list[float],
                    shifts=VALIDATION_SHIFTS_MS) -> dict:
    """How often the posterior names a grid moved by a known amount.

    Each held-back fit is paired with itself moved by k ms both ways; moving a
    grid by k moves its best fit by exactly -k, so no audio is read again.
    'right' names the original, 'wrong' the moved copy, 'tie' neither.
    """
    if len(reference) < 8 or not check:
        return {"status": "too few agreeing audios"}
    lead = float(np.median(reference))
    pdf = offset_density(np.asarray(reference) - lead)
    rows = {}
    for k in shifts:
        tally = {"right": 0, "wrong": 0, "tie": 0}
        for fit in check:
            for moved in (fit - k, fit + k):
                p = pairing_posterior(fit, moved, lead, pdf)
                tally["right" if p >= CONFIDENT else "wrong" if p <= 1 - CONFIDENT else "tie"] += 1
        total = sum(tally.values())
        rows[f"{k:g} ms"] = {name: round(100 * n / total, 1) for name, n in tally.items()}
    return {"reference_audios": len(reference), "check_audios": len(check),
            "lead_ms": lead, "confident_at": CONFIDENT, "percent": rows}


def pairing_audit(manifest: dict, pairs: list[dict], curve=onset_curve) -> dict:
    """Score each disagreeing pair's two grids against the shared audio.

    Diagnostic only: it names which side the onsets favour, with a posterior
    calibrated on agreeing audios and checked on held-back ones by
    validate_shifts. It never removes a map from the score.
    """
    entries = {e["osu_path"]: e for e in manifest["entries"]}
    reader = _Reader(curve)
    halves = reference_fits(manifest, pairs, reader)
    reference = halves["reference"]
    lead = {"audios": len(reference)}
    pdf = None
    if reference:
        q1, median, q3 = np.quantile(reference, [.25, .5, .75])
        lead.update(lead_ms=float(median), iqr_ms=[float(q1), float(q3)])
        if len(reference) >= 8:
            pdf = offset_density(np.asarray(reference) - median)
    results = []
    for pair in (p for p in pairs if p["status"] == "disagree"):
        left, right = entries[pair["left"]], entries[pair["right"]]
        duration = min(left["duration_s"], right["duration_s"])
        sides = [reader.alignment(e, duration) for e in (left, right)]
        if "lead_ms" not in lead or "unscorable" in (sides[0]["status"], sides[1]["status"]):
            verdict, p = "unscorable", None
        else:
            verdict, p = pairing_verdict(*sides, lead["lead_ms"], pdf)
        results.append({**pair, "left_alignment": sides[0], "right_alignment": sides[1],
                        "verdict": verdict, "p_left": p})
    counts = defaultdict(int)
    for row in results:
        counts[row["verdict"]] += 1
    return {"disagreeing_pairs": len(results),
            "audios": len({r["audio_sha1"] for r in results}),
            "cohort_lead": lead,
            "validation": validate_shifts(reference, halves["check"]),
            "verdicts": dict(sorted(counts.items())),
            "pairing_results": results,
            "limitation": "A posterior on onsets, validated on injected shifts of agreeing "
                          "maps; it favours a grid, it does not certify the audio version."}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", type=Path, help="private pair-level output")
    parser.add_argument("--pairing", action="store_true",
                        help="also score each disagreeing pair's grids against the audio")
    args = parser.parse_args(argv)
    if manifest_tools.verify(args.manifest):
        return 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = audit(manifest)
    if args.pairing:
        result["pairing"] = pairing_audit(manifest, result["pair_results"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("pair_results", "crossing_folders")}
    if "pairing" in summary:
        summary["pairing"] = {k: v for k, v in summary["pairing"].items()
                              if k != "pairing_results"}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
