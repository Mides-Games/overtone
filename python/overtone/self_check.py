"""Whether the engine is sure of its own timing, without a map (roadmap 29.5).

A timing is judged against ranked maps with a worst-beat verdict: rankable
means every beat within 10 ms of the ranked map. ``self_check`` asks the same
question of the analysis alone. Per red line it reads the strong attacks (the
top share by weight) against that line's grid:

* how far they sit from the nearest beat: the 95th percentile of the distance
  among those that sit near a beat, and the share that sits near neither a
  beat nor a half beat;
* whether they carry more weight on the half beats than on the beats (the grid
  half a beat out, or an octave in doubt), beside the octave margin the seed
  search stood on (``analysis_evidence``);
* the slow drift: the signed distance's linear trend over the line, which a
  BPM a few thousandths out leaves behind.

The verdict is ``"sure"`` when every red line passes, ``"look"`` otherwise,
naming the first line that did not and why. Read only: no red line moves.
The thresholds (``params.SELF_CHECK_*``) are uncalibrated; ``bench/replay.py``
against ranked maps is where they get their numbers (wanted: at least 95 % of
the songs called "sure" rankable).
"""
from __future__ import annotations

import math

import numpy as np

from . import params, exports


def _section_margin(evidence: list[dict], start_s: float, end_s: float) -> float | None:
    """The octave margin of the fitted section under the middle of a red line."""
    middle = 0.5 * (start_s + end_s)
    for section in evidence:
        if section["start_s"] - 1e-3 <= middle <= section["end_s"] + 1e-3:
            return section.get("octave_margin")
    return None


def _round(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _line_check(times: np.ndarray, weights: np.ndarray, start_s: float, end_s: float,
                bpm: float, margin: float | None) -> dict:
    """One red line's numbers and the reasons it is not sure (empty: it is)."""
    period = 60.0 / bpm
    inside = (times >= start_s) & (times < end_s) & (weights > 0)
    t, w = times[inside], weights[inside]
    out = {"start_s": round(start_s, 3), "end_s": round(end_s, 3), "bpm": round(bpm, 4),
           "strong": 0, "p95_ms": None, "bias_ms": None, "far_share": None,
           "offbeat_ratio": None, "octave_margin": _round(margin, 4),
           "drift_ms_per_min": None, "drift_span_ms": None, "reasons": []}
    reasons = out["reasons"]
    if t.size:
        floor = np.quantile(w, 1.0 - params.SELF_CHECK_STRONG_SHARE)
        strong = w >= floor
        t, w = t[strong], w[strong]
    out["strong"] = int(t.size)
    if t.size < params.SELF_CHECK_MIN_ATTACKS:
        reasons.append("few_attacks")
        return out

    position = (t - start_s) / period
    off = position - np.round(position)                 # beats, in [-0.5, 0.5]
    near = np.abs(off) <= params.SELF_CHECK_NEAR_BEAT
    half = np.abs(np.abs(off) - 0.5) <= params.SELF_CHECK_NEAR_BEAT
    on_w, half_w, total_w = float(w[near].sum()), float(w[half].sum()), float(w.sum())
    far_share = 1.0 - (on_w + half_w) / total_w
    out["far_share"] = round(far_share, 4)
    out["offbeat_ratio"] = round(half_w / on_w, 4) if on_w > 0 else None
    if on_w <= 0 or half_w > params.SELF_CHECK_OFFBEAT_RATIO * on_w:
        reasons.append("offbeat")
    if margin is not None and margin < params.SELF_CHECK_OCTAVE_MARGIN:
        reasons.append("octave")
    if far_share > params.SELF_CHECK_FAR_SHARE:
        reasons.append("off_grid")

    distance_ms = off[near] * period * 1000.0
    if distance_ms.size < params.SELF_CHECK_MIN_ATTACKS:
        if "offbeat" not in reasons:
            reasons.append("few_attacks")
        return out
    p95 = float(np.quantile(np.abs(distance_ms), 0.95))
    out["p95_ms"] = round(p95, 2)
    out["bias_ms"] = round(float(np.median(distance_ms)), 2)
    if p95 > params.SELF_CHECK_P95_MS:
        reasons.append("loose")

    at_min = t[near] / 60.0
    if np.ptp(at_min) > 0:
        slope = float(np.polyfit(at_min, distance_ms, 1)[0])     # ms per minute
        span = abs(slope) * (end_s - start_s) / 60.0
        out["drift_ms_per_min"] = round(slope, 2)
        out["drift_span_ms"] = round(span, 2)
        if span > params.SELF_CHECK_DRIFT_MS:
            reasons.append("drift")
    return out


def self_check(analysis) -> dict:
    """The engine's own verdict on an analysis: ``"sure"`` or ``"look"``.

    Returns ``verdict``, the first line to look at (``look``: its index,
    ``start_s``, ``end_s`` and ``reason``, None when sure), every red line's
    numbers under ``sections``, and ``calibrated`` (False until the thresholds
    are measured against ranked maps). A refusal (None), an analysis with no
    red lines and the fallback tracker's (no attacks to read) are "look".
    """
    result = {"verdict": "look", "calibrated": params.SELF_CHECK_CALIBRATED,
              "look": None, "sections": []}
    if analysis is None:
        result["look"] = {"index": None, "start_s": None, "end_s": None, "reason": "refused"}
        return result
    points = [p for p in (getattr(analysis, "points", None) or [])
              if math.isfinite(float(p.bpm)) and float(p.bpm) > 0
              and math.isfinite(float(p.offset_ms))]
    duration = float(getattr(analysis, "duration", 0.0) or 0.0)
    if not points:
        result["look"] = {"index": None, "start_s": 0.0, "end_s": round(duration, 3),
                          "reason": "no_red_lines"}
        return result

    times = np.asarray(getattr(analysis, "attack_times", []), dtype=np.float64)
    weights = np.asarray(getattr(analysis, "attack_weights", []), dtype=np.float64)
    if weights.shape != times.shape:
        weights = np.ones_like(times)
    good = np.isfinite(times) & np.isfinite(weights)
    times, weights = times[good], weights[good]
    try:
        evidence = exports.analysis_evidence(analysis).get("sections", []) if times.size else []
    except Exception:  # noqa: BLE001 -- the margin is one reading of several, never a blocker
        evidence = []

    for i, p in enumerate(points):
        start = float(p.offset_ms) / 1000.0
        end = float(points[i + 1].offset_ms) / 1000.0 if i + 1 < len(points) else duration
        end = max(end, start)
        line = _line_check(times, weights, start, end, float(p.bpm),
                           _section_margin(evidence, start, end))
        if str(getattr(analysis, "engine", "")) != "precision":
            line["reasons"].insert(0, "fallback_tracker")
        line["index"] = i
        result["sections"].append(line)
        if line["reasons"] and result["look"] is None:
            result["look"] = {"index": i, "start_s": line["start_s"], "end_s": line["end_s"],
                              "reason": line["reasons"][0]}
    if result["look"] is None:
        result["verdict"] = "sure"
    return result
