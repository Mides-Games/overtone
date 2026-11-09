"""Prototype: Beat This!'s beats placed between its 20 ms frames.

The network marks 50 frames a second; its beats sit on that grid, so a stretch
of them cannot place a red line to 10 ms. A parabola through each peak's logit
and its neighbours puts the beat between frames. This measures both against
the ranked maps of single-line songs whose engine grid is rankable: each beat's
distance to the map's nearest beat, less the song's median.

    python proto/beat_subframe.py <report.json>... [--songs 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench")]


def peaks(logits: np.ndarray, subframe: bool) -> np.ndarray:
    """Beat frames (float) as _beat_this_beats finds them, optionally refined
    by a parabola through each peak and its two neighbours."""
    padded = np.pad(logits, 3, constant_values=-1000.0)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 7).max(axis=1)
    frames = np.flatnonzero((logits == local_max) & (logits > 0))
    out, mean, count = [], None, 0
    for f in frames:
        if mean is not None and f - mean <= 1:
            count += 1
            mean += (f - mean) / count
        else:
            if mean is not None:
                out.append(mean)
            mean, count = float(f), 1
    if mean is not None:
        out.append(mean)
    out = np.asarray(out, dtype=float)
    if not subframe:
        return out
    refined = []
    for m in out:
        f = int(round(m))
        if 0 < f < logits.size - 1 and m == f:
            a, b, c = float(logits[f - 1]), float(logits[f]), float(logits[f + 1])
            den = a - 2 * b + c
            refined.append(f + (0.5 * (a - c) / den if den < 0 else 0.0))
        else:
            refined.append(m)
    return np.asarray(refined)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--songs", type=int, default=40)
    args = ap.parse_args(argv)
    import overtone as ta
    import rankable
    from overtone import beat_this, params
    maps = [m for p in args.reports for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]]
    shift = rankable.beat_shift([m for m in maps if m.get("split") == "dev"])
    picked, seen = [], set()
    for m in maps:
        if m["audio_sha1"] in seen or len(rankable.map_lines_of(m)) != 1:
            continue
        if rankable.verdict(m, shift)["verdict"] == "miss":
            continue
        seen.add(m["audio_sha1"])
        picked.append(m)
    picked = picked[:args.songs]
    session = beat_this._load_beat_this()
    import soxr
    rows = {False: [], True: []}
    for m in picked:
        y, sr = ta._load_audio(str(Path(m["folder_path"]) / m["audio_filename"]), lambda _m: None)
        sig = soxr.resample(np.asarray(y, dtype=np.float64), sr, params.BEAT_THIS_SR)
        logits = beat_this._beat_this_logits(session, beat_this._beat_this_spectrogram(sig)).astype(float)
        (o, bpm), = rankable.map_lines_of(m)
        step = 60000.0 / bpm
        for sub in (False, True):
            t = peaks(logits, sub) * params.BEAT_THIS_HOP / params.BEAT_THIS_SR * 1000.0
            r = t - o - np.round((t - o) / step) * step
            r = r[np.abs(r) < 0.25 * step]
            if r.size < 16:
                continue
            r = r - np.median(r)
            rows[sub].append(r)
        print(Path(m["folder_path"]).name[:40], flush=True)
    for sub in (False, True):
        r = np.concatenate(rows[sub])
        a = np.abs(r)
        print(f"{'sub-frame' if sub else 'on frames'}: {r.size} beats, median |err| {np.median(a):.1f} ms, "
              f"p90 {np.percentile(a, 90):.1f} ms, within 5 ms {np.mean(a <= 5) * 100:.0f} %, within 10 ms {np.mean(a <= 10) * 100:.0f} %")
        # a stretch of 16 beats: how far its mean sits from the map
        means = [np.mean(x[i:i + 16]) for x in rows[sub] for i in range(0, x.size - 16, 16)]
        means = np.abs(np.asarray(means))
        print(f"   16-beat stretch mean: median {np.median(means):.1f} ms, p90 {np.percentile(means, 90):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
