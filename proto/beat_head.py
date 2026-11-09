"""Prototype: a small head on a frozen Beat This! that places each beat
between the network's 20 ms frames, trained on the ranked maps.

Beat This! marks 50 frames a second, so its beats sit on a 20 ms grid and a
stretch of them cannot place a red line to 10 ms. A parabola through each
peak (proto/beat_subframe.py) moved beats within 10 ms of the map 70 -> 81 %,
yet no song became rankable. This learns the correction instead: for each
beat the network finds, the audio's band envelope at ~1.5 ms around it and
the network's logits around its peak, and as the target the distance to the
ranked map's nearest beat. The network itself is not trained, so the whole
thing trains on a CPU in minutes.

Two steps, two environments:

    # the project's .venv (onnxruntime, librosa, soxr): one .npz per song
    python proto/beat_head.py build <report.json>... --out DIR [--per-split N] [--jobs 2]
    # any Python with torch: train on dev, score held out against the frames and the parabola
    python proto/beat_head.py train DIR [--epochs 30] [--save head.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

FINE_SR = 22050
FINE_HOP = 32                      # 1.45 ms
FINE_FFT = 256                     # 11.6 ms window
FINE_BANDS = 8
HALF = 41                          # frames each side: about 60 ms
LOGIT_HALF = 3                     # Beat This! frames each side of the peak
TARGET_MS = 40.0                   # a map beat this close is the beat's target
TRAIN_BEATS = 400                  # beats kept per training song


# ---------------------------------------------------------------- build ----

def map_beats(rows: list[dict], end_ms: float) -> np.ndarray:
    """Every beat of the map's red lines, each line up to the next."""
    lines = sorted((float(r["offset_ms"]), float(r["bpm"])) for r in rows)
    out = []
    for n, (o, bpm) in enumerate(lines):
        stop = lines[n + 1][0] if n + 1 < len(lines) else end_ms
        step = 60000.0 / bpm
        out.append(o + step * np.arange(0, max(0, int(np.ceil((stop - o) / step - 1e-6)))))
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


def fine_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """(bands + 1, frames): each band's positive log-mel flux and the broadband
    log energy, at FINE_HOP; frame j is centred on j * FINE_HOP / FINE_SR."""
    import librosa
    import soxr
    sig = soxr.resample(np.asarray(y, dtype=np.float64), sr, FINE_SR) if sr != FINE_SR else np.asarray(y, dtype=np.float64)
    mel = librosa.feature.melspectrogram(y=sig.astype(np.float32), sr=FINE_SR, n_fft=FINE_FFT,
                                         hop_length=FINE_HOP, n_mels=FINE_BANDS, fmin=30.0, power=2.0)
    log = np.log1p(1000.0 * mel)
    flux = np.maximum(0.0, np.diff(log, axis=1, prepend=log[:, :1]))
    energy = np.log1p(1000.0 * mel.sum(axis=0, keepdims=True))
    out = np.concatenate([flux, energy], axis=0).astype(np.float32)
    # one scale per song: the head reads shapes, not loudness
    scale = np.percentile(out, 99, axis=1, keepdims=True) + 1e-6
    return out / scale


def peaks(logits: np.ndarray) -> np.ndarray:
    """Beat frames as the engine finds them (beat_this._beat_this_beats)."""
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
    return np.asarray(out, dtype=float)


def parabola(logits: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Each peak's offset in frames from a parabola through it and its neighbours."""
    out = np.zeros(frames.size)
    for i, m in enumerate(frames):
        f = int(round(m))
        if 0 < f < logits.size - 1 and m == f:
            a, b, c = float(logits[f - 1]), float(logits[f]), float(logits[f + 1])
            den = a - 2 * b + c
            out[i] = 0.5 * (a - c) / den if den < 0 else 0.0
    return out


def build_song(m: dict) -> dict | None:
    sys.path[:0] = [str(ROOT / "python")]
    import overtone as ta
    import soxr
    from overtone import beat_this, params
    y, sr = ta._load_audio(str(Path(m["folder_path"]) / m["audio_filename"]), lambda _m: None)
    session = beat_this._load_beat_this()
    sig = soxr.resample(np.asarray(y, dtype=np.float64), sr, params.BEAT_THIS_SR)
    logits = beat_this._beat_this_logits(session, beat_this._beat_this_spectrogram(sig)).astype(np.float32)
    frames = peaks(logits)
    if frames.size < 16:
        return None
    frame_ms = params.BEAT_THIS_HOP / params.BEAT_THIS_SR * 1000.0
    t = frames * frame_ms
    env = fine_envelope(y, sr)
    fine_ms = FINE_HOP / FINE_SR * 1000.0
    centre = np.round(t / fine_ms).astype(int)
    pad_env = np.pad(env, ((0, 0), (HALF, HALF)))
    windows = np.stack([pad_env[:, c:c + 2 * HALF + 1] for c in centre]).astype(np.float16)
    pad_log = np.pad(logits, LOGIT_HALF, constant_values=-20.0)
    f0 = np.round(frames).astype(int)
    around = np.stack([pad_log[f:f + 2 * LOGIT_HALF + 1] for f in f0]).astype(np.float32)
    around = np.clip(around, -20.0, 20.0)
    beats = map_beats(m["rows"], (m.get("duration_s") or 0.0) * 1000.0)
    if beats.size == 0:
        return None
    idx = np.clip(np.searchsorted(beats, t), 1, beats.size - 1)
    near = np.where(np.abs(beats[idx - 1] - t) <= np.abs(beats[idx] - t), beats[idx - 1], beats[idx])
    target = (near - t).astype(np.float32)
    return {"windows": windows, "logits": around, "parabola": (parabola(logits, frames) * frame_ms).astype(np.float32),
            "t": t.astype(np.float64), "target": target, "split": m.get("split", ""),
            "song": m["audio_sha1"], "lines": len(m["rows"])}


def _build_one(job):
    m, out = job
    path = Path(out) / f"{m['audio_sha1']}.npz"
    if path.exists():
        return path.name, "cached"
    try:
        song = build_song(m)
    except Exception as exc:                    # one bad file must not stop the set
        return path.name, f"error {exc}"
    if song is None:
        return path.name, "skipped"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in song.items()})
    return path.name, f"{song['t'].size} beats"


def build(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked, seen = [], set()
    counts: dict[str, int] = {}
    for p in args.reports:
        for m in json.loads(Path(p).read_text(encoding="utf-8"))["maps"]:
            split = m.get("split", "")
            if m["audio_sha1"] in seen or not m.get("rows") or counts.get((p, split), 0) >= args.per_split:
                continue
            seen.add(m["audio_sha1"])
            counts[(p, split)] = counts.get((p, split), 0) + 1
            picked.append(m)
    print(f"{len(picked)} songs", flush=True)
    jobs = [(m, str(out)) for m in picked]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as pool:
            for n, (name, what) in enumerate(pool.map(_build_one, jobs), 1):
                print(f"{n:5d} {name[:12]} {what}", flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            name, what = _build_one(job)
            print(f"{n:5d} {name[:12]} {what}", flush=True)
    return 0


# ---------------------------------------------------------------- train ----

def load(directory: Path, rng):
    songs = []
    for path in sorted(directory.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        songs.append({k: z[k] for k in z.files})
    train, held = [], []
    for s in songs:
        (train if str(s["split"]) == "dev" else held).append(s)
    return train, held


def stack(songs, rng, limit=None, only_near=True):
    xs, ls, ps, ys, ids = [], [], [], [], []
    for n, s in enumerate(songs):
        keep = np.abs(s["target"]) <= TARGET_MS if only_near else np.ones(s["target"].size, bool)
        idx = np.flatnonzero(keep)
        if limit is not None and idx.size > limit:
            idx = rng.choice(idx, limit, replace=False)
        xs.append(s["windows"][idx].astype(np.float32))
        ls.append(s["logits"][idx])
        ps.append(s["parabola"][idx])
        ys.append(s["target"][idx])
        ids.append(np.full(idx.size, n))
    return (np.concatenate(xs), np.concatenate(ls), np.concatenate(ps),
            np.concatenate(ys), np.concatenate(ids))


def make_head(torch):
    nn = torch.nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            c = FINE_BANDS + 1
            self.conv = nn.Sequential(
                nn.Conv1d(c, 32, 5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ReLU(),
                nn.Conv1d(32, 16, 5, padding=8, dilation=4), nn.ReLU())
            self.fc = nn.Sequential(nn.Linear(16 * (2 * HALF + 1) + 2 * LOGIT_HALF + 2, 64), nn.ReLU(),
                                    nn.Linear(64, 1))

        def forward(self, x, logits, para):
            h = self.conv(x).flatten(1)
            return self.fc(torch.cat([h, logits / 10.0, para[:, None] / 10.0], dim=1))[:, 0]

    return Head()


def score(name, err_by_song):
    """Per beat and per 16-beat stretch, after the set's own median (the
    mappers' lead is a constant the metric fits on its own)."""
    allerr = np.concatenate(err_by_song)
    lead = float(np.median(allerr))
    a = np.abs(allerr - lead)
    stretch = np.abs(np.asarray([np.mean(e[i:i + 16] - lead) for e in err_by_song
                                 for i in range(0, e.size - 16, 16)]))
    # a one-line song's offset error is its beats' median: the mappers' own
    # lead varies 15-47 ms by song, which the one fitted lead cannot follow
    song = np.abs(np.asarray([np.median(e) - lead for e in err_by_song]))
    print(f"  {name:<10} {allerr.size:6d} beats  lead {lead:+5.1f}  median |err| {np.median(a):4.1f} ms  "
          f"<=5 ms {np.mean(a <= 5) * 100:5.1f} %  <=10 ms {np.mean(a <= 10) * 100:5.1f} %  "
          f"p90 {np.percentile(a, 90):5.1f}  16-beat mean <=5 ms {np.mean(stretch <= 5) * 100:5.1f} %  "
          f"song median <=3 / 5 ms {np.mean(song <= 3) * 100:5.1f} / {np.mean(song <= 5) * 100:5.1f} %")


def train(args) -> int:
    import torch
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    train_songs, held_songs = load(Path(args.dir), rng)
    print(f"{len(train_songs)} dev songs, {len(held_songs)} held out", flush=True)
    order = rng.permutation(len(train_songs))
    val_n = max(1, len(train_songs) // 8)
    val = [train_songs[i] for i in order[:val_n]]
    fit = [train_songs[i] for i in order[val_n:]]
    X, L, P, Y, _ = stack(fit, rng, TRAIN_BEATS)
    Xv, Lv, Pv, Yv, _ = stack(val, rng)
    bias = float(np.median(Y))
    print(f"train {Y.size} beats, validate {Yv.size}; target median {bias:+.1f} ms", flush=True)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    X, L, P, Y = t(X), t(L), t(P), t(Y - bias)
    Xv, Lv, Pv, Yv = t(Xv), t(Lv), t(Pv), t(Yv - bias)
    head = make_head(torch)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(delta=5.0)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(Y.shape[0])
        total = 0.0
        for i in range(0, Y.shape[0], args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(head(X[b], L[b], P[b]), Y[b])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * b.numel()
        head.eval()
        with torch.no_grad():
            v = float(torch.cat([loss_fn(head(Xv[i:i + 4096], Lv[i:i + 4096], Pv[i:i + 4096]), Yv[i:i + 4096])[None]
                                 for i in range(0, Yv.shape[0], 4096)]).mean())
        print(f"epoch {epoch + 1:3d}  train {total / Y.shape[0]:.3f}  validate {v:.3f}", flush=True)
        if v < best - 1e-4:
            best, best_state, stale = v, {k: x.clone() for k, x in head.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    for name, songs in (("dev (val)", val), ("held out", held_songs)):
        print(f"{name}: {len(songs)} songs", flush=True)
        rows = {"frames": [], "parabola": [], "head": []}
        for s in songs:
            keep = np.abs(s["target"]) <= TARGET_MS
            if keep.sum() < 16:
                continue
            x = t(s["windows"][keep].astype(np.float32))
            with torch.no_grad():
                pred = head(x, t(s["logits"][keep]), t(s["parabola"][keep])).numpy() + bias
            y = s["target"][keep]
            rows["frames"].append(-y)
            rows["parabola"].append(s["parabola"][keep] - y)
            rows["head"].append(pred - y)
        for k, v in rows.items():
            score(k, v)
    if args.save:
        torch.save({"state": best_state, "bias": bias}, args.save)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("reports", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--per-split", type=int, default=10 ** 9, help="songs per report and split")
    b.add_argument("--jobs", type=int, default=1)
    t = sub.add_parser("train")
    t.add_argument("dir")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--save", default=None)
    args = ap.parse_args(argv)
    return build(args) if args.cmd == "build" else train(args)


if __name__ == "__main__":
    sys.exit(main())
