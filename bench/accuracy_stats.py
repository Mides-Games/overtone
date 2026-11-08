"""Cluster intervals that keep linked recordings from one set together."""
from collections import defaultdict

import numpy as np


def clustered_mean_interval(observations: list[tuple[float, str]], seed: int) -> list | None:
    """Bootstrap independent groups while preserving the per-audio mean.

    Each observation is one audio's score. A group may contain several audios;
    resampling the whole group preserves their dependence and equal audio weight.
    """
    groups = defaultdict(list)
    for value, group in observations:
        groups[group].append(value)
    if len(groups) < 2:
        return None
    totals = np.asarray([sum(groups[k]) for k in sorted(groups)], dtype=float)
    counts = np.asarray([len(groups[k]) for k in sorted(groups)], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(10):
        chosen = rng.integers(len(groups), size=(1000, len(groups)))
        draws.extend(totals[chosen].sum(axis=1) / counts[chosen].sum(axis=1))
    return np.quantile(draws, [.025, .975]).tolist()
