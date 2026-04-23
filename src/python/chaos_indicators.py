"""Chaos indicators: Lyapunov, permutation entropy, 0-1 chaos test."""
from __future__ import annotations

from typing import List

import numpy as np


def largest_lyapunov(x: np.ndarray, m: int = 5, tau: int = 1) -> float:
    """Rosenstein largest Lyapunov exponent via nolds."""
    import nolds
    if len(x) < 50 or np.std(x) == 0:
        return float("nan")
    try:
        return float(nolds.lyap_r(x, emb_dim=m, lag=tau, min_tsep=10, trajectory_len=20))
    except Exception:
        return float("nan")


def permutation_entropy(x: np.ndarray, order: int = 4, delay: int = 1) -> float:
    """Bandt-Pompe permutation entropy via antropy (normalized)."""
    import antropy
    if len(x) < 50:
        return float("nan")
    try:
        return float(antropy.perm_entropy(x, order=order, delay=delay, normalize=True))
    except Exception:
        return float("nan")


def chaos01_K(x: np.ndarray, n_c: int = 100, seed: int = 0) -> float:
    """Gottwald-Melbourne 0-1 chaos test, correlation variant.

    Reference: Gottwald & Melbourne (2004) Proc R Soc A.
    Returns K in [0, 1]; near 0 regular, near 1 chaotic.
    """
    if len(x) < 50 or np.std(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    N = len(x)
    phi = x - np.mean(x)
    n = np.arange(1, N + 1)
    Ks: List[float] = []
    for _ in range(n_c):
        c = rng.uniform(np.pi / 5.0, 4.0 * np.pi / 5.0)
        p_series = np.cumsum(phi * np.cos(c * n))
        q_series = np.cumsum(phi * np.sin(c * n))
        max_lag = min(100, max(N // 4, 10))
        M = np.array([
            np.mean((p_series[j:] - p_series[:-j]) ** 2 + (q_series[j:] - q_series[:-j]) ** 2)
            for j in range(1, max_lag)
        ])
        if len(M) < 5 or np.std(M) == 0:
            continue
        logM = np.log(M + 1e-12)
        logn = np.log(np.arange(1, len(M) + 1))
        r = np.corrcoef(logn, logM)[0, 1]
        if not np.isnan(r):
            Ks.append(r)
    return float(np.median(Ks)) if Ks else float("nan")
