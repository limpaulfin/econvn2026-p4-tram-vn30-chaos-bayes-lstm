"""
IAAFT surrogate-data null test for chaos indicators on VN30 returns.

Refs:
  Theiler et al. 1992, Physica D 58:77 (surrogate-data method).
  Schreiber & Schmitz 1996, PRL 77:635 (IAAFT refinement).
  Federico 2020, Physica A 558:124861 (0-1 test overrejection on leptokurtic series).

Purpose:
  Given an observed chaos-indicator value K on a ticker return series, generate
  N_iaaft linear-stochastic surrogates that match the amplitude distribution
  and power spectrum of the data. Compute K on each surrogate. The one-sided
  p-value is the fraction of surrogates whose K meets or exceeds the observed
  K. A surviving observed K (p <= alpha) supports a non-linear interpretation;
  a failing K (p > alpha) is consistent with the leptokurtic-noise null.

Interface:
  iaaft(x, n_iter=100, rng=None) -> np.ndarray
  surrogate_null_test(x, indicator_fn, n_surrogates=499, n_iter=100, rng=None)
      -> {'observed', 'null_samples', 'p_value', 'reject_null'}

Status:
  STUB with validated IAAFT core. Wire-up against chaos_indicators.py is
  executed in camera-ready via scripts/run_surrogate_null.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def iaaft(x: np.ndarray, n_iter: int = 100, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    n = x.size
    amplitudes = np.sort(x)
    target_spectrum = np.abs(np.fft.rfft(x))
    surrogate = rng.permutation(x)
    for _ in range(n_iter):
        spectrum = np.fft.rfft(surrogate)
        phases = np.angle(spectrum)
        adjusted = target_spectrum * np.exp(1j * phases)
        surrogate = np.fft.irfft(adjusted, n=n)
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = amplitudes[ranks]
    return surrogate


def surrogate_null_test(
    x: np.ndarray,
    indicator_fn: Callable[[np.ndarray], float],
    n_surrogates: int = 499,
    n_iter: int = 100,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    observed = float(indicator_fn(np.asarray(x, dtype=float)))
    null_samples = np.empty(n_surrogates, dtype=float)
    for i in range(n_surrogates):
        surrogate = iaaft(x, n_iter=n_iter, rng=rng)
        null_samples[i] = float(indicator_fn(surrogate))
    p_value = float(np.mean(null_samples >= observed))
    return {
        "observed": observed,
        "null_samples": null_samples,
        "p_value": p_value,
        "reject_null": p_value <= alpha,
        "null_mean": float(np.mean(null_samples)),
        "null_std": float(np.std(null_samples, ddof=1)),
    }
