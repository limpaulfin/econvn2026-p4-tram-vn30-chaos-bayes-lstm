"""Pseudo-BMA stacking + calibration diagnostics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def stacking_weights(val_losses: np.ndarray) -> np.ndarray:
    """Softmax of negative validation loss -> stacking weight."""
    scores = -np.asarray(val_losses)
    w = np.exp(scores - scores.max())
    return w / w.sum()


def bma_loglik(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Average log-likelihood of BMA predictive on test set."""
    from scipy.stats import norm
    pdfs = np.stack([norm.pdf(y, loc=mu[i], scale=sigma[i]) for i in range(mu.shape[0])])
    bma = (w[:, None] * pdfs).sum(axis=0)
    return float(np.log(bma + 1e-12).mean())


def uniform_loglik(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Baseline: simple uniform average log-likelihood."""
    from scipy.stats import norm
    pdfs = np.stack([norm.pdf(y, loc=mu[i], scale=sigma[i]) for i in range(mu.shape[0])])
    return float(np.log(pdfs.mean(axis=0) + 1e-12).mean())


def brier_quantile(mu, sigma, y, q_levels=(0.1, 0.5, 0.9)) -> dict:
    """Quantile Brier scores against empirical coverage."""
    from scipy.stats import norm
    stacked_mu = mu.mean(axis=0)
    stacked_sigma = np.sqrt((sigma ** 2 + mu ** 2).mean(axis=0) - stacked_mu ** 2)
    out = {}
    for q in q_levels:
        level = norm.ppf(q, loc=stacked_mu, scale=np.maximum(stacked_sigma, 1e-6))
        hit = (y <= level).astype(float)
        out[f"q{int(q*100)}_cov"] = float(hit.mean())
        out[f"q{int(q*100)}_target"] = float(q)
    return out


def save_report(out_dir: Path, w: np.ndarray, bma_ll: float,
                uni_ll: float, quantiles: dict, n_members: int, n_test: int) -> None:
    report = {
        "weights": w.tolist(),
        "bma_test_loglik_mean": bma_ll,
        "uniform_avg_test_loglik_mean": uni_ll,
        "quantile_calibration": quantiles,
        "n_members": int(n_members),
        "n_test": int(n_test),
    }
    (out_dir / "bma_weights.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("BMA report saved: bma_ll=%.4f vs uniform=%.4f", bma_ll, uni_ll)
