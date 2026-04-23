"""Publication figures (CUD palette, 300 DPI)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils

LOGGER = logging.getLogger(__name__)
CUD = utils.CUD


def fig1_chaos_concordance(chaos_df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    agg = chaos_df.groupby("ticker")[["lyap", "pe", "K01"]].mean().dropna()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(agg))
    ax.bar(x - 0.25, agg["lyap"].values, 0.25, label="Lyapunov", color=CUD["blue"])
    ax.bar(x, agg["pe"].values, 0.25, label="Perm. entropy", color=CUD["orange"])
    ax.bar(x + 0.25, agg["K01"].values, 0.25, label="0-1 K-value", color=CUD["green"])
    ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=75, fontsize=8)
    ax.set_ylabel("Mean indicator"); ax.set_title("Per-ticker mean chaos indicators")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    LOGGER.info("fig1 %s", out.name)


def fig2_pit_histogram(preds: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    mu, sigma, y = preds["mu"], preds["sigma"], preds["y_true"]
    pit = np.array([norm.cdf(y, loc=mu[i], scale=np.maximum(sigma[i], 1e-6))
                    for i in range(mu.shape[0])]).mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(pit, bins=10, edgecolor=CUD["black"], color=CUD["sky"], alpha=0.85)
    ax.axhline(len(pit) / 10, color=CUD["vermilion"], ls="--", label="Uniform target")
    ax.set_xlabel("PIT"); ax.set_ylabel("Count")
    ax.set_title("PIT histogram (BMA)")
    ax.legend(); fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    LOGGER.info("fig2 %s", out.name)


def fig3_weights(bma_report: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    w = np.array(bma_report["weights"])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(np.arange(len(w)), w, color=CUD["purple"])
    ax.set_xlabel("Member"); ax.set_ylabel("Stacking weight")
    ax.set_title(f"Pseudo-BMA weights (N={len(w)})")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    LOGGER.info("fig3 %s", out.name)


def run(params: dict) -> None:
    p = utils.paths(params)
    utils.setup_mpl(params["figures_dpi"])
    fig_dir = p["root"] / "latex" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if (p["output"] / "chaos_indicators_1D.parquet").exists():
        df = pd.read_parquet(p["output"] / "chaos_indicators_1D.parquet")
        fig1_chaos_concordance(df, fig_dir / "fig1_chaos_concordance.pdf")

    if (p["output"] / "ensemble_preds.npz").exists():
        preds = {k: v for k, v in np.load(p["output"] / "ensemble_preds.npz").items()}
        fig2_pit_histogram(preds, fig_dir / "fig2_pit_histogram.pdf")

    if (p["output"] / "bma_weights.json").exists():
        fig3_weights(json.loads((p["output"] / "bma_weights.json").read_text()),
                     fig_dir / "fig3_bma_weights.pdf")
    LOGGER.info("visualization done")
