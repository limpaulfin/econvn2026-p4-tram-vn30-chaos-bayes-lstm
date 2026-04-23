"""Rolling chaos indicators on 5-min HF data.

Window W = 78 bars (~1 trading day: 27 morning + 21 afternoon, padded).
Stride > 1 to keep compute feasible across 242k bars × 30 tickers.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from . import utils, chaos_indicators

LOGGER = logging.getLogger(__name__)


def _rolling_chaos_5m(
    df: pd.DataFrame, window: int, stride: int, n_c: int
) -> pd.DataFrame:
    rets = df["ret_log"].to_numpy()
    idx = df["time"].to_numpy()
    ticker = df["ticker"].iloc[0]
    rows = {"time": [], "ticker": [], "lyap": [], "pe": [], "K01": []}
    for i in range(window, len(rets), stride):
        win = rets[i - window:i]
        if np.isnan(win).any() or np.std(win) == 0:
            continue
        rows["time"].append(idx[i])
        rows["ticker"].append(ticker)
        rows["lyap"].append(chaos_indicators.largest_lyapunov(win))
        rows["pe"].append(chaos_indicators.permutation_entropy(win))
        rows["K01"].append(chaos_indicators.chaos01_K(win, n_c=n_c))
    return pd.DataFrame(rows)


def run(
    params: dict,
    window: int = 78,
    stride: int = 78,
    n_c: int = 30,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Run rolling chaos on panel_5m. Default stride=window ⇒ one value per day."""
    p = utils.paths(params)
    panel = pd.read_parquet(p["processed"] / "vn30_panel_5m.parquet")
    if tickers is not None:
        panel = panel[panel["ticker"].isin(tickers)]
    pieces: List[pd.DataFrame] = []
    for ticker, grp in panel.groupby("ticker"):
        if len(grp) < window + 5:
            LOGGER.info("skip %s (%d rows)", ticker, len(grp))
            continue
        piece = _rolling_chaos_5m(grp.reset_index(drop=True), window, stride, n_c)
        pieces.append(piece)
        LOGGER.info(
            "chaos_5m %s: %d windows (mean λ=%.4f, PE=%.3f, K=%.3f)",
            ticker, len(piece),
            np.nanmean(piece["lyap"]) if len(piece) else float("nan"),
            np.nanmean(piece["pe"]) if len(piece) else float("nan"),
            np.nanmean(piece["K01"]) if len(piece) else float("nan"),
        )
    if not pieces:
        LOGGER.warning("no chaos_5m rows produced")
        return pd.DataFrame()
    res = pd.concat(pieces, ignore_index=True)
    res.to_parquet(p["output"] / "chaos_indicators_5m.parquet", index=False)
    LOGGER.info("chaos_5m saved: %d rows across %d tickers", len(res), res["ticker"].nunique())
    return res
