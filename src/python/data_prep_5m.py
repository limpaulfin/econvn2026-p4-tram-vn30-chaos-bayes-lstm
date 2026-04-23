"""Load VN30 5-min intraday parquets + compute HF features.

HF PHASE. Operates on VCI 5m bars 2023-09-11 → 2025-12-31.
Trading session filter: morning 09:15-11:30 + afternoon 13:00-14:45.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import utils
from .data_prep import VN30_BASKET_2026_04

LOGGER = logging.getLogger(__name__)

SESSION_MORNING = (pd.Timestamp("09:15:00").time(), pd.Timestamp("11:30:00").time())
SESSION_AFTERNOON = (pd.Timestamp("13:00:00").time(), pd.Timestamp("14:45:00").time())


def _in_session(ts: pd.Series) -> pd.Series:
    t = ts.dt.time
    return ((t >= SESSION_MORNING[0]) & (t <= SESSION_MORNING[1])) | (
        (t >= SESSION_AFTERNOON[0]) & (t <= SESSION_AFTERNOON[1])
    )


def load_ticker_5m(ticker: str, raw_dir: Path) -> pd.DataFrame:
    fp = raw_dir / f"{ticker}_5m_2023-2025_VCI.parquet"
    if not fp.exists():
        LOGGER.warning("Missing %s", fp.name)
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["close"]).loc[_in_session(df["time"])].reset_index(drop=True)
    df["ticker"] = ticker
    return df


def compute_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    df["ret_log"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_abs"] = df["ret_log"].abs()
    df["session_date"] = df["time"].dt.date
    df["rv_5m"] = df["ret_log"] ** 2
    df["parkinson_5m"] = np.sqrt(
        (1.0 / (4.0 * np.log(2.0))) * (np.log(df["high"] / df["low"]) ** 2)
    )
    return df.dropna(subset=["ret_log"]).reset_index(drop=True)


def daily_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min squared returns to daily realized variance."""
    g = df.groupby(["ticker", "session_date"])
    return g.agg(
        rv_daily=("rv_5m", "sum"),
        bars=("ret_log", "count"),
        parkinson_daily=("parkinson_5m", "mean"),
    ).reset_index()


def run(params: dict, tickers: List[str] | None = None) -> Dict[str, int]:
    p = utils.paths(params)
    out: Dict[str, int] = {}
    frames: List[pd.DataFrame] = []
    basket = tickers or VN30_BASKET_2026_04
    for ticker in basket:
        df = load_ticker_5m(ticker, p["raw"])
        if df.empty:
            continue
        df = compute_features_5m(df)
        target = p["processed"] / f"{ticker}_5m_processed.parquet"
        df.to_parquet(target, index=False)
        out[ticker] = len(df)
        frames.append(df)
        LOGGER.info("5m %s: %d rows", ticker, len(df))
    if frames:
        panel = pd.concat(frames, ignore_index=True)
        panel.to_parquet(p["processed"] / "vn30_panel_5m.parquet", index=False)
        rv = daily_realized_vol(panel)
        rv.to_parquet(p["processed"] / "vn30_rv_daily_from_5m.parquet", index=False)
        LOGGER.info("panel_5m %d rows; rv_daily %d rows", len(panel), len(rv))
    return out
