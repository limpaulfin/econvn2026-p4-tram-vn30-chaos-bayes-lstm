"""Load VN30 daily KBS parquets + compute returns + Parkinson volatility.

DAILY PHASE (what we can run NOW on 2021-2025 bulk pull).
HF 5-min phase pending community-tier API key.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from . import utils

LOGGER = logging.getLogger(__name__)

VN30_BASKET_2026_04 = [
    "ACB","BID","CTG","DGC","FPT","GAS","GVR","HDB","HPG","LPB",
    "MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB",
    "TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VPL","VRE",
]


def load_ticker(ticker: str, raw_dir: Path) -> pd.DataFrame:
    """Load one ticker's daily parquet."""
    fp = raw_dir / f"{ticker}_1D_2021-2025_KBS.parquet"
    if not fp.exists():
        LOGGER.warning("Missing %s", fp.name)
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df["ticker"] = ticker
    df["time"] = pd.to_datetime(df["time"])
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns + Parkinson + realized proxy + range-based vol."""
    df = df.sort_values("time").reset_index(drop=True)
    df["ret_log"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_simple"] = df["close"].pct_change()
    # Parkinson 1980: sigma^2 = (1/(4*ln2)) * (ln(H/L))^2
    df["parkinson_vol"] = np.sqrt(
        (1.0 / (4.0 * np.log(2.0))) * (np.log(df["high"] / df["low"]) ** 2)
    )
    df["vol_daily"] = df["ret_log"].rolling(20, min_periods=5).std()
    return df


def run(params: dict) -> Dict[str, int]:
    """Process all 30 tickers. Save to data/processed/."""
    p = utils.paths(params)
    out: Dict[str, int] = {}
    all_frames: List[pd.DataFrame] = []
    for ticker in VN30_BASKET_2026_04:
        df = load_ticker(ticker, p["raw"])
        if df.empty:
            continue
        df = compute_features(df)
        df = df.dropna(subset=["ret_log"])
        target = p["processed"] / f"{ticker}_1D_processed.parquet"
        df.to_parquet(target, index=False)
        out[ticker] = len(df)
        all_frames.append(df)
        LOGGER.info("processed %s: %d rows", ticker, len(df))
    if all_frames:
        panel = pd.concat(all_frames, ignore_index=True)
        panel.to_parquet(p["processed"] / "vn30_panel_1D.parquet", index=False)
        LOGGER.info("panel saved: %d rows across %d tickers", len(panel), len(out))
    return out
