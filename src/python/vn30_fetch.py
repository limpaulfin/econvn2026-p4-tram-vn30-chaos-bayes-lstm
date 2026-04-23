"""Fetch VN30 5-min bars via vnstock community tier + year-chunked retries."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from vnstock import Quote

VN30 = [
    "ACB","BID","CTG","DGC","FPT","GAS","GVR","HDB","HPG","LPB",
    "MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB",
    "TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VPL","VRE",
]
YEARS = [2023, 2024, 2025]
SOURCE = "VCI"
INTERVAL = "5m"  # override from CLI via set_interval()


def set_interval(iv: str) -> None:
    global INTERVAL
    INTERVAL = iv
RATE_DELAY_S = 1.1
MAX_RETRY = 3

log = logging.getLogger("vn30_fetch")


def pull_year(symbol: str, year: int) -> pd.DataFrame:
    q = Quote(source=SOURCE, symbol=symbol)
    for attempt in range(MAX_RETRY):
        try:
            df = q.history(start=f"{year}-01-01", end=f"{year}-12-31", interval=INTERVAL)
            if df is None or df.empty:
                log.warning("%s %d: empty", symbol, year)
                return pd.DataFrame()
            return df
        except Exception as exc:
            wait = (2 ** attempt) * 3
            log.warning("%s %d attempt %d: %s — backoff %ds", symbol, year, attempt, exc, wait)
            time.sleep(wait)
    log.error("%s %d: failed after %d retries", symbol, year, MAX_RETRY)
    return pd.DataFrame()


def pull_ticker(symbol: str, years: list[int], raw_dir: Path) -> dict:
    frames = []
    for y in years:
        df = pull_year(symbol, y)
        if not df.empty:
            df["ticker"] = symbol
            df["year"] = y
            frames.append(df)
        time.sleep(RATE_DELAY_S)
    if not frames:
        return {"ticker": symbol, "rows": 0, "status": "empty"}
    merged = pd.concat(frames, ignore_index=True)
    if "time" in merged.columns:
        merged = merged.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    target = raw_dir / f"{symbol}_{INTERVAL}_{years[0]}-{years[-1]}_{SOURCE}.parquet"
    merged.to_parquet(target, index=False)
    log.info("%s: %d rows -> %s", symbol, len(merged), target.name)
    return {"ticker": symbol, "rows": int(len(merged)), "file": target.name, "status": "ok"}
