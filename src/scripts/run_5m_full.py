"""Full VN30 5-min pipeline: 30 tickers.

Usage:
    python python/scripts/run_5m_full.py
Produces:
    data/processed/vn30_panel_5m.parquet
    data/processed/vn30_rv_daily_from_5m.parquet
    python/output/chaos_indicators_5m.parquet
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from src import utils, data_prep_5m, chaos_5m  # noqa: E402


def main() -> None:
    utils.configure_logging()
    log = logging.getLogger("full_5m")
    params = json.loads((ROOT / "config" / "parameters.json").read_text())
    utils.set_seed(params["random_seed"])

    log.info("STEP 1 — data_prep_5m on full VN30 (30 tickers)")
    prep = data_prep_5m.run(params)
    log.info("prep total tickers: %d", len(prep))

    log.info("STEP 2 — chaos_5m (W=78, stride=78, n_c=30)")
    res = chaos_5m.run(params, window=78, stride=78, n_c=30)

    if res.empty:
        log.error("FAILED: no chaos rows produced")
        sys.exit(2)

    log.info("PASS: %d chaos_5m rows across %d tickers", len(res), res["ticker"].nunique())
    agg = res.groupby("ticker")[["lyap", "pe", "K01"]].mean().round(4)
    log.info("mean indicators per ticker:\n%s", agg.to_string())
    agg.to_csv(ROOT / "python" / "output" / "chaos_5m_summary.csv")
    log.info("summary saved to python/output/chaos_5m_summary.csv")


if __name__ == "__main__":
    main()
