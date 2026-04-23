"""Smoke test: 5-min HF pipeline on 3 tickers (VIC, FPT, ACB).

Usage:
    python python/scripts/run_5m_smoke.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from src import utils, data_prep_5m, chaos_5m  # noqa: E402


SMOKE_TICKERS = ["VIC", "FPT", "ACB"]


def main() -> None:
    utils.configure_logging()
    log = logging.getLogger("smoke_5m")
    params = json.loads((ROOT / "config" / "parameters.json").read_text())
    utils.set_seed(params["random_seed"])

    log.info("STEP 1 — data_prep_5m on %s", SMOKE_TICKERS)
    prep = data_prep_5m.run(params, tickers=SMOKE_TICKERS)
    log.info("prep result: %s", prep)

    log.info("STEP 2 — chaos_5m (W=78, stride=78, n_c=30)")
    res = chaos_5m.run(params, window=78, stride=78, n_c=30, tickers=SMOKE_TICKERS)

    if res.empty:
        log.error("smoke FAILED: no rows produced")
        sys.exit(2)

    log.info("smoke PASS: %d chaos rows", len(res))
    log.info("summary by ticker:\n%s", res.groupby("ticker").size().to_string())
    by_t = res.groupby("ticker")[["lyap", "pe", "K01"]].mean()
    log.info("mean indicators:\n%s", by_t.to_string())


if __name__ == "__main__":
    main()
