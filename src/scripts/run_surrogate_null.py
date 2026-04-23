"""
Run IAAFT surrogate-data null test on one VN30 ticker for K, PE, lambda.

Usage:
  python -m python.scripts.run_surrogate_null \
      --ticker VCB --layer 1D --n-surrogates 499 --alpha 0.05 --out python/output/surrogate_null_VCB_1D.json

Output:
  JSON with observed / null_mean / null_std / p_value / reject_null per indicator.

Status:
  MVP driver. Runs IAAFT against one ticker/layer. Cross-sectional run (30 tickers)
  is wired in camera-ready. This script exists to support the narrative in
  Section 4.2 / 4.5 that K-readings are calibrated against a surrogate null.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.python.chaos_indicators import chaos01_K, largest_lyapunov, permutation_entropy  # noqa: E402
from src.python.surrogates import surrogate_null_test  # noqa: E402


def load_returns(ticker: str, layer: str) -> np.ndarray:
    paper_path = REPO_ROOT / "data" / "processed" / f"{ticker}_{layer}_processed.parquet"
    public_path = REPO_ROOT / "data" / f"{ticker}_{layer}_processed.parquet"
    processed = paper_path if paper_path.exists() else public_path
    df = pd.read_parquet(processed)
    col = "ret" if "ret" in df.columns else "log_return"
    return df[col].dropna().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--layer", choices=["1D", "5m"], default="1D")
    parser.add_argument("--n-surrogates", type=int, default=499)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x = load_returns(args.ticker, args.layer)
    rng = np.random.default_rng(42)

    indicators = {
        "K_01_test": chaos01_K,
        "permutation_entropy": permutation_entropy,
        "lyapunov_rosenstein": largest_lyapunov,
    }

    report: dict = {"ticker": args.ticker, "layer": args.layer, "n_obs": int(x.size), "results": {}}
    for name, fn in indicators.items():
        res = surrogate_null_test(
            x,
            indicator_fn=fn,
            n_surrogates=args.n_surrogates,
            n_iter=args.n_iter,
            alpha=args.alpha,
            rng=rng,
        )
        report["results"][name] = {
            "observed": res["observed"],
            "null_mean": res["null_mean"],
            "null_std": res["null_std"],
            "p_value": res["p_value"],
            "reject_null": bool(res["reject_null"]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
