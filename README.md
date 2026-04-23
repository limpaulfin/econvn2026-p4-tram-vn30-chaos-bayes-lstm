# Chaos-aware Bayesian LSTM on VN30 — ECONVN2026 companion

Public companion code and processed data for the ECONVN2026 submission *"Chaos-aware Bayesian LSTM on Vietnamese high-frequency equity data: a null-calibrated audit of the VN30 basket."*

## Summary

This repository ships the source code, a runnable Jupyter notebook, and a small processed-data slice so that a reader can reproduce the core experiments end-to-end on a laptop. The pipeline computes three classical chaos diagnostics (Rosenstein largest Lyapunov exponent, Bandt-Pompe permutation entropy, Gottwald-Melbourne 0-1 $K$-value) on VN30 daily and five-minute returns, then calibrates the diagnostics against IAAFT (Schreiber-Schmitz 1996) surrogate null data to avoid the leptokurtic-overrejection failure mode documented by Federico (2020) and Webel (2012). A five-member LSTM ensemble is combined by pseudo-BMA stacking (Yao et al., 2018) to produce probabilistic return forecasts, and the predictive distribution is assessed with a PIT coverage diagnostic.

Scope is the ECONVN2026 conference submission. No journal upgrade is pursued from this release.

## System requirements

- Ubuntu 22.04 LTS or later (the scripts are developed on Linux; macOS should work).
- Python 3.10 or later.
- Optional: R 4.3 or later for the `src/R/` verification pipeline.
- Approximately 200 MB disk for data + dependencies.
- Approximately 2 GB RAM for the notebook; CPU is sufficient.

## Installation

```bash
git clone https://github.com/limpaulfin/econvn2026-p4-tram-vn30-chaos-bayes-lstm.git
cd econvn2026-p4-tram-vn30-chaos-bayes-lstm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution

Launch the demo notebook:

```bash
jupyter notebook notebooks/demo_vcb_chaos_bayes_lstm.ipynb
```

Run the IAAFT surrogate null test from the command line (example VCB daily):

```bash
python src/scripts/run_surrogate_null.py \
    --ticker VCB --layer 1D --n-surrogates 499 --out out/surrogate_null_VCB_1D.json
```

## Dataset

Shipped with the repo under `data/`:

| File | Rows | Description |
|------|------|-------------|
| `VCB_1D_processed.parquet` | ~4,000 | VCB daily bars 2008-2025, log-returns included |
| `VCB_5m_processed.parquet` | ~242,000 | VCB five-minute bars 2023-09 to 2025-12 |
| `vn30_panel_1D.parquet` | ~36,000 | Balanced VN30 daily panel 2021-2025 |

All data is adjusted for corporate actions and filtered to in-session hours. Source is the vnstock VCI endpoint; license terms are documented in the paper's data-availability statement.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                             # processed parquet shipped for offline demo
├── notebooks/
│   └── demo_vcb_chaos_bayes_lstm.ipynb
└── src/
    ├── python/                       # chaos + LSTM + BMA + surrogates
    │   ├── chaos_indicators.py
    │   ├── surrogates.py             # IAAFT + null-test driver
    │   ├── lstm_ensemble.py
    │   ├── bma.py
    │   └── ...
    ├── R/                            # nonlinearTseries cross-check
    │   ├── 00-utils.R
    │   ├── 01-data-prep.R
    │   ├── 02-analysis.R
    │   └── 03-visualization.R
    └── scripts/
        ├── run_surrogate_null.py
        ├── run_5m_full.py
        └── run_5m_smoke.py
```

## How to cite

```bibtex
@inproceedings{lim2026econvn_vn30chaosbma,
  author    = {Lim, Paul (Fong) and Tram, {anonymised}},
  title     = {Chaos-aware Bayesian {LSTM} on {V}ietnamese {HF} equity data: a null-calibrated audit of the {VN30} basket},
  booktitle = {Proceedings of ECONVN2026},
  year      = {2026},
  address   = {Ho Chi Minh City, Vietnam}
}
```

## License

MIT. See `LICENSE`.
