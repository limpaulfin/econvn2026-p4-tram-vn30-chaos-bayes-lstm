# VN30 chaos-Bayes-LSTM (ECONVN2026 companion)

Code + small data slice for the ECONVN2026 submission.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/demo_vcb_chaos_bayes_lstm.ipynb
```

## Layout

```
src/python/   chaos indicators, IAAFT surrogates, LSTM, BMA
src/R/        Lyapunov cross-check via nonlinearTseries
src/scripts/  CLI drivers
notebooks/    end-to-end demo (VCB)
data/         processed parquet slice (~2.5 MB)
```
