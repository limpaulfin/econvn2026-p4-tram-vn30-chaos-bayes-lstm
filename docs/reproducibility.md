# Reproducibility notes

This short guide covers the exact software versions and shell invocations used to
regenerate the figures and tables reported in the ECONVN2026 submission.

## Host configuration used for the reference run

- OS: Ubuntu 24.04 LTS (kernel 6.17).
- Python: 3.10 from `/home/fong/Projects/.venv`.
- R: 4.3.3.
- Shell: bash 5.2.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Demo notebook

The notebook at `notebooks/demo_vcb_chaos_bayes_lstm.ipynb` runs end-to-end on the
shipped VCB parquet files. Expected wall-clock on an Intel i7 laptop with no GPU:

- Cells 1 to 4 (load + chaos indicators): under 30 seconds.
- Cell 5 (IAAFT surrogate null with 99 surrogates): 1 to 3 minutes.
- Cell 6 (LSTM ensemble, optional `RUN_LSTM=True`): 8 to 12 minutes on CPU.

## Command-line IAAFT driver

```bash
python src/scripts/run_surrogate_null.py \
    --ticker VCB --layer 1D \
    --n-surrogates 499 --n-iter 100 --alpha 0.05 \
    --out out/surrogate_null_VCB_1D.json
```

The output JSON carries `observed`, `null_mean`, `null_std`, `p_value`, and
`reject_null` per indicator.

## R verification

```bash
Rscript src/R/02-analysis.R
```

This writes `chaos_lyap_R.csv` and a side-by-side comparison against the Python
`chaos_indicators_1D.csv` for per-ticker mean Lyapunov.

## Known caveats

- The ipynb demo limits `n-surrogates` to 99 for interactive speed. The paper uses 499.
- The 0-1 $K$ and permutation-entropy tests will appear to fail the null rejection
  on VCB returns; this is the expected leptokurtic-noise behaviour documented by
  Federico (2020). The paper treats this as a finding, not a bug.
- `lstm_ensemble.train_ensemble_panel_avg` trains on CPU by default. Pass
  `device="cuda"` for GPU if available; results are stable within seed variance.
