"""Orchestrator: chaos indicators + LSTM ensemble + pseudo-BMA (DAILY).

DAILY PHASE — operates on panel saved by data_prep.
Rolling window W = 250 (one trading year). Ensemble N default from params.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from . import utils, chaos_indicators, lstm_ensemble, bma

LOGGER = logging.getLogger(__name__)


def _rolling_chaos(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rets = df["ret_log"].to_numpy()
    idx = df["time"].to_numpy()
    out = {"time": [], "ticker": [], "lyap": [], "pe": [], "K01": []}
    ticker = df["ticker"].iloc[0]
    for i in range(window, len(rets)):
        win = rets[i - window:i]
        if np.isnan(win).any():
            continue
        out["time"].append(idx[i])
        out["ticker"].append(ticker)
        out["lyap"].append(chaos_indicators.largest_lyapunov(win))
        out["pe"].append(chaos_indicators.permutation_entropy(win))
        out["K01"].append(chaos_indicators.chaos01_K(win, n_c=30))
    return pd.DataFrame(out)


def run_chaos(params: dict) -> None:
    p = utils.paths(params)
    panel = pd.read_parquet(p["processed"] / "vn30_panel_1D.parquet")
    window = 250
    pieces: List[pd.DataFrame] = []
    for ticker, grp in panel.groupby("ticker"):
        if len(grp) < window + 10:
            LOGGER.info("skip %s (%d rows)", ticker, len(grp))
            continue
        pieces.append(_rolling_chaos(grp.reset_index(drop=True), window))
        LOGGER.info("chaos %s: done", ticker)
    if pieces:
        res = pd.concat(pieces, ignore_index=True)
        res.to_parquet(p["output"] / "chaos_indicators_1D.parquet", index=False)
        LOGGER.info("chaos saved: %d rows", len(res))


def run_lstm(params: dict) -> None:
    p = utils.paths(params)
    panel = pd.read_parquet(p["processed"] / "vn30_panel_1D.parquet")
    avg = panel.groupby("time")["ret_log"].mean().reset_index()
    rets = avg["ret_log"].dropna().to_numpy(dtype=np.float32)

    seq_len = 20
    X, y = lstm_ensemble.prepare_sequences(rets, seq_len)
    n = len(X)
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    tx, ty = X[:split_train], y[:split_train]
    vx, vy = X[split_train:split_val], y[split_train:split_val]
    te_x, te_y = X[split_val:], y[split_val:]

    N = min(params["lstm"]["ensemble_size"], 5)  # MVP cap
    mus, sigmas, val_losses = [], [], []
    for i in range(N):
        LOGGER.info("training member %d/%d", i + 1, N)
        model, val_loss = lstm_ensemble.train_single_lstm(
            tx, ty, vx, vy, params, seed=20260422 + i,
        )
        mu, sigma = lstm_ensemble.predict_member(model, te_x)
        mus.append(mu); sigmas.append(sigma); val_losses.append(val_loss)

    np.savez(
        p["output"] / "ensemble_preds.npz",
        mu=np.array(mus), sigma=np.array(sigmas),
        y_true=te_y, val_losses=np.array(val_losses),
    )
    LOGGER.info("ensemble saved: %d members x %d test", N, len(te_y))


def run_bma(params: dict) -> None:
    p = utils.paths(params)
    data = np.load(p["output"] / "ensemble_preds.npz")
    mu, sigma = data["mu"], data["sigma"]
    y = data["y_true"]
    w = bma.stacking_weights(data["val_losses"])
    bma_ll = bma.bma_loglik(mu, sigma, y, w)
    uni_ll = bma.uniform_loglik(mu, sigma, y)
    quantiles = bma.brier_quantile(mu, sigma, y)
    bma.save_report(p["output"], w, bma_ll, uni_ll, quantiles, mu.shape[0], len(y))
