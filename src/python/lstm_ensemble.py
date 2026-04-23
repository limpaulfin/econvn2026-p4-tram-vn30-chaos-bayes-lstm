"""LSTM ensemble with heteroscedastic (mu, sigma) head."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def prepare_sequences(x: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows of length `seq_len` -> next-step target."""
    xs, ys = [], []
    for i in range(seq_len, len(x)):
        xs.append(x[i - seq_len:i])
        ys.append(x[i])
    return np.array(xs), np.array(ys)


def train_single_lstm(train_x, train_y, val_x, val_y, params, seed: int):
    """Train one LSTM member. Return (model, best_val_loss)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class LSTMModel(nn.Module):
        def __init__(self, hidden=64, layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=layers,
                                batch_first=True, dropout=dropout)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    model = LSTMModel(
        params["lstm"]["hidden"], params["lstm"]["layers"], params["lstm"]["dropout"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lstm"]["lr"])
    tx = torch.tensor(train_x[..., None], dtype=torch.float32, device=device)
    ty = torch.tensor(train_y, dtype=torch.float32, device=device)
    vx = torch.tensor(val_x[..., None], dtype=torch.float32, device=device)
    vy = torch.tensor(val_y, dtype=torch.float32, device=device)

    best_val, patience, wait = float("inf"), params["lstm"]["early_stop_patience"], 0
    for _ in range(min(params["lstm"]["epochs_max"], 30)):
        model.train(); opt.zero_grad()
        out = model(tx)
        mu, ls = out[:, 0], out[:, 1].clamp(-5, 5)
        loss = 0.5 * (((ty - mu) / torch.exp(ls)) ** 2 + 2 * ls).mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(vx)
            vmu, vls = vout[:, 0], vout[:, 1].clamp(-5, 5)
            vloss = 0.5 * (((vy - vmu) / torch.exp(vls)) ** 2 + 2 * vls).mean().item()
        if vloss < best_val - 1e-4:
            best_val, wait = vloss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model, best_val


def predict_member(model, test_x):
    """Return (mu, sigma) arrays for test sequences."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    tt = torch.tensor(test_x[..., None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        out = model(tt)
        mu = out[:, 0].cpu().numpy()
        sigma = np.exp(out[:, 1].clamp(-5, 5).cpu().numpy())
    return mu, sigma
