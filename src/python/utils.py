"""Shared utilities: logging, seeding, IO."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def paths(params: dict) -> dict:
    root = Path(__file__).resolve().parents[2]
    return {
        "root": root,
        "raw": ensure_dir(root / "data" / "raw"),
        "processed": ensure_dir(root / "data" / "processed"),
        "output": ensure_dir(root / "python" / "output"),
        "logs": ensure_dir(root / "python" / "logs"),
    }


CUD = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermilion": "#D55E00", "purple": "#CC79A7",
}


def setup_mpl(dpi: int) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "figure.dpi": dpi, "savefig.dpi": dpi,
        "axes.grid": True, "grid.alpha": 0.3,
    })
