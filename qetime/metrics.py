"""Evaluation metrics (paper Section 5.4.2): R^2 (Eq. 5), MSE (Eq. 6), NMSE (Eq. 7).

MAPE and Spearman's rho are reported as well because they are robust to the few
very slow (often noisy) measurements that dominate MSE-based metrics.
"""

from __future__ import annotations

import numpy as np


def mse(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean((y - pred) ** 2))


def nmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float)
    var = np.mean((y - y.mean()) ** 2)
    return float(mse(y, pred) / var) if var > 0 else float("nan")


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def mape(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean(np.abs(pred - y) / np.abs(y)))


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(y) < 3:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def all_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y, pred),
        "r2": r2(y, pred),
        "nmse": nmse(y, pred),
        "mape": mape(y, pred),
        "spearman": spearman(y, pred),
        "n": int(len(y)),
    }
