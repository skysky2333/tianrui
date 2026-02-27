from __future__ import annotations

import numpy as np


def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"rankdata expects 1D array; got shape {x.shape}")
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty((n,), dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while (j + 1) < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1

    return ranks


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"pearson_r expects same shapes; got {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")

    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.sqrt(float(np.sum(x * x)) * float(np.sum(y * y))))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_avg_ties(x)
    ry = _rankdata_avg_ties(y)
    return pearson_r(rx, ry)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"r2_score expects same shapes; got {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        return float("nan")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(y_true.mean())) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"rmse expects same shapes; got {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"mae expects same shapes; got {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))

