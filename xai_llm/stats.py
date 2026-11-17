"""Small, dependency-free statistics (numpy only)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Least squares with intercept; returns [intercept, b1..bn]. A tiny ridge
    keeps bootstrap resamples with constant columns solvable."""
    A = np.column_stack([np.ones(len(X)), X])
    G = A.T @ A
    G[np.diag_indices_from(G)] += ridge * len(y)
    return np.linalg.solve(G, A.T @ y)


@dataclass
class Fit:
    intercept: float
    beta: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    p: np.ndarray


def fit_effects(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Fit:
    """OLS effects with percentile-bootstrap CIs and two-sided bootstrap p-values.

    X[i, j] = 1 if word j was KEPT in experiment i. beta[j] is the average change
    in the score caused by keeping word j (versus removing it), holding the other
    words as they were in the random experiments.
    """
    rng = rng or np.random.default_rng(0)
    m = len(y)
    b0 = ols(X, y)
    boots = np.empty((n_boot, X.shape[1]))
    for t in range(n_boot):
        idx = rng.integers(0, m, m)
        boots[t] = ols(X[idx], y[idx])[1:]
    a = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [a, 1.0 - a], axis=0)
    p_le = (np.sum(boots <= 0, axis=0) + 1) / (n_boot + 1)
    p_ge = (np.sum(boots >= 0, axis=0) + 1) / (n_boot + 1)
    p = np.minimum(1.0, 2.0 * np.minimum(p_le, p_ge))

    # A bootstrap p-value cannot go below 2/(B+1). When no resample crossed zero we
    # extrapolate with the normal approximation (z = effect / bootstrap SE) so that
    # strong effects are not capped at an artificial floor (and then inflated by the
    # false-discovery correction).
    beta = b0[1:]
    se = boots.std(axis=0, ddof=1)
    z = np.where(se > 1e-12, np.abs(beta) / np.maximum(se, 1e-12), np.where(np.abs(beta) > 1e-12, np.inf, 0.0))
    p_norm = np.array([math.erfc(v / math.sqrt(2.0)) if np.isfinite(v) else 0.0 for v in z])
    floor = 2.0 / (n_boot + 1)
    p = np.where(p <= floor * 1.0001, np.minimum(p_norm, floor), p)
    p = np.maximum(p, 1e-300)
    return Fit(float(b0[0]), beta, lo, hi, p)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values): controls the false discovery rate."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def cross_val_r2(
    X: np.ndarray, y: np.ndarray, k: int = 5, rng: Optional[np.random.Generator] = None
) -> Tuple[float, np.ndarray]:
    """K-fold cross-validated R^2 of the linear explanation, plus held-out predictions."""
    rng = rng or np.random.default_rng(0)
    m = len(y)
    k = max(2, min(k, m // 2))
    idx = rng.permutation(m)
    pred = np.empty(m)
    for fold in np.array_split(idx, k):
        train = np.setdiff1d(idx, fold, assume_unique=True)
        b = ols(X[train], y[train])
        pred[fold] = b[0] + X[fold] @ b[1:]
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst < 1e-12:
        return float("nan"), pred
    return 1.0 - float(np.sum((y - pred) ** 2)) / sst, pred


def rankdata(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
