"""Statistical helpers for paired AUC comparison and pooled bootstrap CIs.

Two public helpers are exposed:

- :func:`delong_test`: two-tailed DeLong test for paired ROC AUCs on a
  shared set of samples (Sun and Xu, 2014 fast algorithm).
- :func:`bootstrap_pooled_ci`: cohort-stratified bootstrap 95% CI on a
  pooled AUC over LODO held-out predictions. Stratifying by cohort
  preserves the LODO sample-size structure across resamples.

Both functions are vendored from the equivalent routines used in the
crc-metagenomics study (``scripts/auc_comparison.py`` and
``scripts/bootstrap_ci.py``) so the package is self-contained.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank transform with ties resolved by averaging tied positions.

    Used as the inner kernel of the Sun and Xu (2014) DeLong algorithm.
    """
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> dict[str, float]:
    """Two-tailed DeLong test for paired AUCs on a single sample set.

    Implements the fast algorithm of Sun and Xu (2014); the two
    prediction vectors must be scored on the same samples in the same
    order (e.g. paired pooled LODO held-out predictions from two
    classifiers trained on the same folds).

    Parameters
    ----------
    y_true
        Binary ground-truth labels (0/1), shape ``(n,)``.
    y_prob_a, y_prob_b
        Predicted scores from the two classifiers, each shape ``(n,)``.

    Returns
    -------
    result : dict
        Keys: ``auc_a, auc_b, auc_diff, z, p_value, n``. ``p_value`` is
        the two-tailed p-value under the asymptotic normal null.
    """
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("delong_test requires both classes to be present")

    aucs: list[float] = []
    v01s: list[np.ndarray] = []
    v10s: list[np.ndarray] = []
    for y_prob in (y_prob_a, y_prob_b):
        y_prob = np.asarray(y_prob, dtype=float)
        x_pos = y_prob[pos]
        x_neg = y_prob[neg]
        tx = _midrank(x_pos)
        ty = _midrank(x_neg)
        tz = _midrank(np.concatenate([x_pos, x_neg]))
        auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        aucs.append(float(auc))
        v01s.append(v01)
        v10s.append(v10)

    auc_a, auc_b = aucs
    S01 = np.cov(np.vstack(v01s))
    S10 = np.cov(np.vstack(v10s))
    S = S01 / m + S10 / n
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        z = 0.0
        p = 1.0
    else:
        z = float((auc_a - auc_b) / np.sqrt(var_diff))
        p = float(2 * (1 - norm.cdf(abs(z))))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": auc_a - auc_b,
        "z": z,
        "p_value": p,
        "n": int(m + n),
    }


# ---------------------------------------------------------------------------
# Cohort-stratified bootstrap CI on a pooled LODO AUC
# ---------------------------------------------------------------------------


def bootstrap_pooled_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cohort: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cohort-stratified bootstrap CI for a pooled AUC.

    On each bootstrap iteration the samples are resampled with
    replacement within each cohort separately, then concatenated before
    AUC is computed. This preserves the LODO cohort structure (each
    fold's contribution is bounded by its true sample size) and avoids
    cohort-imbalanced resamples that an i.i.d. pooled bootstrap can
    produce.

    Parameters
    ----------
    y_true
        Binary ground-truth labels (0/1), shape ``(n,)``.
    y_prob
        Predicted scores aligned with ``y_true``, shape ``(n,)``.
    cohort
        Cohort identifier per sample, shape ``(n,)``. Any hashable
        dtype is accepted.
    n_boot
        Number of bootstrap resamples. Default 10,000.
    seed
        Seed for the underlying NumPy RNG. Default 42 (matches the
        crc-metagenomics study).
    alpha
        Two-sided CI level; the returned CI covers
        ``[alpha/2, 1 - alpha/2]``. Default 0.05 -> 95% CI.

    Returns
    -------
    result : dict
        Keys: ``auc, ci_low, ci_high, n_boot_kept, alpha, n``. ``auc``
        is the point estimate on the full pooled data. ``n_boot_kept``
        is the number of bootstrap iterations in which both classes
        were present (single-class iterations are dropped).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    cohort = np.asarray(cohort)
    if not (len(y_true) == len(y_prob) == len(cohort)):
        raise ValueError(
            "y_true, y_prob, and cohort must all have the same length"
        )
    if len(y_true) == 0:
        raise ValueError("bootstrap_pooled_ci received an empty input")

    point = float(roc_auc_score(y_true, y_prob))
    rng = np.random.RandomState(seed)
    cohort_to_idx = {
        c: np.where(cohort == c)[0] for c in pd.unique(cohort)
    }
    aucs: list[float] = []
    for _ in range(n_boot):
        sampled = [
            rng.choice(idxs, size=len(idxs), replace=True)
            for idxs in cohort_to_idx.values()
        ]
        idx = np.concatenate(sampled)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, yp)))

    if not aucs:
        raise ValueError(
            "bootstrap_pooled_ci: no resamples retained both classes; "
            "check for severe class imbalance within cohorts"
        )

    aucs_arr = np.asarray(aucs)
    lo = float(np.percentile(aucs_arr, 100 * (alpha / 2.0)))
    hi = float(np.percentile(aucs_arr, 100 * (1 - alpha / 2.0)))
    return {
        "auc": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_kept": len(aucs),
        "alpha": float(alpha),
        "n": int(len(y_true)),
    }


__all__ = ["delong_test", "bootstrap_pooled_ci"]
