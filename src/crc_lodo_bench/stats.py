"""Statistical helpers for paired AUC comparison and pooled bootstrap CIs.

Two public helpers are exposed:

- :func:`delong_test`: two-tailed DeLong test for paired ROC AUCs on a
  shared set of samples (Sun and Xu, 2014 fast algorithm). NOTE: the
  DeLong covariance estimator assumes the two prediction vectors are
  scored on i.i.d. samples from a single population. When applied to
  pooled LODO predictions (where each sample's score is conditional on
  a different held-out training fold) this assumption is approximate;
  treat the resulting p-value as descriptive and corroborate with
  per-fold paired t-tests or the cohort-stratified bootstrap.
- :func:`bootstrap_pooled_ci`: cohort-stratified bootstrap CI on a
  pooled AUC over LODO held-out predictions. Stratifying by cohort
  preserves the LODO sample-size structure across resamples. Supports
  both the percentile method (default, matches the published study) and
  the bias-corrected accelerated (BCa) method of Efron (1987) for
  skewed AUC distributions.

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


def _bca_endpoints(
    point: float,
    boots: np.ndarray,
    jackknife: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected accelerated (BCa) CI endpoints (Efron, 1987).

    ``point`` is the original-sample statistic. ``boots`` is the array
    of bootstrap replicates. ``jackknife`` is the array of leave-one-out
    jackknife replicates of the statistic (used to estimate skewness).
    The returned (lo, hi) are the BCa-adjusted percentile endpoints at
    a two-sided ``alpha``.
    """
    boots = np.asarray(boots, dtype=float)
    # z0: bias correction from the proportion of bootstrap replicates
    # strictly less than the point estimate.
    frac_less = float(np.mean(boots < point))
    # Guard against degenerate 0/1 fractions which would push z0 to
    # +/- infinity; clip just inside the open unit interval.
    eps = 1.0 / (10.0 * len(boots))
    frac_less = float(np.clip(frac_less, eps, 1.0 - eps))
    z0 = float(norm.ppf(frac_less))

    # Acceleration constant a-hat from jackknife replicates.
    jk = np.asarray(jackknife, dtype=float)
    jk_mean = float(np.mean(jk))
    num = float(np.sum((jk_mean - jk) ** 3))
    den = 6.0 * (float(np.sum((jk_mean - jk) ** 2)) ** 1.5)
    a_hat = 0.0 if den == 0.0 else num / den

    z_lo = norm.ppf(alpha / 2.0)
    z_hi = norm.ppf(1.0 - alpha / 2.0)
    alpha_lo = float(norm.cdf(z0 + (z0 + z_lo) / (1.0 - a_hat * (z0 + z_lo))))
    alpha_hi = float(norm.cdf(z0 + (z0 + z_hi) / (1.0 - a_hat * (z0 + z_hi))))
    # Clip into [0, 1] in case the corrected endpoints push past either
    # tail of the bootstrap distribution.
    alpha_lo = float(np.clip(alpha_lo, 0.0, 1.0))
    alpha_hi = float(np.clip(alpha_hi, 0.0, 1.0))
    lo = float(np.percentile(boots, 100.0 * alpha_lo))
    hi = float(np.percentile(boots, 100.0 * alpha_hi))
    return lo, hi


def bootstrap_pooled_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cohort: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    method: str = "percentile",
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
    method
        Either ``"percentile"`` (default; matches the published study)
        or ``"bca"`` for the bias-corrected accelerated (BCa) intervals
        of Efron (1987). BCa is generally preferred for skewed bounded
        statistics like AUC; the percentile default is preserved for
        backwards compatibility with manuscript figures.

    Returns
    -------
    result : dict
        Keys: ``auc, ci_low, ci_high, n_boot_kept, alpha, n, method``.
        ``auc`` is the point estimate on the full pooled data.
        ``n_boot_kept`` is the number of bootstrap iterations in which
        both classes were present (single-class iterations are dropped).
        ``method`` echoes the chosen CI method.
    """
    if method not in {"percentile", "bca"}:
        raise ValueError(
            f"bootstrap_pooled_ci: method must be 'percentile' or 'bca', "
            f"got {method!r}"
        )

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
    if method == "percentile":
        lo = float(np.percentile(aucs_arr, 100 * (alpha / 2.0)))
        hi = float(np.percentile(aucs_arr, 100 * (1 - alpha / 2.0)))
    else:
        # BCa: build leave-one-out jackknife replicates of the pooled AUC.
        # We jackknife on the original samples (not on cohorts); cohort
        # stratification only governs how the bootstrap replicates are
        # drawn, not how skewness is estimated.
        n = len(y_true)
        jk = np.empty(n, dtype=float)
        for k in range(n):
            mask = np.ones(n, dtype=bool)
            mask[k] = False
            yt_k = y_true[mask]
            yp_k = y_prob[mask]
            if len(np.unique(yt_k)) < 2:
                jk[k] = point
            else:
                jk[k] = float(roc_auc_score(yt_k, yp_k))
        lo, hi = _bca_endpoints(point, aucs_arr, jk, alpha)

    return {
        "auc": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_kept": len(aucs),
        "alpha": float(alpha),
        "n": int(len(y_true)),
        "method": method,
    }


__all__ = ["delong_test", "bootstrap_pooled_ci"]
