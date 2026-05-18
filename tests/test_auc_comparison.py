"""Unit tests for ``scripts/auc_comparison.py``.

Validates the DeLong paired-AUC test implementation in two ways:

1. ``delong_roc_test`` returns the same AUC point estimates as
   ``sklearn.metrics.roc_auc_score`` (which is the canonical
   Mann-Whitney-U based AUC and serves as ground truth for the AUC
   component independent of the variance algorithm).
2. The reported (z, p) pair satisfies the DeLong z = (AUC_a - AUC_b) /
   sqrt(var_diff) identity and the two-sided normal p-value identity,
   on a synthetic 200-sample paired-classifier dataset whose ground-
   truth covariance we recompute from scratch with the public Sun & Xu
   (2014) algorithm. This guards against regressions in the rank /
   covariance arithmetic that the smoke test in
   ``tests/test_package_import.py`` cannot catch.

We also test:
    - the identical-predictions corner case (var_diff == 0 -> z=0, p=1);
    - that the function rejects a single-class label vector;
    - that boot_ci returns a (lo, hi) pair bracketing the sample mean;
    - that compare() outputs the documented summary keys.

Run with:
    pytest tests/test_auc_comparison.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# Make scripts/ importable without a package install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import auc_comparison  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic ground-truth fixture: two correlated classifiers on the same 200
# paired samples. We deliberately use a moderate sample size (n=200) so the
# Sun & Xu variance estimate is stable enough to compare with floating-point
# tolerance against an independent recomputation.
# ---------------------------------------------------------------------------


def _make_paired_scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    shared_signal = y + rng.normal(scale=0.6, size=n)
    a = shared_signal + rng.normal(scale=0.2, size=n)
    b = shared_signal + rng.normal(scale=0.4, size=n)
    return y, a, b


def _delong_components(y_true, y_prob):
    """Recompute Sun & Xu (2014) per-positive / per-negative score
    components from scratch as an independent reference implementation."""
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    x = np.asarray(y_prob, dtype=float)[pos]
    z = np.asarray(y_prob, dtype=float)[neg]

    # V10[i] = P(X_i > Y) + 0.5 P(X_i = Y)  for each positive X_i.
    # V01[j] = P(X > Y_j) + 0.5 P(X > Y_j)  symmetrically.
    v10 = np.empty(m)
    for i, xi in enumerate(x):
        v10[i] = (np.sum(xi > z) + 0.5 * np.sum(xi == z)) / n
    v01 = np.empty(n)
    for j, zj in enumerate(z):
        v01[j] = (np.sum(x > zj) + 0.5 * np.sum(x == zj)) / m
    return v10, v01


# ---------------------------------------------------------------------------
# Tests for delong_roc_test
# ---------------------------------------------------------------------------


def test_delong_aucs_match_sklearn():
    """AUC point estimates from delong_roc_test must equal sklearn's."""
    y, a, b = _make_paired_scores()
    auc_a, auc_b, _, _ = auc_comparison.delong_roc_test(y, a, b)
    assert auc_a == pytest.approx(roc_auc_score(y, a), abs=1e-10)
    assert auc_b == pytest.approx(roc_auc_score(y, b), abs=1e-10)


def test_delong_z_and_p_match_reference_recomputation():
    """The reported z and p must match an independent Sun & Xu recomputation
    of the variance term, on the same paired data."""
    y, a, b = _make_paired_scores()
    auc_a, auc_b, z, p = auc_comparison.delong_roc_test(y, a, b)

    v10_a, v01_a = _delong_components(y, a)
    v10_b, v01_b = _delong_components(y, b)
    m = int((y == 1).sum())
    n = int((y == 0).sum())

    S10 = np.cov(np.vstack([v10_a, v10_b]))
    S01 = np.cov(np.vstack([v01_a, v01_b]))
    S = S10 / m + S01 / n
    var_diff_ref = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z_ref = (auc_a - auc_b) / np.sqrt(var_diff_ref)
    p_ref = 2.0 * (1.0 - norm.cdf(abs(z_ref)))

    # The implementation should agree with this reference to ~6 decimals.
    assert z == pytest.approx(z_ref, abs=1e-6)
    assert p == pytest.approx(p_ref, abs=1e-6)


def test_delong_two_sided_p_is_consistent_with_z():
    """Two-tailed p = 2 * (1 - Phi(|z|)) by construction; if this drifts the
    significance numbers in the manuscript will silently change."""
    y, a, b = _make_paired_scores(n=300, seed=2)
    _, _, z, p = auc_comparison.delong_roc_test(y, a, b)
    expected = 2.0 * (1.0 - norm.cdf(abs(z)))
    assert p == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= p <= 1.0


def test_delong_identical_predictions_returns_z0_p1():
    """If both classifiers produce the same scores, the variance of the
    difference is zero and the documented contract is (z=0, p=1)."""
    y, a, _ = _make_paired_scores(n=100, seed=3)
    auc_a, auc_b, z, p = auc_comparison.delong_roc_test(y, a, a)
    assert auc_a == pytest.approx(auc_b, abs=1e-12)
    assert z == 0.0
    assert p == 1.0


def test_delong_rejects_single_class_y_true():
    """The DeLong test is undefined without both classes present."""
    y = np.zeros(50, dtype=int)
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    with pytest.raises(ValueError):
        auc_comparison.delong_roc_test(y, a, b)


def test_midrank_matches_average_rank_method():
    """The internal ``_midrank`` must agree with ``scipy.stats.rankdata``
    using the 'average' method (which is the textbook mid-rank with tie
    averaging)."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(42)
    # Mix of unique and tied values to exercise the tie branch.
    x = np.concatenate([rng.normal(size=50), np.zeros(5), np.ones(5)])
    rng.shuffle(x)
    got = auc_comparison._midrank(x)
    expected = rankdata(x, method="average")
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# Tests for boot_ci and compare
# ---------------------------------------------------------------------------


def test_boot_ci_brackets_sample_mean_and_returns_ordered_pair():
    diffs = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.015, -0.005, 0.02])
    lo, hi = auc_comparison.boot_ci(diffs, n=2000, seed=0)
    assert lo <= diffs.mean() <= hi
    assert lo < hi


def test_compare_returns_documented_summary_keys():
    """``compare`` returns a row dict whose keys are consumed downstream by
    the model_comparison.csv writer; this guards against silent schema drift."""
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.80, scale=0.02, size=10)
    b = rng.normal(loc=0.78, scale=0.02, size=10)
    row = auc_comparison.compare(pd.Series(a), pd.Series(b), label="A vs B")
    for key in (
        "comparison", "mean_a", "mean_b", "mean_diff",
        "ci_low", "ci_high",
        "t_stat", "t_pvalue",
        "wilcoxon_stat", "wilcoxon_pvalue",
        "n_folds",
    ):
        assert key in row, f"compare() missing key {key!r}"
    assert row["n_folds"] == 10
    assert row["comparison"] == "A vs B"
    assert np.isfinite(row["t_stat"])
    assert 0.0 <= row["t_pvalue"] <= 1.0
    assert 0.0 <= row["wilcoxon_pvalue"] <= 1.0
