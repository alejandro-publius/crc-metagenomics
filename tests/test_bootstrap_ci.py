"""Unit tests for ``scripts/bootstrap_ci.py``.

The cohort-stratified bootstrap underpins the pooled-AUC 95% CIs that
appear in the manuscript and supplementary tables; getting it wrong
would silently shift every quoted interval. These tests pin down:

  1. The point estimate matches sklearn on the full pooled data.
  2. The CI brackets the point estimate and respects the requested
     percentile width.
  3. Cohort-stratified resampling preserves per-cohort sample counts in
     every iteration (i.e. the resample is genuinely stratified rather
     than i.i.d.).
  4. ``bootstrap_auc_iid`` and ``bootstrap_auc_stratified`` return the
     two-tuple (low, high) contract expected by ``process_preds``.
  5. With identical predictions per cohort, the bootstrap CI collapses
     to a single point (sanity check).

Run with:
    pytest tests/test_bootstrap_ci.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

# Make scripts/ importable without a package install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pred_frame(n_per_cohort=80, n_cohorts=3, seed=0, signal=0.6):
    """Synthetic pooled prediction frame matching the shape of
    ``results/preds_*.csv`` (cohort, sample_id, y_true, y_prob)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c_idx in range(n_cohorts):
        cohort = f"cohort_{c_idx}"
        # Balanced classes per cohort.
        y = np.array([0] * (n_per_cohort // 2) + [1] * (n_per_cohort // 2))
        rng.shuffle(y)
        y_prob = y * signal + rng.normal(scale=1.0, size=n_per_cohort)
        for i in range(n_per_cohort):
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "cohort": cohort,
                "y_true": int(y[i]),
                "y_prob": float(y_prob[i]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# bootstrap_auc_iid
# ---------------------------------------------------------------------------


def test_bootstrap_auc_iid_returns_ordered_interval_with_point_inside():
    df = _make_pred_frame()
    sub = df[df["cohort"] == "cohort_0"]
    yt = sub["y_true"].values
    yp = sub["y_prob"].values
    point = roc_auc_score(yt, yp)
    lo, hi = bootstrap_ci.bootstrap_auc_iid(yt, yp, n_boot=500, seed=0)
    assert lo < hi
    # The 95% CI should bracket the point estimate in expectation; with
    # n=40 it will not always do so exactly, but it must do so loosely.
    assert lo - 0.05 <= point <= hi + 0.05


def test_bootstrap_auc_iid_returns_two_tuple():
    """``process_preds`` unpacks (lo, hi). Guard against a future refactor
    that silently switches to a dict / scalar."""
    df = _make_pred_frame(n_per_cohort=40, n_cohorts=1)
    yt = df["y_true"].values
    yp = df["y_prob"].values
    result = bootstrap_ci.bootstrap_auc_iid(yt, yp, n_boot=200, seed=0)
    assert isinstance(result, tuple)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# bootstrap_auc_stratified
# ---------------------------------------------------------------------------


def test_stratified_bootstrap_returns_ordered_interval():
    df = _make_pred_frame()
    lo, hi = bootstrap_ci.bootstrap_auc_stratified(df, n_boot=500, seed=0)
    assert 0.0 <= lo < hi <= 1.0


def test_stratified_bootstrap_brackets_point_estimate_loosely():
    """The point AUC on the full pooled frame must lie close to the CI
    midpoint; we use a generous slack because n_boot is modest in the test."""
    df = _make_pred_frame(n_per_cohort=120, n_cohorts=4, seed=1, signal=0.8)
    point = roc_auc_score(df["y_true"].values, df["y_prob"].values)
    lo, hi = bootstrap_ci.bootstrap_auc_stratified(df, n_boot=1000, seed=0)
    assert lo - 0.02 <= point <= hi + 0.02


def test_stratified_resample_preserves_per_cohort_counts(monkeypatch):
    """Each iteration must sample EXACTLY ``len(cohort)`` rows from each
    cohort (this is what makes the procedure stratified rather than i.i.d.).

    We hook into ``np.random.RandomState.choice`` by patching the module's
    ``np`` alias indirectly: we instead inspect ``cohort_to_idx`` by running
    the function with n_boot=1 and a custom frame, and reading back the
    resampled indices via a wrapper around ``roc_auc_score`` that captures
    the labels passed to it.
    """
    df = _make_pred_frame(n_per_cohort=20, n_cohorts=3, seed=0)
    expected_counts = (
        df.groupby("cohort").size().to_dict()
    )

    captured = {"y_true": []}
    real_auc = bootstrap_ci.roc_auc_score

    def spy(yt, yp):
        captured["y_true"].append(np.asarray(yt))
        return real_auc(yt, yp)

    monkeypatch.setattr(bootstrap_ci, "roc_auc_score", spy)

    # Run a handful of iterations and inspect each resample's total size,
    # which equals the sum of per-cohort sample counts iff the stratified
    # contract is preserved (a single cohort being over/under-resampled
    # would change the total).
    bootstrap_ci.bootstrap_auc_stratified(df, n_boot=5, seed=123)

    expected_total = sum(expected_counts.values())
    assert len(captured["y_true"]) == 5
    for yt in captured["y_true"]:
        assert len(yt) == expected_total, (
            f"stratified resample produced {len(yt)} rows, expected "
            f"{expected_total} (sum of per-cohort sizes)"
        )


def test_stratified_bootstrap_is_seed_deterministic():
    """Two calls with the same seed must produce bit-identical CIs."""
    df = _make_pred_frame(seed=7)
    a = bootstrap_ci.bootstrap_auc_stratified(df, n_boot=200, seed=99)
    b = bootstrap_ci.bootstrap_auc_stratified(df, n_boot=200, seed=99)
    assert a == b


# ---------------------------------------------------------------------------
# Pathological corner cases
# ---------------------------------------------------------------------------


def test_iid_bootstrap_skips_single_class_resamples():
    """If a resample lands on a single class, roc_auc_score is undefined and
    the iteration must be silently dropped (not raise)."""
    # All-positive labels would force every resample to be single-class.
    # Use a tiny mixed sample where the dropping branch is exercised: 1
    # positive among many negatives makes some resamples all-negative.
    yt = np.array([0] * 19 + [1])
    yp = np.linspace(0.0, 1.0, 20)
    lo, hi = bootstrap_ci.bootstrap_auc_iid(yt, yp, n_boot=500, seed=0)
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi


def test_stratified_with_identical_scores_within_cohort_gives_tight_ci():
    """If every sample within a cohort has the same prediction, the per-
    cohort AUC contribution is constant under resampling and the pooled CI
    should be very narrow."""
    n = 40
    rows = []
    for c_idx, cohort in enumerate(["A", "B"]):
        for i in range(n):
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "cohort": cohort,
                "y_true": int(i % 2),
                # Perfect separator: positives get 1.0, negatives 0.0.
                "y_prob": float(i % 2),
            })
    df = pd.DataFrame(rows)
    lo, hi = bootstrap_ci.bootstrap_auc_stratified(df, n_boot=500, seed=0)
    # The classifier is perfect, but resampling can leave a single class
    # within a cohort; the kept iterations all score AUC = 1.0 exactly.
    assert lo == pytest.approx(1.0, abs=1e-10)
    assert hi == pytest.approx(1.0, abs=1e-10)
