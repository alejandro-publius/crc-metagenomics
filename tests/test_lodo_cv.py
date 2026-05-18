"""Unit tests for scripts.lodo_cv.run_lodo_cv.

These tests use a tiny synthetic dataset (3 cohorts, two sharing a country)
to verify three contract properties of the country-aware LODO routine:

  1. country_col exclusion: when a cohort is tested, every cohort from the
     same country is excluded from the training set.
  2. feature_filter_fn is called once per fold and only ever sees training
     samples (no test-fold leakage).
  3. The results dict records per-fold AUCs and per-fold n_features.

Stubs are marked with @pytest.mark.skip where assertions need to be
hardened against the project's eventual feature-filter contract; the test
functions themselves import cleanly so pytest collection always succeeds.

Run with:
    pytest tests/test_lodo_cv.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make `import lodo_cv` work without installing the project as a package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import lodo_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic 3-cohort dataset.
#   - cohort_A and cohort_B share country "X"
#   - cohort_C is in country "Y"
# Each cohort gets 20 samples (10 cases, 10 controls) so per-fold AUC is
# defined and reproducible.
# ---------------------------------------------------------------------------

RNG_SEED = 0
N_PER_COHORT = 20
N_FEATURES = 5


def _make_synthetic():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    feats = []
    for cohort, country in [("cohort_A", "X"),
                            ("cohort_B", "X"),
                            ("cohort_C", "Y")]:
        for i in range(N_PER_COHORT):
            label = i % 2  # 10 cases, 10 controls per cohort
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "study_name": cohort,
                "country": country,
                "label": label,
            })
            # Give the cases a mild signal so AUC > 0.5 is feasible.
            feats.append(rng.normal(loc=label * 0.5, scale=1.0,
                                    size=N_FEATURES))
    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(feats,
                     columns=[f"feat_{j}" for j in range(N_FEATURES)],
                     index=metadata.index)
    y = metadata["label"].copy()
    return X, y, metadata


class _ConstantModel:
    """Deterministic stand-in for any sklearn estimator with predict_proba."""

    def fit(self, X, y):
        # Use mean feature_0 of positives vs negatives as a 1-D scorer.
        pos = X[y.values == 1] if hasattr(y, "values") else X[y == 1]
        neg = X[y.values == 0] if hasattr(y, "values") else X[y == 0]
        self.threshold_ = (pos.iloc[:, 0].mean() + neg.iloc[:, 0].mean()) / 2.0
        return self

    def predict_proba(self, X):
        scores = (X.iloc[:, 0].values - self.threshold_)
        # Squash to [0, 1] without depending on scipy.
        probs = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])


def _model_fn():
    return _ConstantModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_country_col_excludes_same_country_cohorts():
    """When cohort_A is held out, cohort_B (same country X) must NOT be in
    train; cohort_C (country Y) must be in train. Symmetric for cohort_B.
    cohort_C as test should train on both cohort_A and cohort_B."""
    _, _, metadata = _make_synthetic()
    splits = list(lodo_cv.get_lodo_splits(
        metadata, label_col="label", cohort_col="study_name",
        country_col="country"))

    by_cohort = {cohort: (train_idx, test_idx, excluded)
                 for cohort, train_idx, test_idx, excluded in splits}

    # All three cohorts produced a fold.
    assert set(by_cohort) == {"cohort_A", "cohort_B", "cohort_C"}

    # cohort_A tested -> cohort_B excluded from training, cohort_C present.
    train_idx, _, excluded = by_cohort["cohort_A"]
    train_cohorts = set(metadata.loc[train_idx, "study_name"])
    assert "cohort_A" not in train_cohorts
    assert "cohort_B" not in train_cohorts, \
        "same-country cohort must be excluded from training"
    assert "cohort_C" in train_cohorts
    assert excluded == {"cohort_B"}

    # Symmetric check for cohort_B as test.
    train_idx, _, excluded = by_cohort["cohort_B"]
    train_cohorts = set(metadata.loc[train_idx, "study_name"])
    assert train_cohorts == {"cohort_C"}
    assert excluded == {"cohort_A"}

    # cohort_C is alone in country Y -> trains on both A and B.
    train_idx, _, excluded = by_cohort["cohort_C"]
    train_cohorts = set(metadata.loc[train_idx, "study_name"])
    assert train_cohorts == {"cohort_A", "cohort_B"}
    assert excluded == set()


def test_feature_filter_called_once_per_fold_with_train_only():
    """feature_filter_fn must be invoked exactly once per LODO fold and
    must receive ONLY training samples (never the held-out test cohort's
    rows). We pair each filter call with its fold via the split iterator
    so the leakage check is per-fold, not global."""
    X, y, metadata = _make_synthetic()

    calls = []

    def feature_filter_fn(X_train):
        calls.append(set(X_train.index))
        return list(X_train.columns[:3])

    lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        feature_filter_fn=feature_filter_fn,
        country_col="country",
    )

    # 3 cohorts -> 3 folds -> 3 calls (one per fold).
    assert len(calls) == 3, f"expected 3 filter calls, got {len(calls)}"

    # Reconstruct the fold order so we can pair each call to its test set
    # and assert the call's training indices are disjoint from that fold's
    # held-out test indices.
    fold_order = [
        (cohort, set(test_idx))
        for cohort, _train_idx, test_idx, _excl
        in lodo_cv.get_lodo_splits(metadata, label_col="label",
                                   cohort_col="study_name",
                                   country_col="country")
    ]
    assert len(fold_order) == len(calls)
    for (cohort, test_idx), train_indices in zip(fold_order, calls):
        leaked = test_idx & train_indices
        assert not leaked, (
            f"feature_filter_fn for fold {cohort!r} received "
            f"{len(leaked)} test-cohort rows"
        )


def test_results_records_per_fold_aucs_and_n_features():
    """The results dict must carry per-fold AUCs (one per cohort), per-fold
    n_features (reflecting the filter), and mean/std summaries."""
    X, y, metadata = _make_synthetic()

    def feature_filter_fn(X_train):
        return list(X_train.columns[:3])  # always keep exactly 3 features

    results = lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        feature_filter_fn=feature_filter_fn,
        country_col="country",
    )

    # AUCs: one per fold, all finite, in [0, 1].
    assert "auc" in results
    assert len(results["auc"]) == 3
    for auc in results["auc"]:
        assert 0.0 <= auc <= 1.0
        assert np.isfinite(auc)

    # n_features recorded per fold and matches the filter output.
    assert "n_features" in results
    assert len(results["n_features"]) == 3
    assert all(n == 3 for n in results["n_features"])

    # Summary stats present.
    assert "mean_auc" in results
    assert "std_auc" in results
    assert np.isfinite(results["mean_auc"])
    assert np.isfinite(results["std_auc"])


@pytest.mark.skip(reason="stub - implement once n_features contract for the "
                          "no-filter path is finalized")
def test_n_features_without_filter_equals_input_width():
    """When feature_filter_fn is None, n_features per fold should equal
    X.shape[1]. Stub left for the contract to be locked down."""
    pass
