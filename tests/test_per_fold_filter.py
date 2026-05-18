"""Unit tests for ``crc_lodo_bench.filters.per_fold_pathway_filter``.

The per-fold filter is the mechanism that prevents test-fold leakage in
prevalence- and mean-abundance-based feature selection: if it ever sees
test-cohort rows, the headline LODO AUCs are no longer leakage-free.

These tests pin down four properties:

  1. **Threshold semantics**: a column at the prevalence threshold is
     kept; below it, dropped. Same for the mean-abundance threshold.
  2. **Passthrough ordering**: passthrough columns are emitted *first*
     in the returned list and always retained.
  3. **Per-fold contract via run_lodo_cv**: the filter is called once
     per fold and only receives the training rows of that fold.
  4. **Defensive validation**: invalid threshold inputs and missing
     filtered_cols raise clear errors.

Run with:
    pytest tests/test_per_fold_filter.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make src/ importable as a package without `pip install -e .`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from crc_lodo_bench.filters import per_fold_pathway_filter  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Threshold semantics
# ---------------------------------------------------------------------------


def test_prevalence_threshold_inclusive():
    """A column whose prevalence equals the threshold must be kept."""
    n = 100
    # exactly 10/100 non-zero -> prevalence = 0.10
    col_at = np.zeros(n)
    col_at[:10] = 1.0
    # 9/100 non-zero -> prevalence = 0.09, below the threshold.
    col_below = np.zeros(n)
    col_below[:9] = 1.0
    # 50/100 non-zero -> well above threshold.
    col_above = np.zeros(n)
    col_above[:50] = 1.0

    X = pd.DataFrame({
        "pw_at": col_at,
        "pw_below": col_below,
        "pw_above": col_above,
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_at", "pw_below", "pw_above"],
        prevalence_threshold=0.10,
        mean_threshold=0.0,  # disable mean filter
    )
    kept = filt(X)
    assert "pw_at" in kept
    assert "pw_below" not in kept
    assert "pw_above" in kept


def test_mean_threshold_inclusive():
    """A column whose mean equals the threshold must be kept."""
    n = 100
    # mean exactly equal to threshold.
    col_at = np.full(n, 1e-6)
    # mean strictly below.
    col_below = np.full(n, 5e-7)
    # mean well above.
    col_above = np.full(n, 1e-3)

    X = pd.DataFrame({
        "pw_at": col_at,
        "pw_below": col_below,
        "pw_above": col_above,
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_at", "pw_below", "pw_above"],
        prevalence_threshold=0.0,  # disable prevalence filter
        mean_threshold=1e-6,
    )
    kept = filt(X)
    assert "pw_at" in kept
    assert "pw_below" not in kept
    assert "pw_above" in kept


def test_both_thresholds_must_pass():
    """A column must clear BOTH thresholds to survive."""
    n = 100
    rows = {
        # high prevalence, high mean -> kept.
        "pw_high_both": np.full(n, 0.5),
        # high prevalence, mean below threshold -> dropped.
        "pw_low_mean": np.full(n, 1e-9),
        # low prevalence, high mean (rare but large) -> dropped.
        "pw_low_prev": np.concatenate([np.full(5, 100.0), np.zeros(95)]),
    }
    X = pd.DataFrame(rows)
    filt = per_fold_pathway_filter(
        filtered_cols=list(rows.keys()),
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
    )
    kept = filt(X)
    assert kept == ["pw_high_both"]


# ---------------------------------------------------------------------------
# 2. Passthrough ordering and retention
# ---------------------------------------------------------------------------


def test_passthrough_columns_emitted_first_in_order():
    n = 50
    X = pd.DataFrame({
        "sp_b": np.ones(n),  # passthrough
        "sp_a": np.ones(n),  # passthrough
        "pw_kept": np.ones(n),
        "pw_dropped": np.zeros(n),
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_kept", "pw_dropped"],
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
        passthrough_cols=["sp_b", "sp_a"],  # NB: deliberately not sorted
    )
    kept = filt(X)
    # Passthrough order must be preserved exactly as given.
    assert kept[:2] == ["sp_b", "sp_a"]
    assert "pw_kept" in kept[2:]
    assert "pw_dropped" not in kept


def test_passthrough_retained_even_if_constant_zero():
    """Passthrough columns must NEVER be subject to the thresholds; even an
    all-zero passthrough column is retained."""
    n = 20
    X = pd.DataFrame({
        "sp_zero": np.zeros(n),
        "pw_x": np.ones(n),
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_x"],
        passthrough_cols=["sp_zero"],
    )
    kept = filt(X)
    assert "sp_zero" in kept


# ---------------------------------------------------------------------------
# 3. Per-fold contract: the filter must see ONLY training rows.
# ---------------------------------------------------------------------------


def _make_3cohort_dataset(seed=0, n_per=30):
    rng = np.random.default_rng(seed)
    rows = []
    feats = []
    for cohort, country in [("cohort_A", "X"),
                            ("cohort_B", "Y"),
                            ("cohort_C", "Z")]:
        for i in range(n_per):
            label = i % 2
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "study_name": cohort,
                "country": country,
                "label": label,
            })
            feats.append(rng.normal(loc=label * 0.5, scale=1.0, size=4))
    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(feats,
                     columns=[f"feat_{j}" for j in range(4)],
                     index=metadata.index)
    y = metadata["label"].copy()
    return X, y, metadata


class _MeanScorer:
    def fit(self, X, y):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        self.pos_ = X[y_arr == 1].mean(axis=0).values
        self.neg_ = X[y_arr == 0].mean(axis=0).values
        return self

    def predict_proba(self, X):
        d_pos = np.linalg.norm(X.values - self.pos_, axis=1)
        d_neg = np.linalg.norm(X.values - self.neg_, axis=1)
        s = d_neg - d_pos
        p = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p, p])


def test_filter_only_sees_training_rows_in_run_lodo_cv():
    """Wire the filter into run_lodo_cv and verify that each call's input
    DataFrame is disjoint from the held-out test rows for that fold."""
    # Import lazily so we exercise the canonical path on the actual repo.
    from crc_lodo_bench import run_lodo_cv, get_lodo_splits

    X, y, metadata = _make_3cohort_dataset()
    test_rows_by_fold = {
        cohort: set(test_idx)
        for cohort, _, test_idx, _ in get_lodo_splits(
            metadata, label_col="label",
            cohort_col="study_name", country_col="country",
        )
    }

    seen_rows = []

    def spying_filter(X_train):
        seen_rows.append(set(X_train.index))
        return list(X_train.columns)

    run_lodo_cv(
        _MeanScorer, X, y, metadata,
        cohort_col="study_name",
        country_col="country",
        feature_filter_fn=spying_filter,
    )

    # 3 cohorts -> 3 folds.
    assert len(seen_rows) == 3
    # Each call's training rows must be disjoint from at least one fold's
    # held-out test rows (specifically: the fold the call corresponds to).
    test_sets = list(test_rows_by_fold.values())
    for call_rows in seen_rows:
        matched = sum(1 for t in test_sets if call_rows.isdisjoint(t))
        # The call must be disjoint from at least its own fold's test rows.
        assert matched >= 1, (
            "filter call leaked test-cohort rows into the training fold"
        )


# ---------------------------------------------------------------------------
# 4. Defensive validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.1, 1.5])
def test_invalid_prevalence_threshold_raises(bad):
    with pytest.raises(ValueError):
        per_fold_pathway_filter(filtered_cols=["a"], prevalence_threshold=bad)


def test_negative_mean_threshold_raises():
    with pytest.raises(ValueError):
        per_fold_pathway_filter(filtered_cols=["a"], mean_threshold=-1e-6)


def test_missing_filtered_cols_raise_keyerror():
    """If the filter is given a column name that the per-fold X_train does
    not have, raise KeyError so the caller notices rather than silently
    dropping the column."""
    X = pd.DataFrame({"present": np.ones(10)})
    filt = per_fold_pathway_filter(filtered_cols=["present", "absent"])
    with pytest.raises(KeyError):
        filt(X)


def test_callable_is_idempotent_on_repeat_calls():
    """Calling the same filter twice on the same X must return the same
    column list (no hidden state)."""
    n = 80
    X = pd.DataFrame({
        f"pw_{i}": np.where(np.arange(n) < 20, 1.0, 0.0)
        for i in range(5)
    })
    filt = per_fold_pathway_filter(
        filtered_cols=list(X.columns),
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
    )
    a = filt(X)
    b = filt(X)
    assert a == b
