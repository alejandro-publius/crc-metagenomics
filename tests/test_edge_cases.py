"""Edge-case tests for the crc_lodo_bench package entry points.

These exercise the validation wrapper layer in
``crc_lodo_bench.lodo`` (which fires before either the canonical
``scripts/lodo_cv.py`` or the vendored backend) and the per-fold filter
in ``crc_lodo_bench.filters``.

Run with:
    pytest tests/test_edge_cases.py -v
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

# Make src/ importable without `pip install -e .`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from crc_lodo_bench import (  # noqa: E402
    get_lodo_splits,
    per_fold_pathway_filter,
    run_lodo_cv,
)


# ---------------------------------------------------------------------------
# Small synthetic dataset used across the validation tests.
# ---------------------------------------------------------------------------


def _make_dataset(n_per: int = 20, n_features: int = 4, seed: int = 0):
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
            feats.append(rng.normal(loc=label * 0.5, size=n_features))
    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(
        feats,
        columns=[f"feat_{j}" for j in range(n_features)],
        index=metadata.index,
    )
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


def _model_fn():
    return _MeanScorer()


# ---------------------------------------------------------------------------
# 1. Single-class test fold: backend skips it, warning is emitted upstream.
# ---------------------------------------------------------------------------


def test_single_class_cohort_is_skipped_with_warning():
    """A cohort with only one label class must trigger a warning and be
    skipped by the LODO loop (roc_auc_score is undefined for one class)."""
    X, y, metadata = _make_dataset(n_per=20)
    # Force cohort_C to be single-class.
    mask = metadata["study_name"] == "cohort_C"
    metadata.loc[mask, "label"] = 0
    y = metadata["label"].copy()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = run_lodo_cv(_model_fn, X, y, metadata,
                              country_col="country")

    # cohort_C must be absent from the results (skipped, not crashed).
    assert "cohort_C" not in results["cohort"]
    assert len(results["cohort"]) == 2
    # And we should have warned about it.
    msgs = [str(item.message) for item in w]
    assert any("cohort_C" in m and "one label class" in m for m in msgs), \
        f"expected single-class warning for cohort_C, got: {msgs}"


# ---------------------------------------------------------------------------
# 2. Duplicate sample_ids -> ValueError.
# ---------------------------------------------------------------------------


def test_duplicate_sample_ids_raise():
    X, y, metadata = _make_dataset(n_per=10)
    metadata.loc[0, "sample_id"] = metadata.loc[1, "sample_id"]
    with pytest.raises(ValueError, match="sample_id.*not unique"):
        run_lodo_cv(_model_fn, X, y, metadata, country_col="country")


# ---------------------------------------------------------------------------
# 3. NaN in label column -> ValueError.
# ---------------------------------------------------------------------------


def test_nan_in_label_raises():
    X, y, metadata = _make_dataset(n_per=10)
    metadata.loc[0, "label"] = np.nan
    y = metadata["label"].copy()
    with pytest.raises(ValueError, match="NaN"):
        run_lodo_cv(_model_fn, X, y, metadata, country_col="country")


# ---------------------------------------------------------------------------
# 4. Misaligned X / y / metadata indices -> ValueError.
# ---------------------------------------------------------------------------


def test_misaligned_indices_raise():
    X, y, metadata = _make_dataset(n_per=10)
    # Shift X's index so it no longer matches metadata.
    X_bad = X.copy()
    X_bad.index = X.index + 100
    with pytest.raises(ValueError, match="index"):
        run_lodo_cv(_model_fn, X_bad, y, metadata, country_col="country")


# ---------------------------------------------------------------------------
# 5. per_fold_pathway_filter with 0 surviving features -> ValueError.
# ---------------------------------------------------------------------------


def test_per_fold_filter_zero_survivors_raises():
    n = 50
    # Both columns have prevalence well below 0.10 -> nothing survives.
    X = pd.DataFrame({
        "pw_a": np.concatenate([np.ones(2), np.zeros(n - 2)]),
        "pw_b": np.concatenate([np.ones(1), np.zeros(n - 1)]),
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_a", "pw_b"],
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
    )
    with pytest.raises(ValueError, match="0 of 2"):
        filt(X)


# ---------------------------------------------------------------------------
# 6. run_lodo_cv with missing country_col -> clear KeyError.
# ---------------------------------------------------------------------------


def test_missing_country_col_raises_keyerror():
    X, y, metadata = _make_dataset(n_per=10)
    with pytest.raises(KeyError, match="country_col.*not found"):
        run_lodo_cv(_model_fn, X, y, metadata, country_col="planet")


# ---------------------------------------------------------------------------
# 7. Empty feature matrix -> ValueError.
# ---------------------------------------------------------------------------


def test_empty_feature_matrix_raises():
    _, y, metadata = _make_dataset(n_per=10)
    X_empty = pd.DataFrame(index=metadata.index)
    with pytest.raises(ValueError, match="0 feature columns"):
        run_lodo_cv(_model_fn, X_empty, y, metadata, country_col="country")


# ---------------------------------------------------------------------------
# 8. Class-imbalance warning on extreme per-cohort skew.
# ---------------------------------------------------------------------------


def test_extreme_class_imbalance_warns():
    X, y, metadata = _make_dataset(n_per=20)
    # Make cohort_A 95% positive (1 negative out of 20).
    mask = metadata["study_name"] == "cohort_A"
    metadata.loc[mask, "label"] = 1
    metadata.loc[metadata[mask].index[0], "label"] = 0
    y = metadata["label"].copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_lodo_cv(_model_fn, X, y, metadata, country_col="country")
    msgs = [str(item.message) for item in w]
    assert any("cohort_A" in m and "minority class" in m for m in msgs), \
        f"expected imbalance warning, got: {msgs}"


# ---------------------------------------------------------------------------
# 9. Small-cohort warning under min_samples_per_cohort.
# ---------------------------------------------------------------------------


def test_small_cohort_warning():
    X, y, metadata = _make_dataset(n_per=20)
    # Trim cohort_C to 3 samples.
    drop_idx = metadata[metadata["study_name"] == "cohort_C"].index[3:]
    metadata = metadata.drop(drop_idx).reset_index(drop=True)
    X = X.drop(drop_idx).reset_index(drop=True)
    y = metadata["label"].copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Use the default threshold (10).
        get_list = list(get_lodo_splits(metadata, country_col="country"))
    msgs = [str(item.message) for item in w]
    assert any("cohort_C" in m and "min_samples_per_cohort" in m for m in msgs), \
        f"expected small-cohort warning, got: {msgs}"
    # And we still produced fold output (the warning is non-fatal).
    assert len(get_list) >= 1


# ---------------------------------------------------------------------------
# 10. NaN in country_col warns rather than silently grouping NaNs together.
# ---------------------------------------------------------------------------


def test_nan_in_country_col_warns():
    X, y, metadata = _make_dataset(n_per=15)
    metadata.loc[0, "country"] = np.nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(get_lodo_splits(metadata, country_col="country"))
    msgs = [str(item.message) for item in w]
    assert any("country" in m and "NaN" in m for m in msgs), \
        f"expected country-NaN warning, got: {msgs}"


# ---------------------------------------------------------------------------
# 11. per_fold_pathway_filter with neither filtered_cols nor passthrough_cols.
# ---------------------------------------------------------------------------


def test_filter_requires_some_cols():
    with pytest.raises(ValueError, match="non-empty"):
        per_fold_pathway_filter(filtered_cols=[])


# ---------------------------------------------------------------------------
# 12. per_fold_pathway_filter with passthrough only (filtered_cols empty
#     but passthrough non-empty) is legal and returns passthrough as-is.
# ---------------------------------------------------------------------------


def test_filter_passthrough_only_is_legal():
    X = pd.DataFrame({"sp_a": np.ones(5), "sp_b": np.ones(5)})
    filt = per_fold_pathway_filter(
        filtered_cols=[],
        passthrough_cols=["sp_a", "sp_b"],
    )
    assert filt(X) == ["sp_a", "sp_b"]


# ---------------------------------------------------------------------------
# 13. per_fold_pathway_filter warns when every filtered_col passes.
# ---------------------------------------------------------------------------


def test_filter_all_pass_emits_noop_warning():
    n = 50
    X = pd.DataFrame({
        "pw_a": np.ones(n),
        "pw_b": np.ones(n),
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_a", "pw_b"],
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kept = filt(X)
    assert kept == ["pw_a", "pw_b"]
    assert any("no-op" in str(item.message) for item in w)
