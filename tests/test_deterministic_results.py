"""Determinism tests for the LODO pipeline.

Reproducibility is a non-negotiable requirement of the headline
benchmark: rerunning the pipeline with the same inputs and seeds must
produce bit-identical AUCs and per-sample probabilities. A silent
non-determinism (e.g. an unset RNG, a hash-randomised dict iteration in
``get_lodo_splits``, or an OMP / BLAS thread-order effect in
``RandomForestClassifier``) would cause every downstream CI and
p-value to drift between runs.

The existing tests check seed-determinism of ``bootstrap_auc_stratified``
in isolation. This file pins down end-to-end determinism of the full
``get_lodo_splits`` -> ``run_lodo_cv`` -> per-sample predictions stack.

Run with:
    pytest tests/test_deterministic_results.py -v
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (SCRIPTS_DIR, SRC_DIR):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import lodo_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset: deterministic given seed.
# ---------------------------------------------------------------------------

N_PER_COHORT = 30
N_FEATURES = 20


def _make_dataset(seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    feats = []
    for cohort, country in [("A", "X"), ("B", "X"), ("C", "Y"), ("D", "Z")]:
        labels = [0] * (N_PER_COHORT // 2) + [1] * (N_PER_COHORT // 2)
        rng.shuffle(labels)
        for label in labels:
            rows.append({
                "sample_id": f"{cohort}_{len(rows)}",
                "study_name": cohort,
                "country": country,
                "label": int(label),
            })
            v = rng.normal(loc=label * 1.0, scale=1.0, size=N_FEATURES)
            feats.append(v)
    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(
        feats,
        columns=[f"feat_{i}" for i in range(N_FEATURES)],
        index=metadata.index,
    )
    y = metadata["label"].copy()
    return X, y, metadata


class _SeededLogReg:
    """sklearn LogisticRegression wrapper with a fixed random_state.
    LogisticRegression with liblinear / lbfgs is deterministic under a
    fixed seed and is faster than RandomForest for this test."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self._inner = LogisticRegression(
            random_state=42, max_iter=2000, solver="lbfgs",
        )

    def fit(self, X, y):
        self._inner.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._inner.predict_proba(X)


def _model_fn():
    return _SeededLogReg()


# ---------------------------------------------------------------------------
# 1. get_lodo_splits is deterministic across calls.
# ---------------------------------------------------------------------------


def test_get_lodo_splits_is_deterministic_across_repeat_calls():
    """``get_lodo_splits`` must yield the same (cohort, train_idx,
    test_idx, excluded) tuples in the same order on every call. A subtle
    failure mode is using ``set(...)`` of cohort names without sorting,
    which would make order depend on Python's hash seed."""
    _, _, metadata = _make_dataset(seed=42)
    a = list(lodo_cv.get_lodo_splits(metadata, cohort_col="study_name",
                                     country_col="country"))
    b = list(lodo_cv.get_lodo_splits(metadata, cohort_col="study_name",
                                     country_col="country"))
    assert len(a) == len(b)
    for (cA, trA, teA, exA), (cB, trB, teB, exB) in zip(a, b):
        assert cA == cB
        assert trA == trB, f"train_idx differs for fold {cA}"
        assert teA == teB, f"test_idx differs for fold {cA}"
        assert exA == exB


# ---------------------------------------------------------------------------
# 2. run_lodo_cv produces bit-identical AUCs and per-sample probabilities
#    across two independent invocations.
# ---------------------------------------------------------------------------


def _run_once(tmp_path, seed):
    X, y, metadata = _make_dataset(seed=seed)
    pred_path = tmp_path / f"preds_{seed}.csv"
    results = lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col="country",
        save_predictions_path=str(pred_path),
    )
    preds = pd.read_csv(pred_path).sort_values("sample_id").reset_index(drop=True)
    return results, preds


def test_run_lodo_cv_is_bitwise_deterministic(tmp_path):
    """Two back-to-back runs of run_lodo_cv on the same synthetic data
    with the same seed must produce IDENTICAL per-fold AUCs and the
    same per-sample y_prob to full float precision."""
    res_a, preds_a = _run_once(tmp_path, seed=42)
    res_b, preds_b = _run_once(tmp_path, seed=42)

    # Per-fold AUCs identical to machine precision.
    assert res_a["cohort"] == res_b["cohort"]
    np.testing.assert_array_equal(
        np.asarray(res_a["auc"]), np.asarray(res_b["auc"])
    )
    assert res_a["mean_auc"] == res_b["mean_auc"]
    assert res_a["std_auc"] == res_b["std_auc"]
    # n_train / n_test must match exactly (split structure is identical).
    assert res_a["n_train"] == res_b["n_train"]
    assert res_a["n_test"] == res_b["n_test"]
    assert res_a["excluded_cohorts"] == res_b["excluded_cohorts"]

    # Per-sample predictions identical row-for-row.
    assert list(preds_a["sample_id"]) == list(preds_b["sample_id"])
    np.testing.assert_array_equal(
        preds_a["y_true"].values, preds_b["y_true"].values
    )
    np.testing.assert_array_equal(
        preds_a["y_prob"].values, preds_b["y_prob"].values
    )


# ---------------------------------------------------------------------------
# 3. Different seed -> different dataset -> different (but still
#    deterministic-per-seed) predictions. Sanity check that seed=42 above
#    is actually being used.
# ---------------------------------------------------------------------------


def test_run_lodo_cv_changes_with_different_seed(tmp_path):
    """If the dataset generator and model fit are wired correctly,
    seed=42 vs seed=99 must produce different per-sample probabilities.
    A test that only asserts equality could pass spuriously if both
    invocations were silently using the same memoised output."""
    _, preds_a = _run_once(tmp_path, seed=42)
    _, preds_c = _run_once(tmp_path, seed=99)
    # The per-sample probabilities should differ at least somewhere; if
    # they don't, the seed argument is being ignored.
    assert not np.array_equal(preds_a["y_prob"].values, preds_c["y_prob"].values)


# ---------------------------------------------------------------------------
# 4. Per-sample predictions never carry NaN / inf.
# ---------------------------------------------------------------------------


def test_run_lodo_cv_predictions_are_finite(tmp_path):
    """A robust pipeline must never emit NaN or inf predictions, even
    on the small synthetic dataset (a regression introducing log(0) or
    division by zero would surface here)."""
    _, preds = _run_once(tmp_path, seed=42)
    assert np.isfinite(preds["y_prob"].values).all()
    assert preds["y_prob"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# 5. Excluded-cohort bookkeeping is determined by country_col, not by call
#    order or by RNG state.
# ---------------------------------------------------------------------------


def test_excluded_cohorts_field_is_function_of_metadata_only():
    """Repeatedly calling get_lodo_splits with different
    sample_id orderings of the same metadata table must produce the same
    ``excluded`` sets per cohort. A subtle leakage would be reading the
    excluded set from ``cohort_country.items()`` dict order, which used
    to be insertion-order-dependent on older Pythons."""
    _, _, metadata = _make_dataset(seed=42)
    md_shuffled = metadata.sample(frac=1.0, random_state=7).reset_index(drop=True)

    a = {c: tuple(sorted(ex)) for c, _, _, ex in
         lodo_cv.get_lodo_splits(metadata, cohort_col="study_name",
                                 country_col="country")}
    b = {c: tuple(sorted(ex)) for c, _, _, ex in
         lodo_cv.get_lodo_splits(md_shuffled, cohort_col="study_name",
                                 country_col="country")}
    assert a == b
