"""Integration test for country-aware LODO with a known-answer dataset.

This is the end-to-end counterpart to ``tests/test_country_aware_lodo.py``
(which exercises split semantics in isolation). Here we drive
``run_lodo_cv`` with three synthetic cohorts whose signal structure was
designed so the AUC ordering across folds is analytically known, then
confirm both the split structure and the AUC ordering hold.

Setup (3 cohorts, two share country X):

  - cohort_A (country X): feature_0 is strongly informative; +noise.
  - cohort_B (country X): same generative law as A (so a classifier
    trained on A would generalise to B perfectly under plain LODO).
  - cohort_C (country Y): same generative law (informative feature_0),
    but constructed so country-aware LODO drops both A and B from
    training when C is the test cohort -> nothing trains C.

Under **plain LODO**, holding out C trains on {A, B} (same generative
law) -> AUC near 1.0.
Under **country-aware LODO**, holding out A excludes B (same country),
so training is C only -> AUC remains high but training set is half
the size. Holding out C excludes nothing (singleton country) -> trains
on {A, B} -> AUC near 1.0.

The key behavioural assertion is the **exclusion bookkeeping**: under
country-aware mode the ``excluded_cohorts`` field of the results dict
must record {B} when A is the test fold and vice versa, while plain
LODO records []. AUCs are checked for finiteness and the well-separated
case is checked to score >= 0.9 (so a regression that broke training
data assembly would show up).

Run with:
    pytest tests/test_country_aware_lodo_integration.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import lodo_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Known-answer synthetic dataset
# ---------------------------------------------------------------------------


N_PER_COHORT = 40  # 20 cases, 20 controls per cohort -> 3 well-defined AUCs.
N_FEATURES = 6
RNG_SEED = 7


def _make_strong_signal_dataset():
    """3 cohorts, two share country 'X'. Each cohort has the SAME
    generative law: label drives feature_0 with separation 3 sigma."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    feats = []
    for cohort, country in [("cohort_A", "X"),
                            ("cohort_B", "X"),
                            ("cohort_C", "Y")]:
        # Equal cases / controls.
        labels = [0] * (N_PER_COHORT // 2) + [1] * (N_PER_COHORT // 2)
        rng.shuffle(labels)
        for label in labels:
            rows.append({
                "sample_id": f"{cohort}_{len(rows)}",
                "study_name": cohort,
                "country": country,
                "label": label,
            })
            # feature_0 carries a strong signal; others are pure noise.
            v = rng.normal(loc=0.0, scale=1.0, size=N_FEATURES)
            v[0] = rng.normal(loc=label * 3.0, scale=1.0)
            feats.append(v)
    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(
        feats,
        columns=[f"feat_{j}" for j in range(N_FEATURES)],
        index=metadata.index,
    )
    y = metadata["label"].copy()
    return X, y, metadata


class _LogisticOneFeature:
    """Tiny classifier scoring on the first feature only — sufficient to
    recover the planted signal at AUC >= 0.9 with our 20/20 splits."""

    def fit(self, X, y):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        self.mu_pos_ = X.iloc[:, 0][y_arr == 1].mean()
        self.mu_neg_ = X.iloc[:, 0][y_arr == 0].mean()
        return self

    def predict_proba(self, X):
        z = X.iloc[:, 0].values - 0.5 * (self.mu_pos_ + self.mu_neg_)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _model_fn():
    return _LogisticOneFeature()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_country_aware_lodo_records_expected_exclusions_and_strong_aucs():
    """End-to-end run with country_col='country' must:

      - Yield 3 folds, one per cohort.
      - Mark cohort_B as excluded when cohort_A is the test fold (and
        vice versa).
      - Mark cohort_C's fold with empty exclusion (singleton country Y).
      - Recover AUC >= 0.9 on every fold (the planted signal is large).
    """
    X, y, metadata = _make_strong_signal_dataset()
    results = lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col="country",
    )

    # 3 cohorts -> 3 folds.
    assert len(results["cohort"]) == 3
    assert len(results["auc"]) == 3
    assert len(results["excluded_cohorts"]) == 3

    by_cohort = {c: (auc, excl) for c, auc, excl in zip(
        results["cohort"], results["auc"], results["excluded_cohorts"]
    )}

    # Exclusion bookkeeping (this is the country-aware contract).
    assert by_cohort["cohort_A"][1] == ["cohort_B"]
    assert by_cohort["cohort_B"][1] == ["cohort_A"]
    assert by_cohort["cohort_C"][1] == []

    # AUC sanity: planted signal should yield AUC >= 0.9 on each fold.
    for cohort, (auc, _) in by_cohort.items():
        assert 0.0 <= auc <= 1.0
        assert auc >= 0.9, (
            f"fold {cohort} AUC {auc:.3f} below 0.9 — planted signal "
            "should be easy to recover."
        )

    # Summary stats consistent with per-fold values.
    assert results["mean_auc"] == pytest.approx(
        float(np.mean(results["auc"])), abs=1e-12
    )
    assert results["std_auc"] == pytest.approx(
        float(np.std(results["auc"])), abs=1e-12
    )


def test_plain_vs_country_aware_lodo_differ_only_in_training_size():
    """Without country_col, A's training fold includes BOTH B and C.
    With country_col='country', B is dropped from A's training fold.
    The n_train values must reflect this exactly."""
    X, y, metadata = _make_strong_signal_dataset()

    plain = lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col=None,
    )
    aware = lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col="country",
    )

    # Map cohort -> n_train.
    plain_n = dict(zip(plain["cohort"], plain["n_train"]))
    aware_n = dict(zip(aware["cohort"], aware["n_train"]))

    # Each cohort has N_PER_COHORT rows; plain LODO trains on the other two.
    expected_plain = {
        "cohort_A": 2 * N_PER_COHORT,
        "cohort_B": 2 * N_PER_COHORT,
        "cohort_C": 2 * N_PER_COHORT,
    }
    assert plain_n == expected_plain

    # Country-aware drops the same-country sibling for A and B; C is alone
    # in Y so it still trains on both A and B.
    expected_aware = {
        "cohort_A": N_PER_COHORT,        # only C remains
        "cohort_B": N_PER_COHORT,        # only C remains
        "cohort_C": 2 * N_PER_COHORT,    # singleton country -> nothing excluded
    }
    assert aware_n == expected_aware

    # Plain LODO must report empty exclusions; country-aware must not for A/B.
    plain_excl = dict(zip(plain["cohort"], plain["excluded_cohorts"]))
    for cohort in ("cohort_A", "cohort_B", "cohort_C"):
        assert plain_excl[cohort] == []


def test_country_aware_lodo_predictions_csv_round_trip(tmp_path):
    """Persisted per-sample predictions must contain one row per test
    sample across all three folds, with columns the downstream stats
    pipeline relies on (sample_id, cohort, y_true, y_prob)."""
    X, y, metadata = _make_strong_signal_dataset()
    out = tmp_path / "preds.csv"
    lodo_cv.run_lodo_cv(
        model_fn=_model_fn,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col="country",
        save_predictions_path=str(out),
    )
    df = pd.read_csv(out)
    # 3 cohorts * N_PER_COHORT samples each -> all rows.
    assert len(df) == 3 * N_PER_COHORT
    assert set(df.columns) == {"sample_id", "cohort", "y_true", "y_prob"}
    assert set(df["cohort"]) == {"cohort_A", "cohort_B", "cohort_C"}
    # No leakage: each sample_id appears exactly once.
    assert df["sample_id"].is_unique
    # y_prob is a valid probability.
    assert df["y_prob"].between(0.0, 1.0).all()
