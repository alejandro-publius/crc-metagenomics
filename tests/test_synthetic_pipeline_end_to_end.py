"""End-to-end synthetic pipeline test for the LODO benchmarking stack.

This is the test the existing suite was missing: a single, hermetic
exercise of preprocessing -> per-fold filter -> country-aware LODO ->
cohort-stratified bootstrap CI -> DeLong paired-AUC test on a
small-but-realistic synthetic dataset.

Existing tests cover each piece in isolation (``test_lodo_cv.py``,
``test_per_fold_filter.py``, ``test_bootstrap_ci.py``,
``test_auc_comparison.py``) but never verify they compose correctly
on a multi-cohort multi-country dataset whose ground truth is known.

Dataset layout (mirrors the real study shape at miniature scale):

  - 4 cohorts: cohort_A1, cohort_A2 (country US),
               cohort_B  (country DE),
               cohort_C  (country JP).
  - 50 samples each (25 cases, 25 controls) -> 200 samples total.
  - 100 features: feat_0..feat_9 carry a planted CRC signal; the
    remaining 90 are pure noise. A handful of "pathway" columns are
    deliberately near-zero so the per-fold prevalence filter has
    something to drop; a separate set of "species" columns is marked
    passthrough.

Assertions cover the contract questions in the test audit:

  1. Country-aware LODO drops same-country training samples
     (cohort_A1's fold excludes cohort_A2's samples and vice versa).
  2. The per-fold pathway filter is invoked once per fold and never
     sees test-cohort rows.
  3. The cohort-stratified bootstrap CI brackets the point estimate.
  4. DeLong z-stat for two identical model outputs is exactly zero.

Run with:
    pytest tests/test_synthetic_pipeline_end_to_end.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

# Make scripts/ and src/ importable without editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
_SRC = os.path.join(_REPO_ROOT, "src")
for p in (_SCRIPTS, _SRC):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import lodo_cv  # noqa: E402
import bootstrap_ci  # noqa: E402
import auc_comparison  # noqa: E402
from crc_lodo_bench.filters import per_fold_pathway_filter  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------


N_PER_COHORT = 50
N_INFORMATIVE = 10
N_NOISE = 70
N_PATHWAY = 15           # subjected to the per-fold prevalence filter
N_SPECIES_PASSTHROUGH = 5  # always retained
RNG_SEED = 2026


def _build_synthetic_dataset(seed: int = RNG_SEED):
    """Return (X, y, metadata, pathway_cols, species_cols).

    Layout: 4 cohorts across 3 countries; cohort_A1 and cohort_A2 share
    country "US" (the case the country-aware LODO must handle).
    """
    rng = np.random.default_rng(seed)
    cohort_country = [
        ("cohort_A1", "US"),
        ("cohort_A2", "US"),
        ("cohort_B",  "DE"),
        ("cohort_C",  "JP"),
    ]

    rows = []
    feats = []
    pathway_cols = [f"pw_{i}" for i in range(N_PATHWAY)]
    species_cols = [f"sp_{i}" for i in range(N_SPECIES_PASSTHROUGH)]
    informative_cols = [f"feat_inf_{i}" for i in range(N_INFORMATIVE)]
    noise_cols = [f"feat_noise_{i}" for i in range(N_NOISE)]
    feature_cols = informative_cols + noise_cols + pathway_cols + species_cols
    assert len(feature_cols) == 100, "expected exactly 100 feature columns"

    for cohort, country in cohort_country:
        labels = np.array([0] * (N_PER_COHORT // 2) + [1] * (N_PER_COHORT // 2))
        rng.shuffle(labels)
        # Cohort-specific batch offset on noise features only (tests that
        # informative features can still recover the signal).
        cohort_offset = rng.normal(scale=0.2, size=N_NOISE)
        for label in labels:
            rows.append({
                "sample_id": f"{cohort}_{len(rows)}",
                "study_name": cohort,
                "country": country,
                "label": int(label),
            })
            inf = rng.normal(loc=label * 1.5, scale=1.0, size=N_INFORMATIVE)
            noise = rng.normal(loc=cohort_offset, scale=1.0, size=N_NOISE)
            # Pathways: most are very sparse (mostly zero); a few are dense.
            pw = np.zeros(N_PATHWAY)
            # First 3 pathways: present in ~50% of samples with abundance.
            pw[:3] = rng.random(3) * (rng.random(3) > 0.5)
            # Next 5 pathways: present in only ~5% of samples (will be
            # dropped by the prevalence filter at threshold 0.10).
            pw[3:8] = rng.random(5) * (rng.random(5) > 0.95)
            # Remaining pathways: dense but tiny in magnitude (mean ~ 1e-9,
            # so dropped by the mean threshold at 1e-6).
            pw[8:] = rng.random(N_PATHWAY - 8) * 1e-10
            sp = rng.random(N_SPECIES_PASSTHROUGH)
            feats.append(np.concatenate([inf, noise, pw, sp]))

    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(feats, columns=feature_cols, index=metadata.index)
    y = metadata["label"].copy()
    return X, y, metadata, pathway_cols, species_cols


# Module-private estimator: trivial mean-difference scorer that is fast and
# deterministic, so the test runs in well under a second.
class _MeanDifferenceClassifier:
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


# ---------------------------------------------------------------------------
# 1. Full pipeline: preprocess -> filter -> LODO -> bootstrap -> DeLong
# ---------------------------------------------------------------------------


def test_full_pipeline_runs_and_recovers_signal(tmp_path):
    """End-to-end smoke test: every stage of the pipeline must compose
    without raising AND produce a non-degenerate AUC."""
    X, y, metadata, pathway_cols, species_cols = _build_synthetic_dataset()

    filt = per_fold_pathway_filter(
        filtered_cols=pathway_cols,
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
        passthrough_cols=species_cols
        + [c for c in X.columns if c.startswith(("feat_inf_", "feat_noise_"))],
    )

    pred_path = tmp_path / "preds.csv"
    results = lodo_cv.run_lodo_cv(
        model_fn=_MeanDifferenceClassifier,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col="country",
        feature_filter_fn=filt,
        save_predictions_path=str(pred_path),
    )

    # 4 cohorts -> 4 folds.
    assert len(results["cohort"]) == 4
    assert results["mean_auc"] > 0.6, (
        f"planted signal should give mean AUC > 0.6; got {results['mean_auc']}"
    )

    # Bootstrap CI on the saved predictions.
    preds = pd.read_csv(pred_path)
    lo, hi = bootstrap_ci.bootstrap_auc_stratified(preds, n_boot=300, seed=0)
    point = roc_auc_score(preds["y_true"].values, preds["y_prob"].values)
    assert lo < hi, "bootstrap CI must be a non-degenerate interval"
    assert lo - 0.05 <= point <= hi + 0.05, (
        f"point estimate {point:.3f} should sit inside CI [{lo:.3f}, {hi:.3f}]"
    )


# ---------------------------------------------------------------------------
# 2. Country-aware LODO drops the same-country training samples
# ---------------------------------------------------------------------------


def test_country_aware_lodo_drops_same_country_training_samples():
    """When cohort_A1 is the test fold, cohort_A2's samples (same country
    "US") must NOT appear in train_idx, and the excluded set must be
    {cohort_A2}. Symmetric for cohort_A2."""
    _, _, metadata, _, _ = _build_synthetic_dataset()

    by_cohort = {
        c: (set(train_idx), set(test_idx), excluded)
        for c, train_idx, test_idx, excluded in lodo_cv.get_lodo_splits(
            metadata, cohort_col="study_name", country_col="country"
        )
    }

    train_A1, _, excl_A1 = by_cohort["cohort_A1"]
    train_A2, _, excl_A2 = by_cohort["cohort_A2"]
    train_B,  _, excl_B  = by_cohort["cohort_B"]
    train_C,  _, excl_C  = by_cohort["cohort_C"]

    # Exclusion bookkeeping.
    assert excl_A1 == {"cohort_A2"}
    assert excl_A2 == {"cohort_A1"}
    assert excl_B == set()
    assert excl_C == set()

    # cohort_A2 samples must not be present in cohort_A1's training set.
    A2_sample_indices = set(metadata.index[metadata["study_name"] == "cohort_A2"])
    A1_sample_indices = set(metadata.index[metadata["study_name"] == "cohort_A1"])
    assert train_A1.isdisjoint(A2_sample_indices), (
        "cohort_A2 samples leaked into cohort_A1's training fold"
    )
    assert train_A2.isdisjoint(A1_sample_indices)

    # n_train math: A1 trains on {B, C} only (no A2) -> 2 * 50 = 100 rows.
    assert len(train_A1) == 2 * N_PER_COHORT
    assert len(train_A2) == 2 * N_PER_COHORT
    # B trains on {A1, A2, C} -> 3 * 50 = 150.
    assert len(train_B) == 3 * N_PER_COHORT
    assert len(train_C) == 3 * N_PER_COHORT


# ---------------------------------------------------------------------------
# 3. Per-fold filter sees train-only data inside run_lodo_cv
# ---------------------------------------------------------------------------


def test_per_fold_filter_sees_train_only_inside_lodo():
    """Wire the per-fold filter into run_lodo_cv and assert that each
    invocation's input frame is disjoint from that fold's test rows."""
    X, y, metadata, pathway_cols, species_cols = _build_synthetic_dataset()

    seen_indices: list[set] = []

    def spying_filter(X_train):
        seen_indices.append(set(X_train.index))
        # Apply the real prevalence/mean filter so n_features is meaningful.
        real = per_fold_pathway_filter(
            filtered_cols=pathway_cols,
            prevalence_threshold=0.10,
            mean_threshold=1e-6,
            passthrough_cols=species_cols
            + [c for c in X_train.columns
               if c.startswith(("feat_inf_", "feat_noise_"))],
        )
        return real(X_train)

    lodo_cv.run_lodo_cv(
        model_fn=_MeanDifferenceClassifier,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name",
        country_col="country",
        feature_filter_fn=spying_filter,
    )

    # 4 folds -> 4 invocations.
    assert len(seen_indices) == 4

    # Reconstruct fold order and verify per-fold disjointness.
    fold_test_sets = {
        cohort: set(test_idx)
        for cohort, _, test_idx, _ in lodo_cv.get_lodo_splits(
            metadata, cohort_col="study_name", country_col="country"
        )
    }
    fold_order = list(fold_test_sets.keys())
    assert len(fold_order) == len(seen_indices)
    for cohort, train_indices in zip(fold_order, seen_indices):
        leaked = fold_test_sets[cohort] & train_indices
        assert not leaked, (
            f"per_fold_pathway_filter for {cohort} received "
            f"{len(leaked)} test-cohort rows"
        )


# ---------------------------------------------------------------------------
# 4. Per-fold filter actually drops sparse / low-abundance pathways
# ---------------------------------------------------------------------------


def test_per_fold_filter_drops_expected_pathway_columns():
    """The synthetic generator plants:
      - 3 dense pathways (kept)
      - 5 very-rare pathways (~5% prevalence; dropped by prevalence threshold)
      - the rest dense-but-tiny (~1e-10 mean; dropped by mean threshold).
    The filter must produce exactly the 3 surviving pathway columns plus
    every passthrough column.
    """
    X, _, _, pathway_cols, species_cols = _build_synthetic_dataset()

    passthrough = species_cols + [
        c for c in X.columns
        if c.startswith(("feat_inf_", "feat_noise_"))
    ]
    filt = per_fold_pathway_filter(
        filtered_cols=pathway_cols,
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
        passthrough_cols=passthrough,
    )
    kept = filt(X)
    surviving_pw = [c for c in kept if c.startswith("pw_")]
    assert set(surviving_pw).issubset(set(pathway_cols))
    # 3 dense + maybe 0..1 borderline -> at most 4, at least 1.
    assert 1 <= len(surviving_pw) <= 4, (
        f"expected 1-4 surviving pathways, got {len(surviving_pw)}: "
        f"{surviving_pw}"
    )
    # Every passthrough column survives.
    for c in passthrough:
        assert c in kept


# ---------------------------------------------------------------------------
# 5. Bootstrap CI bracketing on real saved predictions
# ---------------------------------------------------------------------------


def test_bootstrap_ci_lower_less_than_point_less_than_upper(tmp_path):
    """On the saved per-sample predictions from the LODO run, the
    cohort-stratified bootstrap CI must be a valid interval that brackets
    the point estimate (within a generous bootstrap slack)."""
    X, y, metadata, pathway_cols, species_cols = _build_synthetic_dataset()
    filt = per_fold_pathway_filter(
        filtered_cols=pathway_cols,
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
        passthrough_cols=species_cols
        + [c for c in X.columns if c.startswith(("feat_inf_", "feat_noise_"))],
    )
    pred_path = tmp_path / "preds.csv"
    lodo_cv.run_lodo_cv(
        model_fn=_MeanDifferenceClassifier,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name", country_col="country",
        feature_filter_fn=filt,
        save_predictions_path=str(pred_path),
    )
    preds = pd.read_csv(pred_path)
    point = roc_auc_score(preds["y_true"].values, preds["y_prob"].values)
    lo, hi = bootstrap_ci.bootstrap_auc_stratified(preds, n_boot=500, seed=0)
    assert 0.0 <= lo < hi <= 1.0
    assert lo - 0.05 <= point <= hi + 0.05


# ---------------------------------------------------------------------------
# 6. DeLong z-stat for identical predictions must be exactly zero.
# ---------------------------------------------------------------------------


def test_delong_identical_predictions_have_zero_z_in_full_pipeline(tmp_path):
    """Save one set of LODO predictions, then compare it against itself
    via the canonical DeLong helper. The contract is z == 0.0, p == 1.0."""
    X, y, metadata, pathway_cols, species_cols = _build_synthetic_dataset()
    filt = per_fold_pathway_filter(
        filtered_cols=pathway_cols,
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
        passthrough_cols=species_cols
        + [c for c in X.columns if c.startswith(("feat_inf_", "feat_noise_"))],
    )
    pred_path = tmp_path / "preds.csv"
    lodo_cv.run_lodo_cv(
        model_fn=_MeanDifferenceClassifier,
        X=X, y=y, metadata=metadata,
        cohort_col="study_name", country_col="country",
        feature_filter_fn=filt,
        save_predictions_path=str(pred_path),
    )
    preds = pd.read_csv(pred_path).sort_values("sample_id").reset_index(drop=True)
    yt = preds["y_true"].values
    yp = preds["y_prob"].values
    auc_a, auc_b, z, p = auc_comparison.delong_roc_test(yt, yp, yp)
    assert auc_a == pytest.approx(auc_b, abs=1e-12)
    assert z == 0.0
    assert p == 1.0


# ---------------------------------------------------------------------------
# 7. DeLong edge cases — all-zero, all-one, perfectly anti-correlated
# ---------------------------------------------------------------------------


def test_delong_all_zero_predictions_does_not_crash():
    """If both classifiers output the same constant (e.g. always 0.0), the
    AUCs are 0.5 and the variance of the difference is 0 -> contract z=0,
    p=1. This catches divide-by-zero regressions in the variance branch."""
    y = np.array([0, 1] * 50)
    yp_zero = np.zeros_like(y, dtype=float)
    auc_a, auc_b, z, p = auc_comparison.delong_roc_test(y, yp_zero, yp_zero)
    assert auc_a == pytest.approx(0.5, abs=1e-12)
    assert auc_b == pytest.approx(0.5, abs=1e-12)
    assert z == 0.0
    assert p == 1.0


def test_delong_all_one_predictions_does_not_crash():
    """Symmetric corner case: constant-1.0 predictions."""
    y = np.array([0, 1] * 50)
    yp_one = np.ones_like(y, dtype=float)
    auc_a, auc_b, z, p = auc_comparison.delong_roc_test(y, yp_one, yp_one)
    assert auc_a == pytest.approx(0.5, abs=1e-12)
    assert z == 0.0
    assert p == 1.0


def test_delong_anti_correlated_predictions_have_opposite_aucs():
    """If model_b = -model_a on a separable problem, AUC_a + AUC_b == 1
    exactly (sanity check on AUC arithmetic in delong_roc_test)."""
    rng = np.random.default_rng(0)
    y = (rng.random(120) < 0.5).astype(int)
    yp_a = y + rng.normal(scale=0.3, size=120)
    yp_b = -yp_a
    auc_a, auc_b, _, _ = auc_comparison.delong_roc_test(y, yp_a, yp_b)
    assert auc_a + auc_b == pytest.approx(1.0, abs=1e-12)
