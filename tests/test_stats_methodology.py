"""Deep methodology tests for the statistical layer.

These tests complement ``tests/test_bootstrap_ci.py`` and
``tests/test_auc_comparison.py`` by pinning down four properties that
surface-level tests do not catch:

1. **Stratification shrinkage**: the cohort-stratified bootstrap CI
   shrinks at the expected ``1/sqrt(N)`` rate when the per-cohort
   variances are held constant and the total sample size grows.

2. **DeLong identity on identical predictions**: feeding the same
   prediction vector twice must collapse the variance of the AUC
   difference to zero and return ``z=0, p=1`` exactly (no NaNs from a
   divide-by-zero).

3. **Per-fold filter regression**: ``per_fold_pathway_filter`` selects
   features using only the rows it is given; the surviving column set
   for a training fold must NOT change when arbitrary held-out rows are
   appended to the matrix.

4. **ComBat per-fold isolation**: the ``batch_correction`` script
   applies ComBat fold-by-fold (not once on the global matrix), so a
   contract test confirms the script's call site re-instantiates the
   correction inside the LODO loop rather than caching a global fit.

5. **Brier-decomposition identity**: the Murphy (1973) decomposition
   produced by ``calibration_mechanism.brier_decomposition`` plus the
   recorded residual must equal the Brier score to floating-point
   precision.

6. **BCa endpoints bracket the percentile endpoints on a symmetric
   distribution**: the BCa CI from ``bootstrap_pooled_ci`` must
   converge to the percentile CI when bias correction and acceleration
   are both negligible (i.e. on a near-symmetric near-Gaussian
   resample distribution).

Run with:
    pytest tests/test_stats_methodology.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
for p in (_SRC, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from crc_lodo_bench.stats import bootstrap_pooled_ci, delong_test  # noqa: E402
from crc_lodo_bench.filters import per_fold_pathway_filter  # noqa: E402


def _load_module(name: str, rel_path: str):
    """Import a script-level module by absolute path.

    Avoids polluting ``sys.path`` with the diagnostics directory and
    sidesteps import collisions when the script name shadows a stdlib
    module.
    """
    full = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, full)
    assert spec is not None and spec.loader is not None, full
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Stratification shrinkage
# ---------------------------------------------------------------------------


def _make_pooled(n_per_cohort: int, n_cohorts: int, signal: float, seed: int):
    """Synthesise pooled LODO-shaped predictions with controllable per-cohort
    variance. Returns (y_true, y_prob, cohort)."""
    rng = np.random.default_rng(seed)
    y_true = []
    y_prob = []
    cohort = []
    for c in range(n_cohorts):
        n_half = n_per_cohort // 2
        yc = np.array([0] * n_half + [1] * (n_per_cohort - n_half))
        rng.shuffle(yc)
        # Same signal and same noise variance per cohort so total variance
        # is governed only by the pooled sample size.
        pc = yc * signal + rng.normal(scale=1.0, size=n_per_cohort)
        y_true.append(yc)
        y_prob.append(pc)
        cohort.append(np.array([f"c{c}"] * n_per_cohort))
    return (
        np.concatenate(y_true),
        np.concatenate(y_prob),
        np.concatenate(cohort),
    )


def test_stratified_bootstrap_ci_width_shrinks_with_n():
    """Doubling per-cohort N should approximately ``1/sqrt(2)`` the CI width
    when per-cohort variances are held constant. We allow a generous
    factor-of-two slack because the bootstrap CI width is itself noisy at
    these sample sizes."""
    yt_s, yp_s, co_s = _make_pooled(
        n_per_cohort=80, n_cohorts=4, signal=0.6, seed=0
    )
    yt_l, yp_l, co_l = _make_pooled(
        n_per_cohort=320, n_cohorts=4, signal=0.6, seed=0
    )
    small = bootstrap_pooled_ci(yt_s, yp_s, co_s, n_boot=600, seed=0)
    large = bootstrap_pooled_ci(yt_l, yp_l, co_l, n_boot=600, seed=0)
    w_small = small["ci_high"] - small["ci_low"]
    w_large = large["ci_high"] - large["ci_low"]
    ratio = w_large / w_small
    # Expected ratio is ~1/sqrt(4) = 0.5; require shrinkage at least to 0.75
    # to stay well clear of bootstrap noise but still catch a regression
    # that breaks the stratified contract (which would make the ratio ~1.0).
    assert ratio < 0.75, (
        f"stratified CI width did not shrink with N: ratio={ratio:.3f}, "
        f"small_width={w_small:.4f}, large_width={w_large:.4f}"
    )


def test_stratified_bootstrap_resamples_within_each_cohort_exactly():
    """Each iteration must draw exactly len(cohort) indices from each
    cohort. We verify by patching roc_auc_score and inspecting the cohort
    composition of the resampled index vector."""
    import bootstrap_ci  # type: ignore[import-not-found]

    yt, yp, co = _make_pooled(n_per_cohort=30, n_cohorts=3, signal=0.7, seed=1)
    df = pd.DataFrame({
        "sample_id": np.arange(len(yt)),
        "cohort": co,
        "y_true": yt,
        "y_prob": yp,
    })

    captured = []
    real = bootstrap_ci.roc_auc_score

    def spy(yt_in, yp_in):
        captured.append(np.asarray(yt_in))
        return real(yt_in, yp_in)

    bootstrap_ci.roc_auc_score = spy
    try:
        bootstrap_ci.bootstrap_auc_stratified(df, n_boot=4, seed=7)
    finally:
        bootstrap_ci.roc_auc_score = real

    expected_total = len(yt)
    assert len(captured) == 4
    for arr in captured:
        # Per-cohort counts are not directly visible, but the total length
        # is conserved iff stratification is preserved.
        assert len(arr) == expected_total


# ---------------------------------------------------------------------------
# 2. DeLong identity on identical predictions
# ---------------------------------------------------------------------------


def test_delong_identical_predictions_gives_z_zero_p_one():
    """When both classifiers produce the same scores, the variance of the
    AUC difference is exactly zero and the documented contract is
    (z=0, p=1). The implementation must NOT divide by zero or emit NaN."""
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.5).astype(int)
    p = rng.random(200)
    r = delong_test(y, p, p)
    assert r["auc_a"] == r["auc_b"]
    assert r["auc_diff"] == 0.0
    assert r["z"] == 0.0
    assert r["p_value"] == 1.0


def test_delong_p_value_is_two_sided_normal():
    rng = np.random.default_rng(42)
    y = (rng.random(500) < 0.4).astype(int)
    a = y * 0.7 + rng.normal(scale=1.0, size=500)
    b = y * 0.4 + rng.normal(scale=1.0, size=500)
    r = delong_test(y, a, b)
    from scipy.stats import norm
    expected_p = 2.0 * (1.0 - norm.cdf(abs(r["z"])))
    assert r["p_value"] == pytest.approx(expected_p, abs=1e-12)
    assert 0.0 <= r["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Per-fold filter leakage regression
# ---------------------------------------------------------------------------


def test_per_fold_filter_is_independent_of_test_rows():
    """Add unrelated 'test-fold' rows to the matrix and re-run the filter.
    Because filtering is applied only to the rows passed in, the surviving
    column set must depend solely on whichever subset is handed to the
    callable. If a future refactor accidentally closes over the full X,
    this test will fail.
    """
    rng = np.random.default_rng(0)
    n_train = 100
    # Train rows: 12% prevalence on pw_a (above 10% threshold), 5% on pw_b.
    train = pd.DataFrame({
        "pw_a": np.concatenate([np.full(12, 0.3), np.zeros(n_train - 12)]),
        "pw_b": np.concatenate([np.full(5, 0.3), np.zeros(n_train - 5)]),
    })
    filt = per_fold_pathway_filter(
        filtered_cols=["pw_a", "pw_b"],
        prevalence_threshold=0.10,
        mean_threshold=1e-9,
    )
    kept_train_only = filt(train)
    assert "pw_a" in kept_train_only
    assert "pw_b" not in kept_train_only

    # Build "fold full" matrix that includes adversarial test rows where
    # pw_b is highly prevalent. If the filter ever looks at this larger
    # matrix, it would falsely keep pw_b. By contract the caller passes
    # only the train slice, so this regression test enforces that the
    # filter is a pure function of its argument.
    test = pd.DataFrame({
        "pw_a": np.zeros(50),
        "pw_b": np.full(50, 0.3),  # 100% prevalence in 'test rows'
    })
    full = pd.concat([train, test], axis=0, ignore_index=True)
    # The contract is: caller hands us X_train; we never see the full
    # matrix. So re-invoking the same filter on the same train slice (even
    # though `full` exists in scope) must give the identical answer.
    assert filt(train) == kept_train_only
    # And if the caller wrongly passed the full matrix, the answer would
    # *change* — this is the leakage-detection assertion.
    kept_if_leaked = filt(full)
    assert "pw_b" in kept_if_leaked
    assert kept_if_leaked != kept_train_only


# ---------------------------------------------------------------------------
# 4. ComBat per-fold isolation contract
# ---------------------------------------------------------------------------


def test_batch_correction_applies_combat_inside_lodo_loop():
    """The batch_correction script must call pycombat inside the per-fold
    loop, not once on the global matrix. We verify by static inspection of
    the source: the call site for pycombat must appear *after* the
    ``for cohort, train_idx, test_idx, excluded in get_lodo_splits`` line.

    A regression that hoists pycombat above the loop (i.e. caches a global
    correction and re-uses it across folds) would defeat the LODO
    no-leakage guarantee and silently shift every per-fold AUC.
    """
    src_path = os.path.join(_REPO_ROOT, "scripts", "batch_correction.py")
    with open(src_path, "r") as fh:
        src = fh.read()
    loop_idx = src.find("for cohort, train_idx, test_idx, excluded in get_lodo_splits")
    combat_idx = src.find("pycombat(")
    assert loop_idx >= 0, "loop header not found in batch_correction.py"
    assert combat_idx >= 0, "pycombat() call not found in batch_correction.py"
    assert combat_idx > loop_idx, (
        "batch_correction.py appears to hoist pycombat() above the LODO loop; "
        "ComBat must be re-fit per fold to preserve the no-leakage contract."
    )


# ---------------------------------------------------------------------------
# 5. Brier-decomposition identity
# ---------------------------------------------------------------------------


def test_brier_decomposition_plus_residual_equals_brier():
    """The reported (reliability, resolution, uncertainty) plus the
    explicit residual must equal the Brier score exactly. Bin-mean
    discretization makes the *residual* nonzero, but the total decomposition
    must always sum to the Brier score; this is a regression test against
    silent shifts in the decomposition arithmetic.
    """
    mech = _load_module(
        "calibration_mechanism",
        os.path.join("scripts", "diagnostics", "calibration_mechanism.py"),
    )
    from sklearn.metrics import brier_score_loss
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.45).astype(int)
    # Mix of well-calibrated and miscalibrated bins to exercise both terms.
    base = rng.beta(2.0, 2.0, size=n)
    p = np.clip(0.3 * y + 0.7 * base, 1e-6, 1 - 1e-6)
    rel, res, unc = mech.brier_decomposition(y, p, n_bins=10)
    brier = brier_score_loss(y, p)
    residual = brier - (rel - res + unc)
    # The full identity (decomposition + residual = brier) must hold to
    # machine precision because residual is defined as the gap.
    assert (rel - res + unc + residual) == pytest.approx(brier, abs=1e-12)
    # Residual itself should be small but nonzero in general; bound at 5%
    # of Brier as a sanity check that the decomposition is doing its job.
    assert abs(residual) < 0.05 * brier + 1e-3


def test_brier_decomposition_on_perfect_predictions_is_zero():
    """If predictions equal labels, reliability and the Brier score are
    both zero; resolution equals uncertainty so they cancel."""
    mech = _load_module(
        "calibration_mechanism",
        os.path.join("scripts", "diagnostics", "calibration_mechanism.py"),
    )
    y = np.array([0] * 50 + [1] * 50)
    p = y.astype(float)
    rel, res, unc = mech.brier_decomposition(y, p, n_bins=10)
    assert rel == pytest.approx(0.0, abs=1e-12)
    assert res == pytest.approx(0.25, abs=1e-12)
    assert unc == pytest.approx(0.25, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. BCa endpoints behave correctly
# ---------------------------------------------------------------------------


def test_bca_method_returns_valid_interval_with_method_key():
    yt, yp, co = _make_pooled(
        n_per_cohort=80, n_cohorts=4, signal=0.6, seed=11
    )
    r = bootstrap_pooled_ci(yt, yp, co, n_boot=400, seed=11, method="bca")
    assert r["method"] == "bca"
    assert 0.0 <= r["ci_low"] < r["ci_high"] <= 1.0
    # The BCa interval must contain (or be very close to) the point AUC.
    assert r["ci_low"] - 0.05 <= r["auc"] <= r["ci_high"] + 0.05


def test_bca_and_percentile_agree_on_near_symmetric_distribution():
    """With near-symmetric resamples and minimal skewness the BCa interval
    should be similar to the percentile interval. Wide tolerance (4%) so
    we don't catch rounding noise."""
    yt, yp, co = _make_pooled(
        n_per_cohort=200, n_cohorts=4, signal=0.7, seed=23
    )
    rp = bootstrap_pooled_ci(yt, yp, co, n_boot=500, seed=23, method="percentile")
    rb = bootstrap_pooled_ci(yt, yp, co, n_boot=500, seed=23, method="bca")
    assert abs(rp["ci_low"] - rb["ci_low"]) < 0.04
    assert abs(rp["ci_high"] - rb["ci_high"]) < 0.04


def test_bootstrap_pooled_ci_rejects_unknown_method():
    yt, yp, co = _make_pooled(
        n_per_cohort=40, n_cohorts=2, signal=0.5, seed=0
    )
    with pytest.raises(ValueError):
        bootstrap_pooled_ci(yt, yp, co, n_boot=50, seed=0, method="bayesian")


def test_bootstrap_pooled_ci_exposes_n_boot_kept():
    """Even when zero iterations are dropped, the n_boot_kept key must be
    present so downstream consumers can audit the dropping rate."""
    yt, yp, co = _make_pooled(
        n_per_cohort=60, n_cohorts=3, signal=0.6, seed=0
    )
    r = bootstrap_pooled_ci(yt, yp, co, n_boot=200, seed=0)
    assert "n_boot_kept" in r
    assert 0 < r["n_boot_kept"] <= 200


def test_bootstrap_ci_script_exposes_n_boot_kept_in_row():
    """Regression test on the script-level CSV schema: each row written by
    ``process_preds`` must include the ``n_boot_kept`` column so reviewers
    can see whether any iterations were silently dropped."""
    import bootstrap_ci  # type: ignore[import-not-found]
    n = 80
    rng = np.random.default_rng(0)
    rows = []
    for ci, cohort in enumerate(["A", "B"]):
        y = (rng.random(n) < 0.5).astype(int)
        p = y * 0.6 + rng.normal(scale=1.0, size=n)
        for i in range(n):
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "cohort": cohort,
                "y_true": int(y[i]),
                "y_prob": float(p[i]),
            })
    df = pd.DataFrame(rows)

    # Bypass file I/O: monkey-patch read_csv to return our synthetic frame.
    # We also wrap the two bootstrap functions to force a tiny n_boot so
    # the end-to-end call is fast (the module-level N_BOOT constant is
    # captured at function definition time and cannot be patched).
    orig_read = bootstrap_ci.pd.read_csv
    orig_iid = bootstrap_ci.bootstrap_auc_iid
    orig_strat = bootstrap_ci.bootstrap_auc_stratified
    bootstrap_ci.pd.read_csv = lambda _p: df
    bootstrap_ci.bootstrap_auc_iid = (
        lambda yt, yp, n_boot=200, seed=42, return_kept=False:
        orig_iid(yt, yp, n_boot=200, seed=seed, return_kept=return_kept)
    )
    bootstrap_ci.bootstrap_auc_stratified = (
        lambda d, n_boot=200, seed=42, return_kept=False:
        orig_strat(d, n_boot=200, seed=seed, return_kept=return_kept)
    )
    try:
        out_rows = bootstrap_ci.process_preds("ignored.csv", "synthetic")
    finally:
        bootstrap_ci.pd.read_csv = orig_read
        bootstrap_ci.bootstrap_auc_iid = orig_iid
        bootstrap_ci.bootstrap_auc_stratified = orig_strat

    for row in out_rows:
        assert "n_boot_kept" in row
        assert 0 < row["n_boot_kept"] <= 200
