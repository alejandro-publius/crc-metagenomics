"""Smoke tests for the orchestration scripts ``scripts/bootstrap_ci.py``
and ``scripts/auc_comparison.py``.

The unit tests in ``test_bootstrap_ci.py`` and ``test_auc_comparison.py``
cover the underlying statistical helpers in detail, but nothing in the
existing suite asserts that the scripts themselves can be invoked as
scripts. These two scripts also currently have NO argparse layer: any
positional/keyword flag is silently ignored, and ``--help`` blocks for
~minutes on the real data. This test file documents that gap with an
XFAIL marker so it surfaces in CI rather than rotting silently — if a
future refactor adds argparse, the xfail will flip to XPASS and the
caller will know to remove the marker.

We also include a subprocess-driven end-to-end run of
``bootstrap_ci.py`` against a fixture directory populated with crafted
``preds_*.csv`` files, which exercises file IO, ``process_preds``,
and the CSV writer in a single test.

Run with:
    pytest tests/test_orchestration_scripts.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
BOOTSTRAP_CI = os.path.join(SCRIPTS_DIR, "bootstrap_ci.py")
AUC_COMPARISON = os.path.join(SCRIPTS_DIR, "auc_comparison.py")


# ---------------------------------------------------------------------------
# 1. CLI documentation gap (recorded as xfail).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "scripts/bootstrap_ci.py currently has no argparse layer; "
        "--help is not recognised and the script attempts to read "
        "results/preds_*.csv from the cwd. Add argparse to fix."
    ),
    strict=False,
)
def test_bootstrap_ci_has_cli_help_flag(tmp_path):
    """If argparse is added, ``--help`` exits 0 in well under a second
    AND prints a usage line. This test will XPASS once that lands."""
    r = subprocess.run(
        [sys.executable, BOOTSTRAP_CI, "--help"],
        capture_output=True, text=True,
        cwd=str(tmp_path),  # avoid finding the real results/ files
        timeout=10,
    )
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()


@pytest.mark.xfail(
    reason=(
        "scripts/auc_comparison.py currently has no argparse layer; "
        "--help is not recognised and the script tries to read "
        "results/baseline_results.csv and friends. Add argparse to fix."
    ),
    strict=False,
)
def test_auc_comparison_has_cli_help_flag(tmp_path):
    r = subprocess.run(
        [sys.executable, AUC_COMPARISON, "--help"],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        timeout=10,
    )
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()


# ---------------------------------------------------------------------------
# 2. Scripts exist and are importable as modules without side effects.
# ---------------------------------------------------------------------------


def test_orchestration_scripts_exist():
    """Both scripts must exist on disk and be readable as files."""
    assert os.path.isfile(BOOTSTRAP_CI), (
        f"missing orchestration script: {BOOTSTRAP_CI}"
    )
    assert os.path.isfile(AUC_COMPARISON), (
        f"missing orchestration script: {AUC_COMPARISON}"
    )


def test_orchestration_scripts_importable_without_side_effects():
    """Importing the modules must NOT execute their main(); both use the
    ``if __name__ == '__main__'`` guard."""
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    # Force a fresh import.
    for name in ("bootstrap_ci", "auc_comparison"):
        sys.modules.pop(name, None)

    import bootstrap_ci  # noqa: F401
    import auc_comparison  # noqa: F401

    # Both scripts must expose the documented helpers (this is the test
    # other modules / future refactors depend on).
    assert callable(bootstrap_ci.bootstrap_auc_iid)
    assert callable(bootstrap_ci.bootstrap_auc_stratified)
    assert callable(bootstrap_ci.process_preds)
    assert callable(bootstrap_ci.main)

    assert callable(auc_comparison.delong_roc_test)
    assert callable(auc_comparison.boot_ci)
    assert callable(auc_comparison.compare)
    assert callable(auc_comparison.main)


# ---------------------------------------------------------------------------
# 3. Subprocess invocation of bootstrap_ci.py against a fixture directory.
# ---------------------------------------------------------------------------


def _write_pred_csv(path, n_per_cohort=40, n_cohorts=3, seed=0, signal=0.7):
    """Write a synthetic preds CSV of the shape ``bootstrap_ci`` expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for c_idx in range(n_cohorts):
        cohort = f"cohort_{c_idx}"
        y = np.array([0] * (n_per_cohort // 2) + [1] * (n_per_cohort // 2))
        rng.shuffle(y)
        yp = y * signal + rng.normal(scale=1.0, size=n_per_cohort)
        for i in range(n_per_cohort):
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "cohort": cohort,
                "y_true": int(y[i]),
                "y_prob": float(yp[i]),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.slow
def test_bootstrap_ci_subprocess_writes_results_csv(tmp_path):
    """Run scripts/bootstrap_ci.py as a subprocess inside a fixture cwd
    that has the three expected ``results/preds_*.csv`` files. The script
    must write ``results/bootstrap_ci.csv`` with the documented columns.

    This is the missing integration test for the orchestration layer.
    Marked slow because the script's hardcoded N_BOOT=10000 makes it
    take ~30-60s — there is no CLI flag to override it (yet).
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for label in ("species_rf", "joint_rf", "joint_xgb"):
        _write_pred_csv(results_dir / f"preds_{label}.csv", seed=hash(label) % 1000)

    r = subprocess.run(
        [sys.executable, BOOTSTRAP_CI],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        timeout=120,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if r.returncode != 0:
        pytest.fail(
            f"bootstrap_ci.py exited {r.returncode}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    out_csv = results_dir / "bootstrap_ci.csv"
    assert out_csv.exists(), (
        f"bootstrap_ci.py did not write {out_csv}.\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    df = pd.read_csv(out_csv)
    expected_cols = {"model", "cohort", "auc", "ci_lo", "ci_hi", "n"}
    assert expected_cols.issubset(df.columns), (
        f"missing columns: {expected_cols - set(df.columns)}"
    )
    # 3 models * (3 per-cohort + 1 pooled) = 12 rows.
    assert len(df) == 12
    # Every CI bracket must be ordered and inside [0, 1].
    assert (df["ci_lo"] <= df["ci_hi"]).all()
    assert df["ci_lo"].between(0.0, 1.0).all()
    assert df["ci_hi"].between(0.0, 1.0).all()


@pytest.mark.slow
def test_bootstrap_ci_subprocess_warns_on_missing_preds(tmp_path):
    """If the per-model preds CSV is missing, the script must print a
    WARNING line (not crash) and still produce the output CSV with rows
    for the models it did find. Marked slow for the same reason as the
    sister test above."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_pred_csv(results_dir / "preds_species_rf.csv", seed=0)
    # Deliberately omit preds_joint_rf.csv and preds_joint_xgb.csv.

    r = subprocess.run(
        [sys.executable, BOOTSTRAP_CI],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stdout
    df = pd.read_csv(results_dir / "bootstrap_ci.csv")
    # Only species_rf was processed -> 3 per-cohort + 1 pooled = 4 rows.
    assert len(df) == 4
    assert set(df["model"]) == {"species_rf"}


# ---------------------------------------------------------------------------
# 4. auc_comparison.py main() runs end-to-end on crafted result CSVs.
# ---------------------------------------------------------------------------


def _write_results_csv(path, model_col, seed=0, n_cohorts=4):
    """Write a per-cohort result frame with one AUC column ``model_col``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cohorts):
        rows.append({
            "cohort": f"cohort_{i}",
            model_col: float(rng.uniform(0.65, 0.85)),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_auc_comparison_subprocess_writes_model_comparison_csv(tmp_path):
    """End-to-end run of scripts/auc_comparison.py against crafted
    baseline + joint result CSVs. Must produce ``results/model_comparison.csv``
    with the three documented comparisons."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # baseline_results.csv has 'auc'; joint_results.csv has rf_auc + xgb_auc.
    rng = np.random.default_rng(0)
    n_cohorts = 4
    baseline = pd.DataFrame({
        "cohort": [f"cohort_{i}" for i in range(n_cohorts)],
        "auc": rng.uniform(0.65, 0.85, size=n_cohorts),
    })
    joint = pd.DataFrame({
        "cohort": [f"cohort_{i}" for i in range(n_cohorts)],
        "rf_auc": rng.uniform(0.65, 0.85, size=n_cohorts),
        "xgb_auc": rng.uniform(0.65, 0.85, size=n_cohorts),
    })
    baseline.to_csv(results_dir / "baseline_results.csv", index=False)
    joint.to_csv(results_dir / "joint_results.csv", index=False)

    # Omit preds_*.csv so the DeLong block is skipped (documented behaviour).
    r = subprocess.run(
        [sys.executable, AUC_COMPARISON],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        timeout=60,
    )
    if r.returncode != 0:
        pytest.fail(
            f"auc_comparison.py exited {r.returncode}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    out_csv = results_dir / "model_comparison.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    # 3 pairwise comparisons recorded.
    assert len(df) == 3
    assert set(df["comparison"]) == {
        "Species RF vs Joint RF",
        "Species RF vs Joint XGB",
        "Joint XGB vs Joint RF",
    }
    # DeLong is correctly skipped (no preds CSVs present).
    assert "DeLong skipped" in r.stdout
    assert not (results_dir / "delong_results.csv").exists()
