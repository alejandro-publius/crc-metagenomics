"""Regression tests for train_joint.py's pathway-cohort-coverage guard.

data/raw/pathway_chunks/ ships HUMAnN pathway tables for only a subset of
the 10 analysis cohorts (the rest are omitted from git for size -- see
README.md's "Data-availability boundary" note). Before the guard this
module tests, `train_joint.py` silently trained a "joint" model on
whichever cohorts happened to have pathway data (an inner join on
sample_id drops the rest with no warning), so a fresh clone that skips
`Rscript scripts/export_data.R` would get a plausible-looking but wrong,
undersized joint AUC -- and any comparison against the full 10-cohort
species baseline (e.g. auc_comparison.py) would then fail on a bare shape
mismatch with no explanation.

These tests run scripts/train_joint.py as a subprocess against a tiny
synthetic fixture directory, mirroring the pattern in
tests/test_orchestration_scripts.py.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_JOINT = os.path.join(REPO_ROOT, "scripts", "train_joint.py")

COHORTS = [("cohortA", "US"), ("cohortB", "FR")]
N_PER_COHORT = 20
N_SPECIES = 60
N_PATHWAYS = 60


def _write_fixture(base, cohorts):
    """Write species_filtered.csv, metadata_clean.csv, and
    pathway_abundance.csv (covering only ``cohorts``) into ``base``."""
    rng = np.random.default_rng(0)
    (base / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base / "data" / "raw").mkdir(parents=True, exist_ok=True)

    species_rows, meta_rows = [], []
    for cohort, country in COHORTS:
        for i in range(N_PER_COHORT):
            sid = f"{cohort}_{i}"
            label = 1 if i < N_PER_COHORT // 2 else 0
            meta_rows.append({
                "sample_id": sid, "study_name": cohort,
                "study_condition": "CRC" if label == 1 else "control",
                "label": label, "country": country,
            })
            species_rows.append({
                "sample_id": sid,
                **{f"sp_{j}": float(rng.random()) for j in range(N_SPECIES)},
            })
    pd.DataFrame(species_rows).to_csv(base / "data/processed/species_filtered.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(base / "data/processed/metadata_clean.csv", index=False)

    pathway_rows = []
    for cohort, _ in cohorts:  # only the cohorts passed in get pathway rows
        for i in range(N_PER_COHORT):
            pathway_rows.append({
                "sample_id": f"{cohort}_{i}",
                **{f"PWY{j}": float(rng.random() > 0.3) * rng.random()
                   for j in range(N_PATHWAYS)},
            })
    pd.DataFrame(pathway_rows).to_csv(base / "data/raw/pathway_abundance.csv", index=False)


def _run(cwd):
    return subprocess.run(
        [sys.executable, TRAIN_JOINT],
        capture_output=True, text=True, cwd=str(cwd), timeout=60,
    )


def test_missing_cohort_pathway_data_fails_with_one_clear_line(tmp_path):
    """Only cohortA has pathway rows; cohortB's species-only samples are
    dropped by the inner join. The guard must name cohortB and exit
    non-zero WITHOUT a raw traceback."""
    _write_fixture(tmp_path, cohorts=[("cohortA", "US")])

    r = _run(tmp_path)

    assert r.returncode != 0, (
        f"expected a non-zero exit for incomplete pathway coverage.\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    combined = r.stdout + r.stderr
    assert "cohortB" in combined, f"error did not name the missing cohort:\n{combined}"
    assert "missing pathway data for" in combined
    assert "Traceback (most recent call last)" not in combined, (
        f"guard should fail loudly with one clear line, not a raw traceback:\n{combined}"
    )


def test_full_cohort_pathway_coverage_does_not_trip_the_guard(tmp_path):
    """Both cohorts have pathway rows: the coverage guard must not fire,
    and the script should proceed into (and complete) LODO training."""
    _write_fixture(tmp_path, cohorts=COHORTS)

    r = _run(tmp_path)

    combined = r.stdout + r.stderr
    assert "missing pathway data for" not in combined, (
        f"guard fired despite full cohort coverage:\n{combined}"
    )
    assert r.returncode == 0, (
        f"train_joint.py failed for an unrelated reason.\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    out_csv = tmp_path / "results" / "preds_joint_rf.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert set(df["cohort"].unique()) == {"cohortA", "cohortB"}
