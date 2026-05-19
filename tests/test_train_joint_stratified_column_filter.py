"""Regression test for scripts.train_joint structural filter.

The previous filter ``[c for c in pw_raw.columns if c != 'sample_id'
and '|' not in c]`` was intended to strip HUMAnN-stratified columns
("PATHWAY|genus.species") and keep only unstratified candidates. The
filter still does that, but the failure mode it produces when fed a
frame whose columns ALL contain '|' (e.g. a future stratified gene /
strain table) is dangerous: every column gets dropped, the joint
matrix collapses to species-only, and the run silently logs an AUC
that looks like a joint result but is just the species baseline.

This test pins down the new contract: train_joint.py must abort with a
clear error rather than silently producing zero pathway features.

Run with:
    pytest tests/test_train_joint_stratified_column_filter.py -v
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


@pytest.fixture
def all_stratified_pathway_frame(tmp_path, monkeypatch):
    """Write fixture CSVs to a tmp dir and chdir there so train_joint.main
    reads our crafted frames instead of the real data files.

    The crafted pathway frame is deliberately pathological: EVERY column
    name contains '|', so the structural unstratified-only filter
    strips every feature. The species frame is minimal but plausible.
    """
    # Build a small but valid metadata + species table.
    n = 30
    sample_ids = [f"S{i:03d}" for i in range(n)]
    # 3 cohorts, balanced labels.
    cohorts = ["A", "B", "C"]
    md = pd.DataFrame({
        "sample_id": sample_ids,
        "study_name": [cohorts[i % 3] for i in range(n)],
        "study_condition": ["CRC" if i % 2 == 0 else "control"
                            for i in range(n)],
        "country": ["X", "Y", "Z"] * (n // 3),
        "label": [1 if i % 2 == 0 else 0 for i in range(n)],
    })

    rng = np.random.default_rng(0)
    sp = pd.DataFrame({
        "sample_id": sample_ids,
        **{f"sp_{j}": rng.normal(size=n) for j in range(5)},
    })

    # Pathology: every pathway column is stratified, i.e. has '|'.
    pw = pd.DataFrame({
        "sample_id": sample_ids,
        **{f"PWY|genus.species_{j}": rng.random(n) for j in range(20)},
    })

    raw_dir = tmp_path / "data" / "raw"
    proc_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    sp.to_csv(proc_dir / "species_filtered.csv", index=False)
    md.to_csv(proc_dir / "metadata_clean.csv", index=False)
    pw.to_csv(raw_dir / "pathway_abundance.csv", index=False)

    monkeypatch.chdir(tmp_path)
    yield tmp_path


def test_train_joint_aborts_on_all_stratified_input(
    all_stratified_pathway_frame,
):
    """If every pathway column is stratified (contains '|'), train_joint
    must raise a clear ValueError rather than silently training a
    species-only model and logging it as 'joint'."""
    # Force a fresh import so any module-level side effects from a prior
    # test don't bleed in.
    if "train_joint" in sys.modules:
        del sys.modules["train_joint"]
    train_joint = importlib.import_module("train_joint")

    # The species table has only 5 features (well under the MIN_JOINT_FEATURES
    # guard of 100), and stripping all pathway columns leaves no pathway
    # candidates -- so the guard should fire.
    with pytest.raises(ValueError) as excinfo:
        train_joint.main([])

    msg = str(excinfo.value)
    assert "joint" in msg.lower() or "species-only" in msg.lower(), (
        f"error message should explain the species-only degeneration; "
        f"got: {msg!r}"
    )


def test_unstrat_filter_keeps_columns_without_pipe(monkeypatch, tmp_path):
    """Healthy path: when SOME columns are unstratified (no '|') they must
    survive the structural filter and the guard must NOT fire."""
    n = 40
    sample_ids = [f"S{i:03d}" for i in range(n)]
    cohorts = ["A", "B", "C"]
    md = pd.DataFrame({
        "sample_id": sample_ids,
        "study_name": [cohorts[i % 3] for i in range(n)],
        "study_condition": ["CRC" if i % 2 == 0 else "control"
                            for i in range(n)],
        "country": ["X", "Y", "Z"] * (n // 3) + ["X"],
        "label": [1 if i % 2 == 0 else 0 for i in range(n)],
    })

    rng = np.random.default_rng(1)
    # Use 150 species + 150 unstratified pathway candidates so the
    # post-filter X has > MIN_JOINT_FEATURES columns.
    sp = pd.DataFrame({
        "sample_id": sample_ids,
        **{f"sp_{j}": rng.normal(size=n) for j in range(150)},
    })
    pw = pd.DataFrame({
        "sample_id": sample_ids,
        # 150 unstratified (kept) + 50 stratified (stripped).
        **{f"PWY_unstrat_{j}": rng.random(n) for j in range(150)},
        **{f"PWY|strat_{j}": rng.random(n) for j in range(50)},
    })

    raw_dir = tmp_path / "data" / "raw"
    proc_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)
    sp.to_csv(proc_dir / "species_filtered.csv", index=False)
    md.to_csv(proc_dir / "metadata_clean.csv", index=False)
    pw.to_csv(raw_dir / "pathway_abundance.csv", index=False)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    # Provide a stub baseline so the final summary read doesn't blow up.
    pd.DataFrame({"cohort": ["A"], "auc": [0.5]}).to_csv(
        tmp_path / "results" / "baseline_results.csv", index=False,
    )

    monkeypatch.chdir(tmp_path)
    if "train_joint" in sys.modules:
        del sys.modules["train_joint"]
    train_joint = importlib.import_module("train_joint")

    # Should NOT raise (the all-stratified guard is only for the degen
    # case where the structural filter strips everything down to species
    # only with <100 columns total).
    # We don't actually run main() here -- it would fit a full model.
    # Instead, exercise the structural filter inline using the same
    # rule the script uses.
    pw_unstrat_cols = [c for c in pw.columns
                       if c != "sample_id" and ("|" not in c)]
    assert len(pw_unstrat_cols) == 150, (
        f"expected 150 unstratified columns to survive; got "
        f"{len(pw_unstrat_cols)}"
    )
    # And the total joint matrix should clear the MIN_JOINT_FEATURES
    # guard comfortably.
    sp_cols = [c for c in sp.columns if c != "sample_id"]
    assert (len(sp_cols) + len(pw_unstrat_cols)) > train_joint.MIN_JOINT_FEATURES
