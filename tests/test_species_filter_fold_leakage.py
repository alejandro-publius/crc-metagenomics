"""Regression test: the species feature filter is not fold-safe.

``scripts/preprocessing.py`` step 4 picks the retained species columns
(prevalence >= 10%, mean abundance >= 1e-4) from the FULL 10-cohort
population -- every held-out LODO test cohort included -- before any
train/test split exists. Every downstream script that reads
``data/processed/species_filtered.csv`` (``train_baseline.py``,
``train_joint.py``'s species passthrough, ``train_stratified_joint.py``'s
species passthrough, ``seed_sensitivity.py``, ...) therefore trains on a
feature *set* that was chosen partly using each fold's own held-out
cohort.

This directly contradicts the documented contract in
``docs/ARCHITECTURE.md`` ("Held-out populations cannot select
themselves" / "Feature filtering ... occur[s] inside the training side
of each ... split") and the "Modeling core ... fit leakage-safe
reference models" component-map entry, both of which describe the
*pathway* per-fold filter (``crc_lodo_bench.filters.per_fold_pathway_filter``,
covered by ``tests/test_per_fold_filter.py``) but are stated as if they
applied to the whole modeling core, species included.

This test recomputes, from the committed raw inputs, the species column
list a correct per-fold (train-cohorts-only) filter would retain for
each LODO fold, using the exact thresholds and ordering
``scripts/preprocessing.py`` documents (prevalence >= 0.10 computed on
raw un-renormalised abundances, mean >= 1e-4, same two cohort-level
exclusions). It asserts that list equals the column list actually baked
into the committed ``data/processed/species_filtered.csv``.

It currently fails: for the two largest cohorts (YachidaS_2019, 508
test samples; ThomasAM_2019_c) the fold-restricted list differs from
the committed global list by 27 of ~224-245 columns (about 11-12%);
every one of the 10 folds differs by at least 2 columns. See the PR
this test ships in for a from-scratch fold-safe re-run showing the
practical impact on the headline species-only AUC is small and does
NOT reverse the species-beats-joint finding (mean AUC moves from
0.8075 to 0.8134, i.e. slightly *up*, not down) -- so this is a real,
demonstrated methodology/documentation-accuracy defect, not (on this
data) the explanation for the triple-confirmed null. Fixing it for
real would change every published number that touches
``species_filtered.csv``, so this test is xfail rather than a stealth
fix.

Run with:
    pytest tests/test_species_filter_fold_leakage.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_READS = 1_000_000
EXCLUDE_COHORTS = ["HanniganGD_2017"]
PREVALENCE_THRESHOLD = 0.10
MEAN_THRESHOLD = 1e-4


def _load_raw_population() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Reproduce preprocessing.py steps 1-3: depth filter, cohort exclusion,
    and species/metadata alignment. Returns (species, metadata, feature_cols)
    with no feature SELECTION applied yet.
    """
    species = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/species_abundance.csv"))
    metadata = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/metadata.csv"))

    if "number_reads" in metadata.columns:
        metadata = metadata[metadata["number_reads"] >= MIN_READS].reset_index(drop=True)
    metadata = metadata[~metadata["study_name"].isin(EXCLUDE_COHORTS)].reset_index(drop=True)

    common = set(species["sample_id"]) & set(metadata["sample_id"])
    species = species[species["sample_id"].isin(common)].reset_index(drop=True)
    metadata = metadata[metadata["sample_id"].isin(common)].reset_index(drop=True)
    metadata["label"] = metadata["study_condition"].map(
        {"CRC": 1, "control": 0, "adenoma": -1}
    )

    feature_cols = [c for c in species.columns if c != "sample_id"]
    return species, metadata, feature_cols


def _prevalence_mean_filter(X: pd.DataFrame, feature_cols: list[str]) -> set[str]:
    prev = (X[feature_cols] > 0).mean(axis=0)
    ma = X[feature_cols].mean(axis=0)
    return set(prev[prev >= PREVALENCE_THRESHOLD].index) & set(
        ma[ma >= MEAN_THRESHOLD].index
    )


def _lodo_cohorts(metadata: pd.DataFrame) -> list[str]:
    valid = metadata[metadata["label"].isin([0, 1])]
    return sorted(valid["study_name"].unique())


@pytest.mark.xfail(
    strict=True,
    reason=(
        "scripts/preprocessing.py selects species columns from the FULL "
        "10-cohort population, so every LODO fold's held-out test cohort "
        "influences the training-side feature list. A correct per-fold "
        "filter (computed on training cohorts only, same thresholds) "
        "retains a different column set for every one of the 10 folds. "
        "This contradicts docs/ARCHITECTURE.md's 'leakage-safe reference "
        "models' / 'feature filtering ... occur[s] inside the training "
        "side of each ... split' claim for the species arm specifically. "
        "See test docstring for the measured, non-catastrophic AUC impact."
    ),
)
def test_committed_species_filter_matches_a_fold_safe_recomputation():
    committed = pd.read_csv(
        os.path.join(REPO_ROOT, "data/processed/species_filtered.csv")
    )
    committed_cols = set(c for c in committed.columns if c != "sample_id")

    species, metadata, feature_cols = _load_raw_population()
    mg = metadata.merge(species, on="sample_id", how="inner")
    mask = mg["label"].isin([0, 1])

    mismatches = {}
    for cohort in _lodo_cohorts(metadata):
        train_rows = mg.loc[mask & (mg["study_name"] != cohort)]
        fold_safe_cols = _prevalence_mean_filter(train_rows, feature_cols)
        diff = committed_cols.symmetric_difference(fold_safe_cols)
        if diff:
            mismatches[cohort] = len(diff)

    # A leakage-free pipeline would retain the SAME committed feature set
    # for every fold, because held-out-cohort samples would never have
    # touched the filter's prevalence/mean statistics in the first place.
    assert not mismatches, (
        f"species feature list differs from a fold-safe recomputation in "
        f"{len(mismatches)}/10 LODO folds (columns differing per fold: "
        f"{mismatches}) -- the committed species_filtered.csv was built "
        f"from the full population, not the training side of each split."
    )


def test_per_fold_species_filter_is_internally_consistent_and_nonempty():
    """Sanity check on the fold-safe recomputation itself (not xfail): every
    fold must retain a non-trivial number of species and the two largest
    cohorts (the ones most likely to move the global filter) must be among
    the folds actually exercised, so the xfail test above is not vacuous.
    """
    species, metadata, feature_cols = _load_raw_population()
    mg = metadata.merge(species, on="sample_id", how="inner")
    mask = mg["label"].isin([0, 1])

    cohorts = _lodo_cohorts(metadata)
    assert {"YachidaS_2019", "ThomasAM_2019_c"} <= set(cohorts)

    for cohort in cohorts:
        train_rows = mg.loc[mask & (mg["study_name"] != cohort)]
        kept = _prevalence_mean_filter(train_rows, feature_cols)
        assert len(kept) > 100, f"fold {cohort}: implausibly few species survived ({len(kept)})"
