from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_candidate_taxa import summarize_candidate_taxa  # noqa: E402


def test_dominant_taxon_and_zero_filled_prevalence_are_explicit():
    metadata_rows = []
    for cohort in ["c1", "c2"]:
        for index, label in enumerate([0, 0, 1, 1]):
            metadata_rows.append(
                {
                    "sample_id": f"{cohort}_{index}",
                    "study_name": cohort,
                    "label": label,
                    "country": cohort,
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    long_table = pd.DataFrame(
        [
            {
                "sample_id": "c1_2",
                "study_name": "c1",
                "gene_id": "g1",
                "taxon": "Species_A",
                "abundance": 4.0,
            },
            {
                "sample_id": "c2_2",
                "study_name": "c2",
                "gene_id": "g1",
                "taxon": "Species_A",
                "abundance": 4.0,
            },
            {
                "sample_id": "c1_0",
                "study_name": "c1",
                "gene_id": "g1",
                "taxon": "Species_B",
                "abundance": 1.0,
            },
            {
                "sample_id": "c1_2",
                "study_name": "c1",
                "gene_id": "g1",
                "taxon": "unstratified",
                "abundance": 5.0,
            },
        ]
    )
    summary, evidence = summarize_candidate_taxa(long_table, metadata)
    row = summary.iloc[0]

    assert row["dominant_taxon"] == "Species_A"
    assert row["dominant_taxon_fraction"] == 8 / 9
    assert row["taxonomic_resolution_status"] == "dominant_source_observed"
    assert evidence["crc_prevalence"].eq(0.5).all()
    assert evidence["control_prevalence"].eq(0.0).all()


def test_parent_fraction_includes_strain_suffixed_taxa():
    metadata = pd.DataFrame(
        [
            {"sample_id": "a", "study_name": "c1", "label": 0, "country": "x"},
            {"sample_id": "b", "study_name": "c1", "label": 1, "country": "x"},
        ]
    )
    long_table = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "study_name": "c1",
                "gene_id": "g1",
                "taxon": "g__Bacteroides.s__Bacteroides_fragilis_CAG_47",
                "abundance": 3.0,
            },
            {
                "sample_id": "b",
                "study_name": "c1",
                "gene_id": "g1",
                "taxon": "g__Bacteroides.s__Bacteroides_ovatus",
                "abundance": 1.0,
            },
        ]
    )
    parent_mapping = pd.DataFrame(
        [
            {
                "gene_id": "g1",
                "matched_parent_species_columns": "k__Bacteria|s__Bacteroides_fragilis",
                "mapping_status": "exact_single_match",
            }
        ]
    )
    summary, _ = summarize_candidate_taxa(long_table, metadata, parent_mapping)

    assert summary.loc[0, "parent_taxon_detected"]
    assert summary.loc[0, "parent_taxon_fraction"] == 0.75
    assert summary.loc[0, "best_parent_taxon_rank"] == 1
