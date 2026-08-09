from __future__ import annotations

import numpy as np
import pandas as pd

from crc_lodo_bench.parent_adjustment import (
    ParentAdjustmentThresholds,
    evaluate_parent_adjustment,
    map_candidate_parents,
    summarize_parent_adjustment,
)


def test_parent_mapping_is_exact_and_does_not_infer_synonyms():
    annotations = pd.DataFrame(
        [
            {"gene_id": "g1", "organisms": "Bacteroides fragilis strain X"},
            {"gene_id": "g2", "organisms": "Agathobacter rectalis"},
        ]
    )
    columns = [
        "k__Bacteria|g__Bacteroides|s__Bacteroides_fragilis",
        "k__Bacteria|g__Eubacterium|s__Eubacterium_rectale",
    ]
    mapping = map_candidate_parents(annotations, columns).set_index("gene_id")

    assert mapping.loc["g1", "mapping_status"] == "exact_single_match"
    assert mapping.loc["g2", "mapping_status"] == "no_exact_match"


def test_parent_adjustment_detects_gene_signal_beyond_species():
    rows = []
    gene = []
    species = []
    folds = {}
    for cohort_index in range(7):
        start = len(rows)
        for sample_index, label in enumerate([0, 0, 0, 1, 1, 1]):
            rows.append(
                {
                    "sample_id": f"c{cohort_index}_{sample_index}",
                    "study_name": f"c{cohort_index}",
                    "country": f"country{cohort_index}",
                    "label": label,
                }
            )
            species.append([float(sample_index % 2)])
            gene.append([float(label)])
        test = list(range(start, start + 6))
        train = [index for index in range(42) if index not in test]
        folds[f"c{cohort_index}"] = (train, test)
    metadata = pd.DataFrame(rows)
    index = metadata["sample_id"]
    gene_values = pd.DataFrame(gene, index=index, columns=["g1"])
    species_values = pd.DataFrame(
        species,
        index=index,
        columns=["k__Bacteria|g__Example|s__Example_species"],
    )
    mapping = pd.DataFrame(
        [
            {
                "gene_id": "g1",
                "matched_parent_species_columns": (
                    "k__Bacteria|g__Example|s__Example_species"
                ),
            }
        ]
    )
    results = evaluate_parent_adjustment(
        gene_values, species_values, metadata, mapping, folds
    )
    summary = summarize_parent_adjustment(
        results,
        thresholds=ParentAdjustmentThresholds(
            min_evaluable_folds=7,
            min_positive_delta_fraction=0.70,
            min_median_delta_auc=0.02,
        ),
    )

    assert results["delta_auc"].gt(0).all()
    assert summary.loc[0, "parent_adjustment_gate"] == "pass"
