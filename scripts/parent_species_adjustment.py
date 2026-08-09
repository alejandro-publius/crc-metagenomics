"""Test whether address candidates add signal beyond annotated parent species."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.io import mmread

from crc_lodo_bench.lodo import get_lodo_splits
from crc_lodo_bench.parent_adjustment import (
    ParentAdjustmentThresholds,
    evaluate_parent_adjustment,
    map_candidate_parents,
    summarize_parent_adjustment,
)


def main() -> None:
    output_dir = Path("results/intervention_readiness")
    annotations = pd.read_csv(output_dir / "candidate_annotations.csv")
    annotations = annotations[annotations["internal_nomination"].astype(bool)].copy()

    prefix = Path("data/raw/gene_families_selected")
    matrix = mmread(prefix.with_suffix(".mtx")).tocsr().T
    all_gene_ids = Path(f"{prefix}.features.txt").read_text(encoding="utf-8").splitlines()
    metadata = pd.read_csv(f"{prefix}.samples.csv").reset_index(drop=True)
    feature_index = {gene_id: index for index, gene_id in enumerate(all_gene_ids)}
    missing_genes = sorted(set(annotations["gene_id"]) - set(feature_index))
    if missing_genes:
        raise ValueError(f"nominated genes missing from matrix: {missing_genes}")
    gene_ids = sorted(annotations["gene_id"])
    selected_indices = [feature_index[gene_id] for gene_id in gene_ids]
    gene_values = pd.DataFrame(
        matrix[:, selected_indices].toarray(),
        index=metadata["sample_id"],
        columns=gene_ids,
    )

    species = pd.read_csv("data/processed/species_filtered.csv").set_index("sample_id")
    species = species.loc[metadata["sample_id"]]
    mapping = map_candidate_parents(annotations, species.columns.tolist())
    mapping_bytes = mapping.to_csv(index=False, lineterminator="\n").encode("utf-8")
    mapping_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
    (output_dir / "candidate_parent_species_mapping.csv").write_bytes(mapping_bytes)

    folds = {
        held_out: (train_idx, test_idx)
        for held_out, train_idx, test_idx, _excluded in get_lodo_splits(
            metadata, country_col="country"
        )
    }
    fold_results = evaluate_parent_adjustment(
        gene_values, species, metadata, mapping, folds
    )
    thresholds = ParentAdjustmentThresholds()
    summary = summarize_parent_adjustment(fold_results, thresholds=thresholds)
    fold_results.to_csv(output_dir / "parent_adjustment_fold_results.csv", index=False)
    summary.to_csv(output_dir / "parent_adjustment_summary.csv", index=False)
    audit = {
        "analysis_layer": "parent_species_adjustment",
        "mapping_rule": "exact archived representative binomial to exact MetaPhlAn species; no synonyms",
        "mapping_sha256": mapping_sha256,
        "n_candidates": len(gene_ids),
        "n_candidates_with_parent_match": int(
            mapping["n_matched_parent_species"].gt(0).sum()
        ),
        "n_candidates_passing": int(summary["parent_adjustment_gate"].eq("pass").sum()),
        "thresholds": thresholds.__dict__,
        "interpretation_boundary": (
            "Passing means added held-out discrimination beyond an annotated "
            "parent-species proxy. It does not establish the sequence's true "
            "taxonomic source, causality, specificity, or editability."
        ),
    }
    (output_dir / "parent_adjustment_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
