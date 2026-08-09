"""Run cross-fitted, country-aware discovery of CRC-associated gene families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.io import mmread

from crc_lodo_bench.discovery import (
    DiscoveryThresholds,
    build_cross_fitted_evidence,
    compute_gene_cohort_statistics,
    summarize_cross_fitted_candidates,
)
from crc_lodo_bench.lodo import get_lodo_splits


def load_matrix(prefix: Path):
    matrix = mmread(prefix.with_suffix(".mtx")).tocsr().T.toarray()
    features = Path(f"{prefix}.features.txt").read_text(encoding="utf-8").splitlines()
    samples = pd.read_csv(f"{prefix}.samples.csv").reset_index(drop=True)
    if matrix.shape != (len(samples), len(features)):
        raise ValueError("matrix dimensions disagree with samples/features")
    return matrix, features, samples


def training_cohort_map(metadata: pd.DataFrame) -> dict[str, list[str]]:
    folds: dict[str, list[str]] = {}
    for held_out, train_idx, _test_idx, _excluded in get_lodo_splits(
        metadata, country_col="country"
    ):
        folds[held_out] = sorted(metadata.iloc[train_idx]["study_name"].unique())
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-prefix",
        type=Path,
        default=Path("data/raw/gene_families_selected"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/intervention_readiness")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix, features, metadata = load_matrix(args.matrix_prefix)
    thresholds = DiscoveryThresholds()
    cohort_statistics = compute_gene_cohort_statistics(matrix, features, metadata)
    training_map = training_cohort_map(metadata)
    selections, evidence = build_cross_fitted_evidence(
        cohort_statistics, training_map, thresholds=thresholds
    )
    summary = summarize_cross_fitted_candidates(evidence, thresholds=thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_statistics.to_csv(
        args.output_dir / "gene_cohort_statistics.csv.gz", index=False
    )
    selections.to_csv(args.output_dir / "discovery_fold_selections.csv", index=False)
    evidence.to_csv(args.output_dir / "discovery_cross_fitted_evidence.csv", index=False)
    summary.to_csv(args.output_dir / "discovery_candidate_summary.csv", index=False)
    audit = {
        "analysis_layer": "cross_fitted_gene_family_discovery",
        "matrix_prefix": str(args.matrix_prefix),
        "n_samples": len(metadata),
        "n_gene_families": len(features),
        "n_outer_folds": len(training_map),
        "n_unique_selected_gene_families": int(selections["gene_id"].nunique()),
        "n_internal_nominations": int(summary["internal_nomination"].sum()),
        "thresholds": thresholds.__dict__,
        "selection_boundary": (
            "Each outer fold uses only its training cohorts. Internal nomination "
            "is not external confirmation and does not establish causality."
        ),
    }
    (args.output_dir / "discovery_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
