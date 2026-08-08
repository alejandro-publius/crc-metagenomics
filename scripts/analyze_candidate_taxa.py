"""Summarize the taxonomic sources of nominated gene-family associations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_candidate_taxa(
    long_table: pd.DataFrame, sample_manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "study_name", "gene_id", "taxon", "abundance"}
    missing = sorted(required - set(long_table.columns))
    if missing:
        raise ValueError(f"taxon-resolved table is missing columns: {missing}")
    metadata_required = {"sample_id", "study_name", "label", "country"}
    metadata_missing = sorted(metadata_required - set(sample_manifest.columns))
    if metadata_missing:
        raise ValueError(f"sample manifest is missing columns: {metadata_missing}")
    if sample_manifest["sample_id"].duplicated().any():
        raise ValueError("sample manifest contains duplicate sample IDs")
    if (long_table["abundance"] <= 0).any() or not np.isfinite(
        long_table["abundance"]
    ).all():
        raise ValueError("long-table rows must contain finite positive abundance")

    stratified = long_table[long_table["taxon"].ne("unstratified")].copy()
    source_totals = (
        stratified.groupby(["gene_id", "taxon"], as_index=False)["abundance"]
        .sum()
        .sort_values(
            ["gene_id", "abundance", "taxon"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    if source_totals.empty:
        raise ValueError("no stratified taxon rows were available")
    overall_totals = source_totals.groupby("gene_id")["abundance"].sum().to_dict()
    dominant = source_totals.groupby("gene_id", sort=False).head(1).copy()
    dominant["dominant_taxon_fraction"] = dominant.apply(
        lambda row: row["abundance"] / overall_totals[row["gene_id"]], axis=1
    )
    taxon_counts = source_totals.groupby("gene_id")["taxon"].nunique().to_dict()
    dominant["n_detected_taxa"] = dominant["gene_id"].map(taxon_counts)
    dominant = dominant.rename(
        columns={"taxon": "dominant_taxon", "abundance": "dominant_total_abundance"}
    )

    evidence_rows: list[dict[str, object]] = []
    for candidate in dominant.itertuples(index=False):
        detected = stratified[
            stratified["gene_id"].eq(candidate.gene_id)
            & stratified["taxon"].eq(candidate.dominant_taxon)
        ]
        detected_samples = set(detected["sample_id"])
        for cohort, metadata in sample_manifest.groupby("study_name", sort=True):
            metadata = metadata[metadata["label"].isin([0, 1])]
            crc = metadata[metadata["label"].eq(1)]
            control = metadata[metadata["label"].eq(0)]
            if crc.empty or control.empty:
                continue
            crc_prevalence = float(crc["sample_id"].isin(detected_samples).mean())
            control_prevalence = float(
                control["sample_id"].isin(detected_samples).mean()
            )
            difference = crc_prevalence - control_prevalence
            evidence_rows.append(
                {
                    "gene_id": candidate.gene_id,
                    "dominant_taxon": candidate.dominant_taxon,
                    "cohort": cohort,
                    "country": ";".join(
                        sorted(metadata["country"].dropna().astype(str).unique())
                    ),
                    "n_crc": len(crc),
                    "n_control": len(control),
                    "crc_prevalence": crc_prevalence,
                    "control_prevalence": control_prevalence,
                    "prevalence_difference": difference,
                    "direction": (
                        "crc_enriched"
                        if difference > 0
                        else "control_enriched" if difference < 0 else "tie"
                    ),
                }
            )
    evidence = pd.DataFrame(evidence_rows)
    direction_summary = (
        evidence.groupby("gene_id")
        .agg(
            n_taxon_evidence_cohorts=("cohort", "nunique"),
            n_taxon_crc_enriched_cohorts=(
                "direction", lambda values: int(np.sum(values == "crc_enriched"))
            ),
            median_taxon_prevalence_difference=("prevalence_difference", "median"),
        )
        .reset_index()
    )
    direction_summary["taxon_crc_enriched_fraction"] = (
        direction_summary["n_taxon_crc_enriched_cohorts"]
        / direction_summary["n_taxon_evidence_cohorts"]
    )
    summary = dominant.merge(direction_summary, on="gene_id", how="left")
    summary["taxonomic_resolution_status"] = np.where(
        summary["dominant_taxon_fraction"] >= 0.80,
        "dominant_source_observed",
        "mixed_taxonomic_sources",
    )
    return (
        summary.sort_values("gene_id", kind="mergesort").reset_index(drop=True),
        evidence.sort_values(["gene_id", "cohort"], kind="mergesort").reset_index(
            drop=True
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--long-table",
        type=Path,
        default=Path("data/raw/intervention_candidates_stratified.csv.gz"),
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=Path("data/raw/gene_families_selected.samples.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/intervention_readiness")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long_table = pd.read_csv(args.long_table)
    sample_manifest = pd.read_csv(args.sample_manifest)
    summary, evidence = summarize_candidate_taxa(long_table, sample_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "candidate_taxon_summary.csv", index=False)
    evidence.to_csv(args.output_dir / "candidate_taxon_cohort_evidence.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
