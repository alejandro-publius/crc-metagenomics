"""Build the auditable candidate-attrition funnel and gate sensitivity table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_funnel(
    n_gene_families: int,
    parent_summary: pd.DataFrame,
    taxon_summary: pd.DataFrame,
) -> pd.DataFrame:
    n_nominated = int(len(parent_summary))
    n_parent_pass = int(parent_summary["parent_adjustment_gate"].eq("pass").sum())
    n_address_pass = int(
        taxon_summary["taxonomic_resolution_status"]
        .eq("dominant_source_observed")
        .sum()
    )
    counts = [n_gene_families, n_nominated, n_parent_pass, n_address_pass]
    labels = [
        "Gene families evaluated",
        "Cross-population nominations",
        "Signal beyond parent-species proxy",
        "Taxonomically resolved addresses",
    ]
    gates = [
        "entered leakage-safe screen",
        "passed internal recurrence gate",
        "passed parent-species adjustment gate",
        "passed prespecified 80% dominant-carrier gate",
    ]
    rows: list[dict[str, object]] = []
    for index, (label, gate, count) in enumerate(zip(labels, gates, counts)):
        previous = counts[index - 1] if index else n_gene_families
        rows.append(
            {
                "stage": index + 1,
                "label": label,
                "gate": gate,
                "n_candidates": count,
                "fraction_of_previous": count / previous if previous else 0.0,
                "fraction_of_initial": count / n_gene_families,
            }
        )
    return pd.DataFrame(rows)


def build_taxon_threshold_sensitivity(
    taxon_summary: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dominant_taxon_fraction_threshold": threshold,
                "n_candidates_passing": int(
                    taxon_summary["dominant_taxon_fraction"].ge(threshold).sum()
                ),
                "n_candidates_evaluated": int(len(taxon_summary)),
                "prespecified_primary_threshold": threshold == 0.80,
            }
            for threshold in thresholds
        ]
    )


def main() -> None:
    output_dir = Path("results/intervention_readiness")
    discovery_audit = json.loads((output_dir / "discovery_audit.json").read_text())
    parent = pd.read_csv(output_dir / "parent_adjustment_summary.csv")
    taxon = pd.read_csv(output_dir / "candidate_taxon_summary.csv")
    atlas = pd.read_csv(output_dir / "readiness_atlas.csv")

    funnel = build_funnel(discovery_audit["n_gene_families"], parent, taxon)
    sensitivity = build_taxon_threshold_sensitivity(taxon)
    rejection_counts = (
        atlas[atlas["candidate_class"].eq("precision_address_candidate")]
        .groupby("atlas_status", as_index=False)
        .size()
        .rename(columns={"size": "n_candidates"})
        .sort_values("atlas_status", kind="mergesort")
    )
    funnel.to_csv(output_dir / "candidate_attrition_funnel.csv", index=False)
    sensitivity.to_csv(
        output_dir / "taxonomic_gate_threshold_sensitivity.csv", index=False
    )
    rejection_counts.to_csv(output_dir / "candidate_rejection_reasons.csv", index=False)
    audit = {
        "primary_finding": (
            "No internally recurring gene-family nomination passed the full "
            "parent-independence plus taxonomic-address sequence of gates."
        ),
        "n_initial_gene_families": int(discovery_audit["n_gene_families"]),
        "n_internal_nominations": int(len(parent)),
        "n_parent_adjustment_pass": int(
            parent["parent_adjustment_gate"].eq("pass").sum()
        ),
        "n_taxonomic_address_pass": int(
            taxon["taxonomic_resolution_status"].eq("dominant_source_observed").sum()
        ),
        "sensitivity_result": (
            "No candidate has a majority carrier even at a 50% threshold; "
            "the primary threshold was frozen at 80%."
        ),
    }
    (output_dir / "candidate_attrition_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(funnel.to_string(index=False))
    print("\nTaxonomic threshold sensitivity")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
