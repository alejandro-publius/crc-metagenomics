"""Join known effectors and discovery leads into an explicit-gate atlas."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not cleaned:
        raise ValueError(f"cannot derive a filename from {value!r}")
    return cleaned


def build_atlas(known: pd.DataFrame, discovered: pd.DataFrame) -> pd.DataFrame:
    known_rows: list[dict[str, object]] = []
    for row in known.itertuples(index=False):
        known_rows.append(
            {
                "candidate_id": row.target_id,
                "display_name": row.display_name,
                "candidate_class": "effector_benchmark",
                "human_recurrence_status": row.cross_population_gate,
                "human_recurrence_detail": (
                    f"{row.n_crc_enriched_cohorts}/{row.n_cohorts_evaluable} "
                    f"evaluable cohorts CRC-enriched; median AUC "
                    f"{row.median_association_auc:.3f}"
                    if row.n_cohorts_evaluable
                    else "not evaluable in the current gene-family assay"
                ),
                "function_or_clade_link_status": row.causal_evidence_status,
                "conservation_status": row.conservation_status,
                "specificity_status": row.specificity_status,
                "editability_status": row.editability_evidence_status,
                "external_gene_confirmation_status": "not_yet_assessed",
                "atlas_status": "benchmark_incomplete",
                "claim_boundary": (
                    "A literature-motivated mechanism; human association does not "
                    "by itself establish a safe editing target."
                ),
            }
        )

    discovered_rows: list[dict[str, object]] = []
    for row in discovered.itertuples(index=False):
        if not bool(row.internal_nomination):
            continue
        display_name = str(row.protein_names).split(";")[0]
        discovered_rows.append(
            {
                "candidate_id": row.gene_id,
                "display_name": display_name,
                "candidate_class": "precision_address_candidate",
                "human_recurrence_status": "internal_cross_fitted_pass",
                "human_recurrence_detail": (
                    f"selected in {int(row.n_outer_selections)} outer folds; "
                    f"{row.heldout_crc_enriched_fraction:.0%} held-out direction "
                    f"consistency; median held-out AUC {row.heldout_median_auc:.3f}"
                ),
                "function_or_clade_link_status": "not_yet_established",
                "conservation_status": "not_yet_assessed",
                "specificity_status": "not_yet_assessed",
                "editability_status": "platform_dependent_not_yet_assessed",
                "external_gene_confirmation_status": row.external_confirmation_status,
                "atlas_status": "internal_address_nomination",
                "claim_boundary": (
                    "A recurring gene-family association and possible genomic "
                    "address; not a causal gene or guide sequence."
                ),
            }
        )

    atlas = pd.DataFrame(known_rows + discovered_rows)
    if atlas.empty:
        return atlas
    if atlas["candidate_id"].duplicated().any():
        raise ValueError("atlas candidate_id values must be unique")
    forbidden_ready = atlas["atlas_status"].eq("experiment_ready") & (
        atlas[
            [
                "conservation_status",
                "specificity_status",
                "external_gene_confirmation_status",
            ]
        ]
        .ne("pass")
        .any(axis=1)
    )
    if forbidden_ready.any():
        raise ValueError("a candidate with an incomplete mandatory gate was marked ready")
    return atlas.sort_values(
        ["candidate_class", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def write_dossiers(atlas: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in atlas.itertuples(index=False):
        content = f"""# {candidate.display_name}

- **Candidate ID:** `{candidate.candidate_id}`
- **Class:** {candidate.candidate_class}
- **Current atlas status:** {candidate.atlas_status}

## Why it is in the atlas

{candidate.human_recurrence_detail}

## Evidence gates

| Gate | Current status |
|---|---|
| Human cross-population recurrence | {candidate.human_recurrence_status} |
| Biological function or harmful-clade link | {candidate.function_or_clade_link_status} |
| Sequence conservation | {candidate.conservation_status} |
| Specificity against protected organisms/human sequence | {candidate.specificity_status} |
| Editing and delivery feasibility | {candidate.editability_status} |
| External gene-level confirmation | {candidate.external_gene_confirmation_status} |

## Claim boundary

{candidate.claim_boundary}

## Experiment-enabling next evidence

Resolve every non-passing gate with versioned public data or a prespecified
laboratory experiment. This dossier must not be presented as a treatment
recommendation or institutional endorsement.
"""
        (output_dir / f"{safe_filename(candidate.candidate_id)}.md").write_text(
            content, encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--known",
        type=Path,
        default=Path("results/intervention_readiness/known_target_summary.csv"),
    )
    parser.add_argument(
        "--discovered",
        type=Path,
        default=Path("results/intervention_readiness/candidate_annotations.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/intervention_readiness")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    known = pd.read_csv(args.known)
    discovered = pd.read_csv(args.discovered)
    atlas = build_atlas(known, discovered)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(args.output_dir / "readiness_atlas.csv", index=False)
    write_dossiers(atlas, args.output_dir / "dossiers")
    print(atlas.to_string(index=False))


if __name__ == "__main__":
    main()
