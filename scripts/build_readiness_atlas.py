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


def build_atlas(
    known: pd.DataFrame,
    discovered: pd.DataFrame,
    parent_adjustment: pd.DataFrame | None = None,
    taxon_resolution: pd.DataFrame | None = None,
    mechanism_integrity: pd.DataFrame | None = None,
    guide_conservation: pd.DataFrame | None = None,
    guide_specificity: pd.DataFrame | None = None,
    human_isolate_conservation: pd.DataFrame | None = None,
    guide_exception_resolution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    parent_lookup = (
        parent_adjustment.set_index("gene_id")
        if parent_adjustment is not None
        else None
    )
    taxon_lookup = (
        taxon_resolution.set_index("gene_id") if taxon_resolution is not None else None
    )
    integrity_lookup = (
        mechanism_integrity.set_index("target_id")
        if mechanism_integrity is not None
        else None
    )
    conservation_lookup = (
        guide_conservation.groupby("target_id")
        if guide_conservation is not None
        else None
    )
    specificity_lookup = (
        guide_specificity.groupby("target_id")
        if guide_specificity is not None
        else None
    )
    human_conservation_lookup = (
        human_isolate_conservation.groupby("target_id")
        if human_isolate_conservation is not None
        else None
    )
    known_rows: list[dict[str, object]] = []
    for row in known.itertuples(index=False):
        integrity_status = (
            str(integrity_lookup.loc[row.target_id, "mechanism_integrity_status"])
            if integrity_lookup is not None and row.target_id in integrity_lookup.index
            else "not_yet_assessed"
        )
        conservation_status = row.conservation_status
        conservation_detail = "No frozen sequence-conservation result is available."
        if (
            conservation_lookup is not None
            and row.target_id in conservation_lookup.groups
        ):
            target_conservation = conservation_lookup.get_group(row.target_id)
            n_guides = int(target_conservation["guide_id"].nunique())
            n_pass = int(
                target_conservation["pilot_conservation_gate"].eq("pass").sum()
            )
            n_genomes = int(target_conservation["n_genomes"].max())
            if n_guides > 0 and n_pass == n_guides:
                conservation_status = "reference_panel_pass_human_diversity_pending"
            else:
                conservation_status = "reference_panel_not_passed"
            conservation_detail = (
                f"{n_pass}/{n_guides} published guides passed exact-site coverage "
                f"and uniqueness in a frozen {n_genomes}-genome pks-positive "
                "reference panel; broader human-isolate diversity remains pending."
            )
        if (
            human_conservation_lookup is not None
            and row.target_id in human_conservation_lookup.groups
        ):
            target_human = human_conservation_lookup.get_group(row.target_id)
            n_human_guides = int(target_human["guide_id"].nunique())
            n_human_pass = int(
                target_human["human_isolate_conservation_gate"].eq("pass").sum()
            )
            n_human_genomes = int(target_human["n_genomes"].max())
            if n_human_guides > 0 and n_human_pass == n_human_guides:
                conservation_status = (
                    "human_isolate_panel_pass_global_diversity_pending"
                )
            else:
                conservation_status = "human_isolate_panel_not_passed"
            guide_details = []
            for guide in target_human.sort_values(
                ["benchmark_role", "guide_id"]
            ).itertuples(index=False):
                guide_details.append(
                    f"{guide.guide_id}: {int(guide.n_genomes_covered)}/"
                    f"{int(guide.n_genomes)} covered and "
                    f"{int(guide.n_genomes_unique_site)}/"
                    f"{int(guide.n_genomes)} unique-site"
                )
            conservation_detail = conservation_detail.replace(
                "broader human-isolate diversity remains pending.",
                "the expanded human-isolate gate was evaluated separately.",
            )
            conservation_detail = (
                f"{conservation_detail} Expanded audit: {n_human_pass}/"
                f"{n_human_guides} guides passed the frozen {n_human_genomes}-"
                f"human-isolate panel ({'; '.join(guide_details)}). Global "
                "pks-positive diversity and assembly-independent confirmation "
                "remain pending."
            )
        if row.target_id == "colibactin" and guide_exception_resolution is not None:
            exception_lookup = guide_exception_resolution.set_index("case_id")
            required_cases = {"jml024_duplicate", "upec79_absent"}
            if required_cases.issubset(exception_lookup.index):
                upec = exception_lookup.loc["upec79_absent"]
                jml = exception_lookup.loc["jml024_duplicate"]
                conservation_detail = (
                    f"{conservation_detail} Frozen source-read follow-up resolved "
                    f"UPEC79 as `{upec.resolution_status}` with "
                    f"{int(upec.primary_exact_supporting_reads)} exact primary-site "
                    f"reads. JML024 remains `{jml.resolution_status}` "
                    f"({jml.support_metric})."
                )
        specificity_status = row.specificity_status
        specificity_detail = "No frozen protected-reference result is available."
        if (
            specificity_lookup is not None
            and row.target_id in specificity_lookup.groups
        ):
            target_specificity = specificity_lookup.get_group(row.target_id)
            n_guides = int(target_specificity["guide_id"].nunique())
            n_pass = int(
                target_specificity["protected_reference_pilot_gate"].eq("pass").sum()
            )
            n_references = int(target_specificity["n_references"].max())
            n_flagged = int(target_specificity["n_flagged_sites"].sum())
            primary = target_specificity[
                target_specificity["benchmark_role"].eq("primary")
            ]
            primary_pass = (
                len(primary) == 1
                and primary["protected_reference_pilot_gate"].eq("pass").all()
            )
            if n_guides > 0 and n_pass == n_guides:
                specificity_status = (
                    "protected_reference_pilot_pass_broader_scope_pending"
                )
            elif primary_pass:
                specificity_status = "primary_guide_pilot_pass_secondary_guide_flagged"
            else:
                specificity_status = "protected_reference_pilot_not_passed"
            specificity_detail = (
                f"{n_pass}/{n_guides} published guides passed a frozen "
                f"{n_references}-reference near-match pilot; {n_flagged} flagged "
                "sites were retained. Broader strain and platform-specific "
                "assessment remains pending."
            )
        if row.editability_evidence_status == "targeted_in_vivo_crispri_preprint":
            if (
                conservation_status
                == "human_isolate_panel_pass_global_diversity_pending"
                and specificity_status
                == "primary_guide_pilot_pass_secondary_guide_flagged"
            ):
                known_atlas_status = (
                    "literature_priority_primary_guide_global_and_platform_"
                    "validation_pending"
                )
            elif (
                conservation_status
                == "human_isolate_panel_pass_global_diversity_pending"
                and specificity_status
                == "protected_reference_pilot_pass_broader_scope_pending"
            ):
                known_atlas_status = (
                    "literature_priority_global_and_platform_validation_pending"
                )
            elif (
                conservation_status == "reference_panel_pass_human_diversity_pending"
                and specificity_status
                == "protected_reference_pilot_pass_broader_scope_pending"
            ):
                known_atlas_status = (
                    "literature_priority_human_isolate_and_platform_validation_pending"
                )
            elif (
                conservation_status == "reference_panel_pass_human_diversity_pending"
                and specificity_status
                == "primary_guide_pilot_pass_secondary_guide_flagged"
            ):
                known_atlas_status = (
                    "literature_priority_primary_guide_human_isolate_validation_pending"
                )
            elif conservation_status == "reference_panel_pass_human_diversity_pending":
                known_atlas_status = (
                    "literature_priority_specificity_and_human_diversity_pending"
                )
            else:
                known_atlas_status = "literature_priority_sequence_gates_pending"
        elif row.causal_evidence_status != "pending_structured_review":
            known_atlas_status = "mechanistically_supported_delivery_pending"
        else:
            known_atlas_status = "benchmark_incomplete"
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
                "mechanism_integrity_status": integrity_status,
                "parent_species_independence_status": "not_applicable_effector_benchmark",
                "parent_species_independence_detail": (
                    "Not applicable: the known-effect benchmark is defined by a "
                    "prespecified mechanism rather than a discovered address family."
                ),
                "taxonomic_source_status": "not_applicable_effector_benchmark",
                "taxonomic_source_detail": (
                    "Not applicable to this benchmark-level association result."
                ),
                "conservation_status": conservation_status,
                "conservation_detail": conservation_detail,
                "specificity_status": specificity_status,
                "specificity_detail": specificity_detail,
                "editability_status": row.editability_evidence_status,
                "external_gene_confirmation_status": "not_yet_assessed",
                "atlas_status": known_atlas_status,
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
        parent_status = "not_yet_assessed"
        parent_detail = "parent-species independence has not been evaluated"
        if parent_lookup is not None and row.gene_id in parent_lookup.index:
            parent = parent_lookup.loc[row.gene_id]
            parent_status = str(parent.parent_adjustment_gate)
            if parent_status == "pass":
                parent_detail = (
                    f"median held-out AUC gain {parent.median_delta_auc:.3f}; "
                    f"positive in {parent.positive_delta_fraction:.0%} of evaluable folds"
                )
            elif parent_status == "not_passed":
                parent_detail = (
                    f"did not pass: median held-out AUC gain "
                    f"{parent.median_delta_auc:.3f}; positive in "
                    f"{parent.positive_delta_fraction:.0%} of evaluable folds"
                )
            else:
                parent_detail = (
                    "not evaluable because no exact parent-species match was frozen"
                )

        taxon_status = "not_yet_assessed"
        taxon_detail = "taxon-resolved carrier evidence has not been evaluated"
        if taxon_lookup is not None and row.gene_id in taxon_lookup.index:
            taxon = taxon_lookup.loc[row.gene_id]
            taxon_status = str(taxon.taxonomic_resolution_status)
            taxon_detail = (
                f"dominant carrier {taxon.dominant_taxon} accounted for "
                f"{taxon.dominant_taxon_fraction:.1%} across "
                f"{int(taxon.n_detected_taxa)} detected taxa"
            )

        atlas_status = "internal_address_nomination"
        function_or_clade = "not_yet_established"
        claim_boundary = (
            "A recurring gene-family association and possible genomic address; "
            "not a causal gene or guide sequence."
        )
        if parent_status == "not_passed":
            atlas_status = "rejected_no_parent_independent_signal"
            function_or_clade = "not_assessed_parent_gate_failed"
            claim_boundary = (
                "The family recurs across cohorts but does not add enough held-out "
                "information beyond its annotated parent-species proxy."
            )
        elif parent_status == "not_evaluable":
            atlas_status = "unresolved_no_exact_parent_mapping"
        elif parent_status == "pass" and taxon_status == "mixed_taxonomic_sources":
            atlas_status = "rejected_mixed_taxonomic_sources"
            function_or_clade = "failed_taxonomic_address_gate"
            claim_boundary = (
                "The family carries parent-independent CRC signal, but its abundance "
                "is distributed across multiple taxa; it is not a direct editing address."
            )
        elif parent_status == "pass" and taxon_status == "dominant_source_observed":
            atlas_status = "taxonomic_source_observed_clade_link_pending"
            function_or_clade = "harmful_clade_link_not_yet_established"
        elif parent_status == "pass":
            atlas_status = "parent_independent_signal_taxon_resolution_pending"
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
                "function_or_clade_link_status": function_or_clade,
                "mechanism_integrity_status": "not_applicable_address_candidate",
                "parent_species_independence_status": parent_status,
                "parent_species_independence_detail": parent_detail,
                "taxonomic_source_status": taxon_status,
                "taxonomic_source_detail": taxon_detail,
                "conservation_status": "not_yet_assessed",
                "conservation_detail": (
                    "Not run because the candidate did not pass every preceding "
                    "sequential address gate."
                ),
                "specificity_status": "not_yet_assessed",
                "specificity_detail": (
                    "Not run because the candidate did not pass every preceding "
                    "sequential address gate."
                ),
                "editability_status": "platform_dependent_not_yet_assessed",
                "external_gene_confirmation_status": row.external_confirmation_status,
                "atlas_status": atlas_status,
                "claim_boundary": claim_boundary,
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
        raise ValueError(
            "a candidate with an incomplete mandatory gate was marked ready"
        )
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
| Signal beyond annotated parent species | {candidate.parent_species_independence_status} |
| Taxon-resolved carrier | {candidate.taxonomic_source_status} |
| Biological function or harmful-clade link | {candidate.function_or_clade_link_status} |
| Assay representation of required mechanism | {candidate.mechanism_integrity_status} |
| Sequence conservation | {candidate.conservation_status} |
| Specificity against protected organisms/human sequence | {candidate.specificity_status} |
| Editing and delivery feasibility | {candidate.editability_status} |
| External gene-level confirmation | {candidate.external_gene_confirmation_status} |

## Claim boundary

{candidate.claim_boundary}

## Address-resolution detail

{getattr(candidate, "parent_species_independence_detail", "Not applicable.")}

{getattr(candidate, "taxonomic_source_detail", "Not applicable.")}

## Sequence-conservation detail

{getattr(candidate, "conservation_detail", "Not yet assessed.")}

## Protected-reference specificity detail

{getattr(candidate, "specificity_detail", "Not yet assessed.")}

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
        "--guide-conservation",
        type=Path,
        default=Path(
            "results/intervention_readiness/colibactin_guide_conservation_summary.csv"
        ),
    )
    parser.add_argument(
        "--guide-specificity",
        type=Path,
        default=Path(
            "results/intervention_readiness/colibactin_specificity_pilot_summary.csv"
        ),
    )
    parser.add_argument(
        "--human-isolate-conservation",
        type=Path,
        default=Path(
            "results/intervention_readiness/"
            "colibactin_human_isolate_conservation_summary.csv"
        ),
    )
    parser.add_argument(
        "--guide-exception-resolution",
        type=Path,
        default=Path(
            "results/intervention_readiness/colibactin_exception_resolution.csv"
        ),
    )
    parser.add_argument(
        "--discovered",
        type=Path,
        default=Path("results/intervention_readiness/candidate_annotations.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/intervention_readiness")
    )
    parser.add_argument(
        "--parent-adjustment",
        type=Path,
        default=Path("results/intervention_readiness/parent_adjustment_summary.csv"),
    )
    parser.add_argument(
        "--taxon-resolution",
        type=Path,
        default=Path("results/intervention_readiness/candidate_taxon_summary.csv"),
    )
    parser.add_argument(
        "--mechanism-integrity",
        type=Path,
        default=Path(
            "results/intervention_readiness/known_target_mechanism_integrity.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    known = pd.read_csv(args.known)
    discovered = pd.read_csv(args.discovered)
    parent_adjustment = pd.read_csv(args.parent_adjustment)
    taxon_resolution = pd.read_csv(args.taxon_resolution)
    mechanism_integrity = pd.read_csv(args.mechanism_integrity)
    guide_conservation = pd.read_csv(args.guide_conservation)
    guide_specificity = pd.read_csv(args.guide_specificity)
    human_isolate_conservation = pd.read_csv(args.human_isolate_conservation)
    guide_exception_resolution = pd.read_csv(args.guide_exception_resolution)
    atlas = build_atlas(
        known,
        discovered,
        parent_adjustment,
        taxon_resolution,
        mechanism_integrity,
        guide_conservation,
        guide_specificity,
        human_isolate_conservation,
        guide_exception_resolution,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(args.output_dir / "readiness_atlas.csv", index=False)
    write_dossiers(atlas, args.output_dir / "dossiers")
    print(atlas.to_string(index=False))


if __name__ == "__main__":
    main()
