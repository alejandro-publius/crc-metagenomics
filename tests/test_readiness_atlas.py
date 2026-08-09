from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_readiness_atlas import build_atlas, safe_filename  # noqa: E402


def test_atlas_keeps_effector_and_address_roles_distinct():
    known = pd.DataFrame(
        [
            {
                "target_id": "toxin",
                "display_name": "Example toxin",
                "n_crc_enriched_cohorts": 4,
                "n_cohorts_evaluable": 5,
                "median_association_auc": 0.60,
                "cross_population_gate": "pass",
                "causal_evidence_status": "pending_structured_review",
                "conservation_status": "not_yet_assessed",
                "specificity_status": "not_yet_assessed",
                "editability_evidence_status": "pending_structured_review",
            }
        ]
    )
    discovered = pd.DataFrame(
        [
            {
                "gene_id": "UniRef90_A1",
                "protein_names": "Uncharacterized protein",
                "internal_nomination": True,
                "n_outer_selections": 8,
                "heldout_crc_enriched_fraction": 0.75,
                "heldout_median_auc": 0.58,
                "external_confirmation_status": "not_yet_assessed",
            }
        ]
    )
    atlas = build_atlas(known, discovered).set_index("candidate_id")

    assert atlas.loc["toxin", "candidate_class"] == "effector_benchmark"
    assert atlas.loc["UniRef90_A1", "candidate_class"] == "precision_address_candidate"
    assert not atlas["atlas_status"].eq("experiment_ready").any()


def test_non_nominated_discovery_rows_do_not_enter_atlas():
    known = pd.DataFrame(
        columns=[
            "target_id",
            "display_name",
            "n_crc_enriched_cohorts",
            "n_cohorts_evaluable",
            "median_association_auc",
            "cross_population_gate",
            "causal_evidence_status",
            "conservation_status",
            "specificity_status",
            "editability_evidence_status",
        ]
    )
    discovered = pd.DataFrame(
        [
            {
                "gene_id": "UniRef90_A1",
                "protein_names": "Example",
                "internal_nomination": False,
                "n_outer_selections": 1,
                "heldout_crc_enriched_fraction": 1.0,
                "heldout_median_auc": 1.0,
                "external_confirmation_status": "not_yet_assessed",
            }
        ]
    )
    atlas = build_atlas(known, discovered)
    assert atlas.empty


def test_safe_filename_removes_path_characters():
    assert safe_filename("a/b c") == "a_b_c"


def test_mixed_taxonomic_sources_reject_an_address_candidate():
    known = pd.DataFrame()
    discovered = pd.DataFrame(
        [
            {
                "gene_id": "UniRef90_A1",
                "protein_names": "Example",
                "internal_nomination": True,
                "n_outer_selections": 8,
                "heldout_crc_enriched_fraction": 0.75,
                "heldout_median_auc": 0.58,
                "external_confirmation_status": "not_yet_assessed",
            }
        ]
    )
    parent = pd.DataFrame(
        [
            {
                "gene_id": "UniRef90_A1",
                "parent_adjustment_gate": "pass",
                "median_delta_auc": 0.03,
                "positive_delta_fraction": 0.8,
            }
        ]
    )
    taxon = pd.DataFrame(
        [
            {
                "gene_id": "UniRef90_A1",
                "taxonomic_resolution_status": "mixed_taxonomic_sources",
                "dominant_taxon": "Species_A",
                "dominant_taxon_fraction": 0.4,
                "n_detected_taxa": 4,
            }
        ]
    )

    atlas = build_atlas(known, discovered, parent, taxon)
    assert atlas.loc[0, "atlas_status"] == "rejected_mixed_taxonomic_sources"
    assert (
        atlas.loc[0, "function_or_clade_link_status"] == "failed_taxonomic_address_gate"
    )


def test_known_target_integrates_frozen_sequence_pilots():
    known = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "display_name": "Colibactin",
                "n_crc_enriched_cohorts": 9,
                "n_cohorts_evaluable": 10,
                "median_association_auc": 0.54,
                "cross_population_gate": "not_passed",
                "causal_evidence_status": (
                    "human_signature_plus_animal_targeted_perturbation"
                ),
                "conservation_status": "not_yet_assessed",
                "specificity_status": "not_yet_assessed",
                "editability_evidence_status": "targeted_in_vivo_crispri_preprint",
            }
        ]
    )
    conservation = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": guide,
                "n_genomes": 7,
                "pilot_conservation_gate": "pass",
            }
            for guide in ["g1", "g2"]
        ]
    )
    specificity = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": guide,
                "benchmark_role": role,
                "n_references": 11,
                "n_flagged_sites": 0,
                "protected_reference_pilot_gate": "pass",
            }
            for guide, role in [("g1", "primary"), ("g2", "secondary")]
        ]
    )
    atlas = build_atlas(
        known,
        pd.DataFrame(),
        guide_conservation=conservation,
        guide_specificity=specificity,
    ).set_index("candidate_id")
    assert (
        atlas.loc["colibactin", "specificity_status"]
        == "protected_reference_pilot_pass_broader_scope_pending"
    )
    assert (
        atlas.loc["colibactin", "atlas_status"]
        == "literature_priority_human_isolate_and_platform_validation_pending"
    )


def test_primary_guide_can_advance_when_secondary_guide_is_flagged():
    known = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "display_name": "Colibactin",
                "n_crc_enriched_cohorts": 9,
                "n_cohorts_evaluable": 10,
                "median_association_auc": 0.54,
                "cross_population_gate": "not_passed",
                "causal_evidence_status": (
                    "human_signature_plus_animal_targeted_perturbation"
                ),
                "conservation_status": "not_yet_assessed",
                "specificity_status": "not_yet_assessed",
                "editability_evidence_status": "targeted_in_vivo_crispri_preprint",
            }
        ]
    )
    conservation = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": guide,
                "n_genomes": 7,
                "pilot_conservation_gate": "pass",
            }
            for guide in ["primary", "secondary"]
        ]
    )
    specificity = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": "primary",
                "benchmark_role": "primary",
                "n_references": 11,
                "n_flagged_sites": 0,
                "protected_reference_pilot_gate": "pass",
            },
            {
                "target_id": "colibactin",
                "guide_id": "secondary",
                "benchmark_role": "secondary",
                "n_references": 11,
                "n_flagged_sites": 5,
                "protected_reference_pilot_gate": "not_passed",
            },
        ]
    )
    atlas = build_atlas(
        known,
        pd.DataFrame(),
        guide_conservation=conservation,
        guide_specificity=specificity,
    ).set_index("candidate_id")
    assert (
        atlas.loc["colibactin", "specificity_status"]
        == "primary_guide_pilot_pass_secondary_guide_flagged"
    )
    assert (
        atlas.loc["colibactin", "atlas_status"]
        == "literature_priority_primary_guide_human_isolate_validation_pending"
    )


def test_human_isolate_panel_records_conservation_specificity_tradeoff():
    known = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "display_name": "Colibactin",
                "n_crc_enriched_cohorts": 9,
                "n_cohorts_evaluable": 10,
                "median_association_auc": 0.54,
                "cross_population_gate": "not_passed",
                "causal_evidence_status": (
                    "human_signature_plus_animal_targeted_perturbation"
                ),
                "conservation_status": "not_yet_assessed",
                "specificity_status": "not_yet_assessed",
                "editability_evidence_status": "targeted_in_vivo_crispri_preprint",
            }
        ]
    )
    conservation = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": guide,
                "n_genomes": 7,
                "pilot_conservation_gate": "pass",
            }
            for guide in ["primary", "secondary"]
        ]
    )
    specificity = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": "primary",
                "benchmark_role": "primary",
                "n_references": 11,
                "n_flagged_sites": 0,
                "protected_reference_pilot_gate": "pass",
            },
            {
                "target_id": "colibactin",
                "guide_id": "secondary",
                "benchmark_role": "secondary",
                "n_references": 11,
                "n_flagged_sites": 5,
                "protected_reference_pilot_gate": "not_passed",
            },
        ]
    )
    human = pd.DataFrame(
        [
            {
                "target_id": "colibactin",
                "guide_id": "primary",
                "benchmark_role": "primary",
                "n_genomes": 97,
                "n_genomes_covered": 96,
                "n_genomes_unique_site": 95,
                "human_isolate_conservation_gate": "pass",
            },
            {
                "target_id": "colibactin",
                "guide_id": "secondary",
                "benchmark_role": "secondary",
                "n_genomes": 97,
                "n_genomes_covered": 97,
                "n_genomes_unique_site": 97,
                "human_isolate_conservation_gate": "pass",
            },
        ]
    )
    atlas = build_atlas(
        known,
        pd.DataFrame(),
        guide_conservation=conservation,
        guide_specificity=specificity,
        human_isolate_conservation=human,
    ).set_index("candidate_id")
    assert (
        atlas.loc["colibactin", "conservation_status"]
        == "human_isolate_panel_pass_global_diversity_pending"
    )
    assert (
        "primary: 96/97 covered and 95/97 unique-site"
        in atlas.loc["colibactin", "conservation_detail"]
    )
    assert (
        atlas.loc["colibactin", "atlas_status"]
        == "literature_priority_primary_guide_global_and_platform_validation_pending"
    )
