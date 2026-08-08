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
    assert (
        atlas.loc["UniRef90_A1", "candidate_class"]
        == "precision_address_candidate"
    )
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
