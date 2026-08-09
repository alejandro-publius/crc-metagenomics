"""Verify every headline intervention-readiness claim from generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def verify(results_dir: Path, dossiers_dir: Path) -> dict[str, object]:
    discovery = json.loads((results_dir / "discovery_audit.json").read_text())
    parent = pd.read_csv(results_dir / "parent_adjustment_summary.csv")
    taxon = pd.read_csv(results_dir / "candidate_taxon_summary.csv")
    atlas = pd.read_csv(results_dir / "readiness_atlas.csv")
    sensitivity = pd.read_csv(results_dir / "taxonomic_gate_threshold_sensitivity.csv")
    evidence = pd.read_csv(results_dir / "known_target_evidence_summary.csv")
    integrity = pd.read_csv(results_dir / "known_target_mechanism_integrity.csv")
    guide_conservation = pd.read_csv(
        results_dir / "colibactin_guide_conservation_summary.csv"
    )

    checks = {
        "6755_gene_families": discovery["n_gene_families"] == 6755,
        "16_internal_nominations": len(parent) == 16,
        "4_parent_adjustment_pass": parent["parent_adjustment_gate"].eq("pass").sum()
        == 4,
        "0_taxonomic_addresses": taxon["taxonomic_resolution_status"]
        .eq("dominant_source_observed")
        .sum()
        == 0,
        "carrier_range_13_5_to_42_5_percent": (
            0.134 <= taxon["dominant_taxon_fraction"].min() <= 0.136
            and 0.424 <= taxon["dominant_taxon_fraction"].max() <= 0.426
        ),
        "parent_fraction_at_most_19_6_percent": taxon["parent_taxon_fraction"].max()
        <= 0.196,
        "no_majority_carrier_at_50_percent": int(
            sensitivity.loc[
                sensitivity["dominant_taxon_fraction_threshold"].eq(0.5),
                "n_candidates_passing",
            ].iloc[0]
        )
        == 0,
        "20_atlas_entries": len(atlas) == 20,
        "20_candidate_dossiers": len(list(dossiers_dir.glob("*.md"))) == 20,
        "no_experiment_ready_claim": not atlas["atlas_status"]
        .eq("experiment_ready")
        .any(),
        "4_known_targets_structurally_reviewed": len(evidence) == 4,
        "known_target_review_marks_0_experiment_ready": not evidence["experiment_ready"]
        .astype(bool)
        .any(),
        "colibactin_has_reported_e3_preprint": evidence.set_index("target_id").loc[
            "colibactin", "highest_editability_tier"
        ]
        == "E3",
        "fusobacterial_mechanism_not_represented": integrity.set_index("target_id").loc[
            "fusobacterial_adhesion", "mechanism_integrity_status"
        ]
        == "not_represented_in_frozen_assay",
        "2_published_colibactin_guides_audited": len(guide_conservation) == 2,
        "7_genomes_in_colibactin_pilot": guide_conservation["n_genomes"].eq(7).all(),
        "both_colibactin_guides_pass_reference_pilot": guide_conservation[
            "pilot_conservation_gate"
        ]
        .eq("pass")
        .all(),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"intervention-readiness verification failed: {failed}")
    return {
        "status": "pass",
        "n_checks": len(checks),
        "checks": checks,
        "claim_boundary": (
            "Verification confirms saved computational claims only; it is not "
            "biological or coauthor approval."
        ),
    }


def main() -> None:
    results_dir = Path("results/intervention_readiness")
    report = verify(results_dir, results_dir / "dossiers")
    (results_dir / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
