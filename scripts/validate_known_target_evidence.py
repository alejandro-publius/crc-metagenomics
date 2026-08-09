"""Validate and summarize the structured known-target evidence review."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CAUSAL_RANK = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
EDIT_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}


def validate_and_summarize(
    evidence: pd.DataFrame, registry: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "evidence_id",
        "target_id",
        "source_url",
        "publication_status",
        "primary_source",
        "evidence_model",
        "perturbation",
        "main_observation",
        "causal_tier",
        "editability_tier",
        "key_limitation",
    }
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"evidence review is missing columns: {missing}")
    if evidence["evidence_id"].duplicated().any():
        raise ValueError("evidence IDs must be unique")
    if not evidence["causal_tier"].isin(CAUSAL_RANK).all():
        raise ValueError("invalid causal evidence tier")
    if not evidence["editability_tier"].isin(EDIT_RANK).all():
        raise ValueError("invalid editability evidence tier")
    if not evidence["publication_status"].isin({"peer_reviewed", "preprint"}).all():
        raise ValueError("invalid publication status")
    if not evidence["primary_source"].astype(bool).all():
        raise ValueError("tier-setting rows must be primary research")
    if not evidence["source_url"].str.startswith("http").all():
        raise ValueError("every evidence row must have a source URL")

    registered = set(registry["target_id"])
    observed = set(evidence["target_id"])
    if observed != registered:
        raise ValueError(
            f"evidence target coverage differs from registry: "
            f"missing={sorted(registered - observed)}, extra={sorted(observed - registered)}"
        )

    rows: list[dict[str, object]] = []
    for target_id, frame in evidence.groupby("target_id", sort=True):
        causal = max(frame["causal_tier"], key=CAUSAL_RANK.__getitem__)
        edit = max(frame["editability_tier"], key=EDIT_RANK.__getitem__)
        rows.append(
            {
                "target_id": target_id,
                "n_primary_evidence_rows": len(frame),
                "highest_causal_tier": causal,
                "highest_editability_tier": edit,
                "n_peer_reviewed_rows": int(
                    frame["publication_status"].eq("peer_reviewed").sum()
                ),
                "n_preprint_rows": int(
                    frame["publication_status"].eq("preprint").sum()
                ),
                "causal_evidence_status": {
                    "colibactin": "human_signature_plus_animal_targeted_perturbation",
                    "fragilysin": "animal_isogenic_deletion_necessity",
                    "fusobacterial_adhesion": "animal_genetic_mutant_support",
                    "secondary_bile_acids": "animal_pathway_mutant_support",
                }[target_id],
                "editability_evidence_status": {
                    "colibactin": "targeted_in_vivo_crispri_preprint",
                    "fragilysin": "isogenic_deletion_only",
                    "fusobacterial_adhesion": "isogenic_mutants_only",
                    "secondary_bile_acids": "isogenic_pathway_mutant_only",
                }[target_id],
                "experiment_ready": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    research_dir = Path("research/intervention_readiness")
    output_dir = Path("results/intervention_readiness")
    evidence = pd.read_csv(research_dir / "known_target_evidence_review.csv")
    registry = pd.read_csv(research_dir / "known_target_registry.csv")
    summary = validate_and_summarize(evidence, registry)
    summary.to_csv(output_dir / "known_target_evidence_summary.csv", index=False)
    audit = {
        "status": "complete_structured_primary_evidence_pass",
        "n_targets": int(summary["target_id"].nunique()),
        "n_primary_evidence_rows": int(len(evidence)),
        "n_experiment_ready": 0,
        "rubric": "research/intervention_readiness/evidence_review_protocol.md",
        "interpretation_boundary": (
            "The review records reported causal and engineering evidence. It "
            "does not complete conservation, specificity, delivery safety, or "
            "coauthor biological review."
        ),
    }
    (output_dir / "known_target_evidence_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
