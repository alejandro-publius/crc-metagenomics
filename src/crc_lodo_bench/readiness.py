"""Cross-cohort evidence summaries for microbial intervention targets.

This module deliberately separates *association evidence* from the other
evidence needed to nominate a microbiome-editing target.  A target can recur
in CRC cohorts and still lack causal, sequence-conservation, specificity, or
delivery evidence.  Those evidence gates are joined later in the atlas rather
than being inferred from prediction performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REGISTRY_COLUMNS = {
    "target_id",
    "display_name",
    "score_prefix",
    "parent_taxon",
    "prespecified_genes",
    "target_role",
    "selection_basis",
    "anchor_evidence_url",
    "causal_evidence_status",
    "editability_evidence_status",
    "registry_status",
}

SCORE_METADATA_COLUMNS = {"sample_id", "study_name", "label", "country"}


@dataclass(frozen=True)
class CrossPopulationThresholds:
    """Frozen thresholds for the human cross-population association gate.

    These thresholds screen for a recurring CRC association; passing them is
    not equivalent to being ready for an experiment or a treatment.
    """

    min_evaluable_cohorts: int = 5
    min_crc_enriched_fraction: float = 0.70
    min_median_auc: float = 0.55

    def __post_init__(self) -> None:
        if self.min_evaluable_cohorts < 1:
            raise ValueError("min_evaluable_cohorts must be positive")
        if not 0.5 <= self.min_crc_enriched_fraction <= 1.0:
            raise ValueError("min_crc_enriched_fraction must be in [0.5, 1]")
        if not 0.5 <= self.min_median_auc <= 1.0:
            raise ValueError("min_median_auc must be in [0.5, 1]")


def validate_target_registry(registry: pd.DataFrame) -> None:
    """Validate the target registry before any outcome calculation."""
    missing = sorted(REGISTRY_COLUMNS - set(registry.columns))
    if missing:
        raise ValueError(f"target registry is missing columns: {missing}")
    if registry.empty:
        raise ValueError("target registry cannot be empty")
    if registry["target_id"].isna().any() or registry["target_id"].duplicated().any():
        raise ValueError("target_id values must be non-null and unique")
    if registry["score_prefix"].isna().any() or registry["score_prefix"].duplicated().any():
        raise ValueError("score_prefix values must be non-null and unique")
    if not registry["anchor_evidence_url"].str.startswith("https://").all():
        raise ValueError("every registry entry requires an https evidence URL")


def _safe_auc(y_true: np.ndarray, values: np.ndarray) -> float:
    """Return an oriented association AUC, treating a constant assay as 0.5."""
    if np.unique(y_true).size != 2:
        raise ValueError("association AUC requires both CRC and control samples")
    if np.unique(values).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, values))


def compute_cohort_target_associations(
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    cohort_col: str = "study_name",
    label_col: str = "label",
) -> pd.DataFrame:
    """Compute descriptive CRC-versus-control evidence within every cohort.

    No pooled feature selection or model fitting occurs here.  Each registered
    target is summarized independently within a cohort so a large study cannot
    conceal disagreement among populations.
    """
    validate_target_registry(registry)
    missing_metadata = sorted(SCORE_METADATA_COLUMNS - set(scores.columns))
    if missing_metadata:
        raise ValueError(f"score table is missing columns: {missing_metadata}")
    if scores["sample_id"].duplicated().any():
        raise ValueError("score table contains duplicate sample_id values")

    labels = set(scores[label_col].dropna().astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError("labels must be encoded as 0 (control) and 1 (CRC)")

    rows: list[dict[str, object]] = []
    for target in registry.itertuples(index=False):
        abundance_col = f"{target.score_prefix}__abundance"
        completeness_col = f"{target.score_prefix}__completeness"
        assay_available = abundance_col in scores.columns

        for cohort, cohort_frame in scores.groupby(cohort_col, sort=True):
            frame = cohort_frame[cohort_frame[label_col].isin([0, 1])].copy()
            crc = frame[frame[label_col].eq(1)]
            control = frame[frame[label_col].eq(0)]
            if crc.empty or control.empty:
                continue

            base = {
                "target_id": target.target_id,
                "display_name": target.display_name,
                "cohort": cohort,
                "country": ";".join(sorted(frame["country"].dropna().astype(str).unique())),
                "n_crc": len(crc),
                "n_control": len(control),
                "assay_available": assay_available,
            }
            if not assay_available:
                rows.append(
                    {
                        **base,
                        "evaluable": False,
                        "crc_prevalence": np.nan,
                        "control_prevalence": np.nan,
                        "prevalence_difference": np.nan,
                        "association_auc": np.nan,
                        "direction": "not_assayed",
                        "median_completeness_when_detected": np.nan,
                    }
                )
                continue

            abundance = pd.to_numeric(frame[abundance_col], errors="raise").to_numpy()
            if not np.isfinite(abundance).all() or (abundance < 0).any():
                raise ValueError(f"{abundance_col} must contain finite non-negative values")
            crc_values = pd.to_numeric(crc[abundance_col], errors="raise").to_numpy()
            control_values = pd.to_numeric(control[abundance_col], errors="raise").to_numpy()
            crc_prevalence = float(np.mean(crc_values > 0))
            control_prevalence = float(np.mean(control_values > 0))
            prevalence_difference = crc_prevalence - control_prevalence
            evaluable = bool(np.any(abundance > 0))

            if completeness_col in frame.columns and evaluable:
                completeness = pd.to_numeric(
                    frame.loc[frame[abundance_col] > 0, completeness_col], errors="raise"
                ).to_numpy()
                if not np.isfinite(completeness).all() or np.any(
                    (completeness < 0) | (completeness > 1)
                ):
                    raise ValueError(f"{completeness_col} must be finite and in [0, 1]")
                median_completeness = float(np.median(completeness))
            else:
                median_completeness = np.nan

            if not evaluable or prevalence_difference == 0:
                direction = "tie"
            elif prevalence_difference > 0:
                direction = "crc_enriched"
            else:
                direction = "control_enriched"

            rows.append(
                {
                    **base,
                    "evaluable": evaluable,
                    "crc_prevalence": crc_prevalence,
                    "control_prevalence": control_prevalence,
                    "prevalence_difference": prevalence_difference,
                    "association_auc": _safe_auc(
                        frame[label_col].astype(int).to_numpy(), abundance
                    ),
                    "direction": direction,
                    "median_completeness_when_detected": median_completeness,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target_id", "cohort"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_cross_population_evidence(
    cohort_evidence: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    thresholds: CrossPopulationThresholds | None = None,
) -> pd.DataFrame:
    """Summarize cohort evidence and apply only the human-association gate."""
    validate_target_registry(registry)
    thresholds = thresholds or CrossPopulationThresholds()
    required = {
        "target_id",
        "cohort",
        "evaluable",
        "direction",
        "association_auc",
        "prevalence_difference",
        "median_completeness_when_detected",
    }
    missing = sorted(required - set(cohort_evidence.columns))
    if missing:
        raise ValueError(f"cohort evidence is missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for target in registry.itertuples(index=False):
        frame = cohort_evidence[cohort_evidence["target_id"].eq(target.target_id)]
        evaluable = frame[frame["evaluable"].astype(bool)]
        n_evaluable = len(evaluable)
        n_crc_enriched = int(evaluable["direction"].eq("crc_enriched").sum())
        enriched_fraction = n_crc_enriched / n_evaluable if n_evaluable else np.nan
        median_auc = (
            float(evaluable["association_auc"].median()) if n_evaluable else np.nan
        )
        median_prevalence_difference = (
            float(evaluable["prevalence_difference"].median())
            if n_evaluable
            else np.nan
        )
        completeness = evaluable["median_completeness_when_detected"].dropna()
        median_completeness = float(completeness.median()) if len(completeness) else np.nan

        enough_cohorts = n_evaluable >= thresholds.min_evaluable_cohorts
        consistent = bool(
            n_evaluable
            and enriched_fraction >= thresholds.min_crc_enriched_fraction
        )
        minimum_effect = bool(n_evaluable and median_auc >= thresholds.min_median_auc)
        association_gate = bool(enough_cohorts and consistent and minimum_effect)

        rows.append(
            {
                "target_id": target.target_id,
                "display_name": target.display_name,
                "target_role": target.target_role,
                "n_cohorts_total": len(frame),
                "n_cohorts_evaluable": n_evaluable,
                "n_crc_enriched_cohorts": n_crc_enriched,
                "crc_enriched_fraction": enriched_fraction,
                "median_association_auc": median_auc,
                "median_prevalence_difference": median_prevalence_difference,
                "median_completeness_when_detected": median_completeness,
                "cross_population_gate": "pass" if association_gate else "not_passed",
                "causal_evidence_status": target.causal_evidence_status,
                "conservation_status": "not_yet_assessed",
                "specificity_status": "not_yet_assessed",
                "editability_evidence_status": target.editability_evidence_status,
                "overall_readiness": "not_yet_assessable",
            }
        )

    return pd.DataFrame(rows).sort_values("target_id", kind="mergesort").reset_index(
        drop=True
    )


def required_score_columns(registry: pd.DataFrame) -> Iterable[str]:
    """Return the expected abundance columns for a validated registry."""
    validate_target_registry(registry)
    return (f"{prefix}__abundance" for prefix in registry["score_prefix"])


__all__ = [
    "CrossPopulationThresholds",
    "compute_cohort_target_associations",
    "required_score_columns",
    "summarize_cross_population_evidence",
    "validate_target_registry",
]
