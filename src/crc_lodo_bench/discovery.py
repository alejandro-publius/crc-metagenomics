"""Leakage-safe, cross-fitted discovery of recurring CRC gene families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class DiscoveryThresholds:
    """Frozen internal nomination rules for the de novo discovery track."""

    min_training_cohorts: int = 3
    min_training_crc_enriched_fraction: float = 0.70
    min_training_median_auc: float = 0.55
    max_candidates_per_fold: int = 100
    min_outer_selections: int = 7
    min_heldout_evaluable: int = 7
    min_heldout_crc_enriched_fraction: float = 0.70
    min_heldout_median_auc: float = 0.55

    def __post_init__(self) -> None:
        integer_fields = {
            "min_training_cohorts": self.min_training_cohorts,
            "max_candidates_per_fold": self.max_candidates_per_fold,
            "min_outer_selections": self.min_outer_selections,
            "min_heldout_evaluable": self.min_heldout_evaluable,
        }
        if any(value < 1 for value in integer_fields.values()):
            raise ValueError(f"integer thresholds must be positive: {integer_fields}")
        fractions = {
            "min_training_crc_enriched_fraction": self.min_training_crc_enriched_fraction,
            "min_training_median_auc": self.min_training_median_auc,
            "min_heldout_crc_enriched_fraction": self.min_heldout_crc_enriched_fraction,
            "min_heldout_median_auc": self.min_heldout_median_auc,
        }
        if any(not 0.5 <= value <= 1.0 for value in fractions.values()):
            raise ValueError(f"fraction/AUC thresholds must be in [0.5, 1]: {fractions}")


def _validate_discovery_inputs(
    matrix: np.ndarray,
    feature_ids: Sequence[str],
    metadata: pd.DataFrame,
) -> None:
    required = {"sample_id", "study_name", "country", "label"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape != (len(metadata), len(feature_ids)):
        raise ValueError(
            "matrix dimensions must equal metadata rows x feature_ids: "
            f"{matrix.shape} != ({len(metadata)}, {len(feature_ids)})"
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature_ids must be unique")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("metadata contains duplicate sample_id values")
    if not set(metadata["label"].dropna().astype(int).unique()) <= {0, 1}:
        raise ValueError("labels must be 0 (control) or 1 (CRC)")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("gene-family matrix must be finite and non-negative")


def compute_gene_cohort_statistics(
    matrix: np.ndarray,
    feature_ids: Sequence[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate vectorized within-cohort association evidence for every gene."""
    matrix = np.asarray(matrix)
    _validate_discovery_inputs(matrix, feature_ids, metadata)
    rows: list[pd.DataFrame] = []

    for cohort, cohort_metadata in metadata.groupby("study_name", sort=True):
        positions = metadata.index.get_indexer(cohort_metadata.index)
        if (positions < 0).any():
            raise ValueError("metadata index could not be aligned to the matrix")
        y = cohort_metadata["label"].astype(int).to_numpy()
        n_crc = int(np.sum(y == 1))
        n_control = int(np.sum(y == 0))
        if n_crc == 0 or n_control == 0:
            continue

        values = matrix[positions, :]
        crc_values = values[y == 1, :]
        control_values = values[y == 0, :]
        crc_prevalence = np.mean(crc_values > 0, axis=0)
        control_prevalence = np.mean(control_values > 0, axis=0)

        ranks = rankdata(values, axis=0, method="average")
        positive_rank_sum = np.sum(ranks[y == 1, :], axis=0)
        mann_whitney_u = positive_rank_sum - n_crc * (n_crc + 1) / 2
        auc = mann_whitney_u / (n_crc * n_control)
        evaluable = np.any(values > 0, axis=0)
        auc = np.where(evaluable, auc, np.nan)
        prevalence_difference = crc_prevalence - control_prevalence
        direction = np.where(
            ~evaluable,
            "not_detected",
            np.where(
                prevalence_difference > 0,
                "crc_enriched",
                np.where(prevalence_difference < 0, "control_enriched", "tie"),
            ),
        )

        rows.append(
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "country": ";".join(
                        sorted(cohort_metadata["country"].dropna().astype(str).unique())
                    ),
                    "gene_id": list(feature_ids),
                    "n_crc": n_crc,
                    "n_control": n_control,
                    "evaluable": evaluable,
                    "crc_prevalence": crc_prevalence,
                    "control_prevalence": control_prevalence,
                    "prevalence_difference": prevalence_difference,
                    "association_auc": auc,
                    "direction": direction,
                }
            )
        )

    if not rows:
        raise ValueError("no cohort contained both CRC and control samples")
    return pd.concat(rows, ignore_index=True).sort_values(
        ["cohort", "gene_id"], kind="mergesort"
    ).reset_index(drop=True)


def select_training_candidates(
    cohort_statistics: pd.DataFrame,
    training_cohorts: Sequence[str],
    *,
    thresholds: DiscoveryThresholds | None = None,
) -> pd.DataFrame:
    """Select candidates using only named training cohorts."""
    thresholds = thresholds or DiscoveryThresholds()
    if not training_cohorts:
        raise ValueError("training_cohorts cannot be empty")
    required = {
        "cohort",
        "gene_id",
        "evaluable",
        "direction",
        "association_auc",
        "prevalence_difference",
    }
    missing = sorted(required - set(cohort_statistics.columns))
    if missing:
        raise ValueError(f"cohort_statistics is missing columns: {missing}")

    training = cohort_statistics[
        cohort_statistics["cohort"].isin(training_cohorts)
        & cohort_statistics["evaluable"].astype(bool)
    ].copy()
    grouped = (
        training.groupby("gene_id", sort=False)
        .agg(
            n_training_cohorts=("cohort", "nunique"),
            n_crc_enriched_training=(
                "direction",
                lambda values: int(np.sum(values == "crc_enriched")),
            ),
            training_median_auc=("association_auc", "median"),
            training_median_prevalence_difference=("prevalence_difference", "median"),
        )
        .reset_index()
    )
    grouped["training_crc_enriched_fraction"] = (
        grouped["n_crc_enriched_training"] / grouped["n_training_cohorts"]
    )
    selected = grouped[
        (grouped["n_training_cohorts"] >= thresholds.min_training_cohorts)
        & (
            grouped["training_crc_enriched_fraction"]
            >= thresholds.min_training_crc_enriched_fraction
        )
        & (grouped["training_median_auc"] >= thresholds.min_training_median_auc)
    ].copy()
    selected = selected.sort_values(
        [
            "training_median_auc",
            "training_crc_enriched_fraction",
            "n_training_cohorts",
            "training_median_prevalence_difference",
            "gene_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).head(thresholds.max_candidates_per_fold)
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def build_cross_fitted_evidence(
    cohort_statistics: pd.DataFrame,
    training_cohorts_by_heldout: Mapping[str, Sequence[str]],
    *,
    thresholds: DiscoveryThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover on training cohorts and attach only the corresponding holdout."""
    thresholds = thresholds or DiscoveryThresholds()
    selection_rows: list[pd.DataFrame] = []
    evidence_rows: list[pd.DataFrame] = []

    for held_out, training_cohorts in sorted(training_cohorts_by_heldout.items()):
        selected = select_training_candidates(
            cohort_statistics, training_cohorts, thresholds=thresholds
        )
        selected.insert(0, "held_out_cohort", held_out)
        selected.insert(1, "training_cohorts", ";".join(sorted(training_cohorts)))
        selection_rows.append(selected)

        heldout_statistics = cohort_statistics[
            cohort_statistics["cohort"].eq(held_out)
        ][
            [
                "gene_id",
                "country",
                "evaluable",
                "crc_prevalence",
                "control_prevalence",
                "prevalence_difference",
                "association_auc",
                "direction",
            ]
        ].rename(
            columns={
                "country": "heldout_country",
                "evaluable": "heldout_evaluable",
                "crc_prevalence": "heldout_crc_prevalence",
                "control_prevalence": "heldout_control_prevalence",
                "prevalence_difference": "heldout_prevalence_difference",
                "association_auc": "heldout_association_auc",
                "direction": "heldout_direction",
            }
        )
        evidence = selected.merge(
            heldout_statistics, on="gene_id", how="left", validate="one_to_one"
        )
        if evidence["heldout_evaluable"].isna().any():
            raise ValueError(f"missing held-out statistics for {held_out}")
        evidence_rows.append(evidence)

    empty_columns = [
        "held_out_cohort",
        "training_cohorts",
        "selection_rank",
        "gene_id",
        "n_training_cohorts",
        "n_crc_enriched_training",
        "training_median_auc",
        "training_median_prevalence_difference",
        "training_crc_enriched_fraction",
    ]
    selections = (
        pd.concat(selection_rows, ignore_index=True)
        if selection_rows
        else pd.DataFrame(columns=empty_columns)
    )
    evidence = (
        pd.concat(evidence_rows, ignore_index=True)
        if evidence_rows
        else pd.DataFrame(columns=empty_columns)
    )
    return selections, evidence


def summarize_cross_fitted_candidates(
    cross_fitted_evidence: pd.DataFrame,
    *,
    thresholds: DiscoveryThresholds | None = None,
) -> pd.DataFrame:
    """Apply the frozen internal-nomination rule to outer-fold evidence."""
    thresholds = thresholds or DiscoveryThresholds()
    if cross_fitted_evidence.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "n_outer_selections",
                "n_heldout_evaluable",
                "heldout_crc_enriched_fraction",
                "heldout_median_auc",
                "heldout_median_prevalence_difference",
                "internal_nomination",
                "external_confirmation_status",
            ]
        )

    def summarize_gene(frame: pd.DataFrame) -> pd.Series:
        evaluable = frame[frame["heldout_evaluable"].astype(bool)]
        n_evaluable = len(evaluable)
        enriched_fraction = (
            float(evaluable["heldout_direction"].eq("crc_enriched").mean())
            if n_evaluable
            else np.nan
        )
        median_auc = (
            float(evaluable["heldout_association_auc"].median())
            if n_evaluable
            else np.nan
        )
        median_difference = (
            float(evaluable["heldout_prevalence_difference"].median())
            if n_evaluable
            else np.nan
        )
        nominated = bool(
            len(frame) >= thresholds.min_outer_selections
            and n_evaluable >= thresholds.min_heldout_evaluable
            and enriched_fraction >= thresholds.min_heldout_crc_enriched_fraction
            and median_auc >= thresholds.min_heldout_median_auc
        )
        return pd.Series(
            {
                "n_outer_selections": len(frame),
                "n_heldout_evaluable": n_evaluable,
                "heldout_crc_enriched_fraction": enriched_fraction,
                "heldout_median_auc": median_auc,
                "heldout_median_prevalence_difference": median_difference,
                "internal_nomination": nominated,
                "external_confirmation_status": "not_yet_assessed",
            }
        )

    summary = (
        cross_fitted_evidence.groupby("gene_id", sort=False)
        .apply(summarize_gene, include_groups=False)
        .reset_index()
    )
    return summary.sort_values(
        [
            "internal_nomination",
            "n_outer_selections",
            "heldout_median_auc",
            "heldout_crc_enriched_fraction",
            "gene_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


__all__ = [
    "DiscoveryThresholds",
    "build_cross_fitted_evidence",
    "compute_gene_cohort_statistics",
    "select_training_candidates",
    "summarize_cross_fitted_candidates",
]
