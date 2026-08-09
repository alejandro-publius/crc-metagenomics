"""Cross-fitted tests of gene-family signal beyond parent-species abundance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ParentAdjustmentThresholds:
    """Frozen gate for information added beyond an annotated parent species."""

    min_evaluable_folds: int = 7
    min_positive_delta_fraction: float = 0.70
    min_median_delta_auc: float = 0.02

    def __post_init__(self) -> None:
        if self.min_evaluable_folds < 1:
            raise ValueError("min_evaluable_folds must be positive")
        if not 0.5 <= self.min_positive_delta_fraction <= 1.0:
            raise ValueError("min_positive_delta_fraction must be in [0.5, 1]")
        if not 0 <= self.min_median_delta_auc <= 0.5:
            raise ValueError("min_median_delta_auc must be in [0, 0.5]")


def organism_binomials(organisms: str) -> list[str]:
    """Extract unique genus-species names without inferring synonyms."""
    binomials: set[str] = set()
    for organism in str(organisms).split(";"):
        words = organism.strip().split()
        if len(words) >= 2:
            binomials.add(" ".join(words[:2]))
    return sorted(binomials)


def species_name_from_column(column: str) -> str | None:
    """Return the terminal MetaPhlAn species name in normal spacing."""
    if "s__" not in column:
        return None
    return column.rsplit("s__", 1)[1].replace("_", " ").strip()


def map_candidate_parents(
    annotations: pd.DataFrame, species_columns: Sequence[str]
) -> pd.DataFrame:
    """Map archived representative organisms to exact species-table names.

    The mapping is label-independent and intentionally does not add taxonomic
    synonyms after seeing model results.
    """
    required = {"gene_id", "organisms"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"annotations are missing columns: {missing}")
    lookup = {
        name.casefold(): column
        for column in species_columns
        if (name := species_name_from_column(column)) is not None
    }
    if len(lookup) != len(
        [column for column in species_columns if species_name_from_column(column)]
    ):
        raise ValueError("species columns do not have unique terminal names")

    rows: list[dict[str, object]] = []
    for candidate in annotations.itertuples(index=False):
        binomials = organism_binomials(candidate.organisms)
        matched = sorted(
            {lookup[name.casefold()] for name in binomials if name.casefold() in lookup}
        )
        rows.append(
            {
                "gene_id": candidate.gene_id,
                "archived_representative_binomials": ";".join(binomials),
                "matched_parent_species_columns": ";".join(matched),
                "n_matched_parent_species": len(matched),
                "mapping_status": (
                    "exact_single_match"
                    if len(matched) == 1
                    else "exact_multiple_matches" if matched else "no_exact_match"
                ),
                "mapping_rule": "exact_binomial_only_no_synonyms",
            }
        )
    return pd.DataFrame(rows).sort_values("gene_id", kind="mergesort").reset_index(
        drop=True
    )


def _make_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5_000,
            random_state=42,
        ),
    )


def evaluate_parent_adjustment(
    gene_values: pd.DataFrame,
    species_values: pd.DataFrame,
    metadata: pd.DataFrame,
    parent_mapping: pd.DataFrame,
    folds: Mapping[str, tuple[Sequence[int], Sequence[int]]],
) -> pd.DataFrame:
    """Compare parent-only and parent-plus-gene models in outer holdouts."""
    required_metadata = {"sample_id", "study_name", "label", "country"}
    missing = sorted(required_metadata - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if not (
        metadata["sample_id"].tolist()
        == gene_values.index.tolist()
        == species_values.index.tolist()
    ):
        raise ValueError("metadata, gene values, and species values must share sample order")

    rows: list[dict[str, object]] = []
    mapping = parent_mapping.set_index("gene_id")
    for gene_id in gene_values.columns:
        if gene_id not in mapping.index:
            raise ValueError(f"missing parent mapping for {gene_id}")
        parent_columns = [
            column
            for column in str(mapping.loc[gene_id, "matched_parent_species_columns"]).split(";")
            if column
        ]
        if not parent_columns:
            rows.append(
                {
                    "gene_id": gene_id,
                    "held_out_cohort": "not_evaluable",
                    "n_parent_species": 0,
                    "parent_auc": np.nan,
                    "combined_auc": np.nan,
                    "delta_auc": np.nan,
                    "evaluation_status": "no_exact_parent_species_match",
                }
            )
            continue

        parent = species_values[parent_columns]
        gene = gene_values[[gene_id]]
        combined = pd.concat([parent, gene], axis=1)
        for held_out, (train_idx, test_idx) in sorted(folds.items()):
            y_train = metadata.iloc[list(train_idx)]["label"].astype(int).to_numpy()
            y_test = metadata.iloc[list(test_idx)]["label"].astype(int).to_numpy()
            if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
                continue
            parent_model = _make_model()
            combined_model = _make_model()
            parent_model.fit(parent.iloc[list(train_idx)], y_train)
            combined_model.fit(combined.iloc[list(train_idx)], y_train)
            parent_probability = parent_model.predict_proba(
                parent.iloc[list(test_idx)]
            )[:, 1]
            combined_probability = combined_model.predict_proba(
                combined.iloc[list(test_idx)]
            )[:, 1]
            parent_auc = float(roc_auc_score(y_test, parent_probability))
            combined_auc = float(roc_auc_score(y_test, combined_probability))
            rows.append(
                {
                    "gene_id": gene_id,
                    "held_out_cohort": held_out,
                    "n_parent_species": len(parent_columns),
                    "parent_auc": parent_auc,
                    "combined_auc": combined_auc,
                    "delta_auc": combined_auc - parent_auc,
                    "evaluation_status": "evaluated",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["gene_id", "held_out_cohort"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_parent_adjustment(
    fold_results: pd.DataFrame,
    *,
    thresholds: ParentAdjustmentThresholds | None = None,
) -> pd.DataFrame:
    """Apply the frozen parent-adjustment gate across outer folds."""
    thresholds = thresholds or ParentAdjustmentThresholds()
    rows: list[dict[str, object]] = []
    for gene_id, frame in fold_results.groupby("gene_id", sort=True):
        evaluated = frame[frame["evaluation_status"].eq("evaluated")]
        if evaluated.empty:
            rows.append(
                {
                    "gene_id": gene_id,
                    "n_evaluable_folds": 0,
                    "positive_delta_fraction": np.nan,
                    "median_parent_auc": np.nan,
                    "median_combined_auc": np.nan,
                    "median_delta_auc": np.nan,
                    "parent_adjustment_gate": "not_evaluable",
                }
            )
            continue
        positive_fraction = float((evaluated["delta_auc"] > 0).mean())
        median_delta = float(evaluated["delta_auc"].median())
        passed = bool(
            len(evaluated) >= thresholds.min_evaluable_folds
            and positive_fraction >= thresholds.min_positive_delta_fraction
            and median_delta >= thresholds.min_median_delta_auc
        )
        rows.append(
            {
                "gene_id": gene_id,
                "n_evaluable_folds": len(evaluated),
                "positive_delta_fraction": positive_fraction,
                "median_parent_auc": float(evaluated["parent_auc"].median()),
                "median_combined_auc": float(evaluated["combined_auc"].median()),
                "median_delta_auc": median_delta,
                "parent_adjustment_gate": "pass" if passed else "not_passed",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["parent_adjustment_gate", "median_delta_auc", "gene_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


__all__ = [
    "ParentAdjustmentThresholds",
    "evaluate_parent_adjustment",
    "map_candidate_parents",
    "organism_binomials",
    "species_name_from_column",
    "summarize_parent_adjustment",
]
