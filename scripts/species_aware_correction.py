"""Species-aware study-offset correction for species-resolved functions.

This pilot separates two deployment settings:

* ``source_only`` learns study offsets from training cohorts only. The unseen
  target cohort is never used to fit a transformation.
* ``target_adaptive`` additionally estimates an unlabeled target-cohort offset.
  This is explicitly transductive and is reported separately.

For every stratified pathway with a recognized parent species, the same
multiplicative correction applied to the parent species is propagated to the
pathway abundance. No cancer labels are used to estimate correction factors.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lodo_cv import get_lodo_splits  # noqa: E402


PREVALENCE_THRESHOLD = 0.05
MEAN_THRESHOLD = 1e-7
MAX_PATHWAYS = 10_000
OFFSET_CLIP = 2.0


def parent_species(feature: str) -> Optional[str]:
    """Extract the ``s__`` taxon from a HUMAnN stratified feature name."""
    if "|" not in feature:
        return None
    taxon = feature.rsplit("|", 1)[1].replace(".", "|")
    match = re.search(r"(?:^|\|)(s__[^|]+)$", taxon)
    return match.group(1) if match else None


def species_suffix(column: str) -> Optional[str]:
    match = re.search(r"(?:^|\|)(s__[^|]+)$", column)
    return match.group(1) if match else None


def fit_study_offsets(
    species_log: pd.DataFrame, studies: pd.Series, clip: float = OFFSET_CLIP
) -> pd.DataFrame:
    """Estimate robust log10 offsets for each observed training study."""
    global_median = species_log.median(axis=0)
    offsets = {
        study: (species_log.loc[studies.eq(study)].median(axis=0) - global_median)
        .clip(-clip, clip)
        .fillna(0.0)
        for study in sorted(studies.unique())
    }
    return pd.DataFrame(offsets).T


def estimate_target_offset(
    target_species_log: pd.DataFrame,
    training_species_log: pd.DataFrame,
    clip: float = OFFSET_CLIP,
) -> pd.Series:
    """Estimate one unlabeled target offset relative to the training reference."""
    return (
        target_species_log.median(axis=0) - training_species_log.median(axis=0)
    ).clip(-clip, clip).fillna(0.0)


def correct_species(
    species_log: pd.DataFrame, studies: pd.Series, offsets: pd.DataFrame
) -> pd.DataFrame:
    corrected = species_log.copy()
    for study in studies.unique():
        rows = studies.eq(study).to_numpy()
        if study in offsets.index:
            corrected.iloc[rows] = corrected.iloc[rows].to_numpy() - offsets.loc[
                study
            ].to_numpy()
    return corrected


def propagate_to_pathways(
    pathways: pd.DataFrame,
    studies: pd.Series,
    offsets: pd.DataFrame,
    species_columns: list[str],
) -> tuple[pd.DataFrame, float]:
    """Apply parent-species multiplicative factors to stratified pathways."""
    suffix_to_column = {
        suffix: column
        for column in species_columns
        if (suffix := species_suffix(column)) is not None
    }
    parents = [parent_species(column) for column in pathways.columns]
    mapped = [parent in suffix_to_column for parent in parents]
    values = pathways.to_numpy(dtype=np.float32, copy=True)
    for study in studies.unique():
        rows = studies.eq(study).to_numpy()
        if study not in offsets.index:
            continue
        factors = np.ones(len(pathways.columns), dtype=np.float32)
        for index, parent in enumerate(parents):
            if parent in suffix_to_column:
                species_column = suffix_to_column[parent]
                factors[index] = 10.0 ** (-float(offsets.loc[study, species_column]))
        values[rows] *= factors
    coverage = float(np.mean(mapped)) if mapped else 0.0
    return pd.DataFrame(values, columns=pathways.columns), coverage


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def load_data():
    metadata = pd.read_csv("data/processed/metadata_clean.csv")
    metadata = metadata[metadata["label"].isin([0, 1])].reset_index(drop=True)
    species = pd.read_csv("data/processed/species_filtered.csv").set_index("sample_id")
    pathways = pd.read_csv("data/raw/pathway_stratified_v320.csv").set_index("sample_id")
    ids = metadata["sample_id"]
    species = species.loc[ids].reset_index(drop=True).astype(np.float32)
    pathways = pathways.loc[ids]
    pathway_columns = [
        column
        for column in pathways.columns
        if "|" in column
        and not column.startswith("UNMAPPED")
        and not column.startswith("UNINTEGRATED")
    ]
    pathways = pathways[pathway_columns].reset_index(drop=True).astype(np.float32)
    return metadata, species, pathways


def select_pathways(training: pd.DataFrame) -> list[str]:
    prevalence = training.gt(0).mean(axis=0)
    abundance = training.mean(axis=0)
    eligible = pd.DataFrame(
        {"prevalence": prevalence, "mean_abundance": abundance}
    )
    eligible = eligible[
        eligible["prevalence"].ge(PREVALENCE_THRESHOLD)
        & eligible["mean_abundance"].ge(MEAN_THRESHOLD)
    ]
    eligible = eligible.sort_values(
        ["prevalence", "mean_abundance"], ascending=False, kind="mergesort"
    )
    return eligible.head(MAX_PATHWAYS).index.tolist()


def append_predictions(
    rows: list[dict[str, object]],
    model_name: str,
    metadata: pd.DataFrame,
    test_idx,
    probability: np.ndarray,
) -> None:
    for row, score in zip(test_idx, probability):
        rows.append(
            {
                "model": model_name,
                "sample_id": metadata.iloc[row]["sample_id"],
                "cohort": metadata.iloc[row]["study_name"],
                "y_true": int(metadata.iloc[row]["label"]),
                "y_prob": float(score),
            }
        )


def main() -> None:
    metadata, species, pathways = load_data()
    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    for held_out, train_idx, test_idx, excluded in get_lodo_splits(
        metadata, country_col="country"
    ):
        train_studies = metadata.iloc[train_idx]["study_name"].reset_index(drop=True)
        test_studies = metadata.iloc[test_idx]["study_name"].reset_index(drop=True)
        sp_train = species.iloc[train_idx].reset_index(drop=True)
        sp_test = species.iloc[test_idx].reset_index(drop=True)
        pw_train_raw = pathways.iloc[train_idx].reset_index(drop=True)
        pw_test_raw = pathways.iloc[test_idx].reset_index(drop=True)
        keep = select_pathways(pw_train_raw)
        pw_train_raw = pw_train_raw[keep]
        pw_test_raw = pw_test_raw[keep]

        source_offsets = fit_study_offsets(sp_train, train_studies)
        sp_train_corrected = correct_species(sp_train, train_studies, source_offsets)
        pw_train_corrected, train_coverage = propagate_to_pathways(
            pw_train_raw, train_studies, source_offsets, sp_train.columns.tolist()
        )

        # Strict source-only setting: no target distribution is used.
        sp_test_source = sp_test.copy()
        pw_test_source = pw_test_raw.copy()

        # Explicit unlabeled target-adaptation setting.
        target_offset = estimate_target_offset(sp_test, sp_train)
        target_offsets = pd.DataFrame([target_offset], index=[held_out])
        sp_test_adaptive = correct_species(sp_test, test_studies, target_offsets)
        pw_test_adaptive, test_coverage = propagate_to_pathways(
            pw_test_raw, test_studies, target_offsets, sp_test.columns.tolist()
        )

        y_train = metadata.iloc[train_idx]["label"].astype(int).to_numpy()
        y_test = metadata.iloc[test_idx]["label"].astype(int).to_numpy()

        species_model = make_model()
        species_model.fit(sp_train_corrected, y_train)
        species_probability = species_model.predict_proba(sp_test_source)[:, 1]

        train_joint = pd.concat([sp_train_corrected, pw_train_corrected], axis=1)
        source_test_joint = pd.concat([sp_test_source, pw_test_source], axis=1)
        adaptive_test_joint = pd.concat(
            [sp_test_adaptive, pw_test_adaptive], axis=1
        )
        joint_model = make_model()
        joint_model.fit(train_joint, y_train)
        source_probability = joint_model.predict_proba(source_test_joint)[:, 1]
        adaptive_probability = joint_model.predict_proba(adaptive_test_joint)[:, 1]

        fold_outputs = {
            "species_source_only": (species_probability, species.shape[1]),
            "stratified_source_only": (source_probability, train_joint.shape[1]),
            "stratified_target_adaptive": (
                adaptive_probability,
                train_joint.shape[1],
            ),
        }
        for model_name, (probability, n_features) in fold_outputs.items():
            auc = roc_auc_score(y_test, probability)
            results.append(
                {
                    "model": model_name,
                    "cohort": held_out,
                    "auc": auc,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_features": n_features,
                    "excluded_cohorts": ";".join(sorted(excluded)),
                }
            )
            append_predictions(
                predictions, model_name, metadata, test_idx, probability
            )
        audits.append(
            {
                "cohort": held_out,
                "n_pathways": len(keep),
                "train_pathway_parent_coverage": train_coverage,
                "test_pathway_parent_coverage": test_coverage,
                "mean_abs_target_log10_offset": float(target_offset.abs().mean()),
                "max_abs_target_log10_offset": float(target_offset.abs().max()),
                "target_distribution_used_for_source_only": False,
                "target_distribution_used_for_adaptive": True,
            }
        )
        print(
            f"{held_out:25s} "
            + " ".join(
                f"{name}={roc_auc_score(y_test, output[0]):.3f}"
                for name, output in fold_outputs.items()
            )
        )

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.DataFrame(predictions)
    output = Path("results/species_aware_correction")
    output.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output / "lodo_results.csv", index=False)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "correction_audit.csv", index=False)
    summary = result_frame.groupby("model")["auc"].agg(["mean", "min", "max"])
    summary.to_csv(output / "model_summary.csv")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
