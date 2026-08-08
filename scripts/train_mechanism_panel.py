"""Evaluate frozen CRC mechanism scores without outcome-driven feature search."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lodo_cv import get_lodo_splits


DATA_PREFIX = Path("data/raw/mechanism_panel")
MANIFEST = Path("results/mechanism_panel/frozen_manifest.csv")

PARENT_SPECIES = {
    "colibactin_genotoxicity": "s__Escherichia_coli",
    "b_fragilis_toxin": "s__Bacteroides_fragilis",
    "secondary_bile_acid": "s__Clostridium_scindens",
}


def load_mechanism_matrix(prefix: Path = DATA_PREFIX):
    matrix = mmread(prefix.with_suffix(".mtx")).tocsr().T
    features = Path(f"{prefix}.features.txt").read_text(encoding="utf-8").splitlines()
    samples = pd.read_csv(f"{prefix}.samples.csv")
    if matrix.shape != (len(samples), len(features)):
        raise ValueError("mechanism matrix dimensions disagree with metadata")
    return matrix, features, samples


def build_mechanism_scores(matrix, features: list[str], manifest: pd.DataFrame) -> pd.DataFrame:
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    detected = manifest[manifest["query_status"].eq("frozen_detected")].copy()
    detected = detected.drop_duplicates(
        ["mechanism", "prespecified_gene", "uniref90"]
    )
    output: dict[str, np.ndarray] = {}
    for mechanism, frame in detected.groupby("mechanism"):
        gene_values = []
        for _gene, gene_frame in frame.groupby("prespecified_gene"):
            columns = [
                feature_index[cluster]
                for cluster in gene_frame["uniref90"]
                if cluster in feature_index
            ]
            if columns:
                gene_values.append(np.asarray(matrix[:, columns].sum(axis=1)).ravel())
        if not gene_values:
            continue
        genes = np.column_stack(gene_values)
        output[f"{mechanism}__abundance"] = genes.sum(axis=1)
        output[f"{mechanism}__completeness"] = (genes > 0).mean(axis=1)
    return pd.DataFrame(output)


def find_species_column(columns: list[str], suffix: str) -> str:
    matches = [column for column in columns if column.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one parent-species column for {suffix}, got {matches}")
    return matches[0]


def evaluate_lodo(features: pd.DataFrame, metadata: pd.DataFrame, model_name: str):
    results = []
    predictions = []
    for held_out, train_idx, test_idx, excluded in get_lodo_splits(
        metadata, country_col="country"
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        )
        y_train = metadata.iloc[train_idx]["label"].astype(int).to_numpy()
        y_test = metadata.iloc[test_idx]["label"].astype(int).to_numpy()
        model.fit(features.iloc[train_idx], y_train)
        probability = model.predict_proba(features.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y_test, probability)
        results.append(
            {
                "model": model_name,
                "cohort": held_out,
                "auc": auc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": features.shape[1],
                "excluded_cohorts": ";".join(sorted(excluded)),
            }
        )
        predictions.extend(
            {
                "model": model_name,
                "sample_id": metadata.iloc[row]["sample_id"],
                "cohort": held_out,
                "y_true": int(truth),
                "y_prob": float(score),
            }
            for row, truth, score in zip(test_idx, y_test, probability)
        )
    return pd.DataFrame(results), pd.DataFrame(predictions)


def main() -> None:
    matrix, genes, metadata = load_mechanism_matrix()
    manifest = pd.read_csv(MANIFEST)
    mechanism = build_mechanism_scores(matrix, genes, manifest)

    species = pd.read_csv("data/processed/species_filtered.csv")
    species = species.set_index("sample_id").loc[metadata["sample_id"]]
    parent = pd.DataFrame(
        {
            f"{mechanism_name}__parent_species": species[
                find_species_column(species.columns.tolist(), suffix)
            ].to_numpy()
            for mechanism_name, suffix in PARENT_SPECIES.items()
        }
    )
    combined = pd.concat([mechanism.reset_index(drop=True), parent], axis=1)
    score_table = pd.concat(
        [metadata[["sample_id", "study_name", "label", "country"]], combined],
        axis=1,
    )
    score_table.to_csv("results/mechanism_panel/sample_scores.csv", index=False)

    all_results = []
    all_predictions = []
    for model_name, frame in [
        ("mechanism_only", mechanism),
        ("parent_species_only", parent),
        ("mechanism_plus_parent", combined),
    ]:
        result, prediction = evaluate_lodo(frame, metadata, model_name)
        all_results.append(result)
        all_predictions.append(prediction)

    results = pd.concat(all_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    results.to_csv("results/mechanism_panel/lodo_results.csv", index=False)
    predictions.to_csv("results/mechanism_panel/predictions.csv", index=False)
    summary = results.groupby("model")["auc"].agg(["mean", "min", "max"])
    summary.to_csv("results/mechanism_panel/model_summary.csv")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
