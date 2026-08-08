"""Country-aware LODO benchmark for selected UniRef90 gene families.

Each fold uses the manifest produced exclusively from its training cohorts.
Scaling is also fit on training samples only. The sparse elastic-net logistic
model is a deliberately simple high-dimensional baseline.

Usage:
    python3 scripts/train_gene_family_lodo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler

from lodo_cv import get_lodo_splits


DATA_PREFIX = Path("data/raw/gene_families_selected")
MANIFEST_DIR = Path("results/gene_family_manifests")


def load_selected_matrix(prefix: Path = DATA_PREFIX):
    matrix = mmread(prefix.with_suffix(".mtx")).tocsr().T
    features = Path(f"{prefix}.features.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    samples = pd.read_csv(f"{prefix}.samples.csv")
    if matrix.shape != (len(samples), len(features)):
        raise ValueError(
            "matrix/sample/feature dimensions disagree: "
            f"matrix={matrix.shape}, samples={len(samples)}, "
            f"features={len(features)}"
        )
    return matrix, features, samples


def main() -> None:
    X, features, metadata = load_selected_matrix()
    feature_index = {name: idx for idx, name in enumerate(features)}
    country_col = "country" if "country" in metadata.columns else None

    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for held_out, train_idx, test_idx, excluded in get_lodo_splits(
        metadata, country_col=country_col
    ):
        manifest_path = MANIFEST_DIR / f"{held_out}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing fold manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        selected = [
            feature_index[gene]
            for gene in manifest["gene_id"]
            if gene in feature_index
        ]
        if not selected:
            raise ValueError(f"no selected genes available for {held_out}")

        X_train = X[train_idx][:, selected]
        X_test = X[test_idx][:, selected]
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=5_000,
            random_state=42,
            n_jobs=-1,
        )
        y_train = metadata.iloc[train_idx]["label"].astype(int).to_numpy()
        y_test = metadata.iloc[test_idx]["label"].astype(int).to_numpy()
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probability)

        results.append(
            {
                "cohort": held_out,
                "auc": auc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": len(selected),
                "excluded_cohorts": ";".join(sorted(excluded)),
            }
        )
        for row, truth, score in zip(test_idx, y_test, probability):
            predictions.append(
                {
                    "sample_id": metadata.iloc[row]["sample_id"],
                    "cohort": held_out,
                    "y_true": int(truth),
                    "y_prob": float(score),
                }
            )
        print(
            f"{held_out:25s} AUC={auc:.3f} "
            f"(n_train={len(train_idx)}, n_test={len(test_idx)}, "
            f"p={len(selected)})"
        )

    results_df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    results_df.to_csv("results/gene_family_lodo_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        "results/preds_gene_family_elastic_net.csv", index=False
    )
    print(f"Mean per-cohort AUC: {np.mean(results_df['auc']):.3f}")


if __name__ == "__main__":
    main()
