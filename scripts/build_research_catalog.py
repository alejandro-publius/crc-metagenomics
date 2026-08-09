"""Build a small SQLite catalog from the project's committed CSV outputs.

The database is both a useful query surface for cohort/result auditing and a
real-data SQL learning environment. It is derived and can always be rebuilt.

Usage:
    python3 scripts/build_research_catalog.py
    sqlite3 data/derived/crc_research.sqlite
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd


PREDICTION_FILES = {
    "species_rf": Path("results/preds_species_rf.csv"),
    "joint_rf": Path("results/preds_joint_rf.csv"),
    "joint_xgb": Path("results/preds_joint_xgb.csv"),
    "gene_family_enet": Path("results/preds_gene_family_elastic_net.csv"),
}


def load_fold_results(results_dir: Path) -> pd.DataFrame:
    baseline = pd.read_csv(results_dir / "baseline_results.csv")
    species = baseline.assign(model="species_rf", n_features=pd.NA)[
        ["model", "cohort", "auc", "n_train", "n_test", "n_features"]
    ]

    joint = pd.read_csv(results_dir / "joint_results.csv")
    sample_counts = baseline[["cohort", "n_train", "n_test"]]
    joint = joint.merge(sample_counts, on="cohort", validate="one_to_one")
    joint_rf = joint.rename(
        columns={"rf_auc": "auc", "rf_n_features": "n_features"}
    ).assign(model="joint_rf")
    joint_xgb = joint.rename(
        columns={"xgb_auc": "auc", "xgb_n_features": "n_features"}
    ).assign(model="joint_xgb")
    columns = ["model", "cohort", "auc", "n_train", "n_test", "n_features"]
    frames = [species, joint_rf[columns], joint_xgb[columns]]
    gene_path = results_dir / "gene_family_lodo_results.csv"
    if gene_path.exists():
        gene = pd.read_csv(gene_path).assign(model="gene_family_enet")
        frames.append(gene[columns])
    return pd.concat(frames, ignore_index=True)


def load_predictions(results_dir: Path) -> pd.DataFrame:
    frames = []
    for model, relative_path in PREDICTION_FILES.items():
        path = results_dir / relative_path.name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "model", model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_catalog(
    output_path: Path,
    *,
    metadata_path: Path = Path("data/processed/metadata_clean.csv"),
    results_dir: Path = Path("results"),
    schema_path: Path = Path("learning/sql/schema.sql"),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    samples = pd.read_csv(metadata_path)
    if not samples["sample_id"].is_unique:
        raise ValueError("metadata sample_id must be unique before SQL import")
    samples = samples[
        [
            "sample_id",
            "study_name",
            "study_condition",
            "label",
            "age",
            "gender",
            "BMI",
            "country",
            "sequencing_platform",
            "number_reads",
        ]
    ]
    folds = load_fold_results(results_dir)
    predictions = load_predictions(results_dir)

    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        samples.to_sql("samples", connection, if_exists="append", index=False)
        folds.to_sql("fold_results", connection, if_exists="append", index=False)
        predictions.to_sql(
            "predictions", connection, if_exists="append", index=False
        )
        connection.execute(
            "INSERT INTO catalog_metadata(key, value) VALUES (?, ?)",
            ("source", "crc-metagenomics committed CSV outputs"),
        )
        connection.commit()
        check = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {check}")
    finally:
        connection.close()

    os.replace(temporary, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/crc_research.sqlite"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_catalog(args.output)
    with sqlite3.connect(path) as connection:
        n_samples = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        n_results = connection.execute(
            "SELECT COUNT(*) FROM fold_results"
        ).fetchone()[0]
        n_predictions = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
    print(
        f"Built {path}: {n_samples} samples, {n_results} fold results, "
        f"{n_predictions} predictions"
    )


if __name__ == "__main__":
    main()
