#!/usr/bin/env python3
"""Harmonize external MetaPhlAn profiles and score the locked species RF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_cohort"


def harmonize_metaphlan(table: pd.DataFrame, retained: list[str]) -> pd.DataFrame:
    tax_col = "clade_name" if "clade_name" in table.columns else table.columns[0]
    table = table.set_index(tax_col)
    species = table.loc[
        table.index.astype(str).str.contains(r"(?:^|\|)s__", regex=True)
        & ~table.index.astype(str).str.contains(r"\|t__", regex=True)
    ].T.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    species.index.name = "run_accession"
    aligned = species.reindex(columns=retained, fill_value=0.0)
    row_sum = aligned.sum(axis=1).replace(0, np.nan)
    aligned = aligned.div(row_sum, axis=0).fillna(0.0)
    return np.log10(aligned + 1e-6)


def normalize_species_name(value: str) -> str:
    """Normalize display-only taxonomy differences without synonym guessing."""
    value = str(value).replace("_", " ")
    value = re.sub(r"[\[\]]", "", value)
    return " ".join(value.casefold().split())


def harmonize_gmrepo(table: pd.DataFrame, retained: list[str]) -> pd.DataFrame:
    """Map GMrepo scientific names to the frozen cMD species feature names.

    Mapping is based only on normalized terminal species names. No synonyms or
    performance-informed mappings are allowed. Abundances are renormalized over
    the locked training feature space exactly as in ``harmonize_metaphlan``.
    """
    required = {"run_accession", "scientific_name", "relative_abundance"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"GMrepo table is missing columns: {sorted(missing)}")
    lookup = {
        normalize_species_name(feature.rsplit("s__", 1)[-1]): feature
        for feature in retained
    }
    work = table.copy()
    work["feature"] = work.scientific_name.map(normalize_species_name).map(lookup)
    work["relative_abundance"] = pd.to_numeric(
        work.relative_abundance, errors="coerce"
    ).fillna(0.0)
    run_ids = pd.Index(work.run_accession.astype(str).unique(), name="run_accession")
    matched = work.dropna(subset=["feature"])
    species = matched.pivot_table(
        index="run_accession", columns="feature", values="relative_abundance", aggfunc="sum"
    )
    aligned = species.reindex(index=run_ids, columns=retained, fill_value=0.0).fillna(0.0)
    row_sum = aligned.sum(axis=1).replace(0, np.nan)
    aligned = aligned.div(row_sum, axis=0).fillna(0.0)
    return np.log10(aligned + 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--format", choices=["metaphlan", "gmrepo"], default="metaphlan")
    args = parser.parse_args()

    train_x = pd.read_csv(ROOT / "data/processed/species_filtered.csv")
    train_meta = pd.read_csv(ROOT / "data/processed/metadata_clean.csv")
    training = train_meta.merge(train_x, on="sample_id")
    training = training[training.label.isin([0, 1])]
    retained = [c for c in train_x.columns if c != "sample_id"]

    if args.format == "metaphlan":
        external_raw = pd.read_csv(args.profiles, sep="\t", comment="#")
        external_x = harmonize_metaphlan(external_raw, retained)
        observed = set(external_raw.iloc[:, 0].astype(str))
        feature_overlap = sum(feature in observed for feature in retained)
    else:
        external_raw = pd.read_csv(args.profiles)
        external_x = harmonize_gmrepo(external_raw, retained)
        observed = set(external_raw.scientific_name.map(normalize_species_name))
        feature_overlap = sum(
            normalize_species_name(feature.rsplit("s__", 1)[-1]) in observed
            for feature in retained
        )
    manifest = pd.read_csv(OUT / "manifest.csv").set_index("run_accession")
    common = external_x.index.intersection(manifest.index)
    if len(common) < 2 or manifest.loc[common, "label"].nunique() < 2:
        raise ValueError("Profiles must cover at least one CRC and one control run")

    model = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    model.fit(training[retained], training.label)
    probabilities = model.predict_proba(external_x.loc[common, retained])[:, 1]
    predictions = manifest.loc[common].reset_index()[
        ["run_accession", "sample_alias", "study_condition", "age_group", "label"]
    ]
    predictions["y_prob"] = probabilities
    predictions.to_csv(OUT / "predictions.csv", index=False)
    metrics = {
        "n": len(predictions),
        "n_crc": int(predictions.label.sum()),
        "n_control": int((predictions.label == 0).sum()),
        "auc": float(roc_auc_score(predictions.label, predictions.y_prob)),
        "average_precision": float(average_precision_score(predictions.label, predictions.y_prob)),
        "profile_format": args.format,
        "feature_overlap": int(feature_overlap),
        "retained_features": len(retained),
        "interpretation": "untouched external evaluation",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
