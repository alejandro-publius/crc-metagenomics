"""Select gene families independently inside each country-aware LODO fold.

This is pass 2 of the scalable gene-family workflow. It consumes compact
per-cohort summaries from ``scan_gene_families.R`` and writes one feature
manifest per held-out cohort. Only training-cohort statistics determine a
fold's manifest.

Usage:
    python3 scripts/select_gene_family_manifests.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lodo_cv import get_lodo_splits


REQUIRED_SUMMARY_COLUMNS = {
    "cohort",
    "gene_id",
    "n_samples",
    "n_nonzero",
    "total_abundance",
}


def build_training_cohort_map(metadata: pd.DataFrame) -> dict[str, list[str]]:
    """Return the training cohorts used for each country-aware LODO fold."""
    country_col = "country" if "country" in metadata.columns else None
    folds: dict[str, list[str]] = {}
    for held_out, train_idx, _test_idx, _excluded in get_lodo_splits(
        metadata, country_col=country_col
    ):
        folds[held_out] = sorted(
            metadata.iloc[train_idx]["study_name"].unique().tolist()
        )
    return folds


def select_fold_features(
    summaries: pd.DataFrame,
    training_cohorts: list[str],
    cohort_sample_counts: dict[str, int],
    *,
    min_prevalence: float = 0.05,
    min_cohorts: int = 2,
    max_features: int = 5_000,
) -> pd.DataFrame:
    """Rank genes using only aggregate statistics from training cohorts."""
    if not REQUIRED_SUMMARY_COLUMNS.issubset(summaries.columns):
        missing = sorted(REQUIRED_SUMMARY_COLUMNS - set(summaries.columns))
        raise ValueError(f"summary table is missing columns: {missing}")
    if not training_cohorts:
        raise ValueError("training_cohorts cannot be empty")

    unknown = sorted(set(training_cohorts) - set(cohort_sample_counts))
    if unknown:
        raise ValueError(f"missing sample counts for cohorts: {unknown}")

    train = summaries[summaries["cohort"].isin(training_cohorts)].copy()
    n_train = sum(cohort_sample_counts[c] for c in training_cohorts)
    if n_train <= 0:
        raise ValueError("training sample count must be positive")

    grouped = (
        train.groupby("gene_id", sort=False)
        .agg(
            n_nonzero=("n_nonzero", "sum"),
            total_abundance=("total_abundance", "sum"),
            n_cohorts=("cohort", "nunique"),
        )
        .reset_index()
    )
    grouped["prevalence"] = grouped["n_nonzero"] / n_train
    grouped["mean_abundance"] = grouped["total_abundance"] / n_train
    grouped = grouped[
        (grouped["prevalence"] >= min_prevalence)
        & (grouped["n_cohorts"] >= min_cohorts)
    ]
    grouped = grouped.sort_values(
        ["prevalence", "n_cohorts", "mean_abundance", "gene_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(max_features)
    return grouped.reset_index(drop=True)


def load_summaries(scan_dir: Path) -> pd.DataFrame:
    files = sorted(scan_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"no cohort scans found in {scan_dir}; run scan_gene_families.R first"
        )
    frames = [pd.read_csv(path) for path in files]
    summaries = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_SUMMARY_COLUMNS - set(summaries.columns)
    if missing:
        raise ValueError(f"scan files are missing columns: {sorted(missing)}")
    return summaries


def write_manifests(
    summaries: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    *,
    min_prevalence: float,
    min_cohorts: int,
    max_features: int,
) -> pd.DataFrame:
    metadata = metadata[metadata["label"].isin([0, 1])].reset_index(drop=True)
    sample_counts = metadata.groupby("study_name").size().to_dict()
    training_map = build_training_cohort_map(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, object]] = []
    union: set[str] = set()
    for held_out, training_cohorts in sorted(training_map.items()):
        selected = select_fold_features(
            summaries,
            training_cohorts,
            sample_counts,
            min_prevalence=min_prevalence,
            min_cohorts=min_cohorts,
            max_features=max_features,
        )
        selected.insert(0, "held_out_cohort", held_out)
        selected.insert(1, "rank", range(1, len(selected) + 1))
        selected.to_csv(output_dir / f"{held_out}.csv", index=False)
        union.update(selected["gene_id"])
        audit_rows.append(
            {
                "held_out_cohort": held_out,
                "training_cohorts": ";".join(training_cohorts),
                "n_training_samples": sum(sample_counts[c] for c in training_cohorts),
                "n_selected_genes": len(selected),
            }
        )

    union_path = output_dir / "selected_union.txt"
    union_path.write_text("\n".join(sorted(union)) + "\n", encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "manifest_summary.csv", index=False)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir", type=Path, default=Path("data/interim/gene_family_scan")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/metadata_clean.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/gene_family_manifests")
    )
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--min-cohorts", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.min_prevalence <= 1:
        raise ValueError("--min-prevalence must be in (0, 1]")
    if args.min_cohorts < 1 or args.max_features < 1:
        raise ValueError("--min-cohorts and --max-features must be positive")

    summaries = load_summaries(args.scan_dir)
    metadata = pd.read_csv(args.metadata)
    audit = write_manifests(
        summaries,
        metadata,
        args.output_dir,
        min_prevalence=args.min_prevalence,
        min_cohorts=args.min_cohorts,
        max_features=args.max_features,
    )
    print(audit.to_string(index=False))
    print(f"Wrote fold manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
