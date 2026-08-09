"""Audit published colibactin guides in a frozen human-isolate panel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from audit_published_colibactin_guides import audit_panel, fetch_fasta


GROUP_COLUMNS = [
    "target_id",
    "guide_id",
    "target_gene",
    "benchmark_role",
]


def fetch_missing_fastas(genomes: pd.DataFrame, fasta_dir: Path, workers: int) -> None:
    fasta_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for genome in genomes.itertuples(index=False):
        destination = fasta_dir / f"{genome.accession}.fasta"
        if not destination.exists() or not destination.read_bytes().startswith(b">"):
            missing.append((genome.accession, genome.download_url, destination))
    if not missing:
        return

    def fetch(specification: tuple[str, str, Path]) -> str:
        accession, url, destination = specification
        fetch_fasta(accession, url, destination)
        return accession

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, specification) for specification in missing]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(missing):
                print(
                    f"downloaded {completed}/{len(missing)} missing assemblies",
                    flush=True,
                )


def _coverage_summary(detail: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        detail.groupby(groups, as_index=False)
        .agg(
            n_genomes=("accession", "nunique"),
            n_genomes_covered=("covered", "sum"),
            n_genomes_unique_site=("unique_site", "sum"),
        )
        .assign(
            coverage_fraction=lambda x: x["n_genomes_covered"] / x["n_genomes"],
            unique_site_fraction=lambda x: x["n_genomes_unique_site"] / x["n_genomes"],
        )
    )


def summarize_human_panel(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(
        GROUP_COLUMNS + ["accession", "source_group", "covered", "unique_site"]
    )
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"human-isolate detail is missing columns: {missing}")

    source = _coverage_summary(detail, GROUP_COLUMNS + ["source_group"])
    source["source_coverage_gate"] = (
        source["coverage_fraction"].ge(0.8).map({True: "pass", False: "not_passed"})
    )
    overall = _coverage_summary(detail, GROUP_COLUMNS)
    minimum_source = (
        source.groupby(GROUP_COLUMNS, as_index=False)["coverage_fraction"]
        .min()
        .rename(columns={"coverage_fraction": "minimum_source_coverage_fraction"})
    )
    overall = overall.merge(minimum_source, on=GROUP_COLUMNS, validate="one_to_one")
    passed = (
        overall["coverage_fraction"].ge(0.9)
        & overall["unique_site_fraction"].ge(0.9)
        & overall["minimum_source_coverage_fraction"].ge(0.8)
    )
    overall["human_isolate_conservation_gate"] = passed.map(
        {True: "pass", False: "not_passed"}
    )
    return (
        overall.sort_values(["benchmark_role", "guide_id"]),
        source.sort_values(["benchmark_role", "guide_id", "source_group"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=Path("data/interim/colibactin_human_isolate_panel"),
    )
    parser.add_argument("--download-workers", type=int, default=6)
    args = parser.parse_args()

    research = Path("research/intervention_readiness")
    output = Path("results/intervention_readiness")
    guides = pd.read_csv(research / "published_colibactin_guides.csv")
    panel = pd.read_csv(research / "colibactin_human_isolate_panel.csv")
    genomes = panel.rename(columns={"wgs_accession": "accession"})
    fetch_missing_fastas(genomes, args.fasta_dir, args.download_workers)
    detail, _ = audit_panel(guides, genomes, args.fasta_dir)
    metadata = genomes[
        [
            "accession",
            "source_table_row",
            "host",
            "country",
            "isolation_source",
            "source_group",
            "pathotype",
            "phylogroup",
            "sequence_type",
            "pks_status",
            "short_read_run",
        ]
    ]
    detail = detail.merge(metadata, on="accession", validate="many_to_one")
    detail = detail.sort_values(["benchmark_role", "guide_id", "source_table_row"])
    summary, source_summary = summarize_human_panel(detail)

    detail.to_csv(
        output / "colibactin_human_isolate_conservation_detail.csv", index=False
    )
    summary.to_csv(
        output / "colibactin_human_isolate_conservation_summary.csv", index=False
    )
    source_summary.to_csv(
        output / "colibactin_human_isolate_source_summary.csv", index=False
    )
    audit = {
        "status": "complete_frozen_human_isolate_conservation_audit",
        "n_human_isolates": int(panel["wgs_accession"].nunique()),
        "n_fecal_commensal_isolates": int(
            panel["source_group"].eq("fecal_commensal").sum()
        ),
        "n_extraintestinal_clinical_isolates": int(
            panel["source_group"].eq("extraintestinal_clinical").sum()
        ),
        "n_guides": int(guides["guide_id"].nunique()),
        "n_guides_passing": int(
            summary["human_isolate_conservation_gate"].eq("pass").sum()
        ),
        "protocol": (
            "research/intervention_readiness/colibactin_human_isolate_protocol.md"
        ),
        "claim_boundary": (
            "Exact-site conservation in one Japanese human-isolate collection "
            "does not establish global patient coverage, expression, efficacy, "
            "delivery, safety, or therapeutic readiness."
        ),
    }
    (output / "colibactin_human_isolate_conservation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(source_summary.to_string(index=False))


if __name__ == "__main__":
    main()
