"""Build and freeze a hypothesis-driven CRC microbial mechanism panel.

The candidate genes and organism scopes below come from prespecified
experimental mechanisms, not from associations in this dataset. The script
queries UniProtKB, maps accessions to UniRef90, intersects the mapping with the
label-independent cohort scans, and writes a checksum-protected manifest.

Usage:
    python3 scripts/build_mechanism_panel.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_MAPPING = "https://rest.uniprot.org/idmapping"
SOURCE_SNAPSHOT = "2026-08-08"


@dataclass(frozen=True)
class Candidate:
    mechanism: str
    gene: str
    query: str
    evidence_url: str


def _candidate(mechanism: str, gene: str, query: str, evidence: str) -> Candidate:
    return Candidate(mechanism, gene, query, evidence)


COLIBACTIN_EVIDENCE = "https://www.nature.com/articles/s41586-020-2080-8"
FUSOBACTERIUM_EVIDENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5465824/"
BFT_EVIDENCE = "https://pubmed.ncbi.nlm.nih.gov/29398651/"
BAI_EVIDENCE = "https://pubmed.ncbi.nlm.nih.gov/36343662/"

CANDIDATES = [
    *[
        _candidate(
            "colibactin_genotoxicity",
            f"clb{letter}",
            f"(gene_exact:clb{letter}) AND (taxonomy_id:562)",
            COLIBACTIN_EVIDENCE,
        )
        for letter in "ABCDEFGHIJKLMNOPQRS"
    ],
    _candidate(
        "fusobacterial_adhesion",
        "fadA",
        "(gene_exact:fadA) AND (taxonomy_id:851)",
        FUSOBACTERIUM_EVIDENCE,
    ),
    _candidate(
        "fusobacterial_adhesion",
        "fap2",
        '(gene:fap2) AND (organism_name:"Fusobacterium nucleatum")',
        FUSOBACTERIUM_EVIDENCE,
    ),
    _candidate(
        "b_fragilis_toxin",
        "bft",
        "(gene:bft) AND (taxonomy_id:817)",
        BFT_EVIDENCE,
    ),
    _candidate(
        "secondary_bile_acid",
        "baiA1",
        '(gene:baiA1) AND (organism_name:"Clostridium scindens")',
        BAI_EVIDENCE,
    ),
    _candidate(
        "secondary_bile_acid",
        "baiA2",
        '(gene:baiA2) AND (organism_name:"Clostridium scindens")',
        BAI_EVIDENCE,
    ),
    *[
        _candidate(
            "secondary_bile_acid",
            f"bai{letter}",
            f'(gene:bai{letter}) AND (organism_name:"Clostridium scindens")',
            BAI_EVIDENCE,
        )
        for letter in ["B", "C", "D", "E", "F", "H", "J", "P"]
    ],
]


SEARCH_FIELDS = [
    "accession",
    "gene_names",
    "organism_name",
    "protein_name",
    "reviewed",
]


def search_candidate(session: requests.Session, candidate: Candidate) -> list[dict[str, str]]:
    response = session.get(
        UNIPROT_SEARCH,
        params={
            "query": candidate.query,
            "format": "tsv",
            "fields": ",".join(SEARCH_FIELDS),
            "size": 500,
        },
        timeout=120,
    )
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text), delimiter="\t"))
    return [
        {
            "mechanism": candidate.mechanism,
            "prespecified_gene": candidate.gene,
            "query": candidate.query,
            "evidence_url": candidate.evidence_url,
            "accession": row["Entry"],
            "gene_names": row["Gene Names"],
            "organism": row["Organism"],
            "protein_name": row["Protein names"],
            "reviewed": row["Reviewed"],
            "query_status": "mapped_candidate",
        }
        for row in rows
    ]


def map_accessions_to_uniref90(
    session: requests.Session, accessions: list[str]
) -> dict[str, str]:
    if not accessions:
        return {}
    response = session.post(
        f"{UNIPROT_MAPPING}/run",
        data={
            "from": "UniProtKB_AC-ID",
            "to": "UniRef90",
            "ids": ",".join(sorted(set(accessions))),
        },
        timeout=120,
    )
    response.raise_for_status()
    job_id = response.json()["jobId"]
    for _ in range(120):
        status = session.get(
            f"{UNIPROT_MAPPING}/status/{job_id}", timeout=120
        )
        status.raise_for_status()
        payload = status.json()
        if payload.get("jobStatus") == "FINISHED" or "jobStatus" not in payload:
            break
        if payload.get("jobStatus") == "ERROR":
            raise RuntimeError(f"UniProt mapping job failed: {job_id}")
        time.sleep(1)
    else:
        raise TimeoutError(f"UniProt mapping job did not finish: {job_id}")

    result = session.get(
        f"{UNIPROT_MAPPING}/results/{job_id}",
        params={"format": "tsv", "size": 500},
        timeout=120,
    )
    result.raise_for_status()
    return {
        row["From"]: row["To"]
        for row in csv.DictReader(io.StringIO(result.text), delimiter="\t")
    }


def scan_coverage(scan_dir: Path, clusters: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(scan_dir.glob("*.csv.gz")):
        cohort = path.name.removesuffix(".csv.gz")
        with gzip.open(path, "rt") as handle:
            frame = pd.read_csv(
                handle,
                usecols=["gene_id", "n_samples", "n_nonzero", "total_abundance"],
            )
        found = frame[frame["gene_id"].isin(clusters)].copy()
        if not found.empty:
            found.insert(0, "cohort", cohort)
            rows.append(found)
    if not rows:
        return pd.DataFrame(
            columns=[
                "cohort",
                "gene_id",
                "n_samples",
                "n_nonzero",
                "total_abundance",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["gene_id", "cohort"], kind="mergesort"
    )


def canonical_manifest_bytes(manifest: pd.DataFrame) -> bytes:
    columns = [
        "mechanism",
        "prespecified_gene",
        "accession",
        "uniref90",
        "organism",
        "protein_name",
        "reviewed",
        "evidence_url",
        "query_status",
        "n_cohorts_detected",
    ]
    ordered = manifest[columns].sort_values(
        ["mechanism", "prespecified_gene", "accession"], kind="mergesort"
    )
    return ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")


def build_panel(scan_dir: Path, output_dir: Path) -> None:
    session = requests.Session()
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        hits = search_candidate(session, candidate)
        if hits:
            rows.extend(hits)
        else:
            rows.append(
                {
                    "mechanism": candidate.mechanism,
                    "prespecified_gene": candidate.gene,
                    "query": candidate.query,
                    "evidence_url": candidate.evidence_url,
                    "accession": "",
                    "gene_names": "",
                    "organism": "",
                    "protein_name": "",
                    "reviewed": "",
                    "query_status": "no_uniprot_hit",
                }
            )

    mapping = map_accessions_to_uniref90(
        session, [row["accession"] for row in rows if row["accession"]]
    )
    for row in rows:
        row["uniref90"] = mapping.get(row["accession"], "")
        if row["accession"] and not row["uniref90"]:
            row["query_status"] = "no_uniref90_mapping"

    audit = pd.DataFrame(rows).sort_values(
        ["mechanism", "prespecified_gene", "accession"], kind="mergesort"
    )
    clusters = set(audit.loc[audit["uniref90"] != "", "uniref90"])
    coverage = scan_coverage(scan_dir, clusters)
    cohort_counts = coverage.groupby("gene_id")["cohort"].nunique().to_dict()
    audit["n_cohorts_detected"] = audit["uniref90"].map(cohort_counts).fillna(0).astype(int)
    audit.loc[
        (audit["uniref90"] != "") & (audit["n_cohorts_detected"] == 0),
        "query_status",
    ] = "mapped_but_not_detected"
    audit.loc[
        (audit["uniref90"] != "") & (audit["n_cohorts_detected"] > 0),
        "query_status",
    ] = "frozen_detected"

    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "uniprot_to_uniref90.csv", index=False)
    coverage.to_csv(output_dir / "cohort_coverage.csv", index=False)

    manifest = audit.drop(columns=["query", "gene_names"]).copy()
    manifest_bytes = canonical_manifest_bytes(manifest)
    checksum = hashlib.sha256(manifest_bytes).hexdigest()
    (output_dir / "frozen_manifest.csv").write_bytes(manifest_bytes)
    freeze = {
        "status": "frozen_before_outcome_modeling",
        "source_snapshot": SOURCE_SNAPSHOT,
        "manifest_sha256": checksum,
        "n_prespecified_genes": len(CANDIDATES),
        "n_accessions": int((audit["accession"] != "").sum()),
        "n_uniref90_clusters": int(len(clusters)),
        "n_detected_clusters": int(coverage["gene_id"].nunique()),
        "selection_used_outcome_labels": False,
    }
    (output_dir / "freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scan-dir", type=Path, default=Path("data/interim/gene_family_scan")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/mechanism_panel")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_panel(args.scan_dir, args.output_dir)
