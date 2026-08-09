#!/usr/bin/env python3
"""Run a bounded raw-read recovery pilot for the frozen mechanism panel.

This is deliberately independent of the HUMAnN/UniRef abundance tables used
to define the main mechanism scores. It downloads a deterministic balanced
sample, translates a fixed number of reads, and aligns them directly to the
prespecified UniProt protein sequences with DIAMOND.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import ssl
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
import certifi


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "independent_profiler"
WORK = ROOT / "data" / "interim" / "independent_profiler"


def select_pilot(manifest: pd.DataFrame) -> pd.DataFrame:
    """Choose one control and one CRC sample from each fixed pilot cohort."""
    rows = []
    for cohort in ("GuptaA_2019", "WirbelJ_2018"):
        for label in (0, 1):
            candidates = manifest[
                (manifest.study_name == cohort)
                & (manifest.label == label)
                & manifest.NCBI_accession.notna()
                & ~manifest.NCBI_accession.astype(str).str.contains(";", regex=False)
            ].sort_values(["sample_id", "NCBI_accession"])
            if candidates.empty:
                raise ValueError(f"No single-accession pilot sample for {cohort}/{label}")
            rows.append(candidates.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def download_panel_fasta(mapping: pd.DataFrame, output: Path) -> None:
    accessions = sorted(set(mapping.accession.dropna().astype(str)) - {""})
    context = ssl.create_default_context(cafile=certifi.where())
    records = []
    for accession in accessions:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            text = response.read().decode("utf-8")
        lines = text.rstrip().splitlines()
        if not lines or not lines[0].startswith(">"):
            raise RuntimeError(f"No FASTA returned for {accession}")
        lines[0] = f">{accession}"
        records.append("\n".join(lines))
    output.write_text("\n".join(records) + "\n")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def summarize_hits(pilot: pd.DataFrame, mapping: pd.DataFrame, max_reads: int) -> None:
    annotations = mapping.dropna(subset=["accession"])[
        ["accession", "mechanism", "prespecified_gene"]
    ].drop_duplicates()
    hit_rows = []
    sample_rows = []
    for sample in pilot.itertuples(index=False):
        hit_path = WORK / f"{sample.NCBI_accession}.hits.tsv"
        if hit_path.exists() and hit_path.stat().st_size:
            hits = pd.read_csv(
                hit_path,
                sep="\t",
                names=["read_id", "accession", "pident", "alignment_length",
                       "evalue", "bitscore"],
            )
            hits = hits[hits.alignment_length >= 30].copy()
            hits = hits.merge(annotations, on="accession", how="left")
            hits.insert(0, "sample_id", sample.sample_id)
            hits.insert(1, "study_name", sample.study_name)
            hits.insert(2, "label", sample.label)
            hit_rows.append(hits)
            mechanisms = int(hits.mechanism.nunique())
            genes = int(hits.prespecified_gene.nunique())
            unique_reads = int(hits.read_id.nunique())
        else:
            mechanisms = genes = unique_reads = 0
        sample_rows.append({
            "sample_id": sample.sample_id,
            "study_name": sample.study_name,
            "label": int(sample.label),
            "NCBI_accession": sample.NCBI_accession,
            "reads_requested": max_reads,
            "unique_reads_with_panel_hit": unique_reads,
            "genes_detected": genes,
            "mechanisms_detected": mechanisms,
        })

    columns = ["sample_id", "study_name", "label", "read_id", "accession",
               "pident", "alignment_length", "evalue", "bitscore",
               "mechanism", "prespecified_gene"]
    all_hits = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame(columns=columns)
    all_hits.to_csv(RESULTS / "targeted_hits.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(RESULTS / "pilot_summary.csv", index=False)

    audit = {
        "status": "completed_bounded_pilot",
        "scope": "direct targeted protein recovery; not a full independent functional profile",
        "selection": "first lexical single-accession control and CRC in two prespecified cohorts",
        "cohorts": 2,
        "samples": 4,
        "reads_requested_per_sample": max_reads,
        "panel_accessions": int(annotations.accession.nunique()),
        "samples_with_any_hit": int(sum(r["unique_reads_with_panel_hit"] > 0 for r in sample_rows)),
        "minimum_protein_identity_percent": 90,
        "minimum_alignment_amino_acids": 30,
        "limitations": [
            "The read cap estimates technical recoverability, not absence.",
            "Four samples cannot estimate clinical discrimination.",
            "Short targeted matches do not by themselves establish a complete operon.",
            "A full independent profiler and external cohort remain required for confirmation.",
        ],
    }
    (RESULTS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reads", type=int, default=250_000)
    args = parser.parse_args()
    for executable in ("fastq-dump", "diamond"):
        if shutil.which(executable) is None:
            raise SystemExit(f"Missing required executable: {executable}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULTS / "accession_manifest.csv")
    mapping = pd.read_csv(ROOT / "results" / "mechanism_panel" / "uniprot_to_uniref90.csv")
    pilot = select_pilot(manifest)
    pilot.to_csv(RESULTS / "pilot_samples.csv", index=False)

    fasta = WORK / "mechanism_proteins.fasta"
    database = WORK / "mechanism_proteins"
    download_panel_fasta(mapping, fasta)
    run(["diamond", "makedb", "--in", str(fasta), "-d", str(database), "--quiet"])

    for sample in pilot.itertuples(index=False):
        accession = str(sample.NCBI_accession)
        reads = WORK / f"{accession}.fastq.gz"
        if not reads.exists():
            run(["fastq-dump", "--gzip", "-X", str(args.max_reads),
                 "--outdir", str(WORK), accession])
        hit_path = WORK / f"{accession}.hits.tsv"
        run([
            "diamond", "blastx", "--db", str(database), "--query", str(reads),
            "--out", str(hit_path), "--outfmt", "6", "qseqid", "sseqid",
            "pident", "length", "evalue", "bitscore", "--sensitive",
            "--max-target-seqs", "1", "--evalue", "1e-5", "--quiet",
            "--id", "90",
        ])
    summarize_hits(pilot, mapping, args.max_reads)


if __name__ == "__main__":
    main()
