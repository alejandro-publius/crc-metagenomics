"""Audit published colibactin CRISPRi guides in a frozen reference panel."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd


DNA = re.compile(r"^[ACGT]{20}$")
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    pieces: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(">"):
            if name is not None:
                records.append((name, "".join(pieces).upper()))
            name = raw[1:].split()[0]
            pieces = []
        elif raw.strip():
            pieces.append(raw.strip())
    if name is not None:
        records.append((name, "".join(pieces).upper()))
    if not records:
        raise ValueError(f"no FASTA records found in {path}")
    return records


def find_pam_sites(sequence: str, spacer: str) -> list[dict[str, object]]:
    """Find exact spacer sites with an NGG PAM on either DNA strand."""

    if not DNA.fullmatch(spacer):
        raise ValueError("spacer must contain exactly 20 A/C/G/T bases")
    sequence = sequence.upper()
    rc = reverse_complement(spacer)
    hits: list[dict[str, object]] = []
    for start in range(0, len(sequence) - 22):
        if sequence[start : start + 20] == spacer:
            pam = sequence[start + 20 : start + 23]
            if pam[1:] == "GG":
                hits.append({"start_1based": start + 1, "strand": "+", "pam": pam})
    for start in range(3, len(sequence) - 19):
        if sequence[start : start + 20] == rc:
            pam_on_plus = sequence[start - 3 : start]
            if pam_on_plus[:2] == "CC":
                hits.append(
                    {
                        "start_1based": start + 1,
                        "strand": "-",
                        "pam": reverse_complement(pam_on_plus),
                    }
                )
    return hits


def fetch_fasta(accession: str, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        payload = response.read()
    if url.endswith(".gz"):
        payload = gzip.decompress(payload)
    if not payload.startswith(b">"):
        raise RuntimeError(f"NCBI did not return FASTA for {accession}")
    destination.write_bytes(payload)


def validate_guides(guides: pd.DataFrame) -> None:
    required = {
        "guide_id",
        "target_id",
        "target_gene",
        "spacer_5to3",
        "reverse_complement_5to3",
        "benchmark_role",
    }
    missing = sorted(required - set(guides.columns))
    if missing:
        raise ValueError(f"guide manifest is missing columns: {missing}")
    if guides["guide_id"].duplicated().any():
        raise ValueError("guide IDs must be unique")
    for row in guides.itertuples(index=False):
        spacer = str(row.spacer_5to3).upper()
        if not DNA.fullmatch(spacer):
            raise ValueError(f"invalid spacer for {row.guide_id}")
        if reverse_complement(spacer) != str(row.reverse_complement_5to3).upper():
            raise ValueError(f"reverse-complement mismatch for {row.guide_id}")


def audit_panel(
    guides: pd.DataFrame, genomes: pd.DataFrame, fasta_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_guides(guides)
    if genomes["accession"].duplicated().any():
        raise ValueError("genome accessions must be unique")
    rows: list[dict[str, object]] = []
    for genome in genomes.itertuples(index=False):
        fasta_path = fasta_dir / f"{genome.accession}.fasta"
        if not fasta_path.exists() or not fasta_path.read_bytes().startswith(b">"):
            fetch_fasta(genome.accession, genome.download_url, fasta_path)
        records = read_fasta(fasta_path)
        for guide in guides.itertuples(index=False):
            all_hits: list[dict[str, object]] = []
            for record_id, sequence in records:
                for hit in find_pam_sites(sequence, guide.spacer_5to3):
                    all_hits.append({"record_id": record_id, **hit})
            rows.append(
                {
                    "guide_id": guide.guide_id,
                    "target_id": guide.target_id,
                    "target_gene": guide.target_gene,
                    "benchmark_role": guide.benchmark_role,
                    "accession": genome.accession,
                    "strain": genome.strain,
                    "n_fasta_records": len(records),
                    "n_exact_pam_sites": len(all_hits),
                    "covered": len(all_hits) >= 1,
                    "unique_site": len(all_hits) == 1,
                    "site_details": json.dumps(all_hits, separators=(",", ":")),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["target_id", "guide_id", "target_gene", "benchmark_role"],
            as_index=False,
        )
        .agg(
            n_genomes=("accession", "nunique"),
            n_genomes_covered=("covered", "sum"),
            n_genomes_unique_site=("unique_site", "sum"),
        )
        .sort_values(["benchmark_role", "guide_id"])
    )
    summary["coverage_fraction"] = summary["n_genomes_covered"] / summary["n_genomes"]
    summary["unique_site_fraction"] = (
        summary["n_genomes_unique_site"] / summary["n_genomes"]
    )
    summary["pilot_conservation_gate"] = (
        summary["coverage_fraction"].ge(0.8) & summary["unique_site_fraction"].ge(0.8)
    ).map({True: "pass", False: "not_passed"})
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=Path("data/interim/colibactin_reference_panel"),
    )
    args = parser.parse_args()
    research = Path("research/intervention_readiness")
    output = Path("results/intervention_readiness")
    guides = pd.read_csv(research / "published_colibactin_guides.csv")
    genomes = pd.read_csv(research / "colibactin_reference_genomes.csv")
    detail, summary = audit_panel(guides, genomes, args.fasta_dir)
    detail.to_csv(output / "colibactin_guide_reference_panel.csv", index=False)
    summary.to_csv(output / "colibactin_guide_conservation_summary.csv", index=False)
    audit = {
        "status": "complete_literature_defined_reference_panel_pilot",
        "n_genomes": int(genomes["accession"].nunique()),
        "n_guides": int(guides["guide_id"].nunique()),
        "n_guides_passing_pilot": int(
            summary["pilot_conservation_gate"].eq("pass").sum()
        ),
        "protocol": (
            "research/intervention_readiness/colibactin_sequence_audit_protocol.md"
        ),
        "claim_boundary": (
            "Exact target conservation in a seven-genome reference panel is "
            "not comprehensive human-strain conservation, near-match "
            "specificity, safety, delivery, or therapeutic readiness."
        ),
    }
    (output / "colibactin_guide_conservation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
