"""Screen published colibactin spacers against frozen protected references."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import ssl
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import pandas as pd

from audit_published_colibactin_guides import reverse_complement, validate_guides


DNA = re.compile(r"^[ACGT]{20}$")


def iter_fasta_records(path: Path) -> Iterator[tuple[str, bytearray]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        name: str | None = None
        sequence = bytearray()
        for raw in handle:
            if raw.startswith(b">"):
                if name is not None:
                    yield name, sequence
                name = raw[1:].split(maxsplit=1)[0].decode("ascii")
                sequence = bytearray()
            else:
                sequence.extend(raw.strip().upper())
        if name is not None:
            yield name, sequence


def download_reference(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    user_agent = {"User-Agent": "crc-readiness/0.1"}
    head = urllib.request.Request(url, headers=user_agent, method="HEAD")
    with urllib.request.urlopen(head, timeout=120, context=context) as response:
        content_length = int(response.headers["Content-Length"])

    n_parts = 8 if content_length >= 100_000_000 else 1
    chunk_size = (content_length + n_parts - 1) // n_parts
    ranges = [
        (
            index,
            index * chunk_size,
            min(content_length - 1, (index + 1) * chunk_size - 1),
        )
        for index in range(n_parts)
    ]

    def fetch_part(specification: tuple[int, int, int]) -> Path:
        index, start, stop = specification
        part = temporary.with_suffix(temporary.suffix + f".part{index:02d}")
        headers = {**user_agent, "Range": f"bytes={start}-{stop}"}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=300, context=context) as response:
            if n_parts > 1 and response.status != 206:
                raise RuntimeError("reference server ignored a byte-range request")
            with part.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        expected_size = stop - start + 1
        if part.stat().st_size != expected_size:
            raise RuntimeError(f"incomplete ranged download for {url}")
        return part

    temporary.unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=n_parts) as executor:
        parts = list(executor.map(fetch_part, ranges))
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            part.unlink()
    if temporary.stat().st_size != content_length:
        raise RuntimeError(f"incomplete download for {url}")
    temporary.replace(destination)


def hamming(left: bytes | bytearray, right: bytes) -> int:
    return sum(a != b for a, b in zip(left, right))


def _candidate_starts(
    sequence: bytes | bytearray, expected: bytes, seed_offset: int
) -> set[int]:
    fragments = [(expected[:7], 0), (expected[7:14], 7), (expected[14:], 14)]
    fragments.append((expected[seed_offset : seed_offset + 8], seed_offset))
    starts: set[int] = set()
    for fragment, offset in fragments:
        position = sequence.find(fragment)
        while position >= 0:
            starts.add(position - offset)
            position = sequence.find(fragment, position + 1)
    return starts


def find_flagged_sites(
    sequence: bytes | bytearray, spacer: str
) -> list[dict[str, object]]:
    if not DNA.fullmatch(spacer):
        raise ValueError("spacer must contain exactly 20 A/C/G/T bases")
    guide = spacer.encode("ascii")
    orientations = [
        ("+", guide, 12),
        ("-", reverse_complement(spacer).encode("ascii"), 0),
    ]
    hits: list[dict[str, object]] = []
    for strand, expected, seed_offset in orientations:
        for start in _candidate_starts(sequence, expected, seed_offset):
            if start < 0 or start + 20 > len(sequence):
                continue
            if strand == "+":
                if start + 23 > len(sequence):
                    continue
                pam = bytes(sequence[start + 20 : start + 23])
                if pam[1:] != b"GG":
                    continue
            else:
                if start < 3:
                    continue
                pam_on_plus = bytes(sequence[start - 3 : start])
                if pam_on_plus[:2] != b"CC":
                    continue
                pam = reverse_complement(pam_on_plus.decode("ascii")).encode("ascii")
            observed = bytes(sequence[start : start + 20])
            mismatches = hamming(observed, expected)
            seed_exact = (
                observed[seed_offset : seed_offset + 8]
                == expected[seed_offset : seed_offset + 8]
            )
            if mismatches <= 2 or (seed_exact and mismatches <= 4):
                guide_oriented_candidate = (
                    observed.decode("ascii")
                    if strand == "+"
                    else reverse_complement(observed.decode("ascii"))
                )
                hits.append(
                    {
                        "start_1based": start + 1,
                        "strand": strand,
                        "pam": pam.decode("ascii"),
                        "candidate_spacer_5to3": guide_oriented_candidate,
                        "mismatches": mismatches,
                        "pam_proximal_seed_exact": seed_exact,
                    }
                )
    return sorted(
        hits,
        key=lambda row: (
            int(row["mismatches"]),
            str(row["strand"]),
            int(row["start_1based"]),
        ),
    )


def screen_references(
    guides: pd.DataFrame, references: pd.DataFrame, reference_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_guides(guides)
    required = {
        "reference_id",
        "reference_class",
        "organism",
        "assembly_accession",
        "download_url",
    }
    missing = sorted(required - set(references.columns))
    if missing:
        raise ValueError(f"protected-reference manifest is missing columns: {missing}")
    if references["reference_id"].duplicated().any():
        raise ValueError("protected reference IDs must be unique")

    rows: list[dict[str, object]] = []
    for reference in references.itertuples(index=False):
        path = reference_dir / f"{reference.reference_id}.fna.gz"
        if not path.exists():
            download_reference(reference.download_url, path)
        guide_hits: dict[str, list[dict[str, object]]] = {
            guide.guide_id: [] for guide in guides.itertuples(index=False)
        }
        n_records = 0
        for record_id, sequence in iter_fasta_records(path):
            n_records += 1
            for guide in guides.itertuples(index=False):
                for hit in find_flagged_sites(sequence, guide.spacer_5to3):
                    guide_hits[guide.guide_id].append({"record_id": record_id, **hit})
        for guide in guides.itertuples(index=False):
            hits = guide_hits[guide.guide_id]
            mismatch_values = [int(hit["mismatches"]) for hit in hits]
            rows.append(
                {
                    "target_id": guide.target_id,
                    "guide_id": guide.guide_id,
                    "benchmark_role": guide.benchmark_role,
                    "reference_id": reference.reference_id,
                    "reference_class": reference.reference_class,
                    "organism": reference.organism,
                    "assembly_accession": reference.assembly_accession,
                    "n_fasta_records": n_records,
                    "n_flagged_sites": len(hits),
                    "minimum_mismatches": min(mismatch_values) if hits else pd.NA,
                    "n_sites_at_most_2_mismatches": sum(
                        v <= 2 for v in mismatch_values
                    ),
                    "n_seed_exact_sites_at_most_4_mismatches": sum(
                        bool(hit["pam_proximal_seed_exact"])
                        and int(hit["mismatches"]) <= 4
                        for hit in hits
                    ),
                    "screen_status": "flagged" if hits else "no_flagged_sites",
                    "site_details": json.dumps(hits, separators=(",", ":")),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["target_id", "guide_id", "benchmark_role"], as_index=False)
        .agg(
            n_references=("reference_id", "nunique"),
            n_bacterial_references=(
                "reference_class",
                lambda x: int((x == "gut_bacterial_reference").sum()),
            ),
            n_human_references=(
                "reference_class",
                lambda x: int((x == "human_reference").sum()),
            ),
            n_references_flagged=(
                "screen_status",
                lambda x: int((x == "flagged").sum()),
            ),
            n_flagged_sites=("n_flagged_sites", "sum"),
        )
        .sort_values(["benchmark_role", "guide_id"])
    )
    summary["protected_reference_pilot_gate"] = (
        summary["n_flagged_sites"].eq(0).map({True: "pass", False: "not_passed"})
    )
    return detail, summary


def main() -> None:
    research = Path("research/intervention_readiness")
    output = Path("results/intervention_readiness")
    reference_dir = Path("data/interim/protected_reference_panel")
    guides = pd.read_csv(research / "published_colibactin_guides.csv")
    references = pd.read_csv(research / "protected_reference_panel.csv")
    detail, summary = screen_references(guides, references, reference_dir)
    detail.to_csv(output / "colibactin_protected_reference_detail.csv", index=False)
    summary.to_csv(output / "colibactin_specificity_pilot_summary.csv", index=False)
    audit = {
        "status": "complete_frozen_protected_reference_pilot",
        "n_references": int(references["reference_id"].nunique()),
        "n_bacterial_references": int(
            references["reference_class"].eq("gut_bacterial_reference").sum()
        ),
        "n_human_references": int(
            references["reference_class"].eq("human_reference").sum()
        ),
        "n_guides": int(guides["guide_id"].nunique()),
        "protocol": "research/intervention_readiness/colibactin_specificity_protocol.md",
        "claim_boundary": (
            "A protected-reference pilot cannot establish comprehensive "
            "off-target safety, patient-strain coverage, delivery safety, or "
            "therapeutic readiness."
        ),
    }
    (output / "colibactin_specificity_pilot_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
