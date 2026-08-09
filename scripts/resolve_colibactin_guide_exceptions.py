"""Resolve the two frozen colibactin guide exceptions from source reads."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import ssl
import urllib.request
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import pandas as pd

from audit_published_colibactin_guides import (
    find_pam_sites,
    read_fasta,
    reverse_complement,
)


PRIMARY_GUIDE = "GAACGCGATAGATCTATAGC"
SECONDARY_GUIDE = "ACGAAAGGTACGCTTAACAC"
KMER_SIZE = 31


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - provider checksum, not cryptography
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_read(url: str, destination: Path, expected_bytes: int, md5: str) -> None:
    """Download one provider file with resumable, validated byte ranges."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.exists()
        and destination.stat().st_size == expected_bytes
        and file_md5(destination) == md5
    ):
        return

    context = ssl.create_default_context(cafile=certifi.where())
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    n_parts = 8 if expected_bytes >= 50_000_000 else 1
    chunk_size = (expected_bytes + n_parts - 1) // n_parts
    ranges = [
        (index, index * chunk_size, min(expected_bytes - 1, (index + 1) * chunk_size - 1))
        for index in range(n_parts)
    ]
    user_agent = {"User-Agent": "crc-colibactin-exception-audit/0.1"}

    def fetch_part(specification: tuple[int, int, int]) -> Path:
        index, start, stop = specification
        part = temporary.with_suffix(temporary.suffix + f".part{index:02d}")
        expected_part_bytes = stop - start + 1
        if part.exists() and part.stat().st_size == expected_part_bytes:
            return part
        part.unlink(missing_ok=True)
        request = urllib.request.Request(
            url, headers={**user_agent, "Range": f"bytes={start}-{stop}"}
        )
        with urllib.request.urlopen(request, timeout=300, context=context) as response:
            if n_parts > 1 and response.status != 206:
                raise RuntimeError(f"server ignored a byte-range request for {url}")
            with part.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if part.stat().st_size != expected_part_bytes:
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
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(f"download size mismatch for {url}")
    if file_md5(temporary) != md5:
        raise RuntimeError(f"provider MD5 mismatch for {url}")
    temporary.replace(destination)


def iter_fastq(path: Path) -> Iterator[tuple[str, str]]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        while True:
            name = handle.readline().rstrip()
            if not name:
                break
            sequence = handle.readline().rstrip().upper()
            plus = handle.readline().rstrip()
            quality = handle.readline().rstrip()
            if not name.startswith("@") or not plus.startswith("+"):
                raise ValueError(f"malformed FASTQ record in {path}")
            if len(sequence) != len(quality):
                raise ValueError(f"sequence/quality length mismatch in {path}")
            yield name[1:].split()[0], sequence


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _variant_candidate_starts(sequence: str, expected: str) -> set[int]:
    """Use five exact 4-mers to cover every site with at most four mismatches."""

    starts: set[int] = set()
    for offset in range(0, 20, 4):
        block = expected[offset : offset + 4]
        position = sequence.find(block)
        while position >= 0:
            starts.add(position - offset)
            position = sequence.find(block, position + 1)
    return starts


def find_pam_variants(sequence: str, guide: str, max_mismatches: int = 4) -> list[dict[str, object]]:
    """Find all PAM-compatible sites within a Hamming-distance threshold."""

    sequence = sequence.upper()
    hits: list[dict[str, object]] = []
    for strand, expected in [("+", guide), ("-", reverse_complement(guide))]:
        for start in _variant_candidate_starts(sequence, expected):
            if start < 0 or start + 20 > len(sequence):
                continue
            observed = sequence[start : start + 20]
            if any(base not in "ACGT" for base in observed):
                continue
            if strand == "+":
                pam = sequence[start + 20 : start + 23]
                if len(pam) != 3 or pam[1:] != "GG":
                    continue
            else:
                if start < 3:
                    continue
                pam_on_read = sequence[start - 3 : start]
                if pam_on_read[:2] != "CC":
                    continue
                pam = reverse_complement(pam_on_read)
            mismatches = hamming(observed, expected)
            if mismatches > max_mismatches:
                continue
            candidate = observed if strand == "+" else reverse_complement(observed)
            hits.append(
                {
                    "strand": strand,
                    "pam": pam,
                    "candidate_spacer_5to3": candidate,
                    "mismatches": mismatches,
                }
            )
    return hits


def canonical(sequence: str) -> str:
    return min(sequence, reverse_complement(sequence))


def high_complexity(kmer: str) -> bool:
    return (
        all(base in "ACGT" for base in kmer)
        and len(set(kmer)) == 4
        and max(kmer.count(base) for base in "ACGT") <= 18
    )


def select_probe_groups(
    records: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Select assembly-unique probes around JML024 guide-bearing loci."""

    primary_loci: list[tuple[str, str, int]] = []
    secondary_loci: list[tuple[str, str, int]] = []
    for record_id, sequence in records:
        for hit in find_pam_sites(sequence, PRIMARY_GUIDE):
            primary_loci.append((record_id, sequence, int(hit["start_1based"]) - 1))
        for hit in find_pam_sites(sequence, SECONDARY_GUIDE):
            secondary_loci.append((record_id, sequence, int(hit["start_1based"]) - 1))
    if len(primary_loci) != 2 or len(secondary_loci) != 1:
        raise ValueError("JML024 assembly no longer has the frozen 2-primary/1-secondary pattern")

    primary_loci.sort(key=lambda item: len(item[1]), reverse=True)
    labeled = [
        ("jml024_long_contig", *primary_loci[0]),
        ("jml024_short_contig", *primary_loci[1]),
        ("jml024_secondary_control", *secondary_loci[0]),
    ]
    candidates: dict[str, list[str]] = {}
    all_candidates: set[str] = set()
    for label, _record_id, sequence, target_start in labeled:
        left = max(0, target_start - 250)
        right = min(len(sequence), target_start + 20 + 250)
        values = [
            canonical(sequence[start : start + KMER_SIZE])
            for start in range(left, max(left, right - KMER_SIZE + 1))
            if high_complexity(sequence[start : start + KMER_SIZE])
        ]
        candidates[label] = list(dict.fromkeys(values))
        all_candidates.update(values)

    assembly_counts: Counter[str] = Counter()
    for _record_id, sequence in records:
        for start in range(len(sequence) - KMER_SIZE + 1):
            value = canonical(sequence[start : start + KMER_SIZE])
            if value in all_candidates:
                assembly_counts[value] += 1

    group_sets = {label: set(values) for label, values in candidates.items()}
    selected: dict[str, list[str]] = {}
    for label, values in candidates.items():
        other = set().union(*(items for key, items in group_sets.items() if key != label))
        eligible = [value for value in values if assembly_counts[value] == 1 and value not in other]
        if len(eligible) < 12:
            raise ValueError(f"too few assembly-unique probes for {label}")
        indexes = [round(i * (len(eligible) - 1) / 11) for i in range(12)]
        selected[label] = [eligible[index] for index in indexes]

    structure = {
        "primary_target_records": [
            {"record_id": item[0], "length": len(item[1]), "target_start_1based": item[2] + 1}
            for item in primary_loci
        ],
        "secondary_target_record": {
            "record_id": secondary_loci[0][0],
            "length": len(secondary_loci[0][1]),
            "target_start_1based": secondary_loci[0][2] + 1,
        },
        "redundant_contig_rule_passed": False,
        "note": (
            "The 302-bp target-bearing contig is neither contained in nor at least "
            "500-bp/99.9%-identical to the 108,917-bp target-bearing contig."
        ),
    }
    return selected, structure


def scan_run(
    paths: list[Path],
    run: str,
    strain: str,
    probes: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str], int]:
    guide_rows: list[dict[str, object]] = []
    variants: Counter[tuple[str, str, int]] = Counter()
    probe_counts: Counter[str] = Counter()
    exact_read_counts = {"primary": 0, "secondary": 0}
    exact_site_counts = {"primary": 0, "secondary": 0}
    n_reads = 0

    probe_lookup: dict[str, list[str]] = {}
    for group, values in (probes or {}).items():
        for value in values:
            probe_lookup.setdefault(value, []).append(group)
    probe_forms = {
        form: value
        for value in probe_lookup
        for form in (value, reverse_complement(value))
    }
    probe_pattern = (
        re.compile(f"(?=({'|'.join(probe_forms)}))") if probe_forms else None
    )

    for path in paths:
        for _name, sequence in iter_fastq(path):
            n_reads += 1
            for role, guide in [("primary", PRIMARY_GUIDE), ("secondary", SECONDARY_GUIDE)]:
                hits = find_pam_variants(sequence, guide)
                exact = [hit for hit in hits if int(hit["mismatches"]) == 0]
                if exact:
                    exact_read_counts[role] += 1
                    exact_site_counts[role] += len(exact)
                if role == "primary":
                    observed_variants = {
                        (
                            str(hit["candidate_spacer_5to3"]),
                            str(hit["pam"]),
                            int(hit["mismatches"]),
                        )
                        for hit in hits
                    }
                    variants.update(observed_variants)
            if probe_pattern is not None:
                observed_probes = {
                    probe_forms[match.group(1)] for match in probe_pattern.finditer(sequence)
                }
                for value in observed_probes:
                    for group in probe_lookup[value]:
                        probe_counts[f"{group}|{value}"] += 1
            if n_reads % 200_000 == 0:
                print(f"  scanned {n_reads:,} reads from {run}", flush=True)

    for role in ["primary", "secondary"]:
        guide_rows.append(
            {
                "run_accession": run,
                "strain": strain,
                "guide_role": role,
                "guide_id": "sgclbB_4387" if role == "primary" else "sgclbC_2313",
                "n_reads_scanned": n_reads,
                "exact_supporting_reads": exact_read_counts[role],
                "exact_pam_sites_in_reads": exact_site_counts[role],
            }
        )
    variant_rows = [
        {
            "run_accession": run,
            "strain": strain,
            "candidate_spacer_5to3": spacer,
            "pam": pam,
            "mismatches": mismatches,
            "supporting_reads": count,
        }
        for (spacer, pam, mismatches), count in sorted(
            variants.items(), key=lambda item: (item[0][2], -item[1], item[0][0])
        )
    ]
    return guide_rows, variant_rows, probe_counts, n_reads


def summarize_probe_support(probes: dict[str, list[str]], counts: Counter[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, values in probes.items():
        for value in values:
            rows.append(
                {
                    "probe_group": group,
                    "probe_31mer": value,
                    "supporting_reads": counts[f"{group}|{value}"],
                }
            )
    return pd.DataFrame(rows)


def interpret(
    read_support: pd.DataFrame, variants: pd.DataFrame, probes: pd.DataFrame
) -> pd.DataFrame:
    support = read_support.set_index(["strain", "guide_role"])
    upec_primary = int(support.loc[("UPEC79", "primary"), "exact_supporting_reads"])
    upec_secondary = int(support.loc[("UPEC79", "secondary"), "exact_supporting_reads"])
    upec_near = variants.loc[
        variants["strain"].eq("UPEC79") & variants["mismatches"].between(1, 4)
    ]
    if upec_primary >= 3:
        upec_status = "resolved_assembly_omission"
        upec_reason = "At least three source reads recover the exact primary site."
    elif upec_secondary >= 10 and not upec_near.empty:
        upec_status = "supported_sequence_difference"
        upec_reason = "The secondary control is recovered and a PAM-compatible primary-site variant is supported."
    elif upec_secondary >= 10:
        upec_status = "source_reads_do_not_support_site"
        upec_reason = "The secondary control is recovered but neither the exact primary site nor a <=4-mismatch variant is observed."
    else:
        upec_status = "unresolved_absence"
        upec_reason = "Source-read recovery is insufficient for a frozen-rule decision."

    medians = probes.groupby("probe_group")["supporting_reads"].median()
    long_depth = float(medians.get("jml024_long_contig", 0.0))
    short_depth = float(medians.get("jml024_short_contig", 0.0))
    control_depth = float(medians.get("jml024_secondary_control", 0.0))
    both_supported = long_depth >= 3 and short_depth >= 3
    combined_ratio = (long_depth + short_depth) / control_depth if control_depth else 0.0
    if both_supported and 1.5 <= combined_ratio <= 2.5:
        jml_status = "supported_distinct_copies"
        jml_reason = (
            "Both divergent target neighborhoods have source-read support and their "
            "combined median probe depth is approximately twice the single-copy control."
        )
    else:
        jml_status = "unresolved_duplicate"
        jml_reason = (
            "The contigs are not redundant by the frozen rule, but source-read depth does "
            "not establish approximately two copies relative to the control."
        )

    return pd.DataFrame(
        [
            {
                "case_id": "jml024_duplicate",
                "strain": "JML024",
                "resolution_status": jml_status,
                "decision_reason": jml_reason,
                "primary_exact_supporting_reads": int(support.loc[("JML024", "primary"), "exact_supporting_reads"]),
                "secondary_exact_supporting_reads": int(support.loc[("JML024", "secondary"), "exact_supporting_reads"]),
                "near_variant_supporting_reads": int(
                    variants.loc[
                        variants["strain"].eq("JML024") & variants["mismatches"].between(1, 4),
                        "supporting_reads",
                    ].sum()
                ),
                "support_metric": f"combined_probe_to_control_ratio={combined_ratio:.3f}",
            },
            {
                "case_id": "upec79_absent",
                "strain": "UPEC79",
                "resolution_status": upec_status,
                "decision_reason": upec_reason,
                "primary_exact_supporting_reads": upec_primary,
                "secondary_exact_supporting_reads": upec_secondary,
                "near_variant_supporting_reads": int(upec_near["supporting_reads"].sum()),
                "support_metric": "frozen read-count thresholds",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-dir", type=Path, default=Path("data/interim/colibactin_exception_reads"))
    parser.add_argument("--assembly-dir", type=Path, default=Path("data/interim/colibactin_human_isolate_panel"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    research = Path("research/intervention_readiness")
    output = Path("results/intervention_readiness")
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(research / "colibactin_exception_read_manifest.csv")

    for row in manifest.itertuples(index=False):
        destination = args.read_dir / f"{row.run_accession}_{row.mate}.fastq.gz"
        if not args.skip_download:
            download_read(row.read_url, destination, int(row.bytes), row.md5)
        if not destination.exists() or destination.stat().st_size != int(row.bytes):
            raise FileNotFoundError(f"missing validated read file: {destination}")
        if file_md5(destination) != row.md5:
            raise RuntimeError(f"MD5 mismatch: {destination}")
        print(f"validated {destination.name}", flush=True)

    jml_records = read_fasta(args.assembly_dir / "BFMV01000000.fasta")
    probes, structure = select_probe_groups(jml_records)
    all_support: list[dict[str, object]] = []
    all_variants: list[dict[str, object]] = []
    probe_table = pd.DataFrame()
    for run, strain in [("DRR102722", "JML024"), ("DRR103319", "UPEC79")]:
        paths = [args.read_dir / f"{run}_{mate}.fastq.gz" for mate in [1, 2]]
        run_probes = probes if strain == "JML024" else None
        support, variants, counts, n_reads = scan_run(paths, run, strain, run_probes)
        all_support.extend(support)
        all_variants.extend(variants)
        if strain == "JML024":
            probe_table = summarize_probe_support(probes, counts)
            probe_table.insert(0, "strain", strain)
            probe_table.insert(0, "run_accession", run)
        print(f"scanned {n_reads:,} reads from {run}", flush=True)

    read_support = pd.DataFrame(all_support)
    variants = pd.DataFrame(all_variants)
    if variants.empty:
        variants = pd.DataFrame(
            columns=["run_accession", "strain", "candidate_spacer_5to3", "pam", "mismatches", "supporting_reads"]
        )
    resolution = interpret(read_support, variants, probe_table)
    read_support.to_csv(output / "colibactin_exception_read_support.csv", index=False)
    variants.to_csv(output / "colibactin_exception_variants.csv", index=False)
    probe_table.to_csv(output / "jml024_contig_probe_support.csv", index=False)
    resolution.to_csv(output / "colibactin_exception_resolution.csv", index=False)
    audit = {
        "status": "complete_source_read_reconciliation",
        "protocol": "research/intervention_readiness/colibactin_exception_resolution_protocol.md",
        "read_manifest": "research/intervention_readiness/colibactin_exception_read_manifest.csv",
        "provider_checksums_validated": True,
        "jml024_contig_structure": structure,
        "resolution": resolution.to_dict(orient="records"),
        "independent_assembly_search": {
            "provider": "ENA assembly BioSample mapping",
            "jml024": "GCA_005385205.1 / ASM538520v1 / BFMV01 only",
            "upec79": "GCA_005397665.1 / ASM539766v1 / BGJT01 only",
            "later_or_independent_assembly_found": False,
        },
        "claim_boundary": (
            "This resolves source-read representation only. It does not establish guide "
            "expression, knockdown, colibactin production, delivery, specificity, or safety."
        ),
    }
    (output / "colibactin_exception_resolution_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(resolution.to_string(index=False))


if __name__ == "__main__":
    main()
