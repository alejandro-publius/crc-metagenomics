from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from resolve_colibactin_guide_exceptions import (  # noqa: E402
    PRIMARY_GUIDE,
    find_pam_variants,
    interpret,
    iter_fastq,
)


def mutate(sequence: str, positions: list[int]) -> str:
    values = list(sequence)
    for position in positions:
        values[position] = "A" if values[position] != "A" else "C"
    return "".join(values)


def test_near_variant_search_keeps_all_sites_up_to_four_mismatches():
    candidate = mutate(PRIMARY_GUIDE, [0, 4, 8, 12])
    hits = find_pam_variants("TT" + candidate + "AGG" + "TT", PRIMARY_GUIDE)
    assert len(hits) == 1
    assert hits[0]["candidate_spacer_5to3"] == candidate
    assert hits[0]["mismatches"] == 4


def test_fastq_reader_validates_and_streams_gzip(tmp_path: Path):
    path = tmp_path / "reads.fastq.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write("@r1\nACGT\n+\nIIII\n")
    assert list(iter_fastq(path)) == [("r1", "ACGT")]


def test_frozen_upec_rules_do_not_turn_missing_reads_into_coverage():
    support = pd.DataFrame(
        [
            {"strain": strain, "guide_role": role, "exact_supporting_reads": count}
            for strain, role, count in [
                ("JML024", "primary", 20),
                ("JML024", "secondary", 10),
                ("UPEC79", "primary", 0),
                ("UPEC79", "secondary", 15),
            ]
        ]
    )
    variants = pd.DataFrame(
        columns=["strain", "mismatches", "supporting_reads"]
    )
    probes = pd.DataFrame(
        [
            {"probe_group": group, "supporting_reads": value}
            for group, value in [
                ("jml024_long_contig", 10),
                ("jml024_short_contig", 10),
                ("jml024_secondary_control", 10),
            ]
        ]
    )
    result = interpret(support, variants, probes).set_index("strain")
    assert result.loc["UPEC79", "resolution_status"] == "source_reads_do_not_support_site"
    assert result.loc["JML024", "resolution_status"] == "supported_distinct_copies"
