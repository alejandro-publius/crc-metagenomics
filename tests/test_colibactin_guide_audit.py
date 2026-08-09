from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_published_colibactin_guides import (  # noqa: E402
    audit_panel,
    find_pam_sites,
    reverse_complement,
    validate_guides,
)


def test_finds_pam_site_on_each_strand():
    spacer = "GAACGCGATAGATCTATAGC"
    plus = "AAAA" + spacer + "AGG" + "AAAA"
    minus = "AAAA" + "CCA" + reverse_complement(spacer) + "AAAA"
    assert find_pam_sites(plus, spacer) == [
        {"start_1based": 5, "strand": "+", "pam": "AGG"}
    ]
    assert find_pam_sites(minus, spacer) == [
        {"start_1based": 8, "strand": "-", "pam": "TGG"}
    ]


def test_rejects_reverse_complement_transcription_error():
    guides = pd.DataFrame(
        [
            {
                "guide_id": "g",
                "target_id": "colibactin",
                "target_gene": "clbB",
                "spacer_5to3": "GAACGCGATAGATCTATAGC",
                "reverse_complement_5to3": "AAAAAAAAAAAAAAAAAAAA",
                "benchmark_role": "primary",
            }
        ]
    )
    with pytest.raises(ValueError, match="reverse-complement mismatch"):
        validate_guides(guides)


def test_panel_summary_requires_80_percent_coverage(tmp_path: Path):
    guides = pd.DataFrame(
        [
            {
                "guide_id": "g",
                "target_id": "colibactin",
                "target_gene": "clbB",
                "spacer_5to3": "GAACGCGATAGATCTATAGC",
                "reverse_complement_5to3": "GCTATAGATCTATCGCGTTC",
                "benchmark_role": "primary",
            }
        ]
    )
    genomes = pd.DataFrame(
        [{"accession": f"A{i}", "strain": f"s{i}"} for i in range(5)]
    )
    target = "GAACGCGATAGATCTATAGCAGG"
    for i in range(5):
        sequence = target if i < 4 else "A" * len(target)
        (tmp_path / f"A{i}.fasta").write_text(f">A{i}\n{sequence}\n")
    _, summary = audit_panel(guides, genomes, tmp_path)
    assert summary.loc[0, "coverage_fraction"] == 0.8
    assert summary.loc[0, "pilot_conservation_gate"] == "pass"
