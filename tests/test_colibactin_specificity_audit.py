from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_colibactin_protected_references import find_flagged_sites  # noqa: E402
from audit_published_colibactin_guides import reverse_complement  # noqa: E402


GUIDE = "GAACGCGATAGATCTATAGC"


def mutate(sequence: str, positions: list[int]) -> str:
    chars = list(sequence)
    for position in positions:
        chars[position] = "A" if chars[position] != "A" else "C"
    return "".join(chars)


def test_exact_and_two_mismatch_sites_with_pam_are_flagged():
    sequence = ("TTT" + GUIDE + "AGG" + mutate(GUIDE, [0, 5]) + "TGG").encode()
    hits = find_flagged_sites(sequence, GUIDE)
    assert [hit["mismatches"] for hit in hits] == [0, 2]


def test_seed_exact_site_with_four_total_mismatches_is_flagged():
    observed = mutate(GUIDE, [0, 3, 6, 9])
    hits = find_flagged_sites((observed + "CGG").encode(), GUIDE)
    assert len(hits) == 1
    assert hits[0]["mismatches"] == 4
    assert hits[0]["pam_proximal_seed_exact"]


def test_three_mismatch_site_without_exact_seed_is_not_flagged():
    observed = mutate(GUIDE, [0, 5, 12])
    assert find_flagged_sites((observed + "AGG").encode(), GUIDE) == []


def test_reverse_strand_site_is_flagged_and_non_pam_site_is_not():
    reverse_site = "CCA" + reverse_complement(GUIDE)
    no_pam_site = "TTT" + GUIDE + "AAA"
    hits = find_flagged_sites((reverse_site + no_pam_site).encode(), GUIDE)
    assert len(hits) == 1
    assert hits[0]["strand"] == "-"
    assert hits[0]["pam"] == "TGG"
