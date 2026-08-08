from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_mechanism_panel import CANDIDATES, canonical_manifest_bytes  # noqa: E402


def test_panel_is_prespecified_and_covers_all_four_mechanisms():
    mechanisms = {candidate.mechanism for candidate in CANDIDATES}
    genes = {candidate.gene for candidate in CANDIDATES}

    assert mechanisms == {
        "colibactin_genotoxicity",
        "fusobacterial_adhesion",
        "b_fragilis_toxin",
        "secondary_bile_acid",
    }
    assert {"clbA", "clbS", "fadA", "fap2", "bft", "baiE"} <= genes
    assert all(candidate.evidence_url.startswith("https://") for candidate in CANDIDATES)


def test_manifest_checksum_is_order_stable():
    frame = pd.DataFrame(
        [
            {
                "mechanism": "m",
                "prespecified_gene": "b",
                "accession": "2",
                "uniref90": "UniRef90_2",
                "organism": "o",
                "protein_name": "p",
                "reviewed": "reviewed",
                "evidence_url": "https://example.org",
                "query_status": "frozen_detected",
                "n_cohorts_detected": 2,
            },
            {
                "mechanism": "m",
                "prespecified_gene": "a",
                "accession": "1",
                "uniref90": "UniRef90_1",
                "organism": "o",
                "protein_name": "p",
                "reviewed": "reviewed",
                "evidence_url": "https://example.org",
                "query_status": "frozen_detected",
                "n_cohorts_detected": 3,
            },
        ]
    )
    first = canonical_manifest_bytes(frame)
    second = canonical_manifest_bytes(frame.iloc[::-1])

    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
