from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from annotate_intervention_candidates import (  # noqa: E402
    annotation_category,
    parse_active_uniprot,
    parse_uniparc,
)


def test_parse_active_uniprot_keeps_name_organism_and_sequence():
    payload = {
        "entryType": "UniProtKB unreviewed (TrEMBL)",
        "primaryAccession": "A1",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Example enzyme"}}
        },
        "organism": {"scientificName": "Example bacterium", "taxonId": 123},
        "genes": [{"geneName": {"value": "exampleA"}}],
        "sequence": {"length": 300, "md5": "ABC"},
    }
    parsed = parse_active_uniprot(payload, "A1")

    assert parsed["protein_names"] == "Example enzyme"
    assert parsed["organisms"] == "Example bacterium"
    assert parsed["gene_names"] == "exampleA"
    assert parsed["sequence_length"] == 300


def test_parse_uniparc_prefers_active_cross_references():
    payload = {
        "uniParcId": "UPI1",
        "uniParcCrossReferences": [
            {
                "database": "UniProtKB/TrEMBL",
                "id": "OLD",
                "active": False,
                "proteinName": "Old name",
            },
            {
                "database": "RefSeq",
                "id": "WP_1",
                "active": True,
                "proteinName": "Current enzyme",
                "geneName": "currentA",
                "organism": {"scientificName": "Current bacterium", "taxonId": 456},
            },
        ],
        "sequence": {"length": 200, "md5": "DEF"},
    }
    parsed = parse_uniparc(payload, "A0")

    assert parsed["current_accessions"] == "WP_1"
    assert parsed["protein_names"] == "Current enzyme"
    assert parsed["organisms"] == "Current bacterium"


def test_generic_names_are_not_promoted_to_annotations():
    assert annotation_category("Hypothetical protein") == "uncharacterized"
    assert annotation_category("Uncharacterized protein") == "uncharacterized"
    assert annotation_category("DNA-binding protein") == "annotated"
