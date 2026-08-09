from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_protected_reference_selection import summarize_selection  # noqa: E402


def test_selection_audit_maps_legacy_profile_name_and_ranks_prevalence():
    species = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "k__Bacteria|s__Legacy_name": [-5.0, -6.0, -4.0],
            "k__Bacteria|s__Other": [-5.0, -5.0, -6.0],
        }
    )
    metadata = pd.DataFrame({"sample_id": ["a", "b", "c"], "label": [0, 0, 1]})
    references = pd.DataFrame(
        [
            {
                "reference_id": "legacy",
                "reference_class": "gut_bacterial_reference",
                "profile_species_name": "Legacy_name",
            },
            {
                "reference_id": "human",
                "reference_class": "human_reference",
                "profile_species_name": "",
            },
        ]
    )
    output = summarize_selection(species, metadata, references).set_index(
        "reference_id"
    )
    assert output.loc["legacy", "control_prevalence"] == 0.5
    assert output.loc["legacy", "control_prevalence_rank"] == 2
    assert output.loc["human", "selection_status"] == "human_reference_not_applicable"
