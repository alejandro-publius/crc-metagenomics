from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from species_aware_correction import (  # noqa: E402
    correct_species,
    fit_study_offsets,
    parent_species,
    propagate_to_pathways,
)


def test_parent_species_parser_handles_humann_taxon():
    feature = "PWY-1|g__Bacteroides.s__Bacteroides_fragilis"
    assert parent_species(feature) == "s__Bacteroides_fragilis"
    assert parent_species("PWY-1") is None


def test_source_offsets_are_unchanged_by_held_out_values():
    training = pd.DataFrame({"x|s__A": [-3.0, -2.0, -5.0, -4.0]})
    studies = pd.Series(["one", "one", "two", "two"])

    first = fit_study_offsets(training, studies)
    held_out_a = pd.DataFrame({"x|s__A": [100.0, 200.0]})
    held_out_b = pd.DataFrame({"x|s__A": [-100.0, -200.0]})
    second = fit_study_offsets(training, studies)

    pd.testing.assert_frame_equal(first, second)
    assert not held_out_a.equals(held_out_b)


def test_species_factor_is_propagated_to_matching_pathway():
    species = pd.DataFrame({"k|s__A": [-2.0, -2.0], "k|s__B": [-3.0, -3.0]})
    studies = pd.Series(["batch", "batch"])
    offsets = pd.DataFrame({"k|s__A": [1.0], "k|s__B": [0.0]}, index=["batch"])
    pathways = pd.DataFrame(
        {
            "P1|g__A.s__A": [10.0, 20.0],
            "P2|g__Unknown.s__Unknown": [3.0, 4.0],
        }
    )

    corrected_species = correct_species(species, studies, offsets)
    corrected_pathways, coverage = propagate_to_pathways(
        pathways, studies, offsets, species.columns.tolist()
    )

    assert corrected_species["k|s__A"].tolist() == [-3.0, -3.0]
    assert np.allclose(corrected_pathways["P1|g__A.s__A"], [1.0, 2.0])
    assert corrected_pathways["P2|g__Unknown.s__Unknown"].tolist() == [3.0, 4.0]
    assert coverage == 0.5
