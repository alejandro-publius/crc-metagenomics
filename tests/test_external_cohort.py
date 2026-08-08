import pandas as pd
import numpy as np

from scripts.prepare_external_cohort import build_manifest, label_from_alias
from scripts.score_external_species import harmonize_gmrepo, harmonize_metaphlan


def test_external_alias_labels_are_frozen():
    assert label_from_alias("M_HO_01") == (0, "control", "older")
    assert label_from_alias("M_HY_01") == (0, "control", "younger")
    assert label_from_alias("M_O_01") == (1, "CRC", "older")
    assert label_from_alias("M_Y_01") == (1, "CRC", "younger")


def test_metaphlan_harmonization_reindexes_and_logs():
    table = pd.DataFrame({
        "clade_name": ["k__Bacteria|s__A", "k__Bacteria|s__B|t__strain"],
        "run1": [25.0, 75.0],
    })
    out = harmonize_metaphlan(table, ["k__Bacteria|s__A", "k__Bacteria|s__C"])
    assert list(out.columns) == ["k__Bacteria|s__A", "k__Bacteria|s__C"]
    assert np.isclose(out.loc["run1", "k__Bacteria|s__A"], 0.0, atol=1e-6)
    assert out.loc["run1", "k__Bacteria|s__C"] == -6.0


def test_gmrepo_harmonization_uses_terminal_species_names_only():
    table = pd.DataFrame({
        "run_accession": ["run1", "run1", "run2"],
        "scientific_name": ["[Ruminococcus] torques", "Other bacterium", "Other bacterium"],
        "relative_abundance": [25.0, 75.0, 100.0],
    })
    retained = [
        "k__Bacteria|g__Ruminococcus|s__Ruminococcus_torques",
        "k__Bacteria|g__Missing|s__Missing_species",
    ]
    out = harmonize_gmrepo(table, retained)
    assert list(out.index) == ["run1", "run2"]
    assert np.isclose(out.loc["run1", retained[0]], 0.0, atol=1e-6)
    assert out.loc["run1", retained[1]] == -6.0
    assert (out.loc["run2"] == -6.0).all()
