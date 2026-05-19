"""Tests for the parameterised scripts.verify_results manifest.

Two contracts pinned here:

  1. Default manifest still passes -- the YAML at
     ``results/verify_manifest.yaml`` reflects the current canonical
     state of the saved CSVs and metadata, so ``main()`` returns 0.

  2. A manifest with deliberately wrong numbers fails clearly --
     ``main()`` returns non-zero AND each wrong expectation surfaces
     as a recorded failure (not just a silent pass).

Run with:
    pytest tests/test_verifier_with_manifest.py -v
"""
from __future__ import annotations

import copy
import importlib
import os
import sys

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "results", "verify_manifest.yaml")


@pytest.fixture
def verify_module(monkeypatch):
    """Import verify_results fresh and run from the repo root.

    The script reads CSVs via relative paths like ``results/...``, so we
    chdir into the repo root for the duration of each test.
    """
    monkeypatch.chdir(REPO_ROOT)
    if "verify_results" in sys.modules:
        del sys.modules["verify_results"]
    return importlib.import_module("verify_results")


def test_default_manifest_passes(verify_module, capsys):
    """The shipped default manifest must still match the current CSVs."""
    rc = verify_module.main(["--manifest", DEFAULT_MANIFEST])
    out = capsys.readouterr().out
    assert rc == 0, (
        f"verify_results.main() returned {rc} with the default manifest "
        f"-- the canonical numbers and the saved CSVs have drifted.\n\n"
        f"Output:\n{out}"
    )
    assert "All checks passed." in out


def test_wrong_manifest_fails_clearly(verify_module, tmp_path, capsys):
    """A manifest with deliberately wrong headline numbers must produce
    non-zero return AND list each wrong expectation as a failure."""
    with open(DEFAULT_MANIFEST) as f:
        manifest = yaml.safe_load(f)

    bad = copy.deepcopy(manifest)
    # Push every headline AUC well outside its tolerance.
    bad["lodo"]["baseline_species_rf_mean_auc"]["value"] = 0.123
    bad["lodo"]["joint_rf_mean_auc"]["value"] = 0.456
    bad["lodo"]["joint_xgb_mean_auc"]["value"] = 0.789
    # Push sample counts somewhere clearly wrong.
    bad["dataset"]["n_crc"] = 1
    bad["dataset"]["n_control"] = 2

    bad_path = tmp_path / "bad_manifest.yaml"
    with open(bad_path, "w") as f:
        yaml.safe_dump(bad, f)

    rc = verify_module.main(["--manifest", str(bad_path)])
    out = capsys.readouterr().out

    assert rc != 0, (
        "verify_results.main() returned 0 with a clearly-wrong manifest; "
        "the verifier is not actually consulting the YAML expectations.\n"
        f"Output:\n{out}"
    )
    # The failure messages should name each wrong check explicitly.
    assert "Baseline species RF per-cohort mean AUC" in out
    assert "Joint RF per-cohort mean AUC" in out
    assert "Joint XGB per-cohort mean AUC" in out
    # The new sample-count expectations should also surface.
    assert "CRC sample count = 1" in out
    assert "Control sample count = 2" in out
    # And the closing summary must explicitly mark failures.
    assert "failure(s)" in out


def test_manifest_required_keys_present():
    """Defensive: the shipped manifest contains every key the verifier
    indexes into. A typo here would otherwise show up as a confusing
    KeyError at run time."""
    with open(DEFAULT_MANIFEST) as f:
        m = yaml.safe_load(f)

    # Top-level sections the verifier reads.
    for section in (
        "dataset", "lodo", "joint_features", "prediction_files",
        "delong", "bootstrap_species_rf_pooled", "adenoma",
        "seed_sensitivity", "confounder", "sensitivity_grid",
    ):
        assert section in m, f"manifest missing section {section!r}"

    # Spot-check a few nested keys the verifier dereferences.
    assert "n_samples_binary" in m["dataset"]
    assert "n_crc" in m["dataset"]
    assert "value" in m["lodo"]["baseline_species_rf_mean_auc"]
    assert "tol" in m["lodo"]["baseline_species_rf_mean_auc"]
    assert "species_count" in m["joint_features"]
    assert isinstance(m["prediction_files"], list)
    assert all("path" in e and "expected_n" in e
               for e in m["prediction_files"])
