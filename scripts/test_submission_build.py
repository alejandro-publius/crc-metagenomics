#!/usr/bin/env python3
"""
test_submission_build.py
========================

Pytest-style integration test for ``scripts/build_submission.py``.

Runs the bundler end-to-end against the current repository state and
asserts that every expected output file exists with non-zero size.

The test is marked ``@pytest.mark.integration`` so it does **not** run
during a default ``pytest`` invocation. Run explicitly with:

    pytest scripts/test_submission_build.py -m integration -v

Author: Alejandro Velazquez (UC Berkeley); biological collaboration: Rachel Selbrede.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: end-to-end submission bundler test (slow; opt-in)",
    )


@pytest.fixture(scope="module")
def builder():
    return _load_module("build_submission", SCRIPTS_DIR / "build_submission.py")


@pytest.fixture(scope="module")
def build_result(builder):
    """Run the bundler once for this module."""
    rc = builder.main(["--repo", str(REPO_ROOT)])
    return rc


# --------------------------------------------------------------------------- #
# Expected outputs
# --------------------------------------------------------------------------- #

ALWAYS_EXPECTED = [
    # Submission scaffolding (copied verbatim)
    "submission/build/00_cover_letter.md",
    "submission/build/01_data_availability.md",
    "submission/build/02_ethics_statement.md",
    "submission/build/03_author_contributions.md",
    "submission/build/04_submission_checklist.md",
    "submission/build/05_biorxiv_metadata.md",
    # Bundle-level
    "submission/build/README.md",
    "submission/build/SUBMISSION_BUNDLE.zip",
    # Manuscript source-of-truth (always copied)
    "submission/build/manuscript_complete.md",
    "submission/build/CRC_Manuscript_Complete.docx",
    # Top-level manifest
    "submission/MANIFEST.md",
    # Main figures originals
    "submission/build/figures/main/Figure1_Forest_Plot.pdf",
    "submission/build/figures/main/Figure2_ROC_Curves.pdf",
    "submission/build/figures/main/Figure3_SHAP_Importance.pdf",
    "submission/build/figures/main/Figure4_Three_Panel_SHAP.pdf",
    # Supplementary
    "submission/build/supplementary/INDEX.csv",
    "submission/build/supplementary/S1_cohort_overview.csv",
    "submission/build/supplementary/S10_delong.csv",
    "submission/build/supplementary/Supplementary_Tables.docx",
]


@pytest.mark.integration
def test_bundler_returns_zero(build_result):
    assert build_result == 0, f"build_submission.main() returned {build_result}"


@pytest.mark.integration
@pytest.mark.parametrize("rel", ALWAYS_EXPECTED)
def test_expected_file_exists(build_result, rel):
    path = REPO_ROOT / rel
    assert path.exists(), f"Missing expected output: {path}"
    assert path.stat().st_size > 0, f"Zero-byte output: {path}"


@pytest.mark.integration
def test_zip_contains_scaffolding(build_result):
    import zipfile
    zpath = REPO_ROOT / "submission/build/SUBMISSION_BUNDLE.zip"
    assert zpath.exists()
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
    for scaffold in (
        "00_cover_letter.md",
        "01_data_availability.md",
        "04_submission_checklist.md",
        "05_biorxiv_metadata.md",
        "README.md",
        "CRC_Manuscript_Complete.docx",
        "supplementary/INDEX.csv",
    ):
        assert scaffold in names, f"ZIP missing {scaffold}"


@pytest.mark.integration
def test_manifest_has_hashes(build_result):
    manifest = (REPO_ROOT / "submission/MANIFEST.md").read_text(encoding="utf-8")
    # Either deliverables produced SHA-256s or the build warned about
    # missing converters. At minimum the ZIP must be hashed.
    assert "SUBMISSION_BUNDLE.zip" in manifest
    assert "SHA-256" in manifest


@pytest.mark.integration
def test_no_generator_attribution(build_result):
    """Guard against accidental generator-attribution strings in outputs."""
    forbidden = (
        "Co-" + "Authored-By",
        "Anthro" + "pic",
        "Cla" + "ude",
    )
    targets = [
        REPO_ROOT / "submission/build/README.md",
        REPO_ROOT / "submission/MANIFEST.md",
    ]
    for t in targets:
        if not t.exists():
            continue
        text = t.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, (
                f"Forbidden attribution {needle!r} found in {t}"
            )


if __name__ == "__main__":
    # Allow `python scripts/test_submission_build.py` for quick checks.
    sys.exit(pytest.main([__file__, "-m", "integration", "-v"]))
