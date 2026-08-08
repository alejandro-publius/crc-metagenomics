#!/usr/bin/env python3
"""Build a checksummed, deterministic scientific release candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release"
VERSION = "2.0.0-rc1"

FILES = [
    ".zenodo.json", "CITATION.cff", "CHANGELOG.md", "LICENSE", "README.md",
    "REPRODUCIBILITY.md", "REPRODUCING.md", "environment.yml", "pyproject.toml",
    "requirements.lock",
    "manuscript/generalization_risk/Generalization_Risk_Manuscript.docx",
    "manuscript/generalization_risk/Generalization_Risk_Manuscript.pdf",
    "manuscript/generalization_risk/manuscript.md",
    "manuscript/generalization_risk/coauthor_approval.md",
    "manuscript/generalization_risk/03_external_profile_adapter.md",
    "manuscript/generalization_risk/figures/PortabilityLandscape.png",
    "manuscript/generalization_risk/figures/PortabilityLandscape.pdf",
    "manuscript/generalization_risk/figures/GeneralizationRisk.png",
    "manuscript/generalization_risk/figures/GeneralizationRisk.pdf",
    "submission/08_msystems_cover_letter.md",
    "results/portability_summary.csv",
    "results/external_cohort/manifest.csv",
    "results/external_cohort/audit.json",
    "results/external_cohort/gmrepo_species_long.csv.gz",
    "results/external_cohort/gmrepo_profile_metadata.csv",
    "results/external_cohort/gmrepo_profile_provenance.json",
    "results/external_cohort/predictions.csv",
    "results/external_cohort/metrics.json",
    "results/external_cohort/uncertainty_metrics.csv",
    "results/external_cohort/age_comparison.json",
    "results/external_cohort/profile_coverage.csv",
    "results/external_cohort/profile_coverage_summary.json",
    "results/external_cohort/bootstrap_replicates.csv.gz",
    "results/generalization_risk/observations.csv",
    "results/generalization_risk/outer_cohort_predictions.csv",
    "results/generalization_risk/metrics.csv",
    "results/generalization_risk/external_risk_estimate.json",
    "results/mechanism_panel/freeze.json",
    "results/mechanism_panel/frozen_manifest.csv",
    "results/mechanism_panel/lodo_results.csv",
    "results/species_aware_correction/lodo_results.csv",
    "results/gene_family_lodo_results.csv",
    "scripts/fetch_gmrepo_external_profiles.py",
    "scripts/score_external_species.py",
    "scripts/external_uncertainty.py",
    "scripts/external_generalization_risk.py",
    "scripts/figure_portability_landscape.py",
    "scripts/figure_generalization_risk.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")
    records = [
        {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": digest(ROOT / name)}
        for name in FILES
    ]
    manifest = {
        "version": VERSION,
        "status": "release_candidate_pending_coauthor_approval_and_doi",
        "files": records,
    }
    manifest_path = OUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = OUT / f"crc-metagenomics-{VERSION}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, (ROOT / name).read_bytes(), compresslevel=9)
        bundle.write(manifest_path, "MANIFEST.json")
    (OUT / "ARCHIVE_SHA256.txt").write_text(
        f"{digest(archive)}  {archive.name}\n"
    )


if __name__ == "__main__":
    main()
