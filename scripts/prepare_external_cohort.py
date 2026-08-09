#!/usr/bin/env python3
"""Freeze the publicly released PRJNA763023 shotgun validation subset."""

from __future__ import annotations

import io
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import certifi


PROJECT = "PRJNA763023"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_cohort"
API = "https://www.ebi.ac.uk/ena/portal/api/search"


def ena_search(result: str, fields: list[str]) -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "result": result,
        "query": f'study_accession="{PROJECT}"',
        "fields": ",".join(fields),
        "format": "tsv",
        "limit": "0",
    })
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(f"{API}?{params}", timeout=120,
                                context=context) as response:
        return pd.read_csv(io.BytesIO(response.read()), sep="\t")


def label_from_alias(alias: str) -> tuple[int, str, str]:
    mappings = {
        "M_HO_": (0, "control", "older"),
        "M_HY_": (0, "control", "younger"),
        "M_O_": (1, "CRC", "older"),
        "M_Y_": (1, "CRC", "younger"),
    }
    matches = [value for prefix, value in mappings.items() if str(alias).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Unrecognized or ambiguous metagenome sample alias: {alias}")
    return matches[0]


def build_manifest(runs: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    merged = runs.merge(samples, on="sample_accession", how="left", validate="many_to_one")
    merged = merged[merged.library_strategy.eq("WGS")].copy()
    labels = merged.sample_alias.map(label_from_alias)
    merged[["label", "study_condition", "age_group"]] = pd.DataFrame(
        labels.tolist(), index=merged.index
    )
    columns = [
        "run_accession", "sample_accession", "sample_alias", "label",
        "study_condition", "age_group", "library_strategy", "library_source",
        "instrument_platform", "instrument_model", "library_layout",
        "read_count", "base_count", "fastq_ftp",
    ]
    return merged[columns].sort_values("run_accession").reset_index(drop=True)


def main() -> None:
    run_fields = [
        "run_accession", "sample_accession", "library_strategy", "library_source",
        "instrument_platform", "instrument_model", "library_layout", "read_count",
        "base_count", "fastq_ftp",
    ]
    sample_fields = ["sample_accession", "sample_alias"]
    manifest = build_manifest(
        ena_search("read_run", run_fields),
        ena_search("sample", sample_fields),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "manifest.csv", index=False)
    audit = {
        "snapshot_date": "2026-08-08",
        "project": PROJECT,
        "publication": "https://doi.org/10.1038/s41467-021-27112-y",
        "registry": "ENA Portal API",
        "registry_query": f'study_accession="{PROJECT}" AND library_strategy="WGS"',
        "n_public_wgs_runs": len(manifest),
        "n_crc": int(manifest.label.sum()),
        "n_control": int((manifest.label == 0).sum()),
        "label_rule": "M_O_/M_Y_=CRC; M_HO_/M_HY_=control, as defined by the publication",
        "status": "manifest_frozen_profiles_pending",
        "important_correction": (
            "The earlier scout's claim of a 110-sample Wu cohort was incorrect; "
            "the accession is a different published study with 200 public WGS runs."
        ),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
