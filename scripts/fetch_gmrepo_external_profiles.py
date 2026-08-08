#!/usr/bin/env python3
"""Fetch the frozen PRJNA763023 MetaPhlAn profiles from GMrepo v3.

GMrepo v3 processed WGS runs with MetaPhlAn 4.1.0 and exposes species-level
relative abundances through its public API. This script verifies that the
project contains exactly the run accessions in the already-frozen manifest,
downloads every profile, and writes a compact, checksummed research artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_cohort"
API_ROOT = "https://gmrepo.humangut.info/api"
PROJECT = "PRJNA763023"
GMREPO_METHOD_DOI = "10.1093/nar/gkaf1190"


def post_json(path: str, payload: dict, retries: int = 5) -> dict:
    """POST JSON to the public GMrepo API with bounded retries."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{API_ROOT}/{path.strip('/')}/",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "crc-metagenomics-reproducibility/1.0",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("unreachable")


def fetch_run(run_accession: str) -> dict:
    """Fetch and minimally validate one run-level profile."""
    payload = post_json("getRunDetailsByRunID", {"run_id": run_accession})
    run = payload.get("run", {})
    if run.get("run_id") != run_accession:
        raise ValueError(f"GMrepo returned the wrong run for {run_accession}")
    if int(run.get("QCStatus", 0)) != 1:
        raise ValueError(f"GMrepo QC failed for {run_accession}: {run.get('QCMessage')}")
    species = payload.get("species", [])
    if not species:
        raise ValueError(f"GMrepo returned no species for {run_accession}")
    total = sum(float(row["relative_abundance"]) for row in species)
    if not 99.9 <= total <= 100.1:
        raise ValueError(f"Species abundances sum to {total:.6f} for {run_accession}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest = pd.read_csv(OUT / "manifest.csv")
    expected = set(manifest["run_accession"].astype(str))
    project = post_json("getProjectDetailsByID", {"ncbi_project_id": PROJECT})
    available = {
        run
        for sample in project.get("allsamples", [])
        for run in str(sample.get("run_ids", "")).split(",")
        if run
    }
    if expected != available:
        raise ValueError(
            "GMrepo project runs do not match the frozen manifest: "
            f"missing={sorted(expected - available)}, extra={sorted(available - expected)}"
        )

    profiles: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_run, run): run for run in sorted(expected)}
        for completed, future in enumerate(as_completed(futures), start=1):
            run = futures[future]
            profiles[run] = future.result()
            if completed % 25 == 0 or completed == len(futures):
                print(f"Fetched {completed}/{len(futures)} profiles")

    abundance_rows = []
    metadata_rows = []
    for run_accession in sorted(profiles):
        payload = profiles[run_accession]
        run = payload["run"]
        metadata_rows.append(
            {
                "run_accession": run_accession,
                "loaded_uid": run.get("loaded_uid"),
                "instrument_model": run.get("instrument_model"),
                "nr_reads_sequenced": run.get("nr_reads_sequenced"),
                "country": run.get("country"),
                "qc_status": run.get("QCStatus"),
                "qc_message": run.get("QCMessage"),
                "n_species": len(payload["species"]),
            }
        )
        for species in payload["species"]:
            abundance_rows.append(
                {
                    "run_accession": run_accession,
                    "ncbi_taxon_id": species.get("ncbi_taxon_id"),
                    "scientific_name": species["scientific_name"],
                    "relative_abundance": species["relative_abundance"],
                }
            )

    abundances = pd.DataFrame(abundance_rows).sort_values(
        ["run_accession", "relative_abundance"], ascending=[True, False]
    )
    metadata = pd.DataFrame(metadata_rows).sort_values("run_accession")
    abundance_path = OUT / "gmrepo_species_long.csv.gz"
    metadata_path = OUT / "gmrepo_profile_metadata.csv"
    abundances.to_csv(abundance_path, index=False, compression="gzip")
    metadata.to_csv(metadata_path, index=False)

    canonical = json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "status": "all_profiles_fetched",
        "project": PROJECT,
        "run_count": len(profiles),
        "profile_source": "GMrepo v3 public API",
        "profile_method": "MetaPhlAn 4.1.0, default settings; species abundances normalized to 100%",
        "profile_method_doi": GMREPO_METHOD_DOI,
        "api_root": API_ROOT,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_api_payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "manifest_run_set_matched_exactly": True,
        "all_qc_status_good": bool((metadata.qc_status == 1).all()),
    }
    (OUT / "gmrepo_profile_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
