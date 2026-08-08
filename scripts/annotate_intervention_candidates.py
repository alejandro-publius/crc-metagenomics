"""Resolve internally nominated legacy UniRef90 IDs through UniProt/UniParc.

curatedMetagenomicData gene-family tables use a historical UniRef release, so
some representative UniProt accessions are now inactive.  UniParc preserves
their sequences and cross-references.  This script records that provenance and
does not promote an association to a biological mechanism automatically.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


UNIPROT_API = "https://rest.uniprot.org"


def _protein_name_from_active(payload: dict[str, Any]) -> str:
    description = payload.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    value = recommended.get("fullName", {}).get("value")
    if value:
        return value
    for submitted in description.get("submissionNames", []):
        value = submitted.get("fullName", {}).get("value")
        if value:
            return value
    return "Uncharacterized protein"


def parse_active_uniprot(payload: dict[str, Any], source_id: str) -> dict[str, object]:
    genes = sorted(
        {
            gene["geneName"]["value"]
            for gene in payload.get("genes", [])
            if gene.get("geneName", {}).get("value")
        }
    )
    organism = payload.get("organism", {}).get("scientificName", "")
    protein_name = _protein_name_from_active(payload)
    sequence = payload.get("sequence", {})
    return {
        "source_representative_id": source_id,
        "resolution_source": "UniProtKB_active",
        "uniparc_id": "",
        "current_accessions": payload.get("primaryAccession", source_id),
        "organisms": organism,
        "taxon_ids": payload.get("organism", {}).get("taxonId", ""),
        "gene_names": ";".join(genes),
        "protein_names": protein_name,
        "sequence_length": sequence.get("length", ""),
        "sequence_md5": sequence.get("md5", ""),
    }


def parse_uniparc(payload: dict[str, Any], source_id: str) -> dict[str, object]:
    references = payload.get("uniParcCrossReferences", [])
    active = [reference for reference in references if reference.get("active")]
    preferred = active if active else references

    def values(key: str) -> list[str]:
        return sorted({str(reference[key]) for reference in preferred if reference.get(key)})

    organisms = sorted(
        {
            reference.get("organism", {}).get("scientificName", "")
            for reference in preferred
            if reference.get("organism", {}).get("scientificName")
        }
    )
    taxon_ids = sorted(
        {
            str(reference.get("organism", {}).get("taxonId"))
            for reference in preferred
            if reference.get("organism", {}).get("taxonId") is not None
        }
    )
    protein_names = values("proteinName")
    gene_names = values("geneName")
    sequence = payload.get("sequence", {})
    return {
        "source_representative_id": source_id,
        "resolution_source": "UniParc_sequence_archive",
        "uniparc_id": payload.get("uniParcId", ""),
        "current_accessions": ";".join(values("id")),
        "organisms": ";".join(organisms),
        "taxon_ids": ";".join(taxon_ids),
        "gene_names": ";".join(gene_names),
        "protein_names": ";".join(protein_names) or "Uncharacterized protein",
        "sequence_length": sequence.get("length", ""),
        "sequence_md5": sequence.get("md5", ""),
    }


def annotation_category(protein_names: str) -> str:
    lowered = protein_names.lower()
    generic = ("uncharacterized", "hypothetical", "unknown function")
    return "uncharacterized" if any(term in lowered for term in generic) else "annotated"


def fetch_json(
    session: requests.Session, url: str
) -> tuple[dict[str, Any], dict[str, str]]:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower().startswith("x-uniprot")
    }
    return response.json(), headers


def resolve_identifier(
    session: requests.Session, representative_id: str
) -> tuple[dict[str, object], dict[str, Any], dict[str, str]]:
    if representative_id.startswith("UPI"):
        payload, headers = fetch_json(
            session, f"{UNIPROT_API}/uniparc/{representative_id}.json"
        )
        return parse_uniparc(payload, representative_id), payload, headers

    payload, headers = fetch_json(
        session, f"{UNIPROT_API}/uniprotkb/{representative_id}.json"
    )
    if payload.get("entryType") != "Inactive":
        return parse_active_uniprot(payload, representative_id), payload, headers

    uniparc_id = payload.get("extraAttributes", {}).get("uniParcId")
    if not uniparc_id:
        raise ValueError(f"inactive accession {representative_id} has no UniParc ID")
    uniparc_payload, uniparc_headers = fetch_json(
        session, f"{UNIPROT_API}/uniparc/{uniparc_id}.json"
    )
    combined_headers = {**headers, **uniparc_headers}
    raw_payload = {"inactive_uniprot": payload, "uniparc": uniparc_payload}
    return (
        parse_uniparc(uniparc_payload, representative_id),
        raw_payload,
        combined_headers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("results/intervention_readiness/discovery_candidate_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/intervention_readiness")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    candidates = candidates[candidates["internal_nomination"].astype(bool)].copy()
    if candidates.empty:
        raise ValueError("no internal nominations are available to annotate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "annotation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows: list[dict[str, object]] = []
    release_headers: dict[str, str] = {}
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for candidate in candidates.itertuples(index=False):
        representative_id = candidate.gene_id.removeprefix("UniRef90_")
        annotation, raw_payload, headers = resolve_identifier(session, representative_id)
        release_headers.update(headers)
        (cache_dir / f"{representative_id}.json").write_text(
            json.dumps(raw_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "gene_id": candidate.gene_id,
                **annotation,
                "annotation_category": annotation_category(
                    str(annotation["protein_names"])
                ),
                "biological_review_status": "pending_manual_review",
                "intervention_target_status": "not_yet_established",
                "lookup_url": (
                    f"{UNIPROT_API}/uniprotkb/{representative_id}.json"
                    if not representative_id.startswith("UPI")
                    else f"{UNIPROT_API}/uniparc/{representative_id}.json"
                ),
                "retrieved_at_utc": retrieved_at,
            }
        )

    annotations = candidates.merge(
        pd.DataFrame(rows), on="gene_id", how="left", validate="one_to_one"
    )
    annotations.to_csv(args.output_dir / "candidate_annotations.csv", index=False)
    audit = {
        "source": "UniProt REST API and UniParc sequence archive",
        "source_documentation": "https://www.uniprot.org/help/api_queries",
        "retrieved_at_utc": retrieved_at,
        "release_headers": release_headers,
        "n_internal_nominations": len(candidates),
        "n_annotated_records": len(annotations),
        "n_uncharacterized": int(
            annotations["annotation_category"].eq("uncharacterized").sum()
        ),
        "interpretation_boundary": (
            "Automated names describe archived sequences; they do not establish "
            "a CRC mechanism or an intervention target."
        ),
    }
    (args.output_dir / "annotation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        annotations[
            [
                "gene_id",
                "heldout_median_auc",
                "organisms",
                "protein_names",
                "annotation_category",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
