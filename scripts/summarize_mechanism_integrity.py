"""Summarize how completely the frozen assay represents each known mechanism."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def expand_gene_specification(value: str) -> list[str]:
    genes: list[str] = []
    for token in str(value).split(";"):
        token = token.strip()
        match = re.fullmatch(r"([A-Za-z]+)([A-Za-z])-([A-Za-z]+)?([A-Za-z])", token)
        if match and (match.group(3) is None or match.group(1) == match.group(3)):
            prefix = match.group(1)
            start = ord(match.group(2))
            stop = ord(match.group(4))
            if start > stop:
                raise ValueError(f"invalid descending gene range: {token}")
            genes.extend(f"{prefix}{chr(code)}" for code in range(start, stop + 1))
        elif token:
            genes.append(token)
    if len(genes) != len(set(genes)):
        raise ValueError(f"duplicate prespecified genes in {value!r}")
    return genes


def summarize_integrity(registry: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in registry.itertuples(index=False):
        expected = expand_gene_specification(target.prespecified_genes)
        mechanism_rows = manifest[manifest["mechanism"].eq(target.score_prefix)]
        detected_rows = mechanism_rows[
            mechanism_rows["query_status"].eq("frozen_detected")
        ]
        detected_genes = sorted(
            set(detected_rows["prespecified_gene"].dropna().astype(str)) & set(expected)
        )
        missing_genes = sorted(set(expected) - set(detected_genes))
        n_clusters = detected_rows["uniref90"].replace("", pd.NA).dropna().nunique()
        if not detected_genes:
            status = "not_represented_in_frozen_assay"
        elif len(expected) == 1 and len(detected_genes) == 1:
            status = "single_effector_represented"
        elif len(detected_genes) == len(expected):
            status = "all_prespecified_genes_represented"
        else:
            status = "partial_multigene_representation"
        rows.append(
            {
                "target_id": target.target_id,
                "n_prespecified_genes": len(expected),
                "n_genes_represented": len(detected_genes),
                "represented_gene_fraction": len(detected_genes) / len(expected),
                "n_unique_detected_uniref90_clusters": int(n_clusters),
                "represented_genes": ";".join(detected_genes),
                "missing_genes": ";".join(missing_genes),
                "mechanism_integrity_status": status,
                "integrity_claim_boundary": (
                    "Representation counts frozen assay mappings; it does not "
                    "prove co-location, expression, pathway activity, or causality."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    registry = pd.read_csv("research/intervention_readiness/known_target_registry.csv")
    manifest = pd.read_csv("results/mechanism_panel/frozen_manifest.csv").fillna("")
    output_dir = Path("results/intervention_readiness")
    summary = summarize_integrity(registry, manifest)
    summary.to_csv(output_dir / "known_target_mechanism_integrity.csv", index=False)
    audit = {
        "status": "complete_assay_representation_summary",
        "n_targets": int(len(summary)),
        "n_fully_represented_multigene_mechanisms": int(
            summary["mechanism_integrity_status"]
            .eq("all_prespecified_genes_represented")
            .sum()
        ),
        "n_single_effectors_represented": int(
            summary["mechanism_integrity_status"]
            .eq("single_effector_represented")
            .sum()
        ),
        "interpretation_boundary": (
            "This is assay coverage, not a pass for biological mechanism integrity."
        ),
    }
    (output_dir / "known_target_mechanism_integrity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
