from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_mechanism_integrity import (  # noqa: E402
    expand_gene_specification,
    summarize_integrity,
)


def test_letter_gene_range_expands_inclusively():
    assert expand_gene_specification("clbA-clbD") == ["clbA", "clbB", "clbC", "clbD"]


def test_aliases_sharing_one_cluster_keep_gene_and_cluster_counts_distinct():
    registry = pd.DataFrame(
        [
            {
                "target_id": "t",
                "score_prefix": "m",
                "prespecified_genes": "a;b;c",
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "mechanism": "m",
                "prespecified_gene": "a",
                "query_status": "frozen_detected",
                "uniref90": "u1",
            },
            {
                "mechanism": "m",
                "prespecified_gene": "b",
                "query_status": "frozen_detected",
                "uniref90": "u1",
            },
            {
                "mechanism": "m",
                "prespecified_gene": "c",
                "query_status": "mapped_but_not_detected",
                "uniref90": "u2",
            },
        ]
    )
    row = summarize_integrity(registry, manifest).iloc[0]
    assert row["n_genes_represented"] == 2
    assert row["n_unique_detected_uniref90_clusters"] == 1
    assert row["mechanism_integrity_status"] == "partial_multigene_representation"
