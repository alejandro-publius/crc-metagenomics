from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_research_catalog import build_catalog  # noqa: E402


def test_research_catalog_builds_with_expected_relations(tmp_path):
    database = build_catalog(tmp_path / "catalog.sqlite")

    with sqlite3.connect(database) as connection:
        n_samples = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        n_folds = connection.execute("SELECT COUNT(*) FROM fold_results").fetchone()[0]
        n_models = connection.execute(
            "SELECT COUNT(DISTINCT model) FROM fold_results"
        ).fetchone()[0]
        orphan_predictions = connection.execute(
            """
            SELECT COUNT(*)
            FROM predictions AS p
            LEFT JOIN samples AS s USING (sample_id)
            WHERE s.sample_id IS NULL
            """
        ).fetchone()[0]
        n_comparisons = connection.execute(
            "SELECT COUNT(*) FROM model_comparison"
        ).fetchone()[0]

    assert n_samples == 1522
    assert n_folds == 40
    assert n_models == 4
    assert orphan_predictions == 0
    assert n_comparisons == 10
