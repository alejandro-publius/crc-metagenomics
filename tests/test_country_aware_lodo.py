"""Unit tests for ``get_lodo_splits`` country-aware exclusion semantics.

These tests are deliberately narrow: they only assert the *split*
behaviour (which cohorts go into train, which are excluded, which
become the test fold). End-to-end correctness on a known-answer
3-cohort case is covered by ``tests/test_country_aware_lodo_integration.py``.

The contract under test:

  - When ``country_col=None`` the routine degrades to plain LODO: every
    other cohort goes into training, nothing is excluded.
  - When ``country_col`` is provided, cohorts sharing the held-out
    cohort's majority country are removed from training and reported
    in the ``excluded`` set.
  - Singleton-country cohorts produce an empty exclusion set.
  - Rows whose label is not in {0, 1} are excluded from both train and
    test indices (so adenoma=-1 rows are not silently leaked).
  - Folds with fewer than two label classes in test are skipped.

Run with:
    pytest tests/test_country_aware_lodo.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Test scripts/lodo_cv.py directly (canonical source-of-truth).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import lodo_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metadata(layout):
    """Build a metadata DataFrame from a list of (cohort, country, n_pos, n_neg)."""
    rows = []
    for cohort, country, n_pos, n_neg in layout:
        for i in range(n_pos):
            rows.append({"sample_id": f"{cohort}_pos_{i}",
                         "study_name": cohort, "country": country, "label": 1})
        for i in range(n_neg):
            rows.append({"sample_id": f"{cohort}_neg_{i}",
                         "study_name": cohort, "country": country, "label": 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. No country column -> plain LODO behaviour
# ---------------------------------------------------------------------------


def test_no_country_col_yields_plain_lodo_splits():
    md = _make_metadata([
        ("A", "X", 5, 5),
        ("B", "X", 5, 5),
        ("C", "Y", 5, 5),
    ])
    splits = list(lodo_cv.get_lodo_splits(md, label_col="label",
                                          cohort_col="study_name",
                                          country_col=None))
    by_cohort = {c: (tr, te, ex) for c, tr, te, ex in splits}
    assert set(by_cohort) == {"A", "B", "C"}
    for cohort, (tr, te, ex) in by_cohort.items():
        # No exclusion when country_col is None.
        assert ex == set()
        train_cohorts = set(md.loc[tr, "study_name"])
        # Train must contain every cohort that isn't the test one.
        assert cohort not in train_cohorts
        assert train_cohorts == {"A", "B", "C"} - {cohort}


# ---------------------------------------------------------------------------
# 2. Country-aware exclusion: same-country cohorts dropped from train
# ---------------------------------------------------------------------------


def test_same_country_cohorts_excluded_from_training():
    md = _make_metadata([
        ("A", "X", 5, 5),
        ("B", "X", 5, 5),  # shares country X with A
        ("C", "Y", 5, 5),
        ("D", "Y", 5, 5),  # shares country Y with C
        ("E", "Z", 5, 5),
    ])
    by_cohort = {
        c: (tr, te, ex)
        for c, tr, te, ex in lodo_cv.get_lodo_splits(
            md, cohort_col="study_name", country_col="country"
        )
    }

    # Holding out A excludes B (both in X), keeps {C, D, E}.
    _, _, ex_A = by_cohort["A"]
    assert ex_A == {"B"}
    train_A = set(md.loc[by_cohort["A"][0], "study_name"])
    assert train_A == {"C", "D", "E"}

    # Holding out C excludes D, keeps {A, B, E}.
    _, _, ex_C = by_cohort["C"]
    assert ex_C == {"D"}

    # Holding out E excludes nothing (singleton in country Z).
    _, _, ex_E = by_cohort["E"]
    assert ex_E == set()
    train_E = set(md.loc[by_cohort["E"][0], "study_name"])
    assert train_E == {"A", "B", "C", "D"}


def test_majority_country_resolves_mixed_country_cohort():
    """If a single cohort spans two countries, the majority country wins
    when deciding which other cohorts to exclude."""
    rows = []
    # cohort A: 8 rows in country X, 2 rows in country Y -> majority X.
    for i in range(8):
        rows.append({"sample_id": f"A_x_{i}", "study_name": "A",
                     "country": "X", "label": i % 2})
    for i in range(2):
        rows.append({"sample_id": f"A_y_{i}", "study_name": "A",
                     "country": "Y", "label": i % 2})
    # cohort B is in country X (so it should be excluded when A is held out).
    for i in range(10):
        rows.append({"sample_id": f"B_{i}", "study_name": "B",
                     "country": "X", "label": i % 2})
    # cohort C is in country Y.
    for i in range(10):
        rows.append({"sample_id": f"C_{i}", "study_name": "C",
                     "country": "Y", "label": i % 2})
    md = pd.DataFrame(rows)

    by_cohort = {
        c: (tr, te, ex)
        for c, tr, te, ex in lodo_cv.get_lodo_splits(
            md, cohort_col="study_name", country_col="country"
        )
    }

    # A's majority country is X, so B (also X) must be excluded.
    assert by_cohort["A"][2] == {"B"}
    train_A = set(md.loc[by_cohort["A"][0], "study_name"])
    assert "B" not in train_A
    assert "C" in train_A


# ---------------------------------------------------------------------------
# 3. Label sanitation: non-binary rows are not leaked
# ---------------------------------------------------------------------------


def test_non_binary_label_rows_are_excluded():
    """Adenoma rows (label=-1) must appear in neither train nor test indices."""
    rows = []
    for i in range(5):
        rows.append({"sample_id": f"A_neg_{i}", "study_name": "A",
                     "country": "X", "label": 0})
        rows.append({"sample_id": f"A_pos_{i}", "study_name": "A",
                     "country": "X", "label": 1})
        rows.append({"sample_id": f"A_aden_{i}", "study_name": "A",
                     "country": "X", "label": -1})
        rows.append({"sample_id": f"B_neg_{i}", "study_name": "B",
                     "country": "Y", "label": 0})
        rows.append({"sample_id": f"B_pos_{i}", "study_name": "B",
                     "country": "Y", "label": 1})
        rows.append({"sample_id": f"B_aden_{i}", "study_name": "B",
                     "country": "Y", "label": -1})
    md = pd.DataFrame(rows)

    for cohort, tr, te, _ in lodo_cv.get_lodo_splits(
            md, cohort_col="study_name", country_col="country"):
        # No -1 labels should leak into either split.
        assert (md.loc[tr, "label"].isin([0, 1])).all()
        assert (md.loc[te, "label"].isin([0, 1])).all()


def test_single_class_test_fold_is_skipped():
    """If a cohort only has one class present, it must not be yielded as a
    test fold (the AUC would be undefined)."""
    rows = []
    # Cohort A has both classes.
    for i in range(5):
        rows.append({"sample_id": f"A_neg_{i}", "study_name": "A",
                     "country": "X", "label": 0})
        rows.append({"sample_id": f"A_pos_{i}", "study_name": "A",
                     "country": "X", "label": 1})
    # Cohort B has only positives.
    for i in range(5):
        rows.append({"sample_id": f"B_pos_{i}", "study_name": "B",
                     "country": "Y", "label": 1})
    md = pd.DataFrame(rows)
    yielded = [c for c, *_ in lodo_cv.get_lodo_splits(
        md, cohort_col="study_name", country_col="country")]
    assert "A" in yielded
    assert "B" not in yielded


# ---------------------------------------------------------------------------
# 4. Yield order is deterministic (alphabetical by cohort name)
# ---------------------------------------------------------------------------


def test_split_yield_order_is_sorted_by_cohort():
    md = _make_metadata([
        ("zzz_cohort", "X", 3, 3),
        ("aaa_cohort", "Y", 3, 3),
        ("mmm_cohort", "Z", 3, 3),
    ])
    order = [c for c, *_ in lodo_cv.get_lodo_splits(
        md, cohort_col="study_name", country_col="country")]
    assert order == sorted(order)


# ---------------------------------------------------------------------------
# 5. Vendored fallback in the package matches the canonical implementation
# ---------------------------------------------------------------------------


def test_vendored_lodo_splits_match_canonical():
    """The vendored ``_vendored_get_lodo_splits`` in src/crc_lodo_bench/lodo.py
    must produce the same splits as the canonical scripts/lodo_cv.py on a
    shared synthetic dataset."""
    _SRC = os.path.join(_REPO_ROOT, "src")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from crc_lodo_bench import lodo as pkg_lodo

    md = _make_metadata([
        ("A", "X", 5, 5),
        ("B", "X", 5, 5),
        ("C", "Y", 5, 5),
    ])
    canonical = list(lodo_cv.get_lodo_splits(
        md, cohort_col="study_name", country_col="country"))
    vendored = list(pkg_lodo._vendored_get_lodo_splits(
        md, cohort_col="study_name", country_col="country"))
    assert len(canonical) == len(vendored)
    for (cA, trA, teA, exA), (cV, trV, teV, exV) in zip(canonical, vendored):
        assert cA == cV
        assert trA == trV
        assert teA == teV
        assert exA == exV
