"""Smoke tests for the crc_lodo_bench package.

These tests verify that the package imports cleanly, exposes the
documented public API, and that the headline functions are callable
on a tiny synthetic dataset. They do NOT replicate the per-fold
behavioural assertions in ``tests/test_lodo_cv.py`` (which exercises
the canonical implementation in ``scripts/lodo_cv.py``).

Run with:
    pytest tests/test_package_import.py -v
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make ``crc_lodo_bench`` importable without requiring ``pip install -e .``
# (which is the recommended path; this fallback only matters when running
# the test suite directly against a checkout, e.g. on a Python interpreter
# whose version pre-dates the package's ``requires-python``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def test_package_imports_cleanly():
    """``import crc_lodo_bench`` must succeed and expose ``__version__``."""
    pkg = importlib.import_module("crc_lodo_bench")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__ == "0.1.0"


def test_public_api_is_exported():
    """The five headline names must be importable from the top-level."""
    pkg = importlib.import_module("crc_lodo_bench")
    for name in (
        "get_lodo_splits",
        "run_lodo_cv",
        "per_fold_pathway_filter",
        "delong_test",
        "bootstrap_pooled_ci",
    ):
        assert hasattr(pkg, name), f"crc_lodo_bench is missing {name!r}"
        assert callable(getattr(pkg, name)), f"{name} is not callable"


def test_submodules_importable():
    """Each submodule must import without side effects."""
    for sub in ("lodo", "filters", "stats"):
        importlib.import_module(f"crc_lodo_bench.{sub}")


def _make_tiny_dataset(n_per_cohort: int = 20, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    feats = []
    for cohort, country in [("A", "X"), ("B", "X"), ("C", "Y")]:
        for i in range(n_per_cohort):
            label = i % 2
            rows.append({
                "sample_id": f"{cohort}_{i}",
                "study_name": cohort,
                "country": country,
                "label": label,
            })
            feats.append(rng.normal(loc=label * 0.5, scale=1.0, size=n_features))
    metadata = pd.DataFrame(rows)
    X = pd.DataFrame(
        feats,
        columns=[f"feat_{j}" for j in range(n_features)],
        index=metadata.index,
    )
    y = metadata["label"].copy()
    return X, y, metadata


class _MeanScorer:
    """Tiny estimator with predict_proba so we avoid sklearn fit overhead."""

    def fit(self, X, y):
        self.pos_mean_ = X[y.values == 1].mean(axis=0).values
        self.neg_mean_ = X[y.values == 0].mean(axis=0).values
        return self

    def predict_proba(self, X):
        # Higher score == closer to pos_mean_ than to neg_mean_.
        d_pos = np.linalg.norm(X.values - self.pos_mean_, axis=1)
        d_neg = np.linalg.norm(X.values - self.neg_mean_, axis=1)
        score = d_neg - d_pos
        prob = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1 - prob, prob])


def test_run_lodo_cv_callable_on_synthetic_data():
    """Smoke-run the public entrypoint end-to-end."""
    from crc_lodo_bench import run_lodo_cv

    X, y, metadata = _make_tiny_dataset()
    results = run_lodo_cv(
        _MeanScorer, X, y, metadata,
        cohort_col="study_name",
        country_col="country",
    )
    assert "mean_auc" in results
    assert "auc" in results and len(results["auc"]) == 3
    for auc in results["auc"]:
        assert 0.0 <= auc <= 1.0
    assert np.isfinite(results["mean_auc"])


def test_per_fold_pathway_filter_returns_callable():
    """Factory must return a callable that respects thresholds + passthrough."""
    from crc_lodo_bench import per_fold_pathway_filter

    cols = [f"pw_{i}" for i in range(5)]
    species = [f"sp_{i}" for i in range(2)]
    X = pd.DataFrame({
        **{c: np.linspace(0, 1, 20) for c in cols},
        **{c: np.ones(20) for c in species},
    })
    # Force pw_0 below prevalence (all zeros) so it is dropped; pw_4 kept.
    X["pw_0"] = 0.0

    filt = per_fold_pathway_filter(
        filtered_cols=cols,
        prevalence_threshold=0.10,
        mean_threshold=1e-6,
        passthrough_cols=species,
    )
    kept = filt(X)
    # Species pass through and come first.
    assert kept[: len(species)] == species
    # pw_0 (all zero) must be dropped.
    assert "pw_0" not in kept
    # At least one pathway survives.
    assert any(c.startswith("pw_") for c in kept)


def test_delong_test_on_known_paired_predictions():
    """delong_test must produce finite stats for two scored vectors."""
    from crc_lodo_bench import delong_test

    rng = np.random.default_rng(7)
    n = 200
    y = (rng.random(n) < 0.4).astype(int)
    # Two correlated classifiers with slightly different separation.
    base = y + rng.normal(scale=0.5, size=n)
    a = base + rng.normal(scale=0.1, size=n)
    b = base + rng.normal(scale=0.3, size=n)

    res = delong_test(y, a, b)
    assert {"auc_a", "auc_b", "auc_diff", "z", "p_value", "n"}.issubset(res)
    assert 0 <= res["auc_a"] <= 1
    assert 0 <= res["auc_b"] <= 1
    assert 0 <= res["p_value"] <= 1
    assert np.isfinite(res["z"])
    assert res["n"] == n


def test_bootstrap_pooled_ci_returns_valid_interval():
    """The pooled CI must bracket the point estimate and shrink with n_boot."""
    from crc_lodo_bench import bootstrap_pooled_ci

    rng = np.random.default_rng(11)
    n_per = 80
    cohorts = np.repeat(["A", "B", "C"], n_per)
    y = np.tile([0, 1], (3 * n_per) // 2)
    y_prob = y + rng.normal(scale=0.5, size=len(y))

    res = bootstrap_pooled_ci(y, y_prob, cohorts, n_boot=500, seed=0)
    assert res["ci_low"] <= res["auc"] <= res["ci_high"]
    assert 0 < res["ci_low"] < res["ci_high"] < 1
    assert res["n"] == len(y)
    assert res["n_boot_kept"] > 0


def test_public_functions_have_docstrings():
    """All exported functions must carry a docstring (citability hygiene)."""
    pkg = importlib.import_module("crc_lodo_bench")
    for name in (
        "get_lodo_splits",
        "run_lodo_cv",
        "per_fold_pathway_filter",
        "delong_test",
        "bootstrap_pooled_ci",
    ):
        fn = getattr(pkg, name)
        doc = inspect.getdoc(fn)
        assert doc, f"{name} is missing a docstring"
        assert len(doc) > 20, f"{name} docstring is too short to be useful"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
