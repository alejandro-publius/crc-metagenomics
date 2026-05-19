"""crc_lodo_bench: country-aware LODO benchmarking utilities for
shotgun metagenomic CRC classification.

This package exposes the reusable building blocks from the
crc-metagenomics study as a small, importable library:

    from crc_lodo_bench import run_lodo_cv, get_lodo_splits
    from crc_lodo_bench import per_fold_pathway_filter
    from crc_lodo_bench import delong_test, bootstrap_pooled_ci

The canonical implementation of the LODO loop lives in the project's
top-level ``scripts/lodo_cv.py``; this package re-exports it when run
from a checkout of the repository and falls back to a vendored copy
when installed standalone (so external users can `pip install
crc-lodo-bench` without needing the full research repo). Both export
paths share a defensive input-validation wrapper (see
``crc_lodo_bench.lodo`` for the full edge-case list).
"""

from .lodo import (
    DEFAULT_MIN_MINORITY_FRACTION,
    DEFAULT_MIN_SAMPLES_PER_COHORT,
    get_lodo_splits,
    run_lodo_cv,
)
from .filters import per_fold_pathway_filter
from .stats import bootstrap_pooled_ci, delong_test

__version__ = "0.1.0"

__all__ = [
    "get_lodo_splits",
    "run_lodo_cv",
    "per_fold_pathway_filter",
    "delong_test",
    "bootstrap_pooled_ci",
    "DEFAULT_MIN_SAMPLES_PER_COHORT",
    "DEFAULT_MIN_MINORITY_FRACTION",
    "__version__",
]
