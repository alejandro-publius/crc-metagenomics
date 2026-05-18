"""Per-fold feature filtering for LODO benchmarking.

The headline :func:`per_fold_pathway_filter` factory returns a callable
suitable for the ``feature_filter_fn`` parameter of
:func:`crc_lodo_bench.run_lodo_cv`. It applies a prevalence and mean
abundance threshold to a configurable subset of feature columns and
returns the surviving columns plus any passthrough columns (e.g.
species features that were globally filtered upstream).

The filter is recomputed on the training-cohort samples of each LODO
fold so that no information from the held-out test cohort leaks into
the filter decision.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import pandas as pd


def per_fold_pathway_filter(
    filtered_cols: Sequence[str],
    *,
    prevalence_threshold: float = 0.10,
    mean_threshold: float = 1e-6,
    passthrough_cols: Optional[Iterable[str]] = None,
) -> Callable[[pd.DataFrame], list[str]]:
    """Build a per-fold prevalence + mean abundance filter.

    Parameters
    ----------
    filtered_cols
        Names of the feature columns the filter should evaluate
        (typically the candidate pathway columns).
    prevalence_threshold
        Minimum fraction of training samples in which a column must
        be non-zero. Default 0.10 matches the crc-metagenomics study.
    mean_threshold
        Minimum mean abundance across training samples. Default 1e-6.
    passthrough_cols
        Optional column names to always retain regardless of the
        thresholds (e.g. species features that were already filtered
        globally upstream). These are emitted first in the returned
        list so callers can preserve a stable feature order.

    Returns
    -------
    filter_fn
        Callable mapping a training feature DataFrame to the list of
        surviving column names. Use it as the ``feature_filter_fn``
        argument to :func:`crc_lodo_bench.run_lodo_cv`.

    Example
    -------
    >>> from crc_lodo_bench import run_lodo_cv, per_fold_pathway_filter
    >>> filt = per_fold_pathway_filter(
    ...     filtered_cols=pathway_cols,
    ...     prevalence_threshold=0.10,
    ...     mean_threshold=1e-6,
    ...     passthrough_cols=species_cols,
    ... )
    >>> results = run_lodo_cv(
    ...     model_fn, X, y, metadata,
    ...     feature_filter_fn=filt,
    ...     country_col="country",
    ... )
    """
    filtered_cols = list(filtered_cols)
    passthrough = list(passthrough_cols) if passthrough_cols is not None else []

    if not 0.0 <= prevalence_threshold <= 1.0:
        raise ValueError(
            f"prevalence_threshold must be in [0, 1], got {prevalence_threshold}"
        )
    if mean_threshold < 0:
        raise ValueError(
            f"mean_threshold must be non-negative, got {mean_threshold}"
        )

    def _filter(X_train: pd.DataFrame) -> list[str]:
        missing = [c for c in filtered_cols if c not in X_train.columns]
        if missing:
            raise KeyError(
                f"per_fold_pathway_filter: {len(missing)} filtered_cols "
                f"missing from X_train (e.g. {missing[:3]})"
            )
        sub = X_train[filtered_cols]
        prev = (sub > 0).mean(axis=0)
        ma = sub.mean(axis=0)
        kept = [
            c for c in filtered_cols
            if prev[c] >= prevalence_threshold and ma[c] >= mean_threshold
        ]
        return list(passthrough) + kept

    _filter.__name__ = "per_fold_pathway_filter"
    _filter.__qualname__ = "per_fold_pathway_filter"
    return _filter


__all__ = ["per_fold_pathway_filter"]
