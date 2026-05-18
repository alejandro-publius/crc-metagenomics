"""Country-aware leave-one-dataset-out (LODO) cross-validation.

This module re-exports :func:`get_lodo_splits` and :func:`run_lodo_cv`
from the canonical implementation in ``scripts/lodo_cv.py`` at the root
of the crc-metagenomics research repository.

When the package is installed from a checkout of the research repo
(``pip install -e .`` at the repo root), the import below resolves
to the original module so there is a single source of truth.

When the package is installed standalone (e.g. ``pip install
crc-lodo-bench`` from PyPI / a sdist), the canonical script is not
on the Python path, so we fall back to the vendored copy in
:func:`_vendored_get_lodo_splits` / :func:`_vendored_run_lodo_cv`
below. The vendored implementation is intentionally byte-identical to
the canonical one as of v0.1.0; see the attribution comment on each
function.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _try_import_canonical() -> tuple[Optional[Callable], Optional[Callable]]:
    """Try to import the canonical lodo_cv module from the research repo.

    Walks upward from this file looking for a ``scripts/lodo_cv.py``
    sibling of a ``src/crc_lodo_bench`` directory; if found, that
    directory is prepended to ``sys.path`` and the module is imported.

    Returns (get_lodo_splits, run_lodo_cv) on success, (None, None)
    otherwise. Failures are swallowed so the vendored fallback applies.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    # src/crc_lodo_bench/lodo.py -> repo root is two directories up.
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    scripts_dir = os.path.join(repo_root, "scripts")
    canonical = os.path.join(scripts_dir, "lodo_cv.py")
    if not os.path.isfile(canonical):
        return None, None
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import lodo_cv as _canonical  # type: ignore[import-not-found]
    except Exception:
        return None, None
    return (
        getattr(_canonical, "get_lodo_splits", None),
        getattr(_canonical, "run_lodo_cv", None),
    )


# ---------------------------------------------------------------------------
# Vendored fallback (kept byte-identical to scripts/lodo_cv.py as of v0.1.0).
#
# Attribution: this is a verbatim copy of the country-aware LODO routine
# from `scripts/lodo_cv.py` in the crc-metagenomics repository. It is
# duplicated here only so that the package remains usable when installed
# outside a checkout of the research repo; please prefer the canonical
# implementation when both are available.
# ---------------------------------------------------------------------------


def _vendored_get_lodo_splits(
    metadata: pd.DataFrame,
    label_col: str = "label",
    cohort_col: str = "study_name",
    country_col: Optional[str] = None,
) -> Iterator[tuple[Any, list, list, set]]:
    """Yield ``(cohort, train_indices, test_indices, excluded_cohorts)``
    for each leave-one-dataset-out fold.

    Parameters
    ----------
    metadata
        DataFrame with one row per sample. Must contain ``cohort_col``
        and ``label_col``; optionally contains ``country_col``.
    label_col
        Binary label column (0/1). Rows whose label is not in {0, 1}
        are excluded from both train and test folds.
    cohort_col
        Cohort / study identifier column.
    country_col
        If provided, cohorts that share the same (majority) country as
        the held-out test cohort are excluded from the training fold.
        This prevents population-level confounding when multiple
        cohorts share a geographic origin (e.g. two Japanese cohorts).

    Yields
    ------
    cohort, train_idx, test_idx, excluded
        ``cohort`` is the held-out cohort identifier; ``train_idx`` and
        ``test_idx`` are lists of metadata row indices; ``excluded`` is
        the set of cohorts dropped from training due to country sharing.
    """
    cohort_country: dict = {}
    if country_col is not None and country_col in metadata.columns:
        cohort_country = (
            metadata.groupby(cohort_col)[country_col]
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )

    for cohort in sorted(metadata[cohort_col].unique()):
        test_mask = (metadata[cohort_col] == cohort) & (
            metadata[label_col].isin([0, 1])
        )

        if cohort_country:
            test_country = cohort_country.get(cohort)
            same_country = {
                c for c, ct in cohort_country.items()
                if ct == test_country and c != cohort
            }
            train_mask = (
                (metadata[cohort_col] != cohort)
                & (~metadata[cohort_col].isin(same_country))
                & (metadata[label_col].isin([0, 1]))
            )
        else:
            same_country = set()
            train_mask = (metadata[cohort_col] != cohort) & (
                metadata[label_col].isin([0, 1])
            )

        train_idx = metadata[train_mask].index.tolist()
        test_idx = metadata[test_mask].index.tolist()

        if len(test_idx) == 0:
            continue
        if len(metadata.loc[test_idx, label_col].unique()) < 2:
            continue

        yield cohort, train_idx, test_idx, same_country


def _vendored_run_lodo_cv(
    model_fn: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    cohort_col: str = "study_name",
    save_predictions_path: Optional[str] = None,
    feature_filter_fn: Optional[Callable[[pd.DataFrame], Sequence[str]]] = None,
    country_col: Optional[str] = None,
) -> dict:
    """Run country-aware leave-one-dataset-out cross-validation.

    Parameters
    ----------
    model_fn
        Zero-argument callable returning a fresh sklearn-style
        estimator with ``fit`` and ``predict_proba``.
    X
        Feature matrix aligned with ``metadata`` row index.
    y
        Binary label vector (0/1) aligned with ``metadata`` row index.
    metadata
        Sample metadata with ``cohort_col`` and optionally ``country_col``.
    cohort_col
        Column identifying cohorts for the LODO split.
    save_predictions_path
        If given, per-sample held-out predictions are written to CSV
        with columns ``sample_id, cohort, y_true, y_prob``.
    feature_filter_fn
        Optional callable applied per fold as
        ``feature_filter_fn(X_train) -> list[col_names]``. This is the
        recommended way to prevent test-fold leakage when the feature
        filter (prevalence / mean abundance / variance) depends on data.
    country_col
        Forwarded to :func:`get_lodo_splits` for country-aware LODO.

    Returns
    -------
    results : dict
        Keys: ``cohort, auc, n_train, n_test, n_features,
        excluded_cohorts, mean_auc, std_auc``. The per-fold lists are
        aligned (one entry per yielded LODO fold).
    """
    results: dict = {
        "cohort": [],
        "auc": [],
        "n_train": [],
        "n_test": [],
        "n_features": [],
        "excluded_cohorts": [],
    }
    pred_rows: list = []

    for cohort, train_idx, test_idx, excluded in _vendored_get_lodo_splits(
        metadata, cohort_col=cohort_col, country_col=country_col
    ):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]

        if feature_filter_fn is not None:
            kept = feature_filter_fn(X_tr)
            X_tr = X_tr[kept]
            X_te = X_te[kept]
            n_feat = len(kept)
        else:
            n_feat = X_tr.shape[1]

        model = model_fn()
        model.fit(X_tr, y.iloc[train_idx])
        y_prob = model.predict_proba(X_te)[:, 1]
        y_true = y.iloc[test_idx].values
        auc = roc_auc_score(y_true, y_prob)

        excl_str = f"  [excl: {sorted(excluded)}]" if excluded else ""
        print(
            f"  {cohort:25s}  AUC={auc:.3f}  "
            f"(n_test={len(test_idx)}, n_train={len(train_idx)}, "
            f"p={n_feat}){excl_str}"
        )

        results["cohort"].append(cohort)
        results["auc"].append(auc)
        results["n_train"].append(len(train_idx))
        results["n_test"].append(len(test_idx))
        results["n_features"].append(n_feat)
        results["excluded_cohorts"].append(sorted(excluded))

        if save_predictions_path is not None:
            sids = (
                metadata.loc[test_idx, "sample_id"].values
                if "sample_id" in metadata.columns
                else np.array(test_idx)
            )
            for sid, yt, yp in zip(sids, y_true, y_prob):
                pred_rows.append({
                    "sample_id": sid,
                    "cohort": cohort,
                    "y_true": int(yt),
                    "y_prob": float(yp),
                })

    results["mean_auc"] = float(np.mean(results["auc"])) if results["auc"] else float("nan")
    results["std_auc"] = float(np.std(results["auc"])) if results["auc"] else float("nan")
    print(
        f"\n  Mean AUC: {results['mean_auc']:.3f} "
        f"+/- {results['std_auc']:.3f}"
    )

    if save_predictions_path is not None:
        d = os.path.dirname(save_predictions_path)
        if d:
            os.makedirs(d, exist_ok=True)
        pd.DataFrame(pred_rows).to_csv(save_predictions_path, index=False)
        print(
            f"  Saved {len(pred_rows)} predictions to "
            f"{save_predictions_path}"
        )

    return results


# Resolve canonical vs vendored exports at import time.
_canonical_splits, _canonical_run = _try_import_canonical()

get_lodo_splits = _canonical_splits or _vendored_get_lodo_splits
run_lodo_cv = _canonical_run or _vendored_run_lodo_cv


__all__ = ["get_lodo_splits", "run_lodo_cv"]
