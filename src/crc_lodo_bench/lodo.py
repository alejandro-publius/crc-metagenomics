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

Edge-case validation wrappers
-----------------------------
Both exports are wrapped in :func:`_validate_lodo_inputs` so the same
defensive checks fire regardless of which backend (canonical or
vendored) is dispatched to. The validation layer catches:

  * NaN labels in ``label_col`` (raises ``ValueError``)
  * Duplicate ``sample_id`` (raises ``ValueError``)
  * Misaligned ``X`` / ``y`` / ``metadata`` indices (raises ``ValueError``)
  * Empty feature matrix at entry (raises ``ValueError``)
  * Missing ``country_col`` when requested (raises ``KeyError``)
  * Cohorts smaller than ``min_samples_per_cohort`` (warns)
  * Cohorts with a single label class (warns and skips)
  * NaN values in the ``country_col`` (warns)
  * Extreme per-fold class imbalance (warns)
  * Empty surviving feature set after ``feature_filter_fn`` (raises)
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Defensive validation helpers
# ---------------------------------------------------------------------------

DEFAULT_MIN_SAMPLES_PER_COHORT = 10
DEFAULT_MIN_MINORITY_FRACTION = 0.10


def _validate_lodo_inputs(
    metadata: pd.DataFrame,
    *,
    label_col: str = "label",
    cohort_col: str = "study_name",
    country_col: Optional[str] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    min_samples_per_cohort: int = DEFAULT_MIN_SAMPLES_PER_COHORT,
) -> None:
    """Validate LODO inputs before dispatching to the backend.

    Raises ``ValueError`` / ``KeyError`` for hard failures and emits
    ``UserWarning`` for soft (recoverable) issues. Intentionally
    side-effect free apart from raising / warning.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            f"metadata must be a pandas DataFrame, got {type(metadata).__name__}"
        )
    if cohort_col not in metadata.columns:
        raise KeyError(
            f"cohort_col {cohort_col!r} not found in metadata columns "
            f"(have: {list(metadata.columns)[:8]}...)"
        )
    if label_col not in metadata.columns:
        raise KeyError(
            f"label_col {label_col!r} not found in metadata columns "
            f"(have: {list(metadata.columns)[:8]}...)"
        )

    # NaN labels: refuse silently mis-classifying these rows as 0.
    if metadata[label_col].isna().any():
        n_nan = int(metadata[label_col].isna().sum())
        raise ValueError(
            f"metadata[{label_col!r}] has {n_nan} NaN value(s); LODO refuses "
            f"to silently treat NaN as 0 or drop without explicit upstream "
            f"handling. Pre-filter your metadata before calling run_lodo_cv."
        )

    # Duplicate sample_ids: silently produces wrong joins / predictions.
    if "sample_id" in metadata.columns:
        if metadata["sample_id"].isna().any():
            n_nan = int(metadata["sample_id"].isna().sum())
            raise ValueError(
                f"metadata['sample_id'] has {n_nan} NaN value(s); cannot "
                f"identify samples uniquely."
            )
        if not metadata["sample_id"].is_unique:
            dups = metadata["sample_id"][metadata["sample_id"].duplicated()]
            sample = list(dups.unique())[:5]
            raise ValueError(
                f"metadata['sample_id'] is not unique: "
                f"{metadata['sample_id'].duplicated().sum()} duplicate(s) "
                f"(e.g. {sample}). LODO requires unique sample_ids."
            )

    # country_col handling.
    if country_col is not None:
        if country_col not in metadata.columns:
            raise KeyError(
                f"country_col {country_col!r} not found in metadata columns. "
                f"Either pass country_col=None to disable country-aware "
                f"exclusion, or add the column upstream."
            )
        if metadata[country_col].isna().any():
            n_nan = int(metadata[country_col].isna().sum())
            warnings.warn(
                f"country_col {country_col!r} has {n_nan} NaN value(s); the "
                f"affected rows will be treated as country=NaN and may share "
                f"a country group with each other unexpectedly. Consider "
                f"filling these upstream.",
                UserWarning,
                stacklevel=3,
            )

    # X / y / metadata alignment when X and y are provided (run_lodo_cv path).
    if X is not None and y is not None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X).__name__}"
            )
        if X.shape[1] == 0:
            raise ValueError(
                "X has 0 feature columns; nothing to train on. Check upstream "
                "filtering / column selection."
            )
        if len(X) != len(metadata) or len(y) != len(metadata):
            raise ValueError(
                f"X / y / metadata length mismatch: "
                f"len(X)={len(X)}, len(y)={len(y)}, len(metadata)={len(metadata)}. "
                f"All three must be aligned and indexed identically before "
                f"calling run_lodo_cv."
            )
        if not (X.index.equals(metadata.index) and pd.Index(y.index).equals(metadata.index)):
            raise ValueError(
                "X, y, and metadata must share an identical index "
                "(reset_index(drop=True) on all three before calling). "
                "Mismatched indices cause silent join errors."
            )

    # Per-cohort soft checks: single-class folds and small cohorts.
    valid = metadata[metadata[label_col].isin([0, 1])]
    if len(valid) == 0:
        raise ValueError(
            f"metadata[{label_col!r}] contains no values in {{0, 1}}; "
            f"nothing to split."
        )

    for cohort, sub in valid.groupby(cohort_col):
        n = len(sub)
        if n < min_samples_per_cohort:
            warnings.warn(
                f"cohort {cohort!r} has only {n} sample(s) (< "
                f"min_samples_per_cohort={min_samples_per_cohort}); "
                f"per-fold AUC will be unstable.",
                UserWarning,
                stacklevel=3,
            )
        classes = sub[label_col].unique()
        if len(classes) < 2:
            warnings.warn(
                f"cohort {cohort!r} has only one label class "
                f"({sorted(classes)}); this fold will be skipped by the "
                f"LODO loop (roc_auc_score is undefined for a single class).",
                UserWarning,
                stacklevel=3,
            )
            continue
        # Class imbalance warning.
        minority = float(min((sub[label_col] == 0).mean(),
                             (sub[label_col] == 1).mean()))
        if minority < DEFAULT_MIN_MINORITY_FRACTION:
            warnings.warn(
                f"cohort {cohort!r} has minority class fraction "
                f"{minority:.2%} (< {DEFAULT_MIN_MINORITY_FRACTION:.0%}); "
                f"model may overfit the majority class on this fold.",
                UserWarning,
                stacklevel=3,
            )


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
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
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
                if ct == test_country and c != cohort and ct is not None
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
        When ``None``, all columns in ``X`` are used as features
        (intentional: callers may pre-filter globally upstream).
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
            if len(kept) == 0:
                raise ValueError(
                    f"feature_filter_fn returned 0 surviving features for "
                    f"fold {cohort!r}; cannot fit a model with no features. "
                    f"Loosen thresholds or check input feature columns."
                )
            X_tr = X_tr[kept]
            X_te = X_te[kept]
            n_feat = len(kept)
        else:
            n_feat = X_tr.shape[1]

        if n_feat == 0:
            raise ValueError(
                f"fold {cohort!r}: feature matrix has 0 columns; cannot fit."
            )

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

_backend_get_lodo_splits = _canonical_splits or _vendored_get_lodo_splits
_backend_run_lodo_cv = _canonical_run or _vendored_run_lodo_cv


def get_lodo_splits(
    metadata: pd.DataFrame,
    label_col: str = "label",
    cohort_col: str = "study_name",
    country_col: Optional[str] = None,
    *,
    min_samples_per_cohort: int = DEFAULT_MIN_SAMPLES_PER_COHORT,
) -> Iterator[tuple[Any, list, list, set]]:
    """Validate inputs, then dispatch to the backend split generator.

    The validation layer (see module docstring) runs once at entry; the
    yielded fold tuples are produced by the canonical backend
    (``scripts/lodo_cv.py`` when available, vendored copy otherwise).
    """
    _validate_lodo_inputs(
        metadata,
        label_col=label_col,
        cohort_col=cohort_col,
        country_col=country_col,
        min_samples_per_cohort=min_samples_per_cohort,
    )
    yield from _backend_get_lodo_splits(
        metadata,
        label_col=label_col,
        cohort_col=cohort_col,
        country_col=country_col,
    )


def run_lodo_cv(
    model_fn: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    cohort_col: str = "study_name",
    save_predictions_path: Optional[str] = None,
    feature_filter_fn: Optional[Callable[[pd.DataFrame], Sequence[str]]] = None,
    country_col: Optional[str] = None,
    *,
    min_samples_per_cohort: int = DEFAULT_MIN_SAMPLES_PER_COHORT,
) -> dict:
    """Validate inputs, then dispatch to the backend LODO loop.

    Defensive checks fire before the backend executes (see module
    docstring for the full list). After dispatch, ``feature_filter_fn``
    outputs are also checked per fold for emptiness (handled inside the
    vendored backend; the canonical backend will raise a less helpful
    error in that edge case, but the entry-level checks catch most
    upstream problems first).
    """
    _validate_lodo_inputs(
        metadata,
        label_col="label",
        cohort_col=cohort_col,
        country_col=country_col,
        X=X,
        y=y,
        min_samples_per_cohort=min_samples_per_cohort,
    )

    # Wrap feature_filter_fn so a 0-feature result raises with a clear
    # message regardless of which backend dispatches.
    if feature_filter_fn is not None:
        original_filter = feature_filter_fn

        def _checked_filter(X_train: pd.DataFrame) -> Sequence[str]:
            kept = original_filter(X_train)
            if len(kept) == 0:
                raise ValueError(
                    "feature_filter_fn returned 0 surviving features; "
                    "cannot fit a model with no features. Loosen thresholds "
                    "or check input feature columns."
                )
            return kept

        wrapped_filter: Optional[Callable] = _checked_filter
    else:
        wrapped_filter = None

    return _backend_run_lodo_cv(
        model_fn,
        X,
        y,
        metadata,
        cohort_col=cohort_col,
        save_predictions_path=save_predictions_path,
        feature_filter_fn=wrapped_filter,
        country_col=country_col,
    )


__all__ = [
    "get_lodo_splits",
    "run_lodo_cv",
    "DEFAULT_MIN_SAMPLES_PER_COHORT",
    "DEFAULT_MIN_MINORITY_FRACTION",
]
