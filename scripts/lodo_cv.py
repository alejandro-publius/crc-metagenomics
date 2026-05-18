"""Library: country-aware leave-one-dataset-out cross-validation.

This module is a library imported by `train_baseline.py`,
`train_joint.py`, `seed_sensitivity.py`, `sensitivity_analysis.py`,
`bio_pathway_shortlist.py`, `confounder_adjustment.py`, and
`batch_correction.py`, and exercised by `tests/test_lodo_cv.py`. It is
not intended to be executed directly.

Exposed contracts:

- `get_lodo_splits(metadata, ..., country_col=None)` yields
  `(cohort, train_idx, test_idx, excluded_cohorts)` for each LODO fold.
  When `country_col` is provided, cohorts sharing the test cohort's
  country are removed from the training fold.
- `run_lodo_cv(model_fn, X, y, metadata, ...)` orchestrates the loop
  and optionally calls a per-fold `feature_filter_fn(X_train)` to
  pick training-fold-only features (prevents test-fold leakage).

All module-level code below is function definitions only; importing this
module has no side effects.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def get_lodo_splits(metadata, label_col="label", cohort_col="study_name",
                    country_col=None):
    """Yield (cohort, train_indices, test_indices) for each LODO fold.

    country_col: if provided, cohorts that share the same country as the test
    cohort are excluded from training. This prevents population-level
    confounding when multiple cohorts share the same geographic origin
    (e.g., two Japanese cohorts: training on one while testing on the other
    would let the model learn country-specific — rather than CRC-specific —
    microbial signals).
    """
    # Build cohort -> country map (majority country per cohort)
    cohort_country = {}
    if country_col is not None and country_col in metadata.columns:
        cohort_country = (
            metadata.groupby(cohort_col)[country_col]
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )

    for cohort in sorted(metadata[cohort_col].unique()):
        test_mask = (metadata[cohort_col] == cohort) & \
                    (metadata[label_col].isin([0, 1]))

        if cohort_country:
            test_country = cohort_country.get(cohort)
            # Cohorts from the same country as the test cohort (excluding
            # the test cohort itself) are held out of training.
            same_country = {c for c, ct in cohort_country.items()
                            if ct == test_country and c != cohort}
            train_mask = (
                (metadata[cohort_col] != cohort) &
                (~metadata[cohort_col].isin(same_country)) &
                (metadata[label_col].isin([0, 1]))
            )
        else:
            same_country = set()
            train_mask = (metadata[cohort_col] != cohort) & \
                         (metadata[label_col].isin([0, 1]))

        train_idx = metadata[train_mask].index.tolist()
        test_idx  = metadata[test_mask].index.tolist()

        if len(test_idx) == 0:
            continue
        if len(metadata.loc[test_idx, label_col].unique()) < 2:
            continue

        yield cohort, train_idx, test_idx, same_country


def run_lodo_cv(model_fn, X, y, metadata, cohort_col="study_name",
                save_predictions_path=None, feature_filter_fn=None,
                country_col=None):
    """Run LODO cross-validation.

    Parameters
    ----------
    model_fn : callable -> estimator
    X : DataFrame of features (aligned with metadata index)
    y : Series of labels (0/1)
    metadata : DataFrame with cohort_col and optionally country_col
    cohort_col : column identifying cohorts
    save_predictions_path : optional CSV path for per-sample predictions
    feature_filter_fn : optional callable(X_train) -> list[col names]
        Applied inside each fold to prevent test-fold leakage.
    country_col : optional column for country-aware LODO (see get_lodo_splits)
    """
    results = {"cohort": [], "auc": [], "n_train": [], "n_test": [],
               "n_features": [], "excluded_cohorts": []}
    pred_rows = []

    for cohort, train_idx, test_idx, excluded in get_lodo_splits(
            metadata, cohort_col=cohort_col, country_col=country_col):

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

        excl_str = f'  [excl: {sorted(excluded)}]' if excluded else ''
        print(f'  {cohort:25s}  AUC={auc:.3f}  '
              f'(n_test={len(test_idx)}, n_train={len(train_idx)}, '
              f'p={n_feat}){excl_str}')

        results["cohort"].append(cohort)
        results["auc"].append(auc)
        results["n_train"].append(len(train_idx))
        results["n_test"].append(len(test_idx))
        results["n_features"].append(n_feat)
        results["excluded_cohorts"].append(sorted(excluded))

        if save_predictions_path is not None:
            sids = (metadata.loc[test_idx, 'sample_id'].values
                    if 'sample_id' in metadata.columns
                    else np.array(test_idx))
            for sid, yt, yp in zip(sids, y_true, y_prob):
                pred_rows.append({'sample_id': sid, 'cohort': cohort,
                                  'y_true': int(yt), 'y_prob': float(yp)})

    results["mean_auc"] = np.mean(results["auc"])
    results["std_auc"]  = np.std(results["auc"])
    print(f'\n  Mean AUC: {results["mean_auc"]:.3f} +/- {results["std_auc"]:.3f}')

    if save_predictions_path is not None:
        d = os.path.dirname(save_predictions_path)
        if d:
            os.makedirs(d, exist_ok=True)
        pd.DataFrame(pred_rows).to_csv(save_predictions_path, index=False)
        print(f'  Saved {len(pred_rows)} predictions to {save_predictions_path}')

    return results
