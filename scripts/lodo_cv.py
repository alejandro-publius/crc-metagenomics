import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

def get_lodo_splits(metadata, label_col="label", cohort_col="study_name"):
    for cohort in sorted(metadata[cohort_col].unique()):
        test_mask = (metadata[cohort_col] == cohort) & (metadata[label_col].isin([0, 1]))
        train_mask = (metadata[cohort_col] != cohort) & (metadata[label_col].isin([0, 1]))
        train_idx = metadata[train_mask].index.tolist()
        test_idx = metadata[test_mask].index.tolist()
        if len(metadata.loc[test_idx, label_col].unique()) < 2 or len(test_idx) == 0:
            continue
        yield cohort, train_idx, test_idx

def run_lodo_cv(model_fn, X, y, metadata, cohort_col="study_name",
                save_predictions_path=None, feature_filter_fn=None):
    """Run LODO cross-validation.

    feature_filter_fn: optional callable. Receives the train-fold slice of X
    (DataFrame) and returns a list of column names to retain. Applied inside
    each fold so feature selection uses only training data, with the same
    column set then applied to the held-out test fold. Use this for any
    data-derived feature filtering (e.g., pathway prevalence) to avoid
    test-fold leakage.

    save_predictions_path: optional CSV path. Writes one row per test-fold
    sample (sample_id, cohort, y_true, y_prob) for downstream DeLong tests.
    """
    results = {"cohort": [], "auc": [], "n_train": [], "n_test": [], "n_features": []}
    pred_rows = []
    for cohort, train_idx, test_idx in get_lodo_splits(metadata, cohort_col=cohort_col):
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
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        results["n_train"].append(len(train_idx))
        results["n_test"].append(len(test_idx))
        results["n_features"].append(n_feat)
        print(f'  {cohort:25s}  AUC={auc:.3f}  (n={len(test_idx)}, p={n_feat})')
        if save_predictions_path is not None:
            if 'sample_id' in metadata.columns:
                sids = metadata.loc[test_idx, 'sample_id'].values
            else:
                sids = np.array(test_idx)
            for sid, yt, yp in zip(sids, y_true, y_prob):
                pred_rows.append({'sample_id': sid, 'cohort': cohort,
                                  'y_true': int(yt), 'y_prob': float(yp)})
    results["mean_auc"] = np.mean(results["auc"])
    results["std_auc"] = np.std(results["auc"])
    print(f'\n  Mean AUC: {results["mean_auc"]:.3f} +/- {results["std_auc"]:.3f}')
    if save_predictions_path is not None:
        d = os.path.dirname(save_predictions_path)
        if d:
            os.makedirs(d, exist_ok=True)
        pd.DataFrame(pred_rows).to_csv(save_predictions_path, index=False)
        print(f'  Saved {len(pred_rows)} predictions to {save_predictions_path}')
    return results
