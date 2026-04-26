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
                save_predictions_path=None):
    """Run LODO cross-validation.

    If save_predictions_path is given, writes a long-format CSV with one row
    per test-fold sample (sample_id, cohort, y_true, y_prob). Used by
    auc_comparison for DeLong tests on pooled LODO ROC curves.
    """
    results = {"cohort": [], "auc": [], "n_train": [], "n_test": []}
    pred_rows = []
    for cohort, train_idx, test_idx in get_lodo_splits(metadata, cohort_col=cohort_col):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_true = y.iloc[test_idx].values
        auc = roc_auc_score(y_true, y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        results["n_train"].append(len(train_idx))
        results["n_test"].append(len(test_idx))
        print(f'  {cohort:25s}  AUC={auc:.3f}  (n={len(test_idx)})')
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
