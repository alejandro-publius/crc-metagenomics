import numpy as np
import pandas as pd
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

def run_lodo_cv(model_fn, X, y, metadata, cohort_col="study_name"):
    results = {"cohort": [], "auc": [], "n_train": [], "n_test": []}
    for cohort, train_idx, test_idx in get_lodo_splits(metadata, cohort_col=cohort_col):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        results["n_train"].append(len(train_idx))
        results["n_test"].append(len(test_idx))
        print(f'  {cohort:25s}  AUC={auc:.3f}  (n={len(test_idx)})')
    results["mean_auc"] = np.mean(results["auc"])
    results["std_auc"] = np.std(results["auc"])
    print(f'\n  Mean AUC: {results["mean_auc"]:.3f} +/- {results["std_auc"]:.3f}')
    return results
