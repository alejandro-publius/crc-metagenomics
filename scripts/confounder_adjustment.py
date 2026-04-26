"""Test whether age, sex, and BMI confound the species-only CRC classifier.

Two approaches:
  1. Direct inclusion: append covariates as extra features.
  2. Residualization: regress covariates out of each species feature,
     then classify on residuals.

Covariate imputation (median for continuous, mode for categorical) is
computed per fold using training data only to avoid leakage.

Usage:
    python3 scripts/confounder_adjustment.py
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import get_lodo_splits

def prepare_covariates_per_fold(md_train, md_test):
    """Impute covariates using train-only statistics."""
    cov_cols = []
    for col in ['age', 'BMI']:
        if col in md_train.columns:
            median_val = md_train[col].median()
            md_train = md_train.copy()
            md_test = md_test.copy()
            md_train[col] = md_train[col].fillna(median_val)
            md_test[col] = md_test[col].fillna(median_val)
            cov_cols.append(col)
    if 'gender' in md_train.columns:
        mode_val = md_train['gender'].mode()
        mode_val = mode_val.iloc[0] if len(mode_val) > 0 else 'female'
        md_train = md_train.copy()
        md_test = md_test.copy()
        md_train['gender_num'] = (md_train['gender'].fillna(mode_val) == 'male').astype(float)
        md_test['gender_num'] = (md_test['gender'].fillna(mode_val) == 'male').astype(float)
        cov_cols.append('gender_num')
    return md_train, md_test, cov_cols

def run_lodo_cv_with_covariates(model_fn, X, y, meta, md_full, mode='direct'):
    from sklearn.metrics import roc_auc_score
    results = {"cohort": [], "auc": []}
    for cohort, train_idx, test_idx in get_lodo_splits(meta):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        md_tr, md_te, cov_cols = prepare_covariates_per_fold(
            md_full.iloc[train_idx].copy(), md_full.iloc[test_idx].copy())

        if mode == 'direct':
            for c in cov_cols:
                X_tr[c] = md_tr[c].values
                X_te[c] = md_te[c].values
        elif mode == 'residualize':
            C_tr = md_tr[cov_cols].values
            C_te = md_te[cov_cols].values
            for col in X_tr.columns:
                lr = LinearRegression().fit(C_tr, X_tr[col].values)
                X_tr[col] = X_tr[col].values - lr.predict(C_tr)
                X_te[col] = X_te[col].values - lr.predict(C_te)

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        print(f'  {cohort:25s}  AUC={auc:.3f}')

    mean_auc = np.mean(results["auc"])
    print(f'  Mean AUC: {mean_auc:.3f}')
    results["mean_auc"] = mean_auc
    return results

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    mg = md.merge(sp, on='sample_id', how='inner')
    fc = [c for c in sp.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta = mg.loc[mask, ['sample_id', 'study_name', 'study_condition', 'label']].reset_index(drop=True)
    md_full = mg.loc[mask].reset_index(drop=True)

    def make_rf():
        return RandomForestClassifier(n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')

    def make_xgb():
        return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', n_jobs=-1)

    print('=== Direct inclusion (RF) ===')
    r1 = run_lodo_cv_with_covariates(make_rf, X.copy(), y, meta, md_full, mode='direct')
    print('\n=== Direct inclusion (XGB) ===')
    r2 = run_lodo_cv_with_covariates(make_xgb, X.copy(), y, meta, md_full, mode='direct')
    print('\n=== Residualized (RF) ===')
    r3 = run_lodo_cv_with_covariates(make_rf, X.copy(), y, meta, md_full, mode='residualize')
    print('\n=== Residualized (XGB) ===')
    r4 = run_lodo_cv_with_covariates(make_xgb, X.copy(), y, meta, md_full, mode='residualize')

    rows = [
        {'method': 'direct_rf', 'mean_auc': r1['mean_auc']},
        {'method': 'direct_xgb', 'mean_auc': r2['mean_auc']},
        {'method': 'residualized_rf', 'mean_auc': r3['mean_auc']},
        {'method': 'residualized_xgb', 'mean_auc': r4['mean_auc']},
    ]
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv('results/confounder_results.csv', index=False)
    print('\nSaved results/confounder_results.csv')

if __name__ == '__main__':
    main()
