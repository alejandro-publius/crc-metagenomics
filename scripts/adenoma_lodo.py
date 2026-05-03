"""Cross-cohort LODO validation for adenoma classification.

Runs leave-one-cohort-out across the 3 cohorts with adenoma samples
(FengQ_2015, ZellerG_2014, ThomasAM_2018a). Computes scale_pos_weight
per fold to handle class imbalance correctly.

Usage:
    python3 scripts/adenoma_lodo.py
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import get_lodo_splits

ADENOMA_COHORTS = ['FengQ_2015', 'ZellerG_2014', 'ThomasAM_2018a']

def make_rf(y_train):
    return RandomForestClassifier(n_estimators=500, max_features='sqrt',
        min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')

def make_xgb(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)
    return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', scale_pos_weight=spw, n_jobs=-1)

def run_adenoma_lodo(task_name, label_map, model_fn_factory, X, md):
    """LODO across adenoma cohorts for a given binary task."""
    sub = md[md['study_condition'].isin(label_map.keys()) &
             md['study_name'].isin(ADENOMA_COHORTS)].copy()
    sub['bin_label'] = sub['study_condition'].map(label_map)
    sub = sub.reset_index(drop=True)

    print(f'\n=== {task_name} ===')
    print(f'Total samples: {len(sub)}')
    for cohort in ADENOMA_COHORTS:
        cs = sub[sub['study_name'] == cohort]
        if len(cs) > 0:
            counts = cs['bin_label'].value_counts().to_dict()
            print(f'  {cohort}: {dict(counts)}')

    results = {"cohort": [], "auc": []}
    for cohort in sorted(sub['study_name'].unique()):
        test_mask = sub['study_name'] == cohort
        train_mask = ~test_mask
        if sub.loc[test_mask, 'bin_label'].nunique() < 2:
            print(f'  Skipping {cohort}: only one class in test set')
            continue

        train_idx = sub[train_mask].index.tolist()
        test_idx = sub[test_mask].index.tolist()

        sids_tr = sub.loc[train_idx, 'sample_id']
        sids_te = sub.loc[test_idx, 'sample_id']
        X_tr = X[X.index.isin(sids_tr.index)].iloc[:, :]
        X_te = X[X.index.isin(sids_te.index)].iloc[:, :]

        # Align by sample_id
        X_tr = X.loc[train_idx]
        X_te = X.loc[test_idx]
        y_tr = sub.loc[train_idx, 'bin_label']
        y_te = sub.loc[test_idx, 'bin_label']

        model = model_fn_factory(y_tr)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        print(f'  {cohort:25s}  AUC={auc:.3f}  (n_test={len(test_idx)})')

    if results["auc"]:
        mean_auc = np.mean(results["auc"])
        print(f'  Mean LODO AUC: {mean_auc:.3f}')
        results["mean_auc"] = mean_auc
    return results

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    mg = md.merge(sp, on='sample_id', how='inner')
    fc = [c for c in sp.columns if c != 'sample_id']

    # Build feature matrix aligned to metadata
    X_all = mg[fc].reset_index(drop=True)
    md_all = mg[['sample_id', 'study_name', 'study_condition']].reset_index(drop=True)

    h_vs_a = {'control': 0, 'adenoma': 1}
    a_vs_c = {'adenoma': 0, 'CRC': 1}

    print('--- Healthy vs Adenoma ---')
    r1_rf = run_adenoma_lodo('H-vs-A RF', h_vs_a, make_rf, X_all, md_all)
    r1_xgb = run_adenoma_lodo('H-vs-A XGB', h_vs_a, make_xgb, X_all, md_all)

    print('\n--- Adenoma vs CRC ---')
    r2_rf = run_adenoma_lodo('A-vs-CRC RF', a_vs_c, make_rf, X_all, md_all)
    r2_xgb = run_adenoma_lodo('A-vs-CRC XGB', a_vs_c, make_xgb, X_all, md_all)

    rows = []
    for name, res in [('h_vs_a_rf', r1_rf), ('h_vs_a_xgb', r1_xgb),
                      ('a_vs_crc_rf', r2_rf), ('a_vs_crc_xgb', r2_xgb)]:
        if 'mean_auc' in res:
            rows.append({'task': name, 'mean_lodo_auc': res['mean_auc'],
                         'n_folds': len(res['auc'])})

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv('results/adenoma_lodo_results.csv', index=False)
    print('\nSaved results/adenoma_lodo_results.csv')

if __name__ == '__main__':
    main()
