"""Per-fold ComBat batch correction on species features under LODO CV.

Applies ComBat (Johnson et al. 2007) to correct for study-level batch
effects within each LODO fold, using only training data to estimate
parameters.

Requires: pip install pycombat

Usage:
    python3 scripts/batch_correction.py
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import get_lodo_splits

def run_lodo_cv_combat():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    mg = md.merge(sp, on='sample_id', how='inner')
    fc = [c for c in sp.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta = mg.loc[mask, ['sample_id', 'study_name', 'study_condition', 'label']].reset_index(drop=True)

    try:
        from pycombat import pycombat
    except ImportError:
        print('ERROR: pycombat not installed. Run: pip install pycombat')
        sys.exit(1)

    print('=== Species RF with per-fold ComBat ===')
    results = {"cohort": [], "auc": []}
    for cohort, train_idx, test_idx in get_lodo_splits(meta):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        batch_tr = meta.iloc[train_idx]['study_name'].values

        # ComBat expects features x samples
        try:
            corrected_tr = pycombat(X_tr.T, batch_tr)
            X_tr_c = corrected_tr.T
        except Exception as e:
            print(f'  {cohort}: ComBat failed on train ({e}), using uncorrected')
            X_tr_c = X_tr

        # Test fold is single cohort, no batch to correct; use raw
        X_te_c = X_te

        model = RandomForestClassifier(n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
        model.fit(X_tr_c, y.iloc[train_idx])
        y_prob = model.predict_proba(X_te_c)[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        print(f'  {cohort:25s}  AUC={auc:.3f}')

    mean_auc = np.mean(results["auc"])
    print(f'\n  Mean AUC (ComBat): {mean_auc:.3f}')

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/combat_results.csv', index=False)
    print('Saved results/combat_results.csv')

if __name__ == '__main__':
    run_lodo_cv_combat()
