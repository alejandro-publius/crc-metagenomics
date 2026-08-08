"""Transductive per-fold ComBat pilot on species features under LODO CV.

Applies ComBat (Johnson et al. 2007) to correct for study-level batch
effects within each LODO fold. ComBat is fit jointly on the train and
test feature matrices using batch labels (study_name). Although cancer labels
are not used, the held-out cohort's feature distribution participates in the
transformation. This is therefore an unlabeled target-adaptation experiment,
not a strictly inductive LODO evaluation. Both train and test end up in the same corrected
feature space, which is required for the trained classifier to make
sensible predictions on the held-out cohort. Earlier versions of this
script applied ComBat only to training data and tested on uncorrected
features, leaving train and test in different spaces.

Requires: pip install combat

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
    meta_cols = ['sample_id', 'study_name', 'study_condition', 'label']
    if 'country' in mg.columns:
        meta_cols.append('country')
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)
    country_col = 'country' if 'country' in meta.columns else None

    try:
        from combat.pycombat import pycombat
    except ImportError:
        print('ERROR: combat not installed. Run: pip install combat')
        sys.exit(1)

    print('=== Species RF with per-fold ComBat (country-aware LODO) ===')
    print('ComBat is fit jointly on train+test with batch=study_name; class')
    print('labels are not used, but target features are used: transductive setting.')
    results = {"cohort": [], "auc": []}
    for cohort, train_idx, test_idx, excluded in get_lodo_splits(
            meta, country_col=country_col):
        idx_all = np.concatenate([np.asarray(train_idx), np.asarray(test_idx)])
        X_all = X.iloc[idx_all].copy()
        batch_all = meta.iloc[idx_all]['study_name'].values

        # ComBat expects features x samples; pass batch labels as a Series.
        try:
            X_corr_all = pycombat(X_all.T, pd.Series(batch_all)).T
        except Exception as e:
            print(f'  {cohort}: ComBat failed ({e}); skipping fold')
            continue

        n_train = len(train_idx)
        X_tr_c = X_corr_all.iloc[:n_train]
        X_te_c = X_corr_all.iloc[n_train:]
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        model = RandomForestClassifier(n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
        model.fit(X_tr_c, y_tr)
        y_prob = model.predict_proba(X_te_c)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        print(f'  {cohort:25s}  AUC={auc:.3f}')

    if not results["auc"]:
        print('No folds completed.')
        return
    mean_auc = np.mean(results["auc"])
    print(f'\n  Mean AUC (ComBat): {mean_auc:.3f}')

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/combat_results.csv', index=False)
    print('Saved results/combat_results.csv')

if __name__ == '__main__':
    run_lodo_cv_combat()
