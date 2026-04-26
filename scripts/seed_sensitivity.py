"""Random seed sensitivity analysis for species-only RF LODO.

Runs the baseline classifier at multiple seeds to confirm AUC stability.

Usage:
    python3 scripts/seed_sensitivity.py
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv

SEEDS = [0, 1, 2, 42, 100]

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    mg = md.merge(sp, on='sample_id', how='inner')
    fc = [c for c in sp.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta = mg.loc[mask, ['sample_id', 'study_name', 'study_condition', 'label']].reset_index(drop=True)

    rows = []
    for seed in SEEDS:
        print(f'\n=== seed={seed} ===')
        def make_rf(s=seed):
            return RandomForestClassifier(n_estimators=500, max_features='sqrt',
                min_samples_leaf=5, n_jobs=-1, random_state=s, class_weight='balanced')
        res = run_lodo_cv(make_rf, X, y, meta)
        rows.append({'seed': seed, 'mean_auc': res['mean_auc'], 'std_auc': res['std_auc']})

    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/seed_sensitivity.csv', index=False)
    aucs = df['mean_auc'].values
    print(f'\nAcross {len(SEEDS)} seeds: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')
    print(f'Range: [{np.min(aucs):.4f}, {np.max(aucs):.4f}]')
    print('Saved results/seed_sensitivity.csv')

if __name__ == '__main__':
    main()
