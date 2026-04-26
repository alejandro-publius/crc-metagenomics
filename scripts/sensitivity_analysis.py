"""Vary unstratified pathway filter thresholds and report Joint RF mean
per-cohort AUC across a 20-combination grid.
Reads data/raw/pathway_abundance.csv (output of merge_pathways.py)."""
import pandas as pd, numpy as np, os, sys, re
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv

def main():
    pw_raw = pd.read_csv('data/raw/pathway_abundance.csv')
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    unstrat_cols = [c for c in pw_raw.columns if c != 'sample_id' and '|' not in c]
    print(f'Unstratified pathway candidates in raw: {len(unstrat_cols)}')
    pw_unstrat = pw_raw[['sample_id'] + unstrat_cols]
    prev_grid = [0.05, 0.10, 0.15, 0.20]
    mean_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    rows = []
    for prev_t in prev_grid:
        for mean_t in mean_grid:
            X_pw = pw_unstrat.set_index('sample_id')
            prev = (X_pw > 0).mean(axis=0)
            ma = X_pw.mean(axis=0)
            keep = X_pw.columns[(prev >= prev_t) & (ma >= mean_t)].tolist()
            joint = sp.merge(pw_unstrat[['sample_id'] + keep], on='sample_id')
            mg = md.merge(joint, on='sample_id', how='inner')
            fc = [c for c in joint.columns if c != 'sample_id']
            mask = mg['label'].isin([0, 1])
            X = mg.loc[mask, fc].reset_index(drop=True)
            X.columns = [re.sub(r'[\[\]<>]', '_', str(c)) for c in X.columns]
            y = mg.loc[mask, 'label'].reset_index(drop=True)
            meta = mg.loc[mask, ['sample_id','study_name','study_condition','label']].reset_index(drop=True)
            def make_rf():
                return RandomForestClassifier(n_estimators=500, max_features='sqrt',
                    min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
            print(f'\n--- prev>={prev_t}, mean>={mean_t}: {len(keep)} pathways, {X.shape[1]} features ---')
            res = run_lodo_cv(make_rf, X, y, meta)
            rows.append({'prev_threshold': prev_t, 'mean_threshold': mean_t,
                         'n_pathways': len(keep), 'n_features_total': X.shape[1],
                         'mean_auc': res['mean_auc'], 'std_auc': res['std_auc']})
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv('results/sensitivity_thresholds.csv', index=False)
    aucs = [r['mean_auc'] for r in rows]
    print(f'\nSaved results/sensitivity_thresholds.csv')
    print(f'AUC range: {min(aucs):.3f} to {max(aucs):.3f}, spread {max(aucs)-min(aucs):.3f}')

if __name__ == '__main__':
    main()
