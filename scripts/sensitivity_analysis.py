"""Sweep unstratified pathway filter thresholds and report Joint RF mean
per-cohort LODO AUC across a 20-cell grid (4 prevalence x 5 mean).

The filter is applied PER FOLD using only training-cohort samples, via
the feature_filter_fn hook in run_lodo_cv. Earlier versions of this
script computed prevalence/mean on all samples before LODO, which
leaked test-fold information into the filter. This is the same per-fold
pattern used by train_joint.py for the headline run.

Reads data/raw/pathway_abundance.csv (output of merge_pathways.py).
"""
import pandas as pd, numpy as np, os, sys, re
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv


def sanitize(c):
    return re.sub(r'[\[\]<>]', '_', str(c))


def main():
    pw_raw = pd.read_csv('data/raw/pathway_abundance.csv')
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')

    unstrat_cols = [c for c in pw_raw.columns if c != 'sample_id' and '|' not in c]
    print(f'Unstratified pathway candidates in raw: {len(unstrat_cols)}')
    pw = pw_raw[['sample_id'] + unstrat_cols]

    # Build the joint matrix once; filtering happens per fold below.
    joint = sp.merge(pw, on='sample_id', suffixes=('_sp', '_pw'))
    mg = md.merge(joint, on='sample_id', how='inner')
    fc = [c for c in joint.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    X.columns = [sanitize(c) for c in X.columns]
    sp_feat_cols = [sanitize(c) for c in sp.columns if c != 'sample_id']
    pw_feat_cols = [sanitize(c) for c in unstrat_cols]
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta = mg.loc[mask, ['sample_id', 'study_name', 'study_condition', 'label']].reset_index(drop=True)

    prev_grid = [0.05, 0.10, 0.15, 0.20]
    mean_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    rows = []

    for prev_t in prev_grid:
        for mean_t in mean_grid:
            def make_filter(prev_t=prev_t, mean_t=mean_t):
                def pathway_filter(X_train):
                    Xpw = X_train[pw_feat_cols]
                    prev = (Xpw > 0).mean(axis=0)
                    ma = Xpw.mean(axis=0)
                    keep_pw = [c for c in pw_feat_cols
                               if prev[c] >= prev_t and ma[c] >= mean_t]
                    return sp_feat_cols + keep_pw
                return pathway_filter

            print(f'\n--- prev>={prev_t}, mean>={mean_t} (per-fold filter) ---')
            def make_rf():
                return RandomForestClassifier(n_estimators=500, max_features='sqrt',
                    min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
            res = run_lodo_cv(make_rf, X, y, meta, feature_filter_fn=make_filter())
            mean_n_pw = float(np.mean([n - len(sp_feat_cols) for n in res['n_features']]))
            rows.append({'prev_threshold': prev_t, 'mean_threshold': mean_t,
                         'n_pathways_mean': mean_n_pw,
                         'n_features_mean': float(np.mean(res['n_features'])),
                         'mean_auc': res['mean_auc'], 'std_auc': res['std_auc']})

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv('results/sensitivity_thresholds.csv', index=False)
    aucs = [r['mean_auc'] for r in rows]
    print(f'\nSaved results/sensitivity_thresholds.csv')
    print(f'AUC range: {min(aucs):.3f} to {max(aucs):.3f}, spread {max(aucs)-min(aucs):.3f}')


if __name__ == '__main__':
    main()
