import re, pandas as pd, numpy as np, os, sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv

PREVALENCE_THRESHOLD = 0.10
MEAN_THRESHOLD = 1e-6

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    pw_raw = pd.read_csv('data/raw/pathway_abundance.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')

    # Structural restriction (definitional, not data-derived): unstratified
    # pathways only. Safe to apply pre-fold; equivalent to scoping the
    # candidate feature space.
    pw_unstrat_cols = [c for c in pw_raw.columns if c != 'sample_id' and '|' not in c]
    pw = pw_raw[['sample_id'] + pw_unstrat_cols]
    print(f'Unstratified pathway candidates: {len(pw_unstrat_cols)}')

    joint = sp.merge(pw, on='sample_id', suffixes=('_sp','_pw'))
    mg = md.merge(joint, on='sample_id', how='inner')
    fc = [c for c in joint.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)

    def sanitize(c):
        return re.sub(r'[\[\]<>]', '_', c)

    X.columns = [sanitize(c) for c in X.columns]
    sp_feat_cols = [sanitize(c) for c in sp.columns if c != 'sample_id']
    pw_feat_cols = [sanitize(c) for c in pw_unstrat_cols]
    assert set(sp_feat_cols).isdisjoint(pw_feat_cols), 'species/pathway name collision'
    assert set(sp_feat_cols + pw_feat_cols).issubset(X.columns), 'missing feature columns post-merge'

    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta = mg.loc[mask, ['sample_id','study_name','study_condition','label']].reset_index(drop=True)
    print(f'Samples: {len(X)}, pre-fold features: {X.shape[1]} '
          f'(species={len(sp_feat_cols)}, pathway candidates={len(pw_feat_cols)})')

    def pathway_filter(X_train):
        """Train-fold prevalence/mean filter on pathway columns only.
        Species pass through (their upstream filter is a separate fix)."""
        Xpw = X_train[pw_feat_cols]
        prev = (Xpw > 0).mean(axis=0)
        ma = Xpw.mean(axis=0)
        keep_pw = [c for c in pw_feat_cols
                   if prev[c] >= PREVALENCE_THRESHOLD and ma[c] >= MEAN_THRESHOLD]
        return sp_feat_cols + keep_pw

    print('\n=== Joint Random Forest (per-fold pathway filter) ===')
    def make_rf():
        return RandomForestClassifier(n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
    rf_res = run_lodo_cv(make_rf, X, y, meta,
                         save_predictions_path="results/preds_joint_rf.csv",
                         feature_filter_fn=pathway_filter)

    print('\n=== Joint XGBoost (per-fold pathway filter) ===')
    def make_xgb():
        return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', n_jobs=-1)
    xgb_res = run_lodo_cv(make_xgb, X, y, meta,
                          save_predictions_path="results/preds_joint_xgb.csv",
                          feature_filter_fn=pathway_filter)

    bl = pd.read_csv('results/baseline_results.csv')
    print(f'\n  Species-only RF:  {bl["auc"].mean():.3f}')
    print(f'  Joint RF:         {rf_res["mean_auc"]:.3f}')
    print(f'  Joint XGBoost:    {xgb_res["mean_auc"]:.3f}')
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'cohort': rf_res['cohort'],
                  'rf_auc': rf_res['auc'],
                  'xgb_auc': xgb_res['auc'],
                  'rf_n_features': rf_res['n_features'],
                  'xgb_n_features': xgb_res['n_features'],
        }).to_csv('results/joint_results.csv', index=False)
    print('Saved results/joint_results.csv')

if __name__ == '__main__':
    main()
