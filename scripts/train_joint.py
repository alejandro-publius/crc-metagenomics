import re, pandas as pd, numpy as np, os, sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    pw = pd.read_csv('data/processed/pathway_unstratified.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    joint = sp.merge(pw, on='sample_id', suffixes=('_sp','_pw'))
    mg = md.merge(joint, on='sample_id', how='inner')
    fc = [c for c in joint.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    X.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X.columns]
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta = mg.loc[mask, ['sample_id','study_name','study_condition','label']].reset_index(drop=True)
    print(f'Samples: {len(X)}, Features: {X.shape[1]}')
    print('\n=== Joint Random Forest ===')
    def make_rf():
        return RandomForestClassifier(n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
    rf_res = run_lodo_cv(make_rf, X, y, meta)
    print('\n=== Joint XGBoost ===')
    def make_xgb():
        return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', n_jobs=-1)
    xgb_res = run_lodo_cv(make_xgb, X, y, meta)
    bl = pd.read_csv('results/baseline_results.csv')
    print(f'\n  Species-only RF:  {bl["auc"].mean():.3f}')
    print(f'  Joint RF:         {rf_res["mean_auc"]:.3f}')
    print(f'  Joint XGBoost:    {xgb_res["mean_auc"]:.3f}')
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'cohort': rf_res['cohort'], 'rf_auc': rf_res['auc'],
        'xgb_auc': xgb_res['auc']}).to_csv('results/joint_results.csv', index=False)
    print('Saved results/joint_results.csv')

if __name__ == '__main__':
    main()
