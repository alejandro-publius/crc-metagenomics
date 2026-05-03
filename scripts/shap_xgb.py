"""SHAP on joint XGBoost for all three tasks. Compare to RF SHAP.

Scope: SHAP values come from a model fit on the entire labeled dataset
(or per-task subset). They describe how the fully-fit classifier uses each
feature; they do not measure cross-cohort generalization. Generalization
claims come from the LODO scripts; SHAP here is for feature importance
on the fitted model only.
"""
import re, pandas as pd, numpy as np, os, shap
from xgboost import XGBClassifier

def compute_shap(X, y, name):
    print(f'\n  {name}: {len(X)} samples, {X.shape[1]} features')
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric='logloss', n_jobs=-1,
                        scale_pos_weight=spw)
    xgb.fit(X, y)
    sv = shap.TreeExplainer(xgb).shap_values(X)
    if isinstance(sv, list):
        sv = np.array(sv[1] if len(sv) > 1 else sv[0])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    ms = np.abs(sv).mean(axis=0).flatten()
    imp = pd.DataFrame({'feature': X.columns.tolist(), 'mean_abs_shap': ms}).sort_values('mean_abs_shap', ascending=False)
    for _, r in imp.head(10).iterrows():
        short = r.feature.split('|')[-1].replace('s__','').replace('_',' ') if '|' in r.feature else r.feature
        print(f'    {short:50s} {r.mean_abs_shap:.4f}')
    return imp

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    pw = pd.read_csv('data/processed/pathway_unstratified.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    joint = sp.merge(pw, on='sample_id', suffixes=('_sp', '_pw'))
    mg = md.merge(joint, on='sample_id', how='inner')
    fc = [c for c in joint.columns if c != 'sample_id']
    X_all = mg[fc].copy()
    X_all.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X_all.columns]

    os.makedirs('results', exist_ok=True)

    # CRC vs control (joint XGB)
    m = mg['study_condition'].isin(['CRC', 'control'])
    X = X_all[m].reset_index(drop=True)
    y = (mg.loc[m, 'study_condition'] == 'CRC').astype(int).reset_index(drop=True)
    imp = compute_shap(X, y, 'CRC vs control (joint XGB)')
    imp.to_csv('results/shap_crc_xgb.csv', index=False)

    # Healthy vs adenoma (joint XGB)
    m = mg['study_condition'].isin(['control', 'adenoma'])
    X = X_all[m].reset_index(drop=True)
    y = (mg.loc[m, 'study_condition'] == 'adenoma').astype(int).reset_index(drop=True)
    imp = compute_shap(X, y, 'Healthy vs adenoma (joint XGB)')
    imp.to_csv('results/shap_healthy_vs_adenoma_xgb.csv', index=False)

    # Adenoma vs CRC (joint XGB)
    m = mg['study_condition'].isin(['adenoma', 'CRC'])
    X = X_all[m].reset_index(drop=True)
    y = (mg.loc[m, 'study_condition'] == 'CRC').astype(int).reset_index(drop=True)
    imp = compute_shap(X, y, 'Adenoma vs CRC (joint XGB)')
    imp.to_csv('results/shap_adenoma_vs_crc_xgb.csv', index=False)

    # Concordance check: top 20 RF vs top 20 XGB for each task
    print('\n' + '=' * 60)
    print('TOP-20 OVERLAP: RF SHAP vs XGB SHAP')
    print('=' * 60)
    for rf_path, xgb_path, label in [
        ('results/shap_crc_features.csv', 'results/shap_crc_xgb.csv', 'CRC vs control'),
        ('results/shap_healthy_vs_adenoma.csv', 'results/shap_healthy_vs_adenoma_xgb.csv', 'Healthy vs adenoma'),
        ('results/shap_adenoma_vs_crc.csv', 'results/shap_adenoma_vs_crc_xgb.csv', 'Adenoma vs CRC'),
    ]:
        if not os.path.exists(rf_path):
            print(f'  {label}: RF file missing, skipping')
            continue
        rf_top = set(pd.read_csv(rf_path).head(20)['feature'])
        xgb_top = set(pd.read_csv(xgb_path).head(20)['feature'])
        overlap = len(rf_top & xgb_top)
        print(f'  {label:25s}  {overlap}/20 features overlap in top 20')

if __name__ == '__main__':
    main()
