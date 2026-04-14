import re, pandas as pd, numpy as np, os, shap
from sklearn.ensemble import RandomForestClassifier

def compute_shap(X, y, name):
    print(f'\n  {name}: {len(X)} samples, {X.shape[1]} features')
    rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    sv = shap.TreeExplainer(rf).shap_values(X)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:,:,1]
    elif isinstance(sv, list):
        sv = np.array(sv[1])
    ms = np.abs(sv).mean(axis=0).flatten()
    imp = pd.DataFrame({'feature': X.columns.tolist(), 'mean_abs_shap': ms}).sort_values('mean_abs_shap', ascending=False)
    for _, r in imp.head(10).iterrows():
        short = r.feature.split('|')[-1].replace('s__','').replace('_',' ') if '|' in r.feature else r.feature
        print(f'    {short:40s} {r.mean_abs_shap:.4f}')
    return imp

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    pw = pd.read_csv('data/processed/pathway_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    joint = sp.merge(pw, on='sample_id', suffixes=('_sp','_pw'))
    mg = md.merge(joint, on='sample_id', how='inner')
    fc = [c for c in joint.columns if c != 'sample_id']
    X_all = mg[fc].copy()
    X_all.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X_all.columns]
    os.makedirs('results', exist_ok=True)

    # Healthy vs Adenoma
    m1 = mg['study_condition'].isin(['control','adenoma'])
    X1 = X_all[m1].reset_index(drop=True)
    y1 = (mg.loc[m1,'study_condition']=='adenoma').astype(int).reset_index(drop=True)
    imp1 = compute_shap(X1, y1, 'Healthy vs Adenoma')
    imp1.to_csv('results/shap_healthy_vs_adenoma.csv', index=False)
    print('  Saved results/shap_healthy_vs_adenoma.csv')

    # Adenoma vs CRC (the transition)
    m2 = mg['study_condition'].isin(['adenoma','CRC'])
    X2 = X_all[m2].reset_index(drop=True)
    y2 = (mg.loc[m2,'study_condition']=='CRC').astype(int).reset_index(drop=True)
    imp2 = compute_shap(X2, y2, 'Adenoma vs CRC (transition)')
    imp2.to_csv('results/shap_adenoma_vs_crc.csv', index=False)
    print('  Saved results/shap_adenoma_vs_crc.csv')

if __name__ == '__main__':
    main()
