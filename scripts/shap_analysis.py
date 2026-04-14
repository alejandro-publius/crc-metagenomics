import pandas as pd, numpy as np, os, shap
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
    md = pd.read_csv('data/processed/metadata_clean.csv')
    mg = md.merge(sp, on='sample_id', how='inner')
    fc = [c for c in sp.columns if c != 'sample_id']
    mask = mg['study_condition'].isin(['CRC','control'])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = (mg.loc[mask,'study_condition']=='CRC').astype(int).reset_index(drop=True)
    imp = compute_shap(X, y, 'CRC vs Healthy')
    os.makedirs('results', exist_ok=True)
    imp.to_csv('results/shap_crc_features.csv', index=False)
    print('\n  Saved results/shap_crc_features.csv')

if __name__ == '__main__':
    main()
