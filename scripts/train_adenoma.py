import re, pandas as pd, numpy as np, os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    pw = pd.read_csv('data/processed/pathway_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    joint = sp.merge(pw, on='sample_id', suffixes=('_sp','_pw'))
    mg = md.merge(joint, on='sample_id', how='inner')
    fc = [c for c in joint.columns if c != 'sample_id']
    X_all = mg[fc].copy()
    X_all.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X_all.columns]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def run_binary(X, y, name):
        rf_a, xgb_a = [], []
        for tr, te in skf.split(X, y):
            rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
            rf.fit(X.iloc[tr], y.iloc[tr])
            rf_a.append(roc_auc_score(y.iloc[te], rf.predict_proba(X.iloc[te])[:,1]))
            n_pos = int(y.iloc[tr].sum()); n_neg = int((y.iloc[tr] == 0).sum()); spw = n_neg / n_pos if n_pos > 0 else 1.0; xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=-1, scale_pos_weight=spw)
            xgb.fit(X.iloc[tr], y.iloc[tr])
            xgb_a.append(roc_auc_score(y.iloc[te], xgb.predict_proba(X.iloc[te])[:,1]))
        print(f'  {name}\n    RF:      {np.mean(rf_a):.3f} +/- {np.std(rf_a):.3f}\n    XGBoost: {np.mean(xgb_a):.3f} +/- {np.std(xgb_a):.3f}')
        return np.mean(rf_a), np.mean(xgb_a)

    m1 = mg['study_condition'].isin(['control','adenoma'])
    X1 = X_all[m1].reset_index(drop=True)
    y1 = (mg.loc[m1,'study_condition']=='adenoma').astype(int).reset_index(drop=True)
    print(f'\nHealthy vs Adenoma: {int((y1==0).sum())} ctrl, {int(y1.sum())} adenoma')
    r1_rf, r1_xgb = run_binary(X1, y1, 'Healthy vs Adenoma')

    m2 = mg['study_condition'].isin(['adenoma','CRC'])
    X2 = X_all[m2].reset_index(drop=True)
    y2 = (mg.loc[m2,'study_condition']=='CRC').astype(int).reset_index(drop=True)
    print(f'\nAdenoma vs CRC: {int((y2==0).sum())} adenoma, {int(y2.sum())} CRC')
    r2_rf, r2_xgb = run_binary(X2, y2, 'Adenoma vs CRC')

    m3 = mg['study_condition'].isin(['control','adenoma','CRC'])
    X3 = X_all[m3].reset_index(drop=True)
    y3 = mg.loc[m3,'study_condition'].map({'control':0,'adenoma':1,'CRC':2}).reset_index(drop=True)
    accs = []
    for tr, te in skf.split(X3, y3):
        rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
        rf.fit(X3.iloc[tr], y3.iloc[tr])
        accs.append(accuracy_score(y3.iloc[te], rf.predict(X3.iloc[te])))
    print(f'\n3-class accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}')

    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'task':['healthy_vs_adenoma','adenoma_vs_crc'],
        'rf_auc':[r1_rf, r2_rf], 'xgb_auc':[r1_xgb, r2_xgb]
    }).to_csv('results/adenoma_results.csv', index=False)
    print('Saved results/adenoma_results.csv')

if __name__ == '__main__':
    main()
