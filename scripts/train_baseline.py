import pandas as pd, numpy as np, os, sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0,os.path.dirname(__file__))
from lodo_cv import run_lodo_cv
sp=pd.read_csv('data/processed/species_filtered.csv')
md=pd.read_csv('data/processed/metadata_clean.csv')
mg=md.merge(sp,on='sample_id',how='inner')
fc=[c for c in sp.columns if c!='sample_id']
mask=mg['label'].isin([0,1])
X=mg.loc[mask,fc].reset_index(drop=True)
y=mg.loc[mask,'label'].reset_index(drop=True)
meta=mg.loc[mask,['sample_id','study_name','study_condition','label']].reset_index(drop=True)
print(f'Samples:{len(X)} CRC={int(y.sum())} ctrl={int((y==0).sum())}')
def make_rf(): return RandomForestClassifier(n_estimators=500,max_features='sqrt',min_samples_leaf=5,n_jobs=-1,random_state=42,class_weight='balanced')
print('Running LODO CV...')
res=run_lodo_cv(make_rf,X,y,meta)
os.makedirs('results',exist_ok=True)
pd.DataFrame({'cohort':res['cohort'],'auc':res['auc'],'n_train':res['n_train'],'n_test':res['n_test']}).to_csv('results/baseline_results.csv',index=False)
print('Saved to results/baseline_results.csv')
