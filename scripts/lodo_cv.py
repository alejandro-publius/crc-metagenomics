import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
def get_lodo_splits(md, lc='label', cc='study_name'):
    for co in sorted(md[cc].unique()):
        te=(md[cc]==co)&(md[lc].isin([0,1]))
        tr=(md[cc]!=co)&(md[lc].isin([0,1]))
        tri=md[tr].index.tolist(); tei=md[te].index.tolist()
        if len(md.loc[tei,lc].unique())<2: continue
        yield co,tri,tei
def run_lodo_cv(mf,X,y,md,cc='study_name'):
    r={'cohort':[],'auc':[],'n_train':[],'n_test':[]}
    for co,tri,tei in get_lodo_splits(md,cc=cc):
        m=mf(); m.fit(X.iloc[tri],y.iloc[tri])
        yp=m.predict_proba(X.iloc[tei])[:,1]
        a=roc_auc_score(y.iloc[tei],yp)
        r['cohort'].append(co); r['auc'].append(a)
        r['n_train'].append(len(tri)); r['n_test'].append(len(tei))
        print(f'  {co:25s} AUC={a:.3f} (n={len(tei)})')
    r['mean_auc']=np.mean(r['auc']); r['std_auc']=np.std(r['auc'])
    print(f'  Mean AUC: {r[chr(34)+chr(34)]}' if False else '')
    print(f'  Mean AUC: ' + f'{np.mean(r[chr(39)+"auc"+chr(39)]):.3f} +/- {np.std(r[chr(39)+"auc"+chr(39)]):.3f}')
    return r
