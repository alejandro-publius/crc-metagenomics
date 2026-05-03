"""Bootstrap 95% confidence intervals on LODO AUCs.

Reads saved prediction CSVs (preds_species_rf.csv, preds_joint_rf.csv,
preds_joint_xgb.csv) and computes per-cohort and pooled bootstrap CIs.

The pooled bootstrap is stratified by cohort: each iteration resamples
with replacement within each cohort separately, then concatenates the
resampled cohorts before computing AUC. This preserves the LODO cohort
structure (each fold's contribution is bounded by its true sample size)
and avoids cohort-imbalanced resamples that an i.i.d. pooled bootstrap
can produce.

Usage:
    python3 scripts/bootstrap_ci.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 10000
SEED = 42

def bootstrap_auc_iid(y_true, y_prob, n_boot=N_BOOT, seed=SEED):
    """I.i.d. bootstrap (used for per-cohort CIs)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def bootstrap_auc_stratified(df, n_boot=N_BOOT, seed=SEED):
    """Cohort-stratified bootstrap on the pooled prediction frame."""
    rng = np.random.RandomState(seed)
    cohort_to_idx = {c: np.where(df['cohort'].values == c)[0] for c in df['cohort'].unique()}
    y_true = df['y_true'].values
    y_prob = df['y_prob'].values
    aucs = []
    for _ in range(n_boot):
        sampled = []
        for c, idxs in cohort_to_idx.items():
            sampled.append(rng.choice(idxs, size=len(idxs), replace=True))
        idx = np.concatenate(sampled)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def process_preds(path, label):
    df = pd.read_csv(path)
    print(f'\n=== {label} ===')
    rows = []

    for cohort in sorted(df['cohort'].unique()):
        sub = df[df['cohort'] == cohort]
        yt = sub['y_true'].values
        yp = sub['y_prob'].values
        auc = roc_auc_score(yt, yp)
        lo, hi = bootstrap_auc_iid(yt, yp)
        print(f'  {cohort:25s}  AUC={auc:.3f}  [{lo:.3f}, {hi:.3f}]')
        rows.append({'model': label, 'cohort': cohort, 'auc': auc,
                     'ci_lo': lo, 'ci_hi': hi, 'n': len(sub)})

    yt_all = df['y_true'].values
    yp_all = df['y_prob'].values
    auc_all = roc_auc_score(yt_all, yp_all)
    lo, hi = bootstrap_auc_stratified(df)
    print(f'  {"pooled (stratified)":25s}  AUC={auc_all:.3f}  [{lo:.3f}, {hi:.3f}]')
    rows.append({'model': label, 'cohort': 'pooled', 'auc': auc_all,
                 'ci_lo': lo, 'ci_hi': hi, 'n': len(df)})
    return rows

def main():
    all_rows = []
    for path, label in [
        ('results/preds_species_rf.csv', 'species_rf'),
        ('results/preds_joint_rf.csv', 'joint_rf'),
        ('results/preds_joint_xgb.csv', 'joint_xgb'),
    ]:
        if os.path.exists(path):
            all_rows.extend(process_preds(path, label))
        else:
            print(f'WARNING: {path} not found, skipping')

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(all_rows).to_csv('results/bootstrap_ci.csv', index=False)
    print('\nSaved results/bootstrap_ci.csv')

if __name__ == '__main__':
    main()
