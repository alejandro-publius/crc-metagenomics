"""Paired statistical comparison of model AUCs across LODO folds.
Reads results CSVs (no hardcoded values) and reports paired t-test,
Wilcoxon signed-rank, and bootstrap 95% CIs on AUC differences."""
import pandas as pd
import numpy as np
import os
from scipy import stats

bl = pd.read_csv('results/baseline_results.csv').sort_values('cohort').reset_index(drop=True)
joint = pd.read_csv('results/joint_results.csv').sort_values('cohort').reset_index(drop=True)

assert (bl['cohort'].values == joint['cohort'].values).all(), 'Cohort order mismatch'

cohorts = bl['cohort'].tolist()
baseline = bl['auc'].values
joint_rf = joint['rf_auc'].values
joint_xgb = joint['xgb_auc'].values

def boot_ci(diffs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])

def compare(a, b, label):
    diffs = a - b
    t, p = stats.ttest_rel(a, b)
    w, pw = stats.wilcoxon(a, b)
    ci_lo, ci_hi = boot_ci(diffs)
    print(f'\n=== {label} ===')
    print(f'  Mean A: {a.mean():.4f}   Mean B: {b.mean():.4f}')
    print(f'  Mean diff (A - B): {diffs.mean():+.4f}  [95% CI: {ci_lo:+.4f}, {ci_hi:+.4f}]')
    print(f'  Paired t-test:  t={t:+.3f}, p={p:.4f}')
    print(f'  Wilcoxon:       W={w:.1f}, p={pw:.4f}')
    return {
        'comparison': label,
        'mean_a': a.mean(), 'mean_b': b.mean(),
        'mean_diff': diffs.mean(), 'ci_low': ci_lo, 'ci_high': ci_hi,
        't_stat': t, 't_pvalue': p,
        'wilcoxon_stat': w, 'wilcoxon_pvalue': pw,
        'n_folds': len(a)
    }

print('Cohorts (in order):', cohorts)
print('n folds:', len(cohorts))

rows = [
    compare(baseline, joint_rf,  'Species RF vs Joint RF'),
    compare(baseline, joint_xgb, 'Species RF vs Joint XGB'),
    compare(joint_xgb, joint_rf, 'Joint XGB vs Joint RF'),
]

print('\n=== Per-cohort differences ===')
for i, c in enumerate(cohorts):
    print(f'  {c:20s}  spRF={baseline[i]:.3f}  jRF={joint_rf[i]:.3f}  jXGB={joint_xgb[i]:.3f}')

os.makedirs('results', exist_ok=True)
pd.DataFrame(rows).to_csv('results/model_comparison.csv', index=False)
print('\nSaved results/model_comparison.csv')

print('\nNote: n=7 paired tests have low power. Bootstrap CIs that include 0')
print('mean the difference is not robustly distinguishable from zero.')
print('DeLong test on ROC curves would require saving raw prediction')
print('probabilities from each LODO fold, which is a separate refactor.')
