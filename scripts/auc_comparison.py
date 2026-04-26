"""Paired statistical comparison of model AUCs across LODO folds.
Reads results CSVs (no hardcoded values) and reports paired t-test,
Wilcoxon signed-rank, bootstrap 95% CIs on per-cohort AUC differences,
and DeLong tests on pooled LODO ROC curves."""
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import norm

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


def _midrank(x):
    J = np.argsort(x, kind='mergesort')
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2

def delong_roc_test(y_true, y_prob_a, y_prob_b):
    """Two-tailed DeLong test for paired AUCs (Sun and Xu 2014 fast algorithm).
    Returns (auc_a, auc_b, z, p)."""
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError('Need both classes present')
    aucs = []
    v01s = []
    v10s = []
    for y_prob in [y_prob_a, y_prob_b]:
        y_prob = np.asarray(y_prob, dtype=float)
        x_pos = y_prob[pos]
        x_neg = y_prob[neg]
        tx = _midrank(x_pos)
        ty = _midrank(x_neg)
        tz = _midrank(np.concatenate([x_pos, x_neg]))
        auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        aucs.append(auc)
        v01s.append(v01)
        v10s.append(v10)
    auc_a, auc_b = aucs
    S01 = np.cov(np.vstack(v01s))
    S10 = np.cov(np.vstack(v10s))
    S = S01 / m + S10 / n
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


pred_files = {
    'species_rf': 'results/preds_species_rf.csv',
    'joint_rf':   'results/preds_joint_rf.csv',
    'joint_xgb':  'results/preds_joint_xgb.csv',
}
missing = [p for p in pred_files.values() if not os.path.exists(p)]
if missing:
    print(f'\nDeLong skipped (missing prediction files: {missing}).')
    print('Re-run train_baseline and train_joint to generate them.')
else:
    print('\n=== DeLong tests on pooled LODO predictions ===')
    preds = {k: pd.read_csv(v).sort_values('sample_id').reset_index(drop=True)
             for k, v in pred_files.items()}
    ref = preds['species_rf']['sample_id'].values
    for k in preds:
        assert (preds[k]['sample_id'].values == ref).all(), f'{k} sample order mismatch'
        assert (preds[k]['y_true'].values == preds['species_rf']['y_true'].values).all(), f'{k} y_true mismatch'
    y_true = preds['species_rf']['y_true'].values
    n_total = len(y_true)
    n_pos = int(y_true.sum())
    print(f'  Pooled n={n_total} (CRC={n_pos}, control={n_total - n_pos})')
    delong_rows = []
    for a, b in [('species_rf','joint_rf'), ('species_rf','joint_xgb'), ('joint_xgb','joint_rf')]:
        au_a, au_b, z, p = delong_roc_test(y_true, preds[a]['y_prob'].values, preds[b]['y_prob'].values)
        print(f'  {a:12s} vs {b:12s}  AUC {au_a:.3f} vs {au_b:.3f}  diff={au_a-au_b:+.3f}  z={z:+.3f}  p={p:.4f}')
        delong_rows.append({'model_a': a, 'model_b': b, 'auc_a': au_a, 'auc_b': au_b,
                            'auc_diff': au_a - au_b, 'z': z, 'p_value': p, 'n_samples': n_total})
    pd.DataFrame(delong_rows).to_csv('results/delong_results.csv', index=False)
    print('Saved results/delong_results.csv')
    print('\nNote: DeLong is run on pooled LODO predictions (each sample contributes')
    print('its single held-out cohort prediction). This tests overall classifier')
    print('performance and complements the per-cohort paired tests above.')
