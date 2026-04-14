import numpy as np
from scipy import stats

baseline = [0.838, 0.734, 0.801, 0.828, 0.730, 0.868, 0.821]
joint_rf = [0.790, 0.789, 0.714, 0.824, 0.716, 0.799, 0.823]
joint_xgb = [0.852, 0.736, 0.759, 0.815, 0.707, 0.808, 0.805]
cohorts = ['FengQ_2015','ThomasAM_2018a','ThomasAM_2018b','ThomasAM_2019_c','VogtmannE_2016','YuJ_2015','ZellerG_2014']

print('=== Paired t-test: Baseline RF vs Joint RF ===')
t, p = stats.ttest_rel(baseline, joint_rf)
print(f'  Baseline mean: {np.mean(baseline):.3f}, Joint RF mean: {np.mean(joint_rf):.3f}')
print(f'  t={t:.3f}, p={p:.4f}')
print(f'  {"Significant" if p < 0.05 else "Not significant"} at alpha=0.05')

print('\n=== Paired t-test: Baseline RF vs Joint XGBoost ===')
t2, p2 = stats.ttest_rel(baseline, joint_xgb)
print(f'  Baseline mean: {np.mean(baseline):.3f}, Joint XGB mean: {np.mean(joint_xgb):.3f}')
print(f'  t={t2:.3f}, p={p2:.4f}')
print(f'  {"Significant" if p2 < 0.05 else "Not significant"} at alpha=0.05')

print('\n=== Wilcoxon signed-rank (non-parametric) ===')
w, pw = stats.wilcoxon(baseline, joint_rf)
print(f'  Baseline vs Joint RF:  W={w:.1f}, p={pw:.4f}')
w2, pw2 = stats.wilcoxon(baseline, joint_xgb)
print(f'  Baseline vs Joint XGB: W={w2:.1f}, p={pw2:.4f}')

print('\n=== Per-cohort differences ===')
for i, c in enumerate(cohorts):
    d_rf = baseline[i] - joint_rf[i]
    d_xgb = baseline[i] - joint_xgb[i]
    print(f'  {c:25s}  RF diff={d_rf:+.3f}  XGB diff={d_xgb:+.3f}')
