"""Generate publication-ready supplementary tables (S1-S10) from existing
results CSVs. All outputs land in results/supplementary/.

Reads only existing inputs; does not refit models. Numbers reflect the
10-cohort dataset with HanniganGD_2017 excluded.
"""
import pandas as pd
import numpy as np
import os

OUT = 'results/supplementary'
os.makedirs(OUT, exist_ok=True)

md = pd.read_csv('data/processed/metadata_clean.csv')


def s1_cohort_overview():
    rows = []
    for cohort, sub in md.groupby('study_name'):
        n = len(sub)
        n_crc = (sub['study_condition'] == 'CRC').sum()
        n_ctrl = (sub['study_condition'] == 'control').sum()
        n_ade = (sub['study_condition'] == 'adenoma').sum()
        country = sub['country'].mode().iat[0]
        platform = sub['sequencing_platform'].mode().iat[0]
        reads_med = sub['number_reads'].median() / 1e6
        reads_min = sub['number_reads'].min() / 1e6
        reads_max = sub['number_reads'].max() / 1e6
        age_mean = sub['age'].mean()
        age_sd = sub['age'].std()
        bmi_mean = sub['BMI'].mean()
        bmi_sd = sub['BMI'].std()
        pct_female = (sub['gender'].str.lower() == 'female').mean() * 100
        rows.append({
            'Cohort': cohort, 'Country': country, 'N_total': n,
            'N_CRC': n_crc, 'N_control': n_ctrl, 'N_adenoma': n_ade,
            'Sequencing_platform': platform,
            'Reads_median_Mreads': round(reads_med, 1),
            'Reads_min_Mreads': round(reads_min, 1),
            'Reads_max_Mreads': round(reads_max, 1),
            'Age_mean': round(age_mean, 1), 'Age_SD': round(age_sd, 1),
            'BMI_mean': round(bmi_mean, 1), 'BMI_SD': round(bmi_sd, 1),
            'Pct_female': round(pct_female, 1),
        })
    df = pd.DataFrame(rows).sort_values('Cohort')
    df.to_csv(f'{OUT}/S1_cohort_overview.csv', index=False)
    return df


def s2_per_fold_aucs():
    base = pd.read_csv('results/baseline_results.csv').rename(
        columns={'auc': 'species_rf_auc'})
    joint = pd.read_csv('results/joint_results.csv').rename(
        columns={'rf_auc': 'joint_rf_auc', 'xgb_auc': 'joint_xgb_auc'})
    merged = base.merge(
        joint[['cohort', 'joint_rf_auc', 'joint_xgb_auc']], on='cohort')
    if os.path.exists('results/bio_pathway_results.csv'):
        bio = pd.read_csv('results/bio_pathway_results.csv').rename(
            columns={'bio_pw_auc': 'bio_pathway_rf_auc'})
        merged = merged.merge(
            bio[['cohort', 'bio_pathway_rf_auc']], on='cohort', how='left')
    if os.path.exists('results/combat_results.csv'):
        cb = pd.read_csv('results/combat_results.csv').rename(
            columns={'auc': 'combat_species_rf_auc'})
        merged = merged.merge(cb, on='cohort', how='left')
    means = {'cohort': 'MEAN'}
    for c in merged.columns:
        if c == 'cohort':
            continue
        if pd.api.types.is_numeric_dtype(merged[c]):
            means[c] = round(merged[c].mean(), 4)
    merged = pd.concat([merged, pd.DataFrame([means])], ignore_index=True)
    for c in merged.columns:
        if c not in ('cohort',) and pd.api.types.is_numeric_dtype(merged[c]):
            merged[c] = merged[c].round(4)
    merged.to_csv(f'{OUT}/S2_per_fold_aucs.csv', index=False)
    return merged


def _short_species(name):
    name = str(name)
    if 's__' in name:
        return name.split('s__')[-1].replace('_', ' ')
    return name


def s3_top_shap():
    files = {
        'CRC_vs_control_RF': 'results/shap_crc_features.csv',
        'CRC_vs_control_XGB': 'results/shap_crc_xgb.csv',
        'H_vs_A_RF': 'results/shap_healthy_vs_adenoma.csv',
        'H_vs_A_XGB': 'results/shap_healthy_vs_adenoma_xgb.csv',
        'A_vs_CRC_RF': 'results/shap_adenoma_vs_crc.csv',
        'A_vs_CRC_XGB': 'results/shap_adenoma_vs_crc_xgb.csv',
    }
    tops = []
    for task, path in files.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        score_col = [c for c in df.columns if c != 'feature'][0]
        top = df.sort_values(score_col, ascending=False).head(20).copy()
        top['rank'] = range(1, len(top) + 1)
        top['task'] = task
        top['species'] = top['feature'].map(_short_species)
        top = top[['task', 'rank', 'species', 'feature', score_col]].rename(
            columns={score_col: 'mean_abs_shap'})
        tops.append(top)
    out = pd.concat(tops, ignore_index=True)
    out['mean_abs_shap'] = out['mean_abs_shap'].round(5)
    out.to_csv(f'{OUT}/S3_top_shap_features.csv', index=False)
    return out


def s4_bootstrap_cis():
    bs = pd.read_csv('results/bootstrap_ci.csv').copy()
    for c in ('auc', 'ci_lo', 'ci_hi'):
        if c in bs.columns:
            bs[c] = bs[c].round(4)
    bs.to_csv(f'{OUT}/S4_bootstrap_ci.csv', index=False)
    return bs


def s5_sensitivity_grid():
    sens = pd.read_csv('results/sensitivity_thresholds.csv').copy()
    for c in ('n_pathways_mean', 'n_features_mean'):
        if c in sens.columns:
            sens[c] = sens[c].round(1)
    for c in ('mean_auc', 'std_auc'):
        if c in sens.columns:
            sens[c] = sens[c].round(4)
    sens.to_csv(f'{OUT}/S5_sensitivity_grid.csv', index=False)
    return sens


def s6_adenoma_lodo():
    ade = pd.read_csv('results/adenoma_lodo_results.csv').copy()
    if 'mean_lodo_auc' in ade.columns:
        ade['mean_lodo_auc'] = ade['mean_lodo_auc'].round(4)
    ade.to_csv(f'{OUT}/S6_adenoma_lodo.csv', index=False)
    return ade


def s7_confounder():
    cf = pd.read_csv('results/confounder_results.csv').copy()
    if 'mean_auc' in cf.columns:
        cf['mean_auc'] = cf['mean_auc'].round(4)
    cf.to_csv(f'{OUT}/S7_confounder_adjustment.csv', index=False)
    return cf


def s8_seed_sensitivity():
    seed = pd.read_csv('results/seed_sensitivity.csv').copy()
    for c in ('mean_auc', 'std_auc'):
        if c in seed.columns:
            seed[c] = seed[c].round(4)
    summary = pd.DataFrame([{
        'metric': 'across_seeds',
        'n_seeds': len(seed),
        'grand_mean': round(seed['mean_auc'].mean(), 4),
        'across_seed_std': round(seed['mean_auc'].std(), 4),
        'min': round(seed['mean_auc'].min(), 4),
        'max': round(seed['mean_auc'].max(), 4),
        'spread': round(seed['mean_auc'].max() - seed['mean_auc'].min(), 4),
    }])
    out_path = f'{OUT}/S8_seed_sensitivity.csv'
    with open(out_path, 'w') as f:
        seed.to_csv(f, index=False)
        f.write('\n')
        summary.to_csv(f, index=False)
    return seed


def s9_external_validation():
    if not os.path.exists('results/external_validation.csv'):
        return None
    ext = pd.read_csv('results/external_validation.csv').copy()
    if 'auc' in ext.columns:
        ext['auc'] = ext['auc'].round(4)
    ext.to_csv(f'{OUT}/S9_external_validation.csv', index=False)
    return ext


def s10_delong():
    dl = pd.read_csv('results/delong_results.csv').copy()
    for c in ('auc_a', 'auc_b', 'auc_diff'):
        if c in dl.columns:
            dl[c] = dl[c].round(4)
    if 'z' in dl.columns:
        dl['z'] = dl['z'].round(3)
    if 'p_value' in dl.columns:
        dl['p_value'] = dl['p_value'].map(lambda p: f'{p:.4g}')
    dl.to_csv(f'{OUT}/S10_delong.csv', index=False)
    return dl


def main():
    funcs = [s1_cohort_overview, s2_per_fold_aucs, s3_top_shap,
             s4_bootstrap_cis, s5_sensitivity_grid, s6_adenoma_lodo,
             s7_confounder, s8_seed_sensitivity, s9_external_validation,
             s10_delong]
    for f in funcs:
        try:
            out = f()
            n = len(out) if out is not None else 0
            print(f'{f.__name__:30s} rows={n}')
        except Exception as e:
            print(f'{f.__name__:30s} FAILED: {e}')

    index = pd.DataFrame([
        {'table': 'S1', 'file': 'S1_cohort_overview.csv',
         'description': 'Per-cohort sample counts, demographics, sequencing depth'},
        {'table': 'S2', 'file': 'S2_per_fold_aucs.csv',
         'description': 'Per-fold AUC for species RF, joint RF, joint XGB, bio-pathway RF, and ComBat-corrected species RF'},
        {'table': 'S3', 'file': 'S3_top_shap_features.csv',
         'description': 'Top 20 SHAP species per task (CRC-vs-control, H-vs-A, A-vs-CRC for both RF and XGB)'},
        {'table': 'S4', 'file': 'S4_bootstrap_ci.csv',
         'description': '2000-iteration bootstrap 95% CIs (per-cohort and pooled)'},
        {'table': 'S5', 'file': 'S5_sensitivity_grid.csv',
         'description': 'Per-fold pathway filter threshold sweep (4 prev x 5 mean = 20 cells)'},
        {'table': 'S6', 'file': 'S6_adenoma_lodo.csv',
         'description': 'Adenoma cross-cohort LODO (4 cohorts, 4 tasks)'},
        {'table': 'S7', 'file': 'S7_confounder_adjustment.csv',
         'description': 'Age, sex, BMI adjustment via direct inclusion and residualization'},
        {'table': 'S8', 'file': 'S8_seed_sensitivity.csv',
         'description': 'Species RF LODO at 5 random seeds with summary statistics'},
        {'table': 'S9', 'file': 'S9_external_validation.csv',
         'description': 'Single-cohort holdout AUC for external validation'},
        {'table': 'S10', 'file': 'S10_delong.csv',
         'description': 'DeLong (1988; Sun and Xu 2014) significance tests on pooled LODO predictions'},
    ])
    index.to_csv(f'{OUT}/INDEX.csv', index=False)
    print(f'\nWrote {len(index)} supplementary tables to {OUT}/')


if __name__ == '__main__':
    main()
