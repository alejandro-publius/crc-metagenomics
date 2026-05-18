"""Replicate Wirbel et al. 2019 (Nat Med) pooled LODO meta-analysis on the
same 5 CRC cohorts they used, restricted to species-level features.

Wirbel et al. published the first cross-cohort CRC meta-analysis on five
shotgun cohorts: FengQ_2015 (AUT), VogtmannE_2016 (USA), YuJ_2015 (CHN),
ZellerG_2014 (FRA), and the new German cohort introduced in the paper
(WirbelJ_2018, DEU). Pooled LODO AUC was reported around 0.80-0.85.

This script subsets the existing curatedMetagenomicData-derived
species_filtered.csv + metadata_clean.csv to those 5 cohorts, then runs
the same country-aware LODO RF used in train_baseline.py and writes
per-cohort AUCs + mean to results/wirbel_replication.csv.

Notes on cohort matching:
- All 5 Wirbel 2019 cohorts ARE present in our curatedMetagenomicData
  pull (FengQ_2015, VogtmannE_2016, YuJ_2015, ZellerG_2014, WirbelJ_2018).
- Sample counts may differ slightly from the original publication
  because curatedMetagenomicData re-processes raw reads with its own
  pipeline (MetaPhlAn versions, QC) and a few samples can be excluded.
- The country-aware LODO splitter is unchanged, so when WirbelJ_2018
  (DEU) is the test fold no other DEU cohort is removed (there isn't one
  in this 5-cohort subset). Likewise for the other 4 countries.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv  # noqa: E402


# 5 cohorts in Wirbel et al. 2019 Nat Med
WIRBEL_COHORTS = [
    'FengQ_2015',
    'VogtmannE_2016',
    'YuJ_2015',
    'ZellerG_2014',
    'WirbelJ_2018',
]

# Approximate per-cohort CRC vs control AUCs reported in Wirbel 2019
# (Figure 2 / Supplementary; species LASSO LODO).  Used for the
# comparison column only.  These are rounded from the paper text and
# should be treated as +/- 0.02 reference points.
WIRBEL_REPORTED_AUC = {
    'FengQ_2015':    0.79,
    'VogtmannE_2016':0.80,
    'YuJ_2015':      0.79,
    'ZellerG_2014':  0.84,
    'WirbelJ_2018':  0.84,
}
WIRBEL_REPORTED_MEAN = 0.81  # pooled LODO mean reported ~0.80-0.85


def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')

    present = sorted(set(WIRBEL_COHORTS) & set(md['study_name'].unique()))
    missing = sorted(set(WIRBEL_COHORTS) - set(md['study_name'].unique()))
    print(f'Wirbel 2019 cohorts present in curatedMetagenomicData pull: '
          f'{len(present)}/{len(WIRBEL_COHORTS)}')
    for c in present:
        print(f'  - {c}')
    if missing:
        print(f'Missing (will be skipped): {missing}')

    mg = md.merge(sp, on='sample_id', how='inner')
    mg = mg[mg['study_name'].isin(present)].reset_index(drop=True)

    fc = [c for c in sp.columns if c != 'sample_id']
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta_cols = ['sample_id', 'study_name', 'study_condition', 'label']
    if 'country' in mg.columns:
        meta_cols.append('country')
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)

    print(f'\nSamples (CRC/control only): {len(X)} '
          f'(CRC={int(y.sum())}, control={int((y == 0).sum())})')
    print(f'Features (species): {X.shape[1]}')
    print('Per-cohort breakdown:')
    print(meta.groupby(['study_name', 'study_condition']).size())

    def make_rf():
        return RandomForestClassifier(
            n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42,
            class_weight='balanced')

    print('\nRunning country-aware LODO RF on the Wirbel 5-cohort subset...')
    results = run_lodo_cv(
        make_rf, X, y, meta,
        country_col='country' if 'country' in meta.columns else None,
    )

    out = pd.DataFrame({
        'cohort':            results['cohort'],
        'n_train':           results['n_train'],
        'n_test':            results['n_test'],
        'auc_ours':          results['auc'],
        'auc_wirbel_2019':  [WIRBEL_REPORTED_AUC.get(c, np.nan)
                              for c in results['cohort']],
    })
    out['delta'] = out['auc_ours'] - out['auc_wirbel_2019']

    mean_ours    = float(np.mean(results['auc']))
    mean_wirbel  = float(np.mean([WIRBEL_REPORTED_AUC[c]
                                  for c in results['cohort']
                                  if c in WIRBEL_REPORTED_AUC]))

    summary = pd.DataFrame([{
        'cohort':           'MEAN',
        'n_train':          '',
        'n_test':           '',
        'auc_ours':         mean_ours,
        'auc_wirbel_2019':  mean_wirbel,
        'delta':            mean_ours - mean_wirbel,
    }])
    out = pd.concat([out, summary], ignore_index=True)

    os.makedirs('results', exist_ok=True)
    out.to_csv('results/wirbel_replication.csv', index=False)
    print(f'\nMean AUC (ours):           {mean_ours:.3f}')
    print(f'Mean AUC (Wirbel reported): {mean_wirbel:.3f} '
          f'(~{WIRBEL_REPORTED_MEAN:.2f} headline)')
    print(f'Delta:                      {mean_ours - mean_wirbel:+.3f}')
    print('Saved to results/wirbel_replication.csv')


if __name__ == '__main__':
    main()
