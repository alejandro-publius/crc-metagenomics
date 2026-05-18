# Baseline Species-Only Results (10-cohort dataset)

Random Forest classifier trained on 229 filtered species features (MetaPhlAn,
prevalence >= 10%, mean >= 1e-4, log10-transformed) under country-aware
leave-one-dataset-out (LODO) cross-validation across 10 cohorts, 1339 samples
(674 CRC vs 665 control; 183 adenoma samples excluded from main LODO).

## Per-cohort AUC

| Cohort          | AUC   | n_test | Country | Country exclusion        |
|-----------------|-------|--------|---------|--------------------------|
| FengQ_2015      | 0.840 | 107    | AUT     | —                        |
| GuptaA_2019     | 0.882 | 60     | IND     | —                        |
| ThomasAM_2018a  | 0.694 | 53     | ITA     | ThomasAM_2018b excluded  |
| ThomasAM_2018b  | 0.810 | 60     | ITA     | ThomasAM_2018a excluded  |
| ThomasAM_2019_c | 0.836 | 80     | JPN     | YachidaS_2019 excluded   |
| VogtmannE_2016  | 0.756 | 104    | USA     | —                        |
| WirbelJ_2018    | 0.882 | 125    | DEU     | —                        |
| YachidaS_2019   | 0.708 | 508    | JPN     | ThomasAM_2019_c excluded |
| YuJ_2015        | 0.865 | 128    | CHN     | —                        |
| ZellerG_2014    | 0.803 | 114    | FRA     | —                        |

**Per-cohort mean AUC: 0.808 ± 0.065**
**Pooled LODO AUC: 0.781** (95% CI: 0.756–0.805; 2000-resample bootstrap)

## Comparison to Thomas et al. 2019

Thomas et al. reported a mean LODO AUC of ~0.80 on a similar species-only feature set
across 5 cohorts. Our 10-cohort pooled AUC of 0.781 is consistent with their result,
after accounting for the harder generalization challenge across a larger, more
geographically diverse cohort set and the conservative country-aware exclusion strategy.

## Model configuration

- `RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5,`
  `class_weight='balanced', random_state=42, n_jobs=-1)`
- LODO: train on all cohorts except the test cohort and its country-matched cohorts
- 1339 samples (CRC + control only; 183 adenoma samples excluded from LODO)
- 229 species features (global prevalence/mean filter; see decisions_addendum.md)

## Per-cohort observations

- **YachidaS_2019 (0.708, n=508)** dominates the pooled AUC. With the Japan exclusion,
  only ThomasAM_2019_c remains in training (n_train=751 vs ~1232 for most folds).
  This fold's large test set (n=508) heavily weights the pooled DeLong statistic.
- **ThomasAM_2018a (0.694)** is the lowest per-cohort AUC; small test set (n=53)
  and only non-Italian cohorts in training due to country exclusion.
- **GuptaA_2019 (0.882)** and **WirbelJ_2018 (0.882)** are the highest;
  both have full training sets with no country exclusions.

## Comparison to joint model

Adding 402–406 unstratified pathway features per fold (Joint RF mean 0.804,
Joint XGB 0.797) does not improve over the species-only baseline:

- **Per-cohort paired tests** (n=10): species RF vs Joint RF p=0.87;
  species RF vs Joint XGB p=0.28. Neither significant. (Low power at n=10.)
- **DeLong on pooled predictions** (n=1339): species RF significantly outperforms
  Joint RF (AUC 0.781 vs 0.756, z=3.352, p=0.0008) and Joint XGB (0.781 vs 0.766,
  z=1.996, p=0.046). Signal driven by YachidaS_2019 (species RF 0.708 vs 0.669/0.694).

Both tests agree that pathways add no benefit; DeLong detects a small but statistically
significant degradation at the sample level, driven by the largest fold.

## Files

- `results/baseline_results.csv` — per-cohort AUCs (source of truth)
- `results/preds_species_rf.csv` — per-sample LODO predictions (1339 rows)
- `scripts/train_baseline.py` — produces this result
