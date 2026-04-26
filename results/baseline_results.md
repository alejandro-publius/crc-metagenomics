# Baseline Species-Only Results

Random Forest classifier trained on 247 filtered species features using leave-one-dataset-out (LODO) cross-validation across 7 cohorts.

## Per-cohort AUC

| Cohort          | AUC   | n   |
|-----------------|-------|-----|
| FengQ_2015      | 0.838 | 107 |
| ThomasAM_2018a  | 0.734 | 53  |
| ThomasAM_2018b  | 0.801 | 60  |
| ThomasAM_2019_c | 0.828 | 80  |
| VogtmannE_2016  | 0.730 | 104 |
| YuJ_2015        | 0.868 | 128 |
| ZellerG_2014    | 0.821 | 114 |

**Mean AUC: 0.803 +/- 0.049**

## Comparison to Thomas et al. 2019

Thomas et al. reported a mean LODO AUC of ~0.80 on a similar species-only feature set across 5 cohorts. Our 7-cohort baseline of 0.803 is consistent with their result.

## Model configuration

- RandomForestClassifier(n_estimators=500, max_features=sqrt, min_samples_leaf=5, class_weight=balanced, random_state=42)
- LODO: train on 6 cohorts, test on the 7th, repeat for all 7 folds
- 646 samples used (binary CRC vs control; 116 adenoma samples excluded)

## Per-cohort observations

- YuJ_2015 (0.868) is the easiest cohort, consistent with its larger sample size and balanced class distribution
- VogtmannE_2016 (0.730) and ThomasAM_2018a (0.734) are the hardest, likely due to smaller sample sizes
- All cohorts exceed AUC 0.70, indicating the species-only signal generalizes across studies

## Comparison to joint model

Adding 405 unstratified pathway features (Joint RF 0.783, Joint XGB 0.790) does not improve over the species-only baseline.

- Per-cohort paired tests (n=7 LODO folds) do not detect a difference: paired t p=0.41 vs Joint RF, p=0.45 vs Joint XGB; 95% bootstrap CIs on the mean per-cohort difference include zero.
- DeLong tests on pooled LODO predictions (n=646) find species RF significantly better: p=0.004 vs Joint RF, p=0.013 vs Joint XGB. Pooled AUCs: species RF 0.810, Joint RF 0.777, Joint XGB 0.782.

The two tests answer different questions. Per-cohort tests assess whether the joint model is consistently better across cohorts (it is not, and n=7 has low power). DeLong assesses whether overall sample-level discrimination differs between classifiers given the same held-out predictions. Both agree that pathways do not help; DeLong indicates a small but statistically detectable degradation, likely from added noise in the 405 redundant pathway features.

See results/model_comparison.csv and results/delong_results.csv.

## Files

- results/baseline_results.csv: per-cohort AUCs (source of truth)
- scripts/train_baseline.py: produces this result
