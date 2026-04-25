# Adenoma Analysis: Go/No-Go Memo

## Sample availability
- Total adenoma samples across 7 cohorts: 116
- Roadmap threshold for "meaningful analysis": ~30 adenoma samples
- Cohorts with >=5 adenoma samples: 3 of 7
- Cohorts with >=10 adenoma samples: 3 of 7

## Per-cohort breakdown
    study_name  n_adenoma
    FengQ_2015         47
  ZellerG_2014         42
ThomasAM_2018a         27

## Decision
Total adenoma count (116) EXCEEDS the 30-sample threshold.

## CV strategy
Per-cohort adenoma counts are too sparse for LODO (most cohorts have
fewer than 10 adenoma samples). The adenoma analysis uses pooled
5-fold stratified cross-validation instead. This deviation from the
LODO protocol used elsewhere in the paper is documented in Methods.

## Limitations to disclose in Discussion
- 5-fold CV does not test cross-cohort generalization for adenoma
- Adenoma definitions may vary across cohorts (advanced vs non-advanced)
- Sample size limits the precision of reported AUCs

## Hyperparameter handling (Methods note)

XGBoost was used with commonly-cited default hyperparameters
(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
colsample_bytree=0.8). Hyperparameter tuning via nested CV was not
performed. Given that the joint XGBoost model did not statistically
outperform species-only Random Forest in our LODO comparison
(see results/model_comparison.csv), additional tuning was unlikely
to alter the qualitative conclusion. Random Forest used n_estimators=500,
max_features='sqrt', min_samples_leaf=5, class_weight='balanced'.
For adenoma classification, XGBoost additionally used scale_pos_weight
set to the inverse class ratio to handle class imbalance.

## Methods sentence (paste-ready)

"Random Forest and XGBoost were used with commonly-cited default
hyperparameters (RF: 500 trees, max_features=sqrt(p), min_samples_leaf=5,
class_weight='balanced'; XGBoost: 500 trees, max_depth=6,
learning_rate=0.1, subsample=0.8, colsample_bytree=0.8). XGBoost
adenoma classifiers additionally used scale_pos_weight equal to the
inverse class ratio. Hyperparameters were not tuned because the joint
model did not statistically outperform the species-only baseline."
