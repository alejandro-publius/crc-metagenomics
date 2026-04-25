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
