# Adenoma Analysis: Decision Memo

## Current dataset (10-cohort)

- Total adenoma samples: **183** (up from 116 in the original 7-cohort dataset)
- Cohorts with adenoma samples: **4 of 10**

| Cohort         | n_adenoma | Country |
|----------------|-----------|---------|
| FengQ_2015     | 47        | AUT     |
| ZellerG_2014   | 42        | FRA     |
| ThomasAM_2018a | 27        | ITA     |
| YachidaS_2019  | 67        | JPN     |

All 4 adenoma-containing cohorts are from different countries, so no country-aware
exclusion is needed for the adenoma LODO.

## CV strategy

With 4 adenoma-containing cohorts (≥ 27 samples each), cross-cohort leave-one-cohort-out
(LODO) is used for both adenoma tasks. This matches the main CRC LODO protocol and
directly tests cross-cohort generalization.

An earlier version of this memo (7-cohort dataset, 116 adenoma samples, 3 cohorts)
documented a decision to use pooled 5-fold within-cohort CV instead of LODO.
**That decision is superseded.** The `train_adenoma.py` script (within-cohort 5-fold CV)
and its results in `adenoma_results.csv` are retained for reference only.
All current adenoma results use `adenoma_lodo.py`.

## Results summary

Full results and interpretation: `results/adenoma_lodo_results.csv` and
`results/decisions_addendum.md`.

- **H-vs-A** (control vs adenoma): RF 0.561, XGB 0.579 — near chance, consistent with
  published literature showing weak cross-cohort microbiome signal for adenoma
- **A-vs-CRC** (adenoma vs CRC): RF 0.671, XGB 0.617 — above chance, driven by the
  oral-bacterial CRC signature (Fusobacterium nucleatum, Parvimonas micra,
  Peptostreptococcus stomatis) emerging during malignant transformation

## Hyperparameter configuration

RF: n_estimators=500, max_features='sqrt', min_samples_leaf=5, class_weight='balanced'.
XGBoost: n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
colsample_bytree=0.8; scale_pos_weight = inverse class ratio (computed per fold).
No nested CV tuning — the joint model did not outperform species-only in the main LODO.

## Limitations

- Adenoma definitions vary across cohorts (advanced vs. non-advanced adenoma;
  not uniformly reported in curatedMetagenomicData metadata)
- Per-fold training sets for adenoma are small (e.g., ThomasAM_2018a n_test=27;
  n_train adenoma ~156), limiting classifier performance
- H-vs-A near-chance performance reflects cross-cohort generalization difficulty,
  not necessarily absence of any microbiome signal for adenoma
