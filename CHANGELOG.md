# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-18

First release accompanying the manuscript submission.

### Added
- Full 10-cohort dataset from curatedMetagenomicData: FengQ_2015, GuptaA_2019,
  ThomasAM_2018a, ThomasAM_2018b, ThomasAM_2019_c, VogtmannE_2016,
  WirbelJ_2018, YachidaS_2019, YuJ_2015, ZellerG_2014
  (1,522 unique subjects: 674 CRC, 183 adenoma, 665 controls).
- Country-aware leave-one-dataset-out (LODO) cross-validation in
  `scripts/lodo_cv.py`: when a cohort is held out as the test fold, all
  cohorts from the same country are dropped from training. This corrects a
  population-confounding artefact that previously inflated ThomasAM_2019_c
  AUC to 0.999 (due to YachidaS_2019 in training); the corrected AUC is
  0.836.
- Per-fold pathway filtering: the prevalence (>= 10%) and mean (>= 1e-6)
  filters for HUMAnN unstratified pathways are now computed on the training
  cohorts of each LODO fold rather than once on the pooled dataset. This
  fixes test-fold leakage that was present in v0.1 and yields 402-406
  retained pathways per fold (instead of a single global set).
- Adenoma classification arm (`scripts/adenoma_lodo.py`): 4-cohort LODO for
  healthy-vs-adenoma and adenoma-vs-CRC tasks.
- Robustness battery: seed sensitivity (5 seeds), 4x5 prevalence-by-mean
  filter grid, age/sex/BMI confounder adjustment (direct + residualized),
  10,000-replicate cohort-stratified bootstrap CIs, per-fold ComBat batch
  correction, and a biologically-guided 84-pathway shortlist.
- Reproducibility infrastructure: `requirements.lock`, `environment.yml`,
  `Dockerfile`, GitHub Actions verification workflow, pytest stubs for
  `lodo_cv.run_lodo_cv`, `CITATION.cff`, `.zenodo.json`.
- `scripts/verify_results.py`: 38 headline-number checks tied to manuscript
  values; runs in CI on every push and PR.

### Changed
- Migrated from the v0.1 7-cohort exploratory dataset to the v1.0 10-cohort
  publication dataset. All AUCs, bootstrap CIs, DeLong tests, SHAP values,
  and figures have been recomputed end-to-end.

### Fixed
- `scripts/adenoma_lodo.py`: three bugs that affected the adenoma arm.
  - `ADENOMA_COHORTS` was missing `YachidaS_2019`; it is now included,
    raising the adenoma-arm cohort count from 3 to 4.
  - Label map referenced a non-existent `healthy` value instead of the
    canonical `control` used in `study_condition`; rows were being dropped
    silently.
  - `reset_index()` was called after subsetting without `drop=True`, which
    caused an index mismatch between features and labels in
    `feature_filter_fn` and produced spurious NaN folds.
- `scripts/verify_results.py` was updated to expect the corrected adenoma
  AUCs and the 4-cohort fold count.

### Removed
- HanniganGD_2017 was excluded by pre-specified criteria (mean sequencing
  depth 6.5M reads vs 40-102M for the retained cohorts; 82% species-feature
  sparsity). The exclusion was decided before any LODO AUCs were computed
  and is documented in `results/decisions_addendum.md`.

## [0.1.0] - exploratory baseline (not released)

- 7-cohort exploratory pipeline used during method development. Superseded
  in full by v1.0.0 and not recommended for any use.
