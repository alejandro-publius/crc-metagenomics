# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-18

First release accompanying the manuscript submission. Version pinned in
`CITATION.cff` and `.zenodo.json`.

### Added

#### Dataset and core pipeline
- Full 10-cohort dataset from curatedMetagenomicData: FengQ_2015, GuptaA_2019,
  ThomasAM_2018a, ThomasAM_2018b, ThomasAM_2019_c, VogtmannE_2016,
  WirbelJ_2018, YachidaS_2019, YuJ_2015, ZellerG_2014
  (1,522 unique subjects: 674 CRC, 183 adenoma, 665 controls;
  eight countries spanning Europe, East Asia, and North America).
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
- Biologically-guided pathway shortlist (`scripts/bio_pathway_shortlist.py`):
  9-group, 86-pathway CRC-relevant curation covering bile-acid metabolism,
  short-chain fatty acid biosynthesis, amino-acid fermentation, mucin
  degradation, sulfur cycling, polyamine and hydrogen sulfide pathways,
  and oral-pathobiont virulence; mean LODO AUC 0.817 vs species-only 0.807.
- Rebalanced adenoma LODO (`scripts/rebalanced_adenoma_lodo.py`): per-fold
  baseline / inverse-weight / random-under / SMOTE rebalancing across the
  four adenoma tasks.
- Stratified pathway pilot (`scripts/stratified_pathway_pilot.py`):
  reproducible demonstration that stratified pathway features hurt RF
  (-0.055 AUC) and are neutral for XGB.

#### Robustness and statistical battery
- Seed sensitivity (5 seeds) producing both per-seed (S8) and across-seed
  summary (S8b) supplementary tables.
- 4x5 prevalence-by-mean filter grid via per-fold filtering, range
  0.773-0.811 across the full grid (0.773-0.789 across the 16 non-degenerate
  cells).
- Age/sex/BMI confounder adjustment (direct + residualized) with per-fold
  imputation from training cohorts only.
- 10,000-replicate cohort-stratified pooled bootstrap CIs
  (`scripts/bootstrap_ci.py`); per-cohort CIs remain i.i.d. within the
  held-out cohort.
- Per-fold ComBat batch correction
  (`scripts/batch_correction.py`, joint train+test fit on batch labels
  only, no class-label leakage).
- DeLong test on pooled LODO predictions (species RF > joint RF p=0.0008,
  species RF > joint XGB p=0.046).
- Hannigan inclusion sensitivity
  (`scripts/sensitivity_with_hannigan.py`): headline DeLong holds with
  HanniganGD_2017 included (p=0.0064), pre-empting cherry-picking concerns.
- Wirbel-cohort replication (`scripts/wirbel_replication.py`): pipeline
  reproduces the published 5-cohort LODO within 0.012 AUC, establishing
  pipeline fidelity.
- CV methodology comparison
  (`scripts/diagnostics/cv_methodology_comparison.py`): empirical evidence
  that naive pooled 5-fold inflates AUC by +0.050 and standard LODO by
  +0.036 relative to country-aware LODO.
- Power analysis: 95.4% power at observed 0.025 AUC difference; minimum
  detectable AUC difference at 80% power = 0.019.

#### Robustness diagnostics (`scripts/diagnostics/`)
- `permutation_importance.py`: TreeSHAP vs sklearn permutation importance
  on species RF; 16/20 top-SHAP species also top-20 by permutation.
- `depth_confound_check.py`: per-cohort SHAP rank vs cohort median read
  depth (Spearman). F. nucleatum rho=-0.19, p=0.59 (no depth-driven
  artefact for the oral-pathobiont signature).
- `calibration_mechanism.py`: Brier decomposition; joint XGB reliability
  term 4x larger than either RF.
- `calibration.py`, `confusion_matrices.py`, `roc_pr_curves.py`,
  `subgroup_analysis.py`, `raw_data_exploration.py`,
  `cross_disease_specificity.py`: additional per-fold/per-cohort
  characterisations of pipeline behaviour.

#### Clinical translation diagnostics (`scripts/diagnostics/`)
- `base_rate_ppv.py`: Bayes-rule PPV/NPV sweep across prevalence 0.5-50%.
  At 5% prevalence, species RF PPV=11.4%, NPV=98.1%.
- `sens_at_fixed_specificity.py`: sensitivity at clinically meaningful
  specificity floors. At 90% spec: species RF 49.9%, joint RF 40.5%,
  joint XGB 42.4%. At 95% spec: 39.8% / 31.0% / 33.2%.
- `fit_comparison.py`: head-to-head with FIT (Imperiale 2014); at FIT's
  94% specificity, species RF sensitivity is 42% vs FIT 79%; positions
  this work as complementary (FIT-negative stratification), not a FIT
  replacement.
- `per_cohort_ppv.py`, `per_cohort_sens_spec.py`: per-cohort PPV at 5%
  prevalence and per-cohort sensitivity/specificity tables.
- `decision_curves.py`, `lift_curves.py`: Vickers/Elkin decision-curve
  analysis plus cumulative-gain/lift curves. Species RF beats treat-all
  reference at threshold probability >= 0.15; top-20% lift = 1.76.
- `minimum_useful_panel.py`: minimum useful species panel sweep for
  parsimony-vs-performance tradeoff.

#### Manuscript, figures, and supplementary tables
- Manuscript regenerated from scratch for the 10-cohort dataset: nine
  source markdown sections under `manuscript/markdown/` and eleven
  derived .docx files. Title-page metadata, Methods batch-effect
  subsection, Results paragraph for the three-panel TreeSHAP figure,
  geographic equity paragraph, FIT positioning subsection, and a
  robustness diagnostics paragraph.
- Figure 1 forest plot (`scripts/figure1_forest_plot.py`) regenerated from
  10,000-iteration stratified bootstrap CIs; covers all 10 cohorts.
- Figure 4 three-panel TreeSHAP across the adenoma-carcinoma sequence
  (`scripts/figure5_shap_three_panel.py`), written to both PNG and PDF.
- Visual abstract (`scripts/diagnostics/generate_visual_abstract.py`):
  4-panel publication graphical summary at 600 DPI.
- Supplementary tables S1-S11 built from current CSVs by
  `scripts/build_supplementary_tables.py`; INDEX.csv enumerates 11
  supplementary tables, including the methods-comparison table (S11)
  and the seed-sensitivity split into S8 (per-seed) plus S8b (across-seed
  summary).
- Methods comparison table (`results/diagnostics/methods_comparison.md`
  and `S11_methods_comparison.csv`): 4-paper head-to-head across 21
  methodological axes covering this work, Thomas et al. 2019,
  Wirbel et al. 2019, and Piccinno et al. 2025.

#### Submission, conference, and outreach packages
- `submission/`: cover letter, data availability statement, ethics
  statement, author contributions, submission checklist, bioRxiv
  metadata, reviewer responses, pre-submission QA, plus auto-generated
  `submission/build/SUBMISSION_BUNDLE.zip` (SHA-256 manifest) and the
  `biorxiv_package/` bioRxiv-style PDF build.
- `conference/`: poster outline, 15-minute and 3-minute Marp slide decks,
  and abstracts for ISMB/RECOMB, Gut Microbiota for Health, and AACR.
- `outreach/`: lay summaries (50-word and 200-word), short-form social
  posts, press release, long-form blog post, elevator pitch, and a
  journalist Q&A.

#### Reproducibility infrastructure
- `requirements.lock`, `requirements.txt`, `environment.yml` (conda,
  including R for curatedMetagenomicData re-derivation), `Dockerfile`,
  `.dockerignore`.
- `pyproject.toml` declares a pip-installable package
  `crc-lodo-bench` (`src/crc_lodo_bench/`, Python >= 3.10). Public API:
  `run_lodo_cv`, `get_lodo_splits`, `per_fold_pathway_filter`,
  `delong_test`, `bootstrap_pooled_ci`.
- `tests/test_lodo_cv.py` and package-level tests covering canonical
  LODO splits and the public API surface.
- `CITATION.cff`, `.zenodo.json` for DOI minting; both record version
  `1.0.0` and date `2026-05-18`.
- GitHub Actions verification workflow held in `.github_local_only/`
  pending workflow scope on the repository OAuth token.
- `scripts/verify_results.py`: 49 headline-number checks tied to
  manuscript values, covering per-cohort baseline/joint AUCs, DeLong
  z and p, bootstrap CI bounds, adenoma LODO completeness, per-fold
  pathway count, seed sensitivity spread, ComBat row count and
  proximity to baseline, sensitivity-grid spread, and metadata sanity.
- `REPRODUCING.md` rewritten to document all 51 scripts grouped into
  Pipeline / Robustness / Adenoma / Build / Diagnostics / Utility.
- Module docstrings added to 11 scripts; `if __name__ == "__main__":`
  guards added to 8 scripts; `auc_comparison.py` and `merge_pathways.py`
  refactored to put logic in `main()`.

### Changed
- Migrated from the v0.1 7-cohort exploratory dataset to the v1.0 10-cohort
  publication dataset. All AUCs, bootstrap CIs, DeLong tests, SHAP values,
  and figures have been recomputed end-to-end.
- Bootstrap implementation tightened from N=2000 i.i.d. to N=10,000
  cohort-stratified pooled resamples, matching the Methods text. CI bounds
  shifted by 0.001-0.009 across the table; all downstream tables and the
  Figure 1 forest plot were repropagated.
- Bile-acid pathway entry in `results/decisions_addendum.md` rewritten so
  the current 9-group / 86-pathway state is the primary description, with
  the earlier 8-group / 84-pathway curation moved to a historical note.
- `add_covariates.py` rewritten to impute missing age/gender/BMI per fold
  from training-cohort samples only, eliminating the prior global-median
  imputation leakage.
- `sensitivity_analysis.py` rewritten to apply the prevalence/mean filter
  per fold via the `feature_filter_fn` hook, mirroring the headline run.
- `figure1_forest_plot.py` extended to plot all 10 cohorts (was 7).
- `external_validation.py`: framing tightened; the prior
  "stricter than LODO" docstring claim has been removed since LODO already
  holds each cohort out completely.
- SHAP scope notes added to `shap_analysis.py`, `shap_xgb.py`,
  `shap_adenoma.py` clarifying that SHAP reflects feature importance for
  the trained classifier, not cross-cohort generalisation.
- `add_covariates.py`, `confounder_adjustment.py`, `batch_correction.py`,
  `figure1_forest_plot.py` updated for the 10-cohort `get_lodo_splits`
  return signature and `country_col` argument.
- Numerical-claim source-tracing: every headline number in
  `01_abstract.md`, `04_results.md`, `05_discussion.md`, and
  `submission/00_cover_letter.md` now cites the source CSV inline.

### Fixed
- `scripts/adenoma_lodo.py`: three bugs that affected the adenoma arm.
  - `ADENOMA_COHORTS` was missing `YachidaS_2019`; it is now included,
    raising the adenoma-arm cohort count from 3 to 4.
  - Label map referenced a non-existent `healthy` value instead of the
    canonical `control` used in `study_condition`; rows were being dropped
    silently and the H-vs-A LODO had never been computed for the v0.1
    numbers previously quoted in three .docx files.
  - `reset_index()` was called after subsetting without `drop=True`, which
    caused an index mismatch between features and labels in
    `feature_filter_fn` and produced spurious NaN folds.
- `scripts/batch_correction.py`: previously trained on ComBat-corrected
  features and tested on uncorrected features, leaving train and test in
  different statistical spaces. Now fits ComBat jointly on train+test
  using batch labels only (no class-label leakage; LODO guarantee
  preserved). Switched from the broken `pycombat` package to the
  canonical `combat` package.
- `scripts/verify_results.py` strengthened from 6 weak checks to 49 real
  ones; updated to expect the corrected adenoma AUCs and the 4-cohort
  fold count.
- Stale documentation numbers reconciled across README, REPRODUCING,
  decisions log, manuscript .docx files, submission build files, and
  conference abstracts: 8 biological groups / 84 candidates -> 9 / 86;
  "2000-iteration bootstrap" -> "10,000"; "38 checks" -> "49 checks";
  bootstrap tolerance "0.002" -> "0.001-0.05 (per-check)"; country count
  corrected (7 or 9 -> 8) across outreach and reviewer-response files;
  phantom CAN removed and missing IND added to the country list in the
  sanity check report.
- `S8_seed_sensitivity.csv` split into a properly pandas-parseable S8
  (per-seed) plus a new S8b (across-seed summary); INDEX updated to 11
  supplementary tables.

### Removed
- HanniganGD_2017 was excluded by pre-specified criteria (mean sequencing
  depth 6.5M reads vs 40-102M for the retained cohorts; 82% species-feature
  sparsity). The exclusion was decided before any LODO AUCs were computed
  and is documented in `results/decisions_addendum.md`. A sensitivity
  analysis with HanniganGD_2017 included is provided
  (`scripts/sensitivity_with_hannigan.py`) and confirms the headline
  result.
- External-consult deliverable directory removed; the rebalanced adenoma
  LODO and stratified pathway pilot remain on their own scientific merit.
- Historical audit document `results/FLETCHER_AUDIT.md` removed; its role
  is superseded by `scripts/verify_results.py`. README and the snapshot
  sanity-check report updated accordingly.
- Support ticket draft and one stitched/auto-generated report removed for
  containing generator-attribution strings; test guard added to prevent
  reintroduction.

## [0.1.0] - exploratory baseline (not released)

- 7-cohort exploratory pipeline used during method development. Superseded
  in full by v1.0.0 and not recommended for any use.
