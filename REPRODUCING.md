# Reproducing the Analyses

See README.md and results/decisions_addendum.md for the canonical pipeline and all analytical decisions.

## Environment

```bash
pip install -r requirements.lock
```

All scripts use `random_state=42` and produce deterministic results. Total runtime is approximately 45 minutes on a standard workstation (dominated by LODO CV training).

## Pipeline — preprocessing

### 1. Data export (R)

```bash
Rscript scripts/export_data.R          # pulls curatedMetagenomicData and writes 11 cohorts (~1604 samples) to data/raw/
                                       # HanniganGD_2017 is dropped at the preprocessing step below, leaving 10 cohorts (1,522 samples)
Rscript scripts/audit_subject_ids.R    # verifies no duplicated subject IDs across cohorts
```

### 2. Data processing

```bash
python3 scripts/merge_pathways.py      # concatenates per-cohort HUMAnN pathway chunks -> data/raw/pathway_abundance.csv
python3 scripts/validate_pathways.py   # sanity-checks pathway matrix dimensions and sample-ID overlap
python3 scripts/filter_pathways.py     # global static filter -> data/processed/pathway_unstratified.csv (used by SHAP / adenoma scripts only)
python3 scripts/preprocessing.py       # quality filters: excludes HanniganGD_2017 (low depth),
                                       # drops samples <1M reads; outputs 1,522 samples, 10 cohorts
                                       # -> data/processed/species_filtered.csv, metadata_clean.csv
python3 scripts/generate_table1.py     # Table 1 demographics -> results/table1.csv
python3 scripts/adenoma_counts.py      # per-cohort adenoma sample counts -> results/adenoma_counts_per_cohort.csv
python3 scripts/add_covariates.py      # appends age/sex/BMI/country covariates onto metadata_clean.csv (idempotent)
```

## Pipeline — main classification

### 3. LODO classification

```bash
python3 scripts/train_baseline.py   # species-only RF LODO (country-aware)
                                    # expect: per-cohort mean AUC ~0.807, pooled AUC ~0.781
                                    # outputs: results/baseline_results.csv, preds_species_rf.csv

python3 scripts/train_joint.py      # joint RF + XGBoost LODO (country-aware, per-fold pathway filter)
                                    # expect: Joint RF per-cohort ~0.804 (pooled ~0.756)
                                    #         Joint XGB per-cohort ~0.797 (pooled ~0.766)
                                    # 551 pathway candidates; 402-406 retained per fold
                                    # outputs: results/joint_results.csv, preds_joint_rf.csv, preds_joint_xgb.csv

python3 scripts/auc_comparison.py   # paired tests (t, Wilcoxon) + DeLong on pooled predictions
                                    # expect: species_rf vs joint_rf DeLong z=3.35, p=0.0008
                                    #         species_rf vs joint_xgb DeLong z=2.00, p=0.046
                                    # outputs: results/model_comparison.csv, delong_results.csv
```

### 4. Feature importance (SHAP)

```bash
python3 scripts/shap_analysis.py    # RF TreeSHAP for CRC vs control -> results/shap_crc_features.csv
python3 scripts/shap_xgb.py         # XGBoost TreeSHAP for all three tasks -> results/shap_*_xgb.csv
python3 scripts/shap_adenoma.py     # RF TreeSHAP for adenoma tasks (H-vs-A, A-vs-CRC) -> results/shap_{healthy_vs_adenoma,adenoma_vs_crc}.csv
```

## Pipeline — adenoma classification

### 5. Adenoma LODO

```bash
python3 scripts/adenoma_lodo.py             # cross-cohort LODO across 4 adenoma-containing cohorts
                                            # (FengQ_2015, ZellerG_2014, ThomasAM_2018a, YachidaS_2019)
                                            # expect: H-vs-A RF ~0.561, H-vs-A XGB ~0.579
                                            #         A-vs-CRC RF ~0.671, A-vs-CRC XGB ~0.617
                                            # outputs: results/adenoma_lodo_results.csv
python3 scripts/rebalanced_adenoma_lodo.py  # repeats H-vs-A LODO under inverse-frequency / random-undersampling / SMOTE
                                            # outputs: results/adenoma_rebalanced_lodo.csv, adenoma_rebalanced_summary.csv
```

`scripts/train_adenoma.py` is retained for reference only — it runs the
adenoma classifiers under within-cohort stratified k-fold (not LODO)
and is superseded by the two LODO scripts above.

### 6. Biologically-guided pathway shortlist

```bash
python3 scripts/bio_pathway_shortlist.py    # keyword-selected 86 unique CRC-relevant pathways across 9 groups
                                            # expect: mean per-cohort LODO AUC ~0.817 (vs species-only 0.807)
                                            # outputs: results/bio_pathway_results.csv, preds_bio_pathway_rf.csv, bio_pathway_shortlist.txt
python3 scripts/stratified_pathway_pilot.py # pilot of taxon-stratified pathway features (>4700 columns)
                                            # outputs: results/stratified_pathway_pilot.csv
```

## Pipeline — robustness battery

### 7. Robustness

```bash
python3 scripts/bootstrap_ci.py          # 10,000-resample bootstrap 95% CIs (cohort-stratified pooled)
                                         # expect: species RF pooled 0.781 [0.757, 0.805]
                                         # outputs: results/bootstrap_ci.csv

python3 scripts/seed_sensitivity.py      # seeds {0,1,2,42,100}; expect spread < 0.005
                                         # outputs: results/seed_sensitivity.csv

python3 scripts/sensitivity_analysis.py  # 4x5 prevalence/mean grid (country-aware, per-fold filter)
                                         # expect: joint RF mean per-cohort AUC range 0.781-0.835
                                         # outputs: results/sensitivity_thresholds.csv

python3 scripts/confounder_adjustment.py # direct inclusion + residualization of age, sex, BMI (country-aware)
                                         # expect: per-cohort AUC 0.800-0.814 (within noise of unadjusted 0.807)
                                         # outputs: results/confounder_results.csv, covariate_comparison.csv

python3 scripts/batch_correction.py      # per-fold ComBat on species features (country-aware LODO)
                                         # requires: pip install combat
                                         # expect: mean per-cohort AUC ~0.815 (vs uncorrected ~0.807)
                                         # outputs: results/combat_results.csv

python3 scripts/external_validation.py   # placeholder external-cohort validation hook
                                         # outputs: results/external_validation.csv
```

One-shot wrapper that re-runs the full robustness battery in sequence:

```bash
bash scripts/run_robustness.sh           # chains bootstrap_ci + seed_sensitivity + sensitivity_analysis
                                         # + confounder_adjustment + batch_correction + adenoma_lodo + verify_results
```

## Pipeline — figures and verification

### 8. Figures

```bash
python3 scripts/generate_figures.py          # legacy / draft figures into figures/
python3 scripts/figure1_forest_plot.py       # Figure 1: forest plot of per-cohort + pooled CIs
python3 scripts/figure5_shap_three_panel.py  # Figure 4: three-panel SHAP (H-vs-A, CRC-vs-control, A-vs-CRC)
```

### 9. Verification

```bash
python3 scripts/verify_results.py            # 49 smoke-test assertions against committed CSVs
pytest tests/ -v                             # unit tests for scripts/lodo_cv.py
```

## Optional diagnostics

The `scripts/diagnostics/` directory contains 12 standalone, idempotent
post-hoc scripts that consume the three `results/preds_*.csv` files
(plus, where indicated, `data/processed/`) and write into
`results/diagnostics/` and `figures/diagnostics/`. None are required
for the headline numbers; each addresses an anticipated reviewer
critique or a clinical-translation question.

See `results/diagnostics/README.md` for a one-paragraph description of
each diagnostic and `results/diagnostics/{ROBUSTNESS_SUMMARY.md,
CLINICAL_TRANSLATION_SUMMARY.md, RAW_DATA_PATTERNS.md}` for narrative
roll-ups.

```bash
# Calibration and discrimination
python3 scripts/diagnostics/calibration.py             # Brier + ECE + reliability diagrams
python3 scripts/diagnostics/calibration_mechanism.py   # Brier (Murphy 1973) decomposition; explains the joint-XGB ECE gap
python3 scripts/diagnostics/confusion_matrices.py      # confusion matrices at Youden-J and 0.5
python3 scripts/diagnostics/roc_pr_curves.py           # pooled ROC and PR curves -> auprc_pooled.csv

# Operating characteristics for screening
python3 scripts/diagnostics/per_cohort_sens_spec.py    # per-cohort sens/spec/PPV/NPV at pooled Youden-J
python3 scripts/diagnostics/sens_at_fixed_specificity.py  # sensitivity at 90% and 95% specificity
python3 scripts/diagnostics/base_rate_ppv.py           # PPV / NPV across a CRC-prevalence sweep
python3 scripts/diagnostics/fit_comparison.py          # species RF vs published FIT (Imperiale 2014) at matched specificity

# Robustness against reviewer critiques
python3 scripts/diagnostics/permutation_importance.py  # permutation importance vs TreeSHAP for the species RF
python3 scripts/diagnostics/depth_confound_check.py    # per-cohort sequencing-depth confounding of SHAP rankings
python3 scripts/diagnostics/subgroup_analysis.py       # AUC by age band, sex, BMI category (with 1,000-iter bootstrap CIs)

# Raw-data exploration (pre-modelling)
python3 scripts/diagnostics/raw_data_exploration.py    # cohort composition, Bray-Curtis PCoA, Shannon, depth, top-species heatmap
                                                       # -> results/diagnostics/raw_data_summary.csv + RAW_DATA_PATTERNS.md
```

## Submission build

```bash
python3 manuscript/markdown/_build_docx.py      # rebuild manuscript_complete.md + the 10 .docx section files
python3 scripts/build_supplementary_tables.py   # rebuild results/supplementary/S*.csv + INDEX.csv
python3 scripts/build_submission.py             # assemble submission/build/ + SHA-256 manifest + SUBMISSION_BUNDLE.zip
python3 scripts/build_biorxiv_pdf.py            # produce a single uploadable PDF for bioRxiv
python3 scripts/test_submission_build.py        # integration test for the build pipeline (marked pytest.integration)
```

## Utility scripts (run any time)

```bash
python3 scripts/sanity_check.py        # merged-file shapes, label / study_condition counts, NaN scan
python3 scripts/find_nans.py           # NaN scan across species + pathway + metadata
python3 scripts/check_label_dist.py    # per-cohort label distribution and CSV cohort-order check
```

`scripts/lodo_cv.py` is a library module (imported by every training
script and by `tests/test_lodo_cv.py`); it is not invoked directly.

## Country-aware LODO

When a cohort is held out as the test fold, all training-set cohorts from the same country are excluded. Affected pairs:

- **ThomasAM_2019_c (JPN) ↔ YachidaS_2019 (JPN)**: without this fix ThomasAM_2019_c achieves AUC=0.999 due to geographic signal leakage. With fix: AUC=0.836.
- **ThomasAM_2018a (ITA) ↔ ThomasAM_2018b (ITA)**: each excluded from the other's training fold.

## Key design decisions

See `results/decisions_addendum.md` for the complete log covering: SMOTE vs class weights, DeLong implementation, normalization strategy, per-fold vs global pathway filtering, hyperparameter tuning rationale, HanniganGD_2017 exclusion, and more.
