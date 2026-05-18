# Reproducing the Analyses

See README.md and results/decisions_addendum.md for the canonical pipeline and all analytical decisions.

## Environment

```bash
pip install -r requirements.lock
```

All scripts use `random_state=42` and produce deterministic results. Total runtime is approximately 45 minutes on a standard workstation (dominated by LODO CV training).

## Step-by-step pipeline

### 1. Data export (R)

```bash
Rscript scripts/export_data.R          # pulls curatedMetagenomicData; exports 11 cohorts (~1604 samples)
Rscript scripts/audit_subject_ids.R    # verifies no duplicated subject IDs across cohorts
```

### 2. Data processing

```bash
python3 scripts/merge_pathways.py      # concatenates per-cohort HUMAnN pathway chunks
python3 scripts/validate_pathways.py   # sanity-checks pathway matrix dimensions
python3 scripts/filter_pathways.py     # global static filter (used by SHAP scripts only)
python3 scripts/preprocessing.py       # quality filters: excludes HanniganGD_2017 (low depth),
                                       # drops samples <1M reads; outputs 1522 samples, 10 cohorts
python3 scripts/generate_table1.py     # Table 1 demographics
python3 scripts/adenoma_counts.py      # per-cohort adenoma sample counts
```

### 3. Main LODO classification

```bash
python3 scripts/train_baseline.py   # species-only RF LODO (country-aware)
                                    # expect: per-cohort mean AUC ~0.807, pooled AUC ~0.781

python3 scripts/train_joint.py      # joint RF + XGBoost LODO (country-aware, per-fold pathway filter)
                                    # expect: Joint RF per-cohort ~0.804 (pooled ~0.756)
                                    #         Joint XGB per-cohort ~0.797 (pooled ~0.766)
                                    # 551 pathway candidates; 402-406 retained per fold

python3 scripts/auc_comparison.py   # paired tests (t, Wilcoxon) + DeLong on pooled predictions
                                    # expect: species_rf vs joint_rf DeLong z=3.35, p=0.0008
                                    #         species_rf vs joint_xgb DeLong z=2.00, p=0.046
```

### 4. Feature importance (SHAP)

```bash
python3 scripts/shap_analysis.py    # RF TreeSHAP for CRC vs control
python3 scripts/shap_xgb.py         # XGBoost TreeSHAP for all three tasks
python3 scripts/shap_adenoma.py     # RF TreeSHAP for adenoma tasks (H-vs-A, A-vs-CRC)
```

### 5. Adenoma classification

```bash
python3 scripts/adenoma_lodo.py     # cross-cohort LODO across 4 adenoma-containing cohorts
                                    # (FengQ_2015, ZellerG_2014, ThomasAM_2018a, YachidaS_2019)
                                    # expect: H-vs-A RF ~0.561, H-vs-A XGB ~0.579
                                    #         A-vs-CRC RF ~0.671, A-vs-CRC XGB ~0.617
```

### 6. Biologically-guided pathway shortlist

```bash
python3 scripts/bio_pathway_shortlist.py  # keyword-selected 86 unique CRC-relevant pathways across 9 groups
                                          # expect: mean per-cohort LODO AUC ~0.817 (vs species-only 0.807)
```

### 7. Robustness battery

```bash
python3 scripts/bootstrap_ci.py          # 10,000-resample bootstrap 95% CIs (cohort-stratified pooled)
                                         # expect: species RF pooled 0.781 [0.757, 0.805]

python3 scripts/seed_sensitivity.py      # seeds {0,1,2,42,100}; expect spread < 0.005

python3 scripts/sensitivity_analysis.py  # 4x5 prevalence/mean grid (country-aware, per-fold filter)
                                         # expect: joint RF mean per-cohort AUC range 0.794-0.812

python3 scripts/confounder_adjustment.py # direct inclusion + residualization of age, sex, BMI (country-aware)
                                         # expect: per-cohort AUC 0.800-0.814 (within noise of unadjusted 0.807)

python3 scripts/batch_correction.py      # per-fold ComBat on species features (country-aware LODO)
                                         # requires: pip install combat
                                         # expect: mean per-cohort AUC ~0.815 (vs uncorrected ~0.807)
```

### 8. Figures and verification

```bash
python3 scripts/generate_figures.py          # draft figures
python3 scripts/figure1_forest_plot.py       # Figure 1: forest plot of per-cohort + pooled CIs
python3 scripts/figure5_shap_three_panel.py  # Figure 4: three-panel SHAP (H-vs-A, CRC-vs-control, A-vs-CRC)
python3 scripts/verify_results.py            # smoke-tests all headline numbers against saved CSVs
```

## Sanity checks (run any time)

```bash
python3 scripts/sanity_check.py
python3 scripts/find_nans.py
python3 scripts/check_label_dist.py
```

## Country-aware LODO

When a cohort is held out as the test fold, all training-set cohorts from the same country are excluded. Affected pairs:

- **ThomasAM_2019_c (JPN) ↔ YachidaS_2019 (JPN)**: without this fix ThomasAM_2019_c achieves AUC=0.999 due to geographic signal leakage. With fix: AUC=0.836.
- **ThomasAM_2018a (ITA) ↔ ThomasAM_2018b (ITA)**: each excluded from the other's training fold.

## Key design decisions

See `results/decisions_addendum.md` for the complete log covering: SMOTE vs class weights, DeLong implementation, normalization strategy, per-fold vs global pathway filtering, hyperparameter tuning rationale, HanniganGD_2017 exclusion, and more.
