# Species-level taxonomic features alone outperform joint species-plus-pathway models for colorectal cancer detection

**Alejandro Velazquez and Rachel Selbrede**

A rigorous multi-cohort re-evaluation of the Thomas et al. (2019) CRC classification framework, demonstrating that species-only Random Forest classifiers significantly outperform joint species-plus-pathway models under leave-one-dataset-out (LODO) cross-validation across 10 independent cohorts.

## Key finding

Species-only RF achieves a pooled LODO AUC of **0.781** (95% bootstrap CI: 0.757 to 0.805; 10,000 cohort-stratified resamples on n = 1,339 pooled held-out predictions), significantly outperforming:
- Joint species+pathway RF: AUC 0.756 (DeLong z = 3.35, p = 0.0008)
- Joint species+pathway XGBoost: AUC 0.766 (DeLong z = 2.00, p = 0.046)

This result is stable across random seeds (mean per-cohort AUC 0.810 +/- 0.002 across 5 seeds), filter thresholds (joint RF per-cohort AUC 0.794 to 0.812 across a 4x5 prevalence-by-mean grid; spread 0.018), and confounder adjustments (age, sex, BMI; mean per-cohort AUC 0.800 to 0.814 across direct/residualized x RF/XGB cells, vs unadjusted species-RF baseline 0.807).

## Data

- **Source**: curatedMetagenomicData (Bioconductor)
- **Cohorts**: 10 (FengQ_2015, GuptaA_2019, ThomasAM_2018a, ThomasAM_2018b, ThomasAM_2019_c, VogtmannE_2016, WirbelJ_2018, YachidaS_2019, YuJ_2015, ZellerG_2014)
- **Subjects**: 1,522 unique (674 CRC, 183 adenoma, 665 healthy controls); the metadata `study_condition` field uses the value `control` (not `healthy`)
- **Species features**: 229 (MetaPhlAn, global filter: prevalence >= 10%, mean >= 1e-4; per-sample row-sum renormalization when input is on a percentage scale, then log10(x + 1e-6))
- **Pathway features**: 551 unstratified candidates (HUMAnN, relative abundance, no transform); 402 to 406 retained per LODO fold after per-fold prevalence (>= 10%) and mean (>= 1e-6) filter computed on training-cohort samples only
- **Excluded**: HanniganGD_2017 (mean depth 6.5M reads vs 9.2M-102M per-cohort means for the other 10 cohorts in the candidate set; 82% species feature sparsity vs 61% mean for the other cohorts)

## Manuscript

The complete manuscript is in `manuscript/`:
- `CRC_Manuscript_Complete.docx` (single merged document)
- Individual section files (Title Page, Abstract, Introduction, Methods, Results, Discussion, References, Table 1, Supplementary Tables)
- `figures/` (Figures 1 to 4 in PNG 300 DPI and PDF). Figure 1 = forest plot of per-cohort and pooled LODO AUCs; Figure 2 = ROC curves; Figure 3 = TreeSHAP top-species importance for CRC; Figure 4 = three-panel TreeSHAP across the adenoma-carcinoma sequence (healthy-vs-adenoma | CRC-vs-healthy | adenoma-vs-CRC)

## Reproducing the analyses

See `REPRODUCING.md` for the full step-by-step pipeline. Quick summary:

```bash
pip install -r requirements.lock
Rscript scripts/export_data.R
python3 scripts/preprocessing.py
python3 scripts/train_baseline.py        # Species-only RF LODO
python3 scripts/train_joint.py           # Joint RF + XGBoost LODO
python3 scripts/auc_comparison.py        # DeLong tests
python3 scripts/bootstrap_ci.py          # 95% CIs
python3 scripts/shap_analysis.py         # Feature importance
python3 scripts/bio_pathway_shortlist.py # Biologically-guided pathway subset
python3 scripts/adenoma_lodo.py          # Adenoma cross-cohort LODO
python3 scripts/verify_results.py        # Smoke-test headline numbers
```

All scripts use `random_state=42` and produce deterministic results. Total runtime is approximately 45 minutes on a standard workstation.

## Use this as a library

The country-aware LODO loop, per-fold pathway filter, and statistical helpers (DeLong test, cohort-stratified bootstrap CI) are packaged as `crc_lodo_bench` for reuse on other shotgun-metagenomic cohorts.

```bash
pip install -e .
```

```python
from crc_lodo_bench import run_lodo_cv, per_fold_pathway_filter, bootstrap_pooled_ci

filt = per_fold_pathway_filter(pathway_cols, passthrough_cols=species_cols)
results = run_lodo_cv(make_rf, X, y, metadata, country_col="country", feature_filter_fn=filt)
ci = bootstrap_pooled_ci(preds["y_true"], preds["y_prob"], preds["cohort"])
```

See `src/crc_lodo_bench/README.md` for the full API and a runnable example.

## Methodological contributions

- **Country-aware LODO**: when a cohort is held out as the test fold, all cohorts from the same country are excluded from training. This prevents population-level confounding — without this fix, ThomasAM_2019_c (Japan) achieved AUC=0.999 due to YachidaS_2019 (Japan) in the training set; corrected AUC is 0.836.
- **Biologically-guided pathway shortlist**: 86 unique CRC-relevant pathways selected by keyword matching across 9 biological groups (butyrate/SCFA, fermentation, LPS/inflammation, polyamine, tryptophan, folate/one-carbon, sulfur/methionine, glycan/mucin, bile-acid metabolism). Mean per-cohort LODO AUC 0.817, comparable to the species-only baseline (0.807).

## Robustness battery

- Country-aware leave-one-dataset-out cross-validation (10 cohorts)
- Filter threshold sensitivity (4 x 5 prevalence-by-mean grid; joint RF per-cohort AUC 0.794 to 0.812, spread 0.018)
- Confounder assessment (direct inclusion + residualization of age, sex, BMI; per-cohort AUC 0.800 to 0.814 across the four cells)
- Random seed stability (5 seeds {0, 1, 2, 42, 100}; per-cohort AUC 0.810 +/- 0.002, range 0.807 to 0.811)
- Bootstrap confidence intervals (10,000 resamples; cohort-stratified for the pooled CI)
- Per-fold ComBat batch correction (mean per-cohort AUC 0.815, vs uncorrected 0.807)
- Biologically-guided pathway feature subset (mean per-cohort AUC 0.817)
- Adenoma classification LODO (4 cohorts, 183 adenoma samples)

## Key files

| Path | Description |
|------|-------------|
| `data/processed/species_filtered.csv` | 229 species features (post global filter, log10-transformed) |
| `data/raw/pathway_abundance.csv` | Merged HUMAnN pathways (38,690 cols; 551 unstratified) -- input to per-fold filtering in train_joint.py |
| `data/processed/pathway_unstratified.csv` | 403 pathways after the static global filter (used by SHAP / adenoma scripts only) |
| `results/preds_species_rf.csv` | Per-sample LODO predictions (species RF) |
| `results/preds_joint_rf.csv` | Per-sample LODO predictions (joint RF) |
| `results/preds_joint_xgb.csv` | Per-sample LODO predictions (joint XGBoost) |
| `results/delong_results.csv` | DeLong test statistics |
| `results/bootstrap_ci.csv` | 95% bootstrap confidence intervals |
| `results/shap_crc_features.csv` | SHAP values (RF) |
| `results/shap_crc_xgb.csv` | SHAP values (XGBoost) |
| `results/bio_pathway_results.csv` | Biologically-guided pathway LODO results |
| `results/adenoma_lodo_results.csv` | Adenoma cross-cohort LODO results |
| `results/decisions_addendum.md` | Decision log for all analytical choices |

## Diagnostics and audits

Narrative summaries and read-only audit reports live alongside the
results CSVs:

- [`results/diagnostics/README.md`](results/diagnostics/README.md) — index of the 12 standalone diagnostic scripts and their outputs.
- [`results/diagnostics/ROBUSTNESS_SUMMARY.md`](results/diagnostics/ROBUSTNESS_SUMMARY.md) — TreeSHAP-bias, sequencing-depth, and calibration-mechanism cross-checks for the species RF.
- [`results/diagnostics/CLINICAL_TRANSLATION_SUMMARY.md`](results/diagnostics/CLINICAL_TRANSLATION_SUMMARY.md) — sensitivity at fixed specificity, base-rate-adjusted PPV/NPV, and the head-to-head with FIT.
- [`results/diagnostics/RAW_DATA_PATTERNS.md`](results/diagnostics/RAW_DATA_PATTERNS.md) — pre-modelling exploratory patterns (cohort composition, Bray-Curtis PCoA, alpha diversity, top species, sequencing depth).
- [`results/CITATION_AUDIT.md`](results/CITATION_AUDIT.md) — audit of every reference in `manuscript/markdown/06_references.md`.
- [`results/decisions_addendum.md`](results/decisions_addendum.md) — decision log for all analytical choices (SMOTE vs class weights, DeLong implementation, per-fold filtering, cohort exclusions, bile-acid pathway group, and more).
- [`results/baseline_results.md`](results/baseline_results.md) — narrative summary of the species-only RF per-cohort and pooled results.

## License

MIT
