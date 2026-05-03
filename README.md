# Species-level taxonomic features alone outperform joint species-plus-pathway models for colorectal cancer detection

**Alejandro Velazquez and Rachel Selbrede**

A rigorous multi-cohort re-evaluation of the Thomas et al. (2019) CRC classification framework, demonstrating that species-only Random Forest classifiers significantly outperform joint species-plus-pathway models under leave-one-dataset-out (LODO) cross-validation.

## Key finding

Species-only RF achieves a pooled LODO AUC of **0.810** (95% CI: 0.776 to 0.841, 10,000 cohort-stratified bootstrap resamples), significantly outperforming:
- Joint species+pathway RF: AUC 0.776 (DeLong z = 2.88, p = 0.004)
- Joint species+pathway XGBoost: AUC 0.781 (DeLong z = 2.65, p = 0.008)

This result is stable across random seeds (0.805 +/- 0.002), filter thresholds (joint RF mean per-cohort AUC 0.773 to 0.789 across the 16 substantively non-degenerate cells of the prevalence x mean grid; full 20-cell range 0.773 to 0.811, with the upper bound coming from the degenerate mean >= 1e-3 column that retains only two pathways), and confounder adjustments (age, sex, BMI).

## Data

- **Source**: curatedMetagenomicData (Bioconductor)
- **Cohorts**: 7 (FengQ_2015, YuJ_2015, VogtmannE_2016, ZellerG_2014, ThomasAM_2018a, ThomasAM_2018b, ThomasAM_2019_c)
- **Subjects**: 762 unique (326 CRC, 116 adenoma, 320 controls); the metadata `study_condition` field uses the value `control` (not `healthy`)
- **Species features**: 247 (MetaPhlAn, prevalence >= 10%, mean >= 1e-4, log10-transformed)
- **Pathway features**: 540 unstratified candidates (HUMAnN); 402 to 406 retained per LODO fold after per-fold prevalence/mean filtering

## Manuscript

The complete manuscript is in `manuscript/`:
- `CRC_Manuscript_Complete.docx` (single merged document)
- Individual section files (Title Page, Abstract, Introduction, Methods, Results, Discussion, References, Table 1, Supplementary Tables)
- `figures/` (Figures 1 to 4 in PNG 300 DPI and PDF). Figure 1 = forest plot of per-cohort and pooled LODO AUCs; Figure 2 = ROC curves; Figure 3 = TreeSHAP top-species importance for CRC; Figure 4 = three-panel TreeSHAP across the adenoma-carcinoma sequence (healthy-vs-adenoma | CRC-vs-healthy | adenoma-vs-CRC)

## Reproducing the analyses

See `REPRODUCING.md` for the full step-by-step pipeline. Quick summary:

```bash
pip install -r requirements.txt
Rscript scripts/export_data.R
python3 scripts/preprocessing.py
python3 scripts/train_baseline.py        # Species-only RF LODO
python3 scripts/train_joint.py           # Joint RF + XGBoost LODO
python3 scripts/auc_comparison.py        # DeLong tests
python3 scripts/bootstrap_ci.py          # 95% CIs
python3 scripts/shap_analysis.py         # Feature importance
python3 scripts/verify_results.py        # Smoke-test headline numbers
```

All scripts use `random_state=42` and produce deterministic results. Total runtime is approximately 30 minutes on a standard workstation.

## Robustness battery

- Filter threshold sensitivity (20-combination grid)
- Confounder assessment (direct inclusion + residualization)
- Random seed stability (5 seeds)
- Bootstrap confidence intervals (2,000 resamples)
- Per-fold ComBat batch correction
- Adenoma classification (exploratory, underpowered)

## Key files

| Path | Description |
|------|-------------|
| `data/processed/species_filtered.csv` | 247 species features |
| `data/processed/pathway_unstratified.csv` | 540 pathway candidates |
| `results/preds_species_rf.csv` | Per-sample LODO predictions (species RF) |
| `results/preds_joint_rf.csv` | Per-sample LODO predictions (joint RF) |
| `results/preds_joint_xgb.csv` | Per-sample LODO predictions (joint XGBoost) |
| `results/delong_results.csv` | DeLong test statistics |
| `results/shap_crc_features.csv` | SHAP values (RF) |
| `results/shap_crc_xgb.csv` | SHAP values (XGBoost) |
| `results/decisions_addendum.md` | Decision log for all analytical choices |

## License

MIT
