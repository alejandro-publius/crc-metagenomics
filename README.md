# Species-level taxonomic features alone outperform joint species-plus-pathway models for colorectal cancer detection

**Alejandro Velazquez and Rachel Selbrede**

A rigorous multi-cohort re-evaluation of the Thomas et al. (2019) CRC classification framework, demonstrating that species-only Random Forest classifiers significantly outperform joint species-plus-pathway models under leave-one-dataset-out (LODO) cross-validation across 10 independent cohorts.

## Key finding

Species-only RF achieves a pooled LODO AUC of **0.781** (95% CI: 0.756 to 0.805), significantly outperforming:
- Joint species+pathway RF: AUC 0.756 (DeLong z = 3.35, p = 0.0008)
- Joint species+pathway XGBoost: AUC 0.766 (DeLong z = 2.00, p = 0.046)

This result is stable across random seeds (0.810 +/- 0.002), filter thresholds (AUC range 0.798 to 0.810), and confounder adjustments (age, sex, BMI; AUC 0.807 to 0.816).

## Data

- **Source**: curatedMetagenomicData (Bioconductor)
- **Cohorts**: 10 (FengQ_2015, GuptaA_2019, ThomasAM_2018a, ThomasAM_2018b, ThomasAM_2019_c, VogtmannE_2016, WirbelJ_2018, YachidaS_2019, YuJ_2015, ZellerG_2014)
- **Subjects**: 1,522 unique (674 CRC, 183 adenoma, 665 healthy controls); the metadata `study_condition` field uses the value `control` (not `healthy`)
- **Species features**: 229 (MetaPhlAn, prevalence >= 10%, mean >= 1e-4, log10-transformed)
- **Pathway features**: 551 unstratified candidates (HUMAnN); 402 to 406 retained per LODO fold after per-fold prevalence/mean filtering
- **Excluded**: HanniganGD_2017 (mean depth 6.5M reads vs 40-102M for other cohorts; 82% species feature sparsity)

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

## Methodological contributions

- **Country-aware LODO**: when a cohort is held out as the test fold, all cohorts from the same country are excluded from training. This prevents population-level confounding — without this fix, ThomasAM_2019_c (Japan) achieved AUC=0.999 due to YachidaS_2019 (Japan) in the training set; corrected AUC is 0.836.
- **Biologically-guided pathway shortlist**: 84 CRC-relevant pathways selected by keyword matching across 7 biological groups (butyrate/SCFA, fermentation, LPS/inflammation, polyamine, tryptophan, folate/one-carbon, sulfur/methionine). Mean LODO AUC 0.817, comparable to species-only baseline.

## Robustness battery

- Country-aware leave-one-dataset-out cross-validation (10 cohorts)
- Filter threshold sensitivity (20-combination grid: 4 prevalence × 5 mean thresholds)
- Confounder assessment (direct inclusion + residualization of age, sex, BMI)
- Random seed stability (5 seeds: AUC 0.810 +/- 0.002)
- Bootstrap confidence intervals (2,000 resamples)
- Per-fold ComBat batch correction
- Biologically-guided pathway feature subset
- Adenoma classification LODO (4 cohorts, 183 adenoma samples)

## Key files

| Path | Description |
|------|-------------|
| `data/processed/species_filtered.csv` | 229 species features |
| `data/processed/pathway_unstratified.csv` | 551 pathway candidates |
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

## License

MIT
