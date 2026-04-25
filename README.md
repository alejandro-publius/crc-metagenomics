# CRC Metagenomics

Extension of Thomas et al. 2019: CRC classification from gut microbiome species and pathway abundance, with adenoma classification as a separate task.

## Data
- Source: curatedMetagenomicData (Bioconductor)
- 7 cohorts, 762 unique subjects, 646 used in joint CRC training
- Species: 247 features (MetaPhlAn)
- Pathways: 405 unstratified features (HUMAnN, prevalence>=10%, mean>=1e-6)

## Pipeline
1. `scripts/audit_subject_ids.R` - confirm no cross-cohort overlap
2. `scripts/merge_pathways.py` - merge per-cohort pathway chunks
3. `scripts/validate_pathways.py` - audit raw pathway file
4. `scripts/filter_pathways.py` - prevalence/abundance filter
5. `scripts/train_joint.py` - LODO RF + XGBoost on species+pathway
6. `scripts/auc_comparison.py` - paired tests + bootstrap CIs vs species baseline
7. `scripts/shap_analysis.py`, `scripts/shap_xgb.py` - feature importance
8. `scripts/train_adenoma.py` - 5-fold CV for adenoma tasks
9. `scripts/shap_adenoma.py` - adenoma SHAP
10. `scripts/generate_figures.py` - paper figures

## Key results
- Species-only RF (LODO): AUC 0.803
- Joint RF (LODO): AUC 0.783
- Joint XGBoost (LODO): AUC 0.790
- Adenoma vs healthy XGBoost (5-fold): AUC 0.709
- Adenoma vs CRC XGBoost (5-fold): AUC 0.809
- Joint vs species baseline: not statistically significant (paired t p>0.4)

## Reproducibility
All scripts use random_state=42. Verified identical AUCs across reruns.

## Key files
- `data/processed/species_filtered.csv` - species relative abundance
- `data/processed/pathway_unstratified.csv` - pathway relative abundance
- `data/processed/metadata_clean.csv` - sample metadata
- `results/` - AUCs, SHAP rankings, comparison tests, decisions logs
- `figures/` - fig1-fig4

## Decisions
See `results/decisions_addendum.md` and `results/adenoma_go_nogo_memo.md`.

## Requirements
See `requirements.txt`. R packages installed via BiocManager (curatedMetagenomicData).
