# Submission Package Manifest

Generated: 2026-05-18T11:56:05+00:00
Finished:  2026-05-18T11:56:06+00:00

Master index of everything in the submission package. SHA-256 digests of all final deliverable PDFs and ZIPs are included for integrity tracking.

## `submission/` (scaffolding, human-edited)

| File | Purpose |
|---|---|
| `submission/00_cover_letter.md` | Editable cover-letter template. |
| `submission/01_data_availability.md` | Data Availability statement. |
| `submission/02_ethics_statement.md` | Ethics statement. |
| `submission/03_author_contributions.md` | Author contributions, funding, competing interests. |
| `submission/04_submission_checklist.md` | Pre-submission readiness checklist. |
| `submission/05_biorxiv_metadata.md` | bioRxiv submission-form fields. |
| `submission/MANIFEST.md` | (this file) |

## `submission/build/` (auto-generated)

| Relative path | Size | Note |
|---|---:|---|
| `00_cover_letter.md` | 4.1 KB | submission scaffolding |
| `01_data_availability.md` | 2.0 KB | submission scaffolding |
| `02_ethics_statement.md` | 1.1 KB | submission scaffolding |
| `03_author_contributions.md` | 1.3 KB | submission scaffolding |
| `04_submission_checklist.md` | 4.0 KB | submission scaffolding |
| `05_biorxiv_metadata.md` | 1.5 KB | submission scaffolding |
| `CRC_Manuscript_Complete.docx` | 57.6 KB | manuscript (Word) |
| `README.md` | 4.2 KB | bundle README |
| `SUBMISSION_BUNDLE.zip` | 3.5 MB | final upload archive |
| `figures/Figure1_Forest_Plot.pdf` | 34.8 KB | figure (unmerged) |
| `figures/Figure2_ROC_Curves.pdf` | 25.4 KB | figure (unmerged) |
| `figures/Figure3_SHAP_Importance.pdf` | 38.5 KB | figure (unmerged) |
| `figures/Figure4_Three_Panel_SHAP.pdf` | 42.2 KB | figure (unmerged) |
| `figures/base_rate_ppv.png` | 246.9 KB | figure (unmerged) |
| `figures/calibration_mechanism.png` | 326.5 KB | figure (unmerged) |
| `figures/calibration_reliability.png` | 289.3 KB | figure (unmerged) |
| `figures/confusion_matrices.png` | 363.6 KB | figure (unmerged) |
| `figures/depth_vs_fnucleatum_shap.png` | 180.8 KB | figure (unmerged) |
| `figures/main/Figure1_Forest_Plot.pdf` | 34.8 KB | Figure1_Forest_Plot (PDF) |
| `figures/main/Figure1_Forest_Plot.png` | 216.6 KB | Figure1_Forest_Plot (PNG) |
| `figures/main/Figure2_ROC_Curves.pdf` | 25.4 KB | Figure2_ROC_Curves (PDF) |
| `figures/main/Figure2_ROC_Curves.png` | 152.3 KB | Figure2_ROC_Curves (PNG) |
| `figures/main/Figure3_SHAP_Importance.pdf` | 38.5 KB | Figure3_SHAP_Importance (PDF) |
| `figures/main/Figure3_SHAP_Importance.png` | 396.3 KB | Figure3_SHAP_Importance (PNG) |
| `figures/main/Figure4_Three_Panel_SHAP.pdf` | 42.2 KB | Figure4_Three_Panel_SHAP (PDF) |
| `figures/main/Figure4_Three_Panel_SHAP.png` | 385.7 KB | Figure4_Three_Panel_SHAP (PNG) |
| `figures/per_cohort_sens_spec.png` | 271.2 KB | figure (unmerged) |
| `figures/permutation_vs_shap.png` | 237.3 KB | figure (unmerged) |
| `figures/roc_pr_pooled.png` | 289.4 KB | figure (unmerged) |
| `figures/sens_at_fixed_spec.png` | 162.0 KB | figure (unmerged) |
| `figures/subgroup_auc.png` | 247.3 KB | figure (unmerged) |
| `manuscript_complete.md` | 56.4 KB | manuscript (Markdown source) |
| `supplementary/INDEX.csv` | 955.0 B | supplementary table |
| `supplementary/S10_delong.csv` | 235.0 B | supplementary table |
| `supplementary/S1_cohort_overview.csv` | 990.0 B | supplementary table |
| `supplementary/S2_per_fold_aucs.csv` | 755.0 B | supplementary table |
| `supplementary/S3_top_shap_features.csv` | 18.4 KB | supplementary table |
| `supplementary/S4_bootstrap_ci.csv` | 1.5 KB | supplementary table |
| `supplementary/S5_sensitivity_grid.csv` | 800.0 B | supplementary table |
| `supplementary/S6_adenoma_lodo.csv` | 109.0 B | supplementary table |
| `supplementary/S7_confounder_adjustment.csv` | 98.0 B | supplementary table |
| `supplementary/S8_seed_sensitivity.csv` | 212.0 B | supplementary table |
| `supplementary/S9_external_validation.csv` | 121.0 B | supplementary table |
| `supplementary/Supplementary_Tables.docx` | 38.9 KB | supplementary tables (Word) |

**Total bundle size:** 7.7 MB

## SHA-256 digests (final deliverables)

| File | SHA-256 |
|---|---|
| `SUBMISSION_BUNDLE.zip` | `e295ef5158f883e5e712d4610639ee80aff916b4b747099f67bad33ac12f88f1` |

## Warnings during build

- Could not generate manuscript.pdf: neither pandoc nor soffice produced output. Install pandoc (with a LaTeX engine) or LibreOffice (`soffice`) and rerun.
- Neither pypdf nor PyPDF2 is installed; cannot merge figure PDFs. Install with `pip install pypdf img2pdf` and rerun. Falling back to copying figures individually into build/figures/.

## Regenerate

```bash
python scripts/build_submission.py
python scripts/build_biorxiv_pdf.py   # single bioRxiv PDF
```
