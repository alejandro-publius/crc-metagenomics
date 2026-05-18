# Submission Package Manifest

Generated: 2026-05-18T13:48:51+00:00
Finished:  2026-05-18T13:48:55+00:00

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
| `submission/06_reviewer_responses.md` | Anticipated reviewer-response prep doc. |
| `submission/07_pre_submission_qa.md` | Internal pre-submission self-critique. |
| `submission/MANIFEST.md` | (this file) |

## `submission/build/` (auto-generated)

| Relative path | Size | Note |
|---|---:|---|
| `00_cover_letter.md` | 4.4 KB | submission scaffolding |
| `01_data_availability.md` | 2.0 KB | submission scaffolding |
| `02_ethics_statement.md` | 1.1 KB | submission scaffolding |
| `03_author_contributions.md` | 1.3 KB | submission scaffolding |
| `04_submission_checklist.md` | 4.0 KB | submission scaffolding |
| `05_biorxiv_metadata.md` | 1.6 KB | submission scaffolding |
| `06_reviewer_responses.md` | 34.1 KB | submission scaffolding |
| `07_pre_submission_qa.md` | 10.7 KB | submission scaffolding |
| `CRC_Manuscript_Complete.docx` | 58.3 KB | manuscript (Word) |
| `README.md` | 4.7 KB | bundle README |
| `SUBMISSION_BUNDLE.zip` | 6.8 MB | final upload archive |
| `figures.pdf` | 6.5 MB | merged figures PDF |
| `figures/main/Figure1_Forest_Plot.pdf` | 34.8 KB | Figure1_Forest_Plot (PDF) |
| `figures/main/Figure1_Forest_Plot.png` | 216.6 KB | Figure1_Forest_Plot (PNG) |
| `figures/main/Figure2_ROC_Curves.pdf` | 25.4 KB | Figure2_ROC_Curves (PDF) |
| `figures/main/Figure2_ROC_Curves.png` | 152.3 KB | Figure2_ROC_Curves (PNG) |
| `figures/main/Figure3_SHAP_Importance.pdf` | 38.5 KB | Figure3_SHAP_Importance (PDF) |
| `figures/main/Figure3_SHAP_Importance.png` | 396.3 KB | Figure3_SHAP_Importance (PNG) |
| `figures/main/Figure4_Three_Panel_SHAP.pdf` | 42.2 KB | Figure4_Three_Panel_SHAP (PDF) |
| `figures/main/Figure4_Three_Panel_SHAP.png` | 385.7 KB | Figure4_Three_Panel_SHAP (PNG) |
| `manuscript_complete.md` | 58.5 KB | manuscript (Markdown source) |
| `supplementary/INDEX.csv` | 1.2 KB | supplementary table |
| `supplementary/S10_delong.csv` | 235.0 B | supplementary table |
| `supplementary/S11_methods_comparison.csv` | 3.4 KB | supplementary table |
| `supplementary/S1_cohort_overview.csv` | 990.0 B | supplementary table |
| `supplementary/S2_per_fold_aucs.csv` | 755.0 B | supplementary table |
| `supplementary/S3_top_shap_features.csv` | 18.4 KB | supplementary table |
| `supplementary/S4_bootstrap_ci.csv` | 1.5 KB | supplementary table |
| `supplementary/S5_sensitivity_grid.csv` | 800.0 B | supplementary table |
| `supplementary/S6_adenoma_lodo.csv` | 109.0 B | supplementary table |
| `supplementary/S7_confounder_adjustment.csv` | 98.0 B | supplementary table |
| `supplementary/S8_seed_sensitivity.csv` | 105.0 B | supplementary table |
| `supplementary/S8b_seed_sensitivity_summary.csv` | 106.0 B | supplementary table |
| `supplementary/S9_external_validation.csv` | 121.0 B | supplementary table |
| `supplementary/Supplementary_Tables.docx` | 38.9 KB | supplementary tables (Word) |

**Total bundle size:** 14.8 MB

## SHA-256 digests (final deliverables)

| File | SHA-256 |
|---|---|
| `figures.pdf` | `ff780ffe637116eb540a9ef414623131b3c95164f2e539c269c5754933e6c46a` |
| `SUBMISSION_BUNDLE.zip` | `ccc24ff6dc3f1637fb1c89d8d6b00f28161a259737dc057e421816b1bfd9083d` |

## Warnings during build

- pandoc failed (rc=1); stderr head: 'pandoc: Uncaught exception ghc-internal:GHC.Internal.IO.Exception.IOException:\n\nxelatex: createProcess: find_executable: failed (errnoToString failed)\n\nWhile handling xelatex: createProcess: find_executable: failed (errnoToString failed)\n\nHasCallStack backtrace:\n  throwIO, called at src/Text/Pandoc/'
- pandoc default engine also failed: 'pandoc: Uncaught exception ghc-internal:GHC.Internal.IO.Exception.IOException:\n\npdflatex: createProcess: find_executable: failed (errnoToString failed)\n\nWhile handling pdflatex: createProcess: find_executable: failed (errnoToString failed)\n\nHasCallStack backtrace:\n  throwIO, called at src/Text/Pando'
- Could not generate manuscript.pdf: neither pandoc nor soffice produced output. Install pandoc (with a LaTeX engine) or LibreOffice (`soffice`) and rerun.

## Regenerate

```bash
python scripts/build_submission.py
python scripts/build_biorxiv_pdf.py   # single bioRxiv PDF
```
