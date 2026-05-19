# Submission Readiness Checklist

Internal checklist before submission to any journal. Update each item to
**DONE / PARTIAL / TODO** before pressing submit.

## Manuscript content
- [ ] Title (max 25 words, no abbreviations) — `manuscript/CRC_Title_Page.docx`
- [ ] Abstract (structured, ~250 words; Background, Methods, Results, Conclusions) — `manuscript/CRC_Abstract.docx`
- [ ] Introduction (3-5 paragraphs; ends with explicit study aims) — `manuscript/CRC_Introduction.docx`
- [ ] Methods (full reproducibility detail; cites curatedMetagenomicData, MetaPhlAn, HUMAnN, RF/XGB hyperparameters) — `manuscript/CRC_Methods.docx`
- [ ] Results (in same order as figures; reports per-cohort AND pooled AUCs; DeLong statistics; per-cohort paired tests) — `manuscript/CRC_Results.docx`
- [ ] Discussion (interpretation; pathway negative result framed honestly; oral-CRC biology; cross-cohort generalization; limitations) — `manuscript/CRC_Discussion.docx`
- [ ] References (cite Thomas et al. 2019, Wirbel et al. 2019, Pasolli et al. 2017, Truong et al. 2015, Franzosa et al. 2018, DeLong 1988, Sun and Xu 2014) — `manuscript/CRC_References.docx`

## Figures
- [ ] Figure 1 — LODO per-cohort AUC forest plot — `manuscript/figures/Figure1_Forest_Plot.{pdf,png}`
- [ ] Figure 2 — ROC curves for species RF, joint RF, joint XGB on pooled predictions — `manuscript/figures/Figure2_ROC_Curves.{pdf,png}`
- [ ] Figure 3 — SHAP importance for CRC vs control — `manuscript/figures/Figure3_SHAP_Importance.{pdf,png}`
- [ ] Figure 4 — Three-panel SHAP (CRC vs control, H vs A, A vs CRC) — `manuscript/figures/Figure4_Three_Panel_SHAP.{pdf,png}`
- [ ] All figures vector-format-available (PDF), font embedded, ≥300 DPI raster
- [ ] All axis labels, legends legible at print size

## Tables
- [ ] Table 1 — Cohort overview — `manuscript/CRC_Table1.docx` (source `results/table1.csv`)
- [ ] Supplementary Tables S1 through S11 (plus S8b) — `results/supplementary/` (see `INDEX.csv`)

## Statistics reporting
- [ ] Exact p-values reported (not p < 0.05)
- [ ] Effect sizes with 95% CIs
- [ ] DeLong test statistics include z and exact p
- [ ] Multiple-testing correction noted where applicable
- [ ] Bootstrap CI iteration count specified (10,000)

## Reproducibility
- [ ] GitHub repository public and pinned at submission commit
- [ ] `REPRODUCING.md` end-to-end protocol documented
- [ ] `requirements.lock` includes exact pinned versions
- [ ] `scripts/verify_results.py` 49/49 checks pass
- [ ] DOI for repository snapshot (Zenodo or equivalent) created and cited
- [ ] Per-sample prediction files included for independent DeLong / calibration replication

## Ethics / compliance
- [ ] Data Availability statement (see `submission/01_data_availability.md`)
- [ ] Ethics statement (see `submission/02_ethics_statement.md`)
- [ ] Author contributions (CRediT taxonomy; see `submission/03_author_contributions.md`)
- [ ] Funding statement
- [ ] Competing interests declaration
- [ ] ORCID for each author registered and included on title page

## Pre-submission
- [ ] Spell-check and grammar pass (Grammarly or equivalent)
- [ ] Read through with co-author(s)
- [ ] All in-text citations resolve in reference list
- [ ] Word count within journal limits
- [ ] Figure / table / supplementary count within journal limits
- [ ] Cover letter (`submission/00_cover_letter.md`) tailored to target journal
- [ ] Suggested reviewers list (3 names, no recent collaborators / co-authors / advisors)

## Pre-print (optional but recommended)
- [ ] bioRxiv deposit before / at submission
- [ ] License selected (CC-BY recommended for OA)
- [ ] Pre-print DOI added to cover letter

## Target journals (candidate list)
- [ ] **Genome Medicine** — IF ~12, fits microbiome meta-analysis well
- [ ] **Microbiome** — IF ~14, niche fit
- [ ] **mSystems** — IF ~7, faster review, microbiome focus
- [ ] **Gut Microbes** — IF ~12, CRC + microbiome
- [ ] **Nature Communications** — IF ~17, broad audience (higher bar)
- [ ] **eLife** — IF ~7, open review, fast
