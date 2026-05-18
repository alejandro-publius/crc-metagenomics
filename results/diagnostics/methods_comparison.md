# Methods comparison: this work vs. prior multi-cohort CRC metagenomics benchmarks

This table positions this study against the three most-cited
multi-cohort shotgun-metagenomics benchmarks for colorectal cancer
(CRC) classification: Thomas et al. 2019 (Nat Med 25:667-678), Wirbel
et al. 2019 (Nat Med 25:679-689), and Piccinno et al. 2025 (Nat Med).
Axes were chosen to surface the methodological choices that drive
generalisation claims and reproducibility. Cells are sourced directly
from the published papers where possible; cells marked `[verify]` are
inferences from the paper text or supplements that should be double
checked against the original source before submission.

## Methods comparison table

| Axis | Thomas 2019 | Wirbel 2019 | Piccinno 2025 | This work |
|---|---|---|---|---|
| Cohorts | 5 CRC cohorts (8 if Italian discovery cohorts split out) | 5 CRC cohorts | 18 cohorts | 10 cohorts |
| Total samples (CRC vs control) | n=525 (313 CRC + 212 control) `[verify]` | n=575 (285 CRC + 290 control) `[verify]` | n=3741 (CRC + control + adenoma pooled) | n=1339 CRC vs control (674 CRC + 665 control); n=1522 including 183 adenoma |
| N adenoma | 143 advanced adenoma cases used in a follow-up analysis `[verify]` | Not the main endpoint; adenoma subset only in some cohorts `[verify]` | Adenoma included in pooled set, sub-endpoint reported `[verify]` | 183 adenoma across 4 cohorts; analysed both pooled and LODO |
| CV strategy | LODO across the 5 CRC cohorts plus within-cohort CV | LODO across 5 cohorts; also study-as-fold in SIAMCAT | Pooled stratified k-fold across the 18-cohort meta-dataset; LODO reported as secondary `[verify]` | Country-aware LODO across 10 cohorts (folds defined to avoid same-country train/test leakage); pooled prediction file used for all downstream diagnostics |
| Country-aware exclusion | No (Italian discovery and validation cohorts share country) `[verify]` | No (CN/AT/FR/US/DE pooled without country grouping in LODO) `[verify]` | No (pooled k-fold ignores country; LODO secondary does not enforce country exclusion) `[verify]` | Yes; LODO folds enforce no same-country cohort in train and test |
| Feature filter | Global filter on the pooled training matrix before LODO `[verify]` | Global filter inside SIAMCAT preprocessing (per-feature prevalence/abundance on pooled training data) `[verify]` | Global filter on the meta-dataset before splits `[verify]` | Per-fold filter: prevalence/abundance thresholds re-fit inside each training fold so the held-out cohort never influences which features are selected |
| Pathway features | Yes (MetaCyc pathways via HUMAnN2) | Yes (KEGG / eggNOG functional modules) | Yes (HUMAnN-derived pathways and gene families) | Yes |
| Stratified (per-species) pathway features | No | No | Yes (stratified contributions used in some analyses) `[verify]` | Pilot only; main results use unstratified pathway abundances |
| Biological pathway shortlist | No (data-driven only) | No (data-driven only) | No (data-driven only) `[verify]` | Yes; curated bile-acid / SCFA / mucin / amino-acid-degradation shortlist evaluated as a separate feature set |
| Statistical comparison | Wilcoxon / DeLong for feature-level; AUC reporting without paired tests across model variants `[verify]` | Within-SIAMCAT bootstrap CIs; DeLong not the headline `[verify]` | DeLong on pooled predictions across models `[verify]` | Both: DeLong on pooled LODO predictions (S10) and paired across-cohort comparisons; bootstrap CIs in S4 |
| Bootstrap CI iterations | 100 within-cohort bootstrap `[verify]` | 100 bootstrap (SIAMCAT default) `[verify]` | 1000 bootstrap `[verify]` | 10,000 bootstrap iterations per cohort and pooled |
| Confounder adjustment (age / sex / BMI) | Reported as covariates and tested with PERMANOVA; not folded into the predictor `[verify]` | MaAsLin-style covariate adjustment for biomarker testing; predictor uses raw features `[verify]` | Covariate adjustment in differential-abundance step; predictor variants reported `[verify]` | Yes; two-pronged: direct inclusion of age/sex/BMI in RF and XGB and residualisation of species against age/sex/BMI before LODO (S7) |
| Batch correction | None applied to the headline model; reported as a robustness check `[verify]` | None in the headline; SIAMCAT offers normalisation but not ComBat for the LODO benchmark `[verify]` | DEBIAS-M reported as a comparator method `[verify]` | ComBat reported as a robustness check (not the headline); per-fold filter + LODO is the primary defence |
| Adenoma analysis | Limited; advanced-adenoma follow-up reported on the Italian cohort `[verify]` | Per-cohort adenoma subset where available; no LODO adenoma headline `[verify]` | Adenoma included in pooled benchmark; cross-cohort generalisation reported `[verify]` | Yes; pooled and LODO (4-cohort) for healthy vs adenoma and adenoma vs CRC, plus a rebalanced LODO that downsamples controls to the adenoma prevalence |
| Adenoma LODO | No `[verify]` | No `[verify]` | Yes (pooled k-fold; LODO reported as secondary) `[verify]` | Yes (4-cohort adenoma LODO, S6) |
| Class-rebalanced adenoma analysis | No `[verify]` | No `[verify]` | No `[verify]` | Yes (rebalanced adenoma LODO with prevalence-matched controls) |
| Calibration metrics reported | No `[verify]` | No `[verify]` | Brier / calibration not headline `[verify]` | Yes; Brier score, ECE, Murphy decomposition (reliability − resolution + uncertainty), and reliability curves |
| Operating-point reporting (sensitivity at fixed specificity) | Sensitivity at single thresholds in supplementary `[verify]` | Operating characteristics in supplementary `[verify]` | Operating points reported `[verify]` | Yes; sensitivity at spec=0.90 and 0.95, FIT-matched spec=0.94 and 0.96, plus per-cohort sensitivity / specificity / PPV / NPV at a single pooled Youden-J threshold |
| Open code | Yes (repo released with paper) `[verify]` | Yes (SIAMCAT R package) | Yes (companion repo) `[verify]` | Yes |
| Pinned dependencies | Conda environment files released but not always strictly pinned `[verify]` | R package versions in DESCRIPTION; pinning not enforced `[verify]` | Conda / pip environment released `[verify]` | Yes; `requirements.lock`, `environment.yml`, `Dockerfile`, and `session_info.txt` capture exact versions |
| Reproducibility verification | No automated headline-number check `[verify]` | No automated headline-number check `[verify]` | No automated headline-number check `[verify]` | Yes; `scripts/verify_results.py` runs 49 checks against the saved CSVs and exits non-zero on any drift |

## Where this work sits

Across these 14 methodological axes, this work is the only one of the
four that combines (a) country-aware LODO folds, (b) per-fold feature
filtering (so the held-out cohort never influences feature selection),
(c) a cross-cohort adenoma analysis with an explicit class-rebalanced
variant, (d) headline calibration reporting (Brier + ECE + Murphy
decomposition) alongside the AUC comparisons, and (e) an automated
reproducibility check that re-derives 49 headline numbers from the
saved CSVs on every run. Thomas 2019 and Wirbel 2019 established the
LODO benchmark but used pooled-training feature filtering and did not
enforce country-aware splits; Piccinno 2025 expanded the sample size
roughly three-fold and added DEBIAS-M as a batch-correction comparator
but still relies on pooled k-fold as its headline. The trade-off is
honest: this work is smaller in raw n than Piccinno 2025, so its
contribution is methodological tightness and translation-facing
diagnostics (FIT-matched operating points, base-rate PPV sweep,
per-cohort operating characteristics) rather than scale.
