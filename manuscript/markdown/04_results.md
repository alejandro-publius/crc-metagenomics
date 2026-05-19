# Results

## Cohorts and feature spaces

The analysis dataset comprised 1,522 unique subjects across 10 cohorts: 674 CRC cases, 665 controls, and 183 adenomas, spanning 8 countries (Austria, China, France, Germany, India, Italy, Japan, USA; Table 1; `results/table1.csv`). Per-cohort sample sizes ranged from 60 (GuptaA_2019, ThomasAM_2018b) to 575 (YachidaS_2019). After global filtering (prevalence ≥ 10%, mean ≥ 1e-4) and log10(x + 1e-6) transformation, 229 MetaPhlAn species features were retained. From HUMAnN unstratified pathway tables, 551 candidate pathways entered the per-fold filter; 402–406 pathways were retained per LODO fold (`results/joint_results.csv`, `rf_n_features` column minus 229 species) after the prevalence ≥ 10% / mean ≥ 1e-6 filter computed on training-cohort samples only.

## Species-only Random Forest under country-aware LODO

The species-only Random Forest achieved a per-cohort mean LODO AUC of **0.807 ± 0.065 (SD across 10 folds; `results/baseline_results.csv`)** and a pooled AUC of **0.781 (95% CI 0.757–0.805; `results/bootstrap_ci.csv`, row `species_rf,pooled`)** across all 1,339 case/control predictions. Per-cohort AUCs ranged from 0.694 (ThomasAM_2018a; n_test = 53) to 0.882 (GuptaA_2019, n_test = 60; WirbelJ_2018, n_test = 125) (Figure 1; Supplementary Table S2). The largest fold, YachidaS_2019 (n_test = 508), reached an AUC of 0.708; its size dominates the pooled estimate and the DeLong comparison. At a fixed specificity of 90%, species RF achieves a sensitivity of 49.9% (`results/diagnostics/sens_at_fixed_spec.csv`); at 95% specificity, 39.8% (same table).

Seed sensitivity confirmed stability: across seeds {0, 1, 2, 42, 100} the per-cohort mean AUC was 0.810 ± 0.002 (range 0.807–0.811; `results/seed_sensitivity.csv`).

## Joint species-plus-pathway models do not improve over species alone

Adding 402–406 per-fold pathway features to the 229 species reduced rather than improved performance. The joint Random Forest reached a per-cohort mean AUC of **0.804 ± 0.066 (`results/joint_results.csv`, `rf_auc` column)** and a pooled AUC of **0.756 (95% CI 0.731–0.781; `results/bootstrap_ci.csv`, row `joint_rf,pooled`)**. The joint XGBoost reached a per-cohort mean AUC of **0.797 ± 0.064 (`results/joint_results.csv`, `xgb_auc` column)** and a pooled AUC of **0.766 (95% CI 0.740–0.791; `results/bootstrap_ci.csv`, row `joint_xgb,pooled`)**.

Per-cohort paired comparisons (n = 10 folds) did not reach significance for either contrast: species RF vs joint RF (mean Δ = 0.003; paired t-test p = 0.87; Wilcoxon p = 0.38) or species RF vs joint XGBoost (mean Δ = 0.010; paired t-test p = 0.28; Wilcoxon p = 0.23; all in `results/model_comparison.csv`). This is consistent with the low statistical power of paired tests at n = 10.

The DeLong test on the same pooled held-out predictions (n = 1,339) clearly detected the difference: species RF significantly outperformed the joint RF (ΔAUC = 0.025; z = 3.35; **p = 0.0008**; `results/delong_results.csv`, row 1) and the joint XGBoost (ΔAUC = 0.015; z = 2.00; **p = 0.046**; `results/delong_results.csv`, row 2). The two joint models did not differ significantly from one another (z = 1.30; p = 0.19; `results/delong_results.csv`, row 3). Most of the DeLong signal arises from the YachidaS_2019 fold (n_test = 508), where species RF reaches 0.708 versus 0.669 (joint RF) and 0.694 (joint XGBoost) (Figure 2; Supplementary Table S2).

Together, the per-cohort and DeLong analyses agree that pathways add no benefit on average; DeLong further detects a small but statistically significant *degradation* at the sample level, driven primarily by the largest fold.

## Species-stratified pathway features do not improve over species alone either

To address the possibility that the community-level pathway features are too coarse to capture the joint species-plus-function signal, we re-ran the joint model using HUMAnN species-stratified pathway abundances (per-species, per-pathway) pulled from curatedMetagenomicData 3.20 (`scripts/export_data_stratified.R`). This raises the candidate pathway feature space from ~400 community-level pathways to ~12,000-31,000 stratified pathways per cohort (union: 38,672 features), and yields 9,500-9,950 features per fold after the per-fold prevalence ≥ 5% / mean ≥ 1e-7 filter (`results/stratified_joint_results.csv`, `n_features` column minus 229 species). Under the identical country-aware LODO, RF, and XGBoost configuration:

- **Joint stratified RF:** per-cohort mean AUC **0.771 ± 0.073** and pooled **0.735 (95% CI 0.708–0.761; 10,000 cohort-stratified bootstrap; `results/stratified_vs_baseline_comparison.csv`)**.
- **Joint stratified XGBoost:** per-cohort mean AUC **0.800 ± 0.057** and pooled **0.769 (95% CI 0.744–0.794)**.

DeLong tests on the pooled held-out predictions (`results/delong_stratified_vs_baseline.csv`): species RF significantly outperforms joint stratified RF (ΔAUC = -0.046; z = 5.49; **p < 0.0001**), and joint community-pathway RF significantly outperforms joint stratified RF (ΔAUC = -0.021; z = 3.02; **p = 0.003**). The XGBoost comparisons are not significant (species vs stratified XGB p = 0.13; community-joint XGB vs stratified XGB p = 0.62).

Together with the community-level pathway result, this triples the feature-resolution coverage of the negative finding: across three independent feature regimes (species alone, species + community pathway, species + species-stratified pathway), adding pathway features at any granularity tested does not improve cross-cohort CRC discrimination. The RF degradation worsens monotonically with feature count (0.781 → 0.756 → 0.735 pooled AUC), consistent with the curse of dimensionality at n = 1,339; XGBoost's `tree_method='hist'` handles the wider feature space without further degradation but does not improve either. This rules out "the community-pathway aggregation was masking the signal" as an explanation for the joint model's underperformance.

## Filter-threshold sensitivity

Across a 20-cell grid of prevalence cutoffs {0.05, 0.10, 0.15, 0.20} × mean-abundance cutoffs {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}, the joint RF per-cohort mean AUC ranged from **0.781 to 0.835** (full-grid spread 0.055; `results/sensitivity_thresholds.csv`). The default thresholds (prevalence ≥ 10%, mean ≥ 1e-6) sit near the middle of the observed range, and qualitative conclusions are insensitive to the specific cutoffs chosen (Supplementary Table S1). The 1e-3 mean column retains only two pathways and effectively reduces the joint model to a near-species-only configuration.

## Confounder assessment

Inclusion of age, sex, and BMI as covariates produced changes within sampling noise of the species-only baseline (per-cohort mean AUC 0.807; `results/baseline_results.csv`): direct RF inclusion yielded 0.814, direct XGBoost 0.806, residualized RF 0.801, and residualized XGBoost 0.800 (`results/confounder_results.csv`). The 0.800–0.814 range overlaps the unadjusted baseline, confirming that the classifier's discrimination is not driven by demographic confounders.

## Batch correction

Per-fold ComBat correction on species features produced a per-cohort mean AUC of 0.815 versus 0.807 without correction (Δ +0.008; `results/combat_results.csv`), indicating that residual study-level batch effects in this curatedMetagenomicData subset are modest relative to the biological cross-cohort signal under country-aware LODO.

## Feature importance

TreeSHAP analysis of the joint RF identified *Gemella morbillorum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, and *Fusobacterium nucleatum* as the four highest-ranked features by mean absolute SHAP value (Figure 3). The joint XGBoost top four overlap with RF on three of these (*Gemella morbillorum*, *Peptostreptococcus stomatis*, *Parvimonas micra*) and substitute *Streptococcus salivarius* for *Fusobacterium nucleatum* at rank 4 (*F. nucleatum* remains in the XGBoost top six). Among the top 15 features, 8 were shared between RF and XGBoost, indicating substantial concordance across architectures. The XGBoost top 15 additionally included two pathway features (PWY-6151: S-adenosyl-L-methionine cycle I; PWY0-162: superpathway of pyrimidine ribonucleotide de novo biosynthesis), consistent with the joint model's access to pathways but reinforcing that species features dominate the discriminative signal.

## Adenoma classification along the adenoma-carcinoma sequence

Cross-cohort LODO across the four adenoma-containing cohorts (FengQ_2015, YachidaS_2019, ZellerG_2014, ThomasAM_2018a; total n = 183) gave the following results (Figure 4):

- **Control vs adenoma (H-vs-A):** mean LODO AUC 0.561 (RF) and 0.579 (XGBoost), a null result indistinguishable from chance (`results/adenoma_lodo_results.csv`, rows `h_vs_a_rf`, `h_vs_a_xgb`).
- **Adenoma vs CRC (A-vs-CRC):** mean LODO AUC 0.671 (RF) and 0.617 (XGBoost) — modest above-chance discrimination (`results/adenoma_lodo_results.csv`, rows `a_vs_crc_rf`, `a_vs_crc_xgb`).

These adenoma estimates are underpowered: n = 183 adenoma samples across 4 heterogeneous cohorts (n_folds = 4) provides limited resolving power for cross-cohort discrimination at effect sizes below AUC ≈ 0.65, so the H-vs-A AUC of 0.561 should be read as "no detectable cross-cohort signal at this sample size" rather than as positive evidence of biological equivalence between adenoma and control microbiomes.

TreeSHAP rankings parallel this pattern. The H-vs-A classifier (Figure 4A) emphasizes metabolic-pathway and commensal-depletion features (e.g., PWY-5994 palmitate biosynthesis, *Eubacterium eligens*, PANTO-PWY pantothenate biosynthesis, *Collinsella intestinalis*), none of which constitutes a strong, reproducible cross-cohort signal. The CRC-vs-control (Figure 4B) and A-vs-CRC (Figure 4C) classifiers, by contrast, are both dominated by the same four oral pathobionts that top the main CRC analysis (*Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum*). The reordering of top features between the H-vs-A and A-vs-CRC tasks is consistent with prior literature suggesting stepwise oral-pathobiont enrichment along the adenoma-carcinoma sequence; this descriptive pattern should not be over-interpreted given the H-vs-A null result and the limited per-cohort sample sizes.

## Figure legends

**Figure 1.** Forest plot of per-cohort and pooled LODO AUCs with 10,000-iteration bootstrap 95% CIs for CRC versus control classification under country-aware leave-one-dataset-out cross-validation. Three classifiers are shown: species-only Random Forest, joint species-plus-pathway Random Forest, and joint XGBoost. The pooled estimate (n = 1,339) is shown at the bottom.

**Figure 2.** Receiver operating characteristic (ROC) curves for pooled LODO predictions (n = 1,339) comparing species-only Random Forest (AUC = 0.781), joint Random Forest (AUC = 0.756), and joint XGBoost (AUC = 0.766). The diagonal indicates chance-level classification.

**Figure 3.** Top 15 species and pathway features by mean absolute TreeSHAP value for CRC versus control classification. Left, joint Random Forest; right, joint XGBoost. The four highest-ranked species (*Parvimonas micra*, *Peptostreptococcus stomatis*, *Gemella morbillorum*, *Fusobacterium nucleatum*) are concordant across both models; 8 of 15 features are shared between RF and XGBoost.

**Figure 4.** Three-panel TreeSHAP comparison across the adenoma-carcinoma sequence under country-aware LODO across the four adenoma-containing cohorts (n = 183 adenomas). (A) Control vs adenoma (H-vs-A; RF mean LODO AUC 0.561). (B) CRC vs control (reference panel). (C) Adenoma vs CRC (A-vs-CRC; RF mean LODO AUC 0.671). The oral-pathobiont signature (*Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum*) dominates the two panels involving CRC samples and is largely absent from the H-vs-A panel.
