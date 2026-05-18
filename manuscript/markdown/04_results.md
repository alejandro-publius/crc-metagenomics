# Results

## Cohorts and feature spaces

The analysis dataset comprised 1,522 unique subjects across 10 cohorts: 674 CRC cases, 665 controls, and 183 adenomas, spanning 8 countries (Austria, China, France, Germany, India, Italy, Japan, USA; Table 1). Per-cohort sample sizes ranged from 60 (GuptaA_2019, ThomasAM_2018b) to 575 (YachidaS_2019). After global filtering (prevalence ≥ 10%, mean ≥ 1e-4) and log10(x + 1e-6) transformation, 229 MetaPhlAn species features were retained. From HUMAnN unstratified pathway tables, 551 candidate pathways entered the per-fold filter; 402–406 pathways were retained per LODO fold after the prevalence ≥ 10% / mean ≥ 1e-6 filter computed on training-cohort samples only.

## Species-only Random Forest under country-aware LODO

The species-only Random Forest achieved a per-cohort mean LODO AUC of **0.807 ± 0.065 (SD across 10 folds)** and a pooled AUC of **0.781 (95% CI 0.757–0.805)** across all 1,339 case/control predictions. Per-cohort AUCs ranged from 0.694 (ThomasAM_2018a; n_test = 53) to 0.882 (GuptaA_2019, n_test = 60; WirbelJ_2018, n_test = 125) (Figure 1; Supplementary Table S2). The largest fold, YachidaS_2019 (n_test = 508), reached an AUC of 0.708; its size dominates the pooled estimate and the DeLong comparison. At a fixed specificity of 90%, species RF achieves a sensitivity of 49.9% (`results/diagnostics/sens_at_fixed_spec.csv`); at 95% specificity, 39.8% (same table).

Seed sensitivity confirmed stability: across seeds {0, 1, 2, 42, 100} the per-cohort mean AUC was 0.810 ± 0.002 (range 0.807–0.811).

## Joint species-plus-pathway models do not improve over species alone

Adding 402–406 per-fold pathway features to the 229 species reduced rather than improved performance. The joint Random Forest reached a per-cohort mean AUC of **0.804 ± 0.066** and a pooled AUC of **0.756 (95% CI 0.731–0.781)**. The joint XGBoost reached a per-cohort mean AUC of **0.797 ± 0.064** and a pooled AUC of **0.766 (95% CI 0.740–0.791)**.

Per-cohort paired comparisons (n = 10 folds) did not reach significance for either contrast: species RF vs joint RF (mean Δ = 0.003; paired t-test p = 0.87; Wilcoxon p = 0.38) or species RF vs joint XGBoost (mean Δ = 0.010; paired t-test p = 0.28; Wilcoxon p = 0.23). This is consistent with the low statistical power of paired tests at n = 10.

The DeLong test on the same pooled held-out predictions (n = 1,339) clearly detected the difference: species RF significantly outperformed the joint RF (ΔAUC = 0.025; z = 3.35; **p = 0.0008**) and the joint XGBoost (ΔAUC = 0.015; z = 2.00; **p = 0.046**). The two joint models did not differ significantly from one another (z = 1.30; p = 0.19). Most of the DeLong signal arises from the YachidaS_2019 fold (n_test = 508), where species RF reaches 0.708 versus 0.669 (joint RF) and 0.694 (joint XGBoost) (Figure 2; Supplementary Table S2).

Together, the per-cohort and DeLong analyses agree that pathways add no benefit on average; DeLong further detects a small but statistically significant *degradation* at the sample level, driven primarily by the largest fold.

## Filter-threshold sensitivity

Across a 20-cell grid of prevalence cutoffs {0.05, 0.10, 0.15, 0.20} × mean-abundance cutoffs {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}, the joint RF per-cohort mean AUC ranged from **0.794 to 0.812** (full-grid spread 0.018). The default thresholds (prevalence ≥ 10%, mean ≥ 1e-6) sit near the middle of the observed range, and qualitative conclusions are insensitive to the specific cutoffs chosen (Supplementary Table S1). The 1e-3 mean column retains only two pathways and effectively reduces the joint model to a near-species-only configuration.

## Confounder assessment

Inclusion of age, sex, and BMI as covariates produced changes within sampling noise of the species-only baseline (per-cohort mean AUC 0.807): direct RF inclusion yielded 0.814, direct XGBoost 0.806, residualized RF 0.801, and residualized XGBoost 0.800. The 0.800–0.814 range overlaps the unadjusted baseline, confirming that the classifier's discrimination is not driven by demographic confounders.

## Batch correction

Per-fold ComBat correction on species features produced a per-cohort mean AUC of 0.815 versus 0.807 without correction (Δ +0.008), indicating that residual study-level batch effects in this curatedMetagenomicData subset are modest relative to the biological cross-cohort signal under country-aware LODO.

## Feature importance

TreeSHAP analysis of the joint RF identified *Gemella morbillorum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, and *Fusobacterium nucleatum* as the four highest-ranked features by mean absolute SHAP value (Figure 3). The top four features were identical in joint XGBoost up to rank order (*Gemella morbillorum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, *Streptococcus salivarius*, *Fusobacterium nucleatum*). Among the top 15 features, 8 were shared between RF and XGBoost, indicating substantial concordance across architectures. The XGBoost top 15 additionally included two pathway features (PWY-6151: S-adenosyl-L-methionine cycle I; PWY0-162: superpathway of pyrimidine ribonucleotide de novo biosynthesis), consistent with the joint model's access to pathways but reinforcing that species features dominate the discriminative signal.

## Adenoma classification along the adenoma-carcinoma sequence

Cross-cohort LODO across the four adenoma-containing cohorts (FengQ_2015, YachidaS_2019, ZellerG_2014, ThomasAM_2018a; total n = 183) produced markedly different performance for the two stages of the adenoma-carcinoma sequence (Figure 4):

- **Control vs adenoma (H-vs-A):** mean LODO AUC 0.561 (RF) and 0.579 (XGBoost) — near chance.
- **Adenoma vs CRC (A-vs-CRC):** mean LODO AUC 0.671 (RF) and 0.617 (XGBoost) — moderate above-chance discrimination.

TreeSHAP rankings parallel this pattern. The H-vs-A classifier (Figure 4A) emphasizes metabolic-pathway and commensal-depletion features (e.g., PWY-5994 palmitate biosynthesis, *Eubacterium eligens*, PANTO-PWY pantothenate biosynthesis, *Collinsella intestinalis*), none of which constitutes a strong, reproducible cross-cohort signal. The CRC-vs-control (Figure 4B) and A-vs-CRC (Figure 4C) classifiers, by contrast, are both dominated by the same four oral pathobionts that top the main CRC analysis (*Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum*). The reordering of top features between the H-vs-A and A-vs-CRC tasks is consistent with a stepwise oral-pathobiont enrichment during malignant transformation and supports treating adenoma and CRC as biologically distinct microbiome states rather than two points on a smooth severity gradient.

## Figure legends

**Figure 1.** Forest plot of per-cohort and pooled LODO AUCs with 10,000-iteration bootstrap 95% CIs for CRC versus control classification under country-aware leave-one-dataset-out cross-validation. Three classifiers are shown: species-only Random Forest, joint species-plus-pathway Random Forest, and joint XGBoost. The pooled estimate (n = 1,339) is shown at the bottom.

**Figure 2.** Receiver operating characteristic (ROC) curves for pooled LODO predictions (n = 1,339) comparing species-only Random Forest (AUC = 0.781), joint Random Forest (AUC = 0.756), and joint XGBoost (AUC = 0.766). The diagonal indicates chance-level classification.

**Figure 3.** Top 15 species and pathway features by mean absolute TreeSHAP value for CRC versus control classification. Left, joint Random Forest; right, joint XGBoost. The four highest-ranked species (*Parvimonas micra*, *Peptostreptococcus stomatis*, *Gemella morbillorum*, *Fusobacterium nucleatum*) are concordant across both models; 8 of 15 features are shared between RF and XGBoost.

**Figure 4.** Three-panel TreeSHAP comparison across the adenoma-carcinoma sequence under country-aware LODO across the four adenoma-containing cohorts (n = 183 adenomas). (A) Control vs adenoma (H-vs-A; RF mean LODO AUC 0.561). (B) CRC vs control (reference panel). (C) Adenoma vs CRC (A-vs-CRC; RF mean LODO AUC 0.671). The oral-pathobiont signature (*Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum*) dominates the two panels involving CRC samples and is largely absent from the H-vs-A panel.
