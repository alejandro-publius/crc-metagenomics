# ISMB / RECOMB abstract (250 words)

**Title:** Species-level taxonomic features alone outperform joint species-plus-pathway models for cross-cohort colorectal cancer classification

**Authors:** Alejandro Velazquez¹, Rachel Selbrede²

**Affiliations:**
¹[FILL: Alex's affiliation]
²[FILL: Rachel's affiliation]

**Track / Topic:** [FILL: e.g., ISMB Microbiome COSI, or RECOMB methods track]

**Word count:** 248 (target 250)

---

**Background.** Shotgun gut metagenomic classifiers discriminate colorectal cancer (CRC) cases from controls, but the incremental value of HUMAnN pathway features beyond MetaPhlAn species profiles has not been rigorously tested, and the robustness of cross-cohort classifiers to analytical choices is rarely evaluated systematically.

**Methods.** We assembled 1,522 stool metagenomes from ten curatedMetagenomicData cohorts (674 CRC, 665 controls, 183 adenomas; HanniganGD_2017 excluded a priori for low sequencing depth). MetaPhlAn species (229 features after a 10% prevalence / 1e-4 mean global filter, log10-transformed) and HUMAnN unstratified pathways (551 candidates, **re-filtered per fold** at prevalence >= 10% and mean >= 1e-6 to prevent leakage, retaining 402-406 features) were compared under **country-aware leave-one-dataset-out (LODO)** cross-validation: when a cohort is the test fold, all cohorts sharing its country are also excluded from training. Three classifiers — species-only Random Forest, joint species+pathway RF, joint XGBoost — were compared via the DeLong-Sun-Xu test on pooled held-out predictions and per-cohort paired t / Wilcoxon tests. 95% confidence intervals derive from 10,000-resample cohort-stratified bootstrap.

**Results.** Species-only RF achieved per-cohort mean LODO AUC 0.807 ± 0.065 and pooled AUC **0.781 (95% CI 0.757-0.805)**, significantly outperforming joint RF (0.756; **DeLong z = 3.35, p = 0.0008**) and joint XGBoost (0.766; z = 2.00, p = 0.046). Country-aware LODO removed population-level confounding: ThomasAM_2019_c AUC dropped from 0.998 to 0.836 when same-country YachidaS_2019 was excluded. Results were stable across five random seeds (0.810 ± 0.002), a 20-cell pathway-threshold grid (0.794-0.812), demographic adjustment (0.800-0.814), and per-fold ComBat correction.

**Conclusions.** At current cross-cohort sample sizes, species-level features alone provide superior CRC classification; pathway features increase dimensionality without proportional signal. Formal classifier comparison and systematic robustness evaluation should be standard in metagenomic ML studies.
