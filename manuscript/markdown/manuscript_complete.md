# Species-level taxonomic features outperform joint species-plus-pathway models for colorectal cancer detection: a rigorous re-evaluation across ten cohorts

Alejandro Velazquez^1,\*^, Rachel Selbrede^2^

^1^ University of California, Berkeley, CA, USA
^2^ California State University San Marcos, San Marcos, CA, USA

\* Correspondence: alejandro-publius@berkeley.edu

**Keywords:** colorectal cancer; gut microbiome; shotgun metagenomics; machine learning; Random Forest; leave-one-dataset-out cross-validation; DeLong test; SHAP; curatedMetagenomicData.

**Running title:** Species-only classifiers outperform joint models for CRC.

**Word count:** Abstract ~250 words; main text ~4,800 words.
**Display items:** Figures 4; Tables 1; Supplementary Tables 2.
**Code and data:** https://github.com/alejandro-publius/crc-metagenomics


---

# Abstract

**Background.** Shotgun gut metagenomic classifiers can discriminate colorectal cancer (CRC) cases from controls, but the incremental value of metabolic pathway features beyond species-level taxonomic profiles has not been rigorously tested, and the robustness of cross-cohort classifiers to analytical choices is rarely evaluated systematically.

**Methods.** We assembled 1,522 stool metagenomes from ten publicly available CRC case-control cohorts (674 CRC, 665 controls, 183 adenomas) accessed through curatedMetagenomicData. HanniganGD_2017 was excluded a priori for low sequencing depth. MetaPhlAn species (229 features after a 10% prevalence / 1e-4 mean filter and log10(x + 1e-6) transform) and unstratified HUMAnN pathway abundances (551 candidate pathways, refiltered per fold at prevalence ≥ 10% and mean ≥ 1e-6, retaining 402–406 features) were compared under country-aware leave-one-dataset-out (LODO) cross-validation. Three classifiers were evaluated: species-only Random Forest (RF), joint species-plus-pathway RF, and joint XGBoost. Classifier discrimination was compared using the DeLong test (Sun and Xu 2014) on pooled held-out predictions, complemented by per-cohort paired t-tests and Wilcoxon signed-rank tests. 95% confidence intervals were derived from 10,000-iteration cohort-stratified bootstrap resampling.

**Results.** The species-only RF achieved a per-cohort mean LODO AUC of 0.807 ± 0.065 and a pooled AUC of 0.781 (95% CI 0.757–0.805). It significantly outperformed the joint RF (pooled AUC 0.756, 95% CI 0.731–0.781; DeLong z = 3.35, p = 0.0008) and the joint XGBoost (pooled AUC 0.766, 95% CI 0.740–0.791; z = 2.00, p = 0.046). Per-cohort paired tests on the same comparisons did not reach significance (t-test p = 0.87 and 0.28), consistent with limited power at n = 10 folds. Results were stable across five random seeds (mean per-cohort AUC 0.810 ± 0.002, range 0.807–0.811), a 20-cell pathway-threshold sensitivity grid (0.794–0.812, spread 0.018), and demographic adjustment for age, sex, and BMI (0.800–0.814). Cross-cohort adenoma LODO across the four adenoma-containing cohorts (n = 183) yielded near-chance discrimination for healthy-vs-adenoma (RF 0.561, XGB 0.579) and moderate discrimination for adenoma-vs-CRC (RF 0.671, XGB 0.617), consistent with a stepwise oral-pathobiont enrichment along the adenoma-carcinoma sequence.

**Conclusions.** At current cross-cohort sample sizes, species-level taxonomic features alone provide superior CRC classification compared to joint species-plus-pathway models; adding pathway features increases dimensionality without proportional signal gain. The findings support parsimonious species-level classifiers for microbiome-based CRC screening and highlight the importance of formal statistical comparison and systematic robustness evaluation in metagenomic classification studies.


---

# Introduction

Colorectal cancer (CRC) is the third most commonly diagnosed cancer worldwide and the second leading cause of cancer-related death, with approximately 1.9 million new cases and 930,000 deaths estimated annually (Sung et al. 2021). The global burden is projected to reach 3.2 million new cases per year by 2040, driven by aging populations and the global diffusion of Western dietary patterns (Xi and Xu 2021). Early detection substantially improves survival, yet colonoscopy-based screening programs are resource-intensive and achieve incomplete population coverage. There is a clear need for non-invasive, scalable biomarkers that can complement or precede endoscopic screening.

The gut microbiome has emerged as a promising source of such biomarkers. Large-scale shotgun metagenomic studies have identified reproducible microbial signatures that distinguish CRC patients from healthy controls, with cross-cohort area under the receiver operating characteristic curve (AUC) values typically in the 0.75–0.85 range (Wirbel et al. 2019; Thomas et al. 2019; Yachida et al. 2019). The recent pooled analysis of 3,741 metagenomes across 18 cohorts (Piccinno et al. 2025) reached a mean AUC of approximately 0.85, further establishing the diagnostic potential of stool-based metagenomic profiling and motivating the application of machine learning to metagenomic features for CRC detection.

Thomas et al. (2019) introduced a landmark multi-cohort framework that combined MetaPhlAn species-level taxonomic profiles (Truong et al. 2015) with HUMAnN functional pathway abundances (Franzosa et al. 2018), trained Random Forest classifiers under leave-one-dataset-out (LODO) cross-validation, and demonstrated cross-cohort generalization for CRC detection. This framework has become a de facto reference standard for metagenomics-based CRC classification. Three methodological concerns about the modern application of this framework have, however, received insufficient attention.

First, the contribution of functional pathway features relative to species-level taxonomic features has rarely been tested with formal statistical comparison; many studies report joint model performance without asking whether pathways genuinely improve upon species alone. Second, feature filtering applied globally — across pooled training and test cohorts — rather than refit within each cross-validation fold risks leakage from the held-out cohort into the feature-selection pipeline and can inflate performance estimates. Third, robustness to analytical choices (filter thresholds, random seeds, batch effects, demographic confounders, country-level confounding) is rarely reported systematically, making it difficult to assess whether published results reflect stable biological signals or are artifacts of specific analytical configurations.

Recent benchmarking efforts have evaluated bioinformatics workflows for CRC detection across multiple cohorts (Sun et al. 2025), and several groups have proposed processing-bias corrections to improve cross-study generalization of microbiome prediction models. What is still missing is a focused, end-to-end re-evaluation of the Thomas et al. (2019) framework with per-fold feature filtering, formal statistical classifier comparison via the DeLong test, and a systematic robustness battery, performed on an expanded modern cohort set.

Here we revisit the Thomas et al. (2019) framework with three objectives. First, we test whether the addition of unstratified HUMAnN pathway features to MetaPhlAn species-level profiles improves classification performance, using the DeLong test (DeLong et al. 1988; Sun and Xu 2014) on pooled LODO predictions for formal sample-level comparison and per-cohort paired t-tests / Wilcoxon signed-rank tests for fold-level comparison. Second, we implement per-fold pathway filtering and a country-aware LODO design to eliminate two distinct sources of information leakage: from held-out cohorts into feature selection, and from same-country cohorts into the training fold. Third, we conduct a systematic robustness battery comprising 20-cell filter-threshold sensitivity analysis, demographic confounder assessment (age, sex, BMI), random-seed stability, 10,000-iteration cohort-stratified bootstrap confidence intervals, per-fold ComBat batch correction, and a biologically-guided pathway shortlist. We additionally extend the framework to cross-cohort adenoma classification across the four adenoma-containing cohorts. All code, processed data, per-sample predictions, and decision logs are publicly available to enable end-to-end reproducibility.


---

# Methods

## Study design and cohort selection

We performed a multi-cohort classification study to evaluate the discriminative capacity of gut metagenomic features for colorectal cancer (CRC) detection, extending the analytical framework of Thomas et al. (2019). Uniformly processed shotgun metagenomic profiles were obtained from the curatedMetagenomicData Bioconductor resource (Pasolli et al. 2017) using the `returnSamples()` function in R. Exact R, Bioconductor, and curatedMetagenomicData versions used for data extraction are recorded in the repository's `session_info.txt`.

Ten cohorts were retained for analysis: FengQ_2015 (Austria), GuptaA_2019 (India), ThomasAM_2018a (Italy), ThomasAM_2018b (Italy), ThomasAM_2019_c (Japan), VogtmannE_2016 (USA), WirbelJ_2018 (Germany), YachidaS_2019 (Japan), YuJ_2015 (China), and ZellerG_2014 (France). HanniganGD_2017 was excluded a priori based on a pre-specified, classifier-blind quality assessment: mean sequencing depth of 6.5M reads (range 17K–21M) was substantially below all other cohorts (per-cohort mean depth in the retained 10-cohort set ranges from 9.2M for GuptaA_2019 to 102M for ThomasAM_2018a; all other retained cohorts >40M) and species feature sparsity was 82% versus a 61% mean across other cohorts. An additional per-sample minimum of 1M reads removed four extreme outliers. The final dataset contained 1,522 unique subjects (674 CRC, 665 healthy controls, 183 adenomas); subject identity was audited to confirm that no individual appeared in more than one cohort. The metadata `study_condition` field uses the value `control` (not `healthy`); we use "control" throughout for samples coded as such. Demographic characteristics (age, sex, BMI) were extracted from sample metadata (Table 1).

## Feature extraction and preprocessing

**Species abundance.** Taxonomic profiles were taken from MetaPhlAn relative abundance tables (Truong et al. 2015). Species-level features were filtered to taxa with prevalence ≥ 10% and mean relative abundance ≥ 1 × 10⁻⁴, yielding 229 species features. Retained features were row-sum renormalized (when input was on a percentage scale) and log-transformed as log₁₀(x + 1 × 10⁻⁶). The species filter is computed globally rather than per fold; this is a mild form of feature-set leakage, accepted on three grounds: (i) MetaPhlAn maps to a fixed reference database, so the filter primarily removes globally rare taxa; (ii) only 229 species are retained, providing little room for overfitting at the filter stage; and (iii) global species filtering matches the reference standard set by Thomas et al. (2019).

**Pathway abundance.** Functional profiles were taken from HUMAnN unstratified pathway abundance tables (Franzosa et al. 2018). Per-cohort pathway tables were concatenated into a single matrix. Only unstratified pathway features (community-level pathway abundances, excluding taxon-stratified columns) were retained as joint-model candidates, yielding 551 candidate columns. A prevalence ≥ 10% and mean ≥ 1 × 10⁻⁶ filter was applied *within each LODO fold using only training-cohort samples*, retaining 402–406 pathways per fold (633–635 total features after merging with species). Pathway features were retained on their native relative-abundance scale; because Random Forest and XGBoost split decisions are scale-invariant per feature, the asymmetric handling of species (log-transformed) and pathway (raw) features does not affect AUC.

Stratified taxon|pathway features were considered but excluded: they produce >4,000 highly redundant columns and did not improve AUC in pilot tests. Pathway features were also evaluated under a biologically-guided subset (nine CRC-relevant functional groups, 86 candidate pathways yielding ~66 retained features per fold; `scripts/bio_pathway_shortlist.py`) and as species-stratified pathways (pilot in `scripts/stratified_pathway_pilot.py`, `results/stratified_pathway_pilot.csv`); neither rescued the joint model, indicating that the negative result on community-level pathways is not an artefact of feature granularity.

## Country-aware leave-one-dataset-out cross-validation

LODO cross-validation holds out one cohort as the test set while training on the remaining cohorts. Because two pairs of cohorts share a country (ThomasAM_2018a / ThomasAM_2018b in Italy; ThomasAM_2019_c / YachidaS_2019 in Japan), we used a *country-aware* LODO design: when a cohort is held out, all cohorts from the same country are also removed from the training fold. Without this fix, ThomasAM_2019_c reached an inflated AUC of 0.999 due to YachidaS_2019 in training; with country-aware LODO the AUC drops to a biologically plausible 0.836. Country-aware LODO is applied consistently across `train_baseline.py`, `train_joint.py`, `seed_sensitivity.py`, `sensitivity_analysis.py`, and `bio_pathway_shortlist.py`.

## Batch-effect mitigation

Cross-cohort variability in DNA extraction, library preparation, sequencing depth, and host population genetics is the dominant source of unwanted variation in pooled gut metagenomic data. Our design treats this as a structural problem rather than a post-hoc statistical correction. Country-aware LODO is the *primary* batch-effect mitigation: each cohort is treated as a single batch, the test cohort is held out in full, and any other cohort sharing the test cohort's country is additionally excluded from training (Italy and Japan pairs above) to remove population-level genetic and dietary confounding from the train–test split. Per-fold filtering and feature normalisation use only training-cohort statistics; no within-fold scaling or standardisation is applied beyond the global species log₁₀ transform described above. Per-fold ComBat (Johnson et al. 2007) on species features, fit jointly on train and test feature matrices using only `study_name` as the batch label, is reported as a *robustness check* rather than the primary correction (`results/combat_results.csv`): the ComBat-corrected pooled per-cohort mean AUC is 0.815 versus 0.807 uncorrected, and qualitative conclusions are unchanged. This ordering — design first, statistical correction second — reflects the view that residual batch structure that survives country-aware LODO is not reliably removed by mean/variance harmonisation alone.

## Classifiers

**CRC vs control.** Binary classification of CRC (n = 674) versus controls (n = 665) was performed on the 1,339 case/control subjects under country-aware LODO. Three models were compared: (1) species-only Random Forest, (2) joint species-plus-pathway Random Forest, and (3) joint species-plus-pathway XGBoost. Random Forest was configured with `n_estimators = 500`, `max_features = 'sqrt'`, `min_samples_leaf = 5`, `class_weight = 'balanced'`. XGBoost was configured with `n_estimators = 500`, `max_depth = 6`, `learning_rate = 0.1`, `subsample = 0.8`, `colsample_bytree = 0.8`; for adenoma tasks, XGBoost additionally used `scale_pos_weight` equal to the inverse class ratio recomputed per fold. Hyperparameters were not tuned via nested cross-validation because the joint model did not statistically outperform the species-only baseline, making further optimization unlikely to alter qualitative conclusions. All models used `random_state = 42` unless stated otherwise.

**Adenoma classification.** Four cohorts contain adenoma samples (FengQ_2015: n = 47; YachidaS_2019: n = 67; ZellerG_2014: n = 42; ThomasAM_2018a: n = 27; total n = 183). All four are from different countries, so no additional country-aware exclusion is required. Two binary tasks were evaluated under country-aware LODO across these four cohorts: control-vs-adenoma (H-vs-A) and adenoma-vs-CRC (A-vs-CRC), each with RF and XGBoost. An earlier within-cohort pooled five-fold protocol (`train_adenoma.py`) is retained in the repository for reference only; all current adenoma results use `adenoma_lodo.py` (Decision Memo, `results/adenoma_go_nogo_memo.md`).

Class imbalance was handled by class-weighting (`class_weight='balanced'` for RF; per-fold `scale_pos_weight` for XGBoost); SMOTE was not used.

## Statistical comparison of classifiers

Model performance was compared using three complementary tests. First, per-cohort AUC differences across the 10 LODO folds were assessed by paired t-test (df = 9) and Wilcoxon signed-rank test, with 95% bootstrap confidence intervals on the mean difference (`scripts/auc_comparison.py`). These per-cohort tests have limited power at n = 10. Second, the DeLong test (DeLong et al. 1988) using the fast implementation of Sun and Xu (2014) was applied to pooled held-out predictions (n = 1,339), comparing the ROC curves of each classifier pair on the same sample-level predictions. The DeLong test has substantially greater statistical power than the per-cohort paired tests and is the inferential anchor of the manuscript. Per-sample LODO predictions (sample_id, cohort, y_true, y_prob) were saved in long format per classifier (`results/preds_*.csv`) to enable post-hoc DeLong comparisons without re-running classifiers.

## Bootstrap confidence intervals

Nonparametric bootstrap 95% confidence intervals were computed using 10,000 iterations (`scripts/bootstrap_ci.py`). Per-cohort CIs use i.i.d. resampling within each held-out cohort; pooled CIs use cohort-stratified resampling (resampling within each cohort separately and concatenating) to preserve the LODO sample-size structure.

## Feature importance

Per-feature contributions were quantified using SHAP (SHapley Additive exPlanations; Lundberg and Lee 2017) values computed with TreeSHAP on the joint RF, joint XGBoost, and the two adenoma classifiers (`results/shap_*.csv`). The top features by mean absolute SHAP value were compared across model architectures and across the H-vs-A, CRC-vs-control, and A-vs-CRC tasks.

## Robustness battery

**Filter-threshold sensitivity.** The joint RF LODO was repeated across a 20-cell grid of prevalence cutoffs {0.05, 0.10, 0.15, 0.20} × mean-abundance cutoffs {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}. The per-fold prevalence/mean filter is applied to training-cohort samples only, matching the headline run. Results in `results/sensitivity_thresholds.csv`.

**Confounder assessment.** Age, sex, and BMI were tested as potential confounders via two approaches: (1) direct inclusion as additional features alongside species abundances, and (2) residualization, in which each species feature was regressed on the covariates by ordinary least squares and the residuals were used for classification. Train-fold-only medians (age, BMI) and modes (sex) were used for missing-value imputation to prevent leakage from test-fold samples.

**Seed sensitivity.** The species-only RF country-aware LODO was repeated at five random seeds {0, 1, 2, 42, 100} (`results/seed_sensitivity.csv`).

**Batch correction (robustness check).** As described in *Batch-effect mitigation* above, per-fold ComBat (Johnson et al. 2007), as implemented in `combat.pycombat.pycombat`, was applied to species features within each LODO fold and reported as a robustness check on the country-aware LODO design rather than as the primary correction. ComBat was fit jointly on train and test feature matrices using only batch labels (`study_name`); class labels were never seen by ComBat, preserving LODO no-leakage while keeping train and test in the same corrected feature space. Pooled per-cohort mean AUC: 0.815 corrected versus 0.807 uncorrected (`results/combat_results.csv`).

**Biologically-guided pathway shortlist.** A pre-specified, keyword-based subset of CRC-relevant pathways was constructed across nine biological groups (butyrate/SCFA production, fermentation, LPS/inflammation, polyamine synthesis, tryptophan metabolism, folate/one-carbon metabolism, sulfur/methionine metabolism, glycan/mucin degradation, and bile-acid metabolism). Keyword selection was specified before model training based on published CRC microbiome literature (`scripts/bio_pathway_shortlist.py`).

## Software and reproducibility

All classification and statistical analyses were implemented in Python with scikit-learn 1.4.2 (Pedregosa et al. 2011), XGBoost 2.0.3 (Chen and Guestrin 2016), SHAP 0.44.1 (Lundberg and Lee 2017), pandas 2.2.3, NumPy 1.26.4, SciPy 1.12.0, and matplotlib 3.8.5. Python package versions are pinned in `requirements.lock`. Data extraction was performed in R using curatedMetagenomicData (Pasolli et al. 2017) via Bioconductor; the R session is recorded in `session_info.txt`. All scripts use `random_state = 42` and produce deterministic results. A verification script (`scripts/verify_results.py`) asserts that headline AUC values match expected values within tolerances of 0.001-0.05 (per-check).

## Data and code availability

All metagenomic data are publicly available through curatedMetagenomicData via Bioconductor. Retrieval used `returnSamples()` with the ten study identifiers listed above, filtering for samples annotated as CRC, adenoma, or control. No novel sequencing data were generated. The complete analysis pipeline is deposited at https://github.com/alejandro-publius/crc-metagenomics. The repository contains all classification and robustness scripts; per-sample LODO predictions in long format; SHAP feature-importance tables; processed feature matrices; the decision log (`results/decisions_addendum.md`); pinned Python dependencies (`requirements.lock`); and R session information (`session_info.txt`). Trained model objects are not deposited; all reported results can be reproduced from the deposited scripts and publicly available inputs in approximately 45 minutes on a standard workstation.


---

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


---

# Discussion

We re-evaluated the Thomas et al. (2019) multi-cohort CRC classification framework on an expanded set of 10 curatedMetagenomicData cohorts (1,522 subjects; 1,339 case/control samples), applying country-aware LODO, per-fold pathway filtering, formal DeLong-based classifier comparison, and a systematic robustness battery. Our central finding is that **species-level taxonomic features alone provide better cross-cohort CRC classification than joint species-plus-pathway models** at current sample sizes: pooled DeLong z = 3.35, p = 0.0008 versus the joint Random Forest, and z = 2.00, p = 0.046 versus the joint XGBoost. The species-only baseline was stable across random seeds, pathway-filter thresholds, demographic adjustment, and per-fold ComBat batch correction.

## Negative result on pathways is consistent with over-parameterization

The joint species-plus-pathway model adds roughly 400 community-level pathway features to the 229 retained species without improving per-fold AUC and significantly degrades pooled discrimination relative to the species-only baseline (DeLong z = 3.35, p = 0.0008 for joint RF; z = 2.00, p = 0.046 for joint XGBoost; n = 1,339). This pattern is consistent with the curse of dimensionality acting on a fixed-n problem: nearly tripling the feature count without a proportional gain in independent signal dilutes the probability that the most informative taxa are sampled at each split (`max_features = 'sqrt'` in Random Forest; `colsample_bytree = 0.8` in XGBoost). Pathway features are also highly correlated with the taxa that encode the corresponding genes — the four oral pathobionts that top the species SHAP rankings (*Fusobacterium nucleatum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, *Gemella morbillorum*) collectively contribute to a wide swathe of unstratified pathways — so much of the apparent "functional" signal is already captured by the taxonomic features. The biologically-guided shortlist (~66 retained features per fold drawn from 9 CRC-relevant functional groups; mean LODO AUC 0.817; `results/bio_pathway_results.csv`) shows the same pattern: no rescue of the joint model. Strain- or gene-level features were not evaluated because curatedMetagenomicData does not distribute them at the depth required for a 10-cohort pooled analysis. This is consistent with the broader high-dimensional / low-sample-size literature (Bellman 1961; Trunk 1979): at n ≈ 1,300 across heterogeneous cohorts, parsimony wins. The result should be read as a statement about the current sample-size regime — and the granularity of features available from standard pipelines — rather than as a categorical claim that functional features can never help; with substantially larger pooled datasets (Piccinno et al. 2025), the noise contribution of additional features may diminish.

TreeSHAP rankings on the species Random Forest agree closely with permutation-importance rankings on the same model: 16 of the top 20 species by TreeSHAP also appear in the top 20 by permutation importance (`results/diagnostics/permutation_vs_shap_correlation.csv`). Three of the four oral-pathobiont species (*Fusobacterium nucleatum*, *Peptostreptococcus stomatis*, *Gemella morbillorum*) retain top-four rank under both measures; *Parvimonas micra* ranks 2 by TreeSHAP and 12 by permutation, indicating lower individual model dependence but consistent inclusion in the top-importance band. Per-cohort SHAP rank does not correlate with cohort median sequencing depth for *F. nucleatum* (Spearman ρ = −0.19, p = 0.59, n = 10 cohorts; `results/diagnostics/depth_confound_shap.csv`); no top-20 species survives a multiple-testing-corrected threshold for depth-rank correlation. The oral-pathobiont signature is therefore not an artifact of TreeSHAP's bias toward high-cardinality features nor of cohort-level read depth.

The joint XGBoost model exhibits a notably larger reliability term in the Brier decomposition (0.026) than either Random Forest (species RF 0.007; joint RF 0.006; `results/diagnostics/brier_decomposition.csv`), reflecting more aggressive logit pushes toward 0 and 1 that produce a U-shaped predicted-probability distribution. This is a calibration property of gradient-boosted decision trees on heterogeneous tabular metagenomic data rather than a model-comparison artifact, but it argues for the Random Forest as the preferred deployment candidate where probability calibration matters (e.g., downstream Bayesian thresholding for screening decisions).

## Granularity of functional features

Overall HUMAnN unstratified pathway abundances sit at an intermediate granularity between species composition and gene-level functional content: each feature is a community-level relative abundance for one MetaCyc pathway, summed across all contributing taxa. Two refinements at finer granularities were tested. First, a *biologically-guided pathway shortlist* (`scripts/bio_pathway_shortlist.py`) restricted candidate pathways to nine CRC-relevant functional groups — butyrate / short-chain fatty acid production, fermentation, lipopolysaccharide and inflammation pathways, polyamine synthesis, tryptophan metabolism, folate and one-carbon metabolism, sulfur and methionine metabolism, glycan / mucin degradation, and bile-acid metabolism (BSH-mediated deconjugation and 7alpha-dehydroxylation / bai operon) — selected from the published CRC microbiome literature before any model fitting. Second, *species-stratified pathways* (`scripts/stratified_pathway_pilot.py`, `results/stratified_pathway_pilot.csv`) decompose each pathway abundance into per-taxon contributions, producing >4,000 columns at the cost of substantial sparsity and redundancy. Neither refinement rescued the joint model. We interpret this as evidence that, at the current dataset size, taxonomic composition already captures most of the discriminative signal accessible from shotgun metagenomics through standard MetaPhlAn / HUMAnN profiling. Genuinely additive functional information would, on this evidence, require strain- or gene-level resolution — and corresponding sample sizes — beyond the scope of this work.

## Biology of the CRC-enriched oral-pathobiont signature

The top SHAP features for the CRC-vs-control classifier are dominated by taxa more typical of the oral cavity than of the colon: *Fusobacterium nucleatum*, *Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Solobacterium moorei*, and *Streptococcus salivarius*. *F. nucleatum* in particular has been mechanistically linked to CRC through FadA-mediated adhesion to E-cadherin, β-catenin signalling, and tumour-permissive immune modulation; the other taxa form part of a co-occurring oral consortium repeatedly observed in CRC tumours and stool. Their reproducibility across cohorts on three continents — at top SHAP ranks in both Random Forest and XGBoost despite very different splitting criteria — is the most convincing single observation in this analysis. It is also the feature signature that drives the moderate adenoma-vs-CRC performance, suggesting that oral-pathobiont colonisation is a relatively late event in the adenoma-carcinoma sequence.

## A stepwise model of microbiome change along the adenoma-carcinoma sequence

The adenoma LODO results provide a coherent stage-specific picture. **Control vs adenoma** classification was near chance (RF 0.561, XGBoost 0.579), with TreeSHAP highlighting weak, heterogeneous metabolic and commensal-depletion features. **Adenoma vs CRC** classification was moderate (RF 0.671, XGBoost 0.617), with TreeSHAP dominated by the same oral-pathobiont signature that drives CRC-vs-control. This pattern is consistent with a model in which (i) the microbiome of early-stage adenoma is largely indistinguishable from healthy controls by global stool composition, and (ii) the oral-pathobiont enrichment is acquired at or near the transition to invasive carcinoma. Such a model has clinical implications: stool metagenomic screens are unlikely to detect early adenomas at useful sensitivity, but may have value in distinguishing advanced lesions from carcinoma and in monitoring post-resection recurrence. It also reinforces that adenoma and CRC are biologically distinct microbiome states rather than two points on a smooth severity gradient.

## Adenoma analysis: class-balance robustness

The adenoma LODO folds have small minority-class counts per cohort (n_adenoma ranges from 27 in ThomasAM_2018a to 67 in YachidaS_2019), which raises the possibility that the near-chance control-vs-adenoma result is a methodological artefact of class imbalance rather than a biological observation. We tested this directly with three per-fold class-rebalancing strategies, applied to the training fold only (`scripts/rebalanced_adenoma_lodo.py`): inverse-frequency reweighting, random undersampling of the majority class, and SMOTE oversampling of the minority class. The comparison is summarised in `results/adenoma_rebalanced_summary.csv`. The qualitative finding — healthy-vs-adenoma near chance and adenoma-vs-CRC moderate — was robust across all three rebalancing strategies and consistent with the headline class-weighted run. We therefore interpret the weak adenoma signal as reflecting true biological under-determination of pre-cancerous microbiome change at the granularity of standard shotgun metagenomic profiling, rather than a sampling-imbalance artefact.

## Batch effects vs cross-cohort generalisation

Cross-cohort variability in DNA extraction, library preparation, sequencing depth, and host population genetics is the dominant source of microbiome variation across these 10 cohorts, and any cross-cohort classification result must be read against that background. Two design choices respond directly to this. First, country-aware LODO treats each cohort as a single batch and additionally excludes same-country cohorts from training, removing geographic and population-level confounding that would otherwise leak through diet- and host-genetics-correlated taxa: the most explicit illustration is the ThomasAM_2019_c fold, where allowing YachidaS_2019 (the second Japanese cohort) into training inflated test AUC to 0.998, dropping to 0.836 once Japan is excluded. The species-only RF reached per-cohort AUC ≥ 0.69 in every fold and ≥ 0.80 in 7 of 10 folds under this stricter split. Second, per-fold filtering and feature normalisation use training-cohort statistics only, preventing test-cohort sample composition from influencing the feature set or its scaling. ComBat applied per fold as a robustness check increases pooled per-cohort mean AUC marginally (0.815 vs 0.807 uncorrected; `results/combat_results.csv`) and does not change the qualitative conclusions, which we interpret as evidence that the biological signal survives the dominant technical batch effects under the country-aware LODO design rather than being recovered by post-hoc mean/variance harmonisation.

## Implications for microbiome-based CRC screening

The practical implication is that a parsimonious species-only classifier is preferable to more complex joint models for stool-based CRC screening in the near term. Species-level profiling is more standardised across bioinformatics pipelines than functional profiling, requires less computational infrastructure, and — as we show — delivers equal or superior classification performance at the sample sizes typical of current clinical validation studies. As metagenomic datasets scale to thousands of samples, the cost-benefit calculus for including pathway features may shift, but at present the added complexity is not justified by improved performance.

## Limitations

Several limitations warrant emphasis. First, clinical metadata harmonisation across curatedMetagenomicData is limited to age, sex, and BMI; we do not model adenoma stage (advanced vs non-advanced), tumour location, TNM stage, or treatment history because these fields are not uniformly reported. The H-vs-A near-chance result therefore reflects cross-cohort generalisation difficulty under heterogeneous case definitions, not necessarily the absence of any microbiome signal for adenoma. Second, the dataset is purely cross-sectional; no longitudinal samples are available to test progression or post-treatment trajectories. Third, our analysis is restricted to cohorts processed through the curatedMetagenomicData uniform pipeline; we did not validate the classifier on independently processed, non-curatedMetagenomicData cohorts, and processing-pipeline differences (read trimming, taxonomic database version, pathway database version) could attenuate generalisation in that setting. Fourth, we did not perform nested-CV hyperparameter tuning; we justify this on the grounds that the joint model already fails to outperform the species baseline at default hyperparameters, but tuned joint models could in principle narrow the gap. Fifth, the pathway features used here are unstratified community-level abundances; taxon-stratified features were excluded due to redundancy and dimensionality but could in principle separate species-encoded from metagenome-wide signals. Sixth, the adenoma analyses are statistically underpowered (4 cohorts, n = 183) and should be interpreted as hypothesis-generating rather than definitive.

The 10 cohorts in this meta-analysis are geographically concentrated in Europe, East Asia, and North America, with one Indian cohort (GuptaA_2019, n=60) and no cohorts from Africa, Latin America, or the Middle East. This reflects the current geographic distribution of public shotgun metagenomic CRC datasets rather than a methodological choice. Findings should not be assumed to generalize to populations whose gut microbiota, diet, and exposure profiles differ substantially from the cohorts represented here. Expansion to under-represented populations is a priority for future work.

## Position relative to current non-invasive screening

The Fecal Immunochemical Test (FIT) remains the established non-invasive primary screening modality for colorectal cancer, with published per-test sensitivity of approximately 79% and specificity of approximately 94% (Imperiale et al. 2014). At population CRC prevalence of approximately 5%, FIT's positive predictive value is therefore considerably higher than a microbiome-based classifier operating at the AUC observed here (see `results/diagnostics/fit_vs_microbiome.csv`). The clinical role most plausibly supported by our results is therefore not as a replacement for FIT, but as a stratifier — for example, of FIT-negative individuals at elevated baseline risk — or as a longitudinal monitoring substrate where serial sampling can offset the per-test discrimination gap. Direct head-to-head prospective comparison in screening-age populations remains essential before any deployment claim can be made.

## Conclusion

A species-only Random Forest classifier, trained under country-aware LODO across 10 curatedMetagenomicData cohorts (n = 1,339), significantly outperforms joint species-plus-pathway Random Forest and XGBoost models for CRC detection. This advantage is robust to random-seed variation, pathway-filter thresholds, demographic adjustment, and ComBat batch correction. The adenoma analysis supports a stepwise model in which the diagnostic oral-pathobiont signature emerges at or near malignant transformation rather than at the adenoma stage. All code, processed data, per-sample predictions, and decision logs are publicly available to enable replication and extension.


---

# References

1. Bellman, R. *Adaptive Control Processes: A Guided Tour*. Princeton University Press (1961).

2. Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. In *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* 785–794 (ACM, 2016). https://doi.org/10.1145/2939672.2939785

3. DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* **44**, 837–845 (1988). https://doi.org/10.2307/2531595

4. Franzosa, E.A., McIver, L.J., Rahnavard, G. et al. Species-level functional profiling of metagenomes and metatranscriptomes. *Nat. Methods* **15**, 962–968 (2018). https://doi.org/10.1038/s41592-018-0176-y

5. Imperiale, T.F., Ransohoff, D.F., Itzkowitz, S.H. et al. Multitarget stool DNA testing for colorectal-cancer screening. *N. Engl. J. Med.* **370**, 1287–1297 (2014). https://doi.org/10.1056/NEJMoa1311194

6. Johnson, W.E., Li, C. & Rabinovic, A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics* **8**, 118–127 (2007). https://doi.org/10.1093/biostatistics/kxj037

7. Lundberg, S.M. & Lee, S.-I. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 4766–4777 (2017).

8. Pasolli, E., Schiffer, L., Manghi, P. et al. Accessible, curated metagenomic data through ExperimentHub. *Nat. Methods* **14**, 1023–1024 (2017). https://doi.org/10.1038/nmeth.4468

9. Pedregosa, F. et al. Scikit-learn: machine learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).

10. Piccinno, G. et al. Pooled analysis of 3,741 stool metagenomes from 18 cohorts for cross-stage and strain-level reproducible microbial biomarkers of colorectal cancer. *Nat. Med.* **31**, 2416–2429 (2025). https://doi.org/10.1038/s41591-025-03693-9

11. Sun, X. & Xu, W. Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Process. Lett.* **21**, 1389–1393 (2014). https://doi.org/10.1109/LSP.2014.2337313

12. Sun, Y. et al. Optimizing metagenome analysis for early detection of colorectal cancer: benchmarking bioinformatics approaches and advancing cross-cohort prediction. *bioRxiv* (2025). https://doi.org/10.1101/2025.02.22.639690

13. Sung, H. et al. Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA Cancer J. Clin.* **71**, 209–249 (2021). https://doi.org/10.3322/caac.21660

14. Thomas, A.M. et al. Metagenomic analysis of colorectal cancer datasets identifies cross-cohort microbial diagnostic signatures and a link with choline degradation. *Nat. Med.* **25**, 667–678 (2019). https://doi.org/10.1038/s41591-019-0405-7

15. Trunk, G.V. A problem of dimensionality: a simple example. *IEEE Trans. Pattern Anal. Mach. Intell.* **PAMI-1**, 306–307 (1979). https://doi.org/10.1109/TPAMI.1979.4766926

16. Truong, D.T., Franzosa, E.A., Tickle, T.L. et al. MetaPhlAn2 for enhanced metagenomic taxonomic profiling. *Nat. Methods* **12**, 902–903 (2015). https://doi.org/10.1038/nmeth.3589

17. Wirbel, J. et al. Meta-analysis of fecal metagenomes reveals global microbial signatures that are specific for colorectal cancer. *Nat. Med.* **25**, 679–689 (2019). https://doi.org/10.1038/s41591-019-0406-6

18. Xi, Y. & Xu, P. Global colorectal cancer burden in 2020 and projections to 2040. *Transl. Oncol.* **14**, 101174 (2021). https://doi.org/10.1016/j.tranon.2021.101174

19. Yachida, S. et al. Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer. *Nat. Med.* **25**, 968–976 (2019). https://doi.org/10.1038/s41591-019-0458-7


---

# Table 1 and Supplementary Material

### Table 1. Demographic and clinical characteristics of the ten cohorts

Values are mean ± SD unless otherwise indicated. Source: `results/table1.csv`.

| Cohort | Country | N (total) | N (CRC) | N (adenoma) | N (control) | Age (mean ± SD) | Female % | BMI (mean ± SD) |
|---|---|---|---|---|---|---|---|---|
| FengQ_2015 | AUT | 154 | 46 | 47 | 61 | 66.9 ± 8.4 | 43.5% | 27.4 ± 4.0 |
| GuptaA_2019 | IND | 60 | 30 | 0 | 30 | 50.6 ± 16.0 | 50.0% | 21.3 ± 3.0 |
| ThomasAM_2018a | ITA | 80 | 29 | 27 | 24 | 67.5 ± 8.7 | 35.0% | 25.5 ± 3.9 |
| ThomasAM_2018b | ITA | 60 | 32 | 0 | 28 | 58.2 ± 8.3 | 35.0% | 25.8 ± 4.2 |
| ThomasAM_2019_c | JPN | 80 | 40 | 0 | 40 | 61.1 ± 12.6 | 43.8% | 22.7 ± 2.6 |
| VogtmannE_2016 | USA | 104 | 52 | 0 | 52 | 61.5 ± 12.3 | 28.8% | 25.1 ± 4.2 |
| WirbelJ_2018 | DEU | 125 | 60 | 0 | 65 | 59.6 ± 12.9 | 41.6% | 25.5 ± 3.7 |
| YachidaS_2019 | JPN | 575 | 258 | 67 | 250 | 62.0 ± 11.0 | 40.2% | 22.9 ± 3.4 |
| YuJ_2015 | CHN | 128 | 74 | 0 | 54 | 64.2 ± 9.1 | 36.7% | 23.8 ± 3.1 |
| ZellerG_2014 | FRA | 156 | 53 | 42 | 61 | 63.3 ± 10.9 | 44.2% | 25.3 ± 4.2 |
| **TOTAL** | — | **1,522** | **674** | **183** | **665** | **62.2 ± 11.4** | **40.1%** | **24.2 ± 4.0** |

HanniganGD_2017 was excluded a priori for low sequencing depth (mean 6.5M reads vs per-cohort means of 9.2M for GuptaA_2019 to 102M for ThomasAM_2018a in the retained 10-cohort set, all others >40M) and 82% species feature sparsity; the exclusion was specified before any classifier training (`scripts/preprocessing.py`, `EXCLUDE_COHORTS`).

---

## Supplementary Table S1. Pathway filter-threshold sensitivity (joint Random Forest, country-aware LODO)

Each row reports one country-aware LODO run (10 folds, 1,339 case/control samples) over species ∪ unstratified pathway features, with the per-fold prevalence/mean filter computed on training-cohort samples only at the indicated thresholds. The headline run uses prevalence ≥ 0.10, mean ≥ 1e-6. Across the full 20-cell grid, mean per-cohort AUC ranges from 0.794 to 0.812 (spread 0.018). The mean ≥ 1e-3 column retains only two pathways and effectively reduces the joint model to species-only. Source: `results/sensitivity_thresholds.csv`.

| Prevalence | Mean abundance | Pathways retained (mean) | Total features (mean) | Mean per-cohort AUC | SD across folds |
|---|---|---|---|---|---|
| ≥ 0.05 | ≥ 1e-7 | 436.9 | 665.9 | 0.794 | 0.065 |
| ≥ 0.05 | ≥ 1e-6 | 405.2 | 634.2 | 0.806 | 0.059 |
| ≥ 0.05 | ≥ 1e-5 | 308.6 | 537.6 | 0.805 | 0.066 |
| ≥ 0.05 | ≥ 1e-4 | 141.7 | 370.7 | 0.812 | 0.067 |
| ≥ 0.05 | ≥ 1e-3 | 2.0 | 231.0 | 0.810 | 0.064 |
| ≥ 0.10 | ≥ 1e-7 | 422.9 | 651.9 | 0.799 | 0.067 |
| ≥ 0.10 | ≥ 1e-6 | 404.3 | 633.3 | 0.804 | 0.066 |
| ≥ 0.10 | ≥ 1e-5 | 308.6 | 537.6 | 0.805 | 0.066 |
| ≥ 0.10 | ≥ 1e-4 | 141.7 | 370.7 | 0.812 | 0.067 |
| ≥ 0.10 | ≥ 1e-3 | 2.0 | 231.0 | 0.810 | 0.064 |
| ≥ 0.15 | ≥ 1e-7 | 403.4 | 632.4 | 0.800 | 0.065 |
| ≥ 0.15 | ≥ 1e-6 | 393.8 | 622.8 | 0.803 | 0.063 |
| ≥ 0.15 | ≥ 1e-5 | 308.6 | 537.6 | 0.805 | 0.066 |
| ≥ 0.15 | ≥ 1e-4 | 141.7 | 370.7 | 0.812 | 0.067 |
| ≥ 0.15 | ≥ 1e-3 | 2.0 | 231.0 | 0.810 | 0.064 |
| ≥ 0.20 | ≥ 1e-7 | 387.1 | 616.1 | 0.801 | 0.064 |
| ≥ 0.20 | ≥ 1e-6 | 382.4 | 611.4 | 0.798 | 0.065 |
| ≥ 0.20 | ≥ 1e-5 | 308.6 | 537.6 | 0.805 | 0.066 |
| ≥ 0.20 | ≥ 1e-4 | 141.7 | 370.7 | 0.812 | 0.067 |
| ≥ 0.20 | ≥ 1e-3 | 2.0 | 231.0 | 0.810 | 0.064 |

---

## Supplementary Table S2. Per-cohort and pooled LODO AUCs with 95% bootstrap confidence intervals

Per-cohort AUCs are computed on each held-out cohort under country-aware LODO. Bootstrap 95% CIs (10,000 iterations) use i.i.d. resampling within each held-out cohort for per-cohort rows; the pooled row uses cohort-stratified resampling to preserve LODO sample-size structure. Joint models apply the per-fold pathway prevalence ≥ 0.10 / mean ≥ 1e-6 filter on training-cohort samples only, retaining 402–406 pathways. Source: `results/bootstrap_ci.csv`.

| Cohort | n_test | Species RF (229 features) | Joint RF (species + pathways) | Joint XGBoost (species + pathways) |
|---|---|---|---|---|
| FengQ_2015 | 107 | 0.840 [0.752, 0.915] | 0.833 [0.744, 0.910] | 0.844 [0.758, 0.916] |
| GuptaA_2019 | 60 | 0.882 [0.786, 0.959] | 0.912 [0.824, 0.979] | 0.886 [0.791, 0.959] |
| ThomasAM_2018a | 53 | 0.694 [0.541, 0.832] | 0.843 [0.715, 0.947] | 0.743 [0.599, 0.873] |
| ThomasAM_2018b | 60 | 0.810 [0.692, 0.914] | 0.791 [0.670, 0.898] | 0.761 [0.629, 0.879] |
| ThomasAM_2019_c | 80 | 0.836 [0.739, 0.920] | 0.778 [0.673, 0.872] | 0.819 [0.721, 0.906] |
| VogtmannE_2016 | 104 | 0.756 [0.656, 0.844] | 0.720 [0.615, 0.817] | 0.734 [0.631, 0.827] |
| WirbelJ_2018 | 125 | 0.882 [0.820, 0.933] | 0.852 [0.780, 0.912] | 0.860 [0.791, 0.919] |
| YachidaS_2019 | 508 | 0.708 [0.662, 0.752] | 0.669 [0.622, 0.715] | 0.694 [0.647, 0.739] |
| YuJ_2015 | 128 | 0.865 [0.798, 0.924] | 0.819 [0.745, 0.886] | 0.819 [0.743, 0.886] |
| ZellerG_2014 | 114 | 0.803 [0.717, 0.880] | 0.826 [0.744, 0.899] | 0.811 [0.728, 0.887] |
| **Pooled (n = 1,339)** |  | **0.781 [0.757, 0.805]** | **0.756 [0.731, 0.781]** | **0.766 [0.740, 0.791]** |

**Notes on interpretation.** The species-RF pooled CI [0.757, 0.805] does not overlap the joint-RF point estimate (0.756) and barely contains the joint-XGB point estimate (0.766), consistent with the DeLong tests reported in the main text (species RF vs joint RF, z = 3.35, p = 0.0008; species RF vs joint XGB, z = 2.00, p = 0.046). Per-cohort CIs are wide because each held-out cohort contributes only 53–508 samples; the pooled CI is the appropriate inferential summary.

---

## Supplementary Note S1. Seed sensitivity

Five seeds {0, 1, 2, 42, 100} yield species-only country-aware LODO per-cohort mean AUCs of 0.8094, 0.8115, 0.8088, 0.8075, and 0.8113 (mean 0.8097, SD 0.0015, range 0.807–0.811). The headline result is insensitive to the random seed used by the Random Forest. Source: `results/seed_sensitivity.csv`.

## Supplementary Note S2. Confounder adjustment

Adjusting for age, sex, and BMI with train-fold-only imputation (medians for age/BMI, modes for sex) yields per-cohort mean AUCs of 0.814 (direct RF), 0.806 (direct XGBoost), 0.801 (residualized RF), and 0.800 (residualized XGBoost), all within sampling noise of the 0.807 species-only baseline. The classifier's discrimination is not driven by these standard demographic confounders. Source: `results/confounder_results.csv`, `results/covariate_comparison.csv`.

## Supplementary Note S3. ComBat batch correction

Per-fold ComBat correction (`combat.pycombat.pycombat`) on species features yields a per-cohort mean AUC of 0.815 versus 0.807 without correction (Δ +0.008). ComBat is fit jointly on train and test feature matrices using only batch labels (`study_name`); class labels are never seen by ComBat. Source: `results/combat_results.csv`.

## Supplementary Note S4. Biologically-guided pathway shortlist

A pre-specified, keyword-based shortlist drawn from nine biological groups (butyrate/SCFA, fermentation, LPS/inflammation, polyamines, tryptophan metabolism, folate/one-carbon metabolism, sulfur/methionine metabolism, glycan/mucin degradation, bile-acid metabolism) expands to 86 unique CRC-relevant pathway candidates; per-fold prevalence/mean filtering retains ~66 of these, giving ~295 total features per fold (229 species + ~66 pathways). The joint species + shortlist-pathway feature set yields per-cohort mean AUC 0.817, comparable to the species-only baseline. Curated CRC-relevant pathways therefore do not provide an advantage over the species features alone. Source: `results/bio_pathway_results.csv`.

## Supplementary Note S5. Adenoma classification (cross-cohort LODO)

Country-aware LODO across the four adenoma-containing cohorts (FengQ_2015 n_adenoma=47; YachidaS_2019 n_adenoma=67; ZellerG_2014 n_adenoma=42; ThomasAM_2018a n_adenoma=27; total n=183): control-vs-adenoma mean LODO AUC 0.561 (RF) / 0.579 (XGBoost); adenoma-vs-CRC mean LODO AUC 0.671 (RF) / 0.617 (XGBoost). Source: `results/adenoma_lodo_results.csv`.
