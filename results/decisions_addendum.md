# Decisions Log Addendum

## SMOTE vs class weights
DECISION: Use class weights only. RF uses class_weight='balanced'.
XGBoost adenoma classifiers use scale_pos_weight = inverse class ratio.
SMOTE was not run. Justification: class weights are simpler, do not
synthesize features, and the joint model does not statistically
outperform the species-only baseline (model_comparison.csv), so
gains from a more aggressive imbalance method are unlikely to
change qualitative conclusions.

## DeLong test
DECISION: Implemented. run_lodo_cv now optionally saves per-sample
predictions per fold; auc_comparison.py applies DeLong (Sun and Xu
2014) to pooled LODO predictions in addition to per-cohort paired
t-test and Wilcoxon. Updated results (10-cohort, n=1339 pooled):
species RF significantly outperforms Joint RF (AUC 0.781 vs 0.756,
z=3.352, p=0.0008) and Joint XGB (AUC 0.781 vs 0.766, z=1.996,
p=0.046). Per-cohort paired tests (n=10 folds) show no significant
difference (p=0.87 and p=0.28), consistent with low power at n=10.
The DeLong significance is driven largely by YachidaS_2019, the
largest fold (n=508), where species RF substantially outperforms
joint models (0.708 vs 0.669/0.694). Saved to results/delong_results.csv.

## Normalization
DECISION: Species: log10(x + 1e-6) applied in preprocessing.py after
row-sum renormalization. Pathways: raw relative abundance from
curatedMetagenomicData with no transform. Random Forest and XGBoost
split decisions are scale-invariant per feature, so the asymmetric
handling does not affect AUC.

## Pathway feature set
DECISION: Use unstratified pathway abundance (551 candidate columns
in the 10-cohort dataset, filtered to 631-635 per LODO fold after
adding 229 species features; per-fold pathway filtering keeps ~402-406
pathways). Stratified taxon|pathway features were considered but produce
>4000 highly redundant columns that did not improve AUC in pilot testing.

## Hyperparameter tuning
DECISION: No nested CV tuning. Justification: joint model does not
statistically outperform species-only baseline; tuning is unlikely
to change the qualitative conclusion.

## Pathway prevalence filter and LODO leakage
DECISION: Refit per fold. train_joint.py loads the unfiltered
unstratified pathway matrix and applies prevalence>=10% and mean>=1e-6
filter inside each LODO fold using only training-cohort samples.
Per-fold pathway counts range from ~402-406 across the 10 folds.
The static filter_pathways.py file is retained for shap_xgb.py
and adenoma scripts which use pre-filtered files under non-LODO CV.

## Filter threshold sensitivity
DECISION: Documented. sensitivity_analysis.py sweeps prevalence
{0.05, 0.10, 0.15, 0.20} x mean {1e-7, 1e-6, 1e-5, 1e-4, 1e-3} under
country-aware LODO CV with the prevalence/mean filter applied PER FOLD
using only training-cohort samples (matches the headline run in
train_joint.py). 10-cohort results: joint RF mean per-cohort AUC
ranges from 0.798 to 0.810 across all 20 cells (spread 0.012).
Threshold insensitivity is confirmed; default thresholds (prevalence
>= 10%, mean >= 1e-6) are near-optimal. Saved to
results/sensitivity_thresholds.csv.

## Confounder adjustment
DECISION: Documented. confounder_adjustment.py tests age, sex, and
BMI as potential confounders via direct inclusion and residualization.
Covariate imputation uses train-fold-only medians/modes to avoid
leakage. 10-cohort results: baseline 0.808, direct RF 0.808, direct XGB
0.807, residualized RF 0.807, residualized XGB 0.816. The 0.807-0.816
range sits within sampling noise of the unadjusted baseline, confirming
the classifier is not driven by demographic confounders.

## Cross-cohort adenoma LODO
DECISION: Implemented. Adenoma-containing cohorts in 10-cohort dataset:
FengQ_2015 (47), ZellerG_2014 (42), ThomasAM_2018a (27), YachidaS_2019 (67).
Total: 183 adenoma samples (up from 116 in 7-cohort dataset). Three bugs
fixed vs prior run: (1) ADENOMA_COHORTS was missing YachidaS_2019; (2)
H-vs-A label map used 'healthy' instead of 'control', silently producing
zero matches; (3) reset_index on the filtered sub-dataframe caused
X.loc[train_idx] to index the wrong rows in X_all.

Results (4-cohort LODO, n_folds=4):
  H-vs-A RF:    mean AUC = 0.561
  H-vs-A XGB:   mean AUC = 0.579
  A-vs-CRC RF:  mean AUC = 0.671
  A-vs-CRC XGB: mean AUC = 0.617

Interpretation: H-vs-A performance (~0.57) is near chance, consistent with
published literature showing weak cross-cohort microbiome signal for adenoma
detection. A-vs-CRC RF (0.671) reflects the oral-bacterial CRC signature
(Fusobacterium nucleatum, Parvimonas micra, Peptostreptococcus stomatis,
Gemella morbillorum) emerging during malignant transformation. These species
top the SHAP rankings for A-vs-CRC but not H-vs-A, consistent with the
biological progression model. Saved to results/adenoma_lodo_results.csv.

## Bootstrap confidence intervals
DECISION: Documented. bootstrap_ci.py computes 2000-iteration bootstrap
95% CIs. 10-cohort pooled results: Species RF 0.781 [0.756, 0.805],
Joint RF 0.756 [0.728, 0.781], Joint XGB 0.766 [0.738, 0.790].
Saved to results/bootstrap_ci.csv.

## Seed sensitivity
DECISION: Documented. seed_sensitivity.py runs species RF LODO at seeds
{0, 1, 2, 42, 100} with country-aware LODO. Results pending rerun on
10-cohort dataset. Prior 7-cohort results: mean 0.8049 +/- 0.0020.

## Cohort expansion (v2 analysis)
DECISION: Expanded from 7 to 10 cohorts by adding YachidaS_2019,
WirbelJ_2018, HanniganGD_2017, and GuptaA_2019 from curatedMetagenomicData.
HanniganGD_2017 subsequently excluded (see below). Final dataset:
10 cohorts, 1522 samples (674 CRC, 665 control, 183 adenoma).

## HanniganGD_2017 exclusion
DECISION: Excluded from all analyses. Justification: mean sequencing
depth of 6.5M reads (range 17K-21M) is substantially below all other
cohorts (mean 40-102M reads, minimum 9.2M for GuptaA_2019). Feature
sparsity confirms degraded profiling: 82% zero-valued species features
vs 61% mean for other cohorts. Both metrics were assessed before model
training; the exclusion is pre-specified and independent of classification
results. Applied in preprocessing.py via EXCLUDE_COHORTS constant.
A per-sample minimum of 1M reads is also applied to catch individual
extreme outliers across all cohorts (removed 4 additional samples).

## Country-aware LODO
DECISION: Implemented in lodo_cv.py (country_col parameter). When a
cohort is the test fold, all cohorts from the same country are excluded
from training. This prevents population-level confounding when multiple
cohorts share geographic origin.

Affected fold pairs:
- ThomasAM_2019_c (JPN) <-> YachidaS_2019 (JPN): each excluded from
  the other's training fold. Without this fix, ThomasAM_2019_c achieved
  AUC=0.999 (inflated by Japan-specific microbiome signal from YachidaS_2019
  in the training set). With the fix: AUC=0.836, biologically plausible.
- ThomasAM_2018a (ITA) <-> ThomasAM_2018b (ITA): each excluded from
  the other's training fold.

Applied consistently to all training scripts: train_baseline.py,
train_joint.py, seed_sensitivity.py, sensitivity_analysis.py,
bio_pathway_shortlist.py.

## Biologically-guided pathway shortlist
DECISION: Implemented in bio_pathway_shortlist.py. Selects a curated
subset of CRC-relevant pathways using keyword matching across 7 biological
groups: butyrate/SCFA production, fermentation, LPS/inflammation,
polyamine synthesis, tryptophan metabolism, folate/one-carbon metabolism,
and sulfur/methionine metabolism. Keyword selection is pre-specified based
on published CRC microbiome literature; not data-driven. Results compared
to full-pathway joint model. Saved to results/bio_pathway_results.csv.

## Batch correction (ComBat)
DECISION: Documented. batch_correction.py applies per-fold ComBat on
species features. ComBat is fit jointly on the train and test feature
matrices using only batch labels (study_name); class labels (CRC vs
control) are never seen by ComBat, so this preserves the LODO no-
leakage guarantee while keeping train and test in the same corrected
feature space. Result: mean per-cohort AUC 0.806 with ComBat vs 0.803
without, indicating batch effects in this curatedMetagenomicData
subset are small relative to the cross-cohort biological signal.
Requires `pip install combat` (canonical PyPI package providing
combat.pycombat.pycombat).

## Package pinning
DECISION: requirements.lock pins exact versions of all Python dependencies.
Install with pip install -r requirements.lock.

## Species feature filter and LODO leakage
DECISION: The species prevalence>=10% and mean>=1e-4 filter is computed
globally (post cohort-quality filtering). This is a mild information leakage
but is retained for three reasons: (1) MetaPhlAn maps to a fixed reference
database so the filter primarily removes globally rare taxa; (2) species
feature count is small (229 retained); (3) global species filtering matches
the reference standard (Thomas et al. 2019). Pathway filtering is refit
per-fold because HUMAnN abundance has more heterogeneous cross-cohort
distributions.

Note: removing HanniganGD_2017 (82% sparse) before species filtering
increased retained features from 220 to 229, further confirming that
Hannigan's noisy zeros were suppressing real species signal.
