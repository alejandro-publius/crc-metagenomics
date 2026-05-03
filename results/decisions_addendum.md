# Decisions Log Addendum (final state)

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
2014) to pooled LODO predictions (n=646) in addition to the per-cohort
paired t-test and Wilcoxon (n=7). Result: species RF significantly
outperforms both joint models on the pooled ROC (p=0.004 vs Joint RF,
p=0.013 vs Joint XGB), while the n=7 per-cohort paired tests do not
detect a difference (p>0.4) due to low power. Saved to
results/delong_results.csv.

## Normalization
DECISION: Species: log10(x + 1e-6) applied in preprocessing.py after
row-sum renormalization. Pathways: raw relative abundance from
curatedMetagenomicData with no transform. Random Forest and XGBoost
split decisions are scale-invariant per feature, so the asymmetric
handling does not affect AUC.

## Pathway feature set
DECISION: Use unstratified pathway abundance (540 candidate columns
in the joint model, filtered to 402-406 per LODO fold by prevalence
>=10% and mean>=1e-6 refit on training cohorts; see "Pathway
prevalence filter and LODO leakage" below). Stratified taxon|pathway
features were considered but produce 4589 highly redundant columns
that did not improve AUC in pilot testing.

## Hyperparameter tuning
DECISION: No nested CV tuning. Defaults documented in
adenoma_go_nogo_memo.md. Justification: joint model does not
statistically outperform species-only baseline; tuning is unlikely
to change the qualitative conclusion.

## Pathway prevalence filter and LODO leakage
DECISION: Refit per fold. train_joint.py now loads the unfiltered
unstratified pathway matrix (540 candidate columns) and applies the
prevalence>=10% and mean>=1e-6 filter inside each LODO fold using
only training-cohort samples, via the feature_filter_fn hook added
to run_lodo_cv. Per-fold pathway counts range from 402 to 406 across
the 7 folds, vs 405 under the previous global filter. Headline AUCs
are essentially unchanged (Joint RF 0.783 -> 0.785, Joint XGB 0.790
-> 0.784) and DeLong conclusions hold: species RF significantly
better, p=0.004 vs Joint RF and p=0.008 vs Joint XGB on pooled
predictions. The static filter_pathways.py file is retained because
shap_xgb.py and the adenoma scripts use the pre-filtered file under
non-LODO 5-fold CV, where this leakage concern does not apply.

## Filter threshold sensitivity
DECISION: Documented. sensitivity_analysis.py sweeps prevalence
{0.05, 0.10, 0.15, 0.20} x mean {1e-7, 1e-6, 1e-5, 1e-4, 1e-3} under
LODO CV with the prevalence/mean filter applied PER FOLD using only
training-cohort samples (matches the headline run in train_joint.py).
Joint RF mean per-cohort AUC ranges from 0.773 to 0.811 across all 20
cells (spread 0.038); excluding the degenerate mean >= 1e-3 column
(which retains only ~2 pathways and effectively reduces the joint
model to species-only at AUC 0.811), the spread tightens to 0.016
(range 0.773 to 0.789) across the 16 substantively non-degenerate
cells. Default thresholds (prevalence >= 10%, mean >= 1e-6) yield
0.785 and are near-optimal.

## Confounder adjustment
DECISION: Documented. confounder_adjustment.py tests age, sex, and
BMI as potential confounders via direct inclusion and residualization.
Covariate imputation uses train-fold-only medians/modes to avoid
leakage. Results: baseline 0.803, direct RF 0.808, direct XGB 0.807,
residualized RF 0.807, residualized XGB 0.816. The 0.807-0.816 range
sits within sampling noise of the unadjusted baseline, confirming the
classifier is not driven by demographic confounders.

## Cross-cohort adenoma LODO
DECISION: Documented. adenoma_lodo.py runs leave-one-cohort-out
across the 3 adenoma-containing cohorts (FengQ_2015, ZellerG_2014,
ThomasAM_2018a). H-vs-A LODO mean AUC 0.509 (RF) / 0.453 (XGB), per-fold
range 0.379-0.582, vs 0.681/0.709 in within-cohort 5-fold CV; A-vs-CRC
LODO mean AUC 0.583 (RF) / 0.515 (XGB), per-fold range 0.456-0.631, vs
0.787/0.809 in within-cohort 5-fold CV. The H-vs-A LODO performance is
at or below chance, indicating the within-cohort adenoma signal does
not transfer across cohorts at all. A-vs-CRC LODO performs slightly
above chance but well below the within-cohort 5-fold CV. scale_pos_weight
is recomputed per fold from training labels. Saved to
results/adenoma_lodo_results.csv.

## Bootstrap confidence intervals
DECISION: Documented. bootstrap_ci.py computes 10,000-iteration bootstrap
95% CIs on per-cohort and pooled AUCs for species RF, joint RF, and
joint XGB. Per-cohort CIs use i.i.d. resampling within each held-out
cohort; pooled CIs use cohort-stratified resampling (each iteration
resamples with replacement within each cohort separately, then
concatenates) so the LODO sample-size structure is preserved. Species
RF pooled: 0.810 [0.776, 0.841]. Joint RF pooled: 0.776 [0.740, 0.811].
Joint XGB pooled: 0.781 [0.746, 0.816].

## Seed sensitivity
DECISION: Documented. seed_sensitivity.py runs species RF LODO at
seeds {0, 1, 2, 42, 100}. Mean AUC = 0.8049 +/- 0.0020, range
[0.8035, 0.8084]. Results are stable across random seeds.

## Batch correction (ComBat)
DECISION: Documented. batch_correction.py applies per-fold ComBat on
species features. ComBat is fit jointly on the train and test feature
matrices using only batch labels (study_name); class labels (CRC vs
control) are never seen by ComBat, so this preserves the LODO no-
leakage guarantee while keeping train and test in the same corrected
feature space. (Earlier versions trained on corrected features and
tested on uncorrected features, leaving train and test in different
spaces; that version's AUCs were uninterpretable and have been
discarded.) Result: mean per-cohort AUC 0.806 with ComBat vs 0.803
without, indicating batch effects in this curatedMetagenomicData
subset are small relative to the cross-cohort biological signal.
Requires `pip install combat` (canonical PyPI package providing
combat.pycombat.pycombat).

## Package pinning
DECISION: requirements.lock pins exact versions of all Python
dependencies (pandas 2.2.3, numpy 1.26.4, scikit-learn 1.4.2,
xgboost 2.0.3, shap 0.44.1, matplotlib 3.8.5, scipy 1.12.0).
Install with pip install -r requirements.lock.

## Species feature filter and LODO leakage
DECISION: The species prevalence>=10% and mean>=1e-4 filter in
preprocessing.py is computed on all 762 samples including held-out
cohorts. This is a mild form of information leakage analogous to the
pathway filter we now refit per-fold. We retain the global species filter
for three reasons. First, species presence is structurally stable across
cohorts because MetaPhlAn maps to a fixed reference database, so the
filter primarily removes globally rare taxa rather than cohort-specific
signal. Second, the species feature count is small (247 retained), so
dilution is minor relative to the 540 pathway candidates. Third, Thomas
et al. 2019 and Piccinno et al. 2025 both apply species filtering
globally, matching the reference standard for the comparison. The
pathway case differs because unstratified HUMAnN abundance has
heterogeneous cross-cohort distributions and a global filter risks
removing cohort-specific signal that a per-fold filter retains.
