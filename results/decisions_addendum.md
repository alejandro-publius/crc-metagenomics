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
{0.05..0.25} x mean {1e-7..1e-4} under LODO CV. AUC ranges from
0.773 to 0.789 across all 20 combinations (spread = 0.016),
confirming that the default thresholds (prevalence >= 10%,
mean >= 1e-6) are near-optimal and conclusions are not sensitive
to the specific cutoffs chosen.

## Confounder adjustment
DECISION: Documented. confounder_adjustment.py tests age, sex, and
BMI as potential confounders via direct inclusion and residualization.
Covariate imputation uses train-fold-only medians/modes to avoid
leakage. Results: baseline 0.803, direct RF 0.808, direct XGB 0.812,
residualized RF 0.807, residualized XGB 0.799. Minimal change
confirms the classifier is not driven by demographic confounders.

## Cross-cohort adenoma LODO
DECISION: Documented. adenoma_lodo.py runs leave-one-cohort-out
across the 3 adenoma-containing cohorts (FengQ_2015, ZellerG_2014,
ThomasAM_2018a). H-vs-A LODO AUC 0.58-0.62 (vs 0.68-0.71 in 5-fold
CV); A-vs-CRC LODO AUC 0.67-0.69 (vs 0.79-0.81). The substantial
drop indicates adenoma classification does not generalize well across
cohorts with current sample sizes. scale_pos_weight is recomputed per
fold from training labels.

## Bootstrap confidence intervals
DECISION: Documented. bootstrap_ci.py computes 2000-iteration
bootstrap 95% CIs on per-cohort and pooled AUCs for species RF,
joint RF, and joint XGB. Species RF pooled: 0.810 [0.776, 0.840].

## Seed sensitivity
DECISION: Documented. seed_sensitivity.py runs species RF LODO at
seeds {0, 1, 2, 42, 100}. Mean AUC = 0.8049 +/- 0.0020, range
[0.8035, 0.8084]. Results are stable across random seeds.

## Batch correction (ComBat)
DECISION: Documented. batch_correction.py applies per-fold ComBat
on species features (training cohorts only; test fold is single-cohort
so no correction needed). Requires pycombat. Results should be
compared against uncorrected baseline to assess batch effect magnitude.

## Package pinning
DECISION: requirements.lock pins exact versions of all Python
dependencies (pandas 2.2.3, numpy 1.26.4, scikit-learn 1.4.2,
xgboost 2.0.3, shap 0.44.1, matplotlib 3.8.5, scipy 1.12.0).
Install with pip install -r requirements.lock.
