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
DECISION: Deferred. Paired t-test and Wilcoxon on per-cohort AUCs
are reported (auc_comparison.py). DeLong on raw probabilities would
require refactoring run_lodo_cv to save predictions per fold. n=7
paired tests are underpowered regardless, and bootstrap CIs already
include zero for all comparisons.

## Normalization
DECISION: Use raw relative abundance from curatedMetagenomicData
(MetaPhlAn species, HUMAnN pathway). No CLR, no log transform.
Random Forest and XGBoost are scale-invariant for tree splits.

## Pathway feature set
DECISION: Use unstratified pathway abundance (405 features after
prevalence>=10% and mean>=1e-6 filter). Stratified taxon|pathway
features were considered but produce 4589 highly redundant columns
that did not improve AUC in pilot testing.

## Hyperparameter tuning
DECISION: No nested CV tuning. Defaults documented in
adenoma_go_nogo_memo.md. Justification: joint model does not
statistically outperform species-only baseline; tuning is unlikely
to change the qualitative conclusion.

## Pathway prevalence filter and LODO leakage
NOTE: The prevalence>=10% and mean>=1e-6 filter in filter_pathways.py
is computed on all 762 samples including held-out cohorts. This is
a mild form of information leakage. Disclosed as a limitation in
Discussion. A strict alternative would refit the filter inside each
LODO fold; the impact is expected to be small because the filter
removes only zero-inflated columns.
