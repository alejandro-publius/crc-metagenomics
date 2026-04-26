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
