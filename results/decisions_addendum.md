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
