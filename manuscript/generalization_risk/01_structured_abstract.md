# Draft structured abstract

## Background

Gut-metagenomic classifiers can distinguish colorectal cancer (CRC) from
controls within pooled retrospective datasets, but deployment requires signal
that transfers across studies. It is unclear whether biological specificity,
batch correction, or unlabeled target-cohort diagnostics improve that transfer.

## Methods

We analyzed 1,522 stool metagenomes from ten development cohorts, including
1,339 CRC/control samples, using country-aware leave-one-dataset-out
cross-validation. We compared species, unstratified and species-resolved
pathways, leakage-safe fold-selected UniRef90 gene families, and a
checksum-frozen mechanism panel. Source-only species offsets were propagated to
species-resolved pathways; unlabeled target adaptation was reported separately.
We then evaluated whether prediction confidence and species-distribution shift
could estimate target AUC under an outer leave-one-cohort-out risk model. Before
external scoring, we froze 200 public WGS samples from PRJNA763023, balanced by
CRC status and age group.

## Results

The species RF achieved mean per-cohort AUC 0.807 and pooled AUC 0.781 (95% CI
0.757–0.805). Joint pathway models did not improve pooled discrimination (RF
0.756; XGB 0.766). Gene families averaged 0.693 AUC. The frozen mechanism panel
averaged 0.569 AUC and did not improve its parent-species model (0.655 versus
0.656). Source-only correction produced 0.773 AUC for species-resolved
functions versus 0.771 uncorrected. The label-free risk estimator had higher
error than the historical-model-mean comparator (MAE 0.094 versus 0.062).

## Conclusions

Across ten development studies, greater functional or mechanistic specificity
did not yield more portable CRC classification, and intuitive unlabeled shift
signals did not reliably predict target performance. These results favor a
parsimonious taxonomic reference and strict separation of source-only from
target-adaptive correction. Conclusions about prospective portability remain
conditional on the frozen external evaluation.

> Status: internal results complete; external results intentionally omitted
> until all frozen profiles are processed.
