# Label-free generalization-risk pilot

This analysis asks a deployment-facing question: **before labels arrive, can
we estimate how well a CRC classifier will generalize to a new cohort?**

Each observation is one frozen model evaluated in one held-out cohort. Inputs
to the risk estimate use no target labels: prediction confidence/distribution,
target sample size, species-composition shift, prevalence shift, and the
ability of a classifier to distinguish source from target samples. The outcome
used to evaluate the estimate is the held-out AUC.

Evaluation leaves an entire target cohort out of risk-model development. The
simple comparator is each model's historical mean AUC, also calculated without
the held-out cohort. Results are an internal pilot because ten cohorts are ten
independent deployment environments regardless of the larger model-by-cohort
row count. Prospective claims require a completely untouched external cohort.

Reproduce with:

```bash
python3 scripts/generalization_risk.py
```
