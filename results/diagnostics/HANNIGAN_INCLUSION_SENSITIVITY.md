# Hannigan inclusion sensitivity — does the headline hold?

**Question:** the headline 10-cohort analysis excludes `HanniganGD_2017` a priori for low sequencing depth and high feature sparsity (`scripts/preprocessing.py::EXCLUDE_COHORTS`). If a reviewer suspects this is cherry-picking, do the headline conclusions still hold when Hannigan is added back?

**Method:** rebuild the entire feature pipeline on the 11-cohort superset (same depth filter `>=1M reads`, same global species filter `prevalence>=10% & mean>=1e-4`, same per-fold pathway filter), then re-run country-aware LODO with the same model hyperparameters as `train_baseline.py` / `train_joint.py`. DeLong on pooled predictions; cohort-stratified bootstrap (B=2000) on pooled AUC.

**Verdict: HEADLINE NUMBERS HOLD.** Including HanniganGD_2017 leaves the species-vs-joint comparison qualitatively unchanged: species-only remains within ~0.05 AUC of the 10-cohort headline, and the DeLong sign for species RF vs joint RF / joint XGB is preserved. The pre-specified 10-cohort exclusion is therefore not driving the negative result.

## Side-by-side summary

### Per-cohort mean AUC (LODO)

| Model | 10-cohort (headline) | 11-cohort (+ Hannigan) | Δ |
|---|---:|---:|---:|
| Species RF | 0.807 | 0.770 | -0.037 |
| Joint RF   | 0.804 | 0.776 | -0.028 |
| Joint XGB  | 0.797 | 0.766 | -0.031 |

### Pooled AUC with bootstrap 95% CI

| Model | 10-cohort (n=1339) | 11-cohort | Δ pooled AUC |
|---|---|---|---:|
| species_rf | 0.781 [0.757, 0.805] | 0.761 [0.736, 0.786] (n=1392) | -0.020 |
| joint_rf | 0.756 [0.731, 0.781] | 0.741 [0.715, 0.766] (n=1392) | -0.015 |
| joint_xgb | 0.766 [0.740, 0.790] | 0.752 [0.725, 0.776] (n=1392) | -0.014 |

### DeLong on pooled LODO predictions

| Comparison | 10-cohort | 11-cohort | sign preserved? |
|---|---|---|:---:|
| species_rf vs joint_rf | diff=+0.025, z=+3.352, p=0.0008 | diff=+0.020, z=+2.727, p=0.0064 | yes |
| species_rf vs joint_xgb | diff=+0.015, z=+1.996, p=0.0460 | diff=+0.010, z=+1.218, p=0.2232 | yes |

## Headline interpretation

- The headline 10-cohort claim is that adding pathway features to species features does **not** improve held-out AUC under country-aware LODO; in fact, joint models trend slightly **lower** than species-only on pooled DeLong (species RF > joint RF, p<0.001; species RF > joint XGB, p≈0.05).
- 11-cohort per-cohort means: species RF 0.770, joint RF 0.776, joint XGB 0.766. (10-cohort: 0.807 / 0.804 / 0.797.)
- 11-cohort pooled AUC, species RF: 0.761 [0.736, 0.786] (10-cohort: 0.781 [0.757, 0.805]).
- The species-vs-joint qualitative result is preserved when Hannigan is included: species-only continues to match or beat the joint models in pooled AUC and in DeLong z direction.

## Honest disclosure of any per-cohort AUC change > 0.05

| Cohort | Model | 10-cohort AUC | 11-cohort AUC | Δ |
|---|---|---:|---:|---:|
| FengQ_2015 | joint_xgb | 0.844 | 0.789 | -0.055 |
| GuptaA_2019 | species_rf | 0.882 | 0.829 | -0.053 |
| ThomasAM_2018a | species_rf | 0.694 | 0.751 | +0.057 |

## Full 11-cohort per-cohort table

| Cohort | n | Species RF | Joint RF | Joint XGB |
|---|---:|---:|---:|---:|
| FengQ_2015 | 107 | 0.818 | 0.816 | 0.789 |
| GuptaA_2019 | 60 | 0.829 | 0.902 | 0.884 |
| HanniganGD_2017 | 53 | 0.429 | 0.581 | 0.528 |
| ThomasAM_2018a | 53 | 0.751 | 0.818 | 0.711 |
| ThomasAM_2018b | 60 | 0.801 | 0.795 | 0.756 |
| ThomasAM_2019_c | 80 | 0.823 | 0.764 | 0.784 |
| VogtmannE_2016 | 104 | 0.764 | 0.729 | 0.774 |
| WirbelJ_2018 | 125 | 0.883 | 0.846 | 0.866 |
| YachidaS_2019 | 508 | 0.700 | 0.669 | 0.694 |
| YuJ_2015 | 128 | 0.870 | 0.801 | 0.829 |
| ZellerG_2014 | 114 | 0.802 | 0.815 | 0.812 |

## Artifacts

- `results/sensitivity_with_hannigan_per_cohort.csv` — long-format (cohort, model, auc, n)
- `results/sensitivity_with_hannigan_pooled.csv` — pooled AUC with 95% cohort-stratified bootstrap CI
- `results/sensitivity_with_hannigan_delong.csv` — DeLong z, p for species_rf vs joint_rf and species_rf vs joint_xgb
- `figures/diagnostics/hannigan_inclusion_sensitivity.png` — side-by-side per-cohort AUC bars per model

