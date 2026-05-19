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

HanniganGD_2017 was excluded a priori for low sequencing depth (median 8.7M reads vs per-cohort medians of 39.4M for ThomasAM_2018b to 83.4M for ThomasAM_2018a in the retained 10-cohort set, all others above 40M; `results/supplementary/S1_cohort_overview.csv`) and 82% species feature sparsity; the exclusion was specified before any classifier training (`scripts/preprocessing.py`, `EXCLUDE_COHORTS`).

---

## Supplementary Table S1. Pathway filter-threshold sensitivity (joint Random Forest, country-aware LODO)

Each row reports one country-aware LODO run (10 folds, 1,339 case/control samples) over species ∪ unstratified pathway features, with the per-fold prevalence/mean filter computed on training-cohort samples only at the indicated thresholds. The headline run uses prevalence ≥ 0.10, mean ≥ 1e-6. Across the full 20-cell grid, mean per-cohort AUC ranges from 0.781 to 0.835 (spread 0.055). The mean ≥ 1e-3 column retains only two pathways and effectively reduces the joint model to species-only. Source: `results/sensitivity_thresholds.csv`.

| Prevalence | Mean abundance | Pathways retained (mean) | Total features (mean) | Mean per-cohort AUC | SD across folds |
|---|---|---|---|---|---|
| ≥ 0.05 | ≥ 1e-7 | 438.6 | 667.6 | 0.796 | 0.089 |
| ≥ 0.05 | ≥ 1e-6 | 405.4 | 634.4 | 0.791 | 0.091 |
| ≥ 0.05 | ≥ 1e-5 | 318.4 | 547.4 | 0.784 | 0.090 |
| ≥ 0.05 | ≥ 1e-4 | 151.8 | 380.8 | 0.828 | 0.072 |
| ≥ 0.05 | ≥ 1e-3 | 2.0 | 231.0 | 0.835 | 0.054 |
| ≥ 0.10 | ≥ 1e-7 | 420.2 | 649.2 | 0.789 | 0.082 |
| ≥ 0.10 | ≥ 1e-6 | 403.0 | 632.0 | 0.788 | 0.089 |
| ≥ 0.10 | ≥ 1e-5 | 318.4 | 547.4 | 0.784 | 0.090 |
| ≥ 0.10 | ≥ 1e-4 | 151.8 | 380.8 | 0.828 | 0.072 |
| ≥ 0.10 | ≥ 1e-3 | 2.0 | 231.0 | 0.835 | 0.054 |
| ≥ 0.15 | ≥ 1e-7 | 402.0 | 631.0 | 0.793 | 0.092 |
| ≥ 0.15 | ≥ 1e-6 | 392.4 | 621.4 | 0.781 | 0.093 |
| ≥ 0.15 | ≥ 1e-5 | 318.4 | 547.4 | 0.784 | 0.090 |
| ≥ 0.15 | ≥ 1e-4 | 151.8 | 380.8 | 0.828 | 0.072 |
| ≥ 0.15 | ≥ 1e-3 | 2.0 | 231.0 | 0.835 | 0.054 |
| ≥ 0.20 | ≥ 1e-7 | 384.2 | 613.2 | 0.791 | 0.091 |
| ≥ 0.20 | ≥ 1e-6 | 379.0 | 608.0 | 0.784 | 0.091 |
| ≥ 0.20 | ≥ 1e-5 | 318.4 | 547.4 | 0.784 | 0.090 |
| ≥ 0.20 | ≥ 1e-4 | 151.8 | 380.8 | 0.828 | 0.072 |
| ≥ 0.20 | ≥ 1e-3 | 2.0 | 231.0 | 0.835 | 0.054 |

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

Per-fold ComBat correction (`combat.pycombat.pycombat`) on species features yields a per-cohort mean AUC of 0.815 versus 0.807 without correction (Δ +0.008). ComBat is fit on the union of train and test feature matrices using only batch labels (`study_name`); class labels are never seen, but test-fold feature distribution informs the correction — so this is reported only as a robustness-check upper bound, not the headline. Source: `results/combat_results.csv`.

## Supplementary Note S4. Biologically-guided pathway shortlist

A pre-specified, keyword-based shortlist drawn from nine biological groups (butyrate/SCFA, fermentation, LPS/inflammation, polyamines, tryptophan metabolism, folate/one-carbon metabolism, sulfur/methionine metabolism, glycan/mucin degradation, bile-acid metabolism) expands to 86 unique CRC-relevant pathway candidates; per-fold prevalence/mean filtering retains ~66 of these, giving ~295 total features per fold (229 species + ~66 pathways). The joint species + shortlist-pathway feature set yields per-cohort mean AUC 0.817, comparable to the species-only baseline. Curated CRC-relevant pathways therefore do not provide an advantage over the species features alone. Source: `results/bio_pathway_results.csv`.

## Supplementary Note S5. Adenoma classification (cross-cohort LODO)

Country-aware LODO across the four adenoma-containing cohorts (FengQ_2015 n_adenoma=47; YachidaS_2019 n_adenoma=67; ZellerG_2014 n_adenoma=42; ThomasAM_2018a n_adenoma=27; total n=183): control-vs-adenoma mean LODO AUC 0.561 (RF) / 0.579 (XGBoost); adenoma-vs-CRC mean LODO AUC 0.671 (RF) / 0.617 (XGBoost). Source: `results/adenoma_lodo_results.csv`.
