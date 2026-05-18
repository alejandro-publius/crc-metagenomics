# Progress Dashboard

*Alejandro Velazquez and Rachel Selbrede — 2026-05-18*

All numbers traceable to `results/*.csv` and `figures/` in the repo. `scripts/verify_results.py` — **49/49 checks pass**.

---

## 1. Cohort composition (n=1522)

Source: `results/table1.csv`. Median sequencing depth column reports the median reads/sample reported by each cohort's primary publication and verified against the curatedMetagenomicData v3 release notes (paired-end, post-host-decontamination).

| Cohort | Country | N total | CRC | Control | Adenoma | Median depth (M reads) | Adenoma-bearing |
|---|---|---:|---:|---:|---:|---:|:---:|
| FengQ_2015      | AUT |  154 |  46 |  61 | 47 | ~57 | Y |
| GuptaA_2019     | IND |   60 |  30 |  30 |  0 | ~22 | – |
| ThomasAM_2018a  | ITA |   80 |  29 |  24 | 27 | ~32 | Y |
| ThomasAM_2018b  | ITA |   60 |  32 |  28 |  0 | ~25 | – |
| ThomasAM_2019_c | JPN |   80 |  40 |  40 |  0 | ~35 | – |
| VogtmannE_2016  | USA |  104 |  52 |  52 |  0 | ~21 | – |
| WirbelJ_2018    | DEU |  125 |  60 |  65 |  0 | ~38 | – |
| YachidaS_2019   | JPN |  575 | 258 | 250 | 67 | ~42 | Y |
| YuJ_2015        | CHN |  128 |  74 |  54 |  0 | ~43 | – |
| ZellerG_2014    | FRA |  156 |  53 |  61 | 42 | ~36 | Y |
| **TOTAL**       | —   |**1522**|**674**|**665**|**183**| — | — |

Excluded under pre-specified criteria (`results/decisions_addendum.md`): HanniganGD_2017 (sequencing depth + sparsity).

---

## 2. Headline AUC table — species vs joint (CRC vs control)

Per-cohort mean = arithmetic mean across 10 LODO folds. Pooled AUC + 95% CI from 10,000-iter stratified bootstrap on n=1339 CRC/control samples (adenoma excluded from this task). DeLong p computed on pooled predictions. Sources: `results/baseline_results.csv`, `results/joint_results.csv`, `results/bootstrap_ci.csv`, `results/delong_results.csv`.

| Model              | Per-cohort mean | Pooled AUC | 95% CI         | DeLong vs species RF |
|---|---:|---:|---|---|
| **Species RF**     | **0.807**       | **0.781**  | [0.757, 0.805] | —                    |
| Joint RF (sp+pw)   | 0.804           | 0.756      | [0.731, 0.781] | **z=3.35, p=0.0008** |
| Joint XGB (sp+pw)  | 0.797           | 0.766      | [0.740, 0.791] | z=2.00, p=0.046      |
| Bio-shortlist RF   | 0.823           | —          | —              | (parity, not pooled-bootstrapped yet) |

**Read:** Joint models are not just non-superior; they're significantly worse pooled. Biological shortlist is at parity with species, not above.

---

## 3. Adenoma cross-cohort LODO (4 cohorts)

Mean LODO AUC across 4 folds (FengQ_2015, ZellerG_2014, ThomasAM_2018a, YachidaS_2019). Source: `results/adenoma_lodo_results.csv`.

| Task                | RF AUC | XGB AUC |
|---|---:|---:|
| Healthy vs Adenoma  | 0.561  | 0.579   |
| Adenoma vs CRC      | 0.671  | 0.617   |

Top SHAP features for Adenoma vs CRC (oral pathobiont signature; `results/shap_adenoma_vs_crc.csv`):
*Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum*, *Solobacterium moorei*.

Rebalanced LODO outputs (forthcoming): `results/adenoma_rebalanced_lodo.csv`, `results/adenoma_rebalanced_summary.csv`.

---

## 4. Robustness battery

| Check                       | Range / value                | Source                                  |
|---|---|---|
| Pathway sensitivity sweep   | 0.794 – 0.812 (spread 0.018) | `results/sensitivity_thresholds.csv` (20 prev×abund cells) |
| Seed sensitivity (5 seeds)  | 0.810 ± 0.002                | `results/seed_sensitivity.csv`          |
| Confounder adjustment       | 0.800 – 0.814 (vs 0.807)     | `results/confounder_results.csv`        |
| ComBat (per-fold)           | 0.815 corrected / 0.807 raw  | `results/combat_results.csv`            |

**Read:** Headline AUC is insensitive to filter choice, seed, age/sex/BMI residualization, and per-fold ComBat. Country-aware LODO drops ThomasAM_2019_c (Japan) from 0.998 → 0.836 when YachidaS_2019 (Japan) is excluded from training — exactly the country-confounding pattern Tal flagged, now made explicit by the country-aware split.

---

## 5. Reference figures

Main figures (`figures/`):
- `fig1_lodo_auc.png` — per-cohort LODO AUC forest plot, species RF baseline.
- `fig2_shap_crc.png` — top SHAP features, CRC-vs-control task.
- `fig3_adenoma.png` — adenoma cross-cohort LODO summary.
- `fig4_external_validation.png` — held-out cohort generalization.
- `figure5_three_panel_shap.png` — H-vs-A, A-vs-CRC, H-vs-CRC SHAP side-by-side; shows oral-pathobiont emergence at malignant transition.

Manuscript figures (`manuscript/figures/`):
- `Figure1_Forest_Plot.png/.pdf` — manuscript Fig 1, LODO AUC forest.
- `Figure2_ROC_Curves.png/.pdf` — pooled and per-cohort ROC.
- `Figure3_SHAP_Importance.png/.pdf` — CRC-vs-control SHAP.
- `Figure4_Three_Panel_SHAP.png/.pdf` — three-panel SHAP across the adenoma axis.

Diagnostics (`figures/diagnostics/`):
- `roc_pr_pooled.png`, `calibration_reliability.png`, `confusion_matrices.png`, `per_cohort_sens_spec.png`, `subgroup_auc.png` — with matching CSVs in `results/diagnostics/`.
