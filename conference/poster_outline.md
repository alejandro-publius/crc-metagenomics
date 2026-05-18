# Poster Outline (A0 portrait, 841 x 1189 mm)

**Title:** Species-level taxonomic features alone outperform joint species-plus-pathway models for cross-cohort colorectal cancer classification

**Authors:** Alejandro Velazquez¹, Rachel Selbrede²
**Affiliations:** ¹[FILL: Alex's affiliation, e.g., University of California, Berkeley]; ²[FILL: Rachel's affiliation]
**Contact:** [FILL: corresponding email]

---

## Layout overview

```
+---------------------------------------------------------------+
|  TITLE BAR (full width, ~12% height)                          |
|  Title | Authors | Affiliations | Institution logos           |
+---------------+-------------------------------+---------------+
|  LEFT COL     |  CENTER COL                   |  RIGHT COL    |
|  (~25% width) |  (~45% width)                 |  (~30% width) |
|               |                               |               |
|  P1 INTRO     |  P3 HEADLINE: Species-only    |  P6 DISCUSSION|
|               |     LODO (Fig 1)              |               |
|  P2 METHODS   |  P4 NEGATIVE: Pathways do not |  P7 LIMITS    |
|               |     help (Fig 2 + DeLong tbl) |               |
|               |  P5 ADENOMA + SHAP            |  P8 TAKE-HOMES|
|               |     (Fig 3 + Fig 4)           |               |
+---------------+-------------------------------+---------------+
|  BOTTOM STRIP (full width, ~8% height)                        |
|  References (compact) | QR code | Acknowledgments | Funding   |
+---------------------------------------------------------------+
```

Color/typography: dark navy headers on white background; Open Sans 32 pt body, 60 pt panel titles, 110 pt poster title. Panel titles numbered to guide reading order (1 -> 8).

---

## LEFT COLUMN

### Panel 1 — Introduction / Motivation
**Header:** Why re-evaluate microbiome CRC classifiers?

**Poster text (use verbatim):**
> Colorectal cancer (CRC) is the third most-diagnosed cancer worldwide. Shotgun gut metagenomics can distinguish CRC cases from controls, and prior multi-cohort meta-analyses report cross-cohort AUC near 0.80. Two open questions remain: (i) do HUMAnN functional pathway features add discriminative signal beyond MetaPhlAn species profiles, and (ii) how robust are these classifiers to analytical choices such as cohort selection, batch correction, demographic confounders, and country-level population structure? We re-evaluated the Thomas et al. (2019) framework on an expanded 10-cohort dataset (n = 1,522) with country-aware leave-one-dataset-out (LODO) cross-validation and formal DeLong comparison.

**Visual:** small schematic icon of the gut microbiome -> classifier -> AUC curve (designer-generated, no source figure).

---

### Panel 2 — Methods
**Header:** Data and design

**Poster text:**
> Ten curatedMetagenomicData cohorts spanning eight countries: FengQ_2015 (AUT), GuptaA_2019 (IND), ThomasAM_2018a/b (ITA), ThomasAM_2019_c (JPN), VogtmannE_2016 (USA), WirbelJ_2018 (DEU), YachidaS_2019 (JPN), YuJ_2015 (CHN), ZellerG_2014 (FRA). HanniganGD_2017 excluded a priori for low sequencing depth. Features: 229 MetaPhlAn species (10% prevalence, mean >= 1e-4, log10 transform) and 402-406 HUMAnN unstratified pathways re-filtered per fold to prevent leakage. Three classifiers: species-only Random Forest, joint species+pathway RF, joint species+pathway XGBoost. Country-aware LODO: when a cohort is the test fold, all cohorts from the same country are also excluded from training. Significance: DeLong test on pooled held-out predictions, plus per-cohort paired tests; 10,000-resample cohort-stratified bootstrap CIs.

**Table to render (compact, from `results/table1.csv`):** Cohort | Country | N | CRC | Adenoma | Control
- One row per cohort + TOTAL row (1,522 / 674 / 183 / 665).
- Annotate Italian and Japanese cohort pairs with a tag (e.g., "ITA pair", "JPN pair") to motivate country-aware LODO.

**Visual sidebar (designer-mock, no source file):** small map of cohort countries with sample-size circles.

---

## CENTER COLUMN

### Panel 3 — Headline result: species-only LODO
**Header:** Species-only RF: pooled LODO AUC 0.781 (95% CI 0.757-0.805)

**Poster text:**
> A 500-tree Random Forest trained on 229 species features achieved a per-cohort mean LODO AUC of 0.807 +/- 0.065 across 10 folds (n = 1,339 case/control samples). The pooled AUC on held-out predictions was 0.781 (95% CI 0.757-0.805; 10,000-resample cohort-stratified bootstrap). Per-cohort AUCs ranged from 0.694 (ThomasAM_2018a, n_test = 53) to 0.882 (GuptaA_2019 and WirbelJ_2018). Country-aware exclusion materially mattered: ThomasAM_2019_c reaches AUC 0.998 when YachidaS_2019 (same-country) is in the training set but drops to 0.836 when YachidaS_2019 is excluded, confirming that naive LODO inflates performance through population-level confounding.

**Figure:** `figures/fig1_lodo_auc.png` (per-cohort LODO AUC forest plot for species RF) — render at ~40% panel height.

**Inline table (small, bottom of panel):** Top vs bottom per-cohort folds (source: `results/baseline_results.csv`)
| Fold              | AUC   | n_test | Note                       |
|-------------------|-------|--------|----------------------------|
| GuptaA_2019       | 0.882 | 60     | no country exclusion       |
| WirbelJ_2018      | 0.882 | 125    | no country exclusion       |
| ThomasAM_2019_c   | 0.836 | 80     | JPN; YachidaS excluded     |
| YachidaS_2019     | 0.708 | 508    | JPN; ThomasAM_2019_c excl. |
| ThomasAM_2018a    | 0.694 | 53     | ITA; ThomasAM_2018b excl.  |

Callout box: "Without country-aware LODO, ThomasAM_2019_c AUC = 0.998. With country-aware LODO, AUC = 0.836. Population structure inflates naive LODO."

---

### Panel 4 — Negative result: pathways do not help
**Header:** Adding HUMAnN pathways does NOT improve performance

**Poster text:**
> Joining 402-406 per-fold pathway features to the 229 species features did not improve pooled LODO classification. The joint RF achieved pooled AUC 0.756 (95% CI 0.731-0.781), and the joint XGBoost achieved 0.766 (0.740-0.791) — both lower than the species-only baseline of 0.781. DeLong testing on the 1,339-sample pooled predictions confirmed that species-only significantly outperforms joint RF (z = 3.35, p = 0.0008) and joint XGBoost (z = 2.00, p = 0.046). Per-cohort paired tests (n = 10 folds) did not reach significance (t = 0.87 and 0.28), consistent with limited fold-level power. The effect is driven by the largest fold (YachidaS_2019, n = 508), where species RF reaches 0.708 versus 0.669 (joint RF) and 0.694 (joint XGBoost). Result is stable across 5 random seeds (per-cohort AUC 0.810 +/- 0.002) and a 20-cell pathway-filter sensitivity grid (per-cohort AUC 0.794-0.812).

**Figure:** `figures/diagnostics/roc_pr_pooled.png` (pooled ROC and PR curves for the three classifiers).

**Inline DeLong table (source: `results/delong_results.csv`):**
| Comparison                 | AUC A | AUC B | z      | p       |
|----------------------------|-------|-------|--------|---------|
| species_rf vs joint_rf     | 0.781 | 0.756 | 3.352  | 0.0008  |
| species_rf vs joint_xgb    | 0.781 | 0.766 | 1.996  | 0.046   |
| joint_xgb vs joint_rf      | 0.766 | 0.756 | 1.304  | 0.19    |

Callout: "Parsimony wins at n ~ 1,300. Pathways nearly triple feature dimensionality without proportional signal."

---

### Panel 5 — Adenoma progression + SHAP biology
**Header:** Oral pathobionts emerge near malignant transformation

**Poster text:**
> Cross-cohort adenoma LODO across the four adenoma-containing cohorts (FengQ_2015, ZellerG_2014, ThomasAM_2018a, YachidaS_2019; n_adenoma = 183) yielded near-chance discrimination for healthy-vs-adenoma (RF 0.561, XGB 0.579) and moderate discrimination for adenoma-vs-CRC (RF 0.671, XGB 0.617). TreeSHAP analysis of the CRC-vs-control RF identified four oral pathobionts as the top discriminative features: *Gemella morbillorum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, and *Fusobacterium nucleatum*. The same oral-pathobiont signature dominates the adenoma-vs-CRC SHAP ranking but is weak in healthy-vs-adenoma, supporting a stepwise model in which the CRC-defining oral consortium colonises at or near the transition to invasive carcinoma rather than at the adenoma stage.

**Figures:**
- `figures/fig3_adenoma.png` (left half of panel): stage-stratified AUC bar chart for the four LODO tasks.
- `figures/figure5_three_panel_shap.png` (right half of panel): three-panel SHAP across H-vs-A | CRC-vs-control | A-vs-CRC.

**Inline table (source: `results/shap_crc_features.csv` top rows):**
| Rank | Species                    | Mean |SHAP| |
|------|----------------------------|--------------|
| 1    | Gemella morbillorum        | 0.032        |
| 2    | Parvimonas micra           | 0.029        |
| 3    | Peptostreptococcus stomatis| 0.023        |
| 4    | Fusobacterium nucleatum    | 0.018        |
| 5    | Solobacterium moorei       | 0.015        |

---

## RIGHT COLUMN

### Panel 6 — Discussion
**Header:** Why pathways don't help, and what the signature means

**Poster text:**
> Two explanations for the negative pathway result. (1) Dimensionality: joining ~400 pathways to 229 species nearly triples feature count, and under RF `max_features='sqrt'` the probability of sampling the most informative species at each split decreases as the candidate pool grows. (2) Redundancy: at the unstratified pathway level, the oral pathobionts that top the species SHAP ranking already contribute to most of the pathway signal, so functional features are largely collinear with species. The biologically-guided 84-pathway shortlist (mean LODO AUC 0.817) matches but does not exceed the species baseline (0.807), confirming the redundancy hypothesis. The oral-pathobiont signature is reproducible across RF and XGBoost despite different splitting criteria, across cohorts on three continents, and across the adenoma-carcinoma transition — the most robust observation in this analysis. *F. nucleatum* has known mechanistic ties to CRC via FadA / E-cadherin / beta-catenin signalling.

---

### Panel 7 — Limitations
**Header:** What this poster does *not* claim

**Poster text:**
> Six limitations. (i) Metadata harmonisation across curatedMetagenomicData is limited to age, sex, BMI; we cannot model adenoma stage, tumour location, TNM stage, or treatment history. (ii) Cross-sectional only — no longitudinal samples. (iii) All cohorts processed through the curatedMetagenomicData uniform pipeline; performance on independently processed cohorts is untested. (iv) No nested-CV hyperparameter tuning (justified because the joint model fails to beat the species baseline at defaults). (v) Pathway features are unstratified; taxon-stratified pathways were excluded due to redundancy. (vi) Adenoma analyses are underpowered (n = 183, 4 cohorts) and should be read as hypothesis-generating. The pathway-negative result is a statement about the n ~ 1,300 regime; with thousands of samples (cf. Piccinno et al. 2025, ~3,700 metagenomes, AUC ~0.85), the cost-benefit calculus may shift.

---

### Panel 8 — Take-homes
**Header:** Take-homes

**Poster text (bulleted on poster):**
> 1. Species-only Random Forest is the right default for cross-cohort CRC classification at current sample sizes (pooled LODO AUC 0.781 [0.757, 0.805], n = 1,339).
> 2. Adding HUMAnN unstratified pathway features significantly degrades pooled performance (DeLong z = 3.35, p = 0.0008 against species-only).
> 3. Country-aware LODO is essential: naive LODO inflates ThomasAM_2019_c from 0.836 to 0.998.
> 4. Four oral pathobionts (*F. nucleatum*, *P. stomatis*, *P. micra*, *G. morbillorum*) drive the CRC signal and emerge at or near malignant transformation rather than at the adenoma stage.
> 5. All code, per-sample predictions, decision logs, and figures are public (QR code, bottom).

---

## BOTTOM STRIP

### References (compact, 2-column, 18-20 pt)
1. Thomas AM et al. *Nat Med* 2019; 25:667-678.
2. Wirbel J et al. *Nat Med* 2019; 25:679-689.
3. Piccinno G et al. *[Journal]* 2025; [vol]:[pp]. [FILL: verify citation]
4. Pasolli E et al. (curatedMetagenomicData). *Nat Methods* 2017; 14:1023-1024.
5. Beghini F et al. (MetaPhlAn 3 / HUMAnN 3). *eLife* 2021; 10:e65088.
6. Sun X, Xu W. (DeLong-Sun-Xu test). *IEEE SPL* 2014; 21:1389-1393.
7. Castellarin M et al. (*F. nucleatum* / FadA / CRC). *Genome Res* 2012; 22:299-306.

### QR code
Encodes the URL in `conference/qr_code_target_url.txt` -> https://github.com/alejandro-publius/crc-metagenomics
Caption: "Code, data, predictions, and decision logs."

### Acknowledgments
> We thank the curatedMetagenomicData team and the authors of all 10 contributing cohorts for making their data available. Computation supported by [FILL: HPC / lab resource credit]. AV and RS designed the analyses, AV implemented the pipeline, RS contributed biological interpretation; both authors wrote the manuscript. No external funding to declare. [FILL: confirm funding statement.]

### Figure file paths (for the designer)
- Fig 1 (LODO forest): `figures/fig1_lodo_auc.png`
- Fig 2 (SHAP top species, optional alternate for Panel 5): `figures/fig2_shap_crc.png`
- Fig 3 (adenoma): `figures/fig3_adenoma.png`
- Fig 4 (external validation, reserve): `figures/fig4_external_validation.png`
- Fig 5 (three-panel SHAP across stages): `figures/figure5_three_panel_shap.png`
- Pooled ROC/PR: `figures/diagnostics/roc_pr_pooled.png`

### Source tables (for the designer)
- `results/table1.csv` (cohort demographics)
- `results/baseline_results.csv` (per-cohort species RF AUCs)
- `results/delong_results.csv` (DeLong comparisons)
- `results/bootstrap_ci.csv` (95% CIs)
- `results/shap_crc_features.csv` (CRC SHAP ranking)
- `results/adenoma_lodo_results.csv` (adenoma stage AUCs)
