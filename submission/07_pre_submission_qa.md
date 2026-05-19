# Pre-Submission Internal QA

**Manuscript:** 10-cohort meta-analysis of gut microbiome classifiers for colorectal cancer
**Authors:** Alejandro Velazquez and Rachel Selbrede
**Audience:** Authors only. Honest, non-defensive critique of the manuscript as it stands today.

This is the list of weaknesses the authors themselves can identify, ranked by how likely they are to be raised by a competent reviewer and how damaging they would be if raised. Each item ends with one of four dispositions: **FIX** (do before submission), **ACK** (acknowledge in Discussion before submission), **DEFER** (mention as future work), or **DEFEND** (argue against in response letter only).

---

## 1. No external (non-curatedMetagenomicData) validation cohort

The headline cross-cohort generalization claim is conditioned on the curatedMetagenomicData uniform pipeline. Every retained cohort was reprocessed with the same MetaPhlAn version, HUMAnN version, and trimming protocol. A reviewer can reasonably argue that this is *within-pipeline* generalization, not true external validation, and that classifier performance would drop on independently processed data.

The diagnostics subgroup analysis (`results/diagnostics/subgroup_auc.csv`) and the ComBat batch-correction robustness check (`manuscript/markdown/04_results.md` "Batch correction") are partial answers but do not substitute for an independently processed external cohort.

**Disposition: ACK + DEFER.** Already acknowledged in Discussion "Limitations" item 3; reinforce the framing that the headline AUC is for *within-curatedMetagenomicData LODO*, and explicitly flag external-pipeline validation as future work in the Conclusion.

---

## 2. Headline pooled AUC of 0.781 is not state-of-the-art

Piccinno et al. (2025) report pooled AUC ~0.85 across 18 cohorts and ~3,700 metagenomes. Our pooled AUC of 0.781 (n=1,339) is a clean methodological story but is not the highest number in the field. A reviewer may interpret this as "you ran fewer cohorts and got lower AUC than the current best paper".

The defense is that our central claim is the *negative finding on pathways* and the *country-aware LODO design choice*, not absolute AUC. The headline number is correctly conservative because country-aware LODO removes the country-pair confound that inflated ThomasAM_2019_c from 0.998 to 0.836 (`results/decisions_addendum.md` "Country-aware LODO").

**Disposition: DEFEND.** Make sure the Abstract and Discussion frame "0.781 pooled" alongside "country-aware LODO removes the population-level confound that would otherwise inflate per-cohort AUCs to implausible levels". Cite Piccinno for context without claiming to beat it.

---

## 3. The DeLong significance is driven primarily by one fold (YachidaS_2019)

The DeLong p=0.0008 for species RF vs joint RF is driven mostly by the largest fold, YachidaS_2019 (n_test=508, ~38% of the pooled n). This is documented honestly in `results/decisions_addendum.md` "DeLong test" and Results. A careful reviewer will notice that "the difference is in the biggest fold" can be read two ways: (a) the largest fold is the most powerful and therefore the most informative, or (b) the result is fragile to a single cohort's idiosyncrasies.

The per-cohort paired tests (n=10 folds) are non-significant for both contrasts (p=0.87, p=0.28; `results/model_comparison.csv`), which is consistent with low n=10 power but is *also* consistent with the joint model being approximately as good as species on 9 of 10 cohorts and worse on one.

**Disposition: ACK in Discussion.** Add a one-sentence acknowledgement in "Why pathways do not help" stating that the DeLong signal is concentrated in YachidaS_2019 and that the per-cohort paired tests are non-significant. The calibration evidence (ECE 0.074 vs 0.152 for joint XGB; `results/diagnostics/calibration_metrics.csv`) provides an independent reason to prefer species RF that does not depend on a single fold.

---

## 4. Per-cohort vs DeLong tests give superficially different answers

A reader who reads only the per-cohort paired t-tests (p=0.87 species RF vs joint RF) and ignores the DeLong analysis will conclude that pathways neither help nor hurt. A reader who reads only the DeLong test (p=0.0008) will conclude that pathways *hurt*. The current Results paragraph reconciles these correctly, but a reviewer who is hostile or rushed may attack the reconciliation.

**Disposition: FIX.** The Results section already addresses this in the paragraph "Together, the per-cohort and DeLong analyses agree...". Verify on the final read-through that the framing is "pathways add no benefit on average; DeLong further detects a small but statistically significant *degradation* at the sample level". Do not over-claim degradation; the substantive claim is non-improvement.

---

## 5. Bonferroni correction was not pre-specified

We report two DeLong comparisons as "significant" (p=0.0008 and p=0.046; `results/delong_results.csv`). Under Bonferroni for two a priori comparisons, the second drops to corrected p=0.092 and is no longer significant at alpha=0.05. The headline finding (species RF beats joint RF, corrected p=0.0016) is unaffected, but the joint XGB comparison becomes marginal.

The current Results text reports the uncorrected p=0.046 without flagging the Bonferroni adjustment.

**Disposition: FIX before submission.** Add one sentence to the Results "Joint species-plus-pathway models do not improve over species alone" paragraph: "Under a Bonferroni correction for the two a priori model contrasts, species RF vs joint RF remains significant (corrected p=0.0016) and species RF vs joint XGBoost becomes marginal (corrected p=0.092)."

---

## 6. No hyperparameter tuning

Hyperparameters are fixed at pre-specified defaults; nested CV was not run. The defense in `submission/06_reviewer_responses.md` M5 is sound: tuning would have to selectively lift the joint model without lifting the species model to flip the conclusion, which is implausible because species features are a strict subspace of joint features.

But a reviewer may simply require a one-shot tuned XGBoost result on the joint set as the price of admission. This is a 1-2 day exercise.

**Disposition: DEFEND now; FIX if requested in revision.** Have a tuned-XGBoost script ready to run on revision. We expect a joint XGB AUC gain of <0.02, leaving the qualitative comparison intact.

---

## 7. Species filter is applied globally, not per fold

The species prevalence (>=10%) and mean (>=1e-4) filter is computed once across all cohorts (after HanniganGD_2017 exclusion), not per fold. This is documented in `results/decisions_addendum.md` "Species feature filter and LODO leakage" with three justifications, and the Methods are explicit about it. A reviewer who reads carefully will identify this as a mild form of feature-set leakage.

The argument that this is a mild and defensible leakage is correct (MetaPhlAn is a fixed reference database; only 229 species are retained; it matches the Thomas et al. 2019 reference standard). But it remains a leakage of *some* information from test folds into the feature set, and a strict reviewer will not be satisfied.

**Disposition: DEFEND.** The Methods are already transparent. If a reviewer demands per-fold species filtering as a sensitivity analysis, this is one script change; expect AUC delta well within the per-fold pathway-filter range (`results/sensitivity_thresholds.csv` shows the joint RF AUC spans 0.055 across a 20-cell filter grid, 0.781-0.835).

---

## 8. Adenoma analysis is underpowered (4 cohorts, n=183)

The adenoma LODO uses 4 cohorts and 183 adenomas (FengQ 47, Yachida 67, Zeller 42, Thomas2018a 27; `results/adenoma_go_nogo_memo.md`). At n_folds=4 the cross-cohort means are noisy and the H-vs-A AUC of 0.561 is barely above chance.

The Discussion "Limitations" item 6 acknowledges this honestly. The biological framing - H-vs-A near chance and A-vs-CRC moderate with the oral pathobiont signature - is robust to the small sample size because it is a *contrast* between two tasks on the same cohorts, not an absolute claim about either AUC.

**Disposition: ACK.** Discussion already acknowledges. Resist any temptation to over-claim the A-vs-CRC AUC of 0.671 as clinically meaningful; it is hypothesis-generating evidence for the stepwise model.

---

## 9. Adenoma case definitions are not harmonized

The four adenoma-containing cohorts do not uniformly distinguish advanced vs non-advanced adenoma, polyp histology, or polyp size. This is acknowledged but is a real limit on what the adenoma analysis can claim.

A particularly aggressive reviewer could argue that the "near chance" H-vs-A AUC is *itself* a reflection of case-definition heterogeneity rather than a biological observation. The defense in `submission/06_reviewer_responses.md` R2 (that the H-vs-A vs A-vs-CRC contrast is internally consistent because both draw from the same cohorts) is the right rebuttal.

**Disposition: ACK + DEFER.** Already in Discussion "Limitations" item 1. Future work item: contact PIs for harmonized polyp histology metadata.

---

## 10. We do not report a clinical sensitivity-at-fixed-specificity table

The pooled Youden-J operating point is reported (sensitivity 0.74, specificity 0.70; `results/diagnostics/per_cohort_operating_chars.csv`), but we do not report a table of sensitivity at, e.g., specificity = {0.80, 0.85, 0.90, 0.95}. Clinical screening reviewers (a likely subset of any JAMA/Gastroenterology/Lancet GI audience) will ask for this.

The information is reconstructible from `results/preds_species_rf.csv` in <1 hour of work.

**Disposition: FIX before submission.** Add a Supplementary Table S11 with sensitivity at fixed specificity {0.80, 0.85, 0.90, 0.95} pooled and per cohort. Reference it from Discussion "Implications for microbiome-based CRC screening".

---

## Summary disposition list

| # | Item | Disposition |
|---|---|---|
| 1 | No external non-curatedMetagenomicData cohort | ACK + DEFER |
| 2 | Pooled AUC 0.781 not state-of-the-art | DEFEND |
| 3 | DeLong driven by one fold | ACK in Discussion |
| 4 | Per-cohort vs DeLong reconciliation | FIX (sentence-level) |
| 5 | Bonferroni not pre-specified | FIX before submission |
| 6 | No hyperparameter tuning | DEFEND; FIX if asked |
| 7 | Global species filter | DEFEND |
| 8 | Adenoma analysis underpowered | ACK |
| 9 | Adenoma case definitions heterogeneous | ACK + DEFER |
| 10 | No sensitivity-at-fixed-specificity table | FIX before submission |

**Pre-submission FIX queue:**
- Add Bonferroni-corrected p-values to Results paragraph (item 5)
- Add sensitivity-at-fixed-specificity supplementary table S11 (item 10)
- Verify final Results paragraph reconciles per-cohort and DeLong cleanly (item 4)
- Add one-sentence flag in Discussion "Why pathways do not help" that DeLong signal is concentrated in YachidaS_2019 (item 3)

Estimated total pre-submission work: 4-6 hours.
