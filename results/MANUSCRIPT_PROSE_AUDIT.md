# Manuscript Prose Audit

**Date:** 2026-05-18
**Files audited (read-only):** `manuscript/markdown/00_title.md` through `07_supplementary.md` and `manuscript_complete.md`.
**Scope:** overclaims, hedge mismatches, unsupported speculation, comparative claims, sample-size overconfidence, methodological overreach, causal language, internal contradictions, flow, reviewer-attack vectors.
**Output policy:** read-only audit — no manuscript files modified.

Severity legend:
- **P0** = would mislead a reviewer or invalidate a claim if uncorrected
- **P1** = would weaken the paper / open an easy reviewer attack
- **P2** = polish / consistency

---

## P0 — Would mislead a reviewer

### P0.1  Internal contradiction: ComBat fit *jointly on train + test* contradicts every "no leakage" claim
- **Where:** `03_methods.md:23`, `03_methods.md:53`, `05_discussion.md:30`
- **Quote (methods):** "Per-fold ComBat (Johnson et al. 2007) on species features, fit jointly on train and test feature matrices using only `study_name` as the batch label"
- **Quote (methods, robustness):** "ComBat was fit jointly on train and test feature matrices using only batch labels (`study_name`); class labels were never seen by ComBat, preserving LODO no-leakage"
- **Problem:** Fitting ComBat on the union of train and test *is* feature-distribution leakage, even when class labels are hidden. The empirical-Bayes priors and the per-batch mean/variance estimates for the test cohort are computed from test-cohort feature values, which then alters the training-fold transform. The discussion presents this run as "robustness check" with a +0.008 AUC bump — a reviewer will immediately argue that the +0.008 is exactly the leakage signature, and that the claim "biological signal survives the dominant technical batch effects" is therefore inverted: the result is consistent with either survival *or* with a small batch-induced inflation.
- **Why this is P0:** the paper repeatedly markets per-fold filtering with training-only statistics as a methodological improvement; the ComBat sub-experiment violates that principle in exactly the way the manuscript criticises in others.
- **Proposed fix:** Either (a) re-run ComBat fit on training cohorts only with `transform()` applied to the held-out cohort, OR (b) keep the current run but explicitly label the +0.008 as an upper bound that includes a small leakage contribution, AND state plainly in Methods that the joint-fit choice does not preserve full LODO no-leakage in the feature-distribution sense. Do not claim "preserving LODO no-leakage" when the feature transform sees the test cohort.

### P0.2  Abstract / Results / Conclusion overclaim relative to pre-cohort tests
- **Where:** `01_abstract.md:7`, `01_abstract.md:9`, `04_results.md:21`, `05_discussion.md:3`, `05_discussion.md:48`
- **Quote (abstract conclusion):** "species-level taxonomic features alone provide superior CRC classification compared to joint species-plus-pathway models"
- **Quote (results):** "DeLong further detects a small but statistically significant *degradation* at the sample level"
- **Quote (discussion conclusion):** "A species-only Random Forest classifier ... significantly outperforms joint species-plus-pathway Random Forest and XGBoost models"
- **Problem:** The per-cohort paired tests are non-significant for both contrasts (t-test p = 0.87 and 0.28; Wilcoxon p = 0.38 and 0.23). The DeLong significance is driven primarily by one cohort (YachidaS_2019, n_test=508; explicitly acknowledged at `04_results.md:19`). The abstract and the discussion conclusion present a categorical "superior / significantly outperforms" framing on the basis of a single test that is dominated by one fold.
- **Why this is P0:** a reviewer who recomputes Δ AUC excluding YachidaS_2019 will find the DeLong contrast collapses, and will accuse the authors of overstating a YachidaS-specific effect as a global one.
- **Proposed fix:** Hedge in the abstract conclusion: "At current cross-cohort sample sizes, species-level features alone match or modestly exceed joint species-plus-pathway models; the joint models do not provide a detectable per-cohort improvement, and DeLong on pooled predictions shows a small advantage for species-only that is dominated by the largest cohort (YachidaS_2019, n=508)." Add a one-sentence leave-one-cohort-out DeLong sensitivity in results (or at minimum acknowledge that pooled DeLong significance is fragile to the YachidaS exclusion).

### P0.3  Per-cohort SHAP rank vs depth: only F. nucleatum reported; two species in the panel are uncorrected-significant
- **Where:** `05_discussion.md:9`
- **Quote:** "Per-cohort SHAP rank does not correlate with cohort median sequencing depth for *F. nucleatum* (Spearman ρ = −0.19, p = 0.59, n = 10 cohorts; ...); no top-20 species survives a multiple-testing-corrected threshold for depth-rank correlation."
- **Problem:** Looking at `results/diagnostics/depth_confound_shap.csv`, *Eubacterium eligens* (ρ = 0.65, p = 0.042) and *Parabacteroides distasonis* (ρ = −0.72, p = 0.019) are uncorrected-significant. The "no top-20 species survives a multiple-testing-corrected threshold" sentence is technically true at Bonferroni α = 0.05/20 = 0.0025, but the manuscript sidesteps that two top-20 species (one positive, one negative correlation) are nominally significant. A reviewer scanning the supplied CSV will land on this.
- **Why this is P0:** the claim "the oral-pathobiont signature is therefore not an artifact of ... cohort-level read depth" is supported for the four oral pathobionts, but the manuscript phrasing implies a clean negative for the whole panel that the data does not deliver.
- **Proposed fix:** Replace the sentence with: "Of the 20 species tested, two showed nominally significant rank–depth correlations (*Eubacterium eligens* ρ = +0.65, p = 0.042; *Parabacteroides distasonis* ρ = −0.72, p = 0.019), neither of which survives Bonferroni correction at α = 0.0025; the four oral pathobionts all show |ρ| < 0.3 with p > 0.45."

### P0.4  Internal contradiction in joint-RF / joint-XGB "top 4" feature list
- **Where:** `04_results.md:37` vs `04_results.md:54` (Figure 3 legend)
- **Quote (results body):** "TreeSHAP analysis of the joint RF identified *Gemella morbillorum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, and *Fusobacterium nucleatum* as the four highest-ranked features ... The top four features were identical in joint XGBoost up to rank order (*Gemella morbillorum*, *Parvimonas micra*, *Peptostreptococcus stomatis*, *Streptococcus salivarius*, *Fusobacterium nucleatum*)."
- **Quote (Figure 3 legend):** "The four highest-ranked species (*Parvimonas micra*, *Peptostreptococcus stomatis*, *Gemella morbillorum*, *Fusobacterium nucleatum*) are concordant across both models"
- **Problem 1:** Joint XGBoost top 4 (from `results/shap_crc_xgb.csv`) is *Gemella morbillorum, Parvimonas micra, Peptostreptococcus stomatis, Streptococcus salivarius* — *F. nucleatum* is rank 5, not in top 4. The body text lists FIVE species after claiming "identical ... up to rank order" with FOUR. That is internally inconsistent (5 ≠ 4) and inconsistent with the figure legend, which then drops *S. salivarius* and reasserts the joint-RF four are concordant in both models.
- **Problem 2:** The figure legend says the four highest-ranked species are concordant across both models — but XGBoost's true top 4 includes *S. salivarius* and excludes *F. nucleatum*, so "concordance" is 3/4, not 4/4.
- **Why this is P0:** numerical/feature claims that contradict the supplied SHAP CSV and contradict each other across results body and figure legend are a credibility hit; a reviewer who counts the species will catch it.
- **Proposed fix:** Rewrite results paragraph as: "The top four joint-RF features by mean |SHAP| were *G. morbillorum*, *P. micra*, *P. stomatis*, *F. nucleatum*. Three of these (*G. morbillorum*, *P. micra*, *P. stomatis*) also occupy ranks 1–3 of joint XGBoost; XGBoost's rank-4 feature is *S. salivarius*, with *F. nucleatum* at rank 5." Update Figure 3 legend accordingly. Either say "top 4 shared 3/4" or "top 5 shared 4/5", but pick one and be honest about which.

### P0.5  Numeric contradiction: 0.999 vs 0.998 for the same inflation example
- **Where:** `03_methods.md:19` says **0.999**; `05_discussion.md:30` says **0.998**
- **Quote (methods):** "Without this fix, ThomasAM_2019_c reached an inflated AUC of 0.999 due to YachidaS_2019 in training"
- **Quote (discussion):** "allowing YachidaS_2019 (the second Japanese cohort) into training inflated test AUC to 0.998"
- **Ground truth (`results/diagnostics/cv_methodology_comparison.csv`, row `standard_lodo,ThomasAM_2019_c`):** 0.9981
- **Why this is P0 (borderline P1):** the same load-bearing example carries two different numbers in two adjacent sections; reviewers and copy-editors will flag it.
- **Proposed fix:** Pick one. 0.998 is the correct three-significant-figures rounding. Use 0.998 everywhere.

### P0.6  Overclaim on country-aware LODO as a generalisable methodological recommendation
- **Where:** `05_discussion.md:30`
- **Quote:** "methodological choices accounting for cohort-level distributional heterogeneity should become standard for cross-cohort microbiome benchmarks"
- **Problem:** The +0.050 AUC inflation under pooled 5-fold is from one comparison in one study (this one). Generalising to "should become standard for cross-cohort microbiome benchmarks" is a normative methodological recommendation that exceeds the evidence (n = 1 dataset).
- **Why this is P0 (borderline P1):** reviewer-2 will object to a one-study basis for a field-wide methodological mandate.
- **Proposed fix:** Soften to "is consistent with cohort-level distributional heterogeneity being a meaningful confound in this 10-cohort set, and supports country-aware splitting as a useful default when same-country cohorts are present in pooled CRC microbiome benchmarks."

---

## P1 — Would weaken the paper / easy reviewer attack vector

### P1.1  Unsupported assertions in oral-pathobiont biology paragraph (no citations)
- **Where:** `05_discussion.md:18`
- **Quote:** "*F. nucleatum* in particular has been mechanistically linked to CRC through FadA-mediated adhesion to E-cadherin, β-catenin signalling, and tumour-permissive immune modulation; the other taxa form part of a co-occurring oral consortium repeatedly observed in CRC tumours and stool. The same oral-bacterial taxa have been repeatedly observed in upper-gastrointestinal cancers and Barrett's esophagus, consistent with a broader pattern in which oral pathobionts colonise transformed gastrointestinal epithelium across anatomical sites."
- **Problem:** Three substantive biological claims, zero in-line citations. The reference list does not contain any FadA / β-catenin / Barrett's esophagus references. Reviewer-2 will immediately ask for primary sources.
- **Proposed fix:** Either add primary references (e.g., Rubinstein 2013 for FadA/E-cadherin; Kostic 2013 for F. nucleatum tumour modulation; Snider 2018-ish for Barrett's oral-pathobiont overlap) or remove the mechanistic and Barrett's sentences entirely and keep only what is supported by the analyses presented here.

### P1.2  Unsupported assertion about pathway redundancy with taxa
- **Where:** `05_discussion.md:7`
- **Quote:** "Pathway features are also highly correlated with the taxa that encode the corresponding genes — the four oral pathobionts ... collectively contribute to a wide swathe of unstratified pathways — so much of the apparent 'functional' signal is already captured by the taxonomic features."
- **Problem:** No correlation statistics are reported, no stratified-pathway cross-tabulation is shown, and the dataset deliberately excludes taxon-stratified columns. The claim is plausible but is presented as fact without numerical support.
- **Proposed fix:** Either (a) add a Spearman correlation between top-pathway abundances and top-species abundances to the supplementary, OR (b) downgrade to "is consistent with substantial taxonomic-functional redundancy at the unstratified pathway level, although we did not quantify pairwise taxon–pathway correlations directly."

### P1.3  Causal language on observational data — "drives", "drives the moderate adenoma-vs-CRC performance"
- **Where:** `05_discussion.md:18`
- **Quote:** "It is also the feature signature that drives the moderate adenoma-vs-CRC performance"
- **Problem:** "Drives" implies causal mechanism; the SHAP analysis shows attribution, not causation. Should be "dominates", "accounts for", or "is the top-ranked signature for".
- **Proposed fix:** "It is also the feature signature that is most heavily weighted by the adenoma-vs-CRC classifier."

### P1.4  Causal language — "dilutes the probability that the most informative taxa are sampled at each split"
- **Where:** `05_discussion.md:7`
- **Problem:** Causal-mechanistic framing for an unproven mechanism. Splits are stochastic; the claim about which features are "most informative" is itself dependent on the model. Mild but easy to flag.
- **Proposed fix:** "is consistent with dilution of informative-taxon sampling probability at each split."

### P1.5  Speculation flagged as finding — "may have value in distinguishing advanced lesions"
- **Where:** `05_discussion.md:22`
- **Quote:** "stool metagenomic screens are unlikely to detect early adenomas at useful sensitivity, but may have value in distinguishing advanced lesions from carcinoma and in monitoring post-resection recurrence"
- **Problem:** Adenoma stage (advanced vs non-advanced) is explicitly listed in Limitations as *not modelled* because the data don't support it. The "may have value in distinguishing advanced lesions" sentence then makes exactly that distinction speculatively. Internal contradiction with the Limitations paragraph.
- **Proposed fix:** Drop the "advanced lesions" speculation entirely or move it to a "future directions" sentence with explicit acknowledgment that no advanced-vs-non-advanced data were analysed here.

### P1.6  Unsupported "may" — "may diminish" with substantially larger datasets
- **Where:** `05_discussion.md:7`
- **Quote:** "with substantially larger pooled datasets (Piccinno et al. 2025), the noise contribution of additional features may diminish"
- **Problem:** The cited Piccinno 2025 reaches ~0.85 AUC pooled — but that study did not specifically test species-only vs joint-pathway models with our framework, so it cannot be used to support the inference about future joint-model performance at scale. The "may diminish" is speculation, not finding.
- **Proposed fix:** "We did not test whether the relative performance of species-only vs joint models changes at the n ~3,700 scale of recent pooled analyses (Piccinno et al. 2025); this is an open question."

### P1.7  No comparison to Thomas 2019 / Wirbel 2019 headline numbers in Results
- **Where:** `04_results.md` does not directly compare the 0.781 pooled AUC to published cross-cohort numbers from Wirbel 2019 (~0.80) or Thomas 2019. The replication file `results/wirbel_replication.csv` shows a fold-by-fold comparison with a +0.012 mean delta vs Wirbel 2019, but this is not surfaced in the results text.
- **Problem:** Reviewer-2 will ask "how does this compare to the published reference framework?" The data exists. The results section should preempt the question.
- **Proposed fix:** Add one sentence to Results: "On the five Wirbel 2019-overlapping cohorts (FengQ_2015, VogtmannE_2016, WirbelJ_2018, YuJ_2015, ZellerG_2014), the species-only RF reaches mean AUC 0.824 versus 0.812 reported by Wirbel et al. 2019 on the same folds (Δ +0.012; `results/wirbel_replication.csv`)."

### P1.8  Comparative claim "0.781 vs 0.85 (Piccinno)" never quantified
- **Where:** `02_introduction.md:5`
- **Quote:** "The recent pooled analysis of 3,741 metagenomes across 18 cohorts (Piccinno et al. 2025) reached a mean AUC of approximately 0.85"
- **Problem:** Our headline pooled AUC is 0.781. A reviewer will immediately read the introduction and ask why we are 0.07 below Piccinno. The discussion does not address the gap.
- **Proposed fix:** Add to Discussion (limitations or batch-effects section): "Our pooled AUC of 0.781 is below the ~0.85 reported by Piccinno et al. 2025; the principal differences are (i) our country-aware LODO strictly excludes same-country cohorts from training, deflating folds that share population structure with the held-out set, (ii) Piccinno include 8 additional cohorts not present here, and (iii) we use per-fold rather than global pathway filtering. We interpret 0.781 as a conservative, leakage-controlled lower bound rather than as a competing point estimate."

### P1.9  "Pooled CI does not overlap the joint-RF point estimate" — misuse of CI overlap as significance test
- **Where:** `07_supplementary.md:72`
- **Quote:** "The species-RF pooled CI [0.757, 0.805] does not overlap the joint-RF point estimate (0.756), consistent with the DeLong tests"
- **Problem:** CI-overlap-vs-point-estimate is a non-standard significance test and is widely criticised. The CI [0.757, 0.805] barely excludes 0.756, and the joint-RF *CI* [0.731, 0.781] overlaps the species-RF CI substantially. The intended message ("the difference is significant per DeLong") is fine; the CI-overlap framing is loose.
- **Proposed fix:** Drop the "does not overlap the joint-RF point estimate" framing. Just say: "Pooled CIs are bootstrap-derived under cohort-stratified resampling; for formal classifier comparison we rely on the DeLong test reported in the main text (species RF vs joint RF, p = 0.0008; species RF vs joint XGB, p = 0.046)."

### P1.10  Unsupported assertion about XGBoost calibration property
- **Where:** `05_discussion.md:10`
- **Quote:** "This is a calibration property of gradient-boosted decision trees on heterogeneous tabular metagenomic data rather than a model-comparison artifact, but it argues for the Random Forest as the preferred deployment candidate where probability calibration matters"
- **Problem:** "calibration property of GBDT on heterogeneous tabular metagenomic data" is asserted as known fact with no citation and no within-paper evidence beyond n=1 contrast. Reviewer can attack as overgeneralisation.
- **Proposed fix:** "In this dataset, joint XGBoost shows a notably higher reliability term (worse calibration) than either RF; we therefore prefer RF for screening applications where probability calibration matters, and recommend isotonic or Platt-scaling recalibration if XGBoost is used downstream."

### P1.11  Limitations missing the FIT comparison limitation
- **Where:** `05_discussion.md:44`
- **Quote (Position relative to FIT):** "FIT's positive predictive value is therefore considerably higher than a microbiome-based classifier operating at the AUC observed here"
- **Problem:** The supplied `results/diagnostics/fit_vs_microbiome.csv` shows species RF at FIT-matched 94% specificity reaches sensitivity 0.42 vs FIT 0.79. That is a large clinical gap. The "as a stratifier of FIT-negative individuals at elevated baseline risk" suggestion is speculative: no FIT-stratified subgroup analysis is presented. A clinical reviewer will object that the proposed use case is not tested.
- **Proposed fix:** Add to Limitations: "We did not perform a FIT-stratified subgroup analysis (FIT results are not available in curatedMetagenomicData), so the proposed FIT-negative-stratifier use case is hypothetical and would require prospective evaluation in a screening cohort with both FIT and stool metagenomic data."

### P1.12  Underclaim: rebalancing robustness for adenoma is rigorous but the discussion only summarises it
- **Where:** `05_discussion.md:26`
- **Quote:** "The qualitative finding ... was robust across all three rebalancing strategies"
- **Problem:** This is a real piece of rigour (we ran 4 strategies × 2 models × 4 cohorts) and the result is genuinely informative, but the discussion gives it one paragraph with no numbers. A reviewer who skims will miss it. Per the instruction to flag underclaims: this is a place where the paper hedges on a finding it actually established robustly.
- **Proposed fix:** Add two numbers in the paragraph: "Healthy-vs-adenoma mean LODO AUC stayed within 0.539–0.585 across baseline / random under-sampling / SMOTE / class-weight; adenoma-vs-CRC stayed within 0.617–0.671 (`results/adenoma_rebalanced_summary.csv`). The near-chance H-vs-A signal is therefore not a class-imbalance artefact."

### P1.13  Methods conflate "Random Forest" hyperparameters and treat n_estimators=500 as defaulted with no justification
- **Where:** `03_methods.md:27`
- **Problem:** Reviewer can ask: why 500 trees and not 100 or 1000? Why max_features='sqrt' and not log2? Hyperparameters are picked without justification or sensitivity. The seed-sensitivity addresses *random* variation but not *hyperparameter* variation.
- **Proposed fix:** Add one sentence: "Hyperparameters follow the defaults adopted by Thomas et al. 2019 / Pasolli et al. and are not tuned per-fold because the joint model already fails to outperform the species baseline at default hyperparameters; tuning would be expected to narrow the gap symmetrically rather than reverse it." This is partially said already but is hidden in the middle of the paragraph.

### P1.14  "Most convincing single observation in this analysis" is editorial
- **Where:** `05_discussion.md:18`
- **Quote:** "Their reproducibility across cohorts on three continents — at top SHAP ranks in both Random Forest and XGBoost despite very different splitting criteria — is the most convincing single observation in this analysis."
- **Problem:** Editorial superlative ("most convincing single observation") — fine in a blog post, but a reviewer will pick at it.
- **Proposed fix:** "Their reproducibility across cohorts on three continents — at top SHAP ranks in both Random Forest and XGBoost despite very different splitting criteria — is among the most reproducible findings in this analysis."

### P1.15  Per-fold pathway-filter framing inflates apparent rigour
- **Where:** `02_introduction.md:9` and `03_methods.md:11`
- **Problem:** The species filter is global (Methods admits this on line 11), so the headline framing "we implement per-fold pathway filtering ... to eliminate two distinct sources of information leakage" overstates: only the *pathway* filter is per-fold, the *species* filter is global. The 3 justifications in Methods are reasonable but Introduction loses the qualifier.
- **Proposed fix:** Edit the introduction to read "per-fold pathway filtering (the species filter remains global for the reasons stated in Methods) ..."

### P1.16  Adenoma claim "supports a stepwise oral-pathobiont enrichment" is a strong biological inference from cross-sectional, 4-cohort, n=183 data
- **Where:** `01_abstract.md:7` and `05_discussion.md:22`
- **Quote (abstract):** "consistent with a stepwise oral-pathobiont enrichment along the adenoma-carcinoma sequence"
- **Quote (discussion):** "the oral-pathobiont enrichment is acquired at or near the transition to invasive carcinoma"
- **Problem:** The discussion phrasing slips from the abstract's "consistent with" to "is acquired at or near" — a causal/temporal claim that is not supported by purely cross-sectional cohorts with no longitudinal follow-up. Limitations does acknowledge no longitudinal data, but the discussion narrative doesn't honour that.
- **Proposed fix:** Replace "the oral-pathobiont enrichment is acquired at or near the transition to invasive carcinoma" with "the cross-sectional pattern is consistent with — but does not establish — the oral-pathobiont signature emerging at or near the transition to invasive carcinoma; longitudinal evidence would be required to test the temporal claim directly."

---

## P2 — Polish / consistency

### P2.1  Typo in Methods: `combat.pycombat.pycombat`
- **Where:** `03_methods.md:53`
- **Quote:** "as implemented in `combat.pycombat.pycombat`"
- **Fix:** Should be `combat.pycombat.pycombat` → `combat.pycombat.pycombat()` if it's the function, or check what the actual module path is. As written it is a redundant triple-namespace and reads as a copy-paste artefact.

### P2.2  Title overstates "outperform" given DeLong-only basis
- **Where:** `00_title.md:1`, `00_title.md:12`
- **Quote (running title):** "Species-only classifiers outperform joint models for CRC."
- **Problem:** Same issue as P0.2 — "outperform" is supported by pooled DeLong only; per-cohort paired tests are null. Title and running title carry the strongest possible framing.
- **Proposed fix:** Consider softening the running title to "Species-level features match or exceed joint species-plus-pathway models for CRC." Title can stay if abstract conclusion is hedged per P0.2.

### P2.3  Abstract uses both "above-chance" and "moderate" for the same finding
- **Where:** `01_abstract.md:7` says "moderate discrimination for adenoma-vs-CRC (RF 0.671, XGB 0.617)" and `04_results.md:44` says "moderate above-chance discrimination".
- **Fix:** Pick one wording. "Moderate" alone is fine; "above-chance" is implied.

### P2.4  Word-count claim "~4,800 words" not verified
- **Where:** `00_title.md:14`
- **Fix:** Re-count before submission; many journals enforce a strict word limit and 4,800 is close to the typical 5,000-word cap.

### P2.5  "Approximately 45 minutes on a standard workstation" — unspecified
- **Where:** `03_methods.md:63`
- **Fix:** Either specify the workstation (CPU, RAM) or remove. Reviewers like reproducibility specifics; a bare "standard workstation" reads as hand-waving.

### P2.6  "(Sun et al. 2025)" vs "Sun and Xu (2014)" — author short-form ambiguity
- **Where:** `02_introduction.md:11` ("benchmarking ... (Sun et al. 2025)") vs `02_introduction.md:13` ("Sun and Xu 2014")
- **Problem:** Two unrelated "Sun" first-authors a few sentences apart. Year disambiguates but a hostile reader might misread.
- **Fix:** Use first-author initials on first mention: "Sun Y. et al. 2025" and "Sun X. and Xu W. 2014".

### P2.7  Discussion paragraph on Brier decomposition has no paragraph break before "The joint XGBoost model exhibits..."
- **Where:** `05_discussion.md:9`–`05_discussion.md:10`
- **Problem:** Lines 9 and 10 are one block in the markdown; the Brier paragraph reads as appended to the SHAP-vs-permutation paragraph with no thematic transition. Flow problem.
- **Fix:** Insert blank line + a transition: "We also assessed probability calibration. The joint XGBoost..."

### P2.8  Repetition across Methods / Results / Discussion about ComBat AUC numbers
- **Where:** `03_methods.md:23`, `03_methods.md:53`, `04_results.md:33`, `05_discussion.md:30`, `07_supplementary.md:86`
- **Problem:** The 0.815 vs 0.807 ComBat number appears in five places with minor wording variation. Repetition without new content; reads as padding.
- **Fix:** State once in Methods, once in Results (with the +0.008 Δ); cut from Discussion or merge into the batch-effects paragraph in a single line.

### P2.9  Repetition: "adenoma and CRC are biologically distinct microbiome states rather than two points on a smooth severity gradient"
- **Where:** Same sentence appears verbatim or near-verbatim in `04_results.md:46` and `05_discussion.md:22`
- **Fix:** Cut one — keep the Discussion version, drop from Results.

### P2.10  Inconsistent terminology: "control" vs "healthy" vs "H-vs-A"
- **Where:** Methods (line 7) explicitly notes "we use 'control' throughout for samples coded as such"; Results then introduces "H-vs-A" using "healthy" as the H, and Figure 4 legend says "Control vs adenoma (H-vs-A)" combining both labels in one phrase.
- **Fix:** Replace "H-vs-A" with "C-vs-A" throughout to honour the Methods convention; if the H-shorthand is kept, add a one-line note in Methods explaining "H denotes 'control' for legacy shorthand reasons."

### P2.11  Hedge mismatch on confounder result
- **Where:** `04_results.md:29` says "**confirming** that the classifier's discrimination is not driven by demographic confounders"; `07_supplementary.md:82` says "is not driven by these standard demographic confounders".
- **Problem:** "Confirming" is too strong from a 0.800–0.814 range vs 0.807 baseline. Should be "consistent with" — the analysis cannot rule out small residual confounding within the noise floor.
- **Fix:** Change "confirming" → "consistent with the view that the classifier's discrimination is not materially driven by age, sex, or BMI within this dataset."

### P2.12  "Most of the DeLong signal arises from the YachidaS_2019 fold" — true but buried
- **Where:** `04_results.md:19`
- **Problem:** This is a candid caveat that gets one sentence and is then forgotten. The Discussion does not propagate it; the Conclusion does not propagate it. This understates the caveat's importance.
- **Fix:** Carry this caveat into the Discussion: "The DeLong significance is driven primarily by the largest fold (YachidaS_2019, n_test = 508); the absolute pooled Δ AUC is 0.025 for species-RF vs joint-RF and 0.015 for species-RF vs joint-XGB. The qualitative ranking (species-only ≥ joint) is consistent across all folds and tests, but the strict statistical-significance claim does depend on YachidaS_2019 dominating the pooled sample."

### P2.13  Adenoma sample-size n=183 framed as one number throughout, but per-task n varies
- **Where:** `01_abstract.md:7`, `04_results.md:41`, `05_discussion.md:22`, `07_supplementary.md:94`
- **Problem:** "n=183 adenomas" is correctly stated, but per-task n is different: H-vs-A trains on 4 cohorts with up to 470+ controls + 183 adenomas; A-vs-CRC trains with up to 357+ CRC + 183 adenomas. The single "n=183" number understates the dataset complexity. Per-fold rebalanced LODO CSV shows train sizes 244–528 per fold.
- **Fix:** Add to Results: "Each adenoma-LODO fold trains on 232–528 samples and tests on 51–325 samples per task (`results/adenoma_rebalanced_lodo.csv`)."

### P2.14  Reference (`Piccinno et al. 2025`) used to support TWO different points
- **Where:** `02_introduction.md:5` (motivation), `05_discussion.md:7` (speculation about future joint-model performance)
- **Problem:** Using the same reference to both motivate the work and to back a speculative forward-looking claim is fine, but the Discussion citation is doing work the cited paper does not directly do (they did not specifically test species-only vs joint at n=3,700). See P1.6.

### P2.15  "Drives" word-count
- **Where:** A quick search shows "drives", "driven", "driven by" appear several times in observational contexts (e.g., `04_results.md:9`: "its size dominates the pooled estimate"; `05_discussion.md:18`: "drives the moderate adenoma-vs-CRC"; `05_discussion.md:29`: "the classifier's discrimination is not driven by demographic confounders").
- **Fix:** Audit and replace causal verbs with "associated with", "dominated by", or "weighted toward" in observational contexts. Keep "driven by" only where the noun is something the method explicitly controls (e.g., "driven by the largest fold" is acceptable because the fold size is a deterministic methodological feature, not a causal biological claim).

### P2.16  Conclusion does not mention the adenoma class-imbalance robustness, FIT positioning, or external validation
- **Where:** `05_discussion.md:48`
- **Problem:** The Conclusion summarises species-only vs joint and the adenoma stepwise model, but skips three other things the paper did rigorously: class-imbalance robustness, FIT positioning, and the wirbel_replication sanity-check vs Wirbel 2019. The Conclusion thus undersells the paper.
- **Fix:** Add one sentence: "Robustness across rebalancing strategies, comparison with FIT at matched specificity, and replication of Wirbel et al. 2019 fold-by-fold AUCs are reported in Supplementary Notes."

---

## Reviewer-2 attack vectors — summary table

| # | Attack | Where the reviewer lands | Manuscript counter-evidence | Fix |
|---|---|---|---|---|
| R1 | "Your ComBat 'no-leakage' claim is wrong — you fit on train+test." | `03_methods.md:23`, `03_methods.md:53` | None — the manuscript wording is itself the problem. | See P0.1. |
| R2 | "Your headline DeLong significance is driven by one cohort (YachidaS_2019, n=508)." | `04_results.md:19` already concedes this. | The concession exists in Results but is dropped in Abstract / Discussion / Conclusion. | Propagate the caveat. See P0.2, P2.12. |
| R3 | "Your pooled AUC (0.781) is well below Piccinno 2025 (0.85). Why?" | `02_introduction.md:5` introduces 0.85; Discussion never addresses gap. | None. | Add explanatory paragraph. See P1.8. |
| R4 | "You overclaim a positive biological model for the adenoma–carcinoma sequence from cross-sectional data." | `05_discussion.md:22` "is acquired at or near the transition" | Limitations does state no longitudinal data. | Tighten language. See P1.16. |
| R5 | "Your FadA / β-catenin / Barrett's claims have no citations." | `05_discussion.md:18` | None. | Cite or cut. See P1.1. |
| R6 | "You did not tune hyperparameters or test at different n_estimators." | `03_methods.md:27` | The justification ("joint model fails at defaults") is present but buried. | Surface the justification. See P1.13. |
| R7 | "Your top-feature lists in Results and Figure 3 legend contradict each other and the SHAP CSV." | `04_results.md:37` vs `04_results.md:54` | The SHAP CSV is the ground truth. | Rewrite both. See P0.4. |
| R8 | "Two top-20 species (E. eligens, P. distasonis) show nominally significant per-cohort SHAP-rank vs depth correlations; you only report F. nucleatum." | `05_discussion.md:9` | None — the data are in the supplied CSV. | Add the two species, note Bonferroni. See P0.3. |
| R9 | "0.998 in Discussion, 0.999 in Methods — pick one." | `03_methods.md:19` vs `05_discussion.md:30` | None. | Pick 0.998. See P0.5. |
| R10 | "Your XGBoost calibration claim is over-generalised." | `05_discussion.md:10` | None. | Restrict to this dataset. See P1.10. |
| R11 | "Why no comparison to Wirbel 2019 numbers in the body? You have the data." | `04_results.md` | `results/wirbel_replication.csv` exists but is not surfaced. | Add one sentence. See P1.7. |
| R12 | "Your FIT-stratifier proposal is hypothetical and untested." | `05_discussion.md:44` | None. | Acknowledge in Limitations. See P1.11. |

---

## Quick wins (do these first; high impact / low edit cost)

1. Fix the 0.998 vs 0.999 contradiction (one number swap). P0.5.
2. Rewrite the joint-RF / joint-XGB top-features paragraph and Figure 3 legend to match the SHAP CSV. P0.4.
3. Add the two nominally-significant depth–rank correlations (E. eligens, P. distasonis) and state the Bonferroni threshold. P0.3.
4. Soften the abstract conclusion "superior" to "match or modestly exceed" and surface the YachidaS_2019 caveat. P0.2.
5. Either re-run ComBat with train-only fit OR explicitly state the joint-fit is a known caveat and the +0.008 is an upper bound. P0.1.
6. Cite or cut the FadA / β-catenin / Barrett's sentences. P1.1.
7. Add the explanatory sentence on the 0.781 vs 0.85 (Piccinno) gap. P1.8.
8. Add the one-line Wirbel-replication sentence to Results. P1.7.

---

## What is in good shape

- The hedge structure around the per-cohort vs DeLong contrast is mostly honest (Results body acknowledges YachidaS_2019 dominance, n=10 power problem, etc.).
- Methods are reproducible: scripts, hyperparameters, seeds, and per-fold filter logic are specified.
- Limitations paragraph is genuinely candid about cross-sectional design, geographic gaps, no nested-CV, etc.
- The robustness battery (seeds × thresholds × confounders × ComBat × biology shortlist × stratified pathways × rebalancing) is more thorough than typical for this class of paper.
- Reference list is internally consistent; no orphan refs; in-text citations match (per the prior citation audit).

The paper's core claim survives all these criticisms; the audit findings are about how the claim is *expressed* rather than whether it holds. The main exposure points are the ComBat leakage framing (P0.1) and the abstract/title overclaim relative to the per-cohort null results (P0.2 / P2.2), both of which a hostile reviewer will pick on within the first read.
