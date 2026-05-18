# Methodological Concerns — How Each Has Been Addressed

*Alejandro Velazquez and Rachel Selbrede — 2026-05-18*

Walkthrough of the four methodological concerns Tal raised in the prior meeting, with what was done, evidence, and the remaining open question for each.

---

## 1. Study-level batch effects dominate

> *"Study-level batch effects are going to dominate any signal you think you're seeing. You need to address that head-on before you trust any cross-cohort number."*

**What was done.**
- LODO is the primary evaluation protocol everywhere (`scripts/lodo_cv.py`).
- **Country-aware LODO** as a stress test: when a fold's test cohort shares a country with a training cohort, the training-country sibling is also held out. Makes country confounding visible rather than hidden.
- **Per-fold ComBat** as a robustness check (`scripts/batch_correction.py`, `results/combat_results.csv`) — applied within fold to avoid information leak.
- **Bootstrap CIs and DeLong** on pooled predictions (`results/bootstrap_ci.csv`, `results/delong_results.csv`) so differences are tested, not just compared.

**Evidence.**
- ThomasAM_2019_c (Japan): vanilla LODO AUC **0.998** → country-aware LODO (excluding YachidaS_2019, also Japan) **0.836**. Exactly the inflation pattern country confounding produces.
- ComBat pooled AUC 0.815 vs raw 0.807 — within CI overlap; ranking of species RF vs joint unchanged. Batch correction does not rescue the joint model.

**Open question.** Whether residual country/batch structure is hiding a real pathway signal that ComBat is over-correcting. The stratified pathway pilot (forthcoming `results/stratified_pathway_pilot.csv`) is the next probe.

---

## 2. Joint species + pathway risks over-parameterization

> *"If you throw everything into one model, you're going to over-parameterize and lose generalization. You need to show that the joint model actually does better than species alone, and if it doesn't, that's the result."*

**What was done.**
- Trained species-only, pathway-only, biological-shortlist, and joint (species+pathway) models, same LODO splits, same hyperparameters where comparable (`scripts/train_baseline.py`, `scripts/train_joint.py`, `scripts/bio_pathway_shortlist.py`).
- DeLong test on pooled predictions to test the difference, not just compare points.

**Evidence (`results/delong_results.csv`).**

| Comparison              | ΔAUC    | z     | p       |
|---|---:|---:|---:|
| species RF vs joint RF  | 0.0251  | 3.35  | **0.0008** |
| species RF vs joint XGB | 0.0152  | 2.00  | 0.046   |
| joint XGB vs joint RF   | 0.0099  | 1.30  | 0.19    |

**Read.** This is the clean negative you asked us to surface if it was there. We're calling it as such in the manuscript Discussion (`manuscript/markdown/05_discussion.md`).

**Open question.** Whether the degradation is information-theoretic (species already encodes everything the pathways contribute, plus pathways add noise) or pathway-quality (HUMAnN3 noise from short-read assembly). See concern #3 below.

---

## 3. Granular functional features need careful selection

> *"You can't just throw all 400+ pathways into the model. You need a principled selection strategy, and the selection has to be done inside the fold."*

**What was done.**
- **Prevalence × abundance filter sweep** (`scripts/sensitivity_analysis.py`): 20 cells across prevalence ∈ {0.05, 0.10, 0.15, 0.20} and abundance ∈ {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}. Filter fit on **training cohorts only**, applied to held-out cohort.
- **Validate-then-filter** pipeline in `scripts/filter_pathways.py` and `scripts/validate_pathways.py`.
- All feature selection per-fold, no test-cohort leakage (audited in `scripts/sanity_check.py`).

**Evidence (`results/sensitivity_thresholds.csv`).**
- Per-cohort mean AUC range: **0.794–0.812** across all 20 cells. Spread 0.018 — within bootstrap CI width.
- Best cell: prevalence 0.05, abundance 1e-4 → 0.812 with mean 371 features.

**Read.** Filter choice doesn't move the answer. The negative result is not "we picked the wrong filter."

**Open question.** Same as #2 — is the ceiling set by HUMAnN3 quality, or by species redundancy?

---

## 4. Biological priors should drive pathway selection

> *"If you're using pathway-level features, the selection should be guided by what's biologically plausible for CRC, not just statistical filtering."*

**What was done.**
- `scripts/bio_pathway_shortlist.py` — 8 CRC-relevant pathway groups, ~84 candidates total:
  1. **butyrate / SCFA** (19 pathways)
  2. **fermentation** (15)
  3. **bile-acid metabolism**
  4. **LPS / menaquinone biosynthesis** (incl. PWY-6263, PWY-6478 — both surface in adenoma-vs-CRC SHAP)
  5. **amino-acid degradation** (arginine/lysine/histidine/ornithine — high SHAP rank in `results/shap_adenoma_vs_crc.csv`)
  6. **nucleotide salvage**
  7. **sulfur metabolism**
  8. **mucin degradation**
- Full list: `results/bio_pathway_shortlist.txt`.
- ~84 candidates → ~66 retained per fold after training-cohort-only prevalence/abundance filter.

**Evidence (`results/bio_pathway_results.csv`).**
- Per-cohort mean AUC **0.823** (range 0.679–0.936 across cohorts).
- At parity with species (0.807), not above; same direction as the joint result.
- *Independently:* the top-ranked pathways in the adenoma-vs-CRC SHAP (menaquinone PWY-6263, GDP-D-glycero-α-D-manno-heptose PWY-6478, arginine biosynthesis ARGSYN-PWY, lysine biosynthesis PWY-2941) sit in the LPS / menaquinone and amino-acid groups of the shortlist — the biology lines up.

**Open question.** Whether the shortlist should be tightened further (e.g., drop fermentation/SCFA since the biology there is bidirectional in CRC), or expanded to include KEGG-module-level features at the same granularity.

---

## Summary

| Concern               | Status        | Evidence                                 |
|---|---|---|
| Batch effects         | Addressed     | Country-aware LODO + per-fold ComBat     |
| Over-parameterization | Confirmed     | DeLong p=0.0008 (species RF > joint RF)  |
| Granular selection    | Addressed     | 20-cell sweep, spread 0.018              |
| Biological priors     | Implemented   | 8 groups, ~66 features/fold, parity      |
