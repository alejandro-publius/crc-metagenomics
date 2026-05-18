# 2-Minute Pitch — Speaker Notes

*Alejandro Velazquez and Rachel Selbrede — 2026-05-18*

Three beats, ~2:00 total. Bullets are talking points, not script.

---

### Beat 1 — What we set out to do (~30s)

- Started from the Thomas-2019 LODO benchmark — 7 cohorts, species-only RF, ~0.80 cross-cohort AUC.
- Two questions: **(a)** does the result hold at full curatedMetagenomicData scale, and **(b)** do functional features (HUMAnN pathways) add anything once species are in the model?
- Built out to 10 cohorts, 1522 samples (added YachidaS_2019, WirbelJ_2018, GuptaA_2019; excluded HanniganGD_2017 under pre-specified criteria). Country-aware LODO + per-fold ComBat for the batch-effect concerns you raised.

### Beat 2 — What we found (~60s)

- **Headline:** species-only RF wins. Per-cohort mean 0.807, pooled 0.781 [0.757, 0.805] over 10,000 bootstrap iters.
- Joint species + pathway **degrades**: RF pooled 0.756, DeLong z=3.35, **p=0.0008**. XGB joint p=0.046. So this is not "no improvement" — it's a significant drop pooled across cohorts.
- Biological pathway shortlist (8 CRC-relevant groups, ~66 features/fold after training-cohort-only filtering) gets to **parity** with species (0.823 per-cohort) but does not exceed it. Sensitivity sweep across 20 (prevalence × abundance) filter cells: 0.794–0.812, spread 0.018. Result is robust to the filter, not an artifact of one cell.
- **Interpretation:** at MetaPhlAn3/HUMAnN3 resolution in this corpus, pathway abundance carries no information that species relative abundance does not already encode, and adding it hurts generalization. Framing this as a clean negative result, not a failure.
- **Adenoma:** Healthy-vs-Adenoma is at chance across cohorts (0.56–0.58). Adenoma-vs-CRC is real (0.62–0.67), and the SHAP signature is dominated by oral pathobionts — *F. nucleatum, P. stomatis, P. micra, G. morbillorum*. The oral-pathobiont signal **emerges at the malignant transition**, not at the adenoma precursor. That's a biological finding, not just a modeling artifact.

### Beat 3 — What's next (~30s)

- **Two analyses landing this week:** rebalanced LODO for the adenoma task (parallel agent), and the species-resolved (stratified) pathway pilot — gives us a direct test of whether the pathway signal lives in *who's doing it* vs *what's being done*.
- **Stuck on:** the pathway-quality-vs-pathway-information disentanglement. Rerunning HUMAnN on raw FASTQs is deferred (multi-week compute). Want your input on what surrogate diagnostics would be compelling.
- **Adenoma subtype** — would stratifying by advanced vs non-advanced be worth pursuing if curatedMetagenomicData metadata supports it?
- Manuscript is drafted (`manuscript/`); want your prior on submission venue.

---

**One-sentence ask:** "We have a defensible negative result on joint pathway+species at this resolution, and an interesting positive result on oral pathobionts at the malignant transition — looking for your guidance on the pathway-quality diagnostic and on submission strategy."
