# CRC Metagenomics Update for Tal Korem & George Austin — 2026-05-18

*Alejandro Velazquez and Rachel Selbrede*

---

### Since last meeting
- **Cohort expansion.** Thomas-2019 subset (7 cohorts, 762 samples) extended to all eligible curatedMetagenomicData CRC studies: **10 cohorts, 1522 samples** (674 CRC / 665 control / 183 adenoma). Added YachidaS_2019, WirbelJ_2018, GuptaA_2019; HanniganGD_2017 excluded under pre-specified depth/sparsity criteria (`results/decisions_addendum.md`).
- **Country-aware LODO** with per-fold ComBat robustness check; full robustness battery (sensitivity sweep, seed sensitivity, confounder adjustment, DeLong, bootstrap CIs) re-run on the expanded set.
- **Biologically-guided pathway shortlist** implemented (`scripts/bio_pathway_shortlist.py`, 8 CRC-relevant groups: butyrate/SCFA, fermentation, bile-acid, LPS/menaquinone, amino-acid degradation, nucleotide salvage, sulfur, mucin) with training-cohort-only filtering each fold.
- **Adenoma cross-cohort LODO** rebuilt across the 4 cohorts with adenoma labels (rebalanced variant being finalized; see *Open items*).

### Headline result (clean negative for joint species + pathway features)
Species-only random forest is the strongest model and joint species+pathway features **degrade** performance. Per-cohort means and 10,000-iter pooled bootstrap CIs on n=1339 CRC/control samples (`results/bootstrap_ci.csv`, `results/delong_results.csv`):

| Model | Per-cohort mean AUC | Pooled AUC [95% CI] | DeLong vs species RF |
|---|---|---|---|
| Species RF | 0.807 | 0.781 [0.757, 0.805] | — |
| Joint RF | 0.804 | 0.756 [0.731, 0.781] | **z=3.35, p=0.0008** |
| Joint XGB | 0.797 | 0.766 [0.740, 0.791] | z=2.00, p=0.046 |

The biological pathway shortlist (~66 retained features/fold) holds parity with species (per-cohort mean 0.823, `results/bio_pathway_results.csv`) but does not exceed it. Reading: at MetaPhlAn/HUMAnN resolution, pathway abundances in this corpus carry no information that species relative abundance does not already encode, and adding them inflates parameters in a way the model cannot recover from across cohort shifts.

### Adenoma
Cross-cohort LODO across the 4 adenoma-bearing cohorts (FengQ_2015, ZellerG_2014, ThomasAM_2018a, YachidaS_2019; `results/adenoma_lodo_results.csv`):

| Task | RF AUC | XGB AUC |
|---|---|---|
| Healthy vs Adenoma | 0.561 | 0.579 |
| Adenoma vs CRC | 0.671 | 0.617 |

The healthy→adenoma signal is at chance; the adenoma→CRC SHAP signature is dominated by **oral pathobionts** — *Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum* (`results/shap_adenoma_vs_crc.csv`). The oral-pathobiont signature appears at the malignant transition, not at the precursor. Rebalanced LODO results (`results/adenoma_rebalanced_lodo.csv`) pending and will be slotted into the dashboard.

### Open items
- **Pathway-quality vs pathway-information disentanglement.** HUMAnN was run by curatedMetagenomicData on shotgun reads at upload time; rerunning on raw FASTQs is deferred (multi-week compute, raw FASTQ retrieval). Stratified (species-resolved) pathway pilot is running and will land in `results/stratified_pathway_pilot.csv`.
- **Adenoma sample size.** 183 adenoma across 4 cohorts; no metadata harmonization yet for advanced vs non-advanced subtype.
- **Submission target.** Manuscript draft complete in `manuscript/`; venue not committed.

### Questions for you
1. With raw-FASTQ HUMAnN deferred, what diagnostics would convince you that the joint-feature degradation is **information-theoretic** (species already encodes it) rather than **pathway-quality** (HUMAnN noise)? We considered KEGG module collapsing, MetaCyc super-pathway aggregation, and per-feature mutual information vs species, but want your prior.
2. The adenoma signal is weak (0.56–0.58 H-vs-A). If curatedMetagenomicData metadata supported it, would you stratify by adenoma type (advanced vs non-advanced, tubular vs villous) before declaring the precursor signal absent?
3. Submission strategy — your read on Genome Medicine vs Microbiome vs Gut Microbes for a methods-forward LODO + negative-result-on-pathways paper?

---
**Code:** github.com/alejandro-publius/crc-metagenomics — 49/49 `scripts/verify_results.py` checks pass.
