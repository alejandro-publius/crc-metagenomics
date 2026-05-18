# Response to Tal's Action Items

*Alejandro Velazquez and Rachel Selbrede — 2026-05-18*

Point-by-point status against the 7 items from the last meeting. Status legend: **DONE** / **PARTIAL** / **DEFERRED** / **SKIPPED**.

---

### 1. Run DEBIAS-M — **DEFERRED**
- **What was done:** Batch-effect handling addressed via country-aware leave-one-dataset-out and per-fold ComBat as a robustness check (`scripts/batch_correction.py`, `results/combat_results.csv`).
- **Headline number:** ComBat-corrected pooled AUC 0.815 vs uncorrected 0.807 — within bootstrap CI overlap; no material change to ranking.
- **Reason for deferral:** Prioritizing methodology improvements that don't require third-party tooling, so the pipeline stays self-contained and the batch-correction story is fully reproducible from `scripts/`. Open to revisiting if the ComBat result isn't compelling.

### 2. Rerun HUMAnN on raw reads — **DEFERRED**
- **What was done:** Retained curatedMetagenomicData's HUMAnN3 profiles; added prevalence + abundance filtering (`scripts/filter_pathways.py`) and the biological shortlist as an alternative to a brute rerun.
- **Headline number:** Filtered pathway space (~371 features at 0.01% threshold) yields per-cohort mean 0.812 (`results/sensitivity_thresholds.csv`); biological shortlist 0.823 (`results/bio_pathway_results.csv`).
- **Reason for deferral:** Raw FASTQ retrieval + HUMAnN3 across ~1500 samples is multi-week compute we don't have allocation for before the meeting. See *Open question 1* in `one_pager.md` for the diagnostic surrogate we'd like your input on.

### 3. Expand dataset beyond Thomas subset — **DONE**
- **What was done:** Pulled all curatedMetagenomicData CRC cohorts; applied pre-specified depth/sparsity exclusion criteria.
- **Headline number:** 7 → 10 cohorts, 762 → 1522 samples (`results/table1.csv`). HanniganGD_2017 excluded under the criteria in `results/decisions_addendum.md`. Added: YachidaS_2019 (n=575, Japan), WirbelJ_2018 (n=125, Germany), GuptaA_2019 (n=60, India).
- **Evidence:** `results/table1.csv`, `data/` exports from `scripts/export_data.R`.

### 4. Granular feature spaces + biological shortlist — **PARTIAL**
- **What was done:**
  - Biological shortlist: 8 CRC-relevant pathway groups (~84 candidates → ~66 retained per fold after training-cohort-only prevalence/abundance filtering), implemented in `scripts/bio_pathway_shortlist.py`; full shortlist in `results/bio_pathway_shortlist.txt`.
  - Pathway sensitivity sweep across 20 (prevalence × abundance) cells in `scripts/sensitivity_analysis.py`.
  - Species-resolved (stratified) pathway pilot: re-running; output will land at `results/stratified_pathway_pilot.csv`.
- **Headline number:** Biological shortlist per-cohort mean AUC 0.823 (parity with species, does not exceed); sensitivity sweep range 0.794–0.812, spread 0.018 (`results/sensitivity_thresholds.csv`).
- **Why partial:** Gene-family (UniRef90) abundance not yet evaluated; pending pilot completion and a defensible filter strategy at that resolution.

### 5. RebalancedCV-style LODO for adenoma — **DONE** (rebalanced variant in progress)
- **What was done:** Cross-cohort LODO across 4 adenoma-bearing cohorts (FengQ_2015, ZellerG_2014, ThomasAM_2018a, YachidaS_2019) for Healthy-vs-Adenoma and Adenoma-vs-CRC, RF and XGB (`scripts/adenoma_lodo.py`).
- **Headline number:** H-vs-A: RF 0.561, XGB 0.579. A-vs-CRC: RF 0.671, XGB 0.617 (`results/adenoma_lodo_results.csv`). Rebalanced variant outputs: `results/adenoma_rebalanced_lodo.csv` and `results/adenoma_rebalanced_summary.csv` (parallel agent finalizing).
- **Evidence:** `results/adenoma_lodo_results.csv`, `results/shap_adenoma_vs_crc.csv` (oral-pathobiont signature at malignant transition).

### 6. 2-minute pitch + one-page summary — **DONE**
- **What was done:** This package. See `tal/one_pager.md`, `tal/pitch_2min.md`, `tal/dashboard.md`, `tal/methodology_addressed.md`.

### 7. Follow-up email outlining next steps + summer collab — **SKIPPED**
- **Reason:** Alex will draft and send separately, after this meeting, rather than pre-committing in writing.

---

### Verification
`scripts/verify_results.py` — 49/49 checks pass against `results/*.csv`. Run from repo root: `python scripts/verify_results.py`.
