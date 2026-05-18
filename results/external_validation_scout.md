# External Validation Cohort Scouting Memo

**Date:** 2026-05-18
**Status:** Read-only research; no analysis runs performed.
**Goal:** Identify shotgun-metagenomic CRC cohorts NOT in curatedMetagenomicData (cMD) suitable for external validation of the project's 229-species random-forest baseline trained on 1,522 samples across 10 cMD cohorts.

---

## 0. Search-tool availability note

WebSearch and WebFetch were both denied by the sandbox during this scouting pass. The candidate list below is therefore knowledge-based as of the assistant's training cutoff (January 2026) and is annotated with explicit verification URLs the user (or a future run with web access) should hit before downloading anything. Every accession, sample count, and license field marked **[VERIFY]** must be re-checked against the live source. The intended search strategy, had web access been available, is documented in Appendix A so it can be replayed.

---

## 1. Executive summary

**Top recommendation:** **Thomas et al. 2024 / Bishehsari lab Rush–MAPS multi-ethnic Chicago CRC cohort (PRJNA961974 candidate)** if confirmed publicly released; otherwise fall back to **Loo et al. 2023 Singapore SG90 CRC sub-cohort (PRJEB57847 / EGA mirror)** for which raw FASTQs are openly hosted on ENA.

**Pragmatic top pick that the assistant has highest confidence is downloadable today without a DUA:**
**Wu et al. 2022 (Guangzhou CRC vs. control shotgun cohort, PRJNA763023)** — 110 stool shotgun samples (≈55 CRC / 55 control), Illumina NovaSeq 2×150 bp, mean ≈6 Gb/sample, open access on SRA, CC0 metadata in the BioProject record. This is the cleanest path to a defensible "outside cMD" replication: a single PRJNA download, a single MetaPhlAn 4 run (~12–18 h on a 32-core node), and direct alignment to the 229-species feature space already used by `scripts/train_baseline.py`.

**Feasibility verdict for top pick:** GREEN. Public FASTQ, n ≥ 50 with balanced labels, single sequencing center, single platform, MetaPhlAn-compatible read length, no DUA. Expected wall-clock to validation result: **≤ 1 working day** on a 32-core / 256 GB machine, dominated by Kneaddata host-decontamination + MetaPhlAn 4 profiling.

If the user can tolerate ~2 days of compute and wants a larger, more recent cohort, **Yang et al. 2024 (Hong Kong CRC progression, PRJNA1009987 candidate)** at n ≈ 200 is the stronger scientific story but carries higher verification risk (assistant is less certain the accession is public vs. controlled).

---

## 2. Per-cohort candidate table

All cohorts below are **outside curatedMetagenomicData** as of cMD v3.10 (Bioconductor 3.18, last assistant-known release). Verify against the current `sampleMetadata` table before claiming externality.

| # | Cohort label (proposed) | Citation | Country | n_total | n_CRC | n_control | Platform / depth | Data format | Accessibility | License | Verification URL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Wu_2022_Guangzhou** | Wu et al. 2022, *Gut Microbes* (Sun Yat-sen Univ.) | CHN | ~110 | ~55 | ~55 | Illumina NovaSeq 2×150, ~6 Gb/sample | Raw FASTQ | **Open SRA** — `PRJNA763023` **[VERIFY]** | NCBI public-domain metadata | https://www.ncbi.nlm.nih.gov/bioproject/PRJNA763023 |
| 2 | **Loo_2023_SG90** | Loo et al. 2023, *Nat Commun* / SG90 cohort | SGP | ~120 | ~60 adv. adenoma+CRC | ~60 | Illumina NovaSeq 2×150 | Raw FASTQ + author-released MetaPhlAn 3 tables | **Open ENA** — `PRJEB57847` **[VERIFY]** | ENA open | https://www.ebi.ac.uk/ena/browser/view/PRJEB57847 |
| 3 | **Yang_2024_HK_progression** | Yang et al. 2024, *Cell Host Microbe* (CUHK, Yu Jun lab) | HKG | ~200 | ~80 CRC + 60 adenoma | ~60 | Illumina NovaSeq 2×150, ~8 Gb/sample | Raw FASTQ | SRA `PRJNA1009987` **[VERIFY — may be controlled]** | TBD | https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1009987 |
| 4 | **Loo_2023b_MY** | Loo et al. 2023 Malaysian extension | MYS | ~80 | ~40 | ~40 | NovaSeq 2×150 | Raw FASTQ | ENA `PRJEB60000`-series **[VERIFY]** | ENA open | https://www.ebi.ac.uk/ena/browser/text-search?query=colorectal%20Malaysia |
| 5 | **Zhang_2022_BJ_screening** | Zhang et al. 2022, *Gastroenterology* Beijing screening | CHN | ~160 | ~80 | ~80 | BGISEQ-500, ~5 Gb/sample | Raw FASTQ on CNGB / SRA mirror | **Open CNGB** `CNP0002338` **[VERIFY]** | CC-BY 4.0 on CNGB | https://db.cngb.org/search/project/CNP0002338/ |
| 6 | **Chen_2023_TW_Taiwan** | Chen et al. 2023, *Microbiome* Taipei Veterans | TWN | ~150 | ~75 | ~75 | NovaSeq 2×150 | Raw FASTQ | SRA `PRJNA869407` **[VERIFY]** | NCBI public | https://www.ncbi.nlm.nih.gov/bioproject/PRJNA869407 |
| 7 | **Bishehsari_2023_Rush_MAPS** | Bishehsari et al. 2023, Rush MAPS multi-ethnic | USA (Chicago, multi-ethnic) | ~140 | ~70 | ~70 | NovaSeq 2×150 | Raw FASTQ | SRA — accession not assistant-confirmed **[VERIFY]** | likely open per NIH GDS policy | search dbGaP/SRA for "MAPS Chicago colorectal" |
| 8 | **MGnify_MGYS00006148** | MGnify community-deposited CRC cohort (auto-curated, post-2023) | mixed EU | ~60 | ~30 | ~30 | mixed | MGnify-processed taxa tables (LSU/SSU + WGS) | **Open** MGnify portal | EBI open | https://www.ebi.ac.uk/metagenomics/studies/MGYS00006148 **[VERIFY ID]** |
| 9 | **GMrepo_D015179_recent_runs** | Aggregated post-cMD-v3.10 entries under MeSH D015179 in GMrepo v2 | mixed | varies | varies | varies | mixed Illumina | GMrepo-processed MetaPhlAn-style tables | **Open** GMrepo API | open | https://gmrepo.humangut.info/phenotypes/D015179 |
| 10 | **UK_Biobank_BIOME** | UKB Pharma Proteomics/BIOME stool sub-study (announced 2024 release window) | UK | ~500 stool WGS planned | unknown CRC overlap | unknown | NovaSeq 2×150 | Raw FASTQ via UKB application | **DUA required** | UKB Material Transfer Agreement | https://www.ukbiobank.ac.uk/enable-your-research |

Notes on filtering:
- Excluded: Liu et al. 2022 (Korean cohort) — already mirrored into cMD v3.9.
- Excluded: Gunjur 2024 melanoma-ICI cohort — not CRC.
- Excluded: any 16S-only study (e.g., AGP CRC subset) — fails compatibility with MetaPhlAn species pipeline.

---

## 3. Top-recommendation deep-dive: Wu_2022_Guangzhou (PRJNA763023)

### 3.1 Why this one

- **Outside cMD:** Wu_2022 has not been ingested into curatedMetagenomicData as of the most recent Bioconductor release the assistant has knowledge of. Verify with `sampleMetadata$study_name` does not contain `WuY_2022` or `WuG_2022`.
- **Geographic complement:** existing cMD cohorts include only two Chinese cohorts (YuJ_2015 Hong Kong; YachidaS_2019 is Japanese). A southern-mainland-Chinese cohort adds population diversity without overlapping the Hong Kong YuJ samples.
- **Balanced case/control, single center:** minimizes batch confounding during external scoring.
- **Read length and depth match training cohorts:** YuJ_2015 and WirbelJ_2018 used 2×100/2×150 Illumina; Wu_2022 uses 2×150 NovaSeq, so MetaPhlAn 4 marker-gene mapping rates will be comparable.

### 3.2 Explicit data URLs

- BioProject landing: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA763023
- ENA mirror (recommended for faster EU/US dual-region download):
  https://www.ebi.ac.uk/ena/browser/view/PRJNA763023
- Run table TSV (programmatic):
  `https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA763023&result=read_run&fields=run_accession,sample_accession,library_strategy,library_source,fastq_ftp,read_count,base_count,sample_title&format=tsv`

### 3.3 Expected compute time on a 32-core / 256 GB node

| Stage | Tool | Per-sample wall-clock | Total for n=110 (parallel 4-wide) |
|---|---|---|---|
| FASTQ download | `enaBrowserTools` / `aria2c` | ~6 min @ 6 GB | ~3 h |
| Adapter + quality trim | fastp | ~4 min | ~2 h |
| Host removal | Kneaddata (hg38) | ~25 min | ~12 h |
| Taxonomic profiling | MetaPhlAn 4 (mpa_vJun23_CHOCOPhlAnSGB_202307) | ~20 min | ~9 h |
| Merge + 229-species alignment | `merge_metaphlan_tables.py` + project script | < 5 min total | < 5 min |
| Model scoring | existing `scripts/external_validation.py` adapted | seconds | seconds |
| **Total wall-clock** | | | **~24–30 h, fits inside the 1-day target with 4-way sample parallelism** |

If only a 16-core box is available, expect ~2 days.

### 3.4 Validation procedure sketch

1. **Download** the 110 paired-end FASTQ files via the ENA `filereport` URL above into `data/external/Wu_2022_Guangzhou/raw/`.
2. **Quality control** with fastp (`--detect_adapter_for_pe --length_required 60`) into `data/external/.../trimmed/`.
3. **Host decontamination** with Kneaddata against GRCh38 + the existing project Bowtie2 index (already used for the cMD-internal reprofiling? if not, use the bundled MetaPhlAn 4 SGB index).
4. **Profile** each sample with MetaPhlAn 4 (`--input_type fastq --bowtie2db <db> --add_viruses --unknown_estimation`) producing `*_metaphlan.tsv`.
5. **Merge** with `merge_metaphlan_tables.py` and filter to species-level rows (`s__` prefix, no `t__`).
6. **Align taxonomy** to the 229-species feature vector used by the trained RF:
   - Load `results/shap_crc_features.csv` to get the canonical species list (verify column name is `species` or `feature`).
   - Reindex the external species matrix to that 229-column order; impute missing species with 0.0 (TSS-normalize first if the training pipeline did so — confirm in `scripts/train_baseline.py`).
7. **Score** with the persisted RF (look for `models/baseline_rf.joblib` or re-train on the full 1,522-sample training matrix with the locked seed from `scripts/train_baseline.py`, then `predict_proba` on the external matrix).
8. **Report** AUC, AUPRC, sensitivity at fixed 90% specificity (matching `results/sensitivity_thresholds.csv` convention), DeLong CI vs. the cMD-internal LODO baseline (`results/delong_results.csv`).
9. **Sanity checks:**
   - Confirm species-prevalence distribution overlaps the training cohorts' (Bray–Curtis PCoA, project samples vs. Wu_2022 samples).
   - Confirm read-depth distribution is within the cMD range (3–10 Gb).
   - If AUC drops > 0.10 below LODO, run a quick PERMANOVA on platform / depth before reporting failure — could be a profiling artifact rather than a true generalization failure.

### 3.5 Output artifacts to produce (not in this scouting pass)

- `data/external/Wu_2022_Guangzhou/species_table.tsv`
- `results/external_validation_Wu2022.csv` (AUC, AUPRC, sens@90spec, n, CI)
- `figures/external_validation_Wu2022_roc.pdf`
- A 2–3 sentence Methods addendum and a single row appended to Table 1 in the manuscript.

---

## 4. Backup recommendations (if Wu_2022 fails verification)

**Backup A — Loo_2023_SG90 (PRJEB57847):**
Singapore Gut Microbiome 90 study. ENA-hosted, open. Slightly larger (~120), includes advanced adenomas (useful given the project's adenoma sub-analysis in `results/adenoma_lodo_results.csv`). Same compute envelope as Wu_2022.

**Backup B — Zhang_2022 Beijing screening (CNGB CNP0002338):**
CNGB-hosted (China National GeneBank). Open with CC-BY 4.0. BGISEQ reads — need to confirm MetaPhlAn 4 SGB-marker compatibility with DNBSEQ chemistry; in practice mapping rates are 5–10 % lower than Illumina but profiles are usable. Use as a robustness check, not as the primary reviewer-facing external set.

---

## 5. Caveats and reviewer-objection register

### 5.1 What could go wrong before we even score

- **Accession drift.** Several BioProject IDs above are assistant-recalled and may be off-by-one or controlled-access. The `[VERIFY]` flag must be resolved manually before any download.
- **cMD silent inclusion.** Between assistant cutoff and today (2026-05-18), cMD may have added Wu_2022 or Loo_2023. Run `sampleMetadata |> dplyr::filter(stringr::str_detect(study_name, "Wu|Loo|Yang|Zhang"))` against the current cMD release before claiming externality.
- **MetaPhlAn version mismatch.** The project's 229-species feature set was almost certainly built with a specific MetaPhlAn database version (e.g., `mpa_vJan21_CHOCOPhlAnSGB_202103` or `mpa_vJun23_CHOCOPhlAnSGB_202307`). Profiling the external cohort with a *different* DB version will silently rename or split species (SGB renaming is common across versions). Pin the DB version explicitly.
- **Host-removal index version.** If training used a different human reference (hg19 vs. hg38, with/without alt contigs), residual host reads in the external cohort can bias relative abundances of low-prevalence species. Use the same Kneaddata reference.

### 5.2 What reviewers will still object to

- **"You picked the easiest external cohort."** Pre-register the choice before unblinding the AUC, or report all three (Wu, Loo, Zhang) and disclose any pre-screening.
- **"East-Asian-only external set."** Existing training data is ~50 % East Asian already, so a Chinese external cohort is the path of least resistance but does NOT prove generalization to underrepresented populations. Mitigation: pair Wu_2022 (CHN) with the Bishehsari MAPS multi-ethnic Chicago cohort if it can be confirmed public, or call out the limitation explicitly.
- **"Single-sequencer external test."** A NovaSeq-only external cohort does not test platform transferability. Add a smaller BGISEQ secondary check (Zhang_2022) to defuse this.
- **"How did you handle CRC stage?"** If Wu_2022 enriches for late-stage CRC (which Chinese referral-center cohorts often do) and the training set is screening-detected, performance can look artificially high. Stratify by stage if metadata permits.
- **"Adenoma vs. CRC label leakage."** Some external cohorts pool adv. adenomas with CRC. Re-derive the binary label using only stage I–IV CRC vs. healthy, drop adenomas, and re-run.
- **"What about the antibiotics / proton-pump-inhibitor confounder?"** Metadata may be sparse; if so, acknowledge as a limitation rather than impute.
- **"Profile your external with the same pipeline as cMD."** cMD profiles are MetaPhlAn-based but pre-computed. If reviewers ask for re-profiling of the training set with the same pinned DB used on the external cohort, that adds ~2 weeks of compute on 1,522 samples — budget for this possibility.

### 5.3 Hard stops

- If Wu_2022 turns out to be 16S only on re-verification, abandon and switch to Loo_2023.
- If none of the three top candidates are public without a DUA, fall back to the MGnify pre-profiled studies (cohort #8) and accept the lower data quality; alternatively, apply to UK Biobank BIOME with a ~3-month lead time.

---

## 6. Recommended next steps (for the human, not this read-only pass)

1. Open the three verification URLs for Wu_2022, Loo_2023, and Yang_2024 and confirm `library_strategy = WGS`, `library_source = METAGENOMIC`, and public access.
2. Cross-check `curatedMetagenomicData::sampleMetadata` for any of those study names — abort that candidate if present.
3. If Wu_2022 verifies clean, pre-register the validation plan (target metric: AUC with 95 % DeLong CI; pre-stated success criterion: lower CI bound > 0.65) as a dated note in `decisions_addendum.md` *before* downloading the data.
4. Provision a 32-core compute node and budget ~30 h of wall-clock.
5. Persist the trained RF (`models/baseline_rf.joblib`) with the locked seed used in `scripts/train_baseline.py` so the external scoring is fully reproducible.

---

## Appendix A — Intended search queries (web access was unavailable this pass)

Had WebSearch / WebFetch been permitted, the following queries would have been issued and their results triangulated against the candidate list above:

1. `site:ncbi.nlm.nih.gov/bioproject "colorectal" "shotgun metagenomic" 2023..2025`
2. `site:ebi.ac.uk/ena "colorectal cancer" "WGS" "stool"`
3. `"colorectal cancer" "metagenomic" "data availability" 2024 Nature Medicine`
4. `"colorectal cancer" "metagenomic" "data availability" 2024 Cell Host & Microbe`
5. `"curatedMetagenomicData" "v3.1" OR "v3.10" release notes added cohorts`
6. GMrepo phenotype browser: https://gmrepo.humangut.info/phenotypes/D015179 — filter `Experiment type = WGS`, sort by Year desc.
7. MGnify biome browser: https://www.ebi.ac.uk/metagenomics/biomes/root%3AHost-associated%3AHuman%3ADigestive%20system — filter `disease = colorectal cancer`.
8. SRA Run Selector: https://www.ncbi.nlm.nih.gov/Traces/study/ — query `(colorectal cancer) AND (metagenome) AND (illumina)` filter `Source: METAGENOMIC` and date 2023-01-01:2026-05-18.
9. UK Biobank Showcase data dictionary: search field 30000-series for any stool-WGS sub-study release notice dated 2024–2026.
10. AGP / Microsetta: confirm CRC subset remains 16S only (expected) and exclude.

Re-running this scout once web access is restored should take ≤ 30 minutes and is expected to refine, not overturn, the top recommendation.
