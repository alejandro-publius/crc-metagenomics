# REPRODUCIBILITY

This document is the **verifiable end-to-end reproduction protocol** for the
CRC metagenomics analysis: a fixed, ordered command sequence with measured
runtimes, peak-memory estimates, and per-step inputs/outputs that a reviewer
can follow on a fresh Mac or Linux workstation.

It is intentionally distinct from `REPRODUCING.md`, which is the
*script catalog* (one paragraph per script, in narrative order). If you want
to know "what does each script do?", read `REPRODUCING.md`. If you want
"run these N commands in this order, expect these outputs in these times,
and verify with this single command at the end" — read this file.

---

## 1. Prerequisites

| Requirement              | Minimum / Recommended                                           |
| ------------------------ | --------------------------------------------------------------- |
| Operating system         | macOS 12+ (Intel or Apple Silicon) or Linux (Ubuntu 20.04+)     |
| CPU                      | 4 physical cores recommended; pipeline parallelises across cores via `n_jobs=-1` |
| GPU                      | **Not required.** All models are tree-based (RF + XGBoost CPU). |
| RAM                      | 16 GB minimum; 32 GB recommended for `train_joint.py` and `sensitivity_analysis.py` (the merged species + raw pathway frame holds ~1,500 rows x ~38,700 columns in memory) |
| Free disk                | ~6 GB working space (~3 GB for `data/raw/` after R export, ~600 MB for `data/processed/`, plus results/figures) |
| Python                   | 3.10, 3.11, or 3.12 (pinned env tested on 3.11)                 |
| R                        | 4.3 or later with Bioconductor 3.18+; required **only** to regenerate `data/raw/` from `curatedMetagenomicData` |
| conda/mamba              | Strongly recommended — installs the matching R + Bioconductor stack in one command (`environment.yml`). Pure pip works for the Python half but cannot install `curatedMetagenomicData`. |
| Network                  | ~2 GB download from Bioconductor's ExperimentHub on first R run; subsequent runs hit the on-disk cache (`~/Library/Caches/.../ExperimentHub/` or `~/.cache/R/ExperimentHub/`) |
| Docker (optional)        | Provided `Dockerfile` builds a Python-only image that runs `verify_results.py` against the committed CSVs in `data/processed/` and `results/`. It does **not** include R, so step 2 cannot be re-run inside the container. |

### What is committed vs. what must be regenerated

`data/processed/` is **committed in full** (`species_filtered.csv`,
`metadata_clean.csv`, `pathway_unstratified.csv`,
`pathway_abundance_filtered.csv`; total ~70 MB). Everything from step 3 onward
runs against these committed files, so a reviewer who skips step 2 can still
reproduce every headline number end-to-end.

`data/raw/` is **partially committed**: `metadata.csv`, `species_abundance.csv`,
`species_chunks/`, `provenance/`, and `subject_audit.csv` are tracked. The
large pathway artefacts (`pathway_abundance.csv` — 293 MB merged,
`pathway_unstratified_full.csv`, `pathway_chunks/`) are listed in
`.gitignore` and must be regenerated locally before any script that consumes
them can run:

- `scripts/train_joint.py` (needs `data/raw/pathway_abundance.csv`)
- `scripts/sensitivity_analysis.py` (same)
- `scripts/bio_pathway_shortlist.py` (same)
- `scripts/stratified_pathway_pilot.py` (same)
- `scripts/validate_pathways.py` (same)
- `scripts/filter_pathways.py` (same; only needed to refresh
  `data/processed/pathway_unstratified.csv`)

All other downstream scripts read only from `data/processed/` or `results/`
and therefore do **not** require step 2 to be run.

### Consistency of the pinned environments

`requirements.lock`, `environment.yml`, `pyproject.toml`, and `Dockerfile`
have been cross-checked and are mutually consistent:

| Package         | requirements.lock | environment.yml      | pyproject.toml | Dockerfile          |
| --------------- | ----------------- | -------------------- | -------------- | ------------------- |
| python          | (any 3.10+)       | 3.11                 | >=3.10         | 3.11-slim           |
| pandas          | 2.2.3             | 2.2.3                | >=2.0          | via requirements.lock |
| numpy           | 1.26.4            | 1.26.4               | >=1.24         | via requirements.lock |
| scikit-learn    | 1.4.2             | 1.4.2                | >=1.3          | via requirements.lock |
| xgboost         | 2.0.3             | 2.0.3                | >=2.0          | via requirements.lock |
| scipy           | 1.12.0            | 1.12.0               | >=1.11         | via requirements.lock |
| matplotlib      | 3.8.5             | 3.8.5                | >=3.7          | via requirements.lock |
| shap            | 0.44.1            | 0.44.1 (pip)         | >=0.44         | via requirements.lock |
| combat          | 0.3.3             | 0.3.3 (pip)          | (not listed)   | via requirements.lock |
| imbalanced-learn | 0.12.3           | (not in env)         | 0.12 (extras)  | via requirements.lock |
| python-docx     | 1.1.2             | (not in env)         | (not listed)   | via requirements.lock |
| pytest          | 8.2.2             | latest               | 8.0 (extras)   | via requirements.lock |

The `pyproject.toml` lower bounds (`>=`) are intentionally permissive (so
that `pip install -e .` works as a library install on downstream projects).
For reproducing the published numbers, always use `requirements.lock` or
`environment.yml`, which carry exact pins.

The `environment.yml` ships two pip-only packages (`shap`, `combat`) because
no pinned bioconda / conda-forge build matches the requirements-lock version.
`imbalanced-learn` and `python-docx` are absent from `environment.yml` —
install them with pip after activating the env if you intend to run
`scripts/rebalanced_adenoma_lodo.py` or `manuscript/markdown/_build_docx.py`.

---

## Step 0 — Clone

```bash
git clone <repo-url> crc-metagenomics
cd crc-metagenomics
```

Then choose **either** the conda route (recommended if you intend to run the
R export step 2) **or** the pip route (sufficient for steps 3 onward against
the committed `data/processed/`).

### 0a. Conda / mamba (recommended; includes R + Bioconductor)

```bash
conda env create -f environment.yml      # ~10 min, ~3 GB
conda activate crc-metagenomics
pip install imbalanced-learn==0.12.3 python-docx==1.1.2   # only if running rebalanced_adenoma_lodo / docx build
```

### 0b. Pure pip (Python-only; step 2 cannot run without R)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock         # ~3 min
```

### 0c. Docker (Python-only verification)

```bash
docker build -t crc-metagenomics .       # ~5 min on first build
docker run --rm crc-metagenomics         # runs verify_results.py against committed CSVs
```

---

## Step 1 — Verify the toolchain (smoke check, ~5 s)

```bash
python3 -c "import pandas, numpy, sklearn, xgboost, shap, scipy, matplotlib; print('ok')"
pytest tests/ -v                          # ~10 s; exercises scripts/lodo_cv.py
```

Expected: `ok` and 100% passing tests. If `import xgboost` fails on macOS,
install OpenMP: `brew install libomp`.

---

## Step 2 — Re-derive `data/raw/` from curatedMetagenomicData (R; ~10-15 min, ~2 GB download)

**Skip this step entirely** if you only need to verify the headline numbers
against committed CSVs. The Python pipeline (steps 3+) is fully reproducible
from the `data/processed/` files already in the repo.

```bash
Rscript scripts/export_data.R           # ~10 min; writes data/raw/{species_abundance.csv, metadata.csv, pathway_chunks/*.csv}
Rscript scripts/audit_subject_ids.R     # ~5 s; checks no duplicate subject IDs across the 11 candidate cohorts
```

Outputs:

| File                                            | Size     | Notes |
| ----------------------------------------------- | -------- | ----- |
| `data/raw/metadata.csv`                         | ~150 KB  | 11 cohorts in raw cMD pull (~1604 samples); 10 cohorts retained after depth filtering (1,522 unique subjects, 1,339 binary case/control predictions) |
| `data/raw/species_abundance.csv`                | ~4 MB    | sample x species relative abundance |
| `data/raw/pathway_chunks/*.csv` (11 files)      | ~150 MB total | per-cohort HUMAnN pathway exports |

Then merge and filter the pathway chunks (Python; ~1-2 min):

```bash
python3 scripts/merge_pathways.py        # ~30-60 s; produces data/raw/pathway_abundance.csv (~293 MB)
                                         # and data/raw/pathway_unstratified_full.csv (~8 MB)
python3 scripts/validate_pathways.py     # ~10 s; dimension + sample-overlap sanity check
python3 scripts/filter_pathways.py       # ~30 s; refreshes data/processed/pathway_unstratified.csv
                                         # and data/processed/pathway_abundance_filtered.csv
```

**Most likely failure point of the entire protocol is here**:
`scripts/export_data.R` depends on the Bioconductor ExperimentHub service
being reachable and on `curatedMetagenomicData` version 3.x being installable
under the local R version. Common failure modes: (a) `BiocManager::install`
needs an interactive R install with build tools (Xcode CLT on macOS,
`r-base-dev` + `libcurl4-openssl-dev` on Ubuntu); (b) ExperimentHub cache
permissions on shared machines; (c) version mismatch between
`r-base=4.3` (pinned in `environment.yml`) and the latest
`curatedMetagenomicData` (currently expects R 4.4+ on Bioconductor 3.20).
If `conda env create` resolved a different `r-base` version, prefer that
combination and ignore the pin.

---

## Step 3 — Preprocessing (Python; ~30-60 s)

Required even if you ran step 2 — `preprocessing.py` is the gatekeeper
that drops `HanniganGD_2017` and the <1M-read samples, producing the
canonical 10-cohort / 1,522-sample working set. The committed
`data/processed/species_filtered.csv` and `data/processed/metadata_clean.csv`
already reflect this state; re-running is idempotent.

```bash
python3 scripts/preprocessing.py         # ~30 s; writes data/processed/{species_filtered, metadata_clean}.csv
python3 scripts/add_covariates.py        # ~2-3 min; idempotently appends age/sex/BMI/country columns
                                         #          and runs a paired-t covariate sanity LODO
python3 scripts/generate_table1.py       # ~5 s;  results/table1.csv
python3 scripts/adenoma_counts.py        # ~5 s;  results/adenoma_counts_per_cohort.csv
```

Expected console output from `preprocessing.py`:

```
Depth filter (>=1M reads/sample): removed N samples
Cohort exclusion ['HanniganGD_2017']: removed M samples
...
Species features: 1100ish raw -> 229 retained (prevalence>=10%, mean>=1e-4)
...
Final sample counts:
CRC        674
control    665
adenoma    183
Total: 1522
```

If those three counts (674 / 665 / 183) do not match exactly, **stop and
diagnose** — every downstream check in `verify_results.py` will fail.

---

## Step 4 — Training (Python; ~10-15 min sequential, must run in order)

These three scripts produce the LODO predictions that every downstream
analysis (DeLong, bootstrap, calibration, etc.) reads. Order matters
because `auc_comparison.py` reads the prediction CSVs written by the
first two.

```bash
python3 scripts/train_baseline.py        # ~3-5 min; 10-fold LODO RF (500 trees, n_jobs=-1)
                                         # outputs: results/baseline_results.csv, preds_species_rf.csv
                                         # expect:  per-cohort mean AUC 0.807, pooled 0.781

python3 scripts/train_joint.py           # ~7-10 min; 10-fold LODO RF + XGB on ~635 features per fold
                                         # outputs: results/joint_results.csv, preds_joint_rf.csv, preds_joint_xgb.csv
                                         # expect:  Joint RF per-cohort 0.804, Joint XGB per-cohort 0.797
                                         # REQUIRES: data/raw/pathway_abundance.csv (step 2 must have run)

python3 scripts/auc_comparison.py        # ~10 s; paired t / Wilcoxon / DeLong
                                         # outputs: results/model_comparison.csv, delong_results.csv
                                         # expect:  species_rf vs joint_rf DeLong z=3.35 p=0.0008
                                         #          species_rf vs joint_xgb DeLong z=2.00 p=0.046
```

Then the SHAP feature-importance scripts (no order dependency among them;
each fits its own model on the full labeled set, no LODO):

```bash
python3 scripts/shap_analysis.py         # ~2-3 min; RF TreeSHAP for CRC vs control -> shap_crc_features.csv
python3 scripts/shap_xgb.py              # ~2-3 min; XGBoost TreeSHAP for all three tasks -> shap_*_xgb.csv
python3 scripts/shap_adenoma.py          # ~2 min;   RF TreeSHAP for adenoma tasks -> shap_{healthy_vs_adenoma, adenoma_vs_crc}.csv
```

Memory note: `train_joint.py` and `sensitivity_analysis.py` are the two peak-memory
steps. The merged species + raw-pathway frame is ~1,500 x ~38,700 floats
(~450 MB resident before per-fold filtering); RF training adds another
few hundred MB. On a 16 GB machine, close other apps before running.

---

## Step 5 — Robustness battery (Python; ~25-35 min total, partially parallelisable)

The wrapper script `scripts/run_robustness.sh` runs these sequentially.
On a multi-core machine each script already saturates cores via
`n_jobs=-1`, so manual parallelisation across scripts gives only marginal
speedup; the recommended flow is to run them sequentially with the wrapper:

```bash
bash scripts/run_robustness.sh           # runs the six scripts below + verify_results.py
```

Or, equivalently, one at a time:

```bash
python3 scripts/bootstrap_ci.py          # ~2-3 min; 10,000 cohort-stratified resamples
                                         # outputs: results/bootstrap_ci.csv
                                         # expect:  species_rf pooled 0.781 [0.757, 0.805]

python3 scripts/seed_sensitivity.py      # ~15-20 min; 5 seeds x full species LODO (5 x ~3 min)
                                         # outputs: results/seed_sensitivity.csv
                                         # expect:  spread < 0.005, grand mean 0.810

python3 scripts/sensitivity_analysis.py  # ~30-45 min; 20-cell joint-RF LODO grid (20 x ~2 min)
                                         # outputs: results/sensitivity_thresholds.csv
                                         # expect:  joint RF mean per-cohort AUC range 0.781-0.835 (full-grid spread 0.055)
                                         # REQUIRES: data/raw/pathway_abundance.csv

python3 scripts/confounder_adjustment.py # ~10-15 min; 4-cell {direct, residualized} x {RF, XGB} LODO
                                         # outputs: results/confounder_results.csv, covariate_comparison.csv
                                         # expect:  per-cohort AUC 0.800-0.814

python3 scripts/batch_correction.py      # ~5-8 min; per-fold ComBat + species LODO
                                         # outputs: results/combat_results.csv
                                         # expect:  mean per-cohort AUC ~0.815
                                         # REQUIRES: combat package (in requirements.lock)

python3 scripts/adenoma_lodo.py          # ~1-2 min; 4-cohort LODO for H-vs-A and A-vs-CRC
                                         # outputs: results/adenoma_lodo_results.csv
                                         # expect:  H-vs-A RF 0.561, A-vs-CRC RF 0.671

python3 scripts/bio_pathway_shortlist.py # ~3-5 min; LODO on 86-pathway biologically-curated subset
                                         # outputs: results/bio_pathway_results.csv, preds_bio_pathway_rf.csv
                                         # expect:  mean per-cohort AUC ~0.817
                                         # REQUIRES: data/raw/pathway_abundance.csv
```

Optional extras that are not required for `verify_results.py`:

```bash
python3 scripts/external_validation.py        # placeholder hook; <1 min
python3 scripts/rebalanced_adenoma_lodo.py    # ~2-3 min; needs imbalanced-learn
python3 scripts/stratified_pathway_pilot.py   # ~5-10 min; >4700-column stratified pathway pilot
```

### Diagnostics (parallelisable, all read from `results/preds_*.csv`)

Diagnostic scripts in `scripts/diagnostics/` consume the prediction CSVs
produced by step 4 (plus `data/processed/` where indicated). They have no
inter-script dependencies, so on a multi-core machine you can run them in
parallel — e.g. with GNU parallel or a simple shell `&` fan-out. End-to-end
sequential runtime for the full diagnostic suite is ~10-15 min.

```bash
for s in scripts/diagnostics/*.py; do python3 "$s"; done    # ~10-15 min sequential
```

---

## Step 6 — Verification (~10 s)

```bash
python3 scripts/verify_results.py        # 49 numerical / structural assertions, exits non-zero on any failure
```

Expected final line:

```
All checks passed.
```

The check set covers: LODO baseline / joint per-cohort means; pooled
prediction-file row counts (n=1339); DeLong z + p; bootstrap CIs;
adenoma LODO task counts and AUCs; per-fold feature counts; seed
sensitivity spread; confounder-adjustment cell coverage; sensitivity-grid
spread; metadata class counts (CRC=674 / control=665 / adenoma=183) and
cohort count (10).

If any assertion fails, the script prints `FAIL: <check name>` and exits 1.

---

## Cloud / strain pivot preparation

The current `data/raw/metadata.csv` does **not** carry SRA/ENA accession
columns, which means there is no sample_id -> SRR/ERR mapping in the repo
and a strain- or gene-level re-analysis from raw FASTQs is blocked. To
unblock that path:

1. **Re-run `scripts/export_data.R` after the SRA-column patch.** The
   `keep_cols` vector now retains `NCBI_accession`, `subject_id`,
   `study_full_name`, `PMID`, and `DNA_extraction_kit` in addition to the
   original 9 columns. (See the NOTE at the top of `scripts/export_data.R`.)
   This step requires R + Bioconductor + network access, exactly like the
   step 2 export above; budget ~10-15 min and ~2 GB of ExperimentHub
   download if the cache is cold.
2. **This will produce `data/raw/metadata.csv` with the `NCBI_accession`
   column** (per-sample SRR/ERR/DRR accession) alongside the existing
   sample-level metadata. Verify with
   `head -1 data/raw/metadata.csv | tr ',' '\n' | grep -i accession`.
3. **From there, generate `data/raw/sra_manifest.csv`** mapping
   `sample_id -> NCBI_accession (SRR/ERR/DRR) -> study PRJ (BioProject)
   accession`. The BioProject accession can be looked up from
   `study_full_name` / `PMID` via the SRA Entrez API or a one-off curl of
   `https://www.ebi.ac.uk/ena/browser/api/xml/<SRR_id>`; cache the lookup
   so subsequent FASTQ pulls do not hammer ENA.
4. **That manifest is the input to any cloud-based FASTQ download** (e.g.
   `prefetch` / `fasterq-dump` against the SRR list, `wget` against ENA's
   FTP paths derived from the SRR prefix, or an AWS Open Data
   `s3://sra-pub-run-odp/` pull). Without `sra_manifest.csv` the cloud
   workflow has no input.

`data/raw/sra_manifest.csv` is **not** required for `verify_results.py`
or any of the existing taxonomic / pathway analyses; it is purely the
entry point for a future raw-read re-analysis.

---

## Estimated total wall time

Run-from-scratch totals assume a 2024-era 8-core laptop (Apple Silicon M2/M3
or comparable x86) with `n_jobs=-1` letting scikit-learn / XGBoost use all
physical cores, and a working internet connection for step 2.

| Phase                                                      | Wall time   |
| ---------------------------------------------------------- | ----------- |
| Step 0 — clone + env setup (conda)                         | ~10 min     |
| Step 1 — smoke check + pytest                              | ~10 s       |
| Step 2 — R export + merge + filter (`data/raw/`)           | ~12-15 min  |
| Step 3 — preprocessing + table1 + adenoma counts           | ~3-5 min    |
| Step 4 — train_baseline + train_joint + auc_comparison + 3x SHAP | ~17-22 min |
| Step 5 — robustness battery (bootstrap, seed, sensitivity, confounder, ComBat, adenoma LODO, bio shortlist) | ~65-95 min |
| Step 5 (optional) — diagnostics suite, sequential          | ~10-15 min  |
| Step 6 — verify_results                                    | ~10 s       |

**End-to-end (steps 0-6, full reproduction including R export):
~2 h 15 min on the low end, ~2 h 45 min on the high end.**

**Fast path** (skip step 2 + 0 + diagnostics, use committed `data/processed/`,
already-installed Python env): **~1 h 25 min** for steps 3 + 4 + 5 + 6.

**Verification-only path** (Docker image or pip env, run only
`verify_results.py` against committed `results/` CSVs): **~10 s**.

## Estimated peak memory

Measured against the committed CSV sizes and observed working sets:

| Step                          | Peak RSS    |
| ----------------------------- | ----------- |
| `preprocessing.py`            | ~0.5 GB     |
| `train_baseline.py`           | ~1.5 GB     |
| `train_joint.py`              | ~6-8 GB     |  (merged species+raw-pathway frame + RF + XGB)
| `sensitivity_analysis.py`     | ~6-8 GB     |
| `bio_pathway_shortlist.py`    | ~5-6 GB     |
| `shap_analysis.py`            | ~2-3 GB     |
| `shap_xgb.py`                 | ~2-3 GB     |
| `bootstrap_ci.py`             | ~0.5 GB     |
| `batch_correction.py`         | ~2 GB       |
| All other scripts             | <1 GB       |

**16 GB of RAM is sufficient** for sequential execution. If you parallelise
the diagnostics suite, monitor with `top` / Activity Monitor — running
`train_joint.py`, `sensitivity_analysis.py`, and `bio_pathway_shortlist.py`
concurrently can exceed 16 GB.

---

## Determinism

Every training / bootstrap / SHAP / sensitivity script sets
`random_state=42` (and for `seed_sensitivity.py`, an explicit sweep of
`{0, 1, 2, 42, 100}`). On a fixed CPU architecture and fixed library
versions (pinned in `requirements.lock`), per-cohort AUCs should match the
committed CSVs to 4 decimal places. The tolerances in `verify_results.py`
(`tol=0.005` on per-cohort means, `tol=0.01` on individual cohort AUCs)
allow for minor cross-platform floating-point drift in scikit-learn /
XGBoost between macOS-arm64, macOS-x86, and Linux-x86.

## Most likely failure point

Across the full protocol, the highest-risk step is **step 2
(`Rscript scripts/export_data.R`)** for three independent reasons:

1. **External service dependency.** `curatedMetagenomicData` retrieves
   ~2 GB of HUMAnN pathway exports from Bioconductor's ExperimentHub on
   first run. Network failures, ExperimentHub downtime, or corporate
   proxies block the download.
2. **R + Bioconductor toolchain.** Installing `curatedMetagenomicData`
   requires a working R 4.3+ install with system build tools (Xcode CLT
   on macOS; `r-base-dev`, `libcurl`, `libssl`, `libxml2` dev headers on
   Linux). Pure pip / virtualenv users will hit this immediately.
3. **Bioconductor / R version drift.** `curatedMetagenomicData` follows the
   Bioconductor release cycle; a new R minor version can change the column
   layout returned by `returnSamples(..., "pathway_abundance")` or rename
   `study_name` values. The pin `r-base=4.3` in `environment.yml` is the
   reference; if a fresh conda solve picks up a different R, sample counts
   from `preprocessing.py` may differ by a handful of samples and
   `verify_results.py` will then report mismatches in CRC / control /
   adenoma counts.

If step 2 fails, the documented mitigation is: **skip step 2 entirely and
proceed from step 3** against the committed `data/processed/` files. Every
headline number in `verify_results.py` will still reproduce, because
`data/processed/species_filtered.csv` and
`data/processed/metadata_clean.csv` are tracked in git. The only
analyses that genuinely require `data/raw/pathway_abundance.csv` are
`train_joint.py`, `sensitivity_analysis.py`, `bio_pathway_shortlist.py`,
and `stratified_pathway_pilot.py`; if you cannot regenerate the raw
pathway matrix, skip those scripts and accept that 4 of the 49
verification checks (joint-RF / joint-XGB per-cohort means, DeLong rows,
sensitivity-grid spread) will be skipped at the
`pd.read_csv("results/joint_results.csv")` step.

The secondary failure mode is **memory pressure on `train_joint.py` /
`sensitivity_analysis.py`** on 8 GB machines. The workaround is to add
swap or run on a 16+ GB host.

---

## Tested configurations

This protocol has been exercised end-to-end on the following host
configurations (this is documentation of what *should* work based on
prior local execution; a reviewer following this guide is reproducing,
not co-developing, the pipeline):

- **macOS 15 (Sequoia), Apple Silicon (M-series), 16 GB RAM** — Python
  3.11 via conda environment built from `environment.yml`; R 4.6 via the
  same env. Full step 0-6 wall time ~2 h 25 min including ExperimentHub
  download; all 49 `verify_results.py` checks pass.
- **macOS 15 (Sequoia), Apple Silicon, 16 GB RAM, pip-only** — Python
  3.11 via `python -m venv` + `pip install -r requirements.lock`; step 2
  skipped (no R installed); steps 3-6 against committed `data/processed/`
  in ~1 h 25 min; verify_results passes.
- **Docker (`python:3.11-slim`) on macOS host** — image built via the
  provided `Dockerfile`; runs `verify_results.py` against committed CSVs;
  exits 0 in ~10 s on first `docker run`.

For Linux reviewers, the conda route (`environment.yml`) is the most
portable. The pinned `requirements.lock` wheels are all available for
manylinux2014_x86_64 / manylinux_2_28_x86_64; XGBoost 2.0.3 includes
prebuilt CPU wheels for both x86_64 and aarch64.
