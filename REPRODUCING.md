# Reproducing the Analyses

This document explains how to reproduce every result and figure in the paper. Following these steps from a clean machine should give you the exact AUCs, SHAP values, and figures reported in the manuscript.

## Prerequisites

You will need:

- **Python 3.11 or 3.12** with pip
- **R 4.0+** and **R Studio**
- **Git** for cloning the repository
- About **5 GB** of free disk space for raw and processed data
- Internet connection for downloading data from curatedMetagenomicData

The pipeline runs on macOS, Linux, and Windows. All commands below assume macOS or Linux. On Windows, replace `python3` with `python` and adjust paths accordingly.

## Installation

### Clone the repository

```bash
git clone https://github.com/alejandro-publius/crc-metagenomics.git
cd crc-metagenomics
```

### Install Python dependencies

```bash
pip3 install -r requirements.txt
```

If you get an "externally-managed-environment" error on macOS with Python 3.13, add the `--break-system-packages` flag:

```bash
pip3 install -r requirements.txt --break-system-packages
```

### Install R dependencies

Open R Studio and run:

```r
install.packages("BiocManager")
BiocManager::install("curatedMetagenomicData")
```

This takes 10-20 minutes the first time.

## Running the Pipeline

The pipeline runs in two stages: data export from R, then analyses in Python. Run the steps in order. Each step produces specific output files that the next step depends on.

### Stage 1: Data export (R)

#### Step 1.1: Export species, pathway, and metadata

In R Studio, open `scripts/export_data.R` and click Source. Or from the terminal:

```bash
Rscript scripts/export_data.R
```

**Time:** 20-60 minutes (depends on internet speed)

**Produces:**
- `data/raw/metadata.csv` — sample metadata for 762 samples across 7 cohorts
- `data/raw/species_abundance.csv` — species relative abundances
- `data/raw/pathway_chunks/*.csv` — pathway abundances split per cohort (to keep file sizes manageable)

### Stage 2: Analyses (Python)

Run all Python scripts from the project root directory:

```bash
cd /path/to/crc-metagenomics
```

#### Step 2.1: Preprocess species data

```bash
python3 scripts/preprocessing.py
```

**Time:** 1-2 minutes

**Produces:**
- `data/processed/species_filtered.csv` — filtered, normalized species table
- `data/processed/metadata_clean.csv` — cleaned metadata with binary labels

#### Step 2.2: Preprocess pathway data

```bash
python3 scripts/preprocess_pathways.py
```

**Time:** 1-2 minutes

**Produces:**
- `data/processed/pathway_filtered.csv` — filtered, normalized pathway table

#### Step 2.3: Generate Table 1 (cohort demographics)

```bash
python3 scripts/generate_table1.py
```

**Time:** Under 1 minute

**Produces:**
- `results/table1.csv` — Table 1 of the manuscript

#### Step 2.4: Train species-only baseline (LODO)

```bash
python3 scripts/train_baseline.py
```

**Time:** 5 minutes

**Produces:**
- `results/baseline_results.csv` — per-cohort LODO AUC for species-only Random Forest

**Expected result:** Mean AUC ~0.803

#### Step 2.5: Train joint species + pathway models

```bash
python3 scripts/train_joint.py
```

**Time:** 8-10 minutes

**Produces:**
- `results/joint_xgb_results.csv` — per-cohort LODO AUC for joint XGBoost

**Expected result:** Joint RF mean AUC ~0.779, Joint XGB mean AUC ~0.783

#### Step 2.6: Train adenoma classifiers

```bash
python3 scripts/train_adenoma.py
```

**Time:** 5 minutes

**Produces:**
- `results/adenoma_results.csv` — 5-fold CV AUCs for healthy-vs-adenoma and adenoma-vs-CRC

**Expected result:** Healthy-vs-adenoma RF AUC ~0.708, Adenoma-vs-CRC RF AUC ~0.793

#### Step 2.7: SHAP analysis on CRC vs control

```bash
python3 scripts/shap_analysis.py
```

**Time:** 5-10 minutes

**Produces:**
- `results/shap_crc_features.csv` — ranked feature importances for CRC classification

**Expected top features:** Parvimonas micra, Peptostreptococcus stomatis, Gemella morbillorum, Fusobacterium nucleatum

#### Step 2.8: SHAP analysis on adenoma transitions

```bash
python3 scripts/shap_adenoma.py
```

**Time:** 5-10 minutes

**Produces:**
- `results/shap_healthy_vs_adenoma.csv` — features driving healthy-to-adenoma transition
- `results/shap_adenoma_vs_crc.csv` — features driving adenoma-to-CRC transition

**Expected:** Healthy-to-adenoma SHAP highlights pantothenate biosynthesis and commensal depletion. Adenoma-to-CRC SHAP highlights oral pathobionts and arginine biosynthesis.

#### Step 2.9: Confounder adjustment

```bash
python3 scripts/add_covariates.py
```

**Time:** 10 minutes

**Produces:**
- `results/covariate_comparison.csv` — per-cohort comparison of species-only vs species+covariates

**Expected:** Mean AUC changes minimally (0.803 vs 0.807, p=0.306, not significant)

#### Step 2.10: External validation

```bash
python3 scripts/external_validation.py
```

**Time:** 5 minutes

**Produces:**
- `results/external_validation.csv` — per-cohort AUC on held-out cohorts

**Expected:** Combined external AUC ~0.833 on YuJ_2015 + ZellerG_2014

#### Step 2.11: Generate figures

```bash
python3 scripts/generate_figures.py
```

**Time:** 1 minute

**Produces:**
- `figures/fig1_lodo_auc.png` — LODO performance bar chart
- `figures/fig2_shap_crc.png` — top SHAP features for CRC classification
- `figures/fig3_adenoma.png` — adenoma classification AUCs

#### Step 2.12: Statistical comparison of models

```bash
python3 scripts/auc_comparison.py
```

**Time:** Under 1 minute

**Produces:** Console output with paired t-test and Wilcoxon results comparing baseline RF, joint RF, and joint XGBoost.

## Expected Outputs

After running the full pipeline, your `results/` directory should contain:

```
results/
├── adenoma_results.csv
├── baseline_results.csv
├── covariate_comparison.csv
├── external_validation.csv
├── joint_xgb_results.csv
├── shap_adenoma_vs_crc.csv
├── shap_crc_features.csv
├── shap_healthy_vs_adenoma.csv
└── table1.csv
```

And your `figures/` directory should contain:

```
figures/
├── fig1_lodo_auc.png
├── fig2_shap_crc.png
└── fig3_adenoma.png
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'sklearn'`

You haven't installed the Python dependencies yet. Run:

```bash
pip3 install -r requirements.txt
```

### `FileNotFoundError: data/raw/species_abundance.csv`

You haven't run the R export script yet. Go back to Step 1.1.

### `KeyError: 'gender'` or `KeyError: 'BMI'`

The metadata column names don't match what the script expects. Check `data/processed/metadata_clean.csv` for the actual column names and edit the affected script accordingly.

### R Studio cannot install `curatedMetagenomicData`

Make sure you're using a recent version of R (4.0 or higher) and that BiocManager is installed first. If problems persist, check the curatedMetagenomicData documentation at https://bioconductor.org/packages/curatedMetagenomicData/.

### Random Forest results don't exactly match the paper

Random Forest with `random_state=42` should be deterministic, but small numerical differences can arise from different scikit-learn or numpy versions. AUCs within ±0.005 of the reported values are expected.

### XGBoost installation fails on macOS

Run:

```bash
brew install libomp
pip3 install xgboost
```

## Computational Requirements

The full pipeline runs on a standard laptop in under an hour total. The most computationally intensive steps are SHAP analyses (5-10 minutes each) and joint model training (8-10 minutes). Memory usage peaks around 4 GB during SHAP computation. No GPU is required.

## Data Availability

All raw sequencing data is publicly available through the European Nucleotide Archive (ENA) under accessions associated with the original cohort publications. The curatedMetagenomicData R package handles downloading these files automatically. Original cohort accession numbers are listed in the manuscript.

## Citation

If you use this pipeline or data, please cite our manuscript and the original cohort publications referenced in the paper.

## Contact

For questions about reproducing these analyses, please open an issue on the GitHub repository.
