# Data Availability Statement

All sequencing data analyzed in this study are publicly available through
the `curatedMetagenomicData` Bioconductor package
(<https://bioconductor.org/packages/release/data/experiment/html/curatedMetagenomicData.html>;
Pasolli et al. 2017, *PLoS Computational Biology*). The 10 cohorts used
are: FengQ_2015, GuptaA_2019, ThomasAM_2018a, ThomasAM_2018b,
ThomasAM_2019_c, VogtmannE_2016, WirbelJ_2018, YachidaS_2019, YuJ_2015,
and ZellerG_2014. Original raw-read accession numbers for each cohort
are recorded in the `curatedMetagenomicData` metadata table.

Pre-processed feature matrices (MetaPhlAn relative abundance, HUMAnN
unstratified pathway abundance), per-sample classifier predictions, and
all derived results tables (S1-S10) are deposited alongside the analysis
code at the project repository (see Code Availability).

The HanniganGD_2017 cohort was pre-specified as excluded based on
sequencing-depth and feature-sparsity criteria assessed before
classification; the exclusion rule and its rationale are documented in
`results/decisions_addendum.md` and in the Methods.

# Code Availability Statement

All analysis code is publicly available at
<https://github.com/alejandro-publius/crc-metagenomics> under the
[License] license. The repository includes:

- `scripts/` — end-to-end analysis pipeline (preprocessing, LODO CV,
  joint species + pathway models, adenoma analyses, DeLong tests,
  bootstrap CIs, SHAP, sensitivity analyses).
- `results/` — final tables and per-sample prediction files.
- `figures/` and `manuscript/figures/` — publication figures (PNG + PDF).
- `REPRODUCING.md` — step-by-step reproduction protocol with expected
  outputs.
- `requirements.lock` — pinned Python dependency versions.
- `scripts/verify_results.py` — 38 automated checks confirming that all
  reported numbers match the deposited result files.

Exact versions of all dependencies are pinned in `requirements.lock`.
The pipeline runs end-to-end on a standard laptop (16 GB RAM,
single-machine) in approximately [N] minutes.
