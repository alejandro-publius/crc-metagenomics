# export_data.R
# Exports species abundance, pathway abundance, and metadata
# from curatedMetagenomicData for the CRC metagenomics project.
# Covers 11 cohorts: original 7 (Thomas et al. 2019) + YachidaS_2019,
# WirbelJ_2018, HanniganGD_2017, GuptaA_2019.
#
# Usage: Rscript scripts/export_data.R
# Output: data/raw/species_abundance.csv
#         data/raw/pathway_chunks/*.csv  (pathway data in per-cohort chunks)
#         data/raw/metadata.csv
#
# NOTE (2026-05-18): The keep_cols vector was extended to include SRA accession
# (NCBI_accession), subject_id, study_full_name, PMID, and DNA_extraction_kit
# to unblock raw-FASTQ retrieval and subject-level auditing. After these edits,
# rerun this script to regenerate data/raw/metadata.csv with the SRA accession
# columns. Until then, raw FASTQ pull from SRA/ENA is blocked because no
# sample_id -> SRR/ERR mapping is available anywhere in the repo.

# ── Install / load packages ──────────────────────────────────
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!require("curatedMetagenomicData", quietly = TRUE))
  BiocManager::install("curatedMetagenomicData")

library(curatedMetagenomicData)

# ── 1. Metadata ──────────────────────────────────────────────
data("sampleMetadata")

# Keep CRC, adenoma, and control samples
crc_meta <- sampleMetadata[sampleMetadata$study_condition %in%
                             c("CRC", "adenoma", "control"), ]

# ── Filter to CRC cohorts: original 7 (Thomas et al. 2019) + 4 additional ──
study_cohorts <- c(
  "FengQ_2015",
  "YuJ_2015",
  "VogtmannE_2016",
  "ThomasAM_2018a",
  "ThomasAM_2018b",
  "ThomasAM_2019_c",
  "ZellerG_2014",
  "YachidaS_2019",
  "WirbelJ_2018",
  "HanniganGD_2017",
  "GuptaA_2019"
)
crc_meta <- crc_meta[crc_meta$study_name %in% study_cohorts, ]

# Keep useful columns
# NCBI_accession, subject_id, study_full_name, PMID, and DNA_extraction_kit
# are retained so that (a) we can pull raw FASTQs from SRA/ENA for any
# strain- or gene-level re-analysis (NCBI_accession is the per-sample SRR/ERR
# accession; without it, sample_id -> raw-read mapping is impossible), and
# (b) we can audit subject-vs-sample uniqueness per cohort -- this matters
# especially for YachidaS_2019, where current subject-level coverage sits at
# only ~50% and we cannot tell from study_name + sample_id alone whether
# duplicate subjects are being counted as independent samples. PMID and
# study_full_name make the per-sample provenance audit self-contained, and
# DNA_extraction_kit is needed for batch/extraction-protocol confounder checks.
keep_cols <- c("sample_id", "study_name", "study_condition",
               "age", "gender", "BMI", "country",
               "sequencing_platform", "number_reads",
               "NCBI_accession", "subject_id", "study_full_name",
               "PMID", "DNA_extraction_kit")
keep_cols <- keep_cols[keep_cols %in% colnames(crc_meta)]
crc_meta <- crc_meta[, keep_cols]

cat("Sample counts by condition:\n")
print(table(crc_meta$study_condition))
cat("\nSample counts by condition and cohort:\n")
print(table(crc_meta$study_condition, crc_meta$study_name))
cat("\nTotal samples:", nrow(crc_meta), "\n")

# ── 2. Species abundance ────────────────────────────────────
cat("\nPulling species abundance data (this may take a few minutes)...\n")

species_se <- returnSamples(crc_meta, "relative_abundance")
species_mat <- as.data.frame(t(assay(species_se)))
species_mat$sample_id <- rownames(species_mat)

cat("Species table dimensions:", nrow(species_mat), "samples x",
    ncol(species_mat) - 1, "species\n")

# ── 3. Pathway abundance (exported per-cohort to avoid huge single file) ──
cat("\nPulling pathway abundance data (this may take a few minutes)...\n")

dir.create("data/raw/pathway_chunks", recursive = TRUE, showWarnings = FALSE)
for (cohort in study_cohorts) {
  cat("  Processing", cohort, "...\n")
  cohort_meta <- crc_meta[crc_meta$study_name == cohort, ]
  pw_se <- returnSamples(cohort_meta, "pathway_abundance")
  pw_mat <- as.data.frame(t(assay(pw_se)))
  pw_mat$sample_id <- rownames(pw_mat)
  write.csv(pw_mat, paste0("data/raw/pathway_chunks/", cohort, ".csv"),
            row.names = FALSE)
}

# ── 4. Save to CSV ──────────────────────────────────────────
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

write.csv(crc_meta, "data/raw/metadata.csv", row.names = FALSE)
write.csv(species_mat, "data/raw/species_abundance.csv", row.names = FALSE)

cat("\nDone! Files saved to data/raw/:\n")
cat("  - metadata.csv\n")
cat("  - species_abundance.csv\n")
cat("  - pathway_chunks/*.csv (one per cohort)\n")

# ── 5. Sanity checks ────────────────────────────────────────
cat("\nSanity checks:\n")
cat("  Total samples in metadata:", nrow(crc_meta), "\n")
cat("  Total samples in species table:", nrow(species_mat), "\n")
cat("  Expected: ~1604 (701 CRC + 694 control + 209 adenoma across 11 cohorts)\n")
