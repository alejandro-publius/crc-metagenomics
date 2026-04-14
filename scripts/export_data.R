# export_data.R
# Exports species abundance, pathway abundance, and metadata
# from curatedMetagenomicData for the CRC metagenomics project.
# Restricted to the 7 cohorts used in Thomas et al. (2019).
#
# Usage: Rscript scripts/export_data.R
# Output: data/raw/species_abundance.csv
#         data/raw/pathway_chunks/*.csv  (pathway data in per-cohort chunks)
#         data/raw/metadata.csv

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

# ── Filter to EXACTLY the 7 Thomas et al. (2019) cohorts ────
thomas_cohorts <- c(
  "FengQ_2015",
  "YuJ_2015",
  "VogtmannE_2016",
  "ThomasAM_2018a",
  "ThomasAM_2018b",
  "ThomasAM_2019_c",
  "ZellerG_2014"
)
crc_meta <- crc_meta[crc_meta$study_name %in% thomas_cohorts, ]

# Keep useful columns
keep_cols <- c("sample_id", "study_name", "study_condition",
               "age", "gender", "BMI", "country",
               "sequencing_platform", "number_reads")
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
for (cohort in thomas_cohorts) {
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
cat("  Expected: 762 (326 CRC + 320 control + 116 adenoma)\n")
