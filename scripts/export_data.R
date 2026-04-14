# export_data.R
# Exports species abundance, pathway abundance, and metadata
# from curatedMetagenomicData for the CRC metagenomics project.
#
# Usage: Rscript scripts/export_data.R
# Output: data/raw/species_abundance.csv
#         data/raw/pathway_abundance.csv
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

# Keep useful columns
keep_cols <- c("sample_id", "study_name", "study_condition",
               "age", "gender", "BMI", "country",
               "sequencing_platform", "number_reads")
# Only keep columns that exist in the data
keep_cols <- keep_cols[keep_cols %in% colnames(crc_meta)]
crc_meta <- crc_meta[, keep_cols]

cat("Sample counts by condition:\n")
print(table(crc_meta$study_condition))
cat("\nSample counts by condition and cohort:\n")
print(table(crc_meta$study_condition, crc_meta$study_name))

# ── 2. Species abundance ────────────────────────────────────
# Pull relative abundance for all samples in the metadata
# This uses the returnSamples() function which queries by sample
cat("\nPulling species abundance data (this may take a few minutes)...\n")

species_se <- returnSamples(crc_meta, "relative_abundance")

# Convert to a samples x species matrix
species_mat <- as.data.frame(t(assay(species_se)))

# Add sample_id as a column
species_mat$sample_id <- rownames(species_mat)

cat("Species table dimensions:", nrow(species_mat), "samples x",
    ncol(species_mat) - 1, "species\n")

# ── 3. Pathway abundance ────────────────────────────────────
cat("\nPulling pathway abundance data (this may take a few minutes)...\n")

pathway_se <- returnSamples(crc_meta, "pathway_abundance")

# Convert to a samples x pathways matrix
pathway_mat <- as.data.frame(t(assay(pathway_se)))

# Add sample_id as a column
pathway_mat$sample_id <- rownames(pathway_mat)

cat("Pathway table dimensions:", nrow(pathway_mat), "samples x",
    ncol(pathway_mat) - 1, "pathways\n")

# ── 4. Save to CSV ──────────────────────────────────────────
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

write.csv(crc_meta, "data/raw/metadata.csv", row.names = FALSE)
write.csv(species_mat, "data/raw/species_abundance.csv", row.names = FALSE)
write.csv(pathway_mat, "data/raw/pathway_abundance.csv", row.names = FALSE)

cat("\nDone! Files saved to data/raw/:\n")
cat("  - metadata.csv\n")
cat("  - species_abundance.csv\n")
cat("  - pathway_abundance.csv\n")

# ── 5. Quick sanity checks ──────────────────────────────────
# Verify sample IDs match across all three tables
species_ids <- species_mat$sample_id
pathway_ids <- pathway_mat$sample_id
meta_ids    <- crc_meta$sample_id

cat("\nSanity checks:\n")
cat("  Samples in metadata:", length(meta_ids), "\n")
cat("  Samples in species table:", length(species_ids), "\n")
cat("  Samples in pathway table:", length(pathway_ids), "\n")
cat("  Overlap (all three):",
    length(intersect(intersect(meta_ids, species_ids), pathway_ids)), "\n")

