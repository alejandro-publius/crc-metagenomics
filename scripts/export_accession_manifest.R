#!/usr/bin/env Rscript

# Export only the public raw-read identifiers needed for the independent
# profiler pilot. This leaves the existing processed metadata untouched.
suppressPackageStartupMessages(library(curatedMetagenomicData))

data("sampleMetadata")
processed <- read.csv("data/processed/metadata_clean.csv", check.names = FALSE)

keep <- c(
  "sample_id", "study_name", "study_condition", "country",
  "number_reads", "NCBI_accession"
)
manifest <- sampleMetadata[
  sampleMetadata$sample_id %in% processed$sample_id,
  keep,
  drop = FALSE
]
manifest <- merge(
  processed[, c("sample_id", "label")],
  manifest,
  by = "sample_id",
  all.x = TRUE,
  sort = FALSE
)
manifest <- manifest[order(manifest$study_name, manifest$label,
                           manifest$sample_id), ]

dir.create("results/independent_profiler", recursive = TRUE,
           showWarnings = FALSE)
write.csv(
  manifest,
  "results/independent_profiler/accession_manifest.csv",
  row.names = FALSE,
  na = ""
)

cat("Exported", nrow(manifest), "samples;",
    sum(!is.na(manifest$NCBI_accession) & manifest$NCBI_accession != ""),
    "have raw-read accessions.\n")
