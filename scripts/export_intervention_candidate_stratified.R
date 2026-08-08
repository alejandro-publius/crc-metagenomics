# Export taxon-resolved HUMAnN rows for internally nominated gene families.
#
# This resolves whether an unstratified gene-family signal is primarily carried
# by one taxon or distributed across organisms. Candidate selection has already
# occurred; the export neither ranks candidates nor uses labels.

suppressMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(Matrix)
})

OUR_COHORTS <- c(
  "FengQ_2015", "GuptaA_2019", "ThomasAM_2018a", "ThomasAM_2018b",
  "ThomasAM_2019_c", "VogtmannE_2016", "WirbelJ_2018", "YachidaS_2019",
  "YuJ_2015", "ZellerG_2014"
)

candidates <- read.csv(
  "results/intervention_readiness/discovery_candidate_summary.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
candidate_ids <- unique(candidates$gene_id[candidates$internal_nomination])
if (length(candidate_ids) == 0) {
  stop("No internal nominations found")
}

metadata <- read.csv(
  "data/processed/metadata_clean.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
metadata <- metadata[metadata$label %in% c(0, 1), , drop = FALSE]

rows <- list()
for (cohort in OUR_COHORTS) {
  cat("[", cohort, "] loading taxon-resolved candidate rows\n", sep = "")
  pattern <- paste0("2021-03-31.", cohort, ".gene_families")
  objects <- curatedMetagenomicData(pattern, dryrun = FALSE, counts = FALSE)
  if (length(objects) == 0) {
    warning("No gene-family object found for ", cohort)
    next
  }
  mat <- assay(objects[[1]])
  analysis_ids <- metadata$sample_id[metadata$study_name == cohort]
  kept_ids <- intersect(analysis_ids, colnames(mat))
  feature_ids <- rownames(mat)
  base_ids <- sub("\\|.*$", "", feature_ids)
  kept_rows <- which(base_ids %in% candidate_ids)
  if (length(kept_rows) == 0 || length(kept_ids) == 0) {
    warning("No nominated candidate rows/samples for ", cohort)
    next
  }

  selected <- as(mat[kept_rows, kept_ids, drop = FALSE], "CsparseMatrix")
  nonzero <- Matrix::summary(selected)
  if (nrow(nonzero) > 0) {
    selected_features <- feature_ids[kept_rows]
    source_feature <- selected_features[nonzero$i]
    taxon <- ifelse(
      grepl("|", source_feature, fixed = TRUE),
      sub("^[^|]*\\|", "", source_feature),
      "unstratified"
    )
    frame <- data.frame(
      sample_id = kept_ids[nonzero$j],
      study_name = cohort,
      gene_id = sub("\\|.*$", "", source_feature),
      taxon = taxon,
      abundance = nonzero$x,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    sample_metadata <- metadata[
      match(frame$sample_id, metadata$sample_id),
      c("sample_id", "label", "country"),
      drop = FALSE
    ]
    frame$label <- sample_metadata$label
    frame$country <- sample_metadata$country
    rows[[cohort]] <- frame
    cat("  exported ", nrow(frame), " non-zero rows\n", sep = "")
  }
  rm(objects, mat, selected)
  invisible(gc())
}

if (length(rows) == 0) {
  stop("No taxon-resolved candidate data were exported")
}

output <- do.call(rbind, rows)
rownames(output) <- NULL
output <- output[order(output$gene_id, output$study_name, output$sample_id, output$taxon), ]
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
write.csv(
  output,
  gzfile("data/raw/intervention_candidates_stratified.csv.gz"),
  row.names = FALSE
)
cat("Wrote ", nrow(output), " rows\n", sep = "")
