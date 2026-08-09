# Export taxon-resolved HUMAnN rows for parent-adjustment survivors.
#
# This resolves whether an unstratified gene-family signal is primarily carried
# by one taxon or distributed across organisms. Candidate selection has already
# occurred; the export neither ranks candidates nor uses labels. Each cohort is
# written atomically so a memory failure cannot erase completed work.

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

args <- commandArgs(trailingOnly = TRUE)
cohort_flag <- match("--cohort", args)
if (!is.na(cohort_flag)) {
  if (cohort_flag == length(args)) {
    stop("--cohort requires a cohort name")
  }
  requested <- args[[cohort_flag + 1]]
  if (!requested %in% OUR_COHORTS) {
    stop("Unknown cohort: ", requested)
  }
  cohorts <- requested
} else {
  cohorts <- OUR_COHORTS
}

parent_summary <- read.csv(
  "results/intervention_readiness/parent_adjustment_summary.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
candidate_ids <- unique(
  parent_summary$gene_id[parent_summary$parent_adjustment_gate == "pass"]
)
if (length(candidate_ids) == 0) {
  stop("No parent-adjustment survivors found")
}

metadata <- read.csv(
  "data/processed/metadata_clean.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
metadata <- metadata[metadata$label %in% c(0, 1), , drop = FALSE]

out_dir <- "data/interim/intervention_candidate_stratified"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (cohort in cohorts) {
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
  # Avoid constructing a second vector of roughly three million feature names.
  # The previous implementation exhausted 16 GB of memory at this step.
  candidate_pattern <- paste0(
    "^(?:", paste(candidate_ids, collapse = "|"), ")(?:\\||$)"
  )
  kept_rows <- grep(candidate_pattern, feature_ids, perl = TRUE)
  if (length(kept_rows) == 0 || length(kept_ids) == 0) {
    warning("No nominated candidate rows/samples for ", cohort)
  }

  frame <- data.frame(
    sample_id = character(), study_name = character(), gene_id = character(),
    taxon = character(), abundance = numeric(), label = integer(),
    country = character(), stringsAsFactors = FALSE
  )
  if (length(kept_rows) > 0 && length(kept_ids) > 0) {
    selected <- as(mat[kept_rows, kept_ids, drop = FALSE], "CsparseMatrix")
    nonzero <- Matrix::summary(selected)
  } else {
    selected <- NULL
    nonzero <- data.frame(i = integer(), j = integer(), x = numeric())
  }
  if (nrow(nonzero) > 0L) {
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
    cat("  exported ", nrow(frame), " non-zero rows\n", sep = "")
  }

  out_path <- file.path(out_dir, paste0(cohort, ".csv.gz"))
  tmp_path <- paste0(out_path, ".tmp")
  if (file.exists(tmp_path)) {
    unlink(tmp_path)
  }
  write.csv(frame, gzfile(tmp_path), row.names = FALSE)
  if (!file.rename(tmp_path, out_path)) {
    stop("Could not move completed export into place: ", out_path)
  }
  rm(objects, mat, selected)
  invisible(gc())
}

available <- file.path(out_dir, paste0(OUR_COHORTS, ".csv.gz"))
if (all(file.exists(available))) {
  rows <- lapply(available, read.csv, stringsAsFactors = FALSE, check.names = FALSE)
  output <- do.call(rbind, rows)
  rownames(output) <- NULL
  output <- output[
    order(output$gene_id, output$study_name, output$sample_id, output$taxon),
  ]
  dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
  write.csv(
    output,
    gzfile("data/raw/intervention_candidates_stratified.csv.gz"),
    row.names = FALSE
  )
  cat("Wrote combined export with ", nrow(output), " rows\n", sep = "")
} else {
  cat(
    "Completed requested cohort export; ",
    sum(file.exists(available)), " of ", length(available),
    " cohort files are currently available\n",
    sep = ""
  )
}
