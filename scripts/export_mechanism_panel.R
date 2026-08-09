# Export the frozen, hypothesis-driven mechanism panel from cMD gene families.
# The manifest must be frozen before this script is run.

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

manifest <- read.csv(
  "results/mechanism_panel/frozen_manifest.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
feature_ids <- sort(unique(manifest$uniref90[manifest$query_status == "frozen_detected"]))
feature_ids <- feature_ids[nzchar(feature_ids)]
if (length(feature_ids) == 0) {
  stop("Frozen mechanism manifest contains no detected UniRef90 clusters")
}

metadata <- read.csv(
  "data/processed/metadata_clean.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
metadata <- metadata[metadata$label %in% c(0, 1), , drop = FALSE]

blocks <- list()
sample_rows <- list()
for (cohort in OUR_COHORTS) {
  cat("[", cohort, "] exporting mechanism genes\n", sep = "")
  objects <- curatedMetagenomicData(
    paste0("2021-03-31.", cohort, ".gene_families"),
    dryrun = FALSE,
    counts = FALSE
  )
  if (length(objects) == 0) {
    stop("No gene-family object found for ", cohort)
  }
  mat <- assay(objects[[1]])
  analysis_ids <- metadata$sample_id[metadata$study_name == cohort]
  kept_ids <- intersect(analysis_ids, colnames(mat))
  source_rows <- match(feature_ids, rownames(mat))
  present <- which(!is.na(source_rows))

  block <- Matrix(
    0,
    nrow = length(feature_ids),
    ncol = length(kept_ids),
    sparse = TRUE,
    dimnames = list(feature_ids, kept_ids)
  )
  if (length(present) > 0) {
    block[present, ] <- as(
      mat[source_rows[present], kept_ids, drop = FALSE],
      "sparseMatrix"
    )
  }
  blocks[[cohort]] <- block
  sample_rows[[cohort]] <- metadata[
    match(kept_ids, metadata$sample_id),
    c("sample_id", "study_name", "label", "country"),
    drop = FALSE
  ]
  cat(
    "  samples=", length(kept_ids),
    ", clusters present=", length(present), " / ", length(feature_ids), "\n",
    sep = ""
  )
  rm(objects, mat, block)
  invisible(gc())
}

combined <- do.call(cbind, blocks)
samples <- do.call(rbind, sample_rows)
rownames(samples) <- NULL
out_prefix <- "data/raw/mechanism_panel"
invisible(writeMM(combined, paste0(out_prefix, ".mtx")))
writeLines(feature_ids, paste0(out_prefix, ".features.txt"), useBytes = TRUE)
write.csv(samples, paste0(out_prefix, ".samples.csv"), row.names = FALSE)
cat(
  "Wrote mechanism matrix: ", nrow(combined), " clusters x ",
  ncol(combined), " samples\n", sep = ""
)
