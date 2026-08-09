# Export only the union of training-fold-selected gene families as a sparse
# Matrix Market file. Run after scan_gene_families.R and
# select_gene_family_manifests.py.
#
# Outputs:
#   data/raw/gene_families_selected.mtx
#   data/raw/gene_families_selected.features.txt
#   data/raw/gene_families_selected.samples.csv

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

manifest_path <- "results/gene_family_manifests/selected_union.txt"
if (!file.exists(manifest_path)) {
  stop(
    "Missing ", manifest_path,
    "; run scripts/select_gene_family_manifests.py first"
  )
}

feature_ids <- scan(manifest_path, what = character(), quiet = TRUE)
feature_ids <- unique(feature_ids[nzchar(feature_ids)])
if (length(feature_ids) == 0) {
  stop("The selected gene-family union is empty")
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
  cat("[", cohort, "] exporting selected genes\n", sep = "")
  pattern <- paste0("2021-03-31.", cohort, ".gene_families")
  objects <- curatedMetagenomicData(pattern, dryrun = FALSE, counts = FALSE)
  if (length(objects) == 0) {
    warning("No gene-family object found for ", cohort)
    next
  }

  mat <- assay(objects[[1]])
  analysis_ids <- metadata$sample_id[metadata$study_name == cohort]
  kept_ids <- intersect(analysis_ids, colnames(mat))
  if (length(kept_ids) == 0) {
    warning("No analysis samples overlap the gene-family object for ", cohort)
    next
  }
  mat <- mat[, kept_ids, drop = FALSE]

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
    ", selected genes present=", length(present), "\n",
    sep = ""
  )

  rm(objects, mat, block)
  invisible(gc())
}

if (length(blocks) == 0) {
  stop("No cohort blocks were exported")
}

combined <- do.call(cbind, blocks)
samples <- do.call(rbind, sample_rows)
rownames(samples) <- NULL

out_prefix <- "data/raw/gene_families_selected"
invisible(writeMM(combined, paste0(out_prefix, ".mtx")))
writeLines(feature_ids, paste0(out_prefix, ".features.txt"), useBytes = TRUE)
write.csv(samples, paste0(out_prefix, ".samples.csv"), row.names = FALSE)

cat(
  "Wrote sparse matrix: ", nrow(combined), " genes x ", ncol(combined),
  " samples (", length(combined@x), " nonzero values)\n",
  sep = ""
)
