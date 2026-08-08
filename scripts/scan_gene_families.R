# Scan cMD gene-family tables without materializing a multi-cohort wide CSV.
#
# Pass 1 records compact, per-cohort sufficient statistics for unstratified
# UniRef90 gene families. Feature selection is performed later, separately
# for every LODO training fold, so the held-out cohort cannot determine which
# genes enter its model.
#
# Usage:
#   Rscript scripts/scan_gene_families.R
#   Rscript scripts/scan_gene_families.R --cohort GuptaA_2019
#
# Outputs:
#   data/interim/gene_family_scan/<cohort>.csv.gz
#   data/interim/gene_family_scan/scan_summary.csv

suppressMessages({
  library(curatedMetagenomicData)
  library(Matrix)
  library(SummarizedExperiment)
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

chunk_flag <- match("--chunk-size", args)
if (!is.na(chunk_flag)) {
  if (chunk_flag == length(args)) {
    stop("--chunk-size requires a positive integer")
  }
  chunk_size <- as.integer(args[[chunk_flag + 1]])
  if (is.na(chunk_size) || chunk_size < 1) {
    stop("--chunk-size requires a positive integer")
  }
} else {
  chunk_size <- 50000L
}

metadata <- read.csv(
  "data/processed/metadata_clean.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
metadata <- metadata[metadata$label %in% c(0, 1), , drop = FALSE]

out_dir <- "data/interim/gene_family_scan"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary_path <- file.path(out_dir, "scan_summary.csv")

existing_summary <- if (file.exists(summary_path)) {
  read.csv(summary_path, stringsAsFactors = FALSE)
} else {
  data.frame()
}

summary_rows <- list()

for (cohort in cohorts) {
  cat("[", cohort, "] loading gene families\n", sep = "")
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
  gene_ids <- rownames(mat)
  keep_gene <- !grepl("|", gene_ids, fixed = TRUE) &
    !gene_ids %in% c("UNMAPPED", "UNINTEGRATED")
  gene_rows <- which(keep_gene)
  out_path <- file.path(out_dir, paste0(cohort, ".csv.gz"))
  tmp_path <- paste0(out_path, ".tmp")
  if (file.exists(tmp_path)) {
    unlink(tmp_path)
  }
  out_connection <- gzfile(tmp_path, open = "wt")
  wrote_header <- FALSE
  retained_total <- 0L

  if (inherits(mat, "sparseMatrix")) {
    # cMD stores these assays as triplet sparse matrices. Convert only the
    # analysis columns to compressed-column form, then let Matrix aggregate
    # the non-zero entries without copying millions of mostly-zero row slabs.
    selected <- as(mat[, kept_ids, drop = FALSE], "CsparseMatrix")
    n_nonzero <- as.numeric(Matrix::rowSums(selected != 0))
    total_abundance <- as.numeric(Matrix::rowSums(selected))
    keep_summary <- keep_gene & n_nonzero >= 2 & is.finite(total_abundance)

    if (any(keep_summary)) {
      stats <- data.frame(
        cohort = cohort,
        gene_id = gene_ids[keep_summary],
        n_samples = length(kept_ids),
        n_nonzero = n_nonzero[keep_summary],
        total_abundance = total_abundance[keep_summary],
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      write.table(
        stats,
        out_connection,
        sep = ",",
        row.names = FALSE,
        col.names = TRUE,
        quote = TRUE
      )
      wrote_header <- TRUE
      retained_total <- nrow(stats)
      rm(stats)
    }
    rm(selected, n_nonzero, total_abundance)
    cat("  sparse aggregation complete\n")
  } else {
    # Dense or delayed fall-back: work in row blocks so ``mat != 0`` cannot
    # exceed R's vector-memory limit on a large cohort.
    for (start in seq.int(1L, length(gene_rows), by = chunk_size)) {
      stop_at <- min(start + chunk_size - 1L, length(gene_rows))
      rows <- gene_rows[start:stop_at]
      block <- mat[rows, kept_ids, drop = FALSE]
      n_nonzero <- as.numeric(rowSums(block != 0))
      total_abundance <- as.numeric(rowSums(block))
      keep_summary <- n_nonzero >= 2 & is.finite(total_abundance)
      if (any(keep_summary)) {
        stats <- data.frame(
          cohort = cohort,
          gene_id = gene_ids[rows][keep_summary],
          n_samples = length(kept_ids),
          n_nonzero = n_nonzero[keep_summary],
          total_abundance = total_abundance[keep_summary],
          stringsAsFactors = FALSE,
          check.names = FALSE
        )
        write.table(
          stats,
          out_connection,
          sep = ",",
          row.names = FALSE,
          col.names = !wrote_header,
          quote = TRUE
        )
        wrote_header <- TRUE
        retained_total <- retained_total + nrow(stats)
        rm(stats)
      }
      rm(block, n_nonzero, total_abundance)
    }
  }
  close(out_connection)
  if (!wrote_header) {
    empty <- data.frame(
      cohort = character(), gene_id = character(), n_samples = integer(),
      n_nonzero = integer(), total_abundance = numeric()
    )
    write.csv(empty, gzfile(tmp_path), row.names = FALSE)
  }
  if (!file.rename(tmp_path, out_path)) {
    stop("Could not move completed scan into place: ", out_path)
  }

  summary_rows[[cohort]] <- data.frame(
    cohort = cohort,
    n_analysis_samples = length(kept_ids),
    n_unstratified_genes = length(gene_rows),
    n_genes_with_two_or_more_samples = retained_total,
    output = out_path,
    stringsAsFactors = FALSE
  )
  cat(
    "  samples=", length(kept_ids),
    ", unstratified=", length(gene_rows),
    ", retained summaries=", retained_total, "\n",
    sep = ""
  )

  rm(objects, mat)
  invisible(gc())
}

if (length(summary_rows) > 0) {
  new_summary <- do.call(rbind, summary_rows)
  if (nrow(existing_summary) > 0) {
    existing_summary <- existing_summary[
      !existing_summary$cohort %in% new_summary$cohort,
      ,
      drop = FALSE
    ]
    new_summary <- rbind(existing_summary, new_summary)
  }
  new_summary <- new_summary[order(new_summary$cohort), , drop = FALSE]
  write.csv(new_summary, summary_path, row.names = FALSE)
  cat("Updated ", summary_path, "\n", sep = "")
}
