# Pull HUMAnN species-stratified pathway abundances for our 10 cohorts.
# Output: data/raw/pathway_stratified_v320.csv (samples x features, wide)
#
# This is the finer-resolution feature space (per-species, per-pathway)
# that the existing joint model does NOT use. The existing model uses
# community-level (unstratified) pathway abundances only.
#
# Independent of any external lab or method - purely Bioconductor /
# Segata-lab-curated cMD data.

suppressMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(dplyr)
  library(tibble)
})

OUR_COHORTS <- c("FengQ_2015", "GuptaA_2019", "ThomasAM_2018a", "ThomasAM_2018b",
                 "ThomasAM_2019_c", "VogtmannE_2016", "WirbelJ_2018", "YachidaS_2019",
                 "YuJ_2015", "ZellerG_2014")

OUT_DIR <- "data/raw"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_CSV <- file.path(OUT_DIR, "pathway_stratified_v320.csv")
SUMMARY_CSV <- file.path(OUT_DIR, "pathway_stratified_v320_summary.csv")

cat("=== Pulling stratified pathway abundances ===\n")
cat("cMD version:", as.character(packageVersion("curatedMetagenomicData")), "\n")
cat("Output:", OUT_CSV, "\n\n")

per_cohort <- list()
summary_rows <- list()

for (cohort in OUR_COHORTS) {
  cat("[", cohort, "] ", sep = "")
  pattern <- paste0("2021-03-31.", cohort, ".pathway_abundance")
  se <- curatedMetagenomicData(pattern, dryrun = FALSE, counts = FALSE)
  if (length(se) == 0) {
    cat("MISSING\n")
    next
  }
  se <- se[[1]]
  m <- assay(se)  # features x samples
  # Transpose to samples x features, attach sample_id and study_name
  df <- t(m) %>%
    as.data.frame() %>%
    rownames_to_column("sample_id") %>%
    mutate(study_name = cohort, .after = sample_id)
  per_cohort[[cohort]] <- df
  
  # Count stratified vs unstratified
  n_strat <- sum(grepl("\\|", rownames(m)))
  n_unstrat <- nrow(m) - n_strat
  summary_rows[[cohort]] <- data.frame(
    study_name = cohort,
    n_samples = ncol(m),
    n_features_total = nrow(m),
    n_stratified = n_strat,
    n_unstratified = n_unstrat
  )
  cat("OK (", ncol(m), " samples, ", nrow(m), " features, ", n_strat, " stratified)\n", sep="")
}

# Bind union (different cohorts have different feature universes; bind_rows handles missing as NA)
cat("\n=== Binding all cohorts (union of features) ===\n")
combined <- bind_rows(per_cohort)
cat("Combined shape: ", nrow(combined), " samples x ", ncol(combined) - 2, " features (incl. sample_id+study_name)\n", sep="")
cat("Memory: ~", round(object.size(combined) / 1024^2, 0), " MB\n", sep="")

# Save with NA -> 0 (HUMAnN missing means zero abundance)
combined[is.na(combined)] <- 0

write.csv(combined, OUT_CSV, row.names = FALSE)
cat("Wrote:", OUT_CSV, "(", round(file.size(OUT_CSV) / 1024^2, 0), " MB)\n", sep="")

# Save summary
summary_df <- bind_rows(summary_rows)
write.csv(summary_df, SUMMARY_CSV, row.names = FALSE)
cat("Wrote summary:", SUMMARY_CSV, "\n")

cat("\nDone.\n")
