library(curatedMetagenomicData)
data("sampleMetadata")

thomas_cohorts <- c(
  "FengQ_2015", "YuJ_2015", "VogtmannE_2016",
  "ThomasAM_2018a", "ThomasAM_2018b", "ThomasAM_2019_c",
  "ZellerG_2014"
)

md <- sampleMetadata[
  sampleMetadata$study_condition %in% c("CRC", "adenoma", "control") &
  sampleMetadata$study_name %in% thomas_cohorts,
  c("sample_id", "subject_id", "study_name", "study_condition")
]

cat("Total samples:", nrow(md), "\n")
cat("Unique sample_ids:", length(unique(md$sample_id)), "\n")
cat("Unique subject_ids:", length(unique(md$subject_id)), "\n\n")

subj_cohorts <- aggregate(study_name ~ subject_id, data = md,
                          FUN = function(x) length(unique(x)))
multi_cohort <- subj_cohorts[subj_cohorts$study_name > 1, ]
cat("Subjects in >1 cohort:", nrow(multi_cohort), "\n")

subj_samples <- table(md$subject_id)
multi_sample <- subj_samples[subj_samples > 1]
cat("Subjects with >1 sample:", length(multi_sample), "\n")
cat("Max samples per subject:", max(subj_samples), "\n\n")

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
write.csv(md, "data/raw/subject_audit.csv", row.names = FALSE)

if (nrow(multi_cohort) > 0) {
  cat("WARNING: cross-cohort subjects detected. Examples:\n")
  print(head(multi_cohort, 10))
  write.csv(multi_cohort, "data/raw/cross_cohort_subjects.csv", row.names = FALSE)
} else {
  cat("OK: no subjects appear in more than one cohort.\n")
}

if (length(multi_sample) > 0) {
  cat("\nLongitudinal subjects, top 10 by sample count:\n")
  print(head(sort(multi_sample, decreasing = TRUE), 10))
}
