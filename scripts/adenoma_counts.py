import pandas as pd
import os

md = pd.read_csv("data/processed/metadata_clean.csv")

ade = md[md["study_condition"] == "adenoma"]
counts = ade.groupby("study_name").size().reset_index(name="n_adenoma")
counts = counts.sort_values("n_adenoma", ascending=False).reset_index(drop=True)

total_ade = len(ade)
total_ctrl = (md["study_condition"] == "control").sum()
total_crc = (md["study_condition"] == "CRC").sum()

print(f"Total adenoma samples: {total_ade}")
print(f"Total controls:        {total_ctrl}")
print(f"Total CRC:             {total_crc}\n")
print("Per-cohort adenoma counts:")
print(counts.to_string(index=False))

n_cohorts_total = md["study_name"].nunique()
cohorts_with_ade = counts[counts["n_adenoma"] >= 5]
print(f"\nCohorts with >=5 adenoma samples:  {len(cohorts_with_ade)} of {n_cohorts_total}")
print(f"Cohorts with >=10 adenoma samples: {(counts['n_adenoma'] >= 10).sum()} of {n_cohorts_total}")

os.makedirs("results", exist_ok=True)
counts.to_csv("results/adenoma_counts_per_cohort.csv", index=False)
print("\nSaved results/adenoma_counts_per_cohort.csv")
print("Note: results/adenoma_go_nogo_memo.md is maintained manually; not overwritten here.")
