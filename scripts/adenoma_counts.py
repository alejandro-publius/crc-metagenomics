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

cohorts_with_ade = counts[counts["n_adenoma"] >= 5]
print(f"\nCohorts with >=5 adenoma samples: {len(cohorts_with_ade)} of 7")
print(f"Cohorts with >=10 adenoma samples: {(counts['n_adenoma'] >= 10).sum()} of 7")

os.makedirs("results", exist_ok=True)
counts.to_csv("results/adenoma_counts_per_cohort.csv", index=False)
print("\nSaved results/adenoma_counts_per_cohort.csv")

memo = f"""# Adenoma Analysis: Go/No-Go Memo

## Sample availability
- Total adenoma samples across 7 cohorts: {total_ade}
- Roadmap threshold for "meaningful analysis": ~30 adenoma samples
- Cohorts with >=5 adenoma samples: {len(cohorts_with_ade)} of 7
- Cohorts with >=10 adenoma samples: {(counts['n_adenoma'] >= 10).sum()} of 7

## Per-cohort breakdown
{counts.to_string(index=False)}

## Decision
Total adenoma count ({total_ade}) {'EXCEEDS' if total_ade >= 30 else 'is BELOW'} the 30-sample threshold.

## CV strategy
Per-cohort adenoma counts are too sparse for LODO (most cohorts have
fewer than 10 adenoma samples). The adenoma analysis uses pooled
5-fold stratified cross-validation instead. This deviation from the
LODO protocol used elsewhere in the paper is documented in Methods.

## Limitations to disclose in Discussion
- 5-fold CV does not test cross-cohort generalization for adenoma
- Adenoma definitions may vary across cohorts (advanced vs non-advanced)
- Sample size limits the precision of reported AUCs
"""

with open("results/adenoma_go_nogo_memo.md", "w") as f:
    f.write(memo)
print("Saved results/adenoma_go_nogo_memo.md")
