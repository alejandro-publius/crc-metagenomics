"""
generate_table1.py — Generate Table 1 (demographics summary) for the manuscript.

What this script does:
1. Loads the metadata
2. For each cohort, computes:
   - Number of CRC, adenoma, control samples
   - Age (mean and standard deviation)
   - Gender split (% female)
   - BMI (mean and standard deviation)
   - Country of origin
3. Saves a publication-ready table to results/table1.csv
4. Prints a formatted version that can be copied into Word

Why Table 1 matters:
Every clinical and microbiome paper has a Table 1 showing demographics.
Reviewers expect it. It's how readers verify that your cohorts are
comparable and that the populations are described transparently.

Authors: Alex Velazquez, Rachel Selbrede

Usage:
    Run from the project root:
        python3 scripts/generate_table1.py

Output:
    results/table1.csv (clean version for the paper)
    Console output with a formatted table you can paste into Word
"""

import pandas as pd
import numpy as np
import os


def main():
    print("=" * 70)
    print("CRC Metagenomics — Table 1: Cohort Demographics")
    print("=" * 70)

    # ── Section 1: Load the metadata ──
    metadata = pd.read_csv("data/processed/metadata_clean.csv")
    print(f"\nTotal samples in metadata: {len(metadata)}")

    # ── Section 2: Set up the table structure ──
    # We'll build one row per cohort with all the demographic info
    rows = []

    # Get the unique cohorts in a consistent order
    cohorts = sorted(metadata["study_name"].unique())
    print(f"Number of cohorts: {len(cohorts)}\n")

    # ── Section 3: Compute statistics per cohort ──
    for cohort in cohorts:
        # Filter to just this cohort's samples
        df = metadata[metadata["study_name"] == cohort]

        # Count samples by condition
        # We use .get() with default 0 in case a cohort has no samples in a category
        condition_counts = df["study_condition"].value_counts().to_dict()
        n_crc = condition_counts.get("CRC", 0)
        n_adenoma = condition_counts.get("adenoma", 0)
        n_control = condition_counts.get("control", 0)
        n_total = n_crc + n_adenoma + n_control

        # Age statistics
        # Use .dropna() to handle any missing values
        age_values = df["age"].dropna() if "age" in df.columns else pd.Series([])
        if len(age_values) > 0:
            age_mean = age_values.mean()
            age_std = age_values.std()
            age_str = f"{age_mean:.1f} ± {age_std:.1f}"
        else:
            age_str = "NA"

        # Gender split (percentage female)
        if "gender" in df.columns:
            gender_values = df["gender"].dropna()
            n_with_gender = len(gender_values)
            if n_with_gender > 0:
                # Handle different ways gender might be coded
                female_count = gender_values.isin(["female", "F", "f"]).sum()
                pct_female = 100 * female_count / n_with_gender
                gender_str = f"{pct_female:.1f}%"
            else:
                gender_str = "NA"
        else:
            gender_str = "NA"

        # BMI statistics
        bmi_values = df["BMI"].dropna() if "BMI" in df.columns else pd.Series([])
        if len(bmi_values) > 0:
            bmi_mean = bmi_values.mean()
            bmi_std = bmi_values.std()
            bmi_str = f"{bmi_mean:.1f} ± {bmi_std:.1f}"
        else:
            bmi_str = "NA"

        # Country (most common country in this cohort)
        if "country" in df.columns:
            country_values = df["country"].dropna()
            country = country_values.mode().iloc[0] if len(country_values) > 0 else "NA"
        else:
            country = "NA"

        # Add this cohort's row to the table
        rows.append({
            "Cohort": cohort,
            "Country": country,
            "N (total)": n_total,
            "N (CRC)": n_crc,
            "N (adenoma)": n_adenoma,
            "N (control)": n_control,
            "Age (mean ± SD)": age_str,
            "Female %": gender_str,
            "BMI (mean ± SD)": bmi_str
        })

    # ── Section 4: Add a totals row ──
    # This summarizes the entire dataset for quick reference
    total_n = len(metadata)
    total_crc = (metadata["study_condition"] == "CRC").sum()
    total_adenoma = (metadata["study_condition"] == "adenoma").sum()
    total_control = (metadata["study_condition"] == "control").sum()

    # Overall age
    if "age" in metadata.columns:
        overall_age = metadata["age"].dropna()
        overall_age_str = f"{overall_age.mean():.1f} ± {overall_age.std():.1f}" if len(overall_age) > 0 else "NA"
    else:
        overall_age_str = "NA"

    # Overall gender
    if "gender" in metadata.columns:
        overall_gender = metadata["gender"].dropna()
        if len(overall_gender) > 0:
            overall_female = overall_gender.isin(["female", "F", "f"]).sum()
            overall_gender_str = f"{100 * overall_female / len(overall_gender):.1f}%"
        else:
            overall_gender_str = "NA"
    else:
        overall_gender_str = "NA"

    # Overall BMI
    if "BMI" in metadata.columns:
        overall_bmi = metadata["BMI"].dropna()
        overall_bmi_str = f"{overall_bmi.mean():.1f} ± {overall_bmi.std():.1f}" if len(overall_bmi) > 0 else "NA"
    else:
        overall_bmi_str = "NA"

    rows.append({
        "Cohort": "TOTAL",
        "Country": "—",
        "N (total)": total_n,
        "N (CRC)": int(total_crc),
        "N (adenoma)": int(total_adenoma),
        "N (control)": int(total_control),
        "Age (mean ± SD)": overall_age_str,
        "Female %": overall_gender_str,
        "BMI (mean ± SD)": overall_bmi_str
    })

    # ── Section 5: Save and display ──
    table = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/table1.csv", index=False)

    print("Table 1: Cohort Demographics\n")
    # Print the table with nice formatting
    # The to_string() method displays the full table without truncation
    print(table.to_string(index=False))

    print(f"\n  Saved results/table1.csv")
    print("\n" + "=" * 70)
    print("How to use this in your manuscript:")
    print("  1. Open results/table1.csv in Excel or Numbers")
    print("  2. Copy the table")
    print("  3. Paste it into your manuscript as Table 1")
    print("  4. Format it as a proper publication table (borders, alignment)")
    print("=" * 70)


if __name__ == "__main__":
    main()
