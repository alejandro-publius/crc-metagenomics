"""Headline-numbers verifier. Run after a full results regeneration to
confirm the manuscript's quoted figures still match the saved CSVs.

Exit 0 if all pass, exit 1 on any failure. Tolerances are per-check.

Usage:
    python3 scripts/verify_results.py
"""
import os
import sys
import pandas as pd
import numpy as np

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def near(actual, expected, tol):
    try:
        return abs(float(actual) - float(expected)) <= tol
    except Exception:
        return False


def check_near(name, actual, expected, tol):
    ok = near(actual, expected, tol)
    detail = f"got {float(actual):.4f}, expected {expected:.4f} (tol {tol})"
    check(name, ok, detail)


def main():
    print("=== Verification checks ===\n")

    print("--- LODO baseline / joint per-cohort means (10-cohort dataset) ---")
    bl = pd.read_csv("results/baseline_results.csv")
    jr = pd.read_csv("results/joint_results.csv")
    check_near("Baseline species RF per-cohort mean AUC",
               bl["auc"].mean(), 0.808, tol=0.005)
    check_near("Joint RF per-cohort mean AUC",
               jr["rf_auc"].mean(), 0.804, tol=0.005)
    check_near("Joint XGB per-cohort mean AUC",
               jr["xgb_auc"].mean(), 0.797, tol=0.005)
    check("LODO baseline fold count = 10", len(bl) == 10,
          detail=f"got {len(bl)}")
    check("LODO joint fold count = 10", len(jr) == 10, detail=f"got {len(jr)}")

    print("\n--- Pooled prediction files (n = 1339 binary samples, 10 cohorts) ---")
    pred_files = [
        ("results/preds_species_rf.csv", 1339),
        ("results/preds_joint_rf.csv", 1339),
        ("results/preds_joint_xgb.csv", 1339),
    ]
    for pf, expected_n in pred_files:
        if not os.path.exists(pf):
            check(f"{os.path.basename(pf)} exists", False, detail="file missing")
            continue
        df = pd.read_csv(pf)
        check(f"{os.path.basename(pf)} sample count = {expected_n}",
              len(df) == expected_n,
              detail=f"got {len(df)}")
        check(f"{os.path.basename(pf)} sample_id unique",
              df["sample_id"].is_unique,
              detail=f"{len(df)} rows / {df['sample_id'].nunique()} unique IDs")
        n_cohorts = df["cohort"].nunique()
        check(f"{os.path.basename(pf)} covers 10 cohorts",
              n_cohorts == 10,
              detail=f"got {n_cohorts}")

    print("\n--- DeLong significance on pooled LODO predictions (n=1339) ---")
    dl = pd.read_csv("results/delong_results.csv")
    row_sj = dl[(dl.model_a == "species_rf") & (dl.model_b == "joint_rf")].iloc[0]
    row_sx = dl[(dl.model_a == "species_rf") & (dl.model_b == "joint_xgb")].iloc[0]
    check("DeLong n_samples = 1339",
          int(row_sj["n_samples"]) == 1339,
          detail=f"got {row_sj['n_samples']}")
    check_near("DeLong species_rf vs joint_rf z",    row_sj["z"],       3.352, tol=0.05)
    check_near("DeLong species_rf vs joint_rf p",    row_sj["p_value"], 0.0008, tol=0.001)
    check_near("DeLong species_rf vs joint_xgb z",   row_sx["z"],       1.996, tol=0.05)
    check_near("DeLong species_rf vs joint_xgb p",   row_sx["p_value"], 0.046,  tol=0.01)

    print("\n--- Bootstrap CI (species RF pooled, 10-cohort) ---")
    bc = pd.read_csv("results/bootstrap_ci.csv")
    sp_pool = bc[(bc.model == "species_rf") & (bc.cohort == "pooled")].iloc[0]
    check_near("Bootstrap pooled species_rf AUC",       sp_pool["auc"],   0.781, tol=0.005)
    check_near("Bootstrap pooled species_rf CI lower",  sp_pool["ci_lo"], 0.757, tol=0.010)
    check_near("Bootstrap pooled species_rf CI upper",  sp_pool["ci_hi"], 0.805, tol=0.010)
    check("Bootstrap n_samples = 1339",
          int(sp_pool["n"]) == 1339,
          detail=f"got {sp_pool['n']}")

    print("\n--- Adenoma LODO (4-cohort, YachidaS_2019 included, bugs fixed) ---")
    al = pd.read_csv("results/adenoma_lodo_results.csv")
    expected_tasks = {"h_vs_a_rf", "h_vs_a_xgb", "a_vs_crc_rf", "a_vs_crc_xgb"}
    actual_tasks = set(al["task"])
    check("Adenoma LODO has all 4 task rows",
          expected_tasks == actual_tasks,
          detail=f"missing {expected_tasks - actual_tasks}")
    for task, expected_folds in [("h_vs_a_rf", 4), ("h_vs_a_xgb", 4),
                                  ("a_vs_crc_rf", 4), ("a_vs_crc_xgb", 4)]:
        if task in actual_tasks:
            n = al[al["task"] == task]["n_folds"].iloc[0]
            check(f"{task} n_folds = {expected_folds}",
                  int(n) == expected_folds, detail=f"got {n}")
    if "h_vs_a_rf" in actual_tasks:
        m = al[al["task"] == "h_vs_a_rf"]["mean_lodo_auc"].iloc[0]
        check_near("H-vs-A RF mean LODO AUC (near-chance)", m, 0.561, tol=0.01)
    if "a_vs_crc_rf" in actual_tasks:
        m = al[al["task"] == "a_vs_crc_rf"]["mean_lodo_auc"].iloc[0]
        check_near("A-vs-CRC RF mean LODO AUC", m, 0.671, tol=0.01)

    print("\n--- Per-fold feature counts in joint LODO (229 species + 402-406 pathways) ---")
    species_count = 229
    if "rf_n_features" in jr.columns:
        pw_per_fold = jr["rf_n_features"] - species_count
        in_range = ((pw_per_fold >= 400) & (pw_per_fold <= 410)).all()
        check("Per-fold pathway count in [400, 410]",
              bool(in_range),
              detail=f"got {sorted(pw_per_fold.tolist())}")

    print("\n--- Seed sensitivity (seeds 0,1,2,42,100; 10-cohort) ---")
    if os.path.exists("results/seed_sensitivity.csv"):
        ss = pd.read_csv("results/seed_sensitivity.csv")
        check("Seed sensitivity has 5 rows",
              len(ss) == 5, detail=f"got {len(ss)}")
        if len(ss) >= 2:
            spread = ss["mean_auc"].max() - ss["mean_auc"].min()
            check("Seed sensitivity spread < 0.005",
                  spread < 0.005, detail=f"got {spread:.4f}")
            check_near("Seed sensitivity grand mean", ss["mean_auc"].mean(), 0.810, tol=0.005)

    print("\n--- Confounder adjustment (10-cohort) ---")
    if os.path.exists("results/confounder_results.csv"):
        cf = pd.read_csv("results/confounder_results.csv")
        for method in ("direct_rf", "direct_xgb", "residualized_rf", "residualized_xgb"):
            sub = cf[cf["method"] == method]
            check(f"Confounder method '{method}' present",
                  len(sub) == 1, detail=f"got {len(sub)} rows")
        if len(cf) == 4:
            auc_range = cf["mean_auc"].max() - cf["mean_auc"].min()
            check("Confounder AUC range < 0.015",
                  auc_range < 0.015, detail=f"got {auc_range:.4f}")

    print("\n--- Sensitivity grid (per-fold pathway filter, 10-cohort) ---")
    if os.path.exists("results/sensitivity_thresholds.csv"):
        st = pd.read_csv("results/sensitivity_thresholds.csv")
        check("Sensitivity grid has 20 cells",
              len(st) == 20, detail=f"got {len(st)}")
        spread = st["mean_auc"].max() - st["mean_auc"].min()
        check("Sensitivity grid full spread < 0.020",
              spread < 0.020, detail=f"got {spread:.4f}")

    print("\n--- Metadata integrity (10-cohort dataset) ---")
    md = pd.read_csv("data/processed/metadata_clean.csv")
    sc = set(md["study_condition"].unique())
    check("study_condition values = {CRC, control, adenoma}",
          sc == {"CRC", "control", "adenoma"}, detail=f"got {sc}")
    check("'healthy' NOT in study_condition",
          "healthy" not in sc)
    n_crc  = int((md["study_condition"] == "CRC").sum())
    n_ctrl = int((md["study_condition"] == "control").sum())
    n_aden = int((md["study_condition"] == "adenoma").sum())
    check("CRC sample count = 674",     n_crc  == 674, detail=f"got {n_crc}")
    check("Control sample count = 665", n_ctrl == 665, detail=f"got {n_ctrl}")
    check("Adenoma sample count = 183", n_aden == 183, detail=f"got {n_aden}")
    n_cohorts = md["study_name"].nunique()
    check("10 cohorts in metadata", n_cohorts == 10, detail=f"got {n_cohorts}")
    check("HanniganGD_2017 excluded",
          "HanniganGD_2017" not in md["study_name"].values)
    check("YachidaS_2019 present",
          "YachidaS_2019" in md["study_name"].values)

    print(f"\n{'='*40}")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        print("All checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
