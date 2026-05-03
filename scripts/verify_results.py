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
    detail = f"got {actual:.4f}, expected {expected:.4f} (tol {tol})"
    check(name, ok, detail)


def main():
    print("=== Verification checks ===\n")
    print("--- LODO baseline / joint per-cohort means ---")
    bl = pd.read_csv("results/baseline_results.csv")
    jr = pd.read_csv("results/joint_results.csv")
    check_near("Baseline species RF per-cohort mean AUC",
               bl["auc"].mean(), 0.803, tol=0.005)
    check_near("Joint RF per-cohort mean AUC",
               jr["rf_auc"].mean(), 0.785, tol=0.005)
    check_near("Joint XGB per-cohort mean AUC",
               jr["xgb_auc"].mean(), 0.784, tol=0.005)
    check("LODO baseline fold count = 7", len(bl) == 7,
          detail=f"got {len(bl)}")
    check("LODO joint fold count = 7", len(jr) == 7, detail=f"got {len(jr)}")

    print("\n--- Pooled prediction files (n = 646 binary samples) ---")
    pred_files = [
        ("results/preds_species_rf.csv", 646),
        ("results/preds_joint_rf.csv", 646),
        ("results/preds_joint_xgb.csv", 646),
    ]
    for pf, expected_n in pred_files:
        df = pd.read_csv(pf)
        check(f"{os.path.basename(pf)} sample count",
              len(df) == expected_n,
              detail=f"got {len(df)}, expected {expected_n}")
        # IDs unique
        check(f"{os.path.basename(pf)} sample_id unique",
              df["sample_id"].is_unique,
              detail=f"{len(df)} rows / {df['sample_id'].nunique()} unique IDs")

    print("\n--- DeLong significance (auc_comparison.py) ---")
    dl = pd.read_csv("results/delong_results.csv")
    row_sj = dl[(dl.model_a == "species_rf") & (dl.model_b == "joint_rf")].iloc[0]
    row_sx = dl[(dl.model_a == "species_rf") & (dl.model_b == "joint_xgb")].iloc[0]
    check_near("DeLong species_rf vs joint_rf z",  row_sj["z"], 2.88, tol=0.05)
    check_near("DeLong species_rf vs joint_rf p",  row_sj["p_value"], 0.004, tol=0.005)
    check_near("DeLong species_rf vs joint_xgb z", row_sx["z"], 2.65, tol=0.05)
    check_near("DeLong species_rf vs joint_xgb p", row_sx["p_value"], 0.008, tol=0.005)

    print("\n--- Bootstrap CI (species RF pooled) ---")
    bc = pd.read_csv("results/bootstrap_ci.csv")
    sp_pool = bc[(bc.model == "species_rf") & (bc.cohort == "pooled")].iloc[0]
    check_near("Bootstrap pooled species_rf AUC",     sp_pool["auc"],   0.810, tol=0.005)
    check_near("Bootstrap pooled species_rf CI lower", sp_pool["ci_lo"], 0.777, tol=0.015)
    check_near("Bootstrap pooled species_rf CI upper", sp_pool["ci_hi"], 0.841, tol=0.015)

    print("\n--- Adenoma LODO (post-fix: 4 rows present) ---")
    al = pd.read_csv("results/adenoma_lodo_results.csv")
    expected_tasks = {"h_vs_a_rf", "h_vs_a_xgb", "a_vs_crc_rf", "a_vs_crc_xgb"}
    actual_tasks = set(al["task"])
    check("Adenoma LODO has all 4 task rows",
          expected_tasks == actual_tasks,
          detail=f"missing {expected_tasks - actual_tasks}")
    if "h_vs_a_rf" in actual_tasks:
        m = al[al["task"] == "h_vs_a_rf"]["mean_lodo_auc"].iloc[0]
        check("H-vs-A RF LODO at-or-below-chance (< 0.55)",
              m < 0.55, detail=f"got {m:.3f}")
    if "a_vs_crc_rf" in actual_tasks:
        m = al[al["task"] == "a_vs_crc_rf"]["mean_lodo_auc"].iloc[0]
        check_near("A-vs-CRC RF LODO mean AUC", m, 0.583, tol=0.005)

    print("\n--- Per-fold pathway count in joint LODO (402..406) ---")
    if "rf_n_features" in jr.columns:
        # joint = 247 species + retained pathways; expected pathway count 402..406.
        species_count = 247
        pw_per_fold = jr["rf_n_features"] - species_count
        in_range = ((pw_per_fold >= 402) & (pw_per_fold <= 406)).all()
        check("Per-fold pathway count in [402, 406]",
              bool(in_range),
              detail=f"got {sorted(pw_per_fold.tolist())}")

    print("\n--- Seed sensitivity (seeds 0,1,2,42,100) ---")
    if os.path.exists("results/seed_sensitivity.csv"):
        ss = pd.read_csv("results/seed_sensitivity.csv")
        check("Seed sensitivity has 5 rows",
              len(ss) == 5, detail=f"got {len(ss)}")
        if len(ss) >= 1:
            spread = ss["mean_auc"].max() - ss["mean_auc"].min()
            check("Seed sensitivity spread < 0.01",
                  spread < 0.01, detail=f"got {spread:.4f}")

    print("\n--- ComBat batch-correction results (post-fix) ---")
    if os.path.exists("results/combat_results.csv"):
        cb = pd.read_csv("results/combat_results.csv")
        check("ComBat results have 7 cohort rows",
              len(cb) == 7, detail=f"got {len(cb)}")
        if len(cb) > 0:
            mean_cb = cb["auc"].mean()
            check_near("ComBat mean AUC near uncorrected baseline",
                       mean_cb, bl["auc"].mean(), tol=0.02)

    print("\n--- Confounder adjustment ---")
    if os.path.exists("results/confounder_results.csv"):
        cf = pd.read_csv("results/confounder_results.csv")
        for method in ("direct_rf", "direct_xgb", "residualized_rf", "residualized_xgb"):
            sub = cf[cf["method"] == method]
            check(f"Confounder method {method} present",
                  len(sub) == 1, detail=f"got {len(sub)} rows")

    print("\n--- Sensitivity grid (per-fold pathway filter) ---")
    if os.path.exists("results/sensitivity_thresholds.csv"):
        st = pd.read_csv("results/sensitivity_thresholds.csv")
        check("Sensitivity grid has 20 cells",
              len(st) == 20, detail=f"got {len(st)}")
        # Excluding the degenerate mean=1e-3 column the spread should be small.
        non_deg = st[st["mean_threshold"] < 1e-3]
        if len(non_deg) > 0:
            spread_non_deg = non_deg["mean_auc"].max() - non_deg["mean_auc"].min()
            check("Sensitivity grid (non-degenerate cells) spread < 0.025",
                  spread_non_deg < 0.025,
                  detail=f"got {spread_non_deg:.4f}")

    print("\n--- Metadata sanity ---")
    md = pd.read_csv("data/processed/metadata_clean.csv")
    sc = set(md["study_condition"].unique())
    expected = {"CRC", "control", "adenoma"}
    check("Metadata study_condition values == {CRC, control, adenoma}",
          sc == expected,
          detail=f"got {sc}")
    check("'healthy' label NOT present in metadata "
          "(prevents adenoma_lodo H-vs-A regression)",
          "healthy" not in sc,
          detail=f"present={sc}")
    n_crc = (md["study_condition"] == "CRC").sum()
    n_ctrl = (md["study_condition"] == "control").sum()
    n_aden = (md["study_condition"] == "adenoma").sum()
    check(f"Sample counts (CRC={n_crc}, control={n_ctrl}, adenoma={n_aden})",
          (n_crc, n_ctrl, n_aden) == (326, 320, 116),
          detail=f"got {(n_crc, n_ctrl, n_aden)}")

    print(f"\n{len(failures)} failure(s)")
    if failures:
        for f in failures:
            print(f"  - {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
