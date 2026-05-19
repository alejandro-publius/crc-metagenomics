"""Headline-numbers verifier. Run after a full results regeneration to
confirm the manuscript's quoted figures still match the saved CSVs.

Expected numbers, tolerances, and required cohorts are read from a YAML
manifest (default ``results/verify_manifest.yaml``) rather than being
hard-coded. To re-anchor the verifier on a new run, edit the manifest.

Exit 0 if all pass, exit 1 on any failure.

Usage:
    python3 scripts/verify_results.py [--manifest PATH]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

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
    detail = f"got {float(actual):.4f}, expected {float(expected):.4f} (tol {tol})"
    check(name, ok, detail)


def _vt(node):
    """Return (value, tol) from a {value, tol} manifest node."""
    return float(node["value"]), float(node["tol"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_manifest = os.path.join("results", "verify_manifest.yaml")
    p.add_argument(
        "--manifest",
        default=default_manifest,
        help=f"Path to YAML expectations manifest (default {default_manifest}).",
    )
    return p.parse_args(argv)


def load_manifest(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"verify_results.py: manifest not found at {path}. "
            f"Pass --manifest or create the file."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    args = parse_args(argv)
    m = load_manifest(args.manifest)

    # Reset failures list so main() is safely re-callable from tests.
    failures.clear()

    print(f"=== Verification checks (manifest: {args.manifest}) ===\n")

    n_samples = int(m["dataset"]["n_samples_binary"])
    n_cohorts = int(m["dataset"]["n_cohorts"])
    fold_count = int(m["lodo"]["fold_count"])

    print(f"--- LODO baseline / joint per-cohort means "
          f"({n_cohorts}-cohort dataset) ---")
    bl = pd.read_csv("results/baseline_results.csv")
    jr = pd.read_csv("results/joint_results.csv")
    v, t = _vt(m["lodo"]["baseline_species_rf_mean_auc"])
    check_near("Baseline species RF per-cohort mean AUC",
               bl["auc"].mean(), v, t)
    v, t = _vt(m["lodo"]["joint_rf_mean_auc"])
    check_near("Joint RF per-cohort mean AUC", jr["rf_auc"].mean(), v, t)
    v, t = _vt(m["lodo"]["joint_xgb_mean_auc"])
    check_near("Joint XGB per-cohort mean AUC", jr["xgb_auc"].mean(), v, t)
    check(f"LODO baseline fold count = {fold_count}",
          len(bl) == fold_count, detail=f"got {len(bl)}")
    check(f"LODO joint fold count = {fold_count}",
          len(jr) == fold_count, detail=f"got {len(jr)}")

    print(f"\n--- Pooled prediction files (n = {n_samples} binary samples, "
          f"{n_cohorts} cohorts) ---")
    for entry in m["prediction_files"]:
        pf = entry["path"]
        expected_n = int(entry["expected_n"])
        if not os.path.exists(pf):
            check(f"{os.path.basename(pf)} exists", False, detail="file missing")
            continue
        df = pd.read_csv(pf)
        check(f"{os.path.basename(pf)} sample count = {expected_n}",
              len(df) == expected_n, detail=f"got {len(df)}")
        check(f"{os.path.basename(pf)} sample_id unique",
              df["sample_id"].is_unique,
              detail=f"{len(df)} rows / {df['sample_id'].nunique()} unique IDs")
        n_c = df["cohort"].nunique()
        check(f"{os.path.basename(pf)} covers {n_cohorts} cohorts",
              n_c == n_cohorts, detail=f"got {n_c}")

    print(f"\n--- DeLong significance on pooled LODO predictions "
          f"(n={n_samples}) ---")
    dl = pd.read_csv("results/delong_results.csv")
    row_sj = dl[(dl.model_a == "species_rf") & (dl.model_b == "joint_rf")].iloc[0]
    row_sx = dl[(dl.model_a == "species_rf") & (dl.model_b == "joint_xgb")].iloc[0]
    check(f"DeLong n_samples = {n_samples}",
          int(row_sj["n_samples"]) == n_samples,
          detail=f"got {row_sj['n_samples']}")
    dj = m["delong"]["species_rf_vs_joint_rf"]
    check_near("DeLong species_rf vs joint_rf z",
               row_sj["z"], dj["z"], dj["z_tol"])
    check_near("DeLong species_rf vs joint_rf p",
               row_sj["p_value"], dj["p"], dj["p_tol"])
    dx = m["delong"]["species_rf_vs_joint_xgb"]
    check_near("DeLong species_rf vs joint_xgb z",
               row_sx["z"], dx["z"], dx["z_tol"])
    check_near("DeLong species_rf vs joint_xgb p",
               row_sx["p_value"], dx["p"], dx["p_tol"])

    print(f"\n--- Bootstrap CI (species RF pooled, {n_cohorts}-cohort) ---")
    bc = pd.read_csv("results/bootstrap_ci.csv")
    sp_pool = bc[(bc.model == "species_rf") & (bc.cohort == "pooled")].iloc[0]
    v, t = _vt(m["bootstrap_species_rf_pooled"]["auc"])
    check_near("Bootstrap pooled species_rf AUC", sp_pool["auc"], v, t)
    v, t = _vt(m["bootstrap_species_rf_pooled"]["ci_lo"])
    check_near("Bootstrap pooled species_rf CI lower", sp_pool["ci_lo"], v, t)
    v, t = _vt(m["bootstrap_species_rf_pooled"]["ci_hi"])
    check_near("Bootstrap pooled species_rf CI upper", sp_pool["ci_hi"], v, t)
    check(f"Bootstrap n_samples = {n_samples}",
          int(sp_pool["n"]) == n_samples, detail=f"got {sp_pool['n']}")

    print("\n--- Adenoma LODO (4-cohort, YachidaS_2019 included, bugs fixed) ---")
    al = pd.read_csv("results/adenoma_lodo_results.csv")
    expected_tasks = set(m["adenoma"]["tasks"])
    actual_tasks = set(al["task"])
    check("Adenoma LODO has all 4 task rows",
          expected_tasks == actual_tasks,
          detail=f"missing {expected_tasks - actual_tasks}")
    expected_folds = int(m["adenoma"]["n_folds_per_task"])
    for task in expected_tasks:
        if task in actual_tasks:
            n = al[al["task"] == task]["n_folds"].iloc[0]
            check(f"{task} n_folds = {expected_folds}",
                  int(n) == expected_folds, detail=f"got {n}")
    if "h_vs_a_rf" in actual_tasks:
        v, t = _vt(m["adenoma"]["h_vs_a_rf_mean_auc"])
        meanv = al[al["task"] == "h_vs_a_rf"]["mean_lodo_auc"].iloc[0]
        check_near("H-vs-A RF mean LODO AUC (near-chance)", meanv, v, t)
    if "a_vs_crc_rf" in actual_tasks:
        v, t = _vt(m["adenoma"]["a_vs_crc_rf_mean_auc"])
        meanv = al[al["task"] == "a_vs_crc_rf"]["mean_lodo_auc"].iloc[0]
        check_near("A-vs-CRC RF mean LODO AUC", meanv, v, t)

    species_count = int(m["joint_features"]["species_count"])
    pw_min = int(m["joint_features"]["pathway_per_fold_min"])
    pw_max = int(m["joint_features"]["pathway_per_fold_max"])
    print(f"\n--- Per-fold feature counts in joint LODO "
          f"({species_count} species + {pw_min}-{pw_max} pathways) ---")
    if "rf_n_features" in jr.columns:
        pw_per_fold = jr["rf_n_features"] - species_count
        in_range = ((pw_per_fold >= pw_min) & (pw_per_fold <= pw_max)).all()
        check(f"Per-fold pathway count in [{pw_min}, {pw_max}]",
              bool(in_range),
              detail=f"got {sorted(pw_per_fold.tolist())}")

    print(f"\n--- Seed sensitivity (seeds 0,1,2,42,100; {n_cohorts}-cohort) ---")
    if os.path.exists("results/seed_sensitivity.csv"):
        ss = pd.read_csv("results/seed_sensitivity.csv")
        n_rows = int(m["seed_sensitivity"]["n_rows"])
        spread_max = float(m["seed_sensitivity"]["spread_max"])
        check(f"Seed sensitivity has {n_rows} rows",
              len(ss) == n_rows, detail=f"got {len(ss)}")
        if len(ss) >= 2:
            spread = ss["mean_auc"].max() - ss["mean_auc"].min()
            check(f"Seed sensitivity spread < {spread_max}",
                  spread < spread_max, detail=f"got {spread:.4f}")
            v, t = _vt(m["seed_sensitivity"]["grand_mean"])
            check_near("Seed sensitivity grand mean",
                       ss["mean_auc"].mean(), v, t)

    print(f"\n--- Confounder adjustment ({n_cohorts}-cohort) ---")
    if os.path.exists("results/confounder_results.csv"):
        cf = pd.read_csv("results/confounder_results.csv")
        methods = list(m["confounder"]["methods"])
        for method in methods:
            sub = cf[cf["method"] == method]
            check(f"Confounder method '{method}' present",
                  len(sub) == 1, detail=f"got {len(sub)} rows")
        if len(cf) == len(methods):
            auc_range = cf["mean_auc"].max() - cf["mean_auc"].min()
            range_max = float(m["confounder"]["auc_range_max"])
            check(f"Confounder AUC range < {range_max}",
                  auc_range < range_max, detail=f"got {auc_range:.4f}")

    print(f"\n--- Sensitivity grid (per-fold pathway filter, "
          f"{n_cohorts}-cohort) ---")
    if os.path.exists("results/sensitivity_thresholds.csv"):
        st = pd.read_csv("results/sensitivity_thresholds.csv")
        n_cells = int(m["sensitivity_grid"]["n_cells"])
        spread_max = float(m["sensitivity_grid"]["spread_max"])
        check(f"Sensitivity grid has {n_cells} cells",
              len(st) == n_cells, detail=f"got {len(st)}")
        spread = st["mean_auc"].max() - st["mean_auc"].min()
        check(f"Sensitivity grid full spread < {spread_max}",
              spread < spread_max, detail=f"got {spread:.4f}")

    print(f"\n--- Metadata integrity ({n_cohorts}-cohort dataset) ---")
    md = pd.read_csv("data/processed/metadata_clean.csv")
    sc = set(md["study_condition"].unique())
    expected_sc = set(m["dataset"]["study_conditions"])
    check(f"study_condition values = {sorted(expected_sc)}",
          sc == expected_sc, detail=f"got {sorted(sc)}")
    check("'healthy' NOT in study_condition", "healthy" not in sc)
    n_crc = int((md["study_condition"] == "CRC").sum())
    n_ctrl = int((md["study_condition"] == "control").sum())
    n_aden = int((md["study_condition"] == "adenoma").sum())
    exp_crc = int(m["dataset"]["n_crc"])
    exp_ctrl = int(m["dataset"]["n_control"])
    exp_aden = int(m["dataset"]["n_adenoma"])
    check(f"CRC sample count = {exp_crc}",
          n_crc == exp_crc, detail=f"got {n_crc}")
    check(f"Control sample count = {exp_ctrl}",
          n_ctrl == exp_ctrl, detail=f"got {n_ctrl}")
    check(f"Adenoma sample count = {exp_aden}",
          n_aden == exp_aden, detail=f"got {n_aden}")
    n_c = md["study_name"].nunique()
    check(f"{n_cohorts} cohorts in metadata",
          n_c == n_cohorts, detail=f"got {n_c}")
    for excl in m["dataset"]["excluded_cohorts"]:
        check(f"{excl} excluded", excl not in md["study_name"].values)
    for req in m["dataset"]["required_cohorts"]:
        check(f"{req} present", req in md["study_name"].values)

    print(f"\n{'='*40}")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        print("All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
