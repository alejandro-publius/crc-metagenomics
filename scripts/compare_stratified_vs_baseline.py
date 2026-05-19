"""Compare species-only vs community-joint vs stratified-joint LODO results.

Runs after train_stratified_joint.py completes. Computes:
- Pooled AUC for each of 5 models
- Pairwise DeLong tests (stratified vs species, stratified vs community-joint)
- Cohort-stratified bootstrap CIs for the 2 new models

Output: results/stratified_vs_baseline_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from crc_lodo_bench.stats import delong_test, bootstrap_pooled_ci  # noqa: E402


MODELS = [
    ("species_rf",          "results/preds_species_rf.csv",          "species-only RF"),
    ("joint_rf",            "results/preds_joint_rf.csv",            "joint community-pathway RF"),
    ("joint_xgb",           "results/preds_joint_xgb.csv",           "joint community-pathway XGB"),
    ("stratified_joint_rf", "results/preds_stratified_joint_rf.csv", "joint stratified-pathway RF"),
    ("stratified_joint_xgb","results/preds_stratified_joint_xgb.csv","joint stratified-pathway XGB"),
]


def main():
    rows = []
    preds = {}
    for name, path, desc in MODELS:
        p = REPO / path
        if not p.exists():
            print(f"[SKIP] {name}: {path} missing")
            continue
        df = pd.read_csv(p)
        preds[name] = df
        pooled = roc_auc_score(df["y_true"], df["y_prob"])
        rows.append({"model": name, "desc": desc, "n_samples": len(df), "pooled_auc": pooled})

    summary = pd.DataFrame(rows)
    print("=== Pooled AUC per model ===")
    print(summary.to_string(index=False))

    if "stratified_joint_rf" in preds:
        print("\n=== Bootstrap CIs for new models (10,000 iter, cohort-stratified) ===")
        for new in ("stratified_joint_rf", "stratified_joint_xgb"):
            if new not in preds:
                continue
            d = preds[new]
            ci = bootstrap_pooled_ci(d["y_true"].values, d["y_prob"].values,
                                      d["cohort"].values, n_boot=10000)
            print(f"  {new}: pooled AUC {ci['auc']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] "
                  f"(n={int(ci['n'])}, kept={int(ci['n_boot_kept'])}/10000)")

        print("\n=== Pairwise DeLong tests ===")
        comparisons = [
            ("species_rf", "stratified_joint_rf"),
            ("species_rf", "stratified_joint_xgb"),
            ("joint_rf",   "stratified_joint_rf"),
            ("joint_xgb",  "stratified_joint_xgb"),
        ]
        delong_rows = []
        for a, b in comparisons:
            if a not in preds or b not in preds:
                continue
            da = preds[a].sort_values("sample_id").reset_index(drop=True)
            db = preds[b].sort_values("sample_id").reset_index(drop=True)
            # Verify identical sample sets
            if not da["sample_id"].equals(db["sample_id"]):
                print(f"  [SKIP] {a} vs {b}: sample IDs differ")
                continue
            res = delong_test(da["y_true"].values, da["y_prob"].values, db["y_prob"].values)
            sig = "*" if res["p_value"] < 0.05 else " "
            print(f"  {sig} {a:25s} vs {b:25s}: AUC_a={res['auc_a']:.4f} AUC_b={res['auc_b']:.4f} "
                  f"diff={res['auc_b'] - res['auc_a']:+.4f} z={res['z']:+.3f} p={res['p_value']:.4f}")
            delong_rows.append({
                "model_a": a, "model_b": b,
                "auc_a": res["auc_a"], "auc_b": res["auc_b"],
                "diff": res["auc_b"] - res["auc_a"],
                "z": res["z"], "p_value": res["p_value"],
            })
        if delong_rows:
            pd.DataFrame(delong_rows).to_csv(REPO / "results/delong_stratified_vs_baseline.csv", index=False)

    summary.to_csv(REPO / "results/stratified_vs_baseline_comparison.csv", index=False)
    print(f"\nSaved: results/stratified_vs_baseline_comparison.csv")


if __name__ == "__main__":
    main()
