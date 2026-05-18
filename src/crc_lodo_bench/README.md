# crc_lodo_bench

Country-aware leave-one-dataset-out (LODO) benchmarking utilities for
shotgun metagenomic CRC classification.

This is the reusable library distilled from the crc-metagenomics study
(Velazquez and Selbrede, 2026). It exposes three things:

1. `run_lodo_cv` / `get_lodo_splits` — country-aware LODO loop and
   split generator. When `country_col` is supplied, cohorts that share
   the held-out cohort's country are removed from the training fold so
   population-level signals cannot leak into the classifier.
2. `per_fold_pathway_filter` — factory that builds a per-fold
   prevalence + mean abundance feature filter. Recomputed on
   training-cohort samples inside each fold, preventing the
   test-fold-leakage that a global pre-fold filter would cause.
3. `delong_test` / `bootstrap_pooled_ci` — paired AUC comparison via
   the Sun and Xu (2014) DeLong fast algorithm, and a
   cohort-stratified bootstrap 95% CI for pooled LODO AUCs.

## Install

```bash
pip install -e .            # from a checkout of crc-metagenomics
# or, once published:
pip install crc-lodo-bench
```

Optional extras:

```bash
pip install -e ".[adenoma]"   # adds imbalanced-learn (SMOTE)
pip install -e ".[test]"      # adds pytest
```

## Minimal usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from crc_lodo_bench import (
    run_lodo_cv,
    per_fold_pathway_filter,
    delong_test,
    bootstrap_pooled_ci,
)

# Load your own shotgun-metagenomic data. The contract is:
#   X        : (n_samples, n_features) DataFrame indexed 0..n-1
#   y        : (n_samples,) Series of 0/1 labels
#   metadata : DataFrame with at least 'study_name' and (optional) 'country'
X = pd.read_csv("my_features.csv")
y = pd.read_csv("my_labels.csv")["label"]
metadata = pd.read_csv("my_metadata.csv")

# Always-keep species columns, recompute pathway filter inside each fold.
species_cols = [c for c in X.columns if c.startswith("species__")]
pathway_cols = [c for c in X.columns if c.startswith("pathway__")]

filt = per_fold_pathway_filter(
    filtered_cols=pathway_cols,
    prevalence_threshold=0.10,
    mean_threshold=1e-6,
    passthrough_cols=species_cols,
)

def make_rf():
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, n_jobs=-1,
        random_state=42, class_weight="balanced",
    )

results = run_lodo_cv(
    make_rf, X, y, metadata,
    cohort_col="study_name",
    country_col="country",          # enables country-aware LODO
    feature_filter_fn=filt,
    save_predictions_path="preds.csv",
)
print("Mean per-cohort AUC:", results["mean_auc"])

# Pooled CI and head-to-head DeLong vs a baseline.
preds = pd.read_csv("preds.csv")
ci = bootstrap_pooled_ci(
    preds["y_true"].values, preds["y_prob"].values, preds["cohort"].values,
)
print("Pooled AUC:", ci["auc"], "95% CI:", (ci["ci_low"], ci["ci_high"]))

baseline = pd.read_csv("baseline_preds.csv").sort_values("sample_id")
joint    = preds.sort_values("sample_id")
assert (baseline["sample_id"].values == joint["sample_id"].values).all()
dl = delong_test(
    baseline["y_true"].values,
    baseline["y_prob"].values,
    joint["y_prob"].values,
)
print(f"DeLong: dAUC={dl['auc_diff']:+.3f}  z={dl['z']:+.2f}  p={dl['p_value']:.4f}")
```

## What the country-aware LODO actually does

For each cohort `C` in `metadata['study_name']`:

1. Find the majority value of `metadata['country']` for cohort `C`.
2. Drop every cohort whose majority country equals `C`'s country from
   the training fold (except `C` itself, which is the test fold).
3. Train on the remainder, evaluate on `C`.

Without step 2, two cohorts from the same country can act as
near-duplicates across the train/test boundary and produce
optimistically biased AUCs (in our study, this exact pattern gave a
spurious AUC of 0.999 for a Japanese cohort; the corrected AUC was
0.836).

## Citation

If you use this package, please cite the underlying study:

> Velazquez A, Selbrede R. Species-level taxonomic features alone
> outperform joint species-plus-pathway models for colorectal cancer
> detection. 2026.

See `CITATION.cff` and `.zenodo.json` at the repository root for
machine-readable metadata.

## License

MIT. See `LICENSE` at the repository root.
