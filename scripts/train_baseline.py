"""Species-only Random Forest under country-aware LODO (headline model).

Reads `data/processed/species_filtered.csv` and
`data/processed/metadata_clean.csv`, restricts to the CRC/control
binary subset, and runs country-aware LODO with a 500-tree RF
(`max_features='sqrt'`, `min_samples_leaf=5`,
`class_weight='balanced'`, `random_state=42`). Writes per-fold AUCs to
`results/baseline_results.csv` and per-sample held-out probabilities to
`results/preds_species_rf.csv`. Expected per-cohort mean AUC ~0.807;
pooled bootstrap AUC ~0.781.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv  # noqa: E402


SPECIES_PATH = "data/processed/species_filtered.csv"
METADATA_PATH = "data/processed/metadata_clean.csv"


def _read_csv_with_hint(path: str, *, hint: str) -> pd.DataFrame:
    """Read CSV with a friendly error if the file is missing or unreadable."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            f"  Hint: {hint}"
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(
            f"{path} is empty or unreadable. Re-run the preprocessing "
            f"step that produces it."
        ) from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(
            f"{path} appears truncated or corrupt: {e}. Re-generate it "
            f"with the upstream script."
        ) from e
    if len(df) == 0:
        raise ValueError(f"{path} has 0 rows; nothing to train on.")
    return df


def main():
    sp = _read_csv_with_hint(
        SPECIES_PATH,
        hint="run `python3 scripts/preprocessing.py` first to generate "
             "data/processed/species_filtered.csv from the curatedMetagenomicData export.",
    )
    md = _read_csv_with_hint(
        METADATA_PATH,
        hint="run `python3 scripts/preprocessing.py` first to generate "
             "data/processed/metadata_clean.csv from the curatedMetagenomicData export.",
    )

    if "sample_id" not in sp.columns or "sample_id" not in md.columns:
        raise KeyError(
            "Both species and metadata inputs must have a 'sample_id' column; "
            "check the preprocessing step."
        )
    if not md["sample_id"].is_unique:
        n_dup = int(md["sample_id"].duplicated().sum())
        raise ValueError(
            f"metadata_clean.csv has {n_dup} duplicate sample_id(s); "
            f"LODO requires unique IDs."
        )
    if "label" not in md.columns:
        raise KeyError("metadata_clean.csv must have a 'label' column.")

    mg = md.merge(sp, on='sample_id', how='inner')
    fc = [c for c in sp.columns if c != 'sample_id']
    if len(fc) == 0:
        raise ValueError(
            f"{SPECIES_PATH} has no feature columns beyond 'sample_id'."
        )
    mask = mg['label'].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, 'label'].reset_index(drop=True)
    meta_cols = ['sample_id', 'study_name', 'study_condition', 'label']
    if 'country' in mg.columns:
        meta_cols.append('country')
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)
    if len(X) == 0:
        raise ValueError(
            "After restricting to label in {0, 1} there are 0 samples. "
            "Inspect metadata['label'] upstream."
        )
    print(f'Samples: {len(X)} (CRC={int(y.sum())}, control={int((y==0).sum())})')
    print(f'Features: {X.shape[1]}')
    def make_rf():
        return RandomForestClassifier(n_estimators=500, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight='balanced')
    print('Running LODO CV (country-aware)...')
    results = run_lodo_cv(make_rf, X, y, meta,
                          save_predictions_path="results/preds_species_rf.csv",
                          country_col='country' if 'country' in meta.columns else None)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'cohort': results['cohort'], 'auc': results['auc'],
        'n_train': results['n_train'], 'n_test': results['n_test']
    }).to_csv('results/baseline_results.csv', index=False)
    print('Saved to results/baseline_results.csv')

if __name__ == '__main__':
    main()
