"""Apply a global prevalence/mean filter to the raw HUMAnN pathway matrix.

Reads `data/raw/pathway_abundance.csv`, keeps columns with prevalence
>=10% and mean abundance >=1e-6, and writes two derivative files used by
the (non-LODO) SHAP and adenoma scripts:

- `data/processed/pathway_abundance_filtered.csv` — full filtered matrix
  (both stratified and unstratified columns retained).
- `data/processed/pathway_unstratified.csv` — unstratified pathways only
  (community-level abundance per MetaCyc pathway, no `|` separator).

The main LODO training (`scripts/train_joint.py`) does NOT consume these
files; it refits the prevalence/mean filter per fold on training-cohort
samples only to avoid test-fold leakage. This script's outputs are
explicitly for downstream tools where per-fold filtering is not relevant.
"""
import pandas as pd


def main():
    df = pd.read_csv('data/raw/pathway_abundance.csv')
    print(f'Input: {df.shape[0]} x {df.shape[1]-1}')

    cols = [c for c in df.columns if c != 'sample_id']
    X = df[cols]

    prevalence = (X > 0).mean(axis=0)
    keep_prev = prevalence >= 0.10

    mean_abund = X.mean(axis=0)
    keep_abund = mean_abund >= 1e-6

    keep = keep_prev & keep_abund
    kept_cols = [c for c, k in zip(cols, keep) if k]
    print(f'After prevalence>=10% and mean>=1e-6: {len(kept_cols)} columns')

    unstrat = [c for c in kept_cols if '|' not in c]
    strat = [c for c in kept_cols if '|' in c]
    print(f'  Unstratified: {len(unstrat)}')
    print(f'  Stratified:   {len(strat)}')

    out = df[['sample_id'] + kept_cols]
    out.to_csv('data/processed/pathway_abundance_filtered.csv', index=False)
    print(f'Saved data/processed/pathway_abundance_filtered.csv ({out.shape})')

    unstrat_only = df[['sample_id'] + unstrat]
    unstrat_only.to_csv('data/processed/pathway_unstratified.csv', index=False)
    print(f'Saved data/processed/pathway_unstratified.csv ({unstrat_only.shape})')


if __name__ == "__main__":
    main()
