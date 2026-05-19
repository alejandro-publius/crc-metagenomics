"""Apply a global prevalence/mean filter to the raw HUMAnN pathway matrix.

Reads `data/raw/pathway_abundance.csv`, keeps columns with prevalence
>=10% and mean abundance >=1e-6, and writes two derivative files used by
the (non-LODO) SHAP and adenoma scripts:

- `data/processed/pathway_abundance_filtered.csv` -- full filtered matrix
  (both stratified and unstratified columns retained).
- `data/processed/pathway_unstratified.csv` -- unstratified pathways only
  (community-level abundance per MetaCyc pathway, no `|` separator).

The HUMAnN housekeeping totals `UNMAPPED` and `UNINTEGRATED` are dropped
explicitly: they are not MetaCyc pathway features and should never be
fed to the classifier.

Rows are restricted to the canonical sample set in
`data/processed/metadata_clean.csv`, which already replays the
`EXCLUDE_COHORTS` (HanniganGD_2017) and sub-1M-read sample filters
from `scripts/preprocessing.py`. Without this row alignment, excluded
cohorts/samples present in the raw pathway matrix would silently leak
into downstream analyses.

The main LODO training (`scripts/train_joint.py`) does NOT consume these
files; it refits the prevalence/mean filter per fold on training-cohort
samples only to avoid test-fold leakage. This script's outputs are
explicitly for downstream tools where per-fold filtering is not relevant.
"""
import pandas as pd


# HUMAnN housekeeping rows that appear as pseudo-pathways in the abundance
# table. They sum unmapped / unintegrated reads and must not be treated as
# biological MetaCyc pathway features.
HUMANN_HOUSEKEEPING = ['UNMAPPED', 'UNINTEGRATED']


def main():
    df = pd.read_csv('data/raw/pathway_abundance.csv')
    rows_before = df.shape[0]
    print(f'Input: {rows_before} x {df.shape[1]-1}')

    # Drop UNMAPPED / UNINTEGRATED housekeeping columns (and any
    # UNINTEGRATED|... stratified variants) before any further filtering.
    housekeeping_cols = [
        c for c in df.columns
        if c in HUMANN_HOUSEKEEPING
        or any(c.startswith(f'{h}|') for h in HUMANN_HOUSEKEEPING)
    ]
    if housekeeping_cols:
        df = df.drop(columns=housekeeping_cols)
        print(
            f'Dropped {len(housekeeping_cols)} HUMAnN housekeeping columns '
            f'(UNMAPPED / UNINTEGRATED and stratified variants)'
        )

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

    # Row alignment: restrict to canonical sample set in metadata_clean.csv.
    # This replays the EXCLUDE_COHORTS (HanniganGD_2017) and sub-1M-read
    # sample filters applied by scripts/preprocessing.py, preventing
    # excluded cohorts/samples from leaking through this script.
    metadata = pd.read_csv('data/processed/metadata_clean.csv')
    canonical_samples = set(metadata['sample_id'])

    df_aligned = df[df['sample_id'].isin(canonical_samples)].reset_index(drop=True)
    rows_after = df_aligned.shape[0]
    print(
        f'Row alignment to metadata_clean.csv: {rows_before} -> {rows_after} '
        f'({rows_before - rows_after} dropped)'
    )

    out = df_aligned[['sample_id'] + kept_cols]
    out.to_csv('data/processed/pathway_abundance_filtered.csv', index=False)
    print(f'Saved data/processed/pathway_abundance_filtered.csv ({out.shape})')

    unstrat_only = df_aligned[['sample_id'] + unstrat]
    unstrat_only.to_csv('data/processed/pathway_unstratified.csv', index=False)
    print(f'Saved data/processed/pathway_unstratified.csv ({unstrat_only.shape})')


if __name__ == "__main__":
    main()
