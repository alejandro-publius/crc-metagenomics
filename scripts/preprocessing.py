import pandas as pd, numpy as np, os

# ── Quality filters (documented, applied before any modeling) ─────────────────
# Per-sample: minimum reads for reliable MetaPhlAn species profiling.
MIN_READS = 1_000_000

# Per-cohort: HanniganGD_2017 excluded due to substantially lower sequencing
# depth (mean 6.5M reads, range 17K–21M) compared to all other cohorts
# (mean 40–102M reads), and consequent high feature sparsity (82% zero-valued
# species features vs. 61% mean across other cohorts). Both metrics were
# assessed before model training; the exclusion is pre-specified and
# independent of classification results.
EXCLUDE_COHORTS = ['HanniganGD_2017']

def main():
    species = pd.read_csv('data/raw/species_abundance.csv')
    metadata = pd.read_csv('data/raw/metadata.csv')

    # ── 1. Per-sample depth filter ────────────────────────────────────────────
    if 'number_reads' in metadata.columns:
        before = len(metadata)
        metadata = metadata[metadata['number_reads'] >= MIN_READS].reset_index(drop=True)
        removed = before - len(metadata)
        print(f'Depth filter (>={MIN_READS/1e6:.0f}M reads/sample): removed {removed} samples')

    # ── 2. Cohort exclusion ───────────────────────────────────────────────────
    before = len(metadata)
    metadata = metadata[~metadata['study_name'].isin(EXCLUDE_COHORTS)].reset_index(drop=True)
    print(f'Cohort exclusion {EXCLUDE_COHORTS}: removed {before - len(metadata)} samples')

    print('\nSamples per cohort after quality filters:')
    print(metadata.groupby(['study_name', 'study_condition']).size()
          .unstack(fill_value=0).to_string())

    # ── 3. Align species and metadata ─────────────────────────────────────────
    common = set(species['sample_id']) & set(metadata['sample_id'])
    species  = species[species['sample_id'].isin(common)].reset_index(drop=True)
    metadata = metadata[metadata['sample_id'].isin(common)].reset_index(drop=True)

    # ── 4. Species feature filter (prevalence + mean abundance) ───────────────
    sid = species['sample_id']
    fc = [c for c in species.columns if c != 'sample_id']
    X = species[fc]
    prev = (X > 0).mean()
    ma = X.mean()
    keep = sorted(set(prev[prev >= 0.10].index) & set(ma[ma >= 1e-4].index))
    print(f'\nSpecies features: {len(fc)} raw -> {len(keep)} retained '
          f'(prevalence>=10%, mean>=1e-4)')
    X = X[keep].copy()

    # ── 5. Normalize and log-transform ────────────────────────────────────────
    rs = X.sum(axis=1)
    if rs.mean() > 1.5:
        X = X.div(rs, axis=0)
    X = np.log10(X + 1e-6)
    X.insert(0, 'sample_id', sid)

    # ── 6. Labels ─────────────────────────────────────────────────────────────
    metadata['label'] = metadata['study_condition'].map(
        {'CRC': 1, 'control': 0, 'adenoma': -1})

    print('\nFinal sample counts:')
    print(metadata['study_condition'].value_counts().to_string())
    print(f'Total: {len(metadata)}')

    # ── 7. Save ───────────────────────────────────────────────────────────────
    os.makedirs('data/processed', exist_ok=True)
    X.to_csv('data/processed/species_filtered.csv', index=False)
    metadata.to_csv('data/processed/metadata_clean.csv', index=False)
    print('\nPreprocessing done')

if __name__ == '__main__':
    main()
