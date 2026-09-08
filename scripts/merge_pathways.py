"""Concatenate per-cohort pathway CSVs from data/raw/pathway_chunks/
into a single data/raw/pathway_abundance.csv (full matrix with both
stratified and unstratified pathway columns). Also writes a sidecar
data/raw/pathway_unstratified_full.csv that drops the stratified
(``taxon|pathway``) columns for downstream sensitivity_analysis.py.
Run after the R export script and before filter_pathways.py.

data/raw/pathway_chunks/ ships HUMAnN tables for a subset of cohorts
only (the rest are omitted from git for size; see README/REPRODUCING.md).
This script warns when the chunks it finds cover fewer cohorts than
data/raw/metadata.csv expects, and refuses to silently regress an
already-more-complete committed data/raw/pathway_unstratified_full.csv
with a smaller partial merge (pass --force to override)."""
import glob
import os
import sys

import pandas as pd


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    force = '--force' in argv

    chunk_dir = 'data/raw/pathway_chunks'
    chunks = sorted(glob.glob(os.path.join(chunk_dir, '*.csv')))

    if not chunks:
        print(f'ERROR: no CSVs found in {chunk_dir}/')
        print('Run scripts/export_data.R first to produce per-cohort chunks.')
        sys.exit(1)

    print(f'Found {len(chunks)} chunk files:')
    for c in chunks:
        print(f'  {os.path.basename(c)}')

    dfs = [pd.read_csv(c) for c in chunks]
    merged = pd.concat(dfs, ignore_index=True, sort=False).fillna(0)

    cols = ['sample_id'] + [c for c in merged.columns if c != 'sample_id']
    merged = merged[cols]

    # Report cohort coverage against the stable, always-committed cohort
    # list in metadata.csv, so a partial chunk set is loud rather than a
    # silent downstream surprise (e.g. train_joint.py quietly training on
    # fewer cohorts than the species-only baseline).
    metadata_path = 'data/raw/metadata.csv'
    if os.path.isfile(metadata_path):
        chunk_cohorts = {os.path.splitext(os.path.basename(c))[0] for c in chunks}
        expected_cohorts = set(pd.read_csv(metadata_path)['study_name'].unique())
        missing = sorted(expected_cohorts - chunk_cohorts)
        if missing:
            print(
                f'\nWARNING: {chunk_dir}/ covers {len(chunk_cohorts)} of '
                f'{len(expected_cohorts)} cohorts in {metadata_path}; missing '
                f'pathway chunks for: {", ".join(missing)}. Downstream joint '
                '(species+pathway) models will only cover the cohorts above '
                'unless you re-run `Rscript scripts/export_data.R` with '
                'network access to curatedMetagenomicData to pull the full '
                'chunk set first.'
            )

    out_unstrat = 'data/raw/pathway_unstratified_full.csv'
    if os.path.isfile(out_unstrat) and not force:
        existing_rows = sum(1 for _ in open(out_unstrat)) - 1  # minus header
        if existing_rows > len(merged):
            print(
                f'\nERROR: {out_unstrat} already has {existing_rows} sample '
                f"rows; today's merge of {len(chunks)} chunk file(s) would "
                f'only produce {len(merged)} rows and OVERWRITE it with '
                'less cohort coverage than what is already committed. '
                'Re-run `Rscript scripts/export_data.R` with network access '
                'to curatedMetagenomicData to regenerate the full chunk set '
                'first, or re-run this script with --force to overwrite '
                'intentionally.'
            )
            sys.exit(1)

    out = 'data/raw/pathway_abundance.csv'
    merged.to_csv(out, index=False)
    print(f'\nMerged {sum(len(d) for d in dfs)} rows from {len(chunks)} chunks')
    print(f'Output shape: {merged.shape[0]} samples x {merged.shape[1]-1} pathways')
    print(f'Saved {out}')

    # Also produce an unstratified-only subset for sensitivity_analysis.py
    unstrat_cols = ['sample_id'] + [c for c in merged.columns
                                    if c != 'sample_id' and '|' not in c]
    unstrat = merged[unstrat_cols]
    unstrat.to_csv(out_unstrat, index=False)
    print(f'Unstratified subset: {unstrat.shape[0]} x {unstrat.shape[1]-1} pathways')
    print(f'Saved {out_unstrat}')


if __name__ == '__main__':
    main()
