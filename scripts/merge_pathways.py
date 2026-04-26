"""Concatenate per-cohort pathway CSVs from data/raw/pathway_chunks/
into a single data/raw/pathway_abundance.csv. Run after the R export
script and before filter_pathways.py."""
import pandas as pd
import glob
import os
import sys

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

out = 'data/raw/pathway_abundance.csv'
merged.to_csv(out, index=False)
print(f'\nMerged {sum(len(d) for d in dfs)} rows from {len(chunks)} chunks')
print(f'Output shape: {merged.shape[0]} samples x {merged.shape[1]-1} pathways')
print(f'Saved {out}')

# Also produce an unstratified-only subset for sensitivity_analysis.py
unstrat_cols = ['sample_id'] + [c for c in merged.columns
                                if c != 'sample_id' and '|' not in c]
unstrat = merged[unstrat_cols]
out_unstrat = 'data/raw/pathway_unstratified_full.csv'
unstrat.to_csv(out_unstrat, index=False)
print(f'Unstratified subset: {unstrat.shape[0]} x {unstrat.shape[1]-1} pathways')
print(f'Saved {out_unstrat}')
