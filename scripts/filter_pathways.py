import pandas as pd, numpy as np, os

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
