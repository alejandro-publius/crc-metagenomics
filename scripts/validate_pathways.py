import pandas as pd, numpy as np, os, sys

PATH = 'data/raw/pathway_abundance.csv'
SPECIES = 'data/raw/species_abundance.csv'

if not os.path.exists(PATH):
    sys.exit(f'ERROR: {PATH} not found')

df = pd.read_csv(PATH)
print(f'Shape: {df.shape[0]} samples x {df.shape[1]-1} columns')

cols = [c for c in df.columns if c != 'sample_id']
stratified = [c for c in cols if '|' in c]
unstratified = [c for c in cols if '|' not in c]
print(f'Stratified (taxon-level):   {len(stratified)}')
print(f'Unstratified (pathway sum): {len(unstratified)}')

unique_pwy = set(c.split('|')[0] for c in cols)
print(f'Unique pathway IDs:         {len(unique_pwy)}')

X = df[cols]
zero_frac = (X == 0).mean(axis=0)
print(f'\nColumns >=99% zero: {(zero_frac >= 0.99).sum()}')
print(f'Columns >=95% zero: {(zero_frac >= 0.95).sum()}')
print(f'Columns >=50% zero: {(zero_frac >= 0.50).sum()}')

bad = [c for c in cols if ',' in c or '"' in c]
print(f'\nColumn names with comma or quote: {len(bad)}')
if bad:
    print('  Examples:', bad[:5])

if os.path.exists(SPECIES):
    sp = pd.read_csv(SPECIES)
    pw_ids = set(df['sample_id'])
    sp_ids = set(sp['sample_id']) if 'sample_id' in sp.columns else set(sp.iloc[:,0])
    print(f'\nSpecies samples: {len(sp_ids)}')
    print(f'Pathway samples: {len(pw_ids)}')
    print(f'In both:         {len(pw_ids & sp_ids)}')
    print(f'Pathway only:    {len(pw_ids - sp_ids)}')
    print(f'Species only:    {len(sp_ids - pw_ids)}')
else:
    print(f'\n(species file not at {SPECIES}, skipping ID overlap check)')
