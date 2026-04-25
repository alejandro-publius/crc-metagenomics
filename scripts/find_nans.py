import pandas as pd
sp = pd.read_csv('data/processed/species_filtered.csv')
pw = pd.read_csv('data/processed/pathway_unstratified.csv')
md = pd.read_csv('data/processed/metadata_clean.csv')
joint = sp.merge(pw, on='sample_id', suffixes=('_sp','_pw'))
mg = md.merge(joint, on='sample_id', how='inner')

feats = mg.drop(columns=['sample_id','study_name','study_condition','label'])
nan_per_col = feats.isna().sum()
print('cols with NaN:')
print(nan_per_col[nan_per_col > 0])

print('\nNaN counts per source file:')
print(f'  species: {sp.drop(columns=["sample_id"]).isna().sum().sum()}')
print(f'  pathway: {pw.drop(columns=["sample_id"]).isna().sum().sum()}')
print(f'  metadata (non-label): {md.drop(columns=["sample_id","label"]).isna().sum().sum()}')
