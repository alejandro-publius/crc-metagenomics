import pandas as pd, numpy as np, os

def main():
    species = pd.read_csv('data/raw/species_abundance.csv')
    metadata = pd.read_csv('data/raw/metadata.csv')
    common = set(species['sample_id']) & set(metadata['sample_id'])
    species = species[species['sample_id'].isin(common)].reset_index(drop=True)
    metadata = metadata[metadata['sample_id'].isin(common)].reset_index(drop=True)
    sid = species['sample_id']
    fc = [c for c in species.columns if c != 'sample_id']
    X = species[fc]
    prev = (X > 0).mean()
    ma = X.mean()
    keep = sorted(set(prev[prev>=0.10].index) & set(ma[ma>=1e-4].index))
    print(f'Features: {len(fc)} -> {len(keep)}')
    X = X[keep].copy()
    rs = X.sum(axis=1)
    if rs.mean() > 1.5:
        X = X.div(rs, axis=0)
    X = np.log10(X + 1e-6)
    X.insert(0, 'sample_id', sid)
    metadata['label'] = metadata['study_condition'].map({'CRC':1, 'control':0, 'adenoma':-1})
    print(metadata['study_condition'].value_counts())
    os.makedirs('data/processed', exist_ok=True)
    X.to_csv('data/processed/species_filtered.csv', index=False)
    metadata.to_csv('data/processed/metadata_clean.csv', index=False)
    print('Preprocessing done')

if __name__ == '__main__':
    main()
