import pandas as pd, numpy as np, os

def main():
    pw = pd.read_csv('data/raw/pathway_abundance.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')
    common = set(pw['sample_id']) & set(md['sample_id'])
    pw = pw[pw['sample_id'].isin(common)].reset_index(drop=True)
    print(f'Samples: {len(common)}')
    sid = pw['sample_id']
    fc = [c for c in pw.columns if c != 'sample_id']
    X = pw[fc]
    prev = (X > 0).mean()
    ma = X.mean()
    keep = sorted(set(prev[prev>=0.10].index) & set(ma[ma>=1e-4].index))
    print(f'Pathways: {len(fc)} -> {len(keep)}')
    X = X[keep].copy()
    rs = X.sum(axis=1)
    if rs.mean() > 1.5:
        X = X.div(rs, axis=0)
    X = np.log10(X + 1e-6)
    X.insert(0, 'sample_id', sid)
    os.makedirs('data/processed', exist_ok=True)
    X.to_csv('data/processed/pathway_filtered.csv', index=False)
    print('Saved pathway_filtered.csv')

if __name__ == '__main__':
    main()
