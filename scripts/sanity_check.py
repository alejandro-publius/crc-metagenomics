"""End-to-end sanity check on shapes, label balance, and merge integrity.

Loads the three processed files (species, pathway, metadata), merges
them, and prints per-stage shapes, label / study_condition value counts,
per-cohort sample counts on the binary subset, duplicate-sample-id
counts, and any NaN counts in the feature matrix. Run after any
regeneration of `data/processed/` files.
"""
import pandas as pd


def main():
    sp = pd.read_csv('data/processed/species_filtered.csv')
    pw = pd.read_csv('data/processed/pathway_unstratified.csv')
    md = pd.read_csv('data/processed/metadata_clean.csv')

    print(f'species: {sp.shape}, pathway: {pw.shape}, metadata: {md.shape}')
    print(f'species sample_ids unique: {sp.sample_id.nunique()}')
    print(f'pathway sample_ids unique: {pw.sample_id.nunique()}')
    print(f'metadata sample_ids unique: {md.sample_id.nunique()}')

    joint = sp.merge(pw, on='sample_id', suffixes=('_sp', '_pw'))
    print(f'after sp+pw merge: {joint.shape}')

    mg = md.merge(joint, on='sample_id', how='inner')
    print(f'after md merge: {mg.shape}')

    print(f'\nlabel value counts:')
    print(mg.label.value_counts(dropna=False))

    print(f'\nstudy_condition value counts:')
    print(mg.study_condition.value_counts(dropna=False))

    print(f'\nper-cohort sample counts (label in [0,1] only):')
    print(mg[mg.label.isin([0, 1])].groupby('study_name').size())

    print(f'\ndupe sample_ids in joint? {mg.sample_id.duplicated().sum()}')
    print(f'NaN counts in features: {mg.drop(columns=["sample_id","study_name","study_condition","label"]).isna().sum().sum()}')


if __name__ == "__main__":
    main()
