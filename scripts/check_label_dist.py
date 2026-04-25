import pandas as pd
sp = pd.read_csv('data/processed/species_filtered.csv')
pw = pd.read_csv('data/processed/pathway_unstratified.csv')
md = pd.read_csv('data/processed/metadata_clean.csv')
joint = sp.merge(pw, on='sample_id')
mg = md.merge(joint, on='sample_id', how='inner')
mask = mg['label'].isin([0,1])

print('Label distribution per cohort (joint training set):')
print(mg[mask].groupby('study_name')['label'].value_counts().unstack(fill_value=0))

print('\nStudy_condition per cohort (full set incl. adenoma):')
print(mg.groupby(['study_name','study_condition']).size().unstack(fill_value=0))

print('\nbaseline_results.csv cohort order:')
print(pd.read_csv('results/baseline_results.csv')['cohort'].tolist())

print('joint_results.csv cohort order:')
print(pd.read_csv('results/joint_results.csv')['cohort'].tolist())
