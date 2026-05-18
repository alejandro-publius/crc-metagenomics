"""Biologically-guided pathway shortlist for CRC LODO analysis.

Instead of all 551 unstratified pathway candidates, selects a curated subset
based on CRC-relevant biology: butyrate/SCFA production, fermentation,
LPS/peptidoglycan and inflammation, polyamine synthesis, tryptophan
metabolism, folate/one-carbon metabolism, sulfur/methionine metabolism,
and glycan/mucin degradation (eight groups; 84 unique pathways after
deduplicating cross-group hits such as CENTFERM-PWY).

Rationale (key references in manuscript):
- Butyrate-producing bacteria (Roseburia, Faecalibacterium) are consistently
  depleted in CRC; butyrate is a key colonocyte energy source and
  anti-tumorigenic. Note that Fusobacterium nucleatum, despite being a CRC
  pathobiont, is itself a butyrate-producer, so the depletion signal comes
  from the wider commensal community rather than from removal of one taxon.
- Polyamines (spermine, spermidine, putrescine) promote tumor cell proliferation
  and are elevated in CRC.
- LPS from gram-negative species and peptidoglycan turnover drive colonic
  inflammation; Fusobacterium nucleatum signals through FadA / Fap2 and is
  associated with NF-kB activation in CRC progression.
- Folate/one-carbon metabolism affects DNA methylation and nucleotide synthesis,
  altered in CRC.
- Tryptophan/indole metabolites modulate AhR signaling and immune responses
  in the tumor microenvironment.
- Sulfur amino acid metabolism generates H2S, a colonocyte genotoxin elevated
  in CRC.
- Glycan/mucin-degrading pathways reflect mucus-layer remodeling that
  precedes epithelial barrier disruption.

Usage:
    python3 scripts/bio_pathway_shortlist.py

Output:
    results/bio_pathway_results.csv  (LODO AUCs per cohort)
    results/bio_pathway_shortlist.txt (pathway names selected)
"""
import re, os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import run_lodo_cv

# ── Biological keyword groups ─────────────────────────────────────────────────
BIO_KEYWORDS = {
    'butyrate_SCFA':   ['butanoat', 'butyrat', 'propanoat', 'propionat',
                        'acetate', 'short.chain', 'SCFA', 'CENTFERM',
                        'PWY-5676', 'PWY-5677', 'P108-PWY', 'PROPFERM'],
    'fermentation':    ['ferment', 'homolactic', 'heterolactic', 'mixed.acid',
                        'FERMENTATION-PWY', 'ANAEROFRUCAT', 'P122-PWY',
                        'P461-PWY', 'PWY-5100', 'PWY-6590', 'PWY-4LZ'],
    'LPS_inflammation':['lipopolysaccharide', 'LPSSYN', 'LPS', 'O-antigen',
                        'lipid.A', 'peptidoglycan', 'UDP-N-acetylmuramoyl'],
    'polyamine':       ['polyamine', 'putrescin', 'spermin', 'spermidine',
                        'norspermidine', 'POLYAMSYN', 'POLYAMINSYN',
                        'ARG.POLYAMINE', 'PWY-6305', 'PWY-6562'],
    'tryptophan':      ['tryptophan', 'TRPSYN', 'indole', 'PWY-6629'],
    'folate_onecarbon':['folate', 'tetrahydrofolate', 'FOLSYN', 'PWY-6612',
                        'PWY-3841', '1CMET2', 'N10-formyl', 'methyl.THF'],
    'sulfur_methionine':['methionine', 'homocysteine', 'transsulfuration',
                         'SAM', 'S-adenosyl', 'sulfur.amino', 'sulfate',
                         'cysteine', 'PWY-5347', 'PWY-821', 'MET-SAM',
                         'HSERMETANA', 'METSYN', 'HOMOSER-METSYN',
                         'SULFATE-CYS', 'SO4ASSIM', 'PWY-6151'],
    'glycan_mucin':    ['N-acetylglucosamine', 'N-acetylneuraminate',
                        'sialyl', 'fucosyl', 'mucin', 'glycan',
                        'GLCMANNANAUT', 'P441-PWY', 'UDPNAGSYN'],
}


def select_pathways(pathway_cols):
    """Return (selected_cols, group_counts) using keyword matching."""
    selected = set()
    group_counts = {}
    for group, keywords in BIO_KEYWORDS.items():
        hits = [c for c in pathway_cols
                if any(kw.lower() in c.lower() for kw in keywords)]
        group_counts[group] = len(hits)
        selected.update(hits)
    return sorted(selected), group_counts


def main():
    sp     = pd.read_csv('data/processed/species_filtered.csv')
    pw_raw = pd.read_csv('data/raw/pathway_abundance.csv')
    md     = pd.read_csv('data/processed/metadata_clean.csv')

    unstrat_cols = [c for c in pw_raw.columns
                    if c != 'sample_id' and '|' not in c]
    print(f'Total unstratified pathway candidates: {len(unstrat_cols)}')

    selected, group_counts = select_pathways(unstrat_cols)
    print(f'Bio-guided shortlist: {len(selected)} pathways')
    print('\nPathways per biological group:')
    for grp, n in group_counts.items():
        print(f'  {grp:25s}  {n}')

    # Save the shortlist for transparency
    os.makedirs('results', exist_ok=True)
    with open('results/bio_pathway_shortlist.txt', 'w') as f:
        for grp, kws in BIO_KEYWORDS.items():
            hits = [c for c in unstrat_cols
                    if any(kw.lower() in c.lower() for kw in kws)]
            f.write(f'\n# {grp} ({len(hits)} pathways)\n')
            for h in sorted(hits):
                f.write(f'{h}\n')
    print('Saved results/bio_pathway_shortlist.txt')

    pw_sel = pw_raw[['sample_id'] + selected]
    joint  = sp.merge(pw_sel, on='sample_id', suffixes=('_sp', '_pw'))
    mg     = md.merge(joint, on='sample_id', how='inner')
    fc     = [c for c in joint.columns if c != 'sample_id']

    mask = mg['label'].isin([0, 1])
    X    = mg.loc[mask, fc].reset_index(drop=True)
    X.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X.columns]
    y    = mg.loc[mask, 'label'].reset_index(drop=True)

    meta_cols = ['sample_id', 'study_name', 'study_condition', 'label']
    if 'country' in mg.columns:
        meta_cols.append('country')
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)
    country_col = 'country' if 'country' in meta.columns else None

    sp_feat_cols = [re.sub(r'[\[\]<>]', '_', c)
                    for c in sp.columns if c != 'sample_id']
    pw_feat_cols = [re.sub(r'[\[\]<>]', '_', c) for c in selected]

    def pathway_filter(X_train):
        Xpw = X_train[pw_feat_cols]
        prev = (Xpw > 0).mean(axis=0)
        ma   = Xpw.mean(axis=0)
        keep_pw = [c for c in pw_feat_cols
                   if prev[c] >= 0.10 and ma[c] >= 1e-6]
        return sp_feat_cols + keep_pw

    print(f'\nSamples: {len(X)}  '
          f'(species={len(sp_feat_cols)}, bio pathways={len(pw_feat_cols)})')
    print('\n=== Bio-guided pathway RF (country-aware LODO) ===')

    def make_rf():
        return RandomForestClassifier(
            n_estimators=500, max_features='sqrt', min_samples_leaf=5,
            n_jobs=-1, random_state=42, class_weight='balanced')

    res = run_lodo_cv(make_rf, X, y, meta,
                      save_predictions_path='results/preds_bio_pathway_rf.csv',
                      feature_filter_fn=pathway_filter,
                      country_col=country_col)

    # Compare to baseline
    bl = pd.read_csv('results/baseline_results.csv').sort_values('cohort')
    print('\n=== Per-cohort comparison: Species RF vs Bio-pathway RF ===')
    res_df = pd.DataFrame({
        'cohort':    res['cohort'],
        'bio_pw_auc': res['auc'],
        'n_features': res['n_features'],
    }).sort_values('cohort')
    merged = bl.merge(res_df, on='cohort')
    for _, r in merged.iterrows():
        diff = r['bio_pw_auc'] - r['auc']
        print(f'  {r["cohort"]:25s}  species={r["auc"]:.3f}  '
              f'bio_pw={r["bio_pw_auc"]:.3f}  diff={diff:+.3f}')
    print(f'\n  Species RF mean:  {bl["auc"].mean():.3f}')
    print(f'  Bio-pathway mean: {res["mean_auc"]:.3f}  '
          f'(diff={res["mean_auc"]-bl["auc"].mean():+.3f})')

    res_df.to_csv('results/bio_pathway_results.csv', index=False)
    print('\nSaved results/bio_pathway_results.csv')


if __name__ == '__main__':
    main()
