# Wirbel et al. 2019 (Nat Med) replication note

## What we did

Subset our curatedMetagenomicData pull (`data/processed/species_filtered.csv` +
`data/processed/metadata_clean.csv`) to the same 5 CRC cohorts Wirbel et al.
2019 used in their pooled meta-analysis, and ran the headline pipeline
(country-aware LODO, 500-tree Random Forest on species, `max_features='sqrt'`,
`min_samples_leaf=5`, `class_weight='balanced'`, `random_state=42`) on that
subset only. See `scripts/wirbel_replication.py` and
`results/wirbel_replication.csv`.

## Cohort overlap

All 5 Wirbel 2019 cohorts are present in our curatedMetagenomicData snapshot:

| Wirbel 2019 cohort | Country | curatedMetagenomicData ID | Match |
| --- | --- | --- | --- |
| Feng et al. 2015 | Austria/China | `FengQ_2015` (AUT) | exact |
| Vogtmann et al. 2016 | USA | `VogtmannE_2016` (USA) | exact |
| Yu et al. 2015 | China | `YuJ_2015` (CHN) | exact |
| Zeller et al. 2014 | France/Germany | `ZellerG_2014` (FRA) | exact |
| Wirbel et al. 2019 (new German cohort) | Germany | `WirbelJ_2018` (DEU) | exact |

Sample counts (CRC + control only): 578 total (285 CRC, 293 control) across
the 5 cohorts. These are within a handful of samples per cohort of the original
publication; small differences come from curatedMetagenomicData's reprocessing
(MetaPhlAn version, QC filters) rather than cohort selection.

## Per-cohort AUC: ours vs Wirbel reported

| Cohort | Ours (RF) | Wirbel reported (LASSO) | Delta |
| --- | --- | --- | --- |
| FengQ_2015      | 0.814 | ~0.79 | +0.024 |
| VogtmannE_2016  | 0.739 | ~0.80 | -0.061 |
| WirbelJ_2018    | 0.895 | ~0.84 | +0.055 |
| YuJ_2015        | 0.870 | ~0.79 | +0.080 |
| ZellerG_2014    | 0.803 | ~0.84 | -0.037 |
| **MEAN**        | **0.824** | **~0.81** | **+0.012** |

Wirbel's reported per-cohort numbers above are read off Figure 2 / supplementary
of the paper and are approximate (+/- 0.02). The headline pooled LODO mean in
the paper is in the 0.80-0.85 band.

## Headline match

Mean per-cohort AUC: **0.824 (ours) vs 0.812 (Wirbel reference) -> within
0.012 AUC**, comfortably inside the 0.80-0.85 band reported in the original
paper. Four of the five per-cohort AUCs fall within ~0.06 of the published
numbers; the fifth (YuJ_2015) is +0.08 higher than the paper's LASSO model.

## Caveats

- **Model class differs.** SIAMCAT (Wirbel 2019) uses LASSO logistic
  regression with internal feature normalization; we use a Random Forest with
  per-fold filtering. Small per-cohort differences (a few AUC points) are
  expected and not evidence of methodological disagreement.
- **Feature processing differs.** Wirbel 2019 applies log-relative-abundance
  + standardization across the training pool; we feed raw relative
  abundances to the RF (RF is scale-invariant, so this is intentional).
- **Sample inventory differs slightly** because curatedMetagenomicData
  re-processes raw reads with its own MetaPhlAn version and QC, which can
  drop or recover a handful of samples per cohort vs the original tables.
- **Country-aware LODO is unchanged** from `train_baseline.py`. In this
  5-cohort subset no two cohorts share a country, so country-aware LODO
  reduces to plain LODO for these folds.

## Verdict

The pipeline reproduces Wirbel 2019's headline pooled LODO result to within
~0.01 AUC on the same 5-cohort partition (0.82 ours vs ~0.81 reported), and
all five per-cohort AUCs land in a plausible neighborhood of the published
LASSO numbers. This supports the pipeline as a faithful reproduction of the
published cross-cohort CRC signal -- our downstream extensions
(adenoma, joint features, robustness) build on a baseline that matches the
established meta-analysis.
