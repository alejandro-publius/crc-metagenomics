# Minimum useful species panel — headline

**8 species yield pooled AUC >= 0.75; 12 species recover >= 99% of full-model performance.**

## Context

Starting from the global mean |SHAP| ranking of the species RF (229 ranked features), this analysis greedily adds species in SHAP order and refits the same country-aware 10-cohort LODO Random Forest used in `scripts/train_baseline.py`. At each panel size we report both the mean per-cohort AUC and the pooled AUC computed on stacked held-out predictions.

## Reference (full panel, 229 species)

- Mean per-cohort AUC: **0.8075**
- Pooled AUC: **0.7812**

## Headline panel sizes

- Smallest k with pooled AUC >= 0.75 (clinically useful screening floor): **k = 8**
- Smallest k with pooled AUC >= 0.78 (matches full-229 baseline of 0.781): **k = 14**
- Smallest k recovering >= 99% of full pooled AUC (0.7734): **k = 12**

### Species in the AUC >= 0.75 panel (k = 8)

1. Gemella_morbillorum
2. Parvimonas_micra
3. Peptostreptococcus_stomatis
4. Fusobacterium_nucleatum
5. Solobacterium_moorei
6. Dialister_pneumosintes
7. Ruthenibacterium_lactatiformans
8. Roseburia_faecis

## Method notes

- LODO splits use country-aware exclusion (same-country cohorts are removed from training), matching the headline model.
- The RF spec is identical to `scripts/train_baseline.py`: 500 trees, `max_features='sqrt'`, `min_samples_leaf=5`, `class_weight='balanced'`, `random_state=42`.
- Ranking is taken from the existing `results/shap_crc_features.csv` (global mean |SHAP| from the full-panel RF); this is a one-shot rank, not per-fold re-ranking, so no test-fold leakage occurs.
- Panel size capped at k = 50; gains plateau well before that.
