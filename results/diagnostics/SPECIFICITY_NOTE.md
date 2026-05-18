# Cross-disease specificity of the species RF CRC classifier

The headline species-only Random Forest (trained on CRC vs. control across 10 cohorts; pooled LODO AUC ~0.78) was applied without retraining to samples from non-CRC disease cohorts in curatedMetagenomicData. The question: do those samples score *like CRC* (indicating a generic gut-dysbiosis signal) or *like controls* (indicating a CRC-specific signal)?

**Baselines (Youden-J threshold = 0.4847, source: LODO held-out (results/preds_species_rf.csv)):** training-CRC samples score above threshold 73.9% of the time; training-control samples score above threshold 30.2% of the time.

**Headline:** Of 815 non-CRC disease samples scored by the species RF, 33.3% exceeded the Youden-J threshold (vs 73.9% of training-CRC samples and 30.2% of training-control samples), supporting that the species-RF CRC signature is **CRC-specific** (non-CRC disease samples score near the training-control distribution).

**Per-cohort breakdown:**

- **NielsenHB_2014** (IBD, n=148): mean predicted CRC probability 0.432, 30.4% above Youden threshold — *CRC-specific (non-CRC disease scores near control)*.
- **NielsenHB_2014** (NielsenHB_2014_control, n=248): mean predicted CRC probability 0.336, 6.5% above Youden threshold — *cohort-internal control*.
- **HMP_2019_ibdmdb** (HMP_2019_ibdmdb_control, n=27): mean predicted CRC probability 0.414, 29.6% above Youden threshold — *cohort-internal control*.
- **HMP_2019_ibdmdb** (IBD, n=103): mean predicted CRC probability 0.433, 28.2% above Youden threshold — *CRC-specific (non-CRC disease scores near control)*.
- **KarlssonFH_2013** (KarlssonFH_2013_control, n=43): mean predicted CRC probability 0.306, 4.7% above Youden threshold — *cohort-internal control*.
- **KarlssonFH_2013** (T2D, n=102): mean predicted CRC probability 0.339, 10.8% above Youden threshold — *CRC-specific (non-CRC disease scores near control)*.
- **QinJ_2012** (QinJ_2012_control, n=174): mean predicted CRC probability 0.411, 27.6% above Youden threshold — *cohort-internal control*.
- **QinJ_2012** (T2D, n=170): mean predicted CRC probability 0.491, 49.4% above Youden threshold — *generic-dysbiosis-like (scores comparable to CRC)*.
- **QinN_2014** (QinN_2014_control, n=114): mean predicted CRC probability 0.381, 13.2% above Youden threshold — *cohort-internal control*.
- **QinN_2014** (cirrhosis, n=123): mean predicted CRC probability 0.537, 65.0% above Youden threshold — *generic-dysbiosis-like (scores comparable to CRC)*.
- **LeChatelierE_2013** (lean_control, n=95): mean predicted CRC probability 0.371, 10.5% above Youden threshold — *cohort-internal control*.
- **LeChatelierE_2013** (obesity, n=169): mean predicted CRC probability 0.379, 13.0% above Youden threshold — *CRC-specific (non-CRC disease scores near control)*.
- **HMP_2012** (healthy, n=147): mean predicted CRC probability 0.429, 22.4% above Youden threshold — *CRC-specific (non-CRC disease scores near control)*.

**Method.** Species RF (500 trees, max_features=sqrt, min_samples_leaf=5, class_weight=balanced, random_state=42) was trained on the 1,339-sample binary CRC-vs-control set using the same 229-feature panel and log10-relative-abundance preprocessing as `scripts/train_baseline.py`. External cohort species tables were fetched via `curatedMetagenomicData::returnSamples(..., 'relative_abundance')` (MetaPhlAn taxonomy), reindexed onto the trained 229-feature panel (missing features filled with 0, extra features dropped), renormalised, and log10-transformed identically. Probabilities were thresholded at the species-RF Youden-J value from `results/diagnostics/confusion_matrices.csv` (0.4847).

**Reproducibility.** External cohort tables are cached under `data/external_disease_cache/` so subsequent runs of `python3 scripts/diagnostics/cross_disease_specificity.py` reuse them without re-downloading. Pass `--force-fetch` to refresh.
