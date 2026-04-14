# CRC Metagenomics Classifier

Extending Thomas et al. (2019) CRC metagenomic classifier with pathway features and adenoma classification.

## Overview

Cross-cohort CRC detection using gut metagenomic species and pathway abundance from 7 cohorts via curatedMetagenomicData. 762 samples (326 CRC, 320 control, 116 adenoma).

## Results

**CRC Classification (LODO CV):**
- Species-only RF (baseline): AUC 0.803
- Joint RF (species + pathways): AUC 0.779
- Joint XGBoost: AUC 0.783

**Adenoma Classification (5-Fold CV):**
- Healthy vs Adenoma: RF 0.708, XGBoost 0.725
- Adenoma vs CRC: RF 0.793, XGBoost 0.815

**Top SHAP features:** Parvimonas micra, Peptostreptococcus stomatis, Gemella morbillorum, Fusobacterium nucleatum

## Reproducing

    pip install -r requirements.txt
    python scripts/preprocessing.py
    python scripts/preprocess_pathways.py
    python scripts/train_baseline.py
    python scripts/train_joint.py
    python scripts/train_adenoma.py
    python scripts/shap_analysis.py
    python scripts/generate_figures.py

## Authors

- Alejandro Velazquez, University of California, Berkeley
- Rachel Selbrede, California State University San Marcos

## License

MIT
