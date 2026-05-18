# Contributing

Short, practical guide for working on this repo.

## Run the tests

```bash
pip install -r requirements.lock
pip install pytest
pytest tests/ -v
```

The verification smoke test (49 checks against committed CSVs) runs the
same way CI does:

```bash
python3 scripts/verify_results.py
```

It exits non-zero on any failure.

## Add a new cohort

1. Add the cohort short name to the dataset list in `scripts/export_data.R`
   and re-run `Rscript scripts/export_data.R`. This writes the raw species
   and pathway matrices into `data/raw/`.
2. Run `python3 scripts/preprocessing.py` to rebuild
   `data/processed/metadata_clean.csv`, `species_filtered.csv`, and
   `pathway_unstratified.csv`. Apply the same global filters (prevalence
   >= 10%, mean >= 1e-4 for species).
3. If the new cohort shares a country with an existing one, no code change
   is needed -- `scripts/lodo_cv.py` reads country from metadata and
   excludes same-country cohorts from training automatically.
4. Re-run the full pipeline (see `REPRODUCING.md`).
5. Update the cohort count and sample totals everywhere they appear (start
   with `scripts/verify_results.py`, then README, manuscript, and
   `CHANGELOG.md`).

## Update verify_results.py expected values

`scripts/verify_results.py` is a *manuscript-locked* contract. Touch it only
when the manuscript numbers actually move.

1. Run the full pipeline end-to-end (Rscript -> preprocessing -> all
   train_*.py scripts).
2. Re-read the headline numbers from the regenerated CSVs.
3. Update the expected values and the per-check tolerances in
   `verify_results.py`. Keep tolerances tight (>= 0.005 absolute on AUC, >=
   0.001 on p-values) so drift is caught early.
4. Add a `CHANGELOG.md` entry under the next version explaining which
   numbers changed and why.

## Regenerate figures

```bash
python3 scripts/generate_figures.py            # Figures 1, 2, 3
python3 scripts/figure1_forest_plot.py          # Figure 1 standalone
python3 scripts/figure5_shap_three_panel.py     # Figure 4 (adenoma SHAP)
```

Outputs land in `figures/` as both PNG (300 DPI) and PDF. The figure
scripts read from `results/*.csv`, so re-run the upstream training scripts
first if any numbers changed.

## Local checks before pushing

```bash
pytest tests/ -v
python3 scripts/verify_results.py
```

CI runs the second command on every push and PR to `main` (see
`.github/workflows/verify.yml`); a failure there blocks merging.
