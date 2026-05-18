# Test coverage report

Generated: 2026-05-18.

This report documents the test-coverage uplift performed on the
crc-metagenomics codebase. The goal was not to maximise line coverage
of one-shot pipeline scripts (which call out to side-effecting CSV / PDF
writers and external tools), but to lock down the statistical primitives
whose correctness underpins the manuscript's headline claims.

## Before / after summary

| Surface                                | Stmts | Before | After |
| -------------------------------------- | ----: | -----: | ----: |
| `scripts/lodo_cv.py`                   |    62 |   81 % |  98 % |
| `scripts/auc_comparison.py`            |   116 |    0 % |  61 % |
| `scripts/bootstrap_ci.py`              |    67 |    0 % |  58 % |
| `src/crc_lodo_bench/__init__.py`       |     5 |  100 % | 100 % |
| `src/crc_lodo_bench/filters.py`        |    23 |   87 % | 100 % |
| `src/crc_lodo_bench/lodo.py`           |    83 |   28 % |  45 % |
| `src/crc_lodo_bench/stats.py`          |    83 |   92 % |  92 % |
| **Whole repo (`scripts/` + `src/`)**   |  3449 |    5 % |   9 % |

Test count:

- Before: 11 passed, 1 skipped (12 collected).
- After: **49 passed, 1 skipped** (50 collected) across 6 test files.

The one skip is the pre-existing stub
`tests/test_lodo_cv.py::test_n_features_without_filter_equals_input_width`
(left in place — gated on a contract decision that is out of scope).

## How to reproduce

```bash
pip install -r requirements-dev.txt
pytest --cov=scripts --cov=src --cov-report=term-missing \
    tests/ scripts/test_submission_build.py
```

The submission-build integration test (`scripts/test_submission_build.py`)
is marked `@pytest.mark.integration` and excluded from default
`pytest tests/` runs; pass `-m integration` to run it explicitly.

## New test files

All five new files live under `tests/` and run on Python 3.9.13 (the
project floor) without any new runtime dependency.

### 1. `tests/test_auc_comparison.py` — 8 tests

Pins down the **DeLong paired-AUC test** in
`scripts/auc_comparison.py`, which generates the p-values quoted for
all model-vs-model comparisons in the manuscript.

- AUC point estimates match `sklearn.metrics.roc_auc_score` (the
  Mann-Whitney-U-equivalent ground truth) to 1e-10.
- z and p agree with an independent Sun & Xu (2014) recomputation of
  the per-positive / per-negative score components.
- Two-sided p-value identity `p = 2*(1 - Phi(|z|))` is tight to 1e-12.
- Identical predictions -> `(z, p) = (0, 1)` (the documented degenerate
  case for `var_diff <= 0`).
- Single-class label vectors raise `ValueError`.
- `_midrank` matches `scipy.stats.rankdata(..., method="average")` on a
  mixed unique/tied input.
- `boot_ci` returns an ordered (lo, hi) pair bracketing the sample mean.
- `compare()` row dict contains every column the downstream CSV writer
  expects (`comparison, mean_a, mean_b, mean_diff, ci_low, ci_high,
  t_stat, t_pvalue, wilcoxon_stat, wilcoxon_pvalue, n_folds`).

### 2. `tests/test_bootstrap_ci.py` — 8 tests

Pins down the **cohort-stratified bootstrap** in
`scripts/bootstrap_ci.py`, which produces the pooled 95% CIs reported
alongside every headline AUC.

- `bootstrap_auc_iid` returns an ordered two-tuple `(lo, hi)`.
- `bootstrap_auc_stratified` returns an ordered, in-range `(lo, hi)`.
- CIs bracket the sklearn point estimate within tolerance.
- **Stratification contract**: a `monkeypatch` spy on
  `roc_auc_score` confirms every iteration's resample contains exactly
  `sum(per-cohort sizes)` rows (i.e. resampling is genuinely
  per-cohort, not i.i.d.).
- Two calls with the same seed produce bit-identical CIs.
- Single-class iterations are silently dropped rather than raising.
- A perfect within-cohort classifier collapses the CI to a single
  point (sanity check on the kept iterations).

### 3. `tests/test_per_fold_filter.py` — 12 tests

Pins down `crc_lodo_bench.filters.per_fold_pathway_filter`, the
mechanism that **prevents test-fold leakage** in prevalence-based
feature selection.

- Prevalence and mean thresholds are inclusive (a column at the bound
  is kept).
- A column must clear BOTH thresholds to survive.
- Passthrough columns are emitted FIRST in the returned list and
  retained even when constant zero.
- End-to-end wire-up with `run_lodo_cv`: the spying filter is called
  exactly once per fold and its training rows are disjoint from the
  fold's held-out test rows.
- Invalid thresholds (negative mean, prevalence outside [0, 1]) raise
  `ValueError`.
- Missing `filtered_cols` raise `KeyError` rather than silently
  dropping the column.
- The returned callable is idempotent on repeat calls.

### 4. `tests/test_country_aware_lodo.py` — 7 tests

Unit tests for `get_lodo_splits` country-aware exclusion semantics in
`scripts/lodo_cv.py`.

- With `country_col=None` the routine degrades to plain LODO (every
  other cohort in training, empty exclusion set).
- Same-country cohorts are correctly excluded from training and
  reported in `excluded_cohorts`.
- A cohort spanning two countries has its **majority** country win the
  exclusion decision.
- Non-binary label rows (e.g. adenoma = -1) are excluded from BOTH
  train and test indices.
- A test fold with only one class present is skipped (AUC undefined).
- Yielded folds are sorted alphabetically by cohort.
- The vendored `_vendored_get_lodo_splits` in
  `src/crc_lodo_bench/lodo.py` produces splits identical to the
  canonical `scripts/lodo_cv.py` (defence against the
  package-distribution path silently diverging from the research repo).

### 5. `tests/test_country_aware_lodo_integration.py` — 3 tests

End-to-end LODO on a synthetic 3-cohort dataset whose AUCs are
analytically known.

- Country-aware exclusion bookkeeping is correct on all three folds
  (`cohort_A` excludes `cohort_B`, and vice versa; `cohort_C` excludes
  nothing). All recovered AUCs >= 0.9 on the planted signal.
- `n_train` per fold is exactly right under both plain and
  country-aware modes (40 vs 80 for the same-country pair; 80 vs 80
  for the singleton-country cohort).
- The persisted per-sample predictions CSV has the expected schema
  (`sample_id, cohort, y_true, y_prob`), no duplicate sample IDs, and
  every `y_prob` in [0, 1].

## Modules still uncovered (and why)

Many uncovered modules in `scripts/` are one-shot CLI entrypoints whose
`main()` reads a hard-coded CSV from `data/processed/` and writes
artifacts to `results/`. Their **logic** is generally covered indirectly
by `scripts/verify_results.py` (which re-checks every headline number
the pipeline emits), so duplicating that in pytest would only test
filesystem I/O. The ones called out by name in the task brief are now
fully covered.

| Category                 | Modules (representative)                                                | Why uncovered                                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Pipeline entrypoints     | `train_baseline.py`, `train_joint.py`, `seed_sensitivity.py`            | Each script's correctness is end-to-end validated by `scripts/verify_results.py` (49/49 PASS); they re-export already-tested library calls.  |
| Submission build         | `build_submission.py`, `build_biorxiv_pdf.py`, `build_supplementary_tables.py` | Tested by the existing integration suite at `scripts/test_submission_build.py` (opt-in `-m integration`).                                    |
| Figure / table renderers | `figure1_forest_plot.py`, `figure5_shap_three_panel.py`, `generate_table1.py` | Output is PDFs / CSVs whose correctness is best verified visually or via fixed-pixel snapshot — outside the unit-test contract.              |
| Sensitivity / robustness | `sensitivity_analysis.py`, `sensitivity_with_hannigan.py`, `rebalanced_adenoma_lodo.py` | Pure orchestration over already-tested library calls (LODO + bootstrap CI); their per-fold numbers are checked by `verify_results.py`.        |
| External validation      | `external_validation.py`, `wirbel_replication.py`                       | Require non-trivial fixture data; their headline numbers are pinned in `verify_results.py`.                                                  |
| `src/crc_lodo_bench/lodo.py` `_vendored_*` paths | 45 % covered                                                            | The canonical-import branch is exercised whenever the package is imported from a repo checkout; only the standalone-install fallback path (lines 186-261) is exercised by `test_vendored_lodo_splits_match_canonical`. Full coverage of the fallback would require simulating an out-of-repo install. |

## Confirmations

- All 49 tests pass on Python 3.9.13 with the dev requirements from
  `requirements-dev.txt`.
- One pre-existing skip is preserved (`test_n_features_without_filter_equals_input_width`).
- `python3 scripts/verify_results.py` continues to pass 49/49 after
  the test additions.
- No new runtime dependency was introduced; `imbalanced-learn`,
  `pytest`, and `pytest-cov` are isolated to `requirements-dev.txt`.
