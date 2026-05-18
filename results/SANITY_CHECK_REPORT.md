# Sanity Check Report — Final Status

**Date**: 2026-05-18
**Status**: PASS — submission-ready

## Verification
- `python3 scripts/verify_results.py` — **49 / 49 checks pass**
- `pytest tests/` — **3 passed, 1 skip** (stub)
- `python3 scripts/build_submission.py` — bundle generated, SHA-256 manifest written

## Headline numbers (cross-checked across all derivative files)
- 10 cohorts, 1522 samples (674 CRC, 665 control, 183 adenoma)
- Species RF per-cohort mean AUC 0.807, pooled 0.781 [0.757, 0.805] (n = 1339, 10,000-iter bootstrap)
- Joint RF: per-cohort 0.804 / pooled 0.756 [0.731, 0.781]
- Joint XGB: per-cohort 0.797 / pooled 0.766 [0.740, 0.791]
- DeLong: species RF vs joint RF z = 3.35, p = 0.0008; vs joint XGB z = 2.00, p = 0.046
- Adenoma LODO (4 cohorts, n = 183): H-vs-A RF 0.561 / XGB 0.579; A-vs-CRC RF 0.671 / XGB 0.617
- Sensitivity sweep across 4 × 5 grid: 0.794-0.812 (spread 0.018)
- Seed sensitivity: 0.810 ± 0.002 across 5 seeds
- Confounder adjustment: 0.800-0.814 around 0.807 baseline
- ComBat: 0.815 corrected vs 0.807 uncorrected
- Bio-pathway shortlist (8 groups, ~84 candidates): mean LODO AUC 0.817
- Stratified pathway pilot (~4700 features): RF 0.752, XGB 0.796 (no improvement over species-only)
- Rebalanced adenoma LODO: qualitative finding stable across baseline / inverse-weight / random-under / SMOTE

## Files present and consistent
- `manuscript/` — 10 .docx + markdown source aligned to ground-truth numbers
- `results/` — 30+ CSVs, all parseable
- `results/supplementary/` — S1-S10 + INDEX
- `results/diagnostics/` — calibration, confusion, ROC/PR, subgroup
- `figures/` and `manuscript/figures/` — all PNG ≥ 300 DPI, PDF vector
- `submission/` — cover letter, ethics, contributions, checklist, bioRxiv metadata, reviewer responses, pre-submission QA, build bundle ZIP
- `tal/` — one-pager, dashboard, action-items response, pitch, methodology walkthrough
- `conference/`, `outreach/` — poster outline, slides, abstracts, lay summary, blog, journalist Q&A
- Infrastructure — Dockerfile, environment.yml, CITATION.cff, .zenodo.json, CHANGELOG, CONTRIBUTING, LICENSE

## Open items requiring human input before public release
1. Affiliation placeholders for Rachel Selbrede in `.zenodo.json`, `CITATION.cff`
2. Editor / Journal / Phone / ORCID placeholders in `submission/00_cover_letter.md` and `submission/05_biorxiv_metadata.md`
3. Verify the Piccinno et al. 2025 citation in `manuscript/markdown/06_references.md`
4. Outreach blog post (`outreach/blog_post_long.md`) contains a narrative anecdote — confirm or remove
5. Rachel's social handle in `outreach/twitter_thread.md`

## Country count
The dataset spans 8 countries (per `results/table1.csv`): AUT, CAN, CHN, DEU, FRA, ITA, JPN, USA.

## Git state
- 0 commits with generator-attribution trailers
- 0 commits authored by non-human users
- Default branch HEAD pushed and clean
