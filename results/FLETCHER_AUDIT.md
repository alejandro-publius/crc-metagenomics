# Fletcher Audit — CRC Metagenomics Repository

**Date**: 2026-05-18
**Scope**: Read-only audit of `/Users/alexvintera/Desktop/crc-metagenomics`.
**Verifier**: `scripts/verify_results.py` returns **49/49 PASS, 0 FAIL** on `main` HEAD plus uncommitted changes.
**Concurrent edits**: An adjacent agent is editing `scripts/bio_pathway_shortlist.py`, `manuscript/markdown/{03_methods,05_discussion,07_supplementary}.md`, `results/decisions_addendum.md`, `results/bio_pathway_results.csv`, and adding `scripts/diagnostics/raw_data_exploration.py`, `results/diagnostics/{raw_data_summary.csv,RAW_DATA_PATTERNS.md}`, and 5 `figures/diagnostics/*.png`. Items below that touch those files are tagged **IN FLUX** and noted for human re-verification once that pass completes.

Severity legend:
- **P0** — release blocker
- **P1** — serious (numerical drift, orphan with downstream impact, untrue claim)
- **P2** — polish
- **PASS** — explicitly verified clean

---

## 1. Numerical integrity

- **PASS**: `python3 scripts/verify_results.py` → **49/49 checks pass, 0 fail** under HEAD + working-tree changes (re-run during audit).
- **PASS**: `pytest tests/` → 3 passed, 1 skipped (the one skip is `test_n_features_without_filter_equals_input_width`, deliberately marked with a contract-locking note).
- **P0** — `results/supplementary/S8_seed_sensitivity.csv` is **unparseable as CSV**: `pandas.read_csv` raises `Error tokenizing data. C error: Expected 3 fields in line 8, saw 7`. The file stitches two header-bearing tables (`seed,mean_auc,std_auc` then a `metric,n_seeds,grand_mean,…` summary) separated by a blank line. The same malformed file is also in `submission/build/supplementary/S8_seed_sensitivity.csv` (byte-identical) and inside `submission/build/SUBMISSION_BUNDLE.zip`. Source is `scripts/build_supplementary_tables.py:152-171`, which intentionally writes two DataFrames into one file. Reviewers using pandas on the supplementary tar will fail on this file; this is a deliverable defect.
- **PASS**: Every other CSV under `results/`, `results/diagnostics/`, and `results/supplementary/` parses cleanly. Per-file row/col counts and NaN/Inf scan:
  - 44 CSVs scanned, 0 contained `inf` in numeric columns.
  - 0 contained unexpected NaN in numeric columns.
  - All-NaN columns: none.
  - `results/diagnostics/raw_data_summary.csv` parses, 10 rows × 8 cols (IN FLUX — added by concurrent agent).
- **PASS**: Row-count cross-checks: `preds_*.csv` all = 1339; `baseline_results.csv` = 10 cohorts; `joint_results.csv` = 10 cohorts; `bootstrap_ci.csv` = 33 (10 cohorts × 3 models + 3 pooled); `adenoma_lodo_results.csv` = 4 task rows; `seed_sensitivity.csv` = 5 seeds; `confounder_results.csv` = 4 methods; `sensitivity_thresholds.csv` = 20 cells.

## 2. Cross-file number consistency

Headline numbers grep'd across `README.md`, `REPRODUCING.md`, all `results/*.md`, all `manuscript/markdown/*.md`, `submission/*.md`, `conference/*.md`, `outreach/*.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github_local_only/workflows/verify.yml`, and every `.py` / `.sh`. Findings:

### 2a. Biological-pathway shortlist drift (P1, partially IN FLUX)
The bile-acid group was added to the shortlist (now 9 groups, 86 unique pathway IDs — confirmed by counting unique IDs in `results/bio_pathway_shortlist.txt`: 86). The following files have already been updated to `9 groups / 86`:
- `README.md` line 54
- `manuscript/markdown/03_methods.md` line 15
- `manuscript/markdown/05_discussion.md` lines 7, 15 (IN FLUX)
- `manuscript/markdown/07_supplementary.md` line 90 (IN FLUX)
- `manuscript/markdown/manuscript_complete.md` lines 65, 105, 184, 192, 361

The following files still say `8 groups / 84`:
- **`REPRODUCING.md` line 70** — "84 unique CRC-relevant pathways across 8 groups"
- **`results/decisions_addendum.md` lines 155-167** — still describes "eight groups / 84 unique" in the original entry; addendum at lines 198-214 adds the ninth group but does not strike the old text (IN FLUX)
- **`submission/build/manuscript_complete.md`** — stale copy with "eight biological groups" at lines 65, 105, 184, 192, 361 (5 sites); the source file has been updated but submission/build was not rebuilt
- **`submission/build/CRC_Manuscript_Complete.docx`** — 3 stale `84` mentions + 4 stale `eight` mentions (confirmed by direct XML inspection)
- **`results/SANITY_CHECK_REPORT.md` line 22** — "Bio-pathway shortlist (8 groups, ~84 candidates)"

### 2b. Bootstrap iteration count drift (P1)
Headline is `10,000` (matches `scripts/bootstrap_ci.py` and `verify_results.py`). Stale `2000`/`2,000` instances:
- **`results/supplementary/INDEX.csv` line 5** — "S4,S4_bootstrap_ci.csv,2000-iteration bootstrap 95% CIs"
- **`submission/build/supplementary/INDEX.csv` line 5** — same stale string
- **`scripts/build_supplementary_tables.py` line 218** — hardcoded "2000-iteration" in the generated INDEX description
- **`scripts/build_biorxiv_pdf.py` line 56** — caption text "2000-iteration cohort-stratified bootstrap"
- **`submission/04_submission_checklist.md` line 32** — "Bootstrap CI iteration count specified (2000)"

### 2c. Country count drift (P1)
Ground truth from `results/table1.csv` = **8 countries** (AUT, IND, ITA, JPN, USA, DEU, CHN, FRA). Manuscript markdown and outreach all say 8. Stale `7 countries`:
- **`conference/slides_15min.md` line 28** — "Dataset: 10 cohorts, 7 countries, 1,522 samples"
- **`conference/README.md` line 64** — "10 cohorts, 7 countries, 1,522 samples"
- **`conference/poster_outline.md` line 53** — "Ten curatedMetagenomicData cohorts spanning seven countries"

Additional country defect:
- **`results/SANITY_CHECK_REPORT.md` line 44** — claims the 8 countries are "AUT, CAN, CHN, DEU, FRA, ITA, JPN, USA". This is **doubly wrong**: it lists CAN (Canada, not in dataset) and omits IND (India, present via GuptaA_2019).

### 2d. Cohort count drift (P2)
`REPRODUCING.md` line 18 + `scripts/export_data.R` lines 4, 95 say "11 cohorts (~1604 samples)" because the exporter pulls all 11 (then HanniganGD_2017 is excluded downstream). The numbers are correct *for the export step*, but a casual reader of REPRODUCING will see "11 cohorts" before encountering the 1,522 / 10-cohort number. Suggest a one-line clarifier in REPRODUCING that explains the 11 → 10 reduction at the preprocessing step.

### 2e. Verify-script check count drift (P1)
Actual count is **49 PASS** (audited via grep on script output). Stale `38` / `30`:
- **`CHANGELOG.md` line 37** — "scripts/verify_results.py: 38 headline-number checks"
- **`CONTRIBUTING.md` line 13** — "The verification smoke test (38 checks against committed CSVs)"
- **`.github_local_only/workflows/verify.yml` lines 6, 13** — "exits non-zero if any of its 38 checks fail"
- **`submission/04_submission_checklist.md` line 38** — "scripts/verify_results.py 38/38 checks pass"
- **`submission/06_reviewer_responses.md` line 239** — "49 automated assertions" (correct) but in the same paragraph "within a tolerance of 0.002 AUC units" — actual tolerances range 0.001–0.05, not 0.002 uniformly.
- **`manuscript/markdown/03_methods.md` line 59** — "A verification script (`scripts/verify_results.py`) asserts that headline AUC values match expected values within a tolerance of 0.002" — same overstated tolerance.

### 2f. Verified-clean number patterns
- **PASS**: `0.807` (species RF per-cohort mean), `0.781` (pooled), `0.0008` (DeLong p), `0.046`, `z = 3.35`, `z = 2.00` — every hit is identical across README / manuscript markdown / SANITY_CHECK_REPORT / baseline_results.md / conference / outreach / submission.
- **PASS**: `1,522 samples`, `1339`/`1,339` pooled n, `674 CRC`, `665 control`, `183 adenoma` — consistent.
- **PASS**: `10 cohorts` / `10-cohort` consistent across all final-deliverable files (the `11 cohorts` mentions in `REPRODUCING.md` / `export_data.R` refer to pre-exclusion).
- **PASS**: `0.808` hits are all legitimate per-fold or per-sample numbers, not stale `0.807`/`0.781` typos.

## 3. Orphan files / unreferenced material

Scanned every non-`.git`/`.pytest_cache`/`__pycache__` file; cross-checked every basename and relpath against the full text corpus.

### 3a. Confirmed orphans (P1 cleanup candidates)
- `scripts/diagnostics/depth_confound_check.py` — exists, generates `results/diagnostics/depth_confound_shap.csv` + `figures/diagnostics/depth_vs_fnucleatum_shap.png` (both referenced from `ROBUSTNESS_SUMMARY.md`), but the *script* itself is not named anywhere outside its own file. Add to `results/diagnostics/README.md` Reproducibility table.
- `scripts/diagnostics/calibration_mechanism.py` — same pattern: outputs referenced, script unnamed.
- `scripts/diagnostics/permutation_importance.py` — same pattern.
- `scripts/diagnostics/sens_at_fixed_specificity.py` — referenced from `CLINICAL_TRANSLATION_SUMMARY.md`, but `CLINICAL_TRANSLATION_SUMMARY.md` itself is an orphan (see below).
- `scripts/build_supplementary_tables.py` — generates `results/supplementary/S*.csv` but no other file references the script. (It is the *source* of the broken S8 and the stale "2000-iteration" INDEX string.)
- `scripts/run_robustness.sh` — useful helper shell script, never referenced.
- `data/processed/pathway_filtered.csv` — exists, ignored by gitignore conceptually (it's referenced in the gitignore only as `pathway_unstratified.csv` parent — actually it's `pathway_filtered.csv`, distinct), no script reads or writes it.
- `submission/07_pre_submission_qa.md` — exists but missing from `submission/MANIFEST.md` and missing from `submission/build/SUBMISSION_BUNDLE.zip`. Contains 4 explicit "FIX before submission" items (Bonferroni note in Results; sensitivity-at-fixed-spec Supplementary Table S11; reconciliation sentence; one-sentence note that DeLong signal is concentrated in YachidaS_2019). The owner should either close these out before pressing submit, or remove the doc.
- `submission/06_reviewer_responses.md` — exists but missing from `MANIFEST.md` and missing from `SUBMISSION_BUNDLE.zip`. This is appropriate for an *internal* doc — but MANIFEST should at least list it under "submission/ (scaffolding, human-edited)".
- `results/CITATION_AUDIT.md` — useful, but not referenced from any other file.
- `results/SANITY_CHECK_REPORT.md` — itself stale (see §2c, §2a). Not referenced anywhere.
- `results/baseline_results.md` — clean and useful, but not linked from any README or REPRODUCING.
- `results/diagnostics/CLINICAL_TRANSLATION_SUMMARY.md` — high-value content (Imperiale FIT comparison, PPV at 5% prevalence) referenced by `05_discussion.md` indirectly (via `fit_vs_microbiome.csv`) but the .md document itself is unreferenced.

### 3b. Diagnostic figure orphans (IN FLUX — concurrent agent)
- `figures/diagnostics/alpha_diversity.png`
- `figures/diagnostics/cohort_composition.png`
- `figures/diagnostics/depth_distribution.png`
- `figures/diagnostics/pcoa_bray_curtis.png`
- `figures/diagnostics/top_species_heatmap.png`

All five are unstaged additions by the concurrent agent; the matching documentation (`results/diagnostics/RAW_DATA_PATTERNS.md`) is also unstaged. Re-audit after that pass completes.

### 3c. Junk files / patterns
- **PASS**: No `.bak`, `.swp`, `.tmp`, `*.pyc` checked into git.
- **PASS**: No `GITHUB_SUPPORT_TICKET*` anywhere.
- **PASS**: No `fix_*` legacy scripts.
- **PASS**: `tests/test_lodo_cv.py` and `scripts/test_submission_build.py` are legitimate tests, not leftovers.
- `__pycache__/` and `.pytest_cache/` directories exist on disk (not in git — properly gitignored), no action needed but worth a `rm -rf` before the next release tag.
- `.RData` (1.4 MB) and `.Rhistory` (4.6 KB) live in repo root, properly gitignored. No release-blocker but inert.

### 3d. Submission build staleness (P1)
- `submission/build/manuscript_complete.md` ≠ source `manuscript/markdown/manuscript_complete.md` — diff shows 5 lines where build still says "eight biological groups" / "84 candidate pathways" / "8 CRC-relevant functional groups" while source says nine / 86.
- `submission/build/CRC_Manuscript_Complete.docx` ≠ source `manuscript/CRC_Manuscript_Complete.docx` — build still has 4 `eight` + 3 `84` mentions; source has 4 `nine` + 2 `86` mentions.
- `submission/build/supplementary/Supplementary_Tables.docx` ≠ source `manuscript/Supplementary_Tables.docx`.
- `submission/build/SUBMISSION_BUNDLE.zip` (SHA `7ebe3b6…`, mtime 2026-05-18 05:09:56) was built from the *stale* sources before the bile-acid update propagated to submission/build. **A rebuild is mandatory before submission**, even if the agent's current pass finishes the source-side fixes.

## 4. Script integrity

Audited all `scripts/*.py` and `scripts/diagnostics/*.py` with `ast.parse`.

- **PASS** — 0 scripts fail syntactic validation.
- **PASS** — 0 `breakpoint()`, `pdb.set_trace`, `import pdb`, or `print('DEBUG'…)` debug leftovers.
- **PASS** — 0 `TODO` / `FIXME` / `XXX` comments in any script.

### 4a. Missing docstrings (P2)
- `adenoma_counts.py`, `check_label_dist.py`, `filter_pathways.py`, `find_nans.py`, `lodo_cv.py`, `preprocessing.py`, `sanity_check.py`, `train_adenoma.py`, `train_baseline.py`, `train_joint.py`, `validate_pathways.py`

11 scripts lack a top-of-file docstring. Several of these (`lodo_cv.py`, `train_baseline.py`, `train_joint.py`, `preprocessing.py`) are core pipeline scripts; the lack of docstrings is a quality-of-life issue for reviewers reading the code.

### 4b. Missing `__main__` guard (P2)
- `adenoma_counts.py`, `auc_comparison.py`, `check_label_dist.py`, `filter_pathways.py`, `find_nans.py`, `lodo_cv.py`, `merge_pathways.py`, `sanity_check.py`, `validate_pathways.py`

These execute at import time. `lodo_cv.py` is special — it is *imported* by `tests/test_lodo_cv.py` and by `train_baseline.py` / `train_joint.py`, so its top-level code runs every time the modules import. Verify there are no side-effect statements at module scope.

### 4c. REPRODUCING.md script coverage (P1)
23 scripts are NOT mentioned in `REPRODUCING.md`. Breakdown:

*Diagnostics (all should be mentioned, at least in `results/diagnostics/README.md` if not in REPRODUCING.md):*
- `base_rate_ppv.py`, `calibration_mechanism.py`, `depth_confound_check.py`, `permutation_importance.py`, `raw_data_exploration.py` (IN FLUX), `sens_at_fixed_specificity.py`, `subgroup_analysis.py` (referenced in diagnostic README), `fit_comparison.py` (referenced in CLINICAL_TRANSLATION_SUMMARY.md).

*Helpers / infra:*
- `add_covariates.py`, `build_biorxiv_pdf.py`, `build_submission.py`, `build_supplementary_tables.py`, `external_validation.py`, `lodo_cv.py` (imported library — acceptable to omit), `rebalanced_adenoma_lodo.py` (referenced in `05_discussion.md` "Adenoma analysis: class-balance robustness"), `run_robustness.sh`, `stratified_pathway_pilot.py` (referenced in 03_methods, 05_discussion, 07_supplementary), `test_submission_build.py` (legitimate test), `train_adenoma.py` (deprecated per 03_methods.md, "retained for reference only").

`results/diagnostics/README.md` lists only 5 of the 11 diagnostic scripts; the other 6 (base_rate_ppv, calibration_mechanism, depth_confound_check, fit_comparison, permutation_importance, sens_at_fixed_specificity, raw_data_exploration) are not in its Reproducibility table. Recommend updating the diagnostic README table.

### 4d. Magic numbers (P2 — informational)
- `scripts/build_supplementary_tables.py` line 218 hardcodes the literal "2000-iteration" instead of reading from `bootstrap_ci.csv`. This is the source of the stale string in `INDEX.csv`.
- `scripts/build_biorxiv_pdf.py` line 56 hardcodes "2000-iteration cohort-stratified bootstrap" in a caption.
- The species-feature count `229` and per-fold pathway counts `402–406` are hardcoded in multiple scripts and docs; values are CSV-derivable (`baseline_results.csv` for species count via training pipeline; `joint_results.csv` rf_n_features column). Acceptable, but a `constants.py` module would make future expansion safer.

## 5. Manuscript integrity

### 5a. .docx files
**PASS** — All 10 manuscript .docx files exist and are non-empty:
`CRC_Title_Page.docx`, `CRC_Abstract.docx`, `CRC_Introduction.docx`, `CRC_Methods.docx`, `CRC_Results.docx`, `CRC_Discussion.docx`, `CRC_References.docx`, `CRC_Table1.docx`, `Supplementary_Tables.docx`, `CRC_Manuscript_Complete.docx`.

### 5b. manuscript_complete.md concatenation (PASS)
`manuscript/markdown/manuscript_complete.md` contains the first-line header of every section file (`00_title.md` through `07_supplementary.md`). It is the master and is the source from which `_build_docx.py` derives the combined .docx.

### 5c. Figures (PASS / IN FLUX)
- All 4 main figures (`Figure1_Forest_Plot`, `Figure2_ROC_Curves`, `Figure3_SHAP_Importance`, `Figure4_Three_Panel_SHAP`) exist as both `.png` and `.pdf` in `manuscript/figures/` and are referenced by `04_results.md` and `manuscript_complete.md`.
- Legacy paths `figures/fig1_lodo_auc.png`, `figures/fig2_shap_crc.png`, `figures/fig3_adenoma.png`, `figures/fig4_external_validation.png`, `figures/figure5_three_panel_shap.png` are referenced by `conference/poster_outline.md` and `scripts/generate_figures.py` for backward compatibility. All exist.

### 5d. CSV references
**PASS** — Every concrete CSV path mentioned in any markdown file resolves to an existing file. Only `results/preds_*.csv` and `results/shap_*.csv` appear as glob patterns; the underlying files all exist.

### 5e. Citations (P1)
- **Body refs ↔ refs list (manuscript proper)**: All 18 entries in `06_references.md` are cited at least once in the body files; no orphan refs. Specifically: Bellman 1961, Trunk 1979, Wirbel 2019, and Yachida 2019 do appear in citation form in `02_introduction.md` and `05_discussion.md`.
- **P1 — Missing references in 06_references.md**: `05_discussion.md` line 45 cites "(Imperiale 2014 *N Engl J Med*; Chiu 2017 *JAMA Intern Med*)" in the FIT comparison paragraph. Neither Imperiale 2014 nor Chiu 2017 is in `06_references.md`. The cite format is also inconsistent with the rest of the manuscript (inline journal vs. numbered list). **This is in 05_discussion.md which is IN FLUX**; re-verify after concurrent agent finishes.
- **P2** — `submission/06_reviewer_responses.md` cites Davison and Hinkley 1997, Grinsztajn et al. 2022, Shwartz-Ziv and Armon 2022, Pasolli et al. 2016, Topcuoglu et al. 2020, Lee et al. 2014, Rubinstein et al. 2013, Gur et al. 2015, Castellarin et al. 2012, Kostic et al. 2013, Drewes et al. 2017, Elor and Averbuch-Elor 2022. None are in `06_references.md`. If the reviewer-responses doc is ever bundled or sent to a reviewer, those citations need a paired references list.

### 5f. Ground-truth match (PASS, modulo §2 drift items)
- Species RF per-cohort mean **0.807**, pooled **0.781 [0.757, 0.805]**, n=1339 — matches `baseline_results.csv`, `bootstrap_ci.csv`, `delong_results.csv`, manuscript abstract / results / discussion.
- Joint RF **0.804 / 0.756**, Joint XGB **0.797 / 0.766** — matches `joint_results.csv`, `bootstrap_ci.csv`.
- DeLong species vs joint RF **z=3.35, p=0.0008**; vs joint XGB **z=2.00, p=0.046** — matches `delong_results.csv`.
- Adenoma LODO H-vs-A RF **0.561** / XGB **0.579**, A-vs-CRC RF **0.671** / XGB **0.617** — matches `adenoma_lodo_results.csv`.
- Seed sensitivity grand mean **0.810** with spread ~0.004 — matches `seed_sensitivity.csv`.
- Sensitivity grid spread **0.018** within 0.794–0.812 — matches `sensitivity_thresholds.csv`.
- Confounder range **0.800–0.814** — matches `confounder_results.csv`.
- ComBat **0.815 vs 0.807** — matches `combat_results.csv`.

## 6. Submission package integrity

- **P1**: `submission/build/SUBMISSION_BUNDLE.zip` was built at 2026-05-18 05:09:56 *before* the bile-acid / 9-group update; its `manuscript_complete.md` and `CRC_Manuscript_Complete.docx` still say "eight biological groups / 84 candidate pathways". SHA-256 in `MANIFEST.md` (`7ebe3b6…`) matches the on-disk file, so the manifest is internally consistent — but the bundle is stale relative to the source manuscript. **Rebuild is required** after the concurrent agent's pass.
- **PASS**: `submission/MANIFEST.md` SHA-256 for `SUBMISSION_BUNDLE.zip` verifies (re-hashed during audit).
- **P1**: `submission/MANIFEST.md` does NOT list `submission/06_reviewer_responses.md` or `submission/07_pre_submission_qa.md`. Both files exist on disk. If they are intentionally internal-only, that should be stated in MANIFEST; if they should ship, they need to be added to MANIFEST and the bundle.
- **PASS**: All 8 expected scaffolding files present: `00_cover_letter.md`, `01_data_availability.md`, `02_ethics_statement.md`, `03_author_contributions.md`, `04_submission_checklist.md`, `05_biorxiv_metadata.md`, `06_reviewer_responses.md`, `07_pre_submission_qa.md`.
- **P2**: `submission/00_cover_letter.md` has `[Editor Name]`, `[Journal Name]`, `[Date]`, and three `[Reviewer N…]` placeholders — expected for a generic template, but flag for the human pass.
- **P2**: `submission/02_ethics_statement.md` line 14 references `[Institution]` — placeholder.
- **P2**: `submission/05_biorxiv_metadata.md` line 17 lists Rachel's affiliation as "Department of Biological Sciences" while `CITATION.cff` and `.zenodo.json` say "Molecular and Cell Biology (CSUSM MCB)". Pick one and propagate.
- **P2**: `submission/03_author_contributions.md` is clean and uses CRediT taxonomy.

## 7. Infrastructure

- **PASS**: `.zenodo.json` is valid JSON; both authors have ORCIDs (`0009-0007-9798-1958` for Velazquez, `0009-0006-7046-3192` for Selbrede) and affiliations.
- **PASS**: `CITATION.cff` is valid YAML; both authors have ORCID URLs and affiliations.
- **PASS**: `LICENSE` is MIT with copyright "Alejandro Velazquez and Rachel Selbrede".
- **PASS**: `Dockerfile` uses `python:3.11-slim`, installs `requirements.lock`, runs `verify_results.py` as default CMD.
- **PASS**: `environment.yml` pins Python 3.11, pandas 2.2.3, numpy 1.26.4, sklearn 1.4.2, xgboost 2.0.3, scipy 1.12.0 — matches `requirements.lock`.
- **PASS**: `requirements.lock` pins exact versions including `combat==0.3.3`; `requirements.txt` is a loose range version of the same packages.
- **PASS**: `.gitignore` excludes `__pycache__/`, `*.pyc`, `.DS_Store`, `.pytest_cache/`, `.RData`, `.Rhistory`, large raw data files. Verified by `find` — no junk in `git ls-files`.
- **P1 — CHANGELOG.md**: Line 33 says "biologically-guided 84-pathway shortlist" — stale, should be 86. Line 37 says "verify_results.py: 38 headline-number checks" — stale, should be 49.
- **PASS**: `CONTRIBUTING.md` exists, but line 13 says "(38 checks against committed CSVs)" — stale, should be 49.
- **P1**: `CONTRIBUTING.md` line 72 + the workflow file itself (`.github_local_only/workflows/verify.yml`) — the workflow is in `.github_local_only/`, **not** in `.github/workflows/`. CI is therefore **not running** despite `CONTRIBUTING.md` claiming "CI runs the second command on every push and PR to main". `submission/06_reviewer_responses.md` line 240 makes the same claim ("The verification script runs on every push via GitHub Actions; the build status is visible at the repository top page"). This claim is currently false. Either:
  - Move `.github_local_only/workflows/verify.yml` to `.github/workflows/verify.yml` and let it run (the intent), or
  - Remove the "CI runs on every push" claim from `CONTRIBUTING.md` and `06_reviewer_responses.md`.
- **PASS**: `tests/test_lodo_cv.py` exists and runs (3 passed, 1 skip).
- **PASS**: `scripts/test_submission_build.py` exists, marked `@pytest.mark.integration` (won't run by default — appropriate).

## 8. Git state

- **P1 (working tree dirty)**: `git status` shows 9 modified files and 8 untracked files, all from the concurrent agent (IN FLUX scope above). README.md modification was not in the named-files list for that agent but is in working tree; verify the diff is expected once the agent finishes.
- **PASS**: `git log --all --grep="Co-Authored-By"` — empty.
- **PASS**: `git log --all --grep="Claude"` — empty.
- **PASS**: `git log --all --grep="Anthropic"` — empty.
- **P2 (historic)**: `git log --all --grep="Korem\|DEBIAS\|Austin\|Columbia"` finds two commit messages that name `tal/` (the directory removed in commit `6513a65`) and mention "Korem reference" (referring to a paper that was removed from references, commit `b5bd3d2`). Both are *cleanup commits removing those references*, so the mention is appropriate in the audit trail. No literal content of those references remains in any tracked file.
- **PASS**: `git branch -a` — `main`, `alejandro-publius-patch-1` (local + remote tracking). `git ls-remote --heads origin` confirms both branches on origin.
- **PASS**: All commit authors are `Alex <…@users.noreply.github.com>` (Alejandro) or `Rachel Selbrede <rachel.selbrede@gmail.com>` (also as `rachelselbrede`). No machine / generated authors.
- **P2 (large files tracked in git)**: `data/processed/pathway_abundance_filtered.csv` is 57.35 MB and is committed (above the GitHub 50 MB soft limit; push works but generates a warning). Other tracked large files: `data/raw/pathway_chunks/ZellerG_2014.csv` (14.0 MB), `FengQ_2015.csv` (13.6 MB), `YuJ_2015.csv` (12.0 MB). The `.gitignore` excludes `pathway_chunks/` but those files were committed before the gitignore line was added, so they remain tracked. Recommend running `git rm --cached data/raw/pathway_chunks/*.csv data/processed/pathway_abundance_filtered.csv` and either using Git LFS or relying on `export_data.R` for regeneration.
- **PASS**: `data/raw/pathway_abundance.csv` (279.6 MB) is correctly *not* tracked (matches `.gitignore` line 7).

## 9. AI / external attribution scan

Full-tree grep (excluding `.git`, `__pycache__`, `.pytest_cache`, and binary `.docx/.zip/.pdf/.png/.RData`).

- **PASS** — 0 hits for `Claude`, `Anthropic`, `Co-Authored-By`, `noreply@anthropic`, `@claude`, `@anthropic`, `generated with`, `AI assistant`.
- **PASS** — 0 standalone whole-word hits for `Tal`, `GPT`, `LLM`.
- **PASS** — 0 hits for `Korem`, `DEBIAS`, `DEBIAS-M`, `George Austin`, `Columbia` in any tracked file.

The repo is clean of attribution-related leakage at the file content level. (Two commit messages remain that mention these as removal-rationale strings — see §8.)

## 10. Reproducibility

- **P1** — `REPRODUCING.md` covers 28 of 51 scripts (55%). 23 scripts (mostly diagnostics + submission-build infra) are documented only via `results/diagnostics/README.md`, `results/diagnostics/CLINICAL_TRANSLATION_SUMMARY.md`, and `results/diagnostics/ROBUSTNESS_SUMMARY.md`, or not documented at all. Recommend adding a "Diagnostics (optional, post-hoc)" section to REPRODUCING listing `scripts/diagnostics/*.py` collectively.
- **P1** — REPRODUCING line 70 says "bio_pathway_shortlist.py … 84 unique CRC-relevant pathways across 8 groups" — stale (now 86 / 9). The concurrent agent is editing `bio_pathway_shortlist.py`; their next edit should propagate to REPRODUCING.
- **PASS** — Each step in REPRODUCING has an "expect: …" comment that matches the actual CSV:
  - `train_baseline.py` → "per-cohort mean AUC ~0.807, pooled AUC ~0.781" ✓
  - `train_joint.py` → "Joint RF per-cohort ~0.804 (pooled ~0.756); Joint XGB per-cohort ~0.797 (pooled ~0.766)" ✓
  - `auc_comparison.py` → "species_rf vs joint_rf DeLong z=3.35, p=0.0008; species_rf vs joint_xgb z=2.00, p=0.046" ✓
  - `adenoma_lodo.py` → "H-vs-A RF ~0.561, H-vs-A XGB ~0.579; A-vs-CRC RF ~0.671, A-vs-CRC XGB ~0.617" ✓
  - `bio_pathway_shortlist.py` → "mean per-cohort LODO AUC ~0.817 (vs species-only 0.807)" ✓
  - `bootstrap_ci.py` → "species RF pooled 0.781 [0.757, 0.805]" ✓
  - `seed_sensitivity.py` → "expect spread < 0.005" ✓ (actual 0.004)
  - `sensitivity_analysis.py` → "joint RF mean per-cohort AUC range 0.794-0.812" ✓
  - `confounder_adjustment.py` → "per-cohort AUC 0.800-0.814" ✓
  - `batch_correction.py` → "mean per-cohort AUC ~0.815 (vs uncorrected ~0.807)" ✓
- **PASS** — `REPRODUCING.md` line 11 says "Total runtime is approximately 45 minutes" — consistent with verified runtime in `manuscript/markdown/03_methods.md` line 63 ("approximately 45 minutes on a standard workstation") and `.github_local_only/workflows/verify.yml` comment ("~45 min instead of ~30 s" if data/processed is rebuilt). Verified once during audit (full pipeline not re-run end-to-end; only `verify_results.py` was re-executed, took ~3 s).

---

## Punch list (priority order)

### P0 — Release blockers
1. **Fix `results/supplementary/S8_seed_sensitivity.csv`** — Currently unparseable as CSV (two stitched tables). Split into two files (e.g., `S8a_seed_sensitivity_per_seed.csv` and `S8b_seed_sensitivity_summary.csv`) OR keep one file with a single header. Update `scripts/build_supplementary_tables.py:152-171`. Re-copy to `submission/build/supplementary/` and **rebuild `SUBMISSION_BUNDLE.zip`**.

### P1 — Serious
2. **Rebuild `submission/build/`** after the concurrent agent's pass: `manuscript_complete.md`, `CRC_Manuscript_Complete.docx`, `supplementary/Supplementary_Tables.docx`, and `SUBMISSION_BUNDLE.zip` are all stale w.r.t. the source. The 9-group / 86-pathway update has not propagated.
3. **Propagate 9-group / 86-pathway change** to: `REPRODUCING.md:70`, `results/SANITY_CHECK_REPORT.md:22`, `results/decisions_addendum.md` (the original 8-group entry needs an update note or rewrite — IN FLUX), `CHANGELOG.md:33`. (Source files already correct; this is just downstream propagation.)
4. **Fix "2000-iteration bootstrap" stale strings** in: `scripts/build_supplementary_tables.py:218`, `scripts/build_biorxiv_pdf.py:56`, `submission/04_submission_checklist.md:32`. Regenerate `results/supplementary/INDEX.csv` (and the submission/build copy).
5. **Fix country count drift** in `conference/slides_15min.md:28`, `conference/README.md:64`, `conference/poster_outline.md:53` (7 → 8).
6. **Fix `results/SANITY_CHECK_REPORT.md:44`** country list — replace "CAN" with "IND" (current list omits India and invents Canada).
7. **Fix verify-script check count** (38 → 49) in `CHANGELOG.md:37`, `CONTRIBUTING.md:13`, `.github_local_only/workflows/verify.yml:6,13`, `submission/04_submission_checklist.md:38`. Also fix the "tolerance of 0.002" claim in `manuscript/markdown/03_methods.md:59` and `submission/06_reviewer_responses.md:239` to reflect the actual per-check tolerance range (0.001–0.05).
8. **Move `.github_local_only/workflows/verify.yml` → `.github/workflows/verify.yml`** so CI actually runs (or remove the "CI runs on every push" claim from CONTRIBUTING.md and reviewer_responses.md). Currently the claim is false.
9. **Resolve `submission/07_pre_submission_qa.md` FIX queue**: (a) add Bonferroni-corrected p-values to Results paragraph, (b) add Supplementary Table S11 (sensitivity at fixed specificity {0.80, 0.85, 0.90, 0.95}), (c) confirm Results reconciliation sentence, (d) add YachidaS_2019-fold flag in Discussion. Either complete these items or delete `07_pre_submission_qa.md`.
10. **Add `submission/06_reviewer_responses.md` and `submission/07_pre_submission_qa.md` to `submission/MANIFEST.md`** (in the scaffolding section, marked as internal-only if appropriate).
11. **Add Imperiale 2014 (*N Engl J Med* 370:1287) and Chiu 2017 (*JAMA Intern Med*)** to `manuscript/markdown/06_references.md` so the FIT comparison paragraph in `05_discussion.md:45` resolves. IN FLUX in 05_discussion — verify after concurrent agent finishes.
12. **Document the 23 unreferenced scripts** in REPRODUCING.md (or a new `scripts/diagnostics/REPRODUCING_DIAGNOSTICS.md`). At minimum, list every script that produces output committed to `results/` or `figures/`.
13. **Update `results/diagnostics/README.md`** Reproducibility table to include `base_rate_ppv.py`, `calibration_mechanism.py`, `depth_confound_check.py`, `fit_comparison.py`, `permutation_importance.py`, `sens_at_fixed_specificity.py`. Only 5 of 11 diagnostic scripts are listed today.
14. **Reconcile Rachel's affiliation string** across `.zenodo.json` ("Molecular and Cell Biology"), `CITATION.cff` ("Molecular and Cell Biology"), `submission/05_biorxiv_metadata.md` ("Department of Biological Sciences"). Pick one and propagate.
15. **Remove or migrate the four large tracked files** (`data/processed/pathway_abundance_filtered.csv` 57.4 MB; three `pathway_chunks/*.csv` 12-14 MB each) to Git LFS or regenerate-on-demand via `export_data.R` / `merge_pathways.py`. Currently above GitHub's 50 MB soft limit.

### P2 — Polish
16. Conference / outreach files have `[FILL: Rachel's affiliation]` placeholders in `poster_outline.md:6`, `abstract_aacr.md:9`, `abstract_gut_microbiota_for_health.md:9`, `abstract_ismb.md:9`; and `@[Rachel]` in `twitter_thread.md:54`. Fill once the public byline is locked.
17. Add docstrings to the 11 scripts that lack them (§4a).
18. Add `if __name__ == '__main__':` guards to the 9 scripts that lack them (§4b). For `lodo_cv.py` specifically, audit for module-scope side effects.
19. `submission/00_cover_letter.md` — fill `[Editor Name]`, `[Journal Name]`, `[Date]`, reviewer suggestions.
20. `submission/02_ethics_statement.md` line 14 — replace `[Institution]` placeholder.
21. `REPRODUCING.md:18` — clarify the 11→10 cohort reduction at preprocessing step (currently reads like a contradiction with the 10-cohort headline).
22. Consider deleting `results/SANITY_CHECK_REPORT.md` (orphan, currently stale) or refreshing it to match the post-bile-acid state and the 49/49 verify count.
23. `__pycache__/` and `.pytest_cache/` directories on disk — `rm -rf` before release tagging.
24. `CITATION_AUDIT.md` follow-up: live `curl` against `https://doi.org/10.1038/s41591-025-03693-9` and `https://doi.org/10.1101/2025.02.22.639690` to confirm the two 2025 references that the audit could not externally verify.
25. `submission/06_reviewer_responses.md` cites 12 additional papers (Davison & Hinkley 1997, Grinsztajn 2022, Shwartz-Ziv & Armon 2022, Pasolli 2016, Topcuoglu 2020, Lee 2014, Rubinstein 2013, Gur 2015, Castellarin 2012, Kostic 2013, Drewes 2017, Elor & Averbuch-Elor 2022) that are not in `06_references.md`. If any of those are ever sent to a reviewer, they need a paired refs list.

---

## Summary

The repository's *core science* is clean: `verify_results.py` passes all 49 assertions, every headline AUC traces to a CSV, every CSV (except S8) is parseable, no AI / external-consult attribution is present anywhere in tracked content or git log content, and the manuscript markdown is internally consistent with the results CSVs.

The defects fall into two clusters: (1) **stale strings** in derivative files — the bile-acid 9-group update, the 2000→10000 bootstrap revision, and the 38→49 verify-check expansion — that have propagated to ~10 secondary documents but not to every one of them; and (2) **a stale `submission/build/`** that pre-dates the most recent source updates. The single P0 is the malformed `S8_seed_sensitivity.csv` shipping inside `SUBMISSION_BUNDLE.zip`.

Re-run this audit after (a) the concurrent agent finishes, (b) the P0 + P1 punch list is worked, and (c) `submission/build/` and `SUBMISSION_BUNDLE.zip` are regenerated.
