# Pre-Submission Sanity Check Report

**Generated:** 2026-05-18
**Project:** CRC metagenomics 10-cohort meta-analysis
**Repo HEAD:** `9f80987` (main) — `Audit: regenerate all results for 10-cohort dataset, fix stale files and scripts`
**Verification status:** All 49 `verify_results.py` checks PASS.
**Pytest status:** 3 passed, 1 skipped (no failures).

---

## Executive summary — top 5 issues by severity

1. **BLOCKER** — `submission/build/` contains a **stale rebuild** of the
   manuscript and cover letter, frozen at 01:42 while the source markdown was
   updated to current numbers at 02:01. The bundled `SUBMISSION_BUNDLE.zip`
   (and `submission/build/manuscript_complete.md`, `00_cover_letter.md`)
   still report **per-cohort 0.808** (should be 0.807), **2,000-iteration
   bootstrap** (should be 10,000), and **sensitivity spread 0.012** (should
   be 0.018). **Re-run `python3 scripts/build_submission.py`** before any
   bioRxiv / journal upload.

2. **BLOCKER** — Git history contains a commit (`29f628a`) on the
   `origin/claude/clever-einstein-8808ff` branch (and reachable via
   `git log --all`) whose message ends with `Co-Authored-By: Claude Sonnet
   4.6 <noreply@anthropic.com>`. The same branch name itself
   (`claude/clever-einstein-8808ff`) is published as a remote ref. **Before
   making the repo public** (or before submission audit), either delete
   the remote branch and locally prune, or rewrite/squash that commit on
   any branch that will be exposed.

3. **HIGH** — Author / affiliation placeholders survive in deliverables:
   - `submission/05_biorxiv_metadata.md:14,16,17,19` — `Alex [Last Name]¹*,
     Rachel [Last Name]²`, `[Affiliation 1]`, `[Affiliation 2]`, `[email]`,
     `[iD]`. (Same line repeats in `submission/build/05_biorxiv_metadata.md`.)
   - `.zenodo.json:14` — `"affiliation": "[Affiliation] (Biology)"` for
     Selbrede.
   - `CITATION.cff:28` — same `[Affiliation] (Biology)` for Selbrede.
   - `submission/03_author_contributions.md:16` — `[Institution]` for
     compute provider.
   - `submission/00_cover_letter.md:3-5,12,78-80` — `[Editor Name]`,
     `[Journal Name]`, `[Date]`, `[Reviewer 1/2/3 name and affiliation]`.
   - `outreach/press_release_draft.md:31,34` — `https://github.com/[USER]/...`
     and `[Email] | [Phone]`.

4. **HIGH** — Country count is inconsistent across the corpus, and the
   manuscript text **internally contradicts itself**:
   - Truth from `results/table1.csv`: **8 unique countries** (AUT, IND,
     ITA, JPN, USA, DEU, CHN, FRA).
   - `manuscript/markdown/manuscript_complete.md:118` and
     `manuscript/markdown/04_results.md:5` both say "spanning **9
     countries** (Austria, China, France, Germany, India, Italy, Japan,
     USA; Table 1)" — the prose says 9 but the parenthetical list has 8.
   - `submission/06_reviewer_responses.md:17` says "**9 countries** and 3
     continents".
   - Every outreach file (`elevator_pitch.md`, `blog_post_long.md:21`,
     `press_release_draft.md:11`, `linkedin_post.md:9`,
     `twitter_thread.md:16`, `lay_summary_200w.md:5`,
     `qa_for_journalists.md:38`) and `conference/README.md:64`,
     `conference/slides_15min.md:28` say "**7 countries**".
   - Fix: pick **8 countries** and propagate.

5. **HIGH** — Stale bootstrap-iteration count (2,000 vs 10,000) and stale
   sensitivity-spread (0.012 vs 0.018) survive in **build artifacts,
   indexes, and scripts**:
   - `submission/build/manuscript_complete.md:25,46,85,138,163,284,313` —
     "2,000-iteration"; spread 0.012.
   - `submission/build/00_cover_letter.md:62` — spread 0.012.
   - `submission/04_submission_checklist.md:32` and
     `submission/build/04_submission_checklist.md:38` — "iteration count
     specified (**2000**)".
   - `results/supplementary/INDEX.csv:5` and
     `submission/build/supplementary/INDEX.csv:5` — S4 description still
     "**2000-iteration** bootstrap".
   - `scripts/build_supplementary_tables.py:218` — same string in the
     generator.
   - `scripts/build_biorxiv_pdf.py:56` — "**2000-iteration**" caption text
     used when generating the bioRxiv PDF.
   - `submission/07_pre_submission_qa.md:75` — quotes the obsolete "0.012"
     range.

   The source markdown (`01_abstract.md`, `04_results.md`,
   `manuscript_complete.md`, `03_methods.md`, `07_supplementary.md`) and
   `CHANGELOG.md` / `REPRODUCING.md` are correct (10,000 / 0.018). It is
   the **derived** artifacts that drift.

---

## 1. Verification — `python3 scripts/verify_results.py`

**PASS — 49/49 checks pass.** Categories covered: per-cohort means, pooled
prediction file size and uniqueness, DeLong, bootstrap, adenoma LODO, joint
per-fold pathway count, seed sensitivity, confounder adjustment, sensitivity
grid, metadata integrity.

Minor doc-drift: `CHANGELOG.md:37`, `.github/workflows/verify.yml:5`,
`CONTRIBUTING.md:13`, `submission/04_submission_checklist.md:38`,
`submission/build/04_submission_checklist.md:38`, and
`submission/06_reviewer_responses.md:239,241` all still say **"38 checks"**.
**LOW**, suggested fix: search/replace "38" → "49" in these locations.

---

## 2. Cross-file number consistency — 8 headline numbers

| # | Headline number | Authoritative | Stale instances |
|---|---|---|---|
| 1 | Species RF per-cohort mean = **0.807** ± 0.065 | `results/baseline_results.csv`, `baseline_results.md:23`, `01_abstract.md`, `04_results.md`, manuscript_complete (source, line 27/122) | `submission/build/manuscript_complete.md:27,122,124,142,146,335,339,343` show **0.808**; `conference/README.md:66`, `conference/slides_15min.md:70`, `conference/poster_outline.md:69`, `conference/abstract_ismb.md:21` all say **0.808 ± 0.065** |
| 2 | Pooled species RF AUC = **0.781** [0.757, 0.805] | `results/bootstrap_ci.csv` row 12, README, REPRODUCING, all manuscript markdown | Consistent everywhere; no stale hits. |
| 3 | Joint RF pooled = **0.756** [0.731, 0.781] | `results/bootstrap_ci.csv`, `delong_results.csv` | Consistent everywhere. |
| 4 | DeLong species_rf vs joint_rf **p = 0.0008** (z = 3.35) | `results/delong_results.csv` (0.000801) | Consistent. |
| 5 | Adenoma H-vs-A RF = **0.561** | `results/adenoma_lodo_results.csv:2` (0.5606) | Consistent. |
| 6 | Adenoma A-vs-CRC RF = **0.671** | `results/adenoma_lodo_results.csv` (0.6714) | Consistent. |
| 7 | Bootstrap N = **10,000** | `scripts/bootstrap_ci.py:21` (N_BOOT=10000), README, REPRODUCING, source markdown | **Stale `2,000` / `2000`** in: `submission/build/manuscript_complete.md` (4 places), `submission/build/supplementary/INDEX.csv`, `submission/build/04_submission_checklist.md`, `submission/04_submission_checklist.md:32`, `results/supplementary/INDEX.csv:5`, `scripts/build_supplementary_tables.py:218`, `scripts/build_biorxiv_pdf.py:56`, `submission/06_reviewer_responses.md` (defends the 2,000 choice — text needs updating to defend 10,000 instead, *or* the analysis re-run defended). |
| 8 | Sensitivity grid spread = **0.018** (range 0.794–0.812) | `results/sensitivity_thresholds.csv`, source markdown, README:13/59, `04_results.md:25`, `07_supplementary.md:27` | **Stale `0.012` (range 0.798–0.810)** in: `submission/build/manuscript_complete.md:138,284`, `submission/build/00_cover_letter.md:62`, `submission/07_pre_submission_qa.md:75`. |

Severity per stale set: **BLOCKER** for the `submission/build/*` files (they
ship to the journal); **HIGH** for the supplementary index CSV and the
generator scripts; **MEDIUM** for `submission/06_reviewer_responses.md` and
`07_pre_submission_qa.md` which still defend the old design choice.

**Recommended fix:** Re-run `python3 scripts/build_submission.py` after
updating the two `scripts/build_*.py` strings, and edit `06_reviewer_responses`
to defend "10,000" rather than "2,000".

---

## 3. AI attribution — grep across all text files

**Working tree:** Only one hit, and it is benign by design:
`scripts/test_submission_build.py:136` — `forbidden = ("Co-Authored-By",
"Anthropic", "Claude")`. This is the **negative-test guard**; leave as-is.

**Git history (BLOCKER if repo will be public):**
- Commit `29f628a` (`Expand to 10 cohorts, fix adenoma LODO …`) — message
  ends with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
  Reachable via `git log --all`; sits on `origin/claude/clever-einstein-8808ff`,
  not on `main`.
- Branch name `claude/clever-einstein-8808ff` is itself an attribution
  string published on the remote.
- Merge commits `75b6e06` (`Merge … claude/clever-einstein-8808ff`) and
  `de7de6b` (`Merge pull request #2 from
  alejandro-publius/claude/clever-einstein-8808ff`) also retain the
  branch name in their messages.

**Suggested fix:** `git push origin --delete claude/clever-einstein-8808ff`
and prune locally; consider squashing/amending the two merge messages on a
fresh branch before making the repo public. Optionally `git filter-repo`
to scrub the commit body of `29f628a` (its content is otherwise on `main`
in cleaned form at `c71b2b2` / `9f80987`).

No `Claude` / `Anthropic` / `GPT` / `LLM` strings in any working-tree
`.md`, `.py`, `.yml`, `.json`, `.cff`, or `.txt` file (other than the
test guard above).

---

## 4. Author name consistency

- Authoritative form (used in README, LICENSE, CITATION.cff, .zenodo.json,
  title page, manuscript_complete, all conference and outreach byline lines):
  **Alejandro Velazquez** and **Rachel Selbrede**, byline email
  `thealexschroeder@gmail.com`.
- `scripts/add_covariates.py:16`, `scripts/external_validation.py:25`,
  `scripts/generate_table1.py:20` use the short form **"Alex Velazquez"**.
  **LOW** — inside Python docstrings only; either align to `Alejandro` or
  leave as the author's preferred informal byline.
- **Remaining placeholders** (HIGH, see Executive Summary item 3):
  - `submission/05_biorxiv_metadata.md` and its build copy still have
    `Alex [Last Name]¹*, Rachel [Last Name]²`, `[Affiliation 1/2]`,
    `[email]`, `[iD]`.
  - `.zenodo.json:14` and `CITATION.cff:28` have `[Affiliation] (Biology)`
    for Selbrede.
  - `submission/03_author_contributions.md:16` has `[Institution]`.
  - `submission/00_cover_letter.md:3-5,78-80` has unfilled `[Editor]`,
    `[Journal]`, `[Date]`, three `[Reviewer N name and affiliation]`.
  - `outreach/press_release_draft.md:31,34` has `[USER]`, `[Email]`,
    `[Phone]`.

No misspellings of *Velazquez* or *Selbrede* found.

---

## 5. Script integrity

**All Python files parse via `ast.parse` (0 failures).** Inventory:

`scripts/` (37 .py files):
add_covariates, adenoma_counts, adenoma_lodo, auc_comparison,
batch_correction, bio_pathway_shortlist, bootstrap_ci, build_biorxiv_pdf,
build_submission, build_supplementary_tables, check_label_dist,
confounder_adjustment, export_data.R, external_validation,
figure1_forest_plot, figure5_shap_three_panel, filter_pathways, find_nans,
generate_figures, generate_table1, lodo_cv, merge_pathways, preprocessing,
sanity_check, seed_sensitivity, sensitivity_analysis, shap_adenoma,
shap_analysis, shap_xgb, test_submission_build, train_adenoma,
train_baseline, train_joint, validate_pathways, verify_results,
plus `audit_subject_ids.R`, `run_robustness.sh`.

`scripts/diagnostics/`: calibration, confusion_matrices,
per_cohort_sens_spec, roc_pr_curves, subgroup_analysis. (All parse.)

**5 key scripts entrypoint check:**

| Script | `__main__` block | Invokable as script |
|---|---|---|
| `train_baseline.py` | YES | YES |
| `train_joint.py` | YES | YES |
| `verify_results.py` | YES | YES (verified pass) |
| `lodo_cv.py` | **NO** | Library only — imported by other scripts. Not a defect. |
| `auc_comparison.py` | **NO** | Library + side-effect at import. `REPRODUCING.md:45` invokes it as a script; running it relies on top-level statements. **MEDIUM**: either add an `if __name__ == "__main__":` guard or document that the script runs on import. |

---

## 6. Results CSV integrity

All 41 CSVs in `results/`, `results/diagnostics/`, `results/supplementary/`
exist, non-empty, no all-NaN columns. **Verified row counts:**

- `preds_species_rf.csv`, `preds_joint_rf.csv`, `preds_joint_xgb.csv`,
  `preds_bio_pathway_rf.csv`: all 1339 rows, 4 cols.
- `bootstrap_ci.csv`: 33 rows (≥30 expected). PASS.
- `delong_results.csv`: 3 rows. PASS (≥2 expected).
- `seed_sensitivity.csv`: 5 rows.
- `sensitivity_thresholds.csv`: 20 rows (= 4×5 grid).
- `adenoma_lodo_results.csv`: 4 rows.
- `table1.csv`: 11 rows (10 cohorts + total).

**One file fails strict pandas parse — `results/supplementary/S8_seed_sensitivity.csv`** (`Error tokenizing data. C error: Expected 3 fields in line 8, saw 7`). The file actually contains two tables stitched into one CSV — a 5-row seed table followed by a blank line and a 1-row summary block with 7 columns. **MEDIUM**: either split into `S8a` / `S8b` or use a single consistent header row. Currently any downstream `pd.read_csv` on this file will crash.

**Cross-check of prediction files:** all three `preds_*.csv` share identical sample_id sets and identical cohort assignments (verified row-by-row after sort).

---

## 7. Figure integrity

All 19 PNG/PDF files in `figures/`, `figures/diagnostics/`,
`manuscript/figures/` are valid images at **~300 DPI** (299.9994). No
file > 5 MB. No file < 5 KB (smallest is 80.7 KB,
`figures/fig4_external_validation.png`). All publication-quality.

Two **legacy** PNG copies are kept alongside the canonical manuscript
versions for `REPRODUCING.md` backward compatibility:
`figures/figure5_three_panel_shap.png` (identical bytes to
`manuscript/figures/Figure4_Three_Panel_SHAP.png`) and the `figures/fig*.png`
family. **LOW**: harmless duplication; mark as such in the README or remove
the `figures/figure5_three_panel_shap.png` once REPRODUCING is updated.

---

## 8. Manuscript integrity

- **All 10 `manuscript/CRC_*.docx` files present** (Abstract, Discussion,
  Introduction, Manuscript_Complete, Methods, References, Results,
  Table1, Title_Page, Supplementary_Tables).
- **`manuscript/markdown/manuscript_complete.md` exists** (49,358 bytes,
  mtime 02:01) and structurally references all sections (Abstract,
  Introduction, Methods, Results, Figures 1–4, Discussion, References,
  Table 1, Supplementary Tables S1–S10).
- **No `[citation]` / `[ref]` / `[CITATION]` placeholders** found.
- **All citations in references list**, verified surname-by-surname:
  Austin, Bellman, Chen, DeLong, Franzosa, Johnson, Lundberg, Pasolli,
  Pedregosa, Piccinno, Sun & Xu, Sun et al. 2025, Sung, Thomas, Trunk,
  Truong, Wirbel, Xi, Yachida — all present in `06_references.md`.
- Required citations from the request — **all present**: Thomas 2019 (#14),
  Wirbel 2019 (#17), Pasolli 2017 (#8), Truong 2015 (#16), Franzosa 2018
  (#5), DeLong 1988 (#4), Sun and Xu 2014 (#11).
- **No `manuscript/markdown/06_references.md` ↔ docx mismatch detected.**
- **Figure path references in markdown** point to figure captions (no
  inline `![]()` embeds in source markdown — figures are inserted at
  layout time in the docx build). All 4 caption blocks (Figures 1–4)
  match files on disk in `manuscript/figures/`.

**Internal manuscript contradictions** worth flagging:
- "spanning **9 countries** (Austria, China, France, Germany, India,
  Italy, Japan, USA; Table 1)" — list contains 8. (Items 4 above.)
- `manuscript_complete.md:118` says "Per-cohort sample sizes ranged from
  60 (GuptaA_2019, ThomasAM_2018b) to **575** (YachidaS_2019)" but
  `Table 1` row YachidaS_2019 has N=575 — consistent. Per-fold AUC table
  on line 322 uses `n_test = 508` for the same cohort (the case/control
  subset excluding 67 adenomas). Both are correct; the wording is fine
  but a reader might flag the discrepancy — consider adding a
  parenthetical "(case/control = 508 of 575)" in the Results sentence.
  **LOW.**
- Per-cohort discussion (`manuscript_complete.md:198`) says ComBat is
  "within **0.002** of the uncorrected baseline" but the matching
  Results paragraph (line 146) says "**Δ +0.008**, … corrected 0.815 vs
  uncorrected 0.807". The 0.002 number is stale. **MEDIUM**, fix the
  Discussion line. `05_discussion.md:25` is already correct (0.008).

---

## 9. Submission package

**Present in `submission/`:**
- `00_cover_letter.md`, `01_data_availability.md`, `02_ethics_statement.md`,
  `03_author_contributions.md`, `04_submission_checklist.md` (PRESENT),
  `05_biorxiv_metadata.md`, `06_reviewer_responses.md`,
  `07_pre_submission_qa.md`, `MANIFEST.md`.

**`submission/build/SUBMISSION_BUNDLE.zip` exists** (2.5 MB; 38 files;
~3 MB uncompressed). Contents:
`00..05_*.md`, `CRC_Manuscript_Complete.docx`, `README.md`,
`manuscript_complete.md`, `figures/` (4 main figure PDFs + 5 diagnostic
PNGs + `main/` mirrored copies), `supplementary/` (S1–S10 + INDEX +
`Supplementary_Tables.docx`).

**`submission/MANIFEST.md` is current** (generated 2026-05-18T08:49:35Z)
**but does NOT include `06_reviewer_responses.md` or
`07_pre_submission_qa.md`** (HIGH if these are meant to ship — they live
only in the source tree). The manifest also warns:
- "Could not generate manuscript.pdf: neither pandoc nor soffice produced
  output." → **HIGH**: the bundle is missing the canonical PDF. Either
  install pandoc/soffice and re-build, or add a manual `.pdf` export from
  the `.docx` to `submission/build/` before upload.
- "Neither pypdf nor PyPDF2 is installed; cannot merge figure PDFs."
  → MEDIUM: figures are bundled individually instead of as one merged
  PDF, which most journal portals accept fine.

**`SUBMISSION_BUNDLE.zip` is stale relative to source markdown.** Confirmed:
`submission/build/manuscript_complete.md` mtime = 01:42 (with 0.808 /
2,000 / 0.012), but `manuscript/markdown/manuscript_complete.md` mtime
= 02:01 (with 0.807 / 10,000 / 0.018). **Rebuild before submission.**

`submission/04_submission_checklist.md` exists, but every checkbox is
unchecked. **HIGH**: fill in DONE/PARTIAL before submitting.

---

## 10. Infrastructure

| File | Present | Valid | Notes |
|---|---|---|---|
| `.github/workflows/verify.yml` | YES | valid YAML | references `verify_results.py`; correct branches. Stale: comment "38 checks" (now 49). |
| `Dockerfile` | YES | parses | Python 3.11-slim base; minimal deps. |
| `environment.yml` | YES | valid YAML | conda env mirrors `requirements.lock` + R toolchain. |
| `.zenodo.json` | YES | valid JSON | **`[Affiliation]` placeholder** for Selbrede (HIGH). |
| `CITATION.cff` | YES | valid YAML | **`[Affiliation]` placeholder** for Selbrede (HIGH). |
| `LICENSE` | YES | n/a | MIT, real names. |
| `CHANGELOG.md` | YES | n/a | v1.0.0 entry. Stale "38 headline-number checks" on line 37. |
| `CONTRIBUTING.md` | YES | n/a | Stale "38 checks" on line 13. |
| `tests/test_lodo_cv.py` | YES | parses | `pytest tests/ -v`: **3 passed, 1 skipped** (no failures). |

---

## 11. Git state

**`git status`** (working tree dirty):

- *Modified, not staged:* `.gitignore`, `LICENSE`, `README.md`,
  `REPRODUCING.md`, all 10 `manuscript/CRC_*.docx`,
  `manuscript/figures/Figure1_Forest_Plot.{pdf,png}`, 8 results CSVs/MDs,
  9 scripts including `verify_results.py`, `external_validation.py`,
  `figure1_forest_plot.py`, `figure5_shap_three_panel.py`.
- *Untracked:* `.dockerignore`, `.github/`, `.zenodo.json`, `CHANGELOG.md`,
  `CITATION.cff`, `CONTRIBUTING.md`, `Dockerfile`, `conference/`,
  `environment.yml`, `figures/diagnostics/`, `manuscript/markdown/`,
  `outreach/`, `results/diagnostics/`, `results/supplementary/`,
  `scripts/build_biorxiv_pdf.py`, `scripts/build_submission.py`,
  `scripts/build_supplementary_tables.py`, `scripts/diagnostics/`,
  `scripts/test_submission_build.py`, `submission/`, `tests/`.

**HIGH**: A very large amount of submission-critical material is **not yet
under version control**, including the whole `submission/`, `manuscript/markdown/`,
`conference/`, `outreach/`, and `tests/` trees, and `CITATION.cff` /
`.zenodo.json` / `Dockerfile` / `environment.yml`. Stage and commit before
tagging for Zenodo/DOI.

**`git log --oneline -20`** (top of main):
```
9f80987 Audit: regenerate all results for 10-cohort dataset, fix stale files and scripts
c71b2b2 Expand to 10 cohorts, fix adenoma LODO, add country-aware CV and bio pathway shortlist
d158c6a Update LICENSE
... (license / figure / audit pass commits)
```

**`git log --all --grep`** finds the Co-Authored-By/Claude attribution at
commit `29f628a` (origin/claude/clever-einstein-8808ff). See section 3.
**BLOCKER for public release.**

**`.git/config` remotes:** `origin = https://github.com/alejandro-publius/crc-metagenomics.git` — single remote, expected.

---

## 12. Memory / orphans / cleanup

- **`.RData` (1.3 MB)** and **`.Rhistory` (4.6 KB)** at repo root —
  `.gitignore` excludes them, so they will not be committed, but they
  should be deleted from the working tree before zipping for any review:
  `.RData` will end up in any naive "zip up the project" workflow.
- **`__pycache__/`** present in `scripts/__pycache__/` (13 `.pyc` files)
  — `.gitignore`'d, but again will leak into a manual zip. Suggest
  `find . -name __pycache__ -exec rm -rf {} +` before any archive
  operation.
- **`.pytest_cache/`** present at repo root — `.gitignore`'d.
- **No `.DS_Store`** files found.
- **No files >50 MB outside `data/raw/`.** The three large CSVs
  (`data/processed/pathway_abundance_filtered.csv`,
  `data/raw/pathway_abundance.csv`,
  `data/raw/pathway_chunks/YachidaS_2019.csv`) are correctly listed in
  `.gitignore`. Good — no GitHub-size-limit risk.
- **Orphans / possibly-stale:** legacy `figures/figure5_three_panel_shap.png`
  duplicates `manuscript/figures/Figure4_Three_Panel_SHAP.png` exactly
  (same byte size 394,910). Kept intentionally per
  `scripts/figure5_shap_three_panel.py:42` comment. LOW.

---

## 13. Reproducibility check (REPRODUCING.md)

- All scripts referenced in `REPRODUCING.md` exist in `scripts/`.
- All `# expect:` lines cross-check against the current CSVs (verified
  via `scripts/verify_results.py`).
- **13 scripts NOT mentioned in REPRODUCING.md**:
  `add_covariates.py`, `build_biorxiv_pdf.py`, `build_submission.py`,
  `build_supplementary_tables.py`, `external_validation.py`,
  `lodo_cv.py`, `train_adenoma.py`, `test_submission_build.py`, and all 5
  diagnostics (`calibration.py`, `confusion_matrices.py`,
  `per_cohort_sens_spec.py`, `roc_pr_curves.py`, `subgroup_analysis.py`).
  - `lodo_cv.py` is a library — fine.
  - `build_*.py` are post-processing — fine to omit from the analytical
    pipeline section, but should at least be mentioned under a "Build /
    package" subsection.
  - `add_covariates.py`, `external_validation.py`, `train_adenoma.py`
    are first-class analysis scripts whose outputs *are* shipped in
    `results/` and the submission — **MEDIUM**: add them to
    REPRODUCING.md (with `# expect: …` lines).
  - The diagnostic scripts are documented in `results/diagnostics/README.md`
    but not in REPRODUCING.md — LOW; cross-link.

---

## 14. Citation / biology verification

- **`Piccinno et al. 2025`** — present in `manuscript_complete.md:38,235`,
  `02_introduction.md:5`, `05_discussion.md:13`, `06_references.md:21`
  (with DOI `10.1038/s41591-025-03693-9`, vol 31, pp 2416–2429),
  `submission/06_reviewer_responses.md:15,31,77,133`,
  `submission/07_pre_submission_qa.md:23,27`,
  `conference/slides_15min.md:100`, `conference/poster_outline.md:140,161`,
  `conference/README.md:91`. The conference README itself flags this as
  needing verification ("Piccinno et al. 2025 citation … verify volume
  and page numbers before final submission"). **HIGH** — verify the
  citation exists and the metadata is correct before submission;
  `s41591-025-03693-9` is a *Nature Medicine* DOI in the 2025 issue
  range, so plausible, but confirm independently.
- **All in-text author-year citations resolve in `06_references.md`** —
  19 unique surnames, all in references. ✓
- **Oral pathobionts named consistently** as *Fusobacterium nucleatum*,
  *Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*
  across all manuscript, abstract, conference, outreach, and SHAP CSV
  files. No misspellings detected.

---

## 15. Outreach honesty check

- **`outreach/blog_post_long.md:83`** — "**A senior collaborator pushed
  back** and pointed out that 'a thing everyone expected to work, didn't'
  is exactly the kind of finding the field needs more of. They were
  right." — This is the "senior collaborator anecdote" flagged in your
  instructions. Given the author is described as a Berkeley CS
  undergraduate without an advisor named anywhere in the manuscript or
  acknowledgments, and that Rachel Selbrede is the only named collaborator
  (and is identified as a biology co-author, not a "senior collaborator
  in the field"), **HIGH — verify this anecdote is true.** If no such
  conversation occurred, delete the paragraph or attribute generically
  ("a more experienced collaborator", "Rachel pointed out…"). The
  acknowledgments and CONTRIBUTORS list no other collaborators.

- **`outreach/press_release_draft.md` quotes** —
  - Velazquez quote (line 19): consistent with the paper's headline claim
    (cross-population testing lowers reported AUCs). ✓
  - Selbrede quote (line 21): "Species like Fusobacterium nucleatum and
    Parvimonas micra normally live in the mouth, and they appear to
    colonize colorectal tumors as the disease progresses — they are
    doing real work in distinguishing later from earlier disease stages."
    — Consistent with the SHAP/A-vs-CRC findings in the paper. ✓
  - Press release lines 15, 17 (numerical claims): all consistent with
    the manuscript headline numbers (0.836 corrected ThomasAM_2019_c,
    0.781 pooled, 0.561 H-vs-A, 0.671 A-vs-CRC).
  - **HIGH placeholders**: `[USER]` GitHub URL (line 31), `[Email]`
    and `[Phone]` media contact (line 34).

- **`outreach/lay_summary_*.md`**, **`outreach/linkedin_post.md`**,
  **`outreach/twitter_thread.md`**, **`outreach/qa_for_journalists.md`**:
  all numerical claims match the manuscript. Only issue is the "7
  countries" / "8 countries" / "9 countries" inconsistency (item 4 above).

---

## Final clean checklist — what still needs human attention

**Blockers (do not submit until resolved):**

1. [ ] Re-run `python3 scripts/build_submission.py` after fixing the
       stale strings in the source. The current
       `submission/build/SUBMISSION_BUNDLE.zip` contains **0.808** (vs
       0.807), **2,000-iteration** bootstrap (vs 10,000), and **spread
       0.012** (vs 0.018).
2. [ ] Delete remote branch `claude/clever-einstein-8808ff` and
       (optionally) rewrite commit `29f628a` to strip the
       `Co-Authored-By: Claude` trailer. Confirm `git log --all --grep
       Claude` returns empty before making the repo public.
3. [ ] Generate the manuscript PDF (install pandoc with a LaTeX engine,
       or `brew install --cask libreoffice`, then re-run
       `scripts/build_submission.py`) — the manifest currently warns
       "Could not generate manuscript.pdf".

**Highs (will be embarrassing if submitted as-is):**

4. [ ] Replace `[Last Name]`, `[Affiliation 1/2]`, `[email]`, `[iD]` in
       `submission/05_biorxiv_metadata.md` (and `submission/build/`
       copy).
5. [ ] Replace `[Affiliation] (Biology)` for Selbrede in
       **`.zenodo.json:14`** and **`CITATION.cff:28`**.
6. [ ] Replace `[Institution]` in `submission/03_author_contributions.md:16`.
7. [ ] Fill `[Editor Name]`, `[Journal Name]`, `[Date]`, and the three
       `[Reviewer N name and affiliation]` slots in
       `submission/00_cover_letter.md`.
8. [ ] Resolve country count: **8 countries** is the truth from
       `results/table1.csv`. Update `manuscript_complete.md:118` and
       `04_results.md:5` ("9 countries" → "8 countries"; the
       parenthetical list is already correct at 8 names), then
       `submission/06_reviewer_responses.md:17` ("9 countries" → "8");
       then all conference / outreach files currently saying "7
       countries".
9. [ ] Update `scripts/build_supplementary_tables.py:218` and
       `scripts/build_biorxiv_pdf.py:56` to reference 10,000 (not 2,000)
       bootstrap iterations, then regenerate `results/supplementary/INDEX.csv`.
10. [ ] Update `submission/06_reviewer_responses.md` to defend 10,000
        iterations rather than 2,000 (the analytical pipeline now uses
        10,000 per `scripts/bootstrap_ci.py:21`).
11. [ ] Update `submission/04_submission_checklist.md:32` —
        "(2000)" → "(10000)" — and check the boxes that are actually done.
12. [ ] Include `submission/06_reviewer_responses.md` and
        `submission/07_pre_submission_qa.md` in
        `submission/MANIFEST.md` (or document explicitly that they are
        internal-only).
13. [ ] Verify the **Piccinno et al. 2025** *Nature Medicine* citation
        independently (the conference README flags this as needing
        verification).
14. [ ] Fix `outreach/press_release_draft.md` `[USER]`, `[Email]`,
        `[Phone]` placeholders.
15. [ ] Verify the "senior collaborator pushed back" anecdote in
        `outreach/blog_post_long.md:83` — if no senior collaborator
        exists, rewrite or remove.
16. [ ] Stage and commit all untracked files (`submission/`,
        `manuscript/markdown/`, `conference/`, `outreach/`,
        `.github/`, `tests/`, infra files) before tagging the
        submission commit.

**Mediums:**

17. [ ] Reconcile internal contradiction in
        `manuscript/markdown/manuscript_complete.md`: line 198 says
        "ComBat within **0.002** of the uncorrected baseline" but the
        Results paragraph (line 146) says "Δ +**0.008**, corrected
        0.815 vs uncorrected 0.807". Pick 0.008 (matches
        `combat_results.csv`).
18. [ ] Add `if __name__ == "__main__":` guard to
        `scripts/auc_comparison.py`, or document that it executes at
        import time. `REPRODUCING.md:45` invokes it as a script.
19. [ ] Fix `results/supplementary/S8_seed_sensitivity.csv` — it
        concatenates two tables with different column counts and breaks
        `pd.read_csv` strict parse. Split into `S8a`/`S8b` or use a
        single header.
20. [ ] Add `add_covariates.py`, `external_validation.py`,
        `train_adenoma.py` to `REPRODUCING.md` (with `# expect: …`
        lines).

**Lows (cosmetic / nice-to-have):**

21. [ ] Replace "38 checks" with "49 checks" in `CHANGELOG.md:37`,
        `CONTRIBUTING.md:13`, `.github/workflows/verify.yml:5`,
        `submission/04_submission_checklist.md:38`,
        `submission/06_reviewer_responses.md:239,241`.
22. [ ] Add a parenthetical "(case/control = 508 of 575)" to the
        YachidaS_2019 sentence in
        `manuscript/markdown/manuscript_complete.md:122` to clarify why
        per-fold n_test = 508 rather than the Table 1 total of 575.
23. [ ] Decide whether `figures/figure5_three_panel_shap.png` is still
        needed (it's a byte-identical duplicate of
        `manuscript/figures/Figure4_Three_Panel_SHAP.png`).
24. [ ] Standardize Python docstring author name across
        `scripts/add_covariates.py`, `scripts/external_validation.py`,
        `scripts/generate_table1.py` — currently "Alex Velazquez"; all
        public byline material uses "Alejandro Velazquez".
25. [ ] Delete `.RData` and `.Rhistory` from the working tree, and run
        `find . -name __pycache__ -exec rm -rf {} +` before any manual
        zip/upload (`.gitignore` covers them for git, but not for naive
        archive operations).
