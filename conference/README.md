# Conference materials — CRC metagenomics 10-cohort LODO

**Authors:** Alejandro Velazquez¹, Rachel Selbrede²
**Project root:** `/Users/alexvintera/Desktop/crc-metagenomics/`
**Code & data:** https://github.com/alejandro-publius/crc-metagenomics

This directory holds conference-ready artifacts derived from the manuscript and `results/`. Nothing here is auto-generated — edit and re-render before submission.

---

## File index

| File | Purpose | Target venues |
|------|---------|---------------|
| `poster_outline.md` | A0-portrait scientific poster layout with panel-by-panel poster-ready text, figure paths, and source tables. Hand off to a designer (PowerPoint / Affinity / InDesign). | ISMB poster session, AACR Annual Meeting poster, Gut Microbiota for Health Summit / DDW poster, Microbiome Conference poster, RECOMB poster session. |
| `slides_15min.md` | Marp markdown deck for a 12-15 minute talk (12 slides). Title, motivation, dataset, two methods slides, headline result, negative pathway result, adenoma + SHAP, sensitivity, limitations, conclusions, acknowledgments. | Invited departmental seminars, ISMB COSI talks, AACR mini-symposium, lab group meetings. |
| `slides_3min.md` | Marp markdown deck for a 3-minute lightning talk / RECOMB poster blitz (4 slides). | RECOMB poster blitz, ISMB lightning talks, ASHG late-breaking, departmental lightning sessions, microbiome industry meetups. |
| `abstract_ismb.md` | 250-word structured abstract for ISMB / RECOMB. Methodological emphasis. | ISMB (Microbiome COSI), RECOMB, ML4H, NeurIPS LMRL. |
| `abstract_gut_microbiota_for_health.md` | 250-word structured abstract for the Gut Microbiota for Health Summit / DDW. Clinical translation emphasis. | Gut Microbiota for Health Summit, DDW, UEG Week, Microbiome Movement (CRC track). |
| `abstract_aacr.md` | 250-word structured abstract for AACR Annual Meeting. Cancer-biology emphasis. | AACR Annual Meeting, AACR Special Conference on CRC, SSO Annual Meeting, ESMO. |
| `qr_code_target_url.txt` | Single-line URL for the poster QR code plus a `qrencode` invocation comment. | All posters and slide decks. |
| `README.md` | This file. | — |

---

## Rendering the slides

`slides_15min.md` and `slides_3min.md` are Marp-compatible. With the Marp CLI installed (`npm i -g @marp-team/marp-cli`):

```bash
# from project root
marp conference/slides_15min.md -o conference/slides_15min.pdf
marp conference/slides_3min.md  -o conference/slides_3min.pdf
# or HTML
marp conference/slides_15min.md -o conference/slides_15min.html
```

Figure paths in the decks are relative to `conference/` (`../figures/...`). The Marp CLI resolves them correctly when invoked from project root.

## Generating the QR code

```bash
brew install qrencode   # macOS
qrencode -o conference/qr.png -s 12 -m 2 "$(head -1 conference/qr_code_target_url.txt)"
```

`slides_3min.md` already references `conference/qr.png` on its final slide; the poster bottom strip should embed the same image.

---

## Timing guide

| Format | Duration | File | Notes |
|--------|----------|------|-------|
| Lightning / blitz | 3 min, 4 slides | `slides_3min.md` | One figure (LODO forest), one DeLong result, one QR. |
| Standard contributed talk | 12-15 min + 5 Q&A | `slides_15min.md` | 12 slides at ~1 min each; budget 90 s on the DeLong / pathway-negative result and ~90 s on the SHAP + adenoma model. |
| Poster | Standing presentation, ~60-90 s elevator + 5-10 min walkthrough | `poster_outline.md` | Use Panel 3 (LODO forest) as the elevator visual. Walk left -> center -> right -> bottom strip. |
| Departmental seminar | 45-50 min + Q&A | extend `slides_15min.md` | Add: full Methods detail (5 slides), sensitivity battery breakdown (5 slides), per-cohort SHAP grids, future-work slide. |

---

## Ground-truth numbers (do not edit without updating all artifacts)

- 10 cohorts, 7 countries, 1,522 samples (674 CRC, 665 control, 183 adenoma).
- Case/control LODO uses 1,339 samples (adenomas excluded from the main task).
- Species RF: per-cohort mean LODO AUC **0.807 ± 0.065**; pooled **0.781 (0.757-0.805)**.
- Joint RF pooled AUC 0.756; joint XGBoost pooled 0.766.
- DeLong species_rf vs joint_rf: **z = 3.35, p = 0.0008** (n = 1,339).
- DeLong species_rf vs joint_xgb: z = 2.00, p = 0.046.
- Country-aware LODO impact: ThomasAM_2019_c **0.998 -> 0.836** with YachidaS_2019 excluded.
- Adenoma LODO (4 cohorts, n = 183): H-vs-A RF 0.561 / XGB 0.579; A-vs-CRC RF 0.671 / XGB 0.617.
- Top SHAP species (CRC RF): *G. morbillorum*, *P. micra*, *P. stomatis*, *F. nucleatum*, *S. moorei*.
- Sensitivity: 5 random seeds 0.810 ± 0.002; 20-cell pathway grid 0.794-0.812; demographic 0.800-0.814; ComBat 0.815; external validation (YuJ_2015 + ZellerG_2014 pooled) 0.833.

Sources of truth: `results/baseline_results.csv`, `results/bootstrap_ci.csv`, `results/delong_results.csv`, `results/adenoma_lodo_results.csv`, `results/shap_crc_features.csv`, `results/seed_sensitivity.csv`, `results/sensitivity_thresholds.csv`, `results/confounder_results.csv`, `results/combat_results.csv`, `results/external_validation.csv`, `results/table1.csv`.

---

## Open items — please fill before submission

1. **Author affiliations** for Alex and Rachel (currently `[FILL]` in every artifact). Decide on `Alejandro` vs `Alex` for the public byline; the project uses `Alejandro Velazquez`.
2. **Corresponding author email** for the poster contact strip.
3. **Funding / acknowledgments statement.** Currently shown as "no external funding to declare" — confirm or replace.
4. **HPC / computing acknowledgment** if any (e.g., Berkeley Savio, lab cluster).
5. **Conference deadlines and category selections** — none of the abstracts are yet locked to a specific cycle:
   - AACR Annual Meeting (typically Nov deadline for April meeting)
   - ISMB (typically Feb deadline for July meeting; RECOMB ~Nov for spring)
   - DDW (typically Dec deadline for May meeting)
   - Gut Microbiota for Health Summit (typically Sep-Oct deadline for spring meeting)
   Confirm which cycle is being targeted and adjust title-page / submission category lines accordingly.
6. **Piccinno et al. 2025 citation** (referenced in the slides and poster) — verify volume and page numbers before final submission.
7. **GitHub repo visibility.** The QR code points to `github.com/alejandro-publius/crc-metagenomics` — confirm the repo is public and the README on `main` matches the manuscript before printing the poster.
8. **Designer hand-off format.** `poster_outline.md` is structured for PowerPoint / Affinity Designer / InDesign. If a LaTeX `tikzposter` or `beamerposter` version is preferred, that needs to be generated separately.
