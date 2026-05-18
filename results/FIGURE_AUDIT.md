# Figure Audit — Publication Readiness

Date: 2026-05-18
Scope: `figures/` (including `figures/diagnostics/`) and `manuscript/figures/`
Mode: Read-only. No figure files were modified.

## Methodology

Each PNG was opened with PIL; each PDF with pikepdf. For every file we recorded:

- Pixel dimensions / MediaBox (pt, in)
- Embedded DPI (PNG metadata; PDF: vector vs. embedded raster DPI estimated from `/XObject` widths divided by page width in inches)
- File size
- Color mode
- Rendered-content sanity (PNG): RGB pixel std, fraction of near-white pixels (>245 on all channels), and per-corner / center 30x30 patch std

Flagging rules:

- **P0** (FAIL) — file < 5 KB, image > 98% near-white, or pixel std < 5 (probably blank/broken)
- **P1** — DPI < 300 (FAIL for visually-significant figures; POLISH for the rest)
- **P2** — PNG file size > 5 MB (POLISH; oversize)

For visually-significant figures (`figures/fig*.png`, `manuscript/figures/Figure*.png`, `figures/visual_abstract.{png,pdf}`, `figures/diagnostics/*.png`) we additionally measured:

- Text-legibility heuristic: vertical dark-run lengths sampled from 50-pixel column strips at 200-pixel intervals; reported as `min`, `p10`, `median`, and `frac_lt16` (fraction of glyph-sized runs under 16 px). Runs of length 4-40 px were treated as plausible glyph rows.
- Title presence: any dark content in the top 8% of the canvas.
- Legend presence: dark content in the top-right (rows 5-35%, cols 65-98%) or bottom-right (rows 65-95%, cols 65-98%) regions.

## Headline counts

- Files audited: **38** (32 PNG + 6 PDF)
- **PASS: 36**
- **POLISH: 0**
- **FAIL: 2** (both P1)
- **P0: 0** | **P1: 2** | **P2: 0**

## Special checks

| Check | File | Result |
|---|---|---|
| Forest plot: 10 cohorts + pooled | `manuscript/figures/Figure1_Forest_Plot.png` | **PASS (likely)** — 13 distinct text rows in left margin (heights 31-42 px). 11 cohort/pooled rows + header lines is consistent. Visual confirmation recommended. |
| Three-panel SHAP: 3 panels | `manuscript/figures/Figure4_Three_Panel_SHAP.png` | **PASS** — 3 horizontal content bands (widths 640 / 922 / 826 px) detected on a 5496x1973 canvas. |
| Three-panel SHAP: 3 panels | `figures/figure5_three_panel_shap.png` | **PASS** — identical bytes and layout to `Figure4_Three_Panel_SHAP.png` (both 394,910 B, 5496x1973). Likely the same artifact duplicated; consider deduplicating. |
| Visual abstract: 4 panels @ 600 DPI | `figures/visual_abstract.png` | **PASS** — 4200x4200 px at 600.0 DPI. Column-wise content shows 2 panels (widths 1535 / 1646 px); row-wise shows multiple bands consistent with a 2x2 grid plus titles/captions. |
| Visual abstract PDF | `figures/visual_abstract.pdf` | **PASS** — vector PDF, 7.0 x 7.0 in, 61 KB. |

## Per-file table

Columns: file (relative to project root) | dimensions (px or pt for PDF) | DPI | size (B) | status | flags | notes

### figures/ (top-level)

| File | Dimensions | DPI | Size | Status | Flags | Notes |
|---|---|---|---|---|---|---|
| figures/fig1_lodo_auc.png | 2700 x 1800 | 300.0 | 180,679 | PASS | - | RGBA; std 67; near-white 0.81; title+legend OK; text p10=5 (axis ticks), median=27 |
| figures/fig2_shap_crc.png | 2400 x 1800 | 300.0 | 168,304 | PASS | - | RGBA; title+legend OK; text median=25 |
| figures/fig3_adenoma.png | 1800 x 1500 | 300.0 | 93,703 | PASS | - | RGBA; title+legend OK; text median=23 |
| figures/fig4_external_validation.png | 2100 x 1500 | 300.0 | 80,691 | PASS | - | RGBA; text median=8 (frac_lt16=0.51) — many short runs are tick labels; visual review recommended for axis label size |
| figures/figure5_three_panel_shap.png | 5496 x 1973 | 300.0 | 394,910 | PASS | - | RGBA; identical bytes to `manuscript/figures/Figure4_Three_Panel_SHAP.png`; 3 panels confirmed |
| figures/visual_abstract.png | 4200 x 4200 | 600.0 | 857,261 | PASS | - | RGBA; 600 DPI as required; 2x2 grid layout confirmed |
| figures/visual_abstract.pdf | 504 x 504 pt (7.0 x 7.0 in) | vector | 61,677 | PASS | - | Vector; 1 page |

### manuscript/figures/

| File | Dimensions | DPI | Size | Status | Flags | Notes |
|---|---|---|---|---|---|---|
| manuscript/figures/Figure1_Forest_Plot.png | 2700 x 2250 | 300.0 | 221,840 | PASS | - | RGBA; 13 left-margin text rows (cohort + pooled expected = 11; +header) |
| manuscript/figures/Figure1_Forest_Plot.pdf | 648 x 540 pt (9.0 x 7.5 in) | vector | 35,631 | PASS | - | Vector |
| manuscript/figures/Figure2_ROC_Curves.png | 1725 x 1762 | 300.0 | 155,913 | PASS | - | RGBA; near-white 0.95 (sparse ROC plot — expected) |
| manuscript/figures/Figure2_ROC_Curves.pdf | 414.2 x 422.9 pt (5.75 x 5.87 in) | vector | 26,003 | PASS | - | Vector |
| manuscript/figures/Figure3_SHAP_Importance.png | 3570 x 1941 | 300.0 | 405,854 | PASS | - | RGBA; text median=27, frac_lt16=0.04 (good) |
| manuscript/figures/Figure3_SHAP_Importance.pdf | 857 x 465.8 pt (11.9 x 6.47 in) | vector | 39,455 | PASS | - | Vector |
| manuscript/figures/Figure4_Three_Panel_SHAP.png | 5496 x 1973 | 300.0 | 394,910 | PASS | - | RGBA; 3 panels confirmed; identical to figures/figure5_three_panel_shap.png |
| manuscript/figures/Figure4_Three_Panel_SHAP.pdf | 1319.8 x 473.0 pt (18.33 x 6.57 in) | vector | 43,205 | PASS | - | Vector; very wide aspect — verify it fits the journal column |

### figures/diagnostics/

| File | Dimensions | DPI | Size | Status | Flags | Notes |
|---|---|---|---|---|---|---|
| alpha_diversity.png | 4770 x 1853 | 300.0 | 450,145 | PASS | - | text median=16, frac_lt16=0.48 — many small ticks (multi-panel) |
| base_rate_ppv.png | 3569 x 1544 | 300.0 | 252,787 | PASS | - | near-white 0.95 |
| calibration_mechanism.png | 3928 x 1674 | 300.0 | 334,319 | PASS | - | text median=30, frac_lt16=0.08 (good) |
| calibration_reliability.png | 3359 x 1889 | 300.0 | 296,273 | PASS | - | OK |
| cohort_composition.png | 3269 x 1468 | 300.0 | 209,602 | PASS | - | OK |
| confusion_matrices.png | 2180 x 3297 | 300.0 | 372,285 | PASS | - | text median=30 (good) |
| **cross_disease_specificity.png** | **2181 x 940** | **160.0** | **191,619** | **FAIL** | **P1** | **DPI 160 < 300; rasterize at 300+ for print** |
| cv_methodology.png | 2369 x 1409 | 300.0 | 191,319 | PASS | - | OK |
| decision_curves.png | 2669 x 1617 | 300.0 | 200,997 | PASS | - | near-white 0.95 |
| depth_distribution.png | 3269 x 1468 | 300.0 | 335,609 | PASS | - | text p10=4 median=5 frac_lt16=0.65 — histogram/dense plot; review legibility |
| depth_vs_fnucleatum_shap.png | 1986 x 1675 | 300.0 | 185,141 | PASS | - | near-white 0.94 |
| **hannigan_inclusion_sensitivity.png** | **2234 x 816** | **150.0** | **115,805** | **FAIL** | **P1** | **DPI 150 < 300; rasterize at 300+ for print** |
| lift_curves.png | 3569 x 1545 | 300.0 | 343,858 | PASS | - | near-white 0.94 |
| minimum_panel.png | 2369 x 1620 | 300.0 | 243,117 | PASS | - | near-white 0.93 |
| pcoa_bray_curtis.png | 3869 x 1619 | 300.0 | 893,265 | PASS | - | largest PNG; well under 5 MB cap |
| per_cohort_ppv.png | 4019 x 1710 | 300.0 | 305,600 | PASS | - | OK |
| per_cohort_sens_spec.png | 3719 x 1618 | 300.0 | 277,692 | PASS | - | OK |
| permutation_vs_shap.png | 2291 x 1906 | 300.0 | 242,980 | PASS | - | near-white 0.95 |
| power_curve.png | 2219 x 1470 | 300.0 | 158,781 | PASS | - | near-white 0.95 |
| roc_pr_pooled.png | 3569 x 1618 | 300.0 | 296,356 | PASS | - | near-white 0.94 |
| sens_at_fixed_spec.png | 2520 x 1620 | 300.0 | 165,888 | PASS | - | OK |
| subgroup_auc.png | 2819 x 2055 | 300.0 | 253,257 | PASS | - | near-white 0.95 |
| top_species_heatmap.png | 3089 x 1770 | 300.0 | 361,515 | PASS | - | OK |

All PNGs in the audit are RGBA mode. All PDFs are vector (no rasterized fallbacks).

## Worst 5 offenders

1. **`figures/diagnostics/cross_disease_specificity.png`** — FAIL (P1): rendered at 160 DPI; needs re-export at 300+ DPI for print.
2. **`figures/diagnostics/hannigan_inclusion_sensitivity.png`** — FAIL (P1): rendered at 150 DPI; needs re-export at 300+ DPI for print.
3. **`figures/diagnostics/subgroup_auc.png`** — PASS but ~95% near-white pixels; consider tighter layout or smaller margins to use canvas more efficiently.
4. **`figures/diagnostics/decision_curves.png`** — PASS but ~95% near-white; same suggestion.
5. **`manuscript/figures/Figure2_ROC_Curves.png`** — PASS but ~95% near-white; ROC curves are inherently sparse, but worth confirming line/marker thickness reads at print size.

## Other observations

- **Duplicate file**: `figures/figure5_three_panel_shap.png` is byte-identical to `manuscript/figures/Figure4_Three_Panel_SHAP.png` (both 394,910 B, 5496x1973 px). The lowercase `figure5_*` file appears redundant.
- **`Figure4_Three_Panel_SHAP.pdf`** has a MediaBox of 18.33 x 6.57 in — wider than a standard journal page. Confirm it fits a two-column figure slot, or be prepared to resize.
- **Text-legibility heuristic caveats**: the `min` glyph height of 4 px in many figures is driven by axis tick marks and decimal dots, not body text. Figures where median glyph height is also <16 (notably `fig4_external_validation.png` and `depth_distribution.png`) deserve a manual zoom-in to verify axis-label legibility at print size.
- No PNG approached the 5 MB ceiling; largest is `pcoa_bray_curtis.png` at 0.89 MB.
- No empty/blank or sub-5 KB files detected. All P0 checks passed.

## Recommended actions (read-only audit; no edits performed)

1. Re-render `cross_disease_specificity.png` and `hannigan_inclusion_sensitivity.png` at 300 DPI (priority for any printed supplement).
2. Decide whether to keep both `figure5_three_panel_shap.png` and `Figure4_Three_Panel_SHAP.png`, or drop one to avoid drift.
3. Spot-check axis label legibility on `fig4_external_validation.png` and `depth_distribution.png` at final print width.
4. Verify the wide aspect of `Figure4_Three_Panel_SHAP.pdf` is compatible with the target journal template.
