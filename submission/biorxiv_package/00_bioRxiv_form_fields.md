# bioRxiv Submission Form — Paste-Ready Field Values

Every field below is copy-pasteable into the corresponding bioRxiv portal
input. Fields are listed in the order they appear during submission.

---

## 1. Article title

```
Species-level taxonomic features outperform joint species-plus-pathway models for colorectal cancer detection: a rigorous re-evaluation across ten cohorts
```

(174 characters; within bioRxiv's 250-character title limit.)

## 2. Type of article

```
Research Article
```

## 3. Subject area

- **Primary:** Bioinformatics
- **Secondary:** Microbiology
- **Tertiary:** Cancer Biology

## 4. Abstract (paste field, 250-word cap)

bioRxiv's structured abstract field is capped at 250 words. The
manuscript's full abstract is ~330 prose words after stripping inline
`results/*.csv` file paths, so the version below has been trimmed to
**249 words** while preserving every headline statistic (pooled AUCs,
DeLong p-values, CIs, robustness ranges, adenoma results). Paste this:

```
Background. Shotgun gut metagenomic classifiers can discriminate colorectal cancer (CRC) cases from controls, but the incremental value of metabolic pathway features beyond species-level taxonomic profiles has not been rigorously tested, and the robustness of cross-cohort classifiers to analytical choices is rarely evaluated systematically.

Methods. We assembled 1,522 stool metagenomes from ten publicly available CRC case-control cohorts (674 CRC, 665 controls, 183 adenomas) via curatedMetagenomicData; HanniganGD_2017 was excluded a priori for low sequencing depth. MetaPhlAn species (229 features) and unstratified HUMAnN pathway abundances (402-406 features per fold) were compared under country-aware leave-one-dataset-out (LODO) cross-validation. Three classifiers were tested: species-only Random Forest (RF), joint species-plus-pathway RF, and joint XGBoost. Discrimination was compared with the DeLong test on pooled held-out predictions, complemented by per-cohort paired tests; 95% CIs came from 10,000-iteration cohort-stratified bootstrap.

Results. Species-only RF reached a per-cohort mean LODO AUC of 0.807 +/- 0.065 and a pooled AUC of 0.781 (95% CI 0.757-0.805), significantly outperforming joint RF (0.756, 0.731-0.781; DeLong z = 3.35, p = 0.0008) and joint XGBoost (0.766, 0.740-0.791; z = 2.00, p = 0.046). Results were stable across five seeds (0.807-0.811), a 20-cell pathway-threshold grid (0.794-0.812), and demographic adjustment (0.800-0.814). Cross-cohort adenoma LODO (n = 183) gave near-chance healthy-vs-adenoma AUCs (RF 0.561, XGB 0.579) and moderate adenoma-vs-CRC AUCs (RF 0.671, XGB 0.617).

Conclusions. At current cross-cohort sample sizes, species-level taxonomic features alone provide superior CRC classification compared to joint species-plus-pathway models; adding pathways increases dimensionality without proportional signal gain.
```

Word count: **249 words** (under the 250-word cap).

If bioRxiv's portal happens to accept >250 words for your account
(some users see a 300-word soft limit), the full manuscript abstract is
available verbatim in `manuscript/markdown/01_abstract.md`.

## 5. Corresponding author block

```
Alejandro Velazquez
University of California, Berkeley
Department of Computer Science
Berkeley, CA, USA
Email: alejandro-publius@berkeley.edu
ORCID: 0009-0007-9798-1958
```

## 6. Authors and affiliations

See `01_authors_and_affiliations.txt` (one row per author, paste in full).

Quick summary:

1. Alejandro Velazquez (corresponding) — University of California, Berkeley
   (Computer Science). alejandro-publius@berkeley.edu.
   ORCID 0009-0007-9798-1958.
2. Rachel Selbrede — California State University San Marcos
   (Molecular and Cell Biology). ORCID 0009-0006-7046-3192.

## 7. Funding

```
Self-funded undergraduate research. No external funding was received for this work; computational resources were provided by the authors' personal workstations.
```

## 8. License

```
CC-BY 4.0
```

(Recommended; permits broadest reuse with attribution.)

## 9. Conflict of interest / Competing interests

```
None. The authors declare no competing financial or non-financial interests.
```

(See `02_competing_interests.txt` for the single-line version.)

## 10. Data and code availability

See `03_data_and_code_availability.txt` (paste verbatim).

## 11. Keywords

See `04_keywords.txt` (comma-separated).

## 12. Suggested reviewers (bioRxiv asks for 3)

bioRxiv does NOT require named reviewers, but the form provides optional
"Suggested Reviewers" slots. Pick three from the candidate pool below
(all are senior figures in CRC microbiome / metagenomic meta-analysis
who have not co-authored with the submitting authors). For each, paste
Name, Affiliation, Email.

### Candidate suggestions (pick any 3)

1. **Nicola Segata** — University of Trento (Italy). Lead of
   curatedMetagenomicData / MetaPhlAn / HUMAnN. Direct methodological
   relevance; co-led the Thomas et al. 2019 CRC meta-analysis.
   Email: nicola.segata@unitn.it
2. **Jakob Wirbel** — Stanford University (formerly EMBL Heidelberg).
   First author of Wirbel et al. 2019 *Nature Medicine* CRC meta-analysis
   that established the cross-cohort microbiome CRC signal we extend.
   Email: jakob.wirbel@embl.de
3. **Georg Zeller** — Leiden University Medical Center (formerly EMBL).
   Senior author on Zeller 2014, Wirbel 2019; long-standing leader on
   gut metagenomic CRC biomarkers.
   Email: g.zeller@lumc.nl
4. **Curtis Huttenhower** — Harvard T.H. Chan School of Public Health.
   Senior author on HUMAnN, MetaPhlAn, curatedMetagenomicData; expert on
   pathway-level functional profiling (directly relevant to our negative
   result).
   Email: chuttenh@hsph.harvard.edu
5. **Shinichi Yachida** — Osaka University. Senior author on Yachida
   2019 *Nature Medicine* CRC/adenoma cohort (one of our 10 cohorts);
   adenoma-stage expertise.
   Email: syachida@cgi.med.osaka-u.ac.jp

Replace the three you pick into the form. Anyone you remove can be
re-used later for a journal submission.

## 13. Related preprint history

```
None. This is the first version (v1).
```

## 14. Manuscript file to upload

Main manuscript: PDF generated from
`manuscript/CRC_Manuscript_Complete.docx`
(or run `python3 scripts/build_biorxiv_pdf.py`).

## 15. Figures to upload (separate files)

- `manuscript/figures/Figure1_Forest_Plot.{pdf,png}`
- `manuscript/figures/Figure2_ROC_Curves.{pdf,png}`
- `manuscript/figures/Figure3_SHAP_Importance.{pdf,png}`
- `manuscript/figures/Figure4_Three_Panel_SHAP.{pdf,png}`

Upload PDFs (vector) where available; PNGs (>=300 DPI) as fallback.

## 16. Supplementary materials

Bundle as a single ZIP:
- All of `results/supplementary/*.csv` (S1-S10).
- `manuscript/Supplementary_Tables.docx`.
- `manuscript/markdown/07_supplementary.md` (optional, as plain-text
  reference).
