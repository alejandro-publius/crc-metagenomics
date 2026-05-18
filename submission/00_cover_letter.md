# Cover Letter

[Editor Name]
[Journal Name]
[Date]

Dear Editor,

We are pleased to submit our manuscript, *"Cross-cohort gut microbiome
classification of colorectal cancer: a 10-cohort meta-analysis with
country-aware leave-one-dataset-out validation"*, for consideration at
[Journal Name].

**Significance.** Colorectal cancer (CRC) is the third most common cancer
worldwide, and a non-invasive, microbiome-based screening assay has been
proposed by multiple groups as a complement to colonoscopy. However,
published cross-cohort estimates of microbiome classifier performance
remain inconsistent because (i) most studies pool 5-7 cohorts and report
single-cohort holdouts without geographic confound control, and (ii)
many studies have advocated pathway-level features without rigorous
per-fold filtering, allowing test-fold leakage to inflate estimated
generalization. Our study addresses both gaps.

**Key contributions.**

1. **A 10-cohort meta-analysis** of 1,522 shotgun-metagenomic stool
   samples (674 CRC, 665 controls, 183 adenomas) from
   `curatedMetagenomicData`, expanding the most widely cited prior
   reference (Thomas et al. 2019 *Nature Medicine*, 5 cohorts) by adding
   YachidaS_2019, WirbelJ_2018, and GuptaA_2019. HanniganGD_2017 was
   pre-specified as excluded based on sequencing-depth and feature-sparsity
   criteria assessed independently of classification outcomes.

2. **Country-aware leave-one-dataset-out (LODO) cross-validation.** When
   the held-out cohort shares a country with one or more training cohorts,
   the same-country cohorts are also excluded from training. Without this
   correction, ThomasAM_2019_c (Japan) achieves AUC = 0.998 — clearly
   driven by population-level confounding with YachidaS_2019 (Japan) — and
   collapses to a biologically plausible 0.836 once geographic leakage is
   removed.

3. **A statistically rigorous negative result on pathways.** Adding 402-406
   per-fold-filtered HUMAnN pathway features to the species-only Random
   Forest does not improve mean per-cohort AUC (species 0.807 vs joint
   0.804) and *significantly degrades* sample-level discrimination on
   pooled predictions (DeLong z = 3.35, p = 0.0008, n = 1,339). Per-fold
   pathway filtering is essential: pre-filtering on all samples leaks
   test-fold information and produces spuriously favorable results.

4. **Adenoma stage analysis.** Healthy-vs-adenoma classification is near
   chance across cohorts (AUC 0.561, four cohorts), whereas
   adenoma-vs-CRC reaches moderate performance (AUC 0.671). SHAP
   importances place oral-associated taxa
   (*Fusobacterium nucleatum*, *Peptostreptococcus stomatis*,
   *Parvimonas micra*, *Gemella morbillorum*) at the top of the
   adenoma-vs-CRC ranking but not the healthy-vs-adenoma ranking,
   consistent with the proposed model in which the oral-bacterial CRC
   signature emerges during malignant transformation rather than at the
   adenoma stage.

5. **Comprehensive sensitivity analyses** including (i) per-fold filter
   threshold sweep across a 4 × 5 grid showing AUC spread of 0.018,
   (ii) seed sensitivity across five random seeds (mean 0.810 ± 0.002),
   (iii) age/sex/BMI confounder adjustment via direct inclusion and
   residualization (range 0.800-0.814 around an unadjusted baseline of
   0.807), and (iv) ComBat batch correction (AUC 0.815 corrected vs
   0.807 uncorrected).

All code, predictions, and supplementary tables are publicly available
at <https://github.com/alejandro-publius/crc-metagenomics>, and the
analysis can be reproduced end-to-end via the documented `REPRODUCING.md`
pipeline from the public `curatedMetagenomicData` Bioconductor package.

This work is not under consideration elsewhere and all authors have
approved the submission. We have no conflicts of interest to declare and
suggest the following reviewers with expertise in this area:

- [Reviewer 1 name and affiliation]
- [Reviewer 2 name and affiliation]
- [Reviewer 3 name and affiliation]

Thank you for considering our work.

Sincerely,

Alejandro Velazquez
University of California, Berkeley (Computer Science)
alejandro-publius@berkeley.edu

On behalf of all co-authors.
