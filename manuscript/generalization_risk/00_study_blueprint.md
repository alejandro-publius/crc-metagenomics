# Study blueprint: predicting and explaining cross-study failure

## Plain-language research question

If a colorectal-cancer microbiome model moves to a new hospital or population,
which kinds of evidence still work, which fail, and can we recognize danger
before the new site supplies outcome labels?

## Contribution

This is a portability and failure-analysis paper, not another biomarker search.
It combines six evidence levels under one country-aware leave-one-dataset-out
design:

1. species abundance as the parsimonious reference;
2. community pathways and species-resolved pathways;
3. leakage-safe gene-family screening;
4. a checksum-frozen, experimentally motivated mechanism panel;
5. source-only species-aware correction, separated from target-adaptive use;
6. label-free estimates of target-cohort performance.

The novelty is the prospective structure: the held-out study cannot select its
features, mechanisms, correction offsets, or risk model. A balanced 200-sample
external shotgun cohort is frozen before model scoring.

## What the completed internal work says

- Species RF: mean per-cohort AUC 0.807; pooled AUC 0.781 (95% CI
  0.757–0.805).
- Joint pathways: no improvement; pooled AUC 0.756 for RF and 0.766 for XGB.
- Gene-family elastic net: mean AUC 0.693, range 0.570–0.812.
- Frozen mechanisms: mean AUC 0.569; mechanism plus parent species 0.655,
  parent species alone 0.656.
- Source-only species-aware correction: corrected functions 0.773 versus
  0.771 uncorrected; target-adaptive analysis 0.777 and reported separately.
- Label-free risk estimator: MAE 0.094 versus 0.062 for the historical-model
  mean; it does not yet predict failure usefully.

Together these results support a clear internal conclusion: added biological
specificity and shift correction do not automatically create portability, and
apparently intuitive unlabeled warning signals can themselves overfit a small
number of deployment environments.

## What would falsify or strengthen the conclusion

The frozen PRJNA763023 cohort contains 200 public WGS runs: 50 older-onset CRC,
50 younger-onset CRC, and 50 matched controls for each group. It is untouched
by model development. The paper becomes substantially stronger if the external
species model remains useful while functions/mechanisms remain unstable. A
large external performance drop is also publishable if reported as a genuine
portability failure. No conclusion should depend on choosing the favorable
outcome after profiling.

## Journal ladder

1. **mSystems** after full external profiling and a systems-level explanation
   of which representations transfer. Its scope explicitly includes human
   microbiome, computational microbiology, and multidimensional integration.
2. **GigaScience** if the strongest contribution is the fully reproducible,
   reusable benchmark and external workflow; reproducibility, utility, and
   FAIR research objects are explicit publication criteria.
3. **Bioinformatics Advances** only if the risk-estimation method becomes a
   convincing methodological advance with real external validation. Its
   guidance rejects straightforward applications of established methods.

The current internal-only package is not ready for the top target. External
profiles, frozen evaluation, consolidated figures, and coauthor review are the
submission gate.

## Sources for journal fit

- https://journals.asm.org/journal/msystems/scope
- https://academic.oup.com/gigascience/pages/instructions_to_authors
- https://academic.oup.com/bioinformaticsadvances/pages/author-guidelines
