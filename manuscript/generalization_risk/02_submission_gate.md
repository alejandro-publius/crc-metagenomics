# Submission gate

## Required before submission

- [x] Country-aware leakage-safe internal splits.
- [x] Species, pathway, gene-family, mechanism, and correction comparisons.
- [x] Label-free risk model evaluated by leaving an entire cohort out.
- [x] External project, samples, labels, and scoring rule frozen before AUC.
- [x] Executable external MetaPhlAn harmonization and RF scorer.
- [x] Profile all 200 external WGS samples with one documented database build.
- [x] Run the frozen external scorer once and retain the result regardless of
      direction.
- [x] Bootstrap external AUC/AUPRC confidence intervals and report age-stratum
      estimates as secondary analyses.
- [x] Create one portability figure spanning all representations and one
      predicted-versus-observed risk figure.
- [x] Update the abstract and discussion with the external result.
- [ ] Obtain both authors' approval of wording, author order, affiliations,
      funding, conflicts, and AI-assistance disclosure.
- [ ] Archive the exact environment and derived tables with a persistent DOI.

## Claims that are not currently allowed

- The model is clinically ready or can replace FIT.
- The external AUC establishes clinical readiness or prospective screening utility.
- Mechanism genes are absent when a bounded read pilot does not detect them.
- Target-adaptive correction is equivalent to source-only deployment.
- 120 model-by-cohort rows are 120 independent environments; there are ten.

## Stop rule

No model, feature, threshold, database, or external sample may be changed after
external outcome scoring because its AUC is disappointing. Any unavoidable
pipeline change must be versioned, justified without reference to performance,
and followed by reporting both the original and revised result.
