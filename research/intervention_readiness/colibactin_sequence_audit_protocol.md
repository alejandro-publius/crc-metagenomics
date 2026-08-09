# Published-colibactin-guide sequence audit

**Freeze:** 2026-08-08, after confirming the expected on-target site in the
NC101 experimental reference and before inspecting results in the other six
genomes.

## Question

Do the two already-published, in-vivo-tested colibactin CRISPRi spacers retain
an exact PAM-compatible target site across a small, literature-defined panel of
seven pks-positive *E. coli* genomes?

This is a positive-control audit. It is not de novo guide design, a clinical
recommendation, or a comprehensive survey of human pks-positive diversity.

## Frozen inputs

- Primary benchmark: `sgclbB_4387`; secondary benchmark: `sgclbC_2313`.
- Spacer orientation is transcribed from the capitalized 20-nt sequence in
  Supplementary Table S4. The table's detection primers provide the reverse
  complements and are retained as an orientation check.
- Genome panel: NC101 plus six independent pks-positive references or isolates
  named by Mannion et al. (2016). MIT A4 is excluded because that report says it
  lacks the complete pks island.
- Public sequence accessions and their inclusion reasons are frozen in
  `colibactin_reference_genomes.csv`.

## Frozen rules

1. A target site requires an exact 20-nt spacer match in either orientation
   with the corresponding SpCas9 `NGG` PAM.
2. A genome is covered only when at least one exact PAM-compatible site is
   present.
3. A site is unique within a genome only when exactly one such site is found.
4. The pilot conservation gate is at least 80% genome coverage and at least 80%
   unique-site coverage for the seven-genome panel.
5. Results are retained regardless of direction.

## Claim boundary

Passing this pilot would show only that a published guide is compatible with a
small reference panel. It cannot establish coverage in patients, lack of
near-match effects, safety for beneficial bacteria or human DNA, delivery, or
therapeutic readiness. Those require a larger human-isolate panel and separate
protected-reference specificity analyses.
