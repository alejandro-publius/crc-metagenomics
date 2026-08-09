# Human-isolate colibactin-guide conservation audit

**Freeze:** 2026-08-08, after the seven-genome reference pilot and protected-
reference specificity pilot, but before downloading or searching any genome in
the expanded human-isolate panel.

## Question

Do the two published colibactin CRISPRi spacers retain an exact, PAM-compatible
target site across a larger literature-defined collection of human-derived
*Escherichia coli* reported to carry the pks island?

## Frozen population panel

- Source: Watanabe et al., *Insights into the acquisition of the pks island and
  production of colibactin in the Escherichia coli population*, *Microbial
  Genomics* (2021), doi:10.1099/mgen.0.000579.
- The source paper reports 109 pks-positive strains in Supplementary Table S1.
  The panel retains every row whose `Host` is `human`: 97 isolates, comprising
  62 fecal commensals and 35 extraintestinal clinical isolates from blood or
  urine. No isolate was selected using guide-match results.
- The table is available from Europe PMC at
  `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8209727/supplementaryFiles`.
  The extracted `mgen-7-0579-s002.xlsx` used for the freeze has SHA-256
  `28c154eeb6e84b389d8ae0279a1b003635f109403905f06cb201d5c201275928`.
- Exact source rows, strain metadata, WGS accessions, and DDBJ retrieval URLs
  are frozen in `colibactin_human_isolate_panel.csv`.

The source paper notes that pks carriage does not guarantee colibactin
production and reports disrupted or inactive islands in a minority of the full
collection. This audit asks only whether the published spacer/PAM sequence is
present; it does not relabel a strain as functionally colibactin-producing.

## Frozen sequence and success rules

1. The spacer sequences and orientations remain those transcribed before the
   seven-genome pilot in `published_colibactin_guides.csv`.
2. A genome is covered only when an exact 20-nt spacer match occurs on either
   strand beside the corresponding SpCas9 `NGG` PAM.
3. A site is unique only when exactly one such match occurs in the assembly.
4. A guide passes this expanded audit only if all three conditions hold:
   - at least 90% exact-site coverage across all 97 isolates;
   - at least 90% unique-site coverage across all 97 isolates; and
   - at least 80% exact-site coverage in both the fecal-commensal and
     extraintestinal-clinical source groups.
5. Every missing, duplicated, or subgroup-specific result is retained.

## Claim boundary

This is a single published collection from Japan and is dominated by
phylogroup B2. Its short-read assemblies can contain gaps. Passing would support
sequence conservation in this panel only; it would not establish global
patient coverage, expression, knockdown efficacy, delivery, safety, or
therapeutic readiness. Failing may reflect true sequence variation, disrupted
pks islands, or assembly incompleteness and must be reported without dismissal.
