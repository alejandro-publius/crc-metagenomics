# Intervention-readiness extension: current results

## What is complete

- Four known CRC mechanisms were retained as a frozen benchmark.
- A country-aware, cross-fitted screen evaluated 6,755 UniRef90 gene families.
- Sixteen gene families met the frozen internal nomination rules.
- UniProt/UniParc provenance was resolved for every nomination.
- A provisional atlas separates known **effector targets** from possible
  strain-selective **precision addresses**.
- One explicit-gate evidence dossier was generated for every atlas entry.
- Parent-species adjustment retained four of sixteen nominations.
- A memory-safe, taxon-resolved export recovered 26,246 carrier observations
  across all ten cohorts.
- All four parent-adjustment survivors failed the prespecified 80% dominant-
  carrier address gate; none had a majority carrier even at 50%.
- A structured review of six primary studies separates causal evidence from
  genetic and delivered-edit evidence for the four known benchmarks.
- Colibactin is the only benchmark with reported in vivo delivered CRISPRi that
  changes a CRC-relevant phenotype, but that report is a preprint and the
  benchmark remains incomplete for comprehensive conservation and specificity.
- The two published colibactin spacers each retained exactly one PAM-compatible
  target site in all seven genomes of a literature-defined pks-positive pilot
  panel.
- An expanded panel was frozen before scanning by retaining all 97 human-host
  entries in a published 109-strain pks-positive population table. Primary
  `sgclbB_4387` covered 96/97 and was unique in 95/97; secondary
  `sgclbC_2313` covered and was unique in 97/97. Both passed the predeclared
  overall and fecal-versus-clinical subgroup conservation gates.
- In a frozen panel of ten common gut bacterial references plus GRCh38.p14,
  primary guide `sgclbB_4387` had no flagged near matches. Secondary guide
  `sgclbC_2313` had five GRCh38 sites with three or four mismatches and an exact
  PAM-proximal seed, so it did not pass the protected-reference pilot.
- The two primary-guide assembly exceptions were reconciled against 2,627,284
  checksum-validated source-read records. UPEC79 is a resolved draft-assembly
  omission: 30 reads recover the exact primary target while its assembly has
  none. JML024 remains an explicit unresolved duplicate: the long contig is
  well supported, every selected short-contig probe appears in only one read,
  and combined median probe depth is 0.757 times—not approximately twice—the
  single-copy control. No later or independent assembly for either source
  sample was identified in the provider mapping.
- Frozen assay coverage is partial for colibactin (13/19 prespecified genes) and
  bile-acid conversion (8/10 genes), complete only for the single `bft`
  effector, and absent for `fadA`/`fap2`.

## Current scientific signal

The known mechanisms recur directionally but are weak abundance classifiers:
colibactin is CRC-enriched in 9 of 10 evaluable cohorts with median association
AUC 0.541, while no benchmark passes the frozen cross-population gate. This is
not used to discard the mechanisms or infer that they are biologically
unimportant.

The discovery track produced 16 internally cross-fitted nominations. Four
initially added signal beyond their archived parent-species proxy, but all four
were distributed across 7–23 taxonomic carriers. Their dominant carriers
accounted for only 13.5–42.5% of stratified abundance, and the archived parent
species contributed 0–19.6%. They are therefore rejected as direct editing
addresses, not promoted to guide design.

Ten of the sixteen nominations are uncharacterized. The annotated remainder
include ordinary enzymes, regulators, recombinases, and structural machinery.
The taxon-resolved result shows why representative database taxonomy cannot be
treated as the organism carrying a gene family in a metagenome.

## What remains before a publishable intervention claim

1. Obtain direct biological review of the mechanism-integrity summaries,
   mixed-carrier interpretation, and stopping rule.
2. Extend the completed colibactin pilots beyond the single-country 97-isolate
   panel and into platform-specific assessment. Carry `sgclbB_4387` as the
   cleaner current lead while retaining JML024 as unresolved, redesign rather
   than quietly reuse the flagged `sgclbC_2313` backup, and do not revive
   rejected de novo addresses.
3. Decide whether an untouched gene-level cohort is needed to validate the
   methodological finding. No discovery candidate advances to guide design
   under the current sequential gates.

`readiness_atlas.csv` is a transparent rejection/readiness atlas, not an
experiment-ready or treatment-ready list.
