# Resolution protocol for two primary-guide assembly exceptions

**Freeze:** 2026-08-08, after the 97-human-isolate result and before inspecting
contig overlap, source-read sequence support, or later assemblies for JML024 and
UPEC79.

## Scope

Only the two prespecified `sgclbB_4387` exceptions enter follow-up:

- JML024 (`BFMV01000000`, source run `DRR102722`): two exact PAM-compatible
  sites in the published draft assembly.
- UPEC79 (`BGJT01000000`, source run `DRR103319`): no exact PAM-compatible
  site in the published draft assembly.

The perfectly conserved secondary guide is retained as a within-isolate
sequence-recovery control. No other isolate will be added after results are
viewed.

## Frozen evidence order

1. **Draft-contig structure.** Compare the complete target-bearing contigs and
   at least 500 bases on each available side of the primary-guide site.
2. **Source reads.** Retrieve the accessioned paired-end reads, retain provider
   checksums, and count the exact 20-nt spacer plus correctly oriented NGG PAM
   on either read orientation. Count the secondary guide the same way as a
   positive run-level control.
3. **Local sequence neighborhood.** Search the source reads and assembly for
   PAM-compatible primary-guide variants with up to four spacer mismatches.
4. **Independent public sequence.** Search a later or independently assembled
   genome only when it maps to the same BioSample or source-read accession.

## Frozen interpretation rules

### JML024 duplicate

- `resolved_redundant_draft_contig` if the two target-bearing contigs overlap
  the same locus at at least 99.9% nucleotide identity across at least 500 bp,
  or if one target-bearing contig is contained in the other across that span.
- `supported_distinct_copies` only if the target neighborhoods are genuinely
  different and source-read depth supports approximately two copies relative
  to multiple single-copy pks controls.
- Otherwise `unresolved_duplicate`.

### UPEC79 absence

- `resolved_assembly_omission` if at least three independent source reads
  contain the exact primary spacer/PAM while the assembly does not.
- `supported_sequence_difference` if the source run recovers the secondary
  control and a PAM-compatible primary-site variant, but not the exact primary
  site.
- `source_reads_do_not_support_site` if the secondary control has at least ten
  exact supporting reads while neither an exact primary site nor a qualifying
  PAM-compatible variant is observed.
- Otherwise `unresolved_absence`.

## Claim boundary

Raw-read and assembly reconciliation can resolve sequence representation and
possible contig redundancy. It cannot establish guide expression, CRISPRi
knockdown, colibactin production, delivery, biological specificity, or safety.
AlphaFold predictions will not be used to overrule nucleotide evidence.
