# Frozen study protocol: from CRC biomarkers to editable mechanisms

**Protocol status:** Stage 1 frozen before de novo candidate discovery
**Date:** 2026-08-08

## One-sentence research question

Which colorectal-cancer-linked microbial mechanisms recur across human
populations and are causal, conserved, specific, and technically plausible
enough to justify precision microbiome-editing experiments?

The atlas recognizes two distinct experimental roles. **Effector targets** are
harmful microbial functions to silence. **Precision addresses** are genomic
sequences that may distinguish a harmful strain or clade from organisms that
should remain untouched. An address need not itself cause disease, but it must
be linked to a credible harmful clade and pass stricter conservation and
specificity gates before any guide-design work.

## Primary contribution

The study will create an open, auditable framework and atlas for deciding
whether a CRC microbiome finding is merely a biomarker or a credible
intervention target. It will not claim that association alone establishes a
treatment target.

## Two analysis tracks

### Track A: frozen known-target benchmark

Four literature-motivated mechanisms were frozen before outcome modeling:
colibactin genotoxicity, *Bacteroides fragilis* toxin, fusobacterial adhesion,
and secondary bile-acid conversion. They benchmark whether prominent proposed
mechanisms recur consistently across ten cohorts. They cannot be replaced
after viewing their results.

### Track B: leakage-safe candidate discovery

The de novo track will search gene families for candidates that are repeatedly
CRC-enriched across training populations. Discovery is repeated inside each
country-aware outer fold. A held-out cohort and any cohort from the same
country cannot select, rank, annotate, or tune candidates evaluated in that
fold. Candidates emerging from cross-fitted internal evidence remain
*nominations* until confirmed in an untouched gene-level external dataset.

The internal discovery rules are frozen as follows. Within each outer fold, a
gene family must be evaluable in at least three training cohorts, be
CRC-enriched in at least 70% of them, and have a median training-cohort AUC of
at least 0.55. At most 100 candidates are carried into the held-out cohort,
ranked by median AUC, consistency, cohort coverage, prevalence difference, and
stable gene identifier. An internal nomination must be selected and evaluable
in at least seven outer folds, remain CRC-enriched in at least 70% of those
held-out folds, and retain median held-out AUC of at least 0.55. These are
screening rules, not statistical proof or external confirmation.

For precision-address candidates, a parent-species adjustment gate is also
frozen before running the comparison. Archived representative organism names
are mapped only by exact genus-species name to the existing MetaPhlAn feature
table; no synonyms may be added after viewing results. A candidate must be
evaluable in at least seven outer folds, improve held-out AUC beyond the matched
parent-species model in at least 70% of those folds, and have median held-out
AUC improvement of at least 0.02. Passing this gate indicates possible
strain-level information, not causality or a validated genomic address.

## Evidence gates

No weighted score may hide a failed safety or evidence requirement. Each
candidate receives a separate result for these gates:

1. **Human recurrence:** observed in enough cohorts, CRC-enriched in at least
   70% of evaluable cohorts, and median within-cohort association AUC at least
   0.55. At least five cohorts must be evaluable.
2. **Mechanism integrity:** the assay must support the required functional
   machinery rather than a single ambiguous marker. Exact required-gene rules
   will be frozen after structured biological review and before this gate is
   run.
3. **Causal evidence:** structured evidence review distinguishes association,
   host-cell evidence, animal perturbation, and human mechanistic evidence.
4. **Sequence conservation:** a plausible target region must recur across the
   relevant harmful strains. Genome sources and numerical thresholds will be
   frozen before sequence analysis.
5. **Specificity:** the target must be distinguishable from protected gut
   organisms and human sequences. This computational screen does not establish
   biological safety.
6. **Editability:** delivery range and prior perturbation evidence are recorded
   independently of disease association.
7. **External confirmation:** a locked external cohort is evaluated once at
   gene level. Existing species-only external scores cannot satisfy this gate.

## Constructive outputs regardless of candidate outcome

The primary outputs do not depend on forcing a favorable biological result:

- a versioned target registry with evidence provenance;
- a reusable cross-cohort target-evaluation library;
- a known-target benchmark;
- a leakage-safe discovery and validation benchmark;
- a public intervention-readiness atlas with explicit unknowns and failed
  gates;
- one evidence dossier per leading candidate;
- a laboratory handoff specification describing the experiment that would
  resolve each remaining uncertainty.

The intended secondary result is a validated shortlist of experimental
candidates. The data are allowed to show that a famous mechanism is not yet
ready; it may not be silently redefined or removed.

## Claim boundaries

- Stool metagenomic association does not establish causation.
- A conserved sequence is not automatically a safe CRISPR target.
- The atlas prioritizes research experiments; it does not recommend treatment.
- No private IGI data, implied institutional endorsement, or laboratory access
  is part of this study.
