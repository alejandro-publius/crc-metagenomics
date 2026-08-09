# AlphaFold scope decision for the colibactin guide audit

**Decision:** do not run AlphaFold for the two frozen guide exceptions.

## Why

AlphaFold predicts biomolecular structure and interactions. The present
exceptions ask a different question: whether a 20-base DNA spacer plus its PAM
is represented correctly in two draft bacterial assemblies. Source-read
reconciliation directly answers that question.

- UPEC79 source reads recover the exact primary spacer/PAM, resolving the
  assembly absence without implying a protein change.
- JML024 source reads support the long target-bearing neighborhood and provide
  only one-read support for each short-contig probe. The frozen depth rule does
  not support two biological copies, so the duplicate remains unresolved.
- Neither outcome identifies a protein-changing variant whose structural
  consequence would affect the present guide-ranking decision.

An AlphaFold prediction therefore cannot adjudicate either case and would add
visual complexity without new evidence. In particular, it cannot establish
CRISPRi binding, transcriptional knockdown, colibactin production, bacterial
delivery, off-target activity, or safety.

## Reopening rule

Structural analysis may be reconsidered only if a later, versioned sequence
analysis identifies a reproducible amino-acid-changing variant in a defined
ClbB or ClbC domain and poses a domain-specific structural hypothesis. That
would be a separate protein-function analysis, not validation of these guides.

## Method references

- AlphaFold overview: https://deepmind.google/science/alphafold/
- AlphaFold 3 methods paper: https://www.nature.com/articles/s41586-024-07487-w
- AlphaFold Server output terms: https://alphafoldserver.com/output-terms

This is a scientific scope decision, not a claim of collaboration with Google
DeepMind or access to a private AlphaFold program.
