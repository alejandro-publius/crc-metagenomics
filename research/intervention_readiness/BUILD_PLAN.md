# Complete build plan: CRC microbial intervention-readiness atlas

## Definition of done

The project is complete when a reader can reproduce a locked path from public
human metagenomes to a short, evidence-graded list of microbial mechanisms that
are ready—or not ready—for laboratory evaluation, without outcome leakage or
unsupported therapeutic claims.

## Work packages

| ID | Work package | Concrete output | Completion gate |
|---|---|---|---|
| 1 | Study freeze | Protocol, success rules, claim boundaries | Coauthors approve before discovery outcomes are viewed |
| 2 | Known-target benchmark | Four-target registry and ten-cohort association table | Every target and non-assayed cohort retained |
| 3 | Discovery engine | Country-aware, cross-fitted gene-family nomination code | Synthetic leakage tests and deterministic rerun pass |
| 4 | Biological annotation | Versioned mapping from nominated gene families to mechanism and organism | Every mapping has database version and evidence URL |
| 5 | Mechanism integrity | Required-gene/operon evidence per target | Ambiguous single-gene signals cannot pass |
| 6 | Conservation | Public-genome target coverage and sequence variability tables | Genome set and thresholds frozen before results |
| 7 | Specificity | Protected-commensal and human-reference similarity screen | Hits and exclusions are fully auditable |
| 8 | Editability evidence | Structured delivery and perturbation literature table | Association evidence cannot substitute for editability |
| 9 | External confirmation | Locked gene-level external-cohort results | One-time scoring retained regardless of direction |
| 10 | Atlas and dossiers | Readiness matrix plus leading-candidate evidence packets | Unknown and failed gates remain visible |
| 11 | Manuscript | Figures, methods, limitations, supplement, lay summary | Every numerical claim traced to a generated artifact |
| 12 | Release | Tests, workflow, DOI archive, coauthor signoff | Clean rerun and submission checklist pass |

## Current status

- Existing and reusable: ten internal cohorts, country-aware holdout framework,
  6,755-gene sparse matrix, frozen mechanism panel, raw-read pilot, external
  species validation, packaging, tests, and release workflow.
- Work package 1: protocol and thresholds are written. Alejandro reported
  Rachel's approval to continue the direction; direct final signoff remains
  required before submission.
- Work package 2: computationally complete. All four frozen benchmarks are
  retained, including the mechanism the current assay cannot measure.
- Work package 3: internally complete. The locked cross-fitted screen evaluated
  6,755 families and produced 16 internal nominations.
- Work package 4: complete. Automated UniProt/UniParc provenance and the
  ten-cohort taxon-resolved export are complete. All four parent-adjustment
  survivors failed the frozen taxonomic-address gate.
- Work packages 5–7: not triggered for de novo candidates under the sequential
  stopping rule. No mixed-carrier family advances to conservation, guide
  specificity, or editing design. Required-gene integrity for known benchmarks
  remains pending.
- Work package 8: structured causality and editability review of known
  benchmarks remains pending.
- Work package 9: no de novo candidate qualifies for external confirmation;
  whether to validate the methodological failure in an untouched cohort is a
  manuscript-design decision.
- Work package 10: complete for the current evidence. The atlas records four
  mixed-source rejections, nine parent-adjustment rejections, and three
  unresolved parent mappings, plus four known benchmarks.
- Work package 11: working manuscript, lay summary, and first main figure are
  drafted. Coauthor biological review and revision remain.
- Work package 12: pending final tests, clean rerun, direct coauthor signoff,
  journal formatting, and archival release.
