# Published-colibactin-guide protected-reference specificity pilot

**Freeze:** 2026-08-08, after the seven-genome exact-site conservation result
and before scanning any protected reference.

## Question

Do the two published, in-vivo-tested colibactin CRISPRi spacers have concerning
PAM-compatible near matches in a frozen pilot panel of ten common cultured gut
bacterial references or the GRCh38.p14 human reference?

This is a conservative computational triage screen. It is not a safety study,
does not model delivery exposure, and is not a clinical recommendation.

## Frozen references

- Ten versioned cultured gut reference assemblies were selected without using
  guide-match results. They cover common control-cohort taxa and major gut
  lineages, including legacy-name mappings present in the metagenomic profiles.
- Human reference: GRCh38.p14 (`GCF_000001405.40`).
- Exact assemblies, strains, URLs, and inclusion reasons are frozen in
  `protected_reference_panel.csv`.
- Their legacy-name mappings and healthy-control prevalence ranks are generated
  in `results/intervention_readiness/protected_reference_selection.csv`; nine
  rank within the top 21 profiled species and the broader Roseburia reference
  ranks 40th.

This reference panel is intentionally auditable but not comprehensive. One
assembly cannot represent all strains of a protected species, and GRCh38 does
not represent all human variation.

## Frozen flagging rule

A site is flagged only when it has a SpCas9 `NGG` PAM in the correct
orientation and meets either condition:

1. at most two mismatches across the full 20-nt spacer; or
2. an exact PAM-proximal eight-base seed and at most four total mismatches.

Insertions, deletions, non-NGG PAMs, chromatin or transcriptional context, and
RNA-level interactions are outside this pilot. Every flagged site is retained;
there is no post-result biological dismissal.

## Pilot gate and claim boundary

The pilot passes only if neither guide has a flagged site in any of the eleven
references. A pass means “no flagged site under this frozen rule in this frozen
panel.” It does not mean off-target-free, clinically safe, or ready for an
experiment. A broader isolate panel, platform-specific modeling, and direct
biological review remain mandatory.
