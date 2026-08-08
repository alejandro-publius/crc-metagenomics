# Pre-score declaration: public external profiles

Recorded before any PRJNA763023 outcome score was calculated.

## Unavoidable implementation change

The frozen plan called for downloading and locally profiling 200 paired WGS
runs with one MetaPhlAn database. ENA reports roughly 2.15 trillion sequenced
bases for this cohort, making that route a multi-day, high-bandwidth
recalculation rather than a scientifically necessary part of the locked test.

GMrepo v3 independently provides profiles for exactly the same 200 frozen run
accessions. Its published method processed WGS data with MetaPhlAn 4.1.0 using
default settings and normalized species abundances to 100%. We therefore use
those public profiles as the external abundance source. This changes neither
the frozen samples, labels, training data, feature set, model, nor outcome.

## Mapping frozen before scoring

1. Verify the GMrepo run set equals the committed manifest exactly.
2. Require successful GMrepo QC and species abundances summing to 100% for
   every run.
3. Map each scientific name to a locked training feature using only its
   normalized terminal species name: underscores become spaces, square
   brackets are removed, case and repeated whitespace are ignored.
4. Do not add taxonomic synonyms or hand mappings after viewing performance.
5. Fill absent locked features with zero, renormalize within the locked
   229-feature space, and apply `log10(x + 1e-6)`.
6. Fit the already specified 500-tree RF to all internal CRC/control samples
   and evaluate the 200 external samples once.

The external result will be retained regardless of direction. A future raw-read
recalculation may be reported as a sensitivity analysis, never as a replacement
chosen because it scores better.

## Provenance

- External study: <https://doi.org/10.1038/s41467-021-27112-y>
- Public profile resource: <https://gmrepo.humangut.info/>
- GMrepo v3 processing methods: <https://doi.org/10.1093/nar/gkaf1190>
