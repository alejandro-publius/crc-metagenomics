# Frozen CRC mechanism panel

This analysis tests four experimentally motivated microbial mechanisms without
selecting genes from cancer labels:

- colibactin genotoxicity (`clbA`-`clbS`)
- fusobacterial adhesion (`fadA`, `fap2`)
- *Bacteroides fragilis* toxin (`bft`)
- secondary bile-acid conversion (`bai` genes)

The UniProtKB-to-UniRef90 mapping was frozen before outcome modeling. Its exact
checksum and source snapshot are stored in `freeze.json`. Missing identifiers
were retained as missing rather than replaced by outcome-associated features.
The gene-family representation did not detect the mapped Fusobacterium adhesion
clusters, so that mechanism could not be evaluated in this assay.

## Result

| Representation | Mean LODO AUC | Fold range |
|---|---:|---:|
| Mechanism scores only | 0.569 | 0.539-0.604 |
| Parent species only | 0.656 | 0.522-0.829 |
| Mechanism scores plus parent species | 0.655 | 0.518-0.858 |

The frozen mechanism genes do not improve average cross-cohort discrimination
beyond their parent species in this representation. This result does not argue
that the mechanisms are biologically unimportant. It shows that their relative
abundance, as recovered from these gene-family tables, is not a portable CRC
classifier across the ten cohorts.

## Files

- `frozen_manifest.csv`: checksum-protected mapping used for modeling
- `uniprot_to_uniref90.csv`: query and accession provenance
- `cohort_coverage.csv`: label-independent representation coverage
- `sample_scores.csv`: mechanism and parent-species scores
- `lodo_results.csv`: country-aware held-out-cohort AUCs
- `predictions.csv`: held-out predictions for audit and comparison
