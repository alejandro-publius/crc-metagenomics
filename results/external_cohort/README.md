# Frozen external-cohort path

The external cohort is the publicly released shotgun subset of PRJNA763023,
from *Dysbiosis of human gut microbiome in young-onset colorectal cancer*
([Nature Communications, 2021](https://doi.org/10.1038/s41467-021-27112-y)).
It is outside the ten curatedMetagenomicData development cohorts.

The committed manifest is generated directly from the ENA Portal API. Labels
are frozen from the publication's sample naming: `M_O_` and `M_Y_` are older-
and younger-onset CRC; `M_HO_` and `M_HY_` are their healthy controls.

The earlier scouting memo was wrong about this being a 110-sample Wu cohort.
The verified accession is a different study and exposes 200 public WGS runs:
50 older-onset CRC, 50 younger-onset CRC, and 50 matched controls for each age
group. PRJEB57847 is also not a valid larger fallback—it exposes only 13 public
runs. These corrections were made before viewing any external model
performance.

Reproduce the manifest:

```bash
python3 scripts/prepare_external_cohort.py
```

After profiling every manifest FASTQ with one pinned MetaPhlAn database and
merging the profiles, score the untouched cohort with:

```bash
python3 scripts/score_external_species.py merged_metaphlan_table.tsv
```

The scorer exactly reindexes to the 229 training species, renormalizes across
that locked feature space, applies the training `log10(x + 1e-6)` transform,
fits the locked RF on all internal CRC/control samples, and writes predictions
plus AUC. Until those profiles exist, `audit.json` remains explicitly marked
`manifest_frozen_profiles_pending`; no external performance is claimed.
