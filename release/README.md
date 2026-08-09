# v2.0.0-rc1 release candidate

This package is complete for scientific and coauthor review. It contains the
render-checked manuscript, both final figures, frozen external profiles and
predictions, bootstrap uncertainty, generalization-risk outputs, provenance,
relevant source scripts, and exact environment declarations.

It is not the final archival release. The following external actions remain:

1. both authors approve the manuscript and declarations;
2. the approved commit is merged;
3. Zenodo archives that commit and returns a persistent DOI;
4. the DOI is inserted into the manuscript and metadata;
5. the corresponding author submits through ASM's authenticated portal.

`MANIFEST.json` records every packaged file's SHA-256 digest. The ZIP is built
by `python3 scripts/build_release_candidate.py`.
