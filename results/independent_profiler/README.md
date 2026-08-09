# Independent targeted raw-read pilot

This directory records a bounded independence check for the frozen mechanism
panel. Four samples are selected deterministically: one control and one CRC
sample from each of two prespecified cohorts. Up to 250,000 raw reads per
sample are translated and aligned directly to the frozen UniProt proteins
with DIAMOND. Calls require at least 90% protein identity across at least 30
aligned amino acids. The path does not reuse the HUMAnN/UniRef abundance table used
by the main mechanism analysis.

This is **not** a second full functional profile and is not powered to compare
CRC with controls. A missing hit at this read depth does not establish
biological absence, and short targeted matches do not by themselves establish
a complete operon. In this bounded run, high-stringency colibactin matches were
recovered from both CRC samples and neither control; that four-sample pattern is
technical evidence that the direct path works, not an association estimate.
The committed audit and summaries make that boundary machine-readable. Raw
reads, the protein database, and alignment files remain under ignored
`data/interim/` paths.

Reproduce after installing DIAMOND and the NCBI SRA Toolkit:

```bash
Rscript scripts/export_accession_manifest.R
python3 scripts/run_independent_profiler.py --max-reads 250000
```
