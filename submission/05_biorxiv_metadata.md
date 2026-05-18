# bioRxiv Pre-print Submission Metadata

Paste-ready fields for the bioRxiv submission form.

## Title
Cross-cohort gut microbiome classification of colorectal cancer:
a 10-cohort meta-analysis with country-aware leave-one-dataset-out
validation

## Abstract (paste field)
*Use the final abstract from `manuscript/CRC_Abstract.docx`. 250 words max.*

## Authors
Alex [Last Name]¹*, Rachel [Last Name]²

1. [Affiliation 1]
2. [Affiliation 2]

*Corresponding author. Email: [email]. ORCID: [iD]*

## Subject area
- **Primary**: Bioinformatics
- **Secondary**: Microbiology
- **Tertiary**: Cancer Biology

## Type of article
Research

## Funding
Self-funded undergraduate research. See
`submission/03_author_contributions.md`.

## License
CC-BY 4.0 (recommended; permits reuse with attribution)

## Conflict of interest
None declared.

## Data availability
All data and code publicly available. See
`submission/01_data_availability.md`.

## Manuscript file
Use `manuscript/CRC_Manuscript_Complete.docx` (single-PDF export).
Embed all fonts. Generate PDF via:

```bash
soffice --headless --convert-to pdf manuscript/CRC_Manuscript_Complete.docx \
    --outdir manuscript/
```

## Supplementary file
Bundle `results/supplementary/*.csv` + `manuscript/Supplementary_Tables.docx`
as a single ZIP. Reference S1-S10 in main text.

## Figures
Upload as separate high-resolution files
(`manuscript/figures/Figure{1..4}.{pdf,png}`).

## Related preprint history
None (first version).
