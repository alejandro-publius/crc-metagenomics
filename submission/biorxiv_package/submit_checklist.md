# bioRxiv Submission Checklist

Tick each box as you go. All 10 should be DONE before you click Submit.

- [ ] Manuscript PDF generated — output of `python3 scripts/build_biorxiv_pdf.py`, or `soffice --headless --convert-to pdf manuscript/CRC_Manuscript_Complete.docx --outdir submission/build/`. PDF lives at `submission/build/CRC_Manuscript_Complete.pdf`.
- [ ] Figures uploaded as separate high-resolution files (Figure1-Figure4, PDF preferred, PNG >=300 DPI as fallback) from `manuscript/figures/`.
- [ ] Supplementary files bundled as a single ZIP (`results/supplementary/*.csv` + `manuscript/Supplementary_Tables.docx`) and uploaded.
- [ ] All metadata fields from `00_bioRxiv_form_fields.md` filled in the portal (title, abstract, subject area, type, funding, license, conflict of interest, authors, ORCIDs, suggested reviewers, keywords, data/code availability).
- [ ] bioRxiv account active and email-verified.
- [ ] CC-BY-4.0 license selected.
- [ ] ORCIDs linked: corresponding author 0009-0007-9798-1958, co-author 0009-0006-7046-3192.
- [ ] Preview screenshot saved (PDF render of the bioRxiv preview page, kept in `submission/build/biorxiv_preview.png` for your records).
- [ ] DOI generated post-submission added to `CITATION.cff` (top-level `doi:` field) and `.zenodo.json` (`related_identifiers` entry, relation `isPreprintOf`).
- [ ] Tweet / announcement queued (draft lives in `outreach/`; schedule for the morning after the preprint goes live so the DOI is resolvable).
