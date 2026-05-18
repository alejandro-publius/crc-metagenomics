# bioRxiv Submission Package — Quick-Start Index

Goal: get this preprint posted on bioRxiv in ~15 minutes.

## Order of operations (do these in order)

1. **Open bioRxiv** (https://submit.biorxiv.org) and sign in.
   Confirm your ORCID (0009-0007-9798-1958) is linked to the account.
2. **Generate the PDF** from the .docx if you have not already:
   ```bash
   soffice --headless --convert-to pdf \
     manuscript/CRC_Manuscript_Complete.docx --outdir submission/build/
   ```
   Or run `python3 scripts/build_biorxiv_pdf.py`.
   Output expected: `submission/build/CRC_Manuscript_Complete.pdf`.
3. **Click "New submission"** in the bioRxiv portal. As you walk through
   the form, copy values straight out of the files in this folder:
   - `00_bioRxiv_form_fields.md` — every form-field value, in order.
   - `01_authors_and_affiliations.txt` — paste into the Authors section.
   - `02_competing_interests.txt` — competing-interests line.
   - `03_data_and_code_availability.txt` — data/code availability text.
   - `04_keywords.txt` — keyword list.
4. **Upload files** when prompted:
   - Main manuscript PDF (from step 2).
   - Figures: `manuscript/figures/Figure{1..4}.{pdf,png}` as separate files.
   - Supplementary ZIP: bundle `results/supplementary/*.csv` plus
     `manuscript/Supplementary_Tables.docx` into one .zip and upload.
5. **License:** select **CC-BY 4.0**.
6. **Preview** the submission, take a screenshot for your records, click
   submit. bioRxiv will return a DOI within 24-48h.
7. **After the DOI lands**, update `CITATION.cff` and `.zenodo.json` with
   the bioRxiv DOI, push to GitHub, and queue your announcement
   (see `outreach/`).

Tick items off as you go in `submit_checklist.md`.

If you decide later to send the paper to a journal,
`target_journal_after_preprint.md` ranks fit and realistic acceptance odds.

## File map

| File | Use |
|---|---|
| `README.md` | This index |
| `00_bioRxiv_form_fields.md` | Every bioRxiv form field, paste-ready |
| `01_authors_and_affiliations.txt` | Authors block |
| `02_competing_interests.txt` | Competing-interests one-liner |
| `03_data_and_code_availability.txt` | Data + code availability statement |
| `04_keywords.txt` | Keyword list |
| `submit_checklist.md` | 10-item submission checklist |
| `target_journal_after_preprint.md` | Ranked post-preprint journal targets |

Estimated time-to-submit if PDF and figures are already built: **~15 min**.
Add ~10 min if you need to regenerate the PDF.
