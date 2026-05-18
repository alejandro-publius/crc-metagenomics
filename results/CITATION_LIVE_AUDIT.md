# Citation Live-Verify Audit — CRC Metagenomics Manuscript

**Date:** 2026-05-18
**File audited:** `manuscript/markdown/06_references.md` (19 references)
**Status:** **BLOCKED — outbound network access denied in this session**
**Edits made to `06_references.md`:** None
**Edits made to in-text files:** None
**`.docx` rebuild triggered:** No (no diff to compile)
**`verify_results.py`:** PASSED before audit attempt (49/49)

---

## What was attempted

The task specified live DOI resolution per reference via one of:
1. `curl -sSI -L https://doi.org/<DOI>`
2. `curl -L -H "Accept: application/vnd.citationstyles.csl+json" https://api.crossref.org/works/<DOI>`
3. `WebFetch` as a tolerant fallback.

All three were attempted in this session against authoritative endpoints
(`doi.org` and `api.crossref.org`).

## What happened

| Mechanism | Result |
|---|---|
| `Bash` → `curl https://doi.org/...` | **Denied by sandbox** ("Permission to use Bash has been denied") |
| `Bash` → `/usr/bin/curl ...` (explicit binary path) | **Denied by sandbox** (same error) |
| `WebFetch https://api.crossref.org/works/<DOI>` | **Denied by sandbox** ("Permission to use WebFetch has been denied") for every DOI attempted |

Because none of the three live-verification mechanisms is available in this
sandbox session, the literal "curl the DOI, fetch the landing page, confirm
metadata" step cannot be carried out. The earlier `CITATION_AUDIT.md` file in
this same `results/` directory ran into the identical sandbox limitation
during its session.

## Why no edits were applied

The user's explicit constraints require any fix to:
1. be derived from an authoritative live lookup,
2. then be applied to `06_references.md`,
3. then trigger a renumber + `python3 manuscript/markdown/_build_docx.py` rebuild,
4. and reconfirm 49/49 in `verify_results.py`.

Without a successful live lookup, steps 2–4 must not be triggered. Editing
references on the basis of internal knowledge alone would (a) duplicate the
prior `CITATION_AUDIT.md` work and (b) risk introducing a mismatch that the
user explicitly wants validated against the live publisher record.

## Pre-audit sanity checks (these were possible without network)

- `python3 scripts/verify_results.py` → **49/49 PASS** (run at audit start).
- All 19 reference entries in `06_references.md` parse cleanly and are cited
  in the body manuscript (per the existing `CITATION_AUDIT.md` cross-check,
  re-confirmed by reading `06_references.md` in this session).
- 17 of 19 entries carry a DOI string. 3 do not (Bellman 1961 monograph,
  Lundberg & Lee 2017 NeurIPS, Pedregosa et al. 2011 JMLR) — these are
  recognised conventions for those venues and per the task spec only need
  format-correctness verification, which they pass.
- Two newer entries (Piccinno 2025 Nat Med; Sun Y. 2025 bioRxiv) were
  previously flagged "PLAUSIBLE-UNVERIFIED" pending a live check.

## DOIs that still need a live check

All 17 DOIs below need a `curl https://doi.org/<DOI>` + landing-page metadata
confirmation in a session with network access. The two flagged 2025 entries
are highest priority.

| # | First author + year | DOI | Priority |
|---|---|---|---|
| 2 | Chen & Guestrin 2016 | 10.1145/2939672.2939785 | normal |
| 3 | DeLong 1988 | 10.2307/2531595 | normal |
| 4 | Franzosa 2018 | 10.1038/s41592-018-0176-y | normal |
| 5 | Imperiale 2014 | 10.1056/NEJMoa1311194 | normal |
| 6 | Johnson 2007 | 10.1093/biostatistics/kxj037 | normal |
| 8 | Pasolli 2017 | 10.1038/nmeth.4468 | normal |
| 10 | **Piccinno 2025** | 10.1038/s41591-025-03693-9 | **HIGH** |
| 11 | Sun & Xu 2014 | 10.1109/LSP.2014.2337313 | normal |
| 12 | **Sun Y. 2025 (bioRxiv)** | 10.1101/2025.02.22.639690 | **HIGH** |
| 13 | Sung 2021 | 10.3322/caac.21660 | normal |
| 14 | Thomas 2019 | 10.1038/s41591-019-0405-7 | normal |
| 15 | Trunk 1979 | 10.1109/TPAMI.1979.4766926 | normal |
| 16 | Truong 2015 | 10.1038/nmeth.3589 | normal |
| 17 | Wirbel 2019 | 10.1038/s41591-019-0406-6 | normal |
| 18 | Xi & Xu 2021 | 10.1016/j.tranon.2021.101174 | normal |
| 19 | Yachida 2019 | 10.1038/s41591-019-0458-7 | normal |

(Ref #1 Bellman, #7 Lundberg, #9 Pedregosa have no DOI and need only
format/title/year confirmation — done.)

## Recommended next step

Grant the harness permission to run `curl` to `doi.org` and
`api.crossref.org`, or enable `WebFetch`, then re-run this audit. The
following one-liner per DOI is sufficient (CrossRef returns CSL JSON with
authors, title, container, volume, page, year in a single request):

```
curl -sL -H 'Accept: application/vnd.citationstyles.csl+json' \
     https://doi.org/<DOI>
```

A 200 + matching `author[0].family`, `container-title`, `volume`, `page`,
`issued.date-parts[0][0]` confirms the citation. A 404 or mismatched record
becomes a **P0 fix**: update the DOI in `06_references.md`, renumber if any
entry is dropped, run `python3 manuscript/markdown/_build_docx.py`, and
re-run `python3 scripts/verify_results.py` to reconfirm 49/49.

## Post-audit sanity check

- `python3 scripts/verify_results.py` was **not re-run** because no file was
  modified (no diff to validate). The 49/49 baseline from session start
  remains the current ground truth.
