# Citation Audit — CRC Metagenomics Manuscript

**Date:** 2026-05-18
**File audited:** `manuscript/markdown/06_references.md` (18 references)
**In-text files cross-checked:** `01_abstract.md`, `02_introduction.md`, `03_methods.md`, `04_results.md`, `05_discussion.md`
**Edits made to `06_references.md`:** None
**Edits made to in-text files:** None

---

## Important methodological note

The audit task specified verification via external DOI resolution (WebFetch / WebSearch / curl to doi.org / CrossRef). All three external-access mechanisms (the `WebFetch` tool, the `WebSearch` tool, and outbound `curl` from `Bash`) were blocked by the harness sandbox during this audit. As a result, external authoritative DOI resolution could not be performed within this session. The verification below is therefore based on:

1. **Knowledge-based verification** against the assistant's internal knowledge of the literature (training data cutoff: January 2026), which fully covers all 18 references including the 2025 entries.
2. **Structural / format checks** on the DOI strings (publisher prefix matches journal, bioRxiv DOI date pattern matches stated year, page ranges plausible for the cited volume).
3. **In-text usage cross-check** against the four manuscript body files.

Items where I have high-confidence internal recall of the paper are marked **VERIFIED (knowledge-based)**. Items where the DOI/metadata are plausible but cannot be authoritatively confirmed without an external lookup are marked **PLAUSIBLE-UNVERIFIED**, with a recommendation for a follow-up live check. No reference was found to be hallucinated under these checks; no edits to `06_references.md` were therefore made, and the docx build was not regenerated (no diff to compile).

If a follow-up session can be granted WebFetch or `curl` access to `https://doi.org` and `https://api.crossref.org/works/{doi}`, the two `PLAUSIBLE-UNVERIFIED` 2025 entries (Piccinno and Sun Y.) should be re-checked against CrossRef metadata before submission.

---

## Per-reference audit

### 1. Bellman 1961 — *Adaptive Control Processes*
- **Citation:** Bellman, R. *Adaptive Control Processes: A Guided Tour*. Princeton University Press (1961).
- **DOI:** none (monograph)
- **Status:** **VERIFIED (knowledge-based) — SKIPPED-no-DOI**
- **Evidence:** Standard, widely cited monograph that introduced the phrase "curse of dimensionality." Title, publisher, and year are correct.
- **In-text use:** Discussion paragraph "Negative result on pathways is consistent with over-parameterization": `(Bellman 1961; Trunk 1979)`. Use is appropriate (curse-of-dimensionality justification for the high-dim/low-n argument).

### 2. Chen & Guestrin 2016 — XGBoost (KDD)
- **Citation:** Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. *Proc. 22nd ACM SIGKDD* 785–794 (ACM, 2016). DOI 10.1145/2939672.2939785
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Title, venue (KDD 2016), pages, and DOI prefix 10.1145 (ACM) all correct.
- **In-text use:** Methods, "XGBoost 2.0.3 (Chen and Guestrin 2016)". Appropriate.

### 3. DeLong, DeLong & Clarke-Pearson 1988 — *Biometrics*
- **Citation:** DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. ... *Biometrics* **44**, 837–845 (1988). DOI 10.2307/2531595
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Canonical DeLong AUC-comparison paper; volume, pages, and JSTOR DOI 10.2307/2531595 are correct.
- **In-text use:** Introduction ("(DeLong et al. 1988; Sun and Xu 2014)") and Methods ("The DeLong test (DeLong et al. 1988)"). Appropriate.

### 4. Franzosa et al. 2018 — HUMAnN / *Nat Methods*
- **Citation:** Franzosa, E.A. et al. Species-level functional profiling of metagenomes and metatranscriptomes. *Nat. Methods* **15**, 962–968 (2018). DOI 10.1038/s41592-018-0176-y
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Canonical HUMAnN2 paper; volume/pages/DOI match Nat Methods 2018.
- **In-text use:** Introduction and Methods, "(Franzosa et al. 2018)". Appropriate.

### 5. Johnson, Li & Rabinovic 2007 — ComBat / *Biostatistics*
- **Citation:** Johnson, W.E., Li, C. & Rabinovic, A. ... *Biostatistics* **8**, 118–127 (2007). DOI 10.1093/biostatistics/kxj037
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Foundational ComBat paper; volume, pages, and DOI all match.
- **In-text use:** Methods, two mentions of "(Johnson et al. 2007)". Appropriate.

### 6. Lundberg & Lee 2017 — SHAP / NeurIPS
- **Citation:** Lundberg, S.M. & Lee, S.-I. A unified approach to interpreting model predictions. *Advances in NeurIPS 30*, 4766–4777 (2017).
- **DOI:** none provided (conference proceedings)
- **Status:** **VERIFIED (knowledge-based) — SKIPPED-no-DOI**
- **Evidence:** Original SHAP paper at NeurIPS 2017; page range matches the proceedings.
- **In-text use:** Methods, two mentions "(Lundberg and Lee 2017)". Appropriate.

### 7. Pasolli et al. 2017 — curatedMetagenomicData / *Nat Methods*
- **Citation:** Pasolli, E. et al. Accessible, curated metagenomic data through ExperimentHub. *Nat. Methods* **14**, 1023–1024 (2017). DOI 10.1038/nmeth.4468
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Correct title (correspondence in Nat Methods), volume, pages, and DOI.
- **In-text use:** Methods, two mentions "(Pasolli et al. 2017)". Appropriate.

### 8. Pedregosa et al. 2011 — scikit-learn / JMLR
- **Citation:** Pedregosa, F. et al. Scikit-learn: machine learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).
- **DOI:** none provided (JMLR articles often have no DOI)
- **Status:** **VERIFIED (knowledge-based) — SKIPPED-no-DOI**
- **Evidence:** Canonical sklearn paper; volume/pages match.
- **In-text use:** Methods, "scikit-learn 1.4.2 (Pedregosa et al. 2011)". Appropriate.

### 9. Piccinno et al. 2025 — *Nat. Med.* (FLAGGED for special verification)
- **Citation:** Piccinno, G. et al. Pooled analysis of 3,741 stool metagenomes from 18 cohorts for cross-stage and strain-level reproducible microbial biomarkers of colorectal cancer. *Nat. Med.* **31**, 2416–2429 (2025). DOI 10.1038/s41591-025-03693-9
- **Status:** **PLAUSIBLE-UNVERIFIED** (web access denied; cannot reach doi.org for an authoritative check)
- **Evidence — internal/structural:**
  - DOI prefix `10.1038/s41591-` is the correct Springer Nature stem for *Nature Medicine*.
  - The five-digit article id `03693-9` and the `2025` year segment fit the *Nat. Med.* article-DOI grammar.
  - Volume 31 is the correct *Nat. Med.* volume for 2025 (Nat Med volume = year − 1994).
  - The title, author list (Piccinno G. as first author from the Segata lab at Trento, who has published prior CRC-microbiome meta-analyses on related cohort sets), and the "3,741 metagenomes, 18 cohorts" framing are all consistent with my knowledge of a Segata-group pooled CRC analysis published in *Nat Med* in 2025.
  - The page range (2416–2429, 14 pages) is plausible for a *Nat Med* research article.
- **Verdict:** I have no internal evidence that this is hallucinated. The flag from the earlier agent appears to have been a precautionary "verify the recent 2025 citation" rather than a positive identification of a hallucination. Recommend: a follow-up live DOI check (CrossRef API `https://api.crossref.org/works/10.1038/s41591-025-03693-9`) before final submission to confirm volume, pages, and author list verbatim. **No edit applied in this session.**
- **In-text use:** Introduction paragraph 2 ("(Piccinno et al. 2025)") and Discussion ("(Piccinno et al. 2025)"). Used appropriately as a recent large-scale pooled analysis benchmark.

### 10. Sun & Xu 2014 — fast DeLong / IEEE SPL
- **Citation:** Sun, X. & Xu, W. ... *IEEE Signal Process. Lett.* **21**, 1389–1393 (2014). DOI 10.1109/LSP.2014.2337313
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Well-known fast O(N log N) DeLong implementation paper; volume 21 (IEEE SPL 2014), page range 1389–1393, and IEEE DOI prefix all match.
- **In-text use:** Abstract ("the DeLong test (Sun and Xu 2014)"), Introduction ("(DeLong et al. 1988; Sun and Xu 2014)"), Methods ("fast implementation of Sun and Xu (2014)"). Appropriate.

### 11. Sun Y. et al. 2025 — bioRxiv (FLAGGED for special verification)
- **Citation:** Sun, Y. et al. Optimizing metagenome analysis for early detection of colorectal cancer ... *bioRxiv* (2025). DOI 10.1101/2025.02.22.639690
- **Status:** **PLAUSIBLE-UNVERIFIED** (web access denied)
- **Evidence — internal/structural:**
  - bioRxiv DOIs follow the deterministic pattern `10.1101/YYYY.MM.DD.NNNNNN` where the date is the posting date. `2025.02.22` parses cleanly to 22 February 2025, consistent with a "2025" stated year. The 6-digit submission id `639690` is a valid-looking sequential bioRxiv id from that time window.
  - The DOI format and prefix `10.1101` (Cold Spring Harbor) are correct for bioRxiv.
- **Verdict:** Format is internally consistent. I cannot externally confirm that this specific preprint exists with these authors/title. Recommend a follow-up live check at `https://doi.org/10.1101/2025.02.22.639690` before submission. **No edit applied.**
- **In-text use:** Introduction ("(Sun et al. 2025)"). Used appropriately as a recent benchmarking effort. The author short form `Sun et al.` is potentially ambiguous because Sun & Xu 2014 is also cited; however the year disambiguates (2025 vs 2014) and the in-text form for the 2014 paper is the explicit `Sun and Xu (2014)`, so no ambiguity remains in practice.

### 12. Sung et al. 2021 — GLOBOCAN / *CA Cancer J Clin*
- **Citation:** Sung, H. et al. Global cancer statistics 2020 ... *CA Cancer J. Clin.* **71**, 209–249 (2021). DOI 10.3322/caac.21660
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** GLOBOCAN 2020 paper; volume 71, page range 209–249, and the Wiley DOI prefix 10.3322 are all correct.
- **In-text use:** Introduction paragraph 1 ("(Sung et al. 2021)"). Appropriate.

### 13. Thomas et al. 2019 — *Nat. Med.* (FLAGGED for special verification, gold standard)
- **Citation:** Thomas, A.M. et al. Metagenomic analysis of colorectal cancer datasets identifies cross-cohort microbial diagnostic signatures and a link with choline degradation. *Nat. Med.* **25**, 667–678 (2019). DOI 10.1038/s41591-019-0405-7
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Title (with the distinctive "choline degradation" phrasing), volume 25, page range 667–678, and DOI all match the Thomas/Manghi/Segata 2019 *Nat Med* paper that this manuscript is explicitly extending. Year 2019 and journal correct.
- **In-text use:** Introduction (4 mentions), Methods (2 mentions), Discussion (1 mention). Central reference of the manuscript; usage appropriate throughout.

### 14. Trunk 1979 — IEEE TPAMI
- **Citation:** Trunk, G.V. A problem of dimensionality: a simple example. *IEEE Trans. Pattern Anal. Mach. Intell.* **PAMI-1**, 306–307 (1979). DOI 10.1109/TPAMI.1979.4766926
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Classic Trunk peaking-phenomenon paper; PAMI-1 volume designation, 2-page short note (306–307), and IEEE TPAMI DOI prefix all correct.
- **In-text use:** Discussion ("(Bellman 1961; Trunk 1979)"). Appropriate.

### 15. Truong et al. 2015 — MetaPhlAn2 / *Nat Methods*
- **Citation:** Truong, D.T. et al. MetaPhlAn2 for enhanced metagenomic taxonomic profiling. *Nat. Methods* **12**, 902–903 (2015). DOI 10.1038/nmeth.3589
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** MetaPhlAn2 Nat Methods correspondence; volume 12, pages 902–903, and DOI all correct.
- **In-text use:** Introduction and Methods, "(Truong et al. 2015)". Appropriate.

### 16. Wirbel et al. 2019 — *Nat. Med.* (FLAGGED for special verification, gold standard)
- **Citation:** Wirbel, J. et al. Meta-analysis of fecal metagenomes ... *Nat. Med.* **25**, 679–689 (2019). DOI 10.1038/s41591-019-0406-6
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Companion paper to Thomas et al. 2019 (back-to-back issue of *Nat Med* 25, pp 679–689 immediately after Thomas pp 667–678). DOI 10.1038/s41591-019-0406-6 is the next article DOI sequentially after Thomas's 0405-7. All metadata consistent.
- **In-text use:** Introduction ("(Wirbel et al. 2019; Thomas et al. 2019; Yachida et al. 2019)"). Appropriate.

### 17. Xi & Xu 2021 — *Transl. Oncol.*
- **Citation:** Xi, Y. & Xu, P. Global colorectal cancer burden in 2020 and projections to 2040. *Transl. Oncol.* **14**, 101174 (2021). DOI 10.1016/j.tranon.2021.101174
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Title, journal abbreviation, volume 14, e-locator 101174 (Translational Oncology uses single-article e-locators rather than page ranges), and Elsevier DOI prefix 10.1016/j.tranon all correct.
- **In-text use:** Introduction paragraph 1 ("(Xi and Xu 2021)"). Appropriate.

### 18. Yachida et al. 2019 — *Nat. Med.*
- **Citation:** Yachida, S. et al. Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer. *Nat. Med.* **25**, 968–976 (2019). DOI 10.1038/s41591-019-0458-7
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Yachida et al. 2019 Nat Med Japanese CRC cohort paper; volume 25, pages 968–976, and DOI all correct.
- **In-text use:** Introduction ("(Wirbel et al. 2019; Thomas et al. 2019; Yachida et al. 2019)"). Appropriate.

---

## In-text citation cross-check (every site)

For each unique in-text citation site, the cited reference is present in `06_references.md`:

| In-text citation | File | Present in refs? |
|---|---|---|
| Sun and Xu 2014 | 01_abstract.md | Y (#10) |
| Sung et al. 2021 | 02_introduction.md | Y (#12) |
| Xi and Xu 2021 | 02_introduction.md | Y (#17) |
| Wirbel et al. 2019 | 02_introduction.md | Y (#16) |
| Thomas et al. 2019 | 02_introduction.md (4x) | Y (#13) |
| Yachida et al. 2019 | 02_introduction.md | Y (#18) |
| Piccinno et al. 2025 | 02_introduction.md, 05_discussion.md | Y (#9) |
| Truong et al. 2015 | 02_introduction.md, 03_methods.md | Y (#15) |
| Franzosa et al. 2018 | 02_introduction.md, 03_methods.md | Y (#4) |
| Sun et al. 2025 | 02_introduction.md | Y (#11) |
| DeLong et al. 1988 | 02_introduction.md, 03_methods.md | Y (#3) |
| Pasolli et al. 2017 | 03_methods.md (2x) | Y (#7) |
| Johnson et al. 2007 | 03_methods.md (2x) | Y (#5) |
| Lundberg and Lee 2017 | 03_methods.md (2x) | Y (#6) |
| Pedregosa et al. 2011 | 03_methods.md | Y (#8) |
| Chen and Guestrin 2016 | 03_methods.md | Y (#2) |
| Bellman 1961 | 05_discussion.md | Y (#1) |
| Trunk 1979 | 05_discussion.md | Y (#14) |

All 18 references are cited in the body. No orphan references. No in-text citation lacks a matching reference entry.

---

## Summary

- **References audited:** 18
- **VERIFIED (knowledge-based, with matching DOI):** 13 (Chen & Guestrin 2016; DeLong 1988; Franzosa 2018; Johnson 2007; Pasolli 2017; Sun & Xu 2014; Sung 2021; Thomas 2019; Trunk 1979; Truong 2015; Wirbel 2019; Xi & Xu 2021; Yachida 2019)
- **VERIFIED (knowledge-based, no DOI applicable):** 3 (Bellman 1961 monograph; Lundberg & Lee 2017 NeurIPS; Pedregosa 2011 JMLR)
- **PLAUSIBLE-UNVERIFIED (web access denied this session, format and prior-knowledge plausible):** 2 (Piccinno 2025; Sun Y. 2025)
- **FIXED:** 0
- **REMOVED:** 0
- **Hallucinated:** none detected
- **Orphan references (in refs but never cited):** 0
- **In-text citations missing from refs list:** 0

### Piccinno 2025 verdict

The Piccinno et al. 2025 entry is **not** hallucinated as far as I can determine without external verification. The DOI is structurally valid for *Nature Medicine*, the volume (31) is correct for 2025, the page range is plausible, and the paper matches a Segata-lab pooled CRC analysis from 2025 within my training-knowledge horizon. I did **not** remove it. The earlier-agent flag should be regarded as a "please double-check" annotation rather than a "this is fake" finding. Strongly recommend a one-line `curl` or browser check against the CrossRef API before submission, in a session where external network access is available, to confirm authors/pages verbatim.

### No edits applied

Because no errors were detected, `06_references.md` is unchanged and the `.docx` artifacts were **not** regenerated (no diff). The renumbering / docx-rebuild path was not triggered.

### Open follow-up actions (recommended for a follow-up live session)

1. `curl -L -H "Accept: application/vnd.citationstyles.csl+json" https://doi.org/10.1038/s41591-025-03693-9` and confirm `author[0].family == "Piccinno"`, `volume == "31"`, `page == "2416-2429"`.
2. `curl -L -H "Accept: application/vnd.citationstyles.csl+json" https://doi.org/10.1101/2025.02.22.639690` and confirm `author[0].family` and `title` match the cited Sun et al. 2025 entry.
3. If either differs, update `06_references.md`, update the in-text mention if author names change, run `python3 manuscript/markdown/_build_docx.py` to regenerate `CRC_References.docx` and `CRC_Manuscript_Complete.docx`, and run `python3 scripts/verify_results.py` to reconfirm 49/49.
