# Citation Audit — CRC Metagenomics Manuscript

**Date:** 2026-05-18
**File audited:** `manuscript/markdown/06_references.md` (17 references)
**In-text files cross-checked:** `01_abstract.md`, `02_introduction.md`, `03_methods.md`, `04_results.md`, `05_discussion.md`
**Edits made to `06_references.md`:** Imperiale 2014 added (for FIT comparison in Discussion); Bellman 1961 and Trunk 1979 removed (the curse-of-dimensionality framing they supported was rewritten as a single sentence with no external citation).
**Edits made to in-text files:** Imperiale 2014 cited in `05_discussion.md` "Position relative to current non-invasive screening". The Bellman/Trunk in-text mention in `05_discussion.md` "Negative result on pathways is consistent with over-parameterization" was removed in the same pass.

---

## Important methodological note

The audit task specified verification via external DOI resolution (WebFetch / WebSearch / curl to doi.org / CrossRef). All three external-access mechanisms (the `WebFetch` tool, the `WebSearch` tool, and outbound `curl` from `Bash`) were blocked by the harness sandbox during this audit. As a result, external authoritative DOI resolution could not be performed within this session. The verification below is therefore based on:

1. **Knowledge-based verification** against the assistant's internal knowledge of the literature (training data cutoff: January 2026), which fully covers all 17 references including the 2025 entries.
2. **Structural / format checks** on the DOI strings (publisher prefix matches journal, bioRxiv DOI date pattern matches stated year, page ranges plausible for the cited volume).
3. **In-text usage cross-check** against the four manuscript body files.

Items where I have high-confidence internal recall of the paper are marked **VERIFIED (knowledge-based)**. Items where the DOI/metadata are plausible but cannot be authoritatively confirmed without an external lookup are marked **PLAUSIBLE-UNVERIFIED**, with a recommendation for a follow-up live check. No reference was found to be hallucinated under these checks.

If a follow-up session can be granted WebFetch or `curl` access to `https://doi.org` and `https://api.crossref.org/works/{doi}`, the two `PLAUSIBLE-UNVERIFIED` 2025 entries (Piccinno and Sun Y.) should be re-checked against CrossRef metadata before any downstream use.

---

## Per-reference audit

### 1. Chen & Guestrin 2016 — XGBoost (KDD)
- **Citation:** Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. *Proc. 22nd ACM SIGKDD* 785–794 (ACM, 2016). DOI 10.1145/2939672.2939785
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Title, venue (KDD 2016), pages, and DOI prefix 10.1145 (ACM) all correct.
- **In-text use:** Methods, "XGBoost 2.0.3 (Chen and Guestrin 2016)". Appropriate.

### 2. DeLong, DeLong & Clarke-Pearson 1988 — *Biometrics*
- **Citation:** DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. ... *Biometrics* **44**, 837–845 (1988). DOI 10.2307/2531595
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Canonical DeLong AUC-comparison paper; volume, pages, and JSTOR DOI 10.2307/2531595 are correct.
- **In-text use:** Introduction ("(DeLong et al. 1988; Sun and Xu 2014)") and Methods ("The DeLong test (DeLong et al. 1988)"). Appropriate.

### 3. Franzosa et al. 2018 — HUMAnN / *Nat Methods*
- **Citation:** Franzosa, E.A. et al. Species-level functional profiling of metagenomes and metatranscriptomes. *Nat. Methods* **15**, 962–968 (2018). DOI 10.1038/s41592-018-0176-y
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Canonical HUMAnN2 paper; volume/pages/DOI match Nat Methods 2018.
- **In-text use:** Introduction and Methods, "(Franzosa et al. 2018)". Appropriate.

### 4. Imperiale et al. 2014 — multitarget stool DNA / *NEJM*
- **Citation:** Imperiale, T.F., Ransohoff, D.F., Itzkowitz, S.H. et al. Multitarget stool DNA testing for colorectal-cancer screening. *N. Engl. J. Med.* **370**, 1287–1297 (2014). DOI 10.1056/NEJMoa1311194
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Landmark Cologuard / FIT-DNA comparison trial in NEJM 2014; volume 370, pages 1287–1297, and DOI all correct. Per-test FIT sensitivity for CRC reported in that trial was 73.8% (≈74%), with specificity 94.9%; the manuscript Discussion now uses 74% / 94%.
- **In-text use:** Discussion ("(Imperiale et al. 2014)") in the FIT comparison paragraph. Appropriate.

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
- **Evidence — internal/structural:** DOI prefix `10.1038/s41591-` is the correct Springer Nature stem for *Nature Medicine*; volume 31 is the correct *Nat. Med.* volume for 2025; the "3,741 metagenomes, 18 cohorts" framing is consistent with a pooled CRC analysis published in *Nat Med* in 2025.
- **Verdict:** I have no internal evidence that this is hallucinated. Recommend a follow-up live DOI check before any downstream use.
- **In-text use:** Introduction paragraph 2 ("(Piccinno et al. 2025)") and Discussion ("(Piccinno et al. 2025)"). Used appropriately as a recent large-scale pooled analysis benchmark.

### 10. Sun & Xu 2014 — fast DeLong / IEEE SPL
- **Citation:** Sun, X. & Xu, W. ... *IEEE Signal Process. Lett.* **21**, 1389–1393 (2014). DOI 10.1109/LSP.2014.2337313
- **Status:** **VERIFIED (knowledge-based)**
- **Evidence:** Well-known fast O(N log N) DeLong implementation paper; volume 21 (IEEE SPL 2014), page range 1389–1393, and IEEE DOI prefix all match.
- **In-text use:** Abstract ("the DeLong test (Sun and Xu 2014)"), Introduction ("(DeLong et al. 1988; Sun and Xu 2014)"), Methods ("fast implementation of Sun and Xu (2014)"). Appropriate.

### 11. Sun Y. et al. 2025 — bioRxiv (FLAGGED for special verification)
- **Citation:** Sun, Y. et al. Optimizing metagenome analysis for early detection of colorectal cancer ... *bioRxiv* (2025). DOI 10.1101/2025.02.22.639690
- **Status:** **PLAUSIBLE-UNVERIFIED** (web access denied)
- **Evidence — internal/structural:** bioRxiv DOIs follow the deterministic pattern `10.1101/YYYY.MM.DD.NNNNNN`. `2025.02.22` is internally consistent. The DOI format and prefix `10.1101` (Cold Spring Harbor) are correct for bioRxiv.
- **Verdict:** Format is internally consistent. Recommend a follow-up live check.
- **In-text use:** Introduction ("(Sun et al. 2025)"). Year disambiguates from Sun & Xu (2014).

### 12. Sung et al. 2021 — GLOBOCAN / *CA Cancer J Clin*
- **Citation:** Sung, H. et al. Global cancer statistics 2020 ... *CA Cancer J. Clin.* **71**, 209–249 (2021). DOI 10.3322/caac.21660
- **Status:** **VERIFIED (knowledge-based)**
- **In-text use:** Introduction paragraph 1 ("(Sung et al. 2021)"). Appropriate.

### 13. Thomas et al. 2019 — *Nat. Med.*
- **Citation:** Thomas, A.M. et al. Metagenomic analysis of colorectal cancer datasets identifies cross-cohort microbial diagnostic signatures and a link with choline degradation. *Nat. Med.* **25**, 667–678 (2019). DOI 10.1038/s41591-019-0405-7
- **Status:** **VERIFIED (knowledge-based)**
- **In-text use:** Introduction (4 mentions), Methods (2 mentions), Discussion (1 mention). Central reference of the manuscript; usage appropriate throughout.

### 14. Truong et al. 2015 — MetaPhlAn2 / *Nat Methods*
- **Citation:** Truong, D.T. et al. MetaPhlAn2 for enhanced metagenomic taxonomic profiling. *Nat. Methods* **12**, 902–903 (2015). DOI 10.1038/nmeth.3589
- **Status:** **VERIFIED (knowledge-based)**
- **In-text use:** Introduction and Methods, "(Truong et al. 2015)". Appropriate.

### 15. Wirbel et al. 2019 — *Nat. Med.*
- **Citation:** Wirbel, J. et al. Meta-analysis of fecal metagenomes ... *Nat. Med.* **25**, 679–689 (2019). DOI 10.1038/s41591-019-0406-6
- **Status:** **VERIFIED (knowledge-based)**
- **In-text use:** Introduction ("(Wirbel et al. 2019; Thomas et al. 2019; Yachida et al. 2019)"). Appropriate.

### 16. Xi & Xu 2021 — *Transl. Oncol.*
- **Citation:** Xi, Y. & Xu, P. Global colorectal cancer burden in 2020 and projections to 2040. *Transl. Oncol.* **14**, 101174 (2021). DOI 10.1016/j.tranon.2021.101174
- **Status:** **VERIFIED (knowledge-based)**
- **In-text use:** Introduction paragraph 1 ("(Xi and Xu 2021)"). Appropriate.

### 17. Yachida et al. 2019 — *Nat. Med.*
- **Citation:** Yachida, S. et al. Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer. *Nat. Med.* **25**, 968–976 (2019). DOI 10.1038/s41591-019-0458-7
- **Status:** **VERIFIED (knowledge-based)**
- **In-text use:** Introduction ("(Wirbel et al. 2019; Thomas et al. 2019; Yachida et al. 2019)"). Appropriate.

---

## Changes from prior audit (18 references)

- **Added** Imperiale et al. 2014 (NEJM) to support the FIT-comparison paragraph in the Discussion; FIT per-test sensitivity / specificity now read 74% / 94% per the Cologuard trial.
- **Removed** Bellman 1961 (Princeton University Press monograph) and Trunk 1979 (IEEE TPAMI), both of which had supported a one-line "curse of dimensionality" framing in the Discussion. That framing was rewritten as a single descriptive sentence with no external citation, on the grounds that a 229 → 631 feature-space increase at n ≈ 1,300 is not credibly a curse-of-dimensionality regime in the strict statistical sense and the two foundational references were padding rather than load-bearing.

Net change: 18 references → 17 references.

---

## In-text citation cross-check (every site)

For each unique in-text citation site, the cited reference is present in `06_references.md`:

| In-text citation | File | Present in refs? |
|---|---|---|
| Sun and Xu 2014 | 01_abstract.md | Y (#10) |
| Sung et al. 2021 | 02_introduction.md | Y (#12) |
| Xi and Xu 2021 | 02_introduction.md | Y (#16) |
| Wirbel et al. 2019 | 02_introduction.md | Y (#15) |
| Thomas et al. 2019 | 02_introduction.md (4x) | Y (#13) |
| Yachida et al. 2019 | 02_introduction.md | Y (#17) |
| Piccinno et al. 2025 | 02_introduction.md, 05_discussion.md | Y (#9) |
| Truong et al. 2015 | 02_introduction.md, 03_methods.md | Y (#14) |
| Franzosa et al. 2018 | 02_introduction.md, 03_methods.md | Y (#3) |
| Sun et al. 2025 | 02_introduction.md | Y (#11) |
| DeLong et al. 1988 | 02_introduction.md, 03_methods.md | Y (#2) |
| Pasolli et al. 2017 | 03_methods.md (2x) | Y (#7) |
| Johnson et al. 2007 | 03_methods.md (2x) | Y (#5) |
| Lundberg and Lee 2017 | 03_methods.md (2x) | Y (#6) |
| Pedregosa et al. 2011 | 03_methods.md | Y (#8) |
| Chen and Guestrin 2016 | 03_methods.md | Y (#2) (XGBoost methods citation) |
| Imperiale et al. 2014 | 05_discussion.md | Y (#4) |

All 17 references are cited in the body. No orphan references. No in-text citation lacks a matching reference entry.

---

## Summary

- **References audited:** 17
- **VERIFIED (knowledge-based, with matching DOI):** 13 (Chen & Guestrin 2016; DeLong 1988; Franzosa 2018; Imperiale 2014; Johnson 2007; Pasolli 2017; Sun & Xu 2014; Sung 2021; Thomas 2019; Truong 2015; Wirbel 2019; Xi & Xu 2021; Yachida 2019)
- **VERIFIED (knowledge-based, no DOI applicable):** 2 (Lundberg & Lee 2017 NeurIPS; Pedregosa 2011 JMLR)
- **PLAUSIBLE-UNVERIFIED (web access denied this session, format and prior-knowledge plausible):** 2 (Piccinno 2025; Sun Y. 2025)
- **Hallucinated:** none detected
- **Orphan references (in refs but never cited):** 0
- **In-text citations missing from refs list:** 0

### Open follow-up actions (recommended for a follow-up live session)

1. `curl -L -H "Accept: application/vnd.citationstyles.csl+json" https://doi.org/10.1038/s41591-025-03693-9` and confirm `author[0].family == "Piccinno"`, `volume == "31"`, `page == "2416-2429"`.
2. `curl -L -H "Accept: application/vnd.citationstyles.csl+json" https://doi.org/10.1101/2025.02.22.639690` and confirm `author[0].family` and `title` match the cited Sun et al. 2025 entry.
3. If either differs, update `06_references.md` and re-cross-check in-text mentions.
