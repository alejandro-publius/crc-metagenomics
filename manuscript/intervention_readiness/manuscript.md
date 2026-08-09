# Cross-population colorectal-cancer gene-family biomarkers fail taxonomic address resolution for precision microbiome editing

**Working manuscript — not approved for submission**

Authorship, affiliations, conflicts, and final biological interpretation require
direct coauthor review.

## Abstract

**Background:** Stool metagenomic biomarkers can recur across colorectal cancer
(CRC) cohorts without identifying a sequence that can safely address a harmful
microbial strain. Precision microbiome editing therefore requires evidence
gates beyond disease prediction.

**Methods:** We evaluated 6,755 unstratified UniRef90 gene families across 1,339
CRC and control samples from ten public cohorts. Candidate discovery was
repeated inside country-aware leave-one-dataset-out folds so that held-out
cohorts could not select their own features. Internally recurring candidates
were tested for discrimination beyond an exactly matched annotated parent
species. Parent-adjustment survivors were then traced to their HUMAnN
taxon-stratified carriers. The taxonomic-address gate, frozen before the
taxon-resolved analysis, required one carrier to account for at least 80% of
stratified abundance.

**Results:** Sixteen of 6,755 families (0.24%) passed the internal recurrence
screen. Four of sixteen added a median held-out AUC of at least 0.02 beyond
their annotated parent-species proxy and improved AUC in at least 70% of
evaluable folds. None passed taxonomic address resolution. Each family was
distributed across 7–23 taxa, and its largest carrier accounted for only
13.5–42.5% of stratified abundance. The species implied by the archived UniRef
representative accounted for 0–19.6% and was never the largest carrier. No
candidate passed even a post hoc 50% majority-carrier sensitivity threshold.

**Conclusions:** Cross-population recurrence and parent-species-adjusted
prediction did not identify an editable microbial address. Taxon-resolved
carrier analysis prevented four apparently strain-informative biomarkers from
being promoted to conservation, guide-specificity, or laboratory design. The
reproducible attrition framework is the primary contribution; the current data
do not nominate a CRC microbiome-editing target.

## Introduction

Large cross-cohort studies show that stool metagenomes contain reproducible CRC
signals, including strain- and clade-level associations [1]. At the same time,
new CRISPR-associated systems can modify selected members of native microbial
communities [2]. These developments make a translational question increasingly
important: when does a recurring metagenomic biomarker become a credible
address for a precision perturbation experiment?

Prediction and intervention impose different evidence requirements. A
classifier may benefit from a feature that is shared among many organisms,
tracks ecological restructuring, or is merely correlated with disease. An
editing address must additionally identify the intended organism or harmful
clade, recur within that target, avoid protected organisms and human sequence,
and be compatible with a delivery system. UniRef clusters are useful
nonredundant groups of homologous proteins, but their displayed sequence and
taxonomy summarize a cluster rather than proving which organism contributes
the family in a particular metagenome [3]. HUMAnN taxonomic stratification
provides the information needed to test that distinction [4,5].

We therefore built a sequential, failure-tolerant framework that separates
cross-population association, signal beyond an annotated parent species, and
taxon-resolved addressability. Known CRC mechanisms were retained as frozen
benchmarks, while de novo families were selected only within cross-fitted
training populations. Failed gates remained visible and prevented downstream
target-design work. The study asks not whether a gene family predicts CRC, but
whether public human metagenomes support treating it as a direct microbial
address.

## Methods

### Study design and data

We used uniformly processed public stool metagenomes distributed through
curatedMetagenomicData [6]. Ten cohorts contributed 1,339 samples labeled CRC
or control. Species abundances were MetaPhlAn profiles; gene-family abundances
and taxonomic strata were HUMAnN profiles. Dataset and country were treated as
deployment environments rather than exchangeable random folds.

The protocol, thresholds, and claim boundaries are versioned in
`00_study_protocol.md`. Four literature-motivated benchmarks—colibactin,
*Bacteroides fragilis* toxin, fusobacterial adhesion, and secondary bile-acid
conversion—were frozen before de novo candidate outcomes were examined.

### Leakage-safe gene-family discovery

For every country-aware outer holdout, screening used training cohorts only. A
family had to be evaluable in at least three training cohorts, CRC-enriched in
at least 70%, and have median training-cohort AUC of at least 0.55. At most 100
families per fold advanced to held-out evaluation. An internal nomination had
to be selected and evaluable in at least seven outer folds, remain CRC-enriched
in at least 70% of held-out folds, and retain median held-out AUC of at least
0.55. These rules define an internal screening result, not external validation.

### Annotation and parent-species adjustment

Historical representative accessions were resolved through the UniProt REST
API and UniParc archive. Organism names were mapped to the MetaPhlAn species
table only by exact genus-species text; synonyms were not added after outcomes
were observed. For each candidate and outer fold, balanced L2 logistic models
compared the matched parent-species feature alone with parent species plus the
candidate family. The frozen gate required at least seven evaluable folds,
positive AUC gain in at least 70%, and median AUC gain of at least 0.02.

### Taxon-resolved address gate

Only parent-adjustment survivors entered taxon-resolved analysis. For each
cohort, all HUMAnN rows matching the four UniRef90 identifiers were extracted
without materializing a second copy of the approximately three-million-row
feature index. Every nonzero sample-family-taxon contribution was retained.
Carrier abundance was summed across samples and cohorts. A candidate passed
only if one taxon contributed at least 80% of all stratified abundance. The
largest-carrier threshold was label-independent and frozen before this analysis.
Thresholds from 20% through 90% were reported as sensitivity analysis.

### Sequential stopping rule

A family failing parent independence or taxonomic address resolution did not
advance to sequence conservation, nucleotide guide-specificity, delivery, or
laboratory design. This protects against retrofitting an editing proposal to a
disease-associated but taxonomically ambiguous feature. Failed and unresolved
candidates remain in the public atlas.

## Results

### Known CRC mechanisms were directionally recurrent but weak abundance classifiers

Colibactin was CRC-enriched in 9 of 10 evaluable cohorts, but its median
association AUC was 0.541 and it did not pass the frozen AUC threshold.
Fragilysin was CRC-enriched in 6 of 8 evaluable cohorts (median AUC 0.508), and
secondary bile-acid conversion in 7 of 10 (median AUC 0.537). Fusobacterial
adhesion was not evaluable in the frozen gene-family assay. These results do
not negate experimental evidence for the mechanisms; they show that a known
mechanism need not be a strong abundance classifier in stool.

### Four internally recurring families added information beyond annotated parent species

Sixteen of 6,755 families passed the cross-fitted recurrence rules. Thirteen
could be exactly linked to at least one species feature. Four passed the
parent-adjustment gate: `UniRef90_W6P8M8`, `UniRef90_H1B745`,
`UniRef90_R5RC53`, and `UniRef90_R5REJ2`. Median held-out AUC gains were 0.043,
0.039, 0.038, and 0.021, respectively. At this intermediate stage, these
families appeared to contain information beyond simple detection of the
archived representative's species.

### Taxon-resolved carriers rejected all four as direct genomic addresses

The optimized export recovered 26,246 nonzero sample-family-taxon rows from all
ten cohorts. No candidate approached the prespecified 80% dominant-carrier
threshold (Figure 1). Largest-carrier fractions ranged from 13.5% to 42.5%,
with 7–23 detected taxa per family. The archived parent species contributed
0% for `H1B745`, 3.9% for `R5RC53`, 19.6% for `R5REJ2`, and 19.1% for
`W6P8M8`; none was the largest carrier.

The mismatch was biologically consequential. The largest carrier of `H1B745`
was *Faecalibacterium prausnitzii*, not the archived *Clostridium innocuum*
representative. The three other families were distributed across multiple
*Bacteroides* species. Largest carriers were CRC-enriched in only 4–5 of ten
cohorts. Even at a lenient post hoc majority threshold of 50%, zero candidates
would pass. The atlas therefore records four mixed-source rejections, nine
parent-adjustment rejections, and three unresolved exact parent mappings.

## Discussion

The central result is a failed translation, not a failed analysis. A sequence
family can recur across populations and improve held-out discrimination beyond
one annotated species while still being an unsuitable address for editing.
The representative organism attached to a UniRef cluster was particularly
misleading here: it was never the dominant taxonomic source of a surviving
family. Using that label directly would have created a false impression of
strain-level precision.

This result adds a practical safety gate between biomarker discovery and
microbiome engineering. Current editing platforms demonstrate that
species-specific modification in native communities is becoming technically
plausible [2]. That progress increases, rather than decreases, the importance
of verifying that the proposed sequence belongs to the intended target in the
relevant communities. The proposed framework makes this verification explicit
and preserves every rejection reason.

The finding also clarifies the role of negative results. None of the four known
mechanism benchmarks passed the abundance-association gate, yet colibactin and
other mechanisms have experimental evidence outside this dataset [7–10].
Conversely, four de novo families passed two statistical screens but failed
taxonomic interpretation. Association strength, causal function, and editing
addressability are therefore separate dimensions; no weighted readiness score
should allow one to substitute for another.

### Limitations

The analysis uses processed stool metagenomes rather than isolates, assemblies,
or tumor tissue. HUMAnN strata are computational assignments and do not prove
physical gene location. UniRef90 families combine homologs and are not
nucleotide guide sequences. The study did not run conservation or off-target
guide screens because no de novo candidate passed the preceding address gate.
There is no untouched external gene-level confirmation dataset in the present
analysis. The 80% rule is a prespecified triage threshold, not a universal
biological constant; however, no candidate had a majority carrier at 50%.
Finally, causality, delivery feasibility, and biological safety require
laboratory evidence that cannot be inferred from these observational data.

## Conclusion

Across ten CRC cohorts, no recurring UniRef90 gene-family biomarker survived a
sequential path from association to parent-independent signal to taxonomically
resolved address. The result argues against converting database representative
labels directly into microbial editing targets and provides a reproducible
framework for rejecting unsafe or uninterpretable candidates early.

## References

1. Fackelmann G, et al. Pooled analysis of 3,741 stool metagenomes from 18
   cohorts for cross-stage and strain-level reproducible microbial biomarkers
   of colorectal cancer. *Nature Medicine*. 2025.
   https://doi.org/10.1038/s41591-025-03693-9
2. Gelsinger DR, et al. Metagenomic editing of commensal bacteria in vivo using
   CRISPR-associated transposases. *Science*. 2025.
   https://pubmed.ncbi.nlm.nih.gov/41231980/
3. Suzek BE, et al. UniRef clusters: a comprehensive and scalable alternative
   for improving sequence similarity searches. *Bioinformatics*. 2015.
   https://pubmed.ncbi.nlm.nih.gov/25398609/
4. Franzosa EA, et al. Species-level functional profiling of metagenomes and
   metatranscriptomes. *Nature Methods*. 2018.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6235447/
5. Beghini F, et al. Integrating taxonomic, functional, and strain-level
   profiling of diverse microbial communities with bioBakery 3. *eLife*. 2021.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8096432/
6. Pasolli E, et al. Accessible, curated metagenomic data through ExperimentHub.
   *Nature Methods*. 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5862039/
7. Pleguezuelos-Manzano C, et al. Mutational signature in colorectal cancer
   caused by genotoxic pks-positive *Escherichia coli*. *Nature*. 2020.
   https://www.nature.com/articles/s41586-020-2080-8
8. Chung L, et al. Bacteroides fragilis toxin coordinates a pro-carcinogenic
   inflammatory cascade via targeting of colonic epithelial cells. 2018.
   https://pubmed.ncbi.nlm.nih.gov/29398651/
9. Rubinstein MR, et al. Fusobacterium nucleatum promotes colorectal cancer by
   inducing Wnt/beta-catenin modulator Annexin A1. 2017.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5465824/
10. Cao Y, et al. Gut microbiota and metabolites in colorectal cancer: the role
    of secondary bile acids. 2022. https://pubmed.ncbi.nlm.nih.gov/36343662/
