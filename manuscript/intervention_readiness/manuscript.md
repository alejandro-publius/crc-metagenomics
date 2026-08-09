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
stratified abundance. Separately, two published colibactin CRISPRi spacers were
audited for exact PAM-compatible sites in a frozen seven-genome pks-positive
reference panel, then screened for conservative PAM-compatible near matches in
ten common gut bacterial references and GRCh38.p14. Before expanded
conservation results were inspected, all 97 human-host entries in a published
pks-positive population table and stricter overall and subgroup gates were
frozen. Rules for reconciling the one absent and one duplicated primary-guide
assembly result were then frozen before examining target-bearing contigs or
checksum-validated source reads.

**Results:** Sixteen of 6,755 families (0.24%) passed the internal recurrence
screen. Four of sixteen added a median held-out AUC of at least 0.02 beyond
their annotated parent-species proxy and improved AUC in at least 70% of
evaluable folds. None passed taxonomic address resolution. Each family was
distributed across 7–23 taxa, and its largest carrier accounted for only
13.5–42.5% of stratified abundance. The species implied by the archived UniRef
representative accounted for 0–19.6% and was never the largest carrier. No
candidate passed even a post hoc 50% majority-carrier sensitivity threshold.
Separately, structured primary-evidence review identified colibactin as the
strongest literature-supported experimental benchmark: it has a human
mutational signature and a reported in vivo delivered CRISPRi perturbation in
a mouse CRC model, although the intervention report remains a preprint and
sequence safety gates are incomplete.

Both in-vivo-tested colibactin spacers retained exactly one PAM-compatible
target site in all seven genomes of a frozen pks-positive reference panel. In
the 97-human-isolate panel, `sgclbB_4387` covered 96 isolates and was unique in
95, while `sgclbC_2313` covered and was unique in all 97; both passed the
predeclared conservation gate. In the protected-reference pilot, primary guide
`sgclbB_4387` had no flagged near matches, whereas secondary guide
`sgclbC_2313` retained five GRCh38 sites with three or four mismatches and an
exact PAM-proximal seed. The primary guide was therefore prioritized as the
cleaner current lead. Source reads resolved the UPEC79 absence as a draft-
assembly omission (30 exact supporting reads). JML024 did not satisfy the
frozen approximately two-copy depth rule and remained an unresolved duplicate.
That unresolved exception and the secondary guide's flags prevent an
experiment-ready claim.

**Conclusions:** Cross-population recurrence and parent-species-adjusted
prediction did not identify an editable microbial address. Taxon-resolved
carrier analysis prevented four apparently strain-informative biomarkers from
being promoted to conservation, guide-specificity, or laboratory design. The
reproducible attrition framework is the primary contribution. The independent
positive-control track prioritizes published `sgclbB_4387` over `sgclbC_2313`
as the cleaner default, while the opposing conservation and specificity
rankings support redesigning a secondary guide. The study does not establish an
experiment-ready target.

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

### Published-guide positive-control audit

The two in-vivo-tested colibactin spacers reported by Hamp et al. were
transcribed from Supplementary Table S4, including an explicit reverse-
complement check [11]. After confirming the expected NC101 on-target site and
before inspecting cross-strain results, we froze a seven-genome pks-positive
panel: the complete NC101 genome plus IHE3034, A192PP, and four independent
pks-positive mouse isolates reported to contain complete pks islands [13,14].
For each spacer and genome, both DNA orientations were searched for an exact
20-nt match with the corresponding SpCas9 NGG PAM. The pilot gate required at
least 80% genome coverage and at least 80% of genomes to contain exactly one
PAM-compatible site. This conservation analysis did not assess near matches or
patient-level pks diversity.

### Expanded human-isolate conservation audit

Before downloading or searching any expanded-panel genome, we froze every
human-host entry in Supplementary Table S1 of Watanabe et al., whose table
reports 109 pks-positive *E. coli* strains [16]. This retained 97 isolates from
Japan: 62 fecal commensals and 35 blood- or urine-derived extraintestinal
clinical isolates. Exact source rows, WGS accessions, retrieval URLs, and the
source-spreadsheet checksum were retained. A guide passed only with at least
90% exact-site coverage, at least 90% unique-site coverage, and at least 80%
coverage within both the fecal-commensal and extraintestinal-clinical groups.
All missing and duplicated sites were preserved. The paper reports that pks
carriage did not guarantee colibactin production, so this audit measures
spacer/PAM conservation rather than functional toxin production.

### Primary-guide assembly-exception reconciliation

After the 97-isolate result, the JML024 duplicate and UPEC79 absence were
registered as the only follow-up cases and interpretation rules were frozen.
The complete target-bearing contigs were compared first. Paired FASTQ files for
DRR102722 and DRR103319 were then retrieved from ENA, validated against the
provider byte counts and MD5 checksums, and streamed without subsampling. Both
read orientations were searched for the exact spacer plus NGG PAM and for
PAM-compatible spacer variants with up to four mismatches. The conserved
secondary guide served as a within-run recovery control. For JML024, twelve
assembly-unique 31-base probes were selected from each of the long primary-
target contig, short primary-target contig, and secondary-target neighborhood.
The duplicate could be called distinct only if both neighborhoods were
supported and combined median probe depth was approximately twice the single-
copy control. Public assembly mappings were also checked for a later or
independent assembly of the same source sample.

### Protected-reference specificity pilot

After the conservation result and before scanning any protected reference, we
froze ten cultured gut bacterial assemblies plus GRCh38.p14. Nine bacterial
taxa ranked among the 21 most prevalent species in healthy controls; a tenth
*Roseburia* reference broadened protected butyrate-associated coverage. Exact
assemblies, legacy-name mappings, prevalence ranks, and download URLs were
retained. A site was flagged when it had a correctly oriented SpCas9 NGG PAM
and either no more than two mismatches across the spacer or an exact
PAM-proximal eight-base seed with no more than four total mismatches. A guide
passed the pilot only with zero flagged sites across all eleven references.
Insertions, deletions, non-NGG PAMs, broader strain diversity, exposure, and
platform-specific effects were outside the frozen screen. Bacterial CRISPRi
can produce effects from shorter seed-only matches under some expression
conditions [15], so this operational rule is not an exhaustive off-target
model.

## Results

### Known CRC mechanisms were directionally recurrent but weak abundance classifiers

Colibactin was CRC-enriched in 9 of 10 evaluable cohorts, but its median
association AUC was 0.541 and it did not pass the frozen AUC threshold.
Fragilysin was CRC-enriched in 6 of 8 evaluable cohorts (median AUC 0.508), and
secondary bile-acid conversion in 7 of 10 (median AUC 0.537). Fusobacterial
adhesion was not evaluable in the frozen gene-family assay. These results do
not negate experimental evidence for the mechanisms; they show that a known
mechanism need not be a strong abundance classifier in stool.

### Structured evidence review separates causal support from delivery readiness

Six primary studies were extracted under a frozen tiering rubric. Colibactin
reached C3 because its experimentally reproduced mutational pattern has been
detected in human cancer genomes. It reached E3 only through a 2025 preprint
reporting conjugative CRISPRi delivery, reduced genotoxicity, and lower
tumorigenesis in mice [11]. Fragilysin, FadA/Fap2, and microbial
7alpha-dehydroxylation each reached C2: isogenic microbial mutants changed a
CRC-relevant animal phenotype or tumor targeting [8–10,12]. They remained E1
because the experiments used constructed strains rather than delivering an
edit to an established native community.

The frozen metagenomic assay did not fully represent these mechanisms.
Thirteen of nineteen prespecified colibactin genes were represented. The single
`bft` effector was represented, while neither `fadA` nor `fap2` was recovered.
Eight of ten prespecified bile-acid genes were represented by six distinct
UniRef90 clusters. These counts describe assay coverage, not gene co-location,
expression, or pathway activity. No known benchmark was labeled
experiment-ready.

### Published colibactin guides passed a small reference-panel conservation pilot

Both reported guides passed the frozen pilot. `sgclbB_4387` and
`sgclbC_2313` each had one and only one exact PAM-compatible site in all seven
genomes (7/7 coverage and 7/7 unique-site coverage for each guide). The sites
were found in the expected NC101 genes and conserved across the six additional
references. The atlas therefore records a reference-panel conservation pass
for colibactin while leaving expanded human-isolate conservation and
specificity open at that stage.

### Both guides passed the expanded human-isolate conservation gate

Primary `sgclbB_4387` had an exact PAM-compatible target in 96/97 isolates
(99.0%) and exactly one site in 95/97 (97.9%). It covered all 62 fecal
commensals and 34/35 extraintestinal clinical isolates. Fecal strain JML024 had
two exact sites in its draft assembly, while urine isolate UPEC79 had none.
Secondary `sgclbC_2313` covered and was unique in all 97 isolates, including
both source groups. Both guides passed the predeclared gate. The exceptions
were retained for frozen source-read reconciliation rather than removed post
hoc (Figure 2A).

### Source reads resolved the UPEC79 absence but not the JML024 duplicate

All four FASTQ files passed the provider size and MD5 checks. The scan retained
1,696,398 JML024 and 930,886 UPEC79 read records. In UPEC79, 30 reads recovered
the exact primary spacer/PAM and 24 recovered the secondary control, satisfying
the prespecified assembly-omission rule. Thus, combining the frozen assembly
screen with its prespecified source-read follow-up supported the primary target
sequence across all 97 isolate records even though it was present in only 96/97
draft assemblies.

JML024 contained 24 exact primary-target reads and 29 secondary-control reads.
The two target-bearing contigs were not redundant under the frozen 500-base,
99.9%-identity rule. Median support was 25.5 reads across long-contig probes,
one read across short-contig probes, and 35 reads across secondary-control
probes. The combined target-to-control depth ratio was 0.757, not approximately
twofold. JML024 therefore remained `unresolved_duplicate`; the analysis does
not claim a biological duplication. Provider mappings exposed only the same
original draft assemblies for both samples, not independent later assemblies.

### Protected-reference screening separated the primary and secondary guides

Primary guide `sgclbB_4387` had no flagged site in any of the ten gut bacterial
references or GRCh38.p14 and passed the frozen pilot. Secondary guide
`sgclbC_2313` had no bacterial flag but retained five GRCh38 sites: two with
three mismatches and three with four mismatches, all preserving the exact
PAM-proximal eight-base seed. It therefore did not pass. This sequence-level
ranking independently favors the primary guide and is directionally consistent
with the preprint's RNA-sequencing result, in which `sgclbC_2313` changed more
genes than `sgclbB_4387` [11] (Figure 2B). The human-reference matches do not
demonstrate editing or exposure of human cells by a bacteria-delivered CRISPRi
system.

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

The constructive next target is therefore not one of the de novo gene-family
addresses. Colibactin is the rational positive-control benchmark for the next
sequence-safety work because its causal and preliminary delivery evidence is
strongest. Its two published guides passed a small reference-panel conservation
pilot and the larger 97-human-isolate gate. Their rankings then diverged. The
secondary `clbC` site was perfectly conserved in the expanded panel but
retained five human-reference flags; the primary `clbB` guide passed the
protected-reference pilot but had one absent and one duplicated target in the
human-isolate assemblies. Source reads subsequently showed that the absent
UPEC79 site was an assembly omission, while the JML024 duplicate remained
unresolved under a frozen depth rule. We therefore prioritize `sgclbB_4387` as
the cleaner current default while retaining one unresolved uniqueness
exception and recommending redesign, rather than reuse, of the secondary
guide. That priority comes from independent experimental literature and frozen
sequence screens, not from forcing weak stool abundance AUC to pass.

### Limitations

The analysis uses processed stool metagenomes rather than isolates, assemblies,
or tumor tissue. HUMAnN strata are computational assignments and do not prove
physical gene location. UniRef90 families combine homologs and are not
nucleotide guide sequences. No de novo candidate advanced to conservation or
off-target analysis because none passed the preceding address gate. The
expanded colibactin panel improves on the original seven genomes but is a
single published Japanese collection dominated by phylogroup B2 and largely
represented by short-read draft assemblies. It does not establish global
pks-positive diversity. Source reads resolved the UPEC79 absence as an assembly
omission, but the JML024 duplicate remains unresolved and could reflect a rare
read, strain mixture, or another assembly limitation rather than a biological
copy. The specificity pilot used one assembly
for each of ten protected bacterial taxa and one human reference; it does not
cover strain or human variation, insertions or deletions, alternative PAMs,
delivery exposure, or platform-specific off-target activity. The five
`sgclbC_2313` human-reference flags are sequence motifs, not evidence that
bacteria-delivered dCas9 reaches or perturbs human cells.
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
It also prioritizes colibactin as a literature-supported positive-control
benchmark and `sgclbB_4387` as its cleaner published guide after frozen
reference, 97-human-isolate, source-read, and protected-reference audits. The
UPEC79 source reads repaired the sole assembly-level absence, but JML024
uniqueness remains unresolved and the more assembly-conserved `sgclbC_2313`
guide retained specificity flags. This tradeoff supports a cleaner primary plus
a redesigned secondary, not a treatment claim. Neither is an approved or fully
validated target.

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
9. Guo P, et al. FadA promotes DNA damage and progression of *Fusobacterium
   nucleatum*-induced colorectal cancer through up-regulation of Chk2. 2020.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7523382/
10. Abed J, et al. Fap2 mediates *Fusobacterium nucleatum* colorectal
    adenocarcinoma enrichment by binding to tumor-expressed Gal-GalNAc. 2016.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC5465824/
11. Hamp B, et al. Programmable conjugative CRISPR interference targeting
    genotoxin in the gut. *bioRxiv*. 2025.
    https://pubmed.ncbi.nlm.nih.gov/41278642/
12. Osswald A, et al. Secondary bile acid production by gut bacteria promotes
    Western diet-associated colorectal cancer. *Gut*. 2025.
    https://pubmed.ncbi.nlm.nih.gov/41412727/
13. Mannion A, et al. Draft genome sequences of five novel polyketide
    synthetase-containing mouse *Escherichia coli* strains. *Genome
    Announcements*. 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC5054322/
14. Lopez LR, et al. A nadA mutation confers nicotinic acid auxotrophy in
    pro-carcinogenic intestinal *Escherichia coli* NC101. *Frontiers in
    Microbiology*. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8207962/
15. Cui L, et al. A CRISPRi screen in *E. coli* reveals sequence-specific
    toxicity of dCas9. *Nature Communications*. 2018.
    https://www.nature.com/articles/s41467-018-04209-5
16. Watanabe H, et al. Insights into the acquisition of the pks island and
    production of colibactin in the *Escherichia coli* population. *Microbial
    Genomics*. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8209727/
