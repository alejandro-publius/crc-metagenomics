# Biological detail does not guarantee portability of colorectal cancer metagenomic classifiers

**Running title:** Portability of CRC metagenomic classifiers

**Authors:** Alejandro Velazquez (affiliation 1; corresponding author) and Rachel Selbrede (affiliation 2)

**Affiliations:** 1. University of California, Berkeley, Berkeley, California, USA; 2. California State University San Marcos, San Marcos, California, USA

**Corresponding author:** Alejandro Velazquez, alejandro-publius@berkeley.edu
**Keywords:** colorectal cancer; gut microbiome; metagenomics; machine learning; external validation; dataset shift; reproducibility

## Abstract

Gut-metagenomic classifiers can distinguish colorectal cancer (CRC) from controls, but apparent accuracy may not survive transfer to a new study. We tested whether greater biological specificity, study-effect correction, or unlabeled target-cohort diagnostics improved portability. We analyzed 1,339 CRC/control stool metagenomes from ten development cohorts using country-aware leave-one-dataset-out validation. We compared species abundance, community and species-resolved pathways, leakage-safe gene-family screening, and a checksum-frozen panel of colibactin, *Bacteroides fragilis* toxin, fusobacterial adhesion, and secondary bile-acid genes. Source-only correction was separated from target-adaptive analysis. Twelve frozen models generated 120 model-by-cohort observations for an outer leave-one-cohort-out test of label-free performance estimation. Finally, a 200-sample age-balanced cohort (PRJNA763023) was frozen before scoring. Species abundance remained the most reliable reference (mean development-cohort AUC, 0.807). Gene families averaged 0.693, frozen mechanisms 0.569, and source-only corrected species-resolved pathways 0.773. The unlabeled risk model was worse than the historical model mean internally (mean absolute error, 0.094 versus 0.062) and overestimated external performance (0.840 versus observed 0.798). The untouched external species model achieved AUC 0.798 (95% bootstrap CI, 0.737 to 0.855) and average precision 0.781. More detailed biological representations did not automatically transfer better, and severe measurable dataset shift did not imply classifier failure. Cross-study microbiome evaluation should prioritize prospective freezing, simple references, and auditable external tests over post hoc complexity.

## Importance

Microbiome models are often made more elaborate in the hope that genes, pathways, or biological mechanisms will travel better between hospitals and populations. We tested that assumption across ten development studies and one untouched external study. The simplest species-abundance model transferred about as well externally as it did during development, while gene- and mechanism-level representations were less stable. A model designed to warn of failure before external labels arrived also failed to improve on a simple historical average, even though it detected extreme differences between datasets. The practical lesson extends beyond colorectal cancer: measuring that a new dataset looks different is not the same as knowing a model will fail. Credible portability claims require decisions frozen before outcomes are examined, label-free processing, transparent negative results, and an external cohort that remains untouched until the end.

## Introduction

Colorectal cancer (CRC) remains a major cause of cancer mortality, and outcomes depend strongly on detection before advanced disease [1]. Stool shotgun metagenomics has repeatedly yielded taxonomic signatures that distinguish CRC from controls across studies [2-4]. These results have encouraged increasingly detailed representations, including community pathways, taxon-resolved functions, gene families, and experimentally motivated microbial mechanisms. Greater detail can improve biological interpretation, but it also increases dimensionality, measurement dependence, and sensitivity to reference databases. Whether detail improves transport to a new study is therefore an empirical question rather than an automatic consequence of biological plausibility.

Cross-study validation is difficult because each cohort bundles population, recruitment, storage, extraction, sequencing, and computational differences. Random sample splits can exploit those differences. Holding out an entire dataset is more realistic, but analytical leakage remains possible if the held-out cohort influences filtering, correction, feature selection, or model choice. A second problem appears after deployment: target labels may take months to accrue, so investigators would like to recognize likely failure using only the new cohort's feature distribution and the model's predictions. Confidence, entropy, prevalence shift, or the ease of separating source from target samples are intuitive warning signals, but few studies test whether these signals predict performance in entirely unseen environments.

We reframed an existing CRC benchmark as a prospective stress test of portability. Uniform development data from curatedMetagenomicData [5] were analyzed under country-aware leave-one-dataset-out (LODO) validation. Species abundance served as the parsimonious reference. We then asked whether community pathways, species-resolved pathways, leakage-safe gene-family selection, a frozen mechanism panel, or source-only correction improved held-out performance. Separately, we tested whether label-free target diagnostics predicted AUC under an outer leave-one-cohort-out design. Before completing that method, we registered a balanced 200-sample external WGS cohort from PRJNA763023 [6], froze its sample and label rules, and committed the scoring procedure. This design allows both favorable and unfavorable external results to be informative without changing the test after seeing its outcome.

## Results

### A portability benchmark across biological resolutions

The development set contained 1,339 case/control metagenomes (674 CRC and 665 controls) from ten cohorts in eight countries. An additional 183 adenoma samples were retained for separate analyses but were not used in the binary CRC/control results reported here. Country-aware LODO removed the test cohort and any development cohort from the same country. The reference species matrix contained 229 MetaPhlAn features after filtering and log transformation.

The species Random Forest averaged 0.807 AUC across held-out development cohorts (range, 0.694 to 0.882). Adding broadly selected community pathways did not provide a stable advantage in the original joint comparisons. A biologically guided community-pathway model averaged 0.817, but its range widened to 0.679 to 0.936 and it did not establish a consistent improvement over species. Source-only species-aware correction produced mean AUC 0.814 for species and 0.773 for species-resolved pathways, compared with 0.771 before correction. An explicitly labeled target-adaptive version averaged 0.777. Thus, correcting each species and its assigned functions using source studies changed the species-resolved result by approximately 0.002 AUC and did not rescue that representation.

Leakage-safe elastic-net models screened UniRef90 gene families within each training fold and retained 5,000 features. Their mean held-out AUC was 0.693 (range, 0.570 to 0.812), substantially more heterogeneous than the species reference. The result argues against using the much larger gene-family search space as a default portability strategy at this sample size.

### Frozen mechanisms were interpretable but not portable classifiers

We specified four mechanism groups without consulting CRC labels: colibactin genotoxicity (`clbA`-`clbS`), *B. fragilis* toxin (`bft`), fusobacterial adhesion (`fadA` and `fap2`), and secondary bile-acid conversion (`bai` genes). Forty-eight protein accessions mapped to 31 UniRef90 clusters; the manifest checksum was frozen before outcome modeling. Mapped fusobacterial adhesion clusters were not detected in the available gene-family representation and remained missing rather than being replaced with correlated features.

Mechanism-only models averaged 0.569 AUC (range, 0.539 to 0.604). Parent species alone averaged 0.656, and adding mechanism scores produced 0.655. These findings do not dispute the biological roles of the selected mechanisms. They show that relative abundance recovered from these cross-study gene-family tables did not form a portable case-control classifier and added no average discrimination beyond the organisms carrying them.

### Unlabeled shift diagnostics did not predict held-out accuracy

Twelve frozen model variants produced 120 model-by-target-cohort observations. For each target, the proposed risk estimator used only information available without outcome labels: target sample size; mean and dispersion of predicted probabilities; confidence, entropy, and extreme-prediction fractions; mean, maximum, and prevalence shifts in the species matrix; and the cross-validated AUC of a classifier distinguishing source from target samples. Evaluation left the entire target cohort out of risk-model training.

The risk estimator had mean absolute error 0.094 and root mean squared error 0.135. A simpler comparator—the historical mean AUC of the same model, calculated without the target cohort—had mean absolute error 0.062 and root mean squared error 0.078. Mean cohort-level absolute error was not lower for the risk estimator (one-sided Wilcoxon P=0.968). Intuitive shift measurements therefore added complexity without improving prospective performance estimation across the ten independent environments.

### The untouched external species model retained discrimination

The external manifest comprised all 200 public WGS runs in PRJNA763023: 50 older-onset CRC, 50 younger-onset CRC, and 50 controls for each age group. The run set, label rule, model, feature transform, and stop rule were committed before scoring. Downloading and reprocessing approximately 2.15 trillion sequenced bases locally was replaced before outcome evaluation by public GMrepo v3 profiles for the exact same run set. GMrepo v3 processed WGS data using MetaPhlAn 4.1.0 with default settings [7,8]. Scientific names were mapped to frozen training features by a documented terminal-name normalization only; no outcome-informed synonyms were added.

All 200 profiles passed GMrepo quality control. Of 229 locked training features, 147 were directly observed; their median summed abundance was 46.0% before renormalization within the locked feature space. The single untouched score achieved AUC 0.798 (95% label-stratified bootstrap CI, 0.737 to 0.855) and average precision 0.781 (95% CI, 0.709 to 0.861). At a fixed probability threshold of 0.5, sensitivity was 0.72 and specificity 0.66; these threshold values are descriptive because the model was not calibrated for clinical use.

Performance was similar across age groups. AUC was 0.784 (95% CI, 0.688 to 0.869) among older participants and 0.822 (95% CI, 0.733 to 0.895) among younger participants. The older-minus-younger AUC difference was -0.038 (95% bootstrap CI, -0.157 to 0.085; two-sided bootstrap P=0.557). This study therefore provides no evidence that transfer differed by the cohort's age stratum.

Before accessing target outcomes, the final label-free risk model estimated external AUC 0.840, whereas the historical species-model mean estimated 0.807. Observed AUC was 0.798. A domain classifier distinguished development from external samples perfectly (AUC 1.0), partly reflecting biological and taxonomy-version differences, yet the CRC classifier retained discrimination. This prospective result illustrates the central negative finding: easily detectable dataset shift is not itself a calibrated measure of performance loss.

## Discussion

This study asked a deployment-facing question rather than conducting another unrestricted biomarker search: which levels of metagenomic evidence remain useful when an entire study is new, and can failure be recognized before outcomes arrive? Across ten development cohorts, increasingly detailed functional and mechanistic representations did not become more portable by virtue of their specificity. Gene families and frozen mechanisms were weaker than the species reference, and source-only correction made only marginal changes to species-resolved pathways. The untouched external species result then showed that a simple reference could retain approximately its development performance despite substantial processing and population shift.

The external AUC of 0.798 should not be interpreted as clinical validation. PRJNA763023 is a retrospective case-control study with equal class balance, whereas screening populations have much lower CRC prevalence. The result measures ranking discrimination, not prospective patient benefit, calibration, comparison with fecal immunochemical testing, or readiness for deployment. Its value is methodological: the cohort was outside model development, its sample set was frozen before scoring, and the result was retained without model selection.

The representation comparison helps separate biological importance from predictive transport. Colibactin, enterotoxigenic *B. fragilis*, fusobacterial adhesion, and bile-acid metabolism remain credible CRC mechanisms. Failure of their abundance scores to classify cases portably may reflect incomplete gene recovery, strain specificity, regulation rather than presence, unmeasured exposures, or effects shared across cases and controls. A mechanistic feature can be causally important yet statistically weak as a population classifier. Conversely, a species signature can classify without proving that the organism causes disease.

The risk-estimation result is equally cautionary. Target-cohort shift was real and extreme: source and external samples were perfectly separable. Yet external discrimination remained close to the internal species mean, and the shift-informed estimator was less accurate than that mean. Domain separability can reflect harmless changes, harmful changes, or changes orthogonal to the decision boundary. A useful warning system will require substantially more independent deployment environments, representation-aware calibration, or task-specific causal assumptions; ten cohorts cannot support a broad claim that generic unlabeled diagnostics forecast failure.

Several limitations remain. First, the development species filter was defined globally in the original pipeline, a mild source of feature-set leakage, although model fitting and country exclusions were held out by cohort. Second, functional representations came from available processed tables and were not uniformly regenerated from raw reads. Third, the GMrepo external taxonomy used MetaPhlAn 4.1.0, while the development resource contains earlier taxonomy labels; direct name matching observed 147 of 229 locked features and required renormalization over a median 46% of abundance. This is both a realistic processing stress test and a source of uncertainty. Fourth, the external cohort came from one country and study and cannot establish geographic universality. Fifth, the age subgroup comparison had 100 samples per stratum and wide confidence intervals. Finally, the label-free risk analysis has only ten independent environments regardless of its 120 model-by-cohort rows.

The strongest next test is prospective multisite evaluation with a versioned profiler, prespecified calibration and thresholds, and metadata sufficient to study stage, tumor location, medications, diet, and screening prevalence. Until then, portability studies should report simple references, leakage boundaries, processing compatibility, and negative results alongside more elaborate models.

In conclusion, biological detail did not guarantee cross-study transport in this CRC metagenomic benchmark. A parsimonious species model retained discrimination in an untouched 200-sample cohort, while gene families, mechanism scores, and source-only pathway correction did not offer consistent gains. Unlabeled shift diagnostics recognized that the external data were different but did not accurately translate that difference into expected AUC. Prospective freezing and independent evaluation contributed more credible evidence than additional modeling complexity.

## Materials and Methods

### Development cohorts and preprocessing

Species, pathway, gene-family, and metadata tables were obtained from curatedMetagenomicData [5]. Ten cohorts were retained after a classifier-blind sequencing-depth and sparsity audit. Samples with fewer than one million reads were excluded, as was HanniganGD_2017 because of substantially lower depth and higher species sparsity. CRC and control labels yielded 1,339 binary samples. Species with prevalence at least 10% and mean abundance at least 1e-4 were retained, row-normalized, and transformed as log10(x+1e-6), producing 229 features.

### Country-aware validation and models

Each development cohort was held out in turn. When Italy or Japan was the target, the other cohort from that country was also excluded from training. Random Forest models used 500 trees, square-root feature sampling, minimum leaf size 5, balanced class weights, and random seed 42. Community pathways were filtered inside each training fold. Species-resolved pathways were linked to parent species when possible. Gene families were screened only within training folds and modeled by elastic net. Discrimination was summarized by AUC calculated on each held-out cohort.

### Mechanism-panel freeze

Mechanism genes were selected from experimental literature without consulting cohort outcomes. UniProt accessions were mapped to UniRef90, duplicates were collapsed, unresolved and undetected genes were retained in the audit trail, and the final 48-accession manifest was checksum-frozen. Mechanism scores were compared with corresponding parent-species abundance under the same country-aware folds.

### Study-effect correction

Species offsets were learned from source studies only and applied consistently to each species and recognized species-resolved pathway assigned to it. A separate target-adaptive analysis used the unlabeled target feature distribution and was labeled transductive. Neither analysis used target outcomes during correction.

### Label-free generalization-risk model

For every frozen model and target cohort, probability-distribution and species-shift features were computed without target outcomes. A ridge regression predicted held-out AUC using model identity and the unlabeled features. Evaluation used an outer leave-one-entire-cohort-out loop. The historical mean AUC for the same model, excluding the target cohort, was the comparator. Error was summarized by mean absolute error, root mean squared error, and cohort-level paired comparison.

### External cohort, public profiles, and frozen scoring

The ENA Portal API was queried for PRJNA763023 WGS runs. Sample aliases defined four groups in the source publication [6]: older and younger CRC (`M_O_`, `M_Y_`) and their controls (`M_HO_`, `M_HY_`). The resulting 200-run manifest was committed before profiling. Public GMrepo v3 species profiles [7] covered the run set exactly and were retrieved through its API. GMrepo v3 used MetaPhlAn 4.1.0 with default settings for WGS data [8]. Profiles were required to pass resource QC and sum to 100% at species level.

Scientific names were normalized by replacing underscores with spaces, removing square brackets, folding case, and collapsing whitespace. Exact normalized terminal names were mapped to the 229 locked features; unmatched features were zero-filled. No synonyms were added. Rows were renormalized over the locked feature space and log transformed as in development. The locked Random Forest was then fit once on all development CRC/control samples and scored once on all 200 external samples.

### Statistical analysis and reproducibility

External AUC and average precision confidence intervals used 10,000 bootstrap replicates, resampling within outcome class for the overall cohort and within outcome-by-age strata for subgroup analyses. The age AUC difference used paired replicate differences. Brier score, sensitivity, and specificity at 0.5 were descriptive secondary metrics. Analyses used Python, pandas, NumPy, scikit-learn [9], SciPy, matplotlib, XGBoost [10], and SHAP [11], with versions pinned in the repository. All derived predictions, bootstrap replicates, mapping coverage, figures, checksums, and decision records are included in the release artifact.

## Data, Metadata, and Code Availability

Development data are publicly accessible through curatedMetagenomicData. External raw reads are available under NCBI/ENA BioProject PRJNA763023. The exact external manifest, GMrepo-derived abundance table, per-sample predictions, uncertainty replicates, code, pinned environments, and provenance checksums are available at <https://github.com/alejandro-publius/crc-metagenomics>. A versioned archival DOI will be added after both authors approve the release candidate.

## Ethics Statement

This secondary analysis used only deidentified public data. No new participants were recruited, no intervention was performed, and no identifiable private information was accessed.

## Author Contributions

**Draft pending coauthor approval.** Alejandro Velazquez: conceptualization, software, formal analysis, visualization, data curation, reproducibility, writing-original draft. Rachel Selbrede: biological interpretation, methodology review, validation, writing-review and editing. Both authors must approve the final wording and contribution statement before submission.

## Funding

**Draft pending coauthor confirmation.** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Conflicts of Interest

**Draft pending coauthor confirmation.** The authors declare no competing interests.

## Acknowledgments and AI Assistance

The authors used OpenAI Codex and Anthropic Claude as tools for code assistance, literature organization, and language editing. Human authors defined the research questions, inspected the analyses and source material, controlled all scientific decisions, and remain accountable for the work. No generative system is listed as an author.

## Figure Legends

**Figure 1. Portability across biological representations.** Points show AUC in each of ten held-out development cohorts; large circles show means and horizontal lines show ranges. The red star shows the untouched external species model with a 95% label-stratified bootstrap confidence interval. More detailed gene and mechanism representations did not consistently improve cross-study discrimination.

**Figure 2. Predicted versus observed target performance.** Each circle is one frozen model evaluated in one held-out development cohort. Panel A estimates target AUC from the model's historical mean without the target cohort. Panel B uses the label-free risk model. The red star is the external species model, whose prediction was generated without external outcome labels. The dashed line indicates perfect estimation.

## References

1. Sung H, Ferlay J, Siegel RL, et al. 2021. Global cancer statistics 2020. CA Cancer J Clin 71:209-249. https://doi.org/10.3322/caac.21660.
2. Thomas AM, Manghi P, Asnicar F, et al. 2019. Metagenomic analysis of colorectal cancer datasets identifies cross-cohort microbial diagnostic signatures and a link with choline degradation. Nat Med 25:667-678. https://doi.org/10.1038/s41591-019-0405-7.
3. Wirbel J, Pyl PT, Kartal E, et al. 2019. Meta-analysis of fecal metagenomes reveals global microbial signatures that are specific for colorectal cancer. Nat Med 25:679-689. https://doi.org/10.1038/s41591-019-0406-6.
4. Yachida S, Mizutani S, Shiroma H, et al. 2019. Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer. Nat Med 25:968-976. https://doi.org/10.1038/s41591-019-0458-7.
5. Pasolli E, Schiffer L, Manghi P, et al. 2017. Accessible, curated metagenomic data through ExperimentHub. Nat Methods 14:1023-1024. https://doi.org/10.1038/nmeth.4468.
6. Yang Y, Du L, Shi D, et al. 2021. Dysbiosis of human gut microbiome in young-onset colorectal cancer. Nat Commun 12:6757. https://doi.org/10.1038/s41467-021-27112-y.
7. Liu C, Wang X, Zhang Z, et al. 2025. GMrepo v3: a curated human gut microbiome database with expanded disease coverage and enhanced cross-dataset biomarker analysis. Nucleic Acids Res. https://doi.org/10.1093/nar/gkaf1190.
8. Blanco-Miguez A, Beghini F, Cumbo F, et al. 2023. Extending and improving metagenomic taxonomic profiling with uncharacterized species using MetaPhlAn 4. Nat Biotechnol 41:1633-1644. https://doi.org/10.1038/s41587-023-01688-w.
9. Pedregosa F, Varoquaux G, Gramfort A, et al. 2011. Scikit-learn: machine learning in Python. J Mach Learn Res 12:2825-2830.
10. Chen T, Guestrin C. 2016. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD 785-794. https://doi.org/10.1145/2939672.2939785.
11. Lundberg SM, Lee SI. 2017. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst 30:4765-4774.
12. Franzosa EA, McIver LJ, Rahnavard G, et al. 2018. Species-level functional profiling of metagenomes and metatranscriptomes. Nat Methods 15:962-968. https://doi.org/10.1038/s41592-018-0176-y.
13. DeLong ER, DeLong DM, Clarke-Pearson DL. 1988. Comparing the areas under two or more correlated receiver operating characteristic curves. Biometrics 44:837-845. https://doi.org/10.2307/2531595.
