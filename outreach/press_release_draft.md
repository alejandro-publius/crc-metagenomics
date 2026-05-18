# Press Release (Draft)

**FOR IMMEDIATE RELEASE**

---

## Berkeley Undergraduate Re-Analyzes Decade of Gut-Microbiome Cancer Studies, Reports Honest Cross-Population Numbers for Stool-DNA Cancer Screening

**BERKELEY, Calif.** — A new study from a University of California, Berkeley undergraduate computer-scientist and a biology collaborator finds that a stool-DNA test based on gut bacteria can distinguish colorectal cancer patients from healthy people with roughly 78 percent accuracy across populations — promising, but not yet ready to replace existing screening tools such as colonoscopy or fecal immunochemical testing.

The work, posted today, re-analyzes published data from about 1,500 people across 10 studies in 8 countries to settle inconsistent results from prior single-cohort reports and to correct a subtle statistical bias that had inflated earlier estimates.

The authors, Alejandro Velazquez (Berkeley CS undergraduate, primary author) and Rachel Selbrede (biology collaborator), found that the best machine-learning model uses only the bacterial species present in stool — not the more elaborate metabolic-pathway features that have been a focus of prior work. Adding pathway data, somewhat counterintuitively, made the model slightly worse.

A second methodological contribution is what the authors call country-aware cross-validation. In standard practice, models are trained on some cohorts and tested on others. The authors show that when training and test cohorts come from the same country, performance is inflated by population-level differences in the microbiome unrelated to disease. After excluding same-country cohorts from training, one Japanese cohort that had previously scored near-perfect dropped to a more realistic AUC of 0.836. The same correction lowered overall pooled performance to AUC 0.781.

The team also examined whether the model could detect pre-cancerous growths called adenomas — the stage where intervention is most useful. Here the results were mixed: the model could partly distinguish adenoma from full cancer (driven by oral-cavity bacteria that colonize tumors once they form), but could not reliably distinguish adenoma from healthy.

"Our goal was to give an honest answer to a question that has been bouncing around the literature for almost a decade," said Velazquez. "Stool-microbiome screening for colorectal cancer is a real signal, but the headline numbers in the field are higher than what survives careful cross-population testing — and that distinction matters when you're talking about a screening test."

"What I find most compelling biologically is the role of oral bacteria," said Selbrede. "Species like Fusobacterium nucleatum and Parvimonas micra normally live in the mouth, and they appear to colonize colorectal tumors as the disease progresses — they are doing real work in distinguishing later from earlier disease stages."

Colorectal cancer is the second-leading cause of cancer death in the United States, and is highly curable when caught early. Screening rates remain below national targets, in part because patients find colonoscopy invasive and alternative stool tests have only moderate sensitivity.

The study used the curatedMetagenomicData resource from Bioconductor and applied Random Forest and gradient-boosted-tree classifiers. All analysis code, processed data, and a full decision log are available on GitHub. The work received no external funding; it was completed as an independent undergraduate research project.

**About the authors**
Alejandro Velazquez is an undergraduate in Computer Science at UC Berkeley. Rachel Selbrede is a biology collaborator who led the biological interpretation of the findings.

**Data and code availability**
All code and processed data are publicly available at: https://github.com/[USER]/crc-metagenomics

**Media contact**
[Email] | [Phone]

###
