# Twitter / X Thread

---

**Tweet 1/10**
New preprint. We re-analyzed gut microbiome data from ~1,500 people across 10 studies to ask a simple question: can a stool-DNA test screen for colorectal cancer? Best answer so far: about 78% accuracy. Promising, but not a replacement for colonoscopy. 🧬

---

**Tweet 2/10**
Why this matters: colorectal cancer is highly curable when caught early, but uptake of current screening is low. A more comfortable, more informative stool test could save lives — if it actually works at clinical-grade performance. We wanted an honest answer.

---

**Tweet 3/10**
We pooled 10 published cohorts from 7 countries (curatedMetagenomicData), then trained machine-learning models using leave-one-dataset-out cross-validation. So every test cohort is one the model has truly never seen.
[Figure: figures/fig1_lodo_auc.png]

---

**Tweet 4/10**
Headline number: a Random Forest using only the bacterial species in stool reaches a pooled cross-cohort AUC of 0.78 on 1,339 held-out predictions. Solid. Not yet diagnostic-grade.
[Figure: figures/diagnostics/roc_pr_pooled.png] 📊

---

**Tweet 5/10**
Surprising negative result: adding ~400 metabolic pathway features made the model slightly *worse*, not better. We tried a biologically-curated pathway shortlist too. Same story. Sometimes more data is just more noise.

---

**Tweet 6/10**
Where the signal comes from: the model leans heavily on oral-cavity bacteria that colonize tumors — Fusobacterium nucleatum, Parvimonas micra, Peptostreptococcus stomatis, Gemella morbillorum. These same species also separate adenoma from cancer (AUC 0.67). 🔬
[Figure: figures/fig2_shap_crc.png]

---

**Tweet 7/10**
A methodology fix that matters: when training and test cohorts come from the same country, performance is inflated. One Japanese cohort dropped from AUC 0.998 to 0.836 once we excluded same-country cohorts from training. Country-aware LODO should be standard.

---

**Tweet 8/10**
Honest limits: the model cannot reliably tell pre-cancer (adenoma) from healthy (AUC 0.56). Cohorts are mostly Europe/US/Asia. ~1,500 people is modest. And AUC 0.78 ≠ a clinical test. This is a methodological refinement, not a brand-new biomarker.

---

**Tweet 9/10**
All code, data prep, and decision logs are open. Pipeline is deterministic (single seed, ~45 min on a laptop): https://github.com/[USER]/crc-metagenomics

---

**Tweet 10/10**
Paper preprint: [LINK]. Huge thanks to my collaborator @[Rachel] for the biology side and to the original cohort authors whose open data made the re-analysis possible. First paper out of undergrad — comments welcome.
