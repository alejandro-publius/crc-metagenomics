# LinkedIn Post

---

Sharing my first research paper: a multi-cohort re-analysis of whether the gut microbiome — measured from a small stool sample — can be used to screen for colorectal cancer.

Colorectal cancer is one of the most common and most treatable cancers when caught early, but screening uptake remains low. A stool-DNA test has long been proposed as a more comfortable alternative to colonoscopy. We wanted to know how well such a test could actually work, and how much of the prior optimism survives careful cross-population testing.

Working with data from about 1,500 people across 10 published studies in 8 countries, we trained machine-learning models that look only at which bacterial species are present in stool. The best model distinguishes colorectal cancer patients from healthy controls with roughly 78% accuracy across cohorts it had never seen. Two findings stood out.

First, adding metabolic pathway information to the model made it slightly worse, not better — a useful negative result for anyone designing similar diagnostic pipelines. Second, when training and test data come from the same country, performance is inflated by population-level differences in the microbiome. After controlling for this, the cross-population numbers are noticeably more honest — and noticeably more modest.

We also looked at pre-cancerous adenomas: the model partly separates adenoma from full cancer (driven by oral-cavity bacteria that thrive once tumors form), but cannot reliably separate adenoma from healthy.

The takeaway: microbiome-based stool screening is promising and the underlying biology is real, but AUC 0.78 is not yet a clinical-grade test.

Personal note: I'm a CS undergrad at Berkeley and this is my first paper. Huge thanks to Rachel Selbrede, who carried the biological interpretation work — particularly the oral-pathobiont story — and to the original cohort authors whose open data made the re-analysis possible.

Code, data, and decision logs are public. Comments and critique welcome.

— Alejandro Velazquez
