# Elevator Pitch — Three Versions

These are speaker notes, not scripts. Pause where the dashes are. Use your own words.

---

## 30 Seconds

- I re-analyzed about a decade of published gut-microbiome data — around 1,500 people across 10 studies — to test whether a stool-DNA test could screen for colorectal cancer.
- Best model gets to about 78 percent accuracy across cohorts it has never seen.
- Promising, but not yet good enough to replace colonoscopy or current stool tests.
- Two things stood out: adding metabolic pathway data made the model worse, and prior estimates in the literature were inflated by population effects we now know how to correct for.

---

## 60 Seconds

- Colorectal cancer is highly curable when caught early, but screening uptake is low because colonoscopy is invasive and current stool tests are only moderately sensitive.
- For about a decade, people have proposed using the gut microbiome — the bacteria in a stool sample — as a more comfortable alternative.
- I wanted an honest answer: how well does it actually work across populations?
- I pulled 10 studies, 8 countries, around 1,500 people, and trained machine-learning models with strict cross-cohort validation.
- Headline: about 78 percent accuracy. Real signal, but not clinical grade.
- Two findings I think matter for the field. First, when training and test data come from the same country, results are inflated by population-level microbiome differences — I corrected for that and reported the honest number. Second, adding metabolic pathway features, which everyone expected to help, actually hurt the model.
- The same model can partly distinguish pre-cancer from cancer, driven by oral bacteria that colonize tumors — but cannot reliably catch pre-cancer in healthy people. So the early-detection dream is still further off than the headlines suggest.

---

## 2 Minutes

- I'm a Berkeley CS undergrad. This is my first research project. I worked with a biology collaborator, Rachel Selbrede, who led the interpretation side.
- The question we wanted to answer is one that has been bouncing around the literature for almost a decade: can the bacteria in your stool be used to screen for colorectal cancer?
- The motivation is real. Colorectal cancer is highly curable when caught early, but screening uptake is poor. Colonoscopy works but is invasive. The standard stool test — FIT — is comfortable but only moderately sensitive, particularly for pre-cancer.
- The microbiome idea is compelling. Tumors change the colon environment. Certain bacteria — particularly oral-cavity species like *Fusobacterium nucleatum* — appear to thrive in and around tumors. Several papers have reported good classification accuracy.
- I pulled around 1,500 people across 10 published cohorts in 8 countries, all shotgun-sequenced, and ran machine-learning models with leave-one-dataset-out cross-validation. So every test cohort is one the model has never seen.
- The headline: about 78 percent accuracy across cohorts. Real signal, but not yet clinical grade. FIT is already in that range.
- Two methodological contributions I think are worth flagging. The first is what I'm calling country-aware cross-validation. When you hold out one cohort but leave another from the same country in the training set, you accidentally let the model learn geography instead of disease. We saw one Japanese cohort go from near-perfect AUC down to a much more realistic number after we fixed this. I think this should be standard.
- The second is a negative result. We expected adding metabolic pathway features to help. It didn't. The species-only model is better than the joint model. For anyone designing a similar pipeline, that's worth knowing.
- We also looked at pre-cancer. The model can partly tell adenoma from full cancer, driven by oral bacteria colonizing tumors. But it cannot reliably tell adenoma from healthy. So the early-detection dream — which is the whole point of screening — needs more than what current shotgun microbiome data can provide.
- Bottom line: real biology, honest numbers, useful methodological lessons, not yet a clinical test. Code and decision log are public.
