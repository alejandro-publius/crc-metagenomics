# What I Learned Re-Analyzing a Decade of Gut-Microbiome Cancer Studies

*By Alejandro Velazquez*

This is the story of my first research project. It started as a homework-adjacent curiosity — could I reproduce a published microbiome-and-cancer machine-learning result? — and ended as a small but honest contribution to a field I had no business being in a year ago. Here is what I found, what surprised me, and what I think it means.

## Why colorectal cancer, and why the microbiome

Colorectal cancer (CRC) is one of the most common cancers in the world, and it is also one of the most curable when caught early. The problem is screening. Colonoscopy works but is invasive, expensive, and unpopular. Stool-based tests like FIT are more comfortable but only moderately sensitive, particularly for the pre-cancerous lesions (called adenomas) where intervention is most useful.

For about a decade, researchers have proposed a third option: read the DNA of the bacteria living in a small stool sample, and use machine learning to look for a "cancer signature" in the microbial community. The biological intuition is real. Tumors in the colon change the local environment, and some bacteria — particularly oral-cavity species like *Fusobacterium nucleatum* — appear to thrive in and around CRC tumors. Several studies have reported encouraging classification accuracies on the order of AUC 0.8 or higher.

So why hadn't I heard of a stool-microbiome CRC test in any clinic? That was the question I wanted to answer.

## The original plan: reproduce, then extend

The starting point was a well-known 2019 meta-analysis by Thomas et al., which pooled several CRC cohorts and trained Random Forest classifiers. My first goal was modest: redo their analysis on more recent data and see whether the numbers hold up.

The first surprise was how much had changed since that paper. The Bioconductor resource `curatedMetagenomicData` had grown from 7 CRC cohorts to 10 (and one I eventually excluded for being sequenced too shallowly to be comparable). More cohorts means more population diversity — good for honest evaluation, harder for the model.

I pulled 1,522 unique subjects across 10 cohorts in 8 countries: 674 CRC patients, 183 with adenomas, 665 healthy controls. From this I extracted 229 species-level taxonomic features (after light prevalence and abundance filtering) and 551 functional pathway features.

Then I ran the standard leave-one-dataset-out (LODO) cross-validation: train on 9 cohorts, test on the 10th, rotate.

## The country-aware LODO discovery — the part I'm proudest of

The first round of results looked great. A species-only Random Forest hit a mean per-cohort AUC north of 0.85, and one specific cohort — ThomasAM_2019_c, a Japanese cohort — came in at AUC 0.998. Near perfect.

Anything that looks near-perfect on real biological data should make you nervous.

It took me a while to spot the problem. ThomasAM_2019_c is from Japan. So is YachidaS_2019. Standard LODO holds out *one cohort* at a time — meaning when ThomasAM_2019_c was the test set, YachidaS_2019 was still in the training set. The model wasn't learning "CRC in Japan." It was learning "Japan," because the Japanese gut microbiome differs systematically from European and American ones at the population level, and CRC happened to be present in the test cohort.

The fix is what I now call country-aware LODO: when you hold out a cohort, you also hold out every other cohort from the same country. After this correction:

- ThomasAM_2019_c dropped from AUC 0.998 to 0.836.
- Pooled cross-cohort AUC dropped to 0.781 — a more honest, and more modest, headline.

This sounds like a small fix, but I think it should be standard. Microbiome meta-analyses are awash in geographic structure, and any cross-cohort claim that doesn't account for it is at risk of confusing "where you live" with "are you sick."

## The negative result on pathways

The other thing I expected to find was that adding functional pathway features — what the bacteria are *doing* metabolically — would improve the model. The biology is compelling: short-chain fatty acid producers, polyamine producers, sulfur metabolism, LPS biosynthesis, all have plausible mechanistic links to colon biology.

So I built joint species-plus-pathway models, with both Random Forest and XGBoost. I tried per-fold filtering. I tried a biologically-curated 84-pathway shortlist hand-selected for CRC-relevant biology. I tried ComBat batch correction. I tried residualizing on age, sex, and BMI.

The joint models were consistently a little worse than the species-only model. Pooled AUC 0.756 (joint RF) vs 0.781 (species RF). Even the biologically-curated shortlist barely matched the species baseline.

I want to be careful about how I read this. It does not mean pathway-level information is useless in microbiome biology — pathways are still where mechanism lives. What it means is that, for this particular classification task with this many samples, adding pathway features hurts more than it helps. Probably because pathways are noisier per feature, more correlated with each other than species are, and quantified from short reads with all the upstream uncertainty that implies. The model already gets most of what it needs from "who's there."

For anyone designing a similar pipeline, this is a useful negative result: don't assume more features is better.

## The adenoma stage — where biology gets interesting

Detection is one thing. The dream is *early* detection, ideally at the adenoma stage, before cancer has formed.

The numbers here are sobering. The model can partly tell adenoma from full cancer (AUC about 0.67), but it cannot reliably tell adenoma from healthy (AUC about 0.56 — barely better than a coin flip).

This is where Rachel's biological reading of the SHAP values was crucial. The features driving the adenoma-vs-CRC separation are almost all oral-cavity bacteria: *Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella morbillorum*, *Fusobacterium nucleatum*, *Solobacterium moorei*. These species normally live in the mouth. They are showing up in the colon — and specifically in CRC samples — because the tumor microenvironment becomes hospitable to them after malignant transformation.

In other words: the microbiome signal we are picking up isn't really an "early warning." It is a sign that the colon has already changed enough to host species that don't usually live there. That is biologically interesting and clinically frustrating in the same breath. If we want a screening test that catches pre-cancer, the microbiome alone — at least as measured by current shotgun sequencing — probably isn't enough.

## What this means for the clinic

Let me say the honest thing out loud: AUC 0.78 is not a clinical-grade test. FIT has comparable or better operating characteristics, costs less, and is already integrated into screening guidelines. A microbiome-based stool test would need to either match or beat FIT's sensitivity and specificity *and* offer something FIT cannot — for example, real adenoma-stage detection, or risk stratification for who needs colonoscopy.

Our results suggest that "as currently formulated," shotgun microbiome screening offers neither. The signal is real. The signal is reproducible across populations once you control for geography. But the signal — as measured today — does not appear to clear the bar that current screening already sets.

This does not mean the line of research is dead. It means the next move is not "scale up the same model on more samples." It is probably:

- Combine microbiome features with host markers (occult blood, methylated DNA in stool, immune signatures).
- Look at longitudinal sampling rather than cross-sectional.
- Push deeper than species — strain-level variation, mobile elements, or direct quantitation of specific oral pathobionts.
- Take adenoma-vs-healthy seriously as a harder, more clinically valuable target.

## What I'd do differently

A few honest reflections.

I spent a lot of time chasing performance before I understood the data. The country-aware fix would have been obvious from cohort metadata if I had looked harder, sooner. Lesson: stare at the data before you stare at the model. The first month of this project was spent tuning hyperparameters on numbers that were structurally inflated. None of that work survived.

I also under-estimated how much "decision log discipline" matters. Every analytical choice — which filter threshold, which random seed, which transform, which cohort to exclude on quality grounds — could shift a number by a couple of points. Writing those choices down explicitly, with the rationale, is how you keep yourself honest months later when you have forgotten why you did something. The decisions addendum in our repo is the document I am most quietly proud of; it is also the document I most wish I had started writing on day one instead of day sixty.

The third lesson is about negative results. I almost did not write up the pathway story, because it felt anticlimactic. On reflection, "a thing everyone expected to work, didn't" is exactly the kind of finding the field needs more of. If the only papers that get written are the ones where the experiment worked, we collectively waste a lot of time re-running experiments that have already silently failed.

## Acknowledgments and the meta-story

I'm a Berkeley CS undergrad. This is my first paper. I went into this project assuming the hard part would be the machine learning, and the hard part turned out to be every other part — the data acquisition, the biological interpretation, the careful skepticism about my own results.

I owe a large share of whatever is good here to Rachel Selbrede, who carried the biological interpretation, particularly the oral-pathobiont story, and to the authors of the original cohort papers, who made their data openly available. None of this work would exist without `curatedMetagenomicData`.

The code, the processed data, the figures, and the full decision log are all public. If you find a mistake, please tell me. If you build on it, please tell me even more loudly.

— Alejandro Velazquez
