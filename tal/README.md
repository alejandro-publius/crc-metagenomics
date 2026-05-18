# Meeting Package — Tal Korem & George Austin

*Alejandro Velazquez and Rachel Selbrede — 2026-05-18*

This directory is the meeting deliverable. Everything in it is current as of post-reconciliation; all numbers traceable to `../results/*.csv`. `scripts/verify_results.py` reports **49/49 checks pass**.

## Recommended reading order (≈10 min total)

1. **`one_pager.md`** *(2 min)* — the single page. Headline negative result on joint species+pathway, adenoma summary, open questions for Tal.
2. **`pitch_2min.md`** *(1 min)* — speaker notes for the live 2-minute pitch. Three beats.
3. **`action_items_response.md`** *(3 min)* — point-by-point against the 7 action items from last meeting. Status, evidence, and honest reasons for what was deferred.
4. **`dashboard.md`** *(3 min)* — cohorts, headline AUC table, adenoma LODO, robustness battery, figure references. Read this if you want the numbers all in one place.
5. **`methodology_addressed.md`** *(read if asked)* — explicit walkthrough of each of Tal's four methodological concerns from last meeting (batch effects, over-parameterization, granular features, biological priors), with what was done, evidence, and the open question we'd like input on.

## Where the numbers live

| Claim | File |
|---|---|
| Cohort N + composition          | `../results/table1.csv` |
| Species RF per-cohort + pooled  | `../results/baseline_results.csv`, `../results/bootstrap_ci.csv` |
| Joint RF / XGB                  | `../results/joint_results.csv`, `../results/bootstrap_ci.csv` |
| DeLong tests                    | `../results/delong_results.csv` |
| Biological shortlist            | `../scripts/bio_pathway_shortlist.py`, `../results/bio_pathway_results.csv`, `../results/bio_pathway_shortlist.txt` |
| Adenoma LODO                    | `../results/adenoma_lodo_results.csv` |
| Adenoma rebalanced (in flight)  | `../results/adenoma_rebalanced_lodo.csv`, `../results/adenoma_rebalanced_summary.csv` |
| Stratified pathway pilot (in flight) | `../results/stratified_pathway_pilot.csv` |
| Sensitivity sweep               | `../results/sensitivity_thresholds.csv` |
| Seed sensitivity                | `../results/seed_sensitivity.csv` |
| Confounder adjustment           | `../results/confounder_results.csv` |
| ComBat                          | `../results/combat_results.csv` |
| SHAP (CRC + adenoma axis)       | `../results/shap_*.csv` |
| Figures                         | `../figures/`, `../manuscript/figures/` |
| Manuscript draft                | `../manuscript/markdown/` (and compiled .docx) |

## Open questions we want input on

(Reproduced from `one_pager.md` for visibility.)

1. **Pathway-quality vs pathway-information diagnostic** — with raw-FASTQ HUMAnN rerun deferred, what surrogate would convince you the joint-feature degradation is information-theoretic, not pathway-quality?
2. **Adenoma stratification** — worth pursuing advanced-vs-non-advanced if metadata supports it?
3. **Submission venue** — Genome Medicine vs Microbiome vs Gut Microbes prior?
