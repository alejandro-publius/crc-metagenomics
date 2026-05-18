# Clinical Translation Summary — What the AUC Numbers Actually Mean

**Headline:** At US population CRC prevalence (~5%), the species-only random
forest achieves **PPV = 11.4%** and **NPV = 98.1%** at the Youden-J operating
point (pooled sensitivity 73.9%, pooled specificity 69.8%; pooled LODO
n = 1339). The pooled AUC of 0.781 (95% CI 0.757–0.805) is genuinely above
chance, but PPV is dominated by base rate — the apparently strong
balanced-class operating point yields roughly **9 false positives for every
true positive** at population prevalence. See
`results/diagnostics/base_rate_ppv.csv`.

## 1. Sensitivity at fixed specificity

Two specificity floors are conventional for non-invasive CRC screening: 0.90
(usable) and 0.95 (FIT-territory). At each floor we report sensitivity, the
base-rate-adjusted PPV / NPV at 5% prevalence, and the implied number needed
to test (NNT = 1 / PPV at 5%). Source: `results/diagnostics/sens_at_fixed_spec.csv`.

| Model      | Spec floor | Achieved spec | Sensitivity | PPV (5%) | NPV (5%) | NNT |
|------------|:----------:|:-------------:|:-----------:|:--------:|:--------:|:---:|
| Species RF | 0.90       | 0.9008        | **0.4985**  | 0.2091   | 0.9715   | 4.78 |
| Species RF | 0.95       | 0.9504        | **0.3976**  | 0.2966   | 0.9677   | 3.37 |
| Joint RF   | 0.90       | 0.9023        | 0.4050      | 0.1790   | 0.9665   | 5.59 |
| Joint RF   | 0.95       | 0.9504        | 0.3101      | 0.2475   | 0.9632   | 4.04 |
| Joint XGB  | 0.90       | 0.9008        | 0.4243      | 0.1837   | 0.9675   | 5.44 |
| Joint XGB  | 0.95       | 0.9504        | 0.3323      | 0.2606   | 0.9643   | 3.84 |

The species RF is the strongest of the three at both floors. None of the
three models meets the implicit deployment requirement of "high sensitivity
AND high specificity" simultaneously; the AUC of ~0.78 is consistent with
roughly half of CRC cases being caught at the 90% specificity floor and
roughly 40% at the 95% floor.

## 2. Base-rate context (PPV / NPV across the prevalence sweep)

Because sensitivity and specificity are prevalence-invariant but PPV is not,
the pooled (~50/50) class balance overstates apparent positive-class
precision. At the Youden-J operating point, holding sens / spec fixed:

| Prevalence | PPV   | NPV   |
|:----------:|:-----:|:-----:|
| 0.5%       | 0.012 | 0.998 |
| 5%         | **0.114** | **0.981** |
| 9.5%       | 0.204 | 0.962 |
| 23%        | 0.422 | 0.900 |
| 50% (pooled) | 0.710 | 0.728 |

Full sweep: `results/diagnostics/base_rate_ppv.csv` and
`figures/diagnostics/base_rate_ppv.png`. The take-away is that the
single-cell PPV in `results/diagnostics/per_cohort_operating_chars.csv`
(0.712 pooled) is a function of the artificial 1:1 case-control design and
should not be reported as a deployment-time figure.

## 3. Where does this sit relative to FIT?

Single-test FIT in an average-risk asymptomatic screening cohort
(Imperiale et al., 2014, *NEJM* 370:1287) yields sensitivity ~0.79 and
specificity ~0.94 for CRC. At a matched specificity floor of 0.94, our
species RF reaches sensitivity ~0.42 — a **~37 percentage-point gap**
versus FIT (full table: `results/diagnostics/fit_vs_microbiome.csv`). The
gap is also present in PPV at 5% prevalence (FIT 0.41 vs species RF 0.28
at matched specificity) and is consistent with the AUC differential
implied by FIT's published operating characteristics.

The honest framing for the manuscript Discussion is therefore: the current
species-only microbiome model **does not match FIT** as a stand-alone primary
screen. Where it could still have value is as a complementary tool — for
example, as a stratifier of FIT-negative patients (FIT's high NPV at 5%
prevalence is ~0.99, but a 0.01 residual risk in a screening-eligible
population is large in absolute terms), or as a triage prior to colonoscopy
in symptomatic patients where the pre-test probability is far higher than
5% and PPV correspondingly rises. We are explicit that the present work
does not show the microbiome to be FIT-competitive at FIT's own
specificity floor.

## 4. Number-needed-to-test framing (lay / policy audience)

At 5% population CRC prevalence, deploying the species RF at the 90%
specificity floor would require **~5 tests per true positive identified**
(NNT = 4.78). At the 95% floor, **~3 tests per true positive** (NNT = 3.37) —
better precision, but only ~40% of cases caught. FIT, at its published
operating point, has an NNT of roughly 2.4 at the same prevalence
(`fit_vs_microbiome.csv`, FIT CRC row: 1 / 0.4093 = 2.44). All three
microbiome models exceed FIT's NNT at every operating point we tested.

## Files referenced

- `results/diagnostics/base_rate_ppv.csv`, `figures/diagnostics/base_rate_ppv.png`
- `results/diagnostics/sens_at_fixed_spec.csv`, `figures/diagnostics/sens_at_fixed_spec.png`
- `results/diagnostics/fit_vs_microbiome.csv`
- `results/diagnostics/per_cohort_operating_chars.csv` (existing, for the
  Youden-J operating characteristics by cohort)

## How to regenerate

```bash
python3 scripts/diagnostics/base_rate_ppv.py
python3 scripts/diagnostics/sens_at_fixed_specificity.py
python3 scripts/diagnostics/fit_comparison.py
```

All three scripts are standalone and idempotent. They depend only on the
three `results/preds_*.csv` files already produced by the LODO training
pipeline.
