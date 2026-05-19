---
marp: true
theme: default
paginate: true
size: 16:9
footer: 'Velazquez & Selbrede | github.com/alejandro-publius/crc-metagenomics'
---

# Species-only RF beats joint species+pathway models for cross-cohort CRC classification

**Alejandro Velazquez¹, Rachel Selbrede²**

¹[FILL affiliation], ²[FILL affiliation]
3-minute lightning talk | [FILL conference + date]

---

## Problem

- Gut metagenomics can discriminate CRC from controls; do **functional pathway** features add signal beyond **species** profiles?
- Prior multi-cohort studies use naive LODO and lack formal classifier comparison.
- We re-evaluate the Thomas et al. (2019) framework on 10 cohorts (n = 1,522) with **country-aware LODO** and **DeLong testing**.

---

## Headline result

- Species-only RF: pooled LODO AUC **0.781** (95% CI 0.757–0.805; n = 1,339).
- Joint species+pathway RF: 0.756 — **species_rf vs joint_rf: DeLong z = 3.35, p = 0.0008**.
- Joint XGBoost: 0.766 (z = 2.00, p = 0.046).
- Pathways nearly triple feature dimensionality without proportional gain — parsimony wins at this scale.

![bg right:45% fit](../figures/fig1_lodo_auc.png)

---

## Take-homes + code

- **Species-only RF** is the right default for cross-cohort CRC classification at n ~ 1,300.
- **Country-aware LODO** is essential (inflates ThomasAM_2019_c from 0.836 to 0.998 if same-country cohort kept in training).
- Four **oral pathobionts** (*F. nucleatum*, *P. stomatis*, *P. micra*, *G. morbillorum*) top the CRC SHAP rankings; cross-cohort healthy-vs-adenoma is a null result (n = 183 across 4 cohorts is underpowered), consistent with prior literature suggesting the signature is more prominent at the carcinoma stage.

**Code, data, predictions, decision logs:** github.com/alejandro-publius/crc-metagenomics

![bg right:35% w:80%](qr.png)
<!-- generate qr.png with: qrencode -o conference/qr.png $(cat conference/qr_code_target_url.txt | head -1) -->
