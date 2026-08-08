# Species-aware study-effect correction

This pilot applies each learned species offset to both the species abundance
and every recognized species-resolved pathway assigned to that organism. It
reports two deployment settings separately:

- **source only:** offsets are learned from training studies only; the held-out
  study is never used to fit its transformation
- **target adaptive:** the unlabeled held-out feature distribution is used to
  estimate an additional target offset; cancer labels remain hidden

The earlier ComBat pilot uses the second, transductive setting and is retained
only as an upper-bound robustness check. It is not described as a strictly
inductive evaluation.

## Results

| Representation | Mean LODO AUC | Fold range |
|---|---:|---:|
| Corrected species, source only | 0.814 | 0.688-0.917 |
| Corrected species + stratified pathways, source only | 0.773 | 0.667-0.901 |
| Corrected species + stratified pathways, target adaptive | 0.777 | 0.672-0.891 |

The uncorrected stratified-pathway RF averages 0.771. The source-only change is
therefore approximately +0.002 AUC and the target-adaptive change approximately
+0.006, with heterogeneous fold effects. Study-effect correction does not
rescue the species-resolved functional representation.

Approximately 82% of selected stratified pathway features could be linked to a
parent species retained in the species table. Unmapped pathways remain
unchanged and their coverage is recorded per fold in `correction_audit.csv`.
