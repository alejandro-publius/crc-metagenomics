# Table 1 Reconciliation

Date: 2026-05-18
Source of truth: `data/processed/metadata_clean.csv` (1522 samples, 10 cohorts)
Target under review: `results/table1.csv`
Generator script: `scripts/generate_table1.py`

## Summary

Table 1 was already accurate. All 99 reported cells in `results/table1.csv`
match the values recomputed directly from `data/processed/metadata_clean.csv`
to the published precision (one decimal place). No cell differs by more than
the rounding tolerance (> 1 unit or > 0.5%). The current `results/table1.csv`
is byte-identical to the output produced by re-running
`python3 scripts/generate_table1.py`.

No regeneration was required. `results/table1_PRIOR.csv` was therefore not
created (no prior version to preserve, since the file was not modified).

## Per-cell match status (recomputed vs reported)

Format below: `recomputed | reported -> match`. All matches are exact at one
decimal of precision.

### N (total) / N (CRC) / N (adenoma) / N (control)

| Cohort | N (total) | N (CRC) | N (adenoma) | N (control) | Match |
|---|---|---|---|---|---|
| FengQ_2015      | 154 \| 154   | 46 \| 46   | 47 \| 47   | 61 \| 61   | OK |
| GuptaA_2019     | 60 \| 60     | 30 \| 30   | 0 \| 0     | 30 \| 30   | OK |
| ThomasAM_2018a  | 80 \| 80     | 29 \| 29   | 27 \| 27   | 24 \| 24   | OK |
| ThomasAM_2018b  | 60 \| 60     | 32 \| 32   | 0 \| 0     | 28 \| 28   | OK |
| ThomasAM_2019_c | 80 \| 80     | 40 \| 40   | 0 \| 0     | 40 \| 40   | OK |
| VogtmannE_2016  | 104 \| 104   | 52 \| 52   | 0 \| 0     | 52 \| 52   | OK |
| WirbelJ_2018    | 125 \| 125   | 60 \| 60   | 0 \| 0     | 65 \| 65   | OK |
| YachidaS_2019   | 575 \| 575   | 258 \| 258 | 67 \| 67   | 250 \| 250 | OK |
| YuJ_2015        | 128 \| 128   | 74 \| 74   | 0 \| 0     | 54 \| 54   | OK |
| ZellerG_2014    | 156 \| 156   | 53 \| 53   | 42 \| 42   | 61 \| 61   | OK |
| TOTAL           | 1522 \| 1522 | 674 \| 674 | 183 \| 183 | 665 \| 665 | OK |

### Age (mean +/- SD), Female %, BMI (mean +/- SD), Country

| Cohort | Country | Age | Female % | BMI | Match |
|---|---|---|---|---|---|
| FengQ_2015      | AUT \| AUT | 66.9 +/- 8.4 \| 66.9 +/- 8.4   | 43.5% \| 43.5% | 27.4 +/- 4.0 \| 27.4 +/- 4.0 | OK |
| GuptaA_2019     | IND \| IND | 50.6 +/- 16.0 \| 50.6 +/- 16.0 | 50.0% \| 50.0% | 21.3 +/- 3.0 \| 21.3 +/- 3.0 | OK |
| ThomasAM_2018a  | ITA \| ITA | 67.5 +/- 8.7 \| 67.5 +/- 8.7   | 35.0% \| 35.0% | 25.5 +/- 3.9 \| 25.5 +/- 3.9 | OK |
| ThomasAM_2018b  | ITA \| ITA | 58.2 +/- 8.3 \| 58.2 +/- 8.3   | 35.0% \| 35.0% | 25.8 +/- 4.2 \| 25.8 +/- 4.2 | OK |
| ThomasAM_2019_c | JPN \| JPN | 61.1 +/- 12.6 \| 61.1 +/- 12.6 | 43.8% \| 43.8% | 22.7 +/- 2.6 \| 22.7 +/- 2.6 | OK |
| VogtmannE_2016  | USA \| USA | 61.5 +/- 12.3 \| 61.5 +/- 12.3 | 28.8% \| 28.8% | 25.1 +/- 4.2 \| 25.1 +/- 4.2 | OK |
| WirbelJ_2018    | DEU \| DEU | 59.6 +/- 12.9 \| 59.6 +/- 12.9 | 41.6% \| 41.6% | 25.5 +/- 3.7 \| 25.5 +/- 3.7 | OK |
| YachidaS_2019   | JPN \| JPN | 62.0 +/- 11.0 \| 62.0 +/- 11.0 | 40.2% \| 40.2% | 22.9 +/- 3.4 \| 22.9 +/- 3.4 | OK |
| YuJ_2015        | CHN \| CHN | 64.2 +/- 9.1 \| 64.2 +/- 9.1   | 36.7% \| 36.7% | 23.8 +/- 3.1 \| 23.8 +/- 3.1 | OK |
| ZellerG_2014    | FRA \| FRA | 63.3 +/- 10.9 \| 63.3 +/- 10.9 | 44.2% \| 44.2% | 25.3 +/- 4.2 \| 25.3 +/- 4.2 | OK |
| TOTAL           | em-dash \| em-dash | 62.2 +/- 11.4 \| 62.2 +/- 11.4 | 40.1% \| 40.1% | 24.2 +/- 4.0 \| 24.2 +/- 4.0 | OK |

All 99 cells match exactly. No discrepancies > 1 unit or > 0.5%.

## Cross-check against generator script

Ran `python3 scripts/generate_table1.py`. The script regenerated
`results/table1.csv` with content byte-identical to the prior file. The
recomputed values in this reconciliation match both the file and the script
output. The script and the file are consistent.

## Total counts cross-check

| Quantity | Reported (table1 TOTAL row) | Metadata-derived | Match |
|---|---|---|---|
| N (total)   | 1522 | 1522 | OK |
| N (CRC)     | 674  | 674  | OK |
| N (control) | 665  | 665  | OK |
| N (adenoma) | 183  | 183  | OK |

The headline counts (CRC=674, control=665, adenoma=183, N=1522) are all
confirmed against the metadata.

## Adenoma counts file cross-check

`results/adenoma_counts_per_cohort.csv` reports:

| Cohort         | n_adenoma (file) | n_adenoma (metadata) | Match |
|---|---|---|---|
| YachidaS_2019  | 67 | 67 | OK |
| FengQ_2015     | 47 | 47 | OK |
| ZellerG_2014   | 42 | 42 | OK |
| ThomasAM_2018a | 27 | 27 | OK |
| Sum            | 183 | 183 | OK |

The four cohorts that contribute adenoma samples are the only ones with
non-zero adenoma counts in the metadata, and their per-cohort totals match
exactly. Total adenoma count of 183 in this file equals the metadata adenoma
count and the Table 1 TOTAL row.

## Notes on column scope (structural gap, not a cell discrepancy)

The reconciliation procedure listed two additional columns ("Sequencing
platform" and "Median depth in Mreads") that are not present in either the
current `results/table1.csv` or the current `scripts/generate_table1.py`
output. These are not cell-level discrepancies (no cell differs); they are a
scope-of-columns observation.

For reference, the metadata-derived per-cohort values for these two columns
are recorded below so they are available for any future expansion of Table 1:

| Cohort          | Sequencing platform | Median depth (Mreads) |
|---|---|---|
| FengQ_2015      | IlluminaHiSeq   | 53.8 |
| GuptaA_2019     | IlluminaNextSeq | 8.7  |
| ThomasAM_2018a  | IlluminaHiSeq   | 83.4 |
| ThomasAM_2018b  | IlluminaHiSeq   | 39.4 |
| ThomasAM_2019_c | IlluminaHiSeq   | 42.5 |
| VogtmannE_2016  | IlluminaHiSeq   | 66.6 |
| WirbelJ_2018    | IlluminaHiSeq   | 46.7 |
| YachidaS_2019   | IlluminaHiSeq   | 43.7 |
| YuJ_2015        | IlluminaHiSeq   | 58.5 |
| ZellerG_2014    | IlluminaHiSeq   | 58.7 |
| TOTAL (pooled)  | IlluminaHiSeq (1462) / IlluminaNextSeq (60) | 49.1 |

These were not added to `results/table1.csv` in this reconciliation because
(a) doing so would break the cross-check that the file matches the output of
`scripts/generate_table1.py`, and (b) the reconciliation constraint forbids
touching files other than `results/table1.csv` and this report
(`scripts/generate_table1.py` would need to be updated to keep the file and
its generator in sync). The data is recorded here for whoever decides to
expand Table 1's column set.

## Action taken

- Recomputed all per-cohort and overall Table 1 statistics from
  `data/processed/metadata_clean.csv`.
- Compared every cell (10 cohorts x 9 columns + 1 TOTAL row x 9 columns =
  99 cells) against the values reported in `results/table1.csv`.
- All 99 cells matched within rounding tolerance (in fact, exactly at one
  decimal of precision).
- Ran `scripts/generate_table1.py`; the regenerated `results/table1.csv` is
  byte-identical to the prior file.
- No edits were made to `results/table1.csv`.
- No `results/table1_PRIOR.csv` was created (file was unchanged).
- Cross-checked adenoma per-cohort counts and overall CRC/control/adenoma
  totals; all consistent.

## Conclusion

Table 1 is accurate. The 9 columns currently present all reconcile exactly
against `data/processed/metadata_clean.csv`. The headline counts and the
per-cohort adenoma counts file are also consistent with the metadata.
