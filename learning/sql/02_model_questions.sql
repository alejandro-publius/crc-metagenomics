.headers on
.mode column

-- 1. Compare the four committed models in each held-out cohort.
SELECT *
FROM model_comparison
ORDER BY joint_rf_delta DESC;

-- 2. Join predictions to sample metadata.
SELECT
    p.model,
    s.country,
    COUNT(*) AS n,
    ROUND(AVG(ABS(p.y_true - p.y_prob)), 3) AS mean_absolute_error
FROM predictions AS p
JOIN samples AS s USING (sample_id)
GROUP BY p.model, s.country
ORDER BY p.model, mean_absolute_error DESC;

-- 3. Window functions: rank held-out cohorts within each model by AUC.
SELECT
    model,
    cohort,
    ROUND(auc, 3) AS auc,
    RANK() OVER (PARTITION BY model ORDER BY auc DESC) AS auc_rank
FROM fold_results
ORDER BY model, auc_rank;

-- Exercise: identify cohorts where joint RF underperformed species RF by more
-- than 0.02, and join their sample counts from cohort_overview.

-- Exercise: find the range and standard deviation of gene-family AUCs, then
-- identify the cohorts where gene_family_enet lost more than 0.10 AUC relative
-- to species_rf. What does that suggest about cross-cohort transfer?
