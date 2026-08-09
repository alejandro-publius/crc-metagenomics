-- Open the generated database:
--   sqlite3 data/derived/crc_research.sqlite
-- Then load this file:
--   .read learning/sql/01_cohort_questions.sql

.headers on
.mode column

-- 1. Inspect the ready-made cohort summary view.
SELECT *
FROM cohort_overview
ORDER BY n_samples DESC;

-- 2. Practice WHERE, GROUP BY, and aggregate functions.
SELECT
    country,
    COUNT(*) AS n_samples,
    ROUND(AVG(age), 1) AS mean_age
FROM samples
WHERE study_condition IN ('CRC', 'control')
GROUP BY country
ORDER BY n_samples DESC;

-- Exercise: write a query returning CRC prevalence by cohort.
-- Include study_name, n_crc, n_binary, and crc_prevalence.
