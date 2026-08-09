PRAGMA foreign_keys = ON;

CREATE TABLE samples (
    sample_id TEXT PRIMARY KEY,
    study_name TEXT NOT NULL,
    study_condition TEXT NOT NULL,
    label INTEGER,
    age REAL,
    gender TEXT,
    BMI REAL,
    country TEXT,
    sequencing_platform TEXT,
    number_reads INTEGER
);

CREATE TABLE fold_results (
    model TEXT NOT NULL,
    cohort TEXT NOT NULL,
    auc REAL NOT NULL CHECK (auc >= 0.0 AND auc <= 1.0),
    n_train INTEGER NOT NULL CHECK (n_train > 0),
    n_test INTEGER NOT NULL CHECK (n_test > 0),
    n_features INTEGER,
    PRIMARY KEY (model, cohort)
);

CREATE TABLE predictions (
    model TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    cohort TEXT NOT NULL,
    y_true INTEGER NOT NULL CHECK (y_true IN (0, 1)),
    y_prob REAL NOT NULL CHECK (y_prob >= 0.0 AND y_prob <= 1.0),
    PRIMARY KEY (model, sample_id),
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
);

CREATE TABLE catalog_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX samples_study_idx ON samples(study_name);
CREATE INDEX samples_condition_idx ON samples(study_condition);
CREATE INDEX predictions_cohort_idx ON predictions(cohort);

CREATE VIEW cohort_overview AS
SELECT
    study_name,
    country,
    COUNT(*) AS n_samples,
    SUM(CASE WHEN study_condition = 'CRC' THEN 1 ELSE 0 END) AS n_crc,
    SUM(CASE WHEN study_condition = 'control' THEN 1 ELSE 0 END) AS n_control,
    SUM(CASE WHEN study_condition = 'adenoma' THEN 1 ELSE 0 END) AS n_adenoma,
    ROUND(AVG(age), 1) AS mean_age,
    ROUND(AVG(number_reads), 0) AS mean_reads
FROM samples
GROUP BY study_name, country;

CREATE VIEW model_comparison AS
SELECT
    cohort,
    MAX(CASE WHEN model = 'species_rf' THEN auc END) AS species_rf_auc,
    MAX(CASE WHEN model = 'joint_rf' THEN auc END) AS joint_rf_auc,
    MAX(CASE WHEN model = 'joint_xgb' THEN auc END) AS joint_xgb_auc,
    MAX(CASE WHEN model = 'gene_family_enet' THEN auc END) AS gene_family_enet_auc,
    MAX(CASE WHEN model = 'joint_rf' THEN auc END)
      - MAX(CASE WHEN model = 'species_rf' THEN auc END) AS joint_rf_delta,
    MAX(CASE WHEN model = 'gene_family_enet' THEN auc END)
      - MAX(CASE WHEN model = 'species_rf' THEN auc END) AS gene_family_delta
FROM fold_results
GROUP BY cohort;
