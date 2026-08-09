# Learning track

This directory adds practical SQL and workflow skills using the real research
project. It is intentionally separate from the scientific manuscript and does
not change any analytical claim.

## SQL research catalog

Build the derived SQLite database:

```bash
python3 scripts/build_research_catalog.py
sqlite3 data/derived/crc_research.sqlite
```

Inside SQLite, run:

```sql
.read learning/sql/01_cohort_questions.sql
.read learning/sql/02_model_questions.sql
```

The exercises cover filtering, grouping, joins, views, constraints, indexes,
conditional aggregation, and window functions. Because the database is built
from committed CSVs, every answer corresponds to a real cohort or model result.

## Snakemake

The project workflow in `workflow/Snakefile` rebuilds the SQL catalog and runs
the repository's sanity checks, numerical verifier, and fast tests:

```bash
python3 -m pip install -r requirements-workflow.txt
snakemake --snakefile workflow/Snakefile --cores 1
```

Useful learning commands:

```bash
snakemake --snakefile workflow/Snakefile --dag
snakemake --snakefile workflow/Snakefile --dry-run
snakemake --snakefile workflow/Snakefile --summary
```

The next extension should add a rule only when a scientific analysis has clear
inputs, outputs, and an acceptance check.

## Suggested four-session path

1. **Cohorts in SQL:** build the catalog, run `01_cohort_questions.sql`, and
   explain which cohorts have the largest class imbalance.
2. **Models in SQL:** run `02_model_questions.sql`, complete its exercises, and
   explain why a model's mean AUC can hide cohort-specific failure.
3. **Pipeline mechanics:** run the Snakemake dry-run and DAG commands, then
   identify which rule rebuilds after one input CSV changes.
4. **Research extension:** inspect `results/gene_family_manifests/`, compare
   `gene_family_enet` with `species_rf` in SQL, and write a short interpretation
   that distinguishes a benchmark result from a biological claim.

Completion means being able to rebuild the catalog and workflow, write the
comparison queries without copying them, and explain the leakage protection in
the gene-family manifests. It does not require implementing machine-learning
algorithms from scratch.
