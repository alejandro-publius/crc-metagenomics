"""Quality filters and feature preprocessing for the 10-cohort dataset.

Reads raw species and metadata from `data/raw/` and writes the cleaned
modelling-ready files to `data/processed/`. Steps:

1. Per-sample depth filter: drop samples with `<1,000,000` reads.
2. Per-cohort exclusion: drop HanniganGD_2017 a priori (see
   `EXCLUDE_COHORTS` and `results/decisions_addendum.md`).
3. Align species and metadata on `sample_id`.
4. Filter species features: keep columns with prevalence >=10% AND mean
   abundance >=1e-4.
5. Row-sum renormalisation followed by `log10(x + 1e-6)`. The
   renormalisation policy is controlled by `--data-type`:

   - ``relab`` (relative abundance, e.g. MetaPhlAn fractions or
     percentages): always divide by the per-row sum, then log.
   - ``rpk`` (HUMAnN-style reads-per-kilobase; row sums in thousands):
     do NOT divide by the row sum -- absolute magnitudes carry signal.
   - ``auto`` (default for backward compatibility): use the legacy
     heuristic ``if rs.mean() > 1.5: divide``. In ``auto`` mode we also
     ASSERT that no row sum exceeds 100 -- if it does, the input looks
     like RPK / counts and silently dividing would destroy signal, so
     we fail loudly and tell the user to pass ``--data-type`` explicitly.

6. Map `study_condition` -> `label` (`CRC`->1, `control`->0,
   `adenoma`->-1; adenoma rows are excluded from binary LODO via the
   downstream `label.isin([0, 1])` mask).

Outputs:
- `data/processed/species_filtered.csv`
- `data/processed/metadata_clean.csv`
"""
import argparse
import os

import numpy as np
import pandas as pd

# ── Quality filters (documented, applied before any modeling) ─────────────────
# Per-sample: minimum reads for reliable MetaPhlAn species profiling.
MIN_READS = 1_000_000

# Per-cohort: HanniganGD_2017 excluded due to substantially lower sequencing
# depth (mean 6.5M reads, range 17K–21M) compared to all other cohorts
# (mean 40–102M reads), and consequent high feature sparsity (82% zero-valued
# species features vs. 61% mean across other cohorts). Both metrics were
# assessed before model training; the exclusion is pre-specified and
# independent of classification results.
EXCLUDE_COHORTS = ['HanniganGD_2017']

# Sanity bound for ``auto`` mode: any row sum above this strongly implies
# the input is NOT a relative-abundance table (relab maxes at 1 or 100).
AUTO_MAX_ROW_SUM = 100.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data-type",
        choices=["relab", "rpk", "auto"],
        default="auto",
        help=(
            "How to renormalise the feature matrix before log-transform. "
            "'relab' always divides by the row sum (relative abundance / "
            "percentages). 'rpk' skips renormalisation (HUMAnN RPK or any "
            "count-like matrix where row magnitudes matter). 'auto' "
            "(default) preserves legacy behaviour: divide iff mean row "
            "sum > 1.5 AND no row sum exceeds 100; otherwise raise. "
            "Pick explicitly when the table is gene/strain RPK to avoid "
            "the heuristic firing the wrong way."
        ),
    )
    return p.parse_args(argv)


def renormalise(X, data_type):
    """Apply the configured renormalisation policy to the species frame.

    Mutates a copy and returns it.
    """
    X = X.copy()
    rs = X.sum(axis=1)

    if data_type == "relab":
        X = X.div(rs.replace(0, np.nan), axis=0).fillna(0.0)
    elif data_type == "rpk":
        # Absolute magnitudes carry signal; do not divide.
        pass
    elif data_type == "auto":
        max_rs = float(rs.max())
        if max_rs > AUTO_MAX_ROW_SUM:
            raise ValueError(
                f"preprocessing.py: --data-type=auto saw a row sum of "
                f"{max_rs:.2f} (> {AUTO_MAX_ROW_SUM}). This looks like "
                f"an RPK / count table, not a relative-abundance table, "
                f"and the legacy 'divide if mean>1.5' heuristic would "
                f"silently destroy the signal. Re-run with "
                f"--data-type=rpk (no renormalisation) or "
                f"--data-type=relab (force-divide) to be explicit."
            )
        # Legacy heuristic preserved for backward compatibility on the
        # existing 10-cohort MetaPhlAn species table.
        if rs.mean() > 1.5:
            X = X.div(rs.replace(0, np.nan), axis=0).fillna(0.0)
    else:  # pragma: no cover -- argparse choices guard this.
        raise ValueError(f"Unknown --data-type {data_type!r}")

    return X


def main(argv=None):
    args = parse_args(argv)

    species = pd.read_csv('data/raw/species_abundance.csv')
    metadata = pd.read_csv('data/raw/metadata.csv')

    # ── 1. Per-sample depth filter ────────────────────────────────────────────
    if 'number_reads' in metadata.columns:
        before = len(metadata)
        metadata = metadata[metadata['number_reads'] >= MIN_READS].reset_index(drop=True)
        removed = before - len(metadata)
        print(f'Depth filter (>={MIN_READS/1e6:.0f}M reads/sample): removed {removed} samples')

    # ── 2. Cohort exclusion ───────────────────────────────────────────────────
    before = len(metadata)
    metadata = metadata[~metadata['study_name'].isin(EXCLUDE_COHORTS)].reset_index(drop=True)
    print(f'Cohort exclusion {EXCLUDE_COHORTS}: removed {before - len(metadata)} samples')

    print('\nSamples per cohort after quality filters:')
    print(metadata.groupby(['study_name', 'study_condition']).size()
          .unstack(fill_value=0).to_string())

    # ── 3. Align species and metadata ─────────────────────────────────────────
    common = set(species['sample_id']) & set(metadata['sample_id'])
    species  = species[species['sample_id'].isin(common)].reset_index(drop=True)
    metadata = metadata[metadata['sample_id'].isin(common)].reset_index(drop=True)

    # ── 4. Species feature filter (prevalence + mean abundance) ───────────────
    sid = species['sample_id']
    fc = [c for c in species.columns if c != 'sample_id']
    X = species[fc]
    prev = (X > 0).mean()
    ma = X.mean()
    keep = sorted(set(prev[prev >= 0.10].index) & set(ma[ma >= 1e-4].index))
    print(f'\nSpecies features: {len(fc)} raw -> {len(keep)} retained '
          f'(prevalence>=10%, mean>=1e-4)')
    X = X[keep].copy()

    # ── 5. Normalize and log-transform ────────────────────────────────────────
    print(f'Renormalisation policy: --data-type={args.data_type}')
    X = renormalise(X, args.data_type)
    X = np.log10(X + 1e-6)
    X.insert(0, 'sample_id', sid)

    # ── 6. Labels ─────────────────────────────────────────────────────────────
    metadata['label'] = metadata['study_condition'].map(
        {'CRC': 1, 'control': 0, 'adenoma': -1})

    print('\nFinal sample counts:')
    print(metadata['study_condition'].value_counts().to_string())
    print(f'Total: {len(metadata)}')

    # ── 7. Save ───────────────────────────────────────────────────────────────
    os.makedirs('data/processed', exist_ok=True)
    X.to_csv('data/processed/species_filtered.csv', index=False)
    metadata.to_csv('data/processed/metadata_clean.csv', index=False)
    print('\nPreprocessing done')

if __name__ == '__main__':
    main()
