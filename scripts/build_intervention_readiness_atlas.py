"""Build the first, association-only layer of the intervention-readiness atlas.

The script uses the checksum-frozen known-mechanism panel.  It does not infer
causality, sequence conservation, specificity, or editing feasibility from
CRC prediction performance; those remain explicit, separately auditable gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crc_lodo_bench.readiness import (
    CrossPopulationThresholds,
    compute_cohort_target_associations,
    summarize_cross_population_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/intervention_readiness/known_target_registry.csv"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("results/mechanism_panel/sample_scores.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/intervention_readiness"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry, keep_default_na=False)
    scores = pd.read_csv(args.scores)
    thresholds = CrossPopulationThresholds()

    cohort_evidence = compute_cohort_target_associations(scores, registry)
    summary = summarize_cross_population_evidence(
        cohort_evidence, registry, thresholds=thresholds
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_evidence.to_csv(args.output_dir / "known_target_cohort_evidence.csv", index=False)
    summary.to_csv(args.output_dir / "known_target_summary.csv", index=False)
    audit = {
        "analysis_layer": "known_target_cross_population_association",
        "registry": str(args.registry),
        "scores": str(args.scores),
        "n_registered_targets": len(registry),
        "n_cohort_target_rows": len(cohort_evidence),
        "thresholds": {
            "min_evaluable_cohorts": thresholds.min_evaluable_cohorts,
            "min_crc_enriched_fraction": thresholds.min_crc_enriched_fraction,
            "min_median_auc": thresholds.min_median_auc,
        },
        "important_boundary": (
            "Passing this layer means recurring human association only; it does "
            "not establish causality, safety, editability, or treatment efficacy."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
