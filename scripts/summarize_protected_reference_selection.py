"""Audit healthy-control prevalence for the frozen protected bacterial panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_selection(
    species: pd.DataFrame,
    metadata: pd.DataFrame,
    references: pd.DataFrame,
    detection_floor: float = -6.0,
) -> pd.DataFrame:
    species = species.set_index("sample_id")
    metadata = metadata.set_index("sample_id")
    controls = species.loc[metadata.index[metadata["label"].eq(0)]]
    species_lookup = {
        column.split("s__", 1)[1]: column
        for column in controls.columns
        if "s__" in column
    }
    prevalence = controls.gt(detection_floor).mean().sort_values(ascending=False)
    ranks = prevalence.rank(method="min", ascending=False).astype(int)
    rows: list[dict[str, object]] = []
    for reference in references.itertuples(index=False):
        if reference.reference_class == "human_reference":
            rows.append(
                {
                    "reference_id": reference.reference_id,
                    "profile_species_name": "not_applicable",
                    "control_prevalence": pd.NA,
                    "control_prevalence_rank": pd.NA,
                    "selection_status": "human_reference_not_applicable",
                }
            )
            continue
        profile_name = str(reference.profile_species_name)
        if profile_name not in species_lookup:
            raise ValueError(f"protected profile species is absent: {profile_name}")
        column = species_lookup[profile_name]
        rows.append(
            {
                "reference_id": reference.reference_id,
                "profile_species_name": profile_name,
                "control_prevalence": float(prevalence[column]),
                "control_prevalence_rank": int(ranks[column]),
                "selection_status": "mapped_to_frozen_control_profile",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    references = pd.read_csv(
        "research/intervention_readiness/protected_reference_panel.csv",
        keep_default_na=False,
    )
    species = pd.read_csv("data/processed/species_filtered.csv")
    metadata = pd.read_csv("data/processed/metadata_clean.csv")
    output = summarize_selection(species, metadata, references)
    path = Path("results/intervention_readiness/protected_reference_selection.csv")
    output.to_csv(path, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
