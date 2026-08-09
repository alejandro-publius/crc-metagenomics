from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_colibactin_human_isolates import summarize_human_panel  # noqa: E402


def test_frozen_human_isolate_manifest_matches_source_filter():
    panel = pd.read_csv(
        ROOT
        / "research"
        / "intervention_readiness"
        / "colibactin_human_isolate_panel.csv"
    )
    assert len(panel) == 97
    assert panel["host"].eq("human").all()
    assert panel["pks_status"].eq("positive_reported_by_source").all()
    assert panel["wgs_accession"].nunique() == 97
    assert panel["source_table_row"].nunique() == 97
    assert panel["source_group"].value_counts().to_dict() == {
        "fecal_commensal": 62,
        "extraintestinal_clinical": 35,
    }
    assert panel["phylogroup"].value_counts().to_dict() == {
        "B2": 95,
        "B1": 1,
        "D": 1,
    }
    for row in panel.itertuples(index=False):
        assert f"/na/{row.wgs_accession}?" in row.download_url


def test_expanded_gate_requires_both_source_groups():
    rows = []
    for guide_id, benchmark_role in [
        ("primary", "primary"),
        ("secondary", "secondary"),
    ]:
        for source_group in ["fecal_commensal", "extraintestinal_clinical"]:
            for index in range(10):
                covered = not (
                    guide_id == "secondary"
                    and source_group == "extraintestinal_clinical"
                    and index >= 7
                )
                rows.append(
                    {
                        "target_id": "colibactin",
                        "guide_id": guide_id,
                        "target_gene": "clbB" if guide_id == "primary" else "clbC",
                        "benchmark_role": benchmark_role,
                        "accession": f"{guide_id}-{source_group}-{index}",
                        "source_group": source_group,
                        "covered": covered,
                        "unique_site": covered,
                    }
                )
    summary, source = summarize_human_panel(pd.DataFrame(rows))
    gates = summary.set_index("guide_id")["human_isolate_conservation_gate"]
    assert gates["primary"] == "pass"
    assert gates["secondary"] == "not_passed"
    secondary_extraintestinal = source.loc[
        source["guide_id"].eq("secondary")
        & source["source_group"].eq("extraintestinal_clinical")
    ].iloc[0]
    assert secondary_extraintestinal["coverage_fraction"] == 0.7
    assert secondary_extraintestinal["source_coverage_gate"] == "not_passed"
