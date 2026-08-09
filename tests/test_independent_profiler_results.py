import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_committed_targeted_hits_follow_frozen_stringency():
    hits = pd.read_csv(ROOT / "results/independent_profiler/targeted_hits.csv")
    assert len(hits) > 0
    assert (hits.pident >= 90).all()
    assert (hits.alignment_length >= 30).all()


def test_pilot_is_balanced_and_audit_limits_inference():
    samples = pd.read_csv(ROOT / "results/independent_profiler/pilot_samples.csv")
    assert set(samples.groupby(["study_name", "label"]).size()) == {1}
    audit = json.loads((ROOT / "results/independent_profiler/audit.json").read_text())
    assert audit["samples"] == 4
    assert audit["scope"].startswith("direct targeted protein recovery")
