import pandas as pd

from scripts.run_independent_profiler import select_pilot


def test_pilot_selection_is_balanced_and_uses_single_accessions():
    rows = []
    for cohort in ("GuptaA_2019", "WirbelJ_2018"):
        for label in (0, 1):
            rows.extend([
                {"sample_id": "b", "study_name": cohort, "label": label,
                 "NCBI_accession": "SRR2"},
                {"sample_id": "a", "study_name": cohort, "label": label,
                 "NCBI_accession": "SRR1"},
            ])
    selected = select_pilot(pd.DataFrame(rows))
    assert len(selected) == 4
    assert set(selected.groupby(["study_name", "label"]).size()) == {1}
    assert set(selected.sample_id) == {"a"}
    assert not selected.NCBI_accession.str.contains(";").any()
