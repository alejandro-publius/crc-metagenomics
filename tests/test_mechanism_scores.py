from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_mechanism_panel import build_mechanism_scores  # noqa: E402


def test_mechanism_scores_use_gene_completeness_and_summed_abundance():
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        )
    )
    manifest = pd.DataFrame(
        [
            {"mechanism": "m", "prespecified_gene": "a", "uniref90": "u1",
             "query_status": "frozen_detected"},
            {"mechanism": "m", "prespecified_gene": "a", "uniref90": "u2",
             "query_status": "frozen_detected"},
            {"mechanism": "m", "prespecified_gene": "b", "uniref90": "u3",
             "query_status": "frozen_detected"},
        ]
    )

    scores = build_mechanism_scores(matrix, ["u1", "u2", "u3"], manifest)

    assert scores["m__abundance"].tolist() == [3.0, 0.0, 4.0]
    assert scores["m__completeness"].tolist() == [1.0, 0.0, 0.5]
