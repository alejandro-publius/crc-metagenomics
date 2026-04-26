#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
if [ ! -f data/raw/pathway_abundance.csv ]; then
    echo "==> merge_pathways.py (pathway_abundance.csv missing)"
    python3 scripts/merge_pathways.py
fi
echo "==> bootstrap_ci.py";          python3 scripts/bootstrap_ci.py
echo "==> seed_sensitivity.py";      python3 scripts/seed_sensitivity.py
echo "==> sensitivity_analysis.py";  python3 scripts/sensitivity_analysis.py
echo "==> confounder_adjustment.py"; python3 scripts/confounder_adjustment.py
echo "==> batch_correction.py";      python3 scripts/batch_correction.py
echo "==> adenoma_lodo.py";          python3 scripts/adenoma_lodo.py
echo "==> verify_results.py";        python3 scripts/verify_results.py
echo ""; echo "Done. Latest results files:"; ls -lt results/ | head -25
