"""Smoke-test assertions on headline numbers from REPRODUCING.md.

Exit 0 if all pass, exit 1 on any failure. Tolerance = 0.002.

Usage:
    python3 scripts/verify_results.py
"""
import sys
import pandas as pd
import numpy as np

TOL = 0.002
failures = []

def check(name, actual, expected, tol=TOL):
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f'  [{status}] {name}: got {actual:.4f}, expected {expected:.4f}')
    if not ok:
        failures.append(name)

def main():
    print('=== Verification checks ===\n')

    # 1. Baseline species RF
    bl = pd.read_csv('results/baseline_results.csv')
    check('Baseline species RF mean AUC', bl['auc'].mean(), 0.803)

    # 2. Joint RF
    jr = pd.read_csv('results/joint_results.csv')
    check('Joint RF mean AUC', jr['rf_auc'].mean(), 0.785)

    # 3. Joint XGB
    check('Joint XGB mean AUC', jr['xgb_auc'].mean(), 0.784)

    # 4. Number of LODO folds
    assert len(bl) == 7, f'Expected 7 baseline folds, got {len(bl)}'
    assert len(jr) == 7, f'Expected 7 joint folds, got {len(jr)}'
    print(f'  [PASS] LODO fold count = 7')

    # 5. DeLong results exist
    dl = pd.read_csv('results/delong_results.csv')
    print(f'  [PASS] DeLong results file exists ({len(dl)} rows)')

    # 6. Prediction files have expected sample count
    for pf in ['results/preds_species_rf.csv', 'results/preds_joint_rf.csv',
               'results/preds_joint_xgb.csv']:
        df = pd.read_csv(pf)
        check(f'{pf} sample count', len(df), 646)

    print(f'\n{len(failures)} failure(s)')
    sys.exit(1 if failures else 0)

if __name__ == '__main__':
    main()
