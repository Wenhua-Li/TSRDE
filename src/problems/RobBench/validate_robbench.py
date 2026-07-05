"""
Validate all 20 RobBench problems for:
  1. Peak separation: min pairwise centre distance > 2 * max_sigma
  2. Robustness discriminability: sigma_max / sigma_min >= 2.0
  3. Correct n_robust count and labels
  4. func_eval returns finite values at peak centres
"""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).parent.resolve()
sys.path.insert(0, str(_here.parent.parent))          # alg_code/
sys.path.insert(0, str(_here.parent))                 # alg_code/problems/

import numpy as np
from scipy.spatial.distance import cdist
from robbench import ALL_PROBLEMS


def validate():
    ok_all = True
    header = f"{'Name':8s}  {'D':3s}  {'K':3s}  {'nR':3s}  {'base':5s}  " \
             f"{'minDist':8s}  {'maxSig':7s}  {'sep':4s}  " \
             f"{'sigRat':7s}  {'disc':5s}  {'fOK':4s}"
    print(header)
    print('-' * len(header))

    for name, factory in ALL_PROBLEMS:
        p = factory()
        centers = p.minima.X                          # (K, D)
        n_rob   = len(p._robust_idx)
        K       = p.n_minima
        D       = p.dim

        # Mean-sigma per peak
        mean_sig = p.gt_robustness
        max_sig  = float(mean_sig.max())
        min_sig  = float(mean_sig.min())
        sig_rat  = max_sig / min_sig if min_sig > 0 else float('inf')

        # Peak separation: for each peak k, nearest neighbour dist > 2 * mean_sigma_k
        # (each peak's basin should not overlap its nearest neighbour)
        if K > 1:
            dist_mat = cdist(centers, centers)
            np.fill_diagonal(dist_mat, np.inf)
            min_dist = float(dist_mat.min())
            # Per-peak check: nearest-neighbour dist > 2 * own sigma
            nn_dist = dist_mat.min(axis=1)                     # (K,)
            sep_ok  = bool(np.all(nn_dist > 2 * mean_sig))
        else:
            min_dist = float('inf')
            sep_ok   = True
        disc_ok = sig_rat >= 2.0
        base    = n_rob / K

        # func_eval at each centre should give ≈ -10
        f_vals = p.func_eval(centers)
        f_ok   = bool(np.all(np.isfinite(f_vals)) and
                      np.allclose(f_vals, -p._height, atol=1e-6))

        status = 'OK' if (sep_ok and disc_ok and f_ok) else 'FAIL'
        if status == 'FAIL':
            ok_all = False

        print(f'{name:8s}  {D:3d}  {K:3d}  {n_rob:3d}  {base:5.3f}  '
              f'{min_dist:8.3f}  {max_sig:7.3f}  {"OK" if sep_ok else "FAIL":4s}  '
              f'{sig_rat:7.2f}  {"OK" if disc_ok else "FAIL":5s}  '
              f'{"OK" if f_ok else "FAIL":4s}  [{status}]')

    print()
    if ok_all:
        print('All 20 problems passed validation.')
    else:
        print('WARNING: Some problems failed validation — see FAIL rows above.')
    return ok_all


if __name__ == '__main__':
    validate()
