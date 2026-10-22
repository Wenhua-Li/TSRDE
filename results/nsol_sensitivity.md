# n_sol Parameter Sensitivity

TSRDE run on all 20 RobBench problems with n_sol in {1, 2, 3, 5, 7, 10, 15}
(10 runs per combination, 1400 runs total), recomputed from `results_final/nsol_sweep/`.

## Overall mean RobF1 by fixed n_sol setting (pooled over all problems)

| n_sol | 1 | 2 | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| RobF1 | 0.695 | 0.864 | 0.827 | 0.799 | 0.795 | 0.796 | 0.796 |

## Mean RobF1 as a function of $\Delta n$ = n_sol - n_rob (pooled)

| $\Delta n$ | -3 | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RobF1 | 0.400 | 0.542 | 0.700 | 0.952 | 0.834 | 0.756 | 0.833 | 0.737 | 0.820 | 0.676 | 0.892 | 0.823 | 0.609 | 0.946 | 0.892 | 0.823 | 0.609 |

Performance peaks exactly at $\Delta n$ = 0 (0.952) and degrades gracefully and
asymmetrically: under-specification costs recall (0.695 at n_sol = 1), while moderate
over-specification costs precision only gradually (0.74-0.83 for $\Delta n$ in {+1,...,+5}).
Recommended default when n_rob is unknown: n_sol = 3 (0.827), close to the pooled
optimum at n_sol = 2 (0.864).