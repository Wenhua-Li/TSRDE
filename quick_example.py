"""
Quick-start example: run TSRDE on a CEC2026 problem.

Usage:
    python quick_example.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "problems"))

import numpy as np
from tsrde import TSRDE
from problems.CMMOP.ProblemMM import ProblemMM
from problems.CMMOP.UtilityMethod import UtilityMethod

# ---- Problem setup ----
pid, pin, dim = 1, 1, 2
problem = ProblemMM(pid, pin, dim)
problem.form()
print(f"Problem: CEC pid={pid} pin={pin} D={dim} | {problem.n_minima} peaks")

# ---- Track B: find the n_sol most robust peaks ----
n_sol = 3
algo = TSRDE(seed=42, n_sol=n_sol, search_robust=True)
solutions = algo._run(problem)

print(f"\n--- Track B (robust search, n_sol={n_sol}) ---")
f = problem.func_eval(np.atleast_2d(solutions))
for i, (x, fv) in enumerate(zip(solutions, f)):
    print(f"  x{i+1} = {np.round(x, 4)}  f = {fv:.6f}")

# ---- Track A: find all distinct peaks ----
algo_a = TSRDE(seed=42, n_sol=problem.n_minima, search_robust=False)
solutions_a = algo_a._run(problem)

rpr, _ = UtilityMethod.calc_robust_peak_ratio(
    np.atleast_2d(solutions_a),
    problem.func_eval(np.atleast_2d(solutions_a)),
    problem.minima.X,
    problem.minima.f,
    np.array([1e-5, 1.0])
)
print(f"\n--- Track A (all peaks) ---")
print(f"  Found {len(solutions_a)} solutions, Peak Ratio = {rpr:.3f}")
