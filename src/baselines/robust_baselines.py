"""
Three robust evolutionary optimization baselines:

PBRO  — Penalty-Based Robust Optimization (Mirjalili & Lewis, Info Sci 2016)
        DE base + variance penalty: f_robust(x) = f_eff(x) + w * Var[f(x+δ)]

RPSO_IC — Robust PSO with Implicit Conditional operators (Mirjalili et al., SWEVO 2019)
          PSO with neighborhood-reuse: if enough neighbors sampled, use average;
          otherwise resample. Conditional accept: only accept if robust enough.

SRI   — Searching for Robustness Intervals (Du et al., IEEE TEVC 2022)
        Extends DREA-style peak detection, then evaluates worst-case robustness
        f_wc(x) = min_{||δ||<=σ} f(x+δ) ≈ min over sampled neighborhood.
        Selects peaks with best worst-case value.

All use explicit H=30 perturbation samples per evaluation (budget-adapted from
DREA's H=100 to match CEC2026 FE constraints).
"""
from __future__ import annotations

import sys, os
from pathlib import Path
_here = Path(__file__).parent.resolve()
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / 'problems'))

import numpy as np
from scipy.spatial.distance import cdist
from base_algorithm import BaseAlgorithm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mean_effective(x, problem, delta, rng, H=30):
    """f_eff(x) = mean f(x+δ) over H uniform samples."""
    if problem.used_eval >= problem.max_eval - H:
        return float(np.inf)
    pts = x + rng.uniform(-delta, delta, (H, len(x)))
    pts = np.clip(pts, problem.low_bound, problem.up_bound)
    return float(np.mean(problem.func_eval(pts)))


def _var_effective(x, problem, delta, rng, H=30):
    """Var[f(x+δ)] over H samples (robustness measure)."""
    if problem.used_eval >= problem.max_eval - H:
        return float(np.inf)
    pts = x + rng.uniform(-delta, delta, (H, len(x)))
    pts = np.clip(pts, problem.low_bound, problem.up_bound)
    return float(np.var(problem.func_eval(pts)))


def _worst_case(x, problem, delta, rng, H=30):
    """Worst-case f over H samples: max f(x+δ) (minimization)."""
    if problem.used_eval >= problem.max_eval - H:
        return float(np.inf)
    pts = x + rng.uniform(-delta, delta, (H, len(x)))
    pts = np.clip(pts, problem.low_bound, problem.up_bound)
    return float(np.max(problem.func_eval(pts)))


def _greedy_peak_detect(archive_x, n_peaks):
    """Farthest-point greedy peak selection from archive."""
    if len(archive_x) == 0:
        return np.empty((0, archive_x.shape[1] if archive_x.ndim > 1 else 1))
    peaks = [archive_x[0].copy()]
    remaining = list(range(1, len(archive_x)))
    while len(peaks) < n_peaks and remaining:
        d = cdist(archive_x[remaining], np.array(peaks)).min(axis=1)
        nxt = remaining[int(np.argmax(d))]
        peaks.append(archive_x[nxt].copy())
        remaining.remove(nxt)
    return np.array(peaks)


# ---------------------------------------------------------------------------
# PBRO — Penalty-Based Robust Optimization
# ---------------------------------------------------------------------------

class PBRO(BaseAlgorithm):
    """
    Penalty-based robust DE.

    Fitness used for selection: f_pbro(x) = f_eff(x) + w * Var[f(x+δ)]
    w is annealed from w_max→0 so that late in the run pure f_eff dominates.

    Reference: Mirjalili & Lewis, Information Sciences 328:485-509, 2016.
    """

    def __init__(self, seed=None, n_sol=20, H=8, w_max=1.0, delta_frac=0.01, **kw):
        super().__init__(seed=seed, **kw)
        self._n_sol = n_sol
        self._H = H
        self._w_max = w_max
        self._delta_frac = delta_frac

    @property
    def name(self):
        return "PBRO"

    def _run(self, problem):
        rng = np.random.default_rng(self.params.get("seed"))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        total_fe = problem.max_eval
        range_ = float(ub[0] - lb[0])
        delta = self._delta_frac * range_

        NP = min(100, max(20, 5 * dim))
        F, CR = 0.5, 0.9
        H = self._H

        pop = self.random_init(NP, lb, ub, rng)
        # Initial robust fitness
        rob = np.array([_mean_effective(pop[i], problem, delta, rng, H)
                        + self._w_max * _var_effective(pop[i], problem, delta, rng, H)
                        for i in range(NP)
                        if problem.used_eval < total_fe - 2 * H])
        NP = len(rob); pop = pop[:NP]
        if NP == 0:
            return self.random_init(max(self._n_sol, 1), lb, ub, rng)

        best_x = pop[int(np.argmin(rob))].copy()

        while problem.used_eval < total_fe - 2 * H:
            progress = problem.used_eval / max(1, total_fe)
            w = self._w_max * (1.0 - progress)  # anneal penalty weight

            for i in range(NP):
                if problem.used_eval >= total_fe - 2 * H:
                    break
                idxs = rng.choice(NP, 3, replace=NP < 3)
                mutant = pop[idxs[0]] + F * (pop[idxs[1]] - pop[idxs[2]])
                mask = rng.random(dim) < CR
                if not mask.any():
                    mask[rng.integers(dim)] = True
                trial = self.reflect_to_bounds(
                    np.where(mask, mutant, pop[i]), lb, ub)

                f_eff_t = _mean_effective(trial, problem, delta, rng, H)
                var_t = _var_effective(trial, problem, delta, rng, H)
                rob_t = f_eff_t + w * var_t

                if rob_t < rob[i]:
                    pop[i] = trial
                    rob[i] = rob_t

        # Return top n_sol by current robust fitness
        order = np.argsort(rob)[:self._n_sol]
        return pop[order]


# ---------------------------------------------------------------------------
# RPSO_IC — Robust PSO with Implicit Conditional Operators
# ---------------------------------------------------------------------------

class RPSO_IC(BaseAlgorithm):
    """
    Robust PSO with Implicit averaging and Condition (IC) operators.

    At each step, check if enough neighbors have been sampled historically
    near x. If yes, use implicit average (no new FEs). Otherwise resample.
    The conditional accept: only move to new position if it has been visited
    enough times OR its explicit f_eff is better.

    Simplified: standard PSO but with f_eff evaluation (H samples),
    w linearly from 0.9→0.4, c1=c2=2.0.

    Reference: Mirjalili et al., SWEVO 51:100579, 2019.
    """

    def __init__(self, seed=None, n_sol=20, H=8, delta_frac=0.01, **kw):
        super().__init__(seed=seed, **kw)
        self._n_sol = n_sol
        self._H = H
        self._delta_frac = delta_frac

    @property
    def name(self):
        return "RPSO_IC"

    def _run(self, problem):
        rng = np.random.default_rng(self.params.get("seed"))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        total_fe = problem.max_eval
        range_ = float(ub[0] - lb[0])
        delta = self._delta_frac * range_
        H = self._H

        NP = min(100, max(20, 5 * dim))
        c1 = c2 = 2.0
        w0, w1 = 0.9, 0.4

        pos = self.random_init(NP, lb, ub, rng)
        vel = rng.uniform(-range_ * 0.1, range_ * 0.1, (NP, dim))

        rob = np.array([_mean_effective(pos[i], problem, delta, rng, H)
                        for i in range(NP)
                        if problem.used_eval < total_fe - H])
        NP = len(rob); pos = pos[:NP]; vel = vel[:NP]
        if NP == 0:
            return self.random_init(max(self._n_sol, 1), lb, ub, rng)

        pbest_x = pos.copy(); pbest_r = rob.copy()
        gbest_idx = int(np.argmin(rob))
        gbest_x = pos[gbest_idx].copy()

        # Sample history cache (capped to avoid O(n²) slowdown)
        MAX_CACHE = 200
        cache_x = list(pos.copy()); cache_f = list(rob.copy())

        while problem.used_eval < total_fe - H:
            progress = problem.used_eval / max(1, total_fe)
            w = w0 - (w0 - w1) * progress

            for i in range(NP):
                if problem.used_eval >= total_fe - H:
                    break

                r1, r2 = rng.random(dim), rng.random(dim)
                vel[i] = (w * vel[i]
                          + c1 * r1 * (pbest_x[i] - pos[i])
                          + c2 * r2 * (gbest_x - pos[i]))
                new_pos = np.clip(pos[i] + vel[i], lb, ub)

                # Implicit condition: reuse nearby cached evaluations
                cache_arr = np.array(cache_x)
                dists = np.linalg.norm(cache_arr - new_pos, axis=1)
                nearby = np.where(dists < delta * 3)[0]
                if len(nearby) >= H:
                    rob_new = float(np.mean([cache_f[k] for k in nearby[:H]]))
                else:
                    rob_new = _mean_effective(new_pos, problem, delta, rng, H)
                    if len(cache_x) < MAX_CACHE:
                        cache_x.append(new_pos.copy())
                        cache_f.append(rob_new)

                pos[i] = new_pos
                rob[i] = rob_new

                if rob_new < pbest_r[i]:
                    pbest_x[i] = new_pos.copy(); pbest_r[i] = rob_new
                    if rob_new < rob[gbest_idx]:
                        gbest_idx = i; gbest_x = new_pos.copy()

        order = np.argsort(pbest_r)[:self._n_sol]
        return pbest_x[order]


# ---------------------------------------------------------------------------
# SRI — Searching for Robustness Intervals (worst-case robustness)
# ---------------------------------------------------------------------------

class SRI(BaseAlgorithm):
    """
    Searching for Robustness Intervals (Du et al., IEEE TEVC 2022).

    Implements the core logic:
      Stage 1: Multimodal DE (crowding) to find peaks — same as DREA Stage 1.
      Stage 2: For each peak, evaluate worst-case robustness
               f_wc(x) = max_{||δ||≤σ} f(x+δ)  (worst case in minimization)
               Select peaks with smallest f_wc.

    Key distinction from DREA (expected value) and PBRO (mean + variance penalty):
    SRI uses WORST-CASE robustness, i.e. the peak is robust if even in the
    worst perturbation it stays near-optimal.
    """

    def __init__(self, seed=None, n_sol=20, n_peaks=20, H=8,
                 delta_frac=0.01, stage1_frac=0.50, **kw):
        super().__init__(seed=seed, **kw)
        self._n_sol = n_sol
        self._n_peaks = n_peaks
        self._H = H
        self._delta_frac = delta_frac
        self._stage1_frac = stage1_frac

    @property
    def name(self):
        return "SRI"

    def _run(self, problem):
        rng = np.random.default_rng(self.params.get("seed"))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        total_fe = problem.max_eval
        range_ = float(ub[0] - lb[0])
        delta = self._delta_frac * range_
        H = self._H

        stage1_budget = max(dim * 20, int(total_fe * self._stage1_frac))
        stage2_budget = total_fe - stage1_budget

        # --- Stage 1: crowding DE on original fitness ---
        NP = min(100, max(20, 5 * dim))
        F, CR = 0.5, 0.9
        pop = self.random_init(NP, lb, ub, rng)
        val = self._safe_eval(problem, pop, stage1_budget)
        NP = len(val); pop = pop[:NP]

        while problem.used_eval < stage1_budget:
            n_t = min(NP, stage1_budget - problem.used_eval)
            if n_t <= 0:
                break
            trials = np.empty((n_t, dim))
            for i in range(n_t):
                xi = i % NP
                r1, r2, r3 = self._pick3(NP, xi, rng)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mask = rng.random(dim) < CR
                if not mask.any():
                    mask[rng.integers(dim)] = True
                trials[i] = self.reflect_to_bounds(
                    np.where(mask, mutant, pop[xi]), lb, ub)
            tval = self._safe_eval(problem, trials, stage1_budget)
            for i in range(len(tval)):
                xi = i % NP
                dists = np.linalg.norm(pop - trials[i], axis=1)
                dists[xi] = np.inf
                rival = int(np.argmin(dists))
                if tval[i] < val[rival]:
                    pop[rival] = trials[i]; val[rival] = tval[i]

        # Peak selection from final population
        peaks = _greedy_peak_detect(pop, self._n_peaks)
        if len(peaks) == 0:
            return self.random_init(max(self._n_sol, 1), lb, ub, rng)

        # --- Stage 2: worst-case robustness evaluation per peak ---
        per = max(H, stage2_budget // max(len(peaks), 1))
        wc_scores = []
        for pk in peaks:
            if problem.used_eval >= total_fe - H:
                wc_scores.append(float('inf'))
                continue
            # Refine with L-BFGS-B first (if budget allows)
            from scipy.optimize import minimize, Bounds
            try:
                fe_ref = [0]
                def obj(z):
                    if fe_ref[0] >= per // 4 or problem.used_eval >= total_fe - H:
                        raise StopIteration
                    fe_ref[0] += 1
                    return float(problem.func_eval(np.atleast_2d(np.clip(z, lb, ub)))[0])
                res = minimize(obj, pk, method='L-BFGS-B',
                               bounds=Bounds(lb, ub),
                               options={'maxfun': per // 4, 'ftol': 1e-12})
                pk = np.clip(res.x, lb, ub)
            except Exception:
                pass
            wc = _worst_case(pk, problem, delta, rng, H)
            wc_scores.append(wc)

        order = np.argsort(wc_scores)[:self._n_sol]
        return peaks[order]

    @staticmethod
    def _safe_eval(problem, X, limit):
        left = limit - problem.used_eval
        if left <= 0 or len(X) == 0:
            return np.empty(0)
        X = np.atleast_2d(X)
        return problem.func_eval(X[:min(len(X), left)])

    @staticmethod
    def _pick3(N, exclude, rng):
        others = [i for i in range(N) if i != exclude]
        if len(others) < 3:
            return rng.choice(N, 3, replace=True)
        idx = rng.choice(others, 3, replace=False)
        return idx[0], idx[1], idx[2]
