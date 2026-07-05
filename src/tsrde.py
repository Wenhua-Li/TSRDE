"""
TSRDE: Two-Stage Robust Differential Evolution for Multimodal Optimization

User parameters:
  n_sol (default=3): target peaks to return (Track B only).
    For RobBench problems with _robust_idx, auto-set to len(problem._robust_idx).
  search_robust (default=True): mode switch
    True  → Track B: find n_sol most robust peaks
    False → Track A: find all distinct peaks

Phase 1: k-NN rival crowding DE + archive + pure-random restarts
Phase 2: quality screen → coarse robustness eval → diverse_rob_select →
         competition-elimination loop → deep refinement + dedup

Robustness: E[f(x+δ)] (expected fitness), σ=0.1*Rs, M*(D) theory-grounded adaptive samples.

Track A budget schedule (dim-adaptive, v0.15):
  D< 10: α=0.65, scan=20%  — low-D DE covers space; Phase-2 gets 15%
  D=10:  α=0.525, scan=35% — Phase-2 gets 12.5%; scan supplements DE
  D=20:  α=0.30,  scan=35% — Phase-2 gets 35%; high-D scan+refine dominates
"""
from __future__ import annotations

import sys
from pathlib import Path
_here = Path(__file__).parent.resolve()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist

from base_algorithm import BaseAlgorithm
from evri_utils import compute_robustness, compute_M, SIGMA_FRAC


class TSRDE(BaseAlgorithm):
    """Two-Stage Robust DE for Multimodal Optimization."""

    def __init__(self, seed=None, n_sol=3, search_robust=True, ablation=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._n_sol = n_sol
        self._search_robust = search_robust
        # ablation in {None, 'mmo', 'no_restart', 'no_elim'} — see _run/_phase1/_phase2.
        # None reproduces the exact published TSRDE behaviour (all ablation
        # branches are no-ops when this is None).
        self._ablation = ablation
        self._max_restarts = 0 if ablation == 'no_restart' else 5

    @property
    def name(self):
        return "TSRDE" if self._search_robust else "TSRDE-MMO"

    # =================================================================
    # Main driver
    # =================================================================

    def _run(self, problem):
        rng = np.random.default_rng(self.params.get("seed"))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        total_fe = problem.max_eval
        range_ = float(ub[0] - lb[0])

        # n_sol is passed directly by the caller (run_experiments.py already sets
        # n_sol=n_rob for the main RobBench experiment at the task level).
        # No auto-override here so that the n_sol sensitivity sweep works correctly.
        n_sol = self._n_sol

        N_est = max(20, 5 * dim)
        Rs = 0.5 * range_ / (N_est ** (1.0 / dim)) / 2.5

        # NP: uniform np_mult=4 for both tracks.
        # (np_mult=6 for Track B high-D was tried in v0.13 but reduced generation
        #  count and caused Group A D=10 regression — reverted and kept at 4.)
        NP = min(max(N_est * 4, 15 * dim), total_fe // 10)

        arch_Rs = Rs / 4.0 if not self._search_robust else Rs
        arch_cap = max(6 * N_est, 80)

        # Phase-1 budget fraction α.
        # Track B (robust search): α=0.65 fixed — n_sol is small, Phase-2
        #   robustness eval is cheap per peak, Phase-1 exploration dominates.
        # Track A (MMO, CEC): α shrinks with dimension because at high D the
        #   k-NN rival DE loses exploration efficiency rapidly; shifting budget to
        #   Phase-2's random scan + local refinement recovers high-D peak recall.
        #   Validated in v0.7 local probe: D=10 +112%, D=20 +334% over α=0.65.
        if self._search_robust:
            alpha = 0.65
        else:
            if dim >= 20:
                alpha = 0.30
            elif dim >= 10:
                alpha = 0.525
            else:
                alpha = 0.65
        phase1_budget = int(total_fe * alpha)

        # ── Phase 1: broad multimodal exploration ────────────────────────
        arch_x, arch_f = self._phase1(
            problem, rng, lb, ub, dim, N_est, NP,
            phase1_budget, arch_Rs, arch_cap, range_)

        if not self._search_robust:
            arch_x, arch_f = self._mmo_scan(
                problem, rng, lb, ub, dim, arch_x, arch_f, arch_Rs, arch_cap)

        if len(arch_x) == 0:
            n_fb = max(n_sol, 1)
            return self.random_init(n_fb, lb, ub, rng)

        # Abl-MMO: skip Phase 2 robustness refinement entirely — return the
        # n_sol archive members with the best nominal fitness (paper Sec. V-D).
        if self._ablation == 'mmo':
            top = np.argsort(arch_f)[:max(n_sol, 1)]
            return arch_x[top]

        # ── Phase 2: competition-elimination refinement ───────────────
        budget_left = problem.max_eval - problem.used_eval
        result = self._phase2(
            problem, rng, lb, ub, dim, arch_x, arch_f,
            budget_left, Rs, range_, n_sol)

        return result

    def _mmo_scan(self, problem, rng, lb, ub, dim, arch_x, arch_f, Rs, arch_cap):
        # Scan fraction is dimension-adaptive, paired with the adaptive-α schedule:
        #   D < 10 : 20% — same as v0.13; at low D, Phase-2 refinement is cheap
        #            and Phase-1 DE already covers the space well.
        #   D ≥ 10 : 35% — compensates for the shorter Phase-1 (α=0.525/0.30)
        #            by feeding more random candidates into Phase-2 local search,
        #            which is the primary exploration mechanism at high dimensions.
        # Using 35% for all D would drain Phase-2 budget at D=2/5 (leaves 0 FE).
        scan_frac = 0.35 if dim >= 10 else 0.20
        budget_scan = min(
            problem.max_eval - problem.used_eval,
            max(int(problem.max_eval * scan_frac), 500 * dim))
        budget_scan = max(0, budget_scan)
        if budget_scan <= 0:
            return arch_x, arch_f

        n_scan = min(budget_scan, max(2000 * dim, budget_scan))
        scan_x = self.random_init(n_scan, lb, ub, rng)
        scan_f = self._safe_eval(problem, scan_x, problem.used_eval + budget_scan)
        if len(scan_f) == 0:
            return arch_x, arch_f
        arch_x, arch_f = self._arch_insert(arch_x, arch_f, scan_x[:len(scan_f)], scan_f, Rs, arch_cap)
        return arch_x, arch_f

    # =================================================================
    # Phase 1: k-NN rival crowding DE
    # =================================================================

    def _phase1(self, problem, rng, lb, ub, dim, N_est, NP,
                budget, Rs, arch_cap, range_):
        k_lcf = min(NP - 1, max(3, min(20, 3 * dim)))
        k_rival = max(3, min(10, NP // max(1, N_est)))
        max_restarts = self._max_restarts
        stall_lim = max(5, min(int(0.5 * budget / ((max_restarts + 1) * NP)), NP))

        arch_x = np.empty((0, dim))
        arch_f = np.empty(0)

        pop = self.random_init(NP, lb, ub, rng)
        val = self._safe_eval(problem, pop, budget)
        NP = len(val)
        if NP == 0:
            return arch_x, arch_f

        arch_x, arch_f = self._arch_insert(arch_x, arch_f, pop, val, Rs, arch_cap)
        mu_F, mu_CR = 0.5, 0.9

        arch_size_prev = len(arch_x)
        stall_cnt = 0
        n_restarts = 0

        while problem.used_eval < budget:
            progress = problem.used_eval / max(1, budget)
            n_t = min(NP, budget - problem.used_eval)
            if n_t <= 0:
                break

            k_riv = max(2, k_rival - int(2 * progress))

            trials = np.empty((n_t, dim))
            F_used = np.empty(n_t)
            CR_used = np.empty(n_t)

            for i in range(n_t):
                F = float(np.clip(mu_F + 0.1 * rng.standard_cauchy(), 0.1, 1.0))
                CR = float(np.clip(mu_CR + 0.1 * rng.standard_normal(), 0.0, 1.0))
                xi = i % NP
                trials[i] = self._de_rand1(pop, xi, lb, ub, rng, F, CR)
                F_used[i] = F
                CR_used[i] = CR

            tval = self._safe_eval(problem, trials, budget)
            n_t = len(tval)

            lcf_p = self._lcf(pop, val, min(k_lcf, NP - 1))[0]
            cd_pop = self._crowding_distance(pop)

            sF, sCR = [], []
            for i in range(n_t):
                dists = np.linalg.norm(pop - trials[i], axis=1)
                nbrs = np.argpartition(dists, min(k_riv, NP - 1))[:k_riv]
                scores = lcf_p[nbrs].astype(float) * 1e6 - cd_pop[nbrs]
                rival = nbrs[int(np.argmax(scores))]
                if tval[i] < val[rival]:
                    pop[rival] = trials[i]
                    val[rival] = tval[i]
                    sF.append(F_used[i])
                    sCR.append(CR_used[i])

            mu_F, mu_CR = self._jade_update(mu_F, mu_CR, sF, sCR)

            lcf_final = self._lcf(pop, val, min(k_lcf, NP - 1))[0]
            mask = lcf_final == 0
            if np.any(mask):
                arch_x, arch_f = self._arch_insert(
                    arch_x, arch_f, pop[mask], val[mask], Rs, arch_cap)

            if len(arch_x) == arch_size_prev:
                stall_cnt += 1
            else:
                stall_cnt = 0
                arch_size_prev = len(arch_x)

            if (stall_cnt >= stall_lim and n_restarts < max_restarts
                    and problem.used_eval < budget * 0.9):
                pop = self.random_init(NP, lb, ub, rng)
                val = self._safe_eval(problem, pop, budget)
                NP = len(val)
                if NP == 0:
                    break
                arch_x, arch_f = self._arch_insert(
                    arch_x, arch_f, pop, val, Rs, arch_cap)
                mu_F, mu_CR = 0.5, 0.9
                stall_cnt = 0
                n_restarts += 1

        arch_x, arch_f = self._arch_insert(arch_x, arch_f, pop, val, Rs, arch_cap)
        return arch_x, arch_f

    # =================================================================
    # Phase 2: competition-elimination refinement
    # =================================================================

    def _phase2(self, problem, rng, lb, ub, dim, arch_x, arch_f,
                budget, Rs, range_, n_sol):
        search_robust = self._search_robust
        method = "Nelder-Mead" if dim <= 5 else "L-BFGS-B"
        min_rob_fe = compute_M(dim)

        # ── Step 1: Quality screen (no FE) ──────────────────────────────
        f_best = float(arch_f.min())
        f_range = float(np.percentile(arch_f, 90) - f_best) + 1e-9
        band = max(f_range * 0.05, 0.5 * (float(np.percentile(arch_f, 50)) - f_best))
        qual_mask = arch_f <= f_best + band
        qual_idx = np.where(qual_mask)[0]
        qual_idx = qual_idx[np.argsort(arch_f[qual_idx])]

        n_keep_qual = len(qual_idx) if not search_robust else min(len(qual_idx), max(n_sol * 5, 15))
        K = n_keep_qual
        cand_x = arch_x[qual_idx[:K]].copy()
        cand_f = arch_f[qual_idx[:K]].copy()
        n_cand = len(cand_x)

        if n_cand == 0:
            top = np.argsort(arch_f)[:max(n_sol, 1)]
            return arch_x[top]

        if n_cand <= n_sol:
            cand_x, cand_f = self._deep_refine(
                cand_x, cand_f, budget, problem, rng, lb, ub, dim,
                method, Rs, min_rob_fe, niche_bound=not search_robust)
            cand_x, cand_f = self._dedup(cand_x, cand_f, Rs)
            return np.atleast_2d(cand_x)

        # ── Step 2: Coarse robustness eval → first elimination ──────────
        remaining = problem.max_eval - problem.used_eval
        fe_coarse = min(
            max(min_rob_fe, 5 * dim),
            max(1, remaining // (n_cand * 3)))
        rob = np.zeros(n_cand)
        if search_robust:
            for i in range(n_cand):
                if problem.used_eval >= problem.max_eval - min_rob_fe:
                    break
                rob[i] = compute_robustness(
                    cand_x[i], problem, cand_f[i], Rs, rng,
                    max_fe=fe_coarse)

        M = min(n_cand, max(n_sol * 3, n_sol + 2))
        keep_idx = self._diverse_rob_select(cand_x, rob, M, Rs)
        cand_x = cand_x[keep_idx]
        cand_f = cand_f[keep_idx]
        rob    = rob[keep_idx]

        # Abl-noElim: replace the iterative competitive elimination loop (Step 3)
        # with a single uniform-random draw of n_sol candidates from the pool
        # that survived the initial robustness-based cull (Step 2) — paper Sec. V-D.
        if self._ablation == 'no_elim':
            n_final = min(n_sol, len(cand_x))
            sel = rng.choice(len(cand_x), size=n_final, replace=False)
            cand_x, cand_f = cand_x[sel], cand_f[sel]
            remaining = problem.max_eval - problem.used_eval
            if remaining > dim * 2:
                cand_x, cand_f = self._deep_refine(
                    cand_x, cand_f, remaining, problem, rng, lb, ub, dim,
                    method, Rs, min_rob_fe, niche_bound=not search_robust)
            cand_x, cand_f = self._dedup(cand_x, cand_f, Rs)
            return cand_x if not search_robust else cand_x[:n_sol]

        # ── Step 3: Competition-elimination loop ─────────────────────────
        elim_rate  = 0.33
        round_frac = 0.35

        while len(cand_x) > n_sol:
            remaining = problem.max_eval - problem.used_eval
            if remaining < dim * 5:
                break

            n_c = len(cand_x)
            round_budget = max(1, int(remaining * round_frac))
            per_peak = max(dim * 3, round_budget // n_c)

            for i in range(n_c):
                if problem.used_eval >= problem.max_eval - dim * 2:
                    break
                new_x, new_f = self._local_opt_with_cache(
                    cand_x[i], problem, rng, lb, ub, dim,
                    per_peak, method, Rs,
                    niche_center=None)
                cand_x[i] = new_x
                cand_f[i] = new_f

            if search_robust:
                rob_fe = max(min_rob_fe, max(1,
                    (problem.max_eval - problem.used_eval) // max(n_c * 2, 1)))
                for i in range(n_c):
                    if problem.used_eval >= problem.max_eval - min_rob_fe:
                        break
                    rob[i] = compute_robustness(
                        cand_x[i], problem, cand_f[i], Rs, rng,
                        max_fe=rob_fe)

            n_elim = max(1, int(np.ceil(n_c * elim_rate)))
            n_keep = max(n_sol, n_c - n_elim)
            keep_idx = self._diverse_rob_select(cand_x, rob, n_keep, Rs)
            cand_x = cand_x[keep_idx]
            cand_f = cand_f[keep_idx]
            rob    = rob[keep_idx]

        # ── Step 4: Deep refinement + dedup ─────────────────────────────
        remaining = problem.max_eval - problem.used_eval
        if remaining > dim * 2:
            cand_x, cand_f = self._deep_refine(
                cand_x, cand_f, remaining, problem, rng, lb, ub, dim,
                method, Rs, min_rob_fe, niche_bound=not search_robust)

        cand_x, cand_f = self._dedup(cand_x, cand_f, Rs)
        return cand_x if not search_robust else cand_x[:n_sol]

    # =================================================================
    # Helpers
    # =================================================================

    def _local_opt_with_cache(self, x0, problem, rng, lb, ub, dim,
                               budget, method, Rs, niche_center=None):
        evals = [0]
        best_x = [x0.copy()]
        best_f = [np.inf]

        if niche_center is not None:
            eff_lb = np.maximum(lb, niche_center - Rs)
            eff_ub = np.minimum(ub, niche_center + Rs)
        else:
            eff_lb, eff_ub = lb, ub

        def obj(z):
            if evals[0] >= budget or problem.used_eval >= problem.max_eval:
                raise StopIteration
            evals[0] += 1
            z_clipped = np.clip(z, eff_lb, eff_ub)
            fv = float(problem.func_eval(np.atleast_2d(z_clipped))[0])
            if fv < best_f[0]:
                best_f[0] = fv
                best_x[0] = z_clipped.copy()
            return fv

        try:
            if method == "Nelder-Mead":
                minimize(obj, x0, method="Nelder-Mead",
                         options={"maxfev": budget, "xatol": 1e-9,
                                  "fatol": 1e-11, "adaptive": True})
            else:
                minimize(obj, x0, method="L-BFGS-B",
                         bounds=Bounds(eff_lb, eff_ub),
                         options={"maxfun": budget, "ftol": 1e-12,
                                  "gtol": 1e-8, "maxcor": min(dim, 10)})
        except (StopIteration, Exception):
            pass

        return best_x[0], best_f[0]

    @staticmethod
    def _diverse_rob_select(cand_x, rob, n_keep, Rs):
        n = len(cand_x)
        n_keep = min(n_keep, n)
        if n_keep <= 0:
            return np.array([], dtype=int)
        if n_keep >= n:
            return np.arange(n)

        rob_clean = np.where(np.isfinite(rob), rob,
                             np.nanmax(rob[np.isfinite(rob)], initial=1.0) * 10)
        # penalty_val must be positive and larger than the score range so that
        # adding it to a nearby candidate always pushes it past any non-penalised
        # candidate.  Using max()*2+1 broke when scores are negative (e.g. mean
        # fitness ≈ -10): max would be e.g. -9.68, giving penalty = -18.36,
        # which made penalised candidates look MORE attractive (bug introduced
        # in v0.13 and masked because variance scores were always positive).
        # Fix: use score spread (ptp = peak-to-peak = max-min, always ≥ 0).
        penalty_val = float(np.ptp(rob_clean)) * 2.0 + 1.0

        selected = []
        remaining = list(range(n))
        scores = rob_clean.copy()

        for _ in range(n_keep):
            effective = scores.copy()
            for sel in selected:
                dists = np.linalg.norm(cand_x[remaining] - cand_x[sel], axis=1)
                for j, r in enumerate(remaining):
                    if dists[j] < Rs:
                        effective[r] += penalty_val
            best_j = int(np.argmin([effective[r] for r in remaining]))
            best_r = remaining[best_j]
            selected.append(best_r)
            remaining.remove(best_r)

        return np.array(selected, dtype=int)

    def _deep_refine(self, cand_x, cand_f, budget, problem, rng,
                     lb, ub, dim, method, Rs, min_rob_fe,
                     niche_bound=False):
        n = len(cand_x)
        if n == 0 or budget <= 0:
            return cand_x, cand_f
        per = max(dim * 2, budget // n)
        for i in range(n):
            if problem.used_eval >= problem.max_eval - dim:
                break
            nc = cand_x[i] if niche_bound else None
            new_x, new_f = self._local_opt_with_cache(
                cand_x[i], problem, rng, lb, ub, dim,
                per, method, Rs, niche_center=nc)
            cand_x[i] = new_x
            cand_f[i] = float(new_f)
        return cand_x, cand_f

    @staticmethod
    def _dedup(x, f, Rs):
        if len(x) == 0:
            return x, f
        order = np.argsort(f)
        keep = []
        for idx in order:
            if all(np.linalg.norm(x[idx] - x[k]) >= Rs for k in keep):
                keep.append(idx)
        keep = np.array(keep, dtype=int)
        return x[keep], f[keep]

    # =================================================================
    # Utilities
    # =================================================================

    @staticmethod
    def _de_rand1(pop, i, lb, ub, rng, F=0.8, CR=0.3):
        N, D = pop.shape
        idxs = rng.choice(N, 3, replace=N < 3)
        mutant = pop[idxs[0]] + F * (pop[idxs[1]] - pop[idxs[2]])
        mask = rng.random(D) < CR
        if not mask.any():
            mask[rng.integers(D)] = True
        trial = np.where(mask, mutant, pop[i])
        return BaseAlgorithm.reflect_to_bounds(trial, lb, ub)

    @staticmethod
    def _safe_eval(problem, X, limit):
        left = limit - problem.used_eval
        if left <= 0 or len(X) == 0:
            return np.empty(0)
        X = np.atleast_2d(X)
        n = min(len(X), left)
        return problem.func_eval(X[:n])

    @staticmethod
    def _jade_update(mu_F, mu_CR, sF, sCR, c=0.1):
        if sF:
            sF_arr = np.asarray(sF)
            lehmer = np.sum(sF_arr ** 2) / max(np.sum(sF_arr), 1e-12)
            mu_F = (1 - c) * mu_F + c * lehmer
            mu_CR = (1 - c) * mu_CR + c * float(np.mean(sCR))
        return float(np.clip(mu_F, 0.1, 1.0)), float(np.clip(mu_CR, 0.05, 0.95))

    @staticmethod
    def _lcf(X, f, k):
        N = len(X)
        if N <= 1:
            return np.zeros(N, dtype=int), np.zeros((N, 0), dtype=int)
        k = max(1, min(k, N - 1))
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        nn = np.argpartition(D, k, axis=1)[:, :k]
        lcf = np.sum(f[nn] < f[:, None], axis=1).astype(int)
        return lcf, nn

    @staticmethod
    def _crowding_distance(X):
        N, D = X.shape
        if N <= 2:
            return np.full(N, 1.0)
        cd = np.zeros(N)
        for d in range(D):
            order = np.argsort(X[:, d])
            span = X[order[-1], d] - X[order[0], d]
            if span <= 0:
                continue
            cd[order[0]] += 1.0
            cd[order[-1]] += 1.0
            cd[order[1:-1]] += (X[order[2:], d] - X[order[:-2], d]) / span
        return cd / max(1, D)

    @staticmethod
    def _arch_insert(arch_x, arch_f, new_x, new_f, Rs, cap):
        if len(new_x) == 0:
            return arch_x, arch_f
        for idx in np.argsort(new_f):
            x, fv = new_x[idx], float(new_f[idx])
            if len(arch_x) == 0:
                arch_x = x[None, :].copy()
                arch_f = np.array([fv])
                continue
            dists = np.linalg.norm(arch_x - x, axis=1)
            near = int(np.argmin(dists))
            if dists[near] < Rs:
                if fv < arch_f[near]:
                    arch_x[near] = x.copy()
                    arch_f[near] = fv
            elif len(arch_x) < cap:
                arch_x = np.vstack([arch_x, x[None, :]])
                arch_f = np.append(arch_f, fv)
            else:
                worst = int(np.argmax(arch_f))
                if fv < arch_f[worst]:
                    arch_x[worst] = x.copy()
                    arch_f[worst] = fv
        return arch_x, arch_f
