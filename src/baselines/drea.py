"""
DREA: Dual-Stage Robust Evolutionary Algorithm
Reimplemented based on: arXiv 2401.01070 (IEEE TEVC 2024)
Authors of original paper: Wei Du, Wenxuan Fang, Chen Liang, Yang Tang, Yaochu Jin

Key design:
  Stage 1: Find peaks using NCDE (archive-based, no robust evaluation)
  Stage 2: Search for robust solutions near detected peaks using peak-guided DE
  Robustness: expected fitness = mean of f(x+delta) over H uniform perturbations

Used as comparison algorithm alongside TSRDE.
"""
from __future__ import annotations

import sys, os

import numpy as np
from scipy.spatial.distance import cdist
from base_algorithm import BaseAlgorithm


class DREA(BaseAlgorithm):
    """Dual-Stage Robust Evolutionary Algorithm (arXiv 2401.01070)."""

    def __init__(self, seed=None, n_peaks=3, H=30, delta_frac=0.01,
                 stage1_frac=0.15, **kwargs):
        """
        n_peaks   : number of peaks to detect (Np in paper, default 3)
        H         : perturbation samples for robust fitness (paper uses 100,
                    reduced to 30 for CEC2026 tight FE budgets)
        delta_frac: perturbation radius as fraction of search range
        stage1_frac: fraction of total FE budget given to Stage 1
        """
        super().__init__(seed=seed, **kwargs)
        self._n_peaks = n_peaks
        self._H = H
        self._delta_frac = delta_frac
        self._stage1_frac = stage1_frac

    @property
    def name(self):
        return "DREA"

    def _run(self, problem):
        rng = np.random.default_rng(self.params.get("seed"))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        total_fe = problem.max_eval
        range_ = float(ub[0] - lb[0])

        # FE budget split
        stage1_budget = max(dim * 20, int(total_fe * self._stage1_frac))
        stage2_budget = total_fe - stage1_budget

        # --- Stage 1: peak detection via NCDE ---
        archive = self._stage1_ncde(problem, rng, lb, ub, dim, stage1_budget)

        # Peak detection from archive
        peaks = self._detect_peaks(archive, self._n_peaks)
        if len(peaks) == 0:
            peaks = archive[:1]  # fallback

        # --- Stage 2: refine each peak for robustness ---
        refined = self._stage2_robust(
            problem, rng, lb, ub, dim, peaks,
            stage2_budget, range_)

        # Return all refined peaks as the solution set
        return refined

    # ------------------------------------------------------------------
    # Stage 1: NCDE-based multimodal exploration
    # ------------------------------------------------------------------

    def _stage1_ncde(self, problem, rng, lb, ub, dim, budget):
        """Neighborhood Crowding DE to find peaks (original-fitness, no perturbation)."""
        NP = min(100, max(20, 5 * dim))
        F, CR = 0.5, 0.9
        Na = min(500, budget // 2)  # archive size cap

        pop = self.random_init(NP, lb, ub, rng)
        val = self._safe_eval(problem, pop, budget)
        NP = len(val)

        # archive stores (x, f) pairs
        arch_x = pop.copy()
        arch_f = val.copy()

        while problem.used_eval < budget:
            n_t = min(NP, budget - problem.used_eval)
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
                trial = np.where(mask, mutant, pop[xi])
                trials[i] = self.reflect_to_bounds(trial, lb, ub)

            tval = self._safe_eval(problem, trials, budget)

            # Crowding replacement: trial replaces nearest neighbor if better
            for i in range(len(tval)):
                xi = i % NP
                dists = np.linalg.norm(pop - trials[i], axis=1)
                dists[xi] = np.inf
                rival = int(np.argmin(dists))
                if tval[i] < val[rival]:
                    pop[rival] = trials[i]
                    val[rival] = tval[i]

            # Archive update: keep up to Na points
            for i in range(len(tval)):
                xi = i % NP
                arch_x = np.vstack([arch_x, pop[xi]])
                arch_f = np.append(arch_f, val[xi])

            # Trim archive: keep best + random subset
            if len(arch_f) > Na:
                best_i = int(np.argmin(arch_f))
                rest = np.delete(np.arange(len(arch_f)), best_i)
                keep = rng.choice(rest, Na - 1, replace=False)
                keep = np.append(keep, best_i)
                arch_x = arch_x[keep]
                arch_f = arch_f[keep]

        return arch_x

    # ------------------------------------------------------------------
    # Peak detection: sector-shaped neighborhood (simplified)
    # ------------------------------------------------------------------

    def _detect_peaks(self, archive, n_peaks):
        """
        Simplified peak detection from archive.
        Uses greedy farthest-point selection (distance-based proxy for
        the angular sector method in the paper).
        """
        if len(archive) == 0:
            return np.empty((0, archive.shape[1] if archive.ndim > 1 else 1))

        # Sort by fitness (ascending = minimization)
        order = np.argsort(archive[:, 0]) if archive.ndim == 1 else np.arange(len(archive))

        # Greedy: pick best, then farthest from already-chosen
        peaks = [archive[0]]
        remaining = list(range(1, len(archive)))

        while len(peaks) < n_peaks and remaining:
            dists_to_peaks = cdist(archive[remaining], np.array(peaks)).min(axis=1)
            next_i = remaining[int(np.argmax(dists_to_peaks))]
            peaks.append(archive[next_i])
            remaining.remove(next_i)

        return np.array(peaks)

    # ------------------------------------------------------------------
    # Stage 2: robust search with peak-guided mutation
    # ------------------------------------------------------------------

    def _stage2_robust(self, problem, rng, lb, ub, dim, peaks, budget, range_):
        """Robustly refine each detected peak independently.
        Returns all refined peaks as an array for MMO evaluation.
        """
        n_peaks = len(peaks)
        if n_peaks == 0:
            return peaks

        F, CR = 0.5, 0.9
        delta = self._delta_frac * range_
        per_peak = max(dim * 10, budget // n_peaks)
        NP_peak = max(10, min(50, per_peak // (self._H + 5)))

        results = []
        for pk_x in peaks:
            if problem.used_eval >= budget:
                results.append(pk_x.copy())
                continue

            peak_budget = min(per_peak, budget - problem.used_eval)

            # Small population initialised around this peak
            noise = rng.uniform(-delta * 5, delta * 5, (NP_peak, dim))
            pop = np.clip(pk_x + noise, lb, ub)
            rob_val = np.array([
                self._robust_fitness(pop[i], problem, delta, rng, budget)
                for i in range(NP_peak)
                if problem.used_eval < budget
            ])
            NP_use = len(rob_val)
            pop = pop[:NP_use]
            if NP_use == 0:
                results.append(pk_x.copy())
                continue

            best_x = pop[int(np.argmin(rob_val))].copy()
            peak_end = problem.used_eval + peak_budget

            while problem.used_eval < peak_end:
                n_t = min(NP_use, peak_end - problem.used_eval)
                if n_t <= 0:
                    break
                for i in range(n_t):
                    if problem.used_eval >= peak_end:
                        break
                    xi = i % NP_use
                    xrp = pk_x  # guide toward the detected peak
                    r1, r2, r3 = self._pick3(NP_use, xi, rng)
                    mutant = pop[r1] + F * (xrp - pop[r1]) + F * (pop[r2] - pop[r3])
                    mask = rng.random(dim) < CR
                    if not mask.any():
                        mask[rng.integers(dim)] = True
                    trial = self.reflect_to_bounds(
                        np.where(mask, mutant, pop[xi]), lb, ub)
                    rob_trial = self._robust_fitness(trial, problem, delta, rng, peak_end)
                    if rob_trial < rob_val[xi]:
                        pop[xi] = trial
                        rob_val[xi] = rob_trial
                        if rob_trial < rob_val[int(np.argmin(rob_val))]:
                            best_x = trial.copy()

            results.append(best_x)

        return np.array(results)

    def _robust_fitness(self, x, problem, delta, rng, budget):
        """f_eff(x) = mean f(x + U(-delta, delta)) over H samples."""
        H = self._H
        left = budget - problem.used_eval
        if left <= 0:
            return np.inf
        n = min(H, left)
        samples = x + rng.uniform(-delta, delta, (n, len(x)))
        samples = np.clip(samples, problem.low_bound, problem.up_bound)
        vals = problem.func_eval(samples)
        return float(np.mean(vals))

    @staticmethod
    def _safe_eval(problem, X, limit):
        left = limit - problem.used_eval
        if left <= 0 or len(X) == 0:
            return np.empty(0)
        X = np.atleast_2d(X)
        n = min(len(X), left)
        return problem.func_eval(X[:n])

    @staticmethod
    def _pick3(N, exclude, rng):
        idx = rng.choice(N, 3, replace=False)
        while exclude in idx:
            idx = rng.choice(N, 3, replace=False)
        return idx[0], idx[1], idx[2]
