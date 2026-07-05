"""
NCDE – Nearest-Crowding Differential Evolution
Source: "Exploring the Niching Bias of Differential Evolution", TEC 2012.

Core idea:
  For each individual, find its k nearest neighbours (local neighbourhood)
  and generate a DE trial using only that local pool. The replacement step
  uses crowding: the trial competes with the most similar individual in a
  random subset of the population, preserving population diversity.

Fixes vs. original translation:
  - Correct minimisation direction throughout
  - Added stagnation-driven random restarts to escape dead niches:
    when the best fitness hasn't improved for `restart_patience` generations,
    replace the worst `restart_frac` of the population with fresh random samples
  - _extract_peaks uses ascending sort (lowest f = best)
"""

import numpy as np
from .base_algorithm import BaseAlgorithm
from scipy.spatial.distance import cdist


class NCDE(BaseAlgorithm):

    def __init__(self,
                 pop_size: int = 100,
                 F: float = 0.9,
                 CR: float = 0.1,
                 k_min: int = 5,
                 k_max: int = 20,
                 restart_patience: int = 20,   # gens without improvement → restart
                 restart_frac: float = 0.3,    # fraction of pop replaced on restart
                 seed: int = None):
        super().__init__(pop_size=pop_size, F=F, CR=CR,
                         k_min=k_min, k_max=k_max,
                         restart_patience=restart_patience,
                         restart_frac=restart_frac, seed=seed)

    @property
    def name(self):
        return "NCDE"

    # ------------------------------------------------------------------
    def _run(self, problem):
        rng = np.random.default_rng(self.params.get('seed'))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        NP = self.params['pop_size']
        F = self.params['F']
        CR = self.params['CR']
        k_min = self.params['k_min']
        k_max = min(self.params['k_max'], NP - 1)
        restart_patience = self.params['restart_patience']
        restart_frac = self.params['restart_frac']
        n_restart = max(1, int(NP * restart_frac))

        pop = self.random_init(NP, lb, ub, rng)
        val = problem.func_eval(pop)

        best_val = val.min()
        stagnation = 0

        while problem.used_eval < problem.max_eval:
            budget_left = problem.max_eval - problem.used_eval
            progress = 1.0 - budget_left / problem.max_eval
            # neighbourhood shrinks as search progresses
            k = max(k_min, int(k_max - progress * (k_max - k_min)))
            k = min(k, NP - 1)

            # stagnation check → restart worst individuals
            cur_best = val.min()
            if cur_best < best_val - 1e-10:
                best_val = cur_best
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= restart_patience:
                # replace worst n_restart individuals with fresh random samples
                worst_idx = np.argsort(val)[::-1][:n_restart]
                n_avail = min(n_restart, problem.max_eval - problem.used_eval)
                if n_avail > 0:
                    fresh = self.random_init(n_avail, lb, ub, rng)
                    fresh_val = problem.func_eval(fresh)
                    pop[worst_idx[:n_avail]] = fresh
                    val[worst_idx[:n_avail]] = fresh_val
                stagnation = 0
                best_val = val.min()
                continue

            # build trials
            trials = np.empty_like(pop)
            for j in range(NP):
                dists = np.linalg.norm(pop - pop[j], axis=1)
                dists[j] = np.inf
                nbrs_idx = np.argpartition(dists, k)[:k]
                nbrs = pop[nbrs_idx]

                idxs = rng.choice(len(nbrs), 3, replace=False)
                r1, r2, r3 = nbrs[idxs[0]], nbrs[idxs[1]], nbrs[idxs[2]]
                mutant = r1 + F * (r2 - r3)
                mask = rng.random(dim) < CR
                if not mask.any():
                    mask[rng.integers(dim)] = True
                trial = np.where(mask, mutant, pop[j])
                trials[j] = self.reflect_to_bounds(trial, lb, ub)

            n_eval = min(NP, problem.max_eval - problem.used_eval)
            trial_vals = problem.func_eval(trials[:n_eval])

            for i in range(n_eval):
                # crowding: trial competes with its nearest rival in the population
                dists_to_trial = np.linalg.norm(pop - trials[i], axis=1)
                rival_local = np.argmin(dists_to_trial)
                if trial_vals[i] <= val[rival_local]:
                    pop[rival_local] = trials[i]
                    val[rival_local] = trial_vals[i]

            if n_eval < NP:
                break

        return self._extract_peaks(pop, val, problem.n_minima)

    # ------------------------------------------------------------------
    def _extract_peaks(self, pop, val, n_gm):
        """Greedily pick n_gm best diverse solutions (ascending = best)."""
        order = np.argsort(val)   # ascending: lowest f = best
        pop_s = pop[order]
        reported = []
        for cand in pop_s:
            if len(reported) >= n_gm:
                break
            if len(reported) == 0 or np.min(cdist([cand], reported)) > 1e-8:
                reported.append(cand.copy())
        for cand in pop_s:
            if len(reported) >= n_gm:
                break
            reported.append(cand.copy())
        return np.array(reported[:n_gm])
