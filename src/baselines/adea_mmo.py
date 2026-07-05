"""
ADEA-MMO: Adaptive Dual-Engine Archive for Multimodal Optimization
CEC 2026 MMO Competition entry.

Core mechanism: Local-Convergence Fitness (LCF) niching
---------------------------------------------------------
  LCF(i) = # of k-nearest neighbors (in decision space) that have strictly
  lower objective value. Local optima have LCF=0. This is a parameter-free,
  dimension-agnostic fitness landscape that naturally separates basins.

  Crowding Distance (CD) in decision space as secondary criterion: among
  equally-ranked individuals, prefer the more isolated one (larger CD).

  Replacement: trial replaces its worst k-NN by (LCF_rival, -CD_rival).
  Accept if: tval < rival_val (fitness improvement within a basin).

No niche radius needed during evolution — LCF handles it automatically.
An archive with radius Rs_arch captures distinct peaks for reporting.
Phase 2: Nelder-Mead refinement on archive candidates (15% FE budget).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .base_algorithm import BaseAlgorithm
from .archive import Archive


class ADEAMMO(BaseAlgorithm):
    """DE with Local-Convergence Fitness + Crowding Distance niching."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)

    @property
    def name(self) -> str:
        return "ADEA-MMO"

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _run(self, problem) -> np.ndarray:
        rng = np.random.default_rng(self.params.get("seed"))
        lb, ub = problem.low_bound, problem.up_bound
        dim    = problem.dim
        n_gm   = problem.n_minima
        FE_max = problem.max_eval

        # Archive deduplication radius: large enough to deduplicate same-basin
        # solutions, small enough not to merge distinct peaks.
        # Use 90% of minimum niche radius (niche_rad is half the nearest-neighbor
        # distance, so Rs_arch < niche_rad guarantees distinct peaks both enter)
        Rs_arch = float(np.min(problem.minima.niche_rad)) * 0.9

        refine_reserve = int(FE_max * 0.20)  # 20% for NM refinement
        explore_budget = FE_max - refine_reserve

        # Population size: scales with D for coverage, capped to not waste FE
        NP = max(n_gm * 4, min(n_gm * 20, 20 * dim + n_gm))

        # k for LCF neighborhood — 3*dim to sense wider local basin
        k_lcf   = max(3, min(20, 3 * dim))
        k_lcf   = min(k_lcf, NP - 1)
        # k for rival selection — localized competition within each basin
        k_rival = max(3, min(10, NP // max(1, n_gm)))

        pop = self._lhc_init(NP, lb, ub, rng)
        val = problem.func_eval(pop)

        archive = Archive(capacity=n_gm, Rs=Rs_arch)

        # ── Phase 1: DE with LCF+CD niching ───────────────────────────
        itr    = 0
        while problem.used_eval < explore_budget:
            itr += 1
            n_t = min(NP, explore_budget - problem.used_eval)
            if n_t <= 0:
                break

            # Generate trials
            trials = np.empty((n_t, dim))
            for i in range(n_t):
                trials[i] = self._de_rand1(pop, i % NP, lb, ub, rng)

            tval_raw = problem.func_eval(trials)

            # LCF and CD — recompute fresh each iteration for accuracy
            lcf_p  = self._lcf(pop, val, k_lcf)
            cd_pop = self._crowding_distance(pop)

            # LCF for trials vs pop
            tlcf = self._lcf_query(trials, tval_raw, pop, val, k_lcf)

            # Replacement
            for i in range(n_t):
                dists  = np.linalg.norm(pop - trials[i], axis=1)
                nbrs   = np.argpartition(dists, min(k_rival, NP - 1))[:k_rival]
                # Rival = worst neighbor by (lcf DESC, cd ASC)
                scores = lcf_p[nbrs].astype(float) * 1e6 - cd_pop[nbrs]
                rival  = nbrs[int(np.argmax(scores))]

                # Accept: trial has lower raw fitness (greedy within basin)
                if tval_raw[i] < val[rival]:
                    pop[rival]   = trials[i]
                    val[rival]   = tval_raw[i]
                    lcf_p[rival] = tlcf[i]
                    cd_pop[rival] = 0.0

                    archive.try_insert(trials[i], tval_raw[i], f_opt=-np.inf)

        # Final archive pass from population local-bests
        lcf_final = self._lcf(pop, val, k_lcf)
        lb_order  = np.argsort(lcf_final * 1e9 + val)
        for idx in lb_order:
            if len(archive) >= n_gm:
                break
            archive.try_insert(pop[idx], float(val[idx]), f_opt=-np.inf)

        # ── Phase 2: Nelder-Mead refinement ───────────────────────────
        candidates = list(archive.get_top(n_gm))  # convert to list early

        # Fill if archive sparse
        if len(candidates) < n_gm:
            order = np.argsort(val)
            for idx in order:
                if len(candidates) >= n_gm:
                    break
                x_c = pop[idx]
                if not candidates or float(np.min(
                        np.linalg.norm(np.array(candidates) - x_c, axis=1))) > Rs_arch * 0.5:
                    candidates.append(x_c.copy())

        budget_left = problem.max_eval - problem.used_eval
        if budget_left > 0 and candidates:
            nm_budget = max(20, budget_left // max(1, len(candidates)))
            refined = []
            for x in candidates:
                if problem.used_eval >= problem.max_eval:
                    refined.append(x)
                    continue
                refined.append(self._nm_refine(x, problem, lb, ub, nm_budget))
            # Deduplicate after NM (NM may map distinct candidates to same peak)
            deduped: list[np.ndarray] = []
            for x_r in refined:
                if deduped:
                    dists = np.linalg.norm(np.array(deduped) - x_r, axis=1)
                    if float(np.min(dists)) < Rs_arch:
                        continue
                deduped.append(x_r)
            candidates = deduped

        return self._pad_to_ngm(candidates, list(pop), n_gm, min_dist=Rs_arch)

        return self._pad_to_ngm(candidates, list(pop), n_gm)

    # ------------------------------------------------------------------
    # Local-Convergence Fitness
    # ------------------------------------------------------------------

    @staticmethod
    def _lcf(pop: np.ndarray, val: np.ndarray, k: int) -> np.ndarray:
        """lcf(i) = # k-nearest neighbors (D-space) with strictly lower fitness."""
        N = len(pop)
        if N <= 1:
            return np.zeros(N, dtype=int)
        k = min(k, N - 1)
        dists = cdist(pop, pop)
        np.fill_diagonal(dists, np.inf)
        lcf = np.zeros(N, dtype=int)
        for i in range(N):
            nbr_idx = np.argpartition(dists[i], k)[:k]
            lcf[i] = int(np.sum(val[nbr_idx] < val[i]))
        return lcf

    @staticmethod
    def _lcf_query(trials: np.ndarray, tval: np.ndarray,
                   pop: np.ndarray, val: np.ndarray, k: int) -> np.ndarray:
        """lcf for trial points relative to existing pop."""
        n_t = len(trials)
        k = min(k, len(pop))
        if k == 0:
            return np.zeros(n_t, dtype=int)
        dists = cdist(trials, pop)
        lcf = np.zeros(n_t, dtype=int)
        for i in range(n_t):
            nbr_idx = np.argpartition(dists[i], k)[:k]
            lcf[i] = int(np.sum(val[nbr_idx] < tval[i]))
        return lcf

    # ------------------------------------------------------------------
    # Crowding Distance (decision space)
    # ------------------------------------------------------------------

    @staticmethod
    def _crowding_distance(pop: np.ndarray) -> np.ndarray:
        """Per-individual crowding distance in decision space."""
        N, dim = pop.shape
        if N <= 2:
            return np.full(N, np.inf)
        cd = np.zeros(N)
        for d in range(dim):
            order = np.argsort(pop[:, d])
            sv = pop[order, d]
            span = sv[-1] - sv[0]
            if span < 1e-12:
                continue
            cd[order[0]]  = np.inf
            cd[order[-1]] = np.inf
            for j in range(1, N - 1):
                cd[order[j]] += (sv[j + 1] - sv[j - 1]) / span
        return cd

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _lhc_init(self, n: int, lb: np.ndarray, ub: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
        dim = lb.size
        pop = np.empty((n, dim))
        for d in range(dim):
            cuts = np.linspace(lb[d], ub[d], n + 1)
            pop[:, d] = rng.uniform(cuts[:n], cuts[1:])
            rng.shuffle(pop[:, d])
        return pop

    # ------------------------------------------------------------------
    # DE operator
    # ------------------------------------------------------------------

    def _de_rand1(self, pop: np.ndarray, i: int,
                  lb: np.ndarray, ub: np.ndarray,
                  rng: np.random.Generator,
                  F: float = 0.8, CR: float = 0.3) -> np.ndarray:
        N = len(pop)
        idxs = rng.choice(N, 3, replace=False)
        r1, r2, r3 = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
        mutant = r1 + F * (r2 - r3)
        dim = pop.shape[1]
        mask = rng.random(dim) < CR
        if not mask.any():
            mask[rng.integers(dim)] = True
        trial = np.where(mask, mutant, pop[i])
        return self.reflect_to_bounds(trial, lb, ub)

    # ------------------------------------------------------------------
    # Nelder-Mead refinement
    # ------------------------------------------------------------------

    def _nm_refine(self, x0: np.ndarray, problem,
                   lb: np.ndarray, ub: np.ndarray,
                   budget: int) -> np.ndarray:
        evals = [0]

        def obj(x):
            if evals[0] >= budget or problem.used_eval >= problem.max_eval:
                raise StopIteration
            x_clip = np.clip(x, lb, ub)
            f = problem.func_eval(np.atleast_2d(x_clip))[0]
            evals[0] += 1
            return float(f)

        try:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": budget, "xatol": 1e-9,
                                    "fatol": 1e-11, "adaptive": True})
            return np.clip(res.x, lb, ub)
        except StopIteration:
            return np.clip(x0, lb, ub)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_ngm(candidates: list[np.ndarray],
                    fallback: list[np.ndarray],
                    n_gm: int,
                    min_dist: float = 1e-6) -> np.ndarray:
        """Fill to exactly n_gm solutions, avoiding near-duplicates."""
        result: list[np.ndarray] = list(candidates)
        for x in fallback:
            if len(result) >= n_gm:
                break
            if result:
                dists = np.linalg.norm(np.array(result) - x, axis=1)
                if float(np.min(dists)) < min_dist:
                    continue  # skip near-duplicate
            result.append(x.copy())
        if not result:
            raise RuntimeError("No candidates found — algorithm produced no output")
        while len(result) < n_gm:
            result.append(result[-1].copy())
        return np.array(result[:n_gm])
