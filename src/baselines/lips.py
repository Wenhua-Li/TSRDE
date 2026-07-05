"""
LIPS – Local Information Particle Swarm
Source: "A Novel Approach to Particle Swarms...", TEC 2013.

Core idea:
  Each particle uses only its k nearest personal-best neighbours (based on
  pbest positions) to form a local exemplar FIP, weighted by random
  coefficients per neighbour. Neighbourhood size grows gradually from 2 to 5
  over the run. The inertia weight is the canonical SPSO 2011 value (0.7298).
  Boundary: reflecting.

  This is a faithful Python translation of the MATLAB Lips.m file.
"""

import numpy as np
from .base_algorithm import BaseAlgorithm
from scipy.spatial.distance import cdist


class LIPS(BaseAlgorithm):

    def __init__(self,
                 pop_size: int = 100,
                 nb_min: int = 2,
                 nb_max: int = 5,
                 seed: int = None):
        super().__init__(pop_size=pop_size, nb_min=nb_min, nb_max=nb_max, seed=seed)

    @property
    def name(self):
        return "LIPS"

    # ------------------------------------------------------------------
    def _run(self, problem):
        rng = np.random.default_rng(self.params.get('seed'))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        NP = self.params['pop_size']
        nb_min = self.params['nb_min']
        nb_max = self.params['nb_max']

        # SPSO 2011 constriction coefficient
        cc = np.array([2.05, 2.05])
        phi = cc.sum()
        chi = 2.0 / abs(2.0 - phi - np.sqrt(phi**2 - 4*phi))   # ≈ 0.7298
        mv = 0.5 * (ub - lb)

        pos = self.random_init(NP, lb, ub, rng)
        vel = rng.uniform(-mv, mv, (NP, dim))
        val = problem.func_eval(pos)

        pbest = pos.copy()
        pbest_val = val.copy()

        max_eval = problem.max_eval

        while problem.used_eval < max_eval:
            progress = problem.used_eval / max_eval
            # neighbourhood size grows linearly
            nb = int(nb_min + progress * (nb_max - nb_min))
            nb = max(nb_min, min(nb_max, nb))
            nb = min(nb, NP - 1)

            # pairwise distances between pbest positions
            dist_mat = cdist(pbest, pbest)
            np.fill_diagonal(dist_mat, np.inf)

            for k in range(NP):
                if problem.used_eval >= max_eval:
                    break

                # find nb nearest pbest neighbours
                nbr_idx = np.argpartition(dist_mat[k], nb)[:nb]

                # compute FIP: weighted combination of neighbour pbests
                fi = rng.random((nb, dim)) * cc[0]         # random coefficients
                fip = np.sum(fi * pbest[nbr_idx], axis=0) / np.sum(fi, axis=0)

                # velocity update
                accel = np.sum(fi, axis=0) * (fip - pos[k])
                vel[k] = chi * (vel[k] + accel)

                # clamp velocity
                vel[k] = np.clip(vel[k], -mv, mv)

                # position update
                pos[k] = pos[k] + vel[k]

                # boundary: reflection
                pos[k] = self.reflect_to_bounds(pos[k], lb, ub)

                f_new = problem.func_eval(np.atleast_2d(pos[k]))[0]

                if f_new < pbest_val[k]:   # minimisation: lower is better (func returns 0-based shifted vals)
                    pbest[k] = pos[k].copy()
                    pbest_val[k] = f_new

        # Report: greedy peak extraction from pbest
        return self._extract_peaks(pbest, pbest_val, problem.n_minima)

    # ------------------------------------------------------------------
    def _extract_peaks(self, pop, val, n_gm):
        """Pick n_gm best solutions; prefer diversity."""
        order = np.argsort(val)   # ascending (minimisation)
        pop_s = pop[order]
        reported = []
        for i in range(len(pop_s)):
            if len(reported) >= n_gm:
                break
            cand = pop_s[i]
            if len(reported) == 0 or np.min(cdist([cand], reported)) > 1e-8:
                reported.append(cand)
        # pad if still short
        for i in range(len(pop_s)):
            if len(reported) >= n_gm:
                break
            reported.append(pop_s[i])
        return np.array(reported[:n_gm])
