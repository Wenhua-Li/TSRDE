"""
FOA – Forest Optimization Algorithm (Python translation of MATLAB original)
Source: MultiModal-FOA-master (MATLAB), translated to Python.

Core idea:
  Trees form a forest. Each iteration:
    1. Local Seeding: trees with Age=0 produce LSC offspring by perturbing
       LSC randomly chosen dimensions by ±dx.
    2. Population Limiting: trees exceeding Life_time are moved to a
       candidate pool; remaining trees are clustered, one representative
       per cluster is kept (cluster head Age reset to 0).
    3. Global Seeding: Transfer_rate% of candidates get GSC dimensions
       replaced with uniform random values → new trees injected.

  The forest naturally maintains multiple attractors (cluster heads)
  which correspond to different peaks.
"""

import numpy as np
from scipy.spatial.distance import cdist
from .base_algorithm import BaseAlgorithm


class FOA(BaseAlgorithm):

    def __init__(self,
                 area_limit: int = 100,   # max forest size after limiting
                 life_time: int = 10,     # max age before a tree becomes candidate
                 transfer_rate: float = 10.0,   # % of candidates for global seeding
                 lsc: int = 2,            # local seeding changes (dimensions perturbed)
                 gsc: int = 2,            # global seeding changes (dimensions replaced)
                 dx: float = 0.05,        # local step size (fraction of search range)
                 dist_threshold: float = None,   # clustering threshold; auto if None
                 seed: int = None):
        super().__init__(area_limit=area_limit, life_time=life_time,
                         transfer_rate=transfer_rate, lsc=lsc, gsc=gsc,
                         dx=dx, dist_threshold=dist_threshold, seed=seed)

    @property
    def name(self):
        return "FOA"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_seeding(self, forest, age_col, fit_col, lb, ub, dim, problem, rng):
        """Perturb trees with Age==0 along LSC random dimensions."""
        lsc = self.params['lsc']
        dx_abs = self.params['dx'] * (ub - lb)  # per-dimension step
        new_trees = []
        for i in range(len(forest)):
            if forest[i, age_col] == 0:
                for _ in range(lsc):
                    child = forest[i, :dim].copy()
                    d = rng.integers(dim)
                    child[d] += rng.uniform(-dx_abs[d], dx_abs[d])
                    child = np.clip(child, lb, ub)
                    if problem.used_eval >= problem.max_eval:
                        break
                    f = problem.func_eval(np.atleast_2d(child))[0]
                    row = np.zeros(dim + 2)
                    row[:dim] = child
                    row[fit_col] = f
                    row[age_col] = 0
                    new_trees.append(row)
        # age all existing trees by 1
        forest[:, age_col] += 1
        if new_trees:
            forest = np.vstack([forest, new_trees])
        return forest

    def _population_limiting(self, forest, age_col, fit_col, dim, dist_threshold):
        """Remove old trees, cluster survivors, keep one per cluster."""
        life_time = self.params['life_time']
        area_limit = self.params['area_limit']

        alive_mask = forest[:, age_col] < life_time
        candidates = forest[~alive_mask].copy()
        forest = forest[alive_mask].copy()

        if len(forest) == 0:
            return forest, candidates

        # cluster by fitness-ASCENDING (minimisation: best = lowest f first)
        order = np.argsort(forest[:, fit_col])   # ascending
        forest = forest[order]

        cluster_heads = [0]
        assignments = [0] * len(forest)
        for i in range(1, len(forest)):
            head_pos = forest[cluster_heads, :dim]
            dists = np.linalg.norm(head_pos - forest[i, :dim], axis=1)
            nearest_head = np.argmin(dists)
            if dists[nearest_head] <= dist_threshold:
                assignments[i] = cluster_heads[nearest_head]
            else:
                cluster_heads.append(i)
                assignments[i] = i

        # select trees: distribute area_limit slots proportionally across clusters
        n_clusters = len(cluster_heads)
        avg_per_cluster = max(1, area_limit // n_clusters)

        selected = []
        for ch in cluster_heads:
            members = [i for i, a in enumerate(assignments) if a == ch]
            n_take = min(avg_per_cluster, len(members))
            selected.extend(members[:n_take])
            # reset cluster-head age
            forest[ch, age_col] = 0

        selected = list(set(selected))
        if len(selected) > area_limit:
            selected = selected[:area_limit]

        not_selected = [i for i in range(len(forest)) if i not in selected]
        candidates = np.vstack([candidates, forest[not_selected]]) if not_selected else candidates
        forest = forest[selected]

        return forest, candidates

    def _global_seeding(self, candidates, forest, fit_col, age_col, dim, lb, ub, problem, rng):
        """Replace GSC dimensions of Transfer_rate% of candidates with random values."""
        if len(candidates) == 0:
            return np.empty((0, dim + 2))
        gsc = self.params['gsc']
        tr = self.params['transfer_rate'] / 100.0
        n_revo = max(1, int(len(candidates) * tr))
        chosen = candidates[:n_revo]
        new_trees = []
        for i in range(len(chosen)):
            if problem.used_eval >= problem.max_eval:
                break
            child = chosen[i, :dim].copy()
            dims = rng.choice(dim, gsc, replace=False)
            for d in dims:
                child[d] = rng.uniform(lb[d], ub[d])
            f = problem.func_eval(np.atleast_2d(child))[0]
            row = np.zeros(dim + 2)
            row[:dim] = child
            row[fit_col] = f
            row[age_col] = 0
            new_trees.append(row)
        return np.array(new_trees) if new_trees else np.empty((0, dim + 2))

    # ------------------------------------------------------------------
    def _run(self, problem):
        rng = np.random.default_rng(self.params.get('seed'))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        area_limit = self.params['area_limit']

        fit_col = dim       # index of fitness in forest row
        age_col = dim + 1   # index of age

        dist_threshold = self.params['dist_threshold']
        if dist_threshold is None:
            # default: ~half typical niche radius
            range_span = np.linalg.norm(ub - lb)
            dist_threshold = range_span / (2.0 * problem.n_minima ** (1.0 / dim))

        # initialise forest
        init_pop = self.random_init(area_limit, lb, ub, rng)
        init_f = problem.func_eval(init_pop)
        forest = np.zeros((area_limit, dim + 2))
        forest[:, :dim] = init_pop
        forest[:, fit_col] = init_f
        forest[:, age_col] = 0   # all age 0

        while problem.used_eval < problem.max_eval:
            # 1. Local seeding
            forest = self._local_seeding(forest, age_col, fit_col, lb, ub, dim, problem, rng)
            if problem.used_eval >= problem.max_eval:
                break

            # 2. Population limiting + clustering
            forest, candidates = self._population_limiting(forest, age_col, fit_col, dim, dist_threshold)

            # 3. Global seeding
            new_trees = self._global_seeding(candidates, forest, fit_col, age_col, dim, lb, ub, problem, rng)
            if len(new_trees) > 0:
                forest = np.vstack([forest, new_trees])

        # Report: greedily pick n_minima diverse best solutions from the final forest
        if len(forest) == 0:
            return self.random_init(problem.n_minima, lb, ub, rng)

        # sort by fitness ascending (minimisation: best = lowest f first)
        order = np.argsort(forest[:, fit_col])
        forest = forest[order]

        reported = []
        for row in forest:
            if len(reported) >= problem.n_minima:
                break
            cand = row[:dim]
            if len(reported) == 0 or np.min(cdist([cand], [r[:dim] for r in reported])) > 1e-8:
                reported.append(row[:dim].copy())
        # pad if needed
        for row in forest:
            if len(reported) >= problem.n_minima:
                break
            reported.append(row[:dim].copy())

        return np.array(reported[:problem.n_minima])
