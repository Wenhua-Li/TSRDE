"""
EC-GECCO2020 algorithm: FastNichingEP

Uses Fast Evolutionary Programming (FEP) mutation (Cauchy + self-adaptive eta)
and round-robin tournament selection, adapted from EC-GECCO2020-master.

FastNichingEP:
  Repeatedly re-initialises a fresh population, evolves it, and extracts
  the best peak each round. Already-found niches are repelled (fitness
  penalty) to push subsequent rounds towards undiscovered peaks.
"""

import numpy as np
from scipy.spatial.distance import cdist
from .base_algorithm import BaseAlgorithm


# ======================================================================
# Shared low-level operators (faithful Python translations of originals)
# ======================================================================

def _fep_mutate(individual, eta, lb, ub, rng, max_try=10):
    """Fast EP mutation: Cauchy step scaled by self-adaptive eta."""
    dim = len(individual)
    mutant = individual.copy()
    new_eta = eta.copy()
    for i in range(dim):
        for _ in range(max_try):
            step = eta[i] * rng.standard_cauchy()
            cand = individual[i] + step
            if lb[i] <= cand <= ub[i]:
                mutant[i] = cand
                break
        else:
            mutant[i] = np.clip(individual[i], lb[i], ub[i])
    # update eta with bounds: [1e-6, 0.2*(ub-lb)] to prevent explosion
    tau = 1.0 / np.sqrt(2 * dim)
    tau_prime = 1.0 / np.sqrt(2 * np.sqrt(dim))
    new_eta = eta * np.exp(tau * rng.standard_normal() +
                           tau_prime * rng.standard_normal(dim))
    eta_max = 0.2 * (ub - lb)   # upper bound: 20% of search range per dim
    new_eta = np.clip(new_eta, 1e-6, eta_max)
    return mutant, new_eta


def _round_robin_tournament(pop, new_pop, fit, new_fit, eta, new_eta, q, rng):
    """Round-robin tournament: keep top-N from (pop ∪ new_pop) by win count.
    Minimisation: an individual 'wins' when its fitness is LOWER than the rival's."""
    N = len(pop)
    all_pop = np.vstack([pop, new_pop])
    all_fit = np.concatenate([fit, new_fit])
    all_eta = np.vstack([eta, new_eta])
    total = len(all_pop)

    wins = np.zeros(total, dtype=int)
    for i in range(total):
        rivals = rng.choice(total, size=q, replace=False)
        wins[i] = np.sum(all_fit[i] < all_fit[rivals])   # lower f wins

    survivors = np.argsort(wins)[::-1][:N]
    return all_pop[survivors].copy(), all_fit[survivors].copy(), all_eta[survivors].copy()


# ======================================================================
# FastNichingEP
# ======================================================================

class FastNichingEP(BaseAlgorithm):
    """
    Sequential niche-finding with FEP.
    Divides total budget into `n_niches` sub-runs, each finding one peak.
    Already-found niches penalise fitness, steering later runs away.
    """

    def __init__(self,
                 pop_size: int = 20,
                 q: int = 10,             # round-robin rivals
                 niche_radius: float = None,
                 seed: int = None):
        super().__init__(pop_size=pop_size, q=q,
                         niche_radius=niche_radius, seed=seed)

    @property
    def name(self):
        return "FastNichingEP"

    def _run(self, problem):
        rng = np.random.default_rng(self.params.get('seed'))
        lb, ub = problem.low_bound, problem.up_bound
        dim = problem.dim
        NP = self.params['pop_size']
        q = self.params['q']
        n_niches = problem.n_minima

        niche_radius = self.params['niche_radius']
        if niche_radius is None:
            niche_radius = np.linalg.norm(ub - lb) / 20.0

        budget_per_niche = problem.max_eval // n_niches
        found_peaks = np.empty((0, dim))
        worst_fitness = None

        for _ in range(n_niches):
            if problem.used_eval >= problem.max_eval:
                break

            pop = self.random_init(NP, lb, ub, rng)
            eta = np.ones((NP, dim))
            fit = self._get_penalised_fitness(pop, found_peaks, niche_radius,
                                              worst_fitness, problem)
            if worst_fitness is None and len(fit) > 0:
                worst_fitness = float(np.max(fit))   # worst = highest f (minimisation penalty)

            local_budget = min(budget_per_niche,
                               problem.max_eval - problem.used_eval)
            local_used = NP  # already evaluated during init

            while local_used < local_budget and problem.used_eval < problem.max_eval:
                new_pop = np.empty_like(pop)
                new_eta = np.empty_like(eta)
                for i in range(NP):
                    new_pop[i], new_eta[i] = _fep_mutate(pop[i], eta[i], lb, ub, rng)

                new_fit = self._get_penalised_fitness(new_pop, found_peaks, niche_radius,
                                                      worst_fitness, problem)
                local_used += NP
                pop, fit, eta = _round_robin_tournament(pop, new_pop, fit, new_fit,
                                                         eta, new_eta, q, rng)

            best_idx = np.argmin(fit)   # minimisation: lowest f = best peak
            found_peaks = np.vstack([found_peaks, pop[best_idx]])

        # pad if fewer peaks found than expected
        if len(found_peaks) < n_niches:
            extra = self.random_init(n_niches - len(found_peaks), lb, ub, rng)
            found_peaks = np.vstack([found_peaks, extra])

        return found_peaks[:n_niches]

    def _get_penalised_fitness(self, pop, found_peaks, niche_radius, worst_fitness, problem):
        fit = np.zeros(len(pop))
        for i, ind in enumerate(pop):
            if len(found_peaks) > 0 and \
               np.min(cdist([ind], found_peaks)) < niche_radius:
                fit[i] = worst_fitness if worst_fitness is not None else 0.0
            else:
                if problem.used_eval >= problem.max_eval:
                    fit[i] = worst_fitness if worst_fitness is not None else 0.0
                else:
                    fit[i] = problem.func_eval(np.atleast_2d(ind))[0]
        return fit
