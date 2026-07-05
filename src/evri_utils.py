"""
Robustness measure for TSRDE (Two-Stage Robust DE).

Definition (v0.14, consistent with PhD thesis):
  Robustness(x) = E_{δ ~ U(-sigma, sigma)^D}[f(x + δ)]
  Lower expected fitness = more robust (wide basin stays near peak under perturbation).

  Rationale: E[f(x+δ)] - f(x) ∝ Σ_d (1/s_d²)  [2nd power of inverse width].
  This correctly ranks anisotropic peaks by mean basin width, matching the
  ground-truth criterion mean(σ).  The previous Var metric was ∝ Σ(1/s_d⁴)
  [4th power], which is dominated by the single narrowest dimension and
  misranks peaks that are wide in D-1 dimensions but narrow in one.

sigma = 0.1 × Rs (where Rs is the archive deduplication radius,
derived from N_est and search range).

Sample count formula (theory-grounded):
  M*(D) = max(M_stat(D), M_cov(D))

  where
    M_stat(D) = ceil(z² · (2 − 6/(7D)) / ε²)   (asymptotic RSD bound)
    M_cov(D)  = 2D                                (directional coverage)

  Derivation sketch:
    Near a peak, f(x+δ) ≈ f* + ½ δᵀHδ.
    For δ_k ~ U(-σ,σ), excess kurtosis of Y = ½Σλ_k δ_k² satisfies
      κ_excess = −(6/7) · Σλ_k⁴ / (Σλ_k²)²
    For isotropic H: κ_excess = −6/(7D).
    Variance of sample variance: Var[Ŝ²] ≈ (κ_excess+2)·R²/M.
    Setting RSD ≤ ε at confidence 1−α gives M_stat.
    Coverage bound M_cov = 2D ensures ≥2 projections per dimension.

Default parameters: α=0.1 (z_{0.05}=1.645), ε=0.5.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Statistical parameters for M formula
# ---------------------------------------------------------------------------

_Z2   = 2.706   # z_{0.05}^2 = 1.645^2
_EPS2 = 0.25    # ε^2 = 0.5^2

SIGMA_FRAC = 0.1   # sigma = SIGMA_FRAC × Rs


def compute_M(D: int) -> int:
    """Return theory-based minimum sample count for D-dimensional neighbourhood sampling.

    Originally derived for variance-RSD bound; still valid for mean estimation
    since the same sample set is reused and M_cov=2D ensures directional coverage.
    Combines statistical bound and directional coverage bound.
    """
    kappa  = -6.0 / (7.0 * max(D, 1))          # isotropic near-peak κ_excess
    m_stat = int(np.ceil(_Z2 * (2.0 + kappa) / _EPS2))
    m_cov  = 2 * D
    return max(m_stat, m_cov, 1)


# ---------------------------------------------------------------------------
# Robustness computation
# ---------------------------------------------------------------------------


def compute_robustness(x: np.ndarray, problem, f_x: float,
                       Rs: float,
                       rng: np.random.Generator,
                       max_fe: int = 24) -> float:
    """
    Compute E[f(x + δ)] where δ ~ U(-sigma, sigma)^D.

    Using expected fitness (mean) rather than variance as the robustness proxy.
    Theoretical basis: E[f(x+δ)] - f(x) ≈ (h/2)(σ²/3)·Σ(1/s_d²), which is
    proportional to Σs_d⁻² — a 2nd-power average that correctly orders peaks
    by mean basin width. Variance ∝ Σs_d⁻⁴ is dominated by the single narrowest
    dimension, causing systematic misranking of anisotropic peaks (e.g. Group D).

    Parameters
    ----------
    x       : solution vector
    problem : problem with .func_eval, .low_bound, .up_bound
    f_x     : known fitness of x (avoids re-evaluation; unused, kept for API compat)
    Rs      : archive deduplication radius (scale reference)
    rng     : random generator
    max_fe  : FE budget for this call

    Returns
    -------
    float : expected neighbourhood fitness (lower = more robust, i.e. wider basin)
    """
    dim   = len(x)
    M     = compute_M(dim)
    sigma = SIGMA_FRAC * Rs

    n_new = min(M, max(0, max_fe))
    if n_new == 0 or problem.used_eval >= problem.max_eval - n_new:
        return 0.0

    new_pts = x + rng.uniform(-sigma, sigma, (n_new, dim))
    new_pts = np.clip(new_pts, problem.low_bound, problem.up_bound)
    samples_f = problem.func_eval(new_pts)

    if len(samples_f) == 0:
        return 0.0

    return float(np.mean(samples_f))


# ---------------------------------------------------------------------------
# Legacy alias — retained so existing imports don't break
# ---------------------------------------------------------------------------

def compute_evri(x, problem, f_x, Rs, range_, rng,
                 cache=None, max_fe=24):
    """Backward-compatible alias → calls compute_robustness (expected fitness)."""
    return compute_robustness(x, problem, f_x, Rs, rng, max_fe)
