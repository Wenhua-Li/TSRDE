"""
RobBench: A 20-problem synthetic benchmark for robust multimodal optimization.

Design invariant: all peaks share equal height h=10, so fitness-only algorithms
achieve a random baseline RobPR = n_robust / n_peaks. Ground-truth robustness is
determined by basin width: R(k)=1 iff mean(s_k) ranks in the top n_robust peaks.

Unified function (minimization):
  f(x) = min_{k=1..K} { -h * exp(-0.5 * sum_d (x_d - c_{k,d})^2 / s_{k,d}^2) }

Five groups (4 problems each, 20 total):
  A  Dimensionality scaling       D in {2, 3, 5, 10}, K=4, n_rob=2
  B  Robustness fraction          D=5, K in {4,6},    n_rob/K in {1/4,1/3,1/2,3/4}
  C  Peak multiplicity            D=5, K in {2,4,6,8}, n_rob=K//2
  D  Anisotropic basin geometry   D in {5,10}, K in {4,6}, n_rob in {2,3}
  E  Spatial trap (deceptive)     D in {5,10}, K in {6,8}, n_rob in {1,2}

Difficulty calibration (v2, 2026-06-11):
  Narrow-peak sigma raised from 0.10 → 0.30 globally, and wide-peak sigma
  raised from 1.20 → 1.50, reducing the discrimination ratio from 12x to 5x.
  This makes the benchmark tractable for Track-A MMO algorithms while keeping
  meaningful robustness contrast for Track-B evaluation.
  _WIDTHS_4 sequence: [0.30, 0.50, 0.80, 1.50]  (was [0.10, 0.30, 0.60, 1.20])

Backward compatibility:
  Original RobBench-1 -> RB-A1  (2D, 4 peaks, identical peak definitions)
  Original RobBench-2 -> RB-A3  (5D, 4 peaks, same widths)
  Original RobBench-3 (10D 2-peak) concept -> RB-C1 (5D); RB-A4 covers 10D

API mirrors the original RobBenchProblem:
  .func_eval(X) -> ndarray    evaluate; increments .used_eval
  .is_robust_peak(k) -> bool  ground-truth label
  .peak_info() -> list[dict]  per-peak metadata
  .minima.X, .minima.f, .minima.niche_rad, .minima.n_minima
  .gt_robustness  ndarray of mean-sigma per peak (larger = more robust)
"""
from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Core problem class
# ---------------------------------------------------------------------------

class RobBenchProblem:
    """Synthetic equal-height multi-Gaussian problem with analytically known robust peaks."""

    def __init__(
        self,
        name: str,
        dim: int,
        peaks: list[dict],              # each dict: {'center': ndarray, 'sigma': ndarray}
        height: float = 10.0,
        bounds: tuple[float, float] = (-3.0, 3.0),
        max_eval_factor: int = 20_000,
        n_robust_override: int | None = None,   # explicit n_robust; overrides top-half default
    ):
        self.name = name
        self.pid  = name
        self.dim  = dim
        self.low_bound = np.full(dim, bounds[0])
        self.up_bound  = np.full(dim, bounds[1])
        self.max_eval  = max_eval_factor * dim
        self.used_eval = 0
        self._peaks    = peaks
        self._height   = float(height)
        self.n_minima  = len(peaks)

        # Niche radii: half the nearest-neighbour distance between peak centres
        centers = np.array([pk['center'] for pk in peaks])
        if len(peaks) > 1:
            D = cdist(centers, centers)
            np.fill_diagonal(D, np.inf)
            niche_rad = D.min(axis=0) / 2.0
        else:
            niche_rad = np.full(1, np.linalg.norm(self.up_bound - self.low_bound) / 4.0)

        self.minima = SimpleNamespace(
            X=centers,
            f=-height,
            niche_rad=niche_rad,
            n_minima=len(peaks),
        )

        # Ground-truth robustness: top n_robust peaks by mean sigma (wider = more robust)
        self.gt_robustness = np.array([float(np.mean(pk['sigma'])) for pk in peaks])
        order    = np.argsort(self.gt_robustness)[::-1]   # widest first
        n_robust = (n_robust_override if n_robust_override is not None
                    else max(1, len(peaks) // 2))
        self._robust_idx = set(order[:n_robust].tolist())

    # ------------------------------------------------------------------

    def func_eval(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        n = X.shape[0]
        out = np.empty(n)
        for i in range(n):
            v = 0.0
            for pk in self._peaks:
                z = (X[i] - pk['center']) / pk['sigma']
                v = min(v, -self._height * np.exp(-0.5 * float(np.dot(z, z))))
            out[i] = v
        self.used_eval += n
        return out

    def is_robust_peak(self, peak_idx: int) -> bool:
        return peak_idx in self._robust_idx

    def peak_info(self) -> list[dict]:
        return [
            {
                'peak_idx': k,
                'center':   pk['center'].tolist(),
                'sigma':    (pk['sigma'].tolist() if hasattr(pk['sigma'], 'tolist')
                             else list(pk['sigma'])),
                'gt_robust': float(np.mean(pk['sigma'])),
                'is_robust': self.is_robust_peak(k),
            }
            for k, pk in enumerate(self._peaks)
        ]


# ---------------------------------------------------------------------------
# Peak helpers
# ---------------------------------------------------------------------------

def _iso(center: np.ndarray, sigma: float) -> dict:
    """Isotropic peak: same sigma in all dimensions."""
    return {'center': np.asarray(center, dtype=float),
            'sigma':  np.full(len(center), float(sigma))}


def _aniso(center: np.ndarray, sigma: list | np.ndarray) -> dict:
    """Anisotropic peak: per-dimension sigma vector."""
    return {'center': np.asarray(center, dtype=float),
            'sigma':  np.asarray(sigma, dtype=float)}


# ---------------------------------------------------------------------------
# Group A: Dimensionality scaling  (K=4, n_rob=2, iso, sigma=[0.10,0.30,0.60,1.20])
# Discrimination ratio = 1.20/0.10 = 12x
# ---------------------------------------------------------------------------

# Difficulty calibration v2: narrow=0.30, wide=1.50 → discrimination ratio 5x
# (was [0.10, 0.30, 0.60, 1.20], ratio 12x — too hard for Track-A MMO algorithms)
_WIDTHS_4 = [0.30, 0.50, 0.80, 1.50]   # shared width sequence for A/B/C isotropic problems


def _make_A(name: str, dim: int, seed: int) -> RobBenchProblem:
    rng     = np.random.default_rng(seed)
    centers = rng.uniform(-1.8, 1.8, (4, dim))
    peaks   = [_iso(centers[i], _WIDTHS_4[i]) for i in range(4)]
    return RobBenchProblem(name, dim, peaks)


def make_A1() -> RobBenchProblem:
    """RB-A1 (2D, 4 peaks) — original geometry, widths updated to v2 calibration."""
    peaks = [
        _aniso(np.array([-1.5,  0.0]), [0.30, 0.30]),
        _aniso(np.array([ 1.5,  0.0]), [0.50, 0.50]),
        _aniso(np.array([ 0.0, -1.5]), [0.40, 0.80]),   # anisotropic, mean=0.60
        _aniso(np.array([ 0.0,  1.5]), [1.50, 1.50]),
    ]
    return RobBenchProblem('RB-A1', dim=2, peaks=peaks)


def make_A2() -> RobBenchProblem:
    """RB-A2 (3D, 4 peaks) — intermediate dimension."""
    rng     = np.random.default_rng(15)
    centers = rng.uniform(-2.5, 2.5, (4, 3))
    peaks   = [_iso(centers[i], _WIDTHS_4[i]) for i in range(4)]
    return RobBenchProblem('RB-A2', dim=3, peaks=peaks, bounds=(-3.5, 3.5))


def make_A3() -> RobBenchProblem:
    """RB-A3 (5D, 4 peaks) — mirrors original RobBench-2."""
    return _make_A('RB-A3', dim=5, seed=1)


def make_A4() -> RobBenchProblem:
    """RB-A4 (10D, 4 peaks) — high-dimensional scaling test."""
    return _make_A('RB-A4', dim=10, seed=12)


# ---------------------------------------------------------------------------
# Group B: Robustness fraction  (D=5, n_robust/K in {1/4, 1/3, 1/2, 3/4})
# ---------------------------------------------------------------------------

def make_B1() -> RobBenchProblem:
    """RB-B1 (5D, 4 peaks, n_rob=1) — 1/4 fraction; one wide peak among three narrow."""
    rng     = np.random.default_rng(2)
    centers = rng.uniform(-1.8, 1.8, (4, 5))
    peaks   = [_iso(centers[i], s) for i, s in enumerate([0.30, 0.45, 0.60, 1.50])]
    return RobBenchProblem('RB-B1', dim=5, peaks=peaks, n_robust_override=1)


def make_B2() -> RobBenchProblem:
    """RB-B2 (5D, 6 peaks, n_rob=2) — 1/3 fraction."""
    rng     = np.random.default_rng(25)
    centers = rng.uniform(-1.8, 1.8, (6, 5))
    peaks   = [_iso(centers[i], s) for i, s in
               enumerate([0.30, 0.38, 0.45, 0.60, 0.90, 1.50])]
    return RobBenchProblem('RB-B2', dim=5, peaks=peaks, n_robust_override=2)


def make_B3() -> RobBenchProblem:
    """RB-B3 (5D, 4 peaks, n_rob=2) — 1/2 fraction; Group B anchor."""
    rng     = np.random.default_rng(22)
    centers = rng.uniform(-1.8, 1.8, (4, 5))
    peaks   = [_iso(centers[i], _WIDTHS_4[i]) for i in range(4)]
    return RobBenchProblem('RB-B3', dim=5, peaks=peaks)


def make_B4() -> RobBenchProblem:
    """RB-B4 (5D, 4 peaks, n_rob=3) — 3/4 fraction; only one narrow peak."""
    rng     = np.random.default_rng(22)
    centers = rng.uniform(-1.8, 1.8, (4, 5))
    peaks   = [_iso(centers[i], s) for i, s in enumerate([0.30, 0.80, 1.00, 1.50])]
    return RobBenchProblem('RB-B4', dim=5, peaks=peaks, n_robust_override=3)


# ---------------------------------------------------------------------------
# Group C: Peak multiplicity  (D=5, K in {2,4,6,8}, n_rob=K//2)
# ---------------------------------------------------------------------------

def make_C1() -> RobBenchProblem:
    """RB-C1 (5D, 2 peaks) — minimal case: one sharp, one flat."""
    rng     = np.random.default_rng(3)
    centers = rng.uniform(-1.8, 1.8, (2, 5))
    peaks   = [_iso(centers[0], 0.35), _iso(centers[1], 1.00)]
    return RobBenchProblem('RB-C1', dim=5, peaks=peaks)


def make_C2() -> RobBenchProblem:
    """RB-C2 (5D, 4 peaks) — Group C anchor."""
    rng     = np.random.default_rng(30)
    centers = rng.uniform(-1.8, 1.8, (4, 5))
    peaks   = [_iso(centers[i], _WIDTHS_4[i]) for i in range(4)]
    return RobBenchProblem('RB-C2', dim=5, peaks=peaks)


def make_C3() -> RobBenchProblem:
    """RB-C3 (5D, 6 peaks, n_rob=3)."""
    rng     = np.random.default_rng(31)
    centers = rng.uniform(-2.5, 2.5, (6, 5))
    widths  = [0.30, 0.40, 0.50, 0.70, 1.00, 1.50]
    peaks   = [_iso(centers[i], widths[i]) for i in range(6)]
    return RobBenchProblem('RB-C3', dim=5, peaks=peaks, bounds=(-4.0, 4.0))


def make_C4() -> RobBenchProblem:
    """RB-C4 (5D, 8 peaks, n_rob=4) — largest pool; extended bounds to fit all peaks."""
    rng     = np.random.default_rng(34)
    centers = rng.uniform(-2.5, 2.5, (8, 5))
    widths  = [0.30, 0.35, 0.40, 0.45, 0.55, 0.70, 1.00, 1.50]
    peaks   = [_iso(centers[i], widths[i]) for i in range(8)]
    return RobBenchProblem('RB-C4', dim=5, peaks=peaks, bounds=(-4.0, 4.0))


# ---------------------------------------------------------------------------
# Group D: Anisotropic sigma
# ---------------------------------------------------------------------------

def make_D1() -> RobBenchProblem:
    """RB-D1 (5D, 4 peaks) — mild anisotropy: robust peak wide in 4 dims, narrow in 1."""
    rng     = np.random.default_rng(4)
    centers = rng.uniform(-1.8, 1.8, (4, 5))
    peaks = [
        _iso(centers[0],   0.30),
        _iso(centers[1],   0.45),
        _aniso(centers[2], [0.80, 0.80, 0.80, 0.80, 0.30]),   # robust, mean=0.70
        _iso(centers[3],   1.20),                              # robust
    ]
    return RobBenchProblem('RB-D1', dim=5, peaks=peaks)


def make_D2() -> RobBenchProblem:
    """RB-D2 (10D, 4 peaks) — strong anisotropy: robust peak wide in first 8 of 10 dims."""
    rng     = np.random.default_rng(41)
    centers = rng.uniform(-1.8, 1.8, (4, 10))
    peaks = [
        _iso(centers[0], 0.30),
        _iso(centers[1], 0.45),
        _aniso(centers[2], np.array([0.80]*8 + [0.30]*2)),    # robust, mean=0.70
        _iso(centers[3], 1.20),                               # robust
    ]
    return RobBenchProblem('RB-D2', dim=10, peaks=peaks)


def make_D3() -> RobBenchProblem:
    """RB-D3 (5D, 4 peaks) — structured 2x2 grid in first 2 dims; tests spatial structure."""
    c = np.zeros((4, 5))
    c[0, :2] = [-1.5, -1.5]
    c[1, :2] = [ 1.5, -1.5]
    c[2, :2] = [-1.5,  1.5]
    c[3, :2] = [ 1.5,  1.5]
    peaks = [
        _iso(c[0], 0.30),
        _iso(c[1], 0.50),    # robust, mean=0.50
        _iso(c[2], 0.30),
        _iso(c[3], 1.20),    # robust
    ]
    return RobBenchProblem('RB-D3', dim=5, peaks=peaks)


def make_D4() -> RobBenchProblem:
    """RB-D4 (10D, 6 peaks, n_rob=3) — mixed iso/aniso peaks; random centres."""
    rng     = np.random.default_rng(41)
    centers = rng.uniform(-1.8, 1.8, (6, 10))
    peaks = [
        _iso(centers[0], 0.30),
        _iso(centers[1], 0.40),
        _iso(centers[2], 0.50),
        _iso(centers[3], 0.70),                                # robust, iso
        _aniso(centers[4], np.array([0.80]*5 + [0.50]*5)),    # robust, mean=0.65
        _iso(centers[5], 1.20),                               # robust
    ]
    return RobBenchProblem('RB-D4', dim=10, peaks=peaks, n_robust_override=3)


# ---------------------------------------------------------------------------
# Group E: Spatial trap — clustered narrow peaks + hidden wide peaks
# Random baseline = n_rob/K = 0.125 to 0.333 (most discriminating group)
# ---------------------------------------------------------------------------

def make_E1() -> RobBenchProblem:
    """RB-E1 (5D, 8 peaks, n_rob=1) — 7 narrow peaks clustered + 1 wide peak far away."""
    rng     = np.random.default_rng(5)
    cluster = rng.uniform(-0.8, 0.8, (7, 5))
    wide    = rng.uniform( 1.8, 2.5, (1, 5))[0]
    peaks   = [_iso(cluster[i], 0.30) for i in range(7)]
    peaks.append(_iso(wide, 1.20))
    return RobBenchProblem('RB-E1', dim=5, peaks=peaks, n_robust_override=1)


def make_E2() -> RobBenchProblem:
    """RB-E2 (10D, 8 peaks, n_rob=2) — 6 narrow peaks near origin + 2 wide at corners."""
    rng    = np.random.default_rng(6)
    narrow = rng.uniform(-1.0, 1.0, (6, 10))
    peaks  = [_iso(narrow[i], 0.30) for i in range(6)]
    peaks.append(_iso(np.full(10,  2.2), 0.80))
    peaks.append(_iso(np.full(10, -2.2), 1.00))
    return RobBenchProblem('RB-E2', dim=10, peaks=peaks,
                           bounds=(-3.5, 3.5), n_robust_override=2)


def make_E3() -> RobBenchProblem:
    """RB-E3 (5D, 6 peaks, n_rob=1) — 5 narrow peaks spread + 1 very wide peak isolated."""
    rng    = np.random.default_rng(7)
    narrow = rng.uniform(-1.0, 1.0, (5, 5))
    peaks  = [_iso(narrow[i], 0.32) for i in range(5)]
    peaks.append(_iso(np.full(5, 2.2), 1.50))
    return RobBenchProblem('RB-E3', dim=5, peaks=peaks,
                           bounds=(-3.5, 3.5), n_robust_override=1)


def make_E4() -> RobBenchProblem:
    """RB-E4 (10D, 6 peaks, n_rob=2) — 4 narrow clustered + 2 wide at alternating corners."""
    rng    = np.random.default_rng(8)
    narrow = rng.uniform(-0.5, 0.5, (4, 10))
    c_pos  = np.tile([2.0, -2.0], 5)
    c_neg  = np.tile([-2.0, 2.0], 5)
    peaks  = [_iso(narrow[i], 0.30) for i in range(4)]
    peaks.append(_iso(c_pos, 1.00))
    peaks.append(_iso(c_neg, 0.80))
    return RobBenchProblem('RB-E4', dim=10, peaks=peaks,
                           bounds=(-3.5, 3.5), n_robust_override=2)


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

ALL_PROBLEMS: list[tuple[str, callable]] = [
    ("RB-A1", make_A1), ("RB-A2", make_A2), ("RB-A3", make_A3), ("RB-A4", make_A4),
    ("RB-B1", make_B1), ("RB-B2", make_B2), ("RB-B3", make_B3), ("RB-B4", make_B4),
    ("RB-C1", make_C1), ("RB-C2", make_C2), ("RB-C3", make_C3), ("RB-C4", make_C4),
    ("RB-D1", make_D1), ("RB-D2", make_D2), ("RB-D3", make_D3), ("RB-D4", make_D4),
    ("RB-E1", make_E1), ("RB-E2", make_E2), ("RB-E3", make_E3), ("RB-E4", make_E4),
]

# Legacy aliases — original factory names still work
make_robbench_1 = make_A1
make_robbench_2 = make_A3
make_robbench_3 = make_C1
