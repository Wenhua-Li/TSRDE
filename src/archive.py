"""
Archive: minimal capacity-bounded archive for multimodal optimization.
Stores distinct peak representatives, deduplicates by radius Rs.
"""
from __future__ import annotations
import numpy as np


class Archive:
    """Fixed-capacity archive with radius-based deduplication."""

    def __init__(self, capacity: int, Rs: float):
        self._cap = max(capacity, 1)
        self._Rs = Rs
        self._x: list[np.ndarray] = []
        self._f: list[float] = []

    # ------------------------------------------------------------------

    def try_insert(self, x: np.ndarray, f: float, f_opt: float = -np.inf) -> bool:
        """Insert x if it is distinct (dist > Rs) or improves nearest entry."""
        x = np.asarray(x, dtype=float).ravel()
        if len(self._x) == 0:
            self._x.append(x.copy())
            self._f.append(float(f))
            return True

        dists = np.array([np.linalg.norm(x - xi) for xi in self._x])
        near = int(np.argmin(dists))

        if dists[near] < self._Rs:
            # Same basin: keep better
            if f < self._f[near]:
                self._x[near] = x.copy()
                self._f[near] = float(f)
            return False

        # New basin
        if len(self._x) < self._cap:
            self._x.append(x.copy())
            self._f.append(float(f))
            return True

        # Over capacity: replace worst
        worst = int(np.argmax(self._f))
        if f < self._f[worst]:
            self._x[worst] = x.copy()
            self._f[worst] = float(f)
            return True
        return False

    def get_top(self, n: int) -> np.ndarray:
        """Return up to n best entries sorted by fitness."""
        if not self._x:
            return np.empty((0, 1))
        order = np.argsort(self._f)[:n]
        return np.array([self._x[i] for i in order])

    def __len__(self):
        return len(self._x)
