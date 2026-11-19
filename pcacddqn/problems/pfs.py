"""Pareto-front geometry and front-change operators.

For two objectives the raw front is the graph `f2 = F(u)` with `u = f1` on the
Pareto set. A change operator is then a map on the objective vector, applied
identically inside `evaluate` and inside `true_pf`, so the reference front can
never drift away from the front the algorithm actually sees.

The front change is driven by the SAME m(t) as the Pareto set. Sharing the
schedule is a deliberate choice: with two independent time tables it would be
impossible to say which layer caused an observed response.
"""

from __future__ import annotations

import numpy as np


class Geometry:
    name = "geometry"

    def front(self, u: np.ndarray, f1: np.ndarray | None = None) -> np.ndarray:
        """Raw f2 values for u = f1/g in [0, 1].

        `f1` is the first objective, needed by geometries whose shape depends
        on it rather than on f1/g (DF9/ZDT3 style).
        """
        raise NotImplementedError


class Convex(Geometry):
    name = "convex"

    def front(self, u: np.ndarray, f1: np.ndarray | None = None) -> np.ndarray:
        return 1.0 - np.sqrt(np.clip(u, 0.0, 1.0))


class Concave(Geometry):
    name = "concave"

    def front(self, u: np.ndarray, f1: np.ndarray | None = None) -> np.ndarray:
        return np.sqrt(np.clip(1.0 - u ** 2, 0.0, 1.0))


class Linear(Geometry):
    name = "linear"

    def front(self, u: np.ndarray, f1: np.ndarray | None = None) -> np.ndarray:
        return 1.0 - np.clip(u, 0.0, 1.0)


class Disconnected(Geometry):
    """ZDT3-style front.

    The oscillatory term is a function of f1, not of f1/g: that is what keeps
    f2 increasing in g and makes `dev = 0` the true Pareto set. Using f1/g in
    the sine -- as a naive port would -- lets a point with g > 1 dominate a
    point on the nominal set, and the PS is then wrong.
    """

    name = "disconnected"

    def front(self, u: np.ndarray, f1: np.ndarray | None = None) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        arg = u if f1 is None else np.clip(f1, 0.0, 1.0)
        # ZDT3: h = 1 - sqrt(f1/g) - (f1/g) sin(10 pi f1). The sine argument is
        # f1 while the prefactor is u = f1/g; getting that prefactor wrong makes
        # f2 decrease in g and breaks the Pareto set.
        return 1.0 - np.sqrt(u) - u * np.sin(10.0 * np.pi * arg)


class FrontChange:
    name = "none"

    def apply(self, F: np.ndarray, m: float) -> np.ndarray:
        return F


class Shift(FrontChange):
    name = "shift"

    def __init__(self, amount: float = 0.1):
        self.amount = float(amount)

    def apply(self, F: np.ndarray, m: float) -> np.ndarray:
        return F + self.amount * m


class Scale(FrontChange):
    name = "scale"

    def __init__(self, amount: float = 0.2):
        self.amount = float(amount)

    def apply(self, F: np.ndarray, m: float) -> np.ndarray:
        return F * (1.0 + self.amount * m)


class Curvature(FrontChange):
    """Monotone power warp `F <- sign(F) |F|^H`, identity at m = 0.

    Do NOT clip to [0, 1] before warping. Away from the Pareto set
    `f2 = g h(f1/g)` exceeds 1, and clipping turned every such point into the
    same plateau value -- which erased the gradient, let NSGA-II collapse onto a
    single point, and left three algorithms with bit-identical trajectories.
    The signed power preserves dominance for every real F.
    """

    name = "curvature"

    def __init__(self, amount: float = 0.75):
        self.amount = float(amount)

    def apply(self, F: np.ndarray, m: float) -> np.ndarray:
        H = 1.0 + self.amount * m
        return np.sign(F) * np.abs(F) ** H


GEOMETRIES = {
    "convex": lambda: Convex(),
    "concave": lambda: Concave(),
    "linear": lambda: Linear(),
    "disconnected": lambda: Disconnected(),
    "sphere": None,          # three-objective sphere, handled by the problem
}

CHANGES = {
    "none": lambda **kw: FrontChange(),
    "shift": lambda **kw: Shift(),
    "scale": lambda **kw: Scale(),
    "curvature": lambda **kw: Curvature(),
}
