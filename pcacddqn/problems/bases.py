"""Landscape bases: the structure of g(dev).

Every base must return exactly 1 when dev == 0. That single property is what
keeps the Pareto front analytic for *every* mode: the PS is always the set of
points with dev = 0, so g is 1 there regardless of how the landscape is shaped
away from it.

The old benchmark mixed this layer into `mode` (the `rugged` mode was really a
landscape change, not a PS change). The split here is deliberate.
"""

from __future__ import annotations

import numpy as np


class Base:
    name = "base"

    def configure(self, rng: np.random.Generator, D: int) -> None:
        """Draw instance-dependent constants (rotation, conditioning)."""

    def g(self, dev: np.ndarray, u: float) -> np.ndarray:
        raise NotImplementedError


class Quad(Base):
    name = "quad"

    def g(self, dev: np.ndarray, u: float) -> np.ndarray:
        return 1.0 + np.sum(dev ** 2, axis=1)


class Multi(Base):
    """Rastrigin term whose amplitude ramps to 0.25 over [0, 1].

    Why 0.25: the closed-loop amplitude was the ONE number the old benchmark
    never actually reached. It was written as 0.5(1 - cos(pi t/3)) on a t in
    [0, 3], but the runs stopped at t = 1, so the realised maximum was 0.25.
    Normalising the time axis without rescaling the amplitude lifts it to 1.0,
    and the measured effect is drastic: on C2 the MIGD goes 0.42 -> 11.8 when
    the amplitude goes 0.25 -> 1.0, because g reaches ~180 and the front is no
    longer reachable. 0.25 keeps the landscape the old numbers were tuned on.
    """

    name = "multi"

    def __init__(self, severity: float = 1.0, base_amp: float = 1.0):
        self.severity = float(severity)
        self.base_amp = float(base_amp)

    def g(self, dev: np.ndarray, u: float) -> np.ndarray:
        a = self.base_amp * self.severity * 0.5 * (1.0 - np.cos(np.pi * u))
        core = dev ** 2 + a * (dev ** 2 - 10.0 * np.cos(2.0 * np.pi * dev) + 10.0)
        return 1.0 + np.sum(core, axis=1)


class Link(Base):
    """Nonlinear linkage between neighbouring deviation coordinates."""

    name = "link"

    def g(self, dev: np.ndarray, u: float) -> np.ndarray:
        if dev.shape[1] < 2:
            return 1.0 + np.sum(dev ** 2, axis=1)
        pair = (dev[:, 1:] - dev[:, :-1] ** 2) ** 2
        return 1.0 + np.sum(pair, axis=1) + 0.1 * np.sum(dev ** 2, axis=1)


class Rot(Base):
    """Rotated, anisotropically conditioned quadratic.

    R and lambda are drawn per instance. They are part of the problem, not of
    the algorithm seed, so every algorithm on the same instance meets the same
    landscape.
    """

    name = "rot"

    def __init__(self):
        self.R: np.ndarray | None = None
        self.lam: np.ndarray | None = None

    def configure(self, rng: np.random.Generator, D: int) -> None:
        A = rng.normal(size=(D, D))
        Q, _ = np.linalg.qr(A)
        self.R = Q
        self.lam = np.exp(rng.uniform(-0.5, 0.5, size=D))

    def g(self, dev: np.ndarray, u: float) -> np.ndarray:
        if self.R is None:
            self.R = np.eye(dev.shape[1])
            self.lam = np.ones(dev.shape[1])
        rotated = dev @ self.R
        return 1.0 + np.sum(self.lam * rotated ** 2, axis=1)


BASES = {
    "quad": lambda severity=1.0, base_amp=1.0: Quad(),
    "multi": lambda severity=1.0, base_amp=1.0: Multi(severity=severity, base_amp=base_amp),
    "link": lambda severity=1.0, base_amp=1.0: Link(),
    "rot": lambda severity=1.0, base_amp=1.0: Rot(),
}
