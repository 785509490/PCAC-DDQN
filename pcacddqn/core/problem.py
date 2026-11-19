"""Problem base class.

A dynamic problem owns two things a static one does not:

* `t` is an explicit argument of `evaluate`, never hidden state. The same
  decision vector can therefore be scored in any environment without side
  effects, which is what the oracle sweeps rely on.
* `true_pf(t)` returns the reference front *for that environment*. A problem
  whose front moves must override it; the default falls back to the front at
  t = 0 and is only correct for Type I problems.

Instance identity and algorithm randomness are kept separate by construction:
a problem instance is fully determined by the seed passed to `reseed`, while
algorithms draw from their own generator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .metrics import nondominated_mask


class Problem(ABC):
    D: int = 10
    M: int = 2

    # Time grid. `n_env` environments means `nt = n_env - 1` changes, each
    # followed by `taut` generations; `T0` generations run before the first.
    n_env: int = 21
    nt: int = 20
    taut: int = 20
    T0: int = 50

    def __init__(self, seeded: bool = False) -> None:
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)
        if seeded:
            self.reseed(0)

    # -- interface ---------------------------------------------------------

    @abstractmethod
    def evaluate(self, X: np.ndarray, t: float) -> np.ndarray:
        """Return an (N, M) objective matrix for decision variables X at time t."""

    def pareto_set(self, t: float, n: int = 1000) -> np.ndarray:
        """Sample the Pareto set at time t. Used to build the reference front."""
        raise NotImplementedError(f"{type(self).__name__} has no analytic PS")

    def true_pf(self, t: float, n: int = 2000) -> np.ndarray:
        """Reference front at time t, filtered to the first non-dominated layer."""
        F = self.evaluate(self.pareto_set(t, n), t)
        return F[nondominated_mask(F)]

    def reseed(self, seed: int) -> None:
        """Re-draw everything that makes this a particular instance."""
        self.instance_seed = int(seed)

    # -- helpers -----------------------------------------------------------

    @property
    def environment_times(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_env)

    @property
    def max_fe(self) -> int:
        return self.T0 + self.nt * self.taut

    def clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower, self.upper)

    @staticmethod
    def time_to_u(t: float) -> float:
        """Environment time in [0, 1]. Laws are defined on this, not on t/3."""
        return float(t)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(D={self.D}, M={self.M}, n_env={self.n_env})"
