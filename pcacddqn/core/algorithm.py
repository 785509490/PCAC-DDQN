"""Algorithm base class.

The driver owns the environment loop; an algorithm owns one generation of
search and an optional reaction to a change. Keeping the two apart is what
lets the same NSGA-II be run as a static baseline, as DNSGA-II, or as the
inner solver of the meta-framework without three copies of the driver.

`on_change` receives the population already evaluated in the NEW environment
(the driver re-evaluates it before calling), so a response policy can look at
what actually moved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .population import Population
from .problem import Problem


class Algorithm(ABC):
    name = "algorithm"

    def __init__(self, n_pop: int = 100) -> None:
        self.n_pop = int(n_pop)
        self.rng: np.random.Generator | None = None

    # -- lifecycle ---------------------------------------------------------

    def bind(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def initialize(self, problem: Problem, t: float) -> Population:
        rng = self._rng()
        X = problem.lower + rng.random((self.n_pop, problem.D)) * (
            problem.upper - problem.lower
        )
        return Population(X, problem.evaluate(X, t))

    @abstractmethod
    def step(self, problem: Problem, pop: Population, t: float) -> Population:
        """Advance one generation and return the new population."""

    def on_change(
        self,
        problem: Problem,
        pop: Population,
        t_prev: float,
        t_cur: float,
    ) -> Population:
        """React to a change. Default: do nothing (static baseline)."""
        return pop

    # -- helpers -----------------------------------------------------------

    def _rng(self) -> np.random.Generator:
        if self.rng is None:
            self.rng = np.random.default_rng(0)
        return self.rng
