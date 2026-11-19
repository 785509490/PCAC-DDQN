"""NSGA-II: non-dominated sorting selection with a pluggable variation operator.

The default variation is DE, so every baseline in the comparison shares one
offspring generator and the algorithms differ only in how they react to a
change. Passing `variation="ga"` restores the classical SBX + polynomial
mutation version bit for bit.
"""

from __future__ import annotations

import numpy as np

from ..core.algorithm import Algorithm
from ..core.operators import select_nsga2
from ..core.population import Population
from ..core.problem import Problem
from ..core.variation import DEFAULT_VARIATION, make_variation


class NSGA2(Algorithm):
    def __init__(self, n_pop: int = 100, variation=DEFAULT_VARIATION, **variation_kwargs):
        super().__init__(n_pop)
        self.variation = make_variation(variation, **variation_kwargs)

    @property
    def name(self) -> str:
        return f"NSGA-II({self.variation.family.upper()})"

    def set_config(self, strategy: str, CR: float, F: float) -> None:
        """Mutate the DE configuration at a change point.

        This mirrors `GDE3.set_config`, so the meta-layer's inner solver can be
        swapped between the one-to-one GDE3 selection and NSGA-II selection
        without changing the driver. Only a DE variation can be configured.
        """
        if getattr(self.variation, "family", None) != "de":
            raise ValueError("set_config needs a DE variation")
        self.variation.strategy = strategy
        self.variation.CR = float(CR)
        self.variation.F = float(F)

    def step(self, problem: Problem, pop: Population, t: float) -> Population:
        rng = self._rng()
        children = self.variation.offspring(pop, problem, rng)
        child_obj = problem.evaluate(children, t)

        tag = None
        if pop.tag is not None:
            child_tag = _child_tags(pop, children, self.variation.family)
            tag = np.concatenate([pop.tag, child_tag])

        merged = Population(
            np.vstack([pop.dec, children]),
            np.vstack([pop.obj, child_obj]),
            tag=tag,
        )
        keep = select_nsga2(merged.obj, self.n_pop)
        return merged.take(keep)


def _child_tags(pop: Population, children: np.ndarray, family: str) -> np.ndarray:
    """Origin tag for every child.

    DE keeps the target index: child i is the trial of parent i (the binomial
    crossover always retains the target coordinate), so the tag is exact.
    GA crossover mixes two parents; attribute to the nearer one, which is a
    deterministic approximation that avoids threading parent indices through
    every variation class.
    """
    if family == "de":
        return pop.tag.copy()
    d = ((children[:, None, :] - pop.dec[None, :, :]) ** 2).sum(-1)
    return pop.tag[np.argmin(d, axis=1)]
