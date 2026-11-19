"""Variation operators as pluggable objects.

The algorithm layer owns *selection* and the reaction to change; the variation
layer owns *how offspring are made*. Keeping them apart is what lets one
NSGA-II loop run with differential evolution, with SBX plus polynomial
mutation, or with a family added later, without three copies of the loop.

It is also the axis the meta-layer will grow into. The current action space is
(DE configuration x response); the next step is (operator family x parameters x
response), and that only requires a new entry in `FAMILIES` plus a longer
`config_space`, not a rewrite of the drivers.

Every operator takes the whole population and an explicit generator, so the run
seed remains the only source of algorithmic randomness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .metrics import crowding_distance, ndsort
from .operators import (
    de_mutate,
    polynomial_mutation,
    random_pairs,
    sbx_crossover,
    tournament,
)
from .population import Population
from .problem import Problem

# --- DE configuration space (45, the frozen action axis) -------------------

DE_STRATEGIES = (
    "rand/1",
    "best/1",
    "current-to-best/1",
    "rand/2",
    "current-to-rand/1",
)
DE_CR_VALUES = (0.1, 0.5, 0.9)
DE_F_VALUES = (0.3, 0.5, 0.8)

# --- GA configuration space (9, available but not yet in the action space) --

GA_ETA_C_VALUES = (5.0, 15.0, 20.0)
GA_ETA_M_VALUES = (5.0, 15.0, 20.0)

PARAM_NAMES = {
    "de": ("strategy", "CR", "F"),
    "ga": ("eta_c", "eta_m"),
}


class Variation(ABC):
    """Produces one generation of offspring from the current population."""

    family = "base"

    @abstractmethod
    def offspring(
        self, pop: Population, problem: Problem, rng: np.random.Generator
    ) -> np.ndarray:
        """Return an (n, D) block of offspring decision vectors."""

    def config(self) -> tuple:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.family}:{self.config()}"


class DEVariation(Variation):
    """DE mutation plus binomial crossover.

    `best` is chosen from the current population as the f1-extreme. GDE3 has no
    single best vector, and this choice is deterministic and history-free,
    which keeps the operator reproducible.
    """

    family = "de"

    def __init__(self, strategy: str = "rand/1", CR: float = 0.9, F: float = 0.5):
        if strategy not in DE_STRATEGIES:
            raise ValueError(f"unknown DE strategy {strategy!r}")
        self.strategy = strategy
        self.CR = float(CR)
        self.F = float(F)

    def config(self) -> tuple:
        return (self.strategy, self.CR, self.F)

    def offspring(self, pop: Population, problem: Problem, rng: np.random.Generator) -> np.ndarray:
        best = pop.dec[int(np.argmin(pop.obj[:, 0]))]
        trial = de_mutate(
            pop.dec, self.strategy, self.F, self.CR, best=best, rng=rng
        )
        return problem.clip(trial)


class GAVariation(Variation):
    """Binary tournament, SBX crossover, polynomial mutation."""

    family = "ga"

    def __init__(
        self,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        pc: float = 0.9,
        pm: float | None = None,
    ):
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.pc = float(pc)
        self.pm = None if pm is None else float(pm)

    def config(self) -> tuple:
        return (self.eta_c, self.eta_m)

    def offspring(self, pop: Population, problem: Problem, rng: np.random.Generator) -> np.ndarray:
        ranks = ndsort(pop.obj)
        crowd = np.zeros(len(pop))
        for r in range(int(ranks.max()) + 1 if ranks.size else 0):
            idx = np.where(ranks == r)[0]
            crowd[idx] = crowding_distance(pop.obj, idx)

        parents = tournament(ranks, crowd, len(pop), rng)
        a, b = random_pairs(len(pop), rng)
        c1, c2 = sbx_crossover(
            pop.dec[parents[a]],
            pop.dec[parents[b]],
            problem.lower,
            problem.upper,
            eta=self.eta_c,
            pc=self.pc,
            rng=rng,
        )
        return polynomial_mutation(
            np.vstack([c1, c2]),
            problem.lower,
            problem.upper,
            eta=self.eta_m,
            pm=self.pm,
            rng=rng,
        )


FAMILIES = {"de": DEVariation, "ga": GAVariation}
DEFAULT_VARIATION = "de"


def config_space(family: str = "de") -> list[tuple]:
    """All parameter tuples of a family, in a stable order (index = config id)."""
    family = str(family).lower()
    if family == "de":
        return [
            (s, cr, f)
            for s in DE_STRATEGIES
            for cr in DE_CR_VALUES
            for f in DE_F_VALUES
        ]
    if family == "ga":
        return [(ec, em) for ec in GA_ETA_C_VALUES for em in GA_ETA_M_VALUES]
    raise ValueError(f"unknown variation family {family!r}")


def make_variation(spec="de", params: tuple | None = None, **kwargs) -> Variation:
    """Build a variation object.

    `spec` is a family name ("de", "ga") or an existing Variation. `params` is
    a tuple from `config_space`; explicit keyword arguments override it.
    """
    if isinstance(spec, Variation):
        return spec
    family = str(spec).lower()
    if family not in FAMILIES:
        raise ValueError(
            f"unknown variation family {family!r}; available: {sorted(FAMILIES)}"
        )
    if params is not None:
        names = PARAM_NAMES[family]
        if len(params) != len(names):
            raise ValueError(
                f"{family} expects {len(names)} parameters {names}, got {params!r}"
            )
        kwargs = {**dict(zip(names, params)), **kwargs}
    return FAMILIES[family](**kwargs)
