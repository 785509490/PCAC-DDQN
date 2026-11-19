"""Response operators and the seven-cell action set.

The first six cells are the ones used by the MATLAB closed-loop experiments,
parameterised by `(alpha, sigma, p)`:

    alpha   extrapolate the population along its last step
    sigma   add Gaussian noise, scaled by the box span
    p       replace that fraction with uniform restarts

The seventh cell is a different construction: it keeps the best half of the
population untouched and refills the rest with perturbed copies of the current
Pareto front. It was measured before it was added -- `PPS(variant="diversity")`,
which does exactly this, had the lowest mean MIGD of any simple response
(0.6675 vs 0.6939 for the predictive PPS and 0.8335 for plain NSGA-II). A
global `jitter` cell is not a substitute: applying the same noise to the whole
population was 28% worse than inheriting. What matters is which part of the
population receives the noise.

`kappa` (the affine manifold refit) exists in the MATLAB operator but was zero
in every main experiment. It is deliberately absent here: the argument in the
old report that "the response family only translates" was never tested with
kappa on, and repeating that untested claim in a new codebase would be worse
than leaving the knob out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.metrics import ndsort
from ..core.population import Population
from ..core.problem import Problem


@dataclass(frozen=True)
class Response:
    """One response operator.

    `kind="global"` applies the `(alpha, sigma, p)` pipeline to the whole
    population. `kind="elite_front"` keeps the best `1 - frac` individuals and
    replaces the rest with noisy draws from the current non-dominated set.
    `kind="elite_random"` keeps the same survivors but refills them with
    uniform box restarts; it exists to separate the front restriction from the
    jitter in the mechanism decomposition.
    """

    alpha: float = 0.0
    sigma: float = 0.0
    p: float = 0.0
    kind: str = "global"
    frac: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"global", "elite_front", "elite_random"}:
            raise ValueError(f"unknown response kind {self.kind!r}")
        if self.kind != "global" and not 0.0 < self.frac < 1.0:
            raise ValueError(f"{self.kind} needs 0 < frac < 1")


RESPONSES: tuple[Response, ...] = (
    Response(0.0, 0.0, 0.0),
    Response(0.0, 0.0, 1.0),
    Response(1.0, 0.0, 0.0),
    Response(1.5, 0.0, 0.0),
    Response(0.0, 0.05, 0.0),
    Response(1.0, 0.05, 0.25),
    Response(0.0, 0.05, 0.0, kind="elite_front", frac=0.5),
)

RESPONSE_NAMES = (
    "inherit",
    "restart",
    "extrapolate",
    "over-extrapolate",
    "jitter",
    "extrapolate+jitter+restart",
    "elite-front-jitter",
)


def apply_response(
    problem: Problem,
    pop: Population,
    prev_dec: np.ndarray | None,
    cell: Response,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the decision matrix after applying one response cell."""
    if cell.kind == "elite_front":
        return _elite_front_jitter(problem, pop, cell, rng)
    if cell.kind == "elite_random":
        return _elite_front_random(problem, pop, cell, rng)

    X = pop.dec.copy()
    if cell.alpha and prev_dec is not None and prev_dec.shape == X.shape:
        X = X + cell.alpha * (X - prev_dec)

    if cell.sigma:
        span = problem.upper - problem.lower
        X = X + rng.normal(scale=cell.sigma * span, size=X.shape)

    n_restart = int(round(cell.p * X.shape[0]))
    if n_restart > 0:
        idx = rng.choice(X.shape[0], size=n_restart, replace=False)
        X[idx] = problem.lower + rng.random((n_restart, problem.D)) * (
            problem.upper - problem.lower
        )

    return problem.clip(X)


def _elite_front_jitter(
    problem: Problem,
    pop: Population,
    cell: Response,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep the best `1 - frac`, refill from the current front with noise.

    The sort is by non-domination rank only, with a stable order, matching the
    PPS-diversity control this cell was validated against. Adding crowding here
    would make the cell a different operator than the one measured.
    """
    n = len(pop)
    n_replace = int(round(cell.frac * n))
    n_replace = max(1, min(n - 1, n_replace))
    ranks = ndsort(pop.obj)
    order = np.argsort(ranks, kind="mergesort")
    keep = order[: n - n_replace]

    front = pop.dec[ranks == 0]
    if front.shape[0] == 0:
        front = pop.dec
    idx = rng.choice(front.shape[0], size=n_replace, replace=True)
    span = problem.upper - problem.lower
    new = front[idx] + rng.normal(
        scale=cell.sigma * span, size=(n_replace, problem.D)
    )
    return problem.clip(np.vstack([pop.dec[keep], new]))


def _elite_front_random(
    problem: Problem,
    pop: Population,
    cell: Response,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep the best `1 - frac`, refill with uniform restarts.

    Same survivor rule as `elite_front`, different refill. This separates
    "restricting the refill to the current front" from "adding Gaussian noise
    around the front".
    """
    n = len(pop)
    n_replace = int(round(cell.frac * n))
    n_replace = max(1, min(n - 1, n_replace))
    ranks = ndsort(pop.obj)
    order = np.argsort(ranks, kind="mergesort")
    keep = order[: n - n_replace]
    span = problem.upper - problem.lower
    new = problem.lower + rng.random((n_replace, problem.D)) * span
    return np.vstack([pop.dec[keep], new])
