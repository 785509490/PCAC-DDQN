"""State features read off the population itself.

Six scalars, no extra function evaluations:

    1. centroid displacement, normalised by the box diagonal
    2. relative change in decision-space spread
    3. current spread (ELA, computed on the population that already exists)
    4. fraction of the population on the first non-dominated front (ELA)
    5. sign-change rate of the last centroid displacements
    6. residual of a linear extrapolation of the centroid

The ELA values are biased estimators: they use the current population rather
than a fresh probe sample. That trade is deliberate -- a probe would spend
function evaluations, and the meta-layer must not buy information the inner
algorithm did not pay for. The bias has to be reported with any result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.metrics import ndsort
from ..core.population import Population


def _spread(dec: np.ndarray) -> float:
    if dec.shape[0] < 2:
        return 0.0
    sub = dec if dec.shape[0] <= 200 else dec[:200]
    diff = sub[:, None, :] - sub[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    iu = np.triu_indices(dist.shape[0], k=1)
    return float(dist[iu].mean())


@dataclass
class StateTracker:
    lower: np.ndarray
    upper: np.ndarray
    centroids: list[np.ndarray] = field(default_factory=list)
    spreads: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.diag = float(np.linalg.norm(self.upper - self.lower))

    def update(self, pop: Population, prev_obj: np.ndarray | None = None) -> np.ndarray:
        c = pop.dec.mean(axis=0)
        spread = _spread(pop.dec)
        front_frac = float(np.mean(ndsort(pop.obj) == 0))

        self.centroids.append(c)
        self.spreads.append(spread)
        if len(self.centroids) > 8:
            self.centroids = self.centroids[-8:]
            self.spreads = self.spreads[-8:]

        step = 0.0
        if len(self.centroids) >= 2:
            step = float(np.linalg.norm(self.centroids[-1] - self.centroids[-2]) / self.diag)

        spread_rel = 0.0
        if len(self.spreads) >= 2:
            prev = self.spreads[-2]
            spread_rel = float((self.spreads[-1] - prev) / (abs(prev) + 1e-12))

        spread_norm = float(spread / (self.diag + 1e-12))

        # Direction, not just magnitude: extrapolation is a signed operation,
        # and a feature vector that only carries the norm cannot tell it which
        # way the set moved. This was a real defect in the first version.
        signed_step = 0.0
        if len(self.centroids) >= 2:
            delta = self.centroids[-1] - self.centroids[-2]
            signed_step = float(np.mean(delta) / self.diag)

        sign_rate = 0.0
        residual = 0.0
        cosine = 0.0
        if len(self.centroids) >= 3:
            d1 = self.centroids[-1] - self.centroids[-2]
            d2 = self.centroids[-2] - self.centroids[-3]
            signs = np.sign(d1) * np.sign(d2)
            sign_rate = float(np.mean(signs < 0))
            denom = float(np.linalg.norm(d2))
            residual = float(np.linalg.norm(d1 - d2) / (denom + 1e-12))
            residual = min(residual, 10.0)
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 > 1e-12 and n2 > 1e-12:
                cosine = float(np.dot(d1, d2) / (n1 * n2))

        # Objective-space change. The carried population is re-evaluated in
        # the new environment before this is called, so the old objectives are
        # available for free. Without these, the state cannot tell a hold from
        # a drift: the decision-space centroid also moves because of search.
        df_mean = 0.0
        df_worse = 0.0
        df_shift = 0.0
        if prev_obj is not None and prev_obj.shape == pop.obj.shape:
            span = np.ptp(prev_obj, axis=0) + 1e-12
            delta = pop.obj - prev_obj
            df_mean = float(np.mean(np.abs(delta) / span))
            # f1 alone is the free variable and does not move when the Pareto
            # SET shifts; use the average over objectives instead.
            df_worse = float(np.mean(delta.mean(axis=1) > 0.0))
            # Signed, so a drift and an oscillating shift point opposite ways.
            df_shift = float(np.mean(delta) / (np.mean(span) + 1e-12))

        return np.array(
            [step, signed_step, spread_rel, spread_norm, front_frac,
             sign_rate, residual, cosine, df_mean, df_worse, df_shift],
            dtype=float,
        )
