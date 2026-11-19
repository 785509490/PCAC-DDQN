"""Genetic operators and selection rules.

All operators take whole arrays and an explicit `numpy.random.Generator`; no
module-level RNG state is touched. That is what makes the run seed the only
source of algorithmic randomness.
"""

from __future__ import annotations

import numpy as np

from .metrics import crowding_distance, ndsort


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta: float = 20.0,
    pc: float = 0.9,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover on two equally shaped parent blocks."""
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(p1.shape)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
    )
    mask = rng.random(p1.shape) <= pc
    c1 = np.where(mask, 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2), p1)
    c2 = np.where(mask, 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2), p2)
    return np.clip(c1, lower, upper), np.clip(c2, lower, upper)


def polynomial_mutation(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta: float = 20.0,
    pm: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Polynomial mutation. Default rate is 1/D, as in PlatEMO."""
    rng = np.random.default_rng() if rng is None else rng
    n, d = X.shape
    if pm is None:
        pm = 1.0 / d
    u = rng.random((n, d))
    delta = np.where(
        u < 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
        1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)),
    )
    mask = rng.random((n, d)) < pm
    Y = np.where(mask, X + delta * (upper - lower), X)
    return np.clip(Y, lower, upper)


def tournament(
    rank: np.ndarray,
    crowd: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binary tournament: lower rank wins, ties broken by larger crowding."""
    k = rank.size
    a = rng.integers(0, k, size=n)
    b = rng.integers(0, k, size=n)
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    return np.where(better, a, b)


def random_pairs(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random permutation split into two halves, one mating pair per slot."""
    perm = rng.permutation(n - (n % 2))
    return perm[0::2], perm[1::2]


def de_mutate(
    X: np.ndarray,
    strategy: str,
    F: float,
    CR: float,
    best: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Differential-evolution mutant vectors, excluding the target index.

    Strategies follow PlatEMO's DE operator names:
    `rand/1`, `best/1`, `current-to-best/1`, `rand/2`, `current-to-rand/1`.
    """
    rng = np.random.default_rng() if rng is None else rng
    n, d = X.shape
    if n < 4:
        return X.copy()

    def pick3():
        out = np.empty((3, n), dtype=int)
        for i in range(n):
            pool = np.delete(np.arange(n), i)
            out[:, i] = rng.choice(pool, size=3, replace=False)
        return out

    idx = pick3()
    r1, r2, r3 = X[idx[0]], X[idx[1]], X[idx[2]]

    if strategy == "rand/1":
        V = r1 + F * (r2 - r3)
    elif strategy == "best/1":
        if best is None:
            raise ValueError("best/1 needs the current best vector")
        V = best + F * (r1 - r2)
    elif strategy == "current-to-best/1":
        if best is None:
            raise ValueError("current-to-best/1 needs the current best vector")
        V = X + F * (best - X) + F * (r1 - r2)
    elif strategy == "rand/2":
        idx2 = pick3()
        r4, r5 = X[idx2[0]], X[idx2[1]]
        V = r1 + F * (r2 - r3) + F * (r4 - r5)
    elif strategy == "current-to-rand/1":
        V = X + rng.random() * (r1 - X) + F * (r2 - r3)
    else:
        raise ValueError(f"unknown DE strategy {strategy!r}")

    jrand = rng.integers(0, d, size=n)
    mask = rng.random((n, d)) < CR
    mask[np.arange(n), jrand] = True
    U = np.where(mask, V, X)
    return U


def select_nsga2(F: np.ndarray, n: int) -> np.ndarray:
    """NSGA-II environmental selection on objective rows; returns indices."""
    ranks = ndsort(F)
    chosen: list[int] = []
    for r in range(int(ranks.max()) + 1 if ranks.size else 0):
        idx = np.where(ranks == r)[0]
        if len(chosen) + idx.size <= n:
            chosen.extend(idx.tolist())
            if len(chosen) == n:
                break
        else:
            cd = crowding_distance(F, idx)
            order = np.argsort(-cd, kind="mergesort")
            chosen.extend(idx[order[: n - len(chosen)]].tolist())
            break
    return np.asarray(chosen, dtype=int)
