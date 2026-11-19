"""Vectorised selection and quality metrics.

The metric definitions follow PlatEMO's IGD/HV so that numbers produced here
can be compared with the MATLAB v1 implementation on identical point sets.
NDSort and crowding are re-implemented rather than imported: the framework is
meant to stand alone, and the O(N^2) version is exact at the population sizes
used here (N <= 100 for runs, <= 2000 for reference fronts).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Non-dominated sorting
# ---------------------------------------------------------------------------

def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the first non-dominated front (minimisation).

    For M == 2 this uses the O(N log N) skyline sweep; for M > 2 it falls back
    to the exact O(N^2) pairwise test. Duplicates are kept only once.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)

    if F.shape[1] == 2:
        return _nondominated_mask_2d(F)

    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominated = (le & lt).any(axis=0)
    return ~dominated


def _nondominated_mask_2d(F: np.ndarray) -> np.ndarray:
    """First front for two objectives.

    Sort by f1 then f2; a point survives iff its f2 is strictly below the best
    f2 seen so far. Points with identical (f1, f2) collapse to one survivor.
    """
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep = np.zeros(F.shape[0], dtype=bool)
    best_f2 = np.inf
    for idx in order:
        y = F[idx, 1]
        if y < best_f2:
            keep[idx] = True
            best_f2 = y
    return keep


def ndsort(F: np.ndarray) -> np.ndarray:
    """Fast non-dominated sort. Returns the front index of every point (0 = best)."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    ranks = np.full(n, -1, dtype=int)
    if n == 0:
        return ranks

    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt                    # dom[i, j] == True iff i dominates j
    count = dom.sum(axis=0)          # how many dominate each point

    front = np.where(count == 0)[0]
    rank = 0
    while front.size:
        ranks[front] = rank
        count = count - dom[front].sum(axis=0)
        rank += 1
        front = np.where((count == 0) & (ranks == -1))[0]
    return ranks


def crowding_distance(F: np.ndarray, front: np.ndarray | None = None) -> np.ndarray:
    """Crowding distance inside the given front (all points by default)."""
    F = np.asarray(F, dtype=float)
    if front is None:
        front = np.arange(F.shape[0])
    front = np.asarray(front, dtype=int)
    n = front.size
    dist = np.zeros(n, dtype=float)
    if n <= 2:
        dist[:] = np.inf
        return dist

    sub = F[front]
    for m in range(F.shape[1]):
        order = np.argsort(sub[:, m], kind="mergesort")
        vals = sub[order, m]
        span = vals[-1] - vals[0]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if span <= 0:
            continue
        dist[order[1:-1]] += (vals[2:] - vals[:-2]) / span
    return dist


def environmental_selection_nsga2(F: np.ndarray, n: int) -> np.ndarray:
    """Return indices of the n survivors chosen by NSGA-II's rule."""
    ranks = ndsort(F)
    survivors: list[int] = []
    for r in range(ranks.max() + 1 if ranks.size else 0):
        idx = np.where(ranks == r)[0]
        if len(survivors) + idx.size <= n:
            survivors.extend(idx.tolist())
        else:
            cd = crowding_distance(F, idx)
            take = np.argsort(-cd, kind="mergesort")[: n - len(survivors)]
            survivors.extend(idx[take].tolist())
            break
    return np.asarray(survivors, dtype=int)


# ---------------------------------------------------------------------------
# IGD / HV
# ---------------------------------------------------------------------------

def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between rows of A and rows of B."""
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def igd(F: np.ndarray, R: np.ndarray) -> float:
    """Inverted generational distance: mean distance from each reference point
    to the nearest population point. Both arguments are (N, M) minimisation."""
    F = np.asarray(F, dtype=float)
    R = np.asarray(R, dtype=float)
    if F.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    return float(_pairwise_dist(R, F).min(axis=1).mean())


def hv(F: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume dominated by F with respect to the upper-bound point `ref`."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if F.shape[0] == 0:
        return 0.0
    keep = np.all(F < ref, axis=1)
    F = F[keep]
    if F.shape[0] == 0:
        return 0.0
    if F.shape[1] == 2:
        return _hv2(F, ref)
    if F.shape[1] == 3:
        return _hv3(F, ref)
    raise ValueError("hv supports 2 or 3 objectives")


def _hv2(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume by summing the skyline slabs."""
    F = F[np.all(F < ref, axis=1)]
    if F.shape[0] == 0:
        return 0.0

    order = np.lexsort((F[:, 1], F[:, 0]))
    sx: list[float] = []
    sy: list[float] = []
    best = np.inf
    for x, y in zip(F[order, 0], F[order, 1]):
        if y < best:                 # keep only the non-dominated skyline
            sx.append(float(x))
            sy.append(float(y))
            best = y

    total = 0.0
    for i, (x, y) in enumerate(zip(sx, sy)):
        x_next = sx[i + 1] if i + 1 < len(sx) else ref[0]
        total += (x_next - x) * (ref[1] - y)
    return float(total)


def _hv3(F: np.ndarray, ref: np.ndarray) -> float:
    """Sweep over f1; each slab contributes a 2-D hypervolume in (f2, f3).

    The slabs are disjoint: slab i covers f1 in [x_i, x_{i+1}) and uses every
    point with f1 <= x_i. Summing over slabs is therefore exact.
    """
    order = np.argsort(F[:, 0], kind="mergesort")
    F = F[order]
    xs = F[:, 0]

    total = 0.0
    prev_x = ref[0]
    for i in range(len(F) - 1, -1, -1):
        if i < len(F) - 1 and xs[i] == xs[i + 1]:
            continue
        if prev_x <= xs[i]:
            continue
        sub = F[: i + 1, 1:]
        total += (prev_x - xs[i]) * _hv2(sub, ref[1:])
        prev_x = xs[i]
    return float(total)


def normalise_front(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Divide by the reference point; keeps IGD scale comparable across problems."""
    return np.asarray(F, dtype=float) / np.asarray(ref, dtype=float)
