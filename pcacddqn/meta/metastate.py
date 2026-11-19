"""State features for the per-change meta-policy.

The pre-action block is cheap and policy-independent:

    11 features from `StateTracker` (change + convergence, no extra FE)
     1 objective count
     7 decision-space archive features (no archive re-evaluation)
     5 archive-probe features (paid for, see `probe_archive`)
     1 decision-subspace angle between the population and the archive

The probe is the one part that spends function evaluations. It evaluates a
farthest-point sample of the archive in the new environment *before* the
policy acts, so the policy knows whether memory is worth using instead of
learning that only from the delayed post-response signal. The caller reports
the probe cost separately from the response cost.

The two post-response features (`useful`, `dom`) remain available after a
memory action and are appended to the next state with a mask. No problem id,
instance seed, schedule seed, future law value, or true Pareto front is
exposed here.

Two later additions were tried and removed: initial-context features and a
fixed objective-scale block. Both increased the state to 43 dimensions and
made the one-step LOPO model worse (ridge 0.199 and 0.168 on the five-action
gate, against 0.153 for this 28-dimensional state); the extra context was
overfit, not informative. Keep this file at 28 dimensions unless a new gate
says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.metrics import ndsort
from ..core.population import Population
from ..core.problem import Problem
from .env import StateTracker, _spread
from .memory import farthest_point_sample

META_FEATURE_NAMES: tuple[str, ...] = (
    "step",
    "signed_step",
    "spread_rel",
    "spread_norm",
    "front_frac",
    "sign_rate",
    "residual",
    "cosine",
    "df_mean",
    "df_worse",
    "df_shift",
    "n_obj",
    "taut_norm",
    "n_env_norm",
    "arch_age",
    "arch_log_size",
    "arch_centroid_dist",
    "arch_inside",
    "arch_recent_share",
    "arch_spread_ratio",
    "arch_near_front",
    "probe_useful",
    "probe_dom",
    "probe_keep",
    "probe_dist",
    "probe_inside",
    "arch_subspace",
    "post_useful",
    "post_dom",
    "post_mask",
)

PRE_DIM = 27
POST_DIM = 3
STATE_DIM = PRE_DIM + POST_DIM


@dataclass
class ArchiveProbe:
    """Result of evaluating an archive sample in the new environment."""

    useful: float = 0.0   # non-dominated among the kept archive points
    dom: float = 0.0      # archive points dominating at least one current one
    keep: float = 0.0     # archive points not dominated by the current pop
    dist: float = 0.0     # mean normalised objective distance to the pop
    inside: float = 0.0   # archive points inside the current objective box
    m: int = 0            # evaluations paid for this probe
    obj: np.ndarray | None = None


def probe_archive(problem: Problem, pop: Population, archive: list[np.ndarray],
                  m: int, rng: np.random.Generator, t: float,
                  filter_archive: bool = True) -> ArchiveProbe:
    """Evaluate a farthest-point sample of the archive at the new time.

    This is decision-relevant information the old pre-action state did not
    have: whether the archive still contains points that beat, or at least
    survive against, the carried population. It costs ``min(m, |archive|)``
    evaluations. The caller must add ``probe.m`` to its own probe counter; the
    algorithm's response counter is left untouched.
    """
    if not archive:
        return ArchiveProbe()

    A = np.vstack(archive)
    m = min(int(m), len(A))
    if m <= 0:
        return ArchiveProbe()

    sub = farthest_point_sample(A, m, rng)
    A_sel = A[sub]
    A_obj = problem.evaluate(A_sel, t)

    # An archive point dominated by some current member is strictly worse than
    # what the population already has. On a shape/regime change a whole old
    # front can fall into this class while staying non-dominated among itself.
    le = np.all(pop.obj[None, :, :] <= A_obj[:, None, :], axis=2)
    lt = np.any(pop.obj[None, :, :] < A_obj[:, None, :], axis=2)
    keep_a = ~np.any(le & lt, axis=1) if filter_archive else np.ones(len(A_obj), bool)
    keep = float(np.mean(keep_a))

    A_kept = A_obj[keep_a]
    useful = float(np.mean(ndsort(A_kept) == 0)) if len(A_kept) else 0.0

    dom = 0.0
    if len(A_obj):
        dom = float(np.mean(np.any(
            np.all(A_obj[:, None, :] <= pop.obj[None, :, :], axis=2)
            & np.any(A_obj[:, None, :] < pop.obj[None, :, :], axis=2), axis=1)))

    dist = 0.0
    inside = 0.0
    if len(A_obj):
        scale = np.ptp(pop.obj, axis=0) + 1e-12
        d = np.min(np.linalg.norm(
            (A_obj[:, None, :] - pop.obj[None, :, :]) / scale, axis=2), axis=1)
        dist = float(np.mean(d))
        lo = pop.obj.min(axis=0)
        hi = pop.obj.max(axis=0)
        inside = float(np.mean(np.all((A_obj >= lo) & (A_obj <= hi), axis=1)))

    return ArchiveProbe(useful, dom, keep, dist, inside, m, A_obj)


def _subspace_angle(A: np.ndarray, B: np.ndarray, k: int = 3) -> float:
    """Mean principal angle between the top-k decision subspaces, in [0, 1].

    Zero means the archive's decision-space directions are aligned with the
    current population; one means they are orthogonal. Only decision vectors
    are used, so this costs no evaluations.
    """
    if len(A) < 2 or len(B) < 2:
        return 0.0
    k = min(int(k), A.shape[1], len(A) - 1, len(B) - 1)
    if k <= 0:
        return 0.0
    # Right singular vectors span the decision-space directions; U would span
    # the sample space and has a different row count for the two sets.
    Qa = np.linalg.svd(A - A.mean(axis=0), full_matrices=False)[2][:k].T
    Qb = np.linalg.svd(B - B.mean(axis=0), full_matrices=False)[2][:k].T
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.clip(1.0 - np.mean(np.clip(s, -1.0, 1.0)), 0.0, 1.0))


@dataclass
class MetaStateTracker:
    lower: np.ndarray
    upper: np.ndarray
    M: int
    archive_capacity: int = 20
    recent_fronts: int = 2
    taut: int = 20
    n_env: int = 21
    base: StateTracker = field(init=False)

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.diag = float(np.linalg.norm(self.upper - self.lower)) + 1e-12
        self.base = StateTracker(self.lower, self.upper)
        self.archive_capacity = max(1, int(self.archive_capacity))

    def reset(self, pop: Population) -> np.ndarray:
        self.base.centroids.clear()
        self.base.spreads.clear()
        return self.base.update(pop)

    def pre_action(self, pop: Population, prev_obj: np.ndarray | None,
                   archive: list[np.ndarray],
                   probe: ArchiveProbe | None = None) -> np.ndarray:
        base = self.base.update(pop, prev_obj=prev_obj)
        arch = np.vstack(archive) if archive else np.empty((0, pop.dec.shape[1]))
        feats = np.zeros(13, dtype=float)
        if len(arch):
            n_arch = len(arch)
            feats[0] = min(1.0, len(archive) / self.archive_capacity)
            feats[1] = float(np.log1p(n_arch) / np.log1p(self.archive_capacity))

            centroid = pop.dec.mean(axis=0)
            d = np.linalg.norm(arch - centroid, axis=1) / self.diag
            feats[2] = float(np.clip(np.min(d), 0.0, 2.0) / 2.0)

            lo = pop.dec.min(axis=0)
            hi = pop.dec.max(axis=0)
            feats[3] = float(np.mean(np.all((arch >= lo) & (arch <= hi), axis=1)))

            feats[4] = float(min(self.recent_fronts, len(archive))
                             / max(1, len(archive)))

            pop_spread = _spread(pop.dec)
            arch_spread = _spread(arch)
            feats[5] = float(np.clip(arch_spread / (pop_spread + 1e-12), 0.0, 5.0)
                             / 5.0)

            # Fraction of archive points within one population spread of the
            # current population. This is decision-space only.
            if pop_spread > 1e-12:
                near = np.min(
                    np.linalg.norm(arch[:, None, :] - pop.dec[None, :, :],
                                   axis=2), axis=1,
                ) / pop_spread
                feats[6] = float(np.mean(near <= 1.0))

            # Decision-subspace alignment. A stale archive can look close in
            # centroid distance while pointing in a different direction.
            feats[12] = _subspace_angle(pop.dec, arch)

        if probe is not None and probe.m > 0:
            feats[7] = float(np.clip(probe.useful, 0.0, 1.0))
            feats[8] = float(np.clip(probe.dom, 0.0, 1.0))
            feats[9] = float(np.clip(probe.keep, 0.0, 1.0))
            feats[10] = float(np.clip(probe.dist, 0.0, 10.0) / 10.0)
            feats[11] = float(np.clip(probe.inside, 0.0, 1.0))

        # Known global run parameters. They cost no evaluations and let one
        # policy cover different environment horizons (the joint-taut run).
        glob = np.array([float(self.taut) / 50.0,
                         float(self.n_env) / 21.0], dtype=float)
        return np.concatenate([base, np.array([float(self.M)]), glob, feats])

    def post_response(self, alg, available: bool) -> np.ndarray:
        if available:
            return np.array(
                [float(alg.useful_frac), float(alg.dom_frac), 1.0], dtype=float
            )
        return np.zeros(POST_DIM, dtype=float)
