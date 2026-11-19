"""Archive-memory response with an online trust weight.

This is one population with one variation operator, not a portfolio. At every
change the operator mixes two candidate sources:

    current front   -- what the population converged to in the environment it
                       is leaving (the elite response)
    archive         -- a diverse subset of the Pareto fronts from *earlier*
                       environments, re-evaluated in the new environment

`rho` is the share of the refill slots taken from the archive. It is not a
fixed hyper-parameter: a meta-learned linear law maps observable state to rho
at every change, and the runtime updates the state from two zero-cost signals:

    dmag        realised objective displacement of the carried population
    surv_ratio  survival rate of archive-tagged individuals divided by the
                survival rate of the current-front individuals

Individuals carry an origin tag (0 keep, 1 jitter, 2 archive), so the survival
signal rides on the tag propagation that NSGA-II already does.

The archive subset is chosen by farthest-point sampling in decision space, so
the response costs `m` evaluations per change instead of re-evaluating the
whole archive. `m` is a meta-parameter; the fixed-generation protocol does not
charge response evaluations, so keeping `m` small is a fairness constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..algorithms.nsga2 import NSGA2
from ..core.metrics import ndsort
from ..core.operators import select_nsga2
from ..core.population import Population
from ..core.problem import Problem


@dataclass
class MemoryConfig:
    frac: float = 0.90          # refill share, as in the elite response
    sigma: float = 0.10         # jitter scale for the current-front refill
    m: int = 50                 # archive points evaluated per change
    max_fronts: int = 20        # how many past fronts to keep
    recent_only: bool = False    # sample only the last `recent_fronts` fronts
    archive_sigma: float = 0.0  # jitter around selected archive points
    filter_archive: bool = True  # drop archive points dominated by the current pop
    depth_gate: float | None = None  # useful fraction below which to forget
    recent_fronts: int = 2      # fronts kept by the gate's short-memory branch
    soft_depth: bool = False    # continuous deep/recent split instead of a gate
    depth_tau: float = 0.22
    depth_slope: float = 40.0
    m_recent: int = 40
    dom_gate: float | None = None  # archive-dominance fraction below which to forget
    w_dom: float = 0.0
    dom_rho_low: float | None = None  # switch archive off below this dominance
    dom_rho_high: float = 0.60        # switch it back on above this
    dom_rho_dwell: int = 2            # consecutive changes before a switch
    rho0: float = 0.50          # base archive trust (logit)
    w_dmag: float = 0.0
    w_surv: float = 0.0
    w_useful: float = 0.0
    gamma: float = 0.50         # EMA gain for the survival signal
    online: bool = True         # False = freeze rho at sigmoid(rho0)
    seed_tag: bool = True


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def farthest_point_sample(X: np.ndarray, m: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Greedy max-min subset of size `m` (farthest-point sampling)."""
    n = len(X)
    if n <= m:
        return np.arange(n)
    start = int(rng.integers(n))
    chosen = np.empty(m, dtype=int)
    chosen[0] = start
    d = np.linalg.norm(X - X[start], axis=1)
    for i in range(1, m):
        j = int(np.argmax(d))
        chosen[i] = j
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return chosen


class MemoryNSGA2(NSGA2):
    """NSGA-II with a memory-trust response at every change."""

    def __init__(self, cfg: MemoryConfig | None = None, n_pop: int = 100):
        super().__init__(n_pop=n_pop, variation="de", strategy="rand/1",
                         CR=0.5, F=0.5)
        self.cfg = cfg or MemoryConfig()
        self.archive: list[np.ndarray] = []
        self.last_obj: np.ndarray | None = None
        self.surv_ema: float = 1.0
        self.prev_share: dict[int, float] | None = None
        self.rho: float = _sigmoid(self.cfg.rho0)
        self.response_evals: int = 0
        self.rho_log: list[float] = []
        self.surv_log: list[float] = []
        self.dmag_log: list[float] = []
        self.useful_log: list[float] = []
        self.dist_log: list[float] = []
        self.inside_log: list[float] = []
        self.gate_log: list[int] = []
        self.p_deep_log: list[float] = []
        self.dom_log: list[float] = []
        self.p_deep: float = 1.0
        self.dom_frac: float = 0.5
        self.dom_ema: float = 0.5
        self._rho_gate_on: bool = False
        self._low_count: int = 0
        self.rho_gate_log: list[int] = []
        self.useful_frac: float = 0.5
        self.obj_scale: float = 1.0

    def initialize(self, problem: Problem, t: float) -> Population:
        pop = super().initialize(problem, t)
        # A fixed scale for the change magnitude. Using the *current*
        # population's spread inflates dmag once the population converges:
        # the denominator goes to zero and tanh saturates, which silently
        # turns the online law into a constant.
        self.obj_scale = float(np.mean(np.ptp(pop.obj, axis=0))) + 1e-12
        return pop

    def step(self, problem: Problem, pop: Population, t: float) -> Population:
        out = super().step(problem, pop, t)
        # Objectives before the next change's re-evaluation; the runner
        # evaluates on a copy, so this snapshot survives.
        self.last_obj = out.obj.copy()
        return out

    # -- signals -----------------------------------------------------------

    def _update_signals(self, pop: Population) -> float:
        """Return the normalised change magnitude and update the survival EMA."""
        if self.last_obj is not None and len(self.last_obj) == len(pop.obj):
            dmag = float(np.mean(np.abs(pop.obj - self.last_obj))
                         / self.obj_scale)
        else:
            dmag = 0.0

        if pop.tag is not None and self.prev_share:
            rates = {}
            for tag, share in self.prev_share.items():
                if share <= 1e-9:
                    continue
                rates[tag] = float(np.mean(pop.tag == tag)) / share
            # A ratio of two near-zero survival rates is meaningless; the
            # epsilon keeps a group that vanished from returning 1e7.
            ratio = ((rates.get(2, 0.0) + 0.05)
                     / (rates.get(0, 0.0) + 0.05))
            ratio = float(np.clip(ratio, 0.0, 5.0))
            self.surv_ema = ((1.0 - self.cfg.gamma) * self.surv_ema
                             + self.cfg.gamma * ratio)
        self.dmag_log.append(dmag)
        self.surv_log.append(self.surv_ema)
        return dmag

    def _rho(self, dmag: float) -> float:
        if not self.cfg.online:
            return _sigmoid(self.cfg.rho0)
        # Both features are mapped to a roughly [-1, 1] range without a
        # dataset-dependent scale: dmag is already span-normalised, and the
        # survival ratio is centred on its neutral value 1.
        surv = np.tanh(0.5 * (self.surv_ema - 1.0))
        useful = 2.0 * self.useful_frac - 1.0
        logit = self.cfg.rho0 + self.cfg.w_dmag * np.tanh(0.5 * dmag) \
            + self.cfg.w_surv * surv + self.cfg.w_useful * useful
        return _sigmoid(float(logit))

    # -- response ----------------------------------------------------------

    def _sample_archive(self, pool, m: int, problem: Problem, t: float,
                        rng: np.random.Generator, pop: Population):
        """FPS sample of an archive pool, evaluated and relevance-filtered."""
        A = np.vstack(pool)
        m = min(int(m), len(A))
        sub = farthest_point_sample(A, m, rng)
        A_sel = A[sub]
        A_obj = problem.evaluate(A_sel, t)
        self.response_evals += int(m)
        if self.cfg.filter_archive:
            # An archive point dominated by *some* current member is strictly
            # worse than what the population already has; on a shape/regime
            # change a whole old front can fall into this class while
            # remaining non-dominated among itself.
            le = np.all(pop.obj[None, :, :] <= A_obj[:, None, :], axis=2)
            lt = np.any(pop.obj[None, :, :] < A_obj[:, None, :], axis=2)
            keep_a = ~np.any(le & lt, axis=1)
            A_sel = A_sel[keep_a]
            A_obj = A_obj[keep_a]
        useful = float(np.mean(ndsort(A_obj) == 0)) if len(A_obj) else 0.0
        return A_sel, A_obj, useful

    def _dom_frac(self, A_obj: np.ndarray, pop: Population) -> float:
        """Fraction of archive points that dominate at least one current one."""
        if not len(A_obj):
            return 0.0
        dom = np.any(
            np.all(A_obj[:, None, :] <= pop.obj[None, :, :], axis=2)
            & np.any(A_obj[:, None, :] < pop.obj[None, :, :], axis=2), axis=1)
        return float(np.mean(dom))

    def _log_geo(self, A_obj: np.ndarray, pop: Population) -> None:
        if len(A_obj):
            scale = np.ptp(pop.obj, axis=0) + 1e-12
            d = np.min(np.linalg.norm(
                (A_obj[:, None, :] - pop.obj[None, :, :]) / scale,
                axis=2), axis=1)
            self.dist_log.append(float(np.mean(d)))
            lo = pop.obj.min(axis=0)
            hi = pop.obj.max(axis=0)
            self.inside_log.append(float(np.mean(
                np.all((A_obj >= lo) & (A_obj <= hi), axis=1))))
            self.dom_log.append(self._dom_frac(A_obj, pop))
        else:
            self.dist_log.append(0.0)
            self.inside_log.append(0.0)
            self.dom_log.append(0.0)

    def _take(self, dec: np.ndarray, obj: np.ndarray, n: int,
              rng: np.random.Generator, span: np.ndarray,
              problem: Problem) -> np.ndarray:
        """Pick `n` diverse points from a pool (all of it if it is small)."""
        if len(dec) > n:
            pick = select_nsga2(obj, n)
        else:
            pick = np.arange(len(dec))
        out = dec[pick]
        if self.cfg.archive_sigma > 0.0:
            out = problem.clip(out + rng.normal(
                scale=self.cfg.archive_sigma * span, size=out.shape))
        return out

    def respond(self, problem: Problem, pop: Population, t_prev: float,
                t_cur: float) -> Population:
        """Construct the response from the archive as it was before this change."""
        rng = self._rng()
        cfg = self.cfg
        dmag = self._update_signals(pop)

        span = problem.upper - problem.lower
        n = len(pop)
        ranks = ndsort(pop.obj)
        front = pop.dec[ranks == 0]
        if len(front) == 0:
            front = pop.dec

        # Past memory only: the incoming front is appended afterwards, so the
        # pool never duplicates the current population.
        A_rec = np.empty((0, problem.D))
        O_rec = np.empty((0, problem.M))
        if self.archive:
            pool = self.archive
            if cfg.recent_only and len(self.archive) > cfg.recent_fronts:
                pool = self.archive[-cfg.recent_fronts:]
            A_deep, O_deep, u_deep = self._sample_archive(
                pool, cfg.m, problem, t_cur, rng, pop)
            p_deep = 1.0
            # An old front that no longer beats any current member is stale
            # even when it is non-dominated among itself. This fraction is the
            # signal for the depth gate; it separates shape/regime changes
            # (low dominance) from incremental shifts (high dominance).
            self.dom_frac = self._dom_frac(O_deep, pop)
            self.dom_ema = ((1.0 - cfg.gamma) * self.dom_ema
                            + cfg.gamma * self.dom_frac)
            if cfg.recent_only:
                self.useful_frac = u_deep
            elif (cfg.dom_gate is not None and self.dom_frac < cfg.dom_gate
                    and len(self.archive) > cfg.recent_fronts):
                recent = self.archive[-cfg.recent_fronts:]
                A_deep, O_deep, u_deep = self._sample_archive(
                    recent, cfg.m_recent, problem, t_cur, rng, pop)
                self.useful_frac = u_deep
                self.dom_frac = self._dom_frac(O_deep, pop)
                p_deep = 0.0
            elif (cfg.soft_depth and len(self.archive) > cfg.recent_fronts):
                # Continuous version of the depth gate: a poorly aligned deep
                # archive shifts the fill toward the most recent fronts, but
                # never to zero in one step.
                recent = self.archive[-cfg.recent_fronts:]
                A_rec, O_rec, u_rec = self._sample_archive(
                    recent, cfg.m_recent, problem, t_cur, rng, pop)
                p_deep = _sigmoid(cfg.depth_slope * (u_deep - cfg.depth_tau))
                self.useful_frac = p_deep * u_deep + (1.0 - p_deep) * u_rec
            elif (cfg.depth_gate is not None and u_deep < cfg.depth_gate
                    and len(self.archive) > cfg.recent_fronts):
                recent = self.archive[-cfg.recent_fronts:]
                A_deep, O_deep, u_deep = self._sample_archive(
                    recent, cfg.m_recent, problem, t_cur, rng, pop)
                self.useful_frac = u_deep
                self.dom_frac = self._dom_frac(O_deep, pop)
                p_deep = 0.0
            else:
                self.useful_frac = u_deep
            self.p_deep = p_deep
            self.p_deep_log.append(p_deep)
            self.gate_log.append(int(p_deep < 0.5))
            self._log_geo(
                np.vstack([O_deep, O_rec]) if len(O_rec) else O_deep, pop)
        else:
            A_deep = np.empty((0, problem.D))
            O_deep = np.empty((0, problem.M))
            self.useful_frac = 0.0
            self.p_deep = 1.0
            self.p_deep_log.append(1.0)
            self.gate_log.append(0)
        self.useful_log.append(self.useful_frac)

        rho = self._rho(dmag)
        if cfg.dom_rho_low is not None:
            # Hysteresis on the dominance signal. A single noisy change cannot
            # turn memory off, and once off, a single good change cannot turn
            # it back on. Low dominance means the archive no longer beats any
            # current member, so pure elite is the safer response.
            if self._rho_gate_on:
                if self.dom_ema > cfg.dom_rho_high:
                    self._rho_gate_on = False
            else:
                if self.dom_ema < cfg.dom_rho_low:
                    self._low_count += 1
                else:
                    self._low_count = 0
                if self._low_count >= cfg.dom_rho_dwell:
                    self._rho_gate_on = True
            if self._rho_gate_on:
                rho = 0.0
        self.rho_gate_log.append(int(self._rho_gate_on))
        self.rho = rho
        self.rho_log.append(rho)

        # The incoming front becomes memory for later changes. It was already
        # re-evaluated in the new environment by the driver. `remember` is
        # called by `on_change` *after* `respond`; see the order note there.

        n_keep = max(1, int(round((1.0 - cfg.frac) * n)))
        keep = select_nsga2(pop.obj, n_keep)
        n_fill = n - n_keep
        # A small `m` cannot fill a large archive share; the remainder goes to
        # the current-front refill so the population always has exactly N.
        available = len(A_deep) + len(A_rec)
        n_arch = min(int(round(rho * n_fill)), available) if available else 0
        if cfg.soft_depth and len(A_rec):
            n_deep = min(int(round(self.p_deep * n_arch)), len(A_deep))
            n_rec = min(n_arch - n_deep, len(A_rec))
            n_deep = min(n_arch - n_rec, len(A_deep))
        else:
            n_deep = min(n_arch, len(A_deep))
            n_rec = 0
        n_jit = n_fill - n_deep - n_rec

        parts = [pop.dec[keep]]
        tags = [np.zeros(n_keep, dtype=int)]

        if n_deep > 0:
            parts.append(self._take(A_deep, O_deep, n_deep, rng, span,
                                    problem))
            tags.append(np.full(n_deep, 2, dtype=int))
        if n_rec > 0:
            parts.append(self._take(A_rec, O_rec, n_rec, rng, span, problem))
            tags.append(np.full(n_rec, 2, dtype=int))

        if n_jit > 0:
            idx = rng.integers(0, len(front), n_jit)
            jit = front[idx] + rng.normal(
                scale=cfg.sigma * span, size=(n_jit, problem.D))
            parts.append(problem.clip(jit))
            tags.append(np.ones(n_jit, dtype=int))

        dec = np.vstack(parts)
        tag = np.concatenate(tags)
        # The driver re-evaluates the whole returned population, so the
        # objective matrix here is a placeholder with the right shape.
        obj = np.zeros((len(dec), problem.M))
        self.prev_share = {
            int(t): float(np.mean(tag == t)) for t in np.unique(tag)
        }
        return Population(dec, obj, tag=tag if cfg.seed_tag else None)

    # -- archive -----------------------------------------------------------

    def remember(self, pop: Population) -> None:
        """Append the incoming first front to the archive. No RNG.

        This must run *after* `respond`: the response samples the archive as
        it was before this change, and appending first would put the current
        population into its own memory.
        """
        ranks = ndsort(pop.obj)
        front = pop.dec[ranks == 0]
        if len(front) == 0:
            front = pop.dec
        self.archive.append(front.copy())
        if len(self.archive) > self.cfg.max_fronts:
            self.archive.pop(0)

    def on_change(self, problem: Problem, pop: Population, t_prev: float,
                  t_cur: float) -> Population:
        out = self.respond(problem, pop, t_prev, t_cur)
        self.remember(pop)
        return out
