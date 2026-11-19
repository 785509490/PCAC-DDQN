"""The assembled dynamic problem: base x mode x law x front."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.problem import Problem
from ..core.metrics import nondominated_mask
from .bases import BASES
from .laws import LAWS
from .modes import MODES, ModeContext
from .pfs import CHANGES, GEOMETRIES


@dataclass
class ProblemSpec:
    pid: str
    cls: str
    base: str
    mode: str
    geometry: str
    change: str
    law: str
    M: int = 2
    reg_len: int = 3
    severity: float = 1.0
    base_amp: float = 1.0
    jump_variant: str = "hetero"
    schedule_seed: int | None = None   # not None only for class B
    intensity: bool = True             # heterogeneous per-change magnitude
    intensity_seed: str = "schedule"   # "schedule" (old B) or "instance" (v4)
    note: str = ""


class DynamicProblem(Problem):
    def __init__(self, spec: ProblemSpec, D: int = 10, n_env: int = 21,
                 taut: int = 20, T0: int = 50,
                 intensity_log_sd: float = 0.45,
                 intensity_rho: float = 0.7,
                 intensity_mode: str = "normal",
                 intensity_seed: str | None = None):
        self.spec = spec
        self.intensity_log_sd = float(intensity_log_sd)
        self.intensity_rho = float(intensity_rho)
        self.intensity_mode = str(intensity_mode)
        self.intensity_seed = str(intensity_seed or spec.intensity_seed)
        if self.intensity_seed not in {"schedule", "instance"}:
            raise ValueError(
                f"intensity_seed must be 'schedule' or 'instance', got "
                f"{self.intensity_seed!r}"
            )
        self._m_abs_max = 1.0
        self.D = int(D)
        self.M = int(spec.M)
        self.n_env = int(n_env)
        self.nt = self.n_env - 1
        self.taut = int(taut)
        self.T0 = int(T0)

        self.base = BASES[spec.base](severity=spec.severity, base_amp=spec.base_amp)
        self.mode = MODES[spec.mode]()
        # The three-objective front is a sphere; it has no scalar geometry.
        self.geom = None if spec.geometry == "sphere" else GEOMETRIES[spec.geometry]()
        self.change = CHANGES[spec.change]()
        self.law = LAWS[spec.law](reg_len=spec.reg_len, variant=spec.jump_variant)

        self.severity = float(spec.severity)
        self.instance_seed = 0
        # Per-environment change-intensity profile. Drawn from the instance
        # seed (or the public schedule for class B) in `reseed`; the initial
        # environment is always 1.0 so the starting problem is unchanged.
        self.gain = np.ones(self.n_env)

        # Box: roleswitch may rotate any variable into the free slot, and it
        # must stay non-negative because f1 enters a square root.
        if spec.mode == "roleswitch":
            self.lower = np.zeros(self.D)
            self.upper = np.ones(self.D)
        else:
            self.lower = np.full(self.D, -1.8)
            self.upper = np.full(self.D, 1.8)
            n_free = 2 if self.M == 3 else 1
            self.lower[:n_free] = 0.0
            self.upper[:n_free] = 1.0

        self.reseed(0)

    # -- instance ----------------------------------------------------------

    def reseed(self, seed: int) -> None:
        self.instance_seed = int(seed)

        # The schedule is the *problem identity* for classes C/D/E: two
        # algorithms with the same instance seed must face the same schedule.
        sched = self.spec.schedule_seed
        schedule_seed = int(sched) if sched is not None else int(seed)
        self.law.reseed(np.random.default_rng(schedule_seed), self.n_env, self.spec.reg_len)
        self._m_abs_max = max(abs(float(self.law.m(t)))
                              for t in self.environment_times) or 1.0

        # The intensity profile is part of the instance, exactly like the
        # schedule. The old default draws class B's profile from the public
        # schedule seed, which makes B1/B2 identical across 107/207/307/407.
        # `intensity_seed="instance"` makes the severity instance-dependent
        # while leaving the public timetable intact; new experiments use it.
        intensity_seed = schedule_seed if self.intensity_seed == "schedule" else seed
        self._draw_intensity(intensity_seed)

        # Landscape constants belong to the instance as well, but must not
        # consume the schedule stream: use a separate, decorrelated stream.
        base_seed = int(seed) ^ 0x9E3779B9
        self.base.configure(np.random.default_rng(base_seed), self.D)

    # -- change intensity ---------------------------------------------------

    def _draw_intensity(self, seed: int) -> None:
        """Draw a smooth per-environment magnitude profile.

        `gain_0 = 1` keeps the first environment identical across instances;
        only the *changes* are heterogeneous. The profile is normalised so its
        mean over the changing environments is the instance scale, then capped
        so the Pareto set never leaves the decision box (see `_gain_cap`).
        """
        if not self.spec.intensity:
            self.gain = np.ones(self.n_env)
            return

        rng = np.random.default_rng(int(seed) ^ 0x5EEDBEEF)
        rho = float(self.intensity_rho)
        z = np.empty(self.n_env)
        z[0] = rng.normal()
        for k in range(1, self.n_env):
            z[k] = rho * z[k - 1] + np.sqrt(1.0 - rho ** 2) * rng.normal()
        scale = rng.uniform(0.7, 1.3)
        raw = scale * np.exp(float(self.intensity_log_sd) * z)
        tail = raw[1:]
        raw = raw / (tail.mean() if tail.size and tail.mean() > 0 else 1.0)
        raw[0] = 1.0
        self.gain = self._cap_gain(raw)

    def _gain_cap(self) -> float:
        """Largest uniform gain that keeps the Pareto set inside the box.

        Mode amplitudes are analytic: translate uses `severity * m`, shape adds
        `severity * amp_max`, chain adds `kappa`, roleswitch is centred at 0.5
        with a 0.30 swing in a [0, 1] box. Capping the *whole* profile (rather
        than each environment) preserves the relative heterogeneity.
        """
        times = self.environment_times
        m_abs = max(abs(float(self.law.m(t))) for t in times)
        if m_abs <= 1e-12 or self.spec.mode == "static":
            return float("inf")
        if self.spec.mode.startswith("seg_"):
            # The segmented mode is as restrictive as its stricter half.
            parts = self.spec.mode[4:].split("_")
            return min(self._mode_cap(p, m_abs) for p in parts)
        return self._mode_cap(self.spec.mode, m_abs)

    @staticmethod
    def _mode_cap(mode: str, m_abs: float) -> float:
        if mode == "roleswitch":
            return 0.5 / (0.30 * m_abs)
        if mode == "translate":
            return 1.8 / m_abs
        if mode == "shape":
            return 1.8 / (m_abs + 0.40)
        if mode == "chain":
            return 1.8 / (m_abs + 0.35)
        return float("inf")

    def _cap_gain(self, gain: np.ndarray) -> np.ndarray:
        cap = self._gain_cap()
        peak = float(np.max(np.abs(gain)))
        if np.isfinite(cap) and peak > cap:
            gain = gain * (cap / peak)
            gain[0] = 1.0
        return gain

    def _env_index(self, u: float) -> int:
        # Environment times are `linspace(0, 1, n_env)`; `evaluate` is also
        # called at arbitrary t by admission checks, so snap to the nearest
        # environment rather than assuming an exact grid point.
        k = int(round(float(u) * (self.n_env - 1)))
        return min(max(k, 0), self.n_env - 1)

    def _gain_at(self, u: float) -> float:
        return float(self.gain[self._env_index(u)])

    # -- evaluation --------------------------------------------------------

    def _m_eff(self, u: float) -> tuple[float, float]:
        """Effective change signal and intensity gain.

        `hard` is a diagnostic mode: after the first environment every change
        is a maximal swing in the opposite direction.  It exists to separate
        "the benchmark never asks for a response" from "the credit signal is
        wrong", not as a production setting.
        """
        m = float(self.law.m(u))
        gain = self._gain_at(u)
        if self.intensity_mode == "hard" and self._env_index(u) > 0:
            sign = 1.0 if m >= 0 else -1.0
            return sign * self._m_abs_max, 1.0
        return m, gain

    def _mode_context(self, u: float) -> ModeContext:
        m, gain = self._m_eff(u)
        return ModeContext(
            severity=self.severity,
            m=m,
            amp_max=0.40,
            kappa=0.35,
            gain=gain,
        )

    def evaluate(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        u = float(t)
        m, gain = self._m_eff(u)
        free = self.mode.free_indices(u, self.D, self.M)
        ctx = self._mode_context(u)

        # Every non-free coordinate tracks the value of the free coordinate(s),
        # so ps is built row by row from u_rows rather than from the scalar t.
        # With three objectives the two free coordinates are averaged, matching
        # `pareto_set` exactly; any mismatch would leave g != 1 on the PS.
        if self.M == 2:
            u_rows = X[:, free[0]]
        else:
            u_rows = 0.5 * (X[:, free[0]] + X[:, free[1]])
        ps = np.empty_like(X)
        for i in range(self.D):
            if i in free:
                ps[:, i] = 0.0
            else:
                ps[:, i] = self.mode.ps(i, u_rows, u, ctx)

        dev = X - ps
        for r in free:
            dev[:, r] = 0.0
        g = self.base.g(dev, u)

        if self.M == 2:
            r = free[0]
            f1 = X[:, r]
            h = self.geom.front(np.clip(f1 / g, 0.0, 1.0), f1=f1)
            raw = np.column_stack([f1, g * h])
        elif self.M == 3:
            r1, r2 = free[:2]
            a = 0.5 * np.pi * X[:, r1]
            b = 0.5 * np.pi * X[:, r2]
            raw = np.column_stack([
                g * np.cos(a) * np.cos(b),
                g * np.cos(a) * np.sin(b),
                g * np.sin(a),
            ])
        else:
            raise ValueError("only M = 2 or 3 is supported")

        # The front change is driven by the same schedule and the same
        # per-environment gain as the Pareto set (shared timetable).
        return self.change.apply(raw, m * gain)

    # -- reference front ---------------------------------------------------

    def pareto_set(self, t: float, n: int = 1000) -> np.ndarray:
        u = float(t)
        free = self.mode.free_indices(u, self.D, self.M)
        ctx = self._mode_context(u)

        if self.M == 2:
            grid = np.linspace(0.0, 1.0, int(n))
            X = np.empty((grid.size, self.D))
            for i in range(self.D):
                if i in free:
                    X[:, i] = 0.0
                else:
                    X[:, i] = self.mode.ps(i, grid, u, ctx)
            X[:, free[0]] = grid
            return X

        m_side = max(2, int(np.ceil(np.sqrt(n))))
        u1, u2 = [a.ravel() for a in np.meshgrid(
            np.linspace(0.0, 1.0, m_side), np.linspace(0.0, 1.0, m_side)
        )]
        mid = 0.5 * (u1 + u2)
        X = np.empty((u1.size, self.D))
        for i in range(self.D):
            if i in free:
                X[:, i] = 0.0
            else:
                X[:, i] = self.mode.ps(i, mid, u, ctx)
        X[:, free[0]] = u1
        X[:, free[1]] = u2
        return X

    def true_pf(self, t: float, n: int = 2000) -> np.ndarray:
        F = self.evaluate(self.pareto_set(t, n), t)
        return F[nondominated_mask(F)]

    # -- diagnostics -------------------------------------------------------

    def describe(self, t: float = 0.0) -> dict:
        u = float(t)
        return {
            "pid": self.spec.pid,
            "class": self.spec.cls,
            "base": self.spec.base,
            "mode": self.spec.mode,
            "geometry": self.spec.geometry,
            "change": self.spec.change,
            "law": self.spec.law,
            "M": self.M,
            "instance_seed": self.instance_seed,
            "m": self.law.m(u),
            "free": self.mode.free_indices(u, self.D, self.M),
            "law_detail": self.law.describe(),
        }
