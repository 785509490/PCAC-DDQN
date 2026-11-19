"""Time laws: how the change parameter m(u) evolves over u in [0, 1].

Five laws, and they are five *different functional forms*, not one form with
different parameters:

    drift   clamp(2u - 1)
    swing   sin(2 pi u)
    rush    clamp((2u - 1)^3)
    jump    piecewise-constant, drawn per instance
    regime  a meta-law that switches among hold / drift / oscillate / accelerate

The old MATLAB version wrote these on u = t/3 while the driver only reached
t = 1, so `swing` never completed a half cycle and `drift` covered a third of
its travel. Here u == t by construction; the number of environments is a
separate knob (`nt`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KIND_HOLD, KIND_DRIFT, KIND_OSC, KIND_ACCEL = 1, 2, 3, 4


def _clamp(x, lo=-1.0, hi=1.0):
    return np.clip(x, lo, hi)


class Law:
    name = "law"

    def reseed(self, rng: np.random.Generator, n_env: int, reg_len: int = 3) -> None:
        """Draw whatever this law needs. Stateless laws ignore it."""

    def m(self, u: float) -> float:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"law": self.name}


class Drift(Law):
    name = "drift"

    def m(self, u: float) -> float:
        return float(_clamp(2.0 * u - 1.0))


class Swing(Law):
    name = "swing"

    def m(self, u: float) -> float:
        return float(np.sin(2.0 * np.pi * u))


class Rush(Law):
    name = "rush"

    def m(self, u: float) -> float:
        return float(_clamp((2.0 * u - 1.0) ** 3))


class Jump(Law):
    """Piecewise-constant jumps.

    `hetero` draws a different step size at every jump; `homo` uses one fixed
    step size and a random sign. The distinction is the basis of the E-class
    negative control: with a fixed step, the observed displacement carries no
    information about *how far* the problem moved, so the best response cannot
    depend on the environment.
    """

    name = "jump"

    def __init__(self, variant: str = "hetero", fixed_step: float = 0.6):
        if variant not in {"hetero", "homo"}:
            raise ValueError("variant must be 'hetero' or 'homo'")
        self.variant = variant
        self.fixed_step = float(fixed_step)
        self.values = np.zeros(2)

    def reseed(self, rng: np.random.Generator, n_env: int, reg_len: int = 3) -> None:
        n_jump = n_env
        vals = np.empty(n_jump)
        vals[0] = rng.uniform(-0.5, 0.5)
        for j in range(1, n_jump):
            if self.variant == "hetero":
                mag = np.exp(rng.uniform(np.log(0.05), np.log(0.8)))
            else:
                mag = self.fixed_step
            sign = 1.0 if rng.random() < 0.5 else -1.0
            nxt = vals[j - 1] + sign * mag
            # 反射而不是截断:截断会让两端的跳变幅度变小,把"同分布"破坏掉。
            while nxt > 1.0 or nxt < -1.0:
                if nxt > 1.0:
                    nxt = 2.0 - nxt
                if nxt < -1.0:
                    nxt = -2.0 - nxt
            vals[j] = nxt
        self.values = vals

    def m(self, u: float) -> float:
        j = int(round(u * (len(self.values) - 1)))
        j = min(max(j, 0), len(self.values) - 1)
        return float(self.values[j])


class Regime(Law):
    """A meta-law: the *kind* of motion switches at random boundaries.

    The centre is continuous across a boundary -- a regime change alters the
    dynamics, not the position -- so the problem stays well posed and nothing
    is clipped back into the box.

    `constrained` rejects draws whose type sequence has no contrast (all one
    kind, or only "moving" kinds). That constraint is what guarantees the
    environments actually demand different responses.
    """

    name = "regime"

    def __init__(self, reg_len: int = 3, constrained: bool = True, jitter: float = 0.02):
        self.reg_len = int(reg_len)
        self.constrained = bool(constrained)
        self.jitter = float(jitter)
        self.bounds = np.array([0.0, 1.0])
        self.types = np.array([KIND_DRIFT])
        self.centres = np.array([0.0, 0.5])

    def reseed(self, rng: np.random.Generator, n_env: int, reg_len: int | None = None) -> None:
        if reg_len is not None:
            self.reg_len = int(reg_len)
        n_reg = max(1, int(np.ceil(n_env / self.reg_len)))

        for _ in range(5000):
            types = rng.integers(1, 5, size=n_reg)
            if not self.constrained or self._has_contrast(types):
                break
        else:
            types = np.tile(np.array([KIND_HOLD, KIND_DRIFT]), n_reg)[:n_reg]

        edges = np.linspace(0.0, 1.0, n_reg + 1)
        if n_reg > 1 and self.jitter > 0:
            inner = edges[1:-1] + rng.uniform(-self.jitter, self.jitter, size=n_reg - 1)
            edges[1:-1] = np.sort(np.clip(inner, 0.0, 1.0))
        edges[0], edges[-1] = 0.0, 1.0

        centres = rng.uniform(-0.8, 0.8, size=n_reg + 1)
        # A hold segment stays at its left centre, so the boundary it ends on
        # must carry that same value; otherwise the centre jumps and the
        # "regime change alters dynamics, not position" property is lost.
        for k in range(n_reg):
            if types[k] == KIND_HOLD:
                centres[k + 1] = centres[k]

        self.bounds = edges
        self.types = types
        self.centres = centres

    @staticmethod
    def _has_contrast(types: np.ndarray) -> bool:
        if len(np.unique(types)) < 2:
            return False
        if np.any(types[1:] == types[:-1]):
            return False
        static = np.any((types == KIND_HOLD) | (types == KIND_OSC))
        moving = np.any((types == KIND_DRIFT) | (types == KIND_ACCEL))
        return bool(static and moving)

    def _segment(self, u: float) -> int:
        b = self.bounds
        k = int(np.searchsorted(b, u, side="right") - 1)
        return min(max(k, 0), len(self.types) - 1)

    def m(self, u: float) -> float:
        k = self._segment(u)
        b0, b1 = self.bounds[k], self.bounds[k + 1]
        s = 0.0 if b1 <= b0 else (u - b0) / (b1 - b0)
        m0, m1 = self.centres[k], self.centres[k + 1]
        kind = self.types[k]
        if kind == KIND_HOLD:
            return float(m0)
        if kind == KIND_DRIFT:
            return float(m0 + (m1 - m0) * s)
        if kind == KIND_OSC:
            return float(m0 + (m1 - m0) * s + 0.3 * np.sin(2.0 * np.pi * s))
        if kind == KIND_ACCEL:
            return float(m0 + (m1 - m0) * s ** 3)
        raise ValueError(f"bad regime kind {kind}")

    def describe(self) -> dict:
        return {
            "law": self.name,
            "bounds": self.bounds.tolist(),
            "types": self.types.tolist(),
            "centres": self.centres.tolist(),
        }


LAWS = {
    "drift": lambda **kw: Drift(),
    "swing": lambda **kw: Swing(),
    "rush": lambda **kw: Rush(),
    "jump": lambda variant="hetero", **kw: Jump(variant=variant),
    "regime": lambda reg_len=3, constrained=True, **kw: Regime(
        reg_len=reg_len, constrained=constrained
    ),
}
