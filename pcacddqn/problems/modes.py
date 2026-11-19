"""Modes: how the Pareto set moves (the decision-space layer).

`ps_i(u, t)` gives the target value of the i-th non-free coordinate when the
free coordinate takes value u. Modes never touch g; that is the base's job.

`static` exists so that Type III problems (front moves, set does not) can be
expressed without a special case in the problem class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModeContext:
    severity: float = 1.0
    m: float = 0.0
    amp_max: float = 0.40
    kappa: float = 0.35
    gain: float = 1.0       # per-environment change intensity


class Mode:
    name = "mode"

    def free_indices(self, u: float, D: int, M: int) -> tuple[int, ...]:
        return (0, 1) if M >= 3 else (0,)

    def centre(self, u: float, ctx: ModeContext) -> float:
        return ctx.severity * ctx.m * ctx.gain

    def ps(self, i: int, u: float, t: float, ctx: ModeContext) -> float:
        return self.centre(u, ctx)


class Static(Mode):
    """Pareto set does not move. Used for Type III problems."""

    name = "static"

    def centre(self, u: float, ctx: ModeContext) -> float:
        return 0.5


class Translate(Mode):
    name = "translate"


class Shape(Mode):
    """The set bends while its centre also moves."""

    name = "shape"

    def ps(self, i: int, u: float, t: float, ctx: ModeContext) -> float:
        amp = (ctx.severity * ctx.amp_max * ctx.gain
               * 0.5 * (1.0 - np.cos(np.pi * u)))
        return self.centre(u, ctx) + amp * np.sin(2.0 * np.pi * u + 0.5 * np.pi * t)


class Chain(Mode):
    """A travelling wave along the coordinate index."""

    name = "chain"

    def ps(self, i: int, u: float, t: float, ctx: ModeContext) -> float:
        return self.centre(u, ctx) + ctx.kappa * ctx.gain * np.cos(
            2.0 * np.pi * u + 0.7 * i + 0.4 * np.pi * t
        )


class RoleSwitch(Mode):
    """The free coordinate rotates between variables (the DF2 mechanism)."""

    name = "roleswitch"

    def centre(self, u: float, ctx: ModeContext) -> float:
        return 0.5 + 0.30 * ctx.severity * ctx.m * ctx.gain

    def free_indices(self, u: float, D: int, M: int) -> tuple[int, ...]:
        r1 = int(np.floor((D - 1) * abs(np.sin(0.5 * np.pi * u)))) % D
        if M <= 2:
            return (r1,)
        return (r1, (r1 + 1) % D)


class Segmented(Mode):
    """Switch between two modes at a fixed normalised time.

    Used by the F-class study: the *kind* of motion changes mid-run, so the
    relevance of the archive changes with it. Both children use the same free
    coordinates, which keeps the Pareto set continuous at the boundary.
    """

    def __init__(self, first: Mode, second: Mode, switch: float = 0.5):
        self.first = first
        self.second = second
        self.switch = float(switch)

    def _at(self, u: float) -> Mode:
        return self.first if u < self.switch else self.second

    def free_indices(self, u: float, D: int, M: int) -> tuple[int, ...]:
        return self._at(u).free_indices(u, D, M)

    def centre(self, u: float, ctx: ModeContext) -> float:
        return self._at(u).centre(u, ctx)

    def ps(self, i: int, u, t: float, ctx: ModeContext):
        u_arr = np.asarray(u)
        if u_arr.ndim == 0:
            return self._at(float(u_arr)).ps(i, float(u_arr), t, ctx)
        out = np.empty_like(u_arr, dtype=float)
        mask = u_arr < self.switch
        if mask.any():
            out[mask] = self.first.ps(i, u_arr[mask], t, ctx)
        if (~mask).any():
            out[~mask] = self.second.ps(i, u_arr[~mask], t, ctx)
        return out


MODES = {
    "static": lambda: Static(),
    "translate": lambda: Translate(),
    "shape": lambda: Shape(),
    "chain": lambda: Chain(),
    "roleswitch": lambda: RoleSwitch(),
    "seg_translate_shape": lambda: Segmented(Translate(), Shape()),
    "seg_shape_translate": lambda: Segmented(Shape(), Translate()),
    "seg_translate_chain": lambda: Segmented(Translate(), Chain()),
    "seg_chain_translate": lambda: Segmented(Chain(), Translate()),
}
