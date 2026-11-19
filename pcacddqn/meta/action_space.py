"""The meta action space: a response program plus a DE configuration.

The action table is deliberately small and versioned. It is produced by the
configuration screen (`scripts/screen_actions.py`) and must contain the current
memory operator (m=60, rho=0.5, DE rand/1 CR=F=0.5) as a safe default. The
provisional table below is used by the Phase-0 interface tests and by the
screen itself; the screen overwrites it with a JSON table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..core.population import Population
from ..core.problem import Problem
from ..core.variation import config_space
from .memory import MemoryConfig, MemoryNSGA2
from .response import Response, apply_response


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def config_id(strategy: str, CR: float, F: float) -> int:
    """Index of a DE configuration in `config_space()` (float-safe)."""
    for i, (s, cr, f) in enumerate(config_space()):
        if s == strategy and abs(cr - CR) < 1e-12 and abs(f - F) < 1e-12:
            return i
    raise ValueError(f"unknown DE configuration {(strategy, CR, F)!r}")


@dataclass(frozen=True)
class ResponseProgram:
    kind: str                      # inherit | restart | elite | memory | memory_recent | extrapolate
    rho: float = 0.0
    frac: float = 0.90
    sigma: float = 0.10
    m: int = 60
    max_fronts: int = 20
    recent_only: bool = False
    alpha: float = 0.0
    p: float = 0.0


@dataclass(frozen=True)
class MetaAction:
    name: str
    program: ResponseProgram
    config_id: int

    @property
    def de(self) -> tuple[str, float, float]:
        return config_space()[self.config_id]

    @property
    def is_memory(self) -> bool:
        return self.program.kind in {"memory", "memory_recent"}


C4 = config_id("rand/1", 0.5, 0.5)

ACTION_TABLE: tuple[MetaAction, ...] = (
    MetaAction("inherit", ResponseProgram("inherit"), C4),
    MetaAction("elite-90-10", ResponseProgram("elite", frac=0.90, sigma=0.10), C4),
    MetaAction("memory-r25", ResponseProgram("memory", rho=0.25, m=60), C4),
    MetaAction("memory-r50", ResponseProgram("memory", rho=0.50, m=60), C4),
    MetaAction("memory-r75", ResponseProgram("memory", rho=0.75, m=60), C4),
    MetaAction("memory-recent", ResponseProgram("memory_recent", rho=0.50, m=60,
                                                recent_only=True), C4),
    MetaAction("restart", ResponseProgram("restart"), C4),
    MetaAction("memory-r50-best1",
               ResponseProgram("memory", rho=0.50, m=60),
               config_id("best/1", 0.5, 0.5)),
    MetaAction("memory-r50-ctb1",
               ResponseProgram("memory", rho=0.50, m=60),
               config_id("current-to-best/1", 0.5, 0.5)),
    MetaAction("memory-r50-rand2",
               ResponseProgram("memory", rho=0.50, m=60),
               config_id("rand/2", 0.5, 0.5)),
    MetaAction("memory-r50-ctr1",
               ResponseProgram("memory", rho=0.50, m=60),
               config_id("current-to-rand/1", 0.5, 0.5)),
)

DEFAULT_ACTION = 3
N_ACTIONS = len(ACTION_TABLE)


def memory_cfg(action: MetaAction) -> MemoryConfig:
    p = action.program
    return MemoryConfig(
        m=int(p.m),
        max_fronts=int(p.max_fronts),
        recent_only=bool(p.recent_only or p.kind == "memory_recent"),
        frac=float(p.frac),
        sigma=float(p.sigma),
        rho0=_logit(p.rho),
        online=False,
        filter_archive=True,
    )


def apply_meta_action(
    alg: MemoryNSGA2,
    problem: Problem,
    pop: Population,
    action: MetaAction,
    rng: np.random.Generator,
    t_prev: float,
    t_cur: float,
    prev_dec: np.ndarray | None = None,
) -> Population:
    """Apply one action and return the population the driver must evaluate.

    The caller is responsible for calling `alg.remember(pop)` *after* this
    function. For memory actions this is essential: `respond` samples the
    archive as it was before the change.
    """
    alg.set_config(*action.de)
    p = action.program

    if action.is_memory:
        alg.cfg = memory_cfg(action)
        return alg.respond(problem, pop, t_prev, t_cur)

    if p.kind == "inherit":
        return pop

    if p.kind == "restart":
        n, D = pop.dec.shape
        dec = problem.lower + rng.random((n, D)) * (problem.upper - problem.lower)
        return Population(problem.clip(dec), np.zeros((n, problem.M)))

    if p.kind == "elite":
        cell = Response(kind="elite_front", frac=float(p.frac),
                        sigma=float(p.sigma))
        dec = apply_response(problem, pop, prev_dec, cell, rng)
        return Population(dec, np.zeros((len(dec), problem.M)))

    if p.kind == "extrapolate":
        cell = Response(alpha=float(p.alpha), sigma=float(p.sigma),
                        p=float(p.p), kind="global")
        dec = apply_response(problem, pop, prev_dec, cell, rng)
        return Population(dec, np.zeros((len(dec), problem.M)))

    raise ValueError(f"unknown response kind {p.kind!r}")


def save_action_table(path: str | Path,
                      table: tuple[MetaAction, ...] = ACTION_TABLE) -> None:
    rows = [
        {"name": a.name, "config_id": int(a.config_id),
         "program": asdict(a.program)}
        for a in table
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_action_table(path: str | Path) -> tuple[MetaAction, ...]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    table = []
    for r in rows:
        prog = ResponseProgram(**r["program"])
        table.append(MetaAction(r["name"], prog, int(r["config_id"])))
    return tuple(table)
