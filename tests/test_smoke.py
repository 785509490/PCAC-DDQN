"""Smoke test for the benchmark, the inner loop, and the action table.

No stored data and no experiment pipeline are involved. The test builds three
problems from different classes, runs a short NSGA-II loop, and checks that the
IGD is finite and the action table has the paper's 14 actions.
"""

from __future__ import annotations

import pathlib

import numpy as np

from pcacddqn.algorithms.nsga2 import NSGA2
from pcacddqn.core.metrics import igd
from pcacddqn.meta.action_space import load_action_table
from pcacddqn.problems.suite import build

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_inner_loop_on_three_classes() -> None:
    for pid in ("A1", "B3", "C1"):
        problem = build(pid, 107, n_env=3, taut=20, T0=10)
        t = float(problem.environment_times[0])
        alg = NSGA2(n_pop=20)
        alg.bind(np.random.default_rng(0))
        pop = alg.initialize(problem, t)
        for _ in range(5):
            pop = alg.step(problem, pop, t)
        assert pop.obj.shape == (20, problem.M)
        value = igd(pop.obj, problem.true_pf(t, n=200))
        assert np.isfinite(value) and value >= 0.0


def test_action_table_has_14_actions() -> None:
    table = load_action_table(str(REPO_ROOT / "configs" / "action_table.json"))
    assert len(table) == 14
    kinds = {a.program.kind for a in table}
    assert {"memory", "elite", "inherit"} <= kinds
