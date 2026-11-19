"""Drivers: one closed-loop run and paired batches of runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.algorithm import Algorithm
from ..core.metrics import hv, igd
from ..core.population import Population
from ..core.problem import Problem


@dataclass
class RunResult:
    migd: float
    per_env_igd: np.ndarray
    per_env_hv: np.ndarray
    start_igd: np.ndarray
    population: object = None
    log: dict = field(default_factory=dict)
    per_gen_igd: np.ndarray | None = None
    pop_snapshots: np.ndarray | None = None
    snapshot_env: np.ndarray | None = None
    snapshot_phase: np.ndarray | None = None
    change_gen: np.ndarray | None = None
    total_obj_evals: int = 0


@dataclass
class MetaRunResult:
    migd: float
    per_env_igd: np.ndarray
    per_env_hv: np.ndarray
    start_igd: np.ndarray
    states: np.ndarray          # (n_changes, STATE_DIM), pre-action + previous post
    next_states: np.ndarray     # (n_changes, STATE_DIM)
    actions: np.ndarray         # (n_changes,)
    rewards: np.ndarray         # (n_changes,)
    dones: np.ndarray           # (n_changes,)
    response_evals: int
    probe_evals: int = 0
    population: object = None
    log: dict = field(default_factory=dict)
    per_gen_igd: np.ndarray | None = None
    pop_snapshots: np.ndarray | None = None
    snapshot_env: np.ndarray | None = None
    snapshot_phase: np.ndarray | None = None
    change_gen: np.ndarray | None = None
    total_obj_evals: int = 0



def run_dynamic(
    problem: Problem,
    algorithm: Algorithm,
    instance_seed: int = 0,
    run_seed: int = 0,
    record_pop: bool = False,
    record_trace: bool = False,
) -> RunResult:
    """Run one closed-loop episode.

    Order of operations at a change, matching `runDynamicAAD`:
      1. re-evaluate the carried population in the new environment;
      2. let the algorithm react (it sees the re-evaluated population);
      3. re-evaluate anything the reaction created;
      4. run `taut` generations.

    `record_trace=True` additionally records per-generation IGD and population
    objective snapshots at the start and end of every environment. It does not
    change the trajectory or any metric.
    """
    problem.reseed(instance_seed)
    algorithm.bind(np.random.default_rng(run_seed))

    times = problem.environment_times
    pop = algorithm.initialize(problem, times[0])
    n_pop = len(pop.dec)
    n_changes = len(times) - 1

    per_env_igd: list[float] = []
    per_env_hv: list[float] = []
    start_igd: list[float] = []
    per_gen_igd: list[float] = []
    snapshots: list[np.ndarray] = []
    snap_env: list[int] = []
    snap_phase: list[int] = []

    ref = problem.true_pf(times[0])
    start_igd.append(igd(pop.obj, ref))

    for k, t in enumerate(times):
        if k > 0:
            # The population is evaluated in the environment it has just left;
            # re-evaluate it in the new one before asking for a response.
            pop = pop.copy()
            pop.obj = problem.evaluate(pop.dec, t)
            pop = algorithm.on_change(problem, pop, times[k - 1], t)
            pop.obj = problem.evaluate(pop.dec, t)
            # Reference front of this environment; also used at its end.
            ref = problem.true_pf(t)

        if record_trace:
            snapshots.append(pop.obj.copy())
            snap_env.append(k)
            snap_phase.append(0)          # after the change response

        n_gen = problem.T0 if k == 0 else problem.taut
        for _ in range(n_gen):
            pop = algorithm.step(problem, pop, t)
            if record_trace:
                per_gen_igd.append(igd(pop.obj, ref))

        per_env_igd.append(igd(pop.obj, ref))
        # The HV reference must follow the front: on a Type II problem the
        # front moves, and a reference frozen at t = 0 silently drops points.
        per_env_hv.append(hv(pop.obj, np.max(ref, axis=0) * 1.05))
        if record_trace:
            snapshots.append(pop.obj.copy())
            snap_env.append(k)
            snap_phase.append(1)          # end of the environment
        if k < len(times) - 1:
            start_igd.append(igd(problem.evaluate(pop.dec, times[k + 1]), problem.true_pf(times[k + 1])))

    response_evals = int(getattr(algorithm, "response_evals", 0))
    # Algorithm-side objective evaluations under the generation protocol.
    # Reference-front construction is excluded; response evaluations that the
    # algorithm reports are added explicitly.
    total_obj_evals = int(
        n_pop * (problem.T0 + n_changes * problem.taut)   # one N per generation
        + n_pop * n_changes                               # re-evaluate carried pop
        + n_pop * n_changes                               # evaluate the response
        + n_pop * n_changes                               # start-IGD probe
        + n_pop                                            # initial population
        + response_evals
    )
    change_gen = np.asarray(
        [problem.T0 + (k - 1) * problem.taut for k in range(1, len(times))],
        dtype=int,
    )

    return RunResult(
        migd=float(np.mean(per_env_igd)),
        per_env_igd=np.asarray(per_env_igd),
        per_env_hv=np.asarray(per_env_hv),
        start_igd=np.asarray(start_igd),
        population=pop if record_pop else None,
        log={"times": times, "change_gen": change_gen,
             "total_obj_evals": total_obj_evals,
             "response_evals": response_evals},
        per_gen_igd=np.asarray(per_gen_igd) if record_trace else None,
        pop_snapshots=np.asarray(snapshots) if record_trace else None,
        snapshot_env=np.asarray(snap_env) if record_trace else None,
        snapshot_phase=np.asarray(snap_phase) if record_trace else None,
        change_gen=change_gen,
        total_obj_evals=total_obj_evals,
    )



def run_meta(
    problem: Problem,
    policy,
    instance_seed: int = 0,
    run_seed: int = 0,
    n_pop: int = 100,
    action_table=None,
    reward_scale: float | None = None,
    record_pop: bool = False,
    probe: bool = True,
    record_trace: bool = False,
) -> MetaRunResult:
    """One closed-loop run with a per-change meta-policy.

    This is a sibling of `run_dynamic`; the latter is untouched so the old
    baselines stay bit-reproducible. The order of operations at a change is
    the same:

      1. re-evaluate the carried population in the new environment;
      2. build the state and let the policy choose an action;
      3. apply the response (memory samples the *past* archive first);
      4. append the incoming front to the archive (`remember`);
      5. re-evaluate the response and run `taut` generations.

    The action at change k affects the end-of-environment IGD at k, which is
    the reward. The return is therefore a scaled negative MIGD.
    """
    from ..meta.action_space import ACTION_TABLE, apply_meta_action
    from ..meta.memory import MemoryConfig, MemoryNSGA2
    from ..meta.metastate import (
        PRE_DIM, STATE_DIM, MetaStateTracker, probe_archive,
    )

    table = ACTION_TABLE if action_table is None else action_table
    problem.reseed(instance_seed)
    policy.reset()

    alg = MemoryNSGA2(MemoryConfig(m=60, online=False, rho0=0.0), n_pop=n_pop)
    alg.bind(np.random.default_rng(run_seed))
    tracker = MetaStateTracker(
        problem.lower, problem.upper, problem.M,
        archive_capacity=alg.cfg.max_fronts,
        taut=problem.taut, n_env=problem.n_env,
    )

    times = problem.environment_times
    pop = alg.initialize(problem, times[0])
    n_pop = len(pop.dec)
    n_changes = len(times) - 1
    tracker.reset(pop)
    if reward_scale is None:
        reward_scale = float(np.mean(np.ptp(pop.obj, axis=0))) + 1e-12
    reward_scale = max(float(reward_scale), 1e-12)

    per_env_igd: list[float] = []
    per_env_hv: list[float] = []
    start_igd: list[float] = []
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    per_gen_igd: list[float] = []
    snapshots: list[np.ndarray] = []
    snap_env: list[int] = []
    snap_phase: list[int] = []

    response_rng = np.random.default_rng((run_seed, 0xAAD))
    # The probe has its own stream: it must not perturb the response policy's
    # RNG, or adding the state feature would silently change the action.
    probe_rng = np.random.default_rng((run_seed, 0xBEEF))
    probe_evals = 0
    dec_history: list[np.ndarray] = []
    post = np.zeros(STATE_DIM - PRE_DIM, dtype=float)

    ref = problem.true_pf(times[0])
    start_igd.append(igd(pop.obj, ref))

    for k, t in enumerate(times):
        if k > 0:
            pop = pop.copy()
            prev_obj = pop.obj.copy()
            pop.obj = problem.evaluate(pop.dec, t)

            probe_res = (probe_archive(problem, pop, alg.archive, 60,
                                       probe_rng, t)
                         if probe else None)
            if probe_res is not None:
                probe_evals += probe_res.m
            pre = tracker.pre_action(pop, prev_obj, alg.archive,
                                     probe=probe_res)
            state = np.concatenate([pre, post])
            a_idx = int(policy.act(state))
            if not 0 <= a_idx < len(table):
                raise ValueError(f"action index {a_idx} outside table")
            action = table[a_idx]

            prev_dec = dec_history[-2] if len(dec_history) >= 2 else None
            out = apply_meta_action(
                alg, problem, pop, action, response_rng,
                times[k - 1], t, prev_dec=prev_dec,
            )
            alg.remember(pop)
            pop = Population(out.dec, problem.evaluate(out.dec, t))
            post = tracker.post_response(alg, action.is_memory)
            ref = problem.true_pf(t)

            states.append(state)
            actions.append(a_idx)

        if record_trace:
            snapshots.append(pop.obj.copy())
            snap_env.append(k)
            snap_phase.append(0)          # after the change response

        n_gen = problem.T0 if k == 0 else problem.taut
        for _ in range(n_gen):
            pop = alg.step(problem, pop, t)
            if record_trace:
                per_gen_igd.append(igd(pop.obj, ref))
        dec_history.append(pop.dec.copy())

        env_igd = igd(pop.obj, ref)
        per_env_igd.append(env_igd)
        per_env_hv.append(hv(pop.obj, np.max(ref, axis=0) * 1.05))
        if record_trace:
            snapshots.append(pop.obj.copy())
            snap_env.append(k)
            snap_phase.append(1)          # end of the environment
        if k > 0:
            rewards.append(float(np.clip(-env_igd / reward_scale, -20.0, 0.0)))
        if k < len(times) - 1:
            nt = times[k + 1]
            start_igd.append(
                igd(problem.evaluate(pop.dec, nt), problem.true_pf(nt))
            )

    states_arr = np.asarray(states, dtype=float)
    if len(states_arr) == 0:
        states_arr = np.zeros((0, STATE_DIM))
    next_states = np.vstack([states_arr[1:], np.zeros((1, STATE_DIM))]) \
        if len(states_arr) else states_arr
    dones = np.zeros(len(states_arr), dtype=float)
    if len(dones):
        dones[-1] = 1.0

    response_evals = int(alg.response_evals)
    total_obj_evals = int(
        n_pop * (problem.T0 + n_changes * problem.taut)   # one N per generation
        + n_pop * n_changes                               # re-evaluate carried pop
        + n_pop * n_changes                               # evaluate the response
        + n_pop * n_changes                               # start-IGD probe
        + n_pop                                            # initial population
        + response_evals
        + probe_evals
    )
    change_gen = np.asarray(
        [problem.T0 + (k - 1) * problem.taut for k in range(1, len(times))],
        dtype=int,
    )

    return MetaRunResult(
        migd=float(np.mean(per_env_igd)),
        per_env_igd=np.asarray(per_env_igd),
        per_env_hv=np.asarray(per_env_hv),
        start_igd=np.asarray(start_igd),
        states=states_arr,
        next_states=next_states,
        actions=np.asarray(actions, dtype=int),
        rewards=np.asarray(rewards, dtype=float),
        dones=dones,
        response_evals=response_evals,
        probe_evals=int(probe_evals),
        population=pop if record_pop else None,
        log={"times": times,
             "action_names": [table[a].name for a in actions],
             "probe_evals": int(probe_evals),
             "response_evals": response_evals,
             "change_gen": change_gen,
             "total_obj_evals": total_obj_evals},
        per_gen_igd=np.asarray(per_gen_igd) if record_trace else None,
        pop_snapshots=np.asarray(snapshots) if record_trace else None,
        snapshot_env=np.asarray(snap_env) if record_trace else None,
        snapshot_phase=np.asarray(snap_phase) if record_trace else None,
        change_gen=change_gen,
        total_obj_evals=total_obj_evals,
    )


def paired_runs(
    build,
    algorithms: list[Algorithm],
    instance_seeds: list[int],
    run_seeds: list[int],
) -> dict[str, np.ndarray]:
    """Run several algorithms on the same (instance, run seed) pairs.

    `build(instance_seed)` must return a fresh Problem. Rebuilding per instance
    is what guarantees that every algorithm faces the same instance.
    """
    out: dict[str, list[float]] = {a.name: [] for a in algorithms}
    for inst in instance_seeds:
        for seed in run_seeds:
            for alg in algorithms:
                res = run_dynamic(build(inst), alg, instance_seed=inst, run_seed=seed)
                out[alg.name].append(res.migd)
    return {k: np.asarray(v) for k, v in out.items()}
