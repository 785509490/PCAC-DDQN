"""Train the PCAC-DDQN policy on the 23-problem benchmark.

Episodes are generated on the fly by ``run_meta`` on the 13 training
problems. The reward is the scaled per-environment IGD. Training is offline
and the policy is frozen at deployment.

The default configuration follows the paper. For a quick installation check,
run ``python examples/train.py --smoke``, which trains for one epoch on one
problem, one instance, one seed, and one change interval.
"""

from __future__ import annotations

import argparse
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pcacddqn.meta.action_space import ACTION_TABLE, load_action_table
from pcacddqn.meta.ddqn import DDQNAgent, Normalizer, ReplayBuffer
from pcacddqn.meta.meta_policy import GreedyTablePolicy, QPolicy
from pcacddqn.meta.metastate import STATE_DIM
from pcacddqn.problems.suite import build
from pcacddqn.runner.experiment import run_meta

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TABLE = REPO_ROOT / "configs" / "action_table.json"

TRAIN13 = ["A1", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4",
           "E1", "E3"]
VAL4 = ["C5", "D6", "E2", "E4"]
TRAIN_INST = (107, 207)
TRAIN_SEEDS = (1, 2)
VAL_INST = (107, 207)
VAL_SEED = (3,)
TRAIN_TAUTS = (20, 30, 50)
T0 = 50
TOTAL_GENERATIONS = 450


def _n_env(taut: int) -> int:
    return 1 + (TOTAL_GENERATIONS - T0) // int(taut)


def _default_for_taut(taut: int, table, fallback: int) -> int:
    """The best fixed action changes at taut=50, where elite is preferred."""
    if taut >= 50:
        for i, a in enumerate(table):
            if a.program.kind == "elite":
                return i
    return int(fallback)


def _default_idx(table) -> int:
    """Index of the memory action with m=60, rho=0.5, and the default DE."""
    for i, a in enumerate(table):
        p = a.program
        if (p.kind == "memory" and p.m == 60 and abs(p.rho - 0.5) < 1e-9
                and p.max_fronts == 20 and not p.recent_only
                and a.de == ("rand/1", 0.5, 0.5)):
            return i
    for i, a in enumerate(table):
        if a.name.startswith("mem-m60-r0.50-f20"):
            return i
    return 3


def _map(fn, tasks, workers: int):
    """Run tasks serially for one worker, otherwise in a process pool."""
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def _shadow(task):
    pid, inst, seed, taut, n_env, table, default_idx = task
    res = run_meta(
        build(pid, inst, n_env=n_env, taut=taut, T0=T0,
              intensity_seed="instance"),
        GreedyTablePolicy(default_idx),
        instance_seed=inst, run_seed=seed, action_table=table,
        reward_scale=1.0,
    )
    return (pid, int(inst), int(seed), int(taut),
            float(np.mean(res.per_env_igd)), res.states.copy())


def _episode(task):
    pid, inst, seed, taut, n_env, agent, normalizer, table, epsilon, c_p = task
    agent.rng = np.random.default_rng((seed, 0xDD))
    policy = QPolicy(agent, normalizer, epsilon=epsilon)
    res = run_meta(
        build(pid, inst, n_env=n_env, taut=taut, T0=T0,
              intensity_seed="instance"), policy,
        instance_seed=inst, run_seed=seed, action_table=table,
        reward_scale=c_p,
    )
    return (normalizer.transform(res.states), res.actions, res.rewards,
            normalizer.transform(res.next_states), res.dones,
            float(res.migd), int(res.response_evals))


def _val_task(task):
    pid, inst, seed, taut, agent, normalizer, table, c_p = task
    res = run_meta(
        build(pid, inst, n_env=_n_env(taut), taut=taut, T0=T0,
              intensity_seed="instance"),
        QPolicy(agent, normalizer, epsilon=0.0),
        instance_seed=inst, run_seed=seed, action_table=table,
        reward_scale=c_p,
    )
    return float(res.migd)


def _validate(agent, normalizer, table, cps, tauts, workers=1):
    tasks = []
    for taut in tauts:
        for pid in VAL4:
            for inst in VAL_INST:
                for seed in VAL_SEED:
                    tasks.append((
                        pid, inst, seed, taut, agent, normalizer, table,
                        cps.get((pid, inst, seed, taut), 1.0),
                    ))
    vals = _map(_val_task, tasks, workers)
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--action-table", default=str(DEFAULT_TABLE))
    ap.add_argument("--out", default="artifacts/ddqn_meta.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--default-action", type=int, default=None)
    ap.add_argument("--tauts", default="20,30,50")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many validation checks without gain")
    ap.add_argument("--smoke", action="store_true",
                    help="one problem, one instance, one seed, one taut, "
                         "one epoch, no validation")
    args = ap.parse_args()

    if args.smoke:
        epochs = 1
        workers = 1
        tauts = (20,)
        train_pids = ("A1",)
        train_inst = (107,)
        train_seeds = (1,)
        validate = False
        out = "artifacts/ddqn_smoke.npz"
    else:
        epochs = args.epochs
        workers = args.workers
        tauts = tuple(int(x) for x in args.tauts.split(","))
        train_pids = tuple(TRAIN13)
        train_inst = TRAIN_INST
        train_seeds = TRAIN_SEEDS
        validate = True
        out = args.out

    path = pathlib.Path(args.action_table)
    table = load_action_table(path) if path.exists() else ACTION_TABLE
    n_actions = len(table)
    default_idx = (_default_idx(table) if args.default_action is None
                   else int(args.default_action))
    print(f"actions: {n_actions}; default index {default_idx}")

    train_triples = [(pid, inst, seed, taut, _n_env(taut))
                     for pid in train_pids for inst in train_inst
                     for seed in train_seeds for taut in tauts]
    print(f"train {len(train_triples)} episodes/epoch")

    # Shadow rollouts use the default policy to set the reward scale and the
    # state normaliser. They are not training episodes.
    shadow_tasks = [(pid, inst, seed, taut, n_env, table,
                     _default_for_taut(taut, table, default_idx))
                    for pid, inst, seed, taut, n_env in train_triples]
    cps = {}
    shadow_states = []
    print("shadow rollouts...")
    for pid, inst, seed, taut, c_p, st in _map(_shadow, shadow_tasks, workers):
        cps[(pid, inst, seed, taut)] = float(c_p)
        shadow_states.append(st)
    normalizer = Normalizer.fit(np.vstack(shadow_states))
    print("state normaliser fitted on", sum(len(x) for x in shadow_states),
          "states")

    agent = DDQNAgent(STATE_DIM, n_actions, hidden=args.hidden, seed=0)
    replay = ReplayBuffer(50000, STATE_DIM, np.random.default_rng(0))
    best = float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(1, epochs + 1):
        frac = min(1.0, epoch / max(1.0, 0.6 * epochs))
        epsilon = 0.30 * (1.0 - frac) + 0.05
        tasks = [(pid, inst, seed, taut, n_env, agent, normalizer, table,
                  epsilon, cps[(pid, inst, seed, taut)])
                 for pid, inst, seed, taut, n_env in train_triples]
        ep_migds = []
        n_added = 0
        for s, a, r, s2, d, migd, _ev in _map(_episode, tasks, workers):
            ep_migds.append(migd)
            for i in range(len(a)):
                replay.add(s[i], a[i], r[i], s2[i], d[i])
                n_added += 1
            if len(replay) >= args.batch:
                for _ in range(len(a)):
                    agent.update(replay.sample(args.batch))
                    if agent.t % 500 == 0:
                        agent.target_update(1.0)
        print(f"epoch {epoch:02d} eps={epsilon:.3f} "
              f"train_migd={np.mean(ep_migds):.4f} "
              f"transitions={n_added} replay={len(replay)}", flush=True)

        if validate and (epoch == 1 or epoch % 2 == 0):
            vm = _validate(agent, normalizer, table, cps, tauts,
                           workers=workers)
            print(f"  validation {vm:.4f}", flush=True)
            if vm < best:
                best = vm
                best_epoch = epoch
                stale = 0
                _save(agent, out, normalizer, table, vm, epoch, train_pids,
                      tauts)
                print(f"  saved {out} (best {vm:.4f})", flush=True)
            else:
                stale += 1
                if args.patience > 0 and stale >= args.patience:
                    print(f"early stop: no gain for {stale} checks",
                          flush=True)
                    break

    if validate:
        print(f"best validation {best:.4f} at epoch {best_epoch}")
    else:
        _save(agent, out, normalizer, table, None, epochs, train_pids, tauts)
        print(f"saved {out}")


def _save(agent, out: str, normalizer, table, val_migd, epoch, train_pids,
          tauts) -> None:
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path), meta={
        "mu": normalizer.mean.tolist(),
        "sd": normalizer.std.tolist(),
        "action_names": [a.name for a in table],
        "val_migd": val_migd,
        "epoch": epoch,
        "train_pids": list(train_pids),
        "tauts": list(tauts),
    })


if __name__ == "__main__":
    main()
