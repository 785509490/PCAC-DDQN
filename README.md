# PCAC-DDQN

Reference implementation of per-change automatic configuration for dynamic multiobjective optimization. At every environmental change, the algorithm selects a dynamic response strategy and a differential evolution configuration from the state observed at that instant. The policy is a Double Deep Q-Network trained offline and frozen at deployment.

This repository contains the method and the benchmark only. Experiment scripts, baseline implementations, result tables, and training data are not included.

## Installation

Python 3.10 or newer, with NumPy as the only runtime dependency.

```bash
pip install -e .
```

For the smoke test, install the test extra.

```bash
pip install -e ".[test]"
```

## Quick start

Run one epoch on one problem, one instance, one seed, and one change interval. This checks the installation and writes `artifacts/ddqn_smoke.npz`.

```bash
python examples/train.py --smoke
```

Train with the configuration of the paper. The default run uses 13 training problems, instances 107 and 207, seeds 1 and 2, change intervals 20, 30, and 50, and 30 epochs. It runs for a long time on a CPU. Use `--workers` to set the number of processes and `--epochs` to cap the run.

```bash
python examples/train.py --workers 8 --out artifacts/ddqn_meta.npz
```

The action table in `configs/action_table.json` contains the 14 actions used in the paper. If the file is missing, the code falls back to a smaller built-in table.

## Benchmark

The benchmark builds 23 problems from five layers, the base landscape, the POS motion, the change law, the Pareto front geometry, and the Pareto front change operator. Problems are grouped into five classes by what an algorithm can know at a change instant.

```python
import numpy as np
from pcacddqn.problems.suite import build
from pcacddqn.algorithms.nsga2 import NSGA2
from pcacddqn.core.metrics import igd

problem = build("C1", instance_seed=107, n_env=21, taut=20, T0=50)
t = float(problem.environment_times[0])
algorithm = NSGA2(n_pop=100)
algorithm.bind(np.random.default_rng(0))
population = algorithm.initialize(problem, t)
for _ in range(20):
    population = algorithm.step(problem, population, t)
print(igd(population.obj, problem.true_pf(t)))
```

`build` takes the problem ID, the instance seed, the decision dimensionality, the number of environments, the change interval, and the number of initial generations. Reference fronts are analytic and generated at run time, so no data files are needed.

The benchmark does not encode a split. The training script uses the paper split, with 13 training problems, 4 validation problems, and 6 test problems.

## Repository layout

```
pcacddqn/core/        problem, population, metrics, operators, variation
pcacddqn/problems/    23-problem generator
pcacddqn/algorithms/  NSGA-II inner solver
pcacddqn/meta/        state, actions, memory operators, DDQN, policy
pcacddqn/runner/      per-change loop
configs/              14-action table
examples/train.py     training entry point
tests/test_smoke.py   installation and benchmark smoke test
```

## Tests

```bash
python -m pytest tests/test_smoke.py
```

## Citation

The accompanying paper is under review. Please cite it once the bibliographic details are available.

## License

MIT. See `LICENSE`.
