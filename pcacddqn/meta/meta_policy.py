"""Small policy wrappers for the meta runner and the gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MetaPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, state: np.ndarray) -> int: ...


@dataclass
class GreedyTablePolicy:
    """Always choose one action index. Used for screening and fixed arms."""

    action: int

    def reset(self) -> None:
        return None

    def act(self, state: np.ndarray) -> int:
        return int(self.action)


@dataclass
class RandomMetaPolicy:
    n_actions: int
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, state: np.ndarray) -> int:
        return int(self._rng.integers(self.n_actions))


@dataclass
class QPolicy:
    """Greedy (or epsilon-greedy) policy over a Q-function.

    `agent` only needs a `q_values(state)` method and, for exploration, an
    `act(state, epsilon)` method; this keeps the policy independent of the
    DDQN implementation and lets the gate use a mock.
    """

    agent: object
    normalizer: object | None = None
    epsilon: float = 0.0

    def reset(self) -> None:
        return None

    def act(self, state: np.ndarray) -> int:
        x = np.asarray(state, dtype=float)
        if self.normalizer is not None:
            x = self.normalizer.transform(x[None, :])[0]
        if hasattr(self.agent, "act"):
            return int(self.agent.act(x, epsilon=self.epsilon))
        q = np.asarray(self.agent.q_values(x), dtype=float)
        return int(np.argmax(q))
