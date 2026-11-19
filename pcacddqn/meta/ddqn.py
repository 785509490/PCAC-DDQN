"""A small numpy DDQN: MLP, Adam, replay buffer, target network.

The state and action spaces are small (30 state features and 14 actions), so a
numpy implementation is enough and avoids a PyTorch dependency. The class is deliberately independent
of the benchmark; the training script owns episodes, rewards, and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _huber(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Normalizer":
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.std


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.rng = rng

    def add(self, s, a, r, s2, done) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.s2[i] = s2
        self.d[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, size=batch)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx],
                self.d[idx])

    def __len__(self) -> int:
        return self.n


class DDQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        delta: float = 1.0,
        seed: int = 0,
    ):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.rng = np.random.default_rng(seed)
        h = self.hidden
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / self.state_dim),
                                  (self.state_dim, h))
        self.b1 = np.zeros(h)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / h), (h, h))
        self.b2 = np.zeros(h)
        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / h), (h, self.n_actions))
        self.b3 = np.zeros(self.n_actions)

        self.params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0
        self.target = {k: getattr(self, k).copy() for k in self.params}

    # -- forward / backward ------------------------------------------------

    def _forward(self, X: np.ndarray, target: bool = False, cache=False):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        W1, b1, W2, b2, W3, b3 = (
            (self.target[k] for k in self.params) if target
            else (getattr(self, k) for k in self.params)
        )
        z1 = X @ W1 + b1
        a1 = _relu(z1)
        z2 = a1 @ W2 + b2
        a2 = _relu(z2)
        q = a2 @ W3 + b3
        if cache:
            return q, (X, z1, a1, z2, a2)
        return q[0] if single else q

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self._forward(state, target=False)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def _backward(self, dQ, cache) -> dict:
        X, z1, a1, z2, a2 = cache
        g = {}
        g["W3"] = a2.T @ dQ
        g["b3"] = dQ.sum(axis=0)
        da2 = dQ @ self.W3.T
        dz2 = da2 * (z2 > 0.0)
        g["W2"] = a1.T @ dz2
        g["b2"] = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0.0)
        g["W1"] = X.T @ dz1
        g["b1"] = dz1.sum(axis=0)
        return g

    def update(self, batch, beta1=0.9, beta2=0.999, eps=1e-8) -> float:
        s, a, r, s2, done = batch
        B = len(a)
        if B == 0:
            return 0.0
        q, cache = self._forward(s, cache=True)
        q_next_online = self._forward(s2)
        a_star = np.argmax(q_next_online, axis=1)
        q_next_target = self._forward(s2, target=True)
        target = r + self.gamma * (1.0 - done) * q_next_target[
            np.arange(B), a_star
        ]
        q_sa = q[np.arange(B), a]
        td = q_sa - target
        loss = float(np.mean(_huber(td, self.delta)))
        dQ = np.zeros_like(q)
        dQ[np.arange(B), a] = np.clip(td, -self.delta, self.delta) / B
        grads = self._backward(dQ, cache)

        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g * g)
            mhat = self.m[k] / (1 - beta1 ** self.t)
            vhat = self.v[k] / (1 - beta2 ** self.t)
            setattr(self, k, getattr(self, k) - self.lr * mhat
                    / (np.sqrt(vhat) + eps))
        return loss

    def target_update(self, tau: float = 1.0) -> None:
        for k in self.params:
            if tau >= 1.0:
                self.target[k] = getattr(self, k).copy()
            else:
                self.target[k] = ((1.0 - tau) * self.target[k]
                                  + tau * getattr(self, k))

    # -- persistence -------------------------------------------------------

    def save(self, path: str, meta: dict | None = None) -> None:
        arrays = {k: getattr(self, k) for k in self.params}
        arrays.update({f"t_{k}": v for k, v in self.target.items()})
        arrays.update({f"m_{k}": v for k, v in self.m.items()})
        arrays.update({f"v_{k}": v for k, v in self.v.items()})
        info = {
            "state_dim": self.state_dim, "n_actions": self.n_actions,
            "hidden": self.hidden, "lr": self.lr, "gamma": self.gamma,
            "delta": self.delta, "t": self.t, "meta": meta or {},
        }
        arrays["meta"] = np.array(json.dumps(info))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "DDQNAgent":
        data = np.load(path, allow_pickle=False)
        info = json.loads(str(data["meta"]))
        agent = cls(info["state_dim"], info["n_actions"], info["hidden"],
                    info["lr"], info["gamma"], info["delta"])
        for k in agent.params:
            setattr(agent, k, data[k].copy())
            agent.target[k] = data[f"t_{k}"].copy()
            agent.m[k] = data[f"m_{k}"].copy()
            agent.v[k] = data[f"v_{k}"].copy()
        agent.t = int(info["t"])
        agent.meta = info.get("meta", {})
        return agent
