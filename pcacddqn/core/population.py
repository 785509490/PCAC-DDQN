"""Population container.

A population is deliberately a plain bundle of arrays rather than a list of
solution objects: every operator in this package works on whole arrays, and
the MATLAB version's per-solution loops were the main reason the oracle sweeps
were expensive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Population:
    dec: np.ndarray                       # (N, D) decision variables
    obj: np.ndarray                       # (N, M) objective values
    con: np.ndarray | None = None         # (N, C) constraint violations, <= 0 feasible
    tag: np.ndarray | None = None         # (N,) origin component, for lineage credit

    def __post_init__(self) -> None:
        self.dec = np.asarray(self.dec, dtype=float)
        self.obj = np.asarray(self.obj, dtype=float)
        if self.con is not None:
            self.con = np.asarray(self.con, dtype=float)
        if self.tag is not None:
            self.tag = np.asarray(self.tag, dtype=int)

    def __len__(self) -> int:
        return self.dec.shape[0]

    def copy(self) -> "Population":
        return Population(
            self.dec.copy(),
            self.obj.copy(),
            None if self.con is None else self.con.copy(),
            None if self.tag is None else self.tag.copy(),
        )

    def take(self, idx: np.ndarray) -> "Population":
        return Population(
            self.dec[idx].copy(),
            self.obj[idx].copy(),
            None if self.con is None else self.con[idx].copy(),
            None if self.tag is None else self.tag[idx].copy(),
        )

    @property
    def feasible(self) -> np.ndarray:
        if self.con is None:
            return np.ones(len(self), dtype=bool)
        return np.all(self.con <= 0, axis=1)
