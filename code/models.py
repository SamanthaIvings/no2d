from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NetworkParameters:
    from_nodes: np.ndarray
    to_nodes: np.ndarray
    free_flow_time_h: np.ndarray
    capacity: np.ndarray
    critical_density: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    bpr_eps: float
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return int(self.from_nodes.shape[0])


@dataclass(frozen=True)
class ODData:
    origins: np.ndarray
    destinations: np.ndarray
    demand: np.ndarray
