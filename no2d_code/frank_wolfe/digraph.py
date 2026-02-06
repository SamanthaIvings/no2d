from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Digraph:
    u: NDArray[np.int64]
    v: NDArray[np.int64]
    weight: NDArray[np.float64]
    n_nodes: int
    adj: List[List[Tuple[int, int]]]

    capacity: NDArray[np.float64]
    critical_density: NDArray[np.float64]
    distance_km: NDArray[np.float64]
    speedlimit_kmh: NDArray[np.float64]
    free_flow_travel_h: NDArray[np.float64]

    bpr_params: NDArray[np.float64]

    @classmethod
    def from_edges(cls, edges: pd.DataFrame, eps: float) -> "Digraph":
        u = edges["u"].to_numpy(dtype=int)
        v = edges["v"].to_numpy(dtype=int)

        length_m = edges["length"].to_numpy(dtype=float)
        speedlimit_kmh = edges["speedlim"].to_numpy(dtype=float)
        capacity = edges["capacity"].to_numpy(dtype=float)
        critical_density = edges["criticalDensity"].to_numpy(dtype=float)

        n_nodes = int(max(u.max(), v.max()) + 1)

        bpr_params = np.tile([0.15, 4.0, eps], (edges.shape[0], 1))

        distance_km = length_m / 1000.0
        free_flow_travel_h = distance_km / speedlimit_kmh

        adj: List[List[Tuple[int, int]]] = [[] for _ in range(int(n_nodes))]
        for eid in range(u.size):
            adj[int(u[eid])].append((int(v[eid]), int(eid)))

        return cls(
            u=u,
            v=v,
            weight=length_m,
            n_nodes=int(n_nodes),
            adj=adj,
            capacity=capacity,
            critical_density=critical_density,
            distance_km=distance_km,
            speedlimit_kmh=speedlimit_kmh,
            free_flow_travel_h=free_flow_travel_h,
            bpr_params=bpr_params
        )
