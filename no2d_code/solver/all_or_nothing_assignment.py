from __future__ import annotations

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


def compute_all_or_nothing_flow(
    demand: np.ndarray,
    origin_destination: np.ndarray,
    shortest_path_trees: Dict[int, List],
    n_edges: int,
) -> NDArray[np.float64]:
    flow_y = np.zeros(n_edges, dtype=float)

    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])

        store = shortest_path_trees.get(s)
        if store is None:
            continue

        edgepath = store[t]
        if not edgepath:
            continue

        d = demand[i]
        for e in edgepath:
            flow_y[int(e)] += d

    return flow_y


def build_edge_od_incidence_matrix(
    origin_destination: np.ndarray,
    shortest_path_trees: Dict[int, List],
    n_edges: int,
) -> NDArray[np.float64]:
    xa_gi = np.zeros((n_edges, origin_destination.shape[0]), dtype=float)

    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])

        store = shortest_path_trees.get(s)
        if store is None:
            continue

        edgepath = store[t]
        if not edgepath:
            continue

        xa_gi[np.fromiter((int(e) for e in edgepath), dtype=int), i] = 1.0

    return xa_gi
