from __future__ import annotations

import heapq
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


def shortestpathtree_edges_cell(graph: Digraph, origin: int) -> List[List[int]]:
    n = graph.n_nodes
    dist = np.full(n, np.inf, dtype=float)
    pred_node = np.full(n, -1, dtype=np.int64)
    pred_edge = np.full(n, -1, dtype=np.int64)

    origin = int(origin)
    dist[origin] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, origin)]

    tol = 1e-12

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node] + tol:
            continue

        for nbr, eid in graph.adj[node]:
            nd = d + float(graph.weight[eid])
            if nd + tol < dist[nbr]:
                dist[nbr] = nd
                pred_node[nbr] = node
                pred_edge[nbr] = eid
                heapq.heappush(heap, (nd, nbr))

    E: List[List[int]] = [[] for _ in range(n)]
    for t in range(n):
        if t == origin or pred_edge[t] == -1:
            continue

        path_edges: List[int] = []
        cur = t
        while cur != origin and pred_edge[cur] != -1:
            path_edges.append(int(pred_edge[cur]))
            cur = int(pred_node[cur])

        if cur == origin:
            path_edges.reverse()
            E[t] = path_edges

    return E
