from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
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
    def from_edge_arrays(
        cls,
        *,
        u: NDArray[np.int64],
        v: NDArray[np.int64],
        length_m: NDArray[np.float64],
        speedlimit_kmh: NDArray[np.float64],
        capacity: NDArray[np.float64],
        critical_density: NDArray[np.float64],
        n_nodes: int,
        weight: NDArray[np.float64],
        bpr_params: NDArray[np.float64] | None = None,
    ) -> "Digraph":
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        length_m = np.asarray(length_m, dtype=float)
        speedlimit_kmh = np.asarray(speedlimit_kmh, dtype=float)
        capacity = np.asarray(capacity, dtype=float)
        critical_density = np.asarray(critical_density, dtype=float)
        weight = np.asarray(weight, dtype=float)

        distance_km = length_m / 1000.0
        free_flow_travel_h = distance_km / speedlimit_kmh

        adj: List[List[Tuple[int, int]]] = [[] for _ in range(int(n_nodes))]
        for eid in range(u.size):
            adj[int(u[eid])].append((int(v[eid]), int(eid)))

        return cls(
            u=u,
            v=v,
            weight=weight,
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
