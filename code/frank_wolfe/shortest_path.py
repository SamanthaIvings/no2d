from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from code.models import NetworkParameters, ODData
from code.frank_wolfe.settings import FrankWolfeSettings, format_seconds, log


def all_or_nothing_assignment(
    network: NetworkParameters,
    od_data: ODData,
    link_travel_time_h: np.ndarray,
    settings: FrankWolfeSettings,
) -> np.ndarray:
    start = time.perf_counter()
    adjacency, pair_to_edge_id = _build_adjacency(
        network.from_nodes, network.to_nodes, link_travel_time_h, network.n_nodes
    )

    aon_flow = np.zeros(network.n_edges, dtype=float)
    unique_origins = np.unique(od_data.origins)
    total_origins = int(unique_origins.size)

    for i, origin in enumerate(unique_origins, start=1):
        _, predecessors = dijkstra(adjacency, directed=True, indices=int(origin), return_predecessors=True)

        trips_from_origin = np.where(od_data.origins == origin)[0]
        for k in trips_from_origin:
            edge_path = _reconstruct_edge_path(
                predecessors=predecessors,
                origin=int(origin),
                destination=int(od_data.destinations[k]),
                pair_to_edge_id=pair_to_edge_id,
            )
            if edge_path:
                aon_flow[edge_path] += od_data.demand[k]

        if settings.aon_verbose and (i == 1 or i % settings.origins_progress_every == 0 or i == total_origins):
            log(
                f"  AoN: origin {i}/{total_origins} ({format_seconds(time.perf_counter() - start)} elapsed)",
                enabled=True,
            )

    if settings.aon_verbose:
        log(f"  AoN: finished in {format_seconds(time.perf_counter() - start)}", enabled=True)

    return aon_flow


def _build_adjacency(
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> Tuple[csr_matrix, Dict[Tuple[int, int], int]]:
    u, v, w, pair_to_edge = _min_edge_per_pair(from_nodes, to_nodes, weights, n_nodes)
    return csr_matrix((w, (u, v)), shape=(n_nodes, n_nodes)), pair_to_edge


def _min_edge_per_pair(
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    edge_id = np.arange(len(weights), dtype=int)
    key = from_nodes.astype(np.int64) * np.int64(n_nodes) + to_nodes.astype(np.int64)

    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    u_s = from_nodes[order]
    v_s = to_nodes[order]
    w_s = weights[order]
    id_s = edge_id[order]

    u_out: List[int] = []
    v_out: List[int] = []
    w_out: List[float] = []
    id_out: List[int] = []

    i = 0
    m = len(key_s)
    while i < m:
        j = i + 1
        best_j = i
        best_w = w_s[i]
        while j < m and key_s[j] == key_s[i]:
            if w_s[j] < best_w:
                best_w = w_s[j]
                best_j = j
            j += 1
        u_out.append(int(u_s[best_j]))
        v_out.append(int(v_s[best_j]))
        w_out.append(float(best_w))
        id_out.append(int(id_s[best_j]))
        i = j

    pair_to_edge = {(u, v): eid for u, v, eid in zip(u_out, v_out, id_out)}
    return np.asarray(u_out, int), np.asarray(v_out, int), np.asarray(w_out, float), pair_to_edge


def _reconstruct_edge_path(
    predecessors: np.ndarray,
    origin: int,
    destination: int,
    pair_to_edge_id: Dict[Tuple[int, int], int],
) -> List[int]:
    if destination == origin or predecessors[destination] < 0:
        return []

    nodes_rev = [destination]
    cur = destination
    while cur != origin:
        cur = int(predecessors[cur])
        if cur < 0:
            return []
        nodes_rev.append(cur)

    nodes = nodes_rev[::-1]
    return [pair_to_edge_id[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]
