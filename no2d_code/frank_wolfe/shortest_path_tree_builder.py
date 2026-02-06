from __future__ import annotations

import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np

from no2d_code.frank_wolfe.digraph import Digraph


def get_unique_origins(origin_destination: np.ndarray) -> np.ndarray:
    return np.unique(origin_destination[:, 2].astype(int))


def build_shortest_path_trees(
    graph: Digraph,
    origins: np.ndarray,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Dict[int, List]:
    t0 = perf_counter()
    orig = np.asarray(origins, dtype=int)
    n = int(orig.size)

    if n == 0:
        print("[SPT] Building shortest-path trees for 0 unique origins...")
        print("[SPT] Done in 0.00s  (0.0000s/origin)")
        return {}

    if not parallel:
        e_store: Dict[int, List] = {}
        print(f"[SPT] Building shortest-path trees for {n} unique origins...")
        for j, s in enumerate(orig):
            if j % 100 == 0 or j == n - 1:
                print(f"[SPT]  {j+1}/{n} origins")
            si = int(s)
            e_store[si] = get_shortest_path_tree_edges_cell(graph, si)
        dt = perf_counter() - t0
        print(f"[SPT] Done in {dt:.2f}s  ({dt/max(n,1):.4f}s/origin)")
        return e_store

    print(f"[SPT] Building shortest-path trees for {n} unique origins (threaded)...")
    e_store: Dict[int, List] = {}
    done = 0

    def work(si: int) -> Tuple[int, List]:
        return si, get_shortest_path_tree_edges_cell(graph, si)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, int(s)) for s in orig]
        for fut in as_completed(futs):
            si, spt = fut.result()
            e_store[int(si)] = spt
            done += 1
            if done % 100 == 0 or done == n:
                print(f"[SPT]  {done}/{n} origins")

    dt = perf_counter() - t0
    print(f"[SPT] Done in {dt:.2f}s  ({dt/max(n,1):.4f}s/origin)")
    return e_store


def get_shortest_path_tree_edges_cell(graph: Digraph, origin: int) -> List[List[int]]:
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
