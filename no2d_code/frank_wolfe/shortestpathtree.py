from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Dict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


@dataclass
class Digraph:
    u: NDArray[np.int64]
    v: NDArray[np.int64]
    weight: NDArray[np.float64]
    n_nodes: int

    capacity: NDArray[np.float64]
    critical_density: NDArray[np.float64]
    distance_km: NDArray[np.float64]
    speedlimit_kmh: NDArray[np.float64]
    free_flow_travel_h: NDArray[np.float64]

    bpr_params: NDArray[np.float64]

    adj_indptr: NDArray[np.int64]
    adj_to: NDArray[np.int64]
    adj_eid: NDArray[np.int64]

    @classmethod
    def from_edges(cls, edges: pd.DataFrame) -> "Digraph":
        u = edges["u"].to_numpy(dtype=np.int64)
        v = edges["v"].to_numpy(dtype=np.int64)

        length_m = edges["length"].to_numpy(dtype=np.float64)
        speedlimit_kmh = edges["speedlim"].to_numpy(dtype=np.float64)
        capacity = edges["capacity"].to_numpy(dtype=np.float64)
        critical_density = edges["criticalDensity"].to_numpy(dtype=np.float64)

        n_nodes = int(max(int(u.max()), int(v.max())) + 1)
        m = int(u.size)

        bpr_params = np.tile([0.15, 4.0, 0.0], (m, 1)).astype(np.float64)

        distance_km = length_m / 1000.0
        free_flow_travel_h = distance_km / speedlimit_kmh

        adj_indptr, adj_to, adj_eid = build_csr_adjacency(u, v, n_nodes)

        return cls(
            u=u,
            v=v,
            weight=length_m,
            n_nodes=n_nodes,
            capacity=capacity,
            critical_density=critical_density,
            distance_km=distance_km,
            speedlimit_kmh=speedlimit_kmh,
            free_flow_travel_h=free_flow_travel_h,
            bpr_params=bpr_params,
            adj_indptr=adj_indptr,
            adj_to=adj_to,
            adj_eid=adj_eid,
        )


def build_csr_adjacency(
    u: NDArray[np.int64],
    v: NDArray[np.int64],
    n_nodes: int,
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    m = int(u.size)
    deg = np.zeros(n_nodes, dtype=np.int64)
    for i in range(m):
        deg[int(u[i])] += 1

    indptr = np.empty(n_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(deg, out=indptr[1:])

    to = np.empty(m, dtype=np.int64)
    eid = np.empty(m, dtype=np.int64)

    cur = indptr[:-1].copy()
    for e in range(m):
        a = int(u[e])
        k = int(cur[a])
        to[k] = int(v[e])
        eid[k] = int(e)
        cur[a] = k + 1

    return indptr, to, eid


@dataclass
class SPTWorkspace:
    dist: NDArray[np.float64]
    pred_node: NDArray[np.int64]
    pred_edge: NDArray[np.int64]

    heap_node: NDArray[np.int64]
    heap_dist: NDArray[np.float64]
    heap_pos: NDArray[np.int64]

    node_demand: NDArray[np.float64]
    edge_flow: NDArray[np.float64]

    @classmethod
    def create(cls, n_nodes: int, n_edges: int) -> "SPTWorkspace":
        return cls(
            dist=np.empty(n_nodes, dtype=np.float64),
            pred_node=np.empty(n_nodes, dtype=np.int64),
            pred_edge=np.empty(n_nodes, dtype=np.int64),
            heap_node=np.empty(n_nodes, dtype=np.int64),
            heap_dist=np.empty(n_nodes, dtype=np.float64),
            heap_pos=np.empty(n_nodes, dtype=np.int64),
            node_demand=np.empty(n_nodes, dtype=np.float64),
            edge_flow=np.empty(n_edges, dtype=np.float64),
        )


def spt_predecessors(
    graph: Digraph,
    origin: int,
    edge_cost: Optional[NDArray[np.float64]] = None,
    ws: Optional[SPTWorkspace] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    costs = graph.weight if edge_cost is None else edge_cost
    if ws is None:
        ws = SPTWorkspace.create(graph.n_nodes, costs.size)

    if njit is None:
        return _spt_predecessors_py(graph, int(origin), costs, ws)

    _dijkstra_inplace_numba(
        graph.n_nodes,
        graph.adj_indptr,
        graph.adj_to,
        graph.adj_eid,
        costs,
        int(origin),
        1e-12,
        ws.dist,
        ws.pred_node,
        ws.pred_edge,
        ws.heap_node,
        ws.heap_dist,
        ws.heap_pos,
    )
    return ws.dist, ws.pred_node, ws.pred_edge


def aon_flow_from_tree(
    origin: int,
    dist: NDArray[np.float64],
    pred_node: NDArray[np.int64],
    pred_edge: NDArray[np.int64],
    dest_demands: NDArray[np.float64],
    out_edge_flow: NDArray[np.float64],
) -> None:
    out_edge_flow.fill(0.0)

    order = np.argsort(dist)
    for i in range(order.size - 1, -1, -1):
        t = int(order[i])
        e = int(pred_edge[t])
        if t == origin or e == -1:
            continue
        d = float(dest_demands[t])
        if d != 0.0:
            out_edge_flow[e] += d
            p = int(pred_node[t])
            if p != -1:
                dest_demands[p] += d


def all_or_nothing_assignment_fast(
    graph: Digraph,
    origins: NDArray[np.int64],
    destinations: NDArray[np.int64],
    demands: NDArray[np.float64],
    edge_cost: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    costs = graph.weight if edge_cost is None else edge_cost
    ws = SPTWorkspace.create(graph.n_nodes, costs.size)

    idx = np.argsort(origins, kind="mergesort")
    o_sorted = origins[idx]
    d_sorted = destinations[idx]
    q_sorted = demands[idx]

    uniq, start = np.unique(o_sorted, return_index=True)
    start = np.append(start, o_sorted.size)

    total_flow = np.zeros(costs.size, dtype=np.float64)

    for k in range(uniq.size):
        o = int(uniq[k])
        a = int(start[k])
        b = int(start[k + 1])

        ws.node_demand.fill(0.0)
        np.add.at(ws.node_demand, d_sorted[a:b].astype(np.int64, copy=False), q_sorted[a:b])

        dist, pred_node, pred_edge = spt_predecessors(graph, o, costs, ws)

        aon_flow_from_tree(
            origin=o,
            dist=dist,
            pred_node=pred_node,
            pred_edge=pred_edge,
            dest_demands=ws.node_demand,
            out_edge_flow=ws.edge_flow,
        )
        total_flow += ws.edge_flow

    return total_flow


def paths_for_targets(
    origin: int,
    targets: Iterable[int],
    pred_node: NDArray[np.int64],
    pred_edge: NDArray[np.int64],
) -> Dict[int, list[int]]:
    memo: Dict[int, list[int]] = {int(origin): []}
    out: Dict[int, list[int]] = {}

    for t_raw in targets:
        t = int(t_raw)
        if t in memo:
            out[t] = memo[t]
            continue

        chain: list[int] = []
        cur = t
        while cur != -1 and cur not in memo:
            chain.append(cur)
            cur = int(pred_node[cur])

        base = memo.get(cur, [])
        for node in reversed(chain):
            pe = int(pred_edge[node])
            base = [] if pe == -1 else (base + [pe])
            memo[node] = base

        out[t] = memo[t]

    return out


def _spt_predecessors_py(
    graph: Digraph,
    origin: int,
    costs: NDArray[np.float64],
    ws: SPTWorkspace,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    dist = ws.dist
    pred_node = ws.pred_node
    pred_edge = ws.pred_edge

    dist.fill(np.inf)
    pred_node.fill(-1)
    pred_edge.fill(-1)

    dist[origin] = 0.0
    heap: list[tuple[float, int]] = [(0.0, origin)]
    tol = 1e-12

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node] + tol:
            continue
        s = graph.adj_indptr[node]
        e = graph.adj_indptr[node + 1]
        for k in range(int(s), int(e)):
            nbr = int(graph.adj_to[k])
            eid = int(graph.adj_eid[k])
            nd = d + float(costs[eid])
            if nd + tol < dist[nbr]:
                dist[nbr] = nd
                pred_node[nbr] = node
                pred_edge[nbr] = eid
                heapq.heappush(heap, (nd, nbr))

    return dist, pred_node, pred_edge


if njit is not None:
    @njit(cache=True)
    def _heap_swap(hn, hd, pos, i, j):
        ni = hn[i]
        nj = hn[j]
        di = hd[i]
        dj = hd[j]
        hn[i] = nj
        hn[j] = ni
        hd[i] = dj
        hd[j] = di
        pos[nj] = i
        pos[ni] = j


    @njit(cache=True)
    def _heap_sift_up(hn, hd, pos, i):
        while i > 0:
            p = (i - 1) // 2
            if hd[p] <= hd[i]:
                break
            _heap_swap(hn, hd, pos, i, p)
            i = p


    @njit(cache=True)
    def _heap_sift_down(hn, hd, pos, size, i):
        while True:
            l = 2 * i + 1
            r = l + 1
            if l >= size:
                break
            m = l
            if r < size and hd[r] < hd[l]:
                m = r
            if hd[i] <= hd[m]:
                break
            _heap_swap(hn, hd, pos, i, m)
            i = m


    @njit(cache=True)
    def _heap_push(hn, hd, pos, size, node, dist):
        hn[size] = node
        hd[size] = dist
        pos[node] = size
        _heap_sift_up(hn, hd, pos, size)
        return size + 1


    @njit(cache=True)
    def _heap_pop(hn, hd, pos, size):
        node = hn[0]
        dist = hd[0]
        last = size - 1

        hn[0] = hn[last]
        hd[0] = hd[last]
        pos[hn[0]] = 0
        pos[node] = -1

        size -= 1
        if size > 0:
            _heap_sift_down(hn, hd, pos, size, 0)

        return node, dist, size


    @njit(cache=True)
    def _heap_decrease_key(hn, hd, pos, node, new_dist):
        i = pos[node]
        if i == -1:
            return
        if new_dist >= hd[i]:
            return
        hd[i] = new_dist
        _heap_sift_up(hn, hd, pos, i)


    @njit(cache=True)
    def _dijkstra_inplace_numba(
        n_nodes,
        indptr,
        to,
        eid,
        weight,
        origin,
        tol,
        dist,
        pred_node,
        pred_edge,
        heap_node,
        heap_dist,
        pos,
    ):
        for i in range(n_nodes):
            dist[i] = np.inf
            pred_node[i] = -1
            pred_edge[i] = -1
            pos[i] = -1

        size = 0
        dist[origin] = 0.0
        size = _heap_push(heap_node, heap_dist, pos, size, origin, 0.0)

        while size > 0:
            node, d, size = _heap_pop(heap_node, heap_dist, pos, size)
            if d > dist[node] + tol:
                continue

            start = indptr[node]
            end = indptr[node + 1]
            for k in range(start, end):
                nbr = to[k]
                e = eid[k]
                nd = d + weight[e]
                if nd + tol < dist[nbr]:
                    dist[nbr] = nd
                    pred_node[nbr] = node
                    pred_edge[nbr] = e
                    if pos[nbr] == -1:
                        size = _heap_push(heap_node, heap_dist, pos, size, nbr, nd)
                    else:
                        _heap_decrease_key(heap_node, heap_dist, pos, nbr, nd)


@dataclass
class SPTWorkspace:
    dist: NDArray[np.float64]
    pred_node: NDArray[np.int64]
    pred_edge: NDArray[np.int64]

    heap_node: NDArray[np.int64]
    heap_dist: NDArray[np.float64]
    heap_pos: NDArray[np.int64]

    node_demand: NDArray[np.float64]
    edge_flow: NDArray[np.float64]

    @classmethod
    def create(cls, n_nodes: int, n_edges: int) -> "SPTWorkspace":
        return cls(
            dist=np.empty(n_nodes, dtype=np.float64),
            pred_node=np.empty(n_nodes, dtype=np.int64),
            pred_edge=np.empty(n_nodes, dtype=np.int64),
            heap_node=np.empty(n_nodes, dtype=np.int64),
            heap_dist=np.empty(n_nodes, dtype=np.float64),
            heap_pos=np.empty(n_nodes, dtype=np.int64),
            node_demand=np.empty(n_nodes, dtype=np.float64),
            edge_flow=np.empty(n_edges, dtype=np.float64),
        )


def spt_predecessors(
    graph: "Digraph",
    origin: int,
    edge_cost: Optional[NDArray[np.float64]] = None,
    ws: Optional[SPTWorkspace] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    costs = graph.weight if edge_cost is None else edge_cost
    if ws is None:
        ws = SPTWorkspace.create(graph.n_nodes, costs.size)

    if njit is None:
        return _spt_predecessors_py(graph, int(origin), costs, ws)

    _dijkstra_inplace_numba(
        graph.n_nodes,
        graph.adj_indptr,
        graph.adj_to,
        graph.adj_eid,
        costs,
        int(origin),
        1e-12,
        ws.dist,
        ws.pred_node,
        ws.pred_edge,
        ws.heap_node,
        ws.heap_dist,
        ws.heap_pos,
    )
    return ws.dist, ws.pred_node, ws.pred_edge


def aon_flow_from_tree(
    origin: int,
    dist: NDArray[np.float64],
    pred_node: NDArray[np.int64],
    pred_edge: NDArray[np.int64],
    node_demand: NDArray[np.float64],
    out_edge_flow: NDArray[np.float64],
) -> None:
    out_edge_flow.fill(0.0)
    order = np.argsort(dist)

    for i in range(order.size - 1, -1, -1):
        t = int(order[i])
        if t == origin:
            continue

        e = int(pred_edge[t])
        if e == -1:
            continue

        d = float(node_demand[t])
        if d == 0.0:
            continue

        out_edge_flow[e] += d
        p = int(pred_node[t])
        if p != -1:
            node_demand[p] += d


def write_xagi_from_tree(
    origin: int,
    pred_node: NDArray[np.int64],
    pred_edge: NDArray[np.int64],
    dests: NDArray[np.int64],
    od_cols: NDArray[np.int64],
    Xa_Gi: NDArray[np.float64],
) -> None:
    for j in range(dests.size):
        col = int(od_cols[j])
        cur = int(dests[j])

        while cur != origin:
            e = int(pred_edge[cur])
            if e == -1:
                break
            Xa_Gi[e, col] = 1.0
            cur = int(pred_node[cur])
            if cur == -1:
                break


def _spt_predecessors_py(
    graph: "Digraph",
    origin: int,
    costs: NDArray[np.float64],
    ws: SPTWorkspace,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    import heapq

    dist = ws.dist
    pred_node = ws.pred_node
    pred_edge = ws.pred_edge

    dist.fill(np.inf)
    pred_node.fill(-1)
    pred_edge.fill(-1)

    dist[origin] = 0.0
    heap = [(0.0, origin)]
    tol = 1e-12

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node] + tol:
            continue

        s = int(graph.adj_indptr[node])
        e = int(graph.adj_indptr[node + 1])
        for k in range(s, e):
            nbr = int(graph.adj_to[k])
            eid = int(graph.adj_eid[k])
            nd = d + float(costs[eid])
            if nd + tol < dist[nbr]:
                dist[nbr] = nd
                pred_node[nbr] = node
                pred_edge[nbr] = eid
                heapq.heappush(heap, (nd, nbr))

    return dist, pred_node, pred_edge


if njit is not None:
    @njit(cache=True)
    def _heap_swap(hn, hd, pos, i, j):
        ni = hn[i]
        nj = hn[j]
        di = hd[i]
        dj = hd[j]
        hn[i] = nj
        hn[j] = ni
        hd[i] = dj
        hd[j] = di
        pos[nj] = i
        pos[ni] = j


    @njit(cache=True)
    def _heap_sift_up(hn, hd, pos, i):
        while i > 0:
            p = (i - 1) // 2
            if hd[p] <= hd[i]:
                break
            _heap_swap(hn, hd, pos, i, p)
            i = p


    @njit(cache=True)
    def _heap_sift_down(hn, hd, pos, size, i):
        while True:
            l = 2 * i + 1
            r = l + 1
            if l >= size:
                break
            m = l
            if r < size and hd[r] < hd[l]:
                m = r
            if hd[i] <= hd[m]:
                break
            _heap_swap(hn, hd, pos, i, m)
            i = m


    @njit(cache=True)
    def _heap_push(hn, hd, pos, size, node, dist):
        hn[size] = node
        hd[size] = dist
        pos[node] = size
        _heap_sift_up(hn, hd, pos, size)
        return size + 1


    @njit(cache=True)
    def _heap_pop(hn, hd, pos, size):
        node = hn[0]
        dist = hd[0]
        last = size - 1
        hn[0] = hn[last]
        hd[0] = hd[last]
        pos[hn[0]] = 0
        pos[node] = -1
        size -= 1
        if size > 0:
            _heap_sift_down(hn, hd, pos, size, 0)
        return node, dist, size


    @njit(cache=True)
    def _heap_decrease_key(hn, hd, pos, node, new_dist):
        i = pos[node]
        if i == -1:
            return
        if new_dist >= hd[i]:
            return
        hd[i] = new_dist
        _heap_sift_up(hn, hd, pos, i)


    @njit(cache=True)
    def _dijkstra_inplace_numba(
        n_nodes,
        indptr,
        to,
        eid,
        weight,
        origin,
        tol,
        dist,
        pred_node,
        pred_edge,
        heap_node,
        heap_dist,
        pos,
    ):
        for i in range(n_nodes):
            dist[i] = np.inf
            pred_node[i] = -1
            pred_edge[i] = -1
            pos[i] = -1

        size = 0
        dist[origin] = 0.0
        size = _heap_push(heap_node, heap_dist, pos, size, origin, 0.0)

        while size > 0:
            node, d, size = _heap_pop(heap_node, heap_dist, pos, size)
            if d > dist[node] + tol:
                continue

            start = indptr[node]
            end = indptr[node + 1]
            for k in range(start, end):
                nbr = to[k]
                e = eid[k]
                nd = d + weight[e]
                if nd + tol < dist[nbr]:
                    dist[nbr] = nd
                    pred_node[nbr] = node
                    pred_edge[nbr] = e
                    if pos[nbr] == -1:
                        size = _heap_push(heap_node, heap_dist, pos, size, nbr, nd)
                    else:
                        _heap_decrease_key(heap_node, heap_dist, pos, nbr, nd)

