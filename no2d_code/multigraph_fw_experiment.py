from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODES = ("car", "bus", "bike")


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    facility: str  # "mixed" | "bus" | "bike"
    length_km: float
    cap: float
    speed_car: Optional[float]
    speed_bus: Optional[float]
    speed_bike: Optional[float]
    alpha: float = 0.15
    beta: float = 4.0


class Network:
    def __init__(self, edges: List[Edge]) -> None:
        self.edges = edges
        self.nodes = sorted(set([e.u for e in edges] + [e.v for e in edges]))
        self._out: Dict[Tuple[int, str], List[int]] = {}
        for i, e in enumerate(edges):
            for m in MODES:
                if self._edge_speed(e, m) is not None:
                    self._out.setdefault((e.u, m), []).append(i)

    def outgoing(self, u: int, mode: str) -> List[int]:
        return self._out.get((u, mode), [])

    @staticmethod
    def _edge_speed(e: Edge, mode: str) -> Optional[float]:
        if mode == "car":
            return e.speed_car
        if mode == "bus":
            return e.speed_bus
        if mode == "bike":
            return e.speed_bike
        return None


def _dijkstra(net: Network, mode: str, source: int, edge_cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dist = np.full(max(net.nodes) + 1, np.inf, dtype=float)
    prev_edge = np.full(max(net.nodes) + 1, -1, dtype=int)

    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for ei in net.outgoing(u, mode):
            c = edge_cost[ei]
            if not np.isfinite(c):
                continue
            v = net.edges[ei].v
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                prev_edge[v] = ei
                heapq.heappush(pq, (nd, v))

    return dist, prev_edge


def _reconstruct_path(net: Network, prev_edge: np.ndarray, origin: int, dest: int) -> List[int]:
    path: List[int] = []
    cur = dest
    while cur != origin:
        ei = int(prev_edge[cur])
        if ei < 0:
            raise RuntimeError("Path reconstruction failed.")
        path.append(ei)
        cur = net.edges[ei].u
    path.reverse()
    return path


def _bpr_time_h(t0_h: float, z: float, cap: float, alpha: float, beta: float, eps: float) -> float:
    c = max(cap, eps)
    r = max(z / c, eps)
    return t0_h * (1.0 + alpha * (r**beta))


def _freeflow_time_h(e: Edge, mode: str) -> float:
    sp = Network._edge_speed(e, mode)
    if sp is None:
        return math.inf
    return e.length_km / sp


def build_physical_network() -> Tuple[Dict[int, Tuple[float, float]], Network]:
    pos = {
        0: (0, 2), 1: (1, 2), 2: (2, 2), 3: (3, 2), 4: (4, 2),
        5: (0, 1), 6: (1, 1), 7: (2, 1), 8: (3, 1), 9: (4, 1),
        10: (0, 0), 11: (1, 0), 12: (2, 0), 13: (3, 0), 14: (4, 0),
        15: (2, 3),
        16: (2, -1),
    }

    edges: List[Edge] = []

    def add_bidir(e: Edge) -> None:
        edges.append(e)
        edges.append(
            Edge(
                u=e.v, v=e.u, facility=e.facility, length_km=e.length_km, cap=e.cap,
                speed_car=e.speed_car, speed_bus=e.speed_bus, speed_bike=e.speed_bike,
                alpha=e.alpha, beta=e.beta,
            )
        )

    def mixed(u: int, v: int, length_km: float, cap: float) -> None:
        add_bidir(
            Edge(
                u=u, v=v, facility="mixed", length_km=length_km, cap=cap,
                speed_car=60.0, speed_bus=45.0, speed_bike=18.0,
                alpha=0.15, beta=4.0,
            )
        )

    def bike_lane(u: int, v: int, length_km: float, cap: float = 1800.0) -> None:
        add_bidir(
            Edge(
                u=u, v=v, facility="bike", length_km=length_km, cap=cap,
                speed_car=None, speed_bus=None, speed_bike=26.0,
                alpha=0.15, beta=4.0,
            )
        )

    def bus_lane(u: int, v: int, length_km: float, cap: float = 110.0) -> None:
        add_bidir(
            Edge(
                u=u, v=v, facility="bus", length_km=length_km, cap=cap,
                speed_car=None, speed_bus=55.0, speed_bike=22.0,  # bikes can use bus lanes
                alpha=0.15, beta=4.0,
            )
        )

    for row in [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14)]:
        for a, b in zip(row[:-1], row[1:]):
            mixed(a, b, length_km=1.0, cap=800.0)

    for col in [(0, 5, 10), (1, 6, 11), (2, 7, 12), (3, 8, 13), (4, 9, 14)]:
        for a, b in zip(col[:-1], col[1:]):
            mixed(a, b, length_km=0.9, cap=760.0)

    mixed(6, 8, length_km=1.1, cap=650.0)
    mixed(11, 13, length_km=1.1, cap=650.0)
    mixed(1, 7, length_km=1.2, cap=600.0)
    mixed(7, 3, length_km=1.2, cap=600.0)
    mixed(15, 2, length_km=0.8, cap=520.0)
    mixed(12, 16, length_km=0.8, cap=520.0)

    mixed(7, 8, length_km=1.0, cap=420.0)

    # Fewer dedicated segments:
    # Bike lane only on a part of the top row and one vertical connector
    bike_lane(1, 2, 1.0)
    bike_lane(2, 3, 1.0)
    bike_lane(2, 7, 0.9)

    # Bus lane only on a part of the middle row and one part of bottom row
    bus_lane(5, 6, 1.0)
    bus_lane(6, 7, 1.0)
    bus_lane(12, 13, 1.0)

    return pos, Network(edges)


def compute_edge_costs(
    net: Network,
    x_car: np.ndarray,
    x_bus_pax: np.ndarray,
    x_bike: np.ndarray,
    *,
    occ_bus: float,
    pcu_bus: float,
    pcu_bike_mixed: float,
    pcu_bike_on_bus: float,
    w_bike: float,
    eps: float,
) -> Dict[str, np.ndarray]:
    c = {m: np.full(len(net.edges), np.inf, dtype=float) for m in MODES}

    for i, e in enumerate(net.edges):
        xbus_veh = x_bus_pax[i] / max(occ_bus, eps)

        if e.facility == "mixed":
            z_mixed = x_car[i] + pcu_bus * xbus_veh + pcu_bike_mixed * x_bike[i]
            z_for_bus = x_car[i]  # bus "static-ish": only cars slow it down here
            if e.speed_car is not None:
                t0 = _freeflow_time_h(e, "car")
                c["car"][i] = _bpr_time_h(t0, z_mixed, e.cap, e.alpha, e.beta, eps)
            if e.speed_bus is not None:
                t0 = _freeflow_time_h(e, "bus")
                c["bus"][i] = _bpr_time_h(t0, z_for_bus, e.cap, e.alpha, e.beta, eps)
            if e.speed_bike is not None:
                t0 = _freeflow_time_h(e, "bike")
                c["bike"][i] = w_bike * _bpr_time_h(t0, z_mixed, e.cap, e.alpha, e.beta, eps)

        elif e.facility == "bus":
            z_bus = xbus_veh + pcu_bike_on_bus * x_bike[i]
            if e.speed_bus is not None:
                t0 = _freeflow_time_h(e, "bus")
                c["bus"][i] = t0  # static on bus lanes
            if e.speed_bike is not None:
                t0 = _freeflow_time_h(e, "bike")
                c["bike"][i] = w_bike * _bpr_time_h(t0, z_bus, e.cap, e.alpha, e.beta, eps)

        elif e.facility == "bike":
            z_bike = x_bike[i]
            if e.speed_bike is not None:
                t0 = _freeflow_time_h(e, "bike")
                c["bike"][i] = w_bike * _bpr_time_h(t0, z_bike, e.cap, e.alpha, e.beta, eps)

        else:
            raise ValueError(f"Unknown facility: {e.facility}")

    return c


def shortest_path_cost(net: Network, mode: str, origin: int, dest: int, edge_costs: np.ndarray) -> float:
    dist, _ = _dijkstra(net, mode, origin, edge_costs)
    return float(dist[dest])


def aon_flows_for_mode(
    net: Network,
    mode: str,
    od_list: List[Tuple[int, int, float]],
    edge_costs: np.ndarray,
) -> np.ndarray:
    y = np.zeros(len(net.edges), dtype=float)
    for o, d, q in od_list:
        if q <= 0.0:
            continue
        dist, prev = _dijkstra(net, mode, o, edge_costs)
        if not np.isfinite(dist[d]):
            continue
        path = _reconstruct_path(net, prev, o, d)
        for ei in path:
            y[ei] += q
    return y


def logit_mode_split(
    costs_by_mode: Dict[str, float],
    theta: float,
) -> Dict[str, float]:
    vals = {}
    for m, cm in costs_by_mode.items():
        vals[m] = math.exp(-theta * cm) if math.isfinite(cm) else 0.0
    s = sum(vals.values())
    if s <= 0.0:
        return {m: 0.0 for m in MODES}
    return {m: vals[m] / s for m in MODES}


def run_autobalanced_assignment(
    net: Network,
    od_phys: List[Tuple[int, int, float]],
    *,
    iters: int,
    theta: float,
    occ_bus: float,
    pcu_bus: float,
    pcu_bike_mixed: float,
    pcu_bike_on_bus: float,
    w_bike: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], float]:
    x_car = np.zeros(len(net.edges), dtype=float)
    x_bus = np.zeros(len(net.edges), dtype=float)
    x_bike = np.zeros(len(net.edges), dtype=float)

    for k in range(1, iters + 1):
        c = compute_edge_costs(
            net, x_car, x_bus, x_bike,
            occ_bus=occ_bus,
            pcu_bus=pcu_bus,
            pcu_bike_mixed=pcu_bike_mixed,
            pcu_bike_on_bus=pcu_bike_on_bus,
            w_bike=w_bike,
            eps=eps,
        )

        od_by_mode: Dict[str, List[Tuple[int, int, float]]] = {m: [] for m in MODES}
        mode_totals = {m: 0.0 for m in MODES}

        for o, d, q in od_phys:
            best_costs = {m: shortest_path_cost(net, m, o, d, c[m]) for m in MODES}
            shares = logit_mode_split(best_costs, theta=theta)
            for m in MODES:
                qm = shares[m] * q
                if qm > 0.0 and math.isfinite(best_costs[m]):
                    od_by_mode[m].append((o, d, qm))
                    mode_totals[m] += qm

        y_car = aon_flows_for_mode(net, "car", od_by_mode["car"], c["car"])
        y_bus = aon_flows_for_mode(net, "bus", od_by_mode["bus"], c["bus"])
        y_bike = aon_flows_for_mode(net, "bike", od_by_mode["bike"], c["bike"])

        step = 2.0 / (k + 2.0)
        x_car = (1.0 - step) * x_car + step * y_car
        x_bus = (1.0 - step) * x_bus + step * y_bus
        x_bike = (1.0 - step) * x_bike + step * y_bike

    c_final = compute_edge_costs(
        net, x_car, x_bus, x_bike,
        occ_bus=occ_bus,
        pcu_bus=pcu_bus,
        pcu_bike_mixed=pcu_bike_mixed,
        pcu_bike_on_bus=pcu_bike_on_bus,
        w_bike=w_bike,
        eps=eps,
    )

    total_q = sum(q for _, _, q in od_phys)
    mode_share = {m: (mode_totals[m] / total_q if total_q > 0 else 0.0) for m in MODES}

    mixed_car_costs = []
    for i, e in enumerate(net.edges):
        if e.facility == "mixed" and e.speed_car is not None:
            mixed_car_costs.append(60.0 * float(c_final["car"][i]))
    avg_mixed_car_time_min = float(np.mean(mixed_car_costs)) if mixed_car_costs else 0.0

    return x_car, x_bus, x_bike, mode_share, avg_mixed_car_time_min


def plot_physical_graph(pos: Dict[int, Tuple[float, float]], net: Network) -> None:
    plt.figure(figsize=(11, 5))
    for e in net.edges:
        if e.u > e.v:
            continue
        x1, y1 = pos[e.u]
        x2, y2 = pos[e.v]

        if e.facility == "mixed":
            plt.plot([x1, x2], [y1, y2], "k-", lw=1.6, alpha=0.85)
        elif e.facility == "bike":
            plt.plot([x1, x2], [y1, y2], "g--", lw=2.4, alpha=0.95)
        elif e.facility == "bus":
            plt.plot([x1, x2], [y1, y2], "b--", lw=2.4, alpha=0.95)

    for n, (x, y) in pos.items():
        plt.scatter(x, y, s=60)
        plt.text(x, y + 0.06, str(n), ha="center")

    handles = [
        Line2D([0], [0], color="k", lw=2, linestyle="-", label="Mixed road"),
        Line2D([0], [0], color="g", lw=2.5, linestyle="--", label="Bike lane"),
        Line2D([0], [0], color="b", lw=2.5, linestyle="--", label="Bus lane (bikes allowed)"),
    ]
    plt.legend(handles=handles, loc="upper right")
    plt.title("Physical network (fewer dedicated segments; bikes can use bus lanes)")
    plt.axis("off")
    plt.show()


def main() -> None:
    pos, net = build_physical_network()
    plot_physical_graph(pos, net)

    od_base = [
        (0, 14, 0.55),
        (10, 4, 0.25),
        (15, 16, 0.20),
    ]

    demands = np.linspace(1000.0, 200000.0, 100)

    W_BIKE = 0.55          # bike perceived edge cost smaller
    OCC_BUS = 45.0         # many passengers per bus
    THETA = 0.9            # logit sensitivity to cost (bigger => more deterministic)

    PCU_BUS = 2.2
    PCU_BIKE_MIXED = 0.15
    PCU_BIKE_ON_BUS = 0.03
    EPS = 1e-8

    avg_car_mixed = []
    shares_car = []
    shares_bus = []
    shares_bike = []

    for total_demand in demands:
        od = [(o, d, frac * total_demand) for (o, d, frac) in od_base]
        _, _, _, share, avg_mixed_car_min = run_autobalanced_assignment(
            net,
            od,
            iters=120,
            theta=THETA,
            occ_bus=OCC_BUS,
            pcu_bus=PCU_BUS,
            pcu_bike_mixed=PCU_BIKE_MIXED,
            pcu_bike_on_bus=PCU_BIKE_ON_BUS,
            w_bike=W_BIKE,
            eps=EPS,
        )

        avg_car_mixed.append(avg_mixed_car_min)
        shares_car.append(share["car"])
        shares_bus.append(share["bus"])
        shares_bike.append(share["bike"])

    plt.figure()
    plt.plot(demands, avg_car_mixed, "o-")
    plt.xlabel("Total OD demand (passengers / time unit)")
    plt.ylabel("Avg mixed-road car time (min per link)")
    plt.title("Congestion on mixed roads as demand rises")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(demands, shares_car, "o-", label="Car share")
    plt.plot(demands, shares_bus, "s--", label="Bus share (static on bus lanes)")
    plt.plot(demands, shares_bike, "^-.", label="Bike share (cheaper edges)")
    plt.xlabel("Total OD demand (passengers / time unit)")
    plt.ylabel("Endogenous mode share")
    plt.title("Auto-balanced mode split driven by congested costs")
    plt.grid(True)
    plt.legend()
    plt.show()

    i = len(demands) // 2
    print(f"At demand={demands[i]:.0f}:")
    print(f"  avg mixed-road car time = {avg_car_mixed[i]:.3f} min/link")
    print(f"  mode shares: car {shares_car[i]:.2f}, bus {shares_bus[i]:.2f}, bike {shares_bike[i]:.2f}")


if __name__ == "__main__":
    main()
