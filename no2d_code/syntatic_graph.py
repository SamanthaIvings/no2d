from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from no2d_code.frank_wolfe.frankwolfe_ue_flex import solve_frank_wolfe_user_equilibrium
from no2d_code.frank_wolfe.frank_wolfe_classes import FWRunConfig
from no2d_code.frank_wolfe.digraph import Digraph
from no2d_code.frank_wolfe.shortest_path_tree_builder import get_shortest_path_tree_edges_cell


# =========================
# BPR (flow-based)
# =========================

def bpr_flow_smooth(
    time: np.ndarray,
    flow: np.ndarray,
    capacity: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    alpha = params[:, 0]
    beta = params[:, 1]
    eps = float(params[0, 2])

    ratio = flow / np.maximum(capacity, 1e-12)
    ratio = np.maximum(ratio, eps)

    return time * (1.0 + alpha * np.power(ratio, beta))


# =========================
# Helpers / data
# =========================

@dataclass(frozen=True)
class FlowStats:
    sum: float
    min: float
    max: float
    mean: float
    p50: float
    p90: float
    p99: float
    nnz_frac: float


def _make_synthetic_edges() -> pd.DataFrame:
    rows = [
        (0, 1, 1000.0, 60.0, 600.0, 30.0),
        (1, 3, 1000.0, 60.0, 600.0, 30.0),
        (0, 2, 1000.0, 60.0, 400.0, 30.0),
        (2, 3, 1000.0, 60.0, 400.0, 30.0),
        (1, 2, 400.0, 60.0, 300.0, 30.0),
        (2, 1, 400.0, 60.0, 300.0, 30.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["u", "v", "length", "speedlim", "capacity", "criticalDensity"],
    )


def _make_synthetic_od() -> tuple[np.ndarray, np.ndarray]:
    od = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 2, 3],
        ],
        dtype=np.int64,
    )
    demand = np.array([900.0, 120.0, 80.0, 60.0], dtype=np.float64)
    return od, demand


def _init_logs(out_dir: str, stepbreak: int) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    txt_name = os.path.join(out_dir, "run_log.txt")
    crit_log_name = os.path.join(out_dir, "crit_log.csv")

    crit_log = np.zeros((stepbreak + 5, 1), dtype=np.float64)
    np.savetxt(crit_log_name, crit_log, delimiter=",")

    with open(txt_name, "w", encoding="utf-8") as f:
        f.write("synthetic run\n")

    return txt_name, crit_log_name


def _flow_stats(x: np.ndarray) -> FlowStats:
    x = np.asarray(x, dtype=np.float64)
    return FlowStats(
        sum=float(np.sum(x)),
        min=float(np.min(x)),
        max=float(np.max(x)),
        mean=float(np.mean(x)),
        p50=float(np.quantile(x, 0.50)),
        p90=float(np.quantile(x, 0.90)),
        p99=float(np.quantile(x, 0.99)),
        nnz_frac=float(np.count_nonzero(x) / x.size),
    )


def _edge_report_df(
    graph: Digraph,
    flow: np.ndarray,
    traveltime: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_id": np.arange(flow.size),
            "u": graph.u,
            "v": graph.v,
            "flow": flow,
            "capacity": graph.capacity,
            "v_over_c": flow / np.maximum(graph.capacity, 1e-12),
            "t": traveltime,
            "flow_x_t": flow * traveltime,
        }
    ).sort_values("flow", ascending=False, ignore_index=True)


# =========================
# OD diagnostics (post-UE)
# =========================

def _od_shortest_costs(
    graph: Digraph,
    origins: np.ndarray,
    edge_cost: np.ndarray,
) -> Dict[int, np.ndarray]:
    old_w = graph.weight.copy()
    graph.weight = edge_cost

    out: Dict[int, np.ndarray] = {}
    for s in np.unique(origins):
        E = get_shortest_path_tree_edges_cell(graph, int(s))
        dist = np.full(graph.n_nodes, np.inf)
        dist[s] = 0.0
        for t in range(graph.n_nodes):
            path = E[t]
            if path:
                dist[t] = float(np.sum(edge_cost[np.asarray(path, dtype=int)]))
        out[int(s)] = dist

    graph.weight = old_w
    return out


def _od_analysis_df(
    graph: Digraph,
    origin_destination: np.ndarray,
    demand: np.ndarray,
    edge_cost: np.ndarray,
) -> pd.DataFrame:
    s = origin_destination[:, 2].astype(int)
    t = origin_destination[:, 3].astype(int)

    dist_by_origin = _od_shortest_costs(graph, s, edge_cost)
    shortest = np.array(
        [dist_by_origin[si][ti] for si, ti in zip(s, t)], dtype=float
    )

    return pd.DataFrame(
        {
            "origin": s,
            "dest": t,
            "demand": demand,
            "shortest_cost": shortest,
        }
    )


def _plot_convergence(crit_log_csv: str) -> None:
    A = np.loadtxt(crit_log_csv, delimiter=",")
    if A.ndim != 2:
        return

    it = np.arange(A.shape[0])
    rg = A[:, 0]
    mask = rg > 0.0

    plt.figure()
    plt.semilogy(it[mask], rg[mask])
    plt.xlabel("Iteration")
    plt.ylabel("FW Relative Gap")
    plt.grid(True)
    plt.show()


# =========================
# Main
# =========================

def main() -> None:
    edges = _make_synthetic_edges()
    graph = Digraph.from_edges(edges)

    origin_destination, demand = _make_synthetic_od()

    out_dir = "out_synth"
    eps = 1e-6
    stepbreak = 2000

    txt_name, crit_log_name = _init_logs(out_dir, stepbreak)

    config = FWRunConfig(
        eps=eps,
        steplimit=stepbreak,
        txt_name=txt_name,
        crit_log_name=crit_log_name,
        crit_bests_name="",  # unused
    )

    result = solve_frank_wolfe_user_equilibrium(
        demand=demand,
        graph=graph,
        origin_destination=origin_destination,
        config=config,
    )

    flow = np.asarray(result.flows, dtype=float)

    time0 = np.asarray(graph.free_flow_travel_h, dtype=float)
    cap = np.asarray(graph.capacity, dtype=float)
    params = graph.bpr_params

    tt = bpr_flow_smooth(time0, flow, cap, params)

    print("iterations:", result.iterations)
    print("final FW relgap:", float(result.crit1))

    stats = _flow_stats(flow)
    print("flow stats:", stats)

    df_edges = _edge_report_df(graph, flow, tt)
    df_edges.to_csv(os.path.join(out_dir, "edges_final.csv"), index=False)

    print("total system travel time:", df_edges["flow_x_t"].sum())

    df_od = _od_analysis_df(graph, origin_destination, demand, tt)
    df_od.to_csv(os.path.join(out_dir, "od_costs.csv"), index=False)

    print("\nOD shortest path costs:")
    print(df_od.to_string(index=False))

    _plot_convergence(crit_log_name)


if __name__ == "__main__":
    main()
