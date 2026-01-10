from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from no2d_code.frank_wolfe import frank_wolfe_ue_solver
from no2d_code.frank_wolfe.frank_wolfe_classes import FWRunConfig
from no2d_code.frank_wolfe.shortestpathtree import Digraph


def bpr_flow_smooth(time, flow, capacity, params):
    alpha = params[:, 0]
    beta = params[:, 1]
    eps = float(params[0, 2])

    ratio = flow / np.maximum(capacity, 1e-12)
    ratio = np.maximum(ratio, eps)
    return time * (1.0 + alpha * ratio**beta)


def make_car_network() -> Digraph:
    edges = pd.DataFrame(
        [
            # u, v, length, speedlim, capacity, criticalDensity
            (0, 1, 1.0, 60.0, 600.0, 30.0),
            (1, 3, 1.0, 60.0, 600.0, 30.0),
            (0, 2, 1.0, 60.0, 600.0, 30.0),
            (2, 3, 1.0, 60.0, 600.0, 30.0),
        ],
        columns=["u", "v", "length", "speedlim", "capacity", "criticalDensity"],
    )
    return Digraph.from_edges(edges)


def make_bike_network() -> Digraph:
    edges = pd.DataFrame(
        [
            (0, 4, 1.2, 20.0, 300.0, 30.0),
            (4, 3, 1.2, 20.0, 300.0, 30.0),
        ],
        columns=["u", "v", "length", "speedlim", "capacity", "criticalDensity"],
    )
    return Digraph.from_edges(edges)


def run_modal_ue(gamma: float, total_demand: float, cfg: FWRunConfig):
    od = np.array([[0, 0, 0, 3]], dtype=int)

    demand_car = np.array([(1.0 - gamma) * total_demand], dtype=float)
    demand_bike = np.array([gamma * total_demand], dtype=float)

    car_graph = make_car_network()
    bike_graph = make_bike_network()

    car_res = frank_wolfe_ue_solver(demand_car, car_graph, od, cfg)
    bike_res = frank_wolfe_ue_solver(demand_bike, bike_graph, od, cfg)

    return car_graph, car_res, bike_graph, bike_res


def plot_combined_network():
    plt.figure(figsize=(6, 4))

    node_positions = {
        0: (0, 0),
        1: (1, 1),
        2: (1, -1),
        3: (2, 0),
        4: (1, 0),
    }

    car_edges = [(0, 1), (1, 3), (0, 2), (2, 3)]
    bike_edges = [(0, 4), (4, 3)]

    for u, v in car_edges:
        plt.plot(
            [node_positions[u][0], node_positions[v][0]],
            [node_positions[u][1], node_positions[v][1]],
            "k-",
            lw=2,
            label="car" if (u, v) == car_edges[0] else None,
        )

    for u, v in bike_edges:
        plt.plot(
            [node_positions[u][0], node_positions[v][0]],
            [node_positions[u][1], node_positions[v][1]],
            "g--",
            lw=3,
            label="bike" if (u, v) == bike_edges[0] else None,
        )

    for n, (x, y) in node_positions.items():
        plt.scatter(x, y, s=80)
        plt.text(x, y + 0.05, str(n), ha="center")

    plt.legend()
    plt.title("Car network extended by bicycle availability")
    plt.axis("off")
    plt.show()


def main():
    out_dir = "out_modal_demo"
    os.makedirs(out_dir, exist_ok=True)

    total_demand = 1000.0
    gammas = np.linspace(0.0, 0.6, 13)

    cfg = FWRunConfig(
        eps=1e-6,
        steplimit=1000,
        txt_name=os.path.join(out_dir, "run_log.txt"),
        crit_log_name=os.path.join(out_dir, "crit_log.csv"),
        crit_bests_name=os.path.join(out_dir, "crit_best.csv"),
    )

    car_tstt = []
    bike_tstt = []
    car_max_flow = []

    for g in gammas:
        car_graph, car_res, bike_graph, bike_res = run_modal_ue(
            g, total_demand, cfg
        )

        car_flow = car_res.flows
        bike_flow = bike_res.flows

        car_tt = bpr_flow_smooth(
            car_graph.free_flow_travel_h,
            car_flow,
            car_graph.capacity,
            car_graph.bpr_params,
        )
        bike_tt = bpr_flow_smooth(
            bike_graph.free_flow_travel_h,
            bike_flow,
            bike_graph.capacity,
            bike_graph.bpr_params,
        )

        car_tstt.append(np.dot(car_flow, car_tt))
        bike_tstt.append(np.dot(bike_flow, bike_tt))
        car_max_flow.append(car_flow.max())


    plot_combined_network()

    plt.figure()
    plt.plot(gammas, car_tstt, "o-", label="Car TSTT")
    plt.plot(gammas, bike_tstt, "s--", label="Bike TSTT")
    plt.xlabel("Bike demand share γ")
    plt.ylabel("System travel time")
    plt.title("System travel times vs modal shift")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(gammas, car_max_flow, "r^-")
    plt.xlabel("Bike demand share γ")
    plt.ylabel("Max car link flow")
    plt.title("Congestion relief on car network")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
