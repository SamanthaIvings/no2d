from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from models import NetworkParameters, ODData


def load_inputs(input_directory: Path, distance_tolerance: float) -> Tuple[NetworkParameters, ODData]:
    edges = pd.read_csv(input_directory / "edges.csv")
    nodes = pd.read_csv(input_directory / "nodes.csv")

    network = _build_network_data(edges, nodes)

    od_matrix = _load_od_matrix(input_directory, distance_tolerance)
    demand_vector = _load_demand_vector(input_directory)

    origin_node_ids, destination_node_ids, demand_vector, od_matrix = _align_od_and_demand(
        od_matrix=od_matrix,
        demand_vector=demand_vector
    )

    origin_node_ids, destination_node_ids, demand_vector = _remove_intra_zone_trips(
        od_matrix=od_matrix,
        origin_node_ids=origin_node_ids,
        destination_node_ids=destination_node_ids,
        demand_vector=demand_vector
    )

    od_data = ODData(
        origins=origin_node_ids,
        destinations=destination_node_ids,
        demand=demand_vector
    )

    return network, od_data


def save_results(crit1, crit2, flow, number_of_iterations, lbd, output_dir, suffix):
    np.savetxt(output_dir / f"UE_flow{suffix}.csv", flow, delimiter=",")
    np.savetxt(output_dir / f"UE_crit1and2{suffix}.csv", np.array([crit1, crit2]), delimiter=",")
    np.savetxt(output_dir / f"UE_iter{suffix}.csv", np.array([number_of_iterations], dtype=int), delimiter=",")
    np.savetxt(output_dir / f"UE_LBD{suffix}.csv", np.array([lbd]), delimiter=",")


def _build_network_data(edges: pd.DataFrame, nodes: pd.DataFrame) -> NetworkParameters:
    from_node_ids = edges["u"].to_numpy(dtype=int)
    to_node_ids = edges["v"].to_numpy(dtype=int)

    edge_length_meters = edges["length"].to_numpy(dtype=float)
    speed_limit_km_per_hour = edges["speedlim"].to_numpy(dtype=float)

    edge_capacity = edges["capacity"].to_numpy(dtype=float)
    critical_density = edges["criticalDensity"].to_numpy(dtype=float)

    free_flow_travel_time_hours = (edge_length_meters / 1000.0) / speed_limit_km_per_hour

    number_of_edges = int(edges.shape[0])
    bpr_alpha = np.full(number_of_edges, 0.15, dtype=float)
    bpr_beta = np.full(number_of_edges, 4.0, dtype=float)
    bpr_epsilon = 0.0

    node_count_from_nodes_csv = int(nodes.shape[0])
    node_count_from_edges = int(max(from_node_ids.max(), to_node_ids.max()) + 1)
    number_of_nodes = max(node_count_from_nodes_csv, node_count_from_edges)

    return NetworkParameters(
        from_nodes=from_node_ids,
        to_nodes=to_node_ids,
        free_flow_time_h=free_flow_travel_time_hours,
        capacity=edge_capacity,
        critical_density=critical_density,
        alpha=bpr_alpha,
        beta=bpr_beta,
        bpr_eps=bpr_epsilon,
        n_nodes=number_of_nodes,
    )


def _load_od_matrix(input_directory: Path, distance_tolerance: float) -> np.ndarray:
    od_list_path = input_directory / f"OD_list_tol{distance_tolerance}.csv"
    od_matrix = pd.read_csv(od_list_path, header=None).to_numpy(dtype=float)
    return od_matrix[0:, :]


def _load_demand_vector(input_directory: Path) -> np.ndarray:
    demand_path = input_directory / "demand.csv"
    return pd.read_csv(demand_path, header=None).iloc[:, 0].to_numpy(dtype=float)


def _align_od_and_demand(od_matrix: np.ndarray, demand_vector: np.ndarray):
    origin_node_ids = od_matrix[:, 2].astype(int)
    destination_node_ids = od_matrix[:, 3].astype(int)

    matched_length = min(demand_vector.size, origin_node_ids.size)
    demand_vector = demand_vector[:matched_length]
    origin_node_ids = origin_node_ids[:matched_length]
    destination_node_ids = destination_node_ids[:matched_length]
    od_matrix = od_matrix[:matched_length, :]

    return origin_node_ids, destination_node_ids, demand_vector, od_matrix


def _remove_intra_zone_trips(
    od_matrix: np.ndarray,
    origin_node_ids: np.ndarray,
    destination_node_ids: np.ndarray,
    demand_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_intra_zone_trip = od_matrix[:, 0] == od_matrix[:, 1]

    return (
        origin_node_ids[~is_intra_zone_trip],
        destination_node_ids[~is_intra_zone_trip],
        demand_vector[~is_intra_zone_trip],
    )
