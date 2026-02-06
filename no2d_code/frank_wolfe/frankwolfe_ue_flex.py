from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from no2d_code.frank_wolfe.all_or_nothing_assignment import compute_all_or_nothing_flow, build_edge_od_incidence_matrix
from no2d_code.frank_wolfe.beckmann import beckmann_objective_ue, beckmann_line_search_objective_ue
from no2d_code.frank_wolfe.bpr import bpr_flow
from no2d_code.frank_wolfe.frank_wolfe_classes import FWRunConfig, FWResult
from no2d_code.frank_wolfe.shortest_path_tree_builder import get_unique_origins, build_shortest_path_trees
from no2d_code.frank_wolfe.digraph import Digraph


def _fw_gap_rel_gap_tot_tt(flow, flow_y, traveltime):
    flow_p = flow_y - flow
    fw_gap = float(np.dot(traveltime, -flow_p))
    if fw_gap < 0.0:
        fw_gap = 0.0
    tot_tt = float(np.dot(traveltime, flow))
    rel_gap = fw_gap / max(tot_tt, 1e-12)
    return fw_gap, rel_gap, tot_tt


def _run_line_search(flow, flow_y, time, params, capacity) -> Tuple[float, float]:
    def fun(s: float):
        return beckmann_line_search_objective_ue(s, flow, flow_y, time, params, capacity)

    res = minimize_scalar(fun, bounds=(0.0, 1.0), method="bounded")
    step_new = float(res.x)
    f_star = float(fun(step_new))
    return step_new, f_star


@dataclass
class FWCriteriaState:
    config: FWRunConfig
    criteria_log: NDArray[np.float64]
    critBests: NDArray[np.float64]

    crit1: float = float("inf")
    crit2: float = float("inf")
    best_gap: float = float("inf")

    ue_flows_best: NDArray[np.float64] | None = None
    iter_best: int = 0

    prev_obj: float = float("nan")
    obj_init: float = float("nan")

    def init_best(self, flow: NDArray[np.float64], prev_obj: float):
        self.ue_flows_best = flow.copy()
        self.prev_obj = float(prev_obj)
        self.obj_init = float(prev_obj)

        self.best_gap = float("inf")
        self.crit1 = float("inf")
        self.crit2 = float("inf")
        self.iter_best = 0

    def update_gap(self, rel_gap: float):
        self.crit1 = rel_gap

    def update_move(self, crit2: float):
        self.crit2 = crit2

    def update_objective(self, obj_new: float) -> float:
        delta_obj = self.prev_obj - obj_new
        self.prev_obj = obj_new
        return delta_obj

    def maybe_update_best(self, step: int, flow: NDArray[np.float64]):
        if self.crit1 < self.best_gap:
            self.best_gap = self.crit1
            self.ue_flows_best = flow.copy()
            self.iter_best = int(step)

            self.critBests = np.atleast_2d(self.critBests)
            self.critBests = np.vstack(
                [self.critBests, np.array([self.best_gap, self.crit2, self.iter_best], dtype=float)]
            )
            np.savetxt(self.config.crit_bests_name, self.critBests, delimiter=",")

    def log_step(self, step: int):
        self.criteria_log[step, :] = np.array([self.crit1, self.crit2], dtype=float)
        np.savetxt(self.config.crit_log_name, self.criteria_log, delimiter=",")

    def converged(self) -> bool:
        return self.crit1 <= self.config.eps


def solve_frank_wolfe_user_equilibrium(
    demand: np.ndarray,
    graph: Digraph,
    origin_destination: np.ndarray,
    config: FWRunConfig,
    *,
    spt_parallel: bool = False,
    spt_workers: int | None = None,
) -> FWResult:
    print("[FW] Starting Frank–Wolfe UE solver")
    print(f"[FW] graph: n_nodes={graph.n_nodes}  n_edges={graph.u.size}")
    print(f"[FW] OD rows: {origin_destination.shape[0]}  demand_sum={int(np.sum(demand))}")

    time = graph.free_flow_travel_h
    capacity = graph.capacity
    params = graph.bpr_params

    criteria_log = np.loadtxt(config.crit_log_name, delimiter=",")
    critBests = np.loadtxt(config.crit_bests_name, delimiter=",")

    state = FWCriteriaState(config=config, criteria_log=criteria_log, critBests=critBests)

    flow = np.zeros(graph.u.size, dtype=float)
    graph.weight = bpr_flow(time, flow, capacity, params)

    origins = get_unique_origins(origin_destination)
    print(f"[FW] Unique origins: {int(origins.size)}")

    e_store = build_shortest_path_trees(
        graph,
        origins,
        parallel=spt_parallel,
        max_workers=spt_workers,
    )
    flow = compute_all_or_nothing_flow(demand, origin_destination, e_store, flow.size)

    traveltime = bpr_flow(time, flow, capacity, params)
    graph.weight = traveltime

    init_obj = beckmann_objective_ue(flow, time, params, capacity)
    state.init_best(flow=flow, prev_obj=init_obj)

    print(f"[FW] Objective init (AoN): {state.obj_init:.12e}")

    step = 0
    while not state.converged():
        step += 1

        if step == config.steplimit:
            print(f"[FW] Reached steplimit={config.steplimit}. Building Xa_Gi and stopping.")
            xa_gi_metrics = build_edge_od_incidence_matrix(origin_destination, e_store, flow.size)
            ue_flows = flow.copy()
            break

        traveltime = bpr_flow(time, flow, capacity, params)
        graph.weight = traveltime

        e_store = build_shortest_path_trees(
            graph,
            origins,
            parallel=spt_parallel,
            max_workers=spt_workers,
        )
        flow_y = compute_all_or_nothing_flow(demand, origin_destination, e_store, flow.size)
        flow_p = flow_y - flow

        _, rel_gap, _ = _fw_gap_rel_gap_tot_tt(flow, flow_y, traveltime)
        state.update_gap(rel_gap)

        if state.converged():
            print(f"[FW] Converged at iter={step} (rel_gap={state.crit1:.6e}). Building Xa_Gi...")
            xa_gi_metrics = build_edge_od_incidence_matrix(origin_destination, e_store, flow.size)
            ue_flows = flow.copy()
            state.log_step(step)
            break

        step_new, _ = _run_line_search(flow, flow_y, time, params, capacity)
        flow = flow + step_new * flow_p

        move_norm = float(np.linalg.norm(step_new * flow_p))
        base_norm = max(float(np.linalg.norm(flow)), 1e-12)
        state.update_move(move_norm / base_norm)

        obj_new = beckmann_objective_ue(flow, time, params, capacity)
        delta_obj = state.update_objective(obj_new)

        state.maybe_update_best(step, flow)
        state.log_step(step)

        print(f"[FW] iter={step}  rel_gap={state.crit1:.6e}  step={step_new:.6e}  dObj={delta_obj:.6e}")

        with open(config.txt_name, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}: completed iteration {step}.\n")

    obj_final = beckmann_objective_ue(ue_flows, time, params, capacity)
    obj_best = beckmann_objective_ue(
        state.ue_flows_best if state.ue_flows_best is not None else ue_flows,
        time,
        params,
        capacity,
    )

    print(f"[FW] Objective final:       {obj_final:.12e}")
    print(f"[FW] Objective improvement: {(state.obj_init - obj_final):.12e}")
    print(f"[FW] Objective best(trk):   {obj_best:.12e}  (iter_best={state.iter_best})")

    return FWResult(
        flows=ue_flows,
        flows_best=state.ue_flows_best if state.ue_flows_best is not None else ue_flows,
        crit1=state.crit1,
        crit2=state.crit2,
        crit1_best=state.best_gap,
        crit2_best=state.crit2,
        iterations=step,
        iter_best=state.iter_best,
        Xa_Gi=xa_gi_metrics,
        crit_log=state.criteria_log,
        crit_bests=state.critBests,
    )
