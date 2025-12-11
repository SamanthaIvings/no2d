import os

DATA_INPUTS_DIR = "inputs"
DATA_OUTPUTS_DIR = "outputs"

# input files
EDGES_CSV = "edges.csv"
NODES_CSV = "nodes.csv"
DEMAND_CSV = "demand.csv"

OD_LIST_PATTERN = "OD_list_tol{tol}.csv"


def od_list_filename(tol: float) -> str:
    return OD_LIST_PATTERN.format(tol=tol)


# log / diagnostic files (stored in parentDir root)
OUT_LOG_TXT = "Out_TA_HPC_UE.txt"
ALL_CRIT_CSV = "All_crit1_crit2_UE.csv"
BEST_CRIT_CSV = "Best_crit1_crit2_UE.csv"

# main outputs (stored in outputs subfolder)
UE_FLOW_CSV = "UE_flow.csv"
UE_FLOW_BEST_CSV = "UE_flow_best.csv"
UE_CRIT_CSV = "UE_crit1and2.csv"
UE_CRIT_BEST_CSV = "UE_crit1and2_best.csv"
UE_L_CSV = "UE_L.csv"
UE_L_BEST_CSV = "UE_L_best.csv"
UE_LBD_CSV = "UE_LBD.csv"
UE_LBD_BEST_CSV = "UE_LBD_best.csv"


def join(*parts: str) -> str:
    return os.path.join(*parts)


def inputs_dir(parent_dir: str) -> str:
    return join(parent_dir, DATA_INPUTS_DIR)


def outputs_dir(parent_dir: str) -> str:
    return join(parent_dir, DATA_OUTPUTS_DIR)


def input_path(parent_dir: str, filename: str) -> str:
    return join(parent_dir, DATA_INPUTS_DIR, filename)


def output_path(parent_dir: str, filename: str) -> str:
    return join(parent_dir, DATA_OUTPUTS_DIR, filename)


def log_path(parent_dir: str, filename: str) -> str:
    return join(parent_dir, filename)
