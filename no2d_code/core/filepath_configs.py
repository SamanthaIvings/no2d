import os

DATA_INPUTS_DIR = "inputs/fw_inputs_sy_simplified"
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

UE_RESULTS_FILE = "ue_results.npz"


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
