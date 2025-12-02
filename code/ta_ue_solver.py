from __future__ import annotations

from pathlib import Path

from frank_wolfe import FrankWolfeSettings, frank_wolfe_user_equilibrium
from input_output_utils import load_inputs, save_results


def main() -> None:
    distance_tolerance = 58.6
    data_root_dir = Path("C:/Users/elp24vs/Documents/Samantha Project/no2d/data/")
    input_dir = data_root_dir / "inputs"
    output_dir = data_root_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    network, od_data = load_inputs(input_dir, distance_tolerance)

    settings = FrankWolfeSettings(
        result_difference_tolerance=1e-5,
        step_limit=125000,
        verbose_progress_log=True,
        log_print_frequency=50
    )
    flow, crit1, crit2, iterations, lbd = frank_wolfe_user_equilibrium(network, od_data, settings)

    save_results(crit1, crit2, flow, iterations, lbd, output_dir, "_test")


if __name__ == "__main__":
    main()
