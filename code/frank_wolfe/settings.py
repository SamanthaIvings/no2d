from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrankWolfeSettings:
    result_difference_tolerance: float = 1e-5
    step_limit: int = 200
    verbose_progress_log: bool = True
    log_print_frequency: int = 10
    aon_verbose: bool = False
    origins_progress_every: int = 25


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds/60:.2f}m"
    return f"{seconds/3600:.2f}h"


def should_log_iteration(iteration: int, settings: FrankWolfeSettings) -> bool:
    return iteration == 1 or iteration % settings.log_print_frequency == 0


def log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)
