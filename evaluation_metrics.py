import platform
import resource
from typing import Any, Dict

import torch


def _max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def reset_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def collect_memory_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "process_peak_rss_mb": _max_rss_mb(),
    }

    if torch.cuda.is_available():
        stats.update(
            {
                "device": "cuda",
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
            }
        )
    elif torch.backends.mps.is_available():
        stats.update(
            {
                "device": "mps",
                "mps_current_allocated_mb": torch.mps.current_allocated_memory() / (1024 * 1024),
                "mps_driver_allocated_mb": torch.mps.driver_allocated_memory() / (1024 * 1024),
            }
        )
    else:
        stats["device"] = "cpu"

    return stats
