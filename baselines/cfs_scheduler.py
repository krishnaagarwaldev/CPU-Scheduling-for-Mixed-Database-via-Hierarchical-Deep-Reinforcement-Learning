"""Linux Completely Fair Scheduler (CFS) Baseline.

Implements the standard Linux CFS scheduling algorithm used as the primary
baseline in Table 1 (Xing et al. 2025):
- Tracks process virtual runtime: vruntime += runtime * (NICE_0_LOAD / weight)
- Always dispatches the ready process with minimum vruntime
- Allocates equal proportional shares across categories without semantic database awareness.
"""

from typing import List, Dict, Optional
import numpy as np
from workloads.process import DatabaseProcess


class LinuxCFSScheduler:
    def __init__(self, nice_0_load: float = 1024.0):
        self.nice_0_load = nice_0_load

    def calculate_vruntime_delta(self, runtime_ms: float, weight: int) -> float:
        clamped_weight = max(1, weight)
        return runtime_ms * (self.nice_0_load / clamped_weight)

    def select_process(self, ready_processes: List[DatabaseProcess]) -> Optional[DatabaseProcess]:
        """Picks process with lowest vruntime."""
        if not ready_processes:
            return None
        return min(ready_processes, key=lambda p: p.vruntime_ms)

    def update_vruntime(self, process: DatabaseProcess, runtime_ms: float):
        delta = self.calculate_vruntime_delta(runtime_ms, process.cgroup_weight)
        process.vruntime_ms += delta

    def get_macro_action(self, categories: List[str]) -> Dict[str, float]:
        """CFS is unaware of database categories: assigns equal default shares."""
        n = len(categories)
        return {cat: 1.0 / n for cat in categories}
