"""Static and Priority-Based Scheduling Baselines.

Implements:
- Static CPU budget partitioning (e.g., fixed quotas)
- Strict Priority / FIFO baseline.
"""

from typing import Dict, List, Optional
from workloads.process import DatabaseProcess


class StaticAllocationScheduler:
    def __init__(self, fixed_allocations: Optional[Dict[str, float]] = None):
        self.allocations = fixed_allocations or {
            "OLTP": 0.40,
            "OLAP": 0.30,
            "VECTOR": 0.20,
            "BACKGROUND": 0.10
        }

    def get_macro_action(self) -> Dict[str, float]:
        return self.allocations.copy()


class PriorityFIFOScheduler:
    def __init__(self, use_priority: bool = True):
        self.use_priority = use_priority

    def select_process(self, ready_processes: List[DatabaseProcess]) -> Optional[DatabaseProcess]:
        if not ready_processes:
            return None
        if self.use_priority:
            # Highest priority first, tie break by arrival time
            return max(ready_processes, key=lambda p: (p.priority, -p.arrival_time_ms))
        else:
            # Pure FIFO
            return min(ready_processes, key=lambda p: p.arrival_time_ms)
