"""Category-aware ready queue management and process selection.

Supports:
- Per-category ready queues (OLTP, OLAP, VECTOR, BACKGROUND)
- Cgroups v2 proportional share weighting
- CFS virtual runtime (vruntime) tracking
- Lock dependency awareness (dependency checking).
"""

from typing import List, Dict, Optional
import heapq
from workloads.process import DatabaseProcess


class DatabaseReadyQueue:
    def __init__(self, categories: Optional[List[str]] = None):
        self.categories = categories or ["OLTP", "OLAP", "VECTOR", "BACKGROUND"]
        # Queues indexed by category
        self.queues: Dict[str, List[DatabaseProcess]] = {cat: [] for cat in self.categories}
        self.completed_processes: List[DatabaseProcess] = []

    def add_process(self, process: DatabaseProcess):
        """Inserts an arriving or preempted process into its category queue."""
        cat = process.category if process.category in self.queues else "OLTP"
        self.queues[cat].append(process)

    def is_empty(self) -> bool:
        return all(len(q) == 0 for q in self.queues.values())

    def get_active_processes(self) -> List[DatabaseProcess]:
        """Returns all currently active (ready/running) processes."""
        active = []
        for q in self.queues.values():
            active.extend(q)
        return active

    def select_next_process(
        self,
        category: str,
        scheduling_policy: str = "PROPORTIONAL",
        finished_pids: Optional[set] = None
    ) -> Optional[DatabaseProcess]:
        """Picks the next ready process from category queue according to policy."""
        q = self.queues.get(category, [])
        if not q:
            return None

        finished = finished_pids or set(p.pid for p in self.completed_processes)

        # Filter out processes whose dependencies haven't finished yet
        ready_candidates = [p for p in q if all(dep in finished for dep in p.dependencies)]
        
        # If all candidate processes in this category are blocked on dependencies, pick the first one with minimum lock wait
        if not ready_candidates:
            ready_candidates = q

        if scheduling_policy == "CFS":
            # Pick process with smallest vruntime
            selected = min(ready_candidates, key=lambda p: p.vruntime_ms)
        elif scheduling_policy == "PRIORITY":
            # Highest priority first (10 highest)
            selected = max(ready_candidates, key=lambda p: (p.priority, -p.arrival_time_ms))
        elif scheduling_policy == "FIFO":
            selected = min(ready_candidates, key=lambda p: p.arrival_time_ms)
        else:  # PROPORTIONAL / WEIGHTED
            # Weighted by process cgroup_weight and priority
            selected = max(ready_candidates, key=lambda p: (p.cgroup_weight * p.priority, -p.vruntime_ms))

        q.remove(selected)
        return selected

    def record_completed(self, process: DatabaseProcess):
        self.completed_processes.append(process)
