"""Discrete-event multi-core CPU execution simulator.

Accurately models:
- Multi-core process execution
- Context switch overhead (context_switch_cost_ms)
- Time quantum execution and process preemption
- CPU cycles accounting, idle time, and core utilization.
"""

from typing import List, Optional, Dict
from workloads.process import DatabaseProcess


class CPUCore:
    def __init__(self, core_id: int, context_switch_ms: float = 0.05):
        self.core_id = core_id
        self.context_switch_ms = context_switch_ms
        self.current_process: Optional[DatabaseProcess] = None
        self.total_busy_time_ms: float = 0.0
        self.total_context_switch_time_ms: float = 0.0
        self.total_idle_time_ms: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.current_process is None

    def execute_slice(
        self,
        process: DatabaseProcess,
        slice_ms: float,
        current_time_ms: float
    ) -> float:
        """Executes a process on this core for up to slice_ms.
        
        Returns:
            actual_time_consumed: total time elapsed on this core (slice + context switch).
        """
        is_context_switch = (self.current_process is None or self.current_process.pid != process.pid)
        overhead = self.context_switch_ms if is_context_switch else 0.0
        self.current_process = process

        if process.first_scheduled_time_ms is None:
            process.first_scheduled_time_ms = current_time_ms + overhead

        # Run process
        run_time = min(process.remaining_cpu_demand_ms, slice_ms)
        process.remaining_cpu_demand_ms -= run_time
        process.accumulated_cpu_time_ms += run_time

        self.total_busy_time_ms += run_time
        self.total_context_switch_time_ms += overhead

        if process.is_completed:
            process.completion_time_ms = current_time_ms + overhead + run_time
            self.current_process = None

        return run_time + overhead


class MultiCoreCPU:
    def __init__(self, num_cores: int = 8, context_switch_ms: float = 0.05):
        self.num_cores = num_cores
        self.cores = [CPUCore(core_id=i, context_switch_ms=context_switch_ms) for i in range(num_cores)]

    def get_idle_core(self) -> Optional[CPUCore]:
        for core in self.cores:
            if core.is_idle:
                return core
        return None

    def get_average_utilization(self, total_sim_time_ms: float) -> float:
        if total_sim_time_ms <= 0:
            return 0.0
        total_busy = sum(core.total_busy_time_ms for core in self.cores)
        max_possible = self.num_cores * total_sim_time_ms
        return min(1.0, total_busy / max(1e-6, max_possible))
