"""Database workload generator.

Creates realistic mixed database workloads (OLTP, OLAP, VECTOR, BACKGROUND)
complete with arrival times, CPU demands, priorities, and dependency DAGs.
"""

from typing import List, Dict, Optional
import numpy as np

from .process import DatabaseProcess
from .synthetic_distribution import WorkloadDistribution


class DatabaseWorkloadGenerator:
    def __init__(self, seed: int = 42):
        self.dist = WorkloadDistribution(seed=seed)
        self.rng = np.random.RandomState(seed)
        self.next_pid = 1

    def generate_mixed_workload(
        self,
        duration_ms: float = 10000.0,
        arrival_rates: Optional[Dict[str, float]] = None,
        dependency_prob: float = 0.25
    ) -> List[DatabaseProcess]:
        """Generates mixed database workload over a simulation window.
        
        Args:
            duration_ms: Total duration of workload window in ms.
            arrival_rates: Dictionary of Poisson arrival rates (requests/sec).
                           Defaults to standard mixed benchmark rates.
            dependency_prob: Probability of creating lock/buffer dependency with earlier process.
            
        Returns:
            List of DatabaseProcess objects sorted by arrival time.
        """
        if arrival_rates is None:
            # Default mixed HTAP rates (requests/sec)
            arrival_rates = {
                "OLTP": 60.0,
                "OLAP": 3.0,
                "VECTOR": 15.0,
                "BACKGROUND": 1.0
            }

        processes: List[DatabaseProcess] = []

        for category, rate in arrival_rates.items():
            arrivals = self.dist.sample_poisson_arrivals(rate_lambda=rate, duration_ms=duration_ms)
            for arr_time in arrivals:
                burst = self.dist.sample_burst_time(category)
                priority = self.dist.sample_priority(category)
                op_type = self.dist.sample_op_type(category)
                
                # Sample memory and I/O footprint
                if category == "OLAP":
                    mem = float(self.rng.uniform(256.0, 1024.0))
                    io = float(self.rng.uniform(50.0, 300.0))
                elif category == "BACKGROUND":
                    mem = float(self.rng.uniform(128.0, 512.0))
                    io = float(self.rng.uniform(80.0, 400.0))
                else:
                    mem = float(self.rng.uniform(32.0, 256.0))
                    io = float(self.rng.uniform(5.0, 50.0))
                
                proc = DatabaseProcess(
                    pid=self.next_pid,
                    category=category,
                    arrival_time_ms=arr_time,
                    total_cpu_demand_ms=burst,
                    remaining_cpu_demand_ms=burst,
                    priority=priority,
                    op_type=op_type,
                    memory_mb=mem,
                    io_bandwidth_mb_s=io
                )
                self.next_pid += 1
                processes.append(proc)

        # Sort all processes chronologically by arrival time
        processes.sort(key=lambda p: p.arrival_time_ms)

        # Construct realistic inter-process dependencies (locks / shared buffer pages)
        self._inject_dependencies(processes, dependency_prob=dependency_prob)

        return processes

    def _inject_dependencies(self, processes: List[DatabaseProcess], dependency_prob: float):
        """Creates inter-process relationships (workflow chains and lock waits)."""
        active_pool: List[DatabaseProcess] = []
        
        for i, proc in enumerate(processes):
            # Candidate processes that arrived earlier and might overlap
            candidates = [p for p in active_pool if p.arrival_time_ms + p.total_cpu_demand_ms > proc.arrival_time_ms]
            
            if candidates and self.rng.rand() < dependency_prob:
                target = self.rng.choice(candidates)
                # If target is not already depending on this process (prevent cycles)
                if proc.pid not in target.dependencies:
                    proc.dependencies.append(target.pid)
                    proc.lock_wait_time_ms = float(self.rng.uniform(2.0, 20.0))
            
            # Shared buffer pool pages (for contention graph edges)
            num_pages = self.rng.randint(1, 4)
            proc.shared_buffer_pages = [int(p) for p in self.rng.randint(1, 20, size=num_pages)]
            
            active_pool.append(proc)
            # Prune ancient processes to keep window manageable
            if len(active_pool) > 50:
                active_pool.pop(0)
