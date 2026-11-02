"""Synthetic statistical distributions for database workload generation.

Implements arrival processes, burst time distributions, and diurnal
patterns described in Section 5.3.
"""

from typing import List
import numpy as np


class WorkloadDistribution:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample_poisson_arrivals(self, rate_lambda: float, duration_ms: float) -> List[float]:
        """Generates arrival timestamps using a Poisson process."""
        arrivals = []
        current_time = 0.0
        while current_time < duration_ms:
            # Inter-arrival time in milliseconds
            dt = self.rng.exponential(scale=1000.0 / max(0.1, rate_lambda))
            current_time += dt
            if current_time < duration_ms:
                arrivals.append(current_time)
        return arrivals

    def sample_burst_time(self, category: str) -> float:
        """Samples CPU burst duration (in ms) according to category profile."""
        if category == "OLTP":
            # Short, log-normal bursts centered around 15-30ms
            return float(np.clip(self.rng.lognormal(mean=2.8, sigma=0.5), 5.0, 80.0))
        elif category == "OLAP":
            # Heavy analytical queries (Pareto/log-normal: 500ms - 2500ms)
            return float(np.clip(self.rng.lognormal(mean=6.8, sigma=0.7), 200.0, 4000.0))
        elif category == "VECTOR":
            # Similarity searches (moderate: 50ms - 250ms)
            return float(np.clip(self.rng.lognormal(mean=4.4, sigma=0.6), 25.0, 350.0))
        elif category == "BACKGROUND":
            # Maintenance / Compaction tasks (1000ms - 8000ms)
            return float(np.clip(self.rng.uniform(1000.0, 5000.0), 500.0, 8000.0))
        return 50.0

    def sample_priority(self, category: str) -> int:
        """Sample priority level (1-10) reflecting DB engine scheduling priorities."""
        if category == "OLTP":
            return int(self.rng.randint(7, 11))
        elif category == "VECTOR":
            return int(self.rng.randint(5, 9))
        elif category == "OLAP":
            return int(self.rng.randint(3, 7))
        else:  # BACKGROUND
            return int(self.rng.randint(1, 4))

    def sample_op_type(self, category: str) -> str:
        """Sample database operation type."""
        if category == "OLTP":
            return str(self.rng.choice(["READ", "WRITE", "MIXED"], p=[0.4, 0.4, 0.2]))
        elif category == "OLAP":
            return "READ"
        elif category == "VECTOR":
            return str(self.rng.choice(["READ", "WRITE"], p=[0.8, 0.2]))
        else:
            return "WRITE"
