"""Process representation for database CPU scheduling.

Faithfully implements the database process model specified in Section 4.1.1:
x_i = [r_i; p_i; w_i; h_i] (Equation 9).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DatabaseProcess:
    pid: int
    category: str  # "OLTP", "OLAP", "VECTOR", "BACKGROUND"
    arrival_time_ms: float
    total_cpu_demand_ms: float
    remaining_cpu_demand_ms: float
    priority: int = 5  # 1 to 10
    op_type: str = "MIXED"  # "READ", "WRITE", "MIXED"
    
    # Resource metrics (r_i)
    memory_mb: float = 64.0
    memory_limit_mb: float = 512.0
    io_bandwidth_mb_s: float = 10.0
    network_traffic_kb_s: float = 50.0
    
    # Performance metrics (p_i)
    lock_wait_time_ms: float = 0.0
    buffer_hit_ratio: float = 0.95
    query_throughput_ops: float = 100.0
    
    # Dependencies (workflow / lock waiting)
    dependencies: List[int] = field(default_factory=list)
    shared_buffer_pages: List[int] = field(default_factory=list)
    
    # Scheduling execution state
    accumulated_cpu_time_ms: float = 0.0
    queue_wait_time_ms: float = 0.0
    first_scheduled_time_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None
    vruntime_ms: float = 0.0  # For CFS virtual runtime
    
    # Linux cgroups v2 configuration
    cgroup_weight: int = 100      # cpu.weight in [1, 10000]
    cgroup_max_quota_pct: float = 100.0  # cpu.max percentage
    
    # Historical EWMA tracking for decays beta in {0.9, 0.99, 0.999}
    ewma_cpu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    @property
    def is_completed(self) -> bool:
        return self.remaining_cpu_demand_ms <= 1e-6

    @property
    def turnaround_time_ms(self) -> float:
        if self.completion_time_ms is not None:
            return self.completion_time_ms - self.arrival_time_ms
        return 0.0

    def update_ewma(self, instantaneous_cpu: float):
        decays = np.array([0.9, 0.99, 0.999], dtype=np.float32)
        self.ewma_cpu = decays * self.ewma_cpu + (1.0 - decays) * instantaneous_cpu

    def get_feature_vector(self) -> np.ndarray:
        """Constructs 19-dimensional node feature vector x_i = [r_i; p_i; w_i; h_i] (Eq. 9).
        
        Components:
        - r_i (4): [norm_cpu, mem_ratio, io_mb_s/100, net_traffic/1000]
        - p_i (4): [curr_latency/1000, throughput/1000, lock_wait/100, buffer_hit_ratio]
        - w_i (8): [one_hot_category(4), one_hot_op_type(3), priority/10.0]
        - h_i (3): [ewma_0.9, ewma_0.99, ewma_0.999]
        Total: 4 + 4 + 8 + 3 = 19 dimensions.
        """
        # 1. r_i: Resource utilization metrics (4)
        norm_cpu = min(1.0, self.accumulated_cpu_time_ms / max(1.0, self.total_cpu_demand_ms))
        mem_ratio = min(1.0, self.memory_mb / max(1.0, self.memory_limit_mb))
        norm_io = min(1.0, self.io_bandwidth_mb_s / 500.0)
        norm_net = min(1.0, self.network_traffic_kb_s / 10000.0)
        r_i = [norm_cpu, mem_ratio, norm_io, norm_net]
        
        # 2. p_i: Performance indicators (4)
        curr_latency = (self.completion_time_ms or self.arrival_time_ms) - self.arrival_time_ms
        norm_latency = min(5.0, curr_latency / 1000.0)
        norm_throughput = min(1.0, self.query_throughput_ops / 2000.0)
        norm_lock_wait = min(1.0, self.lock_wait_time_ms / 100.0)
        p_i = [norm_latency, norm_throughput, norm_lock_wait, self.buffer_hit_ratio]
        
        # 3. w_i: Workload attributes (8)
        cat_map = {"OLTP": 0, "OLAP": 1, "VECTOR": 2, "BACKGROUND": 3}
        cat_one_hot = [0.0, 0.0, 0.0, 0.0]
        cat_idx = cat_map.get(self.category, 0)
        cat_one_hot[cat_idx] = 1.0
        
        op_map = {"READ": 0, "WRITE": 1, "MIXED": 2}
        op_one_hot = [0.0, 0.0, 0.0]
        op_idx = op_map.get(self.op_type, 2)
        op_one_hot[op_idx] = 1.0
        
        norm_priority = self.priority / 10.0
        w_i = cat_one_hot + op_one_hot + [norm_priority]
        
        # 4. h_i: Historical EWMA features (3)
        h_i = self.ewma_cpu.tolist()
        
        features = r_i + p_i + w_i + h_i
        return np.array(features, dtype=np.float32)
