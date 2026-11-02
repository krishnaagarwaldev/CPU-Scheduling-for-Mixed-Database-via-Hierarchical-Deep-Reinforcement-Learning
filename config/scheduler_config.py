"""Configuration and Hyperparameters for HDRL Database CPU Scheduler.

Faithfully implements parameter values and constraints from:
'Self-Adapting CPU Scheduling for Mixed Database Workloads via
Hierarchical Deep Reinforcement Learning' (Xing, Wang, Liu - MDPI Symmetry 2025).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class WorkloadConfig:
    """Workload category definitions and SLO targets."""
    categories: List[str] = field(default_factory=lambda: [
        "OLTP",
        "OLAP",
        "VECTOR",
        "BACKGROUND"
    ])
    
    # Category penalty weights lambda_c (Eq. 32 in paper)
    penalty_weights: Dict[str, float] = field(default_factory=lambda: {
        "OLTP": 10.0,       # High latency sensitivity
        "OLAP": 3.0,        # Analytical throughput
        "VECTOR": 5.0,      # Vector similarity search
        "BACKGROUND": 1.0   # Maintenance batch
    })
    
    # Target SLO thresholds (in milliseconds)
    # Calibrated to the paper's steady-state benchmarks (CFS 156.3ms vs HDRL 87.4ms)
    slo_thresholds_ms: Dict[str, float] = field(default_factory=lambda: {
        "OLTP": 100.0,
        "OLAP": 2000.0,
        "VECTOR": 250.0,
        "BACKGROUND": 10000.0
    })
    
    # Priority ranges assigned by DB engine (1 to 10)
    priority_range: Tuple[int, int] = (1, 10)


@dataclass
class ResourceConstraintsConfig:
    """Operational constraints enforcing system stability (Section 5.1.2)."""
    # Eq. 28: sum(alpha_c) == 1.0
    # Eq. 29: Starvation prevention threshold
    alpha_min: float = 0.05
    
    # Eq. 30: Maximum allowed budget change between consecutive steps
    delta_max: float = 0.20
    
    # Target CPU utilization for piecewise reward (Eq. 33)
    u_target: float = 0.85
    beta_1: float = 2.0   # Linear incentive below target
    beta_2: float = 5.0   # Quadratic penalty above target


@dataclass
class MultiObjectiveRewardConfig:
    """Pareto knee-point reward weights (Section 5.1.3, Figure 2)."""
    # Eq. 27: r_t^h = w1*r_SLO + w2*r_util + w3*r_fair + w4*r_stable
    w1_slo: float = 0.40       # SLO Compliance
    w2_util: float = 0.30      # CPU Utilization Efficiency
    w3_fair: float = 0.20      # Symmetric Fairness (Jain's Index)
    w4_stable: float = 0.10    # Allocation Stability Penalty


@dataclass
class GNNConfig:
    """Graph Neural Network architecture parameters (Section 4.1 & 4.2)."""
    # Node feature dimension: r_i(4) + p_i(4) + w_i(8) + h_i(3) = 19
    node_feat_dim: int = 19
    # Edge feature dimension: comm(1) + cont(1) + dep(1) + tau(1) = 4
    edge_feat_dim: int = 4
    hidden_dim: int = 64
    num_layers: int = 2
    leaky_relu_slope: float = 0.2
    ewma_decays: Tuple[float, float, float] = (0.9, 0.99, 0.999)


@dataclass
class ControllerConfig:
    """Hierarchical controller parameters (Section 3.2 & 5.2)."""
    # Time scales
    macro_interval_steps: int = 20   # T_h (Macro decision interval)
    micro_interval_steps: int = 2    # T_l (Micro sub-controller interval)
    
    # Meta-controller (PPO)
    meta_lr: float = 3e-4
    ppo_clip_eps: float = 0.2
    gae_lambda: float = 0.95
    gamma_meta: float = 0.99
    
    # Sub-controller (SAC)
    sub_lr: float = 3e-4
    gamma_sub: float = 0.99
    tau: float = 0.005               # Target network update rate
    init_temperature: float = 0.2
    
    # Curiosity (ICM)
    use_curiosity: bool = True
    eta_0: float = 0.1
    curiosity_decay: float = 0.9999
    
    # Meta-Learning (MAML)
    use_meta_learning: bool = True
    inner_lr: float = 0.01
    meta_lr_outer: float = 1e-4


@dataclass
class SimulatorConfig:
    """Discrete-event CPU simulator and hardware parameters."""
    num_cores: int = 8
    context_switch_cost_ms: float = 0.05
    time_slice_quantum_ms: float = 5.0
    cgroup_default_weight: int = 100
    cgroup_min_weight: int = 1
    cgroup_max_weight: int = 10000
    cgroup_period_us: int = 100000


@dataclass
class PaperBenchmarkTargets:
    """Published steady-state results from Table 1 of Xing et al. 2025."""
    published_results: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "Linux CFS": {
            "p99_latency_ms": (156.3, 8.2),
            "slo_violation_pct": (12.4, 1.1),
            "cpu_util_pct": (67.8, 2.3),
            "fairness_index": (0.72, 0.04)
        },
        "Oracle RM": {
            "p99_latency_ms": (124.7, 6.5),
            "slo_violation_pct": (8.7, 0.8),
            "cpu_util_pct": (71.2, 1.9),
            "fairness_index": (0.83, 0.03)
        },
        "CDBTune": {
            "p99_latency_ms": (132.1, 7.1),
            "slo_violation_pct": (9.3, 0.9),
            "cpu_util_pct": (69.5, 2.1),
            "fairness_index": (0.78, 0.03)
        },
        "MAB": {
            "p99_latency_ms": (118.6, 5.9),
            "slo_violation_pct": (7.2, 0.7),
            "cpu_util_pct": (73.6, 1.7),
            "fairness_index": (0.81, 0.03)
        },
        "DeepRM-Graph": {
            "p99_latency_ms": (108.2, 6.1),
            "slo_violation_pct": (6.3, 0.6),
            "cpu_util_pct": (76.8, 1.8),
            "fairness_index": (0.84, 0.03)
        },
        "HDRL (Ours)": {
            "p99_latency_ms": (87.4, 4.3),
            "slo_violation_pct": (4.0, 0.4),
            "cpu_util_pct": (86.5, 1.2),
            "fairness_index": (0.91, 0.02)
        }
    })
