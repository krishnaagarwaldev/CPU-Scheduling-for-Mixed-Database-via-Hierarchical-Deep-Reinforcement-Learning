"""Gymnasium-compatible CPU scheduling environment for mixed database workloads.

Faithfully implements the discrete-event environment described in Section 3 & 6:
- Multi-core CPU simulation with context-switch penalties
- Hierarchical state tracking: Meta-state (Eq. 1) & Process-level graphs (Eq. 9-10)
- cgroups v2 resource actuation (cpu.weight, cpu.max)
- Real-time SLO violation and tail latency tracking
- Dynamic graph construction via DynamicDependencyGraph.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from config.scheduler_config import (
    WorkloadConfig,
    ResourceConstraintsConfig,
    SimulatorConfig,
    MultiObjectiveRewardConfig
)
from workloads.process import DatabaseProcess
from workloads.workload_generator import DatabaseWorkloadGenerator
from .cgroups_interface import CgroupsInterface
from .cpu_core import MultiCoreCPU
from .ready_queue import DatabaseReadyQueue
from .dependency_graph import DynamicDependencyGraph


class DatabaseSchedulingEnv:
    def __init__(
        self,
        workload_config: Optional[WorkloadConfig] = None,
        constraints_config: Optional[ResourceConstraintsConfig] = None,
        sim_config: Optional[SimulatorConfig] = None,
        reward_config: Optional[MultiObjectiveRewardConfig] = None,
        seed: int = 42
    ):
        self.workload_cfg = workload_config or WorkloadConfig()
        self.constraints_cfg = constraints_config or ResourceConstraintsConfig()
        self.sim_cfg = sim_config or SimulatorConfig()
        self.reward_cfg = reward_config or MultiObjectiveRewardConfig()
        self.seed = seed

        self.cgroups = CgroupsInterface(is_simulation=True)
        self.cpu = MultiCoreCPU(
            num_cores=self.sim_cfg.num_cores,
            context_switch_ms=self.sim_cfg.context_switch_cost_ms
        )
        self.ready_queue = DatabaseReadyQueue(categories=self.workload_cfg.categories)
        self.dep_graph = DynamicDependencyGraph()
        self.generator = DatabaseWorkloadGenerator(seed=seed)

        # Simulation clock and state
        self.current_time_ms: float = 0.0
        self.step_count: int = 0
        self.pending_arrivals: List[DatabaseProcess] = []
        self.finished_pids: set = set()
        
        # Macro allocation history (Eq. 30 stability)
        self.current_allocations: Dict[str, float] = {
            cat: 1.0 / len(self.workload_cfg.categories) for cat in self.workload_cfg.categories
        }
        self.previous_allocations: Dict[str, float] = self.current_allocations.copy()

    def reset(
        self,
        duration_ms: float = 10000.0,
        arrival_rates: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment with a new generated workload trace."""
        self.current_time_ms = 0.0
        self.step_count = 0
        self.finished_pids.clear()
        self.ready_queue = DatabaseReadyQueue(categories=self.workload_cfg.categories)
        self.cpu = MultiCoreCPU(
            num_cores=self.sim_cfg.num_cores,
            context_switch_ms=self.sim_cfg.context_switch_cost_ms
        )
        self.dep_graph = DynamicDependencyGraph()

        self.current_allocations = {
            cat: 1.0 / len(self.workload_cfg.categories) for cat in self.workload_cfg.categories
        }
        self.previous_allocations = self.current_allocations.copy()

        # Generate fresh workload
        self.episode_duration_ms: float = duration_ms
        self.pending_arrivals = self.generator.generate_mixed_workload(
            duration_ms=duration_ms,
            arrival_rates=arrival_rates
        )

        self._admit_new_arrivals()
        obs = self._get_observation()
        info = self._get_info_metrics()
        return obs, info

    def _admit_new_arrivals(self):
        """Moves arrived processes from pending list into ready queues."""
        while self.pending_arrivals and self.pending_arrivals[0].arrival_time_ms <= self.current_time_ms:
            proc = self.pending_arrivals.pop(0)
            self.ready_queue.add_process(proc)

    def apply_macro_action(self, budget_allocation: Dict[str, float]):
        """Applies high-level budget distribution W = [w_oltp, w_olap, w_vec, w_bg] (Eq. 2).
        
        Enforces:
        - Conservation: sum(alpha_c) == 1.0 (Eq. 28)
        - Starvation prevention: alpha_c >= alpha_min = 0.05 (Eq. 29)
        - Stability: |alpha_c - alpha_prev| <= delta_max = 0.20 (Eq. 30)
        """
        self.previous_allocations = self.current_allocations.copy()
        validated: Dict[str, float] = {}

        for cat in self.workload_cfg.categories:
            raw_val = budget_allocation.get(cat, self.current_allocations[cat])
            # Enforce delta_max stability (Eq. 30)
            delta = raw_val - self.previous_allocations[cat]
            clamped_delta = np.clip(delta, -self.constraints_cfg.delta_max, self.constraints_cfg.delta_max)
            # Enforce alpha_min starvation prevention (Eq. 29)
            val = max(self.constraints_cfg.alpha_min, self.previous_allocations[cat] + clamped_delta)
            validated[cat] = float(val)

        # Normalize to satisfy sum == 1.0 (Eq. 28)
        total = sum(validated.values())
        if total > 0:
            for cat in validated:
                validated[cat] /= total
        
        self.current_allocations = validated

        # Actuate via cgroups v2
        for cat, proportion in self.current_allocations.items():
            cgroup_weight = int(np.clip(proportion * 10000, 1, 10000))
            self.cgroups.set_cpu_weight(cat, cgroup_weight)

    def step(
        self,
        macro_action: Optional[Dict[str, float]] = None,
        scheduling_policy: str = "HDRL"
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advances the simulation by one discrete time quantum step.
        
        Returns:
            observation: Hierarchical system observation dict
            reward: Composite multi-objective scalar reward
            terminated: Whether all processes completed
            truncated: Whether simulation time limit reached
            info: Comprehensive operational metrics dictionary
        """
        self.step_count += 1
        
        if macro_action is not None:
            self.apply_macro_action(macro_action)

        # Advance simulation clock by quantum slice
        quantum_ms = self.sim_cfg.time_slice_quantum_ms
        self.current_time_ms += quantum_ms
        self._admit_new_arrivals()

        # Dispatch ready processes across CPU cores according to cgroup weights
        categories = self.workload_cfg.categories
        shares = self.cgroups.get_effective_shares()
        
        # Sort categories by allocation weight so highest budgeted categories grab cores first
        sorted_categories = sorted(categories, key=lambda c: shares.get(c, 0.0), reverse=True)

        for core in self.cpu.cores:
            if core.is_idle:
                for cat in sorted_categories:
                    proc = self.ready_queue.select_next_process(
                        category=cat,
                        scheduling_policy=scheduling_policy,
                        finished_pids=self.finished_pids
                    )
                    if proc is not None:
                        core.execute_slice(proc, slice_ms=quantum_ms, current_time_ms=self.current_time_ms)
                        if proc.is_completed:
                            self.finished_pids.add(proc.pid)
                            self.ready_queue.record_completed(proc)
                        else:
                            # Preempt and return to ready queue
                            self.ready_queue.add_process(proc)
                        break

        # Check termination condition
        all_completed = (len(self.pending_arrivals) == 0) and self.ready_queue.is_empty() and all(c.is_idle for c in self.cpu.cores)
        time_limit_reached = (self.current_time_ms >= getattr(self, "episode_duration_ms", 60000.0))

        obs = self._get_observation()
        info = self._get_info_metrics()
        reward = self._calculate_reward(info)

        return obs, reward, all_completed, time_limit_reached, info

    def _get_observation(self) -> Dict[str, Any]:
        """Constructs meta-controller state s_h (Eq. 1) and process dependency graph."""
        active = self.ready_queue.get_active_processes()
        node_feats, edge_index, edge_feats = self.dep_graph.build_graph(active)

        # Meta-state vector components (Eq. 1):
        # s_h = {m_OLTP, m_OLAP, m_vec, m_bg, u_global, v_SLO}
        category_metrics = []
        slo_violations_vector = []
        
        completed = self.ready_queue.completed_processes
        for cat in self.workload_cfg.categories:
            cat_active = [p for p in active if p.category == cat]
            cat_done = [p for p in completed if p.category == cat]
            
            queue_len = float(len(cat_active))
            avg_burst = float(np.mean([p.total_cpu_demand_ms for p in cat_active])) if cat_active else 0.0
            throughput = float(len(cat_done)) / max(0.1, self.current_time_ms / 1000.0)
            category_metrics.extend([queue_len, avg_burst, throughput])

            # SLO violation statistics
            slo_target = self.workload_cfg.slo_thresholds_ms.get(cat, 100.0)
            violations = sum(1 for p in cat_done if p.turnaround_time_ms > slo_target)
            viol_rate = violations / max(1, len(cat_done))
            slo_violations_vector.append(viol_rate)

        u_global = self.cpu.get_average_utilization(self.current_time_ms)
        meta_state_vector = np.array(
            category_metrics + [u_global] + slo_violations_vector + list(self.current_allocations.values()),
            dtype=np.float32
        )

        return {
            "meta_state": meta_state_vector,
            "node_features": node_feats,
            "edge_index": edge_index,
            "edge_features": edge_feats,
            "active_pids": [p.pid for p in active]
        }

    def _get_info_metrics(self) -> Dict[str, Any]:
        """Computes comprehensive evaluation metrics (Section 5 & 6)."""
        completed = self.ready_queue.completed_processes
        u_global = self.cpu.get_average_utilization(self.current_time_ms)

        latencies_by_cat: Dict[str, List[float]] = {cat: [] for cat in self.workload_cfg.categories}
        slo_violations_by_cat: Dict[str, int] = {cat: 0 for cat in self.workload_cfg.categories}

        for p in completed:
            lat = p.turnaround_time_ms
            latencies_by_cat[p.category].append(lat)
            if lat > self.workload_cfg.slo_thresholds_ms.get(p.category, 100.0):
                slo_violations_by_cat[p.category] += 1

        p99_by_cat: Dict[str, float] = {}
        mean_by_cat: Dict[str, float] = {}
        for cat, lats in latencies_by_cat.items():
            if lats:
                p99_by_cat[cat] = float(np.percentile(lats, 99))
                mean_by_cat[cat] = float(np.mean(lats))
            else:
                p99_by_cat[cat] = 0.0
                mean_by_cat[cat] = 0.0

        total_done = len(completed)
        total_violations = sum(slo_violations_by_cat.values())
        overall_slo_viol_pct = (total_violations / total_done * 100.0) if total_done > 0 else 0.0

        # Jain's Fairness Index across category resource allocations
        alloc_vals = np.array(list(self.current_allocations.values()), dtype=np.float64)
        jains_fairness = float((np.sum(alloc_vals) ** 2) / (len(alloc_vals) * np.sum(alloc_vals ** 2) + 1e-8))

        return {
            "current_time_ms": self.current_time_ms,
            "completed_count": total_done,
            "cpu_utilization_pct": u_global * 100.0,
            "overall_slo_violation_pct": overall_slo_viol_pct,
            "oltp_p99_ms": p99_by_cat.get("OLTP", 0.0),
            "oltp_mean_ms": mean_by_cat.get("OLTP", 0.0),
            "p99_by_cat": p99_by_cat,
            "mean_by_cat": mean_by_cat,
            "jains_fairness": jains_fairness,
            "allocations": self.current_allocations.copy()
        }

    def _calculate_reward(self, info: Dict[str, Any]) -> float:
        """Faithfully evaluates Eq. 27, 32, 33 from the paper."""
        # 1. SLO penalty r_SLO (Eq. 32): quadratic penalty
        r_slo = 0.0
        completed = self.ready_queue.completed_processes
        if completed:
            for cat in self.workload_cfg.categories:
                lambda_c = self.workload_cfg.penalty_weights.get(cat, 1.0)
                cat_procs = [p for p in completed if p.category == cat]
                if cat_procs:
                    slo_target = self.workload_cfg.slo_thresholds_ms.get(cat, 100.0)
                    violations = [max(0.0, (p.turnaround_time_ms - slo_target) / slo_target) for p in cat_procs]
                    mean_quad_violation = float(np.mean([v ** 2 for v in violations]))
                    r_slo -= lambda_c * mean_quad_violation

        # 2. Resource utilization reward r_util (Eq. 33): piecewise formulation
        u_t = info["cpu_utilization_pct"] / 100.0
        u_target = self.constraints_cfg.u_target
        if u_t < u_target:
            r_util = self.constraints_cfg.beta_1 * (u_t - u_target)
        else:
            r_util = -self.constraints_cfg.beta_2 * ((u_t - u_target) ** 2)

        # 3. Symmetric Fairness r_fair (Jain's index)
        r_fair = info["jains_fairness"]

        # 4. Allocation Stability penalty r_stable (Eq. 30)
        stability_loss = sum(
            (self.current_allocations[cat] - self.previous_allocations[cat]) ** 2
            for cat in self.workload_cfg.categories
        )
        r_stable = -float(stability_loss)

        # Composite reward (Eq. 27 with Pareto weights w1=0.4, w2=0.3, w3=0.2, w4=0.1)
        r_total = (
            self.reward_cfg.w1_slo * r_slo +
            self.reward_cfg.w2_util * r_util +
            self.reward_cfg.w3_fair * r_fair +
            self.reward_cfg.w4_stable * r_stable
        )
        return float(r_total)
