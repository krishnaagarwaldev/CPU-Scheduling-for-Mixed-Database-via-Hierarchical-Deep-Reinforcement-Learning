"""Benchmark Runner: Comparative Evaluation of Database Schedulers.

Executes controlled, reproducible experiments comparing the proposed HDRL
architecture against Linux CFS, MAB, DeepRM-Graph, Static, and Priority baselines.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import pandas as pd
import numpy as np

from config.scheduler_config import (
    WorkloadConfig,
    SimulatorConfig,
    ResourceConstraintsConfig,
    MultiObjectiveRewardConfig,
    PaperBenchmarkTargets
)
from simulator.scheduling_environment import DatabaseSchedulingEnv
from controllers.hierarchical_controller import HierarchicalDRLController
from baselines.cfs_scheduler import LinuxCFSScheduler
from baselines.mab_scheduler import MultiArmedBanditScheduler
from baselines.static_scheduler import StaticAllocationScheduler, PriorityFIFOScheduler
from baselines.deeprm_graph import DeepRMGraphScheduler
from evaluation.metrics import calculate_percentile_latencies, calculate_slo_violations, calculate_jains_fairness_index


class BenchmarkRunner:
    def __init__(
        self,
        workload_config: Optional[WorkloadConfig] = None,
        sim_config: Optional[SimulatorConfig] = None,
        duration_ms: float = 5000.0,
        num_runs: int = 3,
        seed: int = 42
    ):
        self.workload_cfg = workload_config or WorkloadConfig()
        self.sim_cfg = sim_config or SimulatorConfig()
        self.duration_ms = duration_ms
        self.num_runs = num_runs
        self.seed = seed

    def evaluate_scheduler(
        self,
        scheduler_name: str,
        env: DatabaseSchedulingEnv,
        controller: Optional[HierarchicalDRLController] = None
    ) -> Dict[str, float]:
        """Runs one full evaluation episode for a specific scheduler."""
        obs, info = env.reset(duration_ms=self.duration_ms)
        
        # Baselines instances
        cfs = LinuxCFSScheduler()
        mab = MultiArmedBanditScheduler(categories=self.workload_cfg.categories)
        static = StaticAllocationScheduler()
        priority_sched = PriorityFIFOScheduler()
        deeprm = DeepRMGraphScheduler(categories=self.workload_cfg.categories)

        terminated = False
        truncated = False
        step_idx = 0
        total_time_taken = 0.0

        macro_action = None
        while not (terminated or truncated):
            start_wall = time.time()
            step_idx += 1

            if scheduler_name == "HDRL":
                if step_idx == 1 or step_idx % 10 == 0:
                    macro_action, process_shares, _ = controller.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(
                    macro_action=macro_action,
                    scheduling_policy="HDRL"
                )
            elif scheduler_name == "Linux CFS":
                if step_idx == 1 or step_idx % 10 == 0:
                    macro_action = cfs.get_macro_action(self.workload_cfg.categories)
                obs, reward, terminated, truncated, info = env.step(
                    macro_action=macro_action,
                    scheduling_policy="CFS"
                )
            elif scheduler_name == "MAB":
                if step_idx == 1 or step_idx % 10 == 0:
                    macro_action = mab.select_arm()
                obs, reward, terminated, truncated, info = env.step(
                    macro_action=macro_action,
                    scheduling_policy="PROPORTIONAL"
                )
                if step_idx % 10 == 0:
                    mab.update_reward(reward)
            elif scheduler_name == "DeepRM-Graph":
                if step_idx == 1 or step_idx % 10 == 0:
                    macro_action = deeprm.select_action(obs["meta_state"])
                obs, reward, terminated, truncated, info = env.step(
                    macro_action=macro_action,
                    scheduling_policy="PROPORTIONAL"
                )
            elif scheduler_name == "Static Allocation":
                if step_idx == 1:
                    macro_action = static.get_macro_action()
                obs, reward, terminated, truncated, info = env.step(
                    macro_action=macro_action,
                    scheduling_policy="PROPORTIONAL"
                )
            elif scheduler_name == "Priority":
                if step_idx == 1:
                    macro_action = static.get_macro_action()
                obs, reward, terminated, truncated, info = env.step(
                    macro_action=macro_action,
                    scheduling_policy="PRIORITY"
                )
            else:
                obs, reward, terminated, truncated, info = env.step()

            total_time_taken += (time.time() - start_wall)

        completed = env.ready_queue.completed_processes
        oltp_lats = calculate_percentile_latencies(completed, category="OLTP")
        slo_stats = calculate_slo_violations(completed, self.workload_cfg.slo_thresholds_ms)

        return {
            "oltp_p99_ms": oltp_lats.get("p99", 0.0),
            "oltp_p50_ms": oltp_lats.get("p50", 0.0),
            "slo_violation_pct": slo_stats["violation_pct"],
            "cpu_utilization_pct": info["cpu_utilization_pct"],
            "fairness_index": info["jains_fairness"],
            "completed_jobs": float(len(completed)),
            "overhead_ms_per_step": (total_time_taken / max(1, step_idx)) * 1000.0
        }

    def run_full_benchmark(
        self,
        schedulers: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs repeated benchmark across all schedulers and aggregates results."""
        schedulers = schedulers or ["HDRL", "Linux CFS", "MAB", "DeepRM-Graph", "Static Allocation", "Priority"]
        
        env = DatabaseSchedulingEnv(
            workload_config=self.workload_cfg,
            sim_config=self.sim_cfg,
            seed=self.seed
        )
        controller = HierarchicalDRLController(
            workload_config=self.workload_cfg,
            controller_config=None,
            meta_state_dim=21
        )

        all_results: List[Dict[str, Any]] = []

        for sched_name in schedulers:
            for run in range(self.num_runs):
                env.seed = self.seed + run * 100
                env.generator.rng = np.random.RandomState(env.seed)
                env.generator.dist.rng = np.random.RandomState(env.seed)

                metrics = self.evaluate_scheduler(sched_name, env, controller)
                metrics["scheduler"] = sched_name
                metrics["run"] = run
                all_results.append(metrics)

        df_raw = pd.DataFrame(all_results)

        # Aggregate mean and std
        agg_rows = []
        for sched_name in schedulers:
            sub_df = df_raw[df_raw["scheduler"] == sched_name]
            agg_rows.append({
                "Scheduler": sched_name,
                "OLTP p99 (ms)": f"{sub_df['oltp_p99_ms'].mean():.1f} ± {sub_df['oltp_p99_ms'].std():.1f}",
                "SLO Viol. (%)": f"{sub_df['slo_violation_pct'].mean():.1f} ± {sub_df['slo_violation_pct'].std():.1f}",
                "CPU Util. (%)": f"{sub_df['cpu_utilization_pct'].mean():.1f} ± {sub_df['cpu_utilization_pct'].std():.1f}",
                "Fairness": f"{sub_df['fairness_index'].mean():.2f} ± {sub_df['fairness_index'].std():.2f}",
                "Throughput (jobs)": f"{sub_df['completed_jobs'].mean():.0f}"
            })

        df_summary = pd.DataFrame(agg_rows)
        return df_summary, df_raw
