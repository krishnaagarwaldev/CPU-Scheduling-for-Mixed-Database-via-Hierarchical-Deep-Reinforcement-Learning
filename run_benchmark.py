"""Comparative Evaluation Experiment Runner for Schedulers.

Executes reproducible evaluation benchmark comparing:
1. HDRL (Hierarchical Deep Reinforcement Learning with GNN)
2. Linux CFS (Completely Fair Scheduler)
3. MAB (Multi-Armed Bandit)
4. DeepRM-Graph (Flat DRL baseline)
5. Static Allocation
6. Strict Priority

Exports CSV summary and generates comparative performance charts.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import config  # bootstraps cache
from evaluation.benchmark_runner import BenchmarkRunner
from config.scheduler_config import PaperBenchmarkTargets


def run_experiments(num_runs: int = 2, duration_ms: float = 2000.0, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(f"RUNNING COMPARATIVE BENCHMARK EXPERIMENTS ({num_runs} runs, {duration_ms}ms window)")
    print("=" * 70)

    runner = BenchmarkRunner(duration_ms=duration_ms, num_runs=num_runs, seed=42)
    df_summary, df_raw = runner.run_full_benchmark()

    # Save CSV outputs
    raw_path = os.path.join(output_dir, "raw_benchmark_runs.csv")
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_raw.to_csv(raw_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENTAL EVALUATION RESULTS (SIMULATED TESTBED):")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("PUBLISHED RESULTS FROM TABLE 1 (Xing et al. 2025, 10-hour testbed):")
    print("=" * 70)
    targets = PaperBenchmarkTargets().published_results
    pub_rows = []
    for s_name, m_dict in targets.items():
        pub_rows.append({
            "Scheduler": s_name,
            "OLTP p99 (ms)": f"{m_dict['p99_latency_ms'][0]} ± {m_dict['p99_latency_ms'][1]}",
            "SLO Viol. (%)": f"{m_dict['slo_violation_pct'][0]} ± {m_dict['slo_violation_pct'][1]}",
            "CPU Util. (%)": f"{m_dict['cpu_util_pct'][0]} ± {m_dict['cpu_util_pct'][1]}",
            "Fairness": f"{m_dict['fairness_index'][0]} ± {m_dict['fairness_index'][1]}"
        })
    df_pub = pd.DataFrame(pub_rows)
    print(df_pub.to_string(index=False))

    # Generate Comparative Bar Charts
    plot_comparisons(df_raw, output_dir)
    print(f"\nBenchmark artifacts and plots successfully saved to: {output_dir}/")
    return df_summary, df_raw


def plot_comparisons(df_raw: pd.DataFrame, output_dir: str):
    schedulers = df_raw["scheduler"].unique()
    means = df_raw.groupby("scheduler").mean(numeric_only=True)
    stds = df_raw.groupby("scheduler").std(numeric_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparative Evaluation Across Schedulers (Simulation Testbed)", fontsize=16, fontweight="bold")

    # 1. OLTP p99 Latency
    ax = axes[0, 0]
    ax.bar(means.index, means["oltp_p99_ms"], yerr=stds["oltp_p99_ms"], capsize=5, color="#1f77b4", alpha=0.85)
    ax.set_title("OLTP p99 Latency (ms) - Lower is Better", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=25)

    # 2. SLO Violations
    ax = axes[0, 1]
    ax.bar(means.index, means["slo_violation_pct"], yerr=stds["slo_violation_pct"], capsize=5, color="#d62728", alpha=0.85)
    ax.set_title("SLO Violation Rate (%) - Lower is Better", fontweight="bold")
    ax.set_ylabel("SLO Violations (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=25)

    # 3. CPU Utilization
    ax = axes[1, 0]
    ax.bar(means.index, means["cpu_utilization_pct"], yerr=stds["cpu_utilization_pct"], capsize=5, color="#2ca02c", alpha=0.85)
    ax.set_title("CPU Utilization (%) - Higher is Better", fontweight="bold")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=25)

    # 4. Jain's Fairness Index
    ax = axes[1, 1]
    ax.bar(means.index, means["fairness_index"], yerr=stds["fairness_index"], capsize=5, color="#9467bd", alpha=0.85)
    ax.set_title("Jain's Fairness Index - Higher is Better", fontweight="bold")
    ax.set_ylabel("Fairness (0 to 1)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "scheduler_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    run_experiments()
