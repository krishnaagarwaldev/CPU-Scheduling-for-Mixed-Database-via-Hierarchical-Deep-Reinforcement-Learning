"""Interactive Streamlit Dashboard for Self-Adapting Database CPU Scheduler.

Faithfully connects to the actual simulation and scheduling environment:
- Algorithm selection (HDRL, Linux CFS, MAB, DeepRM-Graph, Static, Priority)
- Workload parameter configuration (arrival rates, simulation duration, core count)
- Live experiment execution and progress tracking
- Interactive visualization of scheduling decisions, CPU allocations, and tail latency metrics
- Comparative benchmark visualization and CSV export.
"""

import sys
import os
import time

# Ensure repository root is in sys.path so modules (config, simulator, etc.) can be imported on Streamlit Cloud
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Bootstraps local cache for PyTorch/Matplotlib
import config
from config.scheduler_config import (
    WorkloadConfig,
    SimulatorConfig,
    ResourceConstraintsConfig,
    MultiObjectiveRewardConfig,
    PaperBenchmarkTargets
)
from simulator.scheduling_environment import DatabaseSchedulingEnv
from controllers.hierarchical_controller import HierarchicalDRLController
from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.metrics import calculate_percentile_latencies, calculate_slo_violations

st.set_page_config(
    page_title="Self-Adapting CPU Scheduler (HDRL)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("⚡ Self-Adapting CPU Scheduling for Mixed Database Workloads")
st.markdown("""
**Hierarchical Deep Reinforcement Learning with Graph Neural Networks**  
*Implementation of Xing, Wang, Liu (MDPI Symmetry 2025, Article 1109)*
""")

# Sidebar controls
st.sidebar.header("🛠️ Experiment Configuration")

selected_mode = st.sidebar.radio(
    "Select Mode:",
    ["Single Scheduler Simulation", "Comparative Benchmark Suite", "Published Results (Table 1)"]
)

sim_duration_s = st.sidebar.slider("Simulation Duration (seconds)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
num_cores = st.sidebar.slider("CPU Cores", min_value=2, max_value=16, value=8, step=2)

st.sidebar.subheader("Workload Arrival Rates (req/sec)")
oltp_rate = st.sidebar.number_input("OLTP Rate (TPC-C)", min_value=10.0, max_value=500.0, value=60.0, step=10.0)
olap_rate = st.sidebar.number_input("OLAP Rate (TPC-H)", min_value=0.5, max_value=30.0, value=3.0, step=0.5)
vec_rate = st.sidebar.number_input("Vector Search Rate (ANN)", min_value=2.0, max_value=100.0, value=15.0, step=5.0)
bg_rate = st.sidebar.number_input("Background Maintenance", min_value=0.2, max_value=10.0, value=1.0, step=0.2)

arrival_rates = {
    "OLTP": oltp_rate,
    "OLAP": olap_rate,
    "VECTOR": vec_rate,
    "BACKGROUND": bg_rate
}

# MODE 1: Single Scheduler Simulation
if selected_mode == "Single Scheduler Simulation":
    st.subheader("Interactive Single-Run Execution")
    
    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        scheduler_choice = st.selectbox(
            "Select Scheduling Algorithm:",
            ["HDRL (Proposed)", "Linux CFS Baseline", "MAB (Multi-Armed Bandit)", "DeepRM-Graph", "Static Allocation", "Strict Priority"]
        )
        run_btn = st.button("🚀 Start Simulation Run", use_container_width=True)

    with col_info:
        st.info(f"""
        **Running on:** {num_cores} Virtual Cores | **Time Window:** {sim_duration_s} seconds  
        **Workload Mix:** OLTP: {oltp_rate}/s | OLAP: {olap_rate}/s | Vector: {vec_rate}/s | Background: {bg_rate}/s
        """)

    if run_btn:
        with st.spinner(f"Simulating {scheduler_choice}..."):
            sim_cfg = SimulatorConfig(num_cores=num_cores)
            env = DatabaseSchedulingEnv(sim_config=sim_cfg)
            duration_ms = sim_duration_s * 1000.0
            
            # Map choice to runner name
            name_map = {
                "HDRL (Proposed)": "HDRL",
                "Linux CFS Baseline": "Linux CFS",
                "MAB (Multi-Armed Bandit)": "MAB",
                "DeepRM-Graph": "DeepRM-Graph",
                "Static Allocation": "Static Allocation",
                "Strict Priority": "Priority"
            }
            sched_internal_name = name_map[scheduler_choice]
            
            controller = HierarchicalDRLController(meta_state_dim=21)
            runner = BenchmarkRunner(sim_config=sim_cfg, duration_ms=duration_ms)
            
            metrics = runner.evaluate_scheduler(sched_internal_name, env, controller)

        st.success("Simulation Run Completed Successfully!")

        # Metric Tiles
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("OLTP p99 Latency", f"{metrics['oltp_p99_ms']:.1f} ms")
        m2.metric("SLO Violations", f"{metrics['slo_violation_pct']:.1f} %")
        m3.metric("CPU Utilization", f"{metrics['cpu_utilization_pct']:.1f} %")
        m4.metric("Jain's Fairness Index", f"{metrics['fairness_index']:.2f}")
        m5.metric("Completed Queries", f"{metrics['completed_jobs']:.0f}")

        # Visualizations
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Final CPU Allocation Distribution Across Categories")
            allocs = env.current_allocations
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(allocs.values(), labels=allocs.keys(), autopct="%1.1f%%", startangle=140, colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
            ax.axis("equal")
            st.pyplot(fig)

        with c2:
            st.markdown("#### Completed Process Turnaround Distribution")
            completed = env.ready_queue.completed_processes
            if completed:
                df_procs = pd.DataFrame([{
                    "PID": p.pid,
                    "Category": p.category,
                    "Turnaround (ms)": p.turnaround_time_ms,
                    "CPU Demand (ms)": p.total_cpu_demand_ms
                } for p in completed[:200]])
                st.dataframe(df_procs, height=250, use_container_width=True)
                
                # CSV Export
                csv_bytes = df_procs.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export Completed Processes to CSV", data=csv_bytes, file_name=f"{sched_internal_name}_processes.csv", mime="text/csv")


# MODE 2: Comparative Benchmark Suite
elif selected_mode == "Comparative Benchmark Suite":
    st.subheader("Comparative Benchmark: Proposed HDRL vs All Baselines")
    st.markdown("Executes multiple repeated runs across identical workload distributions to calculate mean and standard deviation.")
    
    num_runs = st.slider("Number of Repeated Runs", min_value=1, max_value=5, value=2)
    bench_btn = st.button("📊 Run Full Comparative Benchmark", use_container_width=True)

    if bench_btn:
        with st.spinner("Executing comparative experiments..."):
            runner = BenchmarkRunner(
                duration_ms=sim_duration_s * 1000.0,
                num_runs=num_runs,
                seed=42
            )
            df_summary, df_raw = runner.run_full_benchmark()

        st.subheader("Empirical Benchmark Results (Simulated Testbed)")
        st.dataframe(df_summary, use_container_width=True)

        # Plot comparative bar charts
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        means = df_raw.groupby("scheduler").mean(numeric_only=True)
        
        # p99
        axes[0].bar(means.index, means["oltp_p99_ms"], color="#1f77b4", alpha=0.85)
        axes[0].set_title("OLTP p99 Latency (ms)", fontweight="bold")
        axes[0].set_ylabel("ms")
        axes[0].tick_params(axis="x", rotation=25)

        # SLO Violations
        axes[1].bar(means.index, means["slo_violation_pct"], color="#d62728", alpha=0.85)
        axes[1].set_title("SLO Violation Rate (%)", fontweight="bold")
        axes[1].set_ylabel("%")
        axes[1].tick_params(axis="x", rotation=25)

        # CPU Util
        axes[2].bar(means.index, means["cpu_utilization_pct"], color="#2ca02c", alpha=0.85)
        axes[2].set_title("CPU Utilization (%)", fontweight="bold")
        axes[2].set_ylabel("%")
        axes[2].tick_params(axis="x", rotation=25)

        plt.tight_layout()
        st.pyplot(fig)

        # Download button
        csv_data = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Benchmark Summary to CSV", data=csv_data, file_name="benchmark_summary.csv", mime="text/csv")


# MODE 3: Published Results
else:
    st.subheader("Published Results (Table 1, Xing et al. MDPI 2025)")
    st.markdown("""
    Results reported in the research paper evaluated on a 10-hour benchmark run across 10,000 concurrent database processes:
    """)
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
    st.table(df_pub)

    st.markdown("""
    ### Key Observations from Research Paper:
    - **OLTP p99 Latency:** HDRL achieved **87.4 ms** (a **44.1% reduction** vs Linux CFS at 156.3 ms).
    - **SLO Violations:** Decreased from **12.4% (CFS)** down to **4.0% (HDRL)** (a **67.7% reduction**).
    - **CPU Utilization:** Increased from **67.8% (CFS)** up to **86.5% (HDRL)** (a **27.6% boost**).
    - **Scheduling Overhead:** Kept under **3%**, scaling stably to 10,000 concurrent processes.
    """)
