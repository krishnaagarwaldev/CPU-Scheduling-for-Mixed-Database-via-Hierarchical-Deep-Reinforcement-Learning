# Self-Adapting CPU Scheduling for Mixed Database Workloads via Hierarchical Deep Reinforcement Learning

> **A Faithful Research Reproduction of:**  
> Suchuan Xing, Yihan Wang, Wenhe Liu. *"Self-Adapting CPU Scheduling for Mixed Database Workloads via Hierarchical Deep Reinforcement Learning"*,  
> **Symmetry** (MDPI), Vol. 17, Issue 7, Article 1109 (July 10, 2025).  
> **DOI:** [10.3390/sym17071109](https://doi.org/10.3390/sym17071109)

---

## 1. Project Overview

Live app - https://cpu-scheduling-for-database-via-reinforcement-learning.streamlit.app/

In modern multi-tenant and Hybrid Transactional/Analytical Processing (HTAP) database systems, diverse workloads compete simultaneously for computational resources:
- **OLTP (Online Transaction Processing):** Short, bursty, latency-sensitive transactions (TPC-C / Sysbench) demanding strict $p_{99}$ latency guarantees.
- **OLAP (Online Analytical Processing):** Long-running, multi-threaded analytical queries (TPC-H) demanding high scan throughput.
- **Vector Processing:** High-intensity SIMD/AVX vector similarity searches (ANN-Benchmarks).
- **Background Maintenance:** Periodic, batch tasks (PostgreSQL autovacuum, LSM compaction).

Traditional operating system schedulers (such as Linux Completely Fair Scheduler / CFS) are agnostic to database query semantics, transaction deadlines, and lock dependencies, resulting in severe tail-latency violations.

This project implements the paper's **Symmetric Two-Tier Hierarchical Deep Reinforcement Learning (HDRL)** framework:
1. **Tier-1 Meta-Controller (PPO):** Operates at macro intervals $T_h$ to strategically allocate CPU quotas $\mathbf{W} = (\alpha_{\text{OLTP}}, \alpha_{\text{OLAP}}, \alpha_{\text{vec}}, \alpha_{\text{bg}})$ across workload categories while satisfying simplex conservation ($\sum \alpha_c = 1$), starvation prevention ($\alpha_c \ge 0.05$), and stability constraints ($|\Delta \alpha| \le 0.20$).
2. **Tier-2 Sub-Controllers (SAC):** Dedicated continuous Soft Actor-Critic controllers operating within allocated category budgets to optimize process-level shares (`cpu.weight`) and quota ceilings (`cpu.max`).
3. **Graph Neural Network (MPNN / GAT Attention):** Dynamically constructs an inter-process dependency graph encoding lock contention, shared buffer page conflicts, and pipeline workflow DAGs, generating rich structural embeddings.
4. **Multi-Objective Reward Engine:** Evaluates the exact Pareto knee-point formulation $(w_1=0.4, w_2=0.3, w_3=0.2, w_4=0.1)$ balancing SLO adherence, CPU utilization efficiency, Jain's fairness index, and allocation stability.
5. **Adaptation Mechanisms:** Integrates an Intrinsic Curiosity Module (ICM) forward dynamics model and Model-Agnostic Meta-Learning (MAML) for rapid adaptation.
6. **Linux `cgroups v2` Interface:** Actuates scheduling decisions through native Linux kernel control parameters (`cpu.weight` and `cpu.max`) with an automated discrete-event simulator fallback.

---

## 2. Directory Structure

```
├── config/
│   ├── __init__.py               # Cache environment bootstrapping & config exports
│   └── scheduler_config.py       # Paper hyperparameters, constraints, and published targets
├── workloads/
│   ├── __init__.py
│   ├── process.py                # DatabaseProcess model & 19-dim node feature vector (Eq. 9)
│   ├── synthetic_distribution.py # Poisson arrivals, log-normal bursts, priorities
│   └── workload_generator.py     # Mixed database workload generator with lock/page DAGs
├── simulator/
│   ├── __init__.py
│   ├── cgroups_interface.py      # Linux cgroups v2 abstraction (cpu.weight, cpu.max)
│   ├── cpu_core.py               # Discrete-event multi-core CPU execution & context switching
│   ├── ready_queue.py            # Category-aware queues, vruntime tracking, dependency checks
│   ├── dependency_graph.py       # Dynamic Graph G_t = (V_t, E_t) builder (Eq. 9 & 10)
│   └── scheduling_environment.py # Gymnasium-compatible discrete event scheduling environment
├── gnn/
│   ├── __init__.py
│   └── mpnn_attention.py         # K-layer MPNN with Attention-weighted aggregation (Eq. 11-14)
├── controllers/
│   ├── __init__.py
│   ├── meta_controller.py        # PPO Macro-Allocator with simplex & delta constraints (Eq. 28-30)
│   ├── sub_controller.py         # SAC continuous micro-allocator (Eq. 3, 35)
│   ├── curiosity_module.py       # Intrinsic Curiosity Module (ICM) dynamics model (Eq. 39-42)
│   └── hierarchical_controller.py# End-to-end hierarchical coordinator & checkpoint I/O
├── adaptation/
│   ├── __init__.py
│   ├── meta_learning.py          # MAML inner/outer loop meta-adaptation (Eq. 36-38)
│   └── prioritized_replay.py     # Prioritized Experience Replay with SLO bonus (Eq. 43-44)
├── baselines/
│   ├── __init__.py
│   ├── cfs_scheduler.py          # Linux CFS baseline (virtual runtime proportional sharing)
│   ├── mab_scheduler.py          # Multi-Armed Bandit (UCB1) baseline
│   ├── static_scheduler.py       # Static partitioning & Strict Priority baselines
│   └── deeprm_graph.py           # DeepRM-Graph baseline (flat DRL)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # p99 latency, SLO violation rate, CPU util, Jain's fairness
│   └── benchmark_runner.py       # Automated multi-run comparative benchmark harness
├── tests/
│   ├── test_workload.py          # Workload generation and feature dimension tests
│   ├── test_simulator.py         # Discrete CPU core, context switch, and cgroup tests
│   ├── test_gnn.py               # GNN message passing and attention tensor shape tests
│   └── test_controllers.py       # PPO, SAC, Curiosity, and Fairness tests
├── dashboard/
│   └── app.py                    # Interactive Streamlit UI dashboard
├── results/                      # Generated benchmark CSVs and comparative plots
├── train_verifier.py             # Pipeline verification dry-run and lightweight trainer
├── run_all_tests.py              # Automated test suite executor
├── run_benchmark.py              # Comparative benchmark experiment runner
└── requirements.txt              # Project dependencies
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.13)
* PyTorch 2.0+
* NumPy, Pandas, Matplotlib, Streamlit, Scikit-learn, NetworkX

```bash
# Clone the repository and navigate into the workspace
git clone <repo-url>
cd OS_Project

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Guide

### 4.1 Run Unit Tests
To verify individual components, tensor dimensions, and constraint enforcement:
```bash
python run_all_tests.py
```
*Expected Output:* `12 PASSED, 0 FAILED`.

### 4.2 Run Pipeline Verification (Lightweight Dry-Run)
To verify forward/backward passes, GNN message passing, PPO/SAC gradient steps, curiosity dynamics, and checkpoint saving/loading:
```bash
python train_verifier.py --episodes 2 --steps 15
```

### 4.3 Run Comparative Benchmark Experiments
To run the evaluation harness comparing HDRL against Linux CFS, MAB, DeepRM-Graph, Static Allocation, and Strict Priority:
```bash
python run_benchmark.py
```
*Outputs generated in `results/`:*
- `results/benchmark_summary.csv`
- `results/raw_benchmark_runs.csv`
- `results/scheduler_comparison.png`

### 4.4 Launch Interactive Streamlit Dashboard
To launch the visual dashboard for real-time simulation, configuration, and result exploration:
```bash
streamlit run dashboard/app.py
```

---

## 5. Published Benchmarks (Table 1, Xing et al. 2025)

The table below shows the original published steady-state results reported in the research paper from a 10-hour evaluation on a 10,000-process testbed:

| Scheduler | OLTP $p_{99}$ (ms) | SLO Violations (%) | CPU Utilization (%) | Jain's Fairness Index |
| :--- | :--- | :--- | :--- | :--- |
| **Linux CFS** | $156.3 \pm 8.2$ | $12.4 \pm 1.1$ | $67.8 \pm 2.3$ | $0.72 \pm 0.04$ |
| **Oracle RM** | $124.7 \pm 6.5$ | $8.7 \pm 0.8$ | $71.2 \pm 1.9$ | $0.83 \pm 0.03$ |
| **CDBTune** | $132.1 \pm 7.1$ | $9.3 \pm 0.9$ | $69.5 \pm 2.1$ | $0.78 \pm 0.03$ |
| **MAB** | $118.6 \pm 5.9$ | $7.2 \pm 0.7$ | $73.6 \pm 1.7$ | $0.81 \pm 0.03$ |
| **DeepRM-Graph** | $108.2 \pm 6.1$ | $6.3 \pm 0.6$ | $76.8 \pm 1.8$ | $0.84 \pm 0.03$ |
| **MARL-Sched** | $102.5 \pm 5.7$ | $5.8 \pm 0.5$ | $78.9 \pm 1.6$ | $0.86 \pm 0.02$ |
| **AttentionScheduler** | $95.3 \pm 5.2$ | $5.1 \pm 0.5$ | $81.2 \pm 1.4$ | $0.88 \pm 0.02$ |
| **HDRL (Target Paper)** | **$87.4 \pm 4.3$** | **$4.0 \pm 0.4$** | **$86.5 \pm 1.2$** | **$0.91 \pm 0.02$** |
| **Oracle Upper Bound** | $72.3 \pm 3.8$ | $2.1 \pm 0.2$ | $92.3 \pm 0.8$ | $0.95 \pm 0.01$ |

**Summary of Advantages:**
- **$44.1\%$ reduction** in OLTP $p_{99}$ latency compared to Linux CFS ($26.3\%$ over MAB).
- **$67.7\%$ reduction** in SLO violations compared to Linux CFS.
- **$27.6\%$ boost** in CPU utilization efficiency ($86.5\%$ vs $67.8\%$).
- Runtime scheduling overhead maintained strictly under **$< 3\%$**.
