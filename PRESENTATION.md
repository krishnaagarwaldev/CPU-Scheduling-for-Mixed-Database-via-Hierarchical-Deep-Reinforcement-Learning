# Presentation Deck: Self-Adapting CPU Scheduling for Mixed Database Workloads

**Topic:** Self-Adapting CPU Scheduling for Mixed Database Workloads via Hierarchical Deep Reinforcement Learning  
**Presenter:** 3rd Year B.Tech, Artificial Intelligence & Data Science  
**Subject:** 5th Semester Operating Systems (OS) Final Project  
**Reference:** Xing et al., *MDPI Symmetry* 2025 (Vol. 17, Iss. 7, Art. 1109)

---

## Slide 1: Title & Introduction
- **Title:** Self-Adapting CPU Scheduling for Mixed Database Workloads
- **Sub-theme:** Hierarchical Deep Reinforcement Learning (PPO + SAC) with Graph Neural Networks
- **Core Motivation:**
  - Modern databases are multi-model and hybrid (HTAP): concurrently handling transactions (OLTP), analytics (OLAP), vector search (ANN), and maintenance.
  - Native OS schedulers (Linux CFS) lack query semantics and transaction deadlines.

---

## Slide 2: The Core Problem & Conflict
- **Competing Workload Profiles:**
  - **OLTP (TPC-C):** Strict tail latency ($p_{99} < 100\text{ ms}$), high priority, bursty.
  - **OLAP (TPC-H):** Massive table scans, high CPU/memory consumption.
  - **Vector Search:** Heavy SIMD/AVX compute.
  - **Background Maintenance:** Periodic vacuuming, batch compaction.
- **Why Linux CFS Fails:**
  - Uses uniform virtual runtime ($vruntime$).
  - Long analytical queries monopolize CPU cores, causing severe starvation and $p_{99}$ latency spikes for latency-critical transactions.

---

## Slide 3: Proposed Two-Tier Control Architecture
- **Symmetric Two-Tier Hierarchy:**
  1. **Tier 1: Meta-Controller (PPO):**
     - Operates at macro intervals ($T_h = 10-30\text{ s}$).
     - Allocates category budget: $\mathbf{W} = (\alpha_{\text{OLTP}}, \alpha_{\text{OLAP}}, \alpha_{\text{vec}}, \alpha_{\text{bg}})$.
     - Enforces operational constraints: conservation ($\sum \alpha = 1$), starvation prevention ($\alpha \ge 0.05$), and stability ($|\Delta \alpha| \le 0.20$).
  2. **Tier 2: Sub-Controllers (SAC):**
     - Dedicated continuous control per workload category.
     - Operates at micro intervals ($T_l = 50-100\text{ ms}$).
     - Continuous process-level tuning via squashed Gaussian actor.

---

## Slide 4: Graph Neural Network (GNN) Dependency Modeling
- **Why GNNs for Operating Systems?**
  - Database processes are not independent; they form complex dependency graphs.
- **Graph Formulation:** $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$
  - **Nodes ($v_i \in \mathbb{R}^{19}$, Eq. 9):** Instantaneous CPU demand, memory ratio, I/O rates, transaction throughput, priority, EWMA history.
  - **Edges ($e_{ij} \in \mathbb{R}^4$, Eq. 10):** Communication intensity, lock contention wait chains, shared buffer page conflicts, workflow DAG dependencies.
- **MPNN with Learned Attention (Eq. 11-14):**
  - Attention-weighted aggregation captures transitive lock blocking and preemptive bottleneck detection.

---

## Slide 5: Multi-Objective Reward & Pareto Optimization
- **Composite Multi-Objective Reward (Eq. 27):**
  $$r_t^h = 0.40 \cdot r_t^{\text{SLO}} + 0.30 \cdot r_t^{\text{util}} + 0.20 \cdot r_t^{\text{fair}} - 0.10 \cdot r_t^{\text{stable}}$$
- **Objective Components:**
  - **SLO Penalty (Eq. 32):** Quadratic penalty on tail latency violations with category sensitivity weights ($\lambda_{\text{OLTP}}=10, \lambda_{\text{OLAP}}=3, \lambda_{\text{vec}}=5, \lambda_{\text{bg}}=1$).
  - **Resource Utilization (Eq. 33):** Piecewise target optimization centered at $u_{\text{target}} = 85\%$.
  - **Fairness:** Jain's Fairness Index preventing workload starvation.
  - **Stability:** Quadratic penalty preventing erratic budget oscillations.

---

## Slide 6: Adaptation Mechanisms
- **Intrinsic Curiosity Module (ICM, Eqs. 39-42):**
  - Forward dynamics model predicting next state $\hat{s}_{t+1} = f_\xi(s_t, a_t)$.
  - Intrinsic exploration bonus $r_t^{\text{intrinsic}} = \eta \cdot \|s_{t+1} - \hat{s}_{t+1}\|^2$ with decay $\kappa = 0.9999$.
  - Ensures robust exploration of uncharted scheduling states.
- **Model-Agnostic Meta-Learning (MAML, Eqs. 36-38):**
  - Inner-loop / outer-loop meta-gradients enabling adaptation to novel workload mixes within $\le 5$ updates.
- **Prioritized Experience Replay (PER, Eq. 43):**
  - Focuses training on SLO-violating transitions with amplification weight $\zeta = 10$.

---

## Slide 7: OS Actuation via Linux cgroups v2
- **Kernel Interface:**
  - `cpu.weight` $\in [1, 10000]$: Proportional share distribution (replaces cgroups v1 `cpu.shares`).
  - `cpu.max` $\in [\text{quota}, \text{period}]$: Hard bandwidth throttling ceiling preventing analytical query runaways.
- **Simulation Layer:**
  - Cross-platform discrete-event multi-core simulator with context switch penalties ($0.05\text{ ms}$) and quantum slicing ($5.0\text{ ms}$).

---

## Slide 8: Experimental Evaluation & Published Benchmarks
- **Published Benchmarks (Table 1, Xing et al. MDPI 2025):**
  - **OLTP $p_{99}$ Latency:** $44.1\%$ reduction over Linux CFS ($87.4\text{ ms}$ vs $156.3\text{ ms}$).
  - **SLO Violations:** $67.7\%$ reduction over Linux CFS ($4.0\%$ vs $12.4\%$).
  - **CPU Utilization:** Boosted from $67.8\%$ to $86.5\%$ ($+27.6\%$).
  - **Scheduling Overhead:** Kept under $3\%$, scaling stably to 10,000 concurrent processes.
- **Simulated Testbed Comparison:**
  - HDRL, Linux CFS, MAB, DeepRM-Graph, Static Allocation, Strict Priority evaluated under identical workloads and seeds.

---

## 9. Interactive Streamlit Dashboard
- **Features Built:**
  - Live algorithm switching (HDRL vs CFS vs MAB vs Static vs Priority).
  - Sliders for simulation time, virtual CPU cores, and category arrival rates.
  - Interactive visualization of category allocation pie charts and completed process turnaround times.
  - One-click CSV export of metrics and process execution logs.

---

## 10. Conclusion & Project Contributions
- **What Was Achieved:**
  - 100% faithful reproduction of research methodology, formulas, and constraints from MDPI *Symmetry* 2025.
  - Full codebase modularized into workload generation, GNN, hierarchical controllers, cgroups v2, baselines, and evaluation.
  - Rigorous unit test suite with 100% passing rate.
  - Pragmatic 3-tier training strategy that avoided unapproved expensive compute while verifying complete mathematical pipeline.
