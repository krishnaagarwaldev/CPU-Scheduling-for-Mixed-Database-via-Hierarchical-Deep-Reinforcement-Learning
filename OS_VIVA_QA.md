# Operating Systems (OS) Viva Questions & Answers

**Project Title:** Self-Adapting CPU Scheduling for Mixed Database Workloads via Hierarchical Deep Reinforcement Learning  
**Target Audience:** 3rd Year B.Tech Computer Science / AI & Data Science Students, OS External Examiners, Faculty Reviewers.

---

### Q1: What is the fundamental problem of CPU scheduling in modern database management systems?
**Answer:**  
Modern databases (such as Hybrid Transactional/Analytical Processing or HTAP systems) handle co-located mixed workloads simultaneously:
1. Short, bursty, latency-sensitive transactions (**OLTP**).
2. Long-running, compute-heavy table scans (**OLAP**).
3. Compute-dense similarity searches (**Vector queries**).
4. Periodic maintenance tasks (**Background vacuuming / compaction**).

Standard operating system schedulers (like Linux CFS) operate at thread-level granularity and are agnostic to database query semantics, lock chains, and Service Level Objectives (SLOs). Under heavy analytical or vector load, analytical queries saturate CPU cores, causing severe queue delays and tail-latency ($p_{99}$) spikes for critical transactional queries.

---

### Q2: How does the Linux Completely Fair Scheduler (CFS) work, and why does it struggle with mixed database workloads?
**Answer:**  
CFS models an "ideal multi-tasking CPU" on real hardware. It assigns each runnable task a **virtual runtime ($vruntime$)**:
$$\Delta vruntime = \Delta \text{exec\_time} \times \frac{\text{NICE\_0\_LOAD}}{\text{task\_weight}}$$
CFS maintains runnable tasks in a time-ordered Red-Black tree (`rb_tree`) and always selects the leftmost node (smallest $vruntime$).

**Why it struggles:**
- CFS aims for uniform fairness over time.
- Long-running OLAP queries accumulate large $vruntime$, but when new transactional queries arrive in bursts, CFS grants them short time slices that get quickly preempted as analytical queries catch up.
- CFS does not recognize query deadlines, transaction priorities, or lock dependencies, treating a critical $p_{99}$ banking transaction identically to a background table scan.

---

### Q3: What is the difference between Linux `cgroups v1` and `cgroups v2`, and how are they used here?
**Answer:**  
- **cgroups v1:** Used multiple orthogonal hierarchies for different controllers (`cpu`, `memory`, `blkio`), leading to complex synchronization issues and priority inversions. It used `cpu.shares` (relative proportional shares where base is 1024) and `cpu.cfs_quota_us` / `cpu.cfs_period_us`.
- **cgroups v2:** Unified single-hierarchy model where processes belong to a single unified group tree. It redesigned CPU resource parameters:
  - **`cpu.weight`:** Proportional sharing in range $[1, 10000]$ (default 100).
  - **`cpu.max`:** Hard bandwidth limiting formatted as `[quota_us] [period_us]` (or `max [period_us]`).
- **In our project:** The scheduler translates high-level reinforcement learning decisions directly into `cpu.weight` and `cpu.max` constraints, dynamically adjusting category shares and capping runaway batch queries.

---

### Q4: Why is a Hierarchical Reinforcement Learning (HRL) design used instead of a single flat RL agent?
**Answer:**  
A flat RL agent trying to schedule individual processes across 10,000 concurrent database threads faces the **curse of dimensionality** and massive action-space explosion.
The paper uses a **Symmetric Two-Tier Hierarchy**:
1. **Tier 1 (Meta-Controller):** Operates on a coarse macro timescale ($T_h = 10-30\text{ s}$), focusing solely on strategic high-level CPU budget distribution across 4 workload categories $\mathbf{W} = (\alpha_{\text{OLTP}}, \alpha_{\text{OLAP}}, \alpha_{\text{vec}}, \alpha_{\text{bg}})$.
2. **Tier 2 (Sub-Controllers):** Operate on a faster micro timescale ($T_l = 50-100\text{ ms}$), with specialized continuous controllers (using Soft Actor-Critic) dedicated to each category.

This temporal and functional decomposition reduces training sample complexity by over $3.2\times$ and achieves stable convergence in 500 epochs compared to 1200+ epochs for flat RL.

---

### Q5: Why is a Graph Neural Network (GNN) needed in an operating systems scheduler?
**Answer:**  
Processes in a database are inherently interconnected:
- They coordinate via locks and latches (directed waiting edges).
- They compete for buffer pool pages and memory cache lines (undirected contention edges).
- Multi-stage queries execute in producer-consumer query pipelines (workflow DAG edges).

Standard flat state vectors ignore these relationships. The GNN uses **Message Passing Neural Networks (MPNN) with learned Attention-weighted aggregation** (Equations 11-14) to propagate transitive dependencies (e.g., if Process A holds a lock that Process B is waiting on, and Process B is blocking Process C). The scheduler can preemptively prioritize lock-holding processes to avoid cascading pipeline stalls.

---

### Q6: What are the exact constraints enforced on the Meta-Controller's resource allocation actions?
**Answer:**  
To guarantee operational feasibility and prevent operating system instability, the action $\mathbf{W} = (\alpha_{\text{OLTP}}, \alpha_{\text{OLAP}}, \alpha_{\text{vec}}, \alpha_{\text{bg}})$ must satisfy three strict mathematical constraints:
1. **Resource Conservation (Equation 28):**
   $$\sum_{c \in \mathcal{C}} \alpha_t^c = 1.0$$
   (The scheduler cannot allocate more or less than 100% of available CPU capacity).
2. **Starvation Prevention (Equation 29):**
   $$\alpha_t^c \ge \alpha_{\min} = 0.05, \quad \forall c \in \mathcal{C}$$
   (No workload class is ever completely starved; at least 5% CPU capacity is preserved).
3. **Allocation Stability Enforcement (Equation 30):**
   $$|\alpha_t^c - \alpha_{t-1}^c| \le \delta_{\max} = 0.20, \quad \forall c \in \mathcal{C}$$
   (CPU shares cannot shift by more than 20% in a single step to prevent oscillatory thrashing).

---

### Q7: Explain the Multi-Objective Reward Function and its Pareto Knee Point.
**Answer:**  
Equation 27 defines the composite reward:
$$r_t^h = w_1 r_t^{\text{SLO}} + w_2 r_t^{\text{util}} + w_3 r_t^{\text{fair}} - w_4 r_t^{\text{stable}}$$
Through systematic Pareto frontier analysis across 625 weight configurations (Figure 2 in the paper), the authors selected the balanced knee-point:
$$(w_1, w_2, w_3, w_4) = (0.40, 0.30, 0.20, 0.10)$$
- **$r_t^{\text{SLO}}$ (Eq. 32):** Quadratic penalty on tail latency violations with sensitivity penalty weights ($\lambda_{\text{OLTP}}=10, \lambda_{\text{OLAP}}=3, \lambda_{\text{vec}}=5, \lambda_{\text{bg}}=1$).
- **$r_t^{\text{util}}$ (Eq. 33):** Piecewise function rewarding approaching target utilization $u_{\text{target}} = 85\%$ while penalizing over-saturation.
- **$r_t^{\text{fair}}$:** Jain's Fairness Index ensuring equitable resource distribution.
- **$r_t^{\text{stable}}$:** Penalty on allocation changes $(\Delta \alpha)^2$ to maintain smooth runtime behavior.

---

### Q8: What is Tail Latency ($p_{99}$), and why is it more important than average latency in database systems?
**Answer:**  
Mean or average latency conceals severe outlier delays. If a web service performs 100 sub-queries to render a single user page, the probability that the user experiences the $99^{\text{th}}$ percentile latency approaches $1 - (0.99)^{100} \approx 63.4\%$.
In enterprise databases, Service Level Agreements (SLAs) penalize $p_{99}$ latency spikes, as high tail latency translates directly to dropped requests, cart abandonment, and timeout cascades.

---

### Q9: How do Curiosity-Driven Exploration and Meta-Learning (MAML) work in this framework?
**Answer:**  
- **Intrinsic Curiosity Module (ICM, Eqs. 39-42):** Maintains a forward dynamics model $\hat{s}_{t+1} = f_\xi(s_t, a_t)$. The intrinsic reward $r_t^{\text{intrinsic}} = \eta \cdot \|s_{t+1} - \hat{s}_{t+1}\|^2$ rewards exploring state-action combinations where dynamics prediction error is high, enabling the agent to discover novel scheduling policies even when extrinsic rewards are sparse.
- **Model-Agnostic Meta-Learning (MAML, Eqs. 36-38):** Uses inner-loop adaptation ($\theta' = \theta - \alpha \nabla L$) and outer-loop meta-updates ($\theta \leftarrow \theta - \beta \nabla \sum L$) across diverse workload distributions. When a sudden workload shift occurs (e.g., an unexpected spike in vector searches), the policy adapts within $\le 5$ gradient updates rather than requiring complete retraining.

---

### Q10: What is the scheduling overhead of this method, and how is it kept under 3%?
**Answer:**  
The paper proves that the hierarchical scheduler incurs $< 3\%$ CPU overhead in production:
1. **Timescale separation:** The expensive GNN inference and Meta-Controller optimization run infrequently at macro intervals ($T_h = 10-30\text{ s}$).
2. **Asynchronous execution:** The RL controller runs in a lightweight user-space daemon decoupled from the kernel critical path.
3. **Native kernel offloading:** The scheduler does not dispatch individual instructions; it writes high-level allocations to Linux `cgroups v2` (`cpu.weight` and `cpu.max`), letting the ultra-fast Linux kernel CFS scheduler handle sub-millisecond CPU core dispatching.
