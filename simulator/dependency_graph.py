"""Dynamic Process Dependency Graph construction.

Faithfully implements Section 4.1:
- Node features x_i in R^19 (Equation 9)
- Edge features e_ij in R^4 (Equation 10)
  e_ij = [w_comm, w_cont, w_dep, tau_ij]
- Communication, resource contention (locks/buffer pages), workflow DAGs, and temporal self-loops.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from workloads.process import DatabaseProcess


class DynamicDependencyGraph:
    def __init__(self, ewma_window_size: int = 10):
        self.ewma_window_size = ewma_window_size
        self.edge_history: Dict[Tuple[int, int], int] = {}  # Tracks edge persistence over time
        self.current_step = 0

    def build_graph(
        self,
        active_processes: List[DatabaseProcess]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds graph representation from currently active database processes.
        
        Returns:
            node_features: (N, 19) float32 matrix
            edge_index: (2, E) int64 matrix (source -> target)
            edge_features: (E, 4) float32 matrix
        """
        self.current_step += 1
        N = len(active_processes)
        if N == 0:
            return (
                np.zeros((0, 19), dtype=np.float32),
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32)
            )

        # 1. Node Features (N, 19)
        pid_to_idx = {p.pid: idx for idx, p in enumerate(active_processes)}
        node_features = np.stack([p.get_feature_vector() for p in active_processes], axis=0)

        # 2. Edges and Edge Features
        sources: List[int] = []
        targets: List[int] = []
        features: List[List[float]] = []

        current_edges_set = set()

        for i, p_src in enumerate(active_processes):
            # A. Temporal self-loops: e_ii = [0.0, 0.0, 0.0, 1.0]
            sources.append(i)
            targets.append(i)
            features.append([0.0, 0.0, 0.0, 1.0])
            current_edges_set.add((p_src.pid, p_src.pid))

            # B. Workflow & Lock dependencies (p_src waiting on dependency)
            for dep_pid in p_src.dependencies:
                if dep_pid in pid_to_idx:
                    j = pid_to_idx[dep_pid]
                    sources.append(i)
                    targets.append(j)
                    
                    # Persistence tau_ij
                    hist_count = self.edge_history.get((p_src.pid, dep_pid), 0) + 1
                    tau_ij = min(1.0, hist_count / float(self.ewma_window_size))
                    
                    # Contention severity based on lock wait
                    w_cont = float(np.log1p(max(0.0, p_src.lock_wait_time_ms)))
                    w_comm = min(1.0, (p_src.network_traffic_kb_s + p_src.io_bandwidth_mb_s) / 500.0)
                    w_dep = 1.0  # Workflow flag
                    
                    features.append([w_comm, w_cont, w_dep, tau_ij])
                    current_edges_set.add((p_src.pid, dep_pid))

            # C. Resource contention edges: shared buffer pool pages
            for j in range(i + 1, N):
                p_dst = active_processes[j]
                # Check for shared buffer pool pages
                shared = set(p_src.shared_buffer_pages).intersection(p_dst.shared_buffer_pages)
                if shared:
                    contention_severity = float(np.log1p(len(shared) * 2.0))
                    
                    # Add bidirectional edge between competing processes
                    for u, v, p_u, p_v in [(i, j, p_src, p_dst), (j, i, p_dst, p_src)]:
                        sources.append(u)
                        targets.append(v)
                        hist_count = self.edge_history.get((p_u.pid, p_v.pid), 0) + 1
                        tau_ij = min(1.0, hist_count / float(self.ewma_window_size))
                        w_comm = 0.1
                        w_dep = 0.0
                        features.append([w_comm, contention_severity, w_dep, tau_ij])
                        current_edges_set.add((p_u.pid, p_v.pid))

        # Update edge history tracking
        self.edge_history = {pair: self.edge_history.get(pair, 0) + 1 for pair in current_edges_set}

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_features = np.array(features, dtype=np.float32)

        return node_features, edge_index, edge_features
