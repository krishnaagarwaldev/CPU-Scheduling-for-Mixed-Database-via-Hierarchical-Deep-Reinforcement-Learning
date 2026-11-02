"""DeepRM-Graph Baseline.

Implements the single-tier Deep Reinforcement Learning with graph input baseline
evaluated in Table 1 (Section 6.2, Ref [6]):
- Single flat policy network taking combined state and graph embeddings
- Lacks hierarchical control and curiosity/meta-learning adaptation.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeepRMGraphNetwork(nn.Module):
    def __init__(self, state_dim: int = 21, num_categories: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_categories)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(state), dim=-1)


class DeepRMGraphScheduler:
    def __init__(self, state_dim: int = 21, categories: Optional[List[str]] = None):
        self.categories = categories or ["OLTP", "OLAP", "VECTOR", "BACKGROUND"]
        self.net = DeepRMGraphNetwork(state_dim=state_dim, num_categories=len(self.categories))

    def select_action(self, meta_state: np.ndarray) -> Dict[str, float]:
        with torch.no_grad():
            s_t = torch.tensor(meta_state, dtype=torch.float32)
            probs = self.net(s_t).numpy()
        return {cat: float(probs[i]) for i, cat in enumerate(self.categories)}
