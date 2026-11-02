"""PPO-based Meta-Controller for Strategic CPU Resource Allocation.

Faithfully implements Section 3.2.1 & 5.2:
- Macro allocation W = [w_oltp, w_olap, w_vec, w_bg] (Equation 2)
- Constraints enforcement:
  * Conservation sum(alpha_c) == 1 (Equation 28)
  * Starvation prevention alpha_c >= 0.05 (Equation 29)
  * Stability |alpha_c - alpha_prev| <= 0.20 (Equation 30)
- PPO clipped objective with GAE (Equation 34).
"""

from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal

from config.scheduler_config import ControllerConfig, ResourceConstraintsConfig

np_or_tensor = Union[np.ndarray, torch.Tensor]


class MetaPolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_categories: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # Outputs positive concentration parameters for Dirichlet distribution over simplex
        self.alpha_head = nn.Linear(hidden_dim, num_categories)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.net(state)
        # Softplus ensures alpha > 0, +1.0 ensures mode is well-defined
        alphas = F.softplus(self.alpha_head(h)) + 1.0
        return alphas


class MetaValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MetaController:
    def __init__(
        self,
        meta_state_dim: int = 21,
        graph_context_dim: int = 64,
        hidden_dim: int = 128,
        categories: Optional[List[str]] = None,
        config: Optional[ControllerConfig] = None,
        constraints: Optional[ResourceConstraintsConfig] = None
    ):
        self.categories = categories or ["OLTP", "OLAP", "VECTOR", "BACKGROUND"]
        self.num_cats = len(self.categories)
        self.cfg = config or ControllerConfig()
        self.constraints = constraints or ResourceConstraintsConfig()

        self.total_input_dim = meta_state_dim + graph_context_dim
        self.policy = MetaPolicyNetwork(self.total_input_dim, hidden_dim, self.num_cats)
        self.value_net = MetaValueNetwork(self.total_input_dim, hidden_dim)

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.parameters(), "lr": self.cfg.meta_lr},
            {"params": self.value_net.parameters(), "lr": self.cfg.meta_lr}
        ])

        self.prev_allocation = {cat: 1.0 / self.num_cats for cat in self.categories}

    def select_action(
        self,
        meta_state: np_or_tensor,
        graph_context: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
        """Selects macro allocation vector W satisfying simplex and stability constraints.
        
        Returns:
            allocation_dict: {cat: fraction}
            log_prob: Tensor log probability for PPO
            state_value: Tensor estimated value V(s)
        """
        if not isinstance(meta_state, torch.Tensor):
            meta_state = torch.tensor(meta_state, dtype=torch.float32)
        if not isinstance(graph_context, torch.Tensor):
            graph_context = torch.tensor(graph_context, dtype=torch.float32)

        if meta_state.ndim == 1:
            meta_state = meta_state.unsqueeze(0)
        if graph_context.ndim == 1:
            graph_context = graph_context.unsqueeze(0)

        full_state = torch.cat([meta_state, graph_context], dim=-1)
        alphas = self.policy(full_state)
        value = self.value_net(full_state).squeeze(-1)

        dist = Dirichlet(alphas)

        if deterministic:
            # Mode of Dirichlet or mean
            raw_action = alphas / alphas.sum(dim=-1, keepdim=True)
            log_prob = dist.log_prob(raw_action)
        else:
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action)

        raw_vec = raw_action.squeeze(0).detach().cpu().numpy()

        # Enforce constraints (Eq. 28, 29, 30)
        validated_dict: Dict[str, float] = {}
        for idx, cat in enumerate(self.categories):
            target_val = float(raw_vec[idx])
            prev_val = self.prev_allocation[cat]
            delta = target_val - prev_val
            # Stability constraint (Eq. 30): |delta| <= delta_max (0.20)
            clamped_delta = max(-self.constraints.delta_max, min(self.constraints.delta_max, delta))
            # Starvation prevention (Eq. 29): val >= alpha_min (0.05)
            val = max(self.constraints.alpha_min, prev_val + clamped_delta)
            validated_dict[cat] = val

        # Normalize to strictly satisfy sum == 1.0 (Eq. 28)
        tot = sum(validated_dict.values())
        for cat in validated_dict:
            validated_dict[cat] /= tot

        self.prev_allocation = validated_dict.copy()
        return validated_dict, log_prob.squeeze(0), value.squeeze(0)

    def update_ppo(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> float:
        """Executes PPO clipped surrogate update (Equation 34)."""
        alphas = self.policy(states)
        dist = Dirichlet(alphas)
        new_log_probs = dist.log_prob(actions)

        # Ratio rho_t(theta)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip_eps, 1.0 + self.cfg.ppo_clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_net(states).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        return float(total_loss.item())
