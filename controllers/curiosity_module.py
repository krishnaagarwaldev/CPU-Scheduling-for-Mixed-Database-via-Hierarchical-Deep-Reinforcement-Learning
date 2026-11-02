r"""Intrinsic Curiosity Module (ICM) for Exploration in Dynamic Workloads.

Faithfully implements Section 5.4:
- Forward dynamics model \hat{s}_{t+1} = f_\xi(s_t, a_t) (Equation 39)
- Intrinsic reward r_t^intrinsic = \eta * ||s_{t+1} - \hat{s}_{t+1}||^2 (Equation 41)
- Curiosity decay \eta_t = \eta_0 * \kappa^t with \kappa = 0.9999.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.scheduler_config import ControllerConfig


class ForwardDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)


class IntrinsicCuriosityModule:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        config: Optional[ControllerConfig] = None
    ):
        self.cfg = config or ControllerConfig()
        self.eta = self.cfg.eta_0
        self.decay = self.cfg.curiosity_decay

        self.dynamics_model = ForwardDynamicsModel(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=self.cfg.meta_lr)

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> float:
        """Computes curiosity bonus (Equation 41) and decays eta."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if next_state.ndim == 1:
            next_state = next_state.unsqueeze(0)

        with torch.no_grad():
            predicted_next_state = self.dynamics_model(state, action)
            prediction_error = F.mse_loss(predicted_next_state, next_state)
            intrinsic_reward = self.eta * prediction_error.item()

        # Decay eta
        self.eta *= self.decay
        return float(intrinsic_reward)

    def update_dynamics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> float:
        """Trains forward dynamics model to minimize prediction error (Equation 40)."""
        pred_next = self.dynamics_model(states, actions)
        loss = F.mse_loss(pred_next, next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
