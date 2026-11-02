"""Soft Actor-Critic (SAC) Continuous Sub-Controller for Process-Level Scheduling.

Faithfully implements Section 3.2.2 & 5.2:
- Squashed Gaussian Actor a_l^c = tanh(mu + sigma * eps) (Equation 3)
- Process-level fine-grained CPU share & quota adjustments
- Twin Q-networks with automatic entropy temperature tuning (Equation 35).
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config.scheduler_config import ControllerConfig


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), min=-20.0, max=2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)   # Squash to [-1, 1]
        
        # Enforce log-probability correction for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        # Twin Q-networks (Q1 and Q2) to prevent overestimation bias
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class CategorySubController:
    def __init__(
        self,
        category: str,
        state_dim: int = 19 + 64,  # node feature (19) + category context (64)
        action_dim: int = 1,
        hidden_dim: int = 64,
        config: Optional[ControllerConfig] = None
    ):
        self.category = category
        self.cfg = config or ControllerConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Automatic entropy temperature adjustment (Eq. 35)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.sub_lr)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.sub_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.sub_lr)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> float:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        # Action in [-1, 1], mapped to share multiplier [0.5, 1.5]
        return float(action.squeeze().cpu().item())

    def update_parameters(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Performs SAC maximum entropy update step (Equation 35)."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            y = rewards + (1.0 - dones) * self.cfg.gamma_sub * min_q_target

        # Critic update
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        curr_actions, curr_log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, curr_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * curr_log_probs) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (curr_log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Polyak averaging target network update
        for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * target_p.data)

        return float(critic_loss.item()), float(actor_loss.item()), float(alpha_loss.item())
