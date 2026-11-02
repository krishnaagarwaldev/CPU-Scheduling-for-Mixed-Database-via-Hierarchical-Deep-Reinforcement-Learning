r"""Prioritized Experience Replay (PER) with SLO Violation Bonus.

Faithfully implements Section 5.5:
- Priority p_t = |\delta_t|^\rho + \zeta * I[v_t^SLO > 0] + \xi * |r_t| (Equation 43)
- Sampling probability P(i) = p_i^\nu / \sum_k p_k^\nu (Equation 44)
- Parameters: \rho = 0.6, \zeta = 10, \xi = 0.1, \nu = 0.4.
"""

from typing import List, Tuple, Any, Dict
import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 10000,
        rho: float = 0.6,      # Prioritization strength
        zeta: float = 10.0,    # SLO violation amplifier
        xi: float = 0.1,       # Reward magnitude coefficient
        nu: float = 0.4        # Degree of prioritization
    ):
        self.capacity = capacity
        self.rho = rho
        self.zeta = zeta
        self.xi = xi
        self.nu = nu

        self.buffer: List[Dict[str, Any]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = 1.0,
        slo_violation: bool = False
    ):
        """Adds a transition and computes priority p_t (Equation 43)."""
        # Eq. 43: p_t = |delta_t|^rho + zeta * I[v_t^SLO > 0] + xi * |r_t|
        p_t = (abs(td_error) ** self.rho) + (self.zeta if slo_violation else 0.0) + (self.xi * abs(reward))
        p_t = max(1e-4, float(p_t))

        transition = {
            "state": state,
            "action": action,
            "reward": float(reward),
            "next_state": next_state,
            "done": float(done)
        }

        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = p_t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Samples batch according to probability P(i) (Equation 44)."""
        if self.size == 0:
            return [], np.zeros(0), np.zeros(0)

        actual_batch = min(self.size, batch_size)
        prios = self.priorities[:self.size]
        scaled_prios = prios ** self.nu
        probs = scaled_prios / scaled_prios.sum()

        indices = np.random.choice(self.size, size=actual_batch, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        # Importance sampling weights for bias correction
        weights = (self.size * probs[indices]) ** (-self.nu)
        weights /= weights.max()

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, slo_violations: np.ndarray):
        """Updates transition priorities with new TD errors."""
        for idx, td, viol in zip(indices, td_errors, slo_violations):
            r = self.buffer[idx]["reward"]
            p_t = (abs(td) ** self.rho) + (self.zeta if viol else 0.0) + (self.xi * abs(r))
            self.priorities[idx] = max(1e-4, float(p_t))
