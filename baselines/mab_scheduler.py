"""Multi-Armed Bandit (MAB) Baseline for CPU Allocation.

Implements the MAB baseline cited in Table 1 (Section 6.2, Ref [46]):
- Uses Upper Confidence Bound (UCB) heuristic to dynamically allocate CPU shares
- Evaluates discrete resource budget arms and selects the arm maximizing UCB score.
"""

from typing import List, Dict
import numpy as np


class MultiArmedBanditScheduler:
    def __init__(self, categories: List[str], c_explore: float = 1.0):
        self.categories = categories
        self.c_explore = c_explore

        # Define candidate budget distribution arms
        self.arms = [
            {"OLTP": 0.40, "OLAP": 0.30, "VECTOR": 0.20, "BACKGROUND": 0.10},  # Balanced
            {"OLTP": 0.60, "OLAP": 0.20, "VECTOR": 0.15, "BACKGROUND": 0.05},  # OLTP heavy
            {"OLTP": 0.25, "OLAP": 0.45, "VECTOR": 0.20, "BACKGROUND": 0.10},  # OLAP heavy
            {"OLTP": 0.30, "OLAP": 0.20, "VECTOR": 0.40, "BACKGROUND": 0.10},  # Vector heavy
            {"OLTP": 0.25, "OLAP": 0.25, "VECTOR": 0.25, "BACKGROUND": 0.25}   # Uniform
        ]
        self.num_arms = len(self.arms)
        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)
        self.total_pulls = 0
        self.current_arm = 0

    def select_arm(self) -> Dict[str, float]:
        """Selects budget arm according to UCB1 rule."""
        self.total_pulls += 1
        for a in range(self.num_arms):
            if self.counts[a] == 0:
                self.current_arm = a
                return self.arms[a]

        # UCB calculation
        ucb_scores = self.values + self.c_explore * np.sqrt(np.log(self.total_pulls) / self.counts)
        self.current_arm = int(np.argmax(ucb_scores))
        return self.arms[self.current_arm]

    def update_reward(self, reward: float):
        """Updates empirical value of the selected arm."""
        arm = self.current_arm
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
