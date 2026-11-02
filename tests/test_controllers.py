"""Unit tests for Hierarchical Controller and Multi-Objective Reward calculations."""

import torch
import numpy as np

from controllers.meta_controller import MetaController
from controllers.sub_controller import CategorySubController
from controllers.curiosity_module import IntrinsicCuriosityModule
from evaluation.metrics import calculate_percentile_latencies, calculate_slo_violations, calculate_jains_fairness_index
from workloads.process import DatabaseProcess


def test_meta_controller_action_constraints():
    meta = MetaController(meta_state_dim=21, graph_context_dim=64, hidden_dim=64)
    meta_state = torch.randn(21)
    graph_ctx = torch.randn(64)

    action, log_prob, val = meta.select_action(meta_state, graph_ctx, deterministic=True)
    # Conservation (Eq. 28)
    assert abs(sum(action.values()) - 1.0) < 1e-6
    # Starvation (Eq. 29)
    for cat, val in action.items():
        assert val >= 0.05


def test_sub_controller_sac_action():
    sub = CategorySubController(category="OLTP", state_dim=19 + 64, action_dim=1, hidden_dim=32)
    state = torch.randn(19 + 64)
    action = sub.select_action(state, deterministic=True)
    assert -1.0 <= action <= 1.0


def test_curiosity_bonus_and_decay():
    curiosity = IntrinsicCuriosityModule(state_dim=10, action_dim=4, hidden_dim=32)
    s = torch.randn(10)
    a = torch.randn(4)
    s_next = torch.randn(10)

    bonus1 = curiosity.compute_intrinsic_reward(s, a, s_next)
    bonus2 = curiosity.compute_intrinsic_reward(s, a, s_next)
    assert bonus1 >= 0.0
    # Decayed bonus
    assert bonus2 <= bonus1


def test_jains_fairness_index():
    # Identical allocations => perfectly fair = 1.0
    assert abs(calculate_jains_fairness_index([0.25, 0.25, 0.25, 0.25]) - 1.0) < 1e-6
    # Skewed allocation => lower fairness
    skewed = calculate_jains_fairness_index([0.9, 0.05, 0.03, 0.02])
    assert skewed < 0.6
