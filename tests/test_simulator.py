"""Unit tests for discrete-event CPU simulator and cgroups actuation."""

import numpy as np

from simulator.cgroups_interface import CgroupsInterface
from simulator.cpu_core import MultiCoreCPU
from simulator.scheduling_environment import DatabaseSchedulingEnv


def test_cgroups_interface_constraints():
    cgroups = CgroupsInterface(is_simulation=True)
    cgroups.set_cpu_weight("OLTP", 500)
    cgroups.set_cpu_weight("OLAP", 250)
    cgroups.set_cpu_weight("VECTOR", 150)
    cgroups.set_cpu_weight("BACKGROUND", 100)

    shares = cgroups.get_effective_shares()
    assert abs(sum(shares.values()) - 1.0) < 1e-6
    assert abs(shares["OLTP"] - 0.50) < 1e-3
    assert abs(shares["OLAP"] - 0.25) < 1e-3


def test_multicore_execution_and_preemption():
    cpu = MultiCoreCPU(num_cores=2, context_switch_ms=0.1)
    assert cpu.get_idle_core() is not None

    env = DatabaseSchedulingEnv()
    obs, info = env.reset(duration_ms=1000.0)
    assert "meta_state" in obs
    assert "node_features" in obs
    assert obs["meta_state"].shape == (21,)


def test_macro_action_constraints_enforcement():
    env = DatabaseSchedulingEnv()
    env.reset(duration_ms=1000.0)

    # Test extreme proposal violating starvation (alpha_min) and delta_max
    extreme_action = {"OLTP": 0.99, "OLAP": 0.005, "VECTOR": 0.002, "BACKGROUND": 0.003}
    env.apply_macro_action(extreme_action)

    allocs = env.current_allocations
    # Conservation check (Eq. 28)
    assert abs(sum(allocs.values()) - 1.0) < 1e-6

    # Starvation prevention check (Eq. 29)
    for cat, val in allocs.items():
        assert val >= 0.0499, f"{cat} starved with allocation {val}"
