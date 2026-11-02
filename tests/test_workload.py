"""Unit tests for workload generation and process constraints."""

import numpy as np
from workloads.process import DatabaseProcess
from workloads.workload_generator import DatabaseWorkloadGenerator


def test_process_feature_vector_dimension():
    proc = DatabaseProcess(
        pid=1,
        category="OLTP",
        arrival_time_ms=10.0,
        total_cpu_demand_ms=25.0,
        remaining_cpu_demand_ms=25.0,
        priority=8,
        op_type="READ"
    )
    feat = proc.get_feature_vector()
    # Must match Eq. 9: r_i(4) + p_i(4) + w_i(8) + h_i(3) = 19
    assert feat.shape == (19,)
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_workload_generator_sorting_and_dependencies():
    gen = DatabaseWorkloadGenerator(seed=42)
    procs = gen.generate_mixed_workload(duration_ms=2000.0)
    assert len(procs) > 0

    # Verify chronological arrival order
    arrival_times = [p.arrival_time_ms for p in procs]
    assert arrival_times == sorted(arrival_times)

    # Verify no process depends on an uncreated or future process
    pid_set = set()
    for p in procs:
        pid_set.add(p.pid)
        for dep in p.dependencies:
            assert dep in pid_set, f"Process {p.pid} depends on future/unknown PID {dep}"


def test_process_execution_progress():
    proc = DatabaseProcess(
        pid=1,
        category="OLAP",
        arrival_time_ms=0.0,
        total_cpu_demand_ms=100.0,
        remaining_cpu_demand_ms=100.0
    )
    assert not proc.is_completed
    proc.remaining_cpu_demand_ms -= 100.0
    assert proc.is_completed
