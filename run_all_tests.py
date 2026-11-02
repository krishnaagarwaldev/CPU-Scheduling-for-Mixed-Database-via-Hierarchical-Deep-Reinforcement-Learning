"""Automated test execution runner."""

import sys
import traceback

import config  # bootstraps local cache
from tests.test_workload import (
    test_process_feature_vector_dimension,
    test_workload_generator_sorting_and_dependencies,
    test_process_execution_progress
)
from tests.test_simulator import (
    test_cgroups_interface_constraints,
    test_multicore_execution_and_preemption,
    test_macro_action_constraints_enforcement
)
from tests.test_gnn import (
    test_gnn_shapes_and_attention,
    test_gnn_empty_graph_handling
)
from tests.test_controllers import (
    test_meta_controller_action_constraints,
    test_sub_controller_sac_action,
    test_curiosity_bonus_and_decay,
    test_jains_fairness_index
)

test_functions = [
    test_process_feature_vector_dimension,
    test_workload_generator_sorting_and_dependencies,
    test_process_execution_progress,
    test_cgroups_interface_constraints,
    test_multicore_execution_and_preemption,
    test_macro_action_constraints_enforcement,
    test_gnn_shapes_and_attention,
    test_gnn_empty_graph_handling,
    test_meta_controller_action_constraints,
    test_sub_controller_sac_action,
    test_curiosity_bonus_and_decay,
    test_jains_fairness_index
]

print("=" * 65)
print(f"RUNNING {len(test_functions)} SYSTEM & SCHEDULER UNIT TESTS")
print("=" * 65)

passed = 0
failed = 0

for test_fn in test_functions:
    name = test_fn.__name__
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

print("=" * 65)
print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
print("=" * 65)

sys.exit(0 if failed == 0 else 1)
