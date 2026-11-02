"""Evaluation metrics for database CPU scheduling.

Faithfully implements Section 6.2 (Performance Metrics):
- OLTP p99 Latency (ms)
- Service Level Objective (SLO) violation rate (%)
- Overall CPU utilization (%)
- Symmetric fairness via Jain's Fairness Index
- Allocation stability and runtime overhead.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from workloads.process import DatabaseProcess


def calculate_percentile_latencies(
    processes: List[DatabaseProcess],
    category: Optional[str] = None,
    percentiles: Tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
) -> Dict[str, float]:
    """Calculates tail latencies across completed processes."""
    if category is not None:
        target_procs = [p for p in processes if p.category == category]
    else:
        target_procs = processes

    if not target_procs:
        return {f"p{int(p)}": 0.0 for p in percentiles}

    latencies = [p.turnaround_time_ms for p in target_procs]
    return {f"p{int(p)}": float(np.percentile(latencies, p)) for p in percentiles}


def calculate_slo_violations(
    processes: List[DatabaseProcess],
    slo_thresholds_ms: Dict[str, float]
) -> Dict[str, Any]:
    """Calculates category-wise and overall SLO violation rates."""
    total = len(processes)
    if total == 0:
        return {"total_count": 0, "violation_count": 0, "violation_pct": 0.0, "by_category": {}}

    violations_by_cat = {}
    counts_by_cat = {}

    for p in processes:
        counts_by_cat[p.category] = counts_by_cat.get(p.category, 0) + 1
        threshold = slo_thresholds_ms.get(p.category, 100.0)
        if p.turnaround_time_ms > threshold:
            violations_by_cat[p.category] = violations_by_cat.get(p.category, 0) + 1

    total_violations = sum(violations_by_cat.values())
    cat_rates = {}
    for cat, count in counts_by_cat.items():
        v = violations_by_cat.get(cat, 0)
        cat_rates[cat] = {
            "count": count,
            "violations": v,
            "violation_pct": (v / count * 100.0) if count > 0 else 0.0
        }

    return {
        "total_count": total,
        "violation_count": total_violations,
        "violation_pct": (total_violations / total * 100.0),
        "by_category": cat_rates
    }


def calculate_jains_fairness_index(values: List[float]) -> float:
    """Computes Jain's Fairness Index: J(x) = (sum x_i)^2 / (n * sum x_i^2)."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 1.0
    sum_val = float(np.sum(arr))
    sum_sq = float(np.sum(arr ** 2))
    if sum_sq == 0.0:
        return 1.0
    return float((sum_val ** 2) / (len(arr) * sum_sq))
