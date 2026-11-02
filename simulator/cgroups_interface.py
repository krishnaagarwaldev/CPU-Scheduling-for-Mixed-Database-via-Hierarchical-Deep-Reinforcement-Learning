"""Linux cgroups v2 abstraction and simulation layer.

Faithfully implements the actuation layer specified in Section 3.3.1:
- cpu.weight (proportional sharing in [1, 10000])
- cpu.max (bandwidth limiting [quota, period])
Includes automated fallback for cross-platform simulation (Windows/macOS/unprivileged Linux).
"""

import os
from typing import Dict, Optional, Tuple


class CgroupsInterface:
    def __init__(self, cgroup_root: str = "/sys/fs/cgroup/db_hdrl", is_simulation: bool = True):
        self.cgroup_root = cgroup_root
        self.is_simulation = is_simulation
        # In-memory cgroup state for simulated environments: category -> {weight, quota, period}
        self.cgroups_state: Dict[str, Dict[str, int]] = {
            "OLTP": {"weight": 100, "quota": -1, "period": 100000},
            "OLAP": {"weight": 100, "quota": -1, "period": 100000},
            "VECTOR": {"weight": 100, "quota": -1, "period": 100000},
            "BACKGROUND": {"weight": 100, "quota": -1, "period": 100000}
        }
        self.real_cgroups_available = self._check_real_cgroups() if not is_simulation else False

    def _check_real_cgroups(self) -> bool:
        """Checks if Linux cgroups v2 filesystem is accessible."""
        return os.path.exists("/sys/fs/cgroup/cgroup.controllers") and os.access("/sys/fs/cgroup", os.W_OK)

    def set_cpu_weight(self, category: str, weight: int) -> bool:
        """Sets cgroups v2 cpu.weight in range [1, 10000]."""
        clamped_weight = max(1, min(10000, int(weight)))
        if category in self.cgroups_state:
            self.cgroups_state[category]["weight"] = clamped_weight
        
        if self.real_cgroups_available:
            path = os.path.join(self.cgroup_root, category.lower(), "cpu.weight")
            try:
                with open(path, "w") as f:
                    f.write(str(clamped_weight))
                return True
            except (IOError, OSError):
                return False
        return True

    def set_cpu_max(self, category: str, quota_us: int, period_us: int = 100000) -> bool:
        """Sets cgroups v2 cpu.max bandwidth quota (quota_us over period_us).
        
        quota_us = -1 means max (unconstrained).
        """
        if category in self.cgroups_state:
            self.cgroups_state[category]["quota"] = quota_us
            self.cgroups_state[category]["period"] = period_us

        if self.real_cgroups_available:
            path = os.path.join(self.cgroup_root, category.lower(), "cpu.max")
            val_str = f"{quota_us if quota_us > 0 else 'max'} {period_us}"
            try:
                with open(path, "w") as f:
                    f.write(val_str)
                return True
            except (IOError, OSError):
                return False
        return True

    def get_cgroup_state(self, category: str) -> Dict[str, int]:
        return self.cgroups_state.get(category, {"weight": 100, "quota": -1, "period": 100000})

    def get_effective_shares(self) -> Dict[str, float]:
        """Calculates normalized CPU share proportions across active cgroups."""
        total_weight = sum(state["weight"] for state in self.cgroups_state.values())
        if total_weight <= 0:
            return {cat: 0.25 for cat in self.cgroups_state}
        return {cat: state["weight"] / total_weight for cat, state in self.cgroups_state.items()}
