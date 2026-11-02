import os

# Set local cache paths to avoid sandbox permission errors with PyTorch Inductor / Dynamo
_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache"))
os.makedirs(_cache_dir, exist_ok=True)
os.makedirs(os.path.join(_cache_dir, "torch"), exist_ok=True)
os.makedirs(os.path.join(_cache_dir, "inductor"), exist_ok=True)
os.makedirs(os.path.join(_cache_dir, "dynamo"), exist_ok=True)
os.makedirs(os.path.join(_cache_dir, "matplotlib"), exist_ok=True)
os.environ["MPLCONFIGDIR"] = os.path.join(_cache_dir, "matplotlib")

os.environ["LOCALAPPDATA"] = _cache_dir
os.environ["APPDATA"] = _cache_dir
os.environ["USERPROFILE"] = _cache_dir
os.environ["TEMP"] = os.path.join(_cache_dir, "temp")
os.environ["TMP"] = os.path.join(_cache_dir, "temp")
os.environ["TORCH_HOME"] = os.path.join(_cache_dir, "torch")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(_cache_dir, "inductor")
os.environ["TORCHDYNAMO_CACHE_DIR"] = os.path.join(_cache_dir, "dynamo")

from .scheduler_config import (
    WorkloadConfig,
    ResourceConstraintsConfig,
    MultiObjectiveRewardConfig,
    GNNConfig,
    ControllerConfig,
    SimulatorConfig,
    PaperBenchmarkTargets
)

__all__ = [
    "WorkloadConfig",
    "ResourceConstraintsConfig",
    "MultiObjectiveRewardConfig",
    "GNNConfig",
    "ControllerConfig",
    "SimulatorConfig",
    "PaperBenchmarkTargets"
]
