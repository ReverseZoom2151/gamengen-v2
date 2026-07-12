"""Validated configuration loading shared by command-line entry points."""

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a GameNGen configuration is incomplete or inconsistent."""


_TOP_LEVEL_KEYS = {
    "project_name",
    "experiment_name",
    "seed",
    "device",
    "num_workers",
    "mixed_precision",
    "use_paper_reward",
    "data_dir",
    "checkpoint_dir",
    "log_dir",
    "environment",
    "agent",
    "data_collection",
    "diffusion",
    "decoder",
    "distillation",
    "inference",
    "evaluation",
    "logging",
    "debug",
}


def _require(mapping: Dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context} is missing required keys: {', '.join(missing)}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate invariants shared by collection, training, and inference.

    The validator intentionally rejects typoed top-level settings. Nested settings
    remain extensible while the project is migrated to typed config models.
    """

    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")

    unknown = sorted(set(config) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"unknown top-level configuration keys: {', '.join(unknown)}")

    _require(
        config,
        ("project_name", "experiment_name", "data_dir", "checkpoint_dir", "log_dir"),
        "configuration",
    )
    _require(config, ("environment", "agent", "data_collection", "diffusion"), "configuration")

    environment = config["environment"]
    diffusion = config["diffusion"]
    if not isinstance(environment, dict) or not isinstance(diffusion, dict):
        raise ConfigError("environment and diffusion must be mappings")

    _require(environment, ("name", "num_actions", "resolution"), "environment")
    _require(diffusion, ("context_length", "pretrained_model"), "diffusion")

    resolution = environment["resolution"]
    if not isinstance(resolution, dict):
        raise ConfigError("environment.resolution must be a mapping")
    _require(resolution, ("width", "height"), "environment.resolution")

    for key in ("width", "height"):
        if not isinstance(resolution[key], int) or resolution[key] <= 0:
            raise ConfigError(f"environment.resolution.{key} must be a positive integer")

    if not isinstance(environment["num_actions"], int) or environment["num_actions"] <= 0:
        raise ConfigError("environment.num_actions must be a positive integer")
    if not isinstance(diffusion["context_length"], int) or diffusion["context_length"] <= 0:
        raise ConfigError("diffusion.context_length must be a positive integer")

    if config.get("mixed_precision") and config.get("device") == "cpu":
        raise ConfigError("mixed_precision cannot be enabled when device is explicitly cpu")

    return config


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)
