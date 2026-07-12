"""Validated configuration loading shared by command-line entry points."""

from pathlib import Path
from numbers import Real
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


def _coerce_nonnegative_number(mapping: Dict[str, Any], key: str, context: str) -> None:
    """Normalize YAML scientific-notation strings and reject invalid values."""
    value = mapping[key]
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise ConfigError(f"{context}.{key} must be non-negative") from error
        mapping[key] = value
    if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
        raise ConfigError(f"{context}.{key} must be non-negative")


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
    if environment["name"] not in {"chrome_dino", "vizdoom", "mock"}:
        raise ConfigError("environment.name must be chrome_dino, vizdoom, or mock")
    if not isinstance(diffusion["context_length"], int) or diffusion["context_length"] <= 0:
        raise ConfigError("diffusion.context_length must be a positive integer")

    diffusion_resolution = diffusion.get("resolution")
    if diffusion_resolution is not None:
        if not isinstance(diffusion_resolution, dict):
            raise ConfigError("diffusion.resolution must be a mapping")
        _require(diffusion_resolution, ("width", "height"), "diffusion.resolution")
        if diffusion_resolution != resolution:
            raise ConfigError("diffusion.resolution must match environment.resolution")

    for key in ("batch_size", "gradient_accumulation_steps", "num_train_steps", "save_every_n_steps", "eval_every_n_steps", "keep_last_n_checkpoints", "warmup_steps"):
        if key in diffusion and (not isinstance(diffusion[key], int) or diffusion[key] < 0 or (key not in {"warmup_steps", "keep_last_n_checkpoints"} and diffusion[key] == 0)):
            raise ConfigError(f"diffusion.{key} must be a valid non-negative integer")
    for key in ("learning_rate", "gradient_clip", "weight_decay", "cfg_drop_prob", "cfg_scale"):
        if key in diffusion:
            _coerce_nonnegative_number(diffusion, key, "diffusion")
    if "cfg_drop_prob" in diffusion and diffusion["cfg_drop_prob"] > 1:
        raise ConfigError("diffusion.cfg_drop_prob must be between 0 and 1")
    if "optimizer" in diffusion and diffusion["optimizer"].lower() not in {"adamw", "adafactor"}:
        raise ConfigError("diffusion.optimizer must be AdamW or Adafactor")
    if "lr_scheduler" in diffusion and diffusion["lr_scheduler"].lower() not in {"constant", "linear", "cosine"}:
        raise ConfigError("diffusion.lr_scheduler must be constant, linear, or cosine")

    noise = diffusion.get("noise_augmentation")
    if noise is not None:
        if not isinstance(noise, dict):
            raise ConfigError("diffusion.noise_augmentation must be a mapping")
        if "num_buckets" in noise and (not isinstance(noise["num_buckets"], int) or noise["num_buckets"] <= 0):
            raise ConfigError("diffusion.noise_augmentation.num_buckets must be positive")
        if "max_noise_level" in noise and (not isinstance(noise["max_noise_level"], Real) or not 0 <= noise["max_noise_level"] <= 1):
            raise ConfigError("diffusion.noise_augmentation.max_noise_level must be between 0 and 1")

    if "action_repeat" in environment and (not isinstance(environment["action_repeat"], int) or environment["action_repeat"] <= 0):
        raise ConfigError("environment.action_repeat must be a positive integer")

    agent = config["agent"]
    if "algorithm" in agent and agent["algorithm"] not in {"DQN", "PPO"}:
        raise ConfigError("agent.algorithm must be DQN or PPO")
    if environment["name"] == "chrome_dino" and agent.get("algorithm") == "PPO":
        raise ConfigError("chrome_dino profile must use DQN")
    if environment["name"] == "vizdoom" and agent.get("algorithm") == "DQN":
        raise ConfigError("vizdoom profile must use PPO")

    if config.get("mixed_precision") and config.get("device") == "cpu":
        raise ConfigError("mixed_precision cannot be enabled when device is explicitly cpu")

    return config


def validate_diffusion_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate fields required specifically by the diffusion training entry point."""
    validate_config(config)
    diffusion = config["diffusion"]
    _require(
        diffusion,
        (
            "action_embedding_dim", "batch_size", "num_train_steps", "learning_rate",
            "noise_augmentation", "save_every_n_steps", "eval_every_n_steps",
        ),
        "diffusion training",
    )
    if not isinstance(diffusion["noise_augmentation"], dict):
        raise ConfigError("diffusion.noise_augmentation must be a mapping")
    _require(diffusion["noise_augmentation"], ("num_buckets", "max_noise_level"), "noise_augmentation")
    _require(config, ("logging",), "diffusion training")
    _require(config["logging"], ("log_interval",), "logging")
    if config["logging"]["log_interval"] <= 0:
        raise ConfigError("logging.log_interval must be positive")
    return config


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)
