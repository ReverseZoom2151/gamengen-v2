"""Deterministic paper-ablation configuration planning."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable


def context_length_ablation(config: dict, lengths: Iterable[int] = (1, 2, 4, 8, 16, 32, 64)) -> list[dict]:
    plans = []
    for length in lengths:
        if not isinstance(length, int) or length <= 0:
            raise ValueError("context lengths must be positive integers")
        plan = copy.deepcopy(config)
        plan["diffusion"]["context_length"] = length
        plan["experiment_name"] = f"{config['experiment_name']}-context-{length}"
        plans.append(plan)
    return plans


def noise_ablation(config: dict) -> list[dict]:
    plans = []
    for enabled in (False, True):
        plan = copy.deepcopy(config)
        plan["diffusion"]["noise_augmentation"]["enabled"] = enabled
        plan["experiment_name"] = f"{config['experiment_name']}-noise-{'on' if enabled else 'off'}"
        plans.append(plan)
    return plans


def data_policy_ablation(config: dict, data_dirs: dict[str, str]) -> list[dict]:
    if not data_dirs:
        raise ValueError("at least one named policy dataset is required")
    plans = []
    for policy, data_dir in sorted(data_dirs.items()):
        if not policy or not data_dir:
            raise ValueError("policy names and dataset paths must be non-empty")
        plan = copy.deepcopy(config)
        plan["data_dir"] = data_dir
        plan["experiment_name"] = f"{config['experiment_name']}-policy-{policy}"
        plans.append(plan)
    return plans


def save_experiment_plan(path: str | Path, plans: list[dict]) -> Path:
    """Persist immutable planned configurations with deterministic hashes."""
    if not plans:
        raise ValueError("at least one experiment plan is required")
    entries = []
    for plan in plans:
        encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        entries.append({"experiment_name": plan.get("experiment_name"), "config_sha256": hashlib.sha256(encoded).hexdigest(), "config": plan})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format_version": 1, "experiments": entries}, indent=2, sort_keys=True), encoding="utf-8")
    return path
