"""Dependency preflight for collection, training, and inference commands."""

import argparse
import importlib.util
from typing import Iterable

from src.config import load_config


def required_modules(config: dict, mode: str) -> set[str]:
    """Return optional modules required by one concrete execution mode."""
    modules = set()
    environment = config["environment"]["name"]
    if mode in {"collect", "train", "inference"}:
        modules.update({"gymnasium", "torch"})
    if environment == "chrome_dino" and not config["environment"].get("use_mock", False):
        modules.update({"selenium", "cv2", "PIL"})
    if environment == "vizdoom":
        modules.update({"vizdoom", "cv2", "stable_baselines3"})
    if mode in {"train", "inference"}:
        modules.update({"diffusers", "transformers", "tensorboard", "tqdm"})
    return modules


def missing_modules(modules: Iterable[str]) -> list[str]:
    return sorted(name for name in modules if importlib.util.find_spec(name) is None)


def preflight(config: dict, mode: str) -> list[str]:
    if mode not in {"collect", "train", "inference"}:
        raise ValueError("mode must be collect, train, or inference")
    return missing_modules(required_modules(config, mode))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GameNGen runtime dependencies")
    parser.add_argument("--config", default="configs/tier1_chrome_dino.yaml")
    parser.add_argument("--mode", choices=("collect", "train", "inference"), default="train")
    args = parser.parse_args()
    missing = preflight(load_config(args.config), args.mode)
    if missing:
        raise SystemExit(f"Missing runtime dependencies: {', '.join(missing)}")
    print("Preflight passed")


if __name__ == "__main__":
    main()
