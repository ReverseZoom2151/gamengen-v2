"""Configuration-driven runtime environment construction."""

from __future__ import annotations

from typing import Mapping


def interactive_environment_spec(config: Mapping) -> tuple[str, dict, dict[int, int]]:
    """Return lazy-construction arguments and OpenCV key-to-action bindings."""
    environment = config["environment"]
    resolution = environment["resolution"]
    name = environment["name"]
    if name == "chrome_dino":
        return name, {
            "width": resolution["width"],
            "height": resolution["height"],
            "grayscale": bool(environment.get("grayscale", False)),
            "frame_skip": int(environment.get("action_repeat", 1)),
        }, {ord(" "): 1, 82: 1, 84: 2}
    if name == "vizdoom":
        return name, {
            "scenario": environment.get("scenario", environment.get("config_file", "basic")),
            "width": resolution["width"],
            "height": resolution["height"],
            "frame_skip": int(environment.get("action_repeat", 4)),
            "use_paper_reward": bool(config.get("use_paper_reward", config.get("agent", {}).get("reward_function") == "paper_doom")),
            "visible": False,
            # The interactive neural-engine buffer needs raw RGB frames; PPO's
            # map/history observation is intentionally not part of this context.
            "include_automap": False,
        }, {ord(" "): 7, 82: 1, 84: 2, 81: 5, 83: 6, ord("a"): 3, ord("d"): 4}
    raise ValueError(f"interactive inference does not support environment {name!r}")


def create_interactive_environment(config: Mapping):
    """Instantiate a real configured game environment only when requested."""
    name, kwargs, _ = interactive_environment_spec(config)
    if name == "chrome_dino":
        from src.environment.chrome_dino_env import ChromeDinoEnv

        return ChromeDinoEnv(**kwargs)
    if name == "vizdoom":
        from src.environment.vizdoom_env import create_vizdoom_env

        return create_vizdoom_env(**kwargs)
    raise AssertionError("validated environment name was not handled")
