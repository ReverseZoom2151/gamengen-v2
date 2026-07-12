"""Dependency-free contracts shared by the ViZDoom runtime and unit tests."""

import numpy as np
import random


def resize_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an HWC RGB frame deterministically without an OpenCV dependency."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("expected an HWC RGB frame")
    if width <= 0 or height <= 0:
        raise ValueError("target dimensions must be positive")
    source_height, source_width, _ = frame.shape
    if (source_width, source_height) == (width, height):
        return frame
    y = np.linspace(0, source_height - 1, height).round().astype(np.intp)
    x = np.linspace(0, source_width - 1, width).round().astype(np.intp)
    return frame[y][:, x]


class DoomObservationHistory:
    """Paper-style map and prior-action observation contract.

    ``observe`` intentionally appends the action before producing the next
    observation.  Therefore observation_t contains actions ending at t - 1,
    while observation_t_plus_1 contains action_t.  This avoids an action
    timing ambiguity between the PPO input and recorded diffusion transition.
    """

    def __init__(
        self,
        num_actions: int,
        action_history_length: int = 32,
        screen_width: int = 160,
        screen_height: int = 120,
        map_width: int = 160,
        map_height: int = 120,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if action_history_length <= 0:
            raise ValueError("action_history_length must be positive")
        self.num_actions = num_actions
        self.action_history_length = action_history_length
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.map_width = map_width
        self.map_height = map_height
        self._history = np.zeros(action_history_length, dtype=np.int64)

    def reset(self) -> None:
        self._history.fill(0)

    def observe(
        self,
        screen: np.ndarray,
        automap: np.ndarray,
        action: int | None = None,
    ) -> dict[str, np.ndarray]:
        if action is not None:
            if not 0 <= int(action) < self.num_actions:
                raise ValueError("action is outside the configured action space")
            self._history[:-1] = self._history[1:]
            self._history[-1] = int(action)
        return {
            "screen": resize_rgb_frame(screen, self.screen_width, self.screen_height),
            "automap": resize_rgb_frame(automap, self.map_width, self.map_height),
            "action_history": self._history.copy(),
        }


class ActionRepeatBias:
    """Deterministically choose whether to retain the last applied action."""

    def __init__(self, repeat_probability: float = 0.0, seed: int | None = None):
        if not 0.0 <= repeat_probability <= 1.0:
            raise ValueError("repeat_probability must be between 0 and 1")
        self.repeat_probability = repeat_probability
        self._rng = random.Random(seed)
        self.previous_action: int | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
        self.previous_action = None

    def resolve(self, requested_action: int) -> tuple[int, bool]:
        repeated = self.previous_action is not None and self._rng.random() < self.repeat_probability
        executed_action = self.previous_action if repeated else int(requested_action)
        self.previous_action = executed_action
        return executed_action, repeated


def pad_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-pad an RGB frame; only resize width when the source differs.

    ViZDoom's paper resolution is 320x240, while the diffusion model consumes
    320x256.  Padding preserves geometry that vertical stretching would alter.
    """
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("expected an HWC RGB frame")
    source_height, source_width, _ = frame.shape
    if source_width != width:
        x = np.linspace(0, source_width - 1, width).round().astype(np.intp)
        frame = frame[:, x]
        source_height = frame.shape[0]
    if source_height > height:
        raise ValueError("cannot pad a frame taller than the requested height")
    top = (height - source_height) // 2
    bottom = height - source_height - top
    return np.pad(frame, ((top, bottom), (0, 0), (0, 0)), mode="constant")


def paper_reward(previous: dict, current: dict, visited_positions: set[tuple[int, int]]) -> float:
    """Compute the configured ten-term reward from named game-variable deltas."""
    reward = 0.0
    health_delta = current["health"] - previous["health"]
    if health_delta < 0: reward -= 100.0
    if current["health"] <= 0: reward -= 5000.0
    reward += 300.0 * max(0, current["hitcount"] - previous["hitcount"])
    reward += 1000.0 * max(0, current["killcount"] - previous["killcount"])
    reward += 100.0 * max(0, current["itemcount"] - previous["itemcount"])
    reward += 500.0 * max(0, current["secretcount"] - previous["secretcount"])
    position = (int(current["position_x"] / 100), int(current["position_y"] / 100))
    if position not in visited_positions:
        visited_positions.add(position)
        reward += 20.0 * (1.0 + 0.5 * (abs(current["position_x"]) + abs(current["position_y"])) / 1000.0)
    reward += 10.0 * health_delta
    reward += 10.0 * (current["armor"] - previous["armor"])
    ammo_delta = current["ammo"] - previous["ammo"]
    return reward + 10.0 * max(0, ammo_delta) + min(0, ammo_delta)


class ScenarioSelector:
    """Deterministic sequential/weighted scenario selection independent of ViZDoom."""

    def __init__(self, scenarios: list[str], weights: list[float] | None = None, seed: int | None = None):
        if not scenarios:
            raise ValueError("at least one scenario is required")
        if weights is not None and (len(weights) != len(scenarios) or any(weight < 0 for weight in weights) or not any(weights)):
            raise ValueError("scenario weights must be non-negative, non-zero, and match scenarios")
        self.scenarios, self.weights, self.index = scenarios, weights, 0
        self.rng = random.Random(seed)

    def reset_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def next(self) -> str:
        if self.weights is None:
            self.index = (self.index + 1) % len(self.scenarios)
        else:
            self.index = self.rng.choices(range(len(self.scenarios)), weights=self.weights, k=1)[0]
        return self.scenarios[self.index]
