"""Dependency-free contracts shared by the ViZDoom runtime and unit tests."""

import numpy as np


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
