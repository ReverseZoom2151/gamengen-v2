"""Explicit behavioral failure metrics for gameplay-agent evaluation."""

from __future__ import annotations

import numpy as np


def behavioral_safety_report(actions, positions=None, fired=None, enemy_visible=None, stuck_window: int = 8) -> dict[str, float | int]:
    actions = np.asarray(actions)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("actions must be a non-empty one-dimensional sequence")
    if stuck_window < 2:
        raise ValueError("stuck_window must be at least 2")
    repeats = int(np.sum(actions[1:] == actions[:-1]))
    report: dict[str, float | int] = {"action_repeat_rate": repeats / max(len(actions) - 1, 1), "steps": int(len(actions))}
    if fired is not None and enemy_visible is not None:
        fired, enemy_visible = np.asarray(fired, dtype=bool), np.asarray(enemy_visible, dtype=bool)
        if fired.shape != actions.shape or enemy_visible.shape != actions.shape:
            raise ValueError("fired and enemy_visible must align with actions")
        report["blind_fire_rate"] = float(np.mean(fired & ~enemy_visible))
    if positions is not None:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(actions), 2):
            raise ValueError("positions must have shape (len(actions), 2)")
        movement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        report["stuck_windows"] = int(sum(np.all(movement[start:start + stuck_window - 1] < 1e-6) for start in range(max(0, len(movement) - stuck_window + 2))))
    return report
