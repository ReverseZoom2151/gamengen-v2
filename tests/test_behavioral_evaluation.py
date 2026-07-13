import numpy as np
import pytest

from src.utils.behavior_safety import behavioral_safety_report
from src.utils.behavioral_fidelity import behavioral_fidelity, wasserstein_1d
from src.utils.experiment_plan import modality_ablation, scenario_generalization_plan


def test_behavioral_fidelity_is_zero_for_matching_trajectories():
    trajectory = {"camera_angle": [0, 1, 3, 6], "position": [[0, 0], [1, 0], [1, 1], [2, 1]]}
    assert all(value == 0.0 for value in behavioral_fidelity(trajectory, trajectory).values())
    assert wasserstein_1d(np.array([0]), np.array([2])) == 2.0


def test_safety_flags_blind_fire_and_stuck_windows():
    report = behavioral_safety_report([1, 1, 1, 1], [[0, 0]] * 4, [1, 1, 0, 0], [0, 1, 0, 1], stuck_window=3)
    assert report["action_repeat_rate"] == 1.0
    assert report["blind_fire_rate"] == 0.25
    assert report["stuck_windows"] == 2


def test_generalization_and_modality_plans_are_isolated():
    config = {"experiment_name": "base", "diffusion": {"context_length": 4}}
    plans = scenario_generalization_plan(config, ["basic", "corridor"])
    assert plans[0]["evaluation"]["train_scenarios"] == ["corridor"]
    assert [plan["experiment_name"] for plan in modality_ablation(config, ["rgb", "rgb_depth"])] == ["base-modality-rgb", "base-modality-rgb_depth"]
    with pytest.raises(ValueError):
        scenario_generalization_plan(config, ["basic"])
