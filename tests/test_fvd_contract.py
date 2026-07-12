import pytest
import torch

from src.utils.fvd import FVDCalculator, evaluate_fvd_on_trajectories


def test_fvd_rejects_invalid_self_comparison_helper():
    with pytest.raises(NotImplementedError, match="cannot produce valid FVD"):
        evaluate_fvd_on_trajectories(None, None)


def test_fvd_requires_multiple_paired_videos_without_initializing_i3d():
    calculator = object.__new__(FVDCalculator)
    with pytest.raises(ValueError, match="at least two"):
        FVDCalculator.compute_fvd(calculator, torch.zeros(1, 1), torch.zeros(1, 1))
