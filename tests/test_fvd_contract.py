import pytest
import torch

from src.utils.fvd import FVDCalculator, evaluate_fvd_on_trajectories
from src.utils.human_eval import HumanEvaluationFramework


def test_fvd_rejects_invalid_self_comparison_helper():
    with pytest.raises(NotImplementedError, match="cannot produce valid FVD"):
        evaluate_fvd_on_trajectories(None, None)


def test_fvd_requires_multiple_paired_videos_without_initializing_i3d():
    calculator = object.__new__(FVDCalculator)
    with pytest.raises(ValueError, match="at least two"):
        FVDCalculator.compute_fvd(calculator, torch.zeros(1, 1), torch.zeros(1, 1))


def test_human_protocol_keeps_answer_key_out_of_public_file(tmp_path):
    framework = HumanEvaluationFramework(str(tmp_path))
    framework.create_evaluation_clips(["real.mp4"], ["fake.mp4"], num_clips_per_length=1)
    framework.save_evaluation_protocol()
    assert "real_is_on_left" not in (tmp_path / "evaluation_protocol.json").read_text()
    assert (tmp_path / "answer_key.json").exists()
