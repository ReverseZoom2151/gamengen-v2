import torch
import pytest

from src.utils.evaluation import (
    GameNGenEvaluator,
    evaluate_sampling_step_sweep,
    evaluate_model_comprehensive,
    save_evaluation_report,
)


def test_basic_metrics_work_without_optional_metric_models():
    evaluator = GameNGenEvaluator(device="cpu", enable_lpips=False)
    pred = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    assert evaluator.compute_mse(pred, target) == 1.0
    assert evaluator.compute_psnr(pred, target) > 0


def test_batch_ssim_uses_each_frame_and_lpips_can_be_disabled(monkeypatch):
    evaluator = GameNGenEvaluator(device="cpu", enable_lpips=False)
    observed = []
    monkeypatch.setattr(evaluator, "compute_ssim", lambda pred, target: observed.append(pred) or 0.5)
    metrics = evaluator.compute_all_metrics(torch.zeros(3, 3, 2, 2), torch.ones(3, 3, 2, 2))
    assert metrics["ssim"] == 0.5
    assert len(observed) == 3
    assert "lpips" not in metrics


def test_evaluation_report_is_atomic_and_provenance_linked(tmp_path):
    path = save_evaluation_report(tmp_path / "report.json", {"psnr": 30.0}, {"config_sha256": "abc"})
    assert path.is_file()
    assert "config_sha256" in path.read_text()
    assert not (tmp_path / "report.json.tmp").exists()


def test_comprehensive_evaluation_rejects_empty_loader_without_model_work():
    class Model:
        def eval(self):
            return self

    with pytest.raises(ValueError, match="no trajectories"):
        evaluate_model_comprehensive(Model(), [], device="cpu", enable_lpips=False)


def test_sampling_sweep_uses_fixed_batches_and_restores_model_mode():
    class Model(torch.nn.Module):
        def generate(self, frames, actions, num_inference_steps, guidance_scale):
            return torch.full_like(frames[:, -1], float(num_inference_steps))

    model = Model().train()
    batch = {
        "context_frames": torch.zeros(2, 2, 3, 2, 2),
        "context_actions": torch.zeros(2, 2, dtype=torch.long),
        "target_frame": torch.zeros(2, 3, 2, 2),
    }
    report = evaluate_sampling_step_sweep(model, [batch], [1, 4], max_batches=1)
    assert model.training
    assert report["num_batches"] == 1
    assert [setting["mse"] for setting in report["settings"]] == [1.0, 16.0]
    assert all(setting["frames_per_second"] > 0 for setting in report["settings"])


def test_sampling_sweep_rejects_empty_or_ambiguous_settings():
    with pytest.raises(ValueError, match="positive"):
        evaluate_sampling_step_sweep(None, [], [0])
    with pytest.raises(ValueError, match="unique"):
        evaluate_sampling_step_sweep(None, [], [1, 1])
