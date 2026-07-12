import torch

from src.utils.evaluation import GameNGenEvaluator, save_evaluation_report


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
