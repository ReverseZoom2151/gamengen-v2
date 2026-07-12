import torch

from src.utils.evaluation import GameNGenEvaluator


def test_basic_metrics_work_without_optional_metric_models():
    evaluator = GameNGenEvaluator(device="cpu", enable_lpips=False)
    pred = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    assert evaluator.compute_mse(pred, target) == 1.0
    assert evaluator.compute_psnr(pred, target) > 0
