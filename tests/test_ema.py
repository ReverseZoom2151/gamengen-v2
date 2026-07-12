import torch
import pytest

from src.diffusion.ema import ExponentialMovingAverage, apply_ema_state


def test_ema_updates_serializes_and_restores_parameters():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    ema = ExponentialMovingAverage(model, decay=0.75)
    model.weight.data.fill_(6.0)
    ema.update(model)
    assert ema.shadow["weight"].item() == pytest.approx(3.0)

    restored = torch.nn.Linear(1, 1, bias=False)
    restored_ema = ExponentialMovingAverage(restored, decay=0.5)
    restored_ema.load_state_dict(ema.state_dict())
    restored_ema.copy_to(restored)
    assert restored.weight.item() == pytest.approx(3.0)
    assert restored_ema.num_updates == 1


def test_ema_context_restores_training_parameters_after_evaluation():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    model.weight.data.fill_(6.0)
    ema.update(model)
    with ema.average_parameters(model):
        assert model.weight.item() == pytest.approx(4.0)
    assert model.weight.item() == pytest.approx(6.0)


def test_ema_rejects_incompatible_state():
    model = torch.nn.Linear(1, 1, bias=False)
    ema = ExponentialMovingAverage(model)
    with pytest.raises(ValueError, match="missing"):
        apply_ema_state(model, {})
    with pytest.raises(ValueError, match="strictly"):
        ExponentialMovingAverage(model, decay=1.0)
