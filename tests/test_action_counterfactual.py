import torch
import pytest

from src.utils.action_counterfactual import (
    action_responsiveness_summary,
    generate_action_counterfactuals,
)


class ResponsiveModel(torch.nn.Module):
    num_actions = 4

    def generate(self, frames, actions, **kwargs):
        value = actions[:, -1].to(dtype=frames.dtype).view(-1, 1, 1, 1)
        return value.expand(-1, 3, 2, 2)


def test_counterfactuals_vary_only_current_action_and_restore_training_state():
    model = ResponsiveModel().train()
    frames = torch.zeros(2, 3, 3, 2, 2)
    history = torch.tensor([[1, 2, 3], [0, 1, 2]])
    actions, generated = generate_action_counterfactuals(model, frames, history, [0, 3])
    assert model.training
    assert actions.tolist() == [0, 3]
    assert generated.shape == (2, 2, 3, 2, 2)
    assert generated[0].eq(0).all()
    assert generated[1].eq(3).all()
    summary = action_responsiveness_summary(actions, generated)
    assert summary["action_responsive"] is True
    assert summary["mean_pairwise_l1"] == 3.0


def test_counterfactual_contract_rejects_invalid_or_indistinguishable_inputs():
    model = ResponsiveModel()
    frames = torch.zeros(1, 2, 3, 2, 2)
    history = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="unique"):
        generate_action_counterfactuals(model, frames, history, [1, 1])
    with pytest.raises(ValueError, match="outside"):
        generate_action_counterfactuals(model, frames, history, [4])
    with pytest.raises(ValueError, match="at least two"):
        action_responsiveness_summary(torch.tensor([0]), torch.zeros(1, 1, 3, 2, 2))


def test_counterfactual_summary_detects_control_collapse():
    actions = torch.tensor([0, 1])
    summary = action_responsiveness_summary(actions, torch.zeros(2, 1, 3, 2, 2))
    assert summary["action_responsive"] is False
    assert summary["max_pairwise_l1"] == 0.0
