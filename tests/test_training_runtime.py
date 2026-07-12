import random

import numpy as np
import torch
import pytest

from src.diffusion.train import make_scheduler, train_microbatch
from src.utils.training import atomic_torch_save, capture_rng_state, restore_rng_state, seed_everything


def test_rng_state_round_trip():
    seed_everything(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_atomic_torch_save_replaces_destination(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 3}, path)
    assert torch.load(path, weights_only=False) == {"step": 3}
    assert not path.with_suffix(".pt.tmp").exists()


def test_linear_scheduler_decays_after_warmup():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = make_scheduler(optimizer, "linear", warmup_steps=2, total_steps=4)
    rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == 1.0
    assert rates[-1] == 0.0


def test_non_finite_microbatch_loss_fails_before_backpropagation():
    class Model:
        device = "cpu"

        def __call__(self, *args):
            return torch.tensor(float("nan"), requires_grad=True)

    batch = {
        "target_frame": torch.zeros(1),
        "context_frames": torch.zeros(1),
        "context_actions": torch.zeros(1),
    }
    with pytest.raises(FloatingPointError, match="non-finite"):
        train_microbatch(Model(), batch, torch.amp.GradScaler("cuda", enabled=False), 1, False)
