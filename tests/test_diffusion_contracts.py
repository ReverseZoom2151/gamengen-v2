import torch

from src.diffusion.contracts import velocity_target


def test_velocity_target_matches_endpoints_and_manual_formula():
    clean = torch.tensor([[[[2.0]]], [[[3.0]]]])
    noise = torch.tensor([[[[5.0]]], [[[7.0]]]])
    result = velocity_target(
        clean, noise, torch.tensor([1.0, 0.25, 0.0]), torch.tensor([0, 1])
    )
    assert torch.allclose(result[0], noise[0])
    expected = torch.tensor([[[[0.5 * 7 - (0.75**0.5) * 3]]]])
    assert torch.allclose(result[1], expected)
