"""Action-responsiveness probes for action-conditioned game generators."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

from src.diffusion.conditioning import condition_current_action


@torch.no_grad()
def generate_action_counterfactuals(
    model,
    context_frames: torch.Tensor,
    context_actions: torch.Tensor,
    actions: Iterable[int],
    **generation_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one next frame per possible current action.

    Every sample receives identical context frames and action history except the
    final (current) action token. The returned tensor is ordered by ``actions``
    and has shape ``[actions, batch, channels, height, width]``.
    """
    if context_frames.ndim != 5:
        raise ValueError("context_frames must have shape [batch, time, channels, height, width]")
    if context_actions.ndim != 2 or context_actions.shape[:2] != context_frames.shape[:2]:
        raise ValueError("context_actions must align with the batch and time dimensions")
    action_tensor = torch.as_tensor(list(actions), dtype=torch.long, device=context_actions.device)
    if action_tensor.numel() == 0:
        raise ValueError("at least one counterfactual action is required")
    if action_tensor.unique().numel() != action_tensor.numel():
        raise ValueError("counterfactual actions must be unique")
    if torch.any(action_tensor < 0) or torch.any(action_tensor >= model.num_actions):
        raise ValueError("counterfactual action is outside the model action space")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        outputs = [
            model.generate(
                context_frames,
                condition_current_action(context_actions, int(action)),
                **generation_kwargs,
            )
            for action in action_tensor.tolist()
        ]
    finally:
        model.train(was_training)
    return action_tensor.detach().cpu(), torch.stack(outputs).detach().cpu()


def action_responsiveness_summary(actions: torch.Tensor, generated_frames: torch.Tensor) -> Mapping[str, float | int | bool]:
    """Summarize whether a counterfactual grid changes under different actions."""
    if actions.ndim != 1 or generated_frames.ndim != 5:
        raise ValueError("expected actions [A] and generated_frames [A, B, C, H, W]")
    if len(actions) != len(generated_frames):
        raise ValueError("counterfactual action count does not match generated frames")
    if len(actions) < 2:
        raise ValueError("at least two actions are required to assess responsiveness")
    distances = [
        torch.mean(torch.abs(generated_frames[left] - generated_frames[right])).item()
        for left in range(len(actions))
        for right in range(left + 1, len(actions))
    ]
    return {
        "num_actions": int(len(actions)),
        "mean_pairwise_l1": float(sum(distances) / len(distances)),
        "min_pairwise_l1": float(min(distances)),
        "max_pairwise_l1": float(max(distances)),
        "action_responsive": bool(max(distances) > 0.0),
    }
