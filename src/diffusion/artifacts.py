"""Validation helpers for trusted GameNGen model checkpoint artifacts."""

from typing import Mapping


MODEL_COMPONENTS = ("unet", "action_embedding", "noise_aug_embedding", "action_proj")


def model_state_from_checkpoint(checkpoint: Mapping) -> Mapping:
    """Return a complete model state, accepting the documented v1/v2 layouts."""
    state = checkpoint.get("model", checkpoint)
    missing = [name for name in MODEL_COMPONENTS if name not in state]
    if missing:
        raise ValueError(f"checkpoint is missing model components: {', '.join(missing)}")
    return state
