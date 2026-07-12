"""Mathematical contracts independent of heavyweight model dependencies."""

import torch


def velocity_target(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return v-prediction target ``sqrt(alpha)*eps - sqrt(1-alpha)*x0``."""
    alphas = alphas_cumprod.to(
        device=clean_latents.device, dtype=clean_latents.dtype
    )[timesteps]
    shape = (-1,) + (1,) * (clean_latents.ndim - 1)
    return alphas.sqrt().view(shape) * noise - (1 - alphas).sqrt().view(shape) * clean_latents
