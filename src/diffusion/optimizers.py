"""Optimizer construction for diffusion training."""

from typing import Iterable

import torch


def create_optimizer(optimizer_name: str, parameters: Iterable[torch.nn.Parameter], config: dict) -> torch.optim.Optimizer:
    """Build the configured optimizer with explicit, reproducible settings.

    The paper profile uses the maintained Transformers Adafactor implementation.
    Keeping the implementation in its upstream package avoids silently training
    with a partial local reimplementation.
    """

    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as error:
            raise RuntimeError(
                "Adafactor requires the optional 'transformers' dependency. "
                "Install the diffusion extra before using the paper profile."
            ) from error
        return Adafactor(
            parameters,
            lr=config["learning_rate"],
            weight_decay=0.0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"unknown optimizer: {optimizer_name}")
