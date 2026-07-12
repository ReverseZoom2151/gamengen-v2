"""Checkpointable exponential moving averages for diffusion parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch


class ExponentialMovingAverage:
    """Maintain a deterministic EMA over every trainable model parameter."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be strictly between 0 and 1")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if parameters.keys() < self.shadow.keys():
            raise ValueError("model is missing EMA-tracked parameters")
        with torch.no_grad():
            for name, average in self.shadow.items():
                average.mul_(self.decay).add_(parameters[name].detach(), alpha=1.0 - self.decay)
        self.num_updates += 1

    def copy_to(self, model: torch.nn.Module) -> None:
        apply_ema_state(model, self.state_dict())

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        originals = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in originals:
                        parameter.copy_(originals[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping) -> None:
        if not isinstance(state.get("shadow"), Mapping):
            raise ValueError("EMA state is missing shadow parameters")
        shadow = state["shadow"]
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint parameters do not match the model")
        for name, value in shadow.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA parameter shape mismatch for {name}")
            self.shadow[name].copy_(value.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype))
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))


def apply_ema_state(model: torch.nn.Module, state: Mapping) -> None:
    """Copy a serialized EMA into a matching model without touching buffers."""
    shadow = state.get("shadow")
    if not isinstance(shadow, Mapping):
        raise ValueError("EMA state is missing shadow parameters")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name not in shadow:
                raise ValueError(f"EMA state is missing parameter {name}")
            value = shadow[name]
            if value.shape != parameter.shape:
                raise ValueError(f"EMA parameter shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
