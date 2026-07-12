"""Diffusion components with a lightweight dataset-only import path."""

from .dataset import GameplayDataset, create_dataloader, create_dataloaders
from .conditioning import ActionEmbedding, NoiseAugmentationEmbedding
from .optimizers import create_optimizer
from .ema import ExponentialMovingAverage

__all__ = [
    "ActionEmbedding", "GameplayDataset", "NoiseAugmentationEmbedding",
    "create_dataloader", "create_dataloaders", "create_optimizer", "ExponentialMovingAverage",
]

try:
    from .model import (
        ActionConditionedDiffusionModel,
    )
    __all__ += [
        "ActionConditionedDiffusionModel",
    ]
except ImportError:
    # Diffusers is optional for dataset validation and offline test collection.
    pass
