"""Diffusion components with a lightweight dataset-only import path."""

from .dataset import GameplayDataset, create_dataloader, create_dataloaders
from .optimizers import create_optimizer

__all__ = ["GameplayDataset", "create_dataloader", "create_dataloaders", "create_optimizer"]

try:
    from .model import (
        ActionConditionedDiffusionModel,
        ActionEmbedding,
        NoiseAugmentationEmbedding,
    )
    __all__ += [
        "ActionConditionedDiffusionModel",
        "ActionEmbedding",
        "NoiseAugmentationEmbedding",
    ]
except ImportError:
    # Diffusers is optional for dataset validation and offline test collection.
    pass
