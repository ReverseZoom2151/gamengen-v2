"""Diffusion components with a lightweight dataset-only import path."""

from .dataset import GameplayDataset, create_dataloader

__all__ = ["GameplayDataset", "create_dataloader"]

try:
    from .model import (
        ActionConditionedDiffusionModel,
        ActionEmbedding,
        NoiseAugmentationEmbedding,
    )
    from .optimizers import Adafactor, create_optimizer

    __all__ += [
        "ActionConditionedDiffusionModel",
        "ActionEmbedding",
        "NoiseAugmentationEmbedding",
        "Adafactor",
        "create_optimizer",
    ]
except ImportError:
    # Diffusers is optional for dataset validation and offline test collection.
    pass
