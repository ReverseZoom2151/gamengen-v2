"""Lightweight conditioning modules shared by training and inference."""

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Discrete action tokens with learned temporal positions."""

    def __init__(self, num_actions: int, embedding_dim: int = 128, max_length: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.position = nn.Embedding(max_length, embedding_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2:
            raise ValueError("actions must have shape (batch, sequence)")
        sequence_length = actions.shape[1]
        if sequence_length > self.position.num_embeddings:
            raise ValueError("action sequence exceeds configured context length")
        positions = torch.arange(sequence_length, device=actions.device)
        return self.embedding(actions) + self.position(positions).unsqueeze(0)


class NoiseAugmentationEmbedding(nn.Module):
    """Embedding for discrete context-noise buckets."""

    def __init__(self, num_buckets: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, noise_levels: torch.Tensor) -> torch.Tensor:
        return self.embedding(noise_levels)
