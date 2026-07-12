import torch

from src.diffusion.hierarchical_memory import MemoryCompressor


def test_memory_compressor_accepts_non_8x8_latent_maps():
    compressor = MemoryCompressor(latent_dim=4, compressed_dim=12, hidden_dim=16)
    output = compressor(torch.randn(2, 3, 4, 40, 64))
    assert output.shape == (2, 12)
