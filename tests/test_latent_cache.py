import numpy as np
import pytest

from src.diffusion.latent_cache import build_latent_cache, validate_latent_cache
from src.utils.data_recorder import EpisodeRecorder


def test_latent_cache_preserves_transition_alignment_and_source_provenance(tmp_path):
    recorder = EpisodeRecorder(str(tmp_path / "recordings"), save_frequency=1)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder.add_transition(frame, 1, 0.0, frame + 1, terminated=True)
    recorder.finalize()
    shard = next((tmp_path / "recordings").glob("shard_*.npz"))
    cache = build_latent_cache(shard, tmp_path / "latents.npz", lambda frames: frames.mean(axis=-1, keepdims=True), base_model="model", base_model_revision="a" * 40)
    manifest = validate_latent_cache(cache, shard)
    assert manifest["base_model_revision"] == "a" * 40
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_latent_cache(cache, shard)


def test_latent_cache_rejects_misaligned_encoder_output(tmp_path):
    recorder = EpisodeRecorder(str(tmp_path / "recordings"), save_frequency=1)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder.add_transition(frame, 1, 0.0, frame, terminated=True)
    recorder.finalize()
    shard = next((tmp_path / "recordings").glob("shard_*.npz"))
    with pytest.raises(ValueError, match="one entry"):
        build_latent_cache(shard, tmp_path / "bad.npz", lambda frames: np.zeros((1, 2)), base_model="model", base_model_revision="a" * 40)
