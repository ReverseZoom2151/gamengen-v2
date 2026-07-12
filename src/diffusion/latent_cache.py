"""Safe, provenance-linked latent caches derived from recording shards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from src.utils.data_recorder import load_npz_shard, shard_checksum


LATENT_CACHE_SCHEMA_VERSION = 1


def build_latent_cache(
    recording_shard: str | Path,
    destination: str | Path,
    encode_frames: Callable[[np.ndarray], np.ndarray],
    *,
    base_model: str,
    base_model_revision: str,
) -> Path:
    """Encode one validated recording shard and atomically write its latents.

    ``encode_frames`` receives one episode's uint8 HWC frames and must return a
    numeric array with the same leading frame dimension. Actions and transition
    fields are copied verbatim, preserving the canonical target/action timing.
    """
    source = Path(recording_shard)
    target = Path(destination)
    episodes = load_npz_shard(source)
    arrays, manifest_episodes = {}, []
    for index, episode in enumerate(episodes):
        prefix = f"episode_{index:06d}"
        latents = np.asarray(encode_frames(episode["frames"]))
        if latents.ndim < 2 or len(latents) != len(episode["frames"]) or not np.issubdtype(latents.dtype, np.number):
            raise ValueError("encoder must return numeric latents with one entry per source frame")
        arrays[f"{prefix}_latents"] = latents
        for field in ("actions", "rewards", "terminated", "truncated"):
            arrays[f"{prefix}_{field}"] = episode[field]
        manifest_episodes.append({"prefix": prefix, "episode_id": episode["episode_id"], "env_id": episode["env_id"], "length": episode["length"], "metadata": episode["metadata"]})
    manifest = {"schema_version": LATENT_CACHE_SCHEMA_VERSION, "source_shard": source.name, "source_sha256": shard_checksum(source), "base_model": base_model, "base_model_revision": base_model_revision, "episodes": manifest_episodes}
    arrays["_manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, target)
    return target


def validate_latent_cache(path: str | Path, recording_shard: str | Path | None = None) -> dict:
    """Validate schema, source provenance, and latent/transition alignment."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "_manifest" not in data:
            raise ValueError("latent cache has no manifest")
        manifest = json.loads(str(data["_manifest"].item()))
        if manifest.get("schema_version") != LATENT_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported latent cache schema")
        if recording_shard is not None and manifest.get("source_sha256") != shard_checksum(Path(recording_shard)):
            raise ValueError("latent cache source checksum mismatch")
        for episode in manifest.get("episodes", []):
            prefix, length = episode.get("prefix"), episode.get("length")
            required = [f"{prefix}_{name}" for name in ("latents", "actions", "rewards", "terminated", "truncated")]
            if not prefix or any(name not in data for name in required):
                raise ValueError("latent cache is missing episode arrays")
            if len(data[f"{prefix}_latents"]) != int(length) + 1 or len(data[f"{prefix}_actions"]) != int(length):
                raise ValueError("latent cache transition alignment is invalid")
    return manifest
