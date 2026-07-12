"""
Data Recording Utilities for GameNGen
Records episodes with frames and actions for training diffusion model
"""

import hashlib
import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class EpisodeRecorder:
    """Record isolated environment trajectories as atomic, versioned NPZ shards.

    ``add_transition`` is the canonical API: an action takes ``observation`` to
    ``next_observation``.  ``add_step`` remains as a deprecated compatibility
    adapter for old callers that only stored pre-action observations.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        output_dir: str,
        compress: bool = True,
        save_frequency: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compress = compress
        self.save_frequency = save_frequency

        self._metadata_path = self.output_dir / "metadata.json"
        self._load_existing_state()
        self._episodes: Dict[int, Dict[str, Any]] = {}
        self.batch_buffer: List[Dict[str, Any]] = []

    def _load_existing_state(self) -> None:
        metadata: Dict[str, Any] = {}
        if self._metadata_path.exists():
            with self._metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        shard_ids = []
        for path in self.output_dir.glob("shard_*.npz"):
            try:
                shard_ids.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue

        self.episode_count = int(metadata.get("total_episodes", 0))
        self.total_frames = int(metadata.get("total_frames", 0))
        self.next_shard_id = max(shard_ids, default=-1) + 1
        self.shard_checksums: Dict[str, str] = dict(metadata.get("shard_checksums", {}))

    @staticmethod
    def _new_episode(observation: np.ndarray, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "frames": [np.asarray(observation, dtype=np.uint8)],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "metadata": dict(metadata or {}),
        }

    def add_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool = False,
        truncated: bool = False,
        env_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one canonical ``observation_t + action_t -> observation_t+1`` transition."""

        episode = self._episodes.setdefault(env_id, self._new_episode(observation, metadata))
        if metadata:
            episode["metadata"].update(metadata)

        episode["actions"].append(int(action))
        episode["rewards"].append(float(reward))
        episode["terminated"].append(bool(terminated))
        episode["truncated"].append(bool(truncated))
        episode["frames"].append(np.asarray(next_observation, dtype=np.uint8))

        if terminated or truncated:
            self.finish_episode(env_id)

    def add_step(
        self, frame: np.ndarray, action: int, reward: float, done: bool, env_id: int = 0
    ) -> None:
        """Deprecated compatibility adapter for pre-transition recording callers."""

        warnings.warn(
            "EpisodeRecorder.add_step is deprecated; use add_transition so action/frame timing is explicit.",
            DeprecationWarning,
            stacklevel=2,
        )
        episode = self._episodes.get(env_id)
        if episode is None:
            episode = self._new_episode(frame, None)
            self._episodes[env_id] = episode
        else:
            episode["frames"].append(np.asarray(frame, dtype=np.uint8))
        episode["actions"].append(int(action))
        episode["rewards"].append(float(reward))
        episode["terminated"].append(bool(done))
        episode["truncated"].append(False)
        if done:
            episode["frames"].append(np.asarray(frame, dtype=np.uint8))
            self.finish_episode(env_id)

    def finish_episode(self, env_id: int = 0, force: bool = False) -> None:
        """Finish one environment's episode without affecting other environments."""

        episode = self._episodes.pop(env_id, None)
        if episode is None or not episode["actions"]:
            return

        if len(episode["frames"]) != len(episode["actions"]) + 1:
            if not force:
                raise ValueError("canonical episodes require one more frame than actions")
            episode["frames"].append(episode["frames"][-1].copy())

        episode["episode_id"] = self.episode_count
        episode["env_id"] = env_id
        episode["length"] = len(episode["actions"])
        self.batch_buffer.append(episode)
        self.episode_count += 1
        self.total_frames += len(episode["frames"])

        if len(self.batch_buffer) >= self.save_frequency:
            self._save_batch()

    def _save_batch(self):
        """Save batch of episodes to disk"""
        if len(self.batch_buffer) == 0:
            return

        shard_id = self.next_shard_id
        filename = self.output_dir / f"shard_{shard_id:06d}.npz"
        arrays: Dict[str, np.ndarray] = {}
        manifest = {"schema_version": self.SCHEMA_VERSION, "episodes": []}

        for index, episode in enumerate(self.batch_buffer):
            prefix = f"episode_{index:06d}"
            arrays[f"{prefix}_frames"] = np.stack(episode["frames"]).astype(np.uint8, copy=False)
            arrays[f"{prefix}_actions"] = np.asarray(episode["actions"], dtype=np.int32)
            arrays[f"{prefix}_rewards"] = np.asarray(episode["rewards"], dtype=np.float32)
            arrays[f"{prefix}_terminated"] = np.asarray(episode["terminated"], dtype=np.bool_)
            arrays[f"{prefix}_truncated"] = np.asarray(episode["truncated"], dtype=np.bool_)
            manifest["episodes"].append(
                {
                    "prefix": prefix,
                    "episode_id": episode["episode_id"],
                    "env_id": episode["env_id"],
                    "length": episode["length"],
                    "metadata": episode["metadata"],
                }
            )

        arrays["_manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
        with tempfile.NamedTemporaryFile(dir=self.output_dir, suffix=".npz", delete=False) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays) if self.compress else np.savez(handle, **arrays)
        os.replace(temporary_path, filename)
        checksum = hashlib.sha256(filename.read_bytes()).hexdigest()
        self.shard_checksums[filename.name] = checksum
        self.next_shard_id += 1

        print(f"Saved shard {shard_id} with {len(self.batch_buffer)} episodes to {filename}")

        # Clear buffer
        self.batch_buffer = []

    def finalize(self):
        """Save remaining data and create metadata"""
        # Explicitly flush unfinished episodes rather than silently losing them.
        for env_id in list(self._episodes):
            self.finish_episode(env_id, force=True)

        # Save remaining episodes
        if len(self.batch_buffer) > 0:
            self._save_batch()

        # Create metadata
        metadata = {
            "total_episodes": self.episode_count,
            "total_frames": self.total_frames,
            "schema_version": self.SCHEMA_VERSION,
            "episodes_per_shard": self.save_frequency,
            "compressed": self.compress,
            "next_shard_id": self.next_shard_id,
            "shard_checksums": self.shard_checksums,
            "format": "npz",
        }

        with tempfile.NamedTemporaryFile(dir=self.output_dir, suffix=".json", mode="w", delete=False) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, self._metadata_path)

        print(f"\nRecording complete!")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total frames: {self.total_frames}")
        print(f"Metadata saved to: {self._metadata_path}")


def load_npz_shard(path: Path) -> List[Dict[str, Any]]:
    """Load a versioned NPZ recorder shard without pickle deserialization."""

    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["_manifest"].item()))
        if manifest.get("schema_version") != EpisodeRecorder.SCHEMA_VERSION:
            raise ValueError(f"unsupported recorder schema in {path}")
        episodes = []
        for item in manifest["episodes"]:
            prefix = item["prefix"]
            episode = {
                "frames": data[f"{prefix}_frames"].copy(),
                "actions": data[f"{prefix}_actions"].copy(),
                "rewards": data[f"{prefix}_rewards"].copy(),
                "terminated": data[f"{prefix}_terminated"].copy(),
                "truncated": data[f"{prefix}_truncated"].copy(),
                "episode_id": item["episode_id"],
                "env_id": item["env_id"],
                "length": item["length"],
                "metadata": item.get("metadata", {}),
            }
            episodes.append(episode)
    return episodes


class DatasetLoader:
    """Load versioned NPZ recordings, with warned support for legacy pickles."""

    def __init__(self, data_dir: str, allow_legacy_pickle: bool = True):
        self.data_dir = Path(data_dir)

        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
            self.metadata = {}

        self.shard_files = sorted(self.data_dir.glob("shard_*.npz"))
        self.batch_files = sorted(self.data_dir.glob("batch_*.pkl"))
        if self.batch_files and not allow_legacy_pickle:
            raise ValueError("legacy pickle recordings require allow_legacy_pickle=True")
        if self.batch_files:
            warnings.warn("loading legacy pickle recordings; migrate them before training", RuntimeWarning)
        print(f"Found {len(self.shard_files)} NPZ shards and {len(self.batch_files)} legacy batches")

    def load_batch(self, batch_idx: int) -> List[Dict[str, Any]]:
        sources = self.shard_files + self.batch_files
        if batch_idx >= len(sources):
            raise IndexError(f"batch {batch_idx} not found")
        path = sources[batch_idx]
        if path.suffix == ".npz":
            return load_npz_shard(path)
        with path.open("rb") as handle:
            return pickle.load(handle)

    def iter_episodes(self, shuffle: bool = False):
        batch_indices = list(range(len(self.shard_files) + len(self.batch_files)))
        if shuffle:
            np.random.shuffle(batch_indices)
        for batch_idx in batch_indices:
            batch = self.load_batch(batch_idx)
            if shuffle:
                np.random.shuffle(batch)
            yield from batch


def visualize_episode(episode: Dict[str, Any], output_path: Optional[str] = None):
    """Visualize an episode as video"""
    import cv2
    import imageio

    frames = episode["frames"]
    actions = episode["actions"]

    # Add action labels to frames
    labeled_frames = []
    action_names = ["No Action", "Jump", "Duck"]

    for frame, action in zip(frames, actions):
        # Copy frame
        frame_labeled = frame.copy()

        # Add action text
        cv2.putText(
            frame_labeled,
            f"Action: {action_names[action]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        labeled_frames.append(frame_labeled)

    # Save as video if path provided
    if output_path:
        imageio.mimsave(output_path, labeled_frames, fps=20)
        print(f"Saved video to {output_path}")

    return labeled_frames


if __name__ == "__main__":
    # Test recorder
    print("Testing DataRecorder...")

    output_dir = "data/test_recordings"
    recorder = EpisodeRecorder(output_dir, save_frequency=2)

    # Simulate recording 5 episodes
    for ep in range(5):
        print(f"\nEpisode {ep + 1}")
        for step in range(50):
            # Dummy data
            frame = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            done = step == 49

            recorder.add_step(frame, action, reward, done)

    recorder.finalize()

    # Test loader
    print("\n" + "=" * 50)
    print("Testing DataLoader...")

    loader = DatasetLoader(output_dir)
    print(f"Metadata: {loader.metadata}")

    # Load first episode
    batch = loader.load_batch(0)
    print(f"First batch has {len(batch)} episodes")
    print(f"First episode shape: {batch[0]['frames'].shape}")

    print("\nTest complete!")
