"""
PyTorch Dataset for GameNGen
Loads recorded gameplay episodes and creates training samples
"""

import pickle
import warnings
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.utils.data_recorder import load_npz_shard, validate_npz_shard


class GameplayDataset(Dataset):
    """
    Dataset for training diffusion model on gameplay recordings

    Each sample contains:
    - context_frames: (context_length, 3, H, W) - past frames
    - context_actions: (context_length,) - past actions
    - target_frame: (3, H, W) - next frame to predict
    - target_action: (,) - action taken at target frame
    """

    def __init__(
        self,
        data_dir: str,
        context_length: int = 32,
        resolution: Tuple[int, int] = (256, 512),  # (H, W)
        max_trajectories: Optional[int] = None,
        cache_in_memory: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing recorded episodes
            context_length: Number of past frames to use as context
            resolution: Target resolution (H, W)
            max_trajectories: Maximum number of trajectories to load
            cache_in_memory: Whether to cache all data in RAM
        """
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.resolution = resolution
        self.max_trajectories = max_trajectories
        self.cache_in_memory = cache_in_memory

        # Load metadata if available
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata: {self.metadata}")
        else:
            self.metadata = {}

        # New recordings use safe, versioned NPZ shards. Legacy pickle batches are
        # supported only to make existing local experiments migratable.
        self.shard_files = sorted(self.data_dir.glob("shard_*.npz"))
        self.batch_files = sorted(self.data_dir.glob("batch_*.pkl"))
        self.source_files = self.shard_files + self.batch_files

        if len(self.source_files) == 0:
            raise ValueError(f"No NPZ shards or legacy batch files found in {self.data_dir}")
        if self.batch_files:
            warnings.warn("using legacy pickle recordings; migrate to NPZ shards", RuntimeWarning)
        checksums = self.metadata.get("shard_checksums", {})
        for shard in self.shard_files:
            validate_npz_shard(shard, checksums.get(shard.name))

        print(f"Found {len(self.shard_files)} NPZ shards and {len(self.batch_files)} legacy batches")

        # Create trajectory index
        print("Creating trajectory index...")
        self.trajectories = self._create_trajectory_index()

        print(f"Dataset created with {len(self.trajectories)} trajectories")

        # Cache data if requested
        self.cache = {}
        if cache_in_memory:
            print("Caching data in memory...")
            self._cache_all_data()

    def _create_trajectory_index(self) -> List[Dict]:
        """
        Create index of all valid trajectories across all batches

        Returns:
            List of trajectory descriptors
        """
        trajectories = []

        for batch_idx, batch_file in enumerate(tqdm(self.source_files, desc="Indexing")):
            # Load batch
            batch = self._load_source(batch_idx)

            # Process each episode in batch
            for episode_idx, episode in enumerate(batch):
                frames = episode["frames"]
                actions = episode["actions"]

                # Create trajectories from this episode
                # Need at least context_length + 1 frames
                if len(frames) < self.context_length + 1:
                    continue

                # Create one trajectory for each valid position
                for start_idx in range(len(frames) - self.context_length):
                    trajectory = {
                        "batch_idx": batch_idx,
                        "batch_file": str(batch_file),
                        "episode_idx": episode_idx,
                        "start_idx": start_idx,
                        "episode_id": episode.get("episode_id", -1),
                    }
                    trajectories.append(trajectory)

                    # Stop if we've reached max trajectories
                    if (
                        self.max_trajectories
                        and len(trajectories) >= self.max_trajectories
                    ):
                        return trajectories

        return trajectories

    def _load_episode(self, batch_idx: int, episode_idx: int) -> Dict:
        """Load specific episode from disk"""
        # Check cache first
        cache_key = (batch_idx, episode_idx)
        if cache_key in self.cache:
            return self.cache[cache_key]

        batch = self._load_source(batch_idx)

        episode = batch[episode_idx]

        # Cache if enabled
        if self.cache_in_memory:
            self.cache[cache_key] = episode

        return episode

    def _load_source(self, source_idx: int) -> List[Dict]:
        """Load one recorder source, preferring the safe NPZ schema."""

        path = self.source_files[source_idx]
        if path.suffix == ".npz":
            return load_npz_shard(path, self.metadata.get("shard_checksums", {}).get(path.name))
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _cache_all_data(self):
        """Cache all episodes in memory"""
        for traj in tqdm(self.trajectories, desc="Caching"):
            batch_idx = traj["batch_idx"]
            episode_idx = traj["episode_idx"]
            cache_key = (batch_idx, episode_idx)

            if cache_key not in self.cache:
                episode = self._load_episode(batch_idx, episode_idx)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame

        Args:
            frame: (H, W, 3) numpy array, values in [0, 255]
        Returns:
            (3, H, W) torch tensor, values in [0, 255] float32
        """
        if frame.ndim != 3:
            raise ValueError(f"expected a 3D frame, received shape {frame.shape}")
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.moveaxis(frame, 0, -1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            raise ValueError(f"expected RGB frame, received shape {frame.shape}")

        # Convert to tensor
        frame = torch.from_numpy(frame).float()

        # Permute to (C, H, W)
        frame = frame.permute(2, 0, 1)

        # Resize if needed
        if frame.shape[1:] != self.resolution:
            import torchvision.transforms.functional as TF

            frame = TF.resize(
                frame.unsqueeze(0), self.resolution, antialias=True
            ).squeeze(0)

        return frame

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single training sample

        Returns:
            dict with:
            - context_frames: (context_length, 3, H, W)
            - context_actions: (context_length,)
            - target_frame: (3, H, W)
            - target_action: int
        """
        # Get trajectory info
        traj = self.trajectories[idx]

        # Load episode
        episode = self._load_episode(traj["batch_idx"], traj["episode_idx"])

        frames = episode["frames"]
        actions = episode["actions"]

        start_idx = traj["start_idx"]
        end_idx = start_idx + self.context_length

        # Get context frames and actions
        context_frames = frames[start_idx:end_idx]
        context_actions = actions[start_idx:end_idx]

        # Get target frame and action
        target_frame = frames[end_idx]
        # The final context action produces the target frame.  Legacy datasets
        # can contain an action at end_idx; it is deliberately not used here.
        target_action = actions[end_idx - 1]

        # Preprocess frames
        context_frames = torch.stack(
            [self._preprocess_frame(frame) for frame in context_frames]
        )  # (T, 3, H, W)

        target_frame = self._preprocess_frame(target_frame)  # (3, H, W)

        # Convert actions to tensor
        context_actions = torch.from_numpy(np.array(context_actions, dtype=np.int64))

        target_action = torch.tensor(target_action, dtype=torch.long)

        return {
            "context_frames": context_frames,
            "context_actions": context_actions,
            "target_frame": target_frame,
            "target_action": target_action,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    context_length: int = 32,
    resolution: Tuple[int, int] = (256, 512),
    num_workers: int = 4,
    shuffle: bool = True,
    max_trajectories: Optional[int] = None,
    cache_in_memory: bool = False,
) -> DataLoader:
    """
    Create dataloader for training

    Args:
        data_dir: Directory with recorded episodes
        batch_size: Batch size
        context_length: Number of past frames
        resolution: Target resolution (H, W)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_trajectories: Maximum number of trajectories
        cache_in_memory: Cache all data in RAM

    Returns:
        DataLoader
    """
    dataset = GameplayDataset(
        data_dir=data_dir,
        context_length=context_length,
        resolution=resolution,
        max_trajectories=max_trajectories,
        cache_in_memory=cache_in_memory,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return dataloader


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    context_length: int = 32,
    resolution: Tuple[int, int] = (256, 512),
    num_workers: int = 4,
    validation_fraction: float = 0.05,
    seed: int = 0,
    split_manifest_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create deterministic episode-disjoint train and validation loaders."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    dataset = GameplayDataset(data_dir, context_length, resolution)
    by_episode: Dict[Tuple[int, int], List[int]] = {}
    for index, trajectory in enumerate(dataset.trajectories):
        by_episode.setdefault((trajectory["batch_idx"], trajectory["episode_idx"]), []).append(index)
    episode_keys = sorted(by_episode)
    if len(episode_keys) < 2:
        raise ValueError("at least two recorded episodes are required for a train/validation split")
    validation_keys, train_keys = _episode_split(
        episode_keys, validation_fraction, seed, split_manifest_path
    )
    validation_indices = [index for key in validation_keys for index in by_episode[key]]
    train_indices = [index for key in train_keys for index in by_episode[key]]
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    return (
        DataLoader(Subset(dataset, train_indices), shuffle=True, generator=generator, **common),
        DataLoader(Subset(dataset, validation_indices), shuffle=False, **common),
    )


def _episode_split(
    episode_keys: List[Tuple[int, int]],
    validation_fraction: float,
    seed: int,
    split_manifest_path: Optional[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load or atomically persist a deterministic episode-level split."""
    all_keys = {f"{batch}:{episode}" for batch, episode in episode_keys}
    if split_manifest_path and Path(split_manifest_path).is_file():
        manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
        validation = manifest.get("validation_episodes", [])
        train = manifest.get("train_episodes", [])
        if set(validation) | set(train) != all_keys or set(validation) & set(train):
            raise ValueError("split manifest does not match the available episodes")
        return (
            [tuple(map(int, key.split(":"))) for key in validation],
            [tuple(map(int, key.split(":"))) for key in train],
        )

    shuffled = list(episode_keys)
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    validation_keys, train_keys = shuffled[:validation_count], shuffled[validation_count:]
    if split_manifest_path:
        manifest_path = Path(split_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "validation_episodes": [f"{batch}:{episode}" for batch, episode in validation_keys],
            "train_episodes": [f"{batch}:{episode}" for batch, episode in train_keys],
        }
        with tempfile.NamedTemporaryFile(
            dir=manifest_path.parent, mode="w", encoding="utf-8", delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            temporary = handle.name
        os.replace(temporary, manifest_path)
    return validation_keys, train_keys


def test_dataset(data_dir: str):
    """Test dataset loading"""
    print(f"Testing dataset with data from {data_dir}")

    # Create dataset
    dataset = GameplayDataset(
        data_dir=data_dir,
        context_length=32,
        resolution=(256, 512),
        max_trajectories=100,  # Limit for testing
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print("\nSample shapes:")
    print(f"  context_frames: {sample['context_frames'].shape}")
    print(f"  context_actions: {sample['context_actions'].shape}")
    print(f"  target_frame: {sample['target_frame'].shape}")
    print(f"  target_action: {sample['target_action']}")

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        max_trajectories=100,
    )

    print(f"\nDataloader created with {len(dataloader)} batches")

    # Get a batch
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"  context_frames: {batch['context_frames'].shape}")
    print(f"  context_actions: {batch['context_actions'].shape}")
    print(f"  target_frame: {batch['target_frame'].shape}")
    print(f"  target_action: {batch['target_action'].shape}")

    print("\nDataset test complete!")


if __name__ == "__main__":
    # Test with sample data
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/recordings"

    test_dataset(data_dir)
