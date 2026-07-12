"""Read-only recording corpus verification command."""

import argparse
from typing import Dict

from src.utils.data_recorder import DatasetLoader


def verify_recordings(data_dir: str) -> Dict[str, int]:
    """Validate shards and return corpus counts without loading model dependencies."""
    loader = DatasetLoader(data_dir, allow_legacy_pickle=False, verify_integrity=True)
    episode_ids = set()
    total_frames = 0
    for episode in loader.iter_episodes():
        episode_id = int(episode["episode_id"])
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id in recording corpus: {episode_id}")
        episode_ids.add(episode_id)
        total_frames += len(episode["frames"])
    summary = {
        "episodes": len(episode_ids),
        "frames": total_frames,
        "shards": len(loader.shard_files),
    }
    for metadata_key, summary_key in (("total_episodes", "episodes"), ("total_frames", "frames")):
        if metadata_key in loader.metadata and int(loader.metadata[metadata_key]) != summary[summary_key]:
            raise ValueError(
                f"metadata {metadata_key} does not match reconstructed {summary_key}"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GameNGen NPZ recordings")
    parser.add_argument("data_dir")
    args = parser.parse_args()
    summary = verify_recordings(args.data_dir)
    print("Recording verification passed")
    for name, value in summary.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
