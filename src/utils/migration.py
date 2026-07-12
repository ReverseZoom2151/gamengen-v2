"""Explicit migration of trusted legacy pickle recordings to NPZ shards."""

import argparse
import pickle
from pathlib import Path

import numpy as np

from src.utils.data_recorder import EpisodeRecorder


def migrate_legacy_pickles(source_dir: str, output_dir: str, *, trusted: bool) -> int:
    """Migrate legacy batches after an explicit acknowledgement of pickle risk."""
    if not trusted:
        raise ValueError("legacy pickle migration requires trusted=True")
    sources = sorted(Path(source_dir).glob("batch_*.pkl"))
    if not sources:
        raise ValueError(f"no legacy batch_*.pkl files found in {source_dir}")
    recorder = EpisodeRecorder(output_dir)
    migrated = 0
    for source in sources:
        with source.open("rb") as handle:
            episodes = pickle.load(handle)
        for episode in episodes:
            frames = np.asarray(episode["frames"])
            actions = np.asarray(episode["actions"])
            rewards = np.asarray(episode.get("rewards", np.zeros(len(actions))))
            if len(frames) == len(actions):
                frames = np.concatenate([frames, frames[-1:]], axis=0)
            if len(frames) != len(actions) + 1 or len(rewards) != len(actions):
                raise ValueError(f"invalid legacy episode in {source}")
            for index, action in enumerate(actions):
                recorder.add_transition(
                    frames[index], int(action), float(rewards[index]), frames[index + 1],
                    terminated=index == len(actions) - 1, env_id=0,
                    metadata={"migrated_from": source.name},
                )
            migrated += 1
    recorder.finalize()
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate trusted legacy GameNGen pickle recordings")
    parser.add_argument("source_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--trusted-legacy-input", action="store_true")
    args = parser.parse_args()
    print(f"Migrated {migrate_legacy_pickles(args.source_dir, args.output_dir, trusted=args.trusted_legacy_input)} episodes")


if __name__ == "__main__":
    main()
