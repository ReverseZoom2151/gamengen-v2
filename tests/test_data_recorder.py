from pathlib import Path

import numpy as np

from src.diffusion.dataset import GameplayDataset
from src.utils.data_recorder import DatasetLoader, EpisodeRecorder, load_npz_shard


def frame(value: int) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def add_episode(recorder: EpisodeRecorder, env_id: int, length: int, offset: int = 0) -> None:
    observation = frame(offset)
    for step in range(length):
        next_observation = frame(offset + step + 1)
        recorder.add_transition(
            observation,
            action=step % 3,
            reward=float(step),
            next_observation=next_observation,
            terminated=step == length - 1,
            env_id=env_id,
            metadata={"scenario": "test"},
        )
        observation = next_observation


def test_partial_final_shard_does_not_overwrite_full_shard(tmp_path: Path):
    recorder = EpisodeRecorder(str(tmp_path), save_frequency=2)
    add_episode(recorder, env_id=0, length=2, offset=0)
    add_episode(recorder, env_id=0, length=2, offset=10)
    add_episode(recorder, env_id=0, length=2, offset=20)
    recorder.finalize()

    shards = sorted(tmp_path.glob("shard_*.npz"))
    assert [path.name for path in shards] == ["shard_000000.npz", "shard_000001.npz"]
    assert len(load_npz_shard(shards[0])) == 2
    assert len(load_npz_shard(shards[1])) == 1


def test_parallel_environment_buffers_are_isolated(tmp_path: Path):
    recorder = EpisodeRecorder(str(tmp_path), save_frequency=2)
    recorder.add_transition(frame(1), 1, 1.0, frame(2), env_id=0)
    recorder.add_transition(frame(101), 2, 2.0, frame(102), env_id=1)
    recorder.add_transition(frame(2), 0, 3.0, frame(3), terminated=True, env_id=0)
    recorder.add_transition(frame(102), 1, 4.0, frame(103), terminated=True, env_id=1)
    recorder.finalize()

    episodes = load_npz_shard(tmp_path / "shard_000000.npz")
    assert {episode["env_id"] for episode in episodes} == {0, 1}
    assert np.all(episodes[0]["frames"] < 100)
    assert np.all(episodes[1]["frames"] >= 100)


def test_recorder_resume_allocates_new_shard_ids(tmp_path: Path):
    first = EpisodeRecorder(str(tmp_path), save_frequency=1)
    add_episode(first, env_id=0, length=1)
    first.finalize()

    second = EpisodeRecorder(str(tmp_path), save_frequency=1)
    add_episode(second, env_id=0, length=1, offset=10)
    second.finalize()

    assert sorted(path.name for path in tmp_path.glob("shard_*.npz")) == [
        "shard_000000.npz",
        "shard_000001.npz",
    ]
    loader = DatasetLoader(str(tmp_path))
    assert len(list(loader.iter_episodes())) == 2


def test_npz_dataset_uses_canonical_target_action(tmp_path: Path):
    recorder = EpisodeRecorder(str(tmp_path), save_frequency=1)
    add_episode(recorder, env_id=0, length=3)
    recorder.finalize()

    dataset = GameplayDataset(str(tmp_path), context_length=2, resolution=(4, 6))
    sample = dataset[0]
    assert sample["context_actions"].tolist() == [0, 1]
    assert sample["target_action"].item() == 1
    assert sample["target_frame"].shape == (3, 4, 6)
