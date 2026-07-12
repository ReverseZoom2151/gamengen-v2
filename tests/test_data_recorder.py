from pathlib import Path

import numpy as np

from src.diffusion.dataset import GameplayDataset, create_dataloaders
import pytest

from src.utils.data_recorder import DatasetLoader, EpisodeRecorder, load_npz_shard
from src.utils.migration import migrate_legacy_pickles


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


def test_checksum_validation_detects_tampered_shard(tmp_path: Path):
    recorder = EpisodeRecorder(str(tmp_path), save_frequency=1)
    add_episode(recorder, env_id=0, length=1)
    recorder.finalize()
    shard = tmp_path / "shard_000000.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        DatasetLoader(str(tmp_path))


def test_trusted_legacy_migration_creates_valid_npz_dataset(tmp_path: Path):
    import pickle

    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    with (source / "batch_000000.pkl").open("wb") as handle:
        pickle.dump([{"frames": [frame(1), frame(2)], "actions": [1], "rewards": [2.0]}], handle)
    assert migrate_legacy_pickles(str(source), str(target), trusted=True) == 1
    assert DatasetLoader(str(target)).load_batch(0)[0]["actions"].tolist() == [1]


def test_episode_split_manifest_is_persisted_and_disjoint(tmp_path: Path):
    recorder = EpisodeRecorder(str(tmp_path), save_frequency=3)
    for offset in range(3):
        add_episode(recorder, env_id=0, length=3, offset=offset * 10)
    recorder.finalize()
    manifest = tmp_path / "validation_split.json"
    train, validation = create_dataloaders(
        str(tmp_path), batch_size=1, context_length=2, resolution=(4, 6), num_workers=0,
        validation_fraction=1 / 3, seed=9, split_manifest_path=str(manifest),
    )
    assert manifest.is_file()
    assert set(train.dataset.indices).isdisjoint(validation.dataset.indices)
    train_again, validation_again = create_dataloaders(
        str(tmp_path), batch_size=1, context_length=2, resolution=(4, 6), num_workers=0,
        validation_fraction=0.5, seed=999, split_manifest_path=str(manifest),
    )
    assert train.dataset.indices == train_again.dataset.indices
    assert validation.dataset.indices == validation_again.dataset.indices


def test_dataset_keeps_a_bounded_decoded_source_cache(tmp_path: Path):
    recorder = EpisodeRecorder(str(tmp_path), save_frequency=1)
    add_episode(recorder, env_id=0, length=3)
    add_episode(recorder, env_id=0, length=3, offset=10)
    recorder.finalize()
    dataset = GameplayDataset(str(tmp_path), context_length=2, resolution=(4, 6), source_cache_size=1)
    first = dataset._load_source(0)
    assert first is dataset._load_source(0)
    dataset._load_source(1)
    assert list(dataset._source_cache) == [1]
