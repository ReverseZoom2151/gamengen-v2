import numpy as np
import pytest

from src.utils.data_recorder import EpisodeRecorder
from src.utils.evaluation_manifest import (
    create_evaluation_manifest,
    create_human_benchmark_manifest,
    validate_evaluation_manifest,
)


def test_evaluation_manifest_is_episode_based_and_checksum_bound(tmp_path):
    data = tmp_path / "recordings"
    recorder = EpisodeRecorder(str(data), save_frequency=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder.add_transition(frame, 0, 0.0, frame, terminated=True, metadata={"scenario": "level-a"})
    recorder.finalize()
    manifest = create_evaluation_manifest(data, tmp_path / "evaluation.json")
    payload = validate_evaluation_manifest(manifest, data)
    assert payload["episodes"][0]["scenario"] == "level-a"
    shard = next(data.glob("shard_*.npz"))
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_evaluation_manifest(manifest, data)


def test_human_benchmark_manifest_requires_explicit_source_tag(tmp_path):
    data = tmp_path / "recordings"
    recorder = EpisodeRecorder(str(data), save_frequency=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder.add_transition(frame, 0, 0.0, frame, terminated=True, metadata={"demonstration_source": "human", "participant_id": "p-1"})
    recorder.finalize()
    payload = __import__("json").loads(create_human_benchmark_manifest(data, tmp_path / "human.json").read_text())
    assert payload["kind"] == "human_demonstration_benchmark"
    assert payload["episodes"][0]["participant_id"] == "p-1"
