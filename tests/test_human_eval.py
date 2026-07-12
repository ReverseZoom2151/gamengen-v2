import json

import pytest

from src.utils.human_eval import HumanEvaluationFramework


def _framework(tmp_path):
    framework = HumanEvaluationFramework(str(tmp_path), clip_lengths=[1.6])
    framework.create_evaluation_clips(
        ["source-real.mp4"], ["source-fake.mp4"], num_clips_per_length=2, seed=4
    )
    return framework


def test_public_protocol_is_blinded_and_web_ui_never_embeds_answer_key(tmp_path):
    framework = _framework(tmp_path)
    protocol, answer_key = framework.save_evaluation_protocol()
    text = protocol.read_text()
    assert "real_is_on_left" not in text
    assert "real_video_path" not in text
    assert "fake_video_path" not in text
    assert "source-real.mp4" not in text
    assert "source-fake.mp4" not in text
    assert answer_key.is_file()
    html = framework.build_web_ui().read_text()
    assert "answer_key" not in html
    assert "Export responses" in html


def test_materialized_media_has_neutral_names(tmp_path):
    real = tmp_path / "definitely-real.mp4"
    fake = tmp_path / "definitely-fake.mp4"
    real.write_bytes(b"real")
    fake.write_bytes(b"fake")
    framework = HumanEvaluationFramework(str(tmp_path / "study"), clip_lengths=[1.6])
    framework.create_evaluation_clips([str(real)], [str(fake)], num_clips_per_length=1, seed=1)
    media = framework.materialize_blinded_media()
    assert sorted(path.name for path in media.iterdir()) == ["clip_0000_left.mp4", "clip_0000_right.mp4"]


def test_private_scoring_marks_responses_only_after_collection(tmp_path):
    framework = _framework(tmp_path)
    _, answer_key = framework.save_evaluation_protocol()
    first_is_left = json.loads(answer_key.read_text())["0"]
    responses = {
        "responses": [
            {
                "clip_id": 0,
                "user_choice": "left" if first_is_left else "right",
                "confidence": 4,
                "time_taken_seconds": 2.5,
            }
        ]
    }
    result = framework.score_blinded_responses(responses)
    assert result[0].correct is True
    with pytest.raises(ValueError, match="duplicate"):
        framework.score_blinded_responses({"responses": responses["responses"] * 2})
