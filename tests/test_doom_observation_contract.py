import numpy as np
import pytest

from src.environment.doom_contracts import ActionRepeatBias, DoomObservationHistory


def test_paper_observation_contains_resized_map_and_prior_applied_actions():
    contract = DoomObservationHistory(
        num_actions=5,
        action_history_length=3,
        screen_width=2,
        screen_height=1,
        map_width=2,
        map_height=1,
    )
    screen = np.full((2, 4, 3), 7, dtype=np.uint8)
    automap = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

    initial = contract.observe(screen, automap)
    after_first_action = contract.observe(screen, automap, action=3)
    after_second_action = contract.observe(screen, automap, action=1)

    assert initial["screen"].shape == (1, 2, 3)
    assert initial["automap"].shape == (1, 2, 3)
    assert initial["action_history"].tolist() == [0, 0, 0]
    assert after_first_action["action_history"].tolist() == [0, 0, 3]
    assert after_second_action["action_history"].tolist() == [0, 3, 1]


def test_paper_observation_rejects_action_outside_configured_space():
    contract = DoomObservationHistory(num_actions=2)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="outside"):
        contract.observe(frame, frame, action=2)


def test_action_repeat_bias_is_seeded_resets_between_episodes_and_preserves_applied_action():
    bias = ActionRepeatBias(repeat_probability=1.0, seed=9)
    assert bias.resolve(2) == (2, False)
    assert bias.resolve(4) == (2, True)
    bias.reset(seed=9)
    assert bias.resolve(4) == (4, False)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_action_repeat_bias_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="between"):
        ActionRepeatBias(probability)
