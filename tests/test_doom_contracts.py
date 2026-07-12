import numpy as np

from src.environment.doom_contracts import ScenarioSelector, pad_rgb_frame, paper_reward


def test_320x240_frames_are_center_padded_to_paper_shape():
    frame = np.full((240, 320, 3), 7, dtype=np.uint8)
    padded = pad_rgb_frame(frame, 320, 256)
    assert padded.shape == (256, 320, 3)
    assert np.all(padded[8:248] == 7)
    assert not padded[:8].any() and not padded[248:].any()


def test_paper_reward_uses_named_variable_deltas():
    previous = {"health": 100, "armor": 0, "ammo": 5, "killcount": 0, "hitcount": 0, "itemcount": 0, "secretcount": 0, "position_x": 0, "position_y": 0}
    current = {"health": 90, "armor": 2, "ammo": 4, "killcount": 1, "hitcount": 1, "itemcount": 1, "secretcount": 1, "position_x": 100, "position_y": 0}
    assert paper_reward(previous, current, {(0, 0)}) == 1740.0


def test_weighted_scenario_selection_is_seeded_and_validated():
    first = ScenarioSelector(["a", "b"], [1, 3], seed=4)
    second = ScenarioSelector(["a", "b"], [1, 3], seed=4)
    assert [first.next() for _ in range(8)] == [second.next() for _ in range(8)]
