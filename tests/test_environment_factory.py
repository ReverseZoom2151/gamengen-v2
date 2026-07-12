import pytest

from src.environment.factory import interactive_environment_spec


def _config(name):
    return {
        "environment": {
            "name": name,
            "resolution": {"width": 320, "height": 256},
            "action_repeat": 4,
        },
        "agent": {"reward_function": "paper_doom"},
    }


def test_interactive_factory_spec_uses_real_chrome_runtime_and_controls():
    name, kwargs, controls = interactive_environment_spec(_config("chrome_dino"))
    assert name == "chrome_dino"
    assert kwargs["frame_skip"] == 4
    assert controls[ord(" ")] == 1


def test_interactive_factory_spec_uses_vizdoom_actions_and_raw_context_frames():
    name, kwargs, controls = interactive_environment_spec(_config("vizdoom"))
    assert name == "vizdoom"
    assert kwargs["include_automap"] is False
    assert kwargs["use_paper_reward"] is True
    assert controls[ord(" ")] == 7
    assert controls[81] == 5


def test_interactive_factory_rejects_unsupported_environment():
    with pytest.raises(ValueError, match="does not support"):
        interactive_environment_spec(_config("mock"))
