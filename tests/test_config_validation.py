from pathlib import Path

import pytest

from src.config import ConfigError, load_config, validate_config, validate_diffusion_training_config


def valid_config():
    return {
        "project_name": "test",
        "experiment_name": "unit",
        "data_dir": "data",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "device": "cpu",
        "mixed_precision": False,
        "environment": {
            "name": "mock",
            "num_actions": 3,
            "resolution": {"width": 8, "height": 8},
        },
        "agent": {},
        "data_collection": {},
        "diffusion": {"context_length": 2, "pretrained_model": "test/model"},
    }


def test_validate_config_accepts_complete_minimum():
    assert validate_config(valid_config())["environment"]["num_actions"] == 3


def test_validate_config_rejects_unknown_top_level_key():
    config = valid_config()
    config["typo"] = True
    with pytest.raises(ConfigError, match="unknown top-level"):
        validate_config(config)


def test_load_config_reads_repository_tier_config():
    config_path = Path(__file__).parents[1] / "configs" / "tier1_chrome_dino.yaml"
    config = load_config(str(config_path))
    assert config["diffusion"]["context_length"] == 32


@pytest.mark.parametrize("config_name", ["tier1_chrome_dino.yaml", "tier2_doom_lite.yaml", "tier3_full_doom.yaml"])
def test_all_repository_tiers_are_semantically_valid(config_name):
    config_path = Path(__file__).parents[1] / "configs" / config_name
    assert load_config(str(config_path))["project_name"].startswith("gamengen")


def test_validate_config_rejects_incompatible_resolution():
    config = valid_config()
    config["diffusion"]["resolution"] = {"width": 4, "height": 8}
    with pytest.raises(ConfigError, match="must match"):
        validate_config(config)


def test_validate_config_rejects_unknown_scheduler():
    config = valid_config()
    config["diffusion"]["lr_scheduler"] = "magic"
    with pytest.raises(ConfigError, match="lr_scheduler"):
        validate_config(config)


def test_training_validation_requires_actual_training_fields():
    with pytest.raises(ConfigError, match="diffusion training is missing"):
        validate_diffusion_training_config(valid_config())


def test_validate_config_rejects_unknown_environment_or_incompatible_agent():
    invalid_environment = valid_config()
    invalid_environment["environment"]["name"] = "typo"
    with pytest.raises(ConfigError, match="environment.name"):
        validate_config(invalid_environment)
    invalid_agent = valid_config()
    invalid_agent["environment"]["name"] = "vizdoom"
    invalid_agent["agent"]["algorithm"] = "DQN"
    with pytest.raises(ConfigError, match="vizdoom"):
        validate_config(invalid_agent)


def test_validate_config_rejects_enabled_zero_noise_augmentation():
    config = valid_config()
    config["diffusion"]["noise_augmentation"] = {"enabled": True, "num_buckets": 2, "max_noise_level": 0}
    with pytest.raises(ConfigError, match="must be positive"):
        validate_config(config)
