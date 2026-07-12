from pathlib import Path

import pytest

from src.config import ConfigError, load_config, validate_config


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
