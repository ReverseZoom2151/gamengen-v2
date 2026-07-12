import importlib


def test_public_package_exposes_version_and_legacy_subpackages():
    package = importlib.import_module("gamengen")
    config = importlib.import_module("gamengen.config")
    assert package.__version__ == "0.2.0"
    assert config.ConfigError.__name__ == "ConfigError"
