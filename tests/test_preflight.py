from src.preflight import preflight, required_modules


def config(name="chrome_dino", use_mock=False):
    return {"environment": {"name": name, "use_mock": use_mock}}


def test_preflight_selects_runtime_specific_dependencies():
    assert "selenium" in required_modules(config(), "collect")
    assert "selenium" not in required_modules(config(use_mock=True), "collect")
    assert {"vizdoom", "stable_baselines3"} <= required_modules(config("vizdoom"), "collect")
    assert {"diffusers", "transformers"} <= required_modules(config(), "train")


def test_preflight_reports_missing_modules(monkeypatch):
    monkeypatch.setattr("src.preflight.importlib.util.find_spec", lambda name: None)
    assert "torch" in preflight(config(use_mock=True), "collect")
