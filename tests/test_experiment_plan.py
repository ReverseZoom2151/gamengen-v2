from src.utils.experiment_plan import context_length_ablation, data_policy_ablation, noise_ablation, save_experiment_plan


def _config():
    return {"experiment_name": "base", "data_dir": "data/a", "diffusion": {"context_length": 64, "noise_augmentation": {"enabled": True}}}


def test_ablation_plans_are_isolated_and_named():
    config = _config()
    contexts = context_length_ablation(config, [1, 64])
    assert [item["experiment_name"] for item in contexts] == ["base-context-1", "base-context-64"]
    assert config["diffusion"]["context_length"] == 64
    assert [item["diffusion"]["noise_augmentation"]["enabled"] for item in noise_ablation(config)] == [False, True]
    assert [item["data_dir"] for item in data_policy_ablation(config, {"random": "data/r", "agent": "data/a"})] == ["data/a", "data/r"]


def test_experiment_plan_is_hash_bound_when_saved(tmp_path):
    path = save_experiment_plan(tmp_path / "plan.json", context_length_ablation(_config(), [1]))
    assert "config_sha256" in path.read_text()
