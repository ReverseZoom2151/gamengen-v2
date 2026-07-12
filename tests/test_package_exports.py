import src.diffusion as diffusion
import pytest

from src.diffusion.artifacts import model_state_from_checkpoint


def test_diffusion_package_exports_lightweight_components_without_diffusers():
    assert diffusion.create_dataloaders.__name__ == "create_dataloaders"
    assert diffusion.create_optimizer.__name__ == "create_optimizer"


def test_checkpoint_artifact_requires_every_model_component():
    with pytest.raises(ValueError, match="action_embedding"):
        model_state_from_checkpoint({"model": {"unet": {}}})
    state = {name: {} for name in ("unet", "action_embedding", "noise_aug_embedding", "action_proj")}
    assert model_state_from_checkpoint({"model": state}) is state
