import src.diffusion as diffusion


def test_diffusion_package_exports_lightweight_components_without_diffusers():
    assert diffusion.create_dataloaders.__name__ == "create_dataloaders"
    assert diffusion.create_optimizer.__name__ == "create_optimizer"
