import pytest

from src.diffusion.distill import distill_model


def test_unvalidated_distillation_is_explicitly_quarantined():
    with pytest.raises(NotImplementedError, match="not validated"):
        distill_model({})
