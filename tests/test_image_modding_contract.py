import numpy as np
import pytest

from src.diffusion.image_modding import ImageBasedModding


def test_paste_rejects_negative_or_overflowing_masked_positions():
    modding = ImageBasedModding(object(), device="cpu")
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    thing = np.ones((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-negative"):
        modding.paste_object(base, thing, (-1, 0))
    with pytest.raises(ValueError, match="fit"):
        modding.paste_object(base, thing, (3, 3), np.ones((2, 2), dtype=bool))
