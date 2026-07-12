"""
Simple test script for GameNGen diffusion components (Windows compatible)
"""

import pytest

pytestmark = pytest.mark.integration

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test all imports work"""
    print("\nTesting imports...")

    try:
        from src.diffusion.dataset import GameplayDataset, create_dataloader
        from src.diffusion.model import (ActionConditionedDiffusionModel,
                                         ActionEmbedding,
                                         NoiseAugmentationEmbedding)

        print("  [OK] All diffusion imports successful!")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    print("  (This will download Stable Diffusion v1.4 on first run, ~4GB)")

    try:
        from src.diffusion.model import ActionConditionedDiffusionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        model = ActionConditionedDiffusionModel(
            num_actions=3, context_length=32, device=device, dtype=torch.float32
        )

        print(f"  [OK] Model created!")
        print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"  [FAIL] Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_and_generation():
    """Test forward pass and generation"""
    print("\nTesting forward pass and generation...")

    try:
        from src.diffusion.model import ActionConditionedDiffusionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = ActionConditionedDiffusionModel(
            num_actions=3, context_length=32, device=device, dtype=torch.float32
        )

        # Test forward
        batch_size = 2
        target_frame = torch.randint(
            0, 255, (batch_size, 3, 256, 512), dtype=torch.float32
        ).to(device)
        context_frames = torch.randint(
            0, 255, (batch_size, 32, 3, 256, 512), dtype=torch.float32
        ).to(device)
        actions = torch.randint(0, 3, (batch_size, 32)).to(device)

        model.train()
        loss = model(target_frame, context_frames, actions)

        print(f"  Forward pass - Loss: {loss.item():.4f}")

        # Test generation
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                context_frames[:1], actions[:1], num_inference_steps=4
            )

        print(f"  Generation - Shape: {generated.shape}")
        print("  [OK] Forward and generation tests passed!")
        return True

    except Exception as e:
        print(f"  [FAIL] Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print(" " * 15 + "GameNGen Diffusion Tests")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Forward & Generation", test_forward_and_generation),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] {name} test failed: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] All core tests passed!")
        print("\nYour GameNGen implementation is ready!")
        print("\nNext steps:")
        print("  1. Train RL agent: python src/agent/train_dqn.py")
        print("  2. Train diffusion: python src/diffusion/train.py")
        print(
            "  3. Run inference: python src/diffusion/inference.py --checkpoint <path>"
        )
    else:
        print("[ERROR] Some tests failed. Check errors above.")

    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
