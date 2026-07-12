"""
Comprehensive test script for GameNGen diffusion components
Tests model, dataset, and training/inference functionality
"""

import pytest

pytestmark = pytest.mark.integration

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_action_embedding():
    """Test action embedding layer"""
    print("\n" + "=" * 60)
    print("Testing Action Embedding")
    print("=" * 60)

    from src.diffusion.model import ActionEmbedding

    num_actions = 3
    embedding_dim = 128
    batch_size = 4
    seq_len = 32

    model = ActionEmbedding(num_actions, embedding_dim)
    actions = torch.randint(0, num_actions, (batch_size, seq_len))

    output = model(actions)

    print(f"  Input shape: {actions.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, embedding_dim)
    print("  ✓ Action embedding test passed!")


def test_noise_aug_embedding():
    """Test noise augmentation embedding"""
    print("\n" + "=" * 60)
    print("Testing Noise Augmentation Embedding")
    print("=" * 60)

    from src.diffusion.model import NoiseAugmentationEmbedding

    num_buckets = 10
    embedding_dim = 128
    batch_size = 4

    model = NoiseAugmentationEmbedding(num_buckets, embedding_dim)
    noise_levels = torch.randint(0, num_buckets, (batch_size,))

    output = model(noise_levels)

    print(f"  Input shape: {noise_levels.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, embedding_dim)
    print("  ✓ Noise augmentation embedding test passed!")


def test_model_creation():
    """Test model can be created"""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)

    from src.diffusion.model import ActionConditionedDiffusionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # This will download SD v1.4 if not cached
    print("  Loading Stable Diffusion v1.4...")
    print("  (This may take a few minutes on first run)")

    model = ActionConditionedDiffusionModel(
        num_actions=3, context_length=32, device=device, dtype=torch.float32
    )

    print(f"  ✓ Model created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )


def test_model_forward():
    """Test model forward pass"""
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass")
    print("=" * 60)

    from src.diffusion.model import ActionConditionedDiffusionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActionConditionedDiffusionModel(
        num_actions=3, context_length=32, device=device, dtype=torch.float32
    )

    batch_size = 2
    target_frame = torch.randint(
        0, 255, (batch_size, 3, 256, 512), dtype=torch.float32
    ).to(device)
    context_frames = torch.randint(
        0, 255, (batch_size, 32, 3, 256, 512), dtype=torch.float32
    ).to(device)
    actions = torch.randint(0, 3, (batch_size, 32)).to(device)

    print(f"  Target frame shape: {target_frame.shape}")
    print(f"  Context frames shape: {context_frames.shape}")
    print(f"  Actions shape: {actions.shape}")

    print("  Running forward pass...")
    model.train()
    loss = model(target_frame, context_frames, actions)

    print(f"  Loss: {loss.item():.4f}")

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print("  ✓ Forward pass test passed!")


def test_model_generation():
    """Test model generation"""
    print("\n" + "=" * 60)
    print("Testing Model Generation")
    print("=" * 60)

    from src.diffusion.model import ActionConditionedDiffusionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActionConditionedDiffusionModel(
        num_actions=3, context_length=32, device=device, dtype=torch.float32
    )

    batch_size = 1
    context_frames = torch.randint(
        0, 255, (batch_size, 32, 3, 256, 512), dtype=torch.float32
    ).to(device)
    actions = torch.randint(0, 3, (batch_size, 32)).to(device)

    print(f"  Context frames shape: {context_frames.shape}")
    print(f"  Actions shape: {actions.shape}")

    print("  Running generation (4 DDIM steps)...")
    import time

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            context_frames, actions, num_inference_steps=4, guidance_scale=1.5
        )

    elapsed = time.time() - start_time

    print(f"  Generated shape: {generated.shape}")
    print(f"  Time taken: {elapsed:.3f} seconds ({1/elapsed:.1f} FPS)")

    assert generated.shape == (batch_size, 3, 256, 512)

    print("  ✓ Generation test passed!")


def test_dataset_with_dummy_data():
    """Test dataset with dummy data"""
    print("\n" + "=" * 60)
    print("Testing Dataset (with dummy data)")
    print("=" * 60)

    import shutil
    # Create dummy data
    import tempfile

    from src.diffusion.dataset import GameplayDataset
    from src.utils.data_recorder import EpisodeRecorder

    temp_dir = tempfile.mkdtemp()
    print(f"  Creating dummy data in {temp_dir}")

    try:
        # Record some dummy episodes
        recorder = EpisodeRecorder(temp_dir, save_frequency=2)

        for ep in range(5):
            for step in range(50):
                frame = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
                action = np.random.randint(0, 3)
                reward = np.random.rand()
                done = step == 49

                recorder.add_step(frame, action, reward, done)

        recorder.finalize()

        print(f"  Created {recorder.episode_count} dummy episodes")

        # Create dataset
        dataset = GameplayDataset(
            data_dir=temp_dir,
            context_length=32,
            resolution=(256, 512),
        )

        print(f"  Dataset size: {len(dataset)}")

        # Get sample
        sample = dataset[0]

        print(f"  Sample shapes:")
        print(f"    context_frames: {sample['context_frames'].shape}")
        print(f"    context_actions: {sample['context_actions'].shape}")
        print(f"    target_frame: {sample['target_frame'].shape}")
        print(f"    target_action: {sample['target_action']}")

        assert sample["context_frames"].shape == (32, 3, 256, 512)
        assert sample["context_actions"].shape == (32,)
        assert sample["target_frame"].shape == (3, 256, 512)

        print("  ✓ Dataset test passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_save_and_load():
    """Test model save and load"""
    print("\n" + "=" * 60)
    print("Testing Model Save/Load")
    print("=" * 60)

    import shutil
    import tempfile

    from src.diffusion.model import ActionConditionedDiffusionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = ActionConditionedDiffusionModel(
        num_actions=3, context_length=32, device=device, dtype=torch.float32
    )

    # Save
    temp_dir = tempfile.mkdtemp()
    print(f"  Saving to {temp_dir}")

    try:
        model.save_pretrained(temp_dir)

        # Load
        model2 = ActionConditionedDiffusionModel(
            num_actions=3, context_length=32, device=device, dtype=torch.float32
        )

        print(f"  Loading from {temp_dir}")
        model2.load_pretrained(temp_dir)

        print("  ✓ Save/load test passed!")

    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" " * 20 + "GameNGen Diffusion Tests")
    print("=" * 80)

    tests = [
        ("Action Embedding", test_action_embedding),
        ("Noise Aug Embedding", test_noise_aug_embedding),
        ("Model Creation", test_model_creation),
        ("Model Forward", test_model_forward),
        ("Model Generation", test_model_generation),
        ("Dataset (dummy data)", test_dataset_with_dummy_data),
        ("Save/Load", test_save_and_load),
    ]

    results = {}

    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            print(f"\n  ✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            results[name] = "FAIL"

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)

    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")

    all_passed = all(r == "PASS" for r in results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYour GameNGen implementation is ready!")
        print("\nNext steps:")
        print("1. Train RL agent: python src/agent/train_dqn.py")
        print("2. Train diffusion model: python src/diffusion/train.py")
        print("3. Run inference: python src/diffusion/inference.py --checkpoint <path>")
    else:
        print("✗ Some tests failed. Please fix errors above.")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
