"""
Quick test script to verify installation and setup
Run this after installing dependencies
"""

__test__ = False

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")

    failed = []

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable Baselines3"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]

    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All imports successful!")
        return True


def test_cuda():
    """Test if CUDA is available"""
    print("\nTesting CUDA...")

    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        return True
    else:
        print(f"  ⚠ CUDA not available; CPU-only diagnostics remain supported")
        return True


def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")

    required_paths = [
        "src/agent",
        "src/diffusion",
        "src/environment",
        "src/utils",
        "configs/tier1_chrome_dino.yaml",
        "configs/tier2_doom_lite.yaml",
        "configs/tier3_full_doom.yaml",
        "requirements.txt",
    ]

    all_exist = True
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"  ✓ {path_str}")
        else:
            print(f"  ✗ {path_str} - missing")
            all_exist = False

    if all_exist:
        print("\n✓ Project structure correct!")
    else:
        print("\n❌ Some files/directories missing")

    return all_exist


def test_environment():
    """Test if environment can be created"""
    print("\nTesting environment...")

    try:
        from src.environment.chrome_dino_env import SimpleDinoEnv

        env = SimpleDinoEnv(width=512, height=256)
        obs, info = env.reset()

        print(f"  ✓ Environment created")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")

        # Test one step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        print(f"  ✓ Step executed successfully")

        env.close()
        return True

    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def test_agent():
    """Test if agent can be created"""
    print("\nTesting DQN agent...")

    try:
        from src.agent.dqn_agent import DQNAgent
        from src.environment.chrome_dino_env import SimpleDinoEnv

        env = SimpleDinoEnv(width=512, height=256)
        agent = DQNAgent(env, device="cpu")  # Use CPU for testing

        print(f"  ✓ Agent created")
        print(f"  Device: {agent.device}")
        print(
            f"  Parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}"
        )

        # Test action selection
        obs, _ = env.reset()
        action = agent.select_action(obs, training=False)

        print(f"  ✓ Action selection works")

        env.close()
        return True

    except Exception as e:
        print(f"  ✗ Agent test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("GameNGen - Quick Test Script")
    print("=" * 60)

    results = {}

    results["imports"] = test_imports()
    results["cuda"] = test_cuda()
    results["structure"] = test_project_structure()
    results["environment"] = test_environment()
    results["agent"] = test_agent()

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test.capitalize()}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("1. Review config.yaml")
        print("2. Run: python src/agent/train_dqn.py")
        print("=" * 60)
    else:
        print("❌ Some tests failed. Please fix errors above.")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
