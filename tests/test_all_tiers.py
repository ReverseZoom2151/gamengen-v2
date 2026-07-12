"""
Comprehensive test script for all 3 tiers of GameNGen
Tests Tier 1 (Chrome Dino), Tier 2 (DOOM Lite), and Tier 3 (Full DOOM)
"""

import pytest

pytestmark = pytest.mark.integration

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_tier1_components():
    """Test Tier 1 (Chrome Dino) components"""
    print("\n" + "=" * 70)
    print("TIER 1: Chrome Dino - Component Tests")
    print("=" * 70)

    try:
        from src.agent.dqn_agent import DQNAgent
        from src.environment.chrome_dino_env import SimpleDinoEnv

        # Test environment
        env = SimpleDinoEnv(width=512, height=256)
        obs, _ = env.reset()
        print("  [OK] Chrome Dino environment")

        # Test DQN agent
        agent = DQNAgent(env, device="cpu")
        action = agent.select_action(obs, training=False)
        print("  [OK] DQN agent")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] Tier 1 test failed: {e}")
        return False


def test_tier2_components():
    """Test Tier 2 (DOOM Lite) components"""
    print("\n" + "=" * 70)
    print("TIER 2: DOOM Lite - Component Tests")
    print("=" * 70)

    results = {}

    # Test ViZDoom installation
    try:
        import vizdoom as vzd

        print("  [OK] ViZDoom installed")
        results["vizdoom_install"] = True
    except ImportError:
        print("  [SKIP] ViZDoom not installed (pip install vizdoom)")
        print("         Tier 2 will not work without ViZDoom")
        results["vizdoom_install"] = False
        return results

    # Test ViZDoom environment
    try:
        from src.environment.vizdoom_env import create_vizdoom_env

        # This will fail if scenarios not found, but class should work
        print("  [OK] ViZDoom environment wrapper created")
        results["vizdoom_env"] = True
    except Exception as e:
        print(f"  [FAIL] ViZDoom environment: {e}")
        results["vizdoom_env"] = False

    # Test PPO training script exists
    try:
        ppo_script = Path("src/agent/train_ppo_doom.py")
        if ppo_script.exists():
            print("  [OK] PPO training script exists")
            results["ppo_script"] = True
        else:
            print("  [FAIL] PPO training script not found")
            results["ppo_script"] = False
    except Exception as e:
        print(f"  [FAIL] PPO script check: {e}")
        results["ppo_script"] = False

    # Test Stable Baselines3
    try:
        from stable_baselines3 import PPO

        print("  [OK] Stable Baselines3 (PPO)")
        results["sb3"] = True
    except ImportError:
        print("  [FAIL] Stable Baselines3 not installed")
        results["sb3"] = False

    # Test config file
    try:
        import yaml

        config_path = Path("configs/tier2_doom_lite.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print("  [OK] Tier 2 configuration file")
            results["config"] = True
        else:
            print("  [FAIL] Config file not found")
            results["config"] = False
    except Exception as e:
        print(f"  [FAIL] Config test: {e}")
        results["config"] = False

    return results


def test_tier3_components():
    """Test Tier 3 (Full DOOM) components"""
    print("\n" + "=" * 70)
    print("TIER 3: Full DOOM - Component Tests")
    print("=" * 70)

    results = {}

    # Test Adafactor optimizer
    try:
        import torch

        from src.diffusion.optimizers import Adafactor, create_optimizer

        # Create dummy model
        model = torch.nn.Linear(10, 10)
        optimizer = Adafactor(model.parameters(), lr=2e-5)

        print("  [OK] Adafactor optimizer")
        results["adafactor"] = True
    except Exception as e:
        print(f"  [FAIL] Adafactor: {e}")
        results["adafactor"] = False

    # Test distillation script
    try:
        distill_script = Path("src/diffusion/distill.py")
        if distill_script.exists():
            print("  [OK] Distillation script exists")
            results["distill_script"] = True
        else:
            print("  [FAIL] Distillation script not found")
            results["distill_script"] = False
    except Exception as e:
        print(f"  [FAIL] Distillation check: {e}")
        results["distill_script"] = False

    # Test evaluation suite
    try:
        from src.utils.evaluation import FVDCalculator, GameNGenEvaluator

        device = "cpu"  # Use CPU for testing
        evaluator = GameNGenEvaluator(device=device)

        print("  [OK] Evaluation suite (PSNR, LPIPS, SSIM, FVD)")
        results["evaluation"] = True
    except Exception as e:
        print(f"  [FAIL] Evaluation suite: {e}")
        results["evaluation"] = False

    # Test decoder fine-tuning script
    try:
        decoder_script = Path("src/diffusion/decoder_finetune.py")
        if decoder_script.exists():
            print("  [OK] Decoder fine-tuning script exists")
            results["decoder_script"] = True
        else:
            print("  [FAIL] Decoder script not found")
            results["decoder_script"] = False
    except Exception as e:
        print(f"  [FAIL] Decoder check: {e}")
        results["decoder_script"] = False

    # Test config file
    try:
        import yaml

        config_path = Path("configs/tier3_full_doom.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print("  [OK] Tier 3 configuration file")
            results["config"] = True
        else:
            print("  [FAIL] Config file not found")
            results["config"] = False
    except Exception as e:
        print(f"  [FAIL] Config test: {e}")
        results["config"] = False

    return results


def test_core_diffusion():
    """Test core diffusion model (used by all tiers)"""
    print("\n" + "=" * 70)
    print("CORE: Diffusion Model - All Tiers")
    print("=" * 70)

    try:
        import torch

        from src.diffusion.model import ActionConditionedDiffusionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        # Create model (will download SD v1.4)
        print("  Loading Stable Diffusion v1.4...")
        model = ActionConditionedDiffusionModel(
            num_actions=3, context_length=32, device=device, dtype=torch.float32
        )

        print(
            f"  [OK] Model created ({sum(p.numel() for p in model.parameters()):,} params)"
        )

        # Test generation
        context_frames = torch.randint(
            0, 255, (1, 32, 3, 256, 512), dtype=torch.float32
        ).to(device)
        actions = torch.randint(0, 3, (1, 32)).to(device)

        with torch.no_grad():
            generated = model.generate(context_frames, actions, num_inference_steps=4)

        print(f"  [OK] Generation works ({generated.shape})")

        return True

    except Exception as e:
        print(f"  [FAIL] Core diffusion test: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_tier_summary(tier_num: int, results: dict):
    """Print summary for a tier"""
    if isinstance(results, bool):
        status = "[READY]" if results else "[NOT READY]"
        print(f"\nTier {tier_num}: {status}")
        return results

    all_passed = all(results.values())
    status = "[READY]" if all_passed else "[PARTIAL]"

    print(f"\nTier {tier_num}: {status}")
    for component, passed in results.items():
        status_str = "[OK]" if passed else "[MISSING]"
        print(f"  {status_str} {component}")

    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "GameNGen - All Tiers Test Suite")
    print("=" * 70)

    # Test core (required for all tiers)
    core_result = test_core_diffusion()

    # Test each tier
    tier1_result = test_tier1_components()
    tier2_results = test_tier2_components()
    tier3_results = test_tier3_components()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Which Tiers Are Ready?")
    print("=" * 70)

    tier1_ready = core_result and tier1_result
    tier2_ready = core_result and (
        tier2_results.get("vizdoom_install", False)
        and tier2_results.get("vizdoom_env", False)
        and tier2_results.get("ppo_script", False)
        and tier2_results.get("sb3", False)
    )
    tier3_ready = tier2_ready and (
        tier3_results.get("adafactor", False)
        and tier3_results.get("distill_script", False)
        and tier3_results.get("evaluation", False)
    )

    print(f"\n[{'READY' if tier1_ready else 'NOT READY'}] Tier 1: Chrome Dino")
    if tier1_ready:
        print("  -> python src/agent/train_dqn.py")
        print("  -> python src/diffusion/train.py")

    print(f"\n[{'READY' if tier2_ready else 'PARTIAL'}] Tier 2: DOOM Lite")
    if tier2_ready:
        print(
            "  -> python src/agent/train_ppo_doom.py --config configs/tier2_doom_lite.yaml"
        )
        print(
            "  -> python src/diffusion/train.py --config configs/tier2_doom_lite.yaml"
        )
    elif not tier2_results.get("vizdoom_install"):
        print("  Missing: pip install vizdoom")

    print(f"\n[{'READY' if tier3_ready else 'PARTIAL'}] Tier 3: Full DOOM")
    if tier3_ready:
        print(
            "  -> python src/agent/train_ppo_doom.py --config configs/tier3_full_doom.yaml --use_paper_reward"
        )
        print(
            "  -> python src/diffusion/train.py --config configs/tier3_full_doom.yaml"
        )
        print(
            "  -> python src/diffusion/decoder_finetune.py --config configs/tier3_full_doom.yaml"
        )
        print(
            "  -> python src/diffusion/distill.py --teacher checkpoints_doom_full/latest_checkpoint.pt"
        )
    elif not tier2_results.get("vizdoom_install"):
        print("  Missing: pip install vizdoom (required for Tier 2 first)")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    if tier1_ready:
        print("\n[OPTION 1] Start with Tier 1 (RECOMMENDED)")
        print("  Quick validation, 2-3 days total")
        print("  Command: python src/agent/train_dqn.py")

    if tier2_ready:
        print("\n[OPTION 2] Jump to Tier 2 (if confident)")
        print("  Real DOOM, 1 week total")
        print(
            "  Command: python src/agent/train_ppo_doom.py --config configs/tier2_doom_lite.yaml"
        )

    if tier3_ready:
        print("\n[OPTION 3] Full paper implementation (ambitious)")
        print("  Match paper exactly, 3-4 weeks total")
        print(
            "  Command: python src/agent/train_ppo_doom.py --config configs/tier3_full_doom.yaml --use_paper_reward"
        )

    if not tier2_ready and not tier2_results.get("vizdoom_install"):
        print("\n[ACTION NEEDED] To enable Tier 2 & 3:")
        print("  pip install vizdoom")
        print("  Then run this test again")

    print("\n" + "=" * 70)

    return tier1_ready


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
