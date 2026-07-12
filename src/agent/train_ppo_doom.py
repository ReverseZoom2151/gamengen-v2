"""
Train PPO Agent on DOOM and record gameplay
Based on paper's Section 4.1 - uses PPO with Stable Baselines3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.vizdoom_env import create_vizdoom_env
from src.config import load_config, validate_config
from src.utils.data_recorder import EpisodeRecorder


class RecordingCallback(BaseCallback):
    """
    Callback to record episodes during training
    Paper: "We record the agent's training trajectories throughout
    the entire training process" (Section 3.1)
    """

    def __init__(
        self, recorder: EpisodeRecorder, record_freq: int = 1, verbose: int = 0
    ):
        super().__init__(verbose)
        self.recorder = recorder
        self.record_freq = record_freq

    def _on_step(self) -> bool:
        """Called at each step"""

        if self.n_calls % self.record_freq != 0:
            return True

        # Stable-Baselines invokes callbacks after env.step() but before it
        # updates model._last_obs, so _last_obs is observation_t and new_obs is
        # observation_t_plus_1.  Recording them together preserves the exact
        # transition contract for every vectorized environment independently.
        observations = self.model._last_obs
        for i in range(len(self.locals["infos"])):
            info = self.locals["infos"][i]
            done = bool(self.locals["dones"][i])
            truncated = bool(info.get("TimeLimit.truncated", False))
            self.recorder.add_transition(
                observation=observations[i],
                action=int(self.locals["actions"][i]),
                reward=float(self.locals["rewards"][i]),
                next_observation=self.locals["new_obs"][i],
                terminated=done and not truncated,
                truncated=truncated,
                env_id=i,
                metadata={"environment": "vizdoom"},
            )

        return True


class ProgressCallback(BaseCallback):
    """Callback for progress logging"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at each step"""

        # Check for episode end
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])

                    print(
                        f"\nEpisode {len(self.episode_rewards)} | "
                        f"Reward: {avg_reward:.2f} | "
                        f"Length: {avg_length:.1f}"
                    )

        return True


def make_env(config: dict, rank: int = 0):
    """
    Create a single ViZDoom environment

    Args:
        config: Configuration dict
        rank: Environment rank (for parallel envs)

    Returns:
        Function that creates environment
    """

    def _init():
        environment = config["environment"]
        kwargs = {
            "width": environment["resolution"]["width"],
            "height": environment["resolution"]["height"],
            "frame_skip": environment.get("action_repeat", 4),
            "use_paper_reward": config.get("use_paper_reward", config["agent"].get("reward_function") == "paper_doom"),
        }
        scenarios = environment.get("scenarios")
        if scenarios:
            from src.environment.multi_scenario import MultiScenarioViZDoomEnv

            env = MultiScenarioViZDoomEnv(scenarios=scenarios, **kwargs)
        else:
            env = create_vizdoom_env(scenario=environment.get("scenario", environment.get("config_file", "basic")), visible=False, **kwargs)

        env.reset(seed=int(config.get("seed", 0)) + rank)

        # Wrap with Monitor for episode statistics
        env = Monitor(env)

        return env

    return _init


def train_ppo_doom(config: dict):
    """Train PPO agent on DOOM and record episodes"""

    # Set random seeds
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directories
    output_dir = Path(config["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training PPO Agent on DOOM")
    print("=" * 60)
    print(f"Data directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Log directory: {log_dir}")

    # Create vectorized environments
    # Paper: "We run 8 games in parallel" (Section 4.1)
    num_envs = int(config["agent"].get("num_envs", 8))
    print(f"\nCreating {num_envs} parallel environments...")

    envs = SubprocVecEnv([make_env(config, i) for i in range(num_envs)])

    print("✓ Environments created")

    # Create data recorder
    print("\nCreating data recorder...")
    data_config = config["data_collection"]

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        compress=data_config.get("compress", True),
        save_frequency=data_config.get("save_frequency", 10),
    )

    # PPO hyperparameters from paper (Section 4.1)
    agent_config = config["agent"]

    print("\nCreating PPO agent...")
    print(f"Algorithm: {agent_config['algorithm']}")
    print(f"Total timesteps: {agent_config.get('total_timesteps', 50000000):,}")

    # Create PPO model
    # Paper Section 4.1:
    # - Simple CNN feature network (following Mnih et al. 2015b)
    # - Replay buffer size: 512
    # - Discount factor: 0.99
    # - Entropy coefficient: 0.1
    # - Batch size: 64
    # - 10 epochs per update
    # - Learning rate: 1e-4

    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        "CnnPolicy",
        envs,
        learning_rate=agent_config["learning_rate"],
        n_steps=agent_config["n_steps"],
        batch_size=agent_config["batch_size"],
        n_epochs=agent_config["n_epochs"],
        gamma=agent_config["gamma"],
        gae_lambda=agent_config.get("gae_lambda", 0.95),
        clip_range=agent_config.get("clip_epsilon", 0.2),
        ent_coef=agent_config.get("entropy_coef", 0.1),
        vf_coef=agent_config.get("value_coef", 0.5),
        max_grad_norm=agent_config.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=config.get("device", "cuda"),
    )

    print("✓ PPO agent created")
    print(f"Device: {model.device}")

    # Create callbacks
    recording_callback = RecordingCallback(recorder, record_freq=1)
    progress_callback = ProgressCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=agent_config.get("save_freq", 500)
        * num_envs,  # Adjust for vectorized envs
        save_path=str(checkpoint_dir),
        name_prefix="ppo_doom",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [recording_callback, progress_callback, checkpoint_callback]

    # Training
    total_timesteps = agent_config.get("total_timesteps", 50000000)

    print("\n" + "=" * 60)
    print(f"Starting PPO training")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {num_envs}")
    print(f"Steps per update: {agent_config['n_steps']}")
    print(f"Batch size: {agent_config['batch_size']}")
    print("=" * 60 + "\n")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_model_path = checkpoint_dir / "ppo_doom_final.zip"
    model.save(final_model_path)
    print(f"\n✓ Saved final model to {final_model_path}")

    # Finalize recording
    print("\nFinalizing recordings...")
    recorder.finalize()

    # Close environments
    envs.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total frames recorded: {recorder.total_frames}")
    print(f"Total episodes: {recorder.episode_count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on DOOM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier2_doom_lite.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total timesteps"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="DOOM scenario (basic, deadly_corridor, etc.)",
    )
    parser.add_argument(
        "--use_paper_reward",
        action="store_true",
        help="Use paper's reward function (Appendix A.5)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    config = load_config(str(config_path))

    # Apply overrides
    if args.timesteps:
        config["agent"]["total_timesteps"] = args.timesteps

    if args.scenario:
        config["environment"]["scenario"] = args.scenario

    if args.use_paper_reward:
        config["use_paper_reward"] = True

    validate_config(config)

    # Train
    train_ppo_doom(config)


if __name__ == "__main__":
    main()
