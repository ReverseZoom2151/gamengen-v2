"""
Train DQN Agent on Chrome Dino and record gameplay
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.dqn_agent import DQNAgent
from src.config import load_config, validate_config
from src.environment.chrome_dino_env import ChromeDinoEnv, SimpleDinoEnv
from src.utils.data_recorder import EpisodeRecorder


def train_dqn_agent(config: dict):
    """Train DQN agent and record episodes"""

    # Set random seeds
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directories
    output_dir = Path(config["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    print("Creating environment...")
    env_config = config["environment"]

    env_kwargs = {
        "width": env_config["resolution"]["width"],
        "height": env_config["resolution"]["height"],
    }
    if env_config.get("use_mock", False):
        env = SimpleDinoEnv(**env_kwargs)
    else:
        env = ChromeDinoEnv(
            **env_kwargs,
            frame_skip=env_config.get("action_repeat", 1),
        )

    print(f"Environment: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create DQN agent
    print("\nCreating DQN agent...")
    agent_config = config["agent"]

    agent = DQNAgent(
        env=env,
        learning_rate=agent_config["learning_rate"],
        gamma=agent_config["gamma"],
        epsilon_start=agent_config["epsilon_start"],
        epsilon_end=agent_config["epsilon_end"],
        epsilon_decay=agent_config["epsilon_decay"],
        buffer_size=agent_config["buffer_size"],
        batch_size=agent_config["batch_size"],
        target_update=agent_config["target_update"],
        device=config.get("device", "cuda"),
    )

    print(f"Device: {agent.device}")
    print(
        f"DQN Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}"
    )

    # Create data recorder
    print("\nCreating data recorder...")
    data_config = config["data_collection"]

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        compress=data_config.get("compress", True),
        save_frequency=data_config.get("save_frequency", 10),
    )

    # Training parameters
    total_episodes = agent_config["total_episodes"]
    learning_starts = agent_config.get("learning_starts", 1000)
    train_freq = agent_config.get("train_freq", 4)
    save_freq = agent_config.get("save_freq", 100)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {total_episodes} episodes")
    print(f"{'='*60}\n")

    episode_rewards = []
    episode_lengths = []
    recent_scores = []

    global_step = 0

    for episode in tqdm(range(total_episodes), desc="Training"):
        state, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record step
            recorder.add_transition(
                observation=state,
                action=action,
                reward=reward,
                next_observation=next_state,
                terminated=terminated,
                truncated=truncated,
                metadata={"environment": "chrome_dino"},
            )

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent (after learning_starts steps)
            if global_step >= learning_starts and global_step % train_freq == 0:
                loss = agent.train_step()

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1

        # Update epsilon
        agent.update_epsilon()

        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_scores.append(info.get("score", 0))

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_score = np.mean(recent_scores[-10:])

            tqdm.write(
                f"Episode {episode + 1}/{total_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.1f} | "
                f"Avg Score: {avg_score:.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Steps: {global_step}"
            )

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = checkpoint_dir / f"dqn_episode_{episode + 1}.pt"
            agent.save(checkpoint_path)
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")

        # Update agent's episode counter
        agent.episodes_done = episode + 1

    # Finalize recording
    print("\n" + "=" * 60)
    print("Training complete! Finalizing recordings...")
    recorder.finalize()

    # Save final agent
    final_checkpoint = checkpoint_dir / "dqn_final.pt"
    agent.save(final_checkpoint)
    print(f"Saved final agent to {final_checkpoint}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Training Statistics:")
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {global_step}")
    print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Average length (last 100): {np.mean(episode_lengths[-100:]):.1f}")
    print(f"Average score (last 100): {np.mean(recent_scores[-100:]):.1f}")
    print(f"Total frames recorded: {recorder.total_frames}")
    print("=" * 60)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on Chrome Dino")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier1_chrome_dino.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    config = load_config(str(config_path))

    # Override config with command line arguments
    if args.episodes:
        config["agent"]["total_episodes"] = args.episodes

    if args.output:
        config["data_dir"] = args.output

    if args.device:
        config["device"] = args.device

    validate_config(config)

    # Train agent
    train_dqn_agent(config)


if __name__ == "__main__":
    main()
