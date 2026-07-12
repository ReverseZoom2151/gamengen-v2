"""
DQN Agent for Chrome Dino Game
Based on paper's Appendix A.10 - uses DQN with experience replay
"""

import random
from collections import deque
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    import gymnasium as gym


class DQNNetwork(nn.Module):
    """Deep Q-Network for Chrome Dino"""

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()

        # Input shape: (H, W, C)
        channels = input_shape[2]

        # Convolutional layers
        self.conv = nn.Sequential(
            # First conv block
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Third conv block
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate size of flattened features
        conv_out_size = self._get_conv_output(input_shape)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_conv_output(self, shape):
        """Calculate convolutional output size"""
        with torch.no_grad():
            # Create dummy input (channels first for PyTorch)
            dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
            output = self.conv(dummy_input)
            return int(np.prod(output.size()))

    def forward(self, x):
        """Forward pass"""
        # Input: (batch, H, W, C) -> need (batch, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Permute to channels first
        x = x.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        x = x.float() / 255.0

        # Convolutional layers
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        q_values = self.fc(x)

        return q_values


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def state_dict(self) -> dict:
        return {"capacity": self.buffer.maxlen, "transitions": list(self.buffer)}

    def load_state_dict(self, state: dict) -> None:
        self.buffer = deque(
            state.get("transitions", []),
            maxlen=state.get("capacity", self.buffer.maxlen),
        )


class DQNAgent:
    """DQN Agent for training"""

    def __init__(
        self,
        env: "gym.Env",
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 1000,
        device: str = "cuda",
    ):
        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Get environment info
        self.num_actions = env.action_space.n
        self.obs_shape = env.observation_space.shape

        # Create networks
        self.policy_net = DQNNetwork(self.obs_shape, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.obs_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Training stats
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_t = torch.from_numpy(state).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Compute Q(s, a)
        q_values = self.policy_net(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save agent"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "episodes_done": self.episodes_done,
                "epsilon": self.epsilon,
                "replay_buffer": self.memory.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            path,
        )

    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint.get("steps_done", 0)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        if "replay_buffer" in checkpoint:
            self.memory.load_state_dict(checkpoint["replay_buffer"])
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])


if __name__ == "__main__":
    # Test DQN network
    print("Testing DQN Network...")

    # Create dummy environment to test
    from src.environment.chrome_dino_env import SimpleDinoEnv

    env = SimpleDinoEnv(width=512, height=256)
    agent = DQNAgent(env)

    print(f"Device: {agent.device}")
    print(f"Policy Network:\n{agent.policy_net}")
    print(
        f"Number of parameters: {sum(p.numel() for p in agent.policy_net.parameters())}"
    )

    # Test forward pass
    state, _ = env.reset()
    action = agent.select_action(state)
    print(f"\nSample action: {action}")

    env.close()
    print("Test complete!")
