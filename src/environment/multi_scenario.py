"""
Multi-Scenario Training Support for DOOM
Paper Tier 3 uses multiple DOOM scenarios for diversity
"""

from pathlib import Path
from typing import List, Optional

import gymnasium as gym
from src.environment.doom_contracts import ScenarioSelector


class MultiScenarioViZDoomEnv(gym.Env):
    """
    Wrapper for training on multiple DOOM scenarios

    Paper mentions training on diverse scenarios to ensure the model
    generalizes across different game situations.

    Common scenarios:
    - basic.cfg: Basic navigation
    - deadly_corridor.cfg: Combat focus
    - defend_the_center.cfg: Defense scenario
    - defend_the_line.cfg: Alternative defense
    - health_gathering.cfg: Resource management
    """

    # Available scenarios
    SCENARIOS = [
        "basic.cfg",
        "deadly_corridor.cfg",
        "defend_the_center.cfg",
        "defend_the_line.cfg",
        "health_gathering.cfg",
    ]

    def __init__(
        self,
        scenarios: Optional[List[str]] = None,
        scenario_weights: Optional[List[float]] = None,
        change_scenario_every_n_episodes: int = 10,
        width: int = 320,
        height: int = 256,
        frame_skip: int = 4,
        use_paper_reward: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            scenarios: List of scenario config files (or None for all)
            scenario_weights: Probability weights for each scenario
            change_scenario_every_n_episodes: Switch scenario frequency
            width: Frame width
            height: Frame height
            frame_skip: Apply each action for N frames
            use_paper_reward: Use paper's reward function
        """
        super().__init__()

        # Use all scenarios if none specified
        if scenarios is None:
            scenarios = self.SCENARIOS

        self.scenarios = scenarios
        self.scenario_weights = scenario_weights
        self.change_every_n = change_scenario_every_n_episodes

        self.width = width
        self.height = height
        self.frame_skip = frame_skip
        self.use_paper_reward = use_paper_reward
        self._factory_kwargs = dict(kwargs)
        self.selector = ScenarioSelector(self.scenarios, self.scenario_weights, seed)

        # Current scenario
        self.current_scenario_idx = 0
        self.episode_count = 0

        # Import here to avoid circular imports
        from src.environment.vizdoom_env import create_vizdoom_env

        # Create initial environment
        self.env = create_vizdoom_env(
            scenario=self._get_scenario_name(scenarios[0]),
            width=width,
            height=height,
            frame_skip=frame_skip,
            use_paper_reward=use_paper_reward,
            **self._factory_kwargs,
        )

        # Use same spaces as underlying environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print(f"Multi-scenario environment created with {len(scenarios)} scenarios")

    def _get_scenario_name(self, config_file: str) -> str:
        """Extract scenario name from config file"""
        return Path(config_file).stem

    def _should_change_scenario(self) -> bool:
        """Check if we should change scenario"""
        return self.episode_count % self.change_every_n == 0 and self.episode_count > 0

    def _select_next_scenario(self) -> str:
        """Select next scenario (weighted random or sequential)"""
        scenario = self.selector.next()
        self.current_scenario_idx = self.selector.index
        return scenario

    def reset(self, *, seed=None, options=None):
        """Reset environment, potentially changing scenario"""

        # Check if we should change scenario
        if self._should_change_scenario():
            next_scenario = self._select_next_scenario()

            print(f"\nSwitching to scenario: {next_scenario}")

            # Close current environment
            self.env.close()

            # Create new environment with different scenario
            from src.environment.vizdoom_env import create_vizdoom_env

            self.env = create_vizdoom_env(
                scenario=self._get_scenario_name(next_scenario),
                width=self.width,
                height=self.height,
                frame_skip=self.frame_skip,
                use_paper_reward=self.use_paper_reward,
                **self._factory_kwargs,
            )

        self.episode_count += 1

        if seed is not None:
            self.selector.reset_seed(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Forward to underlying environment"""
        return self.env.step(action)

    def render(self):
        """Forward to underlying environment"""
        return self.env.render()

    def close(self):
        """Close underlying environment"""
        self.env.close()

    def get_current_scenario(self) -> str:
        """Get current scenario name"""
        return self.scenarios[self.current_scenario_idx]


def create_multi_scenario_env(
    num_envs: int = 1, scenarios: Optional[List[str]] = None, **kwargs
) -> gym.Env:
    """
    Factory function to create multi-scenario environment(s)

    Args:
        num_envs: Number of parallel environments
        scenarios: List of scenarios to use
        **kwargs: Additional arguments for MultiScenarioViZDoomEnv

    Returns:
        Single environment or vectorized environment
    """
    if num_envs == 1:
        return MultiScenarioViZDoomEnv(scenarios=scenarios, **kwargs)

    else:
        # Create vectorized environment
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env(rank):
            def _init():
                return MultiScenarioViZDoomEnv(scenarios=scenarios, **kwargs)

            return _init

        return SubprocVecEnv([make_env(i) for i in range(num_envs)])


if __name__ == "__main__":
    # Test multi-scenario environment
    print("Testing Multi-Scenario Environment...")

    try:
        env = MultiScenarioViZDoomEnv(
            scenarios=["basic.cfg", "deadly_corridor.cfg"],
            change_scenario_every_n_episodes=2,
        )

        print("Environment created!")
        print(f"Current scenario: {env.get_current_scenario()}")

        # Test a few episodes
        for ep in range(5):
            print(f"\nEpisode {ep + 1}")
            obs, info = env.reset()
            print(f"  Scenario: {env.get_current_scenario()}")
            print(f"  Observation shape: {obs.shape}")

            # Take a few steps
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)

                if done:
                    break

        env.close()
        print("\nTest complete!")

    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: Requires ViZDoom installation")
        import traceback

        traceback.print_exc()
