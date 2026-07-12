"""
Chrome Dino Game Environment for GameNGen
Implements a Gymnasium-compatible wrapper for Chrome Dino game
"""

import io
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ChromeDinoEnv(gym.Env):
    """
    Chrome Dino Game Environment

    Observation Space: RGB image (256, 512, 3)
    Action Space: Discrete(3) - [no action, jump, duck]
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        width: int = 512,
        height: int = 256,
        grayscale: bool = False,
        frame_skip: int = 1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # Action space: 0 = no action, 1 = jump, 2 = duck
        self.action_space = spaces.Discrete(3)

        # Observation space: RGB or grayscale image
        if grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(height, width, 1), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            )

        # Game state
        self.driver = None
        self.game_started = False
        self.current_score = 0
        self.max_score = 0
        self.step_count = 0

    def _init_driver(self):
        """Initialize Chrome driver with Dino game"""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By

        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run without GUI
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--window-size={self.width},{self.height}")

        self.driver = webdriver.Chrome(options=chrome_options)

        # Navigate to chrome://dino (works when offline)
        self.driver.get("chrome://dino")
        time.sleep(1)

        # Get canvas element
        self.canvas = self.driver.find_element(By.TAG_NAME, "canvas")

    def _get_game_state(self) -> Dict[str, Any]:
        """Extract game state from JavaScript"""
        try:
            script = """
            return {
                crashed: Runner.instance_.crashed,
                playing: Runner.instance_.playing,
                score: Runner.instance_.distanceRan
            };
            """
            state = self.driver.execute_script(script)
            return state
        except Exception as exc:
            raise RuntimeError("could not read Chrome Dino game state") from exc

    def _capture_frame(self) -> np.ndarray:
        """Capture game frame as numpy array"""
        import cv2
        from PIL import Image

        # Get screenshot
        png = self.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))

        # Convert to numpy array
        frame = np.array(img)

        # Resize if needed
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.expand_dims(frame, axis=-1)

        return frame

    def _send_action(self, action: int):
        """Send action to game"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys

        body = self.driver.find_element(By.TAG_NAME, "body")

        if action == 1:  # Jump
            body.send_keys(Keys.SPACE)
        elif action == 2:  # Duck
            body.send_keys(Keys.ARROW_DOWN)
        elif action != 0:
            raise ValueError(f"invalid Chrome Dino action: {action}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize driver if not done
        if self.driver is None:
            self._init_driver()

        # Restart game
        try:
            self.driver.execute_script("Runner.instance_.restart()")
        except Exception:
            # If game not started, send space to start
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys

            body = self.driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.SPACE)

        time.sleep(0.1)

        self.game_started = True
        self.current_score = 0
        self.step_count = 0

        # Get initial observation
        obs = self._capture_frame()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in environment"""
        if not self.game_started:
            raise RuntimeError("Call reset() before step()")

        # Send action and advance the requested number of game frames.
        for _ in range(self.frame_skip):
            self._send_action(action)
            time.sleep(0.05)  # ~20 FPS

        # Get game state
        state = self._get_game_state()

        # Get observation
        obs = self._capture_frame()

        # Calculate reward
        score_delta = state["score"] - self.current_score
        self.current_score = state["score"]

        # Reward: based on score increase
        reward = score_delta / 10.0  # Scale down

        # Check if game over
        terminated = state["crashed"]
        truncated = False

        if terminated:
            reward = -10.0  # Penalty for crashing

        self.step_count += 1

        # Update max score
        if self.current_score > self.max_score:
            self.max_score = self.current_score

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            "score": self.current_score,
            "max_score": self.max_score,
            "step": self.step_count,
        }

    def render(self):
        """Render environment"""
        if self.render_mode == "rgb_array":
            return self._capture_frame()
        elif self.render_mode == "human":
            frame = self._capture_frame()
            # Display frame (could use cv2.imshow for debugging)
            return frame
        return None

    def close(self):
        """Clean up resources"""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None


# Simpler version using a local HTML file (alternative)
class SimpleDinoEnv(gym.Env):
    """
    Simplified Dino Environment using local game file
    This is easier to set up than Selenium version
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, width=512, height=256):
        super().__init__()

        self.width = width
        self.height = height

        # Action space: 0 = no action, 1 = jump, 2 = duck
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

        # Game state
        self.score = 0
        self.game_over = False
        self.frame_count = 0

        print(
            "Warning: SimpleDinoEnv is a placeholder. Use ChromeDinoEnv for actual game."
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.game_over = False
        self.frame_count = 0

        # Return dummy observation
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self.frame_count += 1

        # Deterministic placeholder observation.  This environment is intended
        # only for CPU tests; training defaults to ChromeDinoEnv.
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[-4:, :, :] = 255
        obstacle_x = self.width - (self.frame_count % self.width) - 1
        obs[self.height - 24 : self.height - 4, max(0, obstacle_x - 2) : obstacle_x + 2] = 255
        reward = 0.1
        self.game_over = self.frame_count > 1000
        truncated = False
        info = {"score": self.score}

        return obs, reward, self.game_over, truncated, info

    def close(self):
        pass


if __name__ == "__main__":
    # Test environment
    print("Testing Chrome Dino Environment...")

    # Use simple version for testing
    env = SimpleDinoEnv()

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: action={action}, reward={reward:.2f}, done={done}")

        if done:
            break

    env.close()
    print("Test complete!")
