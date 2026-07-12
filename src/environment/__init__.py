"""Game environment wrappers with optional runtime dependencies."""

try:
    from .chrome_dino_env import ChromeDinoEnv, SimpleDinoEnv
    CHROME_DINO_AVAILABLE = True
except ImportError:
    CHROME_DINO_AVAILABLE = False

try:
    from .vizdoom_env import (ViZDoomEnv, ViZDoomEnvWithPaperReward,
                              create_vizdoom_env)

    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False
