"""Public GameNGen package namespace.

The original research prototype exposed its implementation as ``src``.  The
public package is now ``gamengen``; ``src`` remains supported while scripts are
migrated incrementally.
"""

from importlib import import_module
from pathlib import Path

import src as _legacy

__version__ = "0.2.0"

# Let normal import machinery resolve ``gamengen.diffusion`` et al. from the
# compatibility implementation without duplicating modules or moving user data.
__path__ = list(_legacy.__path__)


def __getattr__(name: str):
    return getattr(import_module("src"), name)


def package_root() -> Path:
    """Return the repository/package root for CLI and configuration discovery."""
    return Path(__file__).resolve().parent.parent
