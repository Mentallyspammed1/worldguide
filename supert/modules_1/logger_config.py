# File: logger_config.py
"""
Handles the setup and configuration of the application logger.
Includes custom logging levels and colorized output for TTY environments.
"""
import logging
import os
import sys
from typing import Any

try:
    from colorama import Back, Fore, Style
except ImportError:
    # Fallback if colorama is not installed, though main script should catch this.
    class Fore: pass
    class Back: pass
    class Style: pass
    for name in ['RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE', 'RESET', 'DIM', 'BRIGHT', 'NORMAL']:
        setattr(Fore, name, '')
        setattr(Back, name, '')
        setattr(Style, name, '')


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": 25,  # Custom level (between INFO and WARNING)
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Set default level to INFO if the env var value is invalid
LOGGING_LEVEL: int = LOGGING_LEVEL_MAP.get(LOGGING_LEVEL_STR, logging.INFO)

# Define custom SUCCESS level if it doesn't exist
SUCCESS_LEVEL: int = LOGGING_LEVEL_MAP["SUCCESS"]
if logging.getLevelName(SUCCESS_LEVEL) == f"Level {SUCCESS_LEVEL}":  # Check if name is default
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# Configure basic logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],  # Output to console
)
logger: logging.Logger = logging.getLogger(__name__)  # Get the root logger


# Define the success method for the logger instance
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add the method to the Logger class (careful with type hinting here)
# Check if method already exists to prevent potential issues if run multiple times
if not hasattr(logging.Logger, "success"):
    logging.Logger.success = log_success  # type: ignore[attr-defined]

# Apply colors if outputting to a TTY (like Termux console or standard terminal)
if sys.stdout.isatty():
    # Define color mappings for levels
    level_colors = {
        logging.DEBUG: f"{Fore.CYAN}{Style.DIM}",
        logging.INFO: f"{Fore.BLUE}",
        SUCCESS_LEVEL: f"{Fore.MAGENTA}{Style.BRIGHT}",  # Use Bright Magenta for Success
        logging.WARNING: f"{Fore.YELLOW}{Style.BRIGHT}",
        logging.ERROR: f"{Fore.RED}{Style.BRIGHT}",
        logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
    }
    # Apply colors to level names by replacing the formatter's levelname part
    for level, color_style in level_colors.items():
        level_name = logging.getLevelName(level)
        if level == SUCCESS_LEVEL or not level_name.startswith("\033"): # type: ignore
            logging.addLevelName(level, f"{color_style}{level_name}{Style.RESET_ALL}") # type: ignore
else:
    if logging.getLevelName(SUCCESS_LEVEL) == f"Level {SUCCESS_LEVEL}":
        logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

```

```python
