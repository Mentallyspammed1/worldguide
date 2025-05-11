# File: logger_setup.py
import logging
import os
import sys
from typing import Any

# Third-party Libraries - colorama is used for colored logging
# colorama_init should be called by the main script before this module is heavily used.
try:
    from colorama import Fore, Back, Style
except ImportError:
    # Create dummy Fore, Back, Style if colorama is not available
    # to prevent NameError, though colors won't work.
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Back, Style = DummyColor(), DummyColor(), DummyColor()

# --- Logger Setup ---
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger("PyrmethusBot") # Named logger

# Custom SUCCESS level and Neon Color Formatting
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)  # pylint: disable=protected-access

logging.Logger.success = log_success # type: ignore[attr-defined]

# Apply colors if TTY
# This part might run before colorama_init if logger_setup is imported very early.
# It's generally safe as colorama handles uninitialized state gracefully by not outputting sequences.
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(
        logging.CRITICAL,
        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}",
    )

# End of logger_setup.py
```

```python
