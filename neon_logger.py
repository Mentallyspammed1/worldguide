#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neon Logger Setup (v1.2) - Enhanced Robustness & Features

Provides a function `setup_logger` to configure a Python logger instance with:
- Colorized console output using a "neon" theme via colorama (TTY only).
- Uses a custom Formatter for cleaner color handling.
- Clean, non-colorized file output.
- Optional log file rotation (size-based).
- Extensive log formatting (timestamp, level, function, line, thread).
- Custom SUCCESS log level.
- Configurable log levels via args or environment variables.
- Option to control verbosity of third-party libraries.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any

# --- Attempt to import colorama ---
try:
    from colorama import Fore, Style, Back, init as colorama_init

    # Initialize colorama (autoreset=True ensures colors reset after each print)
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Define dummy color objects if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""  # Return empty string

    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()
    COLORAMA_AVAILABLE = False
    print(
        "Warning: 'colorama' library not found. Neon console logging disabled.",
        file=sys.stderr,
    )
    print("         Install using: pip install colorama", file=sys.stderr)

# --- Custom Log Level ---
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log method to the Logger instance."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add the method to the Logger class dynamically
if not hasattr(logging.Logger, "success"):
    logging.Logger.success = log_success  # type: ignore[attr-defined]


# --- Neon Color Theme Mapping ---
LOG_LEVEL_COLORS: Dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.BLUE,
    SUCCESS_LEVEL: Fore.MAGENTA,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}


# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is a TTY.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.use_colors = COLORAMA_AVAILABLE and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record and applies colors to the level name."""
        # Store original levelname before potential modification
        original_levelname = record.levelname
        color = LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE)  # Default to white

        if self.use_colors:
            # Temporarily add color codes to the levelname for formatting
            record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"

        # Use the parent class's formatting method
        formatted_message = super().format(record)

        # Restore original levelname to prevent colored output in file logs etc.
        # if the same record object were used by multiple handlers (best practice uses separate formatters)
        record.levelname = original_levelname

        return formatted_message


# --- Log Format Strings ---
# Include thread name for better context in concurrent applications
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s %(funcName)s:%(lineno)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create formatters
console_formatter = ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


# --- Main Setup Function ---
def setup_logger(
    logger_name: str = "AppLogger",
    log_file: Optional[str] = "app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_rotation_bytes: int = 5 * 1024 * 1024,  # 5 MB default max size
    log_backup_count: int = 5,  # Keep 5 backup files
    propagate: bool = False,
    third_party_log_level: int = logging.WARNING,  # Default level for noisy libraries
) -> logging.Logger:
    """
    Sets up and configures a logger instance with neon console, clean file output,
    optional rotation, and control over third-party library logging.

    Looks for environment variables to override default levels/file path:
        - LOG_CONSOLE_LEVEL: (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - LOG_FILE_LEVEL: (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - LOG_FILE_PATH: (e.g., /path/to/your/bot.log)

    Args:
        logger_name: Name for the logger instance.
        log_file: Path to the log file. None disables file logging. Rotation enabled by default.
        console_level: Logging level for console output. Overridden by LOG_CONSOLE_LEVEL env var.
        file_level: Logging level for file output. Overridden by LOG_FILE_LEVEL env var.
        log_rotation_bytes: Max size in bytes before rotating log file. 0 disables rotation.
        log_backup_count: Number of backup log files to keep. Ignored if rotation is disabled.
        propagate: Whether to propagate messages to the root logger (default False).
        third_party_log_level: Level for common noisy libraries (ccxt, urllib3, etc.).

    Returns:
        The configured logging.Logger instance.
    """

    # --- Environment Variable Overrides ---
    env_console_level_str = os.getenv("LOG_CONSOLE_LEVEL")
    env_file_level_str = os.getenv("LOG_FILE_LEVEL")
    env_log_file = os.getenv("LOG_FILE_PATH")

    if env_console_level_str:
        env_console_level = logging.getLevelName(env_console_level_str.upper())
        if isinstance(env_console_level, int):
            print(
                f"Neon Logger: Overriding console level from env LOG_CONSOLE_LEVEL='{env_console_level_str}' -> {logging.getLevelName(env_console_level)}"
            )
            console_level = env_console_level
        else:
            print(
                f"Warning: Invalid LOG_CONSOLE_LEVEL '{env_console_level_str}'. Using default: {logging.getLevelName(console_level)}",
                file=sys.stderr,
            )

    if env_file_level_str:
        env_file_level = logging.getLevelName(env_file_level_str.upper())
        if isinstance(env_file_level, int):
            print(
                f"Neon Logger: Overriding file level from env LOG_FILE_LEVEL='{env_file_level_str}' -> {logging.getLevelName(env_file_level)}"
            )
            file_level = env_file_level
        else:
            print(
                f"Warning: Invalid LOG_FILE_LEVEL '{env_file_level_str}'. Using default: {logging.getLevelName(file_level)}",
                file=sys.stderr,
            )

    if env_log_file:
        print(
            f"Neon Logger: Overriding log file path from env LOG_FILE_PATH='{env_log_file}'"
        )
        log_file = env_log_file

    # --- Get Logger and Set Base Level/Propagation ---
    logger = logging.getLogger(logger_name)
    logger.setLevel(
        logging.DEBUG
    )  # Set logger to lowest level to capture all messages for handlers
    logger.propagate = propagate

    # --- Clear Existing Handlers (if re-configuring) ---
    if logger.hasHandlers():
        print(
            f"Logger '{logger_name}' already configured. Clearing existing handlers.",
            file=sys.stderr,
        )
        for handler in logger.handlers[:]:  # Iterate a copy
            try:
                handler.close()  # Close file handles etc.
                logger.removeHandler(handler)
            except Exception as e:
                print(
                    f"Warning: Error removing/closing existing handler: {e}",
                    file=sys.stderr,
                )

    # --- Console Handler ---
    if console_level is not None and console_level >= 0:
        try:
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setLevel(console_level)
            console_h.setFormatter(console_formatter)  # Use the colored formatter
            logger.addHandler(console_h)
            print(
                f"Neon Logger: Console logging active at level [{logging.getLevelName(console_level)}]."
            )
        except Exception as e:
            print(
                f"{Fore.RED}Error setting up console handler: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
    else:
        print("Neon Logger: Console logging disabled.")

    # --- File Handler (with optional rotation) ---
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

            if log_rotation_bytes > 0 and log_backup_count >= 0:
                # Use Rotating File Handler
                file_h = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=log_rotation_bytes,
                    backupCount=log_backup_count,
                    encoding="utf-8",
                )
                print(
                    f"Neon Logger: Rotating file logging active at level [{logging.getLevelName(file_level)}] to '{log_file}' (Max: {log_rotation_bytes / 1024 / 1024:.1f} MB, Backups: {log_backup_count})."
                )
            else:
                # Use basic File Handler (no rotation)
                file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                print(
                    f"Neon Logger: Basic file logging active at level [{logging.getLevelName(file_level)}] to '{log_file}' (Rotation disabled)."
                )

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter)  # Use the plain (non-colored) formatter
            logger.addHandler(file_h)

        except IOError as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Fatal Error configuring log file '{log_file}': {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Unexpected error setting up file logging: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
    else:
        print("Neon Logger: File logging disabled.")

    # --- Configure Third-Party Log Levels ---
    if third_party_log_level is not None and third_party_log_level >= 0:
        noisy_libraries = [
            "ccxt",
            "urllib3",
            "requests",
            "asyncio",
        ]  # Add others if needed
        print(
            f"Neon Logger: Setting third-party library log level to [{logging.getLevelName(third_party_log_level)}]."
        )
        for lib_name in noisy_libraries:
            try:
                logging.getLogger(lib_name).setLevel(third_party_log_level)
                # Optionally add a specific handler or just let propagation handle it (if enabled)
            except Exception:
                # This might fail if library doesn't use standard logging, ignore gracefully
                # logger.debug(f"Could not set level for third-party logger '{lib_name}': {e}")
                pass
    else:
        print("Neon Logger: Third-party library log level control disabled.")

    # --- Log Test Messages ---
    logger.debug("--- Logger Setup Complete (DEBUG Test) ---")
    logger.info("--- Logger Setup Complete (INFO Test) ---")
    logger.success("--- Logger Setup Complete (SUCCESS Test) ---")
    logger.warning("--- Logger Setup Complete (WARNING Test) ---")
    logger.error("--- Logger Setup Complete (ERROR Test) ---")
    logger.critical("--- Logger Setup Complete (CRITICAL Test) ---")

    return logger


# --- Example Usage ---
if __name__ == "__main__":
    print("-" * 60)
    print("--- Example Neon Logger v1.2 Usage ---")
    print("-" * 60)

    # Example 1: Basic setup with defaults + rotation
    print("\n--- Example 1: Basic Setup (Rotation Enabled) ---")
    basic_logger = setup_logger("BasicBot", "basic_bot.log")
    basic_logger.info("This goes to console (INFO) and file (INFO).")
    basic_logger.debug("This goes ONLY to file (DEBUG).")

    # Example 2: Verbose console, less verbose file, no rotation
    print("\n--- Example 2: Verbose Console, Warning File, No Rotation ---")
    verbose_logger = setup_logger(
        logger_name="VerboseBot",
        log_file="verbose_bot_warn.log",
        console_level=logging.DEBUG,  # Show DEBUG on console
        file_level=logging.WARNING,  # Only WARNING and above in file
        log_rotation_bytes=0,  # Disable rotation
    )
    verbose_logger.debug("Verbose DEBUG on console.")
    verbose_logger.info("Verbose INFO on console.")
    verbose_logger.warning("Warning on console AND file.")

    # Example 3: Using environment variables
    print("\n--- Example 3: Using Environment Variables ---")
    print(
        "Set LOG_CONSOLE_LEVEL=WARNING, LOG_FILE_LEVEL=ERROR, LOG_FILE_PATH=env_bot.log"
    )
    # Set env vars before calling setup (in a real script, these would be set externally)
    os.environ["LOG_CONSOLE_LEVEL"] = "WARNING"
    os.environ["LOG_FILE_LEVEL"] = "ERROR"
    os.environ["LOG_FILE_PATH"] = "env_bot.log"
    env_logger = setup_logger(
        logger_name="EnvBot",
        # Args below will be overridden by env vars
        log_file="default.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    env_logger.info("This INFO message should NOT appear on console (Env WARNING).")
    env_logger.warning("This WARNING message should appear on console (Env WARNING).")
    env_logger.error(
        "This ERROR message should appear on console AND in env_bot.log (Env ERROR)."
    )
    # Cleanup env vars for subsequent tests if needed
    del os.environ["LOG_CONSOLE_LEVEL"]
    del os.environ["LOG_FILE_LEVEL"]
    del os.environ["LOG_FILE_PATH"]

    # Example 4: Quieting third-party libraries
    print("\n--- Example 4: Quieting Third-Party Libs ---")
    quiet_tp_logger = setup_logger(
        logger_name="QuietBot",
        log_file=None,  # Disable file logging
        console_level=logging.DEBUG,
        third_party_log_level=logging.ERROR,  # Only show ERRORS from ccxt etc.
    )
    # Simulate a third-party log message
    logging.getLogger("ccxt").info("This ccxt INFO message should NOT appear.")
    logging.getLogger("urllib3").warning(
        "This urllib3 WARNING message should NOT appear."
    )
    logging.getLogger("ccxt").error("This ccxt ERROR message SHOULD appear.")
    quiet_tp_logger.info("This QuietBot INFO message SHOULD appear.")

    print("-" * 60)
    print("Check console output for neon colors (if supported).")
    print(
        "Check log files ('basic_bot.log', 'verbose_bot_warn.log', 'env_bot.log') for output."
    )
    print("-" * 60)
