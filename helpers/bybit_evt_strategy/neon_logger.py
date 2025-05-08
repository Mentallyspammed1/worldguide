# --- START OF FILE neon_logger.py ---

# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neon Logger Setup (v1.2) - Enhanced Robustness & Features

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
from typing import Any

# --- Attempt to import colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

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
    print("Warning: 'colorama' library not found. Neon console logging disabled.", file=sys.stderr)
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
LOG_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.BLUE,
    SUCCESS_LEVEL: Fore.MAGENTA,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}


# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is a TTY.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: str = "%", validate: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.use_colors = COLORAMA_AVAILABLE and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

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
    log_file: str | None = "app.log",
    console_level_str: str = "INFO",  # Changed to string input
    file_level_str: str = "DEBUG",  # Changed to string input
    log_rotation_bytes: int = 5 * 1024 * 1024,  # 5 MB default max size
    log_backup_count: int = 5,  # Keep 5 backup files
    propagate: bool = False,
    third_party_log_level_str: str = "WARNING",  # Changed to string input
) -> logging.Logger:
    """Sets up and configures a logger instance with neon console, clean file output,
    optional rotation, and control over third-party library logging.

    Reads levels as strings and converts them internally.

    Args:
        logger_name: Name for the logger instance.
        log_file: Path to the log file. None disables file logging. Rotation enabled by default.
        console_level_str: Logging level for console output (e.g., "INFO").
        file_level_str: Logging level for file output (e.g., "DEBUG").
        log_rotation_bytes: Max size in bytes before rotating log file. 0 disables rotation.
        log_backup_count: Number of backup log files to keep. Ignored if rotation is disabled.
        propagate: Whether to propagate messages to the root logger (default False).
        third_party_log_level_str: Level for common noisy libraries (e.g., "WARNING").

    Returns:
        The configured logging.Logger instance.
    """
    func_name = "setup_logger"  # For internal logging if needed

    # --- Convert string levels to logging constants ---
    try:
        console_level = logging.getLevelName(console_level_str.upper())
        file_level = logging.getLevelName(file_level_str.upper())
        third_party_log_level = logging.getLevelName(third_party_log_level_str.upper())

        if not isinstance(console_level, int):
            print(
                f"\033[93mWarning [{func_name}]: Invalid console log level string '{console_level_str}'. Using INFO.\033[0m",
                file=sys.stderr,
            )
            console_level = logging.INFO
        if not isinstance(file_level, int):
            print(
                f"\033[93mWarning [{func_name}]: Invalid file log level string '{file_level_str}'. Using DEBUG.\033[0m",
                file=sys.stderr,
            )
            file_level = logging.DEBUG
        if not isinstance(third_party_log_level, int):
            print(
                f"\033[93mWarning [{func_name}]: Invalid third-party log level string '{third_party_log_level_str}'. Using WARNING.\033[0m",
                file=sys.stderr,
            )
            third_party_log_level = logging.WARNING

    except Exception as e:
        print(
            f"\033[91mError [{func_name}]: Failed converting log level strings: {e}. Using defaults.\033[0m",
            file=sys.stderr,
        )
        console_level, file_level, third_party_log_level = logging.INFO, logging.DEBUG, logging.WARNING

    # --- Get Logger and Set Base Level/Propagation ---
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set logger to lowest level to capture all messages for handlers
    logger.propagate = propagate

    # --- Clear Existing Handlers (if re-configuring) ---
    if logger.hasHandlers():
        print(
            f"\033[94mInfo [{func_name}]: Logger '{logger_name}' already configured. Clearing handlers.\033[0m",
            file=sys.stderr,
        )
        for handler in logger.handlers[:]:  # Iterate a copy
            try:
                handler.close()  # Close file handles etc.
                logger.removeHandler(handler)
            except Exception as e:
                print(f"\033[93mWarning [{func_name}]: Error removing/closing handler: {e}\033[0m", file=sys.stderr)

    # --- Console Handler ---
    if console_level is not None and console_level >= 0:
        try:
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setLevel(console_level)
            console_h.setFormatter(console_formatter)  # Use the colored formatter
            logger.addHandler(console_h)
            print(
                f"\033[94m[{func_name}] Console logging active at level [{logging.getLevelName(console_level)}].\033[0m"
            )
        except Exception as e:
            print(f"\033[91mError [{func_name}] setting up console handler: {e}\033[0m", file=sys.stderr)
    else:
        print(f"\033[94m[{func_name}] Console logging disabled.\033[0m")

    # --- File Handler (with optional rotation) ---
    if log_file:
        try:
            log_file_path = os.path.abspath(log_file)  # Use absolute path
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

            if log_rotation_bytes > 0 and log_backup_count >= 0:
                # Use Rotating File Handler
                file_h = logging.handlers.RotatingFileHandler(
                    log_file_path,  # Use absolute path
                    maxBytes=log_rotation_bytes,
                    backupCount=log_backup_count,
                    encoding="utf-8",
                )
                log_type = "Rotating file"
                log_details = f"(Max: {log_rotation_bytes / 1024 / 1024:.1f} MB, Backups: {log_backup_count})"
            else:
                # Use basic File Handler (no rotation)
                file_h = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
                log_type = "Basic file"
                log_details = "(Rotation disabled)"

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter)  # Use the plain (non-colored) formatter
            logger.addHandler(file_h)
            print(
                f"\033[94m[{func_name}] {log_type} logging active at level [{logging.getLevelName(file_level)}] to '{log_file_path}' {log_details}.\033[0m"
            )

        except OSError as e:
            print(
                f"\033[91mFATAL [{func_name}] Error configuring log file '{log_file}': {e}. File logging disabled.\033[0m",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"\033[91mError [{func_name}] Unexpected error setting up file logging: {e}. File logging disabled.\033[0m",
                file=sys.stderr,
            )
    else:
        print(f"\033[94m[{func_name}] File logging disabled.\033[0m")

    # --- Configure Third-Party Log Levels ---
    if third_party_log_level is not None and third_party_log_level >= 0:
        noisy_libraries = ["ccxt", "urllib3", "requests", "asyncio", "websockets"]  # Add others if needed
        print(
            f"\033[94m[{func_name}] Setting third-party library log level to [{logging.getLevelName(third_party_log_level)}].\033[0m"
        )
        for lib_name in noisy_libraries:
            try:
                lib_logger = logging.getLogger(lib_name)
                if lib_logger:  # Check if logger exists
                    lib_logger.setLevel(third_party_log_level)
                    lib_logger.propagate = False  # Stop noisy logs from reaching our handlers
            except Exception as e_lib:
                # Non-critical error
                print(
                    f"\033[93mWarning [{func_name}]: Could not set level for lib '{lib_name}': {e_lib}\033[0m",
                    file=sys.stderr,
                )
    else:
        print(f"\033[94m[{func_name}] Third-party library log level control disabled.\033[0m")

    # --- Log Test Messages ---
    # logger.debug("--- Logger Setup Complete (DEBUG Test) ---")
    # logger.info("--- Logger Setup Complete (INFO Test) ---")
    # logger.success("--- Logger Setup Complete (SUCCESS Test) ---")
    # logger.warning("--- Logger Setup Complete (WARNING Test) ---")
    # logger.error("--- Logger Setup Complete (ERROR Test) ---")
    # logger.critical("--- Logger Setup Complete (CRITICAL Test) ---")
    logger.info(f"--- Logger '{logger_name}' Setup Complete ---")

    # Cast to include the 'success' method for type hinting upstream
    return logger  # type: ignore


# --- Example Usage ---
if __name__ == "__main__":
    print("-" * 60)
    print("--- Example Neon Logger v1.2 Usage ---")
    print("-" * 60)
    # Example: Set environment variables for testing overrides
    # os.environ["LOG_CONSOLE_LEVEL"] = "DEBUG"
    # os.environ["LOG_FILE_PATH"] = "test_override.log"

    # Basic setup
    logger_instance = setup_logger(
        logger_name="ExampleLogger",
        log_file="example_app.log",
        console_level_str="INFO",  # Use string levels
        file_level_str="DEBUG",
        third_party_log_level_str="WARNING",
    )

    # Log messages at different levels
    logger_instance.debug("This is a detailed debug message (might only go to file).")
    logger_instance.info("This is an informational message.")
    logger_instance.success("Operation completed successfully!")  # Custom level
    logger_instance.warning("This is a warning message.")
    logger_instance.error("An error occurred during processing.")
    try:
        1 / 0
    except ZeroDivisionError:
        logger_instance.critical("A critical error (division by zero) happened!", exc_info=True)

    # Test third-party level suppression (if ccxt installed)
    try:
        import ccxt

        ccxt_logger = logging.getLogger("ccxt")
        print(f"CCXT logger level: {logging.getLevelName(ccxt_logger.getEffectiveLevel())}")
        ccxt_logger.info("This CCXT INFO message should be suppressed by default.")
        ccxt_logger.warning("This CCXT WARNING message should appear.")
    except ImportError:
        print("CCXT not installed, skipping third-party logger test.")

    print("\nCheck console output and log files created ('example_app.log').")
    # Clean up env vars if set for test
    # os.environ.pop("LOG_CONSOLE_LEVEL", None)
    # os.environ.pop("LOG_FILE_PATH", None)

# --- END OF FILE neon_logger.py ---
