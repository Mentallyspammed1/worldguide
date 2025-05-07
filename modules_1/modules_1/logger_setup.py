# File: logger_setup.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone as dt_timezone  # For timezone aware datetime objects

# Import utility functions and classes
from utils import LOG_DIRECTORY, SensitiveFormatter, get_timezone


# --- Custom Formatter to Handle Timezone ---
class TimezoneAwareFormatter(SensitiveFormatter):
    """
    A logging formatter that formats time including timezone info
    using the provided timezone object, without modifying global state.
    Inherits sensitive data masking capabilities.
    """

    def __init__(self, fmt=None, datefmt=None, style="%", timezone=None):
        super().__init__(fmt, datefmt, style)
        self.timezone = timezone or get_timezone()  # Use provided TZ or default from utils

    # Override formatTime to use timezone-aware datetime
    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as text.
        This implementation uses the timezone object provided during
        formatter initialization.
        """
        # record.created is a Unix timestamp (seconds since epoch in UTC)
        # Create a datetime object from the timestamp, making it UTC-aware
        utc_dt = datetime.fromtimestamp(record.created, tz=dt_timezone.utc)
        # Convert to the target timezone
        aware_dt = utc_dt.astimezone(self.timezone)

        if datefmt:
            s = aware_dt.strftime(datefmt)
        else:
            # Default format for the time part of the log record if datefmt is not specified
            # The full log format (including date, level, message) is defined by `fmt` in __init__
            s = aware_dt.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{s},{aware_dt.microsecond // 1000:03d}"  # Append milliseconds
        return s


# --- Global Logging Configuration Function ---
def configure_logging(config: dict):
    """
    Configures the root logger based on application configuration.
    Sets up console and file handlers with appropriate levels and formatting.
    """
    # Get desired log level from config, default to INFO
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get the configured timezone object
    tz = get_timezone()

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Root logger should capture everything

    # Prevent duplicate handlers if configure_logging is somehow called multiple times
    if root_logger.hasHandlers():
        print("Clearing existing root logger handlers...", file=sys.stderr)
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                print(f"Warning: Error removing/closing root handler: {e}", file=sys.stderr)

    # Ensure log directory exists
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = TimezoneAwareFormatter(
        "%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",  # Include Timezone Abbreviation
        timezone=tz,
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # --- Error Console Handler (for stderr) ---
    error_console_handler = logging.StreamHandler(sys.stderr)
    error_console_handler.setFormatter(console_formatter)  # Use same formatter
    error_console_handler.setLevel(logging.WARNING)  # Only show warnings/errors on stderr
    # Check if a similar stderr handler already exists to avoid duplicates
    stderr_exists = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in root_logger.handlers)
    if not stderr_exists:
        root_logger.addHandler(error_console_handler)
    else:
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                handler.setLevel(min(handler.level, logging.WARNING))

    # --- Main File Handler ---
    main_log_filename = os.path.join(LOG_DIRECTORY, "xrscalper_bot.log")
    try:
        file_handler = RotatingFileHandler(
            main_log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_formatter = TimezoneAwareFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S,%f",  # Include milliseconds and full date for file
            timezone=tz,
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not set up main file logger {main_log_filename}: {e}", file=sys.stderr)

    logging.getLogger("xrscalper_bot_init").info(
        f"Logging configured successfully with level: {logging.getLevelName(log_level)}"
    )
