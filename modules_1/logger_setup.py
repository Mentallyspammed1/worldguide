# File: logger_setup.py
"""
Configures the application's logging system.

Sets up handlers for console (stdout/stderr) and rotating files,
using a custom formatter for timezone-aware timestamps and sensitive data masking.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone as dt_timezone  # For timezone aware datetime objects
from typing import Optional  # For type hinting

# Import utility functions and classes
# Assumes utils.py provides:
# - LOG_DIRECTORY: Path to the directory where log files should be stored.
# - SensitiveFormatter: A logging.Formatter subclass that masks sensitive data.
# - get_timezone(): A function that returns a timezone object (e.g., from pytz).
try:
    from utils import LOG_DIRECTORY, SensitiveFormatter, get_timezone
except ImportError:
    print("ERROR: Failed to import required components from 'utils'. Ensure 'utils.py' exists.", file=sys.stderr)
    # Define fallbacks to allow basic script execution, though logging will be limited/incorrect
    LOG_DIRECTORY = os.path.join(os.getcwd(), "logs")
    # Use Python's built-in timezone if pytz/utils not available
    try:
        from zoneinfo import ZoneInfo

        get_timezone = lambda: ZoneInfo("UTC")  # Python 3.9+
    except ImportError:
        from datetime import timezone as dt_tz

        get_timezone = lambda: dt_tz.utc  # Fallback UTC

    # Define a basic SensitiveFormatter if not available
    try:
        SensitiveFormatter  # Check if already defined
    except NameError:

        class SensitiveFormatter(logging.Formatter):
            """Basic fallback formatter."""

            def format(self, record):
                return super().format(record)

        print("Warning: Using fallback basic Formatter instead of SensitiveFormatter.", file=sys.stderr)


# --- Custom Formatter to Handle Timezone ---
class TimezoneAwareFormatter(SensitiveFormatter):
    """
    A logging formatter that includes timezone information in timestamps.
    Uses a specified timezone object for localization. Inherits from SensitiveFormatter.
    """

    def __init__(
        self, fmt: str = None, datefmt: str = None, style: str = "%", timezone: Optional[datetime.tzinfo] = None
    ):
        """Initializes the formatter with timezone support."""
        super().__init__(fmt, datefmt, style)
        try:
            self.timezone = timezone or get_timezone()
            if not isinstance(self.timezone, datetime.tzinfo):
                raise TypeError("Timezone must be a datetime.tzinfo instance.")
        except Exception as e:
            print(f"Warning: Error getting timezone: {e}. Defaulting to UTC.", file=sys.stderr)
            from datetime import timezone as dt_tz  # Ensure fallback import

            self.timezone = dt_tz.utc

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        """Formats the log record's creation time using the configured timezone."""
        # Create a timezone-aware datetime object in UTC
        utc_dt = datetime.fromtimestamp(record.created, tz=dt_timezone.utc)
        # Convert to the target timezone
        local_dt = utc_dt.astimezone(self.timezone)
        # Use the effective date format string
        effective_datefmt = datefmt or self.datefmt
        if effective_datefmt:
            s = local_dt.strftime(effective_datefmt)
        else:  # Default format if no datefmt provided
            s = local_dt.strftime("%Y-%m-%d %H:%M:%S")
            ms = int(local_dt.microsecond / 1000)
            s = f"{s},{ms:03d} {local_dt.strftime('%Z%z')}"  # Include TZ Name and Offset
        return s


# --- Global Logging Configuration Function ---
_logging_configured = False  # Module-level flag


def configure_logging(config: dict):
    """
    Configures the root logger based on application configuration.
    Sets up console (stdout INFO+, stderr WARNING+) and file handlers (DEBUG+).
    """
    global _logging_configured
    if _logging_configured:
        logging.getLogger(__name__).info("Logging already configured. Skipping.")
        return

    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    try:
        tz = get_timezone()
    except Exception as e:
        print(f"CRITICAL: Failed get timezone: {e}. Using UTC.", file=sys.stderr)
        from datetime import timezone as dt_tz

        tz = dt_tz.utc

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all messages at root

    # Clear existing handlers
    if root_logger.hasHandlers():
        print("Clearing existing root logger handlers...", file=sys.stderr)
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                print(f"Warning: Error removing handler {handler}: {e}", file=sys.stderr)

    # Ensure log directory exists
    log_dir_exists = False
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        log_dir_exists = True
    except OSError as e:
        print(f"CRITICAL: Cannot create log dir '{LOG_DIRECTORY}': {e}. File logging disabled.", file=sys.stderr)

    # --- Console Handler (stdout) ---
    try:
        console_stdout = logging.StreamHandler(sys.stdout)
        fmt_stdout = TimezoneAwareFormatter(
            fmt="%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z", timezone=tz
        )
        console_stdout.setFormatter(fmt_stdout)
        console_stdout.setLevel(log_level)  # Filter at handler level
        root_logger.addHandler(console_stdout)
    except Exception as e:
        print(f"CRITICAL: Failed setup stdout handler: {e}", file=sys.stderr)

    # --- Console Handler (stderr) ---
    try:
        console_stderr = logging.StreamHandler(sys.stderr)
        fmt_stderr = TimezoneAwareFormatter(  # Use same format for consistency
            fmt="%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z", timezone=tz
        )
        console_stderr.setFormatter(fmt_stderr)
        console_stderr.setLevel(logging.WARNING)  # Only WARNING and above
        root_logger.addHandler(console_stderr)
    except Exception as e:
        print(f"CRITICAL: Failed setup stderr handler: {e}", file=sys.stderr)

    # --- Rotating File Handler ---
    if log_dir_exists:
        log_filepath = os.path.join(LOG_DIRECTORY, "xrscalper_bot_main.log")  # Use a more specific name
        try:
            file_handler = RotatingFileHandler(
                log_filepath, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=True
            )
            fmt_file = TimezoneAwareFormatter(
                fmt="%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(threadName)s - %(message)s",  # More detailed format
                datefmt="%Y-%m-%d %H:%M:%S",
                timezone=tz,
            )  # Datefmt for the date part, formatter adds ms, TZ etc.
            file_handler.setFormatter(fmt_file)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.critical(f"Failed setup file logger {log_filepath}: {e}", exc_info=True)
            print(f"CRITICAL: Failed setup file logger {log_filepath}: {e}", file=sys.stderr)
    else:
        logging.warning(f"Log directory '{LOG_DIRECTORY}' unusable. File logging disabled.")

    # --- Library Log Levels ---
    library_log_levels = config.get("library_log_levels", {})
    for lib_name, level_str in library_log_levels.items():
        lib_level = getattr(logging, level_str.upper(), None)
        if lib_level:
            logging.getLogger(lib_name).setLevel(lib_level)
            logging.info(f"Set log level for library '{lib_name}' to {level_str.upper()}")
        else:
            logging.warning(f"Invalid log level '{level_str}' for lib '{lib_name}'")
    if not library_log_levels:  # Apply defaults if not configured
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("ccxt").setLevel(logging.INFO)  # Show INFO for CCXT by default

    init_logger = logging.getLogger("App.Init")  # Use hierarchical name
    init_logger.info(f"Logging configured. Console Level: {logging.getLevelName(log_level)}, File Level: DEBUG")
    _logging_configured = True
