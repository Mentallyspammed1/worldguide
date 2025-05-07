# File: logger_setup.py
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import pytz # Import pytz or zoneinfo for timezone objects

# Import utility functions and classes
# Assume utils.py provides LOG_DIRECTORY, SensitiveFormatter (needs modification),
# and get_timezone() which returns a timezone object (like from pytz)
from utils import LOG_DIRECTORY, SensitiveFormatter, get_timezone

# --- Custom Formatter to Handle Timezone ---
# We need a custom formatter or modify SensitiveFormatter to format time with TZ
# without changing the global converter.
class TimezoneAwareFormatter(SensitiveFormatter):
    """
    A logging formatter that formats time including timezone info
    using the provided timezone object, without modifying global state.
    Inherits sensitive data masking capabilities.
    """
    def __init__(self, fmt=None, datefmt=None, style='%', timezone=None):
        super().__init__(fmt, datefmt, style)
        self.timezone = timezone or get_timezone() # Use provided TZ or default from utils

    # Override formatTime to use timezone-aware datetime
    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as text.

        This implementation uses the timezone object provided during
        formatter initialization.
        """
        # record.created is a Unix timestamp (seconds since epoch in UTC)
        # Convert the UTC timestamp to the desired timezone-aware datetime
        dt_utc = datetime.utcfromtimestamp(record.created)
        dt_tz_aware = pytz.utc.localize(dt_utc).astimezone(self.timezone) # Use pytz to make UTC tz-aware then convert

        if datefmt:
            # If a date format is provided, use it. Ensure it can handle timezone (%Z or %z)
            s = dt_tz_aware.strftime(datefmt)
        else:
            # Default format including milliseconds and timezone abbreviation
            # Example: '2023-10-27 10:30:00,123 CDT'
            s = dt_tz_aware.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] # Format with milliseconds
            # Append timezone name/abbreviation if available and not included in %Z/%z via strftime
            # Note: %Z behaviour varies by OS/Python version. Using astimezone handles it correctly.
            # The %Z in strftime should now work as expected with a timezone-aware dt object.
            # Let's rely on %Z in the datefmt if needed. The default format above doesn't use %Z.
            # A common default format is 'YYYY-MM-DD HH:MM:SS,ms'
            # Let's ensure the formatter format string includes %Z or %z if TZ is desired in the output.
            # The base formatter `format` method calls `formatTime`, so the format string from init applies.
            # We just needed to provide a tz-aware datetime object here.

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
    root_logger.setLevel(logging.DEBUG) # Root logger should capture everything

    # Prevent duplicate handlers if configure_logging is somehow called multiple times
    # Clear existing handlers from root logger, which might include default ones
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
    # Logs INFO level and above by default, using configured log level
    console_handler = logging.StreamHandler(sys.stdout)
    # Use the TimezoneAwareFormatter
    # Format string includes timezone abbreviation (%Z)
    console_formatter = TimezoneAwareFormatter(
        "%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z', # Include Timezone Abbreviation
        timezone=tz
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level) # Set console level from config
    root_logger.addHandler(console_handler)
    # Also add a handler for stderr for WARNING/ERROR/CRITICAL
    # This ensures errors are visible even if stdout is redirected
    error_console_handler = logging.StreamHandler(sys.stderr)
    error_console_handler.setFormatter(console_formatter) # Use same formatter
    error_console_handler.setLevel(logging.WARNING) # Only show warnings/errors on stderr
    # Check if a similar stderr handler already exists to avoid duplicates
    stderr_exists = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in root_logger.handlers)
    if not stderr_exists:
        root_logger.addHandler(error_console_handler)
    else:
        # If stderr handler exists, just ensure its level is appropriate
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                 handler.setLevel(min(handler.level, logging.WARNING)) # Ensure it captures at least WARNING


    # --- Main File Handler ---
    # Logs DEBUG level and above to a main log file
    main_log_filename = os.path.join(LOG_DIRECTORY, "xrscalper_bot.log")
    try:
        file_handler = RotatingFileHandler(
            main_log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use the TimezoneAwareFormatter for the file log as well
        file_formatter = TimezoneAwareFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S,%f', # Include milliseconds in file log
            timezone=tz
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Log this error to console before main file logger is fully set up
        # Use a basic print or the console handler directly if available
        print(f"CRITICAL ERROR: Could not set up main file logger {main_log_filename}: {e}", file=sys.stderr)
        # The bot might still run with just console logging


    # --- Optional: Configure levels for chatty libraries ---
    # Example: Suppress DEBUG/INFO messages from ccxt if not needed
    # logging.getLogger('ccxt').setLevel(logging.WARNING)
    # logging.getLogger('urllib3').setLevel(logging.WARNING)
    # logging.getLogger('asyncio').setLevel(logging.WARNING)


    # Log that configuration is complete
    logging.getLogger("xrscalper_bot_init").info(f"Logging configured successfully with level: {logging.getLevelName(log_level)}")


# Note: The original setup_logger function is removed.
# In main.py, after calling configure_logging(CONFIG), you should get loggers
# for specific components or symbols using:
# init_logger = logging.getLogger("xrscalper_bot_init")
# symbol_logger = logging.getLogger("xrscalper_bot_BTC_USDT")
# etc.
# Messages logged by these loggers will propagate up to the root logger
# and be handled by the console and file handlers configured here.
# If symbol-specific *file* logs are required, a dedicated function
# could be added to this module to add a RotatingFileHandler to
# a specific named logger if it doesn't have one, perhaps called
# setup_symbol_file_logger(symbol_name, config). But typically,
# logging to one main file with logger names indicating the source is sufficient.

# Example of a potential function if symbol-specific files ARE needed:
# def setup_symbol_file_logger(symbol_name: str, config: dict):
#     """Sets up a file handler specifically for a symbol logger."""
#     safe_symbol_name = symbol_name.replace('/', '_').replace(':', '-')
#     logger_name = f"xrscalper_bot_{safe_symbol_name}"
#     log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
#     symbol_logger = logging.getLogger(logger_name)
#
#     # Ensure symbol logger's level is low enough to capture desired messages
#     # It might inherit from root, or you might set it explicitly
#     # symbol_logger.setLevel(logging.DEBUG) # Or inherit
#
#     # Check if a file handler for this path already exists to avoid duplicates
#     if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(log_filename) for h in symbol_logger.handlers):
#         try:
#             os.makedirs(LOG_DIRECTORY, exist_ok=True)
#             file_handler = RotatingFileHandler(
#                 log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
#             )
#             tz = get_timezone()
#             file_formatter = TimezoneAwareFormatter(
#                 "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
#                 datefmt='%Y-%m-%d %H:%M:%S,%f', # Include milliseconds
#                 timezone=tz
#             )
#             file_handler.setFormatter(file_formatter)
#             file_handler.setLevel(logging.DEBUG) # Log everything to the symbol file
#             symbol_logger.addHandler(file_handler)
#             symbol_logger.info(f"Symbol-specific file logging enabled for {symbol_name}")
#         except Exception as e:
#             symbol_logger.error(f"Failed to set up symbol file logger for {symbol_name}: {e}", exc_info=True)
#
#     # Important: Ensure propagation is False if you *only* want messages in the symbol file
#     # and not the main log file. If you want them in both, propagation should be True
#     # and the symbol logger level should be high enough to pass messages up.
#     # Given the structure of main.py, keeping propagation True and relying on the
#     # root logger handlers is simpler and messages will appear in both console
#     # and main file, with the logger name identifying the source.
#     # symbol_logger.propagate = False # Decide based on whether you want logs in main file too
