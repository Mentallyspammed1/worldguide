# File: logger_setup.py
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

# Import constants needed for redacting and directory
import constants
# Import ZoneInfo type hint/class
from utils import ZoneInfo

# Fetch API keys from environment for redacting
# Ensure dotenv is loaded in the main script before this is called
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Ensure keys exist before attempting replacement
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def setup_logger(
    logger_name_suffix: str,
    log_directory: str = constants.LOG_DIRECTORY,
    timezone: Any = None # Pass the initialized ZoneInfo object
) -> logging.Logger:
    """Sets up a logger for the given suffix with file and console handlers."""
    # Use a base name and append suffix
    base_logger_name = "xrscalper_bot"
    logger_name = f"{base_logger_name}_{logger_name_suffix}"
    log_filename = os.path.join(log_directory, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if logger already exists and has handlers
    if logger.hasHandlers():
        # If handlers exist, assume it's already configured and return it
        return logger

    logger.setLevel(logging.DEBUG) # Set logger level to DEBUG to capture all messages

    # File Handler (writes DEBUG level and above)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_filename}: {e}")

    # Stream Handler (Console - writes INFO level and above by default)
    stream_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
    stream_formatter = SensitiveFormatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    # Set converter to use the configured TIMEZONE for log record times.
    if timezone:
        logging.Formatter.converter = lambda *args: datetime.now(timezone).timetuple()
    else: # Fallback if timezone object wasn't passed
        logging.Formatter.converter = lambda *args: datetime.now().astimezone().timetuple() # Use system local time

    stream_handler.setFormatter(stream_formatter)
    console_log_level = logging.INFO # Change to DEBUG for more verbose console output
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent messages from propagating to the root logger
    return logger

```

```python
