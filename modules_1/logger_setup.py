```python
# File: logger_setup.py
"""
Configures the application's logging system.

Sets up handlers for console (stdout/stderr) and rotating files,
using a custom formatter for timezone-aware timestamps and potential sensitive data masking.
"""

import logging
import os
import sys
from datetime import datetime, timezone as dt_timezone, tzinfo
from logging.handlers import RotatingFileHandler
from typing import Optional, Type, Dict, Any, List

# --- Constants ---
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_FILE_LOG_LEVEL: str = "DEBUG"
DEFAULT_LOG_DIRECTORY_NAME: str = "logs"
DEFAULT_ROOT_LOG_FILE_NAME: str = "application.log"
MAX_LOG_FILE_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT: int = 5

# Default format strings
CONSOLE_LOG_FORMAT: str = "%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s"
CONSOLE_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S %Z"  # Includes timezone name
FILE_LOG_FORMAT: str = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s "
    "[%(name)s:%(lineno)d] %(threadName)s - %(message)s"
)
# datefmt for file logs; TimezoneAwareFormatter handles localization.
# Milliseconds are added via .%(msecs)03d in FILE_LOG_FORMAT.
FILE_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# --- Utility Imports and Fallbacks ---
try:
    # Attempt to import custom utilities
    # Assumes utils.py provides:
    # - LOG_DIRECTORY: Path to the directory where log files should be stored.
    # - SensitiveFormatter: A logging.Formatter subclass that masks sensitive data.
    # - get_timezone(): A function that returns a tzinfo object (e.g., from pytz or zoneinfo).
    from utils import LOG_DIRECTORY, SensitiveFormatter, get_timezone
except ImportError:
    print(
        "WARNING: Failed to import some components from 'utils'. "
        "Using default fallbacks. Ensure 'utils.py' is correctly set up "
        "for full functionality (e.g., sensitive data masking, custom timezone configuration).",
        file=sys.stderr,
    )

    _log_directory_fallback = os.path.join(os.getcwd(), DEFAULT_LOG_DIRECTORY_NAME)
    LOG_DIRECTORY = os.environ.get("APP_LOG_DIRECTORY", _log_directory_fallback)

    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        _default_timezone_instance = ZoneInfo("UTC")
    except ImportError:
        _default_timezone_instance = dt_timezone.utc  # Python < 3.9 fallback (UTC)

    def get_timezone() -> tzinfo:
        """Fallback timezone retriever, defaults to UTC."""
        return _default_timezone_instance

    class SensitiveFormatter(logging.Formatter):
        """
        Basic fallback formatter. Does not perform sensitive data masking.
        This is used if the custom SensitiveFormatter from 'utils' is unavailable.
        It ensures compatibility with different Python versions' Formatter.__init__ signatures.
        """
        def __init__(
            self,
            fmt: Optional[str] = None,
            datefmt: Optional[str] = None,
            style: str = "%",
            validate: bool = True,
            *, # Keyword-only arguments follow
            defaults: Optional[Dict[str, Any]] = None
        ):
            if sys.version_info >= (3, 10):
                super().__init__(fmt, datefmt, style, validate=validate, defaults=defaults)
            elif sys.version_info >= (3, 8):
                super().__init__(fmt, datefmt, style, validate=validate)
            else:
                super().__init__(fmt, datefmt, style)

        def format(self, record: logging.LogRecord) -> str:
            return super().format(record)

    print(
        "WARNING: Using fallback BasicFormatter instead of SensitiveFormatter. "
        "Sensitive data will NOT be masked.",
        file=sys.stderr,
    )


# --- Custom Timezone-Aware Formatter ---
class TimezoneAwareFormatter(SensitiveFormatter):
    """
    A logging formatter that ensures timestamps are timezone-aware.
    It uses a specified timezone object for localization and inherits
    from SensitiveFormatter for potential sensitive data masking.
    """
    _default_time_only_format: str = "%Y-%m-%d %H:%M:%S"
    _default_full_format_template: str = "%s,%03d %s"  # time_str, msecs, tz_str

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        timezone_obj: Optional[tzinfo] = None,
        validate: bool = True,
        defaults: Optional[Dict[str, Any]] = None
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate, defaults=defaults)

        resolved_timezone: Optional[tzinfo] = None
        try:
            if timezone_obj is not None:
                if not isinstance(timezone_obj, tzinfo):
                    raise TypeError(
                        f"timezone_obj must be a datetime.tzinfo instance, got {type(timezone_obj)}"
                    )
                resolved_timezone = timezone_obj
            else:
                resolved_timezone = get_timezone()
                if not isinstance(resolved_timezone, tzinfo):
                    print(
                        f"WARNING: get_timezone() returned invalid type: {type(resolved_timezone)}. "
                        "Defaulting to UTC.",
                        file=sys.stderr,
                    )
                    resolved_timezone = dt_timezone.utc
            self.timezone = resolved_timezone
        except Exception as e:
            print(
                f"WARNING: Error initializing timezone for formatter: {e}. Defaulting to UTC.",
                file=sys.stderr,
            )
            self.timezone = dt_timezone.utc


    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """
        Formats the log record's creation time using the configured timezone.
        """
        utc_dt = datetime.fromtimestamp(record.created, tz=dt_timezone.utc)
        local_dt = utc_dt.astimezone(self.timezone)
        effective_datefmt = datefmt or self.datefmt

        if effective_datefmt:
            s = local_dt.strftime(effective_datefmt)
        else:
            time_str = local_dt.strftime(self._default_time_only_format)
            ms = int(local_dt.microsecond / 1000)
            
            tz_name_and_offset = local_dt.strftime("%Z%z")
            if not tz_name_and_offset: # Fallback if strftime("%Z%z") is empty
                if self.timezone == dt_timezone.utc:
                    tz_str = "UTC+0000"
                else:
                    offset = local_dt.utcoffset()
                    if offset is not None:
                        total_seconds = offset.total_seconds()
                        hours = int(total_seconds // 3600)
                        minutes = int((total_seconds % 3600) // 60)
                        tz_str = f"UTC{hours:+03d}:{minutes:02d}"
                    else: 
                        tz_str = "UNKNOWN_TZ" # Should be rare for an aware object
            else:
                tz_str = tz_name_and_offset
            s = self._default_full_format_template % (time_str, ms, tz_str)
        return s


# --- Global Logging Configuration State ---
_logging_configured: bool = False
_application_timezone: Optional[tzinfo] = None


# --- Helper for Handler Creation ---
def _create_logging_handler(
    handler_class: Type[logging.Handler],
    level: int,
    formatter: logging.Formatter,
    handler_args: Optional[List[Any]] = None,
    handler_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[logging.Handler]:
    """
    Creates, configures, and returns a logging handler.
    Returns None if handler creation fails, printing an error to stderr.
    """
    handler_args = handler_args or []
    handler_kwargs = handler_kwargs or {}

    try:
        handler = handler_class(*handler_args, **handler_kwargs)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        return handler
    except Exception as e:
        print(
            f"CRITICAL: Failed to create or configure handler {handler_class.__name__}: {e}",
            file=sys.stderr
        )
        return None


# --- Main Configuration Function ---
def configure_logging(app_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configures the root logger based on the provided application configuration.
    """
    global _logging_configured, _application_timezone
    if _logging_configured:
        logging.getLogger(__name__).warning("Logging system already configured. Skipping re-configuration.")
        return

    config = app_config or {}

    if _application_timezone is None:
        try:
            _application_timezone = get_timezone()
            if not isinstance(_application_timezone, tzinfo):
                raise TypeError(
                    f"get_timezone() must return a tzinfo instance, got {type(_application_timezone)}"
                )
        except Exception as e:
            print(
                f"CRITICAL: Failed to get application timezone: {e}. Defaulting to UTC.",
                file=sys.stderr,
            )
            _application_timezone = dt_timezone.utc

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        print("INFO: Clearing existing handlers from root logger.", file=sys.stderr)
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                print(f"WARNING: Error removing existing handler {handler}: {e}", file=sys.stderr)

    console_log_level_str = str(config.get("log_level", DEFAULT_LOG_LEVEL)).upper()
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)

    # Stdout Handler
    stdout_formatter = TimezoneAwareFormatter(
        fmt=CONSOLE_LOG_FORMAT, datefmt=CONSOLE_DATE_FORMAT, timezone_obj=_application_timezone
    )
    stdout_handler = _create_logging_handler(
        logging.StreamHandler, console_log_level, stdout_formatter, handler_args=[sys.stdout]
    )
    if stdout_handler:
        class MaxLevelFilter(logging.Filter):
            def __init__(self, max_level: int):
                super().__init__()
                self.max_level = max_level
            def filter(self, record: logging.LogRecord) -> bool:
                return record.levelno < self.max_level
        
        # Only add filter if stdout isn't already set to WARNING or higher
        if console_log_level < logging.WARNING:
            stdout_handler.addFilter(MaxLevelFilter(logging.WARNING))
        root_logger.addHandler(stdout_handler)

    # Stderr Handler
    stderr_formatter = TimezoneAwareFormatter(
        fmt=CONSOLE_LOG_FORMAT, datefmt=CONSOLE_DATE_FORMAT, timezone_obj=_application_timezone
    )
    stderr_handler = _create_logging_handler(
        logging.StreamHandler, logging.WARNING, stderr_formatter, handler_args=[sys.stderr]
    )
    if stderr_handler:
        root_logger.addHandler(stderr_handler)

    # Rotating File Handler
    file_logging_enabled = False
    actual_file_log_level = logging.NOTSET
    log_filepath = "" # Initialize to prevent usage if not set
    log_directory_path = LOG_DIRECTORY

    try:
        os.makedirs(log_directory_path, exist_ok=True)
        log_dir_exists_and_writable = True
    except OSError as e:
        print(
            f"CRITICAL: Cannot create log directory '{log_directory_path}': {e}. File logging disabled.",
            file=sys.stderr
        )
        log_dir_exists_and_writable = False

    if log_dir_exists_and_writable:
        file_log_level_str = str(config.get("log_file_level", DEFAULT_FILE_LOG_LEVEL)).upper()
        actual_file_log_level = getattr(logging, file_log_level_str, logging.DEBUG)
        log_file_name = str(config.get("log_file_name", DEFAULT_ROOT_LOG_FILE_NAME))
        log_filepath = os.path.join(log_directory_path, log_file_name)

        file_formatter = TimezoneAwareFormatter(
            fmt=FILE_LOG_FORMAT, datefmt=FILE_DATE_FORMAT, timezone_obj=_application_timezone
        )
        file_handler_kwargs = {
            "maxBytes": MAX_LOG_FILE_BYTES, "backupCount": LOG_FILE_BACKUP_COUNT,
            "encoding": "utf-8", "delay": True,
        }
        file_handler = _create_logging_handler(
            RotatingFileHandler, actual_file_log_level, file_formatter,
            handler_args=[log_filepath], handler_kwargs=file_handler_kwargs
        )
        if file_handler:
            root_logger.addHandler(file_handler)
            file_logging_enabled = True
        else:
            actual_file_log_level = logging.NOTSET

    # Library Log Levels
    default_library_levels = {"urllib3": "WARNING", "httpx": "WARNING", "asyncio": "WARNING"}
    library_log_levels_config = config.get("library_log_levels", {})
    final_library_levels = {**default_library_levels, **library_log_levels_config}

    for lib_name, level_str_val in final_library_levels.items():
        level_str = str(level_str_val).upper()
        lib_level = getattr(logging, level_str, None)
        if lib_level is not None:
            logging.getLogger(lib_name).setLevel(lib_level)
            logging.info(f"Set log level for library '{lib_name}' to {level_str}.")
        else:
            logging.warning(f"Invalid log level '{level_str}' for library '{lib_name}'. Ignoring.")

    _logging_configured = True
    init_logger = logging.getLogger("App.Initialization")
    init_logger.info(
        f"Logging configured. Console: {logging.getLevelName(console_log_level)} (stdout <WARNING), "
        f"{logging.getLevelName(logging.WARNING)}+ (stderr)."
    )
    if file_logging_enabled:
        init_logger.info(
            f"File logging: {logging.getLevelName(actual_file_log_level)} to '{log_filepath}'."
        )
    else:
        init_logger.warning(
            f"File logging disabled. Log directory: '{log_directory_path}' (problem or handler error)."
        )
```