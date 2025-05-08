"""Enhanced Trading Bot Script.

This script implements a trading bot using the CCXT library to interact with the Bybit exchange (V5 API).
It utilizes pandas and pandas_ta for technical analysis and incorporates features like:
- Customizable multi-indicator strategy with weighted signals.
- Dynamic position sizing based on risk percentage and ATR.
- ATR-based Stop Loss and Take Profit calculation.
- Optional Trailing Stop Loss via Bybit's native feature.
- Optional Break-Even Stop Loss adjustment.
- Optional MA Cross exit condition.
- Robust error handling and API call retries.
- Detailed logging with timezone awareness and sensitive data redaction.
- Configuration via JSON file (config.json).
- State persistence via JSON file (bot_state.json).
- Support for Bybit Unified Trading Accounts (UTA) and non-UTA accounts.
- Command-line arguments for debugging and configuration overrides.
"""

import argparse
import json
import logging
import math
import os
import re  # Added for parsing error messages
import sys
import time
import traceback
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any

# --- Timezone Handling ---
# Use standard library zoneinfo (Python 3.9+) if available, fallback to pytz
try:
    from zoneinfo import ZoneInfo  # Preferred (Python 3.9+)
    _has_zoneinfo = True
except ImportError:
    _has_zoneinfo = False
    try:
        # Fallback for Python < 3.9 (requires pip install pytz)
        from pytz import timezone as ZoneInfo
        _has_pytz = True
    except ImportError:
        _has_pytz = False
        # Critical dependency, exit if neither is available
        sys.exit(1)

# --- Core Trading Libraries ---
try:
    import ccxt
except ImportError:
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    sys.exit(1)
try:
    import pandas_ta as ta
except ImportError:
    sys.exit(1)

# --- Optional Enhancements ---
# Colorama for colored console output
try:
    from colorama import Fore, Style, init
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Define dummy color variables if colorama is missing
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""  # Return empty string for any attribute
    Fore = DummyColor()
    Style = DummyColor()
    # Dummy init function
    def init(*args: Any, **kwargs: Any) -> None: pass

# Dotenv for loading environment variables from .env file
try:
    from dotenv import load_dotenv
except ImportError:
    # Define dummy function if dotenv is missing
    def load_dotenv(*args: Any, **kwargs: Any) -> None: pass

# --- Initialization ---
# Set precision for Decimal arithmetic operations.
# Note: This primarily affects calculations like Decimal * Decimal.
# Storing and retrieving Decimals generally maintains their inherent precision.
try:
    getcontext().prec = 36  # Sufficient precision for most financial calculations
except Exception:
    pass

# Initialize colorama (or dummy init)
# autoreset=True ensures styles are reset after each print statement
init(autoreset=True)

# Load environment variables from .env file (if python-dotenv is installed)
load_dotenv()

# --- Constants ---

# Bot Identity
BOT_VERSION = "1.1.0"  # Updated version

# Neon Color Scheme (requires colorama)
NEON_GREEN = Fore.LIGHTGREEN_EX if COLORAMA_AVAILABLE else ""
NEON_BLUE = Fore.LIGHTBLUE_EX if COLORAMA_AVAILABLE else ""  # Changed for better readability
NEON_PURPLE = Fore.MAGENTA if COLORAMA_AVAILABLE else ""
NEON_YELLOW = Fore.YELLOW if COLORAMA_AVAILABLE else ""
NEON_RED = Fore.LIGHTRED_EX if COLORAMA_AVAILABLE else ""
NEON_CYAN = Fore.CYAN if COLORAMA_AVAILABLE else ""
RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ""

# --- Environment Variable Loading and Validation ---
# Load API Keys from environment variables
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

# Critical check: Ensure API keys are set
if not API_KEY or not API_SECRET:
    sys.exit(1)

# --- Configuration File and Paths ---
CONFIG_FILE = "config.json"     # Default configuration file name
LOG_DIRECTORY = "bot_logs"      # Directory to store log files
STATE_FILE = "bot_state.json"   # File to store persistent bot state

# Ensure log directory exists, create if it doesn't
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError:
    sys.exit(1)

# --- Timezone Configuration ---
_DEFAULT_TIMEZONE = "America/Chicago"
_FALLBACK_TIMEZONE = "UTC"
try:
    # Attempt to load timezone from environment or use default
    TZ_NAME = os.getenv("BOT_TIMEZONE", _DEFAULT_TIMEZONE)
    TIMEZONE = ZoneInfo(TZ_NAME)
except Exception:
    # Default to UTC using the available library
    try:
        TIMEZONE = ZoneInfo(_FALLBACK_TIMEZONE)
        TZ_NAME = _FALLBACK_TIMEZONE
    except Exception:
        # This should not happen due to initial checks, but as a final safeguard
        sys.exit(1)

# --- API Interaction Constants ---
MAX_API_RETRIES = 4             # Max retries for recoverable API errors (e.g., rate limits, network issues)
RETRY_DELAY_SECONDS = 5         # Initial delay in seconds before retrying API calls (uses exponential backoff)
RATE_LIMIT_BUFFER_SECONDS = 0.5  # Extra buffer added to wait time suggested by rate limit errors
MARKET_RELOAD_INTERVAL_SECONDS = 3600  # How often to reload exchange market data (1 hour)
POSITION_CONFIRM_DELAY = 10     # Seconds to wait after placing entry order before confirming position/price via API
MIN_TICKS_AWAY_FOR_SLTP = 3     # Minimum number of price ticks SL/TP should be away from entry price after quantization

# --- Bot Logic Constants ---
# Supported intervals for OHLCV data (ensure config uses one of these string representations)
VALID_INTERVALS: list[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
# Map bot intervals (strings) to ccxt's expected timeframe format (strings)
CCXT_INTERVAL_MAP: dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}

# Default Indicator/Strategy Parameters (can be overridden by config.json)
# These ensure the bot can run even with a minimal config file.
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14
DEFAULT_STOCH_WINDOW = 14  # Inner RSI period for StochRSI
DEFAULT_K_WINDOW = 3
DEFAULT_D_WINDOW = 3
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2
# Standard Fibonacci levels used for calculation
FIB_LEVELS: list[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# Default loop delay (can be overridden by config)
DEFAULT_LOOP_DELAY_SECONDS = 15

# --- Global Variables ---
loggers: dict[str, logging.Logger] = {}  # Cache for logger instances to avoid duplicate handlers
console_log_level: int = logging.INFO   # Default console log level (can be changed by --debug arg)
QUOTE_CURRENCY: str = "USDT"            # Default quote currency (updated from config)
LOOP_DELAY_SECONDS: int = DEFAULT_LOOP_DELAY_SECONDS  # Actual loop delay (updated from config)
IS_UNIFIED_ACCOUNT: bool = False        # Flag indicating Bybit account type (detected on init)


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive API keys/secrets."""
    REDACTED_STR: str = "***REDACTED***"
    # Store sensitive keys here for redaction. Ensure API_KEY/SECRET are checked for None.
    SENSITIVE_KEYS: list[str] = [
        key for key in [API_KEY, API_SECRET] if key is not None and len(key) > 4
    ]

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive info."""
        # Format the original record first
        formatted = super().format(record)
        # Redact sensitive keys if they exist and are long enough
        for key in self.SENSITIVE_KEYS:
            formatted = formatted.replace(key, self.REDACTED_STR)
        return formatted


class LocalTimeFormatter(SensitiveFormatter):
    """Custom formatter that uses the configured local timezone for console output timestamps.
    Inherits redaction logic from SensitiveFormatter.
    """
    converter = lambda _, timestamp: datetime.fromtimestamp(timestamp, tz=TIMEZONE).timetuple()

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Formats the record's creation time using the local timezone and specified format."""
        dt = datetime.fromtimestamp(record.created, tz=TIMEZONE)
        if datefmt:
            # Use specified date format
            s = dt.strftime(datefmt)
        else:
            # Default format with milliseconds
            s = dt.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{s},{int(record.msecs):03d}"  # Append milliseconds
        return s


def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up and configures a logger instance.

    - Uses rotating file handler with UTC timestamps for persistent logs.
    - Uses stream handler (console) with local timestamps and color formatting.
    - Redacts sensitive API keys/secrets in log messages.
    - Caches logger instances to avoid duplicate handlers.
    - Creates filesystem-safe log file names.

    Args:
        name: The base name for the logger (e.g., 'main', 'BTC/USDT').
        is_symbol_logger: If True, formats the logger name to be filesystem-safe
                          (replaces '/' and ':').

    Returns:
        The configured logging.Logger instance.
    """
    global console_log_level  # Use the global console level setting

    # Create a safe name for file system and logger registry
    safe_name = name.replace('/', '_').replace(':', '-') if is_symbol_logger else name
    logger_instance_name = f"livebot_{safe_name}"

    # Check cache first
    if logger_instance_name in loggers:
        logger = loggers[logger_instance_name]
        # Update console handler level if it changed (e.g., via --debug argument)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level != console_log_level:
                handler.setLevel(console_log_level)
                # logger.debug(f"Updated console log level for '{logger_instance_name}' to {logging.getLevelName(console_log_level)}")
        return logger

    # Create logger instance
    logger = logging.getLogger(logger_instance_name)
    # Set logger level to DEBUG to capture all messages; handlers control output level
    logger.setLevel(logging.DEBUG)

    # --- File Handler (UTC Timestamps, Sensitive Redaction) ---
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    try:
        # Rotate log file when it reaches 10MB, keep up to 5 backup files
        file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        # Use UTC time for file logs for consistency across systems/timezones
        file_formatter = SensitiveFormatter(
            # Format: UTC Timestamp - Level - [LoggerName:LineNo] - Message
            fmt="%(asctime)s.%(msecs)03d UTC - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Explicitly set UTC converter for file handler
        file_formatter.converter = time.gmtime
        file_handler.setFormatter(file_formatter)
        # File handler logs everything (DEBUG level and above)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        # Log error to console if file handler setup fails
        pass

    # --- Stream Handler (Local Timezone Timestamps, Colors, Sensitive Redaction) ---
    try:
        stream_handler = logging.StreamHandler(sys.stdout)
        # Include timezone name and use colors in console output format
        # Format: LocalTimestamp [TZ] - Level - [LoggerName] - Message
        console_fmt = (
            f"{NEON_BLUE}%(asctime)s{RESET} [{TZ_NAME}] - "
            f"{NEON_YELLOW}%(levelname)-8s{RESET} - "
            f"{NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
        )
        # Use LocalTimeFormatter for timezone-aware timestamps and redaction
        # datefmt includes milliseconds (%f) sliced to 3 digits
        stream_formatter = LocalTimeFormatter(console_fmt, datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3])
        stream_handler.setFormatter(stream_formatter)
        # Set console level based on global setting (e.g., INFO default, DEBUG if --debug)
        stream_handler.setLevel(console_log_level)
        logger.addHandler(stream_handler)
    except Exception:
        # Log error to console if stream handler setup fails
        pass

    # Prevent log messages from propagating to the root logger (avoids duplicate output)
    logger.propagate = False
    # Cache the logger instance
    loggers[logger_instance_name] = logger
    logger.info(f"Logger '{logger_instance_name}' initialized. Log file: '{os.path.basename(log_filename)}', Console Level: {logging.getLevelName(console_log_level)}")
    return logger


def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance using the setup_logger function."""
    return setup_logger(name, is_symbol_logger)


# --- Configuration Management ---
def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from the default config are present in the loaded config.
    Adds missing keys with default values and logs warnings. Handles nested dictionaries.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.
        parent_key: String representing the parent key path for nested logging (e.g., "indicators.").

    Returns:
        A tuple containing:
        - The updated configuration dictionary.
        - A boolean indicating if any keys were added or if type mismatches occurred.
    """
    updated_config = config.copy()
    keys_added_or_type_mismatch = False
    current_path = f"{parent_key}." if parent_key else ""

    for key, default_value in default_config.items():
        full_key_path = f"{current_path}{key}"
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            keys_added_or_type_mismatch = True
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_keys_added:
                updated_config[key] = nested_config
                keys_added_or_type_mismatch = True
        elif updated_config.get(key) is not None and type(default_value) != type(updated_config.get(key)):
            # Type mismatch check (allow specific safe promotions/conversions)
            loaded_value = updated_config.get(key)
            # Allow int -> float/Decimal promotion
            is_promoting_num = (isinstance(default_value, (float, Decimal)) and isinstance(loaded_value, int))
            # Allow common string -> bool conversion (handled later in validation)
            is_str_bool_candidate = isinstance(default_value, bool) and isinstance(loaded_value, str)

            if not is_promoting_num and not is_str_bool_candidate:
                pass
                # Note: We keep the user's value despite the type mismatch for non-critical types,
                # but warn them. Validation (_validate_config_values) handles critical conversions/errors.
                # keys_added_or_type_mismatch = True # Optionally flag type mismatches as needing review/update
    return updated_config, keys_added_or_type_mismatch


def _validate_config_values(config: dict[str, Any], logger: logging.Logger) -> bool:
    """Validates critical configuration values for type, range, and format.
    Modifies the input config dictionary *in place* for necessary type conversions
    (e.g., string "true" to bool, string "10" to int).

    Args:
        config: The configuration dictionary to validate (modified in place).
        logger: The logger instance to use for reporting errors.

    Returns:
        True if the configuration is valid after potential modifications, False otherwise.
    """
    is_valid = True
    lg = logger

    # 1. Validate Interval
    interval = config.get("interval")
    if interval not in CCXT_INTERVAL_MAP:
        lg.error(f"Config Error: Invalid 'interval' value '{interval}'. Must be one of {VALID_INTERVALS}")
        is_valid = False
    # Ensure config interval matches CCXT map key (string)
    config["interval"] = str(interval)  # Convert just in case user provided integer

    # 2. Validate Numeric Types and Ranges
    # Format: key: (expected_type, min_value, max_value, required_integer)
    # expected_type is the final target type (int or float)
    # required_integer flag ensures the value is converted to int if possible/necessary
    numeric_params: dict[str, tuple[type, int | float, int | float, bool]] = {
        "loop_delay": (int, 5, 3600, True),         # Min 5 sec delay recommended
        "risk_per_trade": (float, 0.0001, 0.5, False),  # Risk 0.01% to 50% of balance
        "leverage": (int, 1, 125, True),            # Practical leverage limits (exchange may vary)
        "max_concurrent_positions_total": (int, 1, 100, True),
        "atr_period": (int, 2, 500, True),
        "ema_short_period": (int, 2, 500, True),
        "ema_long_period": (int, 3, 1000, True),  # Ensure long > short is checked separately
        "rsi_period": (int, 2, 500, True),
        "bollinger_bands_period": (int, 5, 500, True),
        "bollinger_bands_std_dev": (float, 0.1, 5.0, False),
        "cci_window": (int, 5, 500, True),
        "williams_r_window": (int, 2, 500, True),
        "mfi_window": (int, 5, 500, True),
        "stoch_rsi_window": (int, 5, 500, True),
        "stoch_rsi_rsi_window": (int, 5, 500, True),  # Inner RSI window for StochRSI
        "stoch_rsi_k": (int, 1, 100, True),
        "stoch_rsi_d": (int, 1, 100, True),
        "psar_af": (float, 0.001, 0.5, False),
        "psar_max_af": (float, 0.01, 1.0, False),
        "sma_10_window": (int, 2, 500, True),
        "momentum_period": (int, 2, 500, True),
        "volume_ma_period": (int, 5, 500, True),
        "fibonacci_window": (int, 10, 1000, True),
        "orderbook_limit": (int, 1, 200, True),  # Bybit V5 limit might be 50 or 200 depending on type
        "signal_score_threshold": (float, 0.1, 10.0, False),
        "stoch_rsi_oversold_threshold": (float, 0.0, 50.0, False),
        "stoch_rsi_overbought_threshold": (float, 50.0, 100.0, False),
        "volume_confirmation_multiplier": (float, 0.1, 10.0, False),
        "scalping_signal_threshold": (float, 0.1, 10.0, False),
        "stop_loss_multiple": (float, 0.1, 10.0, False),   # Multiplier of ATR
        "take_profit_multiple": (float, 0.1, 20.0, False),  # Multiplier of ATR
        "trailing_stop_callback_rate": (float, 0.0001, 0.5, False),  # 0.01% to 50% (as distance from price)
        "trailing_stop_activation_percentage": (float, 0.0, 0.5, False),  # 0% to 50% (profit needed to activate)
        "break_even_trigger_atr_multiple": (float, 0.1, 10.0, False),  # Multiplier of ATR
        "break_even_offset_ticks": (int, 0, 100, True),   # Number of ticks above/below entry for BE SL
    }

    for key, (target_type, min_val, max_val, require_int) in numeric_params.items():
        value = config.get(key)
        if value is None: continue  # Skip if optional or handled by ensure_keys

        original_value_repr = repr(value)  # For logging
        try:
            # Attempt conversion to float first for universal range checking
            num_value_float = float(value)

            # Check range
            if not (min_val <= num_value_float <= max_val):
                lg.error(f"Config Error: '{key}' value {num_value_float} (from {original_value_repr}) is outside the recommended range ({min_val} - {max_val}).")
                is_valid = False
                continue  # Skip further processing for this key

            # Convert to target type (int or float)
            if target_type is int:
                # Check if the float value is acceptably close to an integer
                if require_int and not math.isclose(num_value_float, round(num_value_float), rel_tol=1e-9):
                     lg.warning(f"Config Warning: '{key}' requires an integer, but found {num_value_float} (from {original_value_repr}). Truncating to {int(num_value_float)}.")
                config[key] = int(round(num_value_float))  # Round before casting to int
            else:  # Target type is float
                 config[key] = num_value_float  # Store as float

        except (ValueError, TypeError):
            lg.error(f"Config Error: '{key}' value {original_value_repr} could not be converted to a number.")
            is_valid = False

    # Specific check: EMA Long > EMA Short (after potential conversion)
    ema_long = config.get("ema_long_period")
    ema_short = config.get("ema_short_period")
    if isinstance(ema_long, int) and isinstance(ema_short, int) and ema_long <= ema_short:
        lg.error(f"Config Error: 'ema_long_period' ({ema_long}) must be greater than 'ema_short_period' ({ema_short}).")
        is_valid = False

    # 3. Validate Symbols List
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
         lg.error("Config Error: 'symbols' must be a non-empty list.")
         is_valid = False
    elif not all(isinstance(s, str) and '/' in s for s in symbols):  # Basic format check
         lg.error(f"Config Error: 'symbols' list contains invalid formats. Expected 'BASE/QUOTE' or 'BASE/QUOTE:SETTLE'. Found: {symbols}")
         is_valid = False

    # 4. Validate Active Weight Set exists and contains valid weights
    active_set_name = config.get("active_weight_set")
    weight_sets = config.get("weight_sets")
    if not isinstance(weight_sets, dict) or not isinstance(active_set_name, str) or active_set_name not in weight_sets:
        lg.error(f"Config Error: 'active_weight_set' ('{active_set_name}') not found in 'weight_sets'. Available: {list(weight_sets.keys() if isinstance(weight_sets, dict) else [])}")
        is_valid = False
    elif not isinstance(weight_sets[active_set_name], dict):
        lg.error(f"Config Error: Active weight set '{active_set_name}' must be a dictionary of weights.")
        is_valid = False
    else:
        # Validate weights within the active set (should be numeric)
        active_set_weights = weight_sets[active_set_name]
        for indi_key, weight_val in active_set_weights.items():
            try:
                # Try converting to float, store back as float in config for consistency
                active_set_weights[indi_key] = float(weight_val)
            except (ValueError, TypeError):
                 lg.error(f"Config Error: Invalid weight value '{weight_val}' for indicator '{indi_key}' in weight set '{active_set_name}'. Must be numeric.")
                 is_valid = False

    # 5. Validate Boolean types (ensure they are bool, converting common strings)
    bool_params = [
        "enable_trading", "use_sandbox", "enable_ma_cross_exit", "enable_trailing_stop",
        "enable_break_even", "break_even_force_fixed_sl"
    ]
    for key in bool_params:
        if key in config:  # Check if the key exists
            value = config[key]
            if isinstance(value, bool): continue  # Already boolean, skip

            # Try to convert common string representations
            if isinstance(value, str):
                val_str = value.lower().strip()
                if val_str in ['true', 'yes', '1', 'on']:
                    config[key] = True
                    lg.debug(f"Config: Converted '{key}' value '{value}' to True.")
                elif val_str in ['false', 'no', '0', 'off']:
                    config[key] = False
                    lg.debug(f"Config: Converted '{key}' value '{value}' to False.")
                else:
                    lg.error(f"Config Error: '{key}' value '{value}' must be a boolean (true/false/1/0 or actual boolean type).")
                    is_valid = False
            else:  # Not a bool or a string, invalid type
                lg.error(f"Config Error: '{key}' value '{value}' (type: {type(value).__name__}) must be a boolean.")
                is_valid = False

    # 6. Validate indicator enable flags (must be dict of booleans)
    indicators_config = config.get("indicators")
    if indicators_config is None:
        lg.error("Config Error: 'indicators' dictionary is missing.")
        is_valid = False
    elif not isinstance(indicators_config, dict):
         lg.error("Config Error: 'indicators' must be a dictionary.")
         is_valid = False
    else:
        for indi_key, indi_val in indicators_config.items():
            if isinstance(indi_val, bool): continue  # Correct type

            if isinstance(indi_val, str):
                val_str = indi_val.lower().strip()
                if val_str in ['true', 'yes', '1', 'on']:
                    config["indicators"][indi_key] = True  # Update in place
                    lg.debug(f"Config: Converted 'indicators.{indi_key}' value '{indi_val}' to True.")
                elif val_str in ['false', 'no', '0', 'off']:
                    config["indicators"][indi_key] = False  # Update in place
                    lg.debug(f"Config: Converted 'indicators.{indi_key}' value '{indi_val}' to False.")
                else:
                    lg.error(f"Config Error: Indicator enable flag 'indicators.{indi_key}' value '{indi_val}' must be a boolean (true/false/1/0).")
                    is_valid = False
            else:  # Not bool or string
                lg.error(f"Config Error: Indicator enable flag 'indicators.{indi_key}' value '{indi_val}' (type: {type(indi_val).__name__}) must be a boolean.")
                is_valid = False

    # 7. Validate Position Mode
    position_mode = config.get("position_mode", "One-Way")
    valid_modes = ["One-Way", "Hedge"]
    if position_mode not in valid_modes:
        lg.error(f"Config Error: Invalid 'position_mode' value '{position_mode}'. Must be one of {valid_modes}.")
        is_valid = False
    elif position_mode == "Hedge":
        lg.warning(f"{NEON_YELLOW}Config Warning: 'position_mode' is set to 'Hedge'. Ensure your Bybit account and API key are configured for Hedge Mode. Bot logic requires careful handling of positionIdx (1 for Buy, 2 for Sell) in Hedge Mode.{RESET}")

    return is_valid


def load_config(filepath: str, logger: logging.Logger) -> dict[str, Any] | None:
    """Loads, validates, and potentially updates the configuration from a JSON file.

    - Creates a default config file if it doesn't exist.
    - Ensures all necessary keys are present, adding defaults for missing ones (recursively).
    - Validates critical configuration values (type, range, format) and converts types where needed.
    - Saves the updated config (with defaults added or types converted) back to the file if changes were made.

    Args:
        filepath: The path to the configuration JSON file.
        logger: The logger instance for reporting.

    Returns:
        The loaded and validated configuration dictionary, or None if loading/validation fails critically.
    """
    lg = logger
    # --- Default Configuration Structure ---
    # Serves as the template and source of default values.
    default_config: dict[str, Any] = {
        "symbols": ["BTC/USDT:USDT"],  # List of symbols (format: BASE/QUOTE:SETTLE or BASE/QUOTE)
        "interval": "5",              # Kline interval (string from VALID_INTERVALS)
        "loop_delay": DEFAULT_LOOP_DELAY_SECONDS,  # Seconds between bot cycles (integer)
        "quote_currency": "USDT",     # Primary currency for balance checks and calculations
        "enable_trading": False,      # Master switch for placing actual trades (boolean)
        "use_sandbox": True,          # Use Bybit's testnet environment (boolean)
        "risk_per_trade": 0.01,       # Fraction of balance to risk per trade (float, e.g., 0.01 = 1%)
        "leverage": 10,               # Leverage for futures trades (integer)
        "max_concurrent_positions_total": 1,  # Max open positions across all symbols (integer)
        "position_mode": "One-Way",   # "One-Way" or "Hedge" (string)
        # --- Indicator Periods & Parameters ---
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW,
        "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW,
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,  # Inner RSI period for StochRSI
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,         # Parabolic SAR acceleration factor step (float)
        "psar_max_af": DEFAULT_PSAR_MAX_AF,  # Parabolic SAR max acceleration factor (float)
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW,  # Window for calculating Fib levels (integer)
        # --- Strategy & Order Settings ---
        "orderbook_limit": 25,           # Number of levels for order book analysis (integer)
        "signal_score_threshold": 1.5,   # Threshold for combined weighted score (float)
        "stoch_rsi_oversold_threshold": 25.0,  # Float
        "stoch_rsi_overbought_threshold": 75.0,  # Float
        "volume_confirmation_multiplier": 1.5,  # Float
        "scalping_signal_threshold": 2.5,  # Optional higher threshold for 'scalping' set (float)
        "stop_loss_multiple": 1.8,       # ATR multiple for SL distance (float)
        "take_profit_multiple": 0.7,     # ATR multiple for TP distance (float)
        # --- Exit & Position Management ---
        "enable_ma_cross_exit": True,    # Close position on adverse EMA cross (boolean)
        "enable_trailing_stop": True,    # Enable Bybit's native TSL feature (boolean)
        "trailing_stop_callback_rate": 0.005,  # TSL distance as fraction of price (float, e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003,  # Profit % required to activate TSL (float, e.g., 0.003 = 0.3%)
        "enable_break_even": True,       # Enable moving SL to break-even (boolean)
        "break_even_trigger_atr_multiple": 1.0,  # ATR multiple profit needed to trigger BE (float)
        "break_even_offset_ticks": 2,    # Ticks above/below entry for BE SL (integer)
        "break_even_force_fixed_sl": True,  # If true, BE replaces TSL; if false, BE sets SL but TSL might remain active (boolean)
        # --- Indicator Enable Flags (Dictionary of booleans) ---
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,  # Requires fetching order book data
        },
        # --- Weight Sets for Signal Generation (Dictionary of dictionaries) ---
        "weight_sets": {
            "scalping": {  # Example: Focus on faster indicators
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2,
                "orderbook": 0.15,
            },
            "default": {  # Example: More balanced
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2,
                "orderbook": 0.1,
            }
        },
        "active_weight_set": "default"  # Which weight set to use (string)
    }

    loaded_config = None
    config_changed_needs_save = False

    # --- Load or Create Config File ---
    if not os.path.exists(filepath):
        # Config file doesn't exist, create it with defaults
        lg.warning(f"Config file not found at '{filepath}'. Creating default config...")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Use default json serializer, ensure keys are sorted for readability
                json.dump(default_config, f, indent=4, sort_keys=True)
            lg.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            # Use the defaults as the loaded config
            loaded_config = default_config
            # No need to save again immediately as we just created it with defaults
        except OSError as e:
            lg.error(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using in-memory defaults.{RESET}")
            # Fallback to using in-memory defaults if creation fails
            loaded_config = default_config.copy()
    else:
        # Config file exists, load it
        try:
            with open(filepath, encoding="utf-8") as f:
                config_from_file = json.load(f)
            # Store the initially loaded config for comparison later
            original_loaded_config = config_from_file.copy()  # Deep copy might be needed if nested dicts are modified directly

            # Ensure all keys from default_config are present, add if missing
            config_with_defaults, keys_were_added = _ensure_config_keys(config_from_file, default_config)
            loaded_config = config_with_defaults  # Use the potentially updated config
            if keys_were_added:
                config_changed_needs_save = True

        except (FileNotFoundError, json.JSONDecodeError) as e:
            lg.error(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config and attempting to overwrite file with defaults.{RESET}")
            loaded_config = default_config.copy()  # Fallback to defaults
            config_changed_needs_save = True  # Mark for saving the defaults
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error loading configuration: {e}. Using defaults.{RESET}", exc_info=True)
            loaded_config = default_config.copy()  # Fallback to defaults

    # --- Validate and Convert Loaded/Default Config ---
    # _validate_config_values modifies loaded_config in place
    if not _validate_config_values(loaded_config, lg):
        lg.critical("Configuration validation failed. Please check errors above and fix config.json. Exiting.")
        return None  # Indicate critical failure

    # Check if validation/conversion modified the config compared to what was loaded/created
    # This check handles cases where _validate_config_values converts types (e.g., str "10" to int 10)
    # Requires careful comparison, json dump/load might be easiest way to compare structurally
    try:
        # Compare the final validated config with the initial state (either default or loaded)
        initial_state_to_compare = default_config if not os.path.exists(filepath) else original_loaded_config
        if json.dumps(loaded_config, sort_keys=True) != json.dumps(initial_state_to_compare, sort_keys=True):
            config_changed_needs_save = True
    except TypeError as json_err:
         lg.warning(f"Could not compare config states due to non-serializable data: {json_err}")
    except Exception as cmp_err:
         lg.warning(f"Error comparing config states: {cmp_err}")

    # --- Save Updated Config if Necessary ---
    if config_changed_needs_save:
        lg.info(f"Configuration updated (missing keys added or types converted). Saving changes to '{filepath}'...")
        try:
            with open(filepath, "w", encoding="utf-8") as f_write:
                json.dump(loaded_config, f_write, indent=4, sort_keys=True)
            lg.info(f"{NEON_GREEN}Config file updated successfully.{RESET}")
        except OSError as e:
            lg.error(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
            # Continue with the validated config in memory, but saving failed

    lg.info("Configuration loaded and validated successfully.")
    return loaded_config


# --- State Management ---
def load_state(filepath: str, logger: logging.Logger) -> dict[str, Any]:
    """Loads the bot's operational state from a JSON file.
    Handles file not found and JSON decoding errors gracefully.

    Args:
        filepath: Path to the state file (e.g., "bot_state.json").
        logger: Logger instance.

    Returns:
        A dictionary containing the loaded state, or an empty dictionary if loading fails
        or the file doesn't exist.
    """
    lg = logger
    if os.path.exists(filepath):
        try:
            with open(filepath, encoding='utf-8') as f:
                state = json.load(f)
                lg.info(f"Loaded previous state from {filepath}")
                # Basic validation: Ensure it's a dictionary
                if not isinstance(state, dict):
                    lg.error(f"State file {filepath} does not contain a valid dictionary. Starting fresh.")
                    return {}
                # Optional: Add more validation here if state structure is critical
                # e.g., check expected keys for each symbol
                return state
        except json.JSONDecodeError as e:
            lg.error(f"Error decoding JSON from state file {filepath}: {e}. Starting with empty state.")
            return {}
        except OSError as e:
            lg.error(f"Error reading state file {filepath}: {e}. Starting with empty state.")
            return {}
        except Exception as e:
            lg.error(f"Unexpected error loading state: {e}. Starting with empty state.", exc_info=True)
            return {}
    else:
        lg.info(f"No previous state file found ('{filepath}'). Starting with empty state.")
        return {}


def save_state(filepath: str, state: dict[str, Any], logger: logging.Logger) -> None:
    """Saves the bot's current operational state to a JSON file using an atomic write pattern.
    Ensures complex types like Decimal are converted to strings for JSON compatibility.

    Args:
        filepath: Path to the state file (e.g., "bot_state.json").
        state: The dictionary containing the current state to save.
        logger: Logger instance.
    """
    lg = logger
    temp_filepath = filepath + ".tmp"  # Temporary file path for atomic write

    try:
        # Define a custom serializer function to handle non-standard JSON types
        def json_serializer(obj: Any) -> str:
            if isinstance(obj, Decimal):
                # Convert Decimal to string to preserve precision
                return str(obj)
            # Add handlers for other types if needed (e.g., datetime)
            # if isinstance(obj, datetime):
            #     return obj.isoformat()
            # Raise TypeError for unsupported types
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Write to temporary file first using the custom serializer
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            # Use default=json_serializer to handle Decimals etc.
            json.dump(state, f, indent=4, sort_keys=True, default=json_serializer)

        # Atomic rename/replace (os.replace is atomic on most modern systems)
        # This ensures that the original file is only overwritten if the write to temp file was successful.
        os.replace(temp_filepath, filepath)
        lg.debug(f"Saved current state successfully to {filepath}")

    except (OSError, TypeError, json.JSONDecodeError) as e:
        lg.error(f"Error saving state file {filepath}: {e}")
    except Exception as e:
        lg.error(f"Unexpected error saving state: {e}", exc_info=True)
    finally:
        # Clean up the temporary file if it still exists (e.g., due to error before replace)
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError as rm_err:
                lg.error(f"Error removing temporary state file {temp_filepath}: {rm_err}")


# --- CCXT Exchange Setup ---
def initialize_exchange(config: dict[str, Any], logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object.

    - Sets V5 API options, including broker ID and timeouts.
    - Configures sandbox mode based on config.
    - Loads exchange markets.
    - Tests API credentials and detects account type (UTA vs. Non-UTA) by fetching balance.
    - Sets global QUOTE_CURRENCY and IS_UNIFIED_ACCOUNT flags.

    Args:
        config: The bot's configuration dictionary.
        logger: The main logger instance.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails critically.
    """
    lg = logger
    global QUOTE_CURRENCY, IS_UNIFIED_ACCOUNT  # Allow modification of globals

    try:
        # Set global quote currency from config
        QUOTE_CURRENCY = config.get("quote_currency", "USDT")
        lg.info(f"Using Quote Currency: {QUOTE_CURRENCY}")

        # --- CCXT Exchange Options for Bybit V5 ---
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable CCXT's built-in rate limiter
            'rateLimit': 120,        # Default milliseconds between requests (Bybit V5 is faster, but CCXT handles actual limits)
            'options': {
                'defaultType': 'linear',  # Default to linear for futures/swaps (can be overridden by symbol format like BTC/USDT:USDT)
                'adjustForTimeDifference': True,  # Automatically adjust for time differences
                'recvWindow': 10000,     # Increased recvWindow (milliseconds) for potentially slower networks
                # --- Timeouts (milliseconds) - Slightly increased from defaults ---
                'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 25000,
                'createOrderTimeout': 30000, 'fetchOrderTimeout': 25000,
                'fetchPositionsTimeout': 25000, 'cancelOrderTimeout': 25000,
                'fetchOHLCVTimeout': 25000, 'setLeverageTimeout': 25000,
                'fetchMarketsTimeout': 45000,
                # --- Broker ID for identification (Optional, replace with your identifier if needed) ---
                'brokerId': f'EnhTradeBot{BOT_VERSION.replace(".", "")}',  # Example: EnhTradeBot110
                # --- Explicit V5 API Version Mapping (Good practice for clarity & stability) ---
                # Ensures CCXT uses V5 endpoints where intended. Match these paths with Bybit V5 documentation.
                'versions': {
                    'public': {
                        'GET': {
                            'market/tickers': 'v5', 'market/kline': 'v5', 'market/orderbook': 'v5',
                            # Add other public V5 endpoints used...
                        }
                    },
                    'private': {
                        'GET': {
                            'position/list': 'v5', 'account/wallet-balance': 'v5',
                            'order/realtime': 'v5', 'order/history': 'v5',
                            # Add other private GET V5 endpoints used...
                        },
                        'POST': {
                            'order/create': 'v5', 'order/cancel': 'v5',
                            'position/set-leverage': 'v5', 'position/trading-stop': 'v5',
                            # Add other private POST V5 endpoints used...
                        }
                    }
                },
                # --- Default Options hinting CCXT to prefer V5 methods ---
                # Reinforces the use of V5 for common operations.
                'default_options': {
                    'fetchPositions': 'v5', 'fetchBalance': 'v5', 'createOrder': 'v5',
                    'fetchOrder': 'v5', 'fetchTicker': 'v5', 'fetchOHLCV': 'v5',
                    'fetchOrderBook': 'v5', 'setLeverage': 'v5',
                    'private_post_position_trading_stop': 'v5',  # Explicit map for protection endpoint
                },
                # --- Account Type Mapping (Used internally by CCXT) ---
                # Helps CCXT understand Bybit's account structures.
                'accountsByType': {
                    'spot': 'SPOT', 'future': 'CONTRACT', 'swap': 'CONTRACT',
                    'margin': 'UNIFIED', 'option': 'OPTION', 'unified': 'UNIFIED',
                    'contract': 'CONTRACT',  # Explicitly map 'contract'
                },
                'accountsById': {  # Reverse mapping
                    'SPOT': 'spot', 'CONTRACT': 'contract', 'UNIFIED': 'unified', 'OPTION': 'option'
                },
                # --- Bybit Specific Options ---
                'bybit': {
                     # Helps CCXT resolve markets if settle currency isn't explicit in symbol
                    'defaultSettleCoin': QUOTE_CURRENCY
                }
            }
        }

        # Initialize exchange class
        exchange_id = 'bybit'
        exchange_class = getattr(ccxt, exchange_id)
        exchange: ccxt.Exchange = exchange_class(exchange_options)

        # --- Sandbox/Live Mode Configuration ---
        use_sandbox = config.get('use_sandbox', True)
        if use_sandbox:
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Bybit Testnet) - No real funds involved.{RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}--- USING LIVE TRADING MODE (Real Money) --- CAUTION ADVISED!{RESET}")
            # Ensure sandbox mode is explicitly off if not using it
            exchange.set_sandbox_mode(False)

        # --- Load Markets ---
        lg.info(f"Connecting to {exchange.id} (Sandbox: {use_sandbox})...")
        lg.info(f"Loading markets for {exchange.id}... (CCXT Version: {ccxt.__version__})")
        try:
            exchange.load_markets()
            # Store timestamp of last market load for periodic refresh logic
            exchange.last_load_markets_timestamp = time.time()
            lg.info(f"Markets loaded successfully for {exchange.id}. Found {len(exchange.markets)} markets.")
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            lg.critical(f"{NEON_RED}Fatal Error loading markets: {e}. Check network connection and API endpoint status.{RESET}", exc_info=True)
            return None  # Cannot proceed without markets
        except Exception as e:
            lg.critical(f"{NEON_RED}Unexpected Fatal Error loading markets: {e}.{RESET}", exc_info=True)
            return None

        # --- Test API Credentials & Detect Account Type ---
        lg.info(f"Attempting initial balance fetch for {QUOTE_CURRENCY} to test credentials and detect account type...")
        balance_decimal: Decimal | None = None
        account_type_detected: str | None = None
        try:
            # Use helper function to check both UTA and Non-UTA balance endpoints
            temp_is_unified, balance_decimal = _check_account_type_and_balance(exchange, QUOTE_CURRENCY, lg)

            if temp_is_unified is not None:
                IS_UNIFIED_ACCOUNT = temp_is_unified  # Set the global flag
                account_type_detected = "UNIFIED" if IS_UNIFIED_ACCOUNT else "CONTRACT/SPOT (Non-UTA)"
                lg.info(f"Detected Account Type: {account_type_detected}")
            else:
                # This indicates a failure to fetch balance using either common method
                lg.error(f"{NEON_RED}Could not determine account type or fetch initial balance.{RESET}")
                # If trading is enabled, this is critical.
                if config.get("enable_trading"):
                    lg.critical(f"{NEON_RED}Cannot verify balance/account type. Trading is enabled. Aborting initialization for safety.{RESET}")
                    return None
                else:
                    lg.warning(f"{NEON_YELLOW}Proceeding in non-trading mode despite inability to verify balance/account type.{RESET}")
                    # Set IS_UNIFIED_ACCOUNT to a default (e.g., False) or handle downstream checks carefully
                    IS_UNIFIED_ACCOUNT = False  # Default assumption if undetermined in non-trading mode
                    lg.warning(f"{NEON_YELLOW}Assuming Non-UTA account for non-trading operations.{RESET}")

            # Log balance status if fetched successfully
            if balance_decimal is not None:
                if balance_decimal > 0:
                    lg.info(f"{NEON_GREEN}Successfully connected. Initial {QUOTE_CURRENCY} balance: {balance_decimal:.4f}{RESET}")
                else:  # Balance is zero
                    lg.warning(f"{NEON_YELLOW}Successfully connected, but initial {QUOTE_CURRENCY} balance is zero.{RESET}")
            # If balance_decimal is None, the error was already logged by _check_account_type_and_balance or above

        except ccxt.AuthenticationError as auth_err:
            lg.critical(f"{NEON_RED}CCXT Authentication Error during initial setup: {auth_err}{RESET}")
            lg.critical(f"{NEON_RED}>> Please check API Key, API Secret, Permissions (Read/Trade for correct account type), Account Mode (Real/Testnet), and IP Whitelist.{RESET}")
            return None  # Fatal authentication error
        except Exception as balance_err:
            # Catch any other unexpected errors during the balance check phase
            lg.error(f"{NEON_RED}Unexpected error during initial balance check/account detection: {balance_err}{RESET}", exc_info=True)
            if config.get("enable_trading"):
                 lg.critical(f"{NEON_RED}Aborting initialization due to unexpected balance fetch error in trading mode.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}Continuing in non-trading mode despite unexpected balance fetch error.{RESET}")
                 IS_UNIFIED_ACCOUNT = False  # Default assumption
                 lg.warning(f"{NEON_YELLOW}Assuming Non-UTA account for non-trading operations.{RESET}")

        # If all checks passed (or warnings accepted in non-trading mode)
        lg.info("Exchange initialization checks passed.")
        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as e:
        # Catch errors during initial exchange class instantiation or configuration
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        # Catch any other unexpected errors during setup
        lg.critical(f"{NEON_RED}Unexpected error during exchange initialization: {e}{RESET}", exc_info=True)
        return None


def _check_account_type_and_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> tuple[bool | None, Decimal | None]:
    """Attempts to fetch balance using both UNIFIED and CONTRACT/SPOT account types
    to determine the account structure and retrieve the initial balance.

    Args:
        exchange: The CCXT exchange instance.
        currency: The currency (quote asset) to fetch the balance for (e.g., "USDT").
        logger: The logger instance.

    Returns:
        A tuple containing:
        - bool or None: True if UNIFIED, False if CONTRACT/SPOT, None if undetermined.
        - Decimal or None: The available balance as Decimal, or None if fetch failed.
    """
    lg = logger
    unified_balance: Decimal | None = None
    contract_spot_balance: Decimal | None = None
    is_unified_detected: bool | None = None

    # --- Attempt 1: Try fetching as UNIFIED account ---
    try:
        lg.debug("Checking balance with accountType=UNIFIED...")
        params_unified = {'accountType': 'UNIFIED', 'coin': currency}
        # Use safe_ccxt_call but with fewer retries for detection purposes
        # If this fails with auth error, safe_ccxt_call will raise it.
        bal_info_unified = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, retry_delay=2, params=params_unified)
        parsed_balance = _parse_balance_response(bal_info_unified, currency, 'UNIFIED', lg)

        # If parsing returns a valid Decimal (even 0), it means the UNIFIED call likely succeeded structurally.
        if parsed_balance is not None:
            lg.info("Detected UNIFIED account structure (Balance check successful).")
            unified_balance = parsed_balance
            is_unified_detected = True
            # Return immediately if UNIFIED balance is found and parsed
            return is_unified_detected, unified_balance

    except ccxt.ExchangeError as e:
        # Specifically check for errors indicating the account is *not* UNIFIED or lacks permissions
        error_str = str(e).lower()
        # Bybit error codes/messages indicating wrong account type for the API key
        # 30086: UTA not supported / Account Type mismatch
        # 10001 + "accounttype": Parameter error related to account type
        # 10005: Permissions denied for account type
        # Add more codes if discovered...
        is_account_type_error = (
            "30086" in error_str or
            "unified account is not supported" in error_str or
            ("10001" in error_str and "accounttype" in error_str) or
            ("10005" in error_str and ("permission denied" in error_str or "account type" in error_str))
        )
        if is_account_type_error:
             lg.debug("Fetching with UNIFIED failed with account type mismatch error (Code 30086, 10001, or 10005). Assuming Non-UTA.")
             is_unified_detected = False  # Strong indication of Non-UTA
        else:
             # Other exchange errors might be temporary network issues or unrelated problems
             lg.warning(f"ExchangeError checking UNIFIED balance: {e}. Will proceed to check CONTRACT/SPOT.")
             # Do not set is_unified_detected here, let the next check run.
    except Exception as e:
         # Includes potential AuthenticationError raised by safe_ccxt_call, or unexpected errors
         lg.warning(f"Error checking UNIFIED balance: {e}. Will proceed to check CONTRACT/SPOT if possible.")
         # If it was AuthenticationError, the main init function will catch it.

    # --- Attempt 2: Try fetching as CONTRACT/SPOT account (if UNIFIED failed or suggested Non-UTA) ---
    # Only proceed if unified check failed, indicated Non-UTA, or was inconclusive
    if is_unified_detected is False or is_unified_detected is None:
        lg.debug("Checking balance with CONTRACT/SPOT account types...")
        # Check CONTRACT first (common for futures), then SPOT.
        account_types_to_try = ['CONTRACT', 'SPOT']
        accumulated_non_uta_balance = Decimal('0')
        non_uta_balance_found = False

        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Checking balance with accountType={acc_type}...")
                params = {'accountType': acc_type, 'coin': currency}
                bal_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, retry_delay=2, params=params)
                parsed_balance = _parse_balance_response(bal_info, currency, acc_type, lg)

                if parsed_balance is not None:
                     lg.info(f"Successfully fetched balance using {acc_type} account type (Balance: {parsed_balance:.4f}).")
                     accumulated_non_uta_balance += parsed_balance
                     non_uta_balance_found = True
                     is_unified_detected = False  # Confirm Non-UTA structure success
                     # Don't return yet, check SPOT even if CONTRACT worked for non-UTA

            except ccxt.ExchangeError as e:
                 # Log errors but continue trying other types if possible
                 lg.warning(f"ExchangeError checking {acc_type} balance: {e}. Trying next type...")
            except Exception as e:
                 lg.warning(f"Unexpected error checking {acc_type} balance: {e}. Trying next type...")

        # After trying CONTRACT and SPOT:
        if non_uta_balance_found:
            contract_spot_balance = accumulated_non_uta_balance
            return False, contract_spot_balance  # Return False (Non-UTA) and the combined balance

    # --- Conclusion ---
    # This point is reached if:
    # 1) UNIFIED check succeeded (already returned).
    # 2) UNIFIED check failed conclusively (is_unified_detected=False) AND CONTRACT/SPOT check also failed to find balance.
    # 3) UNIFIED check had non-conclusive error AND CONTRACT/SPOT check also failed.
    if is_unified_detected is True:  # Should have returned earlier, but as safeguard
        return True, unified_balance
    if is_unified_detected is False and contract_spot_balance is not None:  # Should have returned earlier
        return False, contract_spot_balance

    # If we reach here, neither method definitively worked or found balance
    lg.error(f"Failed to determine account type OR fetch balance with common types (UNIFIED/CONTRACT/SPOT) for {currency}.")
    return None, None  # Unknown type, no balance retrieved


# --- CCXT API Call Helper with Retries ---
def safe_ccxt_call(
    exchange: ccxt.Exchange,
    method_name: str,
    logger: logging.Logger,
    max_retries: int = MAX_API_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
    *args: Any, **kwargs: Any
) -> Any:
    """Safely calls a CCXT exchange method with robust retry logic for recoverable errors.

    Handles:
    - RateLimitExceeded (with exponential backoff, parsing suggested wait time).
    - NetworkError, RequestTimeout, DDoSProtection (with exponential backoff).
    - Specific non-retryable Bybit V5 ExchangeErrors based on error codes.
    - AuthenticationError (raises immediately).
    - Other potentially temporary ExchangeErrors (retried).

    Args:
        exchange: The initialized CCXT exchange object.
        method_name: The name of the CCXT method to call (e.g., 'fetch_balance').
        logger: The logger instance for reporting errors and warnings.
        max_retries: Maximum number of retries for recoverable errors.
        retry_delay: Initial delay in seconds before the first retry (uses exponential backoff).
        *args: Positional arguments for the CCXT method.
        **kwargs: Keyword arguments for the CCXT method (often includes 'params' for V5).

    Returns:
        The result of the CCXT method call if successful.

    Raises:
        ccxt.AuthenticationError: If authentication fails (not retried).
        ccxt.ExchangeError: If a non-retryable exchange error occurs or retries are exhausted for a retryable one.
        ccxt.NetworkError: If a network error persists after retries.
        ccxt.RequestTimeout: If a timeout persists after retries.
        RuntimeError: If max retries are exceeded without a specific CCXT exception being raised (should not happen).
        Exception: Any other unexpected exception during the call.
    """
    lg = logger
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):  # +1 to include the initial attempt
        try:
            # Debug log for call attempt details (can be verbose)
            # lg.debug(f"Calling {method_name} (Attempt {attempt + 1}/{max_retries + 1}), Args: {args}, Kwargs: {kwargs}")

            # Get the method from the exchange instance
            method = getattr(exchange, method_name)
            # Execute the method
            result = method(*args, **kwargs)

            # Debug log for successful call (optional)
            # lg.debug(f"Call to {method_name} successful.")
            return result  # Success

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Calculate wait time using exponential backoff
            base_wait_time = retry_delay * (2 ** attempt)
            suggested_wait: float | None = None
            # Try to parse suggested wait time from Bybit V5 error message
            try:
                error_msg = str(e).lower()
                # Regex for "Retry after XXXXms" or "Try again in Xs"
                match_ms = re.search(r'(?:retry after|try again in)\s*(\d+)\s*ms', error_msg)
                match_s = re.search(r'(?:retry after|try again in)\s*(\d+)\s*s', error_msg)
                if match_ms:
                    # Convert ms to seconds, ensure minimum wait, add buffer
                    suggested_wait = max(1.0, math.ceil(int(match_ms.group(1)) / 1000) + RATE_LIMIT_BUFFER_SECONDS)
                elif match_s:
                    # Use seconds directly, ensure minimum wait, add buffer
                    suggested_wait = max(1.0, int(match_s.group(1)) + RATE_LIMIT_BUFFER_SECONDS)
                elif "too many visits" in error_msg or "limit" in error_msg or "frequency" in error_msg or "10006" in error_msg or "10018" in error_msg:
                    # Fallback if specific time isn't mentioned but it's clearly a rate limit code/message
                    suggested_wait = base_wait_time + RATE_LIMIT_BUFFER_SECONDS
            except Exception as parse_err:
                 lg.debug(f"Could not parse rate limit wait time from '{str(e)}': {parse_err}")

            # Use suggested wait if available, otherwise use calculated exponential backoff
            final_wait = suggested_wait if suggested_wait is not None else base_wait_time

            if attempt < max_retries:
                lg.warning(f"{NEON_YELLOW}Rate limit hit calling {method_name}. Retrying in {final_wait:.2f}s... (Attempt {attempt + 1}/{max_retries}) Error: {e}{RESET}")
                time.sleep(final_wait)
            else:  # Max retries reached
                lg.error(f"{NEON_RED}Rate limit hit calling {method_name}. Max retries ({max_retries}) reached. Error: {e}{RESET}")
                raise e  # Re-raise the final RateLimitExceeded exception

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            # Handle transient network issues, timeouts, or Cloudflare protection
            last_exception = e
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            if attempt < max_retries:
                lg.warning(f"{NEON_YELLOW}Network/Timeout/DDoS error calling {method_name}: {type(e).__name__}. Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries}){RESET}")
                lg.debug(f"Error details: {e}")
                time.sleep(wait_time)
            else:  # Max retries reached
                 lg.error(f"{NEON_RED}{type(e).__name__} calling {method_name}. Max retries ({max_retries}) reached. Error: {e}{RESET}")
                 raise e  # Re-raise the final exception

        except ccxt.AuthenticationError as e:
            # Authentication errors are critical and not retryable
            lg.critical(f"{NEON_RED}Authentication Error calling {method_name}: {e}. Check API keys/permissions/IP whitelist. Not retrying.{RESET}")
            raise e  # Re-raise immediately

        except ccxt.ExchangeError as e:
            # Handle general exchange errors, potentially parsing Bybit specifics
            last_exception = e
            bybit_code: int | None = None
            ret_msg: str = str(e)

            # --- Try to extract Bybit V5 retCode and retMsg ---
            # Check 'info' attribute first (often populated by CCXT)
            bybit_info = getattr(e, 'info', None)
            if isinstance(bybit_info, dict):
                bybit_code = bybit_info.get('retCode')
                ret_msg = bybit_info.get('retMsg', str(e))
            # Fallback: Try parsing the error string representation if info is missing/unhelpful
            if bybit_code is None:
                try:
                    # Look for JSON-like structure within the error string
                    json_part_match = re.search(r'({.*})', str(e))
                    if json_part_match:
                        details_dict = json.loads(json_part_match.group(1).replace("'", '"'))  # Basic quote fix
                        bybit_code = details_dict.get('retCode')
                        ret_msg = details_dict.get('retMsg', str(e))
                except (json.JSONDecodeError, IndexError, TypeError):
                    lg.debug(f"Could not parse Bybit code/msg from error string: {str(e)}")

            # --- Define known non-retryable Bybit V5 error codes ---
            # Consult Bybit V5 API documentation for the most up-to-date list:
            # https://bybit-exchange.github.io/docs/v5/error_code
            non_retryable_codes: list[int] = [
                # Parameter/Request Errors (10xxx) - Usually indicate client-side issue
                10001,  # Parameter error (check args/kwargs/permissions)
                10002,  # Request method/path not supported
                10003,  # Invalid API key or IP whitelist issue
                10004,  # Invalid sign / Authentication failed (redundant with AuthenticationError but good backup)
                10005,  # Permissions denied for API key
                # 10006 is Rate Limit (handled above)
                10009,  # IP address banned temporarily or permanently
                # 10010 is Request Timeout (handled above)
                10016,  # Service error / System maintenance (Might be temporary, but often requires longer wait or intervention)
                10017,  # Request path not found (likely CCXT/API mismatch)
                # 10018 is Frequency Limit (handled above as RateLimit)
                10020,  # Websocket issue (less relevant for REST but indicates system problem)
                10029,  # Request parameter validation failed (e.g., invalid format)
                # Order/Position Logic Errors (11xxx) - Generally indicate logic/state issues, not temporary network problems
                110001,  # Order placement failed (generic - check message)
                110003,  # Invalid price (precision, range)
                110004,  # Invalid quantity (precision, range)
                110005,  # Quantity too small (min order size)
                110006,  # Quantity too large (max order size)
                110007,  # Insufficient balance / margin
                110008,  # Order cost too small (min cost)
                110009,  # Order cost too large
                110010,  # Invalid order type ('market', 'limit')
                110011,  # Invalid side ('buy', 'sell')
                110012,  # Invalid timeInForce
                110013,  # Price deviates too much from mark price
                110014,  # Order ID not found or invalid (for cancel/amend)
                110015,  # Order already cancelled
                110016,  # Order already filled or partially filled and cannot be amended
                110017,  # Price/Quantity precision error
                110019,  # Cannot amend market orders
                110020,  # Position status prohibits action (e.g., during liquidation)
                110021,  # Risk limit exceeded for account/symbol
                110022,  # Invalid leverage value
                110024,  # Position not found
                110025,  # Position index error (Hedge Mode specific - positionIdx mismatch)
                110028,  # Reduce-only order would increase position size
                110031,  # Order quantity exceeds open order limit for symbol
                110033,  # Cannot set leverage in cross margin mode manually
                110036,  # Cross/Isolated mode mismatch for symbol
                110040,  # TP/SL order parameter error
                110041,  # TP/SL requires an active position
                110042,  # TP/SL price is invalid (e.g., triggers liquidation, wrong side of entry)
                110043,  # Leverage not modified (Often occurs if leverage is already set, handle as success)
                110044,  # Margin mode not modified
                110045,  # Position quantity exceeds risk limit tier
                110047,  # Cannot set TP/SL for Market orders during creation
                110051,  # Position is zero, cannot place reduce-only closing order
                110067,  # Feature requires UTA Pro account upgrade
                # Account/Risk/Position Errors (17xxx) - Indicate deeper issues
                170001,  # Internal error affecting position calculation
                170007,  # Risk limit error (check position value limits)
                170019,  # Account in margin call / liquidation status
                170131,  # TP/SL price invalid (redundant with 110042?)
                170132,  # TP/SL order would trigger immediate liquidation
                170133,  # Cannot set TP/SL/TSL (generic, check position status)
                170140,  # TP/SL/TSL setting requires an active position (redundant with 110041?)
                # Account Type Errors (3xxxx)
                30086,  # UTA not supported / Account Type mismatch with API key/endpoint
                30087,  # UTA feature unavailable for this account
                # Add more critical, non-recoverable codes as identified from Bybit docs or experience
            ]

            # --- Handle Specific Cases ---
            # 110043: Leverage not modified - Treat as success for set_leverage calls
            if bybit_code == 110043 and method_name == 'set_leverage':
                lg.info(f"Leverage already set as requested (Code 110043) when calling {method_name}. Treating as success.")
                return {}  # Return empty dict, often treated as success signal by calling code

            # --- Check if code is non-retryable ---
            if bybit_code in non_retryable_codes:
                # Provide extra context/hints for common non-retryable errors
                extra_info = ""
                if bybit_code == 10001: extra_info = f"{NEON_YELLOW} Hint: Check API call parameters ({args=}, {kwargs=}) or API key permissions/account type.{RESET}"
                elif bybit_code == 110007:
                    balance_currency = QUOTE_CURRENCY  # Default guess
                    try:  # Try harder to guess the currency being checked
                         if 'params' in kwargs and 'coin' in kwargs['params']: balance_currency = kwargs['params']['coin']
                         elif 'symbol' in kwargs and isinstance(kwargs['symbol'], str): balance_currency = kwargs['symbol'].split('/')[1].split(':')[0]
                         elif len(args) > 0 and isinstance(args[0], str) and '/' in args[0]: balance_currency = args[0].split('/')[1].split(':')[0]
                    except Exception: pass
                    extra_info = f"{NEON_YELLOW} Hint: Check available {balance_currency} balance in the correct account (UTA/Contract/Spot).{RESET}"
                elif bybit_code == 30086 or (bybit_code == 10001 and "accounttype" in ret_msg.lower()):
                     extra_info = f"{NEON_YELLOW} Hint: Check 'accountType' param (UNIFIED vs CONTRACT/SPOT) matches your account/API key permissions.{RESET}"
                elif bybit_code == 110025:
                     extra_info = f"{NEON_YELLOW} Hint: Check 'positionIdx' parameter (0 for One-Way, 1/2 for Hedge Mode) and ensure it matches your account's Position Mode setting.{RESET}"
                elif bybit_code in [110003, 110004, 110005, 110006, 110017]:
                     extra_info = f"{NEON_YELLOW} Hint: Check price/amount precision and limits against market data (min/max size, tick size, step size).{RESET}"
                elif bybit_code in [110041, 170140, 110024]:
                     extra_info = f"{NEON_YELLOW} Hint: Action requires an active position, but none found or specified correctly (Code {bybit_code}).{RESET}"

                lg.error(f"{NEON_RED}Non-retryable Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Not retrying.{RESET}{extra_info}")
                raise e  # Re-raise the non-retryable error

            else:  # Unknown or potentially temporary exchange error, proceed with retry logic
                if attempt < max_retries:
                    lg.warning(f"{NEON_YELLOW}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Retrying... (Attempt {attempt + 1}/{max_retries}){RESET}")
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:  # Max retries reached for this unknown/retryable error
                    lg.error(f"{NEON_RED}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Max retries ({max_retries}) reached.{RESET}")
                    raise e  # Re-raise after max retries

        except Exception as e:
            # Catch any other unexpected errors during the method call
            lg.error(f"{NEON_RED}Unexpected Error calling {method_name}: {e}. Not retrying.{RESET}", exc_info=True)
            raise e  # Re-raise unexpected errors immediately

    # --- Max Retries Reached ---
    # This part should only be reached if the loop finishes after max retries for recoverable errors
    lg.error(f"{NEON_RED}Max retries ({max_retries}) reached for {method_name}. Last error: {type(last_exception).__name__}{RESET}")
    if isinstance(last_exception, Exception):
        raise last_exception  # Re-raise the last specific recoverable exception
    else:
        # Should not happen if last_exception is always assigned, but as a fallback
        raise RuntimeError(f"Max retries reached for {method_name} without specific exception recorded.")


# --- Market Info Helper Functions ---
def _determine_category(market: dict[str, Any]) -> str | None:
    """Determines the Bybit V5 category ('linear', 'inverse', 'spot', 'option')
    from CCXT market info dictionary. Prioritizes explicit flags then infers.

    Args:
        market: The market dictionary from ccxt.exchange.markets.

    Returns:
        The category string ('linear', 'inverse', 'spot', 'option') or None if undetermined.
    """
    if not isinstance(market, dict): return None

    # Priority 1: Explicit V5 'category' field in 'info'
    info = market.get('info', {})
    category_info = info.get('category')
    if category_info in ['linear', 'inverse', 'spot', 'option']:
        return category_info

    # Priority 2: Standard CCXT type flags
    market_type = market.get('type')  # 'spot', 'swap', 'future', 'option'
    is_linear = market.get('linear', False)
    is_inverse = market.get('inverse', False)
    is_spot = market.get('spot', False)
    is_option = market.get('option', False)

    if is_spot or market_type == 'spot': return 'spot'
    if is_option or market_type == 'option': return 'option'

    # Priority 3: Infer for contracts (swap/future) based on linear/inverse flags
    if market_type in ['swap', 'future']:
        if is_linear: return 'linear'
        if is_inverse: return 'inverse'
        # Fallback inference based on settle vs quote asset (less reliable)
        settle_asset = market.get('settle', '').upper()
        quote_asset = market.get('quote', '').upper()
        if settle_asset and quote_asset:
            # Linear typically settles in quote (e.g., USDT, USDC)
            # Inverse typically settles in base (e.g., BTC, ETH)
            return 'linear' if settle_asset == quote_asset else 'inverse'

    # If category couldn't be determined
    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict[str, Any] | None:
    """Retrieves and processes detailed market information for a symbol from loaded CCXT markets.

    Extracts key details required for trading logic, including:
    - Precision (tick size, amount step size, digits)
    - Limits (min/max amount, price, cost)
    - Contract details (contract size, type, category, inverse)
    - Stores results as Decimal where appropriate for precision.

    Args:
        exchange: Initialized CCXT exchange object with loaded markets.
        symbol: The standard CCXT trading symbol (e.g., 'BTC/USDT:USDT' or 'BTC/USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing processed market info, or None if the symbol is not found,
        markets are not loaded, or processing fails.
        Keys include: 'symbol', 'id', 'base', 'quote', 'settle', 'type', 'category',
                      'is_contract', 'inverse', 'contract_size' (Decimal),
                      'min_tick_size' (Decimal), 'amount_step_size' (Decimal),
                      'price_precision_digits' (int), 'amount_precision_digits' (int),
                      'min_order_amount' (Decimal), 'max_order_amount' (Decimal),
                      'min_price' (Decimal), 'max_price' (Decimal),
                      'min_order_cost' (Decimal), 'max_order_cost' (Decimal),
                      'raw_market_data' (original dict).
    """
    lg = logger
    if not exchange.markets:
        lg.error(f"Cannot get market info for {symbol}: Markets not loaded on exchange object.")
        return None
    try:
        # Retrieve market data from CCXT's loaded markets
        market = exchange.market(symbol)
        if not market:
            # Try splitting symbol if it contains ':' (e.g., 'BTC/USDT:USDT' -> 'BTC/USDT')
            # Some exchanges might list linear contracts under the base symbol
            if ':' in symbol:
                 base_symbol = symbol.split(':')[0]
                 lg.debug(f"Symbol '{symbol}' not found directly, trying base symbol '{base_symbol}'...")
                 market = exchange.market(base_symbol)

            if not market:  # Still not found
                lg.error(f"Symbol '{symbol}' (or base '{base_symbol if ':' in symbol else ''}') not found in loaded markets.")
                # Log available symbols for debugging if needed (can be very long)
                # lg.debug(f"Available market symbols: {list(exchange.markets.keys())}")
                return None

        # Determine V5 category ('linear', 'inverse', 'spot', 'option')
        category = _determine_category(market)
        if category is None:
            # Log warning but proceed cautiously if category is vital downstream
            lg.warning(f"Could not reliably determine V5 category for symbol {symbol}. Market data: {market}")

        # --- Extract Precision Details ---
        price_precision_val = market.get('precision', {}).get('price')  # This is the tick size
        amount_precision_val = market.get('precision', {}).get('amount')  # This is the amount step size

        # Helper to calculate decimal digits from a precision value (e.g., 0.01 -> 2 digits)
        def get_digits_from_precision(precision_val: float | int | str | None, default_digits: int = 8) -> int:
            if precision_val is None: return default_digits
            try:
                # Use Decimal for accurate calculation, especially for small numbers
                decimal_val = Decimal(str(precision_val))
                if not decimal_val.is_finite() or decimal_val <= 0: return default_digits
                # Calculate digits: -log10(precision). Handle precision >= 1.
                if decimal_val >= 1: return 0  # 0 decimal places for precision 1, 10, etc.
                digits = int(decimal_val.log10().copy_negate())
                return max(0, digits)  # Ensure non-negative digits
            except (InvalidOperation, ValueError, TypeError) as e:
                 lg.warning(f"Could not calculate digits from precision value {precision_val}: {e}. Using default {default_digits}.")
                 return default_digits

        price_digits = get_digits_from_precision(price_precision_val)
        amount_digits = get_digits_from_precision(amount_precision_val)

        # --- Extract Limits ---
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        cost_limits = limits.get('cost', {})

        # Helper to convert limit values to Decimal safely
        def to_decimal_safe(val: Any) -> Decimal | None:
            if val is None: return None
            try:
                # Convert to string first to handle floats accurately
                d = Decimal(str(val))
                # Return Decimal if finite, otherwise None
                return d if d.is_finite() else None
            except (InvalidOperation, TypeError, ValueError): return None

        # --- Contract-Specific Details ---
        # Default contract size to 1 (relevant for spot, linear)
        contract_size = to_decimal_safe(market.get('contractSize', '1')) or Decimal('1')
        is_contract = category in ['linear', 'inverse']
        is_inverse = category == 'inverse'

        # --- Assemble Processed Market Info Dictionary ---
        market_details = {
            'symbol': market.get('symbol', symbol),  # Use the symbol found in market data
            'id': market.get('id'),  # Exchange's internal market ID
            'base': market.get('base'),
            'quote': market.get('quote'),
            'settle': market.get('settle'),
            'type': market.get('type'),  # spot, swap, future
            'category': category,       # linear, inverse, spot, option (best guess)
            'is_contract': is_contract,
            'inverse': is_inverse,
            'contract_size': contract_size,
            # Store precision values as Decimal
            'min_tick_size': to_decimal_safe(price_precision_val),
            'amount_step_size': to_decimal_safe(amount_precision_val),
            # Store calculated digits
            'price_precision_digits': price_digits,
            'amount_precision_digits': amount_digits,
            # Store limits as Decimal
            'min_order_amount': to_decimal_safe(amount_limits.get('min')),
            'max_order_amount': to_decimal_safe(amount_limits.get('max')),
            'min_price': to_decimal_safe(price_limits.get('min')),
            'max_price': to_decimal_safe(price_limits.get('max')),
            'min_order_cost': to_decimal_safe(cost_limits.get('min')),
            'max_order_cost': to_decimal_safe(cost_limits.get('max')),
            'raw_market_data': market  # Include original market data for debugging
        }
        lg.debug(f"Market Info for {symbol}: Cat={category}, Tick={market_details['min_tick_size']}, AmtStep={market_details['amount_step_size']}, ContSize={contract_size}")
        return market_details

    except ccxt.BadSymbol as e:
        # Specific error if CCXT couldn't find the symbol
        lg.error(f"Error getting market info for '{symbol}': {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during processing
        lg.error(f"Unexpected error processing market info for {symbol}: {e}", exc_info=True)
        return None


# --- Data Fetching Wrappers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict) -> Decimal | None:
    """Fetches the current ticker price using V5 API via safe_ccxt_call.
    Parses the 'lastPrice' field from the V5 response structure.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.
        market_info: Market information dictionary (must contain 'category' and 'id').

    Returns:
        The current price as Decimal, or None if fetch/parsing fails or price is invalid.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id')  # Use exchange-specific ID

    if not category or not market_id:
        lg.error(f"Cannot fetch price for {symbol}: Category ('{category}') or Market ID ('{market_id}') missing in market_info."); return None

    try:
        # V5 ticker endpoint requires category and symbol (market_id)
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching ticker for {symbol} (MarketID: {market_id}) with params: {params}")

        # Pass standard CCXT symbol for resolution, and V5 params for the API call
        ticker = safe_ccxt_call(exchange, 'fetch_ticker', lg, symbol=symbol, params=params)

        ticker_data = None
        price_str: str | None = None

        # --- Parse V5 Response Structure ---
        # Bybit V5 ticker response might be nested under info -> result -> list -> [item] (for category requests)
        # or info -> result -> item (less common for single symbol fetch via fetch_ticker?)
        # or directly in the main dict (CCXT standard 'last' field)
        if isinstance(ticker, dict):
            # Check V5 structure (info -> result -> list -> [item])
            if ('info' in ticker and isinstance(ticker['info'], dict) and
                ticker['info'].get('retCode') == 0 and 'result' in ticker['info'] and
                isinstance(ticker['info']['result'], dict) and 'list' in ticker['info']['result'] and
                isinstance(ticker['info']['result']['list'], list) and len(ticker['info']['result']['list']) > 0):

                # Find the ticker matching the market_id within the list
                for item in ticker['info']['result']['list']:
                    if isinstance(item, dict) and item.get('symbol') == market_id:
                        ticker_data = item
                        price_str = ticker_data.get('lastPrice')  # V5 field name
                        lg.debug(f"Parsed price from V5 list structure for {symbol} (MarketID: {market_id})")
                        break  # Found the matching ticker

            # Check alternative V5 structure (info -> result -> single item)
            elif ('info' in ticker and isinstance(ticker['info'], dict) and
                  ticker['info'].get('retCode') == 0 and 'result' in ticker['info'] and
                  isinstance(ticker['info']['result'], dict) and not ticker['info']['result'].get('list')):
                 ticker_data = ticker['info']['result']
                 # Verify symbol match in case API returns unexpected data
                 if ticker_data.get('symbol') == market_id:
                     price_str = ticker_data.get('lastPrice')  # V5 field name
                     lg.debug(f"Parsed price from V5 single result structure for {symbol}")
                 else:
                      lg.warning(f"Ticker result symbol '{ticker_data.get('symbol')}' mismatch for requested MarketID {market_id}")

            # Fallback to standard CCXT 'last' field if V5 parsing fails or structure differs
            if price_str is None and 'last' in ticker and ticker['last'] is not None:
                 price_str = str(ticker['last'])
                 lg.debug(f"Parsed price from standard CCXT 'last' field for {symbol}")

        # --- Validate and Convert Price ---
        if price_str is None or price_str == "":
            lg.warning(f"Could not extract valid 'lastPrice' or 'last' for {symbol}. Ticker response: {ticker}")
            return None

        price_dec = Decimal(price_str)
        # Ensure price is finite and positive
        if price_dec.is_finite() and price_dec > 0:
            lg.debug(f"Current price for {symbol}: {price_dec}")
            return price_dec
        else:
            lg.error(f"Invalid price ('{price_str}') received for {symbol} (non-finite or non-positive).")
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
         lg.error(f"Error converting fetched price '{price_str}' to Decimal for {symbol}: {e}")
         return None
    except Exception as e:
        # Catches errors from safe_ccxt_call (e.g., retries exhausted, non-retryable errors) or other issues
        lg.error(f"Error fetching current price for {symbol}: {e}", exc_info=True)
        return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger, market_info: dict) -> pd.DataFrame:
    """Fetches OHLCV kline data using V5 API via safe_ccxt_call.
    Returns a DataFrame with a UTC timestamp index and Decimal OHLCV columns.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        timeframe: Bot interval string (e.g., "5", "60", "D"). Matches keys in CCXT_INTERVAL_MAP.
        limit: Number of kline candles to fetch.
        logger: Logger instance.
        market_info: Market information dictionary (must contain 'category' and 'id').

    Returns:
        Pandas DataFrame with 'timestamp' (UTC datetime index) and
        'open', 'high', 'low', 'close', 'volume' (Decimal) columns.
        Returns an empty DataFrame on failure or if insufficient data is returned.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id')  # Use exchange-specific ID
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)  # Get CCXT format (e.g., '5m')

    # --- Validate Inputs ---
    if not category or not market_id:
        lg.error(f"Cannot fetch klines for {symbol}: Category ('{category}') or Market ID ('{market_id}') missing."); return pd.DataFrame()
    if not ccxt_timeframe:
        lg.error(f"Invalid timeframe '{timeframe}' provided for {symbol}. Valid keys: {list(CCXT_INTERVAL_MAP.keys())}"); return pd.DataFrame()

    try:
        # Bybit V5 kline endpoint takes category and symbol (market_id)
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching {limit} klines for {symbol} ({ccxt_timeframe}, MarketID: {market_id}) with params: {params}")

        # Adjust limit based on exchange maximum (Bybit V5 kline limit is 1000)
        safe_limit = min(limit, 1000)
        if limit > 1000: lg.warning(f"Requested kline limit {limit} exceeds Bybit's max of 1000. Fetching {safe_limit}.")

        # CCXT fetch_ohlcv maps to the correct V5 endpoint based on category/symbol
        ohlcv = safe_ccxt_call(exchange, 'fetch_ohlcv', lg, symbol=symbol, timeframe=ccxt_timeframe, limit=safe_limit, params=params)

        if not ohlcv:
            lg.warning(f"fetch_ohlcv returned empty data for {symbol} ({ccxt_timeframe}).")
            return pd.DataFrame()

        # --- Convert to DataFrame and Process ---
        # Standard CCXT OHLCV format: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"Kline data for {symbol} resulted in an empty DataFrame after initial creation.")
             return df

        # Convert timestamp (milliseconds) to UTC datetime and set as index
        try:
            # errors='coerce' will turn invalid timestamps into NaT (Not a Time)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            # Drop rows with invalid timestamps before setting index
            initial_len = len(df)
            df.dropna(subset=['timestamp'], inplace=True)
            if len(df) < initial_len:
                lg.warning(f"Dropped {initial_len - len(df)} rows with invalid timestamps for {symbol}.")
            if df.empty:
                 lg.warning(f"DataFrame became empty after dropping invalid timestamps for {symbol}.")
                 return df
            df.set_index('timestamp', inplace=True)
        except Exception as dt_err:
             lg.error(f"Error converting timestamp column for {symbol}: {dt_err}", exc_info=True)
             return pd.DataFrame()  # Cannot proceed without valid index

        # Convert OHLCV columns to Decimal for precision
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_convert:
            if col not in df.columns: continue  # Skip if column somehow missing
            try:
                # Convert to string first -> Decimal avoids potential float inaccuracies.
                # Apply Decimal conversion robustly using .apply()
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
            except (InvalidOperation, TypeError, ValueError) as e:
                 lg.error(f"Error converting column '{col}' to Decimal for {symbol}: {e}. Attempting numeric conversion with NaN coercion.")
                 # Fallback: Convert to numeric, coercing errors to NaN, then try Decimal again
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else None)

        # Final check and drop rows with NaN/None in critical OHLC columns
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        if len(df) < initial_len:
            lg.warning(f"Dropped {initial_len - len(df)} rows with NaN/None values in OHLC columns for {symbol}.")

        if df.empty:
            lg.warning(f"DataFrame became empty after NaN drop for {symbol}.")

        lg.debug(f"Fetched and processed {len(df)} klines for {symbol}.")
        return df

    except Exception as e:
        # Catch errors from safe_ccxt_call or DataFrame processing
        lg.error(f"Error fetching/processing klines for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()  # Return empty DataFrame on error


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: dict) -> dict | None:
    """Fetches order book data using V5 API via safe_ccxt_call.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        limit: Number of order book levels to fetch (e.g., 1, 25, 50). Bybit V5 allows 1, 50, 200.
        logger: Logger instance.
        market_info: Market information dictionary (must contain 'category' and 'id').

    Returns:
        A dictionary containing 'bids' and 'asks' lists (each item is [price_str, amount_str]),
        or None if fetch fails or response is invalid. Prices/amounts are strings as returned by CCXT.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id')  # Use exchange-specific ID

    if not category or not market_id:
        lg.error(f"Cannot fetch orderbook for {symbol}: Category ('{category}') or Market ID ('{market_id}') missing."); return None

    # Validate limit against Bybit V5 allowed values (adjust if necessary)
    valid_limits = [1, 50, 200]
    if limit not in valid_limits:
        # Find closest valid limit (e.g., default to 50)
        if limit < 50: limit = 1
        elif limit < 200: limit = 50
        else: limit = 200
        lg.warning(f"Invalid orderbook limit requested. Adjusted to nearest valid Bybit V5 limit: {limit}")

    try:
        # Bybit V5 orderbook endpoint uses category and symbol, limit in params
        params = {'category': category, 'symbol': market_id, 'limit': limit}
        lg.debug(f"Fetching order book (limit {limit}) for {symbol} (MarketID: {market_id}) with params: {params}")

        # CCXT fetch_order_book should map to the correct V5 endpoint
        orderbook = safe_ccxt_call(exchange, 'fetch_order_book', lg, symbol=symbol, limit=limit, params=params)

        # --- Validate the Structure ---
        if (orderbook and isinstance(orderbook, dict) and
            'bids' in orderbook and isinstance(orderbook['bids'], list) and
            'asks' in orderbook and isinstance(orderbook['asks'], list) and
            'timestamp' in orderbook and orderbook['timestamp'] is not None):  # Check timestamp for validity

            # Basic validation of bid/ask entries format [price_str, amount_str]
            is_valid = True
            for level in orderbook['bids'] + orderbook['asks']:
                 if not (isinstance(level, (list, tuple)) and len(level) == 2 and
                         isinstance(level[0], (str, int, float)) and isinstance(level[1], (str, int, float))):
                     is_valid = False; break
            if not is_valid:
                 lg.warning(f"Order book for {symbol} contains invalid level format. Response: {orderbook}")
                 return None

            # Check if data was returned (lists might be empty if market is illiquid)
            if orderbook['bids'] or orderbook['asks']:
                 lg.debug(f"Fetched order book for {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 # Return the orderbook with string prices/amounts
                 return orderbook
            else:
                 lg.warning(f"Order book fetch for {symbol} returned empty bids/asks lists (possibly illiquid market).")
                 return orderbook  # Return empty book, might be valid state
        else:
            lg.warning(f"Failed to fetch valid order book structure for {symbol}. Response: {orderbook}")
            return None

    except Exception as e:
        # Catch errors from safe_ccxt_call or validation
        lg.error(f"Error fetching order book for {symbol}: {e}", exc_info=True)
        return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency using safe_ccxt_call.
    Adapts the request based on the detected account type (UTA vs. Non-UTA).

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency symbol (e.g., 'USDT').
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetch fails, balance is zero/invalid,
        or parsing fails.
    """
    lg = logger
    account_types_to_query: list[str] = []

    # Determine which account types to query based on the global flag
    if IS_UNIFIED_ACCOUNT:
        account_types_to_query = ['UNIFIED']
        lg.debug(f"Fetching balance specifically for UNIFIED account ({currency}).")
    else:
        # For Non-UTA, CONTRACT usually holds futures balance, SPOT for spot balance.
        # Query both as funds might be in either for trading, sum them up.
        account_types_to_query = ['CONTRACT', 'SPOT']
        lg.debug(f"Fetching balance for Non-UTA account ({currency}), trying types: {account_types_to_query}.")

    last_exception: Exception | None = None
    total_available_balance = Decimal('0')
    balance_successfully_fetched = False

    # Outer retry loop for network/rate limit issues across account type checks
    for attempt in range(MAX_API_RETRIES + 1):
        balance_found_in_attempt = False
        error_in_attempt = False
        current_attempt_balance = Decimal('0')  # Balance accumulated in this attempt cycle

        # Inner loop to try different account types (only relevant for Non-UTA)
        for acc_type in account_types_to_query:
            try:
                params = {'accountType': acc_type, 'coin': currency}
                lg.debug(f"Fetching balance with params={params} (Overall Attempt {attempt + 1})")
                # Use safe_ccxt_call with 0 inner retries; outer loop handles retries for network issues
                balance_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=0, params=params)

                # Parse the response for the specific account type
                current_type_balance = _parse_balance_response(balance_info, currency, acc_type, lg)

                if current_type_balance is not None:
                    balance_found_in_attempt = True
                    current_attempt_balance += current_type_balance
                    # If UTA, we are done once found
                    if IS_UNIFIED_ACCOUNT:
                        break  # Exit inner loop for UTA
                else:
                    lg.debug(f"Balance for {currency} not found or parsing failed for type {acc_type}.")

            except ccxt.ExchangeError as e:
                # Non-retryable errors (like auth, param) should have been raised by safe_ccxt_call
                # Log other exchange errors and continue to next type if applicable (for Non-UTA)
                lg.warning(f"Exchange error fetching balance type {acc_type}: {e}. Trying next type if available.")
                last_exception = e
                continue  # Try the next account type if Non-UTA

            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                 lg.warning(f"Network/RateLimit error during balance fetch type {acc_type}: {e}")
                 last_exception = e
                 error_in_attempt = True
                 break  # Break inner loop, let outer loop handle retry

            except Exception as e:
                 lg.error(f"Unexpected error during balance fetch type {acc_type}: {e}", exc_info=True)
                 last_exception = e
                 error_in_attempt = True
                 break  # Break inner loop, let outer loop handle retry

        # --- After Inner Loop (Account Types) ---
        if balance_found_in_attempt and not error_in_attempt:
            # If we successfully checked all required types without recoverable errors
            total_available_balance = current_attempt_balance
            balance_successfully_fetched = True
            break  # Exit outer retry loop successfully

        if error_in_attempt:  # If broke from inner loop due to recoverable error
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} encountered recoverable error ({type(last_exception).__name__}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue  # Continue outer retry loop
            else:  # Max retries reached for recoverable error
                 lg.error(f"{NEON_RED}Max retries reached fetching balance for {currency} after {type(last_exception).__name__}. Last error: {last_exception}{RESET}")
                 return None  # Failed after retries

        # If inner loop completed without finding balance (and without errors breaking it)
        if not balance_found_in_attempt:
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} failed to find/parse balance for type(s): {account_types_to_query}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue  # Retry outer loop
            else:  # Max retries reached, balance not found/parsed
                 lg.error(f"{NEON_RED}Max retries reached. Failed to fetch/parse balance for {currency} ({' '.join(account_types_to_query)}). Last error: {last_exception}{RESET}")
                 return None  # Failed to get balance

    # --- Return Result ---
    if balance_successfully_fetched:
        account_str = "UNIFIED" if IS_UNIFIED_ACCOUNT else "Non-UTA (CONTRACT+SPOT)"
        lg.info(f"Available {currency} balance ({account_str}): {total_available_balance:.4f}")
        return total_available_balance
    else:
        # Should have returned None earlier if fetching failed after retries
        lg.error(f"{NEON_RED}Balance fetch logic completed unexpectedly without success for {currency}.{RESET}")
        return None


def _parse_balance_response(balance_info: dict | None, currency: str, account_type_checked: str, logger: logging.Logger) -> Decimal | None:
    """Parses the raw response from CCXT's fetch_balance, adapting to Bybit V5 structure.

    Prioritizes fields indicating actually available funds:
    - V5 UNIFIED: 'availableToWithdraw' (most reliable for UTA)
    - V5 CONTRACT/SPOT: 'availableBalance'
    - Falls back to CCXT standard 'free' field if V5 fields are missing.

    Args:
        balance_info: The raw dictionary returned by safe_ccxt_call('fetch_balance').
        currency: The currency symbol (e.g., 'USDT') to look for.
        account_type_checked: The 'accountType' used in the request ('UNIFIED', 'CONTRACT', 'SPOT').
        logger: Logger instance.

    Returns:
        The available balance as Decimal, or None if not found, invalid, or parsing fails.
    """
    if not balance_info or not isinstance(balance_info, dict):
        logger.debug(f"Parsing balance: Received empty or invalid balance_info for {currency} ({account_type_checked}).")
        return None
    lg = logger
    available_balance_str: str | None = None
    field_used: str = "N/A"

    try:
        # --- Strategy 1: Parse Bybit V5 structure (info -> result -> list -> coin[]) ---
        v5_parsed = False
        if ('info' in balance_info and isinstance(balance_info['info'], dict) and
            balance_info['info'].get('retCode') == 0 and 'result' in balance_info['info'] and
            isinstance(balance_info['info']['result'], dict) and 'list' in balance_info['info']['result'] and
            isinstance(balance_info['info']['result']['list'], list)):

            balance_list = balance_info['info']['result']['list']
            lg.debug(f"Parsing V5 balance structure for {currency} ({account_type_checked}). List length: {len(balance_list)}")

            for account_data in balance_list:
                # Find the dictionary matching the account type we requested
                if isinstance(account_data, dict) and account_data.get('accountType') == account_type_checked:
                    coin_list = account_data.get('coin')
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                            if isinstance(coin_data, dict) and coin_data.get('coin') == currency:
                                # --- Determine the most appropriate balance field ---
                                potential_balance = None
                                if account_type_checked == 'UNIFIED':
                                    # For UTA, 'availableToWithdraw' is generally the most reliable available balance
                                    potential_balance = coin_data.get('availableToWithdraw')
                                    field_used = 'V5:availableToWithdraw'
                                    if potential_balance is None:  # Fallback if missing
                                        potential_balance = coin_data.get('availableBalance')
                                        field_used = 'V5:availableBalance (fallback)'
                                else:  # CONTRACT or SPOT
                                    # For non-UTA, 'availableBalance' is the standard available field
                                    potential_balance = coin_data.get('availableBalance')
                                    field_used = 'V5:availableBalance'
                                    if potential_balance is None:  # Fallback if missing
                                         potential_balance = coin_data.get('availableToWithdraw')
                                         field_used = 'V5:availableToWithdraw (fallback)'

                                # Use walletBalance only as a last resort if others are missing (less accurate for available funds)
                                if potential_balance is None:
                                     lg.debug(f"Available balance fields missing for {currency} ({account_type_checked}), checking 'walletBalance'.")
                                     potential_balance = coin_data.get('walletBalance')
                                     field_used = 'V5:walletBalance (last resort)'

                                # Check if we found a non-empty value
                                if potential_balance is not None and str(potential_balance).strip() != "":
                                    available_balance_str = str(potential_balance)
                                    v5_parsed = True
                                    lg.debug(f"Parsed balance from Bybit V5 ({account_type_checked} -> {currency}): Value='{available_balance_str}' (Field: '{field_used}')")
                                    break  # Found currency in this account's coin list
                        if v5_parsed:
                            break  # Found currency in this account type
            if not v5_parsed:
                 lg.debug(f"Currency '{currency}' not found within Bybit V5 list structure for account type '{account_type_checked}'.")

        # --- Strategy 2: Fallback to standard CCXT 'free' balance structure ---
        # This might be populated by CCXT even if V5 structure exists or failed parsing
        if not v5_parsed and currency in balance_info and isinstance(balance_info.get(currency), dict):
            free_val = balance_info[currency].get('free')
            if free_val is not None:
                available_balance_str = str(free_val)
                field_used = "CCXT:free"
                lg.debug(f"Parsed balance via standard CCXT structure ['{currency}']['free']: {available_balance_str}")

        # --- Strategy 3: Fallback to top-level 'free' dictionary (less common for specific currency) ---
        elif not v5_parsed and available_balance_str is None and 'free' in balance_info and isinstance(balance_info.get('free'), dict):
             free_val = balance_info['free'].get(currency)
             if free_val is not None:
                 available_balance_str = str(free_val)
                 field_used = "CCXT:top_level_free"
                 lg.debug(f"Parsed balance via top-level 'free' dictionary ['free']['{currency}']: {available_balance_str}")

        # --- Conversion and Validation ---
        if available_balance_str is None:
            lg.debug(f"Could not extract available balance for {currency} from response structure ({account_type_checked}). Response info keys: {balance_info.get('info', {}).keys() if isinstance(balance_info.get('info'), dict) else 'N/A'}")
            return None

        # Attempt conversion to Decimal
        final_balance = Decimal(available_balance_str)
        # Ensure balance is finite and non-negative
        if final_balance.is_finite() and final_balance >= 0:
            # Return the balance (could be 0)
            return final_balance
        else:
            lg.error(f"Parsed balance for {currency} ('{available_balance_str}' from field '{field_used}') is invalid (negative or non-finite).")
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Failed to convert balance string '{available_balance_str}' (from field '{field_used}') to Decimal for {currency}: {e}.")
        return None
    except Exception as e:
        lg.error(f"Unexpected error parsing balance response structure for {currency}: {e}", exc_info=True)
        return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes market data using pandas_ta to calculate technical indicators and generate
    weighted trading signals based on a configured strategy.

    Manages symbol-specific state (like break-even status) via a shared dictionary.
    Uses Decimal for internal calculations involving prices and quantities where precision is key,
    but converts data to float for pandas_ta compatibility.
    """
    def __init__(
        self,
        df_raw: pd.DataFrame,  # DataFrame with Decimal OHLCV columns from fetch_klines_ccxt
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],
        symbol_state: dict[str, Any],  # Mutable state dict shared with main loop
    ) -> None:
        """Initializes the TradingAnalyzer.

        Args:
            df_raw: DataFrame containing OHLCV data (must have Decimal columns).
            logger: Logger instance for this symbol.
            config: The main configuration dictionary.
            market_info: Processed market information for the symbol.
            symbol_state: Mutable dictionary holding state for this symbol
                          (e.g., 'break_even_triggered', 'last_entry_price').

        Raises:
            ValueError: If df_raw, market_info, or symbol_state is invalid or missing.
        """
        # --- Input Validation ---
        if df_raw is None or df_raw.empty:
            raise ValueError("TradingAnalyzer requires a non-empty raw DataFrame.")
        if not market_info or not isinstance(market_info, dict):
            raise ValueError("TradingAnalyzer requires a valid market_info dictionary.")
        if symbol_state is None or not isinstance(symbol_state, dict):
            raise ValueError("TradingAnalyzer requires a valid symbol_state dictionary.")

        self.df_raw = df_raw  # Keep raw Decimal DF for precise checks (e.g., Fib, price checks)
        self.df = df_raw.copy()  # Work on a copy for TA calculations requiring floats
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol: str = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval: str = config.get("interval", "UNKNOWN_INTERVAL")
        self.symbol_state = symbol_state  # Store reference to the mutable state dict

        # --- Internal State Initialization ---
        self.indicator_values: dict[str, Decimal | None] = {}  # Stores latest indicator values as Decimals
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Current signal state flags
        self.active_weight_set_name: str = config.get("active_weight_set", "default")
        self.weights: dict[str, float] = {}  # Loaded weights (converted to float)
        self.fib_levels_data: dict[str, Decimal] = {}  # Stores calculated Fibonacci levels
        self.ta_strategy: ta.Strategy | None = None  # pandas_ta strategy object
        self.ta_column_map: dict[str, str] = {}  # Maps generic names (e.g., "EMA_Short") to pandas_ta column names (e.g., "EMA_9")

        # Load and validate weights for the active set
        raw_weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        if not raw_weights or not isinstance(raw_weights, dict):
            logger.warning(f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' is empty or not found for {self.symbol}. Signal generation will be disabled.{RESET}")
        else:
            # Convert weights to float during initialization
            for key, val in raw_weights.items():
                try:
                    self.weights[key] = float(val)
                except (ValueError, TypeError):
                     logger.warning(f"Invalid weight '{val}' for '{key}' in set '{self.active_weight_set_name}'. Skipping.")

        # --- Data Preparation and Indicator Calculation ---
        # Convert necessary columns in the copied DataFrame (self.df) to float for pandas_ta
        self._convert_df_for_ta()

        # Initialize and Calculate Indicators if data is valid
        if not self.df.empty:
            self._define_ta_strategy()
            if self.ta_strategy:
                self._calculate_all_indicators()  # Operates on self.df (float version)
                self._update_latest_indicator_values()  # Populates self.indicator_values with Decimals from self.df
            else:
                 logger.warning(f"TA Strategy not defined for {self.symbol}, skipping calculations.")
            # Calculate initial Fib levels using self.df_raw (Decimal version)
            self.calculate_fibonacci_levels()
        else:
            logger.warning(f"DataFrame is empty after float conversion for {self.symbol}. Cannot calculate indicators.")

    def _convert_df_for_ta(self) -> None:
        """Converts necessary DataFrame columns (OHLCV) in the working copy `self.df`
        to float64 for compatibility with pandas_ta. Handles potential errors gracefully.
        """
        if self.df.empty: return
        try:
            cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
            self.logger.debug(f"Converting columns {cols_to_convert} to float64 for TA...")
            for col in cols_to_convert:
                 if col in self.df.columns:
                    # Check if conversion is needed (not already float64)
                    if not pd.api.types.is_float_dtype(self.df[col]) or self.df[col].dtype != np.float64:
                        original_type = self.df[col].dtype
                        # Convert to float64, coercing errors to NaN.
                        # Using pd.to_numeric is robust for various input types (Decimal, str, int).
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(np.float64)
                        if self.df[col].isnull().any():
                            self.logger.debug(f"Column '{col}' (original: {original_type}) converted to float64, NaNs introduced.")
                        # else:
                        #     self.logger.debug(f"Column '{col}' (original: {original_type}) converted to float64.")
                    # else: # Already float64
                    #     self.logger.debug(f"Column '{col}' is already float64.")

            # Verify conversion and log dtypes
            converted_dtypes = {col: self.df[col].dtype for col in cols_to_convert if col in self.df.columns}
            self.logger.debug(f"DataFrame dtypes prepared for TA: {converted_dtypes}")
            # Check if any critical conversion failed resulting in non-float types
            for col, dtype in converted_dtypes.items():
                if not pd.api.types.is_float_dtype(dtype):
                     self.logger.warning(f"Column '{col}' is not float (dtype: {dtype}) after conversion attempt. TA results may be affected.")

        except Exception as e:
             self.logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
             # Set df to empty to prevent further processing with potentially bad data
             self.df = pd.DataFrame()

    # --- State Properties ---
    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even triggered status from the shared symbol state."""
        # Default to False if the key doesn't exist in the state dictionary
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool) -> None:
        """Sets the break-even triggered status in the shared symbol state and logs the change."""
        if not isinstance(value, bool):
            self.logger.error(f"Invalid type for break_even_triggered ({type(value)}). Must be boolean.")
            return
        current_value = self.symbol_state.get('break_even_triggered')
        # Only update and log if the value actually changes
        if current_value != value:
            self.symbol_state['break_even_triggered'] = value
            self.logger.info(f"Break-even status for {self.symbol} set to: {value}")

    def _define_ta_strategy(self) -> None:
        """Defines the pandas_ta Strategy object based on enabled indicators in the config."""
        cfg = self.config
        # Get the dictionary of enabled indicator flags (e.g., {"rsi": True, "macd": False})
        indi_cfg = cfg.get("indicators", {})
        if not isinstance(indi_cfg, dict):
            self.logger.error("Invalid 'indicators' configuration (must be a dictionary). Cannot define TA strategy.")
            return

        # Helper to safely get parameters (already validated as int/float in load_config)
        def get_param(key: str, default: int | float) -> int | float:
            # Config validation ensures these are correct types (int/float)
            val = cfg.get(key, default)
            # Ensure correct type just in case validation missed something (shouldn't happen)
            if isinstance(default, int): return int(val)
            if isinstance(default, float): return float(val)
            return val  # Fallback

        # --- Get parameters for all potential indicators ---
        atr_p = get_param("atr_period", DEFAULT_ATR_PERIOD)
        ema_s = get_param("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
        ema_l = get_param("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
        rsi_p = get_param("rsi_period", DEFAULT_RSI_WINDOW)
        bb_p = get_param("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        bb_std = get_param("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        cci_w = get_param("cci_window", DEFAULT_CCI_WINDOW)
        wr_w = get_param("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        mfi_w = get_param("mfi_window", DEFAULT_MFI_WINDOW)
        stochrsi_w = get_param("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
        stochrsi_rsi_w = get_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
        stochrsi_k = get_param("stoch_rsi_k", DEFAULT_K_WINDOW)
        stochrsi_d = get_param("stoch_rsi_d", DEFAULT_D_WINDOW)
        psar_af = get_param("psar_af", DEFAULT_PSAR_AF)
        psar_max = get_param("psar_max_af", DEFAULT_PSAR_MAX_AF)
        sma10_w = get_param("sma_10_window", DEFAULT_SMA_10_WINDOW)
        mom_p = get_param("momentum_period", DEFAULT_MOMENTUM_PERIOD)
        vol_ma_p = get_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Build the list of indicators for the pandas_ta Strategy
        ta_list: list[dict[str, Any]] = []
        self.ta_column_map: dict[str, str] = {}  # Reset map

        # --- Add Indicators Based on Config Flags and Valid Parameters ---
        # ATR (Always calculated for risk management, ensure period is valid)
        if atr_p > 0:
            col_name = f"ATRr_{atr_p}"  # pandas_ta default ATR col name
            ta_list.append({"kind": "atr", "length": atr_p, "col_names": (col_name,)})
            self.ta_column_map["ATR"] = col_name
        else:
             self.logger.error(f"ATR period ({atr_p}) is invalid. Risk management calculations will fail.")

        # EMAs (Needed if alignment or MA cross exit enabled)
        ema_needed = indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit")
        if ema_needed:
            if ema_s > 0:
                col_name_s = f"EMA_{ema_s}"
                ta_list.append({"kind": "ema", "length": ema_s, "col_names": (col_name_s,)})
                self.ta_column_map["EMA_Short"] = col_name_s
            else: self.logger.warning("EMA Short period invalid, EMA features disabled.")
            if ema_l > ema_s:  # Validation ensures ema_l > ema_s if both are valid
                col_name_l = f"EMA_{ema_l}"
                ta_list.append({"kind": "ema", "length": ema_l, "col_names": (col_name_l,)})
                self.ta_column_map["EMA_Long"] = col_name_l
            elif ema_l <= ema_s: self.logger.warning(f"EMA Long period ({ema_l}) not > EMA Short ({ema_s}), EMA Long disabled.")

        # Momentum
        if indi_cfg.get("momentum") and mom_p > 0:
            col_name = f"MOM_{mom_p}"
            ta_list.append({"kind": "mom", "length": mom_p, "col_names": (col_name,)})
            self.ta_column_map["Momentum"] = col_name

        # Volume SMA (for Volume Confirmation)
        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0:
            col_name = f"VOL_SMA_{vol_ma_p}"
            # Ensure 'volume' column exists and is float before calculating
            if 'volume' in self.df.columns and pd.api.types.is_float_dtype(self.df['volume']):
                ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_p, "col_names": (col_name,)})
                self.ta_column_map["Volume_MA"] = col_name
            else:
                self.logger.warning("Cannot calculate Volume SMA: 'volume' column missing or not float in TA DataFrame.")
                indi_cfg["volume_confirmation"] = False  # Disable check if cannot calculate

        # Stochastic RSI
        if indi_cfg.get("stoch_rsi") and all(p > 0 for p in [stochrsi_w, stochrsi_rsi_w, stochrsi_k, stochrsi_d]):
            k_col = f"STOCHRSIk_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            d_col = f"STOCHRSId_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            ta_list.append({
                "kind": "stochrsi", "length": stochrsi_w, "rsi_length": stochrsi_rsi_w,
                "k": stochrsi_k, "d": stochrsi_d, "col_names": (k_col, d_col)
            })
            self.ta_column_map["StochRSI_K"] = k_col
            self.ta_column_map["StochRSI_D"] = d_col

        # RSI
        if indi_cfg.get("rsi") and rsi_p > 0:
            col_name = f"RSI_{rsi_p}"
            ta_list.append({"kind": "rsi", "length": rsi_p, "col_names": (col_name,)})
            self.ta_column_map["RSI"] = col_name

        # Bollinger Bands
        if indi_cfg.get("bollinger_bands") and bb_p > 0:
            bb_std_str = f"{bb_std:.1f}".replace('.', '_')  # Format std dev for column name
            bbl = f"BBL_{bb_p}_{bb_std_str}"; bbm = f"BBM_{bb_p}_{bb_std_str}"
            bbu = f"BBU_{bb_p}_{bb_std_str}"; bbb = f"BBB_{bb_p}_{bb_std_str}"
            bbp = f"BBP_{bb_p}_{bb_std_str}"
            ta_list.append({
                "kind": "bbands", "length": bb_p, "std": bb_std,
                "col_names": (bbl, bbm, bbu, bbb, bbp)  # Lower, Middle, Upper, Bandwidth, Percent
            })
            self.ta_column_map["BB_Lower"] = bbl
            self.ta_column_map["BB_Middle"] = bbm
            self.ta_column_map["BB_Upper"] = bbu
            # self.ta_column_map["BB_Bandwidth"] = bbb # Optional, not used in default checks
            # self.ta_column_map["BB_Percent"] = bbp # Optional

        # VWAP (Volume Weighted Average Price)
        if indi_cfg.get("vwap"):
            vwap_col = "VWAP_D"  # Default pandas_ta name for daily anchor
            if all(c in self.df.columns for c in ['high', 'low', 'close', 'volume']):
                 # VWAP calculation needs HLCV columns. pandas_ta handles this.
                 ta_list.append({"kind": "vwap", "anchor": "D", "col_names": (vwap_col,)})  # Use daily anchor
                 self.ta_column_map["VWAP"] = vwap_col
            else:
                 self.logger.warning("Cannot calculate VWAP: Missing HLCV columns in TA DataFrame.")
                 indi_cfg["vwap"] = False  # Disable check

        # CCI (Commodity Channel Index)
        if indi_cfg.get("cci") and cci_w > 0:
            cci_col = f"CCI_{cci_w}_0.015"  # Default pandas_ta name includes the constant
            ta_list.append({"kind": "cci", "length": cci_w, "col_names": (cci_col,)})
            self.ta_column_map["CCI"] = cci_col

        # Williams %R
        if indi_cfg.get("wr") and wr_w > 0:
            wr_col = f"WILLR_{wr_w}"
            ta_list.append({"kind": "willr", "length": wr_w, "col_names": (wr_col,)})
            self.ta_column_map["WR"] = wr_col

        # Parabolic SAR
        if indi_cfg.get("psar"):
            # Format AF values for column names (remove trailing zeros/dots)
            psar_af_str = f"{psar_af}".rstrip('0').rstrip('.')
            psar_max_str = f"{psar_max}".rstrip('0').rstrip('.')
            l_col = f"PSARl_{psar_af_str}_{psar_max_str}"  # Long signal line
            s_col = f"PSARs_{psar_af_str}_{psar_max_str}"  # Short signal line
            # af_col = f"PSARaf_{psar_af_str}_{psar_max_str}" # Acceleration Factor (optional)
            # r_col = f"PSARr_{psar_af_str}_{psar_max_str}" # Reversal indicator (optional)
            ta_list.append({
                "kind": "psar", "af": psar_af, "max_af": psar_max,
                "col_names": (l_col, s_col)  # Only request L/S lines by default
            })
            self.ta_column_map["PSAR_Long"] = l_col   # Value when SAR is below (long trend), NaN otherwise
            self.ta_column_map["PSAR_Short"] = s_col  # Value when SAR is above (short trend), NaN otherwise

        # SMA 10
        if indi_cfg.get("sma_10") and sma10_w > 0:
            col_name = f"SMA_{sma10_w}"
            ta_list.append({"kind": "sma", "length": sma10_w, "col_names": (col_name,)})
            self.ta_column_map["SMA10"] = col_name

        # MFI (Money Flow Index)
        if indi_cfg.get("mfi") and mfi_w > 0:
            col_name = f"MFI_{mfi_w}"
            if all(c in self.df.columns for c in ['high', 'low', 'close', 'volume']):
                 ta_list.append({"kind": "mfi", "length": mfi_w, "col_names": (col_name,)})
                 self.ta_column_map["MFI"] = col_name
            else:
                 self.logger.warning("Cannot calculate MFI: Missing HLCV columns in TA DataFrame.")
                 indi_cfg["mfi"] = False  # Disable check

        # --- Create Strategy Object ---
        if not ta_list:
            self.logger.warning(f"No valid indicators enabled or configured for {self.symbol}. TA Strategy not created.")
            return

        try:
            self.ta_strategy = ta.Strategy(
                name="EnhancedMultiIndicatorStrategy",
                description="Calculates multiple TA indicators based on bot config using pandas_ta",
                ta=ta_list
            )
            self.logger.info(f"Defined TA Strategy for {self.symbol} with {len(ta_list)} indicator groups.")
            self.logger.debug(f"TA Column Map: {self.ta_column_map}")
        except Exception as strat_err:
            self.logger.error(f"Error creating pandas_ta Strategy object: {strat_err}", exc_info=True)
            self.ta_strategy = None

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using the defined pandas_ta strategy on the float DataFrame (self.df)."""
        if self.df.empty:
            self.logger.warning(f"TA DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return
        if not self.ta_strategy:
            # Already logged warning in _define_ta_strategy if no indicators were added
            return

        # Check if sufficient data exists for the strategy's requirements
        min_required_data = self.ta_strategy.required if hasattr(self.ta_strategy, 'required') else 50  # pandas_ta built-in or default guess
        buffer = 20  # Add buffer for calculation stability
        required_with_buffer = min_required_data + buffer
        if len(self.df) < required_with_buffer:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} rows) for {self.symbol} TA calculation (min recommended: {required_with_buffer}). Results may be inaccurate or NaN.{RESET}")

        try:
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol} on {len(self.df)} rows...")
            # Apply the strategy to the DataFrame (modifies self.df inplace)
            start_ta_time = time.monotonic()
            # Ensure the DataFrame index is datetime (should be from fetch_klines_ccxt)
            if not isinstance(self.df.index, pd.DatetimeIndex):
                 self.logger.warning("DataFrame index is not DatetimeIndex. TA calculations might fail or be incorrect.")
            # Run the strategy calculation
            self.df.ta.strategy(self.ta_strategy, timed=False)  # timed=True adds overhead
            ta_duration = time.monotonic() - start_ta_time
            self.logger.debug(f"Finished indicator calculations for {self.symbol} in {ta_duration:.3f}s.")
            # Optional: Log columns generated: self.logger.debug(f"DataFrame columns after TA: {self.df.columns.tolist()}")
        except AttributeError as ae:
             # Catch common errors if DataFrame columns are not float or contain NaNs unexpectedly
             if "'float' object has no attribute" in str(ae) or "cannot convert" in str(ae).lower():
                 self.logger.error(f"{NEON_RED}Pandas TA Error ({self.symbol}): Input columns (e.g., HLCV) might contain NaNs or non-numeric data. Check data fetching and float conversion. Error: {ae}{RESET}", exc_info=False)
                 self.logger.debug(f"Problematic DF sample (float converted) tail:\n{self.df.tail()}")
             else:
                 self.logger.error(f"{NEON_RED}Pandas TA attribute error ({self.symbol}): {ae}. Is pandas_ta installed/updated? Check library compatibility.{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta strategy for {self.symbol}: {e}{RESET}", exc_info=True)
            # Don't wipe df here, some indicators might have calculated, but results will be incomplete

    def _update_latest_indicator_values(self) -> None:
        """Updates self.indicator_values (dict of Decimals) with the latest calculated
        values from the float DataFrame (self.df), converting them back to Decimal.
        Handles potential NaN/None values gracefully.
        """
        self.indicator_values = {}  # Reset before populating
        if self.df.empty:
            self.logger.warning(f"TA DataFrame empty for {self.symbol}. Cannot update latest indicator values.")
            return
        try:
            # Get the last row (most recent indicator values)
            latest_series = self.df.iloc[-1]
            if latest_series.isnull().all():
                self.logger.warning(f"Last row of TA DataFrame is all NaN for {self.symbol}. Cannot update latest indicator values.")
                return
        except IndexError:
            self.logger.error(f"DataFrame index out of bounds when accessing last row for {self.symbol}.")
            return
        except Exception as e:
             self.logger.error(f"Error accessing last row of DataFrame for {self.symbol}: {e}", exc_info=True)
             return

        # Helper to safely convert float/object back to Decimal
        def to_decimal_safe(value: Any) -> Decimal | None:
            # Check for pandas NA, numpy NaN, or None
            if pd.isna(value) or value is None: return None
            try:
                # Check for infinity before converting float to string
                if isinstance(value, (float, np.floating)) and not math.isfinite(value):
                    return None
                # Convert to string first for precise Decimal conversion
                dec_val = Decimal(str(value))
                # Final check for non-finite Decimals (just in case)
                return dec_val if dec_val.is_finite() else None
            except (InvalidOperation, ValueError, TypeError):
                # self.logger.debug(f"Could not convert value '{value}' (type: {type(value)}) to Decimal.")
                return None

        # Populate indicator_values using the ta_column_map to get actual column names
        for generic_name, actual_col_name in self.ta_column_map.items():
            if actual_col_name in latest_series:
                raw_value = latest_series.get(actual_col_name)
                self.indicator_values[generic_name] = to_decimal_safe(raw_value)
            else:
                 # This might happen if an indicator calculation failed silently in pandas_ta
                 self.logger.debug(f"Column '{actual_col_name}' not found in TA DataFrame for indicator '{generic_name}' ({self.symbol}). Setting value to None.")
                 self.indicator_values[generic_name] = None

        # Also add latest OHLCV values (from the float df, converted back to Decimal)
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
             if base_col in latest_series:
                 self.indicator_values[base_col.capitalize()] = to_decimal_safe(latest_series.get(base_col))

        # Log summary of updated values
        valid_values_count = sum(1 for v in self.indicator_values.values() if v is not None)
        total_expected = len(self.ta_column_map) + 5  # +5 for OHLCV
        self.logger.debug(f"Latest indicator Decimal values updated for {self.symbol}: Found {valid_values_count}/{total_expected} valid values.")
        # For detailed logging of values:
        # valid_values_str = {k: f"{v:.5f}" if isinstance(v, Decimal) else str(v) for k, v in self.indicator_values.items()}
        # self.logger.debug(f"Indicator Values: {valid_values_str}")

    # --- Precision and Market Info Helpers ---
    def get_min_tick_size(self) -> Decimal | None:
        """Gets the minimum price tick size as a Decimal from market info."""
        tick = self.market_info.get('min_tick_size')
        # Validate that it's a finite, positive Decimal
        if tick is None or not isinstance(tick, Decimal) or not tick.is_finite() or tick <= 0:
            self.logger.warning(f"Invalid or missing min_tick_size ({tick}) for {self.symbol}. Price quantization may fail.")
            return None
        return tick

    def get_price_precision_digits(self) -> int:
        """Gets the number of decimal places for price precision."""
        # Defaults to 8 if not found, a safe fallback (won't truncate too much)
        return self.market_info.get('price_precision_digits', 8)

    def get_amount_precision_digits(self) -> int:
        """Gets the number of decimal places for amount (quantity) precision."""
        return self.market_info.get('amount_precision_digits', 8)

    def get_amount_step_size(self) -> Decimal | None:
        """Gets the minimum amount step size as a Decimal from market info."""
        step = self.market_info.get('amount_step_size')
        # Validate that it's a finite, positive Decimal
        if step is None or not isinstance(step, Decimal) or not step.is_finite() or step <= 0:
            self.logger.warning(f"Invalid or missing amount_step_size ({step}) for {self.symbol}. Amount quantization may use digits fallback.")
            return None
        return step

    def quantize_price(self, price: Decimal | float | str, rounding: str = ROUND_DOWN) -> Decimal | None:
        """Quantizes a price to the market's minimum tick size using specified rounding.

        Args:
            price: The price value to quantize.
            rounding: The rounding mode (e.g., decimal.ROUND_DOWN, decimal.ROUND_UP).

        Returns:
            The quantized price as Decimal, or None if quantization fails.
        """
        min_tick = self.get_min_tick_size()
        if min_tick is None: return None  # Cannot quantize without tick size
        try:
            # Convert input to Decimal using string representation for precision
            price_decimal = Decimal(str(price))
            if not price_decimal.is_finite():
                self.logger.warning(f"Cannot quantize non-finite price: {price}")
                return None
            # Use Decimal's quantize method with the tick size as the exponent template
            quantized = price_decimal.quantize(min_tick, rounding=rounding)
            return quantized
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing price '{price}' with tick '{min_tick}' for {self.symbol}: {e}")
            return None

    def quantize_amount(self, amount: Decimal | float | str, rounding: str = ROUND_DOWN) -> Decimal | None:
        """Quantizes an amount (quantity) to the market's amount step size, falling back
        to precision digits if step size is unavailable.

        Args:
            amount: The amount value to quantize.
            rounding: The rounding mode (e.g., decimal.ROUND_DOWN, decimal.ROUND_UP).

        Returns:
            The quantized amount as Decimal, or None if quantization fails.
        """
        step_size = self.get_amount_step_size()

        # Fallback to using precision digits if step size is missing/invalid
        if step_size is None:
            amount_digits = self.get_amount_precision_digits()
            # Calculate step size from digits (e.g., 3 digits -> step 0.001)
            step_size = Decimal('1') / (Decimal('10') ** amount_digits)
            self.logger.debug(f"Using amount precision digits ({amount_digits}) for step size ({step_size}) for {self.symbol}")

        try:
            # Convert input to Decimal using string representation
            amount_decimal = Decimal(str(amount))
            if not amount_decimal.is_finite():
                self.logger.warning(f"Cannot quantize non-finite amount: {amount}")
                return None

            # Formula: floor/ceil(amount / step_size) * step_size
            # Use Decimal's quantize for precision
            # Calculate how many steps fit into the amount
            quantized_steps = (amount_decimal / step_size).quantize(Decimal('1'), rounding=rounding)
            # Multiply steps back by step size
            quantized_amount = quantized_steps * step_size

            # Re-quantize the result to the exact precision of the step size
            # This handles cases where step_size has more digits than the intermediate result
            # (e.g., step=0.0025). Use ROUND_DOWN for final safety.
            final_quantized = quantized_amount.quantize(step_size, rounding=ROUND_DOWN)
            return final_quantized
        except (InvalidOperation, ValueError, TypeError, DivisionByZeroError) as e:
            self.logger.error(f"Error quantizing amount '{amount}' with step '{step_size}' for {self.symbol}: {e}")
            return None

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels based on the high/low over a specified window.
        Uses the raw DataFrame (`df_raw` with Decimal precision) and quantizes the resulting levels.

        Args:
            window: The lookback period (number of candles). Uses config value if None.

        Returns:
            A dictionary of Fibonacci levels (e.g., "Fib_38.2%") mapped to quantized price Decimals.
            Returns an empty dictionary if calculation is not possible or fails.
        """
        self.fib_levels_data = {}  # Reset previous levels
        window = window or int(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))

        if len(self.df_raw) < window:
            self.logger.debug(f"Not enough data ({len(self.df_raw)} rows) for Fibonacci window ({window}) on {self.symbol}.")
            return {}

        # Use the raw DataFrame (df_raw) which should have Decimal columns
        df_slice = self.df_raw.tail(window)

        try:
            if 'high' not in df_slice.columns or 'low' not in df_slice.columns:
                 self.logger.warning(f"Missing 'high' or 'low' column in df_raw for Fib calculation ({self.symbol}).")
                 return {}

            # Extract High/Low Series (should be Decimal type)
            high_series = df_slice["high"]
            low_series = df_slice["low"]

            # Find max high and min low, handling potential NaNs or non-Decimals gracefully
            # Convert to numeric first, coercing errors, then find max/min, then convert result back to Decimal
            high_numeric = pd.to_numeric(high_series, errors='coerce').dropna()
            low_numeric = pd.to_numeric(low_series, errors='coerce').dropna()

            if high_numeric.empty or low_numeric.empty:
                 self.logger.warning(f"Could not find valid high/low data in the last {window} periods for Fib calculation on {self.symbol} after cleaning.")
                 return {}

            # Convert max/min back to Decimal using string representation for precision
            high = Decimal(str(high_numeric.max()))
            low = Decimal(str(low_numeric.min()))

            if not high.is_finite() or not low.is_finite():
                 self.logger.warning(f"Non-finite high/low values after aggregation for Fib calculation on {self.symbol}. High={high}, Low={low}")
                 return {}
            if high <= low:  # Range must be positive
                 self.logger.debug(f"Invalid range (High <= Low): High={high}, Low={low} in window {window} for Fib calculation on {self.symbol}.")
                 return {}

            diff: Decimal = high - low
            levels: dict[str, Decimal] = {}
            min_tick: Decimal | None = self.get_min_tick_size()

            if diff > 0:
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float))
                    # Calculate raw price level: High - (Range * Percentage)
                    level_price_raw = high - (diff * level_pct)

                    # Quantize the calculated level price if tick size is available
                    if min_tick is not None:
                        # Round down for potential support/resistance levels
                        level_price_quantized = self.quantize_price(level_price_raw, rounding=ROUND_DOWN)
                        if level_price_quantized is not None:
                            level_name = f"Fib_{level_pct * 100:.1f}%"
                            levels[level_name] = level_price_quantized
                        else:
                            self.logger.warning(f"Failed to quantize Fibonacci level {level_pct * 100:.1f}% (Raw: {level_price_raw}) for {self.symbol}")
                    else:
                        # Store raw Decimal if quantization isn't possible
                        self.logger.warning(f"Storing raw (non-quantized) Fibonacci level {level_pct * 100:.1f}% for {self.symbol} due to missing min_tick_size.")
                        levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_raw
            else:  # diff <= 0 case already handled
                 pass

            self.fib_levels_data = levels
            # Log the calculated levels (optional, can be verbose)
            if levels:
                price_prec = self.get_price_precision_digits()
                log_levels = {k: f"{v:.{price_prec}f}" for k, v in levels.items()}
                self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} (Window: {window}, High: {high:.{price_prec}f}, Low: {low:.{price_prec}f}): {log_levels}")
            return levels

        except Exception as e:
            # Catch potential errors during series access, conversion, or calculation
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    # --- Indicator Check Methods ---
    # These methods return Optional[float] score [-1.0, 1.0] or None if unavailable/error.
    # They use the Decimal values stored in self.indicator_values and convert to float for scoring.

    def _get_indicator_float(self, name: str) -> float | None:
        """Safely retrieves an indicator value from self.indicator_values as a float."""
        val_decimal = self.indicator_values.get(name)
        if val_decimal is None or not val_decimal.is_finite(): return None
        try: return float(val_decimal)
        except (ValueError, TypeError): return None

    def _check_ema_alignment(self) -> float | None:
        """Checks if short EMA is above/below long EMA. Score: 1.0 (short > long), -1.0 (short < long), 0.0 (equal/NaN)."""
        ema_s = self._get_indicator_float("EMA_Short")
        ema_l = self._get_indicator_float("EMA_Long")
        if ema_s is None or ema_l is None: return None
        # Add tiny tolerance to avoid score flapping when EMAs are extremely close
        tolerance = 1e-9
        if ema_s > ema_l + tolerance: return 1.0  # Bullish alignment
        if ema_s < ema_l - tolerance: return -1.0  # Bearish alignment
        return 0.0  # Effectively equal or NaN

    def _check_momentum(self) -> float | None:
        """Checks Momentum indicator value. Positive -> bullish, Negative -> bearish. Score scaled & clamped."""
        mom = self._get_indicator_float("Momentum")
        if mom is None: return None
        # Basic scaling attempt: Assumes momentum might need normalization.
        # This factor likely needs tuning based on asset/timeframe volatility.
        # Goal: Map typical positive/negative ranges towards +1/-1.
        scaling_factor = 0.1  # Example: If MOM often +/- 10, this scales it to +/- 1.
        score = mom * scaling_factor
        return max(-1.0, min(1.0, score))  # Clamp score to [-1.0, 1.0]

    def _check_volume_confirmation(self) -> float | None:
        """Checks if current volume exceeds its MA by a multiplier. Score: 0.7 (exceeds), 0.0 (doesn't/NaN)."""
        vol = self._get_indicator_float("Volume")
        vol_ma = self._get_indicator_float("Volume_MA")
        # Need valid volume, and vol_ma > 0 for meaningful comparison
        if vol is None or vol_ma is None or vol_ma <= 0: return None
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        return 0.7 if vol > vol_ma * multiplier else 0.0

    def _check_stoch_rsi(self) -> float | None:
        """Checks Stochastic RSI K and D lines for overbought/oversold and crossovers. Score range [-1.0, 1.0]."""
        k = self._get_indicator_float("StochRSI_K")
        d = self._get_indicator_float("StochRSI_D")
        if k is None or d is None: return None
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        score = 0.0
        # Strong signals: Crossover within OB/OS zones
        if k < oversold and d < oversold: score = 1.0 if k > d else 0.8  # Bullish crossover deep OS (stronger if k>d crosses up)
        elif k > overbought and d > overbought: score = -1.0 if k < d else -0.8  # Bearish crossover deep OB (stronger if k<d crosses down)
        # Weaker signals: Just entering OB/OS without crossover confirmation yet
        elif k < oversold and d >= oversold: score = 0.6  # K entered OS
        elif k > overbought and d <= overbought: score = -0.6  # K entered OB
        # Mid-range crossover signals (weakest confirmation)
        elif k > d and not (k > overbought and d > overbought): score = 0.3  # Bullish crossover mid-range
        elif k < d and not (k < oversold and d < oversold): score = -0.3  # Bearish crossover mid-range
        # Else (k == d mid-range, or both OB/OS but no cross): score remains 0.0
        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_rsi(self) -> float | None:
        """Checks RSI value against OB/OS levels. Score scaled linearly between 0 and 100, clamped."""
        rsi = self._get_indicator_float("RSI")
        if rsi is None: return None
        # Simple linear scale: Score = 1.0 at RSI=0, 0.0 at RSI=50, -1.0 at RSI=100
        # This treats RSI linearly, where lower values are more bullish.
        score = (50.0 - rsi) / 50.0
        # Optional: Add extra weight/boost for extreme readings (e.g., beyond 80/20)
        extreme_ob = 80.0; extreme_os = 20.0
        if rsi >= extreme_ob: score = max(-1.0, score * 1.1)  # Increase bearishness magnitude above extreme OB
        if rsi <= extreme_os: score = min(1.0, score * 1.1)  # Increase bullishness magnitude below extreme OS
        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_cci(self) -> float | None:
        """Checks CCI against standard OB/OS levels (+/- 100). Score scaled linearly between thresholds, clamped."""
        cci = self._get_indicator_float("CCI")
        if cci is None: return None
        ob = 100.0; os = -100.0
        # Scale based on thresholds: Strong signal outside +/-100, linear inside.
        if cci >= ob: score = -1.0  # Strong sell signal above +100
        elif cci <= os: score = 1.0  # Strong buy signal below -100
        else: score = -cci / ob  # Linear scale between -100 and +100 (e.g., cci=50 -> -0.5, cci=-50 -> 0.5)
        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_wr(self) -> float | None:
        """Checks Williams %R against standard OB/OS levels (-20 / -80). Score scaled linearly, clamped."""
        wr = self._get_indicator_float("WR")  # Williams %R typically ranges from -100 (most oversold) to 0 (most overbought)
        if wr is None: return None
        # Scale: Score = 1.0 at WR=-100 (extreme oversold), 0.0 at WR=-50, -1.0 at WR=0 (extreme overbought)
        score = (wr + 50.0) / -50.0
        # Optional: Add extra weight/boost for extreme readings
        extreme_ob = -10.0; extreme_os = -90.0
        if wr >= extreme_ob: score = max(-1.0, score * 1.1)  # Increase bearishness magnitude near 0
        if wr <= extreme_os: score = min(1.0, score * 1.1)  # Increase bullishness magnitude near -100
        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_psar(self) -> float | None:
        """Checks Parabolic SAR position relative to price. Score: 1.0 (SAR below price - bullish), -1.0 (SAR above - bearish), 0.0 (transition/error)."""
        # PSARl (long) column has value when SAR is below price, NaN otherwise.
        # PSARs (short) column has value when SAR is above price, NaN otherwise.
        psar_l_val = self.indicator_values.get("PSAR_Long")  # Decimal or None
        psar_s_val = self.indicator_values.get("PSAR_Short")  # Decimal or None

        # Check if the values are finite Decimals (i.e., not None or NaN from calculation)
        psar_l_active = psar_l_val is not None and psar_l_val.is_finite()
        psar_s_active = psar_s_val is not None and psar_s_val.is_finite()

        if psar_l_active and not psar_s_active: return 1.0  # SAR is below price (Long trend active)
        if psar_s_active and not psar_l_active: return -1.0  # SAR is above price (Short trend active)

        # If both active/inactive (shouldn't happen with standard PSAR) or both NaN (e.g., at start), return neutral.
        if (psar_l_active and psar_s_active) or (not psar_l_active and not psar_s_active):
            self.logger.debug(f"PSAR state ambiguous/neutral for {self.symbol}: PSARl={psar_l_val}, PSARs={psar_s_val}")
            return 0.0
        # Fallback neutral case (should be covered above)
        return 0.0

    def _check_sma10(self) -> float | None:
        """Checks if close price is above/below SMA10. Score: 0.5 (above), -0.5 (below), 0.0 (equal/NaN)."""
        sma = self._get_indicator_float("SMA10")
        close = self._get_indicator_float("Close")
        if sma is None or close is None: return None
        tolerance = 1e-9
        if close > sma + tolerance: return 0.5
        if close < sma - tolerance: return -0.5
        return 0.0  # Effectively equal or NaN

    def _check_vwap(self) -> float | None:
        """Checks if close price is above/below VWAP. Score: 0.6 (above), -0.6 (below), 0.0 (equal/NaN)."""
        vwap = self._get_indicator_float("VWAP")
        close = self._get_indicator_float("Close")
        if vwap is None or close is None: return None
        tolerance = 1e-9
        if close > vwap + tolerance: return 0.6
        if close < vwap - tolerance: return -0.6
        return 0.0  # Effectively equal or NaN

    def _check_mfi(self) -> float | None:
        """Checks Money Flow Index against standard OB/OS levels (80/20). Score scaled linearly, clamped."""
        mfi = self._get_indicator_float("MFI")
        if mfi is None: return None
        # Scale similar to RSI: Score = 1.0 at MFI=0, 0.0 at MFI=50, -1.0 at MFI=100
        score = (50.0 - mfi) / 50.0
        # Optional boost for extremes
        extreme_ob = 90.0; extreme_os = 10.0
        if mfi >= extreme_ob: score = max(-1.0, score * 1.1)  # Boost bearishness magnitude
        if mfi <= extreme_os: score = min(1.0, score * 1.1)  # Boost bullishness magnitude
        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_bollinger_bands(self) -> float | None:
        """Checks close price relative to Bollinger Bands. Score: 1.0 (below lower), -1.0 (above upper), scaled linearly within bands."""
        bbl = self._get_indicator_float("BB_Lower")
        bbu = self._get_indicator_float("BB_Upper")
        close = self._get_indicator_float("Close")
        # Need valid bands (upper > lower) and close price
        if bbl is None or bbu is None or close is None or bbu <= bbl: return None
        # Strong signals outside or touching the bands
        if close <= bbl: return 1.0   # Below or touching lower band (strong buy / reversal potential)
        if close >= bbu: return -1.0  # Above or touching upper band (strong sell / reversal potential)
        # Scale position linearly within the bands
        # Position = (Close - Lower) / (Upper - Lower). Ranges from 0 (at lower) to 1 (at upper).
        band_width = bbu - bbl
        # Avoid division by zero if bands are somehow equal despite earlier check
        if band_width == 0: return 0.0
        position_within_band = (close - bbl) / band_width
        # Convert position [0, 1] to score [+1, -1]
        # Score = 1 - 2 * position. (If position=0, score=1. If position=1, score=-1. If position=0.5, score=0)
        score = 1.0 - (2.0 * position_within_band)
        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_orderbook(self, orderbook_data: dict | None) -> float | None:
        """Calculates Order Book Imbalance (OBI) from fetched order book data.
        OBI = (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)
        Uses top N levels as defined in config. Score: [-1.0, 1.0].

        Args:
            orderbook_data: Dictionary from fetch_orderbook_ccxt containing 'bids' and 'asks'.

        Returns:
            OBI score as float, or None if calculation fails or data is invalid.
        """
        if not orderbook_data or not isinstance(orderbook_data, dict):
             self.logger.debug("Orderbook data missing or invalid type for OBI calculation.")
             return None  # Cannot calculate without data
        try:
            limit = int(self.config.get("orderbook_limit", 10))  # Use configured limit
            # Orderbook data from CCXT usually has string prices/amounts [[price_str, amount_str], ...]
            bids_raw = orderbook_data.get('bids', [])
            asks_raw = orderbook_data.get('asks', [])

            if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
                 self.logger.warning("Orderbook bids/asks are not lists.")
                 return 0.0  # Neutral score if structure is wrong

            # Take top N levels
            top_bids = bids_raw[:limit]
            top_asks = asks_raw[:limit]
            if not top_bids and not top_asks:  # Check if both are empty after slicing
                 self.logger.debug(f"Orderbook empty or insufficient levels (Limit: {limit}) for OBI calc.")
                 return 0.0  # Neutral if no levels available

            # Sum volume (amount) at each level, converting to Decimal robustly
            bid_vol = Decimal('0')
            for b in top_bids:
                # Ensure level format is correct [price, amount]
                if isinstance(b, (list, tuple)) and len(b) > 1:
                    try: bid_vol += Decimal(str(b[1]))  # Use amount (index 1)
                    except (InvalidOperation, TypeError, IndexError, ValueError):
                        self.logger.debug(f"Skipping invalid bid level in OBI calc: {b}")
                        continue  # Skip invalid entries
            ask_vol = Decimal('0')
            for a in top_asks:
                 if isinstance(a, (list, tuple)) and len(a) > 1:
                    try: ask_vol += Decimal(str(a[1]))  # Use amount (index 1)
                    except (InvalidOperation, TypeError, IndexError, ValueError):
                        self.logger.debug(f"Skipping invalid ask level in OBI calc: {a}")
                        continue  # Skip invalid entries

            total_vol = bid_vol + ask_vol
            if total_vol <= 0:
                 self.logger.debug("Total volume in top orderbook levels is zero.")
                 return 0.0  # Avoid division by zero, return neutral

            # Calculate OBI = (BidVol - AskVol) / TotalVol
            obi_decimal = (bid_vol - ask_vol) / total_vol  # Decimal result

            # Clamp to [-1.0, 1.0] and convert to float for the scoring system
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi_decimal)))
            self.logger.debug(f"OBI Calc ({self.symbol}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, OBI={obi_decimal:.4f} -> Score={score:.3f}")
            return score

        except (InvalidOperation, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error calculating Order Book Imbalance for {self.symbol}: {e}")
            return None  # Indicate error / unavailable score
        except Exception as e:
             self.logger.error(f"Unexpected error in OBI calculation for {self.symbol}: {e}", exc_info=True)
             return None

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: dict | None) -> str:
        """Generates a final trading signal ('BUY', 'SELL', 'HOLD') based on weighted scores
        from enabled indicators. Uses Decimal for score accumulation for precision.

        Args:
            current_price_dec: Current market price (Decimal) for logging.
            orderbook_data: Optional order book data for OBI calculation.

        Returns:
            The final signal string: "BUY", "SELL", or "HOLD".
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Reset signals, default HOLD
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores: dict[str, str] = {}  # For detailed logging

        # --- Pre-checks ---
        if not self.indicator_values:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Indicator values not calculated/updated.")
            return "HOLD"
        if not current_price_dec or not current_price_dec.is_finite() or current_price_dec <= 0:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price_dec}).")
            return "HOLD"
        if not self.weights:  # Check if weights dict is empty
            self.logger.debug(f"No weights loaded for active set '{self.active_weight_set_name}'. Holding.")
            return "HOLD"

        # --- Map Indicator Keys to Check Methods ---
        indicator_check_methods = {
            "ema_alignment": self._check_ema_alignment, "momentum": self._check_momentum,
            "volume_confirmation": self._check_volume_confirmation, "stoch_rsi": self._check_stoch_rsi,
            "rsi": self._check_rsi, "bollinger_bands": self._check_bollinger_bands,
            "vwap": self._check_vwap, "cci": self._check_cci, "wr": self._check_wr,
            "psar": self._check_psar, "sma_10": self._check_sma10, "mfi": self._check_mfi,
            # Use lambda for orderbook to pass the data argument
            "orderbook": lambda: self._check_orderbook(orderbook_data),
        }

        # --- Iterate Through Enabled Indicators and Calculate Weighted Score ---
        enabled_indicators_config = self.config.get("indicators", {})

        for indicator_key, check_method in indicator_check_methods.items():
            # Check if indicator is enabled in config
            if not enabled_indicators_config.get(indicator_key, False):
                debug_scores[indicator_key] = "Disabled"
                continue

            # Get the pre-validated float weight for this indicator
            weight = self.weights.get(indicator_key)
            if weight is None:  # Weight missing or was invalid during init
                debug_scores[indicator_key] = "No/Invalid Wt"
                continue

            # Skip if weight is zero
            if math.isclose(weight, 0.0):
                debug_scores[indicator_key] = "Wt=0"
                continue

            # Execute the check method to get the score [-1.0, 1.0] or None
            indicator_score_float: float | None = None
            try:
                indicator_score_float = check_method()
            except Exception as e:
                self.logger.error(f"Error executing check method for '{indicator_key}': {e}", exc_info=True)
                debug_scores[indicator_key] = "Check Error"
                nan_indicator_count += 1
                continue  # Skip this indicator if check failed

            # Process the returned score
            if indicator_score_float is not None and math.isfinite(indicator_score_float):
                try:
                    # Clamp score to [-1.0, 1.0] before converting to Decimal
                    clamped_score_float = max(-1.0, min(1.0, indicator_score_float))
                    indicator_score_decimal = Decimal(str(clamped_score_float))
                    weight_decimal = Decimal(str(weight))  # Convert float weight to Decimal

                    # Calculate weighted score and add to total
                    weighted_score = indicator_score_decimal * weight_decimal
                    final_signal_score += weighted_score
                    total_weight_applied += abs(weight_decimal)  # Sum absolute weights for normalization/debug
                    active_indicator_count += 1
                    # Store score details for debug logging
                    debug_scores[indicator_key] = f"{indicator_score_float:.2f} (x{weight:.2f}) = {weighted_score:.3f}"
                except (InvalidOperation, TypeError) as calc_err:
                    self.logger.error(f"Error processing score/weight for {indicator_key}: {calc_err}")
                    debug_scores[indicator_key] = "Calc Error"
                    nan_indicator_count += 1
            else:
                # Score is None or NaN/Infinite from indicator check
                debug_scores[indicator_key] = "NaN/None"
                nan_indicator_count += 1

        # --- Determine Final Signal Based on Score and Threshold ---
        final_signal = "HOLD"
        normalized_score = Decimal("0.0")
        if total_weight_applied > 0:
            # Normalize the score based on the sum of absolute weights applied
            # This gives a score roughly in [-1, 1] range if individual scores are [-1, 1]
            normalized_score = (final_signal_score / total_weight_applied).quantize(Decimal("0.0001"))
        elif active_indicator_count > 0:
             self.logger.warning(f"Calculated signal score is {final_signal_score} but total weight applied is zero for {self.symbol}. Check weights.")

        # Use specific threshold if active set is 'scalping', otherwise default
        # Thresholds are applied to the RAW final_signal_score (sum of weighted scores)
        is_scalping_set = self.active_weight_set_name == "scalping"
        threshold_key = "scalping_signal_threshold" if is_scalping_set else "signal_score_threshold"
        default_threshold = 2.5 if is_scalping_set else 1.5
        try:
            # Config validation ensures threshold is float
            threshold_float = self.config.get(threshold_key, default_threshold)
            threshold = Decimal(str(threshold_float))
            if not threshold.is_finite() or threshold <= 0: raise ValueError("Threshold must be positive")
        except (ValueError, InvalidOperation, TypeError):
            threshold = Decimal(str(default_threshold))
            self.logger.warning(f"Invalid threshold for '{threshold_key}'. Using default: {threshold}")

        # Compare RAW weighted score against threshold
        if final_signal_score >= threshold: final_signal = "BUY"
        elif final_signal_score <= -threshold: final_signal = "SELL"

        # Log the signal calculation summary
        price_prec = self.get_price_precision_digits()
        signal_color = NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW
        log_msg = (
            f"Signal Calc ({self.symbol} @ {current_price_dec:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Indis(Actv/NaN): {active_indicator_count}/{nan_indicator_count}, "
            f"WtSum: {total_weight_applied:.3f}, RawScore: {final_signal_score:.4f}, NormScore: {normalized_score:.4f}, "
            f"Thresh: +/-{threshold:.3f} -> Signal: {signal_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        if self.logger.level <= logging.DEBUG:  # Log details only if debugging
             # Filter out non-contributing indicators for cleaner debug log
             contributing_scores = {k: v for k, v in debug_scores.items() if v not in ["Disabled", "No/Invalid Wt", "Wt=0"]}
             score_details_str = ", ".join([f"{k}: {v}" for k, v in contributing_scores.items()])
             self.logger.debug(f"  Contributing Scores: {score_details_str}")

        # Update internal signal state flags (used for MA cross check, etc.)
        if final_signal in self.signals:
            self.signals[final_signal] = 1
            self.signals["HOLD"] = 1 if final_signal == "HOLD" else 0
        return final_signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_signal: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates quantized Entry, Take Profit (TP), and Stop Loss (SL) prices based on ATR.
        Uses Decimal for precision and validates results, ensuring SL/TP are a minimum
        number of ticks away from the entry price.

        Args:
            entry_price_signal: The price near which the signal occurred (e.g., current price).
            signal: "BUY" or "SELL".

        Returns:
            Tuple (Quantized Entry Price, Quantized TP Price, Quantized SL Price).
            Returns (None, None, None) if calculation fails (e.g., invalid ATR, price, tick size).
            TP or SL might be None individually if calculation leads to invalid values or cannot be placed safely.
        """
        quantized_entry: Decimal | None = None
        take_profit: Decimal | None = None
        stop_loss: Decimal | None = None

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
             self.logger.error(f"Invalid signal '{signal}' for TP/SL calculation."); return None, None, None
        if not entry_price_signal or not entry_price_signal.is_finite() or entry_price_signal <= 0:
            self.logger.error(f"Invalid entry signal price ({entry_price_signal}) for TP/SL calc."); return None, None, None

        # --- Get ATR and Min Tick ---
        atr_val = self.indicator_values.get("ATR")  # Should be Decimal
        min_tick = self.get_min_tick_size()  # Should be Decimal

        if atr_val is None or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate dynamic TP/SL for {self.symbol}: Invalid ATR value ({atr_val}). SL/TP will be None.{RESET}")
            # We might still be able to quantize entry, but SL/TP depend on ATR.
            quantized_entry = self.quantize_price(entry_price_signal, rounding=ROUND_DOWN if signal == "BUY" else ROUND_UP)
            return quantized_entry, None, None
        if min_tick is None:  # Already logged warning in get_min_tick_size if invalid
             self.logger.error(f"{NEON_RED}Cannot calculate TP/SL for {self.symbol}: Minimum tick size is unavailable.{RESET}")
             return None, None, None  # Cannot quantize without tick size

        # --- Quantize Entry Price ---
        # Use ROUND_DOWN for BUY entry signal, ROUND_UP for SELL entry signal as a reference.
        # Actual fill price for market orders will differ.
        entry_rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
        quantized_entry = self.quantize_price(entry_price_signal, rounding=entry_rounding)
        if quantized_entry is None:
            self.logger.error(f"Failed to quantize entry signal price {entry_price_signal} for {self.symbol}.")
            return None, None, None

        # --- Calculate SL/TP Offsets ---
        try:
            atr = atr_val  # Use the Decimal ATR value
            # Get multipliers from config (validated as float), convert to Decimal
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Calculate raw offsets from entry price
            sl_offset_raw = atr * sl_mult
            tp_offset_raw = atr * tp_mult

            # Ensure offsets are at least the minimum required ticks away in value
            min_offset_value = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)

            sl_offset = max(sl_offset_raw, min_offset_value)
            if sl_offset != sl_offset_raw:
                self.logger.debug(f"Adjusted SL offset from {sl_offset_raw} to minimum {sl_offset} ({MIN_TICKS_AWAY_FOR_SLTP} ticks) for {self.symbol}.")

            tp_offset = max(tp_offset_raw, min_offset_value)
            if tp_offset != tp_offset_raw:
                self.logger.debug(f"Adjusted TP offset from {tp_offset_raw} to minimum {tp_offset} ({MIN_TICKS_AWAY_FOR_SLTP} ticks) for {self.symbol}.")

            # --- Calculate Raw TP/SL Prices ---
            if signal == "BUY":
                sl_raw = quantized_entry - sl_offset
                tp_raw = quantized_entry + tp_offset
                # Quantize SL DOWN (further away), TP UP (further away) initially
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_DOWN)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_UP)
            else:  # SELL
                sl_raw = quantized_entry + sl_offset
                tp_raw = quantized_entry - tp_offset
                # Quantize SL UP (further away), TP DOWN (further away) initially
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_UP)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_DOWN)

            # --- Post-Quantization Validation (Ensure min distance and validity) ---
            # Validate Stop Loss
            if stop_loss is not None:
                sl_boundary = quantized_entry - min_offset_value if signal == "BUY" else quantized_entry + min_offset_value
                needs_sl_adjustment = False
                if signal == "BUY" and stop_loss >= sl_boundary: needs_sl_adjustment = True
                if signal == "SELL" and stop_loss <= sl_boundary: needs_sl_adjustment = True

                if needs_sl_adjustment:
                    rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
                    adjusted_sl = self.quantize_price(sl_boundary, rounding=rounding)
                    # If quantizing the boundary didn't move it far enough, move one more tick
                    if adjusted_sl == stop_loss:
                         adjusted_sl = self.quantize_price(sl_boundary - (min_tick if signal == "BUY" else -min_tick), rounding=rounding)

                    if adjusted_sl is not None and adjusted_sl != stop_loss:
                        self.logger.warning(f"Initial SL ({stop_loss}) too close to entry ({quantized_entry}) after quantization. Adjusting to {adjusted_sl} ({MIN_TICKS_AWAY_FOR_SLTP}+ ticks away).")
                        stop_loss = adjusted_sl
                    else:
                        self.logger.error(f"Could not adjust SL ({stop_loss}) to be minimum distance from entry ({quantized_entry}). Setting SL to None for safety.")
                        stop_loss = None  # Safety measure

                # Final check for zero/negative SL
                if stop_loss is not None and stop_loss <= 0:
                    self.logger.error(f"Calculated SL is zero or negative ({stop_loss}) after adjustments. Setting SL to None.")
                    stop_loss = None

            # Validate Take Profit
            if take_profit is not None:
                tp_boundary = quantized_entry + min_offset_value if signal == "BUY" else quantized_entry - min_offset_value
                needs_tp_adjustment = False
                # Check if TP ended up too close or on the wrong side of entry after quantization
                if signal == "BUY" and take_profit <= tp_boundary: needs_tp_adjustment = True
                if signal == "SELL" and take_profit >= tp_boundary: needs_tp_adjustment = True

                if needs_tp_adjustment:
                    rounding = ROUND_UP if signal == "BUY" else ROUND_DOWN
                    adjusted_tp = self.quantize_price(tp_boundary, rounding=rounding)
                    # If quantizing the boundary didn't move it far enough, move one more tick
                    if adjusted_tp == take_profit:
                        adjusted_tp = self.quantize_price(tp_boundary + (min_tick if signal == "BUY" else -min_tick), rounding=rounding)

                    if adjusted_tp is not None and adjusted_tp != take_profit:
                        self.logger.warning(f"Initial TP ({take_profit}) too close to entry ({quantized_entry}) after quantization. Adjusting to {adjusted_tp} ({MIN_TICKS_AWAY_FOR_SLTP}+ ticks away).")
                        take_profit = adjusted_tp
                    else:
                        self.logger.error(f"Could not adjust TP ({take_profit}) to be minimum distance from entry ({quantized_entry}). Setting TP to None for safety.")
                        take_profit = None  # Safety measure

                # Final check for zero/negative TP
                if take_profit is not None and take_profit <= 0:
                    self.logger.error(f"Calculated TP is zero or negative ({take_profit}) after adjustments. Setting TP to None.")
                    take_profit = None

            # Log results
            prec = self.get_price_precision_digits()
            tp_log = f"{take_profit:.{prec}f}" if take_profit else 'None'
            sl_log = f"{stop_loss:.{prec}f}" if stop_loss else 'None'
            entry_log = f"{quantized_entry:.{prec}f}"
            self.logger.info(f"Calc TP/SL ({signal}): EntryRef={entry_log}, TP={tp_log}, SL={sl_log} (ATR={atr:.{prec + 1}f}, SLx={sl_mult}, TPx={tp_mult})")

            return quantized_entry, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error during TP/SL calculation value conversion for {self.symbol}: {e}{RESET}")
            return quantized_entry, None, None  # Return entry if valid, but no TP/SL
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return quantized_entry, None, None


# --- Position Sizing ---
def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,  # From config (e.g., 0.01 for 1%)
    entry_price: Decimal,
    stop_loss_price: Decimal,
    market_info: dict,
    leverage: int,  # From config
    logger: logging.Logger
) -> Decimal | None:
    """Calculates the position size in base currency units (Spot) or contracts (Futures)
    based on risk percentage, entry/SL prices, and available balance.
    Uses Decimal precision throughout and validates against market limits and available margin.

    Args:
        balance: Available balance in quote currency (Decimal).
        risk_per_trade: Fraction of balance to risk (float, e.g., 0.01 for 1%).
        entry_price: Proposed entry price (Decimal, already quantized potentially).
        stop_loss_price: Proposed stop loss price (Decimal, already quantized).
        market_info: Dictionary containing market details (precision, limits, contract size, type).
        leverage: Leverage to be used (int, relevant for contracts).
        logger: Logger instance.

    Returns:
        Calculated and quantized position size (Decimal) in base units (spot) or contracts (futures),
        or None if calculation fails or constraints (min/max size, min cost, margin) are not met.
    """
    lg = logger
    symbol: str = market_info.get('symbol', 'N/A')
    contract_size: Decimal = market_info.get('contract_size', Decimal('1'))  # Default 1 (for spot/linear)
    min_order_amount: Decimal | None = market_info.get('min_order_amount')
    max_order_amount: Decimal | None = market_info.get('max_order_amount')
    min_order_cost: Decimal | None = market_info.get('min_order_cost')
    amount_step_size: Decimal | None = market_info.get('amount_step_size')
    is_contract: bool = market_info.get('is_contract', False)
    is_inverse: bool = market_info.get('inverse', False)
    quote_currency: str = market_info.get('quote', '?')
    base_currency: str = market_info.get('base', '?')

    # --- Input Validation ---
    if balance is None or not isinstance(balance, Decimal) or balance <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid balance {balance}"); return None
    if not entry_price or not entry_price.is_finite() or entry_price <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid entry price {entry_price}"); return None
    if not stop_loss_price or not stop_loss_price.is_finite() or stop_loss_price <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid SL price {stop_loss_price}"); return None
    if entry_price == stop_loss_price:
        lg.error(f"Size Calc Error ({symbol}): Entry price equals SL price ({entry_price})"); return None
    if amount_step_size is None or amount_step_size <= 0:
        # Try to get digits as fallback if step size is missing
        amount_digits = market_info.get('amount_precision_digits')
        if amount_digits is None:
            lg.error(f"Size Calc Error ({symbol}): Amount step size and precision digits are missing. Cannot quantize size.")
            return None
        else:
            amount_step_size = Decimal('1') / (Decimal('10') ** amount_digits)
            lg.warning(f"Amount step size missing, using precision digits ({amount_digits}) -> step={amount_step_size}")
    if not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Error ({symbol}): Invalid risk_per_trade {risk_per_trade} (must be > 0 and < 1)"); return None
    if is_contract and leverage <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid leverage {leverage} for contract"); return None

    try:
        # --- Calculate Risk Amount and SL Distance ---
        risk_amount_quote: Decimal = balance * Decimal(str(risk_per_trade))  # Risk amount in quote currency
        sl_distance_points: Decimal = abs(entry_price - stop_loss_price)
        if sl_distance_points <= 0:
            lg.error(f"Size Calc Error ({symbol}): SL distance is zero or negative"); return None

        # --- Calculate Risk Per Unit/Contract in Quote Currency ---
        risk_per_unit_or_contract_quote = Decimal('NaN')
        size_unit_name = ""  # For logging

        if is_contract:
            if is_inverse:
                # Inverse: Risk per Contract (in Quote) = Contract Size * |1/SL - 1/Entry| * Entry Price
                if entry_price == 0 or stop_loss_price == 0:
                    lg.error(f"Size Calc Error ({symbol}): Zero price encountered for inverse calculation"); return None
                risk_per_unit_or_contract_quote = contract_size * abs(Decimal('1') / stop_loss_price - Decimal('1') / entry_price) * entry_price
                size_unit_name = f"contracts ({base_currency} base)"
            else:  # Linear Contract
                # Linear: Risk per Contract (in Quote) = ContractSize * |Entry - SL|
                risk_per_unit_or_contract_quote = contract_size * sl_distance_points
                size_unit_name = f"contracts ({base_currency} base)"
        else:  # Spot
            # Spot: Risk per Unit (in Quote) = |Entry - SL| (since contract size is 1 base unit)
            risk_per_unit_or_contract_quote = sl_distance_points
            size_unit_name = f"{base_currency} units"  # Size is in base currency units

        # Validate calculated risk per unit
        if not risk_per_unit_or_contract_quote.is_finite() or risk_per_unit_or_contract_quote <= 0:
            lg.error(f"Size Calc Error ({symbol}): Invalid calculated risk per unit/contract ({risk_per_unit_or_contract_quote})")
            return None

        # --- Calculate Unquantized Size ---
        # Size = Total Risk Amount (Quote) / Risk Per Unit/Contract (Quote)
        size_unquantized = risk_amount_quote / risk_per_unit_or_contract_quote

        if not size_unquantized.is_finite() or size_unquantized <= 0:
            lg.error(f"Size Calc Error ({symbol}): Invalid unquantized size ({size_unquantized}) from RiskAmt={risk_amount_quote:.4f} / RiskPerUnit={risk_per_unit_or_contract_quote:.8f}")
            return None

        lg.debug(f"Size Calc ({symbol}): Bal={balance:.2f}, Risk={risk_per_trade * 100:.2f}%, RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_points:.{market_info.get('price_precision_digits', 4)}f}, RiskPerUnit={risk_per_unit_or_contract_quote:.8f}, UnquantSize={size_unquantized:.8f} {size_unit_name}")

        # --- Quantize Size ---
        # Create a temporary analyzer instance JUST for quantization (slightly hacky)
        # TODO: Refactor quantization to be standalone or accessible differently
        temp_analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info, {})  # Minimal valid init
        quantized_size = temp_analyzer.quantize_amount(size_unquantized, rounding=ROUND_DOWN)

        if quantized_size is None or quantized_size <= 0:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Size is zero or invalid after quantization (Unquantized: {size_unquantized:.8f}). Cannot place order.{RESET}")
            return None

        lg.debug(f"Quantized Size ({symbol}): {quantized_size} {size_unit_name} (Step: {amount_step_size})")

        # --- Validate Against Market Limits (Amount) ---
        if min_order_amount is not None and quantized_size < min_order_amount:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} {size_unit_name} < Min Amount {min_order_amount}. Cannot place order.{RESET}")
            return None
        if max_order_amount is not None and quantized_size > max_order_amount:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} {size_unit_name} > Max Amount {max_order_amount}. Capping size to max.{RESET}")
            # Cap the size and re-quantize using floor (ROUND_DOWN)
            quantized_size = temp_analyzer.quantize_amount(max_order_amount, rounding=ROUND_DOWN)
            if quantized_size is None or quantized_size <= 0 or (min_order_amount is not None and quantized_size < min_order_amount):
                lg.error(f"Size Calc Error ({symbol}): Capped size {quantized_size} is zero or below min amount {min_order_amount}.")
                return None
            lg.info(f"Size capped to market max: {quantized_size} {size_unit_name}")

        # --- Validate Against Market Limits (Cost) & Margin ---
        order_value_quote = Decimal('NaN')
        margin_required = Decimal('NaN')

        if is_contract:
             if is_inverse:
                 # Inverse value in quote = Contracts * ContractSize (value is in base, use price to convert approx)
                 # Value in Quote = contracts * contract_size (This is value in base, not quote!)
                 # Correct: Value in Quote ≈ Contracts * ContractSize / EntryPrice * EntryPrice = Contracts * ContractSize
                 # But this is value in base. Value in quote depends on price... use entry price as reference.
                 order_value_quote = quantized_size * contract_size  # Value in Base at par
                 # Approximate Quote value for cost check:
                 order_value_quote_approx = quantized_size * contract_size * entry_price  # Rough quote value
                 margin_required = (quantized_size * contract_size) / Decimal(leverage) * entry_price  # Approx margin in quote terms
             else:  # Linear Contract
                 # Linear value in quote = Contracts * ContractSize * EntryPrice
                 order_value_quote = quantized_size * contract_size * entry_price
                 margin_required = order_value_quote / Decimal(leverage)
        else:  # Spot
             # Spot value in quote = Amount (Base) * EntryPrice
             order_value_quote = quantized_size * entry_price
             margin_required = order_value_quote  # Margin is full cost for spot

        # Use the most relevant value for cost check (direct quote value if linear/spot, approx if inverse)
        cost_check_value = order_value_quote_approx if is_inverse else order_value_quote

        # Check min cost limit
        if min_order_cost is not None and cost_check_value < min_order_cost:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Order value ~{cost_check_value:.4f} {quote_currency} < Min Cost {min_order_cost}. Cannot place order.{RESET}")
            return None

        # Check margin requirement vs available balance
        # Add a small buffer (e.g., 0.5% = 1.005) for fees/slippage
        buffer_factor = Decimal("1.005")  # 0.5% buffer
        required_margin_with_buffer = margin_required * buffer_factor

        if required_margin_with_buffer > balance:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Required margin ~{margin_required:.4f} {quote_currency} (Buffered: {required_margin_with_buffer:.4f}) > Available balance {balance:.4f}. Cannot place order.{RESET}")
            # TODO: Optional: Reduce size to fit available margin?
            # Requires recalculating size based on max margin: MaxMargin = Balance / BufferFactor
            # MaxValue = MaxMargin * Leverage; MaxSize = MaxValue / (ContractSize * EntryPrice) etc.
            # For now, just reject the trade if margin is insufficient.
            return None

        # --- Success ---
        lg.info(f"Calculated position size for {symbol}: {quantized_size} {size_unit_name} (Value: ~{cost_check_value:.2f} {quote_currency}, Margin: ~{margin_required:.2f} {quote_currency})")
        return quantized_size

    except (InvalidOperation, ValueError, TypeError, DivisionByZeroError) as e:
         lg.error(f"{NEON_RED}Error during position size calculation for {symbol}: {e}{RESET}", exc_info=True)
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


# --- CCXT Trading Action Wrappers ---

def fetch_positions_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict) -> dict | None:
    """Fetches the current non-zero position for a specific symbol using V5 API via safe_ccxt_call.
    Standardizes the returned position dictionary for easier internal use.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The standard CCXT symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.
        market_info: Market information dictionary (must contain 'category' and 'id').

    Returns:
        A dictionary containing standardized position details if a non-zero position exists,
        otherwise None. Includes keys like 'symbol', 'side', 'contracts' (abs size as Decimal),
        'entryPrice' (Decimal), 'liqPrice' (Decimal, optional), 'unrealizedPnl' (Decimal, optional),
        'leverage' (float, optional), 'positionIdx' (int, from info), 'info' (raw position data),
        'market_info' (passed through). Returns None if no position or error occurs.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id')  # Use exchange-specific ID

    # Only fetch positions for derivative markets
    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping position check for non-derivative symbol: {symbol} (Category: {category})")
        return None
    if not market_id:
         lg.error(f"Cannot fetch positions for {symbol}: Market ID missing in market_info."); return None
    if not exchange.has.get('fetchPositions'):
        lg.error(f"Exchange {exchange.id} does not support fetchPositions()."); return None

    try:
        # Bybit V5 requires category and optionally symbol (market_id)
        # Fetching for a specific symbol is generally more efficient
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Use safe_ccxt_call. Pass standard CCXT symbol for potential CCXT client-side filtering,
        # and V5 params for the API call itself.
        all_positions_raw = safe_ccxt_call(exchange, 'fetch_positions', lg, symbols=[symbol], params=params)

        if not isinstance(all_positions_raw, list):
            lg.error(f"fetch_positions did not return a list for {symbol}. Type: {type(all_positions_raw)}. Response: {all_positions_raw}")
            return None

        # Filter the raw response to find the exact symbol and non-zero size
        for pos_raw in all_positions_raw:
            if not isinstance(pos_raw, dict):
                lg.warning(f"Received non-dict item in positions list: {pos_raw}")
                continue

            # --- Match symbol rigorously ---
            # Use market_id from info if available, fallback to CCXT symbol
            position_market_id = pos_raw.get('info', {}).get('symbol')
            position_ccxt_symbol = pos_raw.get('symbol')
            if position_market_id != market_id and position_ccxt_symbol != symbol:
                 lg.debug(f"Skipping position entry, symbol/market_id mismatch: Expected '{symbol}'/'{market_id}', got '{position_ccxt_symbol}'/'{position_market_id}'")
                 continue

            try:
                 # --- Get position size ---
                 # Standard CCXT field is 'contracts', Bybit V5 info field is 'size'. Prioritize info['size'].
                 pos_size_str = pos_raw.get('info', {}).get('size', pos_raw.get('contracts'))
                 if pos_size_str is None or str(pos_size_str).strip() == "":
                     lg.debug(f"Skipping position entry for {symbol}: Missing or empty size field. Data: {pos_raw.get('info', pos_raw)}")
                     continue

                 pos_size = Decimal(str(pos_size_str))

                 # --- Skip zero size positions ---
                 if pos_size.is_zero():
                     continue  # Not an active position

                 # --- Standardize the position dictionary ---
                 standardized_pos: dict[str, Any] = {}
                 standardized_pos['symbol'] = symbol  # Use the requested symbol
                 standardized_pos['side'] = 'long' if pos_size > 0 else 'short'
                 standardized_pos['contracts'] = abs(pos_size)  # Store absolute size as Decimal

                 # Helper to safely convert string price/pnl to Decimal
                 def safe_decimal(value_str: str | float | int | None) -> Decimal | None:
                     if value_str is None or str(value_str).strip() == "": return None
                     try:
                         d = Decimal(str(value_str))
                         return d if d.is_finite() else None
                     except (InvalidOperation, TypeError): return None

                 # Entry price: Standard CCXT is 'entryPrice', Bybit V5 info is 'avgPrice'. Prioritize info['avgPrice'].
                 entry_price_str = pos_raw.get('info', {}).get('avgPrice', standardized_pos.get('entryPrice'))
                 standardized_pos['entryPrice'] = safe_decimal(entry_price_str)

                 # Liquidation price: Standard CCXT is 'liquidationPrice', Bybit V5 info is 'liqPrice'. Prioritize info['liqPrice'].
                 liq_price_str = pos_raw.get('info', {}).get('liqPrice', standardized_pos.get('liquidationPrice'))
                 standardized_pos['liqPrice'] = safe_decimal(liq_price_str)

                 # Unrealized PnL: Standard CCXT is 'unrealizedPnl', Bybit V5 info is 'unrealisedPnl'. Prioritize info.
                 pnl_str = pos_raw.get('info', {}).get('unrealisedPnl', standardized_pos.get('unrealizedPnl'))
                 standardized_pos['unrealizedPnl'] = safe_decimal(pnl_str)

                 # Leverage: Standard CCXT is 'leverage', Bybit V5 info also 'leverage'. Prioritize info. Store as float.
                 leverage_str = pos_raw.get('info', {}).get('leverage', standardized_pos.get('leverage'))
                 try: standardized_pos['leverage'] = float(leverage_str) if leverage_str else None
                 except (ValueError, TypeError): standardized_pos['leverage'] = None

                 # Position Index (Crucial for Hedge Mode): From Bybit V5 info field 'positionIdx'. Store as int.
                 pos_idx_str = pos_raw.get('info', {}).get('positionIdx')
                 try:
                     # Default to 0 (One-Way mode) if missing or invalid
                     standardized_pos['positionIdx'] = int(pos_idx_str) if pos_idx_str is not None else 0
                 except (ValueError, TypeError):
                     lg.warning(f"Invalid positionIdx '{pos_idx_str}' in position data for {symbol}. Defaulting to 0.")
                     standardized_pos['positionIdx'] = 0

                 # Include raw info and market info for downstream use
                 standardized_pos['info'] = pos_raw.get('info', {})  # Store the raw 'info' dictionary
                 standardized_pos['market_info'] = market_info  # Add market info for convenience

                 # --- Log Found Position ---
                 prec = market_info.get('price_precision_digits', 4)
                 entry_log = f"{standardized_pos['entryPrice']:.{prec}f}" if standardized_pos.get('entryPrice') else 'N/A'
                 liq_log = f"Liq={standardized_pos['liqPrice']:.{prec}f}" if standardized_pos.get('liqPrice') else ''
                 pnl_log = f"PnL={standardized_pos['unrealizedPnl']:.2f}" if standardized_pos.get('unrealizedPnl') else ''
                 lev_log = f"Lev={standardized_pos.get('leverage')}x" if standardized_pos.get('leverage') is not None else ''
                 idx_log = f"Idx={standardized_pos.get('positionIdx')}"

                 lg.info(f"Found active {standardized_pos['side']} position for {symbol}: Size={standardized_pos['contracts']}, Entry={entry_log}, {liq_log} {pnl_log} {lev_log} {idx_log}")
                 return standardized_pos  # Return the first non-zero matching position found

            except (InvalidOperation, ValueError, TypeError) as e:
                 lg.error(f"Could not parse position data for {symbol}: {e}. Raw data: {pos_raw}")
                 # Continue loop to check other potential entries in the list
            except Exception as e:
                 lg.error(f"Unexpected error processing position entry for {symbol}: {e}. Raw data: {pos_raw}", exc_info=True)
                 # Continue loop

        # If loop completes without returning a position
        lg.debug(f"No active non-zero position found for {symbol}.")
        return None

    except Exception as e:
        # Catch errors from safe_ccxt_call (e.g., retries exhausted) or other issues
        lg.error(f"{NEON_RED}Error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger, market_info: dict) -> bool:
    """Sets leverage for a symbol using Bybit V5 API via safe_ccxt_call.
    Handles the "leverage not modified" response (code 110043) as success.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        leverage: Target leverage (integer > 0).
        logger: Logger instance.
        market_info: Market information dictionary (must contain 'category' and 'id').

    Returns:
        True if leverage was set successfully (or was already correct), False otherwise.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id')  # Use exchange-specific ID

    # Leverage only applicable to derivatives
    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping leverage setting for non-derivative: {symbol}")
        return True  # Consider success as no action needed
    if not market_id:
         lg.error(f"Cannot set leverage for {symbol}: Market ID missing."); return False
    if not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage()."); return False
    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Invalid leverage ({leverage}) for {symbol}. Must be a positive integer."); return False

    try:
        # Bybit V5 requires category, symbol, and separate buy/sell leverage values.
        # We set both to the same value based on the config.
        params = {
            'category': category,
            'symbol': market_id,
            'buyLeverage': str(leverage),  # API expects string values
            'sellLeverage': str(leverage)
        }
        lg.info(f"Attempting to set leverage for {symbol} (MarketID: {market_id}) to {leverage}x...")
        lg.debug(f"Leverage Params: {params}")

        # CCXT's set_leverage should map to the correct V5 endpoint /v5/position/set-leverage
        # Pass leverage as float (required by CCXT type hint), symbol, and V5 params.
        # safe_ccxt_call handles "leverage not modified" (110043) by returning {} treated as success.
        result = safe_ccxt_call(exchange, 'set_leverage', lg, leverage=float(leverage), symbol=symbol, params=params)

        # Check result: Success could be a dict with info, or empty dict for 'not modified'
        if result is not None and isinstance(result, dict):
            # Check Bybit's retCode in the info field if available
            ret_code = result.get('info', {}).get('retCode')
            # Treat retCode 0 (Success) or empty dict (110043 handled by safe_ccxt_call) as success
            if ret_code == 0 or result == {}:
                 lg.info(f"{NEON_GREEN}Leverage set successfully (or already correct) for {symbol} to {leverage}x.{RESET}")
                 return True
            else:
                 # This case means safe_ccxt_call might not have caught a specific error code
                 ret_msg = result.get('info', {}).get('retMsg', 'Unknown Error')
                 lg.error(f"{NEON_RED}set_leverage call returned non-zero retCode {ret_code}: '{ret_msg}' for {symbol}.{RESET}")
                 return False
        else:
            # This case should ideally not happen if safe_ccxt_call raises or returns {} on 110043
            lg.error(f"{NEON_RED}set_leverage call returned unexpected result type for {symbol}: {type(result)}. Result: {result}{RESET}")
            return False

    except ccxt.ExchangeError as e:
         # Non-retryable errors should have been raised by safe_ccxt_call
         # Logging the error again here might be redundant but ensures visibility
         lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x due to ExchangeError: {e}{RESET}", exc_info=False)
         return False
    except Exception as e:
        # Catch unexpected errors during the call itself
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
        return False


def create_order_ccxt(
    exchange: ccxt.Exchange, symbol: str, order_type: str, side: str,
    amount: Decimal, price: Decimal | None = None, params: dict | None = None,
    logger: logging.Logger | None = None, market_info: dict | None = None
) -> dict | None:
    """Creates an order using safe_ccxt_call, handling V5 parameters and Decimal conversion.
    Includes parameter validation and detailed logging.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        order_type: 'market' or 'limit'.
        side: 'buy' or 'sell'.
        amount: Order quantity (Decimal, must be positive).
        price: Order price for limit orders (Decimal, must be positive if provided).
        params: Additional parameters for the CCXT create_order call
                (e.g., {'reduceOnly': True, 'positionIdx': 0/1/2}).
                Crucially includes 'category' for V5.
        logger: Logger instance.
        market_info: Market information dictionary (required for validation/logging).

    Returns:
        The CCXT order dictionary if the API call was successful and Bybit confirmed
        with retCode=0. Returns None otherwise.
    """
    lg = logger or get_logger('main')  # Use provided logger or default main logger
    if not market_info:
        lg.error(f"Market info required for create_order ({symbol}) but not provided."); return None

    # --- Input Validation ---
    category = market_info.get('category')
    if not category:
        lg.error(f"Unknown category for {symbol}. Cannot place order."); return None
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount <= 0:
        lg.error(f"Order amount must be a positive Decimal ({symbol}, Amount: {amount})"); return None

    order_type_lower = order_type.lower(); side_lower = side.lower()
    if order_type_lower not in ['market', 'limit']:
        lg.error(f"Invalid order type '{order_type}'. Must be 'market' or 'limit'."); return None
    if side_lower not in ['buy', 'sell']:
        lg.error(f"Invalid order side '{side}'. Must be 'buy' or 'sell'."); return None

    price_float: float | None = None
    price_str: str | None = None
    price_digits = market_info.get('price_precision_digits', 8)
    amount_digits = market_info.get('amount_precision_digits', 8)

    if order_type_lower == 'limit':
        if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
            lg.error(f"Valid positive Decimal price required for limit order ({symbol}, Price: {price})"); return None
        # Convert valid Decimal price to float for CCXT method signature
        price_float = float(price)
        price_str = f"{price:.{price_digits}f}"  # Formatted string for logging
    else:  # Market order
        price = None  # Ensure price is None for CCXT market order call

    # Format amount string for logging
    amount_str = f"{amount:.{amount_digits}f}"

    # --- Prepare V5 Parameters ---
    # Base parameters required by Bybit V5 createOrder
    # Category is crucial for V5 routing
    order_params: dict[str, Any] = {'category': category}
    # Merge external params (like reduceOnly, positionIdx, sl/tp) provided by caller
    # Caller is responsible for providing correct `positionIdx` based on position_mode config
    if params:
        order_params.update(params)
        # Ensure positionIdx is int if provided
        if 'positionIdx' in order_params:
             try: order_params['positionIdx'] = int(order_params['positionIdx'])
             except (ValueError, TypeError):
                 lg.error(f"Invalid positionIdx type in params: {order_params['positionIdx']}. Cannot create order.")
                 return None
        # Convert SL/TP prices in params to strings if they are Decimals
        for key in ['stopLoss', 'takeProfit']:
             if key in order_params and isinstance(order_params[key], Decimal):
                  try:
                      val = order_params[key]
                      if val.is_finite() and val > 0:
                           order_params[key] = f"{val:.{price_digits}f}"
                      else:
                           lg.warning(f"Invalid Decimal value {val} for param '{key}'. Removing from params.")
                           del order_params[key]
                  except Exception as fmt_err:
                       lg.error(f"Error formatting Decimal param '{key}': {fmt_err}. Removing.")
                       del order_params[key]

    # --- Convert Amount Decimal to Float for CCXT call ---
    try:
        # CCXT methods typically expect float for amount
        amount_float = float(amount)
    except ValueError as e:
        lg.error(f"Error converting amount '{amount}' to float ({symbol}): {e}"); return None

    # --- Place Order via safe_ccxt_call ---
    try:
        log_price_part = f'@ {price_str}' if price_str else 'at Market'
        log_param_part = f" Params: {order_params}"  # Log combined params
        lg.info(f"Attempting to create {side.upper()} {order_type.upper()} order: {amount_str} {symbol} {log_price_part}{log_param_part}")
        # Debug log showing exact values passed to CCXT
        lg.debug(f"CCXT create_order Args: symbol='{symbol}', type='{order_type}', side='{side}', amount={amount_float}, price={price_float}, params={order_params}")

        order_result = safe_ccxt_call(
            exchange, 'create_order', lg,
            symbol=symbol, type=order_type, side=side,
            amount=amount_float, price=price_float, params=order_params
        )

        # --- Process Result ---
        # Check if the call succeeded and returned a valid order structure with an ID
        if order_result and isinstance(order_result, dict) and order_result.get('id'):
            order_id = order_result['id']
            # Check Bybit's V5 response code within the 'info' field for confirmation
            # retCode=0 indicates success on Bybit's side
            ret_code = order_result.get('info', {}).get('retCode')
            ret_msg = order_result.get('info', {}).get('retMsg', 'Unknown Status')

            if ret_code == 0:
                 lg.info(f"{NEON_GREEN}Successfully created {side.upper()} {order_type.upper()} order for {symbol}. Order ID: {order_id}{RESET}")
                 lg.debug(f"Order Result Info: {order_result.get('info')}")
                 # Return the full CCXT order dict on success
                 return order_result
            else:
                 # Order ID might be generated even if rejected (e.g., insufficient balance)
                 lg.error(f"{NEON_RED}Order placement potentially failed or rejected by Bybit ({symbol}). Order ID: {order_id}, Code={ret_code}, Msg='{ret_msg}'.{RESET}")
                 lg.debug(f"Failed Order Result Info: {order_result.get('info')}")
                 # Provide hints for common rejection codes (already handled in safe_ccxt_call, but good here too)
                 if ret_code == 110007: lg.warning(f"Hint: Order rejected due to insufficient balance (Code {ret_code}).")
                 elif ret_code == 110017: lg.warning(f"Hint: Order rejected due to price/qty precision error (Code {ret_code}). Check market limits.")
                 elif ret_code == 110045 or ret_code == 170007: lg.warning(f"Hint: Order rejected due to risk limit exceeded (Code {ret_code}).")
                 elif ret_code == 110025: lg.warning(f"Hint: Order rejected due to positionIdx mismatch (Code {ret_code}). Ensure Hedge Mode params are correct.")
                 # Treat non-zero retCode as failure
                 return None
        elif order_result:
             # Call succeeded but response format is unexpected (e.g., missing ID)
             lg.error(f"Order API call successful but response missing ID or invalid format ({symbol}). Response: {order_result}")
             return None
        else:  # safe_ccxt_call returned None or raised an exception handled within it
             lg.error(f"Order API call failed or returned None ({symbol}) after retries.")
             return None

    except Exception as e:
        # Catch any unexpected error during the process
        lg.error(f"{NEON_RED}Unexpected error creating order ({symbol}): {e}{RESET}", exc_info=True)
        return None


def set_protection_ccxt(
    exchange: ccxt.Exchange, symbol: str,
    stop_loss_price: Decimal | None = None, take_profit_price: Decimal | None = None,
    trailing_stop_price: Decimal | None = None,  # TSL *distance/offset* value (e.g., 100 for $100 behind)
    trailing_active_price: Decimal | None = None,  # TSL *activation price* trigger
    position_idx: int = 0,  # Required for Hedge mode (0=OneWay, 1=BuyHedge, 2=SellHedge)
    logger: logging.Logger | None = None, market_info: dict | None = None
) -> bool:
    """Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL) for a position
    using Bybit V5 `POST /v5/position/trading-stop` via `private_post_position_trading_stop`.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        stop_loss_price: Price for the stop loss order (Decimal). Set to 0 or None to remove/not set.
        take_profit_price: Price for the take profit order (Decimal). Set to 0 or None to remove/not set.
        trailing_stop_price: The trailing stop distance/offset value (Decimal, positive). Set to 0 or None to remove/not set.
                             Bybit interprets this as the price distance (e.g., 10 for $10 below high for long).
        trailing_active_price: The price at which the TSL should activate (Decimal).
                               Set to 0 or None for immediate activation if trailing_stop_price is set.
        position_idx: Position index (0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge). Must match position.
        logger: Logger instance.
        market_info: Market information dictionary (required for validation/formatting).

    Returns:
        True if the protection was set successfully (API call returns retCode=0), False otherwise.
    """
    lg = logger or get_logger('main')
    if not market_info:
        lg.error(f"Market info required for set_protection ({symbol}) but not provided."); return False

    category = market_info.get('category')
    market_id = market_info.get('id')  # Use exchange-specific ID for API calls
    price_digits = market_info.get('price_precision_digits', 8)

    # Protection setting only applicable to derivatives
    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Cannot set protection for non-derivative {symbol}. Category: {category}"); return False
    if not market_id:
         lg.error(f"Cannot set protection for {symbol}: Market ID missing."); return False

    # --- Prepare V5 Parameters ---
    params: dict[str, Any] = {
        'category': category,
        'symbol': market_id,
        'positionIdx': position_idx,  # Crucial for Hedge Mode, 0 for One-Way
        # --- Optional Parameters (Defaults are usually sufficient) ---
        # 'tpslMode': 'Full', # Apply TP/SL to the entire position ('Partial' also available)
        # 'slTriggerBy': 'LastPrice', # MarkPrice, IndexPrice
        # 'tpTriggerBy': 'LastPrice', # MarkPrice, IndexPrice
        # 'slOrderType': 'Market', # Default is Market, 'Limit' also possible
        # 'tpOrderType': 'Market', # Default is Market, 'Limit' also possible
    }

    # --- Format Price/Value Strings for API ---
    # Bybit API expects prices/distances as strings. "0" is used to cancel/remove existing protection.
    def format_value(value: Decimal | None, name: str, is_distance: bool = False) -> str:
        """Formats Decimal to string for API, validating positivity. Returns '0' if invalid/None."""
        if value is not None and isinstance(value, Decimal) and value.is_finite():
            if value > 0:
                # Format based on price precision. TSL distance might need different formatting
                # if API is strict, but using price precision is usually safe.
                # TODO: Verify Bybit V5 requirements for TSL distance formatting if issues arise.
                return f"{value:.{price_digits}f}"
            elif value == 0:
                return "0"  # Explicitly setting to zero means cancel
            else:  # Negative value is invalid
                 lg.warning(f"Invalid negative value '{value}' provided for '{name}' in set_protection. Using '0' (cancel).")
                 return "0"
        else:  # None or non-finite Decimal
            return "0"  # Treat as "do not set" or "cancel"

    sl_str = format_value(stop_loss_price, "stopLoss")
    tp_str = format_value(take_profit_price, "takeProfit")
    tsl_dist_str = format_value(trailing_stop_price, "trailingStop", is_distance=True)
    tsl_act_str = format_value(trailing_active_price, "activePrice")

    # Add to params only if a value is being set (is not "0")
    # Setting to "0" explicitly cancels that specific protection type.
    action_taken = False
    # Add if value > 0 OR if user explicitly passed Decimal(0) to cancel
    if sl_str != "0" or (stop_loss_price is not None and stop_loss_price == 0):
        params['stopLoss'] = sl_str; action_taken = True
    if tp_str != "0" or (take_profit_price is not None and take_profit_price == 0):
        params['takeProfit'] = tp_str; action_taken = True

    # Trailing Stop: 'trailingStop' is the distance. 'activePrice' is optional trigger.
    if tsl_dist_str != "0" or (trailing_stop_price is not None and trailing_stop_price == 0):
        params['trailingStop'] = tsl_dist_str
        action_taken = True
        # Only include activePrice if TSL distance is non-zero.
        # If distance is "0" (cancelling TSL), activePrice is irrelevant/ignored.
        if tsl_dist_str != "0":
            params['activePrice'] = tsl_act_str  # "0" means immediate activation for activePrice
        # elif 'activePrice' in params: # Should not be needed if logic above is correct
        #     del params['activePrice'] # Remove activePrice if cancelling TSL

    # --- Log Intention ---
    log_parts = []
    if 'stopLoss' in params: log_parts.append(f"SL={params['stopLoss']}")
    if 'takeProfit' in params: log_parts.append(f"TP={params['takeProfit']}")
    if 'trailingStop' in params:
        tsl_log = f"TSL Dist={params['trailingStop']}"
        # Log activation price only if TSL distance is being set (>0)
        if params['trailingStop'] != "0":
            if params.get('activePrice', "0") != "0":
                 tsl_log += f", ActP={params['activePrice']}"
            else:
                 tsl_log += ", Act=Immediate"  # Activation price 0 means immediate
        log_parts.append(tsl_log)

    if not action_taken:
        lg.info(f"No valid protection levels provided or changes needed for set_protection ({symbol}). No API call made.")
        # Consider it success if no action was needed (e.g., called with all Nones)
        return True

    # --- Make API Call ---
    try:
        lg.info(f"Attempting to set protection for {symbol} (Idx: {position_idx}): {', '.join(log_parts)}")
        lg.debug(f"Protection Params: {params}")

        # Explicitly use the private POST method mapped in exchange options during initialization
        # This ensures we hit the correct V5 endpoint: /v5/position/trading-stop
        method_to_call = 'private_post_position_trading_stop'
        if not hasattr(exchange, method_to_call):
            # This check should ideally never fail if initialization mapped correctly.
            lg.error(f"Method '{method_to_call}' not found on exchange object. Check CCXT mapping in initialize_exchange.")
            return False

        # Use safe_ccxt_call to handle retries and errors
        result = safe_ccxt_call(exchange, method_to_call, lg, params=params)

        # --- Process Result ---
        # Check Bybit's V5 response code for success (0)
        if result and isinstance(result, dict) and result.get('retCode') == 0:
            lg.info(f"{NEON_GREEN}Successfully set protection for {symbol}.{RESET}")
            lg.debug(f"Protection Result Info: {result}")
            return True
        elif result:
            # API call returned, but retCode indicates failure
            ret_code = result.get('retCode', -1); ret_msg = result.get('retMsg', 'Unknown Error')
            lg.error(f"{NEON_RED}Failed to set protection ({symbol}). Code={ret_code}, Msg='{ret_msg}'{RESET}")
            lg.debug(f"Protection Failure Info: {result}")
            # Provide hints based on common error codes (some handled by safe_ccxt_call non-retryable list)
            if ret_code in [170140, 110041, 110024]: lg.warning(f"Hint: Set protection failed because position might not exist or size is zero (Code {ret_code}).")
            elif ret_code == 110025: lg.warning(f"Hint: Set protection failed due to positionIdx mismatch (Code {ret_code}). Check Hedge Mode settings.")
            elif ret_code in [170131, 110042]: lg.warning(f"Hint: Set protection failed due to invalid TP/SL price (Code {ret_code}). Check price vs mark/index/entry.")
            elif ret_code == 170133: lg.warning(f"Hint: Cannot set TP/SL/TSL, potentially due to order status or existing orders (Code {ret_code}).")
            return False
        else:  # safe_ccxt_call returned None (likely hit max retries or raised internal error)
            lg.error(f"Set protection API call failed or returned None ({symbol}) after retries.")
            return False

    except Exception as e:
        # Catch unexpected errors during the process
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
        return False


def close_position_ccxt(
    exchange: ccxt.Exchange, symbol: str, position_data: dict,
    logger: logging.Logger | None = None, market_info: dict | None = None
) -> dict | None:
    """Closes an existing position via a Market order with the 'reduceOnly' flag.
    Uses position data to determine the side, size, and position index for the closing order.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        position_data: The standardized position dictionary obtained from fetch_positions_ccxt.
                       Must contain 'side' ('long'/'short'), 'contracts' (Decimal size),
                       and 'positionIdx' (int, especially for Hedge Mode).
        logger: Logger instance.
        market_info: Market information dictionary (required for placing the order).

    Returns:
        The CCXT order dictionary for the closing order if placed successfully (retCode=0 from create_order_ccxt),
        otherwise None.
    """
    lg = logger or get_logger('main')
    if not market_info:
        lg.error(f"Market info required for close_position ({symbol}) but not provided."); return None
    if not position_data or not isinstance(position_data, dict):
        lg.error(f"Valid position data dictionary required for close_position ({symbol})"); return None

    try:
        # --- Extract necessary details from position data ---
        position_side = position_data.get('side')  # 'long' or 'short'
        position_size_dec = position_data.get('contracts')  # Absolute Decimal size
        position_idx = position_data.get('positionIdx')  # Integer index (0 for One-Way, 1/2 for Hedge)

        # --- Validate position data needed for closing ---
        if position_side not in ['long', 'short']:
            lg.error(f"Invalid side ('{position_side}') in position data for closing {symbol}"); return None
        if not isinstance(position_size_dec, Decimal) or position_size_dec <= 0:
             lg.error(f"Invalid size ('{position_size_dec}') in position data for closing {symbol}"); return None
        if position_idx is None or not isinstance(position_idx, int):
             lg.error(f"Missing or invalid positionIdx ('{position_idx}') in position data for closing {symbol}"); return None

        # Determine the side of the closing order (opposite of position side)
        close_side = 'sell' if position_side == 'long' else 'buy'
        amount_to_close = position_size_dec  # Use the absolute size from position data

        lg.info(f"Attempting to close {position_side} position ({symbol}, Size: {amount_to_close}, Idx: {position_idx}) via {close_side.upper()} MARKET order...")

        # --- Prepare Parameters for Closing Order ---
        # 'reduceOnly': True ensures the order only closes or reduces the position.
        # 'positionIdx': Crucial for identifying which position to close in Hedge Mode.
        close_params: dict[str, Any] = {
            'reduceOnly': True,
            'positionIdx': position_idx
            # V5 category will be added by create_order_ccxt
        }

        # Use create_order_ccxt to place the closing market order
        close_order_result = create_order_ccxt(
            exchange=exchange, symbol=symbol, order_type='market', side=close_side,
            amount=amount_to_close, price=None,  # Market order has no price
            params=close_params, logger=lg, market_info=market_info
        )

        # Check if the closing order was placed successfully (create_order_ccxt checks retCode=0)
        if close_order_result and close_order_result.get('id'):
            lg.info(f"{NEON_GREEN}Successfully placed MARKET order to close {position_side} position ({symbol}, Idx: {position_idx}). Close Order ID: {close_order_result.get('id')}{RESET}")
            return close_order_result
        else:
            lg.error(f"{NEON_RED}Failed to place market order to close position ({symbol}, Idx: {position_idx}). Check logs from create_order_ccxt.{RESET}")
            # Reasons could include: insufficient margin for fees, position already closed, network error after retries, etc.
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Error processing position data for closing ({symbol}): {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error attempting to close position ({symbol}): {e}{RESET}", exc_info=True)
        return None


# --- Main Bot Logic ---
def attempt_new_entry(
    exchange: ccxt.Exchange, config: dict[str, Any], logger: logging.Logger,
    analyzer: TradingAnalyzer, signal: str, entry_price_signal: Decimal, current_balance: Decimal
) -> bool:
    """Attempts to enter a new trade based on a BUY or SELL signal. Executes the full workflow:
    1. Calculate TP/SL based on ATR and signal price.
    2. Calculate position size based on risk, balance, entry/SL.
    3. Set leverage (important before opening).
    4. Create the market entry order.
    5. Wait briefly and confirm the position was opened.
    6. Set TP/SL/TSL protection on the confirmed position.

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance for the symbol.
        analyzer: TradingAnalyzer instance with current data and calculation methods.
        signal: The entry signal ("BUY" or "SELL").
        entry_price_signal: The price near which the signal occurred (Decimal).
        current_balance: Current available balance in quote currency (Decimal).

    Returns:
        bool: True if the entry sequence was successfully completed (including setting protection),
              False otherwise.
    """
    lg = logger
    symbol = analyzer.symbol
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state  # Access shared state

    lg.info(f"Attempting new {signal} entry for {symbol}...")

    # --- 1. Calculate Entry/TP/SL ---
    entry_ref, tp_price, sl_price = analyzer.calculate_entry_tp_sl(entry_price_signal, signal)

    # Validate: Need at least an entry reference and a valid stop loss for size calculation
    if entry_ref is None or sl_price is None:
        lg.error(f"Entry Aborted ({symbol}): Failed to calculate valid entry/SL price. EntryRef={entry_ref}, SL={sl_price}")
        return False
    # TP is optional for entry, can be set later or if TSL is used.

    # --- 2. Calculate Position Size ---
    leverage = config.get('leverage', 1)  # Already validated int
    risk_per_trade = config.get('risk_per_trade', 0.01)  # Already validated float
    position_size = calculate_position_size(
        balance=current_balance, risk_per_trade=risk_per_trade,
        entry_price=entry_ref,  # Use the reference entry price for sizing
        stop_loss_price=sl_price, market_info=market_info,
        leverage=leverage, logger=lg
    )

    if position_size is None or position_size <= 0:
        lg.error(f"Entry Aborted ({symbol}): Failed to calculate valid position size.")
        return False

    # --- 3. Set Leverage (if applicable, only for contracts) ---
    if market_info.get('is_contract', False):
        if not set_leverage_ccxt(exchange, symbol, leverage, lg, market_info):
            lg.error(f"Entry Aborted ({symbol}): Failed to set leverage to {leverage}x.")
            # If leverage setting fails, it's risky to proceed with the order.
            return False

    # --- 4. Create Market Entry Order ---
    entry_side = 'buy' if signal == 'BUY' else 'sell'
    # Determine positionIdx based on config and signal (Hedge Mode needs specific index)
    position_mode = config.get("position_mode", "One-Way")
    entry_position_idx = 0  # Default for One-Way
    if position_mode == "Hedge":
        entry_position_idx = 1 if entry_side == 'buy' else 2
        lg.info(f"Using positionIdx={entry_position_idx} for {entry_side} order in Hedge Mode.")

    entry_params: dict[str, Any] = {'positionIdx': entry_position_idx}
    # Note: Bybit V5 doesn't allow setting TP/SL directly on market order creation.
    # We'll set them after confirming the position.

    entry_order = create_order_ccxt(
        exchange=exchange, symbol=symbol, order_type='market', side=entry_side,
        amount=position_size, params=entry_params, logger=lg, market_info=market_info
    )

    if not entry_order or not entry_order.get('id'):
        lg.error(f"Entry Aborted ({symbol}): Failed to create market entry order.")
        return False

    # --- 5. Wait and Confirm Position ---
    lg.info(f"Entry order placed ({symbol}, ID: {entry_order.get('id')}). Waiting {POSITION_CONFIRM_DELAY}s to confirm position...")
    time.sleep(POSITION_CONFIRM_DELAY)

    # Fetch the position again to get the actual entry price and confirm it exists
    confirmed_position = fetch_positions_ccxt(exchange, symbol, lg, market_info)

    if not confirmed_position:
        lg.error(f"{NEON_RED}Entry Failed ({symbol}): Position not found after placing order {entry_order.get('id')}. Manual check required!{RESET}")
        # This is a critical failure - the order might have filled but we can't confirm.
        # Need to prevent further actions on this symbol until resolved.
        # Potential state handling: set a 'needs_manual_check' flag?
        return False

    # --- Validate Confirmed Position ---
    confirmed_entry_price = confirmed_position.get('entryPrice')  # Decimal
    confirmed_side = confirmed_position.get('side')
    confirmed_size = confirmed_position.get('contracts')  # Decimal
    confirmed_pos_idx = confirmed_position.get('positionIdx')  # int

    # Basic sanity checks
    if confirmed_side != entry_side.lower():
        lg.error(f"{NEON_RED}Entry Failed ({symbol}): Confirmed position side ({confirmed_side}) differs from order side ({entry_side}). Manual check required!{RESET}")
        return False
    if confirmed_pos_idx != entry_position_idx:
         lg.error(f"{NEON_RED}Entry Failed ({symbol}): Confirmed position index ({confirmed_pos_idx}) differs from order index ({entry_position_idx}). Manual check required!{RESET}")
         return False
    # Check if size is reasonably close (allow for slight variations due to fees/slippage if needed)
    # Using isclose for Decimals is tricky, compare relative difference.
    if confirmed_size and position_size:
        size_diff_pct = abs(confirmed_size - position_size) / position_size if position_size else Decimal('0')
        if size_diff_pct > Decimal('0.05'):  # Allow up to 5% difference
            lg.warning(f"{NEON_YELLOW}Position size mismatch ({symbol}): Ordered={position_size}, Confirmed={confirmed_size} (Diff: {size_diff_pct:.2%}). Proceeding, but check if expected.{RESET}")
    else:
        lg.error(f"{NEON_RED}Entry Failed ({symbol}): Invalid confirmed size ({confirmed_size}) or order size ({position_size}). Manual check required!{RESET}")
        return False

    if not confirmed_entry_price:
        lg.error(f"{NEON_RED}Entry Failed ({symbol}): Could not retrieve entry price from confirmed position. Cannot set protection accurately.{RESET}")
        # If we can't get entry price, setting accurate SL/TP is impossible. Consider this a failure.
        return False

    lg.info(f"Position Confirmed ({symbol}): Side={confirmed_side}, Size={confirmed_size}, Entry={confirmed_entry_price}, Idx={confirmed_pos_idx}")
    # Store the confirmed entry price in the shared state
    symbol_state['last_entry_price'] = str(confirmed_entry_price)  # Store as string for JSON

    # --- 6. Set Protection (TP/SL/TSL) ---
    # Re-calculate TP/SL slightly based on the *actual* confirmed entry price for accuracy?
    # Or just use the initially calculated ones? Using initial ones is simpler, assumes market order fill is close.
    # Let's stick with initially calculated tp_price and sl_price for simplicity, but log the difference.
    if entry_ref != confirmed_entry_price:
         lg.debug(f"Note: Using initial TP/SL based on EntryRef={entry_ref}, Actual Entry={confirmed_entry_price}. Difference: {abs(entry_ref - confirmed_entry_price)}")

    tsl_price_distance: Decimal | None = None
    tsl_active_price: Decimal | None = None

    if config.get("enable_trailing_stop", False) and market_info.get('is_contract'):
        # Calculate TSL distance and activation price
        tsl_callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))  # e.g., 0.5%
        tsl_activation_perc = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))  # e.g., 0.3%

        # TSL distance is absolute price distance (e.g., 100 for $100)
        # Calculate this based on the entry price and callback rate
        tsl_price_distance = analyzer.quantize_price(confirmed_entry_price * tsl_callback_rate, rounding=ROUND_UP)  # Round distance up

        if tsl_price_distance is not None and tsl_price_distance > 0:
            # Calculate activation price based on entry and percentage gain
            # If activation percentage is 0, active price is 0 (immediate activation)
            if tsl_activation_perc > 0:
                 activation_offset = confirmed_entry_price * tsl_activation_perc
                 raw_active_price = Decimal('0')
                 if confirmed_side == 'long':
                     raw_active_price = confirmed_entry_price + activation_offset
                     tsl_active_price = analyzer.quantize_price(raw_active_price, rounding=ROUND_UP)
                 else:  # Short
                     raw_active_price = confirmed_entry_price - activation_offset
                     tsl_active_price = analyzer.quantize_price(raw_active_price, rounding=ROUND_DOWN)
                 # Ensure activation price is valid (positive)
                 if tsl_active_price is not None and tsl_active_price <= 0:
                     lg.warning(f"Calculated TSL activation price {tsl_active_price} is invalid. Setting activation to immediate (0).")
                     tsl_active_price = Decimal('0')
            else:
                # Immediate activation if percentage is zero
                tsl_active_price = Decimal('0')

            # Validate TSL distance is not too small (e.g., less than a tick)
            min_tick = analyzer.get_min_tick_size()
            if min_tick and tsl_price_distance < min_tick:
                lg.warning(f"Calculated TSL distance {tsl_price_distance} is less than min tick {min_tick}. Disabling TSL.")
                tsl_price_distance = None
                tsl_active_price = None
        else:
            lg.warning(f"Failed to calculate valid TSL distance ({tsl_price_distance}). Disabling TSL.")
            tsl_price_distance = None  # Disable TSL if distance is invalid
            tsl_active_price = None

    # Set protection using the dedicated function
    # Pass the confirmed position index
    protection_set = set_protection_ccxt(
        exchange=exchange, symbol=symbol,
        stop_loss_price=sl_price,          # Use initially calculated SL
        take_profit_price=tp_price,        # Use initially calculated TP
        trailing_stop_price=tsl_price_distance,  # Pass calculated distance
        trailing_active_price=tsl_active_price,  # Pass calculated activation price
        position_idx=confirmed_pos_idx,
        logger=lg, market_info=market_info
    )

    if not protection_set:
        lg.error(f"{NEON_RED}Entry Failed ({symbol}): Failed to set initial TP/SL/TSL protection after confirming position. Closing position for safety...{RESET}")
        # If protection setting fails, close the position immediately to avoid uncontrolled risk.
        close_result = close_position_ccxt(exchange, symbol, confirmed_position, lg, market_info)
        if close_result:
            lg.info(f"Successfully closed position for {symbol} due to failed protection setting.")
        else:
            lg.critical(f"{NEON_RED}CRITICAL: Failed to set protection AND failed to close position for {symbol}. MANUAL INTERVENTION REQUIRED!{RESET}")
        # Reset state regardless of close success to prevent re-entry attempts immediately
        symbol_state.pop('last_entry_price', None)
        analyzer.break_even_triggered = False  # Reset BE flag
        return False  # Entry sequence failed

    # --- Success ---
    lg.info(f"{NEON_GREEN}Entry sequence for {signal} {symbol} completed successfully.{RESET}")
    # Reset break-even trigger flag in state for the new position
    analyzer.break_even_triggered = False
    # last_entry_price was already set after confirmation
    return True


def check_exit_conditions(
    analyzer: TradingAnalyzer, position_data: dict, current_price: Decimal, config: dict[str, Any],
    logger: logging.Logger
) -> tuple[bool, str]:
    """Checks various conditions to determine if an existing position should be closed.

    Args:
        analyzer: TradingAnalyzer instance with latest indicator data.
        position_data: Standardized dictionary of the current active position.
        current_price: Current market price (Decimal).
        config: Bot configuration dictionary.
        logger: Logger instance.

    Returns:
        Tuple (bool, str):
          - bool: True if an exit condition is met, False otherwise.
          - str: Reason for the exit (e.g., "MA Cross", "Opposite Signal"), or empty string.
    """
    if not position_data: return False, ""

    position_side = position_data.get('side')  # 'long' or 'short'
    exit_triggered = False
    exit_reason = ""

    # --- 1. MA Cross Exit ---
    if config.get("enable_ma_cross_exit", False):
        ema_short = analyzer.indicator_values.get("EMA_Short")
        ema_long = analyzer.indicator_values.get("EMA_Long")
        if ema_short is not None and ema_long is not None:
            # Check for adverse cross:
            # Long position exits if short EMA crosses BELOW long EMA
            # Short position exits if short EMA crosses ABOVE long EMA
            # Add tiny tolerance to prevent flapping on exact equality
            tolerance = ema_long * Decimal('0.0001')  # 0.01% tolerance
            if position_side == 'long' and ema_short < ema_long - tolerance:
                exit_triggered = True
                exit_reason = "Adverse MA Cross (Short < Long)"
                logger.info(f"Exit Condition Met ({analyzer.symbol}): {exit_reason}")
                return exit_triggered, exit_reason
            elif position_side == 'short' and ema_short > ema_long + tolerance:
                exit_triggered = True
                exit_reason = "Adverse MA Cross (Short > Long)"
                logger.info(f"Exit Condition Met ({analyzer.symbol}): {exit_reason}")
                return exit_triggered, exit_reason

    # --- 2. Opposite Signal Exit (Optional - Can lead to quicker flips) ---
    # Check if the latest signal from generate_trading_signal is opposite to the current position
    # Note: generate_trading_signal was already called in run_bot_for_symbol before this function
    if analyzer.signals["BUY"] == 1 or analyzer.signals["SELL"] == 1: pass

    # Uncomment below to enable exiting on opposite signal score
    # if (position_side == 'long' and latest_signal == "SELL") or \
    #    (position_side == 'short' and latest_signal == "BUY"):
    #     exit_triggered = True
    #     exit_reason = f"Opposite Signal ({latest_signal})"
    #     logger.info(f"Exit Condition Met ({analyzer.symbol}): {exit_reason}")
    #     return exit_triggered, exit_reason

    # --- Add other potential exit conditions here ---
    # Example: Exit based on specific indicator turning strongly against the position
    # Example: Exit based on time decay / holding period limit

    return exit_triggered, exit_reason


def manage_existing_position(
    exchange: ccxt.Exchange, config: dict[str, Any], logger: logging.Logger,
    analyzer: TradingAnalyzer, position_data: dict, current_price: Decimal
) -> bool:
    """Manages an existing open position based on configured rules like Break-Even SL adjustment.
    Checks for manual/external changes to SL/TP (feature TBD).
    Checks generic exit conditions (MA cross, etc.).

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance.
        analyzer: TradingAnalyzer instance with current data.
        position_data: Standardized dictionary of the current active position.
        current_price: Current market price (Decimal).

    Returns:
        bool: True if the position was closed by this management logic, False otherwise.
    """
    lg = logger
    symbol = analyzer.symbol
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state  # Access shared state

    position_side = position_data.get('side')
    entry_price = position_data.get('entryPrice')
    position_idx = position_data.get('positionIdx', 0)  # Default to 0 for One-Way

    if not position_side or not entry_price:
        lg.error(f"Position management error ({symbol}): Missing side or entry price in position data.")
        return False  # Cannot manage without essential info

    # --- Break-Even Stop Loss Management ---
    enable_be = config.get("enable_break_even", False)
    be_triggered = analyzer.break_even_triggered  # Check state from analyzer property (which reads symbol_state)
    be_force_fixed = config.get("break_even_force_fixed_sl", True)

    if enable_be and not be_triggered:
        lg.debug(f"Checking Break-Even condition for {symbol} ({position_side})...")
        be_trigger_atr_mult = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
        atr_val = analyzer.indicator_values.get("ATR")  # Decimal

        if atr_val is not None and atr_val > 0:
            profit_target_raw = atr_val * be_trigger_atr_mult
            current_profit = Decimal('0')
            if position_side == 'long':
                current_profit = current_price - entry_price
            else:  # Short
                current_profit = entry_price - current_price

            price_prec = market_info.get('price_precision_digits', 4)
            lg.debug(f"BE Check ({symbol}): CurrentProfit={current_profit:.{price_prec}f}, TargetProfit(ATR*Mult) > {profit_target_raw:.{price_prec}f} (ATR={atr_val:.{price_prec + 1}f}, Mult={be_trigger_atr_mult})")

            if current_profit >= profit_target_raw:
                lg.info(f"{NEON_GREEN}Break-Even condition met for {symbol}! Current profit ({current_profit:.{price_prec}f}) >= Target ({profit_target_raw:.{price_prec}f}). Moving SL...{RESET}")

                # Calculate BE SL Price
                be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                min_tick = analyzer.get_min_tick_size()
                if min_tick is None:
                    lg.error(f"Cannot set Break-Even SL for {symbol}: Minimum tick size unavailable.")
                    # Don't trigger BE state if we can't set the SL
                else:
                    offset_value = min_tick * Decimal(be_offset_ticks)
                    be_sl_price_raw = Decimal('0')
                    rounding = ROUND_DOWN  # Default rounding for safety
                    if position_side == 'long':
                        be_sl_price_raw = entry_price + offset_value
                        rounding = ROUND_DOWN  # Ensure SL is at or below the BE point
                    else:  # Short
                        be_sl_price_raw = entry_price - offset_value
                        rounding = ROUND_UP  # Ensure SL is at or above the BE point

                    be_sl_price = analyzer.quantize_price(be_sl_price_raw, rounding=rounding)

                    if be_sl_price is None or be_sl_price <= 0:
                         lg.error(f"Failed to calculate valid BE SL price ({be_sl_price_raw} -> {be_sl_price}) for {symbol}. Cannot set BE SL.")
                    else:
                        lg.info(f"Calculated Break-Even SL price for {symbol}: {be_sl_price:.{price_prec}f} (Entry={entry_price:.{price_prec}f}, Offset={be_offset_ticks} ticks)")

                        # Prepare protection parameters
                        protection_params = {
                            'stop_loss_price': be_sl_price,
                            'position_idx': position_idx,
                            # If forcing fixed SL, potentially cancel existing TP/TSL?
                            # Bybit's API behavior: Setting SL usually leaves TP/TSL unless explicitly cancelled.
                            # If break_even_force_fixed_sl is True, we *only* set the SL.
                            # If False, we could try to preserve existing TP/TSL, but API might reject/override.
                            # Simplest: Just set the SL. If TSL was active, BE SL might override it depending on Bybit logic.
                            # To explicitly remove TSL when BE triggers (if be_force_fixed):
                            # 'trailing_stop_price': Decimal('0') if be_force_fixed else None
                        }
                        # Decide whether to cancel TP/TSL based on config
                        if be_force_fixed:
                            lg.info(f"Break-Even forcing fixed SL. Attempting to remove existing TP/TSL for {symbol}.")
                            # Setting to 0 explicitly cancels them via set_protection_ccxt formatting logic
                            protection_params['take_profit_price'] = Decimal('0')
                            protection_params['trailing_stop_price'] = Decimal('0')
                            protection_params['trailing_active_price'] = Decimal('0')
                        # else: TP/TSL fields are omitted, set_protection_ccxt will use '0' (do nothing) for them

                        # Set the new SL (potentially modifying TP/TSL)
                        be_set_success = set_protection_ccxt(
                            exchange=exchange, symbol=symbol, **protection_params,  # Unpack prepared params
                            logger=lg, market_info=market_info
                        )

                        if be_set_success:
                            lg.info(f"{NEON_GREEN}Break-Even SL successfully set to {be_sl_price:.{price_prec}f} for {symbol}.{RESET}")
                            # Update state ONLY if SL was set successfully
                            analyzer.break_even_triggered = True  # Use setter to update shared state
                        else:
                            lg.error(f"{NEON_RED}Failed to set Break-Even SL for {symbol}. Position remains open with previous protection (if any).{RESET}")
                            # Consider if we should close position here if BE fails? Configurable?
                            # For now, just log the error and continue managing the position.
        else:
            # ATR invalid or profit target not met
            lg.debug(f"BE condition not met for {symbol}.")

    # --- Check Generic Exit Conditions (MA Cross, etc.) ---
    exit_condition_met, exit_reason = check_exit_conditions(analyzer, position_data, current_price, config, lg)

    if exit_condition_met:
        lg.warning(f"{NEON_YELLOW}Exit condition '{exit_reason}' met for {symbol}. Closing position...{RESET}")
        close_result = close_position_ccxt(exchange, symbol, position_data, lg, market_info)
        if close_result:
            lg.info(f"Position closed successfully due to exit condition: {exit_reason}")
            # Reset state after successful close
            symbol_state.pop('last_entry_price', None)
            analyzer.break_even_triggered = False
            return True  # Position was closed
        else:
            lg.error(f"{NEON_RED}Failed to close position for {symbol} despite exit condition '{exit_reason}'. Manual check required!{RESET}")
            # Keep state as is, maybe retry close next loop? Or manual intervention needed.
            return False  # Position closing failed

    # --- Position still open ---
    lg.debug(f"No management actions triggered or position closed for {symbol}. Continuing.")
    return False  # Position remains open


# --- Main Loop for a Single Symbol ---
def run_bot_for_symbol(
    exchange: ccxt.Exchange, symbol: str, config: dict[str, Any],
    bot_state: dict[str, Any], all_market_info: dict[str, dict]
) -> dict | None:
    """Executes one cycle of the trading bot logic for a single symbol.

    1. Fetches market data (price, klines, order book).
    2. Initializes TradingAnalyzer.
    3. Fetches current position.
    4. If in position:
        - Manage position (BE SL, exit conditions).
    5. If not in position:
        - Generate trading signal.
        - Check if entry is allowed (max positions, trading enabled).
        - If signal is BUY/SELL, attempt new entry.
    6. Updates and returns the symbol's state.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The standard CCXT symbol to process.
        config: Bot configuration dictionary.
        bot_state: Dictionary holding the persistent state for all symbols.
        all_market_info: Dictionary mapping symbols to their market info.

    Returns:
        The updated state dictionary for the processed symbol, or None if processing fails critically
        for this symbol in this cycle (e.g., cannot fetch market data).
    """
    symbol_logger = get_logger(symbol, is_symbol_logger=True)  # Get symbol-specific logger
    market_info = all_market_info.get(symbol)
    if not market_info:
        symbol_logger.error(f"Market info not found for {symbol}. Skipping cycle.")
        return bot_state.get(symbol, {})  # Return existing state or empty dict

    # --- Ensure state dictionary exists for this symbol ---
    symbol_state = bot_state.setdefault(symbol, {})
    # Ensure default state values (like BE flag) are present if loading old state
    symbol_state.setdefault('break_even_triggered', False)
    symbol_state.setdefault('last_entry_price', None)

    try:
        # --- Fetch Required Data ---
        # 1. Current Price
        current_price = fetch_current_price_ccxt(exchange, symbol, symbol_logger, market_info)
        if current_price is None:
            symbol_logger.error("Failed to fetch current price. Skipping cycle.")
            return symbol_state  # Return current state, try again next loop

        # 2. Kline Data (needs enough history for indicators)
        # Calculate required klines based on longest period used + buffer
        # TODO: Make this calculation more dynamic based on enabled indicators
        required_klines = max(
            config.get("ema_long_period", 21), config.get("bollinger_bands_period", 20),
            config.get("fibonacci_window", 50), config.get("rsi_period", 14),
            # Add other relevant periods here...
        ) + 50  # Add buffer
        kline_limit = max(200, required_klines)  # Fetch at least 200, or more if needed
        interval = config.get("interval", "5")  # Use validated interval string
        klines_df_raw = fetch_klines_ccxt(exchange, symbol, interval, kline_limit, symbol_logger, market_info)
        if klines_df_raw.empty:
            symbol_logger.error(f"Failed to fetch sufficient kline data ({len(klines_df_raw)} rows). Skipping cycle.")
            return symbol_state  # Return current state

        # 3. Order Book (if enabled/weighted)
        orderbook_data = None
        if config.get("indicators", {}).get("orderbook") and config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {}).get("orderbook", 0.0) != 0.0:
             orderbook_limit = config.get("orderbook_limit", 25)
             orderbook_data = fetch_orderbook_ccxt(exchange, symbol, orderbook_limit, symbol_logger, market_info)
             if orderbook_data is None:
                 symbol_logger.warning("Failed to fetch order book data. Orderbook signal component will be skipped.")

        # --- Initialize Analyzer (Calculates indicators) ---
        try:
            analyzer = TradingAnalyzer(klines_df_raw, symbol_logger, config, market_info, symbol_state)
            # Check if essential indicators (like ATR) calculated successfully
            if analyzer.indicator_values.get("ATR") is None:
                 symbol_logger.error("ATR calculation failed. Cannot proceed with risk management. Skipping cycle.")
                 return symbol_state
        except ValueError as e:
             symbol_logger.error(f"Failed to initialize TradingAnalyzer: {e}. Skipping cycle.")
             return symbol_state
        except Exception as e:
            symbol_logger.error(f"Unexpected error initializing TradingAnalyzer: {e}. Skipping cycle.", exc_info=True)
            return symbol_state

        # --- Fetch Current Position ---
        # Only fetch if market is a contract type
        current_position = None
        if market_info.get('is_contract'):
            current_position = fetch_positions_ccxt(exchange, symbol, symbol_logger, market_info)

        # --- Main Logic: Manage Position or Look for Entry ---
        position_closed_this_cycle = False
        if current_position:
            # --- In Position ---
            position_closed_this_cycle = manage_existing_position(
                exchange, config, symbol_logger, analyzer, current_position, current_price
            )
        else:
            # --- Not in Position ---
            symbol_logger.debug(f"No active position found for {symbol}. Evaluating entry signals...")

            # Generate signal based on latest data
            signal = analyzer.generate_trading_signal(current_price, orderbook_data)

            # Check Entry Conditions
            can_open_new_position = False
            if signal != "HOLD":
                # Check total positions limit
                # Need to fetch *all* positions across symbols (expensive!) or maintain a global count.
                # For now, assume max_concurrent_positions_total=1 simplifies this check.
                # TODO: Implement global position counting if max_concurrent > 1
                max_total_pos = config.get("max_concurrent_positions_total", 1)
                # Simplified check: if max=1, we can only enter if current_position is None (already checked)
                if max_total_pos == 1:  # We already know current_position is None here
                     can_open_new_position = True
                else:
                     # Complex case: Need count of *all* open positions across symbols managed by the bot
                     # Requires fetching positions for all symbols in config or maintaining global state.
                     symbol_logger.warning(f"Max concurrent positions > 1 ({max_total_pos}) requires global position tracking (not fully implemented). Allowing entry attempt for {symbol} assuming limit not reached.")
                     can_open_new_position = True  # Allow attempt, needs refinement

                # Check trading enabled flag
                if not config.get("enable_trading", False):
                    symbol_logger.warning(f"Trading signal '{signal}' generated for {symbol}, but 'enable_trading' is FALSE. No entry order will be placed.")
                    can_open_new_position = False  # Override if trading disabled

            # Attempt Entry if signal and conditions allow
            if can_open_new_position and signal != "HOLD":
                # Fetch fresh balance before attempting entry
                current_balance = fetch_balance(exchange, QUOTE_CURRENCY, symbol_logger)
                if current_balance is not None and current_balance > 0:
                    entry_successful = attempt_new_entry(
                        exchange, config, symbol_logger, analyzer, signal, current_price, current_balance
                    )
                    if entry_successful:
                        # State (BE flag, entry price) updated within attempt_new_entry
                        pass
                    else:
                        symbol_logger.error(f"Entry attempt failed for {signal} {symbol}.")
                        # State should be clean unless entry partially failed (handled in attempt_new_entry)
                else:
                    symbol_logger.error(f"Cannot attempt entry for {symbol}: Failed to fetch balance or balance is zero ({current_balance}).")
            else:
                 symbol_logger.debug(f"Holding {symbol}. Signal: {signal}, Can Open: {can_open_new_position}")

        # --- Cycle End ---
        if position_closed_this_cycle:
             # If closed, ensure state reflects no position (BE flag reset, etc.)
             symbol_state = bot_state.get(symbol, {})  # Re-fetch potentially updated state
             symbol_state['last_entry_price'] = None
             symbol_state['break_even_triggered'] = False
             symbol_logger.debug(f"Resetting state for {symbol} after position closed.")

        return symbol_state

    except ccxt.NetworkError as e:
        symbol_logger.warning(f"Network error during cycle for {symbol}: {e}. Skipping cycle.")
        return symbol_state  # Return existing state, hope connection recovers
    except ccxt.ExchangeError as e:
        symbol_logger.error(f"Exchange error during cycle for {symbol}: {e}. Skipping cycle.", exc_info=True)
        return symbol_state  # Return existing state, might be non-retryable API issue
    except Exception as e:
        symbol_logger.error(f"Unhandled exception in main loop for {symbol}: {e}", exc_info=True)
        symbol_logger.error(traceback.format_exc())  # Ensure traceback is logged
        return symbol_state  # Return existing state, needs investigation


# --- Main Execution ---
def main() -> None:
    """Main function to run the trading bot."""
    global console_log_level, LOOP_DELAY_SECONDS  # Allow modification of globals

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Enhanced CCXT Trading Bot")
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE,
        help=f"Path to the configuration file (default: {CONFIG_FILE})"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG level logging for console output."
    )
    parser.add_argument(
        "--state", type=str, default=STATE_FILE,
        help=f"Path to the state file (default: {STATE_FILE})"
    )
    # Add argument for loop delay override
    parser.add_argument(
        "--delay", type=int,
        help=f"Override loop delay in seconds (default from config or {DEFAULT_LOOP_DELAY_SECONDS}s)."
    )
    args = parser.parse_args()

    # Update console log level if --debug is set *before* setting up loggers
    if args.debug:
        console_log_level = logging.DEBUG

    # --- Logger Setup (Main Logger) ---
    main_logger = get_logger("main")
    main_logger.info(f"--- Enhanced Trading Bot v{BOT_VERSION} Starting ---")
    main_logger.info(f"Using Python {sys.version}")
    main_logger.info(f"Using CCXT {ccxt.__version__}")
    main_logger.info(f"Using pandas {pd.__version__}, pandas_ta {ta.version}")
    main_logger.info(f"Using Timezone: {TZ_NAME}")

    # --- Load Configuration ---
    main_logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config, main_logger)
    if config is None:
        # load_config already logged the critical error
        sys.exit(1)  # Exit if config validation fails

    # --- Update Loop Delay based on Config/Args ---
    if args.delay is not None:
        if 5 <= args.delay <= 3600:
            LOOP_DELAY_SECONDS = args.delay
            main_logger.info(f"Loop delay overridden by command line argument: {LOOP_DELAY_SECONDS}s")
        else:
            main_logger.warning(f"Invalid --delay value ({args.delay}). Must be between 5 and 3600. Using config value.")
            LOOP_DELAY_SECONDS = config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS)
    else:
         # Use validated loop_delay from config
         LOOP_DELAY_SECONDS = config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS)
    main_logger.info(f"Bot loop delay set to: {LOOP_DELAY_SECONDS} seconds")

    # --- Load State ---
    main_logger.info(f"Loading bot state from: {args.state}")
    bot_state = load_state(args.state, main_logger)

    # --- Initialize Exchange ---
    main_logger.info("Initializing exchange...")
    exchange = initialize_exchange(config, main_logger)
    if exchange is None:
        main_logger.critical("Failed to initialize exchange. Exiting.")
        sys.exit(1)
    main_logger.info(f"Exchange {exchange.id} initialized successfully. Sandbox: {config.get('use_sandbox')}. Account Type: {'UNIFIED' if IS_UNIFIED_ACCOUNT else 'Non-UTA'}")

    # --- Pre-fetch Market Info for all Symbols ---
    all_market_info: dict[str, dict] = {}
    symbols_to_trade = config.get("symbols", [])
    if not symbols_to_trade:
         main_logger.critical("No symbols configured in 'symbols' list. Exiting.")
         sys.exit(1)

    main_logger.info(f"Fetching market info for configured symbols: {symbols_to_trade}")
    successful_symbols = []
    for sym in symbols_to_trade:
         market = get_market_info(exchange, sym, main_logger)
         if market:
             all_market_info[sym] = market
             successful_symbols.append(sym)
             # Check if position mode requires Hedge Mode logic
             if config.get("position_mode") == "Hedge" and market.get('category') in ['linear', 'inverse']:
                  main_logger.info(f"Hedge Mode active for symbol {sym}. Ensure positionIdx is handled correctly.")
         else:
             main_logger.error(f"Could not fetch market info for {sym}. It will be skipped.")

    if not successful_symbols:
        main_logger.critical("Failed to fetch market info for ALL configured symbols. Exiting.")
        sys.exit(1)

    main_logger.info(f"Successfully loaded market info for: {successful_symbols}. Bot starting main loop...")

    # --- Main Bot Loop ---
    last_market_reload_time = time.time()
    while True:
        start_time = time.time()
        main_logger.info(f"--- Starting Bot Cycle {datetime.now(tz=TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

        # --- Periodic Market Reload ---
        if time.time() - last_market_reload_time > MARKET_RELOAD_INTERVAL_SECONDS:
            main_logger.info("Reloading exchange markets...")
            try:
                exchange.load_markets(True)  # Force reload
                last_market_reload_time = time.time()
                main_logger.info("Markets reloaded successfully.")
                # Refresh market info cache after reload
                all_market_info.clear()
                temp_successful_symbols = []
                for sym in successful_symbols:  # Only retry symbols that worked initially
                    market = get_market_info(exchange, sym, main_logger)
                    if market:
                        all_market_info[sym] = market
                        temp_successful_symbols.append(sym)
                    else:
                        main_logger.error(f"Could not refresh market info for {sym} after reload. Skipping until next reload.")
                successful_symbols = temp_successful_symbols  # Update list of active symbols
                if not successful_symbols:
                    main_logger.critical("Failed to refresh market info for ALL symbols after reload. Exiting.")
                    break  # Exit loop if no symbols are valid anymore
            except Exception as reload_err:
                main_logger.error(f"Error reloading markets: {reload_err}. Continuing with potentially stale market data.")
                # Reset reload timer to try again sooner? Or keep interval? Keep interval for now.
                last_market_reload_time = time.time()

        # --- Process Each Symbol ---
        temp_state_updates: dict[str, dict] = {}  # Store updates for this cycle
        for symbol in successful_symbols:
            symbol_start_time = time.monotonic()
            main_logger.debug(f"Processing symbol: {symbol}")
            # Pass the full bot_state, run_bot_for_symbol will manage its specific key
            updated_symbol_state = run_bot_for_symbol(
                exchange, symbol, config, bot_state, all_market_info
            )
            # Store the returned state for this symbol
            if updated_symbol_state is not None:
                 temp_state_updates[symbol] = updated_symbol_state
            else:
                 main_logger.error(f"run_bot_for_symbol returned None for {symbol}. State might not be saved correctly.")
                 # Keep the existing state for this symbol in bot_state for saving
                 temp_state_updates[symbol] = bot_state.get(symbol, {})

            symbol_duration = time.monotonic() - symbol_start_time
            main_logger.debug(f"Finished processing {symbol} in {symbol_duration:.3f}s")

        # --- Update and Save State ---
        # Merge updates from this cycle into the main bot_state
        for symbol, state_data in temp_state_updates.items():
            bot_state[symbol] = state_data  # Update the main state dict

        # Save the potentially updated state
        save_state(args.state, bot_state, main_logger)

        # --- Loop Delay ---
        end_time = time.time()
        cycle_duration = end_time - start_time
        main_logger.info(f"--- Bot Cycle Finished in {cycle_duration:.2f} seconds ---")
        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0:
            main_logger.debug(f"Waiting {wait_time:.2f} seconds before next cycle...")
            time.sleep(wait_time)


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
        # Perform any cleanup here if needed
    except Exception as e:
        # Log critical errors that weren't caught elsewhere
        main_logger = get_logger("main")  # Ensure logger exists
        main_logger.critical(f"Critical unhandled exception in main thread: {e}", exc_info=True)
        main_logger.critical(traceback.format_exc())
    finally:
        logging.shutdown()  # Ensure all log handlers are closed properly
