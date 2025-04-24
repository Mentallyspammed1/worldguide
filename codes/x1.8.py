
import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# --- Timezone Handling ---
# Use zoneinfo (Python 3.9+) if available, fallback to pytz
try:
    from zoneinfo import ZoneInfo # Preferred (Python 3.9+)
    _has_zoneinfo = True
except ImportError:
    _has_zoneinfo = False
    try:
        from pytz import timezone as ZoneInfo # Fallback (pip install pytz)
        _has_pytz = True
    except ImportError:
        _has_pytz = False
        print("Error: 'zoneinfo' (Python 3.9+) or 'pytz' package required for timezone handling.")
        print("Please install pytz: pip install pytz")
        sys.exit(1)

# --- Core Trading Libraries ---
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta

# --- Optional Enhancements ---
try:
    from colorama import Fore, Style, init
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Warning: 'colorama' package not found. Colored output will be disabled.")
    print("Install it with: pip install colorama")
    COLORAMA_AVAILABLE = False
    # Define dummy color variables if colorama is missing
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = DummyColor()
    Style = DummyColor()
    def init(*args: Any, **kwargs: Any) -> None: pass # Dummy init function

from dotenv import load_dotenv

# --- Initialization ---
try:
    # Set precision for Decimal arithmetic operations.
    # Note: This affects calculations like Decimal * Decimal.
    # Storing and retrieving Decimals maintains their inherent precision.
    getcontext().prec = 36 # Sufficient precision for most financial calculations
except Exception as e:
    print(f"Warning: Could not set Decimal precision: {e}. Using default.")

if COLORAMA_AVAILABLE:
    init(autoreset=True)    # Initialize colorama (or dummy init)
load_dotenv()           # Load environment variables from .env file

# --- Constants ---

# Bot Identity
BOT_VERSION = "1.0.2"

# Neon Color Scheme (requires colorama)
NEON_GREEN = Fore.LIGHTGREEN_EX if COLORAMA_AVAILABLE else ""
NEON_BLUE = Fore.CYAN if COLORAMA_AVAILABLE else ""
NEON_PURPLE = Fore.MAGENTA if COLORAMA_AVAILABLE else ""
NEON_YELLOW = Fore.YELLOW if COLORAMA_AVAILABLE else ""
NEON_RED = Fore.LIGHTRED_EX if COLORAMA_AVAILABLE else ""
NEON_CYAN = Fore.CYAN if COLORAMA_AVAILABLE else ""
RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ""

# --- Environment Variable Loading and Validation ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    print(f"{NEON_RED}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file.{RESET}")
    sys.exit(1)

# --- Configuration File and Paths ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
STATE_FILE = "bot_state.json"

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Timezone Configuration ---
try:
    # Attempt to load timezone from environment or default to Chicago
    TZ_NAME = os.getenv("BOT_TIMEZONE", "America/Chicago")
    TIMEZONE = ZoneInfo(TZ_NAME)
    print(f"Using Timezone: {TZ_NAME} ({'zoneinfo' if _has_zoneinfo else 'pytz'})")
except Exception as tz_err:
    print(f"{NEON_YELLOW}Warning: Could not load timezone '{TZ_NAME}': {tz_err}. Defaulting to UTC.{RESET}")
    # Default to UTC using the available library
    if _has_zoneinfo: TIMEZONE = ZoneInfo("UTC")
    elif _has_pytz: TIMEZONE = ZoneInfo("UTC")
    else: # Should not happen due to initial check, but as a safeguard
        print(f"{NEON_RED}FATAL: No timezone library available. Exiting.{RESET}")
        sys.exit(1)
    TZ_NAME = "UTC"

# --- API Interaction Constants ---
MAX_API_RETRIES = 4             # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5         # Initial delay before retrying API calls (exponential backoff)
RATE_LIMIT_BUFFER_SECONDS = 0.5 # Extra buffer added to wait time suggested by rate limit errors
MARKET_RELOAD_INTERVAL_SECONDS = 3600 # How often to reload exchange market data (1 hour)
POSITION_CONFIRM_DELAY = 10     # Seconds to wait after placing entry order before confirming position/price
MIN_TICKS_AWAY_FOR_SLTP = 3     # Minimum number of price ticks SL/TP should be away from entry price

# --- Bot Logic Constants ---
# Supported intervals for OHLCV data (ensure config uses one of these)
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
# Map bot intervals to ccxt's expected timeframe format
CCXT_INTERVAL_MAP: Dict[str, str] = {
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
DEFAULT_STOCH_WINDOW = 14 # Inner RSI period for StochRSI
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
FIB_LEVELS: List[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels

# Default loop delay (can be overridden by config)
DEFAULT_LOOP_DELAY_SECONDS = 15

# --- Global Variables ---
loggers: Dict[str, logging.Logger] = {} # Cache for logger instances
console_log_level: int = logging.INFO   # Default console log level (can be changed by args)
QUOTE_CURRENCY: str = "USDT"            # Default quote currency (updated from config)
LOOP_DELAY_SECONDS: int = DEFAULT_LOOP_DELAY_SECONDS # Actual loop delay (updated from config)
IS_UNIFIED_ACCOUNT: bool = False        # Flag to indicate if the account is UTA (detected on init)

# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive API keys/secrets."""
    REDACTED_STR: str = "***REDACTED***"
    SENSITIVE_KEYS: List[str] = [API_KEY or "UNUSED_KEY", API_SECRET or "UNUSED_SECRET"]

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive info."""
        formatted = super().format(record)
        for key in self.SENSITIVE_KEYS:
            # Only redact if key is valid and has reasonable length
            if key and len(key) > 4:
                formatted = formatted.replace(key, self.REDACTED_STR)
        return formatted

class LocalTimeFormatter(SensitiveFormatter):
    """Formatter that uses the configured local timezone for console output."""
    def converter(self, timestamp: float) -> time.struct_time:
        """Converts timestamp to local time tuple."""
        dt = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
        return dt.timetuple()

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Formats the record's creation time using the local timezone."""
        dt = datetime.fromtimestamp(record.created, tz=TIMEZONE)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            # Default format with milliseconds
            s = dt.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{s},{int(record.msecs):03d}" # Add milliseconds
        return s

def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """
    Sets up a logger with rotating file (UTC) and timezone-aware console handlers.
    Caches logger instances to avoid duplicate handlers.

    Args:
        name: The name for the logger (e.g., 'main', 'BTC/USDT').
        is_symbol_logger: If True, formats the logger name for file system safety.

    Returns:
        The configured logging.Logger instance.
    """
    global console_log_level
    # Create a safe name for file system and logger registry
    logger_instance_name = f"livebot_{name.replace('/', '_').replace(':', '-')}" if is_symbol_logger else f"livebot_{name}"

    if logger_instance_name in loggers:
        logger = loggers[logger_instance_name]
        # Update console handler level if it changed (e.g., via --debug)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level != console_log_level:
                handler.setLevel(console_log_level)
        return logger

    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)
    logger.setLevel(logging.DEBUG) # Capture all levels at the logger itself

    # --- File Handler (UTC Timestamps) ---
    try:
        file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Use UTC time for file logs for consistency
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter.converter = time.gmtime # Use UTC time converter
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Stream Handler (Local Timezone Timestamps) ---
    try:
        stream_handler = logging.StreamHandler(sys.stdout)
        tz_name_str = TZ_NAME # Use the determined timezone name
        # Include milliseconds in console output format
        # Using '%f' gets microseconds, slicing `[:-3]` keeps milliseconds
        console_fmt = f"{NEON_BLUE}%(asctime)s{RESET} [{tz_name_str}] - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
        stream_formatter = LocalTimeFormatter(console_fmt, datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3])
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(console_log_level) # Set console level based on global setting
        logger.addHandler(stream_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up stream logger for {name}: {e}{RESET}")

    logger.propagate = False # Prevent log messages from propagating to the root logger
    loggers[logger_instance_name] = logger # Cache the logger
    logger.info(f"Logger '{logger_instance_name}' initialized. File: '{os.path.basename(log_filename)}', Console Level: {logging.getLevelName(console_log_level)}")
    return logger

def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance using setup_logger."""
    return setup_logger(name, is_symbol_logger)

# --- Configuration Management ---
def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures all keys from the default config are present in the loaded config.
    Adds missing keys with default values and logs warnings. Handles nested dictionaries.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.

    Returns:
        A tuple containing:
        - The updated configuration dictionary.
        - A boolean indicating if any keys were added or if type mismatches occurred.
    """
    updated_config = config.copy()
    keys_added_or_type_mismatch = False
    for key, default_value in default_config.items():
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            keys_added_or_type_mismatch = True
            print(f"{NEON_YELLOW}Config Warning: Missing key '{key}'. Added default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_updated_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_value)
            if nested_keys_added:
                updated_config[key] = nested_updated_config
                keys_added_or_type_mismatch = True
        elif updated_config.get(key) is not None and type(default_value) != type(updated_config.get(key)):
            # Type mismatch check (allow int -> float/Decimal promotion, or str -> bool conversion)
            is_promoting_num = (isinstance(default_value, (float, Decimal)) and isinstance(updated_config.get(key), int))
            is_str_bool = isinstance(default_value, bool) and isinstance(updated_config.get(key), str)
            if not is_promoting_num and not is_str_bool:
                print(f"{NEON_YELLOW}Config Warning: Type mismatch for key '{key}'. Expected {type(default_value).__name__}, got {type(updated_config.get(key)).__name__}. Using loaded value: {repr(updated_config.get(key))}.{RESET}")
                # Note: We keep the user's value despite the type mismatch, but warn them.
                # Validation (_validate_config_values) should handle critical type conversions/errors later.
                # keys_added_or_type_mismatch = True # Optionally flag type mismatches too
    return updated_config, keys_added_or_type_mismatch

def _validate_config_values(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Validates specific critical configuration values for type, range, and format.
    Modifies the input config dictionary in place for type conversions (e.g., str to float/int).

    Args:
        config: The configuration dictionary to validate. Modifies dictionary in place.
        logger: The logger instance to use for reporting errors.

    Returns:
        True if the configuration is valid, False otherwise.
    """
    is_valid = True
    # 1. Validate Interval
    interval = config.get("interval")
    if interval not in CCXT_INTERVAL_MAP:
        logger.error(f"Config Error: Invalid 'interval' value '{interval}'. Must be one of {VALID_INTERVALS}")
        is_valid = False

    # 2. Validate Numeric Types and Ranges
    # Format: key: (expected_type, min_value, max_value)
    # Uses float for most parameters initially to allow flexibility in config file (e.g., "leverage": 10.0)
    # Converts to int where strictly necessary later.
    numeric_params: Dict[str, Tuple[type, Union[int, float], Union[int, float]]] = {
        "loop_delay": (float, 5.0, 3600.0),     # Min 5 sec delay recommended
        "risk_per_trade": (float, 0.0001, 0.5), # Risk 0.01% to 50% of balance
        "leverage": (float, 1.0, 125.0),        # Practical leverage limits (exchange may vary)
        "max_concurrent_positions_total": (float, 1.0, 100.0),
        "atr_period": (float, 2.0, 500.0),
        "ema_short_period": (float, 2.0, 500.0),
        "ema_long_period": (float, 3.0, 1000.0), # Ensure long > short is checked separately
        "rsi_period": (float, 2.0, 500.0),
        "bollinger_bands_period": (float, 5.0, 500.0),
        "bollinger_bands_std_dev": (float, 0.1, 5.0),
        "cci_window": (float, 5.0, 500.0),
        "williams_r_window": (float, 2.0, 500.0),
        "mfi_window": (float, 5.0, 500.0),
        "stoch_rsi_window": (float, 5.0, 500.0),
        "stoch_rsi_rsi_window": (float, 5.0, 500.0), # Inner RSI window for StochRSI
        "stoch_rsi_k": (float, 1.0, 100.0),
        "stoch_rsi_d": (float, 1.0, 100.0),
        "psar_af": (float, 0.001, 0.5),
        "psar_max_af": (float, 0.01, 1.0),
        "sma_10_window": (float, 2.0, 500.0),
        "momentum_period": (float, 2.0, 500.0),
        "volume_ma_period": (float, 5.0, 500.0),
        "fibonacci_window": (float, 10.0, 1000.0),
        "orderbook_limit": (float, 1.0, 200.0), # Bybit V5 limit might be 50 or 200 depending on type
        "signal_score_threshold": (float, 0.1, 10.0),
        "stoch_rsi_oversold_threshold": (float, 0.0, 50.0),
        "stoch_rsi_overbought_threshold": (float, 50.0, 100.0),
        "volume_confirmation_multiplier": (float, 0.1, 10.0),
        "scalping_signal_threshold": (float, 0.1, 10.0),
        "stop_loss_multiple": (float, 0.1, 10.0),   # Multiplier of ATR
        "take_profit_multiple": (float, 0.1, 20.0), # Multiplier of ATR
        "trailing_stop_callback_rate": (float, 0.0001, 0.5), # 0.01% to 50% (as distance from price)
        "trailing_stop_activation_percentage": (float, 0.0, 0.5), # 0% to 50% (profit needed to activate)
        "break_even_trigger_atr_multiple": (float, 0.1, 10.0), # Multiplier of ATR
        "break_even_offset_ticks": (float, 0.0, 100.0),   # Number of ticks above/below entry for BE SL
    }
    # Keys that MUST be integers for logic/API calls/pandas_ta lengths
    keys_requiring_int: List[str] = [
        "leverage", "max_concurrent_positions_total", "atr_period", "ema_short_period",
        "ema_long_period", "rsi_period", "bollinger_bands_period", "cci_window",
        "williams_r_window", "mfi_window", "stoch_rsi_window", "stoch_rsi_rsi_window",
        "stoch_rsi_k", "stoch_rsi_d", "sma_10_window", "momentum_period",
        "volume_ma_period", "fibonacci_window", "orderbook_limit", "break_even_offset_ticks",
        "loop_delay" # Loop delay should be int seconds
    ]

    for key, (expected_type, min_val, max_val) in numeric_params.items():
        value = config.get(key)
        if value is None: continue # Skip if optional or handled by ensure_keys

        try:
            # Attempt conversion to float first for range checking
            num_value_float = float(value)

            # Check range
            if not (min_val <= num_value_float <= max_val):
                logger.error(f"Config Error: '{key}' value {num_value_float} is outside the recommended range ({min_val} - {max_val}).")
                is_valid = False
                continue # Skip type conversion if range check failed

            # Store the validated numeric value back (as int if required, else float)
            if key in keys_requiring_int:
                # Check if the float value is actually an integer
                if num_value_float != int(num_value_float):
                    logger.warning(f"Config Warning: '{key}' requires an integer, but found {num_value_float}. Truncating to {int(num_value_float)}.")
                config[key] = int(num_value_float)
            else:
                config[key] = num_value_float # Store as float
        except (ValueError, TypeError):
            logger.error(f"Config Error: '{key}' value '{value}' could not be converted to a number.")
            is_valid = False

    # Specific check: EMA Long > EMA Short (after potential conversion)
    ema_long = config.get("ema_long_period")
    ema_short = config.get("ema_short_period")
    if isinstance(ema_long, int) and isinstance(ema_short, int) and ema_long <= ema_short:
        logger.error(f"Config Error: 'ema_long_period' ({ema_long}) must be greater than 'ema_short_period' ({ema_short}).")
        is_valid = False

    # 3. Validate Symbols List
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
         logger.error("Config Error: 'symbols' must be a non-empty list.")
         is_valid = False
    elif not all(isinstance(s, str) and '/' in s for s in symbols): # Basic format check
         logger.error(f"Config Error: 'symbols' list contains invalid formats. Expected 'BASE/QUOTE' or 'BASE/QUOTE:SETTLE'. Found: {symbols}")
         is_valid = False

    # 4. Validate Active Weight Set exists
    active_set = config.get("active_weight_set")
    weight_sets = config.get("weight_sets")
    if not isinstance(weight_sets, dict) or active_set not in weight_sets:
        logger.error(f"Config Error: 'active_weight_set' ('{active_set}') not found in 'weight_sets'. Available: {list(weight_sets.keys() if isinstance(weight_sets, dict) else [])}")
        is_valid = False
    elif not isinstance(weight_sets[active_set], dict):
        logger.error(f"Config Error: Active weight set '{active_set}' must be a dictionary of weights.")
        is_valid = False
    else:
        # Validate weights within the active set (should be numeric)
        for indi_key, weight_val in weight_sets[active_set].items():
            try:
                float(weight_val) # Check if convertible to float
            except (ValueError, TypeError):
                 logger.error(f"Config Error: Invalid weight value '{weight_val}' for indicator '{indi_key}' in weight set '{active_set}'. Must be numeric.")
                 is_valid = False

    # 5. Validate Boolean types (ensure they are actually bool, converting common strings)
    bool_params = [
        "enable_trading", "use_sandbox", "enable_ma_cross_exit", "enable_trailing_stop",
        "enable_break_even", "break_even_force_fixed_sl"
    ]
    for key in bool_params:
        if key in config and not isinstance(config[key], bool):
            # Try to convert common string representations
            val_str = str(config[key]).lower().strip()
            if val_str in ['true', 'yes', '1', 'on']: config[key] = True
            elif val_str in ['false', 'no', '0', 'off']: config[key] = False
            else:
                logger.error(f"Config Error: '{key}' value '{config[key]}' must be a boolean (true/false/1/0).")
                is_valid = False

    # Validate indicator enable flags (similar boolean conversion)
    if "indicators" in config and isinstance(config["indicators"], dict):
        for indi_key, indi_val in config["indicators"].items():
            if not isinstance(indi_val, bool):
                val_str = str(indi_val).lower().strip()
                if val_str in ['true', 'yes', '1', 'on']: config["indicators"][indi_key] = True
                elif val_str in ['false', 'no', '0', 'off']: config["indicators"][indi_key] = False
                else:
                    logger.error(f"Config Error: Indicator enable flag 'indicators.{indi_key}' value '{indi_val}' must be a boolean (true/false/1/0).")
                    is_valid = False
    elif "indicators" in config:
         logger.error("Config Error: 'indicators' must be a dictionary of boolean flags.")
         is_valid = False

    return is_valid

def load_config(filepath: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Loads configuration from a JSON file.
    - Creates a default config if the file doesn't exist.
    - Ensures all necessary keys are present, adding defaults for missing ones.
    - Validates critical configuration values and converts types where appropriate.
    - Saves the updated config (with defaults added) back to the file if changes were made.

    Args:
        filepath: The path to the configuration JSON file.
        logger: The logger instance for reporting.

    Returns:
        The loaded and validated configuration dictionary, or None if validation fails.
    """
    # Define the structure and default values for the configuration
    default_config: Dict[str, Any] = {
        "symbols": ["BTC/USDT:USDT"], # List of symbols to trade (ensure format matches CCXT)
        "interval": "5",              # Kline interval (e.g., "1", "5", "15", "60", "D")
        "loop_delay": DEFAULT_LOOP_DELAY_SECONDS, # Seconds between bot cycles
        "quote_currency": "USDT",     # Primary currency for balance checks and calculations
        "enable_trading": False,      # Master switch for placing actual trades
        "use_sandbox": True,          # Use Bybit's testnet environment
        "risk_per_trade": 0.01,       # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 10,               # Leverage to use for futures trades
        "max_concurrent_positions_total": 1, # Maximum number of open positions across all symbols
        "position_mode": "One-Way",   # "One-Way" or "Hedge" (Hedge mode requires careful implementation and matching API key permissions)
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
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW, # Inner RSI period for StochRSI
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,         # Parabolic SAR acceleration factor step
        "psar_max_af": DEFAULT_PSAR_MAX_AF, # Parabolic SAR max acceleration factor
        "sma_10_window": DEFAULT_SMA_10_WINDOW, # Example additional indicator period
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW, # Window for calculating Fib levels
        "orderbook_limit": 25,           # Number of levels to fetch for order book analysis
        "signal_score_threshold": 1.5,   # Threshold for combined weighted score to trigger BUY/SELL
        "stoch_rsi_oversold_threshold": 25.0,
        "stoch_rsi_overbought_threshold": 75.0,
        "volume_confirmation_multiplier": 1.5, # How much current vol must exceed MA vol
        "scalping_signal_threshold": 2.5, # Optional higher threshold for 'scalping' weight set
        "stop_loss_multiple": 1.8,       # ATR multiple for initial Stop Loss distance
        "take_profit_multiple": 0.7,     # ATR multiple for initial Take Profit distance
        "enable_ma_cross_exit": True,    # Close position if short/long EMAs cross adversely
        "enable_trailing_stop": True,    # Enable Trailing Stop Loss feature (uses Bybit's native TSL)
        "trailing_stop_callback_rate": 0.005, # TSL distance as fraction of price (e.g., 0.005 = 0.5%) - Used to calculate distance for Bybit
        "trailing_stop_activation_percentage": 0.003, # Profit % required to activate TSL (e.g., 0.003 = 0.3%) - Used to calculate activation price for Bybit
        "enable_break_even": True,       # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # ATR multiple profit needed to trigger BE
        "break_even_offset_ticks": 2,    # How many ticks *above* entry (for longs) or *below* (for shorts) to set BE SL
        "break_even_force_fixed_sl": True, # If true, BE replaces TSL; if false, BE sets SL but TSL might remain active if configured & previously set
        # --- Indicator Enable Flags ---
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Requires fetching order book data
        },
        # --- Weight Sets for Signal Generation ---
        "weight_sets": {
            # Example: Scalping focuses more on faster indicators like Momentum, StochRSI
            "scalping": {
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2,
                "orderbook": 0.15,
            },
            # Example: Default might be more balanced
            "default": {
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2,
                "orderbook": 0.1,
            }
        },
        "active_weight_set": "default" # Which set of weights to use for signal scoring
    }

    config_to_use = default_config # Start with defaults
    keys_updated_in_file = False

    if not os.path.exists(filepath):
        # Config file doesn't exist, create it with defaults
        print(f"{NEON_YELLOW}Config file not found at '{filepath}'. Creating default config...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            print(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using in-memory defaults.{RESET}")
            # Continue with default_config in memory
    else:
        # Config file exists, load it
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config_from_file = json.load(f)

            # Ensure all keys are present, adding defaults if missing
            updated_config_from_file, keys_added = _ensure_config_keys(config_from_file, default_config)
            config_to_use = updated_config_from_file

            # If keys were added, save the updated config back to the file
            if keys_added:
                keys_updated_in_file = True
                print(f"{NEON_YELLOW}Updating config file '{filepath}' with missing default keys...{RESET}")
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(config_to_use, f_write, indent=4, sort_keys=True)
                    print(f"{NEON_GREEN}Config file updated successfully.{RESET}")
                except IOError as e:
                    print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
                    keys_updated_in_file = False # Failed to save

        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Handle file reading or JSON parsing errors
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config and attempting to recreate file.{RESET}")
            config_to_use = default_config # Fallback to defaults
            try:
                # Try to recreate the file with defaults after a load error
                with open(filepath, "w", encoding="utf-8") as f_recreate:
                    json.dump(default_config, f_recreate, indent=4, sort_keys=True)
                print(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create:
                print(f"{NEON_RED}Error recreating default config file after load error: {e_create}{RESET}")
        except Exception as e:
            # Catch any other unexpected errors during loading
            print(f"{NEON_RED}Unexpected error loading configuration: {e}. Using defaults.{RESET}", exc_info=True)
            config_to_use = default_config # Fallback to defaults

    # --- Final Validation and Type Conversion ---
    # Validate the configuration values (whether loaded or default)
    # _validate_config_values modifies config_to_use in place
    if not _validate_config_values(config_to_use, logger):
        logger.critical("Configuration validation failed. Please check errors above and fix config.json. Exiting.")
        return None # Indicate failure

    logger.info("Configuration loaded and validated successfully.")
    return config_to_use

# --- State Management ---
def load_state(filepath: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Loads the bot's operational state from a JSON file.
    Handles file not found and JSON decoding errors gracefully.

    Args:
        filepath: Path to the state file.
        logger: Logger instance.

    Returns:
        A dictionary containing the loaded state, or an empty dictionary if loading fails.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"Loaded previous state from {filepath}")
                # Basic validation: Ensure it's a dictionary
                if not isinstance(state, dict):
                    logger.error(f"State file {filepath} does not contain a valid dictionary. Starting fresh.")
                    return {}
                # Optional: Add more validation here if state structure is critical
                return state
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading state file {filepath}: {e}. Starting with empty state.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}. Starting with empty state.", exc_info=True)
            return {}
    else:
        logger.info(f"No previous state file found ('{filepath}'). Starting with empty state.")
        return {}

def save_state(filepath: str, state: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Saves the bot's current operational state to a JSON file using an atomic write.
    Ensures Decimals are converted to strings for JSON compatibility.

    Args:
        filepath: Path to the state file.
        state: The dictionary containing the current state to save.
        logger: Logger instance.
    """
    temp_filepath = filepath + ".tmp"
    try:
        # Define a function to handle non-serializable types (like Decimal)
        def json_serializer(obj: Any) -> str:
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Write to temporary file first using the custom serializer
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, sort_keys=True, default=json_serializer)

        # Atomic rename/replace (os.replace is atomic on most modern systems)
        os.replace(temp_filepath, filepath)
        logger.debug(f"Saved current state to {filepath}")

    except (IOError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Error saving state file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state: {e}", exc_info=True)
    finally:
        # Clean up temp file if it still exists (e.g., due to error before replace)
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError as rm_err:
                logger.error(f"Error removing temporary state file {temp_filepath}: {rm_err}")


# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object.
    - Sets V5 API options.
    - Configures sandbox mode.
    - Loads markets.
    - Tests API credentials by fetching balance.
    - Detects account type (UTA vs. Non-UTA).

    Args:
        config: The bot's configuration dictionary.
        logger: The main logger instance.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    global QUOTE_CURRENCY, IS_UNIFIED_ACCOUNT # Allow modification of globals

    try:
        QUOTE_CURRENCY = config.get("quote_currency", "USDT")
        lg.info(f"Using Quote Currency: {QUOTE_CURRENCY}")

        # CCXT Exchange Options for Bybit V5
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'rateLimit': 120, # Default CCXT rate limit (adjust if needed, Bybit V5 has higher limits)
            'options': {
                'defaultType': 'linear', # Default to linear for futures/swaps (can be overridden by symbol format)
                'adjustForTimeDifference': True,
                'recvWindow': 10000, # Increased recvWindow for potentially slower networks
                # Timeouts (milliseconds) - Increased slightly
                'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 25000,
                'createOrderTimeout': 30000, 'fetchOrderTimeout': 25000,
                'fetchPositionsTimeout': 25000, 'cancelOrderTimeout': 25000,
                'fetchOHLCVTimeout': 25000, 'setLeverageTimeout': 25000,
                'fetchMarketsTimeout': 45000,
                'brokerId': f'EMTB{BOT_VERSION.replace(".","")}', # Shorter Broker ID example: EMTB102
                # Explicit V5 mapping (CCXT usually handles this, but can be explicit for clarity/backup)
                # Ensure these paths match the latest CCXT implementation or Bybit V5 docs if issues arise
                'versions': {
                    'public': {'GET': {'market/tickers': 'v5', 'market/kline': 'v5', 'market/orderbook': 'v5'}},
                    'private': {
                        'GET': {'position/list': 'v5', 'account/wallet-balance': 'v5', 'order/realtime': 'v5', 'order/history': 'v5'},
                        'POST': {'order/create': 'v5', 'order/cancel': 'v5', 'position/set-leverage': 'v5', 'position/trading-stop': 'v5'}
                    }
                },
                # Default options hinting CCXT to prefer V5 methods
                'default_options': {
                    'fetchPositions': 'v5', 'fetchBalance': 'v5', 'createOrder': 'v5',
                    'fetchOrder': 'v5', 'fetchTicker': 'v5', 'fetchOHLCV': 'v5',
                    'fetchOrderBook': 'v5', 'setLeverage': 'v5',
                    'private_post_position_trading_stop': 'v5', # Explicit map for protection endpoint
                },
                # Map CCXT account types to Bybit API account types
                # Used internally by CCXT and potentially for params
                'accountsByType': {
                    'spot': 'SPOT', 'future': 'CONTRACT', 'swap': 'CONTRACT',
                    'margin': 'UNIFIED', 'option': 'OPTION', 'unified': 'UNIFIED',
                    'contract': 'CONTRACT'
                },
                'accountsById': { # Reverse mapping (useful internally for CCXT)
                    'SPOT': 'spot', 'CONTRACT': 'contract', 'UNIFIED': 'unified', 'OPTION': 'option'
                },
                'bybit': { 'defaultSettleCoin': QUOTE_CURRENCY } # Helps CCXT resolve markets if needed
            }
        }

        exchange_id = 'bybit'
        exchange_class = getattr(ccxt, exchange_id)
        exchange: ccxt.Exchange = exchange_class(exchange_options)

        # --- Sandbox/Live Mode Configuration ---
        if config.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet) - No real funds involved.{RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}--- USING LIVE TRADING MODE (Real Money) --- CAUTION!{RESET}")

        # --- Load Markets ---
        lg.info(f"Connecting to {exchange.id} (Sandbox: {config.get('use_sandbox', True)})...")
        lg.info(f"Loading markets for {exchange.id}... (CCXT Version: {ccxt.__version__})")
        try:
            exchange.load_markets()
            # Store timestamp of last market load for periodic refresh
            exchange.last_load_markets_timestamp = time.time()
            lg.info(f"Markets loaded successfully for {exchange.id}. Found {len(exchange.markets)} markets.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            lg.error(f"{NEON_RED}Fatal Error loading markets: {e}. Check network connection and API endpoint status.{RESET}", exc_info=True)
            return None # Cannot proceed without markets

        # --- Test API Credentials & Check Account Type ---
        lg.info(f"Attempting initial balance fetch for {QUOTE_CURRENCY} to test credentials and detect account type...")
        balance_decimal: Optional[Decimal] = None
        account_type_detected: Optional[str] = None
        try:
            # Use helper function to check both UTA and Non-UTA balance endpoints
            temp_is_unified, balance_decimal = _check_account_type_and_balance(exchange, QUOTE_CURRENCY, lg)

            if temp_is_unified is not None:
                IS_UNIFIED_ACCOUNT = temp_is_unified # Set the global flag
                account_type_detected = "UNIFIED" if IS_UNIFIED_ACCOUNT else "CONTRACT/SPOT (Non-UTA)"
                lg.info(f"Detected Account Type: {account_type_detected}")
            else:
                lg.warning(f"{NEON_YELLOW}Could not definitively determine account type during initial balance check. Proceeding with caution.{RESET}")

            # Check if balance was successfully fetched
            if balance_decimal is not None and balance_decimal > 0:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            elif balance_decimal is not None: # Balance is zero
                 lg.warning(f"{NEON_YELLOW}Successfully connected, but initial {QUOTE_CURRENCY} balance is zero.{RESET} (Account Type: {account_type_detected or 'Unknown'})")
            else: # Balance fetch failed
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch failed (Account Type: {account_type_detected or 'Unknown'}). Check logs above. Ensure API keys have 'Read' permissions for the correct account type (Unified/Contract/Spot).{RESET}")
                 # If trading is enabled, this is a critical failure.
                 if config.get("enable_trading"):
                     lg.error(f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}")
                     return None
                 else:
                     lg.warning("Continuing in non-trading mode despite balance fetch issue.")

        except ccxt.AuthenticationError as auth_err:
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial setup: {auth_err}{RESET}")
            lg.error(f"{NEON_RED}>> Check API Key, API Secret, Permissions (Read/Trade), Account Type (Real/Testnet), and IP Whitelist.{RESET}")
            return None # Fatal error
        except Exception as balance_err:
            lg.error(f"{NEON_RED}Unexpected error during initial balance check: {balance_err}{RESET}", exc_info=True)
            if config.get("enable_trading"):
                 lg.error(f"{NEON_RED}Aborting initialization due to unexpected balance fetch error in trading mode.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}Continuing in non-trading mode despite unexpected balance fetch error: {balance_err}{RESET}")

        # If all checks passed (or warnings were accepted in non-trading mode)
        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during exchange initialization: {e}{RESET}", exc_info=True)
        return None

def _check_account_type_and_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Tuple[Optional[bool], Optional[Decimal]]:
    """
    Tries fetching balance using both UNIFIED and CONTRACT/SPOT account types
    to determine the account structure and retrieve the initial balance.

    Args:
        exchange: The CCXT exchange instance.
        currency: The currency (quote asset) to fetch the balance for.
        logger: The logger instance.

    Returns:
        A tuple containing:
        - bool or None: True if UNIFIED, False if CONTRACT/SPOT, None if undetermined.
        - Decimal or None: The available balance as Decimal, or None if fetch failed.
    """
    lg = logger
    unified_balance: Optional[Decimal] = None
    contract_spot_balance: Optional[Decimal] = None
    is_unified: Optional[bool] = None

    # --- Attempt 1: Try fetching as UNIFIED account ---
    try:
        lg.debug("Checking balance with accountType=UNIFIED...")
        params_unified = {'accountType': 'UNIFIED', 'coin': currency}
        # Use safe_ccxt_call but with fewer retries for detection purposes
        bal_info_unified = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, retry_delay=2, params=params_unified)
        parsed_balance = _parse_balance_response(bal_info_unified, currency, 'UNIFIED', lg)
        if parsed_balance is not None:
            lg.info(f"Successfully fetched balance using UNIFIED account type (Balance: {parsed_balance:.4f}).")
            unified_balance = parsed_balance
            is_unified = True
            # Return immediately if UNIFIED balance is found
            return is_unified, unified_balance
    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        # Bybit error codes/messages indicating wrong account type for the API key
        # 30086: UTA not supported / Account Type mismatch
        # 10001 + "accounttype": Parameter error related to account type
        # 10005: Permissions denied for account type
        if ("accounttype only support" in error_str or "30086" in error_str or
            "unified account is not supported" in error_str or "permission denied" in error_str or
            ("10001" in error_str and "accounttype" in error_str)):
             lg.debug("Fetching with UNIFIED failed (as expected for non-UTA or permission issue), trying CONTRACT/SPOT...")
             is_unified = False # Assume Non-UTA if this specific error occurs
        else:
             lg.warning(f"ExchangeError checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")
    except Exception as e:
         lg.warning(f"Unexpected error checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")

    # --- Attempt 2: Try fetching as CONTRACT/SPOT account (if UNIFIED failed or gave specific error) ---
    # Only proceed if unified check failed or suggested non-UTA
    if is_unified is False or is_unified is None:
        account_types_to_try = ['CONTRACT', 'SPOT'] # Check CONTRACT first, as it's more likely for futures
        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Checking balance with accountType={acc_type}...")
                params = {'accountType': acc_type, 'coin': currency}
                bal_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, retry_delay=2, params=params)
                parsed_balance = _parse_balance_response(bal_info, currency, acc_type, lg)
                if parsed_balance is not None:
                     lg.info(f"Successfully fetched balance using {acc_type} account type (Balance: {parsed_balance:.4f}).")
                     contract_spot_balance = parsed_balance
                     is_unified = False # Confirmed Non-UTA
                     return is_unified, contract_spot_balance # Return found Non-UTA balance
            except ccxt.ExchangeError as e:
                 lg.warning(f"ExchangeError checking {acc_type} balance: {e}. Trying next type...")
            except Exception as e:
                 lg.warning(f"Unexpected error checking {acc_type} balance: {e}. Trying next type...")
                 # Continue to next type unless it's the last one

    # --- Conclusion ---
    if is_unified is True and unified_balance is not None:
        return True, unified_balance
    if is_unified is False and contract_spot_balance is not None:
        return False, contract_spot_balance

    # If all attempts failed to get a definitive balance
    lg.error("Failed to determine account type OR fetch balance with common types (UNIFIED/CONTRACT/SPOT).")
    return None, None # Unknown type, no balance retrieved


# --- CCXT API Call Helper with Retries ---
def safe_ccxt_call(
    exchange: ccxt.Exchange,
    method_name: str,
    logger: logging.Logger,
    max_retries: int = MAX_API_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
    *args: Any, **kwargs: Any
) -> Any:
    """
    Safely calls a CCXT exchange method with robust retry logic.
    Handles RateLimitExceeded, NetworkError, RequestTimeout, DDoSProtection,
    and specific non-retryable ExchangeErrors based on Bybit V5 error codes.

    Args:
        exchange: The initialized CCXT exchange object.
        method_name: The name of the CCXT method to call (e.g., 'fetch_balance').
        logger: The logger instance for reporting errors and warnings.
        max_retries: Maximum number of retries for recoverable errors.
        retry_delay: Initial delay in seconds before the first retry (uses exponential backoff).
        *args: Positional arguments for the CCXT method.
        **kwargs: Keyword arguments for the CCXT method.

    Returns:
        The result of the CCXT method call if successful.

    Raises:
        ccxt.AuthenticationError: If authentication fails (not retried).
        ccxt.ExchangeError: If a non-retryable exchange error occurs.
        ccxt.NetworkError: If a network error persists after retries.
        ccxt.RequestTimeout: If a timeout persists after retries.
        RuntimeError: If max retries are exceeded without a specific CCXT exception being raised.
    """
    lg = logger
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # lg.debug(f"Calling {method_name} (Attempt {attempt + 1}/{max_retries + 1}), Args: {args}, Kwargs: {kwargs}")
            method = getattr(exchange, method_name)
            result = method(*args, **kwargs)
            # lg.debug(f"Call to {method_name} successful. Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            return result

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * (2 ** attempt) # Exponential backoff base
            suggested_wait: Optional[float] = None
            try:
                # Attempt to parse suggested wait time from Bybit error message (V5 format)
                import re
                error_msg = str(e).lower()
                # Example V5 message: "... Too many visits! Retry after 1000ms." or "... Try again in 1s."
                match_ms = re.search(r'(?:try again in|retry after)\s*(\d+)\s*ms', error_msg)
                match_s = re.search(r'(?:try again in|retry after)\s*(\d+)\s*s', error_msg)
                if match_ms:
                    suggested_wait = max(1.0, math.ceil(int(match_ms.group(1)) / 1000) + RATE_LIMIT_BUFFER_SECONDS)
                elif match_s:
                    suggested_wait = max(1.0, int(match_s.group(1)) + RATE_LIMIT_BUFFER_SECONDS)
                elif "too many visits" in error_msg or "limit" in error_msg or "frequency" in error_msg or "10006" in error_msg or "10018" in error_msg:
                    # Fallback if specific time isn't mentioned but it's clearly a rate limit code/message
                    suggested_wait = wait_time + RATE_LIMIT_BUFFER_SECONDS
            except Exception: pass # Ignore parsing errors

            final_wait = suggested_wait if suggested_wait is not None else wait_time
            if attempt < max_retries:
                lg.warning(f"Rate limit hit calling {method_name}. Retrying in {final_wait:.2f}s... (Attempt {attempt + 1}/{max_retries}) Error: {e}")
                time.sleep(final_wait)
            else:
                lg.error(f"Rate limit hit calling {method_name}. Max retries reached. Error: {e}")
                raise e # Re-raise after max retries

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = retry_delay * (2 ** attempt)
            if attempt < max_retries:
                lg.warning(f"Network/DDoS/Timeout error calling {method_name}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                 lg.error(f"Network/DDoS/Timeout error calling {method_name}. Max retries reached. Error: {e}")
                 raise e # Re-raise after max retries

        except ccxt.AuthenticationError as e:
            lg.error(f"{NEON_RED}Authentication Error calling {method_name}: {e}. Check API keys/permissions. Not retrying.{RESET}")
            raise e # Not retryable

        except ccxt.ExchangeError as e:
            last_exception = e
            bybit_code: Optional[int] = None
            ret_msg: str = str(e)
            # Try to extract Bybit's retCode and retMsg from the error string or 'info'
            # Check 'info' first if available
            bybit_info = getattr(e, 'info', None)
            if isinstance(bybit_info, dict):
                bybit_code = bybit_info.get('retCode')
                ret_msg = bybit_info.get('retMsg', str(e))
            else:
                # Fallback to parsing the string representation
                try:
                    json_part_match = re.search(r'({.*})', str(e))
                    if json_part_match:
                        details_dict = json.loads(json_part_match.group(1))
                        bybit_code = details_dict.get('retCode')
                        ret_msg = details_dict.get('retMsg', str(e))
                except (json.JSONDecodeError, IndexError, TypeError): pass # Failed to parse

            # Define known non-retryable Bybit error codes (V5) - Consult Bybit V5 docs for updates
            # See: https://bybit-exchange.github.io/docs/v5/error_code
            non_retryable_codes: List[int] = [
                # Parameter/Request Errors (10xxx)
                10001, # Parameter error (check args/kwargs/permissions)
                10002, # Request not supported
                10003, # Invalid API key or IP whitelist
                10004, # Invalid sign / Authentication failed
                10005, # Permissions denied
                # 10006 is Rate Limit, handled above, but include here as non-retryable if missed
                10009, # IP banned
                # 10010 is Request Timeout, handled above
                10016, # Service error / Maintenance (Might be temporary, but often requires waiting longer)
                10017, # Request path not found
                # 10018 is Frequency Limit, handled above
                10020, # Websocket issue (less relevant for REST)
                10029, # Request parameter validation error
                # Order/Position Errors (11xxx) - Generally indicate logic/state issues
                110001, # Order placement failed (generic)
                110003, # Invalid price
                110004, # Invalid quantity
                110005, # Qty too small
                110006, # Qty too large
                110007, # Insufficient balance
                110008, # Cost too small
                110009, # Cost too large
                110010, # Invalid order type
                110011, # Invalid side
                110012, # Invalid timeInForce
                110013, # Price exceeds deviation limits
                110014, # OrderId not found or invalid
                110015, # Order already cancelled
                110016, # Order already filled
                110017, # Price/Qty precision error
                110019, # Cannot amend market orders
                110020, # Position status prohibits action (e.g., liquidation)
                110021, # Risk limit exceeded
                110022, # Invalid leverage
                110024, # Position not found
                110025, # Position idx error (Hedge Mode specific)
                110028, # Reduce-only order would increase position
                110031, # Order amount exceeds open limit
                110033, # Cannot set leverage in cross margin mode
                110036, # Cross/Isolated mode mismatch
                110040, # TP/SL order parameter error
                110041, # TP/SL requires position
                110042, # TP/SL price invalid
                # 110043 is 'Leverage not modified' - Treat as success below
                110044, # Margin mode not modified
                110045, # Qty exceeds risk limit
                110047, # Cannot set TP/SL for Market orders during creation
                110051, # Position zero, cannot close
                110067, # Feature requires UTA Pro
                # Account/Risk/Position Errors (17xxx)
                170001, # Internal error affecting position
                170007, # Risk limit error
                170019, # Margin call / Liquidation status
                170131, # TP/SL price invalid
                170132, # TP/SL order triggers liquidation
                170133, # Cannot set TP/SL/TSL (generic)
                170140, # TP/SL requires position
                # Account Type Errors (3xxxx)
                30086,  # UTA not supported / Account Type mismatch
                30087,  # UTA feature unavailable
                # Add more critical, non-recoverable codes as identified from Bybit docs
            ]

            if bybit_code in non_retryable_codes:
                # Special handling for "Leverage not modified" - treat as success
                if bybit_code == 110043 and method_name == 'set_leverage':
                    lg.info(f"Leverage already set as requested (Code 110043) when calling {method_name}. Ignoring error.")
                    return {} # Return empty dict, often treated as success by calling code

                extra_info = ""
                if bybit_code == 10001: extra_info = f"{NEON_YELLOW} Hint: Check API call parameters ({args=}, {kwargs=}) or API key permissions/account type.{RESET}"
                elif bybit_code == 110007:
                    balance_currency = QUOTE_CURRENCY # Default guess
                    try: # Try harder to guess the currency being checked
                         if 'params' in kwargs and 'coin' in kwargs['params']: balance_currency = kwargs['params']['coin']
                         elif 'symbol' in kwargs and isinstance(kwargs['symbol'], str): balance_currency = kwargs['symbol'].split('/')[1].split(':')[0]
                         elif len(args) > 0 and isinstance(args[0], str) and '/' in args[0]: balance_currency = args[0].split('/')[1].split(':')[0]
                    except Exception: pass
                    extra_info = f"{NEON_YELLOW} Hint: Check available {balance_currency} balance in the correct account (UTA/Contract/Spot).{RESET}"
                elif bybit_code == 30086 or (bybit_code == 10001 and "accounttype" in ret_msg.lower()):
                     extra_info = f"{NEON_YELLOW} Hint: Check 'accountType' param (UNIFIED vs CONTRACT/SPOT) matches your account/API key permissions.{RESET}"
                elif bybit_code == 110025:
                     extra_info = f"{NEON_YELLOW} Hint: Check 'positionIdx' parameter (0 for One-Way, 1/2 for Hedge Mode) and ensure it matches your account's Position Mode setting.{RESET}"
                elif bybit_code == 110017 or bybit_code == 110005 or bybit_code == 110006:
                     extra_info = f"{NEON_YELLOW} Hint: Check price/amount precision and limits against market data.{RESET}"
                elif bybit_code == 170140:
                     extra_info = f"{NEON_YELLOW} Hint: TP/SL/Protection setting requires an active position.{RESET}"

                lg.error(f"{NEON_RED}Non-retryable Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Not retrying.{RESET}{extra_info}")
                raise e # Re-raise the non-retryable error

            else:
                # Unknown or potentially temporary exchange error, retry
                if attempt < max_retries:
                    lg.warning(f"{NEON_YELLOW}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Retrying... (Attempt {attempt + 1}/{max_retries}){RESET}")
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    lg.error(f"{NEON_RED}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Max retries reached.{RESET}")
                    raise e # Re-raise after max retries

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected Error calling {method_name}: {e}. Not retrying.{RESET}", exc_info=True)
            raise e # Re-raise unexpected errors immediately

    # Max Retries Reached for recoverable errors
    lg.error(f"{NEON_RED}Max retries ({max_retries}) reached for {method_name}. Last error: {last_exception}{RESET}")
    if isinstance(last_exception, Exception):
        raise last_exception # Re-raise the last specific exception
    else:
        # Should not happen if last_exception is always assigned, but as a fallback
        raise RuntimeError(f"Max retries reached for {method_name} without specific exception.")


# --- Market Info Helper Functions ---
def _determine_category(market: Dict[str, Any]) -> Optional[str]:
    """
    Determines the Bybit V5 category ('linear', 'inverse', 'spot', 'option')
    from CCXT market info dictionary. Prioritizes explicit flags then infers.

    Args:
        market: The market dictionary from ccxt.exchange.markets.

    Returns:
        The category string or None if undetermined.
    """
    # Prioritize explicit type/category flags if present in market['info'] (exchange-specific)
    info = market.get('info', {})
    category_info = info.get('category') # V5 specific field
    if category_info in ['linear', 'inverse', 'spot', 'option']:
        return category_info

    # Use CCXT standard flags
    market_type = market.get('type') # 'spot', 'swap', 'future', 'option'
    is_linear = market.get('linear', False)
    is_inverse = market.get('inverse', False)
    is_spot = market.get('spot', False)
    is_option = market.get('option', False)

    if is_spot or market_type == 'spot': return 'spot'
    if is_option or market_type == 'option': return 'option'
    # For contracts (swap/future)
    if market_type in ['swap', 'future']:
        if is_linear: return 'linear'
        if is_inverse: return 'inverse'
        # Fallback inference based on settle asset vs quote asset
        settle_asset = market.get('settle', '').upper()
        quote_asset = market.get('quote', '').upper()
        if settle_asset and quote_asset:
            # Linear settles in quote (e.g., USDT, USDC)
            # Inverse settles in base (e.g., BTC, ETH)
            return 'linear' if settle_asset == quote_asset else 'inverse'

    # If category couldn't be determined
    return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Retrieves and processes market information for a symbol from loaded CCXT markets.
    Extracts key details like precision, limits, contract size, and V5 category.

    Args:
        exchange: Initialized CCXT exchange object with loaded markets.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing processed market info, or None if symbol not found/invalid.
        Keys include: 'symbol', 'id', 'base', 'quote', 'settle', 'type', 'category',
                      'is_contract', 'inverse', 'contract_size' (Decimal),
                      'min_tick_size' (Decimal), 'price_precision_digits' (int),
                      'amount_precision_digits' (int), 'min_order_amount' (Decimal),
                      'max_order_amount' (Decimal), 'min_price' (Decimal), 'max_price' (Decimal),
                      'min_order_cost' (Decimal), 'max_order_cost' (Decimal), 'raw_market_data'.
    """
    lg = logger
    if not exchange.markets:
        lg.error(f"Cannot get market info for {symbol}: Markets not loaded.")
        return None
    try:
        market = exchange.market(symbol)
        if not market:
            lg.error(f"Symbol '{symbol}' not found in loaded markets.")
            return None

        category = _determine_category(market)
        if category is None:
            lg.warning(f"Could not reliably determine V5 category for symbol {symbol}. Proceeding with caution. Market data: {market}")

        # Extract precision details safely
        price_precision_val = market.get('precision', {}).get('price') # This is the tick size
        amount_precision_val = market.get('precision', {}).get('amount') # This is the amount step size

        # Helper to calculate digits from precision value (e.g., 0.01 -> 2 digits)
        def get_digits(precision_val: Optional[Union[float, int, str]]) -> int:
            if precision_val is None or float(precision_val) <= 0: return 8 # Default
            try:
                # Use Decimal for accurate calculation, especially for very small numbers
                # log10(0.01) = -2. negate -> 2 digits
                # log10(1) = 0. negate -> 0 digits
                # log10(10) = 1. negate -> -1 (handle this case -> 0 digits)
                digits = int(Decimal(str(precision_val)).log10().copy_negate())
                return max(0, digits) # Ensure non-negative digits
            except (InvalidOperation, ValueError, TypeError) as e:
                 lg.warning(f"Could not calculate digits from precision {precision_val}: {e}")
                 return 8 # Default

        price_digits = get_digits(price_precision_val)
        amount_digits = get_digits(amount_precision_val)

        # Extract limits safely, converting to Decimal where appropriate
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        cost_limits = limits.get('cost', {})

        # Helper to convert value to Decimal safely
        def to_decimal_safe(val: Any) -> Optional[Decimal]:
            if val is None: return None
            try:
                d = Decimal(str(val))
                return d if d.is_finite() else None
            except (InvalidOperation, TypeError): return None

        min_amount = to_decimal_safe(amount_limits.get('min'))
        max_amount = to_decimal_safe(amount_limits.get('max'))
        min_price = to_decimal_safe(price_limits.get('min'))
        max_price = to_decimal_safe(price_limits.get('max'))
        min_cost = to_decimal_safe(cost_limits.get('min'))
        max_cost = to_decimal_safe(cost_limits.get('max'))

        # Contract-specific details
        contract_size = to_decimal_safe(market.get('contractSize', '1')) or Decimal('1') # Default to 1
        is_contract = category in ['linear', 'inverse']
        is_inverse = category == 'inverse'

        market_details = {
            'symbol': symbol,
            'id': market.get('id'), # Exchange's internal market ID
            'base': market.get('base'),
            'quote': market.get('quote'),
            'settle': market.get('settle'),
            'type': market.get('type'), # spot, swap, future
            'category': category,       # linear, inverse, spot, option (best guess)
            'is_contract': is_contract,
            'inverse': is_inverse,
            'contract_size': contract_size,
            'min_tick_size': to_decimal_safe(price_precision_val), # Store tick size as Decimal
            'amount_step_size': to_decimal_safe(amount_precision_val), # Store amount step size as Decimal
            'price_precision_digits': price_digits,
            'amount_precision_digits': amount_digits,
            'min_order_amount': min_amount,
            'max_order_amount': max_amount,
            'min_price': min_price,
            'max_price': max_price,
            'min_order_cost': min_cost,
            'max_order_cost': max_cost,
            'raw_market_data': market # Include original market data for debugging if needed
        }
        lg.debug(f"Market Info for {symbol}: Cat={category}, Tick={market_details['min_tick_size']}, AmtPrec={amount_digits}, ContSize={contract_size}")
        return market_details

    except ccxt.BadSymbol as e:
        lg.error(f"Error getting market info for '{symbol}': {e}")
        return None
    except Exception as e:
        lg.error(f"Unexpected error processing market info for {symbol}: {e}", exc_info=True)
        return None

# --- Data Fetching Wrappers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Decimal]:
    """
    Fetches the current ticker price using V5 API via safe_ccxt_call.
    Parses the 'lastPrice' field from the V5 response structure.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        The current price as Decimal, or None if fetch/parsing fails.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID
    if not category:
        lg.error(f"Cannot fetch price for {symbol}: Category unknown in market_info."); return None

    try:
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching ticker for {symbol} with params: {params}")
        # Pass symbol for CCXT standard resolution, params for V5 API call
        ticker = safe_ccxt_call(exchange, 'fetch_ticker', lg, symbol=symbol, params=params)

        # Bybit V5 ticker response structure might be nested under 'info' -> 'result' -> 'list'
        ticker_data = None
        price_str = None
        if isinstance(ticker, dict):
            # Check V5 structure (info -> result -> list -> [item]) - Common for category requests
            if ('info' in ticker and isinstance(ticker['info'], dict) and
                ticker['info'].get('retCode') == 0 and 'result' in ticker['info'] and
                isinstance(ticker['info']['result'], dict) and 'list' in ticker['info']['result'] and
                isinstance(ticker['info']['result']['list'], list) and len(ticker['info']['result']['list']) > 0):

                # Find the ticker matching the market_id within the list
                for item in ticker['info']['result']['list']:
                    if isinstance(item, dict) and item.get('symbol') == market_id:
                        ticker_data = item
                        price_str = ticker_data.get('lastPrice') # V5 field
                        lg.debug(f"Parsed price from V5 list structure for {symbol} (MarketID: {market_id})")
                        break # Found the matching ticker
            # Check alternative V5 structure (info -> result -> single item) - Less common?
            elif ('info' in ticker and isinstance(ticker['info'], dict) and
                  ticker['info'].get('retCode') == 0 and 'result' in ticker['info'] and
                  isinstance(ticker['info']['result'], dict) and not ticker['info']['result'].get('list')):
                 ticker_data = ticker['info']['result']
                 # Check if the symbol matches (in case API returns unexpected data)
                 if ticker_data.get('symbol') == market_id:
                     price_str = ticker_data.get('lastPrice') # V5 field
                     lg.debug(f"Parsed price from V5 single result structure for {symbol}")
                 else:
                      lg.warning(f"Ticker result symbol '{ticker_data.get('symbol')}' mismatch for {market_id}")
            # Fallback to standard CCXT 'last' field if V5 parsing fails
            elif 'last' in ticker and ticker['last'] is not None:
                 price_str = str(ticker['last'])
                 lg.debug(f"Parsed price from standard CCXT 'last' field for {symbol}")

        if price_str is None or price_str == "":
            lg.warning(f"Could not extract valid 'lastPrice' or 'last' for {symbol}. Ticker: {ticker}")
            return None

        price_dec = Decimal(price_str)
        if price_dec.is_finite() and price_dec > 0:
            lg.debug(f"Current price for {symbol}: {price_dec}")
            return price_dec
        else:
            lg.error(f"Invalid price ('{price_str}') received for {symbol}.")
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
         lg.error(f"Error converting fetched price '{price_str}' to Decimal for {symbol}: {e}")
         return None
    except Exception as e:
        # Catch errors from safe_ccxt_call or other issues
        lg.error(f"Error fetching current price for {symbol}: {e}", exc_info=True)
        return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger, market_info: Dict) -> pd.DataFrame:
    """
    Fetches OHLCV data using V5 API via safe_ccxt_call.
    Returns a DataFrame with UTC timestamp index and Decimal OHLCV columns.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        timeframe: Bot interval (e.g., "5").
        limit: Number of candles to fetch.
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        Pandas DataFrame with 'timestamp' (UTC datetime index) and
        'open', 'high', 'low', 'close', 'volume' (Decimal) columns.
        Returns an empty DataFrame on failure.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)

    if not category:
        lg.error(f"Cannot fetch klines for {symbol}: Category unknown in market_info."); return pd.DataFrame()
    if not ccxt_timeframe:
        lg.error(f"Invalid timeframe '{timeframe}' provided for {symbol}. Valid: {VALID_INTERVALS}"); return pd.DataFrame()

    try:
        # Bybit V5 kline endpoint takes category and symbol
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching {limit} klines for {symbol} ({ccxt_timeframe}) with params: {params}")
        # Limit might need adjustment based on exchange max (e.g., 1000 for Bybit V5 kline)
        safe_limit = min(limit, 1000)
        if limit > 1000: lg.warning(f"Requested kline limit {limit} > max 1000. Fetching {safe_limit}.")

        # CCXT fetch_ohlcv maps to the correct V5 endpoint
        ohlcv = safe_ccxt_call(exchange, 'fetch_ohlcv', lg, symbol=symbol, timeframe=ccxt_timeframe, limit=safe_limit, params=params)

        if not ohlcv:
            lg.warning(f"fetch_ohlcv returned empty data for {symbol} ({ccxt_timeframe}).")
            return pd.DataFrame()

        # Convert to DataFrame and set column names
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"Kline data for {symbol} resulted in an empty DataFrame.")
             return df

        # Convert timestamp to datetime (UTC) and set as index
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
        except Exception as dt_err:
             lg.error(f"Error converting timestamp column for {symbol}: {dt_err}")
             return pd.DataFrame() # Cannot proceed without valid index

        # Convert OHLCV columns to Decimal for precision
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns: continue # Skip if column somehow missing
            try:
                # Convert to string first to avoid potential float inaccuracies before Decimal
                df[col] = df[col].astype(str).apply(Decimal)
            except (InvalidOperation, TypeError, ValueError) as e:
                 lg.error(f"Error converting column '{col}' to Decimal for {symbol}: {e}. Coercing errors to NaN.")
                 # Coerce errors during conversion to NaN, then potentially handle/drop NaNs later
                 # Use pd.to_numeric to handle potential non-numeric strings robustly
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 # Apply Decimal conversion only to valid numeric (non-NaN) values
                 df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else pd.NA)
                 # Ensure the column type remains compatible if NAs were introduced
                 if df[col].isnull().any():
                     df[col] = df[col].astype('object') # Use object dtype if NAs are present

        # Optional: Drop rows with any NaN/NA in critical columns if needed
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Drop rows where essential prices are missing
        if len(df) < initial_len:
            lg.warning(f"Dropped {initial_len - len(df)} rows with NaN values in OHLC columns for {symbol}.")

        if df.empty:
            lg.warning(f"DataFrame became empty after NaN drop for {symbol}.")

        lg.debug(f"Fetched and processed {len(df)} klines for {symbol}.")
        return df

    except Exception as e:
        lg.error(f"Error fetching/processing klines for {symbol}: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """
    Fetches order book data using V5 API via safe_ccxt_call.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        limit: Number of order book levels to fetch (e.g., 25, 50).
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        A dictionary containing 'bids' and 'asks' lists (each item is [price_str, amount_str]),
        or None if fetch fails. Prices/amounts are strings as returned by the API.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID
    if not category:
        lg.error(f"Cannot fetch orderbook for {symbol}: Category unknown in market_info."); return None

    try:
        # Bybit V5 orderbook endpoint uses category and symbol, limit in params
        params = {'category': category, 'symbol': market_id, 'limit': limit}
        lg.debug(f"Fetching order book (limit {limit}) for {symbol} with params: {params}")
        # CCXT fetch_order_book should map to the correct V5 endpoint
        orderbook = safe_ccxt_call(exchange, 'fetch_order_book', lg, symbol=symbol, limit=limit, params=params)

        # Validate the structure
        if (orderbook and isinstance(orderbook, dict) and
            'bids' in orderbook and isinstance(orderbook['bids'], list) and
            'asks' in orderbook and isinstance(orderbook['asks'], list)):

            # Optional: Convert prices/amounts to Decimal here if needed immediately downstream
            # Example conversion (can be slow for large orderbooks):
            # try:
            #     orderbook['bids'] = [[Decimal(str(p)), Decimal(str(a))] for p, a in orderbook['bids'] if len(p) == 2]
            #     orderbook['asks'] = [[Decimal(str(p)), Decimal(str(a))] for p, a in orderbook['asks'] if len(p) == 2]
            # except (InvalidOperation, TypeError, ValueError, IndexError) as e:
            #     lg.error(f"Error converting orderbook values to Decimal for {symbol}: {e}")
            #     return None # Failed conversion

            # Check if data was returned (lists might be empty if market is illiquid)
            if orderbook['bids'] or orderbook['asks']:
                 lg.debug(f"Fetched order book for {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook
            else:
                 lg.warning(f"Order book fetch for {symbol} returned empty bids/asks lists.")
                 return orderbook # Return empty book, might be valid state
        else:
            lg.warning(f"Failed to fetch valid order book structure for {symbol}. Response: {orderbook}")
            return None
    except Exception as e:
        lg.error(f"Error fetching order book for {symbol}: {e}", exc_info=True)
        return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency using safe_ccxt_call.
    Adapts the request based on the detected account type (UTA vs. Non-UTA).

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency symbol (e.g., 'USDT').
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetch fails or balance is zero/invalid.
    """
    lg = logger
    account_types_to_try: List[str] = []

    # Determine which account types to query based on global flag
    if IS_UNIFIED_ACCOUNT:
        account_types_to_try = ['UNIFIED']
        lg.debug(f"Fetching balance specifically for UNIFIED account ({currency}).")
    else:
        # For Non-UTA, CONTRACT usually holds futures balance, SPOT for spot balance.
        # Query both as funds might be in either for trading.
        account_types_to_try = ['CONTRACT', 'SPOT']
        lg.debug(f"Fetching balance for Non-UTA account ({currency}), trying types: {account_types_to_try}.")

    last_exception: Optional[Exception] = None
    parsed_balance: Optional[Decimal] = None
    total_available_balance = Decimal('0') # Accumulate balance across relevant types if Non-UTA

    # Outer retry loop for network/rate limit issues
    for attempt in range(MAX_API_RETRIES + 1):
        balance_found_in_cycle = False
        error_in_cycle = False

        # Inner loop to try different account types
        for acc_type in account_types_to_try:
            try:
                params = {'accountType': acc_type, 'coin': currency}
                lg.debug(f"Fetching balance with params={params} (Attempt {attempt + 1})")
                # Use safe_ccxt_call with 0 inner retries; outer loop handles retries for network issues
                balance_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=0, params=params)

                current_type_balance = _parse_balance_response(balance_info, currency, acc_type, lg)

                if current_type_balance is not None:
                    balance_found_in_cycle = True
                    if IS_UNIFIED_ACCOUNT:
                        # For UTA, the first successful fetch is the definitive balance
                        total_available_balance = current_type_balance
                        lg.info(f"Available {currency} balance (UNIFIED): {total_available_balance:.4f}")
                        return total_available_balance
                    else:
                        # For Non-UTA, sum balances from CONTRACT and SPOT
                        lg.info(f"Found {currency} balance ({acc_type}): {current_type_balance:.4f}")
                        total_available_balance += current_type_balance
                else:
                    lg.debug(f"Balance for {currency} not found or parsing failed for type {acc_type}.")

            except ccxt.ExchangeError as e:
                # Non-retryable errors (like auth, param) should have been raised by safe_ccxt_call
                # Log other exchange errors and continue to next type if applicable
                lg.debug(f"Exchange error fetching balance type {acc_type}: {e}. Trying next type if available.")
                last_exception = e
                continue # Try the next account type

            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                 lg.warning(f"Network/RateLimit error during balance fetch type {acc_type}: {e}")
                 last_exception = e
                 error_in_cycle = True
                 break # Break inner loop, let outer loop handle retry

            except Exception as e:
                 lg.error(f"Unexpected error during balance fetch type {acc_type}: {e}", exc_info=True)
                 last_exception = e
                 error_in_cycle = True
                 break # Break inner loop, let outer loop handle retry

        # --- After Inner Loop ---
        if not IS_UNIFIED_ACCOUNT and balance_found_in_cycle and not error_in_cycle:
            # If Non-UTA, and we successfully checked all types without errors, return the sum
            lg.info(f"Total available {currency} balance (Non-UTA: CONTRACT+SPOT): {total_available_balance:.4f}")
            return total_available_balance

        if error_in_cycle: # If broke from inner loop due to network/rate limit/unexpected error
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} encountered recoverable error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                total_available_balance = Decimal('0') # Reset sum for retry
                continue # Continue outer retry loop
            else:
                 lg.error(f"{NEON_RED}Max retries reached fetching balance for {currency} after network/unexpected error. Last error: {last_exception}{RESET}")
                 return None

        # If inner loop completed without finding balance (and without errors breaking it)
        if not balance_found_in_cycle:
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} failed to find/parse balance for type(s): {account_types_to_try}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue # Retry outer loop
            else:
                 # Only error if UTA, for Non-UTA zero balance is possible across both
                 if IS_UNIFIED_ACCOUNT:
                    lg.error(f"{NEON_RED}Max retries reached. Failed to fetch/parse balance for {currency} (UNIFIED). Last error: {last_exception}{RESET}")
                    return None
                 else:
                    lg.info(f"Could not find any {currency} balance in CONTRACT or SPOT accounts after retries. Total balance: 0.0000")
                    return Decimal('0') # Return 0 if non-UTA and nothing found

    # Fallback if logic somehow exits loop without returning
    lg.error(f"{NEON_RED}Balance fetch logic completed unexpectedly without returning a value for {currency}.{RESET}")
    return None

def _parse_balance_response(balance_info: Optional[Dict], currency: str, account_type_checked: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Parses the raw response from CCXT's fetch_balance, adapting to Bybit V5 structure.
    Prioritizes 'availableToWithdraw' (for UTA) or 'availableBalance' (Non-UTA Contract/Spot)
    within the nested V5 response structure. Falls back to CCXT standard fields.

    Args:
        balance_info: The raw dictionary returned by safe_ccxt_call('fetch_balance').
        currency: The currency symbol (e.g., 'USDT') to look for.
        account_type_checked: The 'accountType' used in the request ('UNIFIED', 'CONTRACT', 'SPOT').
        logger: Logger instance.

    Returns:
        The available balance as Decimal, or None if not found or invalid.
    """
    if not balance_info:
        logger.debug(f"Parsing balance: Received empty balance_info for {currency} ({account_type_checked}).")
        return None
    lg = logger
    available_balance_str: Optional[str] = None
    field_used: str = "N/A"

    try:
        # --- Strategy 1: Prioritize Bybit V5 structure (info -> result -> list -> coin[]) ---
        # Check for the expected V5 structure returned by fetch_balance
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
                                # Priority:
                                # 1. availableToWithdraw (Primary available balance for UTA, may exist for others)
                                # 2. availableBalance (Available balance for Non-UTA Contract/Spot, might be present in UTA)
                                # 3. walletBalance (Total balance, less useful for available margin)
                                free_val = coin_data.get('availableToWithdraw')
                                field_used = 'availableToWithdraw'
                                if free_val is None or str(free_val).strip() == "":
                                    free_val = coin_data.get('availableBalance')
                                    field_used = 'availableBalance'
                                    if free_val is None or str(free_val).strip() == "":
                                         # Fallback to walletBalance only if others are missing (less preferred)
                                         lg.debug(f"Available balance fields missing for {currency} ({account_type_checked}), checking 'walletBalance'.")
                                         free_val = coin_data.get('walletBalance')
                                         field_used = 'walletBalance'

                                # Check if we found a non-empty value
                                if free_val is not None and str(free_val).strip() != "":
                                    available_balance_str = str(free_val)
                                    lg.debug(f"Parsed balance from Bybit V5 ({account_type_checked} -> {currency}): Value='{available_balance_str}' (Field: '{field_used}')")
                                    break # Found currency in this account's coin list
                        if available_balance_str is not None:
                            break # Found currency in this account type
            if available_balance_str is None:
                 lg.debug(f"Currency '{currency}' not found within Bybit V5 list structure for account type '{account_type_checked}'.")

        # --- Strategy 2: Fallback to standard CCXT 'free' balance structure ---
        # This might be populated by CCXT even if V5 structure exists
        elif available_balance_str is None and currency in balance_info and isinstance(balance_info.get(currency), dict):
            free_val = balance_info[currency].get('free')
            if free_val is not None:
                available_balance_str = str(free_val)
                field_used = "ccxt_free"
                lg.debug(f"Parsed balance via standard CCXT structure ['{currency}']['free']: {available_balance_str}")

        # --- Strategy 3: Fallback to top-level 'free' dictionary (less common for specific currency) ---
        elif available_balance_str is None and 'free' in balance_info and isinstance(balance_info.get('free'), dict):
             free_val = balance_info['free'].get(currency)
             if free_val is not None:
                 available_balance_str = str(free_val)
                 field_used = "ccxt_top_level_free"
                 lg.debug(f"Parsed balance via top-level 'free' dictionary ['free']['{currency}']: {available_balance_str}")

        # --- Conversion and Validation ---
        if available_balance_str is None:
            lg.debug(f"Could not extract available balance for {currency} from response structure ({account_type_checked}). Response info keys: {balance_info.get('info', {}).keys() if isinstance(balance_info.get('info'), dict) else 'N/A'}")
            return None

        # Attempt conversion to Decimal
        final_balance = Decimal(available_balance_str)
        if final_balance.is_finite() and final_balance >= 0:
            # Return the balance, could be 0
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
    """
    Analyzes market data using pandas_ta.Strategy to generate weighted trading signals.
    Calculates technical indicators, Fibonacci levels, and provides risk management values.
    Uses Decimal for internal calculations involving prices and quantities.
    Manages symbol-specific state like break-even status via a shared dictionary.
    """
    def __init__(
        self,
        df_raw: pd.DataFrame, # DataFrame with Decimal OHLCV columns
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
        symbol_state: Dict[str, Any], # Mutable state dict shared with main loop
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df_raw: DataFrame containing OHLCV data (must have Decimal columns).
            logger: Logger instance for this symbol.
            config: The main configuration dictionary.
            market_info: Processed market information for the symbol.
            symbol_state: Mutable dictionary holding state for this symbol
                          (e.g., 'break_even_triggered', 'last_entry_price').

        Raises:
            ValueError: If df_raw, market_info, or symbol_state is invalid.
        """
        if df_raw is None or df_raw.empty:
            raise ValueError("TradingAnalyzer requires a non-empty raw DataFrame.")
        if not market_info:
            raise ValueError("TradingAnalyzer requires valid market_info.")
        if symbol_state is None: # Check for None explicitly
            raise ValueError("TradingAnalyzer requires a valid symbol_state dictionary.")

        self.df_raw = df_raw # Keep raw Decimal DF for precise checks (e.g., Fib, price checks)
        self.df = df_raw.copy() # Work on a copy for TA calculations requiring floats
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol: str = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval: str = config.get("interval", "UNKNOWN_INTERVAL")
        self.symbol_state = symbol_state # Store reference to the mutable state dict

        # --- Internal State ---
        self.indicator_values: Dict[str, Optional[Decimal]] = {} # Stores latest indicator values as Decimals
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1} # Current signal state
        self.active_weight_set_name: str = config.get("active_weight_set", "default")
        self.weights: Dict[str, Union[float, str]] = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated Fibonacci levels
        self.ta_strategy: Optional[ta.Strategy] = None # pandas_ta strategy object
        self.ta_column_map: Dict[str, str] = {} # Map generic names (e.g., "EMA_Short") to pandas_ta column names (e.g., "EMA_9")

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' is empty or not found for {self.symbol}. Signal generation will be disabled.{RESET}")

        # --- Data Preparation for pandas_ta ---
        # Convert necessary columns in the copied DataFrame (self.df) to float
        self._convert_df_for_ta()

        # --- Initialize and Calculate Indicators ---
        if not self.df.empty:
            self._define_ta_strategy()
            self._calculate_all_indicators() # Operates on self.df (float version)
            self._update_latest_indicator_values() # Populates self.indicator_values with Decimals from self.df
            self.calculate_fibonacci_levels() # Calculate initial Fib levels using self.df_raw (Decimal version)
        else:
            logger.warning(f"DataFrame is empty after float conversion for {self.symbol}. Cannot calculate indicators.")


    def _convert_df_for_ta(self) -> None:
        """
        Converts necessary DataFrame columns (OHLCV) in the working copy `self.df`
        to float64 for compatibility with pandas_ta. Handles potential errors gracefully.
        """
        if self.df.empty: return
        try:
            cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
            for col in cols_to_convert:
                 if col in self.df.columns:
                    # Check if conversion is needed (not already float64)
                    if not pd.api.types.is_float_dtype(self.df[col]) or self.df[col].dtype != np.float64:
                        # Convert to float64, coercing errors to NaN.
                        # Using pd.to_numeric is robust for various input types (Decimal, str, int).
                        original_type = self.df[col].dtype
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(np.float64)
                        if self.df[col].isnull().any():
                            self.logger.debug(f"Column '{col}' (original type: {original_type}) converted to float64 for TA, NaNs introduced.")
                        else:
                            self.logger.debug(f"Column '{col}' (original type: {original_type}) converted to float64 for TA.")
                    # else: # Already float64
                    #     self.logger.debug(f"Column '{col}' is already float64.")

            # Verify conversion and log dtypes
            converted_dtypes = {col: self.df[col].dtype for col in cols_to_convert if col in self.df.columns}
            self.logger.debug(f"DataFrame dtypes prepared for TA: {converted_dtypes}")
            # Check if any critical conversion failed resulting in non-float types (shouldn't happen with coerce + astype)
            for col, dtype in converted_dtypes.items():
                if not pd.api.types.is_float_dtype(dtype):
                     self.logger.warning(f"Column '{col}' is not float (dtype: {dtype}) after conversion attempt. TA results may be affected.")

        except Exception as e:
             self.logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
             self.df = pd.DataFrame() # Set to empty to prevent further processing


    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even triggered status from the shared symbol state."""
        # Default to False if the key doesn't exist
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool) -> None:
        """Sets the break-even triggered status in the shared symbol state and logs change."""
        if not isinstance(value, bool):
            self.logger.error(f"Invalid type for break_even_triggered ({type(value)}). Must be boolean.")
            return
        current_value = self.symbol_state.get('break_even_triggered')
        if current_value != value:
            self.symbol_state['break_even_triggered'] = value
            self.logger.info(f"Break-even status for {self.symbol} set to: {value}")


    def _define_ta_strategy(self) -> None:
        """Defines the pandas_ta Strategy object based on enabled indicators in the config."""
        cfg = self.config
        indi_cfg = cfg.get("indicators", {}) # Dictionary of enabled indicators

        # Helper to safely get parameters (already validated as int/float in load_config)
        def get_param(key: str, default: Union[int, float]) -> Union[int, float]:
            # Config validation ensures these are correct types (int/float)
            return cfg.get(key, default)

        # Get parameters for all potential indicators
        atr_p = int(get_param("atr_period", DEFAULT_ATR_PERIOD))
        ema_s = int(get_param("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
        ema_l = int(get_param("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        rsi_p = int(get_param("rsi_period", DEFAULT_RSI_WINDOW))
        bb_p = int(get_param("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        bb_std = float(get_param("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV))
        cci_w = int(get_param("cci_window", DEFAULT_CCI_WINDOW))
        wr_w = int(get_param("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        mfi_w = int(get_param("mfi_window", DEFAULT_MFI_WINDOW))
        stochrsi_w = int(get_param("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW))
        stochrsi_rsi_w = int(get_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)) # Inner RSI
        stochrsi_k = int(get_param("stoch_rsi_k", DEFAULT_K_WINDOW))
        stochrsi_d = int(get_param("stoch_rsi_d", DEFAULT_D_WINDOW))
        psar_af = float(get_param("psar_af", DEFAULT_PSAR_AF))
        psar_max = float(get_param("psar_max_af", DEFAULT_PSAR_MAX_AF))
        sma10_w = int(get_param("sma_10_window", DEFAULT_SMA_10_WINDOW))
        mom_p = int(get_param("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        vol_ma_p = int(get_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))

        # Build the list of indicators for the pandas_ta Strategy
        ta_list: List[Dict[str, Any]] = []
        self.ta_column_map: Dict[str, str] = {} # Reset map

        # --- Add indicators based on config flags and valid parameters ---
        # ATR (Always calculated for risk management)
        if atr_p > 0:
            # pandas_ta default ATR col name is 'ATRr_period'
            col_name = f"ATRr_{atr_p}"
            ta_list.append({"kind": "atr", "length": atr_p, "col_names": (col_name,)})
            self.ta_column_map["ATR"] = col_name
        else:
             self.logger.error(f"ATR period ({atr_p}) is invalid. Risk management calculations will fail.")

        # EMAs (Needed if alignment or MA cross exit enabled)
        ema_enabled = indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit")
        if ema_enabled:
            if ema_s > 0:
                col_name_s = f"EMA_{ema_s}"
                ta_list.append({"kind": "ema", "length": ema_s, "col_names": (col_name_s,)})
                self.ta_column_map["EMA_Short"] = col_name_s
            if ema_l > 0:
                # Validation ensures ema_l > ema_s already
                col_name_l = f"EMA_{ema_l}"
                ta_list.append({"kind": "ema", "length": ema_l, "col_names": (col_name_l,)})
                self.ta_column_map["EMA_Long"] = col_name_l
            elif ema_l <= 0:
                self.logger.warning(f"EMA Long period ({ema_l}) is invalid. Disabling EMA Long.")

        # Momentum
        if indi_cfg.get("momentum") and mom_p > 0:
            col_name = f"MOM_{mom_p}"
            ta_list.append({"kind": "mom", "length": mom_p, "col_names": (col_name,)})
            self.ta_column_map["Momentum"] = col_name

        # Volume SMA
        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0:
            col_name = f"VOL_SMA_{vol_ma_p}"
            # Ensure 'volume' column exists and is float before calculating
            if 'volume' in self.df.columns and pd.api.types.is_float_dtype(self.df['volume']):
                ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_p, "col_names": (col_name,)})
                self.ta_column_map["Volume_MA"] = col_name
            else:
                self.logger.warning(f"Cannot calculate Volume SMA: 'volume' column missing or not float in df.")

        # Stochastic RSI
        if indi_cfg.get("stoch_rsi") and all(p > 0 for p in [stochrsi_w, stochrsi_rsi_w, stochrsi_k, stochrsi_d]):
            # Default pandas_ta column names for StochRSI k and d
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
            # Format std dev for column name (e.g., 2.0 -> 2_0)
            bb_std_str = f"{bb_std:.1f}".replace('.', '_')
            # Default pandas_ta column names
            bbl = f"BBL_{bb_p}_{bb_std_str}"; bbm = f"BBM_{bb_p}_{bb_std_str}"
            bbu = f"BBU_{bb_p}_{bb_std_str}"; bbb = f"BBB_{bb_p}_{bb_std_str}"
            bbp = f"BBP_{bb_p}_{bb_std_str}"
            ta_list.append({
                "kind": "bbands", "length": bb_p, "std": bb_std,
                "col_names": (bbl, bbm, bbu, bbb, bbp) # Lower, Middle, Upper, Bandwidth, Percent
            })
            self.ta_column_map["BB_Lower"] = bbl
            self.ta_column_map["BB_Middle"] = bbm
            self.ta_column_map["BB_Upper"] = bbu
            # self.ta_column_map["BB_Bandwidth"] = bbb # Optional
            # self.ta_column_map["BB_Percent"] = bbp # Optional

        # VWAP
        if indi_cfg.get("vwap"):
            # VWAP calculation needs high, low, close, volume. pandas_ta handles this.
            # Default daily VWAP calculation in pandas_ta might reset daily ('D').
            # Check pandas_ta documentation if different anchoring is needed.
            vwap_col = "VWAP_D" # Default pandas_ta name (may vary based on params/version)
            if all(c in self.df.columns for c in ['high', 'low', 'close', 'volume']):
                 ta_list.append({"kind": "vwap", "anchor": "D", "col_names": (vwap_col,)}) # Use daily anchor
                 self.ta_column_map["VWAP"] = vwap_col
            else:
                 self.logger.warning("Cannot calculate VWAP: Missing HLCV columns.")

        # CCI
        if indi_cfg.get("cci") and cci_w > 0:
            # Default pandas_ta name includes the constant
            cci_col = f"CCI_{cci_w}_0.015"
            ta_list.append({"kind": "cci", "length": cci_w, "col_names": (cci_col,)})
            self.ta_column_map["CCI"] = cci_col

        # Williams %R
        if indi_cfg.get("wr") and wr_w > 0:
            wr_col = f"WILLR_{wr_w}"
            ta_list.append({"kind": "willr", "length": wr_w, "col_names": (wr_col,)})
            self.ta_column_map["WR"] = wr_col

        # Parabolic SAR
        if indi_cfg.get("psar"):
            # Clean format for name (remove trailing zeros/dots)
            psar_af_str = f"{psar_af}".rstrip('0').rstrip('.')
            psar_max_str = f"{psar_max}".rstrip('0').rstrip('.')
            # Default pandas_ta column names
            l_col = f"PSARl_{psar_af_str}_{psar_max_str}" # Long signal line
            s_col = f"PSARs_{psar_af_str}_{psar_max_str}" # Short signal line
            af_col = f"PSARaf_{psar_af_str}_{psar_max_str}" # Acceleration Factor
            r_col = f"PSARr_{psar_af_str}_{psar_max_str}" # Reversal indicator (0 or 1)
            ta_list.append({
                "kind": "psar", "af": psar_af, "max_af": psar_max,
                "col_names": (l_col, s_col, af_col, r_col)
            })
            self.ta_column_map["PSAR_Long"] = l_col   # Value when SAR is below (long trend), NaN otherwise
            self.ta_column_map["PSAR_Short"] = s_col  # Value when SAR is above (short trend), NaN otherwise
            # self.ta_column_map["PSAR_AF"] = af_col    # Optional: Current Acceleration Factor
            # self.ta_column_map["PSAR_Reversal"] = r_col # Optional: 1 if SAR reversed on this bar

        # SMA 10
        if indi_cfg.get("sma_10") and sma10_w > 0:
            col_name = f"SMA_{sma10_w}"
            ta_list.append({"kind": "sma", "length": sma10_w, "col_names": (col_name,)})
            self.ta_column_map["SMA10"] = col_name

        # MFI
        if indi_cfg.get("mfi") and mfi_w > 0:
            # MFI requires high, low, close, volume
            col_name = f"MFI_{mfi_w}"
            if all(c in self.df.columns for c in ['high', 'low', 'close', 'volume']):
                 ta_list.append({"kind": "mfi", "length": mfi_w, "col_names": (col_name,)})
                 self.ta_column_map["MFI"] = col_name
            else:
                 self.logger.warning("Cannot calculate MFI: Missing HLCV columns.")

        # --- Create Strategy ---
        if not ta_list:
            self.logger.warning(f"No valid indicators enabled or configured for {self.symbol}. TA Strategy not created.")
            return

        self.ta_strategy = ta.Strategy(
            name="EnhancedMultiIndicatorStrategy",
            description="Calculates multiple TA indicators based on bot config using pandas_ta",
            ta=ta_list
        )
        self.logger.debug(f"Defined TA Strategy for {self.symbol} with {len(ta_list)} indicator groups.")
        # Log the map for debugging which generic names correspond to which actual columns
        # self.logger.debug(f"TA Column Map: {self.ta_column_map}")


    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using the defined pandas_ta strategy on the float DataFrame (self.df)."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return
        if not self.ta_strategy:
            self.logger.warning(f"TA Strategy not defined for {self.symbol}, cannot calculate indicators.")
            return

        # Check if sufficient data exists (use pandas_ta's internal requirement if available)
        min_required_data = self.ta_strategy.required if hasattr(self.ta_strategy, 'required') else 50 # Default guess
        buffer = 20 # Add buffer for calculation stability, especially for complex indicators
        required_with_buffer = min_required_data + buffer
        if len(self.df) < required_with_buffer:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} rows) for {self.symbol} TA calculation (min recommended: {required_with_buffer}). Results may be inaccurate or NaN.{RESET}")

        try:
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol} on {len(self.df)} rows...")
            # Apply the strategy to the DataFrame (modifies self.df inplace)
            # Ensure the DataFrame has float types before this call (_convert_df_for_ta)
            start_ta_time = time.monotonic()
            self.df.ta.strategy(self.ta_strategy, timed=False) # timed=True adds overhead
            ta_duration = time.monotonic() - start_ta_time
            self.logger.debug(f"Finished indicator calculations for {self.symbol} in {ta_duration:.3f}s.")
            # Optional: Log columns generated: # self.logger.debug(f"DataFrame columns after TA: {self.df.columns.tolist()}")
        except AttributeError as ae:
             # This might catch issues if _convert_df_for_ta failed silently or pandas_ta internal issue
             if "'float' object has no attribute" in str(ae) and ('high' in str(ae) or 'low' in str(ae) or 'close' in str(ae)):
                 self.logger.error(f"{NEON_RED}Pandas TA Error ({self.symbol}): Input columns (e.g., high, low, close) might contain NaNs or non-numeric data after conversion. Error: {ae}{RESET}", exc_info=False)
                 self.logger.debug(f"Problematic DF sample (float converted):\n{self.df.tail()}")
             else:
                 self.logger.error(f"{NEON_RED}Pandas TA attribute error ({self.symbol}): {ae}. Is pandas_ta installed/working correctly? Check library compatibility.{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta strategy for {self.symbol}: {e}{RESET}", exc_info=True)
            # Don't wipe df here, some indicators might have calculated


    def _update_latest_indicator_values(self) -> None:
        """
        Updates self.indicator_values (dict of Decimals) with the latest calculated
        values from the float DataFrame (self.df), converting them back to Decimal.
        Handles potential NaN values gracefully.
        """
        self.indicator_values = {} # Reset before populating
        if self.df.empty:
            self.logger.warning(f"DataFrame empty for {self.symbol}. Cannot update latest indicator values.")
            return
        if self.df.iloc[-1].isnull().all():
            self.logger.warning(f"Last row of DataFrame is all NaN for {self.symbol}. Cannot update latest indicator values.")
            return

        try:
            # Get the last row (most recent indicator values)
            latest_series = self.df.iloc[-1]

            # Helper to safely convert float/object back to Decimal
            def to_decimal(value: Any) -> Optional[Decimal]:
                # Check for pandas NA, numpy NaN, or None
                if pd.isna(value) or value is None: return None
                try:
                    # Convert float to string first for precise Decimal conversion
                    # Check for infinity as well
                    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
                        return None
                    dec_val = Decimal(str(value))
                    # Final check for non-finite Decimals (just in case)
                    return dec_val if dec_val.is_finite() else None
                except (InvalidOperation, ValueError, TypeError):
                    self.logger.debug(f"Could not convert value '{value}' (type: {type(value)}) to Decimal.")
                    return None

            # Populate indicator_values using the ta_column_map to get actual column names
            for generic_name, actual_col_name in self.ta_column_map.items():
                if actual_col_name in latest_series:
                    raw_value = latest_series.get(actual_col_name)
                    self.indicator_values[generic_name] = to_decimal(raw_value)
                    # Log if conversion failed
                    # if raw_value is not None and self.indicator_values[generic_name] is None:
                    #     self.logger.debug(f"Failed Decimal conversion for {generic_name} ({actual_col_name}): {raw_value}")
                else:
                     # This might happen if an indicator calculation failed silently in pandas_ta
                     self.logger.debug(f"Column '{actual_col_name}' not found in DataFrame for indicator '{generic_name}' ({self.symbol}). Setting value to None.")
                     self.indicator_values[generic_name] = None

            # Also add latest OHLCV values (from the float df, converted back to Decimal)
            # These should generally not be NaN if the row itself isn't all NaN
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                 if base_col in latest_series:
                     self.indicator_values[base_col.capitalize()] = to_decimal(latest_series.get(base_col))

            valid_values_count = sum(1 for v in self.indicator_values.values() if v is not None)
            total_expected = len(self.ta_column_map) + 5 # +5 for OHLCV
            self.logger.debug(f"Latest indicator Decimal values updated for {self.symbol}: Count={valid_values_count}/{total_expected}")
            # For detailed logging:
            # valid_values_str = {k: f"{v:.5f}" if isinstance(v, Decimal) else str(v) for k, v in self.indicator_values.items()}
            # self.logger.debug(f"Values: {valid_values_str}")

        except IndexError:
            self.logger.error(f"DataFrame index out of bounds when accessing last row for {self.symbol}.")
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = {} # Clear potentially corrupted values


    # --- Precision and Market Info Helpers ---
    def get_min_tick_size(self) -> Optional[Decimal]:
        """Gets the minimum price tick size as a Decimal from market info."""
        tick = self.market_info.get('min_tick_size')
        if tick is None or not isinstance(tick, Decimal) or not tick.is_finite() or tick <= 0:
            self.logger.warning(f"Invalid or missing min_tick_size ({tick}) for {self.symbol}. Quantization may fail.")
            return None
        return tick

    def get_price_precision_digits(self) -> int:
        """Gets the number of decimal places for price precision."""
        # Returns 8 if not found, which is a safe default (won't truncate too much)
        return self.market_info.get('price_precision_digits', 8)

    def get_amount_precision_digits(self) -> int:
        """Gets the number of decimal places for amount (quantity) precision."""
        return self.market_info.get('amount_precision_digits', 8)

    def get_amount_step_size(self) -> Optional[Decimal]:
        """Gets the minimum amount step size as a Decimal from market info."""
        step = self.market_info.get('amount_step_size')
        if step is None or not isinstance(step, Decimal) or not step.is_finite() or step <= 0:
            self.logger.warning(f"Invalid or missing amount_step_size ({step}) for {self.symbol}. Amount quantization may use digits fallback.")
            return None
        return step

    def quantize_price(self, price: Union[Decimal, float, str], rounding: str = ROUND_DOWN) -> Optional[Decimal]:
        """
        Quantizes a price to the market's minimum tick size using specified rounding.

        Args:
            price: The price value to quantize.
            rounding: The rounding mode (e.g., ROUND_DOWN, ROUND_UP).

        Returns:
            The quantized price as Decimal, or None if quantization fails.
        """
        min_tick = self.get_min_tick_size()
        if min_tick is None: return None
        try:
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

    def quantize_amount(self, amount: Union[Decimal, float, str], rounding: str = ROUND_DOWN) -> Optional[Decimal]:
        """
        Quantizes an amount (quantity) to the market's amount step size (or precision digits).

        Args:
            amount: The amount value to quantize.
            rounding: The rounding mode (e.g., ROUND_DOWN, ROUND_UP).

        Returns:
            The quantized amount as Decimal, or None if quantization fails.
        """
        step_size = self.get_amount_step_size()
        if step_size is None:
            # Fallback to using precision digits if step size is missing
            amount_digits = self.get_amount_precision_digits()
            step_size = Decimal('1') / (Decimal('10') ** amount_digits)
            self.logger.debug(f"Using amount precision digits ({amount_digits}) for step size ({step_size}) for {self.symbol}")

        try:
            amount_decimal = Decimal(str(amount))
            if not amount_decimal.is_finite():
                self.logger.warning(f"Cannot quantize non-finite amount: {amount}")
                return None

            # Formula: floor/ceil(amount / step_size) * step_size
            # Use Decimal's quantize for precision
            quantized_factor = (amount_decimal / step_size).quantize(Decimal('0'), rounding=rounding)
            quantized_amount = quantized_factor * step_size

            # Re-quantize to ensure exact number of decimal places matching step size
            # This handles cases where step_size might be like 0.0025
            final_quantized = quantized_amount.quantize(step_size, rounding=ROUND_DOWN) # Final round down for safety
            return final_quantized
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing amount '{amount}' with step '{step_size}' for {self.symbol}: {e}")
            return None

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """
        Calculates Fibonacci retracement levels based on the high/low over a specified window.
        Uses the raw DataFrame (`df_raw` with Decimal precision) and quantizes the resulting levels.

        Args:
            window: The lookback period (number of candles). Uses config value if None.

        Returns:
            A dictionary of Fibonacci levels (e.g., "Fib_38.2%") mapped to quantized price Decimals.
            Returns an empty dictionary if calculation is not possible.
        """
        self.fib_levels_data = {} # Reset previous levels
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

            # Extract High/Low Series (should be Decimal type, but handle potential mixed types)
            high_series = df_slice["high"]
            low_series = df_slice["low"]

            # Find max high and min low, handling potential NaNs or non-Decimals gracefully
            # Convert to numeric first, coercing errors, then find max/min, then convert result back to Decimal
            # This ensures robustness even if df_raw somehow contains non-Decimal data
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
            if high <= low: # Use <= to handle cases where high == low
                 self.logger.warning(f"Invalid range (High <= Low): High={high}, Low={low} in window {window} for Fib calculation on {self.symbol}. Cannot calculate levels.")
                 return {}

            diff: Decimal = high - low
            levels: Dict[str, Decimal] = {}
            min_tick: Optional[Decimal] = self.get_min_tick_size()

            if diff > 0 and min_tick is not None:
                # Calculate levels only if range is valid and tick size is available
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float))
                    # Calculate raw price level
                    level_price_raw = high - (diff * level_pct)
                    # Quantize the calculated level price (round down for potential support/resistance)
                    level_price_quantized = self.quantize_price(level_price_raw, rounding=ROUND_DOWN)

                    if level_price_quantized is not None:
                        level_name = f"Fib_{level_pct * 100:.1f}%"
                        levels[level_name] = level_price_quantized
                    else:
                        self.logger.warning(f"Failed to quantize Fibonacci level {level_pct*100:.1f}% (Raw: {level_price_raw}) for {self.symbol}")
            elif min_tick is None:
                 # Calculate raw levels if quantization isn't possible
                 self.logger.warning(f"Calculating raw (non-quantized) Fibonacci levels for {self.symbol} due to missing min_tick_size.")
                 for level_pct_float in FIB_LEVELS:
                     level_pct = Decimal(str(level_pct_float))
                     level_price_raw = high - (diff * level_pct)
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_raw # Store raw Decimal
            else: # diff <= 0 case already handled
                 pass

            self.fib_levels_data = levels
            # Log the calculated levels (optional)
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

    def _get_indicator_float(self, name: str) -> Optional[float]:
        """Safely retrieves an indicator value from self.indicator_values as a float."""
        val_decimal = self.indicator_values.get(name)
        if val_decimal is None or not val_decimal.is_finite(): return None
        try: return float(val_decimal)
        except (ValueError, TypeError): return None

    def _check_ema_alignment(self) -> Optional[float]:
        """Checks if short EMA is above/below long EMA. Score: 1.0 (short > long), -1.0 (short < long), 0.0 (equal/NaN)."""
        ema_s = self._get_indicator_float("EMA_Short")
        ema_l = self._get_indicator_float("EMA_Long")
        if ema_s is None or ema_l is None: return None
        if ema_s > ema_l: return 1.0
        if ema_s < ema_l: return -1.0
        return 0.0 # Equal

    def _check_momentum(self) -> Optional[float]:
        """Checks Momentum indicator value. Positive -> bullish, Negative -> bearish. Score scaled & clamped."""
        mom = self._get_indicator_float("Momentum")
        if mom is None: return None
        # Basic scaling: Assumes momentum values might need normalization.
        # This simple scaling factor might need adjustment based on typical observed ranges for the asset/timeframe.
        # Goal is to map typical positive/negative values towards +1/-1.
        scaling_factor = 0.1 # Example: If MOM is often +/- 10, this scales it to +/- 1.
        score = mom * scaling_factor
        return max(-1.0, min(1.0, score)) # Clamp score to [-1.0, 1.0]

    def _check_volume_confirmation(self) -> Optional[float]:
        """Checks if current volume exceeds its MA by a multiplier. Score: 0.7 (exceeds), 0.0 (doesn't/NaN)."""
        vol = self._get_indicator_float("Volume")
        vol_ma = self._get_indicator_float("Volume_MA")
        # Need valid volume, and vol_ma > 0 for meaningful comparison
        if vol is None or vol_ma is None or vol_ma <= 0: return None
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        return 0.7 if vol > vol_ma * multiplier else 0.0

    def _check_stoch_rsi(self) -> Optional[float]:
        """Checks Stochastic RSI K and D lines for overbought/oversold and crossovers. Score range [-1.0, 1.0]."""
        k = self._get_indicator_float("StochRSI_K")
        d = self._get_indicator_float("StochRSI_D")
        if k is None or d is None: return None
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        score = 0.0
        # Strong signals: Crossover within OB/OS zones
        if k < oversold and d < oversold: score = 1.0 if k > d else 0.8 # Bullish crossover deep OS (stronger if k>d)
        elif k > overbought and d > overbought: score = -1.0 if k < d else -0.8 # Bearish crossover deep OB (stronger if k<d)
        # Weaker signals: Just entering OB/OS without crossover confirmation yet
        elif k < oversold: score = 0.6
        elif k > overbought: score = -0.6
        # Mid-range crossover signals (weakest)
        elif k > d: score = 0.3 # Bullish crossover mid-range
        elif k < d: score = -0.3 # Bearish crossover mid-range
        # Else (k == d mid-range): score remains 0.0
        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_rsi(self) -> Optional[float]:
        """Checks RSI value against OB/OS levels. Score scaled linearly between 0 and 100, clamped."""
        rsi = self._get_indicator_float("RSI")
        if rsi is None: return None
        ob = 70.0; os = 30.0 # Common thresholds
        # Simple linear scale: Score = 1.0 at RSI=0, 0.0 at RSI=50, -1.0 at RSI=100
        score = (50.0 - rsi) / 50.0
        # Optional: Add extra weight/boost for extreme readings (beyond standard OB/OS)
        # Example: Slightly increase score magnitude if RSI is < 20 or > 80
        extreme_ob = 80.0; extreme_os = 20.0
        if rsi >= extreme_ob: score = max(-1.0, score * 1.1) # Increase bearishness above extreme OB
        if rsi <= extreme_os: score = min(1.0, score * 1.1) # Increase bullishness below extreme OS
        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_cci(self) -> Optional[float]:
        """Checks CCI against standard OB/OS levels (+/- 100). Score scaled linearly between thresholds, clamped."""
        cci = self._get_indicator_float("CCI")
        if cci is None: return None
        ob = 100.0; os = -100.0
        # Scale based on thresholds: Strong signal outside +/-100, linear inside.
        if cci >= ob: score = -1.0 # Strong sell signal above 100
        elif cci <= os: score = 1.0  # Strong buy signal below -100
        else: score = -cci / ob # Linear scale between -100 and 100 (e.g., cci=50 -> -0.5, cci=-50 -> 0.5)
        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_wr(self) -> Optional[float]:
        """Checks Williams %R against standard OB/OS levels (-20 / -80). Score scaled linearly, clamped."""
        wr = self._get_indicator_float("WR") # Williams %R typically ranges from -100 (most oversold) to 0 (most overbought)
        if wr is None: return None
        ob = -20.0; os = -80.0
        # Scale: Score = 1.0 at WR=-100, 0.0 at WR=-50, -1.0 at WR=0
        score = (wr + 50.0) / -50.0
        # Optional: Add extra weight/boost for extreme readings
        extreme_ob = -10.0; extreme_os = -90.0
        if wr >= extreme_ob: score = max(-1.0, score * 1.1) # Increase bearishness near 0
        if wr <= extreme_os: score = min(1.0, score * 1.1) # Increase bullishness near -100
        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_psar(self) -> Optional[float]:
        """Checks Parabolic SAR position relative to price. Score: 1.0 (SAR below price - bullish), -1.0 (SAR above - bearish), 0.0 (transition/error)."""
        # PSARl (long) column has value when SAR is below price, NaN otherwise.
        # PSARs (short) column has value when SAR is above price, NaN otherwise.
        psar_l_val = self.indicator_values.get("PSAR_Long") # Decimal or None
        psar_s_val = self.indicator_values.get("PSAR_Short") # Decimal or None

        # Check if the values are finite Decimals (i.e., not None or NaN)
        psar_l_active = psar_l_val is not None and psar_l_val.is_finite()
        psar_s_active = psar_s_val is not None and psar_s_val.is_finite()

        if psar_l_active and not psar_s_active: return 1.0  # SAR is below price (Long trend active)
        if psar_s_active and not psar_l_active: return -1.0 # SAR is above price (Short trend active)

        # If both active/inactive (shouldn't happen with standard PSAR) or both NaN (e.g., at start), return neutral.
        # Also handles the case where one might be 0 if calculation somehow resulted in that.
        if (psar_l_active and psar_s_active) or (not psar_l_active and not psar_s_active):
            self.logger.debug(f"PSAR state ambiguous/neutral for {self.symbol}: PSARl={psar_l_val}, PSARs={psar_s_val}")
            return 0.0
        # This case should be covered above, but acts as a fallback
        return 0.0

    def _check_sma10(self) -> Optional[float]:
        """Checks if close price is above/below SMA10. Score: 0.5 (above), -0.5 (below), 0.0 (equal/NaN)."""
        sma = self._get_indicator_float("SMA10")
        close = self._get_indicator_float("Close")
        if sma is None or close is None: return None
        if close > sma: return 0.5
        if close < sma: return -0.5
        return 0.0 # Equal

    def _check_vwap(self) -> Optional[float]:
        """Checks if close price is above/below VWAP. Score: 0.6 (above), -0.6 (below), 0.0 (equal/NaN)."""
        vwap = self._get_indicator_float("VWAP")
        close = self._get_indicator_float("Close")
        if vwap is None or close is None: return None
        if close > vwap: return 0.6
        if close < vwap: return -0.6
        return 0.0 # Equal

    def _check_mfi(self) -> Optional[float]:
        """Checks Money Flow Index against standard OB/OS levels (80/20). Score scaled linearly, clamped."""
        mfi = self._get_indicator_float("MFI")
        if mfi is None: return None
        ob = 80.0; os = 20.0
        # Scale similar to RSI: Score = 1.0 at MFI=0, 0.0 at MFI=50, -1.0 at MFI=100
        score = (50.0 - mfi) / 50.0
        # Optional boost for extremes
        extreme_ob = 90.0; extreme_os = 10.0
        if mfi >= extreme_ob: score = max(-1.0, score * 1.1) # Boost bearishness
        if mfi <= extreme_os: score = min(1.0, score * 1.1) # Boost bullishness
        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_bollinger_bands(self) -> Optional[float]:
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
        position_within_band = (close - bbl) / band_width
        # Convert position [0, 1] to score [+1, -1]
        # Score = 1 - 2 * position. (If position=0, score=1. If position=1, score=-1. If position=0.5, score=0)
        score = 1.0 - (2.0 * position_within_band)
        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_orderbook(self, orderbook_data: Optional[Dict]) -> Optional[float]:
        """
        Calculates Order Book Imbalance (OBI) from fetched order book data.
        OBI = (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)
        Uses top N levels as defined in config. Score: [-1.0, 1.0].

        Args:
            orderbook_data: Dictionary from fetch_orderbook_ccxt containing 'bids' and 'asks'.

        Returns:
            OBI score as float, or None if calculation fails.
        """
        if not orderbook_data or not isinstance(orderbook_data, dict):
             self.logger.debug("Orderbook data missing or invalid type for OBI calculation.")
             return None
        try:
            limit = int(self.config.get("orderbook_limit", 10))
            # Orderbook data from CCXT usually has string prices/amounts
            bids_raw = orderbook_data.get('bids', [])
            asks_raw = orderbook_data.get('asks', [])

            if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
                 self.logger.warning("Orderbook bids/asks are not lists.")
                 return 0.0 # Neutral if structure is wrong

            # Take top N levels
            top_bids = bids_raw[:limit]
            top_asks = asks_raw[:limit]
            if not top_bids or not top_asks:
                 self.logger.debug(f"Orderbook empty or insufficient levels (Limit: {limit}) for OBI calc.")
                 return 0.0 # Neutral if insufficient levels

            # Sum volume (amount) at each level, converting to Decimal robustly
            bid_vol = Decimal('0')
            for b in top_bids:
                if isinstance(b, (list, tuple)) and len(b) > 1:
                    try: bid_vol += Decimal(str(b[1]))
                    except (InvalidOperation, TypeError, IndexError): continue # Skip invalid entries
            ask_vol = Decimal('0')
            for a in top_asks:
                 if isinstance(a, (list, tuple)) and len(a) > 1:
                    try: ask_vol += Decimal(str(a[1]))
                    except (InvalidOperation, TypeError, IndexError): continue # Skip invalid entries

            total_vol = bid_vol + ask_vol
            if total_vol <= 0:
                 self.logger.debug("Total volume in top orderbook levels is zero.")
                 return 0.0 # Avoid division by zero, return neutral

            # Calculate OBI = (BidVol - AskVol) / TotalVol
            obi_decimal = (bid_vol - ask_vol) / total_vol # Decimal result

            # Clamp and convert to float for the scoring system
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi_decimal)))
            # self.logger.debug(f"OBI Calc ({self.symbol}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, OBI={obi_decimal:.4f} -> Score={score:.3f}")
            return score

        except (InvalidOperation, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error calculating Order Book Imbalance for {self.symbol}: {e}")
            return None # Indicate error / unavailable score
        except Exception as e:
             self.logger.error(f"Unexpected error in OBI calculation for {self.symbol}: {e}", exc_info=True)
             return None

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates a final trading signal ('BUY', 'SELL', 'HOLD') based on weighted scores
        from enabled indicators. Uses Decimal for score accumulation for precision.

        Args:
            current_price_dec: Current market price (Decimal) for logging.
            orderbook_data: Optional order book data for OBI calculation.

        Returns:
            The final signal string: "BUY", "SELL", or "HOLD".
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Reset signals, default HOLD
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores: Dict[str, str] = {} # For detailed logging

        # --- Pre-checks ---
        if not self.indicator_values:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Indicator values not calculated.")
            return "HOLD"
        if not current_price_dec.is_finite() or current_price_dec <= 0:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price_dec}).")
            return "HOLD"
        if not self.weights:
            self.logger.debug(f"No weights found for active set '{self.active_weight_set_name}'. Holding.")
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
        active_weights = self.weights
        enabled_indicators_config = self.config.get("indicators", {})

        for indicator_key, check_method in indicator_check_methods.items():
            # Check if indicator is enabled in config
            if not enabled_indicators_config.get(indicator_key, False):
                # self.logger.debug(f"Indicator '{indicator_key}' disabled.")
                debug_scores[indicator_key] = "Disabled"
                continue

            # Check if weight exists for this indicator in the active set
            weight_val = active_weights.get(indicator_key)
            if weight_val is None:
                # self.logger.debug(f"No weight found for enabled indicator '{indicator_key}'.")
                debug_scores[indicator_key] = "No Weight"
                continue

            # Convert weight to Decimal, handling potential errors
            try:
                weight = Decimal(str(weight_val))
            except (ValueError, InvalidOperation, TypeError):
                self.logger.warning(f"Invalid weight '{weight_val}' for '{indicator_key}'. Skipping.")
                debug_scores[indicator_key] = f"Invalid Wt({weight_val})"
                continue

            # Skip if weight is zero
            if weight == Decimal("0"):
                debug_scores[indicator_key] = "Wt=0"
                continue

            # Execute the check method to get the score [-1.0, 1.0] or None
            indicator_score_float: Optional[float] = None
            try:
                indicator_score_float = check_method()
            except Exception as e:
                self.logger.error(f"Error executing check method for '{indicator_key}': {e}", exc_info=True)
                debug_scores[indicator_key] = "Check Error"
                nan_indicator_count += 1
                continue # Skip this indicator if check failed

            # Process the returned score
            if indicator_score_float is not None and math.isfinite(indicator_score_float):
                try:
                    # Clamp score to [-1.0, 1.0] before converting to Decimal
                    clamped_score_float = max(-1.0, min(1.0, indicator_score_float))
                    indicator_score_decimal = Decimal(str(clamped_score_float))

                    # Calculate weighted score and add to total
                    weighted_score = indicator_score_decimal * weight
                    final_signal_score += weighted_score
                    total_weight_applied += abs(weight) # Sum absolute weights for normalization/debug
                    active_indicator_count += 1
                    # Store score details for debug logging
                    debug_scores[indicator_key] = f"{indicator_score_float:.2f} (x{weight:.2f}) = {weighted_score:.3f}"
                except (InvalidOperation, TypeError) as calc_err:
                    self.logger.error(f"Error processing score/weight for {indicator_key}: {calc_err}")
                    debug_scores[indicator_key] = "Calc Error"
                    nan_indicator_count += 1
            else:
                # Score is None or NaN/Infinite
                # self.logger.debug(f"Indicator '{indicator_key}' returned None or non-finite score.")
                debug_scores[indicator_key] = "NaN/None"
                nan_indicator_count += 1

        # --- Determine Final Signal Based on Score and Threshold ---
        final_signal = "HOLD"
        normalized_score = Decimal("0.0")
        if total_weight_applied > 0:
            # Normalize the score based on the sum of absolute weights applied
            # This gives a score roughly in [-1, 1] range if individual scores are [-1, 1]
            # Useful for comparing signal strength across different assets/times
            normalized_score = (final_signal_score / total_weight_applied).quantize(Decimal("0.0001"))
        elif active_indicator_count > 0:
             self.logger.warning(f"Calculated signal score is {final_signal_score} but total weight applied is zero. Check weights.")

        # Use specific threshold if active set is 'scalping', otherwise default
        # Thresholds are applied to the RAW final_signal_score (sum of weighted scores)
        threshold_key = "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
        default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
        try:
            threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
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
        if self.logger.level <= logging.DEBUG: # Log details only if debugging
             # Filter out non-contributing indicators for cleaner debug log
             score_details_str = ", ".join([f"{k}: {v}" for k, v in debug_scores.items() if v not in ["Disabled", "No Weight", "Wt=0"]])
             self.logger.debug(f"  Detailed Scores: {score_details_str}")

        # Update internal signal state (used for MA cross check, etc.)
        if final_signal in self.signals:
            self.signals[final_signal] = 1
            self.signals["HOLD"] = 1 if final_signal == "HOLD" else 0
        return final_signal


    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_signal: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates quantized Entry, Take Profit (TP), and Stop Loss (SL) prices based on ATR.
        Uses Decimal for precision and validates results, ensuring SL/TP are a minimum
        distance away from the entry price.

        Args:
            entry_price_signal: The price near which the signal occurred (e.g., current price).
            signal: "BUY" or "SELL".

        Returns:
            Tuple (Quantized Entry Price, Quantized TP Price, Quantized SL Price).
            Returns (None, None, None) if calculation fails (e.g., invalid ATR, price).
            TP or SL might be None individually if calculation leads to invalid values.
        """
        quantized_entry: Optional[Decimal] = None
        take_profit: Optional[Decimal] = None
        stop_loss: Optional[Decimal] = None

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
             self.logger.error(f"Invalid signal '{signal}' for TP/SL calculation."); return None, None, None
        if not entry_price_signal.is_finite() or entry_price_signal <= 0:
            self.logger.error(f"Invalid entry signal price ({entry_price_signal}) for TP/SL calc."); return None, None, None

        # --- Quantize Entry Price ---
        # Use ROUND_DOWN for BUY entry, ROUND_UP for SELL entry to get a slightly more conservative entry price reference.
        # Actual fill price for market orders will vary.
        entry_rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
        quantized_entry = self.quantize_price(entry_price_signal, rounding=entry_rounding)
        if quantized_entry is None:
            self.logger.error(f"Failed to quantize entry signal price {entry_price_signal} for {self.symbol}.")
            return None, None, None

        # --- Get ATR and Min Tick ---
        atr_val = self.indicator_values.get("ATR")
        min_tick = self.get_min_tick_size()

        if atr_val is None or not atr_val.is_finite() or atr_val <= 0 or min_tick is None:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate dynamic TP/SL for {self.symbol}: Invalid ATR ({atr_val}) or MinTick ({min_tick}). SL/TP will be None.{RESET}")
            # Return the valid entry price, but None for SL/TP
            return quantized_entry, None, None

        # --- Calculate SL/TP Offsets ---
        try:
            atr = atr_val # Already Decimal
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Calculate offsets from entry price
            sl_offset_raw = atr * sl_mult
            tp_offset_raw = atr * tp_mult

            # Ensure SL offset is at least the minimum required ticks away
            min_sl_offset_value = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
            if sl_offset_raw < min_sl_offset_value:
                self.logger.warning(f"Calculated SL offset ({sl_offset_raw}) < minimum {MIN_TICKS_AWAY_FOR_SLTP} ticks ({min_sl_offset_value}). Adjusting SL offset to minimum.")
                sl_offset = min_sl_offset_value
            else:
                sl_offset = sl_offset_raw

            # Ensure TP offset is at least the minimum required ticks away
            min_tp_offset_value = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
            if tp_offset_raw < min_tp_offset_value:
                 self.logger.warning(f"Calculated TP offset ({tp_offset_raw}) < minimum {MIN_TICKS_AWAY_FOR_SLTP} ticks ({min_tp_offset_value}). Adjusting TP offset to minimum.")
                 tp_offset = min_tp_offset_value
            else:
                 tp_offset = tp_offset_raw

            # --- Calculate Raw TP/SL Prices ---
            if signal == "BUY":
                sl_raw = quantized_entry - sl_offset
                tp_raw = quantized_entry + tp_offset
                # Quantize SL DOWN (further away), TP UP (further away) for safety/wider range initially
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_DOWN)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_UP)
            else: # SELL
                sl_raw = quantized_entry + sl_offset
                tp_raw = quantized_entry - tp_offset
                # Quantize SL UP (further away), TP DOWN (further away)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_UP)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_DOWN)

            # --- Post-Calculation Validation (Ensure min distance after quantization) ---
            # Validate Stop Loss
            if stop_loss is not None:
                # Calculate the minimum distance boundary
                sl_boundary = quantized_entry - min_sl_offset_value if signal == "BUY" else quantized_entry + min_sl_offset_value
                needs_sl_adjustment = False
                if signal == "BUY" and stop_loss >= sl_boundary: needs_sl_adjustment = True
                if signal == "SELL" and stop_loss <= sl_boundary: needs_sl_adjustment = True

                if needs_sl_adjustment:
                    # Re-quantize the boundary price safely away
                    rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
                    adjusted_sl = self.quantize_price(sl_boundary, rounding=rounding)
                    if adjusted_sl == stop_loss: # If quantization didn't move it, move one more tick
                         adjusted_sl = self.quantize_price(sl_boundary - (min_tick if signal == "BUY" else -min_tick), rounding=rounding)

                    if adjusted_sl is not None and adjusted_sl != stop_loss:
                        self.logger.warning(f"Initial SL ({stop_loss}) too close to entry ({quantized_entry}) after quantization. Adjusting to {adjusted_sl} ({MIN_TICKS_AWAY_FOR_SLTP} ticks away).")
                        stop_loss = adjusted_sl
                    else:
                        self.logger.error(f"Could not adjust SL ({stop_loss}) to be minimum distance from entry ({quantized_entry}). Setting SL to None.")
                        stop_loss = None # Safety measure

                # Final check for zero/negative SL
                if stop_loss is not None and stop_loss <= 0:
                    self.logger.error(f"Calculated SL is zero/negative ({stop_loss}). Setting SL to None.")
                    stop_loss = None

            # Validate Take Profit
            if take_profit is not None:
                # Calculate the minimum distance boundary
                tp_boundary = quantized_entry + min_tp_offset_value if signal == "BUY" else quantized_entry - min_tp_offset_value
                needs_tp_adjustment = False
                # Check if TP ended up too close or on the wrong side of entry after quantization
                if signal == "BUY" and take_profit <= tp_boundary: needs_tp_adjustment = True
                if signal == "SELL" and take_profit >= tp_boundary: needs_tp_adjustment = True

                if needs_tp_adjustment:
                    rounding = ROUND_UP if signal == "BUY" else ROUND_DOWN
                    adjusted_tp = self.quantize_price(tp_boundary, rounding=rounding)
                    if adjusted_tp == take_profit: # Move one more tick if quantization didn't change it
                        adjusted_tp = self.quantize_price(tp_boundary + (min_tick if signal == "BUY" else -min_tick), rounding=rounding)

                    if adjusted_tp is not None and adjusted_tp != take_profit:
                        self.logger.warning(f"Initial TP ({take_profit}) too close to entry ({quantized_entry}) after quantization. Adjusting to {adjusted_tp} ({MIN_TICKS_AWAY_FOR_SLTP} ticks away).")
                        take_profit = adjusted_tp
                    else:
                        self.logger.error(f"Could not adjust TP ({take_profit}) to be minimum distance from entry ({quantized_entry}). Setting TP to None.")
                        take_profit = None # Safety measure

                # Final check for zero/negative TP
                if take_profit is not None and take_profit <= 0:
                    self.logger.error(f"Calculated TP is zero/negative ({take_profit}). Setting TP to None.")
                    take_profit = None

            # Log results
            prec = self.get_price_precision_digits()
            tp_log = f"{take_profit:.{prec}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{prec}f}" if stop_loss else 'N/A'
            entry_log = f"{quantized_entry:.{prec}f}"
            self.logger.info(f"Calc TP/SL ({signal}): Entry={entry_log}, TP={tp_log}, SL={sl_log} (ATR={atr:.{prec+1}f}, SLx={sl_mult}, TPx={tp_mult})")

            return quantized_entry, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error during TP/SL calculation value conversion for {self.symbol}: {e}{RESET}")
            return quantized_entry, None, None # Return entry if valid, but no TP/SL
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return quantized_entry, None, None


# --- Position Sizing ---
def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # Keep as float from config (0.01 = 1%)
    entry_price: Decimal,
    stop_loss_price: Decimal,
    market_info: Dict,
    leverage: int, # Keep as int from config
    logger: logging.Logger
) -> Optional[Decimal]:
    """
    Calculates the position size in base currency units (Spot) or contracts (Futures)
    using Decimal precision. Validates against market limits and available margin.

    Args:
        balance: Available balance in quote currency (Decimal).
        risk_per_trade: Fraction of balance to risk (float, e.g., 0.01 for 1%).
        entry_price: Proposed entry price (Decimal).
        stop_loss_price: Proposed stop loss price (Decimal).
        market_info: Dictionary containing market details (precision, limits, contract size).
        leverage: Leverage to be used (int, relevant for contracts).
        logger: Logger instance.

    Returns:
        Calculated and quantized position size (Decimal), or None if calculation fails
        or constraints (min/max size, min cost, margin) are not met.
    """
    lg = logger
    symbol: str = market_info.get('symbol', 'N/A')
    contract_size: Decimal = market_info.get('contract_size', Decimal('1')) # Default 1 for spot/linear
    min_order_amount: Optional[Decimal] = market_info.get('min_order_amount')
    max_order_amount: Optional[Decimal] = market_info.get('max_order_amount')
    min_order_cost: Optional[Decimal] = market_info.get('min_order_cost')
    # amount_digits: Optional[int] = market_info.get('amount_precision_digits') # Use step size instead
    amount_step_size: Optional[Decimal] = market_info.get('amount_step_size')
    is_contract: bool = market_info.get('is_contract', False)
    is_inverse: bool = market_info.get('inverse', False)
    quote_currency: str = market_info.get('quote', '?')
    base_currency: str = market_info.get('base', '?')

    # --- Input Validation ---
    if balance <= 0: lg.error(f"Size Calc Error ({symbol}): Balance <= 0"); return None
    if not entry_price.is_finite() or entry_price <= 0: lg.error(f"Size Calc Error ({symbol}): Invalid entry price {entry_price}"); return None
    if not stop_loss_price.is_finite() or stop_loss_price <= 0: lg.error(f"Size Calc Error ({symbol}): Invalid SL price {stop_loss_price}"); return None
    if entry_price == stop_loss_price: lg.error(f"Size Calc Error ({symbol}): Entry price equals SL price"); return None
    if amount_step_size is None or amount_step_size <= 0:
        lg.error(f"Size Calc Error ({symbol}): Amount step size missing or invalid ({amount_step_size}) from market info"); return None
    if not (0 < risk_per_trade < 1): lg.error(f"Size Calc Error ({symbol}): Invalid risk_per_trade value {risk_per_trade} (must be between 0 and 1)"); return None
    if is_contract and leverage <= 0: lg.error(f"Size Calc Error ({symbol}): Invalid leverage {leverage} for contract"); return None

    try:
        # --- Calculate Risk Amount and SL Distance ---
        risk_amount_quote: Decimal = balance * Decimal(str(risk_per_trade))
        sl_distance_points: Decimal = abs(entry_price - stop_loss_price)
        if sl_distance_points <= 0: lg.error(f"Size Calc Error ({symbol}): SL distance points is zero"); return None

        # --- Calculate Risk Per Unit/Contract in Quote Currency ---
        risk_per_unit_or_contract_quote = Decimal('NaN')

        if is_contract:
            if is_inverse:
                # Inverse: Risk per Contract (in Quote) = ContractSize * |Entry - SL| / SL (approx)
                # More accurate: Risk = Contracts * ContractSize * |1/Entry - 1/SL| * ValueAtEntry
                # Let's use the difference-in-reciprocal method for better accuracy
                if entry_price == 0 or stop_loss_price == 0: lg.error(f"Size Calc Error ({symbol}): Zero price encountered for inverse calculation"); return None
                # Risk per contract in BASE currency terms
                risk_per_contract_base = contract_size * abs(Decimal('1')/entry_price - Decimal('1')/stop_loss_price)
                # Convert risk per contract to QUOTE currency terms using entry price as approximation
                risk_per_unit_or_contract_quote = risk_per_contract_base * entry_price
                size_unit_name = f"contracts ({base_currency})"
            else: # Linear Contract
                # Linear: Risk per Contract (in Quote) = ContractSize * |Entry - SL|
                risk_per_unit_or_contract_quote = sl_distance_points * contract_size
                size_unit_name = f"contracts ({base_currency})"
        else: # Spot
            # Spot: Risk per Unit (in Quote) = |Entry - SL| (since contract size is effectively 1 base unit)
            risk_per_unit_or_contract_quote = sl_distance_points
            size_unit_name = f"{base_currency}" # Size is in base currency units

        # Validate calculated risk per unit
        if not risk_per_unit_or_contract_quote.is_finite() or risk_per_unit_or_contract_quote <= 0:
            lg.error(f"Size Calc Error ({symbol}): Invalid calculated risk per unit/contract ({risk_per_unit_or_contract_quote})")
            return None

        # --- Calculate Unquantized Size ---
        # Size = Total Risk Amount (Quote) / Risk Per Unit/Contract (Quote)
        size_unquantized = risk_amount_quote / risk_per_unit_or_contract_quote

        if not size_unquantized.is_finite() or size_unquantized <= 0:
            lg.error(f"Size Calc Error ({symbol}): Invalid unquantized size calculated ({size_unquantized}) from RiskAmt={risk_amount_quote:.4f} / RiskPerUnit={risk_per_unit_or_contract_quote:.8f}")
            return None

        lg.debug(f"Size Calc ({symbol}): Bal={balance:.2f}, Risk={risk_per_trade*100:.2f}%, RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_points}, RiskPerUnit={risk_per_unit_or_contract_quote:.8f}, UnquantSize={size_unquantized:.8f} {size_unit_name}")

        # --- Quantize Size ---
        # Round DOWN using the amount step size to be conservative with risk
        quantized_size = (size_unquantized / amount_step_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * amount_step_size
        # Re-quantize to ensure exact decimal places match step size (handles steps like 0.0025)
        quantized_size = quantized_size.quantize(amount_step_size)

        lg.debug(f"Quantized Size ({symbol}): {quantized_size} {size_unit_name} (Step: {amount_step_size})")

        if quantized_size <= 0:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Size is zero after quantization (Unquantized: {size_unquantized:.8f}). Cannot place order.{RESET}")
            return None

        # --- Validate Against Market Limits (Amount) ---
        if min_order_amount is not None and quantized_size < min_order_amount:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} {size_unit_name} < Min Amount {min_order_amount}. Cannot place order.{RESET}")
            return None
        if max_order_amount is not None and quantized_size > max_order_amount:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} {size_unit_name} > Max Amount {max_order_amount}. Capping size to max.{RESET}")
            # Cap the size and re-quantize using floor (ROUND_DOWN)
            quantized_size = (max_order_amount / amount_step_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * amount_step_size
            quantized_size = quantized_size.quantize(amount_step_size)
            # Check again if capped size is still valid (above min)
            if quantized_size <= 0 or (min_order_amount is not None and quantized_size < min_order_amount):
                lg.error(f"Size Calc Error ({symbol}): Capped size {quantized_size} is zero or below min amount {min_order_amount}.")
                return None

        # --- Validate Against Market Limits (Cost) & Margin ---
        # Calculate the order value in quote currency
        order_value_quote = Decimal('0')
        if is_contract:
             if is_inverse:
                 # Inverse value in quote = Contracts * ContractSize (value is in base, use price to convert approx)
                 # Value = Size (Contracts) * Contract Size (Base/Contract)
                 # Cost in Quote = Value (Base) * Price (Quote/Base) - this isn't right for margin calc
                 # Margin for inverse is typically based on Base qty: Margin = (Contracts * ContractSize) / Leverage
                 # We need value in quote for min_cost check though: Value(Quote) ~ Contracts * ContractSize * EntryPrice
                 order_value_quote = quantized_size * contract_size * entry_price
                 margin_required_base = (quantized_size * contract_size) / Decimal(leverage)
                 margin_required = margin_required_base * entry_price # Approx margin in quote terms
             else: # Linear value in quote = Contracts * ContractSize * EntryPrice
                 order_value_quote = quantized_size * contract_size * entry_price
                 margin_required = order_value_quote / Decimal(leverage)
        else: # Spot value in quote = Amount * EntryPrice
             order_value_quote = quantized_size * entry_price
             margin_required = order_value_quote # No leverage for spot

        # Check min cost limit
        if min_order_cost is not None and order_value_quote < min_order_cost:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Order value {order_value_quote:.4f} {quote_currency} < Min Cost {min_order_cost}. Cannot place order.{RESET}")
            return None

        # Check margin requirement vs available balance
        # Use a small buffer (e.g., 0.5% = 1.005) for fees/slippage, maybe configurable?
        buffer_factor = Decimal("1.005")
        required_margin_with_buffer = margin_required * buffer_factor
        if required_margin_with_buffer > balance:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Required margin {margin_required:.4f} {quote_currency} (incl. buffer: {required_margin_with_buffer:.4f}) > Available balance {balance:.4f}. Cannot place order.{RESET}")
            # TODO: Optionally, reduce size to fit available margin?
            # Requires recalculating size based on max margin: MaxValue = Balance * Leverage; MaxSize = MaxValue / (ContractSize * EntryPrice) etc.
            # For now, just reject the trade.
            return None

        # --- Success ---
        lg.info(f"Calculated position size for {symbol}: {quantized_size} {size_unit_name} (Value: ~{order_value_quote:.2f} {quote_currency}, Margin: ~{margin_required:.2f} {quote_currency})")
        return quantized_size

    except (InvalidOperation, ValueError, TypeError) as e:
         lg.error(f"{NEON_RED}Error during position size calculation for {symbol}: {e}{RESET}", exc_info=True)
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


# --- CCXT Trading Action Wrappers ---

def fetch_positions_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """
    Fetches the current non-zero position for a specific symbol using V5 API via safe_ccxt_call.
    Standardizes the returned position dictionary for internal use.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The standard CCXT symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        A dictionary containing standardized position details if a non-zero position exists,
        otherwise None. Includes keys like 'symbol', 'side', 'contracts' (abs size as Decimal),
        'entryPrice' (Decimal), 'liqPrice' (Decimal, optional), 'unrealizedPnl' (Decimal, optional),
        'leverage' (float, optional), 'positionIdx' (int, from info), 'info' (raw data), 'market_info'.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID

    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping position check for non-derivative symbol: {symbol} (Category: {category})")
        return None
    if not exchange.has.get('fetchPositions'):
        lg.error(f"Exchange {exchange.id} doesn't support fetchPositions()."); return None

    try:
        # Bybit V5 requires category and optionally symbol/market_id
        # Fetching for a specific symbol is generally more efficient
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Use safe_ccxt_call. Pass symbols=[symbol] for potential CCXT client-side filtering,
        # but the API call itself is driven by params['symbol'].
        all_positions_raw = safe_ccxt_call(exchange, 'fetch_positions', lg, symbols=[symbol], params=params)

        if not isinstance(all_positions_raw, list):
            lg.error(f"fetch_positions did not return a list for {symbol}. Type: {type(all_positions_raw)}. Response: {all_positions_raw}")
            return None

        # Filter the raw response to find the exact symbol and non-zero size
        for pos_raw in all_positions_raw:
            if not isinstance(pos_raw, dict):
                lg.warning(f"Received non-dict item in positions list: {pos_raw}")
                continue

            # Match symbol rigorously (CCXT symbol vs position symbol from 'symbol' or 'info')
            position_symbol = pos_raw.get('symbol')
            if position_symbol != symbol:
                lg.debug(f"Skipping position entry, symbol mismatch: Expected '{symbol}', got '{position_symbol}'")
                continue

            try:
                 # Get position size ('contracts' is standard CCXT, 'size' is common in Bybit 'info')
                 # Use Decimal for size comparison
                 pos_size_str = pos_raw.get('contracts', pos_raw.get('info', {}).get('size'))
                 if pos_size_str is None or str(pos_size_str).strip() == "":
                     lg.debug(f"Skipping position entry for {symbol}: Missing or empty size field. Data: {pos_raw.get('info', pos_raw)}")
                     continue

                 pos_size = Decimal(str(pos_size_str))

                 # --- Skip zero size positions ---
                 if pos_size.is_zero():
                     # lg.debug(f"Skipping zero size position for {symbol}.")
                     continue

                 # --- Standardize the position dictionary ---
                 standardized_pos = pos_raw.copy() # Start with the original data from CCXT
                 standardized_pos['side'] = 'long' if pos_size > 0 else 'short'
                 standardized_pos['contracts'] = abs(pos_size) # Store absolute size as Decimal

                 # Helper to safely convert string price/pnl to Decimal
                 def safe_decimal(value_str: Optional[Union[str, float, int]]) -> Optional[Decimal]:
                     if value_str is None or str(value_str).strip() == "": return None
                     try:
                         d = Decimal(str(value_str))
                         return d if d.is_finite() else None
                     except (InvalidOperation, TypeError): return None

                 # Standardize entry price ('entryPrice' is CCXT, 'avgPrice' is Bybit V5 info)
                 entry_price_str = standardized_pos.get('entryPrice', pos_raw.get('info',{}).get('avgPrice'))
                 standardized_pos['entryPrice'] = safe_decimal(entry_price_str)

                 # Add other useful fields, converting to Decimal/float where appropriate
                 standardized_pos['liqPrice'] = safe_decimal(standardized_pos.get('liquidationPrice', pos_raw.get('info',{}).get('liqPrice')))
                 standardized_pos['unrealizedPnl'] = safe_decimal(standardized_pos.get('unrealizedPnl', pos_raw.get('info',{}).get('unrealisedPnl'))) # Note Bybit spelling

                 # Leverage ('leverage' in CCXT, 'leverage' in Bybit info) - store as float
                 leverage_str = standardized_pos.get('leverage', pos_raw.get('info',{}).get('leverage'))
                 try: standardized_pos['leverage'] = float(leverage_str) if leverage_str else None
                 except (ValueError, TypeError): standardized_pos['leverage'] = None

                 # Position Index (Crucial for Hedge Mode) - store as int
                 pos_idx_str = pos_raw.get('info',{}).get('positionIdx')
                 try: standardized_pos['positionIdx'] = int(pos_idx_str) if pos_idx_str is not None else 0 # Default 0 (One-Way)
                 except (ValueError, TypeError): standardized_pos['positionIdx'] = 0

                 standardized_pos['market_info'] = market_info # Add market info for convenience

                 # --- Log Found Position ---
                 entry_log = f"{standardized_pos['entryPrice']:.{market_info.get('price_precision_digits', 4)}f}" if standardized_pos.get('entryPrice') else 'N/A'
                 liq_log = f"Liq={standardized_pos['liqPrice']:.{market_info.get('price_precision_digits', 4)}f}" if standardized_pos.get('liqPrice') else ''
                 pnl_log = f"PnL={standardized_pos['unrealizedPnl']:.2f}" if standardized_pos.get('unrealizedPnl') else ''
                 lev_log = f"Lev={standardized_pos.get('leverage')}x" if standardized_pos.get('leverage') else ''
                 idx_log = f"Idx={standardized_pos.get('positionIdx')}"

                 lg.info(f"Found active {standardized_pos['side']} position for {symbol}: Size={standardized_pos['contracts']}, Entry={entry_log}, {liq_log} {pnl_log} {lev_log} {idx_log}")
                 return standardized_pos # Return the first non-zero matching position found

            except (InvalidOperation, ValueError, TypeError) as e:
                 lg.error(f"Could not parse position data for {symbol}: {e}. Data: {pos_raw}")
            except Exception as e:
                 lg.error(f"Unexpected error processing position entry for {symbol}: {e}. Data: {pos_raw}", exc_info=True)

        # If loop completes without returning a position
        lg.debug(f"No active non-zero position found for {symbol}.")
        return None

    except Exception as e:
        # Catch errors from safe_ccxt_call or other issues
        lg.error(f"{NEON_RED}Error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger, market_info: Dict) -> bool:
    """
    Sets leverage for a symbol using Bybit V5 API via safe_ccxt_call.
    Handles the "leverage not modified" response as success.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        leverage: Target leverage (integer > 0).
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        True if leverage was set successfully (or was already correct), False otherwise.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID

    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping leverage setting for non-derivative: {symbol}")
        return True # Success as no action needed
    if not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} doesn't support setLeverage()."); return False
    if leverage <= 0:
        lg.error(f"Invalid leverage ({leverage}) for {symbol}. Must be > 0."); return False

    try:
        # Bybit V5 requires category, symbol, and separate buy/sell leverage values
        # We set both to the same value.
        params = {
            'category': category,
            'symbol': market_id,
            'buyLeverage': str(leverage), # API expects string value
            'sellLeverage': str(leverage)
        }
        lg.info(f"Attempting to set leverage for {symbol} (MarketID: {market_id}) to {leverage}x...")
        lg.debug(f"Leverage Params: {params}")

        # CCXT's set_leverage should map to the correct V5 endpoint /v5/position/set-leverage
        # Pass leverage as float (required by CCXT type hint), symbol, and V5 params.
        # safe_ccxt_call handles "leverage not modified" (110043) as success (returns {}).
        result = safe_ccxt_call(exchange, 'set_leverage', lg, leverage=float(leverage), symbol=symbol, params=params)

        # Check result: Success could be a dict with info, or empty dict for 'not modified'
        if result is not None and isinstance(result, dict):
            # Check Bybit's retCode in the info field if available
            ret_code = result.get('info', {}).get('retCode')
            if ret_code == 0 or result == {}: # Success or "not modified"
                 lg.info(f"{NEON_GREEN}Leverage set successfully (or already correct) for {symbol} to {leverage}x.{RESET}")
                 return True
            else:
                 # Should generally be caught by safe_ccxt_call non-retryable logic, but double-check
                 ret_msg = result.get('info', {}).get('retMsg', 'Unknown Error')
                 lg.error(f"{NEON_RED}set_leverage call returned success but with error code {ret_code}: '{ret_msg}' for {symbol}.{RESET}")
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
    amount: Decimal, price: Optional[Decimal] = None, params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Creates an order using safe_ccxt_call, handling V5 params and Decimal->float/str conversion.
    Includes basic parameter validation and logging.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        order_type: 'market' or 'limit'.
        side: 'buy' or 'sell'.
        amount: Order quantity (Decimal). Must be positive.
        price: Order price for limit orders (Decimal). Must be positive if provided.
        params: Additional parameters for the CCXT create_order call
                (e.g., {'reduceOnly': True, 'positionIdx': 0/1/2}).
        logger: Logger instance.
        market_info: Market information dictionary (required).

    Returns:
        The CCXT order dictionary if successful (API call successful and retCode=0),
        otherwise None.
    """
    lg = logger or get_logger('main') # Use provided logger or default main logger
    if not market_info:
        lg.error(f"Market info required for create_order ({symbol}) but not provided."); return None

    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID for API calls
    price_digits = market_info.get('price_precision_digits', 8)
    amount_digits = market_info.get('amount_precision_digits', 8)

    # --- Input Validation ---
    if not category:
        lg.error(f"Unknown category for {symbol}. Cannot place order."); return None
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount <= 0:
        lg.error(f"Order amount must be a positive Decimal ({symbol}, Amount: {amount})"); return None
    order_type_lower = order_type.lower(); side_lower = side.lower()
    if order_type_lower not in ['market', 'limit']:
        lg.error(f"Invalid order type '{order_type}'. Must be 'market' or 'limit'."); return None
    if side_lower not in ['buy', 'sell']:
        lg.error(f"Invalid order side '{side}'. Must be 'buy' or 'sell'."); return None
    if order_type_lower == 'limit':
        if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
            lg.error(f"Valid positive Decimal price required for limit order ({symbol}, Price: {price})"); return None
        price_float = float(price) # Convert valid Decimal price to float for CCXT
    else: # Market order
        price = None # Ensure price is None for market orders
        price_float = None

    # --- Format amount/price strings for logging ---
    # Use Decimal formatting capabilities for accurate string representation
    amount_str = f"{amount:.{amount_digits}f}"
    price_str = f"{price:.{price_digits}f}" if order_type_lower == 'limit' and price else None

    # --- Prepare V5 Parameters ---
    # Base parameters required by Bybit V5 createOrder
    order_params: Dict[str, Any] = {'category': category}
    # Merge external params (like reduceOnly, positionIdx) provided by caller
    # Caller is responsible for providing correct `positionIdx` based on position_mode config
    if params:
        order_params.update(params)
        # Ensure positionIdx is int if provided
        if 'positionIdx' in order_params:
             try: order_params['positionIdx'] = int(order_params['positionIdx'])
             except (ValueError, TypeError): lg.error(f"Invalid positionIdx type in params: {order_params['positionIdx']}") ; return None

    # --- Convert Amount Decimal to Float for CCXT call ---
    try:
        # CCXT methods typically expect float for amount
        amount_float = float(amount_str)
    except ValueError as e:
        lg.error(f"Error converting amount '{amount_str}' to float ({symbol}): {e}"); return None

    # --- Place Order via safe_ccxt_call ---
    try:
        log_price_part = f'@ {price_str}' if price_str else 'at Market'
        log_param_part = f" Params: {order_params}" if order_params else ""
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
            ret_code = order_result.get('info', {}).get('retCode')
            ret_msg = order_result.get('info', {}).get('retMsg', 'Unknown Status')

            if ret_code == 0:
                 lg.info(f"{NEON_GREEN}Successfully created {side.upper()} {order_type.upper()} order for {symbol}. Order ID: {order_id}{RESET}")
                 lg.debug(f"Order Result Info: {order_result.get('info')}")
                 # TODO: Optionally parse and return a more standardized order dict here if needed elsewhere
                 return order_result # Return the full CCXT order dict on success
            else:
                 # Order ID might be generated even if rejected (e.g., insufficient balance)
                 lg.error(f"{NEON_RED}Order placement potentially failed or rejected ({symbol}). Order ID: {order_id}, Code={ret_code}, Msg='{ret_msg}'.{RESET}")
                 lg.debug(f"Failed Order Result Info: {order_result.get('info')}")
                 # Provide hints for common rejection codes
                 if ret_code == 110007: lg.warning(f"Hint: Order rejected due to insufficient balance (Code {ret_code}).")
                 elif ret_code == 110017: lg.warning(f"Hint: Order rejected due to price/qty precision error (Code {ret_code}). Check market limits.")
                 elif ret_code == 110045 or ret_code == 170007: lg.warning(f"Hint: Order rejected due to risk limit exceeded (Code {ret_code}).")
                 elif ret_code == 110025: lg.warning(f"Hint: Order rejected due to positionIdx mismatch (Code {ret_code}). Ensure Hedge Mode params are correct.")
                 return None # Treat non-zero retCode as failure
        elif order_result:
             # Call succeeded but response format is unexpected (e.g., missing ID)
             lg.error(f"Order API call successful but response missing ID or invalid format ({symbol}). Response: {order_result}")
             return None
        else: # safe_ccxt_call returned None or raised an exception handled within it
             lg.error(f"Order API call failed or returned None ({symbol}) after retries.")
             return None

    except Exception as e:
        # Catch any unexpected error during the process
        lg.error(f"{NEON_RED}Failed to create order ({symbol}): {e}{RESET}", exc_info=True)
        return None


def set_protection_ccxt(
    exchange: ccxt.Exchange, symbol: str,
    stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
    trailing_stop_price: Optional[Decimal] = None, # TSL *distance/offset* value (e.g., 100 for $100 behind)
    trailing_active_price: Optional[Decimal] = None, # TSL *activation price* trigger
    position_idx: int = 0, # Required for Hedge mode (0=OneWay, 1=BuyHedge, 2=SellHedge)
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> bool:
    """
    Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL) for a position
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
        market_info: Market information dictionary (required).

    Returns:
        True if the protection was set successfully (API call returns retCode=0), False otherwise.
    """
    lg = logger or get_logger('main')
    if not market_info:
        lg.error(f"Market info required for set_protection ({symbol}) but not provided."); return False

    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID for API calls
    price_digits = market_info.get('price_precision_digits', 8)

    if not category or category not in ['linear', 'inverse']:
        lg.warning(f"Cannot set protection for non-derivative {symbol}. Category: {category}"); return False

    # --- Prepare V5 Parameters ---
    params: Dict[str, Any] = {
        'category': category,
        'symbol': market_id,
        'positionIdx': position_idx, # Crucial for Hedge Mode, 0 for One-Way
        # --- Optional Parameters (Defaults are usually sufficient) ---
        # 'tpslMode': 'Full', # Apply TP/SL to the entire position ('Partial' also available)
        # 'slTriggerBy': 'LastPrice', # MarkPrice, IndexPrice
        # 'tpTriggerBy': 'LastPrice', # MarkPrice, IndexPrice
        # 'slOrderType': 'Market', # Default is Market, 'Limit' also possible
        # 'tpOrderType': 'Market', # Default is Market, 'Limit' also possible
    }

    # --- Format Price Values ---
    # Bybit API expects prices/distances as strings. "0" is used to cancel/remove existing protection.
    def format_value(value: Optional[Decimal], name: str) -> str:
        """Formats Decimal to string for API, validating positivity. Returns '0' if invalid/None."""
        if value is not None and isinstance(value, Decimal) and value.is_finite():
            if value > 0:
                # Format based on whether it's a price or distance (TSL distance needs careful formatting)
                # For now, use standard price precision for all. May need adjustment for TSL distance formatting if API is strict.
                return f"{value:.{price_digits}f}"
            elif value == 0:
                return "0" # Explicitly setting to zero means cancel
            else: # Negative value
                 lg.warning(f"Invalid negative value '{value}' provided for '{name}' in set_protection. Using '0'.")
                 return "0"
        else: # None or non-finite Decimal
            return "0" # Treat as "do not set" or "cancel"

    sl_str = format_value(stop_loss_price, "stopLoss")
    tp_str = format_value(take_profit_price, "takeProfit")
    tsl_dist_str = format_value(trailing_stop_price, "trailingStop")
    tsl_act_str = format_value(trailing_active_price, "activePrice")

    # Add to params only if a value is being set (not "0") or if explicitly cancelling
    action_taken = False
    if sl_str != "0" or (stop_loss_price is not None and stop_loss_price == 0):
        params['stopLoss'] = sl_str; action_taken = True
    if tp_str != "0" or (take_profit_price is not None and take_profit_price == 0):
        params['takeProfit'] = tp_str; action_taken = True

    # Trailing Stop: 'trailingStop' is the distance. 'activePrice' is optional trigger.
    if tsl_dist_str != "0" or (trailing_stop_price is not None and trailing_stop_price == 0):
        params['trailingStop'] = tsl_dist_str
        action_taken = True
        # Only include activePrice if TSL distance is non-zero. If distance is "0", activePrice is ignored/invalid.
        if tsl_dist_str != "0":
            params['activePrice'] = tsl_act_str # "0" means immediate activation
        elif 'activePrice' in params:
            # Ensure activePrice is not sent if trailingStop is "0" (API might reject)
             del params['activePrice']

    # --- Log Intention ---
    log_parts = []
    if 'stopLoss' in params: log_parts.append(f"SL={params['stopLoss']}")
    if 'takeProfit' in params: log_parts.append(f"TP={params['takeProfit']}")
    if 'trailingStop' in params:
        tsl_log = f"TSL_Dist={params['trailingStop']}"
        if params.get('activePrice', "0") != "0": tsl_log += f", ActP={params['activePrice']}"
        elif params['trailingStop'] != "0": tsl_log += ", Act=Immediate" # Only relevant if TSL dist > 0
        log_parts.append(tsl_log)

    if not action_taken:
        lg.info(f"No valid protection levels provided or changes needed for set_protection ({symbol}). No API call made.")
        # Consider it success if no action was needed
        return True

    # --- Make API Call ---
    try:
        lg.info(f"Attempting to set protection for {symbol} (Idx: {position_idx}): {', '.join(log_parts)}")
        lg.debug(f"Protection Params: {params}")

        # Explicitly use the private POST method mapped in exchange options
        # This ensures we hit the correct V5 endpoint: /v5/position/trading-stop
        method_to_call = 'private_post_position_trading_stop'
        if not hasattr(exchange, method_to_call):
            # Fallback might be needed if CCXT mapping changes, but rely on initialization mapping first.
            lg.error(f"Method '{method_to_call}' not found directly on exchange object. Ensure it's mapped in exchange options or CCXT version is compatible.")
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
            # Provide hints for common errors
            if ret_code == 170140 or ret_code == 110024: lg.warning(f"Hint: Set protection failed because position might not exist (Code {ret_code}).")
            elif ret_code == 110025: lg.warning(f"Hint: Set protection failed due to positionIdx mismatch (Code {ret_code}). Check Hedge Mode settings.")
            elif ret_code == 170131 or ret_code == 110042: lg.warning(f"Hint: Set protection failed due to invalid TP/SL price (Code {ret_code}). Check price vs mark/index.")
            elif ret_code == 170133: lg.warning(f"Hint: Cannot set TP/SL/TSL, potentially due to order status or existing orders (Code {ret_code}).")
            return False
        else: # safe_ccxt_call returned None (likely hit max retries or raised internal error)
            lg.error(f"Set protection API call failed or returned None ({symbol}) after retries.")
            return False

    except Exception as e:
        # Catch unexpected errors during the process
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
        return False


def close_position_ccxt(
    exchange: ccxt.Exchange, symbol: str, position_data: Dict,
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Closes an existing position via a Market order with 'reduceOnly' flag.
    Uses position data to determine side, size, and position index for closing.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        position_data: The standardized position dictionary obtained from fetch_positions_ccxt.
                       Must contain 'side' ('long'/'short'), 'contracts' (Decimal size),
                       and 'positionIdx' (int, especially for Hedge Mode).
        logger: Logger instance.
        market_info: Market information dictionary (required).

    Returns:
        The CCXT order dictionary for the closing order if placed successfully (retCode=0),
        otherwise None.
    """
    lg = logger or get_logger('main')
    if not market_info:
        lg.error(f"Market info required for close_position ({symbol}) but not provided."); return None
    if not position_data or not isinstance(position_data, dict):
        lg.error(f"Valid position data dictionary required for close_position ({symbol})"); return None

    try:
        position_side = position_data.get('side') # 'long' or 'short'
        position_size_dec = position_data.get('contracts') # Absolute Decimal size
        position_idx = position_data.get('positionIdx') # Integer index (0 for One-Way, 1/2 for Hedge)

        # Validate position data needed for closing
        if position_side not in ['long', 'short']:
            lg.error(f"Invalid side ('{position_side}') in position data for closing {symbol}"); return None
        if not isinstance(position_size_dec, Decimal) or position_size_dec <= 0:
             lg.error(f"Invalid size ('{position_size_dec}') in position data for closing {symbol}"); return None
        if position_idx is None or not isinstance(position_idx, int):
             lg.error(f"Missing or invalid positionIdx ('{position_idx}') in position data for closing {symbol}"); return None

        # Determine the side of the closing order (opposite of position side)
        close_side = 'sell' if position_side == 'long' else 'buy'
        amount_to_close = position_size_dec # Use the absolute size from position data

        lg.info(f"Attempting to close {position_side} position ({symbol}, Size: {amount_to_close}, Idx: {position_idx}) via {close_side.upper()} MARKET order...")

        # --- Prepare Parameters for Closing Order ---
        # 'reduceOnly': True ensures the order only closes or reduces the position.
        # 'positionIdx': Crucial for identifying which position to close in Hedge Mode.
        close_params: Dict[str, Any] = {
            'reduceOnly': True,
            'positionIdx': position_idx
        }

        # Use create_order_ccxt to place the closing market order
        close_order_result = create_order_ccxt(
            exchange=exchange, symbol=symbol, order_type='market', side=close_side,
            amount=amount_to_close, params=close_params, logger=lg, market_info=market_info
        )

        # Check if the closing order was placed successfully (create_order_ccxt checks retCode=0)
        if close_order_result and close_order_result.get('id'):
            lg.info(f"{NEON_GREEN}Successfully placed MARKET order to close {position_side} position ({symbol}, Idx: {position_idx}). Close Order ID: {close_order_result.get('id')}{RESET}")
            return close_order_result
        else:
            lg.error(f"{NEON_RED}Failed to place market order to close position ({symbol}, Idx: {position_idx}).{RESET}")
            # Check logs from create_order_ccxt for specific reason (e.g., insufficient funds, position already closed)
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Error processing position data for closing ({symbol}): {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error attempting to close position ({symbol}): {e}{RESET}", exc_info=True)
        return None


# --- Main Bot Logic ---
def run_bot(exchange: ccxt.Exchange, config: Dict[str, Any], bot_state: Dict[str, Any]) -> None:
    """Main execution loop of the trading bot."""
    main_logger = get_logger('main')
    main_logger.info(f"{NEON_CYAN}=== Starting Enhanced Trading Bot v{BOT_VERSION} (PID: {os.getpid()}) ===")

    # --- Log Initial Bot Configuration ---
    trading_status = f"{NEON_GREEN}Enabled{RESET}" if config.get('enable_trading') else f"{NEON_YELLOW}DISABLED{RESET}"
    sandbox_status = f"{NEON_YELLOW}ACTIVE{RESET}" if config.get('use_sandbox') else f"{NEON_RED}INACTIVE (LIVE!){RESET}"
    account_type = "UNIFIED" if IS_UNIFIED_ACCOUNT else "Non-UTA (Contract/Spot)"
    main_logger.info(f"Mode: Trading={trading_status}, Sandbox={sandbox_status}, Account={account_type}")
    main_logger.info(f"Config: Symbols={config.get('symbols')}, Interval={config.get('interval')}, Quote={QUOTE_CURRENCY}, WeightSet='{config.get('active_weight_set')}'")
    main_logger.info(f"Risk: {config.get('risk_per_trade')*100:.2f}%, Leverage={config.get('leverage')}x, MaxPos={config.get('max_concurrent_positions_total')}")
    main_logger.info(f"Features: TSL={'On' if config.get('enable_trailing_stop') else 'Off'}, BE={'On' if config.get('enable_break_even') else 'Off'}, MACrossExit={'On' if config.get('enable_ma_cross_exit') else 'Off'}")
    main_logger.info(f"Position Mode: {config.get('position_mode', 'One-Way')} (Note: Hedge mode requires config alignment and careful param setting)")

    global LOOP_DELAY_SECONDS
    LOOP_DELAY_SECONDS = int(config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS))

    symbols_to_trade: List[str] = config.get("symbols", [])

    # --- Initialize/Validate state dictionary for each symbol ---
    for symbol in symbols_to_trade:
        if symbol not in bot_state: bot_state[symbol] = {}
        # Ensure default states exist (using .setdefault is idempotent)
        bot_state[symbol].setdefault("break_even_triggered", False)
        bot_state[symbol].setdefault("last_signal", "HOLD")
        # last_entry_price is stored as string if known, otherwise None
        bot_state[symbol].setdefault("last_entry_price", None)

    cycle_count = 0
    last_market_reload_time = getattr(exchange, 'last_load_markets_timestamp', 0)

    # --- Main Loop ---
    while True:
        cycle_count += 1
        start_time = time.monotonic() # Use monotonic clock for duration measurement
        main_logger.info(f"{NEON_BLUE}--- Starting Bot Cycle {cycle_count} ---{RESET}")

        # --- Periodic Market Reload ---
        if time.time() - last_market_reload_time > MARKET_RELOAD_INTERVAL_SECONDS:
            main_logger.info(f"Reloading exchange markets (Interval: {MARKET_RELOAD_INTERVAL_SECONDS}s)...")
            try:
                exchange.load_markets(True) # Force reload
                last_market_reload_time = time.time()
                exchange.last_load_markets_timestamp = last_market_reload_time # Update timestamp on exchange object too
                main_logger.info("Markets reloaded successfully.")
            except Exception as e:
                main_logger.error(f"Failed to reload markets: {e}", exc_info=True)
                # Continue loop, but market info might be stale. Calculations using it might fail.

        # --- Pre-Cycle Checks (Balance, Positions) ---
        current_balance: Optional[Decimal] = None
        if config.get("enable_trading"):
            try:
                current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
                if current_balance is None:
                    main_logger.error(f"{NEON_RED}Failed to fetch {QUOTE_CURRENCY} balance. Trading actions may fail or use stale data.{RESET}")
                elif current_balance <= 0:
                    main_logger.warning(f"{NEON_YELLOW}Available {QUOTE_CURRENCY} balance is {current_balance:.4f}. Cannot open new positions.{RESET}")
                else:
                    main_logger.info(f"Available {QUOTE_CURRENCY} balance: {current_balance:.4f}")
            except Exception as e:
                 main_logger.error(f"Error fetching balance at start of cycle: {e}", exc_info=True)
                 # Proceed cautiously, size calculation will likely fail if balance is None

        # Fetch all active positions for configured symbols at the start of the cycle
        open_positions_count = 0
        active_positions: Dict[str, Dict] = {} # Stores standardized position data {symbol: position_dict}
        main_logger.debug("Fetching active positions for configured symbols...")
        for symbol in symbols_to_trade:
            temp_logger = get_logger(symbol, is_symbol_logger=True) # Use symbol-specific logger
            try:
                market_info = get_market_info(exchange, symbol, temp_logger)
                if not market_info:
                     temp_logger.warning(f"Skipping position check for {symbol}: Could not get market info.")
                     continue
                # Only fetch positions for derivatives (contracts)
                if market_info.get('is_contract'):
                    position = fetch_positions_ccxt(exchange, symbol, temp_logger, market_info)
                    if position: # fetch_positions_ccxt returns None if no non-zero position
                        open_positions_count += 1
                        active_positions[symbol] = position
                        # --- State Synchronization ---
                        # Update state's entry price if missing and position exists
                        # Use the entry price from the fetched position as the source of truth if state is missing or differs significantly?
                        entry_p_state_str = bot_state[symbol].get("last_entry_price")
                        entry_p_pos = position.get('entryPrice') # Decimal
                        if isinstance(entry_p_pos, Decimal):
                            if entry_p_state_str is None:
                                bot_state[symbol]["last_entry_price"] = str(entry_p_pos)
                                temp_logger.info(f"State updated: last_entry_price set from fetched position: {entry_p_pos}")
                            # Optional: Check for significant difference and update/warn?
                            # else:
                            #     try: entry_p_state_dec = Decimal(entry_p_state_str)
                            #     except: entry_p_state_dec = None
                            #     if entry_p_state_dec and abs(entry_p_state_dec - entry_p_pos) / entry_p_pos > Decimal('0.01'): # 1% diff
                            #          temp_logger.warning(f"State entry price ({entry_p_state_str}) differs >1% from fetched ({entry_p_pos}). Updating state.")
                            #          bot_state[symbol]["last_entry_price"] = str(entry_p_pos)
                        elif entry_p_state_str is not None:
                             # Position exists but entry price missing from fetch? Unlikely but possible. Keep state price.
                             temp_logger.warning(f"Position fetched for {symbol}, but entry price missing. Keeping state entry: {entry_p_state_str}")
            except Exception as fetch_pos_err:
                 temp_logger.error(f"Error during position pre-check for {symbol}: {fetch_pos_err}", exc_info=True)

        max_allowed_positions = int(config.get("max_concurrent_positions_total", 1))
        main_logger.info(f"Currently open positions: {open_positions_count} / {max_allowed_positions}")

        # --- Symbol Processing Loop ---
        for symbol in symbols_to_trade:
            symbol_logger = get_logger(symbol, is_symbol_logger=True)
            symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
            symbol_state = bot_state[symbol] # Get reference to this symbol's mutable state dict

            try:
                # Get Market Info (potentially reloaded)
                market_info = get_market_info(exchange, symbol, symbol_logger)
                if not market_info:
                    symbol_logger.error(f"Could not get market info for {symbol}. Skipping cycle for this symbol.")
                    continue

                # Fetch Latest Data (Klines, Price, Orderbook)
                timeframe = config.get("interval", "5")
                # Determine required kline limit based on longest indicator period + buffer
                # Use validated integer/float parameters from config
                periods = [int(v) for k, v in config.items() if ('_period' in k or '_window' in k) and isinstance(v, (int, float)) and v > 0]
                base_limit = max(periods) if periods else 100 # Use 100 as a fallback minimum
                kline_limit = base_limit + 50 # Add buffer for calculation stability
                symbol_logger.debug(f"Required kline limit: {kline_limit} (Base: {base_limit})")

                df_raw = fetch_klines_ccxt(exchange, symbol, timeframe, kline_limit, symbol_logger, market_info)
                # Check if enough data was fetched for the longest period
                if df_raw.empty or len(df_raw) < base_limit:
                    symbol_logger.warning(f"Kline data insufficient ({len(df_raw)} rows, need ~{base_limit}). Skipping analysis for {symbol}.")
                    continue

                current_price_dec = fetch_current_price_ccxt(exchange, symbol, symbol_logger, market_info)
                if current_price_dec is None:
                    symbol_logger.warning(f"Current price unavailable for {symbol}. Skipping analysis.")
                    continue

                orderbook = None
                if config.get("indicators", {}).get("orderbook"):
                    try:
                        ob_limit = int(config.get("orderbook_limit", 25))
                        orderbook = fetch_orderbook_ccxt(exchange, symbol, ob_limit, symbol_logger, market_info)
                    except Exception as ob_err:
                        symbol_logger.warning(f"Failed to fetch order book for {symbol}: {ob_err}")

                # --- Initialize Analyzer (Calculates indicators using df_raw) ---
                try:
                    # Pass the mutable symbol_state dictionary
                    analyzer = TradingAnalyzer(df_raw, symbol_logger, config, market_info, symbol_state)
                except ValueError as analyze_init_err:
                    symbol_logger.error(f"Analyzer initialization failed: {analyze_init_err}. Skipping analysis for {symbol}.")
                    continue
                except Exception as analyze_err:
                    symbol_logger.error(f"Unexpected analyzer initialization error: {analyze_err}", exc_info=True)
                    continue # Skip symbol if analyzer fails

                # --- Manage Existing Position ---
                # Use the position data fetched at the start of the cycle
                current_position = active_positions.get(symbol)
                position_closed_in_manage = False
                if current_position:
                    pos_side = current_position.get('side','?'); pos_size = current_position.get('contracts','?')
                    entry_p_log = f"{current_position.get('entryPrice'):.{market_info.get('price_precision_digits', 4)}f}" if current_position.get('entryPrice') else 'N/A'
                    symbol_logger.info(f"Managing existing {pos_side} position (Size: {pos_size}, Entry: {entry_p_log}).")

                    # Pass analyzer (which holds state reference) and position data
                    position_closed_in_manage = manage_existing_position(
                        exchange, config, symbol_logger, analyzer, current_position, current_price_dec
                    )

                    if position_closed_in_manage:
                        active_positions.pop(symbol, None) # Remove from active list for this cycle
                        open_positions_count = max(0, open_positions_count - 1) # Decrement count
                        symbol_logger.info(f"Position for {symbol} closed during management routine.")
                        # Skip to next symbol as no further action (like new entry) is needed
                        symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---"); time.sleep(0.1); continue
                    else:
                        # If not closed, update the active_positions dict with potentially modified state
                        # (e.g., if BE SL was set, the fetched position next cycle should reflect it)
                        # No direct update needed here, next cycle's fetch will get the latest state.
                        symbol_logger.debug(f"Position for {symbol} remains open after management checks.")

                # --- Check for New Entry ---
                # Conditions:
                # 1. No active position for this symbol currently (either never existed or closed in manage step)
                # 2. It's a contract market (spot trading logic not implemented)
                # 3. Total open positions count is less than the configured maximum
                if not active_positions.get(symbol) and market_info.get('is_contract'):
                    if open_positions_count < max_allowed_positions:
                        symbol_logger.info("No active position. Checking for new entry signals...")
                        # Reset state flags if no position exists (ensure clean state for next potential entry)
                        if analyzer.break_even_triggered: analyzer.break_even_triggered = False # Reset via property
                        if symbol_state.get("last_entry_price") is not None: symbol_state["last_entry_price"] = None

                        # Generate signal based on latest data and indicators
                        signal = analyzer.generate_trading_signal(current_price_dec, orderbook)
                        symbol_state["last_signal"] = signal # Store latest signal regardless of entry

                        # Attempt entry if BUY/SELL signal and trading enabled
                        if signal in ["BUY", "SELL"]:
                            if config.get("enable_trading"):
                                # Ensure balance was fetched successfully and is positive
                                if current_balance is not None and current_balance > 0:
                                     opened_new = attempt_new_entry(
                                         exchange, config, symbol_logger, analyzer,
                                         signal, current_price_dec, current_balance
                                     )
                                     if opened_new:
                                         open_positions_count += 1 # Increment count if entry successful
                                         # Mark position as active for this cycle to prevent immediate re-entry logic?
                                         # Fetching positions at start of next cycle is the main source of truth.
                                         # To be safe, maybe add a placeholder to active_positions?
                                         active_positions[symbol] = {"symbol": symbol, "side": signal.lower(), "contracts": "PENDING_FETCH", "entryPrice": "PENDING_FETCH"} # Placeholder
                                else:
                                    symbol_logger.warning(f"Trading enabled but balance unavailable or zero ({current_balance}). Cannot enter {signal} trade for {symbol}.")
                            else: # Trading disabled
                                symbol_logger.info(f"Entry signal '{signal}' generated for {symbol}, but trading is disabled.")
                    else: # Max positions reached
                        symbol_logger.info(f"Max positions ({open_positions_count}/{max_allowed_positions}) reached. Skipping new entry check for {symbol}.")
                elif not market_info.get('is_contract'):
                    symbol_logger.debug(f"Symbol {symbol} is not a contract market. Skipping position management/entry logic.")
                # Else (position still exists after management check) -> do nothing, wait for next cycle

            except Exception as e:
                symbol_logger.error(f"{NEON_RED}!!! Unhandled error in main loop for symbol {symbol}: {e} !!!{RESET}", exc_info=True)
                # Continue to the next symbol
            finally:
                symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
                # Optional small delay between processing each symbol to avoid API bursts if loop is very fast
                time.sleep(0.2)

        # --- Post-Cycle ---
        end_time = time.monotonic()
        cycle_duration = end_time - start_time
        main_logger.info(f"{NEON_BLUE}--- Bot Cycle {cycle_count} Finished (Duration: {cycle_duration:.3f}s) ---{RESET}")

        # Save state after each full cycle (includes BE status, last entry price, etc.)
        save_state(STATE_FILE, bot_state, main_logger)

        # Calculate wait time for next cycle
        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0:
            main_logger.info(f"Waiting {wait_time:.2f}s for next cycle (Target Loop Time: {LOOP_DELAY_SECONDS}s)...")
            time.sleep(wait_time)
        else:
            main_logger.warning(f"Cycle duration ({cycle_duration:.3f}s) exceeded loop delay ({LOOP_DELAY_SECONDS}s). Starting next cycle immediately.")


def manage_existing_position(
    exchange: ccxt.Exchange, config: Dict[str, Any], logger: logging.Logger,
    analyzer: TradingAnalyzer, position_data: Dict, current_price_dec: Decimal
) -> bool:
    """
    Manages an existing position based on configured exit/management rules.
    1. Checks for MA Cross Exit signal.
    2. Checks for Break-Even trigger and updates SL if met.
    (Note: Trailing Stop Loss (TSL) is primarily managed by the exchange after initial setting via `set_protection_ccxt`.
     This function focuses on MA Cross and Break-Even logic.)

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance for the symbol.
        analyzer: TradingAnalyzer instance with current data and access to shared state.
        position_data: Standardized dictionary of the current position (from fetch_positions_ccxt).
        current_price_dec: Current market price (Decimal).

    Returns:
        bool: True if the position was closed during this management check, False otherwise.
    """
    symbol = position_data.get('symbol')
    position_side = position_data.get('side') # 'long' or 'short'
    entry_price = position_data.get('entryPrice') # Decimal or None
    pos_size = position_data.get('contracts') # Decimal or None
    position_idx = position_data.get('positionIdx') # int (0, 1, or 2)
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state # Access shared state via analyzer

    # --- Validate Inputs ---
    if not all([symbol, position_side, isinstance(entry_price, Decimal), isinstance(pos_size, Decimal), isinstance(position_idx, int)]) or pos_size <= 0:
        logger.error(f"Invalid position data received for management: Symbol={symbol}, Side={position_side}, Entry={entry_price}, Size={pos_size}, Idx={position_idx}"); return False
    if not current_price_dec.is_finite() or current_price_dec <= 0:
         logger.warning(f"Invalid current price ({current_price_dec}) for managing {symbol}"); return False

    position_closed = False

    try:
        # --- 1. Check MA Cross Exit ---
        if config.get("enable_ma_cross_exit"):
            ema_s_f = analyzer._get_indicator_float("EMA_Short")
            ema_l_f = analyzer._get_indicator_float("EMA_Long")
            if ema_s_f is not None and ema_l_f is not None:
                # Check for adverse cross: Short EMA crosses below Long EMA for a LONG position, or above for a SHORT position.
                # Add a small tolerance (e.g., 0.01%) to avoid flapping on near-equal EMAs.
                tolerance_factor = Decimal("0.0001") # 0.01%
                is_adverse_cross = False
                if position_side == 'long' and Decimal(str(ema_s_f)) < Decimal(str(ema_l_f)) * (Decimal(1) - tolerance_factor):
                    is_adverse_cross = True
                    logger.debug(f"MA Cross Check (Long): Short EMA {ema_s_f:.5f} < Long EMA {ema_l_f:.5f} -> Adverse Cross")
                elif position_side == 'short' and Decimal(str(ema_s_f)) > Decimal(str(ema_l_f)) * (Decimal(1) + tolerance_factor):
                    is_adverse_cross = True
                    logger.debug(f"MA Cross Check (Short): Short EMA {ema_s_f:.5f} > Long EMA {ema_l_f:.5f} -> Adverse Cross")

                if is_adverse_cross:
                    logger.warning(f"{NEON_YELLOW}MA Cross Exit Triggered for {position_side} {symbol}! Attempting market close.{RESET}")
                    if config.get("enable_trading"):
                        # Pass the full position_data dict which includes positionIdx
                        close_result = close_position_ccxt(exchange, symbol, position_data, logger, market_info)
                        if close_result:
                            logger.info(f"Position closed successfully via MA Cross for {symbol}.")
                            # Reset state relevant to the closed position
                            analyzer.break_even_triggered = False # Reset BE flag in state via property
                            symbol_state["last_signal"] = "HOLD" # Reset last signal state
                            symbol_state["last_entry_price"] = None # Clear last entry price
                            position_closed = True
                            return True # Exit management, position is closed
                        else:
                            logger.error(f"Failed to place MA Cross close order for {symbol}. Position remains open. Will retry next cycle."); return False
                    else: # Trading disabled
                        logger.info(f"MA Cross exit triggered for {symbol}, but trading is disabled.")
                        # Do not modify state if only simulating

        # --- 2. Check Break-Even (Only if not already triggered and position wasn't just closed by MA cross) ---
        if not position_closed and config.get("enable_break_even") and not analyzer.break_even_triggered:
            atr_val = analyzer.indicator_values.get("ATR") # Decimal
            if atr_val and atr_val.is_finite() and atr_val > 0:
                try:
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    profit_target_points = atr_val * trigger_multiple # Profit needed in price points
                    current_profit_points = Decimal('0')

                    if position_side == 'long': current_profit_points = current_price_dec - entry_price
                    else: current_profit_points = entry_price - current_price_dec

                    # Check if profit meets or exceeds the trigger level
                    if current_profit_points >= profit_target_points:
                        logger.info(f"{NEON_GREEN}Break-Even Trigger Met for {symbol}! (Profit Points: {current_profit_points:.{analyzer.get_price_precision_digits()}f} >= Target: {profit_target_points:.{analyzer.get_price_precision_digits()}f}){RESET}")
                        min_tick = analyzer.get_min_tick_size()
                        offset_ticks = int(config.get("break_even_offset_ticks", 2)) # Number of ticks for offset

                        if min_tick and offset_ticks >= 0:
                            offset_value = min_tick * Decimal(offset_ticks)
                            # Calculate BE stop price: Entry + Offset for Long, Entry - Offset for Short
                            be_stop_price_raw = entry_price + offset_value if position_side == 'long' else entry_price - offset_value
                            # Quantize BE stop price safely away from entry (towards profit zone)
                            rounding_mode = ROUND_UP if position_side == 'long' else ROUND_DOWN # Round towards profit
                            be_stop_price = analyzer.quantize_price(be_stop_price_raw, rounding=rounding_mode)

                            # Sanity check BE price: Ensure it's actually beyond entry after quantization and positive
                            if be_stop_price and be_stop_price.is_finite() and be_stop_price > 0:
                                valid_be_price = False
                                if position_side == 'long' and be_stop_price > entry_price: valid_be_price = True
                                elif position_side == 'short' and be_stop_price < entry_price: valid_be_price = True

                                if not valid_be_price:
                                     # If quantization resulted in price at or before entry, force it one tick beyond
                                     logger.warning(f"BE stop price {be_stop_price} was not beyond entry {entry_price} after quantization. Adjusting.")
                                     be_stop_price = entry_price + min_tick if position_side == 'long' else entry_price - min_tick
                                     # Re-quantize the adjusted price correctly
                                     be_stop_price = analyzer.quantize_price(be_stop_price, rounding=rounding_mode)
                                     if be_stop_price is None or (position_side == 'long' and be_stop_price <= entry_price) or (position_side == 'short' and be_stop_price >= entry_price):
                                         logger.error(f"Failed to calculate valid adjusted BE stop price. Cannot set BE SL.")
                                         be_stop_price = None # Mark as invalid

                            if be_stop_price: # If we have a valid BE price
                                logger.info(f"Calculated Break-Even Stop Price for {symbol}: {be_stop_price}")
                                if config.get("enable_trading"):
                                    # --- Get current TP/TSL state from position info to preserve them if needed ---
                                    # CCXT fetch_positions doesn't reliably return TP/SL/TSL unless specifically implemented.
                                    # Fetching active orders might be needed, but complex.
                                    # Assumption: We only modify SL here. If TSL is active, setting SL might cancel TSL on Bybit.
                                    # If break_even_force_fixed_sl=True (default), we intend to replace TSL with fixed BE SL.
                                    # If break_even_force_fixed_sl=False, we set SL, and TSL might remain active *if* exchange allows both. Bybit likely cancels TSL when SL is set via API.

                                    # We need to fetch current TP to avoid cancelling it unintentionally.
                                    # Fetching orders or relying on `position_data['info']` is needed.
                                    pos_info = position_data.get('info', {})
                                    tp_str = pos_info.get('takeProfit', "0") # Check info field from fetch_positions
                                    current_tp = analyzer.quantize_price(tp_str) if tp_str and tp_str != "0" else None

                                    # Parameters for set_protection: New SL, keep existing TP (if any), potentially remove TSL implicitly.
                                    logger.info(f"Setting BE SL={be_stop_price}, keeping existing TP={current_tp or 'None'}. TSL may be implicitly cancelled by API.")
                                    success = set_protection_ccxt(
                                        exchange, symbol,
                                        stop_loss_price=be_stop_price, # Set the new SL
                                        take_profit_price=current_tp, # Keep existing TP
                                        # Do not set TSL parameters here, effectively cancelling existing TSL if break_even_force_fixed_sl=True
                                        trailing_stop_price=None, # Explicitly None
                                        trailing_active_price=None,
                                        position_idx=position_idx, # Use index from current position
                                        logger=logger, market_info=market_info
                                    )
                                    if success:
                                        logger.info(f"{NEON_GREEN}Successfully set Break-Even SL for {symbol}.{RESET}")
                                        analyzer.break_even_triggered = True # Update state via property setter
                                    else:
                                        logger.error(f"{NEON_RED}Failed to set Break-Even SL via API for {symbol}. Will retry next cycle.{RESET}")
                                else: # Trading disabled
                                    logger.info(f"Break-Even triggered for {symbol}, but trading disabled. State updated.")
                                    analyzer.break_even_triggered = True # Update state even if not trading
                            else: # be_stop_price is None
                                logger.error(f"Invalid Break-Even stop price calculated ({be_stop_price}). Cannot set BE SL.")
                        else: # min_tick invalid or offset_ticks < 0
                            logger.error(f"Cannot calculate BE offset for {symbol}: Invalid min_tick ({min_tick}) or offset_ticks ({offset_ticks}).")
                except (InvalidOperation, ValueError, TypeError) as be_calc_err:
                     logger.error(f"Error during Break-Even calculation for {symbol}: {be_calc_err}")
                except Exception as be_err:
                     logger.error(f"Unexpected error during Break-Even check for {symbol}: {be_err}", exc_info=True)
            else: # ATR invalid
                logger.warning(f"Cannot check Break-Even trigger for {symbol}: Invalid ATR ({atr_val}).")

        # --- Other Management Logic (Future Enhancements) ---
        # e.g., Check if TSL is active and behaving as expected (requires fetching TSL status).
        # e.g., Dynamic TP adjustments based on market conditions.

    except Exception as e:
        logger.error(f"Unexpected error managing position {symbol}: {e}", exc_info=True)
        return False # Return False as we don't know if position was closed

    # Return True only if MA Cross explicitly closed the position in this call
    return position_closed


def attempt_new_entry(
    exchange: ccxt.Exchange, config: Dict[str, Any], logger: logging.Logger,
    analyzer: TradingAnalyzer, signal: str, entry_price_signal: Decimal, current_balance: Decimal
) -> bool:
    """
    Attempts to enter a new trade based on a BUY or SELL signal.
    Executes the full workflow: Calculate TP/SL -> Calculate Size -> Set
