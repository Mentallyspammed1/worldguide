"""Enhanced Multi-Symbol Trading Bot for Bybit (V5 API) - v1.0.2.

Merges features, optimizations, and best practices from previous versions.
Includes: pandas_ta.Strategy, Decimal precision, robust CCXT interaction,
          multi-symbol support, state management, TSL/BE logic, MA cross exit.

This script provides a framework for automated trading on Bybit's V5 API,
focusing on futures (linear/inverse). It incorporates multiple technical
indicators, weighted signal generation, risk management (ATR-based SL/TP,
position sizing), trailing stops, break-even adjustments, and MA cross exits.

Key Features:
- Multi-Symbol Trading: Processes multiple trading pairs concurrently.
- V5 API Integration: Utilizes Bybit's latest API version via CCXT.
- Unified Trading Account (UTA) Support: Auto-detects and handles UTA/Non-UTA.
- Advanced TA: Leverages pandas_ta.Strategy for flexible indicator calculation.
- Weighted Signals: Combines multiple indicator signals with configurable weights.
- Precise Calculations: Uses Python's Decimal type for financial accuracy.
- Robust Error Handling: Includes retries for API calls and handles common errors.
- State Persistence: Saves and loads bot state (e.g., BE status) between runs.
- Risk Management: Calculates position size based on risk % and SL distance.
- Dynamic SL/TP: ATR-based stop-loss and take-profit calculation.
- Trailing Stop Loss (TSL): Configurable TSL with activation price.
- Break-Even (BE): Moves SL to entry + offset after reaching a profit target.
- MA Cross Exit: Option to close positions based on EMA crossover.
- Configuration File: Uses JSON for easy parameter adjustments.
- Detailed Logging: Provides timestamped, timezone-aware logs with redaction.

Changes in v1.0.2:
- Added more explicit handling/logging for Unified Trading Accounts (UTA).
- Enhanced configuration validation on load (interval, numeric types).
- Improved robustness in fetch_balance parsing for UTA.
- Ensured consistent use of Decimal/float where appropriate (Decimal for calcs, float for CCXT).
- Added checks for SL/TP being too close to entry price.
- Added comments regarding Hedge Mode implementation points.
- Standardized logging messages and added library dependency comments.
- Refined state management for last_entry_price (stored as string).
- Minor code clarity improvements and type hinting additions.
- FIXED: AttributeError: module 'pandas.api.types' has no attribute 'is_decimal_dtype'.
- FIXED: pandas_ta warning "dtype incompatible with int64" by ensuring float conversion.
"""

# --- Required Libraries ---
# Ensure these are installed:
# pip install ccxt pandas numpy pandas_ta python-dotenv colorama pytz # (pytz needed if Python < 3.9)
# --------------------------

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any

# --- Timezone Handling ---
# Use zoneinfo (Python 3.9+) if available, fallback to pytz
try:
    from zoneinfo import ZoneInfo  # Preferred (Python 3.9+)
except ImportError:
    try:
        from pytz import timezone as ZoneInfo  # Fallback (pip install pytz)
    except ImportError:
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
    COLORAMA_AVAILABLE = False

    # Define dummy color variables if colorama is missing
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = DummyColor()
    Style = DummyColor()

    def init(*args: Any, **kwargs: Any) -> None:
        pass  # Dummy init function


from dotenv import load_dotenv

# --- Initialization ---
try:
    # Set precision for Decimal arithmetic operations.
    # Note: This affects calculations like Decimal * Decimal.
    # Storing and retrieving Decimals maintains their inherent precision.
    getcontext().prec = 36  # Sufficient precision for most financial calcs
except Exception:
    pass

if COLORAMA_AVAILABLE:
    init(autoreset=True)  # Initialize colorama (or dummy init)
load_dotenv()  # Load environment variables from .env file

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
except Exception:
    TIMEZONE = ZoneInfo("UTC")
    TZ_NAME = "UTC"

# --- API Interaction Constants ---
MAX_API_RETRIES = 4  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Initial delay before retrying API calls (exponential backoff)
RATE_LIMIT_BUFFER_SECONDS = 0.5  # Extra buffer added to wait time suggested by rate limit errors
MARKET_RELOAD_INTERVAL_SECONDS = 3600  # How often to reload exchange market data (1 hour)
POSITION_CONFIRM_DELAY = 10  # Seconds to wait after placing entry order before confirming position/price
MIN_TICKS_AWAY_FOR_SLTP = 3  # Minimum number of price ticks SL/TP should be away from entry price

# --- Bot Logic Constants ---
# Supported intervals for OHLCV data (ensure config uses one of these)
VALID_INTERVALS: list[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
# Map bot intervals to ccxt's expected timeframe format
CCXT_INTERVAL_MAP: dict[str, str] = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
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
FIB_LEVELS: list[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels

# Default loop delay (can be overridden by config)
DEFAULT_LOOP_DELAY_SECONDS = 15

# --- Global Variables ---
loggers: dict[str, logging.Logger] = {}  # Cache for logger instances
console_log_level: int = logging.INFO  # Default console log level (can be changed by args)
QUOTE_CURRENCY: str = "USDT"  # Default quote currency (updated from config)
LOOP_DELAY_SECONDS: int = DEFAULT_LOOP_DELAY_SECONDS  # Actual loop delay (updated from config)
IS_UNIFIED_ACCOUNT: bool = False  # Flag to indicate if the account is UTA (detected on init)


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive API keys/secrets."""

    REDACTED_STR: str = "***REDACTED***"
    SENSITIVE_KEYS: list[str] = [API_KEY or "UNUSED_KEY", API_SECRET or "UNUSED_SECRET"]

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

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Formats the record's creation time using the local timezone."""
        dt = datetime.fromtimestamp(record.created, tz=TIMEZONE)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            # Default format with milliseconds
            s = dt.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{s},{int(record.msecs):03d}"  # Add milliseconds
        return s


def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up a logger with rotating file (UTC) and timezone-aware console handlers.
    Caches logger instances to avoid duplicate handlers.

    Args:
        name: The name for the logger (e.g., 'main', 'BTC/USDT').
        is_symbol_logger: If True, formats the logger name for file system safety.

    Returns:
        The configured logging.Logger instance.
    """
    global console_log_level
    # Create a safe name for file system and logger registry
    logger_instance_name = (
        f"livebot_{name.replace('/', '_').replace(':', '-')}" if is_symbol_logger else f"livebot_{name}"
    )

    if logger_instance_name in loggers:
        logger = loggers[logger_instance_name]
        # Update console handler level if it changed (e.g., via --debug)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level != console_log_level:
                handler.setLevel(console_log_level)
        return logger

    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels at the logger itself

    # --- File Handler (UTC Timestamps) ---
    try:
        file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        # Use UTC time for file logs for consistency
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_formatter.converter = time.gmtime  # Use UTC time converter
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        logger.addHandler(file_handler)
    except Exception:
        pass

    # --- Stream Handler (Local Timezone Timestamps) ---
    try:
        stream_handler = logging.StreamHandler(sys.stdout)
        tz_name_str = TZ_NAME  # Use the determined timezone name
        # Include milliseconds in console output format
        console_fmt = f"{NEON_BLUE}%(asctime)s{RESET} [{tz_name_str}] - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
        stream_formatter = LocalTimeFormatter(console_fmt, datefmt="%Y-%m-%d %H:%M:%S,%f"[:-3])
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(console_log_level)  # Set console level based on global setting
        logger.addHandler(stream_handler)
    except Exception:
        pass

    logger.propagate = False  # Prevent log messages from propagating to the root logger
    loggers[logger_instance_name] = logger  # Cache the logger
    logger.info(
        f"Logger '{logger_instance_name}' initialized. File: '{os.path.basename(log_filename)}', Console Level: {logging.getLevelName(console_log_level)}"
    )
    return logger


def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance using setup_logger."""
    return setup_logger(name, is_symbol_logger)


# --- Configuration Management ---
def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from the default config are present in the loaded config.
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
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_updated_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_value)
            if nested_keys_added:
                updated_config[key] = nested_updated_config
                keys_added_or_type_mismatch = True
        elif updated_config.get(key) is not None and type(default_value) != type(updated_config.get(key)):
            # Type mismatch check (allow int -> float/Decimal promotion)
            is_promoting_num = isinstance(default_value, (float, Decimal)) and isinstance(updated_config.get(key), int)
            if not is_promoting_num:
                pass
                # Note: We keep the user's value despite the type mismatch, but warn them.
                # Optionally flag type mismatches too:
                # keys_added_or_type_mismatch = True
    return updated_config, keys_added_or_type_mismatch


def _validate_config_values(config: dict[str, Any], logger: logging.Logger) -> bool:
    """Validates specific critical configuration values for type, range, and format.

    Args:
        config: The configuration dictionary to validate. Modifies dictionary in place for type conversions.
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
    # Uses float for most parameters to allow flexibility in config file (e.g., "leverage": 10.0)
    # Converts to int where strictly necessary later.
    numeric_params: dict[str, tuple[type, int | float, int | float]] = {
        "loop_delay": (float, 5.0, 3600.0),  # Min 5 sec delay recommended
        "risk_per_trade": (float, 0.0001, 0.5),  # Risk 0.01% to 50% of balance
        "leverage": (float, 1.0, 125.0),  # Practical leverage limits (exchange may vary)
        "max_concurrent_positions_total": (float, 1.0, 100.0),
        "atr_period": (float, 2.0, 500.0),
        "ema_short_period": (float, 2.0, 500.0),
        "ema_long_period": (float, 3.0, 1000.0),  # Ensure long > short is not checked here, but logically required
        "rsi_period": (float, 2.0, 500.0),
        "bollinger_bands_period": (float, 5.0, 500.0),
        "bollinger_bands_std_dev": (float, 0.1, 5.0),
        "cci_window": (float, 5.0, 500.0),
        "williams_r_window": (float, 2.0, 500.0),
        "mfi_window": (float, 5.0, 500.0),
        "stoch_rsi_window": (float, 5.0, 500.0),
        "stoch_rsi_rsi_window": (float, 5.0, 500.0),  # Inner RSI window for StochRSI
        "stoch_rsi_k": (float, 1.0, 100.0),
        "stoch_rsi_d": (float, 1.0, 100.0),
        "psar_af": (float, 0.001, 0.5),
        "psar_max_af": (float, 0.01, 1.0),
        "sma_10_window": (float, 2.0, 500.0),
        "momentum_period": (float, 2.0, 500.0),
        "volume_ma_period": (float, 5.0, 500.0),
        "fibonacci_window": (float, 10.0, 1000.0),
        "orderbook_limit": (float, 1.0, 200.0),  # Bybit V5 limit might be 50 or 200 depending on type
        "signal_score_threshold": (float, 0.1, 10.0),
        "stoch_rsi_oversold_threshold": (float, 0.0, 50.0),
        "stoch_rsi_overbought_threshold": (float, 50.0, 100.0),
        "volume_confirmation_multiplier": (float, 0.1, 10.0),
        "scalping_signal_threshold": (float, 0.1, 10.0),
        "stop_loss_multiple": (float, 0.1, 10.0),  # Multiplier of ATR
        "take_profit_multiple": (float, 0.1, 20.0),  # Multiplier of ATR
        "trailing_stop_callback_rate": (float, 0.0001, 0.5),  # 0.01% to 50% (as distance from price)
        "trailing_stop_activation_percentage": (float, 0.0, 0.5),  # 0% to 50% (profit needed to activate)
        "break_even_trigger_atr_multiple": (float, 0.1, 10.0),  # Multiplier of ATR
        "break_even_offset_ticks": (float, 0.0, 100.0),  # Number of ticks above/below entry for BE SL
    }
    keys_requiring_int: list[str] = [  # Keys that MUST be integers for logic/API calls
        "leverage",
        "max_concurrent_positions_total",
        "atr_period",
        "ema_short_period",
        "ema_long_period",
        "rsi_period",
        "bollinger_bands_period",
        "cci_window",
        "williams_r_window",
        "mfi_window",
        "stoch_rsi_window",
        "stoch_rsi_rsi_window",
        "stoch_rsi_k",
        "stoch_rsi_d",
        "sma_10_window",
        "momentum_period",
        "volume_ma_period",
        "fibonacci_window",
        "orderbook_limit",
        "break_even_offset_ticks",
        "loop_delay",  # Loop delay should be int seconds
    ]

    for key, (_expected_type, min_val, max_val) in numeric_params.items():
        value = config.get(key)
        if value is None:
            continue  # Skip if optional or handled by ensure_keys

        try:
            # Attempt conversion to float first for range checking
            num_value_float = float(value)

            # Check range
            if not (min_val <= num_value_float <= max_val):
                logger.error(
                    f"Config Error: '{key}' value {num_value_float} is outside the recommended range ({min_val} - {max_val})."
                )
                is_valid = False

            # Store the validated numeric value back (as int if required, else float)
            if key in keys_requiring_int:
                if num_value_float != int(num_value_float):
                    logger.warning(
                        f"Config Warning: '{key}' requires an integer, but found {num_value_float}. Truncating to {int(num_value_float)}."
                    )
                config[key] = int(num_value_float)
            else:
                config[key] = num_value_float  # Store as float
        except (ValueError, TypeError):
            logger.error(f"Config Error: '{key}' value '{value}' could not be converted to a number.")
            is_valid = False

    # Specific check: EMA Long > EMA Short
    if config.get("ema_long_period", 0) <= config.get("ema_short_period", 0):
        logger.error(
            f"Config Error: 'ema_long_period' ({config.get('ema_long_period')}) must be greater than 'ema_short_period' ({config.get('ema_short_period')})."
        )
        is_valid = False

    # 3. Validate Symbols List
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        logger.error("Config Error: 'symbols' must be a non-empty list.")
        is_valid = False
    elif not all(isinstance(s, str) and "/" in s for s in symbols):  # Basic format check
        logger.error(
            f"Config Error: 'symbols' list contains invalid formats. Expected 'BASE/QUOTE' or 'BASE/QUOTE:SETTLE'. Found: {symbols}"
        )
        is_valid = False

    # 4. Validate Active Weight Set exists
    active_set = config.get("active_weight_set")
    weight_sets = config.get("weight_sets")
    if not isinstance(weight_sets, dict) or active_set not in weight_sets:
        logger.error(
            f"Config Error: 'active_weight_set' ('{active_set}') not found in 'weight_sets'. Available: {list(weight_sets.keys() if isinstance(weight_sets, dict) else [])}"
        )
        is_valid = False
    elif not isinstance(weight_sets[active_set], dict):
        logger.error(f"Config Error: Active weight set '{active_set}' must be a dictionary of weights.")
        is_valid = False
    else:
        # Validate weights within the active set (should be numeric)
        for indi_key, weight_val in weight_sets[active_set].items():
            try:
                float(weight_val)  # Check if convertible to float
            except (ValueError, TypeError):
                logger.error(
                    f"Config Error: Invalid weight value '{weight_val}' for indicator '{indi_key}' in weight set '{active_set}'. Must be numeric."
                )
                is_valid = False

    # 5. Validate Boolean types (ensure they are actually bool)
    bool_params = [
        "enable_trading",
        "use_sandbox",
        "enable_ma_cross_exit",
        "enable_trailing_stop",
        "enable_break_even",
        "break_even_force_fixed_sl",
    ]
    for key in bool_params:
        if key in config and not isinstance(config[key], bool):
            # Try to convert common string representations
            val_lower = str(config[key]).lower()
            if val_lower in ["true", "yes", "1"]:
                config[key] = True
            elif val_lower in ["false", "no", "0"]:
                config[key] = False
            else:
                logger.error(f"Config Error: '{key}' value '{config[key]}' must be a boolean (true/false).")
                is_valid = False

    # Validate indicator enable flags
    if "indicators" in config and isinstance(config["indicators"], dict):
        for indi_key, indi_val in config["indicators"].items():
            if not isinstance(indi_val, bool):
                # Try conversion
                val_lower = str(indi_val).lower()
                if val_lower in ["true", "yes", "1"]:
                    config["indicators"][indi_key] = True
                elif val_lower in ["false", "no", "0"]:
                    config["indicators"][indi_key] = False
                else:
                    logger.error(
                        f"Config Error: Indicator enable flag 'indicators.{indi_key}' value '{indi_val}' must be a boolean (true/false)."
                    )
                    is_valid = False
    elif "indicators" in config:
        logger.error("Config Error: 'indicators' must be a dictionary of boolean flags.")
        is_valid = False

    return is_valid


def load_config(filepath: str, logger: logging.Logger) -> dict[str, Any] | None:
    """Loads configuration from a JSON file.
    - Creates a default config if the file doesn't exist.
    - Ensures all necessary keys are present, adding defaults for missing ones.
    - Validates critical configuration values.
    - Saves the updated config (with defaults added) back to the file if changes were made.

    Args:
        filepath: The path to the configuration JSON file.
        logger: The logger instance for reporting.

    Returns:
        The loaded and validated configuration dictionary, or None if validation fails.
    """
    # Define the structure and default values for the configuration
    default_config: dict[str, Any] = {
        "symbols": ["BTC/USDT:USDT"],  # List of symbols to trade (ensure format matches CCXT)
        "interval": "5",  # Kline interval (e.g., "1", "5", "15", "60", "D")
        "loop_delay": DEFAULT_LOOP_DELAY_SECONDS,  # Seconds between bot cycles
        "quote_currency": "USDT",  # Primary currency for balance checks and calculations
        "enable_trading": False,  # Master switch for placing actual trades
        "use_sandbox": True,  # Use Bybit's testnet environment
        "risk_per_trade": 0.01,  # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 10,  # Leverage to use for futures trades
        "max_concurrent_positions_total": 1,  # Maximum number of open positions across all symbols
        "position_mode": "One-Way",  # "One-Way" or "Hedge" (Hedge mode requires careful implementation)
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
        "psar_af": DEFAULT_PSAR_AF,  # Parabolic SAR acceleration factor step
        "psar_max_af": DEFAULT_PSAR_MAX_AF,  # Parabolic SAR max acceleration factor
        "sma_10_window": DEFAULT_SMA_10_WINDOW,  # Example additional indicator period
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW,  # Window for calculating Fib levels
        "orderbook_limit": 25,  # Number of levels to fetch for order book analysis
        "signal_score_threshold": 1.5,  # Threshold for combined weighted score to trigger BUY/SELL
        "stoch_rsi_oversold_threshold": 25.0,
        "stoch_rsi_overbought_threshold": 75.0,
        "volume_confirmation_multiplier": 1.5,  # How much current vol must exceed MA vol
        "scalping_signal_threshold": 2.5,  # Optional higher threshold for 'scalping' weight set
        "stop_loss_multiple": 1.8,  # ATR multiple for initial Stop Loss distance
        "take_profit_multiple": 0.7,  # ATR multiple for initial Take Profit distance
        "enable_ma_cross_exit": True,  # Close position if short/long EMAs cross adversely
        "enable_trailing_stop": True,  # Enable Trailing Stop Loss feature
        "trailing_stop_callback_rate": 0.005,  # TSL distance as fraction of price (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003,  # Profit % required to activate TSL (e.g., 0.003 = 0.3%)
        "enable_break_even": True,  # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0,  # ATR multiple profit needed to trigger BE
        "break_even_offset_ticks": 2,  # How many ticks *above* entry (for longs) or *below* (for shorts) to set BE SL
        "break_even_force_fixed_sl": True,  # If true, BE replaces TSL; if false, BE sets SL but TSL might remain active if configured
        # --- Indicator Enable Flags ---
        "indicators": {
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "stoch_rsi": True,
            "rsi": True,
            "bollinger_bands": True,
            "vwap": True,
            "cci": True,
            "wr": True,
            "psar": True,
            "sma_10": True,
            "mfi": True,
            "orderbook": True,  # Requires fetching order book data
        },
        # --- Weight Sets for Signal Generation ---
        "weight_sets": {
            # Example: Scalping focuses more on faster indicators like Momentum, StochRSI
            "scalping": {
                "ema_alignment": 0.2,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "stoch_rsi": 0.6,
                "rsi": 0.2,
                "bollinger_bands": 0.3,
                "vwap": 0.4,
                "cci": 0.3,
                "wr": 0.3,
                "psar": 0.2,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.15,
            },
            # Example: Default might be more balanced
            "default": {
                "ema_alignment": 0.3,
                "momentum": 0.2,
                "volume_confirmation": 0.1,
                "stoch_rsi": 0.4,
                "rsi": 0.3,
                "bollinger_bands": 0.2,
                "vwap": 0.3,
                "cci": 0.2,
                "wr": 0.2,
                "psar": 0.3,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.1,
            },
        },
        "active_weight_set": "default",  # Which set of weights to use for signal scoring
    }

    config_to_use = default_config  # Start with defaults

    if not os.path.exists(filepath):
        # Config file doesn't exist, create it with defaults
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
        except OSError:
            pass
            # Continue with default_config in memory
    else:
        # Config file exists, load it
        try:
            with open(filepath, encoding="utf-8") as f:
                config_from_file = json.load(f)

            # Ensure all keys are present, adding defaults if missing
            updated_config_from_file, keys_added = _ensure_config_keys(config_from_file, default_config)
            config_to_use = updated_config_from_file

            # If keys were added, save the updated config back to the file
            if keys_added:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(config_to_use, f_write, indent=4, sort_keys=True)
                except OSError:
                    pass  # Failed to save

        except (FileNotFoundError, json.JSONDecodeError):
            # Handle file reading or JSON parsing errors
            config_to_use = default_config  # Fallback to defaults
            try:
                # Try to recreate the file with defaults after a load error
                with open(filepath, "w", encoding="utf-8") as f_recreate:
                    json.dump(default_config, f_recreate, indent=4, sort_keys=True)
            except OSError:
                pass
        except Exception:
            # Catch any other unexpected errors during loading
            config_to_use = default_config  # Fallback to defaults

    # --- Final Validation ---
    # Validate the configuration values (whether loaded or default)
    # _validate_config_values modifies config_to_use in place
    if not _validate_config_values(config_to_use, logger):
        logger.critical("Configuration validation failed. Please check errors above and fix config.json. Exiting.")
        return None  # Indicate failure

    logger.info("Configuration loaded and validated successfully.")
    return config_to_use


# --- State Management ---
def load_state(filepath: str, logger: logging.Logger) -> dict[str, Any]:
    """Loads the bot's operational state from a JSON file.
    Handles file not found and JSON decoding errors gracefully.

    Args:
        filepath: Path to the state file.
        logger: Logger instance.

    Returns:
        A dictionary containing the loaded state, or an empty dictionary if loading fails.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, encoding="utf-8") as f:
                state = json.load(f)
                logger.info(f"Loaded previous state from {filepath}")
                # Basic validation: Ensure it's a dictionary
                if not isinstance(state, dict):
                    logger.error(f"State file {filepath} does not contain a valid dictionary. Starting fresh.")
                    return {}
                return state
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading state file {filepath}: {e}. Starting with empty state.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}. Starting with empty state.", exc_info=True)
            return {}
    else:
        logger.info("No previous state file found. Starting with empty state.")
        return {}


def save_state(filepath: str, state: dict[str, Any], logger: logging.Logger) -> None:
    """Saves the bot's current operational state to a JSON file using an atomic write.
    Ensures Decimals are converted to strings for JSON compatibility.

    Args:
        filepath: Path to the state file.
        state: The dictionary containing the current state to save.
        logger: Logger instance.
    """
    temp_filepath = filepath + ".tmp"
    try:
        # Convert Decimals to strings for JSON serialization
        # Use json.dumps with default=str, then load back to ensure pure JSON types
        state_to_save = json.loads(json.dumps(state, default=str))

        # Write to temporary file first
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=4, sort_keys=True)

        # Atomic rename/replace (os.replace is atomic on most modern systems)
        os.replace(temp_filepath, filepath)
        logger.debug(f"Saved current state to {filepath}")

    except (OSError, TypeError, json.JSONDecodeError) as e:
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
def initialize_exchange(config: dict[str, Any], logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object.
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
    global QUOTE_CURRENCY, IS_UNIFIED_ACCOUNT  # Allow modification of globals

    try:
        QUOTE_CURRENCY = config.get("quote_currency", "USDT")
        lg.info(f"Using Quote Currency: {QUOTE_CURRENCY}")

        # CCXT Exchange Options for Bybit V5
        exchange_options = {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "rateLimit": 120,  # Default CCXT rate limit (adjust if needed)
            "options": {
                "defaultType": "linear",  # Default to linear for futures/swaps
                "adjustForTimeDifference": True,
                "recvWindow": 10000,
                # Timeouts (milliseconds)
                "fetchTickerTimeout": 15000,
                "fetchBalanceTimeout": 20000,
                "createOrderTimeout": 25000,
                "fetchOrderTimeout": 20000,
                "fetchPositionsTimeout": 20000,
                "cancelOrderTimeout": 20000,
                "fetchOHLCVTimeout": 20000,
                "setLeverageTimeout": 20000,
                "fetchMarketsTimeout": 30000,
                "brokerId": f"EnhancedWhale71_{BOT_VERSION[:3]}",  # Example: EnhancedWhale71_1.0
                # Explicit V5 mapping (CCXT usually handles this, but can be explicit)
                "versions": {
                    "public": {"GET": {"market/tickers": "v5", "market/kline": "v5", "market/orderbook": "v5"}},
                    "private": {
                        "GET": {
                            "position/list": "v5",
                            "account/wallet-balance": "v5",
                            "order/realtime": "v5",
                            "order/history": "v5",
                        },
                        "POST": {
                            "order/create": "v5",
                            "order/cancel": "v5",
                            "position/set-leverage": "v5",
                            "position/trading-stop": "v5",
                        },
                    },
                },
                # Default options hinting CCXT to prefer V5 methods
                "default_options": {
                    "fetchPositions": "v5",
                    "fetchBalance": "v5",
                    "createOrder": "v5",
                    "fetchOrder": "v5",
                    "fetchTicker": "v5",
                    "fetchOHLCV": "v5",
                    "fetchOrderBook": "v5",
                    "setLeverage": "v5",
                    "private_post_position_trading_stop": "v5",  # Explicit map for protection endpoint
                },
                # Map CCXT account types to Bybit API account types
                "accountsByType": {
                    "spot": "SPOT",
                    "future": "CONTRACT",
                    "swap": "CONTRACT",
                    "margin": "UNIFIED",
                    "option": "OPTION",
                    "unified": "UNIFIED",
                    "contract": "CONTRACT",
                },
                "accountsById": {  # Reverse mapping (useful internally for CCXT)
                    "SPOT": "spot",
                    "CONTRACT": "contract",
                    "UNIFIED": "unified",
                    "OPTION": "option",
                },
                "bybit": {"defaultSettleCoin": QUOTE_CURRENCY},
            },
        }

        exchange_id = "bybit"
        exchange_class = getattr(ccxt, exchange_id)
        exchange: ccxt.Exchange = exchange_class(exchange_options)

        # --- Sandbox/Live Mode Configuration ---
        if config.get("use_sandbox", True):
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
            lg.error(
                f"{NEON_RED}Fatal Error loading markets: {e}. Check network connection and API endpoint status.{RESET}",
                exc_info=True,
            )
            return None  # Cannot proceed without markets

        # --- Test API Credentials & Check Account Type ---
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
                lg.warning(
                    f"{NEON_YELLOW}Could not definitively determine account type during initial balance check. Proceeding with caution.{RESET}"
                )

            # Check if balance was successfully fetched
            if balance_decimal is not None and balance_decimal > 0:
                lg.info(
                    f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})"
                )
            elif balance_decimal is not None:  # Balance is zero
                lg.warning(
                    f"{NEON_YELLOW}Successfully connected, but initial {QUOTE_CURRENCY} balance is zero.{RESET} (Account Type: {account_type_detected or 'Unknown'})"
                )
            else:  # Balance fetch failed
                lg.warning(
                    f"{NEON_YELLOW}Initial balance fetch failed (Account Type: {account_type_detected or 'Unknown'}). Check logs above. Ensure API keys have 'Read' permissions for the correct account type (Unified/Contract/Spot).{RESET}"
                )
                if config.get("enable_trading"):
                    lg.error(
                        f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}"
                    )
                    return None
                else:
                    lg.warning("Continuing in non-trading mode despite balance fetch issue.")

        except ccxt.AuthenticationError as auth_err:
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial setup: {auth_err}{RESET}")
            lg.error(
                f"{NEON_RED}>> Check API Key, API Secret, Permissions (Read/Trade), Account Type (Real/Testnet), and IP Whitelist.{RESET}"
            )
            return None  # Fatal error
        except Exception as balance_err:
            lg.error(f"{NEON_RED}Unexpected error during initial balance check: {balance_err}{RESET}", exc_info=True)
            if config.get("enable_trading"):
                lg.error(
                    f"{NEON_RED}Aborting initialization due to unexpected balance fetch error in trading mode.{RESET}"
                )
                return None
            else:
                lg.warning(
                    f"{NEON_YELLOW}Continuing in non-trading mode despite unexpected balance fetch error: {balance_err}{RESET}"
                )

        # If all checks passed (or warnings were accepted in non-trading mode)
        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during exchange initialization: {e}{RESET}", exc_info=True)
        return None


def _check_account_type_and_balance(
    exchange: ccxt.Exchange, currency: str, logger: logging.Logger
) -> tuple[bool | None, Decimal | None]:
    """Tries fetching balance using both UNIFIED and CONTRACT/SPOT account types
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
    unified_balance: Decimal | None = None
    contract_spot_balance: Decimal | None = None
    is_unified: bool | None = None

    # --- Attempt 1: Try fetching as UNIFIED account ---
    try:
        lg.debug("Checking balance with accountType=UNIFIED...")
        params_unified = {"accountType": "UNIFIED", "coin": currency}
        # Use safe_ccxt_call but with fewer retries for detection purposes
        bal_info_unified = safe_ccxt_call(
            exchange, "fetch_balance", lg, max_retries=1, retry_delay=2, params=params_unified
        )
        parsed_balance = _parse_balance_response(bal_info_unified, currency, "UNIFIED", lg)
        if parsed_balance is not None:
            lg.info("Successfully fetched balance using UNIFIED account type.")
            unified_balance = parsed_balance
            is_unified = True
            # Return immediately if UNIFIED balance is found
            return is_unified, unified_balance
    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        # Bybit error codes/messages indicating wrong account type for the API key
        # 30086: UTA not supported / Account Type mismatch
        # 10001 + "accounttype": Parameter error related to account type
        if (
            "accounttype only support" in error_str
            or "30086" in error_str
            or "unified account is not supported" in error_str
            or ("10001" in error_str and "accounttype" in error_str)
        ):
            lg.debug("Fetching with UNIFIED failed (as expected for non-UTA), trying CONTRACT/SPOT...")
            is_unified = False  # Assume Non-UTA if this specific error occurs
        else:
            lg.warning(f"ExchangeError checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")
    except Exception as e:
        lg.warning(f"Unexpected error checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")

    # --- Attempt 2: Try fetching as CONTRACT/SPOT account (if UNIFIED failed or gave specific error) ---
    # Only proceed if unified check failed or suggested non-UTA
    if is_unified is False or is_unified is None:
        account_types_to_try = ["CONTRACT", "SPOT"]
        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Checking balance with accountType={acc_type}...")
                params = {"accountType": acc_type, "coin": currency}
                bal_info = safe_ccxt_call(exchange, "fetch_balance", lg, max_retries=1, retry_delay=2, params=params)
                parsed_balance = _parse_balance_response(bal_info, currency, acc_type, lg)
                if parsed_balance is not None:
                    lg.info(f"Successfully fetched balance using {acc_type} account type.")
                    contract_spot_balance = parsed_balance
                    is_unified = False  # Confirmed Non-UTA
                    return is_unified, contract_spot_balance  # Return found Non-UTA balance
            except ccxt.ExchangeError as e:
                lg.warning(f"ExchangeError checking {acc_type} balance: {e}.")
            except Exception as e:
                lg.warning(f"Unexpected error checking {acc_type} balance: {e}.")
                break  # Stop checking other types if unexpected error occurs

    # --- Conclusion ---
    if is_unified is True and unified_balance is not None:
        return True, unified_balance
    if is_unified is False and contract_spot_balance is not None:
        return False, contract_spot_balance

    # If all attempts failed to get a definitive balance
    lg.error("Failed to determine account type OR fetch balance with common types (UNIFIED/CONTRACT/SPOT).")
    return None, None  # Unknown type, no balance retrieved


# --- CCXT API Call Helper with Retries ---
def safe_ccxt_call(
    exchange: ccxt.Exchange,
    method_name: str,
    logger: logging.Logger,
    max_retries: int = MAX_API_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Safely calls a CCXT exchange method with robust retry logic.
    Handles RateLimitExceeded, NetworkError, RequestTimeout, DDoSProtection,
    and specific non-retryable ExchangeErrors.

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
        Exception: If an unexpected error occurs or max retries are exceeded.
    """
    lg = logger
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            method = getattr(exchange, method_name)
            result = method(*args, **kwargs)
            return result

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * (2**attempt)  # Exponential backoff base
            suggested_wait: float | None = None
            try:
                # Attempt to parse suggested wait time from Bybit error message
                import re

                error_msg = str(e).lower()
                match_ms = re.search(r"(?:try again in|retry after)\s*(\d+)\s*ms", error_msg)
                match_s = re.search(r"(?:try again in|retry after)\s*(\d+)\s*s", error_msg)
                if match_ms:
                    suggested_wait = max(1.0, math.ceil(int(match_ms.group(1)) / 1000) + RATE_LIMIT_BUFFER_SECONDS)
                elif match_s:
                    suggested_wait = max(1.0, int(match_s.group(1)) + RATE_LIMIT_BUFFER_SECONDS)
                elif "too many visits" in error_msg or "limit" in error_msg:
                    # Fallback if specific time isn't mentioned but it's clearly a rate limit
                    suggested_wait = wait_time + RATE_LIMIT_BUFFER_SECONDS
            except Exception:
                pass  # Ignore parsing errors

            final_wait = suggested_wait if suggested_wait is not None else wait_time
            lg.warning(
                f"Rate limit hit calling {method_name}. Retrying in {final_wait:.2f}s... (Attempt {attempt + 1}/{max_retries + 1}) Error: {e}"
            )
            time.sleep(final_wait)

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = retry_delay * (2**attempt)
            lg.warning(
                f"Network/DDoS/Timeout error calling {method_name}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries + 1})"
            )
            time.sleep(wait_time)

        except ccxt.AuthenticationError as e:
            lg.error(
                f"{NEON_RED}Authentication Error calling {method_name}: {e}. Check API keys/permissions. Not retrying.{RESET}"
            )
            raise e

        except ccxt.ExchangeError as e:
            last_exception = e
            bybit_code: int | None = None
            ret_msg: str = str(e)
            try:
                # Try to extract Bybit's retCode and retMsg from the error string
                if hasattr(e, "args") and len(e.args) > 0:
                    error_details = str(e.args[0])
                    # Find JSON part within the error string (often contains retCode/retMsg)
                    json_start = error_details.find("{")
                    json_end = error_details.rfind("}") + 1
                    if json_start != -1 and json_end != -1:
                        details_dict = json.loads(error_details[json_start:json_end])
                        bybit_code = details_dict.get("retCode")
                        ret_msg = details_dict.get("retMsg", str(e))
            except (json.JSONDecodeError, IndexError, TypeError):
                pass  # Failed to parse

            # Define known non-retryable Bybit error codes (V5)
            # See: https://bybit-exchange.github.io/docs/v5/error_code
            non_retryable_codes: list[int] = [
                10001,  # Parameter error (check args/kwargs/permissions)
                10002,  # Request not supported
                10003,  # Invalid API key or IP whitelist
                10004,  # Invalid sign / Authentication failed
                10005,  # Permissions denied
                10006,  # Rate limit exceeded (should be handled by RateLimitExceeded, but catch here too)
                10009,  # IP banned
                10010,  # Request timeout
                10016,  # Service error / Maintenance
                10017,  # Request path not found
                10018,  # Exceeded frequency limit
                10020,  # Websocket issue (less relevant here)
                10029,  # Request parameter validation error
                110001,  # Order placement failed (generic)
                110003,  # Invalid price
                110004,  # Invalid quantity
                110005,  # Qty too small
                110006,  # Qty too large
                110007,  # Insufficient balance
                110008,  # Cost too small
                110009,  # Cost too large
                110010,  # Invalid order type
                110011,  # Invalid side
                110012,  # Invalid timeInForce
                110013,  # Price exceeds deviation limits
                110014,  # OrderId not found or invalid
                110015,  # Order already cancelled
                110016,  # Order already filled
                110017,  # Price/Qty precision error
                110019,  # Cannot amend market orders
                110020,  # Position status prohibits action
                110021,  # Risk limit exceeded
                110022,  # Invalid leverage
                110024,  # Position not found
                110025,  # Position idx error (Hedge Mode)
                110028,  # Reduce-only order would increase position
                110031,  # Order amount exceeds open limit
                110033,  # Cannot set leverage in cross margin mode
                110036,  # Cross/Isolated mode mismatch
                110040,  # TP/SL order parameter error
                110041,  # TP/SL requires position
                110042,  # TP/SL price invalid
                110043,  # Set leverage not modified (special case)
                110044,  # Margin mode not modified
                110045,  # Qty exceeds risk limit
                110047,  # Cannot set TP/SL for Market orders during creation
                110051,  # Position zero, cannot close
                110067,  # Feature requires UTA Pro
                170001,  # Internal error affecting position
                170007,  # Risk limit exceeded
                170019,  # Margin call / Liquidation status
                170131,  # TP/SL price invalid
                170132,  # TP/SL order triggers liquidation
                170133,  # Cannot set TP/SL/TSL
                170140,  # TP/SL requires position
                30086,  # UTA not supported / Account Type mismatch
                30087,  # UTA feature unavailable
                # Add more critical, non-recoverable codes as identified
            ]

            if bybit_code in non_retryable_codes:
                # Special handling for "Leverage not modified" - treat as success
                if bybit_code == 110043 and method_name == "set_leverage":
                    lg.info(
                        f"Leverage already set as requested (Code 110043) when calling {method_name}. Ignoring error."
                    )
                    return {}  # Return empty dict, often treated as success by calling code

                extra_info = ""
                if bybit_code == 10001:
                    extra_info = f"{NEON_YELLOW} Hint: Check API call parameters ({args=}, {kwargs=}) or API key permissions/account type.{RESET}"
                elif bybit_code == 110007:
                    balance_currency = QUOTE_CURRENCY
                    if "params" in kwargs and "coin" in kwargs["params"]:
                        balance_currency = kwargs["params"]["coin"]
                    elif "symbol" in kwargs:
                        balance_currency = kwargs["symbol"].split("/")[1].split(":")[0]  # Guess quote
                    extra_info = f"{NEON_YELLOW} Hint: Check available {balance_currency} balance in the correct account (UTA/Contract/Spot).{RESET}"
                elif bybit_code == 30086 or (bybit_code == 10001 and "accounttype" in ret_msg.lower()):
                    extra_info = f"{NEON_YELLOW} Hint: Check 'accountType' param (UNIFIED vs CONTRACT/SPOT) matches your account/API key permissions.{RESET}"
                elif bybit_code == 110025:
                    extra_info = f"{NEON_YELLOW} Hint: Check 'positionIdx' parameter (0 for One-Way, 1/2 for Hedge Mode) and ensure it matches your account's Position Mode setting.{RESET}"
                elif bybit_code == 110017 or bybit_code == 110005 or bybit_code == 110006:
                    extra_info = (
                        f"{NEON_YELLOW} Hint: Check price/amount precision and limits against market data.{RESET}"
                    )

                lg.error(
                    f"{NEON_RED}Non-retryable Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Not retrying.{RESET}{extra_info}"
                )
                raise e  # Re-raise the non-retryable error

            else:
                # Unknown or potentially temporary exchange error, retry
                lg.warning(
                    f"{NEON_YELLOW}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Retrying... (Attempt {attempt + 1}/{max_retries + 1}){RESET}"
                )
                wait_time = retry_delay * (2**attempt)
                time.sleep(wait_time)

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected Error calling {method_name}: {e}. Not retrying.{RESET}", exc_info=True)
            raise e

    # Max Retries Reached
    lg.error(f"{NEON_RED}Max retries ({max_retries}) reached for {method_name}. Last error: {last_exception}{RESET}")
    raise (
        last_exception
        if last_exception
        else RuntimeError(f"Max retries reached for {method_name} without specific exception.")
    )


# --- Market Info Helper Functions ---
def _determine_category(market: dict[str, Any]) -> str | None:
    """Determines the Bybit V5 category ('linear', 'inverse', 'spot', 'option') from CCXT market info."""
    market_type = market.get("type")
    is_linear = market.get("linear", False)
    is_inverse = market.get("inverse", False)
    is_spot = market.get("spot", False)
    is_option = market.get("option", False)

    if is_spot:
        return "spot"
    if is_option:
        return "option"
    if is_linear and market_type in ["swap", "future"]:
        return "linear"
    if is_inverse and market_type in ["swap", "future"]:
        return "inverse"

    # Fallback checks if primary flags are missing
    if market_type == "spot":
        return "spot"
    if market_type == "option":
        return "option"
    if market_type in ["swap", "future"]:
        settle_asset = market.get("settle", "").upper()
        quote_asset = market.get("quote", "").upper()
        if settle_asset == quote_asset:
            return "linear"  # Settles in quote (USDT, USDC)
        else:
            return "inverse"  # Settles in base (BTC, ETH)

    return None  # Unable to determine


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict[str, Any] | None:
    """Retrieves and processes market information for a symbol from loaded CCXT markets.
    Extracts key details like precision, limits, contract size, and category.

    Args:
        exchange: Initialized CCXT exchange object with loaded markets.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing processed market info, or None if symbol not found/invalid.
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
            lg.warning(f"Could not determine V5 category for symbol {symbol}. Market data: {market}")

        # Extract precision details safely
        price_precision_val = market.get("precision", {}).get("price")
        amount_precision_val = market.get("precision", {}).get("amount")

        # Calculate precision digits from precision value (e.g., 0.01 -> 2 digits)
        # Use Decimal context for log10 calculation for precision
        price_digits = 8  # Default
        if price_precision_val is not None and price_precision_val > 0:
            try:
                price_digits = int(Decimal(str(price_precision_val)).log10().copy_negate())
            except Exception:
                lg.warning(f"Could not calculate price digits from precision {price_precision_val}")
        amount_digits = 8  # Default
        if amount_precision_val is not None and amount_precision_val > 0:
            try:
                amount_digits = int(Decimal(str(amount_precision_val)).log10().copy_negate())
            except Exception:
                lg.warning(f"Could not calculate amount digits from precision {amount_precision_val}")

        # Extract limits safely, converting to Decimal where appropriate
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        price_limits = limits.get("price", {})
        cost_limits = limits.get("cost", {})

        def to_decimal_safe(val: Any) -> Decimal | None:
            if val is None:
                return None
            try:
                return Decimal(str(val))
            except InvalidOperation:
                return None

        min_amount = to_decimal_safe(amount_limits.get("min"))
        max_amount = to_decimal_safe(amount_limits.get("max"))
        min_price = to_decimal_safe(price_limits.get("min"))
        max_price = to_decimal_safe(price_limits.get("max"))
        min_cost = to_decimal_safe(cost_limits.get("min"))
        max_cost = to_decimal_safe(cost_limits.get("max"))

        # Contract-specific details
        contract_size = to_decimal_safe(market.get("contractSize", "1")) or Decimal("1")  # Default to 1
        is_contract = category in ["linear", "inverse"]
        is_inverse = category == "inverse"

        market_details = {
            "symbol": symbol,
            "id": market.get("id"),  # Exchange's internal market ID
            "base": market.get("base"),
            "quote": market.get("quote"),
            "settle": market.get("settle"),
            "type": market.get("type"),  # spot, swap, future
            "category": category,  # linear, inverse, spot, option
            "is_contract": is_contract,
            "inverse": is_inverse,
            "contract_size": contract_size,
            "min_tick_size": to_decimal_safe(price_precision_val),  # Store tick size as Decimal
            "price_precision_digits": price_digits,
            "amount_precision_digits": amount_digits,
            "min_order_amount": min_amount,
            "max_order_amount": max_amount,
            "min_price": min_price,
            "max_price": max_price,
            "min_order_cost": min_cost,
            "max_order_cost": max_cost,
            "raw_market_data": market,  # Include original market data for debugging if needed
        }
        lg.debug(
            f"Market Info for {symbol}: Cat={category}, Tick={market_details['min_tick_size']}, AmtPrec={amount_digits}, ContSize={contract_size}"
        )
        return market_details

    except ccxt.BadSymbol as e:
        lg.error(f"Error getting market info for '{symbol}': {e}")
        return None
    except Exception as e:
        lg.error(f"Unexpected error processing market info for {symbol}: {e}", exc_info=True)
        return None


# --- Data Fetching Wrappers ---
def fetch_current_price_ccxt(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict
) -> Decimal | None:
    """Fetches the current ticker price using V5 API via safe_ccxt_call."""
    lg = logger
    category = market_info.get("category")
    market_id = market_info.get("id", symbol)  # Use exchange-specific ID
    if not category:
        lg.error(f"Cannot fetch price for {symbol}: Category unknown.")
        return None

    try:
        params = {"category": category, "symbol": market_id}
        lg.debug(f"Fetching ticker for {symbol} with params: {params}")
        ticker = safe_ccxt_call(
            exchange, "fetch_ticker", lg, symbol=symbol, params=params
        )  # Pass symbol for CCXT standard, params for V5

        # Bybit V5 ticker response structure might be nested under 'result' or 'list'
        ticker_data = None
        price_str = None
        if isinstance(ticker, dict):
            # Check V5 structure (result -> list -> [item])
            if (
                "info" in ticker
                and isinstance(ticker["info"], dict)
                and "result" in ticker["info"]
                and isinstance(ticker["info"]["result"], dict)
                and "list" in ticker["info"]["result"]
                and isinstance(ticker["info"]["result"]["list"], list)
                and len(ticker["info"]["result"]["list"]) > 0
            ):
                ticker_data = ticker["info"]["result"]["list"][0]
                price_str = ticker_data.get("lastPrice")  # V5 field
                lg.debug(f"Parsed price from V5 list structure for {symbol}")
            # Check V5 structure (result -> item) - Less common?
            elif (
                "info" in ticker
                and isinstance(ticker["info"], dict)
                and "result" in ticker["info"]
                and isinstance(ticker["info"]["result"], dict)
            ):
                ticker_data = ticker["info"]["result"]
                price_str = ticker_data.get("lastPrice")  # V5 field
                lg.debug(f"Parsed price from V5 result structure for {symbol}")
            # Fallback to standard CCXT 'last' field
            elif "last" in ticker:
                price_str = str(ticker["last"])
                lg.debug(f"Parsed price from standard CCXT 'last' field for {symbol}")

        if price_str is None:
            lg.warning(f"Could not extract 'lastPrice' or 'last' for {symbol}. Ticker: {ticker}")
            return None

        price_dec = Decimal(price_str)
        if price_dec.is_finite() and price_dec > 0:
            # lg.debug(f"Current price for {symbol}: {price_dec}")
            return price_dec
        else:
            lg.error(f"Invalid price ('{price_str}') received for {symbol}.")
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Error converting fetched price '{price_str}' to Decimal for {symbol}: {e}")
        return None
    except Exception as e:
        lg.error(f"Error fetching current price for {symbol}: {e}", exc_info=True)
        return None


def fetch_klines_ccxt(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger, market_info: dict
) -> pd.DataFrame:
    """Fetches OHLCV data using V5 API via safe_ccxt_call.
    Returns a DataFrame with UTC timestamp index and Decimal OHLCV columns.
    """
    lg = logger
    category = market_info.get("category")
    market_id = market_info.get("id", symbol)  # Use exchange-specific ID
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)

    if not category:
        lg.error(f"Cannot fetch klines for {symbol}: Category unknown.")
        return pd.DataFrame()
    if not ccxt_timeframe:
        lg.error(f"Invalid timeframe '{timeframe}' provided for {symbol}.")
        return pd.DataFrame()

    try:
        params = {"category": category, "symbol": market_id}
        lg.debug(f"Fetching {limit} klines for {symbol} ({ccxt_timeframe}) with params: {params}")
        # Limit might need adjustment based on exchange max (e.g., 1000 for Bybit V5 kline)
        safe_limit = min(limit, 1000)
        if limit > 1000:
            lg.warning(f"Requested kline limit {limit} > max 1000. Fetching {safe_limit}.")

        ohlcv = safe_ccxt_call(
            exchange, "fetch_ohlcv", lg, symbol=symbol, timeframe=ccxt_timeframe, limit=safe_limit, params=params
        )

        if not ohlcv:
            lg.warning(f"fetch_ohlcv returned empty data for {symbol}.")
            return pd.DataFrame()

        # Convert to DataFrame and set column names
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            lg.warning(f"Kline data for {symbol} resulted in an empty DataFrame.")
            return df

        # Convert timestamp to datetime (UTC) and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert OHLCV columns to Decimal for precision
        for col in ["open", "high", "low", "close", "volume"]:
            try:
                # Convert to string first to avoid potential float inaccuracies
                df[col] = df[col].astype(str).apply(Decimal)
            except (InvalidOperation, TypeError, ValueError) as e:
                lg.error(f"Error converting column '{col}' to Decimal for {symbol}: {e}. Coercing errors to NaN.")
                # Coerce errors during conversion to NaN, then potentially handle/drop NaNs later
                df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric first
                # Apply Decimal conversion only to non-NaN values
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else pd.NA)

        # Optional: Drop rows with any NaN/NA in critical columns if needed
        initial_len = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            lg.warning(f"Dropped {initial_len - len(df)} rows with NaN values in OHLC columns for {symbol}.")

        lg.debug(f"Fetched and processed {len(df)} klines for {symbol}.")
        return df

    except Exception as e:
        lg.error(f"Error fetching/processing klines for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()  # Return empty DataFrame on error


def fetch_orderbook_ccxt(
    exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: dict
) -> dict | None:
    """Fetches order book data using V5 API via safe_ccxt_call."""
    lg = logger
    category = market_info.get("category")
    market_id = market_info.get("id", symbol)  # Use exchange-specific ID
    if not category:
        lg.error(f"Cannot fetch orderbook for {symbol}: Category unknown.")
        return None

    try:
        params = {"category": category, "symbol": market_id, "limit": limit}  # Pass limit in params for V5
        lg.debug(f"Fetching order book (limit {limit}) for {symbol} with params: {params}")
        # CCXT fetch_order_book might not need limit in top level args if in params for V5
        orderbook = safe_ccxt_call(exchange, "fetch_order_book", lg, symbol=symbol, limit=limit, params=params)

        if orderbook and "bids" in orderbook and "asks" in orderbook:
            # Optional: Convert prices/amounts to Decimal here if needed downstream
            # Example conversion (can be slow for large orderbooks):
            # try:
            #     orderbook['bids'] = [[Decimal(str(p)), Decimal(str(a))] for p, a in orderbook['bids']]
            #     orderbook['asks'] = [[Decimal(str(p)), Decimal(str(a))] for p, a in orderbook['asks']]
            # except (InvalidOperation, TypeError, ValueError) as e:
            #     lg.error(f"Error converting orderbook values to Decimal for {symbol}: {e}")
            #     return None # Failed conversion
            lg.debug(f"Fetched order book for {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
            return orderbook
        else:
            lg.warning(f"Failed to fetch valid order book for {symbol}. Response: {orderbook}")
            return None
    except Exception as e:
        lg.error(f"Error fetching order book for {symbol}: {e}", exc_info=True)
        return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency.
    Adapts the request based on the detected account type (UTA vs. Non-UTA).

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency symbol (e.g., 'USDT').
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetch fails or balance is zero/invalid.
    """
    lg = logger
    account_types_to_try: list[str] = []

    if IS_UNIFIED_ACCOUNT:
        account_types_to_try = ["UNIFIED"]
        lg.debug(f"Fetching balance specifically for UNIFIED account ({currency}).")
    else:
        # For Non-UTA, CONTRACT usually holds futures balance, SPOT for spot balance.
        account_types_to_try = ["CONTRACT", "SPOT"]
        lg.debug(f"Fetching balance for Non-UTA account ({currency}), trying types: {account_types_to_try}.")

    last_exception: Exception | None = None
    parsed_balance: Decimal | None = None

    # Outer retry loop for network/rate limit issues
    for attempt in range(MAX_API_RETRIES + 1):
        balance_info: dict | None = None
        successful_acc_type: str | None = None
        inner_loop_error = False  # Flag if inner loop breaks due to error

        # Inner loop to try different account types
        for acc_type in account_types_to_try:
            try:
                params = {"accountType": acc_type, "coin": currency}
                lg.debug(f"Fetching balance with params={params} (Attempt {attempt + 1})")
                # Use safe_ccxt_call with 0 inner retries; outer loop handles retries
                balance_info = safe_ccxt_call(exchange, "fetch_balance", lg, max_retries=0, params=params)

                parsed_balance = _parse_balance_response(balance_info, currency, acc_type, lg)

                if parsed_balance is not None:
                    successful_acc_type = acc_type
                    lg.info(f"Available {currency} balance ({successful_acc_type}): {parsed_balance:.4f}")
                    return parsed_balance  # Return the found balance
                else:
                    lg.debug(f"Balance for {currency} not found or parsing failed for type {acc_type}.")
                    balance_info = None  # Reset for next type

            except ccxt.ExchangeError as e:
                # Non-retryable errors should have been raised by safe_ccxt_call(max_retries=0)
                # Log other exchange errors and try next type
                lg.debug(f"Exchange error fetching balance type {acc_type}: {e}. Trying next type if available.")
                last_exception = e
                continue  # Try the next account type

            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                lg.warning(f"Network/RateLimit error during balance fetch type {acc_type}: {e}")
                last_exception = e
                inner_loop_error = True
                break  # Break inner loop, let outer loop retry

            except Exception as e:
                lg.error(f"Unexpected error during balance fetch type {acc_type}: {e}", exc_info=True)
                last_exception = e
                inner_loop_error = True
                break  # Break inner loop, let outer loop retry

        # --- After Inner Loop ---
        if inner_loop_error:  # If broke from inner loop due to error
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2**attempt)
                lg.warning(
                    f"Balance fetch attempt {attempt + 1} encountered recoverable error. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue  # Continue outer retry loop
            else:
                lg.error(
                    f"{NEON_RED}Max retries reached fetching balance for {currency} after network/unexpected error. Last error: {last_exception}{RESET}"
                )
                return None

        # If inner loop completed without finding balance AND without errors breaking it
        if parsed_balance is None:
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2**attempt)
                lg.warning(
                    f"Balance fetch attempt {attempt + 1} failed to find/parse balance for type(s): {account_types_to_try}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue  # Retry outer loop
            else:
                lg.error(
                    f"{NEON_RED}Max retries reached. Failed to fetch/parse balance for {currency} using types: {account_types_to_try}. Last error: {last_exception}{RESET}"
                )
                return None

    # Fallback if logic somehow exits loop without returning
    lg.error(f"{NEON_RED}Balance fetch logic completed unexpectedly without returning a value for {currency}.{RESET}")
    return None


def _parse_balance_response(
    balance_info: dict | None, currency: str, account_type_checked: str, logger: logging.Logger
) -> Decimal | None:
    """Parses the raw response from CCXT's fetch_balance, adapting to Bybit V5 structure.
    Specifically looks for 'availableBalance' or 'walletBalance' in the nested V5 response.

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
    available_balance_str: str | None = None

    try:
        # --- Strategy 1: Prioritize Bybit V5 structure (info -> result -> list -> coin[]) ---
        if (
            "info" in balance_info
            and isinstance(balance_info["info"], dict)
            and balance_info["info"].get("retCode") == 0
            and "result" in balance_info["info"]
            and isinstance(balance_info["info"]["result"], dict)
            and "list" in balance_info["info"]["result"]
            and isinstance(balance_info["info"]["result"]["list"], list)
        ):
            balance_list = balance_info["info"]["result"]["list"]
            lg.debug(
                f"Parsing V5 balance structure for {currency} ({account_type_checked}). List length: {len(balance_list)}"
            )

            for account_data in balance_list:
                # Find the dictionary matching the account type we requested
                if isinstance(account_data, dict) and account_data.get("accountType") == account_type_checked:
                    coin_list = account_data.get("coin")
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                            if isinstance(coin_data, dict) and coin_data.get("coin") == currency:
                                # Priority: 1. availableToWithdraw ('availableBalance' for UTA)
                                #           2. walletBalance (total balance)
                                #           3. availableBalance (Non-UTA Contract/Spot available)
                                free = coin_data.get("availableToWithdraw")  # UTA uses this primarily
                                if free is None or str(free).strip() == "":
                                    free = coin_data.get("availableBalance")  # Check this too, esp. for non-UTA
                                if free is None or str(free).strip() == "":
                                    lg.debug(
                                        f"'availableToWithdraw'/'availableBalance' missing/empty for {currency} in {account_type_checked}, trying 'walletBalance'"
                                    )
                                    free = coin_data.get("walletBalance")

                                if free is not None and str(free).strip() != "":
                                    available_balance_str = str(free)
                                    field_used = "unknown"
                                    if free == coin_data.get("availableToWithdraw"):
                                        field_used = "availableToWithdraw"
                                    elif free == coin_data.get("availableBalance"):
                                        field_used = "availableBalance"
                                    elif free == coin_data.get("walletBalance"):
                                        field_used = "walletBalance"
                                    lg.debug(
                                        f"Parsed balance from Bybit V5 ({account_type_checked} -> {currency}): Value='{available_balance_str}' (Field: '{field_used}')"
                                    )
                                    break  # Found currency in this account's coin list
                        if available_balance_str is not None:
                            break  # Found currency in this account type
            if available_balance_str is None:
                lg.debug(
                    f"Currency '{currency}' not found within Bybit V5 list structure for account type '{account_type_checked}'."
                )

        # --- Strategy 2: Fallback to standard CCXT 'free' balance structure ---
        elif (
            available_balance_str is None and currency in balance_info and isinstance(balance_info.get(currency), dict)
        ):
            free_val = balance_info[currency].get("free")
            if free_val is not None:
                available_balance_str = str(free_val)
                lg.debug(f"Parsed balance via standard CCXT structure ['{currency}']['free']: {available_balance_str}")

        # --- Strategy 3: Fallback to top-level 'free' dictionary (less common) ---
        elif available_balance_str is None and "free" in balance_info and isinstance(balance_info.get("free"), dict):
            free_val = balance_info["free"].get(currency)
            if free_val is not None:
                available_balance_str = str(free_val)
                lg.debug(
                    f"Parsed balance via top-level 'free' dictionary ['free']['{currency}']: {available_balance_str}"
                )

        # --- Conversion and Validation ---
        if available_balance_str is None:
            lg.debug(
                f"Could not extract available balance for {currency} from response structure ({account_type_checked}). Response info keys: {balance_info.get('info', {}).keys() if isinstance(balance_info.get('info'), dict) else 'N/A'}"
            )
            return None

        final_balance = Decimal(available_balance_str)
        if final_balance.is_finite() and final_balance >= 0:
            return final_balance
        else:
            lg.error(f"Parsed balance for {currency} ('{available_balance_str}') is invalid (negative or non-finite).")
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}.")
        return None
    except Exception as e:
        lg.error(f"Unexpected error parsing balance response structure for {currency}: {e}", exc_info=True)
        return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes market data using pandas_ta.Strategy to generate weighted trading signals.
    Calculates technical indicators, Fibonacci levels, and provides risk management values.
    Uses Decimal for internal calculations involving prices and quantities.
    Manages symbol-specific state like break-even status via a shared dictionary.
    """

    def __init__(
        self,
        df_raw: pd.DataFrame,  # DataFrame with Decimal OHLCV columns
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
            symbol_state: Mutable dictionary holding state for this symbol (e.g., 'break_even_triggered').

        Raises:
            ValueError: If df_raw, market_info, or symbol_state is invalid.
        """
        if df_raw is None or df_raw.empty:
            raise ValueError("TradingAnalyzer requires a non-empty raw DataFrame.")
        if not market_info:
            raise ValueError("TradingAnalyzer requires valid market_info.")
        if symbol_state is None:  # Check for None explicitly
            raise ValueError("TradingAnalyzer requires a valid symbol_state dictionary.")

        self.df_raw = df_raw  # Keep raw Decimal DF for precise checks (e.g., Fib)
        self.df = df_raw.copy()  # Work on a copy for TA calculations requiring floats
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol: str = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval: str = config.get("interval", "UNKNOWN_INTERVAL")
        self.symbol_state = symbol_state  # Store reference to the mutable state dict

        # --- Internal State ---
        self.indicator_values: dict[str, Decimal | None] = {}  # Stores latest indicator values as Decimals
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Current signal state
        self.active_weight_set_name: str = config.get("active_weight_set", "default")
        self.weights: dict[str, float | str] = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: dict[str, Decimal] = {}  # Stores calculated Fibonacci levels
        self.ta_strategy: ta.Strategy | None = None  # pandas_ta strategy object
        self.ta_column_map: dict[str, str] = {}  # Map generic names to pandas_ta column names

        if not self.weights:
            logger.warning(
                f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' is empty or not found for {self.symbol}. Signal generation will be disabled.{RESET}"
            )

        # --- Data Preparation for pandas_ta ---
        self._convert_df_for_ta()  # Convert self.df columns to float

        # --- Initialize and Calculate Indicators ---
        if not self.df.empty:
            self._define_ta_strategy()
            self._calculate_all_indicators()  # Operates on self.df (float version)
            self._update_latest_indicator_values()  # Populates self.indicator_values with Decimals from self.df
            self.calculate_fibonacci_levels()  # Calculate initial Fib levels using self.df_raw (Decimal version)
        else:
            logger.warning(f"DataFrame is empty after float conversion for {self.symbol}. Cannot calculate indicators.")

    def _convert_df_for_ta(self) -> None:
        """Converts necessary DataFrame columns (OHLCV) in self.df to float for pandas_ta compatibility."""
        try:
            cols_to_convert = ["open", "high", "low", "close", "volume"]
            for col in cols_to_convert:
                if col in self.df.columns:
                    # Check if column is already float
                    if pd.api.types.is_float_dtype(self.df[col]):
                        continue  # Skip if already float
                    # Check if column is Decimal (stored as object) or other numeric type
                    elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_numeric_dtype(self.df[col]):
                        # Convert to float, coercing errors to NaN
                        self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                    else:
                        # Attempt conversion for other types, coercing errors
                        self.logger.debug(
                            f"Attempting float conversion for non-numeric/object column '{col}' (dtype: {self.df[col].dtype})"
                        )
                        self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            # Verify conversion and log dtypes
            converted_dtypes = {col: self.df[col].dtype for col in cols_to_convert if col in self.df.columns}
            self.logger.debug(f"DataFrame dtypes prepared for TA: {converted_dtypes}")
            # Check if any conversion failed resulting in non-float types
            for col, dtype in converted_dtypes.items():
                if not pd.api.types.is_float_dtype(dtype):
                    self.logger.warning(
                        f"Column '{col}' is not float (dtype: {dtype}) after conversion attempt. TA results may be affected."
                    )

        except Exception as e:
            self.logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
            self.df = pd.DataFrame()  # Set to empty to prevent further processing

    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even triggered status from the shared symbol state."""
        return self.symbol_state.get("break_even_triggered", False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool) -> None:
        """Sets the break-even triggered status in the shared symbol state and logs change."""
        if not isinstance(value, bool):
            self.logger.error(f"Invalid type for break_even_triggered ({type(value)}). Must be boolean.")
            return
        current_value = self.symbol_state.get("break_even_triggered")
        if current_value != value:
            self.symbol_state["break_even_triggered"] = value
            self.logger.info(f"Break-even status for {self.symbol} set to: {value}")

    def _define_ta_strategy(self) -> None:
        """Defines the pandas_ta Strategy object based on enabled indicators in the config."""
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})  # Dictionary of enabled indicators

        # Helper to safely get parameters (already validated as int/float in load_config)
        def get_param(key: str, default: int | float) -> int | float:
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
        stochrsi_rsi_w = int(get_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))  # Inner RSI
        stochrsi_k = int(get_param("stoch_rsi_k", DEFAULT_K_WINDOW))
        stochrsi_d = int(get_param("stoch_rsi_d", DEFAULT_D_WINDOW))
        psar_af = float(get_param("psar_af", DEFAULT_PSAR_AF))
        psar_max = float(get_param("psar_max_af", DEFAULT_PSAR_MAX_AF))
        sma10_w = int(get_param("sma_10_window", DEFAULT_SMA_10_WINDOW))
        mom_p = int(get_param("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        vol_ma_p = int(get_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))

        # Build the list of indicators for the pandas_ta Strategy
        ta_list: list[dict[str, Any]] = []
        self.ta_column_map: dict[str, str] = {}  # Reset map

        # --- Add indicators based on config flags and valid parameters ---
        # ATR (Always calculated for risk management)
        if atr_p > 0:
            ta_list.append({"kind": "atr", "length": atr_p})
            self.ta_column_map["ATR"] = f"ATRr_{atr_p}"
        else:
            self.logger.error(f"ATR period ({atr_p}) is invalid. Risk management calculations will fail.")

        # EMAs (Needed if alignment or MA cross exit enabled)
        if indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit"):
            if ema_s > 0:
                col_name_s = f"EMA_{ema_s}"
                ta_list.append({"kind": "ema", "length": ema_s, "col_names": (col_name_s,)})
                self.ta_column_map["EMA_Short"] = col_name_s
            if ema_l > 0 and ema_l > ema_s:
                col_name_l = f"EMA_{ema_l}"
                ta_list.append({"kind": "ema", "length": ema_l, "col_names": (col_name_l,)})
                self.ta_column_map["EMA_Long"] = col_name_l
            elif ema_l <= ema_s:
                self.logger.warning(f"EMA Long period ({ema_l}) must be > Short ({ema_s}). Disabling EMA Long.")

        # Momentum
        if indi_cfg.get("momentum") and mom_p > 0:
            col_name = f"MOM_{mom_p}"
            ta_list.append({"kind": "mom", "length": mom_p, "col_names": (col_name,)})
            self.ta_column_map["Momentum"] = col_name

        # Volume SMA
        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0:
            col_name = f"VOL_SMA_{vol_ma_p}"
            ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_p, "col_names": (col_name,)})
            self.ta_column_map["Volume_MA"] = col_name

        # Stochastic RSI
        if indi_cfg.get("stoch_rsi") and all(p > 0 for p in [stochrsi_w, stochrsi_rsi_w, stochrsi_k, stochrsi_d]):
            k_col = f"STOCHRSIk_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            d_col = f"STOCHRSId_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            ta_list.append(
                {
                    "kind": "stochrsi",
                    "length": stochrsi_w,
                    "rsi_length": stochrsi_rsi_w,
                    "k": stochrsi_k,
                    "d": stochrsi_d,
                    "col_names": (k_col, d_col),
                }
            )
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
            bb_std_str = f"{bb_std:.1f}".replace(".", "_")
            bbl = f"BBL_{bb_p}_{bb_std_str}"
            bbm = f"BBM_{bb_p}_{bb_std_str}"
            bbu = f"BBU_{bb_p}_{bb_std_str}"
            bbb = f"BBB_{bb_p}_{bb_std_str}"
            bbp = f"BBP_{bb_p}_{bb_std_str}"
            ta_list.append({"kind": "bbands", "length": bb_p, "std": bb_std, "col_names": (bbl, bbm, bbu, bbb, bbp)})
            self.ta_column_map["BB_Lower"] = bbl
            self.ta_column_map["BB_Middle"] = bbm
            self.ta_column_map["BB_Upper"] = bbu

        # VWAP
        if indi_cfg.get("vwap"):
            # VWAP calculation needs high, low, close, volume. pandas_ta handles this.
            # Default daily VWAP calculation in pandas_ta might reset daily ('D').
            # If you need session-based VWAP, more complex handling might be needed.
            vwap_col = "VWAP_D"  # Default pandas_ta name (may vary based on params/version)
            ta_list.append({"kind": "vwap", "anchor": "D", "col_names": (vwap_col,)})  # Use daily anchor
            self.ta_column_map["VWAP"] = vwap_col

        # CCI
        if indi_cfg.get("cci") and cci_w > 0:
            cci_col = f"CCI_{cci_w}_0.015"  # Default constant in pandas_ta CCI name
            ta_list.append({"kind": "cci", "length": cci_w, "col_names": (cci_col,)})
            self.ta_column_map["CCI"] = cci_col

        # Williams %R
        if indi_cfg.get("wr") and wr_w > 0:
            wr_col = f"WILLR_{wr_w}"
            ta_list.append({"kind": "willr", "length": wr_w, "col_names": (wr_col,)})
            self.ta_column_map["WR"] = wr_col

        # Parabolic SAR
        if indi_cfg.get("psar"):
            psar_af_str = f"{psar_af}".rstrip("0").rstrip(".")  # Clean format for name
            psar_max_str = f"{psar_max}".rstrip("0").rstrip(".")
            l_col = f"PSARl_{psar_af_str}_{psar_max_str}"
            s_col = f"PSARs_{psar_af_str}_{psar_max_str}"
            af_col = f"PSARaf_{psar_af_str}_{psar_max_str}"
            r_col = f"PSARr_{psar_af_str}_{psar_max_str}"
            ta_list.append(
                {"kind": "psar", "af": psar_af, "max_af": psar_max, "col_names": (l_col, s_col, af_col, r_col)}
            )
            self.ta_column_map["PSAR_Long"] = l_col  # Value when SAR is below (long trend)
            self.ta_column_map["PSAR_Short"] = s_col  # Value when SAR is above (short trend)
            self.ta_column_map["PSAR_AF"] = af_col  # Current Acceleration Factor
            self.ta_column_map["PSAR_Reversal"] = r_col  # 1 if SAR reversed on this bar

        # SMA 10
        if indi_cfg.get("sma_10") and sma10_w > 0:
            col_name = f"SMA_{sma10_w}"
            ta_list.append({"kind": "sma", "length": sma10_w, "col_names": (col_name,)})
            self.ta_column_map["SMA10"] = col_name

        # MFI
        if indi_cfg.get("mfi") and mfi_w > 0:
            # MFI requires high, low, close, volume, which should be present and converted to float
            col_name = f"MFI_{mfi_w}"
            ta_list.append({"kind": "mfi", "length": mfi_w, "col_names": (col_name,)})
            self.ta_column_map["MFI"] = col_name

        # --- Create Strategy ---
        if not ta_list:
            self.logger.warning(
                f"No valid indicators enabled or configured for {self.symbol}. TA Strategy not created."
            )
            return

        self.ta_strategy = ta.Strategy(
            name="EnhancedMultiIndicator",
            description="Calculates multiple TA indicators based on bot config",
            ta=ta_list,
        )
        self.logger.debug(f"Defined TA Strategy for {self.symbol} with {len(ta_list)} indicator groups.")
        # self.logger.debug(f"TA Column Map: {self.ta_column_map}")

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using the defined pandas_ta strategy on the float DataFrame (self.df)."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return
        if not self.ta_strategy:
            self.logger.warning(f"TA Strategy not defined for {self.symbol}, cannot calculate indicators.")
            return

        # Check if sufficient data exists
        min_required_data = self.ta_strategy.required if hasattr(self.ta_strategy, "required") else 50
        buffer = 20  # Add buffer for calculation stability
        if len(self.df) < min_required_data + buffer:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient data ({len(self.df)} rows) for {self.symbol} TA calculation (min recommended: {min_required_data + buffer}). Results may be inaccurate.{RESET}"
            )

        try:
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol}...")
            # Apply the strategy to the DataFrame (modifies self.df inplace)
            # Ensure the DataFrame has float types before this call (_convert_df_for_ta)
            self.df.ta.strategy(self.ta_strategy, timed=False)  # timed=True adds overhead
            self.logger.debug(
                f"Finished indicator calculations for {self.symbol}."
            )  # Columns: {self.df.columns.tolist()}")
        except AttributeError as ae:
            # This might catch issues if _convert_df_for_ta failed silently or pandas_ta internal issue
            if "'Decimal' object has no attribute" in str(ae) or "decimal" in str(ae).lower():
                self.logger.error(
                    f"{NEON_RED}Pandas TA Error ({self.symbol}): Input must be float, not Decimal. Check data conversion. Error: {ae}{RESET}",
                    exc_info=False,
                )
            elif "'float' object has no attribute" in str(ae) and ("high" in str(ae) or "low" in str(ae)):
                self.logger.error(
                    f"{NEON_RED}Pandas TA Error ({self.symbol}): Input columns (e.g., high, low) might contain NaNs or non-numeric data. Error: {ae}{RESET}",
                    exc_info=False,
                )
                self.logger.debug(f"Problematic DF sample:\n{self.df.tail()}")
            else:
                self.logger.error(
                    f"{NEON_RED}Pandas TA attribute error ({self.symbol}): {ae}. Is pandas_ta installed/working?{RESET}",
                    exc_info=True,
                )
        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Error calculating indicators with pandas_ta strategy for {self.symbol}: {e}{RESET}",
                exc_info=True,
            )

    def _update_latest_indicator_values(self) -> None:
        """Updates self.indicator_values (dict of Decimals) with the latest calculated
        values from the float DataFrame (self.df), converting them back to Decimal.
        """
        self.indicator_values = {}  # Reset before populating
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(
                f"DataFrame empty or last row is all NaN for {self.symbol}. Cannot update latest indicator values."
            )
            return

        try:
            latest_series = self.df.iloc[-1]

            # Helper to safely convert float/object back to Decimal
            def to_decimal(value: Any) -> Decimal | None:
                if pd.isna(value) or value is None:
                    return None
                try:
                    # Convert float to string first for precise Decimal conversion
                    dec_val = Decimal(str(value))
                    return dec_val if dec_val.is_finite() else None
                except (InvalidOperation, ValueError, TypeError):
                    return None

            # Populate indicator_values using the ta_column_map
            for generic_name, actual_col_name in self.ta_column_map.items():
                if actual_col_name in latest_series:
                    self.indicator_values[generic_name] = to_decimal(latest_series.get(actual_col_name))
                else:
                    self.logger.debug(
                        f"Column '{actual_col_name}' not found in DataFrame for indicator '{generic_name}' ({self.symbol})."
                    )

            # Also add latest OHLCV values (from the float df, converted back to Decimal)
            for base_col in ["open", "high", "low", "close", "volume"]:
                if base_col in latest_series:
                    self.indicator_values[base_col.capitalize()] = to_decimal(latest_series.get(base_col))

            valid_values_count = sum(1 for v in self.indicator_values.values() if v is not None)
            self.logger.debug(
                f"Latest indicator Decimal values updated for {self.symbol}: Count={valid_values_count}/{len(self.ta_column_map) + 5}"
            )
            # For detailed logging:
            # valid_values_str = {k: f"{v:.5f}" for k, v in self.indicator_values.items() if v is not None}
            # self.logger.debug(f"Values: {valid_values_str}")

        except IndexError:
            self.logger.error(f"DataFrame index out of bounds when updating latest indicator values for {self.symbol}.")
        except Exception as e:
            self.logger.error(
                f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True
            )
            self.indicator_values = {}

    # --- Precision and Market Info Helpers ---
    def get_min_tick_size(self) -> Decimal | None:
        """Gets the minimum price tick size as a Decimal from market info."""
        tick = self.market_info.get("min_tick_size")
        if tick is None or not isinstance(tick, Decimal) or not tick.is_finite() or tick <= 0:
            self.logger.warning(f"Invalid or missing min_tick_size ({tick}) for {self.symbol}. Quantization may fail.")
            return None
        return tick

    def get_price_precision_digits(self) -> int:
        """Gets the number of decimal places for price precision."""
        return self.market_info.get("price_precision_digits", 8)  # Default to 8

    def get_amount_precision_digits(self) -> int:
        """Gets the number of decimal places for amount (quantity) precision."""
        return self.market_info.get("amount_precision_digits", 8)  # Default to 8

    def quantize_price(self, price: Decimal | float | str, rounding: str = ROUND_DOWN) -> Decimal | None:
        """Quantizes a price to the market's minimum tick size using specified rounding."""
        min_tick = self.get_min_tick_size()
        if min_tick is None:
            return None
        try:
            price_decimal = Decimal(str(price))
            if not price_decimal.is_finite():
                return None
            # Formula: quantize(price / tick_size) * tick_size
            # Use Decimal's quantize method with the tick size as the exponent
            quantized = price_decimal.quantize(min_tick, rounding=rounding)
            return quantized
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing price '{price}' for {self.symbol}: {e}")
            return None

    def quantize_amount(self, amount: Decimal | float | str, rounding: str = ROUND_DOWN) -> Decimal | None:
        """Quantizes an amount (quantity) to the market's amount precision (step size)."""
        amount_digits = self.get_amount_precision_digits()
        try:
            amount_decimal = Decimal(str(amount))
            if not amount_decimal.is_finite():
                return None
            # Calculate the step size based on precision digits (e.g., 2 digits -> step 0.01)
            step_size = Decimal("1") / (Decimal("10") ** amount_digits)
            # Quantize using the step size
            quantized = (amount_decimal / step_size).quantize(Decimal("0"), rounding=rounding) * step_size
            # Re-quantize to ensure exact number of decimal places
            return quantized.quantize(step_size)
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing amount '{amount}' for {self.symbol}: {e}")
            return None

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels based on the high/low over a specified window.
        Uses the raw DataFrame (Decimal precision) and quantizes the resulting levels.

        Args:
            window: The lookback period (number of candles). Uses config value if None.

        Returns:
            A dictionary of Fibonacci levels (e.g., "Fib_38.2%") mapped to quantized price Decimals.
            Returns an empty dictionary if calculation is not possible.
        """
        self.fib_levels_data = {}  # Reset previous levels
        window = window or int(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))

        if len(self.df_raw) < window:
            self.logger.debug(
                f"Not enough data ({len(self.df_raw)} rows) for Fibonacci window ({window}) on {self.symbol}."
            )
            return {}

        # Use the raw DataFrame (df_raw) which should have Decimal columns
        df_slice = self.df_raw.tail(window)

        try:
            if "high" not in df_slice.columns or "low" not in df_slice.columns:
                self.logger.warning(f"Missing 'high' or 'low' column in df_raw for Fib calculation ({self.symbol}).")
                return {}

            # Extract High/Low Series (should be Decimal type)
            high_series = df_slice["high"]
            low_series = df_slice["low"]

            # Find max high and min low, handling potential NaNs or non-Decimals gracefully
            # Convert to numeric first, coercing errors, then find max/min, then convert result back to Decimal
            high_numeric = pd.to_numeric(high_series, errors="coerce").dropna()
            low_numeric = pd.to_numeric(low_series, errors="coerce").dropna()

            if high_numeric.empty or low_numeric.empty:
                self.logger.warning(
                    f"Could not find valid high/low data in the last {window} periods for Fib calculation on {self.symbol} after cleaning."
                )
                return {}

            high_raw = high_numeric.max()
            low_raw = low_numeric.min()

            # Convert max/min back to Decimal
            high = Decimal(str(high_raw))
            low = Decimal(str(low_raw))

            if not high.is_finite() or not low.is_finite():
                self.logger.warning(
                    f"Non-finite high/low values after aggregation for Fib calculation on {self.symbol}. High={high}, Low={low}"
                )
                return {}
            if high <= low:  # Use <= to handle cases where high == low
                self.logger.warning(
                    f"Invalid range (High <= Low): High={high}, Low={low} in window for Fib calculation on {self.symbol}. Cannot calculate levels."
                )
                return {}

            diff: Decimal = high - low
            levels: dict[str, Decimal] = {}
            min_tick: Decimal | None = self.get_min_tick_size()

            if diff > 0 and min_tick is not None:
                # Calculate levels only if range is valid and tick size is available
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float))
                    level_price_raw = high - (diff * level_pct)
                    # Quantize the calculated level price (round down for potential support/resistance)
                    level_price_quantized = self.quantize_price(level_price_raw, rounding=ROUND_DOWN)

                    if level_price_quantized is not None:
                        level_name = f"Fib_{level_pct * 100:.1f}%"
                        levels[level_name] = level_price_quantized
                    else:
                        self.logger.warning(
                            f"Failed to quantize Fibonacci level {level_pct * 100:.1f}% (Raw: {level_price_raw}) for {self.symbol}"
                        )
            elif min_tick is None:
                self.logger.warning(
                    f"Calculating raw (non-quantized) Fibonacci levels for {self.symbol} due to missing min_tick_size."
                )
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float))
                    level_price_raw = high - (diff * level_pct)
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_raw  # Store raw Decimal

            self.fib_levels_data = levels
            # Log the calculated levels (optional)
            price_prec = self.get_price_precision_digits()
            log_levels = {k: f"{v:.{price_prec}f}" for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} (Window: {window}): {log_levels}")
            return levels

        except Exception as e:
            # Catch potential errors during series access, conversion, or calculation
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    # --- Indicator Check Methods ---
    # Return Optional[float] score [-1.0, 1.0] or None if unavailable

    def _get_indicator_float(self, name: str) -> float | None:
        """Safely retrieves an indicator value from self.indicator_values as a float."""
        val_decimal = self.indicator_values.get(name)
        if val_decimal is None or not val_decimal.is_finite():
            return None
        try:
            return float(val_decimal)
        except (ValueError, TypeError):
            return None

    def _check_ema_alignment(self) -> float | None:
        """Checks if short EMA is above/below long EMA. Score: 1.0 (short > long), -1.0 (short < long), 0.0 (equal)."""
        ema_s = self._get_indicator_float("EMA_Short")
        ema_l = self._get_indicator_float("EMA_Long")
        if ema_s is None or ema_l is None:
            return None
        if ema_s > ema_l:
            return 1.0
        if ema_s < ema_l:
            return -1.0
        return 0.0

    def _check_momentum(self) -> float | None:
        """Checks Momentum indicator value. Positive -> bullish, Negative -> bearish. Score scaled & clamped."""
        mom = self._get_indicator_float("Momentum")
        if mom is None:
            return None
        # Basic scaling: assumes momentum roughly centered around 0. Adjust if needed.
        # Example: If MOM is typically +/- 5, scale by 0.2. Needs tuning based on typical values.
        scaling_factor = 0.1  # Adjust this based on observed momentum range
        score = mom * scaling_factor
        return max(-1.0, min(1.0, score))  # Clamp to [-1.0, 1.0]

    def _check_volume_confirmation(self) -> float | None:
        """Checks if current volume exceeds its MA by a multiplier. Score: 0.7 (exceeds), 0.0 (doesn't)."""
        vol = self._get_indicator_float("Volume")
        vol_ma = self._get_indicator_float("Volume_MA")
        if vol is None or vol_ma is None or vol_ma <= 0:
            return None  # Need MA > 0 for comparison
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        return 0.7 if vol > vol_ma * multiplier else 0.0

    def _check_stoch_rsi(self) -> float | None:
        """Checks Stochastic RSI K and D lines for overbought/oversold and crossovers. Score range [-1.0, 1.0]."""
        k = self._get_indicator_float("StochRSI_K")
        d = self._get_indicator_float("StochRSI_D")
        if k is None or d is None:
            return None
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        score = 0.0
        # Strong signals in OB/OS zones with crossover confirmation
        if k < oversold and d < oversold:
            score = 1.0 if k > d else 0.8  # Bullish crossover deep OS (stronger if k>d)
        elif k > overbought and d > overbought:
            score = -1.0 if k < d else -0.8  # Bearish crossover deep OB (stronger if k<d)
        # Weaker signals just entering OB/OS
        elif k < oversold:
            score = 0.6
        elif k > overbought:
            score = -0.6
        # Mid-range crossover signals
        elif k > d:
            score = 0.3  # Bullish crossover mid-range
        elif k < d:
            score = -0.3  # Bearish crossover mid-range
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float | None:
        """Checks RSI value against OB/OS levels. Score scaled linearly between 0 and 100, clamped."""
        rsi = self._get_indicator_float("RSI")
        if rsi is None:
            return None
        ob = 70.0
        os = 30.0  # Common thresholds
        # Simple linear scale: 1.0 at 0, 0.0 at 50, -1.0 at 100
        score = (50.0 - rsi) / 50.0
        # Add extra weight for extremes (optional boost beyond OB/OS)
        if rsi >= ob:
            score = max(-1.0, score * 1.1)  # Increase bearishness above OB
        if rsi <= os:
            score = min(1.0, score * 1.1)  # Increase bullishness below OS
        return max(-1.0, min(1.0, score))

    def _check_cci(self) -> float | None:
        """Checks CCI against OB/OS levels (+/- 100). Score scaled linearly between thresholds, clamped."""
        cci = self._get_indicator_float("CCI")
        if cci is None:
            return None
        ob = 100.0
        os = -100.0
        # Scale based on thresholds
        if cci >= ob:
            score = -1.0  # Strong sell signal above 100
        elif cci <= os:
            score = 1.0  # Strong buy signal below -100
        else:
            score = -cci / ob  # Linear scale between -100 and 100 (cci=50 -> -0.5, cci=-50 -> 0.5)
        return max(-1.0, min(1.0, score))

    def _check_wr(self) -> float | None:
        """Checks Williams %R against OB/OS levels (-20 / -80). Score scaled linearly, clamped."""
        wr = self._get_indicator_float("WR")  # Typically -100 to 0
        if wr is None:
            return None
        ob = -20.0
        os = -80.0
        # Scale: 1.0 at -100, 0.0 at -50, -1.0 at 0
        score = (wr + 50.0) / -50.0
        # Add extra weight for extremes (optional boost)
        if wr >= ob:
            score = max(-1.0, score * 1.1)  # Increase bearishness above OB (-20)
        if wr <= os:
            score = min(1.0, score * 1.1)  # Increase bullishness below OS (-80)
        return max(-1.0, min(1.0, score))

    def _check_psar(self) -> float | None:
        """Checks Parabolic SAR position relative to price. Score: 1.0 (SAR below), -1.0 (SAR above), 0.0 (transition/error)."""
        psar_l = self.indicator_values.get("PSAR_Long")  # Decimal or None (value when SAR is long)
        psar_s = self.indicator_values.get("PSAR_Short")  # Decimal or None (value when SAR is short)
        # If PSAR_Long has a value (is finite, not NaN), SAR is below price -> bullish
        psar_l_active = psar_l is not None and psar_l.is_finite()
        # If PSAR_Short has a value, SAR is above price -> bearish
        psar_s_active = psar_s is not None and psar_s.is_finite()

        if psar_l_active and not psar_s_active:
            return 1.0  # SAR below price (bullish trend)
        if psar_s_active and not psar_l_active:
            return -1.0  # SAR above price (bearish trend)
        # If both active/inactive or NaN, implies transition or error
        return 0.0

    def _check_sma10(self) -> float | None:
        """Checks if close price is above/below SMA10. Score: 0.5 (above), -0.5 (below), 0.0 (equal)."""
        sma = self._get_indicator_float("SMA10")
        close = self._get_indicator_float("Close")
        if sma is None or close is None:
            return None
        if close > sma:
            return 0.5
        if close < sma:
            return -0.5
        return 0.0

    def _check_vwap(self) -> float | None:
        """Checks if close price is above/below VWAP. Score: 0.6 (above), -0.6 (below), 0.0 (equal)."""
        vwap = self._get_indicator_float("VWAP")
        close = self._get_indicator_float("Close")
        if vwap is None or close is None:
            return None
        if close > vwap:
            return 0.6
        if close < vwap:
            return -0.6
        return 0.0

    def _check_mfi(self) -> float | None:
        """Checks Money Flow Index against OB/OS levels (80/20). Score scaled linearly, clamped."""
        mfi = self._get_indicator_float("MFI")
        if mfi is None:
            return None
        ob = 80.0
        os = 20.0
        # Scale similar to RSI: 1.0 at 0, 0.0 at 50, -1.0 at 100
        score = (50.0 - mfi) / 50.0
        if mfi >= ob:
            score = max(-1.0, score * 1.1)  # Boost bearishness
        if mfi <= os:
            score = min(1.0, score * 1.1)  # Boost bullishness
        return max(-1.0, min(1.0, score))

    def _check_bollinger_bands(self) -> float | None:
        """Checks close price relative to Bollinger Bands. Score: 1.0 (below lower), -1.0 (above upper), scaled linearly within bands."""
        bbl = self._get_indicator_float("BB_Lower")
        bbu = self._get_indicator_float("BB_Upper")
        close = self._get_indicator_float("Close")
        if bbl is None or bbu is None or close is None or bbu <= bbl:
            return None  # Invalid bands
        if close <= bbl:
            return 1.0  # Below or touching lower band (strong buy/reversal potential)
        if close >= bbu:
            return -1.0  # Above or touching upper band (strong sell/reversal potential)
        # Scale position within bands: 0 at middle, +1 near lower, -1 near upper
        position = (close - bbl) / (bbu - bbl)  # Result is 0 (at lower) to 1 (at upper)
        score = 1.0 - 2.0 * position  # Scale to +1 (at lower) to -1 (at upper)
        return max(-1.0, min(1.0, score))

    def _check_orderbook(self, orderbook_data: dict | None) -> float | None:
        """Calculates Order Book Imbalance (OBI) from fetched order book data. Score: [-1.0, 1.0]."""
        if not orderbook_data:
            return None
        try:
            limit = int(self.config.get("orderbook_limit", 10))
            # Ensure bids/asks are lists of lists/tuples [price, amount]
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])
            if not isinstance(bids, list) or not isinstance(asks, list):
                return 0.0

            # Take top N levels
            top_bids = bids[:limit]
            top_asks = asks[:limit]
            if not top_bids or not top_asks:
                return 0.0

            # Sum volume (amount) at each level, converting to Decimal
            bid_vol = sum(Decimal(str(b[1])) for b in top_bids if len(b) > 1)
            ask_vol = sum(Decimal(str(a[1])) for a in top_asks if len(a) > 1)
            total_vol = bid_vol + ask_vol
            if total_vol <= 0:
                return 0.0

            # Calculate OBI = (BidVol - AskVol) / TotalVol
            obi = (bid_vol - ask_vol) / total_vol  # Decimal result
            return float(max(Decimal("-1.0"), min(Decimal("1.0"), obi)))  # Clamp and convert to float

        except (InvalidOperation, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error calculating Order Book Imbalance for {self.symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in OBI calculation for {self.symbol}: {e}", exc_info=True)
            return None

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: dict | None) -> str:
        """Generates a final trading signal ('BUY', 'SELL', 'HOLD') based on weighted scores
        from enabled indicators. Uses Decimal for score accumulation.
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Reset signals
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores: dict[str, str] = {}

        # --- Pre-checks ---
        if not self.indicator_values:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Indicator values not calculated.")
            return "HOLD"
        if not current_price_dec.is_finite() or current_price_dec <= 0:
            self.logger.warning(
                f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price_dec})."
            )
            return "HOLD"
        if not self.weights:
            self.logger.debug(f"No weights found for active set '{self.active_weight_set_name}'. Holding.")
            return "HOLD"

        # --- Map Indicator Keys to Check Methods ---
        indicator_check_methods = {
            "ema_alignment": self._check_ema_alignment,
            "momentum": self._check_momentum,
            "volume_confirmation": self._check_volume_confirmation,
            "stoch_rsi": self._check_stoch_rsi,
            "rsi": self._check_rsi,
            "bollinger_bands": self._check_bollinger_bands,
            "vwap": self._check_vwap,
            "cci": self._check_cci,
            "wr": self._check_wr,
            "psar": self._check_psar,
            "sma_10": self._check_sma10,
            "mfi": self._check_mfi,
            "orderbook": lambda: self._check_orderbook(orderbook_data),  # Lambda for passing data
        }

        # --- Iterate Through Enabled Indicators and Calculate Weighted Score ---
        active_weights = self.weights
        for indicator_key, is_enabled in self.config.get("indicators", {}).items():
            if not is_enabled:
                # self.logger.debug(f"Indicator '{indicator_key}' disabled.")
                debug_scores[indicator_key] = "Disabled"
                continue

            weight_val = active_weights.get(indicator_key)
            if weight_val is None:
                # self.logger.debug(f"No weight found for enabled indicator '{indicator_key}'.")
                debug_scores[indicator_key] = "No Weight"
                continue

            try:
                weight = Decimal(str(weight_val))
            except (ValueError, InvalidOperation, TypeError):
                self.logger.warning(f"Invalid weight '{weight_val}' for '{indicator_key}'. Skipping.")
                debug_scores[indicator_key] = f"Invalid Wt({weight_val})"
                continue
            if weight == Decimal("0"):
                debug_scores[indicator_key] = "Wt=0"
                continue

            check_method = indicator_check_methods.get(indicator_key)
            if check_method:
                indicator_score_float: float | None = None
                try:
                    indicator_score_float = check_method()
                except Exception as e:
                    self.logger.error(f"Error executing check for '{indicator_key}': {e}", exc_info=True)
                    debug_scores[indicator_key] = "Check Error"
                    nan_indicator_count += 1
                    continue

                # Process valid score
                if indicator_score_float is not None and math.isfinite(indicator_score_float):
                    try:
                        # Clamp score to [-1.0, 1.0] before converting to Decimal
                        clamped_score_float = max(-1.0, min(1.0, indicator_score_float))
                        indicator_score_decimal = Decimal(str(clamped_score_float))

                        weighted_score = indicator_score_decimal * weight
                        final_signal_score += weighted_score
                        total_weight_applied += abs(weight)  # Sum absolute weights for normalization/debug
                        active_indicator_count += 1
                        debug_scores[indicator_key] = (
                            f"{indicator_score_float:.2f} (x{weight:.2f}) = {weighted_score:.3f}"
                        )
                    except (InvalidOperation, TypeError) as calc_err:
                        self.logger.error(f"Error processing score/weight for {indicator_key}: {calc_err}")
                        debug_scores[indicator_key] = "Calc Error"
                        nan_indicator_count += 1
                else:
                    # Score is None or NaN/Infinite
                    debug_scores[indicator_key] = "NaN/None"
                    nan_indicator_count += 1
            elif indicator_key in active_weights:
                # Indicator has a weight but no check method defined
                self.logger.warning(f"Check method missing for enabled indicator with weight: '{indicator_key}'")
                debug_scores[indicator_key] = "No Method"

        # --- Determine Final Signal Based on Score and Threshold ---
        final_signal = "HOLD"
        normalized_score = Decimal("0.0")
        if total_weight_applied > 0:
            # Normalize the score based on the sum of absolute weights applied
            # This gives a score roughly in [-1, 1] range if individual scores are [-1, 1]
            normalized_score = (final_signal_score / total_weight_applied).quantize(Decimal("0.0001"))

        # Use specific threshold if active set is 'scalping', otherwise default
        threshold_key = (
            "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
        )
        default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
        try:
            # Threshold is applied to the RAW final_signal_score
            threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
            if not threshold.is_finite() or threshold <= 0:
                raise ValueError("Threshold must be positive")
        except (ValueError, InvalidOperation, TypeError):
            threshold = Decimal(str(default_threshold))
            self.logger.warning(f"Invalid threshold for '{threshold_key}'. Using default: {threshold}")

        # Compare RAW weighted score against threshold
        if final_signal_score >= threshold:
            final_signal = "BUY"
        elif final_signal_score <= -threshold:
            final_signal = "SELL"

        # Log the signal calculation summary
        price_prec = self.get_price_precision_digits()
        signal_color = NEON_GREEN if final_signal == "BUY" else NEON_RED if final_signal == "SELL" else NEON_YELLOW
        log_msg = (
            f"Signal Calc ({self.symbol} @ {current_price_dec:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Indis(Actv/NaN): {active_indicator_count}/{nan_indicator_count}, "
            f"WtSum: {total_weight_applied:.3f}, RawScore: {final_signal_score:.4f}, NormScore: {normalized_score:.4f}, "
            f"Thresh: +/-{threshold:.3f} -> Signal: {signal_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        if self.logger.level <= logging.DEBUG:  # Log details only if debugging
            # Filter out non-contributing indicators for cleaner debug log
            score_details_str = ", ".join(
                [f"{k}: {v}" for k, v in debug_scores.items() if v not in ["Disabled", "No Weight", "Wt=0"]]
            )
            self.logger.debug(f"  Detailed Scores: {score_details_str}")

        # Update internal signal state
        if final_signal in self.signals:
            self.signals[final_signal] = 1
            self.signals["HOLD"] = 1 if final_signal == "HOLD" else 0
        return final_signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_signal: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates quantized Entry, Take Profit (TP), and Stop Loss (SL) prices based on ATR.
        Uses Decimal for precision and validates results.

        Args:
            entry_price_signal: The price near which the signal occurred (e.g., current price).
            signal: "BUY" or "SELL".

        Returns:
            Tuple (Quantized Entry Price, Quantized TP Price, Quantized SL Price).
            Returns (None, None, None) if calculation fails.
        """
        quantized_entry: Decimal | None = None
        take_profit: Decimal | None = None
        stop_loss: Decimal | None = None

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid signal '{signal}' for TP/SL calculation.")
            return None, None, None
        if not entry_price_signal.is_finite() or entry_price_signal <= 0:
            self.logger.error(f"Invalid entry signal price ({entry_price_signal}) for TP/SL calc.")
            return None, None, None

        # --- Quantize Entry Price ---
        # Use ROUND_DOWN for BUY entry, ROUND_UP for SELL entry to be conservative
        # Market order fills might differ significantly, this is a reference point.
        entry_rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
        quantized_entry = self.quantize_price(entry_price_signal, rounding=entry_rounding)
        if quantized_entry is None:
            self.logger.error(f"Failed to quantize entry signal price {entry_price_signal} for {self.symbol}.")
            return None, None, None

        # --- Get ATR and Min Tick ---
        atr_val = self.indicator_values.get("ATR")
        min_tick = self.get_min_tick_size()
        if atr_val is None or not atr_val.is_finite() or atr_val <= 0 or min_tick is None:
            self.logger.warning(
                f"{NEON_YELLOW}Cannot calculate dynamic TP/SL for {self.symbol}: Invalid ATR ({atr_val}) or MinTick ({min_tick}). SL/TP will be None.{RESET}"
            )
            return quantized_entry, None, None  # Return entry price, but no SL/TP

        # --- Calculate SL/TP Offsets ---
        try:
            atr = atr_val
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Calculate offsets from entry price
            sl_offset_raw = atr * sl_mult
            tp_offset_raw = atr * tp_mult

            # Ensure SL offset is at least the minimum required ticks away
            min_sl_offset_value = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
            if sl_offset_raw < min_sl_offset_value:
                self.logger.warning(
                    f"Calculated SL offset ({sl_offset_raw}) < minimum {MIN_TICKS_AWAY_FOR_SLTP} ticks ({min_sl_offset_value}). Adjusting SL offset."
                )
                sl_offset = min_sl_offset_value
            else:
                sl_offset = sl_offset_raw

            # TP offset doesn't necessarily need a minimum distance, but must be positive
            tp_offset = tp_offset_raw if tp_offset_raw > 0 else min_tick  # Use min tick if calc is zero/negative

            # --- Calculate Raw TP/SL Prices ---
            if signal == "BUY":
                sl_raw = quantized_entry - sl_offset
                tp_raw = quantized_entry + tp_offset
                # Quantize SL DOWN (further away), TP UP (further away) for safety/wider range
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_DOWN)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_UP)
            else:  # SELL
                sl_raw = quantized_entry + sl_offset
                tp_raw = quantized_entry - tp_offset
                # Quantize SL UP (further away), TP DOWN (further away)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_UP)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_DOWN)

            # --- Post-Calculation Validation ---
            # Validate Stop Loss
            if stop_loss is not None:
                min_dist_from_entry = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
                adjusted_sl = None
                # Check if SL is too close (within min ticks, considering potential rounding)
                if signal == "BUY" and stop_loss >= quantized_entry - min_dist_from_entry + (min_tick / Decimal("2")):
                    adjusted_sl = self.quantize_price(quantized_entry - min_dist_from_entry, rounding=ROUND_DOWN)
                elif signal == "SELL" and stop_loss <= quantized_entry + min_dist_from_entry - (
                    min_tick / Decimal("2")
                ):
                    adjusted_sl = self.quantize_price(quantized_entry + min_dist_from_entry, rounding=ROUND_UP)

                if adjusted_sl is not None:
                    self.logger.warning(
                        f"Initial SL ({stop_loss}) too close to entry ({quantized_entry}). Adjusting to {adjusted_sl}."
                    )
                    stop_loss = adjusted_sl

                # Final check for zero/negative SL
                if stop_loss is not None and stop_loss <= 0:
                    self.logger.error(f"Calculated SL is zero/negative ({stop_loss}). Setting SL to None.")
                    stop_loss = None

            # Validate Take Profit
            if take_profit is not None:
                adjusted_tp = None
                # Check if TP ended up on the wrong side of entry
                if signal == "BUY" and take_profit <= quantized_entry:
                    adjusted_tp = self.quantize_price(quantized_entry + min_tick, rounding=ROUND_UP)
                elif signal == "SELL" and take_profit >= quantized_entry:
                    adjusted_tp = self.quantize_price(quantized_entry - min_tick, rounding=ROUND_DOWN)

                if adjusted_tp is not None:
                    self.logger.debug(
                        f"Initial TP ({take_profit}) not beyond entry ({quantized_entry}). Adjusting to {adjusted_tp}."
                    )
                    take_profit = adjusted_tp

                # Final check for zero/negative TP
                if take_profit is not None and take_profit <= 0:
                    self.logger.error(f"Calculated TP is zero/negative ({take_profit}). Setting TP to None.")
                    take_profit = None

            # Log results
            prec = self.get_price_precision_digits()
            tp_log = f"{take_profit:.{prec}f}" if take_profit else "N/A"
            sl_log = f"{stop_loss:.{prec}f}" if stop_loss else "N/A"
            entry_log = f"{quantized_entry:.{prec}f}"
            self.logger.info(
                f"Calc TP/SL ({signal}): Entry={entry_log}, TP={tp_log}, SL={sl_log} (ATR={atr:.{prec + 1}f}, SLx={sl_mult}, TPx={tp_mult})"
            )

            return quantized_entry, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(
                f"{NEON_RED}Error during TP/SL calculation value conversion for {self.symbol}: {e}{RESET}"
            )
            return quantized_entry, None, None  # Return entry if valid, but no TP/SL
        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True
            )
            return quantized_entry, None, None


# --- Position Sizing ---
def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,  # Keep as float from config
    entry_price: Decimal,
    stop_loss_price: Decimal,
    market_info: dict,
    leverage: int,  # Keep as int from config
    logger: logging.Logger,
) -> Decimal | None:
    """Calculates the position size in base currency units or contracts using Decimal precision.
    Validates against market limits and available margin.

    Args:
        balance: Available balance in quote currency (Decimal).
        risk_per_trade: Fraction of balance to risk (float, e.g., 0.01 for 1%).
        entry_price: Proposed entry price (Decimal).
        stop_loss_price: Proposed stop loss price (Decimal).
        market_info: Dictionary containing market details (precision, limits, contract size).
        leverage: Leverage to be used (int).
        logger: Logger instance.

    Returns:
        Calculated and quantized position size (Decimal), or None if calculation fails or constraints are not met.
    """
    lg = logger
    symbol = market_info.get("symbol", "N/A")
    contract_size: Decimal = market_info.get("contract_size", Decimal("1"))
    min_order_amount: Decimal | None = market_info.get("min_order_amount")
    max_order_amount: Decimal | None = market_info.get("max_order_amount")
    min_order_cost: Decimal | None = market_info.get("min_order_cost")
    amount_digits: int | None = market_info.get("amount_precision_digits")
    is_contract: bool = market_info.get("is_contract", False)
    is_inverse: bool = market_info.get("inverse", False)
    quote_currency: str = market_info.get("quote", "?")

    # --- Input Validation ---
    if balance <= 0:
        lg.error(f"Size Calc Error ({symbol}): Balance <= 0")
        return None
    if not entry_price.is_finite() or entry_price <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid entry price {entry_price}")
        return None
    if not stop_loss_price.is_finite() or stop_loss_price <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid SL price {stop_loss_price}")
        return None
    if entry_price == stop_loss_price:
        lg.error(f"Size Calc Error ({symbol}): Entry price equals SL price")
        return None
    if amount_digits is None:
        lg.error(f"Size Calc Error ({symbol}): Amount precision missing from market info")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Error ({symbol}): Invalid risk_per_trade value {risk_per_trade}")
        return None
    if is_contract and leverage <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid leverage {leverage} for contract")
        return None

    try:
        # --- Calculate Risk Amount and SL Distance ---
        risk_amount_quote: Decimal = balance * Decimal(str(risk_per_trade))
        sl_distance_points: Decimal = abs(entry_price - stop_loss_price)
        if sl_distance_points <= 0:
            lg.error(f"Size Calc Error ({symbol}): SL distance points <= 0")
            return None

        # --- Calculate Risk Per Unit/Contract ---
        size_unquantized = Decimal("NaN")
        risk_per_unit_or_contract_quote = Decimal("NaN")

        if is_contract:
            if is_inverse:
                # Inverse: Risk = Contracts * ContractSize * |1/Entry - 1/SL| * ValueAtEntry (approx)
                if entry_price == 0 or stop_loss_price == 0:
                    lg.error(f"Size Calc Error ({symbol}): Zero price encountered for inverse calculation")
                    return None
                # Risk per contract in BASE currency terms
                risk_per_contract_base = contract_size * abs(
                    Decimal("1") / entry_price - Decimal("1") / stop_loss_price
                )
                # Convert risk per contract to QUOTE currency terms (approximate using entry price)
                risk_per_unit_or_contract_quote = risk_per_contract_base * entry_price
            else:  # Linear
                # Linear: Risk = Contracts * ContractSize * |Entry - SL|
                risk_per_unit_or_contract_quote = sl_distance_points * contract_size
        else:  # Spot
            # Spot: Risk = Amount * |Entry - SL|
            risk_per_unit_or_contract_quote = sl_distance_points

        # Validate risk per unit
        if not risk_per_unit_or_contract_quote.is_finite() or risk_per_unit_or_contract_quote <= 0:
            lg.error(
                f"Size Calc Error ({symbol}): Invalid calculated risk per unit/contract ({risk_per_unit_or_contract_quote})"
            )
            return None

        # --- Calculate Unquantized Size ---
        # Size = RiskAmount / RiskPerUnit
        size_unquantized = risk_amount_quote / risk_per_unit_or_contract_quote

        if not size_unquantized.is_finite() or size_unquantized <= 0:
            lg.error(f"Size Calc Error ({symbol}): Invalid unquantized size calculated ({size_unquantized})")
            return None

        lg.debug(
            f"Size Calc ({symbol}): Bal={balance:.2f}, Risk={risk_per_trade * 100:.2f}%, RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_points}, RiskPerUnit={risk_per_unit_or_contract_quote:.8f}, UnquantSize={size_unquantized:.8f}"
        )

        # --- Quantize Size ---
        # Round DOWN to be conservative with risk
        step_size = Decimal("1") / (Decimal("10") ** amount_digits)
        quantized_size = (size_unquantized / step_size).quantize(Decimal("0"), rounding=ROUND_DOWN) * step_size
        quantized_size = quantized_size.quantize(step_size)  # Ensure exact decimal places match step size

        lg.debug(f"Quantized Size ({symbol}): {quantized_size} (Step: {step_size})")

        if quantized_size <= 0:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Size is zero after quantization.{RESET}")
            return None

        # --- Validate Against Market Limits (Amount) ---
        if min_order_amount is not None and quantized_size < min_order_amount:
            lg.warning(
                f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} < Min Amount {min_order_amount}. Cannot place order.{RESET}"
            )
            return None
        if max_order_amount is not None and quantized_size > max_order_amount:
            lg.warning(
                f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} > Max Amount {max_order_amount}. Capping size.{RESET}"
            )
            # Cap the size and re-quantize
            quantized_size = max_order_amount.quantize(step_size, rounding=ROUND_DOWN)
            if quantized_size < min_order_amount if min_order_amount else Decimal("0"):  # Check again after capping
                lg.error(
                    f"Size Calc Error ({symbol}): Capped size {quantized_size} is below min amount {min_order_amount}."
                )
                return None

        # --- Validate Against Market Limits (Cost) & Margin ---
        order_value_quote = Decimal("0")
        if is_contract:
            if is_inverse:
                # Inverse value in quote = Contracts * ContractSize
                order_value_quote = quantized_size * contract_size
            else:  # Linear value in quote = Contracts * ContractSize * EntryPrice
                order_value_quote = quantized_size * contract_size * entry_price
        else:  # Spot value in quote = Amount * EntryPrice
            order_value_quote = quantized_size * entry_price

        # Check min cost
        if min_order_cost is not None and order_value_quote < min_order_cost:
            lg.warning(
                f"{NEON_YELLOW}Size Calc ({symbol}): Order value {order_value_quote:.4f} {quote_currency} < Min Cost {min_order_cost}. Cannot place order.{RESET}"
            )
            return None

        # Check margin requirement vs available balance
        margin_required = order_value_quote / Decimal(leverage) if is_contract else order_value_quote

        # Use a small buffer (e.g., 0.1% = 1.001) for fees/slippage, maybe configurable?
        buffer_factor = Decimal("1.001")
        if margin_required * buffer_factor > balance:
            lg.warning(
                f"{NEON_YELLOW}Size Calc ({symbol}): Required margin {margin_required:.4f} (incl. buffer) > Available balance {balance:.4f}. Cannot place order.{RESET}"
            )
            # TODO: Optionally, reduce size to fit available margin? Requires recalculating size based on max margin.
            # For now, just reject the trade.
            return None

        # --- Success ---
        size_unit = "contracts" if is_contract else market_info.get("base", "units")
        lg.info(f"Calculated position size for {symbol}: {quantized_size} {size_unit}")
        return quantized_size

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Error during position size calculation for {symbol}: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


# --- CCXT Trading Action Wrappers ---


def fetch_positions_ccxt(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict
) -> dict | None:
    """Fetches the current non-zero position for a specific symbol using V5 API.
    Standardizes the returned position dictionary.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The standard CCXT symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        A dictionary containing standardized position details if a non-zero position exists,
        otherwise None. Includes keys like 'symbol', 'side', 'contracts' (abs size),
        'entryPrice' (Decimal), 'info' (raw data), 'market_info'.
    """
    lg = logger
    category = market_info.get("category")
    market_id = market_info.get("id", symbol)  # Use exchange-specific ID

    if not category or category not in ["linear", "inverse"]:
        lg.debug(f"Skipping position check for non-derivative symbol: {symbol}")
        return None
    if not exchange.has.get("fetchPositions"):
        lg.error(f"Exchange {exchange.id} doesn't support fetchPositions().")
        return None

    try:
        # Bybit V5 requires category and optionally symbol/market_id
        params = {"category": category, "symbol": market_id}
        lg.debug(f"Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Pass symbols=[symbol] to CCXT for potential client-side filtering, but params drive the API call
        all_positions = safe_ccxt_call(exchange, "fetch_positions", lg, symbols=[symbol], params=params)

        if not isinstance(all_positions, list):
            lg.error(
                f"fetch_positions did not return a list for {symbol}. Type: {type(all_positions)}. Response: {all_positions}"
            )
            return None

        for pos in all_positions:
            if not isinstance(pos, dict):
                lg.warning(f"Received non-dict item in positions list: {pos}")
                continue

            # Match symbol rigorously (CCXT symbol vs position symbol)
            # CCXT usually returns the standard symbol format in 'symbol' key
            position_symbol = pos.get("symbol")
            if position_symbol != symbol:
                continue  # Skip positions for other symbols if API returned more than requested

            try:
                # Get position size ('contracts' is standard CCXT, 'size' is common in Bybit 'info')
                pos_size_str = pos.get("contracts", pos.get("info", {}).get("size"))
                if pos_size_str is None:
                    lg.debug(f"Skipping position entry for {symbol} due to missing size field. Data: {pos}")
                    continue

                pos_size = Decimal(str(pos_size_str))
                if pos_size == Decimal("0"):
                    # lg.debug(f"Skipping zero size position for {symbol}.")
                    continue  # Skip zero size positions explicitly

                # Determine side and absolute size
                pos_side = "long" if pos_size > 0 else "short"
                abs_size = abs(pos_size)

                # Standardize the position dictionary for internal use
                standardized_pos = pos.copy()  # Start with the original data
                standardized_pos["side"] = pos_side
                standardized_pos["contracts"] = abs_size  # Store absolute size as Decimal

                # Standardize entry price ('entryPrice' is CCXT, 'avgPrice' is Bybit V5 info)
                entry_price_str = standardized_pos.get("entryPrice", pos.get("info", {}).get("avgPrice"))
                if entry_price_str is not None:
                    try:
                        standardized_pos["entryPrice"] = Decimal(str(entry_price_str))
                    except InvalidOperation:
                        standardized_pos["entryPrice"] = None
                else:
                    standardized_pos["entryPrice"] = None

                standardized_pos["market_info"] = market_info  # Add market info for convenience

                entry_log = (
                    f"{standardized_pos['entryPrice']:.{market_info.get('price_precision_digits', 4)}f}"
                    if standardized_pos.get("entryPrice")
                    else "N/A"
                )
                lg.info(f"Found active {pos_side} position for {symbol}: Size={abs_size}, Entry={entry_log}")
                return standardized_pos  # Return the first non-zero matching position

            except (InvalidOperation, ValueError, TypeError) as e:
                lg.error(f"Could not parse position data for {symbol}: {e}. Data: {pos}")
            except Exception as e:
                lg.error(f"Unexpected error processing position entry for {symbol}: {e}. Data: {pos}", exc_info=True)

        # If loop completes without returning a position
        lg.debug(f"No active non-zero position found for {symbol}.")
        return None

    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
        return None


def set_leverage_ccxt(
    exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger, market_info: dict
) -> bool:
    """Sets leverage for a symbol using Bybit V5 API.

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
    category = market_info.get("category")
    market_id = market_info.get("id", symbol)  # Use exchange-specific ID

    if not category or category not in ["linear", "inverse"]:
        lg.debug(f"Skipping leverage setting for non-derivative: {symbol}")
        return True  # Success as no action needed
    if not exchange.has.get("setLeverage"):
        lg.error(f"Exchange {exchange.id} doesn't support setLeverage().")
        return False
    if leverage <= 0:
        lg.error(f"Invalid leverage ({leverage}) for {symbol}. Must be > 0.")
        return False

    try:
        # Bybit V5 requires category, symbol, and separate buy/sell leverage values
        params = {
            "category": category,
            "symbol": market_id,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        lg.info(f"Attempting to set leverage for {symbol} (MarketID: {market_id}) to {leverage}x...")
        lg.debug(f"Leverage Params: {params}")

        # CCXT's set_leverage should map to the correct V5 endpoint
        # safe_ccxt_call handles "leverage not modified" (110043) as success ({})
        result = safe_ccxt_call(exchange, "set_leverage", lg, leverage=float(leverage), symbol=symbol, params=params)

        # Check result: Success could be a dict with info, or empty dict for 'not modified'
        if result is not None and isinstance(result, dict):
            lg.info(f"{NEON_GREEN}Leverage set successfully (or already correct) for {symbol} to {leverage}x.{RESET}")
            return True
        else:
            # This case should ideally not happen if safe_ccxt_call raises or returns {} on 110043
            lg.error(f"{NEON_RED}set_leverage call returned unexpected result for {symbol}: {result}{RESET}")
            return False

    except ccxt.ExchangeError as e:
        # safe_ccxt_call should raise non-retryable errors, but catch just in case
        lg.error(
            f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=False
        )  # Don't need full trace if safe_call logged it
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
        return False


def create_order_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    order_type: str,
    side: str,
    amount: Decimal,
    price: Decimal | None = None,
    params: dict | None = None,
    logger: logging.Logger | None = None,
    market_info: dict | None = None,
) -> dict | None:
    """Creates an order using safe_ccxt_call, handling V5 params and Decimal->float/str conversion.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        order_type: 'market' or 'limit'.
        side: 'buy' or 'sell'.
        amount: Order quantity (Decimal).
        price: Order price for limit orders (Decimal).
        params: Additional parameters for the CCXT create_order call (e.g., {'reduceOnly': True}).
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        The CCXT order dictionary if successful and confirmed by retCode=0, otherwise None.
    """
    lg = logger or get_logger("main")
    if not market_info:
        lg.error(f"Market info required for create_order ({symbol})")
        return None

    category = market_info.get("category")
    market_info.get("id", symbol)  # Use exchange-specific ID
    price_digits = market_info.get("price_precision_digits", 8)
    amount_digits = market_info.get("amount_precision_digits", 8)

    # --- Input Validation ---
    if not category:
        lg.error(f"Unknown category for {symbol}. Cannot place order.")
        return None
    if amount <= 0:
        lg.error(f"Order amount must be positive ({symbol}, Amount: {amount})")
        return None
    order_type_lower = order_type.lower()
    side_lower = side.lower()
    if order_type_lower not in ["market", "limit"]:
        lg.error(f"Invalid order type '{order_type}'")
        return None
    if side_lower not in ["buy", "sell"]:
        lg.error(f"Invalid order side '{side}'")
        return None
    if order_type_lower == "limit" and (price is None or not price.is_finite() or price <= 0):
        lg.error(f"Valid positive price required for limit order ({symbol}, Price: {price})")
        return None

    # --- Format amount/price strings for logging and potential API use ---
    # CCXT generally prefers floats for amount/price args, but string formatting helps logging
    amount_str = f"{amount:.{amount_digits}f}"
    price_str = f"{price:.{price_digits}f}" if order_type_lower == "limit" and price else None

    # --- Prepare V5 Parameters ---
    # Base parameters required by Bybit V5
    order_params: dict[str, Any] = {"category": category}

    # --- Hedge Mode Logic ---
    # If config['position_mode'] == 'Hedge', determine positionIdx based on side.
    # Bybit: 0 for One-Way mode positions, 1 for Buy side hedge, 2 for Sell side hedge.
    # If hedge mode is active, 'positionIdx' MUST be passed in params.
    # If One-Way mode, 'positionIdx' should be 0 (or omitted, default is often 0).
    # Example (needs integration with config):
    # if config.get("position_mode") == "Hedge":
    #     order_params['positionIdx'] = 1 if side_lower == 'buy' else 2
    # else:
    #     order_params['positionIdx'] = 0 # Explicitly One-Way
    # --- End Hedge Mode Placeholder ---
    # For now, assume One-Way or Hedge handled by caller via `params` arg if needed.
    # Merge external params (like reduceOnly, positionIdx) provided by caller
    if params:
        order_params.update(params)

    # --- Convert Decimals to Floats for CCXT call ---
    try:
        # CCXT methods typically expect float for amount/price
        amount_float = float(amount_str)
        price_float = float(price_str) if price_str else None
    except ValueError as e:
        lg.error(f"Error converting amount/price to float ({symbol}): {e}")
        return None

    # --- Place Order via safe_ccxt_call ---
    try:
        log_price_part = f"@ {price_str}" if price_str else "at Market"
        log_param_part = f" Params: {order_params}" if order_params else ""
        lg.info(
            f"Attempting to create {side.upper()} {order_type.upper()} order: {amount_str} {symbol} {log_price_part}{log_param_part}"
        )
        lg.debug(
            f"CCXT Order Args: Symbol={symbol}, Type={order_type}, Side={side}, Amount={amount_float}, Price={price_float}, Params={order_params}"
        )

        order_result = safe_ccxt_call(
            exchange,
            "create_order",
            lg,
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=price_float,
            params=order_params,
        )

        # --- Process Result ---
        # Check if the call succeeded and returned a valid order structure with an ID
        if order_result and isinstance(order_result, dict) and order_result.get("id"):
            order_id = order_result["id"]
            # Check Bybit's V5 response code within the 'info' field for confirmation
            ret_code = order_result.get("info", {}).get("retCode")
            ret_msg = order_result.get("info", {}).get("retMsg", "Unknown Status")

            if ret_code == 0:
                lg.info(
                    f"{NEON_GREEN}Successfully created {side.upper()} {order_type.upper()} order for {symbol}. Order ID: {order_id}{RESET}"
                )
                lg.debug(f"Order Result Info: {order_result.get('info')}")
                return order_result  # Return the full order dict on success
            else:
                # Order ID might be generated even if rejected (e.g., insufficient balance)
                lg.error(
                    f"{NEON_RED}Order placement potentially failed or rejected ({symbol}). Order ID: {order_id}, Code={ret_code}, Msg='{ret_msg}'.{RESET}"
                )
                lg.debug(f"Failed Order Result Info: {order_result.get('info')}")
                return None  # Treat non-zero retCode as failure
        elif order_result:
            # Call succeeded but response format is unexpected (missing ID)
            lg.error(
                f"Order API call successful but response missing ID or invalid format ({symbol}). Response: {order_result}"
            )
            return None
        else:  # safe_ccxt_call returned None or raised an exception handled within it
            lg.error(f"Order API call failed or returned None ({symbol}) after retries.")
            return None

    except Exception as e:
        # Catch any unexpected error during the process
        lg.error(f"{NEON_RED}Failed to create order ({symbol}): {e}{RESET}", exc_info=True)
        return None


def set_protection_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_price: Decimal | None = None,  # TSL distance/offset value
    trailing_active_price: Decimal | None = None,  # TSL activation price trigger
    position_idx: int = 0,  # Required for Hedge mode (0=OneWay, 1=BuyHedge, 2=SellHedge)
    logger: logging.Logger | None = None,
    market_info: dict | None = None,
) -> bool:
    """Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL) for a position
    using Bybit V5 `POST /v5/position/trading-stop`.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        stop_loss_price: Price for the stop loss order (Decimal). Set to 0 or None to remove.
        take_profit_price: Price for the take profit order (Decimal). Set to 0 or None to remove.
        trailing_stop_price: The trailing stop distance/offset value (Decimal). Set to 0 or None to remove.
                             Bybit interprets this as the price distance (e.g., 10 for $10 below high).
        trailing_active_price: The price at which the TSL should activate (Decimal).
                               Set to 0 or None for immediate activation if trailing_stop_price is set.
        position_idx: Position index (0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge).
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        True if the protection was set successfully (retCode=0), False otherwise.
    """
    lg = logger or get_logger("main")
    if not market_info:
        lg.error(f"Market info required for set_protection ({symbol})")
        return False

    category = market_info.get("category")
    market_id = market_info.get("id", symbol)  # Use exchange-specific ID
    price_digits = market_info.get("price_precision_digits", 8)

    if not category or category not in ["linear", "inverse"]:
        lg.warning(f"Cannot set protection for non-derivative {symbol}. Category: {category}")
        return False

    # --- Prepare V5 Parameters ---
    params: dict[str, Any] = {
        "category": category,
        "symbol": market_id,
        "positionIdx": position_idx,  # Crucial for Hedge Mode, 0 for One-Way
        "tpslMode": "Full",  # Apply TP/SL to the entire position ('Partial' also available)
        # 'slOrderType': 'Market', # Default is Market, 'Limit' also possible
        # 'tpOrderType': 'Market', # Default is Market, 'Limit' also possible
    }

    # --- Format Price Values ---
    # Bybit API expects prices as strings. "0" is used to cancel/remove existing SL/TP/TSL.
    def format_price(price: Decimal | None) -> str:
        if price is not None and price.is_finite() and price > 0:
            return f"{price:.{price_digits}f}"
        else:
            return "0"  # Use "0" to signify removal or no setting

    params["stopLoss"] = format_price(stop_loss_price)
    params["takeProfit"] = format_price(take_profit_price)

    # Trailing Stop requires 'trailingStop' (distance) and optionally 'activePrice'
    # Bybit API uses 'trailingStop' for the distance value (e.g., "100" means trail $100 behind price)
    params["trailingStop"] = format_price(trailing_stop_price)
    if params["trailingStop"] != "0":
        # Only set activePrice if TSL distance is being set. "0" means immediate activation.
        params["activePrice"] = format_price(trailing_active_price)
    elif "activePrice" in params:
        # Ensure activePrice is not sent if trailingStop is "0"
        del params["activePrice"]

    # --- Log Intention ---
    log_parts = []
    if params["stopLoss"] != "0":
        log_parts.append(f"SL={params['stopLoss']}")
    if params["takeProfit"] != "0":
        log_parts.append(f"TP={params['takeProfit']}")
    if params["trailingStop"] != "0":
        tsl_log = f"TSL_Dist={params['trailingStop']}"
        if params.get("activePrice", "0") != "0":
            tsl_log += f", ActP={params['activePrice']}"
        else:
            tsl_log += ", Act=Immediate"
        log_parts.append(tsl_log)

    if not log_parts:
        lg.info(f"No valid protection levels provided for set_protection ({symbol}). No API call made.")
        # Consider it success if no action was needed
        return True

    # --- Make API Call ---
    try:
        lg.info(f"Attempting to set protection for {symbol} (Idx: {position_idx}): {', '.join(log_parts)}")
        lg.debug(f"Protection Params: {params}")

        # Explicitly use the private POST method mapped in exchange options
        # This ensures we hit the correct V5 endpoint: /v5/position/trading-stop
        method_to_call = "private_post_position_trading_stop"
        if not hasattr(exchange, method_to_call):
            # Fallback if explicit mapping isn't present (less ideal)
            lg.warning(
                f"Method '{method_to_call}' not found directly on exchange object, attempting generic privatePost."
            )
            # Construct the path manually (prone to errors if CCXT changes internal structure)
            # path = exchange.impl['private']['post']['position/trading-stop'] # Example, might not work
            # result = exchange.privatePost(path, params) # Needs correct path construction
            # Safer to rely on the mapping in initialize_exchange
            lg.error(f"Cannot call {method_to_call}. Ensure it's mapped in exchange options.")
            return False

        result = safe_ccxt_call(exchange, method_to_call, lg, params=params)

        # --- Process Result ---
        # Check Bybit's V5 response code
        if result and isinstance(result, dict) and result.get("retCode") == 0:
            lg.info(f"{NEON_GREEN}Successfully set protection for {symbol}.{RESET}")
            lg.debug(f"Protection Result Info: {result}")
            return True
        elif result:
            # API call returned, but retCode indicates failure
            ret_code = result.get("retCode", -1)
            ret_msg = result.get("retMsg", "Unknown Error")
            lg.error(f"{NEON_RED}Failed to set protection ({symbol}). Code={ret_code}, Msg='{ret_msg}'{RESET}")
            lg.debug(f"Protection Failure Info: {result}")
            # Check for common errors
            if ret_code == 170140:
                lg.warning(f"Hint: Set protection failed because position might not exist (Code {ret_code}).")
            elif ret_code == 110025:
                lg.warning(
                    f"Hint: Set protection failed due to positionIdx mismatch (Code {ret_code}). Check Hedge Mode settings."
                )
            return False
        else:  # safe_ccxt_call returned None (likely hit max retries)
            lg.error(f"Set protection API call failed or returned None ({symbol}) after retries.")
            return False

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
        return False


def close_position_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    position_data: dict,
    logger: logging.Logger | None = None,
    market_info: dict | None = None,
) -> dict | None:
    """Closes an existing position via a Market order with 'reduceOnly' flag.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Standard CCXT symbol.
        position_data: The standardized position dictionary obtained from fetch_positions_ccxt.
                       Must contain 'side' ('long'/'short') and 'contracts' (Decimal size).
                       Should also contain 'info' -> 'positionIdx' if applicable.
        logger: Logger instance.
        market_info: Market information dictionary.

    Returns:
        The CCXT order dictionary for the closing order if placed successfully, otherwise None.
    """
    lg = logger or get_logger("main")
    if not market_info:
        lg.error(f"Market info required for close_position ({symbol})")
        return None
    if not position_data:
        lg.error(f"Position data required for close_position ({symbol})")
        return None

    try:
        position_side = position_data.get("side")  # 'long' or 'short'
        position_size_dec = position_data.get("contracts")  # Absolute Decimal size

        # Validate position data
        if position_side not in ["long", "short"]:
            lg.error(f"Invalid side ('{position_side}') in position data for closing {symbol}")
            return None
        if not isinstance(position_size_dec, Decimal) or position_size_dec <= 0:
            lg.error(f"Invalid size ('{position_size_dec}') in position data for closing {symbol}")
            return None

        # Determine the side of the closing order
        close_side = "sell" if position_side == "long" else "buy"
        amount_to_close = position_size_dec  # Use the absolute size from position data

        lg.info(
            f"Attempting to close {position_side} position ({symbol}, Size: {amount_to_close}) via {close_side.upper()} MARKET order..."
        )

        # --- Prepare Parameters for Closing Order ---
        # 'reduceOnly': True ensures the order only closes or reduces the position, never opens/increases.
        close_params: dict[str, Any] = {"reduceOnly": True}

        # --- Hedge Mode Logic ---
        # If hedging, need to specify which position to close using positionIdx.
        # Get positionIdx from the 'info' field of the fetched position data.
        pos_idx_to_close = position_data.get("info", {}).get("positionIdx")
        # positionIdx is typically 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge
        if pos_idx_to_close is not None:
            close_params["positionIdx"] = pos_idx_to_close
            lg.debug(f"Setting positionIdx={pos_idx_to_close} for closing order ({symbol}).")
        else:
            # If positionIdx is missing from position_data, default might be risky in Hedge Mode.
            # Assuming One-Way (0) if missing, but log a warning if Hedge Mode might be active.
            if config.get("position_mode") == "Hedge":
                lg.warning(
                    f"Position Index missing from position data for {symbol}, but Hedge Mode may be active. Defaulting close order to positionIdx=0. This might fail if not One-Way mode."
                )
            close_params["positionIdx"] = 0  # Default to 0 (One-Way) if not found
        # --- End Hedge Mode Placeholder ---

        # Use create_order_ccxt to place the closing market order
        close_order_result = create_order_ccxt(
            exchange=exchange,
            symbol=symbol,
            order_type="market",
            side=close_side,
            amount=amount_to_close,
            params=close_params,
            logger=lg,
            market_info=market_info,
        )

        # Check if the closing order was placed successfully
        if close_order_result and close_order_result.get("id"):
            lg.info(
                f"{NEON_GREEN}Successfully placed MARKET order to close {position_side} position ({symbol}). Close Order ID: {close_order_result.get('id')}{RESET}"
            )
            return close_order_result
        else:
            lg.error(f"{NEON_RED}Failed to place market order to close position ({symbol}).{RESET}")
            # Log specific reason if available (e.g., from safe_ccxt_call inside create_order_ccxt)
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Error processing position data for closing ({symbol}): {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error attempting to close position ({symbol}): {e}{RESET}", exc_info=True)
        return None


# --- Main Bot Logic ---
def run_bot(exchange: ccxt.Exchange, config: dict[str, Any], bot_state: dict[str, Any]) -> None:
    """Main execution loop of the trading bot."""
    main_logger = get_logger("main")
    main_logger.info(f"{NEON_CYAN}=== Starting Enhanced Trading Bot v{BOT_VERSION} (PID: {os.getpid()}) ===")

    # --- Log Initial Bot Configuration ---
    trading_status = f"{NEON_GREEN}Enabled{RESET}" if config.get("enable_trading") else f"{NEON_YELLOW}DISABLED{RESET}"
    sandbox_status = (
        f"{NEON_YELLOW}ACTIVE{RESET}" if config.get("use_sandbox") else f"{NEON_RED}INACTIVE (LIVE!){RESET}"
    )
    account_type = "UNIFIED" if IS_UNIFIED_ACCOUNT else "Non-UTA (Contract/Spot)"
    main_logger.info(f"Mode: Trading={trading_status}, Sandbox={sandbox_status}, Account={account_type}")
    main_logger.info(
        f"Config: Symbols={config.get('symbols')}, Interval={config.get('interval')}, Quote={QUOTE_CURRENCY}, WeightSet='{config.get('active_weight_set')}'"
    )
    main_logger.info(
        f"Risk: {config.get('risk_per_trade') * 100:.2f}%, Leverage={config.get('leverage')}x, MaxPos={config.get('max_concurrent_positions_total')}"
    )
    main_logger.info(
        f"Features: TSL={'On' if config.get('enable_trailing_stop') else 'Off'}, BE={'On' if config.get('enable_break_even') else 'Off'}, MACrossExit={'On' if config.get('enable_ma_cross_exit') else 'Off'}"
    )
    main_logger.info(
        f"Position Mode: {config.get('position_mode', 'One-Way')} (Note: Hedge mode requires careful param setting)"
    )

    global LOOP_DELAY_SECONDS
    LOOP_DELAY_SECONDS = int(config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS))

    symbols_to_trade: list[str] = config.get("symbols", [])

    # --- Initialize/Validate state dictionary for each symbol ---
    for symbol in symbols_to_trade:
        if symbol not in bot_state:
            bot_state[symbol] = {}
        # Ensure default states exist
        bot_state[symbol].setdefault("break_even_triggered", False)
        bot_state[symbol].setdefault("last_signal", "HOLD")
        bot_state[symbol].setdefault("last_entry_price", None)  # Stored as string, represents last known entry

    cycle_count = 0
    last_market_reload_time = getattr(exchange, "last_load_markets_timestamp", 0)

    # --- Main Loop ---
    while True:
        cycle_count += 1
        start_time = time.time()
        main_logger.info(f"{NEON_BLUE}--- Starting Bot Cycle {cycle_count} ---{RESET}")

        # --- Periodic Market Reload ---
        if time.time() - last_market_reload_time > MARKET_RELOAD_INTERVAL_SECONDS:
            main_logger.info(f"Reloading exchange markets (Interval: {MARKET_RELOAD_INTERVAL_SECONDS}s)...")
            try:
                exchange.load_markets(True)  # Force reload
                last_market_reload_time = time.time()
                exchange.last_load_markets_timestamp = (
                    last_market_reload_time  # Update timestamp on exchange object too
                )
                main_logger.info("Markets reloaded successfully.")
            except Exception as e:
                main_logger.error(f"Failed to reload markets: {e}", exc_info=True)
                # Continue loop, but market info might be stale

        # --- Pre-Cycle Checks (Balance, Positions) ---
        current_balance: Decimal | None = None
        if config.get("enable_trading"):
            try:
                current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
                if current_balance is None:
                    main_logger.error(
                        f"{NEON_RED}Failed to fetch {QUOTE_CURRENCY} balance. Trading actions may fail or use stale data.{RESET}"
                    )
                elif current_balance <= 0:
                    main_logger.warning(
                        f"{NEON_YELLOW}Available {QUOTE_CURRENCY} balance is {current_balance:.4f}. Cannot open new positions.{RESET}"
                    )
                else:
                    main_logger.info(f"Available {QUOTE_CURRENCY} balance: {current_balance:.4f}")
            except Exception as e:
                main_logger.error(f"Error fetching balance: {e}", exc_info=True)

        # Fetch all active positions for configured symbols at the start of the cycle
        open_positions_count = 0
        active_positions: dict[str, dict] = {}  # Stores standardized position data {symbol: position_dict}
        main_logger.debug("Fetching active positions for configured symbols...")
        for symbol in symbols_to_trade:
            temp_logger = get_logger(symbol, is_symbol_logger=True)  # Use symbol-specific logger for fetch
            try:
                market_info = get_market_info(exchange, symbol, temp_logger)
                if not market_info:
                    temp_logger.warning(f"Skipping position check for {symbol}: Could not get market info.")
                    continue
                if market_info.get("is_contract"):  # Only fetch positions for derivatives
                    position = fetch_positions_ccxt(exchange, symbol, temp_logger, market_info)
                    if position:
                        open_positions_count += 1
                        active_positions[symbol] = position
                        # Update state's entry price if missing and position exists
                        # Use the entry price from the fetched position as the source of truth if state is missing
                        entry_p_state = bot_state[symbol].get("last_entry_price")
                        entry_p_pos = position.get("entryPrice")
                        if entry_p_state is None and isinstance(entry_p_pos, Decimal):
                            try:
                                bot_state[symbol]["last_entry_price"] = str(entry_p_pos)
                                temp_logger.info(
                                    f"Updated state's last_entry_price from fetched position: {entry_p_pos}"
                                )
                            except Exception as e:
                                temp_logger.warning(f"Could not update state entry price: {e}")
            except Exception as fetch_pos_err:
                temp_logger.error(f"Error during position pre-check for {symbol}: {fetch_pos_err}", exc_info=True)

        max_allowed_positions = int(config.get("max_concurrent_positions_total", 1))
        main_logger.info(f"Currently open positions: {open_positions_count} / {max_allowed_positions}")

        # --- Symbol Processing Loop ---
        for symbol in symbols_to_trade:
            symbol_logger = get_logger(symbol, is_symbol_logger=True)
            symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
            symbol_state = bot_state[symbol]  # Get reference to this symbol's state dict

            try:
                # Get Market Info (potentially reloaded)
                market_info = get_market_info(exchange, symbol, symbol_logger)
                if not market_info:
                    symbol_logger.error(f"Could not get market info for {symbol}. Skipping cycle for this symbol.")
                    continue

                # Fetch Latest Data (Klines, Price, Orderbook)
                timeframe = config.get("interval", "5")
                # Determine required kline limit based on longest indicator period + buffer
                periods = [
                    int(v)
                    for k, v in config.items()
                    if ("_period" in k or "_window" in k) and isinstance(v, (int, float)) and v > 0
                ]
                base_limit = max(periods) if periods else 100
                kline_limit = base_limit + 50  # Add buffer for calculation stability
                symbol_logger.debug(f"Required kline limit: {kline_limit} (Base: {base_limit})")

                df_raw = fetch_klines_ccxt(exchange, symbol, timeframe, kline_limit, symbol_logger, market_info)
                if df_raw.empty or len(df_raw) < base_limit // 2:  # Check if significantly less than needed
                    symbol_logger.warning(
                        f"Kline data insufficient ({len(df_raw)} rows, need ~{base_limit}). Skipping analysis for {symbol}."
                    )
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

                # --- Initialize Analyzer (Calculates indicators) ---
                try:
                    # Pass the mutable symbol_state dictionary
                    analyzer = TradingAnalyzer(df_raw, symbol_logger, config, market_info, symbol_state)
                except ValueError as analyze_init_err:
                    symbol_logger.error(
                        f"Analyzer initialization failed: {analyze_init_err}. Skipping analysis for {symbol}."
                    )
                    continue
                except Exception as analyze_err:
                    symbol_logger.error(f"Unexpected analyzer initialization error: {analyze_err}", exc_info=True)
                    continue

                # --- Manage Existing Position ---
                current_position = active_positions.get(symbol)
                position_closed_in_manage = False
                if current_position:
                    pos_side = current_position.get("side", "?")
                    pos_size = current_position.get("contracts", "?")
                    entry_p_log = (
                        f"{current_position.get('entryPrice'):.{market_info.get('price_precision_digits', 4)}f}"
                        if current_position.get("entryPrice")
                        else "N/A"
                    )
                    symbol_logger.info(
                        f"Managing existing {pos_side} position (Size: {pos_size}, Entry: {entry_p_log})."
                    )
                    position_closed_in_manage = manage_existing_position(
                        exchange, config, symbol_logger, analyzer, current_position, current_price_dec
                    )
                    if position_closed_in_manage:
                        active_positions.pop(symbol, None)  # Remove from active list
                        open_positions_count = max(0, open_positions_count - 1)  # Decrement count
                        symbol_logger.info(f"Position for {symbol} closed during management routine.")
                        # Skip to next symbol if position was closed
                        symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
                        time.sleep(0.2)
                        continue

                # --- Check for New Entry ---
                # Conditions: No active position for this symbol AND total positions < limit AND it's a contract market
                if not active_positions.get(symbol) and market_info.get("is_contract"):
                    if open_positions_count < max_allowed_positions:
                        symbol_logger.info("No active position. Checking for new entry signals...")
                        # Reset state flags if no position exists
                        if analyzer.break_even_triggered:
                            analyzer.break_even_triggered = False
                        if symbol_state.get("last_entry_price") is not None:
                            symbol_state["last_entry_price"] = None

                        # Generate signal based on latest data
                        signal = analyzer.generate_trading_signal(current_price_dec, orderbook)
                        symbol_state["last_signal"] = signal  # Store latest signal regardless of entry

                        # Attempt entry if BUY/SELL signal and trading enabled
                        if signal in ["BUY", "SELL"]:
                            if config.get("enable_trading"):
                                if current_balance is not None and current_balance > 0:
                                    opened_new = attempt_new_entry(
                                        exchange,
                                        config,
                                        symbol_logger,
                                        analyzer,
                                        signal,
                                        current_price_dec,
                                        current_balance,
                                    )
                                    if opened_new:
                                        open_positions_count += 1  # Increment count if entry successful
                                        # Re-fetch positions immediately? Optional, or wait for next cycle.
                                else:
                                    symbol_logger.warning(
                                        f"Trading enabled but balance unavailable or zero ({current_balance}). Cannot enter {signal} trade for {symbol}."
                                    )
                            else:
                                symbol_logger.info(
                                    f"Entry signal '{signal}' generated for {symbol}, but trading is disabled."
                                )
                    else:
                        # Max positions reached
                        symbol_logger.info(
                            f"Max positions ({open_positions_count}/{max_allowed_positions}) reached. Skipping new entry check for {symbol}."
                        )
                elif not market_info.get("is_contract"):
                    symbol_logger.debug(f"Symbol {symbol} is not a contract. Skipping position management/entry logic.")
                # Else (position still exists after management check) -> do nothing, wait for next cycle

            except Exception as e:
                symbol_logger.error(
                    f"{NEON_RED}!!! Unhandled error in loop for symbol {symbol}: {e} !!!{RESET}", exc_info=True
                )
            finally:
                symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
                time.sleep(0.2)  # Small delay between processing each symbol

        # --- Post-Cycle ---
        end_time = time.time()
        cycle_duration = end_time - start_time
        main_logger.info(
            f"{NEON_BLUE}--- Bot Cycle {cycle_count} Finished (Duration: {cycle_duration:.2f}s) ---{RESET}"
        )

        # Save state after each full cycle
        save_state(STATE_FILE, bot_state, main_logger)

        # Calculate wait time for next cycle
        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0:
            main_logger.info(f"Waiting {wait_time:.2f}s for next cycle...")
            time.sleep(wait_time)
        else:
            main_logger.warning(
                f"Cycle duration ({cycle_duration:.2f}s) exceeded loop delay ({LOOP_DELAY_SECONDS}s). Starting next cycle immediately."
            )


def manage_existing_position(
    exchange: ccxt.Exchange,
    config: dict[str, Any],
    logger: logging.Logger,
    analyzer: TradingAnalyzer,
    position_data: dict,
    current_price_dec: Decimal,
) -> bool:
    """Manages an existing position:
    1. Checks for MA Cross Exit signal.
    2. Checks for Break-Even trigger and updates SL if met.
    (Note: TSL is handled by the exchange after being set, this function mainly checks for BE and MA cross).

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance for the symbol.
        analyzer: TradingAnalyzer instance with current data.
        position_data: Standardized dictionary of the current position.
        current_price_dec: Current market price (Decimal).

    Returns:
        bool: True if the position was closed during this management check, False otherwise.
    """
    symbol = position_data.get("symbol")
    position_side = position_data.get("side")
    entry_price = position_data.get("entryPrice")  # Should be Decimal
    pos_size = position_data.get("contracts")  # Should be Decimal
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state  # Access shared state via analyzer

    # --- Validate Inputs ---
    if (
        not all([symbol, position_side, isinstance(entry_price, Decimal), isinstance(pos_size, Decimal)])
        or pos_size <= 0
    ):
        logger.error(f"Invalid position data received for management: {position_data}")
        return False
    if not current_price_dec.is_finite() or current_price_dec <= 0:
        logger.warning(f"Invalid current price ({current_price_dec}) for managing {symbol}")
        return False

    position_closed = False

    try:
        # --- 1. Check MA Cross Exit ---
        if config.get("enable_ma_cross_exit"):
            ema_s_f = analyzer._get_indicator_float("EMA_Short")
            ema_l_f = analyzer._get_indicator_float("EMA_Long")
            if ema_s_f is not None and ema_l_f is not None:
                # Check for adverse cross (short below long for LONG, short above long for SHORT)
                # Add a small tolerance to avoid flapping on near-equal EMAs
                tolerance = 0.0001  # e.g., 0.01% difference
                is_adverse_cross = False
                if (
                    position_side == "long"
                    and ema_s_f < ema_l_f * (1 - tolerance)
                    or position_side == "short"
                    and ema_s_f > ema_l_f * (1 + tolerance)
                ):
                    is_adverse_cross = True

                if is_adverse_cross:
                    logger.warning(
                        f"{NEON_YELLOW}MA Cross Exit Triggered for {position_side} {symbol}! Attempting close.{RESET}"
                    )
                    if config.get("enable_trading"):
                        close_result = close_position_ccxt(exchange, symbol, position_data, logger, market_info)
                        if close_result:
                            logger.info(f"Position closed successfully via MA Cross for {symbol}.")
                            # Reset state relevant to the closed position
                            analyzer.break_even_triggered = False  # Reset BE flag in state
                            symbol_state["last_signal"] = "HOLD"
                            symbol_state["last_entry_price"] = None
                            position_closed = True
                            return True  # Exit management, position is closed
                        else:
                            logger.error(f"Failed to place MA Cross close order for {symbol}. Position remains open.")
                            return False  # Retry next cycle
                    else:
                        logger.info(f"MA Cross exit triggered for {symbol}, but trading is disabled.")

        # --- 2. Check Break-Even (Only if not already triggered and position wasn't just closed) ---
        if not position_closed and config.get("enable_break_even") and not analyzer.break_even_triggered:
            atr_val = analyzer.indicator_values.get("ATR")
            if atr_val and atr_val.is_finite() and atr_val > 0:
                try:
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    profit_target_points = atr_val * trigger_multiple
                    current_profit_points = Decimal("0")
                    if position_side == "long":
                        current_profit_points = current_price_dec - entry_price
                    else:
                        current_profit_points = entry_price - current_price_dec

                    # Check if profit meets or exceeds the trigger level
                    if current_profit_points >= profit_target_points:
                        logger.info(
                            f"{NEON_GREEN}Break-Even Trigger Met for {symbol}! (Profit: {current_profit_points:.4f} >= Target: {profit_target_points:.4f}){RESET}"
                        )
                        min_tick = analyzer.get_min_tick_size()
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))

                        if min_tick and offset_ticks >= 0:
                            offset_value = min_tick * Decimal(offset_ticks)
                            # Calculate BE stop price: Entry + Offset for Long, Entry - Offset for Short
                            be_stop_price_raw = (
                                entry_price + offset_value if position_side == "long" else entry_price - offset_value
                            )
                            # Quantize BE stop price safely away from entry
                            rounding_mode = (
                                ROUND_DOWN if position_side == "long" else ROUND_UP
                            )  # Round towards safety zone
                            be_stop_price = analyzer.quantize_price(be_stop_price_raw, rounding=rounding_mode)

                            # Sanity check BE price to ensure it's actually beyond entry after quantization
                            if be_stop_price:
                                if position_side == "long" and be_stop_price <= entry_price:
                                    be_stop_price = analyzer.quantize_price(
                                        entry_price + min_tick, ROUND_UP
                                    )  # Move at least one tick beyond
                                elif position_side == "short" and be_stop_price >= entry_price:
                                    be_stop_price = analyzer.quantize_price(
                                        entry_price - min_tick, ROUND_DOWN
                                    )  # Move at least one tick beyond

                            if be_stop_price and be_stop_price.is_finite() and be_stop_price > 0:
                                logger.info(f"Calculated Break-Even Stop Price for {symbol}: {be_stop_price}")
                                if config.get("enable_trading"):
                                    # Get current TP/TSL state from position info to preserve them if needed
                                    pos_info = position_data.get("info", {})
                                    tp_str = pos_info.get("takeProfit", "0")
                                    tsl_dist_str = pos_info.get("trailingStop", "0")
                                    tsl_act_str = pos_info.get("activePrice", "0")
                                    current_tp = Decimal(tp_str) if tp_str != "0" else None
                                    current_tsl_dist = Decimal(tsl_dist_str) if tsl_dist_str != "0" else None
                                    current_tsl_act = Decimal(tsl_act_str) if tsl_act_str != "0" else None

                                    # Determine if TSL should remain active based on config
                                    force_fixed_sl = config.get("break_even_force_fixed_sl", True)
                                    use_tsl = config.get("enable_trailing_stop") and not force_fixed_sl

                                    # Parameters for set_protection: New SL, keep existing TP, maybe keep TSL
                                    tp_to_set = current_tp
                                    tsl_to_set = current_tsl_dist if use_tsl and current_tsl_dist else None
                                    act_to_set = current_tsl_act if use_tsl and tsl_to_set else None
                                    pos_idx = pos_info.get("positionIdx", 0)  # Use index from current position

                                    logger.info(
                                        f"Setting BE SL={be_stop_price}, keeping TP={tp_to_set}, TSL={'Active' if tsl_to_set else 'Inactive'}"
                                    )
                                    success = set_protection_ccxt(
                                        exchange,
                                        symbol,
                                        stop_loss_price=be_stop_price,
                                        take_profit_price=tp_to_set,
                                        trailing_stop_price=tsl_to_set,
                                        trailing_active_price=act_to_set,
                                        position_idx=pos_idx,
                                        logger=logger,
                                        market_info=market_info,
                                    )
                                    if success:
                                        logger.info(f"{NEON_GREEN}Successfully set Break-Even SL for {symbol}.{RESET}")
                                        analyzer.break_even_triggered = True  # Update state via property setter
                                    else:
                                        logger.error(
                                            f"{NEON_RED}Failed to set Break-Even SL via API for {symbol}. Will retry next cycle.{RESET}"
                                        )
                                else:
                                    logger.info(
                                        f"Break-Even triggered for {symbol}, but trading disabled. State updated."
                                    )
                                    analyzer.break_even_triggered = True  # Update state even if not trading
                            else:
                                logger.error(
                                    f"Invalid Break-Even stop price calculated ({be_stop_price}). Cannot set BE SL."
                                )
                        else:
                            logger.error(
                                f"Cannot calculate BE offset for {symbol}: Invalid min_tick ({min_tick}) or offset_ticks ({offset_ticks})."
                            )
                except (InvalidOperation, ValueError, TypeError) as be_calc_err:
                    logger.error(f"Error during Break-Even calculation for {symbol}: {be_calc_err}")
                except Exception as be_err:
                    logger.error(f"Unexpected error during Break-Even check for {symbol}: {be_err}", exc_info=True)
            else:
                logger.warning(f"Cannot check Break-Even trigger for {symbol}: Invalid ATR ({atr_val}).")

        # --- Other Management Logic (e.g., TSL adjustments - currently handled by exchange) ---
        # If TSL is active, the exchange manages it. We might add logic here later
        # to monitor TSL status or make manual adjustments if needed.

    except Exception as e:
        logger.error(f"Unexpected error managing position {symbol}: {e}", exc_info=True)
        return False  # Return False as we don't know if position was closed

    return position_closed  # Return whether the MA cross exit closed the position


def attempt_new_entry(
    exchange: ccxt.Exchange,
    config: dict[str, Any],
    logger: logging.Logger,
    analyzer: TradingAnalyzer,
    signal: str,
    entry_price_signal: Decimal,
    current_balance: Decimal,
) -> bool:
    """Attempts to enter a new trade based on a signal.
    Full workflow: Calc TP/SL -> Calc Size -> Set Leverage -> Place Order -> Confirm -> Set Protection.

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance for the symbol.
        analyzer: TradingAnalyzer instance with current data.
        signal: "BUY" or "SELL".
        entry_price_signal: The price near which the signal occurred (e.g., current price).
        current_balance: Current available balance (Decimal).

    Returns:
        bool: True if the entry was successful (order placed and protection set), False otherwise.
    """
    symbol = analyzer.symbol
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state  # Access shared state via analyzer

    logger.info(
        f"Attempting {signal} entry for {symbol} near {entry_price_signal:.{analyzer.get_price_precision_digits()}f}"
    )

    # --- 1. Calculate TP/SL ---
    quantized_entry, take_profit_price, stop_loss_price = analyzer.calculate_entry_tp_sl(entry_price_signal, signal)
    if not quantized_entry or not stop_loss_price:
        logger.error(f"Cannot enter {signal} ({symbol}): Failed to calculate valid Entry/SL price. Aborting entry.")
        return False
    # Note: TP is optional, but SL is required for size calculation.

    # --- 2. Calculate Position Size ---
    risk = float(config.get("risk_per_trade", 0.01))
    leverage = int(config.get("leverage", 10))
    position_size = calculate_position_size(
        current_balance, risk, quantized_entry, stop_loss_price, market_info, leverage, logger
    )
    if not position_size or position_size <= 0:
        logger.error(
            f"Cannot enter {signal} ({symbol}): Position size calculation failed or resulted in zero/negative size ({position_size}). Aborting entry."
        )
        return False

    # --- 3. Set Leverage (Contracts only) ---
    if market_info.get("is_contract"):
        if not set_leverage_ccxt(exchange, symbol, leverage, logger, market_info):
            logger.error(f"Failed to set leverage {leverage}x for {symbol}. Aborting entry.")
            return False

    # --- 4. Place Entry Order (Market Order) ---
    side = "buy" if signal == "BUY" else "sell"
    entry_order_params: dict[str, Any] = {}
    # --- Hedge Mode Logic ---
    # Add positionIdx if Hedge Mode is configured
    # if config.get("position_mode") == "Hedge":
    #     entry_order_params['positionIdx'] = 1 if side == 'buy' else 2
    # else:
    #     entry_order_params['positionIdx'] = 0 # Explicitly One-Way
    # --- End Hedge Mode Placeholder ---
    # Assuming One-Way for now unless positionIdx is passed via external logic

    entry_order = create_order_ccxt(
        exchange,
        symbol,
        "market",
        side,
        position_size,
        params=entry_order_params,
        logger=logger,
        market_info=market_info,
    )
    if not entry_order or not entry_order.get("id"):
        logger.error(f"Failed to place entry market order for {signal} {symbol}. Aborting entry.")
        return False
    order_id = entry_order["id"]

    # --- 5. Confirm Entry Price & Position (Wait and Fetch) ---
    logger.info(
        f"Entry market order ({order_id}) placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for fill confirmation..."
    )
    time.sleep(POSITION_CONFIRM_DELAY)

    actual_entry_price = quantized_entry  # Default to calculated entry if fetch fails
    filled_size = position_size  # Default to ordered size
    position_idx_filled = entry_order_params.get("positionIdx", 0)  # Default index used
    try:
        logger.debug(f"Re-fetching position for {symbol} to confirm entry details...")
        time.sleep(1.5)  # Extra delay before fetching position after order placement
        updated_position = fetch_positions_ccxt(exchange, symbol, logger, market_info)
        if updated_position and updated_position.get("entryPrice"):
            entry_p_fetched = updated_position.get("entryPrice")  # Should be Decimal
            current_size_fetched = updated_position.get("contracts")  # Should be Decimal
            if isinstance(entry_p_fetched, Decimal) and entry_p_fetched.is_finite() and entry_p_fetched > 0:
                actual_entry_price = entry_p_fetched
            else:
                logger.warning(
                    f"Fetched position for {symbol} has invalid entry price ({entry_p_fetched}). Using calculated: {actual_entry_price}."
                )
            if (
                isinstance(current_size_fetched, Decimal)
                and current_size_fetched.is_finite()
                and current_size_fetched > 0
            ):
                filled_size = abs(current_size_fetched)  # Use absolute size
            else:
                logger.warning(
                    f"Fetched position for {symbol} has invalid size ({current_size_fetched}). Using ordered: {filled_size}."
                )

            # Get the actual position index from the confirmed position
            position_idx_filled = updated_position.get("info", {}).get(
                "positionIdx", position_idx_filled
            )  # Use fetched index if available

            logger.info(
                f"Position Confirmed ({symbol}): Actual Entry={actual_entry_price:.{analyzer.get_price_precision_digits()}f}, Filled Size={filled_size}, PosIdx={position_idx_filled}"
            )
            # Check for significant difference between ordered and filled size
            if abs(filled_size - position_size) / position_size > Decimal("0.05"):  # More than 5% difference
                logger.warning(f"Filled size {filled_size} differs >5% from ordered {position_size} for {symbol}.")
                # Optional: Adjust TP/SL based on actual filled size? Complex. For now, proceed with original TP/SL.
        else:
            logger.warning(
                f"Could not fetch or confirm position details for {symbol} after entry order {order_id}. Using calculated entry price for protection setup."
            )
    except Exception as confirm_err:
        logger.error(
            f"Error confirming entry for {symbol}: {confirm_err}. Using calculated entry price.", exc_info=True
        )

    # --- 6. Calculate and Set Protection (SL/TP/TSL) using actual_entry_price ---
    # Recalculate SL/TP based on actual entry price? Optional, could lead to SL being hit immediately if slippage was bad.
    # Sticking with original SL/TP calculated from signal price for simplicity now.
    # TODO: Revisit if recalculating SL/TP based on actual_entry_price is better.

    # Calculate TSL parameters if enabled
    tsl_distance: Decimal | None = None
    tsl_activation_price: Decimal | None = None
    if config.get("enable_trailing_stop"):
        try:
            cb_rate = Decimal(str(config.get("trailing_stop_callback_rate")))  # Distance as % of price
            act_perc = Decimal(str(config.get("trailing_stop_activation_percentage")))  # Activation profit %
            min_tick = analyzer.get_min_tick_size()

            if cb_rate > 0 and min_tick:
                # Calculate TSL distance in price points
                tsl_dist_raw = actual_entry_price * cb_rate
                # Quantize distance UP to nearest tick or multiple ticks (more conservative)
                tsl_distance = (tsl_dist_raw / min_tick).quantize(Decimal("1"), rounding=ROUND_UP) * min_tick
                tsl_distance = max(tsl_distance, min_tick)  # Ensure at least 1 tick distance

                # Calculate activation price if percentage is set
                if act_perc > 0:
                    profit_offset = actual_entry_price * act_perc
                    act_raw = (
                        actual_entry_price + profit_offset if signal == "BUY" else actual_entry_price - profit_offset
                    )
                    # Quantize activation price away from entry
                    rounding = ROUND_UP if signal == "BUY" else ROUND_DOWN
                    tsl_activation_price = analyzer.quantize_price(act_raw, rounding=rounding)

                    # Validate activation price is actually beyond entry after quantization
                    if tsl_activation_price and tsl_activation_price > 0:
                        if (signal == "BUY" and tsl_activation_price <= actual_entry_price) or (
                            signal == "SELL" and tsl_activation_price >= actual_entry_price
                        ):
                            # Adjust to be at least one tick away if quantization put it at/before entry
                            adj_act_price = actual_entry_price + (min_tick if signal == "BUY" else -min_tick)
                            tsl_activation_price = analyzer.quantize_price(adj_act_price, rounding)
                            logger.debug(
                                f"Adjusted TSL activation price for {symbol} to be beyond entry: {tsl_activation_price}"
                            )
                    else:
                        tsl_activation_price = None  # Invalid activation price calculated
                        logger.warning(
                            f"Could not calculate valid TSL activation price for {symbol}. TSL will activate immediately if set."
                        )
                else:
                    # act_perc is 0, so immediate activation (Bybit default if activePrice="0")
                    tsl_activation_price = None  # Use None to signal immediate activation (API uses "0")

                logger.debug(
                    f"TSL Params Calculated for {symbol}: Dist={tsl_distance}, ActP={tsl_activation_price if tsl_activation_price else 'Immediate'}"
                )
            elif not min_tick:
                logger.warning(f"Cannot calc TSL distance for {symbol}: Min tick size unavailable.")
        except (InvalidOperation, ValueError, TypeError) as tsl_calc_err:
            logger.error(f"Error calculating TSL parameters for {symbol}: {tsl_calc_err}", exc_info=True)
        except Exception as tsl_err:
            logger.error(f"Unexpected error calculating TSL params for {symbol}: {tsl_err}", exc_info=True)

    # Set SL, TP, and TSL using the confirmed position index
    protection_set = set_protection_ccxt(
        exchange,
        symbol,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        trailing_stop_price=tsl_distance,
        trailing_active_price=tsl_activation_price,
        position_idx=position_idx_filled,  # Use the index from the confirmed position
        logger=logger,
        market_info=market_info,
    )

    # --- 7. Final Check and State Update ---
    if not protection_set:
        logger.error(
            f"{NEON_RED}CRITICAL: Failed to set initial protection (SL/TP/TSL) for {symbol} after entry order {order_id}! Position may be unprotected.{RESET}"
        )
        # --- Attempt Emergency Close ---
        # If protection fails, the position is open but potentially without SL/TP. Close it immediately.
        if config.get("enable_trading"):
            logger.warning(f"Attempting emergency market close of unprotected position {symbol}...")
            # Re-fetch the latest position data just before closing
            pos_to_close = fetch_positions_ccxt(exchange, symbol, logger, market_info)
            if pos_to_close:
                close_result = close_position_ccxt(exchange, symbol, pos_to_close, logger, market_info)
                if close_result:
                    logger.info(f"Emergency close order placed successfully for {symbol}.")
                else:
                    logger.error(f"{NEON_RED}EMERGENCY CLOSE FAILED for {symbol}! Manual intervention required!{RESET}")
            else:
                logger.error(
                    f"{NEON_RED}Could not fetch position for emergency close of {symbol}. Manual intervention required!{RESET}"
                )
        # --- End Emergency Close ---
        return False  # Entry ultimately failed because protection wasn't set

    # --- Success ---
    logger.info(f"{NEON_GREEN}Successfully entered {signal} trade for {symbol} with initial protection set.{RESET}")
    # Update state after successful entry and protection
    symbol_state["break_even_triggered"] = False  # Reset BE state on new entry
    symbol_state["last_entry_price"] = str(actual_entry_price)  # Store actual entry price as string
    return True


# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Enhanced Bybit V5 Multi-Symbol Trading Bot v{BOT_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Path to JSON config file.")
    parser.add_argument("--state", type=str, default=STATE_FILE, help="Path to JSON state file.")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Override symbols in config (comma-separated list, e.g., 'BTC/USDT:USDT,ETH/USDT:USDT').",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable LIVE trading. Overrides 'use_sandbox' and 'enable_trading' in config. USE WITH EXTREME CAUTION!",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG console logging level.")
    args = parser.parse_args()

    # Set console log level based on --debug flag
    console_log_level = logging.DEBUG if args.debug else logging.INFO
    if args.debug:
        pass

    # Setup main logger first
    main_logger = get_logger("main")
    main_logger.info(f" --- Bot Starting --- {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} --- ")

    # Load Configuration
    config = load_config(args.config, main_logger)
    if config is None:
        main_logger.critical("Failed to load or validate configuration. Exiting.")
        sys.exit(1)

    # Apply Command-Line Overrides
    if args.symbols:
        try:
            override_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
            # Basic validation of overridden symbols format
            if not override_symbols or not all("/" in s for s in override_symbols):
                raise ValueError("Invalid symbol format. Expected 'BASE/QUOTE' or 'BASE/QUOTE:SETTLE'.")
            main_logger.warning(f"{NEON_YELLOW}Overriding config symbols via command line: {override_symbols}{RESET}")
            config["symbols"] = override_symbols
        except Exception as e:
            main_logger.error(
                f"Invalid --symbols argument: '{args.symbols}'. Error: {e}. Using symbols from config file."
            )

    if args.live:
        main_logger.warning(
            f"{NEON_RED}--- LIVE TRADING ENABLED via --live flag! This overrides config settings for 'enable_trading' and 'use_sandbox'. ---{RESET}"
        )
        config["enable_trading"] = True
        config["use_sandbox"] = False

    # Log effective trading/sandbox mode after potential overrides
    if config.get("enable_trading"):
        main_logger.warning(f"{NEON_RED}--- TRADING ACTIONS ARE ENABLED ---{RESET}")
    else:
        main_logger.info("--- Trading actions are DISABLED (simulation mode) ---")

    if config.get("use_sandbox"):
        main_logger.warning(f"{NEON_YELLOW}--- SANDBOX MODE (Testnet) is ACTIVE ---{RESET}")
    else:
        main_logger.warning(f"{NEON_RED}--- LIVE EXCHANGE MODE is ACTIVE --- Real funds are at risk! ---{RESET}")

    # Load State
    bot_state = load_state(args.state, main_logger)

    # Initialize Exchange
    exchange = initialize_exchange(config, main_logger)

    exit_code = 0
    if exchange:
        main_logger.info(f"{NEON_GREEN}Exchange initialized successfully. Starting main bot loop...{RESET}")
        try:
            # Start the main bot logic
            run_bot(exchange, config, bot_state)
        except KeyboardInterrupt:
            main_logger.info("Bot stopped by user (KeyboardInterrupt).")
        except Exception as e:
            main_logger.critical(f"{NEON_RED}!!! BOT CRASHED UNEXPECTEDLY IN MAIN LOOP: {e} !!!{RESET}", exc_info=True)
            exit_code = 1  # Indicate error exit
        finally:
            # --- Shutdown Procedures ---
            main_logger.info("Initiating shutdown sequence...")
            # Save the final state
            main_logger.info("Saving final bot state...")
            save_state(args.state, bot_state, main_logger)
            # Close any open resources if necessary (e.g., database connections)
            # ...
            main_logger.info(
                f"--- Bot Shutdown Complete --- {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ---"
            )
    else:
        main_logger.critical("Failed to initialize exchange. Bot cannot start.")
        exit_code = 1

    # Ensure all logs are flushed
    logging.shutdown()
    sys.exit(exit_code)
