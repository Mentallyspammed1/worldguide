
dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  self.df.ta.strategy(self.ta_strategy, timed=False) # timed=True adds overhead
2025-04-24 15:12:22 [America/Chicago] - ERROR    - [livebot_DOT_USDT-USDT] - Fibonacci calculation error for DOT/USDT:USDT: module 'pandas.api.types' has no attribute 'is_decimal_dtype'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/codes/x1.5.py", line 1983, in calculate_fibonacci_levels
    high_series = df_slice["high"] if pd.api.types.is_decimal_dtype(df_slice["high"]) else df_slice["high"].astype(str).apply(Decimal)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'pandas.api.types' has no attribute 'is_decimal_dtype'. Did you mean: 'is_period_dtype'?
2025-04-24 15:12:22 [America/Chicago] - INFO     - [livebot_DOT_USDT-USDT] - No active position. Checking for new entry signals...
2025-04-24 15:12:22 [America/Chicago] - INFO     - [livebot_DOT_USDT-USDT] - Signal Calc (DOT/USDT:USDT @ 4.2282): Set='scalping', Indis(Actv/NaN): 13/0, WtSum: 3.450, RawScore: -0.7145, NormScore: -0.2071, Thresh: +/-0.800 -> Signal: HOLD
2025-04-24 15:12:22 [America/Chicago] - INFO     - [livebot_DOT_USDT-USDT] - --- Finished Processing Symbol: DOT/USDT:USDT ---
2025-04-24 15:12:22 [America/Chicago] - INFO     - [livebot_main] - --- Bot Cycle 2 Finished (Duration: 8.62s) ---
2025-04-24 15:12:22 [America/Chicago] - INFO     - [livebot_main] - Waiting 6.38s for next cycle...
2025-04-24 15:12:29 [America/Chicago] - INFO     - [livebot_main] - --- Starting Bot Cycle 3 ---
2025-04-24 15:12:30 [America/Chicago] - INFO     - [livebot_main] - Available USDT balance (UNIFIED):
# -*- coding: utf-8 -*-
"""
Enhanced Multi-Symbol Trading Bot for Bybit (V5 API) - v1.0.2

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
"""

# --- Required Libraries ---
# Ensure these are installed:
# pip install ccxt pandas numpy pandas_ta python-dotenv colorama pytz
# --------------------------

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
except ImportError:
    try:
        from pytz import timezone as ZoneInfo # Fallback (pip install pytz)
    except ImportError:
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
except ImportError:
    print("Warning: 'colorama' package not found. Colored output will be disabled.")
    print("Install it with: pip install colorama")
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
    getcontext().prec = 36
except Exception as e:
    print(f"Warning: Could not set Decimal precision: {e}. Using default.")

init(autoreset=True)    # Initialize colorama (or dummy init)
load_dotenv()           # Load environment variables from .env file

# --- Constants ---

# Bot Identity
BOT_VERSION = "1.0.2"

# Neon Color Scheme (requires colorama)
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

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
    print(f"Using Timezone: {TZ_NAME}")
except Exception as tz_err:
    print(f"{NEON_YELLOW}Warning: Could not load timezone '{TZ_NAME}': {tz_err}. Defaulting to UTC.{RESET}")
    TIMEZONE = ZoneInfo("UTC")
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
DEFAULT_STOCH_WINDOW = 14
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
            if key and len(key) > 4: # Avoid redacting short/empty strings
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
        stream_formatter = LocalTimeFormatter(
            f"{NEON_BLUE}%(asctime)s{RESET} [{tz_name_str}] - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3] # Include milliseconds in console output
        )
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
            # Type mismatch check (allow int -> float/Decimal promotion)
            is_promoting_num = (isinstance(default_value, (float, Decimal)) and isinstance(updated_config.get(key), int))
            if not is_promoting_num:
                print(f"{NEON_YELLOW}Config Warning: Type mismatch for key '{key}'. Expected {type(default_value).__name__}, got {type(updated_config.get(key)).__name__}. Using loaded value: {repr(updated_config.get(key))}.{RESET}")
                # Note: We keep the user's value despite the type mismatch, but warn them.
                # keys_added_or_type_mismatch = True # Optionally flag type mismatches too
    return updated_config, keys_added_or_type_mismatch

def _validate_config_values(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Validates specific critical configuration values for type, range, and format.

    Args:
        config: The configuration dictionary to validate.
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
    numeric_params: Dict[str, Tuple[type, Union[int, float], Union[int, float]]] = {
        "loop_delay": (int, 5, 3600),           # Min 5 sec delay recommended
        "risk_per_trade": (float, 0.0001, 0.5), # Risk 0.01% to 50% of balance
        "leverage": (int, 1, 125),              # Practical leverage limits (exchange may vary)
        "max_concurrent_positions_total": (int, 1, 100),
        "atr_period": (int, 2, 500),
        "ema_short_period": (int, 2, 500),
        "ema_long_period": (int, 3, 1000),      # Ensure long > short is not checked here, but logically required
        "rsi_period": (int, 2, 500),
        "bollinger_bands_period": (int, 5, 500),
        "bollinger_bands_std_dev": (float, 0.1, 5.0),
        "cci_window": (int, 5, 500),
        "williams_r_window": (int, 2, 500),
        "mfi_window": (int, 5, 500),
        "stoch_rsi_window": (int, 5, 500),
        "stoch_rsi_rsi_window": (int, 5, 500),
        "stoch_rsi_k": (int, 1, 100),
        "stoch_rsi_d": (int, 1, 100),
        "psar_af": (float, 0.001, 0.5),
        "psar_max_af": (float, 0.01, 1.0),
        "sma_10_window": (int, 2, 500),
        "momentum_period": (int, 2, 500),
        "volume_ma_period": (int, 5, 500),
        "fibonacci_window": (int, 10, 1000),
        "orderbook_limit": (int, 1, 200),       # Bybit V5 limit might be 50 or 200 depending on type
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
        "break_even_offset_ticks": (int, 0, 100),   # Number of ticks above/below entry for BE SL
    }
    for key, (expected_type, min_val, max_val) in numeric_params.items():
        value = config.get(key)
        if value is None: continue # Skip if optional or handled by ensure_keys

        try:
            # Attempt conversion to the expected numeric type
            if expected_type == int: num_value = int(value)
            elif expected_type == float: num_value = float(value)
            else: num_value = value # Should not happen based on current definitions

            # Check range
            if not (min_val <= num_value <= max_val):
                logger.error(f"Config Error: '{key}' value {num_value} is outside the recommended range ({min_val} - {max_val}).")
                is_valid = False
            # Store the validated (and potentially type-converted) numeric value back
            config[key] = num_value
        except (ValueError, TypeError):
            logger.error(f"Config Error: '{key}' value '{value}' could not be converted to {expected_type.__name__}.")
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

    # 5. Validate Boolean types (ensure they are actually bool)
    bool_params = [
        "enable_trading", "use_sandbox", "enable_ma_cross_exit", "enable_trailing_stop",
        "enable_break_even", "break_even_force_fixed_sl"
    ]
    for key in bool_params:
        if key in config and not isinstance(config[key], bool):
            logger.error(f"Config Error: '{key}' value '{config[key]}' must be a boolean (true/false).")
            is_valid = False
    # Validate indicator enable flags
    if "indicators" in config and isinstance(config["indicators"], dict):
        for indi_key, indi_val in config["indicators"].items():
            if not isinstance(indi_val, bool):
                logger.error(f"Config Error: Indicator enable flag 'indicators.{indi_key}' value '{indi_val}' must be a boolean (true/false).")
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
    - Validates critical configuration values.
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
        "position_mode": "One-Way",   # "One-Way" or "Hedge" (Hedge mode requires careful implementation)
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
        "enable_trailing_stop": True,    # Enable Trailing Stop Loss feature
        "trailing_stop_callback_rate": 0.005, # TSL distance as fraction of price (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003, # Profit % required to activate TSL (e.g., 0.003 = 0.3%)
        "enable_break_even": True,       # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # ATR multiple profit needed to trigger BE
        "break_even_offset_ticks": 2,    # How many ticks *above* entry (for longs) or *below* (for shorts) to set BE SL
        "break_even_force_fixed_sl": True, # If true, BE replaces TSL; if false, BE sets SL but TSL might remain active if configured
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
                print(f"{NEON_YELLOW}Updating config file '{filepath}' with missing/changed default keys...{RESET}")
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
            print(f"{NEON_RED}Unexpected error loading configuration: {e}. Using defaults.{RESET}")
            config_to_use = default_config # Fallback to defaults

    # --- Final Validation ---
    # Validate the configuration values (whether loaded or default)
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
                return state
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading state file {filepath}: {e}. Starting with empty state.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}. Starting with empty state.", exc_info=True)
            return {}
    else:
        logger.info("No previous state file found. Starting with empty state.")
        return {}

def save_state(filepath: str, state: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Saves the bot's current operational state to a JSON file.
    Ensures Decimals are converted to strings for JSON compatibility.

    Args:
        filepath: Path to the state file.
        state: The dictionary containing the current state to save.
        logger: Logger instance.
    """
    try:
        # Convert Decimals to strings for JSON serialization using json.dumps with default=str
        # Then load it back to ensure the structure is pure JSON-compatible types
        state_to_save = json.loads(json.dumps(state, default=str))

        temp_filepath = filepath + ".tmp" # Save to temporary file first
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=4, sort_keys=True)

        # Atomic rename/replace
        os.replace(temp_filepath, filepath)
        logger.debug(f"Saved current state to {filepath}")

    except (IOError, TypeError) as e:
        logger.error(f"Error saving state file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state: {e}", exc_info=True)
    finally:
        # Clean up temp file if it still exists (e.g., due to error before replace)
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass


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
        # These options fine-tune CCXT's behavior for Bybit
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let CCXT handle basic rate limiting
            'rateLimit': 120,         # Milliseconds between requests (adjust based on Bybit limits if needed)
            'options': {
                'defaultType': 'linear',  # Assume linear contracts unless symbol specifies otherwise
                'adjustForTimeDifference': True, # Auto-sync client time with server time
                'recvWindow': 10000,      # Time window for request validity (milliseconds)
                # Timeouts for various API calls (milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'fetchOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'cancelOrderTimeout': 20000,
                'fetchOHLCVTimeout': 20000,
                'setLeverageTimeout': 20000,
                'fetchMarketsTimeout': 30000,
                # Custom Broker ID for tracking (optional)
                'brokerId': f'EnhancedWhale71_{BOT_VERSION[:3]}', # Example: EnhancedWhale71_1.0
                # Explicitly map endpoints to V5 where possible/necessary
                # CCXT often handles this, but explicit mapping can be safer
                'versions': {
                    'public': {
                        'GET': {
                            'market/tickers': 'v5',
                            'market/kline': 'v5',
                            'market/orderbook': 'v5',
                        }
                    },
                    'private': {
                        'GET': {
                            'position/list': 'v5',
                            'account/wallet-balance': 'v5',
                            'order/realtime': 'v5', # For open orders
                            'order/history': 'v5',  # For closed/cancelled orders
                        },
                        'POST': {
                            'order/create': 'v5',
                            'order/cancel': 'v5',
                            'position/set-leverage': 'v5',
                            'position/trading-stop': 'v5', # For SL/TP/TSL
                        }
                    }
                },
                # Default options hinting CCXT to prefer V5 methods
                'default_options': {
                    'fetchPositions': 'v5',
                    'fetchBalance': 'v5',
                    'createOrder': 'v5',
                    'fetchOrder': 'v5',
                    'fetchTicker': 'v5',
                    'fetchOHLCV': 'v5',
                    'fetchOrderBook': 'v5',
                    'setLeverage': 'v5',
                    # Explicit mapping for specific private POST calls if needed
                    'private_post_v5_position_trading_stop': 'v5',
                },
                # Map CCXT account types to Bybit API account types
                'accountsByType': {
                    'spot': 'SPOT',
                    'future': 'CONTRACT', # Unified term for derivatives
                    'swap': 'CONTRACT',   # Unified term for derivatives
                    'margin': 'UNIFIED',  # Mapping for UTA
                    'option': 'OPTION',
                    'unified': 'UNIFIED', # Mapping for UTA
                    'contract': 'CONTRACT',# Unified term for derivatives
                },
                # Reverse mapping (useful internally for CCXT)
                'accountsById': {
                    'SPOT': 'spot',
                    'CONTRACT': 'contract',
                    'UNIFIED': 'unified',
                    'OPTION': 'option',
                },
                # Specific Bybit options
                'bybit': {
                    'defaultSettleCoin': QUOTE_CURRENCY, # Hint for settlement currency
                }
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
            # No action needed, sandbox mode is False by default

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
                # If detection failed, log a warning. The bot might still work if
                # the default balance fetch method succeeds later, but UTA-specific
                # parsing might be less reliable.
                lg.warning(f"{NEON_YELLOW}Could not definitively determine account type during initial balance check. Proceeding with caution.{RESET}")

            # Check if balance was successfully fetched
            if balance_decimal is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 # Balance fetch failed even after trying different account types
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch failed or returned zero/None (Account Type: {account_type_detected or 'Unknown'}). Check logs above. Ensure API keys have 'Read' permissions for the correct account type (Unified/Contract/Spot) and sufficient funds.{RESET}")
                 # If trading is enabled, this is a critical failure
                 if config.get("enable_trading"):
                     lg.error(f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}")
                     return None
                 else:
                     lg.warning("Continuing in non-trading mode despite balance fetch issue.")

        except ccxt.AuthenticationError as auth_err:
            # Specific handling for authentication errors
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial setup: {auth_err}{RESET}")
            lg.error(f"{NEON_RED}>> Check API Key, API Secret, Permissions (Read/Trade), Account Type (Real/Testnet), and IP Whitelist.{RESET}")
            return None # Fatal error
        except Exception as balance_err:
            # Catch any other unexpected errors during the initial check
            lg.error(f"{NEON_RED}Unexpected error during initial balance check: {balance_err}{RESET}", exc_info=True)
            if config.get("enable_trading"):
                 lg.error(f"{NEON_RED}Aborting initialization due to unexpected balance fetch error in trading mode.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}Continuing in non-trading mode despite unexpected balance fetch error: {balance_err}{RESET}")

        # If all checks passed (or warnings were accepted in non-trading mode)
        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as e:
        # Catch CCXT specific errors during class instantiation or initial connection attempts
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        # Catch any other unexpected errors during setup
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

    # --- Attempt 1: Try fetching as UNIFIED account ---
    try:
        lg.debug("Checking balance with accountType=UNIFIED...")
        params_unified = {'accountType': 'UNIFIED', 'coin': currency}
        # Use safe_ccxt_call but with fewer retries for detection purposes
        bal_info_unified = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, retry_delay=2, params=params_unified)
        parsed_balance = _parse_balance_response(bal_info_unified, currency, 'UNIFIED', lg)
        if parsed_balance is not None:
            lg.info("Successfully fetched balance using UNIFIED account type.")
            unified_balance = parsed_balance
            # If successful, we assume it's a Unified Account
            return True, unified_balance
    except ccxt.ExchangeError as e:
        # Check for specific Bybit error codes indicating wrong account type
        # 10001: Parameter error (often implies wrong accountType for the key)
        # 30086: UTA not supported by the endpoint/key permissions
        error_str = str(e).lower()
        if "accounttype only support" in error_str or "30086" in error_str or "unified account is not supported" in error_str or ("10001" in error_str and "accounttype" in error_str):
             lg.debug("Fetching with UNIFIED failed (as expected for non-UTA), trying CONTRACT/SPOT...")
        else:
             # Different ExchangeError occurred
             lg.warning(f"ExchangeError checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")
             # Fall through to check CONTRACT/SPOT anyway
    except Exception as e:
         # Catch other errors during UNIFIED check
         lg.warning(f"Unexpected error checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")
         # Fall through

    # --- Attempt 2: Try fetching as CONTRACT/SPOT account (if UNIFIED failed) ---
    account_types_to_try = ['CONTRACT', 'SPOT']
    for acc_type in account_types_to_try:
        try:
            lg.debug(f"Checking balance with accountType={acc_type}...")
            params = {'accountType': acc_type, 'coin': currency}
            bal_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, retry_delay=2, params=params)
            parsed_balance = _parse_balance_response(bal_info, currency, acc_type, lg)
            if parsed_balance is not None:
                 lg.info(f"Successfully fetched balance using {acc_type} account type.")
                 contract_spot_balance = parsed_balance
                 # If successful, we assume it's a Non-UTA account
                 return False, contract_spot_balance
        except ccxt.ExchangeError as e:
             lg.warning(f"ExchangeError checking {acc_type} balance: {e}.")
             # Continue to next type if available
        except Exception as e:
             lg.warning(f"Unexpected error checking {acc_type} balance: {e}.")
             # Stop checking other types if unexpected error occurs

    # --- Conclusion ---
    if unified_balance is not None: # Should have returned earlier, but as a fallback
        return True, unified_balance
    if contract_spot_balance is not None: # Should have returned earlier
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
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # Get the method dynamically from the exchange object
            method = getattr(exchange, method_name)
            # Execute the method with provided arguments
            result = method(*args, **kwargs)
            # Return the result immediately on success
            return result

        # --- Specific CCXT Error Handling ---
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * (2 ** attempt) # Exponential backoff base
            suggested_wait: Optional[float] = None
            # Try to parse suggested wait time from Bybit's error message
            try:
                import re
                error_msg = str(e).lower()
                # Look for patterns like "try again in X ms" or "retry after Y s"
                match_ms = re.search(r'(?:try again in|retry after)\s*(\d+)\s*ms', error_msg)
                match_s = re.search(r'(?:try again in|retry after)\s*(\d+)\s*s', error_msg)
                if match_ms:
                    suggested_wait = max(1.0, math.ceil(int(match_ms.group(1)) / 1000) + RATE_LIMIT_BUFFER_SECONDS)
                elif match_s:
                    suggested_wait = max(1.0, int(match_s.group(1)) + RATE_LIMIT_BUFFER_SECONDS)
                # Fallback for generic rate limit messages
                elif "too many visits" in error_msg or "limit" in error_msg:
                    suggested_wait = wait_time + RATE_LIMIT_BUFFER_SECONDS
            except Exception:
                pass # Ignore parsing errors, use default backoff

            final_wait = suggested_wait if suggested_wait is not None else wait_time
            lg.warning(f"Rate limit hit calling {method_name}. Retrying in {final_wait:.2f}s... (Attempt {attempt + 1}/{max_retries + 1}) Error: {e}")
            time.sleep(final_wait)

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            # Handle common network/connectivity issues
            last_exception = e
            wait_time = retry_delay * (2 ** attempt)
            lg.warning(f"Network/DDoS/Timeout error calling {method_name}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(wait_time)

        except ccxt.AuthenticationError as e:
            # Authentication errors are critical and usually not recoverable by retry
            lg.error(f"{NEON_RED}Authentication Error calling {method_name}: {e}. Check API keys/permissions. Not retrying.{RESET}")
            raise e # Re-raise immediately

        except ccxt.ExchangeError as e:
            # Handle general exchange errors, distinguishing retryable vs non-retryable
            last_exception = e
            bybit_code: Optional[int] = None
            ret_msg: str = str(e)
            # Attempt to extract Bybit's specific error code and message
            try:
                # CCXT often includes the raw response details in args[0]
                if hasattr(e, 'args') and len(e.args) > 0:
                    error_details = str(e.args[0])
                    # Find the JSON part containing retCode and retMsg
                    json_start = error_details.find('{')
                    json_end = error_details.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                         details_dict = json.loads(error_details[json_start:json_end])
                         bybit_code = details_dict.get('retCode')
                         ret_msg = details_dict.get('retMsg', str(e)) # Use extracted msg if available
            except (json.JSONDecodeError, IndexError, TypeError):
                pass # Failed to parse details, use default message and no code

            # Define known non-retryable Bybit error codes (add more as needed)
            # Examples: Invalid parameters, insufficient balance, invalid order, account issues
            non_retryable_codes: List[int] = [
                10001, # Parameter error (often invalid symbol, type, etc.)
                110007, # Insufficient balance
                110013, # Order price deviates too much from market price
                110017, # Order price/qty is invalid (precision, range)
                110020, # Position status prohibits action (e.g., closing already closed pos)
                110025, # Position idx error (Hedge Mode issue)
                110043, # Set leverage not modified (treat as success for set_leverage)
                110045, # Qty too small/large
                170007, # Risk limit exceeded
                170131, # TP/SL price invalid
                170132, # TP/SL order price triggers liquidation
                170133, # Cannot set TP/SL/TSL (e.g., position direction mismatch)
                170140, # TP/SL requires position
                30086,  # UTA not supported / Account Type mismatch
            ]

            if bybit_code in non_retryable_codes:
                # Special handling for "Leverage not modified" - treat as success
                if bybit_code == 110043 and method_name == 'set_leverage':
                    lg.info(f"Leverage already set as requested (Code 110043) when calling {method_name}. Ignoring error.")
                    return {} # Return empty dict, often treated as success by calling code

                # Provide hints for common errors
                extra_info = ""
                if bybit_code == 10001 and "accountType" in ret_msg:
                    extra_info = f"{NEON_YELLOW} Hint: Check 'accountType' param (UNIFIED vs CONTRACT/SPOT) or API key permissions.{RESET}"
                elif bybit_code == 10001:
                    extra_info = f"{NEON_YELLOW} Hint: Check API call parameters ({args=}, {kwargs=}).{RESET}"
                elif bybit_code == 110007:
                    extra_info = f"{NEON_YELLOW} Hint: Check available balance for {kwargs.get('symbol', 'the symbol')} in the correct account (UTA/Contract/Spot).{RESET}"

                lg.error(f"{NEON_RED}Non-retryable Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Not retrying.{RESET}{extra_info}")
                raise e # Re-raise the non-retryable error

            else:
                # Unknown or potentially temporary exchange error, retry
                lg.warning(f"{NEON_YELLOW}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Retrying... (Attempt {attempt + 1}/{max_retries + 1}){RESET}")
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)

        except Exception as e:
            # Catch any other unexpected exceptions
            lg.error(f"{NEON_RED}Unexpected Error calling {method_name}: {e}. Not retrying.{RESET}", exc_info=True)
            raise e # Re-raise immediately

    # --- Max Retries Reached ---
    # If the loop completes without returning, max retries were exceeded
    lg.error(f"{NEON_RED}Max retries ({max_retries}) reached for {method_name}. Last error: {last_exception}{RESET}")
    raise last_exception if last_exception else RuntimeError(f"Max retries reached for {method_name} without specific exception.")


# --- Market Info Helper Functions ---
def _determine_category(market: Dict[str, Any]) -> Optional[str]:
    """Determines the Bybit V5 category ('linear', 'inverse', 'spot', 'option') from CCXT market info."""
    market_type = market.get('type')
    is_linear = market.get('linear', False)
    is_inverse = market.get('inverse', False)
    is_spot = market.get('spot', False)
    is_option = market.get('option', False)

    if is_spot: return 'spot'
    if is_option: return 'option'
    if is_linear and market_type in ['swap', 'future']: return 'linear'
    if is_inverse and market_type in ['swap', 'future']: return 'inverse'

    # Fallback checks if primary flags are missing
    if market_type == 'spot': return 'spot'
    if market_type == 'option': return 'option'
    if market_type in ['swap', 'future']:
        settle_asset = market.get('settle', '').upper()
        quote_asset = market.get('quote', '').upper()
        if settle_asset == quote_asset: return 'linear' # Settles in quote (USDT, USDC)
        else: return 'inverse' # Settles in base (BTC, ETH)

    return None # Unable to determine

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Retrieves and processes market information for a symbol from loaded CCXT markets.
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
        price_precision = market.get('precision', {}).get('price')
        amount_precision = market.get('precision', {}).get('amount')

        # Calculate precision digits from precision value (e.g., 0.01 -> 2 digits)
        price_digits = int(-math.log10(price_precision)) if price_precision else 8 # Default 8
        amount_digits = int(-math.log10(amount_precision)) if amount_precision else 8 # Default 8

        # Extract limits safely, converting to Decimal where appropriate
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        cost_limits = limits.get('cost', {})

        min_amount = Decimal(str(amount_limits.get('min'))) if amount_limits.get('min') is not None else None
        max_amount = Decimal(str(amount_limits.get('max'))) if amount_limits.get('max') is not None else None
        min_price = Decimal(str(price_limits.get('min'))) if price_limits.get('min') is not None else None
        max_price = Decimal(str(price_limits.get('max'))) if price_limits.get('max') is not None else None
        min_cost = Decimal(str(cost_limits.get('min'))) if cost_limits.get('min') is not None else None
        max_cost = Decimal(str(cost_limits.get('max'))) if cost_limits.get('max') is not None else None

        # Contract-specific details
        contract_size = Decimal(str(market.get('contractSize', '1'))) # Default to 1 for spot/non-contract
        is_contract = category in ['linear', 'inverse']
        is_inverse = category == 'inverse'

        market_details = {
            'symbol': symbol,
            'id': market.get('id'), # Exchange's internal market ID
            'base': market.get('base'),
            'quote': market.get('quote'),
            'settle': market.get('settle'),
            'type': market.get('type'), # spot, swap, future
            'category': category,       # linear, inverse, spot, option
            'is_contract': is_contract,
            'inverse': is_inverse,
            'contract_size': contract_size,
            'min_tick_size': Decimal(str(price_precision)) if price_precision else None,
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
        # lg.debug(f"Market Info for {symbol}: Category={category}, Tick={market_details['min_tick_size']}, AmountPrec={amount_digits}, ContractSize={contract_size}")
        return market_details

    except ccxt.BadSymbol as e:
        lg.error(f"Error getting market info for '{symbol}': {e}")
        return None
    except Exception as e:
        lg.error(f"Unexpected error processing market info for {symbol}: {e}", exc_info=True)
        return None

# --- Data Fetching Wrappers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Decimal]:
    """Fetches the current ticker price using V5 API via safe_ccxt_call."""
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol)
    if not category:
        lg.error(f"Cannot fetch price for {symbol}: Category unknown."); return None

    try:
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching ticker for {symbol} with params: {params}")
        ticker = safe_ccxt_call(exchange, 'fetch_ticker', lg, symbol=symbol, params=params)

        if ticker and ticker.get('last') is not None:
            price_str = str(ticker['last'])
            price_dec = Decimal(price_str)
            if price_dec.is_finite() and price_dec > 0:
                # lg.debug(f"Current price for {symbol}: {price_dec}")
                return price_dec
            else:
                lg.error(f"Invalid price ('{price_str}') received for {symbol}.")
                return None
        else:
            lg.warning(f"Could not fetch 'last' price for {symbol}. Ticker: {ticker}")
            return None
    except Exception as e:
        lg.error(f"Error fetching current price for {symbol}: {e}", exc_info=True)
        return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger, market_info: Dict) -> pd.DataFrame:
    """Fetches OHLCV data using V5 API via safe_ccxt_call and returns a DataFrame."""
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol)
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)

    if not category:
        lg.error(f"Cannot fetch klines for {symbol}: Category unknown."); return pd.DataFrame()
    if not ccxt_timeframe:
        lg.error(f"Invalid timeframe '{timeframe}' provided for {symbol}."); return pd.DataFrame()

    try:
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching {limit} klines for {symbol} ({ccxt_timeframe}) with params: {params}")
        ohlcv = safe_ccxt_call(exchange, 'fetch_ohlcv', lg, symbol=symbol, timeframe=ccxt_timeframe, limit=limit, params=params)

        if not ohlcv:
            lg.warning(f"fetch_ohlcv returned empty data for {symbol}.")
            return pd.DataFrame()

        # Convert to DataFrame and set column names
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"Kline data for {symbol} resulted in an empty DataFrame.")
             return df

        # Convert timestamp to datetime (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision, handling potential errors
        for col in ['open', 'high', 'low', 'close', 'volume']:
            try:
                # Use .astype(str) first to handle potential float inaccuracies before Decimal
                df[col] = df[col].astype(str).apply(Decimal)
            except (InvalidOperation, TypeError, ValueError) as e:
                 lg.error(f"Error converting column '{col}' to Decimal for {symbol}: {e}. Coercing errors to NaN.")
                 # Coerce errors during conversion to NaN, then potentially handle/drop NaNs later
                 df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)

        # Optional: Drop rows with any NaN values if necessary, though TA libraries often handle them
        # df.dropna(inplace=True)
        # lg.debug(f"Fetched and processed {len(df)} klines for {symbol}.")
        return df

    except Exception as e:
        lg.error(f"Error fetching/processing klines for {symbol}: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """Fetches order book data using V5 API via safe_ccxt_call."""
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol)
    if not category:
        lg.error(f"Cannot fetch orderbook for {symbol}: Category unknown."); return None

    try:
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching order book (limit {limit}) for {symbol} with params: {params}")
        orderbook = safe_ccxt_call(exchange, 'fetch_order_book', lg, symbol=symbol, limit=limit, params=params)

        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            # Optional: Convert prices/amounts to Decimal here if needed downstream,
            # but often kept as float/str from CCXT for direct use or simple checks.
            # lg.debug(f"Fetched order book for {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
            return orderbook
        else:
            lg.warning(f"Failed to fetch valid order book for {symbol}. Response: {orderbook}")
            return None
    except Exception as e:
        lg.error(f"Error fetching order book for {symbol}: {e}", exc_info=True)
        return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency.
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

    # Determine which accountType parameter(s) to use based on detection
    if IS_UNIFIED_ACCOUNT:
        account_types_to_try = ['UNIFIED']
        lg.debug(f"Fetching balance specifically for UNIFIED account ({currency}).")
    else:
        # For Non-UTA, CONTRACT usually holds futures balance, SPOT for spot balance.
        # Prioritize CONTRACT for futures trading focus.
        account_types_to_try = ['CONTRACT', 'SPOT']
        lg.debug(f"Fetching balance for Non-UTA account ({currency}), trying types: {account_types_to_try}.")

    last_exception: Optional[Exception] = None

    # Outer retry loop for network/rate limit issues
    for attempt in range(MAX_API_RETRIES + 1):
        balance_info: Optional[Dict] = None
        successful_acc_type: Optional[str] = None
        parsed_balance: Optional[Decimal] = None

        # Inner loop to try different account types if needed (mainly for Non-UTA)
        for acc_type in account_types_to_try:
            try:
                params = {'accountType': acc_type, 'coin': currency}
                lg.debug(f"Fetching balance with params={params} (Attempt {attempt + 1})")
                # Use safe_ccxt_call with 0 inner retries; outer loop handles retries
                balance_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=0, params=params)

                # Parse the response using the dedicated parsing function
                parsed_balance = _parse_balance_response(balance_info, currency, acc_type, lg)

                if parsed_balance is not None:
                    # Successfully fetched and parsed balance for this type
                    successful_acc_type = acc_type
                    lg.info(f"Available {currency} balance ({successful_acc_type}): {parsed_balance:.4f}")
                    return parsed_balance # Return the found balance
                else:
                    # Parsing failed or currency not found for this type, try next type
                    lg.debug(f"Balance for {currency} not found or parsing failed for type {acc_type}.")
                    balance_info = None # Reset for next loop iteration

            except ccxt.ExchangeError as e:
                # Handle specific exchange errors if needed, but safe_ccxt_call(max_retries=0)
                # should have raised non-retryable ones already. Log and try next type.
                lg.debug(f"Exchange error fetching balance type {acc_type}: {e}. Trying next type if available.")
                last_exception = e
                continue # Try the next account type in the inner loop

            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                 # Network/RateLimit errors should trigger the outer retry loop
                 lg.warning(f"Network/RateLimit error during balance fetch type {acc_type}: {e}")
                 last_exception = e
                 break # Break the inner loop (account types) and let the outer loop retry

            except Exception as e:
                 # Unexpected errors should also trigger the outer retry loop
                 lg.error(f"Unexpected error during balance fetch type {acc_type}: {e}", exc_info=True)
                 last_exception = e
                 break # Break the inner loop and let the outer loop retry

        # --- After Inner Loop ---
        # If we broke from inner loop due to network/unexpected error, let outer loop handle retry
        if isinstance(last_exception, (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded, Exception)):
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} encountered recoverable error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue # Continue to the next iteration of the outer retry loop
            else:
                 lg.error(f"{NEON_RED}Max retries reached fetching balance for {currency} after network/unexpected error. Last error: {last_exception}{RESET}")
                 return None # Exhausted retries

        # If inner loop completed without finding balance and without network errors
        # (This might happen if the currency exists but has zero balance and API response is tricky)
        if parsed_balance is None:
            if attempt < MAX_API_RETRIES:
                # Retry outer loop even if no specific error occurred, maybe temporary issue
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} failed to find/parse balance for type(s): {account_types_to_try}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue # Retry outer loop
            else:
                 lg.error(f"{NEON_RED}Max retries reached. Failed to fetch/parse balance for {currency} using types: {account_types_to_try}. Last error: {last_exception}{RESET}")
                 return None

    # Fallback if logic somehow exits loop without returning
    lg.error(f"{NEON_RED}Balance fetch logic completed unexpectedly without returning a value for {currency}.{RESET}")
    return None


def _parse_balance_response(balance_info: Optional[Dict], currency: str, account_type_checked: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Parses the raw response from CCXT's fetch_balance, adapting to Bybit V5 structure.
    Specifically looks for 'availableBalance' in the nested V5 response for UNIFIED/CONTRACT.

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

    try:
        # --- Strategy 1: Prioritize Bybit V5 structure (info -> result -> list -> coin[]) ---
        # This is expected when using 'accountType' parameter with Bybit V5
        if ('info' in balance_info and
            isinstance(balance_info['info'], dict) and
            balance_info['info'].get('retCode') == 0 and
            'result' in balance_info['info'] and
            isinstance(balance_info['info']['result'], dict) and
            'list' in balance_info['info']['result'] and
            isinstance(balance_info['info']['result']['list'], list)):

            balance_list = balance_info['info']['result']['list']
            lg.debug(f"Parsing V5 balance structure for {currency} ({account_type_checked}). List: {balance_list}")

            for account_data in balance_list:
                # Ensure we are looking at the correct account type within the response
                # (UTA response might contain multiple account types)
                if isinstance(account_data, dict) and account_data.get('accountType') == account_type_checked:
                    coin_list = account_data.get('coin')
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                            if isinstance(coin_data, dict) and coin_data.get('coin') == currency:
                                # V5 Field Priority:
                                # 1. 'availableBalance': Balance usable for new orders (most relevant)
                                # 2. 'availableToWithdraw': Sometimes used interchangeably? Less likely needed.
                                # 3. 'walletBalance': Total balance including unrealized PnL etc. (fallback)
                                free = coin_data.get('availableBalance')
                                if free is None or str(free).strip() == "":
                                    lg.debug(f"'availableBalance' missing/empty for {currency} in {account_type_checked}, trying 'walletBalance'")
                                    free = coin_data.get('walletBalance') # Fallback to total wallet balance

                                if free is not None and str(free).strip() != "":
                                    available_balance_str = str(free)
                                    lg.debug(f"Parsed balance from Bybit V5 ({account_type_checked} -> {currency}): Value='{available_balance_str}'")
                                    break # Found the currency in this account type's coin list
                        if available_balance_str is not None:
                            break # Found the currency in this account type
            if available_balance_str is None:
                 lg.debug(f"Currency '{currency}' not found within Bybit V5 list structure for account type '{account_type_checked}'.")

        # --- Strategy 2: Fallback to standard CCXT 'free' balance structure ---
        # This might occur if 'accountType' wasn't used or for different exchanges/API versions
        elif available_balance_str is None and currency in balance_info and isinstance(balance_info.get(currency), dict):
            free_val = balance_info[currency].get('free')
            if free_val is not None:
                available_balance_str = str(free_val)
                lg.debug(f"Parsed balance via standard CCXT structure ['{currency}']['free']: {available_balance_str}")

        # --- Strategy 3: Fallback to top-level 'free' dictionary (less common) ---
        elif available_balance_str is None and 'free' in balance_info and isinstance(balance_info.get('free'), dict):
             free_val = balance_info['free'].get(currency)
             if free_val is not None:
                 available_balance_str = str(free_val)
                 lg.debug(f"Parsed balance via top-level 'free' dictionary ['free']['{currency}']: {available_balance_str}")


        # --- Conversion and Validation ---
        if available_balance_str is None:
            lg.debug(f"Could not extract balance for {currency} from response structure ({account_type_checked}). Response info: {balance_info.get('info', balance_info)}")
            return None

        # Convert the extracted string to Decimal
        final_balance = Decimal(available_balance_str)
        # Ensure the balance is a non-negative finite number
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
    """
    Analyzes market data using pandas_ta.Strategy to generate weighted trading signals.
    Calculates technical indicators, Fibonacci levels, and provides risk management values.
    Uses Decimal for internal calculations involving prices and quantities.
    Manages symbol-specific state like break-even status via a shared dictionary.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
        symbol_state: Dict[str, Any], # Mutable state dict shared with main loop
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: DataFrame containing OHLCV data (must have Decimal columns).
            logger: Logger instance for this symbol.
            config: The main configuration dictionary.
            market_info: Processed market information for the symbol.
            symbol_state: Mutable dictionary holding state for this symbol (e.g., 'break_even_triggered').

        Raises:
            ValueError: If df, market_info, or symbol_state is invalid.
        """
        if df is None or df.empty:
            raise ValueError("TradingAnalyzer requires a non-empty DataFrame.")
        if not market_info:
            raise ValueError("TradingAnalyzer requires valid market_info.")
        if symbol_state is None: # Check for None explicitly
            raise ValueError("TradingAnalyzer requires a valid symbol_state dictionary.")

        self.df_raw = df # Keep raw Decimal DF for potential future use or precise checks
        self.df = df.copy() # Work on a copy for TA calculations
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
        # Map generic indicator names (used in config/logic) to actual column names generated by pandas_ta
        self.ta_column_map: Dict[str, str] = {}

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' is empty or not found for {self.symbol}. Signal generation will be disabled.{RESET}")

        # --- Data Preparation for pandas_ta ---
        # pandas_ta typically requires float inputs. Convert relevant columns.
        self._convert_df_for_ta()

        # --- Initialize and Calculate Indicators ---
        if not self.df.empty:
            self._define_ta_strategy()
            self._calculate_all_indicators()
            self._update_latest_indicator_values() # Populates self.indicator_values with Decimals
            self.calculate_fibonacci_levels() # Calculate initial Fib levels
        else:
            logger.warning(f"DataFrame is empty after float conversion for {self.symbol}. Cannot calculate indicators.")


    def _convert_df_for_ta(self) -> None:
        """Converts necessary DataFrame columns (OHLCV) to float for pandas_ta compatibility."""
        try:
            cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
            for col in cols_to_convert:
                 if col in self.df.columns:
                      # Ensure column exists and is not already float
                      if pd.api.types.is_period_dtype(self.df[col]):
                          # Convert Decimal to float, handling potential non-finite values
                          self.df[col] = self.df[col].apply(lambda x: float(x) if x is not None and x.is_finite() else np.nan)
                      elif not pd.api.types.is_float_dtype(self.df[col]):
                          # If not Decimal and not float, try standard numeric conversion
                          self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            # self.logger.debug(f"DataFrame dtypes prepared for TA: {self.df[cols_to_convert].dtypes.to_dict()}")
        except Exception as e:
             self.logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
             # Mark DataFrame as potentially unusable for TA
             self.df = pd.DataFrame() # Set to empty to prevent further processing


    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even triggered status from the shared symbol state."""
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

        # Helper to safely get numeric parameters from config or use defaults
        def get_num_param(key: str, default: Union[int, float]) -> Union[int, float]:
            val = cfg.get(key, default)
            try:
                return int(val) if isinstance(default, int) else float(val)
            except (ValueError, TypeError):
                self.logger.warning(f"Config Warning: Invalid numeric value '{val}' for '{key}'. Using default: {default}")
                return default

        # Get parameters for all potential indicators
        atr_p = get_num_param("atr_period", DEFAULT_ATR_PERIOD)
        ema_s = get_num_param("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
        ema_l = get_num_param("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
        rsi_p = get_num_param("rsi_period", DEFAULT_RSI_WINDOW)
        bb_p = get_num_param("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        bb_std = get_num_param("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        cci_w = get_num_param("cci_window", DEFAULT_CCI_WINDOW)
        wr_w = get_num_param("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        mfi_w = get_num_param("mfi_window", DEFAULT_MFI_WINDOW)
        stochrsi_w = get_num_param("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
        stochrsi_rsi_w = get_num_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
        stochrsi_k = get_num_param("stoch_rsi_k", DEFAULT_K_WINDOW)
        stochrsi_d = get_num_param("stoch_rsi_d", DEFAULT_D_WINDOW)
        psar_af = get_num_param("psar_af", DEFAULT_PSAR_AF)
        psar_max = get_num_param("psar_max_af", DEFAULT_PSAR_MAX_AF)
        sma10_w = get_num_param("sma_10_window", DEFAULT_SMA_10_WINDOW)
        mom_p = get_num_param("momentum_period", DEFAULT_MOMENTUM_PERIOD)
        vol_ma_p = get_num_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Build the list of indicators for the pandas_ta Strategy
        ta_list: List[Dict[str, Any]] = []
        self.ta_column_map: Dict[str, str] = {} # Reset map

        # --- Add indicators based on config flags and valid parameters ---
        # ATR (Always calculated as it's used for SL/TP/BE)
        if atr_p > 0:
            ta_list.append({"kind": "atr", "length": atr_p})
            self.ta_column_map["ATR"] = f"ATRr_{atr_p}" # pandas_ta default name for ATR value

        # EMAs (Needed if ema_alignment or MA cross exit is enabled)
        if indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit"):
            if ema_s > 0:
                col_name = f"EMA_{ema_s}"
                ta_list.append({"kind": "ema", "length": ema_s, "col_names": (col_name,)})
                self.ta_column_map["EMA_Short"] = col_name
            if ema_l > 0 and ema_l > ema_s: # Ensure long period is longer than short
                col_name = f"EMA_{ema_l}"
                ta_list.append({"kind": "ema", "length": ema_l, "col_names": (col_name,)})
                self.ta_column_map["EMA_Long"] = col_name
            elif ema_l <= ema_s:
                self.logger.warning(f"EMA Long period ({ema_l}) must be greater than Short ({ema_s}) for {self.symbol}. Disabling EMA Long.")

        # Momentum
        if indi_cfg.get("momentum") and mom_p > 0:
            col_name = f"MOM_{mom_p}"
            ta_list.append({"kind": "mom", "length": mom_p, "col_names": (col_name,)})
            self.ta_column_map["Momentum"] = col_name

        # Volume SMA (for Volume Confirmation)
        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0:
            col_name = f"VOL_SMA_{vol_ma_p}"
            # Calculate SMA on the 'volume' column
            ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_p, "col_names": (col_name,)})
            self.ta_column_map["Volume_MA"] = col_name

        # Stochastic RSI
        if indi_cfg.get("stoch_rsi") and stochrsi_w > 0 and stochrsi_rsi_w > 0 and stochrsi_k > 0 and stochrsi_d > 0:
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
            # Generate column names based on parameters
            bb_std_str = f"{bb_std:.1f}".replace('.', '_') # Format std dev for column name
            bbl = f"BBL_{bb_p}_{bb_std_str}" # Lower Band
            bbm = f"BBM_{bb_p}_{bb_std_str}" # Middle Band (SMA)
            bbu = f"BBU_{bb_p}_{bb_std_str}" # Upper Band
            bbb = f"BBB_{bb_p}_{bb_std_str}" # Bandwidth
            bbp = f"BBP_{bb_p}_{bb_std_str}" # %B Position
            ta_list.append({
                "kind": "bbands", "length": bb_p, "std": bb_std,
                "col_names": (bbl, bbm, bbu, bbb, bbp)
            })
            self.ta_column_map["BB_Lower"] = bbl
            self.ta_column_map["BB_Middle"] = bbm
            self.ta_column_map["BB_Upper"] = bbu

        # VWAP (Volume Weighted Average Price)
        if indi_cfg.get("vwap"):
            # VWAP often calculated daily, pandas_ta default might reset daily
            # Ensure 'typical' price (HLC/3) is available if needed, though pandas_ta might calculate it
            if 'typical' not in self.df.columns and all(c in self.df.columns for c in ['high','low','close']):
                 self.df['typical'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
            vwap_col = "VWAP_D" # Default pandas_ta name (might vary)
            ta_list.append({"kind": "vwap", "col_names": (vwap_col,)})
            self.ta_column_map["VWAP"] = vwap_col

        # CCI (Commodity Channel Index)
        if indi_cfg.get("cci") and cci_w > 0:
            cci_col = f"CCI_{cci_w}_0.015" # Default pandas_ta name includes constant
            ta_list.append({"kind": "cci", "length": cci_w, "col_names": (cci_col,)})
            self.ta_column_map["CCI"] = cci_col

        # Williams %R
        if indi_cfg.get("wr") and wr_w > 0:
            wr_col = f"WILLR_{wr_w}"
            ta_list.append({"kind": "willr", "length": wr_w, "col_names": (wr_col,)})
            self.ta_column_map["WR"] = wr_col

        # Parabolic SAR
        if indi_cfg.get("psar"):
            # Format AF parameters for column names, removing trailing zeros/dots
            psar_af_str = f"{psar_af}".rstrip('0').rstrip('.')
            psar_max_str = f"{psar_max}".rstrip('0').rstrip('.')
            # Default PSAR column names from pandas_ta
            l_col = f"PSARl_{psar_af_str}_{psar_max_str}" # Long signal line
            s_col = f"PSARs_{psar_af_str}_{psar_max_str}" # Short signal line
            af_col = f"PSARaf_{psar_af_str}_{psar_max_str}"# Acceleration factor
            r_col = f"PSARr_{psar_af_str}_{psar_max_str}" # Reversal points (0 or 1)
            ta_list.append({
                "kind": "psar", "af": psar_af, "max_af": psar_max,
                "col_names": (l_col, s_col, af_col, r_col)
            })
            self.ta_column_map["PSAR_Long"] = l_col
            self.ta_column_map["PSAR_Short"] = s_col
            self.ta_column_map["PSAR_AF"] = af_col
            self.ta_column_map["PSAR_Reversal"] = r_col

        # SMA 10 (Example additional MA)
        if indi_cfg.get("sma_10") and sma10_w > 0:
            col_name = f"SMA_{sma10_w}"
            ta_list.append({"kind": "sma", "length": sma10_w, "col_names": (col_name,)})
            self.ta_column_map["SMA10"] = col_name

        # MFI (Money Flow Index)
        if indi_cfg.get("mfi") and mfi_w > 0:
            # MFI requires typical price and volume
            if 'typical' not in self.df.columns and all(c in self.df.columns for c in ['high','low','close']):
                 self.df['typical'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
            col_name = f"MFI_{mfi_w}"
            ta_list.append({"kind": "mfi", "length": mfi_w, "col_names": (col_name,)})
            self.ta_column_map["MFI"] = col_name

        # --- Create Strategy ---
        if not ta_list:
            self.logger.warning(f"No valid indicators enabled or configured for {self.symbol}. TA Strategy will not be created.")
            return

        self.ta_strategy = ta.Strategy(
            name="EnhancedMultiIndicator",
            description="Calculates multiple TA indicators based on bot config",
            ta=ta_list
        )
        self.logger.debug(f"Defined TA Strategy for {self.symbol} with {len(ta_list)} indicator groups.")
        # self.logger.debug(f"TA Column Map: {self.ta_column_map}")


    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using the defined pandas_ta strategy."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return
        if not self.ta_strategy:
            self.logger.warning(f"TA Strategy not defined for {self.symbol}, cannot calculate indicators.")
            return

        # Check if sufficient data exists for the strategy's requirements
        min_required_data = self.ta_strategy.required if hasattr(self.ta_strategy, 'required') else 50 # Estimate if 'required' not available
        buffer = 20 # Add a buffer for stability
        if len(self.df) < min_required_data + buffer:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} rows) for {self.symbol} TA calculation (min recommended: {min_required_data + buffer}). Results may be inaccurate or contain NaNs.{RESET}")

        try:
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol}...")
            # Apply the strategy to the DataFrame (modifies self.df inplace)
            self.df.ta.strategy(self.ta_strategy, timed=False) # timed=True adds overhead
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. DataFrame columns: {self.df.columns.tolist()}")
        except AttributeError as ae:
             # Catch errors related to incorrect input types (e.g., Decimal passed to TA function)
             if "'Decimal' object has no attribute" in str(ae):
                 self.logger.error(f"{NEON_RED}Pandas TA Error ({self.symbol}): Input must be float, not Decimal. Check data conversion. Error: {ae}{RESET}", exc_info=False)
             else:
                 self.logger.error(f"{NEON_RED}Pandas TA attribute error ({self.symbol}): {ae}. Is pandas_ta library installed and working correctly?{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta strategy for {self.symbol}: {e}{RESET}", exc_info=True)
            # Depending on severity, might want to clear df or handle differently


    def _update_latest_indicator_values(self) -> None:
        """
        Updates the self.indicator_values dictionary with the latest calculated
        indicator values from the DataFrame, converting them back to Decimal.
        """
        self.indicator_values = {} # Reset before populating
        if self.df.empty:
            self.logger.warning(f"DataFrame empty, cannot update latest indicator values for {self.symbol}.")
            return

        try:
            # Get the last row of the DataFrame
            latest_series = self.df.iloc[-1]
            if latest_series.isnull().all():
                self.logger.warning(f"Last row of DataFrame contains all NaNs for {self.symbol}. Cannot update latest values.")
                return

            # Helper to safely convert float/object back to Decimal
            def to_decimal(value: Any) -> Optional[Decimal]:
                if pd.isna(value) or value is None:
                    return None
                try:
                    # Convert to string first to avoid potential float precision issues
                    dec_val = Decimal(str(value))
                    # Return only if it's a finite number (not NaN or Inf)
                    return dec_val if dec_val.is_finite() else None
                except (InvalidOperation, ValueError, TypeError):
                    # Log conversion errors if they occur unexpectedly
                    # self.logger.warning(f"Could not convert value '{value}' (type {type(value)}) to Decimal.")
                    return None

            # Populate indicator_values using the ta_column_map
            for generic_name, actual_col_name in self.ta_column_map.items():
                if actual_col_name in latest_series:
                    self.indicator_values[generic_name] = to_decimal(latest_series.get(actual_col_name))
                else:
                     # This case should ideally not happen if strategy ran correctly
                     self.logger.debug(f"Column '{actual_col_name}' not found in DataFrame for indicator '{generic_name}' ({self.symbol}).")


            # Also add latest OHLCV values (from the float df, converted back to Decimal)
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                 if base_col in latest_series:
                     self.indicator_values[base_col.capitalize()] = to_decimal(latest_series.get(base_col))

            # Log the updated values (optional, can be verbose)
            valid_values_str = {k: f"{v:.5f}" for k, v in self.indicator_values.items() if v is not None}
            self.logger.debug(f"Latest indicator Decimal values updated for {self.symbol}: Count={len(valid_values_str)}")
            # self.logger.debug(f"Values: {valid_values_str}") # Uncomment for detailed value logging

        except IndexError:
            self.logger.error(f"DataFrame index out of bounds when trying to update latest indicator values for {self.symbol}.")
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = {} # Clear values on error


    # --- Precision and Market Info Helpers ---

    def get_min_tick_size(self) -> Optional[Decimal]:
        """Gets the minimum price tick size as a Decimal from market info."""
        tick = self.market_info.get('min_tick_size')
        if tick is None or not isinstance(tick, Decimal) or tick <= 0:
            self.logger.warning(f"Invalid or missing min_tick_size ({tick}) for {self.symbol}. Quantization may fail.")
            return None
        return tick

    def get_price_precision_digits(self) -> int:
        """Gets the number of decimal places for price precision."""
        return self.market_info.get('price_precision_digits', 8) # Default to 8 if missing

    def get_amount_precision_digits(self) -> int:
        """Gets the number of decimal places for amount (quantity) precision."""
        return self.market_info.get('amount_precision_digits', 8) # Default to 8 if missing

    def quantize_price(self, price: Union[Decimal, float, str], rounding: str = ROUND_DOWN) -> Optional[Decimal]:
        """
        Quantizes a price to the market's minimum tick size using specified rounding.

        Args:
            price: The price to quantize (Decimal, float, or string).
            rounding: The rounding mode (e.g., ROUND_DOWN, ROUND_UP, ROUND_HALF_UP).

        Returns:
            The quantized price as a Decimal, or None if quantization fails.
        """
        min_tick = self.get_min_tick_size()
        if min_tick is None:
            self.logger.error(f"Cannot quantize price for {self.symbol}: Missing or invalid min_tick_size.")
            return None
        try:
            price_decimal = Decimal(str(price)) # Convert input to Decimal
            if not price_decimal.is_finite():
                self.logger.warning(f"Cannot quantize non-finite price '{price}' for {self.symbol}.")
                return None
            # Formula: floor/ceil/round(price / tick_size) * tick_size
            quantized = (price_decimal / min_tick).quantize(Decimal('0'), rounding=rounding) * min_tick
            return quantized
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing price '{price}' for {self.symbol}: {e}")
            return None

    def quantize_amount(self, amount: Union[Decimal, float, str], rounding: str = ROUND_DOWN) -> Optional[Decimal]:
        """
        Quantizes an amount (quantity) to the market's amount precision (step size)
        using specified rounding.

        Args:
            amount: The amount to quantize (Decimal, float, or string).
            rounding: The rounding mode (e.g., ROUND_DOWN, ROUND_UP).

        Returns:
            The quantized amount as a Decimal, or None if quantization fails.
        """
        amount_digits = self.get_amount_precision_digits()
        try:
            amount_decimal = Decimal(str(amount)) # Convert input to Decimal
            if not amount_decimal.is_finite():
                 self.logger.warning(f"Cannot quantize non-finite amount '{amount}' for {self.symbol}.")
                 return None

            # Calculate the step size based on precision digits (e.g., 2 digits -> step 0.01)
            step_size = Decimal('1') / (Decimal('10') ** amount_digits)

            # Quantize using the step size
            quantized = (amount_decimal / step_size).quantize(Decimal('0'), rounding=rounding) * step_size

            # Format the result to the correct number of decimal places to avoid trailing zeros beyond precision
            # Note: This conversion back to string and then Decimal ensures the exact precision representation.
            return Decimal(f"{quantized:.{amount_digits}f}")

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing amount '{amount}' for {self.symbol}: {e}")
            return None

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """
        Calculates Fibonacci retracement levels based on the high/low over a specified window.
        Uses Decimal precision and quantizes the resulting levels to the market's tick size.

        Args:
            window: The lookback period (number of candles). Uses config value if None.

        Returns:
            A dictionary where keys are Fibonacci level descriptions (e.g., "Fib_38.2%")
            and values are the corresponding quantized price levels as Decimals.
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
            # Ensure high/low columns are Decimal or convert them safely
            if 'high' not in df_slice or 'low' not in df_slice:
                 self.logger.warning(f"Missing 'high' or 'low' column in df_raw for Fib calculation ({self.symbol}).")
                 return {}

            # Convert to Decimal if they aren't already, handling potential errors
            try:
                high_series = df_slice["high"] if pd.api.types.is_decimal_dtype(df_slice["high"]) else df_slice["high"].astype(str).apply(Decimal)
                low_series = df_slice["low"] if pd.api.types.is_decimal_dtype(df_slice["low"]) else df_slice["low"].astype(str).apply(Decimal)
            except (InvalidOperation, TypeError, ValueError):
                self.logger.error(f"Error converting high/low to Decimal for Fib calculation ({self.symbol}).")
                return {}

            # Find max high and min low in the window, dropping NaNs
            high_raw = high_series.dropna().max()
            low_raw = low_series.dropna().min()

            # Validate the extracted high and low
            if pd.isna(high_raw) or pd.isna(low_raw) or not high_raw.is_finite() or not low_raw.is_finite():
                self.logger.warning(f"Could not find valid finite high/low in the last {window} periods for Fib calculation on {self.symbol}.")
                return {}

            high: Decimal = high_raw
            low: Decimal = low_raw
            diff: Decimal = high - low

            levels: Dict[str, Decimal] = {}
            min_tick: Optional[Decimal] = self.get_min_tick_size()

            if diff >= 0 and min_tick is not None:
                # Calculate levels only if range is valid and tick size is available
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float)) # Convert float level to Decimal
                    level_price_raw = high - (diff * level_pct)

                    # Quantize the calculated level price (usually round down for support/resistance)
                    level_price_quantized = self.quantize_price(level_price_raw, rounding=ROUND_DOWN)

                    if level_price_quantized is not None:
                        level_name = f"Fib_{level_pct * 100:.1f}%" # e.g., Fib_23.6%
                        levels[level_name] = level_price_quantized
                    else:
                        # Log if quantization failed for a specific level
                        self.logger.warning(f"Failed to quantize Fibonacci level {level_pct*100:.1f}% (Raw: {level_price_raw}) for {self.symbol}")

            elif min_tick is None:
                 # Fallback if tick size is missing (less ideal)
                 self.logger.warning(f"Calculating raw (non-quantized) Fibonacci levels for {self.symbol} due to missing min_tick_size.")
                 for level_pct_float in FIB_LEVELS:
                     level_pct = Decimal(str(level_pct_float))
                     level_price_raw = high - (diff * level_pct)
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_raw # Store raw Decimal

            elif diff < 0:
                 # This shouldn't happen if max/min work correctly
                 self.logger.warning(f"Invalid range detected (high < low?) for Fibonacci calculation on {self.symbol}. High={high}, Low={low}")

            self.fib_levels_data = levels # Store calculated levels
            # Log the calculated levels (optional, can be verbose)
            # price_prec = self.get_price_precision_digits()
            # log_levels = {k: f"{v:.{price_prec}f}" for k, v in levels.items()}
            # self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} (Window: {window}): {log_levels}")
            return levels

        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {} # Clear levels on error
            return {}


    # --- Indicator Check Methods ---
    # These methods check the latest indicator values and return a score typically
    # between -1.0 (strong sell) and 1.0 (strong buy), or None if the indicator
    # value is unavailable. They use float for scoring simplicity.

    def _get_indicator_float(self, name: str) -> Optional[float]:
        """Safely retrieves an indicator value from self.indicator_values as a float."""
        val = self.indicator_values.get(name) # Gets Decimal or None
        if val is None or not val.is_finite():
            # self.logger.debug(f"Indicator '{name}' value is missing or non-finite for {self.symbol}.")
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            # self.logger.warning(f"Could not convert indicator '{name}' Decimal value '{val}' to float.")
            return None

    def _check_ema_alignment(self) -> Optional[float]:
        """Checks if short EMA is above/below long EMA. Score: 1.0 (bullish), -1.0 (bearish), 0.0 (aligned), None (missing)."""
        ema_short = self._get_indicator_float("EMA_Short")
        ema_long = self._get_indicator_float("EMA_Long")
        if ema_short is None or ema_long is None: return None
        if ema_short > ema_long: return 1.0
        if ema_short < ema_long: return -1.0
        return 0.0 # Exactly equal

    def _check_momentum(self) -> Optional[float]:
        """Checks Momentum indicator. Scales score based on value. Score: -1.0 to 1.0, None (missing)."""
        mom = self._get_indicator_float("Momentum")
        if mom is None: return None
        # Scale momentum score (e.g., divide by 10 or apply sigmoid if range is large)
        scaling_factor = 0.1 # Simple linear scaling, adjust as needed
        score = mom * scaling_factor
        return max(-1.0, min(1.0, score)) # Clamp score to [-1, 1]

    def _check_volume_confirmation(self) -> Optional[float]:
        """Checks if current volume significantly exceeds its moving average. Score: 0.7 (confirmed), 0.0 (not confirmed), None (missing)."""
        vol = self._get_indicator_float("Volume")
        vol_ma = self._get_indicator_float("Volume_MA")
        if vol is None or vol_ma is None: return None
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        if vol_ma > 0 and vol > vol_ma * multiplier:
            return 0.7 # Volume confirmation signal (positive score, magnitude less than 1)
        return 0.0 # No strong volume confirmation

    def _check_stoch_rsi(self) -> Optional[float]:
        """Checks Stochastic RSI K and D lines for overbought/oversold conditions and crossovers. Score: -1.0 to 1.0, None (missing)."""
        k = self._get_indicator_float("StochRSI_K")
        d = self._get_indicator_float("StochRSI_D")
        if k is None or d is None: return None

        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        score = 0.0

        # Oversold conditions (stronger buy signal if K crosses above D)
        if k < oversold and d < oversold: score = 0.8 if k > d else 0.6
        # Overbought conditions (stronger sell signal if K crosses below D)
        elif k > overbought and d > overbought: score = -0.8 if k < d else -0.6
        # Check thresholds individually
        elif k < oversold: score = 0.5
        elif k > overbought: score = -0.5
        # Check crossover in the middle range
        elif k > d: score = 0.2
        elif k < d: score = -0.2

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_rsi(self) -> Optional[float]:
        """Checks RSI for overbought/oversold levels. Score: -1.0 (OB) to 1.0 (OS), scaled in between, None (missing)."""
        rsi = self._get_indicator_float("RSI")
        if rsi is None: return None
        score = 0.0
        # Define thresholds (adjust as needed)
        ob_strong = 80.0; ob_weak = 70.0
        os_strong = 20.0; os_weak = 30.0

        if rsi <= os_strong: score = 1.0
        elif rsi <= os_weak: score = 0.7
        elif rsi >= ob_strong: score = -1.0
        elif rsi >= ob_weak: score = -0.7
        # Scale linearly between weak OS and weak OB thresholds
        elif os_weak < rsi < ob_weak:
            score = 1.0 - (rsi - os_weak) * (2.0 / (ob_weak - os_weak))

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_cci(self) -> Optional[float]:
        """Checks CCI for extreme levels. Score: -1.0 (High) to 1.0 (Low), scaled in between, None (missing)."""
        cci = self._get_indicator_float("CCI")
        if cci is None: return None
        score = 0.0
        # Define thresholds (common levels: +/- 100, +/- 200)
        ob_strong = 200.0; ob_weak = 100.0
        os_strong = -200.0; os_weak = -100.0

        if cci <= os_strong: score = 1.0
        elif cci <= os_weak: score = 0.7
        elif cci >= ob_strong: score = -1.0
        elif cci >= ob_weak: score = -0.7
        # Scale linearly between weak thresholds (optional, can be 0)
        elif os_weak < cci < ob_weak:
            # score = -(cci / ob_weak) * 0.3 # Example scaling: small score towards zero
            score = 0.0 # Or treat the middle range as neutral

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_wr(self) -> Optional[float]:
        """Checks Williams %R for overbought/oversold levels. Score: -1.0 (OB) to 1.0 (OS), scaled, None (missing). Note: WR is typically negative."""
        wr = self._get_indicator_float("WR")
        if wr is None: return None
        score = 0.0
        # Define thresholds (common levels: -20 OB, -80 OS)
        ob_strong = -10.0; ob_weak = -20.0
        os_strong = -90.0; os_weak = -80.0

        if wr <= os_strong: score = 1.0
        elif wr <= os_weak: score = 0.7
        elif wr >= ob_strong: score = -1.0
        elif wr >= ob_weak: score = -0.7
        # Scale linearly between weak thresholds
        elif os_weak < wr < ob_weak:
             score = 1.0 - (wr - os_weak) * (2.0 / (ob_weak - os_weak))

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_psar(self) -> Optional[float]:
        """Checks Parabolic SAR position relative to price. Score: 1.0 (below price - bullish), -1.0 (above price - bearish), 0.0 (transition/missing)."""
        # PSAR values are the SAR level itself. We compare with price.
        # PSAR_Long has a value when SAR is below price (potential long)
        # PSAR_Short has a value when SAR is above price (potential short)
        psar_l_val = self.indicator_values.get("PSAR_Long")
        psar_s_val = self.indicator_values.get("PSAR_Short")

        # Check if values exist and are finite Decimals
        psar_l_active = psar_l_val is not None and psar_l_val.is_finite()
        psar_s_active = psar_s_val is not None and psar_s_val.is_finite()

        if psar_l_active and not psar_s_active: return 1.0  # SAR is below price
        if psar_s_active and not psar_l_active: return -1.0 # SAR is above price
        # If both are active/inactive (shouldn't happen with standard PSAR) or values missing
        return 0.0

    def _check_sma10(self) -> Optional[float]:
        """Checks if price is above/below SMA10. Score: 0.5 (above), -0.5 (below), 0.0 (equal), None (missing)."""
        sma = self._get_indicator_float("SMA10")
        close = self._get_indicator_float("Close")
        if sma is None or close is None: return None
        if close > sma: return 0.5
        if close < sma: return -0.5
        return 0.0

    def _check_vwap(self) -> Optional[float]:
        """Checks if price is above/below VWAP. Score: 0.6 (above), -0.6 (below), 0.0 (equal), None (missing)."""
        vwap = self._get_indicator_float("VWAP")
        close = self._get_indicator_float("Close")
        if vwap is None or close is None: return None
        if close > vwap: return 0.6
        if close < vwap: return -0.6
        return 0.0

    def _check_mfi(self) -> Optional[float]:
        """Checks Money Flow Index for overbought/oversold. Score: -1.0 (OB) to 1.0 (OS), scaled, None (missing)."""
        mfi = self._get_indicator_float("MFI")
        if mfi is None: return None
        score = 0.0
        # Define thresholds (common: 80 OB, 20 OS)
        ob_strong = 85.0; ob_weak = 75.0
        os_strong = 15.0; os_weak = 25.0

        if mfi <= os_strong: score = 1.0
        elif mfi <= os_weak: score = 0.7
        elif mfi >= ob_strong: score = -1.0
        elif mfi >= ob_weak: score = -0.7
        # Scale linearly between weak thresholds
        elif os_weak < mfi < ob_weak:
            score = 1.0 - (mfi - os_weak) * (2.0 / (ob_weak - os_weak))

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_bollinger_bands(self) -> Optional[float]:
        """Checks price position relative to Bollinger Bands. Score: 1.0 (at/below lower), -1.0 (at/above upper), scaled in between, None (missing)."""
        bbl = self._get_indicator_float("BB_Lower")
        bbu = self._get_indicator_float("BB_Upper")
        close = self._get_indicator_float("Close")
        if bbl is None or bbu is None or close is None: return None
        score = 0.0

        if close <= bbl: score = 1.0   # Price touching or below lower band (potential buy)
        elif close >= bbu: score = -1.0 # Price touching or above upper band (potential sell)
        else:
            # Price is within the bands, scale score based on position
            band_range = bbu - bbl
            if band_range > 0:
                # Position = 0 near lower band, 1 near upper band
                position_in_band = (close - bbl) / band_range
                # Score = 1 near lower band, -1 near upper band
                score = 1.0 - 2.0 * position_in_band
            else:
                score = 0.0 # Bands are too close, neutral

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_orderbook(self, orderbook_data: Optional[Dict]) -> Optional[float]:
        """Calculates Order Book Imbalance (OBI). Score: -1.0 (Ask heavy) to 1.0 (Bid heavy), None (missing data)."""
        if not orderbook_data:
            # self.logger.debug(f"Order book data missing for {self.symbol}, cannot calculate OBI.")
            return None
        try:
            bids = orderbook_data.get('bids', []) # List of [price, volume]
            asks = orderbook_data.get('asks', []) # List of [price, volume]

            if not bids or not asks:
                # self.logger.debug(f"Empty bids or asks in order book data for {self.symbol}.")
                return 0.0 # Neutral if one side is empty

            # Consider top N levels based on config
            limit = int(self.config.get("orderbook_limit", 10))
            levels_to_consider = min(len(bids), len(asks), limit)

            if levels_to_consider <= 0:
                # self.logger.debug(f"No valid levels to consider in order book for {self.symbol}.")
                return 0.0 # Neutral if limit is 0 or no matching levels

            # Calculate total volume within the considered levels
            # Use Decimal for summation to maintain precision
            bid_volume = sum(Decimal(str(b[1])) for b in bids[:levels_to_consider])
            ask_volume = sum(Decimal(str(a[1])) for a in asks[:levels_to_consider])
            total_volume = bid_volume + ask_volume

            if total_volume <= 0:
                # self.logger.debug(f"Total volume in top {levels_to_consider} levels is zero for {self.symbol}.")
                return 0.0 # Neutral if no volume

            # Calculate Order Book Imbalance (OBI)
            obi = (bid_volume - ask_volume) / total_volume
            # OBI > 0: More bid volume (potential upward pressure)
            # OBI < 0: More ask volume (potential downward pressure)

            # Return as float, clamped between -1 and 1
            return float(max(Decimal("-1.0"), min(Decimal("1.0"), obi)))

        except (InvalidOperation, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error calculating Order Book Imbalance for {self.symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during OBI calculation for {self.symbol}: {e}", exc_info=True)
            return None


    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates a final trading signal ('BUY', 'SELL', 'HOLD') based on the
        weighted scores of enabled indicators.

        Args:
            current_price_dec: The current market price as Decimal.
            orderbook_data: Fetched order book data (if orderbook indicator is enabled).

        Returns:
            The final signal string: "BUY", "SELL", or "HOLD".
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Reset signals
        final_signal_score = Decimal("0.0") # Use Decimal for summing weighted scores
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores: Dict[str, str] = {} # For logging individual contributions

        # --- Input Validation ---
        if not self.indicator_values:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Indicator values not calculated.")
            return "HOLD"
        if not current_price_dec.is_finite() or current_price_dec <= 0:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price_dec}).")
            return "HOLD"
        if not self.weights:
            # Warning already logged in __init__
            return "HOLD"

        active_weights = self.weights # Use the loaded weights for the active set

        # Map indicator keys from config to their checking methods
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
            "sma_10": self._check_sma10, # Example addition
            "mfi": self._check_mfi,
            "orderbook": lambda: self._check_orderbook(orderbook_data), # Use lambda for methods needing args
        }

        # --- Iterate Through Enabled Indicators ---
        for indicator_key, is_enabled in self.config.get("indicators", {}).items():
            if not is_enabled:
                debug_scores[indicator_key] = "Disabled"
                continue

            # Get weight for this indicator from the active weight set
            weight_str = active_weights.get(indicator_key)
            if weight_str is None:
                # Indicator enabled but no weight assigned in the active set
                debug_scores[indicator_key] = "No Weight"
                continue

            # Convert weight to Decimal, handle errors
            try:
                weight = Decimal(str(weight_str))
                if not weight.is_finite(): raise ValueError("Weight is not finite")
            except (ValueError, InvalidOperation, TypeError):
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping.")
                debug_scores[indicator_key] = f"Invalid Wt({weight_str})"
                continue

            # Skip if weight is zero
            if weight == Decimal("0"):
                debug_scores[indicator_key] = "Wt=0"
                continue

            # Find and call the corresponding check method
            check_method = indicator_check_methods.get(indicator_key)
            if check_method:
                indicator_score_float: Optional[float] = None
                try:
                    # Call the check method (which returns float or None)
                    indicator_score_float = check_method()
                except Exception as e:
                    self.logger.error(f"Error executing check method for '{indicator_key}' on {self.symbol}: {e}", exc_info=True)
                    debug_scores[indicator_key] = "Check Error"
                    nan_indicator_count += 1
                    continue # Skip this indicator if check fails

                # Process the returned score
                if indicator_score_float is not None and math.isfinite(indicator_score_float):
                    try:
                        # Clamp score float to [-1.0, 1.0] before converting to Decimal
                        clamped_score_float = max(-1.0, min(1.0, indicator_score_float))
                        indicator_score_decimal = Decimal(str(clamped_score_float))

                        # Calculate weighted score and add to total
                        weighted_score = indicator_score_decimal * weight
                        final_signal_score += weighted_score
                        # Keep track of total absolute weight applied for normalization
                        total_weight_applied += abs(weight)
                        active_indicator_count += 1
                        # Store debug info
                        debug_scores[indicator_key] = f"{indicator_score_float:.2f} (x{weight:.2f}) = {weighted_score:.3f}"
                    except (InvalidOperation, TypeError) as calc_err:
                        self.logger.error(f"Error processing score/weight for {indicator_key}: Score={indicator_score_float}, Weight={weight}. Error: {calc_err}")
                        debug_scores[indicator_key] = "Calc Error"
                        nan_indicator_count += 1
                else:
                    # Indicator returned None or NaN/Inf
                    debug_scores[indicator_key] = "NaN/None"
                    nan_indicator_count += 1
            elif indicator_key in active_weights:
                # Indicator has weight but no check method defined
                self.logger.warning(f"Check method missing for enabled indicator with weight: '{indicator_key}'")
                debug_scores[indicator_key] = "No Method"

        # --- Determine Final Signal ---
        final_signal = "HOLD"
        normalized_score = Decimal("0.0")

        if total_weight_applied > 0:
            # Normalize the score by the sum of absolute weights used
            normalized_score = (final_signal_score / total_weight_applied).quantize(Decimal("0.0001"))
        elif active_indicator_count > 0:
            # This case means indicators were active, but all assigned weights were zero.
            self.logger.warning(f"Signal Calc ({self.symbol}): {active_indicator_count} indicators active, but total weight applied is zero. Defaulting to HOLD.")
        # If active_indicator_count is 0, it means no enabled indicators had valid scores or weights.

        # Get the appropriate threshold based on the active weight set
        threshold_key = "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
        default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
        try:
            threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
            if not threshold.is_finite() or threshold <= 0:
                raise ValueError("Threshold must be positive and finite")
        except (ValueError, InvalidOperation, TypeError):
            threshold = Decimal(str(default_threshold))
            self.logger.warning(f"Invalid or non-positive threshold value for '{threshold_key}' in config. Using default: {threshold}")

        # Compare final score against the threshold
        # Using final_signal_score (raw weighted sum) for threshold comparison
        # Using normalized_score primarily for logging/analysis might be better. Let's stick to raw score vs threshold.
        if final_signal_score >= threshold:
            final_signal = "BUY"
        elif final_signal_score <= -threshold:
            final_signal = "SELL"

        # Log the signal generation details
        price_prec = self.get_price_precision_digits()
        score_details_str = ", ".join([f"{k}: {v}" for k, v in debug_scores.items() if v not in ["Disabled", "No Weight", "Wt=0"]])
        log_msg = (
            f"Signal Calc ({self.symbol} @ {current_price_dec:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', "
            f"Indis(Actv/NaN): {active_indicator_count}/{nan_indicator_count}, "
            f"WtSum: {total_weight_applied:.3f}, "
            f"RawScore: {final_signal_score:.4f}, "
            f"NormScore: {normalized_score:.4f}, "
            f"Thresh: +/-{threshold:.3f} -> "
            f"Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        # Add more detailed score breakdown at DEBUG level if needed
        if nan_indicator_count > 0 or active_indicator_count == 0 or self.logger.level == logging.DEBUG:
             self.logger.debug(f"  Detailed Scores: {debug_scores}")


        # Update the internal signal state
        if final_signal in self.signals:
            self.signals[final_signal] = 1
            self.signals["HOLD"] = 1 if final_signal == "HOLD" else 0

        return final_signal


    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_signal: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates quantized Entry, Take Profit (TP), and Stop Loss (SL) prices.
        Uses ATR for dynamic distances and ensures results are quantized and valid.

        Args:
            entry_price_signal: The price near which the entry signal occurred (used as base).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            A tuple containing:
            - Quantized Entry Price (Decimal, or None if failed)
            - Quantized Take Profit Price (Decimal, or None if failed/disabled)
            - Quantized Stop Loss Price (Decimal, or None if failed/disabled)
        """
        quantized_entry: Optional[Decimal] = None
        take_profit: Optional[Decimal] = None
        stop_loss: Optional[Decimal] = None

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid signal '{signal}' for TP/SL calculation ({self.symbol}).")
            return None, None, None
        if not entry_price_signal.is_finite() or entry_price_signal <= 0:
            self.logger.error(f"Invalid entry_price_signal ({entry_price_signal}) for TP/SL calculation ({self.symbol}).")
            return None, None, None

        # --- Quantize Entry Price ---
        # Quantize the signal price to a valid market price.
        # Using simple ROUND_DOWN for buys and ROUND_UP for sells might slightly improve entry,
        # but simple quantization is safer initially. Let's stick to simple ROUND_DOWN.
        quantized_entry = self.quantize_price(entry_price_signal, rounding=ROUND_DOWN)
        if quantized_entry is None:
            self.logger.error(f"Failed to quantize entry price {entry_price_signal} for {self.symbol}.")
            return None, None, None

        # --- Get ATR and Multipliers ---
        atr_val = self.indicator_values.get("ATR")
        if atr_val is None or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate dynamic TP/SL for {self.symbol}: Invalid or missing ATR value ({atr_val}). SL/TP will be None.{RESET}")
            # Return quantized entry, but None for SL/TP
            return quantized_entry, None, None

        try:
            atr = atr_val
            # Get multipliers from config, converting safely to Decimal
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
            min_tick = self.get_min_tick_size()
            if min_tick is None:
                self.logger.error(f"Cannot calculate TP/SL for {self.symbol}: Min tick size unavailable.")
                return quantized_entry, None, None # Cannot proceed without tick size

            # --- Calculate Offsets ---
            tp_offset = atr * tp_mult
            sl_offset = atr * sl_mult

            # Ensure SL offset is at least the minimum required ticks away from entry
            min_sl_offset_value = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
            if sl_offset < min_sl_offset_value:
                self.logger.warning(f"Calculated SL offset ({sl_offset:.{self.get_price_precision_digits()}f}) based on ATR*Multiplier is less than minimum required {MIN_TICKS_AWAY_FOR_SLTP} ticks ({min_sl_offset_value:.{self.get_price_precision_digits()}f}). Adjusting SL offset to minimum.")
                sl_offset = min_sl_offset_value

            # --- Calculate Raw TP/SL ---
            if signal == "BUY":
                tp_raw = quantized_entry + tp_offset
                sl_raw = quantized_entry - sl_offset
                # Quantize TP UP (further away), SL DOWN (further away)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_UP)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_DOWN)
            else: # SELL
                tp_raw = quantized_entry - tp_offset
                sl_raw = quantized_entry + sl_offset
                # Quantize TP DOWN (further away), SL UP (further away)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_DOWN)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_UP)

            # --- Post-Calculation Validation ---
            # Validate Stop Loss
            if stop_loss is not None:
                # Ensure SL is strictly beyond entry by >= MIN_TICKS_AWAY_FOR_SLTP ticks
                min_dist_from_entry = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
                is_too_close = False
                if signal == "BUY" and stop_loss >= quantized_entry - min_dist_from_entry + min_tick: # Use >= and add one tick buffer for safety
                     is_too_close = True
                     adjusted_sl = self.quantize_price(quantized_entry - min_dist_from_entry, rounding=ROUND_DOWN)
                elif signal == "SELL" and stop_loss <= quantized_entry + min_dist_from_entry - min_tick: # Use <= and subtract one tick buffer
                     is_too_close = True
                     adjusted_sl = self.quantize_price(quantized_entry + min_dist_from_entry, rounding=ROUND_UP)

                if is_too_close:
                    self.logger.warning(f"Initial SL ({stop_loss}) too close to entry ({quantized_entry}). Adjusting to be at least {MIN_TICKS_AWAY_FOR_SLTP} ticks away.")
                    stop_loss = adjusted_sl # Use the adjusted value

                # Final check: SL must be positive
                if stop_loss is not None and stop_loss <= 0:
                    self.logger.error(f"Calculated Stop Loss is zero or negative ({stop_loss}) for {self.symbol}. Setting SL to None.")
                    stop_loss = None

            # Validate Take Profit
            if take_profit is not None:
                 # Ensure TP is strictly beyond entry by at least 1 tick (less strict than SL)
                 is_too_close = False
                 if signal == "BUY" and take_profit <= quantized_entry:
                     is_too_close = True
                     adjusted_tp = self.quantize_price(quantized_entry + min_tick, rounding=ROUND_UP)
                 elif signal == "SELL" and take_profit >= quantized_entry:
                     is_too_close = True
                     adjusted_tp = self.quantize_price(quantized_entry - min_tick, rounding=ROUND_DOWN)

                 if is_too_close:
                     self.logger.debug(f"Initial TP ({take_profit}) not beyond entry ({quantized_entry}). Adjusting by 1 tick.")
                     take_profit = adjusted_tp

                 # Final check: TP must be positive
                 if take_profit is not None and take_profit <= 0:
                     self.logger.error(f"Calculated Take Profit is zero or negative ({take_profit}) for {self.symbol}. Setting TP to None.")
                     take_profit = None

            # --- Logging ---
            prec = self.get_price_precision_digits()
            atr_log = f"{atr:.{prec+1}f}" # Show ATR with more precision
            tp_log = f"{take_profit:.{prec}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{prec}f}" if stop_loss else 'N/A'
            entry_log = f"{quantized_entry:.{prec}f}"
            self.logger.info(f"Calc TP/SL ({signal}): Entry={entry_log}, TP={tp_log}, SL={sl_log} (ATR={atr_log}, SL Mult={sl_mult}, TP Mult={tp_mult})")

            return quantized_entry, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error during TP/SL calculation value conversion for {self.symbol}: {e}{RESET}")
            return quantized_entry, None, None # Return entry, but SL/TP failed
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return quantized_entry, None, None


# --- Position Sizing ---
def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # Keep as float as it's a percentage
    entry_price: Decimal,
    stop_loss_price: Decimal,
    market_info: Dict,
    leverage: int, # Keep as int
    logger: logging.Logger
) -> Optional[Decimal]:
    """
    Calculates the position size in base currency units (e.g., BTC, ETH) or contracts.
    Uses Decimal for calculations involving price and balance.

    Args:
        balance: Available trading balance (in quote currency, e.g., USDT) as Decimal.
        risk_per_trade: Fraction of balance to risk (e.g., 0.01 for 1%).
        entry_price: Planned entry price as Decimal.
        stop_loss_price: Planned stop loss price as Decimal.
        market_info: Dictionary containing market details (precision, limits, type).
        leverage: Leverage to be used (relevant for margin calculation check).
        logger: Logger instance.

    Returns:
        The calculated and quantized position size as Decimal, or None if calculation fails or constraints are not met.
    """
    lg = logger
    symbol = market_info.get('symbol', 'N/A')
    contract_size: Decimal = market_info.get('contract_size', Decimal('1')) # Default 1 (e.g., for spot)
    min_order_amount: Optional[Decimal] = market_info.get('min_order_amount')
    min_order_cost: Optional[Decimal] = market_info.get('min_order_cost')
    amount_digits: Optional[int] = market_info.get('amount_precision_digits')
    is_contract: bool = market_info.get('is_contract', False)
    is_inverse: bool = market_info.get('inverse', False)

    # --- Input Validation ---
    if balance <= 0:
        lg.error(f"Size Calc Error ({symbol}): Balance is zero or negative ({balance}).")
        return None
    if entry_price <= 0 or stop_loss_price <= 0:
        lg.error(f"Size Calc Error ({symbol}): Entry price ({entry_price}) or Stop loss price ({stop_loss_price}) is invalid.")
        return None
    if entry_price == stop_loss_price:
        lg.error(f"Size Calc Error ({symbol}): Entry price cannot be equal to stop loss price.")
        return None
    if amount_digits is None:
        lg.error(f"Size Calc Error ({symbol}): Amount precision (digits) missing in market info.")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Error ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
        return None
    if is_contract and leverage <= 0:
        lg.error(f"Size Calc Error ({symbol}): Invalid leverage ({leverage}) provided for a contract.")
        return None

    try:
        # --- Calculate Risk Amount and SL Distance ---
        risk_amount_quote: Decimal = balance * Decimal(str(risk_per_trade)) # Amount of quote currency to risk
        sl_distance_points: Decimal = abs(entry_price - stop_loss_price) # SL distance in price points

        if sl_distance_points <= 0: # Should be caught by entry==sl check, but safety first
             lg.error(f"Size Calc Error ({symbol}): Stop loss distance is zero or negative.")
             return None

        # --- Calculate Unquantized Size ---
        size_unquantized = Decimal('NaN') # Initialize as Not a Number

        if is_contract:
            # For Futures/Swaps
            if is_inverse:
                # Inverse contracts (e.g., BTC/USD settled in BTC)
                # Risk per contract is value change in base currency
                # Value of 1 contract = contract_size / price (in base currency)
                # Change in value = contract_size * abs(1/entry - 1/sl)
                if entry_price == 0 or stop_loss_price == 0:
                     lg.error(f"Size Calc Error ({symbol}): Cannot calculate inverse contract size with zero price.")
                     return None
                risk_per_contract_base = contract_size * abs(Decimal('1') / entry_price - Decimal('1') / stop_loss_price)
                # Convert risk per contract to quote currency for comparison with risk_amount_quote
                # Use entry price for approximate conversion factor
                risk_per_contract_quote = risk_per_contract_base * entry_price
            else:
                # Linear contracts (e.g., BTC/USDT settled in USDT)
                # Risk per contract = SL distance * contract_size (in quote currency)
                risk_per_contract_quote = sl_distance_points * contract_size

            if risk_per_contract_quote <= 0:
                lg.error(f"Size Calc Error ({symbol}): Calculated risk per contract is zero or negative ({risk_per_contract_quote}).")
                return None
            # Size (in contracts) = Total Risk Amount / Risk Per Contract
            size_unquantized = risk_amount_quote / risk_per_contract_quote
        else:
            # For Spot
            # Risk per unit = SL distance (in quote currency)
            risk_per_unit_quote = sl_distance_points
            if risk_per_unit_quote <= 0: # Should not happen if sl_distance_points > 0
                lg.error(f"Size Calc Error ({symbol}): Spot risk per unit is zero or negative.")
                return None
            # Size (in base currency units) = Total Risk Amount / Risk Per Unit
            size_unquantized = risk_amount_quote / risk_per_unit_quote

        if not size_unquantized.is_finite() or size_unquantized <= 0:
            lg.error(f"Size Calc Error ({symbol}): Calculated unquantized size is invalid ({size_unquantized}). RiskAmt={risk_amount_quote}, SLDist={sl_distance_points}, RiskPerContr={risk_per_contract_quote if is_contract else risk_per_unit_quote}")
            return None

        lg.debug(f"Size Calc ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade*100:.2f}%, RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_points}, UnquantSize={size_unquantized:.8f}")

        # --- Quantize Size ---
        step_size = Decimal('1') / (Decimal('10') ** amount_digits)
        # Use ROUND_DOWN to not exceed risk budget
        quantized_size = (size_unquantized / step_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * step_size
        # Ensure correct number of decimal places in the final Decimal object
        quantized_size = Decimal(f"{quantized_size:.{amount_digits}f}")

        lg.debug(f"Quantized Size ({symbol}): {quantized_size} (Step: {step_size})")

        # --- Validate Against Market Limits ---
        if quantized_size <= 0:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size is zero after quantization. Cannot place trade.{RESET}")
            return None

        # Check minimum order amount
        if min_order_amount is not None and quantized_size < min_order_amount:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Calculated size {quantized_size} is less than minimum order amount {min_order_amount}. Cannot place trade.{RESET}")
            return None

        # --- Check Margin and Cost Limits ---
        # Calculate estimated order value in quote currency
        if is_inverse:
            # Value = Size (contracts) * contract_size / price (in base), convert to quote
            # Approximate value in quote: Size (contracts) * contract_size
            order_value_quote = quantized_size * contract_size
        else:
            # Linear or Spot: Value = Size (contracts/units) * price * contract_size (1 for spot)
            order_value_quote = quantized_size * entry_price * contract_size

        # Calculate estimated margin required
        margin_required = Decimal('0')
        if is_contract:
            if leverage > 0:
                margin_required = order_value_quote / Decimal(leverage)
            else: # Should be caught earlier, but safety check
                 lg.error(f"Size Calc Error ({symbol}): Leverage is zero, cannot calculate margin."); return None
        else: # Spot
            margin_required = order_value_quote # Full cost for spot

        # Check minimum order cost (value)
        if min_order_cost is not None and order_value_quote < min_order_cost:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Estimated order value {order_value_quote:.4f} {market_info.get('quote','Quote')} is less than minimum cost {min_order_cost}. Cannot place trade.{RESET}")
            return None

        # Check if margin required exceeds available balance
        # Use a small buffer (e.g., 0.1%) for fees/slippage if desired
        buffer_factor = Decimal("1.001")
        if margin_required * buffer_factor > balance:
            lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Estimated margin required {margin_required:.4f} (incl. buffer) exceeds available balance {balance:.4f}. Cannot place trade.{RESET}")
            return None

        # --- Success ---
        lg.info(f"Calculated position size for {symbol}: {quantized_size} {'contracts' if is_contract else market_info.get('base', 'units')}")
        return quantized_size

    except (InvalidOperation, ValueError, TypeError) as e:
         lg.error(f"{NEON_RED}Error during position size calculation value conversion for {symbol}: {e}{RESET}", exc_info=True)
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


# --- CCXT Trading Action Wrappers ---

def fetch_positions_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """
    Fetches the current open position for a specific symbol using V5 API.
    Filters for non-zero positions matching the symbol.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.
        market_info: Processed market information for the symbol.

    Returns:
        A dictionary containing the position details if an active, non-zero position exists,
        otherwise None. Includes standardized 'side', 'contracts', and original 'info'.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID

    # Position fetching only relevant for derivatives
    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping position check for non-derivative symbol: {symbol}")
        return None
    # Check if the exchange instance supports fetching positions
    if not exchange.has.get('fetchPositions'):
        lg.error(f"Exchange {exchange.id} does not support fetchPositions(). Cannot check position for {symbol}.")
        return None

    try:
        # Parameters for Bybit V5 fetchPositions
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Call fetch_positions, specifying the symbol to potentially filter server-side
        # Note: Bybit V5 might still return all positions for the category, so client-side filtering is needed.
        all_positions = safe_ccxt_call(exchange, 'fetch_positions', lg, symbols=[symbol], params=params)

        if all_positions is None:
            lg.warning(f"fetch_positions returned None for {symbol}.")
            return None
        if not isinstance(all_positions, list):
            lg.error(f"fetch_positions did not return a list for {symbol}. Response type: {type(all_positions)}")
            return None

        # Iterate through the returned positions to find the matching symbol with non-zero size
        for pos in all_positions:
            # CCXT standardizes the symbol key
            position_symbol = pos.get('symbol')
            if position_symbol == symbol:
                try:
                     # Get position size - Bybit V5 uses 'size' in 'info', CCXT might map to 'contracts'
                     pos_size_str = pos.get('contracts', pos.get('info', {}).get('size'))
                     if pos_size_str is None:
                         lg.warning(f"Position data for {symbol} missing size/contracts field. Data: {pos}")
                         continue # Skip this position entry

                     pos_size = Decimal(str(pos_size_str))

                     # Check if position size is non-zero (ignore closed/empty positions)
                     if pos_size != Decimal('0'):
                          # Determine side ('long' or 'short') based on size sign
                          # Bybit V5: Positive size for long, negative for short
                          pos_side = 'long' if pos_size > 0 else 'short'

                          # Standardize the position dictionary returned
                          standardized_pos = pos.copy() # Work on a copy
                          standardized_pos['side'] = pos_side
                          # Store the absolute size under 'contracts' for consistency
                          standardized_pos['contracts'] = abs(pos_size)
                          # Ensure entry price is Decimal
                          entry_price_str = standardized_pos.get('entryPrice', pos.get('info',{}).get('avgPrice'))
                          if entry_price_str is not None:
                               standardized_pos['entryPrice'] = Decimal(str(entry_price_str))
                          else: standardized_pos['entryPrice'] = None

                          # Add market info for convenience in downstream functions
                          standardized_pos['market_info'] = market_info

                          lg.info(f"Found active {pos_side} position for {symbol}: Size={abs(pos_size)}, Entry={standardized_pos.get('entryPrice')}")
                          return standardized_pos
                     # else: lg.debug(f"Position found for {symbol} but size is 0.")

                except (InvalidOperation, ValueError, TypeError) as e:
                     lg.error(f"Could not parse position data for {symbol}: {e}. Data: {pos}")
                except Exception as e:
                     lg.error(f"Unexpected error processing position entry for {symbol}: {e}. Data: {pos}", exc_info=True)

        # If loop completes without finding a matching non-zero position
        lg.debug(f"No active non-zero position found for {symbol}.")
        return None

    except Exception as e:
        # Catch errors during the safe_ccxt_call or processing
        lg.error(f"{NEON_RED}Error fetching or processing positions for {symbol}: {e}{RESET}", exc_info=True)
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger, market_info: Dict) -> bool:
    """
    Sets the leverage for a given symbol using Bybit V5 API.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        leverage: The desired leverage (integer).
        logger: Logger instance.
        market_info: Processed market information for the symbol.

    Returns:
        True if leverage was set successfully (or was already set correctly), False otherwise.
    """
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID

    # Leverage is only applicable to derivatives
    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping leverage setting for non-derivative symbol: {symbol}")
        return True # Consider it success as no action is needed

    # Check if the exchange instance supports setting leverage
    if not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage(). Cannot set leverage for {symbol}.")
        return False

    if leverage <= 0:
        lg.error(f"Invalid leverage value ({leverage}) provided for {symbol}. Must be positive.")
        return False

    try:
        # Parameters for Bybit V5 setLeverage
        # Requires setting both buy and sell leverage for the symbol
        params = {
            'category': category,
            'symbol': market_id,
            'buyLeverage': str(leverage), # Must be string
            'sellLeverage': str(leverage) # Must be string
        }
        lg.info(f"Setting leverage for {symbol} (MarketID: {market_id}) to {leverage}x...")
        lg.debug(f"Leverage Params: {params}")

        # Use safe_ccxt_call to handle potential errors
        result = safe_ccxt_call(exchange, 'set_leverage', lg, leverage=leverage, symbol=symbol, params=params)

        # safe_ccxt_call handles the "leverage not modified" (110043) error by returning {},
        # which evaluates as True here, indicating success or no change needed.
        if result is not None: # Check if call returned (even if empty dict for 110043)
            lg.info(f"{NEON_GREEN}Leverage set successfully (or already correct) for {symbol} to {leverage}x.{RESET}")
            return True
        else:
            # Should not happen if safe_ccxt_call works correctly, but as a safeguard
            lg.error(f"{NEON_RED}set_leverage call returned None unexpectedly for {symbol}.{RESET}")
            return False

    except ccxt.ExchangeError as e:
         # Catch errors not handled by safe_ccxt_call (e.g., if 110043 handling changes)
         # or non-retryable errors raised by it.
         lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
         return False
    except Exception as e:
        # Catch any other unexpected errors
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
        return False


def create_order_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    order_type: str, # 'market', 'limit'
    side: str,       # 'buy', 'sell'
    amount: Decimal, # Order quantity as Decimal
    price: Optional[Decimal] = None, # Required for limit orders, as Decimal
    params: Optional[Dict] = None,   # Additional parameters for the API call
    logger: Optional[logging.Logger] = None,
    market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Creates an order (market or limit) using safe_ccxt_call.
    Handles V5 parameters, Decimal to float conversion for CCXT, and basic validation.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        order_type: 'market' or 'limit'.
        side: 'buy' or 'sell'.
        amount: Order quantity as Decimal.
        price: Order price as Decimal (required for limit orders).
        params: Dictionary of additional parameters for the exchange API call (e.g., 'reduceOnly', 'positionIdx').
        logger: Logger instance.
        market_info: Processed market information for the symbol.

    Returns:
        The order dictionary returned by CCXT if successful, otherwise None.
    """
    lg = logger or get_logger('main') # Use provided logger or main logger
    if not market_info:
        lg.error(f"Market info is required for create_order_ccxt ({symbol})")
        return None

    category = market_info.get('category')
    market_id = market_info.get('id', symbol) # Use exchange-specific ID

    # --- Input Validation ---
    if not category:
        lg.error(f"Cannot determine category for {symbol}. Cannot place order.")
        return None
    if amount <= 0:
        lg.error(f"Order amount must be positive ({symbol}, Amount: {amount})")
        return None
    order_type_lower = order_type.lower()
    if order_type_lower not in ['market', 'limit']:
        lg.error(f"Invalid order type '{order_type}' for {symbol}. Use 'market' or 'limit'.")
        return None
    if side.lower() not in ['buy', 'sell']:
         lg.error(f"Invalid order side '{side}' for {symbol}. Use 'buy' or 'sell'.")
         return None

    # --- Prepare Price and Amount Strings ---
    price_digits = market_info.get('price_precision_digits', 8)
    amount_digits = market_info.get('amount_precision_digits', 8)

    # Format amount to required precision string
    amount_str = f"{amount:.{amount_digits}f}"

    # Format price only if it's a limit order and price is valid
    price_str: Optional[str] = None
    if order_type_lower == 'limit':
        if price is None or not price.is_finite() or price <= 0:
            lg.error(f"Valid positive price is required for a limit order ({symbol}). Price provided: {price}")
            return None
        price_str = f"{price:.{price_digits}f}"

    # --- Prepare Parameters ---
    # Base parameters required for Bybit V5
    order_params: Dict[str, Any] = {'category': category}

    # --- TODO: Hedge Mode Logic ---
    # If Hedge Mode is enabled in config, determine positionIdx based on side.
    # position_mode = config.get("position_mode", "One-Way") # Assuming config is accessible
    # if position_mode == "Hedge":
    #     # For opening orders in Hedge Mode:
    #     # idx 1 = Hedge Mode Buy position
    #     # idx 2 = Hedge Mode Sell position
    #     order_params['positionIdx'] = 1 if side.lower() == 'buy' else 2
    #     lg.debug(f"Hedge Mode: Setting positionIdx={order_params['positionIdx']} for new {side} order.")
    # else:
    #     # For One-Way Mode or closing orders:
    #     order_params['positionIdx'] = 0 # Default for One-Way mode
    # --- End Hedge Mode Logic ---

    # Merge any additional parameters passed in
    if params:
        order_params.update(params)
        # Ensure positionIdx from params overrides the default if provided explicitly
        if 'positionIdx' in params:
             lg.debug(f"Using explicitly provided positionIdx={params['positionIdx']} from params.")


    # --- Convert Decimals to Floats for CCXT ---
    # CCXT methods typically expect float types for amount and price.
    # Perform this conversion just before the API call.
    try:
        amount_float = float(amount_str)
        price_float = float(price_str) if price_str else None
    except ValueError as e:
        lg.error(f"Error converting amount/price to float for CCXT call ({symbol}): {e}")
        return None

    # --- Place Order via safe_ccxt_call ---
    try:
        log_price_part = f'@ {price_str}' if price_str else 'at Market'
        lg.info(f"Attempting to create {side.upper()} {order_type.upper()} order: {amount_str} {symbol} {log_price_part}")
        lg.debug(f"Final Order Params for CCXT call: Symbol={symbol}, Type={order_type}, Side={side}, Amount={amount_float}, Price={price_float}, Params={order_params}")

        order_result = safe_ccxt_call(
            exchange,
            'create_order',
            lg,
            symbol=symbol,         # CCXT standard symbol
            type=order_type,       # 'market' or 'limit'
            side=side,             # 'buy' or 'sell'
            amount=amount_float,   # Pass float amount
            price=price_float,     # Pass float price (or None)
            params=order_params    # Pass V5 specific params here
        )

        # --- Process Result ---
        if order_result and isinstance(order_result, dict) and order_result.get('id'):
            # Basic check for success: presence of an order ID
            order_id = order_result['id']
            # Check Bybit's return code in the info dictionary for confirmation
            ret_code = order_result.get('info', {}).get('retCode', -1) # Default to -1 if not found
            ret_msg = order_result.get('info', {}).get('retMsg', 'Unknown Status')

            if ret_code == 0:
                 lg.info(f"{NEON_GREEN}Successfully created {side.upper()} {order_type.upper()} order for {symbol}. Order ID: {order_id}{RESET}")
                 # Optional: Add more details from order_result['info'] if needed
                 lg.debug(f"Order Result Info: {order_result.get('info')}")
                 return order_result
            else:
                 # Order ID might be present even if rejected (e.g., insufficient balance after checks)
                 lg.error(f"{NEON_RED}Order placement potentially rejected by exchange ({symbol}). Code={ret_code}, Msg='{ret_msg}'. Order ID: {order_id}{RESET}")
                 # Consider returning None or the result depending on how failures should be handled
                 return None # Treat non-zero retCode as failure
        elif order_result:
             # Call succeeded but response format is unexpected (e.g., missing ID)
             lg.error(f"Order API call successful but response missing Order ID ({symbol}). Response: {order_result}")
             return None
        else:
             # safe_ccxt_call returned None (max retries reached or critical error)
             lg.error(f"Order API call failed or returned None ({symbol}) after retries.")
             return None

    except Exception as e:
        lg.error(f"{NEON_RED}Failed to create order ({symbol}) due to unexpected error: {e}{RESET}", exc_info=True)
        return None


def set_protection_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_price: Optional[Decimal] = None, # Note: This is TSL *distance* for Bybit V5, not trigger price
    trailing_active_price: Optional[Decimal] = None, # Activation price for TSL
    position_idx: int = 0, # Required for Hedge mode (1 for Buy Hedge, 2 for Sell Hedge)
    logger: Optional[logging.Logger] = None,
    market_info: Optional[Dict] = None
) -> bool:
    """
    Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop (TSL) for a position
    using the Bybit V5 `position/trading-stop` endpoint via a private POST call.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        stop_loss_price: The desired SL price (Decimal). Set to 0 or None to remove.
        take_profit_price: The desired TP price (Decimal). Set to 0 or None to remove.
        trailing_stop_price: The desired TSL *distance* from entry/activation (Decimal). Set to 0 or None to remove.
        trailing_active_price: The price at which the TSL should activate (Decimal). Set to 0 or None for immediate activation if TSL distance is set.
        position_idx: Position index for Hedge Mode (0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge).
        logger: Logger instance.
        market_info: Processed market information for the symbol.

    Returns:
        True if the protection was set successfully, False otherwise.
    """
    lg = logger or get_logger('main')
    if not market_info:
        lg.error(f"Market info required for set_protection_ccxt ({symbol})")
        return False

    category = market_info.get('category')
    market_id = market_info.get('id', symbol)
    price_digits = market_info.get('price_precision_digits', 8)

    # Protection setting is primarily for derivatives
    if not category or category not in ['linear', 'inverse']:
        lg.warning(f"Cannot set SL/TP/TSL for non-derivative symbol {symbol}. Category: {category}")
        # Return False as the action cannot be performed for this market type.
        return False

    # --- Prepare Parameters for Bybit V5 position/trading-stop ---
    params: Dict[str, Any] = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full', # Options: 'Full' (SL&TP on whole position), 'Partial' (requires qty)
        # Set TP/SL/TSL values. Use "0" to clear existing ones.
    }

    # --- Hedge Mode Logic ---
    # position_mode = config.get("position_mode", "One-Way") # Assuming config access
    # if position_mode == "Hedge":
    #     if position_idx not in [1, 2]:
    #         lg.error(f"Hedge Mode Error: Invalid positionIdx ({position_idx}) provided for set_protection on {symbol}. Must be 1 (Buy) or 2 (Sell).")
    #         # TODO: Decide how to handle this - maybe try to fetch current positionIdx? Risky.
    #         # For now, default to 0, which will likely fail if the account *is* in Hedge Mode.
    #         params['positionIdx'] = 0
    #     else:
    #          params['positionIdx'] = position_idx
    #          lg.debug(f"Hedge Mode: Setting protection with positionIdx={position_idx}")
    # else: # One-Way Mode
    #     params['positionIdx'] = 0
    # --- End Hedge Mode Logic ---
    # Pass position_idx directly from function arg for now
    params['positionIdx'] = position_idx


    # Helper to format price values to string with correct precision, or "0" if None/invalid
    def format_price(price: Optional[Decimal]) -> str:
        if price is not None and price.is_finite() and price > 0:
            return f"{price:.{price_digits}f}"
        else:
            return "0" # Use "0" to clear/disable the specific protection

    sl_str = format_price(stop_loss_price)
    tp_str = format_price(take_profit_price)
    # Note: Bybit API expects trailingStop as the distance value, not a price level
    tsl_dist_str = format_price(trailing_stop_price)
    tsl_act_str = format_price(trailing_active_price)

    params['stopLoss'] = sl_str
    params['takeProfit'] = tp_str
    params['trailingStop'] = tsl_dist_str # This is the distance value

    # activePrice is only needed if trailingStop distance is set (non-zero)
    if tsl_dist_str != "0":
        params['activePrice'] = tsl_act_str # "0" means activate immediately

    # --- Logging ---
    log_parts = []
    if sl_str != "0": log_parts.append(f"SL={sl_str}")
    if tp_str != "0": log_parts.append(f"TP={tp_str}")
    if tsl_dist_str != "0":
        tsl_log = f"TSL_Dist={tsl_dist_str}"
        if tsl_act_str != "0": tsl_log += f", ActPrice={tsl_act_str}"
        else: tsl_log += ", Act=Immediate"
        log_parts.append(tsl_log)

    if not log_parts:
        lg.warning(f"No valid protection levels provided for set_protection_ccxt ({symbol}). No API call made.")
        # Consider returning True as no action was needed/failed.
        return True

    try:
        lg.info(f"Attempting to set protection for {symbol} (MarketID: {market_id}, PosIdx: {position_idx}): {', '.join(log_parts)}")
        lg.debug(f"Protection Params: {params}")

        # This endpoint requires a private POST call. CCXT might need explicit method name.
        # Check CCXT Bybit overrides or use the direct method name if needed.
        # Example: exchange.private_post_position_trading_stop(params)
        # Using safe_ccxt_call with the expected method name mapping from initialize_exchange
        method_to_call = 'private_post_position_trading_stop' # Check if this maps correctly in your CCXT version/config

        result = safe_ccxt_call(exchange, method_to_call, lg, params=params)

        # --- Process Result ---
        if result and isinstance(result, dict) and result.get('retCode') == 0:
            lg.info(f"{NEON_GREEN}Successfully set protection for {symbol}.{RESET}")
            lg.debug(f"Protection Result Info: {result}")
            return True
        elif result:
            # API call returned, but indicated failure
            ret_code = result.get('retCode', -1)
            ret_msg = result.get('retMsg', 'Unknown Error')
            lg.error(f"{NEON_RED}Failed to set protection ({symbol}). Code={ret_code}, Msg='{ret_msg}'{RESET}")
            lg.debug(f"Protection Failure Info: {result}")
            return False
        else:
            # safe_ccxt_call returned None (max retries or critical error)
            lg.error(f"Set protection API call failed or returned None ({symbol}) after retries.")
            return False

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
        return False


def close_position_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    position_data: Dict, # Position details from fetch_positions_ccxt
    logger: Optional[logging.Logger] = None,
    market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Closes an existing position by placing a Market order in the opposite direction.
    Uses the 'reduceOnly' parameter to ensure it only closes the position.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        position_data: Dictionary containing details of the position to close (must include 'side', 'contracts').
        logger: Logger instance.
        market_info: Processed market information for the symbol.

    Returns:
        The order dictionary of the closing market order if successfully placed, otherwise None.
    """
    lg = logger or get_logger('main')
    if not market_info:
        lg.error(f"Market info required for close_position_ccxt ({symbol})")
        return None
    if not position_data:
        lg.error(f"Position data required for close_position_ccxt ({symbol})")
        return None

    try:
        # Extract necessary details from position data
        position_side = position_data.get('side') # 'long' or 'short'
        # Use 'contracts' which should be the absolute size from fetch_positions_ccxt
        position_size_dec = position_data.get('contracts')

        if position_side not in ['long', 'short']:
            lg.error(f"Missing or invalid 'side' in position data for closing {symbol}. Data: {position_data}")
            return None
        if not isinstance(position_size_dec, Decimal) or position_size_dec <= 0:
             lg.error(f"Missing or invalid 'contracts' (size) in position data for closing {symbol}. Size: {position_size_dec}")
             return None

        amount_to_close: Decimal = position_size_dec # Size is already absolute

        # Determine the side for the closing order
        close_side = 'sell' if position_side == 'long' else 'buy'

        lg.info(f"Attempting to close {position_side} position ({symbol}, Size: {amount_to_close}) via {close_side.upper()} MARKET order...")

        # Parameters for the closing order
        close_params: Dict[str, Any] = {
            'reduceOnly': True # Crucial to ensure this order only reduces/closes the position
        }

        # --- TODO: Hedge Mode Logic ---
        # If Hedge Mode, need to specify the correct positionIdx to close.
        # position_mode = config.get("position_mode", "One-Way") # Assuming config access
        # if position_mode == "Hedge":
        #      # Get the positionIdx from the position data (should be in 'info')
        #      idx = position_data.get('info', {}).get('positionIdx')
        #      if idx in [1, 2]: # Bybit uses 1 for Buy Hedge, 2 for Sell Hedge
        #          close_params['positionIdx'] = idx
        #          lg.debug(f"Hedge Mode: Closing position with positionIdx={idx}")
        #      else:
        #          lg.error(f"Hedge Mode Error: Could not determine positionIdx for closing {symbol}. Found: {idx}. Aborting close.")
        #          return None
        # else: # One-Way Mode
        #     close_params['positionIdx'] = 0 # Default for One-Way
        # --- End Hedge Mode Logic ---
        # For now, assume One-Way (positionIdx=0 is default in create_order_ccxt if not specified)


        # Use create_order_ccxt to place the closing market order
        close_order_result = create_order_ccxt(
            exchange=exchange,
            symbol=symbol,
            order_type='market',
            side=close_side,
            amount=amount_to_close, # Pass the Decimal amount
            params=close_params,
            logger=lg,
            market_info=market_info
        )

        # Check if the closing order was placed successfully
        if close_order_result and close_order_result.get('id'):
            lg.info(f"{NEON_GREEN}Successfully placed MARKET order to close {position_side} position ({symbol}). Close Order ID: {close_order_result.get('id')}{RESET}")
            return close_order_result
        else:
            lg.error(f"{NEON_RED}Failed to place market order to close position ({symbol}). Check previous logs.{RESET}")
            return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Error processing position data for closing ({symbol}): {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error attempting to close position ({symbol}): {e}{RESET}", exc_info=True)
        return None


# --- Main Bot Logic ---
def run_bot(exchange: ccxt.Exchange, config: Dict[str, Any], bot_state: Dict[str, Any]) -> None:
    """
    The main execution loop of the trading bot.
    Iterates through configured symbols, fetches data, analyzes, manages positions,
    and attempts new entries based on signals and configuration.

    Args:
        exchange: Initialized CCXT exchange object.
        config: Loaded configuration dictionary.
        bot_state: Loaded or initialized state dictionary.
    """
    main_logger = get_logger('main')
    main_logger.info(f"{NEON_CYAN}=== Starting Enhanced Trading Bot v{BOT_VERSION} ==="
                     f" (PID: {os.getpid()}) ==={RESET}")

    # Log key configuration settings at startup
    trading_status = f"{NEON_GREEN}Enabled{RESET}" if config.get('enable_trading') else f"{NEON_YELLOW}DISABLED{RESET}"
    sandbox_status = f"{NEON_YELLOW}ACTIVE{RESET}" if config.get('use_sandbox') else f"{NEON_RED}INACTIVE (LIVE!){RESET}"
    account_type = "UNIFIED" if IS_UNIFIED_ACCOUNT else "Non-UTA (Contract/Spot)"
    main_logger.info(f"Mode: Trading={trading_status}, Sandbox={sandbox_status}, Account={account_type}")
    main_logger.info(f"Config: Symbols={config.get('symbols')}, Interval={config.get('interval')}, "
                     f"Quote={QUOTE_CURRENCY}, WeightSet='{config.get('active_weight_set')}'")
    main_logger.info(f"Risk: {config.get('risk_per_trade')*100:.2f}%, Leverage={config.get('leverage')}x, "
                     f"MaxPos={config.get('max_concurrent_positions_total')}")
    main_logger.info(f"Features: TSL={'On' if config.get('enable_trailing_stop') else 'Off'}, "
                     f"BE={'On' if config.get('enable_break_even') else 'Off'}, "
                     f"MACrossExit={'On' if config.get('enable_ma_cross_exit') else 'Off'}")

    global LOOP_DELAY_SECONDS
    LOOP_DELAY_SECONDS = config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS) # Already validated

    symbols_to_trade: List[str] = config.get("symbols", []) # Already validated

    # Initialize/Validate state dictionary for each symbol
    for symbol in symbols_to_trade:
        if symbol not in bot_state:
            bot_state[symbol] = {} # Create state dict if missing
        # Ensure default state keys exist for each symbol
        bot_state[symbol].setdefault("break_even_triggered", False)
        bot_state[symbol].setdefault("last_signal", "HOLD")
        # 'last_entry_price' is stored as string for Decimal precision via JSON
        bot_state[symbol].setdefault("last_entry_price", None)

    cycle_count = 0
    last_market_reload_time = exchange.last_load_markets_timestamp if hasattr(exchange, 'last_load_markets_timestamp') else 0

    # --- Main Loop ---
    while True:
        cycle_count += 1
        start_time = time.time()
        main_logger.info(f"{NEON_BLUE}--- Starting Bot Cycle {cycle_count} ---{RESET}")

        # --- Periodic Market Reload ---
        if time.time() - last_market_reload_time > MARKET_RELOAD_INTERVAL_SECONDS:
            main_logger.info(f"Reloading exchange markets (interval: {MARKET_RELOAD_INTERVAL_SECONDS}s)...")
            try:
                exchange.load_markets(True) # Force reload
                last_market_reload_time = time.time()
                exchange.last_load_markets_timestamp = last_market_reload_time # Update timestamp if attribute exists
                main_logger.info("Markets reloaded successfully.")
            except Exception as e:
                main_logger.error(f"Failed to reload markets: {e}", exc_info=True)
                # Continue running, but market data might be stale

        # --- Pre-Cycle Checks ---
        # 1. Fetch Current Balance (if trading enabled)
        current_balance: Optional[Decimal] = None
        if config.get("enable_trading"):
            try:
                current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
                if current_balance is None:
                    main_logger.error(f"{NEON_RED}Failed to fetch balance for {QUOTE_CURRENCY}. Trading actions requiring balance will be skipped this cycle.{RESET}")
                # else: main_logger.debug(f"Current Balance: {current_balance:.4f} {QUOTE_CURRENCY}")
            except Exception as e:
                main_logger.error(f"Unhandled error fetching balance: {e}", exc_info=True)
                # Continue cycle, but trading actions might fail

        # 2. Fetch All Active Positions
        open_positions_count = 0
        active_positions: Dict[str, Dict] = {} # Store active positions: {symbol: position_data}
        main_logger.debug("Fetching positions for all configured symbols...")
        for symbol in symbols_to_trade:
            # Use a temporary logger for this specific symbol's fetch
            temp_logger = get_logger(symbol, is_symbol_logger=True)
            try:
                market_info = get_market_info(exchange, symbol, temp_logger)
                if not market_info:
                    temp_logger.error(f"Cannot fetch position for {symbol}: Failed to get market info.")
                    continue # Skip this symbol for position check

                # Only fetch positions for contract types
                if market_info.get('is_contract'):
                    position = fetch_positions_ccxt(exchange, symbol, temp_logger, market_info)
                    if position: # fetch_positions_ccxt returns only non-zero positions
                        open_positions_count += 1
                        active_positions[symbol] = position
                        # Update state's entry price if it's missing and available in fetched position
                        if bot_state[symbol].get("last_entry_price") is None and position.get('entryPrice'):
                             try:
                                 # Store as string for JSON compatibility
                                 bot_state[symbol]["last_entry_price"] = str(position['entryPrice'])
                                 temp_logger.info(f"State updated: Set last_entry_price from fetched position: {bot_state[symbol]['last_entry_price']}")
                             except Exception as e:
                                 temp_logger.warning(f"Could not update last_entry_price from fetched position: {e}")
                # else: temp_logger.debug(f"Skipping position fetch for non-contract symbol: {symbol}")
            except Exception as fetch_pos_err:
                 temp_logger.error(f"Error during position fetch pre-check for {symbol}: {fetch_pos_err}", exc_info=True)

        max_allowed_positions = config.get("max_concurrent_positions_total", 1)
        main_logger.info(f"Currently open positions: {open_positions_count} / {max_allowed_positions}")

        # --- Symbol Processing Loop ---
        for symbol in symbols_to_trade:
            symbol_logger = get_logger(symbol, is_symbol_logger=True) # Get dedicated logger for the symbol
            symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
            symbol_state = bot_state[symbol] # Get reference to this symbol's state

            try:
                # 1. Get Market Info (again, ensure it's up-to-date if markets reloaded)
                market_info = get_market_info(exchange, symbol, symbol_logger)
                if not market_info:
                    symbol_logger.error(f"Failed to get market info for {symbol}. Skipping symbol this cycle.")
                    continue

                # 2. Fetch Data (Klines, Price, Orderbook)
                timeframe = config.get("interval", "5") # Already validated
                # Dynamically calculate required kline limit based on longest indicator period
                # Find max period from relevant config values
                periods = [int(cfg_val) for key, cfg_val in config.items() if ('_period' in key or '_window' in key) and isinstance(cfg_val, (int, float)) and cfg_val > 0]
                base_limit = max(periods) if periods else 100 # Use max period or a default
                kline_limit = base_limit + 50 # Add buffer for indicator calculation stability
                symbol_logger.debug(f"Required kline limit based on indicator periods: {kline_limit} (Base: {base_limit})")

                df_raw = fetch_klines_ccxt(exchange, symbol, timeframe, kline_limit, symbol_logger, market_info)
                if df_raw.empty or len(df_raw) < base_limit // 2: # Check if significantly less data than expected
                    symbol_logger.warning(f"Kline data for {symbol} is empty or insufficient ({len(df_raw)} rows). Skipping analysis.")
                    continue

                # Fetch current price after klines
                current_price_dec = fetch_current_price_ccxt(exchange, symbol, symbol_logger, market_info)
                if current_price_dec is None:
                    symbol_logger.warning(f"Current price unavailable for {symbol}. Skipping analysis.")
                    continue

                # Fetch order book if indicator is enabled
                orderbook = None
                if config.get("indicators", {}).get("orderbook"):
                    try:
                        ob_limit = int(config.get("orderbook_limit", 25))
                        orderbook = fetch_orderbook_ccxt(exchange, symbol, ob_limit, symbol_logger, market_info)
                    except Exception as ob_err:
                        symbol_logger.warning(f"Failed to fetch order book data for {symbol}: {ob_err}")

                # 3. Initialize Analyzer (Calculates indicators)
                try:
                    analyzer = TradingAnalyzer(df_raw, symbol_logger, config, market_info, symbol_state)
                except ValueError as analyze_init_err:
                    symbol_logger.error(f"Failed to initialize TradingAnalyzer for {symbol}: {analyze_init_err}. Skipping.")
                    continue
                except Exception as analyze_err:
                     symbol_logger.error(f"Unexpected error initializing TradingAnalyzer for {symbol}: {analyze_err}", exc_info=True)
                     continue


                # 4. Manage Existing Position (if any)
                current_position = active_positions.get(symbol)
                position_closed_in_manage = False
                if current_position:
                    pos_side = current_position.get('side')
                    pos_size_str = current_position.get('contracts', '?') # Should be Decimal
                    symbol_logger.info(f"Managing existing {pos_side} position (Size: {pos_size_str}).")

                    # Pass control to position management function
                    position_closed_in_manage = manage_existing_position(
                        exchange, config, symbol_logger, analyzer,
                        current_position, current_price_dec
                    )
                    # If position was closed, update state immediately
                    if position_closed_in_manage:
                        active_positions.pop(symbol, None) # Remove from active list
                        open_positions_count = max(0, open_positions_count - 1) # Decrement count
                        # State resets (BE, entry price) should happen inside manage_existing_position on close
                        symbol_logger.info(f"Position for {symbol} was closed during management.")
                        # Continue to the next symbol after closing
                        symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
                        time.sleep(0.2) # Small delay before next symbol
                        continue # Skip new entry check for this symbol this cycle


                # 5. Check for New Entry Signal (Only if no position OR position was just closed AND limits allow)
                # Check if still no position (covers case where manage didn't close) AND within limits
                if not active_positions.get(symbol) and market_info.get('is_contract'): # Only contracts for now
                    if open_positions_count < max_allowed_positions:
                        symbol_logger.info("No active position. Checking for new entry signals...")
                        # Reset state vars if entering check after no position exists
                        if analyzer.break_even_triggered:
                            analyzer.break_even_triggered = False # Reset BE state if no position
                        if symbol_state.get("last_entry_price") is not None:
                            symbol_state["last_entry_price"] = None # Clear last entry price if no position

                        # Generate signal using the analyzer
                        signal = analyzer.generate_trading_signal(current_price_dec, orderbook)
                        symbol_state["last_signal"] = signal # Store latest signal attempt in state

                        # Act on BUY/SELL signals
                        if signal in ["BUY", "SELL"]:
                            if config.get("enable_trading"):
                                if current_balance is not None and current_balance > 0:
                                     # Attempt to open the new position
                                     opened_new = attempt_new_entry(
                                         exchange, config, symbol_logger, analyzer,
                                         signal, current_price_dec, current_balance
                                     )
                                     if opened_new:
                                         open_positions_count += 1 # Increment count ONLY if successful
                                         # State (last_entry_price, BE reset) updated inside attempt_new_entry
                                else:
                                     symbol_logger.warning(f"Trading enabled but balance is zero or unavailable ({current_balance}). Cannot enter {signal} trade for {symbol}.")
                            else:
                                # Log signal even if trading is disabled
                                symbol_logger.info(f"Entry signal '{signal}' generated for {symbol}, but trading is disabled.")
                    else:
                         # Log if max positions reached
                         symbol_logger.info(f"Max concurrent positions ({open_positions_count}/{max_allowed_positions}) reached. Skipping new entry check for {symbol}.")

                # --- Placeholder for Spot Market Logic ---
                # elif not market_info.get('is_contract'):
                #      symbol_logger.debug(f"Spot market {symbol}. Entry/Management logic TBD.")
                #      # Add spot specific signal generation, order placement etc. here if needed

            except Exception as e:
                # Catch any unexpected error during the processing of a single symbol
                symbol_logger.error(f"{NEON_RED}!!! Unhandled error in main loop for symbol {symbol}: {e} !!!{RESET}", exc_info=True)
                # Continue to the next symbol

            finally:
                # Log end of processing for this symbol
                symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
                # Optional small delay between processing each symbol to reduce rapid logging/API load
                time.sleep(0.2)

        # --- Post-Cycle ---
        end_time = time.time()
        cycle_duration = end_time - start_time
        main_logger.info(f"{NEON_BLUE}--- Bot Cycle {cycle_count} Finished (Duration: {cycle_duration:.2f}s) ---{RESET}")

        # Save the current state after each full cycle
        save_state(STATE_FILE, bot_state, main_logger)

        # Calculate wait time for the next cycle
        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0:
            main_logger.info(f"Waiting {wait_time:.2f}s for next cycle...")
            time.sleep(wait_time)
        else:
            # Log if the cycle took longer than the configured delay
            main_logger.warning(f"Cycle duration ({cycle_duration:.2f}s) exceeded loop delay ({LOOP_DELAY_SECONDS}s). Starting next cycle immediately.")


def manage_existing_position(
    exchange: ccxt.Exchange,
    config: Dict[str, Any],
    logger: logging.Logger,
    analyzer: TradingAnalyzer, # Provides indicators, state, quantization
    position_data: Dict,      # Current position details
    current_price_dec: Decimal # Current market price
) -> bool:
    """
    Manages an existing open position.
    - Checks for MA Cross exit condition.
    - Checks and triggers Break-Even adjustment.
    - (TSL is handled by the exchange after being set).

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance for the symbol.
        analyzer: TradingAnalyzer instance for the symbol.
        position_data: Dictionary containing details of the open position.
        current_price_dec: Current market price as Decimal.

    Returns:
        bool: True if the position was closed by this function, False otherwise.
    """
    symbol = position_data.get('symbol')
    position_side = position_data.get('side') # 'long' or 'short'
    entry_price = position_data.get('entryPrice') # Should be Decimal from fetch_positions
    pos_size = position_data.get('contracts') # Should be Decimal from fetch_positions
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state # Access shared state via analyzer

    # --- Validate Input ---
    if not all([symbol, position_side, isinstance(entry_price, Decimal), isinstance(pos_size, Decimal)]):
        logger.error(f"Incomplete or invalid position data received for management: {position_data}")
        return False
    if pos_size <= 0:
        logger.warning(f"Position size is zero or negative ({pos_size}) for {symbol} in management function. Skipping.")
        return False
    if not current_price_dec.is_finite() or current_price_dec <= 0:
         logger.warning(f"Invalid current price ({current_price_dec}) for managing position {symbol}. Skipping management checks.")
         return False

    position_closed = False # Flag to indicate if position was closed here

    try:
        # --- 1. Check MA Cross Exit ---
        if config.get("enable_ma_cross_exit"):
            ema_s_f = analyzer._get_indicator_float("EMA_Short")
            ema_l_f = analyzer._get_indicator_float("EMA_Long")

            if ema_s_f is not None and ema_l_f is not None:
                # Add a small tolerance to prevent exits on exact equality or tiny fluctuations
                tolerance = 0.0001 # 0.01% tolerance
                is_adverse_cross = False
                cross_msg = ""

                if position_side == 'long' and ema_s_f < ema_l_f * (1 - tolerance):
                    is_adverse_cross = True
                    cross_msg = f"Long Exit: Short EMA ({ema_s_f:.5f}) crossed below Long EMA ({ema_l_f:.5f})"
                elif position_side == 'short' and ema_s_f > ema_l_f * (1 + tolerance):
                    is_adverse_cross = True
                    cross_msg = f"Short Exit: Short EMA ({ema_s_f:.5f}) crossed above Long EMA ({ema_l_f:.5f})"

                if is_adverse_cross:
                    logger.warning(f"{NEON_YELLOW}MA Cross Exit Triggered! {cross_msg}. Attempting to close position {symbol}.{RESET}")
                    if config.get("enable_trading"):
                        close_result = close_position_ccxt(exchange, symbol, position_data, logger, market_info)
                        if close_result:
                            logger.info(f"Position close order placed successfully for {symbol} due to MA Cross.")
                            # Reset state immediately after successful close order placement
                            symbol_state["break_even_triggered"] = False
                            symbol_state["last_signal"] = "HOLD" # Reset signal state
                            symbol_state["last_entry_price"] = None # Clear entry price
                            position_closed = True # Mark position as closed
                            return True # Exit management logic for this cycle
                        else:
                            logger.error(f"Failed to place MA Cross close order for {symbol}. Position may still be open.")
                            # Do not return yet, allow BE check maybe? Or maybe should return to retry close next cycle?
                            # Let's return False, assuming retry next cycle is better.
                            return False
                    else:
                        logger.info(f"MA Cross exit triggered for {symbol}, but trading is disabled. No action taken.")
                        # Don't return True, as position wasn't actually closed

        # --- 2. Check Break-Even Trigger (Only if not already triggered and position wasn't just closed) ---
        if not position_closed and config.get("enable_break_even") and not analyzer.break_even_triggered:
            atr_val = analyzer.indicator_values.get("ATR")
            if atr_val and atr_val.is_finite() and atr_val > 0:
                try:
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    profit_target_points = atr_val * trigger_multiple # Profit distance required

                    # Calculate current profit in price points
                    current_profit_points = Decimal('0')
                    if position_side == 'long':
                        current_profit_points = current_price_dec - entry_price
                    else: # Short position
                        current_profit_points = entry_price - current_price_dec

                    # Check if profit target is reached
                    if current_profit_points >= profit_target_points:
                        logger.info(f"{NEON_GREEN}Break-Even Trigger Condition Met for {symbol}! "
                                    f"Current Profit Pts ({current_profit_points:.{analyzer.get_price_precision_digits()}f}) "
                                    f">= Target Pts ({profit_target_points:.{analyzer.get_price_precision_digits()}f}){RESET}")

                        min_tick = analyzer.get_min_tick_size()
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))

                        if min_tick and min_tick > 0 and offset_ticks >= 0:
                            offset_value = min_tick * Decimal(offset_ticks)
                            be_stop_price_raw = Decimal('0')

                            # Calculate BE stop price slightly in profit
                            if position_side == 'long':
                                be_stop_price_raw = entry_price + offset_value
                            else: # Short
                                be_stop_price_raw = entry_price - offset_value

                            # Quantize the BE stop price (round towards safety - further from current price)
                            rounding_mode = ROUND_DOWN if position_side == 'long' else ROUND_UP
                            be_stop_price = analyzer.quantize_price(be_stop_price_raw, rounding=rounding_mode)

                            # Sanity check: Ensure BE stop is actually beyond entry after quantization
                            if be_stop_price is not None:
                                if position_side == 'long' and be_stop_price <= entry_price:
                                    be_stop_price = analyzer.quantize_price(entry_price + min_tick, rounding=ROUND_UP) # Move at least 1 tick above
                                elif position_side == 'short' and be_stop_price >= entry_price:
                                    be_stop_price = analyzer.quantize_price(entry_price - min_tick, rounding=ROUND_DOWN) # Move at least 1 tick below

                            if be_stop_price and be_stop_price.is_finite() and be_stop_price > 0:
                                logger.info(f"Calculated Break-Even Stop Price for {symbol}: {be_stop_price}")

                                if config.get("enable_trading"):
                                    # Get current TP/TSL from position info if available
                                    pos_info = position_data.get('info', {})
                                    tp_str = pos_info.get('takeProfit', "0")
                                    tsl_dist_str = pos_info.get('trailingStop', "0")
                                    tsl_act_str = pos_info.get('activePrice', "0")

                                    current_tp = Decimal(tp_str) if tp_str and tp_str != "0" else None
                                    current_tsl_dist = Decimal(tsl_dist_str) if tsl_dist_str and tsl_dist_str != "0" else None
                                    current_tsl_act = Decimal(tsl_act_str) if tsl_act_str and tsl_act_str != "0" else None

                                    # Decide whether to keep TSL active based on config
                                    force_fixed_sl = config.get("break_even_force_fixed_sl", True)
                                    use_tsl = config.get("enable_trailing_stop") and not force_fixed_sl

                                    tsl_to_set = current_tsl_dist if use_tsl and current_tsl_dist else None
                                    act_to_set = current_tsl_act if use_tsl and tsl_to_set else None
                                    tp_to_set = current_tp # Keep existing TP

                                    log_parts_be = [f"BE SL={be_stop_price}"]
                                    if tp_to_set: log_parts_be.append(f"TP={tp_to_set}")
                                    if tsl_to_set: log_parts_be.append(f"TSL={tsl_to_set}")
                                    if force_fixed_sl and current_tsl_dist: log_parts_be.append("(Removing TSL)")
                                    logger.info(f"Attempting to set protection: {', '.join(log_parts_be)}")

                                    pos_idx = pos_info.get('positionIdx', 0) # Get index for Hedge Mode if applicable

                                    # Call set_protection_ccxt to update SL (and potentially remove TSL)
                                    success = set_protection_ccxt(
                                        exchange, symbol,
                                        stop_loss_price=be_stop_price, # Set the new BE SL
                                        take_profit_price=tp_to_set,   # Keep existing TP
                                        trailing_stop_price=tsl_to_set, # Set TSL distance (None if removing)
                                        trailing_active_price=act_to_set,# Set TSL activation (None if removing)
                                        position_idx=pos_idx,
                                        logger=logger,
                                        market_info=market_info
                                    )

                                    if success:
                                        logger.info(f"{NEON_GREEN}Successfully set Break-Even Stop Loss for {symbol}.{RESET}")
                                        # Update the shared state via the analyzer's property
                                        analyzer.break_even_triggered = True
                                    else:
                                        logger.error(f"{NEON_RED}Failed to set Break-Even SL via API for {symbol}. State not updated.{RESET}")
                                else:
                                     logger.info(f"Break-Even triggered for {symbol}, but trading is disabled. No API call made.")
                            else:
                                logger.error(f"Invalid Break-Even stop price calculated ({be_stop_price}) for {symbol}. Cannot set BE SL.")
                        else:
                            logger.error(f"Cannot calculate Break-Even offset for {symbol}: Invalid min_tick ({min_tick}) or offset_ticks ({offset_ticks}).")
                except (InvalidOperation, ValueError, TypeError) as be_calc_err:
                     logger.error(f"Error during Break-Even calculation value conversion for {symbol}: {be_calc_err}")
                except Exception as be_err:
                    logger.error(f"Unexpected error during Break-Even check for {symbol}: {be_err}", exc_info=True)
            else:
                logger.warning(f"Cannot check Break-Even trigger for {symbol}: Invalid or missing ATR value ({atr_val}).")

    except Exception as e:
        logger.error(f"Unexpected error managing position {symbol}: {e}", exc_info=True)
        # Return False as we don't know if the position was closed
        return False

    # Return the status indicating if the position was closed within this function
    return position_closed


def attempt_new_entry(
    exchange: ccxt.Exchange,
    config: Dict[str, Any],
    logger: logging.Logger,
    analyzer: TradingAnalyzer, # Provides indicators, state, quantization, TP/SL calc
    signal: str,               # "BUY" or "SELL"
    entry_price_signal: Decimal, # Price near signal generation
    current_balance: Decimal   # Current available balance
) -> bool:
    """
    Attempts to execute a new trade entry:
    1. Calculates TP/SL based on the signal price and ATR.
    2. Calculates position size based on risk and SL distance.
    3. Sets leverage (for contracts).
    4. Places the market entry order.
    5. Waits and confirms the actual entry price (optional but recommended).
    6. Sets SL, TP, and potentially TSL protection on the position.
    7. Updates bot state (last entry price, reset BE trigger).

    Args:
        exchange: Initialized CCXT exchange object.
        config: Bot configuration dictionary.
        logger: Logger instance for the symbol.
        analyzer: TradingAnalyzer instance.
        signal: The entry signal ("BUY" or "SELL").
        entry_price_signal: The approximate price when the signal occurred.
        current_balance: Current available quote currency balance.

    Returns:
        bool: True if the entry process (order placement and protection setting) was successful, False otherwise.
    """
    symbol = analyzer.symbol
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state # Access shared state

    logger.info(f"Attempting {signal} entry for {symbol} based on signal near price {entry_price_signal:.{analyzer.get_price_precision_digits()}f}")

    # --- 1. Calculate Quantized Entry, TP, SL ---
    # Use the analyzer's method which handles quantization and validation
    quantized_entry, take_profit_price, stop_loss_price = analyzer.calculate_entry_tp_sl(entry_price_signal, signal)

    if not quantized_entry:
        logger.error(f"Cannot enter {signal} ({symbol}): Failed to calculate/quantize entry price.")
        return False
    if not stop_loss_price:
        # SL is crucial for risk management and size calculation
        logger.error(f"Cannot enter {signal} ({symbol}): Failed to calculate a valid Stop Loss price.")
        return False
    # TP is optional, proceed even if TP is None

    # --- 2. Calculate Position Size ---
    risk = float(config.get("risk_per_trade", 0.01))
    leverage = int(config.get("leverage", 10))
    position_size = calculate_position_size(
        balance=current_balance,
        risk_per_trade=risk,
        entry_price=quantized_entry, # Use the calculated entry price
        stop_loss_price=stop_loss_price, # Use the calculated SL price
        market_info=market_info,
        leverage=leverage,
        logger=logger
    )

    if not position_size or position_size <= 0:
        logger.error(f"Cannot enter {signal} ({symbol}): Position size calculation failed or resulted in zero/negative size ({position_size}).")
        return False

    # --- 3. Set Leverage (Only for Contracts) ---
    if market_info.get('is_contract'):
        logger.info(f"Setting leverage to {leverage}x for {symbol} before entry...")
        if not set_leverage_ccxt(exchange, symbol, leverage, logger, market_info):
            logger.error(f"Failed to set leverage {leverage}x for {symbol}. Aborting entry.")
            return False

    # --- 4. Place Entry Order ---
    side = 'buy' if signal == 'BUY' else 'sell'
    entry_order_params: Dict[str, Any] = {}
    # --- TODO: Add positionIdx for Hedge Mode if needed ---
    # position_mode = config.get("position_mode", "One-Way")
    # if position_mode == "Hedge": entry_order_params['positionIdx'] = 1 if side == 'buy' else 2
    # else: entry_order_params['positionIdx'] = 0
    # ---

    # Place the market order using the wrapper function
    entry_order = create_order_ccxt(
        exchange=exchange,
        symbol=symbol,
        order_type='market',
        side=side,
        amount=position_size, # Pass Decimal amount
        params=entry_order_params,
        logger=logger,
        market_info=market_info
    )

    if not entry_order or not entry_order.get('id'):
        logger.error(f"Failed to place entry market order for {symbol}. Aborting entry process.")
        # Note: A partial fill might have occurred if the API call failed after sending.
        # Robust handling might involve checking positions again here.
        return False

    order_id = entry_order['id']
    logger.info(f"Entry market order ({order_id}) placed successfully for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for fill confirmation...")
    time.sleep(POSITION_CONFIRM_DELAY)

    # --- 5. Fetch Actual Entry Price & Confirm Position ---
    actual_entry_price = quantized_entry # Default to calculated price
    filled_size = position_size # Assume full fill initially
    position_idx_filled = 0 # Default for One-Way or if fetch fails
    try:
        logger.debug(f"Attempting to re-fetch position for {symbol} to confirm entry details...")
        # Add a small extra delay before fetching position
        time.sleep(1.5)
        updated_position = fetch_positions_ccxt(exchange, symbol, logger, market_info)

        if updated_position and updated_position.get('entryPrice'):
             entry_p_fetched = updated_position.get('entryPrice') # Should be Decimal
             current_size_fetched = updated_position.get('contracts') # Should be Decimal

             if isinstance(entry_p_fetched, Decimal) and entry_p_fetched.is_finite() and entry_p_fetched > 0:
                 actual_entry_price = entry_p_fetched
             else: logger.warning(f"Fetched position has invalid entry price ({entry_p_fetched}). Using calculated entry: {actual_entry_price}.")

             if isinstance(current_size_fetched, Decimal) and current_size_fetched.is_finite():
                 filled_size = abs(current_size_fetched) # Use absolute value
             else: logger.warning(f"Fetched position has invalid size ({current_size_fetched}). Using ordered size: {filled_size}.")

             # Get positionIdx for Hedge mode protection setting
             position_idx_filled = updated_position.get('info', {}).get('positionIdx', 0)

             logger.info(f"Position Confirmed: Actual Entry={actual_entry_price:.{analyzer.get_price_precision_digits()}f}, Filled Size={filled_size}, PosIdx={position_idx_filled}")

             # Check if filled size significantly deviates from ordered size
             size_diff_ratio = abs(filled_size - position_size) / position_size if position_size > 0 else Decimal('1')
             if size_diff_ratio > Decimal('0.05'): # Allow 5% deviation for partial fills/fees?
                 logger.warning(f"Filled size {filled_size} deviates significantly (>5%) from ordered size {position_size}.")
                 # TODO: Potential handling for significant partial fills? (e.g., adjust SL/TP based on actual fill?)
        else:
             logger.warning(f"Could not confirm entry details by re-fetching position for {symbol}. Using calculated entry price ({actual_entry_price}) and ordered size ({filled_size}). Protection might be based on estimated entry.")

    except Exception as confirm_err:
        logger.error(f"Error confirming entry position details for {symbol}: {confirm_err}. Using calculated entry price.", exc_info=True)

    # --- 6. Calculate and Set SL/TP/TSL Protection ---
    tsl_distance: Optional[Decimal] = None
    tsl_activation_price: Optional[Decimal] = None

    if config.get("enable_trailing_stop"):
        try:
            # Get TSL parameters from config, convert safely
            cb_rate_str = str(config.get("trailing_stop_callback_rate"))
            act_perc_str = str(config.get("trailing_stop_activation_percentage"))
            cb_rate = Decimal(cb_rate_str)
            act_perc = Decimal(act_perc_str)
            min_tick = analyzer.get_min_tick_size()

            if cb_rate > 0 and min_tick:
                # Calculate TSL distance based on callback rate and actual entry price
                tsl_dist_raw = actual_entry_price * cb_rate
                # Quantize the distance UP to the nearest tick size (safer)
                tsl_distance = (tsl_dist_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                # Ensure distance is at least one tick
                if tsl_distance < min_tick:
                    tsl_distance = min_tick
                    logger.debug(f"TSL distance adjusted to minimum tick size: {tsl_distance} for {symbol}")

                # Calculate activation price if percentage is set
                if act_perc > 0:
                    profit_offset = actual_entry_price * act_perc
                    act_raw = Decimal('0')
                    rounding_mode = ROUND_UP # Default rounding
                    if signal == "BUY":
                        act_raw = actual_entry_price + profit_offset
                        rounding_mode = ROUND_UP # Activate further in profit
                    else: # SELL
                        act_raw = actual_entry_price - profit_offset
                        rounding_mode = ROUND_DOWN # Activate further in profit

                    tsl_activation_price = analyzer.quantize_price(act_raw, rounding=rounding_mode)

                    # Validate activation price
                    if tsl_activation_price is not None:
                        if tsl_activation_price <= 0:
                            tsl_activation_price = None # Invalid activation price
                            logger.warning(f"Calculated TSL activation price is zero or negative for {symbol}. Disabling activation price.")
                        # Ensure activation price is actually beyond entry price after quantization
                        elif signal == "BUY" and tsl_activation_price <= actual_entry_price:
                            tsl_activation_price = analyzer.quantize_price(actual_entry_price + min_tick, ROUND_UP)
                            logger.debug(f"Adjusted TSL activation price to be > entry for BUY: {tsl_activation_price}")
                        elif signal == "SELL" and tsl_activation_price >= actual_entry_price:
                            tsl_activation_price = analyzer.quantize_price(actual_entry_price - min_tick, ROUND_DOWN)
                            logger.debug(f"Adjusted TSL activation price to be < entry for SELL: {tsl_activation_price}")
                # If act_perc is 0, tsl_activation_price remains None, meaning immediate activation if tsl_distance is set

                logger.debug(f"Calculated TSL Params ({symbol}): Distance={tsl_distance}, Activation Price={tsl_activation_price} (Based on Entry: {actual_entry_price})")
            elif not min_tick:
                logger.warning(f"Cannot calculate TSL distance for {symbol}: Minimum tick size unavailable.")
            elif cb_rate <= 0:
                logger.debug(f"TSL callback rate is zero or negative ({cb_rate}) for {symbol}. TSL distance not set.")

        except (InvalidOperation, ValueError, TypeError) as tsl_calc_err:
             logger.error(f"Error converting TSL config values for {symbol}: {tsl_calc_err}. TSL may not be set.")
        except Exception as tsl_err:
            logger.error(f"Error calculating TSL parameters for {symbol}: {tsl_err}", exc_info=True)

    # Set the protection using the calculated/fetched values
    protection_set = set_protection_ccxt(
        exchange=exchange,
        symbol=symbol,
        stop_loss_price=stop_loss_price,    # Initial SL
        take_profit_price=take_profit_price, # Initial TP
        trailing_stop_price=tsl_distance,    # TSL distance (or None)
        trailing_active_price=tsl_activation_price, # TSL activation (or None)
        position_idx=position_idx_filled, # Use the index confirmed from position fetch (important for Hedge)
        logger=logger,
        market_info=market_info
    )

    # --- Final Check and State Update ---
    if not protection_set:
        logger.error(f"{NEON_RED}CRITICAL: Failed to set initial SL/TP/TSL protection for {symbol} after placing entry order {order_id}! Position might be unprotected.{RESET}")
        # --- Emergency Action (Optional but Recommended) ---
        # If protection fails, consider immediately closing the position to avoid risk.
        if config.get("enable_trading"): # Only if trading is actually enabled
            logger.warning(f"Attempting emergency market close of unprotected position {symbol} (Order ID: {order_id})...")
            # Fetch the position again to ensure we have the latest data to close
            pos_to_close = fetch_positions_ccxt(exchange, symbol, logger, market_info)
            if pos_to_close:
                close_result = close_position_ccxt(exchange, symbol, pos_to_close, logger, market_info)
                if close_result: logger.info(f"Emergency close order placed for {symbol}.")
                else: logger.error(f"Failed to place emergency close order for {symbol}!")
            else:
                logger.error(f"Could not fetch position details for emergency close of {symbol}. Manual intervention may be required!")
        # --- End Emergency Action ---
        return False # Entry process failed at protection stage

    # --- Success ---
    logger.info(f"{NEON_GREEN}Successfully entered {signal} trade for {symbol} with initial protection set.{RESET}")
    # Update state: Reset BE trigger and store actual entry price (as string)
    symbol_state["break_even_triggered"] = False
    symbol_state["last_entry_price"] = str(actual_entry_price)
    return True


# --- Main Execution Guard ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description=f"Enhanced Bybit V5 Multi-Symbol Trading Bot v{BOT_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE,
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--state", type=str, default=STATE_FILE,
        help="Path to the JSON state file (for saving/loading bot state)."
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Override symbols from config file (comma-separated list, e.g., 'BTC/USDT:USDT,ETH/USDT:USDT')."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable LIVE trading mode. Overrides 'enable_trading' and 'use_sandbox' in config file. USE WITH EXTREME CAUTION!"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG level logging for console output."
    )
    args = parser.parse_args()

    # --- Initial Setup ---
    # Set console log level based on --debug flag
    console_log_level = logging.DEBUG if args.debug else logging.INFO
    if args.debug:
        print(f"{NEON_YELLOW}DEBUG logging enabled for console output.{RESET}")

    # Setup main logger (will respect console_log_level)
    main_logger = get_logger('main')
    main_logger.info(f" --- Bot Starting --- {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} --- ")

    # --- Load Configuration ---
    config = load_config(args.config, main_logger)
    if config is None:
        main_logger.critical("Configuration loading or validation failed. Exiting.")
        sys.exit(1)

    # --- Apply Command-Line Overrides ---
    # Override symbols if provided
    if args.symbols:
        try:
            override_symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
            if not override_symbols: raise ValueError("Symbol list cannot be empty.")
            # Basic validation of override format
            if not all('/' in s for s in override_symbols): raise ValueError("Invalid symbol format in list.")
            main_logger.warning(f"{NEON_YELLOW}Overriding config symbols via command line. Trading ONLY: {override_symbols}{RESET}")
            config["symbols"] = override_symbols
        except ValueError as e:
             main_logger.error(f"Invalid --symbols argument: '{args.symbols}'. Error: {e}. Using symbols from config file.")
        except Exception as e:
             main_logger.error(f"Error parsing --symbols argument: {e}. Using symbols from config file.")


    # Override trading/sandbox mode if --live flag is set
    if args.live:
        main_logger.warning(f"{NEON_RED}--- LIVE TRADING ENABLED via --live command line flag! ---{RESET}")
        if not config.get("enable_trading"):
            main_logger.warning("Overriding config setting 'enable_trading=false'.")
        if config.get("use_sandbox"):
            main_logger.warning("Overriding config setting 'use_sandbox=true'.")
        config["enable_trading"] = True
        config["use_sandbox"] = False

    # Log final effective trading/sandbox mode
    if config.get("enable_trading"):
        main_logger.warning(f"{NEON_RED}--- TRADING ACTIONS ARE ENABLED ---{RESET}")
    else:
        main_logger.info("--- Trading actions are DISABLED (simulation mode) ---")

    if config.get("use_sandbox"):
        main_logger.warning(f"{NEON_YELLOW}--- SANDBOX MODE (Testnet) is ACTIVE ---{RESET}")
    else:
        main_logger.warning(f"{NEON_RED}--- LIVE EXCHANGE MODE is ACTIVE ---{RESET}")
        if not config.get("enable_trading"):
            main_logger.info("--- NOTE: Live exchange mode active, but trading actions are disabled. ---")


    # --- Load State ---
    bot_state = load_state(args.state, main_logger)

    # --- Initialize Exchange ---
    exchange = initialize_exchange(config, main_logger)

    # --- Run Bot ---
    exit_code = 0
    if exchange:
        main_logger.info(f"{NEON_GREEN}Exchange initialized successfully. Starting main bot loop...{RESET}")
        try:
            # Start the main bot logic loop
            run_bot(exchange, config, bot_state)
        except KeyboardInterrupt:
            main_logger.info("Bot stopped by user (KeyboardInterrupt).")
        except Exception as e:
            # Catch any unhandled exceptions from the main loop
            main_logger.critical(f"{NEON_RED}!!! BOT CRASHED UNEXPECTEDLY: {e} !!!{RESET}", exc_info=True)
            exit_code = 1 # Indicate error exit
        finally:
            # --- Shutdown Procedures ---
            main_logger.info("Initiating bot shutdown sequence...")
            # Save the final state before exiting
            main_logger.info("Attempting to save final state...")
            save_state(args.state, bot_state, main_logger)
            main_logger.info(f"--- Bot Shutdown Complete --- {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    else:
        # Exchange initialization failed
        main_logger.critical("Failed to initialize the exchange. Bot cannot start.")
        exit_code = 1 # Indicate error exit

    # Ensure all logs are written
    logging.shutdown()
    print("Bot execution finished.")
    sys.exit(exit_code) # Exit with appropriate code
