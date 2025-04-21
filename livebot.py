```python
# livebot.py
# Enhanced and Upgraded Scalping Bot Framework (Derived from sxs.py)
# Focuses on robust execution, error handling, advanced position management (BE, TSL),
# and Bybit V5 API compatibility for Linear Perpetual contracts.

import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo # Use zoneinfo for modern timezone handling

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialization ---
init(autoreset=True) # Ensure colorama resets styles automatically
load_dotenv() # Load environment variables from .env file

# Set Decimal precision (increased for potentially complex calculations)
getcontext().prec = 36

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Log error and raise immediately if keys are missing
    init_logger = logging.getLogger("InitCheck")
    init_logger.addHandler(logging.StreamHandler()) # Simple handler for startup errors
    init_logger.setLevel(logging.CRITICAL)
    init_logger.critical(f"{NEON_RED}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET environment variables are not set.")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")  # Use IANA timezone database names
MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 7  # Increased default delay
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable

# Default indicator periods (can be overridden by config.json) - Standardized Names
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_PERIOD = 20 # Renamed from WINDOW
DEFAULT_WILLIAMS_R_PERIOD = 14 # Renamed from WINDOW
DEFAULT_MFI_PERIOD = 14 # Renamed from WINDOW
DEFAULT_STOCH_RSI_PERIOD = 14 # Renamed from WINDOW
DEFAULT_STOCH_RSI_RSI_PERIOD = 14 # Renamed from STOCH_WINDOW (underlying RSI)
DEFAULT_STOCH_RSI_K_PERIOD = 3 # Renamed from K_WINDOW
DEFAULT_STOCH_RSI_D_PERIOD = 3 # Renamed from D_WINDOW
DEFAULT_RSI_PERIOD = 14 # Renamed from WINDOW
DEFAULT_BBANDS_PERIOD = 20
DEFAULT_BBANDS_STDDEV = 2.0
DEFAULT_SMA10_PERIOD = 10 # Renamed from WINDOW
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_PERIOD = 50 # Renamed from WINDOW
DEFAULT_PSAR_STEP = 0.02 # Renamed from AF
DEFAULT_PSAR_MAX_STEP = 0.2 # Renamed from MAX_AF

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 10 # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY_SECONDS = 10 # Wait time after placing order before confirming position
# QUOTE_CURRENCY is loaded dynamically from config

os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Global dictionary to store position entry timestamps (symbol -> timestamp)
position_entry_times = {}

# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Use placeholders for safety, even if keys change during runtime (unlikely)
        if API_KEY and API_KEY in msg:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET and API_SECRET in msg:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file, creating default if not found,
    ensuring all default keys are present with validation, and saving updates.
    """
    # Define the default configuration structure and values
    default_config = {
        # Trading pair and timeframe
        "symbol": "BTC/USDT:USDT", # Bybit linear perpetual example
        "interval": "5", # Default timeframe (e.g., "5" for 5 minutes)

        # API and Bot Behavior
        "retry_delay": RETRY_DELAY_SECONDS, # Delay between API retries
        "enable_trading": False, # Safety Feature: Must be explicitly set to true to trade
        "use_sandbox": True, # Safety Feature: Use testnet by default
        "max_concurrent_positions": 1, # Max open positions for this symbol instance (Note: currently logic handles 1)
        "quote_currency": "USDT", # Quote currency for balance checks and sizing
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order before confirming position
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay between main loop cycles

        # Risk Management
        "risk_per_trade": 0.01, # Fraction of balance to risk (e.g., 0.01 = 1%)
        "leverage": 20, # Desired leverage (Ensure supported by exchange/market)
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing/initial fixed SL)
        "take_profit_multiple": 0.7, # ATR multiple for initial TP

        # Order Execution
        "entry_order_type": "market", # "market" or "limit"
        "limit_order_offset_buy": 0.0005, # % offset from price for BUY limit (0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005, # % offset from price for SELL limit

        # Advanced Position Management
        "enable_trailing_stop": True, # Use exchange-native Trailing Stop Loss
        "trailing_stop_callback_rate": 0.005, # Trail distance % from *activation price* (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003, # % profit move from entry to activate TSL
        "enable_break_even": True, # Enable moving SL to break-even point
        "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR
        "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry price
        "time_based_exit_minutes": None, # Optional: Exit after X minutes (e.g., 120)

        # Indicator Periods & Parameters (Using renamed defaults)
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_PERIOD,
        "bollinger_bands_period": DEFAULT_BBANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BBANDS_STDDEV,
        "cci_period": DEFAULT_CCI_PERIOD,
        "williams_r_period": DEFAULT_WILLIAMS_R_PERIOD,
        "mfi_period": DEFAULT_MFI_PERIOD,
        "stoch_rsi_period": DEFAULT_STOCH_RSI_PERIOD,
        "stoch_rsi_rsi_period": DEFAULT_STOCH_RSI_RSI_PERIOD,
        "stoch_rsi_k_period": DEFAULT_STOCH_RSI_K_PERIOD,
        "stoch_rsi_d_period": DEFAULT_STOCH_RSI_D_PERIOD,
        "psar_step": DEFAULT_PSAR_STEP,
        "psar_max_step": DEFAULT_PSAR_MAX_STEP,
        "sma_10_period": DEFAULT_SMA10_PERIOD,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_period": DEFAULT_FIB_PERIOD,

        # Indicator Calculation & Scoring Control
        "orderbook_limit": 25, # Depth of order book levels to fetch/analyze
        "signal_score_threshold": 1.5, # Score needed to trigger BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25, # Threshold for StochRSI oversold score
        "stoch_rsi_overbought_threshold": 75, # Threshold for StochRSI overbought score
        "volume_confirmation_multiplier": 1.5, # Volume > Multiplier * VolMA for confirmation
        "scalping_signal_threshold": 2.5, # Alternative threshold for specific weight sets (if needed)
        "indicators": { # Toggle calculation and scoring contribution
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
            # Ensure consistency in keys (e.g., "wr" matches _check_wr, weight sets)
        },
        "weight_sets": { # Define scoring weights for different strategies
            "scalping": { # Example: Faster, momentum-focused
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Example: Balanced
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
            # Ensure keys here match keys in "indicators" and _check_ methods
        },
        "active_weight_set": "default" # Select the active weight set
    }

    config = default_config.copy() # Start with defaults
    config_loaded = False
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            # Merge loaded config with defaults, ensuring all default keys exist
            config = _merge_configs(loaded_config, default_config)
            print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")
            config_loaded = True
        except (json.JSONDecodeError, IOError) as e:
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
            # Attempt to recreate default file if loading failed and file exists
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(default_config, f_write, indent=4, ensure_ascii=False)
                print(f"{NEON_YELLOW}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create:
                print(f"{NEON_RED}Error recreating default config file: {e_create}{RESET}")
            config = default_config # Use in-memory default
    else:
        # Config file doesn't exist, create it with defaults
        print(f"{NEON_YELLOW}Config file not found. Creating default config at {filepath}{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            config = default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            # Continue with in-memory default config if creation fails

    # --- Validation Section ---
    needs_saving = not config_loaded # Save if we created default or loaded invalid one
    current_config = config # Use current config for validation

    # Helper for validation logging and default setting
    def validate_param(key, default_value, validation_func, error_msg):
        nonlocal needs_saving
        if key not in current_config or not validation_func(current_config[key]):
            print(f"{NEON_RED}{error_msg.format(key=key, value=current_config.get(key), default=default_value)}{RESET}")
            current_config[key] = default_value
            needs_saving = True

    # Validate symbol (must be non-empty string)
    validate_param("symbol", default_config["symbol"],
                   lambda v: isinstance(v, str) and v.strip(),
                   "CRITICAL: '{key}' is missing, empty, or invalid ({value}). Resetting to default: '{default}'.")

    # Validate interval
    validate_param("interval", default_config["interval"],
                   lambda v: v in VALID_INTERVALS,
                   "Invalid interval '{value}' in config for '{key}'. Resetting to default '{default}'. Valid: " + str(VALID_INTERVALS) + ".")

    # Validate entry order type
    validate_param("entry_order_type", default_config["entry_order_type"],
                   lambda v: v in ["market", "limit"],
                   "Invalid entry_order_type '{value}' for '{key}'. Resetting to '{default}'. Must be 'market' or 'limit'.")

    # Validate active weight set exists
    validate_param("active_weight_set", default_config["active_weight_set"],
                   lambda v: v in current_config.get("weight_sets", {}),
                   "Active weight set '{value}' for '{key}' not found in 'weight_sets'. Resetting to '{default}'.")

    # Validate numeric parameters (ranges and types)
    numeric_params = {
        # key: (min_val, max_val, allow_min_equal, allow_max_equal, is_integer, default_val)
        "risk_per_trade": (0, 1, False, False, False, default_config["risk_per_trade"]),
        "leverage": (1, 1000, True, True, True, default_config["leverage"]), # Adjust max leverage realistically
        "stop_loss_multiple": (0, float('inf'), False, True, False, default_config["stop_loss_multiple"]),
        "take_profit_multiple": (0, float('inf'), False, True, False, default_config["take_profit_multiple"]),
        "trailing_stop_callback_rate": (0, 1, False, False, False, default_config["trailing_stop_callback_rate"]),
        "trailing_stop_activation_percentage": (0, 1, True, False, False, default_config["trailing_stop_activation_percentage"]), # Allow 0%
        "break_even_trigger_atr_multiple": (0, float('inf'), False, True, False, default_config["break_even_trigger_atr_multiple"]),
        "break_even_offset_ticks": (0, 100, True, True, True, default_config["break_even_offset_ticks"]),
        "signal_score_threshold": (0, float('inf'), False, True, False, default_config["signal_score_threshold"]),
        "atr_period": (1, 1000, True, True, True, default_config["atr_period"]),
        "ema_short_period": (1, 1000, True, True, True, default_config["ema_short_period"]),
        "ema_long_period": (1, 1000, True, True, True, default_config["ema_long_period"]),
        "rsi_period": (1, 1000, True, True, True, default_config["rsi_period"]),
        "bollinger_bands_period": (1, 1000, True, True, True, default_config["bollinger_bands_period"]),
        "bollinger_bands_std_dev": (0, 10, False, True, False, default_config["bollinger_bands_std_dev"]),
        "cci_period": (1, 1000, True, True, True, default_config["cci_period"]),
        "williams_r_period": (1, 1000, True, True, True, default_config["williams_r_period"]),
        "mfi_period": (1, 1000, True, True, True, default_config["mfi_period"]),
        "stoch_rsi_period": (1, 1000, True, True, True, default_config["stoch_rsi_period"]),
        "stoch_rsi_rsi_period": (1, 1000, True, True, True, default_config["stoch_rsi_rsi_period"]),
        "stoch_rsi_k_period": (1, 1000, True, True, True, default_config["stoch_rsi_k_period"]),
        "stoch_rsi_d_period": (1, 1000, True, True, True, default_config["stoch_rsi_d_period"]),
        "psar_step": (0, 1, False, False, False, default_config["psar_step"]),
        "psar_max_step": (0, 1, False, False, False, default_config["psar_max_step"]),
        "sma_10_period": (1, 1000, True, True, True, default_config["sma_10_period"]),
        "momentum_period": (1, 1000, True, True, True, default_config["momentum_period"]),
        "volume_ma_period": (1, 1000, True, True, True, default_config["volume_ma_period"]),
        "fibonacci_period": (2, 1000, True, True, True, default_config["fibonacci_period"]), # Need at least 2 points
        "orderbook_limit": (1, 100, True, True, True, default_config["orderbook_limit"]),
        "position_confirm_delay_seconds": (0, 60, True, True, False, default_config["position_confirm_delay_seconds"]),
        "loop_delay_seconds": (1, 300, True, True, False, default_config["loop_delay_seconds"]),
        "stoch_rsi_oversold_threshold": (0, 100, True, False, False, default_config["stoch_rsi_oversold_threshold"]),
        "stoch_rsi_overbought_threshold": (0, 100, False, True, False, default_config["stoch_rsi_overbought_threshold"]),
        "volume_confirmation_multiplier": (0, float('inf'), False, True, False, default_config["volume_confirmation_multiplier"]),
        "limit_order_offset_buy": (0, 0.1, True, False, False, default_config["limit_order_offset_buy"]), # 10% offset max?
        "limit_order_offset_sell": (0, 0.1, True, False, False, default_config["limit_order_offset_sell"]),
        "retry_delay": (1, 60, True, True, False, default_config["retry_delay"]),
        "max_concurrent_positions": (1, 10, True, True, True, default_config["max_concurrent_positions"]), # Limit realistically
    }
    for key, (min_val, max_val, allow_min, allow_max, is_integer, default_val) in numeric_params.items():
        value = current_config.get(key)
        is_valid = False
        if value is not None:
            try:
                # Convert to Decimal first for consistent checks, then maybe to int/float
                val_dec = Decimal(str(value))
                if not val_dec.is_finite(): raise ValueError("Value not finite")

                # Check bounds using Decimal comparison
                lower_bound_ok = val_dec >= Decimal(str(min_val)) if allow_min else val_dec > Decimal(str(min_val))
                upper_bound_ok = val_dec <= Decimal(str(max_val)) if allow_max else val_dec < Decimal(str(max_val))

                if lower_bound_ok and upper_bound_ok:
                    # Convert to final type (int or float)
                    final_value = int(val_dec) if is_integer else float(val_dec)
                    # Additional check for integer conversion if required
                    if is_integer and Decimal(final_value) != val_dec.quantize(Decimal('1')):
                         raise ValueError("Non-integer value provided for integer parameter")
                    current_config[key] = final_value # Store validated value
                    is_valid = True

            except (ValueError, TypeError, InvalidOperation):
                 # Invalid format or failed checks, is_valid remains False
                 pass

        if not is_valid:
            validate_param(key, default_val, lambda v: False, # Force reset
                           f"Invalid value for '{{key}}' ({{value}}). Must be {'integer' if is_integer else 'number'} "
                           f"between {min_val} ({'inclusive' if allow_min else 'exclusive'}) and "
                           f"{max_val} ({'inclusive' if allow_max else 'exclusive'}). Resetting to default '{{default}}'.")


    # Specific validation for time_based_exit_minutes (allow None or positive number)
    time_exit = current_config.get("time_based_exit_minutes")
    time_exit_valid = False
    if time_exit is None:
        time_exit_valid = True
    else:
        try:
            time_exit_val = float(time_exit)
            if time_exit_val > 0:
                 current_config["time_based_exit_minutes"] = time_exit_val # Store as float
                 time_exit_valid = True
            else: raise ValueError("Must be positive if set")
        except (ValueError, TypeError):
            pass # Invalid format or non-positive, time_exit_valid remains False

    if not time_exit_valid:
         validate_param("time_based_exit_minutes", default_config["time_based_exit_minutes"], lambda v: False, # Force reset
                        "Invalid value for '{{key}}' ({{value}}). Must be 'None' or a positive number. Resetting to default ('{{default}}').")

    # Validate boolean parameters
    bool_params = ["enable_trading", "use_sandbox", "enable_trailing_stop", "enable_break_even"]
    for key in bool_params:
         validate_param(key, default_config[key], lambda v: isinstance(v, bool),
                        "Invalid value for '{{key}}' ({{value}}). Must be boolean (true/false). Resetting to default '{{default}}'.")

    # Ensure indicators are boolean
    if 'indicators' in current_config and isinstance(current_config.get('indicators'), dict):
        for ind_key, ind_val in current_config['indicators'].items():
            # Check if this indicator key exists in the default config
            if ind_key in default_config['indicators']:
                 if not isinstance(ind_val, bool):
                      print(f"{NEON_RED}Invalid value for 'indicators.{ind_key}' ({ind_val}). Must be boolean. Resetting to default '{default_config['indicators'][ind_key]}'.{RESET}")
                      current_config['indicators'][ind_key] = default_config['indicators'][ind_key]
                      needs_saving = True
            else:
                 # Key exists in loaded config but not in default. Warn and remove? Or keep if boolean?
                 if not isinstance(ind_val, bool):
                      print(f"{NEON_YELLOW}Unknown indicator key '{ind_key}' found with non-boolean value '{ind_val}'. Removing from config.{RESET}")
                      del current_config['indicators'][ind_key]
                      needs_saving = True
                 # else: Keep unknown boolean key if user added it intentionally
    elif 'indicators' in current_config: # Key exists but is not a dict
         validate_param('indicators', default_config['indicators'], lambda v: False, # Force reset
                        "Invalid structure for '{key}' ({{value}}). Must be a dictionary. Resetting to default '{{default}}'.")


    # If config was updated due to invalid values or file creation, save it back
    if needs_saving:
        try:
            with open(filepath, "w", encoding="utf-8") as f_write:
                json.dump(current_config, f_write, indent=4, ensure_ascii=False)
            print(f"{NEON_YELLOW}Updated/created config file {filepath} with corrected/default values.{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

    return current_config

def _merge_configs(loaded_config: Dict, default_config: Dict) -> Dict:
    """
    Recursively merges the loaded configuration with default values.
    Ensures all keys from the default config exist in the final config.
    Prioritizes values from the loaded config. Handles nested dictionaries.
    """
    merged = default_config.copy() # Start with default structure
    for key, value in loaded_config.items():
        if key in merged:
            # If key exists in both and both values are dicts, recurse
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = _merge_configs(value, merged[key])
            # Otherwise, overwrite default with loaded value (type checks handled by validation)
            else:
                merged[key] = value
        else:
            # Key from loaded config doesn't exist in default, add it (allows user extensions)
            merged[key] = value
    # Ensure all default keys are present even if missing from loaded_config
    for key, value in default_config.items():
        if key not in merged:
            merged[key] = value
            # Also handle nested defaults if a whole dict section was missing
            if isinstance(value, dict):
                 merged[key] = _merge_configs({}, value) # Ensure nested defaults exist

    return merged

# --- Logging Setup ---
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with rotating file and colored console handlers."""
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if logger is somehow reused
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Capture all levels at the logger level

    # File Handler (Rotating) - Use UTC timestamps in file logs
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            # Consistent ISO 8601 format with milliseconds and UTC 'Z' indicator
            "%(asctime)s.%(msecs)03dZ %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S' # Use ISO 8601 standard date format
        )
        file_formatter.converter = time.gmtime # Ensure logs use UTC time
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log DEBUG and above to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger {log_filename}: {e}")
        # Fallback to console logging if file fails
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())

    # Console Handler (Colored) - Display timestamps in local timezone
    stream_handler = logging.StreamHandler()
    console_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} {NEON_YELLOW}%(levelname)-8s{RESET} {NEON_PURPLE}[%(name)s]{RESET} %(message)s",
         # Format string includes timezone name (%Z)
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    # Custom converter to generate local time tuples for the console formatter
    def local_time_converter(*args):
        return datetime.now(TIMEZONE).timetuple()

    console_formatter.converter = local_time_converter # Use local time for display
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(level) # Set console level (e.g., INFO)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent duplicate logs in root logger
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with V5 defaults and enhanced error handling."""
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'rateLimit': 150, # Default: ~6.6 req/s. Adjust based on specific V5 endpoint limits.
            'options': {
                'defaultType': 'linear', # Crucial for Bybit V5 USDT/USDC perpetuals/futures
                'adjustForTimeDifference': True,
                'fetchTickerTimeout': 15000, # ms
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 25000, # Increased for potential V5 complexity
                'fetchOHLCVTimeout': 20000,
                'fetchOrderBookTimeout': 15000, # Added timeout
                'setLeverageTimeout': 20000, # Added timeout
                'user-agent': 'livebot/1.0 (+https://github.com/your_repo)', # Optional: Identify your bot
                # Bybit V5 specific options (optional, adjust if needed)
                # 'recvWindow': 10000, # Increase if timestamp errors persist despite adjustForTimeDifference
                # 'brokerId': 'YOUR_BROKER_ID' # If affiliated
            }
        }

        exchange_id = "bybit" # Hardcoded for Bybit focus
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        # --- Sandbox Mode Setup ---
        if config.get('use_sandbox'):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            try:
                # CCXT's generic method (might work depending on CCXT version and Bybit implementation)
                exchange.set_sandbox_mode(True)
                lg.info(f"Sandbox mode enabled via exchange.set_sandbox_mode(True) for {exchange.id}.")
                # Verify the URL was actually changed
                if 'testnet' not in exchange.urls['api']:
                    lg.warning("set_sandbox_mode did not change API URL, attempting manual override.")
                    # Bybit V5 Testnet URL
                    bybit_v5_testnet_url = 'https://api-testnet.bybit.com'
                    exchange.urls['api'] = bybit_v5_testnet_url
                    if 'testnet' in exchange.urls['api']:
                         lg.info(f"Manually set API URL to Testnet: {exchange.urls['api']}")
                    else:
                         lg.error(f"Failed to manually set API URL to Testnet: {bybit_v5_testnet_url}")
            except AttributeError:
                lg.warning(f"{exchange.id} ccxt version might not support set_sandbox_mode. Manually setting API URL.")
                # Manually set Bybit testnet URL (ensure this is correct for V5)
                bybit_v5_testnet_url = 'https://api-testnet.bybit.com'
                exchange.urls['api'] = bybit_v5_testnet_url
                lg.info(f"Manually set Bybit API URL to Testnet: {exchange.urls['api']}")
            except Exception as e:
                lg.error(f"Error enabling sandbox mode: {e}. Ensure API keys are for Testnet.")
        else:
            lg.info(f"{NEON_GREEN}Using LIVE (Real Money) Environment.{RESET}")
            # Ensure API URL is production URL if sandbox was previously set somehow
            if 'testnet' in exchange.urls['api']:
                lg.warning("Detected testnet URL while in live mode. Resetting to production URL.")
                # Bybit V5 Production URL
                bybit_v5_prod_url = 'https://api.bybit.com'
                exchange.urls['api'] = bybit_v5_prod_url
                lg.info(f"Reset API URL to Production: {exchange.urls['api']}")

        lg.info(f"Initializing {exchange.id} (API: {exchange.urls['api']})...")

        # --- Load Markets (Essential for precision, limits, IDs) ---
        lg.info(f"Loading markets for {exchange.id}...")
        try:
             # Load markets only once if possible, use the flag 'reload=False' if markets are already loaded
             # and you just want to ensure they are present. Use 'reload=True' to force a refresh.
             # Let's try loading without reload first, then force reload if symbol not found later.
             if not exchange.markets: # Only load if not already loaded
                 safe_api_call(exchange.load_markets, lg, reload=False)
                 lg.info(f"Initial markets loaded successfully for {exchange.id}.")
             else:
                 lg.info("Markets seem to be already loaded. Skipping initial load.")

             # Quick check if target symbol exists after potential loading
             target_symbol = config.get("symbol")
             if target_symbol and target_symbol not in exchange.markets:
                  lg.warning(f"{NEON_YELLOW}Target symbol '{target_symbol}' not found in initially loaded markets! Attempting force reload...{RESET}")
                  safe_api_call(exchange.load_markets, lg, reload=True) # Force reload
                  if target_symbol not in exchange.markets:
                       lg.critical(f"{NEON_RED}CRITICAL: Target symbol '{target_symbol}' still not found after reloading markets! Check symbol format/availability on {exchange.id}. Exiting.{RESET}")
                       return None # Fatal error if target symbol doesn't exist
                  else:
                       lg.info(f"Target symbol '{target_symbol}' found after market reload.")

        except Exception as market_err:
             lg.critical(f"{NEON_RED}CRITICAL: Failed to load markets: {market_err}. Cannot operate without market data. Exiting.{RESET}")
             return None # Fatal error

        # --- Initial Connection & Permissions Test (Fetch Balance - Crucial for V5 Account Type) ---
        # For V5, fetch balance for the relevant account type (e.g., CONTRACT or UNIFIED)
        # Let's prioritize CONTRACT for perpetuals, but handle potential unified account structures
        account_type_to_test = 'CONTRACT' # Common for derivatives
        lg.info(f"Attempting initial balance fetch (Account Type Hint: {account_type_to_test})...")
        quote_curr = config.get("quote_currency", "USDT")
        balance_decimal = fetch_balance(exchange, quote_curr, lg) # Use the dedicated fetch_balance function

        if balance_decimal is not None:
             lg.info(f"{NEON_GREEN}Successfully connected and fetched initial {quote_curr} balance: {balance_decimal:.4f}{RESET}")
        else:
             # fetch_balance logs errors, just add a warning here
             lg.warning(f"{NEON_YELLOW}Initial balance fetch for {quote_curr} failed or returned zero. Check API permissions, account type (CONTRACT/UNIFIED?), and available funds.{RESET}")
             # Don't necessarily exit, as some actions might still be possible, but warn heavily.

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox')}, Default Type: {exchange.options.get('defaultType')}")
        return exchange

    except ccxt.AuthenticationError as e: # Catch auth errors during class instantiation
        lg.critical(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Check API Key/Secret format and validity in your .env file.{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None

# --- API Call Wrapper with Retries ---
def safe_api_call(func, logger: logging.Logger, *args, **kwargs):
    """Wraps an API call with retry logic for network/rate limit/specific exchange errors."""
    lg = logger
    attempts = 0
    # Load config values within the function call to get potentially updated values if config reloads
    bot_config = load_config(CONFIG_FILE) # Reload config for latest retry settings
    max_retries = bot_config.get("max_api_retries", MAX_API_RETRIES)
    retry_delay = bot_config.get("retry_delay", RETRY_DELAY_SECONDS)
    last_exception = None

    while attempts <= max_retries:
        try:
            result = func(*args, **kwargs)
            # Optional: Log successful call completion (can be verbose)
            # lg.debug(f"API call {func.__name__} successful.")
            return result # Success

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = retry_delay * (1.5 ** attempts) # Exponential backoff
            lg.warning(f"{NEON_YELLOW}Retryable network/availability error in {func.__name__}: {type(e).__name__}. Waiting {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1}). Error: {e}{RESET}")
            time.sleep(wait_time)

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Try to get retry-after header info (might be in seconds or ms)
            retry_after_header = None
            if hasattr(e, 'http_headers'): # CCXT standard location for headers
                 retry_after_header = e.http_headers.get('Retry-After') or e.http_headers.get('retry-after')

            # Default backoff, stronger for rate limits
            wait_time = retry_delay * (2.0 ** attempts)
            if retry_after_header:
                try:
                    # Assume seconds, add buffer
                    header_wait = float(retry_after_header) + 0.5
                    # Check if it looks like milliseconds (e.g., > 10000)
                    if header_wait > 1000: header_wait = (header_wait / 1000.0) + 0.5
                    wait_time = max(wait_time, header_wait) # Use the longer wait time
                    lg.debug(f"Rate limit Retry-After header detected: {retry_after_header} -> Wait: {header_wait:.1f}s")
                except (ValueError, TypeError):
                    lg.warning(f"Could not parse Retry-After header: {retry_after_header}")

            lg.warning(f"{NEON_YELLOW}Rate limit exceeded in {func.__name__}. Waiting {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1}). Error: {e}{RESET}")
            time.sleep(wait_time)

        except ccxt.AuthenticationError as e:
             lg.error(f"{NEON_RED}Authentication Error in {func.__name__}: {e}. Aborting call.{RESET}")
             lg.error(f"{NEON_RED}>> Check API Key/Secret validity, permissions, IP whitelist, and environment (Live/Testnet).{RESET}")
             raise e # Don't retry, re-raise immediately

        except ccxt.ExchangeError as e:
            last_exception = e
            # Bybit V5 specific transient error codes (add more as identified)
            # Reference: https://bybit-exchange.github.io/docs/v5/error_code
            bybit_retry_codes = [
                10001, # Internal server error
                10002, # Service unavailable
                10006, # Too many visits (rate limit related)
                10010, # Request validation failed (can be transient)
                10016, # Service temporarily unavailable due to maintenance
                10018, # Request Duplicate (might happen on retries)
                130150, # System busy
                130021, # Order not found or cancelled (sometimes transient during rapid changes)
                # Add other potentially transient codes based on experience
            ]
            # General retryable messages
            retryable_messages = [
                 "internal server error",
                 "service unavailable",
                 "system busy",
                 # "request validation failed", # Sometimes transient, sometimes permanent
                 "matching engine busy",
                 "order not found or cancelled", # Might retry if state change is expected
            ]

            exchange_code = None
            err_str = str(e).lower()
            # Try extracting Bybit's retCode from the message if possible (e.g., 'bybit {"retCode":10006,...}')
            try:
                if 'retcode' in err_str:
                     # Simple parsing, might need refinement
                     code_part = err_str.split('"retcode":')[1].split(',')[0].strip()
                     if code_part.isdigit(): exchange_code = int(code_part)
            except Exception as parse_err:
                 lg.debug(f"Could not parse retCode from error string: {err_str} ({parse_err})")

            is_retryable_bybit_code = exchange_code in bybit_retry_codes
            is_retryable_message = any(msg in err_str for msg in retryable_messages)
            # Retry on code 10018 only if it's not the first attempt (to avoid loops on genuine duplicates)
            should_retry = is_retryable_bybit_code or is_retryable_message
            if exchange_code == 10018 and attempts == 0:
                should_retry = False # Don't retry duplicate error on the first attempt

            if should_retry:
                 wait_time = retry_delay * (1.5 ** attempts)
                 lg.warning(f"{NEON_YELLOW}Potentially retryable exchange error in {func.__name__}: {e} (Code: {exchange_code}). Waiting {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1})...{RESET}")
                 time.sleep(wait_time) # Sleep here inside the except block
                 # Don't increment attempts here, let the loop handle it at the end
            else:
                 lg.error(f"{NEON_RED}Non-retryable Exchange Error in {func.__name__}: {e} (Code: {exchange_code}){RESET}")
                 raise e # Re-raise non-retryable or unknown exchange errors

        except Exception as e: # Catch any other unexpected error
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error during API call {func.__name__}: {e}{RESET}", exc_info=True)
            raise e # Re-raise unexpected errors immediately

        attempts += 1 # Increment attempt counter


    # If loop completes, max retries exceeded
    lg.error(f"{NEON_RED}Max retries ({max_retries+1}) exceeded for API call {func.__name__}.{RESET}")
    if last_exception:
        raise last_exception # Raise the last known exception that caused retries
    else:
        # Fallback if no exception was captured (shouldn't normally happen with this structure)
        raise ccxt.RequestTimeout(f"Max retries exceeded for {func.__name__} (no specific exception captured during retry loop)")


# --- CCXT Data Fetching (Using safe_api_call) ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks and retries."""
    lg = logger
    try:
        # Use safe_api_call for robustness
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)
        if not ticker:
            lg.error(f"Failed to fetch ticker for {symbol} after retries.")
            return None # Error logged by safe_api_call

        lg.debug(f"Ticker data for {symbol}: {json.dumps(ticker, indent=2)}") # Log full ticker for debug

        # --- Price Extraction Logic ---
        price = None
        # Priorities: last > mark (for contracts if available) > close > average > mid(bid/ask) > ask > bid
        last_price = ticker.get('last')
        mark_price = ticker.get('mark') # Often relevant for funding/liquidation on contracts
        close_price = ticker.get('close', last_price) # Use 'close' if available, fallback to 'last'
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')
        # Calculate average/mid if not provided explicitly
        avg_price = ticker.get('average')
        if avg_price is None and bid_price is not None and ask_price is not None:
            try: avg_price = (Decimal(str(bid_price)) + Decimal(str(ask_price))) / 2
            except: pass # Ignore calculation errors

        # Robust Decimal conversion helper
        def to_decimal(value) -> Optional[Decimal]:
            if value is None: return None
            try:
                d = Decimal(str(value))
                # Ensure the price is finite and positive
                if d.is_finite() and d > 0:
                    return d
                else:
                    lg.warning(f"Invalid price value encountered (non-finite or non-positive): {value}")
                    return None
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Invalid price format encountered, cannot convert to Decimal: {value}")
                return None

        p_last = to_decimal(last_price)
        p_mark = to_decimal(mark_price)
        p_close = to_decimal(close_price)
        p_bid = to_decimal(bid_price)
        p_ask = to_decimal(ask_price)
        p_avg = to_decimal(avg_price)

        # Determine price with priority
        market_info = exchange.market(symbol) if symbol in exchange.markets else {}
        is_contract = market_info.get('contract', False) or market_info.get('type') in ['swap', 'future']

        # Prioritize mark price for contracts if available and seems valid
        if is_contract and p_mark:
            price = p_mark; lg.debug(f"Using 'mark' price (contract): {price}")
        elif p_last:
            price = p_last; lg.debug(f"Using 'last' price: {price}")
        elif p_close: # Use 'close' if 'last' is missing
             price = p_close; lg.debug(f"Using 'close' price (fallback from last): {price}")
        elif p_avg: # Use 'average' (calculated or provided) next
            price = p_avg; lg.debug(f"Using 'average/mid' price: {price}")
        elif p_ask: # Fallback to ask if only ask is valid and reasonable
             # Add a check: is ask reasonably close to bid if bid exists? Avoid skewed books.
             if p_bid and (p_ask / p_bid) > Decimal('1.05'): # e.g., > 5% spread
                  lg.warning(f"Using 'ask' price fallback ({p_ask}) but spread seems large (Bid: {p_bid}).")
             else:
                  lg.debug(f"Using 'ask' price fallback: {p_ask}")
             price = p_ask
        elif p_bid: # Final fallback to bid
            price = p_bid; lg.warning(f"Using 'bid' price fallback (last resort): {price}")

        # Final validation
        if price is not None and price.is_finite() and price > 0:
            # Log the source of the final price
            # Fetch precision for formatting
            price_precision = 8 # Default if market info fails
            try:
                 # Create a temporary analyzer instance just for precision
                 temp_analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info)
                 price_precision = temp_analyzer.get_price_precision()
            except: pass
            lg.info(f"Current price ({symbol}): {price:.{price_precision}f} (Source: determined by priority)")
            return price
        else:
            lg.error(f"{NEON_RED}Failed to extract a valid positive price from ticker data for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except Exception as e:
        # Catch errors raised by safe_api_call or during parsing
        lg.error(f"{NEON_RED}Error fetching/processing current price for {symbol}: {e}{RESET}", exc_info=False) # Keep log concise
        return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries, robust validation, and Decimal conversion."""
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    try:
        # Convert our interval format to CCXT's expected format
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            lg.error(f"Invalid timeframe '{timeframe}' provided. Valid intervals: {list(VALID_INTERVALS)}")
            return pd.DataFrame()

        lg.debug(f"Fetching {limit} klines for {symbol} with timeframe {ccxt_timeframe} ({timeframe})...")
        # Use safe_api_call to handle retries
        ohlcv = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe=ccxt_timeframe, limit=limit)

        if ohlcv is None or not isinstance(ohlcv, list) or len(ohlcv) == 0:
            # Error logged by safe_api_call if failed after retries
            if ohlcv is not None: # Log only if it returned empty list/None without raising error
                lg.warning(f"{NEON_YELLOW}No valid kline data returned by fetch_ohlcv for {symbol} {ccxt_timeframe}.{RESET}")
            return pd.DataFrame()

        # Process the data into a pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            lg.warning(f"Kline data DataFrame is empty after initial creation for {symbol} {ccxt_timeframe}.")
            return df

        # --- Data Cleaning and Type Conversion ---
        # 1. Convert timestamp to datetime objects (UTC), coerce errors, set as index
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
            df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
            if df.empty:
                lg.warning("DataFrame empty after timestamp conversion/dropna.")
                return df
            df.set_index('timestamp', inplace=True)
        except Exception as ts_err:
             lg.error(f"Error processing timestamps: {ts_err}")
             return pd.DataFrame()

        # 2. Convert price/volume columns to Decimal, handling errors robustly
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns: continue # Skip if column missing
            try:
                # Apply robust conversion to Decimal
                # Handle None, empty strings, and potential non-numeric types gracefully
                def safe_to_decimal(x):
                    if pd.isna(x) or str(x).strip() == '': return Decimal('NaN')
                    try:
                        d = Decimal(str(x))
                        # Return NaN for non-finite decimals (inf, -inf) as they cause issues later
                        return d if d.is_finite() else Decimal('NaN')
                    except (InvalidOperation, TypeError, ValueError):
                        # lg.debug(f"Could not convert '{x}' to Decimal in column '{col}', returning NaN.") # Very verbose
                        return Decimal('NaN')

                df[col] = df[col].apply(safe_to_decimal)

            except Exception as conv_err: # Catch unexpected errors during apply
                 lg.error(f"Unexpected error converting column '{col}' to Decimal: {conv_err}", exc_info=True)
                 # As a fallback, try converting to float, then NaN invalid ones
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Drop rows with NaN in essential price columns or non-positive/non-finite close price
        initial_len = len(df)
        essential_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_cols, how='any', inplace=True)

        # Further filter out rows with non-positive close prices (NaNs already dropped)
        if not df.empty and 'close' in df.columns:
             # Check if close column is Decimal
             first_valid_close_idx = df['close'].first_valid_index()
             if first_valid_close_idx is not None and isinstance(df.loc[first_valid_close_idx, 'close'], Decimal):
                 df = df[df['close'] > Decimal('0')]
             else: # Assume numeric (float/int) after fallback
                 df = df[df['close'] > 0]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} {ccxt_timeframe} became empty after cleaning.")
            return pd.DataFrame()

        # 4. Sort by timestamp index and remove duplicates (keeping the last occurrence)
        df.sort_index(inplace=True)
        if df.index.has_duplicates:
            lg.debug(f"Found {df.index.duplicated().sum()} duplicate timestamps. Keeping last entry.")
            df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {ccxt_timeframe}")
        lg.debug(f"Kline check: First row:\n{df.head(1)}\nLast row:\n{df.tail(1)}") # Log head/tail
        return df

    except Exception as e:
        # Catch errors from safe_api_call or during processing
        lg.error(f"{NEON_RED}Error fetching/processing klines for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries, validation, and Decimal conversion."""
    lg = logger
    if not exchange.has['fetchOrderBook']:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    try:
        lg.debug(f"Fetching order book for {symbol} with limit {limit}...")
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)

        if not orderbook: # Error already logged by safe_api_call if it failed
            return None

        # --- Validate Structure and Content ---
        if not isinstance(orderbook, dict) or \
           'bids' not in orderbook or 'asks' not in orderbook or \
           not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
            lg.warning(f"Invalid orderbook structure received for {symbol}. Data: {orderbook}")
            return None

        # --- Convert prices and amounts to Decimal ---
        cleaned_book = {'bids': [], 'asks': [], 'timestamp': orderbook.get('timestamp'), 'datetime': orderbook.get('datetime'), 'nonce': orderbook.get('nonce')}
        conversion_errors = 0

        for side in ['bids', 'asks']:
            for entry in orderbook[side]:
                if isinstance(entry, list) and len(entry) == 2:
                    try:
                        price = Decimal(str(entry[0]))
                        amount = Decimal(str(entry[1]))
                        if price.is_finite() and price > 0 and amount.is_finite() and amount >= 0:
                            cleaned_book[side].append([price, amount])
                        else:
                            # lg.debug(f"Invalid price/amount in {side} entry: {entry}") # Verbose
                            conversion_errors += 1
                    except (InvalidOperation, ValueError, TypeError):
                        # lg.debug(f"Conversion error for {side} entry: {entry}") # Verbose
                        conversion_errors += 1
                else:
                    lg.warning(f"Invalid {side[:-1]} entry format in orderbook: {entry}")
                    conversion_errors += 1 # Treat format errors as conversion errors

        if conversion_errors > 0:
            lg.warning(f"Orderbook for {symbol}: Encountered {conversion_errors} entries with invalid format or non-finite/non-positive values.")

        # Proceed even if some entries failed, but log if book becomes empty
        if not cleaned_book['bids'] and not cleaned_book['asks']:
            lg.warning(f"Orderbook for {symbol} is empty after cleaning/conversion.")
            # Return the empty book structure rather than None if the fetch itself succeeded
            return cleaned_book

        lg.debug(f"Successfully fetched and processed orderbook for {symbol} ({len(cleaned_book['bids'])} bids, {len(cleaned_book['asks'])} asks).")
        return cleaned_book

    except Exception as e:
        # Catch errors raised by safe_api_call or other validation issues
        lg.error(f"{NEON_RED}Error fetching or processing order book for {symbol}: {e}{RESET}", exc_info=False)
        return None

# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta and generates weighted signals."""

    def __init__(
        self,
        df: pd.DataFrame, # Expects OHLCV columns with Decimal type from fetch_klines
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Pandas DataFrame with OHLCV data (expects Decimal values), indexed by timestamp.
            logger: Logger instance for logging messages.
            config: Dictionary containing bot configuration.
            market_info: Dictionary containing market details (precision, limits, etc.).
        """
        self.df = df.copy() # Work on a copy to avoid modifying original DF outside class
        self.logger = logger
        self.config = config
        self.market_info = market_info if market_info else {} # Handle None market_info
        self.symbol = self.market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "5")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval) # Store for reference
        if not self.ccxt_interval:
            self.logger.error(f"Invalid interval '{self.interval}' in config for {self.symbol}. Calculation might fail.")

        # Stores latest indicator values (Decimal for prices/ATR, float for others)
        self.indicator_values: Dict[str, Union[Decimal, float, Any]] = {}
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1} # Default HOLD
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated fib levels
        self.ta_column_names: Dict[str, Optional[str]] = {} # Maps internal name to actual DataFrame column name

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Active weight set '{self.active_weight_set_name}' not found or empty for {self.symbol}. Scoring will be zero.{RESET}")
            self.weights = {} # Use empty dict to prevent errors

        # Perform initial calculations only if DataFrame is valid
        if not self.df.empty:
             # Ensure required columns exist and have some data
             required_cols = ['open', 'high', 'low', 'close', 'volume']
             if not all(col in self.df.columns for col in required_cols):
                 self.logger.error(f"DataFrame missing one or more required columns: {required_cols}. Columns found: {self.df.columns.tolist()}")
             elif self.df[required_cols].isnull().all().all():
                  self.logger.error("DataFrame contains all NaN values in required OHLCV columns.")
             else:
                  self._calculate_all_indicators()
                  self._update_latest_indicator_values() # Run AFTER indicator calculation
                  self.calculate_fibonacci_levels()
        else:
             self.logger.warning(f"TradingAnalyzer initialized with empty DataFrame for {self.symbol}. No calculations performed.")


    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """
        Helper to find the actual column name generated by pandas_ta.
        Searches for expected patterns based on config parameters.
        """
        df_cols = result_df.columns.tolist()
        if not df_cols: return None # No columns to search

        # Define expected patterns dynamically based on current config
        # Use float for std dev formatting as pandas_ta often does
        bb_std_dev_str = f"{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BBANDS_STDDEV)):.1f}"
        expected_patterns = {
            "ATR": [f"ATRr_{self.config.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{self.config.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{self.config.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{self.config.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{self.config.get('cci_period', DEFAULT_CCI_PERIOD)}"], # Base name, suffix might vary
            "Williams_R": [f"WILLR_{self.config.get('williams_r_period', DEFAULT_WILLIAMS_R_PERIOD)}"],
            "MFI": [f"MFI_{self.config.get('mfi_period', DEFAULT_MFI_PERIOD)}"],
            "VWAP": ["VWAP_D"], # Default pandas_ta VWAP often daily anchored, check if config overrides this
            "PSAR_long": [f"PSARl_{float(self.config.get('psar_step', DEFAULT_PSAR_STEP))}_{float(self.config.get('psar_max_step', DEFAULT_PSAR_MAX_STEP))}"],
            "PSAR_short": [f"PSARs_{float(self.config.get('psar_step', DEFAULT_PSAR_STEP))}_{float(self.config.get('psar_max_step', DEFAULT_PSAR_MAX_STEP))}"],
            "SMA_10": [f"SMA_{self.config.get('sma_10_period', DEFAULT_SMA10_PERIOD)}"],
            "StochRSI_K": [f"STOCHRSIk_{self.config.get('stoch_rsi_period', DEFAULT_STOCH_RSI_PERIOD)}_{self.config.get('stoch_rsi_rsi_period', DEFAULT_STOCH_RSI_RSI_PERIOD)}_{self.config.get('stoch_rsi_k_period', DEFAULT_STOCH_RSI_K_PERIOD)}"],
            "StochRSI_D": [f"STOCHRSId_{self.config.get('stoch_rsi_period', DEFAULT_STOCH_RSI_PERIOD)}_{self.config.get('stoch_rsi_rsi_period', DEFAULT_STOCH_RSI_RSI_PERIOD)}_{self.config.get('stoch_rsi_k_period', DEFAULT_STOCH_RSI_K_PERIOD)}_{self.config.get('stoch_rsi_d_period', DEFAULT_STOCH_RSI_D_PERIOD)}"],
            "RSI": [f"RSI_{self.config.get('rsi_period', DEFAULT_RSI_PERIOD)}"],
            "BB_Lower": [f"BBL_{self.config.get('bollinger_bands_period', DEFAULT_BBANDS_PERIOD)}_{bb_std_dev_str}"],
            "BB_Middle": [f"BBM_{self.config.get('bollinger_bands_period', DEFAULT_BBANDS_PERIOD)}_{bb_std_dev_str}"],
            "BB_Upper": [f"BBU_{self.config.get('bollinger_bands_period', DEFAULT_BBANDS_PERIOD)}_{bb_std_dev_str}"],
            "Volume_MA": [f"VOL_SMA_{self.config.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"] # Custom name used
        }

        patterns_to_check = expected_patterns.get(base_name, [])
        if not patterns_to_check:
            self.logger.warning(f"No expected column pattern defined for indicator base name: '{base_name}'")
            return None

        # --- Search Strategy ---
        # 1. Exact Match First (most reliable)
        for pattern in patterns_to_check:
            if pattern in df_cols:
                self.logger.debug(f"Mapped '{base_name}' to column '{pattern}' (Exact Match)")
                return pattern

        # 2. Starts With Match (handles potential suffixes like CCI_20_100.0)
        for pattern in patterns_to_check:
            for col in df_cols:
                 # Ensure the pattern itself is not empty and the column starts with it
                 if pattern and col.startswith(pattern):
                    # Add a check to avoid overly broad matches if pattern is short
                    # e.g., "EMA" matching "EMA_10" and "EMA_20", but also "EMA_CROSS"
                    # If the part after the pattern in the column name contains letters, it might be a different indicator
                    suffix = col[len(pattern):]
                    if not any(c.isalpha() for c in suffix): # Allow numbers, underscores, periods
                        self.logger.debug(f"Mapped '{base_name}' to column '{col}' (StartsWith Match: '{pattern}')")
                        return col

        # 3. Fallback: Simple case-insensitive substring search (use with caution)
        base_lower = base_name.lower().replace('_','') # Simplify e.g., EMA_Short -> emashort
        simple_base = base_lower.split('_')[0] # e.g., "stochrsi_k" -> "stochrsi"

        # Prioritize patterns containing the simple base name
        potential_matches = [col for col in df_cols if simple_base in col.lower()]
        if len(potential_matches) == 1:
             match = potential_matches[0]
             self.logger.debug(f"Mapped '{base_name}' to '{match}' via unique simple substring search ('{simple_base}').")
             return match
        elif len(potential_matches) > 1:
              # If multiple matches, try to find one closer to expected patterns
              for pattern in patterns_to_check:
                   pattern_base = pattern.split('_')[0].lower() # e.g., STOCHRSIk... -> stochrsik
                   for match in potential_matches:
                        if pattern_base in match.lower():
                            self.logger.debug(f"Mapped '{base_name}' to '{match}' via ambiguous substring search (best guess based on pattern '{pattern}').")
                            return match
              # If still ambiguous, log warning and return None or first match?
              self.logger.warning(f"Ambiguous substring match for '{base_name}' ('{simple_base}'): Found {potential_matches}. Could not resolve.")
              return None # Safer to return None if ambiguous

        # If no match found by any method
        self.logger.warning(f"Could not find column name for indicator '{base_name}' (Expected patterns: {patterns_to_check}) in DataFrame columns: {df_cols}")
        return None

    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta, handling types and column names."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty, cannot calculate indicators for {self.symbol}.")
            return

        # Determine minimum required data length based on enabled & weighted indicators
        # Recalculate required periods based on actual config values
        required_periods = []
        indicators_config = self.config.get("indicators", {})
        active_weights = self.weights # Use stored weights

        def add_req_if_active(indicator_key, config_period_key, default_period):
            # Check both enabled flag and if weight is non-zero
            is_enabled = indicators_config.get(indicator_key, False)
            try: weight = float(active_weights.get(indicator_key, 0.0))
            except (ValueError, TypeError): weight = 0.0

            if is_enabled and weight != 0.0:
                required_periods.append(self.config.get(config_period_key, default_period))

        # Add requirements for all potentially used indicators
        add_req_if_active("atr", "atr_period", DEFAULT_ATR_PERIOD) # ATR always useful? Calculate if enabled, even if weight is 0? Decision: Only if weighted.
        add_req_if_active("momentum", "momentum_period", DEFAULT_MOMENTUM_PERIOD)
        add_req_if_active("cci", "cci_period", DEFAULT_CCI_PERIOD)
        add_req_if_active("wr", "williams_r_period", DEFAULT_WILLIAMS_R_PERIOD)
        add_req_if_active("mfi", "mfi_period", DEFAULT_MFI_PERIOD)
        add_req_if_active("sma_10", "sma_10_period", DEFAULT_SMA10_PERIOD)
        add_req_if_active("rsi", "rsi_period", DEFAULT_RSI_PERIOD)
        add_req_if_active("bollinger_bands", "bollinger_bands_period", DEFAULT_BBANDS_PERIOD)
        add_req_if_active("volume_confirmation", "volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Fibonacci requires its own period, calculate if DF large enough
        if len(self.df) >= self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD):
            required_periods.append(self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD))

        # Complex indicators
        is_ema_enabled = indicators_config.get("ema_alignment", False)
        try: ema_weight = float(active_weights.get("ema_alignment", 0.0))
        except (ValueError, TypeError): ema_weight = 0.0
        if is_ema_enabled and ema_weight != 0.0:
             required_periods.append(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
             required_periods.append(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))

        is_stochrsi_enabled = indicators_config.get("stoch_rsi", False)
        try: stochrsi_weight = float(active_weights.get("stoch_rsi", 0.0))
        except (ValueError, TypeError): stochrsi_weight = 0.0
        if is_stochrsi_enabled and stochrsi_weight != 0.0:
            # StochRSI needs its main period and the underlying RSI period
            required_periods.append(self.config.get("stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD))
            required_periods.append(self.config.get("stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD))

        min_required_data = max(required_periods) + 30 if required_periods else 50 # Add buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all active indicators reliably (min recommended: {min_required_data}). Results may contain NaNs.{RESET}")
             # Proceed anyway, but be aware of potential NaNs

        try:
            # Use the class instance's DataFrame directly now (it's already a copy)
            # --- Convert Decimal columns to float for pandas_ta ---
            # Store original types to potentially convert back later if needed (especially ATR)
            original_types = {}
            df_calc = self.df # Reference, not copy, modify in place

            for col in ['open', 'high', 'low', 'close', 'volume']:
                 if col in df_calc.columns:
                     # Check first non-NaN value's type
                     first_valid_idx = df_calc[col].first_valid_index()
                     if first_valid_idx is not None:
                          col_type = type(df_calc.loc[first_valid_idx, col])
                          original_types[col] = col_type
                          if col_type == Decimal:
                               self.logger.debug(f"Converting Decimal column '{col}' to float for TA calculation.")
                               # Apply conversion robustly, handle non-finite Decimals -> NaN
                               df_calc[col] = df_calc[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else np.nan)
                     # If column exists but is all NaN, store type as None and don't convert
                     elif df_calc[col].isnull().all():
                          original_types[col] = None
                          self.logger.debug(f"Column '{col}' is all NaN, skipping conversion.")
                     else: # Not all NaN, but first is NaN. Try inferring type and converting anyway.
                          # This case is tricky, assume it might be Decimal if others were
                          if any(t == Decimal for t in original_types.values()):
                               try:
                                    self.logger.debug(f"Attempting conversion of column '{col}' (starts with NaN) from potential Decimal to float.")
                                    df_calc[col] = df_calc[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else np.nan)
                                    original_types[col] = Decimal # Mark as originally Decimal
                               except Exception as infer_err:
                                    self.logger.warning(f"Could not infer/convert type for column '{col}' starting with NaN: {infer_err}")
                                    original_types[col] = df_calc[col].dtype # Store the actual dtype (likely float/object)

            # --- Calculate Indicators using pandas_ta Strategy ---
            # Create a Strategy object
            ta_strategy = ta.Strategy(
                name="ScalpXRX",
                description="Dynamic TA indicators based on config",
                ta=[] # Initialize empty list, append based on config
            )

            # Map internal keys to pandas_ta function names and parameters
            # Use lambda functions to dynamically get parameters from self.config
            ta_map = {
                 "atr": {"kind": "atr", "length": lambda: self.config.get("atr_period", DEFAULT_ATR_PERIOD)},
                 "ema_short": {"kind": "ema", "length": lambda: self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)},
                 "ema_long": {"kind": "ema", "length": lambda: self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)},
                 "momentum": {"kind": "mom", "length": lambda: self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)}, # 'mom' is pandas_ta name
                 "cci": {"kind": "cci", "length": lambda: self.config.get("cci_period", DEFAULT_CCI_PERIOD)},
                 "wr": {"kind": "willr", "length": lambda: self.config.get("williams_r_period", DEFAULT_WILLIAMS_R_PERIOD)}, # 'willr' is pandas_ta name
                 "mfi": {"kind": "mfi", "length": lambda: self.config.get("mfi_period", DEFAULT_MFI_PERIOD)},
                 "sma_10": {"kind": "sma", "length": lambda: self.config.get("sma_10_period", DEFAULT_SMA10_PERIOD)},
                 "rsi": {"kind": "rsi", "length": lambda: self.config.get("rsi_period", DEFAULT_RSI_PERIOD)},
                 "vwap": {"kind": "vwap"}, # VWAP usually doesn't need length, uses daily anchor by default
                 "psar": {"kind": "psar", "step": lambda: float(self.config.get("psar_step", DEFAULT_PSAR_STEP)), "max_step": lambda: float(self.config.get("psar_max_step", DEFAULT_PSAR_MAX_STEP))},
                 "stoch_rsi": {"kind": "stochrsi",
                               "length": lambda: self.config.get("stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD),
                               "rsi_length": lambda: self.config.get("stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD),
                               "k": lambda: self.config.get("stoch_rsi_k_period", DEFAULT_STOCH_RSI_K_PERIOD),
                               "d": lambda: self.config.get("stoch_rsi_d_period", DEFAULT_STOCH_RSI_D_PERIOD)},
                 "bollinger_bands": {"kind": "bbands",
                                     "length": lambda: self.config.get("bollinger_bands_period", DEFAULT_BBANDS_PERIOD),
                                     "std": lambda: float(self.config.get("bollinger_bands_std_dev", DEFAULT_BBANDS_STDDEV))},
                 # Volume MA needs separate calculation as it uses 'volume' column
            }

            # Always calculate ATR if enabled (needed for SL/TP/BE/Sizing)
            if indicators_config.get("atr", False): # Check if enabled, don't need weight check for ATR
                 if "atr" in ta_map:
                      params = {k: v() for k, v in ta_map["atr"].items() if k != 'kind'}
                      ta_strategy.ta.append(ta.Indicator(ta_map["atr"]["kind"], **params))
                      self.logger.debug(f"Adding ATR to TA strategy with params: {params}")

            # Add other indicators based on config enable/weight
            for key, enabled in indicators_config.items():
                 if key == "atr": continue # Already handled
                 try: weight = float(active_weights.get(key, 0.0))
                 except (ValueError, TypeError): weight = 0.0

                 if not enabled or weight == 0.0: continue # Skip disabled or zero-weight indicators

                 # Handle compound indicators or specific logic
                 if key == "ema_alignment":
                      if "ema_short" in ta_map:
                           params_s = {k: v() for k, v in ta_map["ema_short"].items() if k != 'kind'}
                           ta_strategy.ta.append(ta.Indicator(ta_map["ema_short"]["kind"], **params_s))
                           self.logger.debug(f"Adding EMA_Short to TA strategy with params: {params_s}")
                      if "ema_long" in ta_map:
                           params_l = {k: v() for k, v in ta_map["ema_long"].items() if k != 'kind'}
                           ta_strategy.ta.append(ta.Indicator(ta_map["ema_long"]["kind"], **params_l))
                           self.logger.debug(f"Adding EMA_Long to TA strategy with params: {params_l}")
                 elif key == "volume_confirmation":
                      # Handled separately after main TA run
                      pass
                 elif key in ta_map:
                      # General case: Add indicator from map
                      indicator_def = ta_map[key]
                      params = {k: v() for k, v in indicator_def.items() if k != 'kind'}
                      ta_strategy.ta.append(ta.Indicator(indicator_def["kind"], **params))
                      self.logger.debug(f"Adding {key} to TA strategy with params: {params}")
                 elif key != "orderbook": # Orderbook doesn't have a ta-lib equivalent here
                      self.logger.warning(f"Indicator '{key}' is enabled and weighted but has no definition in ta_map.")

            # --- Run the TA Strategy ---
            if ta_strategy.ta: # Only run if there are indicators to calculate
                 self.logger.info(f"Running pandas_ta strategy '{ta_strategy.name}' with {len(ta_strategy.ta)} indicators...")
                 try:
                     # Use df.ta.strategy() - Ensure df has required float columns
                     df_calc.ta.strategy(ta_strategy, append=True)
                     self.logger.info("Pandas_ta strategy calculation complete.")
                 except Exception as ta_err:
                      self.logger.error(f"{NEON_RED}Error running pandas_ta strategy: {ta_err}{RESET}", exc_info=True)
                      # Continue without these indicators if strategy fails

            # --- Calculate Volume MA Separately ---
            vol_key = "volume_confirmation"
            is_vol_enabled = indicators_config.get(vol_key, False)
            try: vol_weight = float(active_weights.get(vol_key, 0.0))
            except (ValueError, TypeError): vol_weight = 0.0

            if is_vol_enabled and vol_weight != 0.0:
                 try:
                     vol_ma_p = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                     vol_ma_col = f"VOL_SMA_{vol_ma_p}"
                     # Ensure volume column exists and is numeric (should be float now)
                     if 'volume' in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc['volume']):
                          df_calc[vol_ma_col] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_p)
                          self.logger.debug(f"Calculated Volume MA ({vol_ma_col}).")
                     else:
                          self.logger.warning(f"Volume column missing or not numeric, cannot calculate Volume MA.")
                 except Exception as vol_ma_err:
                      self.logger.error(f"Error calculating Volume MA: {vol_ma_err}")

            # --- Map Calculated Column Names ---
            # Map internal names to actual DataFrame column names AFTER calculations
            # Use the internal keys used in the config/weights
            indicator_mapping = {
                # Base Name: Internal Config Key (used in weights/indicators dict)
                "ATR": "atr", "EMA_Short": "ema_alignment", "EMA_Long": "ema_alignment",
                "Momentum": "momentum", "CCI": "cci", "Williams_R": "wr", "MFI": "mfi",
                "SMA_10": "sma_10", "RSI": "rsi", "VWAP": "vwap",
                "PSAR_long": "psar", "PSAR_short": "psar", # PSAR kind maps to both
                "StochRSI_K": "stoch_rsi", "StochRSI_D": "stoch_rsi", # StochRSI kind maps to both K and D
                "BB_Lower": "bollinger_bands", "BB_Middle": "bollinger_bands", "BB_Upper": "bollinger_bands",
                "Volume_MA": "volume_confirmation" # Custom name used
            }
            for internal_name, config_key in indicator_mapping.items():
                 # Only try to map if the indicator was supposed to be calculated
                 # (i.e., enabled AND weighted, or always for ATR if enabled)
                 is_enabled = indicators_config.get(config_key, False)
                 try: weight = float(active_weights.get(config_key, 0.0))
                 except (ValueError, TypeError): weight = 0.0
                 should_map = (is_enabled and weight != 0.0) or (config_key == "atr" and is_enabled)

                 if should_map:
                     self.ta_column_names[internal_name] = self._get_ta_col_name(internal_name, df_calc)

            # --- Convert ATR column back to Decimal (if it was originally Decimal) ---
            atr_col = self.ta_column_names.get("ATR")
            if atr_col and atr_col in df_calc.columns and original_types.get('close') == Decimal: # Check if prices were Decimal
                 try:
                     # Convert float column back to Decimal, handling potential NaNs/infs
                     df_calc[atr_col] = df_calc[atr_col].apply(
                         lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                     )
                     self.logger.debug(f"Converted calculated ATR column '{atr_col}' back to Decimal.")
                 except (ValueError, TypeError, InvalidOperation) as conv_err:
                      self.logger.error(f"Failed to convert ATR column '{atr_col}' back to Decimal: {conv_err}. Leaving as float.")

            # No need to reassign self.df = df_calc, we modified it in place
            self.logger.debug(f"Finished indicator calculations. Final DF columns: {self.df.columns.tolist()}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error during indicator calculation setup or execution: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates indicator_values dict with latest values, handling types and NaNs."""
        # Initialize with NaN based on expected keys from successful calculations + base OHLCV
        # Use mapped TA column names + base OHLCV
        expected_ta_keys = list(self.ta_column_names.keys())
        expected_keys = expected_ta_keys + ["Open", "High", "Low", "Close", "Volume"]

        # Determine default NaN type based on expected output
        def get_default_nan(key):
            price_keys = ["Open", "High", "Low", "Close", "ATR", "BB_Lower", "BB_Middle", "BB_Upper", "PSAR_long", "PSAR_short", "VWAP", "SMA_10", "EMA_Short", "EMA_Long"]
            volume_keys = ["Volume"] # Only base volume is Decimal initially
            return Decimal('NaN') if key in price_keys or key in volume_keys else np.nan

        default_values = {k: get_default_nan(k) for k in expected_keys}

        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame empty for {self.symbol}.")
            self.indicator_values = default_values
            return
        try:
            # Use the last valid index (should be the actual last index if data is clean)
            last_valid_index = self.df.index[-1]
            latest = self.df.loc[last_valid_index]
        except (IndexError, KeyError):
            self.logger.error(f"Error accessing latest row/index for {self.symbol}. DataFrame might be malformed.")
            self.indicator_values = default_values
            return

        updated_values = {}
        # --- Process TA indicators using mapped column names ---
        for key, col_name in self.ta_column_names.items():
            value = get_default_nan(key) # Default to appropriate NaN type
            if col_name and col_name in latest.index:
                raw_value = latest[col_name]
                # Check if the value is valid (not None, not pd.NA, finite if numeric)
                is_valid = pd.notna(raw_value)
                if is_valid and isinstance(raw_value, (float, int, Decimal)):
                    # Use float conversion for finiteness check as Decimal might be NaN
                    try: is_finite = np.isfinite(float(raw_value))
                    except: is_finite = False # Handle conversion errors
                    is_valid = is_finite

                if is_valid:
                    try:
                        # Handle ATR specifically (should be Decimal)
                        if key == "ATR":
                            value = raw_value if isinstance(raw_value, Decimal) else Decimal(str(raw_value))
                        # Handle prices from indicators (like BBands, PSAR, VWAP, SMA) - convert to Decimal
                        elif key in ["BB_Lower", "BB_Middle", "BB_Upper", "PSAR_long", "PSAR_short", "VWAP", "SMA_10", "EMA_Short", "EMA_Long"]:
                             value = raw_value if isinstance(raw_value, Decimal) else Decimal(str(raw_value))
                        # Other indicators (oscillators, etc.) as float
                        else:
                            value = float(raw_value)
                    except (ValueError, TypeError, InvalidOperation) as conv_err:
                        self.logger.warning(f"Could not convert TA value {key} ('{col_name}': {raw_value}): {conv_err}. Storing NaN.")
                        value = get_default_nan(key) # Store appropriate NaN type
                # else: value remains the default NaN type
            else:
                 # Log only if calculation was attempted (key exists in ta_column_names but col_name is None or missing from latest)
                 if key in self.ta_column_names:
                     self.logger.debug(f"Indicator column '{col_name or 'Not Mapped'}' for '{key}' not found in latest data or invalid. Storing NaN.")
            updated_values[key] = value # Store the processed value or NaN

        # --- Process Base OHLCV (should be Decimal from fetch_klines) ---
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            key_name = base_col.capitalize()
            value = Decimal('NaN') # Default to Decimal NaN
            if base_col in latest.index:
                 raw_value = latest[base_col]
                 if isinstance(raw_value, Decimal) and raw_value.is_finite():
                      value = raw_value
                 elif pd.notna(raw_value): # If not Decimal or not finite but not NaN
                      self.logger.warning(f"Base value '{base_col}' ({raw_value}) is not a finite Decimal. Storing NaN.")
            # else: value remains Decimal('NaN')
            updated_values[key_name] = value

        self.indicator_values = updated_values

        # --- Log Summary (formatted) ---
        log_vals = {}
        price_prec = self.get_price_precision()
        amount_prec = self.get_amount_precision_places()
        for k, v in self.indicator_values.items():
            # Log only finite numeric values
            if isinstance(v, Decimal):
                 if v.is_finite():
                      is_price_like = k in ['Open','High','Low','Close','ATR','BB_Lower','BB_Middle','BB_Upper','PSAR_long','PSAR_short','VWAP','SMA_10','EMA_Short','EMA_Long']
                      prec = price_prec if is_price_like else amount_prec if k=='Volume' else 8 # Default precision for other Decimals
                      log_vals[k] = f"{v:.{prec}f}"
                 # else: log_vals[k] = "NaN" # Optionally log NaNs
            elif isinstance(v, (float, int)) and np.isfinite(v):
                 log_vals[k] = f"{v:.5f}" # Format floats/ints
            # else: log_vals[k] = "NaN" # Optionally log NaNs

        if log_vals:
             self.logger.debug(f"Latest values updated ({self.symbol}): {log_vals}")
        else:
             self.logger.warning(f"No valid latest indicator values could be determined for {self.symbol}.")


    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels using Decimal precision based on high/low over a window."""
        window = window or self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD)
        self.fib_levels_data = {} # Clear previous levels

        if 'high' not in self.df.columns or 'low' not in self.df.columns:
             self.logger.error(f"Fibonacci error: Missing 'high' or 'low' columns.")
             return {}
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci ({window} bars) on {self.symbol}.")
            return {}

        df_slice = self.df.tail(window)
        try:
            # Extract high/low series (should be Decimal)
            high_series = df_slice["high"]
            low_series = df_slice["low"]

            # Find max high and min low, handling potential NaNs
            # Ensure we are working with Decimal types
            high_price = high_series.dropna().max()
            low_price = low_series.dropna().min()

            if not isinstance(high_price, Decimal) or not high_price.is_finite() or \
               not isinstance(low_price, Decimal) or not low_price.is_finite():
                self.logger.warning(f"Could not find valid finite high/low Decimal prices for Fibonacci (Window: {window}). High: {high_price}, Low: {low_price}")
                return {}

            # Ensure high is actually higher than low
            if high_price <= low_price:
                 if high_price == low_price:
                      self.logger.debug(f"Fibonacci range is zero (High=Low={high_price}). Setting all levels to this price.")
                      # Set all levels to the single price
                      price_precision = self.get_price_precision()
                      rounding_factor = Decimal('1e-' + str(price_precision))
                      level_price_quantized = high_price.quantize(rounding_factor, rounding=ROUND_DOWN)
                      levels = {f"Fib_{level_pct * 100:.1f}%": level_price_quantized for level_pct in FIB_LEVELS}
                      self.fib_levels_data = levels
                      return levels
                 else: # Should not happen if max/min logic is correct, but handle defensively
                      self.logger.error(f"Fibonacci Calc Error: Max high ({high_price}) <= Min low ({low_price}). Check data.")
                      return {}

            # --- Calculate Levels using Decimal ---
            diff = high_price - low_price
            levels = {}
            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))

            for level_pct in FIB_LEVELS:
                level_name = f"Fib_{level_pct * 100:.1f}%"
                # Retracement Level price = High - (Range * Percentage)
                level_price_raw = high_price - (diff * Decimal(str(level_pct)))
                # Quantize level price based on market precision (round down for support levels from high)
                levels[level_name] = level_price_raw.quantize(rounding_factor, rounding=ROUND_DOWN)

            self.fib_levels_data = levels
            log_levels = {k: f"{v:.{price_precision}f}" for k, v in levels.items()} # Format for logging
            self.logger.debug(f"Calculated Fibonacci levels (Window: {window}, High: {high_price}, Low: {low_price}): {log_levels}")
            return levels

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Fibonacci error: Missing column '{e}'. Ensure OHLCV data is present.{RESET}")
            return {}
        except (ValueError, TypeError, InvalidOperation) as e:
             self.logger.error(f"{NEON_RED}Fibonacci error: Invalid data type or operation during calculation. {e}{RESET}")
             return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error: {e}{RESET}", exc_info=True)
            return {}

    def get_price_precision(self) -> int:
        """Determines price precision (decimal places) from market info. More robust checks."""
        if not self.market_info: return 4 # Default if no market info

        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price') # This might be integer (places) or float/string (tick size)

            if price_precision_val is not None:
                # Case 1: Integer precision (decimal places)
                if isinstance(price_precision_val, int):
                    if price_precision_val >= 0:
                        # self.logger.debug(f"Price precision from market_info.precision.price (int): {price_precision_val}")
                        return price_precision_val
                    else:
                        self.logger.warning(f"Ignoring negative price precision value: {price_precision_val}")

                # Case 2: Float/String precision (likely tick size)
                else:
                    try:
                        tick_size = Decimal(str(price_precision_val))
                        if tick_size.is_finite() and tick_size > 0:
                            precision = abs(tick_size.normalize().as_tuple().exponent)
                            # self.logger.debug(f"Price precision inferred from market_info.precision.price (tick size {tick_size}): {precision}")
                            return precision
                        else:
                             self.logger.warning(f"Ignoring non-finite or non-positive price tick size: {price_precision_val}")
                    except (TypeError, ValueError, InvalidOperation) as e:
                         self.logger.warning(f"Could not parse market_info.precision.price '{price_precision_val}' as tick size: {e}")

            # Fallback 1: Infer from limits.price.min (less reliable, assumes min is the tick)
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_price_tick = Decimal(str(min_price_val))
                    # Heuristic: Only use if it looks like a small fractional tick size
                    if min_price_tick.is_finite() and 0 < min_price_tick < Decimal('1'):
                        precision = abs(min_price_tick.normalize().as_tuple().exponent)
                        # self.logger.debug(f"Price precision inferred from market_info.limits.price.min ({min_price_tick}): {precision}")
                        return precision
                except (TypeError, ValueError, InvalidOperation) as e:
                    self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' for precision: {e}")

            # Fallback 2: Infer from last close price (least reliable, use cautiously)
            last_close = self.indicator_values.get("Close") # Assumes indicator_values is updated
            if isinstance(last_close, Decimal) and last_close.is_finite() and last_close > 0:
                try:
                    # Extract exponent after normalization (removes trailing zeros)
                    precision = abs(last_close.normalize().as_tuple().exponent)
                    # Sanity check: precision should be reasonable (0-8 for most crypto)
                    if 0 <= precision <= 8:
                        # self.logger.debug(f"Price precision inferred from last close price ({last_close}): {precision}")
                        return precision
                    else:
                        self.logger.debug(f"Ignoring precision from last close price ({precision}), out of typical range.")
                except Exception as e:
                     self.logger.debug(f"Could not infer precision from last close price: {e}")

        except Exception as e:
            self.logger.warning(f"Error determining price precision for {self.symbol}: {e}. Falling back.", exc_info=False)

        # --- Final Default Fallback ---
        default_precision = 4 # A common default for USDT pairs, adjust if needed
        self.logger.warning(f"Could not determine price precision for {self.symbol}. Using default: {default_precision}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info. More robust."""
        if not self.market_info: return Decimal('0.0001') # Default if no market info

        try:
            # 1. Try precision.price directly as tick size (most reliable if float/string)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None and not isinstance(price_precision_val, int):
                try:
                    tick = Decimal(str(price_precision_val))
                    if tick.is_finite() and tick > 0:
                        # self.logger.debug(f"Tick size from market_info.precision.price (value): {tick}")
                        return tick
                except (TypeError, ValueError, InvalidOperation) as e:
                     self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")

            # 2. Fallback: Try limits.price.min (often represents tick size)
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_tick = Decimal(str(min_price_val))
                    if min_tick.is_finite() and min_tick > 0:
                        # self.logger.debug(f"Tick size inferred from market_info.limits.price.min: {min_tick}")
                        return min_tick
                except (TypeError, ValueError, InvalidOperation) as e:
                     self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' for tick size: {e}")

            # 3. Fallback: Calculate from integer precision.price if available
            if price_precision_val is not None and isinstance(price_precision_val, int) and price_precision_val >= 0:
                 tick = Decimal('1e-' + str(price_precision_val))
                 # self.logger.debug(f"Tick size calculated from market_info.precision.price (int {price_precision_val}): {tick}")
                 return tick

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}.")

        # --- Final Fallback: Calculate from derived decimal places ---
        # This might be inaccurate if the precision was also derived poorly
        price_precision_places = self.get_price_precision() # Call the robust getter
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.warning(f"Using fallback tick size based on derived precision ({price_precision_places}): {fallback_tick}")
        return fallback_tick

    def get_amount_precision_places(self) -> int:
        """Determines amount precision (decimal places) from market info."""
        if not self.market_info: return 8 # Default if no market info

        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount') # Int (places) or Float/String (step size)

            if amount_precision_val is not None:
                # Case 1: Integer precision
                if isinstance(amount_precision_val, int):
                    if amount_precision_val >= 0:
                         # self.logger.debug(f"Amount precision from market_info.precision.amount (int): {amount_precision_val}")
                         return amount_precision_val
                # Case 2: Float/String (step size)
                else:
                     try:
                          step_size = Decimal(str(amount_precision_val))
                          if step_size.is_finite() and step_size > 0:
                               precision = abs(step_size.normalize().as_tuple().exponent)
                               # self.logger.debug(f"Amount precision inferred from market_info.precision.amount (step {step_size}): {precision}")
                               return precision
                     except (TypeError, ValueError, InvalidOperation): pass # Ignore parsing errors

            # Fallback 1: Infer from limits.amount.min (might be step size)
            min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_val is not None:
                try:
                    min_amount_step = Decimal(str(min_amount_val))
                    # Heuristic: If it's small and fractional, it's likely the step size
                    if min_amount_step.is_finite() and 0 < min_amount_step:
                       # Check if it looks like a step size (non-integer or less than 1)
                       if min_amount_step < 1 or '.' in str(min_amount_val) or 'e' in str(min_amount_val).lower():
                           precision = abs(min_amount_step.normalize().as_tuple().exponent)
                           # self.logger.debug(f"Amount precision inferred from market_info.limits.amount.min ({min_amount_step}): {precision}")
                           return precision
                       elif min_amount_step >= 1 and '.' not in str(min_amount_val) and 'e' not in str(min_amount_val).lower():
                           # Likely an integer minimum amount (e.g., 1 contract), precision is 0
                           # self.logger.debug(f"Amount precision inferred as 0 from integer market_info.limits.amount.min ({min_amount_step})")
                           return 0

                except (TypeError, ValueError, InvalidOperation): pass # Ignore parsing errors

        except Exception as e:
            self.logger.warning(f"Error determining amount precision for {self.symbol}: {e}.")

        # --- Final Default Fallback ---
        default_precision = 8 # Common for crypto base amounts, adjust if needed
        self.logger.warning(f"Could not determine amount precision for {self.symbol}. Using default: {default_precision}.")
        return default_precision

    def get_min_amount_step(self) -> Decimal:
        """Gets the minimum amount increment (step size) from market info."""
        if not self.market_info: return Decimal('1e-8') # Default if no market info

        try:
            # 1. Try precision.amount directly as step size
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None and not isinstance(amount_precision_val, int):
                try:
                    step = Decimal(str(amount_precision_val))
                    if step.is_finite() and step > 0:
                        # self.logger.debug(f"Amount step size from market_info.precision.amount (value): {step}")
                        return step
                except (TypeError, ValueError, InvalidOperation): pass

            # 2. Fallback: Try limits.amount.min (often represents step size)
            min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_val is not None:
                try:
                    min_step = Decimal(str(min_amount_val))
                    if min_step.is_finite() and min_step > 0:
                        # self.logger.debug(f"Amount step size inferred from market_info.limits.amount.min: {min_step}")
                        return min_step
                except (TypeError, ValueError, InvalidOperation): pass

            # 3. Fallback: Calculate from integer precision.amount
            if amount_precision_val is not None and isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                 step = Decimal('1e-' + str(amount_precision_val))
                 # self.logger.debug(f"Amount step size calculated from market_info.precision.amount (int {amount_precision_val}): {step}")
                 return step

        except Exception as e:
            self.logger.warning(f"Could not determine min amount step for {self.symbol}: {e}.")

        # --- Final Fallback: Calculate from derived decimal places ---
        amount_precision_places = self.get_amount_precision_places()
        fallback_step = Decimal('1e-' + str(amount_precision_places))
        self.logger.warning(f"Using fallback amount step based on derived precision ({amount_precision_places}): {fallback_step}")
        return fallback_step


    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            # self.logger.debug(f"Fibonacci levels not calculated or empty for {self.symbol}.")
            return []
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                # Ensure level_price is a valid Decimal before calculating distance
                if isinstance(level_price, Decimal) and level_price.is_finite() and level_price > 0:
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    self.logger.warning(f"Invalid or non-finite Fib level value encountered: {name}={level_price}. Skipping.")

            if not level_distances:
                self.logger.debug("No valid Fibonacci levels found to compare distance.")
                return []

            # Sort by distance (Decimal comparison works correctly)
            level_distances.sort(key=lambda x: x['distance'])

            # Return the name and level price for the nearest N levels
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates final trading signal (BUY/SELL/HOLD) based on weighted score of enabled indicators.
        Uses Decimal for score aggregation for precision.
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Reset to HOLD
        final_score = Decimal("0.0")
        total_weight = Decimal("0.0")
        active_indicator_count = 0
        nan_or_invalid_count = 0
        debug_scores = {} # Store individual scores for logging

        # --- Basic Validation ---
        if not self.indicator_values:
            self.logger.warning("Signal Generation Skipped: Indicator values dictionary is empty."); return "HOLD"

        # Map indicator_values keys (CamelCase) to weight_set keys (snake_case)
        # This allows flexibility in naming conventions between indicator outputs and config
        value_key_to_weight_key = {
            "EMA_Short": "ema_alignment", "EMA_Long": "ema_alignment",
            "Momentum": "momentum",
            "Volume": "volume_confirmation", "Volume_MA": "volume_confirmation", # Check volume_confirmation check method uses these
            "StochRSI_K": "stoch_rsi", "StochRSI_D": "stoch_rsi",
            "RSI": "rsi",
            "BB_Lower": "bollinger_bands", "BB_Middle": "bollinger_bands", "BB_Upper": "bollinger_bands",
            "VWAP": "vwap",
            "CCI": "cci",
            "Williams_R": "wr", # Corrected key from "William's R" to "Williams_R"
            "PSAR_long": "psar", "PSAR_short": "psar",
            "SMA_10": "sma_10",
            "MFI": "mfi",
            # Orderbook handled separately by key "orderbook"
        }

        # Check if at least one *weighted* indicator has a finite value
        core_values_present = False
        active_indicators_config = self.config.get("indicators", {})
        active_weights = self.weights # Use stored weights

        for key, value in self.indicator_values.items():
             weight_key = value_key_to_weight_key.get(key)
             if not weight_key: continue # Skip if no mapping for this value key

             if active_indicators_config.get(weight_key, False): # Check if indicator is enabled
                 try: weight = Decimal(str(active_weights.get(weight_key, 0.0)))
                 except (ValueError, TypeError, InvalidOperation): weight = Decimal('0')

                 if weight > 0: # Only check weighted indicators
                     if isinstance(value, Decimal) and value.is_finite(): core_values_present = True; break
                     if isinstance(value, (float, int)) and np.isfinite(value): core_values_present = True; break

        # Also check if orderbook is enabled, weighted, and data is present
        orderbook_enabled = active_indicators_config.get("orderbook", False)
        try: orderbook_weight = Decimal(str(active_weights.get("orderbook", 0.0)))
        except (ValueError, TypeError, InvalidOperation): orderbook_weight = Decimal('0')
        orderbook_contribution_possible = orderbook_enabled and orderbook_weight > 0 and orderbook_data

        if not core_values_present and not orderbook_contribution_possible:
            self.logger.warning("Signal Generation Skipped: All weighted core indicators have NaN/invalid values (and no valid Orderbook contribution possible).")
            return "HOLD"

        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Signal Generation Skipped: Invalid current price ({current_price})."); return "HOLD"
        if not self.weights:
            self.logger.error("Signal Generation Failed: Active weight set is missing or empty in config."); return "HOLD"

        # --- Iterate through indicators listed in the config's active weight set ---
        # This ensures we only score indicators that have a weight defined
        for indicator_key in self.weights.keys():
            if not active_indicators_config.get(indicator_key, False): continue # Skip disabled indicators

            # Get weight, ensuring it's a valid Decimal
            weight_str = self.weights.get(indicator_key)
            if weight_str is None: continue # Should not happen if iterating keys, but safety check

            try:
                weight = Decimal(str(weight_str))
                if not weight.is_finite(): raise ValueError("Weight not finite")
                if weight == 0: continue # Skip zero weight indicators efficiently
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid weight '{weight_str}' for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping."); continue

            # Find and execute the corresponding check method
            check_method_name = f"_check_{indicator_key}"
            score_float = np.nan # Default score if check fails or method missing
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                try:
                    method = getattr(self, check_method_name)
                    # Special handling for orderbook which needs extra data
                    if indicator_key == "orderbook":
                        if orderbook_data:
                            # Pass Decimal price and orderbook dict (which contains Decimals)
                            score_float = method(orderbook_data, current_price)
                        else:
                            # Log only if orderbook had non-zero weight
                            if weight != 0: self.logger.debug(f"Orderbook check skipped for {self.symbol}: No orderbook data available.")
                    else:
                        score_float = method() # Call the check method

                except Exception as e:
                    self.logger.error(f"Error executing indicator check '{check_method_name}' for {self.symbol}: {e}", exc_info=True)
            elif weight != 0: # Log if method is missing for a weighted indicator
                self.logger.warning(f"Check method '{check_method_name}' not found for weighted indicator '{indicator_key}'.")

            # Store score for debugging (convert finite floats for logging)
            debug_scores[indicator_key] = f"{score_float:.3f}" if pd.notna(score_float) and np.isfinite(score_float) else str(score_float)

            # --- Aggregate Score (using Decimal) ---
            if pd.notna(score_float) and np.isfinite(score_float):
                try:
                    # Convert the float score (-1 to 1) to Decimal for aggregation
                    score_dec = Decimal(str(score_float))
                    # Clamp score just in case a check method returns out of bounds
                    clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_dec))
                    final_score += clamped_score * weight
                    total_weight += weight # Sum weights of indicators that provided a valid score
                    active_indicator_count += 1
                except (ValueError, TypeError, InvalidOperation) as calc_err:
                    self.logger.error(f"Error processing score for {indicator_key} ({score_float}): {calc_err}")
                    nan_or_invalid_count += 1
            else:
                # Score was NaN or infinite from the check method
                nan_or_invalid_count += 1

        # --- Determine Final Signal ---
        final_signal = "HOLD"
        if total_weight <= 0: # Use <= to handle potential precision issues or all-zero weights
            self.logger.warning(f"No indicators contributed valid scores or weights for {self.symbol} (Total Weight: {total_weight}). Defaulting to HOLD.")
        else:
            # Normalize the score? Optional, but can make threshold more intuitive (range -1 to 1)
            # normalized_score = final_score / total_weight
            # Using raw score against threshold might be fine if weights are balanced
            final_score_numeric = final_score # Keep using raw weighted score

            # Get threshold from config
            try:
                threshold_str = self.config.get("signal_score_threshold", "1.5")
                threshold = Decimal(str(threshold_str))
                if not threshold.is_finite() or threshold <= 0: raise ValueError("Threshold non-positive/finite")
            except (ValueError, TypeError, InvalidOperation):
                default_threshold = Decimal("1.5")
                self.logger.warning(f"Invalid signal_score_threshold '{threshold_str}'. Using default {default_threshold}.")
                threshold = default_threshold

            # Compare final score to threshold
            if final_score_numeric >= threshold:
                final_signal = "BUY"
            elif final_score_numeric <= -threshold:
                final_signal = "SELL"
            # else: final_signal remains "HOLD"

        # --- Log Summary ---
        price_prec = self.get_price_precision()
        sig_color = NEON_GREEN if final_signal == "BUY" else NEON_RED if final_signal == "SELL" else NEON_YELLOW
        log_msg = (
            f"Signal ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Ind(Act/Inv):{active_indicator_count}/{nan_or_invalid_count}, "
            f"TotW={total_weight:.2f}, Score={final_score:.4f} (Thr: +/-{threshold:.2f}) "
            f"==> {sig_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        # Log debug scores only if logger level is DEBUG
        if self.logger.isEnabledFor(logging.DEBUG):
             self.logger.debug(f"  Indicator Scores ({self.symbol}): {debug_scores}")

        # Update internal signal state
        self.signals["BUY"] = 1 if final_signal == "BUY" else 0
        self.signals["SELL"] = 1 if final_signal == "SELL" else 0
        self.signals["HOLD"] = 1 if final_signal == "HOLD" else 0
        return final_signal

    # --- Indicator Check Methods (return float score -1.0 to 1.0 or np.nan) ---
    # Ensure methods fetch values from self.indicator_values, handle potential NaN/inf/Decimal/float types

    def _check_ema_alignment(self) -> float:
        """Checks if EMAs are aligned and price confirms trend."""
        ema_s = self.indicator_values.get("EMA_Short") # Expect Decimal
        ema_l = self.indicator_values.get("EMA_Long")  # Expect Decimal
        close = self.indicator_values.get("Close")     # Expect Decimal

        # Validate all inputs are finite Decimals
        if not isinstance(ema_s, Decimal) or not ema_s.is_finite() or \
           not isinstance(ema_l, Decimal) or not ema_l.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
            # self.logger.debug("EMA alignment check skipped: Missing or non-finite Decimal values.")
            return np.nan

        # Strong signals
        if close > ema_s > ema_l: return 1.0 # Strong Bullish trend
        if close < ema_s < ema_l: return -1.0 # Strong Bearish trend

        # Weaker signals (crossing or price disagreeing)
        if ema_s > ema_l: # Bullish cross or alignment
            return 0.3 if close > ema_s else -0.2 # Price confirms weak bullish / Price disagrees
        if ema_s < ema_l: # Bearish cross or alignment
            return -0.3 if close < ema_s else 0.2 # Price confirms weak bearish / Price disagrees

        return 0.0 # EMAs likely equal

    def _check_momentum(self) -> float:
        """Scores based on Momentum indicator value."""
        momentum = self.indicator_values.get("Momentum") # Expect float
        if not isinstance(momentum, (float, int)) or not np.isfinite(momentum): return np.nan

        # Simple scaling: Assume momentum around 0 is neutral.
        # Need a scaling factor based on typical MOM range for the asset/timeframe.
        # This is highly asset-dependent. Use a simple threshold approach for now.
        # Consider using the momentum value directly scaled, e.g., clip(mom / typical_range_scale, -1, 1)
        # For simplicity, stick to thresholds:
        pos_threshold = 0.1 # Example: Adjust based on observation
        neg_threshold = -0.1 # Example: Adjust based on observation

        if momentum > pos_threshold * 5: return 1.0 # Very strong positive momentum
        if momentum > pos_threshold: return 0.6     # Strong positive momentum
        if momentum > 0: return 0.2                # Mild positive momentum
        if momentum < neg_threshold * 5: return -1.0 # Very strong negative momentum
        if momentum < neg_threshold: return -0.6     # Strong negative momentum
        if momentum < 0: return -0.2                # Mild negative momentum

        return 0.0 # Exactly zero or within dead zone

    def _check_volume_confirmation(self) -> float:
        """Scores based on current volume relative to its moving average."""
        current_volume = self.indicator_values.get("Volume")       # Expect Decimal
        volume_ma = self.indicator_values.get("Volume_MA")       # Expect float (from ta)

        # Validate inputs
        if not isinstance(current_volume, Decimal) or not current_volume.is_finite() or \
           not isinstance(volume_ma, (float, int)) or not np.isfinite(volume_ma) or volume_ma <= 0:
            # self.logger.debug("Volume confirmation check skipped: Invalid inputs.")
            return np.nan

        try:
            volume_ma_dec = Decimal(str(volume_ma))
            multiplier = Decimal(str(self.config.get("volume_confirmation_multiplier", 1.5)))
            if multiplier <= 0: multiplier = Decimal('1.5') # Ensure positive multiplier

            ratio = current_volume / volume_ma_dec

            if ratio > multiplier: return 0.7 # High volume confirmation (boost signal)
            if ratio < (Decimal('1') / multiplier): return -0.4 # Unusually low volume (potential lack of interest/confirmation)
            # Could add more granular scoring based on the ratio
            # e.g., scale score between -0.4 and 0.7 based on ratio position relative to 1/mult and mult

            return 0.0 # Volume is within the expected range (neutral confirmation)

        except (ValueError, TypeError, InvalidOperation, ZeroDivisionError) as e:
             self.logger.warning(f"Error during volume confirmation calculation: {e}")
             return np.nan

    def _check_stoch_rsi(self) -> float:
        """Scores based on Stochastic RSI K and D values and thresholds."""
        k = self.indicator_values.get("StochRSI_K") # Expect float
        d = self.indicator_values.get("StochRSI_D") # Expect float

        if not isinstance(k, (float, int)) or not np.isfinite(k) or \
           not isinstance(d, (float, int)) or not np.isfinite(d):
            return np.nan

        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))

        score = 0.0
        # Extreme conditions get max score
        if k < oversold and d < oversold: score = 1.0 # Strong Buy signal (Oversold)
        elif k > overbought and d > overbought: score = -1.0 # Strong Sell signal (Overbought)
        # Readings within OB/OS but not extreme
        elif k < oversold or d < oversold: score = max(score, 0.7) # Moderately oversold
        elif k > overbought or d > overbought: score = min(score, -0.7) # Moderately overbought

        # Consider crosses (K relative to D) outside extreme zones
        cross_threshold = 3 # How much K needs to cross D by to be significant
        if score == 0.0: # Only apply cross logic if not already in OB/OS zones
             if k > d + cross_threshold: # Bullish cross
                 score = max(score, 0.5) # Bullish cross, moderate strength
             elif d > k + cross_threshold: # Bearish cross
                 score = min(score, -0.5) # Bearish cross, moderate strength

        # Dampen signal if near 50 and no cross
        if 40 < k < 60 and 40 < d < 60 and abs(k - d) < cross_threshold:
             score *= 0.3 # Dampen signal significantly if flat in neutral zone

        # Clamp final score just in case
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float:
        """Scores based on RSI value relative to standard overbought/oversold levels."""
        rsi = self.indicator_values.get("RSI") # Expect float
        if not isinstance(rsi, (float, int)) or not np.isfinite(rsi): return np.nan

        # Standard levels: 70 Overbought, 30 Oversold
        # Use a graded score
        if rsi >= 80: return -1.0 # Extremely Overbought
        if rsi >= 70: return -0.7 # Overbought
        if rsi > 60: return -0.3 # Approaching Overbought
        if rsi <= 20: return 1.0 # Extremely Oversold
        if rsi <= 30: return 0.7 # Oversold
        if rsi < 40: return 0.3 # Approaching Oversold

        # Neutral zone
        if 40 <= rsi <= 60: return 0.0

        # Should not be reached, but return 0.0 as default neutral
        return 0.0

    def _check_cci(self) -> float:
        """Scores based on CCI value relative to standard +/-100 levels."""
        cci = self.indicator_values.get("CCI") # Expect float
        if not isinstance(cci, (float, int)) or not np.isfinite(cci): return np.nan

        # Standard levels: +100 Overbought (Sell signal), -100 Oversold (Buy signal)
        if cci >= 200: return -1.0 # Extreme Sell signal
        if cci >= 100: return -0.7 # Standard Sell signal
        if cci > 0: return -0.2   # Mild bearish momentum (above zero line)
        if cci <= -200: return 1.0 # Extreme Buy signal
        if cci <= -100: return 0.7 # Standard Buy signal
        if cci < 0: return 0.2   # Mild bullish momentum (below zero line)

        return 0.0 # Exactly zero

    def _check_wr(self) -> float:
        """Scores based on Williams %R value relative to standard -20/-80 levels."""
        wr = self.indicator_values.get("Williams_R") # Expect float (range -100 to 0)
        if not isinstance(wr, (float, int)) or not np.isfinite(wr): return np.nan

        # Standard levels: -20 Overbought (Sell signal), -80 Oversold (Buy signal)
        if wr >= -10: return -1.0 # Extreme Overbought
        if wr >= -20: return -0.7 # Standard Overbought
        if wr > -50: return -0.2  # In upper half (more bearish)
        if wr <= -90: return 1.0 # Extreme Oversold
        if wr <= -80: return 0.7 # Standard Oversold
        if wr < -50: return 0.2  # In lower half (more bullish)

        return 0.0 # Exactly -50

    def _check_psar(self) -> float:
        """Scores based on Parabolic SAR position relative to price."""
        psar_l = self.indicator_values.get("PSAR_long")  # Expect Decimal or NaN
        psar_s = self.indicator_values.get("PSAR_short") # Expect Decimal or NaN
        close = self.indicator_values.get("Close")       # Expect Decimal

        # PSAR values from pandas_ta might be NaN when inactive, or a price level when active.
        # PSAR Long is active (plots below price) during an uptrend.
        # PSAR Short is active (plots above price) during a downtrend.

        # Check if close is valid
        if not isinstance(close, Decimal) or not close.is_finite(): return np.nan

        l_active = isinstance(psar_l, Decimal) and psar_l.is_finite()
        s_active = isinstance(psar_s, Decimal) and psar_s.is_finite()

        if l_active and not s_active and close > psar_l: return 1.0  # Uptrend confirmed by PSAR Long below price
        elif s_active and not l_active and close < psar_s: return -1.0 # Downtrend confirmed by PSAR Short above price
        elif not l_active and not s_active: return np.nan # Indeterminate or insufficient data
        else:
             # Either both active (error state) or price crossed the active PSAR (potential reversal signal)
             if l_active and close < psar_l: return -0.5 # Price crossed below PSAR_long (potential sell)
             if s_active and close > psar_s: return 0.5  # Price crossed above PSAR_short (potential buy)
             # Log the unusual state if both seem active
             if l_active and s_active:
                 self.logger.warning(f"PSAR check encountered unusual state: Both Long ({psar_l}) and Short ({psar_s}) seem active. Returning neutral.")
             return 0.0 # Default neutral if none of the conditions met

    def _check_sma_10(self) -> float:
        """Scores based on price position relative to the 10-period SMA."""
        sma = self.indicator_values.get("SMA_10")   # Expect Decimal
        close = self.indicator_values.get("Close") # Expect Decimal

        if not isinstance(sma, Decimal) or not sma.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
           return np.nan

        if close > sma: return 0.6  # Price above SMA (Bullish bias)
        if close < sma: return -0.6 # Price below SMA (Bearish bias)
        return 0.0 # Price exactly on SMA

    def _check_vwap(self) -> float:
        """Scores based on price position relative to VWAP."""
        vwap = self.indicator_values.get("VWAP")   # Expect Decimal
        close = self.indicator_values.get("Close") # Expect Decimal

        if not isinstance(vwap, Decimal) or not vwap.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
           return np.nan

        # VWAP acts as a dynamic support/resistance or mean
        if close > vwap: return 0.7  # Price above VWAP (Bullish, potentially overextended short-term)
        if close < vwap: return -0.7 # Price below VWAP (Bearish, potentially oversold short-term)
        return 0.0 # Price exactly on VWAP

    def _check_mfi(self) -> float:
        """Scores based on Money Flow Index relative to standard 20/80 levels."""
        mfi = self.indicator_values.get("MFI") # Expect float
        if not isinstance(mfi, (float, int)) or not np.isfinite(mfi): return np.nan

        # Standard levels: 80 Overbought, 20 Oversold
        if mfi >= 85: return -1.0 # Extreme Overbought
        if mfi >= 80: return -0.7 # Overbought
        if mfi > 65: return -0.3  # Approaching Overbought
        if mfi <= 15: return 1.0 # Extreme Oversold
        if mfi <= 20: return 0.7 # Oversold
        if mfi < 35: return 0.3  # Approaching Oversold

        # Neutral zone
        if 35 <= mfi <= 65: return 0.0

        return 0.0 # Default

    def _check_bollinger_bands(self) -> float:
        """Scores based on price position relative to Bollinger Bands."""
        bbl = self.indicator_values.get("BB_Lower")   # Expect Decimal
        bbm = self.indicator_values.get("BB_Middle")  # Expect Decimal
        bbu = self.indicator_values.get("BB_Upper")   # Expect Decimal
        close = self.indicator_values.get("Close")    # Expect Decimal

        # Validate inputs
        if not isinstance(bbl, Decimal) or not bbl.is_finite() or \
           not isinstance(bbm, Decimal) or not bbm.is_finite() or \
           not isinstance(bbu, Decimal) or not bbu.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
            return np.nan

        # Check for valid band range
        if bbu <= bbl:
            self.logger.debug("BBands check skipped: Upper band <= lower band.")
            return np.nan # Invalid bands

        # --- Scoring Logic ---
        # 1. Touching or exceeding bands (strong reversal signal)
        if close <= bbl: return 1.0 # Strong Buy signal (potential mean reversion)
        if close >= bbu: return -1.0 # Strong Sell signal (potential mean reversion)

        # 2. Position between middle and outer bands (trend indication / fade opportunity)
        # Normalize position within the relevant half-band (0 to 1 scale)
        try:
            if close > bbm: # Above middle band
                 band_width_upper = bbu - bbm
                 if band_width_upper <= 0: return -0.2 # Avoid division by zero, slightly bearish bias
                 position = (close - bbm) / band_width_upper
                 # Score approaches -1 as price nears upper band (fade signal)
                 # Score closer to 0 near middle band
                 # Use a simple linear scale: -0.2 at BBM, -0.8 at BBU
                 score = -0.2 - 0.6 * float(position)
                 return max(-0.9, score) # Clamp score range, slightly stronger near top

            elif close < bbm: # Below middle band
                 band_width_lower = bbm - bbl
                 if band_width_lower <= 0: return 0.2 # Avoid division by zero, slightly bullish bias
                 position = (bbm - close) / band_width_lower
                 # Score approaches +1 as price nears lower band (fade signal)
                 # Score closer to 0 near middle band
                 # Use a simple linear scale: 0.2 at BBM, 0.8 at BBL
                 score = 0.2 + 0.6 * float(position)
                 return min(0.9, score) # Clamp score range, slightly stronger near bottom
            else: # Exactly on middle band
                 return 0.0

        except (ZeroDivisionError, InvalidOperation) as e:
             self.logger.warning(f"BBands calculation error: {e}")
             return np.nan


    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes Order Book Imbalance (OBI) for short-term pressure."""
        if not orderbook_data or not isinstance(orderbook_data.get('bids'), list) or not isinstance(orderbook_data.get('asks'), list):
            self.logger.debug("Orderbook check skipped: Invalid or missing data.")
            return np.nan

        bids = orderbook_data['bids'] # List of [Decimal(price), Decimal(amount)]
        asks = orderbook_data['asks'] # List of [Decimal(price), Decimal(amount)]

        if not bids or not asks:
            self.logger.debug("Orderbook check skipped: Bids or asks list is empty.")
            return np.nan

        try:
            # Use a fixed number of levels as configured
            levels_to_analyze = min(len(bids), len(asks), int(self.config.get("orderbook_limit", 10)))
            if levels_to_analyze <= 0: return 0.0

            # Calculate total volume within the analyzed levels
            # Ensure entries are valid [Decimal, Decimal] and amount is finite
            total_bid_volume = sum(b[1] for b in bids[:levels_to_analyze] if len(b)==2 and isinstance(b[1], Decimal) and b[1].is_finite())
            total_ask_volume = sum(a[1] for a in asks[:levels_to_analyze] if len(a)==2 and isinstance(a[1], Decimal) and a[1].is_finite())

            total_volume = total_bid_volume + total_ask_volume
            if total_volume <= 0:
                # self.logger.debug("Orderbook check: Zero total volume in analyzed levels.")
                return 0.0 # Avoid division by zero

            # Calculate Order Book Imbalance (OBI)
            obi = (total_bid_volume - total_ask_volume) / total_volume
            # OBI > 0 indicates more bid volume (potential upward pressure)
            # OBI < 0 indicates more ask volume (potential downward pressure)

            # Clamp score to [-1, 1] and return as float
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi)))

            # self.logger.debug(f"OB Check ({levels_to_analyze} levels): BidVol={total_bid_volume:.4f}, AskVol={total_ask_volume:.4f}, OBI={obi:.4f} -> Score={score:.4f}")
            return score

        except (IndexError, ValueError, TypeError, InvalidOperation, ZeroDivisionError) as e:
             self.logger.warning(f"Orderbook analysis failed during calculation: {e}", exc_info=False)
             return np.nan


    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential Take Profit and initial Stop Loss based on entry estimate, signal, and ATR.
        Returns (Validated Entry Estimate, TP Price, SL Price) using Decimal precision and market constraints.
        """
        tp_price, sl_price = None, None # Initialize

        # --- Validate Inputs ---
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"TP/SL Calc skipped: Invalid signal '{signal}'.")
            return entry_price_estimate, None, None # Return estimate, no TP/SL

        atr = self.indicator_values.get("ATR") # Expect Decimal
        if not isinstance(atr, Decimal) or not atr.is_finite() or atr <= 0:
            self.logger.warning(f"TP/SL Calc Fail ({signal}): Invalid or non-positive ATR ({atr}). Cannot calculate TP/SL.")
            return entry_price_estimate, None, None

        if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
            self.logger.warning(f"TP/SL Calc Fail ({signal}): Invalid entry price estimate ({entry_price_estimate}).")
            # Cannot calculate reasonable TP/SL without a valid entry point
            return entry_price_estimate, None, None

        try:
            # --- Get Parameters ---
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", "1.0")))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", "1.5")))

            # Fetch precision details using helper methods
            price_prec = self.get_price_precision()
            min_tick = self.get_min_tick_size()
            # Define the quantization factor based on tick size for accurate rounding
            # Use min_tick directly if available and valid, otherwise derive from precision
            quantizer = min_tick if min_tick.is_finite() and min_tick > 0 else Decimal('1e-' + str(price_prec))

            # --- Calculate Raw Offsets ---
            tp_offset = atr * tp_mult
            sl_offset = atr * sl_mult

            # --- Calculate Raw TP/SL Prices ---
            if signal == "BUY":
                tp_raw = entry_price_estimate + tp_offset
                sl_raw = entry_price_estimate - sl_offset
            else: # SELL
                tp_raw = entry_price_estimate - tp_offset
                sl_raw = entry_price_estimate + sl_offset

            # --- Quantize TP/SL using Market Tick Size / Precision ---
            # Round TP towards neutral (less profit) -> Conservative TP
            # Round SL away from neutral (more room) -> Conservative SL (less likely premature stop)
            tp_quantized = None
            if tp_raw.is_finite():
                rounding_mode_tp = ROUND_DOWN if signal == "BUY" else ROUND_UP
                tp_quantized = tp_raw.quantize(quantizer, rounding=rounding_mode_tp)

            sl_quantized = None
            if sl_raw.is_finite():
                # SL Rounding: AWAY from entry price
                rounding_mode_sl = ROUND_DOWN if signal == "BUY" else ROUND_UP
                sl_quantized = sl_raw.quantize(quantizer, rounding=rounding_mode_sl)

            # --- Validation and Final Assignment ---
            final_tp = tp_quantized
            final_sl = sl_quantized

            # Ensure SL is strictly beyond entry by at least one tick
            if final_sl is not None and min_tick > 0: # Check min_tick validity
                if signal == "BUY" and final_sl >= entry_price_estimate:
                    corrected_sl = (entry_price_estimate - min_tick).quantize(quantizer, rounding=ROUND_DOWN)
                    if corrected_sl < final_sl: # Only update if correction actually moved it further
                         self.logger.debug(f"Adjusted BUY SL {final_sl} to be below entry: {corrected_sl}")
                         final_sl = corrected_sl
                    else:
                        final_sl = corrected_sl # Ensure it's at least one tick away even if rounding didn't change much
                        if final_sl >= entry_price_estimate:
                             self.logger.warning(f"BUY SL {final_sl} >= Entry {entry_price_estimate}, correction attempt failed to move it below entry.")
                             final_sl = None # Invalidate SL if it cannot be placed correctly

                elif signal == "SELL" and final_sl <= entry_price_estimate:
                    corrected_sl = (entry_price_estimate + min_tick).quantize(quantizer, rounding=ROUND_UP)
                    if corrected_sl > final_sl: # Only update if correction actually moved it further
                        self.logger.debug(f"Adjusted SELL SL {final_sl} to be above entry: {corrected_sl}")
                        final_sl = corrected_sl
                    else:
                        final_sl = corrected_sl # Ensure it's at least one tick away
                        if final_sl <= entry_price_estimate:
                            self.logger.warning(f"SELL SL {final_sl} <= Entry {entry_price_estimate}, correction attempt failed to move it above entry.")
                            final_sl = None # Invalidate SL


            # Ensure TP offers potential profit (strictly beyond entry)
            if final_tp is not None and min_tick > 0: # Check min_tick validity
                 if signal == "BUY" and final_tp <= entry_price_estimate:
                      corrected_tp = (entry_price_estimate + min_tick).quantize(quantizer, rounding=ROUND_UP)
                      if corrected_tp > final_tp:
                          self.logger.warning(f"BUY TP {final_tp} <= Entry {entry_price_estimate}. Adjusted to {corrected_tp}.")
                          final_tp = corrected_tp
                      else: # If even one tick away is not better than the original invalid TP
                          self.logger.warning(f"BUY TP {final_tp} <= Entry {entry_price_estimate}. Calculation invalid, nullifying TP.")
                          final_tp = None
                 elif signal == "SELL" and final_tp >= entry_price_estimate:
                      corrected_tp = (entry_price_estimate - min_tick).quantize(quantizer, rounding=ROUND_DOWN)
                      if corrected_tp < final_tp:
                          self.logger.warning(f"SELL TP {final_tp} >= Entry {entry_price_estimate}. Adjusted to {corrected_tp}.")
                          final_tp = corrected_tp
                      else:
                          self.logger.warning(f"SELL TP {final_tp} >= Entry {entry_price_estimate}. Calculation invalid, nullifying TP.")
                          final_tp = None

            # Ensure SL/TP are positive numbers
            if final_sl is not None and final_sl <= 0:
                 self.logger.error(f"Calculated SL is zero or negative ({final_sl}). Nullifying SL.")
                 final_sl = None
            if final_tp is not None and final_tp <= 0:
                 self.logger.warning(f"Calculated TP is zero or negative ({final_tp}). Nullifying TP.")
                 final_tp = None

            # --- Log Results ---
            tp_str = f"{final_tp:.{price_prec}f}" if final_tp else "None"
            sl_str = f"{final_sl:.{price_prec}f}" if final_sl else "None"
            self.logger.debug(f"Calc TP/SL ({signal}): EntryEst={entry_price_estimate:.{price_prec}f}, ATR={atr:.{price_prec+2}f}, "
                              f"Tick={min_tick}, TP={tp_str}, SL={sl_str}")

            return entry_price_estimate, final_tp, final_sl

        except Exception as e:
            self.logger.error(f"Unexpected error calculating TP/SL for {signal}: {e}", exc_info=True)
            return entry_price_estimate, None, None


# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches available balance for a specific currency using CCXT, handling V5 specifics.
    Prioritizes 'CONTRACT' account type for Bybit, falls back to default.
    Returns available balance as Decimal, or None if fetch fails or balance not found.
    """
    lg = logger
    balance_info = None
    account_type_tried = None # Keep track of which account type was tried

    # --- Try Fetching with Specific Account Type (Bybit V5) ---
    if exchange.id == 'bybit':
        # Prioritize CONTRACT, could add logic for UNIFIED or SPOT if needed
        # For derivatives (linear specified in init), CONTRACT is usually correct.
        # Unified margin might use UNIFIED. Spot uses SPOT.
        account_type_to_try = 'CONTRACT' # Or determine based on market type?
        lg.debug(f"Attempting Bybit V5 balance fetch for {currency} (Account Type: {account_type_to_try})...")
        try:
            params = {'accountType': account_type_to_try} # Use accountType for V5 fetchBalance override
            balance_info = safe_api_call(exchange.fetch_balance, lg, params=params)
            account_type_tried = account_type_to_try
            lg.debug(f"Raw balance response (Type: {account_type_tried}): {balance_info}")
        except ccxt.ExchangeError as e:
            lg.warning(f"Exchange error fetching balance with type {account_type_to_try}: {e}. Falling back to default fetch.")
            balance_info = None # Ensure fallback occurs
        except Exception as e: # Catch errors from safe_api_call too
             lg.warning(f"Failed fetching balance with type {account_type_to_try}: {e}. Falling back to default fetch.")
             balance_info = None # Ensure fallback occurs

    # --- Fallback to Default Fetch (or if not Bybit or first attempt failed) ---
    if balance_info is None:
        lg.debug(f"Fetching balance for {currency} using default parameters...")
        try:
            balance_info = safe_api_call(exchange.fetch_balance, lg)
            account_type_tried = "Default"
            lg.debug(f"Raw balance response (Type: {account_type_tried}): {balance_info}")
        except Exception as e:
            lg.error(f"Failed to fetch balance info for {currency} even with default params: {e}")
            return None # Both attempts failed

    # --- Parse the balance_info (handle various structures) ---
    if not balance_info:
         lg.error(f"Balance fetch (Type: {account_type_tried}) returned empty or None.")
         return None

    free_balance_str = None

    # Structure 1: Standard CCXT `balance[currency]['free']`
    if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('free') is not None:
        free_balance_str = str(balance_info[currency]['free'])
        lg.debug(f"Found balance via standard ['{currency}']['free']: {free_balance_str}")

    # Structure 2: Top-level `balance['free'][currency]`
    elif not free_balance_str and isinstance(balance_info.get('free'), dict) and balance_info['free'].get(currency) is not None:
         free_balance_str = str(balance_info['free'][currency])
         lg.debug(f"Found balance via top-level 'free' dict: {free_balance_str}")

    # Structure 3: Bybit V5 Specific Parsing (from `info` field - more reliable for V5)
    # This structure can vary based on account type (CONTRACT, UNIFIED, SPOT)
    elif not free_balance_str and exchange.id == 'bybit' and isinstance(balance_info.get('info'), dict):
         # V5 often has a 'result' key, but fetch_balance might parse it differently
         # Look for common patterns:
         # A list of accounts: info['result']['list'][0]...
         # Direct result object: info['result']...
         # Or sometimes directly in info: info['coin']...
         result_data = balance_info['info'].get('result', balance_info['info']) # Use result if present, else info itself

         # Check V5 /v5/account/wallet-balance response format
         if isinstance(result_data.get('list'), list) and result_data['list']:
             # Unified/Contract accounts often return a list
             account_details = result_data['list'][0] # Assume first account in list is relevant
             if isinstance(account_details.get('coin'), list): # Unified account asset list ('coin' is a list)
                 for coin_data in account_details['coin']:
                     if coin_data.get('coin') == currency:
                          # Prioritize availableToWithdraw/availableBalance, fallback to walletBalance
                          free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance')
                          if free is not None: free_balance_str = str(free); break
                          if free_balance_str is None: free_balance_str = str(coin_data.get('walletBalance')) # Fallback
                 if free_balance_str: lg.debug(f"Found Bybit V5 balance via info.result.list[0].coin: {free_balance_str}")

             elif account_details.get('accountType') == account_type_tried: # Contract account structure? (walletBalance, availableBalance)
                 # Check 'coin' key directly under the account details if it's not a list
                 if isinstance(account_details.get(currency), dict): # Older structure? Check just in case
                     free = account_details[currency].get('availableBalance') or account_details[currency].get('walletBalance')
                     if free is not None: free_balance_str = str(free)
                 elif account_details.get('availableBalance') is not None: # More direct V5 structure for CONTRACT
                     free_balance_str = str(account_details['availableBalance'])
                 elif account_details.get('walletBalance') is not None: # Fallback
                     free_balance_str = str(account_details['walletBalance'])

                 if free_balance_str: lg.debug(f"Found Bybit V5 balance via info.result.list[0]: {free_balance_str}")


         # Alternative: Look for the currency directly under 'info' or 'info.result' (less common for V5 balance)
         elif isinstance(result_data.get(currency), dict):
              # Sometimes the direct currency key holds details
              free = result_data[currency].get('available') or result_data[currency].get('free') or result_data[currency].get('availableBalance')
              if free is not None:
                  free_balance_str = str(free)
                  lg.debug(f"Found Bybit V5 balance via info[.result].{currency}: {free_balance_str}")

    # --- Fallback: Use 'total' if 'free' is completely unavailable ---
    if free_balance_str is None:
         total_balance_str = None
         if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('total') is not None:
             total_balance_str = str(balance_info[currency]['total'])
         elif isinstance(balance_info.get('total'), dict) and balance_info['total'].get(currency) is not None:
             total_balance_str = str(balance_info['total'][currency])
         # Add V5 total check if needed (e.g., info.result.list[0].totalEquity or walletBalance)

         if total_balance_str is not None:
              lg.warning(f"{NEON_YELLOW}Could not find 'free' or 'available' balance for {currency}. Using 'total' balance ({total_balance_str}) as fallback. This may include collateral/unrealized PNL.{RESET}")
              free_balance_str = total_balance_str
         else:
              lg.error(f"{NEON_RED}Could not determine any balance ('free', 'available', or 'total') for {currency} after checking known structures.{RESET}")
              lg.debug(f"Full balance_info structure (Type: {account_type_tried}): {json.dumps(balance_info, indent=2)}")
              return None # No balance found

    # --- Convert the found balance string to Decimal ---
    try:
        final_balance = Decimal(free_balance_str)
        if not final_balance.is_finite():
             lg.error(f"Parsed balance for {currency} is not finite ({final_balance}). Treating as zero.")
             final_balance = Decimal('0')
        elif final_balance < 0:
             lg.warning(f"Parsed balance for {currency} is negative ({final_balance}). Treating as zero.")
             final_balance = Decimal('0')

        lg.info(f"Available {currency} balance (Account: {account_type_tried}): {final_balance:.4f}") # Log precision might need adjustment
        return final_balance
    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Failed to convert final balance string '{free_balance_str}' to Decimal for {currency}: {e}")
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information, ensuring markets are loaded, and adds convenience flags."""
    lg = logger
    try:
        # Ensure markets are loaded, reload if necessary using safe_api_call
        if not exchange.markets or symbol not in exchange.markets:
             lg.info(f"Market info for {symbol} not loaded or symbol not found, attempting to load/reload markets...")
             try:
                 # Force reload using safe_api_call for retries
                 safe_api_call(exchange.load_markets, lg, reload=True)
                 lg.info("Markets reloaded successfully.")
             except Exception as load_err:
                  lg.error(f"{NEON_RED}Failed to load/reload markets after retries: {load_err}. Cannot get market info.{RESET}")
                  return None # Cannot proceed without markets

        # Check again after reload attempt
        if symbol not in exchange.markets:
             lg.error(f"{NEON_RED}Market '{symbol}' still not found after reloading markets.{RESET}")
             # Provide hint for common Bybit V5 format
             if '/' in symbol and ':' not in symbol and exchange.id == 'bybit':
                  base, quote = symbol.split('/')
                  suggested_symbol = f"{base}/{quote}:{quote}"
                  lg.warning(f"{NEON_YELLOW}Hint: For Bybit V5 linear perpetuals, try format like '{suggested_symbol}'.{RESET}")
             return None

        # Retrieve the market dictionary
        market = exchange.market(symbol)
        if not market or not isinstance(market, dict):
            # This case should be rare if symbol is in exchange.markets, but check defensively
            lg.error(f"{NEON_RED}Market dictionary structure not found or invalid for validated symbol {symbol}.{RESET}")
            return None

        # --- Add Convenience Flags ---
        market_type = market.get('type', '').lower()
        is_spot = market_type == 'spot'
        is_swap = market_type == 'swap'
        is_future = market_type == 'future'
        is_contract = is_swap or is_future or market.get('contract', False)

        # Linear/Inverse is crucial for contract value calculations
        is_linear = market.get('linear', False)
        is_inverse = market.get('inverse', False)
        # Infer if not explicitly set (common for V5 via defaultType)
        if is_contract and not is_linear and not is_inverse:
            # If defaultType was linear, assume linear. Otherwise, needs check.
            if exchange.options.get('defaultType') == 'linear':
                 is_linear = True
            # Add more specific inference based on quote currency if needed (e.g., quote=USD -> inverse)
            elif market.get('quote','').upper() == 'USD':
                 is_inverse = True
            else: # Default to linear for USDT/USDC etc.
                 is_linear = True

        market['is_spot'] = is_spot
        market['is_contract'] = is_contract
        market['is_linear'] = is_linear
        market['is_inverse'] = is_inverse

        # Log key details for verification
        lg.debug(f"Market Info ({symbol}): ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                 f"Type={market_type}, Contract={is_contract}, Linear={is_linear}, Inverse={is_inverse}, "
                 f"ContractSize={market.get('contractSize', 'N/A')}")
        # Log precision/limits for debugging sizing issues
        lg.debug(f"  Precision: {market.get('precision')}")
        lg.debug(f"  Limits: {market.get('limits')}")

        return market

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Invalid symbol format or symbol not supported by {exchange.id}: '{symbol}'. Error: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # Keep as float from config
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Pass exchange for formatting methods if needed
    logger: Optional[logging.Logger] = None,
    config_override: Optional[Dict] = None # Allow passing config for overrides if needed
) -> Optional[Decimal]:
    """
    Calculates position size based on risk, SL distance, balance, and market constraints.
    Uses Decimal for calculations and applies precision/step limits correctly.
    Handles Linear contracts. Inverse contracts are explicitly not supported yet.
    """
    lg = logger or logging.getLogger(__name__)
    # Reload config within function to get latest values if needed
    cfg = config_override or load_config(CONFIG_FILE)

    # --- Extract Market Info ---
    if not market_info:
        lg.error(f"Size Calc Fail: Missing market info.")
        return None
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quote', cfg.get("quote_currency", "USDT"))
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('is_linear', False)
    is_inverse = market_info.get('is_inverse', False)
    # Unit is base currency for linear/spot
    size_unit = base_currency if is_linear or not is_contract else "Contracts (Inverse?)"

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or not balance.is_finite() or balance <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid or non-positive balance ({balance}).")
        return None
    if not isinstance(risk_per_trade, (float, int)) or not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
        return None
    if not isinstance(initial_stop_loss_price, Decimal) or not initial_stop_loss_price.is_finite() or initial_stop_loss_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid or non-positive initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid or non-positive entry_price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Size Calc Fail ({symbol}): Stop loss price cannot be equal to entry price.")
        return None
    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Size Calc Fail ({symbol}): Market info missing 'limits' or 'precision' dictionary.")
        return None

    # --- Check Contract Type ---
    if is_inverse:
        lg.error(f"{NEON_RED}Inverse contract sizing is not implemented. Aborting sizing for {symbol}.{RESET}")
        return None
    if not is_linear and is_contract:
         lg.warning(f"{NEON_YELLOW}Market {symbol} is a contract but not marked as Linear. Assuming Linear sizing logic. Verify market info.{RESET}")
         # Proceed assuming linear, but warn

    try:
        # --- Initialize Analyzer for Precision/Step ---
        # Pass empty DF, we only need market_info and config here
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, cfg, market_info)
        min_amount_step = analyzer.get_min_amount_step()
        amount_precision_places = analyzer.get_amount_precision_places() # For final formatting if needed

        # --- Calculate Risk Amount ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))

        # --- Calculate SL Distance ---
        sl_distance_per_unit_quote = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit_quote <= 0:
            lg.error(f"Size Calc Fail ({symbol}): Stop loss distance is zero or negative.")
            return None

        # --- Get Contract Size (Value of 1 unit of 'amount' in quote currency) ---
        # For Linear contracts/Spot: contractSize is usually 1 (meaning amount is in base currency)
        # If contractSize is different (e.g., 100 for BTCUSD inverse, or mini contracts), adjust accordingly
        contract_size_val = Decimal('1') # Default for spot/linear
        if is_contract:
            contract_size_str = market_info.get('contractSize')
            if contract_size_str is not None:
                try:
                    cs = Decimal(str(contract_size_str))
                    if cs.is_finite() and cs > 0:
                        contract_size_val = cs
                    else: raise ValueError("Invalid contract size value")
                except (ValueError, TypeError, InvalidOperation):
                    lg.warning(f"Invalid contract size '{contract_size_str}' in market info for {symbol}. Using default value 1.")
            else:
                 lg.warning(f"Contract size not found in market info for {symbol}. Using default value 1 (assuming linear).")


        # --- Calculate Initial Size ---
        # Risk per unit = SL distance (quote/base) * Contract Size (Multiplier, typically 1 for linear)
        # Size (base) = Risk Amount (quote) / Risk per unit (quote/base)

        if not is_inverse: # Linear or Spot
            # Risk per unit is the SL distance in quote currency per unit of base currency, adjusted by contract size multiplier
             risk_per_unit_quote = sl_distance_per_unit_quote * contract_size_val
             if not risk_per_unit_quote.is_finite() or risk_per_unit_quote <= 0:
                 lg.error(f"Size Calc Fail ({symbol}): Risk per unit calculation resulted in zero/negative/NaN ({risk_per_unit_quote}). Check SL distance and contract size.")
                 return None

             calculated_size = risk_amount_quote / risk_per_unit_quote # Result is in Base Currency units (e.g., BTC)

        else: # Inverse case - not handled
             lg.error("Inverse sizing logic required but not implemented.")
             return None


        if not calculated_size.is_finite() or calculated_size <= 0:
            lg.error(f"Initial size calculation resulted in zero/negative/NaN: {calculated_size}. Check inputs (Balance, Risk, SL distance).")
            return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit_quote}")
        lg.info(f"  ContractSize={contract_size_val}, Size Unit={size_unit}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        # Fetch limits with default fallbacks
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})

        min_amount_limit = Decimal(str(amount_limits.get('min', '0')))
        max_amount_limit = Decimal(str(amount_limits.get('max', 'inf')))
        min_cost_limit = Decimal(str(cost_limits.get('min', '0')))
        max_cost_limit = Decimal(str(cost_limits.get('max', 'inf')))

        adjusted_size = calculated_size

        # 1. Apply Amount Step Size (Round DOWN to nearest valid step)
        # Ensure step size is valid before using
        if min_amount_step.is_finite() and min_amount_step > 0:
            original_size_before_step = adjusted_size
            # Use floor division with Decimals
            # adjusted_size = (adjusted_size // min_amount_step) * min_amount_step
            # More robust Decimal quantization for step size rounding DOWN:
            remainder = adjusted_size.remainder_near(min_amount_step) # Find remainder towards nearest step
            if remainder >= 0: # If we are at or above the midpoint towards the next step, round down
                 adjusted_size = (adjusted_size - remainder).quantize(min_amount_step, rounding=ROUND_DOWN)
            else: # If below midpoint, we are closer to the lower step already (or exactly on it)
                 # Ensure it aligns perfectly with the step grid by quantizing
                 adjusted_size = adjusted_size.quantize(min_amount_step, rounding=ROUND_DOWN)

            if adjusted_size != original_size_before_step:
                 lg.info(f"Size adjusted by Amount Step Size ({min_amount_step}): {original_size_before_step:.8f} -> {adjusted_size:.8f} {size_unit}")
        else:
             lg.warning(f"Amount step size is invalid ({min_amount_step}). Skipping step adjustment. Final size might be rejected.")
             # Fallback: round to amount precision places
             quantizer_prec = Decimal('1e-' + str(amount_precision_places))
             adjusted_size = adjusted_size.quantize(quantizer_prec, rounding=ROUND_DOWN)

        # Ensure adjusted size is still positive after rounding
        if adjusted_size <= 0:
            lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Size became zero or negative after step/precision rounding ({adjusted_size}). Increase risk or adjust SL.{RESET}")
            return None

        # 2. Clamp by Amount Limits (Min/Max) AFTER step adjustment
        original_size_before_clamp = adjusted_size
        if min_amount_limit.is_finite() and min_amount_limit > 0 and adjusted_size < min_amount_limit:
             # If rounded size is below min, can we increase it to min? Only if that doesn't violate risk.
             # Decision: Fail the trade if calculated size < min limit. Risk setting should account for this.
             lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Size after step adjustment ({adjusted_size:.8f}) is below minimum limit ({min_amount_limit:.8f}). Increase risk or adjust SL.{RESET}")
             return None # Cannot place order below min amount
        if max_amount_limit.is_finite() and adjusted_size > max_amount_limit:
             adjusted_size = max_amount_limit # Cap at max limit
             # Re-apply step rounding DOWN after capping
             if min_amount_step.is_finite() and min_amount_step > 0:
                  adjusted_size = (adjusted_size // min_amount_step) * min_amount_step
             lg.warning(f"{NEON_YELLOW}Size adjusted by Max Amount Limit: {original_size_before_clamp:.8f} -> {adjusted_size:.8f} {size_unit}{RESET}")

        # 3. Check Cost Limits (Min/Max) with the final adjusted size
        # Estimated Cost = Size (base) * Entry Price (quote/base) * Contract Size (Value Multiplier, usually 1 for linear)
        estimated_cost = adjusted_size * entry_price * contract_size_val
        lg.debug(f"  Cost Check: Final Size={adjusted_size:.8f} {size_unit}, Est. Cost={estimated_cost:.4f} {quote_currency} "
                 f"(Min Limit:{min_cost_limit}, Max Limit:{max_cost_limit})")

        if min_cost_limit.is_finite() and min_cost_limit > 0 and estimated_cost < min_cost_limit:
             lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Estimated cost {estimated_cost:.4f} {quote_currency} is below minimum cost limit {min_cost_limit}. Increase risk/balance or adjust SL.{RESET}")
             return None
        if max_cost_limit.is_finite() and max_cost_limit > 0 and estimated_cost > max_cost_limit:
             # This case means risk % led to a cost exceeding max allowed, cap the size based on max cost.
             # Recalculate size based on max cost: max_size = max_cost / (entry_price * contract_size_val)
             # Then re-apply step size rounding down.
             if entry_price > 0 and contract_size_val > 0:
                  size_based_on_max_cost = max_cost_limit / (entry_price * contract_size_val)
                  # Re-apply step rounding
                  if min_amount_step.is_finite() and min_amount_step > 0:
                       adjusted_size_capped = (size_based_on_max_cost // min_amount_step) * min_amount_step
                  else: # Fallback rounding if step invalid
                       quantizer_prec = Decimal('1e-' + str(amount_precision_places))
                       adjusted_size_capped = size_based_on_max_cost.quantize(quantizer_prec, rounding=ROUND_DOWN)

                  # Ensure capped size is still positive and >= min amount limit before applying
                  if adjusted_size_capped > 0 and (not min_amount_limit.is_finite() or adjusted_size_capped >= min_amount_limit):
                      if adjusted_size_capped < adjusted_size: # Ensure we are reducing the size
                          lg.warning(f"{NEON_YELLOW}Size adjusted by Max Cost Limit: {adjusted_size:.8f} -> {adjusted_size_capped:.8f} {size_unit}{RESET}")
                          adjusted_size = adjusted_size_capped
                      else: # Should not happen if initial cost was > max_cost_limit
                           lg.warning(f"Size capping based on Max Cost Limit resulted in no change or increase. Original Size: {adjusted_size}, Max Cost: {max_cost_limit}")
                  else:
                       lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Size recalculated based on Max Cost Limit ({adjusted_size_capped}) is invalid (<=0 or < Min Amount). Cannot proceed.{RESET}")
                       return None

             else:
                 lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Estimated cost {estimated_cost:.4f} exceeds maximum cost limit {max_cost_limit}, but cannot recalculate size due to zero price/contract size.")
                 return None

        # --- Final Validation ---
        final_size = adjusted_size
        if not final_size.is_finite() or final_size <= 0:
             lg.error(f"{NEON_RED}Final calculated size is zero, negative, or NaN ({final_size}). Aborting trade.{RESET}")
             return None
        # Re-check min amount limit after potential max cost capping
        if min_amount_limit.is_finite() and min_amount_limit > 0 and final_size < min_amount_limit:
             lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Final size after all adjustments ({final_size:.8f}) is below minimum limit ({min_amount_limit:.8f}).{RESET}")
             return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position using fetch_positions with robust parsing for Bybit V5.
    Returns a standardized dictionary for the open position, or None if no position exists.
    Handles potential errors and uses safe_api_call.
    """
    lg = logger
    if not exchange.has.get('fetchPositions') and not exchange.has.get('fetchPosition'): # Check both fetchPositions and fetchPosition
        lg.warning(f"Exchange {exchange.id} supports neither fetchPositions nor fetchPosition. Cannot check position status.")
        return None

    market = get_market_info(exchange, symbol, lg) # Use helper to ensure market info is loaded
    if not market:
        lg.error(f"Cannot get open position: Failed to load market info for {symbol}.")
        return None

    positions: List[Dict] = []
    fetch_method_used = "None"
    market_id = market.get('id')
    if not market_id:
        lg.error(f"Cannot fetch position: Market ID missing for {symbol}.")
        return None

    try:
        # --- Attempt 1: Fetch positions for specific symbol (preferred) ---
        # Note: fetchPosition often returns just one side; fetchPositions(symbols=[...]) gets both
        if exchange.has.get('fetchPositions'):
             fetch_method_used = "fetchPositions (single symbol)"
             lg.debug(f"Attempting {fetch_method_used} for {symbol} (ID: {market_id})...")
             # Bybit V5 Params: need category (linear/inverse)
             category = 'linear' if market.get('is_linear', True) else 'inverse'
             params = {'category': category, 'symbol': market_id} if exchange.id == 'bybit' else {}
             try:
                 # Pass symbol in a list to fetchPositions
                 fetched_data = safe_api_call(exchange.fetch_positions, lg, symbols=[symbol], params=params)
                 if fetched_data is not None: # Can return empty list if no position
                     positions = fetched_data
                     lg.debug(f"{fetch_method_used} successful, found {len(positions)} entries.")
                 else:
                     lg.debug(f"{fetch_method_used} returned None.")
             except ccxt.ArgumentsRequired as e:
                  # Some exchanges don't support filtering fetchPositions by symbol
                  lg.debug(f"{fetch_method_used} filtering not supported ({e}). Falling back to fetch all.")
                  fetch_method_used = "None" # Reset for next attempt
             except Exception as e: # Catch errors from safe_api_call too
                  lg.warning(f"Error during {fetch_method_used} for {symbol}: {e}. Falling back.")
                  fetch_method_used = "None" # Reset for next attempt

        # --- Attempt 2: Fetch ALL positions (fallback) ---
        if not positions and fetch_method_used == "None" and exchange.has.get('fetchPositions'): # Only if previous attempts failed or weren't possible
            fetch_method_used = "fetchPositions (all symbols)"
            lg.debug(f"Attempting {fetch_method_used} as fallback...")
            # Bybit V5 Params: category might still be useful
            category = 'linear' if market.get('is_linear', True) else 'inverse'
            params = {'category': category} if exchange.id == 'bybit' else {}
            try:
                all_positions_data = safe_api_call(exchange.fetch_positions, lg, params=params)
                if all_positions_data:
                     # Filter the results for the target symbol (using market ID for Bybit)
                     target_identifier = market_id if exchange.id == 'bybit' else symbol
                     positions = [p for p in all_positions_data if p.get('symbol') == target_identifier or p.get('info', {}).get('symbol') == target_identifier]
                     lg.debug(f"Fetched {len(all_positions_data)} total positions, found {len(positions)} matching {symbol} (ID: {market_id}).")
                else:
                    lg.warning("Fallback fetch of all positions returned no data.")
            except Exception as e:
                 lg.error(f"Error during fallback fetch of all positions: {e}")
                 return None # Final attempt failed

    except Exception as fetch_err:
         # Catch unexpected errors during the fetch logic itself
         lg.error(f"{NEON_RED}Unexpected error during position fetching process for {symbol}: {fetch_err}{RESET}", exc_info=True)
         return None

    # --- Process the Fetched Position Data ---
    active_position = None
    # Define a sensible threshold for non-zero size using Decimal
    # Use minimum amount step size if available, otherwise a small number
    analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market) # Temp instance for precision/step
    min_step = analyzer.get_min_amount_step()
    size_threshold = min_step if min_step.is_finite() and min_step > 0 else Decimal('1e-9')
    lg.debug(f"Using position size threshold: {size_threshold}")

    if not positions:
        lg.info(f"No position data found for {symbol} after checking via {fetch_method_used}.")
        return None

    # Iterate through the list (usually 0 or 1 entry for one-way mode, maybe 2 for hedge)
    for pos_data in positions:
        if not isinstance(pos_data, dict): continue # Skip invalid entries

        pos_symbol = pos_data.get('symbol') or pos_data.get('info', {}).get('symbol')
        if pos_symbol != market_id: # Compare against market ID for Bybit V5
            lg.debug(f"Skipping position entry for different symbol ID: {pos_symbol} (Expected: {market_id})")
            continue

        # --- Extract Position Size (Crucial for V5) ---
        pos_size_str = None
        # Try standard 'contracts' first
        if pos_data.get('contracts') is not None:
            pos_size_str = str(pos_data['contracts'])
        # Fallback to Bybit V5 'info' field ('size')
        elif isinstance(pos_data.get('info'), dict) and pos_data['info'].get('size') is not None:
             pos_size_str = str(pos_data['info']['size'])
             lg.debug(f"Position size obtained from info.size: {pos_size_str}")
        # Fallback to Bybit V5 'info' field ('contracts', just in case)
        elif isinstance(pos_data.get('info'), dict) and pos_data['info'].get('contracts') is not None:
             pos_size_str = str(pos_data['info']['contracts'])
             lg.debug(f"Position size obtained from info.contracts: {pos_size_str}")


        if pos_size_str is None:
             lg.warning(f"Could not determine position size for an entry of {symbol}. Skipping entry. Data: {pos_data}")
             continue

        # --- Convert Size to Decimal and Check Threshold ---
        try:
            position_size_decimal = Decimal(pos_size_str)
            # Use absolute value comparison against threshold
            if abs(position_size_decimal) >= size_threshold:
                active_position = pos_data.copy() # Work on a copy
                lg.debug(f"Found potential active position entry for {symbol} with size {position_size_decimal}.")
                # Store the decimal size directly
                active_position['contractsDecimal'] = position_size_decimal
                break # Assume only one relevant position per symbol/side/mode (Handle Hedge Mode later if needed)
            else:
                lg.debug(f"Ignoring position entry for {symbol}: Size {position_size_decimal} is below threshold {size_threshold}.")
        except (ValueError, TypeError, InvalidOperation) as parse_err:
            lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol}: {parse_err}. Skipping entry.")
            continue

    # --- Post-Process the Found Active Position ---
    if active_position:
        try:
            # Ensure basic fields exist
            if 'info' not in active_position: active_position['info'] = {}
            info_dict = active_position['info']
            active_position['symbol'] = symbol # Ensure standardized symbol is present

            # --- Standardize Side ---
            # Use 'side' if present, otherwise infer from size or Bybit's info.side
            pos_side = active_position.get('side')
            if pos_side not in ['long', 'short']: pos_side = info_dict.get('side', '').lower() # Check info.side

            if pos_side not in ['long', 'short']:
                size_dec = active_position['contractsDecimal']
                if size_dec >= size_threshold: pos_side = 'long'
                elif size_dec <= -size_threshold: pos_side = 'short'
                else:
                     lg.warning(f"Position size {size_dec} is near zero, cannot determine side reliably. Discarding position.")
                     return None
                lg.debug(f"Inferred position side as '{pos_side}'.")
            active_position['side'] = pos_side

            # --- Standardize Entry Price (Decimal) ---
            # Prioritize info.avgPrice (V5) > entryPrice (ccxt std)
            entry_price_str = info_dict.get('avgPrice') or active_position.get('entryPrice')
            active_position['entryPriceDecimal'] = Decimal(str(entry_price_str)) if entry_price_str is not None else None

            # --- Standardize Liquidation Price (Decimal) ---
            liq_price_str = info_dict.get('liqPrice') or active_position.get('liquidationPrice')
            active_position['liquidationPriceDecimal'] = Decimal(str(liq_price_str)) if liq_price_str is not None and str(liq_price_str).strip() != '0' else None # Ignore '0' liq price

            # --- Standardize Unrealized PNL (Decimal) ---
            pnl_str = info_dict.get('unrealisedPnl') or active_position.get('unrealizedPnl')
            active_position['unrealizedPnlDecimal'] = Decimal(str(pnl_str)) if pnl_str is not None else None

            # --- Extract SL/TP/TSL from 'info' (Bybit V5 is primary source) ---
            # V5 stores these directly in the position info object
            sl_str = info_dict.get('stopLoss')
            tp_str = info_dict.get('takeProfit')
            tsl_dist_str = info_dict.get('trailingStop') # Price distance value (string)
            tsl_act_str = info_dict.get('activePrice')   # Activation price for TSL (string)

            def safe_decimal_or_none(value_str):
                """Helper to convert string to Decimal, return None if invalid or zero."""
                if value_str is None or str(value_str).strip() == '' or str(value_str).strip() == '0':
                    return None
                try:
                    d = Decimal(str(value_str))
                    return d if d.is_finite() else None
                except (InvalidOperation, ValueError, TypeError):
                    return None

            active_position['stopLossPriceDecimal'] = safe_decimal_or_none(sl_str)
            active_position['takeProfitPriceDecimal'] = safe_decimal_or_none(tp_str)
            # Trailing stop distance might be percentage or price, handle as price distance from V5
            active_position['trailingStopLossValueDecimal'] = safe_decimal_or_none(tsl_dist_str) # This is PRICE distance on V5
            active_position['trailingStopActivationPriceDecimal'] = safe_decimal_or_none(tsl_act_str)

            # --- Get Timestamp (Prefer Bybit V5 updatedTime) ---
            # Convert ms timestamp to integer if possible
            timestamp_ms_str = info_dict.get('updatedTime') or active_position.get('timestamp')
            active_position['timestamp_ms'] = None
            timestamp_dt_str = "N/A"
            if timestamp_ms_str is not None:
                try:
                    timestamp_ms = int(timestamp_ms_str)
                    active_position['timestamp_ms'] = timestamp_ms
                    timestamp_dt_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError):
                    timestamp_dt_str = "Invalid Timestamp"

            # --- Log Formatted Position Info ---
            # Use analyzer created earlier for precision
            price_prec = analyzer.get_price_precision()
            amt_prec = analyzer.get_amount_precision_places()

            entry_fmt = f"{active_position['entryPriceDecimal']:.{price_prec}f}" if active_position['entryPriceDecimal'] else 'N/A'
            size_fmt = f"{active_position['contractsDecimal']:.{amt_prec}f}" # Display signed size with correct precision
            liq_fmt = f"{active_position['liquidationPriceDecimal']:.{price_prec}f}" if active_position['liquidationPriceDecimal'] else 'N/A'
            lev_str = info_dict.get('leverage', active_position.get('leverage'))
            lev_fmt = f"{Decimal(str(lev_str)):.1f}x" if lev_str is not None else 'N/A'
            pnl_fmt = f"{active_position['unrealizedPnlDecimal']:.{price_prec}f}" if active_position['unrealizedPnlDecimal'] else 'N/A'
            sl_fmt = f"{active_position['stopLossPriceDecimal']:.{price_prec}f}" if active_position['stopLossPriceDecimal'] else 'None'
            tp_fmt = f"{active_position['takeProfitPriceDecimal']:.{price_prec}f}" if active_position['takeProfitPriceDecimal'] else 'None'
            tsl_d_fmt = f"{active_position['trailingStopLossValueDecimal']:.{price_prec}f}" if active_position['trailingStopLossValueDecimal'] else 'None' # Distance
            tsl_a_fmt = f"{active_position['trailingStopActivationPriceDecimal']:.{price_prec}f}" if active_position['trailingStopActivationPriceDecimal'] else 'None' # Activation Px

            logger.info(f"{NEON_GREEN}Active {pos_side.upper()} position found ({symbol}):{RESET} "
                        f"Size={size_fmt}, Entry={entry_fmt}, Liq={liq_fmt}, Lev={lev_fmt}, PnL={pnl_fmt}, "
                        f"SL={sl_fmt}, TP={tp_fmt}, TSL(Dist/Act): {tsl_d_fmt}/{tsl_a_fmt} (Updated: {timestamp_dt_str})")
            # Log the full processed dict at DEBUG level
            lg.debug(f"Full processed position details: {json.dumps(active_position, default=str, indent=2)}") # Use default=str for Decimal

            # Store entry time if this is the first time seeing this position active
            if symbol not in position_entry_times and active_position['timestamp_ms']:
                 position_entry_times[symbol] = active_position['timestamp_ms']
                 lg.info(f"Recorded position entry time for {symbol}: {timestamp_dt_str}")

            return active_position

        except (ValueError, TypeError, InvalidOperation, KeyError) as proc_err:
             lg.error(f"Error processing active position details for {symbol}: {proc_err}", exc_info=True)
             lg.debug(f"Problematic raw position data: {active_position}")
             # Clear entry time if processing fails after detecting a position
             if symbol in position_entry_times: del position_entry_times[symbol]
             return None # Failed processing

    else:
        # No entries in the fetched list had a size above threshold
        logger.info(f"No active open position found for {symbol} (checked {len(positions)} entries via {fetch_method_used}).")
        # Clear entry time if no position found
        if symbol in position_entry_times:
            lg.debug(f"Clearing recorded entry time for {symbol} as no active position found.")
            del position_entry_times[symbol]
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """
    Sets leverage using CCXT's set_leverage method, handling Bybit V5 specifics.
    Returns True on success or if leverage is already set, False on failure.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.debug(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # No action needed, considered success

    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be a positive integer.")
        return False

    # Check if exchange explicitly supports setLeverage
    if not exchange.has.get('setLeverage'):
        # Some exchanges might handle leverage via set_margin_mode (less common for setting level directly)
        lg.warning(f"Exchange {exchange.id} does not explicitly support setLeverage via ccxt. Attempting anyway...")
        # Proceed, but be aware it might fail or require different methods

    # Get market ID (needed for Bybit V5 params)
    market_id = market_info.get('id')
    if not market_id:
         lg.error(f"Cannot set leverage: Market ID missing for symbol {symbol}.")
         return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} (ID: {market_id}) to {leverage}x...")
        params = {}
        # --- Bybit V5 Specific Parameters ---
        if exchange.id == 'bybit':
            # V5 requires buyLeverage and sellLeverage as strings
            # Ensure leverage is formatted correctly (e.g., "10", not "10.0")
            leverage_str = str(int(leverage))
            params = {'buyLeverage': leverage_str, 'sellLeverage': leverage_str}
            # V5 also needs category
            category = 'linear' if market_info.get('is_linear', True) else 'inverse'
            params['category'] = category
            # Optional: Specify margin mode if needed (e.g., isolated)
            # params['marginMode'] = 'ISOLATED' # or 'CROSSED' (default usually CROSSED)
            lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        # Use safe_api_call to execute
        response = safe_api_call(exchange.set_leverage, lg, leverage, symbol, params=params)

        # --- Check Response (can vary) ---
        lg.debug(f"Set leverage raw response for {symbol}: {response}") # Log raw response for debugging
        # Check Bybit specific success/info codes in response if possible
        if exchange.id == 'bybit' and isinstance(response, dict):
             ret_code = response.get('retCode', -1) # V5 uses retCode
             ret_msg = response.get('retMsg', '')
             if ret_code == 0:
                  lg.info(f"{NEON_GREEN}Leverage for {symbol} set/confirmed to {leverage}x.{RESET}")
                  return True
             # Handle "Leverage not modified" code (e.g., 110045) as success
             elif ret_code == 110045 or "leverage not modified" in ret_msg.lower():
                  lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange code: {ret_code}).{RESET}")
                  return True
             else:
                  # Log specific Bybit error if code indicates failure
                  lg.error(f"{NEON_RED}Failed to set leverage via Bybit API: {ret_msg} (Code: {ret_code}){RESET}")
                  # Provide hints for common Bybit leverage errors
                  if ret_code in [110028, 110009, 110055] or "margin mode" in ret_msg.lower():
                      lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) is compatible with leverage setting.{RESET}")
                  elif ret_code == 110044 or "risk limit" in ret_msg.lower():
                      lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed the current Risk Limit tier for {symbol}. Adjust Risk Limit first.{RESET}")
                  elif ret_code == 110013 or "parameter error" in ret_msg.lower():
                      lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid or out of the allowed range for {symbol}.{RESET}")
                  return False # Failed based on Bybit response code

        # Generic success assumption if no specific error code and no exception raised
        lg.info(f"{NEON_GREEN}Leverage set request for {symbol} to {leverage}x sent successfully (verify confirmation).{RESET}")
        return True

    except ccxt.ExchangeError as e:
        # Catch errors potentially missed by Bybit-specific checks or for other exchanges
        err_str = str(e).lower()
        code = getattr(e, 'code', None) # Try to get standard CCXT error code if available
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {code}){RESET}")
        # Check for common messages indicating already set
        if "leverage not modified" in err_str or "same leverage" in err_str:
             lg.info(f"{NEON_YELLOW}Leverage for {symbol} likely already set to {leverage}x (Message: {e}).{RESET}")
             return True # Treat as success
        # Add hints for other exchanges if known errors exist
    except Exception as e:
        # Catch errors from safe_api_call or other unexpected issues
        lg.error(f"{NEON_RED}Failed to set leverage for {symbol} after retries or due to unexpected error: {e}{RESET}", exc_info=False)

    return False # Return False if any error occurred


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal,
    market_info: Dict,
    logger: Optional[logging.Logger] = None,
    order_type: str = 'market',
    limit_price: Optional[Decimal] = None,
    reduce_only: bool = False,
    params: Optional[Dict] = None # For extra exchange-specific params (e.g., SL/TP on order)
) -> Optional[Dict]:
    """
    Places an order (market or limit) using CCXT with retries, V5 parameters, and enhanced logging.
    Returns the order dictionary on success, None on failure.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', '')
    # Use amount precision for formatting size
    analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance for precision
    amt_prec_places = analyzer.get_amount_precision_places()
    price_prec_places = analyzer.get_price_precision()
    size_unit = base_currency if is_contract or market_info.get('is_spot', False) else "Contracts" # Adjust unit display
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"

    # --- Validate Inputs ---
    if not isinstance(position_size, Decimal) or not position_size.is_finite() or position_size <= 0:
        lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Invalid or non-positive position size {position_size}.")
        return None
    try:
        # CCXT generally requires amount as float, but format using exchange helper first
        amount_str = exchange.amount_to_precision(symbol, float(position_size))
        amount_float = float(amount_str) # Convert the *formatted string* back to float
        if amount_float <= 0: raise ValueError("Position size float conversion resulted in non-positive value after formatting")
    except (ValueError, TypeError, AttributeError, ccxt.ExchangeError) as e:
        lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Failed to convert/format size {position_size}: {e}")
        return None

    price_float = None
    price_str = None
    if order_type == 'limit':
         if not isinstance(limit_price, Decimal) or not limit_price.is_finite() or limit_price <= 0:
             lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Limit order selected but valid positive limit_price not provided ({limit_price}).")
             return None
         try:
             # Format price using exchange helper, then convert formatted string to float
             price_str = exchange.price_to_precision(symbol, float(limit_price))
             price_float = float(price_str)
             if price_float <= 0: raise ValueError("Limit price float conversion resulted in non-positive value after formatting")
         except (ValueError, TypeError, AttributeError, ccxt.ExchangeError):
              lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Invalid limit_price format or failed formatting ({limit_price}).")
              return None
    elif order_type != 'market':
        lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Unsupported order type '{order_type}'. Use 'market' or 'limit'.")
        return None

    # --- Prepare CCXT Parameters ---
    # Base parameters required by create_order
    order_args = {
        'symbol': symbol, # Use the standardized symbol
        'type': order_type,
        'side': side,
        'amount': amount_float, # Use the formatted float amount
        'price': price_float, # Use the formatted float price (None for market orders)
    }

    # Additional parameters for 'params' argument
    ccxt_params = {}
    if reduce_only:
        ccxt_params['reduceOnly'] = True
        # For Bybit Market ReduceOnly, use IOC to prevent it resting if market moves away
        if order_type == 'market' and exchange.id == 'bybit':
             ccxt_params['timeInForce'] = 'IOC' # ImmediateOrCancel

    # Bybit V5 Specifics
    if exchange.id == 'bybit':
        # positionIdx: 0 for one-way mode, 1 for Buy Hedge, 2 for Sell Hedge
        ccxt_params['positionIdx'] = 0 # Assume One-Way mode as default
        # Add category (linear/inverse) - essential for V5
        category = 'linear' if market_info.get('is_linear', True) else 'inverse'
        ccxt_params['category'] = category
        # Add SL/TP directly to order if needed (V5 feature)
        # This should be passed in the 'params' dict if desired
        # Example: params={'stopLoss': sl_price_str, 'takeProfit': tp_price_str, 'tpslMode': 'Full'}

    # Merge external params, ensuring our essential params override
    if params and isinstance(params, dict):
        # Prioritize params passed externally, but don't let them overwrite core args/known V5 params
        core_keys = ['symbol', 'type', 'side', 'amount', 'price', 'reduceOnly', 'positionIdx', 'timeInForce', 'category']
        filtered_external_params = {k: v for k, v in params.items() if k not in core_keys}
        ccxt_params.update(filtered_external_params)

    order_args['params'] = ccxt_params

    # --- Log Order Details ---
    size_fmt = f"{position_size:.{amt_prec_places}f}" # Format original Decimal size for logging
    limit_fmt = f"{limit_price:.{price_prec_places}f}" if order_type == 'limit' and limit_price else "N/A"

    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type.upper()} order for {symbol}:")
    lg.info(f"  Size: {size_fmt} {size_unit} (Formatted Amount: {amount_float})")
    if order_type == 'limit': lg.info(f"  Limit Price: {limit_fmt} (Formatted Price: {price_float})")
    lg.info(f"  ReduceOnly: {reduce_only}")
    if ccxt_params: lg.info(f"  Params: {ccxt_params}")


    # --- Execute Order via safe_api_call ---
    try:
        # Use safe_api_call with unpacked arguments
        order_response = safe_api_call(
            exchange.create_order, lg,
            **order_args # Unpack symbol, type, side, amount, price, params
        )

        # --- Process Response ---
        if order_response and isinstance(order_response, dict) and order_response.get('id'):
            # Order placed successfully (or at least accepted by the exchange)
            order_id = order_response.get('id')
            status = order_response.get('status', 'unknown')
            filled_amount_str = str(order_response.get('filled', 0.0))
            avg_price_str = str(order_response.get('average')) if order_response.get('average') is not None else None

            # Format filled amount and avg price
            try: filled_fmt = f"{Decimal(filled_amount_str):.{amt_prec_places}f}"
            except: filled_fmt = filled_amount_str
            try: avg_price_fmt = f"{Decimal(avg_price_str):.{price_prec_places}f}" if avg_price_str else 'N/A'
            except: avg_price_fmt = avg_price_str if avg_price_str else 'N/A'


            lg.info(f"{NEON_GREEN}{action_desc} Order Placed/Accepted!{RESET}")
            lg.info(f"  ID: {order_id}, Status: {status}, Filled: {filled_fmt}, AvgPx: {avg_price_fmt}")
            lg.debug(f"Raw order response: {json.dumps(order_response, default=str)}")

            # Return the full order dictionary for further processing (e.g., checking status)
            return order_response
        else:
             # Handle cases where safe_api_call returned None (retries failed) or response lacks ID
             # Check for Bybit V5 error messages in response even if 'id' is missing
             if exchange.id == 'bybit' and isinstance(order_response, dict):
                 ret_code = order_response.get('retCode', -1)
                 ret_msg = order_response.get('retMsg', '')
                 if ret_code != 0:
                      lg.error(f"{NEON_RED}Order placement failed ({symbol}): Bybit API Error - {ret_msg} (Code: {ret_code}){RESET}")
                      lg.debug(f"Failed Order Response: {order_response}")
                      return None # Explicit failure
             # Generic failure message
             lg.error(f"{NEON_RED}Order placement failed for {symbol}. Response was invalid or missing ID. Response: {order_response}{RESET}")
             return None

    # --- Specific Error Handling ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {action_desc} {side} order for {symbol}: {e}{RESET}")
        # Check balance again here?
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {action_desc} {side} order ({symbol}): {e}{RESET}")
        err_msg = str(e).lower()
        code = getattr(e, 'code', None)
        if "price" in err_msg or "tick size" in err_msg or (code == 110006 and exchange.id == 'bybit'): lg.error(f"{NEON_YELLOW} >> Hint: Check price precision / tick size. Limit Price: {limit_fmt}{RESET}")
        if "size" in err_msg or "step size" in err_msg or "lot size" in err_msg or (code == 110005 and exchange.id == 'bybit'): lg.error(f"{NEON_YELLOW} >> Hint: Check amount precision / step size. Size: {size_fmt}{RESET}")
        if "minnotional" in err_msg or "minimum order value" in err_msg or "cost" in err_msg or (code == 110007 and exchange.id == 'bybit'): lg.error(f"{NEON_YELLOW} >> Hint: Order cost likely below minimum required.{RESET}")
        if "reduceonly" in err_msg or "reduce-only" in err_msg or (code == 110014 and exchange.id == 'bybit'):
            lg.error(f"{NEON_YELLOW} >> Hint: Reduce-only order failed. Position might be smaller than order size, already closed, or incorrect side.{RESET}")
        if "positionidx" in err_msg and exchange.id == 'bybit': lg.error(f"{NEON_YELLOW} >> Hint: positionIdx issue - check if Hedge Mode is active or if index is incorrect.{RESET}")
    except ccxt.ExchangeError as e:
        # Catch other exchange-specific errors
        code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action_desc} {side} order ({symbol}): {e} (Code: {code}){RESET}")
        # Add hints for specific Bybit codes if needed (e.g., risk limit errors)
        if exchange.id == 'bybit':
            if code == 110044 or "risk limit" in str(e).lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: Position size might exceed risk limit tier for {symbol}.{RESET}")
            if code == 130021: # Order not found / already filled/cancelled (can happen with fast markets/retries)
                 lg.warning(f"{NEON_YELLOW} >> Hint(130021): Order not found/already filled/cancelled. May need to re-check position state.{RESET}")

    except Exception as e:
        # Catch errors from safe_api_call or other unexpected issues
        lg.error(f"{NEON_RED}Unexpected error placing {action_desc} {side} order ({symbol}): {e}{RESET}", exc_info=False)

    return None # Return None if any exception occurred


def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    logger: logging.Logger, stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None, trailing_stop_distance: Optional[Decimal] = None,
    tsl_activation_price: Optional[Decimal] = None,
) -> bool:
    """
    Internal helper using Bybit V5 private API to set SL, TP, or TSL for an existing position.
    Handles parameter formatting and API call. Returns True on success/no change needed, False on failure.
    NOTE: This uses a private endpoint and assumes Bybit V5 structure.
    """
    lg = logger
    # --- Pre-checks ---
    if 'bybit' not in exchange.id.lower():
        lg.error("Protection setting via private_post('/v5/position/set-trading-stop') is specific to Bybit V5.")
        return False
    if not market_info or not market_info.get('is_contract'):
        lg.warning(f"Protection setting skipped for {symbol}: Market is not a contract or market_info missing.")
        return False
    if not position_info or not isinstance(position_info, dict):
        lg.error(f"Cannot set protection for {symbol}: Missing or invalid position_info dictionary.")
        return False

    # --- Extract required info from position_info ---
    pos_side = position_info.get('side')
    # Use Decimal entry price if available, fallback carefully
    entry_price = position_info.get('entryPriceDecimal')
    if entry_price is None and position_info.get('entryPrice'):
        try: entry_price = Decimal(str(position_info['entryPrice']))
        except: entry_price = None

    current_sl = position_info.get('stopLossPriceDecimal')
    current_tp = position_info.get('takeProfitPriceDecimal')
    current_tsl_dist = position_info.get('trailingStopLossValueDecimal') # Price distance
    current_tsl_act = position_info.get('trailingStopActivationPriceDecimal')
    pos_idx_str = position_info.get('info', {}).get('positionIdx') # From Bybit V5 info

    # Validate essential position details
    if pos_side not in ['long', 'short']:
        lg.error(f"Cannot set protection ({symbol}): Invalid position side '{pos_side}'.")
        return False
    if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
        lg.error(f"Cannot set protection ({symbol}): Invalid or missing entry price '{entry_price}'.")
        return False
    try:
        # Determine positionIdx (0 for one-way, 1/2 for hedge)
        pos_idx = int(pos_idx_str) if pos_idx_str is not None else 0
    except (ValueError, TypeError):
        lg.warning(f"Could not parse positionIdx '{pos_idx_str}' for {symbol}, using default 0 (One-Way).")
        pos_idx = 0

    # --- Validate and Prepare Target Protection Parameters ---
    target_sl, target_tp, target_tsl_dist, target_tsl_act = None, None, None, None
    clear_sl, clear_tp, clear_tsl = False, False, False
    needs_update = False # Flag to check if API call is necessary
    analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # For precision/tick
    min_tick = analyzer.get_min_tick_size()

    # --- Helper to check if Decimal price is valid ---
    def is_valid_price(p):
        return isinstance(p, Decimal) and p.is_finite() and p > 0

    # --- Helper to check if two optional Decimal prices are different ---
    def prices_differ(p1: Optional[Decimal], p2: Optional[Decimal]) -> bool:
        p1_valid = is_valid_price(p1)
        p2_valid = is_valid_price(p2)
        if p1_valid != p2_valid: return True # One is valid, the other isn't
        if not p1_valid: return False # Both invalid/None
        # Both are valid Decimals, check if they are different
        return p1 != p2

    # Validate Stop Loss
    if stop_loss_price is None:
        if is_valid_price(current_sl): clear_sl = True; needs_update = True
        target_sl = None
    elif is_valid_price(stop_loss_price):
        if (pos_side == 'long' and stop_loss_price < entry_price) or \
           (pos_side == 'short' and stop_loss_price > entry_price):
            target_sl = stop_loss_price
            if prices_differ(target_sl, current_sl): needs_update = True
        else:
            lg.warning(f"Invalid SL price {stop_loss_price} for {pos_side} position with entry {entry_price}. Ignoring target SL.")
    else:
        lg.warning(f"Ignoring invalid target SL value: {stop_loss_price}")

    # Validate Take Profit
    if take_profit_price is None:
        if is_valid_price(current_tp): clear_tp = True; needs_update = True
        target_tp = None
    elif is_valid_price(take_profit_price):
        if (pos_side == 'long' and take_profit_price > entry_price) or \
           (pos_side == 'short' and take_profit_price < entry_price):
            target_tp = take_profit_price
            if prices_differ(target_tp, current_tp): needs_update = True
        else:
            lg.warning(f"Invalid TP price {take_profit_price} for {pos_side} position with entry {entry_price}. Ignoring target TP.")
    else:
        lg.warning(f"Ignoring invalid target TP value: {take_profit_price}")

    # Validate Trailing Stop (Distance and Activation Price)
    # Distance must be positive Decimal, Activation must be positive Decimal beyond entry
    target_tsl_valid = False
    if trailing_stop_distance is None or tsl_activation_price is None:
        if is_valid_price(current_tsl_dist): # If currently active TSL needs clearing
             clear_tsl = True; needs_update = True
        target_tsl_dist, target_tsl_act = None, None
    elif is_valid_price(trailing_stop_distance) and is_valid_price(tsl_activation_price):
        # Check activation price validity relative to entry price
        if (pos_side == 'long' and tsl_activation_price > entry_price) or \
           (pos_side == 'short' and tsl_activation_price < entry_price):
            target_tsl_dist = trailing_stop_distance
            target_tsl_act = tsl_activation_price
            target_tsl_valid = True
            # Check if TSL values changed
            if prices_differ(target_tsl_dist, current_tsl_dist) or prices_differ(target_tsl_act, current_tsl_act):
                 needs_update = True
        else:
             lg.warning(f"Invalid TSL Activation price {tsl_activation_price} for {pos_side} position with entry {entry_price}. Ignoring target TSL.")
    else:
         lg.warning(f"Ignoring invalid/incomplete target TSL values: Dist={trailing_stop_distance}, Act={tsl_activation_price}")

    # --- Additional Checks for Update Need ---
    # If switching from Fixed SL to TSL
    if target_tsl_valid and is_valid_price(current_sl): needs_update = True
    # If switching from TSL to Fixed SL
    if is_valid_price(target_sl) and is_valid_price(current_tsl_dist): needs_update = True


    # --- Prepare API Parameters if Update is Needed ---
    if not needs_update:
        lg.debug(f"No protection update needed for {symbol}. Current state matches target.")
        return True # Nothing to do, considered success

    # Determine category (linear/inverse) - needed for V5 endpoint
    category = 'linear' if market_info.get('is_linear', True) else 'inverse' # Assume linear if not specified
    market_id = market_info.get('id')
    if not market_id: lg.error("Market ID missing, cannot set protection."); return False

    params = {
        'category': category,
        'symbol': market_id,
        'positionIdx': pos_idx,
        # V5 recommends 'Full' mode for setting SL/TP per position
        'tpslMode': 'Full',
        # Trigger prices (LastPrice, MarkPrice, IndexPrice) - Make configurable? Default to LastPrice for now.
        'tpTriggerBy': 'LastPrice',
        'slTriggerBy': 'LastPrice',
        'trailingStop': '0', # Initialize fields to clear existing ones if not explicitly set
        'stopLoss': '0',
        'takeProfit': '0',
        'activePrice': '0',
    }
    log_parts = [f"Updating protection for {symbol} ({pos_side.upper()} PosIdx:{pos_idx}):"]

    try: # Format parameters to strings with correct precision
        # Use exchange formatting methods where possible
        def fmt_price(p: Optional[Decimal], clear_flag: bool = False) -> Optional[str]:
            if p is None and not clear_flag: return None # Don't send if not setting and not clearing
            if clear_flag or (isinstance(p, Decimal) and p == 0): return '0' # Explicitly allow '0' to clear
            if not is_valid_price(p): return None # Ignore invalid prices
            try: return exchange.price_to_precision(symbol, float(p))
            except Exception as fmt_e: lg.warning(f"Could not format price {p}: {fmt_e}"); return None

        def fmt_tsl_distance(d: Optional[Decimal], clear_flag: bool = False) -> Optional[str]: # TSL distance might need different formatting/precision
            if d is None and not clear_flag: return None
            if clear_flag or (isinstance(d, Decimal) and d == 0): return '0'
            if not is_valid_price(d): return None # Distance must be positive
            try:
                # Use price precision for TSL distance as it's a price value
                # Ensure distance >= min tick size
                dist_float = float(d)
                if min_tick.is_finite() and min_tick > 0 and dist_float < float(min_tick):
                    lg.warning(f"TSL distance {d} is less than min tick {min_tick}. Adjusting up.")
                    dist_float = float(min_tick)
                return exchange.price_to_precision(symbol, dist_float)
            except Exception as fmt_e: lg.warning(f"Could not format TSL distance {d}: {fmt_e}"); return None

        # Set TSL parameters first (as it might override SL)
        # Only set TSL if target_tsl_valid is True
        if target_tsl_valid:
            tsl_d_fmt = fmt_tsl_distance(target_tsl_dist)
            tsl_a_fmt = fmt_price(target_tsl_act)

            if tsl_d_fmt is not None and tsl_a_fmt is not None and tsl_d_fmt != '0':
                 params['trailingStop'] = tsl_d_fmt
                 params['activePrice'] = tsl_a_fmt
                 log_parts.append(f"  Set TSL Dist: {tsl_d_fmt}, ActPx: {tsl_a_fmt}")
                 # Clear target SL as TSL takes precedence
                 target_sl = None
                 clear_sl = True # Ensure fixed SL is cleared in params
            else:
                 lg.error(f"Failed formatting TSL params (Dist: {target_tsl_dist}, Act: {target_tsl_act}). TSL change aborted.")
                 # If formatting failed, ensure TSL is cleared if it was intended
                 if clear_tsl: params['trailingStop'] = '0'; params['activePrice'] = '0'

        elif clear_tsl: # Clear existing TSL if requested
            params['trailingStop'] = '0'
            params['activePrice'] = '0'
            log_parts.append("  Clear TSL")

        # Set Fixed SL (only if TSL wasn't set or is being cleared)
        sl_fmt = fmt_price(target_sl, clear_sl)
        if sl_fmt is not None:
            params['stopLoss'] = sl_fmt
            log_parts.append(f"  Set FixSL: {sl_fmt}")
        elif target_sl is not None: # Log error if formatting failed for a non-None target
             lg.error("Failed formatting Fixed SL price. SL change aborted.")

        # Set Fixed TP
        tp_fmt = fmt_price(target_tp, clear_tp)
        if tp_fmt is not None:
            params['takeProfit'] = tp_fmt
            log_parts.append(f"  Set FixTP: {tp_fmt}")
        elif target_tp is not None:
             lg.error("Failed formatting Fixed TP price. TP change aborted.")

    except Exception as fmt_err:
        lg.error(f"Unexpected error formatting protection parameters: {fmt_err}", exc_info=True)
        return False

    # Remove '0' values if they were just placeholders and not explicit clear requests
    final_params = {}
    for k, v in params.items():
        is_clear_param = k in ['stopLoss', 'takeProfit', 'trailingStop', 'activePrice']
        should_clear = (k == 'stopLoss' and clear_sl) or \
                       (k == 'takeProfit' and clear_tp) or \
                       (k in ['trailingStop', 'activePrice'] and clear_tsl)

        if is_clear_param and v == '0' and not should_clear:
            # If value is '0' but we didn't explicitly intend to clear this param, skip it
            continue
        final_params[k] = v

    # Check if any valid parameters were actually added to the request
    valid_keys_to_send = ['stopLoss', 'takeProfit', 'trailingStop'] # activePrice sent with trailingStop
    if not any(key in final_params and final_params[key] != '0' for key in valid_keys_to_send) and \
       not any([clear_sl, clear_tp, clear_tsl]): # Or if no explicit clear actions needed
        lg.warning(f"No effective protection parameters to update for {symbol}. No API call made.")
        return True


    # --- Make the API Call ---
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call Params: {final_params}")
    endpoint = '/v5/position/set-trading-stop'
    try:
        # Use safe_api_call for the private POST request
        response = safe_api_call(exchange.private_post, lg, endpoint, params=final_params)
        lg.debug(f"Set protection raw response: {json.dumps(response, default=str)}") # Log raw response

        # --- Process Bybit V5 Response ---
        if isinstance(response, dict):
            code = response.get('retCode')
            msg = response.get('retMsg', 'Unknown Error')
            ext = response.get('retExtInfo', {}) # Can contain useful details

            if code == 0:
                # Success, check message for nuances
                if "not modified" in msg.lower() or msg in ["success", "ok", "Success"]:
                    lg.info(f"{NEON_GREEN}Protection update successful or no change needed ({symbol}). Msg: {msg}{RESET}")
                else: # Unexpected success message? Log it.
                     lg.info(f"{NEON_GREEN}Protection update potentially successful ({symbol}). Msg: {msg}{RESET}")
                return True
            else:
                # Failure
                lg.error(f"{NEON_RED}Failed to set protection ({symbol}): {msg} (Code: {code}) ExtInfo: {ext}{RESET}")
                # Add specific hints based on V5 codes if known
                if code == 110013: lg.error(" >> Hint(110013): Parameter Error - Check prices vs entry/mark/index, tick size, TSL values.")
                if code == 110036: lg.error(f" >> Hint(110036): TSL Activation Price '{params.get('activePrice')}' might be invalid (too close, wrong side?).")
                if code == 110086: lg.error(" >> Hint(110086): SL price equals TP price.")
                if code == 110025: lg.error(" >> Hint(110025): Position not found (already closed?) or posIdx mismatch.")
                if code == 110043: lg.error(" >> Hint(110043): SL/TP price triggered immediately (too close to current price?).")
                if code == 170024: lg.error(" >> Hint(170024): Order cost not met for SL/TP order (check size/liquidity?).")
                return False
        else:
            lg.error(f"Received unexpected response format from {endpoint}: {response}")
            return False

    except Exception as e:
        # Catch errors from safe_api_call or other issues
        lg.error(f"Failed API call to {endpoint} for {symbol}: {e}", exc_info=False)
        return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    config: Dict[str, Any], logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional: Set TP concurrently
) -> bool:
    """
    Calculates Trailing Stop Loss parameters based on config percentages and current position,
    then calls _set_position_protection to apply it via Bybit V5 API.

    Note: Bybit V5 TSL uses a fixed *price distance* trail, not a percentage callback rate directly.
          This function calculates an appropriate distance based on the activation price and config rate.
    """
    lg = logger
    if not config.get("enable_trailing_stop"):
        lg.debug(f"Trailing Stop Loss is disabled in config for {symbol}.")
        return False # Not an error, just disabled

    try: # --- Validate Inputs from Config and Position ---
        # trailing_stop_callback_rate: % of activation price used for trail distance
        callback_rate_pct_str = str(config["trailing_stop_callback_rate"])
        # trailing_stop_activation_percentage: % move from entry price to trigger activation
        activation_trigger_pct_str = str(config["trailing_stop_activation_percentage"])

        callback_rate_pct = Decimal(callback_rate_pct_str)
        activation_trigger_pct = Decimal(activation_trigger_pct_str)

        entry_price = position_info.get('entryPriceDecimal')
        pos_side = position_info.get('side')

        # Validate config values
        if not (0 < callback_rate_pct < 1): raise ValueError(f"Invalid trailing_stop_callback_rate ({callback_rate_pct}). Must be between 0 and 1.")
        if not (0 <= activation_trigger_pct < 1): raise ValueError(f"Invalid trailing_stop_activation_percentage ({activation_trigger_pct}). Must be between 0 (inclusive) and 1.")
        # Validate position info
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0: raise ValueError("Invalid entry price in position_info.")
        if pos_side not in ['long','short']: raise ValueError("Invalid side in position_info.")

    except (KeyError, ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Invalid TSL config or position info for {symbol}: {e}.", exc_info=False)
        return False

    try: # --- Calculate TSL Parameters (Activation Price and Trail Distance) ---
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance for precision/tick
        price_prec = analyzer.get_price_precision()
        min_tick = analyzer.get_min_tick_size()
        quantizer = min_tick if min_tick.is_finite() and min_tick > 0 else Decimal(f'1e-{price_prec}')

        # 1. Calculate Activation Price
        activation_offset = entry_price * activation_trigger_pct
        raw_activation_price = None
        rounding_mode_act = None

        if pos_side == 'long':
             raw_activation_price = entry_price + activation_offset
             rounding_mode_act = ROUND_UP # Activate slightly later/higher for long
        else: # short
             raw_activation_price = entry_price - activation_offset
             rounding_mode_act = ROUND_DOWN # Activate slightly later/lower for short

        activation_price = raw_activation_price.quantize(quantizer, rounding=rounding_mode_act)

        # Ensure activation price is strictly beyond entry price by at least one tick
        if min_tick > 0:
            if pos_side == 'long' and activation_price <= entry_price:
                 activation_price = (entry_price + min_tick).quantize(quantizer, rounding=ROUND_UP)
                 lg.debug(f"Adjusted TSL Activation price (long) to be > entry: {activation_price}")
            elif pos_side == 'short' and activation_price >= entry_price:
                 activation_price = (entry_price - min_tick).quantize(quantizer, rounding=ROUND_DOWN)
                 lg.debug(f"Adjusted TSL Activation price (short) to be < entry: {activation_price}")

        if not activation_price.is_finite() or activation_price <= 0:
            lg.error(f"Invalid TSL Activation price calculated ({activation_price}) for {symbol}. Cannot set TSL."); return False

        # 2. Calculate Trail Distance (as price difference, based on activation price)
        # Trail Distance = Activation Price * Callback Rate (%)
        trail_distance_raw = activation_price * callback_rate_pct

        # Quantize distance: Round up to the nearest tick size to ensure it's a valid increment
        trail_distance = trail_distance_raw.quantize(quantizer, rounding=ROUND_UP)

        # Ensure distance is at least one tick size
        if min_tick > 0 and trail_distance < min_tick:
            lg.debug(f"Calculated TSL trail distance ({trail_distance}) was less than min tick ({min_tick}). Adjusting up.")
            trail_distance = min_tick

        if not trail_distance.is_finite() or trail_distance <= 0:
            lg.error(f"Invalid TSL Trail Distance calculated
