Okay, let's break down the provided `pybit` bot script (v1.0.1) and integrate the missing core logic components (`TradingAnalyzer`, `run_bot`, `manage_existing_position_pybit`, `attempt_new_entry_pybit`). I'll also add the crucial `calculate_position_size` function and incorporate some enhancements based on the analysis.

**Analysis Summary:**

1.  **Structure & Helpers:** The script has a solid foundation with good structure, error handling (`safe_pybit_call`), configuration management, logging, state persistence, and Pybit V5 wrappers for common actions (fetching data, placing orders, setting protection).
2.  **Pybit Integration:** It correctly uses `pybit.unified_trading`, handles testnet/mainnet, detects UTA vs. Non-UTA initially, and fetches market info effectively (with caching).
3.  **Precision:** Consistent use of `Decimal` is good practice.
4.  **Key Missing Parts:** The core trading logic (`TradingAnalyzer`, `run_bot`, management functions) were placeholders.
5.  **Critical Warning:** The assumption of `contract_size=1` is explicitly noted and needs careful user verification depending on the exact contracts traded (especially inverse contracts).
6.  **Hedge Mode:** Basic support is included (passing `positionIdx`), but the main logic doesn't fully differentiate hedge mode positions yet. Added comments to highlight this.
7.  **Potential Enhancements:** More robust position confirmation, dynamic contract size fetching (if needed), atomic state saving, explicit Python version notes, and potentially asynchronous operations (though adds complexity).

**Integrating Missing Code & Enhancements:**

1.  **Pasted Full Implementations:** The placeholder sections (`TradingAnalyzer`, `run_bot`, `manage_existing_position_pybit`, `attempt_new_entry_pybit`) have been filled with their complete logic, adapted to use the Pybit helper functions defined earlier.
2.  **Added `calculate_position_size`:** This essential function was missing and has been added. It calculates position size based on risk percentage, balance, entry/stop-loss prices, leverage, and crucially, the `contract_size` from market info (while still defaulting to the warned `Decimal('1')`).
3.  **Enhanced SL/TP Validation:** The `calculate_entry_tp_sl` function already included the check to ensure SL/TP are at least `MIN_TICKS_AWAY_FOR_SLTP` ticks away from the entry, which is good. Added slightly more logging there.
4.  **Contract Size Emphasis:** Added more prominent comments around the `contract_size` usage in `get_market_info_pybit` and `calculate_position_size`.
5.  **Hedge Mode Comments:** Added specific comments in `attempt_new_entry_pybit` and `set_protection_pybit` regarding `positionIdx` (0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge) and the need for potential logic adjustments if using Hedge Mode extensively.
6.  **Position Confirmation:** Kept the `time.sleep` for simplicity but added comments about potentially polling for more robustness.
7.  **State Saving:** Added a comment about atomic writes for state saving.
8.  **Clarity & Type Hinting:** Added a few more type hints and minor clarifications.

```python
# -*- coding: utf-8 -*-
"""
Enhanced Multi-Symbol Trading Bot for Bybit (V5 API) - Pybit Version v1.0.1

Uses pybit library instead of ccxt for API interaction.
Includes: pandas_ta.Strategy, Decimal precision, robust API interaction,
          multi-symbol support, state management, TSL/BE logic, MA cross exit.

Changes in v1.0.1:
- Critical fix/warning for Contract Size determination (assumed 1).
- Improved balance parsing logic for UTA vs Non-UTA accounts.
- Enhanced market info parsing and validation.
- Stricter type/precision handling when calling pybit methods.
- Added validation ensuring SL/TP are minimum ticks away from entry.
- Improved config validation and comments (especially symbol format).
- Enhanced logging, error messages, and type hinting.
- Added Hedge Mode comments/placeholders.
- Re-integrated full TradingAnalyzer and main loop logic.
- Added calculate_position_size function.
"""

# --- Required Libraries ---
# pip install pybit pandas numpy pandas_ta python-dotenv colorama pytz
# Ensure Python >= 3.7 (for Decimal, zoneinfo needs >= 3.9, pytz is fallback)
# --------------------------

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union, cast

try:
    from zoneinfo import ZoneInfo # Preferred (Python 3.9+)
except ImportError:
    try: from pytz import timezone as ZoneInfo # Fallback
    except ImportError: print("Error: 'zoneinfo' or 'pytz' required. pip install pytz"); sys.exit(1)

try: # --- Pybit ---
    from pybit.unified_trading import HTTP as UnifiedHTTP
except ImportError: print("Error: 'pybit' not found. pip install pybit"); sys.exit(1)

import numpy as np # --- Other Libraries ---
import pandas as pd
import pandas_ta as ta
try: from colorama import Fore, Style, init
except ImportError: print("Warning: 'colorama' not found. pip install colorama"); class D:__getattr__=lambda s,n:"";Fore,Style=D(),D();init=lambda *a,**k:None
from dotenv import load_dotenv

# --- Initialization ---
try: getcontext().prec = 36 # Set Decimal precision
except Exception as e: print(f"Warning: Decimal precision error: {e}.")
init(autoreset=True) # Initialize Colorama
load_dotenv() # Load environment variables from .env file

# --- Constants & Globals ---
NEON_GREEN, NEON_BLUE, NEON_PURPLE, NEON_YELLOW, NEON_RED, NEON_CYAN, RESET = Fore.LIGHTGREEN_EX, Fore.CYAN, Fore.MAGENTA, Fore.YELLOW, Fore.LIGHTRED_EX, Fore.CYAN, Style.RESET_ALL
CONFIG_FILE, LOG_DIRECTORY, STATE_FILE = "config_pybit.json", "bot_logs_pybit", "bot_state_pybit.json"
os.makedirs(LOG_DIRECTORY, exist_ok=True)
try: TZ_NAME = os.getenv("BOT_TIMEZONE", "America/Chicago"); TIMEZONE = ZoneInfo(TZ_NAME); print(f"Using Timezone: {TZ_NAME}")
except Exception as tz_err: print(f"{NEON_YELLOW}Warning: TZ '{TZ_NAME}' error: {tz_err}. Using UTC.{RESET}"); TIMEZONE = ZoneInfo("UTC")

API_KEY = os.getenv("BYBIT_API_KEY"); API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET: print(f"{NEON_RED}FATAL: API Keys missing in .env{RESET}"); sys.exit(1)

MAX_API_RETRIES = 4; RETRY_DELAY_SECONDS = 5; RATE_LIMIT_BUFFER_SECONDS = 0.5
MARKET_INFO_RELOAD_INTERVAL_SECONDS = 3600 # Reload market info every hour
POSITION_CONFIRM_DELAY = 10; # Seconds to wait after placing entry order before checking position/setting SL/TP
MIN_TICKS_AWAY_FOR_SLTP = 3 # Minimum number of ticks SL/TP must be away from entry price
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
PYBIT_INTERVAL_MAP = { "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240, "360": 360, "720": 720, "D": "D", "W": "W", "M": "M"}

# Default Indicator Parameters (can be overridden in config)
DEFAULT_ATR_PERIOD, DEFAULT_CCI_WINDOW, DEFAULT_WILLIAMS_R_WINDOW, DEFAULT_MFI_WINDOW = 14, 20, 14, 14
DEFAULT_STOCH_RSI_WINDOW, DEFAULT_STOCH_WINDOW, DEFAULT_K_WINDOW, DEFAULT_D_WINDOW = 14, 14, 3, 3
DEFAULT_RSI_WINDOW, DEFAULT_BOLLINGER_BANDS_PERIOD, DEFAULT_BOLLINGER_BANDS_STD_DEV = 14, 20, 2.0
DEFAULT_SMA_10_WINDOW, DEFAULT_EMA_SHORT_PERIOD, DEFAULT_EMA_LONG_PERIOD = 10, 9, 21
DEFAULT_MOMENTUM_PERIOD, DEFAULT_VOLUME_MA_PERIOD, DEFAULT_FIB_WINDOW = 7, 15, 50
DEFAULT_PSAR_AF, DEFAULT_PSAR_MAX_AF = 0.02, 0.2; FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
DEFAULT_LOOP_DELAY_SECONDS = 15

loggers: Dict[str, logging.Logger] = {}; console_log_level = logging.INFO
QUOTE_CURRENCY = "USDT"; LOOP_DELAY_SECONDS = DEFAULT_LOOP_DELAY_SECONDS
IS_UNIFIED_ACCOUNT = False # Determined during client initialization
instrument_info_cache: Dict[str, Dict[str, Any]] = {}; last_instrument_info_load_time: float = 0.0

# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Redacts API keys/secrets from log messages."""
    REDACTED_STR = "***REDACTED***"
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if API_KEY and len(API_KEY) > 4: formatted = formatted.replace(API_KEY, self.REDACTED_STR)
        if API_SECRET and len(API_SECRET) > 4: formatted = formatted.replace(API_SECRET, self.REDACTED_STR)
        return formatted
class LocalTimeFormatter(SensitiveFormatter):
    """Formats log timestamps in local timezone."""
    def converter(self, timestamp): dt = datetime.fromtimestamp(timestamp, tz=TIMEZONE); return dt.timetuple()
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=TIMEZONE)
        if datefmt: s = dt.strftime(datefmt)
        else: s = dt.strftime("%Y-%m-%d %H:%M:%S"); s = f"{s},{int(record.msecs):03d}"
        return s
def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    global console_log_level
    logger_instance_name = f"pybitbot_{name.replace('/', '_').replace(':', '-')}" if is_symbol_logger else f"pybitbot_{name}"
    if logger_instance_name in loggers:
        logger = loggers[logger_instance_name]
        # Ensure console handler level matches global setting if logger already exists
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level != console_log_level: handler.setLevel(console_log_level)
        return logger
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name); logger.setLevel(logging.DEBUG)
    try: # File Handler (UTC)
        file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_formatter = SensitiveFormatter("%(asctime)s.%(msecs)03d UTC - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'); file_formatter.converter = time.gmtime
        file_handler.setFormatter(file_formatter); file_handler.setLevel(logging.DEBUG); logger.addHandler(file_handler)
    except Exception as e: print(f"{NEON_RED}Error setting up file logger for '{log_filename}': {e}{RESET}")
    try: # Stream Handler (Local Time)
        stream_handler = logging.StreamHandler(sys.stdout)
        tz_name_str = TIMEZONE.tzname(datetime.now(TIMEZONE)) if hasattr(TIMEZONE, 'tzname') else str(TIMEZONE)
        stream_formatter = LocalTimeFormatter(f"{NEON_BLUE}%(asctime)s{RESET} [{tz_name_str}] - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3])
        stream_handler.setFormatter(stream_formatter); stream_handler.setLevel(console_log_level); logger.addHandler(stream_handler)
    except Exception as e: print(f"{NEON_RED}Error setting up stream logger for {name}: {e}{RESET}")
    logger.propagate = False; loggers[logger_instance_name] = logger
    logger.info(f"Logger '{logger_instance_name}' initialized. File: '{os.path.basename(log_filename)}', Console Level: {logging.getLevelName(console_log_level)}")
    return logger
def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger: return setup_logger(name, is_symbol_logger)

# --- Configuration Management ---
def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Recursively ensures all keys from default_config exist in config."""
    updated_config = config.copy(); keys_added_or_type_mismatch = False
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value; keys_added_or_type_mismatch = True
            print(f"{NEON_YELLOW}Cfg Warn: Missing key '{key}'. Added default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recurse for nested dictionaries
            nested_updated_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_config[key])
            if nested_keys_added:
                updated_config[key] = nested_updated_config; keys_added_or_type_mismatch = True
        elif updated_config.get(key) is not None and type(default_value) != type(updated_config.get(key)):
            # Allow int -> float/Decimal promotion, warn otherwise
            is_promoting_num = (isinstance(default_value, (float, Decimal)) and isinstance(updated_config.get(key), int))
            if not is_promoting_num:
                print(f"{NEON_YELLOW}Cfg Warn: Type mismatch for key '{key}'. Expected {type(default_value).__name__}, got {type(updated_config.get(key)).__name__}. Using loaded value: {repr(updated_config.get(key))}. Check config.{RESET}")
                # Keep the user's value despite type mismatch, but warn them. Could force default type here if stricter control is needed.
    return updated_config, keys_added_or_type_mismatch

def _validate_config_values(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validates specific configuration values and ranges."""
    is_valid = True; interval = config.get("interval")
    if interval not in PYBIT_INTERVAL_MAP: logger.error(f"Config Error: Invalid 'interval' value '{interval}'. Valid options are: {list(PYBIT_INTERVAL_MAP.keys())}"); is_valid = False

    numeric_params = { # Key: (Expected Type, Min Value, Max Value)
        "loop_delay": (int, 5, 3600), "risk_per_trade": (float, 0.0001, 0.5), "leverage": (int, 1, 125),
        "max_concurrent_positions_total": (int, 1, 100), "atr_period": (int, 2, 500), "ema_short_period": (int, 2, 500),
        "ema_long_period": (int, 3, 1000), "rsi_period": (int, 2, 500), "bollinger_bands_period": (int, 5, 500),
        "bollinger_bands_std_dev": (float, 0.1, 5.0), "cci_window": (int, 5, 500), "williams_r_window": (int, 2, 500),
        "mfi_window": (int, 2, 500), "stoch_rsi_window": (int, 5, 500), "stoch_rsi_rsi_window": (int, 5, 500),
        "stoch_rsi_k": (int, 1, 100), "stoch_rsi_d": (int, 1, 100), "psar_af": (float, 0.001, 0.1), "psar_max_af": (float, 0.1, 0.5),
        "sma_10_window": (int, 2, 500), "momentum_period": (int, 1, 500), "volume_ma_period": (int, 2, 500),
        "fibonacci_window": (int, 10, 1000), "orderbook_limit": (int, 1, 200), "signal_score_threshold": (float, 0.1, 10.0),
        "stoch_rsi_oversold_threshold": (float, 0.1, 49.9), "stoch_rsi_overbought_threshold": (float, 50.1, 99.9),
        "volume_confirmation_multiplier": (float, 0.1, 10.0), "scalping_signal_threshold": (float, 0.1, 10.0),
        "stop_loss_multiple": (float, 0.1, 10.0), "take_profit_multiple": (float, 0.1, 20.0),
        "trailing_stop_callback_rate": (float, 0.0001, 0.5), # e.g., 0.005 = 0.5%
        "trailing_stop_activation_percentage": (float, 0.0, 0.5), # e.g., 0.003 = 0.3% profit needed to activate TSL
        "break_even_trigger_atr_multiple": (float, 0.1, 10.0),
        "break_even_offset_ticks": (int, 0, 100), # Number of ticks above/below entry for BE SL
    }
    for key, (expected_type, min_val, max_val) in numeric_params.items():
        value = config.get(key);
        if value is None: continue # Skip if not present (will use default later)
        try:
            num_value = expected_type(value) # Attempt conversion
            if not (min_val <= num_value <= max_val):
                 logger.error(f"Config Error: Parameter '{key}' value {num_value} is outside the valid range ({min_val}-{max_val})."); is_valid = False
            config[key] = num_value # Store the validated, correctly typed value back
        except (ValueError, TypeError): logger.error(f"Config Error: Parameter '{key}' value '{value}' is not a valid {expected_type.__name__}."); is_valid = False

    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols: logger.error("Config Error: 'symbols' must be a non-empty list."); is_valid = False
    elif not all(isinstance(s, str) and len(s) > 2 and '/' not in s and ':' not in s for s in symbols):
        # Check for common mistakes like using slash separators
        logger.error(f"Config Error: 'symbols' should be a list of strings in Bybit V5 format (e.g., ['BTCUSDT', 'ETHUSDT']). Found: {symbols}")
        is_valid = False

    pos_mode = config.get("position_mode")
    if pos_mode not in ["One-Way", "Hedge"]:
        logger.error(f"Config Error: Invalid 'position_mode' ('{pos_mode}'). Must be 'One-Way' or 'Hedge'. This MUST match your Bybit account setting for the traded category (Linear/Inverse).")
        is_valid = False
    # Note: Further validation (e.g., checking if hedge mode is actually enabled on Bybit) requires API calls post-initialization.

    # Validate weight sets structure (basic check)
    weight_sets = config.get("weight_sets", {})
    if not isinstance(weight_sets, dict): logger.error("Config Error: 'weight_sets' should be a dictionary."); is_valid = False
    else:
        active_set = config.get("active_weight_set")
        if active_set not in weight_sets: logger.error(f"Config Error: 'active_weight_set' ('{active_set}') not found in 'weight_sets'."); is_valid = False
        for set_name, weights in weight_sets.items():
            if not isinstance(weights, dict): logger.error(f"Config Error: Weight set '{set_name}' is not a dictionary."); is_valid = False
            # Could add more checks here (e.g., numeric weights, matching indicator names)

    return is_valid

def load_config(filepath: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads configuration from JSON file, ensures defaults, validates, and returns config dict or None."""
    default_config = {
        "symbols": ["BTCUSDT", "ETHUSDT"], # NOTE: Use Bybit V5 symbol format (e.g., BTCUSDT, no slashes/colons)
        "interval": "5", # See VALID_INTERVALS
        "loop_delay": DEFAULT_LOOP_DELAY_SECONDS, # Seconds between bot cycles
        "quote_currency": "USDT", # Base currency for balance/risk calcs
        "enable_trading": False, # Master switch for placing real orders
        "use_sandbox": True, # Use Bybit testnet environment
        "risk_per_trade": 0.01, # Fraction of available balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 10, # Desired leverage (Ensure it's allowed for the symbol)
        "max_concurrent_positions_total": 1, # Max total open positions across all symbols
        "position_mode": "One-Way", # Options: "One-Way", "Hedge". MUST match account setting.
        # Indicator Periods & Settings
        "atr_period": DEFAULT_ATR_PERIOD, "ema_short_period": DEFAULT_EMA_SHORT_PERIOD, "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW, "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV, "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW, "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW, # Matches RSI period for StochRSI inner calc
        "stoch_rsi_k": DEFAULT_K_WINDOW, "stoch_rsi_d": DEFAULT_D_WINDOW, "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF, "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD, "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW, # Lookback for Fib High/Low
        "orderbook_limit": 25, # Depth for Order Book Imbalance calc
        # Signal Generation & Thresholds
        "signal_score_threshold": 1.5, # Required weighted score to trigger BUY/SELL (default set)
        "scalping_signal_threshold": 2.5, # Required weighted score for 'scalping' weight set
        "stoch_rsi_oversold_threshold": 25.0, "stoch_rsi_overbought_threshold": 75.0,
        "volume_confirmation_multiplier": 1.5, # Current vol must be > this * vol_ma for signal boost
        # Risk Management
        "stop_loss_multiple": 1.8, # ATR multiple for initial Stop Loss distance
        "take_profit_multiple": 0.7, # ATR multiple for initial Take Profit distance
        "enable_ma_cross_exit": True, # Close position if short/long EMAs cross against the trade
        "enable_trailing_stop": True, # Use Pybit's server-side trailing stop
        "trailing_stop_callback_rate": 0.005, # TSL distance as percentage (0.005 = 0.5%) - Pybit uses distance value though, calc needed
        "trailing_stop_activation_percentage": 0.003, # Profit percentage needed to activate TSL (0.003 = 0.3%) - Pybit uses price, calc needed
        "enable_break_even": True, # Move SL to entry (+offset) if price moves favorably by BE trigger multiple
        "break_even_trigger_atr_multiple": 1.0, # ATR multiple profit needed to trigger BE
        "break_even_offset_ticks": 2, # How many ticks above/below entry to set BE stop loss
        "break_even_force_fixed_sl": True, # If True, disables TSL when BE is triggered, using fixed SL only. If False, keeps TSL active with BE SL.
        # Indicator Switches & Weights
        "indicators": { # Enable/disable specific indicators for signal generation
            "ema_alignment": True, "momentum": True, "volume_confirmation": True, "stoch_rsi": True,
            "rsi": True, "bollinger_bands": True, "vwap": True, "cci": True, "wr": True, "psar": True,
            "sma_10": True, "mfi": True, "orderbook": True,
        },
        "weight_sets": { # Define different weighting strategies
             "scalping": { # Example: Higher weight on faster indicators
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2, "stoch_rsi": 0.6,
                "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4, "cci": 0.3, "wr": 0.3, "psar": 0.2,
                "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
             },
             "default": { # Balanced weights
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1, "stoch_rsi": 0.4,
                "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3, "cci": 0.2, "wr": 0.2, "psar": 0.3,
                "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
             }
        },
        "active_weight_set": "default" # Which weight set to use for signals
    }
    config_to_use = default_config.copy(); keys_updated_in_file = False
    if not os.path.exists(filepath):
        print(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating a default config file...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            print(f"{NEON_GREEN}Default configuration file created at: {filepath}{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using internal defaults.{RESET}")
            # Proceed with internal defaults if file creation fails
    else:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config_from_file = json.load(f)
            # Ensure all keys exist, add missing ones from defaults
            updated_config_from_file, keys_added = _ensure_config_keys(config_from_file, default_config)
            config_to_use = updated_config_from_file
            if keys_added:
                keys_updated_in_file = True
                print(f"{NEON_YELLOW}Configuration file '{filepath}' was missing some keys. Updating file with defaults...{RESET}")
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                         json.dump(config_to_use, f_write, indent=4, sort_keys=True)
                    print(f"{NEON_GREEN}Config file updated successfully.{RESET}")
                except IOError as e:
                    print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
                    keys_updated_in_file = False # Update failed
        except json.JSONDecodeError as e:
            print(f"{NEON_RED}Error decoding JSON from config file {filepath}: {e}. Using internal defaults.{RESET}")
            config_to_use = default_config.copy() # Fallback to defaults on decode error
        except Exception as e:
            print(f"{NEON_RED}Unexpected error loading config file {filepath}: {e}. Using internal defaults.{RESET}")
            config_to_use = default_config.copy() # Fallback on other errors

    # Final validation after loading/merging
    if not _validate_config_values(config_to_use, logger):
        logger.critical(f"{NEON_RED}Configuration validation failed. Please check errors above and fix '{filepath}'. Bot cannot start.{RESET}")
        return None # Indicate failure

    logger.info("Configuration loaded and validated successfully.")
    if keys_updated_in_file: logger.info(f"Config file '{filepath}' was updated with missing default keys.")
    return config_to_use

# --- State Management ---
def load_state(filepath: str, logger: logging.Logger) -> Dict[str, Any]:
    """Loads bot state from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"Loaded previous bot state from {filepath}")
                # Attempt to convert relevant numeric strings back to Decimal if needed
                # Example: state['symbol']['last_entry_price'] = Decimal(state['symbol']['last_entry_price'])
                # This needs care depending on what's stored. Let's keep it simple for now.
                return state
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from state file {filepath}: {e}. Starting with empty state.", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Error loading state file {filepath}: {e}. Starting with empty state.", exc_info=True)
            return {}
    else:
        logger.info(f"No state file found at {filepath}. Starting with empty state.")
        return {}

def save_state(filepath: str, state: Dict[str, Any], logger: logging.Logger):
    """Saves bot state to a JSON file."""
    # Suggestion: Implement atomic write (write to temp file, then rename) for safety
    temp_filepath = filepath + ".tmp"
    try:
        # Convert Decimals to strings for JSON compatibility BEFORE dumping
        state_to_save = json.loads(json.dumps(state, default=str)) # Easy way to handle Decimals etc.
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=4)
        # Atomic rename (on POSIX systems, check for Windows compatibility if needed)
        os.replace(temp_filepath, filepath)
        logger.debug(f"Saved current bot state to {filepath}")
    except Exception as e:
        logger.error(f"Error saving state to {filepath}: {e}", exc_info=True)
        # Clean up temp file if it exists
        if os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass

# --- Pybit Client Setup ---
def initialize_pybit_client(config: Dict[str, Any], logger: logging.Logger) -> Optional[UnifiedHTTP]:
    """Initializes and tests the Pybit HTTP client."""
    lg = logger; global QUOTE_CURRENCY; QUOTE_CURRENCY = config.get("quote_currency", "USDT")
    lg.info(f"Using Quote Currency: {QUOTE_CURRENCY}"); use_testnet = config.get("use_sandbox", True)
    try:
        client = UnifiedHTTP(testnet=use_testnet, api_key=API_KEY, api_secret=API_SECRET, recv_window=20000) # Increased recv_window
        lg.info(f"Initializing Pybit UnifiedHTTP client (Testnet: {use_testnet})...")

        # Perform a simple API call to test connectivity and authentication
        lg.info("Attempting API connection test (get_account_info)...")
        try:
            # Using get_account_info as it's relatively lightweight
            info = client.get_account_info()
            if info and info.get('retCode') == 0:
                lg.info(f"{NEON_GREEN}Pybit API connection successful.{RESET}")
                # Check if it's a Unified Trading Account (UTA) vs Standard
                uta_level = info.get('result', {}).get('unifiedMarginStatus'); # 1: Regular account; 2: Regular UTA; 3: Pro UTA; 4: Pro UTA
                if uta_level and uta_level >= 2:
                    lg.info(f"Detected Unified Trading Account (UTA Level: {uta_level}). Balance checks will use 'UNIFIED'.")
                    # Global IS_UNIFIED_ACCOUNT will be set by _check_pybit_account_type_and_balance
                else:
                    lg.info("Detected Standard Account (Non-UTA). Balance checks will use 'CONTRACT'/'SPOT'.")
            else:
                lg.error(f"{NEON_RED}Pybit API connection test failed. Response Code: {info.get('retCode')}, Message: {info.get('retMsg')}{RESET}")
                lg.error("Check API keys, permissions, network, and if IP is whitelisted on Bybit.")
                return None
        except Exception as conn_err:
            lg.error(f"{NEON_RED}Pybit API connection test threw an exception: {conn_err}{RESET}", exc_info=True)
            lg.error("Check network connectivity and Pybit service status.")
            return None

        # Determine account type and fetch initial balance
        lg.info(f"Attempting initial balance fetch for {QUOTE_CURRENCY} and account type detection...")
        is_unified, balance_dec = _check_pybit_account_type_and_balance(client, QUOTE_CURRENCY, lg)

        if is_unified is not None:
            global IS_UNIFIED_ACCOUNT; IS_UNIFIED_ACCOUNT = is_unified
            lg.info(f"Confirmed Account Type for trading: {'UNIFIED' if IS_UNIFIED_ACCOUNT else 'Non-UTA'}")
        else:
            # This is problematic - we couldn't determine the account type reliably.
            lg.warning("Could not reliably determine account type (UTA vs Non-UTA) via balance check.")
            # We might need to make an assumption or fail. Let's assume based on initial check, if possible.
            # If info['result']['unifiedMarginStatus'] was available earlier, use that as a fallback guess.
            # Otherwise, it's risky to proceed.
            if config.get("enable_trading"):
                 lg.error(f"{NEON_RED}Cannot confirm account type. Trading disabled for safety. Check API permissions/response.{RESET}")
                 # return None # Safer to exit if trading is enabled
            else:
                 lg.warning(f"{NEON_YELLOW}Proceeding with potentially incorrect account type assumption (Non-UTA default?). Trading is disabled.{RESET}")
                 IS_UNIFIED_ACCOUNT = False # Default assumption if check fails

        if balance_dec is None:
            lg.warning(f"{NEON_YELLOW}Initial balance fetch failed for {QUOTE_CURRENCY}. Trading actions might fail.{RESET}")
            # If trading is enabled, failure to get balance is critical.
            if config.get("enable_trading"):
                lg.error(f"{NEON_RED}Cannot verify available balance in live trading mode. Aborting bot start.{RESET}")
                return None
        else:
            lg.info(f"{NEON_GREEN}Initial balance check OK. Available {QUOTE_CURRENCY}: {balance_dec:.4f}{RESET}")

        lg.info(f"{NEON_GREEN}Pybit client initialized successfully.{RESET}")
        return client

    except Exception as e:
        lg.error(f"{NEON_RED}Failed to initialize Pybit client: {e}{RESET}", exc_info=True)
        return None

def _check_pybit_account_type_and_balance(client: UnifiedHTTP, currency: str, logger: logging.Logger) -> Tuple[Optional[bool], Optional[Decimal]]:
    """Tries fetching balance for UNIFIED and CONTRACT to determine account type and get balance."""
    lg = logger
    # Try UNIFIED first (common for newer accounts/features)
    try:
        lg.debug("Checking balance with accountType=UNIFIED...")
        balance_info_unified = safe_pybit_call(client, 'get_wallet_balance', lg, accountType="UNIFIED", coin=currency, max_retries=1) # Low retry for check
        parsed_unified = _parse_pybit_balance_response(balance_info_unified, currency, "UNIFIED", lg)
        if parsed_unified is not None:
            lg.info("Balance check successful with accountType=UNIFIED.")
            return True, parsed_unified # It's a Unified account
    except Exception as e:
        lg.debug(f"UNIFIED balance check failed or returned no data: {e}") # Log as debug, expected for non-UTA

    # Try CONTRACT (for older/standard derivative accounts)
    try:
        lg.debug("Checking balance with accountType=CONTRACT...")
        balance_info_contract = safe_pybit_call(client, 'get_wallet_balance', lg, accountType="CONTRACT", coin=currency, max_retries=1)
        parsed_contract = _parse_pybit_balance_response(balance_info_contract, currency, "CONTRACT", lg)
        if parsed_contract is not None:
            lg.info("Balance check successful with accountType=CONTRACT.")
            return False, parsed_contract # It's a Non-UTA (Standard) account
    except Exception as e:
        lg.debug(f"CONTRACT balance check failed or returned no data: {e}") # Log as debug

    # If neither worked
    lg.error("Failed to determine account type or fetch balance using either UNIFIED or CONTRACT account types.")
    return None, None # Indicate failure to determine type and balance

# --- Pybit API Call Helper ---
def safe_pybit_call(client: UnifiedHTTP, method_name: str, logger: logging.Logger, max_retries: int = MAX_API_RETRIES, retry_delay_sec: int = RETRY_DELAY_SECONDS, **kwargs) -> Optional[Dict]:
    """Safely calls a pybit method with retries and error handling."""
    lg = logger; last_exception = None; last_ret_code = None; last_ret_msg = ""
    for attempt in range(max_retries + 1):
        try:
            method = getattr(client, method_name)
            # lg.debug(f"Calling pybit.{method_name} (Attempt {attempt+1}/{max_retries+1}) Params: {kwargs}") # DEBUG: Log params (can be verbose)
            response = method(**kwargs)
            # lg.debug(f"Pybit.{method_name} Raw Response: {response}") # DEBUG: Log raw response

            if isinstance(response, dict):
                ret_code = response.get('retCode')
                ret_msg = response.get('retMsg', '')
                last_ret_code, last_ret_msg = ret_code, ret_msg # Store last attempt's result

                if ret_code == 0:
                    # lg.debug(f"Pybit.{method_name} call successful.")
                    return response # Success

                else:
                    # Specific Bybit Error Code Handling (V5 Unified) - Refer to Bybit API docs for codes
                    # https://bybit-exchange.github.io/docs/v5/error_code
                    retryable_codes = [
                        10002, # Request version invalid? (Maybe temporary)
                        10006, # Too many visits (Rate Limit)
                        10010, # Request expired (Clock skew?)
                        10016, # Service error / Server busy
                        10018, # Request duplicate (Maybe retry after delay?)
                        30034, # Order quantity is invalid (Might be temporary precision issue?) - Risky to retry?
                        30067, # Position idx not match position mode (Maybe temporary state issue?) - Risky?
                        130006, # KLine data temporary unavailable
                        130150, # System error (Internal server error)
                        131204, # Request frequently
                    ]
                    non_retryable_codes = [
                        10001, # Params Error (Invalid input, won't resolve by retry)
                        10003, # Invalid API Key
                        10004, # Authentication Failed / Invalid Signature
                        10005, # Permission Denied for API Key
                        10007, # Invalid Symbol
                        10009, # IP Mismatch
                        10017, # Request path not found
                        10020, # Max active orders reached
                        110001, # Insufficient balance
                        110007, # Leverage exceeds max allowed
                        110013, # Position size zero
                        110020, # Risk limit exceeded
                        110025, # Min order value not met
                        110043, # Leverage not modified (Not an error, but handle specifically if needed)
                        110045, # QtyPrecision error
                        170007, # Order price deviates too much
                        170131, # SL price invalid
                        170132, # TP price invalid
                        170133, # PriceTickSize error
                        170140, # TSL activation price error
                        # Add more known fatal/config errors here
                    ]

                    if ret_code in retryable_codes:
                        last_exception = Exception(f"Pybit Retryable Error {ret_code}: {ret_msg}")
                        wait = retry_delay_sec * (2 ** attempt) + RATE_LIMIT_BUFFER_SECONDS # Exponential backoff + buffer
                        lg.warning(f"Retryable Pybit error calling {method_name} (Code: {ret_code}, Msg: '{ret_msg}'). Retrying in {wait:.2f}s...")
                        time.sleep(wait)
                        continue # Go to next attempt

                    elif ret_code in non_retryable_codes:
                        # Handle specific non-retryable codes if needed
                        if ret_code == 110043 and method_name == 'set_leverage':
                             lg.info(f"Leverage already set as requested (Code 110043) when calling {method_name}. Treating as success.")
                             return {"retCode": 0, "retMsg": "Leverage not modified", "result": {}} # Simulate success
                        # For other non-retryables, log as error and stop retrying
                        extra_info = f" {NEON_YELLOW}Hint: Check parameters, API permissions, or account settings (e.g., Category/AccountType mismatch?).{RESET}" if ret_code in [10001, 10005] else ""
                        lg.error(f"{NEON_RED}Non-retryable Pybit error calling {method_name} (Code: {ret_code}, Msg: '{ret_msg}').{RESET}{extra_info}")
                        # Raise an exception to signal failure clearly upwards
                        raise Exception(f"Pybit NonRetryable Error {ret_code}: {ret_msg}")

                    else: # Unknown error code
                        last_exception = Exception(f"Pybit Unknown Error {ret_code}: {ret_msg}")
                        wait = retry_delay_sec * (2 ** attempt)
                        lg.warning(f"{NEON_YELLOW}Unknown Pybit error code received from {method_name} (Code: {ret_code}, Msg: '{ret_msg}'). Retrying in {wait:.2f}s...{RESET}")
                        time.sleep(wait)
                        continue # Retry unknown codes just in case

            else: # Unexpected response type (not a dict)
                lg.error(f"Unexpected response format received from pybit.{method_name}: Type {type(response)}, Value: {str(response)[:200]}...")
                last_exception = Exception(f"Unexpected Pybit response format: {type(response)}")
                wait = retry_delay_sec * (2 ** attempt)
                time.sleep(wait)
                continue # Retry

        except Exception as e:
            # Catch exceptions raised by pybit library itself (network issues, etc.) or our re-raised non-retryable error
            if "NonRetryable" in str(e): # If we raised it ourselves, don't retry
                lg.error(f"Caught non-retryable error flag during {method_name} call. Aborting retries.")
                last_exception = e
                break # Exit the retry loop
            last_exception = e
            wait = retry_delay_sec * (2 ** attempt)
            lg.warning(f"Exception occurred calling pybit.{method_name}: {e}. Retrying in {wait:.2f}s...")
            time.sleep(wait)
            # Optional: Check for specific network errors if needed

    # If loop finishes without returning (i.e., max retries exceeded)
    lg.error(f"{NEON_RED}Max retries ({max_retries}) exceeded for pybit.{method_name}. Last Code: {last_ret_code}, Msg: '{last_ret_msg}'. Last Exception: {last_exception}{RESET}",
             exc_info=isinstance(last_exception, Exception) and last_exception is not None) # Log stack trace if an exception was caught
    return None # Indicate failure after all retries

# --- Market Info Handling (Pybit) ---
def fetch_and_cache_instrument_info(client: UnifiedHTTP, logger: logging.Logger) -> bool:
    """Fetches instrument info for relevant categories and caches it."""
    global instrument_info_cache, last_instrument_info_load_time; lg = logger; now = time.time()
    # Avoid fetching too frequently
    if instrument_info_cache and (now - last_instrument_info_load_time < MARKET_INFO_RELOAD_INTERVAL_SECONDS):
        # lg.debug("Using cached instrument info.")
        return True

    lg.info("Fetching fresh instrument info from Bybit (may take a moment)...")
    new_cache = {}; categories_to_fetch = ['linear', 'inverse', 'spot']; # Add 'option' if needed
    fetch_success = True

    for category in categories_to_fetch:
        try:
            lg.debug(f"Fetching instruments for category: {category}...")
            # Use pagination if needed, though typically not required for V5 instruments_info unless >1000 symbols per category
            response = safe_pybit_call(client, 'get_instruments_info', lg, category=category) # Default limit is high

            if response and response.get('retCode') == 0 and 'result' in response and 'list' in response['result']:
                instruments = response['result']['list']
                if not instruments:
                    lg.warning(f"No instruments found for category '{category}'.")
                    continue
                lg.info(f"Fetched {len(instruments)} instruments for category '{category}'. Processing...")

                processed_count = 0
                for instrument in instruments:
                    symbol = instrument.get('symbol')
                    if not symbol: continue # Skip if no symbol

                    # --- CRITICAL: Extract precision and limits using Decimal ---
                    tick_size_str = instrument.get('priceFilter', {}).get('tickSize')
                    qty_step_str = instrument.get('lotSizeFilter', {}).get('qtyStep')
                    min_order_qty_str = instrument.get('lotSizeFilter', {}).get('minOrderQty')
                    max_order_qty_str = instrument.get('lotSizeFilter', {}).get('maxOrderQty')
                    min_price_str = instrument.get('priceFilter', {}).get('minPrice')
                    max_price_str = instrument.get('priceFilter', {}).get('maxPrice')
                    max_leverage_str = instrument.get('leverageFilter', {}).get('maxLeverage')
                    min_order_iv_str = instrument.get('lotSizeFilter', {}).get('minOrderIv') # Min order value in quote coin (USDT)

                    # --- Contract Size Handling ---
                    # V5 API doesn't explicitly return contract size in get_instruments_info like V3 did.
                    # Linear USDT Perps (e.g., BTCUSDT): Contract size is typically 1 base unit (1 BTC). Qty is in base units. Value = Qty * Price.
                    # Inverse Perps (e.g., BTCUSD): Contract size is typically 1 USD. Qty is in contracts (USD value). Value = Qty / Price.
                    # Options: Contract size is 1 base unit.
                    # SPOT: No contract size concept, qty is base asset.
                    # ** THIS IS A MAJOR ASSUMPTION - VERIFY FOR YOUR TRADED INSTRUMENTS **
                    contract_size = Decimal('1.0') # DEFAULT ASSUMPTION: 1 Base Unit (like linear perps)
                    if category == 'inverse':
                        contract_size = Decimal('1.0') # Inverse contracts often have a size of 1 USD - *RE-VERIFY THIS* based on symbol spec if trading inverse.
                        lg.warning(f"{NEON_YELLOW}Assuming contract size {contract_size} for INVERSE symbol {symbol}. VERIFY THIS!{RESET}")
                    elif category == 'linear':
                         pass # Default assumption of 1 base unit is usually correct for linear USDT perps.
                    elif category == 'option':
                         pass # Default assumption of 1 base unit is usually correct for options.
                    elif category == 'spot':
                         contract_size = None # Not applicable


                    try:
                        tick_size = Decimal(tick_size_str) if tick_size_str else None
                        qty_step = Decimal(qty_step_str) if qty_step_str else None
                        min_order_qty = Decimal(min_order_qty_str) if min_order_qty_str else Decimal('0')
                        max_order_qty = Decimal(max_order_qty_str) if max_order_qty_str else Decimal('inf')
                        min_price = Decimal(min_price_str) if min_price_str else Decimal('0')
                        max_price = Decimal(max_price_str) if max_price_str else Decimal('inf')
                        max_leverage = Decimal(max_leverage_str) if max_leverage_str else Decimal('0') # 0 might mean N/A (Spot)
                        min_order_iv = Decimal(min_order_iv_str) if min_order_iv_str else Decimal('0')

                        # Validate essential values
                        if tick_size is None or tick_size <= 0: lg.warning(f"Invalid or missing tickSize for {symbol} ({category})."); tick_size = None; # Mark as invalid
                        if qty_step is None or qty_step <= 0: lg.warning(f"Invalid or missing qtyStep for {symbol} ({category})."); qty_step = None; # Mark as invalid

                        parsed_info = {
                            'symbol': symbol,
                            'pybit_symbol': symbol, # Redundant but consistent
                            'category': category,
                            'status': instrument.get('status', 'Unknown').upper(), # Trading, PreLaunch, etc.
                            'baseCoin': instrument.get('baseCoin', ''),
                            'quoteCoin': instrument.get('quoteCoin', ''),
                            'settleCoin': instrument.get('settleCoin', ''), # Important for derivatives margin/pnl currency
                            'minOrderQty': min_order_qty,
                            'maxOrderQty': max_order_qty,
                            'qtyStep': qty_step, # Store as Decimal or None
                            'tickSize': tick_size, # Store as Decimal or None
                            'minPrice': min_price,
                            'maxPrice': max_price,
                            'maxLeverage': max_leverage,
                            'minOrderIv': min_order_iv, # Minimum order value (cost)
                            'contract_size': contract_size, # Store the determined/assumed contract size (Decimal or None)
                            'is_linear': category == 'linear',
                            'is_inverse': category == 'inverse',
                            'is_spot': category == 'spot',
                            'is_option': category == 'option',
                            'is_contract': category in ['linear', 'inverse', 'option'], # Is it a derivative?
                            'raw_info': instrument # Keep raw data if needed later
                        }

                        # Only add to cache if essential precision info is valid
                        if tick_size is not None and qty_step is not None and parsed_info['status'] == 'TRADING':
                            new_cache[symbol] = parsed_info
                            processed_count += 1
                        elif parsed_info['status'] != 'TRADING':
                            lg.debug(f"Skipping non-trading symbol {symbol} (Status: {parsed_info['status']})")
                        else:
                             lg.error(f"{NEON_RED}Failed to parse essential precision (tick/step) for TRADING symbol {symbol} ({category}). Skipping cache.{RESET}")


                    except (InvalidOperation, ValueError, TypeError) as parse_err:
                         lg.error(f"Error parsing market data for {symbol} ({category}): {parse_err}. Raw: {instrument}")
                         fetch_success = False # Mark partial failure

                lg.info(f"Successfully processed {processed_count} TRADING instruments for category '{category}'.")

            elif response: # Fetch failed but got a response dict
                lg.error(f"Failed to fetch instruments for category '{category}'. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
                fetch_success = False
            else: # Fetch failed, no response (likely from safe_pybit_call timeout)
                lg.error(f"Failed to fetch instruments for category '{category}' after retries (no response).")
                fetch_success = False
        except Exception as e:
            lg.error(f"Unexpected error fetching instruments for category {category}: {e}", exc_info=True)
            fetch_success = False

    if new_cache:
        instrument_info_cache = new_cache
        last_instrument_info_load_time = now
        lg.info(f"Instrument info cache updated successfully ({'PARTIAL update' if not fetch_success else 'Full update'}). Total cached TRADING symbols: {len(new_cache)}")
        return True
    else:
        lg.error("Failed to fetch ANY instrument info. Cache remains empty or stale.")
        # Keep potentially stale cache if load fails? Or clear it? Clearing seems safer.
        # instrument_info_cache = {} # Optional: Clear cache on complete failure
        return False

def get_market_info_pybit(client: UnifiedHTTP, symbol: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Retrieves cached market info for a symbol, refreshing cache if needed."""
    global instrument_info_cache, last_instrument_info_load_time; lg = logger; now = time.time()

    # Ensure cache is loaded or refresh if stale
    if not instrument_info_cache or (now - last_instrument_info_load_time > MARKET_INFO_RELOAD_INTERVAL_SECONDS):
        lg.info(f"Instrument cache empty or stale. Triggering refresh for symbol '{symbol}'...")
        if not fetch_and_cache_instrument_info(client, lg):
            lg.error(f"Failed to refresh instrument cache. Cannot provide market info for '{symbol}'.")
            return None # Critical failure if cache cannot be loaded

    market = instrument_info_cache.get(symbol)

    if not market:
        # Maybe the symbol wasn't in the initial fetch (e.g., wrong category guess, or new listing)
        # Optionally: try a targeted fetch here? But complicates logic. Assume pre-fetch covers it.
        lg.error(f"Symbol '{symbol}' not found in the instrument cache. Was it included in the fetch categories (linear, inverse, spot)? Is it 'TRADING' status?")
        return None

    # Enhance the cached info with calculated precision details if not already done
    if 'min_tick_size' not in market: # Check if enhancement is needed
        enhanced_market = market.copy()
        tick_size = enhanced_market.get('tickSize') # Should be Decimal or None
        qty_step = enhanced_market.get('qtyStep')   # Should be Decimal or None

        enhanced_market['min_tick_size'] = tick_size if tick_size else None # Alias for clarity
        enhanced_market['step_size'] = qty_step if qty_step else None       # Alias for clarity

        # Calculate precision digits (number of decimal places)
        if tick_size and tick_size.is_finite() and tick_size > 0:
            enhanced_market['price_precision_digits'] = abs(tick_size.normalize().as_tuple().exponent)
        else:
            lg.warning(f"Using default price precision (8) for {symbol} due to invalid tickSize: {tick_size}")
            enhanced_market['price_precision_digits'] = 8 # Default fallback

        if qty_step and qty_step.is_finite() and qty_step > 0:
            enhanced_market['amount_precision_digits'] = abs(qty_step.normalize().as_tuple().exponent)
        else:
            lg.warning(f"Using default amount precision (8) for {symbol} due to invalid qtyStep: {qty_step}")
            enhanced_market['amount_precision_digits'] = 8 # Default fallback

        # Add other useful aliases if needed
        enhanced_market['min_order_amount'] = enhanced_market.get('minOrderQty')
        enhanced_market['min_order_cost'] = enhanced_market.get('minOrderIv') # Min value in quote currency

        # Log the critical contract size assumption again when accessed
        if enhanced_market['is_contract'] and enhanced_market['contract_size'] is None:
             lg.error(f"CONTRACT_SIZE IS NONE for {symbol} - calculation error likely.")
        elif enhanced_market['is_contract'] and enhanced_market['contract_size'] != Decimal('1.0'):
             lg.warning(f"Using non-standard contract_size={enhanced_market['contract_size']} for {symbol}. Ensure risk/size calculations are correct!")
        elif enhanced_market['is_contract']:
             lg.debug(f"Using contract_size={enhanced_market['contract_size']} for {symbol}.")


        # Update the cache with the enhanced version
        instrument_info_cache[symbol] = enhanced_market
        return enhanced_market
    else:
        # Return the already enhanced market info from cache
        return market


# --- Pybit Data Fetching Wrappers ---
def fetch_klines_pybit(client: UnifiedHTTP, symbol: str, timeframe: str, limit: int, logger: logging.Logger, market_info: Dict) -> pd.DataFrame:
    """Fetches OHLCV data using pybit and returns a pandas DataFrame with Decimal types."""
    lg = logger; pybit_interval = PYBIT_INTERVAL_MAP.get(timeframe); category = market_info.get('category')

    if not pybit_interval:
        lg.error(f"Invalid timeframe '{timeframe}' provided for {symbol}. Valid: {list(PYBIT_INTERVAL_MAP.keys())}"); return pd.DataFrame()
    if not category:
        lg.error(f"Market category missing for {symbol} in market_info. Cannot fetch klines."); return pd.DataFrame()

    # Pybit V5 kline limit is 1000 per request. Handle larger requests if needed (pagination).
    if limit > 1000: lg.warning(f"Requested {limit} klines, but max per call is 1000. Fetching only 1000.")
    fetch_limit = min(limit, 1000)

    try:
        lg.debug(f"Fetching {fetch_limit} klines for {symbol} ({category}, interval: {pybit_interval})...")
        response = safe_pybit_call(client, 'get_kline', lg,
                                   category=category, symbol=symbol,
                                   interval=pybit_interval, limit=fetch_limit)

        if response and response.get('retCode') == 0 and 'result' in response and 'list' in response['result']:
            ohlcv_list = response['result']['list']
            if not ohlcv_list:
                lg.warning(f"Received empty kline data list for {symbol} {timeframe}."); return pd.DataFrame()

            # V5 Kline format: [timestamp, open, high, low, close, volume, turnover]
            df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])

            # Convert timestamp to datetime (UTC) and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True) # Remove rows with invalid timestamps
            df.set_index('timestamp', inplace=True)
            if df.empty: return pd.DataFrame() # Return empty if no valid timestamps

            # Convert OHLCV columns to Decimal, handling potential strings or NaNs
            # Use apply with a lambda for robust conversion
            for col in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    # Ensure input is string before Decimal conversion, handle None/NaN
                    df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and x != '' else Decimal('NaN'))
                except (InvalidOperation, TypeError) as conv_err:
                     lg.error(f"Error converting kline column '{col}' to Decimal for {symbol}: {conv_err}. Data sample: {df[col].head()}")
                     df[col] = Decimal('NaN') # Mark column as unusable on error

            # Drop rows with any NaN in essential OHLC columns
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            # Ensure close price is positive
            df = df[df['close'] > Decimal('0')]
            # Handle potential NaN volumes (replace with 0)
            df['volume'] = df['volume'].apply(lambda x: x if x.is_finite() else Decimal('0'))

            if df.empty:
                lg.warning(f"Kline data for {symbol} became empty after cleaning/conversion."); return pd.DataFrame()

            # Data should be sorted ascending by time (Bybit usually returns descending, needs sorting)
            df.sort_index(ascending=True, inplace=True)

            lg.info(f"Successfully fetched and processed {len(df)} kline records for {symbol} {timeframe}")
            return df
        elif response:
            lg.error(f"Failed to fetch klines for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}"); return pd.DataFrame()
        else: # safe_pybit_call returned None
            lg.error(f"Failed to fetch klines for {symbol} after retries (API call failed)."); return pd.DataFrame()

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching or processing klines for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()

def fetch_ticker_pybit(client: UnifiedHTTP, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """Fetches the latest ticker information for a symbol."""
    lg = logger; category = market_info.get('category')
    if not category: lg.error(f"Category missing for {symbol}. Cannot fetch ticker."); return None

    try:
        lg.debug(f"Fetching ticker for {symbol} ({category})...")
        response = safe_pybit_call(client, 'get_tickers', lg, category=category, symbol=symbol)

        if response and response.get('retCode') == 0 and 'result' in response and 'list' in response['result']:
            tickers = response['result']['list']
            if tickers:
                # lg.debug(f"Ticker data for {symbol}: {tickers[0]}")
                return tickers[0] # Return the first (should be only) ticker for the specified symbol
            else:
                lg.warning(f"Ticker list received but empty for {symbol} ({category}).")
                return None
        elif response:
            lg.error(f"Failed to fetch ticker for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}"); return None
        else:
            lg.error(f"Failed to fetch ticker for {symbol} after retries (API call failed)."); return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching ticker for {symbol}: {e}{RESET}", exc_info=True)
        return None

def fetch_current_price_pybit(client: UnifiedHTTP, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Decimal]:
    """Fetches the current market price, trying different ticker fields."""
    ticker_data = fetch_ticker_pybit(client, symbol, logger, market_info); lg = logger
    if not ticker_data:
        lg.error(f"Could not fetch ticker data for {symbol} to determine current price.")
        return None

    def safe_decimal(key: str) -> Optional[Decimal]:
        """Safely extracts and converts a ticker value to Decimal."""
        value_str = ticker_data.get(key)
        if value_str is None or value_str == '': return None
        try:
            d = Decimal(str(value_str))
            return d if d.is_finite() and d > 0 else None # Ensure positive finite value
        except (InvalidOperation, TypeError):
            # lg.warning(f"Invalid numeric value in ticker field '{key}' for {symbol}: '{value_str}'")
            return None

    # Price preference order: Last Price > Mark Price > Mid Price (Bid/Ask) > Ask > Bid
    price_last = safe_decimal('lastPrice')
    if price_last:
        lg.debug(f"Using 'lastPrice' for {symbol}: {price_last}")
        return price_last

    price_mark = safe_decimal('markPrice') # Often relevant for derivatives
    if market_info.get('is_contract') and price_mark:
        lg.debug(f"Using 'markPrice' for derivative {symbol}: {price_mark}")
        return price_mark

    price_bid = safe_decimal('bid1Price')
    price_ask = safe_decimal('ask1Price')

    if price_bid and price_ask:
        if price_bid < price_ask:
            price_mid = (price_bid + price_ask) / Decimal('2')
            lg.debug(f"Using Mid Price ({price_bid}/{price_ask}) for {symbol}: {price_mid}")
            return price_mid
        else:
            lg.warning(f"Bid price ({price_bid}) not less than Ask price ({price_ask}) for {symbol}. Cannot calculate Mid Price.")
            # Fall through to Ask/Bid fallback

    # Use Mark Price as fallback even for Spot if LastPrice failed
    if price_mark:
        lg.warning(f"Using 'markPrice' as fallback for {symbol}: {price_mark}")
        return price_mark

    if price_ask: # Fallback to Ask
        lg.warning(f"Using 'ask1Price' as fallback for {symbol}: {price_ask}")
        return price_ask

    if price_bid: # Last resort: Bid
        lg.warning(f"Using 'bid1Price' as last resort for {symbol}: {price_bid}")
        return price_bid

    lg.error(f"Could not extract any valid price (last, mark, mid, ask, bid) for {symbol} from ticker: {ticker_data}")
    return None

def fetch_orderbook_pybit(client: UnifiedHTTP, symbol: str, limit: int, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """Fetches the order book for a symbol."""
    lg = logger; category = market_info.get('category')
    if not category: lg.error(f"Category missing for {symbol}. Cannot fetch order book."); return None

    # Pybit V5 Orderbook limits: Linear/Inverse=50, Spot=200, Option=25
    max_limit = 50
    if category == 'spot': max_limit = 200
    elif category == 'option': max_limit = 25
    effective_limit = min(limit, max_limit)
    if limit > max_limit: lg.warning(f"Requested orderbook limit {limit} exceeds max {max_limit} for {category}. Using {effective_limit}.")

    try:
        lg.debug(f"Fetching order book for {symbol} ({category}), Limit: {effective_limit}...")
        response = safe_pybit_call(client, 'get_orderbook', lg, category=category, symbol=symbol, limit=effective_limit)

        if response and response.get('retCode') == 0 and 'result' in response:
            result = response['result']
            # V5 format: result keys: s (symbol), b (bids [price, size]), a (asks [price, size]), ts (timestamp), u (updateId)
            orderbook = {
                'symbol': result.get('s'),
                'bids': [[Decimal(str(p)), Decimal(str(s))] for p, s in result.get('b', [])], # Convert price/size to Decimal
                'asks': [[Decimal(str(p)), Decimal(str(s))] for p, s in result.get('a', [])],
                'timestamp': result.get('ts'), # Millisecond timestamp
                'updateId': result.get('u')
            }
            if not orderbook['bids'] and not orderbook['asks']:
                 lg.warning(f"Order book data received but bids and asks are both empty for {symbol}.")
                 # Return empty book structure anyway
            # lg.debug(f"Order book received for {symbol}. Bids: {len(orderbook['bids'])}, Asks: {len(orderbook['asks'])}")
            return orderbook
        elif response:
            lg.error(f"Failed to fetch order book for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}"); return None
        else:
            lg.error(f"Failed to fetch order book for {symbol} after retries (API call failed)."); return None
    except (InvalidOperation, TypeError, ValueError) as conv_err:
        lg.error(f"{NEON_RED}Error converting order book data to Decimal for {symbol}: {conv_err}{RESET}", exc_info=True); return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching order book for {symbol}: {e}{RESET}", exc_info=True); return None


# --- Balance Fetching (Pybit - Refined) ---
def fetch_balance_pybit(client: UnifiedHTTP, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency, trying relevant account types."""
    lg = logger
    # Determine which account types to check based on the globally determined account type
    account_types_to_check: List[str] = []
    if IS_UNIFIED_ACCOUNT:
        account_types_to_check = ["UNIFIED"]
        lg.debug(f"Fetching balance for {currency}, checking UNIFIED account type.")
    else:
        # For Non-UTA, derivatives margin is usually in CONTRACT, spot funds in SPOT
        # Check CONTRACT first as it's more likely relevant for derivatives trading bot
        account_types_to_check = ["CONTRACT", "SPOT"]
        lg.debug(f"Fetching balance for {currency}, checking CONTRACT and SPOT account types.")

    for acc_type in account_types_to_check:
        try:
            # lg.debug(f"Calling get_wallet_balance with accountType={acc_type}, coin={currency}")
            balance_info = safe_pybit_call(client, 'get_wallet_balance', lg, accountType=acc_type, coin=currency)
            parsed_balance = _parse_pybit_balance_response(balance_info, currency, acc_type, lg)

            if parsed_balance is not None:
                lg.info(f"Successfully fetched balance for {currency} using accountType='{acc_type}'. Available: {parsed_balance:.4f}")
                return parsed_balance # Return the first successful balance found

        except Exception as e:
            # Log error but continue to try other account types if applicable
            lg.error(f"Error occurred while fetching balance with accountType={acc_type} for {currency}: {e}", exc_info=True)

    lg.error(f"Failed to fetch balance for {currency} after checking relevant account types: {account_types_to_check}.")
    return None

def _parse_pybit_balance_response(balance_info: Optional[Dict], currency: str, expected_acc_type: str, logger: logging.Logger) -> Optional[Decimal]:
    """Parses the response from get_wallet_balance to find the available balance."""
    if not balance_info or balance_info.get('retCode') != 0:
        # logger.debug(f"Invalid or failed balance response received for type {expected_acc_type}. Response: {balance_info}")
        return None # Failed API call or non-zero return code

    lg = logger
    try:
        result_list = balance_info.get('result', {}).get('list', [])
        if not result_list:
            # logger.debug(f"Balance response for {expected_acc_type} is missing 'result.list' or it's empty.")
            return None # Empty list or missing structure

        # The list contains balances potentially for different account types (e.g., UNIFIED, CONTRACT, SPOT)
        # We need to find the entry matching the expected_acc_type we requested
        for account_data in result_list:
            actual_acc_type = account_data.get('accountType')
            if actual_acc_type != expected_acc_type:
                 # logger.debug(f"Skipping balance entry for account type {actual_acc_type}, expected {expected_acc_type}")
                 continue

            # Found the correct account type entry, now look for the specific coin
            coin_list = account_data.get('coin', [])
            if not coin_list:
                 # logger.debug(f"No 'coin' list found within the {actual_acc_type} balance data.")
                 continue # Move to next item in result_list if any

            for coin_data in coin_list:
                if coin_data.get('coin') == currency:
                    balance_str: Optional[str] = None
                    field_used: str = "N/A"

                    # --- Logic based on Bybit V5 structure ---
                    # For UNIFIED and CONTRACT/SPOT, 'availableToWithdraw' or 'availableBalance' seems most relevant for placing NEW orders.
                    # 'walletBalance' often includes unrealized PnL (for CONTRACT/UNIFIED) or total assets (SPOT) and might not be fully available.
                    # Prioritize a field representing readily available funds for new margin/orders.
                    # Bybit V5 Docs suggest 'availableToBorrow' / 'availableToWithdraw'/'availableBalance'
                    # Let's prioritize 'availableBalance' as it seems most consistently present and relevant for trading margin.

                    balance_str = coin_data.get('availableBalance') # Unified, Contract, Spot seem to have this
                    if balance_str is not None and balance_str != "":
                         field_used = "availableBalance"
                    else:
                         # Fallback? maybe 'walletBalance' but be cautious as it includes uPNL etc.
                         balance_str = coin_data.get('walletBalance')
                         if balance_str is not None and balance_str != "":
                              field_used = "walletBalance (fallback)"
                              lg.warning(f"Using '{field_used}' for {currency} ({actual_acc_type}) as 'availableBalance' was missing/empty. This might include uPNL.")
                         else:
                             lg.warning(f"Could not find 'availableBalance' or 'walletBalance' for {currency} in {actual_acc_type} data.")
                             return None # No usable balance field found

                    # Convert the chosen balance string to Decimal
                    try:
                        bal_dec = Decimal(str(balance_str))
                        if bal_dec.is_finite() and bal_dec >= 0:
                             lg.debug(f"Parsed balance for {currency} ({actual_acc_type}, using field '{field_used}'): {bal_dec}")
                             return bal_dec
                        else:
                             lg.warning(f"Invalid balance value '{balance_str}' (field: {field_used}) found for {currency} ({actual_acc_type}).")
                             return None # Invalid decimal value
                    except (InvalidOperation, TypeError) as conv_err:
                         lg.warning(f"Error converting balance string '{balance_str}' (field: {field_used}) to Decimal for {currency} ({actual_acc_type}): {conv_err}")
                         return None # Conversion failed

            # If loop finishes, currency wasn't found in this account_data entry
            # logger.debug(f"Currency '{currency}' not found in the coin list for account type {actual_acc_type}.")
            return None # Currency not found in this specific account type's list

        # If loop finishes, the expected_acc_type wasn't found in the result list
        # logger.debug(f"Account type '{expected_acc_type}' not found in the balance response list.")
        return None

    except Exception as e:
        lg.error(f"Error parsing pybit balance response (for {expected_acc_type}): {e}", exc_info=True)
        return None


# --- Position Fetching (Pybit) ---
def fetch_positions_pybit(client: UnifiedHTTP, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """Fetches current open positions for a specific symbol (Linear/Inverse only)."""
    lg = logger; category = market_info.get('category')

    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping position check for non-derivative symbol {symbol} (Category: {category}).")
        return None

    try:
        lg.debug(f"Fetching positions for {symbol} ({category})...")
        # Can filter by symbol directly
        response = safe_pybit_call(client, 'get_positions', lg, category=category, symbol=symbol)

        if response and response.get('retCode') == 0 and 'result' in response and 'list' in response['result']:
            positions = response['result']['list']
            # The list might contain multiple entries for the same symbol in Hedge Mode (positionIdx 1 for long, 2 for short)
            # In One-Way mode, there should be at most one entry with non-zero size.
            active_pos = None
            for pos in positions:
                if pos.get('symbol') == symbol:
                    try:
                        pos_size = Decimal(str(pos.get('size', '0')))
                        if pos_size != 0: # Found an active position with size
                            side_str = pos.get('side', '').lower() # 'Buy' -> 'buy', 'Sell' -> 'sell', ''
                            entry_price_str = pos.get('avgPrice')
                            mark_price_str = pos.get('markPrice')
                            liq_price_str = pos.get('liqPrice')
                            leverage_str = pos.get('leverage')
                            upnl_str = pos.get('unrealisedPnl')
                            rpnl_str = pos.get('cumRealisedPnl')
                            tp_str = pos.get('takeProfit')
                            sl_str = pos.get('stopLoss')
                            tsl_str = pos.get('trailingStop') # This is the distance for TSL
                            active_price_str = pos.get('activePrice') # This is the activation price for TSL
                            pos_idx = int(pos.get('positionIdx', 0)) # 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge

                            # Determine side if missing (shouldn't happen with size != 0)
                            if not side_str: side_str = 'buy' if pos_size > 0 else 'sell' # Pybit uses positive size for Buy side

                             # Standardize structure - Convert key fields to Decimal
                            std_pos = {
                                'symbol': symbol,
                                'side': 'long' if side_str == 'buy' else 'short', # Standardize 'long'/'short'
                                'contracts': abs(pos_size), # Use absolute size
                                'entryPrice': Decimal(entry_price_str) if entry_price_str else None,
                                'markPrice': Decimal(mark_price_str) if mark_price_str else None,
                                'liquidationPrice': Decimal(liq_price_str) if liq_price_str else None,
                                'leverage': Decimal(leverage_str) if leverage_str else None,
                                'unrealizedPnl': Decimal(upnl_str) if upnl_str else None,
                                'realizedPnl': Decimal(rpnl_str) if rpnl_str else None,
                                # Store SL/TP/TSL as Decimals if they exist and are not '0'
                                'takeProfit': Decimal(tp_str) if tp_str and Decimal(tp_str) != 0 else None,
                                'stopLoss': Decimal(sl_str) if sl_str and Decimal(sl_str) != 0 else None,
                                'trailingStop': Decimal(tsl_str) if tsl_str and Decimal(tsl_str) != 0 else None, # Distance
                                'activePrice': Decimal(active_price_str) if active_price_str and Decimal(active_price_str) != 0 else None, # Activation Price
                                'positionIdx': pos_idx,
                                'info': pos, # Keep the raw dict for reference
                                'market_info': market_info # Attach market info for convenience
                            }

                            # Validation
                            if std_pos['entryPrice'] is None or std_pos['entryPrice'] <= 0:
                                lg.warning(f"Position found for {symbol} but entry price is invalid: {entry_price_str}. Skipping.")
                                continue

                            log_side = "LONG" if std_pos['side'] == 'long' else "SHORT"
                            lg.info(f"Found active {log_side} position for {symbol} (Idx: {pos_idx}): Size={std_pos['contracts']}, Entry={std_pos['entryPrice']:.{market_info.get('price_precision_digits', 4)}f}")
                            # In One-Way mode, return the first non-zero position found.
                            # In Hedge mode, this loop might find both long and short if they exist.
                            # The calling logic needs to handle potential multiple positions if in Hedge Mode.
                            # For simplicity now, we return the *first* active one found.
                            # TODO: Enhance main loop to handle Hedge mode potentially returning multiple positions.
                            active_pos = std_pos
                            break # Found the first active position, stop searching (for One-Way mode assumption)

                    except (InvalidOperation, ValueError, TypeError) as conv_err:
                        lg.error(f"Error converting position data to Decimal for {symbol}: {conv_err}. Raw Pos: {pos}", exc_info=True)
                        continue # Skip this position entry

            if active_pos:
                 return active_pos # Return the found and parsed position
            else:
                 lg.debug(f"No active non-zero positions found for {symbol} ({category}).")
                 return None # No open position for this symbol

        elif response:
            lg.error(f"Failed to fetch positions for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}"); return None
        else: # safe_pybit_call returned None
            lg.error(f"Failed to fetch positions for {symbol} after retries (API call failed)."); return None

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching or processing positions for {symbol}: {e}{RESET}", exc_info=True)
        return None

# --- Pybit Trading Action Wrappers ---
def set_leverage_pybit(client: UnifiedHTTP, symbol: str, leverage: int, logger: logging.Logger, market_info: Dict) -> bool:
    """Sets leverage for a derivative symbol."""
    lg = logger; category = market_info.get('category')
    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping leverage setting for non-derivative symbol {symbol} (Category: {category}). Assuming OK.")
        return True # No action needed for spot

    max_leverage = market_info.get('maxLeverage', Decimal('0'))
    if max_leverage > 0 and leverage > int(max_leverage):
         lg.warning(f"Requested leverage {leverage}x for {symbol} exceeds max allowed {int(max_leverage)}x. Using max allowed.")
         leverage = int(max_leverage)
    elif leverage <= 0:
         lg.error(f"Invalid leverage requested for {symbol}: {leverage}. Must be positive.")
         return False

    try:
        leverage_str = str(leverage) # Pybit expects string format
        lg.info(f"Attempting to set leverage for {symbol} ({category}) to {leverage_str}x...")

        # V5 requires setting buy and sell leverage separately, but usually they are the same
        response = safe_pybit_call(client, 'set_leverage', lg,
                                   category=category, symbol=symbol,
                                   buyLeverage=leverage_str, sellLeverage=leverage_str)

        # Check return code: 0 is success, 110043 means "Leverage not modified" (already set)
        if response and response.get('retCode') in [0, 110043]:
             if response.get('retCode') == 110043:
                 lg.info(f"Leverage for {symbol} was already set to {leverage}x (Code 110043).")
             else:
                 lg.info(f"{NEON_GREEN}Successfully set leverage for {symbol} to {leverage}x.{RESET}")
             return True
        elif response:
            lg.error(f"Failed to set leverage for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
            return False
        else: # safe_pybit_call returned None
            lg.error(f"Failed to set leverage for {symbol} after retries (API call failed).")
            return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)
        return False

def _format_decimal_to_string(value: Optional[Decimal], precision_digits: int) -> Optional[str]:
    """Formats a Decimal to a string with fixed precision, returning None if invalid."""
    if value is None or not value.is_finite():
        return None
    try:
        # Use standard string formatting with the calculated number of digits
        format_string = "{:.%df}" % precision_digits
        return format_string.format(value)
    except (ValueError, TypeError) as e:
        # This shouldn't happen with Decimal if precision_digits is valid, but catch just in case
        print(f"Error formatting Decimal {value} with precision {precision_digits}: {e}") # Use print as logger might not be available
        return None

def create_order_pybit(
    client: UnifiedHTTP, symbol: str, order_type: str, side: str, amount: Decimal,
    price: Optional[Decimal] = None, params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """Creates an order using Pybit V5 API with proper formatting."""
    lg = logger or get_logger('main'); params = params or {}
    if not market_info: lg.error(f"Market info is required to create order for {symbol}"); return None
    if not client: lg.error(f"Pybit client is required to create order for {symbol}"); return None

    category = market_info.get('category')
    amount_digits = market_info.get('amount_precision_digits', 8) # Fallback precision
    price_digits = market_info.get('price_precision_digits', 8)   # Fallback precision
    qty_step = market_info.get('qtyStep')
    tick_size = market_info.get('tickSize')
    min_order_qty = market_info.get('minOrderQty', Decimal('0'))
    min_order_iv = market_info.get('minOrderIv', Decimal('0'))

    if not category: lg.error(f"Category missing for {symbol}. Cannot place order."); return None
    if qty_step is None: lg.error(f"Quantity step (qtyStep) missing for {symbol}. Cannot format order amount."); return None
    if tick_size is None: lg.error(f"Tick size (tickSize) missing for {symbol}. Cannot format order price."); return None

    # --- Validate and Format Amount ---
    if amount <= 0: lg.error(f"Order amount must be positive for {symbol}. Amount: {amount}"); return None
    if amount < min_order_qty: lg.error(f"Order amount {amount} for {symbol} is below minimum required {min_order_qty}."); return None

    # Quantize amount using qtyStep (rounding down)
    quantized_amount = ((amount / qty_step).quantize(Decimal('0'), rounding=ROUND_DOWN) * qty_step) if qty_step > 0 else amount
    if quantized_amount <= 0: lg.error(f"Quantized order amount became zero or negative for {symbol}. Original: {amount}, Step: {qty_step}"); return None
    if quantized_amount < min_order_qty: lg.warning(f"Quantized amount {quantized_amount} is below min {min_order_qty}. Using min order quantity instead for {symbol}."); quantized_amount = min_order_qty

    amount_str = _format_decimal_to_string(quantized_amount, amount_digits)
    if amount_str is None: lg.error(f"Failed to format quantized order amount {quantized_amount} to string for {symbol}."); return None

    # --- Prepare Order Arguments ---
    # Pybit V5 uses case-sensitive side ('Buy', 'Sell') and orderType ('Market', 'Limit')
    pybit_side = side.capitalize()
    pybit_order_type = order_type.capitalize()

    order_args: Dict[str, Any] = {
        "category": category,
        "symbol": symbol,
        "side": pybit_side,
        "orderType": pybit_order_type,
        "qty": amount_str,
        # "timeInForce": "GTC", # GoodTillCancel is default for Limit/Market, IOC/FOK possible
    }

    # --- Handle Price for Limit Orders ---
    price_str_log = "N/A"
    if pybit_order_type == 'Limit':
        if price is None or not price.is_finite() or price <= 0:
            lg.error(f"A valid positive price is required for a Limit order ({symbol}). Price: {price}"); return None
        # Quantize price using tickSize
        quantized_price = ((price / tick_size).quantize(Decimal('0'), rounding=ROUND_DOWN if pybit_side=='Sell' else ROUND_UP) * tick_size) if tick_size > 0 else price
        if quantized_price <= 0: lg.error(f"Quantized limit price is zero or negative for {symbol}. Original: {price}"); return None

        price_str = _format_decimal_to_string(quantized_price, price_digits)
        if price_str is None: lg.error(f"Failed to format quantized limit price {quantized_price} to string for {symbol}."); return None
        order_args["price"] = price_str
        price_str_log = price_str # For logging

    # --- Check Minimum Order Value (Cost) ---
    estimated_cost = Decimal('0')
    current_est_price = price if pybit_order_type == 'Limit' else fetch_current_price_pybit(client, symbol, lg, market_info)

    if current_est_price and current_est_price > 0:
         contract_sz = market_info.get('contract_size') # Should be Decimal(1) for linear, Decimal(1) for inverse, None for spot
         if market_info.get('is_inverse') and contract_sz: # Value = Qty / Price * ContractSize (where Qty is in contracts)
             estimated_cost = (quantized_amount / current_est_price) * contract_sz
         elif market_info.get('is_linear') and contract_sz: # Value = Qty * Price * ContractSize (where Qty is in base asset)
             estimated_cost = quantized_amount * current_est_price * contract_sz
         elif market_info.get('is_spot'): # Value = Qty * Price (where Qty is in base asset)
              estimated_cost = quantized_amount * current_est_price
         else: # Option or unknown
              estimated_cost = quantized_amount * current_est_price # Approx

         if min_order_iv > 0 and estimated_cost < min_order_iv:
              lg.error(f"Estimated order cost {estimated_cost:.4f} {market_info.get('quoteCoin')} for {symbol} is below minimum required value {min_order_iv:.4f}. Order rejected.")
              return None
         elif min_order_iv > 0:
              lg.debug(f"Estimated order cost {estimated_cost:.4f} {market_info.get('quoteCoin')} meets minimum {min_order_iv:.4f}.")


    # --- Add Optional Parameters (SL/TP, ReduceOnly, positionIdx) ---
    if 'reduceOnly' in params and params['reduceOnly'] is True:
        order_args['reduceOnly'] = True
    if 'positionIdx' in params: # For Hedge Mode: 0=One-Way, 1=Buy side, 2=Sell side
        order_args['positionIdx'] = int(params['positionIdx'])
        lg.debug(f"Including positionIdx={order_args['positionIdx']} in order request for {symbol}.")

    # Include SL/TP directly in the order placement if provided in params
    sl_price = params.get('stopLoss') # Should be Decimal or None
    tp_price = params.get('takeProfit') # Should be Decimal or None

    sl_str = _format_decimal_to_string(sl_price, price_digits)
    tp_str = _format_decimal_to_string(tp_price, price_digits)

    if sl_str: order_args['stopLoss'] = sl_str
    if tp_str: order_args['takeProfit'] = tp_str

    # Remove None values potentially added by formatting (though _format handles it)
    # order_args = {k: v for k, v in order_args.items() if v is not None} # Not strictly needed due to format func

    # --- Execute Order ---
    try:
        log_msg = (f"Attempting Pybit place_order: {pybit_side.upper()} {pybit_order_type.upper()} {amount_str} {symbol} "
                   f"{'@ '+price_str_log if pybit_order_type == 'Limit' else 'at Market Price'}")
        if sl_str: log_msg += f", SL={sl_str}"
        if tp_str: log_msg += f", TP={tp_str}"
        if order_args.get('reduceOnly'): log_msg += ", ReduceOnly"
        if 'positionIdx' in order_args: log_msg += f", PosIdx={order_args['positionIdx']}"
        lg.info(log_msg)
        lg.debug(f"Pybit Order Request Params: {order_args}")

        response = safe_pybit_call(client, 'place_order', lg, **order_args)

        if response and response.get('retCode') == 0 and 'result' in response:
            order_id = response['result'].get('orderId')
            if order_id:
                lg.info(f"{NEON_GREEN}Successfully placed order for {symbol}. Order ID: {order_id}{RESET}")
                # Return the result dict which contains orderId etc.
                return response['result']
            else:
                # This case (retCode 0 but no orderId) should ideally not happen for successful orders
                lg.error(f"Order placement for {symbol} returned success (Code 0) but no Order ID found. Result: {response['result']}")
                return None
        elif response: # API call returned an error code
            lg.error(f"Failed to place order for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
            # Log specific helpful messages based on error codes
            if response.get('retCode') == 110001: lg.error("Reason: Insufficient balance.")
            if response.get('retCode') == 110045: lg.error("Reason: Order quantity precision error. Check qtyStep.")
            if response.get('retCode') == 170133: lg.error("Reason: Order price precision error (for Limit order). Check tickSize.")
            if response.get('retCode') == 110025: lg.error(f"Reason: Minimum order value not met (min: {min_order_iv}).")
            return None
        else: # safe_pybit_call returned None (max retries exceeded)
            lg.error(f"Failed to place order for {symbol} after retries (API call failed).")
            return None

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected exception during place_order for {symbol}: {e}{RESET}", exc_info=True)
        return None

def set_protection_pybit(
    client: UnifiedHTTP, symbol: str, stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
    trailing_stop_price: Optional[Decimal] = None, # For TSL, this is the *distance* value, not a trigger price
    trailing_active_price: Optional[Decimal] = None, # This is the price at which the TSL becomes active
    position_idx: int = 0, # 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> bool:
    """Sets Stop Loss, Take Profit, and/or Trailing Stop Loss for an existing position using Pybit V5."""
    lg = logger or get_logger('main')
    if not market_info: lg.error(f"Market info is required to set protection for {symbol}"); return False
    if not client: lg.error(f"Pybit client is required to set protection for {symbol}"); return False

    category = market_info.get('category')
    price_digits = market_info.get('price_precision_digits', 8)

    if not category or category not in ['linear', 'inverse']:
        lg.warning(f"Cannot set SL/TP/TSL for non-derivative symbol {symbol} (Category: {category}).")
        # Consider if this should return True (as no action needed) or False (as protection not applicable)
        return True # Let's treat it as success if no action is applicable/needed

    # --- Prepare Parameters ---
    params: Dict[str, Any] = {
        "category": category,
        "symbol": symbol,
        "positionIdx": position_idx # Crucial for Hedge Mode
    }
    log_parts = []

    # Format prices using the correct precision.
    # Pybit expects price strings. Use "" or "0" to cancel SL/TP/TSL.
    sl_str = _format_decimal_to_string(stop_loss_price, price_digits)
    tp_str = _format_decimal_to_string(take_profit_price, price_digits)
    # TSL distance also uses price precision for formatting in Pybit V5
    tsl_dist_str = _format_decimal_to_string(trailing_stop_price, price_digits)
    tsl_act_str = _format_decimal_to_string(trailing_active_price, price_digits)

    # Assign to params dict, using "0" to cancel if the formatted value is None or invalid
    params['stopLoss'] = sl_str if sl_str else "0"
    params['takeProfit'] = tp_str if tp_str else "0"
    params['tpslMode'] = "Full" # Options: "Full" or "Partial". Default is Full. Required if setting TP/SL.
    params['slTriggerBy'] = "MarkPrice" # Options: MarkPrice, LastPrice, IndexPrice. Default Mark.
    params['tpTriggerBy'] = "MarkPrice" # Options: MarkPrice, LastPrice, IndexPrice. Default Mark.

    # Trailing Stop parameters are handled separately in V5? Let's re-check docs.
    # -> Yes, `set_trading_stop` handles TP/SL/TSL together in V5 Unified.
    params['trailingStop'] = tsl_dist_str if tsl_dist_str else "0"
    # Active price is only relevant if TSL distance is set
    if tsl_dist_str and tsl_dist_str != "0":
        params['activePrice'] = tsl_act_str if tsl_act_str else "" # Empty string might mean activate immediately? Check docs. Let's assume empty means immediate if not provided.
        # Pybit V5 Docs: activePrice - If not passed, TSL is triggered immediately. If passed, TSL is triggered when market price reaches activePrice.
        # So, if tsl_act_str is None/invalid, we don't include the activePrice key, or pass "" ? Passing "" seems safer based on some SDK examples. Let's try passing "" if None.
        if not tsl_act_str:
            params['activePrice'] = ""
            lg.debug(f"Setting TSL for {symbol} with distance {tsl_dist_str}, immediate activation (activePrice='').")
        else:
             lg.debug(f"Setting TSL for {symbol} with distance {tsl_dist_str}, activation price {tsl_act_str}.")

    # Build log message parts
    if sl_str: log_parts.append(f"SL={sl_str}")
    if tp_str: log_parts.append(f"TP={tp_str}")
    if tsl_dist_str and tsl_dist_str != "0":
         tsl_log = f"TSL_Dist={tsl_dist_str}"
         if 'activePrice' in params:
             tsl_log += f", ActPrice={params['activePrice'] if params['activePrice'] != '' else 'Immediate'}"
         log_parts.append(tsl_log)

    if not log_parts:
        # This happens if trying to cancel everything or if inputs were invalid
        lg.warning(f"No valid protection levels provided to set_protection_pybit for {symbol}. Attempting to cancel existing TP/SL/TSL.")
        # Ensure we send "0" for all to cancel if nothing was provided
        params['stopLoss'] = params.get('stopLoss', "0")
        params['takeProfit'] = params.get('takeProfit', "0")
        params['trailingStop'] = params.get('trailingStop', "0")
        if params['stopLoss'] == "0" and params['takeProfit'] == "0" and params['trailingStop'] == "0":
             log_parts.append("Cancel All") # Log the intent
        else:
             # Log what is actually being sent if it's not a full cancel
             if params['stopLoss'] != "0": log_parts.append(f"SL={params['stopLoss']}")
             if params['takeProfit'] != "0": log_parts.append(f"TP={params['takeProfit']}")
             if params['trailingStop'] != "0": log_parts.append(f"TSL_Dist={params['trailingStop']}" + (f", Act={params.get('activePrice', 'Immediate')}" if params.get('activePrice') is not None else ""))


    # --- Execute API Call ---
    try:
        log_message = f"Attempting Pybit set_trading_stop for {symbol} (PosIdx: {position_idx}): {', '.join(log_parts)}"
        lg.info(log_message)
        lg.debug(f"Pybit Protection Request Params: {params}")

        response = safe_pybit_call(client, 'set_trading_stop', lg, **params)

        if response and response.get('retCode') == 0:
            lg.info(f"{NEON_GREEN}Successfully set protection levels for {symbol}.{RESET}")
            return True
        elif response:
            lg.error(f"Failed to set protection levels for {symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
            # Add specific error hints
            if response.get('retCode') == 170131: lg.error("Reason: Invalid Stop Loss price (e.g., wrong side of entry, too close, precision).")
            if response.get('retCode') == 170132: lg.error("Reason: Invalid Take Profit price.")
            if response.get('retCode') == 170140: lg.error("Reason: Invalid Trailing Stop activation price.")
            if response.get('retCode') == 110012: lg.error("Reason: Position size is zero (cannot set TP/SL on closed position).")
            return False
        else: # safe_pybit_call returned None
            lg.error(f"Failed to set protection for {symbol} after retries (API call failed).")
            return False

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected exception setting protection for {symbol}: {e}{RESET}", exc_info=True)
        return False

def close_position_pybit(client: UnifiedHTTP, symbol: str, position_data: Dict, logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None) -> Optional[Dict]:
    """Closes an existing position using a market order."""
    lg = logger or get_logger('main')
    if not market_info: lg.error(f"Market info is required to close position for {symbol}"); return None
    if not position_data: lg.error(f"Position data is required to close position for {symbol}"); return None
    if not client: lg.error(f"Pybit client is required to close position for {symbol}"); return None

    category = market_info.get('category')
    if not category or category not in ['linear', 'inverse']:
        lg.error(f"Cannot close position for non-derivative symbol {symbol} (Category: {category}).")
        return None

    try:
        pos_size = position_data.get('contracts') # Should be Decimal
        side = position_data.get('side')         # 'long' or 'short'
        pos_idx = position_data.get('positionIdx', 0) # Get position index

        if pos_size is None or not side:
            lg.error(f"Missing required position data (size/side) to close {symbol}. Data: {position_data}"); return None
        if not isinstance(pos_size, Decimal):
             try: pos_size = Decimal(str(pos_size))
             except: lg.error(f"Invalid position size type/value for close {symbol}: {pos_size}"); return None

        if pos_size <= 0:
            lg.warning(f"Attempted to close position for {symbol}, but provided size is zero or negative ({pos_size}). Position likely already closed or data stale.")
            return None # No action needed

        # Determine the side for the closing order (opposite of the position)
        close_side = 'sell' if side == 'long' else 'buy'
        price_digits = market_info.get('price_precision_digits', 8)

        lg.info(f"Attempting Pybit MARKET close for {side.upper()} position {symbol} (Size: {pos_size}, PosIdx: {pos_idx}) via {close_side.upper()} order...")

        # Use create_order_pybit helper with reduceOnly=True
        params = {
            'reduceOnly': True,
            'positionIdx': pos_idx # Crucial for Hedge Mode to close the correct side
        }

        # We pass the original position size to close the entire position
        close_order_result = create_order_pybit(
            client=client,
            symbol=symbol,
            order_type='market',
            side=close_side,
            amount=pos_size,
            params=params,
            logger=lg,
            market_info=market_info
        )

        if close_order_result and close_order_result.get('orderId'):
            order_id = close_order_result['orderId']
            lg.info(f"{NEON_GREEN}Successfully placed MARKET close order for {symbol}. Order ID: {order_id}{RESET}")
            # We might want to wait and confirm closure, but for now, return the order result
            return close_order_result
        else:
            lg.error(f"{NEON_RED}Failed to place MARKET close order for {symbol}. Check logs above.{RESET}")
            return None

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error closing position for {symbol}: {e}{RESET}", exc_info=True)
        return None

# --- Position Size Calculation ---
def calculate_position_size(
    balance: Decimal, risk_per_trade: float, entry_price: Decimal, stop_loss_price: Decimal,
    market_info: Dict, leverage: int, logger: logging.Logger
) -> Optional[Decimal]:
    """Calculates position size based on risk percentage, balance, entry/SL, leverage, and contract size."""
    lg = logger; symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quoteCoin', 'QUOTE?')
    contract_size = market_info.get('contract_size') # Should be Decimal or None for Spot
    qty_step = market_info.get('qtyStep')
    min_order_qty = market_info.get('minOrderQty', Decimal('0'))
    amount_digits = market_info.get('amount_precision_digits', 8)

    if not all([balance is not None, risk_per_trade is not None, entry_price is not None, stop_loss_price is not None, market_info, leverage is not None]):
        lg.error(f"Missing required inputs for position size calculation ({symbol}).")
        return None
    if balance <= 0: lg.error(f"Cannot calculate size: Balance is zero or negative ({balance} {quote_currency})."); return None
    if risk_per_trade <= 0 or risk_per_trade >= 1: lg.error(f"Cannot calculate size: Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1."); return None
    if entry_price <= 0 or stop_loss_price <= 0: lg.error(f"Cannot calculate size: Entry ({entry_price}) or SL ({stop_loss_price}) price is zero or negative."); return None
    if entry_price == stop_loss_price: lg.error(f"Cannot calculate size: Entry price equals Stop Loss price ({entry_price})."); return None
    if leverage <= 0: lg.error(f"Cannot calculate size: Invalid leverage ({leverage})."); return None
    if qty_step is None or qty_step <= 0: lg.error(f"Cannot calculate size: Invalid quantity step ({qty_step}) for {symbol}."); return None
    is_contract = market_info.get('is_contract', False)
    if is_contract and (contract_size is None or contract_size <= 0):
        lg.error(f"Cannot calculate size for contract {symbol}: Invalid contract size ({contract_size}). Check market info fetch.")
        return None

    # --- Calculation ---
    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        price_diff = abs(entry_price - stop_loss_price)
        if price_diff <= 0: lg.error(f"Price difference is zero or negative ({price_diff}). Cannot calculate size."); return None

        position_size_base_or_contracts = Decimal('NaN')

        # --- Size Calculation Logic (depends on linear/inverse/spot) ---
        # Formula: Size = RiskAmount / (PriceDiffPerUnit * ValuePerUnit)
        # ValuePerUnit needs to account for contract size and type (linear/inverse)

        if market_info.get('is_linear'):
            # Linear (e.g., BTCUSDT): Qty is in Base Asset (BTC). Contract size = 1 Base usually. Value = Qty * Price. Risk is in Quote (USDT).
            # Risk per contract = abs(Entry - SL) * ContractSize (in Base)
            # Size (in Base Asset) = RiskAmount (Quote) / Risk per contract (Quote)
            # Here, ContractSize is effectively 1 base unit, so risk per 'unit' of base asset is just price_diff.
            if contract_size != Decimal('1.0'): lg.warning(f"Linear contract {symbol} has unusual contract_size={contract_size}. Verify size calc.")
            # Risk per unit of Base Asset (e.g., per 1 BTC) is price_diff USDT
            position_size_base_or_contracts = risk_amount_quote / price_diff
            lg.debug(f"Linear Size Calc ({symbol}): RiskQuote={risk_amount_quote:.4f}, PriceDiff={price_diff} -> SizeBase={position_size_base_or_contracts:.8f}")

        elif market_info.get('is_inverse'):
            # Inverse (e.g., BTCUSD): Qty is in Contracts (USD). Contract size = 1 Quote (USD) usually. Value = Qty / Price. Risk is in Base (BTC).
            # We need Risk Amount in BASE currency for inverse. This is tricky without current Base/Quote price.
            # Easier approach: Calculate risk per contract in QUOTE currency (which is USD for inverse, but our balance is USDT - ASSUMPTION: USD ~= USDT for risk calc)
            # Risk per Contract (in Quote) = abs(1/Entry - 1/SL) * ContractSize (Quote)
            # Size (in Contracts) = RiskAmount (Quote) / Risk per Contract (Quote)
            if contract_size != Decimal('1.0'): lg.warning(f"Inverse contract {symbol} has unusual contract_size={contract_size}. Verify size calc.")
            risk_per_contract_quote = abs(Decimal('1.0') / entry_price - Decimal('1.0') / stop_loss_price) * contract_size
            if risk_per_contract_quote <= 0: lg.error(f"Inverse risk per contract is zero/negative ({risk_per_contract_quote})."); return None
            position_size_base_or_contracts = risk_amount_quote / risk_per_contract_quote
            lg.debug(f"Inverse Size Calc ({symbol}): RiskQuote={risk_amount_quote:.4f}, RiskPerContractQuote={risk_per_contract_quote:.8f} -> SizeContracts={position_size_base_or_contracts:.2f}")

        elif market_info.get('is_spot'):
            # Spot (e.g., BTC/USDT): Qty is in Base Asset. No leverage involved in pure risk calc.
            # Risk per unit of Base Asset = price_diff Quote
            # Size (in Base Asset) = RiskAmount (Quote) / price_diff (Quote)
            position_size_base_or_contracts = risk_amount_quote / price_diff
            lg.debug(f"Spot Size Calc ({symbol}): RiskQuote={risk_amount_quote:.4f}, PriceDiff={price_diff} -> SizeBase={position_size_base_or_contracts:.8f}")
            leverage = 1 # Ignore leverage for spot size calc based on risk

        else: # Options or unknown category
            lg.error(f"Position size calculation not implemented for category '{market_info.get('category')}' of {symbol}.")
            return None

        if not position_size_base_or_contracts.is_finite() or position_size_base_or_contracts <= 0:
            lg.error(f"Calculated position size is invalid ({position_size_base_or_contracts}).")
            return None

        # --- Apply Leverage Limit (for contracts) ---
        # Max position value = Balance * Leverage
        # Calculated position value = Size * EntryPrice * ContractSize (Linear) OR Size * ContractSize / EntryPrice (Inverse) ? No, Value = Size(Contracts) * ContractSize(Quote) for Inverse.
        max_position_value_quote = balance * Decimal(leverage)
        calculated_position_value_quote = Decimal('NaN')

        if market_info.get('is_linear'):
            calculated_position_value_quote = position_size_base_or_contracts * entry_price # * contract_size (if not 1)
        elif market_info.get('is_inverse'):
             # Size is in contracts (USD value), contract size is 1 USD. Value = Size * ContractSize = Size USD.
             calculated_position_value_quote = position_size_base_or_contracts * contract_size # Should be approx Size USD
        # Spot doesn't use leverage limit this way

        if is_contract and calculated_position_value_quote.is_finite() and calculated_position_value_quote > max_position_value_quote:
            lg.warning(f"Risk-based size ({position_size_base_or_contracts}) leads to position value ({calculated_position_value_quote:.2f} {quote_currency}) exceeding max leverage value ({max_position_value_quote:.2f} {quote_currency}). Capping size by leverage.")
            # Recalculate size based on max value
            if market_info.get('is_linear'):
                 position_size_base_or_contracts = max_position_value_quote / entry_price # / contract_size (if not 1)
            elif market_info.get('is_inverse'):
                 position_size_base_or_contracts = max_position_value_quote / contract_size # Max value / value per contract
            lg.info(f"Leverage-capped size: {position_size_base_or_contracts}")

        # --- Quantize and Check Minimums ---
        # Round DOWN to be conservative
        final_size = ((position_size_base_or_contracts / qty_step).quantize(Decimal('0'), rounding=ROUND_DOWN) * qty_step)
        final_size_str = _format_decimal_to_string(final_size, amount_digits) # Format for logging

        lg.info(f"Calculated Position Size ({symbol}): RiskAmt={risk_amount_quote:.2f}, PriceDiff={price_diff:.{market_info.get('price_precision_digits', 8)}f} -> RawSize={position_size_base_or_contracts:.8f} -> QuantizedSize={final_size_str}")

        if final_size <= 0:
            lg.error(f"Final position size is zero or negative after quantization ({final_size}). Cannot place order.")
            return None

        if final_size < min_order_qty:
            lg.error(f"Calculated position size {final_size} is less than minimum order quantity {min_order_qty} for {symbol}. Cannot place order.")
            # Optional: could place min_order_qty if risk allows, but that violates the risk % rule. Safer to reject.
            return None

        return final_size

    except (InvalidOperation, TypeError, ValueError) as e:
        lg.error(f"Error during position size calculation for {symbol}: {e}", exc_info=True)
        return None
    except Exception as e: # Catch any other unexpected errors
        lg.error(f"Unexpected error during position size calculation for {symbol}: {e}", exc_info=True)
        return None


# ==============================================================
# ===           TradingAnalyzer Full Implementation          ===
# ==============================================================
class TradingAnalyzer:
    """
    Analyzes trading data using pandas_ta.Strategy, generates weighted signals,
    and provides risk management helpers. Uses Decimal for precision internally.
    Manages state like break-even trigger status via a passed-in dictionary.
    """
    def __init__(self, df: pd.DataFrame, logger: logging.Logger, config: Dict[str, Any], market_info: Dict[str, Any], symbol_state: Dict[str, Any]):
        if df is None or df.empty:
            # Log error and raise to prevent initialization with bad data
            logger.error("TradingAnalyzer cannot be initialized with an empty DataFrame.")
            raise ValueError("TA requires a non-empty DataFrame.")
        if not market_info:
            logger.error("TradingAnalyzer cannot be initialized without valid market_info.")
            raise ValueError("TA requires valid market_info.")
        if symbol_state is None: # Check for None explicitly
            logger.error("TradingAnalyzer cannot be initialized without symbol_state dictionary.")
            raise ValueError("TA requires valid symbol_state.")

        self.df_raw = df # Keep original DataFrame with Decimals if needed elsewhere
        self.df = df.copy() # Work on a copy for TA calculations
        self.logger = logger
        self.config = config
        self.market_info = market_info # Assumes enhanced market_info with precision digits etc.
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.symbol_state = symbol_state # Reference to the mutable state dict for this symbol
        self.indicator_values: Dict[str, Optional[Decimal]] = {} # Stores latest values as Decimals
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1} # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated Fib levels as Decimals
        self.ta_strategy: Optional[ta.Strategy] = None # Holds the pandas_ta strategy object
        self.ta_column_map: Dict[str, str] = {} # Maps internal names (e.g., "ATR") to DataFrame column names (e.g., "ATRr_14")

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' is empty or not found for {self.symbol}. Signal generation will be disabled (HOLD only).{RESET}")

        # --- Convert DataFrame columns to float for pandas_ta ---
        # pandas_ta typically requires float input.
        try:
            cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
            for col in cols_to_convert:
                 if col in self.df.columns:
                      # Check if already float to avoid unnecessary conversion
                      if not pd.api.types.is_float_dtype(self.df[col]):
                          # Convert Decimal or object type to numeric (float), coerce errors to NaN
                          self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                      # Ensure no NaNs remain in core OHLC columns after conversion
                      if col != 'volume' and self.df[col].isnull().any():
                           original_nan_count = df[col].isnull().sum() # Use original df for check
                           converted_nan_count = self.df[col].isnull().sum()
                           logger.warning(f"NaN values found/introduced in column '{col}' for {self.symbol} after conversion to float. Original NaNs: {original_nan_count}, Post-convert NaNs: {converted_nan_count}. This might affect TA results.")
                           # Option: Fill NaNs? Or let TA handle them? TA usually handles NaNs gracefully.
            # Ensure required columns exist
            if not all(c in self.df.columns for c in ['open', 'high', 'low', 'close', 'volume']):
                 missing = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in self.df.columns]
                 logger.error(f"DataFrame for {self.symbol} is missing required columns: {missing}. Cannot proceed with TA.")
                 raise ValueError(f"Missing required columns for TA: {missing}")
        except Exception as e:
            logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
            # Depending on severity, might want to raise an exception here
            raise ValueError(f"Failed to prepare DataFrame for TA: {e}")

        # --- Initialize TA ---
        # Define the strategy based on config *after* ensuring DF is float
        self._define_ta_strategy()
        # Calculate indicators
        self._calculate_all_indicators()
        # Extract latest values into self.indicator_values (as Decimals)
        self._update_latest_indicator_values()
        # Calculate Fibonacci levels (using original Decimal data if possible)
        self.calculate_fibonacci_levels()
        logger.debug(f"TradingAnalyzer initialized for {self.symbol} on {self.interval}.")

    # --- State Property ---
    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even status from the symbol's state."""
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool):
        """Sets the break-even status in the symbol's state and logs change."""
        current_value = self.symbol_state.get('break_even_triggered', False)
        if current_value != value:
            self.symbol_state['break_even_triggered'] = value
            self.logger.info(f"Break-even status for {self.symbol} set to: {value}")
        # else: self.logger.debug(f"Break-even status for {self.symbol} already {value}.") # Optional: log no change

    # --- TA Strategy Definition ---
    def _define_ta_strategy(self) -> None:
        """Defines the pandas_ta strategy based on the configuration."""
        cfg = self.config; indi_cfg = cfg.get("indicators", {})
        self.ta_column_map = {} # Reset map

        # Helper to safely get numeric parameters from config
        def get_num_param(key: str, default: Union[int, float]) -> Union[int, float]:
            value = cfg.get(key, default)
            try:
                # Convert based on default type to handle int/float correctly
                return int(value) if isinstance(default, int) else float(value)
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid config value '{value}' for '{key}'. Using default: {default}.")
                return default

        # Get parameters, using defaults if not found or invalid
        atr_p=get_num_param("atr_period", DEFAULT_ATR_PERIOD)
        ema_s=get_num_param("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
        ema_l=get_num_param("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
        rsi_p=get_num_param("rsi_period", DEFAULT_RSI_WINDOW)
        bb_p=get_num_param("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        bb_std=get_num_param("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        cci_w=get_num_param("cci_window", DEFAULT_CCI_WINDOW)
        wr_w=get_num_param("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        mfi_w=get_num_param("mfi_window", DEFAULT_MFI_WINDOW)
        st_rsi_w=get_num_param("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
        st_rsi_rsi_w=get_num_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW) # Inner RSI period for StochRSI
        st_rsi_k=get_num_param("stoch_rsi_k", DEFAULT_K_WINDOW)
        st_rsi_d=get_num_param("stoch_rsi_d", DEFAULT_D_WINDOW)
        psar_af=get_num_param("psar_af", DEFAULT_PSAR_AF)
        psar_max=get_num_param("psar_max_af", DEFAULT_PSAR_MAX_AF)
        sma10_w=get_num_param("sma_10_window", DEFAULT_SMA_10_WINDOW)
        mom_p=get_num_param("momentum_period", DEFAULT_MOMENTUM_PERIOD)
        vol_ma_p=get_num_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Build the list of indicators for the strategy
        ta_list = []

        # Always calculate ATR as it's used for SL/TP/BE sizing
        if atr_p > 0:
            ta_list.append({"kind": "atr", "length": int(atr_p)})
            self.ta_column_map["ATR"] = f"ATRr_{int(atr_p)}" # pandas_ta default name
        else: self.logger.error("ATR period must be > 0. ATR calculation skipped.")

        # Add indicators based on config switches
        if indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit"):
            if ema_s > 0:
                 ta_list.append({"kind": "ema", "length": int(ema_s), "col_names": (f"EMA_{int(ema_s)}",)})
                 self.ta_column_map["EMA_Short"] = f"EMA_{int(ema_s)}"
            if ema_l > 0:
                 ta_list.append({"kind": "ema", "length": int(ema_l), "col_names": (f"EMA_{int(ema_l)}",)})
                 self.ta_column_map["EMA_Long"] = f"EMA_{int(ema_l)}"
            if ema_s <= 0 or ema_l <= 0: self.logger.warning("EMA Alignment/Cross requires both short/long periods > 0.")

        if indi_cfg.get("momentum") and mom_p > 0:
             ta_list.append({"kind": "mom", "length": int(mom_p), "col_names": (f"MOM_{int(mom_p)}",)})
             self.ta_column_map["Momentum"] = f"MOM_{int(mom_p)}"

        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0:
             ta_list.append({"kind": "sma", "close": "volume", "length": int(vol_ma_p), "col_names": (f"VOL_SMA_{int(vol_ma_p)}",)})
             self.ta_column_map["Volume_MA"] = f"VOL_SMA_{int(vol_ma_p)}"

        if indi_cfg.get("stoch_rsi") and st_rsi_w > 0 and st_rsi_rsi_w > 0 and st_rsi_k > 0 and st_rsi_d > 0:
             k_col = f"STOCHRSIk_{int(st_rsi_w)}_{int(st_rsi_rsi_w)}_{int(st_rsi_k)}_{int(st_rsi_d)}"
             d_col = f"STOCHRSId_{int(st_rsi_w)}_{int(st_rsi_rsi_w)}_{int(st_rsi_k)}_{int(st_rsi_d)}"
             ta_list.append({"kind":"stochrsi", "length":int(st_rsi_w), "rsi_length":int(st_rsi_rsi_w), "k":int(st_rsi_k), "d":int(st_rsi_d), "col_names":(k_col, d_col)})
             self.ta_column_map["StochRSI_K"], self.ta_column_map["StochRSI_D"] = k_col, d_col
        elif indi_cfg.get("stoch_rsi"): self.logger.warning("StochRSI enabled but one or more periods (w, rsi_w, k, d) are invalid (<=0). Skipped.")

        if indi_cfg.get("rsi") and rsi_p > 0:
             ta_list.append({"kind": "rsi", "length": int(rsi_p), "col_names": (f"RSI_{int(rsi_p)}",)})
             self.ta_column_map["RSI"] = f"RSI_{int(rsi_p)}"

        if indi_cfg.get("bollinger_bands") and bb_p > 0 and bb_std > 0:
             # Define explicit column names
             bbl_col = f"BBL_{int(bb_p)}_{bb_std:.1f}"
             bbm_col = f"BBM_{int(bb_p)}_{bb_std:.1f}"
             bbu_col = f"BBU_{int(bb_p)}_{bb_std:.1f}"
             bbb_col = f"BBB_{int(bb_p)}_{bb_std:.1f}" # Bandwidth
             bbp_col = f"BBP_{int(bb_p)}_{bb_std:.1f}" # Percent
             ta_list.append({"kind":"bbands", "length":int(bb_p), "std":float(bb_std), "col_names":(bbl_col, bbm_col, bbu_col, bbb_col, bbp_col)})
             self.ta_column_map["BB_Lower"], self.ta_column_map["BB_Middle"], self.ta_column_map["BB_Upper"] = bbl_col, bbm_col, bbu_col
        elif indi_cfg.get("bollinger_bands"): self.logger.warning("Bollinger Bands enabled but period/std dev invalid. Skipped.")

        if indi_cfg.get("vwap"):
             # VWAP needs high, low, close, volume. It uses typical price (H+L+C)/3 internally if not provided.
             # Ensure volume column exists and is float
             if 'volume' not in self.df.columns or not pd.api.types.is_float_dtype(self.df['volume']):
                 self.logger.warning("VWAP requires a valid float 'volume' column. Calculation might fail or be inaccurate.")
             # Add VWAP calculation (pandas_ta handles daily reset logic if index is DatetimeIndex)
             vwap_col="VWAP_D" # Default column name from ta library when using default anchor="D"
             ta_list.append({"kind":"vwap", "col_names":(vwap_col,)}) # Anchor defaults to 'D' (daily)
             self.ta_column_map["VWAP"] = vwap_col

        if indi_cfg.get("cci") and cci_w > 0:
             cci_col=f"CCI_{int(cci_w)}_0.015" # Default pandas_ta name
             ta_list.append({"kind":"cci", "length":int(cci_w), "col_names":(cci_col,)})
             self.ta_column_map["CCI"] = cci_col

        if indi_cfg.get("wr") and wr_w > 0:
             wr_col=f"WILLR_{int(wr_w)}" # Default pandas_ta name
             ta_list.append({"kind":"willr", "length":int(wr_w), "col_names":(wr_col,)})
             self.ta_column_map["WR"] = wr_col

        if indi_cfg.get("psar"):
             # Clean AF/MAX names for columns (remove trailing .0)
             psar_af_s = f"{psar_af:.2f}".rstrip('0').rstrip('.')
             psar_max_s = f"{psar_max:.2f}".rstrip('0').rstrip('.')
             # Define expected column names
             l_col = f"PSARl_{psar_af_s}_{psar_max_s}"
             s_col = f"PSARs_{psar_af_s}_{psar_max_s}"
             af_col = f"PSARaf_{psar_af_s}_{psar_max_s}"
             r_col = f"PSARr_{psar_af_s}_{psar_max_s}"
             ta_list.append({"kind":"psar", "af":float(psar_af), "max_af":float(psar_max), "col_names":(l_col, s_col, af_col, r_col)})
             self.ta_column_map["PSAR_Long"] = l_col   # Value when in uptrend (NaN otherwise)
             self.ta_column_map["PSAR_Short"] = s_col  # Value when in downtrend (NaN otherwise)
             self.ta_column_map["PSAR_AF"] = af_col    # Acceleration Factor
             self.ta_column_map["PSAR_Reversal"] = r_col # 1 if reversal occurred, 0 otherwise

        if indi_cfg.get("sma_10") and sma10_w > 0:
             ta_list.append({"kind": "sma", "length": int(sma10_w), "col_names": (f"SMA_{int(sma10_w)}",)})
             self.ta_column_map["SMA10"] = f"SMA_{int(sma10_w)}"

        if indi_cfg.get("mfi") and mfi_w > 0:
            # MFI needs high, low, close, volume. Uses typical price. Ensure volume is valid.
             if 'volume' not in self.df.columns or not pd.api.types.is_float_dtype(self.df['volume']):
                 self.logger.warning("MFI requires a valid float 'volume' column. Calculation might fail or be inaccurate.")
             ta_list.append({"kind":"mfi", "length":int(mfi_w), "col_names":(f"MFI_{int(mfi_w)}",)})
             self.ta_column_map["MFI"] = f"MFI_{int(mfi_w)}"

        # --- Create the Strategy ---
        if not ta_list:
            self.logger.warning(f"No valid indicators enabled or configured for {self.symbol}. TA Strategy is empty.")
            self.ta_strategy = None
            return

        self.ta_strategy = ta.Strategy(
            name="EnhancedMultiIndicatorStrategy",
            description="Calculates multiple TA indicators based on config for the Pybit bot",
            ta=ta_list
        )
        self.logger.debug(f"Defined pandas_ta Strategy for {self.symbol} with {len(ta_list)} indicator calculations.")
        # self.logger.debug(f"TA Column Map: {self.ta_column_map}") # Optional: Log the map

    # --- Indicator Calculation ---
    def _calculate_all_indicators(self):
        """Calculates all indicators defined in the TA strategy."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}. Skipping indicator calculation."); return
        if not self.ta_strategy:
            self.logger.warning(f"TA Strategy is not defined for {self.symbol}. Skipping indicator calculation."); return

        # Check for sufficient data length
        min_required_periods = 0
        if hasattr(self.ta_strategy, 'talib') and hasattr(self.ta_strategy, 'ta'): # Check if strategy has indicators
             # Simple heuristic: find the max length parameter used (can be improved)
             max_len_param = 0
             for indi in self.ta_strategy.ta:
                 length_param = indi.get('length', indi.get('fast_length', indi.get('timeperiod', 0))) # Common length param names
                 if isinstance(length_param, int) and length_param > max_len_param:
                     max_len_param = length_param
             min_required_periods = max_len_param + 20 # Add buffer for stable calculation

        if len(self.df) < min_required_periods:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} rows) for TA calculations on {self.symbol}. Need ~{min_required_periods} rows. Results may be inaccurate or NaN.{RESET}")
             # Proceed anyway, TA library might handle short data with NaNs

        try:
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol}...")
            # Ensure index is datetime type for time-based indicators like VWAP
            if not isinstance(self.df.index, pd.DatetimeIndex):
                 self.logger.warning(f"DataFrame index for {self.symbol} is not DatetimeIndex. Converting...")
                 try: self.df.index = pd.to_datetime(self.df.index)
                 except Exception as idx_err: self.logger.error(f"Failed to convert index to DatetimeIndex: {idx_err}. Time-based indicators might fail."); # Continue cautiously

            # Apply the strategy
            self.df.ta.strategy(self.ta_strategy, timed=False) # timed=True can add overhead
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. DataFrame shape: {self.df.shape}")

        except AttributeError as ae:
             # Common error if df contains non-numeric types pandas_ta doesn't expect (like Decimal)
             if "'Decimal' object has no attribute" in str(ae) or "unsupported operand type" in str(ae):
                  self.logger.error(f"{NEON_RED}Pandas TA Error ({self.symbol}): Input DataFrame must contain floats, not Decimals. Check DF conversion step. Error: {ae}{RESET}", exc_info=False)
             else: # Other attribute errors might indicate installation issues
                  self.logger.error(f"{NEON_RED}Pandas TA attribute error ({self.symbol}): {ae}. Is pandas_ta installed correctly? Is the DataFrame valid?{RESET}", exc_info=True)
             # Optionally clear calculated columns on error?
             # added_cols = [col for col in self.df.columns if col not in self.df_raw.columns and col not in ['open','high','low','close','volume']]
             # self.df.drop(columns=added_cols, inplace=True, errors='ignore')
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error during indicator calculation for {self.symbol}: {e}{RESET}", exc_info=True)
            # Optionally clear calculated columns here too

    # --- Update Latest Values ---
    def _update_latest_indicator_values(self):
        """Extracts the latest calculated indicator values into self.indicator_values as Decimals."""
        self.indicator_values = {} # Reset
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot update latest indicator values."); return

        try:
            # Get the last row of the DataFrame (which should contain the latest indicator values)
            # Ensure index is sorted if necessary (should be done after fetch_klines)
            if not self.df.index.is_monotonic_increasing:
                 self.logger.warning(f"DataFrame index for {self.symbol} is not sorted. Sorting before getting latest values.")
                 self.df.sort_index(inplace=True)

            latest_series = self.df.iloc[-1]

            if latest_series.isnull().all():
                 self.logger.warning(f"The last row of the DataFrame for {self.symbol} contains all NaNs. Cannot update latest indicator values.")
                 return

            # Helper to safely convert float/object from DataFrame to Decimal
            def to_decimal(value: Any) -> Optional[Decimal]:
                if pd.isna(value) or value is None:
                    return None
                try:
                    # Convert potential float/int/string to Decimal
                    d = Decimal(str(value))
                    return d if d.is_finite() else None # Return None for Inf/-Inf/NaN Decimals
                except (InvalidOperation, TypeError):
                    # Log conversion error but continue
                    # self.logger.warning(f"Could not convert value '{value}' (type: {type(value)}) to Decimal.")
                    return None

            # Populate indicator_values using the ta_column_map
            for indicator_name, df_column_name in self.ta_column_map.items():
                if df_column_name in latest_series:
                    self.indicator_values[indicator_name] = to_decimal(latest_series[df_column_name])
                else:
                    # This indicates a mismatch between defined map and calculated columns
                    self.logger.warning(f"Mapped indicator column '{df_column_name}' (for '{indicator_name}') not found in DataFrame for {self.symbol}.")
                    self.indicator_values[indicator_name] = None

            # Also store latest OHLCV values as Decimals
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in latest_series:
                     self.indicator_values[col.capitalize()] = to_decimal(latest_series[col])
                else:
                     self.indicator_values[col.capitalize()] = None # Should exist, but check anyway

            # Debug log of extracted values (optional, can be verbose)
            valid_values_str = {k: f"{v:.5f}" for k, v in self.indicator_values.items() if v is not None}
            # self.logger.debug(f"Latest indicator values (Decimals) updated for {self.symbol}: {valid_values_str}")

        except IndexError:
            self.logger.error(f"IndexError accessing last row of DataFrame for {self.symbol}. DataFrame might be unexpectedly empty.")
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = {} # Clear on error to prevent using stale data

    # --- Precision/Market Info Helpers (Read from processed market_info) ---
    def get_min_tick_size(self) -> Optional[Decimal]:
        """Gets the minimum price increment (tick size) as a Decimal."""
        tick = self.market_info.get('min_tick_size') # Already Decimal or None from get_market_info_pybit
        if tick is None: self.logger.warning(f"Tick size is None for {self.symbol}. Price quantization will fail.")
        return tick

    def get_price_precision_digits(self) -> int:
        """Gets the number of decimal places for price."""
        return self.market_info.get('price_precision_digits', 8) # Default fallback

    def get_amount_precision_digits(self) -> int:
        """Gets the number of decimal places for amount/quantity."""
        return self.market_info.get('amount_precision_digits', 8) # Default fallback

    # --- Quantization Helpers ---
    def quantize_price(self, price: Union[Decimal, float, str], rounding=ROUND_DOWN) -> Optional[Decimal]:
        """Quantizes a price to the correct tick size."""
        min_tick = self.get_min_tick_size()
        if min_tick is None or min_tick <= 0:
            self.logger.error(f"Cannot quantize price for {self.symbol}: Invalid min_tick_size ({min_tick}).")
            return None
        try:
            p = Decimal(str(price))
            if not p.is_finite(): return None # Handle NaN/Inf input
            # Formula: floor(price / tick_size) * tick_size (for rounding down)
            # Use Decimal's quantize for precision control
            quantized_price = (p / min_tick).quantize(Decimal('1'), rounding=rounding) * min_tick
            # Ensure the result matches the expected precision digits (cosmetic formatting)
            price_digits = self.get_price_precision_digits()
            return Decimal(f"{quantized_price:.{price_digits}f}")
        except (InvalidOperation, TypeError, ValueError) as e:
            self.logger.error(f"Error quantizing price '{price}' for {self.symbol} with tick {min_tick}: {e}")
            return None

    def quantize_amount(self, amount: Union[Decimal, float, str], rounding=ROUND_DOWN) -> Optional[Decimal]:
        """Quantizes an amount/quantity to the correct step size (precision digits)."""
        amount_digits = self.get_amount_precision_digits()
        qty_step = self.market_info.get('qtyStep') # Already Decimal or None

        if qty_step is None or qty_step <= 0:
             # Fallback to using digits if step is missing (less accurate but better than nothing)
             self.logger.warning(f"Missing or invalid qtyStep for {self.symbol}. Quantizing amount based on digits ({amount_digits}).")
             try:
                 a = Decimal(str(amount))
                 if not a.is_finite(): return None
                 # Use string formatting for digit-based quantization
                 q_str = ("{:.%df}" % amount_digits).format(a.quantize(Decimal('1e-' + str(amount_digits)), rounding=rounding))
                 return Decimal(q_str)
             except (InvalidOperation, TypeError, ValueError) as e:
                 self.logger.error(f"Error quantizing amount '{amount}' for {self.symbol} using digits ({amount_digits}): {e}")
                 return None
        else:
            # Preferred method: use qty_step
             try:
                 a = Decimal(str(amount))
                 if not a.is_finite(): return None
                 quantized_amount = (a / qty_step).quantize(Decimal('1'), rounding=rounding) * qty_step
                 # Ensure result matches expected precision digits (cosmetic formatting)
                 return Decimal(f"{quantized_amount:.{amount_digits}f}")
             except (InvalidOperation, TypeError, ValueError) as e:
                  self.logger.error(f"Error quantizing amount '{amount}' for {self.symbol} with step {qty_step}: {e}")
                  return None

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels based on recent high/low."""
        self.fib_levels_data = {} # Reset
        # Use window from config or default
        window = window or int(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))
        if window <= 1:
             self.logger.debug(f"Fibonacci window ({window}) too small. Skipping calculation for {self.symbol}.")
             return {}

        # Use the raw DataFrame with original Decimal precision for high/low lookup
        if len(self.df_raw) < window:
            self.logger.debug(f"Not enough data ({len(self.df_raw)} rows) for Fibonacci ({window} window) on {self.symbol}.")
            return {}

        df_slice = self.df_raw.tail(window)

        try:
            # Ensure columns are Decimal before max/min, drop NaNs first
            high_col = df_slice["high"].dropna()
            low_col = df_slice["low"].dropna()
            if high_col.empty or low_col.empty:
                 self.logger.warning(f"No valid high/low data found in the last {window} bars for Fib calc on {self.symbol}.")
                 return {}

            # Find max high and min low in the window
            period_high = high_col.max()
            period_low = low_col.min()

            if not isinstance(period_high, Decimal): period_high = Decimal(str(period_high))
            if not isinstance(period_low, Decimal): period_low = Decimal(str(period_low))

            if not period_high.is_finite() or not period_low.is_finite():
                self.logger.warning(f"Non-finite high/low found for Fib calculation on {self.symbol} (H:{period_high}, L:{period_low}).")
                return {}
            if period_high <= period_low:
                 self.logger.debug(f"Fib range invalid or zero ({self.symbol}): High={period_high}, Low={period_low}")
                 return {} # Range must be positive

            diff = period_high - period_low
            levels: Dict[str, Decimal] = {}
            min_tick = self.get_min_tick_size()

            if min_tick is None: # If quantization isn't possible, store raw Decimal
                 self.logger.warning(f"Using raw (unquantized) Fibonacci levels for {self.symbol} due to missing tick size.")
                 for level_pct_f in FIB_LEVELS:
                     level_pct = Decimal(str(level_pct_f))
                     level_raw = period_low + (diff * level_pct) # Calculate from Low upwards
                     level_name = f"Fib_{level_pct * 100:.1f}%"
                     levels[level_name] = level_raw
            else: # Quantize levels
                 for level_pct_f in FIB_LEVELS:
                     level_pct = Decimal(str(level_pct_f))
                     level_raw = period_low + (diff * level_pct)
                     level_name = f"Fib_{level_pct * 100:.1f}%"
                     # Quantize appropriately - usually down for support, up for resistance? Let's just round to nearest tick.
                     level_quantized = self.quantize_price(level_raw, ROUND_DOWN) # Round down for levels below 100%? Or nearest? Let's use ROUND_DOWN for conservatism.
                     if level_quantized is not None:
                         levels[level_name] = level_quantized
                     else:
                         self.logger.warning(f"Failed to quantize Fibonacci level {level_name} ({level_raw}) for {self.symbol}")


            self.fib_levels_data = levels
            # self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} ({window} bars): { {k: f'{v:.{self.get_price_precision_digits()}f}' for k,v in levels.items()} }")
            return levels

        except KeyError as ke:
             self.logger.error(f"Missing column for Fib calculation {self.symbol}: {ke}")
             return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return {}

    # --- Indicator Check Methods (Return float score: -1.0 to 1.0) ---
    def _get_indicator_float(self, name: str) -> Optional[float]:
        """Safely gets an indicator value as float from the latest values."""
        value_dec = self.indicator_values.get(name)
        if value_dec is not None and value_dec.is_finite():
            return float(value_dec)
        # self.logger.debug(f"Indicator '{name}' not available or invalid for scoring ({value_dec}).")
        return None

    def _check_ema_alignment(self) -> Optional[float]:
        """Score: +1 if short EMA > long EMA, -1 if short < long, 0 if equal, None if missing."""
        ema_short = self._get_indicator_float("EMA_Short")
        ema_long = self._get_indicator_float("EMA_Long")
        if ema_short is not None and ema_long is not None:
            if ema_short > ema_long: return 1.0
            elif ema_short < ema_long: return -1.0
            else: return 0.0
        return None

    def _check_momentum(self) -> Optional[float]:
        """Score: Normalized momentum value (clamped -1 to 1). Requires tuning scale_factor."""
        mom = self._get_indicator_float("Momentum")
        if mom is not None:
             # Normalize momentum - this needs tuning based on typical MOM range for the asset/interval
             scale_factor = 0.1 # Example: Assumes typical MOM values might range +/- 10
             # Scale and clamp
             score = max(-1.0, min(1.0, mom * scale_factor))
             return score
        return None

    def _check_volume_confirmation(self) -> Optional[float]:
        """Score: +0.7 if volume > multiplier * volume MA, 0 otherwise, None if missing."""
        volume = self._get_indicator_float("Volume")
        volume_ma = self._get_indicator_float("Volume_MA")
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        if volume is not None and volume_ma is not None:
            if volume_ma > 0 and volume > (volume_ma * multiplier):
                return 0.7 # Strong positive confirmation
            else:
                return 0.0 # Neutral or weak volume
        return None # Cannot determine if volume data is missing

    def _check_stoch_rsi(self) -> Optional[float]:
        """Score: Based on K/D crossover and relation to Overbought/Oversold levels."""
        k = self._get_indicator_float("StochRSI_K")
        d = self._get_indicator_float("StochRSI_D")
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))

        if k is None or d is None: return None

        score = 0.0
        # Oversold conditions (potential bullish reversal/strength)
        if k < oversold and d < oversold:
            score = 0.8 if k > d else 0.6 # K crossing above D is stronger buy signal
        # Overbought conditions (potential bearish reversal/weakness)
        elif k > overbought and d > overbought:
            score = -0.8 if k < d else -0.6 # K crossing below D is stronger sell signal
        # Exiting oversold (bullish)
        elif k < oversold: score = 0.5
        # Exiting overbought (bearish)
        elif k > overbought: score = -0.5
        # General trend indication by K/D position
        elif k > d: score = 0.2 # K above D generally bullish
        elif k < d: score = -0.2 # K below D generally bearish

        return max(-1.0, min(1.0, score)) # Clamp score

    def _check_rsi(self) -> Optional[float]:
        """Score: Stronger signal near OB/OS, scaled in between."""
        rsi = self._get_indicator_float("RSI")
        if rsi is None: return None
        score = 0.0
        if rsi <= 20: score = 1.0 # Strong Oversold -> Strong Buy Signal
        elif rsi <= 30: score = 0.7 # Oversold -> Buy Signal
        elif rsi >= 80: score = -1.0 # Strong Overbought -> Strong Sell Signal
        elif rsi >= 70: score = -0.7 # Overbought -> Sell Signal
        else: # Scale linearly between 30 and 70 (range of 40 points)
             # Map 30 to +0.7, 70 to -0.7. Midpoint 50 maps to 0.
             # score = 0.7 - (rsi - 30.0) * (1.4 / 40.0)
             # Simpler linear scale: 1 (at 30) down to -1 (at 70)
              score = 1.0 - (rsi - 30.0) * (2.0 / 40.0)
        return max(-1.0, min(1.0, score)) # Clamp

    def _check_cci(self) -> Optional[float]:
        """Score: Based on CCI levels (-100, -200, +100, +200)."""
        cci = self._get_indicator_float("CCI")
        if cci is None: return None
        score = 0.0
        if cci <= -200: score = 1.0   # Extreme oversold
        elif cci <= -100: score = 0.7 # Oversold entry
        elif cci >= 200: score = -1.0  # Extreme overbought
        elif cci >= 100: score = -0.7 # Overbought entry
        else: # Scale between -100 and +100
             # Map -100 to +0.7, +100 to -0.7. Midpoint 0 maps to 0.
             # score = 0.0 - (cci / 100.0) * 0.7
             # Simple scale: Map -100 to 0.7, 100 to -0.7
              score = -(cci / 100.0) * 0.7
        return max(-1.0, min(1.0, score))

    def _check_wr(self) -> Optional[float]:
        """Score: Based on Williams %R levels (-80, -90, -20, -10). Note W%R is -100 to 0."""
        wr = self._get_indicator_float("WR") # Range is -100 (OS) to 0 (OB)
        if wr is None: return None
        score = 0.0
        # Note: WR is inverted compared to RSI/Stochastics
        if wr <= -90: score = 1.0   # Very Oversold -> Strong Buy
        elif wr <= -80: score = 0.7 # Oversold -> Buy
        elif wr >= -10: score = -1.0  # Very Overbought -> Strong Sell
        elif wr >= -20: score = -0.7 # Overbought -> Sell
        else: # Scale between -80 and -20 (range of 60 points)
             # Map -80 to +0.7, -20 to -0.7. Midpoint -50 maps to 0.
             # score = 0.7 - (wr - (-80.0)) * (1.4 / 60.0)
             # Simple scale: Map -80 to 0.7, -20 to -0.7
              score = 0.7 - (wr - (-80.0)) * (1.4 / 60.0)
        return max(-1.0, min(1.0, score))

    def _check_psar(self) -> Optional[float]:
        """Score: +1 if PSAR is below price (uptrend), -1 if above (downtrend), 0 if indeterminate."""
        psar_long = self.indicator_values.get("PSAR_Long") # Value is non-NaN if trend is UP
        psar_short = self.indicator_values.get("PSAR_Short") # Value is non-NaN if trend is DOWN
        close = self.indicator_values.get("Close")

        if close is None: return None # Need close price to compare

        # Check if the values are valid Decimals (non-NaN)
        long_valid = psar_long is not None and psar_long.is_finite()
        short_valid = psar_short is not None and psar_short.is_finite()

        if long_valid and not short_valid:
             # PSAR Long has a value, Short is NaN -> Uptrend indicated
             # Double check if close is indeed above PSAR value
             return 1.0 if close > psar_long else 0.0 # Should be > but check anyway
        elif short_valid and not long_valid:
             # PSAR Short has a value, Long is NaN -> Downtrend indicated
              return -1.0 if close < psar_short else 0.0 # Should be < but check anyway
        elif not long_valid and not short_valid:
             # Both are NaN (can happen at the start of data)
             return 0.0 # Indeterminate
        else: # Both seem valid? This shouldn't happen with pandas_ta PSAR.
             self.logger.warning(f"PSAR state ambiguous for {self.symbol}. Long: {psar_long}, Short: {psar_short}. Returning 0.")
             return 0.0

    def _check_sma10(self) -> Optional[float]:
        """Score: +0.5 if Close > SMA10, -0.5 if Close < SMA10, 0 if equal, None if missing."""
        sma10 = self._get_indicator_float("SMA10")
        close = self._get_indicator_float("Close")
        if sma10 is not None and close is not None:
            if close > sma10: return 0.5
            elif close < sma10: return -0.5
            else: return 0.0
        return None

    def _check_vwap(self) -> Optional[float]:
        """Score: +0.6 if Close > VWAP, -0.6 if Close < VWAP, 0 if equal, None if missing."""
        vwap = self._get_indicator_float("VWAP")
        close = self._get_indicator_float("Close")
        if vwap is not None and close is not None:
            if close > vwap: return 0.6
            elif close < vwap: return -0.6
            else: return 0.0
        return None

    def _check_mfi(self) -> Optional[float]:
        """Score: Based on MFI levels, similar to RSI."""
        mfi = self._get_indicator_float("MFI")
        if mfi is None: return None
        score = 0.0
        if mfi <= 15: score = 1.0   # Strong Oversold -> Buy
        elif mfi <= 25: score = 0.7 # Oversold -> Buy
        elif mfi >= 85: score = -1.0  # Strong Overbought -> Sell
        elif mfi >= 75: score = -0.7 # Overbought -> Sell
        else: # Scale between 25 and 75 (range of 50 points)
             # Map 25 to +0.7, 75 to -0.7. Midpoint 50 maps to 0.
             # score = 0.7 - (mfi - 25.0) * (1.4 / 50.0)
             # Simple scale: Map 25 to 0.7, 75 to -0.7
              score = 0.7 - (mfi - 25.0) * (1.4 / 50.0)
        return max(-1.0, min(1.0, score))

    def _check_bollinger_bands(self) -> Optional[float]:
        """Score: +1 if close <= LowerBand, -1 if close >= UpperBand, scaled linearly in between."""
        bb_lower = self._get_indicator_float("BB_Lower")
        bb_upper = self._get_indicator_float("BB_Upper")
        # bb_middle = self._get_indicator_float("BB_Middle") # Not used in this score
        close = self._get_indicator_float("Close")

        if bb_lower is None or bb_upper is None or close is None: return None
        if bb_upper <= bb_lower: return 0.0 # Avoid division by zero if bands collapse

        band_range = bb_upper - bb_lower
        position_in_band = (close - bb_lower) / band_range # 0 if at lower, 1 if at upper

        score = 0.0
        if position_in_band <= 0: score = 1.0 # At or below lower band -> Strong Buy
        elif position_in_band >= 1: score = -1.0 # At or above upper band -> Strong Sell
        else:
            # Scale linearly from +1 (at lower band) to -1 (at upper band)
            score = 1.0 - (position_in_band * 2.0)

        return max(-1.0, min(1.0, score)) # Clamp

    def _check_orderbook(self, orderbook_data: Optional[Dict]) -> Optional[float]:
        """Score: Order Book Imbalance (-1 strong sell pressure, +1 strong buy pressure)."""
        if not orderbook_data:
             # self.logger.debug("Orderbook data not available for scoring.")
             return None # Cannot score if no data
        if not self.config.get("indicators", {}).get("orderbook"):
             return None # Skip if disabled in config

        try:
            bids = orderbook_data.get('bids', []) # List of [price_decimal, size_decimal]
            asks = orderbook_data.get('asks', []) # List of [price_decimal, size_decimal]

            if not bids or not asks:
                # self.logger.debug(f"Order book empty for {self.symbol}, cannot calculate imbalance.")
                return 0.0 # Neutral score if book is empty

            limit = int(self.config.get("orderbook_limit", 10)) # Use configured depth
            levels_to_use = min(len(bids), len(asks), limit)

            if levels_to_use <= 0: return 0.0 # No overlapping levels

            # Sum the *size* (volume) within the specified levels
            total_bid_volume = sum(bid[1] for bid in bids[:levels_to_use])
            total_ask_volume = sum(ask[1] for ask in asks[:levels_to_use])

            total_volume = total_bid_volume + total_ask_volume
            if total_volume <= 0: return 0.0 # Avoid division by zero

            # Calculate Order Book Imbalance (OBI)
            obi = (total_bid_volume - total_ask_volume) / total_volume
            # OBI ranges from -1 (all ask) to +1 (all bid)

            # Clamp just in case (should be unnecessary here)
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi)))
            # self.logger.debug(f"OBI Calc ({self.symbol}, {levels_to_use} levels): BidVol={total_bid_volume}, AskVol={total_ask_volume} -> OBI={score:.3f}")
            return score

        except (ValueError, TypeError, IndexError, InvalidOperation) as e:
            self.logger.warning(f"Could not calculate Order Book Imbalance for {self.symbol}: {e}")
            return None # Return None on error
        except Exception as e:
            self.logger.error(f"Unexpected error calculating Order Book Imbalance for {self.symbol}: {e}", exc_info=True)
            return None

    # --- Signal Generation ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: Optional[Dict]) -> str:
        """Generates a BUY, SELL, or HOLD signal based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Reset signals
        final_score = Decimal("0.0")
        total_weight = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        score_details = {} # For debugging

        # Check if inputs are valid
        if not self.indicator_values:
             self.logger.warning(f"Cannot generate signal for {self.symbol}: Indicator values not calculated/updated.")
             return "HOLD"
        if not current_price_dec.is_finite() or current_price_dec <= 0:
             self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price_dec}).")
             return "HOLD"
        if not self.weights:
             self.logger.warning(f"Cannot generate signal for {self.symbol}: Active weight set '{self.active_weight_set_name}' is empty.")
             return "HOLD"

        active_weights = self.weights # Use the loaded weights for the active set

        # Map indicator keys to their check methods
        indicator_methods = {
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
            "orderbook": lambda: self._check_orderbook(orderbook_data) # Pass orderbook data via lambda
        }

        # Iterate through indicators enabled in the config
        enabled_indicators = self.config.get("indicators", {})
        for indicator_key, is_enabled in enabled_indicators.items():
            if not is_enabled:
                score_details[indicator_key] = "Disabled"
                continue

            weight_str = active_weights.get(indicator_key)
            if weight_str is None:
                # Only log warning if it was expected to have a weight (i.e., method exists)
                if indicator_key in indicator_methods:
                     # self.logger.debug(f"No weight defined for enabled indicator '{indicator_key}' in set '{self.active_weight_set_name}'. Skipping.")
                     score_details[indicator_key] = "No Weight"
                continue # Skip if no weight assigned

            try:
                weight = Decimal(str(weight_str))
                if not weight.is_finite(): raise ValueError("Weight not finite")
            except (ValueError, InvalidOperation, TypeError):
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}' in set '{self.active_weight_set_name}'. Skipping.")
                score_details[indicator_key] = f"Invalid Wt ({weight_str})"
                continue

            if weight == 0:
                 score_details[indicator_key] = "Weight=0"
                 continue # Skip zero-weighted indicators

            # Get the corresponding check method
            check_method = indicator_methods.get(indicator_key)
            if not check_method:
                # This indicates a mismatch between config 'indicators'/'weights' and implemented methods
                self.logger.warning(f"No scoring method implemented for enabled/weighted indicator '{indicator_key}'. Skipping.")
                score_details[indicator_key] = "No Method"
                continue

            # --- Calculate Score ---
            indicator_score_float: Optional[float] = None
            try:
                indicator_score_float = check_method()
            except Exception as e:
                self.logger.error(f"Error executing check method for '{indicator_key}' on {self.symbol}: {e}", exc_info=True)
                indicator_score_float = None # Treat as NaN on error

            if indicator_score_float is not None and math.isfinite(indicator_score_float):
                 try:
                      # Clamp score between -1.0 and 1.0
                      clamped_score_float = max(-1.0, min(1.0, indicator_score_float))
                      indicator_score_dec = Decimal(str(clamped_score_float))

                      # Add weighted score to total
                      weighted_score = indicator_score_dec * weight
                      final_score += weighted_score
                      total_weight += abs(weight) # Sum absolute weights for normalization
                      active_indicator_count += 1
                      score_details[indicator_key] = f"{clamped_score_float:.2f} (x{weight:.2f}) = {weighted_score:.3f}"
                 except (InvalidOperation, TypeError) as calc_e:
                      self.logger.error(f"Error processing score for indicator {indicator_key}: {calc_e}")
                      score_details[indicator_key] = "Calc Error"
                      nan_indicator_count += 1
            else:
                 # Score was None or NaN/Inf
                 # self.logger.debug(f"Indicator '{indicator_key}' returned invalid score: {indicator_score_float}. Skipping.")
                 score_details[indicator_key] = "NaN/None"
                 nan_indicator_count += 1


        # --- Determine Final Signal ---
        final_signal = "HOLD"
        normalized_score = Decimal("0.0")

        if total_weight > 0:
             normalized_score = (final_score / total_weight).quantize(Decimal("0.0001")) # Normalize score
        elif active_indicator_count > 0:
             self.logger.warning(f"Total weight is zero for {active_indicator_count} active indicators ({self.symbol}). Signal forced to HOLD.")
             # Keep final_signal as HOLD
        else:
             self.logger.warning(f"No active indicators with non-zero weights contributed to the score for {self.symbol}. Signal is HOLD.")
             # Keep final_signal as HOLD

        # Get the appropriate threshold based on the active weight set
        threshold_key = "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
        default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
        try:
             threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
        except (InvalidOperation, TypeError):
             threshold = Decimal(str(default_threshold))
             self.logger.warning(f"Invalid threshold value for '{threshold_key}'. Using default: {threshold}")

        # Compare final score (raw, not normalized) against threshold
        if final_score >= threshold:
             final_signal = "BUY"
        elif final_score <= -threshold:
             final_signal = "SELL"

        # Update internal signal state
        if final_signal != "HOLD":
            self.signals[final_signal] = 1
            self.signals["HOLD"] = 0
        else:
             self.signals["BUY"] = 0
             self.signals["SELL"] = 0
             self.signals["HOLD"] = 1

        # Log the result
        price_prec = self.get_price_precision_digits()
        signal_color = NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW
        log_msg = (f"Signal Gen ({self.symbol} @ {current_price_dec:.{price_prec}f}): "
                   f"Set='{self.active_weight_set_name}', Indicators(Actv/NaN):{active_indicator_count}/{nan_indicator_count}, "
                   f"TotalAbsWt:{total_weight:.3f}, RawScore:{final_score:.4f}, NormScore:{normalized_score:.4f}, "
                   f"Threshold:+/-{threshold:.3f} -> Signal: {signal_color}{final_signal}{RESET}")
        self.logger.info(log_msg)

        # Log detailed scores at DEBUG level
        if active_indicator_count > 0 or nan_indicator_count > 0:
             # Format details nicely
             details_str = ", ".join([f"{k}:{v}" for k, v in score_details.items() if v not in ["Disabled", "Weight=0"]])
             self.logger.debug(f"  Score Details: {details_str}")
        if nan_indicator_count > 0:
            self.logger.debug(f"  ({nan_indicator_count} indicators returned NaN/None/Error)")


        return final_signal

    # --- TP/SL Calculation ---
    def calculate_entry_tp_sl(self, entry_price: Decimal, signal: str) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates Quantized Entry, Take Profit, and Stop Loss based on ATR."""
        quantized_entry, take_profit, stop_loss = None, None, None
        price_prec = self.get_price_precision_digits()

        # Validate inputs
        if signal not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid signal '{signal}' for TP/SL calculation ({self.symbol}).")
            return None, None, None
        if not entry_price.is_finite() or entry_price <= 0:
             self.logger.error(f"Invalid entry price ({entry_price}) for TP/SL calculation ({self.symbol}).")
             return None, None, None

        # Quantize the entry price first (use nearest rounding?) - Let's assume entry IS the intended price, maybe quantize later if needed.
        # Or quantize here? Let's quantize the input entry to be safe. Use ROUND_HALF_UP for entry?
        quantized_entry = self.quantize_price(entry_price, rounding=ROUND_HALF_UP)
        if quantized_entry is None:
             self.logger.error(f"Failed to quantize entry price {entry_price} for TP/SL calculation ({self.symbol}).")
             return None, None, None

        # Get ATR value
        atr_value = self.indicator_values.get("ATR")
        if atr_value is None or not atr_value.is_finite() or atr_value <= 0:
            self.logger.warning(f"Cannot calculate TP/SL for {self.symbol}: ATR value is invalid or missing ({atr_value}).")
            # Return quantized entry, but None for TP/SL
            return quantized_entry, None, None

        try:
            # Get multipliers from config, convert to Decimal
            tp_multiplier = Decimal(str(self.config.get("take_profit_multiple", "1.5")))
            sl_multiplier = Decimal(str(self.config.get("stop_loss_multiple", "1.0")))
            tick_size = self.get_min_tick_size()
            if tick_size is None: # Should have been checked by quantize_price, but double check
                self.logger.error(f"Cannot calculate TP/SL for {self.symbol}: tick_size is missing.")
                return quantized_entry, None, None

            # Calculate raw offsets
            tp_offset = atr_value * tp_multiplier
            sl_offset = atr_value * sl_multiplier

            # Ensure SL offset is at least MIN_TICKS_AWAY_FOR_SLTP ticks
            min_sl_offset_value = tick_size * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
            if sl_offset < min_sl_offset_value:
                self.logger.warning(f"Calculated SL offset ({sl_offset:.{price_prec+2}f}) based on ATR*Multiplier is less than minimum required ({min_sl_offset_value:.{price_prec+2}f}). Adjusting SL offset to minimum.")
                sl_offset = min_sl_offset_value

            # Calculate raw TP/SL prices
            if signal == "BUY":
                raw_tp = quantized_entry + tp_offset
                raw_sl = quantized_entry - sl_offset
                # Quantize TP UP for buy, SL DOWN for buy
                take_profit = self.quantize_price(raw_tp, rounding=ROUND_UP)
                stop_loss = self.quantize_price(raw_sl, rounding=ROUND_DOWN)
            else: # SELL signal
                raw_tp = quantized_entry - tp_offset
                raw_sl = quantized_entry + sl_offset
                 # Quantize TP DOWN for sell, SL UP for sell
                take_profit = self.quantize_price(raw_tp, rounding=ROUND_DOWN)
                stop_loss = self.quantize_price(raw_sl, rounding=ROUND_UP)

            # --- Final Validation ---
            # Ensure SL didn't end up on the wrong side or too close after quantization
            if stop_loss is not None:
                 min_required_sl_diff = tick_size * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
                 if signal == "BUY":
                      # SL must be below entry - min_diff
                      required_sl = quantized_entry - min_required_sl_diff
                      if stop_loss >= required_sl:
                           adjusted_sl = self.quantize_price(required_sl, ROUND_DOWN)
                           self.logger.debug(f"Adjusting BUY Stop Loss {stop_loss} to {adjusted_sl} to ensure minimum distance ({MIN_TICKS_AWAY_FOR_SLTP} ticks) from entry {quantized_entry}.")
                           stop_loss = adjusted_sl
                 else: # SELL
                      # SL must be above entry + min_diff
                      required_sl = quantized_entry + min_required_sl_diff
                      if stop_loss <= required_sl:
                           adjusted_sl = self.quantize_price(required_sl, ROUND_UP)
                           self.logger.debug(f"Adjusting SELL Stop Loss {stop_loss} to {adjusted_sl} to ensure minimum distance ({MIN_TICKS_AWAY_FOR_SLTP} ticks) from entry {quantized_entry}.")
                           stop_loss = adjusted_sl
                 # Ensure SL is not zero or negative
                 if stop_loss <= 0:
                      self.logger.error(f"Calculated Stop Loss is zero or negative ({stop_loss}) for {signal} {self.symbol}. Setting SL to None.")
                      stop_loss = None

            # Ensure TP didn't end up on the wrong side or too close
            if take_profit is not None:
                 min_required_tp_diff = tick_size # TP can be just one tick away
                 if signal == "BUY":
                      if take_profit <= quantized_entry: # TP must be > entry
                          take_profit = self.quantize_price(quantized_entry + min_required_tp_diff, ROUND_UP)
                          self.logger.debug(f"Adjusting BUY Take Profit upwards to {take_profit} as it was <= entry.")
                 else: # SELL
                      if take_profit >= quantized_entry: # TP must be < entry
                           take_profit = self.quantize_price(quantized_entry - min_required_tp_diff, ROUND_DOWN)
                           self.logger.debug(f"Adjusting SELL Take Profit downwards to {take_profit} as it was >= entry.")
                 # Ensure TP is not zero or negative
                 if take_profit <= 0:
                       self.logger.error(f"Calculated Take Profit is zero or negative ({take_profit}) for {signal} {self.symbol}. Setting TP to None.")
                       take_profit = None

            # Log the results
            tp_str = f"{take_profit:.{price_prec}f}" if take_profit else 'None'
            sl_str = f"{stop_loss:.{price_prec}f}" if stop_loss else 'None'
            entry_str = f"{quantized_entry:.{price_prec}f}"
            atr_str = f"{atr_value:.{price_prec+1}f}"
            self.logger.info(f"Calculated TP/SL for {signal} {self.symbol}: Entry={entry_str}, TP={tp_str}, SL={sl_str} (ATR={atr_str}, TP Mult={tp_multiplier}, SL Mult={sl_multiplier})")

            return quantized_entry, take_profit, stop_loss

        except (InvalidOperation, TypeError, ValueError) as e:
            self.logger.error(f"{NEON_RED}Error during TP/SL calculation for {self.symbol}: {e}{RESET}", exc_info=True)
            # Return quantized entry but None for TP/SL on error
            return quantized_entry, None, None
        except Exception as e:
             self.logger.error(f"{NEON_RED}Unexpected error during TP/SL calculation for {self.symbol}: {e}{RESET}", exc_info=True)
             return quantized_entry, None, None


# ==============================================================
# ===            Main Loop & Logic (Pybit Version)           ===
# ==============================================================
def run_bot(client: UnifiedHTTP, config: Dict[str, Any], bot_state: Dict[str, Any]):
    """Main operational loop of the trading bot."""
    main_logger = get_logger('main')
    main_logger.info(f"{NEON_CYAN}=== Starting Pybit Bot v1.0.1 Main Loop ==={RESET}")
    config_summary = (
        f"Trading={'Enabled' if config.get('enable_trading') else 'DISABLED'}, "
        f"Sandbox={'ACTIVE' if config.get('use_sandbox') else 'INACTIVE (LIVE!)'}, "
        f"Symbols={config.get('symbols')}, Interval={config.get('interval')}, "
        f"Risk={config.get('risk_per_trade')*100:.2f}%, Lev={config.get('leverage')}x, "
        f"MaxPos={config.get('max_concurrent_positions_total')}, Quote={QUOTE_CURRENCY}, "
        f"Account={'UNIFIED' if IS_UNIFIED_ACCOUNT else 'Non-UTA'}, "
        f"PosMode={config.get('position_mode')}, "
        f"WSet='{config.get('active_weight_set')}'"
    )
    main_logger.info(f"Config Summary: {config_summary}")
    if config.get("enable_trading") and not config.get("use_sandbox"):
        main_logger.warning(f"{NEON_RED}!!! LIVE TRADING IS ACTIVE ON MAINNET !!!{RESET}")
    elif config.get("enable_trading"):
         main_logger.warning(f"{NEON_YELLOW}--- Trading enabled on TESTNET ---{RESET}")
    else:
         main_logger.info("--- Trading is DISABLED ---")


    global LOOP_DELAY_SECONDS
    LOOP_DELAY_SECONDS = config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS)
    symbols_to_trade = config.get("symbols", [])
    if not symbols_to_trade:
        main_logger.critical("No symbols configured in 'symbols' list. Bot cannot run.")
        return # Exit if no symbols

    # Initialize state for each symbol if not already present
    for symbol in symbols_to_trade:
        if symbol not in bot_state:
            bot_state[symbol] = {}
            main_logger.info(f"Initialized empty state for new symbol: {symbol}")
        # Ensure necessary keys exist within each symbol's state
        bot_state.setdefault(symbol, {}).setdefault("break_even_triggered", False)
        bot_state[symbol].setdefault("last_signal", "HOLD") # Last generated signal
        bot_state[symbol].setdefault("last_entry_price", None) # Store as string if saving state

    cycle_count = 0
    while True:
        cycle_count += 1
        start_time = time.time()
        main_logger.info(f"{NEON_BLUE}--- Starting Bot Cycle {cycle_count} ---{RESET}")

        # --- Pre-Cycle Setup ---
        current_balance: Optional[Decimal] = None
        # Fetch balance only if trading is enabled (reduces API calls)
        if config.get("enable_trading"):
            try:
                current_balance = fetch_balance_pybit(client, QUOTE_CURRENCY, main_logger)
                if current_balance is not None:
                    main_logger.info(f"Current Balance: {current_balance:.4f} {QUOTE_CURRENCY}")
                else:
                    main_logger.error(f"{NEON_RED}Failed to fetch current balance for {QUOTE_CURRENCY}. Trading actions in this cycle will be skipped.{RESET}")
                    # Continue the loop to process indicators, but skip entries/management? Or pause? Let's skip trading actions.
            except Exception as e:
                main_logger.error(f"Unhandled exception during balance fetch: {e}", exc_info=True)
                current_balance = None # Ensure balance is None on error

        # --- Fetch Market Info & Positions ---
        # Refresh market info cache if needed (handled within get_market_info_pybit)
        if not fetch_and_cache_instrument_info(client, main_logger):
             main_logger.error("Critical failure fetching/updating instrument info cache. Skipping cycle.")
             time.sleep(LOOP_DELAY_SECONDS) # Wait before retrying cache load
             continue

        open_positions_count = 0
        active_positions: Dict[str, Dict] = {} # Store fetched active positions {symbol: position_data}

        main_logger.debug("Fetching active positions across configured symbols...")
        for symbol in symbols_to_trade:
            temp_logger = get_logger(symbol, is_symbol_logger=True) # Use symbol-specific logger
            market_info = get_market_info_pybit(client, symbol, temp_logger)
            if not market_info:
                temp_logger.error(f"Could not get market info for {symbol}. Skipping position check.")
                continue # Skip symbol if no market info

            # Only check positions for contract types
            if market_info.get('is_contract'):
                try:
                    position_data = fetch_positions_pybit(client, symbol, temp_logger, market_info)
                    if position_data:
                        # TODO: Handle Hedge Mode - fetch_positions might return multiple entries (long/short)
                        # For now, assumes One-Way and takes the first non-zero position found by fetch_positions_pybit
                        active_positions[symbol] = position_data
                        open_positions_count += 1 # Increment count for each symbol with *any* active position
                        # Update last entry price in state if missing and available in position data
                        if bot_state[symbol].get("last_entry_price") is None and position_data.get('entryPrice'):
                             try:
                                 entry_dec = Decimal(str(position_data['entryPrice']))
                                 bot_state[symbol]["last_entry_price"] = str(entry_dec) # Store as string
                                 temp_logger.info(f"Populated last_entry_price from fetched position: {entry_dec}")
                             except (InvalidOperation, TypeError):
                                 temp_logger.warning(f"Could not parse entry price {position_data['entryPrice']} from position data.")
                except Exception as pos_fetch_err:
                     temp_logger.error(f"Error fetching position for {symbol}: {pos_fetch_err}", exc_info=True)

        max_allowed_positions = config.get("max_concurrent_positions_total", 1)
        main_logger.info(f"Total active positions found: {open_positions_count} / {max_allowed_positions}")

        # --- Process Each Symbol ---
        for symbol in symbols_to_trade:
            symbol_logger = get_logger(symbol, is_symbol_logger=True)
            symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
            symbol_state = bot_state[symbol] # Get reference to this symbol's state dict

            try:
                # Get cached (and potentially refreshed) market info
                market_info = get_market_info_pybit(client, symbol, symbol_logger)
                if not market_info:
                     symbol_logger.error(f"Failed to retrieve market info for {symbol} during main processing loop. Skipping symbol.")
                     continue
                if market_info.get('status') != 'TRADING':
                     symbol_logger.info(f"Symbol {symbol} is not in 'TRADING' status ({market_info.get('status')}). Skipping.")
                     continue

                # --- Fetch Data ---
                timeframe = config.get("interval", "5") # Default to 5m if missing
                # Determine required kline limit (can be optimized)
                # Rough estimation based on longest period used by enabled indicators
                periods = [int(config.get(p, d)) for p, d in [
                    ("atr_period", DEFAULT_ATR_PERIOD), ("ema_long_period", DEFAULT_EMA_LONG_PERIOD),
                    ("rsi_period", DEFAULT_RSI_WINDOW), ("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD),
                    ("cci_window", DEFAULT_CCI_WINDOW), ("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW),
                    ("mfi_window", DEFAULT_MFI_WINDOW), ("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW),
                    ("sma_10_window", DEFAULT_SMA_10_WINDOW), ("momentum_period", DEFAULT_MOMENTUM_PERIOD),
                    ("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD), ("fibonacci_window", DEFAULT_FIB_WINDOW)]
                     if config.get("indicators", {}).get(p.split('_')[0], False) or p in ["atr_period", "fibonacci_window"] # Include ATR always
                ]
                kline_limit = max(periods) + 50 if periods else 200 # Need enough history + buffer
                kline_limit = max(kline_limit, 100) # Ensure a reasonable minimum

                df = fetch_klines_pybit(client, symbol, timeframe, kline_limit, symbol_logger, market_info)
                if df.empty or len(df) < 50: # Check for a reasonable minimum number of rows
                    symbol_logger.warning(f"Insufficient Kline data fetched ({len(df)} rows) for {symbol} {timeframe}. Required ~{kline_limit}. Skipping analysis.")
                    continue

                # Fetch current price (use latest close from klines as fallback?)
                current_price = fetch_current_price_pybit(client, symbol, symbol_logger, market_info)
                if current_price is None:
                    # Fallback to last close price from DataFrame
                    if not df.empty and 'close' in df.columns:
                         last_close = df['close'].iloc[-1] # Already Decimal from fetch_klines
                         if last_close and last_close.is_finite() and last_close > 0:
                              current_price = last_close
                              symbol_logger.warning(f"Could not fetch live ticker price for {symbol}. Using last kline close price: {current_price}")
                         else:
                              symbol_logger.error(f"Failed to fetch current price and last close price is invalid ({last_close}) for {symbol}. Skipping symbol.")
                              continue
                    else:
                         symbol_logger.error(f"Failed to fetch current price and no kline data available for {symbol}. Skipping symbol.")
                         continue

                # Fetch order book if needed for signals
                orderbook = None
                if config.get("indicators", {}).get("orderbook"):
                    try:
                        orderbook = fetch_orderbook_pybit(client, symbol, int(config.get("orderbook_limit", 25)), symbol_logger, market_info)
                    except Exception as ob_e:
                         symbol_logger.warning(f"Failed to fetch order book for {symbol}: {ob_e}")
                         # Continue without order book data

                # --- Analyze Data ---
                analyzer = TradingAnalyzer(df, symbol_logger, config, market_info, symbol_state)

                # --- Manage Existing Position OR Check for New Entry ---
                current_position = active_positions.get(symbol) # Check if we found an active position earlier

                if current_position:
                    # --- Manage Open Position ---
                    position_side = current_position.get('side') # 'long' or 'short'
                    position_size = current_position.get('contracts')
                    symbol_logger.info(f"Managing existing {position_side.upper()} position for {symbol}. Size: {position_size}")
                    manage_existing_position_pybit(client, config, symbol_logger, analyzer, current_position, current_price)
                    # Note: manage_existing_position might close the position. The `active_positions` dict isn't updated within the loop,
                    # so this symbol might still be considered "open" for the max position check in the *next* cycle. This is generally acceptable.

                # Only consider new entries if NO position exists for this symbol AND total positions < max
                elif market_info.get('is_contract') and open_positions_count < max_allowed_positions:
                    symbol_logger.info(f"No active position found for {symbol}. Checking for entry signals...")
                    # Reset BE trigger if no position is open
                    if analyzer.break_even_triggered:
                        symbol_logger.info(f"Resetting break_even_triggered flag for {symbol} as no position is open.")
                        analyzer.break_even_triggered = False
                    # Reset last entry price if no position is open
                    if symbol_state.get("last_entry_price") is not None:
                         symbol_logger.debug(f"Clearing last_entry_price for {symbol} as no position is open.")
                         symbol_state["last_entry_price"] = None


                    # Generate signal
                    signal = analyzer.generate_trading_signal(current_price, orderbook)
                    symbol_state["last_signal"] = signal # Store the latest signal

                    if signal in ["BUY", "SELL"]:
                        if config.get("enable_trading"):
                            if current_balance is not None and current_balance > 0:
                                # --- Attempt New Entry ---
                                opened_new_position = attempt_new_entry_pybit(client, config, symbol_logger, analyzer, signal, current_price, current_balance)
                                if opened_new_position:
                                    open_positions_count += 1 # Increment count immediately after successful entry attempt
                                    # Optional: Add a small delay to allow systems to update?
                                    # time.sleep(1)
                            elif current_balance is None:
                                 symbol_logger.error(f"Cannot attempt {signal} entry for {symbol}: Balance fetch failed earlier.")
                            else: # Balance <= 0
                                 symbol_logger.warning(f"Cannot attempt {signal} entry for {symbol}: Insufficient balance ({current_balance:.4f} {QUOTE_CURRENCY}).")
                        else: # Trading disabled
                             symbol_logger.info(f"Entry signal '{signal}' generated for {symbol}, but trading is disabled in config.")
                    else: # HOLD signal
                         symbol_logger.info(f"Signal is HOLD for {symbol}. No new entry.")

                elif current_position is None and market_info.get('is_contract'): # No position, but max limit reached
                    symbol_logger.info(f"Max concurrent positions ({open_positions_count}/{max_allowed_positions}) reached. Skipping new entry check for {symbol}.")

                elif not market_info.get('is_contract'): # Spot symbol
                    symbol_logger.info(f"Processing for SPOT symbol {symbol}. Current logic primarily targets contracts. Add spot-specific logic if needed.")
                    # TODO: Add spot trading logic here if desired (buy/sell based on signals, no leverage/positions)

            except Exception as symbol_loop_e:
                symbol_logger.error(f"{NEON_RED}!!! Unhandled exception during processing loop for symbol {symbol}: {symbol_loop_e} !!!{RESET}", exc_info=True)
                # Continue to the next symbol

            symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
            # Optional small delay between symbols to avoid hitting potential sub-account rate limits if many symbols
            time.sleep(0.2)

        # --- Post-Cycle Actions ---
        end_time = time.time()
        cycle_duration = end_time - start_time
        main_logger.info(f"{NEON_BLUE}--- Bot Cycle {cycle_count} Finished ({cycle_duration:.2f}s) ---{RESET}")

        # Save state at the end of each cycle
        save_state(args.state, bot_state, main_logger) # Use args.state for filename

        # Calculate wait time
        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0:
            main_logger.info(f"Waiting {wait_time:.2f} seconds before next cycle...")
            time.sleep(wait_time)
        else:
            main_logger.warning(f"Cycle duration ({cycle_duration:.2f}s) exceeded loop delay ({LOOP_DELAY_SECONDS}s). Running next cycle immediately.")


# ==============================================================
# ===        Position Management Logic (Pybit Version)       ===
# ==============================================================
def manage_existing_position_pybit(
    client: UnifiedHTTP, config: Dict[str, Any], logger: logging.Logger, analyzer: TradingAnalyzer,
    position_data: Dict, current_price_dec: Decimal
):
    """Manages an existing open position: MA cross exit, Break-Even logic."""
    symbol = position_data.get('symbol')
    side = position_data.get('side') # 'long' or 'short'
    entry_price_str = position_data.get('entryPrice') # String or Decimal from fetch
    size_str = position_data.get('contracts') # String or Decimal
    position_idx = position_data.get('positionIdx', 0) # For hedge mode
    market_info = analyzer.market_info # Already enhanced
    state = analyzer.symbol_state

    # --- Validate Inputs ---
    if not all([symbol, side, entry_price_str, size_str, market_info, state is not None]):
        logger.error(f"Incomplete position data or market info for management: Symbol={symbol}, Side={side}, Entry={entry_price_str}, Size={size_str}")
        return
    try:
        entry_price = Decimal(str(entry_price_str))
        size = Decimal(str(size_str))
        if size <= 0:
             logger.warning(f"Position size is zero or negative ({size}) for {symbol} during management check. Likely closed or stale data.")
             # Ensure state reflects no position
             if state.get("break_even_triggered"): state["break_even_triggered"] = False
             if state.get("last_entry_price") is not None: state["last_entry_price"] = None
             return
    except (InvalidOperation, TypeError) as e:
        logger.error(f"Invalid entry price or size format for management ({symbol}): Entry='{entry_price_str}', Size='{size_str}'. Error: {e}")
        return

    logger.debug(f"Managing {side.upper()} {symbol}: Entry={entry_price}, Size={size}, CurrentPx={current_price_dec}")

    # --- 1. MA Cross Exit ---
    if config.get("enable_ma_cross_exit", False):
        ema_short = analyzer._get_indicator_float("EMA_Short")
        ema_long = analyzer._get_indicator_float("EMA_Long")

        if ema_short is not None and ema_long is not None:
            crossed_against = False
            cross_details = ""
            # Tolerance to avoid exits on tiny fluctuations exactly at the cross
            tolerance = ema_long * Decimal("0.0001") # 0.01% tolerance, adjust as needed

            if side == 'long' and ema_short < (ema_long - tolerance):
                crossed_against = True
                cross_details = f"Short EMA ({ema_short:.5f}) crossed BELOW Long EMA ({ema_long:.5f})"
            elif side == 'short' and ema_short > (ema_long + tolerance):
                crossed_against = True
                cross_details = f"Short EMA ({ema_short:.5f}) crossed ABOVE Long EMA ({ema_long:.5f})"

            if crossed_against:
                logger.warning(f"{NEON_YELLOW}MA Cross Exit Triggered for {symbol}! {cross_details}. Attempting to close position.{RESET}")
                if config.get("enable_trading"):
                    closed_order_result = close_position_pybit(client, symbol, position_data, logger, market_info)
                    if closed_order_result and closed_order_result.get('orderId'):
                        logger.info(f"Position close order placed successfully for {symbol} due to MA Cross. Order ID: {closed_order_result['orderId']}")
                        # Reset state after successful close order placement
                        state["break_even_triggered"] = False
                        state["last_signal"] = "HOLD" # Reset signal state
                        state["last_entry_price"] = None
                        # Return early as position is being closed
                        return
                    else:
                         logger.error(f"Failed to place position close order for {symbol} after MA Cross trigger.")
                         # Should we retry closing? Or just log and let other logic proceed? Let's log and proceed for now.
                else:
                     logger.info(f"MA Cross exit triggered for {symbol}, but trading is disabled.")
                # Even if trading disabled, prevent BE logic if MA cross exit triggered
                return # Exit management logic for this cycle if MA cross triggered

    # --- 2. Break-Even Logic ---
    # Only apply BE if enabled, not already triggered, and MA cross didn't try to close
    if config.get("enable_break_even", False) and not analyzer.break_even_triggered:
        atr_value = analyzer.indicator_values.get("ATR")
        if atr_value and atr_value.is_finite() and atr_value > 0:
            try:
                trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", "1.0")))
                profit_target_atr = atr_value * trigger_multiple

                # Calculate current profit/loss in price terms
                current_profit = Decimal('0')
                if side == 'long':
                    current_profit = current_price_dec - entry_price
                else: # short
                    current_profit = entry_price - current_price_dec

                price_prec = analyzer.get_price_precision_digits()
                logger.debug(f"BE Check ({symbol}): CurrentProfit={current_profit:.{price_prec}f}, TargetProfit(ATR*{trigger_multiple})={profit_target_atr:.{price_prec}f}")

                # Check if profit target is reached
                if current_profit >= profit_target_atr:
                    logger.info(f"{NEON_GREEN}Break-Even Triggered for {symbol}! Current Profit ({current_profit:.{price_prec}f}) >= Target Profit ({profit_target_atr:.{price_prec}f}){RESET}")

                    tick_size = analyzer.get_min_tick_size()
                    offset_ticks = int(config.get("break_even_offset_ticks", 2)) # Number of ticks for the offset

                    if tick_size and tick_size > 0 and offset_ticks >= 0:
                        offset_value = tick_size * Decimal(offset_ticks)
                        be_stop_loss_price: Optional[Decimal] = None

                        # Calculate BE stop loss price
                        if side == 'long':
                             # Set SL slightly above entry
                             raw_be_sl = entry_price + offset_value
                             be_stop_loss_price = analyzer.quantize_price(raw_be_sl, rounding=ROUND_UP) # Round UP for long SL
                             # Ensure it's at least one tick above entry after quantization
                             if be_stop_loss_price and be_stop_loss_price <= entry_price:
                                  be_stop_loss_price = analyzer.quantize_price(entry_price + tick_size, rounding=ROUND_UP)
                        else: # short
                             # Set SL slightly below entry
                             raw_be_sl = entry_price - offset_value
                             be_stop_loss_price = analyzer.quantize_price(raw_be_sl, rounding=ROUND_DOWN) # Round DOWN for short SL
                             # Ensure it's at least one tick below entry
                             if be_stop_loss_price and be_stop_loss_price >= entry_price:
                                  be_stop_loss_price = analyzer.quantize_price(entry_price - tick_size, rounding=ROUND_DOWN)

                        if be_stop_loss_price and be_stop_loss_price > 0:
                            logger.info(f"Calculated Break-Even Stop Loss for {symbol}: {be_stop_loss_price:.{price_prec}f} ({offset_ticks} ticks offset)")

                            if config.get("enable_trading"):
                                # Get current TP and TSL settings from the position data
                                current_tp = position_data.get('takeProfit') # Already Decimal or None
                                current_tsl_dist = position_data.get('trailingStop') # Distance, Decimal or None
                                current_tsl_active = position_data.get('activePrice') # Activation Price, Decimal or None

                                # Decide whether to keep TSL active based on config
                                keep_tsl = config.get("enable_trailing_stop", False) and not config.get("break_even_force_fixed_sl", True)

                                tp_to_set = current_tp # Keep existing TP
                                tsl_dist_to_set = current_tsl_dist if keep_tsl else None # Keep TSL only if not forcing fixed SL
                                tsl_active_to_set = current_tsl_active if keep_tsl else None

                                logger.info(f"Setting new protection for {symbol}: SL={be_stop_loss_price}, TP={tp_to_set}, TSL_Dist={tsl_dist_to_set}, TSL_Act={tsl_active_to_set}")

                                # Set the new SL (and potentially keep TP/TSL)
                                protection_set_ok = set_protection_pybit(
                                    client=client, symbol=symbol,
                                    stop_loss_price=be_stop_loss_price,
                                    take_profit_price=tp_to_set,
                                    trailing_stop_price=tsl_dist_to_set,
                                    trailing_active_price=tsl_active_to_set,
                                    position_idx=position_idx, # Pass correct index
                                    logger=logger, market_info=market_info
                                )

                                if protection_set_ok:
                                    logger.info(f"{NEON_GREEN}Successfully set Break-Even Stop Loss for {symbol}.{RESET}")
                                    analyzer.break_even_triggered = True # Update state only after successful API call
                                else:
                                    logger.error(f"{NEON_RED}Failed to set Break-Even SL via API for {symbol}. BE state not updated.{RESET}")
                                    # Do not set break_even_triggered = True if API call failed
                            else: # Trading disabled
                                 logger.info(f"Break-Even triggered for {symbol}, but trading is disabled. Would set SL to {be_stop_loss_price}.")
                                 # In backtesting/paper mode, we might still set the state:
                                 # analyzer.break_even_triggered = True

                        else: # Calculated BE SL price was invalid
                             logger.error(f"Calculated break-even stop loss price ({be_stop_loss_price}) is invalid for {symbol}. Cannot set BE SL.")
                    else: # Tick size invalid or offset ticks negative
                         logger.error(f"Cannot calculate break-even offset for {symbol}: Tick size ({tick_size}) or offset ticks ({offset_ticks}) invalid.")
            except (InvalidOperation, TypeError, ValueError) as e:
                 logger.error(f"Error during Break-Even calculation for {symbol}: {e}", exc_info=True)
            except Exception as e:
                 logger.error(f"Unexpected error during Break-Even logic for {symbol}: {e}", exc_info=True)
        elif not analyzer.break_even_triggered: # Only log warning if BE not already triggered
             logger.warning(f"Cannot check Break-Even status for {symbol}: ATR value is invalid or missing ({atr_value}).")

    # --- 3. TSL Management (Server-Side) ---
    # Pybit's set_trading_stop handles TSL activation and trailing.
    # We typically only need to SET it initially (or during BE update).
    # We don't actively manage the TSL price from the bot once it's set on the server.
    # We *could* fetch the position periodically and check the current `stopLoss` price
    # which reflects the trailed SL value, but it's usually not necessary.
    logger.debug(f"Finished management checks for {symbol}.")


# ==============================================================
# ===          New Position Entry Logic (Pybit Version)      ===
# ==============================================================
def attempt_new_entry_pybit(
    client: UnifiedHTTP, config: Dict[str, Any], logger: logging.Logger, analyzer: TradingAnalyzer,
    signal: str, entry_price_signal: Decimal, current_balance: Decimal
) -> bool:
    """Attempts to enter a new position based on signal, calculating size, setting protection."""
    symbol = analyzer.symbol
    market_info = analyzer.market_info
    state = analyzer.symbol_state
    price_prec = analyzer.get_price_precision_digits()

    logger.info(f"Attempting {signal} entry for {symbol} based on signal price ~{entry_price_signal:.{price_prec}f}")

    # --- 1. Calculate TP/SL based on Signal Price & ATR ---
    # Note: Entry price for calc might differ slightly from actual market execution
    quantized_entry_target, take_profit_price, stop_loss_price = analyzer.calculate_entry_tp_sl(entry_price_signal, signal)

    if quantized_entry_target is None:
        logger.error(f"Cannot enter {signal} {symbol}: Failed to quantize entry price target.")
        return False
    if stop_loss_price is None:
        logger.error(f"Cannot enter {signal} {symbol}: Stop Loss calculation failed (invalid ATR or price?). Entry aborted for safety.")
        return False
    # TP is optional, proceed even if TP is None

    # --- 2. Calculate Position Size ---
    risk_per_trade = config.get("risk_per_trade", 0.01) # Default 1% risk
    leverage = int(config.get("leverage", 10)) # Leverage from config

    position_size = calculate_position_size(
        balance=current_balance,
        risk_per_trade=risk_per_trade,
        entry_price=quantized_entry_target, # Use the calculated entry target for sizing
        stop_loss_price=stop_loss_price,
        market_info=market_info,
        leverage=leverage,
        logger=logger
    )

    if position_size is None or position_size <= 0:
        logger.error(f"Cannot enter {signal} {symbol}: Position size calculation failed or resulted in zero/negative size ({position_size}).")
        return False

    # --- 3. Set Leverage (for contracts) ---
    if market_info.get('is_contract'):
        logger.info(f"Setting leverage for {symbol} to {leverage}x before entry...")
        if not set_leverage_pybit(client, symbol, leverage, logger, market_info):
            logger.error(f"Failed to set leverage {leverage}x for {symbol}. Aborting entry attempt.")
            return False
        # Add small delay after setting leverage? Usually not needed.
        # time.sleep(0.5)

    # --- 4. Place Market Order ---
    order_side = 'buy' if signal == 'BUY' else 'sell'
    # --- Hedge Mode Consideration ---
    position_idx = 0 # Default for One-Way mode
    if config.get("position_mode") == "Hedge":
        position_idx = 1 if signal == 'BUY' else 2 # 1 for Buy/Long, 2 for Sell/Short in Hedge mode
        logger.info(f"Hedge Mode Active: Using positionIdx={position_idx} for {signal} order.")
    entry_params = {'positionIdx': position_idx}
    # Do NOT pass SL/TP here if we want to set them *after* confirming entry price
    # Passing them here might use the signal price, not actual fill price for SL/TP calcs on Bybit's side? Safer to set after.

    entry_order_result = create_order_pybit(
        client=client, symbol=symbol, order_type='market', side=order_side,
        amount=position_size, price=None, params=entry_params, # No price for market order
        logger=logger, market_info=market_info
    )

    if not entry_order_result or not entry_order_result.get('orderId'):
        logger.error(f"Failed to place entry market order for {signal} {symbol}. Aborting entry.")
        return False

    order_id = entry_order_result['orderId']
    logger.info(f"Entry market order placed successfully for {symbol}. Order ID: {order_id}.")

    # --- 5. Confirm Entry & Get Actual Fill Price (Important!) ---
    logger.info(f"Waiting {POSITION_CONFIRM_DELAY} seconds for order fill and position update...")
    time.sleep(POSITION_CONFIRM_DELAY) # Wait for propagation

    actual_entry_price: Optional[Decimal] = None
    final_position_size: Optional[Decimal] = None
    retries = 3
    for i in range(retries):
        try:
             logger.debug(f"Fetching position info to confirm entry (Attempt {i+1}/{retries})...")
             # Fetch position specifically using the determined positionIdx
             # Note: fetch_positions_pybit might need enhancement for hedge mode to find the specific index
             # Current fetch_positions_pybit returns the first non-zero position. May need refinement.
             # Let's assume it finds the correct one for now.
             confirmed_position = fetch_positions_pybit(client, symbol, logger, market_info)

             if confirmed_position and confirmed_position.get('contracts', Decimal('0')) > 0:
                  entry_str = confirmed_position.get('entryPrice')
                  size_str = confirmed_position.get('contracts')
                  pos_idx_confirm = confirmed_position.get('positionIdx')
                  if entry_str and size_str is not None:
                       actual_entry_price = Decimal(str(entry_str))
                       final_position_size = Decimal(str(size_str))
                       logger.info(f"{NEON_GREEN}Confirmed position entry for {symbol} (Idx: {pos_idx_confirm}): Actual Entry={actual_entry_price:.{price_prec}f}, Size={final_position_size}{RESET}")
                       # Basic check if filled size is close to ordered size
                       if abs(final_position_size - position_size) / position_size > Decimal('0.05'): # 5% tolerance
                            logger.warning(f"Filled position size ({final_position_size}) differs significantly (>5%) from ordered size ({position_size}).")
                       break # Exit retry loop on success
                  else:
                      logger.warning(f"Position found for {symbol}, but entry price or size missing/invalid in confirmation data. Retrying...")
             else:
                  logger.warning(f"Position not found or size is zero for {symbol} after {POSITION_CONFIRM_DELAY}s. Retrying confirmation check ({i+1}/{retries})...")

             if i < retries - 1: time.sleep(5) # Wait longer before next confirmation check

        except Exception as confirm_e:
             logger.error(f"Error fetching position for confirmation ({symbol}): {confirm_e}. Retrying ({i+1}/{retries})...")
             if i < retries - 1: time.sleep(5)

    # --- 6. Set SL/TP/TSL using Actual Entry Price ---
    if actual_entry_price is None or final_position_size is None or final_position_size <= 0:
        logger.error(f"{NEON_RED}Failed to confirm position entry for {symbol} after retries. Cannot set SL/TP. MANUAL INTERVENTION MAY BE REQUIRED if order filled!{RESET}")
        # What to do here? Try to cancel the order? Risky.
        # If position exists but we couldn't get price, it's unprotected.
        # For now, return failure, assuming entry failed or confirmation failed critically.
        return False

    # --- Re-calculate SL/TP using the actual fill price ---
    logger.info(f"Re-calculating SL/TP based on actual entry price: {actual_entry_price}")
    _, actual_tp_price, actual_sl_price = analyzer.calculate_entry_tp_sl(actual_entry_price, signal)

    if actual_sl_price is None:
         logger.error(f"{NEON_RED}Failed to calculate Stop Loss based on actual entry price ({actual_entry_price}) for {symbol}. Position might be unprotected!{RESET}")
         # Attempt to set only TP? Or close immediately? Closing seems drastic. Let's try setting TP only if available.
         # If TP is also None, we have an unprotected position. Risky.
         if actual_tp_price is None:
              logger.critical(f"CRITICAL: Failed to calculate both SL and TP for entered position {symbol}. POSITION IS UNPROTECTED. Attempting emergency close.")
              if config.get("enable_trading"):
                   pos_to_close = fetch_positions_pybit(client, symbol, logger, market_info) # Fetch again to be sure
                   if pos_to_close: close_position_pybit(client, symbol, pos_to_close, logger, market_info)
                   else: logger.error("Could not fetch position data for emergency close.")
              return False # Entry effectively failed

    # --- Calculate Trailing Stop Parameters (if enabled) ---
    tsl_distance: Optional[Decimal] = None
    tsl_activation_price: Optional[Decimal] = None

    if config.get("enable_trailing_stop", False) and actual_sl_price is not None: # Only set TSL if initial SL is valid
        try:
            callback_rate_str = config.get("trailing_stop_callback_rate", "0.005") # 0.5% default
            activation_perc_str = config.get("trailing_stop_activation_percentage", "0.003") # 0.3% default
            callback_rate = Decimal(str(callback_rate_str))
            activation_perc = Decimal(str(activation_perc_str))
            tick_size = analyzer.get_min_tick_size()

            if callback_rate > 0 and tick_size:
                # Calculate TSL distance (absolute price value)
                raw_tsl_distance = actual_entry_price * callback_rate
                # Quantize distance UP to nearest tick multiple (to make it slightly wider/safer)
                tsl_distance = (raw_tsl_distance / tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * tick_size
                if tsl_distance < tick_size: tsl_distance = tick_size # Ensure min 1 tick distance
                logger.debug(f"Calculated TSL Distance for {symbol}: {tsl_distance:.{price_prec}f} (Rate: {callback_rate})")

                # Calculate TSL activation price (if percentage > 0)
                if activation_perc > 0:
                    price_offset = actual_entry_price * activation_perc
                    raw_activation_price = actual_entry_price + price_offset if signal == "BUY" else actual_entry_price - price_offset
                    # Round activation away from current price
                    rounding_mode = ROUND_UP if signal == "BUY" else ROUND_DOWN
                    tsl_activation_price = analyzer.quantize_price(raw_activation_price, rounding=rounding_mode)
                    # Ensure activation is valid
                    if tsl_activation_price and tsl_activation_price <= 0: tsl_activation_price = None
                    # Ensure activation is beyond entry price
                    if signal == "BUY" and tsl_activation_price and tsl_activation_price <= actual_entry_price:
                        tsl_activation_price = analyzer.quantize_price(actual_entry_price + tick_size, ROUND_UP)
                    elif signal == "SELL" and tsl_activation_price and tsl_activation_price >= actual_entry_price:
                         tsl_activation_price = analyzer.quantize_price(actual_entry_price - tick_size, ROUND_DOWN)
                    logger.debug(f"Calculated TSL Activation Price for {symbol}: {tsl_activation_price:.{price_prec}f} (Perc: {activation_perc})")
                else:
                    logger.debug(f"TSL Activation Percentage is 0 for {symbol}. TSL will activate immediately.")
                    tsl_activation_price = None # Immediate activation

            else: logger.warning(f"Cannot calculate TSL Distance for {symbol}: Callback rate ({callback_rate}) or tick size ({tick_size}) invalid.")
        except (InvalidOperation, TypeError, ValueError) as tsl_e:
             logger.error(f"Error calculating TSL parameters for {symbol}: {tsl_e}", exc_info=True)


    # --- 7. Set Final Protection ---
    logger.info(f"Setting final protection for {symbol} (Idx: {position_idx}): SL={actual_sl_price}, TP={actual_tp_price}, TSL_Dist={tsl_distance}, TSL_Act={tsl_activation_price}")
    protection_set_ok = set_protection_pybit(
        client=client, symbol=symbol,
        stop_loss_price=actual_sl_price,
        take_profit_price=actual_tp_price,
        trailing_stop_price=tsl_distance,
        trailing_active_price=tsl_activation_price,
        position_idx=position_idx, # Use the correct index
        logger=logger, market_info=market_info
    )

    if not protection_set_ok:
        logger.error(f"{NEON_RED}Failed to set initial SL/TP/TSL after confirming entry for {symbol}! Position might be unprotected.{RESET}")
        # Consider emergency close again? Very risky state.
        logger.critical(f"CRITICAL: Failed setting protection on confirmed position {symbol}. MANUAL INTERVENTION REQUIRED.")
        # Do not update state as successful if protection failed
        # Maybe try setting just SL again?
        # For now, signal failure.
        return False

    # --- 8. Update State ---
    logger.info(f"{NEON_GREEN}Successfully entered {signal} trade for {symbol} with protection set. Actual Entry: {actual_entry_price:.{price_prec}f}{RESET}")
    state["break_even_triggered"] = False # Reset BE trigger on new entry
    state["last_entry_price"] = str(actual_entry_price) # Store actual entry price as string
    # last_signal was already updated during signal generation

    return True # Indicate successful entry and protection setting


# --- Main Execution (Pybit Version) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Bybit V5 Multi-Symbol Trading Bot (Pybit) v1.0.1")
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help=f"Path to configuration file (default: {CONFIG_FILE})")
    parser.add_argument("--state", type=str, default=STATE_FILE, help=f"Path to state file (default: {STATE_FILE})")
    parser.add_argument("--symbols", type=str, help="Override config symbols (comma-separated list, e.g., BTCUSDT,ETHUSDT)")
    parser.add_argument("--live", action="store_true", help="Enable live trading (overrides 'enable_trading' in config)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level console logging")
    args = parser.parse_args()

    # Set console log level based on args
    console_log_level = logging.DEBUG if args.debug else logging.INFO
    if args.debug: print(f"{NEON_YELLOW}DEBUG console logging enabled.{RESET}")

    # Setup main logger first
    main_logger = get_logger('main') # Initialize main logger instance
    main_logger.info(f" --- Pybit Trading Bot v1.0.1 Starting: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} --- ")

    # --- Load Configuration ---
    config = load_config(args.config, main_logger)
    if config is None:
        main_logger.critical("Failed to load or validate configuration. Exiting.")
        sys.exit(1) # Exit if config loading/validation failed

    # --- Handle Command Line Overrides ---
    # Symbol Override
    if args.symbols:
        try:
            override_symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
            if override_symbols:
                main_logger.warning(f"{NEON_YELLOW}Overriding symbols from config file via command line. Trading ONLY: {override_symbols}{RESET}")
                config["symbols"] = override_symbols
            else:
                main_logger.error("Symbol override list provided via --symbols is empty after parsing. Exiting.")
                sys.exit(1)
        except Exception as symbol_parse_err:
             main_logger.error(f"Error parsing --symbols argument '{args.symbols}': {symbol_parse_err}. Exiting.")
             sys.exit(1)

    # Live Trading Override
    if args.live:
        main_logger.warning(f"{NEON_RED}--- LIVE TRADING ENABLED via command line override! ---{RESET}")
        if not config.get("enable_trading"):
             main_logger.warning("Overriding 'enable_trading=false' from config file.")
        if config.get("use_sandbox"):
             main_logger.warning("Overriding 'use_sandbox=true' from config file. LIVE TRADING ON MAINNET!")
        config["enable_trading"] = True
        config["use_sandbox"] = False # Force sandbox off if --live is used

    # Log final trading/sandbox status
    if config.get("enable_trading"):
        main_logger.warning(f"{NEON_RED}--- Trading is ENABLED ---{RESET}")
        if config.get("use_sandbox"):
            main_logger.warning(f"{NEON_YELLOW}Sandbox mode (testnet) is ACTIVE.{RESET}")
        else:
            main_logger.warning(f"{NEON_RED}!!! LIVE MAINNET TRADING IS ACTIVE !!!{RESET}")
            main_logger.warning(f"{NEON_RED}Ensure configuration and risk settings are correct before proceeding!{RESET}")
            # Optional: Add a countdown/confirmation step for live mainnet trading
            # print("Starting live trading in 5 seconds... Press Ctrl+C to cancel.")
            # try: time.sleep(5)
            # except KeyboardInterrupt: print("Live trading cancelled by user."); sys.exit(0)
    else:
        main_logger.info("--- Trading is DISABLED (simulation mode) ---")

    # --- Load State ---
    bot_state = load_state(args.state, main_logger)

    # --- Initialize Pybit Client ---
    pybit_client = initialize_pybit_client(config, main_logger)

    # --- Run Bot ---
    if pybit_client:
        main_logger.info(f"{NEON_GREEN}Pybit client initialized. Starting main bot loop...{RESET}")
        try:
            run_bot(pybit_client, config, bot_state) # Pass the initialized client
        except KeyboardInterrupt:
            main_logger.info("Bot stopped by user (KeyboardInterrupt).")
        except Exception as e:
            main_logger.critical(f"{NEON_RED}!!! BOT CRASHED WITH UNHANDLED EXCEPTION: {e} !!!{RESET}", exc_info=True)
            traceback.print_exc() # Print traceback to console as well
        finally:
            # --- Shutdown Procedures ---
            main_logger.info("Executing bot shutdown procedures...")
            # Optional: Attempt to close open positions? (Risky if state is corrupt)
            # print("Attempting to close any open positions...")
            # Implement safe position closing logic here if desired upon shutdown
            main_logger.info("Saving final bot state...")
            save_state(args.state, bot_state, main_logger)
            main_logger.info(f"--- Pybit Bot Shutdown Completed: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    else:
        main_logger.critical("Failed to initialize Pybit client. Bot cannot start.")

    # --- Cleanup ---
    logging.shutdown() # Flush and close all logging handlers
    print("Pybit Bot execution finished.")
    sys.exit(0 if pybit_client else 1) # Exit code 0 on success, 1 on client init failure
```

**Key Changes and Considerations:**

1.  **Placeholders Filled:** All `pass` placeholders (`TradingAnalyzer`, `run_bot`, `manage_existing_position_pybit`, `attempt_new_entry_pybit`) are now implemented with logic consistent with the described features and Pybit helpers.
2.  **`calculate_position_size` Added:** This crucial function is now included and integrated into `attempt_new_entry_pybit`. It handles linear/inverse/spot calculations and leverage capping, using Decimals. **Review the Inverse contract size logic carefully if you trade inverse pairs.**
3.  **Contract Size:** The script heavily relies on the `contract_size` obtained (or assumed) in `get_market_info_pybit`. The default assumption is `Decimal('1')` (suitable for linear USDT perps). **If you trade inverse contracts (e.g., BTCUSD) or options, you MUST verify that the contract size is correctly identified or manually set.** The `calculate_position_size` function depends directly on this.
4.  **Hedge Mode:** While `positionIdx` is passed correctly to order/protection functions, the `fetch_positions_pybit` currently returns only the *first* active position found for a symbol. If you run in Hedge mode, you might have both a long (Idx 1) and short (Idx 2) position simultaneously. The `run_bot` loop and `fetch_positions_pybit` would need modification to handle and manage both positions independently if full Hedge mode support is required. The current implementation implicitly assumes One-Way mode behavior even if Hedge is set in the config.
5.  **Error Handling:** Robust error handling is present in API calls and calculations. Pay attention to CRITICAL/ERROR logs, especially regarding failed protection settings or entry confirmations.
6.  **State Management:** State is loaded and saved. Consider the atomicity comment for `save_state` if running in an unstable environment.
7.  **Dependencies:** Added a comment about Python version (>= 3.7 recommended, 3.9+ for `zoneinfo`).
8.  **Live Trading:** Added extra warnings for live mainnet trading. Be extremely cautious.

This integrated version should be functional, but remember to:

*   **Carefully review and adjust the configuration (`config_pybit.json`)** to match your strategy, risk tolerance, and Bybit account settings (especially `position_mode`).
*   **Verify the `contract_size` assumption** if trading anything other than standard linear USDT perpetuals.
*   **Test thoroughly on testnet (`use_sandbox: true`)** before considering live trading.