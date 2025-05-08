# -*- coding: utf-8 -*-
"""
Enhanced Multi-Symbol Trading Bot for Bybit (V5 API) - v1.0.2

Merges features, optimizations, and best practices from previous versions.
Includes: pandas_ta.Strategy, Decimal precision, robust CCXT interaction,
          multi-symbol support, state management, TSL/BE logic, MA cross exit.

Changes in v1.0.2:
- Added more explicit handling/logging for Unified Trading Accounts (UTA).
- Enhanced configuration validation on load (interval, numeric types).
- Improved robustness in fetch_balance parsing for UTA.
- Ensured consistent use of Decimal/float where appropriate.
- Added checks for SL/TP being too close to entry price.
- Added comments regarding Hedge Mode implementation points.
- Standardized logging messages and added library dependency comments.
- Refined state management for last_entry_price (stored as string).
- Minor code clarity improvements and type hinting additions.
"""

# --- Required Libraries ---
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

try:
    from zoneinfo import ZoneInfo # Preferred (Python 3.9+)
except ImportError:
    # Fallback for older Python (pip install pytz)
    try:
        from pytz import timezone as ZoneInfo
    except ImportError:
        print("Error: 'zoneinfo' (Python 3.9+) or 'pytz' package required for timezone handling.")
        print("Please install pytz: pip install pytz")
        sys.exit(1)

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from colorama import Fore, Style, init
except ImportError:
    print("Warning: 'colorama' package not found. Colored output will be disabled.")
    print("Install it with: pip install colorama")
    # Define dummy color variables if colorama is missing
    class DummyColor:
        def __getattr__(self, name): return ""
    Fore = DummyColor(); Style = DummyColor()
    def init(*args, **kwargs): pass # Dummy init function

from dotenv import load_dotenv

# --- Initialization ---
try:
    getcontext().prec = 36 # Set Decimal precision
except Exception as e:
    print(f"Warning: Could not set Decimal precision: {e}. Using default.")
init(autoreset=True)    # Initialize colorama (or dummy init)
load_dotenv()           # Load environment variables from .env file

# --- Neon Color Scheme ---
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
    print(f"{NEON_RED}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file{RESET}")
    sys.exit(1)

# --- Configuration File and Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
STATE_FILE = "bot_state.json"

os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Timezone for logging and display
try:
    TZ_NAME = os.getenv("BOT_TIMEZONE", "America/Chicago")
    TIMEZONE = ZoneInfo(TZ_NAME)
    print(f"Using Timezone: {TZ_NAME}")
except Exception as tz_err:
    print(f"{NEON_YELLOW}Warning: Could not load timezone '{TZ_NAME}': {tz_err}. Defaulting to UTC.{RESET}")
    TIMEZONE = ZoneInfo("UTC")

# --- API Interaction Constants ---
MAX_API_RETRIES = 4
RETRY_DELAY_SECONDS = 5
RATE_LIMIT_BUFFER_SECONDS = 0.5
MARKET_RELOAD_INTERVAL_SECONDS = 3600 # Reload markets every hour
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status
MIN_TICKS_AWAY_FOR_SLTP = 3 # Minimum number of ticks SL/TP should be away from entry

# --- Bot Logic Constants ---
# Intervals supported by the bot's internal logic (ensure config matches)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
# Map bot intervals to ccxt's expected timeframe format
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}

# Default Indicator/Strategy Parameters (can be overridden by config.json)
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
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

DEFAULT_LOOP_DELAY_SECONDS = 15

# --- Global Variables ---
loggers: Dict[str, logging.Logger] = {}
console_log_level = logging.INFO
QUOTE_CURRENCY = "USDT"
LOOP_DELAY_SECONDS = DEFAULT_LOOP_DELAY_SECONDS
IS_UNIFIED_ACCOUNT = False # Flag to indicate if the account is UTA

# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information."""
    REDACTED_STR = "***REDACTED***"
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if API_KEY and len(API_KEY) > 4: formatted = formatted.replace(API_KEY, self.REDACTED_STR)
        if API_SECRET and len(API_SECRET) > 4: formatted = formatted.replace(API_SECRET, self.REDACTED_STR)
        return formatted

class LocalTimeFormatter(SensitiveFormatter):
    """Formatter that uses the configured local timezone for console output."""
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
        return dt.timetuple()
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=TIMEZONE)
        if datefmt: s = dt.strftime(datefmt)
        else: s = dt.strftime("%Y-%m-%d %H:%M:%S"); s = f"{s},{int(record.msecs):03d}"
        return s

def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up a logger with rotating file and timezone-aware console handlers."""
    global console_log_level
    logger_instance_name = f"livebot_{name.replace('/', '_').replace(':', '-')}" if is_symbol_logger else f"livebot_{name}"

    if logger_instance_name in loggers:
        logger = loggers[logger_instance_name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level != console_log_level:
                handler.setLevel(console_log_level)
        return logger

    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)
    logger.setLevel(logging.DEBUG) # Capture all levels at logger

    try: # File Handler (UTC)
        file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_formatter = SensitiveFormatter("%(asctime)s.%(msecs)03d UTC - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_formatter.converter = time.gmtime
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e: print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    try: # Stream Handler (Local Time)
        stream_handler = logging.StreamHandler(sys.stdout)
        tz_name_str = TIMEZONE.tzname(datetime.now(TIMEZONE)) if hasattr(TIMEZONE, 'tzname') else str(TIMEZONE)
        stream_formatter = LocalTimeFormatter(f"{NEON_BLUE}%(asctime)s{RESET} [{tz_name_str}] - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3])
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(console_log_level)
        logger.addHandler(stream_handler)
    except Exception as e: print(f"{NEON_RED}Error setting up stream logger for {name}: {e}{RESET}")

    logger.propagate = False
    loggers[logger_instance_name] = logger
    logger.info(f"Logger '{logger_instance_name}' initialized. File: '{os.path.basename(log_filename)}', Console Level: {logging.getLevelName(console_log_level)}")
    return logger

def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance."""
    return setup_logger(name, is_symbol_logger)

# --- Configuration Management ---
def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    keys_added_or_type_mismatch = False
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
            keys_added_or_type_mismatch = True
            print(f"{NEON_YELLOW}Config Warning: Missing key '{key}'. Added default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            nested_updated_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_value)
            if nested_keys_added:
                updated_config[key] = nested_updated_config
                keys_added_or_type_mismatch = True
        elif updated_config.get(key) is not None and type(default_value) != type(updated_config.get(key)):
            is_promoting_num = (isinstance(default_value, (float, Decimal)) and isinstance(updated_config.get(key), int))
            if not is_promoting_num:
                print(f"{NEON_YELLOW}Config Warning: Type mismatch for key '{key}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Using loaded value: {repr(updated_config.get(key))}.{RESET}")
    return updated_config, keys_added_or_type_mismatch

def _validate_config_values(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validates specific configuration values."""
    is_valid = True
    # Validate interval
    interval = config.get("interval")
    if interval not in CCXT_INTERVAL_MAP:
        logger.error(f"Config Error: Invalid 'interval' value '{interval}'. Must be one of {VALID_INTERVALS}")
        is_valid = False

    # Validate numeric types and ranges
    numeric_params = {
        "loop_delay": (int, 5, 3600), # Min 5 sec delay
        "risk_per_trade": (float, 0.0001, 0.5), # Risk 0.01% to 50%
        "leverage": (int, 1, 125), # Practical leverage limits
        "max_concurrent_positions_total": (int, 1, 100),
        "atr_period": (int, 2, 500), "ema_short_period": (int, 2, 500),
        "ema_long_period": (int, 3, 1000), "rsi_period": (int, 2, 500),
        # ... add other numeric params ...
        "stop_loss_multiple": (float, 0.1, 10.0),
        "take_profit_multiple": (float, 0.1, 20.0),
        "trailing_stop_callback_rate": (float, 0.0001, 0.5), # 0.01% to 50%
        "trailing_stop_activation_percentage": (float, 0.0, 0.5), # 0% to 50%
        "break_even_trigger_atr_multiple": (float, 0.1, 10.0),
        "break_even_offset_ticks": (int, 0, 100),
    }
    for key, (expected_type, min_val, max_val) in numeric_params.items():
        value = config.get(key)
        if value is None: continue # Skip if optional or handled by ensure_keys
        try:
            if expected_type == int: num_value = int(value)
            elif expected_type == float: num_value = float(value)
            else: num_value = value # Should not happen

            if not (min_val <= num_value <= max_val):
                logger.error(f"Config Error: '{key}' value {num_value} is outside the recommended range ({min_val} - {max_val}).")
                is_valid = False
            # Store the validated numeric type back
            config[key] = num_value
        except (ValueError, TypeError):
            logger.error(f"Config Error: '{key}' value '{value}' could not be converted to {expected_type.__name__}.")
            is_valid = False

    # Validate symbols list
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
         logger.error(f"Config Error: 'symbols' must be a non-empty list.")
         is_valid = False
    elif not all(isinstance(s, str) and '/' in s for s in symbols): # Basic format check
         logger.error(f"Config Error: 'symbols' list contains invalid formats. Expected 'BASE/QUOTE' or 'BASE/QUOTE:SETTLE'. Found: {symbols}")
         is_valid = False

    return is_valid

def load_config(filepath: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Load config, create default, ensure keys, validate values, save updates."""
    default_config = { # Keep defaults definition here
        "symbols": ["BTC/USDT:USDT"], "interval": "5", "loop_delay": DEFAULT_LOOP_DELAY_SECONDS,
        "quote_currency": "USDT", "enable_trading": False, "use_sandbox": True,
        "risk_per_trade": 0.01, "leverage": 10, "max_concurrent_positions_total": 1,
        "position_mode": "One-Way", "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD, "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW, "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV, "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW, "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
        "stoch_rsi_k": DEFAULT_K_WINDOW, "stoch_rsi_d": DEFAULT_D_WINDOW, "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF, "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD, "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW, "orderbook_limit": 25, "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25.0, "stoch_rsi_overbought_threshold": 75.0,
        "volume_confirmation_multiplier": 1.5, "scalping_signal_threshold": 2.5,
        "stop_loss_multiple": 1.8, "take_profit_multiple": 0.7, "enable_ma_cross_exit": True,
        "enable_trailing_stop": True, "trailing_stop_callback_rate": 0.005,
        "trailing_stop_activation_percentage": 0.003, "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0, "break_even_offset_ticks": 2,
        "break_even_force_fixed_sl": True,
        "indicators": {"ema_alignment": True, "momentum": True, "volume_confirmation": True, "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True, "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True, "orderbook": True, },
        "weight_sets": {"scalping": {"ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2, "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4, "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15, },"default": {"ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1, "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3, "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1, }},
        "active_weight_set": "default"
    }
    config_to_use = default_config
    keys_updated_in_file = False
    if not os.path.exists(filepath):
        print(f"{NEON_YELLOW}Config file not found at '{filepath}'. Creating default config...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f: json.dump(default_config, f, indent=4, sort_keys=True)
            print(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
        except IOError as e: print(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using in-memory defaults.{RESET}")
    else:
        try:
            with open(filepath, "r", encoding="utf-8") as f: config_from_file = json.load(f)
            updated_config_from_file, keys_added = _ensure_config_keys(config_from_file, default_config)
            config_to_use = updated_config_from_file
            if keys_added:
                keys_updated_in_file = True
                print(f"{NEON_YELLOW}Updating config file '{filepath}' with missing/changed default keys...{RESET}")
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write: json.dump(config_to_use, f_write, indent=4, sort_keys=True)
                    print(f"{NEON_GREEN}Config file updated successfully.{RESET}")
                except IOError as e: print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}"); keys_updated_in_file = False
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
            config_to_use = default_config
            try:
                with open(filepath, "w", encoding="utf-8") as f_recreate: json.dump(default_config, f_recreate, indent=4, sort_keys=True)
                print(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create: print(f"{NEON_RED}Error recreating default config file after load error: {e_create}{RESET}")
        except Exception as e:
            print(f"{NEON_RED}Unexpected error loading configuration: {e}. Using defaults.{RESET}")
            config_to_use = default_config

    # Validate the final configuration before returning
    if not _validate_config_values(config_to_use, logger):
        logger.critical("Configuration validation failed. Please check errors above and fix config.json. Exiting.")
        return None # Indicate failure

    logger.info("Configuration loaded and validated successfully.")
    return config_to_use

# --- State Management ---
def load_state(filepath: str, logger: logging.Logger) -> Dict[str, Any]:
    """Loads the bot's state from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"Loaded previous state from {filepath}")
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

def save_state(filepath: str, state: Dict[str, Any], logger: logging.Logger):
    """Saves the bot's state to a JSON file."""
    try:
        # Ensure Decimals are saved as strings
        state_to_save = json.loads(json.dumps(state, default=str))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=4)
        logger.debug(f"Saved current state to {filepath}")
    except (IOError, TypeError) as e:
        logger.error(f"Error saving state file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state: {e}", exc_info=True)


# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with V5 settings, error handling, and tests."""
    lg = logger
    global QUOTE_CURRENCY, IS_UNIFIED_ACCOUNT

    try:
        QUOTE_CURRENCY = config.get("quote_currency", "USDT")
        lg.info(f"Using Quote Currency: {QUOTE_CURRENCY}")

        exchange_options = { # Same options as v1.0.1
            'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True, 'rateLimit': 120,
            'options': {
                'defaultType': 'linear', 'adjustForTimeDifference': True, 'recvWindow': 10000,
                'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 20000, 'createOrderTimeout': 25000,
                'fetchOrderTimeout': 20000, 'fetchPositionsTimeout': 20000, 'cancelOrderTimeout': 20000,
                'fetchOHLCVTimeout': 20000, 'setLeverageTimeout': 20000, 'fetchMarketsTimeout': 30000,
                'brokerId': 'EnhancedWhale71',
                'versions': {'public': {'GET': {'market/tickers': 'v5','market/kline': 'v5','market/orderbook': 'v5',}},
                             'private': {'GET': {'position/list': 'v5','account/wallet-balance': 'v5','order/realtime': 'v5','order/history': 'v5',},
                                         'POST': {'order/create': 'v5','order/cancel': 'v5','position/set-leverage': 'v5','position/trading-stop': 'v5',}}},
                'default_options': {'fetchPositions': 'v5', 'fetchBalance': 'v5', 'createOrder': 'v5', 'fetchOrder': 'v5', 'fetchTicker': 'v5', 'fetchOHLCV': 'v5', 'fetchOrderBook': 'v5', 'setLeverage': 'v5', 'private_post_v5_position_trading_stop': 'v5',},
                'accountsByType': {'spot': 'SPOT', 'future': 'CONTRACT', 'swap': 'CONTRACT', 'margin': 'UNIFIED', 'option': 'OPTION', 'unified': 'UNIFIED', 'contract': 'CONTRACT',},
                'accountsById': {'SPOT': 'spot', 'CONTRACT': 'contract', 'UNIFIED': 'unified', 'OPTION': 'option',},
                'bybit': {'defaultSettleCoin': QUOTE_CURRENCY, }
            }
        }

        exchange_id = 'bybit'
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        if config.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}--- USING LIVE TRADING MODE (Real Money) ---{RESET}")

        lg.info(f"Connecting to {exchange.id} (Sandbox: {config.get('use_sandbox', True)})...")
        lg.info(f"Loading markets for {exchange.id}... (CCXT Version: {ccxt.__version__})")
        try:
            exchange.load_markets()
            exchange.last_load_markets_timestamp = time.time()
            lg.info(f"Markets loaded successfully for {exchange.id}. Found {len(exchange.markets)} markets.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            lg.error(f"{NEON_RED}Error loading markets: {e}. Check connection/API settings.{RESET}", exc_info=True)
            return None

        # --- Test API Credentials & Check Account Type ---
        lg.info(f"Attempting initial balance fetch for {QUOTE_CURRENCY} to test credentials and detect account type...")
        balance_decimal = None
        account_type_detected = None
        try:
            # Temporarily set IS_UNIFIED_ACCOUNT based on initial fetch result
            temp_is_unified, balance_decimal = _check_account_type_and_balance(exchange, QUOTE_CURRENCY, lg)
            if temp_is_unified is not None:
                IS_UNIFIED_ACCOUNT = temp_is_unified
                account_type_detected = "UNIFIED" if IS_UNIFIED_ACCOUNT else "CONTRACT/SPOT (Non-UTA)"
                lg.info(f"Detected Account Type: {account_type_detected}")
            else:
                lg.warning("Could not definitively determine account type during initial balance check.")

            if balance_decimal is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch failed or returned None (Account Type: {account_type_detected}). Check logs above. Ensure API keys have 'Read' permissions and the correct account type is accessible.{RESET}")
                 if config.get("enable_trading"):
                     lg.error(f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}")
                     return None
                 else:
                     lg.warning("Continuing in non-trading mode despite balance fetch issue.")

        except ccxt.AuthenticationError as auth_err:
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial setup: {auth_err}{RESET}")
            lg.error(f"{NEON_RED}>> Check API keys, permissions, account type (Real/Testnet), and IP whitelist.{RESET}")
            return None
        except Exception as balance_err:
            lg.error(f"{NEON_RED}Unexpected error during initial balance check: {balance_err}{RESET}", exc_info=True)
            if config.get("enable_trading"):
                 lg.error(f"{NEON_RED}Aborting initialization due to unexpected balance fetch error in trading mode.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}Continuing in non-trading mode despite unexpected balance fetch error: {balance_err}{RESET}")

        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError, Exception) as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None

def _check_account_type_and_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Tuple[Optional[bool], Optional[Decimal]]:
    """Tries fetching balance for UNIFIED and CONTRACT to detect type and get balance."""
    lg = logger
    # Try UNIFIED first, as it's stricter
    try:
        lg.debug("Checking balance with accountType=UNIFIED...")
        params_unified = {'accountType': 'UNIFIED', 'coin': currency}
        bal_info_unified = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, params=params_unified) # Only 1 retry for detection
        # Check if successful and parse
        parsed_balance = _parse_balance_response(bal_info_unified, currency, 'UNIFIED', lg)
        if parsed_balance is not None:
            lg.info("Successfully fetched balance using UNIFIED account type.")
            return True, parsed_balance # It's a Unified Account
    except ccxt.ExchangeError as e:
        # Check if error indicates "account type not support" or similar for UNIFIED
        if "accountType only support" in str(e) or "30086" in str(e) or "unified account is not supported" in str(e).lower():
             lg.debug("Fetching with UNIFIED failed (as expected for non-UTA), trying CONTRACT/SPOT...")
        else:
             lg.warning(f"ExchangeError checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")
             # Fall through to check CONTRACT/SPOT
    except Exception as e:
         lg.warning(f"Unexpected error checking UNIFIED balance: {e}. Proceeding to check CONTRACT/SPOT.")
         # Fall through

    # If UNIFIED failed or wasn't definitively successful, try CONTRACT/SPOT
    try:
        lg.debug("Checking balance with accountType=CONTRACT...")
        params_contract = {'accountType': 'CONTRACT', 'coin': currency}
        bal_info_contract = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, params=params_contract)
        parsed_balance = _parse_balance_response(bal_info_contract, currency, 'CONTRACT', lg)
        if parsed_balance is not None:
             lg.info("Successfully fetched balance using CONTRACT account type.")
             return False, parsed_balance # It's likely a Contract/Standard Account

        # If CONTRACT also failed specifically, try SPOT as last resort?
        lg.debug("Checking balance with accountType=SPOT...")
        params_spot = {'accountType': 'SPOT', 'coin': currency}
        bal_info_spot = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=1, params=params_spot)
        parsed_balance = _parse_balance_response(bal_info_spot, currency, 'SPOT', lg)
        if parsed_balance is not None:
             lg.info("Successfully fetched balance using SPOT account type.")
             return False, parsed_balance # Standard Spot account

    except ccxt.ExchangeError as e:
         lg.warning(f"ExchangeError checking CONTRACT/SPOT balance: {e}.")
         # Could not determine type this way
    except Exception as e:
         lg.warning(f"Unexpected error checking CONTRACT/SPOT balance: {e}.")
         # Could not determine type this way

    # If all attempts failed to get a definitive balance
    lg.error("Failed to determine account type or fetch balance with common types (UNIFIED/CONTRACT/SPOT).")
    return None, None # Unknown type, no balance


# --- CCXT API Call Helper with Retries ---
# safe_ccxt_call remains the same as v1.0.1 (with improved error code handling)
def safe_ccxt_call(
    exchange: ccxt.Exchange,
    method_name: str,
    logger: logging.Logger,
    max_retries: int = MAX_API_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
    *args, **kwargs
) -> Any:
    """Safely calls a CCXT method with retry logic for network errors and rate limits."""
    lg = logger
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            method = getattr(exchange, method_name)
            result = method(*args, **kwargs)
            return result
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * (2 ** attempt); suggested_wait = None
            try:
                import re
                error_msg = str(e).lower()
                match_ms = re.search(r'(?:try again in|retry after)\s*(\d+)\s*ms', error_msg)
                match_s = re.search(r'(?:try again in|retry after)\s*(\d+)\s*s', error_msg)
                if match_ms: suggested_wait = max(1, math.ceil(int(match_ms.group(1)) / 1000) + RATE_LIMIT_BUFFER_SECONDS)
                elif match_s: suggested_wait = max(1, int(match_s.group(1)) + RATE_LIMIT_BUFFER_SECONDS)
                elif "too many visits" in error_msg or "limit" in error_msg: suggested_wait = wait_time + 1.0
            except Exception: pass
            final_wait = suggested_wait if suggested_wait is not None else wait_time
            lg.warning(f"Rate limit hit calling {method_name}. Retrying in {final_wait:.2f}s... (Attempt {attempt + 1}) Error: {e}")
            time.sleep(final_wait)
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e; wait_time = retry_delay * (2 ** attempt)
            lg.warning(f"Network/DDoS/Timeout error calling {method_name}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except ccxt.AuthenticationError as e:
            lg.error(f"{NEON_RED}Authentication Error calling {method_name}: {e}. Check API keys/permissions. Not retrying.{RESET}"); raise e
        except ccxt.ExchangeError as e:
            bybit_code = None; ret_msg = str(e)
            try:
                if hasattr(e, 'args') and len(e.args) > 0:
                    error_details = str(e.args[0])
                    if "retCode" in error_details:
                         details_dict = json.loads(error_details[error_details.find('{'):error_details.rfind('}')+1])
                         bybit_code = details_dict.get('retCode'); ret_msg = details_dict.get('retMsg', str(e))
            except Exception: pass
            non_retryable_codes = [ 10001, 110007, 110013, 110017, 110020, 110025, 110043, 110045, 170007, 170131, 170132, 170133, 170140, 30086 ]
            if bybit_code in non_retryable_codes:
                if bybit_code == 110043 and method_name == 'set_leverage':
                    lg.info(f"Leverage already set (Code 110043) when calling {method_name}. Ignoring.")
                    return {} # Treat as success for leverage
                extra_info = ""
                if bybit_code == 10001 and "accountType" in ret_msg: extra_info = f"{NEON_YELLOW} Hint: Check 'accountType' (UNIFIED vs CONTRACT/SPOT).{RESET}"
                elif bybit_code == 10001: extra_info = f"{NEON_YELLOW} Hint: Check API params ({args=}, {kwargs=}).{RESET}"
                lg.error(f"{NEON_RED}Non-retryable Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Not retrying.{RESET}{extra_info}"); raise e
            else:
                lg.warning(f"{NEON_YELLOW}Retryable/Unknown Exchange Error calling {method_name}: {ret_msg} (Code: {bybit_code}). Retrying... (Attempt {attempt + 1}){RESET}")
                last_exception = e; wait_time = retry_delay * (2 ** attempt); time.sleep(wait_time)
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected Error calling {method_name}: {e}. Not retrying.{RESET}", exc_info=True); raise e
    lg.error(f"{NEON_RED}Max retries ({max_retries}) reached for {method_name}. Last error: {last_exception}{RESET}"); raise last_exception

# --- Market Info Helper Functions ---
# (get_market_info, _determine_category, get_..._from_market functions same as v1.0.1)
# Assume correct based on previous version.

# --- Data Fetching Wrappers ---
# (fetch_current_price_ccxt, fetch_klines_ccxt, fetch_orderbook_ccxt same as v1.0.1)
# Assume correct based on previous version.

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches available balance for a currency, trying appropriate account types.
    Prioritizes UNIFIED if IS_UNIFIED_ACCOUNT flag is set.
    """
    lg = logger
    # Determine account types to try based on detected type
    if IS_UNIFIED_ACCOUNT:
        account_types_to_try = ['UNIFIED']
        lg.debug(f"Fetching balance specifically for UNIFIED account.")
    else:
        account_types_to_try = ['CONTRACT', 'SPOT'] # Try CONTRACT first for non-UTA
        lg.debug(f"Fetching balance for Non-UTA account (trying {account_types_to_try}).")

    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        balance_info = None
        successful_acc_type = None

        for acc_type in account_types_to_try:
            try:
                params = {'accountType': acc_type, 'coin': currency}
                lg.debug(f"Fetching balance with params={params} (Attempt {attempt + 1})")
                balance_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=0, params=params) # No inner retry

                parsed_balance = _parse_balance_response(balance_info, currency, acc_type, lg)
                if parsed_balance is not None:
                    # Found balance, return it directly
                    lg.info(f"Available {currency} balance ({acc_type}): {parsed_balance:.4f}")
                    return parsed_balance
                else:
                    # Parsing failed or currency not found for this type, try next type
                    balance_info = None # Reset for next loop iteration

            except ccxt.ExchangeError as e:
                # Handle specific error codes if needed, but safe_ccxt_call handles non-retryable ones
                lg.debug(f"Exchange error fetching balance type {acc_type}: {e}. Trying next type if available.")
                last_exception = e
                continue # Try next account type
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                 lg.warning(f"Network/RateLimit error during balance fetch type {acc_type}: {e}")
                 last_exception = e; break # Break inner loop, let outer loop retry
            except Exception as e:
                 lg.error(f"Unexpected error during balance fetch type {acc_type}: {e}", exc_info=True)
                 last_exception = e; break # Break inner loop, let outer loop retry

        # If we broke from inner loop due to network/unexpected error, retry outer loop
        if isinstance(last_exception, (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded, Exception)):
            if attempt < MAX_API_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.warning(f"Balance fetch attempt {attempt + 1} encountered network/unexpected error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue # Retry outer loop
            else:
                 lg.error(f"{NEON_RED}Max retries reached fetching balance for {currency} after network/unexpected error. Last error: {last_exception}{RESET}")
                 return None # Exhausted retries

        # If all account types tried and none yielded balance (and no network errors)
        if attempt < MAX_API_RETRIES:
             # This case should ideally not be reached if parsing is correct,
             # but could happen if account has 0 balance and API returns empty list?
             wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
             lg.warning(f"Balance fetch attempt {attempt + 1} failed for type(s): {account_types_to_try}. Retrying in {wait_time}s...")
             time.sleep(wait_time)
        else:
             lg.error(f"{NEON_RED}Max retries reached. Failed to fetch balance for {currency} using types: {account_types_to_try}. Last error: {last_exception}{RESET}")
             return None

    # Should not be reachable if logic is correct, but as a failsafe:
    lg.error(f"{NEON_RED}Balance fetch logic completed unexpectedly without returning a value for {currency}.{RESET}")
    return None


def _parse_balance_response(balance_info: Optional[Dict], currency: str, account_type_checked: str, logger: logging.Logger) -> Optional[Decimal]:
    """Parses the fetch_balance response, prioritizing Bybit V5 structure."""
    if not balance_info: return None
    lg = logger
    available_balance_str = None

    try:
        # 1. Prioritize Bybit V5 structure: info -> result -> list -> coin[]
        if 'info' in balance_info and balance_info['info'].get('retCode') == 0 and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
            balance_list = balance_info['info']['result']['list']
            if isinstance(balance_list, list):
                for account_data in balance_list:
                    # Check if accountType matches what we requested (important for UTA response)
                    if account_data.get('accountType') == account_type_checked:
                        coin_list = account_data.get('coin')
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get('coin') == currency:
                                    # V5 Field: availableBalance (most relevant for new orders)
                                    free = coin_data.get('availableBalance')
                                    if free is None or str(free).strip() == "":
                                        lg.debug(f"'availableBalance' missing for {currency} in {account_type_checked}, trying 'walletBalance'")
                                        free = coin_data.get('walletBalance') # Fallback to total

                                    if free is not None and str(free).strip() != "":
                                        available_balance_str = str(free)
                                        lg.debug(f"Parsed balance from Bybit V5 ({account_type_checked}): {available_balance_str} {currency}")
                                        break # Found currency in this account type
                            if available_balance_str is not None: break # Found currency
                if available_balance_str is None:
                     lg.debug(f"Currency '{currency}' not found within Bybit V5 list for account type '{account_type_checked}'. Response list: {balance_list}")
            else:
                 lg.debug(f"Bybit V5 balance 'list' is not a list for {account_type_checked}. Info: {balance_info.get('info')}")

        # 2. Fallback: Standard CCXT structure (less likely for Bybit V5 with params)
        elif currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('free') is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Parsed balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}")

        # 3. Fallback: Top-level 'free' dictionary (even less likely)
        elif 'free' in balance_info and isinstance(balance_info.get('free'), dict) and currency in balance_info['free']:
            available_balance_str = str(balance_info['free'][currency])
            lg.debug(f"Parsed balance via top-level 'free' dictionary: {available_balance_str} {currency}")

        # If balance still not found
        if available_balance_str is None:
            lg.debug(f"Could not extract balance for {currency} from structure type '{account_type_checked}'.")
            return None

        # --- Convert to Decimal ---
        final_balance = Decimal(available_balance_str)
        if final_balance.is_finite() and final_balance >= 0:
            return final_balance
        else:
            lg.error(f"Parsed balance for {currency} ('{available_balance_str}') is invalid."); return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}."); return None
    except Exception as e:
        lg.error(f"Unexpected error parsing balance structure for {currency}: {e}", exc_info=True); return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes trading data using pandas_ta.Strategy, generates weighted signals,
    and provides risk management helpers. Uses Decimal for precision internally.
    Manages state like break-even trigger status via a passed-in dictionary.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
        symbol_state: Dict[str, Any],
    ) -> None:
        if df is None or df.empty: raise ValueError("TradingAnalyzer requires a non-empty DataFrame.")
        if not market_info: raise ValueError("TradingAnalyzer requires valid market_info.")
        if symbol_state is None: raise ValueError("TradingAnalyzer requires a valid symbol_state dict.")

        self.df_raw = df # Keep raw DF if needed
        self.df = df.copy() # Work on a copy
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.symbol_state = symbol_state # Reference mutable state dict

        self.indicator_values: Dict[str, Optional[Decimal]] = {}
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_strategy: Optional[ta.Strategy] = None
        self.ta_column_map: Dict[str, str] = {}

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' empty for {self.symbol}. Signals disabled.{RESET}")

        # --- Convert DataFrame OHLCV to float for pandas_ta ---
        try:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 if col in self.df.columns:
                      # Ensure column exists before conversion
                      if pd.api.types.is_decimal_dtype(self.df[col]):
                          self.df[col] = self.df[col].apply(lambda x: float(x) if x.is_finite() else np.nan)
                      elif not pd.api.types.is_float_dtype(self.df[col]): # If not decimal and not already float
                          self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            # logger.debug(f"DataFrame dtypes for TA: {self.df[['open', 'high', 'low', 'close', 'volume']].dtypes.to_dict()}")
        except Exception as e:
             logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
             # If conversion fails, TA might fail later.

        # --- Initialize and Calculate ---
        self._define_ta_strategy()
        self._calculate_all_indicators()
        self._update_latest_indicator_values() # Populates self.indicator_values with Decimals
        self.calculate_fibonacci_levels()

    @property
    def break_even_triggered(self) -> bool: return self.symbol_state.get('break_even_triggered', False)
    @break_even_triggered.setter
    def break_even_triggered(self, value: bool):
        if self.symbol_state.get('break_even_triggered') != value:
            self.symbol_state['break_even_triggered'] = value
            self.logger.info(f"Break-even status for {self.symbol} set to: {value}")

    # _define_ta_strategy, _calculate_all_indicators remain same as v1.0.1

    def _define_ta_strategy(self) -> None:
        """Defines the pandas_ta Strategy based on config."""
        cfg = self.config; indi_cfg = cfg.get("indicators", {})
        def get_num_param(key: str, default: Union[int, float]) -> Union[int, float]:
            val = cfg.get(key, default)
            try: return int(val) if isinstance(default, int) else float(val)
            except (ValueError, TypeError): return default

        atr_p = get_num_param("atr_period", DEFAULT_ATR_PERIOD); ema_s = get_num_param("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
        ema_l = get_num_param("ema_long_period", DEFAULT_EMA_LONG_PERIOD); rsi_p = get_num_param("rsi_period", DEFAULT_RSI_WINDOW)
        bb_p = get_num_param("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD); bb_std = get_num_param("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        cci_w = get_num_param("cci_window", DEFAULT_CCI_WINDOW); wr_w = get_num_param("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        mfi_w = get_num_param("mfi_window", DEFAULT_MFI_WINDOW); stochrsi_w = get_num_param("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
        stochrsi_rsi_w = get_num_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW); stochrsi_k = get_num_param("stoch_rsi_k", DEFAULT_K_WINDOW)
        stochrsi_d = get_num_param("stoch_rsi_d", DEFAULT_D_WINDOW); psar_af = get_num_param("psar_af", DEFAULT_PSAR_AF)
        psar_max = get_num_param("psar_max_af", DEFAULT_PSAR_MAX_AF); sma10_w = get_num_param("sma_10_window", DEFAULT_SMA_10_WINDOW)
        mom_p = get_num_param("momentum_period", DEFAULT_MOMENTUM_PERIOD); vol_ma_p = get_num_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        ta_list = []; self.ta_column_map = {}
        ta_list.append({"kind": "atr", "length": atr_p}); self.ta_column_map["ATR"] = f"ATRr_{atr_p}"
        if indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit"):
            if ema_s > 0: ta_list.append({"kind": "ema", "length": ema_s, "col_names": (f"EMA_{ema_s}",)}); self.ta_column_map["EMA_Short"] = f"EMA_{ema_s}"
            if ema_l > 0: ta_list.append({"kind": "ema", "length": ema_l, "col_names": (f"EMA_{ema_l}",)}); self.ta_column_map["EMA_Long"] = f"EMA_{ema_l}"
        if indi_cfg.get("momentum") and mom_p > 0: ta_list.append({"kind": "mom", "length": mom_p, "col_names": (f"MOM_{mom_p}",)}); self.ta_column_map["Momentum"] = f"MOM_{mom_p}"
        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0: ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_p, "col_names": (f"VOL_SMA_{vol_ma_p}",)}); self.ta_column_map["Volume_MA"] = f"VOL_SMA_{vol_ma_p}"
        if indi_cfg.get("stoch_rsi") and stochrsi_w > 0 and stochrsi_rsi_w > 0 and stochrsi_k > 0 and stochrsi_d > 0:
            k_col, d_col = f"STOCHRSIk_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}", f"STOCHRSId_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            ta_list.append({"kind": "stochrsi", "length": stochrsi_w, "rsi_length": stochrsi_rsi_w, "k": stochrsi_k, "d": stochrsi_d, "col_names": (k_col, d_col)}); self.ta_column_map["StochRSI_K"], self.ta_column_map["StochRSI_D"] = k_col, d_col
        if indi_cfg.get("rsi") and rsi_p > 0: ta_list.append({"kind": "rsi", "length": rsi_p, "col_names": (f"RSI_{rsi_p}",)}); self.ta_column_map["RSI"] = f"RSI_{rsi_p}"
        if indi_cfg.get("bollinger_bands") and bb_p > 0:
            bbl, bbm, bbu = f"BBL_{bb_p}_{bb_std:.1f}", f"BBM_{bb_p}_{bb_std:.1f}", f"BBU_{bb_p}_{bb_std:.1f}"
            ta_list.append({"kind": "bbands", "length": bb_p, "std": bb_std, "col_names": (bbl, bbm, bbu, f"BBB_{bb_p}_{bb_std:.1f}", f"BBP_{bb_p}_{bb_std:.1f}")})
            self.ta_column_map["BB_Lower"], self.ta_column_map["BB_Middle"], self.ta_column_map["BB_Upper"] = bbl, bbm, bbu
        if indi_cfg.get("vwap"):
            if 'typical' not in self.df.columns: self.df['typical'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
            vwap_col = "VWAP_D"; ta_list.append({"kind": "vwap", "col_names": (vwap_col,)}); self.ta_column_map["VWAP"] = vwap_col
        if indi_cfg.get("cci") and cci_w > 0: cci_col = f"CCI_{cci_w}_0.015"; ta_list.append({"kind": "cci", "length": cci_w, "col_names": (cci_col,)}); self.ta_column_map["CCI"] = cci_col
        if indi_cfg.get("wr") and wr_w > 0: wr_col = f"WILLR_{wr_w}"; ta_list.append({"kind": "willr", "length": wr_w, "col_names": (wr_col,)}); self.ta_column_map["WR"] = wr_col
        if indi_cfg.get("psar"):
            psar_af_str, psar_max_str = f"{psar_af}".rstrip('0').rstrip('.'), f"{psar_max}".rstrip('0').rstrip('.')
            l, s, af, r = f"PSARl_{psar_af_str}_{psar_max_str}", f"PSARs_{psar_af_str}_{psar_max_str}", f"PSARaf_{psar_af_str}_{psar_max_str}", f"PSARr_{psar_af_str}_{psar_max_str}"
            ta_list.append({"kind": "psar", "af": psar_af, "max_af": psar_max, "col_names": (l, s, af, r)}); self.ta_column_map["PSAR_Long"], self.ta_column_map["PSAR_Short"], self.ta_column_map["PSAR_AF"], self.ta_column_map["PSAR_Reversal"] = l, s, af, r
        if indi_cfg.get("sma_10") and sma10_w > 0: ta_list.append({"kind": "sma", "length": sma10_w, "col_names": (f"SMA_{sma10_w}",)}); self.ta_column_map["SMA10"] = f"SMA_{sma10_w}"
        if indi_cfg.get("mfi") and mfi_w > 0:
            if 'typical' not in self.df.columns: self.df['typical'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
            ta_list.append({"kind": "mfi", "length": mfi_w, "col_names": (f"MFI_{mfi_w}",)}); self.ta_column_map["MFI"] = f"MFI_{mfi_w}"

        if not ta_list: self.logger.warning(f"No indicators enabled or valid for {self.symbol}."); return
        self.ta_strategy = ta.Strategy(name="EnhancedMultiIndicator", description="Calculates multiple TA indicators based on config", ta=ta_list)
        self.logger.debug(f"Defined TA Strategy for {self.symbol} with {len(ta_list)} indicators.")

    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using the defined pandas_ta strategy."""
        if self.df.empty: self.logger.warning(f"DataFrame empty, cannot calculate indicators for {self.symbol}."); return
        if not self.ta_strategy: self.logger.warning(f"TA Strategy not defined for {self.symbol}."); return

        min_required_data = self.ta_strategy.required if hasattr(self.ta_strategy, 'required') else 50; buffer = 20
        if len(self.df) < min_required_data + buffer:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} TA (min recommended: {min_required_data + buffer}). Results may be inaccurate.{RESET}")

        try:
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol}...")
            self.df.ta.strategy(self.ta_strategy, timed=False)
            self.logger.debug(f"Finished indicator calculations for {self.symbol}.")
        except AttributeError as ae:
             if "'Decimal' object has no attribute" in str(ae): self.logger.error(f"{NEON_RED}Pandas TA Error ({self.symbol}): Input must be float, not Decimal. Error: {ae}{RESET}", exc_info=False)
             else: self.logger.error(f"{NEON_RED}Pandas TA attribute error ({self.symbol}): {ae}. Is pandas_ta installed correctly?{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta strategy for {self.symbol}: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest Decimal values from self.df."""
        self.indicator_values = {} # Reset
        if self.df.empty: self.logger.warning(f"DataFrame empty, cannot update latest values for {self.symbol}."); return
        try:
            latest_series = self.df.iloc[-1]
            if latest_series.isnull().all(): self.logger.warning(f"Last row is all NaNs, cannot update latest values for {self.symbol}."); return

            def to_decimal(value: Any) -> Optional[Decimal]:
                if pd.isna(value) or value is None: return None
                try: dec_val = Decimal(str(value)); return dec_val if dec_val.is_finite() else None
                except: return None

            for generic_name, actual_col_name in self.ta_column_map.items():
                self.indicator_values[generic_name] = to_decimal(latest_series.get(actual_col_name))
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                self.indicator_values[base_col.capitalize()] = to_decimal(latest_series.get(base_col))

            valid_values_str = {k: f"{v:.5f}" for k, v in self.indicator_values.items() if v is not None}
            self.logger.debug(f"Latest indicator Decimal values updated for {self.symbol}: {valid_values_str}")
        except IndexError: self.logger.error(f"DataFrame index out of bounds updating latest values for {self.symbol}.")
        except Exception as e: self.logger.error(f"Unexpected error updating latest values for {self.symbol}: {e}", exc_info=True); self.indicator_values = {}

    # --- Precision and Market Info Helpers ---
    def get_min_tick_size(self) -> Optional[Decimal]:
        tick = self.market_info.get('min_tick_size')
        if tick is None or tick <= 0: self.logger.warning(f"Invalid min_tick_size ({tick}) for {self.symbol}"); return None
        return tick
    def get_price_precision_digits(self) -> int: return self.market_info.get('price_precision_digits', 8)
    def get_amount_precision_digits(self) -> int: return self.market_info.get('amount_precision_digits', 8)

    def quantize_price(self, price: Union[Decimal, float, str], rounding=ROUND_DOWN) -> Optional[Decimal]:
        """Quantizes a price to the market's tick size using specified rounding."""
        min_tick = self.get_min_tick_size()
        if min_tick is None: self.logger.error(f"Cannot quantize price for {self.symbol}: Missing min_tick_size."); return None
        try:
            price_decimal = Decimal(str(price))
            if not price_decimal.is_finite(): return None
            quantized = (price_decimal / min_tick).quantize(Decimal('0'), rounding=rounding) * min_tick
            return quantized
        except (InvalidOperation, ValueError, TypeError) as e: self.logger.error(f"Error quantizing price '{price}' for {self.symbol}: {e}"); return None

    def quantize_amount(self, amount: Union[Decimal, float, str], rounding=ROUND_DOWN) -> Optional[Decimal]:
        """Quantizes an amount to the market's amount precision (step size) using specified rounding."""
        amount_digits = self.get_amount_precision_digits()
        try:
            amount_decimal = Decimal(str(amount))
            if not amount_decimal.is_finite(): return None
            step_size = Decimal('1') / (Decimal('10') ** amount_digits)
            quantized = (amount_decimal / step_size).quantize(Decimal('0'), rounding=rounding) * step_size
            return Decimal(f"{quantized:.{amount_digits}f}") # Format to correct digits
        except (InvalidOperation, ValueError, TypeError) as e: self.logger.error(f"Error quantizing amount '{amount}' for {self.symbol}: {e}"); return None

    # --- Fibonacci Calculation ---
    # calculate_fibonacci_levels remains same as v1.0.1
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a window using Decimal and quantization."""
        self.fib_levels_data = {}
        window = window or int(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))
        if len(self.df_raw) < window: self.logger.debug(f"Not enough data ({len(self.df_raw)}) for Fib window ({window}) on {self.symbol}."); return {}

        # Use raw Decimal data if available, otherwise convert float from TA df
        df_slice = self.df_raw.tail(window)
        try:
            if pd.api.types.is_decimal_dtype(df_slice["high"]):
                high_raw = df_slice["high"].dropna().max(); low_raw = df_slice["low"].dropna().min()
            else: high_raw = Decimal(str(df_slice["high"].dropna().max())); low_raw = Decimal(str(df_slice["low"].dropna().min()))

            if not high_raw.is_finite() or not low_raw.is_finite(): self.logger.warning(f"No valid high/low in last {window} periods for Fib on {self.symbol}."); return {}

            high, low, diff = high_raw, low_raw, high_raw - low_raw
            levels = {}; min_tick = self.get_min_tick_size()

            if diff >= 0 and min_tick is not None:
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float))
                    level_price_raw = high - (diff * level_pct)
                    level_price = self.quantize_price(level_price_raw, rounding=ROUND_DOWN)
                    if level_price is not None: levels[f"Fib_{level_pct * 100:.1f}%"] = level_price
                    else: self.logger.warning(f"Failed to quantize Fib level {level_pct*100:.1f}% ({level_price_raw}) for {self.symbol}")
            elif min_tick is None: # Fallback if tick size missing
                 self.logger.warning(f"Using raw Fib values for {self.symbol} due to missing min_tick_size.")
                 for level_pct_float in FIB_LEVELS: levels[f"Fib_{Decimal(str(level_pct_float)) * 100:.1f}%"] = high - (diff * Decimal(str(level_pct_float)))
            elif diff < 0 : self.logger.warning(f"Invalid range (high < low?) for Fib on {self.symbol}. High={high}, Low={low}")

            self.fib_levels_data = levels
            # price_prec = self.get_price_precision_digits(); log_levels = {k: f"{v:.{price_prec}f}" for k, v in levels.items()}
            # self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {log_levels}")
            return levels
        except Exception as e: self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True); return {}


    # --- Indicator Check Methods (Return float score -1.0 to 1.0, or None) ---
    # _get_indicator_float and check methods remain same as v1.0.1
    def _get_indicator_float(self, name: str) -> Optional[float]:
        """Safely gets indicator value as float."""
        val = self.indicator_values.get(name)
        if val is None or not val.is_finite(): return None
        try: return float(val)
        except: return None
    def _check_ema_alignment(self) -> Optional[float]:
        s, l = self._get_indicator_float("EMA_Short"), self._get_indicator_float("EMA_Long"); return 1.0 if s and l and s > l else -1.0 if s and l and s < l else 0.0 if s and l else None
    def _check_momentum(self) -> Optional[float]:
        mom = self._get_indicator_float("Momentum"); sf = 0.1; return max(-1.0, min(1.0, mom * sf)) if mom is not None else None
    def _check_volume_confirmation(self) -> Optional[float]:
        vol, vol_ma = self._get_indicator_float("Volume"), self._get_indicator_float("Volume_MA"); mult = float(self.config.get("volume_confirmation_multiplier", 1.5)); return 0.7 if vol and vol_ma and vol_ma > 0 and vol > vol_ma * mult else 0.0 if vol and vol_ma else None
    def _check_stoch_rsi(self) -> Optional[float]:
        k, d = self._get_indicator_float("StochRSI_K"), self._get_indicator_float("StochRSI_D"); os, ob = float(self.config.get("stoch_rsi_oversold_threshold", 25.0)), float(self.config.get("stoch_rsi_overbought_threshold", 75.0)); score = 0.0
        if k is None or d is None: return None
        if k < os and d < os: score = 0.8 if k > d else 0.6
        elif k > ob and d > ob: score = -0.8 if k < d else -0.6
        elif k < os: score = 0.5; elif k > ob: score = -0.5
        elif k > d: score = 0.2; elif k < d: score = -0.2; return max(-1.0, min(1.0, score))
    def _check_rsi(self) -> Optional[float]:
        rsi = self._get_indicator_float("RSI"); if rsi is None: return None; score = 0.0
        if rsi <= 20: score=1.0; elif rsi <= 30: score=0.7; elif rsi >= 80: score=-1.0; elif rsi >= 70: score=-0.7; elif 30 < rsi < 70: score = 1.0 - (rsi - 30.0) * (2.0 / 40.0); return score
    def _check_cci(self) -> Optional[float]:
        cci = self._get_indicator_float("CCI"); if cci is None: return None; score = 0.0
        if cci <= -200: score=1.0; elif cci <= -100: score=0.7; elif cci >= 200: score=-1.0; elif cci >= 100: score=-0.7; elif -100 < cci < 100: score = -(cci / 100.0) * 0.3; return score
    def _check_wr(self) -> Optional[float]: # Williams %R
        wr = self._get_indicator_float("WR"); if wr is None: return None; score = 0.0
        if wr <= -90: score=1.0; elif wr <= -80: score=0.7; elif wr >= -10: score=-1.0; elif wr >= -20: score=-0.7; elif -80 < wr < -20: score = 1.0 - (wr - (-80.0)) * (2.0 / 60.0); return score
    def _check_psar(self) -> Optional[float]:
        l, s = self.indicator_values.get("PSAR_Long"), self.indicator_values.get("PSAR_Short"); l_act, s_act = l is not None and l.is_finite(), s is not None and s.is_finite()
        if l_act and not s_act: return 1.0; if not l_act and s_act: return -1.0; return 0.0
    def _check_sma10(self) -> Optional[float]:
        sma, close = self._get_indicator_float("SMA10"), self._get_indicator_float("Close"); return 0.5 if sma and close and close > sma else -0.5 if sma and close and close < sma else 0.0 if sma and close else None
    def _check_vwap(self) -> Optional[float]:
        vwap, close = self._get_indicator_float("VWAP"), self._get_indicator_float("Close"); return 0.6 if vwap and close and close > vwap else -0.6 if vwap and close and close < vwap else 0.0 if vwap and close else None
    def _check_mfi(self) -> Optional[float]:
        mfi = self._get_indicator_float("MFI"); if mfi is None: return None; score = 0.0
        if mfi <= 15: score=1.0; elif mfi <= 25: score=0.7; elif mfi >= 85: score=-1.0; elif mfi >= 75: score=-0.7; elif 25 < mfi < 75: score = 1.0 - (mfi - 25.0) * (2.0 / 50.0); return score
    def _check_bollinger_bands(self) -> Optional[float]:
        bbl, bbu, close = self._get_indicator_float("BB_Lower"), self._get_indicator_float("BB_Upper"), self._get_indicator_float("Close")
        if bbl is None or bbu is None or close is None: return None; score = 0.0
        if close <= bbl: score = 1.0; elif close >= bbu: score = -1.0
        else: band_range = bbu - bbl; if band_range > 0: position_in_band = (close - bbl) / band_range; score = 1.0 - 2.0 * position_in_band
        return max(-1.0, min(1.0, score))
    def _check_orderbook(self, orderbook_data: Optional[Dict]) -> Optional[float]:
        if not orderbook_data: return None
        try:
            bids, asks = orderbook_data.get('bids', []), orderbook_data.get('asks', [])
            if not bids or not asks: return 0.0
            lim = int(self.config.get("orderbook_limit", 10)); lvls = min(len(bids), len(asks), lim)
            if lvls <= 0: return 0.0
            bid_vol = sum(Decimal(str(b[1])) for b in bids[:lvls]); ask_vol = sum(Decimal(str(a[1])) for a in asks[:lvls]); total_vol = bid_vol + ask_vol
            if total_vol <= 0: return 0.0
            obi = (bid_vol - ask_vol) / total_vol
            return float(max(Decimal("-1.0"), min(Decimal("1.0"), obi)))
        except (InvalidOperation, ValueError, TypeError, IndexError) as e: self.logger.warning(f"OBI calc error for {self.symbol}: {e}"); return None
        except Exception as e: self.logger.error(f"Unexpected OBI error: {e}"); return None

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: Optional[Dict]) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}; final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0"); active_indicator_count = 0; nan_indicator_count = 0
        debug_scores = {}

        if not self.indicator_values or not current_price_dec.is_finite() or current_price_dec <= 0:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid inputs."); return "HOLD"
        if not self.weights: return "HOLD" # Weights disabled log handled in init

        active_weights = self.weights # Already loaded in init
        indicator_check_methods = { # Map keys directly
            "ema_alignment": self._check_ema_alignment, "momentum": self._check_momentum,
            "volume_confirmation": self._check_volume_confirmation, "stoch_rsi": self._check_stoch_rsi,
            "rsi": self._check_rsi, "bollinger_bands": self._check_bollinger_bands, "vwap": self._check_vwap,
            "cci": self._check_cci, "wr": self._check_wr, "psar": self._check_psar, "sma_10": self._check_sma10,
            "mfi": self._check_mfi, "orderbook": lambda: self._check_orderbook(orderbook_data),
        }

        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: debug_scores[indicator_key] = "Disabled"; continue
            weight_str = active_weights.get(indicator_key)
            if weight_str is None: debug_scores[indicator_key] = "No Weight"; continue
            try: weight = Decimal(str(weight_str)); assert weight.is_finite()
            except: self.logger.warning(f"Invalid weight '{weight_str}' for {indicator_key}. Skipping."); debug_scores[indicator_key] = f"Invalid Wt"; continue
            if weight == 0: debug_scores[indicator_key] = "Wt=0"; continue

            check_method = indicator_check_methods.get(indicator_key)
            if check_method:
                indicator_score_float = None
                try: indicator_score_float = check_method()
                except Exception as e: self.logger.error(f"Error in check method for {indicator_key} on {self.symbol}: {e}", exc_info=True)

                if indicator_score_float is not None and math.isfinite(indicator_score_float):
                    try:
                        clamped_score = max(-1.0, min(1.0, indicator_score_float))
                        indicator_score_decimal = Decimal(str(clamped_score))
                        weighted_score = indicator_score_decimal * weight
                        final_signal_score += weighted_score
                        total_weight_applied += abs(weight)
                        active_indicator_count += 1
                        debug_scores[indicator_key] = f"{indicator_score_float:.2f}(x{weight:.2f})={weighted_score:.3f}"
                    except Exception as calc_err: self.logger.error(f"Error processing score for {indicator_key}: {calc_err}"); debug_scores[indicator_key] = "Calc Err"; nan_indicator_count += 1
                else: debug_scores[indicator_key] = "NaN/None"; nan_indicator_count += 1
            elif indicator_key in active_weights: self.logger.warning(f"Check method missing for enabled indicator with weight: {indicator_key}"); debug_scores[indicator_key] = "No Method"

        final_signal = "HOLD"; normalized_score = Decimal("0.0")
        if total_weight_applied > 0: normalized_score = (final_signal_score / total_weight_applied).quantize(Decimal("0.0001"))
        elif active_indicator_count > 0: self.logger.warning(f"No non-zero weights applied for {active_indicator_count} active indicators on {self.symbol}. HOLDING.")

        threshold_key = "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
        default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
        try: threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
        except: threshold = Decimal(str(default_threshold)); self.logger.warning(f"Invalid threshold value for {threshold_key}. Using default: {threshold}")

        if final_signal_score >= threshold: final_signal = "BUY"
        elif final_signal_score <= -threshold: final_signal = "SELL"

        price_prec = self.get_price_precision_digits()
        score_details_str = ", ".join([f"{k}: {v}" for k, v in debug_scores.items() if v not in ["Disabled", "No Weight", "Wt=0"]])
        log_msg = (f"Signal Calc ({self.symbol} @ {current_price_dec:.{price_prec}f}): Set='{self.active_weight_set_name}', "
                   f"Indis(Actv/NaN): {active_indicator_count}/{nan_indicator_count}, WtSum: {total_weight_applied:.3f}, "
                   f"RawScore: {final_signal_score:.4f}, NormScore: {normalized_score:.4f}, Thresh: +/-{threshold:.3f} -> "
                   f"Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}")
        self.logger.info(log_msg)
        if nan_indicator_count > 0 or active_indicator_count == 0: self.logger.debug(f"  Detailed Scores: {debug_scores}")

        if final_signal in self.signals: self.signals[final_signal] = 1; self.signals["HOLD"] = 1 if final_signal=="HOLD" else 0
        return final_signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates quantized Entry, TP, and SL based on entry price, signal, ATR, config."""
        quantized_entry, take_profit, stop_loss = None, None, None
        if signal not in ["BUY", "SELL"] or not entry_price.is_finite() or entry_price <= 0:
            self.logger.error(f"Invalid input for TP/SL calc: Signal={signal}, Entry={entry_price}")
            return None, None, None

        # Quantize entry price first (use appropriate rounding for entry?) - Let's use ROUND_HALF_UP for entry
        quantized_entry = self.quantize_price(entry_price, rounding=ROUND_UP if signal == "BUY" else ROUND_DOWN) # Adjust entry slightly against signal? Or just quantize simply? Let's quantize simply first.
        quantized_entry = self.quantize_price(entry_price) # Simple quantize down
        if quantized_entry is None: self.logger.error(f"Failed to quantize entry price {entry_price}"); return None, None, None

        atr_val = self.indicator_values.get("ATR")
        if atr_val is None or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"Cannot calculate TP/SL for {self.symbol}: Invalid ATR ({atr_val})."); return quantized_entry, None, None

        try:
            atr = atr_val
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
            min_tick = self.get_min_tick_size()
            if min_tick is None: return quantized_entry, None, None # Should have failed earlier

            tp_offset = atr * tp_mult; sl_offset = atr * sl_mult
            min_sl_offset = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP) # Min distance SL should be

            # Ensure SL offset is at least minimum ticks away
            if sl_offset < min_sl_offset:
                self.logger.warning(f"Calculated SL offset ({sl_offset}) based on ATR is less than minimum {MIN_TICKS_AWAY_FOR_SLTP} ticks ({min_sl_offset}). Using minimum tick distance.")
                sl_offset = min_sl_offset

            if signal == "BUY":
                tp_raw = quantized_entry + tp_offset; sl_raw = quantized_entry - sl_offset
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_UP)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_DOWN)
            else: # SELL
                tp_raw = quantized_entry - tp_offset; sl_raw = quantized_entry + sl_offset
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_DOWN)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_UP)

            # --- Validation ---
            if stop_loss is not None:
                 # Ensure SL is strictly beyond entry by >= MIN_TICKS_AWAY_FOR_SLTP ticks
                 min_dist_from_entry = min_tick * Decimal(MIN_TICKS_AWAY_FOR_SLTP)
                 if signal == "BUY" and stop_loss >= quantized_entry - min_dist_from_entry + min_tick: # Allow SL to be exactly min_dist away
                     stop_loss = self.quantize_price(quantized_entry - min_dist_from_entry, rounding=ROUND_DOWN)
                     self.logger.debug(f"Adjusted BUY SL to be at least {MIN_TICKS_AWAY_FOR_SLTP} ticks away: {stop_loss}")
                 elif signal == "SELL" and stop_loss <= quantized_entry + min_dist_from_entry - min_tick:
                     stop_loss = self.quantize_price(quantized_entry + min_dist_from_entry, rounding=ROUND_UP)
                     self.logger.debug(f"Adjusted SELL SL to be at least {MIN_TICKS_AWAY_FOR_SLTP} ticks away: {stop_loss}")
                 if stop_loss <= 0: self.logger.error(f"Calculated SL <= 0 ({stop_loss}). SL=None."); stop_loss = None

            if take_profit is not None:
                 # Ensure TP is strictly beyond entry by >= 1 tick (less strict for TP)
                 if signal == "BUY" and take_profit <= quantized_entry: take_profit = self.quantize_price(quantized_entry + min_tick, rounding=ROUND_UP)
                 elif signal == "SELL" and take_profit >= quantized_entry: take_profit = self.quantize_price(quantized_entry - min_tick, rounding=ROUND_DOWN)
                 if take_profit <= 0: self.logger.error(f"Calculated TP <= 0 ({take_profit}). TP=None."); take_profit = None

            prec = self.get_price_precision_digits(); atr_log = f"{atr:.{prec+1}f}"
            tp_log = f"{take_profit:.{prec}f}" if take_profit else 'N/A'; sl_log = f"{stop_loss:.{prec}f}" if stop_loss else 'N/A'; entry_log = f"{quantized_entry:.{prec}f}"
            self.logger.info(f"Calc TP/SL ({signal}): Entry={entry_log}, TP={tp_log}, SL={sl_log} (ATR={atr_log})")
            return quantized_entry, take_profit, stop_loss

        except Exception as e: self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True); return quantized_entry, None, None


# --- Position Sizing ---
# calculate_position_size remains same as v1.0.1
def calculate_position_size(
    balance: Decimal, risk_per_trade: float, entry_price: Decimal, stop_loss_price: Decimal,
    market_info: Dict, leverage: int, logger: logging.Logger
) -> Optional[Decimal]:
    """Calculates position size based on balance, risk %, SL distance, market limits, and leverage."""
    lg = logger; symbol = market_info.get('symbol', 'N/A'); contract_size = market_info.get('contract_size', Decimal('1'))
    min_order_amount = market_info.get('min_order_amount'); min_order_cost = market_info.get('min_order_cost')
    amount_digits = market_info.get('amount_precision_digits'); is_contract = market_info.get('is_contract', False); is_inverse = market_info.get('inverse', False)

    if balance <= 0: lg.error(f"Size Calc Error ({symbol}): Balance <= 0 ({balance})."); return None
    if entry_price <= 0 or stop_loss_price <= 0: lg.error(f"Size Calc Error ({symbol}): Invalid Entry ({entry_price}) or SL ({stop_loss_price})."); return None
    if entry_price == stop_loss_price: lg.error(f"Size Calc Error ({symbol}): Entry == SL price."); return None
    if amount_digits is None: lg.error(f"Size Calc Error ({symbol}): Amount precision digits missing."); return None
    if not (0 < risk_per_trade < 1): lg.error(f"Size Calc Error ({symbol}): Invalid risk_per_trade ({risk_per_trade})."); return None
    if leverage <= 0 and is_contract: lg.error(f"Size Calc Error ({symbol}): Invalid leverage ({leverage}) for contract."); return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade)); sl_distance_points = abs(entry_price - stop_loss_price)
        size_unquantized = Decimal('NaN')

        if is_contract:
            if is_inverse:
                risk_per_contract_base = contract_size * abs(Decimal('1') / entry_price - Decimal('1') / stop_loss_price)
                risk_per_contract_quote = risk_per_contract_base * entry_price # Approx conversion
            else: # Linear
                risk_per_contract_quote = sl_distance_points * contract_size
            if risk_per_contract_quote <= 0: lg.error(f"Size Calc Error ({symbol}): Risk per contract <= 0 ({risk_per_contract_quote})."); return None
            size_unquantized = risk_amount_quote / risk_per_contract_quote
        else: # Spot
             if sl_distance_points <= 0: lg.error(f"Size Calc Error ({symbol}): SL distance is zero for spot."); return None
             size_unquantized = risk_amount_quote / sl_distance_points

        if not size_unquantized.is_finite() or size_unquantized <= 0: lg.error(f"Size Calc Error ({symbol}): Unquantized size invalid ({size_unquantized})."); return None
        lg.debug(f"Size Calc ({symbol}): Balance={balance:.2f}, RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_points}, UnquantSize={size_unquantized:.8f}")

        step_size = Decimal('1') / (Decimal('10') ** amount_digits)
        quantized_size = (size_unquantized / step_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * step_size
        quantized_size = Decimal(f"{quantized_size:.{amount_digits}f}")
        lg.debug(f"Quantized Size ({symbol}): {quantized_size} (Step: {step_size})")

        if quantized_size <= 0: lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Quantized size is {quantized_size}. Cannot trade.{RESET}"); return None
        if min_order_amount is not None and quantized_size < min_order_amount: lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Size {quantized_size} < Min Amount {min_order_amount}. Cannot trade.{RESET}"); return None

        order_value = quantized_size * entry_price * (contract_size if is_inverse else Decimal('1'))
        margin_required = order_value / Decimal(leverage) if is_contract and leverage > 0 else order_value
        if min_order_cost is not None and order_value < min_order_cost: lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Est. Value {order_value:.4f} < Min Cost {min_order_cost}. Cannot trade.{RESET}"); return None
        if margin_required > balance: lg.warning(f"{NEON_YELLOW}Size Calc ({symbol}): Est. Margin {margin_required:.4f} > Balance {balance:.4f}. Cannot trade.{RESET}"); return None

        lg.info(f"Calculated position size for {symbol}: {quantized_size}")
        return quantized_size
    except Exception as e: lg.error(f"{NEON_RED}Error calculating position size for {symbol}: {e}{RESET}", exc_info=True); return None


# --- CCXT Trading Action Wrappers ---
# (fetch_positions_ccxt, set_leverage_ccxt, create_order_ccxt, set_protection_ccxt, close_position_ccxt)
# Assume correct based on previous version v1.0.1 unless specific issues found.
def fetch_positions_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict) -> Optional[Dict]:
    """Fetches open position for a specific symbol using V5 API."""
    lg = logger; category = market_info.get('category'); market_id = market_info.get('id', symbol)
    if not category or category not in ['linear', 'inverse']: lg.debug(f"Skipping position check for non-derivative {symbol}"); return None
    if not exchange.has.get('fetchPositions'): lg.error(f"Exchange {exchange.id} does not support fetchPositions()."); return None

    try:
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")
        all_positions = safe_ccxt_call(exchange, 'fetch_positions', lg, symbols=[symbol], params=params)

        if all_positions is None: lg.warning(f"fetch_positions returned None for {symbol}."); return None
        if not isinstance(all_positions, list): lg.error(f"fetch_positions did not return a list for {symbol}."); return None

        for pos in all_positions:
            if pos.get('symbol') == symbol:
                try:
                     pos_size_str = pos.get('contracts', pos.get('info', {}).get('size'))
                     if pos_size_str is None: lg.warning(f"Position data for {symbol} missing size/contracts."); continue
                     pos_size = Decimal(str(pos_size_str))
                     if pos_size != 0:
                          pos_side = pos.get('side', 'long' if pos_size > 0 else 'short')
                          pos['side'] = pos_side; pos['contracts'] = abs(pos_size) # Standardize
                          lg.info(f"Found active {pos_side} position for {symbol}: Size={abs(pos_size)}, Entry={pos.get('entryPrice')}")
                          pos['market_info'] = market_info # Add market info for convenience
                          return pos
                     # else: lg.debug(f"Position found for {symbol} but size is 0.") # Ignore zero size positions
                except Exception as e: lg.error(f"Could not parse position data for {symbol}: {e}. Data: {pos}")
        lg.debug(f"No active non-zero position found for {symbol}.")
        return None
    except Exception as e: lg.error(f"{NEON_RED}Error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True); return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger, market_info: Dict) -> bool:
    """Sets leverage for a symbol using V5 API."""
    lg = logger; category = market_info.get('category'); market_id = market_info.get('id', symbol)
    if not category or category not in ['linear', 'inverse']: lg.debug(f"Skipping leverage set for non-derivative {symbol}"); return True
    if not exchange.has.get('setLeverage'): lg.error(f"Exchange {exchange.id} does not support setLeverage()."); return False

    try:
        params = {'category': category, 'symbol': market_id, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
        lg.info(f"Setting leverage for {symbol} (MarketID: {market_id}) to {leverage}x...")
        result = safe_ccxt_call(exchange, 'set_leverage', lg, leverage=leverage, symbol=symbol, params=params)
        lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x.{RESET}")
        return True
    except ccxt.ExchangeError as e:
         # Handle "Set leverage not modified" specifically by checking the exception from safe_ccxt_call
         # Note: safe_ccxt_call already handles code 110043 by returning {} which evaluates to True here.
         lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
         return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
        return False

def create_order_ccxt(
    exchange: ccxt.Exchange, symbol: str, order_type: str, side: str, amount: Decimal,
    price: Optional[Decimal] = None, params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """Creates an order using safe_ccxt_call with V5 parameters and Decimal inputs."""
    lg = logger or get_logger('main')
    if not market_info: lg.error(f"Market info required for create_order_ccxt ({symbol})"); return None
    category = market_info.get('category'); market_id = market_info.get('id', symbol)
    if not category: lg.error(f"Cannot determine category for {symbol}. Cannot place order."); return None
    if amount <= 0: lg.error(f"Order amount must be positive ({symbol}, Amount: {amount})"); return None

    price_digits = market_info.get('price_precision_digits', 8); amount_digits = market_info.get('amount_precision_digits', 8)
    price_str = f"{price:.{price_digits}f}" if order_type.lower() == 'limit' and price is not None and price > 0 else None
    amount_str = f"{amount:.{amount_digits}f}"
    if order_type.lower() == 'limit' and price_str is None: lg.error(f"Valid positive price required for limit order ({symbol})."); return None

    order_params = {'category': category}
    # --- Hedge Mode Logic Placeholder ---
    # position_mode = config.get("position_mode", "One-Way")
    # if position_mode == "Hedge":
    #     # Determine idx: 0=one-way, 1=buy hedge, 2=sell hedge
    #     order_params['positionIdx'] = 1 if side == 'buy' else 2
    # else: order_params['positionIdx'] = 0 # Explicitly 0 for One-Way
    # --- End Hedge Mode Logic ---
    if params: order_params.update(params)

    try:
        lg.info(f"Attempting to create {side.upper()} {order_type.upper()} order: {amount_str} {symbol} {'@ '+price_str if price_str else 'at Market'}")
        lg.debug(f"Final Order Params: {order_params}")
        amount_float = float(amount_str); price_float = float(price_str) if price_str else None

        order_result = safe_ccxt_call(exchange, 'create_order', lg, symbol=symbol, type=order_type, side=side, amount=amount_float, price=price_float, params=order_params)

        if order_result and order_result.get('id'):
            ret_code = order_result.get('info', {}).get('retCode', -1); ret_msg = order_result.get('info', {}).get('retMsg', 'Unknown'); order_id = order_result['id']
            if ret_code == 0:
                 lg.info(f"{NEON_GREEN}Successfully created {side} {order_type} order for {symbol}. Order ID: {order_id}{RESET}")
                 return order_result
            else:
                 lg.error(f"{NEON_RED}Order placement rejected ({symbol}). Code={ret_code}, Msg='{ret_msg}'. ID: {order_id}{RESET}"); return None
        elif order_result: lg.error(f"Order call success but no Order ID ({symbol}). Response: {order_result}"); return None
        else: lg.error(f"Order call returned None ({symbol}, after retries)."); return None
    except Exception as e: lg.error(f"{NEON_RED}Failed to create order ({symbol}): {e}{RESET}", exc_info=True); return None

def set_protection_ccxt(
    exchange: ccxt.Exchange, symbol: str, stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
    trailing_stop_price: Optional[Decimal] = None, trailing_active_price: Optional[Decimal] = None,
    position_idx: int = 0, # Required for hedge mode if used
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> bool:
    """Sets Stop Loss, Take Profit, and/or Trailing Stop using Bybit V5 position/trading-stop."""
    lg = logger or get_logger('main')
    if not market_info: lg.error(f"Market info required for set_protection_ccxt ({symbol})"); return False
    category = market_info.get('category'); market_id = market_info.get('id', symbol); price_digits = market_info.get('price_precision_digits', 8)
    if not category or category not in ['linear', 'inverse']: lg.warning(f"Cannot set SL/TP/TSL for non-derivative {symbol}"); return False # Changed to False

    params = {'category': category, 'symbol': market_id, 'tpslMode': 'Full'}
    # --- Hedge Mode Logic Placeholder ---
    # position_mode = config.get("position_mode", "One-Way")
    # if position_mode == "Hedge": params['positionIdx'] = position_idx
    # else: params['positionIdx'] = 0 # Explicitly 0 for One-Way
    # --- End Hedge Mode Logic ---

    def format_price(price: Optional[Decimal]) -> str:
        return f"{price:.{price_digits}f}" if price is not None and price.is_finite() and price > 0 else "0"

    sl_str, tp_str, tsl_dist_str, tsl_act_str = format_price(stop_loss_price), format_price(take_profit_price), format_price(trailing_stop_price), format_price(trailing_active_price)
    params['stopLoss'], params['takeProfit'], params['trailingStop'] = sl_str, tp_str, tsl_dist_str
    if tsl_dist_str != "0": params['activePrice'] = tsl_act_str # Only needed if TSL is active

    log_parts = []
    if sl_str != "0": log_parts.append(f"SL={sl_str}")
    if tp_str != "0": log_parts.append(f"TP={tp_str}")
    if tsl_dist_str != "0": log_parts.append(f"TSL_Dist={tsl_dist_str}" + (f", Act={tsl_act_str}" if tsl_act_str != "0" else ", Act=Immediate"))

    if not log_parts: lg.warning(f"No valid protection levels provided for set_protection_ccxt ({symbol})."); return True # Nothing to set = success?

    try:
        lg.info(f"Setting protection for {symbol} (MarketID: {market_id}): {', '.join(log_parts)}")
        lg.debug(f"Protection Params: {params}")
        method_to_call = 'private_post_position_trading_stop'
        result = safe_ccxt_call(exchange, method_to_call, lg, params=params)

        if result and result.get('retCode') == 0:
            lg.info(f"{NEON_GREEN}Successfully set protection for {symbol}.{RESET}"); return True
        elif result:
            ret_code = result.get('retCode', -1); ret_msg = result.get('retMsg', 'Unknown')
            lg.error(f"{NEON_RED}Failed set protection ({symbol}). Code={ret_code}, Msg='{ret_msg}'{RESET}"); return False
        else: lg.error(f"Set protection call failed/returned None ({symbol}, after retries)."); return False
    except Exception as e: lg.error(f"{NEON_RED}Failed to set protection ({symbol}): {e}{RESET}", exc_info=True); return False

def close_position_ccxt(
    exchange: ccxt.Exchange, symbol: str, position_data: Dict,
    logger: Optional[logging.Logger] = None, market_info: Optional[Dict] = None
) -> Optional[Dict]:
    """Closes an existing position by placing a market order in the opposite direction with reduceOnly."""
    lg = logger or get_logger('main')
    if not market_info: lg.error(f"Market info required for close_position_ccxt ({symbol})"); return None
    if not position_data: lg.error(f"Position data required for close_position_ccxt ({symbol})"); return None

    try:
        pos_size_str = position_data.get('contracts', position_data.get('info', {}).get('size'))
        position_side = position_data.get('side')
        if pos_size_str is None or position_side is None: lg.error(f"Missing size/side in position data for closing {symbol}."); return None

        position_size = Decimal(str(pos_size_str))
        amount_to_close = abs(position_size)
        if amount_to_close <= 0: lg.warning(f"Attempted close but size is {position_size} ({symbol})."); return None

        close_side = 'sell' if position_side == 'long' else 'buy'
        lg.info(f"Attempting to close {position_side} position ({symbol}, Size: {amount_to_close}) via {close_side} MARKET order...")

        params = {'reduceOnly': True}
        # --- Hedge Mode Logic Placeholder ---
        # position_mode = config.get("position_mode", "One-Way")
        # if position_mode == "Hedge":
        #      idx = position_data.get('info',{}).get('positionIdx')
        #      if idx is not None: params['positionIdx'] = idx
        #      else: lg.warning(f"Hedge mode: Could not get positionIdx for closing {symbol}")
        # --- End Hedge Mode Logic ---

        close_order = create_order_ccxt(exchange=exchange, symbol=symbol, order_type='market', side=close_side, amount=amount_to_close, params=params, logger=lg, market_info=market_info)

        if close_order and close_order.get('id'):
            lg.info(f"{NEON_GREEN}Successfully placed MARKET order to close position ({symbol}). Close Order ID: {close_order.get('id')}{RESET}")
            return close_order
        else: lg.error(f"{NEON_RED}Failed to place market order to close position ({symbol}). Check logs.{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Error attempting to close position ({symbol}): {e}{RESET}", exc_info=True); return None


# --- Main Bot Logic ---
def run_bot(exchange: ccxt.Exchange, config: Dict[str, Any], bot_state: Dict[str, Any]):
    """Main bot execution loop."""
    main_logger = get_logger('main')
    main_logger.info(f"{NEON_CYAN}=== Starting Enhanced Trading Bot v1.0.2 ==={RESET}")
    # Log key config settings
    main_logger.info(f"Config: Trading={'Enabled' if config.get('enable_trading') else 'DISABLED'}, "
                     f"Sandbox={'ACTIVE' if config.get('use_sandbox') else 'INACTIVE (LIVE!)'}, "
                     f"Symbols={config.get('symbols')}, Interval={config.get('interval')}, "
                     f"Risk={config.get('risk_per_trade')*100:.2f}%, Leverage={config.get('leverage')}x, "
                     f"MaxPos={config.get('max_concurrent_positions_total')}, Quote={QUOTE_CURRENCY}, "
                     f"Account={'UNIFIED' if IS_UNIFIED_ACCOUNT else 'Non-UTA'}, WeightSet='{config.get('active_weight_set')}'")

    global LOOP_DELAY_SECONDS
    LOOP_DELAY_SECONDS = config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS) # Already validated in load_config

    symbols_to_trade = config.get("symbols", []) # Already validated

    # Initialize/Validate state for each symbol
    for symbol in symbols_to_trade:
        if symbol not in bot_state: bot_state[symbol] = {}
        # Ensure default keys exist
        bot_state[symbol].setdefault("break_even_triggered", False)
        bot_state[symbol].setdefault("last_signal", "HOLD")
        bot_state[symbol].setdefault("last_entry_price", None) # Stored as string if set

    cycle_count = 0
    while True:
        cycle_count += 1
        start_time = time.time()
        main_logger.info(f"{NEON_BLUE}--- Starting Bot Cycle {cycle_count} ---{RESET}")

        # --- Pre-Cycle Checks ---
        current_balance: Optional[Decimal] = None
        if config.get("enable_trading"):
            try: current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
            except Exception as e: main_logger.error(f"Error fetching balance: {e}", exc_info=True)
            if current_balance is None: main_logger.error(f"{NEON_RED}Failed fetch balance. Trading actions skipped this cycle.{RESET}")

        open_positions_count = 0
        active_positions: Dict[str, Dict] = {}
        main_logger.debug("Fetching positions for all configured symbols...")
        for symbol in symbols_to_trade:
            temp_logger = get_logger(symbol, is_symbol_logger=True)
            market_info = get_market_info(exchange, symbol, temp_logger)
            if not market_info: temp_logger.error(f"Cannot fetch position for {symbol}: Failed market info."); continue
            if market_info.get('is_contract'):
                position = fetch_positions_ccxt(exchange, symbol, temp_logger, market_info)
                if position: # fetch_positions_ccxt already checks for non-zero size
                    open_positions_count += 1; active_positions[symbol] = position
                    # Update state entry price if missing
                    if bot_state[symbol].get("last_entry_price") is None and position.get('entryPrice'):
                         try: bot_state[symbol]["last_entry_price"] = str(Decimal(str(position['entryPrice'])))
                         except: pass # Ignore conversion errors
            # else: temp_logger.debug(f"Skipping position fetch for non-contract symbol: {symbol}")

        max_allowed_positions = config.get("max_concurrent_positions_total", 1)
        main_logger.info(f"Currently open positions: {open_positions_count} / {max_allowed_positions}")

        # --- Symbol Processing Loop ---
        for symbol in symbols_to_trade:
            symbol_logger = get_logger(symbol, is_symbol_logger=True)
            symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
            symbol_state = bot_state[symbol]

            try:
                # 1. Get Market Info
                market_info = get_market_info(exchange, symbol, symbol_logger)
                if not market_info: symbol_logger.error(f"Failed market info for {symbol}. Skipping."); continue

                # 2. Fetch Data
                timeframe = config.get("interval", "5") # Already validated
                periods = [int(config.get(p, d)) for p, d in [ ("atr_period", 14),("ema_long_period", 21),("rsi_period", 14),("bollinger_bands_period", 20),("cci_window", 20),("williams_r_window", 14),("mfi_window", 14),("stoch_rsi_window", 14),("stoch_rsi_rsi_window", 14),("sma_10_window", 10),("momentum_period", 7),("volume_ma_period", 15),("fibonacci_window", 50),] if int(config.get(p,d)) > 0]
                kline_limit = max(periods) + 50 if periods else 150

                df = fetch_klines_ccxt(exchange, symbol, timeframe, kline_limit, symbol_logger, market_info)
                if df.empty or len(df) < kline_limit / 2: symbol_logger.warning(f"Kline data insufficient ({len(df)} rows) for {symbol}. Skipping."); continue

                current_price_dec = fetch_current_price_ccxt(exchange, symbol, symbol_logger, market_info)
                if current_price_dec is None: symbol_logger.warning(f"Current price unavailable for {symbol}. Skipping."); continue

                orderbook = None
                if config.get("indicators", {}).get("orderbook"):
                    try: orderbook = fetch_orderbook_ccxt(exchange, symbol, int(config.get("orderbook_limit", 25)), symbol_logger, market_info)
                    except Exception as ob_err: symbol_logger.warning(f"Failed orderbook fetch for {symbol}: {ob_err}")

                # 3. Analyze Data
                analyzer = TradingAnalyzer(df, symbol_logger, config, market_info, symbol_state)

                # 4. Manage Existing Position
                current_position = active_positions.get(symbol)
                if current_position:
                    pos_side = current_position.get('side'); pos_size_str = current_position.get('contracts', '?')
                    symbol_logger.info(f"Managing existing {pos_side} position (Size: {pos_size_str}).")
                    manage_existing_position(exchange, config, symbol_logger, analyzer, current_position, current_price_dec)
                    # Position might have been closed by management logic, loop continues to next symbol

                # 5. Check for New Entry (if no position or within limits - contracts only for now)
                elif market_info.get('is_contract') and open_positions_count < max_allowed_positions:
                    symbol_logger.info("No active position. Checking for entry signals...")
                    if analyzer.break_even_triggered: analyzer.break_even_triggered = False # Reset state
                    if symbol_state.get("last_entry_price") is not None: symbol_state["last_entry_price"] = None

                    signal = analyzer.generate_trading_signal(current_price_dec, orderbook)
                    symbol_state["last_signal"] = signal # Store last signal

                    if signal in ["BUY", "SELL"]:
                        if config.get("enable_trading"):
                            if current_balance is not None and current_balance > 0:
                                 opened_new = attempt_new_entry(exchange, config, symbol_logger, analyzer, signal, current_price_dec, current_balance)
                                 if opened_new: open_positions_count += 1 # Increment count
                                 # State updated inside attempt_new_entry
                            else: symbol_logger.warning(f"Trading enabled but balance={current_balance}. Cannot enter {signal} trade.")
                        else: symbol_logger.info(f"Entry signal '{signal}' generated but trading is disabled.")

                elif current_position is None and market_info.get('is_contract'):
                     symbol_logger.info(f"Max concurrent positions ({open_positions_count}) reached. Skipping new entry check.")
                # --- Spot Market Placeholder ---
                # elif not market_info.get('is_contract'):
                #      symbol_logger.debug(f"Spot market {symbol}. Entry logic TBD.")
                #      # Add spot specific entry logic here if needed

            except Exception as e:
                symbol_logger.error(f"{NEON_RED}!!! Unhandled error in symbol loop for {symbol}: {e} !!!{RESET}", exc_info=True)

            symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
            time.sleep(0.2) # Small delay between symbols

        # --- Post-Cycle ---
        end_time = time.time(); cycle_duration = end_time - start_time
        main_logger.info(f"{NEON_BLUE}--- Bot Cycle {cycle_count} Finished (Duration: {cycle_duration:.2f}s) ---{RESET}")

        save_state(STATE_FILE, bot_state, main_logger) # Save state every cycle

        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0: main_logger.info(f"Waiting {wait_time:.2f}s for next cycle..."); time.sleep(wait_time)
        else: main_logger.warning(f"Cycle duration ({cycle_duration:.2f}s) exceeded loop delay ({LOOP_DELAY_SECONDS}s).")


def manage_existing_position(
    exchange: ccxt.Exchange, config: Dict[str, Any], logger: logging.Logger, analyzer: TradingAnalyzer,
    position_data: Dict, current_price_dec: Decimal
):
    """Handles logic for managing an open position (BE, MA Cross Exit)."""
    symbol = position_data.get('symbol'); position_side = position_data.get('side'); entry_price_str = position_data.get('entryPrice')
    pos_size_str = position_data.get('contracts', position_data.get('info', {}).get('size')); market_info = analyzer.market_info; symbol_state = analyzer.symbol_state
    if not all([symbol, position_side, entry_price_str, pos_size_str]): logger.error(f"Incomplete position data for management: {position_data}"); return

    try:
        entry_price = Decimal(str(entry_price_str)); position_size = Decimal(str(pos_size_str))
        if position_size <= 0: logger.warning(f"Position size is {position_size} for {symbol} in management."); return

        # --- 1. Check MA Cross Exit ---
        if config.get("enable_ma_cross_exit"):
            ema_s_f, ema_l_f = analyzer._get_indicator_float("EMA_Short"), analyzer._get_indicator_float("EMA_Long")
            if ema_s_f is not None and ema_l_f is not None:
                tolerance = 0.0001; is_adverse_cross = False
                if position_side == 'long' and ema_s_f < ema_l_f * (1 - tolerance): is_adverse_cross = True; msg = "Long: Short EMA below Long EMA"
                elif position_side == 'short' and ema_s_f > ema_l_f * (1 + tolerance): is_adverse_cross = True; msg = "Short: Short EMA above Long EMA"
                if is_adverse_cross:
                    logger.warning(f"{NEON_YELLOW}MA Cross Exit Triggered ({msg}). Closing position {symbol}.{RESET}")
                    if config.get("enable_trading"):
                        close_result = close_position_ccxt(exchange, symbol, position_data, logger, market_info)
                        if close_result:
                            symbol_state["break_even_triggered"] = False; symbol_state["last_signal"] = "HOLD"; symbol_state["last_entry_price"] = None
                            logger.info(f"Position close order placed for {symbol} due to MA Cross.")
                            return # Exit management logic
                        else: logger.error(f"Failed MA Cross close order placement for {symbol}.")

        # --- 2. Check Break-Even Trigger (Only if not already triggered) ---
        if config.get("enable_break_even") and not analyzer.break_even_triggered:
            atr_val = analyzer.indicator_values.get("ATR")
            if atr_val and atr_val.is_finite() and atr_val > 0:
                try:
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    profit_target_points = atr_val * trigger_multiple
                    current_profit_points = (current_price_dec - entry_price) if position_side == 'long' else (entry_price - current_price_dec)

                    if current_profit_points >= profit_target_points:
                        logger.info(f"{NEON_GREEN}Break-Even Triggered for {symbol}! Profit ({current_profit_points:.5f}) >= Target ({profit_target_points:.5f}){RESET}")
                        min_tick = analyzer.get_min_tick_size(); offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        if min_tick and min_tick > 0 and offset_ticks >= 0:
                            offset_value = min_tick * Decimal(offset_ticks); be_stop_price = None
                            if position_side == 'long':
                                be_stop_price = analyzer.quantize_price(entry_price + offset_value, rounding=ROUND_UP)
                                if be_stop_price and be_stop_price <= entry_price: be_stop_price = analyzer.quantize_price(entry_price + min_tick, rounding=ROUND_UP)
                            else: # SELL
                                be_stop_price = analyzer.quantize_price(entry_price - offset_value, rounding=ROUND_DOWN)
                                if be_stop_price and be_stop_price >= entry_price: be_stop_price = analyzer.quantize_price(entry_price - min_tick, rounding=ROUND_DOWN)

                            if be_stop_price and be_stop_price > 0:
                                logger.info(f"Calculated Break-Even Stop Price: {be_stop_price}")
                                if config.get("enable_trading"):
                                    pos_info = position_data.get('info', {})
                                    tp_str, tsl_dist_str, tsl_act_str = pos_info.get('takeProfit'), pos_info.get('trailingStop'), pos_info.get('activePrice')
                                    current_tp = Decimal(tp_str) if tp_str and tp_str != "0" else None
                                    current_tsl_dist = Decimal(tsl_dist_str) if tsl_dist_str and tsl_dist_str != "0" else None
                                    current_tsl_act = Decimal(tsl_act_str) if tsl_act_str and tsl_act_str != "0" else None
                                    use_tsl = config.get("enable_trailing_stop") and not config.get("break_even_force_fixed_sl")
                                    tsl_to_set = current_tsl_dist if use_tsl and current_tsl_dist else None
                                    act_to_set = current_tsl_act if use_tsl and tsl_to_set else None
                                    tp_to_set = current_tp
                                    logger.info(f"Setting BE SL={be_stop_price}" + (f", TP={tp_to_set}" if tp_to_set else "") + (f", TSL={tsl_to_set}" if tsl_to_set else "") + (f", removing TSL" if not tsl_to_set and current_tsl_dist else "") )
                                    pos_idx = pos_info.get('positionIdx', 0) # Needed for Hedge mode
                                    success = set_protection_ccxt(exchange, symbol, stop_loss_price=be_stop_price, take_profit_price=tp_to_set, trailing_stop_price=tsl_to_set, trailing_active_price=act_to_set, position_idx=pos_idx, logger=logger, market_info=market_info)
                                    if success: logger.info(f"{NEON_GREEN}Set BE SL for {symbol}.{RESET}"); analyzer.break_even_triggered = True
                                    else: logger.error(f"{NEON_RED}Failed BE SL set via API for {symbol}.{RESET}")
                            else: logger.error(f"Invalid BE stop price ({be_stop_price}). Cannot set BE.")
                        else: logger.error(f"Cannot calculate BE offset: Invalid tick ({min_tick}) or offset ({offset_ticks}).")
                except Exception as be_err: logger.error(f"Error during BE calc for {symbol}: {be_err}")
            else: logger.warning(f"Cannot check BE trigger for {symbol}: Invalid ATR ({atr_val}).")

    except Exception as e: logger.error(f"Unexpected error managing position {symbol}: {e}", exc_info=True)

def attempt_new_entry(
    exchange: ccxt.Exchange, config: Dict[str, Any], logger: logging.Logger, analyzer: TradingAnalyzer,
    signal: str, entry_price_signal: Decimal, current_balance: Decimal
) -> bool:
    """Attempts to calculate size, set leverage, place order, and set protection."""
    symbol = analyzer.symbol; market_info = analyzer.market_info; symbol_state = analyzer.symbol_state
    logger.info(f"Attempting {signal} entry for {symbol} @ signal price {entry_price_signal:.{analyzer.get_price_precision_digits()}f}")

    # 1. Calculate Quantized Entry, TP, SL
    quantized_entry, take_profit_price, stop_loss_price = analyzer.calculate_entry_tp_sl(entry_price_signal, signal)
    if not quantized_entry: logger.error(f"Cannot enter {signal} ({symbol}): Failed entry quantization."); return False
    if not stop_loss_price: logger.error(f"Cannot enter {signal} ({symbol}): Failed SL calculation."); return False

    # 2. Calculate Position Size
    risk = float(config.get("risk_per_trade", 0.01)); leverage = int(config.get("leverage", 10))
    position_size = calculate_position_size(current_balance, risk, quantized_entry, stop_loss_price, market_info, leverage, logger)
    if not position_size or position_size <= 0: logger.error(f"Cannot enter {signal} ({symbol}): Invalid size ({position_size})."); return False

    # 3. Set Leverage (Contracts only)
    if market_info.get('is_contract'):
        if not set_leverage_ccxt(exchange, symbol, leverage, logger, market_info): logger.error(f"Failed leverage set for {symbol}. Aborting entry."); return False

    # 4. Place Entry Order
    side = 'buy' if signal == 'BUY' else 'sell'; entry_order_params = {}
    # Add positionIdx for Hedge Mode if needed
    entry_order = create_order_ccxt(exchange, symbol, 'market', side, position_size, params=entry_order_params, logger=logger, market_info=market_info)
    if not entry_order or not entry_order.get('id'): logger.error(f"Failed entry market order placement for {symbol}. Aborting."); return False

    order_id = entry_order['id']
    logger.info(f"Entry order ({order_id}) placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s...")
    time.sleep(POSITION_CONFIRM_DELAY)

    # 5. Fetch Actual Entry Price (Optional but recommended)
    actual_entry_price = quantized_entry # Default to calculated
    try:
        logger.debug(f"Re-fetching position for {symbol} to confirm entry...")
        time.sleep(1) # Extra delay
        updated_position = fetch_positions_ccxt(exchange, symbol, logger, market_info)
        if updated_position and updated_position.get('entryPrice'):
             entry_p_str = updated_position.get('entryPrice'); actual_entry_price = Decimal(str(entry_p_str))
             current_size_str = updated_position.get('contracts', updated_position.get('info',{}).get('size'))
             filled_size = abs(Decimal(str(current_size_str))) if current_size_str else Decimal('0')
             logger.info(f"Confirmed position. Actual Entry: {actual_entry_price}, Size: {filled_size}")
             # Check fill deviation
             if abs(filled_size - position_size) / position_size > Decimal('0.01'): logger.warning(f"Fill size {filled_size} differs >1% from ordered {position_size}.")
        else: logger.warning(f"Could not confirm entry via position fetch. Using calculated entry: {actual_entry_price}.")
    except Exception as confirm_err: logger.error(f"Error confirming entry: {confirm_err}. Using calculated entry: {actual_entry_price}.")

    # 6. Set SL/TP/TSL Protection
    tsl_distance, tsl_activation_price = None, None
    if config.get("enable_trailing_stop"):
        try:
            cb_rate = Decimal(str(config.get("trailing_stop_callback_rate"))); act_perc = Decimal(str(config.get("trailing_stop_activation_percentage")))
            min_tick = analyzer.get_min_tick_size()
            if cb_rate > 0 and min_tick:
                tsl_dist_raw = actual_entry_price * cb_rate
                tsl_distance = (tsl_dist_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                if tsl_distance < min_tick: tsl_distance = min_tick; logger.warning(f"TSL dist adjusted to min tick: {tsl_distance}")
                if act_perc > 0:
                    offset = actual_entry_price * act_perc; act_raw = actual_entry_price + offset if signal == "BUY" else actual_entry_price - offset
                    rounding = ROUND_UP if signal == "BUY" else ROUND_DOWN
                    tsl_activation_price = analyzer.quantize_price(act_raw, rounding=rounding)
                    # Ensure activation price is valid
                    if tsl_activation_price and tsl_activation_price <= 0: tsl_activation_price = None
                    if signal == "BUY" and tsl_activation_price and tsl_activation_price <= actual_entry_price: tsl_activation_price = analyzer.quantize_price(actual_entry_price + min_tick, ROUND_UP)
                    if signal == "SELL" and tsl_activation_price and tsl_activation_price >= actual_entry_price: tsl_activation_price = analyzer.quantize_price(actual_entry_price - min_tick, ROUND_DOWN)
                logger.debug(f"Calc TSL: Dist={tsl_distance}, Act={tsl_activation_price} (Entry: {actual_entry_price})")
            else: logger.warning(f"Cannot calc TSL dist for {symbol}: Rate={cb_rate}, Tick={min_tick}")
        except Exception as tsl_err: logger.error(f"Error calc TSL params for {symbol}: {tsl_err}", exc_info=True)

    pos_idx = 0 # Default for One-Way
    # Add hedge mode logic for positionIdx if needed
    protection_set = set_protection_ccxt(exchange, symbol, stop_loss_price=stop_loss_price, take_profit_price=take_profit_price, trailing_stop_price=tsl_distance, trailing_active_price=tsl_activation_price, position_idx=pos_idx, logger=logger, market_info=market_info)

    if not protection_set:
        logger.error(f"{NEON_RED}Failed to set initial SL/TP/TSL for {symbol}! Position might be unprotected.{RESET}")
        if config.get("enable_trading"):
            logger.warning(f"Attempting emergency close of unprotected position {symbol}...")
            pos_to_close = fetch_positions_ccxt(exchange, symbol, logger, market_info)
            if pos_to_close: close_position_ccxt(exchange, symbol, pos_to_close, logger, market_info)
            else: logger.error(f"Could not fetch position for emergency close {symbol}.")
        return False # Entry failed

    logger.info(f"{NEON_GREEN}Successfully entered {signal} trade for {symbol} with protection.{RESET}")
    symbol_state["break_even_triggered"] = False # Reset BE state
    symbol_state["last_entry_price"] = str(actual_entry_price) # Store actual entry as string
    return True


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Bybit V5 Multi-Symbol Trading Bot v1.0.2")
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help=f"Path to config file (default: {CONFIG_FILE})")
    parser.add_argument("--state", type=str, default=STATE_FILE, help=f"Path to state file (default: {STATE_FILE})")
    parser.add_argument("--symbols", type=str, help="Override config symbols (comma-separated)")
    parser.add_argument("--live", action="store_true", help="Enable live trading (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG console logging")
    args = parser.parse_args()

    console_log_level = logging.DEBUG if args.debug else logging.INFO
    if args.debug: print("DEBUG logging enabled.")

    main_logger = get_logger('main') # Setup main logger with correct level
    main_logger.info(f" --- Bot Starting {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} --- ")

    config = load_config(args.config, main_logger)
    if config is None: # Validation failed
        sys.exit(1)

    if args.symbols: # Override symbols
        override_symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        if override_symbols: main_logger.warning(f"{NEON_YELLOW}Overriding config symbols. Trading ONLY: {override_symbols}{RESET}"); config["symbols"] = override_symbols
        else: main_logger.error("Empty symbol list from --symbols arg."); sys.exit(1)

    if args.live: # Override live/sandbox mode
        main_logger.warning(f"{NEON_RED}--- LIVE TRADING ENABLED via command line! ---{RESET}")
        if not config.get("enable_trading"): main_logger.warning("Overriding config 'enable_trading=false'.")
        if config.get("use_sandbox"): main_logger.warning("Overriding config 'use_sandbox=true'.")
        config["enable_trading"] = True; config["use_sandbox"] = False

    # Log final modes after overrides
    if config.get("enable_trading"): main_logger.warning(f"{NEON_RED}--- Live trading is ENABLED ---{RESET}")
    else: main_logger.info("Live trading is DISABLED.")
    if config.get("use_sandbox"): main_logger.warning(f"{NEON_YELLOW}Sandbox mode (testnet) is ACTIVE.{RESET}")
    else: main_logger.warning(f"{NEON_RED}Sandbox mode is INACTIVE (LIVE exchange).{RESET}")

    bot_state = load_state(args.state, main_logger)
    exchange = initialize_exchange(config, main_logger)

    if exchange:
        main_logger.info(f"{NEON_GREEN}Exchange initialized successfully. Starting main loop...{RESET}")
        try: run_bot(exchange, config, bot_state)
        except KeyboardInterrupt: main_logger.info("Bot stopped by user (KeyboardInterrupt).")
        except Exception as e: main_logger.critical(f"{NEON_RED}!!! BOT CRASHED: {e} !!!{RESET}", exc_info=True)
        finally:
            main_logger.info("Attempting to save final state...")
            save_state(args.state, bot_state, main_logger)
            main_logger.info(f"--- Bot Shutdown {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    else: main_logger.critical("Failed to initialize exchange. Bot cannot start.")

    logging.shutdown()
    print("Bot execution finished.")
    sys.exit(0 if exchange else 1)
