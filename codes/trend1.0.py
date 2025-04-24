# merged_bot.py
# Merged version combining features from livebot7.py (indicator scoring) and volbot5.py (trend/OB)
# Supports both LiveXY and Volbot strategies, configurable via config.json.
# Includes enhanced risk management (ATR-based SL/TP, TSL, BE).

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta  # Ensure pandas_ta is installed: pip install pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo # Use zoneinfo (Python 3.9+) for robust timezone handling

# Initialize colorama and set Decimal precision for financial calculations
init(autoreset=True)
getcontext().prec = 28  # Sufficient precision for most crypto calculations

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# Color Scheme (Consolidated & Consistent)
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN # Duplicate of NEON_BLUE, kept for compatibility if used elsewhere
RESET = Style.RESET_ALL

# Semantic Colors & Log Levels
COLOR_UP = Fore.CYAN + Style.BRIGHT       # Volbot trend up
COLOR_DN = Fore.YELLOW + Style.BRIGHT     # Volbot trend down
COLOR_BULL_BOX = Fore.GREEN               # Volbot bullish OB
COLOR_BEAR_BOX = Fore.RED                 # Volbot bearish OB
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX    # Volbot closed OB
COLOR_INFO = Fore.MAGENTA                 # General info messages
COLOR_HEADER = Fore.BLUE + Style.BRIGHT   # Section headers
COLOR_WARNING = NEON_YELLOW               # Warnings
COLOR_ERROR = NEON_RED                    # Errors
COLOR_SUCCESS = NEON_GREEN                # Success messages (e.g., order filled)

# API Credentials (Loaded from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    print(f"{COLOR_ERROR}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    # Consider raising a more specific error or using logger if available
    raise ValueError("API Key/Secret not found in environment variables.")

# File/Directory Configuration
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True) # Ensure log directory exists

# Time & Retry Configuration
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default timezone if not in config or invalid
try:
    # Attempt to load default timezone immediately to catch errors early
    _default_tz = ZoneInfo(DEFAULT_TIMEZONE_STR)
except Exception as tz_err:
    print(f"{COLOR_ERROR}CRITICAL: Default timezone '{DEFAULT_TIMEZONE_STR}' invalid or system tzdata missing: {tz_err}. Check tz database installation. Exiting.{RESET}")
    exit(1)
TIMEZONE = _default_tz # Set global timezone, will be updated by config load

MAX_API_RETRIES = 3             # Max number of retries for failed API calls
RETRY_DELAY_SECONDS = 5         # Base delay between retries (may increase)
LOOP_DELAY_SECONDS = 15         # Minimum time between the *end* of one full cycle and the start of the next
POSITION_CONFIRM_DELAY = 10     # Seconds to wait after placing entry order before setting protection

# Interval Configuration (Mapping user input to CCXT format)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# API Error Codes for Retry Logic (Common transient HTTP status codes)
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]

# --- Default Indicator/Strategy Parameters (can be overridden by config.json) ---
# LiveXY Defaults (Strategy 1)
DEFAULT_ATR_PERIOD_LIVEXY = 14          # ATR period for LiveXY strategy (if used)
DEFAULT_CCI_WINDOW = 20                 # CCI period
DEFAULT_WILLIAMS_R_WINDOW = 14          # Williams %R period
DEFAULT_MFI_WINDOW = 14                 # Money Flow Index period
DEFAULT_STOCH_RSI_WINDOW = 14           # Stochastic RSI period
DEFAULT_STOCH_WINDOW = 12               # Stochastic %K period (used within StochRSI)
DEFAULT_K_WINDOW = 3                    # Stochastic RSI %K smoothing
DEFAULT_D_WINDOW = 3                    # Stochastic RSI %D smoothing
DEFAULT_RSI_WINDOW = 14                 # RSI period
DEFAULT_BOLLINGER_BANDS_PERIOD = 20     # Bollinger Bands period
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0   # Bollinger Bands standard deviation
DEFAULT_SMA_10_WINDOW = 10              # Simple Moving Average period
DEFAULT_EMA_SHORT_PERIOD = 9            # Short Exponential Moving Average period
DEFAULT_EMA_LONG_PERIOD = 21            # Long Exponential Moving Average period
DEFAULT_MOMENTUM_PERIOD = 7             # Momentum period
DEFAULT_VOLUME_MA_PERIOD = 15           # Volume Moving Average period
DEFAULT_FIB_WINDOW = 50                 # Window for Fibonacci calculation (Note: Not actively used in signal logic yet)
DEFAULT_PSAR_AF = 0.02                  # Parabolic SAR Acceleration Factor start
DEFAULT_PSAR_MAX_AF = 0.2               # Parabolic SAR Acceleration Factor max
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels (for potential future use)

# Volbot Defaults (Strategy 2)
DEFAULT_VOLBOT_LENGTH = 40              # Primary EMA length for Volbot trend
DEFAULT_VOLBOT_ATR_LENGTH = 200         # ATR length for Volbot volatility levels
DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK = 1000 # Lookback for volume normalization max
DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE = 100 # Percentile for volume normalization (100=max)
DEFAULT_VOLBOT_OB_SOURCE = "Wicks"      # Source for Order Block detection: "Wicks" or "Bodys"
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H = 25    # Lookback bars for Pivot High
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H = 25   # Lookforward bars for Pivot High
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L = 25    # Lookback bars for Pivot Low
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L = 25   # Lookforward bars for Pivot Low
DEFAULT_VOLBOT_MAX_BOXES = 50           # Max number of active OBs to track per side

# Default Risk Management Parameters (can be overridden by config.json)
DEFAULT_ATR_PERIOD_RISK = 14            # Default ATR period *specifically* for Risk Management (SL/TP/BE) calculations

# Global QUOTE_CURRENCY placeholder, dynamically loaded from config
QUOTE_CURRENCY = "USDT" # Default fallback, updated by load_config()

# Default console log level (updated by config)
# Use a mutable container like a list to allow setup_logger to update it
_log_level_container = {'level': logging.INFO}

# --- Logger Setup (Enhanced from volbot5) ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that redacts sensitive information like API keys/secrets from log messages."""
    REDACTION_STR = "***REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive info if present."""
        msg = super().format(record)
        # Ensure API keys exist before attempting replacement
        if API_KEY and API_KEY in msg:
            msg = msg.replace(API_KEY, self.REDACTION_STR)
        if API_SECRET and API_SECRET in msg:
            msg = msg.replace(API_SECRET, self.REDACTION_STR)
        return msg

def setup_logger(name_suffix: str) -> logging.Logger:
    """
    Sets up a logger instance with a specified suffix.
    Includes file rotation, colored console output, and sensitive data redaction.
    Prevents adding duplicate handlers and updates console level based on global setting.
    """
    global _log_level_container
    # Sanitize suffix for use in filenames and logger names
    safe_suffix = re.sub(r'[^\w\-]+', '_', name_suffix)
    logger_name = f"merged_bot_{safe_suffix}" # Consistent logger naming
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # If logger already configured, just update console level if needed and return
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                current_console_level = _log_level_container['level']
                if handler.level != current_console_level:
                    logger.debug(f"Updating console handler level for {logger_name} to {logging.getLevelName(current_console_level)}")
                    handler.setLevel(current_console_level)
        return logger

    # Set the base level for the logger (captures all messages >= DEBUG)
    logger.setLevel(logging.DEBUG)

    # --- File Handler ---
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
    except Exception as e:
        # Use print as logger might not be fully set up
        print(f"{COLOR_ERROR}Error setting up file logger for {log_filename}: {e}{RESET}")

    # --- Stream Handler (Console) ---
    stream_handler = logging.StreamHandler()
    level_colors = {
        logging.DEBUG: NEON_BLUE, logging.INFO: NEON_GREEN, logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED, logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    class ColorFormatter(SensitiveFormatter):
        """Custom formatter for colored console output."""
        def format(self, record):
            log_color = level_colors.get(record.levelno, RESET) # Default to reset if level unknown
            record.levelname = f"{log_color}{record.levelname:<8}{RESET}" # Padded level name
            record.asctime = f"{NEON_BLUE}{self.formatTime(record, self.datefmt)}{RESET}"
            # Attempt to extract a more meaningful name part (e.g., symbol or 'init')
            try:
                 # Adjusted split index to handle names like 'merged_bot_BTC_USDT' or 'merged_bot_init'
                base_name = record.name.split('_', 2)[-1] if record.name.count('_') >= 2 else record.name.split('_')[-1]
            except IndexError:
                base_name = record.name # Fallback
            record.name_part = f"{NEON_PURPLE}[{base_name}]{RESET}"
            # Apply color to the message itself only for warning and above
            message_color = log_color if record.levelno >= logging.WARNING else ""
            record.msg = f"{message_color}{record.getMessage()}{RESET}"
            # Use the parent SensitiveFormatter to format the basic string, then apply colors
            formatted_message = super(SensitiveFormatter, self).format(record)
            # Final reset to ensure no color bleed
            return f"{formatted_message}{RESET}"

    stream_formatter = ColorFormatter(
        # Format: Time - Level - [NamePart] - Message
        "%(asctime)s - %(levelname)s - %(name_part)s - %(message)s",
        datefmt='%H:%M:%S' # Concise time format for console
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(_log_level_container['level']) # Set level from global container
    logger.addHandler(stream_handler)

    # Prevent messages from propagating to the root logger
    logger.propagate = False
    return logger

# --- Configuration Loading (Merged & Enhanced) ---
def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    Creates a default config if the file doesn't exist.
    Ensures all default keys are present in the loaded config, adding defaults if missing.
    Updates global settings like QUOTE_CURRENCY, TIMEZONE, and console_log_level.
    Performs basic validation on critical config values.
    """
    global QUOTE_CURRENCY, TIMEZONE, _log_level_container

    # Define the structure and default values for the configuration
    default_config = {
        # --- General Bot Settings ---
        "timezone": DEFAULT_TIMEZONE_STR,           # Timezone for displaying timestamps
        "interval": "5",                            # Kline interval (e.g., "1", "5", "15", "60", "D")
        "retry_delay": RETRY_DELAY_SECONDS,         # Base delay for API retries
        "enable_trading": False,                    # Master switch for placing real orders
        "use_sandbox": True,                        # Use exchange's testnet/sandbox environment
        "risk_per_trade": 0.01,                     # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 10,                             # Desired leverage for futures/margin trading
        "max_concurrent_positions": 1,              # (Currently informational) Max simultaneous positions
        "quote_currency": "USDT",                   # The currency positions are denominated in (e.g., USDT, USD)
        "console_log_level": "INFO",                # Logging level for console output (DEBUG, INFO, WARNING, ERROR)
        "signal_mode": "both_align",                # How to combine signals: 'livexy', 'volbot', 'both_align'

        # --- Risk Management Settings ---
        "atr_period_risk": DEFAULT_ATR_PERIOD_RISK, # ATR period specifically for SL/TP/BE calculations
        "stop_loss_multiple": 1.8,                  # ATR multiple for initial Stop Loss distance
        "take_profit_multiple": 0.7,                # ATR multiple for initial Take Profit distance (0 = no TP)
        "enable_trailing_stop": True,               # Enable Trailing Stop Loss feature
        # TSL callback rate: Bybit accepts percentage (e.g., "0.5%") or price distance (e.g., "50")
        # Use string for flexibility. Bot logic assumes percentage or decimal fraction for now.
        "trailing_stop_callback_rate": "0.5%",      # TSL trail distance/percentage (e.g., "0.005" or "0.5%")
        "trailing_stop_activation_percentage": 0.003, # Profit % required to activate TSL (e.g., 0.003 = 0.3%)
        "enable_break_even": True,                  # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0,     # ATR multiple of profit needed to trigger BE
        "break_even_offset_ticks": 2,               # Number of minimum ticks above/below entry for BE SL

        # --- LiveXY Strategy Settings ---
        "livexy_enabled": True,                     # Master switch for LiveXY strategy calculations & signals
        "livexy_atr_period": DEFAULT_ATR_PERIOD_LIVEXY, # ATR period for LiveXY (if used by indicators)
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD, # LiveXY EMA short period
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,   # LiveXY EMA long period
        "rsi_period": DEFAULT_RSI_WINDOW,           # LiveXY RSI period
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD, # LiveXY BBands period
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV, # LiveXY BBands std dev
        "cci_window": DEFAULT_CCI_WINDOW,           # LiveXY CCI period
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW, # LiveXY Williams %R period
        "mfi_window": DEFAULT_MFI_WINDOW,           # LiveXY MFI period
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, # LiveXY StochRSI period
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW, # LiveXY StochRSI's internal RSI period
        "stoch_rsi_k": DEFAULT_K_WINDOW,            # LiveXY StochRSI %K smoothing
        "stoch_rsi_d": DEFAULT_D_WINDOW,            # LiveXY StochRSI %D smoothing
        "psar_af": DEFAULT_PSAR_AF,                 # LiveXY PSAR acceleration factor start
        "psar_max_af": DEFAULT_PSAR_MAX_AF,         # LiveXY PSAR acceleration factor max
        "sma_10_window": DEFAULT_SMA_10_WINDOW,     # LiveXY SMA period
        "momentum_period": DEFAULT_MOMENTUM_PERIOD, # LiveXY Momentum period
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD, # LiveXY Volume MA period
        "orderbook_limit": 25,                      # Number of levels to fetch for order book analysis
        "signal_score_threshold": 1.5,              # Absolute weighted score needed for LiveXY BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25,         # LiveXY StochRSI oversold level
        "stoch_rsi_overbought_threshold": 75,       # LiveXY StochRSI overbought level
        "volume_confirmation_multiplier": 1.5,      # LiveXY: Volume must be X times MA for confirmation signal
        "fibonacci_window": DEFAULT_FIB_WINDOW,     # (Currently informational) Lookback for potential Fib levels
        "indicators": {                             # Control which LiveXY indicators are calculated & weighted
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": {                            # Define different weighting schemes for LiveXY
            "scalping": { # Example: Weights favoring short-term signals
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Example: More balanced weights
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
            # Add more weight sets as needed
        },
        "active_weight_set": "default",             # Which LiveXY weight set defined above to use

        # --- Volbot Strategy Settings ---
        "volbot_enabled": True,                     # Master switch for Volbot strategy calculations & signals
        "volbot_length": DEFAULT_VOLBOT_LENGTH,     # Volbot primary EMA length
        "volbot_atr_length": DEFAULT_VOLBOT_ATR_LENGTH, # Volbot ATR length
        "volbot_volume_percentile_lookback": DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK, # Volbot volume norm lookback
        "volbot_volume_normalization_percentile": DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE, # Volbot volume norm percentile
        "volbot_ob_source": DEFAULT_VOLBOT_OB_SOURCE, # Volbot OB source ("Wicks" or "Bodys")
        "volbot_pivot_left_len_h": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H,   # Volbot Pivot High lookback
        "volbot_pivot_right_len_h": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H, # Volbot Pivot High lookforward
        "volbot_pivot_left_len_l": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L,   # Volbot Pivot Low lookback
        "volbot_pivot_right_len_l": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L, # Volbot Pivot Low lookforward
        "volbot_max_boxes": DEFAULT_VOLBOT_MAX_BOXES, # Volbot max active OBs
        "volbot_signal_on_trend_flip": True,        # Volbot: Generate signal when trend direction changes
        "volbot_signal_on_ob_entry": True,          # Volbot: Generate signal when price enters a relevant OB
    }

    loaded_config = {}
    config_updated = False # Flag to track if defaults were added

    # Check if config file exists
    if not os.path.exists(filepath):
        print(f"{COLOR_WARNING}Config file not found at '{filepath}'. Creating default config...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            print(f"{COLOR_SUCCESS}Created default config file: {filepath}{RESET}")
            loaded_config = default_config.copy() # Use defaults directly
        except IOError as e:
            print(f"{COLOR_ERROR}CRITICAL: Error creating default config file {filepath}: {e}. Using built-in defaults. Check permissions.{RESET}")
            loaded_config = default_config.copy() # Fallback to in-memory defaults
    else:
        # Config file exists, try to load it
        try:
            with open(filepath, 'r', encoding="utf-8") as f:
                loaded_config_from_file = json.load(f)
            # Ensure all keys from default_config exist, add if missing
            loaded_config, config_updated = _ensure_config_keys(loaded_config_from_file, default_config)
            if config_updated:
                print(f"{COLOR_WARNING}Config file '{filepath}' was missing some keys. Defaults added.{RESET}")
                # Attempt to write the updated config back to the file
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(loaded_config, f_write, indent=4, sort_keys=True)
                    print(f"{COLOR_INFO}Updated config file '{filepath}' saved successfully.{RESET}")
                except IOError as e:
                    print(f"{COLOR_ERROR}Error writing updated config file {filepath}: {e}. Proceeding with updated config in memory.{RESET}")
        except FileNotFoundError:
             # This case should be handled by the initial os.path.exists check, but included for robustness
             print(f"{COLOR_ERROR}Config file {filepath} disappeared unexpectedly. Using defaults.{RESET}")
             loaded_config = default_config.copy()
        except json.JSONDecodeError as e:
            print(f"{COLOR_ERROR}Error decoding JSON from config file {filepath}: {e}. File might be corrupted.{RESET}")
            print(f"{COLOR_WARNING}Using default configuration and attempting to recreate the config file.{RESET}")
            loaded_config = default_config.copy()
            try:
                # Attempt to overwrite the corrupted file with defaults
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4, sort_keys=True)
                print(f"{COLOR_WARNING}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create:
                print(f"{COLOR_ERROR}Error recreating config file after JSON error: {e_create}{RESET}")
        except Exception as e:
            # Catch any other unexpected errors during loading/processing
            print(f"{COLOR_ERROR}Unexpected error loading config {filepath}: {e}. Using defaults.{RESET}")
            loaded_config = default_config.copy()

    # --- Update global settings based on loaded/default config ---
    # Update Quote Currency
    new_quote_currency = loaded_config.get("quote_currency", default_config["quote_currency"]).upper()
    if new_quote_currency != QUOTE_CURRENCY:
        print(f"{COLOR_INFO}Setting QUOTE_CURRENCY to: {new_quote_currency}{RESET}")
        QUOTE_CURRENCY = new_quote_currency

    # Update Console Log Level
    level_name = loaded_config.get("console_log_level", "INFO").upper()
    new_log_level = getattr(logging, level_name, logging.INFO) # Fallback to INFO if invalid
    if new_log_level != _log_level_container['level']:
        print(f"{COLOR_INFO}Setting console log level to: {level_name}{RESET}")
        _log_level_container['level'] = new_log_level
        # Note: Existing loggers' console handlers will be updated by setup_logger() on next call

    # Update Timezone
    config_tz_str = loaded_config.get("timezone", DEFAULT_TIMEZONE_STR)
    try:
        new_tz = ZoneInfo(config_tz_str)
        # Check if timezone actually changed or was initialized
        if TIMEZONE is None or new_tz.key != TIMEZONE.key:
            print(f"{COLOR_INFO}Setting timezone to: {config_tz_str}{RESET}")
            TIMEZONE = new_tz
    except Exception as tz_err:
        print(f"{COLOR_ERROR}Invalid timezone '{config_tz_str}' in config: {tz_err}. Using previous/default '{TIMEZONE.key}'.{RESET}")
        # Correct the config value in memory to prevent repeated errors
        loaded_config["timezone"] = TIMEZONE.key

    # --- Validate specific critical config values ---
    # Interval
    if loaded_config.get("interval") not in VALID_INTERVALS:
        print(f"{COLOR_ERROR}Invalid 'interval': '{loaded_config.get('interval')}' in config. Using default '{default_config['interval']}'.{RESET}")
        loaded_config["interval"] = default_config["interval"]
    # Volbot OB Source
    if loaded_config.get("volbot_ob_source") not in ["Wicks", "Bodys"]:
        print(f"{COLOR_ERROR}Invalid 'volbot_ob_source': '{loaded_config.get('volbot_ob_source')}' in config. Using default '{default_config['volbot_ob_source']}'.{RESET}")
        loaded_config["volbot_ob_source"] = default_config["volbot_ob_source"]
    # Signal Mode
    if loaded_config.get("signal_mode") not in ['livexy', 'volbot', 'both_align']:
        print(f"{COLOR_ERROR}Invalid 'signal_mode': '{loaded_config.get('signal_mode')}' in config. Using default '{default_config['signal_mode']}'.{RESET}")
        loaded_config["signal_mode"] = default_config["signal_mode"]
    # Risk Per Trade (ensure it's a valid percentage)
    try:
        risk_val = float(loaded_config.get("risk_per_trade", default_config["risk_per_trade"]))
        if not (0 < risk_val < 1):
            raise ValueError("Risk per trade must be between 0 and 1 (exclusive).")
        loaded_config["risk_per_trade"] = risk_val # Store as float
    except (ValueError, TypeError) as e:
        print(f"{COLOR_ERROR}Invalid 'risk_per_trade': '{loaded_config.get('risk_per_trade')}'. Error: {e}. Using default {default_config['risk_per_trade']}.{RESET}")
        loaded_config["risk_per_trade"] = default_config["risk_per_trade"]
    # Leverage (ensure positive integer)
    try:
        lev = int(loaded_config.get("leverage", default_config["leverage"]))
        if lev <= 0: raise ValueError("Leverage must be positive.")
        loaded_config["leverage"] = lev
    except (ValueError, TypeError) as e:
        print(f"{COLOR_ERROR}Invalid 'leverage': '{loaded_config.get('leverage')}'. Error: {e}. Using default {default_config['leverage']}.{RESET}")
        loaded_config["leverage"] = default_config["leverage"]
    # Add more validations as needed for other critical parameters...

    return loaded_config

def _ensure_config_keys(loaded_config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures that all keys from the default_config exist in the loaded_config.
    If a key is missing in loaded_config, it's added with the value from default_config.
    Handles nested dictionaries. Returns the potentially modified loaded_config and a boolean
    indicating if any changes were made.
    """
    updated = False
    for key, default_value in default_config.items():
        if key not in loaded_config:
            loaded_config[key] = default_value
            updated = True
            # Log concisely which key was added and its default value (truncated if long)
            default_repr = repr(default_value)
            print(f"{COLOR_INFO}Config Check: Added missing key '{key}' with default value: {default_repr[:80]}{'...' if len(default_repr) > 80 else ''}{RESET}")
        elif isinstance(default_value, dict) and isinstance(loaded_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_config, nested_updated = _ensure_config_keys(loaded_config[key], default_value)
            if nested_updated:
                loaded_config[key] = nested_config # Update the nested dict in the main config
                updated = True # Propagate update status upwards
        # Optional: Check for type mismatches between loaded and default (can add complexity)
        # elif type(loaded_config.get(key)) != type(default_value):
        #     print(f"{COLOR_WARNING}Config Type Mismatch: Key '{key}' has type {type(loaded_config.get(key))} but default is {type(default_value)}. Attempting to use loaded value.{RESET}")

    # Optional: Check for keys in loaded_config that are NOT in default_config (might indicate typos or old keys)
    # extra_keys = set(loaded_config.keys()) - set(default_config.keys())
    # if extra_keys:
    #     print(f"{COLOR_WARNING}Config Check: Found extra keys not in defaults: {', '.join(extra_keys)}. These will be ignored by default logic but kept in file.{RESET}")

    return loaded_config, updated

# Load configuration globally AFTER functions are defined, so functions are available
CONFIG = load_config(CONFIG_FILE)

# --- CCXT Exchange Setup (Enhanced from volbot5) ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object with API keys, rate limiting,
    retry logic, timeouts, and sandbox mode configuration. Tests connection.
    """
    lg = logger
    try:
        exchange_id = 'bybit' # Hardcoded to Bybit for now
        exchange_class = getattr(ccxt, exchange_id)

        # Configure requests session with retry strategy
        session = requests.Session()
        retries = Retry(
            total=MAX_API_RETRIES,
            backoff_factor=0.5, # Exponential backoff delay (0.5 * 2^({retry number}-1))
            status_forcelist=RETRY_ERROR_CODES, # HTTP codes to trigger retry
            allowed_methods=None # Retry on all methods (GET, POST, etc.)
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        # CCXT Exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Use CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',        # Assume linear contracts by default
                'adjustForTimeDifference': True,# Auto-adjust for clock skew
                'recvWindow': 10000,            # Bybit specific: request validity window
                # Timeouts for various operations (in milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 15000,
                'fetchOHLCVTimeout': 20000,
                'fetchPositionTimeout': 15000,
                'fetchPositionsTimeout': 20000,
            },
            'requests_session': session # Inject the session with retry logic
        }

        exchange = exchange_class(exchange_options)

        # Set Sandbox/Testnet mode based on config
        use_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(use_sandbox)
        sandbox_status = f"{COLOR_WARNING}SANDBOX MODE{RESET}" if use_sandbox else f"{COLOR_RED}!!! LIVE TRADING MODE !!!{RESET}"
        lg.warning(f"Initializing exchange {exchange.id} (v{exchange.version}). Status: {sandbox_status}")

        # Load markets - crucial for symbol info, precision, limits
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            # Force reload to ensure fresh market data
            exchange.load_markets(reload=True)
            lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols found).")
        except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
            lg.critical(f"{COLOR_ERROR}CRITICAL: Failed to load markets: {e}. Check network connection and API endpoint status. Cannot proceed.{RESET}")
            return None
        except Exception as e:
            lg.critical(f"{COLOR_ERROR}CRITICAL: Unexpected error loading markets: {e}{RESET}", exc_info=True)
            return None

        # Test API connection and credentials by fetching balance
        lg.info(f"Testing API keys and connection via balance fetch ({QUOTE_CURRENCY})...")
        # Use the dedicated fetch_balance function which handles retries
        test_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if test_balance is not None:
            # Format balance nicely, handling potential small Decimals
            balance_str = f"{test_balance:.8f}".rstrip('0').rstrip('.') if test_balance != Decimal(0) else "0"
            lg.info(f"{COLOR_SUCCESS}API connection successful. Initial {QUOTE_CURRENCY} balance: {balance_str}{RESET}")
        else:
            # Provide more specific guidance on failure
            lg.critical(f"{COLOR_ERROR}CRITICAL: Initial balance fetch failed.{RESET}")
            lg.critical(f"Troubleshooting steps:")
            lg.critical(f"  1. Verify API Key & Secret in '.env' are correct for {'Testnet' if use_sandbox else 'Mainnet'}.")
            lg.critical(f"  2. Check API key permissions on Bybit (read balance, trade permissions).")
            lg.critical(f"  3. Ensure your server's IP address is whitelisted in Bybit API settings if required.")
            lg.critical(f"  4. Check network connectivity and firewall settings.")
            lg.critical(f"  5. Ensure the `use_sandbox` setting in `config.json` matches the API keys being used.")
            return None # Cannot proceed without valid connection/keys

        return exchange

    except ccxt.AuthenticationError as e:
        # Catch auth errors during initialization specifically
        lg.critical(f"{COLOR_ERROR}CRITICAL Authentication Error initializing exchange: {e}{RESET}")
        lg.critical("Please check your API Key and Secret in the .env file.")
        return None
    except Exception as e:
        # Catch any other unexpected errors during setup
        lg.critical(f"{COLOR_ERROR}CRITICAL Unexpected Error initializing exchange: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching (Consolidated & Refined) ---

def safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """
    Safely converts a value to a Decimal object.
    Handles None, empty strings, non-finite numbers (inf, NaN), and various types.
    Returns the default value (or None) if conversion fails or input is invalid.
    """
    if value is None:
        return default
    try:
        # Convert to string first to handle various input types robustly
        str_value = str(value).strip()
        if not str_value: # Handle empty strings
            return default
        d = Decimal(str_value)
        # Check for non-finite values (Infinity, NaN) which are invalid for calculations
        if not d.is_finite():
            # Optionally log this occurrence if it's unexpected
            # print(f"Warning: Non-finite value encountered: {value}")
            return default
        return d
    except (InvalidOperation, ValueError, TypeError):
        # Catch conversion errors
        return default

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using the ticker.
    Uses fallbacks (last, mid-price, ask, bid) and includes retry logic.
    Returns the price as a Decimal, or None if fetching fails.
    """
    lg = logger
    price: Optional[Decimal] = None
    attempt = 0
    max_attempts = MAX_API_RETRIES + 1 # Total attempts = 1 initial + MAX_API_RETRIES

    while attempt < max_attempts:
        attempt += 1
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempt}/{max_attempts})...")
            params = {}
            # Add category hint for Bybit if possible
            if 'bybit' in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    # Determine category based on market info (linear is default)
                    if market:
                        params['category'] = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                    else:
                        params['category'] = 'linear' # Default guess if market info missing somehow
                except Exception:
                     params['category'] = 'linear' # Fallback category

            ticker = exchange.fetch_ticker(symbol, params=params)

            # Try extracting price in order of preference: last, mid, ask, bid
            last = safe_decimal(ticker.get('last'))
            bid = safe_decimal(ticker.get('bid'))
            ask = safe_decimal(ticker.get('ask'))

            if last and last > 0:
                price = last
                lg.debug(f"Using 'last' price: {price}")
                break # Found valid price

            mid_price = None
            if bid and ask and bid > 0 and ask > 0 and bid <= ask:
                mid_price = (bid + ask) / 2
                price = mid_price
                lg.debug(f"Using mid-price (bid={bid}, ask={ask}): {price}")
                break # Found valid price

            if ask and ask > 0:
                price = ask
                lg.debug(f"Using 'ask' price: {price}")
                break # Found valid price

            if bid and bid > 0:
                price = bid
                lg.debug(f"Using 'bid' price: {price}")
                break # Found valid price

            # If none of the above worked
            lg.warning(f"No valid price (last/mid/ask/bid > 0) found in ticker for {symbol} (Attempt {attempt}). Ticker data: {ticker}")
            if attempt < max_attempts:
                time.sleep(RETRY_DELAY_SECONDS) # Wait before retrying

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"Network error fetching price for {symbol} (Attempt {attempt}): {e}. Retrying...")
            if attempt < max_attempts:
                # Exponential backoff could be added here if desired
                time.sleep(RETRY_DELAY_SECONDS * attempt)
        except ccxt.RateLimitExceeded as e:
            # Extract suggested wait time from error message if possible
            wait_match = re.search(r'try again in (\d+)', str(e), re.IGNORECASE)
            wait_time = int(wait_match.group(1)) if wait_match else RETRY_DELAY_SECONDS * (attempt + 1)
            lg.warning(f"Rate limit hit fetching price for {symbol} (Attempt {attempt}). Retrying in {wait_time}s: {e}")
            if attempt < max_attempts:
                time.sleep(wait_time + 1) # Add a small buffer
        except ccxt.BadSymbol as e:
            lg.error(f"{COLOR_ERROR}Invalid symbol '{symbol}' specified for ticker fetch: {e}{RESET}")
            return None # No point retrying if symbol is wrong
        except ccxt.ExchangeError as e:
            # Handle common exchange errors like symbol not found
            if "symbol not found" in str(e).lower() or "invalid symbol" in str(e).lower():
                 lg.error(f"{COLOR_ERROR}Symbol '{symbol}' not found on exchange for ticker: {e}{RESET}")
                 return None # No point retrying
            lg.warning(f"Exchange error fetching price for {symbol} (Attempt {attempt}): {e}. Retrying...")
            if attempt < max_attempts:
                time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e:
            # Catch any other unexpected errors
            lg.error(f"{COLOR_ERROR}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Stop if unexpected error occurs

    # After loop finishes (either success or max retries)
    if price and price > 0:
        return price
    else:
        lg.error(f"{COLOR_ERROR}Failed to fetch valid price for {symbol} after {max_attempts} attempts.{RESET}")
        return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries and validation.
    Cleans the data, converts to appropriate types (Timestamp, Decimal), and returns a DataFrame.
    Returns an empty DataFrame on failure or if insufficient valid data is retrieved.
    """
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    empty_df = pd.DataFrame() # Return this on failure

    try:
        # Check if the exchange supports fetching OHLCV
        if not exchange.has.get('fetchOHLCV'):
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV method.")
            return empty_df

        ohlcv: Optional[List[List[Any]]] = None
        for attempt in range(MAX_API_RETRIES + 1): # +1 for the initial try
            try:
                lg.debug(f"Fetching klines for {symbol}, timeframe={timeframe}, limit={limit} (Attempt {attempt+1})...")
                params = {}
                # Add category hint for Bybit
                if 'bybit' in exchange.id.lower():
                    try:
                         market = exchange.market(symbol)
                         if market: params['category'] = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                         else: params['category'] = 'linear' # Default guess
                    except Exception: params['category'] = 'linear' # Fallback

                # Fetch the data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)

                # Check if data was received
                if ohlcv and len(ohlcv) > 0:
                    lg.debug(f"Received {len(ohlcv)} kline records.")
                    break # Success, exit retry loop
                else:
                    lg.warning(f"fetch_ohlcv returned empty or no data for {symbol} {timeframe} (Attempt {attempt+1}). Retrying...")
                    if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt+1}): {e}.")
                if attempt < MAX_API_RETRIES:
                    lg.warning("Retrying...")
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1)) # Exponential backoff
                else:
                    lg.error(f"Max retries reached for network error fetching klines for {symbol}.")
                    raise e # Re-raise the exception after max retries
            except ccxt.RateLimitExceeded as e:
                 # Try to parse wait time from error message
                 wait_match = re.search(r'(\d+)\s*(?:ms|s)', str(e).lower())
                 if wait_match:
                     wait_time = int(wait_match.group(1))
                     if 'ms' in wait_match.group(0).lower(): wait_time /= 1000
                 else:
                     wait_time = RETRY_DELAY_SECONDS * (attempt + 2) # Default backoff
                 lg.warning(f"Rate limit hit fetching klines for {symbol}. Retrying in {wait_time:.1f}s (Attempt {attempt+1}). Error: {e}")
                 if attempt < MAX_API_RETRIES:
                     time.sleep(wait_time + 0.5) # Add small buffer
                 else:
                     lg.error(f"Max retries reached for rate limit fetching klines for {symbol}.")
                     raise e # Re-raise after max retries
            except ccxt.BadSymbol as e:
                lg.error(f"{COLOR_ERROR}Invalid symbol '{symbol}' for fetching klines: {e}{RESET}")
                return empty_df # Non-recoverable error for this symbol
            except ccxt.ExchangeError as e:
                 # Handle common exchange errors like symbol not found
                 if "symbol not found" in str(e).lower() or "invalid symbol" in str(e).lower():
                     lg.error(f"{COLOR_ERROR}Symbol '{symbol}' not found on exchange for klines: {e}{RESET}")
                     return empty_df # Non-recoverable
                 lg.warning(f"Exchange error fetching klines for {symbol} (Attempt {attempt+1}): {e}. Retrying...")
                 if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                 else:
                     lg.error(f"Max retries reached for exchange error fetching klines for {symbol}.")
                     raise e
            except Exception as e:
                 # Catch any other unexpected errors during fetch
                 lg.error(f"{COLOR_ERROR}Unexpected error fetching klines {symbol}: {e}{RESET}", exc_info=True)
                 raise e # Re-raise to be caught by the outer try/except

        # After the loop, check if we got data
        if not ohlcv:
            lg.warning(f"No kline data retrieved for {symbol} {timeframe} after all retries.")
            return empty_df

        # --- Process the received OHLCV data ---
        # Define standard column names
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(ohlcv, columns=columns)
        if df.empty:
            lg.warning(f"Kline data for {symbol} {timeframe} resulted in an empty DataFrame.")
            return empty_df

        # Convert timestamp to datetime objects (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        # Drop rows where timestamp conversion failed
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
            lg.warning(f"Kline data for {symbol} {timeframe} empty after timestamp conversion/dropna.")
            return empty_df
        # Set timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential errors
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        initial_len = len(df)
        for col in numeric_cols:
            # Apply safe_decimal, replacing errors/invalid values with NaN for now
            df[col] = df[col].apply(lambda x: safe_decimal(x, default=pd.NA)) # Use pd.NA for consistency
            # Convert the column to object type first if it's not, to allow storing Decimals and pd.NA
            if df[col].dtype != 'object':
                df[col] = df[col].astype(object)

        # Drop rows with missing essential price data (O, H, L, C)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Filter out rows with non-positive close price (often indicates bad data)
        # Need to handle potential pd.NA before comparison
        df = df[df['close'].apply(lambda x: x > 0 if pd.notna(x) else False)]

        # Handle missing volume data - fill with Decimal(0)
        df['volume'].fillna(Decimal(0), inplace=True)
        # Ensure volume column remains object type if it contains Decimals
        if not all(isinstance(v, Decimal) for v in df['volume']):
             df['volume'] = df['volume'].astype(object) # Ensure it can hold Decimal(0)

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with invalid price/timestamp data from {symbol} klines.")

        if df.empty:
            lg.warning(f"Kline data empty after cleaning for {symbol} {timeframe}.")
            return empty_df

        # Sort by timestamp index (should be sorted, but good practice)
        df.sort_index(inplace=True)

        # Check for and handle duplicate timestamps (keep the last entry)
        if df.index.duplicated().any():
            duplicates_count = df.index.duplicated().sum()
            lg.warning(f"Found {duplicates_count} duplicate timestamps in kline data for {symbol}. Keeping the last entry for each duplicate.")
            df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Fetched and processed {len(df)} valid klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        # Catch errors during the processing phase
        lg.error(f"{COLOR_ERROR}Error processing kline data for {symbol} {timeframe}: {e}{RESET}", exc_info=True)
        return empty_df # Return empty DataFrame on processing error

def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """
    Fetches the order book for a symbol using CCXT with retries and validation.
    Returns the order book dictionary or None if fetching fails.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        attempts += 1
        try:
            # Check if the exchange supports fetching order books
            if not exchange.has.get('fetchOrderBook'):
                lg.error(f"Exchange {exchange.id} does not support fetchOrderBook method.")
                return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts})...")
            params = {}
            # Add category hint for Bybit
            if 'bybit' in exchange.id.lower():
                 try:
                     market = exchange.market(symbol)
                     if market: params['category'] = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                     else: params['category'] = 'linear' # Default guess
                 except Exception: params['category'] = 'linear' # Fallback

            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            # Validate the structure of the returned order book
            if not orderbook:
                 lg.warning(f"fetch_order_book returned None or empty for {symbol} (Attempt {attempts}).")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                 lg.warning(f"Invalid order book structure received for {symbol} (Attempt {attempts}). Bids/Asks not lists. Response: {str(orderbook)[:200]}...") # Log snippet
            elif not orderbook.get('bids') and not orderbook.get('asks'):
                 # It's possible to receive an empty book if the market is inactive
                 lg.warning(f"Order book received for {symbol} but both bids and asks are empty (Attempt {attempts}). Market might be inactive.")
                 # Return the empty book as it's technically valid data
                 return orderbook
            else:
                 # Successful fetch and basic validation passed
                 lg.debug(f"Fetched order book for {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 # Optional: Further validation (e.g., check bid > ask) could be added here
                 return orderbook

            # If validation failed and not yet max retries, wait and retry
            if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"Network error fetching order book for {symbol} (Attempt {attempts}): {e}. Retrying...")
            if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
        except ccxt.RateLimitExceeded as e:
            # Parse wait time if possible
            wait_match = re.search(r'(\d+)\s*(?:ms|s)', str(e).lower())
            if wait_match:
                wait_time = int(wait_match.group(1))
                if 'ms' in wait_match.group(0).lower(): wait_time /= 1000
            else:
                wait_time = RETRY_DELAY_SECONDS * (attempts + 1)
            lg.warning(f"Rate limit hit fetching order book for {symbol}. Retrying in {wait_time:.1f}s (Attempt {attempts}). Error: {e}")
            if attempts <= MAX_API_RETRIES: time.sleep(wait_time + 0.5)
            else: lg.error(f"Max retries reached for rate limit fetching order book for {symbol}.")
        except ccxt.BadSymbol as e:
             lg.error(f"{COLOR_ERROR}Invalid symbol '{symbol}' for fetching order book: {e}{RESET}")
             return None # Non-recoverable
        except ccxt.ExchangeError as e:
             if "symbol not found" in str(e).lower() or "invalid symbol" in str(e).lower():
                 lg.error(f"{COLOR_ERROR}Symbol '{symbol}' not found on exchange for order book: {e}{RESET}")
                 return None # Non-recoverable
             lg.warning(f"Exchange error fetching order book for {symbol} (Attempt {attempts}): {e}. Retrying...")
             if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
             else: lg.error(f"Max retries reached for exchange error fetching order book for {symbol}.")
        except Exception as e:
             lg.error(f"{COLOR_ERROR}Unexpected error fetching order book for {symbol}: {e}{RESET}", exc_info=True)
             return None # Non-recoverable for unexpected errors

    # If loop completes without returning a valid order book
    lg.error(f"{COLOR_ERROR}Max retries ({MAX_API_RETRIES}) reached fetching order book for {symbol}. Failed.{RESET}")
    return None

# --- Volbot Strategy Calculation Functions (Adapted from volbot5) ---
# Note: These functions now assume input DataFrame columns ('open', 'high', 'low', 'close', 'volume')
# contain Decimal objects or pd.NA where applicable, as prepared by fetch_klines_ccxt.
# They may create temporary float columns for pandas_ta compatibility internally.

def ema_swma(series: pd.Series, length: int, logger: logging.Logger) -> pd.Series:
    """
    Calculates Smoothed Weighted Moving Average (SWMA).
    This specific version is an EMA of a 4-period weighted average of the input series.
    Handles potential pd.NA values and ensures sufficient data.
    Returns a pandas Series with the SWMA results.
    """
    lg = logger
    lg.debug(f"Calculating SWMA with length: {length}...")
    required_periods = 4 # Needs 4 values for the weighted average calculation

    # Ensure the input series is numeric (float) for weighting and EMA calculation
    # Use errors='coerce' to turn non-numeric values (like pd.NA) into NaN
    numeric_series = pd.to_numeric(series, errors='coerce')

    if len(numeric_series) < required_periods:
        lg.warning(f"Series length {len(numeric_series)} is less than required {required_periods} periods for SWMA. Returning standard EMA instead.")
        # Fallback to standard EMA if not enough data
        ema_result = ta.ema(numeric_series.dropna(), length=length, adjust=False)
        # Reindex to match the original series index, filling gaps with NaN
        return ema_result.reindex(series.index) if isinstance(ema_result, pd.Series) else pd.Series(ema_result, index=series.index)

    # Calculate the 4-period weighted average: (price[0]*1 + price[-1]*2 + price[-2]*2 + price[-3]*1) / 6
    # Using fillna(0) temporarily for the weighted sum calculation, but will reset NaNs later
    w0 = numeric_series.fillna(0) / 6
    w1 = numeric_series.shift(1).fillna(0) * 2 / 6
    w2 = numeric_series.shift(2).fillna(0) * 2 / 6
    w3 = numeric_series.shift(3).fillna(0) * 1 / 6
    weighted_series = w0 + w1 + w2 + w3

    # Restore NaNs where the original series had NaNs
    weighted_series[numeric_series.isna()] = np.nan
    # Set initial periods (where shifts produced NaNs) to NaN
    weighted_series.iloc[:required_periods-1] = np.nan

    # Calculate the EMA of the weighted series
    # Drop NaNs before calculating EMA to avoid issues, then reindex
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length, adjust=False)

    # Reindex the result to match the original series' index
    result_series = smoothed_ema.reindex(series.index)

    lg.debug(f"SWMA calculation finished. Result contains {result_series.isna().sum()} NaN values.")
    # Return the result as floats (as calculated by ta.ema)
    return result_series

def calculate_volatility_levels(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates Volumatic Trend indicators based on Volbot strategy logic.
    Includes EMAs, ATR, dynamic volatility levels, trend detection, and volume metrics.
    Assumes input df has Decimal prices/volume. Creates temporary float columns for TA.
    Returns the DataFrame with added 'strat' columns (as float or boolean).
    """
    lg = logger
    lg.info("Calculating Volumatic Trend Levels...")
    length = config.get("volbot_length", DEFAULT_VOLBOT_LENGTH)
    atr_length = config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH)
    volume_lookback = config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK)
    # Determine minimum data length required for calculations
    min_len = max(length + 3, atr_length, volume_lookback) + 10 # Add buffer

    if len(df) < min_len:
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)} rows) for Volumatic calculation (minimum ~{min_len} needed). Skipping.{RESET}")
        # Add placeholder columns to avoid errors later
        placeholder_cols = [
            'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
            'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
            'step_up_strat', 'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat',
            'vol_dn_step_strat', 'vol_trend_up_level_strat', 'vol_trend_dn_level_strat',
            'volume_delta_strat', 'volume_total_strat', 'cum_vol_delta_since_change_strat',
            'cum_vol_total_since_change_strat', 'last_trend_change_idx'
        ]
        for col in placeholder_cols:
            # Assign appropriate dtype if possible (object allows mixed types like float/None)
            df[col] = pd.Series(dtype='object')
        return df

    # Work on a copy to avoid modifying the original DataFrame passed to the function
    df_calc = df.copy()
    temp_float_cols = [] # Keep track of temporary float columns created

    try:
        # --- Prepare float columns for pandas_ta ---
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols + ['volume']:
            float_col_name = f'{col}_float'
            if col in df_calc:
                df_calc[float_col_name] = pd.to_numeric(df_calc[col], errors='coerce')
                temp_float_cols.append(float_col_name)
            else:
                lg.error(f"Missing required column '{col}' in DataFrame for Volumatic calculation.")
                # Add empty placeholder columns and return original df
                for p_col in placeholder_cols: df[p_col] = pd.Series(dtype='object')
                return df # Return original df

        # --- Calculate Core Indicators (using float versions) ---
        # Calculate SWMA (ema1) using the custom function on the float close prices
        df_calc['ema1_strat'] = ema_swma(df_calc['close_float'], length, lg)
        # Calculate standard EMA (ema2) using pandas_ta on float close prices
        df_calc['ema2_strat'] = ta.ema(df_calc['close_float'], length=length, adjust=False)
        # Calculate ATR using pandas_ta on float H, L, C prices
        df_calc['atr_strat'] = ta.atr(df_calc['high_float'], df_calc['low_float'], df_calc['close_float'], length=atr_length)

        # Drop initial rows where EMAs/ATR are NaN
        df_calc.dropna(subset=['ema1_strat', 'ema2_strat', 'atr_strat'], inplace=True)
        if df_calc.empty:
            lg.warning("DataFrame became empty after dropping initial NaN values from EMA/ATR calculations.")
            # Add placeholders to original df and return
            for p_col in placeholder_cols: df[p_col] = pd.Series(dtype='object')
            return df

        # --- Determine Trend ---
        # Trend is up if ema1 > ema2. Use boolean type for clarity.
        df_calc['trend_up_strat'] = (df_calc['ema1_strat'] > df_calc['ema2_strat']).astype('boolean')
        # Detect when the trend changes (diff() will be True/False, fill first NaN with False)
        df_calc['trend_changed_strat'] = df_calc['trend_up_strat'].diff().fillna(False)

        # --- Calculate Dynamic Volatility Levels ---
        # Initialize level columns with NaN
        level_cols = ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat']
        for col in level_cols: df_calc[col] = np.nan

        # Find indices where the trend changed
        change_indices = df_calc.index[df_calc['trend_changed_strat']]

        if not change_indices.empty:
            # Get EMA1 and ATR from the *previous* bar where the trend changed
            ema1_at_change = pd.to_numeric(df_calc['ema1_strat'].shift(1), errors='coerce').loc[change_indices]
            atr_at_change = pd.to_numeric(df_calc['atr_strat'].shift(1), errors='coerce').loc[change_indices]

            # Ensure values are valid (not NaN and ATR > 0)
            valid_mask = pd.notna(ema1_at_change) & pd.notna(atr_at_change) & (atr_at_change > 0)
            valid_indices = change_indices[valid_mask]

            if not valid_indices.empty:
                valid_ema1 = ema1_at_change[valid_mask]
                valid_atr = atr_at_change[valid_mask]

                # Calculate levels based on EMA1 and ATR at the point of trend change
                upper = valid_ema1 + valid_atr * 3.0
                lower = valid_ema1 - valid_atr * 3.0
                # Volatility-adjusted levels
                lower_vol = lower + valid_atr * 4.0
                upper_vol = upper - valid_atr * 4.0
                # Step values for volume influence (clipped at 0)
                step_up = (lower_vol - lower).clip(lower=0.0) / 100.0
                step_dn = (upper - upper_vol).clip(lower=0.0) / 100.0

                # Assign calculated levels to the DataFrame at the change indices
                df_calc.loc[valid_indices, 'upper_strat'] = upper
                df_calc.loc[valid_indices, 'lower_strat'] = lower
                df_calc.loc[valid_indices, 'lower_vol_strat'] = lower_vol
                df_calc.loc[valid_indices, 'upper_vol_strat'] = upper_vol
                df_calc.loc[valid_indices, 'step_up_strat'] = step_up
                df_calc.loc[valid_indices, 'step_dn_strat'] = step_dn

        # Forward fill the calculated levels until the next trend change
        for col in level_cols: df_calc[col] = df_calc[col].ffill()

        # --- Volume Analysis ---
        # Calculate max volume over the lookback period (use float volume)
        # Use rolling max with a minimum number of periods to avoid NaN at the start
        min_vol_periods = max(1, volume_lookback // 10)
        max_vol_lookback = df_calc['volume_float'].rolling(window=volume_lookback, min_periods=min_vol_periods).max()

        # Normalize volume (0-100) based on the rolling max
        # Handle potential division by zero if max_vol_lookback is zero or NaN
        df_calc['vol_norm_strat'] = np.where(
            pd.notna(max_vol_lookback) & (max_vol_lookback > 1e-9), # Avoid division by tiny numbers
            (df_calc['volume_float'].fillna(0.0) / max_vol_lookback * 100.0),
            0.0 # Assign 0 if max_vol is invalid or zero
        ).clip(0.0, 100.0) # Clip result between 0 and 100

        # Calculate volume-influenced step adjustments
        df_calc['vol_up_step_strat'] = (df_calc['step_up_strat'].fillna(0.0) * df_calc['vol_norm_strat'].fillna(0.0))
        df_calc['vol_dn_step_strat'] = (df_calc['step_dn_strat'].fillna(0.0) * df_calc['vol_norm_strat'].fillna(0.0))

        # Calculate final volume-adjusted trend levels
        df_calc['vol_trend_up_level_strat'] = df_calc['lower_strat'].fillna(0.0) + df_calc['vol_up_step_strat'].fillna(0.0)
        df_calc['vol_trend_dn_level_strat'] = df_calc['upper_strat'].fillna(0.0) - df_calc['vol_dn_step_strat'].fillna(0.0)

        # --- Cumulative Volume Delta ---
        # Calculate volume delta per bar (positive if close > open, negative if close < open)
        df_calc['volume_delta_strat'] = np.where(
            df_calc['close_float'] > df_calc['open_float'], df_calc['volume_float'],
            np.where(df_calc['close_float'] < df_calc['open_float'], -df_calc['volume_float'], 0.0)
        ).fillna(0.0)
        # Total volume per bar (just the float volume)
        df_calc['volume_total_strat'] = df_calc['volume_float'].fillna(0.0)

        # Group by trend blocks (defined by cumsum of trend changes)
        trend_block_group = df_calc['trend_changed_strat'].cumsum()
        # Calculate cumulative volume delta and total volume since the last trend change
        df_calc['cum_vol_delta_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_delta_strat'].cumsum()
        df_calc['cum_vol_total_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_total_strat'].cumsum()

        # --- Store Last Trend Change Timestamp ---
        # Find the timestamp of the last trend change for each row
        last_change_ts = df_calc.index.to_series().where(df_calc['trend_changed_strat']).ffill()
        df_calc['last_trend_change_idx'] = last_change_ts # Store as Timestamp

        lg.info("Volumatic Trend Levels calculation complete.")
        # Merge calculated columns back into the original DataFrame structure if needed,
        # or just return the df_calc with results. Let's merge selectively.

        # Select only the calculated 'strat' columns
        strat_cols = [col for col in df_calc.columns if col.endswith('_strat') or col == 'last_trend_change_idx']
        # Merge back into the original df, aligning by index
        df_merged = df.join(df_calc[strat_cols])

        return df_merged

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Volumatic Trend calculation: {e}{RESET}", exc_info=True)
        # Return the original DataFrame with placeholder columns in case of error
        for p_col in placeholder_cols: df[p_col] = pd.Series(dtype='object')
        return df
    finally:
        # Clean up temporary float columns from df_calc if it exists
        if 'df_calc' in locals() and not df_calc.empty:
            try:
                df_calc.drop(columns=temp_float_cols, inplace=True, errors='ignore')
            except Exception as clean_e:
                 lg.warning(f"Could not clean up temporary float columns: {clean_e}")


def calculate_pivot_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Identifies Pivot High (PH) and Pivot Low (PL) points based on configuration.
    These pivots are used later to define potential Order Blocks (OBs).
    Assumes input df has Decimal prices.
    Returns the DataFrame with added 'ph_strat' and 'pl_strat' columns (containing Decimal price or pd.NA).
    """
    lg = logger
    # Get configuration settings for pivot calculation
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE) # "Wicks" or "Bodys"
    left_h = config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    left_l = config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    lg.info(f"Calculating Pivots (Source: {source}, Lookback/forward H: {left_h}/{right_h}, L: {left_l}/{right_l})...")

    # Determine minimum data length required
    min_len = max(left_h + right_h + 1, left_l + right_l + 1)
    if len(df) < min_len:
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)} rows) for Pivot calculation (minimum {min_len} needed). Skipping.{RESET}")
        df['ph_strat'] = pd.NA # Add placeholder columns
        df['pl_strat'] = pd.NA
        return df

    df_calc = df.copy() # Work on a copy

    try:
        # Determine which price columns to use based on the source setting
        high_src_col = 'high' if source == "Wicks" else 'close'
        low_src_col = 'low' if source == "Wicks" else 'open'

        # Check if source columns exist
        if high_src_col not in df_calc or low_src_col not in df_calc:
             lg.error(f"Missing required source columns '{high_src_col}' or '{low_src_col}' for pivot calculation based on source '{source}'. Skipping.")
             df['ph_strat'] = pd.NA
             df['pl_strat'] = pd.NA
             return df

        # Ensure source columns are Decimal or pd.NA (should be from fetch_klines)
        # No explicit conversion needed here if fetch_klines worked correctly.

        # Initialize pivot columns with pd.NA (allows storing Decimals or NA)
        df_calc['ph_strat'] = pd.NA
        df_calc['pl_strat'] = pd.NA
        # Ensure dtype is object to hold Decimal/NA mix
        df_calc['ph_strat'] = df_calc['ph_strat'].astype(object)
        df_calc['pl_strat'] = df_calc['pl_strat'].astype(object)

        # --- Pivot High (PH) Detection ---
        # Iterate through the DataFrame rows where a full lookback/forward window exists
        for i in range(left_h, len(df_calc) - right_h):
            idx = df_calc.index[i]
            pivot_val = df_calc.at[idx, high_src_col]

            # Skip if the pivot candidate value itself is NA
            if pd.isna(pivot_val): continue

            # Get values in the left and right windows
            # Use iloc for positional indexing relative to the current row 'i'
            left_vals = df_calc[high_src_col].iloc[i - left_h : i]
            right_vals = df_calc[high_src_col].iloc[i + 1 : i + right_h + 1]

            # Check conditions:
            # 1. No NA values in the left or right windows
            # 2. Pivot value is strictly greater than all values in the left window
            # 3. Pivot value is strictly greater than all values in the right window
            if not left_vals.isna().any() and not right_vals.isna().any() and \
               (pivot_val > left_vals).all() and (pivot_val > right_vals).all():
                # Found a Pivot High, store its price (Decimal) in ph_strat
                df_calc.loc[idx, 'ph_strat'] = pivot_val

        # --- Pivot Low (PL) Detection ---
        # Similar iteration for Pivot Lows
        for i in range(left_l, len(df_calc) - right_l):
            idx = df_calc.index[i]
            pivot_val = df_calc.at[idx, low_src_col]

            # Skip if the pivot candidate value itself is NA
            if pd.isna(pivot_val): continue

            # Get values in the left and right windows
            left_vals = df_calc[low_src_col].iloc[i - left_l : i]
            right_vals = df_calc[low_src_col].iloc[i + 1 : i + right_l + 1]

            # Check conditions:
            # 1. No NA values in the left or right windows
            # 2. Pivot value is strictly less than all values in the left window
            # 3. Pivot value is strictly less than all values in the right window
            if not left_vals.isna().any() and not right_vals.isna().any() and \
               (pivot_val < left_vals).all() and (pivot_val < right_vals).all():
                # Found a Pivot Low, store its price (Decimal) in pl_strat
                df_calc.loc[idx, 'pl_strat'] = pivot_val

        # --- Finalize ---
        ph_count = df_calc['ph_strat'].notna().sum()
        pl_count = df_calc['pl_strat'].notna().sum()
        lg.info(f"Pivot calculation complete. Found {ph_count} Pivot Highs (PH) and {pl_count} Pivot Lows (PL).")

        # Merge the results back into the original DataFrame
        df['ph_strat'] = df_calc['ph_strat']
        df['pl_strat'] = df_calc['pl_strat']
        return df

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error calculating Pivots: {e}{RESET}", exc_info=True)
        # Return original df with placeholder columns on error
        df['ph_strat'] = pd.NA
        df['pl_strat'] = pd.NA
        return df


def manage_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Identifies, creates, and manages the state ('active', 'closed', 'trimmed') of Order Blocks (OBs)
    based on detected Pivot Highs (for Bearish OBs) and Pivot Lows (for Bullish OBs).
    Assumes input df has Decimal prices and 'ph_strat', 'pl_strat' columns.
    Returns the DataFrame with added 'active_bull_ob_strat' and 'active_bear_ob_strat' columns
    (containing the dictionary of the OB the price is currently inside, or None),
    and two lists containing all identified bull and bear boxes throughout the history.
    """
    lg = logger
    lg.info("Managing Order Block Boxes...")
    # Get configuration settings
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    # Offset determines which bar relative to the pivot defines the OB dimensions
    offset_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    offset_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    max_boxes = config.get("volbot_max_boxes", DEFAULT_VOLBOT_MAX_BOXES)

    df_calc = df.copy() # Work on a copy
    # Lists to store all created boxes during the iteration
    all_bull_boxes: List[Dict] = []
    all_bear_boxes: List[Dict] = []
    # Lists to track currently active boxes during the iteration
    active_bull_boxes: List[Dict] = []
    active_bear_boxes: List[Dict] = []
    box_counter = 0 # Simple counter for unique box IDs

    # Check for required pivot columns
    if 'ph_strat' not in df_calc or 'pl_strat' not in df_calc:
        lg.warning(f"{COLOR_WARNING}Pivot columns ('ph_strat', 'pl_strat') not found. Cannot manage Order Blocks. Skipping.{RESET}")
        # Add placeholder columns and return
        df['active_bull_ob_strat'] = None
        df['active_bear_ob_strat'] = None
        df['active_bull_ob_strat'] = df['active_bull_ob_strat'].astype(object)
        df['active_bear_ob_strat'] = df['active_bear_ob_strat'].astype(object)
        return df, [], [] # Return original df and empty lists

    # Initialize columns to store the *reference* to the active OB dict for each bar
    df_calc['active_bull_ob_strat'] = pd.Series(dtype='object')
    df_calc['active_bear_ob_strat'] = pd.Series(dtype='object')

    # Ensure required price columns are present (as Decimals or pd.NA)
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df_calc:
            lg.error(f"Missing required column '{col}' for OB management. Skipping.")
            df['active_bull_ob_strat'] = None; df['active_bear_ob_strat'] = None
            df['active_bull_ob_strat'] = df['active_bull_ob_strat'].astype(object)
            df['active_bear_ob_strat'] = df['active_bear_ob_strat'].astype(object)
            return df, [], []

    try:
        # Iterate through each bar (row) of the DataFrame
        for i in range(len(df_calc)):
            idx = df_calc.index[i] # Current bar's timestamp index
            # Get current bar's prices (as Decimals or pd.NA)
            current_open = df_calc.at[idx, 'open']
            current_high = df_calc.at[idx, 'high']
            current_low = df_calc.at[idx, 'low']
            current_close = df_calc.at[idx, 'close']

            # Skip iteration if close price is missing (cannot determine OB state)
            if pd.isna(current_close):
                 # Assign None to active OB columns for this row
                 df_calc.loc[idx, 'active_bull_ob_strat'] = None
                 df_calc.loc[idx, 'active_bear_ob_strat'] = None
                 continue

            # --- Update State of Existing Active Boxes ---
            # Check Bullish OBs: Close below bottom closes the box
            next_active_bull = []
            active_bull_ref_for_this_bar = None # Track if price is inside any active bull OB *this* bar
            for box in active_bull_boxes:
                if current_close < box['bottom']:
                    box['state'] = 'closed' # Mark as closed
                    box['end_idx'] = idx    # Record closing timestamp
                    lg.debug(f"Closed Bull OB: {box['id']} at index {idx} (Close={current_close} < Bottom={box['bottom']})")
                    # Closed boxes are not added to next_active_bull
                else:
                    next_active_bull.append(box) # Keep active
                    # Check if the current price is within this still-active box
                    if box['bottom'] <= current_close <= box['top']:
                        # If multiple boxes contain the price, the last one checked (newest?) will be stored.
                        # Consider if logic needs to prioritize (e.g., highest bottom) if overlap occurs.
                        active_bull_ref_for_this_bar = box
            active_bull_boxes = next_active_bull # Update the list of active boxes

            # Check Bearish OBs: Close above top closes the box
            next_active_bear = []
            active_bear_ref_for_this_bar = None # Track if price is inside any active bear OB *this* bar
            for box in active_bear_boxes:
                if current_close > box['top']:
                    box['state'] = 'closed' # Mark as closed
                    box['end_idx'] = idx    # Record closing timestamp
                    lg.debug(f"Closed Bear OB: {box['id']} at index {idx} (Close={current_close} > Top={box['top']})")
                    # Closed boxes are not added to next_active_bear
                else:
                    next_active_bear.append(box) # Keep active
                    # Check if the current price is within this still-active box
                    if box['bottom'] <= current_close <= box['top']:
                        active_bear_ref_for_this_bar = box
            active_bear_boxes = next_active_bear # Update the list of active boxes

            # Store the reference to the active OB (or None) for the current bar
            df_calc.loc[idx, 'active_bull_ob_strat'] = active_bull_ref_for_this_bar
            df_calc.loc[idx, 'active_bear_ob_strat'] = active_bear_ref_for_this_bar

            # --- Create New Boxes based on Pivots found at the current bar ---
            # Check for Pivot High (potential Bearish OB)
            pivot_h_price = df_calc.at[idx, 'ph_strat']
            if pd.notna(pivot_h_price):
                # Determine the index of the bar that defines the OB (offset bars before the pivot)
                ob_bar_iloc = i - offset_h
                if ob_bar_iloc >= 0: # Ensure the offset index is valid
                    ob_idx = df_calc.index[ob_bar_iloc] # Timestamp index of the OB bar
                    # Get prices from the OB bar to define top/bottom
                    ob_open = df_calc.at[ob_idx, 'open']
                    ob_high = df_calc.at[ob_idx, 'high']
                    ob_low = df_calc.at[ob_idx, 'low']
                    ob_close = df_calc.at[ob_idx, 'close']

                    # Define top and bottom based on source ('Wicks' or 'Bodys')
                    top_p, bot_p = pd.NA, pd.NA
                    if source == "Bodys":
                        top_p, bot_p = ob_open, ob_close # Bearish OB: Open to Close of the candle
                    else: # Default to "Wicks"
                        top_p, bot_p = ob_high, ob_close # Bearish OB: High to Close of the candle

                    # Ensure we have valid Decimal prices
                    if pd.notna(top_p) and pd.notna(bot_p):
                        # Ensure top is actually above bottom (handle potential inside bars etc.)
                        top_price, bot_price = max(top_p, bot_p), min(top_p, bot_p)
                        if top_price > bot_price: # Only create if there's a valid range
                            box_counter += 1
                            new_box = {
                                'id': f'BearOB_{box_counter}', 'type': 'bear',
                                'start_idx': ob_idx,      # Timestamp when OB bar occurred
                                'pivot_idx': idx,         # Timestamp when pivot confirmed the OB
                                'end_idx': None,          # Timestamp when closed (initially None)
                                'top': top_price,         # Top price of the OB (Decimal)
                                'bottom': bot_price,      # Bottom price of the OB (Decimal)
                                'state': 'active'         # Initial state
                            }
                            all_bear_boxes.append(new_box)
                            active_bear_boxes.append(new_box) # Add to active list
                            lg.debug(f"Created Bear OB: {new_box['id']} (Top={top_price}, Bot={bot_price}) based on PH at {idx}")

            # Check for Pivot Low (potential Bullish OB)
            pivot_l_price = df_calc.at[idx, 'pl_strat']
            if pd.notna(pivot_l_price):
                # Determine the index of the bar that defines the OB
                ob_bar_iloc = i - offset_l
                if ob_bar_iloc >= 0: # Ensure valid index
                    ob_idx = df_calc.index[ob_bar_iloc]
                    # Get prices from the OB bar
                    ob_open = df_calc.at[ob_idx, 'open']
                    ob_high = df_calc.at[ob_idx, 'high']
                    ob_low = df_calc.at[ob_idx, 'low']
                    ob_close = df_calc.at[ob_idx, 'close']

                    # Define top and bottom based on source
                    top_p, bot_p = pd.NA, pd.NA
                    if source == "Bodys":
                        top_p, bot_p = ob_close, ob_open # Bullish OB: Close to Open of the candle
                    else: # Default to "Wicks"
                        top_p, bot_p = ob_open, ob_low   # Bullish OB: Open to Low of the candle

                    if pd.notna(top_p) and pd.notna(bot_p):
                        top_price, bot_price = max(top_p, bot_p), min(top_p, bot_p)
                        if top_price > bot_price:
                            box_counter += 1
                            new_box = {
                                'id': f'BullOB_{box_counter}', 'type': 'bull',
                                'start_idx': ob_idx, 'pivot_idx': idx, 'end_idx': None,
                                'top': top_price, 'bottom': bot_price, 'state': 'active'
                            }
                            all_bull_boxes.append(new_box)
                            active_bull_boxes.append(new_box)
                            lg.debug(f"Created Bull OB: {new_box['id']} (Top={top_price}, Bot={bot_price}) based on PL at {idx}")

            # --- Trim Excess Active Boxes (Limit Memory Usage/Complexity) ---
            # If number of active boxes exceeds max, remove the oldest ones (based on pivot confirmation time)
            if len(active_bull_boxes) > max_boxes:
                # Sort by pivot index (newest first)
                active_bull_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True)
                # Identify boxes to remove
                removed_bull = active_bull_boxes[max_boxes:]
                # Keep only the newest 'max_boxes'
                active_bull_boxes = active_bull_boxes[:max_boxes]
                # Mark removed boxes state (optional, helps tracking)
                for box in removed_bull:
                    box['state'] = 'trimmed'
                    lg.debug(f"Trimmed oldest Bull OB: {box['id']} due to max_boxes limit.")

            if len(active_bear_boxes) > max_boxes:
                active_bear_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True)
                removed_bear = active_bear_boxes[max_boxes:]
                active_bear_boxes = active_bear_boxes[:max_boxes]
                for box in removed_bear:
                    box['state'] = 'trimmed'
                    lg.debug(f"Trimmed oldest Bear OB: {box['id']} due to max_boxes limit.")

        # --- Finalize ---
        # Count final active boxes at the end of the iteration
        num_active_bull = sum(1 for b in active_bull_boxes)
        num_active_bear = sum(1 for b in active_bear_boxes)
        lg.info(f"OB management complete. Total created: {len(all_bull_boxes)} Bull, {len(all_bear_boxes)} Bear. Final active: {num_active_bull} Bull, {num_active_bear} Bear.")

        # Merge the active OB reference columns back to the original DataFrame
        df['active_bull_ob_strat'] = df_calc['active_bull_ob_strat']
        df['active_bear_ob_strat'] = df_calc['active_bear_ob_strat']

        return df, all_bull_boxes, all_bear_boxes

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during OB management: {e}{RESET}", exc_info=True)
        # Return original df with placeholders and empty lists on error
        df['active_bull_ob_strat'] = None; df['active_bear_ob_strat'] = None
        df['active_bull_ob_strat'] = df['active_bull_ob_strat'].astype(object)
        df['active_bear_ob_strat'] = df['active_bear_ob_strat'].astype(object)
        return df, [], []


# --- Trading Analyzer Class (Merged & Enhanced) ---
class TradingAnalyzer:
    """
    Analyzes market data using configured strategies (LiveXY, Volbot),
    generates trading signals, and calculates risk parameters (SL/TP).
    Handles market precision and rounding.
    """
    def __init__(
        self, df_raw: pd.DataFrame, logger: logging.Logger, config: Dict[str, Any],
        market_info: Dict[str, Any], orderbook_data: Optional[Dict] = None
    ) -> None:
        """
        Initializes the analyzer.

        Args:
            df_raw: Raw OHLCV DataFrame (expecting Decimal prices/volume).
            logger: Logger instance for logging messages.
            config: Dictionary containing bot configuration.
            market_info: Dictionary with market details (precision, limits) from CCXT.
            orderbook_data: Optional dictionary containing fetched order book data.
        """
        self.df_raw = df_raw
        self.df_processed = pd.DataFrame() # DataFrame after adding indicators
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.orderbook_data = orderbook_data # Store fetched orderbook data
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN") # User-friendly interval (e.g., "5")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN") # CCXT format (e.g., "5m")

        # Determine precision and step sizes from market info with fallbacks
        self.min_tick_size = self._determine_min_tick_size()
        self.price_precision = self._determine_price_precision()
        self.amount_step_size = self._determine_amount_step_size()
        self.amount_precision = self._determine_amount_precision()

        # Strategy Enablement Flags
        self.livexy_enabled = config.get("livexy_enabled", True)
        self.volbot_enabled = config.get("volbot_enabled", True)

        # LiveXY specific state
        # Stores latest calculated *float* values for LiveXY signal checks
        self.livexy_indicator_values: Dict[str, float] = {}
        # Stores the actual column names generated by pandas_ta for LiveXY indicators
        self.livexy_ta_column_names: Dict[str, Optional[str]] = {}
        self.livexy_active_weight_set_name = config.get("active_weight_set", "default")
        # Load the active weight set, default to empty dict if not found
        self.livexy_weights = config.get("weight_sets", {}).get(self.livexy_active_weight_set_name, {})
        if not self.livexy_weights and self.livexy_enabled:
             self.logger.warning(f"LiveXY active weight set '{self.livexy_active_weight_set_name}' not found or empty in config.")

        # Volbot specific state
        self.all_bull_boxes: List[Dict] = [] # Stores all historical bull OBs
        self.all_bear_boxes: List[Dict] = [] # Stores all historical bear OBs
        # References to the currently active OBs (updated by _update_latest_strategy_state)
        self.latest_active_bull_ob: Optional[Dict] = None
        self.latest_active_bear_ob: Optional[Dict] = None

        # Combined State - Holds latest values (Decimal where possible) from all enabled strategies
        # Populated by _update_latest_strategy_state after indicator calculation
        self.strategy_state: Dict[str, Any] = {}

        # --- Initial Setup ---
        self.logger.debug(f"Analyzer initialized for {self.symbol}: "
                          f"TickSize={self.min_tick_size}, PricePrec={self.price_precision}, "
                          f"AmtStep={self.amount_step_size}, AmtPrec={self.amount_precision}")

        # Calculate all indicators and populate df_processed
        self._calculate_indicators()

        # Populate the latest state dictionary based on the processed DataFrame
        self._update_latest_strategy_state()

    # --- Precision/Rounding Helpers ---
    def _determine_min_tick_size(self) -> Decimal:
        """Determines the minimum price increment (tick size) from market info."""
        try:
            # Prefer 'precision.price' if available (often represents tick size)
            price_prec_val = self.market_info.get('precision', {}).get('price')
            tick = safe_decimal(price_prec_val)
            if tick and tick > 0:
                self.logger.debug(f"Using tick size from precision.price: {tick}")
                return tick

            # Fallback to 'limits.price.min' if precision.price is not useful
            min_price_limit = self.market_info.get('limits', {}).get('price', {}).get('min')
            tick = safe_decimal(min_price_limit)
            # Heuristic: Check if min price limit looks like a tick size (small value)
            if tick and tick > 0 and tick < Decimal('1'): # Adjust threshold if needed
                self.logger.debug(f"Using tick size from limits.price.min: {tick}")
                return tick

        except Exception as e:
            self.logger.warning(f"Could not reliably determine tick size for {self.symbol} from market info: {e}. Using fallback.")

        # Fallback based on last price (very rough estimate)
        last_price = safe_decimal(self.df_raw['close'].iloc[-1]) if not self.df_raw.empty and 'close' in self.df_raw.columns else None
        if last_price:
            if last_price > 1000: default_tick = Decimal('0.1')
            elif last_price > 10: default_tick = Decimal('0.01')
            elif last_price > 0.1: default_tick = Decimal('0.001')
            elif last_price > 0.001: default_tick = Decimal('0.0001')
            else: default_tick = Decimal('1e-5') # For very low-priced assets
        else:
            default_tick = Decimal('0.0001') # Generic fallback

        self.logger.warning(f"Using estimated/fallback tick size {default_tick} for {self.symbol}.")
        return default_tick

    def _determine_price_precision(self) -> int:
        """Determines the number of decimal places for price based on tick size."""
        try:
            if self.min_tick_size > 0:
                # Calculate precision from the exponent of the normalized tick size
                precision = abs(self.min_tick_size.normalize().as_tuple().exponent)
                self.logger.debug(f"Determined price precision from tick size: {precision}")
                return precision
        except Exception:
            pass # Fall through to default if calculation fails
        self.logger.warning(f"Could not determine price precision for {self.symbol}. Using fallback: 4")
        return 4 # Default fallback precision

    def _determine_amount_step_size(self) -> Decimal:
        """Determines the minimum order quantity increment (step size) from market info."""
        try:
            # Prefer 'precision.amount' if it represents the step size
            amount_prec_val = self.market_info.get('precision', {}).get('amount')
            step_size = safe_decimal(amount_prec_val)
            if step_size and step_size > 0:
                 # Check if it's likely a step size (e.g., 0.001) rather than precision digits (e.g., 8)
                 if step_size <= Decimal('1'):
                     self.logger.debug(f"Using amount step size from precision.amount: {step_size}")
                     return step_size
                 # If precision.amount seems to be #digits, calculate step size
                 elif amount_prec_val == int(amount_prec_val): # Check if it's an integer
                     calculated_step = Decimal('1') / (Decimal('10') ** int(amount_prec_val))
                     self.logger.debug(f"Calculated amount step size from precision.amount digits ({int(amount_prec_val)}): {calculated_step}")
                     return calculated_step

            # Fallback to 'limits.amount.min' if precision.amount isn't useful step size
            min_amount_limit = self.market_info.get('limits', {}).get('amount', {}).get('min')
            step_size = safe_decimal(min_amount_limit)
            # Heuristic: Check if min amount looks like a step size (usually small)
            if step_size and step_size > 0:
                self.logger.debug(f"Using amount step size from limits.amount.min: {step_size}")
                return step_size

        except Exception as e:
            self.logger.warning(f"Could not reliably determine amount step size for {self.symbol} from market info: {e}. Using fallback.")

        # Generic fallback (very small step size)
        default_step = Decimal('1e-8')
        self.logger.warning(f"Using default/fallback amount step size {default_step} for {self.symbol}.")
        return default_step

    def _determine_amount_precision(self) -> int:
        """Determines the number of decimal places for amount based on step size."""
        try:
            if self.amount_step_size > 0:
                # Calculate precision from the exponent of the normalized step size
                precision = abs(self.amount_step_size.normalize().as_tuple().exponent)
                self.logger.debug(f"Determined amount precision from step size: {precision}")
                return precision
        except Exception:
            pass
        self.logger.warning(f"Could not determine amount precision for {self.symbol}. Using fallback: 8")
        return 8 # Default fallback precision

    # --- Public Accessors for Precision Info ---
    def get_price_precision(self) -> int: return self.price_precision
    def get_amount_precision(self) -> int: return self.amount_precision
    def get_min_tick_size(self) -> Decimal: return self.min_tick_size
    def get_amount_step_size(self) -> Decimal: return self.amount_step_size

    # --- Rounding Functions ---
    def round_price(self, price: Union[Decimal, float, str, None], rounding_mode=ROUND_HALF_UP) -> Optional[Decimal]:
        """
        Rounds a given price to the nearest valid tick size using the specified rounding mode.
        Returns None if input price or tick size is invalid.
        Default rounding: ROUND_HALF_UP (standard rounding).
        Use ROUND_UP/ROUND_DOWN for conservative SL/TP rounding.
        """
        price_dec = safe_decimal(price)
        min_tick = self.min_tick_size # Already determined in __init__

        if price_dec is None or min_tick is None or min_tick <= 0:
            self.logger.error(f"Cannot round price: Invalid input price ({price}) or min_tick_size ({min_tick}).")
            return None
        try:
            # Quantize the price to the nearest multiple of the tick size
            rounded_price = (price_dec / min_tick).quantize(Decimal('1'), rounding=rounding_mode) * min_tick
            return rounded_price
        except (InvalidOperation, Exception) as e:
            self.logger.error(f"Error rounding price {price_dec} with tick size {min_tick}: {e}")
            return None

    def round_amount(self, amount: Union[Decimal, float, str, None]) -> Optional[Decimal]:
        """
        Rounds (floors) an order amount DOWN to the nearest valid amount step size.
        Ensures the order quantity meets exchange requirements.
        Returns None if input amount or step size is invalid.
        """
        amount_dec = safe_decimal(amount)
        step_size = self.amount_step_size # Already determined in __init__

        if amount_dec is None or step_size is None or step_size <= 0:
            self.logger.error(f"Cannot round amount: Invalid input amount ({amount}) or amount_step_size ({step_size}).")
            return None
        try:
            # Use floor division to round down to the nearest multiple of step_size
            # This ensures the quantity is always compliant or zero
            if amount_dec < 0:
                # Floor division behaves differently for negatives, consider if adjustment needed
                # For now, log a warning if rounding negative amounts.
                self.logger.warning(f"Attempting to round negative amount: {amount_dec}. Flooring towards negative infinity.")
            rounded_amount = (amount_dec // step_size) * step_size
            # Optional: Check if rounded amount is too small (below limits.amount.min) - requires market_info access
            # min_order_amount = safe_decimal(self.market_info.get('limits', {}).get('amount', {}).get('min'))
            # if min_order_amount and rounded_amount > 0 and rounded_amount < min_order_amount:
            #     self.logger.warning(f"Rounded amount {rounded_amount} is below minimum order size {min_order_amount}. Returning 0.")
            #     return Decimal('0')
            return rounded_amount
        except (InvalidOperation, Exception) as e:
            self.logger.error(f"Error rounding amount {amount_dec} with step size {step_size}: {e}")
            return None

    # --- LiveXY Indicator Calculation Helpers ---
    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
         """
         Helper to find the actual column name generated by pandas_ta for a given indicator.
         Tries specific patterns first, then falls back to searching.
         """
         cfg = self.config
         # Define expected patterns based on pandas_ta naming conventions and config parameters
         # Add more patterns as needed for other indicators
         expected_patterns = {
             "ATR": [f"ATRr_{cfg.get('livexy_atr_period', DEFAULT_ATR_PERIOD_LIVEXY)}"],
             "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
             "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
             "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
             "BBL": [f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)}"],
             "BBM": [f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)}"],
             "BBU": [f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)}"],
             "CCI": [f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_{Decimal('0.015')}"], # Default constant 0.015 for CCI in pandas_ta
             "WR": [f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
             "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
             "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
             "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
             "PSARl": [f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"], # Long stop
             "PSARs": [f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"], # Short stop
             "PSARaf": [f"PSARaf_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],# Acceleration factor
             "PSARr": [f"PSARr_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"], # Reversal boolean
             "SMA_10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
             "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
             "Volume_MA": [f"SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"], # Assuming SMA for volume MA
             "VWAP": ["VWAP_D"], # VWAP often calculated daily, name might vary based on implementation
         }

         patterns_to_check = expected_patterns.get(base_name, [])
         available_columns = result_df.columns.tolist()

         # Try exact pattern matches first
         for pattern in patterns_to_check:
             if pattern in available_columns:
                 self.logger.debug(f"Found column '{pattern}' for indicator '{base_name}'")
                 return pattern

         # Fallback: Search for columns starting with the base name or containing it (case-insensitive)
         base_lower = base_name.lower()
         # Check for base name followed by underscore (common pattern)
         for col in available_columns:
             col_lower = col.lower()
             if col_lower.startswith(base_lower + "_"):
                 self.logger.debug(f"Found column '{col}' for indicator '{base_name}' (prefix match)")
                 return col
         # Check if base name is part of the column name
         for col in available_columns:
              col_lower = col.lower()
              if base_lower in col_lower:
                 self.logger.debug(f"Found column '{col}' for indicator '{base_name}' (substring match)")
                 return col

         self.logger.warning(f"Could not find column name for indicator '{base_name}' using patterns {patterns_to_check} or fallback search in columns: {available_columns}")
         return None

    def _calculate_livexy_indicators(self, df_calc: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates LiveXY strategy indicators using pandas_ta based on config flags.
        Uses temporary float columns for calculations and stores the generated TA column names.
        Returns the DataFrame with added indicator columns (as floats/bools).
        """
        if not self.livexy_enabled:
            return df_calc # Skip if LiveXY is disabled

        self.logger.info("Calculating LiveXY strategy indicators...")
        cfg = self.config
        indi_cfg = cfg.get("indicators", {}) # Which indicators are enabled
        self.livexy_ta_column_names = {} # Reset mappings for this run

        # --- Prepare float columns for pandas_ta ---
        # We expect df_calc to have the original Decimal columns. Create float versions.
        temp_float_cols = []
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in base_cols:
            float_col = f"{col}_float"
            if col in df_calc:
                df_calc[float_col] = pd.to_numeric(df_calc[col], errors='coerce')
                temp_float_cols.append(float_col)
            else:
                 self.logger.error(f"Missing base column '{col}' for LiveXY calculations.")
                 return df_calc # Cannot proceed

        # Assign float column names for convenience
        open_f, high_f, low_f, close_f, volume_f = [f"{col}_float" for col in base_cols]

        try:
            # --- Calculate indicators based on config flags ---
            # Note: pandas_ta appends columns by default if append=True

            # ATR (often used even if not directly weighted, e.g., for other calcs)
            # Use the specific LiveXY ATR period from config
            atr_period = cfg.get("livexy_atr_period", DEFAULT_ATR_PERIOD_LIVEXY)
            df_calc.ta.atr(high=df_calc[high_f], low=df_calc[low_f], close=df_calc[close_f],
                           length=atr_period, append=True)
            self.livexy_ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            if indi_cfg.get("ema_alignment"):
                ema_short_p = cfg.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                ema_long_p = cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                df_calc.ta.ema(close=df_calc[close_f], length=ema_short_p, append=True)
                self.livexy_ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
                df_calc.ta.ema(close=df_calc[close_f], length=ema_long_p, append=True)
                self.livexy_ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)

            if indi_cfg.get("rsi"):
                rsi_p = cfg.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(close=df_calc[close_f], length=rsi_p, append=True)
                self.livexy_ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indi_cfg.get("bollinger_bands"):
                bb_p = cfg.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = cfg.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
                df_calc.ta.bbands(close=df_calc[close_f], length=bb_p, std=bb_std, append=True)
                self.livexy_ta_column_names["BBL"] = self._get_ta_col_name("BBL", df_calc) # Lower band
                self.livexy_ta_column_names["BBM"] = self._get_ta_col_name("BBM", df_calc) # Middle band (SMA)
                self.livexy_ta_column_names["BBU"] = self._get_ta_col_name("BBU", df_calc) # Upper band

            if indi_cfg.get("cci"):
                cci_p = cfg.get("cci_window", DEFAULT_CCI_WINDOW)
                df_calc.ta.cci(high=df_calc[high_f], low=df_calc[low_f], close=df_calc[close_f],
                               length=cci_p, append=True)
                self.livexy_ta_column_names["CCI"] = self._get_ta_col_name("CCI", df_calc)

            if indi_cfg.get("wr"):
                wr_p = cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
                df_calc.ta.willr(high=df_calc[high_f], low=df_calc[low_f], close=df_calc[close_f],
                                 length=wr_p, append=True)
                self.livexy_ta_column_names["WR"] = self._get_ta_col_name("WR", df_calc)

            if indi_cfg.get("mfi"):
                mfi_p = cfg.get("mfi_window", DEFAULT_MFI_WINDOW)
                df_calc.ta.mfi(high=df_calc[high_f], low=df_calc[low_f], close=df_calc[close_f], volume=df_calc[volume_f],
                               length=mfi_p, append=True)
                self.livexy_ta_column_names["MFI"] = self._get_ta_col_name("MFI", df_calc)

            if indi_cfg.get("stoch_rsi"):
                stoch_p = cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)
                rsi_p = cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)
                k_p = cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)
                d_p = cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)
                # Calculate StochRSI - it returns a DataFrame, need to merge columns
                stochrsi_result = df_calc.ta.stochrsi(close=df_calc[close_f], length=stoch_p, rsi_length=rsi_p, k=k_p, d=d_p)
                if stochrsi_result is not None and not stochrsi_result.empty:
                    # Append results to df_calc, overwriting if columns already exist (shouldn't normally)
                    for col in stochrsi_result.columns:
                        df_calc[col] = stochrsi_result[col]
                    self.livexy_ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                    self.livexy_ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)
                else:
                    self.logger.warning(f"StochRSI calculation returned empty or None for {self.symbol}.")
                    self.livexy_ta_column_names["StochRSI_K"] = None
                    self.livexy_ta_column_names["StochRSI_D"] = None

            if indi_cfg.get("psar"):
                psar_af = cfg.get("psar_af", DEFAULT_PSAR_AF)
                psar_max = cfg.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                psar_result = df_calc.ta.psar(high=df_calc[high_f], low=df_calc[low_f], af=psar_af, max_af=psar_max)
                if psar_result is not None and not psar_result.empty:
                    for col in psar_result.columns:
                         df_calc[col] = psar_result[col]
                    self.livexy_ta_column_names["PSARl"] = self._get_ta_col_name("PSARl", df_calc) # Long stop
                    self.livexy_ta_column_names["PSARs"] = self._get_ta_col_name("PSARs", df_calc) # Short stop
                    self.livexy_ta_column_names["PSARaf"] = self._get_ta_col_name("PSARaf", df_calc) # AF
                    self.livexy_ta_column_names["PSARr"] = self._get_ta_col_name("PSARr", df_calc) # Reversal
                else:
                    self.logger.warning(f"PSAR calculation returned empty or None for {self.symbol}.")
                    # Set all PSAR related names to None if calculation failed
                    for key in ["PSARl", "PSARs", "PSARaf", "PSARr"]: self.livexy_ta_column_names[key] = None

            if indi_cfg.get("sma_10"):
                sma_p = cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(close=df_calc[close_f], length=sma_p, append=True)
                self.livexy_ta_column_names["SMA_10"] = self._get_ta_col_name("SMA_10", df_calc)

            if indi_cfg.get("momentum"):
                mom_p = cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                df_calc.ta.mom(close=df_calc[close_f], length=mom_p, append=True)
                self.livexy_ta_column_names["Momentum"] = self._get_ta_col_name("Momentum", df_calc)

            if indi_cfg.get("volume_confirmation"): # Requires Volume MA
                vol_ma_p = cfg.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                # Calculate SMA of volume
                df_calc.ta.sma(close=df_calc[volume_f], length=vol_ma_p, append=True, col_names=("Volume_MA",)) # Specify column name
                # Find the generated column name (might not be exactly "Volume_MA")
                # We need to find the SMA of the volume column
                vol_ma_col = None
                pattern = f"SMA_{vol_ma_p}"
                for col in df_calc.columns:
                    # Check if it's an SMA column and based on the volume_float column implicitly or explicitly
                    if col.startswith(pattern) and col != self.livexy_ta_column_names.get("SMA_10"): # Avoid conflict with price SMA
                        # Heuristic: Assume the last calculated SMA is the volume one if name isn't exact
                        vol_ma_col = col
                        break
                if vol_ma_col:
                     self.livexy_ta_column_names["Volume_MA"] = vol_ma_col
                     self.logger.debug(f"Found Volume MA column: {vol_ma_col}")
                else:
                     self.logger.warning(f"Could not reliably identify Volume MA (SMA_{vol_ma_p}) column.")
                     self.livexy_ta_column_names["Volume_MA"] = None


            if indi_cfg.get("vwap"):
                # VWAP calculation often requires daily resets. Pandas TA might handle this,
                # but results can vary depending on data frequency and implementation.
                # Simple VWAP calculation for the available data chunk:
                df_calc.ta.vwap(high=df_calc[high_f], low=df_calc[low_f], close=df_calc[close_f], volume=df_calc[volume_f], append=True)
                # VWAP column name is usually just 'VWAP_D' or similar, check carefully
                self.livexy_ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            self.logger.info("LiveXY indicator calculations complete.")

        except AttributeError as e:
            # Often happens if pandas_ta strategy method doesn't exist or typo in name
            self.logger.error(f"{COLOR_ERROR}AttributeError calculating LiveXY indicators for {self.symbol}. "
                              f"Check pandas_ta installation, version, and indicator names. Error: {e}{RESET}", exc_info=True)
            # Return df_calc without potentially partial calculations
            return df_calc
        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error calculating LiveXY indicators for {self.symbol}: {e}{RESET}", exc_info=True)
            # Return df_calc, potentially with partial calculations
            return df_calc
        finally:
            # --- Clean up temporary float columns ---
            df_calc.drop(columns=temp_float_cols, inplace=True, errors='ignore')

        return df_calc

    # --- Combined Indicator Calculation ---
    def _calculate_indicators(self) -> None:
        """
        Calculates all enabled indicators:
        1. Risk Management ATR ('atr_risk', 'atr_risk_dec').
        2. LiveXY strategy indicators (if enabled).
        3. Volbot strategy indicators (if enabled).
        Populates self.df_processed with the results.
        """
        if self.df_raw.empty:
            self.logger.warning(f"{COLOR_WARNING}Raw DataFrame is empty for {self.symbol}. Cannot calculate indicators.{RESET}")
            self.df_processed = pd.DataFrame() # Ensure processed df is empty
            return

        # Basic check for minimum data length, although individual calcs might have stricter needs
        min_data_points = 50 # A reasonable minimum for most indicators to start producing values
        if len(self.df_raw) < min_data_points:
             self.logger.warning(f"{COLOR_WARNING}Insufficient raw data ({len(self.df_raw)} points) for reliable analysis of {self.symbol}. "
                                 f"Calculations might result in NaNs or be inaccurate.{RESET}")
             # Proceed with calculation, but be aware of potential issues

        try:
            # Start with a copy of the raw data (which should have Decimal types)
            df_calc = self.df_raw.copy()

            # --- Ensure Base Columns Exist ---
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_calc.columns for col in required_cols):
                 missing = [col for col in required_cols if col not in df_calc.columns]
                 self.logger.error(f"{COLOR_ERROR}Missing required columns in raw data for {self.symbol}: {missing}. Cannot calculate indicators.{RESET}")
                 self.df_processed = pd.DataFrame(); return

            # --- Calculate Risk Management ATR ---
            # Create temporary float columns for TA library
            high_f = pd.to_numeric(df_calc['high'], errors='coerce')
            low_f = pd.to_numeric(df_calc['low'], errors='coerce')
            close_f = pd.to_numeric(df_calc['close'], errors='coerce')

            atr_period_risk = self.config.get("atr_period_risk", DEFAULT_ATR_PERIOD_RISK)
            # Calculate ATR using pandas_ta on float values
            atr_risk_float = ta.atr(high=high_f, low=low_f, close=close_f, length=atr_period_risk)

            # Store both float and Decimal versions
            df_calc['atr_risk'] = atr_risk_float # Float version
            df_calc['atr_risk_dec'] = atr_risk_float.apply(lambda x: safe_decimal(x, default=pd.NA)).astype(object) # Decimal version
            self.logger.debug(f"Calculated Risk ATR (Length: {atr_period_risk}) for {self.symbol}")

            # --- Calculate LiveXY Strategy Indicators ---
            if self.livexy_enabled:
                # This function uses df_calc, adds LiveXY columns, and returns the modified df_calc
                df_calc = self._calculate_livexy_indicators(df_calc)

            # --- Calculate Volbot Strategy Indicators ---
            if self.volbot_enabled:
                # These functions take df_calc (with Decimals), may create temp floats,
                # add Volbot columns ('_strat'), and return the modified df_calc.
                # Volatility Levels adds columns directly to df_calc copy and returns merged.
                df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
                # Pivots adds 'ph_strat', 'pl_strat' to df_calc copy and returns merged.
                df_calc = calculate_pivot_order_blocks(df_calc, self.config, self.logger)
                # OB Management adds active OB refs, returns df and box lists.
                df_calc, self.all_bull_boxes, self.all_bear_boxes = manage_order_blocks(df_calc, self.config, self.logger)

            # --- Finalize ---
            # df_calc now contains raw data + risk ATR + LiveXY cols + Volbot cols
            self.df_processed = df_calc
            self.logger.debug(f"Indicator calculations complete for {self.symbol}. Processed DataFrame has {len(self.df_processed)} rows.")
            # Log columns present in the final processed DataFrame for debugging
            self.logger.debug(f"Processed DF columns: {self.df_processed.columns.tolist()}")

        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error during combined indicator calculation for {self.symbol}: {e}{RESET}", exc_info=True)
            # Ensure df_processed is empty or reset in case of failure
            self.df_processed = pd.DataFrame()

    # --- LiveXY Signal Generation Helpers ---
    def _calculate_livexy_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score based on latest LiveXY indicator values."""
        # Uses self.livexy_indicator_values which should contain latest *float* values
        ema_short = self.livexy_indicator_values.get("EMA_Short", np.nan)
        ema_long = self.livexy_indicator_values.get("EMA_Long", np.nan)
        # Use the 'Close' price from the *strategy state* (which should be Decimal)
        current_price_dec = self.strategy_state.get('close')
        current_price = float(current_price_dec) if isinstance(current_price_dec, Decimal) else np.nan

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            return np.nan # Cannot calculate score if values are missing

        # Score: +1 for bullish alignment, -1 for bearish, 0 otherwise
        if current_price > ema_short > ema_long:
            return 1.0
        elif current_price < ema_short < ema_long:
            return -1.0
        else:
            return 0.0

    def _calculate_livexy_volume_confirmation_score(self) -> float:
        """Calculates volume confirmation score."""
        current_volume_dec = self.strategy_state.get('volume')
        current_volume = float(current_volume_dec) if isinstance(current_volume_dec, Decimal) else np.nan
        volume_ma = self.livexy_indicator_values.get("Volume_MA", np.nan)
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma <= 0:
            return np.nan

        # Score: +1 if volume significantly above MA, 0 otherwise (can refine for low volume)
        if current_volume > volume_ma * multiplier:
            return 1.0
        # Optional: Add negative score for very low volume?
        # elif current_volume < volume_ma * 0.5: return -0.5
        else:
            return 0.0

    def _calculate_livexy_stoch_rsi_score(self) -> float:
        """Calculates StochRSI score considering OB/OS and K/D relationship."""
        k = self.livexy_indicator_values.get("StochRSI_K", np.nan)
        d = self.livexy_indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan

        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
        score = 0.0

        # Strong signal if both K and D are in OB/OS zones
        if k < oversold and d < oversold: score = 1.0 # Oversold -> bullish signal
        elif k > overbought and d > overbought: score = -1.0 # Overbought -> bearish signal
        else:
            # Weaker signal based on K/D crossover or position
            if k > d: # K crossed above D or is leading up
                # Scale score based on how low K is (stronger signal near oversold)
                if k < 50: score = 0.6 - (k / (50 / 0.6)) # Linear scale from 0.6 down to 0
                else: score = 0.1 # Weak bullish indication if above 50
            elif k < d: # K crossed below D or is leading down
                # Scale score based on how high K is (stronger signal near overbought)
                if k > 50: score = -0.6 + ((100 - k) / (50 / 0.6)) # Linear scale from -0.6 up to 0
                else: score = -0.1 # Weak bearish indication if below 50
            # If k == d, score remains 0

        return max(-1.0, min(1.0, score)) # Clamp score between -1 and 1

    def _calculate_livexy_rsi_score(self) -> float:
        """Calculates RSI score based on OB/OS levels."""
        rsi = self.livexy_indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan

        # Simple OB/OS scoring, can be refined (e.g., divergence)
        if rsi <= 30: return 1.0   # Oversold -> Bullish
        if rsi >= 70: return -1.0  # Overbought -> Bearish
        # Add intermediate levels for weaker signals
        if rsi < 40: return 0.3   # Approaching oversold
        if rsi > 60: return -0.3  # Approaching overbought
        # Neutral zone
        return 0.0

    def _calculate_livexy_bbands_score(self) -> float:
        """Calculates Bollinger Bands score based on price position relative to bands."""
        bbl = self.livexy_indicator_values.get("BBL", np.nan)
        bbu = self.livexy_indicator_values.get("BBU", np.nan)
        current_price_dec = self.strategy_state.get('close')
        current_price = float(current_price_dec) if isinstance(current_price_dec, Decimal) else np.nan

        if pd.isna(bbl) or pd.isna(bbu) or pd.isna(current_price) or bbu <= bbl:
            return np.nan

        # Score based on touching/exceeding bands (mean reversion idea)
        if current_price <= bbl: return 1.0   # Touching/below lower band -> Bullish reversal potential
        if current_price >= bbu: return -1.0  # Touching/above upper band -> Bearish reversal potential
        # Optional: Score based on distance from middle band?
        # bbm = self.livexy_indicator_values.get("BBM", np.nan)
        # if pd.notna(bbm): score = (current_price - bbm) / ((bbu - bbl) / 2 + 1e-9) ...
        return 0.0 # Inside the bands

    def _calculate_livexy_cci_score(self) -> float:
        """Calculates CCI score based on extreme levels."""
        cci = self.livexy_indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan

        if cci <= -100: return 1.0  # Below -100 -> Bullish potential
        if cci >= 100: return -1.0   # Above +100 -> Bearish potential
        if cci < 0: return 0.2     # Below 0, slight bullish lean
        if cci > 0: return -0.2    # Above 0, slight bearish lean
        return 0.0

    def _calculate_livexy_wr_score(self) -> float:
        """Calculates Williams %R score based on OB/OS levels."""
        wr = self.livexy_indicator_values.get("WR", np.nan)
        if pd.isna(wr): return np.nan

        # Williams %R is typically -100 to 0
        if wr <= -80: return 1.0  # Oversold (e.g., <= -80) -> Bullish
        if wr >= -20: return -1.0 # Overbought (e.g., >= -20) -> Bearish
        if wr < -50: return 0.2  # Below midpoint, slight bullish lean
        if wr > -50: return -0.2 # Above midpoint, slight bearish lean
        return 0.0

    def _calculate_livexy_mfi_score(self) -> float:
        """Calculates Money Flow Index score based on OB/OS levels."""
        mfi = self.livexy_indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan

        if mfi <= 20: return 1.0   # Oversold -> Bullish
        if mfi >= 80: return -1.0  # Overbought -> Bearish
        if mfi < 40: return 0.3
        if mfi > 60: return -0.3
        return 0.0

    def _calculate_livexy_psar_score(self) -> float:
        """Calculates PSAR score based on position relative to price."""
        psar_l = self.livexy_indicator_values.get("PSARl", np.nan) # PSAR value when trend is long
        psar_s = self.livexy_indicator_values.get("PSARs", np.nan) # PSAR value when trend is short
        # psar_r = self.livexy_indicator_values.get("PSARr", np.nan) # Reversal boolean
        current_price_dec = self.strategy_state.get('close')
        current_price = float(current_price_dec) if isinstance(current_price_dec, Decimal) else np.nan

        if pd.isna(current_price): return np.nan

        # Determine current PSAR value: if PSARl is NaN, use PSARs, otherwise use PSARl
        current_psar = psar_s if pd.isna(psar_l) else psar_l

        if pd.isna(current_psar): return np.nan # Cannot determine score if PSAR is NaN

        # Score: +1 if price is above PSAR (bullish), -1 if below (bearish)
        if current_price > current_psar: return 1.0
        elif current_price < current_psar: return -1.0
        else: return 0.0 # Price exactly on PSAR

    def _calculate_livexy_sma10_score(self) -> float:
        """Calculates score based on price relative to SMA 10."""
        sma10 = self.livexy_indicator_values.get("SMA_10", np.nan)
        current_price_dec = self.strategy_state.get('close')
        current_price = float(current_price_dec) if isinstance(current_price_dec, Decimal) else np.nan

        if pd.isna(sma10) or pd.isna(current_price): return np.nan

        if current_price > sma10: return 1.0
        if current_price < sma10: return -1.0
        return 0.0

    def _calculate_livexy_momentum_score(self) -> float:
        """Calculates score based on the sign of the momentum."""
        momentum = self.livexy_indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan

        if momentum > 0: return 1.0
        if momentum < 0: return -1.0
        return 0.0

    def _calculate_livexy_vwap_score(self) -> float:
        """Calculates score based on price relative to VWAP."""
        vwap = self.livexy_indicator_values.get("VWAP", np.nan)
        current_price_dec = self.strategy_state.get('close')
        current_price = float(current_price_dec) if isinstance(current_price_dec, Decimal) else np.nan

        if pd.isna(vwap) or pd.isna(current_price): return np.nan

        if current_price > vwap: return 1.0
        if current_price < vwap: return -1.0
        return 0.0

    def _calculate_livexy_orderbook_score(self) -> float:
        """Calculates score based on order book imbalance."""
        if not self.orderbook_data:
            self.logger.debug("Order book data not available for scoring.")
            return np.nan
        try:
            bids = self.orderbook_data.get('bids', [])
            asks = self.orderbook_data.get('asks', [])
            if not bids or not asks:
                self.logger.debug("Empty bids or asks in order book data.")
                return np.nan

            # Analyze top N levels (configurable?)
            n_levels = 10 # Consider making this configurable
            # Sum the *quantity* at the top N levels
            bid_vol = sum(safe_decimal(bid[1], Decimal(0)) for bid in bids[:n_levels])
            ask_vol = sum(safe_decimal(ask[1], Decimal(0)) for ask in asks[:n_levels])
            total_vol = bid_vol + ask_vol

            if total_vol <= 0:
                return 0.0 # No volume in top levels

            # Calculate Order Book Imbalance (OBI)
            # OBI = (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)
            # Ranges from -1 (heavy asks) to +1 (heavy bids)
            obi = (bid_vol - ask_vol) / total_vol
            # Convert Decimal OBI to float score
            return float(obi)

        except Exception as e:
            self.logger.warning(f"Error calculating order book score: {e}")
            return np.nan


    def _check_livexy_indicator(self, indicator_key: str) -> float:
         """
         Dispatcher function that calls the appropriate _calculate_livexy_*_score method
         based on the indicator key. Returns the calculated float score, or np.nan if failed.
         """
         # Map indicator keys (from config) to their calculation functions
         score_calculators = {
             "ema_alignment": self._calculate_livexy_ema_alignment_score,
             "momentum": self._calculate_livexy_momentum_score,
             "volume_confirmation": self._calculate_livexy_volume_confirmation_score,
             "stoch_rsi": self._calculate_livexy_stoch_rsi_score,
             "rsi": self._calculate_livexy_rsi_score,
             "bollinger_bands": self._calculate_livexy_bbands_score,
             "vwap": self._calculate_livexy_vwap_score,
             "cci": self._calculate_livexy_cci_score,
             "wr": self._calculate_livexy_wr_score,
             "psar": self._calculate_livexy_psar_score,
             "sma_10": self._calculate_livexy_sma10_score,
             "mfi": self._calculate_livexy_mfi_score,
             "orderbook": self._calculate_livexy_orderbook_score,
         }

         calculator = score_calculators.get(indicator_key)
         if calculator:
             try:
                 score = calculator()
                 # Ensure score is float or NaN
                 return float(score) if pd.notna(score) else np.nan
             except Exception as e:
                  self.logger.error(f"Error calculating score for LiveXY indicator '{indicator_key}': {e}", exc_info=True)
                  return np.nan
         else:
             # Fallback if check method not implemented for an enabled indicator
             self.logger.warning(f"No specific score calculation logic implemented for enabled LiveXY indicator: '{indicator_key}'. Returning NaN.")
             return np.nan

    def _generate_livexy_signal(self) -> Tuple[str, float]:
        """
        Generates a BUY/SELL/HOLD signal based on the weighted sum of enabled LiveXY indicator scores.
        Returns the signal string and the final calculated score (float).
        """
        if not self.livexy_enabled:
            return "HOLD", 0.0
        if not self.livexy_weights:
            self.logger.error("LiveXY signal generation skipped: Active weight set is missing or empty.")
            return "HOLD", 0.0

        signal = "HOLD"
        final_score = Decimal("0.0")
        total_weight = Decimal("0.0")
        contributing_indicators = 0

        # --- Update livexy_indicator_values with the latest float values ---
        # This ensures the _check_livexy_indicator methods use the most recent data
        if self.df_processed.empty:
            self.logger.warning("LiveXY signal generation skipped: Processed DataFrame is empty.")
            return "HOLD", 0.0
        try:
            latest_row = self.df_processed.iloc[-1]
            temp_livexy_vals = {}
            # Extract values for indicators that have calculated column names
            for key, col_name in self.livexy_ta_column_names.items():
                if col_name and col_name in latest_row.index and pd.notna(latest_row[col_name]):
                    try:
                        # Convert to float for consistent scoring logic
                        temp_livexy_vals[key] = float(latest_row[col_name])
                    except (ValueError, TypeError):
                        temp_livexy_vals[key] = np.nan # Assign NaN if conversion fails
                else:
                    temp_livexy_vals[key] = np.nan # Assign NaN if column missing or value is NA
            # Add core values needed by checks (Volume MA needs volume) - get from strategy_state if possible
            # Note: Already handled by individual calculators using strategy_state where needed.
            # self.livexy_indicator_values["Volume"] = float(self.strategy_state.get('volume', np.nan))

            self.livexy_indicator_values = temp_livexy_vals # Update the instance variable
        except IndexError:
             self.logger.error("Failed to get latest row from processed DataFrame for LiveXY values.")
             return "HOLD", 0.0
        except Exception as e:
            self.logger.error(f"Failed to update livexy_indicator_values: {e}", exc_info=True)
            return "HOLD", 0.0
        # --- End update ----

        # --- Calculate Weighted Score ---
        debug_scores = {} # For logging individual scores
        enabled_indicators = self.config.get("indicators", {})

        for key, is_enabled in enabled_indicators.items():
            if not is_enabled: continue # Skip disabled indicators

            # Get weight from the active weight set
            weight_str = self.livexy_weights.get(key)
            if weight_str is None:
                # Log if an enabled indicator is missing a weight
                self.logger.debug(f"LiveXY indicator '{key}' is enabled but missing from active weight set '{self.livexy_active_weight_set_name}'. Skipping.")
                continue

            weight = safe_decimal(weight_str, Decimal(0))
            if weight == Decimal(0): continue # Skip indicators with zero weight

            # Get the score for this indicator (-1 to 1 or NaN)
            score_float = self._check_livexy_indicator(key)

            # Store score for debugging before handling NaN
            debug_scores[key] = f"{score_float:.2f}" if pd.notna(score_float) else "NaN"

            if pd.notna(score_float):
                 # Convert valid score to Decimal for weighted sum
                 score_dec = safe_decimal(score_float, Decimal(0))
                 # Clamp score just in case calculator returned >|1|
                 clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_dec))
                 final_score += clamped_score * weight
                 total_weight += weight
                 contributing_indicators += 1
            else:
                 self.logger.debug(f"LiveXY indicator '{key}' returned NaN score. Not included in final score.")


        # --- Determine Final Signal ---
        if total_weight > 0:
            # Normalize score? Optional. Current logic uses raw weighted score vs threshold.
            # normalized_score = final_score / total_weight
            score_threshold = safe_decimal(self.config.get("signal_score_threshold", 1.5), Decimal(1.5))

            if final_score >= score_threshold:
                signal = "BUY"
            elif final_score <= -score_threshold:
                signal = "SELL"
            # Else: signal remains "HOLD"

            score_details = ", ".join([f"{k}:{v}" for k, v in sorted(debug_scores.items())])
            self.logger.debug(f"LiveXY Signal Calc: Score={final_score:.4f}, TotalWeight={total_weight:.2f}, "
                              f"Threshold={score_threshold}, Signal={signal}")
            if contributing_indicators > 0 and _log_level_container['level'] <= logging.DEBUG:
                 self.logger.debug(f"  LiveXY Scores ({contributing_indicators} contributing): {score_details}")
            elif contributing_indicators == 0:
                 self.logger.warning("LiveXY: No indicators contributed a valid score.")

        elif contributing_indicators == 0:
             self.logger.warning("LiveXY: No enabled indicators provided valid scores or weights. Signal: HOLD")
        else: # Should not happen if contributing_indicators > 0, but safety check
             self.logger.warning(f"LiveXY: Total weight is zero despite {contributing_indicators} contributing indicators? Signal: HOLD")

        # Return signal and the raw final score (as float)
        return signal, float(final_score)

    # --- Volbot Signal Generation Helper ---
    def _generate_volbot_signal(self) -> str:
        """
        Generates a BUY/SELL/HOLD signal based on Volbot strategy rules (trend, OB entry).
        Uses the latest data stored in self.strategy_state.
        Returns the signal string.
        """
        if not self.volbot_enabled:
            return "HOLD"

        signal = "HOLD"
        try:
            # Extract latest Volbot state values
            is_trend_up = self.strategy_state.get('trend_up_strat') # Boolean or None
            trend_changed = self.strategy_state.get('trend_changed_strat', False) # Boolean
            # Check if price is currently inside an *active* OB
            is_in_bull_ob = self.strategy_state.get('is_in_active_bull_ob', False) # Boolean derived in state update
            is_in_bear_ob = self.strategy_state.get('is_in_active_bear_ob', False) # Boolean derived in state update

            # Get config flags for signal generation triggers
            signal_on_flip = self.config.get("volbot_signal_on_trend_flip", True)
            signal_on_ob = self.config.get("volbot_signal_on_ob_entry", True)

            # Need trend direction to generate signals
            if is_trend_up is None:
                self.logger.debug("Volbot signal: HOLD (Trend direction undetermined - likely insufficient data or calculation error).")
                return "HOLD"

            # --- Log Current State ---
            trend_str = f"{COLOR_UP}UP{RESET}" if is_trend_up else f"{COLOR_DN}DOWN{RESET}"
            ob_status = ""
            if is_in_bull_ob: ob_status += f" {COLOR_BULL_BOX}InBullOB{RESET}"
            if is_in_bear_ob: ob_status += f" {COLOR_BEAR_BOX}InBearOB{RESET}"
            if not is_in_bull_ob and not is_in_bear_ob: ob_status = " NoActiveOB"
            self.logger.debug(f"Volbot State Check: Trend={trend_str}, Changed={trend_changed},{ob_status}")

            # --- Signal Logic ---
            # 1. Signal on Trend Flip
            if signal_on_flip and trend_changed:
                signal = "BUY" if is_trend_up else "SELL"
                reason = f"Trend flipped to {trend_str}"
                color = COLOR_UP if is_trend_up else COLOR_DN
                self.logger.debug(f"{color}Volbot Signal Trigger: {signal} (Reason: {reason}){RESET}")
                return signal # Return immediately on flip signal

            # 2. Signal on Order Block Entry (only if trend flip didn't trigger)
            if signal_on_ob:
                # Buy signal: Trend is UP and price entered an ACTIVE Bullish OB
                if is_trend_up and is_in_bull_ob:
                    signal = "BUY"
                    ob_id = self.latest_active_bull_ob.get('id', 'N/A') if self.latest_active_bull_ob else 'N/A'
                    reason = f"Price entered Bull OB '{ob_id}' during UP trend"
                    self.logger.debug(f"{COLOR_BULL_BOX}Volbot Signal Trigger: {signal} (Reason: {reason}){RESET}")
                    return signal # Return immediately on OB entry signal

                # Sell signal: Trend is DOWN and price entered an ACTIVE Bearish OB
                elif not is_trend_up and is_in_bear_ob:
                    signal = "SELL"
                    ob_id = self.latest_active_bear_ob.get('id', 'N/A') if self.latest_active_bear_ob else 'N/A'
                    reason = f"Price entered Bear OB '{ob_id}' during DOWN trend"
                    self.logger.debug(f"{COLOR_BEAR_BOX}Volbot Signal Trigger: {signal} (Reason: {reason}){RESET}")
                    return signal # Return immediately on OB entry signal

            # If neither flip nor OB entry triggered a signal
            self.logger.debug(f"Volbot Signal: HOLD (No triggering condition met based on config: Flip={signal_on_flip}, OBEntry={signal_on_ob})")
            return "HOLD"

        except KeyError as e:
            # This might happen if expected keys are missing from strategy_state
            self.logger.error(f"{COLOR_ERROR}Volbot Signal Error: Missing key in strategy state: '{e}'. Defaulting to HOLD.{RESET}")
            return "HOLD"
        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error during Volbot signal generation: {e}{RESET}", exc_info=True)
            return "HOLD"

    # --- Combined Signal Generation ---
    def generate_trading_signal(self) -> str:
        """
        Generates the final trading signal ('BUY', 'SELL', 'HOLD') based on the
        enabled strategies (LiveXY, Volbot) and the configured 'signal_mode'.
        Logs the decision process and the final signal.
        """
        final_signal = "HOLD"
        livexy_signal, livexy_score = "HOLD", 0.0
        volbot_signal = "HOLD"

        # Get signals from enabled strategies
        if self.livexy_enabled:
            livexy_signal, livexy_score = self._generate_livexy_signal()
        if self.volbot_enabled:
            volbot_signal = self._generate_volbot_signal()

        # Determine final signal based on configured mode
        mode = self.config.get("signal_mode", "both_align")
        price_fmt = f".{self.price_precision}f"
        current_price = self.strategy_state.get('close', None) # Get latest close price (Decimal or None)
        price_str = f"{current_price:{price_fmt}}" if isinstance(current_price, Decimal) else "N/A"

        log_prefix = f"Signal Gen ({self.symbol} @ {price_str}):"
        enabled_str = f"LiveXY={'ON' if self.livexy_enabled else 'OFF'}, Volbot={'ON' if self.volbot_enabled else 'OFF'}, Mode='{mode}'"

        # --- Decision Logic ---
        if not self.livexy_enabled and not self.volbot_enabled:
            # Case 0: No strategies enabled
            self.logger.warning(f"{log_prefix} Final Signal: {signal_to_color('HOLD')} (Reason: No strategies enabled)")
            final_signal = "HOLD"

        elif self.livexy_enabled and not self.volbot_enabled:
            # Case 1: Only LiveXY enabled
            final_signal = livexy_signal
            self.logger.info(f"{log_prefix} Final Signal: {signal_to_color(final_signal)} (Source: LiveXY Score={livexy_score:.2f}) | {enabled_str}")

        elif not self.livexy_enabled and self.volbot_enabled:
            # Case 2: Only Volbot enabled
            final_signal = volbot_signal
            self.logger.info(f"{log_prefix} Final Signal: {signal_to_color(final_signal)} (Source: Volbot) | {enabled_str}")

        else:
            # Case 3: Both strategies enabled
            if mode == 'livexy':
                # Mode 'livexy': Prioritize LiveXY signal
                final_signal = livexy_signal
                self.logger.info(f"{log_prefix} Final Signal: {signal_to_color(final_signal)} (Priority: LiveXY Score={livexy_score:.2f}, Volbot Signal={volbot_signal}) | {enabled_str}")
            elif mode == 'volbot':
                # Mode 'volbot': Prioritize Volbot signal
                final_signal = volbot_signal
                self.logger.info(f"{log_prefix} Final Signal: {signal_to_color(final_signal)} (Priority: Volbot, LiveXY Signal={livexy_signal}) | {enabled_str}")
            elif mode == 'both_align':
                # Mode 'both_align': Signal only if both agree (and are not HOLD)
                if livexy_signal == volbot_signal and livexy_signal != "HOLD":
                    final_signal = livexy_signal # Signals align and are actionable
                    self.logger.info(f"{log_prefix} Final Signal: {signal_to_color(final_signal)} (Reason: Signals Aligned, LiveXY={livexy_signal}, Volbot={volbot_signal}) | {enabled_str}")
                else:
                    final_signal = "HOLD" # Signals don't align or both are HOLD
                    reason = "Signals Not Aligned" if livexy_signal != volbot_signal else "Both HOLD"
                    self.logger.info(f"{log_prefix} Final Signal: {signal_to_color(final_signal)} (Reason: {reason}, LiveXY={livexy_signal}, Volbot={volbot_signal}) | {enabled_str}")
            else:
                # Unknown mode - default to HOLD
                self.logger.error(f"{log_prefix} Unknown signal_mode '{mode}' in config. Defaulting to HOLD.")
                final_signal = "HOLD"

        return final_signal

    # --- Combined State Update ---
    def _update_latest_strategy_state(self) -> None:
        """
        Updates the combined `self.strategy_state` dictionary with the latest available values
        from the `self.df_processed` DataFrame. Converts relevant numeric values to Decimal.
        Also updates `self.latest_active_bull_ob` and `self.latest_active_bear_ob`.
        """
        self.strategy_state = {} # Reset state dictionary
        self.latest_active_bull_ob = None
        self.latest_active_bear_ob = None

        if self.df_processed.empty:
            self.logger.warning(f"Cannot update strategy state for {self.symbol}: Processed DataFrame is empty.")
            return
        if len(self.df_processed) == 0:
             self.logger.warning(f"Cannot update strategy state for {self.symbol}: Processed DataFrame has zero rows.")
             return

        try:
            # Get the last row of the processed DataFrame
            latest_row = self.df_processed.iloc[-1]

            # Check if the last row contains only NaNs (can happen with insufficient data)
            if latest_row.isnull().all():
                self.logger.warning(f"Cannot update strategy state for {self.symbol}: Last row of processed DataFrame contains all NaNs.")
                return

            # Define columns to extract and their target type/key in strategy_state
            # Format: 'column_name_in_df': 'key_name_in_state' or True (use col name, convert to Decimal) or False (use col name, keep original type)
            cols_to_extract = {
                # Core Data (already Decimal/Timestamp in df_processed)
                'open': True, 'high': True, 'low': True, 'close': True, 'volume': True,
                # Risk ATR (use pre-calculated Decimal column)
                'atr_risk_dec': 'atr_risk', # Use the Decimal version directly
                # LiveXY Indicators (mostly kept as float in df_processed, store as float/bool in state for now)
                # Note: LiveXY signal logic uses self.livexy_indicator_values (floats) separately
                # Volbot Indicators
                'ema1_strat': True, 'ema2_strat': True, 'atr_strat': True, # Convert Volbot floats to Decimal
                'trend_up_strat': False, # Keep as boolean
                'trend_changed_strat': False, # Keep as boolean
                'upper_strat': True, 'lower_strat': True, 'lower_vol_strat': True, 'upper_vol_strat': True,
                'step_up_strat': True, 'step_dn_strat': True,
                'vol_norm_strat': False, # Keep normalized volume as float (0-100)
                'vol_up_step_strat': True, 'vol_dn_step_strat': True,
                'vol_trend_up_level_strat': True, 'vol_trend_dn_level_strat': True,
                'cum_vol_delta_since_change_strat': True, 'cum_vol_total_since_change_strat': True,
                'last_trend_change_idx': False, # Keep as Timestamp or NaT
                'ph_strat': False, # Keep as Decimal or pd.NA (object type)
                'pl_strat': False, # Keep as Decimal or pd.NA (object type)
                'active_bull_ob_strat': False, # Keep as Dict or None (object type)
                'active_bear_ob_strat': False, # Keep as Dict or None (object type)
            }

            extracted_state = {}
            for col, state_key_or_flag in cols_to_extract.items():
                if col in latest_row.index: # Check if column exists
                    value = latest_row[col]
                    if pd.notna(value): # Only process non-NA values
                        key_name = state_key_or_flag if isinstance(state_key_or_flag, str) else col
                        convert_to_decimal = isinstance(state_key_or_flag, bool) and state_key_or_flag

                        if convert_to_decimal:
                             # Attempt conversion to Decimal
                             decimal_value = safe_decimal(value, default=None)
                             if decimal_value is not None:
                                 extracted_state[key_name] = decimal_value
                             else:
                                 # Log if conversion failed for a value that should be Decimal
                                 self.logger.warning(f"Failed to convert value '{value}' from column '{col}' to Decimal for state.")
                                 extracted_state[key_name] = None # Store None if conversion fails
                        else:
                             # Keep original type (bool, Timestamp, Decimal, object, dict, etc.)
                             extracted_state[key_name] = value
                    else:
                        # If value is NA/NaN, store None in state for consistency
                        key_name = state_key_or_flag if isinstance(state_key_or_flag, str) else col
                        extracted_state[key_name] = None
                # else: # Optionally log if an expected column is missing from df_processed
                #     self.logger.debug(f"Column '{col}' not found in processed DataFrame for state update.")


            # Update the main state dictionary
            self.strategy_state = extracted_state

            # Update latest OB references and derived boolean flags
            self.latest_active_bull_ob = self.strategy_state.get('active_bull_ob_strat')
            self.latest_active_bear_ob = self.strategy_state.get('active_bear_ob_strat')
            self.strategy_state['is_in_active_bull_ob'] = self.latest_active_bull_ob is not None
            self.strategy_state['is_in_active_bear_ob'] = self.latest_active_bear_ob is not None

            # --- Log the updated state (compactly) ---
            log_state = {}
            # Define formats for logging different types of values
            price_fmt = f".{self.price_precision}f"
            vol_fmt = ".2f" # Example format for volume
            atr_fmt = ".5f" # Example format for ATR
            default_fmt = ".8f" # Default for other Decimals

            for k, v in self.strategy_state.items():
                # Skip logging the full OB dictionaries
                if k in ['active_bull_ob_strat', 'active_bear_ob_strat']: continue

                if isinstance(v, Decimal):
                     # Choose format based on key name heuristic
                     fmt = price_fmt if any(p in k for p in ['price', 'level', 'strat', 'open', 'high', 'low', 'close', 'tp', 'sl', 'upper', 'lower', 'pivot']) \
                           else vol_fmt if 'vol' in k and 'level' not in k \
                           else atr_fmt if 'atr' in k \
                           else default_fmt
                     log_state[k] = f"{v:{fmt}}"
                elif isinstance(v, (bool, type(None), pd._libs.missing.NAType)):
                     log_state[k] = str(v) # Simple string representation
                elif isinstance(v, pd.Timestamp):
                     log_state[k] = v.strftime('%Y-%m-%d %H:%M:%S') # Full timestamp or just '%H:%M:%S'
                elif isinstance(v, float):
                     log_state[k] = f"{v:.4f}" # Format floats nicely
                else:
                     log_state[k] = repr(v)[:50] # Represent other types concisely

            self.logger.debug(f"Latest strategy state updated for {self.symbol}: {log_state}")
            # Log active OB IDs separately if they exist
            if self.latest_active_bull_ob:
                self.logger.debug(f"  Latest Active Bull OB: ID={self.latest_active_bull_ob.get('id', 'N/A')}, "
                                  f"Range=({self.latest_active_bull_ob.get('bottom')}, {self.latest_active_bull_ob.get('top')})")
            if self.latest_active_bear_ob:
                self.logger.debug(f"  Latest Active Bear OB: ID={self.latest_active_bear_ob.get('id', 'N/A')}, "
                                  f"Range=({self.latest_active_bear_ob.get('bottom')}, {self.latest_active_bear_ob.get('top')})")

        except IndexError:
            # This error occurs if iloc[-1] fails (e.g., empty DataFrame after filtering)
            self.logger.error(f"Error accessing latest row in processed DataFrame for {self.symbol}. State not updated.")
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest strategy state for {self.symbol}: {e}", exc_info=True)

    # --- Risk Management Calculation ---
    def calculate_entry_tp_sl(self, entry_price: Decimal, signal: str) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial Take Profit (TP) and Stop Loss (SL) levels based on the entry price,
        signal direction, Risk ATR, and configured multiples.
        Applies conservative rounding based on the minimum tick size.

        Args:
            entry_price: The entry price of the potential trade (as Decimal).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            A tuple containing: (entry_price, tp_price, sl_price).
            tp_price and sl_price will be Decimals if calculable and valid, otherwise None.
            Returns (entry_price, None, None) if inputs are invalid or calculation fails.
        """
        if signal not in ["BUY", "SELL"]:
            self.logger.warning(f"Cannot calculate TP/SL for non-BUY/SELL signal: {signal}")
            return entry_price, None, None

        # Get necessary values from state and config
        atr_val = self.strategy_state.get("atr_risk") # Use Risk ATR (should be Decimal or None)
        tp_mult_cfg = self.config.get("take_profit_multiple", 1.0)
        sl_mult_cfg = self.config.get("stop_loss_multiple", 1.5)
        min_tick = self.min_tick_size # Already validated Decimal

        # --- Input Validation ---
        valid = True
        tp_mult = safe_decimal(tp_mult_cfg)
        sl_mult = safe_decimal(sl_mult_cfg)

        if not isinstance(entry_price, Decimal) or entry_price <= 0:
            self.logger.error(f"TP/SL Calculation Error: Invalid entry_price ({entry_price}).")
            valid = False
        if not isinstance(atr_val, Decimal):
            # Allow calculation even if ATR is None/invalid, but SL/TP will likely be None
            self.logger.warning(f"TP/SL Calculation Warning: Risk ATR is invalid ({atr_val}). SL/TP will be based on entry only if multiples are 0.")
            atr_val = None # Treat invalid ATR as None for calculation
        elif atr_val <= 0:
            # Allow calculation with zero ATR (SL/TP will be based on entry + rounding)
             self.logger.warning(f"TP/SL Calculation Warning: Risk ATR is zero or negative ({atr_val}). Offsets will be zero.")
             atr_val = Decimal('0') # Treat non-positive ATR as zero offset
        if tp_mult is None or tp_mult < 0:
            self.logger.error(f"TP/SL Calculation Error: Invalid take_profit_multiple ({tp_mult_cfg}). Must be >= 0.")
            valid = False
        if sl_mult is None or sl_mult <= 0:
            self.logger.error(f"TP/SL Calculation Error: Invalid stop_loss_multiple ({sl_mult_cfg}). Must be > 0.")
            valid = False
        # min_tick is assumed valid from __init__

        if not valid:
            self.logger.error(f"Aborting TP/SL calculation due to invalid inputs for {self.symbol} {signal}.")
            return entry_price, None, None
        # --- End Validation ---

        try:
            tp_price, sl_price = None, None
            price_fmt = f'.{self.price_precision}f' # Format for logging

            # Calculate offsets only if ATR is valid
            tp_offset = atr_val * tp_mult if atr_val is not None else Decimal('0')
            sl_offset = atr_val * sl_mult if atr_val is not None else Decimal('0') # Should always have valid sl_mult > 0

            # Calculate raw TP/SL prices
            if signal == "BUY":
                tp_raw = entry_price + tp_offset
                sl_raw = entry_price - sl_offset
                tp_round_mode = ROUND_DOWN # Round TP down for longs (conservative)
                sl_round_mode = ROUND_DOWN # Round SL down for longs (aggressive, closer stop) - adjust if needed
            else: # signal == "SELL"
                tp_raw = entry_price - tp_offset
                sl_raw = entry_price + sl_offset
                tp_round_mode = ROUND_UP   # Round TP up for shorts (conservative)
                sl_round_mode = ROUND_UP   # Round SL up for shorts (aggressive, closer stop) - adjust if needed

            # --- Round TP ---
            # Only set TP if multiplier is > 0
            if tp_mult > 0 and tp_raw is not None:
                 tp_price = self.round_price(tp_raw, rounding_mode=tp_round_mode)
                 if tp_price is not None:
                     # Final check: Ensure rounded TP didn't cross entry price due to rounding
                     if signal == "BUY" and tp_price <= entry_price:
                         self.logger.warning(f"Rounded BUY TP {tp_price:{price_fmt}} is <= entry {entry_price:{price_fmt}}. Adjusting TP up by one tick.")
                         tp_price += min_tick
                     elif signal == "SELL" and tp_price >= entry_price:
                         self.logger.warning(f"Rounded SELL TP {tp_price:{price_fmt}} is >= entry {entry_price:{price_fmt}}. Adjusting TP down by one tick.")
                         tp_price -= min_tick
                     # Ensure TP is still positive after adjustment
                     if tp_price <= 0:
                          self.logger.error(f"Calculated TP {tp_price:{price_fmt}} is zero or negative. Setting TP to None.")
                          tp_price = None
                 else:
                     self.logger.error(f"Failed to round TP price {tp_raw}. Setting TP to None.")

            # --- Round SL ---
            if sl_raw is not None:
                sl_price = self.round_price(sl_raw, rounding_mode=sl_round_mode)
                if sl_price is not None:
                    # Final check: Ensure rounded SL didn't cross entry price
                    if signal == "BUY" and sl_price >= entry_price:
                        self.logger.warning(f"Rounded BUY SL {sl_price:{price_fmt}} is >= entry {entry_price:{price_fmt}}. Adjusting SL down by one tick.")
                        sl_price -= min_tick
                    elif signal == "SELL" and sl_price <= entry_price:
                         self.logger.warning(f"Rounded SELL SL {sl_price:{price_fmt}} is <= entry {entry_price:{price_fmt}}. Adjusting SL up by one tick.")
                         sl_price += min_tick
                    # Ensure SL is positive
                    if sl_price <= 0:
                         self.logger.error(f"Calculated SL {sl_price:{price_fmt}} is zero or negative. Setting SL to None.")
                         sl_price = None # Critical: Cannot place zero/negative SL
                else:
                    self.logger.error(f"Failed to round SL price {sl_raw}. Setting SL to None.")
                    sl_price = None # Critical: Need a valid SL

            # Log results
            tp_str = f"{tp_price:{price_fmt}}" if tp_price else "None"
            sl_str = f"{sl_price:{price_fmt}}" if sl_price else "None"
            atr_str = f"{atr_val:.5f}" if atr_val is not None else "N/A"
            self.logger.info(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_price:{price_fmt}}, "
                             f"RiskATR={atr_str}, TP={tp_str} (x{tp_mult}), SL={sl_str} (x{sl_mult})")

            return entry_price, tp_price, sl_price

        except Exception as e:
            self.logger.error(f"Unexpected error calculating TP/SL for {self.symbol}: {e}", exc_info=True)
            return entry_price, None, None

# --- Helper Functions (Leveraging potentially enhanced versions if available) ---
# Assume these functions exist elsewhere or are defined below, potentially adapted from volbot5
# Make sure they use the Analyzer instance for rounding/precision where appropriate.

# Placeholder implementations if not provided - REPLACE WITH ACTUAL IMPLEMENTATIONS
def fetch_balance(exchange: ccxt.Exchange, currency_code: str, logger: logging.Logger) -> Optional[Decimal]:
    lg = logger; lg.debug(f"Fetching balance for {currency_code}...")
    try:
        balance = exchange.fetch_balance()
        free_bal = safe_decimal(balance.get('free', {}).get(currency_code))
        total_bal = safe_decimal(balance.get('total', {}).get(currency_code))
        # Prefer total balance for risk calculation, fallback to free
        bal_to_return = total_bal if total_bal is not None else free_bal
        if bal_to_return is not None:
             lg.info(f"Balance for {currency_code}: Total={total_bal}, Free={free_bal}. Using: {bal_to_return}")
             return bal_to_return
        else:
             lg.warning(f"Could not find balance for {currency_code} in fetch_balance response.")
             return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: lg.error(f"Error fetching balance: {e}"); return None
    except Exception as e: lg.error(f"Unexpected error fetching balance: {e}", exc_info=True); return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    lg = logger; lg.debug(f"Getting market info for {symbol}...")
    try:
        market = exchange.market(symbol)
        if market:
             # Add contract type info if available
             market['is_contract'] = market.get('contract', False)
             market['contract_type'] = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
             lg.debug(f"Market info retrieved for {symbol}: Type={market['contract_type']}")
             return market
        else:
             lg.error(f"Market info not found for symbol: {symbol}")
             return None
    except ccxt.BadSymbol: lg.error(f"Invalid symbol format or not found on exchange: {symbol}"); return None
    except Exception as e: lg.error(f"Error getting market info for {symbol}: {e}", exc_info=True); return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    lg = logger; lg.debug(f"Checking for open position in {symbol}...")
    try:
        params = {}
        market = exchange.market(symbol)
        if 'bybit' in exchange.id.lower() and market and market.get('linear'):
             params['category'] = 'linear'
        positions = exchange.fetch_positions([symbol], params=params)
        # Filter for positions with non-zero size
        open_positions = [p for p in positions if p.get('contracts') is not None and safe_decimal(p.get('contracts')) != Decimal(0)]

        if not open_positions:
            lg.debug(f"No open position found for {symbol}.")
            return None
        elif len(open_positions) > 1:
            # This shouldn't happen in hedge mode (default for Bybit linear) but log if it does
            lg.warning(f"Found multiple ({len(open_positions)}) open positions for {symbol}. Using the first one found.")
            # Additional logic might be needed here depending on exchange/mode

        pos = open_positions[0]
        # --- Convert relevant fields to Decimal ---
        pos_data = {}
        fields_to_decimal = ['contracts', 'contractSize', 'entryPrice', 'initialMargin', 'leverage',
                             'liquidationPrice', 'maintenanceMargin', 'markPrice', 'notional',
                             'unrealizedPnl', 'realizedPnl', 'takeProfitPrice', 'stopLossPrice']
        for key, value in pos.items():
            if key in fields_to_decimal:
                pos_data[key] = safe_decimal(value)
            else:
                pos_data[key] = value # Keep other fields as they are (string, bool, etc.)

        # Determine side more reliably
        pos_size = pos_data.get('contracts', Decimal(0))
        pos_data['side'] = 'long' if pos_size > 0 else 'short' if pos_size < 0 else None

        # Add TSL info if available (Bybit specific example)
        if 'bybit' in exchange.id.lower() and 'info' in pos:
            bybit_info = pos.get('info', {})
            pos_data['trailingStopLossPrice'] = safe_decimal(bybit_info.get('trailingStop'))
            pos_data['trailingStopLossActive'] = bybit_info.get('activePrice') is not None and safe_decimal(bybit_info.get('activePrice')) != Decimal(0) # Approximation

        lg.info(f"Open position found for {symbol}: Side={pos_data['side']}, Size={pos_data.get('contracts')}, Entry={pos_data.get('entryPrice')}")
        return pos_data

    except ccxt.ExchangeNotAvailable as e: lg.error(f"Exchange not available checking position: {e}"); return None
    except ccxt.NetworkError as e: lg.error(f"Network error checking position: {e}"); return None
    except ccxt.ExchangeError as e: lg.error(f"Exchange error checking position: {e}"); return None
    except Exception as e: lg.error(f"Unexpected error checking position for {symbol}: {e}", exc_info=True); return None

def calculate_position_size(
    balance: Decimal, risk_per_trade: float, sl_price: Decimal, entry_price: Decimal,
    market_info: Dict, analyzer: TradingAnalyzer, logger: logging.Logger
) -> Optional[Decimal]:
    """Calculates position size based on risk percentage, SL distance, and balance."""
    lg = logger
    if not all([isinstance(balance, Decimal), isinstance(sl_price, Decimal), isinstance(entry_price, Decimal)]):
        lg.error("Invalid input types for position size calculation (need Decimals).")
        return None
    if balance <= 0 or risk_per_trade <= 0 or risk_per_trade >= 1 or sl_price <= 0 or entry_price <= 0:
        lg.error(f"Invalid inputs for position size: Balance={balance}, Risk={risk_per_trade}, SL={sl_price}, Entry={entry_price}")
        return None
    if sl_price == entry_price:
        lg.error("Stop loss price cannot be the same as entry price.")
        return None

    risk_amount = balance * Decimal(str(risk_per_trade)) # Amount of quote currency to risk
    risk_per_contract = abs(entry_price - sl_price) # Risk per contract in quote currency

    if risk_per_contract <= 0:
        lg.error(f"Risk per contract is zero or negative ({risk_per_contract}). Cannot calculate position size.")
        return None

    # Calculate raw position size (number of contracts)
    position_size_raw = risk_amount / risk_per_contract

    # Adjust for contract size if it's not 1 (common in inverse contracts)
    contract_size = safe_decimal(market_info.get('contractSize', '1'), Decimal('1'))
    if contract_size != Decimal('1'):
        lg.debug(f"Adjusting position size for contract size: {contract_size}")
        position_size_raw /= contract_size # This gives size in base currency if contractSize is in quote

    # Round the position size DOWN to the nearest valid step size using the analyzer's rounding
    rounded_size = analyzer.round_amount(position_size_raw)

    if rounded_size is None or rounded_size <= 0:
        lg.error(f"Calculated position size is zero or invalid after rounding. Raw size={position_size_raw}, Rounded={rounded_size}")
        return None

    # --- Optional: Add checks against min/max order size from market_info ---
    min_order_size = safe_decimal(market_info.get('limits', {}).get('amount', {}).get('min'))
    max_order_size = safe_decimal(market_info.get('limits', {}).get('amount', {}).get('max'))

    if min_order_size is not None and rounded_size < min_order_size:
        lg.warning(f"Calculated position size {rounded_size} is below minimum order size {min_order_size}. Cannot place order.")
        return None
    if max_order_size is not None and rounded_size > max_order_size:
        lg.warning(f"Calculated position size {rounded_size} exceeds maximum order size {max_order_size}. Capping size.")
        # Cap the size and re-round (floor)
        rounded_size = analyzer.round_amount(max_order_size)
        if rounded_size is None or rounded_size <= 0: return None # Check again after capping

    lg.info(f"Calculated Position Size: {rounded_size} {market_info.get('base', '')} "
            f"(Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount:.2f}, "
            f"Entry={entry_price}, SL={sl_price}, Risk/Contract={risk_per_contract})")
    return rounded_size

def place_market_order(
    exchange: ccxt.Exchange, symbol: str, side: str, amount: Decimal,
    market_info: Dict, analyzer: TradingAnalyzer, logger: logging.Logger, params: Optional[Dict] = None
) -> Optional[Dict]:
    """Places a market order using CCXT with proper rounding and error handling."""
    lg = logger
    if side not in ['buy', 'sell']: lg.error(f"Invalid order side: {side}"); return None
    if not isinstance(amount, Decimal) or amount <= 0: lg.error(f"Invalid order amount: {amount}"); return None

    # Round amount using analyzer
    rounded_amount = analyzer.round_amount(amount)
    if rounded_amount is None or rounded_amount <= 0:
        lg.error(f"Order amount {amount} rounded to zero or invalid ({rounded_amount}). Cannot place order.")
        return None

    lg.info(f"Placing MARKET {side.upper()} order for {rounded_amount} {market_info.get('base','')} of {symbol}...")

    order_params = {}
    # Add category hint for Bybit linear contracts
    if 'bybit' in exchange.id.lower() and market_info.get('contract_type') == 'linear':
        order_params['category'] = 'linear'
    # Merge any additional specific params
    if params: order_params.update(params)

    try:
        order = exchange.create_market_order(symbol, side, float(rounded_amount), params=order_params) # CCXT usually expects float amount
        lg.info(f"{COLOR_SUCCESS}Market {side.upper()} order placed successfully for {symbol}. Order ID: {order.get('id')}{RESET}")
        lg.debug(f"Order details: {order}")
        # Optional: Add delay or fetch order status immediately to confirm fill
        return order
    except ccxt.InsufficientFunds as e: lg.error(f"{COLOR_ERROR}Insufficient funds to place {side} order for {rounded_amount} {symbol}: {e}{RESET}"); return None
    except ccxt.ExchangeError as e: lg.error(f"{COLOR_ERROR}Exchange error placing {side} order for {symbol}: {e}{RESET}"); return None
    except ccxt.NetworkError as e: lg.error(f"{COLOR_ERROR}Network error placing {side} order for {symbol}: {e}{RESET}"); return None
    except Exception as e: lg.error(f"{COLOR_ERROR}Unexpected error placing {side} order for {symbol}: {e}{RESET}", exc_info=True); return None

def set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, analyzer: TradingAnalyzer,
    sl_price: Optional[Decimal], tp_price: Optional[Decimal], tsl_params: Optional[Dict] = None,
    current_position: Optional[Dict] = None, logger: logging.Logger = None
) -> bool:
    """
    Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL) for an open position.
    Uses exchange-specific methods if available (e.g., Bybit's unified endpoint).
    Compares with current protection to avoid redundant API calls.
    Handles rounding of prices.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        market_info: Market information dictionary.
        analyzer: TradingAnalyzer instance for rounding.
        sl_price: Desired stop loss price (Decimal), or None to potentially cancel. Use 0 to explicitly cancel on some exchanges.
        tp_price: Desired take profit price (Decimal), or None to potentially cancel. Use 0 to explicitly cancel.
        tsl_params: Dictionary with trailing stop parameters (e.g., {'trailingStop': '50', 'activePrice': '...'}) or None.
                    Use {'trailingStop': '0'} or similar to cancel TSL on Bybit.
        current_position: Optional dictionary of the current position details to check existing protection.
        logger: Logger instance.

    Returns:
        True if the API call was attempted successfully, False otherwise.
        Note: Success only means the call was sent, not that protection is guaranteed active.
    """
    lg = logger or logging.getLogger(__name__)
    lg.info(f"Attempting to set protection for {symbol}...")

    # --- Prepare Parameters ---
    params = {}
    needs_update = False
    price_fmt = f".{analyzer.get_price_precision()}f"

    # Add category for Bybit linear
    if 'bybit' in exchange.id.lower() and market_info.get('contract_type') == 'linear':
        params['category'] = 'linear'

    # Get current protection if position data is available
    current_sl = current_position.get('stopLossPrice') if current_position else None
    current_tp = current_position.get('takeProfitPrice') if current_position else None
    # Note: Current TSL details might be harder to get consistently across exchanges via fetch_positions
    # We might rely on the logic triggering the update rather than comparing existing TSL state perfectly.
    current_tsl_active = current_position.get('trailingStopLossActive', False) if current_position else False # Basic check

    # --- Stop Loss ---
    rounded_sl = None
    if sl_price is not None and sl_price > 0:
        rounded_sl = analyzer.round_price(sl_price) # Use appropriate rounding mode if needed (default should be ok)
        if rounded_sl is None:
            lg.error("Failed to round SL price. Cannot set SL.")
            return False # Cannot proceed without valid SL price if one was intended
        params['stopLoss'] = f"{rounded_sl:{price_fmt}}" # Format as string for API
        if rounded_sl != current_sl: needs_update = True; lg.debug(f"SL change detected: {current_sl} -> {rounded_sl}")
    elif sl_price == Decimal(0): # Explicit cancellation request
        params['stopLoss'] = '0' # Bybit uses '0' to cancel
        if current_sl is not None and current_sl != 0: needs_update = True; lg.debug("Explicit SL cancellation requested.")
    # If sl_price is None, we don't add 'stopLoss' to params, potentially leaving existing SL untouched.

    # --- Take Profit ---
    rounded_tp = None
    if tp_price is not None and tp_price > 0:
        rounded_tp = analyzer.round_price(tp_price)
        if rounded_tp is None:
            lg.error("Failed to round TP price. Cannot set TP.")
            # Might still proceed to set SL/TSL
        else:
            params['takeProfit'] = f"{rounded_tp:{price_fmt}}"
            if rounded_tp != current_tp: needs_update = True; lg.debug(f"TP change detected: {current_tp} -> {rounded_tp}")
    elif tp_price == Decimal(0): # Explicit cancellation
        params['takeProfit'] = '0'
        if current_tp is not None and current_tp != 0: needs_update = True; lg.debug("Explicit TP cancellation requested.")
    # If tp_price is None, leave existing TP.

    # --- Trailing Stop Loss ---
    if tsl_params is not None:
        # Example for Bybit: expects 'trailingStop' (distance/rate) and optionally 'activePrice'
        tsl_value = tsl_params.get('trailingStop')
        active_price = tsl_params.get('activePrice')

        if tsl_value is not None:
            params['trailingStop'] = str(tsl_value) # Ensure string format
            needs_update = True # Assume TSL update always requires API call
            lg.debug(f"TSL parameter 'trailingStop' set to: {tsl_value}")
            if active_price is not None:
                 # Round active price if it's a Decimal
                 active_price_dec = safe_decimal(active_price)
                 if active_price_dec:
                     rounded_active_price = analyzer.round_price(active_price_dec)
                     if rounded_active_price:
                         params['activePrice'] = f"{rounded_active_price:{price_fmt}}"
                         lg.debug(f"TSL parameter 'activePrice' set to: {params['activePrice']}")
                     else: lg.warning("Failed to round TSL active price.")
                 else: # If active_price was already a formatted string
                      params['activePrice'] = str(active_price)
                      lg.debug(f"TSL parameter 'activePrice' set to: {params['activePrice']}")
        else:
            lg.warning("TSL parameters provided but 'trailingStop' key is missing.")


    # --- Check if any update is needed ---
    if not needs_update:
        lg.info(f"No changes detected in protection levels for {symbol}. No API call needed.")
        return True # Considered successful as no action was required

    # --- Make the API Call ---
    # Use exchange-specific method if available, otherwise fallback
    lg.info(f"Sending protection update for {symbol}. Params: {params}")
    try:
        if 'bybit' in exchange.id.lower() and hasattr(exchange, 'private_post_position_set_trading_stop'):
            # Bybit specific endpoint for SL/TP/TSL
            response = exchange.private_post_position_set_trading_stop({**params, 'symbol': exchange.market_id(symbol)})
            lg.info(f"{COLOR_SUCCESS}Bybit protection update API call successful for {symbol}.{RESET}")
            lg.debug(f"Protection response: {response}")
            return True
        elif hasattr(exchange, 'edit_order'):
             # Fallback: Try modifying existing SL/TP orders (requires order IDs, more complex)
             lg.warning("Exchange does not have a dedicated protection endpoint like Bybit's. "
                        "Attempting to modify orders individually is not implemented in this function.")
             # TODO: Implement logic to find and edit existing SL/TP orders if needed
             return False
        else:
             lg.error("Cannot set position protection: Neither Bybit endpoint nor 'edit_order' available/implemented.")
             return False

    except ccxt.ExchangeError as e:
        lg.error(f"{COLOR_ERROR}Exchange error setting protection for {symbol}: {e}{RESET}")
        # Log specific Bybit errors if possible
        if 'bybit' in exchange.id.lower() and hasattr(e, 'args') and e.args:
             try: error_info = json.loads(str(e.args[0]).split(' ', 1)[1]); lg.error(f"  Bybit Error Code: {error_info.get('retCode')}, Msg: {error_info.get('retMsg')}")
             except: pass # Ignore parsing errors
        return False
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error setting protection for {symbol}: {e}{RESET}")
        return False
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
        return False

# Helper to color signal logs
def signal_to_color(signal: str) -> str:
    """Applies color formatting to signal strings for logging."""
    if signal == "BUY": return f"{NEON_GREEN}{signal}{RESET}"
    if signal == "SELL": return f"{NEON_RED}{signal}{RESET}"
    if signal == "HOLD": return f"{NEON_YELLOW}{signal}{RESET}"
    return signal # Return original string if not recognized

# --- Main Trading Loop Function (Merged & Enhanced) ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Performs one cycle of analysis and potential trading action for a single symbol.
    Fetches data, runs analysis, checks position, manages risk, and places orders if configured.
    """
    lg = logger
    lg.info(f"{COLOR_HEADER}---== Analyzing {symbol} ==---{RESET}")
    cycle_start_time = time.monotonic() # Track cycle duration

    try:
        # 1. Get Market Info (Precision, Limits, Type)
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info:
            lg.error(f"Skipping cycle for {symbol}: Could not retrieve valid market info.")
            return
        symbol = market_info['symbol'] # Use the validated symbol from market info

        # 2. Determine Timeframe and Fetch Kline Data
        interval_str = config.get("interval", "5") # e.g., "5"
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(interval_str) # e.g., "5m"
        if not ccxt_timeframe:
            lg.error(f"Invalid interval '{interval_str}' configured for {symbol}. Skipping cycle.")
            return

        # Calculate required kline limit dynamically (rough estimate)
        kline_limit = 300 # Default fallback
        try:
             buffer = 100 # Extra candles for stability
             # Find max period length from enabled indicators across both strategies + risk
             periods = [config.get("atr_period_risk", DEFAULT_ATR_PERIOD_RISK)]
             if config.get("livexy_enabled"):
                 periods.extend([
                     cfg_val for key, cfg_val in config.items()
                     if isinstance(cfg_val, (int, float)) and any(sub in key for sub in ['period', 'window', 'length']) and 'livexy' in key
                 ])
                 periods.append(config.get("fibonacci_window", DEFAULT_FIB_WINDOW))
             if config.get("volbot_enabled"):
                 periods.extend([
                     config.get("volbot_length", DEFAULT_VOLBOT_LENGTH),
                     config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
                     config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
                     config.get("volbot_pivot_left_len_h", 0) + config.get("volbot_pivot_right_len_h", 0),
                     config.get("volbot_pivot_left_len_l", 0) + config.get("volbot_pivot_right_len_l", 0),
                 ])
             # Use max period + buffer, ensure reasonable minimum
             required_data = max(periods) if periods else 0
             kline_limit = max(200, required_data + buffer) # Ensure at least 200 candles
             lg.debug(f"Calculated kline fetch limit based on indicator periods: {kline_limit} (MaxPeriod={required_data}, Buffer={buffer})")
        except Exception as e:
             lg.error(f"Error calculating optimal kline limit: {e}. Using fallback: {kline_limit}")

        df_klines = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, limit=kline_limit, logger=lg)
        min_acceptable_klines = 50 # Absolute minimum required to attempt analysis
        if df_klines.empty or len(df_klines) < min_acceptable_klines:
            lg.warning(f"Insufficient valid kline data for {symbol} (Got {len(df_klines)}, needed >{min_acceptable_klines}). Skipping analysis.")
            return

        # 3. Fetch Order Book (if LiveXY uses it and weight > 0)
        orderbook_data = None
        active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
        if config.get("livexy_enabled") and \
           config.get("indicators",{}).get("orderbook", False) and \
           safe_decimal(active_weights.get("orderbook", 0)) != 0:
            ob_limit = config.get("orderbook_limit", 25)
            lg.debug(f"Fetching order book data (limit={ob_limit}) as required by LiveXY config.")
            orderbook_data = fetch_orderbook_ccxt(exchange, symbol, ob_limit, lg)
            if orderbook_data is None:
                 lg.warning("Failed to fetch order book data for LiveXY analysis.")

        # 4. Initialize Analyzer & Calculate Indicators
        analyzer = TradingAnalyzer(
            df_raw=df_klines,
            logger=lg,
            config=config,
            market_info=market_info,
            orderbook_data=orderbook_data
        )
        # Check if analysis was successful
        if analyzer.df_processed.empty or not analyzer.strategy_state:
            lg.error(f"Indicator calculation or state update failed for {symbol}. Skipping trade logic.")
            return

        # 5. Check Current Position & Price
        current_position = get_open_position(exchange, symbol, lg)
        has_open_position = current_position is not None
        # Extract key position details safely
        position_side = current_position.get('side') if has_open_position else None # 'long' or 'short'
        position_entry_price = current_position.get('entryPrice') if has_open_position else None # Decimal
        position_size = current_position.get('contracts', Decimal('0')) if has_open_position else Decimal('0') # Decimal

        # Fetch current price (more reliable than last close for decisions)
        current_price = fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price is None:
            # Fallback to last close from analyzer state if fetch fails
            last_close = analyzer.strategy_state.get('close')
            if isinstance(last_close, Decimal) and last_close > 0:
                current_price = last_close
                lg.warning(f"Failed to fetch current ticker price. Using last close price ({current_price}) from klines.")
            else:
                lg.error(f"Failed to get current price or valid last close for {symbol}. Cannot proceed with trade logic.")
                return
        price_fmt = f".{analyzer.get_price_precision()}f" # Price format string

        # 6. Generate Trading Signal
        signal = analyzer.generate_trading_signal() # Returns "BUY", "SELL", or "HOLD"

        # 7. Trading Logic
        trading_enabled = config.get("enable_trading", False)

        # --- Scenario 1: Currently IN a Position ---
        if has_open_position and isinstance(position_size, Decimal) and abs(position_size) > Decimal('1e-9') and position_side:
            lg.info(f"Currently IN {position_side.upper()} position: Size={position_size}, Entry={position_entry_price:{price_fmt} if position_entry_price else 'N/A'}")

            # --- Check for Exit Signal ---
            # Exit if signal opposes current position direction
            exit_signal_triggered = (position_side == 'long' and signal == "SELL") or \
                                    (position_side == 'short' and signal == "BUY")

            if exit_signal_triggered:
                reason = f"Opposing signal ({signal}) generated"
                lg.info(f"{signal_to_color(signal)} EXIT Signal Triggered: {reason}. Closing {position_side.upper()} position.")
                if trading_enabled:
                    close_side = 'sell' if position_side == 'long' else 'buy'
                    # Place market order to close the entire position
                    # Use abs(position_size) as amount is always positive
                    close_order = place_market_order(
                        exchange, symbol, close_side, abs(position_size),
                        market_info, analyzer, lg,
                        params={'reduceOnly': True} # Ensure it only closes position
                    )
                    if close_order:
                        lg.info(f"{COLOR_SUCCESS}Market order placed to close position (Order ID: {close_order.get('id')}).")
                        # Attempt to cancel existing SL/TP/TSL after sending close order
                        time.sleep(1) # Brief pause before cancelling stops
                        lg.info("Attempting to cancel existing protection orders (SL/TP/TSL)...")
                        try:
                            # Use set_position_protection with zero/None values to cancel
                            cancel_success = set_position_protection(
                                exchange, symbol, market_info, analyzer,
                                sl_price=Decimal(0), # Explicit SL cancel
                                tp_price=Decimal(0), # Explicit TP cancel
                                tsl_params={'trailingStop': '0'}, # Explicit TSL cancel (Bybit specific)
                                logger=lg
                            )
                            if cancel_success:
                                lg.info("Cancellation request for existing stops sent.")
                            else:
                                lg.warning("Could not confirm cancellation of stops via API. Manual check advised.")
                        except Exception as cancel_err:
                            lg.warning(f"Error attempting to cancel stops after closing: {cancel_err}")
                    else:
                        lg.error(f"{COLOR_ERROR}Failed to place market order to close position. MANUAL INTERVENTION REQUIRED!{RESET}")
                else:
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have placed market order to close {position_side} position.{RESET}")
                # End cycle for this symbol after attempting closure
                return
            else:
                # --- No Exit Signal: Perform In-Position Risk Management ---
                lg.info(f"Signal is '{signal}'. Holding {position_side} position. Performing risk management...")
                new_sl, new_tsl_params, needs_protection_update = None, None, False

                # Get current protection details from position data if available
                current_sl = current_position.get('stopLossPrice') # Decimal or None
                current_tp = current_position.get('takeProfitPrice') # Decimal or None
                current_tsl_active = current_position.get('trailingStopLossActive', False) # Boolean guess
                current_tsl_price = current_position.get('trailingStopLossPrice') # Decimal or None

                # --- Check Break-Even (BE) ---
                enable_be = config.get("enable_break_even", True)
                # Option: Disable BE if TSL is already active (can be config choice)
                disable_be_if_tsl_active = True
                run_be_check = enable_be and not (disable_be_if_tsl_active and current_tsl_active)

                if run_be_check and position_entry_price is not None:
                    risk_atr = analyzer.strategy_state.get("atr_risk") # Decimal or None
                    min_tick = analyzer.get_min_tick_size() # Decimal

                    if isinstance(risk_atr, Decimal) and risk_atr > 0 and min_tick > 0:
                        be_trigger_mult = safe_decimal(config.get("break_even_trigger_atr_multiple", 1.0), Decimal(1.0))
                        be_offset_ticks = int(config.get("break_even_offset_ticks", 2))

                        # Calculate profit needed to trigger BE
                        profit_target_for_be = risk_atr * be_trigger_mult
                        # Calculate current profit/loss
                        current_pnl = (current_price - position_entry_price) if position_side == 'long' else (position_entry_price - current_price)

                        lg.debug(f"BE Check: Current PnL={current_pnl:{price_fmt}}, Target Profit={profit_target_for_be:{price_fmt}} (ATR={risk_atr:.5f} * {be_trigger_mult})")

                        if current_pnl >= profit_target_for_be:
                            # Calculate BE SL price (entry + small offset)
                            offset_amount = min_tick * be_offset_ticks
                            be_sl_raw = position_entry_price + offset_amount if position_side == 'long' else position_entry_price - offset_amount
                            # Round BE SL conservatively (further from entry)
                            be_round_mode = ROUND_UP if position_side == 'long' else ROUND_DOWN
                            be_sl_price = analyzer.round_price(be_sl_raw, rounding_mode=be_round_mode)

                            if be_sl_price and be_sl_price > 0:
                                # Check if this new BE SL is actually better (higher for long, lower for short) than the current SL
                                is_sl_improvement = current_sl is None or \
                                                    (position_side == 'long' and be_sl_price > current_sl) or \
                                                    (position_side == 'short' and be_sl_price < current_sl)

                                if is_sl_improvement:
                                    lg.info(f"{COLOR_SUCCESS}Break-Even Triggered! Profit target reached. Proposed BE SL: {be_sl_price:{price_fmt}}{RESET}")
                                    new_sl = be_sl_price # Set the new SL target
                                    needs_protection_update = True
                                else:
                                    lg.debug(f"BE triggered, but proposed BE SL {be_sl_price:{price_fmt}} is not an improvement over current SL {current_sl:{price_fmt}}. No SL change.")
                            else:
                                lg.warning("BE triggered, but failed to calculate valid BE SL price.")
                    elif enable_be: # Log if BE enabled but couldn't run check
                        lg.warning(f"Cannot calculate BE: Invalid Risk ATR ({risk_atr}), Min Tick ({min_tick}), or Entry Price ({position_entry_price}).")

                # --- Check Trailing Stop Loss (TSL) Activation ---
                enable_tsl = config.get("enable_trailing_stop", True)
                # Only try to activate if TSL is enabled and not already active
                if enable_tsl and not current_tsl_active:
                    tsl_rate_str = str(config.get("trailing_stop_callback_rate", "0.5%")) # e.g., "0.5%" or "50"
                    tsl_act_perc_cfg = config.get("trailing_stop_activation_percentage", 0.003) # e.g., 0.003 for 0.3%
                    tsl_act_perc = safe_decimal(tsl_act_perc_cfg)

                    # Validate TSL rate format (simple check for number/percentage)
                    is_valid_rate = False
                    try:
                        # Check if it's a number or ends with %
                        rate_val_str = tsl_rate_str.replace('%', '')
                        rate_val = safe_decimal(rate_val_str)
                        if rate_val is not None and rate_val > 0: is_valid_rate = True
                    except: pass

                    if is_valid_rate and tsl_act_perc is not None and tsl_act_perc >= 0 and position_entry_price is not None and position_entry_price > 0:
                        activate_tsl = False
                        # Activate immediately if threshold is 0 or negative
                        if tsl_act_perc <= 0:
                            activate_tsl = True
                            lg.info("TSL activation threshold <= 0, attempting immediate TSL activation.")
                        else:
                            # Calculate current profit percentage
                            profit_perc = (current_price / position_entry_price) - 1 if position_side == 'long' else 1 - (current_price / position_entry_price)
                            lg.debug(f"TSL Activation Check: Current Profit%={profit_perc:.4%}, Activation Threshold%={tsl_act_perc:.4%}")
                            if profit_perc >= tsl_act_perc:
                                activate_tsl = True
                                lg.info(f"{COLOR_SUCCESS}TSL activation profit threshold reached.{RESET}")

                        if activate_tsl:
                            # Prepare TSL parameters for the API call
                            new_tsl_params = {'trailingStop': tsl_rate_str}
                            # Calculate activation price if threshold > 0 (Bybit specific)
                            if tsl_act_perc > 0:
                                act_price_raw = position_entry_price * (1 + tsl_act_perc) if position_side == 'long' else position_entry_price * (1 - tsl_act_perc)
                                # Round activation price away from entry (conservative activation)
                                act_round_mode = ROUND_UP if position_side == 'long' else ROUND_DOWN
                                act_price = analyzer.round_price(act_price_raw, rounding_mode=act_round_mode)
                                if act_price and act_price > 0:
                                     new_tsl_params['activePrice'] = f"{act_price:{price_fmt}}"
                                     lg.info(f"Calculated TSL Activation Price: {new_tsl_params['activePrice']}")
                                else: lg.warning("Could not calculate valid TSL activation price.")

                            lg.info(f"Attempting to activate TSL with rate: {tsl_rate_str}")
                            needs_protection_update = True
                    elif enable_tsl: # Log if TSL enabled but couldn't check activation
                         lg.warning(f"Cannot check TSL activation: Invalid rate ('{tsl_rate_str}'), "
                                    f"activation percentage ({tsl_act_perc_cfg}), or entry price ({position_entry_price}).")


                # --- Update Protection via API Call ---
                if needs_protection_update and trading_enabled:
                    lg.info("Change detected, attempting to update position protection (SL/TP/TSL)...")
                    # Prioritize the new SL from BE if it was set
                    final_sl = new_sl if new_sl is not None else current_sl
                    # Keep existing TP unless explicitly changed (not implemented here, but possible)
                    final_tp = current_tp
                    # Use new TSL params if activation was triggered
                    final_tsl = new_tsl_params if new_tsl_params is not None else None # Don't send TSL params if not activating

                    # Call the protection function
                    success = set_position_protection(
                        exchange, symbol, market_info, analyzer,
                        sl_price=final_sl, # Use the potentially updated SL
                        tp_price=final_tp, # Keep current TP
                        tsl_params=final_tsl, # Send new TSL params if activating
                        current_position=current_position, # Pass current position for comparison
                        logger=lg
                    )
                    if success:
                        lg.info(f"{COLOR_SUCCESS}Protection update request sent successfully.{RESET}")
                    else:
                        lg.error(f"{COLOR_ERROR}Failed to send protection update request.{RESET}")
                elif needs_protection_update and not trading_enabled:
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have updated protection (NewSL={new_sl}, NewTSL={new_tsl_params}).{RESET}")
                else:
                    lg.info("No risk management actions triggered requiring protection update.")

        # --- Scenario 2: Currently OUT of Position ---
        else:
            lg.info(f"No open position found for {symbol}. Signal: {signal_to_color(signal)}")

            # --- Check for Entry Signal ---
            if signal in ["BUY", "SELL"]:
                lg.info(f"{signal_to_color(signal)} ENTRY Signal Detected at current price: {current_price:{price_fmt}}")

                # 1. Calculate initial TP/SL based on entry price and ATR
                # Use current_price as potential entry price for calculation
                _, potential_tp, potential_sl = analyzer.calculate_entry_tp_sl(current_price, signal)

                # CRITICAL: Must have a valid SL to calculate position size and enter
                if potential_sl is None:
                    lg.error(f"{COLOR_ERROR}Cannot enter {signal}: Failed to calculate a valid initial Stop Loss.{RESET}")
                    return # Do not proceed without SL

                # 2. Calculate Position Size based on risk
                quote_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if quote_balance is None:
                    lg.error(f"Cannot calculate position size: Failed to fetch {QUOTE_CURRENCY} balance.")
                    return
                if quote_balance <= 0:
                    lg.error(f"Cannot calculate position size: {QUOTE_CURRENCY} balance is zero or negative ({quote_balance}).")
                    return

                risk_fraction = config.get("risk_per_trade", 0.01) # Float (e.g., 0.01 for 1%)
                position_size_contracts = calculate_position_size(
                    balance=quote_balance,
                    risk_per_trade=risk_fraction,
                    sl_price=potential_sl,
                    entry_price=current_price, # Use current price as estimated entry
                    market_info=market_info,
                    analyzer=analyzer,
                    logger=lg
                )

                if position_size_contracts is None or position_size_contracts <= 0:
                    lg.error(f"{COLOR_ERROR}Cannot enter {signal}: Position size calculation failed or resulted in zero/negative size ({position_size_contracts}).{RESET}")
                    return

                # 3. Place Order and Set Protection (if trading enabled)
                tp_str = f"{potential_tp:{price_fmt}}" if potential_tp else "None"
                sl_str = f"{potential_sl:{price_fmt}}" # Should not be None here
                lg.info(f"Potential Entry Details: Signal={signal}, Size={position_size_contracts}, Est. Entry={current_price:{price_fmt}}, SL={sl_str}, TP={tp_str}")

                if trading_enabled:
                    # --- Set Leverage ---
                    if market_info.get('is_contract'):
                        leverage = int(config.get("leverage", 10))
                        try:
                            lg.info(f"Attempting to set leverage to {leverage}x for {symbol}...")
                            # Bybit requires category for setting leverage on linear contracts
                            params_lev = {}
                            if 'bybit' in exchange.id.lower() and market_info.get('contract_type') == 'linear':
                                params_lev = {'category': 'linear', 'buyLeverage': str(float(leverage)), 'sellLeverage': str(float(leverage))}

                            # Use set_leverage method (might vary slightly by exchange in CCXT)
                            exchange.set_leverage(leverage, symbol, params=params_lev)
                            lg.info(f"Leverage set to {leverage}x successfully.")
                        except ccxt.ExchangeError as e:
                             # Common issue: Setting leverage when position exists, or margin mode mismatch
                             lg.warning(f"{COLOR_WARNING}Could not modify leverage to {leverage}x for {symbol}. Error: {e}. Check if position exists or margin mode allows.{RESET}")
                        except Exception as e:
                             lg.warning(f"{COLOR_WARNING}Unexpected error setting leverage to {leverage}x: {e}{RESET}")
                    else: lg.debug("Leverage setting skipped (not a contract market).")


                    # --- Place Entry Order ---
                    entry_order = place_market_order(
                        exchange, symbol, signal.lower(), position_size_contracts,
                        market_info, analyzer, lg
                    )

                    # --- Set Protection After Entry ---
                    if entry_order and entry_order.get('id'):
                        # Order placed, now try to set SL/TP/TSL
                        lg.info(f"{COLOR_SUCCESS}Entry order placed (ID: {entry_order.get('id')}). Waiting {POSITION_CONFIRM_DELAY}s before setting protection...{RESET}")
                        time.sleep(POSITION_CONFIRM_DELAY) # Wait for order fill propagation

                        # Check position again to confirm entry (optional but recommended)
                        # entry_pos_check = get_open_position(exchange, symbol, lg)
                        # if not entry_pos_check: lg.warning("Position not confirmed after entry order!") # Handle this case?

                        # Prepare TSL params for initial setting if TSL is enabled
                        tsl_entry_params = None
                        if config.get("enable_trailing_stop", True):
                            tsl_rate_str = str(config.get("trailing_stop_callback_rate","0.5%"))
                            tsl_act_perc_cfg = config.get("trailing_stop_activation_percentage",0.003)
                            tsl_act_perc = safe_decimal(tsl_act_perc_cfg)
                            # Validate rate
                            is_valid_rate = False; try: rate_val=safe_decimal(tsl_rate_str.replace('%','')); is_valid_rate=(rate_val is not None and rate_val>0) except: pass

                            if is_valid_rate and tsl_act_perc is not None and tsl_act_perc >= 0:
                                tsl_entry_params = {'trailingStop': tsl_rate_str}
                                if tsl_act_perc > 0:
                                    # Use filled price if available, else estimated entry
                                    filled_price = safe_decimal(entry_order.get('average', entry_order.get('price'))) or current_price
                                    act_price_raw = filled_price * (1 + tsl_act_perc) if signal == "BUY" else filled_price * (1 - tsl_act_perc)
                                    act_round = ROUND_UP if signal == "BUY" else ROUND_DOWN
                                    act_price = analyzer.round_price(act_price_raw, rounding_mode=act_round)
                                    if act_price and act_price > 0:
                                        tsl_entry_params['activePrice'] = f"{act_price:{price_fmt}}"
                                    else: lg.warning("Could not calculate valid initial TSL activation price.")
                                lg.info(f"Prepared initial TSL parameters: {tsl_entry_params}")
                            else: lg.warning(f"Cannot set initial TSL: Invalid rate ('{tsl_rate_str}') or activation percentage ({tsl_act_perc_cfg}).")

                        # Set SL, TP, and potentially TSL
                        protection_success = set_position_protection(
                            exchange, symbol, market_info, analyzer,
                            sl_price=potential_sl,
                            tp_price=potential_tp,
                            tsl_params=tsl_entry_params,
                            logger=lg
                        )

                        if protection_success:
                            lg.info(f"{COLOR_SUCCESS}Initial protection (SL={sl_str}, TP={tp_str}, TSL={'Active' if tsl_entry_params else 'Inactive'}) set for new {signal} position.{RESET}")
                        else:
                            lg.error(f"{COLOR_ERROR}Failed to set initial protection for {symbol} after entry. MANUAL CHECK REQUIRED!{RESET}")
                    elif entry_order:
                        # Order was returned but might not have ID or failed status
                        lg.error(f"Entry order status uncertain (Details: {entry_order}). Cannot set protection. MANUAL CHECK REQUIRED!")
                    else:
                        # place_market_order returned None
                        lg.error(f"Entry order FAILED for {signal} {symbol}. No protection set.")
                else:
                    # Trading disabled
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have entered {signal} | Size={position_size_contracts} | SL={sl_str} | TP={tp_str}{RESET}")

            elif signal == "HOLD":
                lg.info("Signal is HOLD. No position open. No entry action taken.")

    except ccxt.AuthenticationError as e:
        # Critical error, should stop the bot
        lg.critical(f"{COLOR_ERROR}CRITICAL AUTHENTICATION ERROR during cycle for {symbol}: {e}. Check API keys/permissions.{RESET}", exc_info=True)
        raise # Re-raise to stop the main loop
    except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
        # Allow retry for network issues
        lg.error(f"Network Error during cycle for {symbol}: {e}. Cycle will repeat.")
    except ccxt.ExchangeError as e:
        # Log exchange errors, allow retry unless critical (like auth error)
        lg.error(f"Exchange Error during cycle for {symbol}: {e}.", exc_info=True)
    except Exception as e:
        # Catch any other unexpected errors within the symbol loop
        lg.error(f"{COLOR_ERROR}!!! UNHANDLED EXCEPTION in cycle for {symbol} !!!: {e}{RESET}", exc_info=True)
        # Continue to the next symbol or cycle unless it's critical

    finally:
        cycle_duration = time.monotonic() - cycle_start_time
        lg.info(f"{COLOR_HEADER}---== Finished {symbol} cycle ({cycle_duration:.2f}s) ==---{RESET}")


# --- Main Execution ---
def main() -> None:
    """Initializes the bot, selects symbols, and runs the main trading loop."""
    # Use a dedicated logger for initialization messages
    init_logger = setup_logger("init")
    init_logger.info(f"{COLOR_HEADER}--- Merged Bot Initializing ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    init_logger.info(f"Using Config: {os.path.abspath(CONFIG_FILE)}")
    init_logger.info(f"Logging to Directory: {os.path.abspath(LOG_DIRECTORY)}")
    init_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"Trading Enabled: {CONFIG.get('enable_trading')}")
    init_logger.info(f"Using Sandbox: {CONFIG.get('use_sandbox')}")
    init_logger.info(f"Kline Interval: {CONFIG.get('interval')} ({CCXT_INTERVAL_MAP.get(CONFIG.get('interval'), 'N/A')})")
    init_logger.info(f"Timezone: {TIMEZONE.key}")
    init_logger.info(f"Console Log Level: {logging.getLevelName(_log_level_container['level'])}")
    init_logger.info(f"Signal Mode: {CONFIG.get('signal_mode')}")
    init_logger.info(f"Strategies Enabled: LiveXY={CONFIG.get('livexy_enabled')}, Volbot={CONFIG.get('volbot_enabled')}")

    # Initialize Exchange
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical("Exchange initialization failed. Bot cannot start. Exiting.")
        return

    # --- Symbol Selection ---
    symbols_to_trade: List[str] = []
    init_logger.info(f"{COLOR_HEADER}--- Symbol Selection ---{RESET}")
    # Interactive loop for symbol selection
    while True:
        list_str = f"Current Symbols: {', '.join(symbols_to_trade) if symbols_to_trade else 'None'}"
        print(f"\n{list_str}")
        prompt = (f"Enter symbol (e.g., BTC/USDT:USDT), '{COLOR_CYAN}all{RESET}' "
                  f"(active linear {QUOTE_CURRENCY} perpetuals), '{COLOR_YELLOW}clear{RESET}', "
                  f"or {COLOR_GREEN}Enter{RESET} to start ({len(symbols_to_trade)} symbols): ")
        try:
            symbol_input = input(prompt).strip()
        except EOFError: # Handle non-interactive environments (e.g., Docker)
             if not symbols_to_trade:
                 init_logger.warning("Non-interactive mode detected and no symbols pre-loaded. Exiting.")
                 return # Or load symbols from config/env var?
             symbol_input = "" # Treat as "Enter" to start if symbols exist

        if not symbol_input and symbols_to_trade:
            init_logger.info(f"Starting trading loop with {len(symbols_to_trade)} selected symbols.")
            break # Exit selection loop and start trading
        if not symbol_input and not symbols_to_trade:
            print(f"{COLOR_WARNING}Symbol list is empty. Please add symbols or type 'all'.{RESET}")
            continue
        if symbol_input.lower() == 'clear':
            symbols_to_trade = []
            print("Symbol list cleared.")
            continue
        if symbol_input.lower() == 'all':
            init_logger.info(f"Fetching all active linear {QUOTE_CURRENCY} perpetual swaps...")
            try:
                # Force reload of markets to get the latest active list
                all_mkts = exchange.load_markets(True)
                linear_swaps = [
                    m['symbol'] for m in all_mkts.values()
                    if m.get('active') and m.get('linear') and m.get('swap') and m.get('quote','').upper() == QUOTE_CURRENCY
                ]
                if not linear_swaps:
                     init_logger.warning(f"No active linear perpetual swaps found for quote currency {QUOTE_CURRENCY}.")
                     continue

                init_logger.info(f"Found {len(linear_swaps)} active linear {QUOTE_CURRENCY} swaps.")
                # Add only unique symbols to the list
                new_symbols = [s for s in linear_swaps if s not in symbols_to_trade]
                symbols_to_trade.extend(new_symbols)
                init_logger.info(f"Added {len(new_symbols)} new symbols to the list. Total: {len(symbols_to_trade)}.")
            except Exception as e:
                init_logger.error(f"Error fetching 'all' symbols: {e}", exc_info=True)
            continue

        # Validate single symbol input (case-insensitive match, use market info)
        init_logger.info(f"Validating symbol '{symbol_input}'...")
        # Use get_market_info for validation
        market_info = get_market_info(exchange, symbol_input.upper(), init_logger)
        if market_info:
            validated_symbol = market_info['symbol'] # Get the correctly formatted symbol
            if validated_symbol not in symbols_to_trade:
                symbols_to_trade.append(validated_symbol)
                print(f"{COLOR_SUCCESS}Added symbol: {validated_symbol}{RESET}")
            else:
                print(f"{COLOR_YELLOW}{validated_symbol} is already in the list.{RESET}")
        else:
            # get_market_info already logged the error
            print(f"{COLOR_ERROR}Symbol '{symbol_input}' seems invalid or not found on the exchange.{RESET}")

    # Check if any symbols were selected
    if not symbols_to_trade:
        init_logger.critical("No valid symbols selected to trade. Exiting.")
        return

    # --- Final Safety Check before Live Trading ---
    if CONFIG.get("enable_trading"):
        print("\n" + "="*40)
        init_logger.warning(f"{COLOR_RED}!!! LIVE TRADING IS ENABLED !!!{RESET}")
        if not CONFIG.get("use_sandbox"):
            init_logger.warning(f"{COLOR_RED}!!! BOT WILL USE REAL MONEY ON MAINNET !!!{RESET}")
        else:
            init_logger.warning(f"{COLOR_YELLOW}Bot is configured to use SANDBOX (Testnet).{RESET}")
        # Display key risk parameters
        init_logger.warning(f"Risk Per Trade: {CONFIG['risk_per_trade']:.2%}")
        init_logger.warning(f"Leverage: {CONFIG['leverage']}x")
        init_logger.warning(f"Trailing Stop: {'Enabled' if CONFIG['enable_trailing_stop'] else 'Disabled'}")
        init_logger.warning(f"Break Even: {'Enabled' if CONFIG['enable_break_even'] else 'Disabled'}")
        init_logger.warning(f"Trading Symbols: {', '.join(symbols_to_trade)}")
        print("="*40 + "\n")
        try:
            # Final confirmation prompt
            confirm = input(f"Press {COLOR_GREEN}Enter{RESET} to confirm and START TRADING, or type 'N' / Ctrl+C to abort: ")
            if confirm.strip().lower() == 'n':
                init_logger.info("User aborted startup.")
                return
            init_logger.info("User confirmed. Starting main loop.")
        except KeyboardInterrupt:
            init_logger.info("User aborted startup via Ctrl+C.")
            return
    else:
        init_logger.info(f"{COLOR_YELLOW}TRADING IS DISABLED. Running in analysis-only mode.{RESET}")
        init_logger.info(f"Trading Symbols: {', '.join(symbols_to_trade)}")

    # --- Setup Loggers for Each Symbol ---
    # Sanitize symbol names for logger/file names
    symbol_loggers = {
        symbol: setup_logger(symbol.replace('/', '_').replace(':', '-'))
        for symbol in symbols_to_trade
    }

    # --- Main Trading Loop ---
    init_logger.info(f"{COLOR_HEADER}--- Starting Main Trading Loop for {len(symbols_to_trade)} Symbols ---{RESET}")
    try:
        while True:
            loop_start_time = time.time() # Track start of the full loop iteration

            # --- Optional: Reload Config Each Cycle ---
            # Uncomment below to reload config file at the start of each loop iteration.
            # Be cautious as this can change behavior mid-operation.
            # try:
            #     CONFIG = load_config(CONFIG_FILE)
            #     # Update log levels for all symbol loggers if changed
            #     current_console_level = _log_level_container['level']
            #     for sym_logger in symbol_loggers.values():
            #         for handler in sym_logger.handlers:
            #             if isinstance(handler, logging.StreamHandler):
            #                 if handler.level != current_console_level:
            #                     sym_logger.debug(f"Updating console level to {logging.getLevelName(current_console_level)}")
            #                     handler.setLevel(current_console_level)
            #     init_logger.info("Configuration reloaded.")
            # except Exception as config_reload_err:
            #     init_logger.error(f"Error reloading config: {config_reload_err}. Using previous config.")
            # --- End Optional Config Reload ---

            # Iterate through each selected symbol
            for symbol in symbols_to_trade:
                symbol_logger = symbol_loggers[symbol]
                try:
                    # Analyze and potentially trade the current symbol
                    analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except ccxt.AuthenticationError:
                    # Critical auth error should stop the bot immediately
                    init_logger.critical(f"Authentication Error detected during cycle for {symbol}. Stopping bot.")
                    raise # Re-raise to be caught by the outer loop's handler
                except Exception as symbol_err:
                    # Log unhandled errors specific to a symbol's cycle, but try to continue
                    symbol_logger.error(f"Unhandled error during cycle for {symbol}: {symbol_err}", exc_info=True)
                    symbol_logger.info(f"Attempting to continue with the next symbol or cycle despite error in {symbol}.")
                # Optional small delay between symbols within a loop iteration
                # time.sleep(1)

            # --- Loop Delay Calculation ---
            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)

            init_logger.debug(f"Main loop cycle finished in {elapsed_time:.2f}s. Waiting {sleep_time:.2f}s before next cycle...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        init_logger.info("Keyboard Interrupt detected. Shutting down gracefully...")
    except ccxt.AuthenticationError:
        # Already logged when raised from analyze_and_trade_symbol
        init_logger.critical("BOT STOPPED due to authentication failure.")
    except Exception as e:
        # Catch any critical unhandled errors in the main loop itself
        init_logger.critical(f"Critical unhandled error in main loop: {e}", exc_info=True)
        init_logger.critical("BOT STOPPED.")
    finally:
        # --- Shutdown Procedures ---
        init_logger.info(f"{COLOR_HEADER}--- Merged Bot Stopping ---{RESET}")
        # Close the exchange connection if possible
        if exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Closing CCXT exchange connection...")
                exchange.close()
                init_logger.info("Exchange connection closed.")
            except Exception as ce:
                init_logger.error(f"Error closing CCXT connection: {ce}")
        # Ensure all log handlers are flushed and closed
        logging.shutdown()
        print(f"{NEON_PURPLE}Bot stopped.{RESET}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()

