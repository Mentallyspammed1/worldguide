~/worldguide/codes main* ⇡ ❯ python x1.0.py BCH/USDT:USDT                                                   2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Logger 'livebot_BCH_USDT-USDT' initialized. File: 'bot_logs/livebot_BCH_USDT-USDT.log', Console Level: INFO                                                  2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - ---=== Whale 2.0 Enhanced Trading Bot Initializing ===---                                              2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Symbol: BCH/USDT:USDT                            2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Config File: config.json                         2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Log Directory: bot_logs                          2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Timezone: America/Chicago                        2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Trading Enabled: True                            2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Sandbox Mode: False                              2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Quote Currency: USDT                             2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Risk Per Trade: 1.00%                            2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Leverage: 20x                                    2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Interval: 3                                      2025-04-24 12:37:37 - WARNING  - [livebot_BCH_USDT-USDT] - USING LIVE TRADING MODE (Real Money)             2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Connecting to bybit (Sandbox: False)...          2025-04-24 12:37:37 - INFO     - [livebot_BCH_USDT-USDT] - Loading markets for bybit...                     2025-04-24 12:37:50 - INFO     - [livebot_BCH_USDT-USDT] - Markets loaded successfully for bybit.           2025-04-24 12:37:50 - INFO     - [livebot_BCH_USDT-USDT] - CCXT exchange initialized (bybit). CCXT Version: 4.4.72                                                2025-04-24 12:37:50 - INFO     - [livebot_BCH_USDT-USDT] - Attempting initial balance fetch for USDT...     2025-04-24 12:37:53 - ERROR    - [livebot_BCH_USDT-USDT] - Exchange error fetching balance: bybit {"retCode":10001,"retMsg":"accountType only support UNIFIED.","result":{},"retExtInfo":{},"time":1745516274021}. Not retrying.                                              2025-04-24 12:37:53 - WARNING  - [livebot_BCH_USDT-USDT] - Initial balance fetch returned None or failed. Check logs. Ensure API keys have 'Read' permissions and correct account type (CONTRACT/UNIFIED) is accessible.2025-04-24 12:37:53 - ERROR    - [livebot_BCH_USDT-USDT] - Cannot verify balance. Trading is enabled, aborting initialization for safety.                         2025-04-24 12:37:53 - CRITICAL - [livebot_BCH_USDT-USDT] - Failed to initialize exchange. Bot cannot start.                                                       Bot execution

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, Union
import argparse # Added for command-line arguments

# Third-party libraries - alphabetized
import ccxt
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv
# requests, requests.adapters, urllib3.util.retry are typically used by ccxt internally,
# direct import might not be needed unless making custom HTTP calls. Keep for now if unsure.
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo # Preferred over pytz for modern Python

# Initialize colorama and set decimal precision
getcontext().prec = 28  # Set precision for Decimal calculations
init(autoreset=True)
load_dotenv() # Load environment variables from .env file

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
    raise ValueError(f"{NEON_RED}BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file{RESET}")

# --- Configuration File and Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Timezone for logging and display (adjust as needed in config or here)
# Defaulting here, but ideally loaded from config later if needed more dynamically
try:
    # Example: Allow overriding timezone via env var for Docker setups
    TZ_NAME = os.getenv("BOT_TIMEZONE", "America/Chicago")
    TIMEZONE = ZoneInfo(TZ_NAME)
except Exception:
    print(f"{NEON_YELLOW}Warning: Could not load timezone '{TZ_NAME}'. Defaulting to UTC.{RESET}")
    TIMEZONE = ZoneInfo("UTC")

# --- API Interaction Constants ---
MAX_API_RETRIES = 3 # Max retries for recoverable API errors (e.g., network, rate limit)
RETRY_DELAY_SECONDS = 5 # Base delay between retries
# HTTP status codes considered retryable (Network/Server issues)
RETRYABLE_HTTP_STATUS = [429, 500, 502, 503, 504]
# Intervals supported by the bot's internal logic (ensure config matches)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
# Map bot intervals to ccxt's expected timeframe format
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# --- Default Indicator/Strategy Parameters (can be overridden by config.json) ---
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14 # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 12     # Window for underlying RSI in StochRSI
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float for calculations
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels

# --- Bot Timing and Delays ---
LOOP_DELAY_SECONDS = 15 # Time between the end of one cycle and the start of the next (configurable)
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status

# QUOTE_CURRENCY will be dynamically loaded from config after it's read


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        original_message = super().format(record)
        redacted_message = original_message
        if API_KEY:
            redacted_message = redacted_message.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            redacted_message = redacted_message.replace(API_SECRET, "***API_SECRET***")
        return redacted_message

def setup_logger(symbol: str, console_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger for the given symbol with rotating file and console handlers.

    Args:
        symbol: The trading symbol (used for naming log files).
        console_level: The logging level for the console output (e.g., logging.INFO, logging.DEBUG).

    Returns:
        The configured logger instance.
    """
    # Clean symbol for filename (replace characters invalid in filenames)
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"livebot_{safe_symbol}" # Unique logger name per symbol
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists and is configured
    if logger.hasHandlers():
        # Ensure existing handlers have the correct level (if changed dynamically)
        for handler in logger.handlers:
             if isinstance(handler, logging.StreamHandler):
                  handler.setLevel(console_level) # Update console level if needed
        logger.info(f"Logger '{logger_name}' already configured.")
        return logger

    # Set base logging level to DEBUG to capture everything for the file
    logger.setLevel(logging.DEBUG)

    # --- File Handler ---
    # Rotates logs, keeping backups. Includes line numbers for detailed debugging.
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S' # Consistent timestamp format
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(file_handler)
    except Exception as e:
        # Use print here as logger setup might have failed
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")

    # --- Stream Handler (Console) ---
    # Colored output for readability, level controlled by `console_level`.
    stream_handler = logging.StreamHandler()
    # Use local timezone for console timestamps
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Timestamp format for console
    )
    # Apply the local timezone to the console formatter's asctime
    logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()

    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level) # Set level passed from main()
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs if root logger is configured)
    logger.propagate = False

    logger.info(f"Logger '{logger_name}' initialized. File: '{log_filename}', Console Level: {logging.getLevelName(console_level)}")
    return logger

# --- Configuration Management ---
def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    keys_added = False
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
            keys_added = True
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_updated_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_value)
            if nested_keys_added:
                updated_config[key] = nested_updated_config
                keys_added = True
        # Optional: Add type mismatch warning/handling if needed
        # elif type(default_value) != type(updated_config.get(key)):
        #     print(f"{NEON_YELLOW}Warning: Config type mismatch for key '{key}'. Default: {type(default_value)}, Loaded: {type(updated_config.get(key))}. Using loaded value.{RESET}")
        #     # Decide whether to overwrite with default or keep loaded value
    return updated_config, keys_added

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    Creates a default config if the file doesn't exist.
    Ensures all default keys are present in the loaded config, adding missing ones.
    Saves the updated config back to the file if keys were added.
    """
    default_config = {
        "interval": "5", # Default interval (ensure it's in VALID_INTERVALS)
        "retry_delay": RETRY_DELAY_SECONDS, # Base retry delay
        "loop_delay": LOOP_DELAY_SECONDS, # Delay between analysis cycles
        "quote_currency": "USDT", # Currency for balance check and sizing
        "enable_trading": False, # SAFETY FIRST: Default to False
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet)
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Desired leverage (check exchange limits!)
        "max_concurrent_positions": 1, # Informational, not enforced by this script version
        # --- Indicator Periods & Settings ---
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
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF,
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        # --- Signal Generation & Thresholds ---
        "orderbook_limit": 25, # Depth of orderbook to fetch
        "signal_score_threshold": 1.5, # Score needed to trigger BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25.0, # Use float
        "stoch_rsi_overbought_threshold": 75.0, # Use float
        "volume_confirmation_multiplier": 1.5, # Volume spike confirmation factor
        "scalping_signal_threshold": 2.5, # Example: Separate threshold for 'scalping' weight set
        # --- Risk Management Multipliers (based on ATR) ---
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        # --- Exit Strategies ---
        "enable_ma_cross_exit": True, # Close position on adverse EMA cross
        # --- Trailing Stop Loss Config (Exchange-based TSL) ---
        "enable_trailing_stop": True, # Enable exchange TSL
        # Trail distance as a percentage of entry price (used for calculation)
        "trailing_stop_callback_rate": 0.005, # e.g., 0.5% trail distance relative to entry
        # Activation profit percentage (0 for immediate activation via API)
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate when 0.3% in profit
        # --- Break-Even Stop Config ---
        "enable_break_even": True, # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # Trigger BE when profit = X * Current ATR
        "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry price
        # --- Indicator Enable/Disable Flags ---
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Flag to enable fetching and scoring orderbook data
        },
        # --- Indicator Weighting Sets ---
        "weight_sets": {
            "scalping": { # Example: Fast scalping weights
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Example: Balanced weights
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
            # Add more weight sets here as needed
        },
        "active_weight_set": "default" # Choose which weight set to use
    }

    # --- File Handling ---
    if not os.path.exists(filepath):
        print(f"{NEON_YELLOW}Config file not found at '{filepath}'. Creating default config...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using in-memory defaults.{RESET}")
            # Return default config anyway if file creation fails
            return default_config

    # --- Load Existing Config and Merge Defaults ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config_from_file = json.load(f)
        # Ensure all default keys exist in the loaded config recursively
        updated_config, keys_added = _ensure_config_keys(config_from_file, default_config)
        # Save back if keys were added during the update
        if keys_added:
            print(f"{NEON_YELLOW}Updating config file '{filepath}' with missing default keys...{RESET}")
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                print(f"{NEON_GREEN}Config file updated successfully.{RESET}")
            except IOError as e:
                print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
        return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config
    except Exception as e:
        print(f"{NEON_RED}Unexpected error loading configuration: {e}. Using defaults.{RESET}")
        return default_config

# Load configuration globally after defining the function
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency for global use

# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object with appropriate settings,
    error handling, and basic connection/authentication tests.

    Args:
        config: The loaded configuration dictionary.
        logger: The logger instance for logging messages.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger # Alias for convenience
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'linear', # Assume linear contracts (USDT margined)
                'adjustForTimeDifference': True, # Auto-sync time with server
                # Connection timeouts (milliseconds) - adjust as needed
                'fetchTickerTimeout': 10000,    # 10 seconds
                'fetchBalanceTimeout': 15000,   # 15 seconds
                'createOrderTimeout': 20000,    # 20 seconds for placing orders
                'fetchOrderTimeout': 15000,     # 15 seconds for fetching orders
                'fetchPositionsTimeout': 15000, # 15 seconds for fetching positions
                'cancelOrderTimeout': 15000,    # 15 seconds for cancelling orders
                # Bybit V5 Specific Options (Crucial)
                'default_options': {
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                    'recvWindow': 10000, # Optional: Increase recvWindow if needed
                    # Explicitly request V5 API for key endpoints
                    'fetchPositions': 'v5',
                    'fetchBalance': 'v5',
                    'createOrder': 'v5',
                    'fetchOrder': 'v5',
                    'cancelOrder': 'v5',
                    'setLeverage': 'v5',
                    'private_post_v5_position_trading_stop': 'v5', # For SL/TP/TSL setting
                    # Add other endpoints as needed
                },
                'accounts': { # Define V5 account types for ccxt mapping
                    'future': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'swap': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'option': {'unified': 'OPTION'}, # Unified account type
                    'spot': {'unified': 'SPOT'},     # Unified account type
                    # Add other mappings if using different account structures
                },
                 # Optional: Add a broker ID for tracking on Bybit's side
                'brokerId': 'livebot71',
            }
        }

        # Select Bybit exchange class dynamically
        exchange_id = 'bybit' # Use lowercase id
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        # Set sandbox mode if configured
        if config.get('use_sandbox', True): # Default to sandbox if key missing
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}USING LIVE TRADING MODE (Real Money){RESET}")


        # --- Test Connection & Load Markets ---
        lg.info(f"Connecting to {exchange.id} (Sandbox: {config.get('use_sandbox', True)})...")
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            exchange.load_markets()
            lg.info(f"Markets loaded successfully for {exchange.id}.")
        except ccxt.NetworkError as e:
            lg.error(f"{NEON_RED}Network error loading markets: {e}. Check connection.{RESET}")
            return None
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error loading markets: {e}{RESET}")
            return None # Cannot proceed without markets
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error loading markets: {e}{RESET}", exc_info=True)
            return None

        lg.info(f"CCXT exchange initialized ({exchange.id}). CCXT Version: {ccxt.__version__}")

        # --- Test API Credentials & Permissions (Fetch Balance) ---
        # Use the specific QUOTE_CURRENCY from config
        quote_currency = config.get("quote_currency", "USDT")
        lg.info(f"Attempting initial balance fetch for {quote_currency}...")
        try:
            # Use the enhanced fetch_balance function
            balance_decimal = fetch_balance(exchange, quote_currency, lg)
            if balance_decimal is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({quote_currency} available: {balance_decimal:.4f})")
            else:
                 # fetch_balance logs specific errors, add a general warning here
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch returned None or failed. Check logs. Ensure API keys have 'Read' permissions and correct account type (CONTRACT/UNIFIED) is accessible.{RESET}")
                 # Decide if this is critical. If trading is enabled, it probably is.
                 if config.get("enable_trading"):
                     lg.error(f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}")
                     return None
                 else:
                     lg.warning("Continuing in non-trading mode despite balance fetch issue.")

        except ccxt.AuthenticationError as auth_err:
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            lg.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None # Critical failure, cannot proceed
        except Exception as balance_err:
            # Catch other potential errors during balance fetch
            lg.warning(f"{NEON_YELLOW}Warning during initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")
            # Only abort if trading is enabled
            if config.get("enable_trading"):
                 lg.error(f"{NEON_RED}Aborting initialization due to balance fetch error in trading mode.{RESET}")
                 return None


        return exchange

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        lg.error(f"{NEON_RED}>> Check API keys, permissions, IP whitelist, and Real/Testnet selection.{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None


# --- CCXT Data Fetching with Retries ---
def _handle_rate_limit(error: ccxt.RateLimitExceeded, logger: logging.Logger) -> int:
    """Parses rate limit error message to find suggested wait time."""
    default_wait = RETRY_DELAY_SECONDS * 3 # Longer default for rate limits
    try:
        error_msg = str(error).lower()
        if 'try again in' in error_msg:
            parts = error_msg.split('try again in')
            if len(parts) > 1:
                time_part = parts[1].split('ms')[0].strip()
                wait_ms = int(time_part)
                wait_sec = max(1, math.ceil(wait_ms / 1000) + 1) # Add buffer, min 1s
                logger.warning(f"Rate limit suggests waiting {wait_sec}s.")
                return wait_sec
        elif 'rate limit' in error_msg: # Generic message, look for digits
            import re
            match = re.search(r'(\d+)\s*(ms|s)', error_msg)
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                wait_sec = num / 1000 if unit == 'ms' else num
                wait_sec = max(1, math.ceil(wait_sec + 1)) # Add buffer, min 1s
                logger.warning(f"Rate limit suggests waiting {wait_sec}s.")
                return wait_sec
    except Exception as parse_err:
        logger.warning(f"Could not parse rate limit wait time from '{error}': {parse_err}. Using default {default_wait}s.")
    return default_wait

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetch the current price of a trading symbol using CCXT ticker with retries and fallbacks.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: The logger instance.

    Returns:
        The current price as a Decimal, or None if fetching fails.
    """
    lg = logger
    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})...")
            ticker = exchange.fetch_ticker(symbol)
            lg.debug(f"Raw ticker data for {symbol}: {ticker}")

            price = None
            last_price = ticker.get('last')
            bid_price = ticker.get('bid')
            ask_price = ticker.get('ask')

            # Helper to safely convert to Decimal and check positivity
            def safe_decimal(value: Any) -> Optional[Decimal]:
                if value is None: return None
                try:
                    d_val = Decimal(str(value))
                    return d_val if d_val > 0 else None
                except (InvalidOperation, ValueError, TypeError):
                    return None

            # 1. Try 'last' price
            price = safe_decimal(last_price)
            if price:
                lg.debug(f"Using 'last' price for {symbol}: {price}")
            else:
                # 2. If 'last' is invalid, try bid/ask midpoint
                bid_decimal = safe_decimal(bid_price)
                ask_decimal = safe_decimal(ask_price)
                if bid_decimal and ask_decimal:
                    if bid_decimal <= ask_decimal:
                        price = (bid_decimal + ask_decimal) / 2
                        lg.debug(f"Using bid/ask midpoint for {symbol}: {price} (Bid: {bid_decimal}, Ask: {ask_decimal})")
                    else:
                        lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' as fallback.")
                        price = ask_decimal # Use ask as a safer fallback
                else:
                    # 3. If midpoint fails, try ask price
                    price = ask_decimal
                    if price:
                        lg.warning(f"Using 'ask' price as fallback for {symbol}: {price}")
                    else:
                        # 4. If ask fails, try bid price
                        price = bid_decimal
                        if price:
                             lg.warning(f"Using 'bid' price as last resort fallback for {symbol}: {price}")

            # Final check
            if price:
                return price
            else:
                lg.error(f"{NEON_RED}Failed to fetch a valid positive current price for {symbol} from ticker data.{RESET}")
                return None # Return None if no valid price found in this attempt

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(f"Rate limit hit fetching ticker for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"Network error fetching ticker for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})")
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching ticker for {symbol}: {e}. Not retrying.{RESET}")
            return None # Don't retry on definitive exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching ticker for {symbol}: {e}. Not retrying.{RESET}", exc_info=True)
            return None # Don't retry on unexpected errors

    lg.error(f"{NEON_RED}Max retries reached fetching ticker for {symbol}. Last error: {last_exception}{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Fetch OHLCV kline data using CCXT with retries and basic validation.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        timeframe: The timeframe string expected by ccxt (e.g., '1m', '1h', '1d').
        limit: The number of klines to fetch.
        logger: The logger instance.

    Returns:
        A pandas DataFrame with OHLCV data and datetime index, or an empty DataFrame on failure.
    """
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    ohlcv = None
    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
            # Add category param for Bybit V5 if needed (linear/inverse)
            params = {}
            if 'bybit' in exchange.id.lower():
                 # Attempt to infer category from market info if available
                 try:
                     market = exchange.market(symbol)
                     category = 'linear' if market.get('linear', True) else 'inverse'
                 except Exception:
                     # Fallback guess if market info fails
                     category = 'linear' if 'USDT' in symbol else 'inverse'
                 params['category'] = category
                 lg.debug(f"Using params for fetch_ohlcv: {params}")

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)

            if ohlcv is not None and len(ohlcv) > 0:
                # --- Data Processing and Validation ---
                try:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    if df.empty:
                        lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} after creation.{RESET}")
                        # Potentially retry if empty dataframe is transient? For now, return empty.
                        return pd.DataFrame()

                    # Convert timestamp to datetime and set as index
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                    df.dropna(subset=['timestamp'], inplace=True)
                    if df.empty:
                        lg.warning(f"{NEON_YELLOW}Kline data DataFrame empty after timestamp conversion/dropna for {symbol} {timeframe}.{RESET}")
                        return pd.DataFrame()
                    df.set_index('timestamp', inplace=True)

                    # Ensure numeric types, coerce errors to NaN
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Data Cleaning: Drop rows with NaN in critical columns or invalid prices/volume
                    initial_len = len(df)
                    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Require OHLC
                    df = df[df['close'] > 0] # Require positive close price
                    # Optional: Require positive volume? Depends on exchange/data source
                    # df = df[df['volume'] > 0]

                    rows_dropped = initial_len - len(df)
                    if rows_dropped > 0:
                        lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price/volume data for {symbol}.")

                    if df.empty:
                        lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
                        return pd.DataFrame()

                    # Sort index just in case data isn't perfectly ordered
                    df.sort_index(inplace=True)

                    lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
                    return df # Success

                except Exception as proc_err:
                    lg.error(f"{NEON_RED}Error processing kline data for {symbol}: {proc_err}. Not retrying.{RESET}", exc_info=True)
                    return pd.DataFrame() # Return empty on processing error
            else:
                # fetch_ohlcv returned None or empty list
                lg.warning(f"fetch_ohlcv returned no data for {symbol} (Attempt {attempt + 1}). Retrying...")
                last_exception = ValueError("API returned no kline data") # Set placeholder exception
                time.sleep(1) # Short delay before retry on empty data

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(f"Rate limit hit fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"Network error fetching klines for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})")
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}. Not retrying.{RESET}")
            return pd.DataFrame() # Don't retry on definitive exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}. Not retrying.{RESET}", exc_info=True)
            return pd.DataFrame() # Don't retry on unexpected errors

    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol}. Last error: {last_exception}{RESET}")
    return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """
    Fetch orderbook data using ccxt with retries and basic validation.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol.
        limit: The maximum number of bids/asks to fetch.
        logger: The logger instance.

    Returns:
        The orderbook dictionary, or None if fetching fails.
    """
    lg = logger
    if not exchange.has['fetchOrderBook']:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
            # Add category param for Bybit V5 if needed
            params = {}
            if 'bybit' in exchange.id.lower():
                 try:
                     market = exchange.market(symbol)
                     category = 'linear' if market.get('linear', True) else 'inverse'
                 except Exception:
                     category = 'linear' if 'USDT' in symbol else 'inverse' # Fallback guess
                 params['category'] = category
                 lg.debug(f"Using params for fetch_order_book: {params}")

            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            # --- Validation ---
            if not orderbook:
                lg.warning(f"fetch_order_book returned None or empty data for {symbol} (Attempt {attempt + 1}). Retrying...")
                last_exception = ValueError("API returned no orderbook data")
                time.sleep(1) # Short delay
                continue
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"{NEON_YELLOW}Invalid orderbook structure (bids/asks not lists) for {symbol}. Attempt {attempt + 1}. Response: {orderbook}{RESET}")
                last_exception = TypeError("Invalid orderbook structure received")
                time.sleep(RETRY_DELAY_SECONDS) # Retry on structure issues
                continue
            elif not orderbook['bids'] and not orderbook['asks']:
                 # Valid structure, but empty book (thin liquidity)
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempt + 1}).{RESET}")
                 return orderbook # Return the empty book, let analysis handle it
            else:
                 # Looks valid
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook # Success

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(f"Rate limit hit fetching orderbook for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"Network error fetching orderbook for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})")
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}. Not retrying.{RESET}")
            return None # Don't retry on definitive exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}. Not retrying.{RESET}", exc_info=True)
            return None # Don't retry on unexpected errors

    lg.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}. Last error: {last_exception}{RESET}")
    return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes trading data using pandas_ta, generates weighted signals,
    and provides risk management calculation helpers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any], # Pass market info for precision etc.
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Pandas DataFrame with OHLCV data, indexed by timestamp.
            logger: The logger instance.
            config: The loaded configuration dictionary.
            market_info: Market information dictionary from ccxt.
        """
        if df is None or not isinstance(df, pd.DataFrame):
             raise ValueError("TradingAnalyzer requires a valid pandas DataFrame.")
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config

        if not market_info or not isinstance(market_info, dict):
            raise ValueError("TradingAnalyzer requires a valid market_info dictionary.")
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")

        # --- Internal State ---
        self.indicator_values: Dict[str, float] = {} # Stores latest indicator float values
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1} # Simple signal state, default HOLD
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated fib levels (as Decimals)
        self.ta_column_names: Dict[str, Optional[str]] = {} # Stores actual column names generated by pandas_ta
        # Track break-even status per instance (cleared each cycle by re-initializing analyzer)
        self.break_even_triggered = False

        if not self.weights:
            logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will not work.{RESET}")

        # --- Initialization Steps ---
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Calculate Fib levels after indicators are ready

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """Helper to find the actual column name generated by pandas_ta, handling variations."""
        cfg = self.config # Shortcut
        # Ensure float conversion for std dev is safe
        try:
            bb_std_dev = float(cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV))
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid bollinger_bands_std_dev '{cfg.get('bollinger_bands_std_dev')}' in config. Using default {DEFAULT_BOLLINGER_BANDS_STD_DEV}.")
            bb_std_dev = DEFAULT_BOLLINGER_BANDS_STD_DEV

        # Define expected patterns based on config values
        expected_patterns = {
            "ATR": [f"ATRr_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}", f"ATR_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_0.015"], # Default const suffix
            "Williams_R": [f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP", "VWAP_D"], # Handle potential suffix like _D for daily reset
            "PSAR_long": [f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            # Use string formatting correctly for float std dev
            "BB_Lower": [f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "BB_Middle": [f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "BB_Upper": [f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "Volume_MA": [f"VOL_SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"] # Custom name
        }

        patterns_to_check = expected_patterns.get(base_name, [])
        available_columns = result_df.columns.tolist()

        # 1. Check exact expected patterns
        for pattern in patterns_to_check:
            if pattern in available_columns:
                self.logger.debug(f"Found exact TA column '{pattern}' for base '{base_name}'.")
                return pattern

        # 2. Check variations (e.g., without const suffix like CCI)
        for pattern in patterns_to_check:
            base_pattern_parts = pattern.split('_')
            if len(base_pattern_parts) > 2 and base_pattern_parts[-1].replace('.','').isdigit():
                pattern_no_suffix = '_'.join(base_pattern_parts[:-1])
                if pattern_no_suffix in available_columns:
                    self.logger.debug(f"Found TA column '{pattern_no_suffix}' for base '{base_name}' (without const suffix).")
                    return pattern_no_suffix

        # 3. Check for simpler base name with parameters (less common but possible)
        for pattern in patterns_to_check:
             base_only = pattern.split('_')[0] # e.g., "ATRr"
             if base_only in available_columns:
                  self.logger.debug(f"Found TA column '{base_only}' for base '{base_name}' (parameter-less variation).")
                  return base_only

        # 4. Fallback: Search based on base name (case-insensitive)
        #    Prioritize prefix match (e.g., "CCI_...") over contains match
        base_lower = base_name.lower()
        prefix_match = None
        contains_match = None
        for col in available_columns:
            col_lower = col.lower()
            if col_lower.startswith(base_lower + "_"):
                prefix_match = col
                break # Found best prefix match
            if not contains_match and base_lower in col_lower:
                contains_match = col # Keep first contains match as fallback

        if prefix_match:
            self.logger.debug(f"Found TA column '{prefix_match}' for base '{base_name}' using prefix fallback.")
            return prefix_match
        if contains_match:
            self.logger.debug(f"Found TA column '{contains_match}' for base '{base_name}' using contains fallback.")
            return contains_match

        # If nothing found
        self.logger.warning(f"{NEON_YELLOW}Could not find column name for indicator '{base_name}' in DataFrame columns: {available_columns}{RESET}")
        return None

    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # --- Check for sufficient data length ---
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        periods_needed = []
        # Always calculate ATR as it's used for sizing/SL/TP/BE
        periods_needed.append(cfg.get("atr_period", DEFAULT_ATR_PERIOD))
        # Check EMA periods only if alignment or MA cross exit is enabled
        if indi_cfg.get("ema_alignment", False) or cfg.get("enable_ma_cross_exit", False):
            periods_needed.append(max(cfg.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD),
                                      cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)))
        # Add periods for other enabled indicators
        if indi_cfg.get("momentum"): periods_needed.append(cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        if indi_cfg.get("cci"): periods_needed.append(cfg.get("cci_window", DEFAULT_CCI_WINDOW))
        if indi_cfg.get("wr"): periods_needed.append(cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        if indi_cfg.get("mfi"): periods_needed.append(cfg.get("mfi_window", DEFAULT_MFI_WINDOW))
        if indi_cfg.get("sma_10"): periods_needed.append(cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW))
        if indi_cfg.get("stoch_rsi"): periods_needed.append(cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + cfg.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)) # StochRSI needs RSI window too
        if indi_cfg.get("rsi"): periods_needed.append(cfg.get("rsi_period", DEFAULT_RSI_WINDOW))
        if indi_cfg.get("bollinger_bands"): periods_needed.append(cfg.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        if indi_cfg.get("volume_confirmation"): periods_needed.append(cfg.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
        # PSAR, VWAP don't have simple period dependencies in the same way

        # Calculate minimum required length (max period + buffer for calculation stability)
        min_required_data = max(periods_needed) + 20 if periods_needed else 50 # Add buffer

        if len(self.df) < min_required_data:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators reliably (min recommended: {min_required_data}). Results may be inaccurate or NaN.{RESET}")
            # Continue calculation, but expect NaNs at the start

        try:
            # Work on a copy to preserve original OHLCV data if needed elsewhere
            df_calc = self.df.copy()

            # --- Calculate Indicators using pandas_ta ---
            indicators_config = self.config.get("indicators", {})
            calculate_emas = indicators_config.get("ema_alignment", False) or self.config.get("enable_ma_cross_exit", False)

            # --- Always calculate ATR (required for risk management) ---
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # --- Calculate other indicators based on config flags ---
            if calculate_emas:
                ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                df_calc.ta.ema(length=ema_short, append=True)
                self.ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
                df_calc.ta.ema(length=ema_long, append=True)
                self.ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)

            if indicators_config.get("momentum", False):
                mom_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                df_calc.ta.mom(length=mom_period, append=True)
                self.ta_column_names["Momentum"] = self._get_ta_col_name("Momentum", df_calc)

            if indicators_config.get("cci", False):
                cci_period = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
                df_calc.ta.cci(length=cci_period, append=True)
                self.ta_column_names["CCI"] = self._get_ta_col_name("CCI", df_calc)

            if indicators_config.get("wr", False):
                wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
                df_calc.ta.willr(length=wr_period, append=True)
                self.ta_column_names["Williams_R"] = self._get_ta_col_name("Williams_R", df_calc)

            if indicators_config.get("mfi", False):
                mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
                df_calc.ta.mfi(length=mfi_period, append=True)
                self.ta_column_names["MFI"] = self._get_ta_col_name("MFI", df_calc)

            if indicators_config.get("vwap", False):
                # VWAP calculation depends on frequency (e.g., daily reset). pandas_ta handles this based on index.
                df_calc.ta.vwap(append=True)
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                # Use try-except as PSAR can sometimes fail on certain data patterns
                try:
                    psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                    if psar_result is not None and not psar_result.empty:
                        # Append safely, avoiding duplicate columns
                        for col in psar_result.columns:
                            if col not in df_calc.columns: df_calc[col] = psar_result[col]
                            else: df_calc[col] = psar_result[col] # Overwrite if exists
                        self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                        self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)
                    else: self.logger.warning(f"PSAR calculation returned empty result for {self.symbol}.")
                except Exception as e_psar: self.logger.error(f"Error calculating PSAR for {self.symbol}: {e_psar}")

            if indicators_config.get("sma_10", False):
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                try:
                    stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                    if stochrsi_result is not None and not stochrsi_result.empty:
                        for col in stochrsi_result.columns:
                            if col not in df_calc.columns: df_calc[col] = stochrsi_result[col]
                            else: df_calc[col] = stochrsi_result[col] # Overwrite
                        self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                        self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)
                    else: self.logger.warning(f"StochRSI calculation returned empty result for {self.symbol}.")
                except Exception as e_stoch: self.logger.error(f"Error calculating StochRSI for {self.symbol}: {e_stoch}")

            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
                try:
                    bb_std_float = float(bb_std) # Ensure float
                    bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std_float)
                    if bbands_result is not None and not bbands_result.empty:
                        for col in bbands_result.columns:
                             if col not in df_calc.columns: df_calc[col] = bbands_result[col]
                             else: df_calc[col] = bbands_result[col] # Overwrite
                        self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                        self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                        self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)
                    else: self.logger.warning(f"Bollinger Bands calculation returned empty result for {self.symbol}.")
                except Exception as e_bb: self.logger.error(f"Error calculating Bollinger Bands for {self.symbol}: {e_bb}")

            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}" # Custom name for clarity
                # Calculate SMA on volume column, handle potential NaNs in volume robustly
                try:
                    # Fill NaN volumes with 0 before calculating MA, then assign
                    df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)
                    self.ta_column_names["Volume_MA"] = vol_ma_col_name
                except Exception as e_vol: self.logger.error(f"Error calculating Volume MA for {self.symbol}: {e_vol}")

            # Assign the DataFrame with calculated indicators back to self.df
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
            # This might happen if a pandas_ta method name is incorrect or version incompatible
            self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} (check pandas_ta method name/version?): {e}{RESET}", exc_info=True)
            # self.df remains the original data without calculated indicators
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)
            # Keep original df for potential partial analysis? Or clear? Keep original for now.

    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest float values from self.df."""
        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame is empty for {self.symbol}.")
            self.indicator_values = {} # Ensure empty if df is empty
            return

        try:
            # Check if the DataFrame has rows before accessing iloc[-1]
            if len(self.df) == 0:
                self.logger.warning(f"Cannot update latest values: DataFrame has zero rows for {self.symbol}.")
                self.indicator_values = {}
                return

            latest = self.df.iloc[-1]

            # Check if the last row contains only NaNs (except potentially index)
            if latest.isnull().all():
                self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: Last row of DataFrame contains all NaNs for {self.symbol}.{RESET}")
                self.indicator_values = {}
                return

            updated_values = {}
            # --- Populate from calculated indicators using stored column names ---
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index: # Check column exists
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            updated_values[key] = float(value) # Store as float
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) to float for {self.symbol}.")
                            updated_values[key] = float('nan') # Store NaN on conversion failure
                    else:
                        updated_values[key] = float('nan') # Value is NaN in DataFrame
                else:
                    # If col_name is None or not found, store NaN
                    # Log only if the indicator was supposed to be calculated (i.e., key exists in expected names)
                    if key in self.ta_column_names: # Check if we expected this key
                         self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found or invalid in latest data for {self.symbol}. Storing NaN.")
                    updated_values[key] = float('nan')

            # --- Add essential price/volume data from the original DataFrame columns ---
            for base_col in ['close', 'volume', 'high', 'low', 'open']: # Added 'open'
                key_name = base_col.capitalize() # e.g., 'Close'
                value = latest.get(base_col) # Use .get() for safety if base columns somehow missing
                if pd.notna(value):
                    try:
                        updated_values[key_name] = float(value)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert base value for '{base_col}' ({value}) to float for {self.symbol}.")
                        updated_values[key_name] = float('nan')
                else:
                    updated_values[key_name] = float('nan')

            self.indicator_values = updated_values
            # Filter out NaN for debug log brevity
            valid_values = {k: f"{v:.5f}" if isinstance(v, float) else v for k, v in self.indicator_values.items() if pd.notna(v)}
            # Log at DEBUG level as it can be verbose
            self.logger.debug(f"Latest indicator float values updated for {self.symbol}: {valid_values}")

        except IndexError: # Catch error if df becomes empty between checks
            self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be empty or too short.")
            self.indicator_values = {} # Reset values
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = {} # Reset values

    # --- Precision and Market Info Helpers ---
    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        try:
            precision_info = self.market_info.get('precision', {})
            # 1. Prefer 'tick' if available and valid
            tick_val = precision_info.get('tick')
            if tick_val is not None:
                try:
                    tick_size = Decimal(str(tick_val))
                    if tick_size > 0: return tick_size
                except (InvalidOperation, ValueError, TypeError): pass # Try next

            # 2. Try 'price' if it represents step size (float/string, not int decimals)
            price_prec_val = precision_info.get('price')
            if isinstance(price_prec_val, (float, str)):
                try:
                    tick_size_from_price = Decimal(str(price_prec_val))
                    # Heuristic: If it's not an integer > 1, assume it's a tick size
                    if tick_size_from_price > 0 and not (tick_size_from_price.is_integer() and tick_size_from_price > 1):
                        return tick_size_from_price
                except (InvalidOperation, ValueError, TypeError): pass # Try next

            # 3. Try Bybit specific 'info.tickSize'
            bybit_tick_size = self.market_info.get('info', {}).get('tickSize')
            if bybit_tick_size is not None:
                 try:
                      tick_size_bybit = Decimal(str(bybit_tick_size))
                      if tick_size_bybit > 0: return tick_size_bybit
                 except (InvalidOperation, ValueError, TypeError): pass # Try next

            # 4. Try 'limits.price.min' (sometimes holds tick size, but risky)
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_tick_from_limit = Decimal(str(min_price_val))
                    # Use only if plausible (e.g., < 100)
                    if 0 < min_tick_from_limit < 100:
                        self.logger.debug(f"Using tick size from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                        return min_tick_from_limit
                except (InvalidOperation, ValueError, TypeError): pass # Ignore

        except Exception as e:
            self.logger.warning(f"Error determining min tick size for {self.symbol} from market info: {e}. Using fallback.")

        # Absolute fallback: Very small number if all else fails
        fallback_tick = Decimal('0.00000001') # Adjust if needed for specific exchanges
        self.logger.warning(f"{NEON_YELLOW}Could not reliably determine tick size for {self.symbol}. Using fallback: {fallback_tick}. Price quantization may be inaccurate.{RESET}")
        return fallback_tick

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info."""
        try:
            min_tick = self.get_min_tick_size()
            if min_tick > 0:
                # Calculate decimal places from tick size using Decimal properties
                precision = abs(min_tick.normalize().as_tuple().exponent)
                return precision
        except Exception as e:
            self.logger.warning(f"Error deriving precision from min tick size for {self.symbol}: {e}. Trying other methods.")

        # Fallback 1: Use 'precision.price' if it's an integer (decimal places)
        try:
            price_precision_val = self.market_info.get('precision', {}).get('price')
            if isinstance(price_precision_val, int) and price_precision_val >= 0:
                return price_precision_val
            # Handle if it's float/string representing tick size (infer precision)
            elif isinstance(price_precision_val, (float, str)):
                 tick_from_price = Decimal(str(price_precision_val))
                 if tick_from_price > 0 and not (tick_from_price.is_integer() and tick_from_price > 1):
                      return abs(tick_from_price.normalize().as_tuple().exponent)
        except Exception: pass # Ignore errors

        # Fallback 2: Infer from last close price format (less reliable)
        try:
            last_close = self.indicator_values.get("Close") # Uses float value
            if last_close and pd.notna(last_close) and last_close > 0:
                s_close = format(Decimal(str(last_close)), 'f') # Avoid scientific notation
                return len(s_close.split('.')[-1]) if '.' in s_close else 0
        except Exception: pass # Ignore errors

        # Default fallback precision
        default_precision = 4 # Common default for USDT pairs
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a specified window using Decimal."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}. Skipping.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure high/low are valid numbers before converting to Decimal
            high_price_raw = df_slice["high"].dropna().max()
            low_price_raw = df_slice["low"].dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                self.logger.warning(f"Could not find valid high/low in the last {window} periods for Fibonacci on {self.symbol}.")
                self.fib_levels_data = {}
                return {}

            high = Decimal(str(high_price_raw))
            low = Decimal(str(low_price_raw))
            diff = high - low

            levels = {}
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                self.logger.warning(f"Invalid min_tick_size ({min_tick}) for Fibonacci quantization on {self.symbol}. Levels will not be quantized.")
                min_tick = None # Disable quantization

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level assuming retracement from High towards Low
                    level_price = high - (diff * Decimal(str(level_pct)))
                    # Quantize the result DOWN to the nearest tick size
                    if min_tick:
                        level_price = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    levels[level_name] = level_price
            else: # Handle zero range (high == low)
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}. Setting levels to high/low.")
                 level_price = high
                 if min_tick: # Quantize the single level too
                      level_price = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price

            self.fib_levels_data = levels
            # Log levels formatted as strings for readability
            log_levels = {k: f"{v:.{self.get_price_precision()}f}" for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {log_levels}")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 3) -> list[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels (name, price) to the current price."""
        if not self.fib_levels_data:
            return []
        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal): # Ensure level is Decimal
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    self.logger.warning(f"Non-decimal value found in fib_levels_data for {self.symbol}: {name}={level_price} ({type(level_price)})")

            level_distances.sort(key=lambda x: x['distance'])
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score based on latest values. Returns float score or NaN."""
        ema_short = self.indicator_values.get("EMA_Short", float('nan'))
        ema_long = self.indicator_values.get("EMA_Long", float('nan'))
        current_price = self.indicator_values.get("Close", float('nan'))

        if math.isnan(ema_short) or math.isnan(ema_long) or math.isnan(current_price):
            return float('nan') # Return NaN if data is missing

        # Bullish alignment: Price > Short EMA > Long EMA
        if current_price > ema_short > ema_long: return 1.0
        # Bearish alignment: Price < Short EMA < Long EMA
        elif current_price < ema_short < ema_long: return -1.0
        # Other cases are neutral or mixed signals
        else: return 0.0

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores.

        Args:
            current_price: The current price as a Decimal.
            orderbook_data: Fetched orderbook data dictionary, or None.

        Returns:
            The generated signal string ("BUY", "SELL", or "HOLD").
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Reset signals, default HOLD
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {} # For detailed logging

        # --- Essential Data Checks ---
        if not self.indicator_values:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values dictionary is empty.{RESET}")
            return "HOLD"
        if current_price is None or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
            return "HOLD"

        # --- Get Active Weights ---
        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
            self.logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' is missing or empty in config for {self.symbol}. Cannot generate signal.{RESET}")
            return "HOLD"

        # --- Iterate Through Enabled Indicators with Weights ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue # Skip disabled indicators

            weight_str = active_weights.get(indicator_key)
            if weight_str is None: continue # Skip if no weight defined

            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue # Skip zero weight
            except (InvalidOperation, ValueError, TypeError):
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping.")
                continue

            # Find and call the check method dynamically
            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method = getattr(self, check_method_name)
                indicator_score_float = float('nan') # Default to NaN
                try:
                    # Pass specific arguments if needed
                    if indicator_key == "orderbook":
                        if orderbook_data: indicator_score_float = method(orderbook_data, current_price)
                        else: self.logger.debug(f"Orderbook data not available for {self.symbol}, skipping check.")
                    else:
                        indicator_score_float = method() # Returns float score or NaN

                except Exception as e:
                    self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                    # indicator_score_float remains NaN

                # --- Process Score ---
                debug_scores[indicator_key] = f"{indicator_score_float:.2f}" if not math.isnan(indicator_score_float) else "NaN"
                if not math.isnan(indicator_score_float):
                    try:
                        # Clamp score between -1 and 1 before applying weight
                        clamped_score = max(-1.0, min(1.0, indicator_score_float))
                        score_contribution = Decimal(str(clamped_score)) * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight
                        active_indicator_count += 1
                    except (InvalidOperation, ValueError, TypeError, Exception) as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} ({indicator_score_float}): {calc_err}")
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1 # Count indicators that returned NaN
            else:
                self.logger.warning(f"Check method '{check_method_name}' not found or not callable for enabled/weighted indicator: {indicator_key} ({self.symbol})")

        # --- Determine Final Signal ---
        if total_weight_applied == 0:
            self.logger.warning(f"No indicators contributed to the signal score for {self.symbol} (Total Weight Applied = 0). Defaulting to HOLD.")
            final_signal = "HOLD"
        else:
            try:
                # Use specific threshold if active weight set is 'scalping', else default
                threshold_key = "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
                default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
                threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
            except (InvalidOperation, ValueError, TypeError):
                self.logger.warning(f"Invalid {threshold_key} in config. Using default {default_threshold}.")
                threshold = Decimal(str(default_threshold))

            if final_signal_score >= threshold: final_signal = "BUY"
            elif final_signal_score <= -threshold: final_signal = "SELL"
            else: final_signal = "HOLD"

        # --- Log Summary ---
        price_precision = self.get_price_precision()
        score_details = ", ".join([f"{k}: {v}" for k, v in debug_scores.items()]) # Full details for DEBUG
        log_msg = (
            f"Signal Calculation Summary ({self.symbol} @ {current_price:.{price_precision}f}):\n"
            f"  Weight Set: {self.active_weight_set_name}\n"
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN)\n"
            f"  Total Weight Applied: {total_weight_applied:.3f}\n"
            f"  Final Weighted Score: {final_signal_score:.4f}\n"
            f"  Signal Threshold: +/- {threshold:.3f}\n"
            f"  ==> Final Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        self.logger.debug(f"  Detailed Scores: {score_details}") # Log details only at DEBUG level

        # Update internal signal state
        if final_signal in self.signals:
            self.signals[final_signal] = 1
            if final_signal != "HOLD": self.signals["HOLD"] = 0 # Unset HOLD if BUY/SELL

        return final_signal

    # --- Indicator Check Methods ---
    # Each method should return a float score between -1.0 and 1.0, or float('nan')

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment. Relies on calculate_ema_alignment_score."""
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
            self.logger.debug(f"EMA Alignment check skipped for {self.symbol}: EMAs not calculated/available.")
            return float('nan')
        # calculate_ema_alignment_score handles NaNs internally
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        """Checks Momentum indicator."""
        momentum = self.indicator_values.get("Momentum", float('nan'))
        if math.isnan(momentum): return float('nan')
        # Simple scaling: Assume momentum oscillates around 0. Scale linearly.
        # Need asset-specific tuning for scale_factor. Example: Scale +/- 0.5 to +/- 1.0
        scale_factor = 2.0
        score = momentum * scale_factor
        return max(-1.0, min(1.0, score)) # Clamp

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume supports potential move (relative to MA). Score is direction-neutral."""
        current_volume = self.indicator_values.get("Volume", float('nan'))
        volume_ma = self.indicator_values.get("Volume_MA", float('nan'))

        if math.isnan(current_volume) or math.isnan(volume_ma) or volume_ma <= 0:
            return float('nan')

        try:
            multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        except (ValueError, TypeError):
             self.logger.warning(f"Invalid volume_confirmation_multiplier in config. Using 1.5.")
             multiplier = 1.5

        if current_volume > volume_ma * multiplier: return 0.7 # High volume = Confirmation/Significance
        elif current_volume < volume_ma / multiplier: return -0.4 # Low volume = Lack of confirmation
        else: return 0.0 # Neutral volume

    def _check_stoch_rsi(self) -> float:
        """Checks Stochastic RSI K and D lines using configured thresholds."""
        k = self.indicator_values.get("StochRSI_K", float('nan'))
        d = self.indicator_values.get("StochRSI_D", float('nan'))
        if math.isnan(k) or math.isnan(d): return float('nan')

        try:
            oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
            overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        except (ValueError, TypeError):
            self.logger.warning("Invalid Stoch RSI thresholds in config. Using defaults 25/75.")
            oversold, overbought = 25.0, 75.0

        score = 0.0
        # 1. Extreme Zones (strongest signals)
        if k < oversold and d < oversold: score = 1.0 # Both deep oversold -> Strong bullish
        elif k > overbought and d > overbought: score = -1.0 # Both deep overbought -> Strong bearish
        # 2. K vs D Crossover/Momentum (Simplified check: K above/below D)
        elif k > d: score = max(score, 0.5) # K above D -> Bullish momentum bias
        elif k < d: score = min(score, -0.5) # K below D -> Bearish momentum bias
        # 3. Position within range (Refinement)
        if oversold <= k <= overbought:
             # Optional: Scale score based on proximity to midpoint (50) for finer grading
             range_width = overbought - oversold
             if range_width > 0:
                  mid_range_score = (k - (oversold + range_width / 2)) / (range_width / 2) # Scales -1 to 1 within range
                  # Combine gently with existing score (e.g., weighted average)
                  score = (score * 0.7) + (mid_range_score * 0.3)

        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_rsi(self) -> float:
        """Checks RSI indicator using standard levels (30/70)."""
        rsi = self.indicator_values.get("RSI", float('nan'))
        if math.isnan(rsi): return float('nan')

        if rsi <= 30: return 1.0 # Oversold -> Bullish
        if rsi >= 70: return -1.0 # Overbought -> Bearish
        # Scale linearly between 30 and 70 for smoother signal
        if 30 < rsi < 70:
             # Maps 30 to +1, 70 to -1 linearly
             return 1.0 - (rsi - 30.0) * (2.0 / 40.0)
        return 0.0 # Should not be reached if logic is correct

    def _check_cci(self) -> float:
        """Checks CCI indicator using standard levels (+/-100, +/-150)."""
        cci = self.indicator_values.get("CCI", float('nan'))
        if math.isnan(cci): return float('nan')

        if cci <= -150: return 1.0 # Strong Oversold -> Bullish
        if cci >= 150: return -1.0 # Strong Overbought -> Bearish
        if cci <= -100: return 0.6 # Moderate Oversold
        if cci >= 100: return -0.6 # Moderate Overbought
        # Scale linearly between -100 and 100
        if -100 < cci < 100:
             # Maps -100 to +0.6, +100 to -0.6
             return - (cci / 100.0) * 0.6
        return 0.0 # Fallback

    def _check_wr(self) -> float: # Williams %R
        """Checks Williams %R indicator using standard levels (-80/-20)."""
        wr = self.indicator_values.get("Williams_R", float('nan'))
        if math.isnan(wr): return float('nan')
        # WR range: -100 (most oversold) to 0 (most overbought)
        if wr <= -80: return 1.0 # Oversold -> Bullish
        if wr >= -20: return -1.0 # Overbought -> Bearish
        # Scale linearly between -80 and -20
        if -80 < wr < -20:
            # Maps -80 to +1.0, -20 to -1.0
            return 1.0 - (wr - (-80.0)) * (2.0 / 60.0)
        return 0.0 # Fallback (shouldn't be reached)

    def _check_psar(self) -> float:
        """Checks Parabolic SAR relative to price."""
        # PSAR values indicate the stop level. Signal comes from which one is active (non-NaN).
        psar_l = self.indicator_values.get("PSAR_long", float('nan')) # Value below price if active
        psar_s = self.indicator_values.get("PSAR_short", float('nan')) # Value above price if active

        if not math.isnan(psar_l) and math.isnan(psar_s): return 1.0 # PSAR below price -> Uptrend
        elif math.isnan(psar_l) and not math.isnan(psar_s): return -1.0 # PSAR above price -> Downtrend
        else:
            # Both NaN (start of data) or both have values (shouldn't happen with ta.psar)
            self.logger.debug(f"PSAR state ambiguous/NaN for {self.symbol} (L={psar_l}, S={psar_s})")
            return 0.0 # Neutral or undetermined

    def _check_sma_10(self) -> float: # Example using SMA10 vs Close
        """Checks price relative to SMA10."""
        sma_10 = self.indicator_values.get("SMA10", float('nan'))
        last_close = self.indicator_values.get("Close", float('nan'))
        if math.isnan(sma_10) or math.isnan(last_close): return float('nan')

        if last_close > sma_10: return 0.6 # Price above SMA -> Bullish bias
        if last_close < sma_10: return -0.6 # Price below SMA -> Bearish bias
        return 0.0

    def _check_vwap(self) -> float:
        """Checks price relative to VWAP."""
        vwap = self.indicator_values.get("VWAP", float('nan'))
        last_close = self.indicator_values.get("Close", float('nan'))
        if math.isnan(vwap) or math.isnan(last_close): return float('nan')

        if last_close > vwap: return 0.7 # Price above VWAP -> Bullish bias
        if last_close < vwap: return -0.7 # Price below VWAP -> Bearish bias
        return 0.0

    def _check_mfi(self) -> float:
        """Checks Money Flow Index using standard levels (20/80)."""
        mfi = self.indicator_values.get("MFI", float('nan'))
        if math.isnan(mfi): return float('nan')

        if mfi <= 20: return 1.0 # Oversold -> Bullish
        if mfi >= 80: return -1.0 # Overbought -> Bearish
        # Scale linearly between 20 and 80
        if 20 < mfi < 80:
             # Maps 20 to +1, 80 to -1
             return 1.0 - (mfi - 20.0) * (2.0 / 60.0)
        return 0.0 # Fallback

    def _check_bollinger_bands(self) -> float:
        """Checks price relative to Bollinger Bands."""
        bb_lower = self.indicator_values.get("BB_Lower", float('nan'))
        bb_upper = self.indicator_values.get("BB_Upper", float('nan'))
        bb_middle = self.indicator_values.get("BB_Middle", float('nan'))
        last_close = self.indicator_values.get("Close", float('nan'))
        if math.isnan(bb_lower) or math.isnan(bb_upper) or math.isnan(bb_middle) or math.isnan(last_close):
            return float('nan')

        # 1. Price touching/outside outer bands (strong mean reversion signal)
        if last_close <= bb_lower: return 1.0 # Below lower band -> Strong bullish potential
        if last_close >= bb_upper: return -1.0 # Above upper band -> Strong bearish potential

        # 2. Price relative to middle band (trend indication within bands)
        # Scale score based on position between middle and outer bands
        upper_range = bb_upper - bb_middle
        lower_range = bb_middle - bb_lower

        if last_close > bb_middle and upper_range > 0:
            # Price in upper half: Scale from 0 (at middle) to -1 (at upper band)
            proximity_to_upper = (last_close - bb_middle) / upper_range
            score = 0.0 - proximity_to_upper # Max 0, min -1
            return max(-1.0, min(0.0, score)) # Clamp result

        elif last_close < bb_middle and lower_range > 0:
            # Price in lower half: Scale from 0 (at middle) to +1 (at lower band)
            proximity_to_lower = (bb_middle - last_close) / lower_range
            score = 0.0 + proximity_to_lower # Max +1, min 0
            return max(0.0, min(1.0, score)) # Clamp result

        return 0.0 # Exactly on middle band or if ranges are zero

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes order book depth for immediate pressure using Order Book Imbalance (OBI)."""
        if not orderbook_data:
            self.logger.debug(f"Orderbook check skipped for {self.symbol}: No data provided.")
            return float('nan')

        try:
            bids = orderbook_data.get('bids', []) # List of [price_str, volume_str]
            asks = orderbook_data.get('asks', [])

            if not bids or not asks:
                self.logger.debug(f"Orderbook check skipped for {self.symbol}: Bids or asks list is empty.")
                return float('nan') # Need both sides for imbalance

            # --- Simple Order Book Imbalance (OBI) within N levels ---
            num_levels_to_check = 10 # Configurable? Check top N levels
            try:
                bid_volume_sum = sum(Decimal(str(bid[1])) for bid in bids[:num_levels_to_check])
                ask_volume_sum = sum(Decimal(str(ask[1])) for ask in asks[:num_levels_to_check])
            except (InvalidOperation, ValueError, TypeError, IndexError) as e:
                self.logger.warning(f"Error parsing orderbook levels for OBI calculation ({self.symbol}): {e}")
                return float('nan')

            total_volume = bid_volume_sum + ask_volume_sum
            if total_volume <= 0: # Use <= to handle potential edge cases
                self.logger.debug(f"Orderbook check: Zero or negative total volume within top {num_levels_to_check} levels for {self.symbol}.")
                return 0.0 # Neutral if no volume

            # Calculate Order Book Imbalance (OBI) ratio: (Bids - Asks) / Total
            obi = (bid_volume_sum - ask_volume_sum) / total_volume

            # OBI naturally ranges from -1 (all asks) to +1 (all bids)
            score = float(obi)

            self.logger.debug(f"Orderbook check ({self.symbol}): Top {num_levels_to_check} Levels: "
                              f"BidVol={bid_volume_sum:.4f}, AskVol={ask_volume_sum:.4f}, "
                              f"OBI={obi:.4f}, Score={score:.4f}")
            return score

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed unexpectedly for {self.symbol}: {e}{RESET}", exc_info=True)
            return float('nan')

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on entry price, ATR, and configured multipliers. Uses Decimal precision
        and market tick size for quantization.

        Args:
            entry_price: The entry price (Decimal).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            Tuple (entry_price, take_profit, stop_loss), all as Decimal or None.
            SL/TP will be None if calculation fails or signal is "HOLD".
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None # No TP/SL needed for HOLD

        atr_val_float = self.indicator_values.get("ATR")
        if atr_val_float is None or math.isnan(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float)) # Convert valid float ATR to Decimal

            # Get multipliers from config, convert to Decimal safely
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision and tick size for quantization
            price_precision = self.get_price_precision()
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"{NEON_RED}Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).{RESET}")
                 return entry_price, None, None

            take_profit = None
            stop_loss = None
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            if signal == "BUY":
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Quantize TP UP, SL DOWN to the nearest tick size (away from entry)
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick

            elif signal == "SELL":
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Quantize TP DOWN, SL UP to the nearest tick size (away from entry)
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

            # --- Validation and Adjustment ---
            # Ensure SL is strictly beyond entry by at least one tick
            if signal == "BUY" and stop_loss >= entry_price:
                stop_loss = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation resulted in SL >= Entry. Adjusted SL down to {stop_loss}.{RESET}")
            elif signal == "SELL" and stop_loss <= entry_price:
                stop_loss = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation resulted in SL <= Entry. Adjusted SL up to {stop_loss}.{RESET}")

            # Ensure TP is strictly profitable relative to entry by at least one tick
            if signal == "BUY" and take_profit <= entry_price:
                take_profit = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}BUY signal TP calculation resulted in TP <= Entry. Adjusted TP up to {take_profit}.{RESET}")
            elif signal == "SELL" and take_profit >= entry_price:
                take_profit = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}SELL signal TP calculation resulted in TP >= Entry. Adjusted TP down to {take_profit}.{RESET}")

            # Final checks: Ensure SL/TP are positive prices
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in zero or negative price ({stop_loss}) for {self.symbol}. Setting SL to None.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}Take profit calculation resulted in zero or negative price ({take_profit}) for {self.symbol}. Setting TP to None.{RESET}")
                take_profit = None

            # --- Logging ---
            tp_log = f"{take_profit:.{price_precision}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{price_precision}f}" if stop_loss else 'N/A'
            atr_log = f"{atr:.{price_precision+1}f}" # Log ATR with more precision
            entry_log = f"{entry_price:.{price_precision}f}"
            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log} (based on ATR={atr_log})")

            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency, handling Bybit V5 structures and retries.

    Args:
        exchange: Initialized ccxt exchange object.
        currency: The currency code (e.g., "USDT").
        logger: The logger instance.

    Returns:
        The available balance as a Decimal, or None if fetching fails or balance is invalid.
    """
    lg = logger
    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            balance_info = None
            # Bybit V5: Try specific account types first
            account_types_to_try = ['CONTRACT', 'UNIFIED'] # Prioritize based on typical usage
            successful_acc_type = None # Track which type worked

            for acc_type in account_types_to_try:
                try:
                    lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency} (Attempt {attempt + 1})")
                    balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                    successful_acc_type = acc_type # Mark type used for successful call
                    # Check if the structure contains the currency directly or nested (V5)
                    if balance_info and (currency in balance_info or 'info' in balance_info):
                        lg.debug(f"Received balance structure using accountType '{acc_type}'.")
                        break # Found a potentially valid structure, proceed to parse
                    else:
                        lg.debug(f"Balance structure for accountType '{acc_type}' seems empty or missing currency. Trying next.")
                        balance_info = None # Reset to try next type
                except ccxt.ExchangeError as e:
                    # Ignore errors indicating the account type doesn't exist/support, try next
                    ignore_msgs = ["account type not support", "invalid account type"]
                    if any(msg in str(e).lower() for msg in ignore_msgs):
                        lg.debug(f"Account type '{acc_type}' not supported or error fetching: {e}. Trying next.")
                        continue
                    else: # Re-raise other exchange errors within this attempt
                        raise e
                # Let outer try-except handle NetworkError, RateLimitExceeded etc.

            # If specific account types failed, try default fetch_balance without params
            if not balance_info:
                lg.debug(f"Fetching balance using default parameters for {currency} (Attempt {attempt + 1})...")
                balance_info = exchange.fetch_balance()
                successful_acc_type = None # Mark as default fetch

            # --- Parse the balance_info ---
            if not balance_info:
                 lg.warning(f"Failed to fetch any balance information (Attempt {attempt + 1}).")
                 last_exception = ValueError("API returned no balance information")
                 time.sleep(RETRY_DELAY_SECONDS) # Wait before retry
                 continue # Go to next attempt

            available_balance_str = None
            # 1. Standard CCXT: balance_info[currency]['free']
            if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('free') is not None:
                available_balance_str = str(balance_info[currency]['free'])
                lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}")
            # 2. Bybit V5 Nested: info -> result -> list -> coin[]
            elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                balance_list = balance_info['info']['result']['list']
                for account in balance_list:
                    current_account_type = account.get('accountType')
                    # Match the successful account type OR check all if default fetch was used
                    if successful_acc_type is None or current_account_type == successful_acc_type:
                        coin_list = account.get('coin')
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get('coin') == currency:
                                    # Prefer Bybit V5 'availableToWithdraw' or 'availableBalance'
                                    free = coin_data.get('availableToWithdraw', coin_data.get('availableBalance'))
                                    # Fallback: 'walletBalance' (might include unrealized PnL)
                                    if free is None: free = coin_data.get('walletBalance')
                                    if free is not None:
                                        available_balance_str = str(free)
                                        lg.debug(f"Found balance via Bybit V5 nested structure: {available_balance_str} {currency} (Account: {current_account_type or 'Default'})")
                                        break # Found currency in this account
                            if available_balance_str is not None: break # Found currency
                if available_balance_str is None:
                    lg.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]' structure for relevant account type(s).")
            # 3. Fallback: Top-level 'free' dictionary
            elif 'free' in balance_info and isinstance(balance_info.get('free'), dict) and currency in balance_info['free']:
                available_balance_str = str(balance_info['free'][currency])
                lg.debug(f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency}")

            # 4. Last Resort: Check 'total' balance if 'free' still missing
            if available_balance_str is None:
                total_balance_str = None
                # Standard total
                if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('total') is not None:
                    total_balance_str = str(balance_info[currency]['total'])
                # Bybit V5 nested total ('walletBalance')
                elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                     balance_list = balance_info['info']['result']['list']
                     for account in balance_list:
                        current_account_type = account.get('accountType')
                        if successful_acc_type is None or current_account_type == successful_acc_type:
                            coin_list = account.get('coin')
                            if isinstance(coin_list, list):
                                for coin_data in coin_list:
                                    if coin_data.get('coin') == currency:
                                        total = coin_data.get('walletBalance')
                                        if total is not None:
                                            total_balance_str = str(total)
                                            lg.debug(f"Using 'walletBalance' ({total_balance_str}) from nested structure as 'total' fallback.")
                                            break
                                if total_balance_str is not None: break
                        if total_balance_str is not None: break

                if total_balance_str:
                    lg.warning(f"{NEON_YELLOW}Could not determine 'free'/'available' balance for {currency}. Using 'total' ({total_balance_str}) as fallback (may include collateral/unrealized PnL).{RESET}")
                    available_balance_str = total_balance_str
                else:
                    lg.error(f"{NEON_RED}Could not determine any balance for {currency}. Balance info structure not recognized or currency missing.{RESET}")
                    lg.debug(f"Full balance_info structure: {balance_info}")
                    return None # Critical failure if no balance found

            # --- Convert to Decimal ---
            try:
                final_balance = Decimal(available_balance_str)
                if final_balance >= 0: # Allow zero balance
                    lg.info(f"Available {currency} balance: {final_balance:.4f}")
                    return final_balance
                else:
                    lg.error(f"{NEON_RED}Parsed balance for {currency} is negative ({final_balance}). Returning None.{RESET}")
                    return None
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.error(f"{NEON_RED}Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}.{RESET}")
                return None # Conversion failed

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(f"Rate limit hit fetching balance for {currency}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"Network error fetching balance for {currency}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})")
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.AuthenticationError as e:
            lg.error(f"{NEON_RED}Authentication error fetching balance: {e}. Check API key permissions. Not retrying.{RESET}")
            raise e # Re-raise auth error as it's critical
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching balance: {e}. Not retrying.{RESET}")
            last_exception = e
            # Treat exchange errors during balance fetch as potentially critical, don't retry by default
            # unless specific error codes are known to be transient.
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}. Not retrying.{RESET}", exc_info=True)
            last_exception = e
            return None # Don't retry on unexpected errors

    lg.error(f"{NEON_RED}Max retries reached fetching balance for {currency}. Last error: {last_exception}{RESET}")
    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Gets market information like precision, limits, contract type, ensuring markets are loaded.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: The logger instance.

    Returns:
        The market dictionary from ccxt, enhanced with 'is_contract' flag, or None on failure.
    """
    lg = logger
    try:
        # Ensure markets are loaded; reload if symbol is missing or markets empty
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded or symbol missing, reloading markets...")
            exchange.load_markets(reload=True) # Force reload

        # Check again after reloading
        if not exchange.markets or symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Market {symbol} still not found after reloading markets. Check symbol spelling and availability on {exchange.id}.{RESET}")
            # Log available markets for debugging if needed (can be very long)
            # lg.debug(f"Available markets: {list(exchange.markets.keys())}")
            return None

        market = exchange.market(symbol)
        if market:
            # --- Enhance market dictionary ---
            market_type = market.get('type', 'unknown') # spot, swap, future etc.
            # Determine if it's a contract market
            market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']
            # Add contract type string for clarity
            market['contract_type'] = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "Spot/Unknown"
            # Ensure market ID is present (used in API calls)
            if 'id' not in market or not market['id']:
                market['id'] = market.get('info', {}).get('symbol', symbol.replace('/', '').split(':')[0]) # Fallback ID construction
                lg.debug(f"Market ID constructed as '{market['id']}' for {symbol}.")

            # --- Log Key Details ---
            precision = market.get('precision', {})
            limits = market.get('limits', {})
            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract={market['is_contract']} ({market['contract_type']}), "
                f"Precision(Price/Amount/Tick): {precision.get('price')}/{precision.get('amount')}/{precision.get('tick', 'N/A')}, "
                f"Limits(Amount Min/Max): {limits.get('amount', {}).get('min')}/{limits.get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {limits.get('cost', {}).get('min')}/{limits.get('cost', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            return market
        else:
            # Should have been caught by the 'in exchange.markets' check, but safeguard
            lg.error(f"{NEON_RED}Market dictionary not found for {symbol} even after checking exchange.markets.{RESET}")
            return None

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Symbol '{symbol}' is not supported by {exchange.id} or is incorrectly formatted: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error loading market info for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error loading market info for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
    return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal, # Calculated initial SL price (Decimal)
    entry_price: Decimal, # Estimated entry price (Decimal)
    market_info: Dict,
    exchange: ccxt.Exchange, # Pass exchange object for formatting helpers
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size based on risk percentage, initial SL distance, balance,
    and market constraints (min/max size, precision, contract size, cost).

    Args:
        balance: Available quote currency balance (Decimal).
        risk_per_trade: Risk fraction per trade (e.g., 0.01 for 1%).
        initial_stop_loss_price: Calculated initial SL price (Decimal).
        entry_price: Estimated entry price (Decimal).
        market_info: Market dictionary from ccxt.
        exchange: Initialized ccxt exchange object.
        logger: Logger instance.

    Returns:
        Calculated and validated position size as Decimal, or None if calculation fails.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('linear', not market_info.get('inverse', False)) # Assume linear if not inverse
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if not market_info or 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Position sizing failed ({symbol}): Missing or incomplete market_info.")
        return None
    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid or zero balance ({balance} {quote_currency}).")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
        return None
    if initial_stop_loss_price is None or entry_price is None or entry_price <= 0:
        lg.error(f"Position sizing failed ({symbol}): Missing or invalid entry_price ({entry_price}) or initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed ({symbol}): Stop loss price cannot equal entry price.")
        return None
    # Allow SL <= 0 if calculated that way, but check distance later
    if initial_stop_loss_price <= 0:
        lg.warning(f"Position sizing ({symbol}): Initial SL price ({initial_stop_loss_price}) is zero or negative. Ensure calculation is correct.")

    try:
        # --- Calculate Risk Amount and Initial Size ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)

        if sl_distance_per_unit <= 0:
            lg.error(f"Position sizing failed ({symbol}): Stop loss distance is zero or negative ({sl_distance_per_unit}). Entry={entry_price}, SL={initial_stop_loss_price}")
            return None

        # Get contract size (value of 1 contract in base currency for linear, quote for inverse - check docs!)
        # Bybit Linear: contractSize = 1 (usually means 1 base unit, e.g., 1 BTC)
        # Bybit Inverse: contractSize = 1 (usually means 1 quote unit, e.g., 1 USD)
        contract_size_str = market_info.get('contractSize', '1')
        contract_size = Decimal(str(contract_size_str)) if contract_size_str is not None else Decimal('1')
        if contract_size <= 0:
            lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}. Defaulting to 1.")
            contract_size = Decimal('1')

        # --- Calculate Size based on Market Type ---
        calculated_size = Decimal('0')
        if is_linear or not is_contract: # Spot or Linear Contract
            # Size (Base/Contracts) = Risk Amount (Quote) / (SL Distance (Quote/Base) * Contract Size (Base/Contract))
            risk_per_contract_quote = sl_distance_per_unit * contract_size
            if risk_per_contract_quote <= 0:
                lg.error(f"Position sizing failed ({symbol}): Risk per contract is zero or negative ({risk_per_contract_quote}).")
                return None
            calculated_size = risk_amount_quote / risk_per_contract_quote
            lg.debug(f"  Linear/Spot Sizing: RiskAmt={risk_amount_quote:.4f} / (SLDist={sl_distance_per_unit} * ContSize={contract_size}) = {calculated_size}")
        else: # Inverse Contract
            lg.debug(f"{NEON_YELLOW}Inverse contract sizing for {symbol}. Contract Size ({contract_size}) assumed Quote value/contract.{RESET}")
            # Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
            # Risk per Contract (Quote) = SL Distance (Quote/Base) * Value of 1 Contract (Base)
            # Value of 1 Contract (Base) = Contract Size (Quote/Contract) / Entry Price (Quote/Base)
            contract_value_quote = contract_size # Assumed value like 1 USD
            if entry_price <= 0:
                 lg.error(f"Position sizing failed ({symbol}): Entry price is zero or negative for inverse calculation.")
                 return None
            value_of_1_contract_base = contract_value_quote / entry_price
            risk_per_contract_quote = sl_distance_per_unit * value_of_1_contract_base
            if risk_per_contract_quote <= 0:
                 lg.error(f"Position sizing failed ({symbol}): Inverse risk per contract is zero or negative ({risk_per_contract_quote}).")
                 return None
            calculated_size = risk_amount_quote / risk_per_contract_quote
            lg.debug(f"  Inverse Sizing: Val1ContBase={value_of_1_contract_base:.8f}, RiskPerContQuote={risk_per_contract_quote:.4f}")
            lg.debug(f"  Inverse Sizing: RiskAmt={risk_amount_quote:.4f} / RiskPerContQuote={risk_per_contract_quote} = {calculated_size}")

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, Risk Amount={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Distance={sl_distance_per_unit}")
        lg.info(f"  Contract Size={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount') # Can be int (decimals) or float (step size)

        # Helper to safely get Decimal limit or default
        def get_limit(limit_dict: dict, key: str, default: Decimal) -> Decimal:
            val_str = limit_dict.get(key)
            try: return Decimal(str(val_str)) if val_str is not None else default
            except (InvalidOperation, ValueError, TypeError): return default

        min_amount = get_limit(amount_limits, 'min', Decimal('0'))
        max_amount = get_limit(amount_limits, 'max', Decimal('inf'))
        min_cost = get_limit(cost_limits, 'min', Decimal('0'))
        max_cost = get_limit(cost_limits, 'max', Decimal('inf'))

        # 1. Adjust size based on MIN/MAX AMOUNT limits
        adjusted_size = calculated_size
        if adjusted_size < min_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} is below min amount {min_amount:.8f}. Adjusting to minimum.{RESET}")
            adjusted_size = min_amount
        elif adjusted_size > max_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} exceeds max amount {max_amount:.8f}. Capping at maximum.{RESET}")
            adjusted_size = max_amount

        # 2. Check COST limits with the amount-adjusted size
        current_cost = Decimal('0')
        if is_linear or not is_contract: # Spot or Linear
            # Cost = Size (Base/Contracts) * Entry Price (Quote/Base) * Contract Size (Base/Contract)
            current_cost = adjusted_size * entry_price * contract_size
        else: # Inverse
            # Cost = Size (Contracts) * Contract Size (Quote/Contract)
            current_cost = adjusted_size * contract_size # Assuming contract_size is Quote value
        lg.debug(f"  Cost Check: Amount-Adjusted Size={adjusted_size:.8f}, Estimated Cost={current_cost:.4f} {quote_currency} (Limits: Min={min_cost}, Max={max_cost})")

        cost_limit_adjusted = False
        if min_cost > 0 and current_cost < min_cost:
            # Allow very small tolerance for floating point issues
            if not math.isclose(float(current_cost), float(min_cost), rel_tol=1e-6):
                lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} is below min cost {min_cost:.4f}. Attempting to increase size.{RESET}")
                required_size_for_min_cost = Decimal('0')
                try:
                    if is_linear or not is_contract:
                        if entry_price > 0 and contract_size > 0:
                            required_size_for_min_cost = min_cost / (entry_price * contract_size)
                        else: raise ValueError("Invalid entry price or contract size for linear min cost calc")
                    else: # Inverse
                        if contract_size > 0:
                            required_size_for_min_cost = min_cost / contract_size # Assuming contract_size is Quote value
                        else: raise ValueError("Invalid contract size for inverse min cost calc")

                    lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
                    # Check feasibility
                    if required_size_for_min_cost > max_amount:
                        lg.error(f"{NEON_RED}Cannot meet min cost {min_cost:.4f} without exceeding max amount {max_amount:.8f}. Trade aborted.{RESET}")
                        return None
                    elif required_size_for_min_cost < min_amount:
                         lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f}, but min amount is {min_amount:.8f}. Trade aborted.{RESET}")
                         return None
                    else:
                         lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f}")
                         adjusted_size = required_size_for_min_cost
                         cost_limit_adjusted = True
                except (ValueError, InvalidOperation, ZeroDivisionError) as calc_err:
                     lg.error(f"{NEON_RED}Error calculating required size for min cost: {calc_err}. Trade aborted.{RESET}")
                     return None
            else:
                 lg.debug(f"Estimated cost {current_cost:.4f} is very close to min cost {min_cost:.4f}. Proceeding.")

        elif max_cost > 0 and current_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} exceeds max cost {max_cost:.4f}. Reducing size.{RESET}")
            adjusted_size_for_max_cost = Decimal('0')
            try:
                if is_linear or not is_contract:
                    if entry_price > 0 and contract_size > 0:
                        adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                    else: raise ValueError("Invalid entry price or contract size for linear max cost calc")
                else: # Inverse
                    if contract_size > 0:
                        adjusted_size_for_max_cost = max_cost / contract_size # Assuming contract_size is Quote value
                    else: raise ValueError("Invalid contract size for inverse max cost calc")

                lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f}")
                # Check feasibility
                if adjusted_size_for_max_cost < min_amount:
                    lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is now below min amount {min_amount:.8f}. Cannot meet limits. Trade aborted.{RESET}")
                    return None
                else:
                    adjusted_size = adjusted_size_for_max_cost
                    cost_limit_adjusted = True
            except (ValueError, InvalidOperation, ZeroDivisionError) as calc_err:
                lg.error(f"{NEON_RED}Error calculating size reduction for max cost: {calc_err}. Trade aborted.{RESET}")
                return None

        # 3. Apply Amount Precision/Step Size (rounding DOWN using ccxt helper)
        final_size = Decimal('0')
        try:
            # Use exchange.amount_to_precision with TRUNCATE (floor/round down for positive numbers)
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except (ccxt.BaseError, ValueError, TypeError) as fmt_err:
             lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision for {symbol} ({fmt_err}). Using manual rounding (ROUND_DOWN).{RESET}")
             final_size = _manual_amount_rounding(adjusted_size, amount_precision_val, lg, symbol)
        except Exception as fmt_err: # Catch any other unexpected formatting error
            lg.error(f"{NEON_RED}Unexpected error during amount formatting for {symbol}: {fmt_err}. Using manual rounding (ROUND_DOWN).{RESET}", exc_info=True)
            final_size = _manual_amount_rounding(adjusted_size, amount_precision_val, lg, symbol)


        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{NEON_RED}Position size became zero or negative ({final_size}) after adjustments/rounding for {symbol}. Trade aborted.{RESET}")
            return None

        # Final check against min amount AFTER formatting/rounding (with tolerance)
        if final_size < min_amount and not math.isclose(float(final_size), float(min_amount), rel_tol=1e-9):
             lg.error(f"{NEON_RED}Final formatted size {final_size} {size_unit} is below minimum amount {min_amount} {size_unit}. Trade aborted.{RESET}")
             return None

        # Final check against min cost AFTER formatting/rounding (only if cost limits were not the reason for adjustment)
        if not cost_limit_adjusted and min_cost > 0:
            final_cost = Decimal('0')
            if is_linear or not is_contract: final_cost = final_size * entry_price * contract_size
            else: final_cost = final_size * contract_size # Inverse

            if final_cost < min_cost and not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6):
                 lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} below min cost {min_cost:.4f} (likely due to rounding). Trade aborted.{RESET}")
                 return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(f"{NEON_RED}Position sizing error ({symbol}): Missing market info key {e}. Market: {market_info}{RESET}")
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Decimal/Type error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
    return None

def _manual_amount_rounding(size: Decimal, precision_val: Any, lg: logging.Logger, symbol: str) -> Decimal:
    """Manual fallback for rounding amount DOWN based on precision (step or decimals)."""
    final_size = size # Default if precision invalid
    step_size = None
    num_decimals = None

    if isinstance(precision_val, int) and precision_val >= 0:
        num_decimals = precision_val
    elif isinstance(precision_val, (float, str)): # Assume step size
        try:
            step_size = Decimal(str(precision_val))
            if step_size <= 0: step_size = None
        except (InvalidOperation, ValueError, TypeError): pass

    if step_size is not None:
        final_size = (size // step_size) * step_size # Floor division then multiply
        lg.info(f"Applied manual amount step size ({step_size}), rounded down: {size:.8f} -> {final_size}")
    elif num_decimals is not None:
        rounding_factor = Decimal('1e-' + str(num_decimals))
        final_size = size.quantize(rounding_factor, rounding=ROUND_DOWN)
        lg.info(f"Applied manual amount precision ({num_decimals} decimals), rounded down: {size:.8f} -> {final_size}")
    else:
        lg.warning(f"{NEON_YELLOW}Amount precision value ('{precision_val}') invalid for manual rounding ({symbol}). Using unrounded size: {size:.8f}{RESET}")

    return final_size


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using fetch_positions.
    Returns the unified position dictionary from CCXT if an active position exists,
    enhanced with Decimal SL/TP/TSL info, otherwise None.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: The logger instance.

    Returns:
        Enhanced position dictionary or None.
    """
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        market = None # Fetch market info for context and params

        # Prepare params for Bybit V5 fetch_positions
        params = {}
        market_id = symbol # Default if market lookup fails
        if 'bybit' in exchange.id.lower():
            try:
                # Get market info to determine category and correct ID
                market = get_market_info(exchange, symbol, lg)
                if not market: raise ValueError("Failed to get market info for position fetch")
                market_id = market['id'] # Use exchange ID (e.g., BTCUSDT)
                category = 'linear' if market.get('linear', True) else 'inverse'
                params['category'] = category
                params['symbol'] = market_id # V5 requires symbol for single fetch
                lg.debug(f"Using params for fetch_positions: {params}")
            except Exception as e:
                lg.warning(f"Error getting market info for position fetch ({symbol}): {e}. Using defaults.")
                # Attempt fallback category guess
                category = 'linear' if 'USDT' in symbol else 'inverse'
                params['category'] = category
                params['symbol'] = symbol.replace('/', '').split(':')[0] # Guess market ID
                market_id = params['symbol'] # Update market_id for filtering later

        # --- Fetch Positions ---
        try:
            # Bybit V5: fetch_positions with symbol param fetches that specific symbol
            positions = exchange.fetch_positions(symbols=None, params=params) # Rely on params for V5

        except ccxt.ArgumentsRequired:
            # Fallback if exchange requires fetching all (less common now)
            lg.warning(f"Exchange {exchange.id} requires fetching all positions, then filtering. Fetching all...")
            all_positions = exchange.fetch_positions()
            # Filter by the correct symbol field ('symbol' in ccxt standard)
            positions = [p for p in all_positions if p.get('symbol') == symbol]
        except ccxt.ExchangeError as e:
            # Handle specific "no position" errors gracefully
            no_pos_codes_v5 = [110021] # Bybit V5: Position is closed or does not exist
            no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
            err_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            if err_code in no_pos_codes_v5 or any(msg in err_str for msg in no_pos_msgs):
                lg.info(f"No position found for {symbol} (Exchange confirmation: {e}).")
                return None
            # Re-raise other exchange errors
            lg.error(f"Exchange error fetching position for {symbol}: {e}", exc_info=True)
            return None # Treat as failure if error occurs here
        # Let outer try-except handle NetworkError etc.

        # --- Process Fetched Positions ---
        active_position = None
        if not positions:
            lg.info(f"No position entries returned for {symbol}.")
            return None

        # Find the first entry with a meaningful non-zero size
        # Determine a sensible threshold slightly above zero noise
        min_size_threshold = Decimal('1e-9') # Default very small threshold
        if market:
            try:
                 min_amt = market.get('limits', {}).get('amount', {}).get('min')
                 if min_amt is not None:
                     # Use a fraction of min size, or a small absolute value
                     min_size_threshold = max(Decimal('1e-9'), Decimal(str(min_amt)) * Decimal('0.01'))
            except Exception: pass # Ignore errors fetching min size

        for pos in positions:
            pos_size_str = None
            # Prefer standard 'contracts', fallback to Bybit V5 'size', others
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size'])
            elif pos.get('contractSize') is not None: pos_size_str = str(pos['contractSize']) # Less common standard field
            # Add other potential fields if needed for specific exchanges

            if pos_size_str is None:
                lg.debug(f"Skipping position entry for {symbol}: Could not find size field. Entry: {pos}")
                continue

            try:
                position_size = Decimal(pos_size_str)
                # Check absolute size against threshold
                if abs(position_size) > min_size_threshold:
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break # Use the first active entry found
                else:
                    lg.debug(f"Position entry size {position_size} <= threshold {min_size_threshold}. Skipping.")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse position size '{pos_size_str}' as Decimal for {symbol}. Skipping entry. Error: {e}")
                continue

        # --- Post-Process the Found Active Position ---
        if not active_position:
            lg.info(f"No active open position found for {symbol} (checked {len(positions)} entries).")
            return None

        # --- Determine Side (Crucial) ---
        pos_side = active_position.get('side') # Standard 'long' or 'short'
        info_side = active_position.get('info', {}).get('side', 'None') # Bybit V5: 'Buy', 'Sell', or 'None'
        size_decimal = Decimal('0') # For inferring side if needed
        try:
            size_str_for_side = active_position.get('contracts', active_position.get('info',{}).get('size', '0'))
            size_decimal = Decimal(str(size_str_for_side))
        except (InvalidOperation, ValueError, TypeError): pass

        if pos_side not in ['long', 'short']:
            if info_side == 'Buy': pos_side = 'long'
            elif info_side == 'Sell': pos_side = 'short'
            # Infer from size sign as last resort (less reliable for V5 where size is usually positive)
            elif size_decimal > min_size_threshold: pos_side = 'long'
            elif size_decimal < -min_size_threshold: pos_side = 'short'
            else:
                lg.warning(f"Could not reliably determine side for position {symbol} (size={size_decimal}). Treating as no position.")
                return None
            active_position['side'] = pos_side # Add inferred side
            lg.debug(f"Inferred position side as '{pos_side}' for {symbol}.")

        # Ensure 'contracts' field holds absolute size (ccxt standard)
        if active_position.get('contracts') is not None:
             try:
                  current_contracts = Decimal(str(active_position['contracts']))
                  active_position['contracts'] = abs(current_contracts)
             except (InvalidOperation, ValueError, TypeError): pass # Ignore conversion fails


        # --- Enhance with SL/TP/TSL info (Decimal) ---
        info_dict = active_position.get('info', {})

        def get_decimal_from_pos(key_standard: str, key_info: str) -> Optional[Decimal]:
            """Helper to get Decimal value from standard or info dict, checking > 0."""
            val_str = active_position.get(key_standard, info_dict.get(key_info))
            if val_str is not None and str(val_str).strip() not in ['', '0', '0.0']: # Check not empty or zero string
                try:
                    d_val = Decimal(str(val_str).strip())
                    if d_val > 0: return d_val
                except (InvalidOperation, ValueError, TypeError): pass
            return None

        active_position['stopLossPriceDecimal'] = get_decimal_from_pos('stopLossPrice', 'stopLoss')
        active_position['takeProfitPriceDecimal'] = get_decimal_from_pos('takeProfitPrice', 'takeProfit')

        # TSL specific fields (distance/value and activation price) - Can be '0'
        tsl_dist_str = info_dict.get('trailingStop', '0')
        tsl_act_str = info_dict.get('activePrice', '0')
        active_position['trailingStopLossDistance'] = tsl_dist_str # Store raw string
        active_position['tslActivationPrice'] = tsl_act_str # Store raw string
        try: active_position['trailingStopLossDistanceDecimal'] = Decimal(tsl_dist_str) if tsl_dist_str else Decimal('0')
        except (InvalidOperation, ValueError, TypeError): active_position['trailingStopLossDistanceDecimal'] = Decimal('0')
        try: active_position['tslActivationPriceDecimal'] = Decimal(tsl_act_str) if tsl_act_str else Decimal('0')
        except (InvalidOperation, ValueError, TypeError): active_position['tslActivationPriceDecimal'] = Decimal('0')

        # Ensure Decimal values are non-negative
        if active_position['trailingStopLossDistanceDecimal'] < 0: active_position['trailingStopLossDistanceDecimal'] = Decimal('0')
        if active_position['tslActivationPriceDecimal'] < 0: active_position['tslActivationPriceDecimal'] = Decimal('0')


        # --- Log Position Details ---
        log_precision = 8 # Default precision
        amount_precision = 8
        if market: # Use market info if available
            try:
                 # Need a dummy analyzer to use helper methods
                 dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                 dummy_df.index.name = 'timestamp'
                 analyzer_temp = TradingAnalyzer(dummy_df, lg, {}, market) # Minimal config needed
                 log_precision = analyzer_temp.get_price_precision()
                 # Get amount precision (decimals from step size)
                 amount_prec_val = market.get('precision', {}).get('amount')
                 if amount_prec_val is not None:
                     if isinstance(amount_prec_val, int): amount_precision = amount_prec_val
                     else: amount_precision = abs(Decimal(str(amount_prec_val)).normalize().as_tuple().exponent)
            except Exception: pass # Ignore errors getting precision

        # Helper to format log values safely
        def format_log(val: Any, prec: int, allow_zero: bool = False) -> str:
            if val is None: return 'N/A'
            try:
                d_val = Decimal(str(val))
                if d_val > 0: return f"{d_val:.{prec}f}"
                elif d_val == 0 and allow_zero: return '0.0' # Show inactive TSL as 0.0
                else: return 'N/A' # Treat 0 as N/A otherwise
            except (InvalidOperation, ValueError, TypeError): return str(val) # Raw on error

        entry_price = format_log(active_position.get('entryPrice', info_dict.get('avgPrice')), log_precision)
        contracts = format_log(active_position.get('contracts', info_dict.get('size')), amount_precision)
        liq_price = format_log(active_position.get('liquidationPrice'), log_precision)
        leverage_str = active_position.get('leverage', info_dict.get('leverage'))
        leverage = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str else 'N/A'
        pnl = format_log(active_position.get('unrealizedPnl'), 4) # PnL needs fewer decimals
        sl_price = format_log(active_position.get('stopLossPriceDecimal'), log_precision)
        tp_price = format_log(active_position.get('takeProfitPriceDecimal'), log_precision)
        tsl_active = active_position.get('trailingStopLossDistanceDecimal', Decimal(0)) > 0
        tsl_dist = format_log(active_position.get('trailingStopLossDistanceDecimal'), log_precision, allow_zero=True)
        tsl_act = format_log(active_position.get('tslActivationPriceDecimal'), log_precision, allow_zero=True)

        logger.info(f"{NEON_GREEN}Active {pos_side.upper()} position found for {symbol}:{RESET} "
                    f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                    f"Leverage={leverage}, PnL={pnl}, SL={sl_price}, TP={tp_price}, "
                    f"TSL Active: {tsl_active} (Dist={tsl_dist}/Act={tsl_act})")
        logger.debug(f"Full position details for {symbol}: {active_position}")
        return active_position

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching positions for {symbol}: {e}{RESET}")
        # Depending on strategy, may need to raise or handle differently
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching positions for {symbol}: {e}{RESET}")
        # Position status is critical, maybe retry here or raise
    except ccxt.ExchangeError as e:
        # Catch potential exchange errors not handled during fetch
        lg.error(f"{NEON_RED}Unhandled Exchange error processing positions for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if any error occurs during the process


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """
    Sets leverage for a symbol using CCXT, handling Bybit V5 specifics and verification.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol.
        leverage: The desired leverage integer (e.g., 10 for 10x).
        market_info: Market dictionary from ccxt.
        logger: Logger instance.

    Returns:
        True if leverage setting was successful or confirmed unnecessary, False otherwise.
    """
    lg = logger
    if not market_info:
        lg.error(f"Leverage setting failed ({symbol}): Missing market_info.")
        return False
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped ({symbol}): Not a contract market.")
        return True # Success (not applicable)
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped ({symbol}): Invalid leverage value ({leverage}). Must be > 0.")
        return False

    # --- Check exchange capability (often unreliable, proceed with caution) ---
    # if not exchange.has.get('setLeverage'):
    #     lg.warning(f"{NEON_YELLOW}Exchange {exchange.id} does not report set_leverage capability. Attempting anyway...{RESET}")

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")

        # --- Prepare Bybit V5 specific parameters ---
        params = {}
        market_id = market_info.get('id')
        if not market_id:
             lg.error(f"Leverage setting failed ({symbol}): Missing market ID.")
             return False

        if 'bybit' in exchange.id.lower():
            # Bybit V5 requires buyLeverage and sellLeverage, and category
            category = 'linear' if market_info.get('linear', True) else 'inverse'
            params = {
                'buyLeverage': str(leverage), # String format for Bybit API
                'sellLeverage': str(leverage),
                'category': category,
                # 'symbol': market_id # Symbol is passed as main argument to set_leverage
            }
            lg.debug(f"Using Bybit V5 specific params for set_leverage: {params}")

        # --- Call set_leverage ---
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # --- Verification ---
        # Success is often indicated by lack of exception and/or specific response codes.
        verified = False
        if response is not None and isinstance(response, dict):
             ret_code = response.get('retCode', response.get('info', {}).get('retCode'))
             if ret_code == 0:
                 lg.debug(f"Set leverage call for {symbol} confirmed success (retCode 0).")
                 verified = True
             elif ret_code == 110045: # Leverage not modified (already set correctly)
                 lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation: Code {ret_code}).{RESET}")
                 verified = True
             elif ret_code is not None: # Other non-zero retCode
                  ret_msg = response.get('retMsg', response.get('info', {}).get('retMsg', 'Unknown Error'))
                  lg.warning(f"Set leverage call for {symbol} returned non-zero retCode {ret_code} ({ret_msg}). Treating as failure.")
             else: # No retCode found, rely on lack of exception
                  lg.debug("Set leverage call returned response without retCode. Assuming success based on no error.")
                  verified = True
        elif response is None:
             # Bybit V5 via ccxt might return None on success sometimes. Assume OK if no exception.
             lg.debug("Set leverage call returned None. Assuming success based on no error.")
             verified = True
        else: # Response is not dict or None
             lg.debug(f"Set leverage call returned unexpected response type ({type(response)}). Assuming success based on no error.")
             verified = True


        if verified:
            lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/confirmed at {leverage}x.{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Leverage setting failed for {symbol} based on response analysis.{RESET}")
            return False

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
        # Handle common Bybit errors with hints
        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True # Treat as success
        elif bybit_code in [110028, 110009] or "margin mode" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Ensure Margin Mode (Isolated/Cross) is correct for {symbol} *before* setting leverage. Check Bybit settings.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x might exceed the risk limit tier. Check Bybit Risk Limit docs.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for {symbol}. Check allowed range on Bybit.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal, # Size calculated previously (Decimal)
    market_info: Dict,
    logger: Optional[logging.Logger] = None,
    reduce_only: bool = False # Flag for closing trades
) -> Optional[Dict]:
    """
    Places a market order using CCXT for opening or closing positions.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol.
        trade_signal: "BUY" or "SELL".
        position_size: The absolute size of the order (Decimal).
        market_info: Market dictionary from ccxt.
        logger: Logger instance.
        reduce_only: Set to True for closing orders.

    Returns:
        The order dictionary from ccxt on success, None on failure.
    """
    lg = logger or logging.getLogger(__name__)
    if not market_info:
        lg.error(f"Trade aborted ({symbol}): Missing market_info.")
        return None

    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')

    # Convert Decimal size to float for ccxt amount parameter, ensure positive
    try:
        amount_float = float(abs(position_size))
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} reduce={reduce_only}): Invalid position size ({position_size}). Must be positive.")
            return None
    except (ValueError, TypeError) as e:
        lg.error(f"Trade aborted ({symbol} {side} reduce={reduce_only}): Failed to convert position size {position_size} to float: {e}")
        return None

    # --- Prepare Order Parameters ---
    params = {
        # Bybit V5: Specify position index (0 for One-Way mode)
        'positionIdx': 0,
        'reduceOnly': reduce_only,
        # Optional: Add timeInForce if needed (e.g., 'IOC' or 'FOK' for market)
        # 'timeInForce': 'IOC',
    }
    market_id = market_info.get('id')
    if not market_id:
        lg.error(f"Trade aborted ({symbol}): Missing market ID.")
        return None

    if 'bybit' in exchange.id.lower():
        category = 'linear' if market_info.get('linear', True) else 'inverse'
        params['category'] = category

    action = "Closing" if reduce_only else "Opening"
    lg.info(f"Attempting to place {side.upper()} {order_type} order ({action}) for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {size_unit} | Params: {params}")

    try:
        # --- Execute Market Order ---
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None, # Market order doesn't need price
            params=params
        )

        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A') # Market orders might be 'closed' or 'open' initially
        lg.info(f"{NEON_GREEN}Trade Order Placed Successfully ({action})! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} reduce={reduce_only}): {order}")

        # IMPORTANT: Caller must verify the resulting position state after a delay.
        return order # Return the order dictionary

    # --- Error Handling ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order ({action}) for {symbol}: {e}{RESET}")
        # Log balance for context if possible
        try:
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            lg.error(f"  Available {QUOTE_CURRENCY} balance: {balance if balance is not None else 'Fetch Failed'}")
        except Exception as bal_err: lg.error(f"  Could not fetch balance for context: {bal_err}")
        # Add Bybit V5 specific hints
        if getattr(e, 'code', None) == 110007: lg.error(f"{NEON_YELLOW} >> Hint (110007): Check available margin, leverage. Cost ~ Size * Price / Leverage.{RESET}")

    except ccxt.InvalidOrder as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Invalid order parameters for {symbol} ({action}): {e} (Code: {bybit_code}){RESET}")
        lg.error(f"  Order Details: Size={amount_float}, Type={order_type}, Side={side}, Params={params}")
        lg.error(f"  Market Limits: Amount={market_info.get('limits',{}).get('amount')}, Cost={market_info.get('limits',{}).get('cost')}")
        lg.error(f"  Market Precision: Amount={market_info.get('precision',{}).get('amount')}, Price={market_info.get('precision',{}).get('price')}")
        # Add Bybit V5 specific hints
        if bybit_code in [10001, 110013] and "parameter" in str(e).lower(): lg.error(f"{NEON_YELLOW} >> Hint ({bybit_code}): Check size/price vs precision/limits.{RESET}")
        elif bybit_code == 110017: lg.error(f"{NEON_YELLOW} >> Hint (110017): Order size {amount_float} violates min/max quantity per order.{RESET}")
        elif bybit_code == 110040: lg.error(f"{NEON_YELLOW} >> Hint (110040): Order size {amount_float} is below minimum. Check size calculation/limits.{RESET}")
        elif bybit_code == 110014 and reduce_only: lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only close failed. Size ({amount_float}) > open position? Position closed? API issue?{RESET}")

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing order ({action}) for {symbol}: {e}{RESET}")
        # Network errors during order placement are risky. Assume failure.
        # Consider adding state management to check order status later if this happens.

    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing order ({action}) for {symbol}: {e} (Code: {bybit_code}){RESET}")
        # Add Bybit V5 specific hints
        if bybit_code == 110007: lg.error(f"{NEON_YELLOW} >> Hint (110007): Insufficient balance/margin. Check balance, leverage. Cost ~ Size*Price/Leverage.{RESET}")
        elif bybit_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint (110043): Order cost exceeds available balance or risk limits.{RESET}")
        elif bybit_code == 110044: lg.error(f"{NEON_YELLOW} >> Hint (110044): Position size would exceed risk limit tier. Check Bybit risk limits.{RESET}")
        elif bybit_code == 110055: lg.error(f"{NEON_YELLOW} >> Hint (110055): Mismatch 'positionIdx' ({params.get('positionIdx')}) and account's Position Mode (One-Way vs Hedge).{RESET}")
        elif bybit_code == 10005 or "order link id exists" in str(e).lower(): lg.warning(f"{NEON_YELLOW}Duplicate order ID detected (Code {bybit_code}). Order might already exist. Check manually!{RESET}")
        elif bybit_code == 110025 and reduce_only: # Position not found when trying to close
             lg.warning(f"{NEON_YELLOW} >> Hint (110025): Position not found when attempting to close (reduceOnly=True). Already closed?{RESET}")
             # Treat this potentially as success (position is closed)
             return {'id': 'N/A', 'status': 'closed', 'info': {'reason': 'Position not found on close attempt'}}

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing order ({action}) for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if order failed


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Union[Decimal, str]] = None, # Allow '0' for cancel
    take_profit_price: Optional[Union[Decimal, str]] = None, # Allow '0' for cancel
    trailing_stop_distance: Optional[Union[Decimal, str]] = None, # Price distance (Decimal or '0')
    tsl_activation_price: Optional[Union[Decimal, str]] = None, # Price (Decimal or '0')
) -> bool:
    """
    Internal helper to set SL, TP, or TSL for an existing position via Bybit's V5 API.
    Uses the single `/v5/position/set-trading-stop` endpoint.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol.
        market_info: Market dictionary.
        position_info: Confirmed position dictionary from get_open_position.
        logger: Logger instance.
        stop_loss_price: Fixed SL price (Decimal) or '0' to cancel.
        take_profit_price: Fixed TP price (Decimal) or '0' to cancel.
        trailing_stop_distance: TSL distance in price points (Decimal) or '0' to cancel.
        tsl_activation_price: Price to activate TSL (Decimal) or '0' for immediate.

    Returns:
        True on successful API call (or if no change needed), False on failure.
    """
    lg = logger
    if not market_info or not position_info:
        lg.error(f"Cannot set protection ({symbol}): Missing market or position info.")
        return False
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped ({symbol}): Not a contract market.")
        return True # Not applicable

    # --- Validate Inputs and Determine Action ---
    pos_side = position_info.get('side')
    if pos_side not in ['long', 'short']:
        lg.error(f"Cannot set protection ({symbol}): Invalid position side ('{pos_side}').")
        return False

    # Determine position index (Default to 0 for One-Way)
    position_idx = 0
    try:
        # Bybit V5 uses positionIdx in info or potentially top level id if unified
        pos_idx_val = position_info.get('id', position_info.get('info', {}).get('positionIdx'))
        if pos_idx_val is not None:
             pos_idx_int = int(str(pos_idx_val))
             if pos_idx_int in [0, 1, 2]: position_idx = pos_idx_int # Bybit uses 0, 1, 2
             else: lg.warning(f"Invalid positionIdx ({pos_idx_int}). Defaulting to 0.")
        else: lg.debug(f"PositionIdx not found, defaulting to 0 (One-Way) for {symbol}.")
    except (ValueError, TypeError) as e:
        lg.warning(f"Could not parse positionIdx ({symbol}): {e}. Defaulting to {position_idx}.")

    # Helper to check if a value is a valid Decimal >= 0 or the cancel string '0'
    def is_valid_param(val: Optional[Union[Decimal, str]]) -> bool:
        if val is None: return False
        if isinstance(val, str) and val == '0': return True
        if isinstance(val, Decimal): return val >= 0
        return False

    # Check intent vs validity
    has_sl_intent = stop_loss_price is not None
    has_tp_intent = take_profit_price is not None
    has_tsl_intent = trailing_stop_distance is not None # Activation is optional

    is_sl_valid = has_sl_intent and is_valid_param(stop_loss_price)
    is_tp_valid = has_tp_intent and is_valid_param(take_profit_price)
    # TSL requires valid distance; activation defaults to '0' if None
    is_tsl_valid = (has_tsl_intent and is_valid_param(trailing_stop_distance) and
                    (tsl_activation_price is None or is_valid_param(tsl_activation_price)))

    if not is_sl_valid and not is_tp_valid and not is_tsl_valid:
        lg.info(f"No valid protection parameters provided for {symbol}. No action taken.")
        return True # No action needed = success

    # --- Prepare API Parameters ---
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    market_id = market_info.get('id')
    if not market_id:
        lg.error(f"Cannot set protection ({symbol}): Missing market ID.")
        return False

    params = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full', # Apply to whole position
        'slTriggerBy': 'LastPrice', # Configurable? Defaulting to LastPrice
        'tpTriggerBy': 'LastPrice', # Configurable? Defaulting to LastPrice
        'slOrderType': 'Market', # Preferred for SL/TP triggers
        'tpOrderType': 'Market',
        'positionIdx': position_idx
    }
    log_parts = [f"Setting protection for {symbol} ({pos_side.upper()}, Idx: {position_idx}):"]
    params_to_send = {} # Build the specific params to send (sl, tp, tsl)

    # --- Format and Add Parameters using Exchange Helpers ---
    try:
        # Helper to format Decimal or '0' using exchange precision
        def format_value(val: Union[Decimal, str], type: str = 'price') -> str:
            if isinstance(val, str) and val == '0': return '0'
            if isinstance(val, Decimal):
                if val < 0: raise ValueError(f"Negative value {val} invalid for protection.")
                if val == 0: return '0' # Treat Decimal 0 as cancel
                try:
                    # Use price_to_precision for both price levels and price distances (TSL)
                    return exchange.price_to_precision(symbol, float(val))
                except (ccxt.BaseError, ValueError, TypeError) as fmt_err:
                    raise ValueError(f"Formatting failed for {val}: {fmt_err}")
            raise TypeError(f"Invalid type {type(val)} for formatting.")

        # TSL Handling (Prioritize if active TSL distance provided)
        tsl_distance_formatted = None
        if is_tsl_valid:
            tsl_distance_formatted = format_value(trailing_stop_distance, 'price')
            # If TSL distance is active (not '0'), set TSL params
            if tsl_distance_formatted != '0':
                params_to_send['trailingStop'] = tsl_distance_formatted
                # Handle activation price (default to '0' if None)
                activation_price = tsl_activation_price if tsl_activation_price is not None else Decimal('0')
                params_to_send['activePrice'] = format_value(activation_price, 'price')
                log_parts.append(f"  Trailing SL: Distance={params_to_send['trailingStop']}, Activation={params_to_send['activePrice']}")
            else: # Explicitly cancelling TSL
                params_to_send['trailingStop'] = '0'
                # Explicitly reset activation price too when cancelling TSL
                params_to_send['activePrice'] = '0'
                log_parts.append("  Trailing SL: Cancelling (Distance='0')")

        # Fixed SL Handling (Only if TSL is not being actively set OR TSL is being cancelled)
        if is_sl_valid:
            if tsl_distance_formatted is None or tsl_distance_formatted == '0':
                 params_to_send['stopLoss'] = format_value(stop_loss_price, 'price')
                 log_parts.append(f"  Fixed SL: {params_to_send['stopLoss']}")
            elif has_sl_intent: # SL was intended, but active TSL took precedence
                 lg.warning(f"Ignoring Fixed SL for {symbol} as active TSL is being set.")

        # Fixed TP Handling
        if is_tp_valid:
            params_to_send['takeProfit'] = format_value(take_profit_price, 'price')
            log_parts.append(f"  Fixed TP: {params_to_send['takeProfit']}")

    except (ValueError, TypeError, InvalidOperation) as fmt_err:
        lg.error(f"{NEON_RED}Error formatting protection parameters for {symbol}: {fmt_err}. API call aborted.{RESET}")
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error preparing protection parameters for {symbol}: {e}{RESET}", exc_info=True)
        return False

    # Check if any actual protection parameters were added
    if not params_to_send:
        lg.info(f"No protection parameters to set/modify for {symbol} after processing. No API call needed.")
        return True

    # Combine base params with specific protection params
    params.update(params_to_send)

    # Log the final parameters being sent
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call Params: {params}")

    # --- Make the API Call ---
    try:
        # Use the correct CCXT method for Bybit V5 endpoint
        method_name = 'private_post_v5_position_trading_stop'
        if not hasattr(exchange, method_name):
             raise NotImplementedError(f"Exchange object missing '{method_name}'. Update CCXT?")

        api_call_func = getattr(exchange, method_name)
        response = api_call_func(params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        # --- Check Response ---
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {})

        if ret_code == 0:
            # Check for messages indicating no change needed
            no_change_msgs = ["not modified", "same tpsl"]
            if any(msg in ret_msg.lower() for msg in no_change_msgs):
                lg.info(f"{NEON_YELLOW}Position protection already set to target values for {symbol} (Exchange: {ret_msg}).{RESET}")
            else:
                lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            return True
        else:
            # Log specific error hints
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            # Add hints based on common error codes
            if ret_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint (110043): Set tpsl failed. Check trigger prices (SL/TP correct side?), `tpslMode`, `retExtInfo`.{RESET}")
            elif ret_code == 110025: lg.error(f"{NEON_YELLOW} >> Hint (110025): Position not found/zero size. Closed already? `positionIdx` mismatch?{RESET}")
            elif ret_code == 110055: lg.error(f"{NEON_YELLOW} >> Hint (110055): 'positionIdx' ({params.get('positionIdx')}) mismatch with Position Mode (One-Way vs Hedge).{RESET}")
            elif ret_code == 110013: lg.error(f"{NEON_YELLOW} >> Hint (110013): Parameter error. Invalid SL/TP/TSL value (tick size)? `activePrice` invalid? SL/TP wrong side? Too close to mark/last?{RESET}")
            elif ret_code == 110036: lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price ({params.get('activePrice')}) invalid (too close, wrong side, passed?).{RESET}")
            elif ret_code in [110084, 110085, 110086]: lg.error(f"{NEON_YELLOW} >> Hint ({ret_code}): SL/TP trigger price error (wrong side, too close, same value?).{RESET}")
            # Add more hints as needed
            return False

    except NotImplementedError as e: # Catch if method doesn't exist
        lg.error(f"{NEON_RED}{e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting protection for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e: # Catch potential errors from the call itself
        lg.error(f"{NEON_RED}Exchange error during protection API call for {symbol}: {e}{RESET}")
    except ccxt.BaseError as e:
        lg.error(f"{NEON_RED}CCXT BaseError setting protection for {symbol}: {e}{RESET}")
    except KeyError as e:
        lg.error(f"{NEON_RED}Error setting protection ({symbol}): Missing key {e} in market/position info.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional TP price (Decimal)
) -> bool:
    """
    Calculates TSL parameters based on config and position, then calls the internal
    `_set_position_protection` helper to set TSL (and optionally TP), cancelling fixed SL.

    Args:
        exchange: Initialized ccxt exchange object.
        symbol: The trading symbol.
        market_info: Market dictionary.
        position_info: Confirmed position dictionary.
        config: The loaded configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional fixed TP price (Decimal) to set alongside TSL.

    Returns:
        True if the protection API call is attempted successfully, False otherwise.
    """
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss is disabled in config for {symbol}. Skipping TSL setup.")
        return False # TSL wasn't set because it's disabled
    if not market_info or not position_info:
        lg.error(f"Cannot set TSL ({symbol}): Missing market or position info.")
        return False

    # --- Get TSL parameters from config ---
    try:
        callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
        if callback_rate <= 0: raise ValueError("trailing_stop_callback_rate must be positive")
        if activation_percentage < 0: raise ValueError("trailing_stop_activation_percentage must be non-negative")
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format/value in config for {symbol}: {e}. Cannot calculate TSL.{RESET}")
        return False

    # --- Extract required position details ---
    try:
        entry_price_str = position_info.get('entryPrice', position_info.get('info', {}).get('avgPrice'))
        pos_side = position_info.get('side')
        if entry_price_str is None or pos_side not in ['long', 'short']:
            raise ValueError("Missing required position info (entryPrice, side)")
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0: raise ValueError(f"Invalid entry price ({entry_price})")
    except (TypeError, ValueError, KeyError, InvalidOperation) as e:
        lg.error(f"{NEON_RED}Error parsing position info for TSL calculation ({symbol}): {e}. Position: {position_info}{RESET}")
        return False

    # --- Calculate TSL parameters for Bybit API ---
    try:
        # Use TradingAnalyzer helpers for precision/tick size
        # Need a temporary analyzer instance for this
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        dummy_df.index.name = 'timestamp'
        try:
            temp_analyzer = TradingAnalyzer(df=dummy_df, logger=lg, config={}, market_info=market_info)
            price_precision = temp_analyzer.get_price_precision()
            min_tick_size = temp_analyzer.get_min_tick_size()
        except ValueError as init_err: # Catch error if market_info was invalid for analyzer
            lg.error(f"{NEON_RED}Failed to initialize temporary analyzer for TSL calc ({symbol}): {init_err}{RESET}")
            return False
        if min_tick_size <= 0: raise ValueError(f"Invalid min tick size ({min_tick_size})")

        # 1. Calculate Activation Price
        activation_price = Decimal('0') # Default to immediate activation
        if activation_percentage > 0:
            activation_offset = entry_price * activation_percentage
            rounding_mode = ROUND_UP if pos_side == 'long' else ROUND_DOWN
            raw_activation = entry_price + activation_offset if pos_side == 'long' else entry_price - activation_offset
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=rounding_mode) * min_tick_size
            # Ensure activation is strictly away from entry
            if (pos_side == 'long' and activation_price <= entry_price):
                activation_price = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
            elif (pos_side == 'short' and activation_price >= entry_price):
                activation_price = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)
            if activation_price <= 0: # Check if activation price became invalid
                 lg.warning(f"Calculated TSL activation price ({activation_price}) is zero or negative. Defaulting to immediate activation ('0').")
                 activation_price = Decimal('0')
        else:
            lg.info(f"TSL activation percentage is zero for {symbol}, setting immediate activation.")

        # 2. Calculate Trailing Stop Distance (price points)
        trailing_distance_raw = entry_price * callback_rate
        # Round distance UP to nearest tick, ensure minimum of one tick
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        if trailing_distance < min_tick_size: trailing_distance = min_tick_size
        if trailing_distance <= 0: raise ValueError(f"Calculated TSL distance ({trailing_distance}) is zero or negative.")

        # --- Logging and API Call ---
        act_price_log = f"{activation_price:.{price_precision}f}" if activation_price > 0 else '0 (Immediate)'
        trail_dist_log = f"{trailing_distance:.{price_precision}f}"
        tp_log = f"{take_profit_price:.{price_precision}f}" if take_profit_price else "None"
        lg.info(f"Calculated TSL Params for {symbol} ({pos_side.upper()}): Activation={act_price_log}, Distance={trail_dist_log}, TP={tp_log}")

        # Call the helper function to set TSL (and TP), explicitly cancelling fixed SL
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price='0', # <<< CRITICAL: Cancel fixed SL when setting TSL
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance, # Pass calculated distance
            tsl_activation_price=activation_price # Pass calculated activation price
        )
    except ValueError as ve: # Catch specific calculation value errors
        lg.error(f"{NEON_RED}Error during TSL calculation for {symbol}: {ve}{RESET}")
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL parameters for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Analyzes a single symbol and executes/manages trades based on signals and config.
    This function represents one cycle of the bot's operation for the symbol.
    """
    lg = logger # Use the symbol-specific logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Market Info & Data ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping analysis cycle.{RESET}")
        return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
        lg.error(f"{NEON_RED}Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe for {symbol}. Skipping cycle.{RESET}")
        return

    # Determine required kline history (adjust based on longest indicator lookback)
    kline_limit = 500 # Default, ensure enough for common indicators + buffer

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50: # Need a reasonable minimum history
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping analysis cycle.{RESET}")
        return

    # Fetch current price with fallback to last close
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch ticker price for {symbol}. Using last close from klines.{RESET}")
        try:
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0:
                current_price = Decimal(str(last_close_val))
                lg.info(f"Using last close price: {current_price}")
            else: raise ValueError(f"Last close price ({last_close_val}) is invalid.")
        except (IndexError, ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"{NEON_RED}Error getting valid last close price for {symbol}: {e}. Cannot proceed.{RESET}")
            return

    # Fetch order book if enabled and weighted
    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators",{}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
        orderbook_limit = config.get("orderbook_limit", 25)
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, orderbook_limit, lg)


    # --- 2. Analyze Data & Generate Signal ---
    try:
        # Initialize analyzer each cycle for fresh state (e.g., break_even_triggered)
        analyzer = TradingAnalyzer(
            df=klines_df.copy(), # Pass a copy
            logger=lg,
            config=config,
            market_info=market_info
        )
    except ValueError as e_analyzer:
        lg.error(f"{NEON_RED}Failed to initialize TradingAnalyzer for {symbol}: {e_analyzer}. Skipping cycle.{RESET}")
        return
    except Exception as e_analyzer_other:
         lg.error(f"{NEON_RED}Unexpected error initializing TradingAnalyzer for {symbol}: {e_analyzer_other}. Skipping cycle.{RESET}", exc_info=True)
         return

    # Check if analyzer initialized correctly and has data
    if not analyzer.indicator_values:
        lg.error(f"{NEON_RED}Indicator calculation failed or produced no values for {symbol}. Skipping signal generation.{RESET}")
        return

    # Generate the trading signal
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)

    # Calculate potential initial SL/TP based on *current* data (used for sizing if opening trade)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr_float = analyzer.indicator_values.get("ATR", float('nan')) # Get float ATR


    # --- 3. Log Analysis Summary ---
    # Signal details are logged within generate_trading_signal
    atr_log = f"{current_atr_float:.{price_precision+1}f}" if not math.isnan(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision}f}" if tp_potential else 'N/A'
    lg.info(f"Current ATR: {atr_log}")
    lg.info(f"Potential Initial SL (for new trade): {sl_pot_log}")
    lg.info(f"Potential Initial TP (for new trade): {tp_pot_log}")
    # Optional: Log Fib levels
    # fib_levels = analyzer.get_nearest_fibonacci_levels(current_price)
    # lg.debug(f"Nearest Fib Levels: " + ", ".join([f"{name}={level:.{price_precision}f}" for name, level in fib_levels]))

    tsl_enabled = config.get('enable_trailing_stop')
    be_enabled = config.get('enable_break_even')
    ma_exit_enabled = config.get('enable_ma_cross_exit')
    lg.info(f"Configured Protections: TSL={'On' if tsl_enabled else 'Off'} | BE={'On' if be_enabled else 'Off'} | MA Exit={'On' if ma_exit_enabled else 'Off'}")


    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.info(f"{NEON_YELLOW}Trading is disabled in config. Analysis complete, no trade actions taken.{RESET}")
    else:
        # --- Get Current Position Status ---
        open_position = get_open_position(exchange, symbol, lg) # Returns enhanced dict or None

        # --- Scenario 1: No Open Position ---
        if open_position is None:
            if signal in ["BUY", "SELL"]:
                lg.info(f"*** {signal} Signal & No Open Position: Initiating Trade Sequence for {symbol} ***")

                # --- Pre-Trade Checks & Setup ---
                balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if balance is None or balance <= 0:
                    lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot proceed, invalid balance ({balance} {QUOTE_CURRENCY}).{RESET}")
                    return

                if sl_potential is None:
                    lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot size trade, Potential Initial SL calculation failed (ATR invalid?).{RESET}")
                    return

                if market_info.get('is_contract', False):
                    leverage = int(config.get("leverage", 1))
                    if leverage > 0:
                        if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set/confirm leverage to {leverage}x. Cannot proceed safely.{RESET}")
                            return
                    else: lg.warning(f"Leverage setting skipped: Configured leverage is {leverage}.")
                else: lg.info("Leverage setting skipped (Spot market).")

                position_size = calculate_position_size(
                    balance=balance, risk_per_trade=config["risk_per_trade"],
                    initial_stop_loss_price=sl_potential, entry_price=current_price,
                    market_info=market_info, exchange=exchange, logger=lg
                )
                if position_size is None or position_size <= 0:
                    lg.error(f"{NEON_RED}Trade Aborted ({signal}): Invalid position size calculated ({position_size}). Check balance, risk, SL, limits.{RESET}")
                    return

                # --- Place Initial Market Order ---
                lg.info(f"==> Placing {signal} market order | Size: {position_size} <==")
                trade_order = place_trade(
                    exchange=exchange, symbol=symbol, trade_signal=signal,
                    position_size=position_size, market_info=market_info,
                    logger=lg, reduce_only=False
                )

                # --- Post-Order: Verify Position and Set Protection ---
                if trade_order and trade_order.get('id'):
                    order_id = trade_order['id']
                    lg.info(f"Order {order_id} placed. Waiting {POSITION_CONFIRM_DELAY}s for position confirmation...")
                    time.sleep(POSITION_CONFIRM_DELAY)

                    lg.info(f"Confirming position status after order {order_id}...")
                    confirmed_position = get_open_position(exchange, symbol, lg)

                    if confirmed_position:
                        try:
                            entry_price_actual_str = confirmed_position.get('entryPrice', confirmed_position.get('info', {}).get('avgPrice'))
                            pos_size_actual_str = confirmed_position.get('contracts', confirmed_position.get('info', {}).get('size'))
                            entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else None
                            pos_size_actual = Decimal(str(pos_size_actual_str)) if pos_size_actual_str else None

                            if not entry_price_actual or entry_price_actual <= 0 or not pos_size_actual or pos_size_actual == 0:
                                raise ValueError(f"Confirmed position has invalid entry/size: Entry={entry_price_actual}, Size={pos_size_actual}")

                            lg.info(f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_price_actual:.{price_precision}f}, Size: {pos_size_actual}{RESET}")

                            # --- Recalculate SL/TP based on ACTUAL entry ---
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                            # --- Set Protection (TSL or Fixed SL/TP) ---
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting Trailing Stop Loss (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=confirmed_position, config=config, logger=lg,
                                    take_profit_price=tp_actual # Pass optional TP
                                )
                            else: # Use Fixed SL/TP
                                lg.info(f"Setting Fixed Stop Loss ({sl_actual}) and Take Profit ({tp_actual})...")
                                if sl_actual or tp_actual:
                                    protection_set_success = _set_position_protection(
                                        exchange=exchange, symbol=symbol, market_info=market_info,
                                        position_info=confirmed_position, logger=lg,
                                        stop_loss_price=sl_actual, take_profit_price=tp_actual,
                                        trailing_stop_distance='0', tsl_activation_price='0' # Cancel TSL
                                    )
                                else:
                                    lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed after entry. No fixed protection set.{RESET}")
                                    protection_set_success = True # No protection needed = success

                            if protection_set_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({signal}) ===")
                            else: lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION ({signal}) ===")

                        except (ValueError, TypeError, InvalidOperation, KeyError) as post_trade_err:
                            lg.error(f"{NEON_RED}Error during post-trade processing for {symbol}: {post_trade_err}{RESET}", exc_info=True)
                            lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")
                    else: # Position not confirmed
                        lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position after {POSITION_CONFIRM_DELAY}s delay!{RESET}")
                        lg.warning(f"{NEON_YELLOW}Order might have failed, filled partially, or API delay. Manual investigation required!{RESET}")
                        # Optional: Fetch order status
                        try:
                            order_status = exchange.fetch_order(order_id, symbol)
                            lg.info(f"Status of order {order_id}: {order_status}")
                        except Exception as fetch_order_err: lg.warning(f"Could not fetch status for order {order_id}: {fetch_order_err}")
                else: # place_trade failed
                    lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({signal}). See previous logs. ===")
            else: # No position and signal is HOLD
                lg.info(f"Signal is HOLD and no open position. No trade action taken.")

        # --- Scenario 2: Existing Open Position Found ---
        else: # open_position is not None
            pos_side = open_position.get('side', 'unknown')
            pos_size_str = open_position.get('contracts', open_position.get('info',{}).get('size', 'N/A'))
            entry_price_str = open_position.get('entryPrice', open_position.get('info', {}).get('avgPrice', 'N/A'))
            # Use Decimal protection info parsed by get_open_position
            current_sl_price_dec = open_position.get('stopLossPriceDecimal')
            current_tp_price_dec = open_position.get('takeProfitPriceDecimal')
            tsl_distance_dec = open_position.get('trailingStopLossDistanceDecimal', Decimal(0))
            is_tsl_active = tsl_distance_dec > 0
            sl_log_str = f"{current_sl_price_dec:.{price_precision}f}" if current_sl_price_dec else 'N/A'
            tp_log_str = f"{current_tp_price_dec:.{price_precision}f}" if current_tp_price_dec else 'N/A'

            lg.info(f"Existing {pos_side.upper()} position found. Size: {pos_size_str}, Entry: {entry_price_str}, SL: {sl_log_str}, TP: {tp_log_str}, TSL Active: {is_tsl_active}")

            # --- ** Check Exit Conditions ** ---
            exit_signal_triggered = False
            if (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY"):
                exit_signal_triggered = True
                lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position. ***{RESET}")

            ma_cross_exit = False
            if not exit_signal_triggered and config.get("enable_ma_cross_exit", False):
                ema_short = analyzer.indicator_values.get("EMA_Short")
                ema_long = analyzer.indicator_values.get("EMA_Long")
                if not math.isnan(ema_short) and not math.isnan(ema_long):
                    if (pos_side == 'long' and ema_short < ema_long):
                        ma_cross_exit = True
                        lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bearish): Short EMA ({ema_short:.{price_precision}f}) < Long EMA ({ema_long:.{price_precision}f}). Closing LONG. ***{RESET}")
                    elif (pos_side == 'short' and ema_short > ema_long):
                        ma_cross_exit = True
                        lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bullish): Short EMA ({ema_short:.{price_precision}f}) > Long EMA ({ema_long:.{price_precision}f}). Closing SHORT. ***{RESET}")
                else: lg.warning("MA cross exit check skipped: EMA values unavailable.")

            # --- Execute Position Close if Exit Condition Met ---
            if exit_signal_triggered or ma_cross_exit:
                lg.info(f"Attempting to close {pos_side} position with a market order (reduceOnly=True)...")
                try:
                    close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                    size_to_close_str = open_position.get('contracts', open_position.get('info',{}).get('size'))
                    if size_to_close_str is None: raise ValueError("Could not determine position size to close.")
                    size_to_close = abs(Decimal(str(size_to_close_str)))
                    if size_to_close <= 0: raise ValueError(f"Position size ({size_to_close}) is zero or invalid.")

                    close_order = place_trade(
                        exchange=exchange, symbol=symbol, trade_signal=close_side_signal,
                        position_size=size_to_close, market_info=market_info,
                        logger=lg, reduce_only=True # CRITICAL
                    )

                    if close_order:
                         order_id = close_order.get('id', 'N/A')
                         # Handle case where place_trade returned dummy success for 'position not found'
                         if close_order.get('info',{}).get('reason') == 'Position not found on close attempt':
                              lg.info(f"{NEON_GREEN}Position for {symbol} confirmed already closed.{RESET}")
                         else:
                              lg.info(f"{NEON_GREEN}Closing order {order_id} placed. Waiting {POSITION_CONFIRM_DELAY}s to verify closure...{RESET}")
                              time.sleep(POSITION_CONFIRM_DELAY)
                              final_position = get_open_position(exchange, symbol, lg)
                              if final_position is None: lg.info(f"{NEON_GREEN}=== POSITION successfully closed. ===")
                              else: lg.error(f"{NEON_RED}*** POSITION CLOSE FAILED after reduceOnly order {order_id}. Position still detected: {final_position}{RESET}")
                    else: # place_trade failed
                        lg.error(f"{NEON_RED}Failed to place closing order. Manual intervention required.{RESET}")

                except (ValueError, InvalidOperation, TypeError) as size_err:
                     lg.error(f"{NEON_RED}Error determining size for closing order: {size_err}. Manual intervention required.{RESET}")
                except Exception as close_err:
                     lg.error(f"{NEON_RED}Unexpected error during position close: {close_err}{RESET}", exc_info=True)

            # --- Check Break-Even Condition (Only if NOT Exiting) ---
            elif config.get("enable_break_even", False) and not analyzer.break_even_triggered:
                try:
                    entry_price_dec = Decimal(str(entry_price_str))
                    current_atr_dec = Decimal(str(current_atr_float)) if not math.isnan(current_atr_float) else None
                    if entry_price_dec > 0 and current_atr_dec and current_atr_dec > 0 and min_tick_size > 0:
                        trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        min_tick = min_tick_size

                        profit_target_offset = current_atr_dec * trigger_multiple
                        be_stop_price = None
                        trigger_met = False
                        trigger_price = None

                        if pos_side == 'long':
                            trigger_price = entry_price_dec + profit_target_offset
                            if current_price >= trigger_price:
                                trigger_met = True
                                be_stop_raw = entry_price_dec + (min_tick * offset_ticks)
                                be_stop_price = (be_stop_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                                if be_stop_price <= entry_price_dec: be_stop_price = (entry_price_dec + min_tick).quantize(min_tick, rounding=ROUND_UP)
                        elif pos_side == 'short':
                            trigger_price = entry_price_dec - profit_target_offset
                            if current_price <= trigger_price:
                                trigger_met = True
                                be_stop_raw = entry_price_dec - (min_tick * offset_ticks)
                                be_stop_price = (be_stop_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                                if be_stop_price >= entry_price_dec: be_stop_price = (entry_price_dec - min_tick).quantize(min_tick, rounding=ROUND_DOWN)

                        if trigger_met and be_stop_price and be_stop_price > 0:
                            lg.warning(f"{NEON_PURPLE}*** BREAK-EVEN Triggered for {pos_side.upper()} position! ***")
                            lg.info(f"  Current Price: {current_price:.{price_precision}f}, Trigger Price: {trigger_price:.{price_precision}f}")
                            lg.info(f"  Moving SL to: {be_stop_price:.{price_precision}f} (Entry {entry_price_dec:.{price_precision}f} +/- {offset_ticks} ticks)")

                            modify_sl_needed = True
                            if current_sl_price_dec: # Check if SL already better
                                if (pos_side == 'long' and current_sl_price_dec >= be_stop_price) or \
                                   (pos_side == 'short' and current_sl_price_dec <= be_stop_price):
                                     lg.info(f"  SL ({sl_log_str}) already at/beyond BE target. No modification needed.")
                                     analyzer.break_even_triggered = True
                                     modify_sl_needed = False

                            if modify_sl_needed:
                                lg.info(f"  Modifying protection: Setting Fixed SL to {be_stop_price:.{price_precision}f}, keeping TP {tp_log_str}, cancelling TSL.")
                                be_success = _set_position_protection(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=open_position, logger=lg,
                                    stop_loss_price=be_stop_price, # New BE SL
                                    take_profit_price=current_tp_price_dec, # Keep existing TP
                                    trailing_stop_distance='0', tsl_activation_price='0' # Cancel TSL
                                )
                                if be_success:
                                    lg.info(f"{NEON_GREEN}Break-even SL successfully set/updated.{RESET}")
                                    analyzer.break_even_triggered = True
                                else: lg.error(f"{NEON_RED}Failed to set break-even SL.{RESET}")
                        elif trigger_met: # Trigger met but BE price invalid
                            lg.error(f"{NEON_RED}Break-even triggered, but calculated BE stop price ({be_stop_price}) is invalid.{RESET}")
                    else: # Conditions for BE check not met (price, ATR, etc.)
                        lg.debug("Break-even check conditions not met (price, ATR, or min_tick invalid).")
                except (InvalidOperation, ValueError, TypeError, KeyError) as be_err:
                    lg.error(f"{NEON_RED}Error calculating/applying break-even logic: {be_err}{RESET}", exc_info=False) # Keep log cleaner

            # --- If no exit or BE modification, log HOLD ---
            elif not exit_signal_triggered and not ma_cross_exit and not analyzer.break_even_triggered:
                 lg.info(f"Signal is {signal}. Holding existing {pos_side.upper()} position. No management action.")

    # --- End of Trading Logic ---

    # --- 5. Log Cycle Timing ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End ({cycle_end_time - cycle_start_time:.2f}s) ==---")


# --- Main Function ---
def main():
    """Main function to parse arguments, initialize, and run the bot loop."""
    parser = argparse.ArgumentParser(description="Enhanced Bybit Trading Bot")
    parser.add_argument("symbol", help="Trading symbol in CCXT format (e.g., BTC/USDT:USDT or ETH/USDT)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable DEBUG level logging to console")
    args = parser.parse_args()
    symbol = args.symbol # Use symbol from command line

    # Set console log level based on argument
    console_log_level = logging.DEBUG if args.debug else logging.INFO

    # Initialize logger for the specified symbol *after* setting console level
    logger = setup_logger(symbol, console_log_level)

    logger.info(f"---=== {NEON_GREEN}Whale 2.0 Enhanced Trading Bot Initializing{RESET} ===---")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Config File: {CONFIG_FILE}")
    logger.info(f"Log Directory: {LOG_DIRECTORY}")
    logger.info(f"Timezone: {TIMEZONE.key}")
    logger.info(f"Trading Enabled: {NEON_RED if CONFIG.get('enable_trading') else NEON_YELLOW}{CONFIG.get('enable_trading', False)}{RESET}")
    logger.info(f"Sandbox Mode: {NEON_YELLOW if CONFIG.get('use_sandbox') else NEON_RED}{CONFIG.get('use_sandbox', True)}{RESET}")
    logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    logger.info(f"Risk Per Trade: {CONFIG.get('risk_per_trade', 0.01):.2%}")
    logger.info(f"Leverage: {CONFIG.get('leverage', 1)}x")
    logger.info(f"Interval: {CONFIG.get('interval')}")

    # Validate interval from config
    if CONFIG.get("interval") not in VALID_INTERVALS:
        logger.critical(f"{NEON_RED}Invalid 'interval' ({CONFIG.get('interval')}) in config. Must be one of {VALID_INTERVALS}. Exiting.{RESET}")
        return

    # Initialize exchange
    exchange = initialize_exchange(CONFIG, logger)
    if not exchange:
        logger.critical(f"{NEON_RED}Failed to initialize exchange. Bot cannot start.{RESET}")
        return

    # Validate symbol and get initial market info
    market_info = get_market_info(exchange, symbol, logger)
    if not market_info:
        logger.critical(f"{NEON_RED}Symbol {symbol} not found or invalid on {exchange.id}. Exiting.{RESET}")
        return
    # Log key market details obtained
    try:
         is_contract = market_info.get('is_contract', False)
         min_tick = market_info.get('precision',{}).get('tick')
         min_amt = market_info.get('limits',{}).get('amount', {}).get('min')
         min_cost = market_info.get('limits',{}).get('cost', {}).get('min')
         logger.info(f"Symbol Details: Contract={is_contract}, TickSize={min_tick}, MinAmount={min_amt}, MinCost={min_cost}")
    except Exception as e:
         logger.warning(f"Could not log detailed market info: {e}")

    # --- Bot Main Loop ---
    logger.info(f"{NEON_GREEN}Initialization complete. Starting main trading loop for {symbol}...{RESET}")
    loop_count = 0
    while True:
        loop_start_utc = datetime.now(ZoneInfo("UTC"))
        loop_start_local = loop_start_utc.astimezone(TIMEZONE)
        loop_count += 1
        logger.debug(f"--- Loop Cycle {loop_count} starting at {loop_start_local.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

        try:
            # Core logic execution for the symbol
            analyze_and_trade_symbol(exchange, symbol, CONFIG, logger)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down gracefully...")
            # Optional: Add logic here to close open positions if desired on exit
            # Example: close_open_positions(exchange, symbol, logger)
            logger.info("Shutdown complete.")
            break
        except ccxt.AuthenticationError as e:
            logger.critical(f"{NEON_RED}CRITICAL: Authentication Error during main loop: {e}. Bot stopped. Check API keys/permissions.{RESET}")
            break # Stop on authentication errors
        except ccxt.NetworkError as e:
            logger.error(f"{NEON_RED}Network Error in main loop: {e}. Retrying after longer delay...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 6) # Exponential backoff might be better
        except ccxt.RateLimitExceeded as e:
             wait_time = _handle_rate_limit(e, logger)
             logger.warning(f"{NEON_YELLOW}Rate Limit Exceeded in main loop. Waiting {wait_time}s...{RESET}")
             time.sleep(wait_time)
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"{NEON_RED}Exchange Not Available: {e}. Retrying after significant delay...{RESET}")
            time.sleep(config.get("loop_delay", LOOP_DELAY_SECONDS) * 10) # e.g., wait 2.5 minutes
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            logger.error(f"{NEON_RED}Exchange Error in main loop: {e} (Code: {bybit_code}){RESET}")
            if bybit_code == 10016 or "system maintenance" in err_str:
                logger.warning(f"{NEON_YELLOW}Exchange likely in maintenance. Waiting longer...{RESET}")
                time.sleep(config.get("loop_delay", LOOP_DELAY_SECONDS) * 20) # Wait longer
            else:
                # For other exchange errors, retry after moderate delay
                time.sleep(RETRY_DELAY_SECONDS * 3)
        except Exception as e:
            logger.error(f"{NEON_RED}An unexpected critical error occurred in the main loop: {e}{RESET}", exc_info=True)
            logger.critical("Bot encountered a potentially fatal error. Stopping for safety.")
            # Consider sending a notification here (email, Telegram, etc.)
            break # Stop the bot

        # --- Loop Delay ---
        try:
             loop_delay = int(config.get("loop_delay", LOOP_DELAY_SECONDS))
             loop_delay = max(1, loop_delay) # Ensure minimum 1 second delay
        except (ValueError, TypeError):
            loop_delay = LOOP_DELAY_SECONDS
            logger.warning(f"Invalid loop_delay in config. Using default {loop_delay}s.")

        logger.debug(f"Loop cycle finished. Sleeping for {loop_delay} seconds...")
        time.sleep(loop_delay)

    # --- End of Main Loop ---
    logger.info(f"---=== {NEON_RED}Trading Bot for {symbol} has stopped.{RESET} ===---")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any uncaught exceptions during initialization or main() execution
        print(f"\n{NEON_RED}---=== UNHANDLED EXCEPTION CAUGHT ===---{RESET}")
        print(f"{NEON_RED}Error: {e}{RESET}")
        # Attempt to log if logger might be available
        try:
            root_logger = logging.getLogger()
            if root_logger and root_logger.hasHandlers():
                root_logger.critical("Unhandled exception caused script termination.", exc_info=True)
            else: # Fallback to print traceback if logger failed
                import traceback
                print("\n--- Traceback ---")
                traceback.print_exc()
                print("-----------------\n")
        except Exception as log_ex:
             print(f"(Logging failed during exception handling: {log_ex})")
    finally:
         print(f"\n{NEON_BLUE}Bot execution finished.{RESET}")
