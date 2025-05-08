```python
# scalpxrx.py
# Enhanced and Upgraded Scalping Bot Framework
# Derived from xrscalper.py, focusing on robust execution, error handling,
# advanced position management (BE, TSL), and Bybit V5 compatibility.

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

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta  # Import pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

# Initialize colorama and set Decimal precision
getcontext().prec = 36  # Increased precision for complex calculations
init(autoreset=True)
load_dotenv()

# Neon Color Scheme
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
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Timezone for logging and display
TIMEZONE = ZoneInfo("America/Chicago")  # Adjust as needed
MAX_API_RETRIES = 5  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 7  # Increased delay between retries
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable
# Default indicator periods (can be overridden by config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14
DEFAULT_STOCH_WINDOW = 12
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

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 10 # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY_SECONDS = 10 # Increased wait time after placing order before confirming position
# QUOTE_CURRENCY dynamically loaded from config

os.makedirs(LOG_DIRECTORY, exist_ok=True)

class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file, creating default if not found,
    and ensuring all default keys are present with validation.
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
        "max_concurrent_positions": 1, # Max open positions for this symbol instance
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
        "trailing_stop_callback_rate": 0.005, # Trail distance % (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003, # % profit move from entry to activate TSL
        "enable_break_even": True, # Enable moving SL to break-even point
        "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR
        "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry price
        "time_based_exit_minutes": None, # Optional: Exit after X minutes (e.g., 60)

        # Indicator Periods & Parameters
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW,
        "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, # StochRSI main window
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW, # Underlying RSI window for StochRSI
        "stoch_rsi_k": DEFAULT_K_WINDOW, # StochRSI K period
        "stoch_rsi_d": DEFAULT_D_WINDOW, # StochRSI D period
        "psar_af": DEFAULT_PSAR_AF, # PSAR Acceleration Factor
        "psar_max_af": DEFAULT_PSAR_MAX_AF, # PSAR Max Acceleration Factor
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW,

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
        },
        "active_weight_set": "default" # Select the active weight set
    }

    config = default_config.copy()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            # Merge loaded config with defaults, ensuring all keys exist
            config = _merge_configs(loaded_config, default_config)
            print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
            # Attempt to recreate default file if loading failed
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(default_config, f_write, indent=4)
                print(f"{NEON_YELLOW}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create:
                print(f"{NEON_RED}Error recreating default config file: {e_create}{RESET}")
            config = default_config # Use in-memory default
    else:
        # Config file doesn't exist, create it with defaults
        print(f"{NEON_YELLOW}Config file not found. Creating default config at {filepath}{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            config = default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            # Continue with in-memory default config if creation fails

    # --- Validation Section ---
    updated = False # Flag to track if config needs saving back

    # Validate symbol
    if not config.get("symbol") or not isinstance(config.get("symbol"), str):
         print(f"{NEON_RED}CRITICAL: 'symbol' is missing, empty, or invalid in config. Resetting to default: '{default_config['symbol']}'{RESET}")
         config["symbol"] = default_config["symbol"]
         updated = True

    # Validate interval
    if config.get("interval") not in VALID_INTERVALS:
        print(f"{NEON_RED}Invalid interval '{config.get('interval')}' in config. Resetting to default '{default_config['interval']}'. Valid: {VALID_INTERVALS}{RESET}")
        config["interval"] = default_config["interval"]
        updated = True

    # Validate entry order type
    if config.get("entry_order_type") not in ["market", "limit"]:
        print(f"{NEON_RED}Invalid entry_order_type '{config.get('entry_order_type')}' in config. Resetting to 'market'.{RESET}")
        config["entry_order_type"] = "market"
        updated = True

    # Validate active weight set exists
    if config.get("active_weight_set") not in config.get("weight_sets", {}):
         print(f"{NEON_RED}Active weight set '{config.get('active_weight_set')}' not found in 'weight_sets'. Resetting to 'default'.{RESET}")
         config["active_weight_set"] = "default" # Ensure 'default' exists in defaults
         updated = True

    # Validate numeric parameters (ranges and types)
    numeric_params = {
        # key: (min_val, max_val, allow_min_equal, allow_max_equal, is_integer)
        "risk_per_trade": (0, 1, False, False, False),
        "leverage": (1, 1000, True, True, True), # Adjust max leverage realistically
        "stop_loss_multiple": (0, float('inf'), False, True, False),
        "take_profit_multiple": (0, float('inf'), False, True, False),
        "trailing_stop_callback_rate": (0, 1, False, False, False),
        "trailing_stop_activation_percentage": (0, 1, True, False, False), # Allow 0%
        "break_even_trigger_atr_multiple": (0, float('inf'), False, True, False),
        "break_even_offset_ticks": (0, 100, True, True, True),
        "signal_score_threshold": (0, float('inf'), False, True, False),
        "atr_period": (1, 1000, True, True, True),
        "ema_short_period": (1, 1000, True, True, True),
        "ema_long_period": (1, 1000, True, True, True),
        "rsi_period": (1, 1000, True, True, True),
        "bollinger_bands_period": (1, 1000, True, True, True),
        "bollinger_bands_std_dev": (0, 10, False, True, False),
        "cci_window": (1, 1000, True, True, True),
        "williams_r_window": (1, 1000, True, True, True),
        "mfi_window": (1, 1000, True, True, True),
        "stoch_rsi_window": (1, 1000, True, True, True),
        "stoch_rsi_rsi_window": (1, 1000, True, True, True),
        "stoch_rsi_k": (1, 1000, True, True, True),
        "stoch_rsi_d": (1, 1000, True, True, True),
        "psar_af": (0, 1, False, False, False),
        "psar_max_af": (0, 1, False, False, False),
        "sma_10_window": (1, 1000, True, True, True),
        "momentum_period": (1, 1000, True, True, True),
        "volume_ma_period": (1, 1000, True, True, True),
        "fibonacci_window": (2, 1000, True, True, True), # Need at least 2 points
        "orderbook_limit": (1, 100, True, True, True),
        "position_confirm_delay_seconds": (0, 60, True, True, False),
        "loop_delay_seconds": (1, 300, True, True, False),
        "stoch_rsi_oversold_threshold": (0, 100, True, False, False),
        "stoch_rsi_overbought_threshold": (0, 100, False, True, False),
        "volume_confirmation_multiplier": (0, float('inf'), False, True, False),
        "limit_order_offset_buy": (0, 0.1, True, False, False), # 10% offset max?
        "limit_order_offset_sell": (0, 0.1, True, False, False),
    }
    for key, (min_val, max_val, allow_min, allow_max, is_integer) in numeric_params.items():
        try:
            value_str = str(config[key]) # Convert potential int/float to str first
            value = Decimal(value_str) if not is_integer else int(Decimal(value_str))

            # Check bounds
            lower_bound_ok = value >= min_val if allow_min else value > min_val
            upper_bound_ok = value <= max_val if allow_max else value < max_val

            if not (lower_bound_ok and upper_bound_ok):
                raise ValueError(f"Value {value} out of range "
                                 f"({min_val} {'<=' if allow_min else '<'} x {'<=' if allow_max else '<'} {max_val})")

            # Store the validated value (could be int or float/Decimal)
            config[key] = int(value) if is_integer else float(value) # Store float for simplicity unless int needed

        except (ValueError, TypeError, KeyError, InvalidOperation) as e:
            print(f"{NEON_RED}Invalid value for '{key}' ({config.get(key)}): {e}. Resetting to default '{default_config[key]}'.{RESET}")
            config[key] = default_config[key]
            updated = True

    # Specific validation for time_based_exit_minutes (allow None or positive number)
    time_exit = config.get("time_based_exit_minutes")
    if time_exit is not None:
        try:
            time_exit_val = float(time_exit)
            if time_exit_val <= 0: raise ValueError("Must be positive if set")
            config["time_based_exit_minutes"] = time_exit_val # Store as float
        except (ValueError, TypeError) as e:
             print(f"{NEON_RED}Invalid value for 'time_based_exit_minutes' ({time_exit}): {e}. Resetting to default (None).{RESET}")
             config["time_based_exit_minutes"] = None
             updated = True

    # If config was updated due to invalid values, save it back
    if updated:
        try:
            with open(filepath, "w", encoding="utf-8") as f_write:
                # Use ensure_ascii=False for better readability if non-ASCII chars exist
                json.dump(config, f_write, indent=4, ensure_ascii=False)
            print(f"{NEON_YELLOW}Updated config file {filepath} with corrected/default values.{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

    return config

def _merge_configs(loaded_config: Dict, default_config: Dict) -> Dict:
    """
    Recursively merges the loaded configuration with default values.
    Ensures all keys from the default config exist in the final config.
    Prioritizes values from the loaded config.
    """
    merged = default_config.copy()
    for key, value in loaded_config.items():
        # If key exists in both and both values are dicts, recurse
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_configs(value, merged[key])
        else:
            # Otherwise, overwrite default with loaded value
            merged[key] = value
    return merged

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with rotating file and colored console handlers."""
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if logger is reused
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            try: handler.close()
            except: pass # Ignore errors closing handlers
            logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG) # Capture all levels at the logger level

    # File Handler (Rotating)
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log DEBUG and above to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger {log_filename}: {e}")

    # Console Handler (Colored)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z' # Include Timezone abbreviation
    )
    # Use UTC internally for consistency, display local time in logs via formatter
    stream_formatter.converter = time.gmtime # Log record times in UTC
    # Set the formatter's timezone for display purposes (using datefmt %Z)
    # Ensure the TIMEZONE object is used correctly by the formatter
    logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()

    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(level) # Set console level (e.g., INFO)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent duplicate logs in root logger
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with enhanced error handling."""
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Use CCXT's built-in rate limiter
            'rateLimit': 150, # Adjust based on Bybit V5 limits (e.g., 100ms for 10/s might be safer)
            'options': {
                'defaultType': 'linear', # Essential for Bybit V5 USDT perpetuals
                'adjustForTimeDifference': True, # Helps with timestamp sync issues
                # Increased timeouts
                'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000, 'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000, 'fetchOHLCVTimeout': 20000,
                # Add user agent for potential identification
                'user-agent': 'ScalpXRX Bot v1.0',
                # Consider Bybit V5 specific options if needed
                # 'recvWindow': 10000 # Example: Increase if timestamp errors persist
            }
        }

        # Default to Bybit, could be made configurable
        exchange_id = "bybit"
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        # --- Sandbox Mode Setup ---
        if config.get('use_sandbox'):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            try:
                exchange.set_sandbox_mode(True)
                lg.info(f"Sandbox mode explicitly enabled for {exchange.id}.")
            except AttributeError:
                lg.warning(f"{exchange.id} does not support set_sandbox_mode via ccxt. Ensuring API keys are for Testnet.")
                # Manually set Bybit testnet URL if needed
                if exchange.id == 'bybit':
                    exchange.urls['api'] = 'https://api-testnet.bybit.com'
                    lg.info("Manually set Bybit API URL to Testnet.")
            except Exception as e:
                lg.error(f"Error enabling sandbox mode: {e}")
        else:
            lg.info(f"{NEON_GREEN}Using LIVE (Real Money) Environment.{RESET}")


        lg.info(f"Initializing {exchange.id}...")
        # --- Initial Connection & Permissions Test (Fetch Balance) ---
        account_type_to_test = 'CONTRACT' # Prioritize CONTRACT for V5 derivatives
        lg.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            params = {'type': account_type_to_test} if exchange.id == 'bybit' else {}
            # Use safe_api_call for robustness
            balance = safe_api_call(exchange.fetch_balance, lg, params=params)

            if balance:
                quote_curr = config.get("quote_currency", "USDT")
                # Handle potential differences in balance structure more robustly
                available_quote_val = None
                if quote_curr in balance:
                    available_quote_val = balance[quote_curr].get('free')
                if available_quote_val is None and 'free' in balance and quote_curr in balance['free']:
                    available_quote_val = balance['free'][quote_curr]
                # Add check for Bybit V5 structure if needed (though fetch_balance often standardizes)

                available_quote_str = str(available_quote_val) if available_quote_val is not None else 'N/A'
                lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {quote_curr} available: {available_quote_str})")
            else:
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch (Type: {account_type_to_test}) returned no data. Check connection/permissions.{RESET}")

        except ccxt.AuthenticationError as auth_err:
             lg.error(f"{NEON_RED}Authentication Error during initial balance fetch: {auth_err}{RESET}")
             lg.error(f"{NEON_RED}>> Ensure API keys (in .env) are correct, have permissions (Read, Trade), match environment (Real/Testnet), and IP whitelist is correct.{RESET}")
             return None # Fatal error
        except ccxt.ExchangeError as balance_err:
             # Fallback if specific account type failed
             lg.warning(f"{NEON_YELLOW}Exchange error fetching balance (Type: {account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
             try:
                  balance = safe_api_call(exchange.fetch_balance, lg)
                  if balance:
                       quote_curr = config.get("quote_currency", "USDT")
                       available_quote_val = None
                       if quote_curr in balance:
                           available_quote_val = balance[quote_curr].get('free')
                       if available_quote_val is None and 'free' in balance and quote_curr in balance['free']:
                           available_quote_val = balance['free'][quote_curr]
                       available_quote_str = str(available_quote_val) if available_quote_val is not None else 'N/A'
                       lg.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {quote_curr} available: {available_quote_str})")
                  else:
                       lg.warning(f"{NEON_YELLOW}Default balance fetch also returned no data.{RESET}")
             except Exception as fallback_err:
                  lg.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type/network.{RESET}")
        except Exception as balance_err: # Catches errors from safe_api_call
             lg.warning(f"{NEON_YELLOW}Could not perform initial balance fetch after retries or due to error: {balance_err}. Proceeding cautiously.{RESET}")


        # --- Load Markets (Crucial for market info, precision, etc.) ---
        lg.info(f"Loading markets for {exchange.id}...")
        try:
             safe_api_call(exchange.load_markets, lg, reload=True) # Force reload
             lg.info(f"Markets loaded successfully for {exchange.id}.")
        except Exception as market_err:
             lg.error(f"{NEON_RED}Failed to load markets after retries: {market_err}. Cannot operate without market data. Exiting.{RESET}")
             return None # Fatal error if markets cannot be loaded

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox')}")
        return exchange

    except ccxt.AuthenticationError as e: # Catch auth errors during class instantiation
        lg.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        lg.error(f"{NEON_RED}>> Check API Key/Secret format and validity in your .env file.{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None

# --- API Call Wrapper with Retries ---
def safe_api_call(func, logger: logging.Logger, *args, **kwargs):
    """Wraps an API call with retry logic for network/rate limit/specific exchange errors."""
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            result = func(*args, **kwargs)
            # lg.debug(f"API call {func.__name__} successful.") # Optional success log
            return result # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (1.5 ** attempts) # Exponential backoff
            lg.warning(f"{NEON_YELLOW}Retryable network/availability error in {func.__name__}: {type(e).__name__}. Waiting {wait_time:.1f}s (Attempt {attempts+1}/{MAX_API_RETRIES}). Error: {e}{RESET}")
            time.sleep(wait_time)
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time_header = getattr(e, 'retry_after', None)
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts) # Stronger backoff
            if wait_time_header:
                try: wait_time = max(wait_time, float(wait_time_header) + 0.5) # Add buffer
                except: pass # Ignore invalid header
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded in {func.__name__}. Waiting {wait_time:.1f}s (Attempt {attempts+1}/{MAX_API_RETRIES}). Error: {e}{RESET}")
            time.sleep(wait_time)
        except ccxt.AuthenticationError as e:
             lg.error(f"{NEON_RED}Authentication Error in {func.__name__}: {e}. Aborting call.{RESET}")
             raise e # Don't retry, re-raise immediately
        except ccxt.ExchangeError as e:
            last_exception = e
            bybit_retry_codes = [
                10001, # Internal server error
                10006, # Request frequent
                # Add other transient error codes based on Bybit docs/experience
            ]
            exchange_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            is_retryable_exchange_err = exchange_code in bybit_retry_codes or \
                                        "internal server error" in err_str or \
                                        "request validation failed" in err_str # Check message too

            if is_retryable_exchange_err:
                 wait_time = RETRY_DELAY_SECONDS * (1.5 ** attempts)
                 lg.warning(f"{NEON_YELLOW}Potentially retryable exchange error in {func.__name__}: {e} (Code: {exchange_code}). Waiting {wait_time:.1f}s (Attempt {attempts+1}/{MAX_API_RETRIES})...{RESET}")
                 time.sleep(wait_time)
            else:
                 lg.error(f"{NEON_RED}Non-retryable Exchange Error in {func.__name__}: {e} (Code: {exchange_code}){RESET}")
                 raise e # Re-raise non-retryable ones
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error in {func.__name__}: {e}{RESET}", exc_info=True)
            raise e # Re-raise unexpected errors immediately

        attempts += 1

    # If loop completes, max retries exceeded
    lg.error(f"{NEON_RED}Max retries ({MAX_API_RETRIES}) exceeded for {func.__name__}.{RESET}")
    if last_exception:
        raise last_exception # Raise the last known exception
    else:
        # Fallback if no exception was captured (shouldn't normally happen)
        raise ccxt.RequestTimeout(f"Max retries exceeded for {func.__name__} (no specific exception captured)")


# --- CCXT Data Fetching (Using safe_api_call) ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks and retries."""
    lg = logger
    try:
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)
        if not ticker:
            return None # Error logged by safe_api_call

        lg.debug(f"Ticker data for {symbol}: {ticker}")
        price = None
        # Prioritize 'last', then 'average', then mid-price, then ask/bid
        last_price = ticker.get('last')
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')
        avg_price = ticker.get('average')

        # Robust Decimal conversion helper
        def to_decimal(value) -> Optional[Decimal]:
            if value is None: return None
            try:
                d = Decimal(str(value))
                return d if d.is_finite() and d > 0 else None
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Invalid price format encountered: {value}")
                return None

        p_last = to_decimal(last_price)
        p_bid = to_decimal(bid_price)
        p_ask = to_decimal(ask_price)
        p_avg = to_decimal(avg_price)

        # Determine price with priority
        if p_last:
            price = p_last; lg.debug(f"Using 'last' price: {price}")
        elif p_avg:
            price = p_avg; lg.debug(f"Using 'average' price: {price}")
        elif p_bid and p_ask:
            price = (p_bid + p_ask) / 2; lg.debug(f"Using bid/ask midpoint: {price}")
        elif p_ask:
            price = p_ask; lg.warning(f"Using 'ask' price fallback (bid invalid/missing): {price}")
        elif p_bid:
            price = p_bid; lg.warning(f"Using 'bid' price fallback (ask invalid/missing): {price}")

        # Final validation
        if price is not None and price.is_finite() and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to extract a valid price from ticker data for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching current price for {symbol}: {e}{RESET}", exc_info=False)
        return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and robust validation."""
    lg = logger or logging.getLogger(__name__)
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    try:
        ohlcv = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe=timeframe, limit=limit)

        if ohlcv is None or not isinstance(ohlcv, list) or len(ohlcv) == 0:
            if ohlcv is not None:
                lg.warning(f"{NEON_YELLOW}No valid kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            lg.warning(f"Kline data DataFrame is empty for {symbol} {timeframe}.")
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # Convert price/volume columns to numeric Decimal
        for col in ['open', 'high', 'low', 'close', 'volume']:
             try:
                  df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and str(x).strip() != '' else Decimal('NaN'))
             except (TypeError, ValueError, InvalidOperation) as conv_err:
                  lg.warning(f"Could not convert column '{col}' to Decimal, attempting numeric fallback: {conv_err}")
                  df[col] = pd.to_numeric(df[col], errors='coerce')

        # Data Cleaning
        initial_len = len(df)
        close_col = df['close']
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        if not df.empty:
            if isinstance(close_col.iloc[0], Decimal):
                df = df[close_col.apply(lambda x: x.is_finite() and x > 0)]
            elif pd.api.types.is_numeric_dtype(close_col.dtype):
                df = df[np.isfinite(close_col) & (close_col > 0)]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} {timeframe} empty after cleaning.")
            return pd.DataFrame()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching/processing klines for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and validation."""
    lg = logger
    if not exchange.has['fetchOrderBook']:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    try:
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)

        if not orderbook:
            return None
        if not isinstance(orderbook, dict) or 'bids' not in orderbook or 'asks' not in orderbook or \
           not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
            lg.warning(f"Invalid orderbook structure received for {symbol}. Data: {orderbook}")
            return None

        if not orderbook['bids'] and not orderbook['asks']:
            lg.warning(f"Orderbook received but both bids and asks lists are empty for {symbol}.")
            return orderbook # Return empty but valid book

        # Basic validation of entry format
        valid = True
        for side in ['bids', 'asks']:
             if orderbook[side]:
                  entry = orderbook[side][0]
                  if not (isinstance(entry, list) and len(entry) == 2):
                       lg.warning(f"Invalid {side[:-1]} entry format: {entry}"); valid = False; break
                  try: # Check numeric format
                       _ = float(entry[0]); _ = float(entry[1])
                  except (ValueError, TypeError):
                       lg.warning(f"Non-numeric data in {side[:-1]} entry: {entry}"); valid = False; break
        if not valid:
             lg.error("Orderbook data format validation failed."); return None

        lg.debug(f"Successfully fetched orderbook for {symbol} ({len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks).")
        return orderbook

    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching order book for {symbol}: {e}{RESET}", exc_info=False)
        return None

# --- Trading Analyzer Class (Enhancements Incorporated) ---
# Assuming TradingAnalyzer class definition is identical to the input sx.py
# with the enhancements already discussed (robust TA conversion, precision handling, etc.)
# (Class definition omitted here for brevity, assuming it's the enhanced version from sx.py)
# ... [TradingAnalyzer class definition from sx.py goes here] ...

# --- Trading Logic Helper Functions (Enhancements Incorporated) ---
# Assuming these functions are identical to the input sx.py
# with the enhancements already discussed (robust parsing, validation, error handling)
# (Function definitions omitted here for brevity, assuming they are the enhanced versions from sx.py)
# ... [fetch_balance function definition from sx.py goes here] ...
# ... [get_market_info function definition from sx.py goes here] ...
# ... [calculate_position_size function definition from sx.py goes here] ...
# ... [get_open_position function definition from sx.py goes here] ...
# ... [set_leverage_ccxt function definition from sx.py goes here] ...
# ... [place_trade function definition from sx.py goes here] ...
# ... [_set_position_protection function definition from sx.py goes here] ...
# ... [set_trailing_stop_loss function definition from sx.py goes here] ...

# --- Main Analysis and Trading Loop (Enhancements Incorporated) ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Performs one cycle of analysis and trading logic for a single symbol."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    try:
        # --- Get Market Info (Critical) ---
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info: raise ValueError(f"Fatal: Failed to get market info for {symbol}.")

        # --- Fetch Data ---
        ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
        if not ccxt_interval: raise ValueError(f"Invalid interval '{config['interval']}'.")
        klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=500, logger=lg)
        if klines_df.empty or len(klines_df) < 50: raise ValueError(f"Insufficient kline data ({len(klines_df)}).")

        current_price = fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price is None: # Fallback to last close
             lg.warning("Using last close from klines as current price.")
             try:
                 last_close = klines_df['close'].iloc[-1] # Assumes klines_df uses Decimal
                 if not isinstance(last_close, Decimal) or not last_close.is_finite():
                     raise ValueError("Last close is not a valid Decimal.")
                 current_price = last_close
                 if current_price <= 0: raise ValueError("Last close price non-positive.")
             except (IndexError, KeyError, ValueError, TypeError, InvalidOperation) as e:
                 raise ValueError(f"Failed to get valid last close price: {e}")

        # Fetch order book if needed for scoring
        orderbook_data = None
        active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
        if config.get("indicators",{}).get("orderbook", False) and float(active_weights.get("orderbook", 0)) != 0:
            orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)
            if not orderbook_data: lg.warning(f"Failed orderbook fetch for {symbol}. Proceeding without.")

        # --- Analyze Data & Generate Signal ---
        analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
        if not analyzer.indicator_values: raise ValueError("Indicator calculation failed.")
        signal = analyzer.generate_trading_signal(current_price, orderbook_data)

        # --- Calculate Potential TP/SL & Log Summary ---
        _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal)
        price_prec = analyzer.get_price_precision(); current_atr = analyzer.indicator_values.get("ATR")
        lg.info(f"Current Price: {current_price:.{price_prec}f}")
        lg.info(f"ATR: {current_atr:.{price_prec+2}f}" if isinstance(current_atr, Decimal) else 'ATR: N/A')
        lg.info(f"Calc Initial SL (sizing): {sl_calc or 'N/A'}, TP (target): {tp_calc or 'N/A'}")
        lg.info(f"Mgmt: TSL={'On' if config['enable_trailing_stop'] else 'Off'}, BE={'On' if config['enable_break_even'] else 'Off'}, TimeExit={config.get('time_based_exit_minutes') or 'Off'}")

        # --- Trading Execution Logic ---
        if not config.get("enable_trading"):
            lg.debug("Trading disabled. Cycle complete."); return

        open_position = get_open_position(exchange, symbol, lg)

        # --- Scenario 1: No Open Position ---
        if open_position is None:
            if signal in ["BUY", "SELL"]:
                lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence ***")
                balance = fetch_balance(exchange, config["quote_currency"], lg)
                if balance is None or balance <= 0: raise ValueError("Balance fetch failed or zero/negative.")
                if sl_calc is None: raise ValueError("Initial SL calculation failed (required for sizing).")

                if market_info.get('is_contract') and int(config.get("leverage", 0)) > 0:
                    if not set_leverage_ccxt(exchange, symbol, int(config["leverage"]), market_info, lg):
                        raise ValueError("Failed to set leverage.")

                pos_size = calculate_position_size(balance, config["risk_per_trade"], sl_calc, current_price, market_info, exchange, lg)
                if pos_size is None or pos_size <= 0: raise ValueError(f"Position size calculation failed ({pos_size}).")

                # --- Place Order (Market or Limit) ---
                entry_type = config.get("entry_order_type", "market"); limit_px = None
                if entry_type == "limit":
                     try:
                         offset = Decimal(str(config[f"limit_order_offset_{signal.lower()}"]))
                         rnd_f = Decimal(f'1e-{price_prec}')
                         raw_px = current_price * (1 - offset if signal=='BUY' else 1 + offset)
                         limit_px = raw_px.quantize(rnd_f, rounding=ROUND_DOWN if signal=='BUY' else ROUND_UP)
                         if limit_px <= 0: raise ValueError("Limit price non-positive")
                         lg.info(f"Calc Limit Entry for {signal}: {limit_px}")
                     except (KeyError, ValueError, InvalidOperation) as e:
                          lg.error(f"Limit price calc failed ({e}). Switching to Market."); entry_type="market"; limit_px=None

                trade_order = place_trade(exchange, symbol, signal, pos_size, market_info, lg, entry_type, limit_px)

                # --- Post-Order Handling ---
                if trade_order and trade_order.get('id'):
                    order_id, status = trade_order['id'], trade_order.get('status')
                    # If filled immediately (market or fast limit)
                    if status == 'closed' or entry_type == 'market':
                        delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                        lg.info(f"Order {order_id} placed/filled. Waiting {delay}s for confirmation...")
                        time.sleep(delay)
                        confirmed_pos = get_open_position(exchange, symbol, lg)
                        if confirmed_pos:
                            lg.info(f"{NEON_GREEN}Position Confirmed!{RESET}")
                            # --- Set Protection ---
                            try:
                                entry_act = confirmed_pos.get('entryPriceDecimal') or current_price # Use actual or estimate
                                lg.info(f"Actual Entry ~ {entry_act:.{price_prec}f}")
                                _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_act, signal) # Recalculate based on actual entry
                                protection_ok = False
                                if config["enable_trailing_stop"]:
                                     lg.info(f"Setting TSL (TP target: {tp_final})...")
                                     protection_ok = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_pos, config, lg, tp_final)
                                elif sl_final or tp_final: # Use fixed if TSL disabled AND calc valid
                                     lg.info(f"Setting Fixed SL ({sl_final}) / TP ({tp_final})...")
                                     protection_ok = _set_position_protection(exchange, symbol, market_info, confirmed_pos, lg, sl_final, tp_final)
                                else: lg.warning("No valid protection calculated (TSL disabled or calc failed).")

                                if protection_ok: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION COMPLETE ({symbol} {signal}) ===")
                                else: lg.error(f"{NEON_RED}=== TRADE PLACED BUT PROTECTION FAILED ({symbol} {signal}) ===\n{NEON_YELLOW}>>> MANUAL MONITORING REQUIRED! <<<")
                            except Exception as post_err:
                                 lg.error(f"Error setting protection: {post_err}", exc_info=True); lg.warning(f"{NEON_YELLOW}Position open, manual check needed!{RESET}")
                        else: lg.error(f"{NEON_RED}Order {order_id} placed/filled BUT POSITION NOT CONFIRMED! Manual check!{RESET}")
                    # If limit order is open
                    elif status == 'open' and entry_type == 'limit':
                         lg.info(f"Limit order {order_id} OPEN. Will check status next cycle.")
                         # TODO: Optionally add logic to track open orders and cancel if stale
                    else: # Order failed or other status
                         lg.error(f"Order {order_id} status: {status}. Trade did not open as expected.")
                else: # place_trade failed
                     lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED. Order placement error. ===")
            else: lg.info("Signal HOLD, no position. No action.")

        # --- Scenario 2: Existing Open Position ---
        else:
            pos_side = open_position['side']; pos_size = open_position['contractsDecimal']
            entry_price = open_position['entryPriceDecimal']; pos_ts_ms = open_position['timestamp_ms']
            lg.info(f"Managing existing {pos_side.upper()} position. Size: {pos_size}, Entry: {entry_price}")

            # --- Check for Exit Signal (Opposite Direction) ---
            if (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY"):
                lg.warning(f"{NEON_YELLOW}*** EXIT Signal ({signal}) opposes {pos_side} position. Closing... ***{RESET}")
                try:
                    close_sig = "SELL" if pos_side == 'long' else "BUY"
                    size_close = abs(pos_size)
                    if size_close <= 0: raise ValueError("Position size is zero/negative.")
                    lg.info(f"==> Placing {close_sig} MARKET order (reduceOnly=True) | Size: {size_close} <==")
                    close_order = place_trade(exchange, symbol, close_sig, size_close, market_info, lg, 'market', reduce_only=True)
                    if close_order: lg.info(f"{NEON_GREEN}Close order placed successfully. ID: {close_order.get('id','?')}{RESET}")
                    else: lg.error(f"{NEON_RED}Failed placing CLOSE order! Manual check!{RESET}")
                except Exception as close_err:
                    lg.error(f"Error closing position: {close_err}", exc_info=True); lg.warning(f"{NEON_YELLOW}Manual close may be needed!{RESET}")
                return # Exit cycle after close attempt

            # --- Check for Time-Based Exit ---
            time_exit = config.get("time_based_exit_minutes")
            if isinstance(time_exit, (int, float)) and time_exit > 0 and pos_ts_ms:
                 try:
                      elapsed_mins = (time.time() * 1000 - pos_ts_ms) / 60000
                      lg.debug(f"Time Exit Check: Elapsed={elapsed_mins:.2f}m, Limit={time_exit}m")
                      if elapsed_mins >= time_exit:
                           lg.warning(f"{NEON_YELLOW}*** TIME-BASED EXIT ({elapsed_mins:.1f} >= {time_exit}m). Closing... ***{RESET}")
                           # Execute Close Logic (identical to signal-based exit)
                           close_sig = "SELL" if pos_side == 'long' else "BUY"
                           size_close = abs(pos_size)
                           if size_close <= 0: raise ValueError("Position size is zero/negative.")
                           close_order = place_trade(exchange, symbol, close_sig, size_close, market_info, lg, 'market', reduce_only=True)
                           if close_order: lg.info(f"{NEON_GREEN}Time-based CLOSE order placed. ID: {close_order.get('id','?')}{RESET}")
                           else: lg.error(f"{NEON_RED}Failed time-based CLOSE order! Manual check!{RESET}")
                           return # Exit cycle
                 except Exception as time_err: lg.error(f"Error in time exit check: {time_err}")

            # --- Position Management (Break-Even) ---
            # Check if TSL is already active on the exchange
            is_tsl_active = open_position.get('trailingStopLossValueDecimal') is not None
            if config["enable_break_even"] and not is_tsl_active:
                 lg.debug("Checking Break-Even conditions...")
                 try:
                     if entry_price is None or not entry_price.is_finite() or entry_price <= 0: raise ValueError("Invalid entry price for BE")
                     if not isinstance(current_atr, Decimal) or not current_atr.is_finite() or current_atr <= 0: raise ValueError("Invalid ATR for BE")

                     be_trig_atr = Decimal(str(config["break_even_trigger_atr_multiple"]))
                     be_off_ticks = int(config["break_even_offset_ticks"])
                     min_tick = analyzer.get_min_tick_size()

                     price_diff = current_price - entry_price if pos_side == 'long' else entry_price - current_price
                     profit_atr = price_diff / current_atr if current_atr > 0 else Decimal(0)
                     lg.debug(f"BE Check: ProfitATRs={profit_atr:.2f}, TargetATRs={be_trig_atr}")

                     if profit_atr >= be_trig_atr:
                          tick_offset = min_tick * be_off_ticks
                          be_sl = (entry_price + tick_offset).quantize(min_tick, rounding=ROUND_UP) if pos_side=='long' else \
                                  (entry_price - tick_offset).quantize(min_tick, rounding=ROUND_DOWN)
                          if not be_sl.is_finite() or be_sl <= 0: raise ValueError(f"Calculated BE SL non-positive/finite ({be_sl})")

                          curr_sl = open_position.get('stopLossPriceDecimal')
                          update_needed = False
                          if curr_sl is None: update_needed = True; lg.info("BE triggered: No current SL.")
                          elif pos_side=='long' and be_sl > curr_sl: update_needed = True; lg.info(f"BE triggered: Target {be_sl} > Current {curr_sl}.")
                          elif pos_side=='short' and be_sl < curr_sl: update_needed = True; lg.info(f"BE triggered: Target {be_sl} < Current {curr_sl}.")
                          else: lg.debug(f"BE triggered but current SL {curr_sl} already adequate.")

                          if update_needed:
                               lg.warning(f"{NEON_PURPLE}*** Moving SL to Break-Even ({symbol} @ {be_sl}) ***{RESET}")
                               curr_tp = open_position.get('takeProfitPriceDecimal') # Preserve existing TP
                               success = _set_position_protection(exchange, symbol, market_info, open_position, lg, be_sl, curr_tp)
                               if success: lg.info(f"{NEON_GREEN}Break-Even SL updated.{RESET}")
                               else: lg.error(f"{NEON_RED}Failed updating Break-Even SL.{RESET}")
                     else: lg.debug("BE Profit target not reached.")
                 except ValueError as ve: lg.warning(f"BE Check skipped: {ve}")
                 except Exception as be_err: lg.error(f"Error during BE check: {be_err}", exc_info=True)
            elif is_tsl_active: lg.debug("BE check skipped: TSL active.")
            else: lg.debug("BE check skipped: Disabled in config.")

            # Placeholder for other potential management logic (e.g., TSL adjustments)
            # lg.debug("End of position management checks.")

    # --- Error Handling for the entire cycle ---
    except ValueError as data_err: # Catch data/config related errors
        lg.error(f"{NEON_RED}Data/Config Error ({symbol}): {data_err}. Skipping cycle.{RESET}")
    except ccxt.AuthenticationError as auth_err: # Catch critical auth errors
         lg.critical(f"{NEON_RED}CRITICAL: Authentication Failed: {auth_err}. Stopping bot.{RESET}")
         raise SystemExit("Authentication Failed") # Stop the bot
    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as net_err:
         lg.error(f"{NEON_RED}Network/Exchange Availability Error ({symbol}): {net_err}. Skipping cycle.{RESET}")
    except Exception as cycle_err: # Catch unexpected errors
        lg.error(f"{NEON_RED}Unexpected Cycle Error ({symbol}): {cycle_err}{RESET}", exc_info=True)
        # Decide behavior: continue or stop? For now, just log and continue.

    finally:
        # --- Cycle End Logging ---
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY # Allow modification of globals

    # Setup initial logger
    init_logger = setup_logger("ScalpXRX_Init", level=logging.INFO)
    start_time_str = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
    init_logger.info(f"--- Starting ScalpXRX Bot ({start_time_str}) ---")

    try:
        CONFIG = load_config(CONFIG_FILE)
        QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
        TARGET_SYMBOL = CONFIG.get("symbol")

        if not TARGET_SYMBOL:
            init_logger.critical("CRITICAL: 'symbol' not defined in config.json. Exiting.")
            return

        # Setup logger specific to the target symbol for the main loop
        safe_symbol_name = TARGET_SYMBOL.replace('/', '_').replace(':', '-')
        symbol_logger_name = f"ScalpXRX_{safe_symbol_name}"
        main_logger = setup_logger(symbol_logger_name, level=logging.INFO) # Use INFO for console by default
        main_logger.info(f"Logging initialized for symbol: {TARGET_SYMBOL}")
        main_logger.info(f"Config loaded. Quote: {QUOTE_CURRENCY}, Interval: {CONFIG['interval']}")
        try: ta_version = ta.version
        except AttributeError: ta_version = "N/A" # Handle if version attribute is missing
        main_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={ta_version}")


        # --- Trading Enabled Warning ---
        if CONFIG.get("enable_trading"):
            main_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
            env_type = "SANDBOX (Testnet)" if CONFIG.get("use_sandbox") else f"{NEON_RED}!!! REAL MONEY !!!"
            main_logger.warning(f"Environment: {env_type}{RESET}")
            risk_pct = CONFIG.get('risk_per_trade', 0) * 100
            lev = CONFIG.get('leverage', 1)
            main_logger.warning(f"Settings: Risk/Trade={risk_pct:.2f}%, Leverage={lev}x")
            for i in range(3, 0, -1):
                main_logger.warning(f"Starting in {i}...")
                time.sleep(1)
        else:
            main_logger.info("Trading is disabled in config. Running in analysis-only mode.")

        # --- Initialize Exchange ---
        exchange = initialize_exchange(CONFIG, main_logger)
        if not exchange:
            main_logger.critical("Failed to initialize exchange. Exiting.")
            return

        # --- Main Loop ---
        main_logger.info(f"Starting main analysis loop for {TARGET_SYMBOL}...")
        loop_interval = max(1, CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)) # Ensure positive delay
        while True:
            try:
                analyze_and_trade_symbol(exchange, TARGET_SYMBOL, CONFIG, main_logger)
            except SystemExit as e: # Catch SystemExit for clean shutdown
                 main_logger.critical(f"SystemExit triggered: {e}. Shutting down.")
                 break
            except KeyboardInterrupt: # Allow Ctrl+C to break loop
                 main_logger.info("KeyboardInterrupt detected in loop. Shutting down.")
                 break
            except Exception as loop_err:
                # Catch unexpected errors from analyze_and_trade_symbol if they weren't caught internally
                main_logger.error(f"{NEON_RED}Error in main loop iteration: {loop_err}{RESET}", exc_info=True)
                # Decide whether to continue or stop based on the error type?
                # For now, log and continue, but could add logic to stop on critical errors.

            # Delay before next cycle
            main_logger.debug(f"Waiting {loop_interval} seconds before next cycle...")
            time.sleep(loop_interval)

    except KeyboardInterrupt:
        init_logger.info("KeyboardInterrupt received during startup/shutdown. Shutting down...")
    except Exception as startup_err:
        init_logger.critical(f"Critical error during startup: {startup_err}", exc_info=True)
    finally:
        end_time_str = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
        init_logger.info(f"--- ScalpXRX Bot Shutdown ({end_time_str}) ---")
        logging.shutdown() # Ensure all logs are flushed


if __name__ == "__main__":
    main()
```
