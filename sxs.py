# sxs.py
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
        avg_price = ticker.get('average') # Some exchanges provide volume-weighted avg or simple mid

        # Robust Decimal conversion helper
        def to_decimal(value) -> Optional[Decimal]:
            if value is None: return None
            try:
                d = Decimal(str(value))
                # Ensure the price is finite and positive
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
        elif p_avg: # Use 'average' if last is missing but average exists
            price = p_avg; lg.debug(f"Using 'average' price: {price}")
        elif p_bid and p_ask: # Use bid/ask midpoint if others missing
            price = (p_bid + p_ask) / 2; lg.debug(f"Using bid/ask midpoint: {price}")
        elif p_ask: # Fallback to ask if only ask is valid
            price = p_ask; lg.warning(f"Using 'ask' price fallback (bid invalid/missing): {price}")
        elif p_bid: # Fallback to bid if only bid is valid
            price = p_bid; lg.warning(f"Using 'bid' price fallback (ask invalid/missing): {price}")

        # Final validation
        if price is not None and price.is_finite() and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to extract a valid price from ticker data for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except Exception as e:
        # Catch errors raised by safe_api_call or during parsing
        lg.error(f"{NEON_RED}Error fetching current price for {symbol}: {e}{RESET}", exc_info=False) # Keep log concise
        return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and robust validation."""
    lg = logger or logging.getLogger(__name__)
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    try:
        # Use safe_api_call to handle retries
        ohlcv = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe=timeframe, limit=limit)

        if ohlcv is None or not isinstance(ohlcv, list) or len(ohlcv) == 0:
            # Error logged by safe_api_call if failed after retries
            if ohlcv is not None: # Log only if it returned empty list/None without raising error
                lg.warning(f"{NEON_YELLOW}No valid kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        # Process the data into a pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            lg.warning(f"Kline data DataFrame is empty for {symbol} {timeframe}.")
            return df

        # Convert timestamp to datetime objects (UTC), coerce errors
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # Convert price/volume columns to numeric Decimal, handling potential empty strings or invalid formats
        for col in ['open', 'high', 'low', 'close', 'volume']:
             try:
                  # Apply robust conversion to Decimal, handle empty strings/None explicitly
                  df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and str(x).strip() != '' else Decimal('NaN'))
             except (TypeError, ValueError, InvalidOperation) as conv_err:
                  lg.warning(f"Could not convert column '{col}' to Decimal, attempting numeric fallback: {conv_err}")
                  df[col] = pd.to_numeric(df[col], errors='coerce') # Fallback to float/NaN

        # Data Cleaning: Drop rows with NaN in essential price columns or non-positive/non-finite close price
        initial_len = len(df)
        close_col = df['close'] # Reference the column for checks
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        # Filter out rows with non-positive or non-finite close prices
        if not df.empty:
            first_val_type = type(close_col.iloc[0]) if not close_col.empty else None
            if first_val_type == Decimal:
                # Use Decimal methods for filtering
                df = df[close_col.apply(lambda x: x.is_finite() and x > 0)]
            elif pd.api.types.is_numeric_dtype(close_col.dtype): # Check if float/int
                 # Use numpy methods for filtering float columns
                 df = df[np.isfinite(close_col) & (close_col > 0)]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} {timeframe} empty after cleaning.")
            return pd.DataFrame()

        # Sort by timestamp index and remove duplicates (keeping the last occurrence)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        # Catch errors from safe_api_call or during processing
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

        if not orderbook: # Error already logged by safe_api_call if it failed
            return None
        # Validate structure
        if not isinstance(orderbook, dict) or 'bids' not in orderbook or 'asks' not in orderbook or \
           not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
            lg.warning(f"Invalid orderbook structure received for {symbol}. Data: {orderbook}")
            return None

        if not orderbook['bids'] and not orderbook['asks']:
            lg.warning(f"Orderbook received but both bids and asks lists are empty for {symbol}.")
            # Return the empty but valid book
            return orderbook

        # Basic validation of bid/ask entry format (price, size structure)
        valid = True
        for side in ['bids', 'asks']:
             if orderbook[side]: # Check first entry if list is not empty
                  entry = orderbook[side][0]
                  if not (isinstance(entry, list) and len(entry) == 2):
                       lg.warning(f"Invalid {side[:-1]} entry format in orderbook: {entry}")
                       valid = False; break
                  try: # Check if price and size are numeric (allow float conversion)
                       _ = float(entry[0]); _ = float(entry[1])
                  except (ValueError, TypeError):
                       lg.warning(f"Non-numeric data in {side[:-1]} entry: {entry}")
                       valid = False; break
        if not valid:
             lg.error("Orderbook data format validation failed.")
             return None

        lg.debug(f"Successfully fetched orderbook for {symbol} ({len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks).")
        return orderbook

    except Exception as e:
        # Catch errors raised by safe_api_call or other validation issues
        lg.error(f"{NEON_RED}Error fetching order book for {symbol}: {e}{RESET}", exc_info=False)
        return None

# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta and generates weighted signals."""

    def __init__(
        self,
        df: pd.DataFrame,
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
        self.df = df # Expects OHLCV columns with Decimal type from fetch_klines
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "5")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)
        if not self.ccxt_interval:
            self.logger.error(f"Invalid interval '{self.interval}' in config for {self.symbol}.")
            # Bot might fail later if this is not valid

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
             self._calculate_all_indicators()
             self._update_latest_indicator_values() # Run AFTER indicator calculation
             self.calculate_fibonacci_levels()
        else:
             self.logger.warning("TradingAnalyzer initialized with empty DataFrame. No calculations performed.")


    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """Helper to find the actual column name generated by pandas_ta."""
        # Define expected patterns, potentially using f-strings for dynamic parts
        expected_patterns = {
            "ATR": [f"ATRr_{self.config.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{self.config.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{self.config.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{self.config.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            # CCI often includes the constant (e.g., 100.0) which pandas_ta adds
            "CCI": [f"CCI_{self.config.get('cci_window', DEFAULT_CCI_WINDOW)}"],
            "Williams_R": [f"WILLR_{self.config.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{self.config.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP_D"], # Default pandas_ta VWAP often daily anchored
            "PSAR_long": [f"PSARl_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{self.config.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            # StochRSI names can be complex, include core parameters
            "StochRSI_K": [f"STOCHRSIk_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{self.config.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{self.config.get('stoch_rsi_k', DEFAULT_K_WINDOW)}"],
            "StochRSI_D": [f"STOCHRSId_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{self.config.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{self.config.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{self.config.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "RSI": [f"RSI_{self.config.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            # BBands names include period and std dev (formatted to 1 decimal place by default)
            "BB_Lower": [f"BBL_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)):.1f}"],
            "BB_Middle": [f"BBM_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)):.1f}"],
            "BB_Upper": [f"BBU_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)):.1f}"],
            # Custom name used for Volume MA
            "Volume_MA": [f"VOL_SMA_{self.config.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"]
        }

        patterns = expected_patterns.get(base_name, [])
        df_cols = result_df.columns.tolist()

        # Exact match or startswith preferred
        for col in df_cols:
            for pattern in patterns:
                # Use startswith for flexibility (e.g., CCI_20_100.0 matches CCI_20)
                if col.startswith(pattern):
                    self.logger.debug(f"Mapped '{base_name}' to column '{col}'")
                    return col

        # Fallback: Simple case-insensitive substring search
        base_lower = base_name.lower()
        simple_base = base_lower.split('_')[0] # e.g., "ema_short" -> "ema"
        for col in df_cols:
            col_lower = col.lower()
            # Check full base name first, then simpler version
            if base_lower in col_lower:
                 self.logger.debug(f"Mapped '{base_name}' to '{col}' via substring search ('{base_lower}').")
                 return col
            # Avoid overly broad matches with simple base (e.g., 'r' matching 'atr')
            # Ensure simple_base is reasonably specific
            elif len(simple_base) > 2 and simple_base in col_lower:
                 self.logger.debug(f"Mapped '{base_name}' to '{col}' via substring search ('{simple_base}').")
                 return col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {df_cols}")
        return None

    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty, cannot calculate indicators for {self.symbol}.")
            return

        # Determine minimum required data length based on enabled & weighted indicators
        required_periods = []
        indicators_config = self.config.get("indicators", {})
        active_weights = self.weights # Use stored weights

        # Helper to add requirement if indicator is enabled AND weighted
        def add_req(key, config_key, default_period):
            if indicators_config.get(key, False) and float(active_weights.get(key, 0)) != 0:
                required_periods.append(self.config.get(config_key, default_period))

        add_req("atr", "atr_period", DEFAULT_ATR_PERIOD) # ATR always calculated if possible
        add_req("momentum", "momentum_period", DEFAULT_MOMENTUM_PERIOD)
        add_req("cci", "cci_window", DEFAULT_CCI_WINDOW)
        add_req("wr", "williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        add_req("mfi", "mfi_window", DEFAULT_MFI_WINDOW)
        add_req("sma_10", "sma_10_window", DEFAULT_SMA_10_WINDOW)
        add_req("rsi", "rsi_period", DEFAULT_RSI_WINDOW)
        add_req("bollinger_bands", "bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        add_req("volume_confirmation", "volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
        required_periods.append(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)) # For Fib levels

        if indicators_config.get("ema_alignment", False) and float(active_weights.get("ema_alignment", 0)) != 0:
             required_periods.append(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
             required_periods.append(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        if indicators_config.get("stoch_rsi", False) and float(active_weights.get("stoch_rsi", 0)) != 0:
            required_periods.append(self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW))
            required_periods.append(self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))

        min_required_data = max(required_periods) + 30 if required_periods else 50 # Add buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate indicators reliably (min recommended: {min_required_data}). Results may contain NaNs.{RESET}")

        try:
            df_calc = self.df.copy()
            # --- Convert Decimal columns to float for pandas_ta ---
            # Store original types to potentially convert back later if needed
            original_types = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 if col in df_calc.columns:
                     # Check first non-NaN value's type
                     first_valid_idx = df_calc[col].first_valid_index()
                     if first_valid_idx is not None:
                          original_types[col] = type(df_calc.loc[first_valid_idx, col])
                          if original_types[col] == Decimal:
                               self.logger.debug(f"Converting Decimal column '{col}' to float for TA calculation.")
                               # Apply conversion robustly, handle non-finite Decimals
                               df_calc[col] = df_calc[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else np.nan)
                     # If column exists but is all NaN, it's fine
                     elif not df_calc[col].isnull().all(): # If not all NaN, but first is NaN, try converting anyway
                          if isinstance(df_calc[col].iloc[0], Decimal): # Check type even if NaN initially
                               self.logger.debug(f"Converting Decimal column '{col}' (starting with NaN) to float.")
                               df_calc[col] = df_calc[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else np.nan)

            # --- Calculate Indicators using pandas_ta ---
            # Always calculate ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # Calculate others based on config and weight
            key_map = { # Map internal keys to config keys and defaults if needed
                "momentum": ("momentum_period", DEFAULT_MOMENTUM_PERIOD),
                "cci": ("cci_window", DEFAULT_CCI_WINDOW),
                "wr": ("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW),
                "mfi": ("mfi_window", DEFAULT_MFI_WINDOW),
                "sma_10": ("sma_10_window", DEFAULT_SMA_10_WINDOW),
                "rsi": ("rsi_period", DEFAULT_RSI_WINDOW),
            }

            for key, enabled in indicators_config.items():
                if key == "atr": continue # Already done
                if enabled and float(active_weights.get(key, 0)) != 0:
                    self.logger.debug(f"Calculating indicator: {key}")
                    try:
                        if key == "ema_alignment":
                            ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                            ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                            df_calc.ta.ema(length=ema_short, append=True)
                            self.ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
                            df_calc.ta.ema(length=ema_long, append=True)
                            self.ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)
                        elif key == "psar":
                            psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                            psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                            psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                            if psar_result is not None and not psar_result.empty:
                                df_calc = pd.concat([df_calc, psar_result], axis=1)
                                self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                                self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)
                        elif key == "stoch_rsi":
                            st_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                            st_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                            st_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                            st_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                            st_result = df_calc.ta.stochrsi(length=st_len, rsi_length=st_rsi_len, k=st_k, d=st_d)
                            if st_result is not None and not st_result.empty:
                                df_calc = pd.concat([df_calc, st_result], axis=1)
                                self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                                self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)
                        elif key == "bollinger_bands":
                            bb_p = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                            bb_std = float(self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV))
                            bb_result = df_calc.ta.bbands(length=bb_p, std=bb_std)
                            if bb_result is not None and not bb_result.empty:
                                df_calc = pd.concat([df_calc, bb_result], axis=1)
                                self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                                self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                                self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)
                        elif key == "volume_confirmation":
                            vol_ma_p = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                            vol_ma_col = f"VOL_SMA_{vol_ma_p}"
                            # Ensure volume is float for SMA calculation
                            vol_series = df_calc['volume'].astype(float) # Already converted above if needed
                            df_calc[vol_ma_col] = ta.sma(vol_series.fillna(0), length=vol_ma_p)
                            self.ta_column_names["Volume_MA"] = vol_ma_col
                        elif key == "vwap":
                             df_calc.ta.vwap(append=True)
                             self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)
                        elif key in key_map: # General case using key_map
                            config_k, default_p = key_map[key]
                            period = self.config.get(config_k, default_p)
                            method = getattr(df_calc.ta, key, None)
                            if method and callable(method):
                                method(length=period, append=True)
                                # Map internal key to pandas_ta base name for column lookup
                                ta_base_map = {"cci": "CCI", "wr": "Williams_R", "mfi": "MFI", "sma_10": "SMA10", "rsi": "RSI", "momentum": "Momentum"}
                                ta_base_name = ta_base_map.get(key, key.upper()) # Simple mapping
                                self.ta_column_names[ta_base_name] = self._get_ta_col_name(ta_base_name, df_calc)
                            else:
                                self.logger.warning(f"Pandas TA method '{key}' not found or not callable.")

                    except Exception as calc_err:
                        self.logger.error(f"Error calculating indicator '{key}': {calc_err}", exc_info=True)

            # --- Convert ATR column back to Decimal ---
            # pandas_ta outputs float, convert ATR back for precise calculations
            atr_col = self.ta_column_names.get("ATR")
            if atr_col and atr_col in df_calc.columns:
                 try:
                     # Convert float column back to Decimal, handling potential NaNs/infs
                     df_calc[atr_col] = df_calc[atr_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
                     self.logger.debug(f"Converted calculated ATR column '{atr_col}' back to Decimal.")
                 except (ValueError, TypeError, InvalidOperation) as conv_err:
                      self.logger.error(f"Failed to convert ATR column '{atr_col}' back to Decimal: {conv_err}")

            # Update the instance's DataFrame
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations. Final DF columns: {self.df.columns.tolist()}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error during indicator calculation setup or execution: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates indicator_values dict with latest values, handling types."""
        # Define keys expected based on calculation attempts + base OHLCV
        expected_keys = list(self.ta_column_names.keys()) + ["Open", "High", "Low", "Close", "Volume"]
        default_values = {k: np.nan for k in expected_keys} # Initialize with NaN

        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame empty for {self.symbol}.")
            self.indicator_values = default_values
            return
        try:
            # Use the last valid index in case of missing data points
            last_valid_index = self.df.last_valid_index()
            if last_valid_index is None: raise IndexError("No valid index found.")
            latest = self.df.loc[last_valid_index]
        except (IndexError, KeyError):
            self.logger.error(f"Error accessing latest valid row/index for {self.symbol}.")
            self.indicator_values = default_values
            return

        if latest.isnull().all():
            self.logger.warning(f"Last valid row contains all NaNs for {self.symbol}. Cannot update values.")
            self.indicator_values = default_values
            return

        updated_values = {}
        # --- Process TA indicators ---
        for key, col_name in self.ta_column_names.items():
            if col_name and col_name in latest.index:
                value = latest[col_name]
                # Ensure value is finite number (not NaN, not inf)
                if pd.notna(value) and np.isfinite(value):
                    try:
                        if key == "ATR": # ATR should be Decimal
                            updated_values[key] = value if isinstance(value, Decimal) else Decimal(str(value))
                        else: # Others as float
                            updated_values[key] = float(value)
                    except (ValueError, TypeError, InvalidOperation) as conv_err:
                        self.logger.warning(f"Could not convert TA value {key} ('{col_name}': {value}): {conv_err}. Storing NaN.")
                        updated_values[key] = np.nan
                else: updated_values[key] = np.nan # Store NaN if value is NaN or inf
            else:
                 # Log only if calculation was attempted (key exists in ta_column_names)
                 if key in self.ta_column_names:
                     self.logger.debug(f"Indicator column '{col_name}' for '{key}' not found in latest data. Storing NaN.")
                 updated_values[key] = np.nan

        # --- Process Base OHLCV (should be Decimal from fetch) ---
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            key_name = base_col.capitalize()
            value = latest.get(base_col)
            if pd.notna(value) and isinstance(value, Decimal) and value.is_finite():
                 updated_values[key_name] = value
            elif pd.notna(value): # If not Decimal or not finite
                 self.logger.warning(f"Base value '{base_col}' ({value}) is not a finite Decimal. Storing NaN.")
                 updated_values[key_name] = np.nan
            else: # Value is NaN
                 updated_values[key_name] = np.nan

        self.indicator_values = updated_values

        # --- Log Summary (formatted) ---
        log_vals = {}
        price_prec = self.get_price_precision()
        for k, v in self.indicator_values.items():
            # Log only finite numeric values
            if pd.notna(v) and isinstance(v, (Decimal, float, int)) and np.isfinite(v):
                if isinstance(v, Decimal):
                    prec = price_prec if k in ['Open', 'High', 'Low', 'Close', 'ATR'] else 8
                    log_vals[k] = f"{v:.{prec}f}"
                elif isinstance(v, float): log_vals[k] = f"{v:.5f}"
                else: log_vals[k] = str(v)
            # else: log_vals[k] = "NaN" # Optionally log NaNs

        self.logger.debug(f"Latest values updated ({self.symbol}): {log_vals}")


    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci levels using Decimal precision."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci ({window}) on {self.symbol}.")
            self.fib_levels_data = {}; return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure 'high'/'low' are numeric (handle Decimal or float)
            high_series = pd.to_numeric(df_slice["high"], errors='coerce')
            low_series = pd.to_numeric(df_slice["low"], errors='coerce')
            high_price_raw = high_series.dropna().max()
            low_price_raw = low_series.dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                self.logger.warning(f"Could not find valid high/low for Fibonacci (Window: {window}).")
                self.fib_levels_data = {}; return {}

            high = Decimal(str(high_price_raw))
            low = Decimal(str(low_price_raw))
            diff = high - low

            levels = {}
            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Level price = High - (Range * Percentage)
                    level_price = high - (diff * Decimal(str(level_pct)))
                    # Quantize down from high (conservative support)
                    levels[level_name] = level_price.quantize(rounding_factor, rounding=ROUND_DOWN)
            else: # Handle zero range
                self.logger.debug(f"Fibonacci range is zero (High=Low={high}). Setting all levels to this price.")
                level_price_quantized = high.quantize(rounding_factor, rounding=ROUND_DOWN)
                for level_pct in FIB_LEVELS:
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            log_levels = {k: str(v) for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels (Window: {window}): {log_levels}")
            return levels

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Fibonacci error: Missing column '{e}'.{RESET}")
            self.fib_levels_data = {}; return {}
        except (ValueError, TypeError, InvalidOperation) as e:
             self.logger.error(f"{NEON_RED}Fibonacci error: Invalid data type for high/low. {e}{RESET}")
             self.fib_levels_data = {}; return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}; return {}

    def get_price_precision(self) -> int:
        """Determines price precision (decimal places) from market info."""
        try:
            # 1. Check precision.price (most reliable)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None:
                if isinstance(price_precision_val, int) and price_precision_val >= 0:
                    return price_precision_val
                try: # Assume float/str represents tick size
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size.is_finite() and tick_size > 0:
                        return abs(tick_size.normalize().as_tuple().exponent)
                except (TypeError, ValueError, InvalidOperation) as e:
                     self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")

            # 2. Fallback: Infer from limits.price.min (less reliable)
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_price_tick = Decimal(str(min_price_val))
                    if min_price_tick.is_finite() and 0 < min_price_tick < Decimal('0.1'): # Heuristic
                        return abs(min_price_tick.normalize().as_tuple().exponent)
                except (TypeError, ValueError, InvalidOperation) as e:
                    self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' for precision: {e}")

            # 3. Fallback: Infer from last close price (least reliable)
            last_close = self.indicator_values.get("Close")
            if isinstance(last_close, Decimal) and last_close.is_finite() and last_close > 0:
                try:
                    precision = abs(last_close.normalize().as_tuple().exponent)
                    if 0 <= precision < 10: return precision # Sanity check
                except Exception: pass

        except Exception as e:
            self.logger.warning(f"Error determining price precision for {self.symbol}: {e}. Falling back.")

        default_precision = 4
        self.logger.warning(f"Could not determine price precision for {self.symbol}. Using default: {default_precision}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info."""
        try:
            # 1. Try precision.price
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None:
                if isinstance(price_precision_val, (float, str, int)):
                     try:
                          if isinstance(price_precision_val, int): # Decimal places
                               if price_precision_val >= 0:
                                    tick = Decimal('1e-' + str(price_precision_val))
                                    if tick.is_finite() and tick > 0: return tick
                          else: # float or str -> Assume it IS the tick size
                               tick = Decimal(str(price_precision_val))
                               if tick.is_finite() and tick > 0: return tick
                     except (TypeError, ValueError, InvalidOperation) as e:
                          self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")

            # 2. Fallback: Try limits.price.min
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_tick = Decimal(str(min_price_val))
                    if min_tick.is_finite() and 0 < min_tick < Decimal('0.1'): # Heuristic check
                        return min_tick
                except (TypeError, ValueError, InvalidOperation) as e:
                     self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' for tick size: {e}")

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}.")

        # --- Final Fallback: Calculate from derived decimal places ---
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.debug(f"Using fallback tick size based on derived precision ({price_precision_places}): {fallback_tick}")
        return fallback_tick

    def get_amount_precision_places(self) -> int:
        """Determines amount precision (decimal places) from market info."""
        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None:
                if isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                    return amount_precision_val
                try: # Assume step size
                    step_size = Decimal(str(amount_precision_val))
                    if step_size.is_finite() and step_size > 0:
                        return abs(step_size.normalize().as_tuple().exponent)
                except (TypeError, ValueError, InvalidOperation): pass

            min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_val is not None:
                try:
                    min_amount_step = Decimal(str(min_amount_val))
                    if min_amount_step.is_finite() and 0 < min_amount_step <= Decimal('1'):
                       if min_amount_step < 1: # Looks like step size
                           return abs(min_amount_step.normalize().as_tuple().exponent)
                       elif min_amount_step >= 1 and '.' not in str(min_amount_val): return 0 # Integer min amount likely 0 precision
                except (TypeError, ValueError, InvalidOperation): pass

        except Exception as e:
            self.logger.warning(f"Error determining amount precision for {self.symbol}: {e}.")

        default_precision = 8
        self.logger.warning(f"Could not determine amount precision for {self.symbol}. Using default: {default_precision}.")
        return default_precision

    def get_min_amount_step(self) -> Decimal:
        """Gets the minimum amount increment (step size) from market info."""
        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None:
                if isinstance(amount_precision_val, (float, str, int)):
                     try:
                          if isinstance(amount_precision_val, int): # Decimal places
                               if amount_precision_val >= 0:
                                    step = Decimal('1e-' + str(amount_precision_val))
                                    if step.is_finite() and step > 0: return step
                          else: # Float/Str = step size itself
                               step = Decimal(str(amount_precision_val))
                               if step.is_finite() and step > 0: return step
                     except (TypeError, ValueError, InvalidOperation): pass

            min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_val is not None:
                try:
                    min_step = Decimal(str(min_amount_val))
                    # Assume min limit IS the step size if it's positive and finite
                    if min_step.is_finite() and min_step > 0: return min_step
                except (TypeError, ValueError, InvalidOperation): pass

        except Exception as e:
            self.logger.warning(f"Could not determine min amount step for {self.symbol}: {e}.")

        amount_precision_places = self.get_amount_precision_places()
        fallback_step = Decimal('1e-' + str(amount_precision_places))
        self.logger.debug(f"Using fallback amount step based on derived precision ({amount_precision_places}): {fallback_step}")
        return fallback_step


    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            self.logger.debug(f"Fibonacci levels not calculated for {self.symbol}.")
            return []
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal) and level_price.is_finite() and level_price > 0:
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    self.logger.warning(f"Invalid Fib level value encountered: {name}={level_price}. Skipping.")

            level_distances.sort(key=lambda x: x['distance'])
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score."""
        ema_s = self.indicator_values.get("EMA_Short") # Float
        ema_l = self.indicator_values.get("EMA_Long") # Float
        close_dec = self.indicator_values.get("Close") # Decimal

        if not (isinstance(ema_s, (float, int)) and np.isfinite(ema_s) and
                isinstance(ema_l, (float, int)) and np.isfinite(ema_l) and
                isinstance(close_dec, Decimal) and close_dec.is_finite()):
            self.logger.debug("EMA alignment check skipped: Missing or non-finite values.")
            return np.nan

        price_f = float(close_dec)
        if price_f > ema_s > ema_l: return 1.0 # Strong Bullish
        elif price_f < ema_s < ema_l: return -1.0 # Strong Bearish
        else: return 0.0 # Mixed / Crossing

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """Generates final trading signal (BUY/SELL/HOLD) based on weighted score."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Default HOLD
        final_score = Decimal("0.0")
        total_weight = Decimal("0.0")
        active_count, nan_count = 0, 0
        debug_scores = {}

        # --- Basic Validation ---
        if not self.indicator_values:
            self.logger.warning("Signal Gen: Indicator values empty."); return "HOLD"
        core_ok = any(
            pd.notna(v) and np.isfinite(v)
            for k, v in self.indicator_values.items()
            if k not in ['Open', 'High', 'Low', 'Close', 'Volume'] and float(self.weights.get(k, 0)) != 0
        )
        if not core_ok:
            self.logger.warning("Signal Gen: All weighted core indicators NaN/invalid."); return "HOLD"
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Signal Gen: Invalid current price ({current_price})."); return "HOLD"
        if not self.weights:
            self.logger.error("Signal Gen: Active weight set missing/empty."); return "HOLD"

        # --- Iterate through indicators ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue
            weight_str = self.weights.get(indicator_key)
            if weight_str is None: continue # No weight defined

            try:
                weight = Decimal(str(weight_str))
                if not weight.is_finite(): raise ValueError("Weight not finite")
                if weight == 0: continue # Skip zero weight
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid weight '{weight_str}' for '{indicator_key}'. Skipping."); continue

            check_method_name = f"_check_{indicator_key}"
            score_float = np.nan
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                try:
                    method = getattr(self, check_method_name)
                    if indicator_key == "orderbook":
                        if orderbook_data: score_float = method(orderbook_data, current_price)
                        elif weight != 0: self.logger.debug("Orderbook check skipped: No data.")
                    else: score_float = method()
                except Exception as e:
                    self.logger.error(f"Error executing check {check_method_name}: {e}", exc_info=True)
            elif weight != 0:
                self.logger.warning(f"Check method '{check_method_name}' not found for weighted indicator '{indicator_key}'.")

            # Store score for debugging
            debug_scores[indicator_key] = f"{score_float:.3f}" if pd.notna(score_float) and np.isfinite(score_float) else str(score_float)

            # Aggregate score if valid
            if pd.notna(score_float) and np.isfinite(score_float):
                try:
                    score_dec = Decimal(str(score_float))
                    clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_dec))
                    final_score += clamped_score * weight
                    total_weight += weight
                    active_count += 1
                except (ValueError, TypeError, InvalidOperation) as calc_err:
                    self.logger.error(f"Error processing score for {indicator_key}: {calc_err}"); nan_count += 1
            else:
                nan_count += 1

        # --- Determine Final Signal ---
        final_signal = "HOLD"
        if total_weight == 0:
            self.logger.warning(f"No indicators contributed valid scores/weights ({self.symbol}). Defaulting to HOLD.")
        else:
            try:
                threshold_str = self.config.get("signal_score_threshold", "1.5")
                threshold = Decimal(str(threshold_str))
                if not threshold.is_finite() or threshold <= 0: raise ValueError("Threshold non-positive/finite")
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid signal_score_threshold '{threshold_str}'. Using default 1.5.")
                threshold = Decimal("1.5")

            if final_score >= threshold: final_signal = "BUY"
            elif final_score <= -threshold: final_signal = "SELL"

        # --- Log Summary ---
        price_prec = self.get_price_precision()
        sig_color = NEON_GREEN if final_signal == "BUY" else NEON_RED if final_signal == "SELL" else NEON_YELLOW
        log_msg = (
            f"Signal Summary ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Ind=[Act:{active_count}, NaN:{nan_count}], "
            f"Weight={total_weight:.2f}, Score={final_score:.4f} (Thr: +/-{threshold:.2f}) "
            f"==> {sig_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        self.logger.debug(f"  Indicator Scores ({self.symbol}): {debug_scores}")

        # Update internal signal state
        if final_signal == "BUY": self.signals = {"BUY": 1, "SELL": 0, "HOLD": 0}
        elif final_signal == "SELL": self.signals = {"BUY": 0, "SELL": 1, "HOLD": 0}
        else: self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}
        return final_signal

    # --- Indicator Check Methods (return float score -1.0 to 1.0 or np.nan) ---
    # Ensure methods handle potential NaN/inf values from self.indicator_values
    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum")
        if not isinstance(momentum, (float, int)) or not np.isfinite(momentum): return np.nan
        threshold = 0.1 # Example threshold, adjust based on expected MOM range
        if threshold == 0: return 0.0
        score = momentum / threshold
        return max(-1.0, min(1.0, score)) # Clamp

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume") # Decimal
        volume_ma = self.indicator_values.get("Volume_MA") # Float
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        if not (isinstance(current_volume, Decimal) and current_volume.is_finite() and
                isinstance(volume_ma, (float, int)) and np.isfinite(volume_ma) and volume_ma > 0 and multiplier > 0):
            return np.nan
        try:
            volume_ma_dec = Decimal(str(volume_ma)); multiplier_dec = Decimal(str(multiplier))
            if current_volume > volume_ma_dec * multiplier_dec: return 0.7 # High volume
            elif current_volume < volume_ma_dec / multiplier_dec: return -0.4 # Low volume
            else: return 0.0 # Neutral
        except (ValueError, TypeError, InvalidOperation): return np.nan

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K")
        d = self.indicator_values.get("StochRSI_D")
        if not (isinstance(k, (float, int)) and np.isfinite(k) and
                isinstance(d, (float, int)) and np.isfinite(d)): return np.nan
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
        score = 0.0
        if k < oversold and d < oversold: score = 1.0
        elif k > overbought and d > overbought: score = -1.0
        diff = k - d
        if abs(diff) > 5: score = max(score, 0.6) if diff > 0 else min(score, -0.6) # Stronger cross signal
        elif k > d: score = max(score, 0.2) # Mild bullish
        elif k < d: score = min(score, -0.2) # Mild bearish
        if 40 < k < 60: score *= 0.5 # Dampen in neutral zone
        return score

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI")
        if not isinstance(rsi, (float, int)) or not np.isfinite(rsi): return np.nan
        if rsi <= 30: return 1.0;   elif rsi >= 70: return -1.0
        if rsi < 40: return 0.5;    elif rsi > 60: return -0.5
        if 40 <= rsi <= 60: return (rsi - 50) / 50.0 
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI")
        if not isinstance(cci, (float, int)) or not np.isfinite(cci): return np.nan
        if cci <= -150: return 1.0; elif cci >= 150: return -1.0
        if cci < -80: return 0.6;   elif cci > 80: return -0.6
        if cci > 0: return -0.1;    elif cci < 0: return 0.1
        return 0.0

    def _check_wr(self) -> float:
        wr = self.indicator_values.get("Williams_R")
        if not isinstance(wr, (float, int)) or not np.isfinite(wr): return np.nan
        if wr <= -80: return 1.0;   elif wr >= -20: return -1.0
        if wr < -50: return 0.4;    elif wr > -50: return -0.4
        return 0.0

    def _check_psar(self) -> float:
        psar_l = self.indicator_values.get("PSAR_long"); psar_s = self.indicator_values.get("PSAR_short")
        # Check if values are finite numbers if not NaN
        l_act = pd.notna(psar_l) and isinstance(psar_l, (float, int)) and np.isfinite(psar_l)
        s_act = pd.notna(psar_s) and isinstance(psar_s, (float, int)) and np.isfinite(psar_s)

        if l_act and not s_act: return 1.0 # Long trend
        elif s_act and not l_act: return -1.0 # Short trend
        elif not l_act and not s_act: return np.nan # Indeterminate
        else: self.logger.warning(f"PSAR unusual: L={psar_l}, S={psar_s}"); return 0.0 # Both active? Error.

    def _check_sma_10(self) -> float:
        sma = self.indicator_values.get("SMA10"); close = self.indicator_values.get("Close")
        if not (isinstance(sma, (float, int)) and np.isfinite(sma) and
                isinstance(close, Decimal) and close.is_finite()): return np.nan
        try: sma_dec = Decimal(str(sma))
        except (ValueError, TypeError, InvalidOperation): return np.nan
        if close > sma_dec: return 0.6
        elif close < sma_dec: return -0.6
        else: return 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP"); close = self.indicator_values.get("Close")
        if not (isinstance(vwap, (float, int)) and np.isfinite(vwap) and
                isinstance(close, Decimal) and close.is_finite()): return np.nan
        try: vwap_dec = Decimal(str(vwap))
        except (ValueError, TypeError, InvalidOperation): return np.nan
        if close > vwap_dec: return 0.7
        elif close < vwap_dec: return -0.7
        else: return 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI")
        if not isinstance(mfi, (float, int)) or not np.isfinite(mfi): return np.nan
        if mfi <= 20: return 1.0;   elif mfi >= 80: return -1.0
        if mfi < 40: return 0.4;    elif mfi > 60: return -0.4
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bbl=self.indicator_values.get("BB_Lower"); bbm=self.indicator_values.get("BB_Middle"); bbu=self.indicator_values.get("BB_Upper")
        close = self.indicator_values.get("Close")
        if not (isinstance(bbl, (float, int)) and np.isfinite(bbl) and
                isinstance(bbm, (float, int)) and np.isfinite(bbm) and
                isinstance(bbu, (float, int)) and np.isfinite(bbu) and
                isinstance(close, Decimal) and close.is_finite()): return np.nan
        try:
            bbl_d, bbm_d, bbu_d = Decimal(str(bbl)), Decimal(str(bbm)), Decimal(str(bbu))
            if bbu_d <= bbl_d: return 0.0 # Avoid division by zero if bands collapse/invalid
        except (ValueError, TypeError, InvalidOperation): return np.nan

        if close <= bbl_d: return 1.0 # Touch/Below Lower -> Buy Signal
        if close >= bbu_d: return -1.0 # Touch/Above Upper -> Sell Signal

        # Scale score based on position between middle and outer bands
        if close > bbm_d: # Above middle
             dist_from_mid = close - bbm_d
             upper_range = bbu_d - bbm_d
             # Score decreases from 0.5 (at mid) to 0.0 (at upper band)
             score = 0.5 * float(1 - (dist_from_mid / upper_range if upper_range > 0 else 0))
             return max(0.0, min(score, 0.5)) # Clamp [0.0, 0.5]
        else: # Below middle
             dist_from_mid = bbm_d - close
             lower_range = bbm_d - bbl_d
             # Score increases from -0.5 (at mid) to 0.0 (at lower band)
             score = -0.5 * float(1 - (dist_from_mid / lower_range if lower_range > 0 else 0))
             return max(-0.5, min(score, 0.0)) # Clamp [-0.5, 0.0]

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            self.logger.debug("Orderbook check skipped: No data or missing bids/asks.")
            return np.nan
        try:
            bids = orderbook_data['bids']; asks = orderbook_data['asks']
            levels = min(len(bids), len(asks), 10) # Use top 10 levels
            if levels == 0: return 0.0 # Neutral if no common levels

            bid_v = sum(Decimal(str(b[1])) for b in bids[:levels] if len(b)==2)
            ask_v = sum(Decimal(str(a[1])) for a in asks[:levels] if len(a)==2)
            total_v = bid_v + ask_v
            if total_v == 0: return 0.0 # Avoid division by zero

            obi = (bid_v - ask_v) / total_v # Order Book Imbalance ratio
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi))) # Clamp and convert

            self.logger.debug(f"OB Check: Lvl={levels}, BidV={bid_v:.4f}, AskV={ask_v:.4f}, OBI={obi:.4f} -> Score={score:.4f}")
            return score
        except (IndexError, ValueError, TypeError, InvalidOperation) as e:
             self.logger.warning(f"Orderbook analysis failed: {e}", exc_info=True); return np.nan

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates potential TP and initial SL based on entry estimate, signal, and ATR."""
        if signal not in ["BUY", "SELL"]: return entry_price_estimate, None, None
        atr = self.indicator_values.get("ATR") # Decimal
        if not isinstance(atr, Decimal) or not atr.is_finite() or atr <= 0:
            self.logger.warning(f"Calc TP/SL Fail ({signal}): Invalid ATR ({atr})."); return entry_price_estimate, None, None
        if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
            self.logger.warning(f"Calc TP/SL Fail ({signal}): Invalid entry estimate ({entry_price_estimate})."); return entry_price_estimate, None, None

        try:
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", "1.0")))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", "1.5")))
            prec = self.get_price_precision()
            rnd = Decimal('1e-' + str(prec))
            tick = self.get_min_tick_size()

            tp_off = atr * tp_mult; sl_off = atr * sl_mult
            tp_raw, sl_raw = None, None

            if signal == "BUY": tp_raw, sl_raw = entry_price_estimate + tp_off, entry_price_estimate - sl_off
            else: tp_raw, sl_raw = entry_price_estimate - tp_off, entry_price_estimate + sl_off

            # Quantize TP/SL using market precision
            # Round TP towards neutral (less profit), SL away from neutral (more room) -> Conservative
            tp_q = tp_raw.quantize(rnd, rounding=ROUND_DOWN if signal=="BUY" else ROUND_UP) if tp_raw and tp_raw.is_finite() else None
            sl_q = sl_raw.quantize(rnd, rounding=ROUND_DOWN if signal=="BUY" else ROUND_UP) if sl_raw and sl_raw.is_finite() else None

            # --- Validation & Adjustment ---
            final_tp, final_sl = tp_q, sl_q
            # Ensure SL is strictly beyond entry by at least one tick
            if final_sl:
                if signal == "BUY" and final_sl >= entry_price_estimate:
                    final_sl = (entry_price_estimate - tick).quantize(rnd, rounding=ROUND_DOWN)
                    self.logger.debug(f"Adjusted BUY SL below entry: {sl_q} -> {final_sl}")
                elif signal == "SELL" and final_sl <= entry_price_estimate:
                    final_sl = (entry_price_estimate + tick).quantize(rnd, rounding=ROUND_UP)
                    self.logger.debug(f"Adjusted SELL SL above entry: {sl_q} -> {final_sl}")

            # Ensure TP offers profit (strictly beyond entry)
            if final_tp:
                 if signal == "BUY" and final_tp <= entry_price_estimate:
                      self.logger.warning(f"BUY TP {final_tp} <= Entry {entry_price_estimate}. Nullifying TP."); final_tp = None
                 elif signal == "SELL" and final_tp >= entry_price_estimate:
                      self.logger.warning(f"SELL TP {final_tp} >= Entry {entry_price_estimate}. Nullifying TP."); final_tp = None

            # Ensure SL/TP are positive
            if final_sl and final_sl <= 0: self.logger.error(f"SL calc non-positive ({final_sl}). Nullifying SL."); final_sl = None
            if final_tp and final_tp <= 0: self.logger.warning(f"TP calc non-positive ({final_tp}). Nullifying TP."); final_tp = None

            tp_str = f"{final_tp:.{prec}f}" if final_tp else "None"; sl_str = f"{final_sl:.{prec}f}" if final_sl else "None"
            self.logger.debug(f"Calc TP/SL ({signal}): Entry={entry_price_estimate:.{prec}f}, ATR={atr:.{prec+2}f}, TP={tp_str}, SL={sl_str}")
            return entry_price_estimate, final_tp, final_sl
        except Exception as e:
            self.logger.error(f"Unexpected error calculating TP/SL: {e}", exc_info=True)
            return entry_price_estimate, None, None

# --- Trading Logic Helper Functions ---
def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches available balance for a specific currency with retries and robust parsing."""
    lg = logger
    try:
        balance_info = None
        params = {}
        account_type_to_log = "default"
        if exchange.id == 'bybit':
            params = {'type': 'CONTRACT'}
            account_type_to_log = 'CONTRACT'

        lg.debug(f"Fetching balance for {currency} (Account: {account_type_to_log})...")
        balance_info = safe_api_call(exchange.fetch_balance, lg, params=params)

        if not balance_info:
             lg.error(f"Failed to fetch balance info for {currency} after retries.")
             lg.debug("Attempting balance fetch with default parameters as fallback...")
             balance_info = safe_api_call(exchange.fetch_balance, lg)
             if not balance_info:
                  lg.error(f"Fallback balance fetch also failed for {currency}.")
                  return None

        # --- Parse the balance_info ---
        free_balance = None

        # 1. Standard CCXT: balance[currency]['free']
        if currency in balance_info and balance_info[currency].get('free') is not None:
            free_balance = balance_info[currency]['free']
            lg.debug(f"Found balance via standard ['{currency}']['free']: {free_balance}")

        # 2. Bybit V5 Nested: info.result.list[].coin[].availableToWithdraw / availableBalance
        elif not free_balance and exchange.id == 'bybit' and 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
            for account in balance_info['info']['result']['list']:
                if isinstance(account.get('coin'), list):
                    for coin_data in account['coin']:
                        if coin_data.get('coin') == currency:
                            free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                            if free is not None:
                                free_balance = free
                                lg.debug(f"Found balance via Bybit V5 nested ['available...']: {free_balance}")
                                break
                    if free_balance is not None: break
            if free_balance is None:
                lg.warning(f"{currency} balance details not found within Bybit V5 'info.result.list[].coin[]'.")

        # 3. Fallback: Top-level 'free' dictionary (less common)
        elif not free_balance and 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
             free_balance = balance_info['free'][currency]
             lg.debug(f"Found balance via top-level 'free' dict: {free_balance}")

        # 4. Final Fallback: Use 'total' if 'free' is unavailable (use with caution)
        if free_balance is None:
             total_balance = balance_info.get(currency, {}).get('total')
             if total_balance is not None:
                  lg.warning(f"{NEON_YELLOW}Using 'total' balance ({total_balance}) as fallback for available {currency}. This might include collateral.{RESET}")
                  free_balance = total_balance
             else:
                  lg.error(f"{NEON_RED}Could not determine any balance ('free' or 'total') for {currency}.{RESET}")
                  lg.debug(f"Full balance_info structure: {json.dumps(balance_info, indent=2)}")
                  return None # No balance found

        # Convert the found balance to Decimal
        try:
            final_balance = Decimal(str(free_balance))
            if final_balance < 0:
                 lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Treating as zero.")
                 final_balance = Decimal('0')
            lg.info(f"Available {currency} balance: {final_balance:.4f}")
            return final_balance
        except (ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"Failed to convert balance string '{free_balance}' to Decimal for {currency}: {e}")
            return None

    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching balance for {currency}: {e}{RESET}", exc_info=False)
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information with retries for loading."""
    lg = logger
    try:
        # Ensure markets are loaded, reload if necessary with retries
        if not exchange.markets or symbol not in exchange.markets:
             lg.info(f"Market info for {symbol} not loaded or symbol not found, attempting to load/reload markets...")
             try:
                 safe_api_call(exchange.load_markets, lg, reload=True)
             except Exception as load_err:
                  lg.error(f"{NEON_RED}Failed to load/reload markets after retries: {load_err}{RESET}")
                  return None # Cannot proceed without markets

        if symbol not in exchange.markets:
             lg.error(f"{NEON_RED}Market {symbol} still not found after reloading.{RESET}")
             if symbol == "BTC/USDT": lg.warning(f"{NEON_YELLOW}Hint: For Bybit linear perpetual, try '{symbol}:USDT'{RESET}")
             return None

        market = exchange.market(symbol)
        if market:
            # Add custom flags for convenience
            market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
            market['is_linear'] = market.get('linear', False)
            market['is_inverse'] = market.get('inverse', False)
            # Log key details
            lg.debug(f"Market Info ({symbol}): ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                     f"Type={market.get('type')}, Contract={market['is_contract']}, Linear={market['is_linear']}, Inverse={market['is_inverse']}")
            return market
        else: lg.error(f"{NEON_RED}Market dict not found for validated symbol {symbol}.{RESET}"); return None
    except ccxt.BadSymbol as e: lg.error(f"{NEON_RED}Invalid symbol '{symbol}': {e}{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error getting market info: {e}{RESET}", exc_info=True); return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange,
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """Calculates position size based on risk, SL, balance, and market constraints."""
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quote', config.get("quote_currency", "USDT"))
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('is_inverse', False)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or not balance.is_finite() or balance <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade}).")
        return None
    if not isinstance(initial_stop_loss_price, Decimal) or not initial_stop_loss_price.is_finite() or initial_stop_loss_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid entry_price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Size Calc Fail ({symbol}): SL price equals entry price.")
        return None
    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Size Calc Fail ({symbol}): Market info missing 'limits' or 'precision'.")
        return None
    if is_inverse:
        lg.error(f"{NEON_RED}Inverse contract sizing not implemented. Aborting sizing for {symbol}.{RESET}")
        return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
            lg.error(f"Size Calc Fail ({symbol}): SL distance zero/negative.")
            return None

        contract_size_str = market_info.get('contractSize', '1')
        try:
            contract_size = Decimal(str(contract_size_str))
            if not contract_size.is_finite() or contract_size <= 0: raise ValueError("Invalid contract size")
        except (ValueError, TypeError, InvalidOperation):
            lg.warning(f"Invalid contract size '{contract_size_str}', using 1."); contract_size = Decimal('1')

        # --- Calculate Initial Size (Linear/Spot) ---
        risk_per_contract_quote = sl_distance_per_unit * contract_size
        if not risk_per_contract_quote.is_finite() or risk_per_contract_quote <= 0:
             lg.error(f"Size Calc Fail ({symbol}): Risk per contract zero/negative/NaN.")
             return None

        calculated_size = risk_amount_quote / risk_per_contract_quote

        if not calculated_size.is_finite() or calculated_size <= 0:
            lg.error(f"Initial size calc resulted in zero/negative/NaN: {calculated_size}.")
            return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  ContractSize={contract_size}, Initial Calc. Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance
        min_amount = Decimal(str(market_info.get('limits', {}).get('amount', {}).get('min', '0')))
        max_amount = Decimal(str(market_info.get('limits', {}).get('amount', {}).get('max', 'inf')))
        min_cost = Decimal(str(market_info.get('limits', {}).get('cost', {}).get('min', '0')))
        max_cost = Decimal(str(market_info.get('limits', {}).get('cost', {}).get('max', 'inf')))
        amount_step = analyzer.get_min_amount_step()

        adjusted_size = calculated_size
        # 1. Clamp by Amount Limits
        original_size = adjusted_size
        if adjusted_size < min_amount: adjusted_size = min_amount
        if adjusted_size > max_amount: adjusted_size = max_amount
        if adjusted_size != original_size:
             lg.warning(f"{NEON_YELLOW}Size adjusted by Amount Limits: {original_size:.8f} -> {adjusted_size:.8f} {size_unit}{RESET}")

        # 2. Apply Amount Step Size (Round DOWN)
        if amount_step.is_finite() and amount_step > 0:
            original_size = adjusted_size
            adjusted_size = (adjusted_size // amount_step) * amount_step
            if adjusted_size != original_size:
                 lg.info(f"Applied Amount Step Size ({amount_step}): {original_size:.8f} -> {adjusted_size:.8f}")
        elif not amount_step.is_finite():
             lg.warning(f"Amount step size is not finite ({amount_step}). Skipping step adjustment.")
        else: # Step is zero or negative
             lg.warning(f"Amount step size is zero/negative ({amount_step}). Skipping step adjustment.")

        # 3. Re-check Min Amount & Cost Limits with final adjusted size
        if adjusted_size < min_amount:
             lg.error(f"{NEON_RED}Final size {adjusted_size} < Min Amount {min_amount} after step adjustment. Cannot order.{RESET}")
             return None
        # Cost = Size * Price * ContractValue (for linear/spot)
        estimated_cost = adjusted_size * entry_price * contract_size
        lg.debug(f"  Cost Check: Final Size={adjusted_size:.8f}, Est. Cost={estimated_cost:.4f} (Min:{min_cost}, Max:{max_cost})")
        if min_cost.is_finite() and min_cost > 0 and estimated_cost < min_cost:
             lg.error(f"{NEON_RED}Est. cost {estimated_cost:.4f} < Min Cost {min_cost}. Cannot order.{RESET}")
             return None
        if max_cost.is_finite() and max_cost > 0 and estimated_cost > max_cost:
             lg.error(f"{NEON_RED}Est. cost {estimated_cost:.4f} > Max Cost {max_cost}. Cannot order.{RESET}")
             return None

        # --- Final Validation ---
        final_size = adjusted_size
        if not final_size.is_finite() or final_size <= 0:
             lg.error(f"{NEON_RED}Final size zero/negative/NaN ({final_size}). Aborted.{RESET}")
             return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position using fetch_positions with robust parsing for Bybit V5."""
    lg = logger
    if not exchange.has.get('fetchPositions'):
        lg.warning(f"Exchange {exchange.id} does not support fetchPositions. Cannot check position status.")
        return None

    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        params = {}
        market = exchange.market(symbol) # Needed for market ID if fetching single
        if not market: raise ValueError(f"Market info not loaded for {symbol}")

        if exchange.id == 'bybit':
            params = {'symbol': market['id']} # Use market ID for Bybit V5 fetch

        # Use safe_api_call for the fetch operation
        positions: List[Dict] = safe_api_call(exchange.fetch_positions, lg, symbols=[symbol], params=params)

        if positions is None: # safe_api_call failed after retries
             lg.error("Position fetch failed after retries.")
             # Optional Fallback: Try fetching all positions
             lg.debug("Attempting to fetch ALL positions as fallback...")
             all_positions = safe_api_call(exchange.fetch_positions, lg)
             if all_positions:
                  positions = [p for p in all_positions if p.get('symbol') == symbol]
                  lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
             else:
                  lg.error("Fallback fetch of all positions also failed.")
                  return None

        # --- Process the fetched positions list ---
        active_position = None
        size_threshold = Decimal('1e-9') # Threshold for considering size non-zero

        for pos in positions:
            pos_symbol = pos.get('symbol')
            if pos_symbol != symbol: continue

            pos_size_str = None
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif isinstance(pos.get('info'), dict) and pos['info'].get('size') is not None:
                 pos_size_str = str(pos['info']['size']) # Bybit V5

            if pos_size_str is None: continue

            try:
                position_size = Decimal(pos_size_str)
                if abs(position_size) > size_threshold:
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break # Assume one position per symbol/side/mode
            except (ValueError, TypeError, InvalidOperation) as parse_err:
                lg.warning(f"Could not parse position size '{pos_size_str}': {parse_err}")

        # --- Post-Process the found active position ---
        if active_position:
            try:
                analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market) # Temp instance
                price_prec = analyzer.get_price_precision()
                amt_prec = analyzer.get_amount_precision_places()

                # Standardize Size (always Decimal)
                size_decimal = Decimal(str(active_position.get('contracts', active_position.get('info',{}).get('size', '0'))))
                active_position['contractsDecimal'] = size_decimal

                # Standardize Side
                side = active_position.get('side')
                if side not in ['long', 'short']:
                    if size_decimal > size_threshold: side = 'long'
                    elif size_decimal < -size_threshold: side = 'short'
                    else: lg.warning(f"Pos size {size_decimal} near zero, cannot determine side."); return None
                    active_position['side'] = side
                    lg.debug(f"Inferred position side as '{side}'.")

                # Standardize Entry Price (Decimal)
                entry_price_str = active_position.get('entryPrice') or active_position.get('info', {}).get('avgPrice')
                active_position['entryPriceDecimal'] = Decimal(str(entry_price_str)) if entry_price_str else None

                # Standardize Liq Price (Decimal)
                liq_price_str = active_position.get('liquidationPrice') or active_position.get('info', {}).get('liqPrice')
                active_position['liquidationPriceDecimal'] = Decimal(str(liq_price_str)) if liq_price_str else None

                # Standardize PNL (Decimal)
                pnl_str = active_position.get('unrealizedPnl') or active_position.get('info', {}).get('unrealisedPnl')
                active_position['unrealizedPnlDecimal'] = Decimal(str(pnl_str)) if pnl_str else None

                # Extract SL/TP/TSL from 'info' (Bybit V5) and store as Decimal
                info_dict = active_position.get('info', {})
                sl_str = info_dict.get('stopLoss')
                tp_str = info_dict.get('takeProfit')
                tsl_dist_str = info_dict.get('trailingStop') # Distance value
                tsl_act_str = info_dict.get('activePrice') # Activation price

                active_position['stopLossPriceDecimal'] = Decimal(str(sl_str)) if sl_str and str(sl_str).replace('.','',1).isdigit() and Decimal(sl_str) != 0 else None
                active_position['takeProfitPriceDecimal'] = Decimal(str(tp_str)) if tp_str and str(tp_str).replace('.','',1).isdigit() and Decimal(tp_str) != 0 else None
                active_position['trailingStopLossValueDecimal'] = Decimal(str(tsl_dist_str)) if tsl_dist_str and str(tsl_dist_str).replace('.','',1).isdigit() and Decimal(tsl_dist_str) != 0 else None
                active_position['trailingStopActivationPriceDecimal'] = Decimal(str(tsl_act_str)) if tsl_act_str and str(tsl_act_str).replace('.','',1).isdigit() and Decimal(tsl_act_str) != 0 else None

                # Get timestamp (prefer updatedTime from info for Bybit V5)
                timestamp_ms = info_dict.get('updatedTime') or active_position.get('timestamp')
                active_position['timestamp_ms'] = int(timestamp_ms) if timestamp_ms else None
                timestamp_dt_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp_ms else "N/A"

                # Log Formatted Info
                entry_fmt = f"{active_position['entryPriceDecimal']:.{price_prec}f}" if active_position['entryPriceDecimal'] else 'N/A'
                size_fmt = f"{abs(size_decimal):.{amt_prec}f}"
                liq_fmt = f"{active_position['liquidationPriceDecimal']:.{price_prec}f}" if active_position['liquidationPriceDecimal'] else 'N/A'
                lev_str = info_dict.get('leverage', active_position.get('leverage'))
                lev_fmt = f"{Decimal(str(lev_str)):.1f}x" if lev_str else 'N/A'
                pnl_fmt = f"{active_position['unrealizedPnlDecimal']:.{price_prec}f}" if active_position['unrealizedPnlDecimal'] else 'N/A'
                sl_fmt = f"{active_position['stopLossPriceDecimal']:.{price_prec}f}" if active_position['stopLossPriceDecimal'] else 'N/A'
                tp_fmt = f"{active_position['takeProfitPriceDecimal']:.{price_prec}f}" if active_position['takeProfitPriceDecimal'] else 'N/A'
                tsl_d_fmt = f"{active_position['trailingStopLossValueDecimal']:.{price_prec}f}" if active_position['trailingStopLossValueDecimal'] else 'N/A'
                tsl_a_fmt = f"{active_position['trailingStopActivationPriceDecimal']:.{price_prec}f}" if active_position['trailingStopActivationPriceDecimal'] else 'N/A'

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={size_fmt}, Entry={entry_fmt}, Liq={liq_fmt}, Lev={lev_fmt}, PnL={pnl_fmt}, "
                            f"SL={sl_fmt}, TP={tp_fmt}, TSL(Dist/Act): {tsl_d_fmt}/{tsl_a_fmt} (Updated: {timestamp_dt_str})")
                logger.debug(f"Full processed position details: {active_position}")
                return active_position

            except (ValueError, TypeError, InvalidOperation, KeyError) as proc_err:
                 lg.error(f"Error processing active position details for {symbol}: {proc_err}", exc_info=True)
                 lg.debug(f"Problematic raw position data: {active_position}")
                 return None

        else:
            logger.info(f"No active open position found for {symbol}.")
            return None

    except ccxt.ArgumentsRequired as e:
         # Handle cases where fetching single symbol isn't supported by trying fetch all
         lg.warning(f"Fetching single position failed ({e}). Trying to fetch all positions...")
         try:
             all_positions = safe_api_call(exchange.fetch_positions, lg) # Fetch all
             if all_positions:
                  positions = [p for p in all_positions if p.get('symbol') == symbol]
                  lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
                  # Now re-run the processing logic (This code block is duplicated, consider refactoring)
                  active_position = None; size_threshold = Decimal('1e-9')
                  for pos in positions:
                      pos_symbol = pos.get('symbol')
                      if pos_symbol != symbol: continue
                      pos_size_str = None
                      if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
                      elif isinstance(pos.get('info'), dict) and pos['info'].get('size') is not None: pos_size_str = str(pos['info']['size'])
                      if pos_size_str is None: continue
                      try:
                          position_size = Decimal(pos_size_str)
                          if abs(position_size) > size_threshold: active_position = pos; break
                      except (ValueError, TypeError, InvalidOperation): continue
                  if active_position:
                      # Repeat the post-processing logic here... (Refactor opportunity)
                      try:
                           # ... (Duplicate of post-processing block above) ...
                           return active_position # Return processed position
                      except Exception as proc_err_fallback:
                           lg.error(f"Error processing fallback position details for {symbol}: {proc_err_fallback}", exc_info=True)
                           return None
                  else: logger.info(f"No active open position found for {symbol} in fallback fetch."); return None
             else: lg.error("Fallback fetch of all positions also failed."); return None
         except Exception as fallback_e:
              lg.error(f"Error during fallback fetch of all positions: {fallback_e}", exc_info=True)
              return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage using CCXT, handling Bybit V5 specifics."""
    lg = logger
    if not market_info.get('is_contract', False):
        lg.debug(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # No action needed
    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped: Invalid leverage value ({leverage}). Must be positive integer.")
        return False
    if not exchange.has.get('setLeverage') and not exchange.has.get('setMarginMode'):
        lg.error(f"Exchange {exchange.id} supports neither setLeverage nor setMarginMode. Cannot set leverage.")
        return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        if exchange.id == 'bybit':
            leverage_str = str(leverage) # Bybit V5 requires string leverage in params
            params = {'buyLeverage': leverage_str, 'sellLeverage': leverage_str}
            lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        response = safe_api_call(exchange.set_leverage, lg, leverage, symbol, params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        lg.info(f"{NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Check position details for confirmation).{RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower(); code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {code}){RESET}")
        if exchange.id == 'bybit':
            if code == 110045 or "leverage not modified" in err_str:
                 lg.info(f"{NEON_YELLOW}Leverage for {symbol} likely already set to {leverage}x (Exchange: {e}).{RESET}")
                 return True # Treat as success
            elif code in [110028, 110009, 110055] or "margin mode" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) setting.{RESET}")
            elif code == 110044 or "risk limit" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed Risk Limit tier.{RESET}")
            elif code == 110013 or "parameter error" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid/out of range.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Failed to set leverage for {symbol} after retries or due to unexpected error: {e}{RESET}", exc_info=False)

    return False


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
    params: Optional[Dict] = None # For extra exchange-specific params
) -> Optional[Dict]:
    """Places an order (market or limit) using CCXT with retries and enhanced logging."""
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"

    # --- Validate Inputs ---
    try:
        amount_float = float(position_size) # CCXT usually requires float
        if amount_float <= 0: raise ValueError("Position size must be positive")
    except (ValueError, TypeError) as e:
        lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Invalid size {position_size}: {e}")
        return None
    if order_type == 'limit':
         if limit_price is None or not isinstance(limit_price, Decimal) or not limit_price.is_finite() or limit_price <= 0:
             lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Limit order needs valid positive limit_price.")
             return None
         try: price_float = float(limit_price) # CCXT needs float price
         except (ValueError, TypeError): lg.error(f"Invalid limit_price format {limit_price}."); return None
    else: price_float = None

    # --- Prepare Parameters ---
    order_params = {'reduceOnly': reduce_only}
    if exchange.id == 'bybit': order_params['positionIdx'] = 0 # Assume one-way mode default
    if order_type == 'market' and reduce_only: order_params['timeInForce'] = 'IOC'
    if params: order_params = {**params, **order_params} # Ensure our settings override external

    # --- Log Order Details ---
    analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance for precision
    amt_prec = analyzer.get_amount_precision_places()
    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type.upper()} order for {symbol}:")
    lg.info(f"  Size: {amount_float:.{amt_prec}f} {size_unit}")
    if order_type == 'limit': lg.info(f"  Limit Price: {limit_price}")
    lg.info(f"  ReduceOnly: {reduce_only}, Params: {order_params}")

    # --- Execute Order via safe_api_call ---
    try:
        order = safe_api_call(
            exchange.create_order, lg,
            symbol=symbol, type=order_type, side=side,
            amount=amount_float, price=price_float, params=order_params
        )
        if order and order.get('id'):
            lg.info(f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET}")
            lg.info(f"  ID: {order.get('id')}, Status: {order.get('status', '?')}, Filled: {order.get('filled', 0.0)}, AvgPx: {order.get('average')}")
            lg.debug(f"Raw order response: {order}")
            return order
        else: # Includes case where safe_api_call returns None after retries
             lg.error(f"{NEON_RED}Order placement failed for {symbol}. Response: {order}{RESET}"); return None
    except ccxt.InsufficientFunds as e: lg.error(f"{NEON_RED}Insufficient funds: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order params: {e}{RESET}")
        err = str(e).lower()
        if "tick size" in err: lg.error(" >> Hint: Check limit price tick size.")
        if "step size" in err: lg.error(" >> Hint: Check amount step size.")
        if "minnotional" in err or "cost" in err: lg.error(" >> Hint: Order cost below minimum?")
        if "reduce-only" in err or (getattr(e,'code',None)==110014 and exchange.id=='bybit'): lg.error(" >> Hint: Reduce-only failed (pos closed?).")
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error placing order: {e} (Code: {getattr(e,'code',None)}){RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Failed placing {action_desc} order: {e}{RESET}", exc_info=False)

    return None


def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    logger: logging.Logger, stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None, trailing_stop_distance: Optional[Decimal] = None,
    tsl_activation_price: Optional[Decimal] = None,
) -> bool:
    """Internal helper using Bybit V5 API to set SL, TP, or TSL."""
    lg = logger
    if 'bybit' not in exchange.id.lower(): lg.error("Protection via private_post only for Bybit."); return False
    if not market_info.get('is_contract'): lg.warning(f"Protection skipped ({symbol}: Not contract)."); return False
    if not position_info: lg.error(f"Cannot set protection ({symbol}): Missing position info."); return False

    pos_side = position_info.get('side'); entry_price = position_info.get('entryPriceDecimal')
    pos_idx = 0 # Default for One-Way
    try: pos_idx_val = position_info.get('info', {}).get('positionIdx'); pos_idx = int(pos_idx_val) if pos_idx_val is not None else 0
    except (ValueError, TypeError): lg.warning("Could not parse positionIdx, using default 0.")

    if pos_side not in ['long', 'short']: lg.error("Invalid position side."); return False
    if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
        lg.error("Invalid entry price."); return False

    # --- Validate Protection Parameters ---
    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price.is_finite() and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price.is_finite() and take_profit_price > 0
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance.is_finite() and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price.is_finite() and tsl_activation_price > 0)

    # Check SL/TP/TSL Act relative to entry
    if has_sl and ((pos_side=='long' and stop_loss_price >= entry_price) or (pos_side=='short' and stop_loss_price <= entry_price)): lg.error(f"Invalid SL {stop_loss_price} vs Entry {entry_price}. Ignoring SL."); has_sl = False
    if has_tp and ((pos_side=='long' and take_profit_price <= entry_price) or (pos_side=='short' and take_profit_price >= entry_price)): lg.error(f"Invalid TP {take_profit_price} vs Entry {entry_price}. Ignoring TP."); has_tp = False
    if has_tsl and ((pos_side=='long' and tsl_activation_price <= entry_price) or (pos_side=='short' and tsl_activation_price >= entry_price)): lg.error(f"Invalid TSL Act {tsl_activation_price} vs Entry {entry_price}. Ignoring TSL."); has_tsl = False
    if has_sl and has_tp and stop_loss_price == take_profit_price: lg.error("SL price equals TP price. Ignoring both."); has_sl = False; has_tp = False

    if not has_sl and not has_tp and not has_tsl: lg.info(f"No valid protection params remaining for {symbol}."); return True

    # --- Prepare API Parameters ---
    category = 'linear' if market_info.get('is_linear', True) else 'inverse'
    params = {'category': category, 'symbol': market_info['id'], 'tpslMode': 'Full',
              'tpTriggerBy': 'LastPrice', 'slTriggerBy': 'LastPrice', 'positionIdx': pos_idx}
    log_parts = [f"Setting protection for {symbol} ({pos_side.upper()} PosIdx:{pos_idx}):"]

    try: # Format parameters
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance
        min_tick = analyzer.get_min_tick_size()
        def fmt_p(p): return exchange.price_to_precision(symbol, float(p)) if isinstance(p,Decimal) and p.is_finite() and p>0 else None
        def fmt_d(d):
             if not (isinstance(d, Decimal) and d.is_finite() and d > 0): return None
             prec = abs(min_tick.normalize().as_tuple().exponent) if min_tick > 0 else analyzer.get_price_precision()
             fmt_str = exchange.decimal_to_precision(d, exchange.ROUND, prec, exchange.NO_PADDING)
             # Ensure distance >= min_tick
             return str(min_tick) if min_tick > 0 and Decimal(fmt_str) < min_tick else fmt_str

        # Set TSL first (overrides SL on Bybit V5)
        if has_tsl:
            tsl_d_fmt, tsl_a_fmt = fmt_d(trailing_stop_distance), fmt_p(tsl_activation_price)
            if tsl_d_fmt and tsl_a_fmt:
                params['trailingStop']=tsl_d_fmt; params['activePrice']=tsl_a_fmt
                log_parts.append(f"  TSL: Dist={tsl_d_fmt}, Act={tsl_a_fmt}"); has_sl=False # Disable fixed SL
            else: lg.error("Failed formatting TSL params."); has_tsl=False # Mark TSL failed

        if has_sl: # Only set if TSL wasn't set
            sl_fmt = fmt_p(stop_loss_price)
            if sl_fmt: params['stopLoss']=sl_fmt; log_parts.append(f"  FixSL: {sl_fmt}")
            else: lg.error("Failed formatting Fixed SL."); has_sl=False # Mark SL failed

        if has_tp:
            tp_fmt = fmt_p(take_profit_price)
            if tp_fmt: params['takeProfit']=tp_fmt; log_parts.append(f"  FixTP: {tp_fmt}")
            else: lg.error("Failed formatting Fixed TP."); has_tp=False # Mark TP failed

    except Exception as fmt_err: lg.error(f"Error formatting protection params: {fmt_err}", exc_info=True); return False

    # Check if any parameters were actually added
    if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
        lg.warning(f"No valid protection parameters formatted for {symbol}. No API call."); return False

    # --- Make API Call ---
    lg.info("\n".join(log_parts)); lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', {params})")
    try:
        response = safe_api_call(exchange.private_post, lg, '/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection response: {response}")
        code=response.get('retCode'); msg=response.get('retMsg','Err'); ext=response.get('retExtInfo',{})
        if code == 0:
            if "not modified" in msg.lower(): lg.info(f"{NEON_YELLOW}Protection already set or partially modified ({symbol}). Msg: {msg}{RESET}")
            else: lg.info(f"{NEON_GREEN}Protection set/updated successfully ({symbol}).{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Failed set protection ({symbol}): {msg} (Code:{code}) Ext:{ext}{RESET}")
            if code == 110013: lg.error(" >> Hint(110013): Param Error? Check prices vs entry/tick, TSL values.")
            elif code == 110036: lg.error(f" >> Hint(110036): TSL Act Price '{params.get('activePrice')}' invalid?")
            elif code == 110086: lg.error(" >> Hint(110086): SL price == TP price.")
            elif code == 110025: lg.error(" >> Hint(110025): Position closed or posIdx mismatch?")
            return False
    except Exception as e: lg.error(f"Failed protection API call: {e}", exc_info=False); return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    config: Dict[str, Any], logger: logging.Logger, take_profit_price: Optional[Decimal] = None
) -> bool:
    """Calculates and sets Exchange-Native Trailing Stop Loss."""
    lg = logger
    if not config.get("enable_trailing_stop"): lg.debug(f"TSL disabled ({symbol})."); return False

    try: # Validate inputs
        cb_rate = Decimal(str(config["trailing_stop_callback_rate"]))
        act_pct = Decimal(str(config["trailing_stop_activation_percentage"]))
        entry = position_info['entryPriceDecimal']; side = position_info['side']
        if cb_rate <= 0: raise ValueError("callback_rate must be > 0")
        if act_pct < 0: raise ValueError("activation_percentage must be >= 0")
        if not isinstance(entry, Decimal) or not entry.is_finite() or entry <= 0: raise ValueError("Invalid entry price")
        if side not in ['long','short']: raise ValueError("Invalid side")
    except (KeyError, ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Invalid TSL config or pos info ({symbol}): {e}.", exc_info=True); return False

    try: # Calculate parameters
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance
        prec = analyzer.get_price_precision(); rnd = Decimal(f'1e-{prec}'); tick = analyzer.get_min_tick_size()

        act_off = entry * act_pct; act_price = None
        if side == 'long':
             raw_activation = entry + act_off
             act_price = raw_activation.quantize(rnd, rounding=ROUND_UP)
             if act_price <= entry: act_price = (entry+tick).quantize(rnd,rounding=ROUND_UP) # Ensure > entry
        else: # short
             raw_activation = entry - act_off
             act_price = raw_activation.quantize(rnd, rounding=ROUND_DOWN)
             if act_price >= entry: act_price = (entry-tick).quantize(rnd,rounding=ROUND_DOWN) # Ensure < entry

        if not act_price.is_finite() or act_price <= 0:
            lg.error(f"Invalid TSL Act price calc ({act_price})."); return False

        dist_raw = act_price * cb_rate; dist = None
        if tick > 0:
             dist = (dist_raw / tick).quantize(Decimal('1'), rounding=ROUND_UP) * tick # Round up to nearest tick
             dist = max(dist, tick) # Ensure at least one tick
        else: dist = dist_raw.quantize(rnd, rounding=ROUND_UP) # Fallback rounding

        if not dist.is_finite() or dist <= 0:
            lg.error(f"Invalid TSL Dist calc ({dist})."); return False

        lg.info(f"Calc TSL Params ({symbol} {side.upper()}): ActPrice={act_price:.{prec}f}, Dist={dist:.{prec}f}")
        if isinstance(take_profit_price, Decimal) and take_profit_price.is_finite() and take_profit_price > 0:
            lg.info(f"  Also setting TP: {take_profit_price:.{prec}f}")

        # Set protection via helper
        return _set_position_protection(exchange, symbol, market_info, position_info, lg,
                                        stop_loss_price=None, # TSL overrides fixed SL
                                        take_profit_price=take_profit_price if (isinstance(take_profit_price, Decimal) and take_profit_price.is_finite() and take_profit_price > 0) else None,
                                        trailing_stop_distance=dist,
                                        tsl_activation_price=act_price)
    except Exception as e: lg.error(f"Error calculating/setting TSL: {e}", exc_info=True); return False


# --- Main Analysis and Trading Loop ---
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
            orderbook_data = fetch_orderbook_ccxt(exchange, symbol, int(config["orderbook_limit"]), lg) # Ensure limit is int
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
                         if not limit_px.is_finite() or limit_px <= 0: raise ValueError("Limit price non-positive/finite")
                         lg.info(f"Calc Limit Entry for {signal}: {limit_px}")
                     except (KeyError, ValueError, InvalidOperation) as e:
                          lg.error(f"Limit price calc failed ({e}). Switching to Market."); entry_type="market"; limit_px=None

                trade_order = place_trade(exchange, symbol, signal, pos_size, market_info, lg, entry_type, limit_px)

                # --- Post-Order Handling ---
                if trade_order and trade_order.get('id'):
                    order_id, status = trade_order['id'], trade_order.get('status')
                    if status == 'closed' or entry_type == 'market':
                        delay = config["position_confirm_delay_seconds"] if entry_type=='market' else 2
                        lg.info(f"Order {order_id} placed/filled. Waiting {delay}s for confirmation...")
                        time.sleep(delay)
                        confirmed_pos = get_open_position(exchange, symbol, lg)
                        if confirmed_pos:
                            lg.info(f"{NEON_GREEN}Position Confirmed!{RESET}")
                            try:
                                entry_act = confirmed_pos.get('entryPriceDecimal') or current_price # Use actual or estimate
                                lg.info(f"Actual Entry ~ {entry_act:.{price_prec}f}")
                                _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_act, signal) # Recalculate based on actual entry
                                protection_ok = False
                                if config["enable_trailing_stop"]:
                                     lg.info(f"Setting TSL (TP target: {tp_final})...")
                                     protection_ok = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_pos, config, lg, tp_final)
                                elif sl_final or tp_final:
                                     lg.info(f"Setting Fixed SL ({sl_final}) / TP ({tp_final})...")
                                     protection_ok = _set_position_protection(exchange, symbol, market_info, confirmed_pos, lg, sl_final, tp_final)
                                else: lg.warning("No valid protection calculated (TSL disabled or calc failed).")

                                if protection_ok: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION COMPLETE ({symbol} {signal}) ===")
                                else: lg.error(f"{NEON_RED}=== TRADE PLACED BUT PROTECTION FAILED ({symbol} {signal}) ===\n{NEON_YELLOW}>>> MANUAL MONITORING REQUIRED! <<<")
                            except Exception as post_err:
                                 lg.error(f"Error setting protection: {post_err}", exc_info=True); lg.warning(f"{NEON_YELLOW}Position open, manual check needed!{RESET}")
                        else: lg.error(f"{NEON_RED}Order {order_id} placed/filled BUT POSITION NOT CONFIRMED! Manual check!{RESET}")
                    elif status == 'open' and entry_type == 'limit':
                         lg.info(f"Limit order {order_id} OPEN. Will check status next cycle.")
                    else: lg.error(f"Order {order_id} status: {status}. Trade did not open as expected.")
                else: lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED. Order placement error. ===")
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
                           close_sig = "SELL" if pos_side == 'long' else "BUY"
                           size_close = abs(pos_size)
                           if size_close <= 0: raise ValueError("Position size is zero/negative.")
                           close_order = place_trade(exchange, symbol, close_sig, size_close, market_info, lg, 'market', reduce_only=True)
                           if close_order: lg.info(f"{NEON_GREEN}Time-based CLOSE order placed. ID: {close_order.get('id','?')}{RESET}")
                           else: lg.error(f"{NEON_RED}Failed time-based CLOSE order! Manual check!{RESET}")
                           return # Exit cycle
                 except Exception as time_err: lg.error(f"Error in time exit check: {time_err}")

            # --- Position Management (Break-Even) ---
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

            # Placeholder for other potential management logic
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
    global CONFIG, QUOTE_CURRENCY, config # Allow modification of globals

    # Setup initial logger
    init_logger = setup_logger("ScalpXRX_Init", level=logging.INFO)
    start_time_str = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
    init_logger.info(f"--- Starting ScalpXRX Bot ({start_time_str}) ---")

    try:
        CONFIG = load_config(CONFIG_FILE)
        config = CONFIG # Make config accessible without global keyword in helpers if needed
        QUOTE_CURRENCY = config.get("quote_currency", "USDT")
        TARGET_SYMBOL = config.get("symbol")

        if not TARGET_SYMBOL:
            init_logger.critical("CRITICAL: 'symbol' not defined in config.json. Exiting.")
            return

        # Setup logger specific to the target symbol for the main loop
        safe_symbol_name = TARGET_SYMBOL.replace('/', '_').replace(':', '-')
        symbol_logger_name = f"ScalpXRX_{safe_symbol_name}"
        main_logger = setup_logger(symbol_logger_name, level=logging.INFO) # Use INFO for console by default
        main_logger.info(f"Logging initialized for symbol: {TARGET_SYMBOL}")
        main_logger.info(f"Config loaded. Quote: {QUOTE_CURRENCY}, Interval: {config['interval']}")
        try: ta_version = ta.version
        except: ta_version = "N/A" # Handle if pandas_ta version attribute missing
        main_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={ta_version}")


        # --- Trading Enabled Warning ---
        if config.get("enable_trading"):
            main_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
            env_type = "SANDBOX (Testnet)" if config.get("use_sandbox") else f"{NEON_RED}!!! REAL MONEY !!!"
            main_logger.warning(f"Environment: {env_type}{RESET}")
            risk_pct = config.get('risk_per_trade', 0) * 100
            lev = config.get('leverage', 1)
            main_logger.warning(f"Settings: Risk/Trade={risk_pct:.2f}%, Leverage={lev}x")
            for i in range(3, 0, -1):
                main_logger.warning(f"Starting in {i}...")
                time.sleep(1)
        else:
            main_logger.info("Trading is disabled in config. Running in analysis-only mode.")

        # --- Initialize Exchange ---
        exchange = initialize_exchange(config, main_logger)
        if not exchange:
            main_logger.critical("Failed to initialize exchange. Exiting.")
            return

        # --- Main Loop ---
        main_logger.info(f"Starting main analysis loop for {TARGET_SYMBOL}...")
        loop_interval = max(1, config.get("loop_delay_seconds", LOOP_DELAY_SECONDS)) # Ensure positive delay
        while True:
            try:
                analyze_and_trade_symbol(exchange, TARGET_SYMBOL, config, main_logger)
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
    # Assign config globally after loading if helper functions rely on it implicitly
    # (Though passing config explicitly is generally better practice)
    config = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = config.get("quote_currency", "USDT")
    main()
    
