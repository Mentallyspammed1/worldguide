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
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present with validation."""
    default_config = {
        "symbol": "BTC/USDT:USDT", # Specify the full symbol including contract type if needed
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
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
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,
        "scalping_signal_threshold": 2.5,
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False,
        "use_sandbox": True,
        "risk_per_trade": 0.01, # e.g., 1%
        "leverage": 20,
        "max_concurrent_positions": 1, # Per symbol managed by this instance
        "quote_currency": "USDT",
        "entry_order_type": "market", # "market" or "limit"
        "limit_order_offset_buy": 0.0005, # Percentage offset (0.05%)
        "limit_order_offset_sell": 0.0005, # Percentage offset (0.05%)
        "enable_trailing_stop": True,
        "trailing_stop_callback_rate": 0.005, # e.g., 0.5% trail distance
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate when profit reaches 0.3%
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "time_based_exit_minutes": None, # e.g., 60 to exit after 1 hour
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": {
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default"
    }

    config = default_config.copy()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            config = _merge_configs(loaded_config, default_config)
            print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
            # Optionally recreate default if load failed badly
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(default_config, f_write, indent=4)
                print(f"{NEON_YELLOW}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create:
                print(f"{NEON_RED}Error recreating default config file: {e_create}{RESET}")
    else:
        print(f"{NEON_YELLOW}Config file not found. Creating default config at {filepath}{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            config = default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            # Continue with in-memory default config

    # --- Validation ---
    updated = False
    if config.get("interval") not in VALID_INTERVALS:
        print(f"{NEON_RED}Invalid interval '{config.get('interval')}' in config. Resetting to default '{default_config['interval']}'.{RESET}")
        config["interval"] = default_config["interval"]
        updated = True
    if config.get("entry_order_type") not in ["market", "limit"]:
        print(f"{NEON_RED}Invalid entry_order_type '{config.get('entry_order_type')}' in config. Resetting to 'market'.{RESET}")
        config["entry_order_type"] = "market"
        updated = True
    if config.get("active_weight_set") not in config.get("weight_sets", {}):
         print(f"{NEON_RED}Active weight set '{config.get('active_weight_set')}' not found in 'weight_sets'. Resetting to 'default'.{RESET}")
         config["active_weight_set"] = "default" # Assume 'default' exists
         updated = True
    # Add more validation (numeric ranges, types) as needed
    for key in ["risk_per_trade", "leverage", "stop_loss_multiple", "take_profit_multiple",
                "trailing_stop_callback_rate", "trailing_stop_activation_percentage",
                "break_even_trigger_atr_multiple", "break_even_offset_ticks"]:
        try:
             val = config[key]
             # Check basic numeric types and ranges
             if key == "risk_per_trade" and not (0 < float(val) < 1): raise ValueError("must be between 0 and 1")
             if key == "leverage" and not (int(val) >= 1): raise ValueError("must be >= 1")
             if key in ["stop_loss_multiple", "take_profit_multiple", "break_even_trigger_atr_multiple"] and not (float(val) > 0): raise ValueError("must be > 0")
             if key in ["trailing_stop_callback_rate", "trailing_stop_activation_percentage"] and not (float(val) >= 0): raise ValueError("must be >= 0")
             if key == "break_even_offset_ticks" and not (int(val) >= 0): raise ValueError("must be >= 0")
        except (ValueError, TypeError, KeyError) as e:
            print(f"{NEON_RED}Invalid value for '{key}' ({config.get(key)}): {e}. Resetting to default '{default_config[key]}'.{RESET}")
            config[key] = default_config[key]
            updated = True

    # If config was updated due to invalid values, save it back
    if updated:
        try:
            with open(filepath, "w", encoding="utf-8") as f_write:
                json.dump(config, f_write, indent=4)
            print(f"{NEON_YELLOW}Updated config file {filepath} with corrected/default values.{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

    return config

def _merge_configs(loaded_config: Dict, default_config: Dict) -> Dict:
    """Recursively merges loaded config with defaults, ensuring all keys exist."""
    merged = default_config.copy()
    for key, value in loaded_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_configs(value, merged[key])
        else:
            merged[key] = value
    return merged

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Sets up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        # Clear existing handlers to ensure clean setup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        # return logger # Option: return existing logger

    logger.setLevel(level)

    # File Handler (Rotating)
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log all levels to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger {log_filename}: {e}")

    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z' # Include Timezone
    )
    stream_formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
    stream_handler.setFormatter(stream_formatter)
    # Set console level (e.g., INFO for less verbosity, DEBUG for more)
    console_log_level = logging.INFO
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent duplicate messages in parent loggers
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with enhanced error handling."""
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Assume linear for Bybit V5
                'adjustForTimeDifference': True,
                # Increased timeouts
                'fetchTickerTimeout': 15000, # 15s
                'fetchBalanceTimeout': 20000, # 20s
                'createOrderTimeout': 25000, # 25s
                'cancelOrderTimeout': 20000, # 20s
                'fetchPositionsTimeout': 20000, # 20s
                'fetchOHLCVTimeout': 20000, # 20s
                # Bybit V5 specific options might be needed here if issues persist
                # 'recvWindow': 10000 # Example: Increase recvWindow if needed
            }
        }

        # Default to Bybit, can be made dynamic if needed
        exchange_id = "bybit"
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            try:
                exchange.set_sandbox_mode(True)
                lg.info(f"Sandbox mode explicitly enabled for {exchange.id}.")
            except AttributeError:
                lg.warning(f"{exchange.id} does not support set_sandbox_mode via ccxt. Ensure API keys are for Testnet.")
                # Attempt to set URLs manually for Bybit if needed
                if exchange.id == 'bybit':
                    exchange.urls['api'] = 'https://api-testnet.bybit.com'
                    lg.info("Manually set Bybit API URL to Testnet.")
            except Exception as e:
                lg.error(f"Error enabling sandbox mode: {e}")

        lg.info(f"Initializing {exchange.id}...")
        # Test connection and API keys by fetching balance early
        account_type_to_test = 'CONTRACT' # For Bybit V5, try CONTRACT or UNIFIED
        lg.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            params = {'type': account_type_to_test} if exchange.id == 'bybit' else {}
            balance = exchange.fetch_balance(params=params)
            # Check common quote currencies for available balance display
            quote_curr = CONFIG.get("quote_currency", "USDT")
            available_quote = balance.get(quote_curr, {}).get('free', 'N/A')
            lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {quote_curr} available: {available_quote})")

        except ccxt.AuthenticationError as auth_err:
             lg.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             lg.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on the exchange.{RESET}")
             return None
        except ccxt.ExchangeError as balance_err:
             lg.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
             try:
                  balance = exchange.fetch_balance()
                  quote_curr = CONFIG.get("quote_currency", "USDT")
                  available_quote = balance.get(quote_curr, {}).get('free', 'N/A')
                  lg.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {quote_curr} available: {available_quote})")
             except Exception as fallback_err:
                  lg.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type/network.{RESET}")
        except ccxt.NetworkError as net_err:
             lg.error(f"{NEON_RED}Network Error during initial balance fetch: {net_err}{RESET}")
             lg.warning("Proceeding, but balance checks might fail later.")
             # Decide if we should return None here or allow proceeding cautiously
             # return None
        except Exception as balance_err:
             lg.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type/network. Proceeding cautiously.{RESET}")


        # Load markets after initial connection test
        lg.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        return exchange

    except ccxt.AuthenticationError as e:
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
    """Wraps an API call with retry logic for network/rate limit errors."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            result = func(*args, **kwargs)
            return result # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Network error in {func.__name__}: {e}. Retrying ({attempts+1}/{MAX_API_RETRIES})...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts) # Exponential backoff
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded in {func.__name__}: {e}. Waiting {wait_time:.1f}s ({attempts+1}/{MAX_API_RETRIES})...{RESET}")
            time.sleep(wait_time)
            attempts += 1 # Consume attempt after wait
            continue # Skip standard delay
        except ccxt.ExchangeNotAvailable as e: # Server maintenance or outage
             wait_time = RETRY_DELAY_SECONDS * 5 * (2 ** attempts) # Longer exponential backoff
             lg.error(f"{NEON_RED}Exchange not available ({func.__name__}): {e}. Waiting {wait_time:.1f}s ({attempts+1}/{MAX_API_RETRIES})...{RESET}")
             time.sleep(wait_time)
             attempts += 1
             continue
        except ccxt.AuthenticationError as e:
             # Don't retry auth errors, they need fixing
             lg.error(f"{NEON_RED}Authentication Error in {func.__name__}: {e}. Aborting call.{RESET}")
             raise e # Re-raise to be caught by caller
        except ccxt.ExchangeError as e:
            # Decide if specific exchange errors are retryable
            # Example: Bybit internal server error (e.g., code 10001) might be temporary
            bybit_retry_codes = [10001, 10006] # Example: Internal error, Rate limit system error
            exchange_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            if exchange_code in bybit_retry_codes or "internal server error" in err_str:
                 lg.warning(f"{NEON_YELLOW}Retryable exchange error in {func.__name__}: {e} (Code: {exchange_code}). Retrying ({attempts+1}/{MAX_API_RETRIES})...{RESET}")
            else:
                 # Non-retryable ExchangeError
                 lg.error(f"{NEON_RED}Non-retryable Exchange Error in {func.__name__}: {e} (Code: {exchange_code}){RESET}")
                 raise e # Re-raise
        except Exception as e:
            # Unexpected errors - typically don't retry these
            lg.error(f"{NEON_RED}Unexpected error in {func.__name__}: {e}{RESET}", exc_info=True)
            raise e # Re-raise

        # Standard delay before next attempt
        attempts += 1
        if attempts <= MAX_API_RETRIES:
             delay = RETRY_DELAY_SECONDS * (1.5 ** (attempts -1)) # Mild exponential backoff
             time.sleep(delay)

    lg.error(f"{NEON_RED}Max retries ({MAX_API_RETRIES}) exceeded for {func.__name__}.{RESET}")
    # Raise the last exception encountered after exhausting retries
    # Using 'e' from the last loop iteration might be problematic if the last error wasn't caught by a specific type
    # It's better to explicitly raise a custom error or return None/False indicating failure
    raise ccxt.RequestTimeout(f"Max retries exceeded for {func.__name__}")


# --- CCXT Data Fetching (Using safe_api_call) ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks and retries."""
    lg = logger
    try:
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)
        if not ticker:
            lg.error(f"Failed to fetch ticker for {symbol} after retries.")
            return None

        lg.debug(f"Ticker data for {symbol}: {ticker}")
        price = None
        # Prioritize 'last', then mid-price, then ask/bid
        last_price = ticker.get('last')
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')

        # Try converting to Decimal robustly
        def to_decimal(value) -> Optional[Decimal]:
            if value is None: return None
            try:
                d = Decimal(str(value))
                return d if d > 0 else None # Ensure positive price
            except InvalidOperation:
                lg.warning(f"Invalid price format: {value}")
                return None

        p_last = to_decimal(last_price)
        p_bid = to_decimal(bid_price)
        p_ask = to_decimal(ask_price)

        if p_last:
            price = p_last; lg.debug(f"Using 'last' price: {price}")
        elif p_bid and p_ask:
            price = (p_bid + p_ask) / 2; lg.debug(f"Using bid/ask midpoint: {price}")
        elif p_ask:
            price = p_ask; lg.warning(f"Using 'ask' price fallback: {price}")
        elif p_bid:
            price = p_bid; lg.warning(f"Using 'bid' price fallback: {price}")

        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to get a valid price from ticker data for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except Exception as e:
        # Catch errors raised by safe_api_call (like AuthError or max retries exceeded) or parsing issues
        lg.error(f"{NEON_RED}Error fetching current price for {symbol}: {e}{RESET}", exc_info=False) # Don't need full trace if safe_api_call logged it
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
            lg.warning(f"{NEON_YELLOW}No valid kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # Process the data into a pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Basic validation of the DataFrame structure
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe}.{RESET}")
            return df

        # Convert timestamp to datetime objects (UTC), coerce errors
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # Convert price/volume columns to numeric Decimal, coercing errors
        for col in ['open', 'high', 'low', 'close', 'volume']:
             try:
                  # Use Decimal for price/volume columns for precision
                  df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
             except (TypeError, ValueError, InvalidOperation) as conv_err:
                  lg.warning(f"Could not convert column '{col}' to Decimal, attempting numeric: {conv_err}")
                  df[col] = pd.to_numeric(df[col], errors='coerce') # Fallback to numeric

        # Data Cleaning: Drop rows with NaN in essential price columns or zero close price
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Ensure close price is positive Decimal or float
        df = df[df['close'] > Decimal(0) if isinstance(df['close'].iloc[0], Decimal) else df['close'] > 0]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{RESET}")
            return pd.DataFrame()

        # Sort by timestamp index
        df.sort_index(inplace=True)
        # Ensure no duplicate timestamps remain
        df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing klines for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and validation."""
    lg = logger
    if not exchange.has['fetchOrderBook']:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    try:
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)

        # Validate structure
        if not orderbook:
            lg.warning(f"fetch_order_book returned None/empty for {symbol} after retries.")
            return None
        elif not isinstance(orderbook, dict):
            lg.warning(f"Invalid orderbook type received for {symbol}. Expected dict, got {type(orderbook)}.")
            return None
        elif 'bids' not in orderbook or 'asks' not in orderbook:
            lg.warning(f"Invalid orderbook structure for {symbol}: missing 'bids' or 'asks'. Keys: {list(orderbook.keys())}")
            return None
        elif not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
            lg.warning(f"Invalid orderbook structure for {symbol}: 'bids'/'asks' not lists. Types: bid={type(orderbook['bids'])}, ask={type(orderbook['asks'])}")
            return None
        elif not orderbook['bids'] and not orderbook['asks']:
            lg.warning(f"Orderbook received but both bids and asks lists are empty for {symbol}.")
            # Return the empty but validly structured book
            return orderbook
        else:
            lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
            # Add basic validation of bid/ask format (price, size)
            valid = True
            for side in ['bids', 'asks']:
                 if orderbook[side]: # Check first entry if list is not empty
                      entry = orderbook[side][0]
                      if not (isinstance(entry, list) and len(entry) == 2):
                           lg.warning(f"Invalid {side[:-1]} entry format in orderbook: {entry}")
                           valid = False; break
                      try:
                           # Attempt conversion to float/Decimal to check numeric format
                           _ = float(entry[0]); _ = float(entry[1])
                           # Or: _ = Decimal(str(entry[0])); _ = Decimal(str(entry[1]))
                      except (ValueError, TypeError, InvalidOperation):
                           lg.warning(f"Non-numeric data in {side[:-1]} entry: {entry}")
                           valid = False; break
            if not valid:
                 lg.error("Orderbook data format validation failed.")
                 return None # Reject invalid format
            return orderbook

    except Exception as e:
        # Catch errors raised by safe_api_call or validation
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
        self.df = df
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "5")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)
        if not self.ccxt_interval:
            self.logger.error(f"Invalid interval '{self.interval}' in config, cannot map to CCXT timeframe for {self.symbol}.")
            # Consider raising an error or handling fallback more explicitly

        # Stores latest calculated indicator values (Decimal for prices/ATR, float for others)
        self.indicator_values: Dict[str, Union[Decimal, float, Any]] = {}
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_column_names: Dict[str, Optional[str]] = {} # Maps internal name to actual DataFrame column name

        if not self.weights:
            logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}.")
            # Fallback to an empty dict to avoid errors, but scoring will be zero
            self.weights = {}

        # Initial calculations
        self._calculate_all_indicators()
        self._update_latest_indicator_values() # Needs to run after indicator calc
        self.calculate_fibonacci_levels()

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
        # This is less reliable but might catch unexpected variations
        # Convert base name to lower for comparison
        base_lower = base_name.lower()
        # Split common indicator names if needed (e.g., "EMA_Short" -> "ema")
        simple_base = base_lower.split('_')[0]
        for col in df_cols:
            col_lower = col.lower()
            if base_lower in col_lower or simple_base in col_lower:
                # Be cautious with very short base names (e.g., 'r' in 'atr', 'wr')
                if len(simple_base) > 1 and simple_base in col_lower:
                     self.logger.debug(f"Found column '{col}' for base '{base_name}' using fallback substring search ('{simple_base}').")
                     return col
                elif base_lower in col_lower:
                     self.logger.debug(f"Found column '{col}' for base '{base_name}' using fallback substring search ('{base_lower}').")
                     return col


        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {df_cols}")
        return None

    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty, cannot calculate indicators for {self.symbol}.")
            return

        # Determine minimum required data length
        required_periods = []
        indicators_config = self.config.get("indicators", {})
        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name, {})

        # Add period only if indicator is enabled AND has a non-zero weight
        def add_req(key, default):
            if indicators_config.get(key, False) and float(active_weights.get(key, 0)) != 0:
                required_periods.append(self.config.get(key.replace("window", "period").replace("period","period"), default)) # Basic key mapping

        # Add specific periods based on config keys
        add_req("atr_period", DEFAULT_ATR_PERIOD)
        if indicators_config.get("ema_alignment", False) and float(active_weights.get("ema_alignment", 0)) != 0:
             required_periods.append(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
             required_periods.append(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        add_req("momentum_period", DEFAULT_MOMENTUM_PERIOD)
        add_req("cci_window", DEFAULT_CCI_WINDOW)
        add_req("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        add_req("mfi_window", DEFAULT_MFI_WINDOW)
        add_req("sma_10_window", DEFAULT_SMA_10_WINDOW)
        if indicators_config.get("stoch_rsi", False) and float(active_weights.get("stoch_rsi", 0)) != 0:
            required_periods.append(self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW))
            required_periods.append(self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))
        add_req("rsi_period", DEFAULT_RSI_WINDOW)
        add_req("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        add_req("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
        # Fibonacci window for price range
        required_periods.append(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))

        min_required_data = max(required_periods) + 30 if required_periods else 50 # Add more buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators reliably (min recommended: {min_required_data}). Results may contain NaNs.{RESET}")
             # Continue calculation, but expect NaNs

        try:
            # Work on a copy
            df_calc = self.df.copy()
            # Ensure OHLCV columns are numeric (float or Decimal) for ta library
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 if col in df_calc.columns and not pd.api.types.is_numeric_dtype(df_calc[col]):
                      self.logger.debug(f"Converting column '{col}' to numeric for TA calculations.")
                      # Convert Decimal to float if needed by pandas_ta
                      if isinstance(df_calc[col].iloc[0], Decimal):
                           df_calc[col] = df_calc[col].astype(float)
                      else: # Handle other non-numeric types
                           df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            # --- Always calculate ATR ---
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # --- Calculate other indicators based on config and weight ---
            if indicators_config.get("ema_alignment", False) and float(active_weights.get("ema_alignment", 0)) != 0:
                ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                df_calc.ta.ema(length=ema_short, append=True)
                self.ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
                df_calc.ta.ema(length=ema_long, append=True)
                self.ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)

            if indicators_config.get("momentum", False) and float(active_weights.get("momentum", 0)) != 0:
                mom_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                df_calc.ta.mom(length=mom_period, append=True)
                self.ta_column_names["Momentum"] = self._get_ta_col_name("Momentum", df_calc)

            if indicators_config.get("cci", False) and float(active_weights.get("cci", 0)) != 0:
                cci_period = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
                df_calc.ta.cci(length=cci_period, append=True)
                self.ta_column_names["CCI"] = self._get_ta_col_name("CCI", df_calc)

            if indicators_config.get("wr", False) and float(active_weights.get("wr", 0)) != 0:
                wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
                df_calc.ta.willr(length=wr_period, append=True)
                self.ta_column_names["Williams_R"] = self._get_ta_col_name("Williams_R", df_calc)

            if indicators_config.get("mfi", False) and float(active_weights.get("mfi", 0)) != 0:
                mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
                df_calc.ta.mfi(length=mfi_period, append=True)
                self.ta_column_names["MFI"] = self._get_ta_col_name("MFI", df_calc)

            if indicators_config.get("vwap", False) and float(active_weights.get("vwap", 0)) != 0:
                df_calc.ta.vwap(append=True)
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            if indicators_config.get("psar", False) and float(active_weights.get("psar", 0)) != 0:
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                if psar_result is not None and not psar_result.empty:
                    df_calc = pd.concat([df_calc, psar_result], axis=1)
                    self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                    self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)

            if indicators_config.get("sma_10", False) and float(active_weights.get("sma_10", 0)) != 0:
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)

            if indicators_config.get("stoch_rsi", False) and float(active_weights.get("stoch_rsi", 0)) != 0:
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                if stochrsi_result is not None and not stochrsi_result.empty:
                    df_calc = pd.concat([df_calc, stochrsi_result], axis=1)
                    self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                    self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)

            if indicators_config.get("rsi", False) and float(active_weights.get("rsi", 0)) != 0:
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False) and float(active_weights.get("bollinger_bands", 0)) != 0:
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = float(self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV))
                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std)
                if bbands_result is not None and not bbands_result.empty:
                    df_calc = pd.concat([df_calc, bbands_result], axis=1)
                    self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                    self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                    self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)

            if indicators_config.get("volume_confirmation", False) and float(active_weights.get("volume_confirmation", 0)) != 0:
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}"
                # Ensure volume is float for SMA calculation
                vol_series = df_calc['volume'].astype(float) if isinstance(df_calc['volume'].iloc[0], Decimal) else df_calc['volume']
                df_calc[vol_ma_col_name] = ta.sma(vol_series.fillna(0), length=vol_ma_period)
                self.ta_column_names["Volume_MA"] = vol_ma_col_name

            # Convert calculated indicator columns (usually float) back to Decimal if desired?
            # For now, keep them as float, except ATR which needs Decimal precision.
            # Convert ATR column to Decimal if it's not already
            atr_col = self.ta_column_names.get("ATR")
            if atr_col and atr_col in df_calc.columns and not isinstance(df_calc[atr_col].iloc[-1], Decimal):
                 self.logger.debug(f"Converting calculated ATR column '{atr_col}' to Decimal.")
                 df_calc[atr_col] = df_calc[atr_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))


            # Update the instance's DataFrame
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
             self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol}: {e}{RESET}. Check pandas_ta usage and data.", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates indicator_values dict with latest values, handling types."""
        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame empty for {self.symbol}.")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Open", "High", "Low", "Close", "Volume"]}
            return
        try:
            latest = self.df.iloc[-1]
        except IndexError:
            self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be empty or too short.")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Open", "High", "Low", "Close", "Volume"]}
            return

        if latest.isnull().all():
            self.logger.warning(f"Cannot update latest values: Last row contains all NaNs for {self.symbol}.")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Open", "High", "Low", "Close", "Volume"]}
            return

        updated_values = {}
        # --- Process TA indicators ---
        for key, col_name in self.ta_column_names.items():
            if col_name and col_name in latest.index:
                value = latest[col_name]
                if pd.notna(value):
                    try:
                        # Store ATR as Decimal, others as float
                        if key == "ATR":
                            updated_values[key] = Decimal(str(value)) if not isinstance(value, Decimal) else value
                        else:
                            updated_values[key] = float(value)
                    except (ValueError, TypeError, InvalidOperation) as conv_err:
                        self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}): {conv_err}. Storing NaN.")
                        updated_values[key] = np.nan
                else:
                    updated_values[key] = np.nan
            else:
                 if key in self.ta_column_names: # Only log if calc was attempted
                     self.logger.debug(f"Indicator column '{col_name}' for '{key}' not found in latest data. Storing NaN.")
                 updated_values[key] = np.nan

        # --- Process Base OHLCV (ensure Decimal) ---
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            key_name = base_col.capitalize()
            value = latest.get(base_col)
            if pd.notna(value):
                 try:
                      # Ensure base values are stored as Decimal
                      updated_values[key_name] = Decimal(str(value)) if not isinstance(value, Decimal) else value
                 except (ValueError, TypeError, InvalidOperation) as conv_err:
                      self.logger.warning(f"Could not convert base '{base_col}' ({value}) to Decimal: {conv_err}. Storing NaN.")
                      updated_values[key_name] = np.nan
            else:
                 updated_values[key_name] = np.nan

        self.indicator_values = updated_values

        # --- Log Summary (formatted) ---
        log_vals = {}
        price_prec = self.get_price_precision()
        for k, v in self.indicator_values.items():
            if pd.notna(v):
                if isinstance(v, Decimal):
                    # Use appropriate precision for Decimals
                    prec = price_prec if k in ['Open', 'High', 'Low', 'Close', 'ATR'] else 8 # More precision for vol/others
                    log_vals[k] = f"{v:.{prec}f}"
                elif isinstance(v, float):
                    log_vals[k] = f"{v:.5f}" # Standard float precision
                else:
                    log_vals[k] = str(v)
            # else: log_vals[k] = "NaN" # Optionally log NaNs

        self.logger.debug(f"Latest indicator values updated for {self.symbol}: {log_vals}")


    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci levels using Decimal precision."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)} points) for Fibonacci window ({window}) on {self.symbol}.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure 'high' and 'low' are numeric before max/min
            high_series = pd.to_numeric(df_slice["high"], errors='coerce')
            low_series = pd.to_numeric(df_slice["low"], errors='coerce')

            high_price_raw = high_series.dropna().max()
            low_price_raw = low_series.dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                self.logger.warning(f"Could not find valid high/low for Fibonacci calculation (Window: {window}) on {self.symbol}.")
                self.fib_levels_data = {}
                return {}

            high = Decimal(str(high_price_raw))
            low = Decimal(str(low_price_raw))
            diff = high - low

            levels = {}
            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    level_price = high - (diff * Decimal(str(level_pct)))
                    levels[level_name] = level_price.quantize(rounding_factor, rounding=ROUND_DOWN)
            else: # Handle zero range
                self.logger.debug(f"Fibonacci range is zero (High=Low={high}) for {self.symbol} (Window: {window}).")
                level_price_quantized = high.quantize(rounding_factor, rounding=ROUND_DOWN)
                for level_pct in FIB_LEVELS:
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            log_levels = {k: str(v) for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} (Window: {window}): {log_levels}")
            return levels

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Fibonacci error for {self.symbol}: Missing column '{e}'. Ensure 'high'/'low' exist.{RESET}")
            self.fib_levels_data = {}
            return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Determines price precision (decimal places) from market info."""
        try:
            # 1. Check precision.price (most reliable)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None:
                if isinstance(price_precision_val, int) and price_precision_val >= 0:
                    return price_precision_val # Direct decimal places
                try: # Assume it represents tick size
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0:
                        precision = abs(tick_size.normalize().as_tuple().exponent)
                        return precision
                except (TypeError, ValueError, InvalidOperation) as e:
                     self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")

            # 2. Fallback: Infer from limits.price.min (less reliable)
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_price_tick = Decimal(str(min_price_val))
                    if 0 < min_price_tick < Decimal('0.1'): # Heuristic: looks like tick size
                        precision = abs(min_price_tick.normalize().as_tuple().exponent)
                        return precision
                except (TypeError, ValueError, InvalidOperation) as e:
                    self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' for precision: {e}")

            # 3. Fallback: Infer from last close price (least reliable)
            last_close = self.indicator_values.get("Close")
            if isinstance(last_close, Decimal) and last_close > 0:
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
                          # If int, it's decimal places, calculate tick size
                          if isinstance(price_precision_val, int):
                               if price_precision_val >= 0:
                                    tick = Decimal('1e-' + str(price_precision_val))
                                    if tick > 0: return tick
                          else: # float or str, assume it IS the tick size
                               tick = Decimal(str(price_precision_val))
                               if tick > 0: return tick
                     except (TypeError, ValueError, InvalidOperation) as e:
                          self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")

            # 2. Fallback: Try limits.price.min
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                try:
                    min_tick = Decimal(str(min_price_val))
                    if 0 < min_tick < Decimal('0.1'): # Heuristic check
                        return min_tick
                except (TypeError, ValueError, InvalidOperation) as e:
                     self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' for tick size: {e}")

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using precision fallback.")

        # --- Final Fallback: Calculate from derived decimal places ---
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.debug(f"Using fallback tick size based on derived precision ({price_precision_places}): {fallback_tick}")
        return fallback_tick

    def get_amount_precision_places(self) -> int:
        """Determines amount precision (decimal places) from market info."""
        # Similar logic to get_price_precision, but using 'amount' field
        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None:
                if isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                    return amount_precision_val # Direct decimal places
                try: # Assume it represents step size
                    step_size = Decimal(str(amount_precision_val))
                    if step_size > 0:
                        precision = abs(step_size.normalize().as_tuple().exponent)
                        return precision
                except (TypeError, ValueError, InvalidOperation): pass

            min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_val is not None:
                try:
                    min_amount_step = Decimal(str(min_amount_val))
                    if 0 < min_amount_step < Decimal('1'): # Heuristic
                        precision = abs(min_amount_step.normalize().as_tuple().exponent)
                        return precision
                except (TypeError, ValueError, InvalidOperation): pass

        except Exception as e:
            self.logger.warning(f"Error determining amount precision for {self.symbol}: {e}.")

        default_precision = 8 # Common default for crypto amounts
        self.logger.warning(f"Could not determine amount precision for {self.symbol}. Using default: {default_precision}.")
        return default_precision

    def get_min_amount_step(self) -> Decimal:
        """Gets the minimum amount increment (step size) from market info."""
        # Similar logic to get_min_tick_size, using 'amount' field
        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None:
                if isinstance(amount_precision_val, (float, str, int)):
                     try:
                          if isinstance(amount_precision_val, int):
                               if amount_precision_val >= 0:
                                    step = Decimal('1e-' + str(amount_precision_val))
                                    if step > 0: return step
                          else:
                               step = Decimal(str(amount_precision_val))
                               if step > 0: return step
                     except (TypeError, ValueError, InvalidOperation): pass

            min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_val is not None:
                try:
                    min_step = Decimal(str(min_amount_val))
                    # Assume min limit might be the step size if it's small
                    if 0 < min_step < Decimal('1'): return min_step
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
        if not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal) and level_price > 0:
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    self.logger.warning(f"Invalid Fib level value: {name}={level_price}. Skipping.")

            level_distances.sort(key=lambda x: x['distance'])
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score."""
        ema_short = self.indicator_values.get("EMA_Short")
        ema_long = self.indicator_values.get("EMA_Long")
        close_decimal = self.indicator_values.get("Close")
        # Convert close to float for comparison, default to NaN
        current_price_float = float(close_decimal) if isinstance(close_decimal, Decimal) else np.nan

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price_float):
            self.logger.debug("EMA alignment check skipped: Missing values.")
            return np.nan

        if current_price_float > ema_short > ema_long: return 1.0 # Strong Bullish
        elif current_price_float < ema_short < ema_long: return -1.0 # Strong Bearish
        else: return 0.0 # Mixed / Crossing

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """Generates final trading signal (BUY/SELL/HOLD) based on weighted score."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {}

        if not self.indicator_values:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Indicator values empty.")
            return "HOLD"
        core_indicators_present = any(
            pd.notna(v) for k, v in self.indicator_values.items()
            if k not in ['Open', 'High', 'Low', 'Close', 'Volume']
        )
        if not core_indicators_present:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: All core indicator values are NaN.")
            return "HOLD"
        if pd.isna(current_price) or not isinstance(current_price, Decimal) or current_price <= 0:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).")
            return "HOLD"

        active_weights = self.weights # Already fetched in __init__
        if not active_weights:
            self.logger.error(f"Active weight set '{self.active_weight_set_name}' missing or empty. Cannot generate signal.")
            return "HOLD"

        # --- Iterate through configured indicators ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue
            weight_str = active_weights.get(indicator_key)
            if weight_str is None: continue # Skip if no weight defined

            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue # Skip zero weight indicators
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid weight format '{weight_str}' for '{indicator_key}'. Skipping.")
                continue

            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method_to_call = getattr(self, check_method_name)
                indicator_score_float = np.nan

                try:
                    if indicator_key == "orderbook":
                        if orderbook_data:
                            indicator_score_float = method_to_call(orderbook_data, current_price)
                        elif weight != 0:
                             self.logger.debug(f"Orderbook check skipped: No data provided.")
                    else:
                        indicator_score_float = method_to_call()

                except Exception as e:
                    self.logger.error(f"Error executing check {check_method_name}: {e}", exc_info=True)

                # Store score for debugging
                debug_scores[indicator_key] = f"{indicator_score_float:.3f}" if pd.notna(indicator_score_float) else "NaN"

                if pd.notna(indicator_score_float):
                    try:
                        score_decimal = Decimal(str(indicator_score_float))
                        # Clamp score to [-1, 1]
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_decimal))
                        final_signal_score += clamped_score * weight
                        total_weight_applied += weight
                        active_indicator_count += 1
                    except (ValueError, TypeError, InvalidOperation) as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} (Score: {indicator_score_float}, Weight: {weight}): {calc_err}")
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1
            elif weight != 0: # Log only if weighted but missing
                self.logger.warning(f"Indicator check method '{check_method_name}' not found for enabled/weighted indicator: {indicator_key}")

        # --- Determine Final Signal ---
        final_signal = "HOLD"
        if total_weight_applied == 0:
            self.logger.warning(f"No indicators contributed valid scores for {self.symbol}. Defaulting to HOLD.")
        else:
            try:
                threshold_str = self.config.get("signal_score_threshold", "1.5")
                threshold = Decimal(str(threshold_str))
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid signal_score_threshold '{threshold_str}'. Using default 1.5.")
                threshold = Decimal("1.5")

            if final_signal_score >= threshold: final_signal = "BUY"
            elif final_signal_score <= -threshold: final_signal = "SELL"

        # --- Log Summary ---
        price_prec = self.get_price_precision()
        log_msg = (
            f"Signal Summary ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Indicators=[Active:{active_indicator_count}, NaN:{nan_indicator_count}], "
            f"TotalWeight={total_weight_applied:.2f}, "
            f"FinalScore={final_signal_score:.4f} (Threshold: +/-{threshold:.2f}) "
            f"==> {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        self.logger.debug(f"  Indicator Scores ({self.symbol}): {debug_scores}")

        if final_signal == "BUY": self.signals = {"BUY": 1, "SELL": 0, "HOLD": 0}
        elif final_signal == "SELL": self.signals = {"BUY": 0, "SELL": 1, "HOLD": 0}
        else: self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}

        return final_signal

    # --- Indicator Check Methods (return float score -1.0 to 1.0 or np.nan) ---
    def _check_ema_alignment(self) -> float:
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
             self.logger.debug("EMA Alignment check skipped: Values not found.")
             return np.nan
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum")
        if pd.isna(momentum): return np.nan
        threshold = 0.1 # Example threshold, needs tuning
        if momentum > threshold: return 1.0
        elif momentum < -threshold: return -1.0
        else: return float(momentum / threshold) # Scale within threshold

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume") # Decimal
        volume_ma_float = self.indicator_values.get("Volume_MA") # Float
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))

        if pd.isna(current_volume) or not isinstance(current_volume, Decimal) or \
           pd.isna(volume_ma_float) or volume_ma_float <= 0:
            return np.nan
        try:
            volume_ma = Decimal(str(volume_ma_float))
            multiplier_decimal = Decimal(str(multiplier))
            if current_volume > volume_ma * multiplier_decimal: return 0.7 # High volume confirmation
            elif current_volume < volume_ma / multiplier_decimal: return -0.4 # Low volume lack of confirmation
            else: return 0.0 # Neutral volume
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.warning(f"Error during volume confirmation check: {e}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K")
        d = self.indicator_values.get("StochRSI_D")
        if pd.isna(k) or pd.isna(d): return np.nan
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))

        score = 0.0
        if k < oversold and d < oversold: score = 1.0
        elif k > overbought and d > overbought: score = -1.0

        diff = k - d
        if abs(diff) > 5: # Significant crossing potential
             score = max(score, 0.6) if diff > 0 else min(score, -0.6)
        elif k > d: score = max(score, 0.2) # Mildly bullish momentum
        elif k < d: score = min(score, -0.2) # Mildly bearish momentum

        if 40 < k < 60: score *= 0.5 # Dampen score in mid-range
        return score

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI")
        if pd.isna(rsi): return np.nan
        if rsi <= 30: return 1.0   # Oversold
        if rsi >= 70: return -1.0  # Overbought
        if rsi < 40: return 0.5
        if rsi > 60: return -0.5
        if 40 <= rsi <= 60: return (rsi - 50) / 50.0 # Scale mid-range
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI")
        if pd.isna(cci): return np.nan
        if cci <= -150: return 1.0
        if cci >= 150: return -1.0
        if cci < -80: return 0.6
        if cci > 80: return -0.6
        if cci > 0: return -0.1 # Slightly bearish tendency above zero
        if cci < 0: return 0.1  # Slightly bullish tendency below zero
        return 0.0

    def _check_wr(self) -> float:
        wr = self.indicator_values.get("Williams_R")
        if pd.isna(wr): return np.nan
        # WR range is -100 to 0
        if wr <= -80: return 1.0   # Oversold -> Buy
        if wr >= -20: return -1.0  # Overbought -> Sell
        if wr < -50: return 0.4    # Approaching midpoint from oversold
        if wr > -50: return -0.4    # Approaching midpoint from overbought
        return 0.0

    def _check_psar(self) -> float:
        psar_long = self.indicator_values.get("PSAR_long") # Price if long active, else NaN
        psar_short = self.indicator_values.get("PSAR_short") # Price if short active, else NaN

        long_active = pd.notna(psar_long)
        short_active = pd.notna(psar_short)

        if long_active and not short_active: return 1.0 # Uptrend
        elif short_active and not long_active: return -1.0 # Downtrend
        elif not long_active and not short_active: return np.nan # Indeterminate / Start of data
        else: # Should not happen (both active)
             self.logger.warning(f"PSAR check unusual state: Long={psar_long}, Short={psar_short}")
             return 0.0

    def _check_sma_10(self) -> float:
        sma_10 = self.indicator_values.get("SMA10")
        last_close = self.indicator_values.get("Close") # Decimal
        if pd.isna(sma_10) or pd.isna(last_close) or not isinstance(last_close, Decimal): return np.nan
        # Convert SMA float to Decimal for comparison
        sma_10_dec = Decimal(str(sma_10))
        if last_close > sma_10_dec: return 0.6
        elif last_close < sma_10_dec: return -0.6
        else: return 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP")
        last_close = self.indicator_values.get("Close") # Decimal
        if pd.isna(vwap) or pd.isna(last_close) or not isinstance(last_close, Decimal): return np.nan
        vwap_dec = Decimal(str(vwap))
        if last_close > vwap_dec: return 0.7
        elif last_close < vwap_dec: return -0.7
        else: return 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI")
        if pd.isna(mfi): return np.nan
        if mfi <= 20: return 1.0
        if mfi >= 80: return -1.0
        if mfi < 40: return 0.4
        if mfi > 60: return -0.4
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_lower = self.indicator_values.get("BB_Lower")
        bb_middle = self.indicator_values.get("BB_Middle")
        bb_upper = self.indicator_values.get("BB_Upper")
        last_close = self.indicator_values.get("Close") # Decimal
        if pd.isna(bb_lower) or pd.isna(bb_middle) or pd.isna(bb_upper) or \
           pd.isna(last_close) or not isinstance(last_close, Decimal): return np.nan

        # Convert band floats to Decimal
        try:
             bb_l, bb_m, bb_u = Decimal(str(bb_lower)), Decimal(str(bb_middle)), Decimal(str(bb_upper))
        except (ValueError, TypeError, InvalidOperation):
             self.logger.warning("Could not convert BB values to Decimal.")
             return np.nan

        if last_close <= bb_l: return 1.0 # Touch/Below Lower Band -> Buy Signal
        if last_close >= bb_u: return -1.0 # Touch/Above Upper Band -> Sell Signal

        # Position relative to middle band
        band_width = bb_u - bb_l
        if band_width > 0:
             # Scale score based on proximity to bands within the middle range
             if last_close > bb_m: # Above middle
                  proximity_to_upper = (last_close - bb_m) / (bb_u - bb_m) if (bb_u - bb_m) > 0 else Decimal(0)
                  return float(Decimal(0.5) * (1 - proximity_to_upper)) # Score decreases closer to upper band
             elif last_close < bb_m: # Below middle
                  proximity_to_lower = (bb_m - last_close) / (bb_m - bb_l) if (bb_m - bb_l) > 0 else Decimal(0)
                  return float(Decimal(-0.5) * (1 - proximity_to_lower)) # Score increases (less negative) closer to lower band
        return 0.0 # On middle band or bands collapsed

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes order book imbalance."""
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            self.logger.debug("Orderbook check skipped: No data or missing bids/asks.")
            return np.nan
        try:
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            num_levels = min(len(bids), len(asks), 10) # Check top 10 levels, or fewer if available
            if num_levels == 0: return 0.0 # Neutral if no common levels

            # Sum sizes (Decimal) within N levels
            bid_vol = sum(Decimal(str(b[1])) for b in bids[:num_levels] if len(b) == 2)
            ask_vol = sum(Decimal(str(a[1])) for a in asks[:num_levels] if len(a) == 2)
            total_vol = bid_vol + ask_vol

            if total_vol == 0: return 0.0 # Avoid division by zero

            # Order Book Imbalance (OBI)
            obi = (bid_vol - ask_vol) / total_vol
            score = float(obi) # Convert final ratio to float score

            self.logger.debug(
                f"Orderbook check ({self.symbol}): Top {num_levels} levels -> "
                f"BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, OBI={obi:.4f} -> Score={score:.4f}"
            )
            # Clamp score just in case (should already be -1 to 1)
            return max(-1.0, min(1.0, score))

        except (IndexError, ValueError, TypeError, InvalidOperation) as e:
            self.logger.warning(f"Orderbook analysis failed for {self.symbol}: {e}", exc_info=True)
            return np.nan

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates potential TP and initial SL based on entry estimate, signal, and ATR."""
        if signal not in ["BUY", "SELL"]:
            return entry_price_estimate, None, None

        atr_val = self.indicator_values.get("ATR") # Should be Decimal
        if not isinstance(atr_val, Decimal) or pd.isna(atr_val) or atr_val <= 0:
            self.logger.warning(f"Cannot calculate TP/SL for {signal}: Invalid ATR ({atr_val}).")
            return entry_price_estimate, None, None
        if not isinstance(entry_price_estimate, Decimal) or pd.isna(entry_price_estimate) or entry_price_estimate <= 0:
            self.logger.warning(f"Cannot calculate TP/SL for {signal}: Invalid entry price estimate ({entry_price_estimate}).")
            return entry_price_estimate, None, None

        try:
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", "1.0")))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", "1.5")))
            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))
            min_tick = self.get_min_tick_size()

            tp_offset = atr_val * tp_multiple
            sl_offset = atr_val * sl_multiple

            take_profit_raw: Optional[Decimal] = None
            stop_loss_raw: Optional[Decimal] = None

            if signal == "BUY":
                take_profit_raw = entry_price_estimate + tp_offset
                stop_loss_raw = entry_price_estimate - sl_offset
            elif signal == "SELL":
                take_profit_raw = entry_price_estimate - tp_offset
                stop_loss_raw = entry_price_estimate + sl_offset

            # Quantize TP/SL to Market Precision
            take_profit_quantized: Optional[Decimal] = None
            stop_loss_quantized: Optional[Decimal] = None

            if take_profit_raw is not None:
                 # Quantize TP conservatively (ROUND_DOWN for BUY, ROUND_UP for SELL - less profit potential)
                 # Or round normally? Let's round normally first.
                 # tp_rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP # Conservative
                 tp_rounding = ROUND_HALF_UP # Standard rounding
                 take_profit_quantized = take_profit_raw.quantize(rounding_factor, rounding=tp_rounding)

            if stop_loss_raw is not None:
                 # Quantize SL conservatively (ROUND_DOWN for BUY, ROUND_UP for SELL - wider SL)
                 sl_rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
                 stop_loss_quantized = stop_loss_raw.quantize(rounding_factor, rounding=sl_rounding)

            # --- Validation and Adjustments ---
            final_tp = take_profit_quantized
            final_sl = stop_loss_quantized

            # 1. Ensure SL is strictly beyond entry by at least one tick
            if final_sl is not None:
                if signal == "BUY" and final_sl >= entry_price_estimate:
                    final_sl = (entry_price_estimate - min_tick).quantize(rounding_factor, rounding=ROUND_DOWN)
                    self.logger.debug(f"Adjusted BUY SL below entry: {stop_loss_quantized} -> {final_sl}")
                elif signal == "SELL" and final_sl <= entry_price_estimate:
                    final_sl = (entry_price_estimate + min_tick).quantize(rounding_factor, rounding=ROUND_UP)
                    self.logger.debug(f"Adjusted SELL SL above entry: {stop_loss_quantized} -> {final_sl}")

            # 2. Ensure TP provides potential profit (strictly beyond entry)
            if final_tp is not None:
                if signal == "BUY" and final_tp <= entry_price_estimate:
                    self.logger.warning(f"{NEON_YELLOW}BUY TP calculation non-profitable (TP {final_tp} <= Entry {entry_price_estimate}). Setting TP to None.{RESET}")
                    final_tp = None
                elif signal == "SELL" and final_tp >= entry_price_estimate:
                    self.logger.warning(f"{NEON_YELLOW}SELL TP calculation non-profitable (TP {final_tp} >= Entry {entry_price_estimate}). Setting TP to None.{RESET}")
                    final_tp = None

            # 3. Ensure SL/TP are positive
            if final_sl is not None and final_sl <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in non-positive price ({final_sl}). Setting SL to None.{RESET}")
                final_sl = None
            if final_tp is not None and final_tp <= 0:
                self.logger.warning(f"{NEON_YELLOW}Take profit calculation resulted in non-positive price ({final_tp}). Setting TP to None.{RESET}")
                final_tp = None

            tp_str = f"{final_tp:.{price_precision}f}" if final_tp else "None"
            sl_str = f"{final_sl:.{price_precision}f}" if final_sl else "None"
            self.logger.debug(
                f"Calculated TP/SL for {signal}: EntryEst={entry_price_estimate:.{price_precision}f}, "
                f"ATR={atr_val:.{price_precision+2}f}, TP={tp_str} (Mult: {tp_multiple}), SL={sl_str} (Mult: {sl_multiple})"
            )
            return entry_price_estimate, final_tp, final_sl

        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"{NEON_RED}Error converting values during TP/SL calculation for {signal}: {e}{RESET}")
            return entry_price_estimate, None, None
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {signal}: {e}{RESET}", exc_info=True)
            return entry_price_estimate, None, None

# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches available balance for a specific currency with retries and robust parsing."""
    lg = logger
    try:
        balance_info = None
        # Prioritize specific account types for Bybit V5
        params = {}
        account_type_to_log = "default"
        if exchange.id == 'bybit':
            # Use 'CONTRACT' for derivatives, could also try 'UNIFIED'
            params = {'type': 'CONTRACT'}
            account_type_to_log = 'CONTRACT'

        lg.debug(f"Fetching balance for {currency} (Account: {account_type_to_log})...")
        # Use safe_api_call for fetching
        balance_info = safe_api_call(exchange.fetch_balance, lg, params=params)

        if not balance_info:
             lg.error(f"Failed to fetch balance info for {currency} after retries.")
             # Optionally try default fetch without params as fallback
             lg.debug("Attempting balance fetch with default parameters as fallback...")
             balance_info = safe_api_call(exchange.fetch_balance, lg)
             if not balance_info:
                  lg.error(f"Fallback balance fetch also failed for {currency}.")
                  return None

        # --- Parse the balance_info ---
        available_balance_str = None
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
                            # Prefer availableToWithdraw > availableBalance > walletBalance
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

        # 4. Fallback: Use 'total' if 'free' is unavailable (use with caution)
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
            # Allow zero balance, but log warning if negative
            if final_balance < 0:
                 lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Treating as zero.")
                 final_balance = Decimal('0')
            lg.info(f"Available {currency} balance: {final_balance:.4f}")
            return final_balance
        except (ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"Failed to convert balance string '{free_balance}' to Decimal for {currency}: {e}")
            return None

    except Exception as e:
        # Catch errors raised by safe_api_call or during parsing
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
             # Suggest alternatives for common symbols
             if symbol == "BTC/USDT": lg.warning(f"{NEON_YELLOW}Hint: For Bybit linear perpetual, try '{symbol}:USDT'{RESET}")
             return None

        market = exchange.market(symbol)
        if market:
            # --- Extract and Log Details ---
            market_type = market.get('type', 'unknown') # spot, future, swap, option
            is_contract = market.get('contract', False) or market_type in ['swap', 'future']
            is_linear = market.get('linear', False)
            is_inverse = market.get('inverse', False)
            contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "N/A" if not is_contract else "Unknown Contract"
            price_prec_val = market.get('precision', {}).get('price')
            amount_prec_val = market.get('precision', {}).get('amount')
            min_amount = market.get('limits', {}).get('amount', {}).get('min')
            max_amount = market.get('limits', {}).get('amount', {}).get('max')
            min_cost = market.get('limits', {}).get('cost', {}).get('min')
            max_cost = market.get('limits', {}).get('cost', {}).get('max')
            contract_size = market.get('contractSize', 'N/A')

            lg.debug(
                f"Market Info ({symbol}): ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                f"Type={market_type}, Contract={is_contract}, ContractType={contract_type}, Size={contract_size}, "
                f"Precision(Price/Amount): {price_prec_val}/{amount_prec_val}, "
                f"Limits(Amount Min/Max): {min_amount}/{max_amount}, "
                f"Limits(Cost Min/Max): {min_cost}/{max_cost}"
            )
            # Add custom flags for easier checks
            market['is_contract'] = is_contract
            market['is_linear'] = is_linear
            market['is_inverse'] = is_inverse
            return market
        else:
             lg.error(f"{NEON_RED}Market dictionary unexpectedly not found for validated symbol {symbol}.{RESET}")
             return None

    except ccxt.BadSymbol as e:
         lg.error(f"{NEON_RED}Symbol '{symbol}' is invalid or not supported by {exchange.id}: {e}{RESET}")
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Needed for formatting helpers
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """Calculates position size based on risk, SL, balance, and market constraints."""
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('is_inverse', False)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or not isinstance(balance, Decimal) or balance <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade}).")
        return None
    if initial_stop_loss_price is None or initial_stop_loss_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if entry_price is None or entry_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid entry_price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Size Calc Fail ({symbol}): SL price equals entry price.")
        return None
    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Size Calc Fail ({symbol}): Market info missing 'limits' or 'precision'.")
        return None
    if is_inverse:
        lg.error(f"{NEON_RED}Inverse contract sizing not fully implemented. Aborting sizing for {symbol}.{RESET}")
        return None # Exit for inverse contracts

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
            lg.error(f"Size Calc Fail ({symbol}): SL distance is zero/negative.")
            return None

        # Get Contract Size (value per contract, usually in base currency)
        contract_size_str = market_info.get('contractSize', '1') # Defaults to 1 for spot/linear if not specified
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0: raise ValueError("Contract size must be positive")
        except (ValueError, TypeError, InvalidOperation):
            lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}, using 1.")
            contract_size = Decimal('1')

        # --- Calculate Initial Size (Assuming Linear/Spot) ---
        # Formula: Size = RiskAmountQuote / RiskPerUnitQuote
        # RiskPerUnitQuote = StopLossDistanceQuote * ContractSizeValueQuote
        # For Linear: ContractSizeValueQuote = contract_size (in base) * entry_price (quote/base) -> this is wrong.
        # Let's simplify: RiskPerUnitQuote = SL_Distance_Quote_Per_Unit * ContractSize_Base_Units
        # For Spot: Size = RiskQuote / SL_Quote_Per_Base
        # For Linear: Contract represents 'contract_size' amount of Base currency.
        #             Risk per contract (in Quote) = SL_Distance_Quote_Per_Base * ContractSize_Base
        #             SizeInContracts = RiskQuote / (SL_Distance_Quote_Per_Base * ContractSize_Base)
        risk_per_contract_quote = sl_distance_per_unit * contract_size
        if risk_per_contract_quote <= 0:
             lg.error(f"Size Calc Fail ({symbol}): Denominator zero/negative in size calc (risk per contract: {risk_per_contract_quote}).")
             return None

        calculated_size = risk_amount_quote / risk_per_contract_quote

        if calculated_size <= 0:
            lg.error(f"Initial size calc resulted in zero/negative: {calculated_size}. RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_per_unit}, ContrSize={contract_size}")
            return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  ContractSize={contract_size}, Initial Calc. Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        # Use helper functions to get Decimal values for limits
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info) # Temp instance for helpers
        min_amount = Decimal(str(amount_limits.get('min', '0')))
        max_amount = Decimal(str(amount_limits.get('max', 'inf')))
        min_cost = Decimal(str(cost_limits.get('min', '0')))
        max_cost = Decimal(str(cost_limits.get('max', 'inf')))
        amount_step = analyzer.get_min_amount_step()

        adjusted_size = calculated_size

        # 1. Clamp by MIN/MAX AMOUNT limits (before step size adjustment)
        original_size = adjusted_size
        if adjusted_size < min_amount: adjusted_size = min_amount
        if adjusted_size > max_amount: adjusted_size = max_amount
        if adjusted_size != original_size:
             lg.warning(f"{NEON_YELLOW}Size adjusted by Amount Limits: {original_size:.8f} -> {adjusted_size:.8f} {size_unit} (Min: {min_amount}, Max: {max_amount}){RESET}")

        # 2. Apply Amount Step Size (Round DOWN - conservative)
        if amount_step > 0:
            original_size = adjusted_size
            adjusted_size = (adjusted_size // amount_step) * amount_step
            if adjusted_size != original_size:
                 lg.info(f"Applied Amount Step Size ({amount_step}): {original_size:.8f} -> {adjusted_size:.8f} {size_unit}")
        else:
            lg.warning(f"Amount step size is zero or negative ({amount_step}). Skipping step size adjustment.")

        # Re-check MIN amount after step-size rounding
        if adjusted_size < min_amount:
             lg.error(f"{NEON_RED}Size ({adjusted_size}) is below Min Amount ({min_amount}) after step size adjustment. Cannot place order.{RESET}")
             # This can happen if min_amount is not a multiple of amount_step
             return None

        # 3. Check COST limits with the step-adjusted size
        # Cost = Size * EntryPrice (for spot) or Size * EntryPrice * ContractSize (for linear)
        estimated_cost = adjusted_size * entry_price * contract_size
        lg.debug(f"  Cost Check: Final Adjusted Size={adjusted_size:.8f}, Estimated Cost={estimated_cost:.4f} {quote_currency} (Min: {min_cost}, Max: {max_cost})")

        if min_cost > 0 and estimated_cost < min_cost:
            lg.error(f"{NEON_RED}Estimated cost {estimated_cost:.4f} is below Min Cost {min_cost} after all adjustments. Cannot place order.{RESET}")
            lg.error(f"  >> Check if Min Amount ({min_amount}) and Min Cost ({min_cost}) limits conflict for this price/size.")
            return None
        if max_cost > 0 and estimated_cost > max_cost:
            lg.error(f"{NEON_RED}Estimated cost {estimated_cost:.4f} exceeds Max Cost {max_cost} after all adjustments. Cannot place order.{RESET}")
            # This implies risk % or balance leads to too large an order even after max amount check
            return None

        # --- Final Validation ---
        final_size = adjusted_size
        if final_size <= 0:
             lg.error(f"{NEON_RED}Final position size became zero or negative ({final_size}) after adjustments. Aborted.{RESET}")
             return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position using fetch_positions with robust parsing for Bybit V5."""
    lg = logger
    if not exchange.has.get('fetchPositions'):
        lg.warning(f"Exchange {exchange.id} does not support fetchPositions. Cannot check position status.")
        return None

    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        # Some exchanges require symbols list, others fetch all if symbol omitted
        # Bybit V5: Can fetch specific symbol or category=linear/inverse
        params = {}
        if exchange.id == 'bybit':
            # Try fetching specific symbol directly, which works for unified/contract v5
            params = {'symbol': symbol} # Should use the exchange's ID for the symbol
            # Alternative: params = {'category': 'linear'} # Fetch all linear positions

        # Use safe_api_call for the fetch operation
        positions: List[Dict] = safe_api_call(exchange.fetch_positions, lg, symbols=[symbol], params=params) # Pass symbol in list

        if positions is None: # safe_api_call failed after retries
             lg.error("Position fetch failed after retries.")
             # Option: Try fetching all positions as a fallback?
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
            if pos_symbol != symbol: continue # Ensure correct symbol

            pos_size_str = None
            # Try standard 'contracts', fallback to 'info.size' (Bybit V5)
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif isinstance(pos.get('info'), dict) and pos['info'].get('size') is not None:
                 pos_size_str = str(pos['info']['size'])

            if pos_size_str is None: continue # Skip if size unavailable

            try:
                position_size = Decimal(pos_size_str)
                if abs(position_size) > size_threshold:
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break # Assume only one position per symbol/side/mode needed
            except (ValueError, TypeError, InvalidOperation) as parse_err:
                lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol}: {parse_err}")

        # --- Post-Process the found active position ---
        if active_position:
            try:
                market = exchange.market(symbol) # Get market info again for context
                analyzer = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market) # Temp instance for helpers
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
                    else: lg.warning(f"Position size {size_decimal} near zero, cannot determine side."); return None
                    active_position['side'] = side
                    lg.debug(f"Inferred position side as '{side}' based on size.")

                # Standardize Entry Price (Decimal)
                entry_price_str = active_position.get('entryPrice') or active_position.get('info', {}).get('avgPrice')
                active_position['entryPriceDecimal'] = Decimal(str(entry_price_str)) if entry_price_str else None

                # Standardize Liq Price (Decimal)
                liq_price_str = active_position.get('liquidationPrice') or active_position.get('info', {}).get('liqPrice')
                active_position['liquidationPriceDecimal'] = Decimal(str(liq_price_str)) if liq_price_str else None

                # Standardize PNL (Decimal)
                pnl_str = active_position.get('unrealizedPnl') or active_position.get('info', {}).get('unrealisedPnl')
                active_position['unrealizedPnlDecimal'] = Decimal(str(pnl_str)) if pnl_str else None

                # Extract SL/TP/TSL from 'info' (Bybit V5) and store as Decimal where applicable
                info_dict = active_position.get('info', {})
                sl_str = info_dict.get('stopLoss')
                tp_str = info_dict.get('takeProfit')
                tsl_dist_str = info_dict.get('trailingStop') # Distance value
                tsl_act_str = info_dict.get('activePrice') # Activation price

                active_position['stopLossPriceDecimal'] = Decimal(str(sl_str)) if sl_str and str(sl_str) != '0' else None
                active_position['takeProfitPriceDecimal'] = Decimal(str(tp_str)) if tp_str and str(tp_str) != '0' else None
                active_position['trailingStopLossValueDecimal'] = Decimal(str(tsl_dist_str)) if tsl_dist_str and str(tsl_dist_str) != '0' else None
                active_position['trailingStopActivationPriceDecimal'] = Decimal(str(tsl_act_str)) if tsl_act_str and str(tsl_act_str) != '0' else None

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
                pnl_fmt = f"{active_position['unrealizedPnlDecimal']:.{price_prec}f}" if active_position['unrealizedPnlDecimal'] else 'N/A' # Use price precision for PNL
                sl_fmt = f"{active_position['stopLossPriceDecimal']:.{price_prec}f}" if active_position['stopLossPriceDecimal'] else 'N/A'
                tp_fmt = f"{active_position['takeProfitPriceDecimal']:.{price_prec}f}" if active_position['takeProfitPriceDecimal'] else 'N/A'
                tsl_d_fmt = f"{active_position['trailingStopLossValueDecimal']:.{price_prec}f}" if active_position['trailingStopLossValueDecimal'] else 'N/A' # Format distance like price
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
         # Handle cases where fetching single symbol isn't supported
         lg.warning(f"Fetching single position failed ({e}). Trying to fetch all positions...")
         try:
             all_positions = safe_api_call(exchange.fetch_positions, lg) # Fetch all
             if all_positions:
                  positions = [p for p in all_positions if p.get('symbol') == symbol]
                  lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
                  # Now re-run the processing logic from above
                  # This part is duplicated, consider refactoring into a helper
                  active_position = None
                  size_threshold = Decimal('1e-9')
                  for pos in positions:
                      # ... (repeat processing logic as above) ...
                      # Standardize Size (always Decimal)
                        pos_size_str = None
                        if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
                        elif isinstance(pos.get('info'), dict) and pos['info'].get('size') is not None: pos_size_str = str(pos['info']['size'])
                        if pos_size_str is None: continue
                        try:
                            position_size = Decimal(pos_size_str)
                            if abs(position_size) > size_threshold:
                                active_position = pos
                                break
                        except (ValueError, TypeError, InvalidOperation): continue
                  if active_position:
                      # ... (repeat post-processing logic as above) ...
                       return active_position # Return processed position
                  else: logger.info(f"No active open position found for {symbol} in fallback fetch."); return None
             else:
                 lg.error("Fallback fetch of all positions also failed.")
                 return None
         except Exception as fallback_e:
              lg.error(f"Error during fallback fetch of all positions: {fallback_e}", exc_info=True)
              return None
    except Exception as e:
        # Catch errors raised by safe_api_call or other unexpected issues
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
        # Bybit V5 requires buy/sell leverage, ensure string format
        if exchange.id == 'bybit':
            leverage_str = str(leverage)
            params = {'buyLeverage': leverage_str, 'sellLeverage': leverage_str}
            lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        # Use safe_api_call to wrap the exchange call
        response = safe_api_call(exchange.set_leverage, lg, leverage, symbol, params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # Basic success check (no exception raised by safe_api_call)
        lg.info(f"{NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Check position details for confirmation).{RESET}")
        return True

    except ccxt.ExchangeError as e:
        # Handle specific errors more informatively
        err_str = str(e).lower()
        code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {code}){RESET}")
        if exchange.id == 'bybit':
            if code == 110045 or "leverage not modified" in err_str:
                 lg.info(f"{NEON_YELLOW}Leverage for {symbol} likely already set to {leverage}x (Exchange: {e}).{RESET}")
                 return True # Treat as success
            elif code in [110028, 110009, 110055] or "margin mode" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) setting. May need `set_margin_mode` first or ensure compatibility.{RESET}")
            elif code == 110044 or "risk limit" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed Risk Limit tier. Check Bybit Risk Limits.{RESET}")
            elif code == 110013 or "parameter error" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid/out of range for {symbol}.{RESET}")
    except Exception as e:
        # Catch errors raised by safe_api_call (like AuthError, Max Retries) or unexpected ones
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
        amount_float = float(position_size) # CCXT generally requires float amount
        if amount_float <= 0: raise ValueError("Position size must be positive")
    except (ValueError, TypeError) as e:
        lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Invalid size {position_size}: {e}")
        return None
    if order_type == 'limit':
         if limit_price is None or not isinstance(limit_price, Decimal) or limit_price <= 0:
             lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Limit order needs valid positive limit_price.")
             return None
         try: # CCXT needs price as float
              price_float = float(limit_price)
         except (ValueError, TypeError):
              lg.error(f"Trade Aborted ({action_desc} {side} {symbol}): Invalid limit_price format {limit_price}.")
              return None
    else: price_float = None


    # --- Prepare Parameters ---
    # Base parameters needed by many exchanges/modes
    order_params = {
        'reduceOnly': reduce_only,
        # Bybit V5: positionIdx often needed (0 for one-way, 1/2 for hedge buy/sell)
        # Assume one-way mode if not specified otherwise
        # 'positionIdx': 0, # Uncomment or make configurable if needed
    }
    if order_type == 'market' and reduce_only:
         # Use IOC or FOK for market close to avoid unexpected partial fills hanging
         order_params['timeInForce'] = 'IOC' # ImmediateOrCancel
         # order_params['timeInForce'] = 'FOK' # FillOrKill (alternative)

    # Merge external params carefully
    if params:
        base_params = order_params.copy() # Keep our base settings safe
        base_params.update(params) # Add external ones
        order_params = base_params # Use the merged dict


    # --- Log Order Details ---
    log_price = f"Limit @ {limit_price}" if order_type == 'limit' else "Market"
    amt_prec = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info).get_amount_precision_places()
    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type.upper()} order for {symbol}:")
    lg.info(f"  Size: {amount_float:.{amt_prec}f} {size_unit}")
    if order_type == 'limit': lg.info(f"  Limit Price: {limit_price}")
    lg.info(f"  ReduceOnly: {reduce_only}")
    lg.info(f"  Params: {order_params}")

    # --- Execute Order via safe_api_call ---
    try:
        order = safe_api_call(
            exchange.create_order, lg,
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=price_float, # Will be None for market orders
            params=order_params
        )

        if order and order.get('id'):
            order_id = order.get('id')
            status = order.get('status', 'unknown')
            filled = order.get('filled', 0.0)
            avg_price = order.get('average')
            lg.info(f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET}")
            lg.info(f"  ID: {order_id}, Initial Status: {status}, Filled: {filled}, AvgPrice: {avg_price}")
            lg.debug(f"Raw order response ({action_desc} {side} {symbol}): {order}")
            return order
        elif order is None: # safe_api_call failed after retries
             lg.error(f"{NEON_RED}Order placement failed for {symbol} after retries.{RESET}")
             return None
        else: # Call succeeded but response missing ID (unusual)
             lg.error(f"{NEON_RED}Order placement call succeeded but response lacks ID for {symbol}. Response: {order}{RESET}")
             return None

    # --- Handle Specific CCXT Exceptions Raised by safe_api_call ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds for {action_desc} {side} {symbol}: {e}{RESET}")
        try: balance = fetch_balance(exchange, QUOTE_CURRENCY, lg); lg.info(f"Current Balance: {balance} {QUOTE_CURRENCY}")
        except: pass
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {action_desc} {side} {symbol}: {e}{RESET}")
        lg.error(f"  > Used: amount={amount_float}, price={limit_price}, params={order_params}")
        err_str = str(e).lower()
        if "tick size" in err_str: lg.error("  >> Hint: Check limit_price aligns with market tick size.")
        if "step size" in err_str: lg.error("  >> Hint: Check position_size aligns with market amount step size.")
        if "minnotional" in err_str or "cost" in err_str: lg.error("  >> Hint: Order cost (size*price) might be below minimum required.")
        if "reduce-only" in err_str or (getattr(e, 'code', None) == 110014 and exchange.id == 'bybit'):
             lg.error(f"{NEON_YELLOW}  >> Hint: Reduce-only failed. Position closed? Size/side wrong?{RESET}")
    except ccxt.ExchangeError as e:
        code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action_desc} order ({symbol}): {e} (Code: {code}){RESET}")
        if reduce_only and code == 110025 and exchange.id == 'bybit': # Position closed/not found
            lg.warning(f"{NEON_YELLOW} >> Hint (Bybit 110025): Position might have closed before reduce-only order placed.{RESET}")
    except Exception as e:
        # Catch other errors (like AuthError, RequestTimeout from safe_api_call, or unexpected)
        lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol}: {e}{RESET}", exc_info=False)

    return None


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Contains needed context like side, positionIdx
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # Price distance/offset
    tsl_activation_price: Optional[Decimal] = None,
) -> bool:
    """Internal helper using Bybit V5 API to set SL, TP, or TSL."""
    lg = logger
    if 'bybit' not in exchange.id.lower():
        lg.error(f"Protection setting via private_post currently only for Bybit.")
        return False
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not contract).")
        return False
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position info.")
        return False

    pos_side = position_info.get('side')
    entry_price = position_info.get('entryPriceDecimal')
    pos_idx = 0 # Default for One-Way
    try: # Get positionIdx from info if available (for Hedge Mode)
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: pos_idx = int(pos_idx_val)
    except (ValueError, TypeError): lg.warning("Could not parse positionIdx, using default 0.")

    if pos_side not in ['long', 'short']:
        lg.error(f"Cannot set protection: Invalid position side ('{pos_side}').")
        return False
    if entry_price is None:
        lg.error(f"Cannot set protection: Missing entry price in position info.")
        return False

    # --- Validate Protection Parameters ---
    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0)

    # Validate SL/TP relative to entry price and side
    if has_sl:
         if pos_side == 'long' and stop_loss_price >= entry_price:
              lg.error(f"Invalid SL for LONG: SL price {stop_loss_price} >= Entry price {entry_price}. Ignoring SL.")
              has_sl = False
         elif pos_side == 'short' and stop_loss_price <= entry_price:
              lg.error(f"Invalid SL for SHORT: SL price {stop_loss_price} <= Entry price {entry_price}. Ignoring SL.")
              has_sl = False
    if has_tp:
         if pos_side == 'long' and take_profit_price <= entry_price:
              lg.error(f"Invalid TP for LONG: TP price {take_profit_price} <= Entry price {entry_price}. Ignoring TP.")
              has_tp = False
         elif pos_side == 'short' and take_profit_price >= entry_price:
              lg.error(f"Invalid TP for SHORT: TP price {take_profit_price} >= Entry price {entry_price}. Ignoring TP.")
              has_tp = False
    # Validate TSL Activation relative to entry
    if has_tsl:
        if pos_side == 'long' and tsl_activation_price <= entry_price:
            lg.error(f"Invalid TSL Act. for LONG: Act. price {tsl_activation_price} <= Entry price {entry_price}. Ignoring TSL.")
            has_tsl = False
        elif pos_side == 'short' and tsl_activation_price >= entry_price:
            lg.error(f"Invalid TSL Act. for SHORT: Act. price {tsl_activation_price} >= Entry price {entry_price}. Ignoring TSL.")
            has_tsl = False
    # Check if SL and TP are the same
    if has_sl and has_tp and stop_loss_price == take_profit_price:
         lg.error(f"Invalid protection: Stop Loss price ({stop_loss_price}) cannot be equal to Take Profit price ({take_profit_price}). Ignoring both.")
         has_sl = False
         has_tp = False

    if not has_sl and not has_tp and not has_tsl:
         lg.info(f"No valid protection parameters provided or remaining after validation for {symbol} (PosIdx: {pos_idx}).")
         return True # No action needed/possible

    # --- Prepare API Parameters ---
    category = 'linear' if market_info.get('is_linear', True) else 'inverse'
    params = {
        'category': category,
        'symbol': market_info['id'], # Use exchange-specific ID
        'tpslMode': 'Full', # Apply to whole position
        'tpTriggerBy': 'LastPrice',
        'slTriggerBy': 'LastPrice',
        'positionIdx': pos_idx
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} PosIdx: {pos_idx}):"]

    # --- Format and Add Protection Params ---
    try:
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info) # Temp instance for helpers
        price_prec = analyzer.get_price_precision()
        min_tick = analyzer.get_min_tick_size()

        # Helper to format price string using ccxt price_to_precision
        def format_price_str(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try: return exchange.price_to_precision(symbol, float(price_decimal))
            except Exception as e: lg.warning(f"Failed price formatting ({price_decimal}): {e}"); return None

        # Helper to format distance value string using decimal_to_precision based on tick size
        def format_distance_str(dist_decimal: Optional[Decimal]) -> Optional[str]:
             if not isinstance(dist_decimal, Decimal) or dist_decimal <= 0: return None
             try:
                 dist_prec = abs(min_tick.normalize().as_tuple().exponent) if min_tick > 0 else price_prec
                 formatted = exchange.decimal_to_precision(dist_decimal, exchange.ROUND, dist_prec, exchange.NO_PADDING)
                 # Ensure formatted distance is at least min tick
                 if min_tick > 0 and Decimal(formatted) < min_tick:
                      formatted = str(min_tick)
                 return formatted
             except Exception as e: lg.warning(f"Failed distance formatting ({dist_decimal}): {e}"); return None

        # Set TSL first (overrides SL on Bybit V5 if set)
        if has_tsl:
            tsl_dist_fmt = format_distance_str(trailing_stop_distance)
            tsl_act_fmt = format_price_str(tsl_activation_price)
            if tsl_dist_fmt and tsl_act_fmt:
                params['trailingStop'] = tsl_dist_fmt
                params['activePrice'] = tsl_act_fmt
                log_parts.append(f"  Trailing SL: Dist={tsl_dist_fmt}, Act={tsl_act_fmt}")
                has_sl = False # Mark fixed SL as inactive if TSL is set
                lg.debug("TSL parameters added. Fixed SL will be ignored by Bybit V5.")
            else: lg.error("Failed to format TSL parameters. TSL will not be set.")

        # Set Fixed SL only if TSL was NOT set
        if has_sl:
            sl_fmt = format_price_str(stop_loss_price)
            if sl_fmt: params['stopLoss'] = sl_fmt; log_parts.append(f"  Fixed SL: {sl_fmt}")
            else: lg.error("Failed to format Fixed SL price. Fixed SL will not be set.")

        # Set Fixed TP
        if has_tp:
            tp_fmt = format_price_str(take_profit_price)
            if tp_fmt: params['takeProfit'] = tp_fmt; log_parts.append(f"  Fixed TP: {tp_fmt}")
            else: lg.error("Failed to format Fixed TP price. Fixed TP will not be set.")

    except Exception as fmt_err:
         lg.error(f"Error during formatting of protection parameters: {fmt_err}", exc_info=True)
         return False

    # --- Check if any protection parameters were successfully added ---
    if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
        lg.warning(f"No valid protection parameters could be formatted for {symbol} (PosIdx: {pos_idx}). No API call made.")
        return False # Failed to format intended protection

    # --- Make the API Call via safe_api_call ---
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', {params})")
    try:
        response = safe_api_call(exchange.private_post, lg, '/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response: {response}")

        # --- Parse Bybit V5 Response ---
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {})

        if ret_code == 0:
            if "not modified" in ret_msg.lower():
                 lg.info(f"{NEON_YELLOW}Position protection already set or only partially modified for {symbol} (PosIdx: {pos_idx}). Response: {ret_msg}{RESET}")
            else:
                 lg.info(f"{NEON_GREEN}Position protection set/updated successfully for {symbol} (PosIdx: {pos_idx}).{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Failed to set protection for {symbol} (PosIdx: {pos_idx}): {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            # Add specific hints based on error codes
            if ret_code == 110013: lg.error(f"{NEON_YELLOW} >> Hint (110013 - Param Error): Check prices vs entry/current, tick size, TSL values.{RESET}")
            elif ret_code == 110036: lg.error(f"{NEON_YELLOW} >> Hint (110036 - TSL Price Invalid): Activation price '{params.get('activePrice')}' likely invalid.{RESET}")
            elif ret_code == 110086: lg.error(f"{NEON_YELLOW} >> Hint (110086): SL price equals TP price.{RESET}")
            elif ret_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint (110043): Position status prevents modification (liquidation?).{RESET}")
            elif ret_code == 110025: lg.error(f"{NEON_YELLOW} >> Hint (110025): Position not found/closed, or positionIdx mismatch?{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: TSL distance '{params.get('trailingStop')}' likely invalid (size/tick).{RESET}")
            return False

    except Exception as e:
        # Catches errors from safe_api_call (Max retries, AuthError, etc.)
        lg.error(f"{NEON_RED}Failed protection API call for {symbol}: {e}{RESET}", exc_info=False)
        return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional TP to set alongside
) -> bool:
    """Calculates and sets Exchange-Native Trailing Stop Loss using _set_position_protection."""
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.debug(f"TSL disabled in config for {symbol}. Skipping setup.")
        return False # Indicate TSL not set

    try:
        callback_rate = Decimal(str(config["trailing_stop_callback_rate"]))
        activation_percentage = Decimal(str(config["trailing_stop_activation_percentage"]))
        entry_price = position_info.get('entryPriceDecimal')
        side = position_info.get('side')

        if callback_rate <= 0: raise ValueError("callback_rate must be positive")
        if activation_percentage < 0: raise ValueError("activation_percentage cannot be negative")
        if entry_price is None or entry_price <= 0: raise ValueError("Missing/invalid entry price")
        if side not in ['long', 'short']: raise ValueError("Missing/invalid position side")

    except (ValueError, TypeError, KeyError, InvalidOperation) as e:
        lg.error(f"{NEON_RED}Invalid TSL config or position info ({symbol}): {e}. Cannot calculate TSL.{RESET}")
        return False

    try:
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance
        price_prec = analyzer.get_price_precision()
        price_rounding = Decimal('1e-' + str(price_prec))
        min_tick_size = analyzer.get_min_tick_size()

        # 1. Calculate Activation Price
        activation_offset = entry_price * activation_percentage
        activation_price: Optional[Decimal] = None
        if side == 'long':
            raw_activation = entry_price + activation_offset
            activation_price = raw_activation.quantize(price_rounding, rounding=ROUND_UP)
            # Ensure activation > entry (by at least one tick)
            if activation_price <= entry_price:
                 activation_price = (entry_price + min_tick_size).quantize(price_rounding, rounding=ROUND_UP)
                 lg.debug(f"Adjusted LONG TSL activation to tick above entry: {activation_price}")
        else: # short
            raw_activation = entry_price - activation_offset
            activation_price = raw_activation.quantize(price_rounding, rounding=ROUND_DOWN)
            # Ensure activation < entry (by at least one tick)
            if activation_price >= entry_price:
                 activation_price = (entry_price - min_tick_size).quantize(price_rounding, rounding=ROUND_DOWN)
                 lg.debug(f"Adjusted SHORT TSL activation to tick below entry: {activation_price}")

        if activation_price is None or activation_price <= 0:
             lg.error(f"{NEON_RED}Invalid TSL activation price calculated ({activation_price}). Cannot set TSL.{RESET}")
             return False

        # 2. Calculate Trailing Distance (based on callback * ACTIVATION price)
        # Bybit V5 requires the distance value. Round to tick size.
        trailing_distance_raw = activation_price * callback_rate
        trailing_distance: Optional[Decimal] = None
        if min_tick_size > 0:
             # Round distance UP to nearest tick increment (conservative trail)
             trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
             # Ensure distance is at least one tick
             if trailing_distance < min_tick_size: trailing_distance = min_tick_size
        else: # Fallback if tick size is zero (shouldn't happen)
             trailing_distance = trailing_distance_raw.quantize(price_rounding, rounding=ROUND_UP) # Round like price

        if trailing_distance is None or trailing_distance <= 0:
             lg.error(f"{NEON_RED}Invalid TSL distance calculated ({trailing_distance}). Cannot set TSL.{RESET}")
             return False

        lg.info(f"Calculated TSL Parameters for {symbol} ({side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_prec}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price (Target): {activation_price:.{price_prec}f}")
        lg.info(f"  => Trailing Distance (Target): {trailing_distance:.{price_prec}f}")
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0:
             lg.info(f"  Take Profit (Target): {take_profit_price:.{price_prec}f}")

        # 3. Call internal helper to set protection
        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info,
            position_info=position_info, logger=lg,
            stop_loss_price=None, # TSL overrides fixed SL on Bybit V5
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Performs one cycle of analysis and trading logic for a single symbol."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    try:
        # --- Get Market Info ---
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info: raise ValueError(f"Failed to get market info for {symbol}.")

        # --- Fetch Data ---
        ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
        if not ccxt_interval: raise ValueError(f"Invalid interval '{config['interval']}'.")

        kline_limit = 500 # Ample data for indicators
        klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
        if klines_df.empty or len(klines_df) < 50:
            raise ValueError(f"Insufficient kline data ({len(klines_df)}) for {symbol}.")

        current_price = fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price is None:
            lg.warning(f"Failed ticker price fetch. Using last close from klines.")
            try:
                # Check if 'close' column is Decimal or float before converting
                last_close_val = klines_df['close'].iloc[-1]
                if pd.notna(last_close_val):
                    current_price = Decimal(str(last_close_val))
                    if current_price <= 0: raise ValueError("Last close price is non-positive.")
                    lg.info(f"Using last close price: {current_price}")
                else: raise ValueError("Last close price is NaN.")
            except (IndexError, KeyError, ValueError, TypeError, InvalidOperation) as e:
                raise ValueError(f"Failed to get valid last close price from klines: {e}")

        # Fetch order book if needed
        orderbook_data = None
        active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
        if config.get("indicators",{}).get("orderbook", False) and float(active_weights.get("orderbook", 0)) != 0:
            lg.debug(f"Fetching order book for {symbol}...")
            orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)
            if not orderbook_data: lg.warning(f"Failed to fetch orderbook for {symbol}, proceeding without.")
        else: lg.debug(f"Orderbook analysis skipped (Disabled/Zero Weight).")

        # --- Analyze Data ---
        analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
        if not analyzer.indicator_values:
            raise ValueError(f"Indicator calculation failed for {symbol}.")

        # --- Generate Signal ---
        signal = analyzer.generate_trading_signal(current_price, orderbook_data)

        # --- Calculate Potential TP/SL (based on current price estimate) ---
        _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal)
        price_prec = analyzer.get_price_precision()
        min_tick_size = analyzer.get_min_tick_size()
        current_atr = analyzer.indicator_values.get("ATR") # Decimal

        # --- Log Analysis Summary ---
        lg.info(f"Current Price: {current_price:.{price_prec}f}")
        lg.info(f"ATR: {current_atr:.{price_prec+2}f}" if isinstance(current_atr, Decimal) else 'ATR: N/A')
        lg.info(f"Calc. Initial SL (sizing): {sl_calc if sl_calc else 'N/A'}")
        lg.info(f"Calc. Initial TP (target): {tp_calc if tp_calc else 'N/A'}")
        tsl_enabled = config.get('enable_trailing_stop')
        be_enabled = config.get('enable_break_even')
        time_exit_minutes = config.get('time_based_exit_minutes')
        time_exit_str = f"{time_exit_minutes} min" if time_exit_minutes else "Disabled"
        lg.info(f"Position Mgmt: TSL={'On' if tsl_enabled else 'Off'}, BE={'On' if be_enabled else 'Off'}, TimeExit={time_exit_str}")

        # --- Trading Execution Check ---
        if not config.get("enable_trading", False):
            lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
            return # Exit function here

        # ==============================================
        # === Position Management Logic            ===
        # ==============================================
        open_position = get_open_position(exchange, symbol, lg)

        # --- Scenario 1: No Open Position ---
        if open_position is None:
            # Check concurrent positions (this simplistic check assumes this script is the only one trading this symbol)
            # A more robust check might involve querying all orders or using external state management.
            # For now, rely on get_open_position returning None as indication we can enter.
            # If max_concurrent_positions > 1, this logic needs modification.
            if config.get("max_concurrent_positions", 1) > 1:
                 lg.warning("max_concurrent_positions > 1 logic not fully implemented in this basic check.")

            if signal in ["BUY", "SELL"]:
                lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***")

                balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if balance is None or balance <= 0:
                    raise ValueError("Cannot fetch balance or balance is zero/negative.")
                if sl_calc is None:
                    raise ValueError("Initial SL calculation failed. Cannot calculate position size.")

                if market_info.get('is_contract', False):
                    leverage = int(config.get("leverage", 1))
                    if leverage > 0:
                        if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                            raise ValueError(f"Failed to set leverage to {leverage}x.")
                    else: lg.info(f"Leverage setting skipped (Config: {leverage}).")

                position_size = calculate_position_size(balance, config["risk_per_trade"], sl_calc, current_price, market_info, exchange, lg)
                if position_size is None or position_size <= 0:
                    raise ValueError(f"Position size calculation failed ({position_size}).")

                entry_order_type = config.get("entry_order_type", "market")
                limit_entry_price: Optional[Decimal] = None
                if entry_order_type == "limit":
                    offset_buy = Decimal(str(config.get("limit_order_offset_buy", "0.0005")))
                    offset_sell = Decimal(str(config.get("limit_order_offset_sell", "0.0005")))
                    rounding_factor = Decimal('1e-' + str(price_prec))
                    if signal == "BUY":
                        raw_limit = current_price * (Decimal(1) - offset_buy)
                        limit_entry_price = raw_limit.quantize(rounding_factor, rounding=ROUND_DOWN)
                    else: # SELL
                        raw_limit = current_price * (Decimal(1) + offset_sell)
                        limit_entry_price = raw_limit.quantize(rounding_factor, rounding=ROUND_UP)
                    if limit_entry_price <= 0:
                         lg.error(f"Limit price calc resulted in {limit_entry_price}. Switching to Market order.")
                         entry_order_type = "market"; limit_entry_price = None
                    else: lg.info(f"Calculated Limit Entry Price for {signal}: {limit_entry_price}")

                lg.info(f"==> Placing {signal} {entry_order_type.upper()} order | Size: {position_size} <==")
                trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, entry_order_type, limit_entry_price, reduce_only=False)

                if trade_order and trade_order.get('id'):
                    order_id = trade_order['id']
                    order_status = trade_order.get('status')

                    # Post-order actions depend on type and status
                    # Market orders: Wait and confirm position, then set protection
                    # Limit orders (open): Wait for next cycle
                    # Limit orders (closed immediately): Treat like market order
                    if order_status == 'closed' or entry_order_type == 'market':
                         if entry_order_type == 'market':
                              confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                              lg.info(f"Market order {order_id} placed. Waiting {confirm_delay}s for confirmation...")
                              time.sleep(confirm_delay)
                         else: # Limit filled immediately
                              lg.info(f"Limit order {order_id} filled immediately. Confirming position...")
                              time.sleep(2) # Short delay

                         lg.info(f"Attempting position confirmation for {symbol}...")
                         confirmed_position = get_open_position(exchange, symbol, lg)

                         if confirmed_position:
                             lg.info(f"{NEON_GREEN}Position Confirmed after order {order_id}!{RESET}")
                             # --- Set Protection Based on Actual Entry ---
                             try:
                                 entry_price_actual = confirmed_position.get('entryPriceDecimal')
                                 if not isinstance(entry_price_actual, Decimal) or entry_price_actual <= 0:
                                     lg.warning(f"Could not get valid actual entry price. Using estimate {current_price} for protection.")
                                     entry_price_actual = current_price # Fallback

                                 lg.info(f"Actual Entry Price: ~{entry_price_actual:.{price_prec}f}")
                                 _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                                 protection_set = False
                                 if config.get("enable_trailing_stop", False):
                                      lg.info(f"Setting Exchange Trailing Stop Loss (TP target: {tp_final})...")
                                      protection_set = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_position, config, lg, tp_final)
                                 else:
                                      lg.info(f"Setting Fixed SL ({sl_final}) and TP ({tp_final})...")
                                      if sl_final or tp_final:
                                          protection_set = _set_position_protection(exchange, symbol, market_info, confirmed_position, lg, sl_final, tp_final)
                                      else: lg.warning("Fixed SL/TP calculation failed. No fixed protection set.")

                                 if protection_set: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                                 else: lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ===\n{NEON_YELLOW}>>> MANUAL MONITORING REQUIRED! <<<")
                             except Exception as post_trade_err:
                                  lg.error(f"{NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                                  lg.warning(f"{NEON_YELLOW}Position open but protection setup failed. Manual check needed!")
                         else:
                             lg.error(f"{NEON_RED}Order {order_id} placed/filled, but FAILED TO CONFIRM open position! Manual check needed!{RESET}")

                    elif order_status == 'open' and entry_order_type == 'limit':
                        lg.info(f"Limit order {order_id} placed and is OPEN. Will check status next cycle.")
                        # Store order_id if monitoring specific orders is needed

                    else: # Order failed, cancelled, rejected etc.
                        lg.error(f"Order {order_id} placement resulted in status: {order_status}. Trade did not open as expected.")
                else:
                    lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). See previous logs. ===")
            else: # signal == HOLD
                lg.info(f"Signal is HOLD and no position exists for {symbol}. No action.")

        # --- Scenario 2: Existing Open Position ---
        else:
            pos_side = open_position.get('side', 'unknown')
            pos_size = open_position.get('contractsDecimal', Decimal('0'))
            entry_price = open_position.get('entryPriceDecimal')
            pos_timestamp_ms = open_position.get('timestamp_ms')

            lg.info(f"Managing existing {pos_side.upper()} position. Size: {pos_size}, Entry: {entry_price}")

            # --- Check for Exit Signal ---
            exit_signal_triggered = (pos_side == 'long' and signal == "SELL") or \
                                    (pos_side == 'short' and signal == "BUY")
            if exit_signal_triggered:
                lg.warning(f"{NEON_YELLOW}*** EXIT Signal ({signal}) opposes existing {pos_side} position. Closing position... ***{RESET}")
                try:
                    close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                    size_to_close = abs(pos_size)
                    if size_to_close <= 0: raise ValueError(f"Position size zero/negative ({size_to_close}).")

                    lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                    close_order = place_trade(exchange, symbol, close_side_signal, size_to_close, market_info, lg, 'market', reduce_only=True)
                    if close_order: lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully. ID: {close_order.get('id', 'N/A')}{RESET}")
                    else: lg.error(f"{NEON_RED}Failed to place CLOSE order. Manual check required!{RESET}")
                    return # Exit management after close attempt
                except Exception as close_err:
                    lg.error(f"{NEON_RED}Error attempting to close position {symbol}: {close_err}{RESET}", exc_info=True)
                    lg.warning(f"{NEON_YELLOW}Manual intervention likely needed!{RESET}")
                return # Exit management after close attempt

            # --- Check for Time-Based Exit ---
            time_exit_minutes_config = config.get("time_based_exit_minutes")
            if isinstance(time_exit_minutes_config, (int, float)) and time_exit_minutes_config > 0:
                 if pos_timestamp_ms:
                      try:
                           current_time_ms = time.time() * 1000
                           time_elapsed_ms = current_time_ms - pos_timestamp_ms
                           time_elapsed_minutes = time_elapsed_ms / (1000 * 60)
                           lg.debug(f"Time Exit Check: Elapsed={time_elapsed_minutes:.2f}m, Limit={time_exit_minutes_config}m")
                           if time_elapsed_minutes >= time_exit_minutes_config:
                                lg.warning(f"{NEON_YELLOW}*** TIME-BASED EXIT Triggered ({time_elapsed_minutes:.1f} >= {time_exit_minutes_config} min). Closing position... ***{RESET}")
                                # Execute Close Logic (same as above)
                                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                                size_to_close = abs(pos_size)
                                if size_to_close > 0:
                                     close_order = place_trade(exchange, symbol, close_side_signal, size_to_close, market_info, lg, 'market', reduce_only=True)
                                     if close_order: lg.info(f"{NEON_GREEN}Time-based CLOSE order placed successfully. ID: {close_order.get('id', 'N/A')}{RESET}")
                                     else: lg.error(f"{NEON_RED}Failed time-based CLOSE order. Manual check!{RESET}")
                                else: lg.warning("Time exit triggered but size is zero.")
                                return # Exit management after close attempt
                      except Exception as time_err: lg.error(f"Error in time exit check: {time_err}")
                 else: lg.warning("Time exit enabled, but position timestamp missing.")

            # --- If Holding, Manage Position (BE / TSL Update?) ---
            if not exit_signal_triggered:
                 lg.info(f"Signal ({signal}) allows holding {pos_side} position. Performing management checks...")

                 is_tsl_active_exchange = open_position.get('trailingStopLossValueDecimal') is not None

                 # --- Check Break-Even ---
                 if config.get("enable_break_even", False) and not is_tsl_active_exchange:
                     lg.debug(f"Checking Break-Even conditions for {symbol}...")
                     try:
                         if entry_price is None or entry_price <= 0: raise ValueError("Invalid entry price for BE")
                         if not isinstance(current_atr, Decimal) or current_atr <= 0: raise ValueError("Invalid ATR for BE")

                         be_trigger_atr_mult = Decimal(str(config.get("break_even_trigger_atr_multiple", "1.0")))
                         be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                         profit_target_atr = be_trigger_atr_mult

                         price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                         profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('0')

                         lg.debug(f"BE Check: PriceDiff={price_diff:.{price_prec}f}, ProfitATRs={profit_in_atr:.2f}, TargetATRs={profit_target_atr}")

                         if profit_in_atr >= profit_target_atr:
                              tick_offset = min_tick_size * be_offset_ticks
                              be_stop_price: Optional[Decimal] = None
                              if pos_side == 'long':
                                   be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP)
                              else: # short
                                   be_stop_price = (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                              if be_stop_price is None or be_stop_price <= 0: raise ValueError("Invalid BE stop price calc.")

                              current_sl_price = open_position.get('stopLossPriceDecimal')
                              update_be_sl = False
                              if current_sl_price is None: update_be_sl = True; lg.info("BE triggered: No current SL found.")
                              elif pos_side == 'long' and be_stop_price > current_sl_price: update_be_sl = True; lg.info(f"BE triggered: Target {be_stop_price} > Current {current_sl_price}.")
                              elif pos_side == 'short' and be_stop_price < current_sl_price: update_be_sl = True; lg.info(f"BE triggered: Target {be_stop_price} < Current {current_sl_price}.")
                              else: lg.debug(f"BE Triggered, but current SL ({current_sl_price}) already >= target BE SL ({be_stop_price}).")

                              if update_be_sl:
                                   lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                                   current_tp_price = open_position.get('takeProfitPriceDecimal')
                                   success = _set_position_protection(exchange, symbol, market_info, open_position, lg, be_stop_price, current_tp_price)
                                   if success: lg.info(f"{NEON_GREEN}Break-Even SL set/updated successfully.{RESET}")
                                   else: lg.error(f"{NEON_RED}Failed to set/update Break-Even SL.{RESET}")
                         else: lg.debug(f"BE Profit target not reached.")
                     except ValueError as ve: lg.warning(f"BE Check skipped: {ve}")
                     except Exception as be_err: lg.error(f"Error during BE check: {be_err}", exc_info=True)
                 elif is_tsl_active_exchange: lg.debug("BE check skipped: Exchange TSL is active.")
                 else: lg.debug("BE check skipped: Disabled in config.")

                 # --- Placeholder for other management ---
                 # E.g., Re-evaluate TSL settings if config changed? Partial TP?

    except ValueError as data_err:
        lg.error(f"{NEON_RED}Data Error for {symbol}: {data_err}. Skipping cycle.{RESET}")
    except ccxt.AuthenticationError as auth_err:
         lg.critical(f"{NEON_RED}CRITICAL: Authentication Failed during cycle: {auth_err}. Stopping bot.{RESET}")
         # Consider raising a specific exception to stop the main loop gracefully
         raise SystemExit("Authentication Failed")
    except Exception as cycle_err:
        lg.error(f"{NEON_RED}Unexpected error during analysis/trading cycle for {symbol}: {cycle_err}{RESET}", exc_info=True)
        # Decide if this error should stop the bot or just skip the cycle

    finally:
        # --- Cycle End Logging ---
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY # Allow modification of globals

    # Setup initial logger
    init_logger = setup_logger("ScalpXRX_Init")
    init_logger.info(f"--- Starting ScalpXRX Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    CONFIG = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
    TARGET_SYMBOL = CONFIG.get("symbol")
    if not TARGET_SYMBOL:
         init_logger.critical("CRITICAL: 'symbol' not defined in config.json. Exiting.")
         return

    # Setup logger specific to the target symbol for the main loop
    symbol_logger_name = f"ScalpXRX_{TARGET_SYMBOL.replace('/', '_').replace(':', '-')}"
    main_logger = setup_logger(symbol_logger_name)
    main_logger.info(f"Logging initialized for symbol: {TARGET_SYMBOL}")
    main_logger.info(f"Config loaded. Quote: {QUOTE_CURRENCY}, Interval: {CONFIG['interval']}")
