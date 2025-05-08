# volbot.py
# Incorporates Volumatic Trend + Order Block strategy into a trading framework using ccxt.
# Enhanced version with improved structure, error handling, logging, clarity, and robustness.

import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo  # Use zoneinfo (Python 3.9+) for timezone handling

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize colorama and set Decimal precision
init(autoreset=True)
getcontext().prec = 28  # Increased precision for complex financial calculations
load_dotenv()

# --- Constants ---
# Color Scheme
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# Strategy-Specific Colors & Log Levels
COLOR_UP = Fore.CYAN + Style.BRIGHT
COLOR_DN = Fore.YELLOW + Style.BRIGHT
COLOR_BULL_BOX = Fore.GREEN
COLOR_BEAR_BOX = Fore.RED
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX
COLOR_INFO = Fore.MAGENTA
COLOR_HEADER = Fore.BLUE + Style.BRIGHT
COLOR_WARNING = NEON_YELLOW
COLOR_ERROR = NEON_RED
COLOR_SUCCESS = NEON_GREEN

# API Credentials (Loaded from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print here as logger might not be set up yet
    raise ValueError("API Key/Secret not found in environment variables.")

# File/Directory Configuration
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Time & Retry Configuration
# Timezone is loaded within load_config to allow override
DEFAULT_TIMEZONE_STR = "America/Chicago"
try:
    TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_STR)  # Default if config fails early
except Exception:
    exit(1)  # Cannot proceed without a valid timezone

MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Base delay between retries
LOOP_DELAY_SECONDS = 15  # Min time between the end of one cycle and the start of the next

# Interval Configuration
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# API Error Codes for Retry Logic (HTTP status codes)
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]  # Add relevant exchange-specific codes if needed

# --- Default Volbot Strategy Parameters (overridden by config.json) ---
DEFAULT_VOLBOT_LENGTH = 40
DEFAULT_VOLBOT_ATR_LENGTH = 200
DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK = 1010
DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE = 100  # Use max volume in lookback
DEFAULT_VOLBOT_OB_SOURCE = "Wicks"  # "Wicks" or "Bodys"
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L = 25
DEFAULT_VOLBOT_MAX_BOXES = 50

# Default Risk Management Parameters (overridden by config.json)
DEFAULT_ATR_PERIOD = 14  # Risk Management ATR (for SL/TP/BE)

# Global QUOTE_CURRENCY placeholder, dynamically loaded from config
QUOTE_CURRENCY = "USDT"  # Default fallback

# Default console log level (updated by config)
console_log_level = logging.INFO


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that redacts sensitive information like API keys/secrets from log messages."""
    REDACTION_STR = "***REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive info."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, self.REDACTION_STR)
        if API_SECRET:
            msg = msg.replace(API_SECRET, self.REDACTION_STR)
        # Add other sensitive patterns if needed
        # msg = re.sub(r'password=.*', 'password=***REDACTED***', msg)
        return msg


def setup_logger(name_suffix: str) -> logging.Logger:
    """Sets up a logger instance with specified suffix, file rotation, and console output.
    Prevents adding duplicate handlers. Updates console handler level based on global setting.

    Args:
        name_suffix: A string suffix for the logger name and filename (e.g., symbol or 'init').

    Returns:
        The configured logging.Logger instance.
    """
    global console_log_level  # Ensure we use the potentially updated level
    safe_suffix = re.sub(r'[^\w\-]+', '_', name_suffix)  # Make suffix filesystem-safe
    logger_name = f"volbot_{safe_suffix}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Check if handlers already exist
    if logger.hasHandlers():
        # Update existing console handler level if necessary
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.level != console_log_level:
                    logger.debug(f"Updating console handler level for {logger_name} to {logging.getLevelName(console_log_level)}")
                    handler.setLevel(console_log_level)
        return logger  # Logger already configured

    logger.setLevel(logging.DEBUG)  # Set root level to DEBUG to allow handlers to filter

    # File Handler (DEBUG level)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        # Use print as logger might not be fully functional yet
        pass

    # Console Handler (configurable level)
    stream_handler = logging.StreamHandler()
    # Define colors for different log levels
    level_colors = {
        logging.DEBUG: NEON_BLUE,
        logging.INFO: NEON_GREEN,
        logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    class ColorFormatter(SensitiveFormatter):
        """Custom formatter to add colors and specific formatting to console logs."""
        def format(self, record) -> str:
            log_color = level_colors.get(record.levelno, RESET)
            record.levelname = f"{log_color}{record.levelname:<8}{RESET}"  # Pad level name
            record.asctime = f"{NEON_BLUE}{self.formatTime(record, self.datefmt)}{RESET}"
            # Extract the base logger name (e.g., 'volbot_BTCUSDT' -> 'BTCUSDT')
            base_name = record.name.split('_', 1)[-1] if '_' in record.name else record.name
            record.name_part = f"{NEON_PURPLE}[{base_name}]{RESET}"
            # Format the final message using the parent's method after modifications
            formatted_message = super().format(record)
            # Ensure final message has color reset
            return f"{formatted_message}{RESET}"

    stream_formatter = ColorFormatter(
        # Format: Timestamp - LEVEL    - [Name] - Message
        "%(asctime)s - %(levelname)s - %(name_part)s - %(message)s",
        datefmt='%H:%M:%S'  # Use shorter time format for console
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level)  # Use the global level
    logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent messages from reaching the root logger
    return logger


# --- Configuration Loading ---
def load_config(filepath: str) -> dict[str, Any]:
    """Loads configuration from a JSON file. Creates a default config if the file
    doesn't exist. Ensures all default keys are present, adding missing ones
    with default values and updating the file if necessary. Updates global settings.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        A dictionary containing the configuration settings.
    """
    global QUOTE_CURRENCY, TIMEZONE, console_log_level  # Allow updating global variables

    default_config = {
        # --- General Bot Settings ---
        "timezone": DEFAULT_TIMEZONE_STR,  # Timezone for logging and potentially scheduling (e.g., "Europe/London")
        "interval": "5",              # Default trading interval (string format from VALID_INTERVALS)
        "retry_delay": RETRY_DELAY_SECONDS,  # API retry delay in seconds
        "enable_trading": False,      # MASTER SWITCH: Set to true to allow placing real orders. Default: False.
        "use_sandbox": True,          # Use exchange's testnet/sandbox environment. Default: True.
        "risk_per_trade": 0.01,       # Max percentage of account balance to risk per trade (0.01 = 1%)
        "leverage": 10,               # Desired leverage for contract trading (applied if possible)
        "max_concurrent_positions": 1,  # Max open positions allowed per symbol by this bot instance (currently informational, logic not fully implemented)
        "quote_currency": "USDT",     # Currency for balance checks and position sizing (MUST match exchange pairs, e.g., USDT for BTC/USDT)
        "console_log_level": "INFO",  # Console logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        # --- Volbot Strategy Settings ---
        "volbot_enabled": True,         # Enable/disable Volbot strategy calculations and signals
        "volbot_length": DEFAULT_VOLBOT_LENGTH,  # Main period for Volbot EMAs
        "volbot_atr_length": DEFAULT_VOLBOT_ATR_LENGTH,  # ATR period for Volbot dynamic levels
        "volbot_volume_percentile_lookback": DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK,  # Lookback for volume normalization
        "volbot_volume_normalization_percentile": DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE,  # Percentile (usually 100=max)
        "volbot_ob_source": DEFAULT_VOLBOT_OB_SOURCE,  # "Wicks" or "Bodys" for Order Block detection
        "volbot_pivot_left_len_h": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H,  # Left bars for Pivot High
        "volbot_pivot_right_len_h": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H,  # Right bars for Pivot High
        "volbot_pivot_left_len_l": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L,  # Left bars for Pivot Low
        "volbot_pivot_right_len_l": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L,  # Right bars for Pivot Low
        "volbot_max_boxes": DEFAULT_VOLBOT_MAX_BOXES,  # Max number of active Order Blocks to track
        "volbot_signal_on_trend_flip": True,  # Generate BUY/SELL on Volbot trend direction change
        "volbot_signal_on_ob_entry": True,   # Generate BUY/SELL on price entering an Order Block matching trend

        # --- Risk Management Settings ---
        "atr_period": DEFAULT_ATR_PERIOD,  # ATR period for SL/TP/BE calculations (Risk Management ATR)
        "stop_loss_multiple": 1.8,  # Risk ATR multiple for initial Stop Loss distance
        "take_profit_multiple": 0.7,  # Risk ATR multiple for initial Take Profit distance

        # --- Trailing Stop Loss Config (Exchange-based TSL) ---
        "enable_trailing_stop": True,  # Attempt to set an exchange-based Trailing Stop Loss on entry
        "trailing_stop_callback_rate": "0.005",  # Trail distance as percentage (0.005=0.5%) or price distance (e.g., "50" for $50). String for Bybit API. Check exchange API docs.
        "trailing_stop_activation_percentage": 0.003,  # Profit percentage to activate TSL (0.003=0.3%). 0 for immediate activation if supported.

        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even + offset
        "break_even_trigger_atr_multiple": 1.0,  # Profit needed (in multiples of Risk ATR) to trigger BE
        "break_even_offset_ticks": 2,           # Number of minimum price ticks to offset BE SL from entry (for fees/slippage)
    }

    config_updated = False
    loaded_config = {}

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            loaded_config = default_config
            config_updated = True  # File was created
        except OSError:
            loaded_config = default_config  # Use defaults if creation fails
    else:
        try:
            with open(filepath, encoding="utf-8") as f:
                loaded_config = json.load(f)
            # Ensure all default keys exist, add missing ones
            loaded_config, config_updated = _ensure_config_keys(loaded_config, default_config)
            if config_updated:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(loaded_config, f_write, indent=4, sort_keys=True)
                except OSError:
                    pass
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            # Attempt to recreate default config if loading failed badly
            loaded_config = default_config  # Start with defaults
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4, sort_keys=True)
                config_updated = True
            except OSError:
                pass

    # Update global settings based on loaded/default config
    QUOTE_CURRENCY = loaded_config.get("quote_currency", default_config["quote_currency"])

    # Update global console log level
    level_name = loaded_config.get("console_log_level", "INFO").upper()
    new_log_level = getattr(logging, level_name, logging.INFO)
    if new_log_level != console_log_level:
        console_log_level = new_log_level
        # Note: Existing loggers' console handlers will be updated by setup_logger() if called again.

    # Update global timezone
    config_tz_str = loaded_config.get("timezone", DEFAULT_TIMEZONE_STR)
    try:
        new_tz = ZoneInfo(config_tz_str)
        if new_tz.key != TIMEZONE.key:
            TIMEZONE = new_tz
    except Exception:
        pass

    # Validate interval
    if loaded_config.get("interval") not in VALID_INTERVALS:
         loaded_config["interval"] = "5"  # Correct in loaded config

    # Validate OB source
    ob_source = loaded_config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    if ob_source not in ["Wicks", "Bodys"]:
         loaded_config["volbot_ob_source"] = DEFAULT_VOLBOT_OB_SOURCE

    return loaded_config


def _ensure_config_keys(loaded_config: dict[str, Any], default_config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Recursively ensures default keys exist in loaded config. Returns updated config and a flag indicating if changes were made."""
    updated = False
    for key, default_value in default_config.items():
        if key not in loaded_config:
            loaded_config[key] = default_value
            updated = True
        elif isinstance(default_value, dict) and isinstance(loaded_config.get(key), dict):
            # Recurse for nested dictionaries
            loaded_config[key], nested_updated = _ensure_config_keys(loaded_config[key], default_value)
            if nested_updated:
                updated = True
        # Optional: Add type checking/validation here if desired
        # elif type(loaded_config.get(key)) != type(default_value):
        #     print(f"{COLOR_WARNING}Config: Type mismatch for key '{key}'. Expected {type(default_value)}, got {type(loaded_config.get(key))}. Using loaded value.{RESET}")
    return loaded_config, updated


# Load configuration globally AFTER functions are defined
CONFIG = load_config(CONFIG_FILE)


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with configuration settings.

    Args:
        logger: The logger instance for initialization steps.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    try:
        exchange_id = 'bybit'  # Hardcoded to Bybit for this script
        exchange_class = getattr(ccxt, exchange_id)

        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable built-in rate limiter
            'options': {
                'defaultType': 'linear',  # Prefer linear contracts (USDT margined)
                'adjustForTimeDifference': True,  # Auto-sync time with server
                # Increased timeouts (milliseconds) for potentially slow networks/API
                'recvWindow': 10000,  # Bybit recommended recv_window
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 15000,
                'fetchOHLCVTimeout': 20000,
                'fetchPositionTimeout': 15000,  # Added timeout for positions
                'fetchPositionsTimeout': 20000,  # Added timeout for positions
            }
        }

        exchange = exchange_class(exchange_options)

        # Set Sandbox Mode based on config
        use_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(use_sandbox)
        sandbox_status = f"{COLOR_WARNING}SANDBOX MODE{RESET}" if use_sandbox else f"{COLOR_ERROR}LIVE TRADING MODE{RESET}"
        lg.warning(f"Exchange {exchange.id} initialized. Status: {sandbox_status}")

        # Load Markets - Crucial for accessing symbol info, precision, limits
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            exchange.load_markets()
            lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols).")
        except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
            lg.error(f"{COLOR_ERROR}Failed to load markets: {e}. Check connection and API status. Cannot proceed.{RESET}")
            return None  # Critical failure if markets can't be loaded

        # Test API Connection with Balance Fetch (using the robust fetch_balance)
        lg.info(f"Attempting initial balance fetch ({QUOTE_CURRENCY}) to test API keys and connection...")
        test_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if test_balance is not None:
             lg.info(f"{COLOR_SUCCESS}API keys and connection successful. Initial {QUOTE_CURRENCY} balance: {test_balance:.4f}{RESET}")
        else:
            # fetch_balance logs detailed errors, add a summary here
            lg.error(f"{COLOR_ERROR}Initial balance fetch failed. This is critical.{RESET}")
            lg.error(f"{COLOR_ERROR}Possible issues:{RESET}")
            lg.error(f"{COLOR_ERROR}- Invalid API Key/Secret.{RESET}")
            lg.error(f"{COLOR_ERROR}- Incorrect API permissions (Read required, Trade needed for execution).{RESET}")
            lg.error(f"{COLOR_ERROR}- IP Whitelist mismatch on Bybit account settings.{RESET}")
            lg.error(f"{COLOR_ERROR}- Using Live keys on Testnet or vice-versa.{RESET}")
            lg.error(f"{COLOR_ERROR}- Network/Firewall issues blocking connection to Bybit API.{RESET}")
            lg.error(f"{COLOR_ERROR}- Exchange API endpoint issues or maintenance.{RESET}")
            return None  # Critical failure if cannot authenticate/connect

        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{COLOR_ERROR}CCXT Authentication Error during initialization: {e}. Check API Key/Secret.{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{COLOR_ERROR}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{COLOR_ERROR}CCXT Network Error initializing: {e}. Check connection/firewall.{RESET}")
    except Exception as e:
        lg.critical(f"{COLOR_ERROR}Unexpected error initializing CCXT exchange: {e}{RESET}", exc_info=True)
    return None


# --- CCXT Data Fetching ---
def safe_decimal(value: Any, default: Decimal | None = None) -> Decimal | None:
    """Safely convert a value to Decimal, handling None, strings, floats, and potential InvalidOperation.
    Returns default if conversion fails or input is invalid.
    """
    if value is None:
        return default
    try:
        # Handle potential scientific notation in strings robustly
        str_value = str(value).strip()
        d = Decimal(str_value)
        # Check for NaN or Infinity which are invalid for most financial ops
        if not d.is_finite():
            # Log this occurrence if necessary
            # print(f"Warning: safe_decimal encountered non-finite value: {value}")
            return default
        return d
    except (InvalidOperation, ValueError, TypeError):
        # Log this occurrence if necessary
        # print(f"Warning: safe_decimal failed for value: {value} (type: {type(value)})")
        return default


def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the current market price for a symbol using the exchange's ticker,
    with robust fallbacks (last, mid, ask, bid) and validation.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        The current price as a Decimal, or None if fetching fails or price is invalid/zero.
    """
    lg = logger
    price: Decimal | None = None
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        # Use appropriate params for Bybit V5 ticker fetching
        params = {}
        if 'bybit' in exchange.id.lower():
            market = exchange.market(symbol)  # Assume market loaded
            if market:
                category = 'linear' if market.get('linear', True) else 'inverse'
                params['category'] = category
            else:
                lg.warning(f"Market info not found for {symbol} when fetching ticker. Assuming 'linear'.")
                params['category'] = 'linear'

        ticker = exchange.fetch_ticker(symbol, params=params)
        lg.debug(f"Ticker data received for {symbol}: Keys={list(ticker.keys())}")

        # Order of preference for price:
        # 1. 'last' price
        last_price = safe_decimal(ticker.get('last'))
        if last_price is not None and last_price > 0:
            price = last_price
            lg.debug(f"Using 'last' price: {price}")
        else:
            # 2. Bid/Ask Midpoint
            bid = safe_decimal(ticker.get('bid'))
            ask = safe_decimal(ticker.get('ask'))
            if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
                price = (bid + ask) / Decimal('2')
                lg.debug(f"Using bid/ask midpoint: {price} (Bid: {bid}, Ask: {ask})")
            else:
                # 3. 'ask' price (price you can buy at)
                if ask is not None and ask > 0:
                    price = ask
                    lg.debug(f"Using 'ask' price (fallback): {price}")
                # 4. 'bid' price (price you can sell at - last resort)
                elif bid is not None and bid > 0:
                    price = bid
                    lg.debug(f"Using 'bid' price (last resort): {price}")

        # Final validation
        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{COLOR_ERROR}Failed to fetch a valid positive current price for {symbol} from ticker.{RESET}")
            lg.debug(f"Invalid or zero price values in ticker data: {ticker}")
            return None

    except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        lg.error(f"{COLOR_ERROR}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeNotAvailable as e:
        lg.error(f"{COLOR_ERROR}Exchange not available fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Check for specific "symbol not found" or invalid instrument errors
        if "symbol not found" in err_str or "instrument not found" in err_str or "invalid symbol" in err_str:
             lg.error(f"{COLOR_ERROR}Symbol {symbol} not found on exchange ticker endpoint: {e}{RESET}")
        else:
             lg.error(f"{COLOR_ERROR}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT with retries, validation, DataFrame conversion,
    and robust data cleaning.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        timeframe: CCXT timeframe string (e.g., '5m', '1h').
        limit: Maximum number of klines to fetch.
        logger: Logger instance.

    Returns:
        A pandas DataFrame with OHLCV data indexed by timestamp, or an empty DataFrame on failure.
    """
    lg = logger or logging.getLogger(__name__)  # Use provided logger or get default
    empty_df = pd.DataFrame()
    try:
        if not exchange.has['fetchOHLCV']:
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return empty_df

        ohlcv: list[list[Any]] | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
                # Add category param for Bybit V5 klines
                params = {}
                if 'bybit' in exchange.id.lower():
                    market = exchange.market(symbol)  # Assume market loaded
                    if market:
                        category = 'linear' if market.get('linear', True) else 'inverse'
                        params['category'] = category
                    else:
                         lg.warning(f"Market info not found for {symbol} when fetching klines. Assuming 'linear'.")
                         params['category'] = 'linear'

                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)

                if ohlcv and len(ohlcv) > 0:
                    lg.debug(f"Received {len(ohlcv)} klines from API.")
                    break  # Success
                else:
                    lg.warning(f"fetch_ohlcv returned empty list for {symbol} (Attempt {attempt + 1}). Retrying...")
                    if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"Max retries exceeded for network error fetching klines: {e}")
                    raise e  # Raise after max retries
            except ccxt.RateLimitExceeded as e:
                # Extract wait time suggestion from error message if possible
                wait_time_match = re.search(r'try again in (\d+)', str(e), re.IGNORECASE)
                wait_time = int(wait_time_match.group(1)) if wait_time_match else RETRY_DELAY_SECONDS * (attempt + 2)  # Exponential backoff
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                if attempt < MAX_API_RETRIES: time.sleep(wait_time)
                else: raise e
            except ccxt.ExchangeError as e:
                 err_str = str(e).lower()
                 # Check for "symbol not found" or invalid instrument errors specifically
                 if "symbol not found" in err_str or "instrument invalid" in err_str or "invalid symbol" in err_str:
                      lg.error(f"{COLOR_ERROR}Symbol {symbol} not found on exchange kline endpoint: {e}{RESET}")
                      return empty_df  # Cannot recover from invalid symbol
                 lg.error(f"Exchange error during fetch_ohlcv for {symbol}: {e}")
                 # Retry generic exchange errors cautiously
                 if attempt < MAX_API_RETRIES // 2:  # Retry fewer times for generic errors
                     time.sleep(RETRY_DELAY_SECONDS)
                 else: raise e

        if not ohlcv:
            lg.warning(f"{COLOR_WARNING}No kline data returned for {symbol} {timeframe} after all retries.{RESET}")
            return empty_df

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"DataFrame conversion resulted in empty DF for {symbol}.")
            return empty_df

        # --- Data Cleaning and Processing ---
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # Drop rows where timestamp conversion failed
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision, then to numeric for calculations
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # Use safe_decimal first for robust conversion
            df[col] = df[col].apply(lambda x: safe_decimal(x, default=np.nan))
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        # Drop rows with any NaN in OHLC (essential price data)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Ensure close price is positive (filter out bad data points)
        df = df[df['close'] > 0]
        # Fill NaN volume with 0 (often indicates no trades in that period)
        df['volume'].fillna(0, inplace=True)

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price or non-positive close for {symbol}.")

        if df.empty:
            lg.warning(f"{COLOR_WARNING}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return empty_df

        # Ensure data is sorted chronologically by timestamp index
        df.sort_index(inplace=True)

        # Optional: Check for duplicate timestamps (can happen with API issues)
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].unique()
            lg.warning(f"{COLOR_WARNING}Found {len(duplicates)} duplicate timestamps in kline data for {symbol}. Keeping last entry for each.{RESET}")
            df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.BadSymbol as e:
         lg.error(f"{COLOR_ERROR}Invalid symbol {symbol} for klines: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error fetching klines for {symbol} after retries: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{COLOR_ERROR}Exchange error processing klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return empty_df

# --- Volbot Strategy Calculation Functions ---


def ema_swma(series: pd.Series, length: int, logger: logging.Logger) -> pd.Series:
    """Calculates a Smoothed Weighted Moving Average (SWMA), an EMA applied
    to a weighted average of the last 4 values (weights: 1/6, 2/6, 2/6, 1/6).
    Uses pandas_ta.ema with adjust=False for TV-like calculation. Handles NaNs.

    Args:
        series: Input pandas Series (e.g., 'close' prices).
        length: EMA length parameter.
        logger: Logger instance.

    Returns:
        A pandas Series with SWMA values, aligned with the input series index.
    """
    lg = logger
    lg.debug(f"Calculating SWMA with length: {length}...")
    required_periods = 4  # Need 4 periods for the weighting
    if len(series) < required_periods:
        lg.warning(f"Series length ({len(series)}) < {required_periods}. SWMA requires {required_periods} periods. Returning standard EMA.")
        # Fallback to standard EMA, use adjust=False for consistency
        # Ensure result is a Series even if input is short
        ema_result = ta.ema(series, length=length, adjust=False)
        return ema_result if isinstance(ema_result, pd.Series) else pd.Series(ema_result, index=series.index)

    # Calculate the weighted average: (1/6)*P[t] + (2/6)*P[t-1] + (2/6)*P[t-2] + (1/6)*P[t-3]
    # Ensure correct alignment using shift, fill NaNs created by shift with 0 temporarily for calculation
    # Note: This assumes prices are positive; adjust if handling negative values is needed.
    weighted_series = (series.fillna(0) / 6 +
                       series.shift(1).fillna(0) * 2 / 6 +
                       series.shift(2).fillna(0) * 2 / 6 +
                       series.shift(3).fillna(0) * 6)

    # Set initial values (where shifts caused NaNs) back to NaN before EMA
    weighted_series.iloc[:required_periods - 1] = np.nan

    # Calculate EMA on the weighted series
    # Use adjust=False for behavior closer to TradingView's EMA calculation
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length, adjust=False)  # Apply EMA only on valid weighted values

    # Reindex to match the original series index, forward-filling initial NaNs if needed
    # This ensures the output series has the same length and index as the input
    result_series = smoothed_ema.reindex(series.index)
    # Forward fill might be too aggressive, consider leaving initial NaNs as NaN
    # result_series = result_series.ffill() # Optional: ffill to handle NaNs at the very beginning

    lg.debug(f"SWMA calculation finished. Result length: {len(result_series)}")
    return result_series


def calculate_volatility_levels(df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Calculates core Volumatic Trend indicators: EMAs, ATR, dynamic levels,
    normalized volume, and cumulative volume metrics. Includes data length checks.

    Args:
        df: DataFrame with OHLCV data.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        DataFrame with added Volbot strategy columns. Returns input df if insufficient data.
    """
    lg = logger
    lg.info("Calculating Volumatic Trend Levels...")
    length = config.get("volbot_length", DEFAULT_VOLBOT_LENGTH)
    atr_length = config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH)
    volume_lookback = config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK)
    # Volume normalization percentile is not directly used in this calculation part, but good to note it exists
    # vol_norm_perc = config.get("volbot_volume_normalization_percentile", DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE)

    # Check for sufficient data length for calculations
    # Need enough for EMA/SWMA (length + ~3), ATR (atr_length), and volume lookback
    min_len = max(length + 3, atr_length, volume_lookback) + 10  # Add buffer
    if len(df) < min_len:
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)} rows) for Volumatic Trend calculation (min ~{min_len}). Skipping.{RESET}")
        # Add placeholder columns to prevent errors later
        placeholder_cols = [
            'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
            'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
            'step_up_strat', 'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat',
            'vol_dn_step_strat', 'vol_trend_up_level_strat', 'vol_trend_dn_level_strat',
            'volume_delta_strat', 'volume_total_strat', 'cum_vol_delta_since_change_strat',
            'cum_vol_total_since_change_strat', 'last_trend_change_idx'
        ]
        for col in placeholder_cols:
            df[col] = np.nan
        return df

    df_calc = df.copy()  # Work on a copy

    try:
        # Calculate Strategy EMAs and ATR
        df_calc['ema1_strat'] = ema_swma(df_calc['close'], length, lg)
        df_calc['ema2_strat'] = ta.ema(df_calc['close'], length=length, adjust=False)
        df_calc['atr_strat'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=atr_length)

        # Determine Trend Direction (UP if smoothed EMA > standard EMA, handle NaNs)
        df_calc['trend_up_strat'] = (df_calc['ema1_strat'] > df_calc['ema2_strat']).astype('boolean')  # Use nullable boolean
        # Identify exact points where trend changes (True on the first bar of the new trend)
        df_calc['trend_changed_strat'] = df_calc['trend_up_strat'].diff().fillna(False)  # True where trend flips (NaN diff becomes False)

        # Initialize level columns
        level_cols = ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
                      'step_up_strat', 'step_dn_strat']
        for col in level_cols:
            df_calc[col] = np.nan  # Initialize as float NaNs

        # --- Calculate Dynamic Levels Based on Trend Changes ---
        # Levels are reset/recalculated ONLY on the bar where the trend flip occurs.
        # They remain constant until the next flip. Use vectorization for efficiency.

        # Find indices where trend changed
        change_indices = df_calc.index[df_calc['trend_changed_strat']]

        # Get EMA1 and ATR values at the bar *before* the change occurred
        # Shift data to align previous bar's values with the change index
        ema1_at_change = df_calc['ema1_strat'].shift(1).loc[change_indices]
        atr_at_change = df_calc['atr_strat'].shift(1).loc[change_indices]

        # Calculate levels where valid EMA and ATR exist
        valid_change = pd.notna(ema1_at_change) & pd.notna(atr_at_change) & (atr_at_change > 0)
        valid_indices = change_indices[valid_change]

        if not valid_indices.empty:
            valid_ema1 = ema1_at_change[valid_change]
            valid_atr = atr_at_change[valid_change]

            # Calculate levels based on EMA1 +/- 3*ATR
            upper = valid_ema1 + valid_atr * 3
            lower = valid_ema1 - valid_atr * 3
            # Intermediate volatility levels (adjust width by 4*ATR)
            lower_vol = lower + valid_atr * 4  # Top of lower vol zone
            upper_vol = upper - valid_atr * 4  # Bottom of upper vol zone

            # Step sizes (per 1% of normalized volume)
            step_up = (lower_vol - lower).clip(lower=0) / 100  # Ensure non-negative
            step_dn = (upper - upper_vol).clip(lower=0) / 100  # Ensure non-negative

            # Assign calculated levels to the DataFrame at the valid change indices
            df_calc.loc[valid_indices, 'upper_strat'] = upper
            df_calc.loc[valid_indices, 'lower_strat'] = lower
            df_calc.loc[valid_indices, 'lower_vol_strat'] = lower_vol
            df_calc.loc[valid_indices, 'upper_vol_strat'] = upper_vol
            df_calc.loc[valid_indices, 'step_up_strat'] = step_up
            df_calc.loc[valid_indices, 'step_dn_strat'] = step_dn

        # Forward fill the calculated levels until the next change
        # This propagates the levels set at each trend change point
        for col in level_cols:
             df_calc[col] = df_calc[col].ffill()

        # --- Calculate Volume Metrics ---
        # Normalized Volume (0-100 based on max volume in rolling lookback window)
        max_vol_lookback = df_calc['volume'].rolling(window=volume_lookback, min_periods=max(1, volume_lookback // 10)).max()  # Require some min periods
        # Avoid division by zero if max_vol is 0 or NaN
        df_calc['vol_norm_strat'] = np.where(
            pd.notna(max_vol_lookback) & (max_vol_lookback > 0),
            (df_calc['volume'].fillna(0) / max_vol_lookback * 100),
            0  # Set to 0 if max_vol is invalid or zero
        ).clip(0, 100)  # Clip result between 0 and 100

        # Volume-adjusted step amount (handle potential NaNs in steps or norm_vol)
        df_calc['vol_up_step_strat'] = (df_calc['step_up_strat'].fillna(0) * df_calc['vol_norm_strat'].fillna(0))
        df_calc['vol_dn_step_strat'] = (df_calc['step_dn_strat'].fillna(0) * df_calc['vol_norm_strat'].fillna(0))

        # Final Volume-Adjusted Trend Levels
        df_calc['vol_trend_up_level_strat'] = df_calc['lower_strat'] + df_calc['vol_up_step_strat']
        df_calc['vol_trend_dn_level_strat'] = df_calc['upper_strat'] - df_calc['vol_dn_step_strat']

        # --- Cumulative Volume Since Last Trend Change ---
        # Calculate volume delta (+ for green candle, - for red, 0 for doji)
        df_calc['volume_delta_strat'] = np.where(df_calc['close'] > df_calc['open'], df_calc['volume'],
                                           np.where(df_calc['close'] < df_calc['open'], -df_calc['volume'], 0)).fillna(0)
        df_calc['volume_total_strat'] = df_calc['volume'].fillna(0)

        # Create a grouping key based on when the trend changes
        trend_block_group = df_calc['trend_changed_strat'].cumsum()
        # Calculate cumulative sums within each trend block
        df_calc['cum_vol_delta_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_delta_strat'].cumsum()
        df_calc['cum_vol_total_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_total_strat'].cumsum()

        # Track index (timestamp) of last trend change for reference
        # Get the timestamp where trend changed, forward fill it
        last_change_ts = df_calc.index.to_series().where(df_calc['trend_changed_strat']).ffill()
        df_calc['last_trend_change_idx'] = last_change_ts  # Store the timestamp

        lg.info("Volumatic Trend Levels calculation complete.")
        return df_calc

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Volumatic Trend calculation: {e}{RESET}", exc_info=True)
        # Return original DataFrame on error, potentially adding NaN columns if needed elsewhere
        return df


def calculate_pivot_order_blocks(df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Identifies Pivot High (PH) and Pivot Low (PL) points based on configuration,
    used for Order Block detection. Includes data length checks.

    Args:
        df: DataFrame with OHLCV data.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        DataFrame with added 'ph_strat' (Pivot High price) and 'pl_strat' (Pivot Low price) columns.
        Returns input df with NaN columns if insufficient data.
    """
    lg = logger
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    left_h = config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    left_l = config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    lg.info(f"Calculating Pivot Points (Source: {source}, Left/Right H: {left_h}/{right_h}, L: {left_l}/{right_l})...")

    # Check for sufficient data length for pivot calculation window
    min_len_h = left_h + right_h + 1
    min_len_l = left_l + right_l + 1
    if len(df) < max(min_len_h, min_len_l):
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)} rows) for Pivot calculation (min H ~{min_len_h}, L ~{min_len_l}). Skipping.{RESET}")
        df['ph_strat'] = np.nan
        df['pl_strat'] = np.nan
        return df

    df_calc = df.copy()
    try:
        # Select price series based on source config ('high'/'low' for Wicks, 'close'/'open' for Bodys)
        # Ensure columns exist before accessing
        high_source_col = 'high' if source == "Wicks" else 'close'
        low_source_col = 'low' if source == "Wicks" else 'open'

        if high_source_col not in df_calc.columns or low_source_col not in df_calc.columns:
             lg.error(f"Missing required columns '{high_source_col}' or '{low_source_col}' for pivot calculation. Skipping.")
             df_calc['ph_strat'] = np.nan
             df_calc['pl_strat'] = np.nan
             return df_calc

        high_col = df_calc[high_source_col]
        low_col = df_calc[low_source_col]

        df_calc['ph_strat'] = np.nan
        df_calc['pl_strat'] = np.nan

        # --- Calculate Pivot Highs (PH) ---
        # A bar `i` is PH if high_col[i] is strictly greater than highs in `left_h` bars before
        # AND strictly greater than highs in `right_h` bars after.
        # Vectorized approach is complex for this pattern, loop is clearer and acceptable for typical kline lengths
        for i in range(left_h, len(df_calc) - right_h):
            pivot_val = high_col.iloc[i]
            if pd.isna(pivot_val): continue  # Skip if pivot candidate value is NaN

            # Check left: pivot_val > all left values (strict)
            is_higher_than_left = (pivot_val > high_col.iloc[i - left_h : i]).all()
            if not is_higher_than_left: continue

            # Check right: pivot_val > all right values (strict)
            is_higher_than_right = (pivot_val > high_col.iloc[i + 1 : i + right_h + 1]).all()

            if is_higher_than_right:
                # Store the high price at the pivot bar index
                df_calc.loc[df_calc.index[i], 'ph_strat'] = pivot_val

        # --- Calculate Pivot Lows (PL) ---
        # A bar `i` is PL if low_col[i] is strictly lower than lows in `left_l` bars before
        # AND strictly lower than lows in `right_l` bars after.
        for i in range(left_l, len(df_calc) - right_l):
            pivot_val = low_col.iloc[i]
            if pd.isna(pivot_val): continue  # Skip if pivot candidate value is NaN

            # Check left: pivot_val < all left values (strict)
            is_lower_than_left = (pivot_val < low_col.iloc[i - left_l : i]).all()
            if not is_lower_than_left: continue

            # Check right: pivot_val < all right values (strict)
            is_lower_than_right = (pivot_val < low_col.iloc[i + 1 : i + right_l + 1]).all()

            if is_lower_than_right:
                 # Store the low price at the pivot bar index
                 df_calc.loc[df_calc.index[i], 'pl_strat'] = pivot_val

        lg.info(f"Pivot Point calculation complete. Found {pd.notna(df_calc['ph_strat']).sum()} PH, {pd.notna(df_calc['pl_strat']).sum()} PL.")
        return df_calc

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Pivot calculation: {e}{RESET}", exc_info=True)
        # Return original DataFrame with NaN columns on error
        df['ph_strat'] = np.nan
        df['pl_strat'] = np.nan
        return df


def manage_order_blocks(df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Identifies, creates, and manages the state (active/closed) of Order Blocks (OBs)
    based on pivot points and subsequent price action. Limits number of active OBs.

    Args:
        df: DataFrame with OHLCV and pivot columns ('ph_strat', 'pl_strat').
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Tuple: (DataFrame with active OB references, list of all tracked bull OBs, list of all tracked bear OBs).
               Returns input df and empty lists if insufficient data or error.
    """
    lg = logger
    lg.info("Managing Order Block Boxes...")
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    # Use pivot right lengths to determine the OB candle relative to the pivot
    ob_candle_offset_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    ob_candle_offset_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    max_boxes = config.get("volbot_max_boxes", DEFAULT_VOLBOT_MAX_BOXES)

    df_calc = df.copy()
    bull_boxes: list[dict] = []  # Stores all created bull boxes (active or closed)
    bear_boxes: list[dict] = []  # Stores all created bear boxes
    active_bull_boxes: list[dict] = []  # Stores currently active bull boxes
    active_bear_boxes: list[dict] = []  # Stores currently active bear boxes
    box_counter = 0

    # Check if pivot columns exist
    if 'ph_strat' not in df_calc.columns or 'pl_strat' not in df_calc.columns:
        lg.warning(f"{COLOR_WARNING}Pivot columns ('ph_strat', 'pl_strat') not found. Skipping OB management.{RESET}")
        # Add placeholder columns to prevent errors later
        df_calc['active_bull_ob_strat'] = None
        df_calc['active_bear_ob_strat'] = None
        return df_calc, bull_boxes, bear_boxes

    # Initialize columns to store references to the active OB dict for each bar
    df_calc['active_bull_ob_strat'] = pd.Series(dtype='object')
    df_calc['active_bear_ob_strat'] = pd.Series(dtype='object')

    try:
        # Iterate through each bar to potentially create new OBs and manage existing ones
        for i in range(len(df_calc)):
            current_idx = df_calc.index[i]
            current_close = df_calc.loc[current_idx, 'close']
            df_calc.loc[current_idx, 'high']
            df_calc.loc[current_idx, 'low']

            # --- Manage Existing Active Boxes ---
            # Check for mitigation BEFORE adding new boxes for the current bar
            next_active_bull = []
            active_bull_ref_for_current_bar = None
            for box in active_bull_boxes:
                # Bull OB Mitigation: Close below the OB bottom
                if current_close < box['bottom']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx  # Record mitigation timestamp
                    lg.debug(f"Closed Bullish OB: {box['id']} at {current_idx} (Close {current_close:.5f} < Bottom {box['bottom']:.5f})")
                    # Don't add to next_active_bull
                else:
                    # Box remains active
                    next_active_bull.append(box)
                    # Check if current price (e.g., close) is inside this active box
                    if box['bottom'] <= current_close <= box['top']:
                         # If multiple active boxes contain the price, prioritize the most recent one?
                         # Current logic takes the *last* one checked in the loop. Sorting active_boxes might be needed if specific priority is required.
                         active_bull_ref_for_current_bar = box

            active_bull_boxes = next_active_bull  # Update the list of active bull boxes

            next_active_bear = []
            active_bear_ref_for_current_bar = None
            for box in active_bear_boxes:
                # Bear OB Mitigation: Close above the OB top
                if current_close > box['top']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx
                    lg.debug(f"Closed Bearish OB: {box['id']} at {current_idx} (Close {current_close:.5f} > Top {box['top']:.5f})")
                    # Don't add to next_active_bear
                else:
                    # Box remains active
                    next_active_bear.append(box)
                    # Check if current price is inside this active box
                    if box['bottom'] <= current_close <= box['top']:
                        active_bear_ref_for_current_bar = box

            active_bear_boxes = next_active_bear  # Update the list of active bear boxes

            # Store the active OB reference (or None) in the DataFrame for this bar *before* creating new ones
            df_calc.at[current_idx, 'active_bull_ob_strat'] = active_bull_ref_for_current_bar
            df_calc.at[current_idx, 'active_bear_ob_strat'] = active_bear_ref_for_current_bar

            # --- Create New Bearish OB (Based on Pivot Highs) ---
            # PH confirmed at index 'i'. OB candle is 'ob_candle_offset_h' bars *before* 'i'.
            if pd.notna(df_calc.loc[current_idx, 'ph_strat']):
                ob_candle_iloc = i - ob_candle_offset_h
                if ob_candle_iloc >= 0:
                    ob_candle_idx = df_calc.index[ob_candle_iloc]
                    # Define Bearish OB range based on source
                    top_p, bottom_p = np.nan, np.nan
                    if source == "Bodys":  # Bearish Body OB: Open to Close
                        top_p = df_calc.loc[ob_candle_idx, 'open']
                        bottom_p = df_calc.loc[ob_candle_idx, 'close']
                    else:  # Wicks (Default): High to Close
                        top_p = df_calc.loc[ob_candle_idx, 'high']
                        bottom_p = df_calc.loc[ob_candle_idx, 'close']

                    # Validate prices and create box
                    if pd.notna(top_p) and pd.notna(bottom_p):
                        top_price = max(top_p, bottom_p)
                        bottom_price = min(top_p, bottom_p)
                        # Avoid creating zero-height boxes
                        if top_price > bottom_price:
                            box_counter += 1
                            new_box = {
                                'id': f'BearOB_{box_counter}', 'type': 'bear',
                                'start_idx': ob_candle_idx,  # Timestamp of the OB candle
                                'pivot_idx': current_idx,   # Timestamp where pivot was confirmed
                                'end_idx': None,           # Timestamp when mitigated (null when active)
                                'top': top_price, 'bottom': bottom_price, 'state': 'active'
                            }
                            bear_boxes.append(new_box)
                            active_bear_boxes.append(new_box)  # Add to active list
                            lg.debug(f"Created Bearish OB: {new_box['id']} (Pivot: {current_idx}, Candle: {ob_candle_idx}, Range: [{bottom_price:.5f}, {top_price:.5f}])")

            # --- Create New Bullish OB (Based on Pivot Lows) ---
            # PL confirmed at index 'i'. OB candle is 'ob_candle_offset_l' bars *before* 'i'.
            if pd.notna(df_calc.loc[current_idx, 'pl_strat']):
                ob_candle_iloc = i - ob_candle_offset_l
                if ob_candle_iloc >= 0:
                    ob_candle_idx = df_calc.index[ob_candle_iloc]
                    # Define Bullish OB range based on source
                    top_p, bottom_p = np.nan, np.nan
                    if source == "Bodys":  # Bullish Body OB: Close to Open
                        top_p = df_calc.loc[ob_candle_idx, 'close']
                        bottom_p = df_calc.loc[ob_candle_idx, 'open']
                    else:  # Wicks (Default): Open to Low
                        top_p = df_calc.loc[ob_candle_idx, 'open']
                        bottom_p = df_calc.loc[ob_candle_idx, 'low']

                    # Validate prices and create box
                    if pd.notna(top_p) and pd.notna(bottom_p):
                        top_price = max(top_p, bottom_p)
                        bottom_price = min(top_p, bottom_p)
                        if top_price > bottom_price:
                            box_counter += 1
                            new_box = {
                                'id': f'BullOB_{box_counter}', 'type': 'bull',
                                'start_idx': ob_candle_idx, 'pivot_idx': current_idx,
                                'end_idx': None,
                                'top': top_price, 'bottom': bottom_price, 'state': 'active'
                            }
                            bull_boxes.append(new_box)
                            active_bull_boxes.append(new_box)  # Add to active list
                            lg.debug(f"Created Bullish OB: {new_box['id']} (Pivot: {current_idx}, Candle: {ob_candle_idx}, Range: [{bottom_price:.5f}, {top_price:.5f}])")

            # --- Limit Number of Active Boxes ---
            # Keep only the 'max_boxes' most recent *active* boxes based on pivot confirmation time
            if len(active_bull_boxes) > max_boxes:
                active_bull_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True)  # Sort newest first
                removed_boxes = active_bull_boxes[max_boxes:]
                active_bull_boxes = active_bull_boxes[:max_boxes]
                for box in removed_boxes:
                    box['state'] = 'trimmed'  # Mark as trimmed, not closed by price
                    lg.debug(f"Trimmed older active Bull OB: {box['id']} (Pivot: {box['pivot_idx']})")

            if len(active_bear_boxes) > max_boxes:
                active_bear_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True)
                removed_boxes = active_bear_boxes[max_boxes:]
                active_bear_boxes = active_bear_boxes[:max_boxes]
                for box in removed_boxes:
                    box['state'] = 'trimmed'
                    lg.debug(f"Trimmed older active Bear OB: {box['id']} (Pivot: {box['pivot_idx']})")

        # End of loop

        # The `bull_boxes` and `bear_boxes` lists contain *all* boxes ever created in this run.
        # The `active_bull_boxes` and `active_bear_boxes` lists contain only those currently active.
        # The DataFrame columns `active_bull_ob_strat` / `active_bear_ob_strat` contain the reference
        # to the specific active box the price was inside at that bar's close.

        num_active_bull = len(active_bull_boxes)
        num_active_bear = len(active_bear_boxes)
        lg.info(f"Order Block management complete. Total created: {len(bull_boxes)} Bull, {len(bear_boxes)} Bear. Currently active: {num_active_bull} Bull, {num_active_bear} Bear.")
        # Return the DataFrame and the lists containing *all* boxes (active, closed, trimmed)
        return df_calc, bull_boxes, bear_boxes

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Order Block management: {e}{RESET}", exc_info=True)
        # Return original df state with NaN columns if error occurs
        df['active_bull_ob_strat'] = None
        df['active_bear_ob_strat'] = None
        return df, [], []


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes trading data using Volbot strategy, generates signals, and calculates risk metrics.
    Handles market precision and provides utility methods.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],  # Pass the fetched market info
    ) -> None:
        """Initializes the analyzer with data, config, logger, and market info.
        Calculates indicators upon initialization.

        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame.
            logger (logging.Logger): Logger instance.
            config (Dict[str, Any]): Configuration dictionary.
            market_info (Dict[str, Any]): Market information dictionary from ccxt.
        """
        self.df_raw = df
        self.df_processed = pd.DataFrame()  # Populated by _calculate_indicators
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN")  # User-friendly interval
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN")  # CCXT format

        # Determine precision and tick size from market_info
        self.min_tick_size = self._determine_min_tick_size()
        self.price_precision = self._determine_price_precision()  # Decimals for price
        self.amount_precision = self._determine_amount_precision()  # Decimals for amount/size

        self.logger.debug(f"Analyzer initialized for {self.symbol}: "
                          f"TickSize={self.min_tick_size}, PricePrec={self.price_precision}, AmountPrec={self.amount_precision}")

        # Strategy state variables
        self.strategy_state: dict[str, Any] = {}  # Stores latest indicator values
        self.latest_active_bull_ob: dict | None = None  # Ref to latest active bull OB dict
        self.latest_active_bear_ob: dict | None = None  # Ref to latest active bear OB dict
        self.all_bull_boxes: list[dict] = []  # All bull OBs generated
        self.all_bear_boxes: list[dict] = []  # All bear OBs generated

        # Calculate indicators immediately on initialization
        self._calculate_indicators()
        # Update state with the latest calculated values
        self._update_latest_strategy_state()

    def _determine_min_tick_size(self) -> Decimal:
        """Determine minimum price increment (tick size) from market info."""
        try:
            # Prefer precision info 'price' which often *is* the tick size
            price_prec_val = self.market_info.get('precision', {}).get('price')
            if price_prec_val is not None:
                 tick = safe_decimal(price_prec_val)
                 if tick and tick > 0:
                     self.logger.debug(f"Tick size determined from precision.price: {tick}")
                     return tick

            # Fallback to limits info 'price' 'min' (less common for tick size)
            min_price_limit = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_limit is not None:
                tick = safe_decimal(min_price_limit)
                # Check if it looks like a tick size (e.g., 0.01, 0.5) rather than a minimum trading price (e.g., 1000)
                if tick and tick > 0 and tick < 10:  # Heuristic: ticks are usually small
                    self.logger.debug(f"Tick size determined from limits.price.min: {tick}")
                    return tick

        except Exception as e:
            self.logger.warning(f"Could not reliably determine tick size for {self.symbol} from market info: {e}. Using fallback.")

        # Absolute fallback (adjust based on typical market, e.g., 0.1 for BTC, 0.01 for ETH)
        # Determine fallback based on typical price magnitude if possible
        last_price = safe_decimal(self.df_raw['close'].iloc[-1]) if not self.df_raw.empty else None
        if last_price:
            if last_price > 1000: default_tick = Decimal('0.1')
            elif last_price > 10: default_tick = Decimal('0.01')
            elif last_price > 0.1: default_tick = Decimal('0.001')
            else: default_tick = Decimal('0.00001')
        else:
            default_tick = Decimal('0.0001')  # Generic fallback

        self.logger.warning(f"Using default/fallback tick size {default_tick} for {self.symbol}.")
        return default_tick

    def _determine_price_precision(self) -> int:
        """Determine decimal places for price formatting based on tick size."""
        try:
            tick_size = self.min_tick_size  # Use the already determined tick size
            if tick_size > 0:
                # Calculate decimal places from the tick size
                # normalize() removes trailing zeros, as_tuple().exponent gives power of 10
                return abs(tick_size.normalize().as_tuple().exponent)
        except Exception as e:
            self.logger.warning(f"Could not determine price precision from tick size ({self.min_tick_size}) for {self.symbol}: {e}. Using default.")
        # Default fallback precision
        return 4

    def _determine_amount_precision(self) -> int:
         """Determine decimal places for amount/size formatting from market info."""
         try:
            # Prefer precision.amount if it's an integer (decimal places)
            amount_precision_val = self.market_info.get('precision', {}).get('amount')
            if isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                self.logger.debug(f"Amount precision determined from precision.amount (integer): {amount_precision_val}")
                return amount_precision_val

            # If precision.amount is float/str, assume it's the step size
            if isinstance(amount_precision_val, (float, str)):
                step_size = safe_decimal(amount_precision_val)
                if step_size and step_size > 0:
                    precision = abs(step_size.normalize().as_tuple().exponent)
                    self.logger.debug(f"Amount precision determined from precision.amount (step size {step_size}): {precision}")
                    return precision

            # Fallback: Check limits.amount.min if it looks like a step size
            min_amount_limit = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount_limit is not None:
                 step_size = safe_decimal(min_amount_limit)
                 # Heuristic: Step size is usually small, often power of 10
                 if step_size and step_size > 0 and step_size <= 1:
                      precision = abs(step_size.normalize().as_tuple().exponent)
                      self.logger.debug(f"Amount precision determined from limits.amount.min (step size {step_size}): {precision}")
                      return precision

         except Exception as e:
             self.logger.warning(f"Could not determine amount precision for {self.symbol}: {e}. Using default.")
         # Default fallback precision for amount
         default_prec = 8  # Common default for crypto base amounts
         self.logger.warning(f"Using default amount precision {default_prec} for {self.symbol}.")
         return default_prec

    def _calculate_indicators(self) -> None:
        """Calculates Risk Management ATR and Volbot strategy indicators. Populates df_processed."""
        if self.df_raw.empty:
            self.logger.warning(f"{COLOR_WARNING}Raw DataFrame empty, cannot calculate indicators for {self.symbol}.{RESET}")
            self.df_processed = pd.DataFrame()  # Ensure it's empty
            return

        # Check minimum data length required by the longest lookback period
        # Add a buffer for calculation stability (e.g., initial NaNs)
        buffer = 50
        try:
             min_len_volbot = max(
                 self.config.get("volbot_length", DEFAULT_VOLBOT_LENGTH) + 3,  # SWMA needs ~3 extra
                 self.config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
                 self.config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
                 self.config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H) + self.config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H) + 1,
                 self.config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L) + self.config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L) + 1
             ) if self.config.get("volbot_enabled", True) else 0

             min_len_risk = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
             min_required_data = max(min_len_volbot, min_len_risk) + buffer
        except Exception as e:
             self.logger.error(f"Error calculating minimum required data length: {e}. Using fallback.")
             min_required_data = 250  # Fallback minimum

        if len(self.df_raw) < min_required_data:
            self.logger.warning(
                f"{COLOR_WARNING}Insufficient data ({len(self.df_raw)} points) for {self.symbol}. "
                f"Need ~{min_required_data} for reliable calculations. Results may be inaccurate or missing.{RESET}"
            )
            # Proceed, but calculations might return NaNs or be unreliable

        try:
            df_calc = self.df_raw.copy()

            # 1. Calculate Risk Management ATR (for SL/TP/BE)
            atr_period_risk = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc['atr_risk'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=atr_period_risk)
            self.logger.debug(f"Calculated Risk Management ATR (Length: {atr_period_risk})")

            # 2. Calculate Volbot Strategy Indicators (if enabled)
            if self.config.get("volbot_enabled", True):
                df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
                df_calc = calculate_pivot_order_blocks(df_calc, self.config, self.logger)
                # Pass the modified df_calc to manage_order_blocks
                df_calc, self.all_bull_boxes, self.all_bear_boxes = manage_order_blocks(df_calc, self.config, self.logger)
            else:
                 self.logger.info("Volbot strategy calculation skipped (disabled in config).")
                 # Add NaN placeholders if strategy disabled but columns might be expected later
                 placeholder_cols = [
                    'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
                    'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat',
                    'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat', 'vol_dn_step_strat',
                    'vol_trend_up_level_strat', 'vol_trend_dn_level_strat', 'volume_delta_strat',
                    'volume_total_strat', 'cum_vol_delta_since_change_strat', 'cum_vol_total_since_change_strat',
                    'last_trend_change_idx', 'ph_strat', 'pl_strat', 'active_bull_ob_strat', 'active_bear_ob_strat'
                 ]
                 for col in placeholder_cols:
                     if col not in df_calc.columns: df_calc[col] = np.nan

            # Convert calculated columns to Decimal where appropriate for consistency
            decimal_cols = ['atr_risk', 'ema1_strat', 'ema2_strat', 'atr_strat', 'upper_strat',
                            'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat',
                            'step_dn_strat', 'vol_up_step_strat', 'vol_dn_step_strat',
                            'vol_trend_up_level_strat', 'vol_trend_dn_level_strat', 'ph_strat', 'pl_strat']
            for col in decimal_cols:
                if col in df_calc.columns:
                     # Apply safe_decimal, keeping NaN where conversion fails
                     df_calc[col] = df_calc[col].apply(lambda x: safe_decimal(x, default=np.nan))

            self.df_processed = df_calc
            self.logger.debug(f"Indicator calculations complete for {self.symbol}. Processed DF has {len(self.df_processed)} rows.")

        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Error calculating indicators for {self.symbol}: {e}{RESET}", exc_info=True)
            self.df_processed = pd.DataFrame()  # Ensure empty on error

    def _update_latest_strategy_state(self) -> None:
        """Updates `strategy_state` dictionary with the latest available values
        from the last row of `df_processed`. Converts relevant values to Decimal.
        """
        self.strategy_state = {}  # Reset state each time
        self.latest_active_bull_ob = None
        self.latest_active_bear_ob = None

        if self.df_processed.empty or len(self.df_processed) == 0:
            self.logger.warning(f"Cannot update state: Processed DataFrame is empty for {self.symbol}.")
            return

        try:
            # Get the last row of the processed data
            latest = self.df_processed.iloc[-1]

            if latest.isnull().all():
                self.logger.warning(f"{COLOR_WARNING}Last row of processed DataFrame contains all NaNs for {self.symbol}. Check data source or indicator calculations.{RESET}")
                return

            # List of columns to potentially extract (core + risk + strategy)
            all_possible_cols = [
                'open', 'high', 'low', 'close', 'volume', 'atr_risk',
                # Volbot specific
                'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
                'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
                'step_up_strat', 'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat',
                'vol_dn_step_strat', 'vol_trend_up_level_strat', 'vol_trend_dn_level_strat',
                'cum_vol_delta_since_change_strat', 'cum_vol_total_since_change_strat',
                'last_trend_change_idx', 'ph_strat', 'pl_strat',
                'active_bull_ob_strat', 'active_bear_ob_strat'
            ]

            # Extract values, handling missing columns and converting to Decimal where needed
            for col in all_possible_cols:
                if col in latest.index:
                    value = latest[col]
                    # Handle specific types
                    if col in ['active_bull_ob_strat', 'active_bear_ob_strat']:
                        # These are expected to be dicts or None/NaN
                        self.strategy_state[col] = value if isinstance(value, dict) else None
                    elif col in ['trend_up_strat', 'trend_changed_strat']:
                        # Handle boolean/nullable boolean
                        self.strategy_state[col] = bool(value) if pd.notna(value) else None
                    elif col == 'last_trend_change_idx':
                        # Store timestamp as is, or None
                        self.strategy_state[col] = value if pd.notna(value) else None
                    else:
                        # Attempt Decimal conversion for numeric types
                        decimal_value = safe_decimal(value, default=None)
                        self.strategy_state[col] = decimal_value  # Store Decimal or None

            # Update latest active OB references separately
            self.latest_active_bull_ob = self.strategy_state.get('active_bull_ob_strat')
            self.latest_active_bear_ob = self.strategy_state.get('active_bear_ob_strat')
            # Add convenience boolean flags
            self.strategy_state['is_in_active_bull_ob'] = self.latest_active_bull_ob is not None
            self.strategy_state['is_in_active_bear_ob'] = self.latest_active_bear_ob is not None

            # Log the updated state compactly, formatting Decimals
            log_state = {}
            price_fmt = f".{self.price_precision}f"
            for k, v in self.strategy_state.items():
                 if isinstance(v, Decimal):
                     # Format price-like values, show more precision for others if needed
                     is_price = any(p in k for p in ['price', 'level', 'strat', 'open', 'high', 'low', 'close', 'tp', 'sl'])
                     log_state[k] = f"{v:{price_fmt}}" if is_price else f"{v:.8f}"  # Use 8 decimals for non-price Decimals
                 elif isinstance(v, (bool, pd._libs.missing.NAType)):  # Handle boolean and Pandas NA
                     log_state[k] = str(v) if pd.notna(v) else 'None'
                 elif isinstance(v, pd.Timestamp):
                     log_state[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                 elif v is not None and k not in ['active_bull_ob_strat', 'active_bear_ob_strat']:  # Avoid logging full OB dicts here
                     log_state[k] = v

            self.logger.debug(f"Latest strategy state updated for {self.symbol}: {log_state}")
            if self.latest_active_bull_ob: self.logger.debug(f"  Latest Active Bull OB: ID={self.latest_active_bull_ob.get('id')}, Range=[{self.latest_active_bull_ob.get('bottom'):.5f}, {self.latest_active_bull_ob.get('top'):.5f}]")
            if self.latest_active_bear_ob: self.logger.debug(f"  Latest Active Bear OB: ID={self.latest_active_bear_ob.get('id')}, Range=[{self.latest_active_bear_ob.get('bottom'):.5f}, {self.latest_active_bear_ob.get('top'):.5f}]")

        except IndexError:
            self.logger.error(f"{COLOR_ERROR}Error accessing latest row (index -1) for {self.symbol}. Processed DataFrame might be empty or too short.{RESET}")
        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error updating latest strategy state for {self.symbol}: {e}{RESET}", exc_info=True)

    # --- Utility Functions ---
    def get_price_precision(self) -> int:
        """Returns the number of decimal places for price."""
        return self.price_precision

    def get_amount_precision(self) -> int:
        """Returns the number of decimal places for amount/size."""
        return self.amount_precision

    def get_min_tick_size(self) -> Decimal:
        """Returns the minimum price increment as a Decimal."""
        return self.min_tick_size

    def round_price(self, price: Decimal | float | str) -> Decimal | None:
        """Rounds a given price to the symbol's minimum tick size."""
        price_decimal = safe_decimal(price)
        min_tick = self.min_tick_size
        if price_decimal is None or min_tick is None or min_tick <= 0:
            self.logger.error(f"Cannot round price: Invalid input price ({price}) or min_tick ({min_tick})")
            return None
        try:
            # Quantize to the tick size using ROUND_HALF_UP (common rounding)
            # Or adjust rounding method if needed (e.g., ROUND_DOWN for bids, ROUND_UP for asks)
            rounded_price = (price_decimal / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
            return rounded_price
        except Exception as e:
             self.logger.error(f"Error rounding price {price_decimal} with tick {min_tick}: {e}")
             return None

    def round_amount(self, amount: Decimal | float | str) -> Decimal | None:
        """Rounds (truncates) a given amount to the symbol's amount precision (step size)."""
        amount_decimal = safe_decimal(amount)
        amount_prec_digits = self.amount_precision
        if amount_decimal is None:
            self.logger.error(f"Cannot round amount: Invalid input amount ({amount})")
            return None

        try:
            # Determine the step size based on precision digits
            # Handles both integer precision (decimal places) and step size directly if precision was derived from it
            amount_prec_val = self.market_info.get('precision', {}).get('amount')
            step_size = None
            if isinstance(amount_prec_val, (float, str)):
                 step_size = safe_decimal(amount_prec_val)
            elif isinstance(amount_prec_val, int):
                 step_size = Decimal('1e-' + str(amount_prec_digits))

            if step_size and step_size > 0:
                 # Truncate/floor to the nearest step size
                 rounded_amount = (amount_decimal // step_size) * step_size
                 return rounded_amount
            else:  # Fallback using decimal places if step size invalid
                 rounding_factor = Decimal('1e-' + str(amount_prec_digits))
                 rounded_amount = amount_decimal.quantize(rounding_factor, rounding=ROUND_DOWN)  # Truncate
                 return rounded_amount

        except Exception as e:
             self.logger.error(f"Error rounding amount {amount_decimal} with precision {amount_prec_digits}: {e}")
             return None

    # --- Signal Generation ---
    def generate_trading_signal(self) -> str:
        """Generates "BUY", "SELL", or "HOLD" signal based on Volbot rules defined in config.
        Relies on values in `self.strategy_state`.

        Returns:
            str: "BUY", "SELL", or "HOLD".
        """
        signal = "HOLD"  # Default signal
        if not self.strategy_state:
            self.logger.debug("Cannot generate signal: Strategy state is empty.")
            return signal
        if not self.config.get("volbot_enabled", True):
            self.logger.debug("Cannot generate signal: Volbot strategy is disabled in config.")
            return signal

        try:
            # Get relevant state values (handle potential None)
            is_trend_up = self.strategy_state.get('trend_up_strat')  # Boolean or None
            trend_changed = self.strategy_state.get('trend_changed_strat', False)  # Default to False if missing
            is_in_bull_ob = self.strategy_state.get('is_in_active_bull_ob', False)
            is_in_bear_ob = self.strategy_state.get('is_in_active_bear_ob', False)

            # Get signal generation rules from config
            signal_on_flip = self.config.get("volbot_signal_on_trend_flip", True)
            signal_on_ob = self.config.get("volbot_signal_on_ob_entry", True)

            # Check if trend is determined
            if is_trend_up is None:
                self.logger.debug("Volbot signal: HOLD (Trend state could not be determined - likely insufficient data)")
                return "HOLD"

            trend_str = f"{COLOR_UP}UP{RESET}" if is_trend_up else f"{COLOR_DN}DOWN{RESET}"
            ob_status = ""
            if is_in_bull_ob: ob_status += f"{COLOR_BULL_BOX} InBullOB{RESET}"
            if is_in_bear_ob: ob_status += f"{COLOR_BEAR_BOX} InBearOB{RESET}"

            # Rule 1: Trend Flip Signal (Highest Priority if enabled)
            if signal_on_flip and trend_changed:
                signal = "BUY" if is_trend_up else "SELL"
                reason = f"Trend flipped to {trend_str}"
                color = COLOR_UP if is_trend_up else COLOR_DN
                self.logger.info(f"{color}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                return signal

            # Rule 2: Order Block Entry Signal (If no flip and enabled)
            if signal_on_ob:
                if is_trend_up and is_in_bull_ob:
                    signal = "BUY"
                    ob_id = self.latest_active_bull_ob.get('id', 'N/A') if self.latest_active_bull_ob else 'N/A'
                    reason = f"Price in Bull OB '{ob_id}' during {trend_str} Trend"
                    self.logger.info(f"{COLOR_BULL_BOX}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                    return signal
                elif not is_trend_up and is_in_bear_ob:
                    signal = "SELL"
                    ob_id = self.latest_active_bear_ob.get('id', 'N/A') if self.latest_active_bear_ob else 'N/A'
                    reason = f"Price in Bear OB '{ob_id}' during {trend_str} Trend"
                    self.logger.info(f"{COLOR_BEAR_BOX}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                    return signal

            # Rule 3: Default to HOLD if no entry conditions met
            # Log current state for HOLD signal
            self.logger.info(f"Volbot Signal: HOLD (Conditions: Trend={trend_str},{ob_status})")

        except Exception as e:
             self.logger.error(f"{COLOR_ERROR}Error generating signal: {e}{RESET}", exc_info=True)
             return "HOLD"  # Default to HOLD on any error during signal generation

        return signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential Take Profit (TP) and initial Stop Loss (SL) based on entry price,
        signal, Risk Management ATR, and configuration multiples. Rounds results precisely
        to the market's minimum tick size and validates them.

        Args:
            entry_price: The potential or actual entry price (Decimal).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            Tuple (entry_price, take_profit, stop_loss):
                - entry_price (Decimal): The input entry price.
                - take_profit (Optional[Decimal]): Calculated TP price, rounded, or None if invalid.
                - stop_loss (Optional[Decimal]): Calculated SL price, rounded, or None if invalid.
        """
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"Cannot calculate TP/SL: Signal is '{signal}'.")
            return entry_price, None, None

        # --- Get Inputs & Validate ---
        atr_val = self.strategy_state.get("atr_risk")  # Use Risk Management ATR
        tp_multiple = safe_decimal(self.config.get("take_profit_multiple", 1.0))
        sl_multiple = safe_decimal(self.config.get("stop_loss_multiple", 1.5))
        min_tick = self.min_tick_size  # Already determined Decimal tick size

        # Validate inputs needed for calculation
        valid_inputs = True
        if not (isinstance(entry_price, Decimal) and entry_price > 0):
            self.logger.error(f"TP/SL Calc Error: Invalid entry_price ({entry_price})")
            valid_inputs = False
        if not (isinstance(atr_val, Decimal) and atr_val > 0):
            # Allow calculation even if ATR is zero/small, but log warning
            if isinstance(atr_val, Decimal) and atr_val <= 0:
                 self.logger.warning(f"{COLOR_WARNING}TP/SL Calc Warning: Risk ATR is zero or negative ({atr_val}). SL/TP offsets will be zero.{RESET}")
                 atr_val = Decimal('0')  # Proceed with zero offset
            else:
                 self.logger.error(f"TP/SL Calc Error: Invalid Risk ATR value ({atr_val})")
                 valid_inputs = False
        if not (isinstance(tp_multiple, Decimal) and tp_multiple >= 0):  # Allow 0 TP multiple
            self.logger.error(f"TP/SL Calc Error: Invalid take_profit_multiple ({tp_multiple})")
            valid_inputs = False
        if not (isinstance(sl_multiple, Decimal) and sl_multiple > 0):  # SL multiple must be positive
            self.logger.error(f"TP/SL Calc Error: Invalid stop_loss_multiple ({sl_multiple})")
            valid_inputs = False
        if not (isinstance(min_tick, Decimal) and min_tick > 0):
            self.logger.error(f"TP/SL Calc Error: Invalid min_tick_size ({min_tick})")
            valid_inputs = False

        if not valid_inputs:
            self.logger.warning(f"{COLOR_WARNING}Cannot calculate TP/SL for {self.symbol} due to invalid inputs.{RESET}")
            return entry_price, None, None

        try:
            # --- Calculate Offsets ---
            tp_offset = atr_val * tp_multiple
            sl_offset = atr_val * sl_multiple
            take_profit_raw, stop_loss_raw = None, None

            # --- Calculate Raw Prices ---
            if signal == "BUY":
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
            elif signal == "SELL":
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset

            # --- Round TP/SL to Tick Size ---
            take_profit, stop_loss = None, None

            # Round TP: Down for SELL, UP for BUY to be conservative (harder to reach)
            if take_profit_raw is not None:
                tp_rounding = ROUND_DOWN if signal == "SELL" else ROUND_UP
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=tp_rounding) * min_tick

            # Round SL: UP for SELL, DOWN for BUY to be conservative (easier to hit)
            if stop_loss_raw is not None:
                sl_rounding = ROUND_UP if signal == "SELL" else ROUND_DOWN
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=sl_rounding) * min_tick

            # --- Validate Rounded Prices ---
            price_fmt = f".{self.price_precision}f"  # Format for logging

            # Ensure SL is strictly on the losing side of entry
            if stop_loss is not None:
                if signal == "BUY" and stop_loss >= entry_price:
                    adjusted_sl = (entry_price - min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    self.logger.warning(f"{COLOR_WARNING}BUY SL ({stop_loss:{price_fmt}}) >= entry ({entry_price:{price_fmt}}). Adjusting SL down by one tick to {adjusted_sl:{price_fmt}}.{RESET}")
                    stop_loss = adjusted_sl
                elif signal == "SELL" and stop_loss <= entry_price:
                    adjusted_sl = (entry_price + min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                    self.logger.warning(f"{COLOR_WARNING}SELL SL ({stop_loss:{price_fmt}}) <= entry ({entry_price:{price_fmt}}). Adjusting SL up by one tick to {adjusted_sl:{price_fmt}}.{RESET}")
                    stop_loss = adjusted_sl

            # Ensure TP is strictly on the winning side of entry (if TP multiple > 0)
            if take_profit is not None and tp_multiple > 0:
                if signal == "BUY" and take_profit <= entry_price:
                    adjusted_tp = (entry_price + min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                    self.logger.warning(f"{COLOR_WARNING}BUY TP ({take_profit:{price_fmt}}) <= entry ({entry_price:{price_fmt}}). Adjusting TP up by one tick to {adjusted_tp:{price_fmt}}.{RESET}")
                    take_profit = adjusted_tp
                elif signal == "SELL" and take_profit >= entry_price:
                    adjusted_tp = (entry_price - min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    self.logger.warning(f"{COLOR_WARNING}SELL TP ({take_profit:{price_fmt}}) >= entry ({entry_price:{price_fmt}}). Adjusting TP down by one tick to {adjusted_tp:{price_fmt}}.{RESET}")
                    take_profit = adjusted_tp

            # Ensure final SL/TP are positive prices
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{COLOR_ERROR}SL calculation resulted in zero/negative price ({stop_loss:{price_fmt}}). Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                # If TP multiple was 0, a zero TP might be intentional (e.g., no TP). Otherwise, it's an error.
                if tp_multiple > 0:
                     self.logger.error(f"{COLOR_ERROR}TP calculation resulted in zero/negative price ({take_profit:{price_fmt}}). Cannot set TP.{RESET}")
                     take_profit = None
                else:
                     self.logger.info(f"TP calculation resulted in zero/negative price ({take_profit:{price_fmt}}) but TP multiple was zero. Setting TP to None.")
                     take_profit = None  # Treat as no TP

            # Log final results
            tp_str = f"{take_profit:{price_fmt}}" if take_profit else "None"
            sl_str = f"{stop_loss:{price_fmt}}" if stop_loss else "None"
            self.logger.info(f"Calculated TP/SL for {self.symbol} {signal} (Risk ATR={atr_val:{price_fmt}}): "
                             f"Entry={entry_price:{price_fmt}}, TP={tp_str}, SL={sl_str}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the *available* balance for a specific currency, handling various
    account types (Contract, Unified) and response structures robustly, especially for Bybit V5.

    Args:
        exchange: Initialized ccxt.Exchange object.
        currency: Currency code (e.g., "USDT").
        logger: Logger instance.

    Returns:
        Available balance as Decimal, or None on failure or if balance is zero/negative.
    """
    lg = logger
    balance_info: dict | None = None
    available_balance: Decimal | None = None

    # Prioritize specific account types relevant for derivatives (Bybit V5 uses 'CONTRACT' or 'UNIFIED')
    # Others might use 'future', 'swap', 'funding', 'trading' etc.
    account_types_to_try = []
    if 'bybit' in exchange.id.lower():
        account_types_to_try = ['CONTRACT', 'UNIFIED']  # Bybit V5: Unified covers Spot, Linear, Options. Contract for older Inverse.
    else:
        # Generic order for other exchanges (adjust as needed)
        account_types_to_try = ['swap', 'future', 'contract', 'trading', 'funding', 'spot']

    # Try fetching balance with specific account types first
    for acc_type in account_types_to_try:
        try:
            lg.debug(f"Fetching balance with params={{'type': '{acc_type}'}} for {currency}...")
            # Bybit V5 uses 'accountType' in params for fetchBalance
            params = {'accountType': acc_type} if 'bybit' in exchange.id.lower() else {'type': acc_type}
            balance_info = exchange.fetch_balance(params=params)
            # Store the attempted type for parsing logic
            balance_info['params_used'] = params

            # Attempt to parse the balance from this response
            parsed_balance = _parse_balance_from_response(balance_info, currency, lg)
            if parsed_balance is not None and parsed_balance > 0:
                available_balance = parsed_balance
                lg.debug(f"Found positive balance ({available_balance}) in account type '{acc_type}'.")
                break  # Found a usable balance, stop searching
            elif parsed_balance is not None:  # Found balance but it's zero
                 lg.debug(f"Balance found in '{acc_type}' is zero. Checking next type.")
                 balance_info = None  # Reset to try next type cleanly
            else:
                 lg.debug(f"Balance for {currency} not found in '{acc_type}' account type structure. Checking next type.")
                 balance_info = None  # Reset

        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            err_str = str(e).lower()
            # Ignore errors indicating the account type isn't supported/valid and try the next one
            if "account type not support" in err_str or "invalid account type" in err_str or "account type invalid" in err_str:
                lg.debug(f"Account type '{acc_type}' not supported/invalid for fetch_balance. Trying next.")
                continue
            else:
                # Log other errors but continue trying other types if possible
                lg.warning(f"Exchange/Network error fetching balance (type {acc_type}): {e}. Trying next.")
                if acc_type == account_types_to_try[-1]:  # If last attempt failed with error
                     lg.error("Failed to fetch balance for all attempted specific account types due to errors.")
                     # Proceed to default fetch attempt below
                continue
        except Exception as e:
            lg.warning(f"Unexpected error fetching balance (type {acc_type}): {e}. Trying next.")
            continue

    # If no positive balance found with specific types, try default fetch_balance (no params)
    if available_balance is None:
        lg.debug(f"No positive balance found with specific account types. Fetching balance using default parameters for {currency}...")
        try:
            balance_info = exchange.fetch_balance()
            balance_info['params_used'] = {'type': 'default'}  # Mark as default fetch
            parsed_balance = _parse_balance_from_response(balance_info, currency, lg)
            if parsed_balance is not None and parsed_balance > 0:
                 available_balance = parsed_balance
            elif parsed_balance is not None:
                 lg.info(f"Default balance fetch returned zero balance for {currency}.")
            else:
                 lg.warning(f"Default balance fetch did not find balance for {currency}.")

        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            lg.error(f"{COLOR_ERROR}Failed to fetch balance using default parameters: {e}{RESET}")
            # Cannot proceed if default fetch fails
            return None
        except Exception as e:
             lg.error(f"{COLOR_ERROR}Unexpected error during default balance fetch: {e}{RESET}", exc_info=True)
             return None

    # --- Final Result ---
    if available_balance is not None and available_balance > 0:
        lg.info(f"Final available {currency} balance: {available_balance:.4f}")
        return available_balance
    elif available_balance is not None and available_balance <= 0:
        lg.warning(f"{COLOR_WARNING}Available balance for {currency} is zero or negative ({available_balance:.4f}).{RESET}")
        return None  # Treat zero/negative available balance as unusable
    else:
        lg.error(f"{COLOR_ERROR}Could not determine available balance for {currency} after all attempts.{RESET}")
        # Log the last structure checked for debugging
        lg.debug(f"Last balance_info structure checked: {json.dumps(balance_info, indent=2) if balance_info else 'None'}")
        return None


def _parse_balance_from_response(balance_info: dict | None, currency: str, logger: logging.Logger) -> Decimal | None:
    """Helper function to parse the *available* balance from various potential structures
    within a ccxt fetch_balance response dictionary. Prioritizes 'free' or 'available' fields.

    Args:
        balance_info: The dictionary returned by exchange.fetch_balance().
        currency: The currency code (e.g., "USDT").
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if not found or parsing fails.
    """
    lg = logger
    if not balance_info:
        lg.debug("_parse_balance: Input balance_info is None.")
        return None

    attempted_params = balance_info.get('params_used', {})
    lg.debug(f"_parse_balance: Attempting to parse for {currency} with params {attempted_params}. Structure keys: {list(balance_info.keys())}")

    available_balance_str: str | None = None
    parse_source = "N/A"

    try:
        # --- Bybit V5 Specific Structure (Highest Priority if detected) ---
        # Bybit V5 often nests the useful data under info.result.list[]
        info_dict = balance_info.get('info', {})
        result_dict = info_dict.get('result', {})
        balance_list = result_dict.get('list')

        if isinstance(balance_list, list):
            lg.debug(f"_parse_balance: Found Bybit V5 'info.result.list' structure with {len(balance_list)} account(s).")
            target_account_type = attempted_params.get('accountType')  # e.g., CONTRACT, UNIFIED
            parsed_v5_acc_type = 'N/A'

            for account_data in balance_list:
                current_account_type = account_data.get('accountType')
                # Match if a specific type was requested OR if it was a default fetch (check all accounts)
                match_type = (target_account_type is None or current_account_type == target_account_type)

                if match_type and isinstance(account_data.get('coin'), list):
                    lg.debug(f"_parse_balance: Checking coins in V5 account type '{current_account_type}'...")
                    for coin_data in account_data['coin']:
                        if coin_data.get('coin') == currency:
                            # Priority keys for available balance in Bybit V5:
                            # 1. availableToWithdraw / availableBalance (preferred)
                            # 2. equity (use as fallback if others missing, might include PnL)
                            # 3. walletBalance (total balance, less preferred for available)
                            keys_to_check = ['availableToWithdraw', 'availableBalance', 'equity']
                            for key in keys_to_check:
                                free = coin_data.get(key)
                                if free is not None:
                                    available_balance_str = str(free)
                                    parsed_v5_acc_type = current_account_type or 'Unknown'
                                    parse_source = f"Bybit V5 list ['{key}'] (Account: {parsed_v5_acc_type})"
                                    lg.debug(f"_parse_balance: Found {currency} balance via {parse_source}: {available_balance_str}")
                                    break  # Found balance for this currency using preferred key
                            if available_balance_str is not None: break  # Exit coin loop
                    if available_balance_str is not None: break  # Exit account loop
            if available_balance_str is None:
                lg.debug(f"_parse_balance: {currency} not found within Bybit V5 'info.result.list[].coin[]' for requested type '{target_account_type}'.")

        # --- Standard ccxt Structure (if V5 parse failed or not V5 structure) ---
        if available_balance_str is None:
            # 1. Standard ccxt 'free' balance (top-level currency dict)
            if currency in balance_info and isinstance(balance_info[currency], dict) and 'free' in balance_info[currency]:
                free_val = balance_info[currency]['free']
                if free_val is not None:
                    available_balance_str = str(free_val)
                    parse_source = f"standard ['{currency}']['free']"
                    lg.debug(f"_parse_balance: Found balance via {parse_source}: {available_balance_str}")

            # 2. Alternative top-level 'free' dictionary structure (less common)
            elif 'free' in balance_info and isinstance(balance_info['free'], dict) and currency in balance_info['free']:
                 free_val = balance_info['free'][currency]
                 if free_val is not None:
                    available_balance_str = str(free_val)
                    parse_source = "top-level 'free' dict"
                    lg.debug(f"_parse_balance: Found balance via {parse_source}: {available_balance_str}")

        # --- Final Conversion ---
        if available_balance_str is not None:
            final_balance = safe_decimal(available_balance_str)
            if final_balance is not None:
                # Return the balance (can be zero, handled by caller)
                lg.info(f"Parsed available balance for {currency} via {parse_source}: {final_balance:.8f}")
                return final_balance
            else:
                lg.warning(f"{COLOR_WARNING}_parse_balance: Failed to convert parsed balance string '{available_balance_str}' from {parse_source} to Decimal.{RESET}")
                return None  # Conversion failed

        # --- Fallback (If 'free'/'available' not found) ---
        # Consider using 'total' or 'equity' as a last resort, but log clearly as it's not truly 'available'.
        # This implementation prioritizes strictly available funds. If total is needed, call fetch_total_balance.
        lg.debug(f"_parse_balance: Could not find 'free' or 'available' balance field for {currency} in the response.")
        return None  # Indicate available balance wasn't found

    except Exception as e:
        lg.error(f"{COLOR_ERROR}_parse_balance: Error parsing balance response: {e}{RESET}", exc_info=True)
        lg.debug(f"Balance info structure during parse error: {balance_info}")
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Retrieves, validates, and enhances market information for a symbol from ccxt.
    Ensures essential precision and limit info is present.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT' or 'BTC/USDT').
        logger: Logger instance.

    Returns:
        Enhanced market info dictionary, or None if not found or validation fails.
        Adds 'is_contract', 'is_linear', 'is_inverse'.
    """
    lg = logger
    try:
        # Ensure markets are loaded. Load again if symbol not found initially.
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market '{symbol}' not found or markets not loaded. Reloading markets...")
            try:
                exchange.load_markets(reload=True)
            except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as reload_err:
                 lg.error(f"Failed to reload markets: {reload_err}")
                 return None  # Cannot proceed if reload fails

            # Check again after reload
            if symbol not in exchange.markets:
                # Try simplifying symbol (e.g., BTC/USDT:USDT -> BTC/USDT) if applicable
                simplified_symbol = symbol.split(':')[0] if ':' in symbol else None
                if simplified_symbol and simplified_symbol != symbol and simplified_symbol in exchange.markets:
                    lg.warning(f"Original symbol '{symbol}' not found, but simplified '{simplified_symbol}' found. Using simplified.")
                    symbol = simplified_symbol
                else:
                    lg.error(f"{COLOR_ERROR}Market '{symbol}' not found even after reloading markets. "
                             f"Check symbol format and availability on {exchange.id}. Available keys sample: {list(exchange.markets.keys())[:10]}{RESET}")
                    return None

        market = exchange.market(symbol)
        if not market:
            # This case should be rare if the symbol key exists in exchange.markets
            lg.error(f"{COLOR_ERROR}ccxt returned None for market('{symbol}') despite symbol key existing in markets dict.{RESET}")
            return None

        # --- Enhance with derived info for easier access ---
        mkt_copy = market.copy()  # Work on a copy
        mkt_copy['is_contract'] = mkt_copy.get('contract', False) or mkt_copy.get('type') in ['swap', 'future', 'option']
        mkt_copy['is_linear'] = mkt_copy.get('linear', False)
        mkt_copy['is_inverse'] = mkt_copy.get('inverse', False)
        # Ensure basic type info is present
        mkt_type = mkt_copy.get('type', 'unknown')
        contract_type = "Linear" if mkt_copy['is_linear'] else "Inverse" if mkt_copy['is_inverse'] else "N/A"

        # --- Log key details ---
        price_prec_info = mkt_copy.get('precision', {}).get('price', 'N/A')
        amount_prec_info = mkt_copy.get('precision', {}).get('amount', 'N/A')
        min_amount = mkt_copy.get('limits', {}).get('amount', {}).get('min', 'N/A')
        min_cost = mkt_copy.get('limits', {}).get('cost', {}).get('min', 'N/A')
        contract_size = mkt_copy.get('contractSize', 'N/A')

        lg.debug(f"Market Info {symbol}: Type={mkt_type}, ContractType={contract_type}, IsContract={mkt_copy['is_contract']}, "
                 f"PricePrecInfo={price_prec_info}, AmtPrecInfo={amount_prec_info}, MinAmt={min_amount}, MinCost={min_cost}, ContractSize={contract_size}")

        # --- Validate essential info ---
        precision = mkt_copy.get('precision', {})
        limits = mkt_copy.get('limits', {})
        amount_limits = limits.get('amount', {})
        # Price precision/tick size is crucial
        if precision.get('price') is None:
             lg.error(f"{COLOR_ERROR}Market {symbol} lacks essential 'precision.price' (tick size) information.{RESET}")
             return None
        # Amount precision/step size is crucial
        if precision.get('amount') is None:
             lg.error(f"{COLOR_ERROR}Market {symbol} lacks essential 'precision.amount' (step size / places) information.{RESET}")
             return None
        # Minimum amount limit is needed for order validation
        if amount_limits.get('min') is None:
             lg.warning(f"{COLOR_WARNING}Market {symbol} lacks 'limits.amount.min'. Order size validation might be incomplete.{RESET}")
             # Decide if this is critical - for now, allow proceeding with warning.

        # Validate that precision/limits are usable numbers
        if safe_decimal(precision.get('price')) is None or safe_decimal(precision.get('price')) <= 0:
            lg.error(f"{COLOR_ERROR}Market {symbol} has invalid 'precision.price': {precision.get('price')}.{RESET}")
            return None
        if safe_decimal(precision.get('amount')) is None:  # Allow 0 for integer precision, but step size must be > 0 if float/str
             amount_prec_val = precision.get('amount')
             if not isinstance(amount_prec_val, int):  # If it's not int (places), it must be a valid step size
                 if safe_decimal(amount_prec_val) is None or safe_decimal(amount_prec_val) <= 0:
                      lg.error(f"{COLOR_ERROR}Market {symbol} has invalid 'precision.amount': {amount_prec_val}.{RESET}")
                      return None
        if safe_decimal(amount_limits.get('min')) is None:
            lg.warning(f"{COLOR_WARNING}Market {symbol} has invalid 'limits.amount.min': {amount_limits.get('min')}. Defaulting to 0 for checks.{RESET}")
            # Adjust the market info copy if needed, or handle downstream
            mkt_copy['limits']['amount']['min'] = 0

        return mkt_copy

    except ccxt.BadSymbol as e:
        lg.error(f"{COLOR_ERROR}Symbol '{symbol}' is invalid or not supported by {exchange.id}: {e}{RESET}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
        lg.error(f"{COLOR_ERROR}API Error getting market info for {symbol}: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: dict,
    exchange: ccxt.Exchange,  # Pass exchange for formatting/precision methods
    logger: logging.Logger | None = None
) -> Decimal | None:
    """Calculates position size considering balance, risk percentage, SL distance,
    contract type (linear/inverse/spot), and market constraints (min/max amount, cost, precision).

    Args:
        balance: Available quote currency balance (Decimal).
        risk_per_trade: Risk percentage (e.g., 0.01 for 1%).
        initial_stop_loss_price: Initial SL price (Decimal). Must be different from entry.
        entry_price: Potential entry price (Decimal).
        market_info: Enhanced market info dictionary from get_market_info().
        exchange: Initialized ccxt.Exchange object (for formatting methods).
        logger: Logger instance.

    Returns:
        Calculated and adjusted position size (Decimal) in base currency or contracts,
        rounded to the market's amount precision/step size, or None on failure.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quote', 'QUOTE')
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('is_linear', False)
    is_inverse = market_info.get('is_inverse', False)
    # Determine the unit of the calculated size for logging
    size_unit = base_currency  # Default for Spot and Linear Contracts
    if is_inverse:
        size_unit = "Contracts"  # Inverse contracts are typically sized in contracts (e.g., USD value)

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or balance <= 0:
        lg.error(f"Pos Sizing Fail ({symbol}): Invalid or non-positive balance ({balance}).")
        return None
    if not (isinstance(risk_per_trade, (float, int)) and 0 < risk_per_trade < 1):
        lg.error(f"Pos Sizing Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be a number > 0 and < 1.")
        return None
    if not isinstance(entry_price, Decimal) or entry_price <= 0:
         lg.error(f"Pos Sizing Fail ({symbol}): Invalid or non-positive entry price ({entry_price}).")
         return None
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= 0:
         lg.error(f"Pos Sizing Fail ({symbol}): Invalid or non-positive SL price ({initial_stop_loss_price}).")
         return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Pos Sizing Fail ({symbol}): Stop Loss price cannot be equal to entry price.")
        return None

    # Validate required market info keys
    if 'precision' not in market_info or 'limits' not in market_info:
         lg.error(f"Pos Sizing Fail ({symbol}): Market info missing 'precision' or 'limits'. Cannot proceed.")
         return None

    try:
        # --- Core Calculation ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))  # Amount of quote currency to risk
        sl_distance_price = abs(entry_price - initial_stop_loss_price)  # Price difference for SL

        if sl_distance_price <= 0:  # Should be caught by earlier check, but safety first
            lg.error(f"Pos Sizing Fail ({symbol}): SL distance is zero or negative ({sl_distance_price}).")
            return None

        # Get contract size (use Decimal, default to 1 if missing/invalid)
        contract_size = safe_decimal(market_info.get('contractSize', '1'), Decimal('1'))
        if contract_size <= 0:
             lg.warning(f"{COLOR_WARNING}Invalid contract size ({market_info.get('contractSize')}) for {symbol}. Defaulting to 1.{RESET}")
             contract_size = Decimal('1')

        calculated_size = Decimal('0')
        risk_per_unit_quote = Decimal('0')

        # Calculate size based on contract type
        if not is_contract or is_linear:  # Spot or Linear Contract
            # Size is in Base Currency (e.g., BTC for BTC/USDT)
            # Risk per unit (of base currency) = SL distance * contract size (usually 1 for linear/spot)
            risk_per_unit_quote = sl_distance_price * contract_size
            if risk_per_unit_quote > 0:
                calculated_size = risk_amount_quote / risk_per_unit_quote
                lg.debug(f"Pos Sizing (Linear/Spot {symbol}): RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_price}, CtrSize={contract_size}, RiskPerUnit={risk_per_unit_quote:.8f}")
            else:
                 lg.error(f"Pos Sizing Fail (Linear/Spot {symbol}): Risk per unit is zero or negative ({risk_per_unit_quote}). Check inputs.")
                 return None
        elif is_inverse:  # Inverse Contract
            # Size is in Contracts (e.g., number of USD contracts for BTC/USD inverse)
            # Risk per Contract = (SL distance * Contract Value in Quote) / Entry Price
            # Contract Value is typically the contract_size (e.g., 1 USD)
            if entry_price > 0:
                risk_per_unit_quote = (sl_distance_price * contract_size) / entry_price  # Risk per 1 Contract in Quote currency
                if risk_per_unit_quote > 0:
                    calculated_size = risk_amount_quote / risk_per_unit_quote
                    lg.debug(f"Pos Sizing (Inverse {symbol}): RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_price}, CtrSize={contract_size}, Entry={entry_price}, RiskPerContract={risk_per_unit_quote:.8f}")
                else:
                    lg.error(f"Pos Sizing Fail (Inverse {symbol}): Risk per contract is zero or negative ({risk_per_unit_quote}). Check inputs.")
                    return None
            else:  # Should have been caught by input validation
                 lg.error(f"Pos Sizing Fail (Inverse {symbol}): Entry price is zero or negative.")
                 return None
        else:  # Unknown contract type?
             lg.error(f"Pos Sizing Fail ({symbol}): Unknown contract type (Not Spot, Linear, or Inverse). Market Info: {market_info}")
             return None

        # --- Log Initial Calculation ---
        price_fmt = f".{abs(entry_price.normalize().as_tuple().exponent)}f"  # Dynamic price formatting
        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price:{price_fmt}}, SL={initial_stop_loss_price:{price_fmt}}, SLDist={sl_distance_price:{price_fmt}}")
        lg.info(f"  Contract Type: {'Spot' if not is_contract else 'Linear' if is_linear else 'Inverse'}, ContractSize={contract_size}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Constraints ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})

        # Get min/max limits (use safe_decimal with defaults)
        min_amount = safe_decimal(amount_limits.get('min'), Decimal('0'))
        max_amount = safe_decimal(amount_limits.get('max'), Decimal('Infinity'))
        min_cost = safe_decimal(cost_limits.get('min'), Decimal('0'))
        max_cost = safe_decimal(cost_limits.get('max'), Decimal('Infinity'))

        adjusted_size = calculated_size

        # 1. Clamp size by Min/Max Amount Limits
        original_calc_size_before_amount_limits = adjusted_size
        if adjusted_size < min_amount:
            lg.warning(f"{COLOR_WARNING}Calculated size {adjusted_size:.8f} < min amount {min_amount}. Adjusting to min amount.{RESET}")
            adjusted_size = min_amount
        if adjusted_size > max_amount:  # Check against max_amount AFTER min_amount adjustment
            lg.warning(f"{COLOR_WARNING}Calculated size {adjusted_size:.8f} > max amount {max_amount}. Capping at max amount.{RESET}")
            adjusted_size = max_amount
        if adjusted_size != original_calc_size_before_amount_limits:
             lg.info(f"  Size after Amount Limits: {adjusted_size:.8f} {size_unit}")

        # 2. Check and Adjust by Min/Max Cost Limits
        # Estimate cost based on the potentially amount-adjusted size
        current_cost = Decimal('0')
        if is_linear or not is_contract:  # Cost = Size * Price * ContractSize (usually 1)
            current_cost = adjusted_size * entry_price * contract_size
        elif is_inverse:  # Inverse Cost = Size (in Contracts) * ContractSize (Quote Value per contract)
             current_cost = adjusted_size * contract_size

        lg.debug(f"  Cost Check: Size={adjusted_size:.8f}, Est. Cost={current_cost:.4f} {quote_currency}, MinCost={min_cost}, MaxCost={max_cost}")

        cost_adjusted = False

        # Adjust for Min Cost
        if min_cost is not None and min_cost > 0 and current_cost < min_cost:
            lg.warning(f"{COLOR_WARNING}Est. cost {current_cost:.4f} < min cost {min_cost}. Attempting to increase size to meet min cost.{RESET}")
            required_size_for_min_cost = Decimal('0')
            try:
                if is_linear or not is_contract:
                     if entry_price > 0 and contract_size > 0: required_size_for_min_cost = min_cost / (entry_price * contract_size)
                elif is_inverse:
                     if contract_size > 0: required_size_for_min_cost = min_cost / contract_size

                if required_size_for_min_cost <= 0: raise ValueError("Calculated required size is non-positive")
            except (ValueError, InvalidOperation, ZeroDivisionError) as calc_err:
                 lg.error(f"{COLOR_ERROR}Cannot calculate required size for min cost due to error: {calc_err}. Aborting size calculation.{RESET}")
                 return None

            lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f} {size_unit}")

            # Check if required size exceeds max amount or max cost
            if required_size_for_min_cost > max_amount:
                lg.error(f"{COLOR_ERROR}Cannot meet min cost ({min_cost}): Required size {required_size_for_min_cost:.8f} exceeds max amount limit ({max_amount}). Aborted.{RESET}")
                return None
            required_cost = Decimal('0')
            if is_linear or not is_contract: required_cost = required_size_for_min_cost * entry_price * contract_size
            else: required_cost = required_size_for_min_cost * contract_size
            if max_cost is not None and max_cost > 0 and required_cost > max_cost:
                 lg.error(f"{COLOR_ERROR}Cannot meet min cost ({min_cost}): Required size {required_size_for_min_cost:.8f} results in cost ({required_cost:.4f}) exceeding max cost limit ({max_cost}). Aborted.{RESET}")
                 return None

            # Use the size required for min cost
            adjusted_size = required_size_for_min_cost
            cost_adjusted = True

        # Adjust for Max Cost (only if min cost didn't apply or didn't cause adjustment)
        elif max_cost is not None and max_cost > 0 and current_cost > max_cost:
            lg.warning(f"{COLOR_WARNING}Est. cost {current_cost:.4f} > max cost {max_cost}. Reducing size to meet max cost.{RESET}")
            allowed_size_for_max_cost = Decimal('0')
            try:
                if is_linear or not is_contract:
                    if entry_price > 0 and contract_size > 0: allowed_size_for_max_cost = max_cost / (entry_price * contract_size)
                elif is_inverse:
                    if contract_size > 0: allowed_size_for_max_cost = max_cost / contract_size

                if allowed_size_for_max_cost <= 0: raise ValueError("Calculated allowed size is non-positive")
            except (ValueError, InvalidOperation, ZeroDivisionError) as calc_err:
                 lg.error(f"{COLOR_ERROR}Cannot calculate allowed size for max cost due to error: {calc_err}. Aborting size calculation.{RESET}")
                 return None

            lg.info(f"  Allowed size for max cost: {allowed_size_for_max_cost:.8f} {size_unit}")

            # Check if allowed size is below min amount
            if allowed_size_for_max_cost < min_amount:
                lg.error(f"{COLOR_ERROR}Cannot meet max cost ({max_cost}): Allowed size {allowed_size_for_max_cost:.8f} is below min amount limit ({min_amount}). Aborted.{RESET}")
                return None

            # Use the size allowed by max cost
            adjusted_size = allowed_size_for_max_cost
            cost_adjusted = True

        if cost_adjusted:
            lg.info(f"  Size after Cost Limits: {adjusted_size:.8f} {size_unit}")

        # 3. Apply Amount Precision/Step Size (Truncate/Floor)
        final_size = Decimal('0')
        precision.get('amount')  # Could be int (places) or float/str (step)

        try:
            # Use the analyzer's rounding method which handles both step size and decimal places
            final_size = TradingAnalyzer(pd.DataFrame(), lg, {}, market_info).round_amount(adjusted_size)

            if final_size is None:
                lg.error(f"{COLOR_ERROR}Failed to apply amount precision/step size to {adjusted_size}. Aborting.{RESET}")
                return None

            lg.info(f"  Applied amount precision/step (Truncated/Floored): {adjusted_size:.8f} -> {final_size} {size_unit}")

        except Exception as fmt_err:
            lg.error(f"{COLOR_ERROR}Error applying amount precision for {symbol}: {fmt_err}. Using unrounded value and logging error.{RESET}", exc_info=True)
            # Fallback might be risky, safer to abort
            # final_size = adjusted_size # Fallback - use with caution
            return None

        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{COLOR_ERROR}Position size became zero or negative ({final_size}) after precision/limit adjustments. Aborted.{RESET}")
            return None

        # Final check against min amount (allow very small tolerance for floating point issues if conversion happened)
        if final_size < min_amount and not math.isclose(float(final_size), float(min_amount), rel_tol=1e-9, abs_tol=1e-9):
            lg.error(f"{COLOR_ERROR}Final size {final_size} after precision is still less than min amount {min_amount}. Aborted.{RESET}")
            return None

        # Final check against min cost if applicable (allow small tolerance)
        if min_cost is not None and min_cost > 0:
            final_cost = Decimal('0')
            if is_linear or not is_contract: final_cost = final_size * entry_price * contract_size
            elif is_inverse: final_cost = final_size * contract_size

            if final_cost < min_cost and not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6, abs_tol=1e-6):
                 lg.error(f"{COLOR_ERROR}Final cost {final_cost:.4f} after precision is still less than min cost {min_cost}. Aborted.{RESET}")
                 return None

        lg.info(f"{COLOR_SUCCESS}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(f"{COLOR_ERROR}Pos Sizing Error ({symbol}): Missing key in market_info: {e}. Check market data validity. Info: {market_info}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Fetches and validates the *single* open position for a specific symbol,
    handling different ccxt methods and response structures (especially Bybit V5).
    Enhances the position dictionary with standardized 'side', SL/TP prices, and TSL info.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        logger: Logger instance.

    Returns:
        Enhanced position dictionary if an active position exists (size != 0),
        or None if no active position is found or an error occurs.
    """
    lg = logger
    position: dict | None = None
    size_threshold = Decimal('1e-9')  # Threshold to consider a position size non-zero

    try:
        lg.debug(f"Fetching position for symbol: {symbol}")
        params = {}
        # Add Bybit V5 category parameter based on market info
        if 'bybit' in exchange.id.lower():
            try:
                market = exchange.market(symbol)  # Assume loaded
                if market:
                    category = 'linear' if market.get('linear', True) else 'inverse'
                    params['category'] = category
                    lg.debug(f"Using params for fetch_position(s): {params}")
                else:
                     lg.warning(f"Market info not found for {symbol} during position check. Assuming 'linear' category.")
                     params['category'] = 'linear'
            except (KeyError, ccxt.BadSymbol) as e:
                 lg.warning(f"Error getting market info for {symbol} during position check ({e}). Assuming 'linear' category.")
                 params['category'] = 'linear'

        # --- Attempt fetching position ---
        fetched_positions: list[dict] = []

        # 1. Try fetch_position (singular) first if supported (more efficient)
        if exchange.has.get('fetchPosition'):
             try:
                 lg.debug(f"Attempting exchange.fetch_position('{symbol}', params={params})")
                 # Note: fetch_position might return a dict with size 0 if no pos, or raise NotSupported/ExchangeError
                 single_pos_data = exchange.fetch_position(symbol, params=params)
                 # Check if the returned dict represents an actual position
                 pos_size_str = single_pos_data.get('contracts') or single_pos_data.get('info', {}).get('size')
                 pos_size = safe_decimal(pos_size_str, Decimal('0'))

                 if abs(pos_size) > size_threshold:
                      fetched_positions = [single_pos_data]
                      lg.debug("fetch_position returned an active position.")
                 else:
                      lg.debug("fetch_position returned data, but size is zero or missing. Assuming no active position.")
                      # Treat as no position found

             except ccxt.NotSupported as e:
                  lg.debug(f"fetch_position not supported by exchange or for this symbol/params: {e}. Falling back.")
             except ccxt.ExchangeError as e:
                 # Specific codes indicating "no position"
                 no_pos_codes = ['110025']  # Bybit V5: position idx not exist / position does not exist
                 no_pos_msgs = ['position does not exist', 'no position found', 'position idx not exist']
                 err_str = str(e).lower()
                 err_code = getattr(e, 'code', None)  # Get error code if available
                 if str(err_code) in no_pos_codes or any(msg in err_str for msg in no_pos_msgs):
                      lg.info(f"No position found for {symbol} via fetch_position (Code/Msg: {e}).")
                      # This is expected, not an error state.
                 else:
                      # Re-raise other exchange errors if fetchPosition fails unexpectedly
                      lg.warning(f"fetch_position failed unexpectedly for {symbol}: {e}. Falling back to fetch_positions.")
             except Exception as e:
                  lg.warning(f"Unexpected error in fetch_position for {symbol}: {e}. Falling back to fetch_positions.")

        # 2. Fallback to fetch_positions (plural) if fetchPosition failed, not supported, or returned no active pos
        if not fetched_positions and exchange.has.get('fetchPositions'):
            try:
                 # Try fetching positions for the specific symbol first
                 lg.debug(f"Attempting exchange.fetch_positions(symbols=['{symbol}'], params={params})")
                 fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
                 lg.debug(f"fetch_positions returned {len(fetched_positions)} entries for {symbol}.")
            except ccxt.ArgumentsRequired:
                 # If exchange requires fetching all positions
                 lg.debug("Fetching all positions as exchange requires it...")
                 all_positions = exchange.fetch_positions(params=params)  # Pass category here too if needed
                 fetched_positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Filtered {len(fetched_positions)} positions for {symbol} from all positions.")
            except ccxt.ExchangeError as e:
                 # Handle errors indicating no position found within fetch_positions response
                 no_pos_msgs = ['no position found', 'position does not exist', 'position idx not exist']
                 err_str = str(e).lower()
                 if any(msg in err_str for msg in no_pos_msgs):
                     lg.info(f"No position found for {symbol} via fetch_positions (Exchange message: {err_str}).")
                     fetched_positions = []  # Ensure list is empty
                 else:
                     lg.error(f"Exchange error during fetch_positions for {symbol}: {e}", exc_info=True)
                     return None  # Treat other errors as failure
            except Exception as e:
                 lg.error(f"Unexpected error fetching positions for {symbol}: {e}", exc_info=True)
                 return None

        # --- Find and Validate the Active Position from the fetched list ---
        active_position_raw: dict | None = None
        for pos_data in fetched_positions:
            # Consolidate size fetching from various possible keys
            size_val = pos_data.get('contracts')  # Standard ccxt v1
            if size_val is None: size_val = pos_data.get('contractSize')  # Alternative standard v1
            if size_val is None: size_val = pos_data.get('info', {}).get('size')  # Bybit V5 info size
            if size_val is None: size_val = pos_data.get('info', {}).get('positionAmt')  # Binance info size
            if size_val is None: size_val = pos_data.get('amount')  # Another possible key (less common for positions)

            if size_val is None:
                lg.debug(f"Skipping position entry, missing/null size field: {pos_data}")
                continue

            position_size = safe_decimal(size_val)
            if position_size is not None and abs(position_size) > size_threshold:
                if active_position_raw is not None:
                     lg.warning(f"{COLOR_WARNING}Multiple active position entries found for {symbol}. Using the first one found. Check exchange mode (Hedge vs One-Way).{RESET}")
                     # Consider adding logic here if hedge mode needs specific handling
                else:
                    active_position_raw = pos_data.copy()  # Work on a copy
                    lg.debug(f"Found candidate active position entry: Size={position_size}")
                    # Keep checking in case of multiple entries (warning will trigger)
                # break # Uncomment if you strictly want only the first one found

        # --- Process and Enhance the Found Active Position ---
        if active_position_raw:
            position = active_position_raw  # Use the found raw position
            info_dict = position.get('info', {})  # Standardized info dict

            # --- Standardize Side ('long' or 'short') ---
            side = position.get('side')  # Standard ccxt side
            size_decimal = safe_decimal(position.get('contracts', info_dict.get('size', '0')), Decimal('0'))  # Get size again reliably

            if side not in ['long', 'short']:
                info_side = info_dict.get('side', '').lower()  # Bybit: 'Buy'/'Sell'/'None'
                if info_side == 'buy': side = 'long'
                elif info_side == 'sell': side = 'short'
                elif size_decimal > size_threshold: side = 'long'  # Infer from positive size
                elif size_decimal < -size_threshold: side = 'short'  # Infer from negative size
                else: side = None  # Cannot determine side

            if side is None:
                 lg.warning(f"Position found for {symbol}, but size ({size_decimal}) is near zero or side undetermined. Treating as no active position.")
                 return None

            position['side'] = side  # Store standardized side

            # --- Populate Standard CCXT Fields from Info Dict if Missing ---
            # Ensures consistency regardless of which fetch method worked
            if position.get('entryPrice') is None: position['entryPrice'] = info_dict.get('entryPrice', info_dict.get('avgPrice'))  # entryPrice preferred, avgPrice fallback
            if position.get('markPrice') is None: position['markPrice'] = info_dict.get('markPrice')
            if position.get('liquidationPrice') is None: position['liquidationPrice'] = info_dict.get('liqPrice')
            if position.get('unrealizedPnl') is None: position['unrealizedPnl'] = info_dict.get('unrealisedPnl', info_dict.get('unrealizedPnl'))  # Check both spellings
            if position.get('collateral') is None: position['collateral'] = info_dict.get('positionIM', info_dict.get('collateral'))  # Initial margin or collateral
            if position.get('leverage') is None: position['leverage'] = info_dict.get('leverage')
            if position.get('contracts') is None: position['contracts'] = info_dict.get('size')  # Ensure size is in standard field
            if position.get('symbol') is None: position['symbol'] = info_dict.get('symbol')  # Ensure symbol is present

            # --- Enhance with SL/TP/TSL from 'info' (Focus on Bybit V5 structure) ---
            price_prec = 4  # Default price precision for logging SL/TP
            try:
                market = exchange.market(symbol)
                tick_size = safe_decimal(market['precision']['price'])
                if tick_size: price_prec = abs(tick_size.normalize().as_tuple().exponent)
            except Exception: pass  # Ignore errors getting market precision here

            def get_valid_price_from_info(key: str) -> Decimal | None:
                """Safely gets and validates a price field from info dict."""
                val_str = info_dict.get(key)
                val_dec = safe_decimal(val_str)
                # Treat '0' or '0.0' etc. as None (meaning not set)
                return val_dec if val_dec and val_dec > 0 else None

            # Parse SL/TP (Bybit V5: 'stopLoss', 'takeProfit')
            sl_price = get_valid_price_from_info('stopLoss')
            tp_price = get_valid_price_from_info('takeProfit')
            position['stopLossPrice'] = sl_price  # Add to standard dict
            position['takeProfitPrice'] = tp_price  # Add to standard dict

            # Parse TSL (Bybit V5: 'trailingStop' is distance/rate string, 'activePrice' is string trigger)
            tsl_value_str = info_dict.get('trailingStop', '0')  # Is a string like "0", "50", "0.005"
            tsl_value_dec = safe_decimal(tsl_value_str)
            tsl_active = tsl_value_dec is not None and tsl_value_dec > 0  # TSL is considered active if distance > 0

            tsl_activation_price = get_valid_price_from_info('activePrice')  # Price level for activation

            position['trailingStopLossValue'] = tsl_value_dec if tsl_active else None  # Store Decimal distance or None
            position['trailingStopLossActive'] = tsl_active  # Boolean flag
            position['trailingStopLossActivationPrice'] = tsl_activation_price  # Store Decimal activation price or None

            # Log enhanced position details
            details = {
                'Symbol': position.get('symbol', symbol),  # Use symbol from pos data if available
                'Side': side.upper(),
                'Size': f"{size_decimal}",
                'Entry': f"{safe_decimal(position.get('entryPrice')):.{price_prec}f}" if position.get('entryPrice') else 'N/A',
                'Mark': f"{safe_decimal(position.get('markPrice')):.{price_prec}f}" if position.get('markPrice') else 'N/A',
                'Liq': f"{safe_decimal(position.get('liquidationPrice')):.{price_prec}f}" if position.get('liquidationPrice') else 'N/A',
                'uPnL': f"{safe_decimal(position.get('unrealizedPnl')):.4f}" if position.get('unrealizedPnl') else 'N/A',
                'Coll': f"{safe_decimal(position.get('collateral')):.4f}" if position.get('collateral') else 'N/A',
                'Lev': f"{safe_decimal(position.get('leverage')):.1f}x" if position.get('leverage') else 'N/A',
                'TP': f"{tp_price:.{price_prec}f}" if tp_price else 'None',
                'SL': f"{sl_price:.{price_prec}f}" if sl_price else 'None',
                'TSL': (f"Active (Val={tsl_value_dec}, ActAt={tsl_activation_price:.{price_prec}f})"
                        if tsl_active and tsl_activation_price else
                        f"Active (Val={tsl_value_dec})" if tsl_active else 'None'),
            }
            details_str = ', '.join(f"{k}={v}" for k, v in details.items())
            lg.info(f"{COLOR_SUCCESS}Active Position Found: {details_str}{RESET}")
            return position  # Return the enhanced dictionary
        else:
            lg.info(f"No active position found for {symbol} after checking fetch results.")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{COLOR_ERROR}Authentication error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Avoid logging expected "no position" errors again
        no_pos_codes = ['110025']
        no_pos_msgs = ['position does not exist', 'no position found', 'position idx not exist']
        err_str = str(e).lower()
        err_code = getattr(e, 'code', None)
        if not (str(err_code) in no_pos_codes or any(msg in err_str for msg in no_pos_msgs)):
             lg.error(f"{COLOR_ERROR}Exchange error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error checking positions for {symbol}: {e}{RESET}", exc_info=True)

    return None  # Return None if any error occurred or no active position found


def place_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,  # 'buy' or 'sell'
    amount: Decimal,  # Base currency or contracts
    market_info: dict,
    logger: logging.Logger,
    params: dict | None = None
) -> dict | None:
    """Places a market order with safety checks, correct parameters (incl. Bybit V5),
    and robust error handling.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        side: 'buy' or 'sell'.
        amount: Order size (positive Decimal). Unit depends on market (base for spot/linear, contracts for inverse).
        market_info: Enhanced market info dictionary.
        logger: Logger instance.
        params: Additional parameters for create_order (e.g., {'reduceOnly': True}).

    Returns:
        The order dictionary from ccxt if successful, or None if failed or trading disabled.
    """
    lg = logger
    if not CONFIG.get("enable_trading", False):
        lg.warning(f"{COLOR_WARNING}TRADING DISABLED. Skipping market {side} order for {amount} {symbol}.{RESET}")
        return None

    if not isinstance(amount, Decimal) or amount <= 0:
        lg.error(f"{COLOR_ERROR}Invalid amount for market order: {amount}. Must be a positive Decimal.{RESET}")
        return None

    # Ensure amount respects market precision/step size before sending
    analyzer_temp = TradingAnalyzer(pd.DataFrame(), lg, {}, market_info)  # Temp analyzer for rounding
    rounded_amount = analyzer_temp.round_amount(amount)
    del analyzer_temp
    if rounded_amount is None or rounded_amount <= 0:
         lg.error(f"{COLOR_ERROR}Amount {amount} became invalid ({rounded_amount}) after rounding for {symbol}. Cannot place order.{RESET}")
         return None
    if rounded_amount != amount:
         lg.info(f"Market order amount rounded from {amount} to {rounded_amount} {market_info.get('base', '')} for {symbol} precision.")
         amount = rounded_amount  # Use the rounded amount

    # --- Prepare Order Parameters ---
    order_params = params.copy() if params else {}
    is_reduce_only = order_params.get('reduceOnly', False)

    # --- Add Exchange Specific Params (Bybit V5 Example) ---
    if 'bybit' in exchange.id.lower():
        # Category (Linear/Inverse)
        if 'category' not in order_params:
            category = 'linear' if market_info.get('is_linear', True) else 'inverse'
            order_params['category'] = category
            lg.debug(f"Setting Bybit category='{category}' for market order.")

        # Position Mode (Assume One-way if not specified) - Crucial for Bybit V5
        # 0=One-Way, 1=Buy Hedge, 2=Sell Hedge
        # We primarily operate in One-Way for this bot logic
        if 'positionIdx' not in order_params:
            order_params['positionIdx'] = 0
            lg.debug("Setting default Bybit positionIdx=0 (One-way mode) for market order.")

        # Add timeInForce if needed (Market orders usually default correctly)
        # order_params['timeInForce'] = 'ImmediateOrCancel' # Or 'FillOrKill' - check API docs

    # Determine order description for logging
    size_unit = market_info.get('base', '') if market_info.get('linear', True) or not market_info.get('is_contract') else "Contracts"
    order_desc = f"{side.upper()} {amount} {size_unit} {symbol} MARKET"
    if is_reduce_only: order_desc += " [REDUCE_ONLY]"
    lg.info(f"Attempting to place order: {order_desc} with params: {order_params}")

    try:
        # Convert Decimal amount to float for ccxt create_order function
        amount_float = float(amount)

        # --- Place Order ---
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount_float,
            params=order_params
        )

        # --- Post-Order Handling ---
        # Market orders fill quickly, but response structure varies.
        # Add basic info if order seems successful but lacks details (rare)
        if order and not order.get('id'):
            order['id'] = f'simulated_market_fill_{int(time.time())}'
            lg.warning("Market order response lacked ID, using simulated ID.")
        if order and not order.get('status'):
            # Assume 'closed' (filled) for market orders if status missing, common on some exchanges
            order['status'] = 'closed'
            lg.warning("Market order response lacked status, assuming 'closed' (filled).")

        # Check status (should be 'closed' or 'filled')
        order_status = order.get('status', 'unknown').lower()
        if order_status in ['closed', 'filled']:
             lg.info(f"{COLOR_SUCCESS}Successfully placed and likely filled market order: {order_desc}. Order ID: {order.get('id')}{RESET}")
        else:
             # This shouldn't happen often for market orders, but log if it does
             lg.warning(f"{COLOR_WARNING}Market order placed ({order_desc}, ID: {order.get('id')}), but status is '{order_status}'. Manual check advised.{RESET}")

        lg.debug(f"Market order API response details: {order}")
        return order

    # --- Error Handling ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{COLOR_ERROR}Insufficient funds for {order_desc}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        # Common reasons: Size too small/large, cost too small, precision issue, reduceOnly mismatch
        err_str = str(e).lower()
        if "order size is invalid" in err_str or "amount" in err_str or "precision" in err_str:
             lg.error(f"{COLOR_ERROR}Invalid market order size/precision for {order_desc}: {e}. Check limits/rounding.{RESET}")
        elif "order cost" in err_str or "value" in err_str or "minimum" in err_str:
             lg.error(f"{COLOR_ERROR}Market order cost below minimum for {order_desc}: {e}. Check minCost limit.{RESET}")
        elif "reduce-only" in err_str:
             lg.error(f"{COLOR_ERROR}Reduce-only market order conflict for {order_desc}: {e}. Position size mismatch?{RESET}")
        else:
             lg.error(f"{COLOR_ERROR}Invalid market order {order_desc}: {e}. Check params/limits.{RESET}")
    except ccxt.ExchangeError as e:
        # Handle other specific exchange errors if known
        lg.error(f"{COLOR_ERROR}Exchange error placing {order_desc}: {e}{RESET}", exc_info=True)
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error placing {order_desc}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error placing {order_desc}: {e}{RESET}", exc_info=True)

    return None  # Return None if order placement failed


def set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_params: dict | None = None,  # e.g., {'trailingStop': '50', 'activePrice': '...'} String values for Bybit!
    current_position: dict | None = None,  # Pass pre-fetched enhanced position to avoid redundant API call
    logger: logging.Logger = None
) -> bool:
    """Sets or modifies Stop Loss (SL), Take Profit (TP), and Trailing Stop Loss (TSL)
    for an existing position using the appropriate ccxt method or specific API endpoint (e.g., Bybit V5).
    Only sends API request if changes are detected compared to the current position state.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        market_info: Enhanced market info dict.
        stop_loss_price: Desired SL price (Decimal). Use 0 or None to remove or leave unchanged.
        take_profit_price: Desired TP price (Decimal). Use 0 or None to remove or leave unchanged.
        trailing_stop_params: Dict with TSL parameters specific to the exchange API.
                              For Bybit V5: {'trailingStop': 'distance_str', 'activePrice': 'price_str' (optional)}
                              Use {'trailingStop': '0'} to remove TSL. None leaves it unchanged.
        current_position: Optional pre-fetched and enhanced position dictionary. If None, it will be fetched.
        logger: Logger instance.

    Returns:
        True if protection was set/modified successfully or no change was needed, False otherwise.
    """
    lg = logger or logging.getLogger(__name__)
    if not CONFIG.get("enable_trading", False):
        lg.warning(f"{COLOR_WARNING}TRADING DISABLED. Skipping protection setting for {symbol}.{RESET}")
        return False  # Cannot set protection if trading off

    # --- 1. Get Current Position Info (Fetch if not provided) ---
    position = current_position
    if position is None:
        lg.debug(f"Fetching position for {symbol} before setting protection...")
        position = get_open_position(exchange, symbol, lg)  # Use enhanced fetcher
        if position is None:
            lg.info(f"No open position found for {symbol}. Cannot set protection.")
            # This isn't an error if we just closed it, it's the expected state.
            return True  # No action needed

    # --- 2. Validate Position State ---
    position_side = position.get('side')  # 'long' or 'short'
    position_size_str = position.get('contracts', position.get('info', {}).get('size'))
    position_size = safe_decimal(position_size_str)
    size_threshold = Decimal('1e-9')

    if not position_side or position_size is None or abs(position_size) <= size_threshold:
        lg.info(f"Position side/size invalid or near zero for {symbol}. No protection action needed. Pos Info: {position}")
        return True  # No active position to protect

    # --- 3. Determine Current Protection State from Enhanced Position Data ---
    # Prices are already Decimals from get_open_position if they exist
    current_sl: Decimal | None = position.get('stopLossPrice')
    current_tp: Decimal | None = position.get('takeProfitPrice')
    current_tsl_val: Decimal | None = position.get('trailingStopLossValue')  # The distance/rate value
    current_tsl_active: bool = position.get('trailingStopLossActive', False)
    current_tsl_act_price: Decimal | None = position.get('trailingStopLossActivationPrice')

    lg.debug(f"Current Protection State ({symbol}): SL={current_sl}, TP={current_tp}, "
             f"TSL Active={current_tsl_active} (Val={current_tsl_val}, ActPrice={current_tsl_act_price})")

    # --- 4. Prepare API Parameters and Detect Changes ---
    params = {}
    log_parts = []
    needs_api_call = False

    # Get precision for formatting prices in logs and potentially API params
    price_prec = 4
    try:
        analyzer_temp = TradingAnalyzer(pd.DataFrame(), lg, {}, market_info)
        price_prec = analyzer_temp.get_price_precision()
        analyzer_temp.get_min_tick_size()
        del analyzer_temp
    except Exception:
         lg.warning("Could not get price precision from market info for protection setting.")
         Decimal('0.0001')  # Fallback tick for formatting

    # --- Exchange/API Specific Setup (Bybit V5 Example) ---
    is_bybit_v5 = 'bybit' in exchange.id.lower() and exchange.version == 'v5'
    if is_bybit_v5:
        # Category needed for v5 position endpoints
        params['category'] = 'linear' if market_info.get('is_linear', True) else 'inverse'
        # Position Index (0 for One-Way, 1/2 for Hedge) - Get from fetched position if possible
        params['positionIdx'] = position.get('info', {}).get('positionIdx', 0)  # Default to 0 (One-Way)
        # Bybit V5 uses a single endpoint '/v5/position/trading-stop' for TP/SL/TSL
        # It requires string format for prices/values.
        # It requires tpslMode ('Full' or 'Partial') and trigger prices.
        params['tpTriggerBy'] = 'MarkPrice'  # Common trigger setting
        params['slTriggerBy'] = 'MarkPrice'
        # tpslMode determined later based on which params are set
        lg.debug(f"Using Bybit V5 params: category={params['category']}, positionIdx={params['positionIdx']}")

    # --- Process Stop Loss ---
    sl_change_detected = False
    target_sl_str = None  # String representation for API
    if stop_loss_price is not None:  # If a new SL value is provided
        # Round the target SL price to the tick size
        rounded_sl = TradingAnalyzer(pd.DataFrame(), lg, {}, market_info).round_price(stop_loss_price)
        if rounded_sl is None:
            lg.error(f"Invalid target SL price {stop_loss_price} after rounding. Cannot set SL.")
            return False  # Cannot proceed with invalid SL

        target_sl_str = "0" if rounded_sl <= 0 else f"{rounded_sl:.{price_prec}f}"

        # Compare rounded target SL with current SL (handle None)
        if rounded_sl <= 0 and current_sl is not None:  # Request to remove existing SL
            sl_change_detected = True
            log_parts.append("SL=Remove")
        elif rounded_sl > 0 and rounded_sl != current_sl:  # Request to set/change SL
            sl_change_detected = True
            log_parts.append(f"SL={target_sl_str}")
        # else: No change needed

    if sl_change_detected:
        needs_api_call = True
        if is_bybit_v5: params['stopLoss'] = target_sl_str
        # Add logic for other exchange params if needed
    elif is_bybit_v5 and current_sl is not None:
         # If not changing SL, but other params might change, need to resubmit current SL value for Bybit V5
         params['stopLoss'] = f"{current_sl:.{price_prec}f}"

    # --- Process Take Profit ---
    tp_change_detected = False
    target_tp_str = None
    if take_profit_price is not None:  # If a new TP value is provided
        rounded_tp = TradingAnalyzer(pd.DataFrame(), lg, {}, market_info).round_price(take_profit_price)
        if rounded_tp is None:
            lg.error(f"Invalid target TP price {take_profit_price} after rounding. Cannot set TP.")
            return False

        target_tp_str = "0" if rounded_tp <= 0 else f"{rounded_tp:.{price_prec}f}"

        if rounded_tp <= 0 and current_tp is not None:  # Request to remove
            tp_change_detected = True
            log_parts.append("TP=Remove")
        elif rounded_tp > 0 and rounded_tp != current_tp:  # Request to set/change
            tp_change_detected = True
            log_parts.append(f"TP={target_tp_str}")
        # else: No change needed

    if tp_change_detected:
        needs_api_call = True
        if is_bybit_v5: params['takeProfit'] = target_tp_str
        # Add logic for other exchanges
    elif is_bybit_v5 and current_tp is not None:
         # Resubmit current TP if needed for Bybit V5 API call
         params['takeProfit'] = f"{current_tp:.{price_prec}f}"

    # --- Process Trailing Stop ---
    tsl_change_detected = False
    target_tsl_params = {}  # Store params specifically for TSL part of API call
    if trailing_stop_params is not None:  # If new TSL instructions provided
        # Extract target values (expect strings for Bybit V5)
        target_tsl_dist_str = trailing_stop_params.get('trailingStop')  # e.g., "50", "0.005", "0"
        target_tsl_act_str = trailing_stop_params.get('activePrice')   # e.g., "20000.5" or None

        target_tsl_dist_dec = safe_decimal(target_tsl_dist_str)  # Convert target distance for comparison

        # Validate target distance
        if target_tsl_dist_dec is None:
             lg.error(f"Invalid trailingStop value provided: '{target_tsl_dist_str}'. Cannot process TSL.")
             # Don't set needs_api_call = True based on invalid input
        else:
            # Check if TSL distance needs changing
            update_tsl_dist = False
            if target_tsl_dist_dec <= 0 and current_tsl_active:  # Request to remove TSL
                update_tsl_dist = True
                target_tsl_dist_str = "0"  # Ensure removal value is '0' string
                log_parts.append("TSL=Remove")
            elif target_tsl_dist_dec > 0 and target_tsl_dist_dec != current_tsl_val:  # Request to set/change TSL distance
                update_tsl_dist = True
                log_parts.append(f"TSL_Dist={target_tsl_dist_str}")
            # else: TSL distance is the same or wasn't active and target is 0

            if update_tsl_dist:
                tsl_change_detected = True
                target_tsl_params['trailingStop'] = target_tsl_dist_str  # Use string for Bybit

                # Handle Activation Price only if TSL distance > 0
                if target_tsl_dist_dec > 0:
                    target_tsl_act_dec = safe_decimal(target_tsl_act_str)  # Convert target activation for comparison
                    rounded_tsl_act = TradingAnalyzer(pd.DataFrame(), lg, {}, market_info).round_price(target_tsl_act_dec) if target_tsl_act_dec else None

                    if rounded_tsl_act and rounded_tsl_act > 0 and rounded_tsl_act != current_tsl_act_price:
                        act_price_str = f"{rounded_tsl_act:.{price_prec}f}"
                        target_tsl_params['activePrice'] = act_price_str
                        log_parts.append(f"TSL_ActAt={act_price_str}")
                    elif target_tsl_act_str is None and current_tsl_act_price is not None:
                        # If new params omit activation but it existed, Bybit might require sending '0' or omitting the key.
                        # Omitting seems safer based on docs, but test carefully.
                         log_parts.append("TSL_ActAt=Omitted(Previously set)")
                         # params.pop('activePrice', None) # Let Bybit handle default if omitted
                    # else: Activation price same or not provided / not changing

                else:  # Removing TSL (distance <= 0), ensure activePrice is not sent
                     target_tsl_params.pop('activePrice', None)
                     log_parts.append("TSL_ActAt=Removed(TSL off)")

            # else: No change needed for TSL distance

    if tsl_change_detected:
        needs_api_call = True
        if is_bybit_v5:
            params.update(target_tsl_params)  # Add TSL specific params
        # Add logic for other exchanges
    elif is_bybit_v5 and current_tsl_active:
         # Resubmit current TSL if needed for Bybit V5 API call when other params change
         params['trailingStop'] = str(current_tsl_val)  # Use current value as string
         if current_tsl_act_price:
              params['activePrice'] = f"{current_tsl_act_price:.{price_prec}f}"

    # --- 5. Make API Call Only If Changes Detected ---
    if not needs_api_call:
        lg.info(f"No changes detected for position protection ({symbol}). No API call needed.")
        return True

    # --- Finalize Bybit V5 Parameters ---
    if is_bybit_v5:
        # Ensure required fields are present with '0' if not explicitly set/changed
        params.setdefault('stopLoss', '0')
        params.setdefault('takeProfit', '0')
        params.setdefault('trailingStop', '0')  # Important: TSL defaults to 0 if not provided

        # Determine tpslMode ('Full' or 'Partial') - Crucial for Bybit V5
        # Use 'Partial' if setting TSL, or if setting BOTH TP and SL simultaneously.
        # Use 'Full' if setting ONLY TP or ONLY SL.
        has_tp = safe_decimal(params.get('takeProfit', '0'), 0) > 0
        has_sl = safe_decimal(params.get('stopLoss', '0'), 0) > 0
        has_tsl = safe_decimal(params.get('trailingStop', '0'), 0) > 0

        if has_tsl or (has_tp and has_sl):
             params['tpslMode'] = 'Partial'
        elif has_tp or has_sl:  # Only one of TP or SL is being set (or potentially modified)
             params['tpslMode'] = 'Full'
        else:  # Only removing TP/SL/TSL - Partial seems safer default
             params['tpslMode'] = 'Partial'
             lg.debug("Setting tpslMode to Partial as only removing protection.")

    # --- Log and Execute ---
    set_desc = f"Set Protection for {symbol} ({position_side.upper()} {position_size:.8f}): {', '.join(log_parts)}"
    lg.info(f"Attempting: {set_desc}")
    lg.debug(f"  API Call Params: {params}")

    try:
        response = None
        # --- Select Appropriate API Method ---
        if is_bybit_v5:
            lg.debug("Using Bybit V5 private_post /v5/position/trading-stop")
            response = exchange.private_post('/v5/position/trading-stop', params)
        elif exchange.has.get('editPosition'):  # Generic check (might not work for all exchanges)
             lg.debug("Attempting generic exchange.edit_position (Experimental)")
             # Need to translate params to what editPosition expects (highly variable)
             # This part requires specific implementation per exchange if not Bybit V5
             lg.warning("editPosition logic not fully implemented for non-Bybit exchanges.")
             # response = exchange.edit_position(symbol, params=params) # Example structure
             return False  # Indicate failure until implemented
        else:
            # Check for separate set_stop_loss, set_take_profit methods if needed
            lg.error(f"Protection setting (TP/SL/TSL) not implemented for exchange {exchange.id} / version {exchange.version}. Cannot proceed.")
            return False

        # --- Process Response ---
        lg.info(f"{COLOR_SUCCESS}Protection setting API request successful for {symbol}.{RESET}")
        lg.debug(f"Protection setting API response: {response}")

        # --- Response Validation (Bybit V5 Example) ---
        if isinstance(response, dict):
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', '')
            if ret_code == 0:
                lg.debug("API response indicates success (retCode 0).")
                return True  # Success
            else:
                # Check for non-critical "errors" that mean success or no change needed
                # Codes based on Bybit V5 documentation and common scenarios:
                # 110043: Set TP/SL orders can only be modified or cancelled (Might mean already set as requested)
                # 34036:  The order is not modified (No change detected by exchange)
                # 110025: position idx not match position mode (Warning if mode seems correct)
                # 110068: The trailing stop loss is not modified
                # 110067: The take profit/stop loss is not modified
                # 110090: tp/sl order maybe executed or cancelled (Order already closed/triggered?)
                non_error_codes = [110043, 34036, 110025, 110067, 110068, 110090]
                no_change_msgs = ["not modified", "same tpsl", "already closed", "already cancelled"]  # Keywords for non-errors
                err_str = ret_msg.lower()

                if ret_code in non_error_codes or any(msg in err_str for msg in no_change_msgs):
                    lg.warning(f"{COLOR_WARNING}Protection setting for {symbol} - Non-critical response code {ret_code}: '{ret_msg}'. Assuming success or no change required.{RESET}")
                    return True  # Treat as success/no action needed
                else:
                    lg.error(f"{COLOR_ERROR}Protection setting failed ({symbol}). API Code: {ret_code}, Msg: {ret_msg}{RESET}")
                    return False
        else:
            # If response is not a dict or format is unknown, assume success if no exception occurred (less reliable)
            lg.warning("Protection setting response format unknown or not dictionary, assuming success based on lack of exception.")
            return True

    # --- Error Handling for API Call ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{COLOR_ERROR}Insufficient funds during protection setting for {symbol}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{COLOR_ERROR}Invalid order parameters setting protection for {symbol}: {e}. Check values relative to price/position/liquidation.{RESET}")
    except ccxt.ExchangeError as e:
        # Handle already logged non-critical errors again just in case exception mapping catches them
        err_code = getattr(e, 'code', None)
        err_str = str(e).lower()
        non_error_codes = [110043, 34036, 110025, 110067, 110068, 110090]
        no_change_msgs = ["not modified", "same tpsl", "already closed", "already cancelled"]
        if str(err_code) in non_error_codes or any(msg in err_str for msg in no_change_msgs):
             lg.warning(f"{COLOR_WARNING}Protection setting for {symbol} - Caught non-critical error {err_code}: '{err_str}'. Assuming success/no change.{RESET}")
             return True  # Treat as success
        lg.error(f"{COLOR_ERROR}Exchange error setting protection for {symbol}: {e}{RESET}", exc_info=True)
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error setting protection for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

    return False  # Return False if any exception occurred


# --- Main Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger) -> None:
    """Performs one full cycle of analysis and potential trading action for a single symbol.
    Fetches data, analyzes, checks position, generates signals, manages risk, and executes trades.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: The trading symbol to analyze (e.g., 'BTC/USDT').
        config: The global configuration dictionary.
        logger: The logger instance specific to this symbol.
    """
    lg = logger
    lg.info(f"---== Analyzing {symbol} ==---")
    cycle_start_time = time.monotonic()

    try:
        # --- 1. Fetch Market Info (Crucial for precision, limits) ---
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info:
            lg.error(f"Skipping cycle for {symbol}: Could not retrieve valid market info.")
            return  # Cannot proceed without market info

        # --- 2. Fetch Kline Data ---
        interval_str = config.get("interval", "5")  # User interval (e.g., "5")
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(interval_str)
        if not ccxt_timeframe:
            lg.error(f"Invalid interval '{interval_str}' in config for {symbol}. Skipping cycle.")
            return

        # Determine appropriate lookback limit based on indicator needs
        min_required_data = 1010  # Fallback
        try:
             # Calculate min required length based on config dynamically
             buffer = 1010  # Add ample buffer for initial NaNs and stability
             min_len_volbot = 0
             if config.get("volbot_enabled", True):
                 min_len_volbot = max(
                     config.get("volbot_length", DEFAULT_VOLBOT_LENGTH) + 3,  # SWMA
                     config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
                     config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
                     config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H) + config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H) + 1,
                     config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L) + config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L) + 1
                 )
             min_len_risk = config.get("atr_period", DEFAULT_ATR_PERIOD)
             min_required_data = max(min_len_volbot, min_len_risk) + buffer
        except Exception as e:
             lg.error(f"Error calculating min required data length: {e}. Using fallback {min_required_data}.")

        kline_limit = min_required_data
        lg.debug(f"Fetching {kline_limit} klines for {symbol} {ccxt_timeframe}...")
        df_klines = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, limit=kline_limit, logger=lg)

        # Validate fetched klines
        min_acceptable_klines = 20  # Need a reasonable minimum for any analysis
        if df_klines.empty or len(df_klines) < min_acceptable_klines:
            lg.warning(f"{COLOR_WARNING}Insufficient kline data for {symbol} (got {len(df_klines)}, needed ~{min_required_data}, min acceptable {min_acceptable_klines}). Skipping analysis cycle.{RESET}")
            return

        # --- 3. Initialize Analyzer & Calculate Indicators ---
        analyzer = TradingAnalyzer(df=df_klines, logger=lg, config=config, market_info=market_info)
        if analyzer.df_processed.empty or not analyzer.strategy_state:
            lg.error(f"Failed to calculate indicators or update state for {symbol}. Skipping trading logic.")
            return  # Cannot proceed without analysis results

        # --- 4. Check Current Position ---
        # Use the enhanced get_open_position which returns None if no active position
        current_position = get_open_position(exchange, symbol, lg)  # Returns enhanced dict or None
        has_open_position = current_position is not None
        position_side = current_position.get('side') if has_open_position else None  # 'long' or 'short'
        position_entry_price = safe_decimal(current_position.get('entryPrice')) if has_open_position else None
        position_size_str = current_position.get('contracts') if has_open_position else '0'
        position_size = safe_decimal(position_size_str, Decimal('0'))  # Size in contracts/base

        # --- 5. Fetch Current Market Price ---
        current_price = fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price is None:
            # Fallback to last close price from klines if ticker fails
            last_close = analyzer.strategy_state.get('close')  # Already Decimal or None
            if isinstance(last_close, Decimal) and last_close > 0:
                current_price = last_close
                lg.warning(f"{COLOR_WARNING}Ticker price fetch failed for {symbol}. Using last kline close ({current_price}) for checks. Signal/execution might be based on slightly stale data.{RESET}")
            else:
                lg.error(f"{COLOR_ERROR}Failed to get current price for {symbol} from ticker and no valid last close price available. Cannot proceed with trading logic.{RESET}")
                return

        price_fmt = f".{analyzer.get_price_precision()}f"  # Use analyzer's precision for logging

        # --- 6. Generate Trading Signal ---
        signal = analyzer.generate_trading_signal()  # Returns "BUY", "SELL", or "HOLD"

        # --- 7. Position Management & Trading Logic ---
        trading_enabled = config.get("enable_trading", False)

        # =============================
        # == Scenario 1: IN a Position ==
        # =============================
        if has_open_position and position_entry_price and abs(position_size) > Decimal('1e-9'):
            pos_size_fmt = f".{analyzer.get_amount_precision()}f"
            lg.info(f"Managing existing {position_side.upper()} position ({position_size:{pos_size_fmt}} {symbol} @ {position_entry_price:{price_fmt}}). Current Price: {current_price:{price_fmt}}")

            # --- 7a. Check for Exit Signal ---
            # Exit if signal flips against the current position direction
            exit_signal_triggered = (position_side == 'long' and signal == "SELL") or \
                                    (position_side == 'short' and signal == "BUY")

            if exit_signal_triggered:
                reason = f"Opposing signal ({signal}) received while in {position_side.upper()} position"
                color = COLOR_DN if signal == "SELL" else COLOR_UP
                lg.info(f"{color}Exit Signal Triggered: {reason}. Attempting to close position.{RESET}")

                if trading_enabled:
                    close_side = 'sell' if position_side == 'long' else 'buy'
                    # Place market order to close, using reduceOnly flag
                    close_order = place_market_order(
                        exchange, symbol, close_side, abs(position_size),
                        market_info, lg, params={'reduceOnly': True}
                    )
                    if close_order:
                        lg.info(f"{COLOR_SUCCESS}Market order placed to close {position_side} position.{RESET}")
                        # Attempt to cancel associated SL/TP orders (best effort)
                        # Bybit V5 uses set_position_protection with SL/TP=0 to cancel existing orders
                        lg.info(f"Attempting to cancel any existing SL/TP orders for {symbol}...")
                        try:
                             cancel_success = set_position_protection(
                                 exchange, symbol, market_info,
                                 stop_loss_price=Decimal(0),  # Signal to remove SL
                                 take_profit_price=Decimal(0),  # Signal to remove TP
                                 trailing_stop_params={'trailingStop': '0'},  # Signal to remove TSL
                                 current_position=None,  # Force refetch to confirm closure if needed by API
                                 logger=lg
                             )
                             if cancel_success: lg.info("Cancellation request for SL/TP/TSL sent successfully.")
                             else: lg.warning(f"{COLOR_WARNING}Could not confirm cancellation of SL/TP/TSL orders for {symbol} after closing.{RESET}")
                        except Exception as cancel_err:
                            lg.warning(f"{COLOR_WARNING}Error attempting to cancel stop orders for {symbol}: {cancel_err}{RESET}")
                    else:
                        lg.error(f"{COLOR_ERROR}Failed to place market order to close {position_side} position. MANUAL INTERVENTION MAY BE REQUIRED!{RESET}")
                    # End cycle for this symbol after attempting closure
                    return
                else:
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have closed {position_side} position ({reason}).{RESET}")
                    # Even if trading disabled, stop further management for this cycle if exit triggered
                    return

            # --- 7b. Risk Management (Only if NO exit signal) ---
            else:
                lg.info(f"No exit signal. Performing risk management checks for {position_side} position...")
                # Store potential updates, only call API if something changes
                new_sl_price: Decimal | None = None  # Potential new SL (e.g., from BE)
                new_tsl_params: dict | None = None  # Potential new TSL settings (e.g., activation)
                needs_protection_update = False

                # Get current protection state directly from the enhanced position object
                current_sl = current_position.get('stopLossPrice')  # Decimal or None
                current_tp = current_position.get('takeProfitPrice')  # Decimal or None
                current_tsl_active = current_position.get('trailingStopLossActive', False)
                current_position.get('trailingStopLossValue')  # Decimal distance or None

                # --- i. Break-Even Logic ---
                enable_be = config.get("enable_break_even", True)
                if enable_be and not current_tsl_active:  # Optional: Disable BE if TSL is already active? Or let them compete? Current: Allow BE even if TSL active.
                    risk_atr = analyzer.strategy_state.get("atr_risk")  # Risk ATR (Decimal or None)
                    min_tick = analyzer.get_min_tick_size()  # Decimal

                    if risk_atr and risk_atr > 0 and min_tick > 0:
                        be_trigger_multiple = safe_decimal(config.get("break_even_trigger_atr_multiple", 1.0), Decimal(1.0))
                        be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        profit_target_for_be = risk_atr * be_trigger_multiple  # Profit needed in price points

                        # Calculate current profit in price points
                        current_profit_points = (current_price - position_entry_price) if position_side == 'long' else (position_entry_price - current_price)

                        lg.debug(f"BE Check: CurrentProfit={current_profit_points:{price_fmt}}, TargetProfit={profit_target_for_be:{price_fmt}}")

                        # Check if profit target reached
                        if current_profit_points >= profit_target_for_be:
                            # Calculate BE price (entry + offset)
                            offset_amount = min_tick * be_offset_ticks
                            be_price_raw = position_entry_price + offset_amount if position_side == 'long' else position_entry_price - offset_amount
                            # Round BE price away from entry to cover costs/slippage
                            be_rounding = ROUND_UP if position_side == 'long' else ROUND_DOWN
                            be_price_rounded = (be_price_raw / min_tick).quantize(Decimal('1'), rounding=be_rounding) * min_tick

                            # Check if this BE SL is actually better than the current SL (or if no SL exists)
                            is_be_sl_better = False
                            if current_sl is None:
                                is_be_sl_better = True  # No current SL, so BE is better
                            elif position_side == 'long' and be_price_rounded > current_sl or position_side == 'short' and be_price_rounded < current_sl:
                                is_be_sl_better = True

                            if is_be_sl_better:
                                lg.info(f"{COLOR_SUCCESS}Break-Even Triggered! Profit {current_profit_points:{price_fmt}} >= Target {profit_target_for_be:{price_fmt}}.{RESET}")
                                lg.info(f"  Moving SL from {current_sl:{price_fmt} if current_sl else 'None'} to BE price: {be_price_rounded:{price_fmt}}")
                                new_sl_price = be_price_rounded  # Store the potential new SL price
                                needs_protection_update = True
                            else:
                                lg.debug(f"BE triggered, but proposed BE SL {be_price_rounded:{price_fmt}} is not better than current SL {current_sl:{price_fmt} if current_sl else 'None'}. No change.")
                        # else: Profit target not reached for BE
                    elif enable_be:  # Log if BE enabled but inputs missing
                        lg.warning(f"Cannot calculate BE for {symbol}: Invalid Risk ATR ({risk_atr}) or Min Tick ({min_tick}).")

                # --- ii. Trailing Stop Activation / Management ---
                # This logic assumes the TSL, once set via set_position_protection, is managed by the exchange.
                # We only check if we need to *initially* set the TSL if it's enabled but not yet active.
                enable_tsl = config.get("enable_trailing_stop", True)
                if enable_tsl and not current_tsl_active:
                    # Get TSL parameters from config
                    tsl_callback_rate_str = str(config.get("trailing_stop_callback_rate", "0.005"))  # String for API
                    tsl_activation_perc = safe_decimal(config.get("trailing_stop_activation_percentage", "0.003"), Decimal('0'))

                    # Check profit percentage for activation (only if activation percentage > 0)
                    activate_tsl = False
                    if tsl_activation_perc <= 0:
                         # Activate immediately if percentage is zero (should have been set on entry)
                         # This logic block might indicate TSL should have been set on entry but failed, or config changed.
                         lg.warning(f"{COLOR_WARNING}TSL enabled with 0 activation threshold, but not active. Attempting to set TSL now.{RESET}")
                         activate_tsl = True
                    else:
                        # Calculate current profit percentage
                        current_profit_perc = Decimal('0')
                        if position_entry_price > 0:
                            current_profit_perc = (current_price / position_entry_price) - 1 if position_side == 'long' else 1 - (current_price / position_entry_price)

                        lg.debug(f"TSL Activation Check: CurrentProfit%={current_profit_perc:.4%}, ActivationThreshold%={tsl_activation_perc:.4%}")
                        if current_profit_perc >= tsl_activation_perc:
                            lg.info(f"{COLOR_SUCCESS}Trailing Stop activation profit threshold reached ({current_profit_perc:.2%} >= {tsl_activation_perc:.2%}).{RESET}")
                            activate_tsl = True

                    # If activation conditions met, prepare TSL params
                    if activate_tsl:
                        lg.info(f"Preparing to activate TSL with distance/rate: {tsl_callback_rate_str}")
                        # For Bybit V5, setting 'trailingStop' with a value > 0 activates it.
                        # 'activePrice' can be used to set a specific trigger price, otherwise Bybit uses its own logic.
                        # Let's omit activePrice here unless config explicitly requires it, to use Bybit's default activation.
                        new_tsl_params = {'trailingStop': tsl_callback_rate_str}
                        # Optional: Calculate and add activePrice if needed based on config/exchange behavior
                        # act_price_raw = position_entry_price * (1 + tsl_activation_perc) if position_side == 'long' else position_entry_price * (1 - tsl_activation_perc)
                        # rounded_act_price = analyzer.round_price(act_price_raw)
                        # if rounded_act_price: new_tsl_params['activePrice'] = f"{rounded_act_price:{price_fmt}}"

                        needs_protection_update = True
                        # Note: If BE also triggered, both SL and TSL will be sent in the update.
                        # Exchange behavior might prioritize one or handle combined updates. Test carefully.
                        if new_sl_price is not None:
                            lg.warning(f"{COLOR_WARNING}Both Break-Even SL ({new_sl_price:{price_fmt}}) and TSL activation triggered. Sending both updates. Exchange behavior may vary.{RESET}")

                # --- iii. Update Protection via API Call ---
                if needs_protection_update and trading_enabled:
                    lg.info(f"Attempting to update position protection for {symbol}...")
                    # Use the potential new SL from BE if set, otherwise keep current SL (or None)
                    final_sl_to_set = new_sl_price if new_sl_price is not None else current_sl
                    # TP generally remains unchanged during risk management (unless specific logic added)
                    final_tp_to_set = current_tp
                    # TSL params are set if activation occurred
                    final_tsl_params_to_set = new_tsl_params

                    success = set_position_protection(
                        exchange, symbol, market_info,
                        stop_loss_price=final_sl_to_set,  # Can be None
                        take_profit_price=final_tp_to_set,  # Can be None
                        trailing_stop_params=final_tsl_params_to_set,  # Can be None
                        current_position=current_position,  # Pass current state to avoid refetch
                        logger=lg
                    )
                    if success:
                        lg.info(f"{COLOR_SUCCESS}Successfully sent protection update request for {symbol}.{RESET}")
                    else:
                        lg.error(f"{COLOR_ERROR}Failed to update position protection for {symbol}. Check logs.{RESET}")

                elif needs_protection_update and not trading_enabled:
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have updated protection. Proposed SL={new_sl_price:{price_fmt} if new_sl_price else 'N/A'}, Proposed TSL Params={new_tsl_params}{RESET}")
                else:
                     lg.info("No risk management actions triggered requiring protection update.")

        # ==================================
        # == Scenario 2: OUT of a Position ==
        # ==================================
        else:
            lg.info(f"No open position for {symbol}. Checking for entry signal ({signal}).")

            # --- 7c. Check for Entry Signal ---
            if signal in ["BUY", "SELL"]:
                entry_color = COLOR_UP if signal == "BUY" else COLOR_DN
                lg.info(f"{entry_color}Entry Signal Detected: {signal}. Current Price: {current_price:{price_fmt}}{RESET}")

                # Check available balance
                quote_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if quote_balance is None or quote_balance <= 0:
                    lg.error(f"{COLOR_ERROR}Cannot enter {signal} for {symbol}: Invalid or zero {QUOTE_CURRENCY} balance ({quote_balance}).{RESET}")
                    return  # Stop processing for this symbol

                # Calculate initial SL/TP based on current price as entry estimate
                # Use the analyzer's method which includes rounding and validation
                _, potential_tp, potential_sl = analyzer.calculate_entry_tp_sl(current_price, signal)

                # Stop Loss is mandatory for entry based on risk calculation
                if potential_sl is None:
                    lg.error(f"{COLOR_ERROR}Cannot enter {signal} for {symbol}: Failed to calculate a valid initial Stop Loss.{RESET}")
                    return

                # Calculate position size based on risk, SL distance, balance, and market limits
                risk_per_trade = config.get("risk_per_trade", 0.01)
                position_size = calculate_position_size(
                    balance=quote_balance,
                    risk_per_trade=risk_per_trade,
                    initial_stop_loss_price=potential_sl,
                    entry_price=current_price,  # Use current price as estimate for sizing
                    market_info=market_info,
                    exchange=exchange,  # Pass exchange for formatting/precision
                    logger=lg
                )

                if position_size is None or position_size <= 0:
                    lg.error(f"{COLOR_ERROR}Cannot enter {signal} for {symbol}: Position size calculation failed or resulted in zero/negative size.{RESET}")
                    return

                # --- Execute Entry ---
                if trading_enabled:
                    lg.info(f"Attempting to enter {signal} position for {symbol}:")
                    lg.info(f"  Size = {position_size} ({analyzer.get_amount_precision()} decimals)")
                    lg.info(f"  Est. Entry = {current_price:{price_fmt}}")
                    lg.info(f"  Initial SL = {potential_sl:{price_fmt}}")
                    lg.info(f"  Initial TP = {potential_tp:{price_fmt} if potential_tp else 'None'}")

                    # --- Set Leverage (Best Effort, before placing order) ---
                    if market_info.get('is_contract', False):
                        desired_leverage = int(config.get("leverage", 10))  # Ensure integer leverage
                        try:
                            lg.info(f"Attempting to set leverage to {desired_leverage}x for {symbol}")
                            # Bybit V5 requires category and separate buy/sell leverage params
                            params = {'buyLeverage': float(desired_leverage), 'sellLeverage': float(desired_leverage)}
                            if 'bybit' in exchange.id.lower() and exchange.version == 'v5':
                                 params['category'] = 'linear' if market_info.get('is_linear', True) else 'inverse'
                            # Use set_leverage method
                            exchange.set_leverage(desired_leverage, symbol, params=params)
                            lg.info(f"Leverage set/confirmed at {desired_leverage}x for {symbol}.")
                        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                            err_str = str(e).lower()
                            # Non-fatal leverage errors (already set, positions exist, etc.)
                            if "leverage not modified" in err_str or "position open" in err_str or "same leverage" in err_str:
                                lg.warning(f"{COLOR_WARNING}Could not modify leverage to {desired_leverage}x (Reason: {e}). Proceeding with existing leverage.{RESET}")
                            else:  # Potentially more serious errors
                                lg.error(f"{COLOR_ERROR}Failed to set leverage for {symbol}: {e}. Check permissions/exchange status. May impact margin requirements.{RESET}")
                                # Decide whether to abort if leverage fails critically:
                                # return # Uncomment to abort if leverage must be set
                        except Exception as e:
                            lg.error(f"{COLOR_ERROR}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

                    # --- Place Market Order to Enter ---
                    entry_order = place_market_order(
                        exchange, symbol, signal.lower(),  # 'buy' or 'sell'
                        position_size, market_info, lg
                    )

                    # --- Set Protection After Entry ---
                    if entry_order and entry_order.get('id') and entry_order.get('status', '').lower() in ['closed', 'filled']:
                        lg.info(f"{COLOR_SUCCESS}Entry market order placed and likely filled (ID: {entry_order.get('id')}).{RESET}")
                        # Wait briefly for position to fully register on exchange before setting SL/TP
                        entry_fill_wait_s = 3  # Shorter wait after market fill confirmation
                        lg.debug(f"Waiting {entry_fill_wait_s}s for position update before setting protection...")
                        time.sleep(entry_fill_wait_s)

                        # Prepare TSL parameters if enabled
                        tsl_params_for_entry = None
                        if config.get("enable_trailing_stop", True):
                            tsl_callback_rate_str = str(config.get("trailing_stop_callback_rate", "0.005"))
                            # For Bybit V5, just setting the distance activates it. Activation price optional.
                            tsl_params_for_entry = {'trailingStop': tsl_callback_rate_str}
                            # Optional: Add activePrice if activation_percentage > 0 and needed
                            # tsl_activation_perc = safe_decimal(config.get("trailing_stop_activation_percentage", "0.003"), 0)
                            # if tsl_activation_perc > 0: ... calculate and add activePrice ...
                            lg.info(f"Prepared initial TSL params for new position: {tsl_params_for_entry}")

                        # Set initial SL, TP, and potentially TSL
                        lg.info(f"Attempting to set initial protection for new {signal} position...")
                        protection_success = set_position_protection(
                            exchange, symbol, market_info,
                            stop_loss_price=potential_sl,
                            take_profit_price=potential_tp,  # Can be None
                            trailing_stop_params=tsl_params_for_entry,  # Can be None
                            current_position=None,  # Force refetch of position state after entry
                            logger=lg
                        )
                        if protection_success:
                            lg.info(f"{COLOR_SUCCESS}Initial protection (SL/TP/TSL) set successfully for new {signal} position.{RESET}")
                        else:
                            lg.error(f"{COLOR_ERROR}Failed to set initial protection after entry for {symbol}. MANUAL CHECK REQUIRED!{RESET}")
                    elif entry_order:
                         lg.error(f"{COLOR_ERROR}Entry market order placed (ID: {entry_order.get('id')}) but status uncertain ('{entry_order.get('status')}'). Cannot reliably set protection. MANUAL CHECK REQUIRED!{RESET}")
                    else:
                        lg.error(f"{COLOR_ERROR}Entry market order failed for {signal} {symbol}. See previous logs.{RESET}")
                else:
                    # Log potential trade if trading disabled
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have entered {signal} for {symbol}.{RESET}")
                    lg.warning(f"  Size = {position_size:.8f}, SL = {potential_sl:{price_fmt}}, TP = {potential_tp:{price_fmt} if potential_tp else 'None'}")

            # --- 7d. No Signal, No Position ---
            elif signal == "HOLD":
                lg.info(f"Signal is HOLD and no open position for {symbol}. No trading action.")
            else:  # Should not happen if signal is BUY/SELL/HOLD
                 lg.error(f"Unexpected signal value '{signal}' encountered for {symbol}.")

    except ccxt.AuthenticationError as e:
        lg.critical(f"{COLOR_ERROR}CRITICAL AUTH ERROR during analysis/trade for {symbol}: {e}. Check keys/permissions. Stopping bot cycle for safety.{RESET}")
        raise  # Re-raise critical auth errors to potentially stop the main loop

    except ccxt.NetworkError as e:
         lg.error(f"{COLOR_ERROR}Network Error during analysis/trade for {symbol}: {e}. Will retry next cycle.{RESET}")
         # Allow loop to continue to next symbol/cycle

    except ccxt.ExchangeError as e:
         lg.error(f"{COLOR_ERROR}Exchange Error during analysis/trade for {symbol}: {e}.{RESET}", exc_info=True)
         # Allow loop to continue unless it's critical (like auth error handled above)

    except Exception as e:
        lg.error(f"{COLOR_ERROR}!!! Unhandled Exception in analyze_and_trade_symbol({symbol}) !!!: {e}{RESET}", exc_info=True)
        # Allow loop to continue

    finally:
        cycle_duration = time.monotonic() - cycle_start_time
        lg.info(f"---== Finished {symbol} ({cycle_duration:.2f}s) ==---")


# --- Main Execution ---
def main() -> None:
    """Initializes the bot, selects symbols, and runs the main trading loop,
    handling setup, symbol iteration, and graceful shutdown.
    """
    # Use a generic init logger for setup phase
    init_logger = setup_logger("init")  # Initial logger setup
    init_logger.info(f"{COLOR_HEADER}--- Volbot Initializing ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    init_logger.info(f"Config File: {os.path.abspath(CONFIG_FILE)}")
    init_logger.info(f"Log Directory: {os.path.abspath(LOG_DIRECTORY)}")
    init_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"Trading Enabled: {CONFIG.get('enable_trading')}")
    init_logger.info(f"Use Sandbox: {CONFIG.get('use_sandbox')}")
    init_logger.info(f"Default Interval: {CONFIG.get('interval')} ({CCXT_INTERVAL_MAP.get(CONFIG.get('interval', ''))})")
    init_logger.info(f"Timezone: {TIMEZONE.key}")
    init_logger.info(f"Console Log Level: {logging.getLevelName(console_log_level)}")

    # Initialize Exchange
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{COLOR_ERROR}Exchange initialization failed. Please check API keys, permissions, connection, and logs. Exiting.{RESET}")
        return  # Stop execution

    # --- Symbol Selection ---
    symbols_to_trade: list[str] = []
    init_logger.info(f"{COLOR_HEADER}--- Symbol Selection ---{RESET}")
    while True:
        f"Current list: {', '.join(symbols_to_trade)}" if symbols_to_trade else "Current list is empty"
        prompt = (
            f"\nEnter symbol (e.g., BTC/USDT), "
            f"'{COLOR_CYAN}all{RESET}' for active linear {QUOTE_CURRENCY} perps, "
            f"'{COLOR_YELLOW}clear{RESET}' to reset list, "
            f"or press {COLOR_GREEN}Enter{RESET} to start with current list ({len(symbols_to_trade)} symbols): "
        )
        try:
            symbol_input = input(prompt).strip().upper()  # Convert to uppercase for consistency
        except EOFError:  # Handle case where input stream is closed (e.g., running non-interactively)
             init_logger.warning("EOFError received during input. Attempting to proceed if symbols already selected.")
             if not symbols_to_trade:
                 init_logger.error("No symbols selected and input stream closed. Exiting.")
                 return
             symbol_input = ""  # Treat as pressing Enter

        if not symbol_input and symbols_to_trade:
            init_logger.info(f"Starting analysis with {len(symbols_to_trade)} selected symbols: {', '.join(symbols_to_trade)}")
            break  # Exit selection loop
        elif not symbol_input and not symbols_to_trade:
            continue
        elif symbol_input.lower() == 'clear':
             symbols_to_trade = []
             continue
        elif symbol_input.lower() == 'all':
            init_logger.info(f"Attempting to fetch all active linear {QUOTE_CURRENCY} perpetual swaps from {exchange.id}...")
            try:
                # Ensure markets are loaded/reloaded
                all_markets = exchange.load_markets(reload=True)
                linear_swaps = [
                    mkt['symbol'] for mkt in all_markets.values()
                    if mkt.get('active', True)  # Check if market is active
                       and mkt.get('linear', False)  # Check if linear contract
                       and mkt.get('swap', True)  # Check if it's a perpetual swap
                       and mkt.get('quote', '').upper() == QUOTE_CURRENCY  # Match quote currency
                ]
                if not linear_swaps:
                     init_logger.error(f"{COLOR_ERROR}No active linear {QUOTE_CURRENCY} perpetual swaps found on {exchange.id}. Please enter symbols manually.{RESET}")
                     symbols_to_trade = []  # Clear list if 'all' fails
                     continue

                symbols_to_trade = sorted(linear_swaps)  # Add all found symbols, sorted
                init_logger.info(f"Selected {len(symbols_to_trade)} active linear {QUOTE_CURRENCY} perpetual swaps.")
                preview_count = 10
                f"{', '.join(symbols_to_trade[:preview_count])}{', ...' if len(symbols_to_trade) > preview_count else ''}"
                # Ask user to confirm or continue adding
                confirm = input(f"Press {COLOR_GREEN}Enter{RESET} to start with these {len(symbols_to_trade)} symbols, or enter another symbol to add: ").strip()
                if not confirm:
                     break  # Start with the 'all' list
                else:
                     continue  # Go back to prompt to allow adding more

            except Exception as e:
                init_logger.error(f"{COLOR_ERROR}Error fetching/filtering markets for 'all': {e}. Please enter symbols manually.{RESET}")
                continue
        else:
            # Validate the entered symbol format (basic check)
            if '/' not in symbol_input:
                 continue

            # Validate the entered symbol against the exchange
            market_info = get_market_info(exchange, symbol_input, init_logger)
            if market_info:
                if symbol_input not in symbols_to_trade:
                    symbols_to_trade.append(symbol_input)
                else:
                    pass
            else:
                # get_market_info already logged the error reason
                pass

    # Final check if symbols list is empty
    if not symbols_to_trade:
        init_logger.critical(f"{COLOR_ERROR}No valid symbols selected for trading. Exiting.{RESET}")
        return

    # --- Setup Loggers for Each Selected Symbol ---
    init_logger.info(f"{COLOR_HEADER}--- Setting up Symbol Loggers ---{RESET}")
    symbol_loggers: dict[str, logging.Logger] = {}
    for sym in symbols_to_trade:
        # Setup logger with potentially updated console level from config
        # Use symbol directly as suffix (setup_logger handles sanitization)
        symbol_loggers[sym] = setup_logger(sym)
        init_logger.debug(f"Logger initialized for {sym}")

    # --- Main Trading Loop ---
    init_logger.info(f"{COLOR_HEADER}--- Starting Main Trading Loop ({len(symbols_to_trade)} symbols) ---{RESET}")
    stop_bot_flag = False  # Flag to signal graceful shutdown
    try:
        while not stop_bot_flag:
            loop_start_time = time.monotonic()
            now = datetime.now(TIMEZONE)
            init_logger.info(f"*** New Main Loop Cycle: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ***")

            # Iterate through a copy of the list in case of future modifications
            for symbol in list(symbols_to_trade):
                if stop_bot_flag: break  # Check flag before processing next symbol

                symbol_logger = symbol_loggers.get(symbol)
                if not symbol_logger:  # Should not happen, but safety check
                     init_logger.warning(f"Logger for {symbol} not found, re-initializing.")
                     symbol_logger = setup_logger(symbol)
                     symbol_loggers[symbol] = symbol_logger

                try:
                    analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except ccxt.AuthenticationError as e:
                    # Critical error, stop the bot entirely
                    symbol_logger.critical(f"{COLOR_ERROR}CRITICAL AUTH ERROR for {symbol}: {e}. Stopping bot! Check API keys/permissions immediately.{RESET}")
                    stop_bot_flag = True  # Signal loop to stop
                    break  # Exit inner symbol loop
                except Exception as sym_e:
                    # Log other errors for the specific symbol but continue the main loop
                    symbol_logger.error(f"{COLOR_ERROR}Unhandled Exception during analysis for {symbol}: {sym_e}{RESET}", exc_info=True)
                    # Optional: Add logic to temporarily disable a symbol if it errors repeatedly

                # Add a small delay between analyzing symbols to respect rate limits
                # Adjust delay based on number of symbols and exchange limits
                inter_symbol_delay = 0.2  # seconds
                time.sleep(inter_symbol_delay)

            # --- Loop Delay ---
            if stop_bot_flag: break  # Exit outer loop if flag is set

            loop_end_time = time.monotonic()
            cycle_duration = loop_end_time - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
            init_logger.info(f"*** Main Loop Cycle Completed in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s... ***")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        init_logger.info(f"\n{COLOR_WARNING}--- KeyboardInterrupt received. Initiating graceful shutdown... ---{RESET}")
        stop_bot_flag = True  # Signal shutdown
        # Optional: Add specific cleanup logic here (e.g., attempt to close open positions)
        if CONFIG.get("enable_trading", False):
             init_logger.warning("Trading was enabled. Check exchange interface for any open positions/orders.")
             # Example cleanup (USE WITH EXTREME CAUTION - VERIFY LOGIC):
             # print(f"{COLOR_YELLOW}Attempting to close open positions (if any)...{RESET}")
             # for sym in symbols_to_trade:
             #     try:
             #         pos = get_open_position(exchange, sym, init_logger)
             #         pos_size = safe_decimal(pos.get('contracts')) if pos else Decimal('0')
             #         if pos and abs(pos_size) > Decimal('1e-9'):
             #             print(f"Closing position for {sym}...")
             #             close_side = 'sell' if pos.get('side') == 'long' else 'buy'
             #             mkt_info = get_market_info(exchange, sym, init_logger)
             #             if mkt_info:
             #                  place_market_order(exchange, sym, close_side, abs(pos_size), mkt_info, init_logger, params={'reduceOnly': True})
             #             time.sleep(1) # Small delay between closures
             #     except Exception as close_err:
             #          print(f"{COLOR_ERROR}Error closing position for {sym} during shutdown: {close_err}{RESET}")

    except Exception as e:
        init_logger.critical(f"{COLOR_ERROR}--- FATAL UNHANDLED ERROR IN MAIN LOOP ---{RESET}", exc_info=True)
        init_logger.critical(f"{COLOR_ERROR}Error: {e}{RESET}")
        stop_bot_flag = True  # Ensure shutdown on fatal error
    finally:
        init_logger.info(f"{COLOR_HEADER}--- Volbot Shutting Down ---{RESET}")
        logging.shutdown()  # Ensure all log handlers flush and close properly


if __name__ == "__main__":
    # Set console log level based on config *before* setting up initial logger
    # This ensures the initial messages respect the configured level
    try:
        # Temporarily load config just to get the log level
        temp_config = load_config(CONFIG_FILE)
        config_level_name = temp_config.get("console_log_level", "INFO").upper()
        initial_console_level = getattr(logging, config_level_name, logging.INFO)
        # Update the global variable before setting up the first logger
        console_log_level = initial_console_level
        del temp_config  # Clean up temporary config load
    except Exception:
        pass
        # Keep default INFO level if config read fails early

    # Start the main application logic
    main()
