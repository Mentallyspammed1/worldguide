# volbot.py
# Incorporates Volumatic Trend + Order Block strategy into the livexy trading framework.

import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta  # Import pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize colorama and set precision
getcontext().prec = 18  # Increased precision for calculations
init(autoreset=True)
load_dotenv()

# --- Constants ---
# Neon Color Scheme (from livexy) + Strategy Colors (from volbot)
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# Volbot Colors
COLOR_UP = Fore.CYAN + Style.BRIGHT
COLOR_DN = Fore.YELLOW + Style.BRIGHT
COLOR_BULL_BOX = Fore.GREEN
COLOR_BEAR_BOX = Fore.RED
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX
COLOR_INFO = Fore.MAGENTA  # Reuse Neon Purple/Magenta
COLOR_HEADER = Fore.BLUE + Style.BRIGHT  # Reuse Neon Blue/Cyan

# API Credentials (Loaded from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# File/Directory Configuration
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Time & Retry Configuration
TIMEZONE = ZoneInfo("America/Chicago")  # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Delay between retries
LOOP_DELAY_SECONDS = 15  # Time between the end of one cycle and the start of the next

# Interval Configuration
VALID_INTERVALS = [
    "1",
    "3",
    "5",
    "15",
    "30",
    "60",
    "120",
    "240",
    "D",
    "W",
    "M",
]  # Intervals supported by the bot's logic
CCXT_INTERVAL_MAP = {  # Map our intervals to ccxt's expected format
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}

# API Error Codes for Retry Logic
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]  # HTTP status codes considered retryable

# --- Default Volbot Strategy Parameters (overridden by config.json) ---
DEFAULT_VOLBOT_LENGTH = 40
DEFAULT_VOLBOT_ATR_LENGTH = 200  # ATR period used for Volbot strategy levels
DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK = 1000  # Lookback for normalizing volume
DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE = 100  # Use 100th percentile (max) for normalization
DEFAULT_VOLBOT_OB_SOURCE = "Wicks"  # "Wicks" or "Bodys" - determines candle parts for OB calculation
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H = 25  # Left lookback for pivot high
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H = 25  # Right lookback for pivot high
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L = 25  # Left lookback for pivot low
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L = 25  # Right lookback for pivot low
DEFAULT_VOLBOT_MAX_BOXES = 50  # Max number of active Bull/Bear boxes to track

# Default Risk Management Parameters (from livexy, overridden by config.json)
DEFAULT_ATR_PERIOD = 14  # Default ATR period for SL/TP/BE calculations (Risk Management ATR)

# QUOTE_CURRENCY is dynamically loaded from config after it's read

console_log_level = logging.INFO  # Default console level, can be changed in main()


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that redacts sensitive information like API keys from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting API keys."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def setup_logger(name_suffix: str) -> logging.Logger:
    """Sets up a logger instance with specified suffix, file rotation, and console output.

    Args:
        name_suffix: A string suffix to append to the logger name and filename (e.g., symbol or 'init').

    Returns:
        The configured logging.Logger instance.
    """
    safe_suffix = re.sub(r"[^\w\-]+", "_", name_suffix)  # Make suffix filesystem-safe
    logger_name = f"volbot_{safe_suffix}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # If logger already exists with handlers, just ensure console level is correct
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

    logger.setLevel(logging.DEBUG)  # Set root level to DEBUG to allow handlers to filter

    # File Handler (DEBUG level)
    try:
        file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        pass

    # Console Handler (INFO level by default)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Use local time format
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent messages from reaching the root logger
    return logger


# --- Configuration Loading ---
def load_config(filepath: str) -> dict[str, Any]:
    """Loads configuration from a JSON file. Creates a default config if the file
    doesn't exist. Ensures all default keys are present in the loaded config,
    adding missing ones with default values and updating the file if necessary.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        A dictionary containing the configuration settings.
    """
    default_config = {
        # --- General Bot Settings ---
        "interval": "5",  # Default trading interval (string format for internal logic)
        "retry_delay": RETRY_DELAY_SECONDS,  # API retry delay
        "enable_trading": False,  # SAFETY FIRST: Master switch for placing orders. Default: False.
        "use_sandbox": True,  # SAFETY FIRST: Use exchange's testnet/sandbox. Default: True.
        "risk_per_trade": 0.01,  # Percentage of account balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 10,  # Desired leverage for contract trading (ignored for spot)
        "max_concurrent_positions": 1,  # Max open positions allowed for this symbol by the bot (usually 1 for single strategy)
        "quote_currency": "USDT",  # Currency for balance checking and position sizing (e.g., USDT, USD)
        # --- Volbot Strategy Settings ---
        "volbot_enabled": True,  # Enable/disable the Volbot strategy logic calculations and signals
        "volbot_length": DEFAULT_VOLBOT_LENGTH,  # Main period for Volbot EMAs
        "volbot_atr_length": DEFAULT_VOLBOT_ATR_LENGTH,  # ATR period for calculating Volbot dynamic levels
        "volbot_volume_percentile_lookback": DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK,  # Lookback period for volume normalization
        "volbot_volume_normalization_percentile": DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE,  # Percentile used for volume normalization (usually 100=max)
        "volbot_ob_source": DEFAULT_VOLBOT_OB_SOURCE,  # Source for Order Block detection: "Wicks" or "Bodys"
        "volbot_pivot_left_len_h": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H,  # Left bars to check for Pivot High
        "volbot_pivot_right_len_h": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H,  # Right bars to check for Pivot High
        "volbot_pivot_left_len_l": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L,  # Left bars to check for Pivot Low
        "volbot_pivot_right_len_l": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L,  # Right bars to check for Pivot Low
        "volbot_max_boxes": DEFAULT_VOLBOT_MAX_BOXES,  # Maximum number of active (unmitigated) Order Blocks to track
        "volbot_signal_on_trend_flip": True,  # Generate BUY/SELL signal immediately when Volbot trend direction changes
        "volbot_signal_on_ob_entry": True,  # Generate BUY/SELL signal when price enters an Order Block matching the current trend
        # --- Risk Management Settings ---
        "atr_period": DEFAULT_ATR_PERIOD,  # ATR period used for calculating Stop Loss, Take Profit, and Break Even triggers (separate from strategy ATR)
        "stop_loss_multiple": 1.8,  # ATR multiple for initial Stop Loss calculation (used for position sizing)
        "take_profit_multiple": 0.7,  # ATR multiple for initial Take Profit calculation
        # --- Trailing Stop Loss Config (Exchange-based TSL) ---
        "enable_trailing_stop": True,  # Enable setting an exchange-based Trailing Stop Loss
        "trailing_stop_callback_rate": 0.005,  # Trail distance as a percentage of entry/activation price (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003,  # Profit percentage required to activate the TSL (e.g., 0.003 = 0.3%). Use 0 for immediate activation.
        # --- Break-Even Stop Config ---
        "enable_break_even": True,  # Enable moving Stop Loss to break-even + offset
        "break_even_trigger_atr_multiple": 1.0,  # ATR multiple of profit required to trigger break-even (e.g., 1.0 = profit equals 1x Risk ATR)
        "break_even_offset_ticks": 2,  # Number of minimum price ticks to offset the break-even SL from entry price
        # --- Livexy Legacy/Optional Settings (can be removed if not used) ---
        # These are likely remnants from the original livexy framework and not used by the core Volbot logic.
        # "signal_score_threshold": 1.5,
        # "indicators": { ... },
        # "weight_sets": { ... },
        # "active_weight_set": "default",
    }

    # Create default config if file doesn't exist
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            return default_config
        except OSError:
            # Return default config anyway if creation fails
            return default_config

    # Load config from file and ensure all keys are present
    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist, add missing ones
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # If the config was updated, write it back to the file
            if updated_config != config_from_file:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                except OSError:
                    pass
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError):
        # Attempt to recreate default config if loading failed
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
        except OSError:
            pass
        return default_config


def _ensure_config_keys(loaded_config: dict[str, Any], default_config: dict[str, Any]) -> dict[str, Any]:
    """Recursively ensures that all keys from the default_config exist in the loaded_config.
    If a key is missing in loaded_config, it's added with the value from default_config.

    Args:
        loaded_config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.

    Returns:
        The updated configuration dictionary with all default keys present.
    """
    updated_config = loaded_config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        # If the value is a dictionary, recurse
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config


# Load configuration globally
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")  # Get quote currency from config


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with API keys, rate limiting,
    sandbox mode handling, and basic connectivity tests.

    Args:
        logger: The logger instance to use for logging messages.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    try:
        exchange_options = {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,  # Let ccxt handle basic rate limiting
            "options": {
                "defaultType": "linear",  # Default to linear contracts for Bybit
                "adjustForTimeDifference": True,  # Auto-sync time with server
                # Timeouts for various operations (in milliseconds)
                "fetchTickerTimeout": 10000,
                "fetchBalanceTimeout": 15000,
                "createOrderTimeout": 20000,
            },
        }

        exchange_class = ccxt.bybit  # Explicitly use bybit
        exchange = exchange_class(exchange_options)

        # Enable Sandbox Mode if configured
        if CONFIG.get("use_sandbox", True):  # Default to sandbox if not specified
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}USING LIVE TRADING ENVIRONMENT{RESET}")

        # Load markets to ensure symbol information is available
        lg.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        lg.info(f"Markets loaded. CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Perform an initial balance fetch to test API key validity and permissions
        account_type_to_test = "CONTRACT"  # Bybit V5 default for derivatives
        lg.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            # Try fetching CONTRACT balance first
            balance = exchange.fetch_balance(params={"type": account_type_to_test})
            available_quote = balance.get(QUOTE_CURRENCY, {}).get("free", "N/A")
            lg.info(
                f"{NEON_GREEN}Successfully connected and fetched initial balance ({account_type_to_test}).{RESET} "
                f"(Example: {QUOTE_CURRENCY} available: {available_quote})"
            )
        except ccxt.AuthenticationError as auth_err:
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            lg.error(
                f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), "
                f"match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}"
            )
            return None
        except ccxt.ExchangeError as balance_err:
            lg.warning(
                f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. "
                f"Trying default fetch...{RESET}"
            )
            # Fallback to default fetch_balance if specific type fails
            try:
                balance = exchange.fetch_balance()
                available_quote = balance.get(QUOTE_CURRENCY, {}).get("free", "N/A")
                lg.info(
                    f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} "
                    f"(Example: {QUOTE_CURRENCY} available: {available_quote})"
                )
            except Exception as fallback_err:
                lg.warning(
                    f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. "
                    f"Continuing, but check API permissions/account type if trading fails.{RESET}"
                )
        except Exception as balance_err:
            # Catch other potential errors during balance fetch
            lg.warning(
                f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. "
                f"Continuing, but check API permissions/account type if trading fails.{RESET}"
            )

        return exchange

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
    return None


# --- CCXT Data Fetching ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the current market price for a symbol using the exchange's ticker.
    Uses fallbacks (last, bid/ask midpoint, ask, bid) to find a valid price.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: The logger instance.

    Returns:
        The current price as a Decimal, or None if fetching fails or price is invalid.
    """
    lg = logger
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        lg.debug(f"Ticker data received for {symbol}: {ticker}")

        price: Decimal | None = None

        # Prioritize 'last' price if available and valid
        last_price = ticker.get("last")
        if last_price is not None:
            try:
                last_decimal = Decimal(str(last_price))
                if last_decimal > 0:
                    price = last_decimal
                    lg.debug(f"Using 'last' price: {price}")
            except Exception:
                lg.warning(f"Could not parse 'last' price '{last_price}' as Decimal.")

        # Fallback to bid/ask midpoint if 'last' is invalid/missing
        if price is None:
            bid_price = ticker.get("bid")
            ask_price = ticker.get("ask")
            if bid_price is not None and ask_price is not None:
                try:
                    bid_decimal = Decimal(str(bid_price))
                    ask_decimal = Decimal(str(ask_price))
                    if bid_decimal > 0 and ask_decimal > 0 and bid_decimal <= ask_decimal:
                        price = (bid_decimal + ask_decimal) / Decimal("2")
                        lg.debug(f"Using bid/ask midpoint: {price}")
                    elif ask_decimal > 0:
                        price = ask_decimal  # Fallback to ask if bid is invalid
                        lg.debug(f"Using 'ask' price (midpoint fallback): {price}")
                except Exception:
                    lg.warning(f"Could not parse bid '{bid_price}' or ask '{ask_price}' for midpoint.")

        # Fallback to 'ask' price if midpoint failed
        if price is None and ask_price is not None:
            try:
                ask_decimal = Decimal(str(ask_price))
                if ask_decimal > 0:
                    price = ask_decimal
                    lg.debug(f"Using 'ask' price (direct fallback): {price}")
            except Exception:
                lg.warning(f"Could not parse 'ask' price '{ask_price}' as Decimal.")

        # Fallback to 'bid' price as last resort
        if price is None and bid_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                if bid_decimal > 0:
                    price = bid_decimal
                    lg.debug(f"Using 'bid' price (last resort): {price}")
            except Exception:
                lg.warning(f"Could not parse 'bid' price '{bid_price}' as Decimal.")

        # Final validation
        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to fetch a valid positive current price for {symbol} from ticker.{RESET}")
            return None

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None


def fetch_klines_ccxt(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT's fetch_ohlcv method with retries
    for network issues and rate limits, basic data validation, and conversion to a pandas DataFrame.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        timeframe: The CCXT timeframe string (e.g., '1m', '5m', '1h', '1d').
        limit: The maximum number of klines to fetch.
        logger: The logger instance. Defaults to root logger if None.

    Returns:
        A pandas DataFrame containing the OHLCV data, indexed by timestamp,
        or an empty DataFrame if fetching or processing fails.
    """
    lg = logger or logging.getLogger(__name__)
    try:
        if not exchange.has["fetchOHLCV"]:
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        ohlcv: list[list[Any]] | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(
                    f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})"
                )
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

                # Check if data was received
                if ohlcv is not None and len(ohlcv) > 0:
                    lg.debug(f"Received {len(ohlcv)} klines from API.")
                    break  # Success
                else:
                    lg.warning(f"fetch_ohlcv returned empty list for {symbol} (Attempt {attempt + 1}). Retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)  # Short delay before retry

            except (
                ccxt.NetworkError,
                ccxt.RequestTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(
                        f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. "
                        f"Retrying in {RETRY_DELAY_SECONDS}s..."
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"Max retries exceeded for network error fetching klines: {e}")
                    raise e  # Raise after max retries
            except ccxt.RateLimitExceeded as e:
                # Extract suggested wait time from error message if possible, otherwise use a longer delay
                wait_time_match = re.search(r"(\d+)\s*seconds", str(e))
                wait_time = int(wait_time_match.group(1)) if wait_time_match else RETRY_DELAY_SECONDS * 5
                lg.warning(
                    f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})"
                )
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                lg.error(f"Exchange error during fetch_ohlcv for {symbol}: {e}")
                raise e  # Non-retryable exchange error

        # If no data after retries
        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after all retries.{RESET}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df  # Return empty if conversion failed

        # --- Data Cleaning and Processing ---
        # Convert timestamp to datetime objects and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)  # Drop rows where timestamp conversion failed
        df.set_index("timestamp", inplace=True)

        # Convert OHLCV columns to numeric, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN in essential price columns or zero close price
        initial_len = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df = df[df["close"] > 0]  # Ensure close price is positive
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return pd.DataFrame()

        # Ensure data is sorted chronologically
        df.sort_index(inplace=True)

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching klines for {symbol} after retries: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error processing klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()


# --- Volbot Strategy Calculation Functions ---


def ema_swma(series: pd.Series, length: int, logger: logging.Logger) -> pd.Series:
    """Calculates a Smoothed Weighted Moving Average (SWMA), which is an EMA applied
    to a weighted average of the last 4 values of the input series.
    This specific weighting (1/6, 2/6, 2/6, 1/6) aims to provide smoothing.
    Uses pandas_ta.ema for the final EMA calculation.

    Args:
        series: The pandas Series to calculate the SWMA on (e.g., 'close' prices).
        length: The length parameter for the final EMA calculation.
        logger: The logger instance.

    Returns:
        A pandas Series containing the calculated SWMA values, aligned with the input series index.
    """
    lg = logger
    lg.debug(f"Calculating Smoothed Weighted EMA (SWMA) with length: {length}...")
    if len(series) < 4:
        lg.warning(f"Series length ({len(series)}) is less than 4. SWMA requires 4 periods. Returning standard EMA.")
        # Fallback to standard EMA if series is too short
        return ta.ema(series, length=length, adjust=False)  # Use adjust=False for TV-like behavior

    # Calculate the weighted average of the last 4 points
    # Weights: Current(1/6), Prev1(2/6), Prev2(2/6), Prev3(1/6)
    weighted_series = (
        series.shift(3) / 6 + series.shift(2) * 2 / 6 + series.shift(1) * 2 / 6 + series / 6
    )  # Corrected: Use current series value with weight 1/6

    # Calculate EMA on the weighted series
    # Use adjust=False for behavior closer to TradingView's EMA calculation (less weight on early points)
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length, adjust=False)

    # Reindex to match the original series index, filling potential NaNs at the start
    return smoothed_ema.reindex(series.index)


def calculate_volatility_levels(df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Calculates the core Volumatic Trend indicators, including EMAs, ATR,
    dynamic volatility-based levels, normalized volume, and cumulative volume metrics.

    Args:
        df: The DataFrame containing OHLCV data.
        config: The configuration dictionary.
        logger: The logger instance.

    Returns:
        The DataFrame with added Volbot strategy columns.
    """
    lg = logger
    lg.info("Calculating Volumatic Trend Levels...")
    length = config.get("volbot_length", DEFAULT_VOLBOT_LENGTH)
    atr_length = config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH)
    volume_percentile_lookback = config.get(
        "volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK
    )
    # volume_normalization_percentile = config.get("volbot_volume_normalization_percentile", DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE) # Hardcoded to 100 (max) for simplicity

    # Calculate Strategy EMAs and ATR
    # ema1_strat: Smoothed Weighted MA (specific Volbot smoothing)
    df["ema1_strat"] = ema_swma(df["close"], length, lg)
    # ema2_strat: Standard EMA for trend comparison
    df["ema2_strat"] = ta.ema(df["close"], length=length, adjust=False)
    # atr_strat: ATR used specifically for calculating the dynamic Volbot levels
    df["atr_strat"] = ta.atr(df["high"], df["low"], df["close"], length=atr_length)

    # Determine Trend Direction and Changes
    # Trend is considered UP if the smoothed EMA (ema1) crosses *above* the standard EMA (ema2)
    df["trend_up_strat"] = df["ema1_strat"] > df["ema2_strat"]  # Corrected logic: ema1 > ema2 for uptrend
    df["trend_changed_strat"] = df["trend_up_strat"] != df["trend_up_strat"].shift(1)

    # Initialize level columns with NaN
    level_cols = [
        "upper_strat",
        "lower_strat",
        "lower_vol_strat",
        "upper_vol_strat",
        "step_up_strat",
        "step_dn_strat",
        "last_trend_change_index_strat",
    ]
    for col in level_cols:
        df[col] = np.nan

    # --- Calculate Dynamic Levels Based on Trend Changes ---
    # These levels are recalculated only when the trend flips.
    last_trend_change_idx = 0
    for i in range(1, len(df)):
        current_index = df.index[i]
        previous_index = df.index[i - 1]

        if df.loc[current_index, "trend_changed_strat"]:
            # Use values from the *previous* bar where the trend was stable
            ema1_val = df.loc[previous_index, "ema1_strat"]
            atr_val = df.loc[previous_index, "atr_strat"]

            if pd.notna(ema1_val) and pd.notna(atr_val) and atr_val > 0:
                # Calculate base levels based on EMA1 +/- 3*ATR at the time of the flip
                upper = ema1_val + atr_val * 3
                lower = ema1_val - atr_val * 3
                # Calculate intermediate volatility levels (adjust width based on ATR)
                lower_vol = lower + atr_val * 4  # Top of the lower volatility zone
                upper_vol = upper - atr_val * 4  # Bottom of the upper volatility zone

                # Calculate step sizes for volume influence (scaled by 100 for percentage volume)
                step_up = (lower_vol - lower) / 100 if lower_vol > lower else 0
                step_dn = (upper - upper_vol) / 100 if upper > upper_vol else 0

                # Assign the newly calculated levels from this bar onwards until the next change
                df.loc[current_index:, "upper_strat"] = upper
                df.loc[current_index:, "lower_strat"] = lower
                df.loc[current_index:, "lower_vol_strat"] = lower_vol
                df.loc[current_index:, "upper_vol_strat"] = upper_vol
                df.loc[current_index:, "step_up_strat"] = step_up
                df.loc[current_index:, "step_dn_strat"] = step_dn

                last_trend_change_idx = i  # Store the numerical index of the change
                df.loc[current_index:, "last_trend_change_index_strat"] = last_trend_change_idx
            else:
                # If EMA or ATR is NaN at trend change, propagate previous levels
                lg.debug(f"NaN encountered for EMA/ATR at trend change index {i}. Propagating previous levels.")
                df.loc[current_index, level_cols] = df.loc[previous_index, level_cols]
                # Also propagate the last known change index
                df.loc[current_index, "last_trend_change_index_strat"] = df.loc[
                    previous_index, "last_trend_change_index_strat"
                ]

        elif i > 0:
            # If trend didn't change, propagate the levels from the previous bar
            df.loc[current_index, level_cols] = df.loc[previous_index, level_cols]

    # --- Calculate Volume Metrics ---
    # Normalized Volume (as percentage of max volume in lookback period)
    df["percentile_vol_strat"] = df["volume"].rolling(window=volume_percentile_lookback, min_periods=1).max()
    df["vol_norm_strat"] = np.where(
        df["percentile_vol_strat"] > 0,
        (df["volume"] / df["percentile_vol_strat"] * 100),  # Scale volume 0-100 based on max in lookback
        0,  # Avoid division by zero
    ).fillna(0)

    # Calculate volume-adjusted steps
    df["vol_up_step_strat"] = df["step_up_strat"] * df["vol_norm_strat"]
    df["vol_dn_step_strat"] = df["step_dn_strat"] * df["vol_norm_strat"]

    # Calculate final volume-adjusted trend levels
    # Uptrend: Lower base level + volume-adjusted upward step
    df["vol_trend_up_level_strat"] = df["lower_strat"] + df["vol_up_step_strat"]
    # Downtrend: Upper base level - volume-adjusted downward step
    df["vol_trend_dn_level_strat"] = df["upper_strat"] - df["vol_dn_step_strat"]

    # --- Calculate Cumulative Volume Since Last Trend Change ---
    # Volume Delta: Positive if close > open, Negative if close < open
    df["volume_delta_strat"] = np.where(
        df["close"] > df["open"], df["volume"], np.where(df["close"] < df["open"], -df["volume"], 0)
    )  # Zero delta for dojis
    df["volume_total_strat"] = df["volume"]

    # Group by trend blocks (identified by cumulative sum of trend changes)
    trend_block_group = df["trend_changed_strat"].cumsum()
    # Calculate cumulative volume delta and total volume within each trend block
    df["cum_vol_delta_since_change_strat"] = df.groupby(trend_block_group)["volume_delta_strat"].cumsum()
    df["cum_vol_total_since_change_strat"] = df.groupby(trend_block_group)["volume_total_strat"].cumsum()

    lg.info("Volumatic Trend Levels calculation complete.")
    return df


def calculate_pivot_order_blocks(df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Identifies Pivot High (PH) and Pivot Low (PL) points based on surrounding bars,
    which are used as the basis for detecting Order Blocks.

    Args:
        df: The DataFrame containing OHLCV data.
        config: The configuration dictionary.
        logger: The logger instance.

    Returns:
        The DataFrame with added 'ph_strat' (Pivot High price) and 'pl_strat' (Pivot Low price) columns.
        Values are NaN if no pivot occurs at that bar.
    """
    lg = logger
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    left_h = config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    left_l = config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    lg.info(
        f"Calculating Pivot Points for Order Blocks (Source: {source}, Left/Right H: {left_h}/{right_h}, L: {left_l}/{right_l})..."
    )

    # Select the price series based on the 'source' config (Wicks or Bodys)
    # Pivot High uses High (Wicks) or Close (Bodys)
    high_col = df["high"] if source == "Wicks" else df["close"]
    # Pivot Low uses Low (Wicks) or Open (Bodys - as per original volbot interpretation)
    low_col = df["low"] if source == "Wicks" else df["open"]

    df["ph_strat"] = np.nan  # Initialize Pivot High column
    df["pl_strat"] = np.nan  # Initialize Pivot Low column

    # --- Calculate Pivot Highs (PH) ---
    # A bar is a Pivot High if its 'high_col' is strictly higher than all 'high_col'
    # values within 'left_h' bars to the left and 'right_h' bars to the right.
    # Note: Original Pine script uses '>' for left and '>=' for right check. Replicating that.
    # Correction: Let's use >= for left and > for right to match common definitions more closely
    # and avoid issues with consecutive equal highs.
    for i in range(left_h, len(df) - right_h):
        is_pivot_high = True
        pivot_high_val = high_col.iloc[i]

        # Check left side: Current high must be >= all left highs
        for j in range(1, left_h + 1):
            if high_col.iloc[i - j] > pivot_high_val:  # Changed from >= to >
                is_pivot_high = False
                break
        if not is_pivot_high:
            continue

        # Check right side: Current high must be > all right highs
        for j in range(1, right_h + 1):
            if high_col.iloc[i + j] >= pivot_high_val:  # Changed from > to >=
                is_pivot_high = False
                break

        if is_pivot_high:
            # Store the high price at the pivot point index 'i'
            df.loc[df.index[i], "ph_strat"] = pivot_high_val

    # --- Calculate Pivot Lows (PL) ---
    # A bar is a Pivot Low if its 'low_col' is strictly lower than all 'low_col'
    # values within 'left_l' bars to the left and 'right_l' bars to the right.
    # Using <= for left check and < for right check.
    for i in range(left_l, len(df) - right_l):
        is_pivot_low = True
        pivot_low_val = low_col.iloc[i]

        # Check left side: Current low must be <= all left lows
        for j in range(1, left_l + 1):
            if low_col.iloc[i - j] < pivot_low_val:  # Changed from <= to <
                is_pivot_low = False
                break
        if not is_pivot_low:
            continue

        # Check right side: Current low must be < all right lows
        for j in range(1, right_l + 1):
            if low_col.iloc[i + j] <= pivot_low_val:  # Changed from < to <=
                is_pivot_low = False
                break

        if is_pivot_low:
            # Store the low price at the pivot point index 'i'
            df.loc[df.index[i], "pl_strat"] = pivot_low_val

    lg.info("Pivot Point calculation complete.")
    return df


def manage_order_blocks(
    df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Identifies, creates, and manages the state of Order Block (OB) boxes based on
    detected pivot points and subsequent price action. Tracks active and closed boxes.

    Order Block Definition Used Here:
    - Bearish OB: Forms after a Pivot High (PH). The OB candle is `right_h` bars *before* the PH bar.
      - Source "Wicks": Top = High, Bottom = Close of OB candle.
      - Source "Bodys": Top = Close, Bottom = Open of OB candle.
    - Bullish OB: Forms after a Pivot Low (PL). The OB candle is `right_l` bars *before* the PL bar.
      - Source "Wicks": Top = Open, Bottom = Low of OB candle.
      - Source "Bodys": Top = Close, Bottom = Open of OB candle.
    - State Management:
      - 'active': The OB is potentially valid.
      - 'closed': Price has moved beyond the OB, invalidating it (e.g., close > Bear OB top, close < Bull OB bottom).

    Args:
        df: DataFrame with OHLCV data and calculated pivot columns ('ph_strat', 'pl_strat').
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Tuple containing:
        - df: The input DataFrame with added columns ('active_bull_ob_strat', 'active_bear_ob_strat')
              referencing the active OB dictionary if price is currently within one.
        - bull_boxes: A list of all identified bullish OB dictionaries.
        - bear_boxes: A list of all identified bearish OB dictionaries.
    """
    lg = logger
    lg.info("Managing Order Block Boxes...")
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)  # Bars from PH back to OB candle
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)  # Bars from PL back to OB candle
    max_boxes = config.get("volbot_max_boxes", DEFAULT_VOLBOT_MAX_BOXES)

    # Lists to store OB dictionaries
    bull_boxes: list[dict] = []  # Stores active bullish OBs: [ {id, type, start_idx, end_idx, top, bottom, state} ]
    bear_boxes: list[dict] = []  # Stores active bearish OBs: [ {id, type, start_idx, end_idx, top, bottom, state} ]
    box_counter = 0  # Simple counter for unique OB IDs

    # Add columns to DataFrame to store a *reference* to the active OB dictionary for the current bar
    # Using 'object' dtype allows storing dictionaries or None
    df["active_bull_ob_strat"] = pd.Series(dtype="object")
    df["active_bear_ob_strat"] = pd.Series(dtype="object")  # Changed from pd.series to pd.Series

    # Iterate through each bar to create and manage OBs
    for i in range(len(df)):
        current_idx = df.index[i]  # Timestamp index
        current_close = df.loc[current_idx, "close"]

        # --- Create New Bearish Order Blocks (Based on Pivot Highs) ---
        # A PH is confirmed at index 'i'. The OB candle is 'right_h' bars *before* 'i'.
        if pd.notna(df.loc[current_idx, "ph_strat"]):
            ob_candle_iloc = i - right_h  # Integer location of the OB candle
            if ob_candle_iloc >= 0:
                ob_candle_idx = df.index[ob_candle_iloc]  # Timestamp index of the OB candle
                top_price, bottom_price = np.nan, np.nan

                # Determine OB boundaries based on source
                if source == "Bodys":
                    # Bearish Body OB: Close to Open of the OB candle
                    top_price = df.loc[ob_candle_idx, "close"]
                    bottom_price = df.loc[ob_candle_idx, "open"]
                else:  # Wicks (Default)
                    # Bearish Wick OB: High to Close of the OB candle
                    top_price = df.loc[ob_candle_idx, "high"]
                    bottom_price = df.loc[ob_candle_idx, "close"]

                # Ensure valid prices and correct order (top > bottom)
                if pd.notna(top_price) and pd.notna(bottom_price):
                    if bottom_price > top_price:
                        top_price, bottom_price = bottom_price, top_price  # Swap if needed
                    box_counter += 1
                    new_box = {
                        "id": f"BearOB_{box_counter}",
                        "type": "bear",
                        "start_idx": ob_candle_idx,  # Timestamp when OB candle formed
                        "pivot_idx": current_idx,  # Timestamp when pivot confirmed OB
                        "end_idx": current_idx,  # Initially ends at pivot confirmation bar
                        "top": top_price,
                        "bottom": bottom_price,
                        "state": "active",  # Starts as active
                    }
                    bear_boxes.append(new_box)
                    lg.debug(
                        f"Created Bearish OB: {new_box['id']} (Pivot: {current_idx}, Candle: {ob_candle_idx}, "
                        f"Range: {bottom_price:.5f}-{top_price:.5f})"
                    )

        # --- Create New Bullish Order Blocks (Based on Pivot Lows) ---
        # A PL is confirmed at index 'i'. The OB candle is 'right_l' bars *before* 'i'.
        if pd.notna(df.loc[current_idx, "pl_strat"]):
            ob_candle_iloc = i - right_l  # Integer location of the OB candle
            if ob_candle_iloc >= 0:
                ob_candle_idx = df.index[ob_candle_iloc]  # Timestamp index of the OB candle
                top_price, bottom_price = np.nan, np.nan

                # Determine OB boundaries based on source
                if source == "Bodys":
                    # Bullish Body OB: Close to Open of the OB candle
                    top_price = df.loc[ob_candle_idx, "close"]
                    bottom_price = df.loc[ob_candle_idx, "open"]
                else:  # Wicks (Default)
                    # Bullish Wick OB: Open to Low of the OB candle
                    top_price = df.loc[ob_candle_idx, "open"]
                    bottom_price = df.loc[ob_candle_idx, "low"]

                # Ensure valid prices and correct order (top > bottom)
                if pd.notna(top_price) and pd.notna(bottom_price):
                    if bottom_price > top_price:
                        top_price, bottom_price = bottom_price, top_price  # Swap if needed
                    box_counter += 1
                    new_box = {
                        "id": f"BullOB_{box_counter}",
                        "type": "bull",
                        "start_idx": ob_candle_idx,  # Timestamp when OB candle formed
                        "pivot_idx": current_idx,  # Timestamp when pivot confirmed OB
                        "end_idx": current_idx,  # Initially ends at pivot confirmation bar
                        "top": top_price,
                        "bottom": bottom_price,
                        "state": "active",  # Starts as active
                    }
                    bull_boxes.append(new_box)
                    lg.debug(
                        f"Created Bullish OB: {new_box['id']} (Pivot: {current_idx}, Candle: {ob_candle_idx}, "
                        f"Range: {bottom_price:.5f}-{top_price:.5f})"
                    )

        # --- Manage Existing Boxes State & Identify Active OB for Current Bar ---
        active_bull_ref = None
        for box in bull_boxes:
            if box["state"] == "active":
                # Check if the current bar's close invalidates (mitigates) the Bull OB
                if current_close < box["bottom"]:
                    box["state"] = "closed"
                    box["end_idx"] = current_idx  # Mark the closing bar
                    lg.debug(f"Closed Bullish OB: {box['id']} (Close {current_close:.5f} < Bottom {box['bottom']:.5f})")
                else:
                    # If still active, update its end_idx to the current bar
                    box["end_idx"] = current_idx
                    # Check if the current close is *inside* this active Bull OB
                    if box["bottom"] <= current_close <= box["top"]:
                        active_bull_ref = box  # Store reference to the box dictionary

        active_bear_ref = None
        for box in bear_boxes:
            if box["state"] == "active":
                # Check if the current bar's close invalidates (mitigates) the Bear OB
                if current_close > box["top"]:
                    box["state"] = "closed"
                    box["end_idx"] = current_idx  # Mark the closing bar
                    lg.debug(f"Closed Bearish OB: {box['id']} (Close {current_close:.5f} > Top {box['top']:.5f})")
                else:
                    # If still active, update its end_idx to the current bar
                    box["end_idx"] = current_idx
                    # Check if the current close is *inside* this active Bear OB
                    if box["bottom"] <= current_close <= box["top"]:
                        active_bear_ref = box  # Store reference to the box dictionary

        # Store the reference to the active OB (or None) in the DataFrame for this bar
        # Using .at for direct assignment by index label (timestamp)
        df.at[current_idx, "active_bull_ob_strat"] = active_bull_ref
        df.at[current_idx, "active_bear_ob_strat"] = active_bear_ref

        # --- Clean Up Old Active Boxes (Limit Memory Usage) ---
        # Filter to get currently active boxes
        active_bull_boxes = [b for b in bull_boxes if b["state"] == "active"]
        if len(active_bull_boxes) > max_boxes:
            # Identify the oldest *active* boxes to remove conceptually
            num_to_remove = len(active_bull_boxes) - max_boxes
            # Sort active boxes by their start timestamp (ascending)
            active_bull_boxes.sort(key=lambda b: b["start_idx"])
            # Get IDs of the oldest active boxes to remove
            ids_to_remove = {b["id"] for b in active_bull_boxes[:num_to_remove]}
            # Rebuild the main list, keeping closed boxes and newer active boxes
            bull_boxes = [b for b in bull_boxes if b["state"] == "closed" or b["id"] not in ids_to_remove]
            lg.debug(f"Removed {num_to_remove} oldest active Bull OBs to maintain max count ({max_boxes}).")

        active_bear_boxes = [b for b in bear_boxes if b["state"] == "active"]
        if len(active_bear_boxes) > max_boxes:
            num_to_remove = len(active_bear_boxes) - max_boxes
            active_bear_boxes.sort(key=lambda b: b["start_idx"])
            ids_to_remove = {b["id"] for b in active_bear_boxes[:num_to_remove]}
            bear_boxes = [b for b in bear_boxes if b["state"] == "closed" or b["id"] not in ids_to_remove]
            lg.debug(f"Removed {num_to_remove} oldest active Bear OBs to maintain max count ({max_boxes}).")

    lg.info(f"Order Block management complete. Total Bull OBs: {len(bull_boxes)}, Bear OBs: {len(bear_boxes)}")
    # Return df (with active OB refs) and the complete lists of all OBs found
    return df, bull_boxes, bear_boxes


# --- Trading Analyzer Class (Modified for Volbot Strategy) ---
class TradingAnalyzer:
    """Analyzes trading data using the Volbot strategy and generates trading signals.
    Also calculates risk management metrics (ATR, SL/TP levels).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],
    ) -> None:
        """Initializes the TradingAnalyzer.

        Args:
            df: Raw OHLCV DataFrame fetched from the exchange.
            logger: Logger instance.
            config: Configuration dictionary.
            market_info: Market information dictionary from ccxt.
        """
        self.df_raw = df  # Store raw klines
        self.df_processed = pd.DataFrame()  # Store df after strategy calculations
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")

        # Stores latest relevant state values from the strategy calculations
        self.strategy_state: dict[str, Any] = {}
        # Stores references to the latest active OB dictionaries (or None)
        self.latest_active_bull_ob: dict | None = None
        self.latest_active_bear_ob: dict | None = None
        # Keep track of all boxes found during the calculation (for potential future analysis/plotting)
        self.all_bull_boxes: list[dict] = []
        self.all_bear_boxes: list[dict] = []

        # Calculate indicators and update state immediately on initialization
        self._calculate_strategy_indicators()
        self._update_latest_strategy_state()

    def _calculate_strategy_indicators(self) -> None:
        """Calculates all necessary indicators: Risk Management ATR and Volbot strategy indicators.
        Stores the results in self.df_processed.
        """
        if self.df_raw.empty:
            self.logger.warning(
                f"{NEON_YELLOW}Raw DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}"
            )
            return

        # Determine minimum required data length based on longest lookback periods
        strat_lookbacks = [
            self.config.get("volbot_length", DEFAULT_VOLBOT_LENGTH),
            self.config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
            self.config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
            # Pivot lookbacks require left + right + 1 candle
            self.config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
            + self.config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
            + 1,
            self.config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
            + self.config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
            + 1,
        ]
        risk_lookback = self.config.get("atr_period", DEFAULT_ATR_PERIOD)  # For SL/TP ATR
        # Add a buffer (e.g., 50) for initial NaNs and smoother calculations
        min_required_data = max(strat_lookbacks + [risk_lookback]) + 50

        if len(self.df_raw) < min_required_data:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient data ({len(self.df_raw)} points) for {self.symbol} "
                f"to calculate all indicators reliably (recommend min: ~{min_required_data}). "
                f"Results may be inaccurate or contain NaNs.{RESET}"
            )
            # Continue calculation, but be aware results might be compromised

        try:
            # Work on a copy to avoid modifying the raw data
            df_calc = self.df_raw.copy()

            # --- Calculate Risk Management ATR ---
            # This ATR uses 'atr_period' from config and is used for SL/TP/BE calculations.
            # It's separate from the 'volbot_atr_length' used within the Volbot level calcs.
            atr_period_risk = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc["atr_risk"] = ta.atr(df_calc["high"], df_calc["low"], df_calc["close"], length=atr_period_risk)
            self.logger.debug(f"Calculated Risk Management ATR (Length: {atr_period_risk})")

            # --- Calculate Volbot Strategy Indicators ---
            if self.config.get("volbot_enabled", True):
                self.logger.info("Calculating Volbot strategy indicators...")
                df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
                df_calc = calculate_pivot_order_blocks(df_calc, self.config, self.logger)
                # Manage OBs and store the results (df updated, lists of all boxes returned)
                df_calc, self.all_bull_boxes, self.all_bear_boxes = manage_order_blocks(
                    df_calc, self.config, self.logger
                )
                self.logger.info("Volbot strategy indicator calculation complete.")
            else:
                self.logger.info("Volbot strategy calculation skipped (disabled in config).")

            # Assign the DataFrame with all calculated indicators
            self.df_processed = df_calc
            self.logger.debug(
                f"Finished indicator calculations for {self.symbol}. Processed DF columns: {self.df_processed.columns.tolist()}"
            )

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators for {self.symbol}: {e}{RESET}", exc_info=True)
            # df_processed remains empty or incomplete if error occurs

    def _update_latest_strategy_state(self) -> None:
        """Updates the `strategy_state` dictionary with the latest values (last row)
        from the processed DataFrame (`self.df_processed`). Also updates
        `latest_active_bull_ob` and `latest_active_bear_ob`.
        """
        if self.df_processed.empty:
            self.logger.warning(f"Cannot update latest state: Processed DataFrame is empty for {self.symbol}.")
            self.strategy_state = {}
            self.latest_active_bull_ob = None
            self.latest_active_bear_ob = None
            return

        try:
            # Get the last row of the processed DataFrame
            latest = self.df_processed.iloc[-1]

            # Check if the last row contains only NaNs (can happen with insufficient data)
            if latest.isnull().all():
                self.logger.warning(
                    f"{NEON_YELLOW}Cannot update latest state: Last row of processed DataFrame contains all NaNs for {self.symbol}. "
                    f"Likely due to insufficient history ({len(self.df_processed)} rows).{RESET}"
                )
                self.strategy_state = {}
                self.latest_active_bull_ob = None
                self.latest_active_bear_ob = None
                return

            updated_state = {}

            # --- Extract Core Price/Volume Data ---
            for col in ["open", "high", "low", "close", "volume"]:
                updated_state[col] = latest.get(col, np.nan)

            # --- Extract Risk Management ATR ---
            updated_state["atr_risk"] = latest.get("atr_risk", np.nan)

            # --- Extract Volbot Strategy State (if enabled) ---
            if self.config.get("volbot_enabled", True):
                volbot_cols = [
                    "trend_up_strat",
                    "trend_changed_strat",  # Core trend info
                    "vol_trend_up_level_strat",
                    "vol_trend_dn_level_strat",  # Volume-adjusted levels
                    "lower_strat",
                    "upper_strat",  # Base levels
                    "cum_vol_delta_since_change_strat",
                    "cum_vol_total_since_change_strat",  # Cumulative volume
                    "last_trend_change_index_strat",  # Index of last trend change
                    "ema1_strat",
                    "ema2_strat",
                    "atr_strat",  # Base indicators for strategy
                    # OB references are handled separately below
                ]
                for col in volbot_cols:
                    # Use .get() to safely access columns that might be missing if calculation failed
                    updated_state[col] = latest.get(col, np.nan)

                # Get the latest active OB references (these are dicts or None from manage_order_blocks)
                self.latest_active_bull_ob = latest.get("active_bull_ob_strat", None)
                self.latest_active_bear_ob = latest.get("active_bear_ob_strat", None)

                # Add boolean flags to state for easier checking in signal logic
                updated_state["is_in_active_bull_ob"] = self.latest_active_bull_ob is not None
                updated_state["is_in_active_bear_ob"] = self.latest_active_bear_ob is not None

            # Update the main state dictionary
            self.strategy_state = updated_state

            # Log the updated state (filter out NaNs/None for brevity)
            valid_state = {
                k: f"{v:.5f}" if isinstance(v, (float, np.float64, Decimal)) else v
                for k, v in self.strategy_state.items()
                if pd.notna(v) and v is not None
            }
            self.logger.debug(f"Latest strategy state updated for {self.symbol}: {valid_state}")
            if self.latest_active_bull_ob:
                self.logger.debug(
                    f"  Latest Active Bull OB: ID={self.latest_active_bull_ob.get('id', 'N/A')}, "
                    f"Range=[{self.latest_active_bull_ob.get('bottom', 0):.5f}, {self.latest_active_bull_ob.get('top', 0):.5f}]"
                )
            if self.latest_active_bear_ob:
                self.logger.debug(
                    f"  Latest Active Bear OB: ID={self.latest_active_bear_ob.get('id', 'N/A')}, "
                    f"Range=[{self.latest_active_bear_ob.get('bottom', 0):.5f}, {self.latest_active_bear_ob.get('top', 0):.5f}]"
                )

        except IndexError:
            self.logger.error(
                f"Error accessing latest row (iloc[-1]) for {self.symbol}. Processed DataFrame might be unexpectedly empty or too short."
            )
            self.strategy_state = {}
            self.latest_active_bull_ob = None
            self.latest_active_bear_ob = None
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest strategy state for {self.symbol}: {e}", exc_info=True)
            self.strategy_state = {}
            self.latest_active_bull_ob = None
            self.latest_active_bear_ob = None

    # --- Utility Functions (Market Info Access) ---
    def get_price_precision(self) -> int:
        """Gets the number of decimal places for price formatting from market info.
        Uses ccxt market['precision']['price']. Falls back to limits or estimating from price.

        Returns:
            The integer number of decimal places for price.
        """
        try:
            # Prefer precision info
            precision_info = self.market_info.get("precision", {})
            price_precision_val = precision_info.get("price")  # This is often the tick size (e.g., 0.01)

            if price_precision_val is not None:
                # If it's an integer, it directly represents decimal places (less common now)
                if isinstance(price_precision_val, int):
                    return price_precision_val
                # If it's float/str, assume it's the tick size and calculate decimals
                elif isinstance(price_precision_val, (float, str)):
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0:
                        # Calculate decimal places from the tick size
                        return abs(tick_size.normalize().as_tuple().exponent)

            # Fallback to limits info if precision is missing
            limits_info = self.market_info.get("limits", {})
            price_limits = limits_info.get("price", {})
            min_price_val = price_limits.get("min")  # Min price often implies tick size
            if min_price_val is not None:
                min_price_tick = Decimal(str(min_price_val))
                if min_price_tick > 0:
                    return abs(min_price_tick.normalize().as_tuple().exponent)

            # Fallback: Estimate from last close price if available
            last_close = self.strategy_state.get("close")
            if last_close and pd.notna(last_close) and last_close > 0:
                s_close = format(Decimal(str(last_close)), "f")  # Format to string
                if "." in s_close:
                    return len(s_close.split(".")[-1])

        except Exception as e:
            self.logger.warning(f"Could not reliably determine price precision for {self.symbol}: {e}. Using default.")

        # Default precision if all else fails
        default_precision = 4
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) as a Decimal from market info.
        Uses ccxt market['precision']['price'] or falls back to limits.

        Returns:
            The minimum tick size as a Decimal.
        """
        try:
            # Prefer precision info (price precision usually IS the tick size)
            precision_info = self.market_info.get("precision", {})
            price_precision_val = precision_info.get("price")
            if price_precision_val is not None:
                # If it's float/str, assume it's the tick size
                if isinstance(price_precision_val, (float, str)):
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0:
                        return tick_size
                # If it's an integer (decimal places), calculate tick size
                elif isinstance(price_precision_val, int):
                    return Decimal("1e-" + str(price_precision_val))

            # Fallback to limits info
            limits_info = self.market_info.get("limits", {})
            price_limits = limits_info.get("price", {})
            min_price_val = price_limits.get("min")  # Min price often implies tick size
            if min_price_val is not None:
                min_tick_from_limit = Decimal(str(min_price_val))
                if min_tick_from_limit > 0:
                    return min_tick_from_limit

        except Exception as e:
            self.logger.warning(f"Could not reliably determine tick size for {self.symbol}: {e}. Using fallback.")

        # Fallback: Calculate from price precision decimal places
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal("1e-" + str(price_precision_places))
        self.logger.debug(f"Using fallback tick size based on precision places for {self.symbol}: {fallback_tick}")
        return fallback_tick

    # --- Signal Generation (Based on Volbot Rules) ---
    def generate_trading_signal(self) -> str:
        """Generates a trading signal ("BUY", "SELL", or "HOLD") based on the
        latest calculated Volbot strategy state.

        Rules:
        1. If `volbot_signal_on_trend_flip` is True and trend changed:
           - BUY on flip to UP trend.
           - SELL on flip to DOWN trend.
        2. If `volbot_signal_on_ob_entry` is True and no trend flip occurred:
           - BUY if trend is UP and price is inside an active Bullish OB.
           - SELL if trend is DOWN and price is inside an active Bearish OB.
        3. Otherwise, HOLD.

        Returns:
            The trading signal string: "BUY", "SELL", or "HOLD".
        """
        signal = "HOLD"  # Default signal

        if not self.strategy_state or not self.config.get("volbot_enabled", True):
            self.logger.debug("Cannot generate Volbot signal: State is empty or strategy disabled.")
            return signal

        # --- Extract latest state variables ---
        is_trend_up = self.strategy_state.get("trend_up_strat", None)  # Boolean (True/False) or None if undetermined
        trend_changed = self.strategy_state.get("trend_changed_strat", False)
        is_in_bull_ob = self.strategy_state.get("is_in_active_bull_ob", False)
        is_in_bear_ob = self.strategy_state.get("is_in_active_bear_ob", False)

        # --- Get config flags for signal generation triggers ---
        signal_on_flip = self.config.get("volbot_signal_on_trend_flip", True)
        signal_on_ob = self.config.get("volbot_signal_on_ob_entry", True)

        # --- Determine Signal based on rules ---
        if is_trend_up is None:
            self.logger.debug("Volbot signal: HOLD (Trend state could not be determined)")
            return "HOLD"

        # Rule 1: Trend Flip Signal (Highest Priority)
        if signal_on_flip and trend_changed:
            if is_trend_up:
                signal = "BUY"
                self.logger.info(f"{COLOR_UP}Volbot Signal: BUY (Reason: Trend flipped to UP){RESET}")
            else:  # Trend flipped down
                signal = "SELL"
                self.logger.info(f"{COLOR_DN}Volbot Signal: SELL (Reason: Trend flipped to DOWN){RESET}")
            return signal  # Return immediately on trend flip signal

        # Rule 2: Order Block Entry Signal (If no trend flip)
        if signal_on_ob:
            if is_trend_up and is_in_bull_ob:
                # Check if price *entered* the OB on this bar (optional refinement)
                # requires checking previous bar state. For now, signal if inside + trend matches.
                signal = "BUY"
                ob_id = self.latest_active_bull_ob.get("id", "N/A") if self.latest_active_bull_ob else "N/A"
                self.logger.info(
                    f"{COLOR_BULL_BOX}Volbot Signal: BUY (Reason: Price in Bull OB '{ob_id}' during Uptrend){RESET}"
                )
                return signal
            elif not is_trend_up and is_in_bear_ob:
                signal = "SELL"
                ob_id = self.latest_active_bear_ob.get("id", "N/A") if self.latest_active_bear_ob else "N/A"
                self.logger.info(
                    f"{COLOR_BEAR_BOX}Volbot Signal: SELL (Reason: Price in Bear OB '{ob_id}' during Downtrend){RESET}"
                )
                return signal

        # Rule 3: Default to HOLD if no other signals triggered
        if signal == "HOLD":
            trend_str = "UP" if is_trend_up else "DOWN"
            self.logger.info(
                f"Volbot Signal: HOLD (Conditions: Trend={trend_str}, In Bull OB={is_in_bull_ob}, In Bear OB={is_in_bear_ob})"
            )

        return signal

    # --- Risk Management Calculations (Using Risk ATR) ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential Take Profit (TP) and initial Stop Loss (SL) levels
        based on a potential entry price, the generated signal ("BUY" or "SELL"),
        the Risk Management ATR (`atr_risk`), and configured multipliers.
        Rounds results to the market's minimum tick size.

        Args:
            entry_price: The potential entry price (as Decimal).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            A tuple containing (entry_price, take_profit, stop_loss),
            all as Decimal or None if calculation is not possible or invalid.
        """
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"Cannot calculate TP/SL: Signal is '{signal}'.")
            return entry_price, None, None

        # --- Get necessary values ---
        atr_val_float = self.strategy_state.get("atr_risk")  # Use the separate Risk Management ATR
        tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
        sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
        price_precision = self.get_price_precision()  # For logging
        min_tick = self.get_min_tick_size()

        # --- Validate inputs ---
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(
                f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Risk ATR ('atr_risk') is invalid ({atr_val_float}). Check data/config.{RESET}"
            )
            return entry_price, None, None
        if entry_price is None or entry_price <= 0:
            self.logger.warning(
                f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}"
            )
            return entry_price, None, None
        if min_tick <= 0:
            self.logger.error(
                f"{NEON_RED}Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).{RESET}"
            )
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float))
            take_profit, stop_loss = None, None

            # --- Calculate Raw SL/TP ---
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            if signal == "BUY":
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Round TP up to nearest tick, SL down to nearest tick
                take_profit = (take_profit_raw / min_tick).quantize(Decimal("1"), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal("1"), rounding=ROUND_DOWN) * min_tick
            elif signal == "SELL":
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Round TP down to nearest tick, SL up to nearest tick
                take_profit = (take_profit_raw / min_tick).quantize(Decimal("1"), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal("1"), rounding=ROUND_UP) * min_tick

            # --- Validate Calculated SL/TP ---
            min_sl_distance = min_tick  # Ensure SL is at least one tick away
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_sl_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(
                    f"{NEON_YELLOW}BUY SL ({stop_loss:.{price_precision}f}) calculated >= entry ({entry_price:.{price_precision}f}). Adjusting SL to {adjusted_sl:.{price_precision}f}.{RESET}"
                )
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_sl_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(
                    f"{NEON_YELLOW}SELL SL ({stop_loss:.{price_precision}f}) calculated <= entry ({entry_price:.{price_precision}f}). Adjusting SL to {adjusted_sl:.{price_precision}f}.{RESET}"
                )
                stop_loss = adjusted_sl

            min_tp_distance = min_tick  # Ensure TP allows for at least one tick profit
            if signal == "BUY" and take_profit <= entry_price:
                adjusted_tp = (entry_price + min_tp_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(
                    f"{NEON_YELLOW}BUY TP ({take_profit:.{price_precision}f}) non-profitable vs entry ({entry_price:.{price_precision}f}). Adjusting TP to {adjusted_tp:.{price_precision}f}.{RESET}"
                )
                take_profit = adjusted_tp
            elif signal == "SELL" and take_profit >= entry_price:
                adjusted_tp = (entry_price - min_tp_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(
                    f"{NEON_YELLOW}SELL TP ({take_profit:.{price_precision}f}) non-profitable vs entry ({entry_price:.{price_precision}f}). Adjusting TP to {adjusted_tp:.{price_precision}f}.{RESET}"
                )
                take_profit = adjusted_tp

            # Ensure SL/TP are positive prices
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(
                    f"{NEON_RED}SL calculation resulted in zero/negative price ({stop_loss}). Cannot set SL.{RESET}"
                )
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(
                    f"{NEON_RED}TP calculation resulted in zero/negative price ({take_profit}). Cannot set TP.{RESET}"
                )
                take_profit = None

            # Log the results
            tp_str = f"{take_profit:.{price_precision}f}" if take_profit else "N/A"
            sl_str = f"{stop_loss:.{price_precision}f}" if stop_loss else "N/A"
            self.logger.info(
                f"Calculated TP/SL for {self.symbol} {signal} using Risk ATR ({atr:.{price_precision + 1}f}): "
                f"Entry={entry_price:.{price_precision}f}, TP={tp_str}, SL={sl_str}"
            )
            return entry_price, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None


# --- Trading Logic Helper Functions ---


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency from the exchange.
    Handles different account types for Bybit (CONTRACT, UNIFIED) and parses
    various balance response structures (standard ccxt, Bybit V5 nested).

    Args:
        exchange: Initialized ccxt.Exchange object.
        currency: The currency code to fetch the balance for (e.g., "USDT").
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetching fails or balance is not found.
    """
    lg = logger
    try:
        balance_info: dict | None = None
        # Account types relevant for Bybit V5 API balance fetching
        account_types_to_try = ["CONTRACT", "UNIFIED"]  # Try derivatives first, then unified

        # Attempt fetching balance for specific account types first (more reliable on Bybit V5)
        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Fetching balance using params={{'type': '{acc_type}'}} for {currency}...")
                balance_info = exchange.fetch_balance(params={"type": acc_type})

                # Check if currency exists directly in the balance structure
                if currency in balance_info:
                    # Check for standard 'free' balance
                    if balance_info[currency].get("free") is not None:
                        lg.debug(f"Found '{currency}' in balance (type '{acc_type}') with 'free' key.")
                        break  # Found balance, exit loop
                    # Check for Bybit V5 nested structure within 'info'
                    elif (
                        "info" in balance_info
                        and "result" in balance_info["info"]
                        and "list" in balance_info["info"]["result"]
                    ):
                        balance_list = balance_info["info"]["result"]["list"]
                        if isinstance(balance_list, list):
                            for account in balance_list:
                                # Check if the account type matches the one we requested
                                if account.get("accountType") == acc_type:
                                    coin_list = account.get("coin")
                                    if isinstance(coin_list, list):
                                        # Check if the currency exists within this account's coin list
                                        if any(coin_data.get("coin") == currency for coin_data in coin_list):
                                            lg.debug(
                                                f"Found '{currency}' nested within V5 structure (type '{acc_type}')."
                                            )
                                            break  # Found nested currency
                            if any(coin_data.get("coin") == currency for coin_data in coin_list):
                                break  # Break outer loop too if found

                    # If currency key exists but no usable balance found yet, log and try next type
                    lg.debug(
                        f"Currency '{currency}' found (type '{acc_type}'), but usable balance (free/V5) not found. Trying next type."
                    )
                    balance_info = None  # Reset balance_info to continue loop

                # Check if currency exists ONLY in the nested V5 structure (not top-level key)
                elif (
                    "info" in balance_info
                    and "result" in balance_info["info"]
                    and "list" in balance_info["info"]["result"]
                ):
                    balance_list = balance_info["info"]["result"]["list"]
                    if isinstance(balance_list, list):
                        found_nested = False
                        for account in balance_list:
                            if account.get("accountType") == acc_type:
                                coin_list = account.get("coin")
                                if isinstance(coin_list, list):
                                    if any(coin_data.get("coin") == currency for coin_data in coin_list):
                                        lg.debug(
                                            f"Found '{currency}' ONLY nested within V5 structure (type '{acc_type}')."
                                        )
                                        found_nested = True
                                        break
                        if found_nested:
                            break  # Break outer loop if found

                # If currency not found at all for this account type
                if balance_info is None:  # Check if we reset it or it wasn't found nested
                    lg.debug(f"Currency '{currency}' not found in balance structure (type '{acc_type}'). Trying next.")
                    balance_info = None  # Ensure it's None to continue loop

            except ccxt.ExchangeError as e:
                # Ignore errors related to unsupported account types and try the next one
                if "account type not support" in str(e).lower() or "invalid account type" in str(e).lower():
                    lg.debug(f"Account type '{acc_type}' not supported/invalid. Trying next.")
                    continue
                else:
                    # Log other exchange errors but continue trying other types/default
                    lg.warning(f"Exchange error fetching balance (type {acc_type}): {e}. Trying next.")
                    continue
            except Exception as e:
                lg.warning(f"Unexpected error fetching balance (type {acc_type}): {e}. Trying next.")
                continue

        # If no balance found with specific account types, try default fetch_balance
        if not balance_info:
            lg.debug(f"Fetching balance using default parameters for {currency}...")
            try:
                balance_info = exchange.fetch_balance()
            except Exception as e:
                lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                return None  # Cannot proceed without balance info

        # --- Parse the Balance from the final balance_info structure ---
        available_balance_str: str | None = None

        # 1. Check standard ccxt 'free' balance
        if currency in balance_info and "free" in balance_info[currency] and balance_info[currency]["free"] is not None:
            available_balance_str = str(balance_info[currency]["free"])
            lg.debug(f"Parsing balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}")

        # 2. Check Bybit V5 nested structure ('info.result.list') if standard failed
        elif (
            "info" in balance_info
            and "result" in balance_info["info"]
            and isinstance(balance_info["info"]["result"].get("list"), list)
        ):
            balance_list = balance_info["info"]["result"]["list"]
            # Determine which account type was successful if params were used
            successful_acc_type = balance_info.get("params", {}).get("type")
            parsed_account_type = "N/A"
            for account in balance_list:
                current_account_type = account.get("accountType")
                # Process if it matches the successful type, or if no specific type was successful (default fetch)
                if successful_acc_type is None or current_account_type == successful_acc_type:
                    coin_list = account.get("coin")
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                            if coin_data.get("coin") == currency:
                                # Try different keys for available balance in V5 structure
                                free = coin_data.get(
                                    "availableToWithdraw",
                                    coin_data.get("availableBalance", coin_data.get("walletBalance")),
                                )  # Last resort: walletBalance
                                if free is not None:
                                    available_balance_str = str(free)
                                    parsed_account_type = current_account_type or "Unknown"
                                    break  # Found balance for the currency
                        if available_balance_str is not None:
                            break  # Found balance, exit account loop
            if available_balance_str:
                lg.debug(
                    f"Parsing balance via Bybit V5 nested list: {available_balance_str} {currency} (Account: {parsed_account_type})"
                )
            else:
                lg.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]'.")

        # 3. Check older/alternative top-level 'free' dictionary structure
        elif (
            "free" in balance_info
            and isinstance(balance_info["free"], dict)
            and currency in balance_info["free"]
            and balance_info["free"][currency] is not None
        ):
            available_balance_str = str(balance_info["free"][currency])
            lg.debug(f"Parsing balance via top-level 'free' dictionary: {available_balance_str} {currency}")

        # --- Handle Missing 'free' Balance ---
        if available_balance_str is None:
            # Fallback to 'total' balance if 'free' is unavailable
            total_balance = balance_info.get(currency, {}).get("total")
            if total_balance is not None:
                lg.warning(
                    f"{NEON_YELLOW}Could not find 'free'/'available' balance for {currency}. "
                    f"Using 'total' balance ({total_balance}) as fallback. This might include collateral.{RESET}"
                )
                available_balance_str = str(total_balance)
            else:
                # If neither free nor total found
                lg.error(f"{NEON_RED}Could not determine any balance ('free' or 'total') for {currency}.{RESET}")
                lg.debug(f"Full balance_info structure: {balance_info}")
                return None

        # --- Convert final balance string to Decimal ---
        try:
            final_balance = Decimal(available_balance_str)
            if final_balance >= 0:
                lg.info(f"Available {currency} balance: {final_balance:.4f}")
                return final_balance
            else:
                lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Returning None.")
                return None
        except Exception as e:
            lg.error(f"Failed to convert final balance string '{available_balance_str}' to Decimal: {e}")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching balance: {e}. Check API keys/permissions.{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Retrieves detailed market information for a given symbol from the exchange.
    Ensures markets are loaded and handles potential errors. Adds an 'is_contract' flag.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT', 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing market information (precision, limits, type, etc.),
        or None if the symbol is not found or an error occurs.
    """
    lg = logger
    try:
        # Ensure markets are loaded; reload if symbol not found initially
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded or symbol missing, reloading markets...")
            exchange.load_markets(reload=True)  # Force reload

        # Check again after reloading
        if symbol not in exchange.markets:
            lg.error(
                f"{NEON_RED}Market {symbol} still not found after reloading. "
                f"Check if the symbol exists on {exchange.id} in the correct format (e.g., with :QUOTE for Bybit linear).{RESET}"
            )
            return None

        # Retrieve market details
        market = exchange.market(symbol)
        if market:
            # Determine market type and contract type for logging/logic
            market_type = market.get("type", "unknown")  # spot, swap, future, etc.
            is_linear = market.get("linear", False)
            is_inverse = market.get("inverse", False)
            contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "N/A"

            # Add a convenient boolean flag to check if it's a contract market
            is_contract_market = market.get("contract", False) or market_type in ["swap", "future"]
            market["is_contract"] = is_contract_market  # Add flag to the dict

            # Log detailed market info for debugging
            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract={contract_type}, "
                f"IsContract={is_contract_market}, "
                f"Precision(Price/Amount/Tick): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}/{market.get('precision', {}).get('tick', 'N/A')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            return market
        else:
            # This case should be rare if exchange.market(symbol) succeeded but returned None
            lg.error(f"{NEON_RED}Market dictionary could not be retrieved for {symbol} despite being listed.{RESET}")
            return None

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Symbol '{symbol}' is not supported or invalid on {exchange.id}: {e}{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error loading markets or getting info for {symbol}: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error loading markets or getting info for {symbol}: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: dict,
    exchange: ccxt.Exchange,
    logger: logging.Logger | None = None,
) -> Decimal | None:
    """Calculates the appropriate position size based on account balance, risk percentage,
    stop-loss distance, and market constraints (precision, limits, contract size).

    Args:
        balance: Available balance in the quote currency (as Decimal).
        risk_per_trade: Risk percentage per trade (e.g., 0.01 for 1%).
        initial_stop_loss_price: Calculated initial stop-loss price (as Decimal).
        entry_price: Potential or actual entry price (as Decimal).
        market_info: Market information dictionary from get_market_info.
        exchange: Initialized ccxt.Exchange object.
        logger: Logger instance.

    Returns:
        The calculated and adjusted position size as a Decimal, or None if calculation fails.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
    quote_currency = market_info.get("quote", QUOTE_CURRENCY)  # e.g., USDT
    base_currency = market_info.get("base", "BASE")  # e.g., BTC
    is_contract = market_info.get("is_contract", False)
    # Determine unit for logging size
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):  # Risk should be between 0 and 1 (exclusive)
        lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be > 0 and < 1.")
        return None
    if initial_stop_loss_price is None or entry_price is None or entry_price <= 0:
        lg.error(
            f"Position sizing failed ({symbol}): Invalid entry price ({entry_price}) or SL price ({initial_stop_loss_price})."
        )
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed ({symbol}): Stop loss price cannot be equal to entry price.")
        return None
    # Allow SL=0 for certain scenarios, but log warning if it's non-positive unexpectedly
    if initial_stop_loss_price <= 0:
        lg.warning(
            f"Position sizing ({symbol}): Calculated initial SL price ({initial_stop_loss_price}) is zero or negative."
        )

    if "limits" not in market_info or "precision" not in market_info:
        lg.error(f"Position sizing failed ({symbol}): Missing market 'limits' or 'precision' info.")
        return None

    try:
        # --- Core Calculation ---
        # 1. Calculate Risk Amount in Quote Currency
        risk_amount_quote = balance * Decimal(str(risk_per_trade))

        # 2. Calculate Stop Loss Distance (Risk per Unit) in Quote Currency
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
            lg.error(f"Position sizing failed ({symbol}): SL distance is zero or negative ({sl_distance_per_unit}).")
            return None

        # 3. Get Contract Size (Value of 1 contract/unit)
        contract_size_str = market_info.get("contractSize", "1")  # Default to 1 if missing
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0:
                raise ValueError("Contract size must be positive")
        except (ValueError, TypeError, Exception) as cs_err:
            lg.warning(
                f"Could not parse contract size '{contract_size_str}' as positive Decimal ({cs_err}). Defaulting to 1."
            )
            contract_size = Decimal("1")

        # 4. Calculate Initial Position Size
        calculated_size = Decimal("0")
        # For Linear contracts or Spot: Size = Risk Amount / (SL Distance * Contract Size)
        # Contract size here is typically 1 for spot/linear futures (meaning 1 unit of base currency)
        if market_info.get("linear", True) or not is_contract:
            value_per_unit = sl_distance_per_unit * contract_size
            if value_per_unit > 0:
                calculated_size = risk_amount_quote / value_per_unit
            else:
                lg.error(f"Pos sizing failed (linear/spot {symbol}): Risk per unit zero/negative ({value_per_unit}).")
                return None
        # For Inverse contracts: Size = Risk Amount / (SL Distance * Contract Size / Entry Price)
        # Contract size here is typically the quote value of 1 contract (e.g., 1 USD)
        else:  # Inverse contract
            lg.warning(
                f"{NEON_YELLOW}Inverse contract {symbol} detected. Sizing logic assumes 'contractSize' ({contract_size}) is the value of 1 contract in Quote currency.{RESET}"
            )
            if entry_price > 0:
                # Risk per contract in Quote currency: (Price Diff * Quote Value per Contract) / Entry Price
                risk_per_contract_quote = sl_distance_per_unit * contract_size / entry_price
                if risk_per_contract_quote > 0:
                    calculated_size = risk_amount_quote / risk_per_contract_quote
                else:
                    lg.error(
                        f"Pos sizing failed (inverse {symbol}): Calculated risk per contract is zero/negative ({risk_per_contract_quote})."
                    )
                    return None
            else:
                lg.error(f"Pos sizing failed (inverse {symbol}): Entry price is zero or negative.")
                return None

        lg.info(
            f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, Risk Amount={risk_amount_quote:.4f} {quote_currency}"
        )
        lg.info(
            f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit:.{abs(entry_price.as_tuple().exponent)}f}"
        )
        lg.info(f"  ContractSize={contract_size}, IsLinear/Spot={market_info.get('linear', True) or not is_contract}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Constraints ---
        limits = market_info.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        precision = market_info.get("precision", {})
        # Amount precision can be integer (decimals) or float/str (step size)
        amount_precision_val = precision.get("amount")

        # Get min/max amount limits
        min_amount_str = amount_limits.get("min")
        max_amount_str = amount_limits.get("max")
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal("0")
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal("inf")

        # Get min/max cost limits
        min_cost_str = cost_limits.get("min")
        max_cost_str = cost_limits.get("max")
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal("0")
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal("inf")

        # 1. Adjust size based on Min/Max Amount Limits
        adjusted_size = calculated_size
        if adjusted_size < min_amount:
            lg.warning(
                f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} < min amount {min_amount}. Adjusting size up to minimum.{RESET}"
            )
            adjusted_size = min_amount
        elif adjusted_size > max_amount:
            lg.warning(
                f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} > max amount {max_amount}. Capping size to maximum.{RESET}"
            )
            adjusted_size = max_amount

        # 2. Check and Adjust size based on Min/Max Cost Limits
        # Estimate cost of the amount-adjusted size
        current_cost = Decimal("0")
        if market_info.get("linear", True) or not is_contract:
            # Cost = Size * Price * ContractSize (ContractSize is often 1 for linear/spot)
            current_cost = adjusted_size * entry_price * contract_size
        else:  # Inverse
            # Cost = Size * ContractSize (ContractSize is quote value, e.g., 1 USD)
            contract_value_quote = contract_size  # Value of 1 contract in quote currency
            current_cost = adjusted_size * contract_value_quote

        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Estimated Cost={current_cost:.4f} {quote_currency}")

        # Adjust for Min Cost
        if min_cost > 0 and current_cost < min_cost:
            lg.warning(
                f"{NEON_YELLOW}Estimated cost {current_cost:.4f} < min cost {min_cost}. Attempting to increase size to meet min cost.{RESET}"
            )
            required_size_for_min_cost = Decimal("0")
            if market_info.get("linear", True) or not is_contract:
                if entry_price > 0 and contract_size > 0:
                    required_size_for_min_cost = min_cost / (entry_price * contract_size)
                else:
                    lg.error(
                        f"{NEON_RED}Cannot calculate required size for min cost (linear/spot): Invalid price/contract size.{RESET}"
                    )
                    return None
            else:  # Inverse
                contract_value_quote = contract_size
                if contract_value_quote > 0:
                    required_size_for_min_cost = min_cost / contract_value_quote
                else:
                    lg.error(
                        f"{NEON_RED}Cannot calculate required size for min cost (inverse): Invalid contract value.{RESET}"
                    )
                    return None

            lg.info(f"  Required size to meet min cost: {required_size_for_min_cost:.8f} {size_unit}")
            # Check if this required size violates other limits
            if required_size_for_min_cost > max_amount:
                lg.error(
                    f"{NEON_RED}Cannot meet min cost ({min_cost}): Required size {required_size_for_min_cost:.8f} exceeds max amount limit ({max_amount}). Aborted.{RESET}"
                )
                return None
            # If required size is less than min amount (after initial adjustment), something is wrong with limits
            if required_size_for_min_cost < min_amount:
                lg.error(
                    f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f} which is less than min amount {min_amount}. Aborted.{RESET}"
                )
                return None

            lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f}")
            adjusted_size = required_size_for_min_cost

        # Adjust for Max Cost
        elif max_cost > 0 and current_cost > max_cost:
            lg.warning(
                f"{NEON_YELLOW}Estimated cost {current_cost:.4f} > max cost {max_cost}. Reducing size to meet max cost.{RESET}"
            )
            adjusted_size_for_max_cost = Decimal("0")
            if market_info.get("linear", True) or not is_contract:
                if entry_price > 0 and contract_size > 0:
                    adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                else:
                    lg.error(
                        f"{NEON_RED}Cannot calculate reduced size for max cost (linear/spot): Invalid price/contract size.{RESET}"
                    )
                    return None
            else:  # Inverse
                contract_value_quote = contract_size
                if contract_value_quote > 0:
                    adjusted_size_for_max_cost = max_cost / contract_value_quote
                else:
                    lg.error(
                        f"{NEON_RED}Cannot calculate reduced size for max cost (inverse): Invalid contract value.{RESET}"
                    )
                    return None

            lg.info(f"  Reduced size required to meet max cost: {adjusted_size_for_max_cost:.8f} {size_unit}")
            # Check if reduced size violates min amount limit
            if adjusted_size_for_max_cost < min_amount:
                lg.error(
                    f"{NEON_RED}Cannot meet max cost ({max_cost}): Reduced size {adjusted_size_for_max_cost:.8f} is below min amount limit ({min_amount}). Aborted.{RESET}"
                )
                return None

            adjusted_size = adjusted_size_for_max_cost

        # 3. Apply Amount Precision (Rounding/Truncation)
        final_size = Decimal("0")
        try:
            # Use ccxt's built-in precision handling if possible
            # TRUNCATE (or ROUND_DOWN) is generally safer for position sizing to avoid exceeding margin
            formatted_size_str = exchange.amount_to_precision(
                symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE
            )
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied exchange amount precision (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as fmt_err:
            lg.warning(
                f"{NEON_YELLOW}Could not use exchange.amount_to_precision ({fmt_err}). Using manual rounding/step size.{RESET}"
            )
            # Manual fallback using precision/limit info
            step_size, num_decimals = None, None
            if amount_precision_val is not None:
                if isinstance(amount_precision_val, (float, str)):
                    try:
                        step_size = Decimal(str(amount_precision_val))
                        if step_size <= 0:
                            step_size = None
                    except:
                        pass
                elif isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                    num_decimals = amount_precision_val

            if step_size is not None:
                # Apply step size (floor division)
                final_size = (adjusted_size // step_size) * step_size
                lg.info(f"Applied manual step size ({step_size}): {adjusted_size:.8f} -> {final_size}")
            elif num_decimals is not None:
                # Apply decimal place rounding (down)
                rounding_factor = Decimal("1e-" + str(num_decimals))
                final_size = adjusted_size.quantize(rounding_factor, rounding=ROUND_DOWN)
                lg.info(f"Applied manual precision ({num_decimals} decimals): {adjusted_size:.8f} -> {final_size}")
            else:
                lg.warning(
                    f"Amount precision value ('{amount_precision_val}') invalid or missing. Using limit-adjusted size without final precision: {adjusted_size:.8f}"
                )
                final_size = adjusted_size  # Use the size adjusted only by limits

        # --- Final Validation ---
        if final_size <= 0:
            lg.error(
                f"{NEON_RED}Position size became zero or negative ({final_size}) after precision/limit adjustments. Aborted.{RESET}"
            )
            return None

        # Ensure final size still meets min amount (allow for tiny floating point differences)
        if final_size < min_amount and not math.isclose(float(final_size), float(min_amount), rel_tol=1e-9):
            lg.error(
                f"{NEON_RED}Final size {final_size} after precision is less than min amount {min_amount}. Aborted.{RESET}"
            )
            return None

        # Final check on minimum cost if applicable
        final_cost = Decimal("0")
        if min_cost > 0:
            if market_info.get("linear", True) or not is_contract:
                final_cost = final_size * entry_price * contract_size
            else:
                contract_value_quote = contract_size
                final_cost = final_size * contract_value_quote
            # Check if final cost dropped below min cost due to precision rounding (allow small tolerance)
            if final_cost < min_cost and not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6):
                lg.error(
                    f"{NEON_RED}Final cost {final_cost:.4f} after precision is less than min cost {min_cost}. Aborted.{RESET}"
                )
                return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(
            f"{NEON_RED}Position sizing error ({symbol}): Missing expected key in market_info: {e}. Market Info: {market_info}{RESET}"
        )
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks if there is an open position for the specified symbol using ccxt.
    Enhances the returned position dictionary by parsing SL/TP/TSL from the 'info' field if needed.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing the active position details (enhanced with SL/TP/TSL),
        or None if no active position is found or an error occurs.
    """
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: list[dict] = []
        fetch_all = False  # Flag to fetch all positions if single symbol fetch fails

        # Try fetching position for the specific symbol first (more efficient)
        try:
            params = {}
            # Add Bybit V5 category parameter if applicable
            if "bybit" in exchange.id.lower():
                market = None
                try:
                    market = exchange.market(symbol)
                except ccxt.BadSymbol:
                    lg.warning(f"Market info not found for {symbol} during position check. Assuming 'linear'.")
                    market = {"linear": True}  # Assume linear if market info fails here
                # Determine category based on market info (linear/inverse)
                category = "linear" if market.get("linear", True) else "inverse"
                params["category"] = category
                lg.debug(f"Using params for fetch_positions: {params}")

            # Fetch positions for the single symbol
            positions = exchange.fetch_positions(symbols=[symbol], params=params)

        except ccxt.ArgumentsRequired:
            # Some exchanges might require fetching all positions
            lg.debug(f"Exchange {exchange.id} requires fetching all positions. Fetching all...")
            fetch_all = True
        except ccxt.ExchangeError as e:
            # Handle specific Bybit V5 errors indicating no position
            # 110025: position idx not exist error / positionStatus=NOT_HOLDING (or similar)
            # 110021: Order not found or insufficient balance for order (can sometimes imply no position)
            no_pos_codes_v5 = [110025, 110021]
            no_pos_msgs = ["position idx not exist", "no position found", "position does not exist"]
            err_str = str(e).lower()
            err_code = getattr(e, "code", None)

            if "symbol not found" in err_str or "instrument not found" in err_str:
                lg.warning(f"Symbol {symbol} not found when fetching position: {e}. Assuming no position.")
                return None  # Treat as no position if symbol is invalid for positions endpoint
            if err_code in no_pos_codes_v5 or any(msg in err_str for msg in no_pos_msgs):
                lg.info(f"No position found for {symbol} (Exchange code/message: {err_code}/{err_str}).")
                return None  # Explicitly no position found
            # Log other exchange errors but might indicate temporary issue, return None for safety
            lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=True)
            return None
        except Exception as e:
            lg.error(f"Unexpected error fetching single position for {symbol}: {e}", exc_info=True)
            return None  # Unknown error, assume no position for safety

        # If single symbol fetch failed or wasn't supported, fetch all positions
        if fetch_all:
            try:
                all_positions = exchange.fetch_positions()  # Fetch all positions
                # Filter for the target symbol
                positions = [p for p in all_positions if p.get("symbol") == symbol]
                lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
            except Exception as e:
                lg.error(f"Error fetching all positions after single fetch failed for {symbol}: {e}", exc_info=True)
                return None

        # --- Find the Active Position ---
        active_position: dict | None = None
        for pos in positions:
            # Determine position size from various possible keys ('contracts', 'contractSize', info fields)
            pos_size_str = pos.get(
                "contracts",  # Standard ccxt field
                pos.get(
                    "contractSize",  # Sometimes used interchangeably
                    pos.get("info", {}).get(
                        "size",  # Bybit V5 info field
                        pos.get("info", {}).get("positionAmt"),
                    ),
                ),
            )  # Binance info field (example)

            if pos_size_str is None:
                lg.debug(f"Skipping position entry, missing size field: {pos}")
                continue  # Skip if size cannot be determined

            try:
                # Check if size is non-zero (use Decimal for precision)
                position_size = Decimal(str(pos_size_str))
                size_threshold = Decimal("1e-9")  # Threshold to consider a position non-zero
                if abs(position_size) > size_threshold:
                    active_position = pos
                    break  # Found the first active position for the symbol
            except Exception as parse_err:
                lg.warning(
                    f"Could not parse position size '{pos_size_str}' as Decimal: {parse_err}. Skipping entry: {pos}"
                )
                continue

        # --- Process and Enhance the Active Position ---
        if active_position:
            # Standardize 'side' ('long' or 'short')
            side = active_position.get("side")  # Standard ccxt field
            size_decimal = Decimal("0")
            try:
                # Use 'size' from info dict for Bybit V5 side determination if needed
                size_str_for_side = active_position.get("info", {}).get("size", "0")
                size_decimal = Decimal(str(size_str_for_side))
            except Exception:
                pass

            if side not in ["long", "short"]:
                # Fallback using 'info' fields if standard 'side' is missing/invalid
                info_side = active_position.get("info", {}).get("side", "None")  # Bybit V5 uses 'Buy'/'Sell'/'None'
                if info_side == "Buy":
                    side = "long"
                elif info_side == "Sell":
                    side = "short"
                # Fallback based on size if info side is also missing
                elif size_decimal > size_threshold:
                    side = "long"
                elif size_decimal < -size_threshold:
                    side = "short"
                else:
                    lg.warning(
                        f"Position found for {symbol}, but size ({size_decimal}) is near zero. Cannot determine side. Assuming no active position."
                    )
                    return None
                active_position["side"] = side  # Update the dictionary with standardized side

            # --- Enhance with SL/TP/TSL from 'info' dict (Bybit V5 specific parsing) ---
            info_dict = active_position.get("info", {})

            # Helper to safely get price strings from info, ignoring '0' or empty strings
            def get_valid_price_from_info(key: str) -> str | None:
                val_str = info_dict.get(key)
                if val_str and str(val_str).strip():  # Check if not None and not empty
                    try:
                        # Check if it's a valid positive number, ignore "0", "0.0" etc.
                        if Decimal(str(val_str)) > 0:
                            return str(val_str)
                    except Exception:
                        pass  # Ignore parsing errors
                return None

            # If standard SL/TP fields are missing, try parsing from 'info'
            if active_position.get("stopLossPrice") is None:
                sl_val_str = get_valid_price_from_info("stopLoss")
                if sl_val_str:
                    active_position["stopLossPrice"] = sl_val_str
            if active_position.get("takeProfitPrice") is None:
                tp_val_str = get_valid_price_from_info("takeProfit")
                if tp_val_str:
                    active_position["takeProfitPrice"] = tp_val_str

            # Parse TSL info from Bybit V5 'info' dict
            active_position["trailingStopLoss"] = info_dict.get("trailingStop", "0")  # TSL distance/value
            active_position["tslActivationPrice"] = info_dict.get("activePrice", "0")  # TSL activation price

            # --- Log Position Details ---
            # Helper for formatting prices/values for logging
            def format_log_val(key, value, precision=6):
                if value is None or value == "":
                    return "N/A"
                try:
                    d_val = Decimal(str(value))
                    # Format prices/PnL with precision, allow 0 for TSL fields
                    if d_val > 0 or (d_val == 0 and key in ["trailingStopLoss", "tslActivationPrice"]):
                        # Use appropriate precision for different fields
                        if "Price" in key or key == "stopLoss" or key == "takeProfit":
                            # Use price precision from market info if possible
                            price_prec = 4  # Default
                            try:
                                market_p = exchange.market(symbol)["precision"]["price"]
                                if market_p:
                                    price_prec = abs(Decimal(str(market_p)).normalize().as_tuple().exponent)
                            except:
                                pass
                            return f"{d_val:.{price_prec}f}"
                        elif key == "trailingStopLoss":
                            # Format as percentage or value depending on how it's set
                            return f"{d_val}"  # Keep raw value for now
                        else:
                            return f"{d_val:.{precision}f}"
                    else:
                        return str(value)  # Return original if not positive (or not TSL zero)
                except Exception:
                    return str(value)  # Return original string if conversion fails

            pos_details = {
                "Symbol": active_position.get("symbol", "N/A"),
                "Side": active_position.get("side", "N/A").upper(),
                "Size": format_log_val("size", active_position.get("contracts", info_dict.get("size"))),
                "EntryPrice": format_log_val(
                    "entryPrice", active_position.get("entryPrice", info_dict.get("avgPrice"))
                ),
                "MarkPrice": format_log_val("markPrice", active_position.get("markPrice", info_dict.get("markPrice"))),
                "LiqPrice": format_log_val(
                    "liquidationPrice", active_position.get("liquidationPrice", info_dict.get("liqPrice")), precision=2
                ),
                "uPnL": format_log_val(
                    "unrealizedPnl", active_position.get("unrealizedPnl", info_dict.get("unrealisedPnl")), precision=4
                ),
                "Collateral": format_log_val(
                    "collateral", active_position.get("collateral", info_dict.get("positionIM")), precision=4
                ),  # Initial Margin as collateral proxy
                "Leverage": format_log_val(
                    "leverage", active_position.get("leverage", info_dict.get("leverage")), precision=1
                ),
                "TP": format_log_val("takeProfitPrice", active_position.get("takeProfitPrice"), precision=4),
                "SL": format_log_val("stopLossPrice", active_position.get("stopLossPrice"), precision=4),
                "TSL": format_log_val("trailingStopLoss", active_position.get("trailingStopLoss"), precision=4),
                "TSL_Act": format_log_val("tslActivationPrice", active_position.get("tslActivationPrice"), precision=4),
            }
            details_str = ", ".join(f"{k}={v}" for k, v in pos_details.items())
            lg.info(f"{NEON_GREEN}Active Position Found: {details_str}{RESET}")
            return active_position
        else:
            lg.info(f"No active position found for {symbol}.")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching positions: {e}{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching positions: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.error(
            f"{NEON_RED}Exchange error fetching positions: {e}{RESET}", exc_info=True
        )  # Log traceback for exchange errors
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error checking positions for {symbol}: {e}{RESET}", exc_info=True)
        return None


def place_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,  # 'buy' or 'sell'
    amount: Decimal,
    market_info: dict,
    logger: logging.Logger,
    params: dict | None = None,
) -> dict | None:
    """Places a market order using ccxt with retries for specific errors.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        side: 'buy' or 'sell'.
        amount: Order size (in base currency or contracts).
        market_info: Market info dictionary.
        logger: Logger instance.
        params: Additional parameters for the create_order call.

    Returns:
        The order dictionary returned by ccxt, or None if the order fails.
    """
    lg = logger
    if not CONFIG.get("enable_trading", False):
        lg.warning(
            f"{NEON_YELLOW}TRADING DISABLED in config. Skipping market {side} order for {amount} {symbol}.{RESET}"
        )
        return None

    if amount <= 0:
        lg.error(f"Invalid amount for market order: {amount}. Must be positive.")
        return None

    order_params = params or {}
    # Ensure position mode is set for Bybit (One-way default, can be overridden in params if needed)
    if "bybit" in exchange.id.lower() and "positionIdx" not in order_params:
        # 0 for one-way mode, 1 for hedge mode long, 2 for hedge mode short
        order_params["positionIdx"] = 0  # Default to one-way mode
        lg.debug("Setting default Bybit positionIdx=0 (One-way mode) for market order.")

    # Add category for Bybit V5
    if "bybit" in exchange.id.lower() and "category" not in order_params:
        category = "linear" if market_info.get("linear", True) else "inverse"
        order_params["category"] = category
        lg.debug(f"Setting Bybit category='{category}' for market order.")

    order_desc = f"{side.upper()} {amount} {symbol} MARKET"
    lg.info(f"Attempting to place order: {order_desc} with params: {order_params}")

    try:
        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=float(amount),  # ccxt usually expects float for amount
            params=order_params,
        )
        lg.info(f"{NEON_GREEN}Successfully placed market order: {order_desc}. Order ID: {order.get('id')}{RESET}")
        lg.debug(f"Market order details: {order}")
        return order
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place market order {order_desc}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid market order parameters for {order_desc}: {e}. Check size/limits/params.{RESET}")
    except ccxt.OrderImmediatelyFillable as e:
        lg.warning(
            f"{NEON_YELLOW}Market order {order_desc} likely filled immediately (exchange response): {e}. Assuming success.{RESET}"
        )
        # Try to fetch the order status or assume success for market orders
        # For simplicity, we'll return a placeholder indicating likely success
        return {
            "id": "assumed_filled",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "status": "filled",
            "info": str(e),
        }
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error placing market order {order_desc}: {e}{RESET}", exc_info=True)
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing market order {order_desc}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing market order {order_desc}: {e}{RESET}", exc_info=True)

    return None


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_params: dict | None = None,
    current_position: dict | None = None,  # Pass current position to avoid redundant fetches
    logger: logging.Logger = None,
) -> bool:
    """Sets or modifies Stop Loss (SL), Take Profit (TP), and potentially Trailing Stop Loss (TSL)
    for an existing position using ccxt's unified `edit_position` or exchange-specific methods.
    Handles Bybit V5 API specifics.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        market_info: Market info dictionary.
        stop_loss_price: The desired SL price (Decimal). Set to 0 or None to remove.
        take_profit_price: The desired TP price (Decimal). Set to 0 or None to remove.
        trailing_stop_params: Dictionary with TSL parameters (e.g., {'trailingStop': '0.005', 'activePrice': '...'})
                              Set to None to potentially remove TSL if API supports it, or leave existing TSL.
        current_position: Optional pre-fetched current position dictionary.
        logger: Logger instance.

    Returns:
        True if the protection was successfully set/modified, False otherwise.
    """
    lg = logger or logging.getLogger(__name__)
    if not CONFIG.get("enable_trading", False):
        lg.warning(f"{NEON_YELLOW}TRADING DISABLED. Skipping setting protection for {symbol}.{RESET}")
        return False

    # --- Fetch current position if not provided ---
    position = current_position
    if position is None:
        position = get_open_position(exchange, symbol, lg)
        if position is None:
            lg.warning(f"Cannot set protection for {symbol}: No open position found.")
            # Return True because there's no position to protect, not an error state.
            return True

    position_side = position.get("side")  # 'long' or 'short'
    position_size_str = position.get("contracts", position.get("info", {}).get("size"))
    if not position_side or position_size_str is None:
        lg.error(
            f"Could not determine side or size of position for {symbol}. Cannot set protection. Position: {position}"
        )
        return False
    position_size = Decimal(str(position_size_str))
    if abs(position_size) <= Decimal("1e-9"):
        lg.info(f"Position size for {symbol} is zero or negligible. No protection needed.")
        return True  # No protection needed for zero size

    # --- Prepare Parameters for API Call ---
    params = {}
    sl_tp_set_parts = []  # For logging

    # Determine Bybit category and positionIdx
    if "bybit" in exchange.id.lower():
        category = "linear" if market_info.get("linear", True) else "inverse"
        params["category"] = category
        # Position Index: 0 for one-way, 1 for Buy hedge, 2 for Sell hedge
        # Assuming one-way mode is used, otherwise this needs adjustment based on actual mode
        params["positionIdx"] = position.get("info", {}).get("positionIdx", 0)  # Use existing or default to 0
        lg.debug(
            f"Using Bybit params for set protection: category={params['category']}, positionIdx={params['positionIdx']}"
        )

    # Add Stop Loss
    if stop_loss_price is not None:
        sl_float = float(stop_loss_price) if stop_loss_price > 0 else 0  # Use 0 to remove SL on some exchanges
        params["stopLoss"] = sl_float  # Unified param name
        params["stopLossPrice"] = sl_float  # Sometimes needed explicitly
        # Bybit V5 specific param (check if needed, usually stopLoss works)
        # params['stopLoss'] = str(stop_loss_price) if stop_loss_price > 0 else '0'
        sl_tp_set_parts.append(
            f"SL={stop_loss_price:.{market_info['precision'].get('price', 4)}f}" if sl_float > 0 else "SL=Remove"
        )

    # Add Take Profit
    if take_profit_price is not None:
        tp_float = float(take_profit_price) if take_profit_price > 0 else 0
        params["takeProfit"] = tp_float  # Unified param name
        params["takeProfitPrice"] = tp_float  # Sometimes needed explicitly
        # Bybit V5 specific param
        # params['takeProfit'] = str(take_profit_price) if take_profit_price > 0 else '0'
        sl_tp_set_parts.append(
            f"TP={take_profit_price:.{market_info['precision'].get('price', 4)}f}" if tp_float > 0 else "TP=Remove"
        )

    # Add Trailing Stop Loss (Handle Bybit V5 specifics)
    if trailing_stop_params is not None:
        if "bybit" in exchange.id.lower():
            # Bybit V5 expects string values for TSL parameters
            tsl_distance = trailing_stop_params.get(
                "trailingStop"
            )  # e.g., '50' (price distance) or '0.005' (percentage)
            tsl_activation = trailing_stop_params.get("activePrice")  # e.g., '29500'

            if tsl_distance is not None:
                # Ensure it's a string, format if Decimal/float
                if isinstance(tsl_distance, (Decimal, float)):
                    # Determine precision needed (e.g., for percentage)
                    # Heuristic: more decimals if < 1, fewer if large number
                    prec = 5 if abs(float(tsl_distance)) < 1 else 2
                    tsl_distance_str = f"{Decimal(str(tsl_distance)):.{prec}f}"
                else:
                    tsl_distance_str = str(tsl_distance)

                # Use '0' to remove TSL if distance is effectively zero
                if Decimal(tsl_distance_str) <= 0:
                    params["tpslMode"] = "Full"  # Required when setting TP/SL without TSL
                    params["trailingStop"] = "0"
                    sl_tp_set_parts.append("TSL=Remove")
                else:
                    params["trailingStop"] = tsl_distance_str
                    sl_tp_set_parts.append(f"TSL={tsl_distance_str}")
                    # Activation price is optional, only include if provided and > 0
                    if tsl_activation is not None:
                        if isinstance(tsl_activation, (Decimal, float)):
                            tsl_activation_str = (
                                f"{Decimal(str(tsl_activation)):.{market_info['precision'].get('price', 4)}f}"
                            )
                        else:
                            tsl_activation_str = str(tsl_activation)

                        if Decimal(tsl_activation_str) > 0:
                            params["activePrice"] = tsl_activation_str
                            sl_tp_set_parts.append(f"TSL_Act={tsl_activation_str}")
                        else:
                            # Remove activation price if set to 0/empty
                            params.pop("activePrice", None)
                    # Bybit requires tpslMode=Partial when TSL is active with TP/SL
                    # If only TSL is set, Full mode might be required? Test this.
                    # Safest is Partial if any TP/SL is also being set. Assume Full if only TSL.
                    if stop_loss_price or take_profit_price:
                        params["tpslMode"] = "Partial"  # TP/SL take precedence
                    else:
                        params["tpslMode"] = "Full"  # Assumed for TSL only, verify if needed

            # Ensure tpslMode is set if TP or SL is set, even without TSL change
            elif (stop_loss_price is not None or take_profit_price is not None) and "tpslMode" not in params:
                # If existing position had TSL, we might need Partial, otherwise Full.
                # Check existing TSL from position info
                existing_tsl = position.get("info", {}).get("trailingStop", "0")
                if Decimal(str(existing_tsl)) > 0:
                    params["tpslMode"] = "Partial"
                    # Also need to resubmit existing TSL values if only changing TP/SL
                    params["trailingStop"] = str(existing_tsl)
                    existing_act_price = position.get("info", {}).get("activePrice", "0")
                    if Decimal(str(existing_act_price)) > 0:
                        params["activePrice"] = str(existing_act_price)
                    lg.debug("Setting tpslMode=Partial and resubmitting existing TSL due to TP/SL change.")
                else:
                    params["tpslMode"] = "Full"
                    lg.debug("Setting tpslMode=Full as only TP/SL are being modified.")

        else:
            lg.warning(
                f"Trailing stop parameters provided, but handling is not implemented for exchange {exchange.id}. Skipping TSL."
            )
            # Unified TSL support in ccxt is limited, often needs exchange-specific params

    # Check if any parameters were actually prepared
    if not params or len(sl_tp_set_parts) == 0:
        lg.info(f"No protection parameters provided or changed for {symbol}. No action taken.")
        return True  # No action needed, considered success

    set_desc = f"Set Protection for {symbol} ({position_side.upper()} {position_size}): {', '.join(sl_tp_set_parts)}"
    lg.info(f"Attempting: {set_desc} with params: {params}")

    # --- API Call ---
    try:
        # Use edit_position if available (preferred unified method)
        if exchange.has.get("editPosition"):
            lg.debug("Using exchange.edit_position...")
            response = exchange.edit_position(symbol, params=params)
        # Fallback to edit_order (might work for modifying SL/TP associated with position) - less common
        # elif exchange.has.get('editOrder'):
        #     lg.warning("edit_position not available, attempting fallback with edit_order (may not work as expected).")
        #     # Requires an order ID associated with the position, which is complex to manage reliably.
        #     # Need a way to link SL/TP orders back to the main position entry.
        #     # For now, we'll rely on methods that modify the position directly.
        #     lg.error("edit_order fallback for position protection not implemented.")
        #     return False
        # Fallback to set_leverage (Bybit uses this endpoint for SL/TP/TSL)
        elif "bybit" in exchange.id.lower() and exchange.has.get("setLeverage"):
            lg.debug("Using exchange.set_leverage (Bybit endpoint for SL/TP)...")
            # setLeverage needs leverage param, but we don't want to change it.
            # Fetch current leverage and pass it along.
            current_leverage = position.get("leverage", position.get("info", {}).get("leverage"))
            if current_leverage is None:
                lg.error("Could not determine current leverage for Bybit setLeverage call.")
                return False
            # Note: setLeverage requires leverage argument, even if not changing it.
            response = exchange.set_leverage(leverage=float(current_leverage), symbol=symbol, params=params)
        # Fallback to private POST with endpoint (Example for Bybit V5)
        elif "bybit" in exchange.id.lower():
            lg.debug("Using private POST /v5/position/trading-stop (Bybit specific)...")
            # Endpoint: /v5/position/trading-stop
            # Required params: category, symbol, stopLoss, takeProfit, tpslMode, slTriggerBy, tpTriggerBy
            # Optional: trailingStop, activePrice, tpLimitPrice, slLimitPrice, slOrderType, tpOrderType
            bybit_params = {
                "category": params["category"],
                "symbol": symbol,
                # Trigger prices by Mark Price is common
                "tpTriggerBy": "MarkPrice",
                "slTriggerBy": "MarkPrice",
                # Ensure SL/TP are strings, use '0' if not set or removing
                "stopLoss": str(params.get("stopLoss", "0")),
                "takeProfit": str(params.get("takeProfit", "0")),
                "tpslMode": params.get("tpslMode", "Full"),  # Default to Full if not specified
            }
            if "trailingStop" in params and Decimal(params["trailingStop"]) > 0:
                bybit_params["trailingStop"] = params["trailingStop"]
                if "activePrice" in params:
                    bybit_params["activePrice"] = params["activePrice"]

            lg.debug(f"Bybit V5 Request Params: {bybit_params}")
            response = exchange.private_post("/v5/position/trading-stop", bybit_params)
        else:
            lg.error(
                f"Cannot set position protection: No suitable method found for exchange {exchange.id} (edit_position, setLeverage, or specific endpoint)."
            )
            return False

        lg.info(f"{NEON_GREEN}Successfully executed protection setting request for {symbol}.{RESET}")
        lg.debug(f"Protection setting response: {response}")
        # Basic check: Does the response indicate an error? (Varies by exchange)
        # Bybit V5 usually returns {'retCode': 0, ...} on success
        if isinstance(response, dict):
            ret_code = response.get("retCode", response.get("code"))  # Check common keys for status code
            ret_msg = response.get("retMsg", response.get("msg"))
            if ret_code is not None and int(ret_code) != 0:
                # Check for common non-error codes (e.g., already set, no change)
                # Bybit 110043: Set Take Profit/Stop Loss orders can only be modified or cancelled
                # Bybit 110025: Position idx not match position mode (Warning, but might still work if mode correct)
                # Bybit 34036: The order is not modified; triggers if SL/TP values are the same as existing.
                non_error_codes = [110043, 110025, 34036]
                if int(ret_code) in non_error_codes:
                    lg.warning(
                        f"{NEON_YELLOW}Protection setting for {symbol} resulted in non-critical exchange code {ret_code}: '{ret_msg}'. Assuming no change needed or modification applied.{RESET}"
                    )
                    return True  # Treat as success if it's just an info code
                else:
                    lg.error(
                        f"{NEON_RED}Protection setting failed for {symbol}. Exchange Response Code: {ret_code}, Msg: {ret_msg}{RESET}"
                    )
                    return False
            # If no error code or code is 0, assume success
            return True
        # If response is not a dict, assume success if no exception occurred
        return True

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds error during protection setting for {symbol}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        # This can happen if SL/TP values are invalid (e.g., SL > entry for long)
        lg.error(
            f"{NEON_RED}Invalid order parameters setting protection for {symbol}: {e}. Check SL/TP values relative to current price.{RESET}"
        )
    except ccxt.OrderNotFound as e:
        # Might occur if trying to edit a non-existent SL/TP order via edit_order fallback
        lg.error(f"{NEON_RED}Order not found error setting protection for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Catch specific exchange errors
        # Bybit 110043: "Set Take Profit/Stop Loss orders can only be modified or cancelled" -> Means already set, treat as success?
        # Bybit 110025: "position idx not match position mode" -> Check hedge vs one-way mode
        # Bybit 34036: "The order is not modified" -> SL/TP same as current, treat as success
        err_code = getattr(e, "code", None)
        err_str = str(e)
        non_error_codes = [110043, 110025, 34036]
        if err_code in non_error_codes or "order is not modified" in err_str.lower():
            lg.warning(
                f"{NEON_YELLOW}Protection setting for {symbol} resulted in non-critical exchange error {err_code}: '{err_str}'. Assuming no change needed or modification applied.{RESET}"
            )
            return True  # Treat as success
        else:
            lg.error(f"{NEON_RED}Exchange error setting protection for {symbol}: {e}{RESET}", exc_info=True)
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting protection for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

    return False


# --- Main Trading Loop Function ---
def analyze_and_trade_symbol(
    exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger
) -> None:
    """Performs one cycle of analysis and potential trading for a single symbol.
    Fetches data, calculates indicators, generates signals, manages positions,
    and handles risk management (SL/TP/BE/TSL).

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: The trading symbol to analyze.
        config: The configuration dictionary.
        logger: The logger instance for this symbol.
    """
    lg = logger
    lg.info(f"--- Starting analysis cycle for {symbol} ---")

    # --- 1. Fetch Market Info ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"Could not retrieve market info for {symbol}. Skipping cycle.")
        return  # Cannot proceed without market info

    # --- 2. Fetch Kline Data ---
    interval_str = config.get("interval", "5")  # User-friendly interval (e.g., "5")
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(interval_str)
    if not ccxt_timeframe:
        lg.error(f"Invalid interval '{interval_str}' in config. Check VALID_INTERVALS. Skipping cycle.")
        return

    # Determine lookback based on longest requirement + buffer
    # Reuse logic from TradingAnalyzer init for consistency
    strat_lookbacks = [
        config.get("volbot_length", DEFAULT_VOLBOT_LENGTH),
        config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
        config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
        config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
        + config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
        + 1,
        config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
        + config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
        + 1,
    ]
    risk_lookback = config.get("atr_period", DEFAULT_ATR_PERIOD)
    kline_limit = max(strat_lookbacks + [risk_lookback]) + 100  # Add generous buffer

    df_klines = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, limit=kline_limit, logger=lg)
    if df_klines.empty or len(df_klines) < 50:  # Check for minimum reasonable data points
        lg.warning(
            f"{NEON_YELLOW}Insufficient kline data received for {symbol} (got {len(df_klines)}, needed ~{kline_limit}). Skipping analysis.{RESET}"
        )
        return

    # --- 3. Initialize Analyzer & Calculate Indicators ---
    analyzer = TradingAnalyzer(df=df_klines, logger=lg, config=config, market_info=market_info)
    if analyzer.df_processed.empty or not analyzer.strategy_state:
        lg.error(f"Failed to calculate indicators or update state for {symbol}. Skipping further analysis.")
        return

    # --- 4. Check Current Position ---
    current_position = get_open_position(exchange, symbol, lg)
    has_open_position = current_position is not None
    position_side = current_position.get("side") if has_open_position else None  # 'long' or 'short'
    position_entry_price_str = (
        current_position.get("entryPrice", current_position.get("info", {}).get("avgPrice"))
        if has_open_position
        else None
    )
    position_size_str = (
        current_position.get("contracts", current_position.get("info", {}).get("size")) if has_open_position else None
    )
    position_entry_price = Decimal(str(position_entry_price_str)) if position_entry_price_str else None
    position_size = Decimal(str(position_size_str)) if position_size_str else Decimal("0")

    # --- 5. Fetch Current Price ---
    # Use ticker fetch as primary source for latest price
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        # Fallback to last close from klines if ticker fails
        last_close = analyzer.strategy_state.get("close")
        if last_close and pd.notna(last_close):
            current_price = Decimal(str(last_close))
            lg.warning(
                f"{NEON_YELLOW}Failed to fetch ticker price, using last kline close price ({current_price}) for decisions.{RESET}"
            )
        else:
            lg.error(
                f"{NEON_RED}Failed to fetch current price for {symbol} via ticker and klines. Cannot proceed.{RESET}"
            )
            return

    # --- 6. Generate Trading Signal ---
    signal = analyzer.generate_trading_signal()  # "BUY", "SELL", or "HOLD"

    # --- 7. Position Management Logic ---
    trading_enabled = config.get("enable_trading", False)
    config.get("max_concurrent_positions", 1)  # Max allowed by config for this symbol

    # == Scenario 1: Currently IN a position ==
    if has_open_position:
        lg.info(
            f"Managing existing {position_side.upper()} position ({position_size} {symbol} @ {position_entry_price}). Current Price: {current_price}"
        )

        # --- 7a. Check for Exit Signal ---
        exit_signal = False
        if position_side == "long" and signal == "SELL":
            lg.info(f"{COLOR_DN}Exit Signal: SELL signal received while in LONG position. Closing position.{RESET}")
            exit_signal = True
        elif position_side == "short" and signal == "BUY":
            lg.info(f"{COLOR_UP}Exit Signal: BUY signal received while in SHORT position. Closing position.{RESET}")
            exit_signal = True

        if exit_signal:
            if trading_enabled:
                close_side = "sell" if position_side == "long" else "buy"
                close_order = place_market_order(
                    exchange, symbol, close_side, abs(position_size), market_info, lg, params={"reduceOnly": True}
                )
                if close_order:
                    lg.info(f"{NEON_GREEN}Successfully closed {position_side} position based on exit signal.{RESET}")
                    # Optional: Cancel all open SL/TP orders for this symbol after closing
                    try:
                        exchange.cancel_all_orders(
                            symbol, params={"orderFilter": "StopOrder"}
                        )  # Bybit example for TP/SL orders
                        lg.info(f"Cancelled pending Stop/TP orders for {symbol} after closing position.")
                    except Exception as cancel_err:
                        lg.warning(f"Could not cancel stop orders for {symbol} after closing: {cancel_err}")
                else:
                    lg.error(f"{NEON_RED}Failed to place market order to close {position_side} position.{RESET}")
                return  # End cycle after attempting to close
            else:
                lg.warning(
                    f"{NEON_YELLOW}TRADING DISABLED: Would have closed {position_side} position due to exit signal.{RESET}"
                )
                # Continue to risk management even if trading disabled, to log potential actions
                pass  # Fall through to risk management checks

        # --- 7b. Risk Management (SL/TP/BE/TSL) - Only if no exit signal ---
        if not exit_signal:
            lg.info("Performing risk management checks for open position...")
            new_sl_price, _new_tp_price, new_tsl_params = None, None, None
            needs_update = False

            # Get current SL/TP/TSL from position data (parsed in get_open_position)
            current_sl_str = current_position.get("stopLossPrice")
            current_tp_str = current_position.get("takeProfitPrice")
            current_tsl_val = current_position.get("trailingStopLoss")
            current_tsl_act = current_position.get("tslActivationPrice")

            current_sl = Decimal(str(current_sl_str)) if current_sl_str and Decimal(str(current_sl_str)) > 0 else None
            current_tp = Decimal(str(current_tp_str)) if current_tp_str and Decimal(str(current_tp_str)) > 0 else None
            # Use Decimal for TSL value check too
            has_current_tsl = current_tsl_val and Decimal(str(current_tsl_val)) > 0

            lg.debug(
                f"Current Protection: SL={current_sl}, TP={current_tp}, TSL Active={has_current_tsl} (Val={current_tsl_val}, Act={current_tsl_act})"
            )

            # --- i. Break-Even Logic ---
            enable_be = config.get("enable_break_even", True)
            if enable_be and current_sl != position_entry_price:  # Only trigger BE if SL isn't already at entry
                be_trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                risk_atr_val = analyzer.strategy_state.get("atr_risk")
                min_tick = analyzer.get_min_tick_size()

                if risk_atr_val and pd.notna(risk_atr_val) and risk_atr_val > 0 and min_tick > 0:
                    risk_atr = Decimal(str(risk_atr_val))
                    profit_target_for_be = risk_atr * be_trigger_multiple
                    current_profit = Decimal("0")
                    if position_side == "long":
                        current_profit = current_price - position_entry_price
                    else:
                        current_profit = position_entry_price - current_price

                    if current_profit >= profit_target_for_be:
                        # Calculate BE price with offset
                        offset_amount = min_tick * be_offset_ticks
                        be_price = (
                            position_entry_price + offset_amount
                            if position_side == "long"
                            else position_entry_price - offset_amount
                        )
                        # Round BE price to tick size (away from entry)
                        rounding = ROUND_UP if position_side == "long" else ROUND_DOWN
                        be_price_rounded = (be_price / min_tick).quantize(Decimal("1"), rounding=rounding) * min_tick

                        # Check if new BE SL is more favorable than current SL
                        sl_price_to_compare = (
                            current_sl
                            if current_sl
                            else (Decimal("-Infinity") if position_side == "long" else Decimal("Infinity"))
                        )
                        if (position_side == "long" and be_price_rounded > sl_price_to_compare) or (
                            position_side == "short" and be_price_rounded < sl_price_to_compare
                        ):
                            lg.info(
                                f"{NEON_GREEN}Break-Even Triggered! Profit {current_profit:.4f} >= Target {profit_target_for_be:.4f}. Moving SL.{RESET}"
                            )
                            lg.info(f"  Old SL: {current_sl}. New BE SL: {be_price_rounded}")
                            new_sl_price = be_price_rounded
                            needs_update = True
                        else:
                            lg.debug(
                                f"BE condition met, but calculated BE SL ({be_price_rounded}) not better than current SL ({current_sl}). No BE move."
                            )
                    else:
                        lg.debug(
                            f"BE not triggered. Current Profit: {current_profit:.4f}, Target: {profit_target_for_be:.4f}"
                        )
                else:
                    lg.warning(f"Cannot calculate BE: Invalid Risk ATR ({risk_atr_val}) or Min Tick ({min_tick}).")

            # --- ii. Trailing Stop Loss Logic (Exchange-based) ---
            # We primarily set the TSL on entry. This section could potentially update it if needed,
            # but exchange-based TSL usually manages itself after activation.
            # We mainly check if it *should* be active and hasn't been set.
            enable_tsl = config.get("enable_trailing_stop", True)
            if enable_tsl and not has_current_tsl:  # Check if TSL should be set but isn't
                tsl_callback_rate = config.get("trailing_stop_callback_rate", 0.005)  # e.g., 0.5%
                tsl_activation_perc = config.get("trailing_stop_activation_percentage", 0.003)  # e.g., 0.3%

                # Check if activation profit reached (only relevant if TSL not yet active)
                current_profit_perc = Decimal("0")
                if position_entry_price > 0:
                    if position_side == "long":
                        current_profit_perc = (current_price / position_entry_price) - 1
                    else:
                        current_profit_perc = 1 - (current_price / position_entry_price)

                if current_profit_perc >= Decimal(str(tsl_activation_perc)):
                    lg.info(
                        f"{NEON_GREEN}Trailing Stop activation condition met (Profit {current_profit_perc:.2%} >= {tsl_activation_perc:.2%}). Ensuring TSL is set.{RESET}"
                    )

                    # Calculate activation price (optional for some exchanges, Bybit uses it)
                    # If percentage activation, the activation price is entry * (1 +/- activation_perc)
                    act_price = None
                    if tsl_activation_perc > 0:
                        if position_side == "long":
                            act_price_raw = position_entry_price * (1 + Decimal(str(tsl_activation_perc)))
                            act_price = (act_price_raw / analyzer.get_min_tick_size()).quantize(
                                Decimal("1"), rounding=ROUND_UP
                            ) * analyzer.get_min_tick_size()
                        else:
                            act_price_raw = position_entry_price * (1 - Decimal(str(tsl_activation_perc)))
                            act_price = (act_price_raw / analyzer.get_min_tick_size()).quantize(
                                Decimal("1"), rounding=ROUND_DOWN
                            ) * analyzer.get_min_tick_size()

                    # Prepare TSL params (use percentage for Bybit if callback < 1)
                    # Bybit V5: Use percentage string directly if rate < 1
                    tsl_value_str = (
                        str(tsl_callback_rate * 100) if tsl_callback_rate < 1 else str(tsl_callback_rate)
                    )  # Convert rate to percentage for Bybit if needed? Check API docs. Bybit might expect price distance or percent string. Assume rate IS the value/percent string needed.
                    tsl_value_str = str(tsl_callback_rate)  # Keep as configured for now

                    new_tsl_params = {"trailingStop": tsl_value_str}
                    if act_price is not None:
                        new_tsl_params["activePrice"] = str(act_price)

                    lg.info(f"  Prepared TSL Params: {new_tsl_params}")
                    needs_update = True
                else:
                    lg.debug(
                        f"TSL activation condition not met (Profit {current_profit_perc:.2%} < {tsl_activation_perc:.2%})"
                    )

            # --- iii. Update Protection if Needed ---
            if needs_update and trading_enabled:
                # Pass existing TP if not changing, pass new SL, pass new TSL params
                # If BE moved SL, use new_sl_price. If TSL activated, use current_sl (unless BE also triggered)
                final_sl_to_set = new_sl_price if new_sl_price is not None else current_sl
                final_tp_to_set = current_tp  # TP generally doesn't change unless strategy dictates

                # Important: If BE sets SL, disable TSL? Or let TSL take over? Configurable?
                # Current logic: If BE sets SL, we pass that SL. If TSL activates, we pass TSL params.
                # If both happen, BE SL might override TSL activation intention depending on API call structure.
                # Let's prioritize BE: if new_sl_price is set (by BE), don't pass new_tsl_params unless specifically designed.
                final_tsl_params = new_tsl_params
                if new_sl_price is not None and new_tsl_params is not None:
                    lg.warning(
                        f"Both Break-Even SL ({new_sl_price}) and TSL activation ({new_tsl_params}) triggered. Prioritizing BE SL. TSL might need manual check/reset if desired."
                    )
                    final_tsl_params = (
                        None  # Prioritize BE move, potentially removing TSL if params are {'trailingStop':'0'}
                    )

                lg.info(
                    f"Updating position protection: SL={final_sl_to_set}, TP={final_tp_to_set}, TSL Params={final_tsl_params}"
                )
                success = _set_position_protection(
                    exchange,
                    symbol,
                    market_info,
                    stop_loss_price=final_sl_to_set,
                    take_profit_price=final_tp_to_set,
                    trailing_stop_params=final_tsl_params,
                    current_position=current_position,  # Pass position to avoid refetch
                    logger=lg,
                )
                if success:
                    lg.info(f"{NEON_GREEN}Successfully updated position protection.{RESET}")
                else:
                    lg.error(f"{NEON_RED}Failed to update position protection.{RESET}")
            elif needs_update and not trading_enabled:
                lg.warning(
                    f"{NEON_YELLOW}TRADING DISABLED: Would have updated protection (SL={new_sl_price}, TSL Params={new_tsl_params}).{RESET}"
                )

    # == Scenario 2: Currently OUT of a position ==
    else:  # Not in a position
        lg.info(f"No open position for {symbol}. Checking for entry signals.")

        # --- 7c. Check for Entry Signal ---
        if signal in ["BUY", "SELL"]:
            lg.info(f"Entry Signal Detected: {signal}. Current Price: {current_price}")

            # Fetch available balance for sizing
            quote_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if quote_balance is None or quote_balance <= 0:
                lg.error(
                    f"{NEON_RED}Cannot enter trade: Failed to fetch valid {QUOTE_CURRENCY} balance or balance is zero.{RESET}"
                )
                return

            # Calculate SL/TP based on potential entry (current price as estimate)
            # Use current_price for initial calculation, actual entry might differ slightly
            entry_price_est = current_price
            _, potential_tp, potential_sl = analyzer.calculate_entry_tp_sl(entry_price_est, signal)

            if potential_sl is None:
                lg.error(
                    f"{NEON_RED}Cannot enter {signal} trade: Failed to calculate a valid Stop Loss based on current ATR and price.{RESET}"
                )
                return  # Cannot proceed without a valid SL for sizing

            # Calculate position size
            risk_per_trade = config.get("risk_per_trade", 0.01)
            position_size = calculate_position_size(
                balance=quote_balance,
                risk_per_trade=risk_per_trade,
                initial_stop_loss_price=potential_sl,
                entry_price=entry_price_est,
                market_info=market_info,
                exchange=exchange,
                logger=lg,
            )

            if position_size is None or position_size <= 0:
                lg.error(
                    f"{NEON_RED}Cannot enter {signal} trade: Position size calculation failed or resulted in zero/negative size.{RESET}"
                )
                return

            # --- Execute Entry Order ---
            if trading_enabled:
                lg.info(
                    f"Attempting to enter {signal} position: Size={position_size}, Est. Entry={entry_price_est}, SL={potential_sl}, TP={potential_tp}"
                )

                # Set leverage if applicable (only for contracts)
                if market_info.get("is_contract", False):
                    desired_leverage = config.get("leverage", 10)
                    try:
                        lg.info(f"Setting leverage to {desired_leverage}x for {symbol}")
                        # Use set_leverage if available, otherwise might need params in order
                        if exchange.has.get("setLeverage"):
                            # Bybit V5 needs buyLeverage and sellLeverage if setting individually
                            params = {"buyLeverage": float(desired_leverage), "sellLeverage": float(desired_leverage)}
                            if "bybit" in exchange.id.lower():
                                params["category"] = "linear" if market_info.get("linear", True) else "inverse"
                            exchange.set_leverage(desired_leverage, symbol, params=params)
                        else:
                            lg.warning(
                                f"Exchange {exchange.id} does not support setLeverage. Leverage might need manual setup or order params."
                            )
                    except ccxt.ExchangeError as e:
                        # Handle common leverage errors (e.g., already set, open orders/positions)
                        err_str = str(e).lower()
                        if "leverage not modified" in err_str or "same leverage" in err_str:
                            lg.info(f"Leverage for {symbol} already set to {desired_leverage}x.")
                        elif "position open" in err_str or "order active" in err_str:
                            lg.warning(
                                f"{NEON_YELLOW}Cannot set leverage: Active position or orders exist for {symbol}. Proceeding with existing leverage.{RESET}"
                            )
                        else:
                            lg.error(f"{NEON_RED}Failed to set leverage for {symbol}: {e}{RESET}")
                            # Decide whether to proceed without setting leverage or abort
                            # return # Abort if leverage setting is critical and failed unexpectedly
                    except Exception as e:
                        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)
                        # return # Abort if leverage setting is critical

                # Place the market order
                entry_order = place_market_order(exchange, symbol, signal.lower(), position_size, market_info, lg)

                if entry_order and entry_order.get("id"):
                    lg.info(
                        f"{NEON_GREEN}Successfully placed {signal} market order (ID: {entry_order.get('id')}). Size: {position_size}{RESET}"
                    )

                    # --- Set Initial SL/TP/TSL after entry ---
                    # Allow time for order fill and position update on exchange
                    time.sleep(config.get("retry_delay", 5))  # Wait a few seconds

                    # Prepare TSL parameters if enabled
                    tsl_params = None
                    if config.get("enable_trailing_stop", True):
                        tsl_callback_rate = config.get("trailing_stop_callback_rate", 0.005)
                        tsl_activation_perc = config.get("trailing_stop_activation_percentage", 0.003)
                        # For initial setting, activation price is usually not needed unless > 0% activation AND exchange requires it upfront
                        # Bybit V5: can set TSL distance without activation price (activates immediately or when profit allows)
                        tsl_value_str = str(tsl_callback_rate)
                        tsl_params = {"trailingStop": tsl_value_str}
                        # Optionally calculate and add activation price if needed by API and config requires > 0% activation
                        # This part might need refinement based on specific exchange API behavior for initial TSL setting
                        lg.info(f"Prepared initial TSL params: {tsl_params}")

                    lg.info(
                        f"Setting initial protection: SL={potential_sl}, TP={potential_tp}, TSL={tsl_params is not None}"
                    )
                    protection_success = _set_position_protection(
                        exchange,
                        symbol,
                        market_info,
                        stop_loss_price=potential_sl,
                        take_profit_price=potential_tp,
                        trailing_stop_params=tsl_params,
                        # Fetch position again here to ensure we use updated data after entry
                        current_position=None,  # Force refetch
                        logger=lg,
                    )
                    if protection_success:
                        lg.info(f"{NEON_GREEN}Successfully set initial protection for new {signal} position.{RESET}")
                    else:
                        lg.error(
                            f"{NEON_RED}Failed to set initial protection after entering {signal} position. Manual check recommended!{RESET}"
                        )

                else:
                    lg.error(f"{NEON_RED}Failed to place entry market order for {signal}.{RESET}")
            else:
                lg.warning(
                    f"{NEON_YELLOW}TRADING DISABLED: Would have entered {signal} position. Size={position_size}, Est. Entry={entry_price_est}, SL={potential_sl}, TP={potential_tp}{RESET}"
                )

        # --- 7d. No Signal, No Position ---
        else:  # signal == "HOLD" and not has_open_position
            lg.info("Signal is HOLD and no open position. No trading action taken.")

    lg.info(f"--- Finished analysis cycle for {symbol} ---")


# --- Main Execution ---
def main() -> None:
    """Main function to initialize the bot, run the trading loop for specified symbols."""
    # Use a generic init logger for setup messages
    init_logger = setup_logger("init")
    init_logger.info(f"{COLOR_HEADER}--- Volbot Initializing ---{RESET}")
    init_logger.info(f"Using config file: {CONFIG_FILE}")
    init_logger.info(f"Log directory: {LOG_DIRECTORY}")
    init_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"Trading Enabled: {CONFIG.get('enable_trading')}")
    init_logger.info(f"Using Sandbox: {CONFIG.get('use_sandbox')}")
    init_logger.info(f"Default Interval: {CONFIG.get('interval')}")

    # Initialize Exchange
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Exchange initialization failed. Exiting.{RESET}")
        return

    # --- Symbol Selection and Validation ---
    symbols_to_trade: list[str] = []
    while True:
        symbol_input = input(
            f"Enter trading symbol (e.g., BTC/USDT:USDT or BTC/USDT, 'all' for all linear contracts, or press Enter to start with current list {symbols_to_trade}): "
        ).strip()
        if not symbol_input and symbols_to_trade:
            init_logger.info(f"Starting analysis with symbols: {', '.join(symbols_to_trade)}")
            break
        elif not symbol_input and not symbols_to_trade:
            continue
        elif symbol_input.lower() == "all":
            init_logger.info("Attempting to fetch all linear contract symbols...")
            try:
                all_markets = exchange.load_markets()
                # Filter for active linear perpetual swaps (common use case)
                linear_swaps = [
                    mkt["symbol"]
                    for mkt in all_markets.values()
                    if mkt.get("active", True)
                    and mkt.get("linear", False)
                    and mkt.get("swap", True)
                    # Optional: Filter by quote currency if needed
                    and mkt.get("quote", "").upper() == QUOTE_CURRENCY
                ]
                if not linear_swaps:
                    init_logger.error(
                        f"{NEON_RED}No active linear {QUOTE_CURRENCY} perpetual swaps found on {exchange.id}. Please enter symbols manually.{RESET}"
                    )
                    continue
                symbols_to_trade = linear_swaps
                init_logger.info(
                    f"Selected {len(symbols_to_trade)} linear {QUOTE_CURRENCY} swaps. Starting analysis..."
                )
                break
            except Exception as e:
                init_logger.error(
                    f"{NEON_RED}Error fetching or filtering markets: {e}. Please enter symbols manually.{RESET}"
                )
                continue
        else:
            # Validate the entered symbol
            market_info = get_market_info(exchange, symbol_input, init_logger)
            if market_info:
                if symbol_input not in symbols_to_trade:
                    symbols_to_trade.append(symbol_input)
                else:
                    pass
            else:
                # get_market_info already logged the error
                pass

    if not symbols_to_trade:
        init_logger.critical(f"{NEON_RED}No valid symbols selected. Exiting.{RESET}")
        return

    # --- Setup Loggers for Each Symbol ---
    symbol_loggers: dict[str, logging.Logger] = {}
    for sym in symbols_to_trade:
        symbol_loggers[sym] = setup_logger(sym)  # Use symbol-specific log files

    # --- Main Trading Loop ---
    init_logger.info(f"{COLOR_HEADER}--- Starting Main Trading Loop ({len(symbols_to_trade)} symbols) ---{RESET}")
    try:
        while True:
            loop_start_time = time.monotonic()
            now = datetime.now(TIMEZONE)
            init_logger.info(f"*** New Cycle Start: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ***")

            for symbol in symbols_to_trade:
                symbol_logger = symbol_loggers[symbol]
                try:
                    analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except ccxt.AuthenticationError as e:
                    symbol_logger.critical(
                        f"{NEON_RED}CRITICAL: Authentication Error for {symbol}: {e}. Check API keys/permissions. Stopping analysis for this symbol.{RESET}"
                    )
                    # Potentially remove symbol from list or stop the bot entirely
                    # For now, just log critically and continue with others
                except ccxt.NetworkError as e:
                    symbol_logger.error(
                        f"{NEON_RED}Network Error during analysis for {symbol}: {e}. Will retry next cycle.{RESET}"
                    )
                except ccxt.ExchangeError as e:
                    symbol_logger.error(
                        f"{NEON_RED}Exchange Error during analysis for {symbol}: {e}.{RESET}", exc_info=True
                    )
                except Exception as e:
                    symbol_logger.error(
                        f"{NEON_RED}Unhandled Exception during analysis for {symbol}: {e}{RESET}", exc_info=True
                    )

            # --- Loop Delay ---
            loop_end_time = time.monotonic()
            cycle_duration = loop_end_time - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
            init_logger.info(f"Cycle duration: {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        init_logger.info(f"{NEON_YELLOW}KeyboardInterrupt received. Shutting down gracefully...{RESET}")
        # Optional: Add cleanup logic here (e.g., close open positions, cancel orders)
        if CONFIG.get("enable_trading", False):
            init_logger.warning("Trading was enabled. Consider manually checking open positions/orders.")
    except Exception as e:
        init_logger.critical(f"{NEON_RED}--- FATAL UNHANDLED ERROR IN MAIN LOOP ---{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Error: {e}{RESET}")
    finally:
        init_logger.info(f"{COLOR_HEADER}--- Volbot Shut Down ---{RESET}")
        logging.shutdown()  # Ensure all log handlers are closed properly


if __name__ == "__main__":
    main()
