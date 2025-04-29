# livebot7.1.py
# Enhanced version focusing on stop-loss/take-profit mechanisms,
# including break-even logic and MA cross exit condition.
# Improvements: Robust error handling, Decimal precision, Bybit V5 specifics,
# enhanced logging, clearer structure, type hinting, config validation.

import argparse
import contextlib
import json
import logging
import math
import os
import sys
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
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

# Initialize colorama and set Decimal precision
getcontext().prec = 38  # Increased precision for financial calculations
init(autoreset=True)
load_dotenv()

# --- Constants ---
# Neon Color Scheme for Logging
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# API Credentials (Loaded from .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit(1)  # Exit if keys are missing

# Configuration and Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"

# Timezone for logging and display (adjust as needed)
DEFAULT_TIMEZONE_NAME = "America/Chicago"  # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
try:
    TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_NAME)
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# API Call Settings
MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Base delay between retries (may increase for rate limits)
# HTTP status codes considered retryable (e.g., rate limits, temporary server issues)
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]

# Timeframe Mapping
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]  # Intervals supported by the bot's logic
CCXT_INTERVAL_MAP = {  # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Default Indicator Periods (can be overridden by config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14  # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 12     # Window for underlying RSI in StochRSI
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0  # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

# Other Constants
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 15  # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY = 10  # Seconds to wait after placing order before checking position status
# QUOTE_CURRENCY is dynamically loaded from config

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Ensure keys are strings before replacing
        api_key_str = str(API_KEY) if API_KEY else ""
        api_secret_str = str(API_SECRET) if API_SECRET else ""
        if api_key_str:
            msg = msg.replace(api_key_str, "***API_KEY***")
        if api_secret_str:
            msg = msg.replace(api_secret_str, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
    and ensuring all default keys are present recursively. Validates types.
    """
    default_config = {
        "interval": "5",  # Default to 5 minute interval (string format for our logic)
        "retry_delay": RETRY_DELAY_SECONDS,  # Use constant
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
        "orderbook_limit": 25,  # Depth of orderbook to fetch
        "signal_score_threshold": 1.5,  # Score needed to trigger BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8,  # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7,  # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,  # How much higher volume needs to be than MA
        "scalping_signal_threshold": 2.5,  # Separate threshold for 'scalping' weight set
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False,  # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet), disable consciously
        "risk_per_trade": 0.01,  # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions": 1,  # Limit open positions for this symbol (common strategy) - Currently informational, not enforced
        "quote_currency": "USDT",  # Currency for balance check and sizing
        # --- MA Cross Exit Config ---
        "enable_ma_cross_exit": True,  # Enable closing position on adverse EMA cross
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True,  # Default to enabling TSL (exchange TSL)
        # Trail distance as a percentage of the activation/high-water-mark price (e.g., 0.5%)
        # Bybit API expects absolute distance, this percentage is used for calculation.
        "trailing_stop_callback_rate": 0.005,  # Example: 0.5% trail distance relative to entry/activation
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry.
        "trailing_stop_activation_percentage": 0.003,  # Example: Activate when 0.3% in profit
        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        # Move SL when profit (in price points) reaches X * Current ATR
        "break_even_trigger_atr_multiple": 1.0,  # Example: Trigger BE when profit = 1x ATR
        # Place BE SL this many minimum price increments (ticks) beyond entry price
        # E.g., 2 ticks to cover potential commission/slippage on exit
        "break_even_offset_ticks": 2,
        # --- End Protection Config ---
        "indicators": {  # Control which indicators are calculated and contribute to score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,  # Flag to enable fetching and scoring orderbook data
        },
        "weight_sets": {  # Define different weighting strategies
            "scalping": {  # Example weighting for a fast scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": {  # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default"  # Choose which weight set to use ("default" or "scalping")
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            return default_config
        except OSError:
            return default_config  # Return default config anyway if file creation fails

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively and validate types
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added or types corrected during the update
            if updated_config != config_from_file:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                except OSError:
                    pass
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError):
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
        except OSError:
            pass
        return default_config


def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any]) -> dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config.
    Attempts to convert loaded values to the default type if they differ.
    """
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Check type mismatch and attempt conversion
        elif not isinstance(updated_config.get(key), type(default_value)):
            try:
                # Attempt to convert loaded value to the default type
                converted_value = type(default_value)(updated_config[key])
                updated_config[key] = converted_value
                # Check if conversion might lose precision (e.g., float to int) - keep original if so?
                # For simplicity, we accept the conversion for now.

            except (ValueError, TypeError):
                updated_config[key] = default_value

    return updated_config


# Load configuration globally
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")  # Get quote currency from config
# Global variable for console log level, can be changed by args
console_log_level = logging.INFO


# --- Logger Setup ---
def setup_logger(symbol: str, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    global console_log_level
    console_log_level = level  # Update global level

    # Clean symbol for filename (replace / and : which are invalid in filenames)
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"livebot_{safe_symbol}"  # Use safe symbol in logger name
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists
    # Check if handlers exist and update levels if needed
    if logger.hasHandlers():
        handlers_updated = False
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level != console_log_level:
                handler.setLevel(console_log_level)  # Update console level
                handlers_updated = True
            # Optionally update file handler level if needed, though typically stays DEBUG
        if handlers_updated:
            logger.debug(f"Updated existing logger handler levels for {logger_name}.")
        return logger

    # Set base logging level to DEBUG to capture everything internally
    logger.setLevel(logging.DEBUG)

    # File Handler (writes DEBUG and above, includes line numbers)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Add line number to file logs for easier debugging
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to the file
        logger.addHandler(file_handler)
    except Exception:
        pass

    # Stream Handler (console, level set by argument or default INFO)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'  # Add timestamp format to console
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs)
    logger.propagate = False

    logger.info(f"Logger '{logger_name}' initialized. Console Level: {logging.getLevelName(console_log_level)}, File Level: DEBUG.")
    return logger


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with error handling and validation."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'linear',  # Assume linear contracts (USDT margined) - adjust if needed
                'adjustForTimeDifference': True,  # Auto-sync time with server
                # Connection timeouts (milliseconds) - Increased slightly
                'fetchTickerTimeout': 15000,  # 15 seconds
                'fetchBalanceTimeout': 20000,  # 20 seconds
                'createOrderTimeout': 25000,  # Timeout for placing orders
                'fetchOrderTimeout': 20000,  # Timeout for fetching orders
                'fetchPositionsTimeout': 20000,  # Timeout for fetching positions
                # Add any exchange-specific options if needed
                # 'recvWindow': 10000, # Example for Binance if needed
                'brokerId': 'livebot71enhanced',  # Example: Add a broker ID for Bybit if desired
            }
        }

        # Select Bybit class
        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            logger.warning(f"{NEON_RED}USING LIVE TRADING ENVIRONMENT{RESET}")

        # Test connection by fetching markets (essential for market info)
        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"Markets loaded for {exchange.id}.")

        # Test API credentials and permissions by fetching balance
        # Specify account type for Bybit V5 (CONTRACT for linear/USDT, UNIFIED if using that)
        account_type_to_test = 'CONTRACT'  # Or 'UNIFIED' based on your account setup
        logger.info(f"Attempting initial balance fetch (Using enhanced function, checking Account Type: {account_type_to_test})...")
        try:
            # Use our enhanced balance function
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, logger)  # Enhanced function handles account types
            if balance_decimal is not None:
                 logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 logger.warning(f"{NEON_YELLOW}Initial balance fetch returned None. Check API permissions, account type (CONTRACT/UNIFIED), and ensure funds exist in the correct account type ({QUOTE_CURRENCY}).{RESET}")
                 # Still return exchange object, but warn user
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None  # Critical failure, cannot proceed
        except ccxt.ExchangeError as balance_err:
            # Handle potential errors if account type is wrong etc.
            logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")
            # Check if error suggests wrong account type
            if "account type" in str(balance_err).lower():
                logger.warning(f"{NEON_YELLOW} >> Hint: The error might indicate the wrong account type was specified or used by default. Ensure API key is for the correct account (CONTRACT or UNIFIED) holding {QUOTE_CURRENCY}.{RESET}")
        except Exception as balance_err:
            logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")

        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        logger.error(f"{NEON_RED}>> Check API keys, permissions, IP whitelist, and Real/Testnet selection.{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None


# --- CCXT Data Fetching ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks, retries, and Decimal precision."""
    lg = logger
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})...")
            ticker = exchange.fetch_ticker(symbol)
            lg.debug(f"Ticker data for {symbol}: {ticker}")

            price = None
            last_price = ticker.get('last')
            bid_price = ticker.get('bid')
            ask_price = ticker.get('ask')

            # Helper to parse and validate price string to positive Decimal
            def parse_price(price_str: Any, field_name: str) -> Decimal | None:
                if price_str is None: return None
                try:
                    price_dec = Decimal(str(price_str))
                    if price_dec > 0:
                        return price_dec
                    else:
                        lg.warning(f"Ticker field '{field_name}' ({price_dec}) is not positive for {symbol}.")
                        return None
                except (InvalidOperation, ValueError, TypeError) as e:
                    lg.warning(f"Could not parse ticker field '{field_name}' ({price_str}) for {symbol}: {e}")
                    return None

            # 1. Try 'last' price first
            price = parse_price(last_price, 'last')
            if price:
                lg.debug(f"Using 'last' price for {symbol}: {price}")

            # 2. If 'last' is invalid, try bid/ask midpoint
            if price is None and bid_price is not None and ask_price is not None:
                bid_decimal = parse_price(bid_price, 'bid')
                ask_decimal = parse_price(ask_price, 'ask')
                if bid_decimal and ask_decimal:
                    # Ensure bid is not higher than ask (can happen in volatile/illiquid moments)
                    if bid_decimal <= ask_decimal:
                        price = (bid_decimal + ask_decimal) / Decimal('2')
                        lg.debug(f"Using bid/ask midpoint for {symbol}: {price} (Bid: {bid_decimal}, Ask: {ask_decimal})")
                    else:
                        lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' as fallback.")
                        price = ask_decimal  # Use ask as a safer fallback in this case
                elif ask_decimal:  # If only ask is valid, use ask
                    price = ask_decimal
                    lg.warning(f"Using 'ask' price as fallback (bid invalid) for {symbol}: {price}")
                elif bid_decimal:  # If only bid is valid, use bid
                    price = bid_decimal
                    lg.warning(f"Using 'bid' price as fallback (ask invalid) for {symbol}: {price}")

            # 3. If midpoint fails or wasn't used, try ask price
            if price is None and ask_price is not None:
                ask_decimal = parse_price(ask_price, 'ask')
                if ask_decimal:
                    price = ask_decimal
                    lg.warning(f"Using 'ask' price as fallback for {symbol}: {price}")

            # 4. If ask fails, try bid price
            if price is None and bid_price is not None:
                bid_decimal = parse_price(bid_price, 'bid')
                if bid_decimal:
                    price = bid_decimal
                    lg.warning(f"Using 'bid' price as fallback for {symbol}: {price}")

            # --- Final Check ---
            if price is not None and price > 0:
                return price
            else:
                # If we got here after successful fetch but no valid price, don't retry immediately unless specific error
                lg.error(f"{NEON_RED}Failed to find a valid positive current price for {symbol} from ticker data.{RESET}")
                return None  # No valid price found in this attempt's data

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ReadTimeout) as e:
            lg.warning(f"Network error fetching price for {symbol} (Attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES:
                lg.warning(f"Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached fetching price for {symbol} after network errors.{RESET}")
                return None
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 2  # Default wait time
            try:  # Try to parse recommended wait time
                 if 'try again in' in str(e).lower():
                     wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                     wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                 elif 'rate limit' in str(e).lower():
                     import re
                     match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                     if match:
                         num = int(match.group(1))
                         unit = match.group(2)
                         wait_time = num / 1000 if unit == 'ms' else num
                         wait_time = max(1, int(wait_time + 1))
            except Exception as parse_err:
                 lg.debug(f"Could not parse wait time from rate limit error: {parse_err}")
            lg.warning(f"Rate limit exceeded fetching price for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Decide if retryable. Some exchange errors are permanent (bad symbol).
            # Let's not retry on generic ExchangeError for price fetch unless known retryable code.
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Don't retry unexpected errors

    # If loop finishes without returning a price
    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after all retries.{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries, basic validation, and proper formatting."""
    lg = logger or logging.getLogger(__name__)
    try:
        if not exchange.has['fetchOHLCV']:
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if ohlcv is not None and len(ohlcv) > 0:  # Basic check if data was returned and not empty
                    break  # Success
                else:
                    lg.warning(f"fetch_ohlcv returned {type(ohlcv)} (length {len(ohlcv) if ohlcv is not None else 'N/A'}) for {symbol} (Attempt {attempt + 1}). Retrying...")
                    # Optional: Add a small delay even on None/empty return if it might be transient
                    time.sleep(1)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}")
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                    return pd.DataFrame()  # Return empty DataFrame on final failure
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 5  # Default wait time
                try:  # Try to parse recommended wait time
                     if 'try again in' in str(e).lower():
                         wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                         wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                     elif 'rate limit' in str(e).lower():
                         import re
                         match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                         if match:
                             num = int(match.group(1))
                             unit = match.group(2)
                             wait_time = num / 1000 if unit == 'ms' else num
                             wait_time = max(1, int(wait_time + 1))
                except Exception as parse_err:
                    lg.debug(f"Could not parse wait time from rate limit error: {parse_err}")
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                # Depending on the error, might not be retryable (e.g., invalid timeframe/symbol)
                return pd.DataFrame()  # Return empty DataFrame on non-retryable exchange errors

        if not ohlcv:  # Check if list is empty or still None after retries
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} immediately after creation.{RESET}")
            return df

        # Convert timestamp to datetime and set as index
        try:
            # Ensure timestamps are integers/floats before conversion
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid timestamps after conversion
        except Exception as ts_err:
            lg.error(f"Error converting timestamp column for {symbol}: {ts_err}. Columns: {df.columns}, Head:\n{df.head()}")
            return pd.DataFrame()

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame empty after timestamp conversion/validation for {symbol} {timeframe}.{RESET}")
            return df
        df.set_index('timestamp', inplace=True)

        # Ensure numeric types for OHLCV, coerce errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with any NaN in critical price columns
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Drop rows with non-positive close price (invalid data)
        df = df[df['close'] > 0]
        # Optional: Handle NaN volumes (e.g., fill with 0 or drop rows)
        df['volume'].fillna(0, inplace=True)  # Fill NaN volumes with 0 for calculations
        # Optional: Drop rows with zero volume if it indicates bad data (can be market specific)
        # df = df[df['volume'] > 0]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return pd.DataFrame()

        # Optional: Sort index just in case data isn't perfectly ordered (though fetch_ohlcv usually is)
        df.sort_index(inplace=True)

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> dict | None:
    """Fetch orderbook data using ccxt with retries, basic validation, and clearer logging."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.has['fetchOrderBook']:
                lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            orderbook = exchange.fetch_order_book(symbol, limit=limit)

            # --- Validation ---
            if not orderbook:
                lg.warning(f"fetch_order_book returned None or empty data for {symbol} (Attempt {attempts + 1}).")
                # Continue to retry logic
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"{NEON_YELLOW}Invalid orderbook structure (bids/asks not lists) for {symbol}. Attempt {attempts + 1}. Response: {orderbook}{RESET}")
                 # Treat as potentially retryable issue
            elif not orderbook['bids'] and not orderbook['asks']:
                 # Exchange might return empty lists if orderbook is thin or during low liquidity
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 # Return the empty book, signal generation needs to handle this
                 # Ensure structure is minimal ccxt standard
                 return {
                    'bids': [],
                    'asks': [],
                    'timestamp': orderbook.get('timestamp'),
                    'datetime': orderbook.get('datetime'),
                    'nonce': orderbook.get('nonce'),
                    'symbol': symbol
                 }
            else:
                 # Looks valid
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5  # Default wait time
            try:  # Try to parse recommended wait time
                 if 'try again in' in str(e).lower():
                     wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                     wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                 elif 'rate limit' in str(e).lower():
                     import re
                     match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                     if match:
                         num = int(match.group(1))
                         unit = match.group(2)
                         wait_time = num / 1000 if unit == 'ms' else num
                         wait_time = max(1, int(wait_time + 1))
            except Exception as parse_err:
                 lg.debug(f"Could not parse wait time from rate limit error: {parse_err}")
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time}s... (Attempt {attempts + 1})")
            time.sleep(wait_time)  # Use delay from error msg if possible
            # Increment attempt counter here so it doesn't bypass retry limit due to sleep
            attempts += 1
            continue  # Skip the standard delay at the end
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            # Don't retry on definitive exchange errors (e.g., bad symbol) unless specifically handled
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Don't retry on unexpected errors

        # Increment attempt counter and wait before retrying network/validation issues
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            lg.warning(f"Retrying orderbook fetch in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None


# --- Trading Analyzer Class (Using pandas_ta) ---
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta and generates weighted signals."""

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],  # Pass market info for precision etc.
    ) -> None:
        """Initializes the analyzer.

        Args:
            df: DataFrame with OHLCV data, indexed by timestamp.
            logger: Logger instance.
            config: Configuration dictionary.
            market_info: Market information dictionary from CCXT.
        """
        self.df = df  # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")
        self.indicator_values: dict[str, float] = {}  # Stores latest indicator float values
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}  # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: dict[str, Decimal] = {}  # Stores calculated fib levels (as Decimals)
        self.ta_column_names: dict[str, str | None] = {}  # Stores actual column names generated by pandas_ta

        if not self.weights:
            logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will not work.")

        # Calculate indicators immediately on initialization
        self._calculate_all_indicators()
        # Update latest values immediately after calculation
        self._update_latest_indicator_values()
        # Calculate Fibonacci levels (can be done after indicators)
        self.calculate_fibonacci_levels()

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> str | None:
        """Helper to find the actual column name generated by pandas_ta, handling variations."""
        # Common prefixes/suffixes used by pandas_ta
        # Examples: ATRr_14, EMA_10, MOM_10, CCI_20_0.015, WILLR_14, MFI_14, VWAP,
        # PSARl_0.02_0.2, PSARs_0.02_0.2, SMA_10, STOCHRSIk_14_14_3_3, STOCHRSId_14_14_3_3,
        # RSI_14, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, VOL_SMA_15 (custom)
        cfg = self.config  # Shortcut
        # Ensure config values used for matching are correct type (int/float)
        try:
            bb_std_dev_float = float(cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV))
        except (ValueError, TypeError):
            self.logger.error(f"Invalid bollinger_bands_std_dev format in config: {cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)}. Using default {DEFAULT_BOLLINGER_BANDS_STD_DEV}.")
            bb_std_dev_float = DEFAULT_BOLLINGER_BANDS_STD_DEV

        # Define expected patterns based on default names and config values
        expected_patterns = {
            "ATR": [f"ATRr_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}", f"ATR_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_0.015"],  # pandas-ta often appends the constant
            "Williams_R": [f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP", "VWAP_D"],  # Handle potential suffix like _D for daily reset
            "PSAR_long": [f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev_float:.1f}"],
            "BB_Middle": [f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev_float:.1f}"],
            "BB_Upper": [f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev_float:.1f}"],
            "Volume_MA": [f"VOL_SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"]  # Custom name used in calculation
        }

        patterns_to_check = expected_patterns.get(base_name, [])
        df_columns_lower = [col.lower() for col in result_df.columns]

        for pattern in patterns_to_check:
            if pattern in result_df.columns:
                self.logger.debug(f"Found exact match column '{pattern}' for base '{base_name}'.")
                return pattern
            # Check case-insensitive match
            if pattern.lower() in df_columns_lower:
                match_index = df_columns_lower.index(pattern.lower())
                original_case_col = result_df.columns[match_index]
                self.logger.debug(f"Found column '{original_case_col}' for base '{base_name}' (case-insensitive match).")
                return original_case_col

            # Check variation without const suffix (e.g., CCI_20 instead of CCI_20_0.015)
            base_pattern_parts = pattern.split('_')
            # Check if last part looks like a number (could be float or int)
            if len(base_pattern_parts) > 2 and base_pattern_parts[-1].replace('.', '', 1).isdigit():
                pattern_no_suffix = '_'.join(base_pattern_parts[:-1])
                if pattern_no_suffix in result_df.columns:
                    self.logger.debug(f"Found column '{pattern_no_suffix}' for base '{base_name}' (without const suffix).")
                    return pattern_no_suffix
                if pattern_no_suffix.lower() in df_columns_lower:
                    match_index = df_columns_lower.index(pattern_no_suffix.lower())
                    original_case_col = result_df.columns[match_index]
                    self.logger.debug(f"Found column '{original_case_col}' for base '{base_name}' (case-insensitive match, no suffix).")
                    return original_case_col

        # Fallback: Search for base name (case-insensitive) as prefix or exact match
        # More specific check: starts with base name + underscore (e.g., "CCI_")
        prefix_match = base_name.lower() + "_"
        for col in result_df.columns:
            if col.lower().startswith(prefix_match):
                 self.logger.debug(f"Found column '{col}' for base '{base_name}' using prefix fallback search.")
                 return col
        # Exact match (case-insensitive) - less likely if prefix failed but check anyway
        for i, col_lower in enumerate(df_columns_lower):
            if col_lower == base_name.lower():
                 original_case_col = result_df.columns[i]
                 self.logger.debug(f"Found column '{original_case_col}' for base '{base_name}' using exact match fallback search.")
                 return original_case_col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {result_df.columns.tolist()}")
        return None

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check for sufficient data length based on configured periods
        periods_needed = []
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        # Always calculate ATR as it's used for SL/TP sizing and BE logic
        periods_needed.append(cfg.get("atr_period", DEFAULT_ATR_PERIOD))

        # Check which other indicators need calculating based on config
        if indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit"):  # Need EMAs if alignment OR MA cross exit enabled
            periods_needed.append(cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        if indi_cfg.get("momentum"): periods_needed.append(cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        if indi_cfg.get("cci"): periods_needed.append(cfg.get("cci_window", DEFAULT_CCI_WINDOW))
        if indi_cfg.get("wr"): periods_needed.append(cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        if indi_cfg.get("mfi"): periods_needed.append(cfg.get("mfi_window", DEFAULT_MFI_WINDOW))
        if indi_cfg.get("sma_10"): periods_needed.append(cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW))
        if indi_cfg.get("stoch_rsi"): periods_needed.append(cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + cfg.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))  # StochRSI needs underlying RSI window too
        if indi_cfg.get("rsi"): periods_needed.append(cfg.get("rsi_period", DEFAULT_RSI_WINDOW))
        if indi_cfg.get("bollinger_bands"): periods_needed.append(cfg.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        if indi_cfg.get("volume_confirmation"): periods_needed.append(cfg.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
        # VWAP doesn't have a fixed period in the same way, relies on session/day start. PSAR also doesn't have a 'period' dependency like others.
        if indi_cfg.get("fibonacci"): periods_needed.append(cfg.get("fibonacci_window", DEFAULT_FIB_WINDOW))  # If using Fib levels from TA

        # Calculate minimum required data length (max period + some buffer)
        min_required_data = max(periods_needed) + 20 if periods_needed else 50  # Add buffer (e.g., 20 bars)

        if len(self.df) < min_required_data:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate or NaN.{RESET}")
             # Continue calculation, but expect NaNs

        try:
            df_calc = self.df.copy()  # Work on a copy to avoid modifying original DF passed in

            # --- Calculate indicators using pandas_ta ---
            indicators_config = self.config.get("indicators", {})
            calculate_emas = indicators_config.get("ema_alignment", False) or self.config.get("enable_ma_cross_exit", False)

            # Always calculate ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # Calculate other indicators based on config flags
            if calculate_emas:  # Calculate EMAs if needed for alignment or MA cross exit
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
                # VWAP calculation might depend on the frequency (e.g., daily reset)
                # pandas_ta vwap usually assumes daily reset based on timestamp index
                df_calc.ta.vwap(append=True)
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                if psar_result is not None and not psar_result.empty:
                    # Append safely, avoiding duplicate columns if they somehow exist
                    for col in psar_result.columns:
                        if col not in df_calc.columns:
                            df_calc[col] = psar_result[col]
                        else:
                             self.logger.debug(f"Column {col} from PSAR result already exists in DataFrame. Overwriting.")
                             df_calc[col] = psar_result[col]  # Overwrite if exists
                    self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                    self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)
                else:
                    self.logger.warning(f"PSAR calculation returned empty result for {self.symbol}.")

            if indicators_config.get("sma_10", False):
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                if stochrsi_result is not None and not stochrsi_result.empty:
                    # Append safely
                    for col in stochrsi_result.columns:
                        if col not in df_calc.columns:
                            df_calc[col] = stochrsi_result[col]
                        else:
                             self.logger.debug(f"Column {col} from StochRSI result already exists. Overwriting.")
                             df_calc[col] = stochrsi_result[col]
                    self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                    self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)
                else:
                    self.logger.warning(f"StochRSI calculation returned empty result for {self.symbol}.")

            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
                try:
                    bb_std_float = float(bb_std)
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid bollinger_bands_std_dev format in config: {bb_std}. Using default {DEFAULT_BOLLINGER_BANDS_STD_DEV}.")
                    bb_std_float = DEFAULT_BOLLINGER_BANDS_STD_DEV

                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std_float)
                if bbands_result is not None and not bbands_result.empty:
                     # Append safely
                    for col in bbands_result.columns:
                        if col not in df_calc.columns:
                            df_calc[col] = bbands_result[col]
                        else:
                             self.logger.debug(f"Column {col} from BBands result already exists. Overwriting.")
                             df_calc[col] = bbands_result[col]
                    self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                    self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                    self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)
                else:
                    self.logger.warning(f"Bollinger Bands calculation returned empty result for {self.symbol}.")

            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}"  # Custom name
                # Calculate SMA on volume column, handle potential NaNs in volume
                # Ensure volume exists and handle potential initial NaNs in the MA result
                if 'volume' in df_calc.columns:
                    df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)  # Use ta.sma directly
                    self.ta_column_names["Volume_MA"] = vol_ma_col_name
                else:
                    self.logger.warning(f"Volume column missing in DataFrame, cannot calculate Volume MA for {self.symbol}.")

            # Assign the df with calculated indicators back to self.df
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
            # Common error if pandas_ta is not installed or method name changes
            self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} (check pandas_ta method name/version/installation?): {e}{RESET}", exc_info=True)
             # self.df remains the original data without calculated indicators
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)
            # Decide how to handle - clear df or keep original? Keep original for now.

        # Note: _update_latest_indicator_values is called after this in __init__

    def _update_latest_indicator_values(self) -> None:
        """Updates the indicator_values dict with the latest float values from self.df."""
        # Define placeholder for all potential keys to ensure dict structure consistency
        potential_keys = list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]
        nan_placeholder = dict.fromkeys(potential_keys, np.nan)

        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame is empty for {self.symbol}.")
            self.indicator_values = nan_placeholder
            return

        try:
            latest = self.df.iloc[-1]  # Get the last row (Series)
            if latest.isnull().all():
                self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: Last row of DataFrame contains all NaNs for {self.symbol}.{RESET}")
                self.indicator_values = nan_placeholder
                return
        except IndexError:
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty or too short.")
             self.indicator_values = nan_placeholder  # Reset values
             return

        try:
            updated_values = {}

            # Use the dynamically stored column names from self.ta_column_names
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index:  # Check if column name exists and is valid
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            updated_values[key] = float(value)  # Store as float
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) to float for {self.symbol}.")
                            updated_values[key] = np.nan
                    else:
                        updated_values[key] = np.nan  # Value is NaN in DataFrame
                else:
                    # If col_name is None or not in index, store NaN
                    # Only log if the indicator was intended to be calculated
                    indicator_enabled_or_needed = False
                    if key == "ATR": indicator_enabled_or_needed = True  # Always needed
                    elif key == "EMA_Short" or key == "EMA_Long": indicator_enabled_or_needed = self.config.get("indicators", {}).get("ema_alignment", False) or self.config.get("enable_ma_cross_exit", False)
                    else:
                        # Map internal key back to config key
                        config_key_map = {
                            "Momentum": "momentum", "CCI": "cci", "Williams_R": "wr", "MFI": "mfi", "VWAP": "vwap",
                            "PSAR_long": "psar", "PSAR_short": "psar", "SMA10": "sma_10", "StochRSI_K": "stoch_rsi",
                            "StochRSI_D": "stoch_rsi", "RSI": "rsi", "BB_Lower": "bollinger_bands",
                            "BB_Middle": "bollinger_bands", "BB_Upper": "bollinger_bands", "Volume_MA": "volume_confirmation"
                        }
                        config_key = config_key_map.get(key)
                        if config_key:
                            indicator_enabled_or_needed = self.config.get("indicators", {}).get(config_key, False)

                    if indicator_enabled_or_needed:  # Only log if it was supposed to be calculated
                        self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found or invalid in latest data for {self.symbol}. Storing NaN.")
                    updated_values[key] = np.nan

            # Add essential price/volume data from the original DataFrame columns
            for base_col in ['close', 'volume', 'high', 'low']:
                value = latest.get(base_col, np.nan)
                key_name = base_col.capitalize()  # e.g., 'Close'
                if pd.notna(value):
                    try:
                        updated_values[key_name] = float(value)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert base value for '{base_col}' ({value}) to float for {self.symbol}.")
                        updated_values[key_name] = np.nan
                else:
                    updated_values[key_name] = np.nan

            # Ensure all potential keys are present, filling with NaN if missing
            for p_key in potential_keys:
                if p_key not in updated_values:
                    updated_values[p_key] = np.nan

            self.indicator_values = updated_values
            # Filter out NaN for debug log brevity
            valid_values = {k: f"{v:.5f}" if isinstance(v, float) else v for k, v in self.indicator_values.items() if pd.notna(v)}
            self.logger.debug(f"Latest indicator float values updated for {self.symbol}: {valid_values}")

        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = nan_placeholder  # Reset values

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a specified window using Decimal and market tick size."""
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
                min_tick = None  # Disable quantization if tick size is invalid

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level: High - (Range * Pct) assumes retracement from High towards Low
                    level_price = (high - (diff * Decimal(str(level_pct))))

                    # Quantize the result to the market's price precision (using tick size) if possible
                    if min_tick:
                        # Round level price DOWN to the nearest tick size (for potential support levels)
                        level_price_quantized = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    else:
                        level_price_quantized = level_price  # Use raw value if quantization fails

                    levels[level_name] = level_price_quantized
            else:
                 # If high == low, all levels will be the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}. Setting levels to high/low.")
                 if min_tick > 0:
                     # Round down to be safe
                     level_price_quantized = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 else:
                     level_price_quantized = high  # Use raw if quantization fails
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            # Log levels formatted as strings with appropriate precision
            price_precision_digits = self.get_price_precision()
            log_levels = {k: f"{v:.{price_precision_digits}f}" for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { log_levels }")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info, using min_tick_size as primary source."""
        try:
            min_tick = self.get_min_tick_size()  # Rely on tick size first
            if min_tick > 0:
                # Calculate decimal places from tick size
                # normalize() removes trailing zeros, as_tuple().exponent gives power of 10
                # Use max(0, ...) to handle cases like tick size '1' resulting in exponent 0
                precision = max(0, abs(min_tick.normalize().as_tuple().exponent))
                # self.logger.debug(f"Derived price precision {precision} from min tick size {min_tick} for {self.symbol}")
                return precision
            else:
                 self.logger.warning(f"Min tick size ({min_tick}) is invalid for {self.symbol}. Attempting fallback precision methods.")

        except Exception as e:
            self.logger.warning(f"Error getting/using min tick size for precision derivation ({self.symbol}): {e}. Attempting fallback methods.")

        # Fallback 1: Infer from ccxt market precision.price if it's an integer (representing decimal places)
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if isinstance(price_precision_val, int):
                # CCXT sometimes uses integer for decimal places directly
                self.logger.debug(f"Using price precision from market_info.precision.price (integer): {price_precision_val}")
                return price_precision_val
        except Exception as e:
            self.logger.debug(f"Could not get integer precision from market_info.precision.price: {e}")

        # Fallback 2: Infer from last close price format if tick size failed
        try:
            last_close = self.indicator_values.get("Close")  # Uses float value
            if last_close and pd.notna(last_close) and last_close > 0:
                try:
                    # Use Decimal formatting to handle scientific notation and get accurate decimal places
                    s_close = format(Decimal(str(last_close)), 'f')  # Format to standard decimal string
                    if '.' in s_close:
                        # Count digits after decimal, removing trailing zeros
                        precision = len(s_close.split('.')[-1].rstrip('0'))
                        self.logger.debug(f"Inferring price precision from last close price ({s_close}) as {precision} for {self.symbol}.")
                        return precision
                    else:
                        return 0  # No decimal places if integer
                except Exception as e_close:
                    self.logger.warning(f"Error inferring precision from close price {last_close}: {e_close}")
        except Exception as e_outer:
            self.logger.warning(f"Could not access/parse close price for precision fallback: {e_outer}")

        # Default fallback precision if all else fails
        default_precision = 4  # Common default for USDT pairs, adjust if needed
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal, with improved fallbacks."""
        try:
            # 1. Check market_info['precision']['tick'] (often explicitly provided)
            precision_info = self.market_info.get('precision', {})
            tick_val = precision_info.get('tick')  # Check for explicit 'tick' key first
            if tick_val is not None:
                try:
                    tick_size = Decimal(str(tick_val))
                    if tick_size > 0:
                        # self.logger.debug(f"Using tick size from precision.tick: {tick_size} for {self.symbol}")
                        return tick_size
                except (InvalidOperation, ValueError, TypeError):
                     self.logger.debug(f"Could not parse precision.tick '{tick_val}' as Decimal for {self.symbol}. Trying precision.price.")

            # 2. Check market_info['precision']['price'] (ccxt standard, often the tick size as float/str)
            price_precision_val = precision_info.get('price')  # This is usually the tick size
            if price_precision_val is not None:
                try:
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0:
                        # self.logger.debug(f"Using tick size from precision.price: {tick_size} for {self.symbol}")
                        return tick_size
                except (InvalidOperation, ValueError, TypeError):
                    self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as Decimal for {self.symbol}. Trying other methods.")

            # 3. Check Bybit specific 'tickSize' in info (V5)
            info_dict = self.market_info.get('info', {})
            bybit_tick_size = info_dict.get('tickSize')
            if bybit_tick_size is not None:
                 try:
                      tick_size_bybit = Decimal(str(bybit_tick_size))
                      if tick_size_bybit > 0:
                          # self.logger.debug(f"Using tick size from market_info['info']['tickSize']: {tick_size_bybit} for {self.symbol}")
                          return tick_size_bybit
                 except (InvalidOperation, ValueError, TypeError):
                      self.logger.debug(f"Could not parse info['tickSize'] '{bybit_tick_size}' as Decimal for {self.symbol}.")

            # 4. Fallback: Check limits.price.min (sometimes this represents tick size, less reliable)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                try:
                    # Heuristic: If min price is very small (e.g., < 1), it might be the tick size
                    min_tick_from_limit = Decimal(str(min_price_val))
                    # Add a check: if min_price is a power of 10 (like 0.1, 0.01), it's likely tick size
                    is_power_of_10 = min_tick_from_limit.log10().is_integer() if min_tick_from_limit > 0 else False
                    if min_tick_from_limit > 0 and (min_tick_from_limit < 1 or is_power_of_10):
                        self.logger.debug(f"Using tick size derived from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                        return min_tick_from_limit
                except (InvalidOperation, ValueError, TypeError):
                    self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' as Decimal for {self.symbol}.")

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using default fallback.")

        # Absolute fallback: A very small number if everything else fails
        fallback_tick = Decimal('0.00000001')  # Small default, ensure it's less than likely prices
        self.logger.warning(f"{NEON_YELLOW}Could not reliably determine tick size for {self.symbol}. Using extremely small fallback tick size: {fallback_tick}. Price quantization may be inaccurate.{RESET}")
        return fallback_tick

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels (name, price) to the current price."""
        if not self.fib_levels_data:
            # self.logger.debug(f"Fibonacci levels not calculated or empty for {self.symbol}. Cannot find nearest.")
            return []

        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal):  # Ensure level is Decimal
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    self.logger.warning(f"Non-decimal value found in fib_levels_data for {self.symbol}: {name}={level_price} ({type(level_price)})")

            # Sort by distance (ascending)
            level_distances.sort(key=lambda x: x['distance'])

            # Return the names and levels of the nearest N
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score based on latest values. Returns float score or NaN."""
        # Relies on 'EMA_Short', 'EMA_Long', 'Close' being in self.indicator_values
        ema_short = self.indicator_values.get("EMA_Short", np.nan)
        ema_long = self.indicator_values.get("EMA_Long", np.nan)
        current_price = self.indicator_values.get("Close", np.nan)

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            # self.logger.debug(f"EMA alignment check skipped for {self.symbol}: Missing required values (EMA_Short={ema_short}, EMA_Long={ema_long}, Close={current_price}).")
            return np.nan  # Return NaN if data is missing

        # Bullish alignment: Price > Short EMA > Long EMA
        if current_price > ema_short > ema_long:
            return 1.0
        # Bearish alignment: Price < Short EMA < Long EMA
        elif current_price < ema_short < ema_long:
            return -1.0
        # Other cases are neutral or mixed signals
        else:
            return 0.0

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: dict | None
    ) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0}  # Reset signals
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {}  # For logging individual scores

        # --- Essential Data Checks ---
        if not self.indicator_values:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values dictionary is empty.{RESET}")
            return "HOLD"

        # --- Check if any relevant indicator has a valid value ---
        core_indicators_present = False
        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
            self.logger.error(f"Active weight set '{self.active_weight_set_name}' is missing or empty in config for {self.symbol}. Cannot generate signal.")
            return "HOLD"

        for key, weight_str in active_weights.items():
            try:
                weight = Decimal(str(weight_str))
                # Check only if indicator is enabled, weighted, AND relevant values exist
                if weight != 0 and self.config.get("indicators", {}).get(key, False):
                    # Map indicator key to relevant keys in self.indicator_values
                    value_keys_to_check = []
                    if key == 'ema_alignment': value_keys_to_check = ['EMA_Short', 'EMA_Long', 'Close']
                    elif key == 'momentum': value_keys_to_check = ['Momentum']
                    elif key == 'volume_confirmation': value_keys_to_check = ['Volume', 'Volume_MA']
                    elif key == 'stoch_rsi': value_keys_to_check = ['StochRSI_K', 'StochRSI_D']
                    elif key == 'rsi': value_keys_to_check = ['RSI']
                    elif key == 'bollinger_bands': value_keys_to_check = ['BB_Lower', 'BB_Middle', 'BB_Upper', 'Close']
                    elif key == 'vwap': value_keys_to_check = ['VWAP', 'Close']
                    elif key == 'cci': value_keys_to_check = ['CCI']
                    elif key == 'wr': value_keys_to_check = ['Williams_R']
                    elif key == 'psar': value_keys_to_check = ['PSAR_long', 'PSAR_short']  # Check if either is present
                    elif key == 'sma_10': value_keys_to_check = ['SMA10', 'Close']
                    elif key == 'mfi': value_keys_to_check = ['MFI']
                    # Orderbook checked separately

                    # Check if *at least one* relevant value is not NaN
                    if any(pd.notna(self.indicator_values.get(v_key)) for v_key in value_keys_to_check):
                        core_indicators_present = True
                        break  # Found at least one valid indicator value

                    # Special check for orderbook (only need data if weight > 0)
                    if key == 'orderbook' and orderbook_data and orderbook_data.get('bids') and orderbook_data.get('asks'):
                        core_indicators_present = True
                        break  # Valid orderbook data counts

            except (InvalidOperation, ValueError, TypeError): pass  # Ignore invalid weights

        if not core_indicators_present:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: All score-contributing indicator values are NaN or orderbook data missing/empty.{RESET}")
             return "HOLD"

        if current_price is None or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
            return "HOLD"

        # --- Iterate Through Enabled Indicators with Weights ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue  # Skip disabled indicators

            weight_str = active_weights.get(indicator_key)
            if weight_str is None: continue  # Skip if no weight defined for this enabled indicator

            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue  # Skip if weight is zero
            except (InvalidOperation, ValueError, TypeError):
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping.")
                continue

            # Find and call the check method
            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method = getattr(self, check_method_name)
                indicator_score = np.nan  # Default to NaN
                try:
                    # Pass specific arguments if needed (e.g., orderbook)
                    if indicator_key == "orderbook":
                        # Check if valid OB data exists before calling
                        if orderbook_data and orderbook_data.get('bids') and orderbook_data.get('asks'):
                            indicator_score = method(orderbook_data, current_price)
                        else:
                            if weight != 0:  # Only log if it was expected to contribute
                                self.logger.debug(f"Orderbook data not available or empty for {self.symbol}, skipping orderbook check.")
                            indicator_score = np.nan  # Treat as NaN if data missing/empty
                    else:
                        indicator_score = method()  # Returns float score or np.nan

                except Exception as e:
                    self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                    indicator_score = np.nan  # Treat as NaN on error

                # --- Process Score ---
                # Log score before clamping for debugging visibility
                debug_scores[indicator_key] = f"{indicator_score:.3f}" if pd.notna(indicator_score) else "NaN"
                if pd.notna(indicator_score):
                    try:
                        # Ensure score is a valid float/int before converting to Decimal
                        if not isinstance(indicator_score, (float, int, np.number)):
                            raise TypeError(f"Indicator score for {indicator_key} is not a number: {indicator_score} ({type(indicator_score)})")

                        score_decimal = Decimal(str(indicator_score))  # Convert float score to Decimal
                        # Clamp score between -1 and 1 before applying weight
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_decimal))
                        score_contribution = clamped_score * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight
                        active_indicator_count += 1
                        # self.logger.debug(f"  {indicator_key}: Score={indicator_score:.3f}, Clamped={clamped_score}, Weight={weight}, Contrib={score_contribution:.4f}")
                    except (InvalidOperation, ValueError, TypeError, Exception) as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} ({indicator_score}): {calc_err}")
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1
            else:
                self.logger.warning(f"Check method '{check_method_name}' not found or not callable for enabled/weighted indicator: {indicator_key} ({self.symbol})")

        # --- Determine Final Signal ---
        if total_weight_applied == 0:
            self.logger.warning(f"No indicators contributed to the signal score for {self.symbol} (Total Weight Applied = 0). Defaulting to HOLD.")
            final_signal = "HOLD"
        else:
            # Use threshold defined in config
            try:
                threshold = Decimal(str(self.config.get("signal_score_threshold", 1.5)))
            except (InvalidOperation, ValueError, TypeError):
                 self.logger.error(f"Invalid signal_score_threshold in config: {self.config.get('signal_score_threshold', 1.5)}. Using default 1.5")
                 threshold = Decimal("1.5")

            # Apply threshold to the final weighted score
            if final_signal_score >= threshold:
                final_signal = "BUY"
            elif final_signal_score <= -threshold:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"

        # --- Log Summary ---
        # Format score contributions for logging
        score_details = ", ".join([f"{k}: {v}" for k, v in debug_scores.items()])
        price_precision = self.get_price_precision()  # Get precision for logging current price
        log_msg = (
            f"Signal Calculation Summary ({self.symbol} @ {current_price:.{price_precision}f}):\n"
            f"  Weight Set: {self.active_weight_set_name}\n"
            # f"  Scores: {score_details}\n" # Logged at DEBUG level below
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN)\n"
            f"  Total Weight Applied: {total_weight_applied:.3f}\n"
            f"  Final Weighted Score: {final_signal_score:.4f}\n"
            f"  Signal Threshold: +/- {threshold:.3f}\n"
            f"  ==> Final Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        # Log detailed scores only if console level is DEBUG or lower
        if console_log_level <= logging.DEBUG:
            self.logger.debug(f"  Detailed Scores: {score_details}")

        # Update internal signal state
        if final_signal in self.signals:
            self.signals[final_signal] = 1

        return final_signal

    # --- Indicator Check Methods ---
    # Each method should return a float score between -1.0 and 1.0, or np.nan if data invalid/missing

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment using pre-calculated latest values."""
        # Ensure the indicator was calculated if this check is enabled
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
            # Check if EMAs were supposed to be calculated
            should_calculate = self.config.get("indicators", {}).get("ema_alignment", False) or self.config.get("enable_ma_cross_exit", False)
            if should_calculate:
                self.logger.debug(f"EMA Alignment check skipped for {self.symbol}: EMAs not in indicator_values (maybe calculation failed?).")
            return np.nan
        # calculate_ema_alignment_score handles NaNs internally
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        """Checks Momentum indicator. Score reflects direction and magnitude (clamped)."""
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan
        # Simple scaling: Assume significant momentum is > 0 or < 0.
        # Positive momentum -> bullish bias, negative -> bearish bias.
        # Scale score: e.g., MOM=0.1 -> score=0.5, MOM=-0.1 -> score=-0.5
        # This needs tuning based on typical asset volatility and timeframe.
        # Let's use a simple scaling factor, clamping the result.
        scale_factor = 5.0  # Example: Scales +/- 0.2 to +/- 1.0
        score = momentum * scale_factor
        return max(-1.0, min(1.0, score))  # Clamp score

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume supports potential move (relative to MA). Score is direction-neutral, indicating significance."""
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        try:
            multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))  # Use float
        except (ValueError, TypeError):
            self.logger.error(f"Invalid volume_confirmation_multiplier in config: {self.config.get('volume_confirmation_multiplier', 1.5)}. Using default 1.5")
            multiplier = 1.5

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma <= 0:
            # self.logger.debug("Volume confirmation check skipped: Missing values.")
            return np.nan

        try:
            # High volume indicates significance (could be breakout or exhaustion)
            if current_volume > volume_ma * multiplier:
                # Positive score indicates significance/confirmation, not direction.
                return 0.7  # Strong confirmation/significance
            # Very low volume suggests lack of interest or weak move
            elif current_volume < volume_ma / multiplier:
                # Negative score indicates lack of confirmation/significance
                return -0.4
            else:
                # Neutral volume
                return 0.0
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check calculation failed for {self.symbol}: {e}{RESET}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Checks Stochastic RSI K and D lines for overbought/oversold and crossover signals."""
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan

        try:
            oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
            overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
            if oversold >= overbought:
                self.logger.warning(f"Stoch RSI oversold threshold ({oversold}) >= overbought ({overbought}). Check config. Using defaults 25/75.")
                oversold = 25.0
                overbought = 75.0
        except (ValueError, TypeError):
            self.logger.error("Invalid Stoch RSI threshold format in config. Using defaults (25/75).")
            oversold = 25.0
            overbought = 75.0

        # --- Scoring Logic ---
        score = 0.0
        # 1. Extreme Zones (strongest signals)
        if k < oversold and d < oversold:
            score = 1.0  # Both deep oversold -> Strong bullish signal
        elif k > overbought and d > overbought:
            score = -1.0  # Both deep overbought -> Strong bearish signal

        # 2. K vs D Relationship (momentum indication)
        # Give higher weight to K crossing D than just K being above/below D
        # Use difference as proxy for crossover momentum
        diff = k - d
        cross_threshold = 5.0  # Threshold for significant difference/potential cross (tune this)
        if abs(diff) > cross_threshold:
            if diff > 0:  # K moved above D (or is significantly above) -> Bullish momentum
                # Boost score towards bullish, stronger if not already strongly bearish
                score = max(score, 0.6) if score >= -0.5 else 0.6
            else:  # K moved below D (or is significantly below) -> Bearish momentum
                # Boost score towards bearish, stronger if not already strongly bullish
                score = min(score, -0.6) if score <= 0.5 else -0.6
        else:  # K and D are close or recently crossed
            # Slight bias based on which is higher
            if k > d: score = max(score, 0.2)  # Weak bullish bias
            elif k < d: score = min(score, -0.2)  # Weak bearish bias

        # 3. Consider position within range (0-100) - less important than extremes/crosses
        if oversold <= k <= overbought:  # Inside normal range
            # Optionally scale based on proximity to mid-point (50)
            range_width = overbought - oversold
            if range_width > 0:
                 mid_point = oversold + range_width / 2.0
                 half_range_width = range_width / 2.0
                 if half_range_width > 0:
                     # Scales -1 (at oversold) to +1 (at overbought) within the range
                     mid_range_pos = (k - mid_point) / half_range_width
                     # Map this to score: higher value -> more bearish, lower value -> more bullish
                     # Apply a small weight to this mid-range position score
                     mid_range_score_contribution = -mid_range_pos * 0.3  # Max contribution +/- 0.3
                     # Combine with existing score (simple addition or weighted average)
                     score += mid_range_score_contribution
                 # else: score remains unchanged
            # else: score remains unchanged if range is zero

        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_rsi(self) -> float:
        """Checks RSI indicator for overbought/oversold conditions."""
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan

        # Define thresholds (could be moved to config)
        oversold = 30.0
        overbought = 70.0
        lower_mid = 40.0  # Start leaning bullish below this
        upper_mid = 60.0  # Start leaning bearish above this

        if rsi <= oversold: return 1.0  # Strong bullish signal
        if rsi >= overbought: return -1.0  # Strong bearish signal
        if rsi < lower_mid: return 0.5  # Leaning oversold/bullish
        if rsi > upper_mid: return -0.5  # Leaning overbought/bearish

        # Smoother transition in the middle neutral zone (lower_mid to upper_mid)
        if lower_mid <= rsi <= upper_mid:
            # Linear scale from +0.5 (at lower_mid) down to -0.5 (at upper_mid)
            mid_range = upper_mid - lower_mid
            if mid_range > 0:
                # Calculate position within mid-range (0 at lower_mid, 1 at upper_mid)
                position_in_mid = (rsi - lower_mid) / mid_range
                # Scale score linearly from +0.5 down to -0.5
                score = 0.5 - (position_in_mid * 1.0)
                return score
            else:  # Avoid division by zero if thresholds are equal
                return 0.0  # Neutral if range is zero

        return 0.0  # Fallback (should not be reached if logic is correct)

    def _check_cci(self) -> float:
        """Checks CCI indicator for extremes indicating potential trend reversals."""
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan
        # Define thresholds (common values, could be configurable)
        strong_oversold = -150.0
        strong_overbought = 150.0
        mod_oversold = -80.0
        mod_overbought = 80.0

        if cci <= strong_oversold: return 1.0  # Strong Oversold -> Bullish signal
        if cci >= strong_overbought: return -1.0  # Strong Overbought -> Bearish signal
        if cci < mod_oversold: return 0.6  # Moderately Oversold -> Leaning Bullish
        if cci > mod_overbought: return -0.6  # Moderately Overbought -> Leaning Bearish

        # Trend confirmation near zero line - scale based on value
        # Scale linearly between mod_oversold (+0.6) and mod_overbought (-0.6)
        if mod_oversold <= cci <= mod_overbought:
            mid_range = mod_overbought - mod_oversold
            if mid_range > 0:
                 # Calculate position within mid-range (0 at mod_oversold, 1 at mod_overbought)
                 position_in_mid = (cci - mod_oversold) / mid_range
                 # Scale score linearly from +0.6 down to -0.6
                 score = 0.6 - (position_in_mid * 1.2)  # Total range is 1.2
                 return score
            else:  # Avoid division by zero
                 return 0.0  # Neutral if range is zero

        return 0.0  # Fallback

    def _check_wr(self) -> float:  # Williams %R
        """Checks Williams %R indicator for overbought/oversold conditions."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        # WR range: -100 (most oversold) to 0 (most overbought)
        # Thresholds are typically -80 and -20
        oversold = -80.0
        overbought = -20.0

        if wr <= oversold: return 1.0  # Oversold -> Bullish signal
        if wr >= overbought: return -1.0  # Overbought -> Bearish signal

        # Scale linearly in the middle range (-80 to -20)
        if oversold < wr < overbought:
            mid_range = overbought - oversold  # Should be 60.0
            if mid_range > 0:
                # Calculate position within mid-range (0 at -80, 1 at -20)
                position_in_mid = (wr - oversold) / mid_range
                # Scale score linearly from +1.0 down to -1.0
                score = 1.0 - (position_in_mid * 2.0)  # Total range is 2.0
                return score
            else:  # Should not happen with defined thresholds
                return 0.0  # Neutral if range is zero

        return 0.0  # Fallback (shouldn't be reached with above logic)

    def _check_psar(self) -> float:
        """Checks Parabolic SAR relative to price to determine trend direction."""
        psar_l = self.indicator_values.get("PSAR_long", np.nan)
        psar_s = self.indicator_values.get("PSAR_short", np.nan)
        # PSAR values themselves indicate the stop level.
        # The signal comes from which one is active (non-NaN).
        # If PSAR_long has a value, SAR is below price (uptrend).
        # If PSAR_short has a value, SAR is above price (downtrend).

        if pd.notna(psar_l) and pd.isna(psar_s):
            # PSAR is below price (long value is active) -> Uptrend indication
            return 1.0
        elif pd.notna(psar_s) and pd.isna(psar_l):
            # PSAR is above price (short value is active) -> Downtrend indication
            return -1.0
        elif pd.isna(psar_l) and pd.isna(psar_s):
            # Both NaN (likely start of data, insufficient history)
            # self.logger.debug(f"PSAR state undetermined (both NaN) for {self.symbol}")
            return 0.0  # Neutral or undetermined
        else:
            # Both have values? This shouldn't happen with pandas_ta psar implementation.
            self.logger.warning(f"Ambiguous PSAR state for {self.symbol} (PSAR_long={psar_l}, PSAR_short={psar_s}). Treating as neutral.")
            return 0.0  # Neutral or ambiguous

    def _check_sma_10(self) -> float:  # Example using SMA10
        """Checks price relative to SMA10 as a simple trend indicator."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return np.nan

        # Simple crossover score - magnitude could be added based on distance from SMA
        if last_close > sma_10: return 0.6  # Price above SMA -> Bullish bias
        if last_close < sma_10: return -0.6  # Price below SMA -> Bearish bias
        return 0.0  # Price is exactly on SMA

    def _check_vwap(self) -> float:
        """Checks price relative to VWAP, often used as an intraday mean/support/resistance."""
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return np.nan

        # VWAP acts as a dynamic support/resistance or fair value indicator for the session
        # Score magnitude could be increased based on distance from VWAP
        if last_close > vwap: return 0.7  # Price above VWAP -> Bullish intraday sentiment
        if last_close < vwap: return -0.7  # Price below VWAP -> Bearish intraday sentiment
        return 0.0  # Price is exactly on VWAP

    def _check_mfi(self) -> float:
        """Checks Money Flow Index for overbought/oversold conditions, incorporating volume."""
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan
        # Define thresholds (similar to RSI, but MFI includes volume)
        oversold = 20.0
        overbought = 80.0
        lower_mid = 40.0
        upper_mid = 60.0

        if mfi <= oversold: return 1.0  # Oversold -> Bullish potential (money flowing in despite price drops)
        if mfi >= overbought: return -1.0  # Overbought -> Bearish potential (money flowing out despite price rises)
        if mfi < lower_mid: return 0.4  # Leaning oversold/potential accumulation
        if mfi > upper_mid: return -0.4  # Leaning overbought/potential distribution

        # Scale linearly in the middle neutral range (lower_mid to upper_mid)
        if lower_mid <= mfi <= upper_mid:
            mid_range = upper_mid - lower_mid
            if mid_range > 0:
                 # Scale from +0.4 (at lower_mid) down to -0.4 (at upper_mid)
                 position_in_mid = (mfi - lower_mid) / mid_range
                 score = 0.4 - (position_in_mid * 0.8)  # Total range is 0.8
                 return score
            else:  # Avoid division by zero
                 return 0.0  # Neutral if range is zero
        return 0.0  # Fallback

    def _check_bollinger_bands(self) -> float:
        """Checks price relative to Bollinger Bands for mean reversion and trend context."""
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        bb_middle = self.indicator_values.get("BB_Middle", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)

        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(last_close):
            return np.nan

        # 1. Price relative to outer bands (mean reversion signals)
        if last_close < bb_lower: return 1.0  # Below lower band -> Strong bullish potential (oversold relative to volatility)
        if last_close > bb_upper: return -1.0  # Above upper band -> Strong bearish potential (overbought relative to volatility)

        # 2. Price relative to middle band (trend confirmation / position within bands)
        # Calculate position relative to the bands' width
        band_width = bb_upper - bb_lower
        if band_width <= 0: return 0.0  # Avoid division by zero if bands are flat or invalid

        # Normalize position between lower (-1) and upper (+1) band
        # Position = (Price - Mid) / (Half Band Width)
        # Ensure half-width is positive
        half_width = band_width / 2.0
        if half_width <= 0: return 0.0  # Should not happen if band_width > 0, but safety check

        relative_position = (last_close - bb_middle) / half_width
        # Clamp relative_position between -1 and 1 (approx, as price can exceed bands)
        clamped_relative_pos = max(-1.0, min(1.0, relative_position))

        # Map this relative position to a score:
        # Closer to upper band (positive relative_pos) -> more bearish bias (-ve score)
        # Closer to lower band (negative relative_pos) -> more bullish bias (+ve score)
        # We already handled touching/exceeding bands, so scale score between +/- 0.7 for within bands
        score = -clamped_relative_pos * 0.7

        return score

    def _check_orderbook(self, orderbook_data: dict | None, current_price: Decimal) -> float:
        """Analyzes order book depth for immediate pressure using Order Book Imbalance (OBI)."""
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            # self.logger.debug(f"Orderbook check skipped for {self.symbol}: No data or empty bids/asks.")
            return np.nan

        try:
            # Convert bid/ask data to Decimal, handling potential errors
            def parse_level(level: list) -> tuple[Decimal, Decimal] | None:
                try:
                    price = Decimal(str(level[0]))
                    volume = Decimal(str(level[1]))
                    if price > 0 and volume >= 0:  # Allow zero volume
                        return price, volume
                except (InvalidOperation, ValueError, TypeError, IndexError):
                    pass
                return None

            bids = [parse_level(b) for b in orderbook_data['bids']]
            asks = [parse_level(a) for a in orderbook_data['asks']]
            bids = [b for b in bids if b is not None]  # Filter out parsing errors
            asks = [a for a in asks if a is not None]

            if not bids or not asks:
                self.logger.debug(f"Orderbook check skipped for {self.symbol}: No valid bid/ask levels after parsing.")
                return np.nan

            # --- Simple Order Book Imbalance (OBI) within N levels ---
            num_levels_to_check = 10  # Check top N levels (configurable?)
            bid_volume_sum = sum(level[1] for level in bids[:num_levels_to_check])
            ask_volume_sum = sum(level[1] for level in asks[:num_levels_to_check])

            total_volume = bid_volume_sum + ask_volume_sum
            if total_volume == 0:
                self.logger.debug(f"Orderbook check: No volume within top {num_levels_to_check} levels for {self.symbol}.")
                return 0.0  # Neutral if no volume

            # Calculate Order Book Imbalance (OBI) ratio: (Bids - Asks) / Total
            obi = (bid_volume_sum - ask_volume_sum) / total_volume

            # Scale OBI to a score between -1 and 1
            # Direct scaling: OBI already ranges from -1 to 1 theoretically
            score = float(obi)  # Convert Decimal OBI to float score

            # Optional: Apply non-linear scaling to emphasize stronger imbalances
            # Example: score = float(obi)**3 # Cube enhances values closer to +/- 1
            # Example: score = math.tanh(float(obi) * 2) # Tanh scales smoothly

            self.logger.debug(f"Orderbook check ({self.symbol}): Top {num_levels_to_check} Levels: "
                              f"BidVol={bid_volume_sum:.4f}, AskVol={ask_volume_sum:.4f}, "
                              f"OBI={obi:.4f}, Score={score:.4f}")
            # Clamp score just in case of floating point issues or unexpected OBI > 1
            return max(-1.0, min(1.0, score))

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan  # Return NaN on error

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on the provided entry price, ATR, config multipliers, and market tick size. Uses Decimal precision.
        The initial SL calculated here is primarily used for position sizing and setting initial protection.
        Returns (entry_price, take_profit, stop_loss), all as Decimal or None if calculation fails.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None  # No TP/SL needed for HOLD

        atr_val_float = self.indicator_values.get("ATR")  # Get float ATR value
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or not isinstance(entry_price, Decimal) or pd.isna(entry_price) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float))  # Convert valid float ATR to Decimal

            # Get multipliers from config, convert to Decimal
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            if tp_multiple <= 0 or sl_multiple <= 0:
                self.logger.warning(f"TP ({tp_multiple}) or SL ({sl_multiple}) multiple is zero or negative. Check config. Cannot calculate valid levels.")
                return entry_price, None, None

            # Get market precision and tick size for quantization
            price_precision = self.get_price_precision()
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).")
                 return entry_price, None, None

            take_profit = None
            stop_loss = None

            # Calculate raw offsets
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            if signal == "BUY":
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Quantize TP UP (away from entry), SL DOWN (away from entry)
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick

            elif signal == "SELL":
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Quantize TP DOWN (away from entry), SL UP (away from entry)
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

            # --- Validation ---
            # Ensure SL is actually beyond entry by at least one tick after rounding
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation ({stop_loss}) is too close to or above entry ({entry_price}) after rounding. Adjusting SL down by 1 tick to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation ({stop_loss}) is too close to or below entry ({entry_price}) after rounding. Adjusting SL up by 1 tick to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl

            # Ensure TP is potentially profitable relative to entry after rounding
            if signal == "BUY" and take_profit <= entry_price:
                adjusted_tp = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}BUY signal TP calculation ({take_profit}) resulted in non-profitable level after rounding. Adjusting TP up by 1 tick to {adjusted_tp}.{RESET}")
                take_profit = adjusted_tp
            elif signal == "SELL" and take_profit >= entry_price:
                adjusted_tp = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}SELL signal TP calculation ({take_profit}) resulted in non-profitable level after rounding. Adjusting TP down by 1 tick to {adjusted_tp}.{RESET}")
                take_profit = adjusted_tp

            # Final checks: Ensure SL/TP are still valid (positive price) after adjustments
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in zero or negative price ({stop_loss}) for {self.symbol} after adjustments. Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}Take profit calculation resulted in zero or negative price ({take_profit}) for {self.symbol} after adjustments. Cannot set TP.{RESET}")
                take_profit = None

            # Format for logging
            tp_log = f"{take_profit:.{price_precision}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{price_precision}f}" if stop_loss else 'N/A'
            atr_log = f"{atr:.{price_precision + 1}f}" if atr else 'N/A'  # Log ATR with more precision
            entry_log = f"{entry_price:.{price_precision}f}" if entry_price else 'N/A'

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log}, ATR={atr_log}")
            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency, handling Bybit V5 structures
    (CONTRACT/UNIFIED), retries, and returning Decimal or None.
    """
    lg = logger
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            balance_info = None
            fetched_from_type = "default"  # Track which type was successful

            # For Bybit V5, try specific account types first
            account_types_to_try = []
            if 'bybit' in exchange.id.lower():
                account_types_to_try = ['CONTRACT', 'UNIFIED']  # Prioritize CONTRACT

            for acc_type in account_types_to_try:
                try:
                    lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency} (Attempt {attempt + 1})...")
                    params = {'accountType': acc_type}
                    balance_info = exchange.fetch_balance(params=params)
                    # Check if this structure contains the currency directly or nested
                    if currency in balance_info or \
                       ('info' in balance_info and isinstance(balance_info['info'].get('result', {}).get('list'), list)):
                        balance_list = balance_info.get('info', {}).get('result', {}).get('list', [])
                        # Check if currency exists in this account type's coin list
                        if currency in balance_info or \
                           any(acc.get('accountType') == acc_type and
                               any(coin.get('coin') == currency for coin in acc.get('coin', []))
                               for acc in balance_list):
                            lg.debug(f"Balance data structure potentially containing {currency} found using type '{acc_type}'.")
                            fetched_from_type = acc_type
                            break  # Found a structure likely containing the currency for this type
                    # If not found in this structure, reset balance_info to try next type
                    balance_info = None
                    lg.debug(f"Currency '{currency}' not found or structure invalid for type '{acc_type}'.")

                except ccxt.ExchangeError as e:
                    # Ignore errors indicating the account type doesn't exist or is invalid for the API key
                    if "account type not support" in str(e).lower() or "invalid account type" in str(e).lower() or "api key permission" in str(e).lower():
                        lg.debug(f"Account type '{acc_type}' not supported or permission issue: {e}. Trying next.")
                        continue
                    else:  # Log other exchange errors but continue trying types/default
                        lg.warning(f"Exchange error fetching balance for account type {acc_type}: {e}. Trying next.")
                        continue
                except Exception as e:  # Log other errors but continue
                    lg.warning(f"Unexpected error fetching balance for account type {acc_type}: {e}. Trying next.")
                    continue

                # If balance_info was successfully populated in the loop, break outer loop
                if balance_info is not None:
                    break

            # If specific account types failed or not Bybit, try default fetch_balance
            if balance_info is None:
                lg.debug(f"Fetching balance using default parameters for {currency} (Attempt {attempt + 1})...")
                try:
                    balance_info = exchange.fetch_balance()
                    fetched_from_type = "default"
                except Exception as e:
                    lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                    # Raise to trigger retry logic below if applicable
                    raise e

            # --- Parse the balance_info ---
            if not balance_info:  # Check if fetch failed entirely
                 lg.error("Failed to fetch any balance info after trying account types and default.")
                 # Trigger retry logic
                 raise ccxt.ExchangeError("Failed to fetch balance info structure")

            available_balance_str = None
            parse_source_detail = f"(Fetched using type: {fetched_from_type})"

            # 1. Standard CCXT structure: balance_info[currency]['free']
            if currency in balance_info and 'free' in balance_info[currency] and balance_info[currency]['free'] is not None:
                available_balance_str = str(balance_info[currency]['free'])
                lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str} {parse_source_detail}")

            # 2. Bybit V5 structure (often nested): Check 'info' -> 'result' -> 'list'
            elif 'info' in balance_info and isinstance(balance_info['info'].get('result', {}).get('list'), list):
                balance_list = balance_info['info']['result']['list']
                found_in_v5 = False
                for account in balance_list:
                    # If we know the successful type, only check that. Otherwise, check any account type returned.
                    current_account_type = account.get('accountType')
                    if fetched_from_type == 'default' or current_account_type == fetched_from_type:
                        coin_list = account.get('coin')
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get('coin') == currency:
                                    # Prefer Bybit V5 'availableBalance' or 'availableToWithdraw'
                                    free = coin_data.get('availableBalance')  # Bybit's preferred field for available trading balance
                                    source_field = 'availableBalance'
                                    if free is None:
                                        free = coin_data.get('availableToWithdraw')  # Alternative 'free' balance
                                        source_field = 'availableToWithdraw'
                                    # Fallback to walletBalance only if others are missing (less ideal)
                                    if free is None:
                                        free = coin_data.get('walletBalance')
                                        source_field = 'walletBalance'
                                        if free is not None:
                                            lg.warning(f"{NEON_YELLOW}Using 'walletBalance' as balance fallback for {currency}. May include unrealized PnL.{RESET}")

                                    if free is not None:
                                        available_balance_str = str(free)
                                        lg.debug(f"Found balance via Bybit V5 nested structure: {available_balance_str} {currency} (Field: {source_field}, Account: {current_account_type or 'N/A'}) {parse_source_detail}")
                                        found_in_v5 = True
                                        break  # Found the currency in this account's coin list
                            if found_in_v5: break  # Stop searching accounts if found
                if not found_in_v5:
                    lg.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]' structure for relevant account type(s). {parse_source_detail}")

            # 3. Fallback: Check top-level 'free' dictionary if present (less common structure)
            elif 'free' in balance_info and isinstance(balance_info['free'], dict) and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                available_balance_str = str(balance_info['free'][currency])
                lg.debug(f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency} {parse_source_detail}")

            # 4. Last Resort: Check 'total' balance if 'free'/'available' still missing
            if available_balance_str is None:
                total_balance_str = None
                # Check standard total
                if currency in balance_info and 'total' in balance_info[currency] and balance_info[currency]['total'] is not None:
                    total_balance_str = str(balance_info[currency]['total'])
                    lg.debug(f"Using standard 'total' balance as fallback for {currency}: {total_balance_str} {parse_source_detail}")
                # Check nested V5 total (walletBalance) if standard failed
                elif 'info' in balance_info and isinstance(balance_info['info'].get('result', {}).get('list'), list):
                     balance_list = balance_info['info']['result']['list']
                     found_total_v5 = False
                     for account in balance_list:
                        current_account_type = account.get('accountType')
                        if fetched_from_type == 'default' or current_account_type == fetched_from_type:
                            coin_list = account.get('coin')
                            if isinstance(coin_list, list):
                                for coin_data in coin_list:
                                    if coin_data.get('coin') == currency:
                                        total = coin_data.get('walletBalance')  # Use walletBalance as total proxy
                                        if total is not None:
                                             total_balance_str = str(total)
                                             lg.debug(f"Using nested V5 'walletBalance' as 'total' fallback for {currency}: {total_balance_str} (Account: {current_account_type or 'N/A'}) {parse_source_detail}")
                                             found_total_v5 = True
                                             break
                                if found_total_v5: break
                     if found_total_v5:
                         available_balance_str = total_balance_str  # Use the found total

                if available_balance_str is not None:
                    lg.warning(f"{NEON_YELLOW}Could not determine 'free'/'available' balance for {currency}. Using 'total' balance ({available_balance_str}) as fallback. This might include collateral/unrealized PnL.{RESET}")
                else:
                    lg.error(f"{NEON_RED}Could not determine any balance for {currency}. Balance info structure not recognized or currency missing. {parse_source_detail}{RESET}")
                    lg.debug(f"Full balance_info structure: {balance_info}")  # Log structure for debugging
                    # Continue to retry logic by raising an error or returning None after retries
                    if attempt < MAX_API_RETRIES: continue  # Explicitly continue to next retry attempt
                    else: return None  # Failed to find balance after all retries

            # --- Convert to Decimal ---
            try:
                final_balance = Decimal(available_balance_str)
                if final_balance >= 0:  # Allow zero balance
                    lg.info(f"Available {currency} balance: {final_balance:.4f} {parse_source_detail}")
                    return final_balance  # Success
                else:
                    lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Returning None.")
                    return None
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
                # Trigger retry logic or return None after retries
                if attempt < MAX_API_RETRIES: continue
                else: return None

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ReadTimeout) as e:
            lg.warning(f"Network error fetching balance (Attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES:
                lg.warning(f"Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached fetching balance after network errors.{RESET}")
                return None
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 2
            try:  # Parse wait time
                 if 'try again in' in str(e).lower():
                     wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                     wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                 elif 'rate limit' in str(e).lower():
                     import re
                     match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                     if match:
                         num = int(match.group(1)); unit = match.group(2)
                         wait_time = max(1, int((num / 1000 if unit == 'ms' else num) + 1))
            except Exception as parse_err:
                 lg.debug(f"Could not parse wait time from rate limit error: {parse_err}")
            lg.warning(f"Rate limit exceeded fetching balance. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except ccxt.AuthenticationError as e:
            lg.error(f"{NEON_RED}Authentication error fetching balance: {e}. Check API key permissions.{RESET}")
            return None  # Don't retry auth errors
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
            # Don't retry generic exchange errors unless known retryable
            # If it was an account type error handled above, we might retry with default.
            # If it happens on default or is another error, don't retry further.
            if attempt < MAX_API_RETRIES:
                 lg.warning(f"Retrying after exchange error in {RETRY_DELAY_SECONDS}s...")
                 time.sleep(RETRY_DELAY_SECONDS)  # Allow retry for potentially recoverable exchange errors
            else:
                 lg.error(f"{NEON_RED}Max retries reached after ExchangeError fetching balance.{RESET}")
                 return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            # Don't retry unexpected errors
            return None

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after all retries.{RESET}")
    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Gets market information like precision, limits, contract type, ensuring markets are loaded.
    Converts relevant limits/precision to Decimal. Derives tick size/precision if missing.
    """
    lg = logger
    try:
        # Ensure markets are loaded; reload if symbol is missing (with retry)
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                if not exchange.markets or symbol not in exchange.markets:
                    lg.info(f"Market info for {symbol} not loaded or symbol missing, reloading markets (Attempt {attempt + 1})...")
                    exchange.load_markets(reload=True)  # Force reload

                # Check again after reloading
                if symbol in exchange.markets:
                    break  # Success
                else:
                    lg.warning(f"Market {symbol} still not found after reloading markets (Attempt {attempt + 1}).")
                    if attempt < MAX_API_RETRIES:
                        time.sleep(RETRY_DELAY_SECONDS)  # Wait before retrying load_markets
                    else:
                        lg.error(f"{NEON_RED}Market {symbol} still not found after all reload attempts. Check symbol spelling and availability on {exchange.id}.{RESET}")
                        return None
            except (ccxt.NetworkError, ccxt.RequestTimeout) as net_err:
                 lg.warning(f"Network error loading markets (Attempt {attempt + 1}): {net_err}")
                 if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                 else: raise  # Raise final network error
            except ccxt.RateLimitExceeded:
                 # Simplified rate limit handling for market loading
                 wait_time = RETRY_DELAY_SECONDS * 2
                 lg.warning(f"Rate limit exceeded loading markets. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                 time.sleep(wait_time)
            except ccxt.ExchangeError as ex_err:
                 lg.error(f"Exchange error loading markets: {ex_err}")
                 raise  # Raise exchange errors immediately during loading

        market = exchange.market(symbol)
        if not market:
            # Should have been caught by the 'in exchange.markets' check, but safeguard
            lg.error(f"{NEON_RED}Market dictionary not found for {symbol} even after checking exchange.markets.{RESET}")
            return None

        # --- Enhance Market Info ---
        # Add contract flag for easier checking later
        market_type = market.get('type', 'unknown')  # spot, swap, future etc.
        market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']
        contract_type = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "Spot/Other"

        # Ensure precision and limits dictionaries exist
        if 'precision' not in market: market['precision'] = {}
        if 'limits' not in market: market['limits'] = {}

        # Derive and store tick size and price precision using helper methods if missing
        # Requires a temporary analyzer instance
        try:
            # Create with minimal dummy data/config
            dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            dummy_df.index.name = 'timestamp'
            analyzer_temp = TradingAnalyzer(dummy_df, lg, {}, market)  # Pass minimal config, but real market info
            min_tick = analyzer_temp.get_min_tick_size()
            price_prec_digits = analyzer_temp.get_price_precision()

            # Store derived tick size as Decimal in precision dict if valid
            if min_tick > 0:
                market['precision']['tick_derived'] = min_tick  # Store derived tick separately
                # If 'tick' or 'price' precision fields are missing, populate them with derived tick
                if market['precision'].get('tick') is None: market['precision']['tick'] = str(min_tick)
                if market['precision'].get('price') is None: market['precision']['price'] = str(min_tick)
            if price_prec_digits >= 0 and market['precision'].get('price_digits_derived') is None:
                 market['precision']['price_digits_derived'] = price_prec_digits

        except Exception as e:
            lg.warning(f"Could not derive tick size/precision for {symbol}: {e}")

        # Convert existing limits/precision values to Decimal where applicable
        for limit_type in ['amount', 'cost', 'price']:
            if limit_type in market['limits']:
                for bound in ['min', 'max']:
                    val = market['limits'][limit_type].get(bound)
                    if val is not None:
                        try: market['limits'][limit_type][bound] = Decimal(str(val))
                        except (InvalidOperation, TypeError, ValueError):
                            lg.warning(f"Could not parse market limit {limit_type}.{bound} ('{val}') as Decimal. Setting to None.")
                            market['limits'][limit_type][bound] = None  # Invalidate bad limit

        # Convert precision fields (often step sizes) to Decimal
        for prec_type in ['amount', 'price', 'cost', 'tick']:
             if prec_type in market['precision']:
                  val = market['precision'][prec_type]
                  if val is not None:
                       try:
                           # Store precision step values as Decimal
                           market['precision'][prec_type] = Decimal(str(val))
                       except (InvalidOperation, TypeError, ValueError):
                            lg.warning(f"Could not parse market precision {prec_type} ('{val}') as Decimal. Setting to None.")
                            market['precision'][prec_type] = None  # Invalidate bad precision

        # Log key details for confirmation and debugging
        lg.debug(
            f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, ContractType={contract_type}, Base={market.get('base')}, Quote={market.get('quote')}, "
            f"IsContract={market['is_contract']}, Active={market.get('active')}"
        )
        # Log precision/limits using .get safely
        prec = market.get('precision', {})
        lim = market.get('limits', {})
        amt_lim = lim.get('amount', {})
        cost_lim = lim.get('cost', {})
        lg.debug(
            f"  Precision(Price/Amount/Tick/Cost): {prec.get('price')}/{prec.get('amount')}/{prec.get('tick')}/{prec.get('cost')}"
        )
        lg.debug(
            f"  Limits(Amount Min/Max): {amt_lim.get('min')}/{amt_lim.get('max')}"
        )
        lg.debug(
            f"  Limits(Cost Min/Max): {cost_lim.get('min')}/{cost_lim.get('max')}"
        )
        lg.debug(f"  Contract Size: {market.get('contractSize', 'N/A')}")

        if not market.get('active', True):
            lg.warning(f"{NEON_YELLOW}Market {symbol} is marked as inactive by the exchange.{RESET}")
            # Decide if inactive markets should be allowed? For now, allow but warn.

        return market

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Symbol '{symbol}' is not supported by {exchange.id} or is incorrectly formatted: {e}{RESET}")
        return None
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
        lg.error(f"{NEON_RED}Network error loading market info for {symbol} after retries: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error loading market info for {symbol}: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,  # Keep as float for config simplicity
    initial_stop_loss_price: Decimal,  # The calculated initial SL price (Decimal)
    entry_price: Decimal,  # Estimated entry price (e.g., current market price) (Decimal)
    market_info: dict,
    exchange: ccxt.Exchange,  # Pass exchange object for formatting helpers
    logger: logging.Logger | None = None
) -> Decimal | None:
    """Calculates position size based on risk percentage, initial SL distance, balance,
    and market constraints (min/max size, precision, contract size). Uses Decimal precision.
    Returns Decimal size or None if sizing fails or results in zero/invalid size.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed for {symbol}: Invalid or zero balance ({balance} {quote_currency}).")
        return None
    try:
        risk_per_trade_dec = Decimal(str(risk_per_trade))
        if not (0 < risk_per_trade_dec < 1):
             raise ValueError("Risk per trade must be between 0 and 1")
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Position sizing failed for {symbol}: Invalid risk_per_trade ({risk_per_trade}): {e}.")
        return None

    if initial_stop_loss_price is None or entry_price is None or entry_price <= 0:
        lg.error(f"Position sizing failed for {symbol}: Missing or invalid entry_price ({entry_price}) or initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed for {symbol}: Stop loss price ({initial_stop_loss_price}) cannot be the same as entry price ({entry_price}).")
        return None
    # Allow zero or negative stop loss price if it resulted from calculation, but warn
    if initial_stop_loss_price <= 0:
        lg.warning(f"Position sizing for {symbol}: Calculated initial stop loss price ({initial_stop_loss_price}) is zero or negative. Ensure SL calculation is correct.")
        # Do not return None here, let the distance calculation handle it

    # Ensure market info has required Decimal values
    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Position sizing failed for {symbol}: Market info missing 'limits' or 'precision'. Market: {market_info}")
        return None

    try:
        # --- Calculate Risk Amount and Initial Size ---
        risk_amount_quote = balance * risk_per_trade_dec
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)

        if sl_distance_per_unit <= 0:
            lg.error(f"Position sizing failed for {symbol}: Stop loss distance is zero or negative ({sl_distance_per_unit}). Entry={entry_price}, SL={initial_stop_loss_price}")
            return None

        # Get contract size (should be Decimal if market info processed correctly)
        contract_size_val = market_info.get('contractSize', '1')  # Default to 1 if missing
        try:
            contract_size = Decimal(str(contract_size_val))
            if contract_size <= 0: raise ValueError("Contract size must be positive")
        except (InvalidOperation, ValueError, TypeError):
            lg.warning(f"Could not parse contract size '{contract_size_val}' for {symbol}. Defaulting to 1.")
            contract_size = Decimal('1')

        # --- Calculate Size based on Market Type ---
        calculated_size = Decimal('0')
        if market_info.get('linear', True) or not is_contract:  # Treat spot as linear type
            # Size (Base/Contracts) = Risk Amount (Quote) / (SL Distance (Quote/Base) * Contract Size (Base/Contract))
            risk_per_unit_quote = sl_distance_per_unit * contract_size
            if risk_per_unit_quote <= 0:
                lg.error(f"Position sizing failed for {symbol}: Risk per unit is zero or negative ({risk_per_unit_quote}). Check SL distance or contract size.")
                return None
            calculated_size = risk_amount_quote / risk_per_unit_quote
            lg.debug(f"  Linear/Spot Sizing: RiskAmt={risk_amount_quote:.4f} / (SLDist={sl_distance_per_unit} * ContSize={contract_size}) = {calculated_size}")
        else:  # Inverse contract logic
            lg.warning(f"{NEON_YELLOW}Inverse contract detected for {symbol}. Sizing assumes `contractSize` ({contract_size}) represents the value of 1 contract in the *BASE* currency (e.g., 100 for BTCUSD contract worth $100). Verify this for {exchange.id}!{RESET}")
            # Formula: Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
            # Risk per Contract (Quote) = Contract Value (Base) * SL Distance (Quote / Base)
            contract_value_base = contract_size  # Assumed value of 1 contract in BASE currency (e.g., 100 USD for BTCUSD)
            risk_per_contract_quote = contract_value_base * sl_distance_per_unit

            if risk_per_contract_quote > 0:
                calculated_size = risk_amount_quote / risk_per_contract_quote
                lg.debug(f"  Inverse Sizing: RiskPerContQuote = ContValBase({contract_value_base}) * SLDist({sl_distance_per_unit}) = {risk_per_contract_quote}")
                lg.debug(f"  Inverse Sizing: Size = RiskAmt({risk_amount_quote:.4f}) / RiskPerContQuote({risk_per_contract_quote}) = {calculated_size}")
            else:
                 lg.error(f"Position sizing failed for inverse contract {symbol}: Risk per contract calculation is zero or negative ({risk_per_contract_quote}).")
                 return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade_dec:.2%}, Risk Amount={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Distance={sl_distance_per_unit}")
        lg.info(f"  Contract Size={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})  # Cost = size * price
        precision = market_info.get('precision', {})
        # Get amount precision/step size (should be Decimal from get_market_info)
        amount_step_size = precision.get('amount')  # This is the step size (e.g., 0.001)

        # Get min/max amount limits (size limits, should be Decimal)
        min_amount = amount_limits.get('min')  # Should be Decimal or None
        max_amount = amount_limits.get('max')  # Should be Decimal or None

        # Get min/max cost limits (value limits in quote currency, should be Decimal)
        min_cost = cost_limits.get('min')  # Should be Decimal or None
        max_cost = cost_limits.get('max')  # Should be Decimal or None

        # Use sensible defaults if limits are missing/None
        if min_amount is None: min_amount = Decimal('0')
        if max_amount is None: max_amount = Decimal('inf')  # Use infinity if no max limit
        if min_cost is None: min_cost = Decimal('0')
        if max_cost is None: max_cost = Decimal('inf')

        # Ensure limits are valid Decimals
        if not isinstance(min_amount, Decimal) or not isinstance(max_amount, Decimal) or \
           not isinstance(min_cost, Decimal) or not isinstance(max_cost, Decimal):
            lg.error(f"Invalid limit types found for {symbol}. Amount(min/max): {type(min_amount)}/{type(max_amount)}, Cost(min/max): {type(min_cost)}/{type(max_cost)}. Cannot proceed.")
            return None

        # 1. Adjust size based on MIN/MAX AMOUNT limits
        adjusted_size = calculated_size
        if adjusted_size < min_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} {size_unit} is below minimum amount {min_amount} {size_unit}. Adjusting to minimum.{RESET}")
            adjusted_size = min_amount
        # Check against max_amount only if it's finite
        elif max_amount.is_finite() and adjusted_size > max_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} {size_unit} exceeds maximum amount {max_amount} {size_unit}. Capping at maximum.{RESET}")
            adjusted_size = max_amount

        # 2. Check COST limits with the amount-adjusted size
        # Cost = Size * Entry Price (for spot)
        # Cost = Size * Entry Price * Contract Size (for linear contracts, if contract size is base units per contract)
        # Cost = Size (Contracts) * Contract Value (Quote) (for inverse contracts, needs careful check of contract value source)
        current_cost = Decimal('0')
        if market_info.get('linear', True) or not is_contract:
            # Linear/Spot: Cost = Size (base/contracts) * Entry Price (quote/base) * Contract Size (base/contract) -> Quote
            # If spot, contract_size is effectively 1 base unit / 1 unit
            current_cost = adjusted_size * entry_price * contract_size
        else:  # Inverse
            # Cost = Size (contracts) * Contract Value (Quote / contract)
            # We need the contract value in QUOTE currency.
            # If `contract_size` holds BASE value (e.g., 100 USD for BTCUSD):
            # Value (Quote) = Contract Value (Base) / Entry Price (Quote / Base) -- THIS SEEMS WRONG.
            # Let's assume `contractSize` IS the quote value for inverse for now, but needs verification.
            # E.g., if 1 contract is 1 BTC, its quote value changes.
            # Bybit V5 API info often provides `contractVal` and `contractValCurrency`
            # If contractValCurrency is QUOTE, use contractVal directly.
            # If contractValCurrency is BASE, need conversion (ValueQuote = contractVal * Price)
            # Let's assume simpler case first and require user verification:
            lg.warning(f"{NEON_YELLOW}Inverse Cost Calculation: Assuming Size ({adjusted_size}) is in contracts and cost limit applies to Quote value. Needs verification of how {exchange.id} defines cost limits for inverse.{RESET}")
            # Placeholder: Estimate cost as size * entry_price (value of size contracts at entry) - ROUGH ESTIMATE
            current_cost = adjusted_size * entry_price  # Very rough estimate, may be wrong

        lg.debug(f"  Cost Check: Amount-Adjusted Size={adjusted_size:.8f} {size_unit}, Estimated Cost={current_cost:.4f} {quote_currency}")
        lg.debug(f"  Cost Limits: Min={min_cost}, Max={max_cost}")

        # Check MIN Cost
        if min_cost > 0 and current_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} {quote_currency} (Size: {adjusted_size:.8f}) is below minimum cost {min_cost} {quote_currency}. Attempting to increase size.{RESET}")
            # Calculate required size to meet min cost (re-evaluate this based on verified cost formula)
            required_size_for_min_cost = Decimal('0')
            try:
                if market_info.get('linear', True) or not is_contract:
                     if entry_price > 0 and contract_size > 0: required_size_for_min_cost = min_cost / (entry_price * contract_size)
                     else: raise ValueError("Invalid entry price or contract size for linear cost recalc")
                else:  # Inverse (using rough estimate again)
                     if entry_price > 0: required_size_for_min_cost = min_cost / entry_price
                     else: raise ValueError("Invalid entry price for inverse cost recalc")

                lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f} {size_unit}")

                # Check if this required size is feasible (within amount limits)
                if max_amount.is_finite() and required_size_for_min_cost > max_amount:
                    lg.error(f"{NEON_RED}Cannot meet minimum cost {min_cost} without exceeding maximum amount {max_amount}. Trade aborted.{RESET}")
                    return None
                elif required_size_for_min_cost < min_amount:
                     lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f}, but min amount is {min_amount}. Trade aborted.{RESET}")
                     return None
                else:
                     lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f} {size_unit}")
                     adjusted_size = required_size_for_min_cost
            except (ValueError, InvalidOperation) as recalc_err:
                 lg.error(f"Could not calculate required size for min cost: {recalc_err}. Trade aborted.")
                 return None

        # Check MAX Cost
        elif max_cost.is_finite() and current_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} {quote_currency} exceeds maximum cost {max_cost} {quote_currency}. Reducing size.{RESET}")
            adjusted_size_for_max_cost = Decimal('0')
            try:
                if market_info.get('linear', True) or not is_contract:
                    if entry_price > 0 and contract_size > 0: adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                    else: raise ValueError("Invalid entry price or contract size for linear cost recalc")
                else:  # Inverse (using rough estimate)
                    if entry_price > 0: adjusted_size_for_max_cost = max_cost / entry_price
                    else: raise ValueError("Invalid entry price for inverse cost recalc")

                lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f} {size_unit}")

                # Check if this reduced size is now below min amount
                if adjusted_size_for_max_cost < min_amount:
                    lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is now below minimum amount {min_amount}. Cannot meet conflicting limits. Trade aborted.{RESET}")
                    return None
                else:
                    adjusted_size = adjusted_size_for_max_cost
            except (ValueError, InvalidOperation) as recalc_err:
                 lg.error(f"Could not calculate required size for max cost: {recalc_err}. Trade aborted.")
                 return None

        # 3. Apply Amount Precision/Step Size (rounding DOWN using step size)
        final_size = Decimal('0')
        if amount_step_size is not None and isinstance(amount_step_size, Decimal) and amount_step_size > 0:
            try:
                # Use Decimal floor division (//) to round down to the nearest step size increment
                final_size = (adjusted_size // amount_step_size) * amount_step_size
                lg.info(f"Applied amount step size ({amount_step_size}), rounded down: {adjusted_size:.8f} -> {final_size} {size_unit}")
            except Exception as step_err:
                lg.warning(f"{NEON_YELLOW}Error applying amount step size {amount_step_size} for {symbol}: {step_err}. Using size adjusted only for limits: {adjusted_size:.8f}{RESET}")
                final_size = adjusted_size  # Fallback, but might fail on exchange
        else:
             # Fallback: If step size is invalid or not provided, attempt using exchange.amount_to_precision
             lg.warning(f"{NEON_YELLOW}Amount step size invalid or missing ({amount_step_size}). Attempting exchange.amount_to_precision (TRUNCATE).{RESET}")
             try:
                # amount_to_precision usually expects a float input
                # Use exchange.amount_to_precision with TRUNCATE (equivalent to floor/ROUND_DOWN for positive numbers)
                formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
                final_size = Decimal(formatted_size_str)
                lg.info(f"Applied exchange.amount_to_precision (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
             except Exception as fmt_err:
                 lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision for {symbol} ({fmt_err}). Using size adjusted only for limits: {adjusted_size:.8f}{RESET}")
                 final_size = adjusted_size  # Use the size adjusted for limits only, likely to fail

        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{NEON_RED}Position size became zero or negative ({final_size}) after adjustments/rounding for {symbol}. Trade aborted.{RESET}")
            return None

        # Final check against min amount AFTER formatting/rounding
        if final_size < min_amount:
            # Use a small tolerance for comparison if needed, but direct Decimal compare is better
            if not math.isclose(float(final_size), float(min_amount), rel_tol=1e-9, abs_tol=1e-12) and final_size < min_amount:
                 lg.error(f"{NEON_RED}Final formatted size {final_size} {size_unit} is below minimum amount {min_amount} {size_unit} for {symbol}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.warning(f"Final formatted size {final_size} is extremely close to min amount {min_amount}. Proceeding cautiously.")

        # Final check against min cost AFTER formatting/rounding (recalculate cost)
        final_cost = Decimal('0')
        try:
            if market_info.get('linear', True) or not is_contract:
                 final_cost = final_size * entry_price * contract_size
            else:  # Inverse (rough estimate)
                 final_cost = final_size * entry_price
        except Exception: pass  # Ignore calculation error here, focus on size

        if min_cost > 0 and final_cost < min_cost:
             # Check if it's close enough due to rounding? Use tolerance.
            if not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6, abs_tol=1e-9) and final_cost < min_cost:
                 lg.error(f"{NEON_RED}Final formatted size {final_size} {size_unit} results in cost {final_cost:.4f} {quote_currency} which is below minimum cost {min_cost} {quote_currency} for {symbol}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.warning(f"Final cost {final_cost} is very close to min cost {min_cost}. Proceeding cautiously.")

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(f"{NEON_RED}Position sizing error for {symbol}: Missing market info key {e}. Market: {market_info}{RESET}")
        return None
    except (InvalidOperation, ValueError, TypeError) as e:
         lg.error(f"{NEON_RED}Position sizing error for {symbol}: Invalid number format or operation. Check limits/precision in market info and calculations. Error: {e}{RESET}")
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks for an open position for the given symbol using fetch_positions.
    Returns the unified position dictionary from CCXT if an active position exists, otherwise None.
    Handles variations in position reporting (size > 0, side field, different API versions).
    Enhances the returned dict with SL/TP/TSL info parsed from the 'info' dict if available (as Decimals).
    """
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: list[dict] = []
        market = None  # Fetch market info for context

        # Fetch market info first to determine category for Bybit V5
        try:
            market = get_market_info(exchange, symbol, lg)
            if not market:
                 # If market info fails, we might not be able to fetch positions correctly for Bybit V5
                 lg.error(f"Cannot reliably fetch position for {symbol} without valid market info.")
                 # Optionally, could try fetching without category, but might fail or return wrong type
                 return None  # Safer to return None if market info fails
        except Exception as e:
             lg.error(f"Error fetching market info for {symbol} before getting position: {e}. Cannot proceed.")
             return None

        # Prepare params for fetch_positions (especially Bybit V5)
        params = {}
        fetch_symbols = [symbol]  # Standard way to fetch for one symbol
        if 'bybit' in exchange.id.lower():
            category = 'linear' if market.get('linear', True) else 'inverse'
            params['category'] = category
            # Bybit V5 fetch_positions usually requires the symbol in params OR as arg
            params['symbol'] = symbol  # Explicitly add symbol to params for V5
            # Some ccxt versions might still need symbols=[] arg, others use params
            # Let's use both for broader compatibility? Or rely on params.
            # fetch_symbols = [] # Test if using params['symbol'] is sufficient
            lg.debug(f"Using params for fetch_positions: {params}")

        # --- Fetch Positions with Retries ---
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                positions = exchange.fetch_positions(symbols=fetch_symbols, params=params)
                lg.debug(f"Fetched positions response (Attempt {attempt + 1}): {positions}")
                break  # Success

            except ccxt.ArgumentsRequired as e:
                # This exchange might need fetching all positions
                lg.warning(f"Exchange {exchange.id} requires fetching all positions (Attempt {attempt + 1}). Error: {e}")
                try:
                     all_positions = exchange.fetch_positions(params=params)  # Still pass params if needed
                     # Filter for the specific symbol (case-insensitive check just in case)
                     positions = [p for p in all_positions if p.get('symbol', '').upper() == symbol.upper()]
                     lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
                     break  # Success after fetching all
                except Exception as e_all:
                     lg.error(f"Error fetching ALL positions for {symbol} (Attempt {attempt + 1}): {e_all}")
                     # Continue to retry logic below

            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                lg.warning(f"Network error fetching positions for {symbol} (Attempt {attempt + 1}): {e}")
                if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                else: lg.error("Max retries reached fetching positions after network errors."); return None
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 2
                try:  # Parse wait time
                    if 'try again in' in str(e).lower():
                         wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                         wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                except Exception: pass
                lg.warning(f"Rate limit exceeded fetching positions. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                # Handle errors that might indicate no position gracefully
                no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
                no_pos_codes_v5 = [110025, 110021]  # Bybit V5: Position not found / is closed
                err_str = str(e).lower()
                err_code = getattr(e, 'code', None)  # Bybit V5 retCode often mapped here

                if any(msg in err_str for msg in no_pos_msgs) or (err_code in no_pos_codes_v5):
                    lg.info(f"No open position found for {symbol} (Confirmed by exchange error: {e}).")
                    return None
                # Re-raise other exchange errors if not retryable
                lg.error(f"Unhandled Exchange error fetching positions for {symbol}: {e} (Code: {err_code}){RESET}")
                if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)  # Allow retry for some exchange errors
                else: return None  # Fail after retries
            except Exception as e:
                 lg.error(f"Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
                 return None  # Fail on unexpected errors

        # --- Process the fetched positions (should be 0 or 1 for the symbol in One-Way mode) ---
        active_position = None
        if positions is None:  # Check if fetch failed after retries
             lg.error(f"Failed to fetch positions for {symbol} after all retries.")
             return None

        for pos in positions:
            # Ensure the position entry matches the symbol we're interested in
            if pos.get('symbol', '').upper() != symbol.upper():
                lg.debug(f"Skipping position entry for different symbol: {pos.get('symbol')}")
                continue

            # --- Determine Position Size ---
            # Standardized field is 'contracts'. Bybit V5 uses 'size' in info. Binance uses 'positionAmt'.
            pos_size_str = None
            size_source = "unknown"
            if pos.get('contracts') is not None:
                pos_size_str = str(pos['contracts'])
                size_source = "contracts"
            elif pos.get('info', {}).get('size') is not None:  # Bybit V5
                pos_size_str = str(pos['info']['size'])
                size_source = "info.size"
            elif pos.get('info', {}).get('positionAmt') is not None:  # Binance
                pos_size_str = str(pos['info']['positionAmt'])
                size_source = "info.positionAmt"
            elif pos.get('contractSize') is not None:  # Less common fallback
                pos_size_str = str(pos['contractSize'])
                size_source = "contractSize"

            if pos_size_str is None:
                lg.debug(f"Could not find position size field in position data for {symbol}: {pos}")
                continue  # Skip this entry if size cannot be determined

            # Check if size is valid and meaningfully non-zero
            try:
                position_size = Decimal(pos_size_str)
                # Use a small threshold based on minimum contract size if available, else small number
                min_size_threshold = Decimal('1e-9')  # Default threshold
                if market:  # Use market info if available
                    try:
                        # Amount precision/step size should be Decimal
                        min_amt_step = market.get('precision', {}).get('amount')
                        if isinstance(min_amt_step, Decimal) and min_amt_step > 0:
                            min_size_threshold = min_amt_step  # Use the actual step size as threshold
                        else:
                             # Fallback to limits.amount.min if step size not available/valid
                             min_amt_limit = market.get('limits', {}).get('amount', {}).get('min')
                             if isinstance(min_amt_limit, Decimal) and min_amt_limit > 0:
                                 min_size_threshold = min_amt_limit  # Use min limit as threshold
                    except Exception: pass  # Ignore errors getting min size

                # Check absolute size against threshold. Binance 'positionAmt' is signed. Others usually unsigned.
                if abs(position_size) >= min_size_threshold:
                    # Found an active position
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size} (Source: {size_source}).")
                    break  # Stop after finding the first active position entry matching symbol
                else:
                    lg.debug(f"Position size {position_size} (Source: {size_source}) is below threshold {min_size_threshold}. Not an active position.")

            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse position size '{pos_size_str}' (Source: {size_source}) as Decimal for {symbol}. Skipping entry. Error: {e}")
                continue

        # --- Post-Process the found active position (if any) ---
        if active_position:
            # --- Determine Side ---
            side = active_position.get('side')  # Standard 'long' or 'short'
            size_decimal = Decimal('0')  # Recalculate size for side inference if needed
            try:
                # Re-read size string determined earlier
                size_str_for_side = None
                if size_source == "contracts": size_str_for_side = str(active_position['contracts'])
                elif size_source == "info.size": size_str_for_side = str(active_position['info']['size'])
                elif size_source == "info.positionAmt": size_str_for_side = str(active_position['info']['positionAmt'])
                elif size_source == "contractSize": size_str_for_side = str(active_position['contractSize'])

                if size_str_for_side: size_decimal = Decimal(str(size_str_for_side))
                else: raise ValueError("Size string missing")

            except (InvalidOperation, ValueError, TypeError) as parse_err:
                 lg.error(f"Error parsing size {size_str_for_side} for side inference: {parse_err}")
                 # Cannot reliably determine side if size parse fails
                 return None

            # Re-calculate threshold for comparison
            min_size_threshold = Decimal('1e-9')
            if market:
                try:
                    min_amt_step = market.get('precision', {}).get('amount')
                    if isinstance(min_amt_step, Decimal) and min_amt_step > 0: min_size_threshold = min_amt_step
                    else:
                         min_amt_limit = market.get('limits', {}).get('amount', {}).get('min')
                         if isinstance(min_amt_limit, Decimal) and min_amt_limit > 0: min_size_threshold = min_amt_limit
                except Exception: pass

            # Infer side if 'side' field is missing, 'none', or ambiguous
            # Bybit V5: 'side' in info dict can be 'Buy', 'Sell', or 'None'
            if side not in ['long', 'short']:
                info_side = active_position.get('info', {}).get('side', 'None')  # Bybit V5 side
                if info_side == 'Buy': side = 'long'
                elif info_side == 'Sell': side = 'short'
                # If still not determined, use size sign (esp. for Binance 'positionAmt')
                elif size_source == 'info.positionAmt' and size_decimal != 0:  # Binance uses signed 'positionAmt'
                    if size_decimal >= min_size_threshold: side = 'long'
                    elif size_decimal <= -min_size_threshold: side = 'short'
                # Fallback using unsigned size (less reliable if side field missing, Bybit uses this)
                # If side is None/Unknown AND size is positive, assume side based on how entry happened? Ambiguous.
                # For Bybit, if info.side is 'None' and size > 0, it means flat.
                elif info_side == 'None' and abs(size_decimal) < min_size_threshold:
                    lg.debug(f"Position side is 'None' and size {size_decimal} is near zero. Treating as flat.")
                    return None  # No active position
                elif info_side == 'None' and abs(size_decimal) >= min_size_threshold:
                    lg.warning(f"Ambiguous position state: Bybit side is 'None' but size {size_decimal} > 0. Cannot determine side. Treating as error.")
                    return None
                else:  # Default guess if side is missing entirely (less reliable)
                    lg.warning(f"Position side field missing/invalid ('{side}'). Attempting inference from size sign (may be inaccurate).")
                    if size_decimal >= min_size_threshold: side = 'long'
                    elif size_decimal <= -min_size_threshold: side = 'short'  # Should only happen if size field was signed
                    else: side = None  # Cannot determine

                # If side could not be determined, treat as error/no position
                if side not in ['long', 'short']:
                    lg.error(f"Could not determine side for position with size {size_decimal}. Position data: {active_position}")
                    return None

                # Add inferred side back to the dictionary for consistency
                active_position['side'] = side

            # Ensure 'contracts' field holds the ABSOLUTE size as float/Decimal if it exists
            # CCXT standard is absolute size here. Bybit V5 'size' is also absolute.
            if active_position.get('contracts') is not None:
                 try:
                      current_contracts = Decimal(str(active_position['contracts']))
                      active_position['contracts'] = abs(current_contracts)  # Store absolute value
                 except (InvalidOperation, ValueError, TypeError): pass  # Ignore if conversion fails

            # --- Enhance with SL/TP/TSL info directly if available ---
            # Check 'info' dict for exchange-specific fields (e.g., Bybit V5)
            info_dict = active_position.get('info', {})

            # Helper to extract and validate price/value from info dict, returning Decimal or None
            # Handles '0', '0.0', '' as inactive/None
            def get_valid_decimal_from_info(key: str) -> Decimal | None:
                val_str = info_dict.get(key)
                if val_str is not None and str(val_str).strip() not in ['', '0', '0.0']:
                    try:
                        val_dec = Decimal(str(val_str).strip())
                        if val_dec > 0: return val_dec
                        # Allow 0 only for specific fields like TSL activation if it means immediate
                        elif val_dec == 0 and key == 'activePrice': return Decimal('0')
                    except (InvalidOperation, ValueError, TypeError): pass  # Ignore conversion errors
                return None  # Return None if inactive ('0', '', None) or parsing fails

            # Populate standard fields if missing, using validated values from info
            # Store these also as Decimal for internal use
            sl_val_dec = get_valid_decimal_from_info('stopLoss')
            tp_val_dec = get_valid_decimal_from_info('takeProfit')
            tsl_dist_dec = get_valid_decimal_from_info('trailingStop')  # Distance/Value
            tsl_act_dec = get_valid_decimal_from_info('activePrice')  # Activation Price (can be 0)

            # Add parsed Decimal versions to the dict for easier checks later
            active_position['stopLossPriceDecimal'] = sl_val_dec
            active_position['takeProfitPriceDecimal'] = tp_val_dec
            active_position['trailingStopLossDistanceDecimal'] = tsl_dist_dec if tsl_dist_dec is not None else Decimal('0')
            active_position['tslActivationPriceDecimal'] = tsl_act_dec if tsl_act_dec is not None else Decimal('0')

            # Update standard ccxt float fields if they were missing/None
            if active_position.get('stopLossPrice') is None and sl_val_dec is not None:
                active_position['stopLossPrice'] = float(sl_val_dec)
            if active_position.get('takeProfitPrice') is None and tp_val_dec is not None:
                active_position['takeProfitPrice'] = float(tp_val_dec)

            # --- Log details of the confirmed active position ---
            # Get precision for logging
            log_precision = 8
            amount_precision = 8
            if market:
                 log_precision = market.get('precision', {}).get('price_digits_derived', 8)
                 try:  # Get amount precision (digits from step size)
                    amount_prec_val = market.get('precision', {}).get('amount')
                    if isinstance(amount_prec_val, Decimal) and amount_prec_val > 0:
                        amount_precision = max(0, abs(amount_prec_val.normalize().as_tuple().exponent))
                    elif isinstance(amount_prec_val, int): amount_precision = amount_prec_val
                 except Exception: pass

            # Helper function to format Decimal/float/string for logging
            def format_log_value(value: Any, precision: int) -> str:
                if value is None: return 'N/A'
                try:
                    d_value = Decimal(str(value))
                    # Format non-zero values with precision
                    if d_value != 0: return f"{d_value:.{precision}f}"
                    # Show 0 for TSL activation/distance if it means inactive/immediate
                    elif isinstance(value, Decimal) and value == 0: return "0.0"
                    else: return 'N/A'  # Treat other zeros or non-numerics as N/A
                except (InvalidOperation, ValueError, TypeError):
                    return str(value)  # Return raw string if not convertible

            entry_price_val = active_position.get('entryPrice', info_dict.get('avgPrice'))
            entry_price = format_log_value(entry_price_val, log_precision)
            contracts_val = active_position.get('contracts')  # Should be absolute Decimal now
            contracts = format_log_value(contracts_val, amount_precision)
            liq_price = format_log_value(active_position.get('liquidationPrice'), log_precision)
            leverage_str = active_position.get('leverage', info_dict.get('leverage'))
            leverage = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str is not None else 'N/A'
            pnl_val = active_position.get('unrealizedPnl')
            pnl = format_log_value(pnl_val, 4)  # PnL usually shown with fewer decimals

            # Use the parsed Decimal values for SL/TP/TSL logging
            sl_log = format_log_value(active_position['stopLossPriceDecimal'], log_precision)
            tp_log = format_log_value(active_position['takeProfitPriceDecimal'], log_precision)
            tsl_dist_log = format_log_value(active_position['trailingStopLossDistanceDecimal'], log_precision)
            tsl_act_log = format_log_value(active_position['tslActivationPriceDecimal'], log_precision)
            is_tsl_active_log = active_position['trailingStopLossDistanceDecimal'] > 0

            logger.info(f"{NEON_GREEN}Active {side.upper()} position found for {symbol}:{RESET} "
                        f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                        f"Leverage={leverage}, PnL={pnl}, SL={sl_log}, TP={tp_log}, "
                        f"TSL Active: {is_tsl_active_log} (Dist={tsl_dist_log}/Act={tsl_act_log})")
            logger.debug(f"Full position details for {symbol}: {active_position}")
            return active_position
        else:
            # If loop completes without finding a non-zero position matching the symbol
            logger.info(f"No active open position found for {symbol}.")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching positions for {symbol}: {e}{RESET}")
    # Network errors handled by retry loop above
    # Exchange errors handled by retry loop or return None above
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error in get_open_position logic for {symbol}: {e}{RESET}", exc_info=True)

    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics and verification hints."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True  # Success if not applicable

    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be > 0.")
        return False

    # --- Check exchange capability ---
    if not exchange.has.get('setLeverage'):
        lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage method via CCXT capabilities.{RESET}")
        # Future: Implement direct API call here if needed as a fallback.
        return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")

        # --- Prepare Bybit V5 specific parameters ---
        params = {}
        if 'bybit' in exchange.id.lower():
            # Bybit V5 requires buyLeverage and sellLeverage, and category.
            # Assume One-Way mode, setting both to the same value.
            params = {
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage),
                'category': 'linear' if market_info.get('linear', True) else 'inverse'
            }
            lg.debug(f"Using Bybit V5 specific params for set_leverage: {params}")

        # --- Call set_leverage with Retries ---
        response = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                # The `leverage` argument is the primary value, `params` provides extra details
                response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
                lg.debug(f"Set leverage raw response (Attempt {attempt + 1}): {response}")
                # If call succeeds without exception, break retry loop
                break

            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                lg.warning(f"Network error setting leverage (Attempt {attempt + 1}): {e}")
                if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                else: raise  # Re-raise after final attempt
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 2
                try:  # Parse wait time
                    if 'try again in' in str(e).lower():
                         wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                         wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                except Exception: pass
                lg.warning(f"Rate limit setting leverage. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                 # Check for "leverage not modified" code - treat as success
                 bybit_code = getattr(e, 'code', None)
                 if bybit_code == 110045 or "leverage not modified" in str(e).lower():
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation: Code {bybit_code}).{RESET}")
                     return True  # Treat as success
                 # Re-raise other exchange errors to potentially retry or fail
                 raise e

        # --- Verification (after successful call or final retry) ---
        verified = False
        if response is not None:
            if isinstance(response, dict):
                 # Check Bybit V5 specific response structure
                 ret_code = response.get('retCode', response.get('info', {}).get('retCode'))
                 if ret_code == 0:
                     lg.debug(f"Set leverage call for {symbol} confirmed success (retCode 0).")
                     verified = True
                 elif ret_code is not None:  # Got a non-zero retCode (should have been caught by ExchangeError?)
                      ret_msg = response.get('retMsg', response.get('info', {}).get('retMsg', 'Unknown Error'))
                      lg.warning(f"Set leverage call for {symbol} returned non-zero retCode {ret_code} ({ret_msg}) despite no exception. Treating as failure.")
                      verified = False
                 else:  # No retCode found, rely on lack of exception from call
                      lg.debug(f"Set leverage call for {symbol} returned a response (no retCode found). Assuming success as no error was raised.")
                      verified = True
            else:  # Response is not a dict, rely on lack of exception
                 lg.debug(f"Set leverage call for {symbol} returned non-dict response. Assuming success as no error was raised.")
                 verified = True
        else:  # Response is None (might happen on success for some ccxt versions/methods)
            lg.debug(f"Set leverage call for {symbol} returned None. Assuming success as no error was raised.")
            verified = True

        if verified:
            lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/confirmed to {leverage}x.{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Leverage setting failed for {symbol} based on response analysis or lack of response.{RESET}")
            return False

    # --- Error Handling (Catch errors from final attempt or specific non-retryable errors) ---
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)  # CCXT often maps Bybit retCode to e.code
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
        # --- Handle Common Bybit V5 Leverage Errors ---
        # Code 110045 handled in retry loop, but double-check
        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation: Code {bybit_code}).{RESET}")
            return True  # Treat as success
        elif bybit_code == 110028 or "set margin mode first" in err_str or "switch margin mode" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Ensure Margin Mode (Isolated/Cross) is set correctly for {symbol} *before* setting leverage. Check Bybit Account Settings. Cannot set Isolated leverage in Cross mode.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x might exceed the risk limit for your account tier or selected margin mode. Check Bybit Risk Limit documentation.{RESET}")
        elif bybit_code == 110009 or "position is in cross margin mode" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Cannot set leverage individually if symbol is using Cross Margin. Switch symbol to Isolated first, or set cross leverage for the whole account margin coin.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
            lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for this specific symbol {symbol}. Check allowed leverage range on Bybit.{RESET}")
        elif "available balance not enough" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: May indicate insufficient available margin if using Isolated Margin and increasing leverage requires more margin allocation possibility.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,  # "BUY" or "SELL"
    position_size: Decimal,  # Size calculated previously (Decimal, positive)
    market_info: dict,
    logger: logging.Logger | None = None,
    reduce_only: bool = False  # Flag for closing trades
) -> dict | None:
    """Places a market order using CCXT with robustness for Bybit V5.
    Handles opening (reduce_only=False) or closing (reduce_only=True).
    Returns the order dictionary on success, None on failure.
    SL/TP/TSL should be set *after* opening trades and verifying position.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    base_currency = market_info.get('base', '')
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else base_currency

    # Convert positive Decimal size to float for ccxt amount parameter
    try:
        amount_float = float(position_size)
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} reduce={reduce_only}): Invalid position size for order amount ({amount_float}). Must be positive.")
            return None
    except Exception as e:
        lg.error(f"Trade aborted ({symbol} {side} reduce={reduce_only}): Failed to convert position size {position_size} to float: {e}")
        return None

    # --- Prepare Order Parameters ---
    params = {
        'reduceOnly': reduce_only,  # Use the passed flag
        # 'timeInForce': 'IOC', # Optional for market orders
    }
    # Add Bybit V5 specific parameters
    if 'bybit' in exchange.id.lower():
        # Assuming One-Way mode (positionIdx=0) is default/required. Hedge mode needs 1 or 2.
        params['positionIdx'] = 0
        params['category'] = 'linear' if market_info.get('linear', True) else 'inverse'
        # Bybit V5 might prefer size as string in params for market orders? Test this.
        # params['qty'] = str(position_size) # Alternative to amount arg? Let's stick to amount arg for now.

    action = "Closing" if reduce_only else "Opening"
    lg.info(f"Attempting to place {side.upper()} {order_type} order ({action}) for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {size_unit}")  # Log the float amount being sent
    lg.info(f"  Params: {params}")

    try:
        # --- Execute Market Order with Retries ---
        order = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                order = exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount_float,  # Use float amount
                    price=None,  # Market order doesn't need price
                    params=params
                )
                # If order placement succeeds without exception, break retry loop
                lg.debug(f"Create order call successful (Attempt {attempt + 1}).")
                break

            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                lg.warning(f"Network error placing order ({action}, Attempt {attempt + 1}): {e}")
                if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                else: raise  # Re-raise after final attempt
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 2
                try:  # Parse wait time
                    if 'try again in' in str(e).lower():
                         wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                         wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
                except Exception: pass
                lg.warning(f"Rate limit placing order. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                 # Check for specific non-fatal errors like "position not found" when closing
                 bybit_code = getattr(e, 'code', None)
                 if reduce_only and (bybit_code == 110025 or "position idx not match" in str(e).lower()):
                     lg.warning(f"{NEON_YELLOW}Position not found when attempting to close (reduceOnly=True, Code {bybit_code}). Might have already been closed.{RESET}")
                     # Return a dummy success indicating position is likely closed
                     return {'id': 'N/A', 'status': 'closed', 'info': {'reason': 'Position not found on close attempt'}}
                 # Re-raise other exchange errors to potentially retry or fail
                 raise e

        # --- Check if order was successfully placed after retries ---
        if order is None:
             lg.error(f"{NEON_RED}Failed to place order for {symbol} after all retries.{RESET}")
             return None

        # --- Log Success and Basic Order Details ---
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A')  # Market orders might be 'closed' quickly or 'open' initially
        lg.info(f"{NEON_GREEN}Trade Order Placed Successfully ({action})! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} reduce={reduce_only}): {order}")

        # IMPORTANT: The calling function MUST wait and verify the resulting position.
        return order  # Return the order dictionary

    # --- Error Handling (Catch errors from final attempt or specific non-retryable errors) ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order ({action}) for {symbol}: {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        # Log balance info if possible
        try:
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)  # Use enhanced fetch_balance
            lg.error(f"  Available {QUOTE_CURRENCY} balance: {balance if balance is not None else 'Fetch Failed'}")
            if bybit_code == 110007: lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Check available margin, leverage, and order cost vs balance.{RESET}")
        except Exception as bal_err: lg.error(f"  Could not fetch balance for context: {bal_err}")

    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {symbol} ({action}): {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        lg.error(f"  Size: {amount_float}, Type: {order_type}, Side: {side}, Params: {params}")
        lg.error(f"  Market Limits: Amount={market_info.get('limits', {}).get('amount')}, Cost={market_info.get('limits', {}).get('cost')}")
        lg.error(f"  Market Precision: Amount={market_info.get('precision', {}).get('amount')}, Price={market_info.get('precision', {}).get('price')}")
        # Add hints based on Bybit V5 codes
        if bybit_code == 10001 and "parameter error" in str(e).lower(): lg.error(f"{NEON_YELLOW} >> Hint (10001): Check if size/price violates precision or limits.{RESET}")
        elif bybit_code == 110017: lg.error(f"{NEON_YELLOW} >> Hint (110017): Order size {amount_float} might violate exchange's min/max quantity per order.{RESET}")
        elif bybit_code == 110040: lg.error(f"{NEON_YELLOW} >> Hint (110040): Order size {amount_float} is below the minimum allowed. Check `calculate_position_size`.{RESET}")
        elif bybit_code == 110014 and reduce_only: lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only close failed. Size ({amount_float}) might exceed open position?{RESET}")
        elif bybit_code == 110071: lg.error(f"{NEON_YELLOW} >> Hint (110071): Order cost (Size * Price / Leverage) is below minimum required value.{RESET}")

    except ccxt.ExchangeError as e:
        # Handle specific Bybit V5 error codes for better diagnostics
        bybit_code = getattr(e, 'code', None)
        err_str = str(e).lower()
        lg.error(f"{NEON_RED}Exchange error placing order ({action}) for {symbol}: {e} (Code: {bybit_code}){RESET}")
        # --- Bybit V5 Specific Error Hints ---
        if bybit_code == 110007: lg.error(f"{NEON_YELLOW} >> Hint (110007): Insufficient balance/margin. Cost ~ Size * Price / Leverage.{RESET}")
        elif bybit_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint (110043): Problem related to SL/TP settings? Ensure they are not sent with market order.{RESET}")
        elif bybit_code == 110044: lg.error(f"{NEON_YELLOW} >> Hint (110044): Opening this position would exceed Bybit's risk limit tier. Check limits or reduce size/leverage.{RESET}")
        elif bybit_code == 110055: lg.error(f"{NEON_YELLOW} >> Hint (110055): Mismatch between 'positionIdx' ({params.get('positionIdx')}) and account's Position Mode (One-Way vs Hedge).{RESET}")
        elif bybit_code == 10005 or "order link id exists" in err_str: lg.warning(f"{NEON_YELLOW}Duplicate order ID detected (Code {bybit_code}). Order might be already placed? Check manually!{RESET}")  # Treat as failure for safety
        elif "risk limit can't be place order" in err_str: lg.error(f"{NEON_YELLOW} >> Hint: Order blocked by risk limits. Check Bybit tiers vs leverage/position size.{RESET}")
        elif bybit_code == 110025 and reduce_only: lg.warning(f"{NEON_YELLOW} >> Hint (110025): Position not found when closing (already handled above, but log again).{RESET}")  # Should be caught earlier

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing order ({action}) for {symbol}: {e}{RESET}", exc_info=True)

    return None  # Return None if order failed


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,  # Confirmed position dict from get_open_position
    logger: logging.Logger,
    stop_loss_price: Decimal | str | None = None,  # Allow '0' string for cancel
    take_profit_price: Decimal | str | None = None,  # Allow '0' string for cancel
    trailing_stop_distance: Decimal | str | None = None,  # Allow '0' string for cancel
    tsl_activation_price: Decimal | str | None = None,  # Allow '0' string for cancel
) -> bool:
    """Internal helper to set SL, TP, or TSL for an existing position using Bybit's V5 API
    (`/v5/position/set-trading-stop`) via CCXT's `private_post` method with retries.
    Returns True on success (API call successful, retCode=0), False on failure.
    Prioritizes TSL if valid TSL and SL are provided. Pass '0' string to cancel.
    """
    lg = logger
    is_contract = market_info.get('is_contract', False)
    if not is_contract:
        lg.warning(f"Position protection (SL/TP/TSL) is typically for contract markets. Skipping for {symbol}.")
        return False  # Not applicable, not a failure

    # --- Validate Inputs ---
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    pos_side = position_info.get('side')
    pos_size_str = position_info.get('contracts', position_info.get('info', {}).get('size'))
    if pos_side not in ['long', 'short'] or pos_size_str is None:
        lg.error(f"Cannot set protection for {symbol}: Invalid or missing position side ('{pos_side}') or size in position_info.")
        return False

    # Helper to check if input value is valid for setting or canceling
    def is_valid_or_cancel(val: Decimal | str | None, allow_zero: bool = False) -> bool:
        if val is None: return False  # No intent to set/cancel
        if isinstance(val, str) and val.strip() == '0': return True  # Explicit cancel signal
        try:
            val_dec = Decimal(str(val).strip())
            if allow_zero: return val_dec >= 0  # Allow 0 for TSL activation/distance
            else: return val_dec > 0  # Require > 0 for SL/TP prices
        except (InvalidOperation, ValueError, TypeError):
            return False  # Invalid format

    has_sl_intent = stop_loss_price is not None
    has_tp_intent = take_profit_price is not None
    has_tsl_intent = trailing_stop_distance is not None or tsl_activation_price is not None

    is_sl_valid_input = has_sl_intent and is_valid_or_cancel(stop_loss_price, allow_zero=False)
    is_tp_valid_input = has_tp_intent and is_valid_or_cancel(take_profit_price, allow_zero=False)
    # TSL requires distance > 0 (or '0') and activation >= 0 (or '0')
    is_tsl_dist_valid = trailing_stop_distance is not None and is_valid_or_cancel(trailing_stop_distance, allow_zero=True) and (str(trailing_stop_distance).strip() == '0' or Decimal(str(trailing_stop_distance)) > 0)
    is_tsl_act_valid = tsl_activation_price is not None and is_valid_or_cancel(tsl_activation_price, allow_zero=True)
    # Valid TSL input requires *both* valid distance and activation price if setting/modifying
    # If only one is provided with intent, it's ambiguous, unless canceling ('0')
    is_tsl_valid_input = has_tsl_intent and is_tsl_dist_valid and is_tsl_act_valid

    if not is_sl_valid_input and not is_tp_valid_input and not is_tsl_valid_input:
        if has_tsl_intent and (not is_tsl_dist_valid or not is_tsl_act_valid):
             lg.warning(f"TSL intent detected but requires both valid distance (>0 or '0', got '{trailing_stop_distance}') and activation (>=0 or '0', got '{tsl_activation_price}'). Skipping TSL.")
        else:
             lg.info(f"No valid protection parameters provided for {symbol}. No protection set/modified.")
        return True  # Return True if no valid action requested

    # --- Prepare API Parameters ---
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    position_idx = 0  # Default for One-Way mode
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
        if position_idx not in [0, 1, 2]: position_idx = 0
    except (ValueError, TypeError): pass

    params = {
        'category': category,
        'symbol': market_info['id'],
        'tpslMode': 'Full',  # Apply to the whole position
        'slTriggerBy': 'LastPrice',
        'tpTriggerBy': 'LastPrice',
        'slOrderType': 'Market',
        'tpOrderType': 'Market',
        'positionIdx': position_idx
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} position, Idx: {position_idx}):"]

    # --- Format and Add Parameters ---
    tsl_added_to_params = False
    sl_added_to_params = False
    tp_added_to_params = False
    try:
        # Helper to format price/value using exchange's precision rules
        def format_api_value(value: Decimal | str | None, precision_type: str = 'price') -> str | None:
            if value is None: return None
            val_str = str(value).strip()
            if val_str == '0': return '0'  # Pass cancel signal directly

            try:
                val_dec = Decimal(val_str)
                # Validate non-negative (already done by is_valid_or_cancel, but double check)
                if val_dec < 0:
                    lg.warning(f"Invalid negative value '{val_dec}' provided for {precision_type}. Skipping.")
                    return None
                if val_dec == 0 and precision_type not in ['amount', 'distance']:  # Only allow 0 for distance/activation if needed
                    return '0'  # Treat 0 as cancel for prices

                # Use appropriate formatting helper (price precision for prices and distance)
                return exchange.price_to_precision(symbol, float(val_dec))
            except (InvalidOperation, ValueError, TypeError) as fmt_err:
                lg.warning(f"Could not format value '{value}' using exchange precision ({precision_type}): {fmt_err}. Skipping.")
                return None

        # Trailing Stop Loss (Set first, as it might override fixed SL)
        if is_tsl_valid_input:
            formatted_tsl_distance = format_api_value(trailing_stop_distance, 'amount')  # Distance uses price precision
            formatted_activation_price = format_api_value(tsl_activation_price, 'price')

            # Check if both are valid after formatting (format_api_value returns None on error)
            if formatted_tsl_distance is not None and formatted_activation_price is not None:
                params['trailingStop'] = formatted_tsl_distance
                params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Distance={formatted_tsl_distance}, Activation={formatted_activation_price}")
                tsl_added_to_params = True
            else:
                lg.error(f"Failed to format valid TSL parameters for {symbol}. Cannot set TSL.")
                is_tsl_valid_input = False  # Mark TSL as failed

        # Fixed Stop Loss - Add only if SL intent exists AND TSL was NOT successfully added OR TSL is being cancelled ('0')
        if is_sl_valid_input:
            tsl_is_active_and_not_cancelling = tsl_added_to_params and params.get('trailingStop') != '0'
            if not tsl_is_active_and_not_cancelling:
                formatted_sl = format_api_value(stop_loss_price, 'price')
                if formatted_sl is not None:
                    params['stopLoss'] = formatted_sl
                    log_parts.append(f"  Fixed SL: {formatted_sl}")
                    sl_added_to_params = True
                else:  # Formatting failed
                    if has_sl_intent: is_sl_valid_input = False
            elif has_sl_intent:
                 lg.warning(f"Both valid 'stopLoss' and active 'trailingStop' provided for {symbol}. Prioritizing TSL. Fixed 'stopLoss' ignored.")
                 is_sl_valid_input = False

        # Fixed Take Profit
        if is_tp_valid_input:
            formatted_tp = format_api_value(take_profit_price, 'price')
            if formatted_tp is not None:
                params['takeProfit'] = formatted_tp
                log_parts.append(f"  Fixed TP: {formatted_tp}")
                tp_added_to_params = True
            else:  # Formatting failed
                if has_tp_intent: is_tp_valid_input = False

    except Exception as fmt_err:
        lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
        return False

    # Check if any protection is actually being set after formatting and prioritization
    if not sl_added_to_params and not tp_added_to_params and not tsl_added_to_params:
        lg.warning(f"No valid protection parameters could be formatted or remain after adjustments for {symbol}. No API call made.")
        original_intent_present = has_sl_intent or has_tp_intent or has_tsl_intent
        valid_params_remain = is_sl_valid_input or is_tp_valid_input or is_tsl_valid_input
        if original_intent_present and not valid_params_remain: return False  # Intent failed
        else: return True  # No intent or intent was valid but superseded (e.g., SL by TSL)

    # Log the attempt
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={params})")

    # --- Call Bybit V5 API Endpoint with Retries ---
    response = None
    last_error = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = exchange.private_post('/v5/position/set-trading-stop', params)
            lg.debug(f"Set protection raw response (Attempt {attempt + 1}): {response}")

            # --- Check Response ---
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown Error')
            ret_ext = response.get('retExtInfo', {})

            if ret_code == 0:
                # Check for "NotModified" messages which indicate success but no change
                if "notmodified" in ret_msg.lower().replace(" ", ""):
                    lg.info(f"{NEON_YELLOW}Position protection already set to target values for {symbol} (Exchange confirmation).{RESET}")
                else:
                    lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}. Response: '{ret_msg}'{RESET}")
                return True  # Success
            else:
                # Handle specific retryable errors or log non-retryable ones
                lg.warning(f"Failed attempt {attempt + 1} to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}")
                last_error = f"{ret_msg} (Code: {ret_code}) Ext: {ret_ext}"
                # Decide if retryable based on code? For now, retry all non-zero codes.
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Retrying protection set in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                     lg.error(f"{NEON_RED}Max retries reached. Failed to set protection for {symbol}: {last_error}{RESET}")
                     # Add hints based on common error codes
                     if ret_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint (110043): Check trigger prices (SL below entry for long?), `retExtInfo`.{RESET}")
                     elif ret_code == 110025: lg.error(f"{NEON_YELLOW} >> Hint (110025): Position might have closed or `positionIdx` mismatch?{RESET}")
                     elif ret_code == 110055: lg.error(f"{NEON_YELLOW} >> Hint (110055): Ensure 'positionIdx' matches account's Position Mode.{RESET}")
                     elif ret_code == 110013: lg.error(f"{NEON_YELLOW} >> Hint (110013): Parameter Error. Check SL/TP/TSL values (tick size, side), `activePrice`.{RESET}")
                     elif ret_code == 110036: lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price invalid (too close, wrong side?).{RESET}")
                     elif ret_code == 110086: lg.error(f"{NEON_YELLOW} >> Hint (110086): SL price cannot be the same as TP price.{RESET}")
                     elif "trailing stop value invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: TSL distance invalid (too small/large?).{RESET}")
                     elif "stop loss price is invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: Fixed SL price invalid (wrong side, tick size?).{RESET}")
                     elif "take profit price is invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: Fixed TP price invalid (wrong side, tick size?).{RESET}")
                     return False  # Failed after retries

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            lg.warning(f"Network error setting protection (Attempt {attempt + 1}): {e}")
            last_error = str(e)
            if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"{NEON_RED}Max retries reached setting protection after network errors.{RESET}"); return False
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 2
            try:  # Parse wait time
                if 'try again in' in str(e).lower():
                     wait_ms_str = str(e).lower().split('try again in')[1].split('ms')[0].strip()
                     wait_time = max(1, int(int(wait_ms_str) / 1000 + 1))
            except Exception: pass
            lg.warning(f"Rate limit setting protection. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except ccxt.ExchangeError as e:  # Catch errors from private_post call itself
            lg.warning(f"Exchange error setting protection (Attempt {attempt + 1}): {e}")
            last_error = str(e)
            if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)  # Retry potentially recoverable exchange errors
            else: lg.error(f"{NEON_RED}Max retries reached setting protection after ExchangeError: {e}{RESET}"); return False
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
            return False  # Fail immediately on unexpected errors

    # Should not be reached if logic is correct, but acts as a fallback
    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after loop. Last error: {last_error}{RESET}")
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,  # Pass the confirmed position info
    config: dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Decimal | None = None  # Allow passing pre-calculated TP price
) -> bool:
    """Calculates TSL parameters (activation price, distance) based on config and position,
    then calls the internal `_set_position_protection` helper function to set TSL (and optionally TP).
    Returns True if the protection API call is attempted successfully, False otherwise.
    """
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss is disabled in config for {symbol}. Skipping TSL setup.")
        return False  # Return False indicating TSL wasn't set due to config

    # --- Get TSL parameters from config ---
    try:
        callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format in config for {symbol}: {e}. Cannot calculate TSL.{RESET}")
        return False

    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_callback_rate ({callback_rate}) in config for {symbol}. Must be positive.{RESET}")
        return False
    if activation_percentage < 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_activation_percentage ({activation_percentage}) in config for {symbol}. Must be non-negative.{RESET}")
        return False

    # --- Extract required position details ---
    try:
        entry_price_str = position_info.get('entryPrice', position_info.get('info', {}).get('avgPrice'))
        side = position_info.get('side')

        if entry_price_str is None or side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing required position info (entryPrice, side) to calculate TSL for {symbol}. Position: {position_info}{RESET}")
            return False

        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0:
            lg.error(f"{NEON_RED}Invalid entry price ({entry_price}) from position info for {symbol}. Cannot calculate TSL.{RESET}")
            return False

    except (InvalidOperation, TypeError, ValueError, KeyError) as e:
        lg.error(f"{NEON_RED}Error parsing position info for TSL calculation ({symbol}): {e}. Position: {position_info}{RESET}")
        return False

    # --- Calculate TSL parameters for Bybit API ---
    try:
        # Use helper methods from a temporary analyzer instance for precision/tick size
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        dummy_df.index.name = 'timestamp'
        if not market_info: raise ValueError("Market info not available.")
        temp_analyzer = TradingAnalyzer(df=dummy_df, logger=lg, config=config, market_info=market_info)
        price_precision = temp_analyzer.get_price_precision()
        min_tick_size = temp_analyzer.get_min_tick_size()
        if min_tick_size <= 0: raise ValueError(f"Invalid min tick size ({min_tick_size}).")

        # 1. Calculate Activation Price
        activation_price = None
        if activation_percentage > 0:
            activation_offset = entry_price * activation_percentage
            if side == 'long':
                raw_activation = entry_price + activation_offset
                activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                if activation_price <= entry_price: activation_price = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
            else:  # short
                raw_activation = entry_price - activation_offset
                activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                if activation_price >= entry_price: activation_price = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)
        else:  # Immediate activation
            activation_price = Decimal('0')
            lg.info(f"TSL activation percentage is zero for {symbol}, setting immediate activation (API value '0').")

        if activation_price is None or activation_price < 0:
            raise ValueError(f"Calculated TSL activation price ({activation_price}) is invalid.")

        # 2. Calculate Trailing Stop Distance (absolute price points)
        trailing_distance_raw = entry_price * callback_rate
        # Round distance UP to ensure it's at least the calculated value, respecting ticks
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size

        if trailing_distance < min_tick_size:
            lg.warning(f"{NEON_YELLOW}Calculated TSL distance {trailing_distance} < min tick {min_tick_size}. Adjusting to min tick size.{RESET}")
            trailing_distance = min_tick_size
        elif trailing_distance <= 0:
            raise ValueError(f"Calculated TSL distance is zero or negative ({trailing_distance}).")

        # Format for logging
        act_price_log = f"{activation_price:.{price_precision}f}" if activation_price > 0 else '0 (Immediate)'
        trail_dist_log = f"{trailing_distance:.{price_precision}f}"
        tp_log = f"{take_profit_price:.{price_precision}f}" if take_profit_price and take_profit_price > 0 else "N/A"

        lg.info(f"Calculated TSL Params for {symbol} ({side.upper()}):")
        lg.info(f"  Entry Price: {entry_price:.{price_precision}f}")
        lg.info(f"  => Activation Price (API): {act_price_log}")
        lg.info(f"  => Trailing Distance (API): {trail_dist_log}")
        lg.info(f"  Take Profit (Optional): {tp_log}")

        # 3. Call the helper function to set TSL (and TP if provided)
        # Explicitly set fixed SL to '0' (cancel) when enabling TSL
        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info,
            position_info=position_info, logger=lg,
            stop_loss_price='0',  # Cancel fixed SL
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price  # Can be 0
        )

    except ValueError as ve:
        lg.error(f"{NEON_RED}Error calculating TSL parameters for {symbol}: {ve}{RESET}")
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""
    lg = logger  # Use the symbol-specific logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Market Info & Data ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping analysis cycle.{RESET}")
        return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
        lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe for {symbol}.")
        return

    # Determine required kline history (ensure enough for longest lookback + buffer)
    kline_limit = 500  # Example: Ensure enough for long EMAs, Fib, etc.

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50:  # Need a reasonable minimum history
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping analysis cycle.{RESET}")
        return

    # Fetch current price
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close from klines as fallback.{RESET}")
        try:
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0:
                current_price = Decimal(str(last_close_val))
                lg.info(f"Using last close price: {current_price}")
            else:
                lg.error(f"{NEON_RED}Last close price ({last_close_val}) is also invalid. Cannot proceed for {symbol}.{RESET}")
                return
        except (IndexError, ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"{NEON_RED}Error getting last close price for {symbol}: {e}. Cannot proceed.{RESET}")
            return

    # Fetch order book if enabled and weighted
    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    orderbook_weight = Decimal('0')
    with contextlib.suppress(Exception): orderbook_weight = Decimal(str(active_weights.get("orderbook", 0)))
    if config.get("indicators", {}).get("orderbook", False) and orderbook_weight != 0:
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config.get("orderbook_limit", 25), lg)

    # --- 2. Analyze Data & Generate Signal ---
    analyzer = TradingAnalyzer(
        df=klines_df.copy(),  # Pass a copy
        logger=lg, config=config, market_info=market_info
    )

    if not analyzer.indicator_values or pd.isna(analyzer.indicator_values.get("Close")):  # Check if close price is valid
        lg.error(f"{NEON_RED}Indicator calculation failed or produced invalid latest values for {symbol}. Skipping signal generation.{RESET}")
        return

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr_float = analyzer.indicator_values.get("ATR")

    # --- 3. Log Analysis Summary ---
    atr_log = f"{current_atr_float:.{price_precision + 1}f}" if current_atr_float and pd.notna(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision}f}" if tp_potential else 'N/A'
    lg.info(f"ATR: {atr_log}")
    lg.info(f"Potential Initial SL (for new trade sizing): {sl_pot_log}")
    lg.info(f"Potential Initial TP (for new trade): {tp_pot_log}")
    tsl_enabled = config.get('enable_trailing_stop')
    be_enabled = config.get('enable_break_even')
    ma_exit_enabled = config.get('enable_ma_cross_exit')
    lg.info(f"Trailing Stop: {'Enabled' if tsl_enabled else 'Disabled'} | Break Even: {'Enabled' if be_enabled else 'Disabled'} | MA Cross Exit: {'Enabled' if ma_exit_enabled else 'Disabled'}")

    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # --- Get Current Position Status ---
    open_position = get_open_position(exchange, symbol, lg)  # Enhanced function returns SL/TP/TSL info

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Open Position: Initiating Trade Sequence for {symbol} ***")

            # --- Pre-Trade Checks & Setup ---
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot proceed, failed to fetch balance or balance is zero/negative ({balance} {QUOTE_CURRENCY}).{RESET}")
                return
            if sl_potential is None:
                lg.error(f"{NEON_RED}Trade Aborted ({signal}): Potential Initial SL calculation failed (ATR invalid?). Cannot determine risk.{RESET}")
                return

            if market_info.get('is_contract', False):
                try: leverage = int(config.get("leverage", 1))
                except (ValueError, TypeError): leverage = 1
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set/confirm leverage to {leverage}x.{RESET}")
                        return
                else: lg.warning(f"Leverage setting skipped: Configured leverage <= 0 ({leverage}).")
            else: lg.info("Leverage setting skipped (Spot market).")

            position_size = calculate_position_size(
                balance=balance, risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_potential, entry_price=current_price,
                market_info=market_info, exchange=exchange, logger=lg
            )
            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({signal}): Invalid position size calculated ({position_size}). Check logs.{RESET}")
                return

            # --- Place Initial Market Order ---
            lg.info(f"==> Placing {signal} market order for {symbol} | Size: {position_size} <==")
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

                lg.info(f"Attempting to confirm position for {symbol} after order {order_id}...")
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        entry_price_actual_str = confirmed_position.get('entryPrice', confirmed_position.get('info', {}).get('avgPrice'))
                        pos_size_actual_str = confirmed_position.get('contracts', confirmed_position.get('info', {}).get('size'))
                        entry_price_actual = Decimal('0')
                        valid_entry = False
                        if entry_price_actual_str and pos_size_actual_str:
                            try:
                                entry_price_actual = Decimal(str(entry_price_actual_str))
                                pos_size_actual = Decimal(str(pos_size_actual_str))  # Absolute size
                                if entry_price_actual > 0 and pos_size_actual > 0: valid_entry = True
                                else: lg.error(f"Confirmed position invalid entry/size: Entry={entry_price_actual}, Size={pos_size_actual}")
                            except Exception as parse_err: lg.error(f"Error parsing confirmed pos entry/size: {parse_err}")
                        else: lg.error("Confirmed position missing entryPrice or size.")

                        if valid_entry:
                            lg.info(f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_price_actual:.{price_precision}f}, Size: {pos_size_actual}{RESET}")
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting Trailing Stop Loss (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(
                                    exchange, symbol, market_info, confirmed_position, config, lg, tp_actual
                                )
                            else:
                                lg.info(f"Setting Fixed SL ({sl_actual}) and TP ({tp_actual})...")
                                if sl_actual or tp_actual:
                                    protection_set_success = _set_position_protection(
                                        exchange, symbol, market_info, confirmed_position, lg,
                                        stop_loss_price=sl_actual, take_profit_price=tp_actual
                                    )
                                else:
                                    lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed. No fixed protection set.{RESET}")
                                    protection_set_success = True  # No error, just no protection

                            if protection_set_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({signal}) ===")
                            else:
                                lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION ({signal}) ===")
                                lg.warning(f"{NEON_YELLOW}Position open without automated protection. Manual monitoring required!{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Trade placed, but confirmed position data invalid. Cannot set protection.{RESET}")
                            lg.warning(f"{NEON_YELLOW}Manual check needed! Position might be open without protection.{RESET}")

                    except Exception as post_trade_err:
                        lg.error(f"{NEON_RED}Error during post-trade processing: {post_trade_err}{RESET}", exc_info=True)
                        lg.warning(f"{NEON_YELLOW}Position might be open without protection. Manual check needed!{RESET}")
                else:
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM position after {POSITION_CONFIRM_DELAY}s!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have failed, filled partially, or API delay. Manual investigation required!{RESET}")
                    try:  # Attempt to fetch order status for clues
                        order_status = exchange.fetch_order(order_id, symbol)
                        lg.info(f"Status of order {order_id}: {order_status.get('status', 'N/A')}, Filled: {order_status.get('filled', 'N/A')}")
                    except Exception as fetch_order_err: lg.warning(f"Could not fetch status for order {order_id}: {fetch_order_err}")
            else:
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({signal}). See previous logs. ===")
        else:  # Signal is HOLD, no position
            lg.info("Signal is HOLD and no open position. No trade action taken.")

    # --- Scenario 2: Existing Open Position Found ---
    else:  # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size_str = open_position.get('contracts', open_position.get('info', {}).get('size', 'N/A'))
        entry_price_str = open_position.get('entryPrice', open_position.get('info', {}).get('avgPrice', 'N/A'))
        # Use parsed Decimal values from enhanced get_open_position
        current_sl_price_dec = open_position.get('stopLossPriceDecimal')  # Decimal or None
        current_tp_price_dec = open_position.get('takeProfitPriceDecimal')  # Decimal or None
        tsl_distance_dec = open_position.get('trailingStopLossDistanceDecimal', Decimal(0))  # Default to 0
        is_tsl_active = tsl_distance_dec > 0

        sl_log_str = f"{current_sl_price_dec:.{price_precision}f}" if current_sl_price_dec else 'N/A'
        tp_log_str = f"{current_tp_price_dec:.{price_precision}f}" if current_tp_price_dec else 'N/A'
        lg.info(f"Existing {pos_side.upper()} position found. Size: {pos_size_str}, Entry: {entry_price_str}, SL: {sl_log_str}, TP: {tp_log_str}, TSL Active: {is_tsl_active}")

        # --- Check Exit Conditions ---
        exit_signal_triggered = False
        if (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY"):
            exit_signal_triggered = True
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position. ***{RESET}")

        ma_cross_exit = False
        if not exit_signal_triggered and config.get("enable_ma_cross_exit", False):
            ema_short = analyzer.indicator_values.get("EMA_Short")
            ema_long = analyzer.indicator_values.get("EMA_Long")
            if pd.notna(ema_short) and pd.notna(ema_long):
                if pos_side == 'long' and ema_short < ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bearish): Short EMA ({ema_short:.{price_precision}f}) < Long EMA ({ema_long:.{price_precision}f}). Closing LONG. ***{RESET}")
                elif pos_side == 'short' and ema_short > ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bullish): Short EMA ({ema_short:.{price_precision}f}) > Long EMA ({ema_long:.{price_precision}f}). Closing SHORT. ***{RESET}")
            else: lg.warning("MA cross exit check skipped: EMA values unavailable.")

        # --- Execute Position Close if Exit Condition Met ---
        if exit_signal_triggered or ma_cross_exit:
            lg.info(f"Attempting to close {pos_side} position with a market order (reduceOnly=True)...")
            try:
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close_str = open_position.get('contracts', open_position.get('info', {}).get('size'))
                if size_to_close_str is None: raise ValueError("Could not determine size to close.")
                size_to_close = abs(Decimal(str(size_to_close_str)))  # Use absolute size
                if size_to_close <= 0: raise ValueError(f"Position size {size_to_close_str} is zero/invalid.")

                close_order = place_trade(
                    exchange, symbol, close_side_signal, size_to_close, market_info, lg, reduce_only=True
                )
                if close_order:
                     order_id = close_order.get('id', 'N/A')
                     if close_order.get('info', {}).get('reason') == 'Position not found on close attempt':
                          lg.info(f"{NEON_GREEN}Position CLOSE confirmed (was likely already closed).{RESET}")
                     else:
                          lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully. Order ID: {order_id}{RESET}")
                          lg.info(f"{NEON_YELLOW}Verify position closure on exchange if necessary.{RESET}")
                else:
                     lg.error(f"{NEON_RED}Failed to place position CLOSE order. Manual intervention required!{RESET}")
            except (InvalidOperation, ValueError) as ve: lg.error(f"{NEON_RED}Error preparing to close position: {ve}{RESET}")
            except Exception as e: lg.error(f"{NEON_RED}Unexpected error closing position: {e}{RESET}", exc_info=True)

        # --- Manage Existing Position (No Exit Signal) ---
        elif signal == "HOLD" or (signal == "BUY" and pos_side == 'long') or (signal == "SELL" and pos_side == 'short'):
            lg.info(f"Signal ({signal}) allows holding existing {pos_side} position.")

            # --- Break-Even Stop Logic ---
            # Check only if enabled AND TSL is NOT currently active AND fixed SL exists (or no SL exists)
            if config.get("enable_break_even", False) and not is_tsl_active:
                lg.debug("Checking Break-Even conditions...")
                try:
                    if not entry_price_str or str(entry_price_str).lower() == 'n/a': raise ValueError("Missing entry price")
                    entry_price = Decimal(str(entry_price_str))
                    if entry_price <= 0: raise ValueError(f"Invalid entry price ({entry_price})")
                    if current_atr_float is None or pd.isna(current_atr_float) or current_atr_float <= 0: raise ValueError("Invalid ATR")
                    current_atr_decimal = Decimal(str(current_atr_float))
                    if min_tick_size <= 0: raise ValueError(f"Invalid min_tick_size ({min_tick_size})")

                    profit_target_atr_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                    profit_target_price_diff = profit_target_atr_multiple * current_atr_decimal

                    lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision}f}, Target Diff={profit_target_price_diff:.{price_precision}f}")

                    if price_diff >= profit_target_price_diff:
                        lg.info("Break-even profit target reached.")
                        tick_offset = min_tick_size * offset_ticks
                        if pos_side == 'long':
                            be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP)
                        else:  # short
                            be_stop_price = (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                        if be_stop_price <= 0: raise ValueError(f"Calculated BE stop price invalid ({be_stop_price})")

                        update_be = False
                        if current_sl_price_dec is None:  # No current SL, set BE SL
                            update_be = True
                            lg.info("Profit target hit & no current SL. Setting BE SL.")
                        elif pos_side == 'long' and be_stop_price > current_sl_price_dec:
                            update_be = True
                            lg.info(f"Profit target hit. Current SL {current_sl_price_dec} < Target BE SL {be_stop_price}. Updating.")
                        elif pos_side == 'short' and be_stop_price < current_sl_price_dec:
                            update_be = True
                            lg.info(f"Profit target hit. Current SL {current_sl_price_dec} > Target BE SL {be_stop_price}. Updating.")
                        else: lg.debug(f"BE Triggered, but current SL ({current_sl_price_dec}) already better than target ({be_stop_price}).")

                        if update_be:
                            lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even at {be_stop_price:.{price_precision}f} ***{RESET}")
                            success = _set_position_protection(
                                exchange, symbol, market_info, open_position, lg,
                                stop_loss_price=be_stop_price,  # Set the new BE SL
                                take_profit_price=current_tp_price_dec,  # Preserve existing TP
                                trailing_stop_distance='0',  # Ensure TSL is cancelled
                                tsl_activation_price='0'
                            )
                            if success: lg.info(f"{NEON_GREEN}Break-Even SL update successful.{RESET}")
                            else: lg.error(f"{NEON_RED}Failed to update SL to Break-Even.{RESET}")
                    else: lg.debug("BE Profit target not yet reached.")

                except ValueError as ve: lg.warning(f"Cannot check break-even: {ve}")
                except Exception as be_err: lg.error(f"Error during Break-Even check: {be_err}", exc_info=True)
            elif is_tsl_active:
                lg.debug("Break-Even check skipped: Trailing Stop Loss is active.")
            # else: BE disabled or TSL active

    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")


# --- Main Execution ---
def main() -> None:
    """Main function to run the trading bot."""
    parser = argparse.ArgumentParser(description="Enhanced Bybit Trading Bot")
    parser.add_argument("symbol", help="Trading symbol (e.g., BTC/USDT:USDT)")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable DEBUG level logging to console"
    )
    args = parser.parse_args()

    # Setup logger with appropriate level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(args.symbol, level=log_level)

    logger.info("*" * 60)
    logger.info(f"Initializing Live Trading Bot for {args.symbol}")
    logger.info(f"Using Config: {CONFIG_FILE}")
    logger.info("*" * 60)

    exchange = initialize_exchange(logger)
    if not exchange:
        logger.critical("Failed to initialize exchange. Exiting.")
        sys.exit(1)

    # Validate symbol using market info
    market_info = get_market_info(exchange, args.symbol, logger)
    if not market_info:
         logger.critical(f"Symbol {args.symbol} is not valid or available on {exchange.id}. Exiting.")
         sys.exit(1)
    # Ensure the symbol format matches CCXT's internal format if needed
    # symbol_ccxt = market
