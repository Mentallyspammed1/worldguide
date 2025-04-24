# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).

# --- Core Libraries ---
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from zoneinfo import ZoneInfo # Requires tzdata package

# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy
import pandas as pd # Requires pandas
import pandas_ta as ta # Requires pandas_ta
import requests # Requires requests
# import websocket # Requires websocket-client (Imported but unused, placeholder for potential future WS integration)
import ccxt # Requires ccxt
from colorama import Fore, Style, init # Requires colorama
from dotenv import load_dotenv # Requires python-dotenv
# Note: requests automatically uses urllib3, no need for separate import unless customizing adapters/retries outside ccxt

# --- Initialize Environment and Settings ---
getcontext().prec = 28  # Set Decimal precision for calculations
init(autoreset=True) # Initialize Colorama for colored console output
load_dotenv() # Load environment variables from a .env file

# --- Constants ---
# API Credentials (Loaded from .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# Configuration and Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago") # << IMPORTANT: Set to your local timezone or preferred timezone for logs >>

# API Interaction Settings
MAX_API_RETRIES = 3 # Max retries for recoverable API errors (Network, 429, 5xx)
RETRY_DELAY_SECONDS = 5 # Base delay between retries (may increase for rate limits)
POSITION_CONFIRM_DELAY_SECONDS = 8 # Wait time after placing order before confirming position state
LOOP_DELAY_SECONDS = 15 # Min time between the end of one cycle and the start of the next

# Timeframe Settings
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Intervals supported by the bot logic
CCXT_INTERVAL_MAP = { # Map bot intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling
DEFAULT_FETCH_LIMIT = 750 # Ensure enough data for indicator lookbacks (adjust based on strategy needs)
MAX_DF_LEN = 2000 # Max rows to keep in DataFrame to manage memory

# Default Strategy/Indicator Parameters (can be overridden by config.json)
# Volumatic Trend Params
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 1000
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0

# Order Block Params
DEFAULT_OB_SOURCE = "Wicks" # "Wicks" or "Bodys"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50

# Dynamically loaded from config: QUOTE_CURRENCY

# Neon Color Scheme for Logging Output
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Configuration Loading ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive API keys from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record and redacts API keys."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger instance for a given name (e.g., 'init' or symbol).
    Configures file logging (DEBUG level) and console logging (INFO level by default).
    Uses SensitiveFormatter to prevent API keys leaking into logs.

    Args:
        name: The name for the logger (used in log messages and filename).

    Returns:
        Configured logging.Logger instance.
    """
    safe_name = name.replace('/', '_').replace(':', '-') # Sanitize name for filename
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times if logger already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Set base level to capture all messages

    # File Handler - Logs everything (DEBUG level and above)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to print error if file logging fails
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")

    # Console Handler - Logs INFO level and above by default (less verbose)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Use standard date format
    )
    stream_handler.setFormatter(stream_formatter)
    # Set desired console log level (INFO for normal operation, DEBUG for detailed tracing)
    console_log_level = logging.INFO
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent messages propagating to the root logger
    return logger

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively ensures that all keys from the default configuration exist
    in the loaded configuration, adding missing keys with default values.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The default configuration dictionary.

    Returns:
        An updated configuration dictionary with all default keys present.
    """
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        # Recurse if both values are dictionaries
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file. Creates a default config file if
    it doesn't exist. Ensures all default keys are present in the loaded config
    and validates the 'interval' setting. Updates the file if keys were added
    or interval was corrected.

    Args:
        filepath: Path to the configuration JSON file.

    Returns:
        The loaded (and potentially updated) configuration dictionary.
    """
    default_config = {
        "interval": "5", # Default analysis interval (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS, # Use constant defined above
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Use constant
        "orderbook_limit": 25, # Depth for order book fetching (if used)
        "enable_trading": False, # << SAFETY: Default to False (dry run mode) >>
        "use_sandbox": True,     # << SAFETY: Default to True (use testnet) >>
        "risk_per_trade": 0.01, # Risk percentage (e.g., 0.01 = 1% of balance)
        "leverage": 20,          # Desired leverage (must be supported by exchange/symbol)
        "max_concurrent_positions": 1, # Max open positions for this script instance/symbol
        "quote_currency": "USDT", # Currency for balance/sizing (e.g., USDT, USDC)
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Use constant
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Use constant

        # --- Strategy Parameters (Volumatic Trend & OB) ---
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,
            "ob_source": DEFAULT_OB_SOURCE, # "Wicks" or "Bodys"
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005, # Price must be within 0.5% *beyond* OB edge for entry
            "ob_exit_proximity_factor": 1.001 # Price must be within 0.1% *beyond* OB edge for exit signal
        },

        # --- Position Protection Settings ---
        "protection": {
             "enable_trailing_stop": True,
             "trailing_stop_callback_rate": 0.005, # Trail distance as % (e.g., 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Activate TSL when profit is % of entry (e.g., 0.3%)
             "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0, # Move SL to BE when profit >= X * ATR
             "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry (uses market tick size)
             "initial_stop_loss_atr_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
             "initial_take_profit_atr_multiple": 0.7 # ATR multiple for initial TP (optional fixed target)
        }
    }

    config_needs_saving = False
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using default values.{RESET}")
            return default_config

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config_from_file = json.load(f)

        # Ensure all default keys exist, preserving existing values
        updated_config = _ensure_config_keys(config_from_file, default_config)

        # Check if any keys were added from default during the merge
        if updated_config != config_from_file:
            config_needs_saving = True
            print(f"{NEON_YELLOW}Config file updated with missing default keys.{RESET}")

        # Validate interval value after merging defaults
        interval_from_config = updated_config.get("interval")
        if interval_from_config not in VALID_INTERVALS:
            print(f"{NEON_RED}Invalid interval '{interval_from_config}' in config. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True # Mark for saving if interval was corrected

        # Save the updated config back to the file if changes were made
        if config_needs_saving:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{NEON_YELLOW}Saved updated configuration to: {filepath}{RESET}")
             except IOError as e:
                 print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

        return updated_config

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Attempting to recreate default config.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e_create:
             print(f"{NEON_RED}Error creating default config file after load error: {e_create}. Using default values.{RESET}")
             return default_config
    except Exception as e:
        print(f"{NEON_RED}Unexpected error loading config: {e}. Using default values.{RESET}")
        return default_config


# --- Load Configuration Globally (Can be updated later if dynamic reloading is needed) ---
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

# --- Logger Setup (Instantiate the 'init' logger) ---
init_logger = setup_logger("init") # Logger for initial setup phases

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object.
    Handles API keys, sandbox mode, rate limits, timeouts, market loading,
    and performs an initial connection test by fetching balance.

    Args:
        logger: The logger instance to use for messages.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear', # Assume linear contracts (USDT/USDC perpetuals) unless overridden
                'adjustForTimeDifference': True, # Auto-adjust for clock skew
                # Increased timeouts for potentially slow API calls
                'fetchTickerTimeout': 15000, # ms
                'fetchBalanceTimeout': 20000, # ms
                'createOrderTimeout': 30000, # ms
                'cancelOrderTimeout': 20000, # ms
            }
        }
        exchange_class = ccxt.bybit # Explicitly use the Bybit class
        exchange = exchange_class(exchange_options)

        # Set sandbox mode based on config
        if CONFIG.get('use_sandbox', True): # Default to sandbox if key is missing
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
             lg.warning(f"{NEON_RED}USING LIVE TRADING ENVIRONMENT{RESET}")

        # Load markets with retries
        lg.info(f"Loading markets for {exchange.id}...")
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                exchange.load_markets(reload=True if attempt > 0 else False) # Force reload on retries
                lg.info(f"Markets loaded successfully for {exchange.id}.")
                break # Success
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error loading markets (Attempt {attempt+1}/{MAX_API_RETRIES+1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.critical(f"{NEON_RED}Max retries reached loading markets for {exchange.id}. Error: {e}. Exiting.{RESET}")
                    return None # Critical failure
            except ccxt.ExchangeError as e:
                 lg.critical(f"{NEON_RED}Exchange error loading markets: {e}. Exiting.{RESET}")
                 return None
            except Exception as e:
                 lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                 return None
        if not exchange.markets: # Check if markets actually loaded
             lg.critical(f"{NEON_RED}Market loading failed, exchange.markets is empty. Exiting.{RESET}")
             return None

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys by fetching balance
        lg.info(f"Attempting initial balance fetch (Quote Currency: {QUOTE_CURRENCY})...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_val:.4f})")
            else:
                 # fetch_balance returning None after retries usually indicates a persistent issue
                 lg.critical(f"{NEON_RED}Initial balance fetch failed after retries. Check API key permissions, network, or account type.{RESET}")
                 return None # Treat persistent balance failure as critical
        except ccxt.AuthenticationError as auth_err:
             lg.critical(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             lg.critical(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
             return None
        except Exception as balance_err:
             # Non-auth errors during initial fetch are warnings, but problematic if persistent
             lg.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type if trading fails.{RESET}")
             # Proceed cautiously, but trading might fail later

        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None


# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using the exchange's ticker.
    Includes retries for network issues and rate limits, and fallbacks for price sources.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: Logger instance for messages.

    Returns:
        The current price as a Decimal, or None if fetching fails.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            # lg.debug(f"Ticker data for {symbol}: {ticker}") # Uncomment for verbose debugging

            price = None
            # Prioritize 'last', then mid-price ('bid'+'ask')/2, then 'ask', then 'bid'
            last_price_str = ticker.get('last')
            bid_price_str = ticker.get('bid')
            ask_price_str = ticker.get('ask')

            # Try 'last' price
            if last_price_str is not None:
                try:
                    p = Decimal(str(last_price_str))
                    if p > 0: price = p; lg.debug(f"Using 'last' price: {p}")
                except Exception: lg.warning(f"Invalid 'last' price format: {last_price_str}")

            # Try bid/ask midpoint if 'last' failed or was invalid
            if price is None and bid_price_str is not None and ask_price_str is not None:
                try:
                    bid = Decimal(str(bid_price_str))
                    ask = Decimal(str(ask_price_str))
                    if bid > 0 and ask > 0 and ask >= bid: # Sanity check: ask >= bid
                        price = (bid + ask) / Decimal('2')
                        lg.debug(f"Using bid/ask midpoint: {price} (Bid: {bid}, Ask: {ask})")
                    else: lg.warning(f"Invalid bid/ask values: Bid={bid}, Ask={ask}")
                except Exception: lg.warning(f"Invalid bid/ask format: {bid_price_str}, {ask_price_str}")

            # Fallback to 'ask' price
            if price is None and ask_price_str is not None:
                 try:
                      p = Decimal(str(ask_price_str))
                      if p > 0: price = p; lg.warning(f"{NEON_YELLOW}Using 'ask' price fallback: {p}{RESET}")
                 except Exception: lg.warning(f"Invalid 'ask' price format: {ask_price_str}")

            # Fallback to 'bid' price
            if price is None and bid_price_str is not None:
                 try:
                      p = Decimal(str(bid_price_str))
                      if p > 0: price = p; lg.warning(f"{NEON_YELLOW}Using 'bid' price fallback: {p}{RESET}")
                 except Exception: lg.warning(f"Invalid 'bid' price format: {bid_price_str}")

            # Check if a valid price was found
            if price is not None and price > 0:
                # Price formatting/quantization happens later during order placement/SL setting
                return price
            else:
                lg.warning(f"Failed to get a valid positive price from ticker (Attempt {attempts + 1}).")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5 # Wait longer for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time - RETRY_DELAY_SECONDS) # Account for standard delay below
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Don't retry most exchange errors (e.g., bad symbol) immediately
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch a valid current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries, validation, and processing.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        timeframe: CCXT timeframe string (e.g., '1m', '5m', '1h').
        limit: Number of candles to fetch.
        logger: Logger instance.

    Returns:
        A pandas DataFrame with Decimal columns ['open', 'high', 'low', 'close', 'volume']
        and a DatetimeIndex (UTC), or an empty DataFrame on failure.
    """
    lg = logger
    if not exchange.has['fetchOHLCV']:
         lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
         return pd.DataFrame()

    ohlcv = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
            # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if ohlcv is not None and len(ohlcv) > 0:
                # Basic validation: Check if the last timestamp seems reasonably recent
                try:
                    last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    interval_duration = pd.Timedelta(exchange.parse_timeframe(timeframe), unit='s')
                    # Allow some reasonable lag (e.g., 5 intervals)
                    if (now_utc - last_ts) < interval_duration * 5:
                         lg.debug(f"Received {len(ohlcv)} klines. Last timestamp: {last_ts}")
                         break # Success
                    else:
                         lg.warning(f"Received {len(ohlcv)} klines, but last timestamp {last_ts} seems too old (> 5 intervals). Retrying...")
                except Exception as ts_err:
                    lg.warning(f"Error validating timestamp: {ts_err}. Proceeding cautiously.")
                    break # Proceed even if timestamp check fails
            else:
                lg.warning(f"fetch_ohlcv returned None or empty list for {symbol} (Attempt {attempt+1}). Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempt < MAX_API_RETRIES:
                lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors: {e}{RESET}")
                return pd.DataFrame() # Return empty DF on persistent network failure
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"Rate limit exceeded fetching klines for {symbol}: {e}. Retrying in {wait_time}s... (Attempt {attempt+1})")
            time.sleep(wait_time)
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
            # Depending on the error, might not be retryable (e.g., bad symbol)
            return pd.DataFrame() # Return empty DF on unrecoverable exchange error
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching klines: {e}{RESET}", exc_info=True)
            return pd.DataFrame() # Return empty DF on unexpected error

    if not ohlcv:
        lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
        return pd.DataFrame()

    # --- Data Processing ---
    try:
        # Ensure standard column names, handle potential missing 'turnover'
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Bybit sometimes includes 'turnover' (quote volume)
        if len(ohlcv[0]) > 6: columns.append('turnover')

        df = pd.DataFrame(ohlcv, columns=columns[:len(ohlcv[0])]) # Use only available columns

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric (using Decimal for price/volume precision)
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            if col in df.columns:
                # Apply Decimal conversion, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))

        # Validate data: drop rows with NaNs in key price columns or non-positive close
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0] # Ensure close price is positive
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{RESET}")
             return pd.DataFrame()

        # Ensure data is sorted chronologically (fetch_ohlcv usually returns oldest first)
        df.sort_index(inplace=True)

        # Limit DataFrame size to prevent memory issues
        if len(df) > MAX_DF_LEN:
             lg.debug(f"Trimming DataFrame from {len(df)} to {MAX_DF_LEN} rows.")
             df = df.iloc[-MAX_DF_LEN:]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df.copy() # Return a copy to avoid modifying cached data unintentionally

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on processing failure

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Retrieves market information for a symbol, including precision, limits,
    and contract type, with retries for network issues.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        logger: Logger instance.

    Returns:
        A dictionary containing market details, or None if the symbol is invalid
        or fetching fails after retries.
    """
    lg = logger
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            # Check if markets are loaded; reload if necessary or if symbol not found
            if not exchange.markets or symbol not in exchange.markets:
                 lg.info(f"Market info for {symbol} not loaded or missing. Reloading markets (Attempt {attempt+1})...")
                 exchange.load_markets(reload=True)

            # Check again after potential reload
            if symbol not in exchange.markets:
                 # If still not found after reload, it's likely an invalid symbol
                 if attempt == 0: continue # Allow one reload attempt before failing
                 lg.error(f"{NEON_RED}Market '{symbol}' still not found after reloading markets.{RESET}")
                 return None

            market = exchange.market(symbol)
            if market:
                # Enhance market info with derived flags for convenience
                market_type = market.get('type', 'unknown') # e.g., spot, swap, future
                is_linear = market.get('linear', False)
                is_inverse = market.get('inverse', False)
                # Consider swap/future as contracts, check 'contract' flag as fallback
                is_contract = market.get('contract', False) or market_type in ['swap', 'future']
                contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "Spot/Other"

                # Add derived flags to the market dictionary
                market['is_contract'] = is_contract
                market['contract_type_str'] = contract_type

                # Log key details
                lg.debug(
                    f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract Type={contract_type}, "
                    f"Precision(Price/Amount): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}, "
                    f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                    f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                    f"Contract Size: {market.get('contractSize', 'N/A')}"
                )
                return market # Success
            else:
                 # Should not happen if symbol is in exchange.markets, but handle defensively
                 lg.error(f"{NEON_RED}Market dictionary unexpectedly None for '{symbol}' even though key exists in markets.{RESET}")
                 return None # Treat as failure

        except ccxt.BadSymbol as e:
             lg.error(f"{NEON_RED}Symbol '{symbol}' not supported or invalid on {exchange.id}: {e}{RESET}")
             return None # Bad symbol is not retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
             if attempt < MAX_API_RETRIES:
                  lg.warning(f"Network error getting market info for {symbol} (Attempt {attempt+1}): {e}. Retrying...")
                  time.sleep(RETRY_DELAY_SECONDS)
             else:
                  lg.error(f"{NEON_RED}Max retries reached getting market info for {symbol} after network errors: {e}{RESET}")
                  return None
        except ccxt.ExchangeError as e:
            lg.warning(f"{NEON_YELLOW}Exchange error getting market info for {symbol}: {e}. Retrying...{RESET}")
            # Some exchange errors might be temporary, allow retry
            if attempt < MAX_API_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached getting market info for {symbol} after exchange errors: {e}{RESET}")
                return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
            return None # Unexpected errors usually not retryable

    return None # Should only be reached if all retries fail

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency, handling various Bybit V5
    API response structures and including retries.

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency code to fetch the balance for (e.g., 'USDT').
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetching fails after retries.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info = None
            available_balance_str = None
            found_structure = False

            # --- Try fetching with specific Bybit V5 account types ---
            # UNIFIED often holds USDT for linear contracts, CONTRACT is for derivatives.
            account_types_to_try = ['UNIFIED', 'CONTRACT']
            for acc_type in account_types_to_try:
                 try:
                     lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency}...")
                     # Bybit V5 uses 'accountType' in params for fetch_balance
                     balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                     lg.debug(f"Raw balance response (type {acc_type}): {balance_info}") # Verbose

                     # Structure 1: Standard CCXT (less common for Bybit V5 derivatives)
                     if currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free'] structure: {available_balance_str}")
                         found_structure = True; break

                     # Structure 2: Bybit V5 Unified/Contract ('info' field)
                     elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             if account.get('accountType') == acc_type and isinstance(account.get('coin'), list):
                                 for coin_data in account['coin']:
                                     if coin_data.get('coin') == currency:
                                         # Prefer 'availableToWithdraw' or 'availableBalance'
                                         free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                         if free is not None:
                                             available_balance_str = str(free)
                                             lg.debug(f"Found balance via V5 info.result.list[].coin[] ({acc_type}): {available_balance_str}")
                                             found_structure = True; break
                                 if found_structure: break # Exit outer loop if found
                         if found_structure: break # Exit account type loop if found
                         lg.debug(f"Currency '{currency}' not found within V5 info.result.list structure for type '{acc_type}'.")

                 except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                     lg.debug(f"API error fetching balance for type '{acc_type}': {e}. Trying next type.")
                     continue # Try next account type
                 except Exception as e:
                     lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}. Trying next type.")
                     continue

            # --- Fallback: Try default fetch_balance (no params) ---
            if not found_structure:
                 lg.debug(f"Fetching balance using default parameters for {currency} (fallback)...")
                 try:
                      balance_info = exchange.fetch_balance()
                      lg.debug(f"Raw balance response (default): {balance_info}") # Verbose
                      # Check standard structure again
                      if currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free'] (default fetch): {available_balance_str}")
                         found_structure = True
                      # Check top-level 'free' dict (older CCXT versions?)
                      elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                         available_balance_str = str(balance_info['free'][currency])
                         lg.debug(f"Found balance via top-level 'free' dict (default fetch): {available_balance_str}")
                         found_structure = True
                      # Check V5 structure again in default response
                      elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                           for account in balance_info['info']['result']['list']:
                              if isinstance(account.get('coin'), list):
                                  for coin_data in account['coin']:
                                      if coin_data.get('coin') == currency:
                                          free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                          if free is not None:
                                              available_balance_str = str(free)
                                              lg.debug(f"Found balance via V5 nested structure (default fetch): {available_balance_str}")
                                              found_structure = True; break
                                  if found_structure: break
                              if found_structure: break

                 except Exception as e:
                      lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                      # Allow retry loop to handle this

            # --- Process the extracted balance string ---
            if found_structure and available_balance_str is not None:
                try:
                    final_balance = Decimal(available_balance_str)
                    if final_balance >= 0:
                         lg.info(f"Available {currency} balance: {final_balance:.4f}")
                         return final_balance # Success
                    else:
                         lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Check account.")
                         # Treat negative balance as an issue, maybe retry? For now, raise error.
                         raise ccxt.ExchangeError(f"Negative balance detected for {currency}")
                except Exception as e:
                    lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
                    # Treat conversion failure as an issue, raise error to trigger retry
                    raise ccxt.ExchangeError(f"Balance conversion failed for {currency}")
            else:
                # If still no balance found after all attempts
                lg.error(f"{NEON_RED}Could not determine available balance for {currency} after checking known structures.{RESET}")
                lg.debug(f"Last balance_info structure checked: {balance_info}")
                # Raise error to trigger retry
                raise ccxt.ExchangeError(f"Balance not found for {currency} in response")

        # --- Retry Logic for Handled Exceptions ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance for {currency}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time - RETRY_DELAY_SECONDS) # Account for standard delay below
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API keys/permissions.{RESET}")
             return None # Auth errors are fatal, don't retry
        except ccxt.ExchangeError as e:
            # Includes errors raised internally above (negative balance, conversion fail, not found)
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance for {currency}: {e}. Retrying...{RESET}")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error during balance fetch for {currency}: {e}{RESET}", exc_info=True)
            # Allow retry for unexpected errors, but log stack trace

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            # Increase delay slightly on subsequent retries
            time.sleep(RETRY_DELAY_SECONDS * (attempts + 1))
        else:
            lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            return None # Return None after all retries fail

    return None # Should not be reached, but satisfies static analysis

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using fetch_positions.
    Handles Bybit V5 API specifics, parses position details robustly, and includes retries.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        logger: Logger instance.

    Returns:
        A dictionary containing details of the open position if found (standardized format),
        otherwise None.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for symbol: {symbol} (Attempt {attempts+1})")
            positions: List[Dict] = []
            # Bybit V5 prefers fetching specific symbol if possible, specify category
            params = {'category': 'linear'} # Default to linear, adjust if handling inverse needed elsewhere
            # TODO: Determine category dynamically based on market_info if available?
            # if market_info.get('inverse'): params['category'] = 'inverse'

            try:
                 # Attempt fetching only the target symbol
                 positions = exchange.fetch_positions([symbol], params=params)
                 lg.debug(f"Fetched positions using fetch_positions([symbol]): {positions}")
            except ccxt.ArgumentsRequired:
                 # Fallback if exchange requires fetching all positions
                 lg.debug("Fetching all positions (exchange doesn't support single symbol fetch).")
                 all_positions = exchange.fetch_positions(params=params) # Add params here too
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
            except ccxt.ExchangeError as e:
                 # Handle specific Bybit "position not found" errors gracefully
                 # Bybit V5 code for position not found: 110025 ("Position not found")
                 no_pos_codes_v5 = [110025]
                 err_str = str(e).lower()
                 if (hasattr(e, 'code') and e.code in no_pos_codes_v5) or \
                    "position not found" in err_str or "no position found" in err_str:
                      lg.info(f"No position found for {symbol} (Exchange confirmed: {e}).")
                      return None # Confirmed no position
                 else:
                      # Re-raise other exchange errors to trigger retry logic
                      raise e
            # Let other exceptions (NetworkError, AuthError, etc.) propagate to outer handler

            active_position = None
            # Define a small threshold to consider a position size non-zero
            # Use market info for amount precision if possible, else a small default
            size_threshold = Decimal('1e-9') # Very small default threshold
            try:
                market = exchange.market(symbol)
                amount_precision_step = market.get('precision', {}).get('amount')
                if amount_precision_step:
                    # Threshold slightly smaller than the smallest step size
                    size_threshold = Decimal(str(amount_precision_step)) * Decimal('0.1')
            except Exception as market_err:
                lg.debug(f"Could not get market precision for position size threshold ({market_err}), using default.")

            lg.debug(f"Using position size threshold: {size_threshold}")

            # Iterate through potentially multiple position entries (e.g., hedge mode)
            for pos in positions:
                # --- Determine Position Size (robustly checking multiple fields) ---
                pos_size_str = None
                if pos.get('contracts') is not None: # Standard CCXT field
                    pos_size_str = str(pos['contracts'])
                elif pos.get('info', {}).get('size') is not None: # Bybit V5 often uses 'size' in 'info'
                    pos_size_str = str(pos['info']['size'])
                elif pos.get('contractSize') is not None: # Sometimes size is per contract
                     pos_size_str = str(pos.get('contractSize')) # Less common for total size

                if pos_size_str is None:
                    lg.debug(f"Skipping position entry, could not determine size: {pos.get('info', {})}")
                    continue

                try:
                    position_size = Decimal(pos_size_str)
                    # Check if absolute size exceeds the threshold
                    if abs(position_size) > size_threshold:
                        active_position = pos # Found a potentially active position
                        lg.debug(f"Found potential active position entry for {symbol} with size {position_size}. Details: {pos.get('info', {})}")
                        break # Assume first non-zero position is the one we manage (for non-hedge mode)
                except Exception as parse_err:
                     lg.warning(f"Could not parse position size '{pos_size_str}': {parse_err}. Skipping entry.")
                     continue

            if active_position:
                # --- Standardize Key Position Details ---
                # Make a copy to avoid modifying the original list item
                std_pos = active_position.copy()
                info_dict = std_pos.get('info', {}) # Convenience accessor for 'info' field

                # Size (ensure Decimal, prefer 'contracts')
                size_str = str(std_pos.get('contracts', info_dict.get('size', '0')))
                std_pos['size_decimal'] = Decimal(size_str) # Store standardized Decimal size

                # Side ('long' or 'short') - Derive robustly
                side = std_pos.get('side') # Standard CCXT field
                if side not in ['long', 'short']:
                    pos_side_v5 = info_dict.get('side', '').lower() # Bybit V5 'Buy'/'Sell'
                    if pos_side_v5 == 'buy': side = 'long'
                    elif pos_side_v5 == 'sell': side = 'short'
                    # Fallback: derive from size if side is missing/ambiguous
                    elif std_pos['size_decimal'] > size_threshold: side = 'long'
                    elif std_pos['size_decimal'] < -size_threshold: side = 'short'
                    else:
                        lg.warning(f"Position size {std_pos['size_decimal']} near zero or side ambiguous. Cannot determine side.")
                        return None # Cannot reliably determine side
                std_pos['side'] = side # Store standardized side

                # Entry Price
                entry_price_str = std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice') # Check multiple fields
                std_pos['entryPrice'] = entry_price_str # Ensure standardized field exists

                # Leverage
                leverage_str = std_pos.get('leverage') or info_dict.get('leverage')
                std_pos['leverage'] = leverage_str

                # Liquidation Price
                liq_price_str = std_pos.get('liquidationPrice') or info_dict.get('liqPrice')
                std_pos['liquidationPrice'] = liq_price_str

                # Unrealized PnL
                pnl_str = std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl')
                std_pos['unrealizedPnl'] = pnl_str

                # --- Extract Protection Info (SL/TP/TSL from Bybit V5 info dict) ---
                # Note: CCXT might not parse these consistently across exchanges/versions
                sl_price_str = std_pos.get('stopLossPrice') or info_dict.get('stopLoss')
                tp_price_str = std_pos.get('takeProfitPrice') or info_dict.get('takeProfit')
                tsl_distance_str = info_dict.get('trailingStop') # Bybit V5: Distance/Value
                tsl_activation_str = info_dict.get('activePrice') # Bybit V5: Activation price for TSL

                # Store potentially missing protection info back into the main dict if found in info
                if sl_price_str and not std_pos.get('stopLossPrice'): std_pos['stopLossPrice'] = sl_price_str
                if tp_price_str and not std_pos.get('takeProfitPrice'): std_pos['takeProfitPrice'] = tp_price_str
                # Add TSL info using consistent keys if found
                std_pos['trailingStopLoss'] = tsl_distance_str
                std_pos['tslActivationPrice'] = tsl_activation_str

                # --- Log Formatted Position Info ---
                # Helper to format values for logging, using market precision if available
                def format_log_val(val, precision_type='price', default_prec=4):
                    if val is None or str(val).strip() == '' or str(val).lower() == 'nan': return 'N/A'
                    try:
                        d_val = Decimal(str(val))
                        # Attempt to get market precision
                        prec = default_prec
                        try:
                            market = exchange.market(symbol) # Requires market loaded
                            prec_val = market.get('precision', {}).get(precision_type)
                            if prec_val:
                                # CCXT precision is step size, convert to decimal places
                                prec = abs(Decimal(str(prec_val)).normalize().as_tuple().exponent)
                        except Exception: pass # Ignore errors getting precision

                        # Format using determined precision
                        return f"{d_val:.{prec}f}"
                    except Exception:
                        return str(val) # Fallback to string representation

                entry_price_fmt = format_log_val(std_pos.get('entryPrice'), 'price')
                contracts_fmt = format_log_val(abs(std_pos['size_decimal']), 'amount') # Show absolute size
                liq_price_fmt = format_log_val(std_pos.get('liquidationPrice'), 'price')
                leverage_fmt = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str and Decimal(str(leverage_str)) > 0 else 'N/A'
                pnl_fmt = format_log_val(std_pos.get('unrealizedPnl'), 'price', 4) # PnL often uses quote precision
                sl_price_fmt = format_log_val(std_pos.get('stopLossPrice'), 'price')
                tp_price_fmt = format_log_val(std_pos.get('takeProfitPrice'), 'price')
                tsl_dist_fmt = format_log_val(std_pos.get('trailingStopLoss'), 'price') # TSL distance often uses price precision
                tsl_act_fmt = format_log_val(std_pos.get('tslActivationPrice'), 'price')

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={contracts_fmt}, Entry={entry_price_fmt}, Liq={liq_price_fmt}, "
                            f"Lev={leverage_fmt}, PnL={pnl_fmt}, SL={sl_price_fmt}, TP={tp_price_fmt}, "
                            f"TSL(Dist/Act): {tsl_dist_fmt}/{tsl_act_fmt}")
                logger.debug(f"Full standardized position details for {symbol}: {std_pos}")
                return std_pos # Success, return the standardized dictionary
            else:
                logger.info(f"No active open position found for {symbol} (checked {len(positions)} entries).")
                return None # No non-zero position found

        # --- Retry Logic for Handled Exceptions ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching position for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching position: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time - RETRY_DELAY_SECONDS) # Account for standard delay below
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching position: {e}. Stopping checks.{RESET}")
             return None # Fatal
        except ccxt.ExchangeError as e:
            # Specific errors like 'position not found' are handled inside the try block.
            # Retry other exchange errors.
            lg.warning(f"{NEON_YELLOW}Exchange error fetching position for {symbol}: {e}. Retrying...{RESET}")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
            # Don't retry unexpected errors immediately, let outer loop handle maybe

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay on retries
        else:
            lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            return None

    return None # Should not be reached

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """
    Sets leverage for a derivatives symbol using CCXT, handling Bybit V5 specifics.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        leverage: The desired integer leverage value.
        market_info: Market dictionary containing contract details.
        logger: Logger instance.

    Returns:
        True if leverage was set successfully or already correct, False otherwise.
    """
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract).")
        return True # No action needed for non-contracts, considered success

    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage ({leverage}). Must be > 0.")
        return False

    if not exchange.has.get('setLeverage'):
         lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage via CCXT.{RESET}")
         # Check if setMarginMode is available as an alternative for some exchanges/modes
         return False

    try:
        # Optional: Check current leverage first to avoid unnecessary API calls
        # current_pos = get_open_position(exchange, symbol, lg) # Needs error handling
        # current_lev = current_pos.get('leverage') if current_pos else None
        # if current_lev and int(float(current_lev)) == leverage:
        #     lg.info(f"Leverage for {symbol} already set to {leverage}x.")
        #     return True

        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        # Bybit V5 requires setting buy and sell leverage separately for the symbol
        if 'bybit' in exchange.id.lower():
             # Leverage value must be passed as a string in params for Bybit V5 API
             params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
             lg.debug(f"Using Bybit V5 params for set_leverage: {params}")
             # Note: This assumes Isolated margin. For Cross margin, leverage might be set per-currency.
             # Consider adding set_margin_mode call if needed: exchange.set_margin_mode('isolated', symbol, {'leverage': leverage})

        # CCXT's set_leverage signature: setLeverage(leverage, symbol=None, params={})
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}") # Contains exchange-specific response

        # Verification (Optional - uncomment if needed, adds latency)
        # time.sleep(1) # Short delay before verification
        # updated_pos = get_open_position(exchange, symbol, lg)
        # if updated_pos and updated_pos.get('leverage') and int(float(updated_pos['leverage'])) == leverage:
        #      lg.info(f"{NEON_GREEN}Leverage for {symbol} confirmed set to {leverage}x.{RESET}")
        #      return True
        # elif updated_pos:
        #      lg.warning(f"{NEON_YELLOW}Leverage set request sent, but confirmation shows leverage is {updated_pos.get('leverage')}. Check manually.{RESET}")
        #      return False # Treat confirmation failure as potential issue
        # else:
        #      lg.warning(f"{NEON_YELLOW}Leverage set request sent, but could not confirm position status afterwards.{RESET}")
        #      return False # Cannot confirm

        # If not verifying, assume success if no exception was raised
        lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Bybit V5 Specific Error Codes for Leverage:
        # 110045: Leverage not modified (already set to this value)
        # 110028 / 110009 / 110055: Position/order exists, margin mode conflict (Isolated vs Cross)
        # 110044: Exceed risk limit (leverage too high for position size tier)
        # 110013: Parameter error (leverage value invalid/out of range)
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")

        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True # Already set is considered success
        elif bybit_code in [110028, 110009, 110055] or "margin mode" in err_str or "position exists" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Cannot change leverage. Check Margin Mode (Isolated/Cross), open orders, or existing position for {symbol}.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed risk limit tier for current/potential position size. Check Bybit Risk Limits documentation.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for {symbol}. Check allowed range on Bybit.{RESET}")
        # Add more specific error handling as needed
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False # Return False if any error occurred and wasn't handled as success

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # e.g., 0.01 for 1%
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Needed for formatting amount
    logger: logging.Logger
) -> Optional[Decimal]:
    """
    Calculates the position size based on account balance, risk percentage,
    stop-loss distance, contract specifications, and market limits/precision.

    Args:
        balance: Available account balance (in quote currency) as Decimal.
        risk_per_trade: Risk percentage per trade (e.g., 0.01 for 1%).
        initial_stop_loss_price: Calculated initial stop-loss price as Decimal.
        entry_price: Estimated or actual entry price as Decimal.
        market_info: Market dictionary from get_market_info.
        exchange: Initialized CCXT exchange object (for formatting).
        logger: Logger instance.

    Returns:
        The calculated and adjusted position size as a Decimal, or None if calculation fails.
    """
    lg = logger
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY) # e.g., USDT
    base_currency = market_info.get('base', 'BASE') # e.g., BTC
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('inverse', False)
    # Determine the unit for size (Contracts for derivatives, Base currency for Spot)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
         lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
         return None
    if initial_stop_loss_price is None or initial_stop_loss_price <= 0 or \
       entry_price is None or entry_price <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid entry ({entry_price}) or SL ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
         lg.error(f"Position sizing failed ({symbol}): SL price cannot equal entry price.")
         return None
    if 'limits' not in market_info or 'precision' not in market_info:
         lg.error(f"Position sizing failed ({symbol}): Market info missing limits/precision.")
         # Attempting to reload market info is handled at a higher level if needed
         return None

    try:
        # --- Risk Amount ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")

        # --- Stop-Loss Distance ---
        sl_distance_price = abs(entry_price - initial_stop_loss_price)
        if sl_distance_price <= 0:
             lg.error(f"Position sizing failed ({symbol}): SL distance is zero/negative ({sl_distance_price}).")
             return None
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_price}")

        # --- Contract Size ---
        # Value of 1 unit of 'amount' (1 contract or 1 base unit for spot)
        contract_size_str = market_info.get('contractSize', '1') # Defaults to 1 (e.g., for spot)
        try:
             contract_size = Decimal(str(contract_size_str))
             if contract_size <= 0: raise ValueError("Contract size must be positive")
        except Exception as e:
             lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}, using 1. Error: {e}")
             contract_size = Decimal('1')
        lg.info(f"  ContractSize={contract_size}, Type={'Linear/Spot' if not is_inverse else 'Inverse'}")


        # --- Calculate Initial Size based on Risk and SL Distance ---
        calculated_size = Decimal('0')
        if not is_inverse: # Linear Contract or Spot
             # Risk = Size * SL_Distance_Price * ContractSize_Quote
             # Size = Risk / (SL_Distance_Price * ContractSize_Quote)
             risk_per_unit_amount = sl_distance_price * contract_size
             if risk_per_unit_amount > 0:
                 calculated_size = risk_amount_quote / risk_per_unit_amount
             else:
                 lg.error(f"Sizing failed ({symbol}): Calculated risk per unit amount is zero/negative."); return None
        else: # Inverse Contract
             # Risk (Quote) = Size * ContractSize_Base * |(1/SL_Price) - (1/Entry_Price)|
             # Size = Risk / (ContractSize_Base * |(1/SL_Price) - (1/Entry_Price)|)
             lg.warning(f"Inverse contract detected ({symbol}). Ensure balance and risk are in Quote ({quote_currency}).")
             if entry_price > 0 and initial_stop_loss_price > 0:
                  # Calculate risk per contract in quote currency
                  risk_per_contract_quote = contract_size * abs(Decimal('1') / initial_stop_loss_price - Decimal('1') / entry_price)
                  if risk_per_contract_quote > 0:
                      calculated_size = risk_amount_quote / risk_per_contract_quote
                  else:
                      lg.error(f"Sizing failed ({symbol}): Inverse contract risk per contract is zero/negative."); return None
             else:
                 lg.error(f"Sizing failed ({symbol}): Invalid entry/SL price for inverse contract calculation."); return None

        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {}) # Limits on position size (in contracts or base currency)
        cost_limits = limits.get('cost', {})     # Limits on position value (in quote currency)
        precision = market_info.get('precision', {})
        amount_precision_step = precision.get('amount') # Step size for amount (e.g., 0.001)
        price_precision_step = precision.get('price') # Needed for cost calculation accuracy

        if amount_precision_step is None or price_precision_step is None:
            lg.error(f"Sizing failed ({symbol}): Missing amount or price precision in market info.")
            return None

        # Amount Limits (Min/Max Size)
        min_amount_str = amount_limits.get('min')
        max_amount_str = amount_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')

        # Cost Limits (Min/Max Value in Quote Currency)
        min_cost_str = cost_limits.get('min')
        max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')

        adjusted_size = calculated_size
        # Apply Min/Max Amount Limits first
        if min_amount > 0 and adjusted_size < min_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} < min amount {min_amount}. Adjusting to min amount.{RESET}")
             adjusted_size = min_amount
        if max_amount > 0 and adjusted_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} > max amount {max_amount}. Adjusting to max amount.{RESET}")
             adjusted_size = max_amount

        # Calculate Estimated Cost (in Quote Currency) for Cost Limit Check
        estimated_cost = Decimal('0')
        if not is_inverse: # Linear / Spot
            # Cost = Size * EntryPrice * ContractSize
            estimated_cost = adjusted_size * entry_price * contract_size
        else: # Inverse
            # Cost (in Quote) = Size * ContractSize / EntryPrice (value in base * quote/base rate)
            if entry_price > 0:
                 estimated_cost = (adjusted_size * contract_size) / entry_price
            else: lg.warning(f"Cannot estimate cost for inverse contract ({symbol}) due to zero entry price."); # Skip cost check?

        lg.debug(f"  Size after Amount Limits: {adjusted_size:.8f} {size_unit}")
        lg.debug(f"  Estimated Cost ({quote_currency}): {estimated_cost:.4f}")

        # Apply Min/Max Cost Limits (adjusting size if needed)
        cost_adjusted = False
        if min_cost > 0 and estimated_cost < min_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} < min cost {min_cost}. Attempting to increase size.{RESET}")
             required_size_for_min_cost = None
             try:
                 if not is_inverse: # Linear / Spot
                     if entry_price > 0 and contract_size > 0:
                         required_size_for_min_cost = min_cost / (entry_price * contract_size)
                 else: # Inverse
                     if contract_size > 0: # Requires Price > 0 implicitly
                         required_size_for_min_cost = (min_cost * entry_price) / contract_size
             except Exception as cost_calc_err: lg.error(f"Error calculating required size for min cost: {cost_calc_err}")

             if required_size_for_min_cost is None or required_size_for_min_cost <= 0:
                 lg.error(f"{NEON_RED}Cannot meet min cost {min_cost}. Calculation failed or resulted in zero/negative size. Aborted.{RESET}"); return None
             lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")

             if max_amount > 0 and required_size_for_min_cost > max_amount:
                 lg.error(f"{NEON_RED}Cannot meet min cost {min_cost} without exceeding max amount {max_amount}. Aborted.{RESET}"); return None

             # Adjust size up to meet min cost, ensuring it doesn't drop below min amount itself
             adjusted_size = max(min_amount, required_size_for_min_cost)
             cost_adjusted = True

        elif max_cost > 0 and estimated_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} > max cost {max_cost}. Reducing size.{RESET}")
             adjusted_size_for_max_cost = None
             try:
                 if not is_inverse: # Linear / Spot
                     if entry_price > 0 and contract_size > 0:
                         adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                 else: # Inverse
                     if contract_size > 0: # Requires Price > 0 implicitly
                          adjusted_size_for_max_cost = (max_cost * entry_price) / contract_size
             except Exception as cost_calc_err: lg.error(f"Error calculating max size for max cost: {cost_calc_err}")

             if adjusted_size_for_max_cost is None or adjusted_size_for_max_cost <= 0:
                  lg.error(f"{NEON_RED}Cannot reduce size to meet max cost {max_cost}. Calculation failed or resulted in zero/negative size. Aborted.{RESET}"); return None
             lg.info(f"  Max size allowed by max cost: {adjusted_size_for_max_cost:.8f}")

             # Reduce size to meet max cost, ensuring it doesn't fall below min amount
             adjusted_size = max(min_amount, min(adjusted_size, adjusted_size_for_max_cost))
             cost_adjusted = True

        if cost_adjusted:
             lg.info(f"  Size after Cost Limits: {adjusted_size:.8f} {size_unit}")

        # --- Apply Amount Precision (Step Size) ---
        final_size = adjusted_size
        try:
            # Use ccxt's amount_to_precision which correctly handles step sizes.
            # Use TRUNCATE (equivalent to ROUND_DOWN) to ensure we don't exceed risk/cost limits due to rounding up.
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size)) # Defaults to truncate usually
            # Or explicitly: formatted_size_str = exchange.decimal_to_precision(adjusted_size, exchange.TRUNCATE, precision['amount'], exchange.TICK_SIZE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except KeyError:
             lg.warning(f"{NEON_YELLOW}Market '{symbol}' might not have amount precision info in CCXT. Using manual rounding (less reliable).{RESET}")
             # Manual rounding fallback (less reliable than ccxt method for complex step sizes)
             try:
                  step_size = Decimal(str(amount_precision_step))
                  if step_size > 0:
                       # Floor division to round down to the nearest step size multiple
                       final_size = (adjusted_size // step_size) * step_size
                       lg.info(f"Applied manual amount step size ({step_size}): {adjusted_size:.8f} -> {final_size} {size_unit}")
                  else: raise ValueError("Step size must be positive")
             except Exception as manual_err:
                  lg.warning(f"{NEON_YELLOW}Invalid amount precision value '{amount_precision_step}' ({manual_err}). Using size adjusted for limits only.{RESET}")
                  final_size = adjusted_size # Use limit-adjusted size without precision formatting
        except Exception as fmt_err:
            lg.error(f"{NEON_RED}Error applying amount precision for {symbol}: {fmt_err}. Using unformatted size.{RESET}", exc_info=True)
            final_size = adjusted_size # Fallback to limit-adjusted size

        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{NEON_RED}Position size became zero/negative ({final_size}) after adjustments. Aborted.{RESET}")
            return None
        # Ensure final size still meets minimum amount limit after truncation
        if min_amount > 0 and final_size < min_amount:
            # This can happen if min_amount itself is not a multiple of step size
            lg.error(f"{NEON_RED}Final size {final_size} is below minimum amount {min_amount} after precision formatting. Aborted.{RESET}")
            # Alternative: could try rounding UP to min_amount if very close, but safer to abort.
            # Alternative 2: If cost adjustment forced size up, maybe allow it? Complex. Abort is safest.
            return None

        # Recalculate final cost and check against min_cost again (important after rounding down)
        final_cost = Decimal('0')
        try:
            if not is_inverse: final_cost = final_size * entry_price * contract_size
            else:
                 if entry_price > 0: final_cost = (final_size * contract_size) / entry_price
        except Exception: lg.warning("Could not recalculate final cost.")

        if min_cost > 0 and final_cost < min_cost:
             lg.debug(f"Final size {final_size} results in cost {final_cost:.4f} which is below min cost {min_cost:.4f}.")
             # Attempt to bump size up by one step if it meets min cost and doesn't exceed max amount/cost
             try:
                 step_size = Decimal(str(amount_precision_step))
                 next_step_size = final_size + step_size

                 # Calculate cost of next step size
                 next_step_cost = Decimal('0')
                 if not is_inverse: next_step_cost = next_step_size * entry_price * contract_size
                 else:
                      if entry_price > 0: next_step_cost = (next_step_size * contract_size) / entry_price

                 # Check if next step is valid
                 valid_next_step = True
                 if next_step_cost < min_cost: valid_next_step = False
                 if max_amount > 0 and next_step_size > max_amount: valid_next_step = False
                 if max_cost > 0 and next_step_cost > max_cost: valid_next_step = False

                 if valid_next_step:
                      lg.warning(f"{NEON_YELLOW}Final size cost {final_cost:.4f} < min cost {min_cost}. Bumping size to next step {next_step_size} ({size_unit}) to meet minimums.{RESET}")
                      final_size = next_step_size
                 else:
                      lg.error(f"{NEON_RED}Final size {final_size} cost is below minimum, and next step size {next_step_size} is invalid (violates min cost or max limits). Aborted.{RESET}")
                      return None
             except Exception as bump_err:
                  lg.error(f"{NEON_RED}Final size cost is below minimum. Error trying to bump size: {bump_err}. Aborted.{RESET}")
                  return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None

def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL" (direction of the trade)
    position_size: Decimal, # Positive size to trade
    market_info: Dict,
    logger: logging.Logger,
    reduce_only: bool = False,
    params: Optional[Dict] = None # Allow passing extra params if needed
) -> Optional[Dict]:
    """
    Places a market order using CCXT.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        trade_signal: "BUY" or "SELL" indicating the order direction.
        position_size: The absolute size of the order (must be positive).
        market_info: Market dictionary for context.
        logger: Logger instance.
        reduce_only: If True, set the reduceOnly flag (for closing positions).
        params: Optional dictionary of extra parameters for create_order.

    Returns:
        The order dictionary returned by CCXT on success, None on failure.
    """
    lg = logger
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')
    action_desc = "Close" if reduce_only else "Open/Increase"

    try:
        # Convert size to float for CCXT create_order, ensure it's positive
        amount_float = float(position_size)
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Invalid position size ({position_size}). Size must be positive.")
            return None
    except Exception as e:
        lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Failed to convert size {position_size} to float: {e}")
        return None

    # --- Prepare Parameters ---
    order_params = {
        # Bybit V5: positionIdx=0 for one-way mode (default)
        # Use 1 for Buy Hedge, 2 for Sell Hedge if using Hedge Mode (requires separate logic)
        'positionIdx': 0,
        'reduceOnly': reduce_only,
    }
    if reduce_only:
        # Use IOC for reduceOnly market orders to prevent accidental opening if position closes unexpectedly
        # or if the order lingers partially filled.
        order_params['timeInForce'] = 'IOC' # Immediate Or Cancel
        lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type} order for {symbol}:")
    else:
        # Default timeInForce for market orders is usually GTC or FOK depending on exchange,
        # but market orders typically execute immediately anyway.
        lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type} order for {symbol}:")

    lg.info(f"  Side: {side.upper()}, Size: {amount_float:.8f} {size_unit}")

    # Merge any additional custom params passed in
    if params:
        order_params.update(params)
    lg.debug(f"  Params: {order_params}")

    try:
        # Use create_order: create_order(symbol, type, side, amount, price=None, params={})
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None, # Market order doesn't need a specific price
            params=order_params
        )
        order_id = order.get('id', 'N/A')
        # Status might be 'open' briefly then 'closed' for market orders
        order_status = order.get('status', 'N/A')
        avg_fill_price = order.get('average') # Price at which the order was filled
        filled_amount = order.get('filled') # Amount actually filled

        lg.info(f"{NEON_GREEN}{action_desc} Trade Placed Successfully!{RESET}")
        lg.info(f"  Order ID: {order_id}, Initial Status: {order_status}")
        if avg_fill_price: lg.info(f"  Avg Fill Price: ~{avg_fill_price}")
        if filled_amount: lg.info(f"  Filled Amount: {filled_amount}")
        lg.debug(f"Raw order response ({symbol} {side} {action_desc}): {order}")
        return order

    # --- Specific Error Handling ---
    except ccxt.InsufficientFunds as e:
         lg.error(f"{NEON_RED}Insufficient funds to place {action_desc} {side} order ({symbol}): {e}{RESET}")
         # Hint: Check available balance in the correct account (Unified/Contract).
    except ccxt.InvalidOrder as e:
        # Includes issues like size precision, limit violations, parameter errors
        lg.error(f"{NEON_RED}Invalid order parameters placing {action_desc} {side} order ({symbol}): {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        # Bybit V5 Specific Codes:
        # 110007: Order quantity invalid / below minimum / exceeds maximum / precision error
        # 110013: Parameter error (e.g., invalid positionIdx, timeInForce)
        # 110014: Reduce-only order failed (no position, size > position, side mismatch)
        # 110040: Order cost invalid / below minimum / exceeds maximum
        if reduce_only and (bybit_code == 110014 or "reduce-only" in str(e).lower()):
             lg.error(f"{NEON_YELLOW} >> Hint (Reduce-Only Fail): Position might be closed, size incorrect, or wrong side specified? Check active position.{RESET}")
        elif bybit_code == 110007 or "order quantity" in str(e).lower():
             lg.error(f"{NEON_YELLOW} >> Hint (Quantity Error): Check order size ({amount_float}) against market precision (step size) and amount limits (min/max).{RESET}")
        elif bybit_code == 110040 or "order cost" in str(e).lower():
             lg.error(f"{NEON_YELLOW} >> Hint (Cost Error): Check estimated order value ({amount_float} * price) against market cost limits (min/max).{RESET}")
        elif bybit_code == 110013:
             lg.error(f"{NEON_YELLOW} >> Hint (Parameter Error): Review order parameters: {order_params}. Ensure they are valid for Bybit V5.{RESET}")
    except ccxt.NetworkError as e:
         lg.error(f"{NEON_RED}Network error placing {action_desc} order ({symbol}): {e}{RESET}")
         # Network errors might warrant a retry at a higher level or manual check
    except ccxt.ExchangeError as e:
        # Catch-all for other exchange-specific errors
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action_desc} order ({symbol}): {e} (Code: {bybit_code}){RESET}")
        # Bybit V5 Specific Codes:
        # 110025: Position not found (can affect reduce_only or TP/SL placement)
        # 110043: Set margin mode failed (can happen during order if mode conflicts)
        # 30086: Risk limit exceeded (position size + order size > allowed for leverage tier)
        if reduce_only and bybit_code == 110025:
             lg.warning(f"{NEON_YELLOW} >> Hint (Position Not Found): Position might have been closed already when trying to place reduce-only order.{RESET}")
        elif bybit_code == 30086 or "risk limit" in str(e).lower():
             lg.error(f"{NEON_YELLOW} >> Hint (Risk Limit): Order size + existing position size may exceed the risk limit tier for the current leverage. Reduce size/leverage or check Bybit docs.{RESET}")
        # Add more known error codes as needed
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing {action_desc} order ({symbol}): {e}{RESET}", exc_info=True)

    return None # Return None if any error occurred

def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # Value/Distance, not price
    tsl_activation_price: Optional[Decimal] = None, # Price at which TSL becomes active
) -> bool:
    """
    Internal helper to set Stop Loss, Take Profit, or Trailing Stop Loss for an
    existing position using Bybit's V5 API endpoint (/v5/position/set-trading-stop).

    Handles parameter validation, formatting according to market precision,
    and making the API call. TSL settings override fixed SL on Bybit V5.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        market_info: Market dictionary containing precision, limits, etc.
        position_info: Position dictionary (standardized format from get_open_position).
        logger: Logger instance.
        stop_loss_price: Target fixed SL price (Decimal). Ignored if TSL is set.
        take_profit_price: Target fixed TP price (Decimal).
        trailing_stop_distance: Target TSL distance (Decimal, positive value).
        tsl_activation_price: Target TSL activation price (Decimal).

    Returns:
        True if the protection was successfully set or updated, False otherwise.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract).")
        return False # Cannot set SL/TP/TSL on non-contracts
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    # --- Extract necessary info from position ---
    pos_side = position_info.get('side') # 'long' or 'short'
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
         lg.error(f"Cannot set protection for {symbol}: Invalid position side ('{pos_side}') or missing entry price.")
         return False
    try:
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0: raise ValueError("Entry price must be positive")
    except Exception as e:
        lg.error(f"Invalid entry price format ('{entry_price_str}') for protection check: {e}"); return False

    # --- Validate and Format Parameters ---
    params_to_set = {} # Dictionary to hold parameters for the API call
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} @ {entry_price}):"]
    params_valid = False # Flag to track if any valid protection is being set

    try:
        # Get price precision (tick size) for formatting prices accurately
        price_prec_str = market_info.get('precision', {}).get('price')
        if price_prec_str is None: raise ValueError("Missing price precision in market info")
        min_tick_size = Decimal(str(price_prec_str))
        if min_tick_size <= 0: raise ValueError(f"Invalid tick size: {min_tick_size}")
        price_precision_places = abs(min_tick_size.normalize().as_tuple().exponent)

        # Helper to format price according to market precision using ccxt
        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            """Formats price using exchange.price_to_precision."""
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try:
                # Use ccxt's formatter which handles tick size rounding correctly
                formatted = exchange.price_to_precision(symbol, float(price_decimal))
                # Double-check formatted value is positive
                if Decimal(formatted) <= 0:
                     lg.warning(f"Formatted price {formatted} became non-positive for input {price_decimal}.")
                     return None
                return formatted
            except Exception as e:
                 lg.error(f"Failed to format price {price_decimal} using exchange precision: {e}. Returning None.")
                 return None

        # --- Trailing Stop ---
        # Bybit V5: Setting 'trailingStop' (distance) > 0 activates TSL.
        # Requires 'activePrice' > 0. TSL overrides fixed 'stopLoss'.
        set_tsl = False
        if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and \
           isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0:

            # Validate TSL Activation Price relative to entry (must be beyond entry)
            # Add a small buffer (e.g., one tick) to avoid issues near entry
            activation_valid = False
            if pos_side == 'long' and tsl_activation_price > entry_price: activation_valid = True
            elif pos_side == 'short' and tsl_activation_price < entry_price: activation_valid = True

            if not activation_valid:
                lg.error(f"{NEON_RED}TSL Activation Price ({tsl_activation_price}) is not strictly beyond entry price ({entry_price}) for {pos_side} position. Cannot set TSL.{RESET}")
            else:
                # Format TSL distance (needs to be positive and adhere to price precision rules)
                # Ensure distance is at least one tick size
                tsl_dist_quantized = max(trailing_stop_distance, min_tick_size)
                # Use decimal_to_precision for formatting the distance value
                formatted_tsl_distance = exchange.decimal_to_precision(
                    tsl_dist_quantized, exchange.ROUND, # Round distance, maybe ROUND_UP is safer?
                    precision=price_precision_places, padding_mode=exchange.NO_PADDING
                )
                if Decimal(formatted_tsl_distance) <= 0:
                      lg.error(f"Formatted TSL distance ({formatted_tsl_distance}) is invalid.")
                      formatted_tsl_distance = None

                # Format TSL activation price
                formatted_activation_price = format_price(tsl_activation_price)

                if formatted_tsl_distance and formatted_activation_price:
                    params_to_set['trailingStop'] = formatted_tsl_distance
                    params_to_set['activePrice'] = formatted_activation_price
                    log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                    params_valid = True
                    set_tsl = True # Mark that TSL is being set
                    # Clear fixed SL if TSL is being set, as TSL takes precedence on Bybit V5
                    if 'stopLoss' in params_to_set: del params_to_set['stopLoss']
                    log_parts.append(f"  (Fixed SL will be ignored/removed as TSL is active)")
                else:
                     lg.error(f"Failed to format valid TSL parameters (Dist: {formatted_tsl_distance}, Act: {formatted_activation_price}). TSL not set.")

        # --- Fixed Stop Loss (Only if TSL is NOT being set) ---
        if not set_tsl and stop_loss_price is not None:
            # Validate SL price relative to entry (must be beyond entry)
            sl_valid = False
            if pos_side == 'long' and stop_loss_price < entry_price: sl_valid = True
            elif pos_side == 'short' and stop_loss_price > entry_price: sl_valid = True

            if not sl_valid:
                lg.error(f"{NEON_RED}Stop Loss Price ({stop_loss_price}) is not strictly beyond entry price ({entry_price}) for {pos_side} position. Cannot set SL.{RESET}")
            else:
                formatted_sl = format_price(stop_loss_price)
                if formatted_sl:
                    params_to_set['stopLoss'] = formatted_sl
                    log_parts.append(f"  Fixed SL: {formatted_sl}")
                    params_valid = True
                else:
                    lg.error(f"Failed to format valid SL price: {stop_loss_price}. Fixed SL not set.")

        # --- Fixed Take Profit ---
        if take_profit_price is not None:
             # Validate TP price relative to entry (must be beyond entry)
             tp_valid = False
             if pos_side == 'long' and take_profit_price > entry_price: tp_valid = True
             elif pos_side == 'short' and take_profit_price < entry_price: tp_valid = True

             if not tp_valid:
                lg.error(f"{NEON_RED}Take Profit Price ({take_profit_price}) is not strictly beyond entry price ({entry_price}) for {pos_side} position. Cannot set TP.{RESET}")
             else:
                formatted_tp = format_price(take_profit_price)
                if formatted_tp:
                    params_to_set['takeProfit'] = formatted_tp
                    log_parts.append(f"  Fixed TP: {formatted_tp}")
                    params_valid = True
                else:
                    lg.error(f"Failed to format valid TP price: {take_profit_price}. Fixed TP not set.")

    except ValueError as ve:
         lg.error(f"Validation error processing protection parameters for {symbol}: {ve}", exc_info=False)
         return False
    except Exception as fmt_err:
         lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    # --- Check if any valid parameters were actually set ---
    # Note: Sending empty values (e.g., "stopLoss": "0") clears that protection on Bybit V5.
    # We only send parameters if they were successfully validated and formatted above.
    # If the intention is to *clear* protection, specific logic should handle that by setting "0".
    # Example: To clear SL, pass stop_loss_price=Decimal('0') ? Needs Bybit V5 documentation confirmation.
    # For now, assume we only set valid, non-zero protection levels.

    if not params_valid or not params_to_set:
        # Check if the intention might have been to clear protection (e.g., SL=0 was passed but failed validation)
        # If SL/TP/TSL inputs were None, this is expected. If they were invalid, error was logged above.
        if stop_loss_price is None and take_profit_price is None and trailing_stop_distance is None:
             lg.info(f"No protection parameters provided for {symbol}. No API call needed.")
             return True # Considered success as no action was requested/needed
        else:
             lg.warning(f"No valid protection parameters to set for {symbol} after validation/formatting. No API call made.")
             return False # Requested action failed validation/formatting

    # --- Prepare API Call ---
    # Determine category (linear/inverse) based on market info
    category = 'linear' if market_info.get('linear', True) else 'inverse'

    # Determine positionIdx (for hedge mode compatibility)
    position_idx = 0 # Default for one-way mode
    try:
        # Try getting positionIdx from the position info if available
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
    except Exception:
        lg.warning(f"Could not parse positionIdx from position info, using default {position_idx}.")

    # Construct final parameters for Bybit V5 endpoint: /v5/position/set-trading-stop
    final_api_params = {
        'category': category,
        'symbol': market_info['id'], # Use exchange-specific symbol ID
        'tpslMode': 'Full',          # Affects entire position ('Partial' requires size param)
        # Trigger prices (consider making these configurable)
        'slTriggerBy': 'LastPrice',  # Options: MarkPrice, IndexPrice, LastPrice
        'tpTriggerBy': 'LastPrice',
        # Order types for SL/TP triggers (Market is usually preferred)
        'slOrderType': 'Market',     # Options: Market, Limit
        'tpOrderType': 'Market',
        'positionIdx': position_idx, # 0 for one-way, 1/2 for hedge
    }
    # Add the validated & formatted SL/TP/TSL parameters
    final_api_params.update(params_to_set)

    lg.info("\n".join(log_parts)) # Log the intended action
    lg.debug(f"  API Call Params: {final_api_params}")

    # --- Execute API Call ---
    try:
        # Use the specific V5 endpoint via private_post
        response = exchange.private_post('/v5/position/set-trading-stop', params=final_api_params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        # --- Process Response ---
        # Bybit V5 response structure: {retCode: 0, retMsg: "OK", result: {}, time: ..., retExtInfo: {}}
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        # ret_ext = response.get('retExtInfo', {}) # Contains additional details on failure

        if ret_code == 0:
            # Check for specific success messages indicating no actual change was needed
            # (e.g., SL already at the target price) - Treat these as success.
            no_change_msgs = [
                "tpsl order cost not modified", "take profit is not modified",
                "stop loss is not modified", "trailing stop is not modified"
            ]
            if any(msg in ret_msg.lower() for msg in no_change_msgs):
                 lg.info(f"{NEON_YELLOW}Position protection already set to target values or no change needed for {symbol}. Response: {ret_msg}{RESET}")
            else:
                 lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            return True # Success
        else:
            # Log specific Bybit error codes and messages
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}){RESET}") # Ext: {ret_ext}
            # Provide hints for common V5 errors related to set-trading-stop
            if ret_code == 110013: # Parameter Error
                lg.error(f"{NEON_YELLOW} >> Hint (110013): Check SL/TP/TSL values against entry/market price, ensure they comply with tick size, price limits, and range requirements.{RESET}")
            elif ret_code == 110036: # TSL Activation Price Error
                lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price invalid (already passed, wrong side of entry, too close/far from current market price?).{RESET}")
            elif ret_code == 110086: # SL = TP Error
                lg.error(f"{NEON_YELLOW} >> Hint (110086): Stop Loss price cannot be equal to Take Profit price.{RESET}")
            elif ret_code == 110084: # TP/SL trigger price error
                 lg.error(f"{NEON_YELLOW} >> Hint (110084/110085): SL/TP price is invalid (too close to market price, outside limits?). Check Bybit's required distance from market price.{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower():
                lg.error(f"{NEON_YELLOW} >> Hint: TSL distance invalid (too small/large, violates tick size?). Check required min/max distance on Bybit.{RESET}")
            # Add more hints based on observed errors from Bybit V5 documentation
            return False

    except ccxt.ExchangeError as e:
        # Handle potential CCXT-level errors during the raw API call (e.g., network issues wrapped by ccxt)
        lg.error(f"{NEON_RED}CCXT ExchangeError during protection API call for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during protection API call for {symbol}: {e}{RESET}", exc_info=True)

    return False # Return False if API call failed or encountered an exception

def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional fixed TP to set alongside TSL
) -> bool:
    """
    Calculates Trailing Stop Loss parameters based on configuration and current position,
    then calls the internal helper `_set_position_protection` to set the TSL
    (and optionally a fixed Take Profit) via the Bybit V5 API.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol string.
        market_info: Market dictionary from `get_market_info`.
        position_info: Position dictionary from `get_open_position`.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional fixed TP target (Decimal) to set simultaneously.

    Returns:
        True if TSL (and optional TP) were successfully requested/set, False otherwise.
    """
    lg = logger
    protection_cfg = config.get("protection", {})
    if not protection_cfg.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss disabled in config for {symbol}. Skipping TSL setup.")
        # Return True because the desired state (TSL off) is achieved.
        # Or return False if the caller expects this function only to succeed if TSL is *set*?
        # Let's return True, as no action was required by config.
        return True

    # --- Validate Inputs ---
    if not market_info or not position_info:
        lg.error(f"Cannot calculate TSL for {symbol}: Missing market or position info.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"{NEON_RED}Missing required position info (side, entryPrice) for TSL calculation ({symbol}).{RESET}")
        return False

    try:
        # Extract parameters and convert to Decimal
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(protection_cfg.get("trailing_stop_callback_rate", 0.005))) # e.g., 0.005 = 0.5%
        activation_percentage = Decimal(str(protection_cfg.get("trailing_stop_activation_percentage", 0.003))) # e.g., 0.003 = 0.3% profit trigger

        # Basic validation of parameters
        if entry_price <= 0: raise ValueError("Entry price must be positive")
        if callback_rate <= 0: raise ValueError("Callback rate must be positive")
        if activation_percentage < 0: raise ValueError("Activation percentage cannot be negative") # 0 is allowed (activate immediately)

        # Get price precision (tick size) for rounding calculations
        price_prec_str = market_info.get('precision', {}).get('price')
        if price_prec_str is None: raise ValueError("Missing price precision in market info")
        min_tick_size = Decimal(str(price_prec_str))
        if min_tick_size <= 0: raise ValueError(f"Invalid tick size: {min_tick_size}")
        price_precision_places = abs(min_tick_size.normalize().as_tuple().exponent)

    except ValueError as ve:
        lg.error(f"{NEON_RED}Invalid TSL parameter format or position info ({symbol}): {ve}. Cannot calculate TSL.{RESET}")
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Error parsing TSL parameters ({symbol}): {e}. Cannot calculate TSL.{RESET}")
        return False

    try:
        # --- Calculate Activation Price ---
        # Price at which the trailing stop becomes active.
        activation_price = None
        activation_offset = entry_price * activation_percentage

        # Activation price must be strictly beyond entry price (by at least one tick).
        if pos_side == 'long':
            raw_activation = entry_price + activation_offset
            # Round UP to the nearest tick, ensuring it's > entry_price
            # Use quantize with ROUND_UP based on tick size
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
            # Ensure it's strictly greater than entry after rounding
            if activation_price <= entry_price:
                 activation_price = entry_price + min_tick_size
        else: # short
            raw_activation = entry_price - activation_offset
            # Round DOWN to the nearest tick, ensuring it's < entry_price
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
            # Ensure it's strictly less than entry after rounding
            if activation_price >= entry_price:
                 activation_price = entry_price - min_tick_size

        # Validate calculated activation price
        if activation_price <= 0:
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) is zero/negative for {symbol}. Cannot set TSL.{RESET}")
             return False

        # --- Calculate Trailing Stop Distance ---
        # This is the fixed distance/value the SL trails behind the best price once activated.
        # Bybit V5 uses a distance value (not percentage callback directly in API).
        # Calculate distance based on the activation price * callback rate.
        # Note: Some exchanges calculate distance based on entry price. Bybit's API uses a fixed value.
        trailing_distance_raw = activation_price * callback_rate

        # Distance must be positive and generally rounded UP to the nearest tick size increment.
        # It also might have a minimum required value by the exchange (e.g., 0.1% of entry?). Check Bybit docs.
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        # Ensure minimum distance is at least one tick size
        if trailing_distance < min_tick_size:
             lg.debug(f"Calculated TSL distance {trailing_distance_raw} rounded to {trailing_distance}, which is less than min tick {min_tick_size}. Setting distance to one tick.")
             trailing_distance = min_tick_size

        if trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance zero/negative ({trailing_distance}) for {symbol}. Cannot set TSL.{RESET}")
             return False

        # --- Log Calculated Parameters ---
        lg.info(f"Calculated TSL Params for {symbol} ({pos_side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision_places}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price: {activation_price:.{price_precision_places}f}")
        lg.info(f"  => Trailing Distance: {trailing_distance:.{price_precision_places}f}")

        # Format optional TP for logging if provided
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0:
             tp_log_str = f"{take_profit_price:.{price_precision_places}f}"
             try: # Try formatting with exchange precision for log clarity
                 tp_log_str = exchange.price_to_precision(symbol, float(take_profit_price))
             except Exception: pass
             lg.info(f"  Take Profit Price: {tp_log_str} (Will be set alongside TSL)")
        else:
             take_profit_price = None # Ensure it's None if invalid input was passed


        # --- Call Helper to Set Protection via API ---
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None, # TSL overrides fixed SL, so set SL to None
            take_profit_price=take_profit_price, # Pass optional fixed TP
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy Implementation ---

class OrderBlock(TypedDict):
    """Represents a detected Order Block with its properties."""
    id: str           # Unique identifier (e.g., type_timestamp)
    type: str         # 'bull' or 'bear'
    left_idx: pd.Timestamp # Timestamp of the bar where OB formed (pivot bar)
    right_idx: pd.Timestamp # Timestamp of the last bar OB is considered valid for (or violated)
    top: Decimal      # Top price level of the OB
    bottom: Decimal   # Bottom price level of the OB
    active: bool      # Is the OB still considered valid (not violated)?
    violated: bool    # Has price closed beyond the OB boundaries?

class StrategyAnalysisResults(TypedDict):
    """Structured container for results from the strategy analysis."""
    dataframe: pd.DataFrame        # DataFrame with all indicator calculations (Decimal columns)
    last_close: Decimal            # Latest close price as Decimal
    current_trend_up: Optional[bool] # True=UP, False=DOWN, None=Undetermined
    trend_just_changed: bool       # True if trend changed on the last completed bar
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    vol_norm_int: Optional[int]    # Latest Volume Norm (0-200 integer), or None
    atr: Optional[Decimal]         # Latest ATR as Decimal, or None
    upper_band: Optional[Decimal]  # Latest Volumatic upper band as Decimal, or None
    lower_band: Optional[Decimal]  # Latest Volumatic lower band as Decimal, or None

class VolumaticOBStrategy:
    """
    Implements the Volumatic Trend and Pivot Order Block strategy.
    Calculates indicators based on Pine Script logic interpretation, manages
    Order Block state (creation, violation, pruning), and returns analysis results.
    """

    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the strategy engine with configuration parameters.

        Args:
            config: The main bot configuration dictionary.
            market_info: Market details dictionary (used for context).
            logger: Logger instance.
        """
        self.config = config
        self.market_info = market_info # Store for reference (e.g., interval)
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})

        # --- Store Configurable Strategy Parameters ---
        self.vt_length = int(strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH))
        self.vt_atr_period = int(strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD))
        self.vt_vol_ema_length = int(strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH))
        # Use Decimal for multipliers for precision
        self.vt_atr_multiplier = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))
        self.vt_step_atr_multiplier = Decimal(str(strategy_cfg.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER)))

        self.ob_source = strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE)
        self.ph_left = int(strategy_cfg.get("ph_left", DEFAULT_PH_LEFT))
        self.ph_right = int(strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT))
        self.pl_left = int(strategy_cfg.get("pl_left", DEFAULT_PL_LEFT))
        self.pl_right = int(strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT))
        self.ob_extend = bool(strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND))
        self.ob_max_boxes = int(strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES))

        # --- Internal State Variables ---
        # Store detected order blocks persistently between updates
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Calculate minimum data length required based on the longest lookback period
        self.min_data_len = max(
             self.vt_length * 2, # EMA/SWMA need buffer to stabilize
             self.vt_atr_period,
             self.vt_vol_ema_length,
             self.ph_left + self.ph_right + 1, # Pivot needs left+pivot+right bars
             self.pl_left + self.pl_right + 1
         ) + 10 # Add a general safety buffer

        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy Engine...{RESET}")
        self.logger.info(f"  VT Params: Len={self.vt_length}, ATRLen={self.vt_atr_period}, VolLen={self.vt_vol_ema_length}, ATRMult={self.vt_atr_multiplier}, StepMult={self.vt_step_atr_multiplier}")
        self.logger.info(f"  OB Params: Src={self.ob_source}, PH={self.ph_left}/{self.ph_right}, PL={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxBoxes={self.ob_max_boxes}")
        self.logger.info(f"  Minimum historical data points recommended: {self.min_data_len}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates a custom EMA based on a 4-period Symmetrically Weighted Moving Average (SWMA).
        This attempts to replicate a specific Pine Script pattern: `ema(swma(source), length)`.
        The SWMA uses weights [1, 2, 2, 1] / 6.

        Args:
            series: Input pandas Series (e.g., close prices). Must be float/numeric.
            length: The length parameter for the final EMA.

        Returns:
            A pandas Series containing the calculated EMA of SWMA.
        """
        if len(series) < 4 or length <= 0:
            # Return NaNs if data is too short or length is invalid
            return pd.Series(np.nan, index=series.index, dtype=float)

        # Pine Script's swma(src) is often interpreted as a 4-period WMA with weights [1, 2, 2, 1].
        # Weights applied to [x[3], x[2], x[1], x[0]] where x[0] is the current bar.
        # Pandas rolling window operates on [t-3, t-2, t-1, t]. We apply weights [1, 2, 2, 1] / 6.0.
        weights = np.array([1, 2, 2, 1]) / 6.0

        # Use rolling apply with the defined weights. Ensure raw=True for performance.
        # min_periods=4 ensures we only calculate when we have a full window.
        swma = series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate the EMA of the resulting SWMA series using pandas_ta
        # Ensure the input series to ema is numeric (float) and handle potential NaNs from swma
        swma_numeric = pd.to_numeric(swma, errors='coerce')
        ema_of_swma = ta.ema(swma_numeric, length=length, fillna=np.nan)

        return ema_of_swma

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes the input historical data (OHLCV DataFrame) to calculate
        Volumatic Trend indicators and identify/manage Pivot Order Blocks.

        Args:
            df_input: pandas DataFrame with Decimal columns ['open', 'high', 'low', 'close', 'volume']
                      index must be a DatetimeIndex, sorted chronologically. Should contain
                      at least `self.min_data_len` rows for reliable calculations.

        Returns:
            StrategyAnalysisResults dictionary containing the updated DataFrame,
            latest trend state, ATR, bands, volume norm, and lists of active order blocks.
            Returns a default/empty result structure if analysis fails or data is insufficient.
        """
        # --- Default empty result structure ---
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        # --- Pre-computation Checks ---
        if df_input.empty:
             self.logger.error("Strategy update received an empty DataFrame.")
             return empty_results

        # Work on a copy to avoid modifying the original DataFrame passed in
        df = df_input.copy()

        # Validate index and sorting
        if not isinstance(df.index, pd.DatetimeIndex):
             self.logger.error("DataFrame index is not a DatetimeIndex. Analysis aborted.")
             # Consider attempting conversion? Safer to abort if index is wrong type.
             # try: df.index = pd.to_datetime(df.index, utc=True)
             # except Exception: self.logger.error("Failed to convert index to DatetimeIndex."); return empty_results
             return empty_results
        if not df.index.is_monotonic_increasing:
             self.logger.warning("DataFrame index is not monotonically increasing. Sorting.")
             df.sort_index(inplace=True)

        # Validate data length
        if len(df) < self.min_data_len:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(df)} rows, need >= {self.min_data_len}) for full strategy analysis. Results may be incomplete or inaccurate.{RESET}")
            # Proceed with calculation, but expect NaNs or potentially unreliable results initially

        self.logger.debug(f"Starting strategy analysis on {len(df)} candles.")

        # --- Data Preparation for TA-Lib/Pandas TA ---
        # Convert necessary Decimal columns to float for compatibility with most TA libraries.
        # Keep the original Decimal DataFrame (`df`) for precise calculations later (OB levels, results).
        try:
            df_float = df.copy()
            float_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in float_cols:
                if col in df_float.columns:
                     # Use .astype(float) for direct conversion, handles NaN correctly
                     df_float[col] = df_float[col].astype(float)
                else:
                    self.logger.error(f"Required column '{col}' not found in DataFrame. Analysis aborted.")
                    return empty_results
        except Exception as e:
            self.logger.error(f"Error converting DataFrame columns to float for TA calculations: {e}")
            return empty_results # Abort if conversion fails

        # --- Volumatic Trend Calculations (using df_float) ---
        try:
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)
            # Calculate the two EMAs for trend direction
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length) # Custom EMA(SWMA)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan) # Standard EMA

            # Trend Detection Logic (based on Pine interpretation: UpTrend = ema1[1] < ema2)
            # Python equivalent: Is previous ema1 less than current ema2?
            df_float['trend_up'] = (df_float['ema1'].shift(1) < df_float['ema2'])
            # Forward fill the boolean trend to handle initial NaNs after the shift
            df_float['trend_up'] = df_float['trend_up'].ffill() # Now contains True/False where trend is determined

            # Detect Trend Change Points
            # Change occurs when 'trend_up' flips compared to the previous bar
            df_float['trend_changed'] = (df_float['trend_up'] != df_float['trend_up'].shift(1)) & \
                                        df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'].fillna(False, inplace=True) # First valid trend point is not a 'change'

            # --- Stateful Band Calculation (using df_float) ---
            # Bands are based on EMA1 and ATR *at the point of the last trend change*.
            # Capture ema1 and atr values only on bars where the trend changed.
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

            # Forward fill these values to propagate the reference levels until the next change.
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate the upper and lower bands using the forward-filled values
            # Ensure multipliers are float for calculation with float columns
            atr_mult_float = float(self.vt_atr_multiplier)
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_mult_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_mult_float)

            # --- Volume Calculation & Normalization (using df_float) ---
            # Calculate rolling max volume over the specified lookback period
            df_float['vol_max'] = df_float['volume'].rolling(
                window=self.vt_vol_ema_length,
                min_periods=max(1, self.vt_vol_ema_length // 10) # Require some periods, e.g., 10% of window
            ).max()

            # Normalize volume (0-100 range relative to recent max)
            # Handle potential division by zero or NaN denominator
            df_float['vol_norm'] = np.where(
                df_float['vol_max'].notna() & (df_float['vol_max'] > 1e-9), # Check vol_max is valid and > 0
                (df_float['volume'] / df_float['vol_max'] * 100.0), # Normalize to 0-100
                0.0 # Default to 0 if max is NaN or zero
            )
            df_float['vol_norm'].fillna(0.0, inplace=True) # Treat any remaining NaN volume norm as 0
            # Clip extreme values (e.g., if current volume massively exceeds historical max temporarily)
            df_float['vol_norm'] = df_float['vol_norm'].clip(0.0, 200.0) # Allow up to 200% of recent max

            # --- Calculate Step Sizes and Vol Bar Levels (Optional - for visualization/other logic) ---
            # step_atr_mult_float = float(self.vt_step_atr_multiplier)
            # df_float['lower_vol_ref'] = df_float['lower_band'] + df_float['atr_for_bands'] * step_atr_mult_float
            # df_float['upper_vol_ref'] = df_float['upper_band'] - df_float['atr_for_bands'] * step_atr_mult_float
            # df_float['step_up_size'] = np.where(
            #     df_float['vol_norm'] > 1e-9,
            #      (df_float['lower_vol_ref'] - df_float['lower_band']) / df_float['vol_norm'], # Size per 1% norm? Seems complex.
            #      0.0
            # ) * (df_float['vol_norm'] / 100.0) # Scale by actual norm %
            # df_float['step_dn_size'] = np.where(
            #     df_float['vol_norm'] > 1e-9,
            #      (df_float['upper_band'] - df_float['upper_vol_ref']) / df_float['vol_norm'],
            #      0.0
            # ) * (df_float['vol_norm'] / 100.0)
            # df_float['step_up_size'] = df_float['step_up_size'].clip(lower=0).fillna(0)
            # df_float['step_dn_size'] = df_float['step_dn_size'].clip(lower=0).fillna(0)
            # df_float['vol_bar_up_top'] = df_float['lower_band'] + df_float['step_up_size']
            # df_float['vol_bar_dn_bottom'] = df_float['upper_band'] - df_float['step_dn_size']

        except Exception as e:
            self.logger.error(f"Error during Volumatic Trend indicator calculation: {e}", exc_info=True)
            return empty_results # Abort if VT calcs fail

        # --- Copy Calculated Float Columns back to the main Decimal DataFrame ---
        # Convert numeric results back to Decimal for precision in final results/OB levels.
        cols_to_copy = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed',
                        'upper_band', 'lower_band', 'vol_norm']
                        # Add 'vol_bar_up_top', 'vol_bar_dn_bottom' if calculated and needed
        try:
            for col in cols_to_copy:
                if col in df_float.columns:
                    if df_float[col].dtype == 'bool' or pd.api.types.is_object_dtype(df_float[col]):
                         # Copy booleans/objects directly
                         df[col] = df_float[col]
                    else: # Convert numeric types back to Decimal
                         # Apply conversion carefully, handling potential NaNs/Infs from float calcs
                         df[col] = df_float[col].apply(
                             lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                         )
        except Exception as e:
            self.logger.error(f"Error converting calculated float columns back to Decimal: {e}", exc_info=True)
            return empty_results

        # --- Data Cleaning After Calculations ---
        # Drop initial rows where essential indicators might still be NaN due to lookback periods
        # (e.g., before the first trend is established or ATR is calculated).
        initial_len_before_drop = len(df)
        required_cols_for_signal = ['upper_band', 'lower_band', 'atr', 'trend_up', 'close']
        df.dropna(subset=required_cols_for_signal, inplace=True)
        rows_dropped = initial_len_before_drop - len(df)
        if rows_dropped > 0:
             self.logger.debug(f"Dropped {rows_dropped} initial rows lacking essential indicator values after calculation.")

        # Check if DataFrame became empty after dropping NaNs
        if df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame empty after calculating indicators and dropping NaNs. Insufficient data or no stable trend/ATR established yet.{RESET}")
            return empty_results

        self.logger.debug("Volumatic Trend calculations complete. Proceeding to Order Blocks.")

        # --- Pivot Order Block Calculations & Management ---
        try:
            # Determine the high/low series for pivot detection based on config
            if self.ob_source == "Wicks":
                high_series = df_float['high'] # Use float series for ta.pivot
                low_series = df_float['low']
            else: # "Bodys"
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)

            # Calculate pivot points using pandas_ta.pivot
            # ta.pivot returns 1.0 at index `i` if a pivot high/low formed `right` bars ago is confirmed at `i`.
            # Fillna(0) and convert to bool for easier checking.
            df_float['ph_signal'] = ta.pivot(high_series, left=self.ph_left, right=self.ph_right, high_low='high').fillna(0).astype(bool)
            df_float['pl_signal'] = ta.pivot(low_series, left=self.pl_left, right=self.pl_right, high_low='low').fillna(0).astype(bool)

            # --- Identify NEW Pivots Confirmed in the Latest Data ---
            # We only need to check the last few bars where a pivot confirmation could occur.
            # Look back enough bars to cover the 'right' lookback period + buffer.
            check_recent_bars = max(self.ph_right, self.pl_right) + 5
            # Use indices from the main Decimal DF (`df`) as it might have fewer rows after NaN drop
            recent_indices = df.index[-check_recent_bars:]

            new_boxes_found_count = 0
            for confirmation_idx in recent_indices:
                # Ensure the confirmation index exists in the float DataFrame where signals are stored
                if confirmation_idx not in df_float.index: continue

                # --- Check for New Bearish OB (from Pivot High Confirmation) ---
                if df_float.loc[confirmation_idx, 'ph_signal']:
                    # Pivot HIGH confirmed at `confirmation_idx`. The actual pivot candle occurred `ph_right` bars BEFORE this.
                    try:
                        pivot_bar_loc_in_float = df_float.index.get_loc(confirmation_idx) - self.ph_right
                        if pivot_bar_loc_in_float >= 0:
                            pivot_bar_idx = df_float.index[pivot_bar_loc_in_float] # Timestamp of the actual pivot bar

                            # Check if this pivot bar index exists in the main Decimal DF (`df`)
                            if pivot_bar_idx in df.index:
                                # Avoid creating duplicate OBs for the same pivot bar
                                if not any(b['left_idx'] == pivot_bar_idx and b['type'] == 'bear' for b in self.bear_boxes):
                                    # Get the candle data from the precise Decimal DataFrame
                                    ob_candle = df.loc[pivot_bar_idx]
                                    box_top, box_bottom = Decimal('NaN'), Decimal('NaN')

                                    # Define Bearish OB range based on source config
                                    # Standard: Last up-candle before down move. Often uses High and Open.
                                    if self.ob_source == "Wicks":
                                        box_top = ob_candle['high']    # Wick High
                                        box_bottom = ob_candle['open'] # Wick Base (Open)
                                    else: # "Bodys"
                                        # Assumes pivot high formed on an up-candle (close > open)
                                        box_top = ob_candle['close']   # Body Top
                                        box_bottom = ob_candle['open'] # Body Bottom
                                    # Ensure top > bottom (swap if pivot high occurred on a down candle)
                                    if pd.notna(box_top) and pd.notna(box_bottom) and box_bottom > box_top:
                                        box_top, box_bottom = box_bottom, box_top

                                    # Create the box if levels are valid
                                    if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                                        self.bear_boxes.append(OrderBlock(
                                            id=f"bear_{pivot_bar_idx.strftime('%Y%m%d%H%M%S')}",
                                            type='bear', left_idx=pivot_bar_idx, right_idx=df.index[-1], # Initial end is current bar
                                            top=box_top, bottom=box_bottom, active=True, violated=False
                                        ))
                                        self.logger.debug(f"{NEON_RED}New Bearish OB created at {pivot_bar_idx} [{box_bottom} - {box_top}]{RESET}")
                                        new_boxes_found_count += 1
                    except KeyError: continue # Index not found, skip
                    except Exception as e: self.logger.warning(f"Error processing PH signal at {confirmation_idx}: {e}")


                # --- Check for New Bullish OB (from Pivot Low Confirmation) ---
                if df_float.loc[confirmation_idx, 'pl_signal']:
                    # Pivot LOW confirmed at `confirmation_idx`. Pivot candle is `pl_right` bars BEFORE.
                    try:
                        pivot_bar_loc_in_float = df_float.index.get_loc(confirmation_idx) - self.pl_right
                        if pivot_bar_loc_in_float >= 0:
                            pivot_bar_idx = df_float.index[pivot_bar_loc_in_float]

                            if pivot_bar_idx in df.index:
                                if not any(b['left_idx'] == pivot_bar_idx and b['type'] == 'bull' for b in self.bull_boxes):
                                    ob_candle = df.loc[pivot_bar_idx]
                                    box_top, box_bottom = Decimal('NaN'), Decimal('NaN')

                                    # Define Bullish OB range based on source config
                                    # Standard: Last down-candle before up move. Often uses Open and Low.
                                    if self.ob_source == "Wicks":
                                        box_top = ob_candle['open'] # Wick Base (Open)
                                        box_bottom = ob_candle['low']  # Wick Low
                                    else: # "Bodys"
                                        # Assumes pivot low formed on a down-candle (close < open)
                                        box_top = ob_candle['open']    # Body Top
                                        box_bottom = ob_candle['close']# Body Bottom
                                    # Ensure top > bottom (swap if pivot low occurred on an up candle)
                                    if pd.notna(box_top) and pd.notna(box_bottom) and box_bottom > box_top:
                                        box_top, box_bottom = box_bottom, box_top

                                    if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                                        self.bull_boxes.append(OrderBlock(
                                            id=f"bull_{pivot_bar_idx.strftime('%Y%m%d%H%M%S')}",
                                            type='bull', left_idx=pivot_bar_idx, right_idx=df.index[-1],
                                            top=box_top, bottom=box_bottom, active=True, violated=False
                                        ))
                                        self.logger.debug(f"{NEON_GREEN}New Bullish OB created at {pivot_bar_idx} [{box_bottom} - {box_top}]{RESET}")
                                        new_boxes_found_count += 1
                    except KeyError: continue
                    except Exception as e: self.logger.warning(f"Error processing PL signal at {confirmation_idx}: {e}")


            if new_boxes_found_count > 0:
                self.logger.debug(f"Found {new_boxes_found_count} new OB(s). Total counts: {len(self.bull_boxes)} Bull, {len(self.bear_boxes)} Bear.")

            # --- Manage Existing Order Blocks (Violation Check & Extension) ---
            if not df.empty and 'close' in df.columns and pd.notna(df['close'].iloc[-1]):
                last_close = df['close'].iloc[-1] # Use Decimal close price
                last_bar_idx = df.index[-1]

                # Check Bullish OBs for violation
                for box in self.bull_boxes:
                    if box['active']:
                        # Violation criteria: Price CLOSES below the bottom of a bull box
                        if last_close < box['bottom']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_bar_idx # Mark violation time
                            self.logger.debug(f"Bull Box {box['id']} ({box['bottom']}-{box['top']}) VIOLATED by close {last_close} at {last_bar_idx}.")
                        # If not violated and extension is enabled, update the right boundary
                        elif self.ob_extend:
                            box['right_idx'] = last_bar_idx

                # Check Bearish OBs for violation
                for box in self.bear_boxes:
                    if box['active']:
                        # Violation criteria: Price CLOSES above the top of a bear box
                        if last_close > box['top']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_bar_idx
                            self.logger.debug(f"Bear Box {box['id']} ({box['bottom']}-{box['top']}) VIOLATED by close {last_close} at {last_bar_idx}.")
                        elif self.ob_extend:
                            box['right_idx'] = last_bar_idx
            else:
                self.logger.warning("Cannot check OB violations: Last close price is missing or invalid.")


            # --- Prune Older Order Blocks ---
            # Keep only the N most recent *active* boxes, plus maybe some recent inactive ones for context.
            # Sort by the pivot bar timestamp (left_idx) descending (most recent first).
            active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)
            inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda b: b['left_idx'], reverse=True)
            # Keep max active boxes + half max inactive boxes
            self.bull_boxes = active_bull[:self.ob_max_boxes] + inactive_bull[:self.ob_max_boxes // 2]

            active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)
            inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda b: b['left_idx'], reverse=True)
            self.bear_boxes = active_bear[:self.ob_max_boxes] + inactive_bear[:self.ob_max_boxes // 2]

            active_bull_count = len(active_bull) # Count before pruning for logging
            active_bear_count = len(active_bear)
            self.logger.debug(f"Pruned OBs. Kept {len(self.bull_boxes)} Bull ({active_bull_count} active), {len(self.bear_boxes)} Bear ({active_bear_count} active).")

        except Exception as e:
             self.logger.error(f"Error during Pivot Order Block processing: {e}", exc_info=True)
             # Return results calculated so far, but OBs might be unreliable
             pass # Allow returning partial results from VT section


        # --- Prepare Final Results ---
        # Extract latest values from the main Decimal DataFrame (`df`)
        last_row = df.iloc[-1] if not df.empty else None
        final_close = last_row.get('close', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_atr = last_row.get('atr', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_upper_band = last_row.get('upper_band', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_lower_band = last_row.get('lower_band', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_vol_norm = last_row.get('vol_norm', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        latest_trend_up = last_row.get('trend_up') if last_row is not None else None # Can be bool or None/NaN
        latest_trend_changed = last_row.get('trend_changed', False) if last_row is not None else False

        # Sanitize final values before returning
        final_trend_up_bool = bool(latest_trend_up) if isinstance(latest_trend_up, (bool, np.bool_)) else None
        # Convert vol_norm to int, handling NaN
        final_vol_norm_int = int(final_vol_norm) if pd.notna(final_vol_norm) and np.isfinite(final_vol_norm) else None
        # Ensure ATR, bands, close are valid Decimals or None/0
        final_atr_dec = final_atr if pd.notna(final_atr) and final_atr > 0 else None
        final_upper_band_dec = final_upper_band if pd.notna(final_upper_band) else None
        final_lower_band_dec = final_lower_band if pd.notna(final_lower_band) else None
        final_close_dec = final_close if pd.notna(final_close) else Decimal('0') # Default to 0 if close is NaN

        # Filter final box lists to only include active ones for the results
        active_bull_boxes_final = [b for b in self.bull_boxes if b['active']]
        active_bear_boxes_final = [b for b in self.bear_boxes if b['active']]

        results = StrategyAnalysisResults(
            dataframe=df, # Return the main Decimal DataFrame with indicators
            last_close=final_close_dec,
            current_trend_up=final_trend_up_bool,
            trend_just_changed=bool(latest_trend_changed),
            active_bull_boxes=active_bull_boxes_final,
            active_bear_boxes=active_bear_boxes_final,
            vol_norm_int=final_vol_norm_int,
            atr=final_atr_dec,
            upper_band=final_upper_band_dec,
            lower_band=final_lower_band_dec
        )

        # Log key summary results
        trend_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] else \
                    f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{results['atr']:.4f}" if results['atr'] else "N/A"
        last_idx_time = df.index[-1].strftime('%Y-%m-%d %H:%M:%S') if not df.empty else "N/A"
        self.logger.debug(f"Strategy Results ({last_idx_time}): "
                          f"Close={results['last_close']:.4f}, Trend={trend_str}, TrendChg={results['trend_just_changed']}, "
                          f"ATR={atr_str}, VolNorm={results['vol_norm_int']}, "
                          f"Active OBs (Bull/Bear): {len(results['active_bull_boxes'])}/{len(results['active_bear_boxes'])}")

        return results

# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """
    Generates trading signals (BUY, SELL, HOLD, EXIT_LONG, EXIT_SHORT)
    based on the analysis results from VolumaticOBStrategy and the current
    position state. Also calculates initial TP/SL levels for potential trades.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the SignalGenerator with configuration parameters.

        Args:
            config: The main bot configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})
        try:
            # Proximity factors for OB interaction (use Decimal)
            # Factor > 1 means price can penetrate the OB edge slightly.
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg.get("ob_entry_proximity_factor", 1.005)))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg.get("ob_exit_proximity_factor", 1.001)))
            # Ensure factors are at least 1 (no negative penetration)
            if self.ob_entry_proximity_factor < 1: self.ob_entry_proximity_factor = Decimal("1.0")
            if self.ob_exit_proximity_factor < 1: self.ob_exit_proximity_factor = Decimal("1.0")

            # ATR Multipliers for initial TP/SL calculation (used for sizing)
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg.get("initial_take_profit_atr_multiple", 0.7)))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg.get("initial_stop_loss_atr_multiple", 1.8)))
            if self.initial_sl_atr_multiple <= 0:
                 raise ValueError("initial_stop_loss_atr_multiple must be positive for valid sizing.")

        except ValueError as ve:
             self.logger.error(f"Error initializing SignalGenerator with config values: {ve}. Using defaults.")
             # Set safe defaults if config parsing fails
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8") # Ensure positive default SL multiple
        except Exception as e:
             self.logger.error(f"Unexpected error initializing SignalGenerator: {e}. Using defaults.", exc_info=True)
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")

        self.logger.info("Signal Generator Initialized.")
        self.logger.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor}, OB Exit Proximity Factor: {self.ob_exit_proximity_factor}")
        self.logger.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple}, Initial SL ATR Multiple: {self.initial_sl_atr_multiple}")


    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """
        Determines the appropriate trading signal based on strategy analysis
        (trend, OBs, price action) and the current open position status.

        Args:
            analysis_results: The results dictionary from VolumaticOBStrategy.update().
            open_position: A dictionary representing the current open position, or None.

        Returns:
            A string signal: "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT".
        """
        # --- Input Validation ---
        # Check for essential analysis results needed for signal generation
        if not isinstance(analysis_results, dict) or analysis_results.get('dataframe') is None or \
           analysis_results['dataframe'].empty or \
           analysis_results.get('current_trend_up') is None or \
           not isinstance(analysis_results.get('last_close'), Decimal) or analysis_results['last_close'] <= 0 or \
           not isinstance(analysis_results.get('atr'), Decimal) or analysis_results['atr'] <= 0:
            self.logger.warning(f"{NEON_YELLOW}Incomplete or invalid strategy analysis results received. Cannot generate signal. Holding.{RESET}")
            self.logger.debug(f"  Problematic Analysis Results: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
            return "HOLD"

        # Extract key values from results
        latest_close = analysis_results['last_close']
        is_trend_up = analysis_results['current_trend_up'] # True or False
        trend_just_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        # ATR is guaranteed valid Decimal > 0 here due to initial check

        current_pos_side = open_position.get('side') if open_position else None # 'long' or 'short'
        signal = "HOLD" # Default signal

        # --- Signal Logic ---

        # === 1. Check for EXIT Signal (only if a position is currently open) ===
        if current_pos_side == 'long':
            # Exit Condition 1: Trend flips DOWN
            if not is_trend_up: # Trend is now DOWN
                 # Exit immediately on trend change? Or only if trend *just* changed?
                 # Current logic: Exit only if trend *just* changed (less aggressive)
                 if trend_just_changed:
                      signal = "EXIT_LONG"
                      self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped DOWN.{RESET}")
                 else:
                      # Optionally, exit if trend is down even if not a fresh change (more aggressive)
                      # signal = "EXIT_LONG"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend is DOWN (persistent).{RESET}")
                      pass # Default: Hold if trend flipped previously but no other exit condition met

            # Exit Condition 2: Price approaches/exceeds the nearest active Bearish OB (Resistance)
            # Check only if trend hasn't already triggered an exit
            if signal != "EXIT_LONG" and active_bear_obs:
                try:
                    # Find the bear OB whose top edge is closest to the current price
                    closest_bear_ob = min(active_bear_obs, key=lambda ob: abs(ob['top'] - latest_close))
                    # Calculate the exit threshold slightly *inside* the bear OB top (using exit proximity)
                    # exit_threshold = OB_Top - (OB_Top * (Factor - 1)) = OB_Top * (2 - Factor)
                    # Example: Factor=1.001 -> Threshold = Top * 0.999
                    exit_threshold = closest_bear_ob['top'] * (Decimal("2") - self.ob_exit_proximity_factor)

                    if latest_close >= exit_threshold:
                         signal = "EXIT_LONG"
                         self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price ({latest_close}) approached/crossed closest Bear OB {closest_bear_ob['id']} top ({closest_bear_ob['top']}). Exit Threshold >= {exit_threshold}{RESET}")
                except Exception as e:
                     self.logger.warning(f"Error checking Bear OB exit condition: {e}")

            # Add other potential exit conditions here (e.g., price crossing below key MA/band)

        elif current_pos_side == 'short':
             # Exit Condition 1: Trend flips UP
             if is_trend_up: # Trend is now UP
                 if trend_just_changed:
                     signal = "EXIT_SHORT"
                     self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped UP.{RESET}")
                 else:
                     # Optional: Exit if trend is UP persistently
                     # signal = "EXIT_SHORT"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend is UP (persistent).{RESET}")
                     pass

             # Exit Condition 2: Price approaches/exceeds the nearest active Bullish OB (Support)
             if signal != "EXIT_SHORT" and active_bull_obs:
                 try:
                    # Find the bull OB whose bottom edge is closest to the current price
                    closest_bull_ob = min(active_bull_obs, key=lambda ob: abs(ob['bottom'] - latest_close))
                    # Calculate the exit threshold slightly *inside* the bull OB bottom (using exit proximity)
                    # exit_threshold = OB_Bottom * Factor
                    exit_threshold = closest_bull_ob['bottom'] * self.ob_exit_proximity_factor

                    if latest_close <= exit_threshold:
                          signal = "EXIT_SHORT"
                          self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price ({latest_close}) approached/crossed closest Bull OB {closest_bull_ob['id']} bottom ({closest_bull_ob['bottom']}). Exit Threshold <= {exit_threshold}{RESET}")
                 except Exception as e:
                     self.logger.warning(f"Error checking Bull OB exit condition: {e}")

             # Add other potential exit conditions here

        # If an exit signal was generated, return it immediately. Do not check for entries.
        if signal in ["EXIT_LONG", "EXIT_SHORT"]:
             return signal

        # === 2. Check for ENTRY Signal (Only if NO position is currently open) ===
        if current_pos_side is None:
            # Entry Conditions:
            # a) Volumatic Trend matches the desired direction (UP for BUY, DOWN for SELL)
            # b) Price interacts with an *active* Order Block of the corresponding type
            #    (within the OB range, allowing slight penetration based on entry proximity factor)

            # Check for BUY (Long Entry)
            if is_trend_up and active_bull_obs:
                entry_ob = None
                for ob in active_bull_obs:
                    # Define entry zone: from OB bottom up to slightly above OB top
                    lower_bound = ob['bottom']
                    # upper_bound = OB_Top * Factor
                    upper_bound = ob['top'] * self.ob_entry_proximity_factor
                    # Check if latest close price falls within this zone
                    if lower_bound <= latest_close <= upper_bound:
                        entry_ob = ob
                        break # Found a potential entry OB, use the first one encountered

                if entry_ob:
                    signal = "BUY"
                    self.logger.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price ({latest_close}) in/near Bull OB {entry_ob['id']} ({entry_ob['bottom']}-{entry_ob['top']}). Entry Zone: [{lower_bound}, {upper_bound}]{RESET}")

            # Check for SELL (Short Entry)
            elif not is_trend_up and active_bear_obs: # Trend is DOWN
                 entry_ob = None
                 for ob in active_bear_obs:
                     # Define entry zone: from slightly below OB bottom up to OB top
                     # lower_bound = OB_Bottom * (2 - Factor)
                     lower_bound = ob['bottom'] * (Decimal("2") - self.ob_entry_proximity_factor)
                     upper_bound = ob['top']
                     # Check if latest close price falls within this zone
                     if lower_bound <= latest_close <= upper_bound:
                         entry_ob = ob
                         break

                 if entry_ob:
                     signal = "SELL"
                     self.logger.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price ({latest_close}) in/near Bear OB {entry_ob['id']} ({entry_ob['bottom']}-{entry_ob['top']}). Entry Zone: [{lower_bound}, {upper_bound}]{RESET}")

        # === 3. Log HOLD Reason ===
        if signal == "HOLD":
             if current_pos_side:
                 # Holding an existing position
                 self.logger.debug(f"HOLD Signal: Trend ({'UP' if is_trend_up else 'DOWN'}) compatible with existing {current_pos_side} position. Price ({latest_close}) not triggering exit conditions.")
             else:
                 # No position and no entry signal
                 trend_status = 'UP' if is_trend_up else 'DOWN'
                 ob_status = "No relevant active OBs found for current trend."
                 if is_trend_up and active_bull_obs: ob_status = f"Price ({latest_close}) not within entry zone of active Bull OBs."
                 elif not is_trend_up and active_bear_obs: ob_status = f"Price ({latest_close}) not within entry zone of active Bear OBs."
                 self.logger.debug(f"HOLD Signal: No position open. Trend is {trend_status}. {ob_status}")

        return signal

    def calculate_initial_tp_sl(
        self,
        entry_price: Decimal,
        signal: str, # "BUY" or "SELL"
        atr: Decimal,
        market_info: Dict,
        exchange: ccxt.Exchange # Needed for formatting
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial Take Profit (TP) and Stop Loss (SL) levels based on
        entry price, current ATR, configured multipliers, and market precision.
        These values are primarily used for position sizing and potentially for
        setting initial fixed protection orders (if TSL is disabled).

        Args:
            entry_price: The estimated or actual entry price (Decimal).
            signal: The trade direction signal ("BUY" or "SELL").
            atr: The current ATR value (Decimal, must be > 0).
            market_info: Market dictionary containing precision info.
            exchange: Initialized CCXT exchange object (for formatting).


        Returns:
            Tuple (take_profit_price, stop_loss_price):
            Calculated TP and SL prices as Decimals. TP can be None if disabled
            by multiplier=0. SL will be None only if calculation fails, as it's
            required for sizing.
        """
        lg = self.logger # Use the class logger

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            lg.error(f"Cannot calculate initial TP/SL: Invalid signal '{signal}'.")
            return None, None
        if not isinstance(entry_price, Decimal) or entry_price <= 0:
            lg.error(f"Cannot calculate initial TP/SL: Invalid entry price ({entry_price}).")
            return None, None
        if not isinstance(atr, Decimal) or atr <= 0:
            lg.error(f"Cannot calculate initial TP/SL: Invalid ATR value ({atr}).")
            return None, None
        if not market_info or 'precision' not in market_info or market_info['precision'].get('price') is None:
             lg.error(f"Cannot calculate initial TP/SL: Missing market price precision info.")
             return None, None

        try:
            # Get price precision (tick size) for rounding
            price_prec_str = market_info['precision']['price']
            min_tick_size = Decimal(str(price_prec_str))
            if min_tick_size <= 0: raise ValueError(f"Invalid tick size in market info: {min_tick_size}")

            # Get ATR multipliers from config (already validated in __init__)
            tp_multiple = self.initial_tp_atr_multiple
            sl_multiple = self.initial_sl_atr_multiple # Guaranteed > 0 by __init__

            # Calculate offsets from entry price
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple # Always positive offset

            take_profit_raw = None
            stop_loss_raw = None

            # Calculate raw levels based on signal direction
            if signal == "BUY":
                if tp_multiple > 0: take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
            elif signal == "SELL":
                if tp_multiple > 0: take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset

            # --- Format Prices Using Exchange Precision ---
            # Helper to format and validate
            def format_level(price_decimal: Optional[Decimal], level_type: str) -> Optional[Decimal]:
                if price_decimal is None: return None
                if price_decimal <= 0:
                    lg.warning(f"Calculated {level_type} is zero/negative ({price_decimal}). Setting to None.")
                    return None
                try:
                    # Use exchange.price_to_precision for correct rounding based on tick size
                    formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_decimal))
                    formatted_decimal = Decimal(formatted_str)
                    if formatted_decimal <= 0: # Double check after formatting
                        lg.warning(f"Formatted {level_type} became zero/negative ({formatted_decimal}). Setting to None.")
                        return None
                    return formatted_decimal
                except Exception as e:
                     lg.error(f"Error formatting {level_type} price {price_decimal}: {e}. Setting to None.")
                     return None

            take_profit = format_level(take_profit_raw, "TP")
            stop_loss = format_level(stop_loss_raw, "SL")

            # --- Final Validation ---
            # Ensure SL is strictly beyond entry price after formatting
            if stop_loss is not None:
                if signal == "BUY" and stop_loss >= entry_price:
                    lg.warning(f"Formatted BUY SL ({stop_loss}) is not below entry ({entry_price}). Adjusting down by one tick.")
                    stop_loss = entry_price - min_tick_size
                elif signal == "SELL" and stop_loss <= entry_price:
                    lg.warning(f"Formatted SELL SL ({stop_loss}) is not above entry ({entry_price}). Adjusting up by one tick.")
                    stop_loss = entry_price + min_tick_size
                # Re-check if adjustment made it non-positive
                if stop_loss <= 0:
                     lg.error(f"Adjusted SL became zero/negative ({stop_loss}). Cannot set valid SL.")
                     stop_loss = None

            # Ensure TP is strictly beyond entry price if enabled
            if take_profit is not None:
                if signal == "BUY" and take_profit <= entry_price:
                     lg.warning(f"Formatted BUY TP ({take_profit}) is not above entry ({entry_price}). Setting TP to None.")
                     take_profit = None
                elif signal == "SELL" and take_profit >= entry_price:
                     lg.warning(f"Formatted SELL TP ({take_profit}) is not below entry ({entry_price}). Setting TP to None.")
                     take_profit = None

            # Log final calculated values
            tp_str = f"{take_profit}" if take_profit else "None (Disabled or Invalid)"
            sl_str = f"{stop_loss}" if stop_loss else "None (Calculation Failed)"
            lg.debug(f"Calculated Initial Protection Levels: TP={tp_str}, SL={sl_str}")

            # Crucially, SL must be valid for position sizing to work
            if stop_loss is None:
                 lg.error(f"{NEON_RED}Stop Loss calculation failed or resulted in an invalid level. Cannot proceed with sizing.{RESET}")
                 return take_profit, None # Return TP if valid, but SL is None

            return take_profit, stop_loss

        except ValueError as ve:
             lg.error(f"{NEON_RED}Error calculating initial TP/SL (ValueError): {ve}{RESET}")
             return None, None
        except Exception as e:
             lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
             return None, None


# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy, # Passed instance
    signal_generator: SignalGenerator,   # Passed instance
    market_info: Dict                    # Passed validated market info
) -> None:
    """
    Performs one cycle of the trading logic for a single symbol:
    1. Fetches latest kline data.
    2. Runs the strategy analysis engine.
    3. Checks current position status.
    4. Generates a trading signal (BUY, SELL, HOLD, EXIT_*).
    5. Executes trades (if enabled): Enters, Exits, or Manages protections (BE, TSL).

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol to analyze.
        config: Current bot configuration dictionary.
        logger: Logger instance for this symbol.
        strategy_engine: Instantiated VolumaticOBStrategy object.
        signal_generator: Instantiated SignalGenerator object.
        market_info: Validated market information dictionary.
    """
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config.get('interval', 'N/A')}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe. Skipping cycle.")
         return # Should not happen if config validation works, but defensive check

    # Determine fetch limit, ensuring it meets strategy minimum requirements
    fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    min_strategy_len = strategy_engine.min_data_len
    if fetch_limit < min_strategy_len:
         lg.warning(f"{NEON_YELLOW}Configured fetch_limit ({fetch_limit}) is less than strategy's minimum required ({min_strategy_len}). Increasing fetch limit to {min_strategy_len + 50}.{RESET}")
         fetch_limit = min_strategy_len + 50 # Add buffer

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit, logger=lg)
    if klines_df.empty or len(klines_df) < min_strategy_len:
        # fetch_klines_ccxt logs errors, just add context here
        lg.error(f"Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}, need >= {min_strategy_len}). Skipping analysis cycle.")
        return

    # --- 2. Run Strategy Analysis ---
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
         lg.error(f"{NEON_RED}Error during strategy analysis update for {symbol}: {analysis_err}{RESET}", exc_info=True)
         return # Skip cycle on analysis error

    # Validate essential results needed for signal generation and actions
    if not isinstance(analysis_results, dict) or \
       analysis_results.get('current_trend_up') is None or \
       not isinstance(analysis_results.get('last_close'), Decimal) or analysis_results['last_close'] <= 0 or \
       not isinstance(analysis_results.get('atr'), Decimal) or analysis_results['atr'] <= 0:
         lg.error(f"{NEON_RED}Strategy analysis produced incomplete/invalid results for {symbol}. Cannot proceed with signal generation. Skipping rest of cycle.{RESET}")
         lg.debug(f"  Analysis Results Dump: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
         return

    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr'] # Guaranteed non-None and > 0 Decimal here

    # --- 3. Check Position & Generate Signal ---
    open_position = get_open_position(exchange, symbol, lg) # Returns standardized dict or None

    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err:
         lg.error(f"{NEON_RED}Error during signal generation for {symbol}: {signal_err}{RESET}", exc_info=True)
         return # Skip cycle on signal generation error

    # --- 4. Trading Logic ---
    trading_enabled = config.get("enable_trading", False)
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading disabled. Generated Signal: {signal}. Analysis complete for {symbol}.{RESET}")
        # Log intended actions if trading were enabled (optional)
        # if signal in ["BUY", "SELL"] and open_position is None: lg.info("  (Would attempt entry)")
        # elif signal in ["EXIT_LONG", "EXIT_SHORT"] and open_position: lg.info("  (Would attempt exit)")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis-Only Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return # End cycle here if trading is disabled

    # ===========================================
    # --- EXECUTION LOGIC (Trading Enabled) ---
    # ===========================================
    lg.debug(f"Trading enabled. Signal: {signal}. Position: {'Yes' if open_position else 'No'}")

    # === Scenario 1: No Open Position ===
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {NEON_GREEN if signal == 'BUY' else NEON_RED}{BRIGHT}{signal} Signal & No Position: Initiating Trade Entry Sequence for {symbol}{RESET} ***")

            # --- Pre-Trade Checks ---
            # Concurrent position check (simple, assumes this script instance controls the symbol exclusively)
            # max_pos = config.get("max_concurrent_positions", 1) # Check config limit
            # if max_pos <= 0: # Effectively disable new entries via config
            #      lg.warning(f"Max concurrent positions set to {max_pos}. Skipping new entry for {symbol}.")
            #      return # Skip entry if limit is 0 or less

            # Fetch current balance
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= Decimal('0'): # Check balance > 0
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative ({balance}).{RESET}")
                return

            # Calculate initial SL/TP for sizing (using latest close as entry estimate)
            lg.debug(f"Calculating initial TP/SL for sizing using Entry={latest_close}, ATR={current_atr}")
            initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(
                 entry_price=latest_close, signal=signal, atr=current_atr,
                 market_info=market_info, exchange=exchange
            )

            # SL is MANDATORY for risk-based sizing
            if initial_sl_calc is None:
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed. Cannot size position.{RESET}")
                 return
            # TP is optional, log if it failed but proceed if SL is valid
            if initial_tp_calc is None:
                 lg.warning(f"{NEON_YELLOW}Initial TP calculation failed or disabled. Proceeding without fixed TP target for sizing.{RESET}")

            # Set Leverage (if applicable for contracts)
            leverage_set_success = True # Assume success for spot or if leverage <= 0
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 0)) # Default leverage 0 if missing
                if leverage > 0:
                    leverage_set_success = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg)
                    if not leverage_set_success:
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage to {leverage}x.{RESET}")
                         return # Stop entry if leverage setting fails
                else:
                    lg.info(f"Leverage setting skipped: Leverage ({leverage}) is zero or negative in config.")
            # else: lg.info(f"Leverage setting skipped ({symbol} is Spot).") # Already logged by set_leverage_ccxt

            # Calculate Position Size
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=initial_sl_calc, # Use the validated SL
                entry_price=latest_close, # Use latest close as entry estimate for sizing
                market_info=market_info,
                exchange=exchange,
                logger=lg
            )

            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size calculated ({position_size}). Check balance, risk, SL distance, and market limits.{RESET}")
                return

            # --- Place Entry Trade ---
            lg.info(f"==> Placing {signal} market order | Size: {position_size} {market_info.get('base', '')} <==")
            trade_order = place_trade(
                exchange, symbol, signal, position_size, market_info, lg, reduce_only=False
            )

            # --- Post-Trade: Verify Position & Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                lg.info(f"Confirming position status for {symbol} after order {order_id}...")
                # Re-fetch position state after delay
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        # Get actual entry price from confirmed position for more accurate protection setting
                        entry_price_actual_str = confirmed_position.get('entryPrice')
                        entry_price_actual = None
                        if entry_price_actual_str:
                            try: entry_price_actual = Decimal(str(entry_price_actual_str))
                            except Exception: pass

                        if not entry_price_actual or entry_price_actual <= 0:
                             lg.warning(f"Could not get valid actual entry price from confirmed position {confirmed_position.get('info', {})}. Using initial estimate {latest_close} for protection.")
                             entry_price_actual = latest_close # Fallback to estimate
                        else:
                              lg.info(f"{NEON_GREEN}Position Confirmed! Side: {confirmed_position.get('side')}, Actual Entry: ~{entry_price_actual}{RESET}")


                        # --- Set Protection based on Config ---
                        protection_set_success = False
                        protection_cfg = config.get("protection", {})

                        # Option 1: Trailing Stop Loss (if enabled)
                        if protection_cfg.get("enable_trailing_stop", True): # Default TSL to True if key missing? Safer maybe.
                             lg.info(f"Setting Trailing Stop Loss using actual entry {entry_price_actual}...")
                             # Recalculate TP target based on *actual* entry for TSL setup
                             tp_for_tsl, _ = signal_generator.calculate_initial_tp_sl(
                                  entry_price=entry_price_actual, signal=signal, atr=current_atr,
                                  market_info=market_info, exchange=exchange
                             )
                             protection_set_success = set_trailing_stop_loss(
                                 exchange=exchange, symbol=symbol, market_info=market_info,
                                 position_info=confirmed_position, config=config, logger=lg,
                                 take_profit_price=tp_for_tsl # Pass optional recalculated TP target
                             )
                         # Option 2: Fixed SL/TP (if TSL disabled but fixed SL/TP enabled via multipliers)
                         elif protection_cfg.get("initial_stop_loss_atr_multiple", 0) > 0 or \
                              protection_cfg.get("initial_take_profit_atr_multiple", 0) > 0:
                             lg.info(f"Setting Fixed SL/TP using actual entry {entry_price_actual}...")
                             # Recalculate both SL and TP based on actual entry
                             tp_final_fixed, sl_final_fixed = signal_generator.calculate_initial_tp_sl(
                                  entry_price=entry_price_actual, signal=signal, atr=current_atr,
                                  market_info=market_info, exchange=exchange
                             )
                             # Need at least one valid level to make the API call
                             if sl_final_fixed or tp_final_fixed:
                                 protection_set_success = _set_position_protection(
                                     exchange=exchange, symbol=symbol, market_info=market_info,
                                     position_info=confirmed_position, logger=lg,
                                     stop_loss_price=sl_final_fixed,
                                     take_profit_price=tp_final_fixed
                                 )
                             else:
                                 lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed based on actual entry. No fixed protection set.{RESET}")
                                 protection_set_success = True # No action needed, consider success
                         else:
                              lg.info("Neither Trailing Stop nor Fixed SL/TP multipliers enabled in config. No protection set.")
                              protection_set_success = True # No action required, considered success

                        # --- Log Overall Outcome ---
                        if protection_set_success:
                             lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                        else:
                             # _set_position_protection or set_trailing_stop_loss logs the specific failure reason
                             lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/UPDATE PROTECTION ({symbol} {signal}) ===")
                             lg.warning(f"{NEON_YELLOW}MANUAL MONITORING REQUIRED! Position is open without fully configured automated protection.{RESET}")

                    except Exception as post_trade_err:
                         lg.error(f"{NEON_RED}Critical error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                         lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")
                else:
                    # Position not found after placing order and waiting
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position for {symbol} after {confirm_delay}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Possible reasons: Order rejected, filled & closed instantly (check history), API delay, fetch_positions issue. Manual investigation required!{RESET}")
            else:
                # Trade placement itself failed (place_trade logs the error)
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). See previous logs. ===")
        else: # signal == HOLD and no position
            lg.info(f"Signal is HOLD and no open position for {symbol}. No entry action taken.")

    # === Scenario 2: Existing Open Position ===
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size_str = open_position.get('size_decimal', 'N/A') # Use standardized Decimal size if available
        if pos_size_str == 'N/A': # Fallback
             pos_size_str = open_position.get('contracts') or open_position.get('info',{}).get('size', 'N/A')

        lg.info(f"Existing {pos_side.upper()} position found for {symbol} (Size: {pos_size_str}). Signal: {signal}")

        # --- Check for Exit Signal ---
        exit_signal_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                                (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal Triggered for existing {pos_side} position on {symbol}. Initiating Close Sequence... ***{RESET}")
            try:
                # Determine close side and get size from standardized position info
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                # Use the reliable Decimal size calculated in get_open_position
                size_to_close = abs(open_position.get('size_decimal', Decimal('0')))

                if size_to_close <= 0:
                    # This might happen if position info is slightly stale and it was just closed by SL/TP/Manual action
                    lg.warning(f"Position size to close is zero or negative ({size_to_close}). Position might already be closed. Skipping close attempt.")
                else:
                    lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                    close_order = place_trade(
                        exchange, symbol, close_side_signal, size_to_close,
                        market_info, lg, reduce_only=True
                    )

                    if close_order and close_order.get('id'):
                        lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                        # Optional: Wait and verify closure with another get_open_position call?
                        # time.sleep(confirm_delay)
                        # final_pos_check = get_open_position(exchange, symbol, lg)
                        # if final_pos_check is None: lg.info("Position closure confirmed.")
                        # else: lg.warning("Position closure order sent, but position still detected.")
                    else:
                        # place_trade logs the specific error
                        lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check required! Position might still be open.{RESET}")

            except KeyError as ke:
                 lg.error(f"{NEON_RED}Error preparing close order for {symbol}: Missing key {ke} in position info.{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error closing position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")

        # --- Manage Existing Position (HOLD signal or compatible signal) ---
        else:
            lg.info(f"Signal ({signal}) allows holding or is aligned with existing {pos_side} position. Managing protections...")

            # --- Manage Protections: Break-Even (BE) and Trailing Stop Loss (TSL) ---
            protection_cfg = config.get("protection", {})

            # Check if TSL is currently active on the position (based on Bybit V5 fields)
            is_tsl_active_on_pos = False
            try:
                # Check if 'trailingStop' distance is set and positive in position info
                tsl_dist_str = open_position.get('trailingStopLoss') # Standardized field from get_open_position
                if tsl_dist_str and Decimal(str(tsl_dist_str)) > 0:
                     # Also check if activation price exists (might be 0 if activated immediately)
                     # tsl_act_str = open_position.get('tslActivationPrice')
                     # if tsl_act_str is not None: # Presence implies TSL setup
                     is_tsl_active_on_pos = True
                     lg.debug("Trailing Stop Loss appears active on current position (TSL distance > 0).")
            except Exception as tsl_check_err:
                lg.warning(f"Could not reliably check if TSL is active: {tsl_check_err}")

            # --- Break-Even Logic ---
            # Apply BE only if:
            # 1. BE is enabled in config.
            # 2. TSL is NOT currently active on the position (BE usually precedes TSL activation).
            if protection_cfg.get("enable_break_even", True) and not is_tsl_active_on_pos:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    entry_price_str = open_position.get('entryPrice')
                    if not entry_price_str: raise ValueError("Missing entry price for BE check")
                    entry_price = Decimal(str(entry_price_str))

                    # Get BE parameters from config
                    be_trigger_atr_mult = Decimal(str(protection_cfg.get("break_even_trigger_atr_multiple", 1.0)))
                    be_offset_ticks = int(protection_cfg.get("break_even_offset_ticks", 2))
                    if be_trigger_atr_mult <= 0: raise ValueError("BE trigger ATR multiple must be positive")
                    if be_offset_ticks < 0: raise ValueError("BE offset ticks cannot be negative")

                    # Calculate current profit in terms of ATR multiples
                    price_diff = (latest_close - entry_price) if pos_side == 'long' else (entry_price - latest_close)
                    # current_atr is guaranteed > 0 from earlier check
                    profit_in_atr = price_diff / current_atr

                    lg.debug(f"BE Check: Entry={entry_price}, Close={latest_close}, Diff={price_diff}, ATR={current_atr}")
                    lg.debug(f"  Profit ATRs = {profit_in_atr:.2f}, Trigger ATRs = {be_trigger_atr_mult}")

                    # Check if profit target for BE is reached
                    if profit_in_atr >= be_trigger_atr_mult:
                        # Calculate the target BE stop price (entry + offset)
                        price_prec_str = market_info['precision']['price']
                        min_tick_size = Decimal(str(price_prec_str))
                        tick_offset_value = min_tick_size * be_offset_ticks

                        # Calculate BE stop price, rounding away from entry
                        be_stop_price = (entry_price + tick_offset_value).quantize(min_tick_size, rounding=ROUND_UP) if pos_side == 'long' \
                                   else (entry_price - tick_offset_value).quantize(min_tick_size, rounding=ROUND_DOWN)

                        if be_stop_price <= 0: raise ValueError("Calculated BE stop price is zero/negative")

                        # Get current SL price from position info to see if BE is an improvement
                        current_sl_price = None
                        # Use standardized field from get_open_position
                        current_sl_str = open_position.get('stopLossPrice')
                        if current_sl_str and str(current_sl_str) != '0': # Check if SL is set
                            try: current_sl_price = Decimal(str(current_sl_str))
                            except Exception: lg.warning(f"Could not parse current SL price: {current_sl_str}")

                        # Determine if we should update the SL to the BE level
                        update_be = False
                        if current_sl_price is None:
                             update_be = True # No current SL exists, so set the BE SL
                             lg.info("BE triggered: No current fixed SL detected.")
                        elif pos_side == 'long' and be_stop_price > current_sl_price:
                             update_be = True # BE target is better (higher) than current SL
                             lg.info(f"BE triggered: Target BE SL {be_stop_price} > Current SL {current_sl_price}.")
                        elif pos_side == 'short' and be_stop_price < current_sl_price:
                             update_be = True # BE target is better (lower) than current SL
                             lg.info(f"BE triggered: Target BE SL {be_stop_price} < Current SL {current_sl_price}.")
                        else:
                             lg.debug(f"BE Profit target reached, but current SL ({current_sl_price}) is already better than or equal to BE target ({be_stop_price}). No BE SL update needed.")

                        if update_be:
                            lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                            # Get current TP to preserve it when setting the new BE SL
                            current_tp_price = None
                            current_tp_str = open_position.get('takeProfitPrice') # Standardized field
                            if current_tp_str and str(current_tp_str) != '0':
                                 try: current_tp_price = Decimal(str(current_tp_str))
                                 except Exception: pass

                            # Call the protection helper to set the new SL (and potentially keep existing TP)
                            # TSL distance/activation should be None here as we are setting a fixed BE SL
                            success = _set_position_protection(
                                exchange, symbol, market_info, open_position, lg,
                                stop_loss_price=be_stop_price,
                                take_profit_price=current_tp_price, # Preserve existing TP if any
                                trailing_stop_distance=None,
                                tsl_activation_price=None
                            )
                            if success:
                                lg.info(f"{NEON_GREEN}Break-Even SL set/updated successfully.{RESET}")
                                # Optional: Update local position_info state? get_open_position will refresh next cycle.
                            else:
                                lg.error(f"{NEON_RED}Failed to set/update Break-Even SL via API.{RESET}")
                    else:
                        lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {be_trigger_atr_mult} ATRs).")

                except ValueError as ve:
                    lg.error(f"{NEON_RED}Error during break-even calculation ({symbol}): {ve}{RESET}")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Unexpected error during break-even check ({symbol}): {be_err}{RESET}", exc_info=True)
            elif not protection_cfg.get("enable_break_even", True):
                lg.debug("Break-even check skipped: Disabled in config.")
            elif is_tsl_active_on_pos:
                 lg.debug("Break-even check skipped: Trailing Stop Loss is already active on the position.")

            # --- Trailing Stop Management (Recovery) ---
            # If TSL is enabled in config, but we detected it's NOT active on the position,
            # attempt to set it now. This acts as a recovery if initial TSL setting failed,
            # or if BE was active previously and now TSL should take over (if profit increased further).
            # Note: This might override a manually set fixed SL if TSL is enabled.
            if protection_cfg.get("enable_trailing_stop", True) and not is_tsl_active_on_pos:
                 lg.warning(f"{NEON_YELLOW}TSL is enabled in config but not detected as active on the current {pos_side} position. Attempting TSL setup/recovery...{RESET}")
                 # Need entry price and current ATR to set TSL
                 entry_price_str = open_position.get('entryPrice')
                 if entry_price_str:
                      try:
                          entry_price = Decimal(str(entry_price_str))
                          # Recalculate TP target based on entry for TSL setup consistency
                          tp_recalc, _ = signal_generator.calculate_initial_tp_sl(
                               entry_price=entry_price, signal=pos_side.upper(), atr=current_atr,
                               market_info=market_info, exchange=exchange
                          )
                          # Attempt to set TSL (which will call _set_position_protection)
                          tsl_set_success = set_trailing_stop_loss(
                               exchange=exchange, symbol=symbol, market_info=market_info,
                               position_info=open_position, config=config, logger=lg,
                               take_profit_price=tp_recalc # Pass optional TP target
                          )
                          if tsl_set_success: lg.info(f"TSL setup/recovery attempt successful for {symbol}.")
                          else: lg.error(f"TSL setup/recovery attempt failed for {symbol}.")
                      except Exception as e:
                           lg.error(f"Error during TSL recovery attempt: {e}")
                 else:
                      lg.error(f"{NEON_RED}Cannot attempt TSL recovery: Missing entry price in position info.{RESET}")
            # else: # TSL disabled or already active
            #      lg.debug("TSL recovery check skipped.")


    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """
    Main function to initialize the bot, set up the exchange connection,
    get user input for symbol and interval, and run the main trading loop.
    """
    global CONFIG, QUOTE_CURRENCY # Allow access to global config state

    # Use the dedicated 'init' logger for setup phases
    init_logger.info(f"{BRIGHT}--- Starting Pyrmethus Volumatic Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{RESET}")

    # Initial configuration load handled globally above

    init_logger.info(f"Configuration Loaded: Quote Currency={QUOTE_CURRENCY}, Trading Enabled={CONFIG.get('enable_trading')}, Sandbox={CONFIG.get('use_sandbox')}")
    # Log library versions for debugging
    try:
        init_logger.info(f"Using Versions: Python={os.sys.version.split()[0]}, CCXT={ccxt.__version__}, Pandas={pd.__version__}, NumPy={np.__version__}, PandasTA={getattr(ta, 'version', 'N/A')}")
    except Exception as e:
        init_logger.warning(f"Could not determine all library versions: {e}")

    # --- User Confirmation for Live Trading ---
    if CONFIG.get("enable_trading", False): # Check if trading is truly enabled
         init_logger.warning(f"{NEON_YELLOW}{BRIGHT}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox", True): # Double-check sandbox setting
              init_logger.warning(f"{NEON_YELLOW}Mode: SANDBOX (Testnet) Environment{RESET}")
         else:
              init_logger.warning(f"{NEON_RED}{BRIGHT}Mode: LIVE (Real Money) Environment{RESET}")

         # Display key risk parameters for review
         protection_cfg = CONFIG.get("protection", {})
         init_logger.warning(f"{BRIGHT}--- Key Settings Review ---{RESET}")
         init_logger.warning(f"  Risk Per Trade: {CONFIG.get('risk_per_trade', 0)*100:.2f}%")
         init_logger.warning(f"  Leverage: {CONFIG.get('leverage', 0)}x")
         init_logger.warning(f"  Trailing SL: {'ENABLED' if protection_cfg.get('enable_trailing_stop', True) else 'DISABLED'}")
         init_logger.warning(f"  Break Even: {'ENABLED' if protection_cfg.get('enable_break_even', True) else 'DISABLED'}")
         init_logger.warning(f"---------------------------")
         try:
             # Prompt user for confirmation before proceeding with live trading
             input(f"{BRIGHT}>>> Review settings CAREFULLY. Press {NEON_GREEN}Enter{RESET}{BRIGHT} to continue, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to abort... {RESET}")
             init_logger.info("User confirmed live trading settings. Proceeding...")
         except KeyboardInterrupt:
             init_logger.info("User aborted startup during confirmation.")
             print(f"\n{NEON_YELLOW}Bot startup aborted by user.{RESET}")
             return # Exit script
    else:
         init_logger.info(f"{NEON_YELLOW}Trading is disabled in config. Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange Connection ---
    init_logger.info("Initializing exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize exchange connection. Cannot continue. Exiting.{RESET}")
        return
    init_logger.info(f"Exchange {exchange.id} initialized successfully.")

    # --- Get and Validate Trading Symbol ---
    target_symbol = None
    market_info = None
    while target_symbol is None: # Loop until a valid symbol is confirmed
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter the trading symbol (e.g., BTC/USDT, ETH/USDT:USDT): {RESET}").strip().upper()
            if not symbol_input_raw: continue # Ask again if input is empty

            # Standardize common variations (e.g., Bybit perpetuals often use :)
            # Try direct input first, then common forms like BASE/QUOTE:QUOTE
            symbols_to_try = [symbol_input_raw]
            if '/' in symbol_input_raw and ':' not in symbol_input_raw:
                 symbols_to_try.append(f"{symbol_input_raw}:{QUOTE_CURRENCY}") # e.g., BTC/USDT -> BTC/USDT:USDT
            elif ':' in symbol_input_raw and '/' not in symbol_input_raw:
                 symbols_to_try.append(symbol_input_raw.replace(':', '/')) # e.g., BTC:USDT -> BTC/USDT

            symbols_to_try = list(dict.fromkeys(symbols_to_try)) # Remove duplicates

            for symbol_attempt in symbols_to_try:
                init_logger.info(f"Validating symbol '{symbol_attempt}'...")
                market_info_attempt = get_market_info(exchange, symbol_attempt, init_logger)

                if market_info_attempt:
                    target_symbol = market_info_attempt['symbol'] # Use the validated symbol from CCXT
                    market_info = market_info_attempt # Store the validated market info
                    market_type_desc = market_info.get('contract_type_str', "Unknown Type")
                    init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                    break # Exit the inner loop once a valid symbol is found
            else:
                 # If loop finishes without breaking
                 if market_info is None:
                      init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' (and variations {symbols_to_try}) not found or invalid on {exchange.id}. Please check the symbol and try again.{RESET}")

        except KeyboardInterrupt:
            init_logger.info("User aborted startup during symbol input.")
            print(f"\n{NEON_YELLOW}Bot startup aborted by user.{RESET}")
            return
        except Exception as e:
            init_logger.error(f"Error during symbol validation: {e}", exc_info=True)
            # Loop will continue to ask for symbol

    # --- Get and Validate Analysis Interval ---
    selected_interval = None
    while selected_interval is None: # Loop until valid interval
        default_interval = CONFIG.get('interval', '5') # Get default from current config
        interval_input = input(f"{NEON_YELLOW}Enter analysis interval {VALID_INTERVALS} (default: {default_interval}): {RESET}").strip()
        if not interval_input:
            interval_input = default_interval # Use default if input is empty

        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             CONFIG["interval"] = selected_interval # Update config dictionary in memory
             ccxt_tf = CCXT_INTERVAL_MAP.get(selected_interval)
             init_logger.info(f"Using interval: {selected_interval} (CCXT Timeframe: {ccxt_tf})")
             # Optionally save the updated config back to file here if desired
             # save_config(CONFIG, CONFIG_FILE)
             break
        else:
             init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Please choose from {VALID_INTERVALS}.{RESET}")

    # --- Setup Symbol-Specific Logger ---
    symbol_logger = setup_logger(target_symbol) # Use the validated symbol for logger name
    symbol_logger.info(f"---=== {BRIGHT}Starting Trading Loop for {target_symbol} ({CONFIG['interval']}){RESET} ===---")
    symbol_logger.info(f"Trading Enabled: {trading_enabled}, Sandbox Mode: {CONFIG.get('use_sandbox')}")
    # Log key parameters again for the symbol-specific log
    protection_cfg = CONFIG.get("protection", {})
    symbol_logger.info(f"Settings: Risk={CONFIG['risk_per_trade']:.2%}, Leverage={CONFIG['leverage']}x, TSL={'ON' if protection_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if protection_cfg.get('enable_break_even') else 'OFF'}")
    symbol_logger.debug(f"Strategy Params: {json.dumps(CONFIG.get('strategy_params', {}))}")

    # --- Instantiate Strategy Engine and Signal Generator ---
    # Pass the current config, validated market_info, and symbol-specific logger
    try:
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_init_err:
        symbol_logger.critical(f"{NEON_RED}Failed to initialize strategy engine or signal generator: {engine_init_err}. Exiting.{RESET}", exc_info=True)
        return

    # --- Main Trading Loop ---
    symbol_logger.info(f"{BRIGHT}Entering main trading loop... Press Ctrl+C to stop.{RESET}")
    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Start: {datetime.now(TIMEZONE).strftime('%H:%M:%S %Z')}")

            # --- Core Logic Execution ---
            try:
                # --- Optional: Dynamic Config Reload ---
                # Uncomment below to reload config file each cycle.
                # Be cautious, as this can change behavior mid-run.
                # current_config = load_config(CONFIG_FILE)
                # if current_config != CONFIG:
                #     symbol_logger.info("Configuration changed, reloading...")
                #     CONFIG = current_config
                #     QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
                #     # Update components that depend on config
                #     strategy_engine.config = CONFIG
                #     signal_generator.config = CONFIG
                #     # Re-log key settings if needed
                #     protection_cfg = CONFIG.get("protection", {})
                #     symbol_logger.info(f"Reloaded Settings: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if protection_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if protection_cfg.get('enable_break_even') else 'OFF'}")
                # --- End Optional Reload ---

                analyze_and_trade_symbol(
                    exchange=exchange,
                    symbol=target_symbol,
                    config=CONFIG, # Pass current config state
                    logger=symbol_logger,
                    strategy_engine=strategy_engine,
                    signal_generator=signal_generator,
                    market_info=market_info # Pass validated market info
                )
            # --- Robust Error Handling for Common Issues within the Loop ---
            except ccxt.RateLimitExceeded as e:
                 symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting 60 seconds...{RESET}")
                 time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e:
                 # Catch common network/connection issues
                 symbol_logger.error(f"{NEON_RED}Network error encountered: {e}. Waiting {RETRY_DELAY_SECONDS * 3}s before next cycle...{RESET}")
                 time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                 # API keys invalid or expired - Fatal error
                 symbol_logger.critical(f"{NEON_RED}CRITICAL Authentication Error: {e}. API keys may be invalid, expired, or permissions revoked. Stopping bot.{RESET}")
                 break # Exit the main trading loop
            except ccxt.ExchangeNotAvailable as e:
                 # Exchange temporarily unavailable (e.g., Cloudflare issues, DDOS)
                 symbol_logger.error(f"{NEON_RED}Exchange not available: {e}. Waiting 60 seconds...{RESET}")
                 time.sleep(60)
            except ccxt.OnMaintenance as e:
                 # Exchange is under scheduled maintenance
                 symbol_logger.error(f"{NEON_RED}Exchange under maintenance: {e}. Waiting 5 minutes...{RESET}")
                 time.sleep(300)
            except ccxt.ExchangeError as e:
                 # Catch other specific exchange errors (e.g., invalid parameters not caught earlier)
                 symbol_logger.error(f"{NEON_RED}Unhandled Exchange Error in main loop: {e}{RESET}", exc_info=True)
                 symbol_logger.warning(f"{NEON_YELLOW}Pausing for 10 seconds before next cycle.{RESET}")
                 time.sleep(10) # Short pause
            except Exception as loop_error:
                 # Catch any other unexpected errors during the cycle
                 symbol_logger.error(f"{NEON_RED}Critical unexpected error in main trading loop: {loop_error}{RESET}", exc_info=True)
                 symbol_logger.warning(f"{NEON_YELLOW}Pausing for 15 seconds due to unexpected error.{RESET}")
                 time.sleep(15)

            # --- Loop Delay Calculation ---
            elapsed_time = time.time() - loop_start_time
            loop_delay = config.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_time = max(0, loop_delay - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle processed in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Initiating graceful shutdown...")
    except Exception as critical_error:
        # Catch errors outside the inner try/except (e.g., during loop setup)
        init_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR outside main loop: {critical_error}{RESET}", exc_info=True)
    finally:
        # --- Shutdown Procedures ---
        shutdown_msg = f"--- Pyrmethus Volumatic Bot for {target_symbol or 'N/A'} Stopping ---"
        init_logger.info(shutdown_msg)
        # Log shutdown message to symbol logger if it was initialized
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
             symbol_logger.info(shutdown_msg)

        # Close exchange connection if it was successfully initialized
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Attempting to close CCXT exchange connection...")
                # Some exchanges might require explicit close, others might not
                # exchange.close() # Uncomment if needed, check ccxt docs for your exchange
                init_logger.info("Exchange connection closed (or no action needed).")
            except Exception as close_err:
                init_logger.error(f"Error closing exchange connection: {close_err}")

        logging.shutdown() # Flush and close all logging handlers properly
        print(f"\n{NEON_YELLOW}{BRIGHT}Bot stopped.{RESET}")


if __name__ == "__main__":
    # This script should be saved as 'pyrmethus_volumatic_bot.py' and run directly.
    # Ensure all dependencies listed at the top are installed:
    # pip install ccxt pandas numpy pandas_ta requests python-dotenv colorama tzdata
    main()
