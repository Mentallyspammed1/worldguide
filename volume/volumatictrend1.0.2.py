# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL).

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
from typing import Any, Dict, Optional, Tuple, List, TypedDict
from zoneinfo import ZoneInfo # Requires tzdata package

import numpy as np
import pandas as pd
import pandas_ta as ta # Requires pandas_ta
import requests # Requires requests
import websocket # Requires websocket-client (For potential future WS integration)
import ccxt # Requires ccxt
from colorama import Fore, Style, init # Requires colorama
from dotenv import load_dotenv # Requires python-dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry # requests automatically uses urllib3

# --- Initialize Environment and Settings ---
getcontext().prec = 28  # High precision for Decimal calculations
init(autoreset=True) # Init Colorama
load_dotenv() # Load environment variables from .env file

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago") # Adjust timezone as needed
MAX_API_RETRIES = 3 # Max retries for recoverable API errors (Network, 429, 5xx)
RETRY_DELAY_SECONDS = 5 # Delay between retries for network/rate limit errors
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Intervals supported by the bot logic
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable

# Default Strategy/Indicator Periods (can be overridden by config.json)
# Volumatic Trend Params
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200 # ATR period used in Volumatic Trend calc
DEFAULT_VT_VOL_EMA_LENGTH = 1000 # Length for Volume smoothing
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

# Fetch limit for initial historical data
DEFAULT_FETCH_LIMIT = 750 # Ensure enough data for indicator lookbacks
MAX_DF_LEN = 2000 # Keep DataFrame size manageable

LOOP_DELAY_SECONDS = 15 # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY_SECONDS = 8 # Wait time after placing order before confirming position

# QUOTE_CURRENCY dynamically loaded from config

# Neon Color Scheme for logging
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Configuration Loading ---
class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger for the given name (e.g., 'init' or symbol) with file and console handlers."""
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding multiple handlers if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels for potential debugging

    # File Handler (logs everything)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_filename}: {e}")

    # Console Handler (logs INFO and above by default)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level (e.g., INFO for normal operation, DEBUG for detailed tracing)
    console_log_level = logging.INFO
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent root logger from handling messages again
    return logger

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present and interval is valid."""
    default_config = {
        "interval": "5", # Bot's internal interval name (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS,
        "fetch_limit": DEFAULT_FETCH_LIMIT,
        "orderbook_limit": 25, # Depth of orderbook to fetch (if used in future)
        "enable_trading": False, # SAFETY FIRST: Default to False (dry run)
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet)
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade (e.g., 0.01)
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions": 1, # Limit open positions for this symbol/script instance
        "quote_currency": "USDT", # Currency for balance check and sizing (e.g., USDT, USDC)
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Pause between main loop cycles
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait after order before checking position

        # --- Strategy Parameters (Volumatic Trend & OB) ---
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,
            "ob_source": DEFAULT_OB_SOURCE, # "Wicks" or "Bodys" - defines OB candle range
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT, # Pivot High lookback/forward periods
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT, # Pivot Low lookback/forward periods
            "ob_extend": DEFAULT_OB_EXTEND, # Extend OB lines until violated?
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES, # Limit number of OBs stored/checked
            "ob_entry_proximity_factor": 1.005, # How close price needs to be to OB edge for entry (e.g., 1.005 allows price 0.5% *beyond* the box edge)
            "ob_exit_proximity_factor": 1.001 # How close price needs to be to OB edge for exit signal (tighter)
        },

        # --- Protection Settings ---
        "protection": {
             "enable_trailing_stop": True,
             "trailing_stop_callback_rate": 0.005, # Trail distance as % of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Activate TSL when profit reaches X% of entry (e.g., 0.003 = 0.3%)
             "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0, # Move SL to BE when profit >= X * ATR
             "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry price (using market tick size)
             "initial_stop_loss_atr_multiple": 1.8, # ATR multiple for initial SL (used for sizing & potentially fixed SL)
             "initial_take_profit_atr_multiple": 0.7 # ATR multiple for initial TP (if not using TSL or as TSL target)
        }
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            # Return default anyway to allow bot to potentially run
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
        # Ensure all default keys exist, preserving existing values
        updated_config = _ensure_config_keys(config_from_file, default_config)

        # Validate interval value after loading and merging defaults
        interval_from_config = updated_config.get("interval")
        if interval_from_config not in VALID_INTERVALS:
            print(f"{NEON_RED}Invalid interval '{interval_from_config}' found in config. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            # Mark config for saving back if interval was corrected
            config_needs_saving = True
        else:
             # Check if any other keys were added from default
             config_needs_saving = (updated_config != config_from_file)


        if config_needs_saving:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{NEON_YELLOW}Updated config file with missing default keys or corrected interval: {filepath}{RESET}")
             except IOError as e:
                 print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

        return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        try:
            # Attempt to recreate default config if loading failed
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
             print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        # If the key exists and both values are dictionaries, recurse
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config

# Initial config load for global constants
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

# --- Logger Setup ---
# setup_logger function defined above

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with error handling and connection test."""
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear', # Default to linear for USDT/USDC perpetuals
                'adjustForTimeDifference': True, # Adjust for clock skew
                # Increase default timeouts for potentially slow operations
                'fetchTickerTimeout': 15000, # ms
                'fetchBalanceTimeout': 20000, # ms
                'createOrderTimeout': 30000, # ms
                'cancelOrderTimeout': 20000, # ms
            }
        }
        exchange_class = ccxt.bybit # Explicitly use bybit
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        lg.info(f"Loading markets for {exchange.id}...")
        # Use a retry mechanism for load_markets
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                exchange.load_markets()
                lg.info(f"Markets loaded successfully for {exchange.id}.")
                break # Success
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error loading markets (Attempt {attempt+1}/{MAX_API_RETRIES+1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.critical(f"{NEON_RED}Max retries reached loading markets for {exchange.id}. Exiting.{RESET}")
                    return None # Critical failure if markets can't be loaded
            except ccxt.ExchangeError as e:
                 lg.critical(f"{NEON_RED}Exchange error loading markets: {e}. Exiting.{RESET}")
                 return None
            except Exception as e:
                 lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                 return None

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys with balance fetch
        account_type_to_test = 'CONTRACT' # Try contract first for perpetuals
        lg.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            # Use fetch_balance helper which handles different structures
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_val:.4f})")
            else:
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch returned None, but connection might be okay. Check API key permissions if trading fails.{RESET}")
        except ccxt.AuthenticationError as auth_err:
             lg.critical(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             lg.critical(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
             return None
        except Exception as balance_err:
             # Non-auth errors during initial fetch are warnings, as fetch_balance has fallbacks
             lg.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type if trading fails.{RESET}")

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
    """Fetch the current price of a trading symbol using CCXT ticker with retries and fallbacks."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            # lg.debug(f"Ticker data for {symbol}: {ticker}") # Can be verbose

            price = None
            # Prioritize 'last', then mid-price (bid+ask)/2, then ask, then bid
            last_price = ticker.get('last')
            bid_price = ticker.get('bid')
            ask_price = ticker.get('ask')

            if last_price is not None:
                try:
                    p = Decimal(str(last_price))
                    if p > 0: price = p; lg.debug(f"Using 'last' price: {p}")
                except Exception: lg.warning(f"Invalid 'last' price format: {last_price}")

            if price is None and bid_price is not None and ask_price is not None:
                try:
                    bid = Decimal(str(bid_price))
                    ask = Decimal(str(ask_price))
                    if bid > 0 and ask > 0 and ask >= bid: # Sanity check ask >= bid
                        price = (bid + ask) / 2
                        lg.debug(f"Using bid/ask midpoint: {price} (Bid: {bid}, Ask: {ask})")
                    else: lg.warning(f"Invalid bid/ask values: Bid={bid}, Ask={ask}")
                except Exception: lg.warning(f"Invalid bid/ask format: {bid_price}, {ask_price}")

            if price is None and ask_price is not None: # Fallback to ask
                 try:
                      p = Decimal(str(ask_price))
                      if p > 0: price = p; lg.warning(f"{NEON_YELLOW}Using 'ask' price fallback: {p}{RESET}")
                 except Exception: lg.warning(f"Invalid 'ask' price format: {ask_price}")

            if price is None and bid_price is not None: # Fallback to bid
                 try:
                      p = Decimal(str(bid_price))
                      if p > 0: price = p; lg.warning(f"{NEON_YELLOW}Using 'bid' price fallback: {p}{RESET}")
                 except Exception: lg.warning(f"Invalid 'bid' price format: {bid_price}")

            if price is not None and price > 0:
                # Price formatting/quantization happens later during order placement/SL setting
                return price
            else:
                lg.warning(f"Failed to get a valid positive price from ticker attempt {attempts + 1}.")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting longer ({RETRY_DELAY_SECONDS*5}s)...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 4) # Wait 4*delay before the standard 1*delay below
            attempts += 1 # Count this attempt but skip standard delay
            continue
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Don't retry on most exchange errors (e.g., bad symbol)
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry on unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch a valid current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = DEFAULT_FETCH_LIMIT, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    try:
        if not exchange.has['fetchOHLCV']:
             lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
             return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
             try:
                  lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
                  # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

                  if ohlcv is not None and len(ohlcv) > 0: # Basic check if data was returned
                    # Check if the last timestamp seems recent (within a few intervals)
                    last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    interval_duration = pd.Timedelta(exchange.parse_timeframe(timeframe), unit='s')
                    if (now_utc - last_ts) < interval_duration * 5: # Allow some lag
                         lg.debug(f"Received {len(ohlcv)} klines. Last timestamp: {last_ts}")
                         break # Success
                    else:
                         lg.warning(f"Received {len(ohlcv)} klines, but last timestamp {last_ts} seems too old. Retrying...")
                  else:
                    lg.warning(f"fetch_ohlcv returned None or empty list for {symbol} (Attempt {attempt+1}). Retrying...")

             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                      raise e # Raise the final error to be caught by the main loop
             except ccxt.RateLimitExceeded as e:
                 lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {RETRY_DELAY_SECONDS * 5}s... (Attempt {attempt+1})")
                 time.sleep(RETRY_DELAY_SECONDS * 5)
             except ccxt.ExchangeError as e:
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 # Depending on the error, might not be retryable (e.g., bad symbol)
                 raise e # Re-raise non-network errors immediately
             except Exception as e:
                  lg.error(f"{NEON_RED}Unexpected error fetching klines: {e}{RESET}", exc_info=True)
                  raise e # Re-raise unexpected errors

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        # Ensure standard column names, handle potential missing 'turnover'
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if len(ohlcv[0]) > 6: columns.append('turnover') # Add turnover if present

        df = pd.DataFrame(ohlcv, columns=columns[:len(ohlcv[0])]) # Use only available columns

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric (Decimal for precision)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
        if 'turnover' in df.columns:
             df['turnover'] = df['turnover'].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))

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
        return df.copy() # Return a copy to avoid modifying cache outside function

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        # Errors raised after retries or non-retryable errors
        lg.error(f"{NEON_RED}Failed to fetch/process klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)

    return pd.DataFrame() # Return empty DataFrame on failure

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision, limits, contract type with retries."""
    lg = logger
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            # Check if markets are loaded, reload if necessary or if symbol not found
            if not exchange.markets or symbol not in exchange.markets:
                 lg.info(f"Market info for {symbol} not loaded or missing. Reloading markets (Attempt {attempt+1})...")
                 exchange.load_markets(reload=True)

            # Check again after reloading
            if symbol not in exchange.markets:
                 # If still not found after reload, it's likely an invalid symbol
                 if attempt == 0: continue # Allow one reload attempt
                 lg.error(f"{NEON_RED}Market {symbol} still not found after reloading.{RESET}")
                 return None

            market = exchange.market(symbol)
            if market:
                # Enhance market info with useful flags
                market_type = market.get('type', 'unknown') # spot, swap, future
                is_linear = market.get('linear', False)
                is_inverse = market.get('inverse', False)
                is_contract = market.get('contract', False) or market_type in ['swap', 'future']
                contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "Spot/Other"

                # Log retrieved info
                lg.debug(
                    f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract Type={contract_type}, "
                    f"Precision(Price/Amount): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}, "
                    f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                    f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                    f"Contract Size: {market.get('contractSize', 'N/A')}"
                )
                # Add derived flags to the dictionary for easier access
                market['is_contract'] = is_contract
                market['contract_type_str'] = contract_type # Add descriptive string
                return market # Success
            else:
                 # Should not happen if symbol is in exchange.markets, but handle defensively
                 lg.error(f"{NEON_RED}Market dictionary unexpectedly not found for {symbol} even though key exists.{RESET}")
                 return None # Treat as failure

        except ccxt.BadSymbol as e:
             lg.error(f"{NEON_RED}Symbol '{symbol}' not supported or invalid on {exchange.id}: {e}{RESET}")
             return None # Bad symbol is not retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
             if attempt < MAX_API_RETRIES:
                  lg.warning(f"Network error getting market info for {symbol} (Attempt {attempt+1}): {e}. Retrying...")
                  time.sleep(RETRY_DELAY_SECONDS)
             else:
                  lg.error(f"{NEON_RED}Max retries reached getting market info for {symbol} after network errors.{RESET}")
                  return None
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error getting market info for {symbol}: {e}{RESET}")
            # Some exchange errors might be temporary, allow retry
            if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
            else: return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
            return None # Unexpected errors usually not retryable

    return None # Should only be reached if all retries fail

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency, handling various Bybit structures."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info = None
            available_balance_str = None

            # Try fetching with common account types first (Bybit V5 preference)
            account_types_to_try = ['UNIFIED', 'CONTRACT'] # Unified often holds USDT for linear contracts too
            found_structure = False

            for acc_type in account_types_to_try:
                 try:
                     lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency}...")
                     # Note: Bybit V5 uses 'accountType' in params
                     balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                     lg.debug(f"Raw balance response (type {acc_type}): {balance_info}")

                     # --- Try extracting balance from various possible structures ---
                     # 1. Standard CCXT structure (less common for Bybit V5 derivatives)
                     if currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free'] structure: {available_balance_str}")
                         found_structure = True; break

                     # 2. Bybit V5 structure (often nested in 'info')
                     elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             # Unified account structure
                             if account.get('accountType') == acc_type and isinstance(account.get('coin'), list):
                                 for coin_data in account['coin']:
                                     if coin_data.get('coin') == currency:
                                         # Prefer 'availableToWithdraw', fallback to others
                                         free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                         if free is not None:
                                             available_balance_str = str(free)
                                             lg.debug(f"Found balance via V5 info.result.list[].coin[] ({acc_type}): {available_balance_str}")
                                             found_structure = True; break
                                 if found_structure: break
                         if found_structure: break
                         lg.debug(f"Currency '{currency}' not found within V5 info.result.list using type '{acc_type}'.")

                 except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                     lg.debug(f"API error fetching balance for type '{acc_type}': {e}. Trying next.")
                     continue # Try next account type
                 except Exception as e:
                     lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}. Trying next.")
                     continue

            # 3. Fallback: Try default fetch_balance (no params)
            if not found_structure:
                 lg.debug(f"Fetching balance using default parameters for {currency}...")
                 try:
                      balance_info = exchange.fetch_balance()
                      lg.debug(f"Raw balance response (default): {balance_info}")
                      # Check standard structure again with default response
                      if currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free'] (default fetch): {available_balance_str}")
                      # Check top-level 'free' dict (older CCXT versions?)
                      elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                         available_balance_str = str(balance_info['free'][currency])
                         lg.debug(f"Found balance via top-level 'free' dict (default fetch): {available_balance_str}")
                      # Check V5 structure again in default response
                      elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                           for account in balance_info['info']['result']['list']:
                              if isinstance(account.get('coin'), list):
                                  for coin_data in account['coin']:
                                      if coin_data.get('coin') == currency:
                                          free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                          if free is not None: available_balance_str = str(free); break
                                  if available_balance_str is not None: break
                              if available_balance_str is not None: break
                           if available_balance_str: lg.debug(f"Found balance via V5 nested structure (default fetch): {available_balance_str}")

                 except Exception as e:
                      lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                      # Allow retry loop to handle this

            # --- Process the extracted balance string ---
            if available_balance_str is not None:
                try:
                    final_balance = Decimal(available_balance_str)
                    if final_balance >= 0:
                         lg.info(f"Available {currency} balance: {final_balance:.4f}")
                         return final_balance # Success
                    else:
                         lg.error(f"Parsed balance for {currency} is negative ({final_balance}).")
                         # Treat negative balance as an issue, maybe retry
                except Exception as e:
                    lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
                    # Treat conversion failure as an issue, maybe retry
            else:
                # If still no balance string found after all attempts
                lg.error(f"{NEON_RED}Could not determine available balance for {currency} after checking known structures.{RESET}")
                lg.debug(f"Last balance_info structure checked: {balance_info}")
                # Allow retry loop

            # If we reached here, balance wasn't found or was invalid, retry if possible
            raise ccxt.ExchangeError("Balance not found or invalid in response") # Raise generic error to trigger retry


        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance for {currency}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting longer...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 4) # Add extra delay for rate limit
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API keys/permissions.{RESET}")
             return None # Auth errors are fatal
        except ccxt.ExchangeError as e:
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance for {currency}: {e}. Retrying...{RESET}")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error during balance fetch for {currency}: {e}{RESET}", exc_info=True)
            # Unexpected errors might be temporary, allow retry

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (attempts + 1)) # Increase delay slightly on retries
        else:
            lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            return None # Return None after all retries fail

    return None # Should not be reached, but satisfies static analysis

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions, handling Bybit V5."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for symbol: {symbol} (Attempt {attempts+1})")
            positions: List[Dict] = []
            # Bybit V5 prefers fetching specific symbol if possible
            try:
                 # Use params for V5 category if needed (linear default)
                 params = {'category': 'linear'} if 'bybit' in exchange.id.lower() else {}
                 positions = exchange.fetch_positions([symbol], params=params)
                 lg.debug(f"Fetched positions using fetch_positions([symbol]): {positions}")
            except ccxt.ArgumentsRequired:
                 # Fallback if exchange requires fetching all positions
                 lg.debug("Fetching all positions as exchange doesn't support single symbol fetch.")
                 all_positions = exchange.fetch_positions(params=params) # Add params here too
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total, found {len(positions)} matching {symbol}.")
            except ccxt.ExchangeError as e:
                 # Handle specific Bybit "position not found" errors gracefully
                 # Bybit V5 code for position not found: 110025 ("Position not found")
                 # Other potential indicators: "symbol: invalid" if symbol is wrong type?
                 no_pos_codes_v5 = [110025]
                 err_str = str(e).lower()
                 if (hasattr(e, 'code') and e.code in no_pos_codes_v5) or \
                    "position not found" in err_str or \
                    "no position found" in err_str:
                      lg.info(f"No position found for {symbol} (Exchange confirmed: {e}).")
                      return None # Confirmed no position
                 else:
                      lg.error(f"Exchange error fetching position for {symbol}: {e}", exc_info=False)
                      # Allow retry for other exchange errors
                      if attempts >= MAX_API_RETRIES: return None
                      raise e # Re-raise to trigger retry logic
            except Exception as e:
                 lg.error(f"Error fetching position for {symbol}: {e}", exc_info=True)
                 if attempts >= MAX_API_RETRIES: return None
                 raise e # Re-raise to trigger retry logic

            active_position = None
            # Define a small threshold to consider a position non-zero
            # Use market info if available for amount precision, else small default
            size_threshold = Decimal('1e-9')
            try:
                market = exchange.market(symbol)
                amount_precision = market.get('precision', {}).get('amount')
                if amount_precision: size_threshold = Decimal(str(amount_precision)) * Decimal('0.1') # 1/10th of smallest step
            except Exception: pass # Ignore errors getting market for threshold

            lg.debug(f"Using position size threshold: {size_threshold}")

            # Iterate through potentially multiple position entries (e.g., hedge mode)
            for pos in positions:
                # --- Determine Position Size ---
                pos_size_str = None
                # CCXT standard field
                if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
                # Bybit V5 often uses 'size' in 'info'
                elif pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size'])

                if pos_size_str is None:
                    lg.debug(f"Skipping position entry, could not determine size: {pos.get('info', {})}")
                    continue

                try:
                    position_size = Decimal(pos_size_str)
                    # Check if absolute size exceeds threshold
                    if abs(position_size) > size_threshold:
                        active_position = pos # Found a potential active position
                        lg.debug(f"Found potential active position entry for {symbol} with size {position_size}. Details: {pos.get('info', {})}")
                        break # Assume first non-zero position is the one we manage
                except Exception as parse_err:
                     lg.warning(f"Could not parse position size '{pos_size_str}': {parse_err}")
                     continue

            if active_position:
                # --- Standardize Key Position Details ---
                info_dict = active_position.get('info', {})

                # Size (use Decimal, prefer 'contracts' if available)
                size_str = str(active_position.get('contracts', info_dict.get('size', '0')))
                size_decimal = Decimal(size_str)

                # Side (derive if necessary)
                side = active_position.get('side') # 'long' or 'short'
                if side not in ['long', 'short']:
                    pos_side_v5 = info_dict.get('side', '').lower() # Bybit V5 'Buy'/'Sell'
                    if pos_side_v5 == 'buy': side = 'long'
                    elif pos_side_v5 == 'sell': side = 'short'
                    # Fallback derivation from size
                    elif size_decimal > size_threshold: side = 'long'
                    elif size_decimal < -size_threshold: side = 'short'
                    else: lg.warning(f"Position size {size_decimal} near zero or side ambiguous, cannot determine side."); return None # Cannot determine side
                    active_position['side'] = side # Store standardized side

                # Entry Price
                entry_price_str = active_position.get('entryPrice') or info_dict.get('avgPrice')
                active_position['entryPrice'] = entry_price_str # Ensure it's set

                # Leverage
                leverage_str = active_position.get('leverage') or info_dict.get('leverage')
                active_position['leverage'] = leverage_str

                # Liquidation Price
                liq_price_str = active_position.get('liquidationPrice') or info_dict.get('liqPrice')
                active_position['liquidationPrice'] = liq_price_str

                # Unrealized PnL
                pnl_str = active_position.get('unrealizedPnl') or info_dict.get('unrealisedPnl')
                active_position['unrealizedPnl'] = pnl_str

                # --- Protection Info (SL/TP/TSL from Bybit V5 info dict) ---
                # Note: CCXT might not parse these consistently across exchanges/versions
                sl_price_str = active_position.get('stopLossPrice') or info_dict.get('stopLoss')
                tp_price_str = active_position.get('takeProfitPrice') or info_dict.get('takeProfit')
                tsl_distance_str = info_dict.get('trailingStop') # Distance (value)
                tsl_activation_str = info_dict.get('activePrice') # Activation price for TSL

                # Store potentially missing protection info back into the main dict if found in info
                if sl_price_str and not active_position.get('stopLossPrice'): active_position['stopLossPrice'] = sl_price_str
                if tp_price_str and not active_position.get('takeProfitPrice'): active_position['takeProfitPrice'] = tp_price_str
                # Add TSL info if found
                active_position['trailingStopLoss'] = tsl_distance_str
                active_position['tslActivationPrice'] = tsl_activation_str


                # --- Log Formatted Position Info ---
                def format_log_val(val, precision=6, is_size=False):
                    if val is None or str(val).strip() == '' or str(val).lower() == 'nan': return 'N/A'
                    try:
                         d_val = Decimal(str(val))
                         # Ignore zero values unless it's size (where 0 might be valid temporarily)
                         if not is_size and d_val.is_zero(): return 'N/A'

                         market = exchange.market(symbol) # Requires market loaded
                         if is_size:
                             amount_prec = abs(Decimal(str(market['precision']['amount'])).normalize().as_tuple().exponent)
                             return f"{abs(d_val):.{amount_prec}f}" # Show absolute size
                         else:
                             price_prec = abs(Decimal(str(market['precision']['price'])).normalize().as_tuple().exponent)
                             return f"{d_val:.{price_prec}f}"
                    except Exception: return str(val) # Fallback to string if formatting fails

                entry_price = format_log_val(active_position.get('entryPrice'))
                contracts = format_log_val(size_decimal, is_size=True)
                liq_price = format_log_val(active_position.get('liquidationPrice'))
                leverage = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str and Decimal(str(leverage_str)) > 0 else 'N/A'
                pnl = format_log_val(active_position.get('unrealizedPnl'), 4)
                sl_price = format_log_val(active_position.get('stopLossPrice'))
                tp_price = format_log_val(active_position.get('takeProfitPrice'))
                tsl_dist = format_log_val(active_position.get('trailingStopLoss'))
                tsl_act = format_log_val(active_position.get('tslActivationPrice'))

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                            f"Lev={leverage}, PnL={pnl}, SL={sl_price}, TP={tp_price}, "
                            f"TSL(Dist/Act): {tsl_dist}/{tsl_act}")
                logger.debug(f"Full standardized position details for {symbol}: {active_position}")
                return active_position # Success
            else:
                logger.info(f"No active open position found for {symbol} (checked {len(positions)} entries).")
                return None # No non-zero position found

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching position for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching position: {e}. Waiting longer...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 4)
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching position: {e}. Stopping checks.{RESET}")
             return None # Fatal
        except ccxt.ExchangeError as e:
            # Retry logic handled within the try block for specific codes
            lg.warning(f"{NEON_YELLOW}Exchange error fetching position for {symbol}: {e}. Retrying...{RESET}")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
            # Don't retry unexpected errors immediately

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay on retries
        else:
            lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            return None

    return None # Should not be reached

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract).")
        return True # Considered success as no action needed
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage ({leverage}).")
        return False
    if not exchange.has.get('setLeverage'):
         lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage via CCXT.{RESET}")
         return False

    try:
        # Check current leverage first? (Optional, adds an API call)
        # current_position = get_open_position(exchange, symbol, lg)
        # current_leverage = current_position.get('leverage') if current_position else None
        # if current_leverage and int(float(current_leverage)) == leverage:
        #     lg.info(f"Leverage for {symbol} already set to {leverage}x.")
        #     return True

        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        # Bybit V5 requires setting buy and sell leverage separately
        if 'bybit' in exchange.id.lower():
             # Ensure leverage is passed as string for Bybit V5 params
             params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
             lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        # CCXT's set_leverage function signature: setLeverage(leverage, symbol=None, params={})
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # Verify leverage setting (optional, requires another API call)
        # time.sleep(1) # Short delay before verification
        # updated_pos = get_open_position(exchange, symbol, lg) # Fetch position again to check leverage
        # if updated_pos and updated_pos.get('leverage') and int(float(updated_pos['leverage'])) == leverage:
        #      lg.info(f"{NEON_GREEN}Leverage for {symbol} confirmed set to {leverage}x.{RESET}")
        #      return True
        # else:
        #      lg.warning(f"{NEON_YELLOW}Leverage for {symbol} set request sent, but confirmation failed or pending. Current: {updated_pos.get('leverage') if updated_pos else 'N/A'}{RESET}")
        #      return False # Treat confirmation failure as potential issue

        # If not verifying, assume success if no exception
        lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Bybit V5 Specific Error Codes for Leverage:
        # 110045: Leverage not modified (already set)
        # 110028 / 110009 / 110055: Position/order exists, margin mode conflict (Isolated vs Cross)
        # 110044: Exceed risk limit (leverage too high for position size tier)
        # 110013: Parameter error (leverage value invalid)
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")

        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True # Already set is considered success
        elif bybit_code in [110028, 110009, 110055] or "margin mode" in err_str or "position exists" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Cannot change leverage. Check Margin Mode (Isolated/Cross), open orders, or existing position.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed risk limit tier for current/potential position size. Check Bybit Risk Limits.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for {symbol}. Check allowed range on Bybit.{RESET}")
        # Add more specific error handling as needed
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False # Return False if any error occurred


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # e.g., 0.01 for 1%
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Needed for formatting
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size based on risk percentage, SL distance, balance, and market constraints.
    Handles linear/inverse contract nuances (simplified) and applies precision/limits.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('inverse', False)
    size_unit = "Contracts" if is_contract else base_currency # Unit for size display

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
         # Attempt to reload market info once?
         market_info = get_market_info(exchange, symbol, lg)
         if not market_info or 'limits' not in market_info or 'precision' not in market_info:
              lg.error(f"Position sizing failed ({symbol}): Still missing market info after reload.")
              return None

    try:
        # --- Core Calculation ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_price = abs(entry_price - initial_stop_loss_price)
        if sl_distance_price <= 0:
             lg.error(f"Position sizing failed ({symbol}): SL distance zero/negative ({sl_distance_price}).")
             return None

        # Get contract size (value of 1 contract in quote currency for linear, or base for inverse)
        contract_size_str = market_info.get('contractSize', '1') # Default to 1 if not specified
        try:
             contract_size = Decimal(str(contract_size_str))
             if contract_size <= 0: raise ValueError("Contract size must be positive")
        except Exception as e:
             lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}, using 1. Error: {e}")
             contract_size = Decimal('1')

        # Calculate size based on contract type
        calculated_size = Decimal('0')
        if not is_contract or market_info.get('linear', True):
             # Linear Contract or Spot: Risk = Size * SL_Distance_Price * ContractSize_Quote
             value_per_unit = sl_distance_price * contract_size # Risk per unit of 'amount' (contract or base currency)
             if value_per_unit > 0:
                 calculated_size = risk_amount_quote / value_per_unit
             else: lg.error(f"Sizing failed ({symbol}): Calculated value per unit is zero/negative."); return None
        else: # Inverse Contract: Risk = Size * ContractSize_Base * (1/SL_Price - 1/Entry_Price)
             lg.warning(f"Inverse contract detected ({symbol}). Using simplified sizing (may be less accurate).")
             # Simplified: Risk approx Size * ContractSize_Base * SL_Distance_Price / (Entry * SL)
             # Or using value per contract change: Risk = Size * ContractSize_Base / SL_Price - Size * ContractSize_Base / Entry_Price
             # Approximate risk per contract: ContractSize * (1/SL - 1/Entry)
             if entry_price > 0 and initial_stop_loss_price > 0:
                  risk_per_contract_quote = abs(contract_size / initial_stop_loss_price - contract_size / entry_price)
                  if risk_per_contract_quote > 0:
                      calculated_size = risk_amount_quote / risk_per_contract_quote
                  else: lg.error(f"Sizing failed ({symbol}): Inverse contract risk per contract is zero/negative."); return None
             else: lg.error(f"Sizing failed ({symbol}): Invalid entry/SL price for inverse contract calc."); return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_price}")
        lg.info(f"  ContractSize={contract_size}, Type={'Linear/Spot' if not is_inverse else 'Inverse'}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount') # This is the step size (e.g., 0.001)

        # Amount Limits (Min/Max Contracts or Base Currency)
        min_amount_str = amount_limits.get('min')
        max_amount_str = amount_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')

        # Cost Limits (Min/Max Quote Currency Value)
        min_cost_str = cost_limits.get('min')
        max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')

        adjusted_size = calculated_size
        # Apply Min/Max Amount Limits first
        if adjusted_size < min_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} < min amount {min_amount:.8f}. Adjusting to min amount.{RESET}")
             adjusted_size = min_amount
        elif adjusted_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} > max amount {max_amount:.8f}. Adjusting to max amount.{RESET}")
             adjusted_size = max_amount

        # Calculate Estimated Cost
        # Linear/Spot: Cost = Size * EntryPrice * ContractSize
        # Inverse: Cost = Size * ContractSize / EntryPrice (Cost is in Base currency value, convert to Quote if needed?)
        # For simplicity, assuming cost limits are in Quote currency.
        estimated_cost = Decimal('0')
        if not is_inverse:
            estimated_cost = adjusted_size * entry_price * contract_size
        else:
            # Inverse cost limit check is tricky. Assume limit applies to quote value equivalent?
            # Approximate Quote Cost = Size * ContractSize (value in Base)
            estimated_cost = adjusted_size * contract_size # Value in Base, needs conversion? Let's assume cost limits apply to this for now.
            lg.warning(f"Cost limit check for Inverse contract ({symbol}) is approximate (using Base value).")

        lg.debug(f"  Size after Amount Limits: {adjusted_size:.8f} {size_unit}")
        lg.debug(f"  Estimated Cost ({quote_currency if not is_inverse else base_currency}): {estimated_cost:.4f}")

        # Apply Min/Max Cost Limits
        cost_adjusted = False
        if min_cost > 0 and estimated_cost < min_cost :
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} below min cost {min_cost:.4f}. Attempting to increase size.{RESET}")
             # Calculate required size to meet min cost
             required_size_for_min_cost = None
             if not is_inverse and entry_price > 0 and contract_size > 0:
                 required_size_for_min_cost = min_cost / (entry_price * contract_size)
             elif is_inverse and contract_size > 0: # Approximate for inverse
                 required_size_for_min_cost = min_cost / contract_size
             # Add more accurate inverse cost calc if needed

             if required_size_for_min_cost is None: lg.error("Cannot calculate size for min cost."); return None
             lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
             if required_size_for_min_cost > max_amount: lg.error(f"{NEON_RED}Cannot meet min cost {min_cost:.4f} without exceeding max amount {max_amount:.8f}. Aborted.{RESET}"); return None
             # Ensure it doesn't fall below min amount after potential rounding later
             adjusted_size = max(min_amount, required_size_for_min_cost)
             cost_adjusted = True

        elif max_cost > 0 and estimated_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} exceeds max cost {max_cost:.4f}. Reducing size.{RESET}")
             # Calculate max size allowed by max cost
             adjusted_size_for_max_cost = None
             if not is_inverse and entry_price > 0 and contract_size > 0:
                 adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
             elif is_inverse and contract_size > 0: # Approximate for inverse
                 adjusted_size_for_max_cost = max_cost / contract_size

             if adjusted_size_for_max_cost is None: lg.error("Cannot calculate size for max cost."); return None
             lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f}")
             # Ensure it doesn't fall below min amount
             adjusted_size = max(min_amount, min(adjusted_size, adjusted_size_for_max_cost)) # Take the minimum of current adjusted and max cost allowed, but not less than min_amount
             cost_adjusted = True

        if cost_adjusted:
             lg.info(f"  Size after Cost Limits: {adjusted_size:.8f} {size_unit}")


        # --- Apply Amount Precision (Step Size) ---
        final_size = adjusted_size
        try:
            # Use ccxt's amount_to_precision which handles step sizes correctly
            # Use TRUNCATE (ROUND_DOWN equivalent) to ensure we don't exceed risk/cost limits due to rounding up
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision ({fmt_err}). Using manual rounding.{RESET}")
            # Manual rounding fallback (less reliable than ccxt method)
            amount_prec_step = precision.get('amount')
            if amount_prec_step is not None:
                 try:
                      step_size = Decimal(str(amount_prec_step))
                      if step_size > 0:
                           # Floor division to round down to the nearest step size
                           final_size = (adjusted_size // step_size) * step_size
                           lg.info(f"Applied manual amount step size ({step_size}): {adjusted_size:.8f} -> {final_size} {size_unit}")
                      else: raise ValueError("Step size must be positive")
                 except Exception as manual_err:
                      lg.warning(f"{NEON_YELLOW}Invalid amount precision value '{amount_prec_step}' ({manual_err}). Using size adjusted for limits only.{RESET}")
                      final_size = adjusted_size # Use limit-adjusted size without precision formatting
            else:
                 lg.warning(f"{NEON_YELLOW}Amount precision not defined. Using size adjusted for limits only.{RESET}")
                 final_size = adjusted_size


        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{NEON_RED}Position size became zero/negative ({final_size}) after adjustments. Aborted.{RESET}")
            return None
        if final_size < min_amount:
            # This can happen if min_amount itself is not a multiple of step size and rounding down pushes below it.
            lg.error(f"{NEON_RED}Final size {final_size} is below minimum amount {min_amount} after precision formatting. Aborted.{RESET}")
            # Alternative: could try rounding UP to min_amount if very close, but safer to abort.
            return None

        # Recalculate final cost and check against min_cost again (important after rounding down)
        final_cost = Decimal('0')
        if not is_inverse: final_cost = final_size * entry_price * contract_size
        else: final_cost = final_size * contract_size # Approximate cost check

        if min_cost > 0 and final_cost < min_cost:
             # Check if the *next* step up would satisfy min cost
             try:
                 step_size = Decimal(str(precision.get('amount')))
                 next_step_size = final_size + step_size
                 next_step_cost = (next_step_size * entry_price * contract_size) if not is_inverse else (next_step_size * contract_size)

                 if next_step_cost >= min_cost and next_step_size <= max_amount:
                      lg.warning(f"{NEON_YELLOW}Final size {final_size} cost {final_cost:.4f} < min cost {min_cost:.4f}. Bumping to next step size {next_step_size} to meet min cost.{RESET}")
                      final_size = next_step_size
                 else:
                      lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} below minimum cost {min_cost:.4f}, and next step size is invalid/too large. Aborted.{RESET}")
                      return None
             except Exception:
                  lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} below minimum cost {min_cost:.4f}. Aborted (failed step check).{RESET}")
                  return None


        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None

def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal,
    market_info: Dict,
    logger: Optional[logging.Logger] = None,
    reduce_only: bool = False
) -> Optional[Dict]:
    """
    Places a market order using CCXT. Returns the order dictionary on success, None on failure.
    Uses reduce_only flag for closing orders. Handles Bybit V5 params.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')
    action = "Close" if reduce_only else "Open/Increase"

    try:
        # Convert size to float for CCXT, ensure it's positive
        amount_float = float(position_size)
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} {action}): Invalid position size ({amount_float}). Size must be positive.")
            return None
    except Exception as e:
        lg.error(f"Trade aborted ({symbol} {side} {action}): Failed to convert size {position_size} to float: {e}")
        return None

    # --- Prepare Parameters ---
    params = {
        # Bybit V5: positionIdx=0 for one-way mode (assumed default)
        # 1 for Buy Hedge, 2 for Sell Hedge if using Hedge Mode
        'positionIdx': 0,
        'reduceOnly': reduce_only,
    }
    if reduce_only:
        # Use IOC for reduceOnly market orders to prevent accidental opening if position closed unexpectedly
        params['timeInForce'] = 'IOC' # Immediate Or Cancel
        lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")
    else:
        lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")

    lg.info(f"  Size: {amount_float:.8f} {size_unit}")
    lg.debug(f"  Params: {params}")

    try:
        # Use create_order: create_order(symbol, type, side, amount, price=None, params={})
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None, # Market order doesn't need price
            params=params
        )
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A') # e.g., 'open', 'closed', 'canceled'
        lg.info(f"{NEON_GREEN}{action} Trade Placed Successfully! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} {action}): {order}")
        return order

    # --- Specific Error Handling ---
    except ccxt.InsufficientFunds as e:
         lg.error(f"{NEON_RED}Insufficient funds to place {action} {side} order ({symbol}): {e}{RESET}")
    except ccxt.InvalidOrder as e:
        # Includes issues like size precision, limits, parameter errors
        lg.error(f"{NEON_RED}Invalid order parameters placing {action} {side} order ({symbol}): {e}{RESET}")
        # Bybit V5 Specific Codes:
        # 110007: Quantity invalid (precision/limits)
        # 110013: Parameter error
        # 110014: Reduce-only order failed (size mismatch, no position)
        bybit_code = getattr(e, 'code', None)
        if reduce_only and bybit_code == 110014:
             lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only order failed. Position might already be closed, size incorrect, or API issue? Check position status manually.{RESET}")
        elif "order quantity" in str(e).lower() or bybit_code == 110007:
             lg.error(f"{NEON_YELLOW} >> Hint: Check order size ({amount_float}) against market precision and limits (min/max amount).{RESET}")
    except ccxt.NetworkError as e:
         lg.error(f"{NEON_RED}Network error placing {action} order ({symbol}): {e}{RESET}")
         # Network errors might warrant a retry at a higher level or manual check
    except ccxt.ExchangeError as e:
        # Generic exchange errors, check Bybit codes
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action} order ({symbol}): {e} (Code: {bybit_code}){RESET}")
        # Bybit V5 Specific Codes:
        # 110025: Position not found (relevant for reduce_only)
        # 110043: Set margin mode failed (can happen during order if mode conflicts)
        if reduce_only and bybit_code == 110025:
             lg.warning(f"{NEON_YELLOW} >> Hint (110025): Position might have been closed already when trying to place reduce-only order.{RESET}")
        # Add more known error codes as needed
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing {action} order ({symbol}): {e}{RESET}", exc_info=True)

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
    Internal helper to set SL, TP, or TSL for an existing position using Bybit's V5 API endpoint.
    Handles parameter formatting, validation, and API call.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract).")
        return False # Cannot set protection on non-contracts
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    pos_side = position_info.get('side') # Should be 'long' or 'short'
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
         lg.error(f"Cannot set protection for {symbol}: Invalid position side ('{pos_side}') or missing entry price.")
         return False
    try: entry_price = Decimal(str(entry_price_str))
    except Exception: lg.error(f"Invalid entry price format for protection check: {entry_price_str}"); return False

    # --- Validate and Format Parameters ---
    params_to_set = {}
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} @ {entry_price}):"]
    params_valid = False

    try:
        # Get price precision (tick size) for formatting
        price_prec_str = market_info.get('precision', {}).get('price', '0.0001') # Default if missing
        min_tick_size = Decimal(str(price_prec_str))
        if min_tick_size <= 0: raise ValueError(f"Invalid tick size: {min_tick_size}")
        price_precision_places = abs(min_tick_size.normalize().as_tuple().exponent)

        # Helper to format price according to market precision
        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try:
                # Use exchange.price_to_precision for rounding based on tick size
                formatted = exchange.price_to_precision(symbol, float(price_decimal))
                # Ensure it's positive after formatting
                if Decimal(formatted) <= 0: return None
                return formatted
            except Exception as e:
                 lg.warning(f"Failed to format price {price_decimal} using exchange precision: {e}. Using manual quantize.")
                 # Manual fallback quantization
                 quantized = price_decimal.quantize(min_tick_size, rounding=ROUND_DOWN if pos_side == 'long' else ROUND_UP) # Round defensively
                 if quantized <= 0: return None
                 return f"{quantized:.{price_precision_places}f}" # Format to correct places

        # --- Trailing Stop ---
        # TSL requires both distance > 0 AND activation price > 0
        # TSL overrides fixed SL on Bybit V5 when set
        if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and \
           isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0:

            # Validate TSL Activation Price relative to entry
            if (pos_side == 'long' and tsl_activation_price <= entry_price) or \
               (pos_side == 'short' and tsl_activation_price >= entry_price):
                lg.error(f"{NEON_RED}TSL Activation Price ({tsl_activation_price}) is not beyond entry price ({entry_price}) for {pos_side} position. Cannot set TSL.{RESET}")
            else:
                # Format TSL distance (treated like a price value for precision)
                # Ensure distance is at least one tick size
                tsl_dist_quantized = max(trailing_stop_distance, min_tick_size)
                formatted_tsl_distance = exchange.decimal_to_precision(
                    tsl_dist_quantized, exchange.ROUND, precision=price_precision_places, padding_mode=exchange.NO_PADDING
                )
                if Decimal(formatted_tsl_distance) <= 0: # Double check after formatting
                      lg.error(f"Formatted TSL distance ({formatted_tsl_distance}) invalid.")
                      formatted_tsl_distance = None

                # Format TSL activation price
                formatted_activation_price = format_price(tsl_activation_price)

                if formatted_tsl_distance and formatted_activation_price:
                    params_to_set['trailingStop'] = formatted_tsl_distance
                    params_to_set['activePrice'] = formatted_activation_price
                    log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                    params_valid = True
                    stop_loss_price = None # Explicitly clear fixed SL if TSL is being set
                    log_parts.append(f"  (Fixed SL will be removed/ignored as TSL is active)")
                else:
                     lg.error(f"Failed to format valid TSL parameters (Dist: {formatted_tsl_distance}, Act: {formatted_activation_price}).")

        # --- Fixed Stop Loss (Only if TSL is not being set) ---
        if stop_loss_price is not None and 'trailingStop' not in params_to_set:
            # Validate SL price relative to entry
            if (pos_side == 'long' and stop_loss_price >= entry_price) or \
               (pos_side == 'short' and stop_loss_price <= entry_price):
                lg.error(f"{NEON_RED}Stop Loss Price ({stop_loss_price}) is not beyond entry price ({entry_price}) for {pos_side} position. Cannot set SL.{RESET}")
            else:
                formatted_sl = format_price(stop_loss_price)
                if formatted_sl:
                    params_to_set['stopLoss'] = formatted_sl
                    log_parts.append(f"  Fixed SL: {formatted_sl}")
                    params_valid = True
                else: lg.error(f"Failed to format valid SL price: {stop_loss_price}")

        # --- Fixed Take Profit ---
        if take_profit_price is not None:
             # Validate TP price relative to entry
            if (pos_side == 'long' and take_profit_price <= entry_price) or \
               (pos_side == 'short' and take_profit_price >= entry_price):
                lg.error(f"{NEON_RED}Take Profit Price ({take_profit_price}) is not beyond entry price ({entry_price}) for {pos_side} position. Cannot set TP.{RESET}")
            else:
                formatted_tp = format_price(take_profit_price)
                if formatted_tp:
                    params_to_set['takeProfit'] = formatted_tp
                    log_parts.append(f"  Fixed TP: {formatted_tp}")
                    params_valid = True
                else: lg.error(f"Failed to format valid TP price: {take_profit_price}")

    except Exception as fmt_err:
         lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    # If no valid parameters were formatted, don't make API call
    if not params_valid or not params_to_set:
        lg.warning(f"No valid protection parameters to set for {symbol} after validation/formatting. No API call made.")
        # Consider returning True if the intent was to clear protection and params are empty?
        # For now, return False as the requested action wasn't fully formed.
        return False

    # --- Prepare API Call ---
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    position_idx = 0 # Assume one-way mode
    try: # Try getting positionIdx from info if available (hedge mode)
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
    except Exception: lg.warning(f"Could not parse positionIdx from position info, using default {position_idx}.")

    # Construct final parameters for Bybit V5 endpoint
    final_api_params = {
        'category': category,
        'symbol': market_info['id'], # Use exchange-specific ID
        'tpslMode': 'Full', # Options: 'Full' or 'Partial' (default is Full)
        'slTriggerBy': 'LastPrice', # Options: MarkPrice, IndexPrice, LastPrice
        'tpTriggerBy': 'LastPrice',
        'slOrderType': 'Market', # Options: Market, Limit
        'tpOrderType': 'Market',
        'positionIdx': position_idx
    }
    # Add the validated & formatted SL/TP/TSL params
    final_api_params.update(params_to_set)

    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={final_api_params})")

    # --- Execute API Call ---
    try:
        response = exchange.private_post('/v5/position/set-trading-stop', params=final_api_params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        # --- Process Response ---
        # Bybit V5 response structure: {retCode: 0, retMsg: "OK", result: {}, ...}
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {}) # Contains additional details on failure

        if ret_code == 0:
            # Check for specific success messages that indicate no change was needed
            if "tpsl order cost not modified" in ret_msg.lower() or \
               "take profit is not modified" in ret_msg.lower() or \
               "stop loss is not modified" in ret_msg.lower() or \
               "trailing stop is not modified" in ret_msg.lower():
                 lg.info(f"{NEON_YELLOW}Position protection already set to target values or no change needed for {symbol}. Response: {ret_msg}{RESET}")
            else:
                 lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            return True # Success
        else:
            # Log specific Bybit error codes and messages
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            # Provide hints for common errors
            if ret_code == 110013 and "parameter error" in ret_msg.lower():
                lg.error(f"{NEON_YELLOW} >> Hint (110013): Check SL/TP prices vs entry, TSL dist/act prices, compliance with tick size/price limits.{RESET}")
            elif ret_code == 110036:
                lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price likely invalid (already passed, wrong side, too close to current price?).{RESET}")
            elif ret_code == 110086:
                lg.error(f"{NEON_YELLOW} >> Hint (110086): SL price cannot equal TP price.{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower():
                lg.error(f"{NEON_YELLOW} >> Hint: TSL distance invalid (too small/large, violates tick size?). Check required min distance on Bybit.{RESET}")
            elif "stop loss price is invalid" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: SL price invalid (too close to market price, outside limits?). Check required distance from market price.{RESET}")
            elif "take profit price is invalid" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: TP price invalid (too close to market price, outside limits?).{RESET}")
            # Add more hints based on observed errors
            return False

    except ccxt.ExchangeError as e:
        # Handle potential CCXT-level errors during the raw API call
        lg.error(f"{NEON_RED}CCXT ExchangeError during protection API call for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during protection API call for {symbol}: {e}{RESET}", exc_info=True)

    return False # Return False if API call failed


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
    Calculates Trailing Stop Loss parameters based on config and current position,
    then calls the internal helper `_set_position_protection` to set TSL (and optionally TP).

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol string.
        market_info: Market dictionary from `get_market_info`.
        position_info: Position dictionary from `get_open_position`.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional fixed TP target (Decimal).

    Returns:
        True if TSL (and TP) were successfully requested/set, False otherwise.
    """
    lg = logger
    protection_cfg = config.get("protection", {})
    if not protection_cfg.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss disabled in config for {symbol}. Skipping TSL setup.")
        return False # Not an error, just disabled

    # --- Validate Inputs ---
    if not market_info or not position_info:
        lg.error(f"Cannot calculate TSL for {symbol}: Missing market or position info.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"{NEON_RED}Missing required position info (side, entryPrice) for TSL calc ({symbol}).{RESET}")
        return False

    try:
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(protection_cfg.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(protection_cfg.get("trailing_stop_activation_percentage", 0.003)))

        if entry_price <= 0: raise ValueError("Entry price must be positive")
        if callback_rate <= 0: raise ValueError("Callback rate must be positive")
        if activation_percentage < 0: raise ValueError("Activation percentage cannot be negative")

        # Get price precision (tick size)
        price_prec_str = market_info.get('precision', {}).get('price', '0.0001')
        min_tick_size = Decimal(str(price_prec_str))
        if min_tick_size <= 0: raise ValueError(f"Invalid tick size: {min_tick_size}")
        price_precision_places = abs(min_tick_size.normalize().as_tuple().exponent)

    except Exception as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format or position info ({symbol}): {e}. Cannot calculate TSL.{RESET}")
        return False

    try:
        # --- Calculate Activation Price ---
        activation_price = None
        activation_offset = entry_price * activation_percentage

        # Activation price must be strictly beyond entry price by at least one tick
        if pos_side == 'long':
            raw_activation = entry_price + activation_offset
            # Round UP to the nearest tick, ensuring it's > entry_price
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
            if activation_price <= entry_price:
                 activation_price = entry_price + min_tick_size # Ensure it's at least one tick away
        else: # short
            raw_activation = entry_price - activation_offset
            # Round DOWN to the nearest tick, ensuring it's < entry_price
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
            if activation_price >= entry_price:
                 activation_price = entry_price - min_tick_size # Ensure it's at least one tick away

        if activation_price <= 0:
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) is zero/negative for {symbol}.{RESET}")
             return False

        # --- Calculate Trailing Stop Distance ---
        # Distance is based on activation price * callback rate
        # Note: Some interpretations base distance on entry price. Using activation price seems more common for Bybit.
        trailing_distance_raw = activation_price * callback_rate
        # Distance must be positive and generally rounded UP to the nearest tick size increment
        # It also might have a minimum required value by the exchange (check Bybit docs if issues occur)
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        # Ensure minimum distance is at least one tick
        if trailing_distance < min_tick_size:
             trailing_distance = min_tick_size
             lg.debug(f"TSL distance bumped to minimum tick size: {min_tick_size}")

        if trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance zero/negative ({trailing_distance}) for {symbol}.{RESET}")
             return False

        # --- Log Calculated Parameters ---
        lg.info(f"Calculated TSL Params for {symbol} ({pos_side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision_places}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price: {activation_price:.{price_precision_places}f}")
        lg.info(f"  => Trailing Distance: {trailing_distance:.{price_precision_places}f}")
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0:
             # Format TP for logging
             tp_log_str = f"{take_profit_price:.{price_precision_places}f}"
             try: tp_log_str = exchange.price_to_precision(symbol, float(take_profit_price))
             except: pass
             lg.info(f"  Take Profit Price: {tp_log_str} (Will be set alongside TSL)")
        else:
             take_profit_price = None # Ensure it's None if invalid


        # --- Call Helper to Set Protection ---
        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
            stop_loss_price=None, # TSL overrides fixed SL
            take_profit_price=take_profit_price, # Pass optional fixed TP
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy ---

class OrderBlock(TypedDict):
    """Represents a detected Order Block."""
    id: str # Unique ID (e.g., type_timestamp)
    type: str # 'bull' or 'bear'
    left_idx: pd.Timestamp # DatetimeIndex of the bar where OB formed (the pivot bar)
    right_idx: pd.Timestamp # DatetimeIndex of the last bar OB is considered valid for (or violated)
    top: Decimal # Top price level of the OB (use Decimal)
    bottom: Decimal # Bottom price level of the OB
    active: bool # Still considered valid (not violated)?
    violated: bool # Has price closed beyond the OB?

class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis."""
    dataframe: pd.DataFrame # DataFrame with all indicator calculations
    last_close: Decimal # Latest close price as Decimal
    current_trend_up: Optional[bool] # True=UP, False=DOWN, None=Undetermined/InsufficientData
    trend_just_changed: bool # True if trend changed on the last completed bar
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    vol_norm_int: Optional[int] # Latest Volume Norm (0-200) as int, or None
    atr: Optional[Decimal] # Latest ATR as Decimal, or None
    upper_band: Optional[Decimal] # Latest upper band as Decimal, or None
    lower_band: Optional[Decimal] # Latest lower band as Decimal, or None

class VolumaticOBStrategy:
    """
    Calculates Volumatic Trend and Pivot Order Blocks based on Pine Script logic interpretation.
    Manages OB state and generates analysis results including trend state and active OBs.
    """

    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        """Initializes the strategy engine with configuration parameters."""
        self.config = config
        self.market_info = market_info # Store market info for reference (e.g., interval for pivot timing)
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})

        # --- Store Config Params ---
        self.vt_length = strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH)
        self.vt_atr_period = strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD)
        self.vt_vol_ema_length = strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH)
        self.vt_atr_multiplier = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))
        self.vt_step_atr_multiplier = Decimal(str(strategy_cfg.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER)))

        self.ob_source = strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE)
        self.ph_left = strategy_cfg.get("ph_left", DEFAULT_PH_LEFT)
        self.ph_right = strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT)
        self.pl_left = strategy_cfg.get("pl_left", DEFAULT_PL_LEFT)
        self.pl_right = strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT)
        self.ob_extend = strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND)
        self.ob_max_boxes = strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES)

        # --- State Variables ---
        # Store boxes persistently between updates
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Minimum data length required for all indicators
        # Based on longest lookback periods used in calculations
        self.min_data_len = max(
             self.vt_length * 2, # Allow buffer for EMA/SWMA stabilization
             self.vt_atr_period,
             self.vt_vol_ema_length,
             self.ph_left + self.ph_right + 1, # Pivot calculation needs left+right+pivot bar
             self.pl_left + self.pl_right + 1
         ) + 10 # Add a safety buffer

        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy Engine...{RESET}")
        self.logger.info(f"  VT Params: Len={self.vt_length}, ATRLen={self.vt_atr_period}, VolLen={self.vt_vol_ema_length}, ATRMult={self.vt_atr_multiplier}, StepMult={self.vt_step_atr_multiplier}")
        self.logger.info(f"  OB Params: Src={self.ob_source}, PH={self.ph_left}/{self.ph_right}, PL={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxBoxes={self.ob_max_boxes}")
        self.logger.info(f"  Minimum data points recommended: {self.min_data_len}")


    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates the custom SWMA (Symmetrically Weighted Moving Average) then EMA,
        attempting to replicate the specific Pine Script logic.
        SWMA uses weights [1, 2, 2, 1] / 6 over 4 periods.
        """
        if len(series) < 4:
            return pd.Series(np.nan, index=series.index, dtype=float) # Use float for numpy compatibility

        # Pine Script's swma(src) seems to be a specific 4-period WMA variant.
        # Weights [1, 2, 2, 1] / 6 applied to [x[3], x[2], x[1], x[0]] where x[0] is current bar.
        weights = np.array([1, 2, 2, 1]) / 6.0

        # Use rolling apply with explicit weights. Pandas rolling window gives [t-3, t-2, t-1, t].
        # We need to dot product with weights reversed: [w3, w2, w1, w0] -> [1, 2, 2, 1]/6.
        swma = series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate the EMA of the SWMA series using pandas_ta
        # Ensure the input series to ema is numeric (float)
        swma_numeric = pd.to_numeric(swma, errors='coerce')
        ema_of_swma = ta.ema(swma_numeric, length=length, fillna=np.nan)

        return ema_of_swma

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Calculates all strategy components based on the input DataFrame.
        Processes data, updates OB state, and returns structured results.

        Args:
            df_input: pandas DataFrame with Decimal columns ['open', 'high', 'low', 'close', 'volume']
                      index must be DatetimeIndex, sorted chronologically. Should include
                      enough historical data as required by min_data_len.

        Returns:
            StrategyAnalysisResults dictionary containing latest state and active boxes.
            Returns a partial/default dictionary if analysis fails or data is insufficient.
        """
        # --- Pre-computation Checks ---
        if df_input.empty:
             self.logger.error("Strategy update received empty DataFrame.")
             # Return default empty results
             return StrategyAnalysisResults(dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
                                            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
                                            vol_norm_int=None, atr=None, upper_band=None, lower_band=None)

        # Work on a copy to avoid modifying the original DataFrame from outside
        df = df_input.copy()

        # Ensure data is sorted and index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             self.logger.error("DataFrame index is not DatetimeIndex. Attempting conversion.")
             try: df.index = pd.to_datetime(df.index, utc=True)
             except Exception: self.logger.error("Failed to convert index to DatetimeIndex."); # Proceed with partial results
        if not df.index.is_monotonic_increasing:
             self.logger.warning("DataFrame index is not monotonically increasing. Sorting.")
             df.sort_index(inplace=True)

        # Ensure data is long enough
        if len(df) < self.min_data_len:
            self.logger.warning(f"{NEON_YELLOW}Not enough data ({len(df)}/{self.min_data_len}) for full strategy analysis. Results may be inaccurate or incomplete.{RESET}")
            # Proceed with calculation, but results might have NaNs or be less reliable

        self.logger.debug(f"Analyzing {len(df)} candles for strategy.")

        # --- Convert Decimal columns to float for pandas_ta compatibility ---
        # Keep original Decimal columns for precision where needed later
        try:
            df_float = df.copy()
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                if col in df_float.columns:
                     # Use .astype(float) for direct conversion, handles NaN
                     df_float[col] = df_float[col].astype(float)
        except Exception as e:
            self.logger.error(f"Error converting DataFrame columns to float for TA: {e}")
            # Return partial results if conversion fails
            last_close_dec = df['close'].iloc[-1] if not df.empty and pd.notna(df['close'].iloc[-1]) else Decimal('0')
            return StrategyAnalysisResults(dataframe=df, last_close=last_close_dec, current_trend_up=None, trend_just_changed=False,
                                           active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None)


        # --- Volumatic Trend Calculations (using df_float for TA) ---
        df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)
        df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length) # Custom EMA/SWMA
        df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan) # Standard EMA

        # Trend Detection: Pine: `UpTrend = ema1[1] < ema2` -> Python: `ema1.shift(1) < ema2`
        df_float['trend_up'] = (df_float['ema1'].shift(1) < df_float['ema2'])
        # Forward fill the boolean trend to handle initial NaNs after shift
        df_float['trend_up'] = df_float['trend_up'].ffill() # Now contains True/False where trend is determined

        # Detect trend change points
        df_float['trend_changed'] = (df_float['trend_up'] != df_float['trend_up'].shift(1)) & \
                                    df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
        df_float['trend_changed'].fillna(False, inplace=True) # First valid trend point is not a 'change'

        # --- Stateful Band Calculation (using df_float) ---
        # Bands are reset based on EMA1 and ATR *at the point of trend change*.
        df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
        df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

        # Forward fill the values from the last change point
        df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
        df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

        # Calculate bands using the ffilled values and Decimal multipliers
        df_float['upper_band'] = df_float['ema1_for_bands'] + df_float['atr_for_bands'] * float(self.vt_atr_multiplier)
        df_float['lower_band'] = df_float['ema1_for_bands'] - df_float['atr_for_bands'] * float(self.vt_atr_multiplier)

        # --- Volume Calculation & Normalization (using df_float) ---
        # Calculate rolling max volume over the lookback period
        df_float['vol_max'] = df_float['volume'].rolling(
            window=self.vt_vol_ema_length,
            min_periods=max(1, self.vt_vol_ema_length // 10) # Require some minimum periods
        ).max()

        # Normalize volume (0-100 range relative to recent max)
        # Avoid division by zero or NaN denominator
        df_float['vol_norm'] = np.where(
            df_float['vol_max'].notna() & (df_float['vol_max'] > 1e-9), # Check max > 0
            (df_float['volume'] / df_float['vol_max'] * 100),
            0 # Default to 0 if max is NaN or zero
        )
        df_float['vol_norm'].fillna(0, inplace=True) # Treat NaN volume norm as 0
        # Clip extreme values (e.g., if current volume massively exceeds historical max)
        df_float['vol_norm'] = df_float['vol_norm'].clip(0, 200)

        # Calculate volume step reference levels (potentially for visualization or other logic)
        df_float['lower_vol_ref'] = df_float['lower_band'] + df_float['atr_for_bands'] * float(self.vt_step_atr_multiplier)
        df_float['upper_vol_ref'] = df_float['upper_band'] - df_float['atr_for_bands'] * float(self.vt_step_atr_multiplier)

        # Calculate step size per 1% of normalized volume
        df_float['step_up_size'] = np.where(
            df_float['vol_norm'] > 1e-9, # Avoid division by zero if norm is zero
             (df_float['lower_vol_ref'] - df_float['lower_band']) / df_float['vol_norm'], # Original had /100, seems redundant if norm is 0-100
             0
        ) * (df_float['vol_norm'] / 100) # Apply scaling based on vol_norm

        df_float['step_dn_size'] = np.where(
            df_float['vol_norm'] > 1e-9,
             (df_float['upper_band'] - df_float['upper_vol_ref']) / df_float['vol_norm'],
             0
        ) * (df_float['vol_norm'] / 100)

        df_float['step_up_size'] = df_float['step_up_size'].clip(lower=0).fillna(0)
        df_float['step_dn_size'] = df_float['step_dn_size'].clip(lower=0).fillna(0)

        # Calculate top/bottom of volume bars (relative to bands)
        df_float['vol_bar_up_top'] = df_float['lower_band'] + df_float['step_up_size'] # Simpler: offset = step_size * norm
        df_float['vol_bar_dn_bottom'] = df_float['upper_band'] - df_float['step_dn_size']


        # --- Copy calculated float columns back to the main Decimal DataFrame ---
        # Keep precision where it matters (bands, atr) by converting back
        cols_to_copy = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed',
                        'upper_band', 'lower_band', 'vol_norm',
                        'vol_bar_up_top', 'vol_bar_dn_bottom']
        for col in cols_to_copy:
            if col in df_float.columns:
                if df_float[col].dtype == 'bool' or df_float[col].dtype == 'object': # Handle booleans/objects separately
                     df[col] = df_float[col]
                else: # Convert numeric back to Decimal
                     df[col] = df_float[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))

        # Drop rows where essential calculations (like bands) are still NaN
        # This happens before the first trend change or if ATR is NaN initially
        initial_len_before_drop = len(df)
        df.dropna(subset=['upper_band', 'lower_band', 'atr', 'trend_up'], inplace=True)
        rows_dropped = initial_len_before_drop - len(df)
        if rows_dropped > 0:
             self.logger.debug(f"Dropped {rows_dropped} initial rows lacking necessary indicator values.")

        if df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame empty after calculating indicators and dropping NaNs. Insufficient data or no trend established yet.{RESET}")
            # Return partial empty results
            return StrategyAnalysisResults(dataframe=df, last_close=Decimal('0'), current_trend_up=None, trend_just_changed=False,
                                           active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None)

        self.logger.debug("Volumatic Trend calculations complete.")


        # --- Pivot Order Block Calculations & Management ---
        # Use the float DataFrame for pandas_ta pivot calculation
        # Define source series based on config
        if self.ob_source == "Wicks":
            high_series = df_float['high']
            low_series = df_float['low']
        else: # "Bodys"
            high_series = df_float[['open', 'close']].max(axis=1)
            low_series = df_float[['open', 'close']].min(axis=1)

        # `ta.pivot` returns 1/True at index `i` if pivot is CONFIRMED at `i`.
        # The actual pivot high/low bar occurred `right` bars BEFORE index `i`.
        df_float['ph_signal'] = ta.pivot(high_series, left=self.ph_left, right=self.ph_right, high_low='high').fillna(0).astype(bool)
        df_float['pl_signal'] = ta.pivot(low_series, left=self.pl_left, right=self.pl_right, high_low='low').fillna(0).astype(bool)

        # --- Identify NEW Pivots confirmed in the latest data ---
        # We only need to check the last few bars where a pivot could have been confirmed.
        # A pivot confirmed at index `i` relates to a candle at `i - right_bars`.
        check_recent_bars = max(self.ph_right, self.pl_right) + 5 # Check enough bars back for confirmation lag + buffer
        recent_indices = df.index[-check_recent_bars:] # Use index from the main Decimal DF

        new_boxes_found = False
        for idx in recent_indices:
            if idx not in df_float.index: continue # Skip if index missing in float df (e.g., due to NaN drop)

            # --- Check for New Bearish OB (from Pivot High) ---
            if df_float.loc[idx, 'ph_signal']:
                # Pivot HIGH confirmed at `idx`, the actual pivot candle is `ph_right` bars before.
                pivot_bar_loc_in_float = df_float.index.get_loc(idx) - self.ph_right
                if pivot_bar_loc_in_float >= 0:
                    pivot_bar_idx = df_float.index[pivot_bar_loc_in_float] # Timestamp of the pivot bar

                    # Check if this pivot bar index exists in the main Decimal DF
                    if pivot_bar_idx in df.index:
                        # Check if a bear box starting at this exact timestamp already exists
                        if not any(b['left_idx'] == pivot_bar_idx for b in self.bear_boxes):
                            # Get the actual OB candle data from the Decimal DataFrame
                            ob_candle = df.loc[pivot_bar_idx]
                            box_top, box_bottom = Decimal('NaN'), Decimal('NaN')

                            # Define OB range based on source (using standard definitions)
                            # Bearish OB: Last up-move candle before down-move. Typically uses High and Open.
                            if self.ob_source == "Wicks":
                                box_top = ob_candle['high']
                                box_bottom = ob_candle['open'] # Base of the wick OB is often the open
                            else: # "Bodys"
                                box_top = ob_candle['close'] # Body top (assuming up candle)
                                box_bottom = ob_candle['open'] # Body bottom

                            # Ensure top > bottom, swap if candle was actually down
                            if pd.notna(box_top) and pd.notna(box_bottom) and box_bottom > box_top:
                                box_top, box_bottom = box_bottom, box_top

                            if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                                self.bear_boxes.append({
                                    'id': f"bear_{pivot_bar_idx.strftime('%Y%m%d%H%M%S')}", # More unique ID
                                    'type': 'bear',
                                    'left_idx': pivot_bar_idx,
                                    'right_idx': df.index[-1], # Extend to current bar initially
                                    'top': box_top,
                                    'bottom': box_bottom,
                                    'active': True,
                                    'violated': False
                                })
                                self.logger.debug(f"{NEON_RED}New Bearish OB created at {pivot_bar_idx} [{box_bottom:.4f} - {box_top:.4f}]{RESET}")
                                new_boxes_found = True

            # --- Check for New Bullish OB (from Pivot Low) ---
            if df_float.loc[idx, 'pl_signal']:
                # Pivot LOW confirmed at `idx`, the actual pivot candle is `pl_right` bars before.
                pivot_bar_loc_in_float = df_float.index.get_loc(idx) - self.pl_right
                if pivot_bar_loc_in_float >= 0:
                    pivot_bar_idx = df_float.index[pivot_bar_loc_in_float]

                    if pivot_bar_idx in df.index:
                        if not any(b['left_idx'] == pivot_bar_idx for b in self.bull_boxes):
                            ob_candle = df.loc[pivot_bar_idx]
                            box_top, box_bottom = Decimal('NaN'), Decimal('NaN')

                            # Define OB range based on source (using standard definitions)
                            # Bullish OB: Last down-move candle before up-move. Typically uses Open and Low.
                            if self.ob_source == "Wicks":
                                box_top = ob_candle['open'] # Top of the wick OB is often the open
                                box_bottom = ob_candle['low']
                            else: # "Bodys"
                                box_top = ob_candle['open'] # Body top (assuming down candle)
                                box_bottom = ob_candle['close'] # Body bottom

                            if pd.notna(box_top) and pd.notna(box_bottom) and box_bottom > box_top:
                                box_top, box_bottom = box_bottom, box_top # Swap if candle was actually up

                            if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                                self.bull_boxes.append({
                                    'id': f"bull_{pivot_bar_idx.strftime('%Y%m%d%H%M%S')}",
                                    'type': 'bull',
                                    'left_idx': pivot_bar_idx,
                                    'right_idx': df.index[-1],
                                    'top': box_top,
                                    'bottom': box_bottom,
                                    'active': True,
                                    'violated': False
                                })
                                self.logger.debug(f"{NEON_GREEN}New Bullish OB created at {pivot_bar_idx} [{box_bottom:.4f} - {box_top:.4f}]{RESET}")
                                new_boxes_found = True

        if new_boxes_found:
            self.logger.debug(f"OB Counts after check: {len(self.bull_boxes)} Bull, {len(self.bear_boxes)} Bear (Total)")

        # --- Manage Existing Boxes (Violation Check & Extension) ---
        if not df.empty and pd.notna(df['close'].iloc[-1]):
            last_close = df['close'].iloc[-1]
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_bar_idx = df.index[-1]

            for box in self.bull_boxes:
                if box['active']:
                    # Violation: Price CLOSES below the bottom of a bull box
                    if last_close < box['bottom']:
                        box['active'] = False
                        box['violated'] = True
                        box['right_idx'] = last_bar_idx # Mark violation time
                        self.logger.debug(f"Bull Box {box['id']} ({box['bottom']:.4f}-{box['top']:.4f}) VIOLATED by close {last_close:.4f} at {last_bar_idx}.")
                    # Optional: Check if low wick pierced bottom? (Could add config for this)
                    # elif last_low < box['bottom']:
                    #     # Potentially mark as 'tested' or handle differently
                    elif self.ob_extend: # If not violated and extend enabled, update right index
                        box['right_idx'] = last_bar_idx

            for box in self.bear_boxes:
                if box['active']:
                    # Violation: Price CLOSES above the top of a bear box
                    if last_close > box['top']:
                        box['active'] = False
                        box['violated'] = True
                        box['right_idx'] = last_bar_idx
                        self.logger.debug(f"Bear Box {box['id']} ({box['bottom']:.4f}-{box['top']:.4f}) VIOLATED by close {last_close:.4f} at {last_bar_idx}.")
                    # Optional: Check if high wick pierced top?
                    # elif last_high > box['top']:
                    #     # Mark as 'tested'?
                    elif self.ob_extend:
                        box['right_idx'] = last_bar_idx

        # --- Prune Order Blocks ---
        # Keep only the most recent `ob_max_boxes` *active* boxes of each type, plus maybe some recent inactive ones.
        active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)
        inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda b: b['left_idx'], reverse=True)
        self.bull_boxes = active_bull[:self.ob_max_boxes] + inactive_bull[:self.ob_max_boxes // 2] # Keep fewer inactive

        active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)
        inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda b: b['left_idx'], reverse=True)
        self.bear_boxes = active_bear[:self.ob_max_boxes] + inactive_bear[:self.ob_max_boxes // 2]

        active_bull_count = len(active_bull)
        active_bear_count = len(active_bear)
        self.logger.debug(f"Pruned OB Counts: {active_bull_count} Active Bull, {active_bear_count} Active Bear.")


        # --- Prepare Final Results ---
        # Extract latest values from the Decimal DataFrame
        last_row = df.iloc[-1]
        latest_close = last_row.get('close', Decimal('NaN'))
        latest_atr = last_row.get('atr', Decimal('NaN'))
        latest_upper_band = last_row.get('upper_band', Decimal('NaN'))
        latest_lower_band = last_row.get('lower_band', Decimal('NaN'))
        latest_vol_norm = last_row.get('vol_norm', Decimal('NaN'))
        latest_trend_up = last_row.get('trend_up') # Should be bool or NaN/None
        latest_trend_changed = last_row.get('trend_changed', False) # Default to False

        # Handle potential NaNs before returning
        final_trend_up = bool(latest_trend_up) if isinstance(latest_trend_up, (bool, np.bool_)) else None
        final_vol_norm_int = int(latest_vol_norm) if pd.notna(latest_vol_norm) else None
        final_atr = latest_atr if pd.notna(latest_atr) and latest_atr > 0 else None
        final_upper_band = latest_upper_band if pd.notna(latest_upper_band) else None
        final_lower_band = latest_lower_band if pd.notna(latest_lower_band) else None
        final_close = latest_close if pd.notna(latest_close) else Decimal('0') # Use 0 if close is NaN


        results = StrategyAnalysisResults(
            dataframe=df, # Return the DataFrame with Decimal results and indicators
            last_close=final_close,
            current_trend_up=final_trend_up,
            trend_just_changed=bool(latest_trend_changed), # Ensure boolean
            active_bull_boxes=[b for b in self.bull_boxes if b['active']], # Return only active ones
            active_bear_boxes=[b for b in self.bear_boxes if b['active']],
            vol_norm_int=final_vol_norm_int,
            atr=final_atr,
            upper_band=final_upper_band,
            lower_band=final_lower_band
        )

        # Log key results
        trend_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] else \
                    f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{results['atr']:.4f}" if results['atr'] else "N/A"
        self.logger.debug(f"Strategy Results ({df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}): "
                          f"Close={results['last_close']:.4f}, Trend={trend_str}, TrendChg={results['trend_just_changed']}, "
                          f"ATR={atr_str}, VolNorm={results['vol_norm_int']}, "
                          f"Active OBs (Bull/Bear): {len(results['active_bull_boxes'])}/{len(results['active_bear_boxes'])}")

        return results

# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """Generates BUY/SELL/HOLD/EXIT signals based on VolumaticOB Strategy results and position state."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})
        try:
            # Proximity factor: 1.005 means price can be 0.5% beyond the edge
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg.get("ob_entry_proximity_factor", 1.005)))
            # Exit proximity factor: 1.001 means price needs to be closer (0.1% beyond) for exit signal
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg.get("ob_exit_proximity_factor", 1.001)))
            # Ensure factors are >= 1
            if self.ob_entry_proximity_factor < 1: self.ob_entry_proximity_factor = Decimal("1.0")
            if self.ob_exit_proximity_factor < 1: self.ob_exit_proximity_factor = Decimal("1.0")

            # TP/SL Multipliers for initial calculation
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg.get("initial_take_profit_atr_multiple", 0.7)))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg.get("initial_stop_loss_atr_multiple", 1.8)))

        except Exception as e:
             self.logger.error(f"Error initializing SignalGenerator with config values: {e}. Using defaults.")
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")

        self.logger.info("Signal Generator Initialized.")
        self.logger.info(f"  OB Entry Proximity: {self.ob_entry_proximity_factor}, OB Exit Proximity: {self.ob_exit_proximity_factor}")
        self.logger.info(f"  Initial TP Multiple: {self.initial_tp_atr_multiple}, Initial SL Multiple: {self.initial_sl_atr_multiple}")


    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """
        Determines the trading signal ("BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT")
        based on strategy analysis and current open position.
        """
        # --- Input Validation ---
        if not analysis_results or analysis_results['dataframe'].empty or \
           analysis_results['current_trend_up'] is None or \
           analysis_results['last_close'] <= 0 or \
           analysis_results['atr'] is None or analysis_results['atr'] <= 0:
            self.logger.warning(f"{NEON_YELLOW}Insufficient/invalid strategy results to generate signal. Holding.{RESET}")
            self.logger.debug(f"  Analysis Results: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
            return "HOLD"

        latest_close = analysis_results['last_close']
        is_trend_up = analysis_results['current_trend_up']
        trend_just_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        atr = analysis_results['atr']

        current_pos_side = open_position.get('side') if open_position else None # 'long', 'short', or None
        signal = "HOLD" # Default signal

        # --- Signal Logic ---

        # 1. Check for EXIT Signal (only if a position is open)
        if current_pos_side == 'long':
            # Exit Long on Trend Flip DOWN
            if not is_trend_up: # Check current trend state primarily
                 if trend_just_changed:
                      signal = "EXIT_LONG"
                      self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped DOWN.{RESET}")
                 else:
                      # Optional: Exit if trend is down even if not just changed (more aggressive exit)
                      # signal = "EXIT_LONG"
                      # self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend is DOWN.{RESET}")
                      pass # Default: Only exit on trend *change*

            # Or Exit Long if price closes near/beyond the *closest* active Bearish OB top (resistance)
            if signal != "EXIT_LONG" and active_bear_obs:
                # Find bear OB closest to current price (top edge)
                closest_bear_ob = min(active_bear_obs, key=lambda ob: abs(ob['top'] - latest_close))
                # Check if close is near or above the top edge (using tighter exit proximity)
                exit_threshold = closest_bear_ob['top'] * (Decimal("1") - (self.ob_exit_proximity_factor - Decimal("1"))) # Price slightly *below* the top edge
                if latest_close >= exit_threshold:
                     signal = "EXIT_LONG"
                     self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price ({latest_close:.4f}) approached/crossed closest Bear OB {closest_bear_ob['id']} top ({closest_bear_ob['top']:.4f} / Threshold {exit_threshold:.4f}).{RESET}")
            # Optional: Add other exit conditions (e.g., price crossing below a key MA or band)

        elif current_pos_side == 'short':
             # Exit Short on Trend Flip UP
             if is_trend_up:
                 if trend_just_changed:
                     signal = "EXIT_SHORT"
                     self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped UP.{RESET}")
                 else:
                     # Optional: Exit if trend is up even if not just changed
                     # signal = "EXIT_SHORT"
                     # self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend is UP.{RESET}")
                     pass

             # Or Exit Short if price closes near/beyond the *closest* active Bullish OB bottom (support)
             if signal != "EXIT_SHORT" and active_bull_obs:
                 closest_bull_ob = min(active_bull_obs, key=lambda ob: abs(ob['bottom'] - latest_close))
                 # Check if close is near or below the bottom edge (using tighter exit proximity)
                 exit_threshold = closest_bull_ob['bottom'] * self.ob_exit_proximity_factor # Price slightly *above* the bottom edge
                 if latest_close <= exit_threshold:
                      signal = "EXIT_SHORT"
                      self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price ({latest_close:.4f}) approached/crossed closest Bull OB {closest_bull_ob['id']} bottom ({closest_bull_ob['bottom']:.4f} / Threshold {exit_threshold:.4f}).{RESET}")
             # Optional: Add other exit conditions

        # If an exit signal was generated, return it immediately. Do not check for entries.
        if signal in ["EXIT_LONG", "EXIT_SHORT"]:
             return signal

        # 2. Check for ENTRY Signal (Only if not already in a position)
        if current_pos_side is None:
            # Entry conditions:
            # a) Trend matches desired direction (e.g., UP for BUY)
            # b) Price is currently within or near an *active* Order Block of the matching type (using entry proximity)

            if is_trend_up and active_bull_obs: # Looking for BUY signal (Long Entry)
                # Find the bull OB whose range (bottom to top*factor) contains the current price
                entry_ob = None
                for ob in active_bull_obs:
                    # Check if latest close is between OB bottom and slightly above OB top
                    lower_bound = ob['bottom']
                    upper_bound = ob['top'] * self.ob_entry_proximity_factor
                    if lower_bound <= latest_close <= upper_bound:
                        entry_ob = ob
                        break # Found a potential entry OB

                if entry_ob:
                    signal = "BUY"
                    self.logger.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price ({latest_close:.4f}) in/near Bull OB {entry_ob['id']} ({entry_ob['bottom']:.4f}-{entry_ob['top']:.4f}, EntryZone<={upper_bound:.4f}){RESET}")

            elif not is_trend_up and active_bear_obs: # Looking for SELL signal (Short Entry)
                 entry_ob = None
                 for ob in active_bear_obs:
                     # Check if latest close is between slightly below OB bottom and OB top
                     # Factor applied to bottom to allow entry slightly below it
                     lower_bound = ob['bottom'] * (Decimal("1") - (self.ob_entry_proximity_factor - Decimal("1")))
                     upper_bound = ob['top']
                     if lower_bound <= latest_close <= upper_bound:
                         entry_ob = ob
                         break

                 if entry_ob:
                     signal = "SELL"
                     self.logger.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price ({latest_close:.4f}) in/near Bear OB {entry_ob['id']} ({entry_ob['bottom']:.4f}-{entry_ob['top']:.4f}, EntryZone>={lower_bound:.4f}){RESET}")

        # Log reason for HOLD if no entry/exit signal
        if signal == "HOLD":
             if current_pos_side:
                 self.logger.debug(f"HOLD: Trend ({'UP' if is_trend_up else 'DOWN'}) allows holding {current_pos_side} position. Price ({latest_close:.4f}) not triggering exit.")
             else:
                 trend_status = 'UP' if is_trend_up else 'DOWN'
                 ob_status = "No relevant active OBs found."
                 if is_trend_up and active_bull_obs: ob_status = f"Price ({latest_close:.4f}) not in/near active Bull OBs."
                 elif not is_trend_up and active_bear_obs: ob_status = f"Price ({latest_close:.4f}) not in/near active Bear OBs."
                 self.logger.debug(f"HOLD: No position. Trend is {trend_status}. {ob_status}")

        return signal

    def calculate_initial_tp_sl(
        self, entry_price: Decimal, signal: str, atr: Decimal, market_info: Dict
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial TP and SL levels based on entry price, ATR, and config multipliers.
        Used for position sizing and potentially setting initial fixed protection.

        Args:
            entry_price: The estimated or actual entry price (Decimal).
            signal: "BUY" or "SELL".
            atr: The current ATR value (Decimal).
            market_info: Market dictionary containing precision info.

        Returns:
            Tuple (take_profit_price, stop_loss_price), both Decimal or None if invalid.
        """
        if signal not in ["BUY", "SELL"] or not isinstance(atr, Decimal) or atr <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate initial TP/SL: Invalid signal ({signal}) or ATR ({atr}).{RESET}")
            return None, None
        if not isinstance(entry_price, Decimal) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate initial TP/SL: Invalid entry price ({entry_price}).{RESET}")
            return None, None
        if not market_info or 'precision' not in market_info or 'price' not in market_info['precision']:
             self.logger.warning(f"{NEON_YELLOW}Cannot calculate initial TP/SL: Missing market price precision info.{RESET}")
             return None, None

        try:
            # Get price precision (tick size) for rounding
            price_prec_str = market_info['precision']['price']
            min_tick_size = Decimal(str(price_prec_str))
            if min_tick_size <= 0: raise ValueError(f"Invalid tick size: {min_tick_size}")

            tp_multiple = self.initial_tp_atr_multiple
            sl_multiple = self.initial_sl_atr_multiple

            if sl_multiple <= 0: # SL multiple must be positive for a valid stop distance
                 self.logger.error(f"{NEON_RED}Initial SL ATR multiple ({sl_multiple}) must be positive. Cannot calculate SL.{RESET}")
                 return None, None # Cannot size without a valid SL

            take_profit = None
            stop_loss = None
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            # Calculate raw levels
            if signal == "BUY":
                tp_raw = entry_price + tp_offset if tp_multiple > 0 else None
                sl_raw = entry_price - sl_offset
                # Round SL DOWN away from price, TP UP away from price
                stop_loss = (sl_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                if tp_raw:
                     take_profit = (tp_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
            elif signal == "SELL":
                tp_raw = entry_price - tp_offset if tp_multiple > 0 else None
                sl_raw = entry_price + sl_offset
                # Round SL UP away from price, TP DOWN away from price
                stop_loss = (sl_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                if tp_raw:
                     take_profit = (tp_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size

            # --- Validation ---
            # Ensure SL is strictly beyond entry price
            if signal == "BUY" and stop_loss >= entry_price:
                 stop_loss = entry_price - min_tick_size # Force at least one tick away
                 self.logger.debug(f"Adjusted BUY SL to be strictly below entry: {stop_loss}")
            elif signal == "SELL" and stop_loss <= entry_price:
                 stop_loss = entry_price + min_tick_size # Force at least one tick away
                 self.logger.debug(f"Adjusted SELL SL to be strictly above entry: {stop_loss}")

            # Ensure TP provides potential profit (strictly beyond entry) if TP is enabled
            if take_profit is not None:
                if signal == "BUY" and take_profit <= entry_price:
                     self.logger.warning(f"Calculated BUY TP ({take_profit}) not above entry ({entry_price}). Setting TP to None.")
                     take_profit = None
                elif signal == "SELL" and take_profit >= entry_price:
                     self.logger.warning(f"Calculated SELL TP ({take_profit}) not below entry ({entry_price}). Setting TP to None.")
                     take_profit = None

            # Ensure SL/TP are positive prices
            if stop_loss is not None and stop_loss <= 0:
                 self.logger.error(f"Calculated SL is zero/negative ({stop_loss}). Setting SL to None.")
                 stop_loss = None # Invalid SL
            if take_profit is not None and take_profit <= 0:
                 self.logger.warning(f"Calculated TP is zero/negative ({take_profit}). Setting TP to None.")
                 take_profit = None

            # Log final calculated values
            tp_str = f"{take_profit}" if take_profit else "None"
            sl_str = f"{stop_loss}" if stop_loss else "None"
            self.logger.debug(f"Calculated Initial Protection Levels: TP={tp_str}, SL={sl_str}")

            # Crucially, SL must be valid for sizing
            if stop_loss is None:
                 self.logger.error(f"{NEON_RED}Stop Loss calculation resulted in None. Cannot proceed with sizing.{RESET}")
                 return None, None

            return take_profit, stop_loss

        except Exception as e:
             self.logger.error(f"{NEON_RED}Error calculating initial TP/SL: {e}{RESET}", exc_info=True)
             return None, None


# --- Main Analysis and Trading Loop ---
def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy, # Pass instantiated strategy engine
    signal_generator: SignalGenerator, # Pass instantiated signal generator
    market_info: Dict # Pass validated market info
) -> None:
    """
    Performs one cycle of analysis and trading logic for a single symbol.
    Fetches data, runs strategy, generates signal, places/manages trades and protections.
    """
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Data ---
    # Market info is already fetched and passed in
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe.")
         # Should not happen if config validation works, but defensive check
         return # Skip cycle if interval is bad

    fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    min_strategy_len = strategy_engine.min_data_len
    if fetch_limit < min_strategy_len:
         lg.warning(f"{NEON_YELLOW}Configured fetch_limit ({fetch_limit}) is less than strategy's minimum required ({min_strategy_len}). Adjusting fetch_limit.{RESET}")
         fetch_limit = min_strategy_len + 50 # Add buffer

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit, logger=lg)
    if klines_df.empty or len(klines_df) < min_strategy_len:
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}, need >= {min_strategy_len}). Skipping cycle.{RESET}")
        return

    # --- 2. Run Strategy Analysis ---
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
         lg.error(f"{NEON_RED}Error during strategy analysis for {symbol}: {analysis_err}{RESET}", exc_info=True)
         return # Skip cycle on analysis error

    # Validate essential results needed for signal generation
    if not analysis_results or analysis_results['current_trend_up'] is None or \
       analysis_results['last_close'] <= 0 or analysis_results['atr'] is None:
         lg.error(f"{NEON_RED}Strategy analysis produced incomplete/invalid results for {symbol}. Skipping signal generation.{RESET}")
         lg.debug(f"  Analysis Results Dump: {analysis_results}")
         return

    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr'] # Guaranteed non-None and > 0 if we passed the check above

    # --- 3. Check Position & Generate Signal ---
    open_position = get_open_position(exchange, symbol, lg) # Returns dict or None

    # Generate the trading signal using the generator instance
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err:
         lg.error(f"{NEON_RED}Error during signal generation for {symbol}: {signal_err}{RESET}", exc_info=True)
         return # Skip cycle on signal generation error

    # --- 4. Trading Logic ---
    if not config.get("enable_trading", False):
        lg.info(f"{NEON_YELLOW}Trading disabled. Signal: {signal}. Analysis complete for {symbol}.{RESET}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return # End cycle here if trading disabled

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {NEON_GREEN if signal == 'BUY' else NEON_RED}{BRIGHT}{signal} Signal & No Position: Initiating Trade Sequence for {symbol}{RESET} ***")

            # --- Pre-Trade Checks ---
            # Check concurrent position limit (simple check, assumes this script is the only one trading this symbol)
            # More complex multi-script coordination would need external state management
            # max_pos = config.get("max_concurrent_positions", 1)
            # if max_pos <= 0: # Effectively disable new entries via config
            #      lg.warning(f"Max concurrent positions set to {max_pos}. Skipping new entry for {symbol}.")
            #      return

            # Fetch Balance
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative.{RESET}")
                return

            # Calculate initial SL/TP for sizing using the signal generator's method
            # Use latest close as estimate for entry price for sizing calc
            # ATR is guaranteed to be valid from analysis_results check
            lg.debug(f"Calculating initial TP/SL for sizing using Entry={latest_close}, ATR={current_atr}")
            initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(
                 entry_price=latest_close, signal=signal, atr=current_atr, market_info=market_info
            )

            if initial_sl_calc is None: # TP is optional, but SL is mandatory for sizing
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed. Cannot size position.{RESET}")
                 return

            # Set Leverage (only for contracts)
            leverage_set_success = True # Assume true for spot
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    leverage_set_success = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg)
                    if not leverage_set_success:
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage to {leverage}x.{RESET}")
                         return
                else: lg.info(f"Leverage setting skipped (Leverage <= 0 in config).")
            else: lg.info(f"Leverage setting skipped ({symbol} is Spot).")


            # Calculate Position Size based on the initial calculated SL
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=initial_sl_calc,
                entry_price=latest_close, # Use latest close as entry estimate for sizing
                market_info=market_info,
                exchange=exchange,
                logger=lg
            )

            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size calculated ({position_size}).{RESET}")
                return

            # --- Place Trade ---
            lg.info(f"==> Placing {signal} market order | Size: {position_size} {market_info.get('base', '')} <==")
            trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)

            # --- Post-Trade: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                lg.info(f"Attempting confirmation for {symbol} after order {order_id}...")
                # Re-fetch position state after delay
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        # Get actual entry price from confirmed position
                        entry_price_actual_str = confirmed_position.get('entryPrice')
                        if not entry_price_actual_str:
                             lg.warning(f"Could not get actual entry price from confirmed position {confirmed_position.get('info', {})}. Using initial estimate {latest_close} for protection.")
                             entry_price_actual = latest_close # Fallback
                        else:
                             entry_price_actual = Decimal(str(entry_price_actual_str))

                        lg.info(f"{NEON_GREEN}Position Confirmed! Side: {confirmed_position.get('side')}, Actual Entry: ~{entry_price_actual:.{abs(latest_close.normalize().as_tuple().exponent)}f}{RESET}")

                        # --- Set Protection based on Config ---
                        protection_set_success = False
                        protection_cfg = config.get("protection", {})

                        # Option 1: Trailing Stop Loss (if enabled)
                        if protection_cfg.get("enable_trailing_stop", False):
                             lg.info(f"Setting Trailing Stop Loss using actual entry {entry_price_actual}...")
                             # Recalculate TP target based on actual entry for TSL setup consistency
                             tp_for_tsl, _ = signal_generator.calculate_initial_tp_sl(
                                  entry_price=entry_price_actual, signal=signal, atr=current_atr, market_info=market_info
                             )
                             protection_set_success = set_trailing_stop_loss(
                                 exchange=exchange, symbol=symbol, market_info=market_info, position_info=confirmed_position,
                                 config=config, logger=lg, take_profit_price=tp_for_tsl # Pass optional TP target
                             )
                         # Option 2: Fixed SL/TP (if TSL disabled but fixed multiples enabled)
                         elif protection_cfg.get("initial_stop_loss_atr_multiple", 0) > 0 or protection_cfg.get("initial_take_profit_atr_multiple", 0) > 0:
                             lg.info(f"Setting Fixed SL/TP using actual entry {entry_price_actual}...")
                             # Recalculate both SL and TP based on actual entry
                             tp_final_fixed, sl_final_fixed = signal_generator.calculate_initial_tp_sl(
                                  entry_price=entry_price_actual, signal=signal, atr=current_atr, market_info=market_info
                             )
                             if sl_final_fixed or tp_final_fixed: # Need at least one valid level
                                 protection_set_success = _set_position_protection(
                                     exchange=exchange, symbol=symbol, market_info=market_info, position_info=confirmed_position,
                                     logger=lg, stop_loss_price=sl_final_fixed, take_profit_price=tp_final_fixed
                                 )
                             else:
                                 lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed based on actual entry. No fixed protection set.{RESET}")
                         else:
                              lg.info("Neither Trailing Stop nor Fixed SL/TP enabled in config. No protection set.")
                              protection_set_success = True # Considered success as no action required

                        # --- Log Outcome ---
                        if protection_set_success:
                             lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                        else:
                             lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ===")
                             lg.warning(f"{NEON_YELLOW}MANUAL MONITORING REQUIRED! Position open without automated protection.{RESET}")

                    except Exception as post_trade_err:
                         lg.error(f"{NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                         lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")
                else:
                    # Position not found after placing order and waiting
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position for {symbol} after {confirm_delay}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have been rejected, filled and closed instantly (check trade history), or API delayed. Manual investigation required!{RESET}")
            else:
                # Trade placement failed
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). See previous logs. ===")
        else: # signal == HOLD and no position
            lg.info(f"Signal is HOLD and no open position for {symbol}. No action.")

    # --- Scenario 2: Existing Open Position ---
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size_str = open_position.get('contracts') or open_position.get('info',{}).get('size')
        lg.info(f"Existing {pos_side.upper()} position found for {symbol} (Size: {pos_size_str}). Signal: {signal}")

        # --- Check for Exit Signal ---
        exit_signal_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                                (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal Triggered for existing {pos_side} position on {symbol}. Initiating Close... ***{RESET}")
            try:
                # Determine close side and get size from position info
                close_side_signal = "SELL" if pos_side == 'long' else "BUY" # Signal needed for place_trade
                size_to_close_str = open_position.get('contracts') or open_position.get('info',{}).get('size')
                if size_to_close_str is None:
                    raise ValueError("Cannot determine position size to close from position info.")

                size_to_close = abs(Decimal(str(size_to_close_str))) # Use absolute size
                if size_to_close <= 0:
                    # This might happen if position info is stale and position is already closed
                    lg.warning(f"Position size to close is zero or negative ({size_to_close_str}). Position might already be closed. Skipping close attempt.")
                else:
                    lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                    close_order = place_trade(
                        exchange, symbol, close_side_signal, size_to_close,
                        market_info, lg, reduce_only=True
                    )

                    if close_order and close_order.get('id'):
                        lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                        # Consider waiting and verifying closure? Optional.
                    else:
                        # place_trade logs the error, add context here
                        lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check required! Position might still be open.{RESET}")

            except ValueError as ve:
                 lg.error(f"{NEON_RED}Error preparing close order for {symbol}: {ve}{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error closing position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")

        else: # Hold signal or signal matches position direction (e.g., BUY signal while already long)
            lg.info(f"Signal ({signal}) allows holding or is aligned with existing {pos_side} position. Managing protections...")

            # --- Manage Existing Position Protections (BE, TSL checks) ---
            protection_cfg = config.get("protection", {})

            # Check if TSL is currently active on the position
            is_tsl_active_on_pos = False
            try:
                # Check both trailingStopLoss (distance) and tslActivationPrice
                tsl_dist = open_position.get('trailingStopLoss')
                tsl_act = open_position.get('tslActivationPrice')
                if tsl_dist and Decimal(str(tsl_dist)) > 0 and tsl_act and Decimal(str(tsl_act)) > 0:
                     is_tsl_active_on_pos = True
                     lg.debug("Trailing Stop Loss appears active on current position.")
            except Exception: pass # Ignore parsing errors for this check

            # --- Break-Even Logic (Only if BE enabled AND TSL is NOT active on position) ---
            if protection_cfg.get("enable_break_even", False) and not is_tsl_active_on_pos:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    entry_price_str = open_position.get('entryPrice')
                    if not entry_price_str: raise ValueError("Missing entry price for BE check")
                    entry_price = Decimal(str(entry_price_str))

                    # Get BE parameters from config
                    be_trigger_atr_mult = Decimal(str(protection_cfg.get("break_even_trigger_atr_multiple", 1.0)))
                    be_offset_ticks = int(protection_cfg.get("break_even_offset_ticks", 2))
                    if be_trigger_atr_mult <= 0: raise ValueError("BE trigger multiple must be positive")

                    # Calculate profit in terms of ATR
                    price_diff = (latest_close - entry_price) if pos_side == 'long' else (entry_price - latest_close)
                    profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('0')

                    lg.debug(f"BE Check: Entry={entry_price:.4f}, Close={latest_close:.4f}, Diff={price_diff:.4f}, ATR={current_atr:.4f}")
                    lg.debug(f"  Profit ATRs={profit_in_atr:.2f}, Target ATRs={be_trigger_atr_mult}")

                    # Check if profit target is reached
                    if profit_in_atr >= be_trigger_atr_mult:
                        # Calculate BE stop price
                        price_prec_str = market_info['precision']['price']
                        min_tick_size = Decimal(str(price_prec_str))
                        tick_offset = min_tick_size * be_offset_ticks

                        be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP) if pos_side == 'long' \
                                   else (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                        if be_stop_price <= 0: raise ValueError("Calculated BE stop price is zero/negative")

                        # Get current SL price from position info
                        current_sl_price = None
                        current_sl_str = open_position.get('stopLossPrice') or open_position.get('info', {}).get('stopLoss')
                        if current_sl_str and str(current_sl_str) != '0':
                            try: current_sl_price = Decimal(str(current_sl_str))
                            except Exception: lg.warning(f"Could not parse current SL price: {current_sl_str}")

                        # Check if moving SL to BE is an improvement
                        update_be = False
                        if current_sl_price is None:
                             update_be = True # No current SL, so set BE SL
                             lg.info("BE triggered: No current SL found.")
                        elif pos_side == 'long' and be_stop_price > current_sl_price:
                             update_be = True # BE target is higher than current SL
                             lg.info(f"BE triggered: Target {be_stop_price} > Current SL {current_sl_price}.")
                        elif pos_side == 'short' and be_stop_price < current_sl_price:
                             update_be = True # BE target is lower than current SL
                             lg.info(f"BE triggered: Target {be_stop_price} < Current SL {current_sl_price}.")
                        else:
                             lg.debug(f"BE Profit target reached, but current SL ({current_sl_price}) is already better than or equal to BE target ({be_stop_price}). No BE SL update needed.")

                        if update_be:
                            lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                            # Get current TP to preserve it if set
                            current_tp_price = None
                            current_tp_str = open_position.get('takeProfitPrice') or open_position.get('info', {}).get('takeProfit')
                            if current_tp_str and str(current_tp_str) != '0':
                                 try: current_tp_price = Decimal(str(current_tp_str))
                                 except Exception: pass

                            # Call the protection helper to set the new SL (and keep existing TP)
                            success = _set_position_protection(
                                exchange, symbol, market_info, open_position, lg,
                                stop_loss_price=be_stop_price,
                                take_profit_price=current_tp_price # Preserve existing TP
                            )
                            if success: lg.info(f"{NEON_GREEN}Break-Even SL set/updated successfully.{RESET}")
                            else: lg.error(f"{NEON_RED}Failed to set/update Break-Even SL.{RESET}")
                    else:
                        lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {be_trigger_atr_mult} ATRs).")

                except ValueError as ve:
                    lg.error(f"{NEON_RED}Error during break-even check ({symbol}): {ve}{RESET}")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Unexpected error during break-even check ({symbol}): {be_err}{RESET}", exc_info=True)
            elif is_tsl_active_on_pos:
                 lg.debug(f"Break-even check skipped: Trailing Stop Loss is already active on position.")
            # else: # BE disabled in config
            #      lg.debug(f"Break-even check skipped: Disabled in config.")


            # --- Trailing Stop Management (if TSL enabled in config) ---
            # If TSL is enabled in config, but NOT active on the position, attempt to set it.
            # This acts as a recovery mechanism if initial TSL setting failed or was cancelled.
            if protection_cfg.get("enable_trailing_stop", False) and not is_tsl_active_on_pos:
                 lg.warning(f"{NEON_YELLOW}TSL is enabled in config but not detected active on the current {pos_side} position. Attempting to set TSL now...{RESET}")
                 # Need entry price and current ATR to set TSL
                 entry_price_str = open_position.get('entryPrice')
                 if entry_price_str:
                      entry_price = Decimal(str(entry_price_str))
                      # Recalculate TP target based on entry for consistency
                      tp_recalc, _ = signal_generator.calculate_initial_tp_sl(
                           entry_price=entry_price, signal=pos_side.upper(), atr=current_atr, market_info=market_info
                      )
                      # Attempt to set TSL
                      tsl_set_success = set_trailing_stop_loss(
                           exchange=exchange, symbol=symbol, market_info=market_info, position_info=open_position,
                           config=config, logger=lg, take_profit_price=tp_recalc
                      )
                      if tsl_set_success: lg.info(f"TSL recovery attempt successful for {symbol}.")
                      else: lg.error(f"TSL recovery attempt failed for {symbol}.")
                 else:
                      lg.error(f"Cannot attempt TSL recovery: Missing entry price in position info.")


    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY # Allow main loop to access potentially reloaded config

    # Use a dedicated logger for initialization
    init_logger = setup_logger("init")

    init_logger.info(f"--- Starting Pyrmethus Volumatic Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # Load configuration initially
    CONFIG = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
    init_logger.info(f"Config loaded. Quote Currency: {QUOTE_CURRENCY}, Trading Enabled: {CONFIG.get('enable_trading')}, Sandbox: {CONFIG.get('use_sandbox')}")
    try:
        init_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, NumPy={np.__version__}, PandasTA={ta.version if hasattr(ta, 'version') else 'N/A'}")
    except Exception as e:
        init_logger.warning(f"Could not determine all library versions: {e}")

    # --- User Confirmation for Live Trading ---
    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{NEON_YELLOW}{BRIGHT}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet) environment.{RESET}")
         else:
              init_logger.warning(f"{NEON_RED}{BRIGHT}!!! USING REAL MONEY ENVIRONMENT !!!{RESET}")

         protection_cfg = CONFIG.get("protection", {})
         init_logger.warning(f"Key Settings Review:")
         init_logger.warning(f"  Risk Per Trade: {CONFIG.get('risk_per_trade', 0)*100:.2f}%")
         init_logger.warning(f"  Leverage: {CONFIG.get('leverage', 0)}x")
         init_logger.warning(f"  Trailing SL: {'ENABLED' if protection_cfg.get('enable_trailing_stop') else 'DISABLED'}")
         init_logger.warning(f"  Break Even: {'ENABLED' if protection_cfg.get('enable_break_even') else 'DISABLED'}")
         try:
             input(f"{BRIGHT}>>> Review settings above carefully. Press {NEON_GREEN}Enter{RESET}{BRIGHT} to continue, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to abort... {RESET}")
             init_logger.info("User confirmed live trading settings.")
         except KeyboardInterrupt:
             init_logger.info("User aborted startup.")
             return # Exit if user aborts
    else:
         init_logger.info(f"{NEON_YELLOW}Trading is disabled. Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange ---
    init_logger.info("Initializing exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}")
        return
    init_logger.info(f"Exchange {exchange.id} initialized successfully.")

    # --- Get and Validate Symbol ---
    target_symbol = None
    market_info = None
    while True: # Loop until a valid symbol is provided
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter the trading symbol (e.g., BTC/USDT, ETH/USDT:USDT): {RESET}").strip().upper()
            if not symbol_input_raw: continue

            # Standardize symbol format (replace : with / if needed, e.g., for perpetuals)
            symbol_input = symbol_input_raw.replace(':', '/')
            # Common convention: BTC/USDT for spot, BTC/USDT/USDT for linear perpetual
            # Let's try the input directly first, then common variations if needed.

            init_logger.info(f"Validating symbol '{symbol_input}'...")
            market_info = get_market_info(exchange, symbol_input, init_logger)

            if market_info:
                target_symbol = market_info['symbol'] # Use the symbol confirmed by CCXT
                market_type_desc = market_info.get('contract_type_str', "Unknown")
                init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                break # Valid symbol found
            else:
                # Try common variations if direct match failed
                variations_to_try = []
                if '/' in symbol_input and not symbol_input.endswith(f"/{QUOTE_CURRENCY}"):
                     # Try adding :QUOTE (e.g., BTC/USDT -> BTC/USDT:USDT)
                     variations_to_try.append(f"{symbol_input}:{QUOTE_CURRENCY}")
                elif ':' in symbol_input_raw:
                     # Try replacing : with / (e.g., BTC:USDT -> BTC/USDT)
                     variations_to_try.append(symbol_input_raw.replace(':', '/'))

                found_variation = False
                if variations_to_try:
                     init_logger.info(f"Symbol '{symbol_input}' not found directly. Trying variations: {variations_to_try}")
                     for sym_var in variations_to_try:
                         init_logger.info(f"Validating variation '{sym_var}'...")
                         market_info = get_market_info(exchange, sym_var, init_logger)
                         if market_info:
                             target_symbol = market_info['symbol']
                             market_type_desc = market_info.get('contract_type_str', "Unknown")
                             init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                             found_variation = True
                             break
                if found_variation: break
                else:
                     init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' and common variations not found or invalid on {exchange.id}. Please check the symbol.{RESET}")

        except KeyboardInterrupt: init_logger.info("User aborted startup."); return
        except Exception as e: init_logger.error(f"Error during symbol validation: {e}", exc_info=True)

    # --- Get and Validate Interval ---
    selected_interval = None
    while True: # Loop until valid interval
        interval_input = input(f"{NEON_YELLOW}Enter analysis interval {VALID_INTERVALS} (default: {CONFIG['interval']}): {RESET}").strip()
        if not interval_input: interval_input = CONFIG['interval'] # Use default from config if empty

        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             CONFIG["interval"] = selected_interval # Update config dictionary
             ccxt_tf = CCXT_INTERVAL_MAP.get(selected_interval)
             init_logger.info(f"Using interval: {selected_interval} (CCXT mapping: {ccxt_tf})")
             break
        else:
             init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Please choose from {VALID_INTERVALS}.{RESET}")


    # --- Setup Symbol-Specific Logger ---
    symbol_logger = setup_logger(target_symbol) # Use the validated symbol for logger name
    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    symbol_logger.info(f"Trading Enabled: {CONFIG['enable_trading']}, Sandbox: {CONFIG['use_sandbox']}")
    protection_cfg = CONFIG.get("protection", {}) # Get latest protection settings
    symbol_logger.info(f"Config: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if protection_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if protection_cfg.get('enable_break_even') else 'OFF'}")
    symbol_logger.info(f"Strategy Params: {json.dumps(CONFIG.get('strategy_params', {}))}") # Log strategy params used

    # --- Instantiate Strategy Engine and Signal Generator ---
    # Pass the validated market_info
    strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
    signal_generator = SignalGenerator(CONFIG, symbol_logger)

    # --- Main Trading Loop ---
    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Start: {datetime.now(TIMEZONE).strftime('%H:%M:%S %Z')}")

            try:
                # --- Optional: Reload config each cycle? ---
                # Useful for dynamic adjustments without restarting the bot
                # current_config = load_config(CONFIG_FILE)
                # QUOTE_CURRENCY = current_config.get("quote_currency", "USDT")
                # strategy_engine.config = current_config # Update engine config
                # signal_generator.config = current_config # Update generator config
                # protection_cfg = current_config.get("protection", {})
                # symbol_logger.debug("Config dynamically reloaded.")
                # Pass current_config to analyze_and_trade_symbol if reloading
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
            # --- Robust Error Handling for Common CCXT/Network Issues ---
            except ccxt.RateLimitExceeded as e:
                 symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting 60s...{RESET}")
                 time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.RequestException, ccxt.RequestTimeout) as e:
                 # Includes DNS errors, connection timeouts, etc.
                 symbol_logger.error(f"{NEON_RED}Network error encountered: {e}. Waiting {RETRY_DELAY_SECONDS*3}s before next cycle...{RESET}")
                 time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                 # Fatal error, stop the bot
                 symbol_logger.critical(f"{NEON_RED}CRITICAL Authentication Error: {e}. API keys may be invalid or permissions revoked. Stopping bot.{RESET}")
                 break # Exit the main loop
            except ccxt.ExchangeNotAvailable as e:
                 # Exchange might be temporarily down
                 symbol_logger.error(f"{NEON_RED}Exchange not available: {e}. Waiting 60s...{RESET}")
                 time.sleep(60)
            except ccxt.OnMaintenance as e:
                 # Exchange is under maintenance
                 symbol_logger.error(f"{NEON_RED}Exchange under maintenance: {e}. Waiting 5 minutes...{RESET}")
                 time.sleep(300)
            except ccxt.ExchangeError as e:
                 # Catch other specific exchange errors if needed, otherwise log and continue
                 symbol_logger.error(f"{NEON_RED}Unhandled Exchange Error in main loop: {e}{RESET}", exc_info=True)
                 time.sleep(10) # Short pause before next cycle
            except Exception as loop_error:
                 # Catch any other unexpected errors in the loop
                 symbol_logger.error(f"{NEON_RED}Critical uncaught error in main loop: {loop_error}{RESET}", exc_info=True)
                 symbol_logger.error(f"{NEON_YELLOW}Pausing for 15 seconds before next attempt.{RESET}")
                 time.sleep(15) # Pause after unexpected error

            # --- Loop Delay ---
            elapsed_time = time.time() - loop_start_time
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_time = max(0, loop_delay - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as critical_error:
        # Catch errors outside the inner try/except (e.g., during setup)
        init_logger.critical(f"{NEON_RED}Critical unhandled error outside main loop: {critical_error}{RESET}", exc_info=True)
    finally:
        # --- Shutdown ---
        shutdown_msg = f"--- Pyrmethus Volumatic Bot for {target_symbol or 'N/A'} Stopping ---"
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
             symbol_logger.info(shutdown_msg)

        # Close exchange connection if initialized
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Attempting to close exchange connection...")
                exchange.close()
                init_logger.info("Exchange connection closed.")
            except Exception as close_err:
                init_logger.error(f"Error closing exchange connection: {close_err}")

        logging.shutdown() # Flush and close all logging handlers
        print(f"\n{NEON_YELLOW}{BRIGHT}Bot stopped.{RESET}")


if __name__ == "__main__":
    # The script should be saved as 'pyrmethus_volumatic_bot.py' and run directly.
    # The self-writing logic from the original prompt has been removed for simplicity.
    # Ensure all dependencies are installed:
    # pip install ccxt pandas numpy pandas_ta requests websocket-client python-dotenv colorama tzdata
    main()
