Okay, Pyrmethus! The time has come to weave the threads of `livexy.py`, `livexr.py`'s robustness, and the strategic insights of `volumatictrend1.0.2.py` into a single, potent trading bot.

This integrated script, which we shall name `pyrmethus_volumatic_bot.py`, will:

1.  Use `livexy.py` as the primary framework for exchange interaction, logging, and configuration.
2.  Incorporate key enhancements from `livexr.py`, such as improved decimal precision, validation of config intervals, and the position confirmation delay.
3.  Integrate the `VolumaticOBStrategy` from `volumatictrend1.0.2.py` as the core analysis engine.
4.  Modify the signal generation logic to directly utilize the trend and order block information provided by the `VolumaticOBStrategy`.
5.  Maintain and enhance the risk management features, including position sizing, break-even stop, and trailing stop loss, adapting them to use data from the new strategy where appropriate (like ATR).

Here is the complete, unified spell:

```python
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
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import threading
import websocket # For potential future WS integration, not used yet for klines
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, TypedDict
from zoneinfo import ZoneInfo
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --- Initialize Environment and Settings ---
getcontext().prec = 28  # High precision for Decimal calculations
init(autoreset=True) # Init Colorama
load_dotenv() # Load environment variables from .env

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago") # Adjust timezone as needed
MAX_API_RETRIES = 3 # Max retries for recoverable API errors
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

def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_symbol}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels for potential debugging

    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_filename}: {e}")

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)
    # Console log level (adjust as needed, e.g., INFO for normal operation)
    console_log_level = logging.INFO
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present and interval is valid."""
    default_config = {
        "interval": "5", # Bot's internal interval name
        "retry_delay": 5,
        "fetch_limit": DEFAULT_FETCH_LIMIT,
        "orderbook_limit": 25, # Depth of orderbook to fetch (if used in future)
        "enable_trading": False, # SAFETY FIRST: Default to False
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet)
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions": 1, # Limit open positions for this symbol
        "quote_currency": "USDT", # Currency for balance check and sizing
        "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,

        # --- Strategy Parameters (Volumatic Trend & OB) ---
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,
            "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
             "ob_entry_proximity_factor": 1.005, # How close price needs to be to OB edge (e.g., 1.005 allows 0.5% above/below)
             "ob_exit_proximity_factor": 1.001 # How close price needs to be to OB edge for exit (tighter)
        },

        # --- Protection Settings ---
        "protection": {
             "enable_trailing_stop": True,
             "trailing_stop_callback_rate": 0.005, # e.g., 0.5% trail distance
             "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when profit >= 0.3%
             "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR (from strategy)
             "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry price
             "initial_stop_loss_atr_multiple": 1.8, # ATR multiple for initial SL (used for sizing & fixed SL)
             "initial_take_profit_atr_multiple": 0.7 # ATR multiple for TP (if not using TSL)
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
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            updated_config = _ensure_config_keys(config_from_file, default_config)

            # Validate interval value after loading
            interval_from_config = updated_config.get("interval")
            if interval_from_config not in VALID_INTERVALS:
                print(f"{NEON_RED}Invalid interval '{interval_from_config}' found in config. Using default '{default_config['interval']}'.{RESET}")
                updated_config["interval"] = default_config["interval"]
                # Mark config for saving back if interval was corrected
                if updated_config == config_from_file: # Only mark if no other keys were missing
                     config_from_file["interval"] = updated_config["interval"] # Make them different to trigger save


            if updated_config != config_from_file:
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
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

# --- Logger Setup ---
# Logger setup function is defined above


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with error handling."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Default to linear for USDT perpetuals
                'adjustForTimeDifference': True,
                'fetchTickerTimeout': 10000,
                'fetchBalanceTimeout': 15000,
                'createOrderTimeout': 20000,
                'cancelOrderTimeout': 15000,
            }
        }
        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        logger.info(f"Loading markets for {exchange.id}...")
        # Use a retry mechanism for load_markets
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                exchange.load_markets()
                logger.info(f"Markets loaded for {exchange.id}.")
                break # Success
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    logger.warning(f"Network error loading markets (Attempt {attempt+1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.critical(f"{NEON_RED}Max retries reached loading markets for {exchange.id}. Exiting.{RESET}")
                    return None # Critical failure

        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys with balance fetch
        account_type_to_test = 'CONTRACT'
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            balance = exchange.fetch_balance(params={'type': account_type_to_test})
            available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
            logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
        except ccxt.AuthenticationError as auth_err:
             logger.critical(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             logger.critical(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
             return None
        except ccxt.ExchangeError as balance_err:
             logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
             try:
                  balance = exchange.fetch_balance()
                  available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
                  logger.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
             except Exception as fallback_err:
                  logger.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type if trading fails.{RESET}")
        except Exception as balance_err:
             logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type if trading fails.{RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.critical(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.critical(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

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
            # Prioritize 'last', then mid-price, then ask/bid
            last_price = ticker.get('last')
            bid_price = ticker.get('bid')
            ask_price = ticker.get('ask')

            if last_price is not None:
                try:
                    p = Decimal(str(last_price))
                    if p > 0: price = p; lg.debug(f"Using 'last' price: {p}")
                except Exception: lg.warning(f"Invalid 'last' price: {last_price}")

            if price is None and bid_price is not None and ask_price is not None:
                try:
                    bid = Decimal(str(bid_price))
                    ask = Decimal(str(ask_price))
                    if bid > 0 and ask > 0:
                        price = (bid + ask) / 2
                        lg.debug(f"Using bid/ask midpoint: {price}")
                except Exception: lg.warning(f"Invalid bid/ask: {bid_price}, {ask_price}")

            if price is None and ask_price is not None: # Fallback to ask
                 try:
                      p = Decimal(str(ask_price));
                      if p > 0: price = p; lg.warning(f"Using 'ask' price fallback: {p}")
                 except Exception: lg.warning(f"Invalid 'ask' price: {ask_price}")

            if price is None and bid_price is not None: # Fallback to bid
                 try:
                      p = Decimal(str(bid_price));
                      if p > 0: price = p; lg.warning(f"Using 'bid' price fallback: {p}")
                 except Exception: lg.warning(f"Invalid 'bid' price: {bid_price}")

            if price is not None and price > 0:
                # Ensure price is quantized to market precision
                try:
                    # Requires market info access which isn't available here easily.
                    # Rely on exchange.price_to_precision later if needed for orders.
                    # For simple price checks/comparisons, Decimal is sufficient.
                    return price
                except Exception: # If quantization fails, return raw decimal
                     lg.warning(f"Failed to quantize price {price}, returning raw Decimal.")
                     return price
            else:
                lg.warning(f"Failed to get a valid positive price from ticker attempt {attempts + 1}.")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting longer ({RETRY_DELAY_SECONDS*5}s)...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 5)
            attempts += 1 # Count this attempt
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Don't retry on most exchange errors (e.g., bad symbol)
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch a valid current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = DEFAULT_FETCH_LIMIT, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__)
    try:
        if not exchange.has['fetchOHLCV']:
             lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
             return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
             try:
                  lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                  if ohlcv is not None and len(ohlcv) > 0: # Basic check if data was returned
                    break # Success
                  else:
                    lg.warning(f"fetch_ohlcv returned None or empty list for {symbol} (Attempt {attempt+1}). Retrying...")

             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                      # Raise the error only if it's not the last attempt
                      if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS) # Small delay before final raise
                      else: raise e
             except ccxt.RateLimitExceeded as e:
                 lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {RETRY_DELAY_SECONDS * 5}s... (Attempt {attempt+1})")
                 time.sleep(RETRY_DELAY_SECONDS * 5)
             except ccxt.ExchangeError as e:
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 # Depending on the error, might not be retryable
                 raise e # Re-raise non-network errors immediately

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        # Bybit returns newest first, reverse for chronological order
        df = df.iloc[::-1].reset_index(drop=True) # Reset index after reversing

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{RESET}")
             return pd.DataFrame()

        df.sort_index(inplace=True) # Ensure index is sorted
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df.tail(MAX_DF_LEN).copy() # Return a copy of the tail to keep DF size limited

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching klines for {symbol} after final retry: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error processing klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision, limits, contract type."""
    lg = logger
    try:
        if not exchange.markets or symbol not in exchange.markets:
             lg.info(f"Market info for {symbol} not loaded, reloading markets...")
             exchange.load_markets(reload=True)

        if symbol not in exchange.markets:
             lg.error(f"{NEON_RED}Market {symbol} still not found after reloading.{RESET}")
             return None

        market = exchange.market(symbol)
        if market:
            market_type = market.get('type', 'unknown')
            contract_type = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "Spot/Other"
            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract Type={contract_type}, "
                f"Precision(Price/Amount): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']
            return market
        else:
             lg.error(f"{NEON_RED}Market dictionary not found for {symbol}.{RESET}")
             return None
    except ccxt.BadSymbol as e:
         lg.error(f"{NEON_RED}Symbol '{symbol}' not supported or invalid: {e}{RESET}")
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency."""
    lg = logger
    try:
        balance_info = None
        account_types_to_try = ['CONTRACT', 'UNIFIED']
        found_structure = False

        for acc_type in account_types_to_try:
             try:
                 lg.debug(f"Fetching balance using params={{'type': '{acc_type}'}} for {currency}...")
                 balance_info = exchange.fetch_balance(params={'type': acc_type})
                 if currency in balance_info and balance_info[currency].get('free') is not None:
                     found_structure = True; break
                 elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                     for account in balance_info['info']['result']['list']:
                         if isinstance(account.get('coin'), list):
                             if any(coin_data.get('coin') == currency for coin_data in account['coin']):
                                 found_structure = True; break
                     if found_structure: break
                 lg.debug(f"Currency '{currency}' not directly found using type '{acc_type}'.")
             except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                 lg.debug(f"Error fetching balance for type '{acc_type}': {e}. Trying next.")
                 continue
             except Exception as e:
                 lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}. Trying next.")
                 continue

        if not found_structure:
             lg.debug(f"Fetching balance using default parameters for {currency}...")
             try: balance_info = exchange.fetch_balance()
             except Exception as e: lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}"); return None

        available_balance_str = None
        if currency in balance_info and balance_info[currency].get('free') is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ['{currency}']['free']: {available_balance_str}")
        elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
            for account in balance_info['info']['result']['list']:
                if isinstance(account.get('coin'), list):
                    for coin_data in account['coin']:
                         if coin_data.get('coin') == currency:
                             free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                             if free is not None: available_balance_str = str(free); break
                    if available_balance_str is not None: break
            if available_balance_str: lg.debug(f"Found balance via Bybit V5 nested: {available_balance_str}")
            else: lg.warning(f"{currency} not found within V5 'info.result.list[].coin[]'.")
        elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
             available_balance_str = str(balance_info['free'][currency])
             lg.debug(f"Found balance via top-level 'free' dict: {available_balance_str}")

        if available_balance_str is None:
             total_balance = balance_info.get(currency, {}).get('total')
             if total_balance is not None:
                  lg.warning(f"{NEON_YELLOW}Using 'total' balance ({total_balance}) as fallback for {currency}.{RESET}")
                  available_balance_str = str(total_balance)
             else:
                  lg.error(f"{NEON_RED}Could not determine any balance for {currency}.{RESET}")
                  lg.debug(f"Full balance_info structure: {balance_info}")
                  return None

        try:
            final_balance = Decimal(available_balance_str)
            if final_balance >= 0:
                 lg.info(f"Available {currency} balance: {final_balance:.4f}")
                 return final_balance
            else:
                 lg.error(f"Parsed balance for {currency} is negative ({final_balance}).")
                 return None
        except Exception as e:
            lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
            return None

    except Exception as e:
        lg.error(f"{NEON_RED}Critical error during balance fetch for {currency}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions."""
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        fetch_all = False

        try: positions = exchange.fetch_positions([symbol])
        except ccxt.ArgumentsRequired: fetch_all = True
        except ccxt.ExchangeError as e:
             no_pos_codes_v5 = [110025]
             if "symbol not found" in str(e).lower() or (hasattr(e, 'code') and e.code in no_pos_codes_v5):
                  lg.info(f"No position found for {symbol} (Exchange confirmed: {e}).")
                  return None
             lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=False); return None
        except Exception as e:
             lg.error(f"Error fetching single position for {symbol}: {e}", exc_info=True); return None

        if fetch_all:
            try:
                 all_positions = exchange.fetch_positions()
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total, found {len(positions)} matching {symbol}.")
            except Exception as e: lg.error(f"Error fetching all positions for {symbol}: {e}", exc_info=True); return None

        active_position = None
        size_threshold = Decimal('1e-9')

        for pos in positions:
            pos_size_str = None
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size'])

            if pos_size_str is None: continue

            try:
                position_size = Decimal(pos_size_str)
                if abs(position_size) > size_threshold:
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break
            except Exception: continue

        if active_position:
            size_decimal = Decimal(str(active_position.get('contracts', active_position.get('info',{}).get('size', '0'))))
            side = active_position.get('side')
            if side not in ['long', 'short']:
                if size_decimal > size_threshold: side = 'long'
                elif size_decimal < -size_threshold: side = 'short'
                else: lg.warning(f"Position size {size_decimal} near zero, cannot determine side."); return None
                active_position['side'] = side

            info_dict = active_position.get('info', {})
            if active_position.get('stopLossPrice') is None: active_position['stopLossPrice'] = info_dict.get('stopLoss')
            if active_position.get('takeProfitPrice') is None: active_position['takeProfitPrice'] = info_dict.get('takeProfit')
            active_position['trailingStopLoss'] = info_dict.get('trailingStop') # Distance (value)
            active_position['tslActivationPrice'] = info_dict.get('activePrice') # Activation price


            def format_log_val(val, precision=6, is_size=False):
                if val is None or str(val).strip() == '' or str(val) == '0' or str(val) == '0.0': return 'N/A'
                try:
                     d_val = Decimal(str(val))
                     if is_size:
                         try: market = exchange.market(symbol); amount_prec = abs(Decimal(str(market['precision']['amount'])).normalize().as_tuple().exponent)
                         except: amount_prec = 8
                         return f"{abs(d_val):.{amount_prec}f}"
                     else:
                         try: market = exchange.market(symbol); price_prec = abs(Decimal(str(market['precision']['price'])).normalize().as_tuple().exponent)
                         except: price_prec = precision
                         return f"{d_val:.{price_prec}f}"
                except Exception: return str(val)

            entry_price = format_log_val(active_position.get('entryPrice', info_dict.get('avgPrice')))
            contracts = format_log_val(active_position.get('contracts', info_dict.get('size')), is_size=True)
            liq_price = format_log_val(active_position.get('liquidationPrice'))
            leverage_str = active_position.get('leverage', info_dict.get('leverage'))
            leverage = f"{Decimal(leverage_str):.1f}x" if leverage_str is not None and Decimal(leverage_str) > 0 else 'N/A'
            pnl_str = active_position.get('unrealizedPnl')
            pnl = format_log_val(pnl_str, 4)
            sl_price = format_log_val(active_position.get('stopLossPrice'))
            tp_price = format_log_val(active_position.get('takeProfitPrice'))
            tsl_dist = format_log_val(active_position.get('trailingStopLoss'))
            tsl_act = format_log_val(active_position.get('tslActivationPrice'))

            logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                        f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                        f"Lev={leverage}, PnL={pnl}, SL={sl_price}, TP={tp_price}, "
                        f"TSL(Dist/Act): {tsl_dist}/{tsl_act}")
            logger.debug(f"Full position details for {symbol}: {active_position}")
            return active_position
        else:
            logger.info(f"No active open position found for {symbol}.")
            return None

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract).")
        return True
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage ({leverage}).")
        return False
    if not exchange.has.get('setLeverage'):
         lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage via CCXT.{RESET}")
         return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        if 'bybit' in exchange.id.lower():
             params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
             lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")
        lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True
        elif bybit_code in [110028, 110009, 110055] or "margin mode" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) compatibility with leverage setting.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed risk limit tier. Check Bybit Risk Limits.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for {symbol}. Check allowed range.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange,
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size based on risk, SL distance, balance, and market constraints.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else base_currency

    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid balance ({balance}).")
        return None
    if not (Decimal(str(risk_per_trade)) > 0 and Decimal(str(risk_per_trade)) < 1):
         lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}).")
         return None
    if initial_stop_loss_price is None or initial_stop_loss_price <= 0 or \
       entry_price is None or entry_price <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid entry ({entry_price}) or SL ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
         lg.error(f"Position sizing failed ({symbol}): SL price equals entry price.")
         return None
    if 'limits' not in market_info or 'precision' not in market_info:
         lg.error(f"Position sizing failed ({symbol}): Market info missing limits/precision.")
         return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
             lg.error(f"Position sizing failed ({symbol}): SL distance zero/negative ({sl_distance_per_unit}).")
             return None

        contract_size_str = market_info.get('contractSize', '1')
        try: contract_size = Decimal(str(contract_size_str)); assert contract_size > 0
        except Exception: lg.warning(f"Invalid contract size '{contract_size_str}', using 1."); contract_size = Decimal('1')

        if market_info.get('linear', True) or not is_contract:
             calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)
        else:
             lg.warning(f"Inverse contract detected for {symbol}. Sizing calculation assumes standard inverse logic (may be inaccurate)!")
             # Simplified inverse calc placeholder
             calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)


        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount')

        min_amount_str = amount_limits.get('min')
        max_amount_str = amount_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')

        min_cost_str = cost_limits.get('min')
        max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')

        adjusted_size = calculated_size
        adjusted_size = max(min_amount, min(adjusted_size, max_amount))
        if adjusted_size != calculated_size:
             lg.warning(f"{NEON_YELLOW}Size adjusted by amount limits: {calculated_size:.8f} -> {adjusted_size:.8f} {size_unit}{RESET}")

        current_cost = adjusted_size * entry_price * contract_size
        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Estimated Cost={current_cost:.4f} {quote_currency}")

        if min_cost > 0 and current_cost < min_cost :
             lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} below min cost {min_cost:.4f}. Attempting to increase size.{RESET}")
             required_size_for_min_cost = min_cost / (entry_price * contract_size) if entry_price > 0 and contract_size > 0 else None
             if required_size_for_min_cost is None: lg.error("Cannot calculate size for min cost."); return None
             lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
             if required_size_for_min_cost > max_amount: lg.error(f"{NEON_RED}Cannot meet min cost {min_cost:.4f} without exceeding max amount {max_amount:.8f}. Aborted.{RESET}"); return None
             elif required_size_for_min_cost < min_amount: lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f}, but min amount is {min_amount:.8f}. Aborted.{RESET}"); return None
             else: adjusted_size = required_size_for_min_cost

        elif max_cost > 0 and current_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} exceeds max cost {max_cost:.4f}. Reducing size.{RESET}")
             adjusted_size_for_max_cost = max_cost / (entry_price * contract_size) if entry_price > 0 and contract_size > 0 else None
             if adjusted_size_for_max_cost is None: lg.error("Cannot calculate size for max cost."); return None
             lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f}")
             if adjusted_size_for_max_cost < min_amount: lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is below min amount {min_amount:.8f}. Aborted.{RESET}"); return None
             else: adjusted_size = adjusted_size_for_max_cost

        try:
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision ({fmt_err}). Using manual rounding based on precision.{RESET}")
            amount_prec = precision.get('amount')
            if amount_prec is not None:
                 step_size = None
                 if isinstance(amount_prec, (float, str)):
                      try: step_size = Decimal(str(amount_prec)); assert step_size > 0
                      except: pass
                 elif isinstance(amount_prec, int) and amount_prec >= 0:
                      step_size = Decimal('1e-' + str(amount_prec))

                 if step_size is not None and step_size > 0:
                      final_size = (adjusted_size // step_size) * step_size
                      lg.info(f"Applied manual amount step size ({step_size}): {adjusted_size:.8f} -> {final_size} {size_unit}")
                 else:
                      lg.warning(f"{NEON_YELLOW}Invalid amount precision value '{amount_prec}'. Using unrounded size adjusted for limits.{RESET}")
                      final_size = adjusted_size
            else:
                 lg.warning(f"{NEON_YELLOW}Amount precision not defined. Using unrounded size adjusted for limits.{RESET}")
                 final_size = adjusted_size


        if final_size <= 0: lg.error(f"{NEON_RED}Position size became zero/negative ({final_size}) after adjustments. Aborted.{RESET}"); return None
        if final_size < min_amount: lg.error(f"{NEON_RED}Final size {final_size} is below minimum amount {min_amount}. Aborted.{RESET}"); return None
        final_cost = final_size * entry_price * contract_size
        if min_cost > 0 and final_cost < min_cost: lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} below minimum cost {min_cost:.4f}. Aborted.{RESET}"); return None


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
    Uses reduce_only flag for closing orders.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')
    action = "Close" if reduce_only else "Open/Increase"

    try:
        amount_float = float(position_size)
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} {action}): Invalid position size ({amount_float}).")
            return None
    except Exception as e:
        lg.error(f"Trade aborted ({symbol} {side} {action}): Failed to convert size {position_size} to float: {e}")
        return None

    params = {
        'positionIdx': 0,
        'reduceOnly': reduce_only,
    }
    if reduce_only:
        params['timeInForce'] = 'IOC'
        lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")
    else:
        lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")

    lg.info(f"  Size: {amount_float:.8f} {size_unit}")
    lg.debug(f"  Params: {params}")

    try:
        order = exchange.create_order(
            symbol=symbol, type=order_type, side=side,
            amount=amount_float, price=None, params=params
        )
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A')
        lg.info(f"{NEON_GREEN}{action} Trade Placed Successfully! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} {action}): {order}")
        return order

    except ccxt.InsufficientFunds as e: lg.error(f"{NEON_RED}Insufficient funds to {action} {side} order ({symbol}): {e}{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters to {action} {side} order ({symbol}): {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        if reduce_only and bybit_code == 110014: lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only order failed. Position might already be closed, size incorrect, or API issue?{RESET}")
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error placing {action} order ({symbol}): {e}{RESET}")
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action} order ({symbol}): {e} (Code: {bybit_code}){RESET}")
        if reduce_only and bybit_code == 110025: lg.warning(f"{NEON_YELLOW} >> Hint (110025): Position might have been closed already when trying to place reduce-only order.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing {action} order ({symbol}): {e}{RESET}", exc_info=True)

    return None

def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None,
    tsl_activation_price: Optional[Decimal] = None,
) -> bool:
    """
    Internal helper to set SL, TP, or TSL for an existing position using Bybit's V5 API.
    Handles parameter formatting and API call.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract).")
        return False
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    pos_side = position_info.get('side')
    if pos_side not in ['long', 'short']:
         lg.error(f"Cannot set protection for {symbol}: Invalid position side ('{pos_side}').")
         return False

    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    # TSL needs both distance > 0 AND activation price > 0
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0)


    if not has_sl and not has_tp and not has_tsl:
         lg.info(f"No valid protection parameters provided for {symbol}. No protection set/updated.")
         return True

    category = 'linear' if market_info.get('linear', True) else 'inverse'
    position_idx = 0
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
    except Exception: lg.warning(f"Could not parse positionIdx, using {position_idx}.")

    params = {
        'category': category,
        'symbol': market_info['id'],
        'tpslMode': 'Full',
        'slTriggerBy': 'LastPrice',
        'tpTriggerBy': 'LastPrice',
        'slOrderType': 'Market',
        'tpOrderType': 'Market',
        'positionIdx': position_idx
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} PosIdx: {position_idx}):"]

    try:
        # Need market info to get precision for formatting prices/distances
        price_precision_places = abs(Decimal(str(market_info.get('precision', {}).get('price', '0.0001'))).normalize().as_tuple().exponent)
        # Tick size is precision.price value itself
        min_tick_size = Decimal(str(market_info.get('precision', {}).get('price', '0.0001')))
        if min_tick_size <= 0: min_tick_size = Decimal(str(10**(-price_precision_places))) # Fallback if precision.price is int or invalid

        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try: return exchange.price_to_precision(symbol, float(price_decimal))
            except Exception as e: lg.warning(f"Failed to format price {price_decimal}: {e}"); return None

        if has_tsl:
            try:
                 # TSL distance is treated as a price value and needs price precision
                 formatted_tsl_distance = exchange.decimal_to_precision(
                     trailing_stop_distance, exchange.ROUND, precision=price_precision_places, padding_mode=exchange.NO_PADDING
                 )
                 if Decimal(formatted_tsl_distance) <= 0: # Ensure minimum distance is positive
                      formatted_tsl_distance = str(min_tick_size) if min_tick_size > 0 else '0.0001' # Fallback to tiny value
                      lg.warning(f"Calculated TSL distance {trailing_stop_distance} <= 0, setting to min tick/fallback {formatted_tsl_distance}")

            except Exception as e:
                 lg.warning(f"Failed to format TSL distance {trailing_stop_distance}: {e}"); formatted_tsl_distance = None

            formatted_activation_price = format_price(tsl_activation_price)

            if formatted_tsl_distance and Decimal(formatted_tsl_distance) > 0 and formatted_activation_price:
                params['trailingStop'] = formatted_tsl_distance
                params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                has_sl = False # Setting TSL overrides fixed SL on Bybit

        if has_sl:
            formatted_sl = format_price(stop_loss_price)
            if formatted_sl:
                params['stopLoss'] = formatted_sl
                log_parts.append(f"  Fixed SL: {formatted_sl}")
            else: has_sl = False

        if has_tp:
            formatted_tp = format_price(take_profit_price)
            if formatted_tp:
                params['takeProfit'] = formatted_tp
                log_parts.append(f"  Fixed TP: {formatted_tp}")
            else: has_tp = False

    except Exception as fmt_err:
         lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
        lg.warning(f"No valid protection parameters could be formatted or remain after adjustments for {symbol}. No API call made.")
        return False

    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={params})")

    try:
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {})

        if ret_code == 0:
            if "not modified" in ret_msg.lower():
                 lg.info(f"{NEON_YELLOW}Position protection already set to target values or partially modified for {symbol}. Response: {ret_msg}{RESET}")
            else:
                 lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            if ret_code == 110013 and "parameter error" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint (110013): Check SL/TP prices vs entry, TSL dist/act prices, tick size compliance.{RESET}")
            elif ret_code == 110036: lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price likely invalid (already passed, wrong side, too close?).{RESET}")
            elif ret_code == 110086: lg.error(f"{NEON_YELLOW} >> Hint (110086): SL price cannot equal TP price.{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: TSL distance invalid (too small/large/violates tick?).{RESET}")
            return False

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during protection API call for {symbol}: {e}{RESET}", exc_info=True)
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None
) -> bool:
    """
    Calculates TSL parameters and calls the internal helper to set TSL (and optionally TP).
    """
    lg = logger
    protection_cfg = config.get("protection", {})
    if not protection_cfg.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss disabled in config for {symbol}. Skipping TSL setup.")
        return False

    try:
        callback_rate = Decimal(str(protection_cfg.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(protection_cfg.get("trailing_stop_activation_percentage", 0.003)))
    except Exception as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format in config ({symbol}): {e}. Cannot calculate TSL.{RESET}")
        return False
    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_callback_rate ({callback_rate}) for {symbol}.{RESET}")
        return False
    if activation_percentage < 0:
         lg.error(f"{NEON_RED}Invalid trailing_stop_activation_percentage ({activation_percentage}) for {symbol}.{RESET}")
         return False

    try:
        entry_price_str = position_info.get('entryPrice') or position_info.get('info', {}).get('avgPrice')
        side = position_info.get('side')
        if entry_price_str is None or side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing required position info (entryPrice, side) for TSL calc ({symbol}).{RESET}")
            return False
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0:
             lg.error(f"{NEON_RED}Invalid entry price ({entry_price}) for TSL calc ({symbol}).{RESET}")
             return False
    except Exception as e:
        lg.error(f"{NEON_RED}Error parsing position info for TSL calculation ({symbol}): {e}.{RESET}")
        return False

    try:
        # Need market info to get precision for formatting prices/distances
        price_precision_places = abs(Decimal(str(market_info.get('precision', {}).get('price', '0.0001'))).normalize().as_tuple().exponent)
        min_tick_size = Decimal(str(market_info.get('precision', {}).get('price', '0.0001')))
        if min_tick_size <= 0: min_tick_size = Decimal(str(10**(-price_precision_places)))


        # 1. Calculate Activation Price
        activation_price = None
        activation_offset = entry_price * activation_percentage
        if side == 'long':
            raw_activation = entry_price + activation_offset
            activation_price = raw_activation.quantize(min_tick_size, rounding=ROUND_UP)
            if activation_percentage > 0 and activation_price <= entry_price: activation_price = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
            elif activation_percentage == 0: activation_price = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
        else: # short
            raw_activation = entry_price - activation_offset
            activation_price = raw_activation.quantize(min_tick_size, rounding=ROUND_DOWN)
            if activation_percentage > 0 and activation_price >= entry_price: activation_price = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)
            elif activation_percentage == 0: activation_price = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)

        if activation_price is None or activation_price <= 0:
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) invalid for {symbol}.{RESET}")
             return False

        # 2. Calculate Trailing Stop Distance
        trailing_distance_raw = activation_price * callback_rate
        # Round distance UP to nearest tick size increment
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        if trailing_distance < min_tick_size: trailing_distance = min_tick_size
        if trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance zero/negative ({trailing_distance}) for {symbol}.{RESET}")
             return False

        lg.info(f"Calculated TSL Params for {symbol} ({side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision_places}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price: {activation_price:.{price_precision_places}f}")
        lg.info(f"  => Trailing Distance: {trailing_distance:.{price_precision_places}f}")
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0:
             lg.info(f"  Take Profit Price: {take_profit_price:.{price_precision_places}f} (Will be set alongside TSL)")


        # 3. Call helper to set TSL (and TP)
        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
            stop_loss_price=None,
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy ---

class OrderBlock(TypedDict):
    id: str # Unique ID (e.g., bar_index timestamp string)
    type: str # 'bull' or 'bear'
    left_idx: pd.Timestamp # DatetimeIndex of bar where OB formed
    right_idx: pd.Timestamp # DatetimeIndex of last bar OB is valid for
    top: Decimal # Use Decimal for price levels
    bottom: Decimal
    active: bool # Still considered valid?
    violated: bool # Was it violated?

class StrategyAnalysisResults(TypedDict):
    dataframe: pd.DataFrame # DataFrame with all indicator calculations
    last_close: Decimal # Latest close price as Decimal
    current_trend_up: Optional[bool] # True=UP, False=DOWN, None=Undetermined
    trend_just_changed: bool # True if trend changed on the last completed bar
    active_bull_boxes: List[OrderBlock]
    active_bear_boxes: List[OrderBlock]
    vol_norm_int: Optional[int] # Latest Volume Norm as int
    atr: Optional[Decimal] # Latest ATR as Decimal
    upper_band: Optional[Decimal] # Latest upper band as Decimal
    lower_band: Optional[Decimal] # Latest lower band as Decimal


class VolumaticOBStrategy:
    """
    Calculates Volumatic Trend and Pivot Order Blocks based on Pine Script logic.
    Generates analysis results including trend state and active OBs.
    """

    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        """Initializes the analyzer with configuration parameters."""
        self.config = config
        self.market_info = market_info
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})

        # --- Store Config ---
        self.trend_length = strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH)
        self.atr_length = strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD)
        self.vol_ema_length = strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH)
        self.vol_atr_mult = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))
        self.vol_step_atr_mult = Decimal(str(strategy_cfg.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER)))

        self.ob_source = strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE)
        self.ph_left = strategy_cfg.get("ph_left", DEFAULT_PH_LEFT)
        self.ph_right = strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT)
        self.pl_left = strategy_cfg.get("pl_left", DEFAULT_PL_LEFT)
        self.pl_right = strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT)
        self.ob_extend = strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND)
        self.ob_max_boxes = strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES)

        # --- State Variables (will be populated during analysis) ---
        # Store boxes persistently between updates
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Minimum data length required for all indicators
        self.min_data_len = max(
             self.trend_length * 2, # Allow for initial ema/swma calculation
             self.atr_length,
             self.vol_ema_length,
             self.ph_left + self.ph_right + 1,
             self.pl_left + self.pl_right + 1
         ) + 5 # Add a small buffer


        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy Engine...{RESET}")
        self.logger.info(f"Params: TrendLen={self.trend_length}, ATRLen={self.atr_length}, VolLen={self.vol_ema_length}")
        self.logger.info(f"OB: Source={self.ob_source}, PH={self.ph_left}/{self.ph_right}, PL={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxBoxes={self.ob_max_boxes}")
        self.logger.info(f"Minimum data points recommended: {self.min_data_len}")


    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates the custom SWMA then EMA from the Pine Script."""
        if len(series) < 4:
            return pd.Series(np.nan, index=series.index)

        # Pine Script's swma(src) is ta.sma(ta.wma(src, 4), 1) or similar,
        # but the weights [1, 2, 2, 1] / 6 suggest a specific weighted average.
        # Using rolling apply with explicit weights mapping Pine's x[0]...x[3] to t, t-1...t-3
        weights = np.array([1, 2, 2, 1]) / 6.0
        # Need to reverse the series for dot product if rolling window is in natural order
        # Pandas rolling window default is 'right' (inclusive of current).
        # To apply weights [w0, w1, w2, w3] to [src[t], src[t-1], src[t-2], src[t-3]],
        # and rolling window gives [src[t-3], src[t-2], src[t-1], src[t]],
        # we need to dot product with [w3, w2, w1, w0].
        reversed_weights = weights[::-1]

        # Use min_periods to match Pine's behavior where function output starts later
        # min_periods=4 ensures 4 points are available for SWMA
        swma = series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, reversed_weights), raw=True)

        # Calculate the EMA of the SWMA series
        # Need min_periods for EMA as well
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan) # fillna=np.nan propagates NaNs

        return ema_of_swma

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
        """
        Finds pivot points similar to Pine Script's ta.pivothigh/low.
        A pivot occurs at index `i` if the condition holds for `left` bars
        before it and `right` bars after it. The pivot value is reported
        at index `i`.
        """
        if self.ob_source == "Wicks":
             source_col = 'high' if is_high else 'low'
        else: # Bodys
             # Pine uses close for PH body, open for PL body based on common OB defs
             source_col = 'close' if is_high else 'open'

        if source_col not in df.columns:
            self.logger.error(f"Source column '{source_col}' not found in DataFrame for pivot calculation.")
            return pd.Series(np.nan, index=df.index)

        source_series = df[source_col]
        pivots = pd.Series(np.nan, index=df.index) # Store the pivot price at the pivot index

        # Iterate through potential pivot points.
        # A point at index `i` can be a pivot only if there are `left` bars before
        # and `right` bars after it within the current series length.
        # Adjusted range: Check bars from `left` up to `len(df) - 1 - right`.
        start_idx = left
        end_idx = len(df) - 1 - right

        for i in range(start_idx, end_idx + 1):
            pivot_val = source_series.iloc[i]
            if pd.isna(pivot_val): continue # Skip if pivot value is NaN

            is_pivot = True

            # Check left side (indices i-left to i-1)
            # Check values are strictly LESS THAN pivot_val for HIGH pivot
            # Check values are strictly GREATER THAN pivot_val for LOW pivot
            for j in range(1, left + 1):
                left_val = source_series.iloc[i - j]
                if pd.isna(left_val): continue # Ignore NaN neighbors in comparison
                if (is_high and left_val > pivot_val) or \
                   (not is_high and left_val < pivot_val):
                    is_pivot = False
                    break # Not a pivot, move to next i
            if not is_pivot:
                continue

            # Check right side (indices i+1 to i+right)
            # Check values are LESS THAN OR EQUAL TO pivot_val for HIGH pivot
            # Check values are GREATER THAN OR EQUAL TO pivot_val for LOW pivot
            for j in range(1, right + 1):
                right_val = source_series.iloc[i + j]
                if pd.isna(right_val): continue # Ignore NaN neighbors in comparison
                if (is_high and right_val >= pivot_val) or \
                   (not is_high and right_val <= pivot_val):
                    is_pivot = False
                    break # Not a pivot, move to next i

            if is_pivot:
                # Store the actual pivot price at the index where the pivot occurred
                pivots.iloc[i] = pivot_val

        return pivots

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Calculates all strategy components based on the input DataFrame.
        Processes all data, including potential new pivots and box status updates.

        Args:
            df_input: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                      index must be DatetimeIndex, sorted chronologically. Should include
                      enough historical data as required by min_data_len.

        Returns:
            StrategyAnalysisResults dictionary containing latest state and active boxes.
            Returns a partial dictionary if analysis fails early.
        """
        # Work on a copy to avoid modifying the original DataFrame passed from outside
        df = df_input.copy()

        # Ensure data is sorted and index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             self.logger.error("DataFrame index is not DatetimeIndex.")
             df.index = pd.to_datetime(df.index)
             df.sort_index(inplace=True)
        if not df.index.is_monotonic_increasing:
             self.logger.warning("DataFrame index is not monotonically increasing. Sorting.")
             df.sort_index(inplace=True)

        # Ensure data is long enough
        if len(df) < self.min_data_len:
            self.logger.warning(f"{NEON_YELLOW}Not enough data points ({len(df)}/{self.min_data_len}) for full strategy analysis. Expect NaNs.{RESET}")
            # Return partial empty results
            return StrategyAnalysisResults(
                dataframe=df, last_close=Decimal(str(df['close'].iloc[-1])) if not df.empty and pd.notna(df['close'].iloc[-1]) else Decimal('0'),
                current_trend_up=None, trend_just_changed=False, active_bull_boxes=[],
                active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None
            )

        self.logger.debug(f"Analyzing {len(df)} candles for strategy.")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length, fillna=np.nan)
        df['ema1'] = self._ema_swma(df['close'], length=self.trend_length)
        df['ema2'] = ta.ema(df['close'], length=self.trend_length, fillna=np.nan)

        # Trend Detection: Pine script uses `ema1[1] < ema2` for `UpTrend`
        # In vectorized pandas, `df['ema1'].shift(1) < df['ema2']` corresponds to this logic for the *current* bar
        df['trend_up'] = df['ema1'].shift(1) < df['ema2']
        # This will have NaN at the start where shift(1) is NaN. Forward fill the boolean trend.
        df['trend_up'] = df['trend_up'].ffill()

        # Detect trend change based on the boolean trend series
        # Check current trend vs previous trend (ignoring NaNs at the very start)
        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) & \
                              df['trend_up'].notna() & df['trend_up'].shift(1).notna()
        df['trend_changed'].fillna(False, inplace=True) # Fill initial NaNs with False

        # --- Stateful Band Calculation ---
        # New band levels are set ONLY on a trend change, based on EMA1 and ATR at the change point.
        # We need to find the values of EMA1 and ATR *at the bar where trend_changed is True*.
        # Then, these values are held forward until the *next* trend change.
        atr_mult = self.vol_atr_mult
        step_atr_mult = self.vol_step_atr_mult

        # Create temporary columns holding EMA1/ATR only at change points
        df['ema1_at_change'] = np.where(df['trend_changed'], df['ema1'], np.nan)
        df['atr_at_change'] = np.where(df['trend_changed'], df['atr'], np.nan)

        # Forward fill these values to get the EMA1/ATR relevant for current band calculations
        df['ema1_for_bands'] = df['ema1_at_change'].ffill()
        df['atr_for_bands'] = df['atr_at_change'].ffill()

        # Calculate the band levels using the ffilled EMA1 and ATR
        df['upper_band'] = df['ema1_for_bands'] + df['atr_for_bands'] * atr_mult
        df['lower_band'] = df['ema1_for_bands'] - df['atr_for_bands'] * atr_mult

        # Drop rows where bands are still NaN (period before the first trend change occurs)
        initial_len_before_band_drop = len(df)
        df.dropna(subset=['upper_band', 'lower_band'], inplace=True)
        if len(df) < initial_len_before_band_drop:
             self.logger.debug(f"Dropped {initial_len_before_band_drop - len(df)} rows before first trend change for band calc.")

        if df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame empty after calculating trend bands. Not enough data for a trend change?{RESET}")
            # Return partial empty results
            return StrategyAnalysisResults(
                 dataframe=df, last_close=Decimal('0'), current_trend_up=None, trend_just_changed=False,
                 active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None
            )


        # --- Volume Calculation & Normalization ---
        # Calculate rolling percentile
        df['vol_percentile_val'] = df['volume'].rolling(
            window=self.vol_ema_length,
            min_periods=min(self.vol_ema_length // 2, 50) # Require reasonable number of periods
        ).apply(
            lambda x: np.percentile(x[x.notna() & (x > 0)], 100) if np.any(x.notna() & (x > 0)) else np.nan, # 100th percentile = max
            raw=True
        )
        # Normalize volume (0-100 range relative to recent max)
        # Avoid division by zero or NaN
        df['vol_norm'] = (df['volume'] / df['vol_percentile_val'] * 100)
        df['vol_norm'].fillna(0, inplace=True) # Treat NaN volume norm as 0
        df['vol_norm'] = df['vol_norm'].clip(0, 200) # Clip extreme values, similar to gradient cap

        # Calculate volume levels for plotting/potential future use
        # Ensure levels and steps are valid numbers (use the ffilled ATR for steps)
        df['lower_vol_ref'] = df['lower_band'] + df['atr_for_bands'] * step_atr_mult
        df['upper_vol_ref'] = df['upper_band'] - df['atr_for_bands'] * step_atr_mult

        df['step_up_size'] = (df['lower_vol_ref'] - df['lower_band']) / 100
        df['step_dn_size'] = (df['upper_band'] - df['upper_vol_ref']) / 100
        df['step_up_size'] = df['step_up_size'].clip(lower=0)
        df['step_dn_size'] = df['step_dn_size'].clip(lower=0)

        df['vol_step_up_offset'] = (df['step_up_size'] * df['vol_norm']).fillna(0)
        df['vol_step_dn_offset'] = (df['step_dn_size'] * df['vol_norm']).fillna(0)

        # Calculate the top/bottom of the volume bars (relative to the bands)
        df['vol_bar_up_top'] = df['lower_band'] + df['vol_step_up_offset']
        df['vol_bar_dn_bottom'] = df['upper_band'] - df['vol_step_dn_offset']

        self.logger.debug("Volumatic Trend calculations complete.")


        # --- Pivot Order Block Calculations & Management ---
        # Find pivots *on the full dataset* available in this update cycle
        if self.ob_source == "Wicks":
            high_series = df['high']
            low_series = df['low']
        else: # "Bodys"
            high_series = df[['open', 'close']].max(axis=1)
            low_series = df[['open', 'close']].min(axis=1)

        # `ta.pivot` returns 1 at the index where the pivot BAR ends (confirmed)
        df['ph_signal'] = ta.pivot(high_series, left=self.ph_left, right=self.ph_right, high_low='high')
        df['pl_signal'] = ta.pivot(low_series, left=low_series, right=self.pl_right, high_low='low') # Fix: use low_series for pivot calc

        # Find indices where new pivot signals appear in the *latest part* of the DF
        # Only need to check the most recent bars, as older pivots would have been found already
        check_recent_bars = max(self.ph_right, self.pl_right) + 5 # Check few bars back
        recent_df = df.tail(check_recent_bars)

        # Create new boxes for pivot signals confirmed in the `recent_df`
        for idx in recent_df.index:
            # Pivot High signal at `idx` means the pivot BAR occurred at `idx - ph_right`
            if pd.notna(recent_df.loc[idx, 'ph_signal']):
                pivot_bar_idx = idx - pd.Timedelta(minutes=int(self.config.get("interval", "5")) * self.ph_right) # Approximate datetime index
                # Ensure pivot_bar_idx is within the current df's index range
                if pivot_bar_idx >= df.index.min() and pivot_bar_idx in df.index: # Check if index exists
                    # Check if a box starting at this pivot bar index already exists
                    if not any(b['left_idx'] == pivot_bar_idx for b in self.bear_boxes):
                        pivot_bar_loc = df.index.get_loc(pivot_bar_idx) # Get integer location
                        ob_candle = df.iloc[pivot_bar_loc]
                        box_top = ob_candle['high'] if self.ob_source == "Wicks" else ob_candle['close']
                        box_bottom = ob_candle['close'] if self.ob_source == "Wicks" else ob_candle['open']
                        if box_bottom > box_top: box_top, box_bottom = box_bottom, box_top # Swap

                        if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                            self.bear_boxes.append({
                                'id': f"bear_{pivot_bar_idx.strftime('%Y%m%d%H%M')}",
                                'type': 'bear',
                                'left_idx': pivot_bar_idx,
                                'right_idx': df.index[-1], # Extend to current bar initially
                                'top': Decimal(str(box_top)),
                                'bottom': Decimal(str(box_bottom)),
                                'active': True,
                                'violated': False
                            })
                            self.logger.debug(f"{NEON_RED}New Bearish OB created at {pivot_bar_idx}.{RESET}")


            # Pivot Low signal at `idx` means the pivot BAR occurred at `idx - pl_right`
            if pd.notna(recent_df.loc[idx, 'pl_signal']):
                 pivot_bar_idx = idx - pd.Timedelta(minutes=int(self.config.get("interval", "5")) * self.pl_right)
                 if pivot_bar_idx >= df.index.min() and pivot_bar_idx in df.index:
                     if not any(b['left_idx'] == pivot_bar_idx for b in self.bull_boxes):
                         pivot_bar_loc = df.index.get_loc(pivot_bar_idx)
                         ob_candle = df.iloc[pivot_bar_loc]
                         box_top = ob_candle['open'] if self.ob_source == "Wicks" else ob_candle['open']
                         box_bottom = ob_candle['low'] if self.ob_source == "Wicks" else ob_candle['close']
                         if box_bottom > box_top: box_top, box_bottom = box_bottom, box_top

                         if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                             self.bull_boxes.append({
                                 'id': f"bull_{pivot_bar_idx.strftime('%Y%m%d%H%M')}",
                                 'type': 'bull',
                                 'left_idx': pivot_bar_idx,
                                 'right_idx': df.index[-1],
                                 'top': Decimal(str(box_top)),
                                 'bottom': Decimal(str(box_bottom)),
                                 'active': True,
                                 'violated': False
                             })
                             self.logger.debug(f"{NEON_GREEN}New Bullish OB created at {pivot_bar_idx}.{RESET}")
        self.logger.debug(f"Current OB Counts: {len(self.bull_boxes)} Bull, {len(self.bear_boxes)} Bear (Total)")

        # --- Manage existing boxes (close or extend) ---
        if not df.empty and pd.notna(df['close'].iloc[-1]):
            last_close = Decimal(str(df['close'].iloc[-1]))
            last_bar_idx = df.index[-1]

            for box in self.bull_boxes:
                if box['active']:
                    if last_close < box['bottom']: # Price closed below bull box
                        box['active'] = False
                        box['violated'] = True
                        box['right_idx'] = last_bar_idx # Close the box visually
                        self.logger.debug(f"Bull Box {box['id']} violated at {last_bar_idx}.")
                    elif self.ob_extend: # Extend if active and enabled
                        box['right_idx'] = last_bar_idx

            for box in self.bear_boxes:
                if box['active']:
                    if last_close > box['top']: # Price closed above bear box
                        box['active'] = False
                        box['violated'] = True
                        box['right_idx'] = last_bar_idx
                        self.logger.debug(f"Bear Box {box['id']} violated at {last_bar_idx}.")
                    elif self.ob_extend:
                        box['right_idx'] = last_bar_idx

        # --- Prune Order Blocks ---
        # Sort by activity status (active=True first) then by index (newest first)
        self.bull_boxes.sort(key=lambda b: (b['active'], b['left_idx']), reverse=True)
        # Keep max_boxes active ones and a few recent inactive ones for context/display
        active_bull = [b for b in self.bull_boxes if b['active']]
        inactive_bull = [b for b in self.bull_boxes if not b['active']]
        self.bull_boxes = active_bull[:self.ob_max_boxes] + inactive_bull[:self.ob_max_boxes // 2] # Keep fewer inactive

        self.bear_boxes.sort(key=lambda b: (b['active'], b['left_idx']), reverse=True)
        active_bear = [b for b in self.bear_boxes if b['active']]
        inactive_bear = [b for b in self.bear_boxes if not b['active']]
        self.bear_boxes = active_bear[:self.ob_max_boxes] + inactive_bear[:self.ob_max_boxes // 2]

        self.logger.debug(f"Active OB Counts: {len(active_bull)} Bull, {len(active_bear)} Bear.")


        # --- Prepare Results ---
        last_row = df.iloc[-1] if not df.empty else {}
        latest_close = Decimal(str(last_row.get('close', '0'))) if pd.notna(last_row.get('close')) else Decimal('0')
        latest_atr = Decimal(str(last_row.get('atr', '0'))) if pd.notna(last_row.get('atr')) else Decimal('0')
        latest_upper_band = Decimal(str(last_row.get('upper_band', '0'))) if pd.notna(last_row.get('upper_band')) else Decimal('0')
        latest_lower_band = Decimal(str(last_row.get('lower_band', '0'))) if pd.notna(last_row.get('lower_band')) else Decimal('0')


        results = StrategyAnalysisResults(
            dataframe=df, # Return the DataFrame with all calcs
            last_close=latest_close,
            current_trend_up=bool(last_row.get('trend_up', False)) if pd.notna(last_row.get('trend_up')) else None,
            trend_just_changed=bool(last_row.get('trend_changed', False)),
            active_bull_boxes=[b for b in self.bull_boxes if b['active']],
            active_bear_boxes=[b for b in self.bear_boxes if b['active']],
            vol_norm_int=int(last_row.get('vol_norm', 0)) if pd.notna(last_row.get('vol_norm')) else 0,
            atr=latest_atr,
            upper_band=latest_upper_band,
            lower_band=latest_lower_band
        )

        # Log key results
        trend_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] else f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else f"{NEON_YELLOW}N/A{RESET}"
        self.logger.debug(f"Strategy Results ({df.index[-1]}): Close={results['last_close']:.4f}, Trend={trend_str}, "
                          f"TrendChg={results['trend_just_changed']}, ATR={results['atr']:.4f}, "
                          f"Active Bull OBs={len(results['active_bull_boxes'])}, Active Bear OBs={len(results['active_bear_boxes'])}")


        return results

# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """Generates BUY/SELL/HOLD/EXIT signals based on VolumaticOB Strategy results."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})
        self.ob_entry_proximity_factor = Decimal(str(strategy_cfg.get("ob_entry_proximity_factor", 1.005)))
        self.ob_exit_proximity_factor = Decimal(str(strategy_cfg.get("ob_exit_proximity_factor", 1.001)))

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """
        Determines the trading signal based on the latest strategy analysis results
        and the current open position.

        Args:
            analysis_results: The output dictionary from VolumaticOBStrategy.update().
            open_position: The CCXT position dictionary or None if no position.

        Returns:
            "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
        """
        if not analysis_results or analysis_results['dataframe'].empty or analysis_results['current_trend_up'] is None or analysis_results['last_close'] <= 0:
            self.logger.warning(f"{NEON_YELLOW}Insufficient/invalid strategy analysis results to generate signal.{RESET}")
            return "HOLD"

        latest_close = analysis_results['last_close']
        is_trend_up = analysis_results['current_trend_up']
        trend_just_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        atr = analysis_results['atr'] # ATR from the strategy calc

        current_pos_side = open_position.get('side') if open_position else None # 'long', 'short', or None


        # --- Signal Logic ---
        signal = "HOLD" # Default signal

        # 1. Check for Exit Signal (Trend Change or Price Violating OB edge)
        if current_pos_side == 'long':
            # Exit Long on Trend Flip DOWN
            if not is_trend_up and trend_just_changed:
                 signal = "EXIT_LONG"
                 self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG: Trend flipped DOWN.{RESET}")
            # Or Exit Long if price approaches/crosses a recent Bearish OB top (resistance)
            # Find the newest active bear box
            newest_bear_ob = sorted(active_bear_obs, key=lambda x: x['left_idx'], reverse=True)[:1]
            if newest_bear_ob:
                bear_ob = newest_bear_ob[0]
                # Check if close is at or above the top of the newest active bear OB (with buffer)
                if latest_close >= bear_ob['top'] * (Decimal("1") - (self.ob_exit_proximity_factor - 1)): # Check slightly below top edge
                     signal = "EXIT_LONG"
                     self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG: Price ({latest_close:.4f}) approached/crossed Bear OB {bear_ob['id']} top ({bear_ob['top']:.4f}).{RESET}")
            # Optional: Exit Long if price closes significantly below a relevant level (e.g., EMA, band)

        elif current_pos_side == 'short':
             # Exit Short on Trend Flip UP
             if is_trend_up and trend_just_changed:
                 signal = "EXIT_SHORT"
                 self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT: Trend flipped UP.{RESET}")
             # Or Exit Short if price approaches/crosses a recent Bullish OB bottom (support)
             # Find the newest active bull box
             newest_bull_ob = sorted(active_bull_obs, key=lambda x: x['left_idx'], reverse=True)[:1]
             if newest_bull_ob:
                 bull_ob = newest_bull_ob[0]
                 # Check if close is at or below the bottom of the newest active bull OB (with buffer)
                 if latest_close <= bull_ob['bottom'] * (Decimal("1") + (self.ob_exit_proximity_factor - 1)): # Check slightly above bottom edge
                      signal = "EXIT_SHORT"
                      self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT: Price ({latest_close:.4f}) approached/crossed Bull OB {bull_ob['id']} bottom ({bull_ob['bottom']:.4f}).{RESET}")
             # Optional: Exit Short if price closes significantly above a relevant level

        # If an exit signal was generated, return it immediately. Do not check for entries.
        if signal in ["EXIT_LONG", "EXIT_SHORT"]:
             return signal

        # 2. Check for Entry Signal (Only if not already in a position)
        if current_pos_side is None:
            # Entry conditions:
            # a) Trend matches desired direction
            # b) Price is currently within or near an *active* Order Block of the matching type

            if is_trend_up: # Looking for BUY signal (Long Entry)
                # Look for price interacting with active Bullish OBs
                for ob in active_bull_obs:
                    # Check if latest close is within or slightly above the Bull OB range
                    if ob['bottom'] <= latest_close <= ob['top'] * self.ob_entry_proximity_factor:
                        signal = "BUY"
                        self.logger.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price ({latest_close:.4f}) in/near Bull OB {ob['id']} ({ob['bottom']:.4f}-{ob['top']:.4f}){RESET}")
                        break # Found an entry signal, stop checking other OBs

            elif not is_trend_up: # Looking for SELL signal (Short Entry)
                 # Look for price interacting with active Bearish OBs
                 for ob in active_bear_obs:
                     # Check if latest close is within or slightly below the Bear OB range
                     if ob['bottom'] * (Decimal("1") - (self.ob_entry_proximity_factor - 1)) <= latest_close <= ob['top']:
                         signal = "SELL"
                         self.logger.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price ({latest_close:.4f}) in/near Bear OB {ob['id']} ({ob['bottom']:.4f}-{ob['top']:.4f}){RESET}")
                         break # Found an entry signal

        # If signal is still HOLD, log why (optional, for debug)
        if signal == "HOLD" and current_pos_side is None:
             self.logger.debug(f"HOLD: Trend is {'UP' if is_trend_up else 'DOWN' if is_trend_up is False else 'N/A'}. Price ({latest_close:.4f}) not in/near active OBs.")


        return signal

    def calculate_initial_tp_sl(
        self, entry_price: Decimal, signal: str, atr: Optional[Decimal], market_info: Dict
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial TP and SL levels based on entry price, ATR, and config multipliers.
        Used for position sizing and initial order protection.
        Returns (take_profit, stop_loss), both Decimal or None.
        """
        if signal not in ["BUY", "SELL"] or atr is None or atr <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate initial TP/SL: Invalid signal ({signal}) or ATR ({atr}).{RESET}")
            return None, None
        if entry_price is None or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate initial TP/SL: Invalid entry price ({entry_price}).{RESET}")
            return None, None

        protection_cfg = self.config.get("protection", {})
        try:
            tp_multiple = Decimal(str(protection_cfg.get("initial_take_profit_atr_multiple", 0.7)))
            sl_multiple = Decimal(str(protection_cfg.get("initial_stop_loss_atr_multiple", 1.8)))

            # Need market info to get precision
            price_precision_places = abs(Decimal(str(market_info.get('precision', {}).get('price', '0.0001'))).normalize().as_tuple().exponent)
            min_tick_size = Decimal(str(market_info.get('precision', {}).get('price', '0.0001')))
            if min_tick_size <= 0: min_tick_size = Decimal(str(10**(-price_precision_places)))

            take_profit = None
            stop_loss = None
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            if signal == "BUY":
                tp_raw = entry_price + tp_offset
                sl_raw = entry_price - sl_offset
                take_profit = tp_raw.quantize(min_tick_size, rounding=ROUND_UP)
                stop_loss = sl_raw.quantize(min_tick_size, rounding=ROUND_DOWN)
            elif signal == "SELL":
                tp_raw = entry_price - tp_offset
                sl_raw = entry_price + sl_offset
                take_profit = tp_raw.quantize(min_tick_size, rounding=ROUND_DOWN)
                stop_loss = sl_raw.quantize(min_tick_size, rounding=ROUND_UP)

            # Validation: Ensure SL is strictly beyond entry by at least one tick
            if signal == "BUY" and stop_loss >= entry_price:
                 stop_loss = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)
                 self.logger.debug(f"Adjusted BUY SL below entry: {stop_loss}")
            elif signal == "SELL" and stop_loss <= entry_price:
                 stop_loss = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
                 self.logger.debug(f"Adjusted SELL SL above entry: {stop_loss}")

            # Validation: Ensure TP is potentially profitable (strictly beyond entry)
            if signal == "BUY" and take_profit <= entry_price: take_profit = None
            elif signal == "SELL" and take_profit >= entry_price: take_profit = None

            # Validation: Ensure SL/TP are positive
            if stop_loss is not None and stop_loss <= 0: stop_loss = None
            if take_profit is not None and take_profit <= 0: take_profit = None

            self.logger.debug(f"Calculated Initial TP/SL: TP={take_profit}, SL={stop_loss}")
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
    signal_generator: SignalGenerator # Pass instantiated signal generator
) -> None:
    """Analyzes a single symbol and executes/manages trades based on strategy signals and config."""

    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Market Info & Data ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed market info for {symbol}. Skipping cycle.{RESET}")
        return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}'. Cannot map to CCXT timeframe.")
         return

    fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    if fetch_limit < strategy_engine.min_data_len:
         lg.warning(f"{NEON_YELLOW}Configured fetch_limit ({fetch_limit}) is less than strategy's minimum required data ({strategy_engine.min_data_len}). Adjusting fetch_limit.{RESET}")
         fetch_limit = strategy_engine.min_data_len + 50 # Add some buffer

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit, logger=lg)
    if klines_df.empty or len(klines_df) < strategy_engine.min_data_len:
        lg.error(f"{NEON_RED}Failed sufficient kline data for {symbol} (fetched {len(klines_df)}, need {strategy_engine.min_data_len}). Skipping cycle.{RESET}")
        return

    # --- 2. Run Strategy Analysis ---
    analysis_results = strategy_engine.update(klines_df)
    if not analysis_results or analysis_results['dataframe'].empty or analysis_results['current_trend_up'] is None or analysis_results['last_close'] <= 0:
         lg.error(f"{NEON_RED}Strategy analysis failed or produced insufficient results for {symbol}. Skipping signal.{RESET}")
         return

    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr']

    # --- 3. Check Position & Generate Signal ---
    open_position = get_open_position(exchange, symbol, lg) # Returns dict or None

    # Generate the trading signal based on analysis results and current position
    signal = signal_generator.generate_signal(analysis_results, open_position)

    # --- 4. Trading Logic ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***")

            # --- Pre-Trade Checks ---
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative.{RESET}")
                return

            # Calculate initial SL/TP for sizing purposes using the strategy's ATR
            _, initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(
                 entry_price=latest_close, signal=signal, atr=current_atr, market_info=market_info
            )

            if initial_sl_calc is None or initial_sl_calc <= 0:
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed ({initial_sl_calc}). Cannot size position.{RESET}")
                 return

            # Set Leverage (only for contracts)
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage.{RESET}")
                         return
            else: lg.info(f"Leverage setting skipped (Spot).")

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
            lg.info(f"==> Placing {signal} market order | Size: {position_size} <==")
            trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)

            # --- Post-Trade: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                lg.info(f"Attempting confirmation for {symbol} after order {order_id}...")
                # Re-fetch position state
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        entry_price_actual_str = confirmed_position.get('entryPrice') or confirmed_position.get('info', {}).get('avgPrice')
                        entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else latest_close # Fallback

                        lg.info(f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_price_actual:.{abs(latest_close.normalize().as_tuple().exponent)}f}{RESET}")

                        # Recalculate TP based on ACTUAL entry price (SL is managed later by BE/TSL)
                        # Use the same initial TP logic but with actual entry
                        tp_final, _ = signal_generator.calculate_initial_tp_sl(
                             entry_price=entry_price_actual, signal=signal, atr=current_atr, market_info=market_info
                        )

                        protection_set_success = False
                        protection_cfg = config.get("protection", {})

                        if protection_cfg.get("enable_trailing_stop", False):
                             lg.info(f"Setting Trailing Stop Loss (TP target: {tp_final})...")
                             protection_set_success = set_trailing_stop_loss(
                                 exchange=exchange, symbol=symbol, market_info=market_info, position_info=confirmed_position,
                                 config=config, logger=lg, take_profit_price=tp_final
                             )
                        elif protection_cfg.get("initial_stop_loss_atr_multiple", 0) > 0 or protection_cfg.get("initial_take_profit_atr_multiple", 0) > 0:
                             # If TSL is off but fixed SL/TP multipliers exist, calculate fixed SL based on actual entry & ATR
                             _, sl_final = signal_generator.calculate_initial_tp_sl(
                                  entry_price=entry_price_actual, signal=signal, atr=current_atr, market_info=market_info
                             )
                             lg.info(f"Setting Fixed SL ({sl_final}) and TP ({tp_final})...")
                             if sl_final or tp_final:
                                 protection_set_success = _set_position_protection(
                                     exchange=exchange, symbol=symbol, market_info=market_info, position_info=confirmed_position,
                                     logger=lg, stop_loss_price=sl_final, take_profit_price=tp_final
                                 )
                             else: lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed based on actual entry. No fixed protection set.{RESET}")

                        if protection_set_success:
                             lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                        else:
                             lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ===")
                             lg.warning(f"{NEON_YELLOW}MANUAL MONITORING REQUIRED! Position open without automated protection.{RESET}")

                    except Exception as post_trade_err:
                         lg.error(f"{NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                         lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")
                else:
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM position after {confirm_delay}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order may have failed/rejected, or API delay. Manual investigation required!{RESET}")
            else:
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). See logs. ===")
        else: # signal == HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No action.")

    # --- Scenario 2: Existing Open Position ---
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        lg.info(f"Existing {pos_side.upper()} position found for {symbol}.")

        # Check for Exit Signal
        exit_signal_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered for {symbol} ({signal}). Closing... ***{RESET}")
            try:
                close_side = 'sell' if pos_side == 'long' else 'buy'
                size_to_close_str = open_position.get('contracts') or open_position.get('info',{}).get('size')
                if size_to_close_str is None: raise ValueError("Cannot determine position size to close.")
                size_to_close = abs(Decimal(str(size_to_close_str)))
                if size_to_close <= 0: raise ValueError(f"Position size {size_to_close_str} invalid.")

                lg.info(f"==> Placing {close_side.upper()} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                # Use "SELL" signal for long close, "BUY" for short close as per place_trade
                trade_signal_for_close = "SELL" if pos_side == 'long' else "BUY"
                close_order = place_trade(exchange, symbol, trade_signal_for_close, size_to_close, market_info, lg, reduce_only=True)

                if close_order:
                    lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check required!{RESET}")

            except Exception as close_err:
                 lg.error(f"{NEON_RED}Error closing position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")

        else: # Hold signal or signal matches position direction
            lg.info(f"Signal ({signal}) allows holding existing {pos_side} position.")

            # --- Manage Existing Position (BE, TSL checks) ---
            protection_cfg = config.get("protection", {})
            is_tsl_active = False
            try: is_tsl_active = Decimal(str(open_position.get('trailingStopLoss', '0'))) > 0
            except: pass # Ignore parsing error for TSL check

            # --- Break-Even Check (only if BE enabled AND TSL is NOT active) ---
            if protection_cfg.get("enable_break_even", False) and not is_tsl_active:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    entry_price_str = open_position.get('entryPrice') or open_position.get('info', {}).get('avgPrice')
                    entry_price = Decimal(str(entry_price_str))
                    if current_atr is None or current_atr <= 0:
                        lg.warning("BE Check skipped: Invalid ATR.")
                    else:
                        profit_target_atr = Decimal(str(protection_cfg.get("break_even_trigger_atr_multiple", 1.0)))
                        offset_ticks = int(protection_cfg.get("break_even_offset_ticks", 2))

                        price_precision_places = abs(Decimal(str(market_info.get('precision', {}).get('price', '0.0001'))).normalize().as_tuple().exponent)
                        min_tick_size = Decimal(str(market_info.get('precision', {}).get('price', '0.0001')))
                        if min_tick_size <= 0: min_tick_size = Decimal(str(10**(-price_precision_places)))

                        price_diff = (latest_close - entry_price) if pos_side == 'long' else (entry_price - latest_close)
                        profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('0')

                        lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision_places}f}, Profit ATRs={profit_in_atr:.2f}, Target ATRs={profit_target_atr}")

                        if profit_in_atr >= profit_target_atr:
                            tick_offset = min_tick_size * offset_ticks
                            be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP) if pos_side == 'long' \
                                       else (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                            current_sl_price = None
                            current_sl_str = open_position.get('stopLossPrice') or open_position.get('info', {}).get('stopLoss')
                            if current_sl_str and str(current_sl_str) != '0':
                                try: current_sl_price = Decimal(str(current_sl_str))
                                except Exception: pass

                            update_be = False
                            if be_stop_price > 0:
                                if current_sl_price is None: update_be = True; lg.info("BE triggered: No current SL found.")
                                elif pos_side == 'long' and be_stop_price > current_sl_price: update_be = True; lg.info(f"BE triggered: Target {be_stop_price} > Current SL {current_sl_price}.")
                                elif pos_side == 'short' and be_stop_price < current_sl_price: update_be = True; lg.info(f"BE triggered: Target {be_stop_price} < Current SL {current_sl_price}.")
                                else: lg.debug(f"BE Triggered, but current SL ({current_sl_price}) already better than target ({be_stop_price}).")

                            if update_be:
                                lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                                current_tp_price = None
                                current_tp_str = open_position.get('takeProfitPrice') or open_position.get('info', {}).get('takeProfit')
                                if current_tp_str and str(current_tp_str) != '0':
                                     try: current_tp_price = Decimal(str(current_tp_str))
                                     except Exception: pass

                                success = _set_position_protection(
                                    exchange, symbol, market_info, open_position, lg,
                                    stop_loss_price=be_stop_price,
                                    take_profit_price=current_tp_price
                                )
                                if success: lg.info(f"{NEON_GREEN}Break-Even SL set successfully.{RESET}")
                                else: lg.error(f"{NEON_RED}Failed to set Break-Even SL.{RESET}")
                        else:
                            lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {profit_target_atr} ATRs).")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check ({symbol}): {be_err}{RESET}", exc_info=True)
            elif is_tsl_active:
                 lg.info(f"Break-even check skipped: Trailing Stop Loss is active.")
            elif not protection_cfg.get("enable_break_even", False):
                 lg.debug(f"Break-even check skipped: Disabled in config.")

            # --- Trailing Stop Logic (if enabled) ---
            # If TSL is enabled, ensure it IS active. If not, attempt to set it.
            # This handles cases where TSL might fail on entry or get cancelled.
            if protection_cfg.get("enable_trailing_stop", False) and not is_tsl_active:
                 lg.info(f"TSL enabled but not active. Attempting to set TSL...")
                 # Recalculate TP based on actual entry for consistency, pass to set_trailing_stop_loss
                 entry_price_actual_str = open_position.get('entryPrice') or open_position.get('info', {}).get('avgPrice')
                 entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else latest_close # Fallback
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(
                      entry_price=entry_price_actual, signal=pos_side.upper(), atr=current_atr, market_info=market_info # Use position side as signal
                 )
                 set_trailing_stop_loss(
                     exchange=exchange, symbol=symbol, market_info=market_info, position_info=open_position,
                     config=config, logger=lg, take_profit_price=tp_recalc
                 )


    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY

    setup_logger("init")
    init_logger = logging.getLogger("init")

    init_logger.info(f"--- Starting Pyrmethus Volumatic Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    CONFIG = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
    init_logger.info(f"Config loaded. Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={ta.version if hasattr(ta, 'version') else 'N/A'}")

    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet).{RESET}")
         else:
              init_logger.warning(f"{NEON_RED}!!! USING REAL MONEY ENVIRONMENT !!!{RESET}")
         protection_cfg = CONFIG.get("protection", {})
         init_logger.warning(f"Settings: Risk={CONFIG.get('risk_per_trade', 0)*100:.2f}%, Lev={CONFIG.get('leverage', 0)}x, TSL={'ON' if protection_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if protection_cfg.get('enable_break_even') else 'OFF'}")
         try:
             input(f">>> Review settings. Press {NEON_GREEN}Enter{RESET} to continue, or {NEON_RED}Ctrl+C{RESET} to abort... ")
             init_logger.info("User confirmed live trading settings.")
         except KeyboardInterrupt: init_logger.info("User aborted."); return
    else:
         init_logger.info(f"{NEON_YELLOW}Trading is disabled. Running in analysis-only mode.{RESET}")

    init_logger.info("Initializing exchange...")
    exchange = initialize_exchange(init_logger)
    if not exchange: init_logger.critical(f"{NEON_RED}Failed exchange initialization. Exiting.{RESET}"); return
    init_logger.info(f"Exchange {exchange.id} initialized.")

    target_symbol = None
    while True:
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter symbol (e.g., BTC/USDT): {RESET}").strip().upper()
            if not symbol_input_raw: continue
            symbol_input = symbol_input_raw.replace('-', '/')

            init_logger.info(f"Validating symbol '{symbol_input}'...")
            market_info = get_market_info(exchange, symbol_input, init_logger)

            if market_info:
                target_symbol = market_info['symbol']
                market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                break
            else:
                variations_to_try = []
                if '/' in symbol_input:
                    base, quote = symbol_input.split('/', 1)
                    variations_to_try.append(f"{base}/{quote}:{quote}")

                found_variation = False
                if variations_to_try:
                     init_logger.info(f"Symbol '{symbol_input}' not found. Trying variations: {variations_to_try}")
                     for sym_var in variations_to_try:
                         market_info = get_market_info(exchange, sym_var, init_logger)
                         if market_info:
                             target_symbol = market_info['symbol']
                             market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                             init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                             found_variation = True
                             break
                if found_variation: break
                else: init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' and variations not validated.{RESET}")

        except Exception as e: init_logger.error(f"Error validating symbol: {e}", exc_info=True)

    selected_interval = None
    while True:
        interval_input = input(f"{NEON_YELLOW}Enter interval [{'/'.join(VALID_INTERVALS)}] (default: {CONFIG['interval']}): {RESET}").strip()
        if not interval_input: interval_input = CONFIG['interval']
        if interval_input in VALID_INTERVALS: # Validate against our list
             selected_interval = interval_input
             CONFIG["interval"] = selected_interval
             # Ensure CCXT map entry exists, though validated against VALID_INTERVALS first
             ccxt_tf = CCXT_INTERVAL_MAP.get(selected_interval)
             if ccxt_tf:
                  init_logger.info(f"Using interval: {selected_interval} (CCXT: {ccxt_tf})")
                  break
             else: # Should not happen if VALID_INTERVALS maps correctly, but as a safeguard
                  init_logger.error(f"{NEON_RED}Internal Error: Interval '{selected_interval}' not mapped to CCXT timeframe.{RESET}")

        else: init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Choose from {VALID_INTERVALS}.{RESET}")


    symbol_logger = setup_logger(target_symbol)
    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    protection_cfg = CONFIG.get("protection", {})
    symbol_logger.info(f"Config: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if protection_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if protection_cfg.get('enable_break_even') else 'OFF'}, Trading={'ENABLED' if CONFIG['enable_trading'] else 'DISABLED'}")
    symbol_logger.info(f"Strategy Params: {CONFIG.get('strategy_params', {})}")

    # Instantiate Strategy Engine and Signal Generator once
    strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
    signal_generator = SignalGenerator(CONFIG, symbol_logger)


    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Start: {datetime.now(TIMEZONE).strftime('%H:%M:%S')}")

            try:
                # --- Optional Config Reload ---
                # CONFIG = load_config(CONFIG_FILE) # Reload config if needed dynamically
                # QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
                # protection_cfg = CONFIG.get("protection", {})
                # symbol_logger.debug("Config reloaded (if uncommented).")
                # --- End Optional Reload ---

                analyze_and_trade_symbol(
                    exchange=exchange,
                    symbol=target_symbol,
                    config=CONFIG, # Pass potentially reloaded config
                    logger=symbol_logger,
                    strategy_engine=strategy_engine, # Pass the strategy engine
                    signal_generator=signal_generator # Pass the signal generator
                )
            except ccxt.RateLimitExceeded as e: symbol_logger.warning(f"{NEON_YELLOW}Rate limit: {e}. Waiting 60s...{RESET}"); time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.RequestException) as e: symbol_logger.error(f"{NEON_RED}Network error: {e}. Waiting {RETRY_DELAY_SECONDS*3}s...{RESET}"); time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"{NEON_RED}CRITICAL Auth Error: {e}. Stopping.{RESET}"); break
            except ccxt.ExchangeNotAvailable as e: symbol_logger.error(f"{NEON_RED}Exchange unavailable: {e}. Waiting 60s...{RESET}"); time.sleep(60)
            except ccxt.OnMaintenance as e: symbol_logger.error(f"{NEON_RED}Exchange Maintenance: {e}. Waiting 5m...{RESET}"); time.sleep(300)
            except Exception as loop_error: symbol_logger.error(f"{NEON_RED}Uncaught loop error: {loop_error}{RESET}", exc_info=True); time.sleep(15)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS) - elapsed_time) # Use config delay
            symbol_logger.debug(f"<<< Loop cycle finished in {elapsed_time:.2f}s. Sleeping {sleep_time:.2f}s...")
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except Exception as critical_error: init_logger.critical(f"{NEON_RED}Critical unhandled error: {critical_error}{RESET}", exc_info=True)
    finally:
        shutdown_msg = f"--- Pyrmethus Volumatic Bot for {target_symbol or 'N/A'} Stopping ---"
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger): symbol_logger.info(shutdown_msg)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); exchange.close(); init_logger.info("Connection closed.")
            except Exception as close_err: init_logger.error(f"Error closing connection: {close_err}")
        logging.shutdown()
        print(f"{NEON_YELLOW}Bot stopped.{RESET}")


if __name__ == "__main__":
    # Write the enhanced script content to 'pyrmethus_volumatic_bot.py'
    output_filename = "pyrmethus_volumatic_bot.py"
    try:
        with open(__file__, 'r', encoding='utf-8') as current_file:
             script_content = current_file.read()
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            header = f"# {output_filename}\n# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy\n# with advanced position management (SL/TP, BE, TSL).\n\n"
            # Clean up potential duplicate headers/comments from previous merges/prompts
            import re
            # Remove lines starting with '# livex' or '# pyrmethus_' at the very beginning
            script_content = re.sub(r'^(# livex.*|\# pyrmethus\_.*|\# Enhanced version.*)\n', '', script_content, count=1)
            # Remove the second occurrence of the header if it exists
            script_content = re.sub(r'# pyrmethus\_volumatic\_bot\.py.*?\n', '', script_content, count=1, flags=re.DOTALL)

            output_file.write(header + script_content)
        print(f"Enhanced script content written to {output_filename}")

        # Run the main logic
        main()
    except Exception as e:
        print(f"{NEON_RED}Error writing script to {output_filename} or running main: {e}{RESET}")


