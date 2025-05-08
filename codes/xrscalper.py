```python
# xrscalper.py
# Merged and enhanced version derived from livexy.py and incorporating concepts from ScalpingBot,
# focusing on robust stop-loss/take-profit mechanisms, including break-even and exchange-native trailing stops.

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
from typing import Any, Dict, Optional, Tuple, List

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta # Import pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

# Initialize colorama and set precision
getcontext().prec = 28  # Increased precision for calculations
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
TIMEZONE = ZoneInfo("America/Chicago") # Adjust as needed
MAX_API_RETRIES = 3 # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5 # Delay between retries
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Intervals supported by the bot's logic
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable
# Default periods (can be overridden by config.json)
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
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float
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
POSITION_CONFIRM_DELAY_SECONDS = 8 # Wait time after placing order before confirming position
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
       and ensuring all default keys are present."""
    default_config = {
        "interval": "5", # Default to '5' (map to 5m later)
        "retry_delay": 5,
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
        "orderbook_limit": 25, # Depth of orderbook to fetch
        "signal_score_threshold": 1.5, # Score needed to trigger BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5, # How much higher volume needs to be than MA
        "scalping_signal_threshold": 2.5, # Separate threshold for 'scalping' weight set
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False, # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet), disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions": 1, # Limit open positions for this symbol
        "quote_currency": "USDT", # Currency for balance check and sizing
        "entry_order_type": "market", # "market" or "limit"
        "limit_order_offset_buy": 0.0005, # Percentage offset from current price for BUY limit orders (e.g., 0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005, # Percentage offset from current price for SELL limit orders (e.g., 0.0005 = 0.05%)

        # --- Trailing Stop Loss Config (Exchange-Native) ---
        "enable_trailing_stop": True,           # Default to enabling TSL (exchange TSL)
        "trailing_stop_callback_rate": 0.005,   # e.g., 0.5% trail distance (as decimal) from high water mark
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when price moves 0.3% in favor from entry (set to 0 for immediate activation)

        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR
        "break_even_offset_ticks": 2,           # Place BE SL X ticks beyond entry price (in profit direction)

        # --- Position Management ---
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order before checking position status
        "time_based_exit_minutes": None, # Optional: Exit position after X minutes (e.g., 60). Set to None or 0 to disable.

        # --- Indicator Control ---
        "indicators": { # Control which indicators are calculated and contribute to score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Flag to enable fetching and scoring orderbook data
        },
        "weight_sets": { # Define different weighting strategies
            "scalping": { # Example weighting for a fast scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default" # Choose which weight set to use ("default" or "scalping")
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            return default_config # Return default if creation failed

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
        # Ensure all keys from default are present, add missing ones
        updated_config = _ensure_config_keys(config_from_file, default_config)
        # If updates were made, write them back
        if updated_config != config_from_file:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
             except IOError as e:
                 print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

        # Validate crucial values after loading/updating
        if updated_config.get("interval") not in VALID_INTERVALS:
            print(f"{NEON_RED}Invalid interval '{updated_config.get('interval')}' found in config. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            # Save back the corrected interval
            try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{NEON_YELLOW}Corrected invalid interval in config file: {filepath}{RESET}")
            except IOError as e:
                 print(f"{NEON_RED}Error writing corrected interval to config file {filepath}: {e}{RESET}")
        if updated_config.get("entry_order_type") not in ["market", "limit"]:
             print(f"{NEON_RED}Invalid entry_order_type '{updated_config.get('entry_order_type')}' in config. Using default 'market'.{RESET}")
             updated_config["entry_order_type"] = "market"
             # Save back the corrected type
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{NEON_YELLOW}Corrected invalid entry_order_type in config file: {filepath}{RESET}")
             except IOError as e:
                 print(f"{NEON_RED}Error writing corrected entry_order_type to config file {filepath}: {e}{RESET}")

        return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        try:
            # Attempt to recreate default if loading failed badly
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
             print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config # Return default


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Optional: Check type consistency for non-dict items if needed
        # elif not isinstance(updated_config.get(key), type(default_value)):
        #     print(f"Warning: Config type mismatch for key '{key}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Using default.")
        #     updated_config[key] = default_value
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"xrscalper_bot_{safe_symbol}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Set logger level to DEBUG to capture all messages

    # File Handler (writes DEBUG level and above)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use SensitiveFormatter for file logs as well
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_filename}: {e}")

    # Stream Handler (Console - writes INFO level and above by default)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z' # Include Timezone
    )
    stream_formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple() # Use local timezone for console display
    stream_handler.setFormatter(stream_formatter)
    console_log_level = logging.INFO # Change to DEBUG for more verbose console output
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent messages from propagating to the root logger
    return logger

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
                # Increased timeouts for robustness
                'fetchTickerTimeout': 10000, # 10 seconds
                'fetchBalanceTimeout': 15000, # 15 seconds
                'createOrderTimeout': 20000, # 20 seconds
                'cancelOrderTimeout': 15000, # 15 seconds
                'fetchPositionsTimeout': 15000, # 15 seconds
                'fetchOHLCVTimeout': 15000, # 15 seconds
            }
        }

        # Dynamically select the exchange class based on config if needed, default Bybit
        exchange_id = CONFIG.get("exchange_id", "bybit").lower()
        if not hasattr(ccxt, exchange_id):
             logger.error(f"Exchange ID '{exchange_id}' not found in CCXT.")
             return None
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            # Check if the exchange supports sandbox mode setting
            if hasattr(exchange, 'set_sandbox_mode'):
                exchange.set_sandbox_mode(True)
            else:
                logger.warning(f"{exchange.id} does not support set_sandbox_mode via ccxt. Ensure API keys are for Testnet.")
                # Attempt to set URLs manually if known (example for Bybit)
                if exchange.id == 'bybit':
                     exchange.urls['api'] = 'https://api-testnet.bybit.com'


        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys by fetching balance
        account_type_to_test = 'CONTRACT' # For Bybit V5, try CONTRACT or UNIFIED
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            # Use params suitable for the specific exchange (e.g., Bybit V5)
            params = {}
            if exchange.id == 'bybit':
                 params={'type': account_type_to_test}

            balance = exchange.fetch_balance(params=params)
            available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
            logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")

        except ccxt.AuthenticationError as auth_err:
             logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on the exchange.{RESET}")
             return None
        except ccxt.ExchangeError as balance_err:
             logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
             # Fallback to default fetch_balance call
             try:
                  balance = exchange.fetch_balance()
                  available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
                  logger.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
             except Exception as fallback_err:
                  logger.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type.{RESET}")
        except Exception as balance_err:
             # Catch other potential errors like network issues during balance fetch
             logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type/network.{RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        # This catches auth errors during the initial exchange class instantiation
        logger.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        logger.error(f"{NEON_RED}>> Check API Key/Secret format and validity in your .env file.{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None


# --- CCXT Data Fetching (Adapted from livexy.py) ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            lg.debug(f"Ticker data for {symbol}: {ticker}")

            price = None
            # Prioritize 'last', then mid-price, then ask/bid
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
                    if bid > 0 and ask > 0 and ask >= bid:
                        price = (bid + ask) / 2 # Midpoint
                        lg.debug(f"Using bid/ask midpoint: {price}")
                    elif ask > 0: # If only ask is valid (e.g., thin book)
                         price = ask
                         lg.debug(f"Using 'ask' price (bid invalid): {price}")
                    elif bid > 0: # If only bid is valid
                         price = bid
                         lg.debug(f"Using 'bid' price (ask invalid): {price}")
                except Exception: lg.warning(f"Invalid bid/ask format: {bid_price}, {ask_price}")

            # Fallbacks if midpoint failed or wasn't available
            if price is None and ask_price is not None:
                 try:
                      p = Decimal(str(ask_price));
                      if p > 0: price = p; lg.warning(f"Using 'ask' price fallback: {p}")
                 except Exception: lg.warning(f"Invalid 'ask' price format: {ask_price}")

            if price is None and bid_price is not None:
                 try:
                      p = Decimal(str(bid_price));
                      if p > 0: price = p; lg.warning(f"Using 'bid' price fallback: {p}")
                 except Exception: lg.warning(f"Invalid 'bid' price format: {bid_price}")

            # Final validation
            if price is not None and price > 0:
                return price
            else:
                lg.warning(f"Failed to get a valid price from ticker data on attempt {attempts + 1}.")
                # Continue to retry logic

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting longer...{RESET}")
            # Apply a longer delay specifically for rate limit errors
            time.sleep(RETRY_DELAY_SECONDS * 5)
            attempts += 1 # Consume an attempt
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Decide if retryable based on error message/code if needed
            # For now, assume most exchange errors aren't retryable for fetching price
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry unexpected errors

        # Standard delay before next attempt
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch a valid current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    try:
        # Check if the exchange supports fetching OHLCV data
        if not exchange.has['fetchOHLCV']:
             lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
             return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
             try:
                  lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
                  # Fetch the data
                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

                  # Basic check if data was returned
                  if ohlcv is not None and isinstance(ohlcv, list): # Ensure it's a list
                    # Optional: Check if list is not empty
                    # if not ohlcv:
                    #    lg.warning(f"fetch_ohlcv returned an empty list for {symbol} (Attempt {attempt+1}). Retrying...")
                    # else:
                    break # Success
                  else:
                    lg.warning(f"fetch_ohlcv returned invalid data (None or not list) for {symbol} (Attempt {attempt+1}). Retrying...")

             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                      raise e # Re-raise the last error
             except ccxt.RateLimitExceeded as e:
                 # Use a longer delay for rate limits
                 wait_time = RETRY_DELAY_SECONDS * 5
                 lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                 time.sleep(wait_time)
             except ccxt.ExchangeError as e:
                 # Non-network/rate-limit errors from the exchange
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 # Consider if specific exchange errors are retryable, otherwise raise
                 # Example: Bad symbol error should not be retried
                 if "symbol" in str(e).lower(): raise e
                 if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                 else: raise e # Re-raise after retries
             except Exception as e:
                # Catch any other unexpected error during the fetch attempt
                lg.error(f"{NEON_RED}Unexpected error during kline fetch attempt {attempt+1} for {symbol}: {e}{RESET}", exc_info=True)
                if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                else: raise e # Re-raise after retries


        # After the loop, check if we successfully got data
        if not ohlcv or not isinstance(ohlcv, list):
            lg.warning(f"{NEON_YELLOW}No valid kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # Process the data into a pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Basic validation of the DataFrame structure
        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe}.{RESET}")
             return df

        # Convert timestamp to datetime objects
        # Use errors='coerce' to handle potential conversion errors gracefully
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        # Drop rows where timestamp conversion failed
        df.dropna(subset=['timestamp'], inplace=True)
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Convert price/volume columns to numeric, coercing errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Data Cleaning: Drop rows with NaN in essential price columns or zero close price
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Ensure close price is positive (relevant for log returns, some indicators)
        df = df[df['close'] > 0]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        # Check if DataFrame became empty after cleaning
        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{RESET}")
             return pd.DataFrame()

        # Sort by timestamp index to ensure chronological order
        df.sort_index(inplace=True)

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    # Catch errors that occur outside the retry loop (e.g., during DataFrame processing)
    except ccxt.NetworkError as e:
        # This might be hit if the initial check for fetchOHLCV fails due to network
        lg.error(f"{NEON_RED}Network error occurred during kline processing for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error occurred during kline processing for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)

    # Return an empty DataFrame in case of any caught exception during processing
    return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and basic validation."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            # Check if the exchange supports fetching order book data
            if not exchange.has['fetchOrderBook']:
                 lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                 return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts+1}/{MAX_API_RETRIES + 1})")
            orderbook = exchange.fetch_order_book(symbol, limit=limit)

            # Validate the received orderbook structure
            if not orderbook:
                lg.warning(f"fetch_order_book returned None/empty for {symbol} (Attempt {attempts+1}).")
            elif not isinstance(orderbook, dict):
                 lg.warning(f"{NEON_YELLOW}Invalid orderbook type received for {symbol}. Expected dict, got {type(orderbook)}. Attempt {attempts + 1}.{RESET}")
            elif 'bids' not in orderbook or 'asks' not in orderbook:
                 lg.warning(f"{NEON_YELLOW}Invalid orderbook structure for {symbol}: missing 'bids' or 'asks'. Attempt {attempts + 1}. Response keys: {list(orderbook.keys())}{RESET}")
            elif not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
                 lg.warning(f"{NEON_YELLOW}Invalid orderbook structure for {symbol}: 'bids' or 'asks' are not lists. Attempt {attempts + 1}. bids type: {type(orderbook['bids'])}, asks type: {type(orderbook['asks'])}{RESET}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 # It's possible to receive an empty book, especially on inactive markets
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 return orderbook # Return the empty but validly structured book
            else:
                 # Basic validation passed
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 # Optional: Deeper validation (check price/size format)
                 # for i, bid in enumerate(orderbook['bids']):
                 #     if not (isinstance(bid, list) and len(bid) == 2 and isinstance(bid[0], (float, int)) and isinstance(bid[1], (float, int))):
                 #          lg.warning(f"Invalid bid format at index {i}: {bid}")
                 #          # Handle error or continue
                 # for i, ask in enumerate(orderbook['asks']):
                 #      if not (isinstance(ask, list) and len(ask) == 2 and isinstance(ask[0], (float, int)) and isinstance(ask[1], (float, int))):
                 #          lg.warning(f"Invalid ask format at index {i}: {ask}")
                 #          # Handle error or continue
                 return orderbook

        # Handle specific CCXT exceptions
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5 # Longer wait for rate limits
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempts += 1 # Consume attempt
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            # Decide if retryable based on error message/code
            # Example: Bad symbol should not be retried
            if "symbol" in str(e).lower(): return None
            # Otherwise, retry for potential temporary exchange issues
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            # Decide whether to retry unexpected errors or not
            # For now, let's not retry these by default
            return None

        # Standard delay before next attempt
        attempts += 1
        if attempts <= MAX_API_RETRIES:
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
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Pandas DataFrame with OHLCV data, indexed by timestamp.
            logger: Logger instance for logging messages.
            config: Dictionary containing bot configuration.
            market_info: Dictionary containing market details (precision, limits, etc.).
        """
        self.df = df
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        # Get interval from config, default to '5' if missing
        self.interval = config.get("interval", "5")
        # Map to CCXT format, handle missing mapping
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)
        if not self.ccxt_interval:
             self.logger.error(f"Invalid interval '{self.interval}' in config, cannot map to CCXT timeframe. Defaulting calculation logic if possible, but fetching will fail.")
             # Handle fallback or raise error depending on desired behavior
             # For now, let it proceed but log the error.

        # Stores latest calculated indicator values (float or Decimal)
        self.indicator_values: Dict[str, Any] = {}
        # Stores binary signal states (BUY:1, SELL:1, HOLD:1) - only one can be 1
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1} # Default HOLD
        # Get the name of the active weight set from config
        self.active_weight_set_name = config.get("active_weight_set", "default")
        # Get the actual weight dictionary for the active set
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        # Stores calculated Fibonacci levels
        self.fib_levels_data: Dict[str, Decimal] = {}
        # Stores the actual column names generated by pandas_ta for mapping
        self.ta_column_names: Dict[str, Optional[str]] = {}

        if not self.weights:
             logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}.")
             # Handle this case, maybe use default weights or stop

        # Perform initial calculations upon instantiation
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Calculate Fib levels based on initial data


    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """
        Helper to find the actual column name generated by pandas_ta based on common patterns.

        Args:
            base_name: The conceptual name of the indicator (e.g., "ATR", "EMA_Short").
            result_df: The DataFrame containing the calculated indicator columns.

        Returns:
            The actual column name if found, otherwise None.
        """
        # Define expected patterns for pandas_ta column names
        # These might need adjustment based on pandas_ta version and parameters used
        expected_patterns = {
            "ATR": [f"ATRr_{self.config.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{self.config.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{self.config.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{self.config.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{self.config.get('cci_window', DEFAULT_CCI_WINDOW)}"], # May have suffix like _100.0
            "Williams_R": [f"WILLR_{self.config.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{self.config.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP_D"], # Often includes anchor like 'D' for daily
            "PSAR_long": [f"PSARl_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{self.config.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            # StochRSI patterns can vary, try common ones
            "StochRSI_K": [
                f"STOCHRSIk_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{self.config.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{self.config.get('stoch_rsi_k', DEFAULT_K_WINDOW)}",
                f"STOCHRSIk_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}" # Simpler fallback
            ],
            "StochRSI_D": [
                f"STOCHRSId_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{self.config.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{self.config.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{self.config.get('stoch_rsi_d', DEFAULT_D_WINDOW)}",
                 f"STOCHRSId_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}" # Simpler fallback
            ],
            "RSI": [f"RSI_{self.config.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            # BBands patterns often include std dev
            "BB_Lower": [
                f"BBL_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)):.1f}",
                f"BBL_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}" # Simpler fallback
            ],
            "BB_Middle": [
                f"BBM_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)):.1f}",
                f"BBM_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}"
            ],
            "BB_Upper": [
                f"BBU_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV)):.1f}",
                f"BBU_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}"
            ],
            "Volume_MA": [f"VOL_SMA_{self.config.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"] # Custom name used in calc
        }

        patterns = expected_patterns.get(base_name, [])
        # Search dataframe columns for a match starting with any pattern
        for col in result_df.columns:
             for pattern in patterns:
                 # Use startswith for flexibility (e.g., CCI might have CCI_20_100.0)
                 if col.startswith(pattern):
                     self.logger.debug(f"Mapped '{base_name}' to column '{col}'")
                     return col

        # Fallback: Simple substring search (less reliable but might catch variations)
        for col in result_df.columns:
            # Make search case-insensitive and handle simple names
            if base_name.lower() in col.lower():
                self.logger.debug(f"Found column '{col}' for base '{base_name}' using fallback substring search.")
                return col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {result_df.columns.tolist()}")
        return None


    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # --- Check Sufficient Data Length ---
        # Create a list of required periods based on enabled indicators in config
        required_periods = []
        indicators_config = self.config.get("indicators", {})
        if indicators_config.get("atr_period"): required_periods.append(self.config.get("atr_period", DEFAULT_ATR_PERIOD))
        if indicators_config.get("ema_alignment"):
            required_periods.append(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
            required_periods.append(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        if indicators_config.get("momentum"): required_periods.append(self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        if indicators_config.get("cci"): required_periods.append(self.config.get("cci_window", DEFAULT_CCI_WINDOW))
        if indicators_config.get("wr"): required_periods.append(self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        if indicators_config.get("mfi"): required_periods.append(self.config.get("mfi_window", DEFAULT_MFI_WINDOW))
        # VWAP doesn't have a standard period parameter in basic ta.vwap
        # PSAR doesn't have a length period
        if indicators_config.get("sma_10"): required_periods.append(self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW))
        if indicators_config.get("stoch_rsi"):
            required_periods.append(self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW))
            required_periods.append(self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))
            # K and D periods are applied after StochRSI calc, so main windows are limiting factors
        if indicators_config.get("rsi"): required_periods.append(self.config.get("rsi_period", DEFAULT_RSI_WINDOW))
        if indicators_config.get("bollinger_bands"): required_periods.append(self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        if indicators_config.get("volume_confirmation"): required_periods.append(self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
        # Fibonacci window for price range, not strictly an indicator period but needs data
        required_periods.append(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))

        # Determine minimum data length needed (max period + some buffer)
        min_required_data = max(required_periods) + 20 if required_periods else 50 # Default min 50 if no periods found

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators reliably (min recommended: {min_required_data}). Results may contain NaNs.{RESET}")
             # Continue calculation, but expect potential issues

        try:
            # Work on a copy to avoid modifying the original DataFrame passed to the class
            df_calc = self.df.copy()

            # --- Always calculate ATR as it's crucial for SL/TP/Sizing ---
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            # Use pandas_ta's atr method
            df_calc.ta.atr(length=atr_period, append=True)
            # Find and store the actual column name generated by ta.atr
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)
            # --- Calculate indicators based on config ---
            if indicators_config.get("ema_alignment", False):
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
                # Note: pandas_ta might add a constant suffix like '_100.0'
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
                # Standard VWAP often anchors daily ('D'). Ensure your data covers enough time.
                # If using intraday data, VWAP might reset daily or calculate cumulatively.
                # pandas_ta's default might be daily anchored. Check behavior if needed.
                df_calc.ta.vwap(append=True) # Use default settings first
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                # psar returns a DataFrame with multiple columns (long, short, af, reversal)
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                if psar_result is not None and not psar_result.empty:
                    # Concatenate the results back to the main calculation DataFrame
                    df_calc = pd.concat([df_calc, psar_result], axis=1)
                    # Get the column names for long and short PSAR signals
                    self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                    self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)

            if indicators_config.get("sma_10", False):
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                # stochrsi returns a DataFrame with K and D columns
                stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                if stochrsi_result is not None and not stochrsi_result.empty:
                     df_calc = pd.concat([df_calc, stochrsi_result], axis=1)
                     # Get the specific column names for K and D
                     self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                     self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)

            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                # Ensure std_dev is float for pandas_ta
                bb_std = float(self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV))
                # bbands returns a DataFrame with lower, middle, upper, bandwidth, percent
                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std)
                if bbands_result is not None and not bbands_result.empty:
                    # Concatenate results back
                    df_calc = pd.concat([df_calc, bbands_result], axis=1)
                    # Get column names for lower, middle, upper bands
                    self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                    self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                    self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)

            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                # Use a distinct name for the volume MA column
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}"
                # Calculate SMA on the 'volume' column, filling potential NaNs with 0 for calculation
                df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)
                # Store the custom column name
                self.ta_column_names["Volume_MA"] = vol_ma_col_name

            # Update the instance's DataFrame with the calculated indicators
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
             # This can happen if pandas_ta methods are called incorrectly or on incompatible data
             self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol}: {e}{RESET}. Check pandas_ta usage and data.", exc_info=True)
        except Exception as e:
            # Catch any other unexpected errors during calculation
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest (most recent) values from self.df."""
        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame empty for {self.symbol}.")
            # Initialize with NaNs if empty
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}
            return
        if self.df.iloc[-1].isnull().all():
            self.logger.warning(f"Cannot update latest values: Last row contains all NaNs for {self.symbol}.")
            # Initialize with NaNs if last row is unusable
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}
            return

        try:
            # Get the last row (most recent data)
            latest = self.df.iloc[-1]
            updated_values = {}

            # --- Process TA indicator columns stored in ta_column_names ---
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index:
                    value = latest[col_name]
                    # Check if the value is valid (not NaN)
                    if pd.notna(value):
                        try:
                            # Store ATR as Decimal for precision in SL/TP calculations
                            if key == "ATR":
                                updated_values[key] = Decimal(str(value))
                            # Store most other indicators as float for general use in scoring
                            else:
                                updated_values[key] = float(value)
                        except (ValueError, TypeError) as conv_err:
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) for {self.symbol}. Storing NaN. Error: {conv_err}")
                            updated_values[key] = np.nan
                    else:
                        # Store NaN if the value in the DataFrame is NaN
                        updated_values[key] = np.nan
                else:
                    # Log if the column was expected but not found
                    if key in self.ta_column_names: # Only log if calculation was attempted
                        self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found in latest data row for {self.symbol}. Storing NaN.")
                    updated_values[key] = np.nan

            # --- Add essential price/volume data as Decimal for precision ---
            # These are directly from the OHLCV data, not calculated indicators via TA
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                 key_name = base_col.capitalize() # e.g., 'Open', 'Close'
                 value = latest.get(base_col) # Use .get() for safety if column might be missing
                 if pd.notna(value):
                      try:
                           updated_values[key_name] = Decimal(str(value))
                      except (ValueError, TypeError) as conv_err:
                           self.logger.warning(f"Could not convert base value for '{base_col}' ({value}) to Decimal for {self.symbol}. Storing NaN. Error: {conv_err}")
                           updated_values[key_name] = np.nan
                 else:
                      updated_values[key_name] = np.nan # Store NaN if base value is missing/NaN

            # Update the instance's dictionary
            self.indicator_values = updated_values

            # --- Log the updated values (formatted for readability) ---
            # Create a dictionary for logging, formatting Decimals and floats appropriately
            valid_values_log = {}
            for k, v in self.indicator_values.items():
                 if pd.notna(v):
                     if isinstance(v, Decimal):
                          # Determine precision for logging (more for price/ATR, less for volume?)
                          # Use market price precision for price-related values
                          prec = self.get_price_precision() if k in ['Open','High','Low','Close','ATR'] else 6
                          # Avoid scientific notation for small numbers if possible, format nicely
                          valid_values_log[k] = f"{v:.{prec}f}"
                     elif isinstance(v, float):
                          # Format floats with reasonable precision for indicators
                          valid_values_log[k] = f"{v:.5f}"
                     else: # Handle other types if necessary (e.g., strings, though unlikely here)
                          valid_values_log[k] = str(v)
                 # Optionally include NaN values in the log
                 # else:
                 #     valid_values_log[k] = "NaN"

            self.logger.debug(f"Latest indicator values updated for {self.symbol}: {valid_values_log}")

        except IndexError:
             # This error occurs if the DataFrame is empty or iloc[-1] fails
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be empty or too short after cleaning.")
             # Reset to NaNs if access fails
             self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}
        except Exception as e:
             # Catch any other unexpected errors during the update process
             self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
             # Reset to NaNs as a safety measure
             self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """
        Calculates Fibonacci retracement levels based on the high/low over a specified window.
        Uses Decimal for precision. Stores results in self.fib_levels_data.

        Args:
            window: The number of recent periods (candles) to consider for high/low.
                    Defaults to the value in config['fibonacci_window'].

        Returns:
            A dictionary where keys are Fibonacci level names (e.g., "Fib_23.6%")
            and values are the corresponding price levels as Decimal objects.
            Returns an empty dictionary if calculation is not possible.
        """
        # Use provided window or get from config, falling back to default
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)

        # Check if DataFrame is long enough for the window
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)} points) for Fibonacci window ({window}) on {self.symbol}.")
            self.fib_levels_data = {} # Clear any previous data
            return {}

        # Get the relevant slice of the DataFrame
        df_slice = self.df.tail(window)

        try:
            # Find the maximum high and minimum low in the window, drop NaNs first
            high_price_raw = df_slice["high"].dropna().max()
            low_price_raw = df_slice["low"].dropna().min()

            # Check if valid high/low prices were found
            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                 self.logger.warning(f"Could not find valid high/low prices within the last {window} periods for Fibonacci calculation on {self.symbol}.")
                 self.fib_levels_data = {}
                 return {}

            # Convert raw high/low to Decimal
            high = Decimal(str(high_price_raw))
            low = Decimal(str(low_price_raw))

            # Calculate the difference (range)
            diff = high - low

            # Initialize the levels dictionary
            levels = {}
            # Get market price precision for rounding
            price_precision = self.get_price_precision()
            # Create a Decimal quantizer based on precision (e.g., '0.01' for 2 decimal places)
            rounding_factor = Decimal('1e-' + str(price_precision))

            # Check if there's a valid range (high > low)
            if diff > 0:
                # Calculate each Fibonacci level
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level price: High - (Range * Percentage)
                    level_price = high - (diff * Decimal(str(level_pct)))
                    # Quantize the calculated price to the market's tick size/precision
                    # Use ROUND_DOWN for levels based on range from high (conservative support)
                    level_price_quantized = level_price.quantize(rounding_factor, rounding=ROUND_DOWN)
                    levels[level_name] = level_price_quantized
            else:
                 # Handle case where high == low (no range)
                 self.logger.debug(f"Fibonacci range is zero or negative (High={high}, Low={low}) for {self.symbol} over last {window} periods. All levels set to High/Low.")
                 # Quantize the single price level
                 level_price_quantized = high.quantize(rounding_factor, rounding=ROUND_DOWN)
                 # Assign this price to all levels
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            # Store the calculated levels in the instance variable
            self.fib_levels_data = levels
            # Log the calculated levels (convert Decimals to strings for logging)
            log_levels = {k: str(v) for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} (Window: {window}): {log_levels}")
            return levels

        except KeyError as e:
             self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: Missing column '{e}'. Ensure 'high' and 'low' columns exist.{RESET}")
             self.fib_levels_data = {}
             return {}
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}


    def get_price_precision(self) -> int:
        """
        Determines the number of decimal places required for price values
        based on the market information provided by the exchange.

        Uses 'precision' and 'limits' fields from the market info.
        Falls back to inferring from the last close price or a default value.

        Returns:
            The number of decimal places (integer).
        """
        try:
            # 1. Check 'precision.price' (most common field)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')

            if price_precision_val is not None:
                 # If it's an integer, it usually represents decimal places directly
                 if isinstance(price_precision_val, int):
                      if price_precision_val >= 0:
                           self.logger.debug(f"Using price precision (decimal places) from market_info.precision.price: {price_precision_val}")
                           return price_precision_val
                 # If it's float/str, it often represents the tick size
                 elif isinstance(price_precision_val, (float, str)):
                      try:
                           tick_size = Decimal(str(price_precision_val))
                           # Ensure tick size is positive
                           if tick_size > 0:
                                # Calculate decimal places from tick size
                                # normalize() removes trailing zeros, as_tuple().exponent gives the exponent
                                precision = abs(tick_size.normalize().as_tuple().exponent)
                                self.logger.debug(f"Calculated price precision from market_info.precision.price (tick size {tick_size}): {precision}")
                                return precision
                      except Exception as e:
                           self.logger.warning(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")

            # 2. Fallback: Check 'limits.price.min' (sometimes represents tick size)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')

            if min_price_val is not None:
                 try:
                      min_price_tick = Decimal(str(min_price_val))
                      if min_price_tick > 0:
                           # Heuristic: Check if min_price looks like a tick size (small value)
                           # rather than just a minimum orderable price (e.g., 0.1).
                           # Tick sizes are usually << 1. Adjust threshold if needed.
                           if min_price_tick < Decimal('0.1'):
                                precision = abs(min_price_tick.normalize().as_tuple().exponent)
                                self.logger.debug(f"Inferred price precision from limits.price.min ({min_price_tick}): {precision}")
                                return precision
                           else:
                                self.logger.debug(f"limits.price.min ({min_price_tick}) seems too large for tick size, likely minimum order price. Ignoring for precision.")
                 except Exception as e:
                      self.logger.warning(f"Could not parse limits.price.min '{min_price_val}' for precision inference: {e}")

            # 3. Fallback: Infer from the last known close price's decimal places
            # This is less reliable as prices can fluctuate (e.g., 10.0 vs 10.123)
            last_close = self.indicator_values.get("Close") # Uses Decimal value if available
            if last_close and isinstance(last_close, Decimal) and last_close > 0:
                 try:
                      # Get the number of decimal places from the Decimal object
                      precision = abs(last_close.normalize().as_tuple().exponent)
                      self.logger.debug(f"Inferring price precision from last close price ({last_close}) as {precision} for {self.symbol}.")
                      # Add a small sanity check - avoid excessively high precision from weird prices
                      if precision < 10: # Arbitrary sanity limit
                           return precision
                      else:
                           self.logger.warning(f"Inferred precision {precision} from last close price seems too high. Skipping.")
                 except Exception as e:
                     self.logger.warning(f"Could not infer precision from last close price {last_close}: {e}")

        except Exception as e:
            self.logger.warning(f"Error determining price precision for {self.symbol} from market info: {e}. Falling back.")

        # --- Final Fallback ---
        # Use a reasonable default if no other method worked
        default_precision = 4 # Common default, adjust if needed for your typical markets
        self.logger.warning(f"Could not determine price precision for {self.symbol}. Using default: {default_precision}.")
        return default_precision


    def get_min_tick_size(self) -> Decimal:
        """
        Gets the minimum price increment (tick size) from market info using Decimal.

        Returns:
            The minimum tick size as a Decimal object. Falls back based on precision.
        """
        try:
            # 1. Try precision.price (often the tick size as float/str)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None:
                 if isinstance(price_precision_val, (float, str)):
                      try:
                           tick_size = Decimal(str(price_precision_val))
                           if tick_size > 0:
                                self.logger.debug(f"Using tick size from precision.price: {tick_size} for {self.symbol}")
                                return tick_size
                      except Exception as e:
                            self.logger.warning(f"Could not parse precision.price '{price_precision_val}' as tick size: {e}")
                 # If it's an integer (decimal places), calculate tick size
                 elif isinstance(price_precision_val, int) and price_precision_val >= 0:
                      tick_size = Decimal('1e-' + str(price_precision_val))
                      self.logger.debug(f"Calculated tick size from precision.price (decimal places {price_precision_val}): {tick_size} for {self.symbol}")
                      return tick_size

            # 2. Fallback: Try limits.price.min (sometimes represents tick size)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                try:
                    min_tick_from_limit = Decimal(str(min_price_val))
                    if min_tick_from_limit > 0:
                        # Heuristic check: if it's very small, assume it's the tick size
                        if min_tick_from_limit < Decimal('0.1'):
                             self.logger.debug(f"Using tick size from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                             return min_tick_from_limit
                        else:
                             self.logger.debug(f"limits.price.min ({min_tick_from_limit}) seems too large for tick size, potentially min order price.")
                except Exception as e:
                    self.logger.warning(f"Could not parse limits.price.min '{min_price_val}' for tick size inference: {e}")

        except Exception as e:
             self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using precision fallback.")

        # --- Final Fallback: Calculate from get_price_precision (decimal places) ---
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.debug(f"Using fallback tick size based on derived precision places ({price_precision_places}): {fallback_tick} for {self.symbol}")
        return fallback_tick


    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """
        Finds the N nearest Fibonacci levels (support and resistance) to the current price.

        Args:
            current_price: The current market price as a Decimal.
            num_levels: The maximum number of nearest levels (combined support/resistance) to return.

        Returns:
            A list of tuples, where each tuple contains (level_name, level_price).
            The list is sorted by proximity to the current price. Returns empty list on error.
        """
        # Check if Fibonacci levels have been calculated
        if not self.fib_levels_data:
            self.logger.debug(f"Fibonacci levels not calculated yet for {self.symbol}. Cannot find nearest.")
            return []
        # Validate input price
        if not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) provided for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            # Iterate through the calculated Fibonacci levels
            for name, level_price in self.fib_levels_data.items():
                # Ensure the level price is a valid Decimal
                if isinstance(level_price, Decimal) and level_price > 0:
                    # Calculate the absolute distance between current price and level price
                    distance = abs(current_price - level_price)
                    # Store name, price, and distance
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                     self.logger.warning(f"Invalid or non-decimal value found in fib_levels_data: {name}={level_price}. Skipping.")

            # Sort the levels based on their distance to the current price (ascending)
            level_distances.sort(key=lambda x: x['distance'])

            # Return the top N nearest levels as (name, price) tuples
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """
        Calculates EMA alignment score based on the latest EMA_Short, EMA_Long, and Close price.

        Returns:
             1.0 if Price > EMA_Short > EMA_Long (Strong Bullish Alignment)
            -1.0 if Price < EMA_Short < EMA_Long (Strong Bearish Alignment)
             0.0 otherwise (Mixed or Crossing)
             np.nan if any required value is missing.
        """
        # Retrieve latest values (should be floats or NaN)
        ema_short = self.indicator_values.get("EMA_Short")
        ema_long = self.indicator_values.get("EMA_Long")
        # Retrieve Close price (should be Decimal or NaN) and convert to float for comparison
        close_decimal = self.indicator_values.get("Close")
        current_price_float = float(close_decimal) if isinstance(close_decimal, Decimal) else np.nan

        # Check if all necessary values are available and valid numbers
        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price_float):
            self.logger.debug("EMA alignment check skipped: Missing required values (EMA_Short, EMA_Long, or Close).")
            return np.nan # Return NaN if data is missing

        # Check for bullish alignment
        if current_price_float > ema_short > ema_long:
            return 1.0
        # Check for bearish alignment
        elif current_price_float < ema_short < ema_long:
            return -1.0
        # Otherwise, EMAs are crossed or price is between them (neutral/mixed alignment)
        else:
            return 0.0


    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """
        Generates a final trading signal (BUY/SELL/HOLD) based on a weighted score
        from various enabled indicator checks.

        Args:
            current_price: The current market price (Decimal).
            orderbook_data: Optional order book data dictionary from fetch_orderbook_ccxt.

        Returns:
            "BUY", "SELL", or "HOLD" string signal.
        """
        # Reset signal states
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Default to HOLD
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {} # For detailed logging

        # --- Pre-checks ---
        if not self.indicator_values:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values dictionary is empty.{RESET}")
             return "HOLD"
        # Check if at least one core indicator has a valid value
        core_indicators_present = any(
            pd.notna(v) for k, v in self.indicator_values.items()
            if k not in ['Open', 'High', 'Low', 'Close', 'Volume'] # Exclude raw OHLCV
        )
        if not core_indicators_present:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: All core indicator values are NaN.{RESET}")
            return "HOLD"
        # Check current price validity
        if pd.isna(current_price) or not isinstance(current_price, Decimal) or current_price <= 0:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
             return "HOLD"

        # Get the active weight set from config
        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
             self.logger.error(f"Active weight set '{self.active_weight_set_name}' missing or empty in config for {self.symbol}. Cannot generate signal.")
             return "HOLD"

        # --- Iterate through configured indicators ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            # Skip if indicator is disabled in the config
            if not enabled: continue

            # Get the weight for this indicator from the active weight set
            weight_str = active_weights.get(indicator_key)
            # Skip if no weight is defined for this enabled indicator
            if weight_str is None: continue

            try:
                # Convert weight to Decimal, skip if weight is zero
                weight = Decimal(str(weight_str))
                if weight == 0: continue
            except Exception:
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping.")
                continue

            # --- Call the corresponding check method ---
            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method_to_call = getattr(self, check_method_name)
                indicator_score_float = np.nan # Initialize score as NaN

                try:
                    # Special case for orderbook check which needs extra data
                    if indicator_key == "orderbook":
                         # Only call if orderbook data is available
                         if orderbook_data:
                             indicator_score_float = method_to_call(orderbook_data, current_price)
                         else:
                             # Log if orderbook indicator is enabled/weighted but data is missing
                              if weight != 0: # Only log if it would have contributed
                                 self.logger.debug(f"Orderbook check skipped for {self.symbol}: No orderbook data provided.")
                    else:
                         # Call standard check methods
                         indicator_score_float = method_to_call() # Expected to return float score or np.nan

                except Exception as e:
                    self.logger.error(f"Error executing indicator check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                    # Keep score as NaN

                # Store score for debugging, format nicely
                debug_scores[indicator_key] = f"{indicator_score_float:.3f}" if pd.notna(indicator_score_float) else "NaN"

                # --- Aggregate score if valid ---
                if pd.notna(indicator_score_float):
                    try:
                        # Convert float score to Decimal for weighted sum
                        score_decimal = Decimal(str(indicator_score_float))
                        # Clamp score between -1 and 1 before applying weight
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_decimal))
                        # Calculate contribution to final score
                        score_contribution = clamped_score * weight
                        final_signal_score += score_contribution
                        # Track total weight applied for normalization/debugging
                        total_weight_applied += weight
                        active_indicator_count += 1
                    except Exception as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} (Score: {indicator_score_float}, Weight: {weight}): {calc_err}")
                        nan_indicator_count += 1 # Count as NaN if processing failed
                else:
                    # Count indicators that returned NaN
                    nan_indicator_count += 1
            else:
                # Log warning if a check method is missing for an enabled/weighted indicator
                self.logger.warning(f"Indicator check method '{check_method_name}' not found for enabled/weighted indicator: {indicator_key} ({self.symbol})")


        # --- Determine Final Signal based on Score ---
        final_signal = "HOLD" # Default
        if total_weight_applied == 0:
             self.logger.warning(f"No indicators contributed valid scores to the signal calculation for {self.symbol}. Defaulting to HOLD.")
        else:
            # Normalize score (optional, but can be useful for consistent thresholding)
            # normalized_score = final_signal_score / total_weight_applied if total_weight_applied else Decimal("0.0")
            # Use raw score for thresholding as weights already scale contributions
            threshold_str = self.config.get("signal_score_threshold", "1.5")
            try:
                threshold = Decimal(str(threshold_str))
            except:
                self.logger.warning(f"Invalid signal_score_threshold '{threshold_str}'. Using default 1.5.")
                threshold = Decimal("1.5")

            if final_signal_score >= threshold:
                final_signal = "BUY"
            elif final_signal_score <= -threshold:
                final_signal = "SELL"
            # else: final_signal remains "HOLD"

        # --- Log Summary ---
        price_prec = self.get_price_precision()
        log_msg = (
            f"Signal Summary ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Indicators=[Active:{active_indicator_count}, NaN:{nan_indicator_count}], "
            f"TotalWeight={total_weight_applied:.2f}, "
            f"FinalScore={final_signal_score:.4f} (Threshold: +/-{threshold:.2f}) "
            # Colorize the final signal
            f"==> {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        # Log detailed scores only at DEBUG level
        self.logger.debug(f"  Indicator Scores ({self.symbol}): {debug_scores}")

        # Update the signals dictionary
        if final_signal == "BUY": self.signals = {"BUY": 1, "SELL": 0, "HOLD": 0}
        elif final_signal == "SELL": self.signals = {"BUY": 0, "SELL": 1, "HOLD": 0}
        else: self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # HOLD

        return final_signal


    # --- Indicator Check Methods (returning float score -1.0 to 1.0 or np.nan) ---
    # Each method should access self.indicator_values to get the latest calculated data.

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment. Requires EMA_Short, EMA_Long, Close."""
        # Check if required values are present (they might be NaN if calc failed)
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
             self.logger.debug("EMA Alignment check skipped: EMA values not found in indicator_values.")
             return np.nan
        # Delegate to the calculation method which handles NaN checks inside
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        """Checks Momentum indicator."""
        momentum = self.indicator_values.get("Momentum") # Should be float or NaN
        if pd.isna(momentum):
            return np.nan

        # Simple thresholding example (can be refined)
        # Scale momentum to a -1 to 1 range based on some expected magnitude
        # Example: Assume significant momentum is > 0.1 or < -0.1
        threshold = 0.1
        if momentum > threshold:
            return 1.0 # Strong positive momentum
        elif momentum < -threshold:
            return -1.0 # Strong negative momentum
        else:
            # Linearly scale momentum between -threshold and +threshold to -1 and 1
            return momentum / threshold


    def _check_volume_confirmation(self) -> float:
        """Checks if current volume confirms the potential trend."""
        current_volume = self.indicator_values.get("Volume") # Should be Decimal or NaN
        volume_ma_float = self.indicator_values.get("Volume_MA") # Should be float or NaN
        # Get multiplier from config, default to 1.5
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))

        # Check if values are available and valid
        if pd.isna(current_volume) or not isinstance(current_volume, Decimal) or \
           pd.isna(volume_ma_float) or volume_ma_float <= 0:
            return np.nan

        try:
            # Convert Volume MA float to Decimal for comparison
            volume_ma = Decimal(str(volume_ma_float))
            multiplier_decimal = Decimal(str(multiplier))

            # Compare current volume to MA * multiplier
            if current_volume > volume_ma * multiplier_decimal:
                # High volume can confirm trend (positive score, magnitude tunable)
                return 0.7
            elif current_volume < volume_ma / multiplier_decimal:
                # Low volume might indicate lack of confirmation (negative score)
                return -0.4
            else:
                # Neutral volume
                return 0.0
        except Exception as e:
            self.logger.warning(f"Error during volume confirmation check: {e}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Checks Stochastic RSI K and D lines."""
        k = self.indicator_values.get("StochRSI_K") # Float or NaN
        d = self.indicator_values.get("StochRSI_D") # Float or NaN

        if pd.isna(k) or pd.isna(d):
            return np.nan

        # Get thresholds from config
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))

        score = 0.0
        # Oversold condition -> Bullish signal
        if k < oversold and d < oversold:
            score = 1.0
        # Overbought condition -> Bearish signal
        elif k > overbought and d > overbought:
            score = -1.0

        # Consider K/D crossing or relationship for additional nuance
        diff = k - d
        # Significant crossing potential (adjust threshold 5 if needed)
        if abs(diff) > 5:
             if diff > 0: # K crossing above D -> Bullish momentum
                  # Enhance existing score or set if neutral
                  score = max(score, 0.6) if score >= 0 else 0.6
             else: # K crossing below D -> Bearish momentum
                  # Enhance existing score or set if neutral
                  score = min(score, -0.6) if score <= 0 else -0.6
        # Weaker signal if K is just above/below D without strong crossing
        elif k > d :
             score = max(score, 0.2) # Mildly bullish
        elif k < d:
             score = min(score, -0.2) # Mildly bearish

        # Dampen score if in the mid-range (e.g., 40-60), less decisive
        if 40 < k < 60:
            score *= 0.5

        return score

    def _check_rsi(self) -> float:
        """Checks standard RSI."""
        rsi = self.indicator_values.get("RSI") # Float or NaN
        if pd.isna(rsi):
            return np.nan

        # Standard RSI interpretation
        if rsi <= 30: return 1.0   # Oversold -> Strong Buy Signal
        if rsi >= 70: return -1.0  # Overbought -> Strong Sell Signal
        # Intermediate levels for weaker signals
        if rsi < 40: return 0.5   # Approaching Oversold -> Moderate Buy
        if rsi > 60: return -0.5  # Approaching Overbought -> Moderate Sell
        # Mid-range (40-60) - potentially scale score linearly
        if 40 <= rsi <= 60:
             # Scale from 0.0 (at RSI 50) to +/- 0.2 at boundaries 40/60
             return (rsi - 50) / 50.0 # Results in -0.2 at 40, 0 at 50, 0.2 at 60
             # Alternative: Flat zero in mid-range
             # return 0.0
        return 0.0 # Should not be reached if logic covers all ranges


    def _check_cci(self) -> float:
        """Checks Commodity Channel Index (CCI)."""
        cci = self.indicator_values.get("CCI") # Float or NaN
        if pd.isna(cci):
            return np.nan

        # CCI Interpretation (Common thresholds: +/- 100, +/- 200)
        if cci <= -150: return 1.0  # Strongly Oversold -> Buy
        if cci >= 150: return -1.0 # Strongly Overbought -> Sell
        if cci < -80: return 0.6   # Moderately Oversold
        if cci > 80: return -0.6  # Moderately Overbought
        # Consider zero line cross or direction near zero
        if cci > 0: return -0.1  # Weak bearish (above zero)
        if cci < 0: return 0.1   # Weak bullish (below zero)
        return 0.0


    def _check_wr(self) -> float:
        """Checks Williams %R."""
        wr = self.indicator_values.get("Williams_R") # Float or NaN
        if pd.isna(wr):
            return np.nan

        # Williams %R Interpretation (Range -100 to 0)
        # Oversold: -100 to -80 -> Buy Signal
        # Overbought: -20 to 0 -> Sell Signal
        if wr <= -80: return 1.0
        if wr >= -20: return -1.0
        # Intermediate signals
        if wr < -50: return 0.4 # Moving out of oversold potentially
        if wr > -50: return -0.4 # Moving out of overbought potentially
        return 0.0

    def _check_psar(self) -> float:
        """Checks Parabolic SAR (PSAR) trend direction."""
        psar_long_signal = self.indicator_values.get("PSAR_long") # Price if long signal active, else NaN
        psar_short_signal = self.indicator_values.get("PSAR_short") # Price if short signal active, else NaN

        # Check which signal is active (non-NaN)
        long_active = pd.notna(psar_long_signal)
        short_active = pd.notna(psar_short_signal)

        if long_active and not short_active:
            return 1.0 # Uptrend indicated by PSAR
        elif short_active and not long_active:
            return -1.0 # Downtrend indicated by PSAR
        elif not long_active and not short_active:
            # This might happen at the start of data or if calculation failed
            return np.nan
        else:
            # Should not happen with standard PSAR calculation (both active simultaneously)
            self.logger.warning(f"PSAR check encountered unexpected state: Long={psar_long_signal}, Short={psar_short_signal}")
            return 0.0 # Neutral or error state


    def _check_sma_10(self) -> float:
        """Checks price position relative to SMA 10."""
        sma_10 = self.indicator_values.get("SMA10") # Float or NaN
        last_close_decimal = self.indicator_values.get("Close") # Decimal or NaN

        # Convert close to float for comparison, handle NaNs
        last_close_float = float(last_close_decimal) if isinstance(last_close_decimal, Decimal) else np.nan

        if pd.isna(sma_10) or pd.isna(last_close_float):
            return np.nan

        # Simple check: Price above SMA is bullish, below is bearish
        if last_close_float > sma_10:
            return 0.6 # Moderate bullish signal
        elif last_close_float < sma_10:
            return -0.6 # Moderate bearish signal
        else:
            return 0.0 # Price is exactly on SMA


    def _check_vwap(self) -> float:
        """Checks price position relative to VWAP."""
        vwap = self.indicator_values.get("VWAP") # Float or NaN
        last_close_decimal = self.indicator_values.get("Close") # Decimal or NaN

        # Convert close to float for comparison, handle NaNs
        last_close_float = float(last_close_decimal) if isinstance(last_close_decimal, Decimal) else np.nan

        if pd.isna(vwap) or pd.isna(last_close_float):
            return np.nan

        # Price above VWAP is generally considered bullish intraday, below is bearish
        if last_close_float > vwap:
            return 0.7 # Stronger bullish signal than SMA 10 perhaps
        elif last_close_float < vwap:
            return -0.7 # Stronger bearish signal
        else:
            return 0.0


    def _check_mfi(self) -> float:
        """Checks Money Flow Index (MFI)."""
        mfi = self.indicator_values.get("MFI") # Float or NaN
        if pd.isna(mfi):
            return np.nan

        # MFI Interpretation (Similar to RSI, but volume-weighted)
        # Oversold: < 20 -> Buy Signal
        # Overbought: > 80 -> Sell Signal
        if mfi <= 20: return 1.0
        if mfi >= 80: return -1.0
        # Intermediate levels
        if mfi < 40: return 0.4
        if mfi > 60: return -0.4
        return 0.0 # Mid-range


    def _check_bollinger_bands(self) -> float:
        """Checks price position relative to Bollinger Bands."""
        bb_lower = self.indicator_values.get("BB_Lower") # Float or NaN
        bb_middle = self.indicator_values.get("BB_Middle") # Float or NaN
        bb_upper = self.indicator_values.get("BB_Upper") # Float or NaN
        last_close_decimal = self.indicator_values.get("Close") # Decimal or NaN

        # Convert close to float for comparison, handle NaNs
        last_close_float = float(last_close_decimal) if isinstance(last_close_decimal, Decimal) else np.nan

        if pd.isna(bb_lower) or pd.isna(bb_middle) or pd.isna(bb_upper) or pd.isna(last_close_float):
            return np.nan

        # Check if price touches or crosses bands (potential reversal/breakout)
        if last_close_float <= bb_lower:
             # Price at or below lower band -> Potential bounce (Buy signal)
             return 1.0
        if last_close_float >= bb_upper:
             # Price at or above upper band -> Potential pullback (Sell signal)
             return -1.0

        # Check position relative to middle band (often an SMA)
        # Calculate distance relative to band width for scaling
        band_width = bb_upper - bb_lower
        if band_width > 0: # Avoid division by zero if bands collapse
            if last_close_float > bb_middle:
                 # Price above middle band, closer to upper band is less bullish
                 proximity_to_upper = (last_close_float - bb_middle) / (bb_upper - bb_middle) if (bb_upper - bb_middle) > 0 else 0
                 # Scale score from +0.5 (just above middle) down to 0 (near upper)
                 return 0.5 * (1 - proximity_to_upper)
            elif last_close_float < bb_middle:
                 # Price below middle band, closer to lower band is less bearish
                 proximity_to_lower = (bb_middle - last_close_float) / (bb_middle - bb_lower) if (bb_middle - bb_lower) > 0 else 0
                 # Scale score from -0.5 (just below middle) down to 0 (near lower)
                 return -0.5 * (1 - proximity_to_lower)
        # If price is exactly on middle band or bands collapsed
        return 0.0


    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """
        Analyzes order book depth (imbalance) as a sentiment indicator.
        Returns float score (-1.0 to 1.0) or NaN.
        """
        if not orderbook_data:
            self.logger.debug("Orderbook check skipped: No data provided.")
            return np.nan

        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])

            # Ensure we have both bids and asks to compare
            if not bids or not asks:
                self.logger.debug("Orderbook check skipped: Missing bids or asks.")
                return np.nan

            # --- Simple Order Book Imbalance (OBI) Calculation ---
            # Consider N levels deep (e.g., first 10 levels)
            num_levels_to_check = 10 # Make this configurable?
            top_bids = bids[:num_levels_to_check]
            top_asks = asks[:num_levels_to_check]

            # Sum the sizes (quantities) at these levels
            # Use Decimal for summation to maintain precision
            bid_volume_sum = sum(Decimal(str(bid[1])) for bid in top_bids if len(bid) == 2)
            ask_volume_sum = sum(Decimal(str(ask[1])) for ask in top_asks if len(ask) == 2)

            # Calculate total volume in the checked range
            total_volume = bid_volume_sum + ask_volume_sum

            # Avoid division by zero if total volume is zero (inactive market?)
            if total_volume == 0:
                self.logger.debug(f"Orderbook check ({self.symbol}): Zero total volume in top {num_levels_to_check} levels.")
                return 0.0 # Neutral signal

            # Calculate Order Book Imbalance ratio
            # OBI = (BidVolume - AskVolume) / TotalVolume
            # Ranges from -1 (all asks) to +1 (all bids)
            obi_decimal = (bid_volume_sum - ask_volume_sum) / total_volume

            # Convert OBI Decimal to float score
            score = float(obi_decimal)

            # --- Log the analysis ---
            self.logger.debug(
                f"Orderbook check ({self.symbol}): Top {num_levels_to_check} levels -> "
                f"BidVol={bid_volume_sum:.4f}, AskVol={ask_volume_sum:.4f}, "
                f"OBI={obi_decimal:.4f} -> Score={score:.4f}"
            )

            # --- Refinements (Optional) ---
            # 1. Weighted OBI: Give more weight to levels closer to the spread.
            # 2. Volume Clusters: Look for large individual orders ("walls").
            # 3. Spread Analysis: Consider the bid-ask spread size.

            # Return the calculated score
            return score

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan # Return NaN on error


    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential Take Profit (TP) and initial Stop Loss (SL) levels
        based on an estimated entry price, the signal direction, ATR, and config multipliers.

        This initial SL is primarily used for position sizing. The actual SL set on the
        exchange might be different (e.g., adjusted by BE logic or replaced by TSL).

        Args:
            entry_price_estimate: An estimated entry price (Decimal) for the trade.
                                  (Could be current price or anticipated limit fill price).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            A tuple containing: (entry_price_estimate, take_profit, stop_loss)
            All values are Decimals or None if calculation fails.
        """
        # Only calculate for valid BUY/SELL signals
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"TP/SL calculation skipped: Signal is '{signal}'.")
            return entry_price_estimate, None, None

        # --- Retrieve necessary values ---
        atr_val = self.indicator_values.get("ATR") # Should be Decimal or NaN

        # --- Validate Inputs ---
        if not isinstance(atr_val, Decimal) or pd.isna(atr_val) or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol} {signal}: Invalid or missing ATR ({atr_val}).{RESET}")
            return entry_price_estimate, None, None
        if not isinstance(entry_price_estimate, Decimal) or pd.isna(entry_price_estimate) or entry_price_estimate <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol} {signal}: Invalid entry price estimate ({entry_price_estimate}).{RESET}")
            return entry_price_estimate, None, None

        try:
            # --- Get Multipliers from Config ---
            # Convert multipliers to Decimal for precise calculations
            tp_multiple_str = self.config.get("take_profit_multiple", "1.0") # Default 1.0 if missing
            sl_multiple_str = self.config.get("stop_loss_multiple", "1.5")   # Default 1.5 if missing
            tp_multiple = Decimal(str(tp_multiple_str))
            sl_multiple = Decimal(str(sl_multiple_str))

            # --- Get Market Precision Info ---
            price_precision = self.get_price_precision()
            # Quantizer for rounding to the correct number of decimal places
            rounding_factor = Decimal('1e-' + str(price_precision))
            # Minimum price increment (tick size) for validation/adjustments
            min_tick = self.get_min_tick_size()

            # --- Calculate Offsets ---
            tp_offset = atr_val * tp_multiple
            sl_offset = atr_val * sl_multiple

            # --- Calculate Raw TP/SL Prices ---
            take_profit_raw: Optional[Decimal] = None
            stop_loss_raw: Optional[Decimal] = None

            if signal == "BUY":
                take_profit_raw = entry_price_estimate + tp_offset
                stop_loss_raw = entry_price_estimate - sl_offset
            elif signal == "SELL":
                take_profit_raw = entry_price_estimate - tp_offset
                stop_loss_raw = entry_price_estimate + sl_offset

            # --- Quantize TP/SL to Market Precision ---
            # Quantize TP towards profit direction (UP for BUY, DOWN for SELL)
            # Quantize SL towards loss direction (DOWN for BUY, UP for SELL) - more conservative SL
            take_profit_quantized: Optional[Decimal] = None
            stop_loss_quantized: Optional[Decimal] = None

            if take_profit_raw is not None:
                 tp_rounding = ROUND_UP if signal == "BUY" else ROUND_DOWN
                 take_profit_quantized = take_profit_raw.quantize(rounding_factor, rounding=tp_rounding)

            if stop_loss_raw is not None:
                 sl_rounding = ROUND_DOWN if signal == "BUY" else ROUND_UP
                 stop_loss_quantized = stop_loss_raw.quantize(rounding_factor, rounding=sl_rounding)

            # --- Validation and Adjustments ---
            final_tp = take_profit_quantized
            final_sl = stop_loss_quantized

            # 1. Ensure SL is strictly beyond entry by at least one tick
            if final_sl is not None:
                 if signal == "BUY" and final_sl >= entry_price_estimate:
                      original_sl = final_sl
                      final_sl = (entry_price_estimate - min_tick).quantize(rounding_factor, rounding=ROUND_DOWN)
                      self.logger.debug(f"Adjusted BUY SL below entry: {original_sl} -> {final_sl}")
                 elif signal == "SELL" and final_sl <= entry_price_estimate:
                      original_sl = final_sl
                      final_sl = (entry_price_estimate + min_tick).quantize(rounding_factor, rounding=ROUND_UP)
                      self.logger.debug(f"Adjusted SELL SL above entry: {original_sl} -> {final_sl}")

            # 2. Ensure TP provides potential profit (strictly beyond entry)
            if final_tp is not None:
                 if signal == "BUY" and final_tp <= entry_price_estimate:
                      self.logger.warning(f"{NEON_YELLOW}BUY TP calculation non-profitable (TP {final_tp} <= Entry {entry_price_estimate}). Setting TP to None.{RESET}")
                      final_tp = None
                 elif signal == "SELL" and final_tp >= entry_price_estimate:
                      self.logger.warning(f"{NEON_YELLOW}SELL TP calculation non-profitable (TP {final_tp} >= Entry {entry_price_estimate}). Setting TP to None.{RESET}")
                      final_tp = None

            # 3. Ensure SL/TP are positive prices
            if final_sl is not None and final_sl <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in non-positive price ({final_sl}). Setting SL to None.{RESET}")
                final_sl = None
            if final_tp is not None and final_tp <= 0:
                self.logger.warning(f"{NEON_YELLOW}Take profit calculation resulted in non-positive price ({final_tp}). Setting TP to None.{RESET}")
                final_tp = None

            # --- Log Calculation Results ---
            tp_str = f"{final_tp:.{price_precision}f}" if final_tp else "None"
            sl_str = f"{final_sl:.{price_precision}f}" if final_sl else "None"
            self.logger.debug(
                f"Calculated TP/SL for {self.symbol} {signal}: "
                f"EntryEst={entry_price_estimate:.{price_precision}f}, "
                f"ATR={atr_val:.{price_precision+1}f}, " # Show ATR with more precision
                f"TP={tp_str} (Mult: {tp_multiple}), "
                f"SL={sl_str} (Mult: {sl_multiple})"
            )

            return entry_price_estimate, final_tp, final_sl

        except Exception as e:
             # Catch any unexpected errors during calculation
             self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol} {signal}: {e}{RESET}", exc_info=True)
             return entry_price_estimate, None, None


# --- Trading Logic Helper Functions (Adapted from livexy.py) ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency with retries and robust parsing."""
    lg = logger
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            balance_info = None
            # Prioritize specific account types if needed (e.g., Bybit V5)
            account_types_to_try = []
            if exchange.id == 'bybit':
                 account_types_to_try = ['CONTRACT', 'UNIFIED'] # Try V5 types first

            found_structure = False
            # Try specific account types first
            for acc_type in account_types_to_try:
                 try:
                     lg.debug(f"Fetching balance using params={{'type': '{acc_type}'}} for {currency}... (Attempt {attempt+1})")
                     balance_info = exchange.fetch_balance(params={'type': acc_type})
                     # Check standard CCXT structure first
                     if currency in balance_info and balance_info[currency].get('free') is not None:
                         found_structure = True; break
                     # Check Bybit V5 specific structure within 'info'
                     elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             if isinstance(account.get('coin'), list):
                                 # Find the specific coin data within the list
                                 if any(coin_data.get('coin') == currency for coin_data in account['coin']):
                                     found_structure = True; break # Found the currency
                         if found_structure: break # Exit outer loop too
                     lg.debug(f"Currency '{currency}' not directly found using type '{acc_type}'. Checking V5 structure...")
                 except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                     # Log specific errors but continue trying other types or default fetch
                     lg.debug(f"Error fetching balance for type '{acc_type}': {e}. Trying next.")
                     continue
                 except Exception as e:
                     lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}. Trying next.")
                     continue

            # If not found with specific types, try default fetch_balance
            if not found_structure:
                 lg.debug(f"Fetching balance using default parameters for {currency}... (Attempt {attempt+1})")
                 try:
                     balance_info = exchange.fetch_balance()
                 except Exception as e:
                     # If default fetch also fails, log and proceed to retry logic
                     lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                     raise e # Re-raise to trigger retry

            # --- Parse the final balance_info ---
            if balance_info:
                available_balance_str = None
                # 1. Standard CCXT: balance[currency]['free']
                if currency in balance_info and balance_info[currency].get('free') is not None:
                    available_balance_str = str(balance_info[currency]['free'])
                    lg.debug(f"Found balance via standard ['{currency}']['free']: {available_balance_str}")

                # 2. Bybit V5 Nested: info.result.list[].coin[].availableToWithdraw/availableBalance
                elif not available_balance_str and 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                    for account in balance_info['info']['result']['list']:
                        if isinstance(account.get('coin'), list):
                            for coin_data in account['coin']:
                                 if coin_data.get('coin') == currency:
                                     # Prioritize availableToWithdraw > availableBalance > walletBalance
                                     free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                     if free is not None:
                                         available_balance_str = str(free)
                                         lg.debug(f"Found balance via Bybit V5 nested ['available...']: {available_balance_str}")
                                         break # Found it
                            if available_balance_str is not None: break # Exit account loop
                    if not available_balance_str:
                         lg.warning(f"{currency} balance details not found within Bybit V5 'info.result.list[].coin[]'.")

                # 3. Fallback: Top-level 'free' dictionary
                elif not available_balance_str and 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                     available_balance_str = str(balance_info['free'][currency])
                     lg.debug(f"Found balance via top-level 'free' dict: {available_balance_str}")

                # 4. Final Fallback: Use 'total' balance if 'free' is unavailable
                if available_balance_str is None:
                     total_balance = balance_info.get(currency, {}).get('total')
                     if total_balance is not None:
                          lg.warning(f"{NEON_YELLOW}Using 'total' balance ({total_balance}) as fallback for available {currency}.{RESET}")
                          available_balance_str = str(total_balance)
                     else:
                          lg.error(f"{NEON_RED}Could not determine any balance ('free' or 'total') for {currency}.{RESET}")
                          lg.debug(f"Full balance_info structure: {balance_info}")
                          # Continue to retry logic or return None if retries exhausted

                # Convert the found string balance to Decimal
                if available_balance_str is not None:
                    try:
                        final_balance = Decimal(available_balance_str)
                        if final_balance >= 0:
                             lg.info(f"Available {currency} balance: {final_balance:.4f}")
                             return final_balance
                        else:
                             lg.error(f"Parsed balance for {currency} is negative ({final_balance}).")
                             # Treat negative balance as an error, may retry
                    except Exception as e:
                        lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
                        # Treat conversion error as failure, may retry
            else:
                # If balance_info itself is None after attempts
                lg.error(f"Balance info was None after fetch attempt {attempt + 1}.")


            # If we got here, something failed in parsing or fetching, proceed to retry
            raise ccxt.ExchangeError("Balance parsing failed or data missing") # Trigger retry

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            lg.warning(f"Network error fetching balance: {e}. Retrying ({attempt+1}/{MAX_API_RETRIES})...")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s ({attempt+1}/{MAX_API_RETRIES})...")
            time.sleep(wait_time)
            continue # Skip standard delay after rate limit wait
        except ccxt.AuthenticationError as e:
             lg.error(f"{NEON_RED}Authentication error fetching balance: {e}. Aborting balance fetch.{RESET}")
             return None # Don't retry auth errors
        except ccxt.ExchangeError as e:
            lg.warning(f"Exchange error fetching balance: {e}. Retrying ({attempt+1}/{MAX_API_RETRIES})...")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            # Decide if unexpected errors should be retried

        # Standard delay before next attempt
        if attempt < MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information (precision, limits, type) using exchange.market()."""
    lg = logger
    try:
        # Ensure markets are loaded
        if not exchange.markets or symbol not in exchange.markets:
             lg.info(f"Market info for {symbol} not loaded or symbol not found, reloading markets...")
             try:
                 exchange.load_markets(reload=True)
             except Exception as load_err:
                  lg.error(f"{NEON_RED}Failed to reload markets: {load_err}{RESET}")
                  return None # Cannot proceed without markets

        # Check again after reload
        if symbol not in exchange.markets:
             lg.error(f"{NEON_RED}Market {symbol} still not found after reloading.{RESET}")
             # Suggest alternatives if common variations exist (e.g., PERP)
             if '/' in symbol:
                 base, quote = symbol.split('/', 1)
                 perp_sym = f"{symbol}:USDT" # Example for Bybit linear perps
                 if perp_sym in exchange.markets:
                     lg.warning(f"{NEON_YELLOW}Did you mean '{perp_sym}'?{RESET}")
             return None

        # Retrieve market details
        market = exchange.market(symbol)
        if market:
            # --- Extract relevant details ---
            market_type = market.get('type', 'unknown') # spot, future, swap, option
            is_contract = market.get('contract', False) or market_type in ['swap', 'future']
            contract_type = "N/A"
            if is_contract:
                if market.get('linear'): contract_type = "Linear"
                elif market.get('inverse'): contract_type = "Inverse"
                else: contract_type = "Unknown Contract"

            # Log key details
            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                f"Type={market_type}, IsContract={is_contract}, ContractType={contract_type}, "
                f"Precision(Price/Amount): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            # Add custom 'is_contract' flag for easier checks later
            market['is_contract'] = is_contract
            return market
        else:
             # Should not happen if symbol is in exchange.markets, but handle defensively
             lg.error(f"{NEON_RED}Market dictionary unexpectedly not found for validated symbol {symbol}.{RESET}")
             return None

    except ccxt.BadSymbol as e:
         # This might occur if the symbol format is wrong despite passing initial checks
         lg.error(f"{NEON_RED}Symbol '{symbol}' is invalid or not supported by {exchange.id}: {e}{RESET}")
         return None
    except ccxt.NetworkError as e:
         lg.error(f"{NEON_RED}Network error getting market info for {symbol}: {e}{RESET}")
         return None # Network errors might be temporary, but critical for this step
    except Exception as e:
        # Catch any other unexpected errors
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # e.g., 0.01 for 1%
    initial_stop_loss_price: Decimal, # Calculated SL price (must be validated before calling)
    entry_price: Decimal, # Estimated or actual entry price
    market_info: Dict, # From get_market_info()
    exchange: ccxt.Exchange, # Needed for formatting helpers
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates the position size in base currency or contracts based on risk percentage,
    stop-loss distance, available balance, and market constraints (precision, limits).

    Args:
        balance: Available balance in QUOTE currency (Decimal).
        risk_per_trade: Risk percentage per trade (float, e.g., 0.01 for 1%).
        initial_stop_loss_price: The calculated stop-loss price (Decimal).
        entry_price: The estimated or actual entry price (Decimal).
        market_info: The market dictionary from CCXT.
        exchange: The CCXT exchange instance (for formatting).
        logger: Logger instance.

    Returns:
        Calculated position size (Decimal) in base currency (spot) or contracts (futures),
        or None if calculation fails or constraints are violated.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY) # e.g., USDT
    base_currency = market_info.get('base', 'BASE')       # e.g., BTC
    is_contract = market_info.get('is_contract', False)
    # Determine unit based on market type
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or not isinstance(balance, Decimal) or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid or zero balance ({balance}).")
        return None
    if not isinstance(risk_per_trade, (float, int)) or not (0 < risk_per_trade < 1):
         lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
         return None
    if initial_stop_loss_price is None or not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if entry_price is None or not isinstance(entry_price, Decimal) or entry_price <= 0:
         lg.error(f"Position sizing failed ({symbol}): Invalid entry_price ({entry_price}).")
         return None
    if initial_stop_loss_price == entry_price:
         lg.error(f"Position sizing failed ({symbol}): Stop loss price cannot be equal to entry price.")
         return None
    if 'limits' not in market_info or 'precision' not in market_info:
         lg.error(f"Position sizing failed ({symbol}): Market info missing 'limits' or 'precision'.")
         return None

    try:
        # --- Calculate Risk Amount ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))

        # --- Calculate SL Distance per Unit ---
        # This is the risk per unit (contract or base currency) in quote currency
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0: # Should be caught by earlier check, but defense-in-depth
             lg.error(f"Position sizing failed ({symbol}): Stop loss distance is zero or negative ({sl_distance_per_unit}).")
             return None

        # --- Get Contract Size (for contracts) ---
        # Defaults to 1 (for spot or if contractSize is missing/invalid)
        contract_size_str = market_info.get('contractSize', '1')
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0: raise ValueError("Contract size must be positive")
        except Exception:
            lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}, using 1.")
            contract_size = Decimal('1')

        # --- Calculate Initial Size based on Risk ---
        # Formula: Size = RiskAmount / (StopLossDistancePerUnit * ValuePerUnit)
        # For Linear Contracts/Spot: ValuePerUnit is contract_size (in base currency value per contract)
        # For Inverse Contracts: ValuePerUnit depends on price, more complex. Assuming Linear/Spot here.

        calculated_size: Optional[Decimal] = None
        if market_info.get('linear', True) or not is_contract: # Assume linear or spot
             # Risk is in quote, SL distance is in quote, contract_size converts contracts to base value units
             # For Spot: contract_size is 1, Size = RiskQuote / SL_Quote_Per_Base
             # For Linear: contract_size is base_per_contract, Size = RiskQuote / (SL_Quote_Per_Contract)
             # SL_Quote_Per_Contract = SL_Quote_Per_Base * base_per_contract = sl_distance_per_unit * contract_size
             if sl_distance_per_unit * contract_size > 0:
                 calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)
             else:
                 lg.error(f"Position sizing failed ({symbol}): Denominator zero/negative in size calculation.")
                 return None
        else: # Inverse Contract Placeholder
             # Sizing inverse contracts based on fixed quote risk is complex
             # Risk is in quote, but position is sized in contracts (valued in base)
             # Requires converting quote risk to base risk at entry price, then calculating contracts
             # Example (simplified, verify accuracy): BaseRisk = RiskQuote / EntryPrice
             # SizeInContracts = BaseRisk / (SL_Distance_Base * ContractValueBase)
             # This needs careful implementation based on exchange specifics.
             lg.error(f"{NEON_RED}Inverse contract sizing not fully implemented. Aborting sizing for {symbol}.{RESET}")
             # calculated_size = ... # Implement inverse logic here if needed
             return None # Abort for now

        if calculated_size is None or calculated_size <= 0:
             lg.error(f"Initial position size calculation resulted in zero or negative: {calculated_size}. RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_per_unit}, ContractSize={contract_size}")
             return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  ContractSize={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {}) # Cost limits are in Quote currency
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount') # Usually step size (float/str) or decimal places (int)

        # Min/Max Amount Limits (in base currency or contracts)
        min_amount_str = amount_limits.get('min')
        max_amount_str = amount_limits.get('max')
        # Use Decimal for limits, handle None with appropriate defaults
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')

        # Min/Max Cost Limits (in quote currency)
        min_cost_str = cost_limits.get('min')
        max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')


        # 1. Adjust for MIN/MAX AMOUNT limits
        adjusted_size = calculated_size
        if adjusted_size < min_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {calculated_size:.8f} is below min amount {min_amount}. Adjusting to min amount.{RESET}")
             adjusted_size = min_amount
        elif adjusted_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {calculated_size:.8f} exceeds max amount {max_amount}. Adjusting to max amount.{RESET}")
             adjusted_size = max_amount


        # 2. Check COST limits (Estimate cost based on adjusted size)
        # Cost calculation assumes Linear/Spot: Cost = Size * EntryPrice * ContractSize
        estimated_cost = adjusted_size * entry_price * contract_size
        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Estimated Cost={estimated_cost:.4f} {quote_currency}")

        # Check Min Cost
        if min_cost > 0 and estimated_cost < min_cost :
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} is below min cost {min_cost}. Attempting to increase size.{RESET}")
             # Calculate the size needed to meet min cost
             required_size_for_min_cost: Optional[Decimal] = None
             denominator = entry_price * contract_size
             if denominator > 0:
                 required_size_for_min_cost = min_cost / denominator
             else:
                 lg.error("Cannot calculate required size for min cost: EntryPrice or ContractSize is zero/negative.")
                 return None

             if required_size_for_min_cost is None: return None # Should be caught above

             lg.info(f"  Required size to meet min cost: {required_size_for_min_cost:.8f} {size_unit}")

             # Check if required size violates other limits
             if required_size_for_min_cost > max_amount:
                  lg.error(f"{NEON_RED}Cannot meet min cost {min_cost} without exceeding max amount limit {max_amount}. Aborted.{RESET}")
                  return None
             # This check might be redundant if min_amount adjustment happened first, but good safety check
             if required_size_for_min_cost < min_amount:
                 lg.error(f"{NEON_RED}Cannot meet min cost: Required size {required_size_for_min_cost:.8f} is below min amount {min_amount}. Aborted.{RESET}")
                 # This indicates conflicting limits on the exchange
                 return None
             else:
                 # Adjust size up to meet min cost
                 lg.info(f"  Adjusting size to meet min cost: {adjusted_size:.8f} -> {required_size_for_min_cost:.8f}")
                 adjusted_size = required_size_for_min_cost
                 # Recalculate estimated cost with the new size for max cost check
                 estimated_cost = adjusted_size * entry_price * contract_size


        # Check Max Cost (after potential min cost adjustment)
        elif max_cost > 0 and estimated_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} exceeds max cost {max_cost}. Reducing size.{RESET}")
             # Calculate the maximum size allowed by max cost
             adjusted_size_for_max_cost: Optional[Decimal] = None
             denominator = entry_price * contract_size
             if denominator > 0:
                 adjusted_size_for_max_cost = max_cost / denominator
             else:
                 lg.error("Cannot calculate max size for max cost: EntryPrice or ContractSize is zero/negative.")
                 return None

             if adjusted_size_for_max_cost is None: return None # Should be caught above

             lg.info(f"  Reduced size allowed by max cost: {adjusted_size_for_max_cost:.8f} {size_unit}")

             # Ensure the reduced size is still above the minimum amount
             if adjusted_size_for_max_cost < min_amount:
                  lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is below min amount {min_amount}. Aborted.{RESET}")
                  return None
             else:
                 # Adjust size down to meet max cost
                 lg.info(f"  Adjusting size to meet max cost: {adjusted_size:.8f} -> {adjusted_size_for_max_cost:.8f}")
                 adjusted_size = adjusted_size_for_max_cost


        # 3. Apply Amount Precision/Step Size
        # Use ccxt's amount_to_precision for reliable formatting based on market info
        try:
            # Convert Decimal to float for ccxt function
            amount_float = float(adjusted_size)
            # Use TRUNCATE (rounding down) to be conservative with size
            formatted_size_str = exchange.amount_to_precision(symbol, amount_float) # Default rounding mode might be ok too
            # Some exchanges might require specific padding modes, check ccxt docs if needed
            # formatted_size_str = exchange.amount_to_precision(symbol, amount_float, padding_mode=exchange.TRUNCATE)

            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step size: {adjusted_size:.8f} -> {final_size} {size_unit}")

        except ccxt.ExchangeError as fmt_err:
             lg.warning(f"{NEON_YELLOW}CCXT formatting error applying amount precision ({fmt_err}). Check market data or try manual rounding.{RESET}")
             # Fallback: Manual rounding using step size (if amount precision is step size)
             if isinstance(amount_precision_val, (float, str)):
                 try:
                     amount_step = Decimal(str(amount_precision_val))
                     if amount_step > 0:
                         # Round down to the nearest step size
                         final_size = (adjusted_size // amount_step) * amount_step
                         lg.info(f"Applied manual amount step size ({amount_step}): {adjusted_size:.8f} -> {final_size} {size_unit}")
                     else: raise ValueError("Amount step size is not positive")
                 except Exception as manual_err:
                     lg.error(f"Manual step size rounding failed: {manual_err}. Using unrounded size: {adjusted_size}", exc_info=True)
                     final_size = adjusted_size # Use unrounded as last resort
             else:
                 lg.error(f"Cannot determine amount step size for manual rounding. Using unrounded size: {adjusted_size}")
                 final_size = adjusted_size


        # --- Final Validation ---
        if final_size <= 0:
             lg.error(f"{NEON_RED}Position size became zero or negative ({final_size}) after adjustments. Aborted.{RESET}")
             return None
        # Final check against min amount after precision formatting
        if final_size < min_amount:
             # This can happen if min_amount itself doesn't align with step size
             lg.error(f"{NEON_RED}Final size {final_size} is below minimum amount {min_amount} after precision formatting. Aborted.{RESET}")
             return None
        # Final check against min cost after precision formatting
        final_cost = final_size * entry_price * contract_size
        if min_cost > 0 and final_cost < min_cost:
            # This implies the minimum amount constraint resulted in a cost below minimum cost
            lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} which is below minimum cost {min_cost}. Exchange limits conflict? Aborted.{RESET}")
            return None


        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        # Catch any unexpected errors during the entire process
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions."""
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        fetch_all = False # Flag to fetch all symbols if single fetch fails

        # 1. Attempt to fetch position for the specific symbol (more efficient)
        # Some exchanges require a list of symbols, even for one
        if exchange.has.get('fetchPositions'):
            try:
                # Try fetching single symbol first (works on Bybit V5 unified, maybe others)
                positions = exchange.fetch_positions([symbol])
                lg.debug(f"Fetched single symbol position data for {symbol}. Count: {len(positions)}")
            except ccxt.ArgumentsRequired:
                 # If fetchPositions requires no args (fetches all), set flag
                 lg.debug(f"fetchPositions for {exchange.id} requires no arguments. Fetching all.")
                 fetch_all = True
            except ccxt.ExchangeError as e:
                 # Handle specific errors indicating no position exists cleanly
                 no_pos_codes_v5 = [110025] # Bybit V5: Position idx not match / Position is closed
                 no_pos_messages = ["position not found", "position is closed"]
                 err_str = str(e).lower()
                 bybit_code = getattr(e, 'code', None)

                 if any(msg in err_str for msg in no_pos_messages) or (bybit_code in no_pos_codes_v5):
                      lg.info(f"No position found for {symbol} (Exchange confirmed: {e}).")
                      return None # Confirmed no position
                 else:
                      # Log other exchange errors but consider them temporary failures
                      lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=False)
                      # Decide whether to fetch all as fallback or return None
                      fetch_all = True # Fallback to fetching all positions
                      # return None # Option: Treat error as failure
            except Exception as e:
                 # Handle other unexpected errors during single fetch
                 lg.error(f"Error fetching single position for {symbol}: {e}", exc_info=True)
                 fetch_all = True # Fallback to fetching all positions
                 # return None # Option: Treat error as failure
        else:
             lg.warning(f"Exchange {exchange.id} does not support fetchPositions. Cannot check position status.")
             return None


        # 2. Fetch all positions if single fetch failed or wasn't attempted
        if fetch_all:
            lg.debug(f"Attempting to fetch all positions for {exchange.id}...")
            try:
                 all_positions = exchange.fetch_positions()
                 # Filter for the target symbol
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
            except Exception as e:
                 lg.error(f"Error fetching all positions for {symbol}: {e}", exc_info=True)
                 return None # Failed to get any position data


        # --- Process the fetched positions list ---
        active_position = None
        # Define a small threshold to consider a position size non-zero
        size_threshold = Decimal('1e-9') # Adjust if needed based on minimum contract size

        for pos in positions:
            pos_size_str = None
            # Find position size - check standard 'contracts' and Bybit V5 'info.size'
            if pos.get('contracts') is not None:
                pos_size_str = str(pos['contracts'])
            elif isinstance(pos.get('info'), dict) and pos['info'].get('size') is not None:
                pos_size_str = str(pos['info']['size']) # Common in Bybit V5

            if pos_size_str is None:
                lg.debug(f"Skipping position entry, could not determine size: {pos}")
                continue # Skip if size cannot be determined

            try:
                position_size = Decimal(pos_size_str)
                # Check if the absolute size is greater than the threshold
                if abs(position_size) > size_threshold:
                    # Found an active position
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break # Assume only one position per symbol (or handle multiple if needed)
            except Exception as parse_err:
                lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol}: {parse_err}")
                continue


        # --- Post-Process the found active position (if any) ---
        if active_position:
            # Ensure essential fields are present and standardized
            try:
                # Standardize Size
                size_decimal = Decimal(str(active_position.get('contracts', active_position.get('info',{}).get('size', '0'))))
                active_position['contractsDecimal'] = size_decimal # Store Decimal size

                # Standardize Side
                side = active_position.get('side')
                # Infer side from size if missing (common issue)
                if side not in ['long', 'short']:
                    if size_decimal > size_threshold: side = 'long'
                    elif size_decimal < -size_threshold: side = 'short'
                    else:
                        lg.warning(f"Position size {size_decimal} near zero for {symbol}, cannot reliably determine side.")
                        return None # Cannot use position if side unknown
                    active_position['side'] = side # Store inferred side
                    lg.debug(f"Inferred position side as '{side}' based on size {size_decimal}.")

                # Standardize Entry Price
                entry_price_str = active_position.get('entryPrice') or active_position.get('info', {}).get('avgPrice')
                if entry_price_str:
                    active_position['entryPriceDecimal'] = Decimal(str(entry_price_str))
                else: active_position['entryPriceDecimal'] = None

                # Standardize SL/TP/TSL from 'info' if not top-level
                info_dict = active_position.get('info', {})
                if active_position.get('stopLossPrice') is None: active_position['stopLossPrice'] = info_dict.get('stopLoss')
                if active_position.get('takeProfitPrice') is None: active_position['takeProfitPrice'] = info_dict.get('takeProfit')
                # TSL info might be in different fields depending on exchange/version
                active_position['trailingStopLossValue'] = info_dict.get('trailingStop') # Bybit V5: distance value
                active_position['trailingStopActivationPrice'] = info_dict.get('activePrice') # Bybit V5: activation price

                # Get timestamp (usually milliseconds)
                timestamp_ms = active_position.get('timestamp') or info_dict.get('updatedTime') # Use 'timestamp' or fallback
                active_position['timestamp_ms'] = timestamp_ms

                # --- Log Formatted Position Info ---
                # Helper to format values safely for logging
                def format_log_val(val, is_price=True, is_size=False):
                     if val is None or str(val).strip() == '' or str(val) == '0': return 'N/A'
                     try:
                          d_val = Decimal(str(val))
                          if is_size:
                              # Get amount precision dynamically if possible, else default
                              try: amt_prec = abs(Decimal(str(market_info['precision']['amount'])).normalize().as_tuple().exponent)
                              except: amt_prec = 8
                              return f"{abs(d_val):.{amt_prec}f}" # Show absolute size
                          elif is_price:
                              # Use market price precision
                              try: price_prec = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info).get_price_precision()
                              except: price_prec = 6 # Fallback precision
                              return f"{d_val:.{price_prec}f}"
                          else: # Other values like PNL
                              return f"{d_val:.4f}" # Default formatting
                     except: return str(val) # Fallback to string if conversion fails

                entry_price_fmt = format_log_val(active_position.get('entryPriceDecimal'))
                contracts_fmt = format_log_val(size_decimal, is_size=True)
                liq_price_fmt = format_log_val(active_position.get('liquidationPrice'))
                leverage_str = active_position.get('leverage', info_dict.get('leverage'))
                leverage_fmt = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str is not None else 'N/A'
                pnl_fmt = format_log_val(active_position.get('unrealizedPnl'), is_price=False)
                sl_price_fmt = format_log_val(active_position.get('stopLossPrice'))
                tp_price_fmt = format_log_val(active_position.get('takeProfitPrice'))
                tsl_dist_fmt = format_log_val(active_position.get('trailingStopLossValue'), is_price=False) # TSL distance is a value/rate
                tsl_act_fmt = format_log_val(active_position.get('trailingStopActivationPrice'))

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={contracts_fmt}, Entry={entry_price_fmt}, Liq={liq_price_fmt}, "
                            f"Lev={leverage_fmt}, PnL={pnl_fmt}, SL={sl_price_fmt}, TP={tp_price_fmt}, "
                            f"TSL(Dist/Act): {tsl_dist_fmt}/{tsl_act_fmt}")
                logger.debug(f"Full position details for {symbol}: {active_position}")

                return active_position # Return the processed position dictionary

            except Exception as proc_err:
                 lg.error(f"Error processing active position details for {symbol}: {proc_err}", exc_info=True)
                 lg.debug(f"Problematic position data: {active_position}")
                 return None # Failed to process essential details

        else:
            # No position with size > threshold found
            logger.info(f"No active open position found for {symbol}.")
            return None

    except Exception as e:
        # Catch errors occurring outside the fetch loops (e.g., initial capability check)
        lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling exchange specifics (like Bybit V5)."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    # Validate inputs and market type
    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # No action needed, considered success
    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be a positive integer.")
        return False
    # Check if the exchange instance supports setting leverage
    if not exchange.has.get('setLeverage'):
         # Check if setMarginMode is an alternative (some exchanges combine them)
         if not exchange.has.get('setMarginMode'):
              lg.error(f"{NEON_RED}Exchange {exchange.id} does not support setLeverage or setMarginMode via CCXT. Cannot set leverage.{RESET}")
              return False
         else:
              lg.warning(f"{NEON_YELLOW}Exchange {exchange.id} uses setMarginMode for leverage. Attempting via setMarginMode...{RESET}")
              # Fall through to try setMarginMode logic if needed (though setLeverage is preferred)

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        # --- Prepare parameters ---
        params = {}
        # Handle Bybit V5 requirement for buy/sell leverage (needed even for same value)
        if 'bybit' in exchange.id.lower():
             # Ensure leverage is passed as string for Bybit V5 params
             leverage_str = str(leverage)
             params = {'buyLeverage': leverage_str, 'sellLeverage': leverage_str}
             lg.debug(f"Using Bybit V5 params for set_leverage: {params}")
        # Add other exchange-specific params here if necessary

        # --- Call setLeverage ---
        # Use the main leverage argument and the params dict
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # --- Verification (Difficult without fetching position again) ---
        # CCXT setLeverage often doesn't return detailed confirmation.
        # Success is usually assumed if no exception is raised.
        # Some exchanges might return a confirmation message in the response.
        # Example check (adapt based on actual response structure):
        # if isinstance(response, dict) and response.get('info', {}).get('retMsg') == 'OK':
        #      lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set to {leverage}x (Confirmed by response).{RESET}")
        # else:
        lg.info(f"{NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Check position details for confirmation).{RESET}")
        return True

    except ccxt.ExchangeError as e:
        # Handle specific exchange errors related to leverage setting
        err_str = str(e).lower()
        exchange_code = getattr(e, 'code', None) # Get exchange-specific error code if available
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {exchange_code}){RESET}")

        # Add hints for common errors based on exchange and code
        if 'bybit' in exchange.id.lower():
             if exchange_code == 110045 or "leverage not modified" in err_str:
                 lg.info(f"{NEON_YELLOW}Leverage for {symbol} likely already set to {leverage}x (Exchange confirmation).{RESET}")
                 return True # Treat as success
             elif exchange_code in [110028, 110009, 110055] or "margin mode" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) compatibility with leverage setting. May need to set margin mode first.{RESET}")
             elif exchange_code == 110044 or "risk limit" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed the current risk limit tier for {symbol}. Check Bybit Risk Limits.{RESET}")
             elif exchange_code == 110013 or "parameter error" in err_str:
                  lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid (e.g., too high) for {symbol}. Check allowed range.{RESET}")
             elif "set margin mode" in err_str:
                   lg.error(f"{NEON_YELLOW} >> Hint: Operation might require setting margin mode first/again using `set_margin_mode`.{RESET}")
        # Add hints for other exchanges if needed
        # elif 'binance' in exchange.id.lower(): ...

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
        # Network errors are usually temporary, but setting leverage is critical
        # Consider retrying or aborting trade based on strategy
    except Exception as e:
        # Catch any other unexpected errors
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    # Return False if any error occurred that wasn't handled as success
    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL" (determines side)
    position_size: Decimal, # Size in base currency or contracts
    market_info: Dict, # Market details
    logger: Optional[logging.Logger] = None,
    order_type: str = 'market', # 'market' or 'limit'
    limit_price: Optional[Decimal] = None, # Required if order_type is 'limit'
    reduce_only: bool = False, # True for closing orders
    params: Optional[Dict] = None # Extra parameters for create_order
) -> Optional[Dict]:
    """
    Places an order (market or limit) using CCXT.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        trade_signal: "BUY" or "SELL".
        position_size: Order size (Decimal).
        market_info: Market dictionary.
        logger: Logger instance.
        order_type: 'market' or 'limit'.
        limit_price: Price for limit order (Decimal). Required if order_type='limit'.
        reduce_only: Set True for closing/reducing positions.
        params: Additional exchange-specific parameters.

    Returns:
        The order dictionary from CCXT on success, None on failure.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"

    # --- Validate Inputs ---
    try:
        amount_float = float(position_size)
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Invalid position size ({position_size}).")
            return None
    except Exception as e:
        lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Failed to convert size {position_size} to float: {e}")
        return None

    if order_type == 'limit' and (limit_price is None or not isinstance(limit_price, Decimal) or limit_price <= 0):
        lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Limit order requested but invalid limit_price ({limit_price}) provided.")
        return None

    # --- Prepare Parameters ---
    order_params = {
        # Bybit V5 specific: positionIdx=0 for one-way mode is often required
        #'positionIdx': 0, # Uncomment if explicitly needed for your Bybit mode
        'reduceOnly': reduce_only,
    }
    # Merge external params, ensuring internal ones (like reduceOnly) take precedence if conflict
    if params:
         order_params.update(params) # External params added first
         order_params['reduceOnly'] = reduce_only # Ensure our flag overrides external

    # Adjust params for closing orders if needed
    if reduce_only:
        # Using IOC for market close orders helps prevent partial fills hanging
        if order_type == 'market':
             order_params['timeInForce'] = 'IOC' # ImmediateOrCancel

    # --- Log Order Details ---
    log_price = f"Limit @ {limit_price}" if order_type == 'limit' else "Market"
    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type.upper()} order for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {size_unit}")
    if order_type == 'limit': lg.info(f"  Limit Price: {limit_price}")
    lg.info(f"  ReduceOnly: {reduce_only}")
    lg.info(f"  Params: {order_params}")


    # --- Execute Order ---
    try:
        order: Optional[Dict] = None
        if order_type == 'market':
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount_float,
                price=None, # Market orders don't have a price argument
                params=order_params
            )
        elif order_type == 'limit':
            # Ensure limit_price is converted to float for CCXT
            price_float = float(limit_price) if limit_price else None
            if price_float:
                order = exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=amount_float,
                    price=price_float, # Price is required for limit orders
                    params=order_params
                )
            else: # Should be caught by validation, but double-check
                 lg.error("Internal Error: Limit order type but price_float is invalid.")
                 return None
        else:
            lg.error(f"Unsupported order type '{order_type}' in place_trade function.")
            return None

        # --- Log Success ---
        if order:
            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'N/A') # e.g., 'open', 'closed', 'canceled'
            filled_amount = order.get('filled', 0.0)
            avg_price = order.get('average') # Price at which it was filled (if applicable)

            lg.info(f"{NEON_GREEN}{action_desc} Trade Placed Successfully!{RESET}")
            lg.info(f"  Order ID: {order_id}, Initial Status: {order_status}")
            if filled_amount > 0: lg.info(f"  Filled Amount: {filled_amount}")
            if avg_price: lg.info(f"  Average Fill Price: {avg_price}")
            lg.debug(f"Raw order response ({symbol} {side} {action_desc}): {order}")
            return order
        else:
             # This case should ideally not be reached if create_order worked without error
             lg.error(f"{NEON_RED}Order placement call returned None without raising an exception for {symbol}.{RESET}")
             return None

    # --- Handle Specific CCXT Exceptions ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} {order_type} order ({symbol}): {e}{RESET}")
        # Log current balance if possible/helpful
        try: balance = fetch_balance(exchange, QUOTE_CURRENCY, lg); lg.info(f"Current Balance: {balance} {QUOTE_CURRENCY}")
        except: pass
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {side} {order_type} order ({symbol}): {e}{RESET}")
        lg.error(f"  > Used Parameters: amount={amount_float}, price={limit_price}, params={order_params}")
        # Add hints based on common InvalidOrder reasons
        if "Order price is not following the tick size" in str(e): lg.error("  >> Hint: Check limit_price alignment with market tick size.")
        if "Order size is not following the step size" in str(e): lg.error("  >> Hint: Check position_size alignment with market amount step size.")
        if "minNotional" in str(e) or "cost" in str(e).lower(): lg.error("  >> Hint: Order cost might be below the minimum required by the exchange.")
        # Handle reduce-only specific errors
        exchange_code = getattr(e, 'code', None)
        if reduce_only and exchange_code == 110014: # Bybit: Reduce-only order failed
             lg.error(f"{NEON_YELLOW}  >> Hint (Bybit 110014): Reduce-only order failed. Position might be closed, size incorrect, or side wrong?{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing {action_desc} order ({symbol}): {e}{RESET}")
        # Network errors might warrant a retry mechanism outside this function or careful state management
    except ccxt.ExchangeError as e:
        exchange_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action_desc} order ({symbol}): {e} (Code: {exchange_code}){RESET}")
        # Add hints for common exchange errors
        if reduce_only and exchange_code == 110025: # Bybit: Position not found/closed
            lg.warning(f"{NEON_YELLOW} >> Hint (Bybit 110025): Position might have been closed already when trying to place reduce-only order.{RESET}")
        # Add more specific hints...
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing {action_desc} order ({symbol}): {e}{RESET}", exc_info=True)

    # Return None if any exception occurred
    return None


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Required for positionIdx and context
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # Distance in price points (Decimal)
    tsl_activation_price: Optional[Decimal] = None, # Price to activate TSL (Decimal)
) -> bool:
    """
    Internal helper to set SL, TP, or TSL for an existing position using Bybit's V5 API endpoint.
    This function handles the specific parameter formatting and API call structure for Bybit.

    NOTE: This uses a direct API call (`private_post`) because setting SL/TP/TSL simultaneously
    or setting TSL often requires specific endpoints not fully standardized in base CCXT yet.

    Args:
        exchange: CCXT Bybit instance.
        symbol: Trading symbol.
        market_info: Market dictionary.
        position_info: The dictionary representing the open position from `get_open_position`.
        logger: Logger instance.
        stop_loss_price: Target SL price (Decimal).
        take_profit_price: Target TP price (Decimal).
        trailing_stop_distance: Target TSL distance/offset (Decimal).
        tsl_activation_price: Target TSL activation price (Decimal).

    Returns:
        True if the API call was successful (or likely successful), False otherwise.
    """
    lg = logger
    # Ensure this is only called for Bybit (or adapt for other exchanges)
    if 'bybit' not in exchange.id.lower():
        lg.error(f"Protection setting via private_post is currently implemented only for Bybit. Cannot set for {exchange.id}.")
        return False
    # Ensure it's a contract market
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract market).")
        return False # Or True? If no action needed, maybe True. Let's use False as intent failed.
    # Ensure we have position info
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    # --- Get Position Context ---
    pos_side = position_info.get('side')
    if pos_side not in ['long', 'short']:
         lg.error(f"Cannot set protection for {symbol}: Invalid or missing position side ('{pos_side}') in position_info.")
         return False
    # Get position index (Crucial for Bybit Hedge Mode, default 0 for One-Way)
    position_idx = 0 # Default for One-Way mode
    try:
        # Bybit V5 often has positionIdx in info dict
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
             position_idx = int(pos_idx_val)
             lg.debug(f"Using positionIdx: {position_idx} from position info.")
        # Add logic here if positionIdx is stored differently for your Bybit mode/version
    except Exception as idx_err:
        lg.warning(f"Could not parse positionIdx from position info ({idx_err}), using default {position_idx}.")

    # --- Validate Input Protection Values ---
    # Check if they are positive Decimals if provided
    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    # TSL requires both distance and activation price
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0)

    # If no valid protection parameters are given, no need to call API
    if not has_sl and not has_tp and not has_tsl:
         lg.info(f"No valid protection parameters provided for {symbol} (PosIdx: {position_idx}). No protection set/updated.")
         # Consider this success as no action was needed/intended
         return True

    # --- Prepare API Parameters for Bybit V5 /v5/position/set-trading-stop ---
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    # Base parameters for the API call
    params = {
        'category': category,
        'symbol': market_info['id'], # Use exchange-specific ID
        'tpslMode': 'Full',          # Apply to the entire position ('Full' or 'Partial')
        'slTriggerBy': 'LastPrice',  # Trigger SL based on LastPrice (or MarkPrice, IndexPrice)
        'tpTriggerBy': 'LastPrice',  # Trigger TP based on LastPrice (or MarkPrice, IndexPrice)
        # Note: Bybit V5 uses Market orders for SL/TP triggers by default via this endpoint.
        # If Limit orders are needed, different endpoints or parameters might be required.
        # 'slOrderType': 'Market', # Often implicit for this endpoint
        # 'tpOrderType': 'Market', # Often implicit
        'positionIdx': position_idx  # Crucial for hedge mode, 0 for one-way
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} PosIdx: {position_idx}):"]

    # --- Format and Add Protection Parameters ---
    try:
        # Create a temporary analyzer instance to access precision/tick helpers
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info) # Needs dummy DF
        price_prec = analyzer.get_price_precision()
        min_tick = analyzer.get_min_tick_size()

        # Helper to format price using ccxt's price_to_precision
        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try:
                # Convert Decimal to float for ccxt helper
                return exchange.price_to_precision(symbol, float(price_decimal))
            except Exception as e:
                 lg.warning(f"Failed to format price {price_decimal} using price_to_precision: {e}. Price will not be set.")
                 return None

        # --- Trailing Stop Handling ---
        # Bybit V5: Setting 'trailingStop' often overrides 'stopLoss'. Set TSL first if applicable.
        formatted_tsl_distance = None
        formatted_activation_price = None
        if has_tsl:
            # Format TSL distance (requires precision relative to tick size)
            try:
                 # Determine the number of decimal places for the distance based on tick size
                 dist_prec = abs(min_tick.normalize().as_tuple().exponent)
                 # Use decimal_to_precision to format the distance value
                 formatted_tsl_distance = exchange.decimal_to_precision(
                     trailing_stop_distance,
                     exchange.ROUND, # Use standard rounding for distance
                     precision=dist_prec,
                     padding_mode=exchange.NO_PADDING # No extra zeros needed usually
                 )
                 # Ensure the formatted distance is at least the minimum tick size
                 if Decimal(formatted_tsl_distance) < min_tick:
                      lg.warning(f"Calculated TSL distance {formatted_tsl_distance} is less than min tick {min_tick}. Adjusting to min tick.")
                      formatted_tsl_distance = str(min_tick) # Use min tick as string
            except Exception as e:
                 lg.warning(f"Failed to format TSL distance {trailing_stop_distance} using decimal_to_precision: {e}. TSL distance will not be set.")
                 formatted_tsl_distance = None # Mark as failed

            # Format activation price
            formatted_activation_price = format_price(tsl_activation_price)

            # Add to params only if both parts are valid
            if formatted_tsl_distance and formatted_activation_price and Decimal(formatted_tsl_distance) > 0:
                params['trailingStop'] = formatted_tsl_distance
                params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                # If TSL is successfully set, the exchange usually ignores 'stopLoss' param.
                # We mark `has_sl` as False to prevent adding the 'stopLoss' param later.
                has_sl = False
                lg.debug("TSL parameters added. Fixed SL will be ignored by the exchange.")
            else:
                lg.error(f"Failed to format valid TSL parameters for {symbol}. TSL will not be set.")
                has_tsl = False # Mark TSL setting as failed

        # --- Fixed Stop Loss Handling ---
        # Add 'stopLoss' only if TSL was *not* successfully prepared to be set.
        if has_sl:
            formatted_sl = format_price(stop_loss_price)
            if formatted_sl:
                params['stopLoss'] = formatted_sl
                log_parts.append(f"  Fixed SL: {formatted_sl}")
            else:
                has_sl = False # Mark SL setting as failed if formatting failed

        # --- Fixed Take Profit Handling ---
        if has_tp:
            formatted_tp = format_price(take_profit_price)
            if formatted_tp:
                params['takeProfit'] = formatted_tp
                log_parts.append(f"  Fixed TP: {formatted_tp}")
            else:
                has_tp = False # Mark TP setting as failed if formatting failed

    except Exception as fmt_err:
         lg.error(f"Error during formatting/preparation of protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False # Cannot proceed if formatting fails


    # --- Check if any protection parameters remain to be set ---
    # Check the actual keys added to the params dictionary
    if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
        lg.warning(f"No valid protection parameters could be formatted or remained after adjustments for {symbol} (PosIdx: {position_idx}). No API call made.")
        # If the intent was to set nothing (e.g., all inputs were invalid/None), return True.
        # If formatting failed for valid inputs, it should return False.
        # Let's assume if we reach here with no params, it was due to formatting failures or intent=None.
        # We return False if any *initial* valid protection was requested but failed formatting.
        # If no initial protection was requested, we already returned True earlier.
        return False if (stop_loss_price or take_profit_price or (trailing_stop_distance and tsl_activation_price)) else True


    # --- Make the API Call ---
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: exchange.private_post('/v5/position/set-trading-stop', params={params})")

    try:
        # Use CCXT's generic private_post method to call the specific Bybit endpoint
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        # --- Parse Bybit V5 Response ---
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {}) # Extra info, often empty

        if ret_code == 0:
            # Success code 0, but check message for nuances
            if "not modified" in ret_msg.lower():
                 # This happens if the SL/TP/TSL are already set to the target values
                 lg.info(f"{NEON_YELLOW}Position protection already set to target values or only partially modified for {symbol} (PosIdx: {position_idx}). Response: {ret_msg}{RESET}")
            else:
                 # Generic success message
                 lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol} (PosIdx: {position_idx}).{RESET}")
            return True # Success
        else:
            # API call failed, log error details
            lg.error(f"{NEON_RED}Failed to set protection for {symbol} (PosIdx: {position_idx}): {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            # Add hints for common Bybit V5 error codes for this endpoint
            if ret_code == 110013: # Parameter error
                 lg.error(f"{NEON_YELLOW} >> Hint (110013 - Parameter Error): Check SL/TP prices vs entry price, TSL distance/activation validity, tick size compliance, tpslMode.{RESET}")
            elif ret_code == 110036: # TSL Active price invalid
                 lg.error(f"{NEON_YELLOW} >> Hint (110036 - TSL Price Invalid): TSL Activation price '{params.get('activePrice')}' likely invalid (already passed? wrong side? too close to current price?).{RESET}")
            elif ret_code == 110086: # SL Price cannot be equal to TP price
                 lg.error(f"{NEON_YELLOW} >> Hint (110086): Stop Loss price cannot be equal to Take Profit price.{RESET}")
            elif ret_code == 110043: # Position status is not normal
                  lg.error(f"{NEON_YELLOW} >> Hint (110043): Position status prevents modification (e.g., during liquidation?).{RESET}")
            elif ret_code == 110025: # Position not found / closed
                 lg.error(f"{NEON_YELLOW} >> Hint (110025): Position may have closed before protection could be set, or positionIdx mismatch?{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower(): # Check message substring
                 lg.error(f"{NEON_YELLOW} >> Hint: Trailing Stop distance '{params.get('trailingStop')}' likely invalid (too small? too large? violates tick size rules?).{RESET}")
            # Add more specific error code handling as encountered...
            return False # Failure

    except ccxt.AuthenticationError as e:
         lg.error(f"{NEON_RED}Authentication error during protection API call for {symbol}: {e}{RESET}")
         # Auth errors usually mean API keys are wrong/expired, not retryable
         return False
    except ccxt.NetworkError as e:
         lg.error(f"{NEON_RED}Network error during protection API call for {symbol}: {e}{RESET}")
         # Network errors might be temporary, but failure could leave position unprotected.
         # Consider state management or manual intervention flags.
         return False # Assume failure for now
    except Exception as e:
        # Catch any other unexpected errors during the API call
        lg.error(f"{NEON_RED}Unexpected error during protection API call for {symbol}: {e}{RESET}", exc_info=True)
        return False # Failure

    # Fallthrough case, should ideally not be reached
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Confirmed position dict from get_open_position
    config: Dict[str, Any], # Bot configuration
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional: Set TP alongside TSL
) -> bool:
    """
    Calculates Trailing Stop Loss parameters based on configuration and position details,
    then calls the internal helper `_set_position_protection` to set the TSL (and optionally TP)
    on the exchange (specifically implemented for Bybit V5).

    Args:
        exchange: CCXT Bybit instance.
        symbol: Trading symbol.
        market_info: Market dictionary.
        position_info: Dictionary representing the open position.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional target TP price (Decimal) to set simultaneously.

    Returns:
        True if TSL (and TP if provided) was successfully set/requested, False otherwise.
    """
    lg = logger

    # Check if TSL is enabled in config
    if not config.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss is disabled in config for {symbol}. Skipping TSL setup.")
        # Return True because no action was intended, or False because TSL wasn't set?
        # Let's return False to indicate TSL specifically was not actioned.
        return False

    # --- Validate TSL Config Parameters ---
    try:
        # Convert config values to Decimal for calculation
        callback_rate_str = config.get("trailing_stop_callback_rate", "0.005") # e.g., 0.5%
        activation_perc_str = config.get("trailing_stop_activation_percentage", "0.003") # e.g., 0.3% profit move
        callback_rate = Decimal(str(callback_rate_str))
        activation_percentage = Decimal(str(activation_perc_str))
    except Exception as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format in config ({symbol}): {e}. Cannot calculate TSL.{RESET}")
        lg.error(f"  >> Check 'trailing_stop_callback_rate' ({callback_rate_str}) and 'trailing_stop_activation_percentage' ({activation_perc_str}).")
        return False
    # Ensure callback rate is positive
    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid 'trailing_stop_callback_rate' ({callback_rate}) in config. Must be positive for {symbol}.{RESET}")
        return False
    # Ensure activation percentage is non-negative
    if activation_percentage < 0:
         lg.error(f"{NEON_RED}Invalid 'trailing_stop_activation_percentage' ({activation_percentage}) in config. Cannot be negative for {symbol}.{RESET}")
         return False

    # --- Get Required Position Info ---
    try:
        # Use the processed Decimal entry price if available
        entry_price = position_info.get('entryPriceDecimal')
        side = position_info.get('side')

        # Check if essential info is present and valid
        if entry_price is None or not isinstance(entry_price, Decimal) or entry_price <= 0:
            lg.error(f"{NEON_RED}Missing or invalid entry price ({entry_price}) in position info for TSL calc ({symbol}).{RESET}")
            return False
        if side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing or invalid position side ('{side}') in position info for TSL calc ({symbol}).{RESET}")
            return False
    except Exception as e:
        # Catch errors if position_info structure is unexpected
        lg.error(f"{NEON_RED}Error accessing position info for TSL calculation ({symbol}): {e}.{RESET}")
        lg.debug(f"Position info received: {position_info}")
        return False

    # --- Calculate TSL Parameters ---
    try:
        # Need market helpers for precision and tick size
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance
        price_precision = analyzer.get_price_precision()
        price_rounding = Decimal('1e-' + str(price_precision)) # Quantizer for price
        min_tick_size = analyzer.get_min_tick_size()

        # 1. Calculate Activation Price
        activation_price: Optional[Decimal] = None
        # Calculate offset based on percentage of entry price
        activation_offset = entry_price * activation_percentage

        if side == 'long':
            # Activation price is entry + offset, rounded UP towards profit
            raw_activation = entry_price + activation_offset
            activation_price = raw_activation.quantize(price_rounding, rounding=ROUND_UP)
            # Ensure activation is strictly > entry if percentage > 0
            if activation_percentage > 0 and activation_price <= entry_price:
                activation_price = (entry_price + min_tick_size).quantize(price_rounding, rounding=ROUND_UP)
                lg.debug(f"Adjusted LONG TSL activation price to be at least one tick above entry: {activation_price}")
            # For immediate activation (0%), set slightly above entry to meet exchange requirement (usually needs to be different from entry)
            elif activation_percentage == 0:
                 activation_price = (entry_price + min_tick_size).quantize(price_rounding, rounding=ROUND_UP)
                 lg.debug(f"Immediate TSL activation (0%) requested. Setting activation slightly above entry: {activation_price}")

        else: # side == 'short'
            # Activation price is entry - offset, rounded DOWN towards profit
            raw_activation = entry_price - activation_offset
            activation_price = raw_activation.quantize(price_rounding, rounding=ROUND_DOWN)
            # Ensure activation is strictly < entry if percentage > 0
            if activation_percentage > 0 and activation_price >= entry_price:
                 activation_price = (entry_price - min_tick_size).quantize(price_rounding, rounding=ROUND_DOWN)
                 lg.debug(f"Adjusted SHORT TSL activation price to be at least one tick below entry: {activation_price}")
            # For immediate activation (0%), set slightly below entry
            elif activation_percentage == 0:
                 activation_price = (entry_price - min_tick_size).quantize(price_rounding, rounding=ROUND_DOWN)
                 lg.debug(f"Immediate TSL activation (0%) requested. Setting activation slightly below entry: {activation_price}")

        # Validate calculated activation price
        if activation_price is None or activation_price <= 0:
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) is invalid for {symbol}. Cannot set TSL.{RESET}")
             return False

        # 2. Calculate Trailing Stop Distance (based on callback rate * activation price)
        # Bybit V5 'trailingStop' parameter is the distance/offset value.
        # The distance should ideally align with tick size.
        trailing_distance_raw = activation_price * callback_rate
        # Round the distance UP to the nearest tick size increment (more conservative trail)
        if min_tick_size > 0:
            trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        else:
            # Should not happen, but fallback if tick size is zero
            lg.warning("Min tick size is zero, cannot round trailing distance accurately.")
            trailing_distance = trailing_distance_raw # Use raw value

        # Ensure distance is at least one tick size
        if min_tick_size > 0 and trailing_distance < min_tick_size:
            lg.warning(f"Calculated TSL distance {trailing_distance} is smaller than min tick {min_tick_size}. Adjusting to min tick.")
            trailing_distance = min_tick_size
        # Ensure distance is positive
        if trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance is zero or negative ({trailing_distance}) for {symbol}. Cannot set TSL.{RESET}")
             return False

        # --- Log Calculated Parameters ---
        lg.info(f"Calculated TSL Parameters for {symbol} ({side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price (Target): {activation_price:.{price_precision}f}")
        lg.info(f"  => Trailing Distance (Target): {trailing_distance:.{price_precision}f}") # Log distance with price precision
        # Log TP if it's being set alongside
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0:
             tp_fmt = f"{take_profit_price:.{price_precision}f}"
             lg.info(f"  Take Profit Price (Target): {tp_fmt} (Will be set simultaneously)")
        else:
            lg.debug("  Take Profit: Not being set or updated with TSL.")


        # 3. Call the internal helper function to set TSL (and TP if provided)
        # Pass None for stop_loss_price as TSL overrides it on Bybit V5
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info, # Pass the whole position dict
            logger=lg,
            stop_loss_price=None, # Explicitly None when setting TSL
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        # Catch any unexpected errors during calculation or the API call preparation
        lg.error(f"{NEON_RED}Unexpected error calculating or preparing TSL parameters for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Performs one cycle of analysis and trading logic for a single symbol.
    Fetches data, analyzes indicators, generates signals, manages positions (entry, exit, BE, TSL).
    """
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- Get Market Info ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping cycle.{RESET}")
        return # Cannot proceed without market info

    # --- Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe for {symbol}. Skipping cycle.")
         return

    kline_limit = 500 # Fetch ample data for indicators
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50: # Check for minimum reasonable data length
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping cycle.{RESET}")
        return

    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
         lg.warning(f"{NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close from klines as fallback.{RESET}")
         try:
             # Ensure index is datetime and sorted
             if isinstance(klines_df.index, pd.DatetimeIndex):
                 last_close_val = klines_df['close'].iloc[-1]
                 if pd.notna(last_close_val) and last_close_val > 0:
                      current_price = Decimal(str(last_close_val))
                      lg.info(f"Using last close price as current price: {current_price}")
                 else:
                     lg.error(f"{NEON_RED}Last close price from klines is invalid ({last_close_val}). Cannot proceed without current price.{RESET}")
                     return
             else:
                 lg.error(f"{NEON_RED}Kline DataFrame index is not DatetimeIndex. Cannot reliably get last close.{RESET}"); return
         except IndexError:
             lg.error(f"{NEON_RED}Kline DataFrame is empty or index error getting last close.{RESET}"); return
         except Exception as e:
             lg.error(f"{NEON_RED}Error getting last close price from klines: {e}. Cannot proceed.{RESET}"); return

    # Fetch order book data only if the indicator is enabled and weighted
    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    orderbook_enabled = config.get("indicators",{}).get("orderbook", False)
    orderbook_weight = Decimal(str(active_weights.get("orderbook", "0"))) # Default weight 0
    if orderbook_enabled and orderbook_weight != 0:
         lg.debug(f"Fetching order book for {symbol} (Weight: {orderbook_weight})...")
         orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)
         if not orderbook_data:
             lg.warning(f"{NEON_YELLOW}Failed to fetch orderbook data for {symbol}, proceeding without it.{RESET}")
    else:
         lg.debug(f"Orderbook analysis skipped (Disabled or Zero Weight).")


    # --- Analyze Data ---
    analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    if not analyzer.indicator_values:
         lg.error(f"{NEON_RED}Indicator calculation failed or produced no values for {symbol}. Skipping signal generation.{RESET}")
         return

    # --- Generate Signal ---
    signal = analyzer.generate_trading_signal(current_price, orderbook_data) # Returns "BUY", "SELL", or "HOLD"

    # --- Calculate Potential TP/SL (based on current price estimate) ---
    # This SL is primarily for position sizing calculation.
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal)
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr = analyzer.indicator_values.get("ATR") # Should be Decimal

    # --- Log Analysis Summary ---
    lg.info(f"Current Price: {current_price:.{price_precision}f}")
    lg.info(f"ATR: {current_atr:.{price_precision+1}f}" if isinstance(current_atr, Decimal) else 'ATR: N/A')
    lg.info(f"Calculated Initial SL (for sizing): {sl_calc if sl_calc else 'N/A'}")
    lg.info(f"Calculated Initial TP (potential target): {tp_calc if tp_calc else 'N/A'}")
    tsl_enabled = config.get('enable_trailing_stop')
    be_enabled = config.get('enable_break_even')
    time_exit_minutes = config.get('time_based_exit_minutes')
    time_exit_str = f"{time_exit_minutes} min" if time_exit_minutes else "Disabled"
    lg.info(f"Position Management: TSL={'Enabled' if tsl_enabled else 'Disabled'}, BE={'Enabled' if be_enabled else 'Disabled'}, TimeExit={time_exit_str}")


    # --- Trading Execution Logic ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading is disabled in config. Analysis complete for {symbol}.")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # --- Check Existing Position ---
    # Use a clean fetch here to get the latest state before making decisions
    open_position = get_open_position(exchange, symbol, lg) # Returns dict or None

    # ==============================================
    # === Scenario 1: No Open Position           ===
    # ==============================================
    if open_position is None:
        if len(open_position) >= config.get("max_concurrent_positions", 1):
            lg.info(f"Max concurrent positions ({config.get('max_concurrent_positions', 1)}) reached. Not entering new position for {symbol}.")

        elif signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***")

            # 1. Fetch Balance
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative.{RESET}")
                return # Stop trade sequence

            # 2. Check if Initial SL for sizing is valid
            if sl_calc is None:
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed (ATR invalid?). Cannot calculate position size.{RESET}")
                 return # Stop trade sequence

            # 3. Set Leverage (if applicable)
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1)) # Default to 1x if not set
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage to {leverage}x.{RESET}")
                         return # Stop trade sequence
                else: lg.info(f"Leverage setting skipped: Leverage config is zero or negative ({leverage}).")
            else:
                lg.info(f"Leverage setting skipped (Spot market).")

            # 4. Calculate Position Size
            # Use the initial SL calculated earlier for sizing
            # Use current_price as the entry price estimate for sizing
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_calc,
                entry_price=current_price,
                market_info=market_info,
                exchange=exchange,
                logger=lg
            )
            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Position size calculation failed or resulted in zero/negative ({position_size}).{RESET}")
                return # Stop trade sequence

            # 5. Determine Order Type and Price
            entry_order_type = config.get("entry_order_type", "market")
            limit_entry_price: Optional[Decimal] = None

            if entry_order_type == "limit":
                 offset_buy = Decimal(str(config.get("limit_order_offset_buy", "0.0005")))
                 offset_sell = Decimal(str(config.get("limit_order_offset_sell", "0.0005")))
                 rounding_factor = Decimal('1e-' + str(price_precision))

                 if signal == "BUY":
                      raw_limit = current_price * (Decimal(1) - offset_buy)
                      # Round down for buy limit (get a potentially better price)
                      limit_entry_price = raw_limit.quantize(rounding_factor, rounding=ROUND_DOWN)
                 else: # SELL
                      raw_limit = current_price * (Decimal(1) + offset_sell)
                      # Round up for sell limit (get a potentially better price)
                      limit_entry_price = raw_limit.quantize(rounding_factor, rounding=ROUND_UP)

                 # Ensure limit price is positive
                 if limit_entry_price <= 0:
                      lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Calculated limit entry price is non-positive ({limit_entry_price}). Switching to Market order.{RESET}")
                      entry_order_type = "market"
                      limit_entry_price = None
                 else:
                      lg.info(f"Calculated Limit Entry Price for {signal}: {limit_entry_price}")

            # 6. Place Entry Order
            lg.info(f"==> Placing {signal} {entry_order_type.upper()} order | Size: {position_size} <==")
            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                trade_signal=signal,
                position_size=position_size,
                market_info=market_info,
                logger=lg,
                order_type=entry_order_type,
                limit_price=limit_entry_price, # Will be None for market orders
                reduce_only=False,
                params=None # Add specific entry params if needed
            )

            # 7. Handle Order Placement Result
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                order_status = trade_order.get('status') # 'open', 'closed', 'canceled', etc.

                # If Market Order -> Confirm Position Immediately
                if entry_order_type == 'market':
                    confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                    lg.info(f"Market order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
                    time.sleep(confirm_delay)

                    lg.info(f"Attempting position confirmation for {symbol} after market order {order_id}...")
                    confirmed_position = get_open_position(exchange, symbol, lg)

                    if confirmed_position:
                        lg.info(f"{NEON_GREEN}Position Confirmed after Market Order!{RESET}")
                        # Proceed to set protection based on actual entry price
                        try:
                            entry_price_actual = confirmed_position.get('entryPriceDecimal')
                            if entry_price_actual is None or entry_price_actual <= 0:
                                lg.warning(f"Could not get valid actual entry price from confirmed position. Using initial estimate {current_price} for protection.")
                                entry_price_actual = current_price # Fallback

                            lg.info(f"Actual Entry Price: ~{entry_price_actual:.{price_precision}f}")

                            # Recalculate protection levels based on ACTUAL entry price
                            _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                            # Set Protection (TSL or Fixed SL/TP)
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                 lg.info(f"Setting Exchange Trailing Stop Loss (TP target: {tp_final})...")
                                 protection_set_success = set_trailing_stop_loss(
                                     exchange=exchange, symbol=symbol, market_info=market_info,
                                     position_info=confirmed_position, config=config, logger=lg,
                                     take_profit_price=tp_final # Pass recalculated TP target
                                 )
                            else:
                                 # Set Fixed SL/TP if TSL is disabled
                                 lg.info(f"Setting Fixed SL ({sl_final}) and TP ({tp_final})...")
                                 if sl_final or tp_final: # Only set if values are valid
                                     protection_set_success = _set_position_protection(
                                         exchange=exchange, symbol=symbol, market_info=market_info,
                                         position_info=confirmed_position, logger=lg,
                                         stop_loss_price=sl_final, take_profit_price=tp_final
                                     )
                                 else:
                                     lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed based on actual entry or returned None. No fixed protection set.{RESET}")

                            # Log final status
                            if protection_set_success:
                                 lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                            else:
                                 lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ===")
                                 lg.warning(f"{NEON_YELLOW}>>> MANUAL MONITORING REQUIRED! <<<")

                        except Exception as post_trade_err:
                             lg.error(f"{NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                             lg.warning(f"{NEON_YELLOW}Position is open but protection setup failed. Manual check needed!{RESET}")
                    else:
                        # Market order placed, but position not confirmed
                        lg.error(f"{NEON_RED}Market trade order {order_id} placed, but FAILED TO CONFIRM open position after {confirm_delay}s delay!{RESET}")
                        lg.warning(f"{NEON_YELLOW}Order may have failed, been rejected, or API/exchange delay. Manual investigation required!{RESET}")

                # If Limit Order -> Log and wait for next cycle (or implement order monitoring)
                elif entry_order_type == 'limit':
                     if order_status == 'open':
                         lg.info(f"Limit order {order_id} placed successfully and is OPEN.")
                         lg.info("Will check status and set protection on next cycle if filled.")
                         # Optional: Add logic here to monitor the open limit order specifically
                         # e.g., store order_id and check its status in subsequent loops
                     elif order_status == 'closed':
                          lg.warning(f"Limit order {order_id} filled immediately (status: {order_status}). Attempting confirmation...")
                          # Treat like market order - try to confirm and set protection
                          # (Add similar confirmation/protection logic as above)
                          lg.info(f"Attempting position confirmation for {symbol} after immediate limit fill {order_id}...")
                          confirmed_position = get_open_position(exchange, symbol, lg)
                          if confirmed_position:
                              # ... (Add protection setting logic here, identical to market order success path) ...
                               lg.info(f"{NEON_GREEN}Position Confirmed after Immediate Limit Fill!{RESET}")
                               # Proceed to set protection based on actual entry price
                               try:
                                   entry_price_actual = confirmed_position.get('entryPriceDecimal')
                                   if entry_price_actual is None or entry_price_actual <= 0:
                                       lg.warning(f"Could not get valid actual entry price from confirmed position. Using limit price {limit_entry_price} for protection.")
                                       entry_price_actual = limit_entry_price # Fallback to intended limit

                                   lg.info(f"Actual Entry Price: ~{entry_price_actual:.{price_precision}f}")
                                   _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                                   protection_set_success = False
                                   if config.get("enable_trailing_stop", False):
                                        lg.info(f"Setting Exchange Trailing Stop Loss (TP target: {tp_final})...")
                                        protection_set_success = set_trailing_stop_loss(
                                            exchange=exchange, symbol=symbol, market_info=market_info,
                                            position_info=confirmed_position, config=config, logger=lg,
                                            take_profit_price=tp_final
                                        )
                                   else:
                                        lg.info(f"Setting Fixed SL ({sl_final}) and TP ({tp_final})...")
                                        if sl_final or tp_final:
                                            protection_set_success = _set_position_protection(
                                                exchange=exchange, symbol=symbol, market_info=market_info,
                                                position_info=confirmed_position, logger=lg,
                                                stop_loss_price=sl_final, take_profit_price=tp_final
                                            )
                                        else: lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed. No fixed protection set.{RESET}")

                                   if protection_set_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                                   else: lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ==="); lg.warning(f"{NEON_YELLOW}>>> MANUAL MONITORING REQUIRED! <<<")
                               except Exception as post_trade_err:
                                    lg.error(f"{NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                                    lg.warning(f"{NEON_YELLOW}Position open but protection setup failed. Manual check needed!{RESET}")
                          else:
                              lg.error(f"{NEON_RED}Limit order {order_id} reported 'closed', but FAILED TO CONFIRM open position! Manual check needed!{RESET}")
                     else:
                          # Limit order failed or was cancelled immediately
                          lg.error(f"Limit order {order_id} placement resulted in status: {order_status}. Trade did not open.")
            else:
                # place_trade returned None
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). Order placement function returned None. See previous logs. ===")
        else: # signal == HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No entry action taken.")

    # ==============================================
    # === Scenario 2: Existing Open Position     ===
    # ==============================================
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size = open_position.get('contractsDecimal', Decimal('0'))
        entry_price = open_position.get('entryPriceDecimal')
        pos_timestamp_ms = open_position.get('timestamp_ms') # Get timestamp in ms

        lg.info(f"Managing existing {pos_side.upper()} position for {symbol}. Size: {pos_size}, Entry: {entry_price}")

        # --- Check for Exit Signal ---
        exit_signal_triggered = (pos_side == 'long' and signal == "SELL") or \
                                (pos_side == 'short' and signal == "BUY")

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position. Closing position... ***{RESET}")
            try:
                # Determine close side based on position side
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close = abs(pos_size) # Close the absolute size
                if size_to_close <= 0:
                    raise ValueError(f"Position size to close is zero or negative ({size_to_close}). Cannot close.")

                lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                # Use market order for closing to ensure exit
                close_order = place_trade(
                    exchange=exchange, symbol=symbol, trade_signal=close_side_signal,
                    position_size=size_to_close, market_info=market_info, logger=lg,
                    order_type='market', reduce_only=True
                )

                if close_order:
                    lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                    # Assuming market close is effective. Further checks could poll position status again.
                    # Important: Exit the current cycle after placing close order to avoid conflicting actions
                    return
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check/intervention required!{RESET}")
                    # Consider what state the bot should be in - maybe skip further management this cycle

            except Exception as close_err:
                 lg.error(f"{NEON_RED}Error attempting to close position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")
            # After attempting close, exit the management logic for this cycle
            return

        # --- If No Exit Signal, Proceed with Position Management ---
        else:
            lg.info(f"Signal ({signal}) allows holding the existing {pos_side} position. Performing position management checks...")

            # --- Check for Time-Based Exit ---
            time_exit_minutes_config = config.get("time_based_exit_minutes")
            if time_exit_minutes_config and time_exit_minutes_config > 0:
                if pos_timestamp_ms:
                    try:
                        current_time_ms = time.time() * 1000
                        time_elapsed_ms = current_time_ms - pos_timestamp_ms
                        time_elapsed_minutes = time_elapsed_ms / (1000 * 60)

                        lg.debug(f"Time-Based Exit Check: Elapsed = {time_elapsed_minutes:.2f} min, Limit = {time_exit_minutes_config} min")

                        if time_elapsed_minutes >= time_exit_minutes_config:
                            lg.warning(f"{NEON_YELLOW}*** TIME-BASED EXIT Triggered ({time_elapsed_minutes:.1f} >= {time_exit_minutes_config} min). Closing position... ***{RESET}")
                            # --- Execute Close Logic (similar to above) ---
                            close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                            size_to_close = abs(pos_size)
                            if size_to_close > 0:
                                close_order = place_trade(exchange, symbol, close_side_signal, size_to_close, market_info, lg, order_type='market', reduce_only=True)
                                if close_order: lg.info(f"{NEON_GREEN}Time-based CLOSE order placed successfully for {symbol}. ID: {close_order.get('id', 'N/A')}{RESET}")
                                else: lg.error(f"{NEON_RED}Failed to place time-based CLOSE order for {symbol}. Manual check required!{RESET}")
                            else: lg.warning("Time-based exit triggered but position size is zero. No close order placed.")
                            # Exit management logic after triggering time-based close
                            return
                    except Exception as time_err:
                         lg.error(f"{NEON_RED}Error during time-based exit check: {time_err}{RESET}")
                else:
                    lg.warning("Time-based exit enabled, but position timestamp not found. Cannot perform check.")


            # --- Check if TSL is Active on the Exchange ---
            # We need to rely on the 'trailingStopLossValue' field populated by get_open_position
            is_tsl_active_exchange = False
            try:
                 # Check if the trailing stop distance value is present and positive
                 tsl_value_str = open_position.get('trailingStopLossValue')
                 if tsl_value_str and str(tsl_value_str) != '0': # Check if not None, empty or '0' string
                      tsl_value = Decimal(str(tsl_value_str))
                      if tsl_value > 0:
                           is_tsl_active_exchange = True
                           lg.debug("Exchange Trailing Stop Loss appears to be active for this position.")
            except Exception as tsl_check_err:
                 lg.warning(f"Could not reliably determine if exchange TSL is active: {tsl_check_err}")


            # --- Check Break-Even Conditions ---
            # Only run BE check if:
            # 1. BE is enabled in config.
            # 2. Exchange TSL is NOT currently active (BE usually gets disabled once TSL takes over).
            if config.get("enable_break_even", False) and not is_tsl_active_exchange:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    # Ensure we have valid entry price and ATR
                    if entry_price is None or entry_price <= 0: raise ValueError("Invalid entry price for BE check")
                    if not isinstance(current_atr, Decimal) or current_atr <= 0: raise ValueError("Invalid ATR for BE check")

                    # Get BE config parameters
                    be_trigger_atr_mult_str = config.get("break_even_trigger_atr_multiple", "1.0")
                    be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    profit_target_atr = Decimal(str(be_trigger_atr_mult_str))

                    # Calculate current profit/loss in price points and ATR multiples
                    price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                    # Avoid division by zero if ATR is somehow zero
                    profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('0')

                    lg.debug(f"BE Check: CurrentPrice={current_price:.{price_precision}f}, Entry={entry_price:.{price_precision}f}")
                    lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision}f}, Profit ATRs={profit_in_atr:.2f}, Target ATRs={profit_target_atr}")

                    # Check if profit target is reached
                    if profit_in_atr >= profit_target_atr:
                        # --- Calculate Target Break-Even Stop Price ---
                        # Add/subtract a small offset (in ticks) from entry price
                        tick_offset = min_tick_size * be_offset_ticks
                        be_stop_price: Optional[Decimal] = None
                        if pos_side == 'long':
                            # Place BE SL slightly above entry
                            be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP)
                        else: # short
                            # Place BE SL slightly below entry
                            be_stop_price = (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                        if be_stop_price is None or be_stop_price <= 0:
                             raise ValueError(f"Calculated BE stop price invalid: {be_stop_price}")

                        # --- Get Current Stop Loss from Position Info ---
                        current_sl_price: Optional[Decimal] = None
                        # Check both standard field and info dict fallback
                        current_sl_str = open_position.get('stopLossPrice') or open_position.get('info', {}).get('stopLoss')
                        # Ensure it's not None, empty string, or '0' string before converting
                        if current_sl_str and str(current_sl_str) != '0':
                            try:
                                current_sl_price = Decimal(str(current_sl_str))
                            except Exception as sl_parse_err:
                                lg.warning(f"Could not parse current stop loss '{current_sl_str}': {sl_parse_err}")

                        # --- Determine if SL Update is Needed ---
                        update_be_sl = False
                        if current_sl_price is None:
                            # No current SL set, so set the BE SL
                            update_be_sl = True
                            lg.info("BE triggered: No current SL found. Setting BE SL.")
                        elif pos_side == 'long' and be_stop_price > current_sl_price:
                            # Current SL is below target BE SL, move it up
                            update_be_sl = True
                            lg.info(f"BE triggered: Target BE SL {be_stop_price} is tighter than Current SL {current_sl_price}. Updating.")
                        elif pos_side == 'short' and be_stop_price < current_sl_price:
                            # Current SL is above target BE SL, move it down
                            update_be_sl = True
                            lg.info(f"BE triggered: Target BE SL {be_stop_price} is tighter than Current SL {current_sl_price}. Updating.")
                        else:
                            # BE triggered, but current SL is already at or better than BE target
                            lg.debug(f"BE Triggered, but current SL ({current_sl_price}) is already better than or equal to target BE SL ({be_stop_price}). No SL update needed.")

                        # --- Execute SL Update if Needed ---
                        if update_be_sl:
                            lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                            # Preserve existing Take Profit if one is set
                            current_tp_price: Optional[Decimal] = None
                            current_tp_str = open_position.get('takeProfitPrice') or open_position.get('info', {}).get('takeProfit')
                            if current_tp_str and str(current_tp_str) != '0':
                                 try: current_tp_price = Decimal(str(current_tp_str))
                                 except: pass # Ignore parsing error for TP

                            # Call the protection function to update SL (and preserve TP)
                            success = _set_position_protection(
                                exchange=exchange, symbol=symbol, market_info=market_info,
                                position_info=open_position, logger=lg,
                                stop_loss_price=be_stop_price,
                                take_profit_price=current_tp_price # Pass existing TP to preserve it
                            )
                            if success: lg.info(f"{NEON_GREEN}Break-Even SL set/updated successfully.{RESET}")
                            else: lg.error(f"{NEON_RED}Failed to set/update Break-Even SL.{RESET}")
                            # Optional: If BE SL set successfully, maybe disable further BE checks for this position?

                    else:
                        # Profit target not yet reached
                        lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {profit_target_atr} ATRs).")

                except ValueError as ve: # Catch validation errors for entry price/ATR
                     lg.warning(f"BE Check skipped for {symbol}: {ve}")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check ({symbol}): {be_err}{RESET}", exc_info=True)
            elif is_tsl_active_exchange:
                 lg.debug(f"Break-even check skipped: Exchange Trailing Stop Loss is active.")
            else: # BE disabled in config
                 lg.debug(f"Break-even check skipped: Disabled in config.")

            # --- Placeholder for other potential management logic ---
            # - Check if TSL activation price hit, but TSL failed to set previously? Retry?
            # - Partial profit taking?
            # - Adding to position? (Requires careful risk management adjustment)


    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY # Allow modification of globals

    # Use a general logger for initial setup
    setup_logger("init") # Create logger for initialization phase
    init_logger = logging.getLogger("init") # Get the logger instance

    init_logger.info(f"--- Starting XR Scalper Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    # Load/Update config at start
    CONFIG = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Update global QUOTE_CURRENCY
    init_logger.info(f"Config loaded from {CONFIG_FILE}. Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={ta.version if hasattr(ta, 'version') else 'N/A'}")


    # --- Trading Enabled Warning ---
    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet) Environment.{RESET}")
         else:
              # Extra warning for real money
              init_logger.warning(f"{NEON_RED}!!! CAUTION: USING REAL MONEY ENVIRONMENT !!!{RESET}")

         # Display critical settings before proceeding
         risk_pct = CONFIG.get('risk_per_trade', 0) * 100
         leverage = CONFIG.get('
