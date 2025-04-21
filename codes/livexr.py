```python
# livexy.py
# Enhanced version focusing on stop-loss/take-profit mechanisms, including break-even and trailing stops.

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
getcontext().prec = 28  # Increased precision for calculations, especially intermediate ones
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
# Timezone for logging and display (adjust as needed)
TIMEZONE = ZoneInfo("America/Chicago") # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
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
LOOP_DELAY_SECONDS = 15 # Time between the end of one cycle and the start of the next
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
        "max_concurrent_positions": 1, # Limit open positions for this symbol (common strategy)
        "quote_currency": "USDT", # Currency for balance check and sizing
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True,           # Default to enabling TSL (exchange TSL)
        "trailing_stop_callback_rate": 0.005,   # e.g., 0.5% trail distance (as decimal) from high water mark
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when price moves 0.3% in favor from entry (set to 0 for immediate activation)
        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR
        "break_even_offset_ticks": 2,           # Place BE SL X ticks beyond entry price (in profit direction)
        # --- Position Management ---
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order before checking position status
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
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            updated_config = _ensure_config_keys(config_from_file, default_config)
            if updated_config != config_from_file:
                 try:
                     with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                     print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
                 except IOError as e:
                     print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
            # Validate interval value after loading
            if updated_config.get("interval") not in VALID_INTERVALS:
                print(f"{NEON_RED}Invalid interval '{updated_config.get('interval')}' found in config. Using default '{default_config['interval']}'.{RESET}")
                updated_config["interval"] = default_config["interval"]
                # Optionally save back the corrected interval
                try:
                     with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                     print(f"{NEON_YELLOW}Corrected invalid interval in config file: {filepath}{RESET}")
                except IOError as e:
                     print(f"{NEON_RED}Error writing corrected interval to config file {filepath}: {e}{RESET}")
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
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"livexy_bot_{safe_symbol}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

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

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Consistent time format
    )
    stream_handler.setFormatter(stream_formatter)
    console_log_level = logging.INFO # Change to DEBUG for more verbose console output
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False
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
                'fetchTickerTimeout': 10000,
                'fetchBalanceTimeout': 15000,
                'createOrderTimeout': 20000, # Longer timeout for placing orders
                'cancelOrderTimeout': 15000,
            }
        }

        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys
        account_type_to_test = 'CONTRACT' # Unified might also be 'UNIFIED'
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            balance = exchange.fetch_balance(params={'type': account_type_to_test})
            available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
            logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
        except ccxt.AuthenticationError as auth_err:
             logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
             return None
        except ccxt.ExchangeError as balance_err:
             logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
             try:
                  balance = exchange.fetch_balance() # Fallback
                  available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
                  logger.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
             except Exception as fallback_err:
                  logger.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type.{RESET}")
        except Exception as balance_err:
             logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type.{RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None


# --- CCXT Data Fetching ---
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
                return price
            else:
                lg.warning(f"Failed to get a valid price from ticker attempt {attempts + 1}.")
                # Continue to retry logic if price is None or invalid

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting longer...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 5)
            attempts += 1 # Count this attempt
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Don't retry on most exchange errors (e.g., bad symbol)
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            # Don't retry unexpected errors
            return None

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch a valid current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
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
                  if ohlcv is not None: # Basic check if data was returned
                    break # Success
                  else:
                    lg.warning(f"fetch_ohlcv returned None for {symbol} (Attempt {attempt+1}). Retrying...")

             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                      raise e # Re-raise the last error
             except ccxt.RateLimitExceeded as e:
                 lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {RETRY_DELAY_SECONDS * 5}s... (Attempt {attempt+1})")
                 time.sleep(RETRY_DELAY_SECONDS * 5) # Longer delay for rate limits
             except ccxt.ExchangeError as e:
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 raise e # Re-raise non-network errors immediately

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe}.{RESET}")
             return df

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
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


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and basic validation."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.has['fetchOrderBook']:
                 lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                 return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts+1}/{MAX_API_RETRIES + 1})")
            orderbook = exchange.fetch_order_book(symbol, limit=limit)

            if not orderbook:
                lg.warning(f"fetch_order_book returned None/empty for {symbol} (Attempt {attempts+1}).")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                 lg.warning(f"{NEON_YELLOW}Invalid orderbook structure for {symbol}. Attempt {attempts + 1}. Response: {orderbook}{RESET}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 return orderbook # Return empty book
            else:
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {RETRY_DELAY_SECONDS * 5}s...")
            time.sleep(RETRY_DELAY_SECONDS * 5)
            attempts += 1
            continue
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None

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
        self.df = df
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")
        self.indicator_values: Dict[str, Any] = {} # Stores latest indicator values (float or Decimal)
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_column_names: Dict[str, Optional[str]] = {}

        if not self.weights:
             logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}.")

        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """Helper to find the actual column name generated by pandas_ta."""
        # Simplified patterns for better matching
        expected_patterns = {
            "ATR": [f"ATRr_{self.config.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{self.config.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{self.config.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{self.config.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{self.config.get('cci_window', DEFAULT_CCI_WINDOW)}"], # Might have const suffix
            "Williams_R": [f"WILLR_{self.config.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{self.config.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP"], # Usually just VWAP_D or similar if daily anchor used
            "PSAR_long": [f"PSARl_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{self.config.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}"], # Simpler pattern
            "StochRSI_D": [f"STOCHRSId_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}"], # Simpler pattern
            "RSI": [f"RSI_{self.config.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [f"BBL_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}"], # Simpler pattern
            "BB_Middle": [f"BBM_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}"], # Simpler pattern
            "BB_Upper": [f"BBU_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}"], # Simpler pattern
            "Volume_MA": [f"VOL_SMA_{self.config.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"] # Custom name
        }
        patterns = expected_patterns.get(base_name, [])
        # Search dataframe columns for a match starting with any pattern
        for col in result_df.columns:
             for pattern in patterns:
                 if col.startswith(pattern):
                     return col
        # Fallback: simple base name search (less reliable)
        for col in result_df.columns:
            if base_name.lower() in col.lower():
                self.logger.debug(f"Found column '{col}' for base '{base_name}' using fallback search.")
                return col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {result_df.columns.tolist()}")
        return None


    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check sufficient data length
        # ... (length check logic remains similar) ...
        required_periods = [self.config.get(k, v) for k,v in {
            "atr_period": DEFAULT_ATR_PERIOD, "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
            "momentum_period": DEFAULT_MOMENTUM_PERIOD, "cci_window": DEFAULT_CCI_WINDOW,
            "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW, "mfi_window": DEFAULT_MFI_WINDOW,
            "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
            "rsi_period": DEFAULT_RSI_WINDOW, "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
            "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD, "sma_10_window": DEFAULT_SMA_10_WINDOW,
            "fibonacci_window": DEFAULT_FIB_WINDOW
        }.items()]
        min_required_data = max(required_periods) + 20 if required_periods else 50

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}).{RESET}")
             # Continue, but expect NaNs

        try:
            df_calc = self.df.copy()
            indicators_config = self.config.get("indicators", {})

            # Always calculate ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

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
                # VWAP might need daily anchor depending on ta version/usage
                df_calc.ta.vwap(append=True)
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                if psar_result is not None and not psar_result.empty:
                    df_calc = pd.concat([df_calc, psar_result], axis=1)
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
                stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                if stochrsi_result is not None and not stochrsi_result.empty:
                     df_calc = pd.concat([df_calc, stochrsi_result], axis=1)
                     self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                     self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)

            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = float(self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV))
                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std)
                if bbands_result is not None and not bbands_result.empty:
                    df_calc = pd.concat([df_calc, bbands_result], axis=1)
                    self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                    self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                    self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)

            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}" # Custom name
                df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)
                self.ta_column_names["Volume_MA"] = vol_ma_col_name

            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
             self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol}: {e}{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest values from self.df."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(f"Cannot update latest values: DataFrame empty or last row is all NaN for {self.symbol}.")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}
            return

        try:
            latest = self.df.iloc[-1]
            updated_values = {}

            # Process TA indicator columns
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index:
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            # Store most as float for general use, ATR as Decimal
                            if key == "ATR":
                                updated_values[key] = Decimal(str(value))
                            else:
                                updated_values[key] = float(value)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) for {self.symbol}.")
                            updated_values[key] = np.nan
                    else:
                        updated_values[key] = np.nan
                else:
                    if key in self.ta_column_names: # Log only if calc was attempted
                        self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found in latest data for {self.symbol}. Storing NaN.")
                    updated_values[key] = np.nan

            # Add essential price/volume data as Decimal for precision
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                 value = latest.get(base_col)
                 key_name = base_col.capitalize()
                 if pd.notna(value):
                      try:
                           updated_values[key_name] = Decimal(str(value))
                      except (ValueError, TypeError):
                           self.logger.warning(f"Could not convert base value for '{base_col}' ({value}) to Decimal for {self.symbol}.")
                           updated_values[key_name] = np.nan
                 else:
                      updated_values[key_name] = np.nan

            self.indicator_values = updated_values
            # Format for logging, handling both float and Decimal
            valid_values_log = {}
            for k, v in self.indicator_values.items():
                 if pd.notna(v):
                     if isinstance(v, Decimal):
                          # Determine precision based on type (price or other)
                          prec = self.get_price_precision() if k in ['Open','High','Low','Close','ATR'] else 6
                          valid_values_log[k] = f"{v:.{prec}f}"
                     elif isinstance(v, float):
                          valid_values_log[k] = f"{v:.5f}"
                     else: # Handle other types if necessary
                          valid_values_log[k] = str(v)

            self.logger.debug(f"Latest indicator values updated for {self.symbol}: {valid_values_log}")

        except IndexError:
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be empty/short.")
             self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}
        except Exception as e:
             self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
             self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a specified window using Decimal."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            high_price_raw = df_slice["high"].dropna().max()
            low_price_raw = df_slice["low"].dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                 self.logger.warning(f"Could not find valid high/low for Fibonacci on {self.symbol}.")
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
                    level_price = (high - (diff * Decimal(str(level_pct))))
                    level_price_quantized = level_price.quantize(rounding_factor, rounding=ROUND_DOWN)
                    levels[level_name] = level_price_quantized
            else:
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol}.")
                 level_price_quantized = high.quantize(rounding_factor, rounding=ROUND_DOWN)
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { {k: str(v) for k,v in levels.items()} }")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info."""
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')

            if price_precision_val is not None:
                 if isinstance(price_precision_val, int):
                      if price_precision_val >= 0: return price_precision_val
                 elif isinstance(price_precision_val, (float, str)):
                      tick_size = Decimal(str(price_precision_val))
                      if tick_size > 0: return abs(tick_size.normalize().as_tuple().exponent)

            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                min_price_tick = Decimal(str(min_price_val))
                if min_price_tick > 0:
                    self.logger.debug(f"Inferring price precision from min price limit ({min_price_tick}) for {self.symbol}.")
                    return abs(min_price_tick.normalize().as_tuple().exponent)

            last_close = self.indicator_values.get("Close") # Uses Decimal value
            if last_close and isinstance(last_close, Decimal) and last_close > 0:
                 # Use Decimal's exponent info
                 precision = abs(last_close.normalize().as_tuple().exponent)
                 self.logger.debug(f"Inferring price precision from last close price ({last_close}) as {precision} for {self.symbol}.")
                 return precision

        except Exception as e:
            self.logger.warning(f"Could not determine price precision for {self.symbol}: {e}. Falling back to default.")

        default_precision = 4
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info."""
        try:
            # Try precision.price first (often the tick size as float/str)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None and isinstance(price_precision_val, (float, str)):
                 tick_size = Decimal(str(price_precision_val))
                 if tick_size > 0: return tick_size

            # Fallback: limits.price.min (sometimes represents tick size)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                min_tick_from_limit = Decimal(str(min_price_val))
                if min_tick_from_limit > 0:
                    # Check if this seems plausible as a tick size (not just a minimum order price like 0.1)
                    # A tick size is usually much smaller than 1, often with many decimal places.
                    # This heuristic might need refinement based on exchange.
                    if min_tick_from_limit < Decimal('0.1'):
                         self.logger.debug(f"Using tick size from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                         return min_tick_from_limit
                    else:
                         self.logger.debug(f"limits.price.min ({min_tick_from_limit}) seems too large for tick size, potentially min order price.")

            # Fallback: calculate from decimal places if precision.price was an int
            if price_precision_val is not None and isinstance(price_precision_val, int):
                 tick_size = Decimal('1e-' + str(price_precision_val))
                 self.logger.debug(f"Calculated tick size from precision.price (decimals): {tick_size} for {self.symbol}")
                 return tick_size

        except Exception as e:
             self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using precision fallback.")

        # Final fallback: Use get_price_precision (decimal places)
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.debug(f"Using fallback tick size based on precision places for {self.symbol}: {fallback_tick}")
        return fallback_tick

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels (name, price) to the current price."""
        if not self.fib_levels_data: return []
        if not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal):
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                     self.logger.warning(f"Non-decimal value found in fib_levels_data: {name}={level_price}")

            level_distances.sort(key=lambda x: x['distance'])
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score based on latest values. Returns float score or NaN."""
        ema_short = self.indicator_values.get("EMA_Short") # Float
        ema_long = self.indicator_values.get("EMA_Long") # Float
        current_price_float = float(self.indicator_values.get("Close", np.nan)) # Convert Decimal to float for comparison

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price_float):
            return np.nan

        if current_price_float > ema_short > ema_long: return 1.0
        elif current_price_float < ema_short < ema_long: return -1.0
        else: return 0.0

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {}

        if not self.indicator_values:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values empty.{RESET}")
             return "HOLD"
        core_indicators_present = any(pd.notna(v) for k, v in self.indicator_values.items() if k not in ['Close', 'Volume', 'High', 'Low', 'Open'])
        if not core_indicators_present:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: All core indicator values are NaN.{RESET}")
            return "HOLD"
        if pd.isna(current_price) or current_price <= 0:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
             return "HOLD"

        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
             self.logger.error(f"Active weight set '{self.active_weight_set_name}' missing or empty for {self.symbol}. Cannot generate signal.")
             return "HOLD"

        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue
            weight_str = active_weights.get(indicator_key)
            if weight_str is None: continue

            try:
                weight = Decimal(str(weight_str));
                if weight == 0: continue
            except Exception:
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}'. Skipping.")
                continue

            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method = getattr(self, check_method_name)
                indicator_score = np.nan
                try:
                    if indicator_key == "orderbook":
                         indicator_score = method(orderbook_data, current_price) if orderbook_data else np.nan
                    else:
                         indicator_score = method() # Returns float score or np.nan

                except Exception as e:
                    self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)

                debug_scores[indicator_key] = f"{indicator_score:.2f}" if pd.notna(indicator_score) else "NaN"
                if pd.notna(indicator_score):
                    try:
                        score_decimal = Decimal(str(indicator_score))
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_decimal))
                        score_contribution = clamped_score * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight
                        active_indicator_count += 1
                    except Exception as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} ({indicator_score}): {calc_err}")
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1
            else:
                self.logger.warning(f"Check method '{check_method_name}' not found for enabled/weighted indicator: {indicator_key} ({self.symbol})")

        if total_weight_applied == 0:
             self.logger.warning(f"No indicators contributed to the signal score for {self.symbol}. Defaulting to HOLD.")
             final_signal = "HOLD"
        else:
            threshold = Decimal(str(self.config.get("signal_score_threshold", 1.5)))
            if final_signal_score >= threshold: final_signal = "BUY"
            elif final_signal_score <= -threshold: final_signal = "SELL"
            else: final_signal = "HOLD"

        price_prec = self.get_price_precision()
        log_msg = (
            f"Signal Summary ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set={self.active_weight_set_name}, Indis=[{active_indicator_count}:{nan_indicator_count} NaN], Wgt={total_weight_applied:.2f}, "
            f"Score={final_signal_score:.4f} (Thresh: +/-{threshold:.2f}) "
            f"==> {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        self.logger.debug(f"  Indicator Scores: {debug_scores}") # Log details at debug level

        if final_signal in self.signals: self.signals[final_signal] = 1
        return final_signal


    # --- Indicator Check Methods (returning float score -1.0 to 1.0 or np.nan) ---
    # (Implementations remain largely the same as before, ensure they use self.indicator_values correctly)

    def _check_ema_alignment(self) -> float:
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
             return np.nan
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum") # Float
        if pd.isna(momentum): return np.nan
        # Simple thresholding (adjust thresholds as needed)
        if momentum > 0.1: return 1.0
        if momentum < -0.1: return -1.0
        return momentum * 10.0 # Linear scale between -0.1 and 0.1

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume") # Decimal
        volume_ma_float = self.indicator_values.get("Volume_MA") # Float
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))

        if pd.isna(current_volume) or pd.isna(volume_ma_float) or volume_ma_float <= 0: return np.nan
        volume_ma = Decimal(str(volume_ma_float)) # Convert MA to Decimal for comparison

        try:
            if current_volume > volume_ma * Decimal(str(multiplier)): return 0.7 # High volume significance
            elif current_volume < volume_ma / Decimal(str(multiplier)): return -0.4 # Low volume lack of confirmation
            else: return 0.0 # Neutral volume
        except Exception: return np.nan

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K") # Float
        d = self.indicator_values.get("StochRSI_D") # Float
        if pd.isna(k) or pd.isna(d): return np.nan
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
        score = 0.0
        if k < oversold and d < oversold: score = 1.0
        elif k > overbought and d > overbought: score = -1.0
        diff = k - d
        if abs(diff) > 5: # Crossing potential
             if diff > 0: score = max(score, 0.6) if score >= 0 else 0.6
             else: score = min(score, -0.6) if score <= 0 else -0.6
        elif k > d : score = max(score, 0.2) # Weak bullish
        elif k < d: score = min(score, -0.2) # Weak bearish
        if 40 < k < 60: score *= 0.5 # Dampen in mid-range
        return score

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI") # Float
        if pd.isna(rsi): return np.nan
        if rsi <= 30: return 1.0
        if rsi >= 70: return -1.0
        if rsi < 40: return 0.5
        if rsi > 60: return -0.5
        if 40 <= rsi <= 60: return 0.5 - (rsi - 40) * (1.0 / 20.0)
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI") # Float
        if pd.isna(cci): return np.nan
        if cci <= -150: return 1.0
        if cci >= 150: return -1.0
        if cci < -80: return 0.6
        if cci > 80: return -0.6
        if cci > 0: return -0.1
        if cci < 0: return 0.1
        return 0.0

    def _check_wr(self) -> float:
        wr = self.indicator_values.get("Williams_R") # Float
        if pd.isna(wr): return np.nan
        if wr <= -80: return 1.0
        if wr >= -20: return -1.0
        if wr < -50: return 0.4
        if wr > -50: return -0.4
        return 0.0

    def _check_psar(self) -> float:
        psar_l = self.indicator_values.get("PSAR_long") # Float
        psar_s = self.indicator_values.get("PSAR_short") # Float
        if pd.notna(psar_l) and pd.isna(psar_s): return 1.0 # Uptrend
        elif pd.notna(psar_s) and pd.isna(psar_l): return -1.0 # Downtrend
        else: return 0.0

    def _check_sma_10(self) -> float:
        sma_10 = self.indicator_values.get("SMA10") # Float
        last_close_float = float(self.indicator_values.get("Close", np.nan)) # Float for comparison
        if pd.isna(sma_10) or pd.isna(last_close_float): return np.nan
        if last_close_float > sma_10: return 0.6
        if last_close_float < sma_10: return -0.6
        return 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP") # Float
        last_close_float = float(self.indicator_values.get("Close", np.nan)) # Float for comparison
        if pd.isna(vwap) or pd.isna(last_close_float): return np.nan
        if last_close_float > vwap: return 0.7
        if last_close_float < vwap: return -0.7
        return 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI") # Float
        if pd.isna(mfi): return np.nan
        if mfi <= 20: return 1.0
        if mfi >= 80: return -1.0
        if mfi < 40: return 0.4
        if mfi > 60: return -0.4
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_lower = self.indicator_values.get("BB_Lower") # Float
        bb_upper = self.indicator_values.get("BB_Upper") # Float
        bb_middle = self.indicator_values.get("BB_Middle") # Float
        last_close_float = float(self.indicator_values.get("Close", np.nan)) # Float for comparison
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(last_close_float):
            return np.nan

        if last_close_float < bb_lower: return 1.0
        if last_close_float > bb_upper: return -1.0
        if last_close_float > bb_middle:
             proximity_to_upper = (last_close_float - bb_middle) / (bb_upper - bb_middle) if (bb_upper - bb_middle) > 0 else 0
             return 0.3 * (1 - proximity_to_upper)
        if last_close_float < bb_middle:
             proximity_to_lower = (bb_middle - last_close_float) / (bb_middle - bb_lower) if (bb_middle - bb_lower) > 0 else 0
             return -0.3 * (1 - proximity_to_lower)
        return 0.0

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes order book depth. Returns float score or NaN."""
        if not orderbook_data: return np.nan
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            if not bids or not asks: return np.nan

            num_levels_to_check = 10
            bid_volume_sum = sum(Decimal(str(bid[1])) for bid in bids[:num_levels_to_check])
            ask_volume_sum = sum(Decimal(str(ask[1])) for ask in asks[:num_levels_to_check])
            total_volume = bid_volume_sum + ask_volume_sum
            if total_volume == 0: return 0.0

            obi = (bid_volume_sum - ask_volume_sum) / total_volume
            score = float(obi) # OBI is already -1 to 1

            self.logger.debug(f"Orderbook check ({self.symbol}): Top {num_levels_to_check}: BidVol={bid_volume_sum:.4f}, AskVol={ask_volume_sum:.4f}, OBI={obi:.4f}, Score={score:.4f}")
            return score

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential TP and initial SL levels based on entry estimate, ATR,
        and multipliers. Returns (entry_price_estimate, take_profit, stop_loss), all Decimal or None.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price_estimate, None, None

        atr_val = self.indicator_values.get("ATR") # Should be Decimal
        if not isinstance(atr_val, Decimal) or pd.isna(atr_val) or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid ATR ({atr_val}).{RESET}")
            return entry_price_estimate, None, None
        if not isinstance(entry_price_estimate, Decimal) or pd.isna(entry_price_estimate) or entry_price_estimate <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid entry price estimate ({entry_price_estimate}).{RESET}")
            return entry_price_estimate, None, None

        try:
            entry_price = entry_price_estimate # Use the provided estimate

            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))
            min_tick = self.get_min_tick_size()

            take_profit = None
            stop_loss = None
            tp_offset = atr_val * tp_multiple
            sl_offset = atr_val * sl_multiple

            if signal == "BUY":
                tp_raw = entry_price + tp_offset
                sl_raw = entry_price - sl_offset
                # Quantize TP UP, SL DOWN
                take_profit = tp_raw.quantize(rounding_factor, rounding=ROUND_UP)
                stop_loss = sl_raw.quantize(rounding_factor, rounding=ROUND_DOWN)
            elif signal == "SELL":
                tp_raw = entry_price - tp_offset
                sl_raw = entry_price + sl_offset
                # Quantize TP DOWN, SL UP
                take_profit = tp_raw.quantize(rounding_factor, rounding=ROUND_DOWN)
                stop_loss = sl_raw.quantize(rounding_factor, rounding=ROUND_UP)

            # --- Validation ---
            # Ensure SL is strictly beyond entry by at least one tick
            if signal == "BUY" and stop_loss >= entry_price:
                 stop_loss = (entry_price - min_tick).quantize(rounding_factor, rounding=ROUND_DOWN)
                 self.logger.debug(f"Adjusted BUY SL below entry: {stop_loss}")
            elif signal == "SELL" and stop_loss <= entry_price:
                 stop_loss = (entry_price + min_tick).quantize(rounding_factor, rounding=ROUND_UP)
                 self.logger.debug(f"Adjusted SELL SL above entry: {stop_loss}")

            # Ensure TP is potentially profitable (strictly beyond entry)
            if signal == "BUY" and take_profit <= entry_price:
                 self.logger.warning(f"{NEON_YELLOW}BUY TP calculation non-profitable (TP {take_profit} <= Entry {entry_price}). Setting TP to None.{RESET}")
                 take_profit = None
            elif signal == "SELL" and take_profit >= entry_price:
                 self.logger.warning(f"{NEON_YELLOW}SELL TP calculation non-profitable (TP {take_profit} >= Entry {entry_price}). Setting TP to None.{RESET}")
                 take_profit = None

            # Ensure SL/TP are positive
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calc resulted in non-positive price ({stop_loss}). Setting SL to None.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.warning(f"{NEON_YELLOW}Take profit calc resulted in non-positive price ({take_profit}). Setting TP to None.{RESET}")
                take_profit = None

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_price:.{price_precision}f}, TP={take_profit}, SL={stop_loss}, ATR={atr_val:.{price_precision+1}f}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
             self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
             return entry_price_estimate, None, None


# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency."""
    lg = logger
    try:
        balance_info = None
        # Prioritize CONTRACT type for Bybit V5 derivatives
        account_types_to_try = ['CONTRACT', 'UNIFIED']
        found_structure = False

        for acc_type in account_types_to_try:
             try:
                 lg.debug(f"Fetching balance using params={{'type': '{acc_type}'}} for {currency}...")
                 balance_info = exchange.fetch_balance(params={'type': acc_type})
                 # Check standard and V5 nested structure
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

        # Parse the final balance_info
        available_balance_str = None
        # 1. Standard CCXT
        if currency in balance_info and balance_info[currency].get('free') is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ['{currency}']['free']: {available_balance_str}")
        # 2. Bybit V5 Nested
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
        # 3. Top-level 'free' dict
        elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
             available_balance_str = str(balance_info['free'][currency])
             lg.debug(f"Found balance via top-level 'free' dict: {available_balance_str}")
        # 4. Fallback to 'total'
        if available_balance_str is None:
             total_balance = balance_info.get(currency, {}).get('total')
             if total_balance is not None:
                  lg.warning(f"{NEON_YELLOW}Using 'total' balance ({total_balance}) as fallback for {currency}.{RESET}")
                  available_balance_str = str(total_balance)
             else:
                  lg.error(f"{NEON_RED}Could not determine any balance for {currency}.{RESET}")
                  lg.debug(f"Full balance_info structure: {balance_info}")
                  return None

        # Convert to Decimal
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

    except Exception as e: # Catch errors from the initial try/except blocks if fetch failed entirely
        lg.error(f"{NEON_RED}Critical error during balance fetch for {currency}: {e}{RESET}", exc_info=True)
        return None

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

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal, # Must be calculated and validated before calling
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

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
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

        # --- Calculate Initial Size (Linear/Spot assumed) ---
        # Adapt for inverse if needed, but requires careful handling of contract value.
        if market_info.get('linear', True) or not is_contract:
             calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)
        else: # Inverse Placeholder - Needs Accurate Logic
             lg.warning(f"Inverse contract sizing for {symbol} using placeholder logic - VERIFY!")
             calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
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

        # 1. Adjust for MIN/MAX AMOUNT
        adjusted_size = max(min_amount, min(calculated_size, max_amount))
        if adjusted_size != calculated_size:
             lg.warning(f"{NEON_YELLOW}Size adjusted by amount limits: {calculated_size:.8f} -> {adjusted_size:.8f} {size_unit}{RESET}")

        # 2. Check COST limits (Linear/Spot assumed for cost calc)
        current_cost = adjusted_size * entry_price * contract_size
        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Estimated Cost={current_cost:.4f} {quote_currency}")

        if min_cost > 0 and current_cost < min_cost :
             lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} below min cost {min_cost:.4f}. Attempting to increase size.{RESET}")
             required_size_for_min_cost = min_cost / (entry_price * contract_size) if entry_price > 0 and contract_size > 0 else None
             if required_size_for_min_cost is None:
                 lg.error("Cannot calculate size for min cost.")
                 return None
             lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
             if required_size_for_min_cost > max_amount:
                  lg.error(f"{NEON_RED}Cannot meet min cost {min_cost:.4f} without exceeding max amount {max_amount:.8f}. Aborted.{RESET}")
                  return None
             elif required_size_for_min_cost < min_amount:
                 lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f}, but min amount is {min_amount:.8f}. Aborted.{RESET}")
                 return None
             else:
                 lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f}")
                 adjusted_size = required_size_for_min_cost
                 current_cost = adjusted_size * entry_price * contract_size # Recalculate cost

        elif max_cost > 0 and current_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} exceeds max cost {max_cost:.4f}. Reducing size.{RESET}")
             adjusted_size_for_max_cost = max_cost / (entry_price * contract_size) if entry_price > 0 and contract_size > 0 else None
             if adjusted_size_for_max_cost is None:
                  lg.error("Cannot calculate size for max cost.")
                  return None
             lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f}")
             if adjusted_size_for_max_cost < min_amount:
                  lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is below min amount {min_amount:.8f}. Aborted.{RESET}")
                  return None
             else:
                 adjusted_size = adjusted_size_for_max_cost

        # 3. Apply Amount Precision/Step Size (using ccxt helper)
        try:
            # Use TRUNCATE (equivalent to ROUND_DOWN for positive numbers) to be conservative
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision ({fmt_err}). Using manual rounding.{RESET}")
            # Fallback manual rounding
            analyzer = TradingAnalyzer(df=pd.DataFrame(), logger=lg, config=CONFIG, market_info=market_info) # Temp instance for helpers
            amount_step = analyzer.get_min_tick_size() # Assuming amount step is same as price tick for simplicity - **CHECK THIS**
            # If amount precision is defined differently (e.g., integer decimal places), use that instead.
            # Let's assume step size is the most common way for amount precision.
            if amount_step > 0:
                final_size = (adjusted_size // amount_step) * amount_step # Round down to step size
                lg.info(f"Applied manual amount step size ({amount_step}): {adjusted_size:.8f} -> {final_size} {size_unit}")
            else:
                lg.warning("Could not determine amount step size, using unrounded adjusted size.")
                final_size = adjusted_size


        # --- Final Validation ---
        if final_size <= 0:
             lg.error(f"{NEON_RED}Position size became zero/negative ({final_size}) after adjustments. Aborted.{RESET}")
             return None
        if final_size < min_amount:
             lg.error(f"{NEON_RED}Final size {final_size} is below minimum amount {min_amount}. Aborted.{RESET}")
             return None
        # Final cost check
        final_cost = final_size * entry_price * contract_size
        if min_cost > 0 and final_cost < min_cost:
            lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} below minimum cost {min_cost:.4f}. Aborted.{RESET}")
            return None


        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions."""
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        fetch_all = False

        # Attempt fetch single symbol (preferred for Bybit V5)
        try: positions = exchange.fetch_positions([symbol])
        except ccxt.ArgumentsRequired: fetch_all = True
        except ccxt.ExchangeError as e:
             no_pos_codes_v5 = [110025] # Position idx not match / Position is closed
             if "symbol not found" in str(e).lower() or (hasattr(e, 'code') and e.code in no_pos_codes_v5):
                  lg.info(f"No position found for {symbol} (Exchange confirmed: {e}).")
                  return None
             lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=False); return None # Treat other errors as failure
        except Exception as e:
             lg.error(f"Error fetching single position for {symbol}: {e}", exc_info=True); return None

        if fetch_all:
            try:
                 all_positions = exchange.fetch_positions()
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total, found {len(positions)} matching {symbol}.")
            except Exception as e: lg.error(f"Error fetching all positions for {symbol}: {e}", exc_info=True); return None

        # --- Process fetched positions ---
        active_position = None
        size_threshold = Decimal('1e-9') # Threshold to consider position non-zero

        for pos in positions:
            pos_size_str = None
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size']) # Bybit V5

            if pos_size_str is None: continue

            try:
                position_size = Decimal(pos_size_str)
                if abs(position_size) > size_threshold:
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break
            except Exception: continue

        # --- Post-Process the found active position ---
        if active_position:
            size_decimal = Decimal(str(active_position.get('contracts', active_position.get('info',{}).get('size', '0'))))
            side = active_position.get('side')
            # Infer side if missing
            if side not in ['long', 'short']:
                if size_decimal > size_threshold: side = 'long'
                elif size_decimal < -size_threshold: side = 'short'
                else: lg.warning(f"Position size {size_decimal} near zero, cannot determine side."); return None
                active_position['side'] = side # Store inferred side

            # Enhance with SL/TP/TSL info from 'info' dict if not in standard fields
            info_dict = active_position.get('info', {})
            if active_position.get('stopLossPrice') is None: active_position['stopLossPrice'] = info_dict.get('stopLoss')
            if active_position.get('takeProfitPrice') is None: active_position['takeProfitPrice'] = info_dict.get('takeProfit')
            active_position['trailingStopLoss'] = info_dict.get('trailingStop')
            active_position['tslActivationPrice'] = info_dict.get('activePrice')

            # Helper to format price/value for logging
            def format_log_val(val, precision=6, is_size=False):
                if val is None or str(val).strip() == '' or str(val) == '0': return 'N/A' # Treat '0' string as N/A for SL/TP/TSL
                try:
                     d_val = Decimal(str(val))
                     # Size formatting
                     if is_size:
                         # Get amount precision dynamically if possible
                         try: amount_prec = abs(get_market_info(exchange, symbol, lg)['precision']['amount'].normalize().as_tuple().exponent)
                         except: amount_prec = 8 # Default precision for size
                         return f"{abs(d_val):.{amount_prec}f}" # Show absolute size
                     # Price/Value formatting
                     else:
                         # Use market price precision
                         try: price_prec = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, get_market_info(exchange, symbol, lg)).get_price_precision()
                         except: price_prec = precision # Fallback precision
                         return f"{d_val:.{price_prec}f}"
                except Exception: return str(val)

            entry_price = format_log_val(active_position.get('entryPrice', info_dict.get('avgPrice')))
            contracts = format_log_val(active_position.get('contracts', info_dict.get('size')), is_size=True)
            liq_price = format_log_val(active_position.get('liquidationPrice'))
            leverage_str = active_position.get('leverage', info_dict.get('leverage'))
            leverage = f"{Decimal(leverage_str):.1f}x" if leverage_str is not None else 'N/A'
            pnl_str = active_position.get('unrealizedPnl')
            pnl = format_log_val(pnl_str, 4) # More precision for PNL
            sl_price = format_log_val(active_position.get('stopLossPrice'))
            tp_price = format_log_val(active_position.get('takeProfitPrice'))
            tsl_dist = format_log_val(active_position.get('trailingStopLoss')) # Distance (value)
            tsl_act = format_log_val(active_position.get('tslActivationPrice')) # Activation price

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
        # Bybit V5 requires buy/sell leverage parameters
        params = {}
        if 'bybit' in exchange.id.lower():
             params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
             lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")
        # Assume success if no exception is raised, verification is complex.
        lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True # Success
        elif bybit_code in [110028, 110009, 110055] or "margin mode" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) compatibility with leverage setting.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed risk limit tier. Check Bybit Risk Limits.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for {symbol}. Check allowed range.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal,
    market_info: Dict,
    logger: Optional[logging.Logger] = None,
    reduce_only: bool = False # Added flag for closing orders
) -> Optional[Dict]:
    """
    Places a market order using CCXT. Returns the order dictionary on success, None on failure.
    Uses reduce_only flag for closing positions.
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
        'positionIdx': 0,  # Assume One-Way Mode
        'reduceOnly': reduce_only,
    }
    if reduce_only: # Closing order specific parameters
        params['timeInForce'] = 'IOC' # ImmediateOrCancel recommended for closing market orders
        lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")
    else: # Opening order
        lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")

    lg.info(f"  Size: {amount_float:.8f} {size_unit}")
    lg.info(f"  Params: {params}")

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

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to {action} {side} order ({symbol}): {e}{RESET}")
        # ... (balance logging and hints remain similar) ...
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters to {action} {side} order ({symbol}): {e}{RESET}")
        # ... (parameter logging and hints remain similar) ...
        bybit_code = getattr(e, 'code', None)
        if reduce_only and bybit_code == 110014: # Reduce-only error
             lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only order failed. Position might already be closed, size incorrect, or API issue?{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing {action} order ({symbol}): {e}{RESET}")
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action} order ({symbol}): {e} (Code: {bybit_code}){RESET}")
        # ... (hints for common codes remain similar) ...
        if reduce_only and bybit_code == 110025: # Position not found/zero on close attempt
            lg.warning(f"{NEON_YELLOW} >> Hint (110025): Position might have been closed already when trying to place reduce-only order.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing {action} order ({symbol}): {e}{RESET}", exc_info=True)

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

    # Validate inputs are positive Decimals if provided
    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0)

    if not has_sl and not has_tp and not has_tsl:
         lg.info(f"No valid protection parameters provided for {symbol}. No protection set/updated.")
         return True # Considered success as no action needed

    # --- Prepare API Parameters ---
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    position_idx = 0 # Default for One-Way
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
    except Exception: lg.warning(f"Could not parse positionIdx, using {position_idx}.")

    params = {
        'category': category,
        'symbol': market_info['id'],
        'tpslMode': 'Full', # Apply to whole position
        'slTriggerBy': 'LastPrice', # Can change to MarkPrice if preferred/needed
        'tpTriggerBy': 'LastPrice',
        'slOrderType': 'Market',
        'tpOrderType': 'Market',
        'positionIdx': position_idx
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} PosIdx: {position_idx}):"]

    # --- Format and Add Parameters ---
    try:
        # Temp analyzer for precision/tick helpers
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info)
        price_prec = analyzer.get_price_precision()
        min_tick = analyzer.get_min_tick_size()

        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try: return exchange.price_to_precision(symbol, float(price_decimal))
            except Exception as e: lg.warning(f"Failed to format price {price_decimal}: {e}"); return None

        # Handle TSL first, as it might override fixed SL
        if has_tsl:
            # Format distance (needs tick size precision, treated like a price difference)
            try:
                 # Use decimal_to_precision for distance formatting relative to tick size
                 dist_prec = abs(min_tick.normalize().as_tuple().exponent)
                 formatted_tsl_distance = exchange.decimal_to_precision(
                     trailing_stop_distance, exchange.ROUND, precision=dist_prec, padding_mode=exchange.NO_PADDING
                 )
                 if Decimal(formatted_tsl_distance) < min_tick:
                      formatted_tsl_distance = str(min_tick) # Ensure minimum distance is at least one tick
            except Exception as e:
                 lg.warning(f"Failed to format TSL distance {trailing_stop_distance}: {e}"); formatted_tsl_distance = None

            formatted_activation_price = format_price(tsl_activation_price)

            if formatted_tsl_distance and formatted_activation_price and Decimal(formatted_tsl_distance) > 0:
                params['trailingStop'] = formatted_tsl_distance
                params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                # If TSL is set, Bybit ignores 'stopLoss'. No need to explicitly remove 'stopLoss' param.
                has_sl = False # Mark fixed SL as effectively overridden/not set
            else:
                lg.error(f"Failed to format valid TSL parameters for {symbol}. Cannot set TSL.")
                has_tsl = False # Mark TSL as failed

        # Fixed Stop Loss (only if TSL wasn't successfully prepared)
        if has_sl:
            formatted_sl = format_price(stop_loss_price)
            if formatted_sl:
                params['stopLoss'] = formatted_sl
                log_parts.append(f"  Fixed SL: {formatted_sl}")
            else:
                has_sl = False # Failed

        # Fixed Take Profit
        if has_tp:
            formatted_tp = format_price(take_profit_price)
            if formatted_tp:
                params['takeProfit'] = formatted_tp
                log_parts.append(f"  Fixed TP: {formatted_tp}")
            else:
                has_tp = False # Failed

    except Exception as fmt_err:
         lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    # Check if any protection is actually being sent after formatting and TSL override logic
    if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
        lg.warning(f"No valid protection parameters could be formatted or remain after adjustments for {symbol}. No API call made.")
        # Return True if the *intent* was to set nothing (e.g., only invalid inputs given)
        # Return False if formatting failed for valid inputs? Let's return False to indicate failure.
        return False

    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={params})")

    # --- Call Bybit V5 API Endpoint ---
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
            # ... (Add specific error code hints as before) ...
            if ret_code == 110013 and "parameter error" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint (110013): Check SL/TP prices vs entry, TSL dist/act prices, tick size compliance.{RESET}")
            elif ret_code == 110036: # TSL Active price invalid
                 lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price {params.get('activePrice')} likely invalid (already passed, wrong side, too close?).{RESET}")
            elif ret_code == 110086: # SL=TP
                 lg.error(f"{NEON_YELLOW} >> Hint (110086): SL price cannot equal TP price.{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: TSL distance {params.get('trailingStop')} invalid (too small/large/violates tick?).{RESET}")
            # Add more hints as needed...
            return False

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during protection API call for {symbol}: {e}{RESET}", exc_info=True)
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Confirmed position dict
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional TP to set alongside TSL
) -> bool:
    """
    Calculates TSL parameters and calls the internal helper to set TSL (and optionally TP).
    """
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss disabled in config for {symbol}. Skipping TSL setup.")
        return False # TSL wasn't set

    try:
        callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
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
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance for helpers
        price_precision = analyzer.get_price_precision()
        price_rounding = Decimal('1e-' + str(price_precision))
        min_tick_size = analyzer.get_min_tick_size()

        # 1. Calculate Activation Price
        activation_price = None
        # Ensure activation percentage calculation uses Decimal
        activation_offset = entry_price * activation_percentage
        if side == 'long':
            raw_activation = entry_price + activation_offset
            activation_price = raw_activation.quantize(price_rounding, rounding=ROUND_UP)
            # Ensure activation strictly > entry for positive activation percentage
            if activation_percentage > 0 and activation_price <= entry_price:
                activation_price = (entry_price + min_tick_size).quantize(price_rounding, rounding=ROUND_UP)
            # For immediate activation (0%), set slightly above entry
            elif activation_percentage == 0:
                 activation_price = (entry_price + min_tick_size).quantize(price_rounding, rounding=ROUND_UP)
        else: # short
            raw_activation = entry_price - activation_offset
            activation_price = raw_activation.quantize(price_rounding, rounding=ROUND_DOWN)
            # Ensure activation strictly < entry for positive activation percentage
            if activation_percentage > 0 and activation_price >= entry_price:
                 activation_price = (entry_price - min_tick_size).quantize(price_rounding, rounding=ROUND_DOWN)
            # For immediate activation (0%), set slightly below entry
            elif activation_percentage == 0:
                 activation_price = (entry_price - min_tick_size).quantize(price_rounding, rounding=ROUND_DOWN)

        if activation_price is None or activation_price <= 0:
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) invalid for {symbol}.{RESET}")
             return False

        # 2. Calculate Trailing Stop Distance (based on activation price)
        trailing_distance_raw = activation_price * callback_rate
        # Round distance UP to nearest tick size increment
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        if trailing_distance < min_tick_size: trailing_distance = min_tick_size # Enforce minimum tick size
        if trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance zero/negative ({trailing_distance}) for {symbol}.{RESET}")
             return False

        lg.info(f"Calculated TSL Params for {symbol} ({side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price: {activation_price:.{price_precision}f}")
        lg.info(f"  => Trailing Distance: {trailing_distance:.{price_precision}f}")
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0:
             lg.info(f"  Take Profit Price: {take_profit_price:.{price_precision}f} (Will be set alongside TSL)")

        # 3. Call helper to set TSL (and TP)
        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
            stop_loss_price=None, # Explicitly None
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed market info for {symbol}. Skipping cycle.{RESET}")
        return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}'. Cannot map to CCXT timeframe.")
         return

    kline_limit = 500
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50:
        lg.error(f"{NEON_RED}Failed sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping cycle.{RESET}")
        return

    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
         lg.warning(f"{NEON_YELLOW}Failed current price for {symbol}. Using last close fallback.{RESET}")
         try:
             last_close_val = klines_df['close'].iloc[-1]
             if pd.notna(last_close_val) and last_close_val > 0:
                  current_price = Decimal(str(last_close_val))
                  lg.info(f"Using last close price: {current_price}")
             else: lg.error(f"{NEON_RED}Last close price invalid. Cannot proceed.{RESET}"); return
         except Exception as e: lg.error(f"{NEON_RED}Error getting last close: {e}. Cannot proceed.{RESET}"); return

    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators",{}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
         orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)

    analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    if not analyzer.indicator_values:
         lg.error(f"{NEON_RED}Indicator calculation failed for {symbol}. Skipping signal.{RESET}")
         return

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal) # Based on current price estimate
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr = analyzer.indicator_values.get("ATR") # Decimal

    # --- Log Analysis Summary ---
    lg.info(f"ATR: {current_atr:.{price_precision+1}f}" if isinstance(current_atr, Decimal) else 'N/A')
    lg.info(f"Calc Initial SL (sizing): {sl_calc if sl_calc else 'N/A'}")
    lg.info(f"Calc Initial TP (potential): {tp_calc if tp_calc else 'N/A'}")
    tsl_enabled = config.get('enable_trailing_stop')
    be_enabled = config.get('enable_break_even')
    lg.info(f"TSL: {'Enabled' if tsl_enabled else 'Disabled'} | BE: {'Enabled' if be_enabled else 'Disabled'}")

    # --- Trading Logic ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    open_position = get_open_position(exchange, symbol, lg) # Returns dict or None

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***")

            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative.{RESET}")
                return
            if sl_calc is None:
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed (ATR invalid?).{RESET}")
                 return

            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage.{RESET}")
                         return
            else: lg.info(f"Leverage setting skipped (Spot).")

            position_size = calculate_position_size(balance, config["risk_per_trade"], sl_calc, current_price, market_info, exchange, lg)
            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size ({position_size}).{RESET}")
                return

            lg.info(f"==> Placing {signal} market order | Size: {position_size} <==")
            trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)

            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                lg.info(f"Attempting confirmation for {symbol} after order {order_id}...")
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        entry_price_actual_str = confirmed_position.get('entryPrice') or confirmed_position.get('info', {}).get('avgPrice')
                        entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else current_price # Fallback
                        lg.info(f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_price_actual:.{price_precision}f}{RESET}")

                        # Recalculate protection based on ACTUAL entry price
                        _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                        protection_set_success = False
                        if config.get("enable_trailing_stop", False):
                             lg.info(f"Setting Trailing Stop Loss (TP target: {tp_final})...")
                             protection_set_success = set_trailing_stop_loss(
                                 exchange, symbol, market_info, confirmed_position, config, lg,
                                 take_profit_price=tp_final # Pass recalculated TP
                             )
                        else: # Set Fixed SL/TP
                             lg.info(f"Setting Fixed SL ({sl_final}) and TP ({tp_final})...")
                             if sl_final or tp_final:
                                 protection_set_success = _set_position_protection(
                                     exchange, symbol, market_info, confirmed_position, lg,
                                     stop_loss_price=sl_final, take_profit_price=tp_final
                                 )
                             else: lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed based on actual entry. No fixed protection set.{RESET}")

                        if protection_set_success:
                             lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                        else:
                             lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ===")
                             lg.warning(f"{NEON_YELLOW}MANUAL MONITORING REQUIRED!{RESET}")

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
        exit_signal_triggered = (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY")

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position. Closing... ***{RESET}")
            try:
                close_side = 'sell' if pos_side == 'long' else 'buy'
                size_to_close_str = open_position.get('contracts') or open_position.get('info',{}).get('size')
                if size_to_close_str is None: raise ValueError("Cannot determine position size to close.")
                size_to_close = abs(Decimal(str(size_to_close_str)))
                if size_to_close <= 0: raise ValueError(f"Position size {size_to_close_str} invalid.")

                lg.info(f"==> Placing {close_side.upper()} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                close_order = place_trade(exchange, symbol, "SELL" if pos_side == 'long' else "BUY", size_to_close, market_info, lg, reduce_only=True)

                if close_order:
                    lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                    # Assume close is successful due to reduceOnly market order. Further checks could poll position.
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check required!{RESET}")

            except Exception as close_err:
                 lg.error(f"{NEON_RED}Error closing position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")

        else: # Hold signal or signal matches position
            lg.info(f"Signal ({signal}) allows holding existing {pos_side} position.")

            # --- Manage Existing Position (BE, TSL checks) ---
            is_tsl_active = False
            try: is_tsl_active = Decimal(str(open_position.get('trailingStopLoss', '0'))) > 0
            except: pass # Ignore parsing error for TSL check

            # --- Break-Even Check (only if BE enabled AND TSL is NOT active) ---
            if config.get("enable_break_even", False) and not is_tsl_active:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    entry_price = Decimal(str(open_position.get('entryPrice') or open_position.get('info', {}).get('avgPrice')))
                    if not isinstance(current_atr, Decimal) or current_atr <= 0:
                        lg.warning("BE Check skipped: Invalid ATR.")
                    else:
                        profit_target_atr = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                        profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('0')

                        lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision}f}, Profit ATRs={profit_in_atr:.2f}, Target ATRs={profit_target_atr}")

                        if profit_in_atr >= profit_target_atr:
                            # Calculate target BE SL
                            tick_offset = min_tick_size * offset_ticks
                            be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP) if pos_side == 'long' \
                                       else (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                            # Get current SL as Decimal
                            current_sl_price = None
                            current_sl_str = open_position.get('stopLossPrice') or open_position.get('info', {}).get('stopLoss')
                            if current_sl_str and str(current_sl_str) != '0':
                                try: current_sl_price = Decimal(str(current_sl_str))
                                except Exception: pass

                            # Determine if update needed
                            update_be = False
                            if be_stop_price > 0:
                                if current_sl_price is None: update_be = True; lg.info("BE triggered: No current SL found.")
                                elif pos_side == 'long' and be_stop_price > current_sl_price: update_be = True; lg.info(f"BE triggered: Target {be_stop_price} > Current SL {current_sl_price}.")
                                elif pos_side == 'short' and be_stop_price < current_sl_price: update_be = True; lg.info(f"BE triggered: Target {be_stop_price} < Current SL {current_sl_price}.")
                                else: lg.debug(f"BE Triggered, but current SL ({current_sl_price}) already better than target ({be_stop_price}).")

                            if update_be:
                                lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                                # Preserve existing TP if possible
                                current_tp_price = None
                                current_tp_str = open_position.get('takeProfitPrice') or open_position.get('info', {}).get('takeProfit')
                                if current_tp_str and str(current_tp_str) != '0':
                                     try: current_tp_price = Decimal(str(current_tp_str))
                                     except Exception: pass

                                success = _set_position_protection(
                                    exchange, symbol, market_info, open_position, lg,
                                    stop_loss_price=be_stop_price,
                                    take_profit_price=current_tp_price # Preserve TP
                                )
                                if success: lg.info(f"{NEON_GREEN}Break-Even SL set successfully.{RESET}")
                                else: lg.error(f"{NEON_RED}Failed to set Break-Even SL.{RESET}")
                        else:
                            lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {profit_target_atr} ATRs).")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check ({symbol}): {be_err}{RESET}", exc_info=True)
            elif is_tsl_active:
                 lg.info(f"Break-even check skipped: Trailing Stop Loss is active.")
            else: # BE disabled
                 lg.debug(f"Break-even check skipped: Disabled in config.")

            # --- Placeholder for other management logic ---
            # E.g., Check if TSL failed previously and needs retry?


    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY

    setup_logger("init")
    init_logger = logging.getLogger("init")

    init_logger.info(f"--- Starting LiveXY Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
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
         init_logger.warning(f"Settings: Risk={CONFIG.get('risk_per_trade', 0)*100:.2f}%, Lev={CONFIG.get('leverage', 0)}x, TSL={'ON' if CONFIG.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if CONFIG.get('enable_break_even') else 'OFF'}")
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
            symbol_input = symbol_input_raw.replace('-', '/') # Normalize

            init_logger.info(f"Validating symbol '{symbol_input}'...")
            market_info = get_market_info(exchange, symbol_input, init_logger)

            if market_info:
                target_symbol = market_info['symbol']
                market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                break
            else: # Try variations like BASE/QUOTE:QUOTE
                variations_to_try = []
                if '/' in symbol_input:
                    base, quote = symbol_input.split('/', 1)
                    variations_to_try.append(f"{base}/{quote}:{quote}") # Common linear perp format

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
        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            selected_interval = interval_input
            CONFIG["interval"] = selected_interval # Update config in memory
            ccxt_tf = CCXT_INTERVAL_MAP[selected_interval]
            init_logger.info(f"Using interval: {selected_interval} (CCXT: {ccxt_tf})")
            break
        else: init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Choose from {VALID_INTERVALS}.{RESET}")

    symbol_logger = setup_logger(target_symbol)
    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    symbol_logger.info(f"Config: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if CONFIG['enable_trailing_stop'] else 'OFF'}, BE={'ON' if CONFIG['enable_break_even'] else 'OFF'}, Trading={'ENABLED' if CONFIG['enable_trading'] else 'DISABLED'}")

    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Start: {datetime.now(TIMEZONE).strftime('%H:%M:%S')}")
            try:
                # --- Optional Config Reload ---
                # CONFIG = load_config(CONFIG_FILE)
                # QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
                # --- End Optional Reload ---
                analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger)
            except ccxt.RateLimitExceeded as e: symbol_logger.warning(f"{NEON_YELLOW}Rate limit: {e}. Waiting 60s...{RESET}"); time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.RequestException) as e: symbol_logger.error(f"{NEON_RED}Network error: {e}. Waiting {RETRY_DELAY_SECONDS*3}s...{RESET}"); time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"{NEON_RED}CRITICAL Auth Error: {e}. Stopping.{RESET}"); break
            except ccxt.ExchangeNotAvailable as e: symbol_logger.error(f"{NEON_RED}Exchange unavailable: {e}. Waiting 60s...{RESET}"); time.sleep(60)
            except ccxt.OnMaintenance as e: symbol_logger.error(f"{NEON_RED}Exchange Maintenance: {e}. Waiting 5m...{RESET}"); time.sleep(300)
            except Exception as loop_error: symbol_logger.error(f"{NEON_RED}Uncaught loop error: {loop_error}{RESET}", exc_info=True); time.sleep(15)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle finished in {elapsed_time:.2f}s. Sleeping {sleep_time:.2f}s...")
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except Exception as critical_error: init_logger.critical(f"{NEON_RED}Critical unhandled error: {critical_error}{RESET}", exc_info=True)
    finally:
        shutdown_msg = f"--- LiveXY Bot for {target_symbol or 'N/A'} Stopping ---"
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger): symbol_logger.info(shutdown_msg)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); exchange.close(); init_logger.info("Connection closed.")
            except Exception as close_err: init_logger.error(f"Error closing connection: {close_err}")
        logging.shutdown()
        print(f"{NEON_YELLOW}Bot stopped.{RESET}")


if __name__ == "__main__":
    # Write the enhanced script content to 'livexy.py'
    output_filename = "livexy.py"
    try:
        with open(__file__, 'r', encoding='utf-8') as current_file:
             script_content = current_file.read()
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            header = f"# {output_filename}\n# Enhanced version focusing on stop-loss/take-profit mechanisms, including break-even and trailing stops.\n\n"
            # Clean up potential duplicate headers/comments from prompt
            import re
            script_content = re.sub(r'^(# livex[xy]\.py.*\n)+', '', script_content)
            output_file.write(header + script_content)
        print(f"Enhanced script content written to {output_filename}")

        # Run the main logic
        main()
    except Exception as e:
        print(f"{NEON_RED}Error writing script to {output_filename} or running main: {e}{RESET}")
```
