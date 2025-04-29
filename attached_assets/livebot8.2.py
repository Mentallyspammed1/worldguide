```python
# livexy.py
# Enhanced version focusing on stop-loss/take-profit mechanisms, including break-even logic.

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, Union

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
getcontext().prec = 36  # Increased precision for financial calculations, especially intermediate steps
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
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable (Generic, ccxt handles specific ones)
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
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status
# QUOTE_CURRENCY dynamically loaded from config

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    _secrets = [] # Class variable to hold secrets

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add secrets only if they exist
        if API_KEY: SensitiveFormatter._secrets.append(API_KEY)
        if API_SECRET: SensitiveFormatter._secrets.append(API_SECRET)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for secret in SensitiveFormatter._secrets:
            # Redact partially, ensuring secret is a string
            secret_str = str(secret)
            msg = msg.replace(secret_str, f"***{secret_str[:3]}...{secret_str[-3:]}***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present."""
    default_config = {
        "interval": "5", # Default to 5 minute interval (string format for our logic)
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
        "max_concurrent_positions": 1, # Limit open positions for this symbol (common strategy) - Currently informational, not enforced
        "quote_currency": "USDT", # Currency for balance check and sizing
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to enabling TSL (exchange TSL)
        # Trail distance as a percentage of the activation/high-water-mark price (e.g., 0.5%)
        # Bybit API expects absolute distance, this percentage is used for calculation.
        "trailing_stop_callback_rate": 0.005, # Example: 0.5% trail distance relative to entry/activation
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry.
        "trailing_stop_activation_percentage": 0.003, # Example: Activate when 0.3% in profit
        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        # Move SL when profit (in price points) reaches X * Current ATR
        "break_even_trigger_atr_multiple": 1.0, # Example: Trigger BE when profit = 1x ATR
        # Place BE SL this many minimum price increments (ticks) beyond entry price
        # E.g., 2 ticks to cover potential commission/slippage on exit
        "break_even_offset_ticks": 2,
        # --- End Protection Config ---
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
            # Return default config anyway if file creation fails
            return default_config

    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added during the update
            if updated_config != config_from_file:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                    print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
                except IOError as e:
                    print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
            return updated_config
    except FileNotFoundError:
        print(f"{NEON_RED}Config file not found at {filepath}. Using default config.{RESET}")
        # Attempt to create default if not found
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after not found: {e_create}{RESET}")
        return default_config
    except json.JSONDecodeError as e:
        print(f"{NEON_RED}Error decoding JSON in config file {filepath}: {e}. Using default config.{RESET}")
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath} after JSON error.{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after JSON error: {e_create}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Optional: Handle type mismatches if needed (could log warning)
        elif type(default_value) != type(updated_config.get(key)):
             # Be careful with type mismatches, especially numeric types (int vs float)
             # Only warn if it's not a simple numeric difference (e.g., 2 vs 2.0)
             if not (isinstance(default_value, (int, float)) and isinstance(updated_config.get(key), (int, float))):
                 print(f"{NEON_YELLOW}Warning: Type mismatch for key '{key}' in config. Default: {type(default_value)}, Loaded: {type(updated_config.get(key))}. Using loaded value.{RESET}")
             # Decide whether to use loaded or default value in case of significant type mismatch
             # For now, keeping the loaded value seems safer unless strict typing is enforced.
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config
console_log_level = logging.INFO # Default console level, can be changed in main()

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    # Clean symbol for filename (replace / and : which are invalid in filenames)
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"livexy_bot_{safe_symbol}" # Use safe symbol in logger name
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        # Ensure existing handlers have the correct level (e.g., if changed dynamically)
        for handler in logger.handlers:
             if isinstance(handler, logging.StreamHandler):
                  handler.setLevel(console_log_level) # Update console level
        return logger

    # Set base logging level to DEBUG to capture everything
    logger.setLevel(logging.DEBUG)

    # File Handler (writes DEBUG and above, includes line numbers)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Add line number to file logs for easier debugging
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")


    # Stream Handler (console, writes INFO and above by default for cleaner output)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Add timestamp format to console
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level based on global variable
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs)
    logger.propagate = False

    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with error handling."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'linear', # Assume linear contracts (USDT margined) - adjust if needed
                'adjustForTimeDifference': True, # Auto-sync time with server
                # Connection timeouts (milliseconds)
                'fetchTickerTimeout': 10000, # 10 seconds
                'fetchBalanceTimeout': 15000, # 15 seconds
                'createOrderTimeout': 20000, # Timeout for placing orders
                'fetchOrderTimeout': 15000,  # Timeout for fetching orders
                'fetchPositionsTimeout': 15000, # Timeout for fetching positions
                # Add any exchange-specific options if needed
                # 'recvWindow': 10000, # Example for Binance if needed
                'brokerId': 'livexyBotV2', # Example: Add a broker ID for Bybit if desired
            }
        }

        # Select Bybit class
        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        # Test connection by fetching markets (essential for market info)
        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test API credentials and permissions by fetching balance
        # Specify account type for Bybit V5 (CONTRACT for linear/USDT, UNIFIED if using that)
        account_type_to_test = 'CONTRACT' # Or 'UNIFIED' based on your account
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            # Use our enhanced balance function
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, logger)
            if balance_decimal is not None:
                 logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 # fetch_balance logs errors, this is a fallback warning
                 logger.warning(f"{NEON_YELLOW}Initial balance fetch failed or returned None. Check API permissions/account type if trading fails.{RESET}")
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None # Critical failure, cannot proceed
        except ccxt.ExchangeError as balance_err:
            # Handle potential errors if account type is wrong etc.
            logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")
        except Exception as balance_err:
            logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")

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
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks."""
    lg = logger
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        lg.debug(f"Ticker data for {symbol}: {ticker}")

        price = None
        last_price = ticker.get('last')
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')

        # Helper to parse and validate price string to positive Decimal
        def parse_price(price_str: Optional[Union[str, float]], name: str) -> Optional[Decimal]:
            if price_str is None: return None
            try:
                d_price = Decimal(str(price_str))
                if d_price > 0:
                    return d_price
                else:
                    lg.warning(f"'{name}' price ({d_price}) is not positive for {symbol}.")
                    return None
            except Exception as e:
                lg.warning(f"Could not parse '{name}' price ({price_str}) for {symbol}: {e}")
                return None

        # 1. Try 'last' price first
        price = parse_price(last_price, 'last')
        if price:
            lg.debug(f"Using 'last' price for {symbol}: {price}")

        # 2. If 'last' is invalid, try bid/ask midpoint
        if price is None:
            bid_decimal = parse_price(bid_price, 'bid')
            ask_decimal = parse_price(ask_price, 'ask')
            if bid_decimal and ask_decimal:
                if bid_decimal <= ask_decimal:
                    price = (bid_decimal + ask_decimal) / 2
                    lg.debug(f"Using bid/ask midpoint for {symbol}: {price} (Bid: {bid_decimal}, Ask: {ask_decimal})")
                else:
                    lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' as fallback.")
                    price = ask_decimal # Use ask as a safer fallback in this case
            elif ask_decimal:
                 # If only ask is valid, use it as fallback
                 price = ask_decimal
                 lg.warning(f"Bid invalid, using 'ask' price as fallback for {symbol}: {price}")
            elif bid_decimal:
                 # If only bid is valid, use it as fallback
                 price = bid_decimal
                 lg.warning(f"Ask invalid, using 'bid' price as fallback for {symbol}: {price}")

        # 3. Final check - we should have a price from last, midpoint, ask, or bid fallback by now
        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to fetch a valid positive current price for {symbol} from ticker.{RESET}")
            lg.debug(f"Final ticker values: last={last_price}, bid={bid_price}, ask={ask_price}")
            return None

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
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
                if ohlcv is not None and len(ohlcv) > 0: # Basic check if data was returned and not empty
                    break # Success
                else:
                    lg.warning(f"fetch_ohlcv returned {type(ohlcv)} (length {len(ohlcv) if ohlcv is not None else 'N/A'}) for {symbol} (Attempt {attempt+1}). Retrying...")
                    # Optional: Add a small delay even on None/empty return if it might be transient
                    time.sleep(1)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                    raise e # Re-raise the last error
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 5 # Default wait time
                try:
                     # Try to parse recommended wait time from Bybit/other exchange messages
                     if 'try again in' in str(e).lower():
                         wait_time = int(str(e).lower().split('try again in')[1].split('ms')[0].strip()) / 1000
                         wait_time = max(1, int(wait_time + 1)) # Add a buffer and ensure minimum 1s
                     elif 'rate limit' in str(e).lower(): # Generic rate limit message
                         match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                         if match:
                             num = int(match.group(1))
                             unit = match.group(2)
                             wait_time = num / 1000 if unit == 'ms' else num
                             wait_time = max(1, int(wait_time + 1))
                except Exception:
                     pass # Use default if parsing fails
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                # Check for "Invalid timeframe" error
                if "invalid timeframe" in str(e).lower() or "timeframe not supported" in str(e).lower():
                    lg.error(f"{NEON_RED}Exchange error: Timeframe '{timeframe}' is invalid or not supported by {exchange.id} for {symbol}. {e}{RESET}")
                    return pd.DataFrame() # Return empty DataFrame, not retryable
                # Check for bad symbol error
                elif "symbol" in str(e).lower() and ("not found" in str(e).lower() or "invalid" in str(e).lower()):
                     lg.error(f"{NEON_RED}Exchange error: Symbol '{symbol}' not found or invalid on {exchange.id}. {e}{RESET}")
                     return pd.DataFrame() # Not retryable

                lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                # Depending on the error, might not be retryable
                raise e # Re-raise other non-network errors immediately

        if not ohlcv: # Check if list is empty or still None after retries
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} immediately after creation.{RESET}")
            return df

        # Convert timestamp to datetime and set as index
        # Use errors='coerce' to handle potential timestamp issues gracefully
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame empty after timestamp conversion for {symbol} {timeframe}.{RESET}")
            return df
        df.set_index('timestamp', inplace=True)

        # Ensure numeric types, coerce errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with any NaN in critical price/volume columns
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        # Drop rows with non-positive close price (invalid data)
        df = df[df['close'] > 0]
        # Optional: Drop rows with zero volume if it indicates bad data (can be market specific)
        # df = df[df['volume'] > 0]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price/volume data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return pd.DataFrame()

        # Optional: Sort index just in case data isn't perfectly ordered (though fetch_ohlcv usually is)
        df.sort_index(inplace=True)

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e: # Catch error if retries fail
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

            # --- Validation ---
            if not orderbook:
                lg.warning(f"fetch_order_book returned None or empty data for {symbol} (Attempt {attempts+1}).")
                # Continue to retry logic
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"{NEON_YELLOW}Invalid orderbook structure (bids/asks not lists) for {symbol}. Attempt {attempts + 1}. Response: {orderbook}{RESET}")
                 # Treat as potentially retryable issue
            elif not orderbook['bids'] and not orderbook['asks']:
                 # Exchange might return empty lists if orderbook is thin or during low liquidity
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 return orderbook # Return the empty book, signal generation needs to handle this
            else:
                 # Basic validation of first bid/ask format [price, amount]
                 valid_format = True
                 if orderbook['bids'] and (not isinstance(orderbook['bids'][0], list) or len(orderbook['bids'][0]) != 2):
                     valid_format = False
                 if orderbook['asks'] and (not isinstance(orderbook['asks'][0], list) or len(orderbook['asks'][0]) != 2):
                     valid_format = False

                 if not valid_format:
                     lg.warning(f"{NEON_YELLOW}Invalid orderbook entry format (expected [price, amount]) for {symbol}. Attempt {attempts + 1}. Response: {orderbook}{RESET}")
                     # Treat as potentially retryable issue
                 else:
                     # Looks valid
                     lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                     return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5 # Default wait time
            try:
                 if 'try again in' in str(e).lower():
                     wait_time = int(str(e).lower().split('try again in')[1].split('ms')[0].strip()) / 1000
                     wait_time = max(1, int(wait_time + 1))
                 elif 'rate limit' in str(e).lower():
                     match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                     if match:
                         num = int(match.group(1))
                         unit = match.group(2)
                         wait_time = num / 1000 if unit == 'ms' else num
                         wait_time = max(1, int(wait_time + 1))
            except Exception: pass
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time}s... (Attempt {attempts+1})")
            time.sleep(wait_time) # Use delay from error msg if possible
            # Increment attempt counter here so it doesn't bypass retry limit due to sleep
            attempts += 1
            continue # Skip the standard delay at the end
        except ccxt.ExchangeError as e:
            # Check for bad symbol error
            if "symbol" in str(e).lower() and ("not found" in str(e).lower() or "invalid" in str(e).lower()):
                 lg.error(f"{NEON_RED}Exchange error: Symbol '{symbol}' not found or invalid on {exchange.id} when fetching orderbook. {e}{RESET}")
                 return None # Not retryable
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            # Don't retry on definitive exchange errors unless specifically handled
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry on unexpected errors

        # Increment attempt counter and wait before retrying network/validation issues
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
        market_info: Dict[str, Any], # Pass market info for precision etc.
    ) -> None:
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")
        self.indicator_values: Dict[str, float] = {} # Stores latest indicator float values
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0} # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated fib levels (as Decimals)
        self.ta_column_names: Dict[str, Optional[str]] = {} # Stores actual column names generated by pandas_ta

        if not self.weights:
            logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will not work.{RESET}")

        # Calculate indicators immediately on initialization
        self._calculate_all_indicators()
        # Update latest values immediately after calculation
        self._update_latest_indicator_values()
        # Calculate Fibonacci levels (can be done after indicators)
        self.calculate_fibonacci_levels()

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """Helper to find the actual column name generated by pandas_ta, handling variations."""
        # Common prefixes/suffixes used by pandas_ta
        # Examples: ATRr_14, EMA_10, MOM_10, CCI_20_0.015, WILLR_14, MFI_14, VWAP,
        # PSARl_0.02_0.2, PSARs_0.02_0.2, SMA_10, STOCHRSIk_14_14_3_3, STOCHRSId_14_14_3_3,
        # RSI_14, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, VOL_SMA_15 (custom)
        cfg = self.config # Shortcut
        # Ensure numeric config values are correct type for formatting
        bb_std_dev_float = float(cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV))

        expected_patterns = {
            "ATR": [f"ATRr_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}", f"ATR_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}"], # Added ATR_ variation
            "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_0.015"], # Default const suffix
            "Williams_R": [f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP_D", "VWAP"], # Handle potential suffix like _D for daily reset (pandas_ta >= 0.3.14 adds _D)
            "PSAR_long": [f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev_float:.1f}"],
            "BB_Middle": [f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev_float:.1f}"],
            "BB_Upper": [f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev_float:.1f}"],
            "Volume_MA": [f"VOL_SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"] # Custom name
        }
        patterns = expected_patterns.get(base_name, [])
        for pattern in patterns:
            if pattern in result_df.columns:
                return pattern
            # Check variation without const suffix (e.g., CCI)
            base_pattern_parts = pattern.split('_')
            if len(base_pattern_parts) > 2 and base_pattern_parts[-1].replace('.','').isdigit():
                pattern_no_suffix = '_'.join(base_pattern_parts[:-1])
                if pattern_no_suffix in result_df.columns:
                    self.logger.debug(f"Found column '{pattern_no_suffix}' for base '{base_name}' (without const suffix).")
                    return pattern_no_suffix

        # Check for common parameter-less suffix variations if specific pattern not found
        for pattern in patterns:
             base_pattern = pattern.split('_')[0] # e.g., "ATRr" -> "ATRr"
             if base_pattern in result_df.columns:
                  self.logger.debug(f"Found column '{base_pattern}' for base '{base_name}' (parameter-less variation).")
                  return base_pattern
             # Try base name itself (e.g., "CCI" if "CCI_20_0.015" not found)
             # Convert base_name to uppercase to match pandas_ta conventions if it uses uppercase
             base_name_upper = base_name.upper()
             if base_name_upper in result_df.columns:
                  self.logger.debug(f"Found column '{base_name_upper}' for base '{base_name}'.")
                  return base_name_upper


        # Fallback: search for base name (case-insensitive) if specific pattern not found
        # This is less reliable but might catch variations
        for col in result_df.columns:
            # More specific check: starts with base name + underscore (e.g., "CCI_")
            if col.lower().startswith(base_name.lower() + "_"):
                 self.logger.debug(f"Found column '{col}' for base '{base_name}' using prefix fallback search.")
                 return col
            # Less specific check: contains base name
            # Be careful with this, "RSI" could match "STOCHRSIk"
            # Only use if more specific checks fail
            # if base_name.lower() in col.lower():
            #    self.logger.debug(f"Found column '{col}' for base '{base_name}' using basic fallback search.")
            #    return col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {result_df.columns.tolist()}")
        return None


    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check for sufficient data length
        periods_needed = []
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        # Always calculate ATR for risk management
        periods_needed.append(cfg.get("atr_period", DEFAULT_ATR_PERIOD))
        if indi_cfg.get("ema_alignment"): periods_needed.append(cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        if indi_cfg.get("momentum"): periods_needed.append(cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        if indi_cfg.get("cci"): periods_needed.append(cfg.get("cci_window", DEFAULT_CCI_WINDOW))
        if indi_cfg.get("wr"): periods_needed.append(cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        if indi_cfg.get("mfi"): periods_needed.append(cfg.get("mfi_window", DEFAULT_MFI_WINDOW))
        if indi_cfg.get("sma_10"): periods_needed.append(cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW))
        if indi_cfg.get("stoch_rsi"): periods_needed.append(cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + cfg.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))
        if indi_cfg.get("rsi"): periods_needed.append(cfg.get("rsi_period", DEFAULT_RSI_WINDOW))
        if indi_cfg.get("bollinger_bands"): periods_needed.append(cfg.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        if indi_cfg.get("volume_confirmation"): periods_needed.append(cfg.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
        # VWAP doesn't have a fixed period in the same way, relies on session/day start
        # PSAR starts calculation quickly

        min_required_data = max(periods_needed) + 20 if periods_needed else 50 # Add buffer

        if len(self.df) < min_required_data:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate or NaN.{RESET}")
             # Continue calculation, but expect NaNs

        try:
            # --- Calculate indicators using pandas_ta ---
            # Use the df directly, pandas_ta appends columns by default
            indicators_config = self.config.get("indicators", {})

            # Always calculate ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            self.df.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", self.df)

            # Calculate other indicators based on config flags
            if indicators_config.get("ema_alignment", False):
                ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                self.df.ta.ema(length=ema_short, append=True)
                self.ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", self.df)
                self.df.ta.ema(length=ema_long, append=True)
                self.ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", self.df)

            if indicators_config.get("momentum", False):
                mom_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                self.df.ta.mom(length=mom_period, append=True)
                self.ta_column_names["Momentum"] = self._get_ta_col_name("Momentum", self.df)

            if indicators_config.get("cci", False):
                cci_period = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
                self.df.ta.cci(length=cci_period, append=True)
                self.ta_column_names["CCI"] = self._get_ta_col_name("CCI", self.df)

            if indicators_config.get("wr", False):
                wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
                self.df.ta.willr(length=wr_period, append=True)
                self.ta_column_names["Williams_R"] = self._get_ta_col_name("Williams_R", self.df)

            if indicators_config.get("mfi", False):
                mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
                self.df.ta.mfi(length=mfi_period, append=True)
                self.ta_column_names["MFI"] = self._get_ta_col_name("MFI", self.df)

            if indicators_config.get("vwap", False):
                # VWAP calculation might depend on the frequency (e.g., daily reset)
                # pandas_ta vwap usually assumes daily reset based on timestamp index
                self.df.ta.vwap(append=True)
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", self.df)

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                # PSAR calculation appends columns directly
                self.df.ta.psar(af=psar_af, max_af=psar_max_af, append=True)
                self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", self.df)
                self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", self.df)

            if indicators_config.get("sma_10", False):
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                self.df.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", self.df)

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                # StochRSI calculation appends columns directly
                self.df.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d, append=True)
                self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", self.df)
                self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", self.df)

            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                self.df.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", self.df)

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
                bb_std_float = float(bb_std)
                # BBands calculation appends columns directly
                self.df.ta.bbands(length=bb_period, std=bb_std_float, append=True)
                self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", self.df)
                self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", self.df)
                self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", self.df)

            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}" # Custom name
                # Calculate SMA on volume column, handle potential NaNs in volume
                self.df[vol_ma_col_name] = ta.sma(self.df['volume'].fillna(0), length=vol_ma_period) # Use ta.sma directly
                self.ta_column_names["Volume_MA"] = vol_ma_col_name


            # Assign the df with calculated indicators back to self.df (already modified in place)
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
            # Check if it's a pandas_ta specific error
            if 'DataFrameTA' in str(e) or '.ta.' in str(e):
                 self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} using pandas_ta. Check method name, version, or input data: {e}{RESET}", exc_info=True)
            else:
                 self.logger.error(f"{NEON_RED}AttributeError during indicator calculation: {e}{RESET}", exc_info=True)
             # self.df might have partial calculations, but some indicators will be missing
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)
            # Decide how to handle - clear df or keep original? Keep partially calculated for now.

        # Note: _update_latest_indicator_values is called after this in __init__


    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest float values from self.df."""
        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame is empty for {self.symbol}.")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
            return

        try:
            # Check if the index is valid datetime index
            if not isinstance(self.df.index, pd.DatetimeIndex):
                self.logger.error(f"DataFrame index is not a DatetimeIndex for {self.symbol}. Cannot reliably get latest values.")
                self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
                return

            latest = self.df.iloc[-1] # Get the last row

            # Check if the last row contains any non-NaN values before proceeding
            if latest.isnull().all():
                self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: Last row of DataFrame contains all NaNs for {self.symbol} at {self.df.index[-1]}.{RESET}")
                self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
                return
        except IndexError:
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty or too short.")
             self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]} # Reset values
             return

        try:
            # latest = self.df.iloc[-1] # Already fetched above
            updated_values = {}

            # Use the dynamically stored column names from self.ta_column_names
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index: # Check if column name exists and is valid
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            updated_values[key] = float(value) # Store as float
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) to float for {self.symbol}.")
                            updated_values[key] = np.nan
                    else:
                        updated_values[key] = np.nan # Value is NaN in DataFrame
                else:
                    # If col_name is None or not in index, store NaN
                    # Log only if the indicator was supposed to be calculated (i.e., key exists)
                    if key in self.ta_column_names: # Check if key was attempted
                        self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found or invalid in latest data for {self.symbol}. Storing NaN.")
                    updated_values[key] = np.nan

            # Add essential price/volume data from the original DataFrame columns
            for base_col in ['close', 'volume', 'high', 'low']:
                value = latest.get(base_col, np.nan)
                key_name = base_col.capitalize() # e.g., 'Close'
                if pd.notna(value):
                    try:
                        updated_values[key_name] = float(value)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert base value for '{base_col}' ({value}) to float for {self.symbol}.")
                        updated_values[key_name] = np.nan
                else:
                    updated_values[key_name] = np.nan


            self.indicator_values = updated_values
            # Filter out NaN for debug log brevity
            valid_values = {k: f"{v:.5f}" if isinstance(v, float) else v for k, v in self.indicator_values.items() if pd.notna(v)}
            self.logger.debug(f"Latest indicator float values updated for {self.symbol} (Timestamp: {self.df.index[-1]}): {valid_values}")

        except IndexError: # Catch again just in case df became empty between checks
            self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty or too short.")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]} # Reset values
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]} # Reset values


    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a specified window using Decimal."""
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
            if diff > 0:
                min_tick = self.get_min_tick_size()
                if min_tick <= 0:
                    self.logger.warning(f"Invalid min_tick_size ({min_tick}) for Fibonacci quantization on {self.symbol}. Levels will not be quantized.")
                    min_tick = None # Disable quantization if tick size is invalid

                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level: High - (Range * Pct) for downtrend assumption (standard)
                    # Or Low + (Range * Pct) for uptrend assumption
                    # Using High - diff * level assumes retracement from High towards Low
                    level_price = (high - (diff * Decimal(str(level_pct))))

                    # Quantize the result to the market's price precision (using tick size) if possible
                    if min_tick:
                        # Quantize DOWN for potential support levels (0-100%)
                        level_price_quantized = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    else:
                        level_price_quantized = level_price # Use raw value if quantization fails

                    levels[level_name] = level_price_quantized
            else:
                 # If high == low, all levels will be the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}. Setting levels to high/low.")
                 min_tick = self.get_min_tick_size()
                 if min_tick > 0:
                     level_price_quantized = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 else:
                     level_price_quantized = high # Use raw if quantization fails
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { {k: str(v) for k,v in levels.items()} }") # Log as strings
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info, using min_tick_size as primary source."""
        try:
            min_tick = self.get_min_tick_size() # Rely on tick size first
            if min_tick > 0:
                # Calculate decimal places from tick size
                # normalize() removes trailing zeros, as_tuple().exponent gives power of 10
                precision = abs(min_tick.normalize().as_tuple().exponent)
                # self.logger.debug(f"Derived price precision {precision} from min tick size {min_tick} for {self.symbol}")
                return precision
            else:
                 self.logger.warning(f"Min tick size ({min_tick}) is invalid for {self.symbol}. Attempting fallback precision methods.")

        except Exception as e:
            self.logger.warning(f"Error getting/using min tick size for precision derivation ({self.symbol}): {e}. Attempting fallback methods.")

        # Fallback 1: Use ccxt market['precision']['price'] if it's an integer (decimal places)
        try:
             precision_info = self.market_info.get('precision', {})
             price_precision_val = precision_info.get('price')
             if isinstance(price_precision_val, int):
                 self.logger.debug(f"Using price precision from market_info.precision.price (integer): {price_precision_val} for {self.symbol}")
                 return price_precision_val
        except Exception as e:
             self.logger.debug(f"Could not get integer precision from market_info.precision.price: {e}")


        # Fallback 2: Infer from last close price format if tick size failed
        try:
            last_close = self.indicator_values.get("Close") # Uses float value
            if last_close and pd.notna(last_close) and last_close > 0:
                try:
                    # Use Decimal formatting to handle scientific notation properly
                    s_close = format(Decimal(str(last_close)), 'f') # Format to avoid scientific notation
                    if '.' in s_close:
                        precision = len(s_close.split('.')[-1].rstrip('0')) # Count digits after point, ignore trailing zeros
                        self.logger.debug(f"Inferring price precision from last close price ({s_close}) as {precision} for {self.symbol}.")
                        return precision
                    else:
                        return 0 # No decimal places
                except Exception as e_close:
                    self.logger.warning(f"Error inferring precision from close price {last_close}: {e_close}")
        except Exception as e_outer:
            self.logger.warning(f"Could not access/parse close price for precision fallback: {e_outer}")

        # Default fallback precision
        default_precision = 4 # Common default for USDT pairs, adjust if needed
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision


    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        try:
            # CCXT precision structure often contains tick size directly or indirectly
            precision_info = self.market_info.get('precision', {})
            # Priority 1: Check for 'tick' key specifically (added by our get_market_info enhancement)
            tick_val = precision_info.get('tick')
            if tick_val is not None:
                try:
                    tick_size = Decimal(str(tick_val))
                    if tick_size > 0:
                        # self.logger.debug(f"Using tick size from precision.tick: {tick_size} for {self.symbol}")
                        return tick_size
                except Exception:
                    self.logger.debug(f"Could not parse precision.tick '{tick_val}' as Decimal for {self.symbol}.")

            # Priority 2: Use precision.price (often represents tick size)
            price_precision_val = precision_info.get('price') # This is usually the tick size as a float/string
            if price_precision_val is not None:
                try:
                    # Avoid interpreting integer precision as tick size here
                    if isinstance(price_precision_val, (float, str)):
                        tick_size = Decimal(str(price_precision_val))
                        if tick_size > 0:
                            # self.logger.debug(f"Using tick size from precision.price: {tick_size} for {self.symbol}")
                            return tick_size
                except Exception:
                    self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as Decimal for {self.symbol}. Trying other methods.")

            # Fallback 3: Check limits.price.min (sometimes this represents tick size)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                try:
                    min_tick_from_limit = Decimal(str(min_price_val))
                    if min_tick_from_limit > 0:
                        self.logger.debug(f"Using tick size from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                        return min_tick_from_limit
                except Exception:
                    self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' as Decimal for {self.symbol}.")

            # Fallback 4: Check Bybit specific 'tickSize' in info (V5)
            info_dict = self.market_info.get('info', {})
            bybit_tick_size = info_dict.get('tickSize')
            if bybit_tick_size is not None:
                 try:
                      tick_size_bybit = Decimal(str(bybit_tick_size))
                      if tick_size_bybit > 0:
                          self.logger.debug(f"Using tick size from market_info['info']['tickSize']: {tick_size_bybit} for {self.symbol}")
                          return tick_size_bybit
                 except Exception:
                      self.logger.debug(f"Could not parse info['tickSize'] '{bybit_tick_size}' as Decimal for {self.symbol}.")


        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using default fallback.")

        # Absolute fallback: A very small number if everything else fails
        fallback_tick = Decimal('0.00000001') # Small default
        self.logger.warning(f"Using extremely small fallback tick size for {self.symbol}: {fallback_tick}. Price quantization may be inaccurate.")
        return fallback_tick


    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels (name, price) to the current price."""
        if not self.fib_levels_data:
            # Don't recalculate here, rely on initial calculation
            # self.logger.debug(f"Fibonacci levels not calculated or empty for {self.symbol}. Cannot find nearest.")
            return []

        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal): # Ensure level is Decimal
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
            # self.logger.debug(f"EMA alignment check skipped for {self.symbol}: Missing required values.")
            return np.nan # Return NaN if data is missing

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
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0} # Reset signals
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {} # For logging individual scores

        # --- Essential Data Checks ---
        if not self.indicator_values:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values dictionary is empty.{RESET}")
            return "HOLD"
        # Check if *any* core indicator value (excluding price/vol/high/low) is non-NaN
        core_indicators_present = any(pd.notna(v) for k, v in self.indicator_values.items() if k not in ['Close', 'Volume', 'High', 'Low'])
        if not core_indicators_present:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: All core indicator values are NaN.{RESET}")
            return "HOLD"
        if current_price is None or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
            return "HOLD"

        # --- Get Active Weights ---
        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
            self.logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' is missing or empty in config for {self.symbol}. Cannot generate signal.{RESET}")
            return "HOLD"

        # --- Iterate Through Enabled Indicators with Weights ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue # Skip disabled indicators

            weight_str = active_weights.get(indicator_key)
            if weight_str is None:
                # Log if indicator is enabled but has no weight defined
                self.logger.debug(f"Indicator '{indicator_key}' enabled but has no weight in set '{self.active_weight_set_name}'. Skipping.")
                continue

            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue # Skip if weight is zero
            except Exception:
                self.logger.warning(f"{NEON_YELLOW}Invalid weight format '{weight_str}' for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping.{RESET}")
                continue

            # Find and call the check method
            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method = getattr(self, check_method_name)
                indicator_score = np.nan # Default to NaN
                try:
                    # Pass specific arguments if needed (e.g., orderbook)
                    if indicator_key == "orderbook":
                        if orderbook_data: # Only call if data exists
                            indicator_score = method(orderbook_data, current_price)
                        else:
                            self.logger.debug(f"Orderbook data not available for {self.symbol}, skipping orderbook check.")
                            indicator_score = np.nan # Treat as NaN if data missing
                    else:
                        indicator_score = method() # Returns float score or np.nan

                except Exception as e:
                    self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                    indicator_score = np.nan # Treat as NaN on error

                # --- Process Score ---
                debug_scores[indicator_key] = f"{indicator_score:.2f}" if pd.notna(indicator_score) else "NaN"
                if pd.notna(indicator_score):
                    try:
                        score_decimal = Decimal(str(indicator_score)) # Convert float score to Decimal
                        # Clamp score between -1 and 1 before applying weight
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_decimal))
                        score_contribution = clamped_score * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight
                        active_indicator_count += 1
                        # Detailed debug log inside the loop can be verbose, moved summary outside
                    except Exception as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} ({indicator_score}): {calc_err}")
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1
            else:
                self.logger.warning(f"Check method '{check_method_name}' not found or not callable for enabled/weighted indicator: {indicator_key} ({self.symbol})")


        # --- Determine Final Signal ---
        if total_weight_applied == 0:
            self.logger.warning(f"{NEON_YELLOW}No indicators contributed to the signal score for {self.symbol} (Total Weight Applied = 0). Defaulting to HOLD.{RESET}")
            final_signal = "HOLD"
        else:
            # Normalize score? Optional. Threshold works on weighted sum.
            # normalized_score = final_signal_score / total_weight_applied if total_weight_applied != 0 else Decimal(0)
            threshold = Decimal(str(self.config.get("signal_score_threshold", 1.5)))

            if final_signal_score >= threshold:
                final_signal = "BUY"
            elif final_signal_score <= -threshold:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"

        # --- Log Summary ---
        # Format score contributions for logging
        score_details = ", ".join([f"{k}: {v}" for k, v in debug_scores.items()])
        price_precision = self.get_price_precision() # Get precision for logging current price
        log_msg = (
            f"Signal Calculation Summary ({self.symbol} @ {current_price:.{price_precision}f}):\n"
            f"  Weight Set: {self.active_weight_set_name}\n"
            # f"  Scores: {score_details}\n" # Can be very long, log at DEBUG if needed
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN)\n"
            f"  Total Weight Applied: {total_weight_applied:.3f}\n"
            f"  Final Weighted Score: {final_signal_score:.4f}\n"
            f"  Signal Threshold: +/- {threshold:.3f}\n"
            f"  ==> Final Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        if console_log_level <= logging.DEBUG: # Only log detailed scores if console level is DEBUG
            self.logger.debug(f"  Detailed Scores: {score_details}")

        # Update internal signal state
        if final_signal in self.signals:
            self.signals[final_signal] = 1

        return final_signal


    # --- Indicator Check Methods ---
    # Each method should return a float score between -1.0 and 1.0, or np.nan if data invalid/missing

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment. Relies on calculate_ema_alignment_score."""
        # Ensure the indicator was calculated if this check is enabled
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
            self.logger.debug(f"EMA Alignment check skipped for {self.symbol}: EMAs not in indicator_values.")
            return np.nan
        # calculate_ema_alignment_score already handles NaNs internally
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        """Checks Momentum indicator."""
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan
        # Normalize score based on momentum magnitude (simple example)
        # These thresholds might need tuning based on typical MOM values for the asset/interval
        # Example: If MOM is typically +/- 0.5, scale based on that range
        # Let's assume a typical range of +/- 0.2 for this example scaling
        # A more robust approach might use ATR or % change
        scale_factor = 5.0 # Scales +/-0.2 to +/-1.0
        score = momentum * scale_factor
        return max(-1.0, min(1.0, score)) # Clamp score between -1 and 1

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume supports potential move (relative to MA). Score is direction-neutral."""
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5)) # Use float

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma <= 0:
            return np.nan

        try:
            if current_volume > volume_ma * multiplier:
                # High volume suggests stronger conviction or potential climax/exhaustion.
                # Positive score indicates significance, not direction.
                return 0.7 # Strong confirmation/significance
            elif current_volume < volume_ma / multiplier:
                # Low volume suggests lack of interest or consolidation.
                return -0.4 # Negative score indicates lack of confirmation
            else:
                return 0.0 # Neutral volume
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check calculation failed for {self.symbol}: {e}{RESET}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Checks Stochastic RSI K and D lines."""
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan

        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))

        # --- Scoring Logic ---
        score = 0.0
        # 1. Extreme Zones (strongest signals)
        if k < oversold and d < oversold:
            score = 1.0 # Both deep oversold -> Strong bullish
        elif k > overbought and d > overbought:
            score = -1.0 # Both deep overbought -> Strong bearish

        # 2. K vs D Relationship (momentum indication)
        # Give higher weight to K crossing D than just K being above/below D
        # Requires previous state, approximating: if K is significantly different from D
        diff = k - d
        cross_threshold = 5.0 # Threshold for significant difference/cross potential (tune this)
        if abs(diff) > cross_threshold:
            if diff > 0: # K moved above D (or is significantly above)
                # Combine bullish momentum with zone score (stronger if oversold)
                score = max(score, 0.6) if score >= 0 else 0.6 # If already bearish (-1), overwrite partially
            else: # K moved below D (or is significantly below)
                # Combine bearish momentum with zone score (stronger if overbought)
                score = min(score, -0.6) if score <= 0 else -0.6 # If already bullish (+1), overwrite partially
        else: # K and D are close
            # Less strong signal based on which is higher
            if k > d : score = max(score, 0.2) # Weak bullish bias if not already strongly signaled
            elif k < d: score = min(score, -0.2) # Weak bearish bias

        # 3. Consider position within range (0-100) - less important than extremes/crosses
        if oversold <= k <= overbought: # Inside normal range
            # Optionally scale based on proximity to mid-point (50)
            range_width = overbought - oversold
            if range_width > 0:
                 mid_range_score = (k - (oversold + range_width / 2)) / (range_width / 2) # Scales -1 to 1 within the range
                 # Combine with existing score (e.g., average or weighted average)
                 # Give mid-range position less weight than extremes or crosses
                 score = (score + mid_range_score * 0.3) / 1.3
            # else: score remains unchanged if range is zero

        return max(-1.0, min(1.0, score)) # Clamp final score

    def _check_rsi(self) -> float:
        """Checks RSI indicator."""
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan

        if rsi <= 30: return 1.0
        if rsi >= 70: return -1.0
        if rsi < 40: return 0.5 # Leaning oversold/bullish
        if rsi > 60: return -0.5 # Leaning overbought/bearish
        # Smoother transition in the middle
        if 40 <= rsi <= 60:
            # Linear scale from +0.5 (at 40) to -0.5 (at 60)
            return 0.5 - (rsi - 40) * (1.0 / 20.0)
        return 0.0 # Fallback

    def _check_cci(self) -> float:
        """Checks CCI indicator."""
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan
        # CCI extremes often signal reversal potential
        if cci <= -150: return 1.0 # Strong Oversold -> Bullish
        if cci >= 150: return -1.0 # Strong Overbought -> Bearish
        if cci < -80: return 0.6 # Moderately Oversold
        if cci > 80: return -0.6 # Moderately Overbought
        # Trend confirmation near zero line - scale based on value
        if -80 <= cci <= 80:
            # Linear scale from +0.6 (at -80) to -0.6 (at 80)
             return - (cci / 80.0) * 0.6
        return 0.0 # Fallback

    def _check_wr(self) -> float: # Williams %R
        """Checks Williams %R indicator."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        # WR: -100 (most oversold) to 0 (most overbought)
        if wr <= -80: return 1.0 # Oversold -> Bullish
        if wr >= -20: return -1.0 # Overbought -> Bearish
        # Scale linearly in the middle range (-80 to -20)
        if -80 < wr < -20:
            # Scale from +1.0 (at -80) to -1.0 (at -20)
            return 1.0 - (wr - (-80.0)) * (2.0 / 60.0)
        # Handle edge cases (exactly -80 or -20) or values outside range
        elif wr == -80: return 1.0
        elif wr == -20: return -1.0
        elif wr < -80: return 1.0 # Treat below -80 as 1.0
        elif wr > -20: return -1.0 # Treat above -20 as -1.0
        return 0.0 # Fallback (shouldn't be reached with above logic)


    def _check_psar(self) -> float:
        """Checks Parabolic SAR relative to price."""
        psar_l = self.indicator_values.get("PSAR_long", np.nan)
        psar_s = self.indicator_values.get("PSAR_short", np.nan)
        # PSAR values themselves indicate the stop level.
        # The signal comes from which one is active (non-NaN).
        if pd.notna(psar_l) and pd.isna(psar_s):
            # PSAR is below price (long value is active) -> Uptrend
            return 1.0
        elif pd.notna(psar_s) and pd.isna(psar_l):
            # PSAR is above price (short value is active) -> Downtrend
            return -1.0
        else:
            # Both NaN (start of data) or both have values (shouldn't happen with ta.psar)
            # self.logger.debug(f"PSAR state ambiguous/NaN for {self.symbol} (PSAR_long={psar_l}, PSAR_short={psar_s})")
            return 0.0 # Neutral or undetermined

    def _check_sma_10(self) -> float: # Example using SMA10
        """Checks price relative to SMA10."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return np.nan
        # Simple crossover score
        if last_close > sma_10: return 0.6 # Price above SMA -> Bullish bias
        if last_close < sma_10: return -0.6 # Price below SMA -> Bearish bias
        return 0.0

    def _check_vwap(self) -> float:
        """Checks price relative to VWAP."""
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return np.nan
        # VWAP acts as a dynamic support/resistance or fair value indicator
        if last_close > vwap: return 0.7 # Price above VWAP -> Bullish intraday sentiment
        if last_close < vwap: return -0.7 # Price below VWAP -> Bearish intraday sentiment
        return 0.0

    def _check_mfi(self) -> float:
        """Checks Money Flow Index."""
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan
        # MFI combines price and volume for overbought/oversold signals
        if mfi <= 20: return 1.0 # Oversold -> Bullish potential
        if mfi >= 80: return -1.0 # Overbought -> Bearish potential
        if mfi < 40: return 0.4 # Leaning oversold/accumulation
        if mfi > 60: return -0.4 # Leaning overbought/distribution
        # Scale linearly in the middle range (40 to 60)
        if 40 <= mfi <= 60:
            # Scale from +0.4 (at 40) to -0.4 (at 60)
            return 0.4 - (mfi - 40) * (0.8 / 20.0)
        return 0.0 # Fallback

    def _check_bollinger_bands(self) -> float:
        """Checks price relative to Bollinger Bands."""
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        bb_middle = self.indicator_values.get("BB_Middle", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(last_close):
            return np.nan

        # 1. Price relative to outer bands (mean reversion signals)
        if last_close < bb_lower: return 1.0 # Below lower band -> Strong bullish potential
        if last_close > bb_upper: return -1.0 # Above upper band -> Strong bearish potential

        # 2. Price relative to middle band (trend confirmation)
        # Bandwidth calculation for volatility context could also be added
        band_width = (bb_upper - bb_lower)
        if band_width <= 0: return 0.0 # Avoid division by zero if bands are flat

        # Avoid division by zero if middle band = upper/lower band
        upper_range = bb_upper - bb_middle
        lower_range = bb_middle - bb_lower

        if last_close > bb_middle:
            # Price above middle band -> weaker bullish signal (already in upper half)
            # Scale score based on proximity to upper band: closer to upper band -> weaker bullish / stronger bearish potential
            # Example: Score from +0.5 (at middle) to -0.5 (at upper band)
            proximity_to_upper = (last_close - bb_middle) / upper_range if upper_range > 0 else 0
            score = 0.5 - proximity_to_upper # Max +0.5, min -0.5 (approx)
            return max(-0.5, min(0.5, score)) # Clamp result

        if last_close < bb_middle:
            # Price below middle band -> weaker bearish signal
            # Scale score based on proximity to lower band: closer to lower band -> weaker bearish / stronger bullish potential
            # Example: Score from -0.5 (at middle) to +0.5 (at lower band)
            proximity_to_lower = (bb_middle - last_close) / lower_range if lower_range > 0 else 0
            score = -0.5 + proximity_to_lower # Max +0.5, min -0.5 (approx)
            return max(-0.5, min(0.5, score)) # Clamp result

        return 0.0 # Exactly on middle band


    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes order book depth for immediate pressure. Returns float score or NaN."""
        if not orderbook_data:
            self.logger.debug(f"Orderbook check skipped for {self.symbol}: No data provided.")
            return np.nan

        try:
            bids = orderbook_data.get('bids', []) # List of [price (str), volume (str)]
            asks = orderbook_data.get('asks', []) # List of [price (str), volume (str)]

            if not bids or not asks:
                self.logger.debug(f"Orderbook check skipped for {self.symbol}: Bids or asks list is empty.")
                return np.nan # Need both sides for imbalance calculation

            # --- Simple Order Book Imbalance (OBI) within N levels ---
            # Alternative: Use % range around current price
            num_levels_to_check = min(10, len(bids), len(asks)) # Check top N levels, limited by available depth
            if num_levels_to_check == 0:
                self.logger.debug(f"Orderbook check: Not enough depth (bids: {len(bids)}, asks: {len(asks)}) for {self.symbol}.")
                return 0.0 # Neutral if not enough depth

            bid_volume_sum = sum(Decimal(str(bid[1])) for bid in bids[:num_levels_to_check])
            ask_volume_sum = sum(Decimal(str(ask[1])) for ask in asks[:num_levels_to_check])

            total_volume = bid_volume_sum + ask_volume_sum
            if total_volume == 0:
                self.logger.debug(f"Orderbook check: No volume within top {num_levels_to_check} levels for {self.symbol}.")
                return 0.0 # Neutral if no volume

            # Calculate Order Book Imbalance (OBI) ratio: (Bids - Asks) / Total
            obi = (bid_volume_sum - ask_volume_sum) / total_volume

            # Scale OBI to a score between -1 and 1
            # Direct scaling: OBI already ranges from -1 to 1
            score = float(obi)

            # Optional: Make the score more sensitive to strong imbalances
            # Example: score = float(obi) ** 3 # Cube enhances values closer to +/- 1

            self.logger.debug(f"Orderbook check ({self.symbol}): Top {num_levels_to_check} Levels: "
                              f"BidVol={bid_volume_sum:.4f}, AskVol={ask_volume_sum:.4f}, "
                              f"OBI={obi:.4f}, Score={score:.4f}")
            return score

        except (ValueError, TypeError) as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol} due to parsing error: {e}{RESET}")
            return np.nan # Return NaN on parsing error
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Unexpected orderbook analysis failure for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan # Return NaN on other errors


    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on the provided entry price, ATR, and configured multipliers. Uses Decimal precision.
        The initial SL calculated here is primarily used for position sizing and setting the initial protection.
        Returns (entry_price, take_profit, stop_loss), all as Decimal or None.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None # No TP/SL needed for HOLD

        atr_val_float = self.indicator_values.get("ATR") # Get float ATR value
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float)) # Convert valid float ATR to Decimal

            # Get multipliers from config, convert to Decimal
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision for logging
            price_precision = self.get_price_precision()
            # Get minimum price increment (tick size) for quantization
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"{NEON_RED}Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).{RESET}")
                 return entry_price, None, None


            take_profit = None
            stop_loss = None

            if signal == "BUY":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Quantize TP UP (more profit), SL DOWN (more room) to the nearest tick size
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick

            elif signal == "SELL":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Quantize TP DOWN (more profit), SL UP (more room) to the nearest tick size
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick


            # --- Validation ---
            # Ensure SL is actually beyond entry by at least one tick
            min_sl_distance = min_tick # Minimum distance from entry for SL
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_sl_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation ({stop_loss:.{price_precision}f}) is too close to or above entry ({entry_price:.{price_precision}f}). Adjusting SL down to {adjusted_sl:.{price_precision}f}.{RESET}")
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_sl_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation ({stop_loss:.{price_precision}f}) is too close to or below entry ({entry_price:.{price_precision}f}). Adjusting SL up to {adjusted_sl:.{price_precision}f}.{RESET}")
                stop_loss = adjusted_sl

            # Ensure TP is potentially profitable relative to entry (at least one tick away)
            min_tp_distance = min_tick # Minimum distance from entry for TP
            if signal == "BUY" and take_profit <= entry_price:
                adjusted_tp = (entry_price + min_tp_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}BUY signal TP calculation ({take_profit:.{price_precision}f}) resulted in non-profitable level. Adjusting TP up to {adjusted_tp:.{price_precision}f}.{RESET}")
                take_profit = adjusted_tp
            elif signal == "SELL" and take_profit >= entry_price:
                adjusted_tp = (entry_price - min_tp_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}SELL signal TP calculation ({take_profit:.{price_precision}f}) resulted in non-profitable level. Adjusting TP down to {adjusted_tp:.{price_precision}f}.{RESET}")
                take_profit = adjusted_tp

            # Final checks: Ensure SL/TP are still valid (positive price) after adjustments
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in zero or negative price ({stop_loss}) for {self.symbol}. Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}Take profit calculation resulted in zero or negative price ({take_profit}) for {self.symbol}. Cannot set TP.{RESET}")
                take_profit = None

            # Format for logging
            tp_log = f"{take_profit:.{price_precision}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{price_precision}f}" if stop_loss else 'N/A'
            atr_log = f"{atr:.{price_precision+1}f}" if atr else 'N/A'
            entry_log = f"{entry_price:.{price_precision}f}" if entry_price else 'N/A'

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log}, ATR={atr_log}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency, handling Bybit V5 structures."""
    lg = logger
    try:
        balance_info = None
        # For Bybit V5, specify the account type relevant to the market (e.g., CONTRACT, UNIFIED)
        # Prioritize 'CONTRACT' for derivatives, then 'UNIFIED'.
        account_types_to_try = ['CONTRACT', 'UNIFIED']
        fetched_type = None # Track which type succeeded

        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency}...")
                # Bybit V5 expects 'accountType' in params
                balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                fetched_type = acc_type # Mark this type as fetched

                # Check if the desired currency is present and has usable balance info
                # Primary check: Use Bybit V5 nested structure
                if 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                    balance_list = balance_info['info']['result']['list']
                    for account in balance_list:
                        # Match the account type we requested
                        if account.get('accountType') == acc_type:
                            coin_list = account.get('coin')
                            if isinstance(coin_list, list):
                                for coin_data in coin_list:
                                    if coin_data.get('coin') == currency:
                                        # Prefer Bybit V5 'availableToWithdraw' or 'availableBalance'
                                        free = coin_data.get('availableToWithdraw', coin_data.get('availableBalance'))
                                        if free is not None:
                                            lg.debug(f"Found balance via Bybit V5 nested structure (Type: {acc_type}, Field: 'availableToWithdraw'/'availableBalance')")
                                            balance_info['parsed_balance'] = str(free) # Store parsed value
                                            break # Found currency in this account type
                                if 'parsed_balance' in balance_info: break # Found in coin list
                    if 'parsed_balance' in balance_info: break # Found in account list

                # Secondary check: Standard CCXT structure (less likely for Bybit V5 with params)
                elif currency in balance_info and balance_info[currency].get('free') is not None:
                    lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free'] (Type: {acc_type})")
                    balance_info['parsed_balance'] = str(balance_info[currency]['free'])
                    break # Found balance

                # If neither found for this type, reset balance_info to try next type
                lg.debug(f"Currency '{currency}' balance not found in expected structure for type '{acc_type}'. Trying next.")
                balance_info = None
                fetched_type = None

            except ccxt.ExchangeError as e:
                # Ignore errors indicating the account type doesn't exist or mismatch, try the next one
                if "account type not support" in str(e).lower() or "invalid account type" in str(e).lower() or "account type mismatch" in str(e).lower():
                    lg.debug(f"Account type '{acc_type}' not supported or mismatch error: {e}. Trying next.")
                    balance_info = None # Ensure reset
                    fetched_type = None
                    continue
                else:
                    lg.warning(f"Exchange error fetching balance for account type {acc_type}: {e}. Trying next.")
                    balance_info = None
                    fetched_type = None
                    continue
            except Exception as e:
                lg.warning(f"Unexpected error fetching balance for account type {acc_type}: {e}. Trying next.")
                balance_info = None
                fetched_type = None
                continue

        # If specific account types failed, try default fetch_balance without params (might work for Spot or older API versions)
        if not balance_info:
            lg.debug(f"Specific account type fetches failed. Fetching balance using default parameters for {currency}...")
            try:
                balance_info = exchange.fetch_balance()
                fetched_type = 'DEFAULT'
                # Check standard structure after default fetch
                if currency in balance_info and balance_info[currency].get('free') is not None:
                    lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free'] (Default fetch)")
                    balance_info['parsed_balance'] = str(balance_info[currency]['free'])
                # Check Bybit nested structure again (less likely without type param but possible)
                elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                     balance_list = balance_info['info']['result']['list']
                     for account in balance_list: # Check all account types in the list
                         coin_list = account.get('coin')
                         if isinstance(coin_list, list):
                             for coin_data in coin_list:
                                 if coin_data.get('coin') == currency:
                                     free = coin_data.get('availableToWithdraw', coin_data.get('availableBalance'))
                                     if free is not None:
                                         lg.debug(f"Found balance via Bybit V5 nested structure (Default fetch, Account: {account.get('accountType')})")
                                         balance_info['parsed_balance'] = str(free)
                                         break
                             if 'parsed_balance' in balance_info: break
                     if 'parsed_balance' not in balance_info:
                         lg.debug("Currency balance not found in nested structure during default fetch.")

                elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                    # Fallback check: Top-level 'free' dictionary
                    lg.debug(f"Found balance via top-level 'free' dictionary (Default fetch)")
                    balance_info['parsed_balance'] = str(balance_info['free'][currency])
                else:
                    lg.warning(f"Could not find '{currency}' balance in default fetch_balance response structure.")
                    balance_info = None # Mark as failed

            except Exception as e:
                lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                return None


        # --- Parse the balance_info ---
        available_balance_str = None
        if balance_info and 'parsed_balance' in balance_info:
            available_balance_str = balance_info['parsed_balance']
        elif balance_info:
            # Last Resort: Check 'total' balance if 'free'/'available' is still missing
            total_balance = None
            # Standard total
            if currency in balance_info and balance_info[currency].get('total') is not None:
                total_balance = balance_info[currency]['total']
            # Bybit V5 nested total (walletBalance)
            elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                 balance_list = balance_info['info']['result']['list']
                 for account in balance_list:
                    # If we know the successful type, check that first, else check all
                    if fetched_type == 'DEFAULT' or account.get('accountType') == fetched_type:
                        coin_list = account.get('coin')
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get('coin') == currency:
                                    total_balance = coin_data.get('walletBalance') # Use walletBalance as total proxy
                                    if total_balance is not None: break
                            if total_balance is not None: break
                    if total_balance is not None: break # Found in some account

            if total_balance is not None:
                lg.warning(f"{NEON_YELLOW}Could not determine 'free'/'available' balance for {currency} (Type: {fetched_type}). Using 'total' balance ({total_balance}) as fallback. This might include collateral/unrealized PnL.{RESET}")
                available_balance_str = str(total_balance)

        # If still no balance string after all checks
        if available_balance_str is None:
            lg.error(f"{NEON_RED}Could not determine any balance for {currency}. Balance info structure not recognized or currency missing.{RESET}")
            lg.debug(f"Full balance_info structure (Type: {fetched_type}): {balance_info}") # Log structure for debugging
            return None

        # --- Convert to Decimal ---
        try:
            final_balance = Decimal(available_balance_str)
            if final_balance >= 0: # Allow zero balance
                lg.info(f"Available {currency} balance: {final_balance:.4f} (Fetched Type: {fetched_type})")
                return final_balance
            else:
                lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Returning None.")
                return None
        except Exception as e:
            lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching balance: {e}. Check API key permissions.{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}")
        # Consider if balance fetch should be retried or if failure is critical
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
        return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision, limits, contract type, ensuring markets are loaded."""
    lg = logger
    try:
        # Ensure markets are loaded; reload if symbol is missing
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded or symbol missing, reloading markets...")
            exchange.load_markets(reload=True) # Force reload

        # Check again after reloading
        if symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Market {symbol} still not found after reloading markets. Check symbol spelling and availability on {exchange.id}.{RESET}")
            # Provide suggestions if possible
            try:
                possible_matches = [m for m in exchange.symbols if symbol.split('/')[0] in m] # Simple base match
                if possible_matches:
                     lg.info(f"Possible matches found: {possible_matches[:10]}")
            except Exception: pass
            return None

        market = exchange.market(symbol)
        if market:
            # Log key details for confirmation and debugging
            market_type = market.get('type', 'unknown') # spot, swap, future etc.
            contract_type = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "N/A"
            # Add contract flag for easier checking later
            market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']

            # --- Enhance market info with derived tick size ---
            # Create a temporary analyzer instance just to use utility methods
            # We need a valid DataFrame structure for initialization, even if empty
            dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            dummy_df.index.name = 'timestamp'
            # Use an empty config for the temp analyzer
            analyzer_temp = TradingAnalyzer(dummy_df, lg, {}, market) # Temp instance for helper
            min_tick = analyzer_temp.get_min_tick_size()
            if min_tick > 0:
                 if 'precision' not in market: market['precision'] = {}
                 # Store as float for consistency with CCXT's typical precision structure
                 # Avoid overwriting if 'tick' already exists and is valid
                 if market['precision'].get('tick') is None or float(market['precision'].get('tick', 0)) <= 0:
                     market['precision']['tick'] = float(min_tick)
                     lg.debug(f"Added/Updated derived min tick size {min_tick} to market info precision.")
            else:
                 lg.warning(f"Could not derive a valid positive tick size for {symbol} to add to market info.")

            # Log amount precision details
            amount_precision_val = market.get('precision', {}).get('amount')
            amount_precision_type = "N/A"
            if isinstance(amount_precision_val, int): amount_precision_type = f"Decimals ({amount_precision_val})"
            elif isinstance(amount_precision_val, (float, str)): amount_precision_type = f"Step ({amount_precision_val})"


            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract={contract_type}, "
                f"Precision(Price/Amount/Tick): {market.get('precision', {}).get('price')}/{amount_precision_type}/{market.get('precision', {}).get('tick', 'N/A')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            return market
        else:
            # Should have been caught by the 'in exchange.markets' check, but safeguard
            lg.error(f"{NEON_RED}Market dictionary not found for {symbol} even after checking exchange.markets.{RESET}")
            return None
    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Symbol '{symbol}' is not supported by {exchange.id} or is incorrectly formatted: {e}{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error loading markets info for {symbol}: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error loading markets info for {symbol}: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal, # The calculated initial SL price (Decimal)
    entry_price: Decimal, # Estimated entry price (e.g., current market price) (Decimal)
    market_info: Dict,
    exchange: ccxt.Exchange, # Pass exchange object for formatting helpers
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size based on risk percentage, initial SL distance, balance,
    and market constraints (min/max size, precision, contract size). Returns Decimal size or None.
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
    if not (0 < risk_per_trade < 1):
        lg.error(f"Position sizing failed for {symbol}: Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1 (exclusive).")
        return None
    if initial_stop_loss_price is None or entry_price is None or entry_price <= 0:
        lg.error(f"Position sizing failed for {symbol}: Missing or invalid entry_price ({entry_price}) or initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed for {symbol}: Stop loss price ({initial_stop_loss_price}) cannot be the same as entry price ({entry_price}).")
        return None
    # Allow zero or negative stop loss price if it resulted from calculation, but warn heavily
    if initial_stop_loss_price <= 0:
        lg.warning(f"{NEON_YELLOW}Position sizing for {symbol}: Calculated initial stop loss price ({initial_stop_loss_price}) is zero or negative. Ensure SL calculation is correct. Proceeding cautiously.{RESET}")
        # Do not return None here, let the distance calculation handle it, but distance must be > 0

    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Position sizing failed for {symbol}: Market info missing 'limits' or 'precision'. Market: {market_info}")
        return None

    try:
        # --- Calculate Risk Amount and Initial Size ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)

        if sl_distance_per_unit <= 0:
            lg.error(f"Position sizing failed for {symbol}: Stop loss distance is zero or negative ({sl_distance_per_unit}). Entry={entry_price}, SL={initial_stop_loss_price}")
            return None

        # Get contract size (usually 1 for linear/spot, can vary for inverse/options)
        contract_size_str = market_info.get('contractSize', '1')
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0: raise ValueError("Contract size must be positive")
        except Exception:
            lg.warning(f"{NEON_YELLOW}Could not parse contract size '{contract_size_str}' for {symbol}. Defaulting to 1.{RESET}")
            contract_size = Decimal('1')


        # --- Calculate Size based on Market Type ---
        # For Linear contracts (Base/Quote, e.g., BTC/USDT) or Spot:
        # Size (in Base units or Contracts) = Risk Amount (Quote) / (SL Distance per Unit (Quote/Base) * Contract Size (Base/Contract))
        # For Inverse contracts (Quote/Base, e.g., USD/BTC -> BTC value per contract):
        # **WARNING**: Inverse calculation below is a simplified placeholder and likely needs exchange-specific adjustment.
        # It assumes `contractSize` represents the QUOTE value per contract (e.g., 1 for 1 USD).
        calculated_size = Decimal('0')
        if market_info.get('linear', True) or not is_contract: # Treat spot as linear type
            # Denominator represents risk per unit (base/contract) in quote currency
            risk_per_unit_quote = sl_distance_per_unit * contract_size
            if risk_per_unit_quote <= 0:
                lg.error(f"Position sizing failed for {symbol}: Risk per unit is zero or negative ({risk_per_unit_quote}). Check SL distance or contract size.")
                return None
            calculated_size = risk_amount_quote / risk_per_unit_quote
            lg.debug(f"  Linear/Spot Sizing: RiskAmt={risk_amount_quote:.4f} / (SLDist={sl_distance_per_unit} * ContSize={contract_size}) = {calculated_size}")
        else: # Potential Inverse contract logic
            lg.warning(f"{NEON_YELLOW}Inverse contract detected for {symbol}. Sizing calculation assumes `contractSize` ({contract_size}) is the Quote value per contract. VERIFY THIS LOGIC FOR {exchange.id}!{RESET}")
            # Formula: Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
            # Risk per Contract (Quote) = SL Distance (Quote/Base) * Contract Value (Base/Contract)
            # Contract Value (Base/Contract) = Contract Value (Quote/Contract) / Entry Price (Quote/Base)
            # Here, assume Contract Value (Quote/Contract) IS the `contract_size` field from market_info.
            contract_value_quote = contract_size
            if entry_price > 0:
                 # Risk per contract in quote currency = size_of_1_contract_in_base * sl_distance_in_quote
                 # size_of_1_contract_in_base = contract_value_quote / entry_price
                 size_of_1_contract_in_base = contract_value_quote / entry_price
                 risk_per_contract_quote = size_of_1_contract_in_base * sl_distance_per_unit

                 if risk_per_contract_quote > 0:
                     calculated_size = risk_amount_quote / risk_per_contract_quote
                     lg.debug(f"  Inverse Sizing: Size1Base={size_of_1_contract_in_base}, RiskPerContQuote={risk_per_contract_quote}")
                     lg.debug(f"  Inverse Sizing: RiskAmt={risk_amount_quote:.4f} / RiskPerContQuote={risk_per_contract_quote} = {calculated_size}")
                 else:
                     lg.error(f"Position sizing failed for inverse contract {symbol}: Risk per contract calculation is zero or negative ({risk_per_contract_quote}).")
                     return None
            else:
                 lg.error(f"Position sizing failed for inverse contract {symbol}: Entry price is zero.")
                 return None


        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, Risk Amount={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Distance={sl_distance_per_unit}")
        lg.info(f"  Contract Size={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}")


        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {}) # Cost = size * price
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount') # Can be int (decimals) or float (step size)

        # Get min/max amount limits (size limits)
        min_amount_str = amount_limits.get('min')
        max_amount_str = amount_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')

        # Get min/max cost limits (value limits in quote currency)
        min_cost_str = cost_limits.get('min')
        max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')


        # 1. Adjust size based on MIN/MAX AMOUNT limits
        adjusted_size = calculated_size
        if adjusted_size < min_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} {size_unit} is below minimum amount {min_amount:.8f} {size_unit}. Adjusting to minimum.{RESET}")
            adjusted_size = min_amount
        elif adjusted_size > max_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} {size_unit} exceeds maximum amount {max_amount:.8f} {size_unit}. Capping at maximum.{RESET}")
            adjusted_size = max_amount

        # 2. Check COST limits with the amount-adjusted size
        # Cost = Size * Entry Price * Contract Size (Contract size handles units -> quote value for linear)
        # For inverse, Cost = Size (Contracts) * Contract Value (Quote/Contract)
        current_cost = Decimal('0')
        if market_info.get('linear', True) or not is_contract:
            current_cost = adjusted_size * entry_price * contract_size
        else: # Inverse Cost (Size in Contracts * Contract Value in Quote)
            contract_value_quote = contract_size # Assumes contractSize holds the quote value
            current_cost = adjusted_size * contract_value_quote
            lg.debug(f"  Inverse Cost Calculation: Size={adjusted_size} * ContractValueQuote={contract_value_quote} = {current_cost}")


        lg.debug(f"  Cost Check: Amount-Adjusted Size={adjusted_size:.8f} {size_unit}, Estimated Cost={current_cost:.4f} {quote_currency}")
        lg.debug(f"  Cost Limits: Min={min_cost}, Max={max_cost}")

        if min_cost > 0 and current_cost < min_cost :
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} {quote_currency} (Size: {adjusted_size:.8f}) is below minimum cost {min_cost:.4f} {quote_currency}. Attempting to increase size.{RESET}")
            # Calculate required size to meet min cost
            required_size_for_min_cost = Decimal('0')
            if market_info.get('linear', True) or not is_contract:
                if entry_price > 0 and contract_size > 0:
                    required_size_for_min_cost = min_cost / (entry_price * contract_size)
                else:
                    lg.error("Cannot calculate size for min cost (linear) due to invalid entry price or contract size.")
                    return None
            else: # Inverse
                contract_value_quote = contract_size # Use contract_size as value
                if contract_value_quote > 0:
                     required_size_for_min_cost = min_cost / contract_value_quote
                else:
                     lg.error("Cannot calculate size for min cost (inverse) due to invalid contract value.")
                     return None

            lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f} {size_unit}")

            # Check if this required size is feasible (within amount limits)
            if required_size_for_min_cost > max_amount:
                lg.error(f"{NEON_RED}Cannot meet minimum cost {min_cost:.4f} without exceeding maximum amount {max_amount:.8f}. Trade aborted.{RESET}")
                return None
            # Check if required size is now below min_amount (conflicting limits)
            # Use a small tolerance here
            elif required_size_for_min_cost < min_amount and not math.isclose(float(required_size_for_min_cost), float(min_amount), rel_tol=1e-9):
                 lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f}, but min amount is {min_amount:.8f}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f} {size_unit}")
                 adjusted_size = required_size_for_min_cost
                 # Re-calculate cost for logging/final checks
                 if market_info.get('linear', True) or not is_contract:
                    current_cost = adjusted_size * entry_price * contract_size
                 else: # Inverse
                    contract_value_quote = contract_size # Use contract_size as value
                    current_cost = adjusted_size * contract_value_quote

        elif max_cost > 0 and current_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} {quote_currency} exceeds maximum cost {max_cost:.4f} {quote_currency}. Reducing size.{RESET}")
            adjusted_size_for_max_cost = Decimal('0')
            if market_info.get('linear', True) or not is_contract:
                if entry_price > 0 and contract_size > 0:
                    adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                else:
                    lg.error("Cannot calculate size for max cost (linear) due to invalid entry price or contract size.")
                    return None
            else: # Inverse
                contract_value_quote = contract_size # Use contract_size as value
                if contract_value_quote > 0:
                    adjusted_size_for_max_cost = max_cost / contract_value_quote
                else:
                    lg.error("Cannot calculate size for max cost (inverse) due to invalid contract value.")
                    return None


            lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f} {size_unit}")

            # Check if this reduced size is now below min amount
            if adjusted_size_for_max_cost < min_amount and not math.isclose(float(adjusted_size_for_max_cost), float(min_amount), rel_tol=1e-9):
                lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is now below minimum amount {min_amount:.8f}. Cannot meet conflicting limits. Trade aborted.{RESET}")
                return None
            else:
                adjusted_size = adjusted_size_for_max_cost


        # 3. Apply Amount Precision/Step Size (using ccxt helper ideally, rounding DOWN)
        final_size = Decimal('0')
        try:
            # amount_to_precision usually expects a float input
            # Use exchange.amount_to_precision with TRUNCATE (equivalent to floor/ROUND_DOWN for positive numbers)
            # to ensure we don't exceed balance/risk due to rounding up size.
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit} using exchange helper.")
        except ccxt.ExchangeError as fmt_err:
            # Handle cases where ccxt doesn't have precision info properly
            lg.warning(f"{NEON_YELLOW}ccxt.ExchangeError using amount_to_precision for {symbol}: {fmt_err}. Using manual rounding (ROUND_DOWN).{RESET}")
            # Fallback to manual rounding based on precision value (int decimals or float step)
            final_size = _apply_manual_precision(adjusted_size, amount_precision_val, ROUND_DOWN, lg, symbol, "amount")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision for {symbol} ({fmt_err}). Using manual rounding (ROUND_DOWN).{RESET}")
            # Fallback to manual rounding based on precision value (int decimals or float step)
            final_size = _apply_manual_precision(adjusted_size, amount_precision_val, ROUND_DOWN, lg, symbol, "amount")


        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{NEON_RED}Position size became zero or negative ({final_size}) after adjustments/rounding for {symbol}. Trade aborted.{RESET}")
            return None

        # Final check against min amount AFTER formatting/rounding
        if final_size < min_amount:
            # Use a small tolerance to account for potential float/decimal precision issues during comparison
            if not math.isclose(float(final_size), float(min_amount), rel_tol=1e-9):
                 lg.error(f"{NEON_RED}Final formatted size {final_size} {size_unit} is below minimum amount {min_amount} {size_unit} for {symbol}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}Final formatted size {final_size} is extremely close to min amount {min_amount}. Proceeding cautiously.{RESET}")


        # Final check against min cost AFTER formatting/rounding
        final_cost = Decimal('0')
        if market_info.get('linear', True) or not is_contract:
            final_cost = final_size * entry_price * contract_size
        else: # Inverse placeholder
            contract_value_quote = contract_size # Use contract_size as value
            final_cost = final_size * contract_value_quote

        if min_cost > 0 and final_cost < min_cost:
             # Check if it's close enough due to rounding? Use tolerance.
            if not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6):
                 lg.error(f"{NEON_RED}Final formatted size {final_size} {size_unit} results in cost {final_cost:.4f} {quote_currency} which is below minimum cost {min_cost:.4f} {quote_currency} for {symbol}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}Final cost {final_cost} is very close to min cost {min_cost}. Proceeding cautiously.{RESET}")


        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(f"{NEON_RED}Position sizing error for {symbol}: Missing market info key {e}. Market: {market_info}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None

def _apply_manual_precision(value: Decimal, precision_val: Any, rounding_mode: str, lg: logging.Logger, symbol: str, value_type: str) -> Decimal:
    """Helper to apply manual precision/step size rounding."""
    if precision_val is not None:
        num_decimals = None
        step_size = None
        if isinstance(precision_val, int):
            num_decimals = precision_val
        elif isinstance(precision_val, (float, str)): # Assume it's step size
            try:
                step_size = Decimal(str(precision_val))
                if step_size <= 0: step_size = None # Invalidate non-positive step size
                else:
                    # Infer decimals from step size for quantization logging
                    num_decimals = abs(step_size.normalize().as_tuple().exponent)
            except Exception: pass # Ignore error if step size is invalid

        if step_size is not None and step_size > 0:
            # Round down/up to the nearest step size increment
            if rounding_mode == ROUND_DOWN:
                rounded_value = (value // step_size) * step_size
            else: # ROUND_UP
                # Add a small epsilon before dividing for ceiling effect with integer division
                rounded_value = ((value + step_size - Decimal('1e-18')) // step_size) * step_size
            lg.info(f"Applied manual {value_type} step size ({step_size}), rounded {rounding_mode}: {value:.8f} -> {rounded_value} {symbol}")
            return rounded_value
        elif num_decimals is not None and num_decimals >= 0:
            rounding_factor = Decimal('1e-' + str(num_decimals))
            rounded_value = value.quantize(rounding_factor, rounding=rounding_mode)
            lg.info(f"Applied manual {value_type} precision ({num_decimals} decimals), rounded {rounding_mode}: {value:.8f} -> {rounded_value} {symbol}")
            return rounded_value
        else:
            lg.warning(f"{NEON_YELLOW}{value_type.capitalize()} precision value ('{precision_val}') invalid for {symbol}. Using unrounded value: {value:.8f}{RESET}")
            return value
    else:
        lg.warning(f"{NEON_YELLOW}{value_type.capitalize()} precision not defined for {symbol}. Using unrounded value: {value:.8f}{RESET}")
        return value

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using fetch_positions.
    Returns the unified position dictionary from CCXT if an active position exists, otherwise None.
    Handles variations in position reporting (size > 0, side field, different API versions).
    Enhances the returned dict with SL/TP/TSL info parsed from the 'info' dict if available.
    Adds 'trailingStopLossDistanceDecimal' and 'tslActivationPriceDecimal' for easier checks.
    """
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        market = None # Fetch market info for context

        # Bybit V5: fetch_positions usually requires the symbol and category
        fetch_all = False
        try:
            # Try fetching specific symbol first
            params = {}
            if 'bybit' in exchange.id.lower():
                 try:
                     market = get_market_info(exchange, symbol, lg) # Use helper to ensure markets loaded
                     if market and market.get('is_contract', False):
                         category = 'linear' if market.get('linear', True) else 'inverse'
                         params['category'] = category
                         lg.debug(f"Using params for fetch_positions: {params}")
                     elif market and not market.get('is_contract', False):
                          lg.debug(f"Symbol {symbol} is Spot, not setting category for fetch_positions.")
                     else:
                         lg.warning(f"Could not get market info for {symbol} within get_open_position. Params may be incomplete.")
                 except Exception as e:
                     lg.warning(f"Error fetching market info for {symbol} within get_open_position: {e}. Assuming 'linear' for category if contract.")
                     # Attempt to guess category if market fetch failed but symbol looks like a derivative
                     if ':' in symbol or 'PERP' in symbol:
                         params['category'] = 'linear' # Default assumption

            # Fetch positions for the specific symbol(s)
            # Use a slightly higher timeout if possible? Default is used if not specified
            positions = exchange.fetch_positions(symbols=[symbol], params=params)

        except ccxt.ArgumentsRequired:
            lg.warning(f"Exchange {exchange.id} requires fetching all positions, then filtering. Fetching all...")
            fetch_all = True
        except ccxt.ExchangeError as e:
            # Handle errors like "symbol not found" if fetching single symbol failed
            if "symbol not found" in str(e).lower() or "instrument not found" in str(e).lower():
                lg.warning(f"Symbol {symbol} not found when fetching position: {e}. Assuming no position.")
                return None
            # Handle Bybit V5 specific "no position found" error codes if fetching single
            no_pos_codes_v5 = [110025, 110021] # Position idx not match / Position is closed or not exist
            bybit_code = getattr(e, 'code', None)
            if isinstance(bybit_code, str):
                try: bybit_code = int(bybit_code) # Convert code to int if string
                except ValueError: pass # Keep as string if not integer

            if bybit_code in no_pos_codes_v5:
                lg.info(f"No position found for {symbol} (Exchange error code: {bybit_code} - {e}).")
                return None
            # Handle "Invalid symbol" type errors
            if "invalid symbol" in str(e).lower():
                lg.error(f"{NEON_RED}Invalid symbol '{symbol}' when fetching position: {e}.{RESET}")
                return None
            # Re-raise other exchange errors during single fetch
            lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=True)
            # Fallback to fetch_all might be risky, return None here.
            return None
        except Exception as e:
            lg.error(f"Error fetching single position for {symbol}: {e}", exc_info=True)
            # Fallback to fetching all? Assume failure for now.
            return None

        # Fetch all positions if required or as fallback
        if fetch_all:
            try:
                # Fetch all positions without symbol/params
                all_positions = exchange.fetch_positions()
                # Filter for the specific symbol
                positions = [p for p in all_positions if p.get('symbol') == symbol]
                lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
            except Exception as e:
                lg.error(f"Error fetching all positions for {symbol}: {e}", exc_info=True)
                return None # Cannot determine position state if fetch_all fails


        # --- Process the fetched positions (should be 0 or 1 for the symbol in One-Way mode) ---
        active_position = None
        for pos in positions:
            # Standardized field for size is 'contracts' (float or string)
            # Fallbacks: 'contractSize' (less common), 'info' dict specifics
            pos_size_str = None
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            # elif pos.get('contractSize') is not None: pos_size_str = str(pos['contractSize']) # Less reliable, might be 1
            elif pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size']) # Bybit V5 'size'
            elif pos.get('info', {}).get('positionAmt') is not None: pos_size_str = str(pos['info']['positionAmt']) # Binance 'positionAmt'

            if pos_size_str is None:
                lg.debug(f"Could not find position size field in position data for {symbol}: {pos}")
                continue # Skip this entry if size cannot be determined

            # Check if size is valid and meaningfully non-zero
            try:
                position_size = Decimal(pos_size_str)
                # Use a small threshold based on minimum contract size if available, else small number
                min_size_threshold = Decimal('1e-9') # Default threshold
                if market: # Use market info fetched earlier if available
                    try:
                        min_amt = market.get('limits', {}).get('amount', {}).get('min')
                        if min_amt is not None:
                            # Use a fraction of min size, ensuring it's not zero
                            min_size_threshold = max(min_size_threshold, Decimal(str(min_amt)) * Decimal('0.01'))
                    except Exception: pass # Ignore errors fetching min size

                if abs(position_size) > min_size_threshold:
                    # Found an active position
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                    break # Stop after finding the first active position entry
            except Exception as e:
                lg.warning(f"{NEON_YELLOW}Could not parse position size '{pos_size_str}' as Decimal for {symbol}. Skipping entry. Error: {e}{RESET}")
                continue

        # --- Post-Process the found active position (if any) ---
        if active_position:
            # --- Determine Side ---
            side = active_position.get('side') # Standard 'long' or 'short'
            size_decimal = Decimal('0')
            try:
                # Try to get size from standard field or info dict again for side inference
                size_str_for_side = active_position.get('contracts', active_position.get('info',{}).get('size', '0'))
                size_decimal = Decimal(str(size_str_for_side))
            except Exception: pass # Ignore errors, fallback below

            size_threshold = Decimal('1e-9') # Re-define here for clarity

            # Infer side from size sign if 'side' field is missing, 'none', or ambiguous
            # Bybit V5: 'side' in info dict can be 'Buy', 'Sell', or 'None'
            if side not in ['long', 'short']:
                info_side = active_position.get('info', {}).get('side', 'None')
                if info_side == 'Buy': side = 'long'
                elif info_side == 'Sell': side = 'short'
                # If still not determined, use size sign
                elif size_decimal > size_threshold: # Use threshold again
                    side = 'long'
                    lg.debug(f"Inferred side as 'long' from positive size {size_decimal} for {symbol}.")
                elif size_decimal < -size_threshold: # Use threshold again
                    side = 'short'
                    lg.debug(f"Inferred side as 'short' from negative size {size_decimal} for {symbol}.")
                    # CCXT often stores the absolute size in 'contracts' even for short.
                    # Ensure 'contracts' field holds the absolute value if we modify it.
                    # Bybit V5 'size' is usually positive, side is in 'side' field.
                    if 'contracts' in active_position and active_position['contracts'] is not None:
                        try:
                            # Store absolute value in 'contracts', keep original size in 'signed_contracts' maybe?
                            original_contracts_val = Decimal(str(active_position['contracts']))
                            active_position['contracts'] = abs(original_contracts_val) # Standardize 'contracts' field to positive
                            # active_position['signed_contracts'] = original_contracts_val # Optional: Store signed value if needed elsewhere
                        except Exception: pass # Ignore conversion errors
                else:
                    # This case should ideally not happen if size check passed, but handle defensively
                    lg.warning(f"{NEON_YELLOW}Position size {size_decimal} is close to zero, could not determine side for {symbol}. Treating as no position.{RESET}")
                    return None # Treat as no position if side cannot be reliably determined

                # Add inferred side back to the dictionary for consistency
                active_position['side'] = side

            # Ensure 'contracts' field is positive float/Decimal if it exists and side is determined
            if active_position.get('contracts') is not None:
                 try:
                      current_contracts = Decimal(str(active_position['contracts']))
                      if current_contracts < 0:
                           active_position['contracts'] = abs(current_contracts)
                           lg.debug(f"Standardized 'contracts' field to positive value: {active_position['contracts']}")
                 except Exception: pass # Ignore if conversion fails


            # --- Enhance with SL/TP/TSL info directly if available ---
            # CCXT aims to standardize 'stopLossPrice', 'takeProfitPrice'
            # Check 'info' dict for exchange-specific fields (e.g., Bybit V5)
            info_dict = active_position.get('info', {})

            # Helper to extract and validate price from info dict, returning Decimal or None
            def get_valid_decimal_price_from_info(key: str) -> Optional[Decimal]:
                val_str = info_dict.get(key)
                # Bybit uses '0' or '0.0' for inactive, check non-empty and not just '0' variants
                if val_str and str(val_str).strip() and Decimal(str(val_str).strip()) != 0:
                    try:
                        price_dec = Decimal(str(val_str).strip())
                        if price_dec > 0:
                            return price_dec
                    except Exception: pass # Ignore conversion errors
                return None

            # Populate standard fields if missing, using validated values from info
            if active_position.get('stopLossPrice') is None or float(active_position.get('stopLossPrice', 0)) <= 0:
                sl_val_dec = get_valid_decimal_price_from_info('stopLoss')
                if sl_val_dec: active_position['stopLossPrice'] = float(sl_val_dec) # CCXT expects float

            if active_position.get('takeProfitPrice') is None or float(active_position.get('takeProfitPrice', 0)) <= 0:
                tp_val_dec = get_valid_decimal_price_from_info('takeProfit')
                if tp_val_dec: active_position['takeProfitPrice'] = float(tp_val_dec) # CCXT expects float

            # Add TSL info if available - distance/value and activation price
            # Bybit V5: 'trailingStop' is the distance/value, 'activePrice' is TSL activation price
            # Store them even if '0' (inactive) for clarity
            tsl_dist_str = info_dict.get('trailingStop', '0')
            tsl_act_str = info_dict.get('activePrice', '0')
            active_position['trailingStopLossDistance'] = tsl_dist_str # Store raw string
            active_position['tslActivationPrice'] = tsl_act_str # Store raw string

            # Add parsed Decimal versions for easier checks later
            try:
                 # Ensure distance is non-negative
                 tsl_dist_val = Decimal(tsl_dist_str) if tsl_dist_str and str(tsl_dist_str).strip() else Decimal('0')
                 active_position['trailingStopLossDistanceDecimal'] = max(Decimal(0), tsl_dist_val)
            except Exception:
                 active_position['trailingStopLossDistanceDecimal'] = Decimal('0')
            try:
                 # Activation can be 0, ensure non-negative
                 tsl_act_val = Decimal(tsl_act_str) if tsl_act_str and str(tsl_act_str).strip() else Decimal('0')
                 active_position['tslActivationPriceDecimal'] = max(Decimal(0), tsl_act_val)
            except Exception:
                 active_position['tslActivationPriceDecimal'] = Decimal('0')


            # Helper function to format price/value for logging, handling None, '0', errors
            def format_log_value(key: str, value: Union[str, float, Decimal, None], precision: int = 6) -> str:
                if value is None or str(value).strip() == '': return 'N/A'
                try:
                    d_value = Decimal(str(value))
                    # Allow zero only for specific fields like TSL distance/activation if inactive
                    if d_value > 0:
                         # Format with dynamic precision
                         return f"{d_value:.{precision}f}"
                    elif d_value == 0 and ('trailingstop' in key.lower() or 'tslactivation' in key.lower()):
                         return '0.0 (Inactive)' # Show inactive TSL as 0.0
                    elif d_value == 0:
                         # For other fields like entry, SL, TP, size, 0 is usually invalid or means N/A
                         return 'N/A (Zero)'
                    else: # Negative values
                        # For PnL, format negative. For Size/Contracts, format absolute. For others, treat as invalid?
                        if 'pnl' in key.lower():
                             return f"{d_value:.{precision}f}"
                        elif key.lower() in ['size', 'contracts']:
                            # Use amount precision for size
                            amount_precision = 8 # Default
                            if market:
                                try:
                                    ap_val = market.get('precision',{}).get('amount')
                                    if isinstance(ap_val, int): amount_precision = ap_val
                                    elif isinstance(ap_val, (float, str)):
                                        amount_precision = abs(Decimal(str(ap_val)).normalize().as_tuple().exponent)
                                except Exception: pass
                            return f"{abs(d_value):.{amount_precision}f}" # Show absolute size
                        else:
                             return f'Invalid ({d_value})'
                except Exception:
                    return str(value) # Return raw if conversion fails

            # Re-fetch market info if it wasn't available earlier
            if not market:
                 try:
                     market = get_market_info(exchange, symbol, lg)
                 except Exception: pass # Ignore errors, use defaults

            log_precision = 6 # Default precision for logging prices
            amount_precision = 8 # Default precision for logging amounts/sizes
            if market:
                # Try to get price precision from market info
                try:
                    # Create temp analyzer to use helper
                    dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                    dummy_df.index.name = 'timestamp'
                    analyzer_temp = TradingAnalyzer(dummy_df, lg, {}, market)
                    log_precision = analyzer_temp.get_price_precision()
                except Exception: pass
                # Try to get amount precision from market info
                try:
                    ap_val = market.get('precision', {}).get('amount')
                    if isinstance(ap_val, int): amount_precision = ap_val
                    elif isinstance(ap_val, (float, str)):
                         amount_precision = abs(Decimal(str(ap_val)).normalize().as_tuple().exponent)
                except Exception: pass


            entry_price_val = active_position.get('entryPrice', info_dict.get('avgPrice'))
            entry_price = format_log_value('entryPrice', entry_price_val, log_precision)
            contracts_val = active_position.get('contracts', info_dict.get('size'))
            # Use amount precision for contracts log
            contracts = format_log_value('contracts', contracts_val, amount_precision)
            liq_price = format_log_value('liquidationPrice', active_position.get('liquidationPrice'), log_precision)
            leverage_str = active_position.get('leverage', info_dict.get('leverage'))
            leverage = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str is not None else 'N/A'
            pnl_val = active_position.get('unrealizedPnl')
            pnl = format_log_value('unrealizedPnl', pnl_val, 4) # PnL precision usually lower
            sl_price_val = active_position.get('stopLossPrice') # Use potentially updated value
            tp_price_val = active_position.get('takeProfitPrice') # Use potentially updated value
            sl_price = format_log_value('stopLossPrice', sl_price_val, log_precision)
            tp_price = format_log_value('takeProfitPrice', tp_price_val, log_precision)
            # Use the parsed Decimal for check, format the string for log
            is_tsl_active_log = active_position.get('trailingStopLossDistanceDecimal', Decimal(0)) > 0
            tsl_dist_log = format_log_value('trailingStopLossDistance', active_position.get('trailingStopLossDistance'), log_precision)
            tsl_act_log = format_log_value('tslActivationPrice', active_position.get('tslActivationPrice'), log_precision)


            logger.info(f"{NEON_GREEN}Active {side.upper()} position found for {symbol}:{RESET} "
                        f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                        f"Leverage={leverage}, PnL={pnl}, SL={sl_price}, TP={tp_price}, "
                        f"TSL Active: {is_tsl_active_log} (Dist={tsl_dist_log}/Act={tsl_act_log})")
            logger.debug(f"Full position details for {symbol}: {active_position}")
            return active_position
        else:
            # If loop completes without finding a non-zero position
            logger.info(f"No active open position found for {symbol}.")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Handle errors that might indicate no position gracefully (e.g., Bybit V5 retCode != 0)
        no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
        no_pos_codes = [110025, 110021] # Bybit V5: Position not found / is closed
        err_str = str(e).lower()
        err_code = getattr(e, 'code', None)
        if isinstance(err_code, str):
            try: err_code = int(err_code)
            except ValueError: pass

        if any(msg in err_str for msg in no_pos_msgs) or (err_code in no_pos_codes):
            lg.info(f"No open position found for {symbol} (Confirmed by exchange error: {e}).")
            return None
        # Bybit V5 might return success (retCode 0) but an empty list if no position - handled by the loop logic above.
        lg.error(f"{NEON_RED}Unhandled Exchange error fetching positions for {symbol}: {e} (Code: {err_code}){RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)

    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics and verification hints."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # Success if not applicable

    if leverage <= 0:
        lg.warning(f"{NEON_YELLOW}Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be > 0.{RESET}")
        return False

    # --- Check exchange capability ---
    if not exchange.has.get('setLeverage'):
        lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage method via CCXT capabilities.{RESET}")
        # TODO: Could implement direct API call here if needed as a fallback.
        return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")

        # --- Prepare Bybit V5 specific parameters (if applicable) ---
        # Bybit V5 often requires setting buy and sell leverage, especially for Isolated Margin.
        # For Cross Margin, setting just one might suffice, but setting both is safer.
        # Assume we need to set both for robustness unless we know margin mode.
        params = {}
        if 'bybit' in exchange.id.lower():
            # Fetch current margin mode if possible to avoid errors
            margin_mode = None
            trade_mode_val = None # 0: Cross, 1: Isolated
            try:
                # Using fetch_position for margin mode info (might be in 'info')
                # Need to handle case where there's NO position yet
                # We can try fetch_position, but it might fail if no position exists
                # An alternative is a dedicated endpoint if available, or rely on user setting.
                # Let's try fetch_positions first, as get_open_position might not be called yet
                pos_check_params = {'category': 'linear' if market_info.get('linear', True) else 'inverse'}
                positions = exchange.fetch_positions(symbols=[symbol], params=pos_check_params)
                if positions:
                    pos_info = positions[0] # Assume one-way mode, take first entry
                    margin_mode = pos_info.get('marginMode') # Standard ccxt field
                    if margin_mode is None:
                        trade_mode_val = pos_info.get('info', {}).get('tradeMode') # Bybit V5 'tradeMode' (0: Cross, 1: Isolated)
                        if trade_mode_val == 0: margin_mode = 'cross'
                        elif trade_mode_val == 1: margin_mode = 'isolated'
                    lg.debug(f"Detected margin mode from existing position for {symbol}: {margin_mode} (TradeMode: {trade_mode_val})")
                else:
                    # If no position, we can't reliably get the *symbol's* current margin mode.
                    # We might need to call a specific endpoint like /v5/account/info if ccxt doesn't provide it easily.
                    # For now, assume the user has set it correctly on the exchange.
                    lg.debug(f"No existing position for {symbol}, cannot detect margin mode automatically. Assuming Isolated/Cross based on account default.")

            except ccxt.ExchangeError as e:
                 # Ignore "no position" errors
                 no_pos_codes_v5 = [110025, 110021]
                 bybit_code = getattr(e, 'code', None)
                 if isinstance(bybit_code, str):
                     try: bybit_code = int(bybit_code)
                     except ValueError: pass # Keep as string if not integer
                 if bybit_code in no_pos_codes_v5:
                      lg.debug(f"No existing position found for {symbol} when checking margin mode.")
                 else:
                      lg.warning(f"Exchange error checking margin mode for {symbol}: {e}. Proceeding cautiously.")
            except Exception as e:
                lg.warning(f"Could not reliably fetch margin mode for {symbol} before setting leverage: {e}. Proceeding cautiously.")

            # Bybit V5 set_leverage endpoint requires buy/sell for both modes now.
            # Use string representation for leverage in params for Bybit.
            params = {
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage),
                # 'positionIdx': 0 # Not needed for setLeverage endpoint usually
            }
            # Category is needed
            category = 'linear' if market_info.get('linear', True) else 'inverse'
            params['category'] = category
            lg.debug(f"Using Bybit V5 specific params for set_leverage: {params}")


        # --- Call set_leverage ---
        # The `leverage` argument is the primary value, `params` provides extra details
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)

        # Log response for debugging, but success isn't guaranteed by non-exception response
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # --- Verification ---
        # A successful response (no exception, or specific success code/message) implies success.
        # Bybit V5 response might contain retCode=0 on success.
        verified = False
        if response is not None:
            if isinstance(response, dict):
                 # Check Bybit V5 specific response structure
                 ret_code = response.get('retCode', response.get('info', {}).get('retCode')) # Check top level or nested info
                 if isinstance(ret_code, str):
                     try: ret_code = int(ret_code) # Convert code to int if string
                     except ValueError: pass # Keep as string if not integer

                 if ret_code == 0:
                     lg.debug(f"Set leverage call for {symbol} confirmed success (retCode 0).")
                     verified = True
                 elif ret_code is not None: # Got a non-zero retCode
                      # Check for "leverage not modified" code
                      if ret_code == 110045:
                          lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation: Code {ret_code}).{RESET}")
                          verified = True # Treat as success
                      else:
                          ret_msg = response.get('retMsg', response.get('info', {}).get('retMsg', 'Unknown Error'))
                          lg.warning(f"{NEON_YELLOW}Set leverage call for {symbol} returned non-zero retCode {ret_code} ({ret_msg}). Treating as failure.{RESET}")
                          verified = False # Treat as failure if non-zero code returned
                 else:
                      # No retCode found, rely on lack of exception (less reliable)
                      lg.debug(f"Set leverage call for {symbol} returned a response (no retCode found). Assuming success if no error.")
                      verified = True
            else:
                 # Response is not a dict, rely on lack of exception
                 lg.debug(f"Set leverage call for {symbol} returned non-dict response. Assuming success if no error.")
                 verified = True
        else:
            # If response is None, it *might* still be okay, but less certain.
            # Bybit V5 setLeverage might return None on success via ccxt sometimes. Rely on no exception.
            lg.debug(f"Set leverage call for {symbol} returned None. Assuming success as no error was raised.")
            verified = True # Tentative


        if verified:
            lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Leverage setting failed for {symbol} based on response analysis.{RESET}")
            return False


    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None) # CCXT often maps Bybit retCode to e.code
        if isinstance(bybit_code, str):
             try: bybit_code = int(bybit_code)
             except ValueError: pass # Keep as string if not integer

        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")

        # --- Handle Common Bybit V5 Leverage Errors ---
        # Code 110045: Leverage not modified (already set to the target value)
        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation: Code {bybit_code}).{RESET}")
            return True # Treat as success

        # Code 110028: Need to set margin mode first (e.g., switch to Isolated)
        elif bybit_code == 110
