# livexy.py
# Enhanced version focusing on stop-loss/take-profit mechanisms, including break-even and trailing stops.
~/worldguide/codes main* 7m 21s ❯ python liveexr.py
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/codes/liveexr.py", line 3418, in <module>
    main()
  File "/data/data/com.termux/files/home/worldguide/codes/liveexr.py", line 3223, in main
    try: pandas_ta_version = ta.version()
                             ^^^^^^^^^^^^
TypeError: 'str' object is not callable 
evel based on global variable
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
                'brokerId': 'livebot71', # Example: Add a broker ID for Bybit if desired
                # Consider explicitly setting V5 API version if issues persist with defaults
                # Explicitly request V5 API
                'default_options': {
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                    'recvWindow': 10000, # Optional: Set recvWindow
                    'fetchPositions': 'v5', # Request V5 for fetchPositions
                    'fetchBalance': 'v5',   # Request V5 for fetchBalance
                    'createOrder': 'v5',    # Request V5 for createOrder
                    'fetchOrder': 'v5',     # Request V5 for fetchOrder
                    # Add other endpoints as needed
                    'setLeverage': 'v5',
                    'private_post_v5_position_trading_stop': 'v5', # Ensure protection uses V5
                },
                'accounts': { # Define V5 account types if needed by ccxt version
                    'future': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'swap': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'option': {'unified': 'OPTION'},
                    'spot': {'unified': 'SPOT'},
                },
            }
        }

        # Select Bybit class
        exchange_class = getattr(ccxt, 'bybit') # Use getattr for robustness
        exchange = exchange_class(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        # Test connection by fetching markets (essential for market info)
        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"CCXT exchange
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
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
getcontext().prec = 28  # Sufficient precision for financial calculations
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
        "trailing_stop_callback_rate": 0.005,   # e.g., 0.5% trail distance (as decimal) from high water mark (or activation price)
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
            # Return default even if file creation failed, so the bot can try to run
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
            # Attempt to create a default file if loading failed badly
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file after load error: {filepath}{RESET}")
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
            # Recursively check nested dictionaries (like 'indicators' or 'weight_sets')
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Ensure leaf node values are of the correct type if possible (simple cases)
        elif type(default_value) != type(updated_config.get(key)):
            # Basic type correction (e.g., str to float/int if applicable, bool)
            try:
                if isinstance(default_value, bool): updated_config[key] = bool(updated_config[key])
                elif isinstance(default_value, int): updated_config[key] = int(updated_config[key])
                elif isinstance(default_value, float): updated_config[key] = float(updated_config[key])
                # Add more type conversions if needed
            except (ValueError, TypeError):
                print(f"{NEON_YELLOW}Config Warning: Type mismatch for key '{key}'. Using default value.{RESET}")
                updated_config[key] = default_value
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

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels in the logger itself

    # File Handler
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use SensitiveFormatter to redact keys in file logs too
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to console if file logging fails
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")

    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z' # Add timezone abbreviation
    )
    # Apply timezone to formatter's date formatting
    stream_formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
    file_formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple() # Also for file

    stream_handler.setFormatter(stream_formatter)
    console_log_level = logging.INFO # Default console level (change to DEBUG for verbose)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent log messages from propagating to the root logger
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
                 # Increased timeouts for potentially slower operations
                'fetchTickerTimeout': 15000, # 15 seconds
                'fetchOHLCVTimeout': 20000, # 20 seconds
                'fetchBalanceTimeout': 20000, # 20 seconds
                'createOrderTimeout': 30000, # 30 seconds
                'cancelOrderTimeout': 20000, # 20 seconds
                'fetchPositionsTimeout': 20000, # 20 seconds
                'fetchOrderBookTimeout': 15000, # 15 seconds
                'privatePostV5PositionSetTradingStopTimeout': 30000, # Specific for SL/TP/TSL
            }
        }

        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
             logger.warning(f"{NEON_RED}!!! USING REAL MONEY ENVIRONMENT !!!{RESET}")


        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys - crucial step
        account_type_to_test = 'CONTRACT' # Preferred for V5 derivatives
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            balance = exchange.fetch_balance(params={'type': account_type_to_test})
            # Try to find the quote currency balance for logging
            available_quote = 'N/A'
            if balance and QUOTE_CURRENCY in balance and balance[QUOTE_CURRENCY].get('free') is not None:
                 available_quote = f"{Decimal(str(balance[QUOTE_CURRENCY]['free'])):.4f}"
            # Check Bybit V5 structure if standard fails
            elif balance and 'info' in balance and 'result' in balance['info'] and isinstance(balance['info']['result'].get('list'), list):
                 for account in balance['info']['result']['list']:
                      if isinstance(account.get('coin'), list):
                           for coin_data in account['coin']:
                                if coin_data.get('coin') == QUOTE_CURRENCY:
                                     free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                     if free is not None: available_quote = f"{Decimal(str(free)):.4f}"; break
                           if available_quote != 'N/A': break

            logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")

        except ccxt.AuthenticationError as auth_err:
             logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade for Derivatives/Contracts), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
             return None
        except ccxt.ExchangeError as balance_err:
             logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
             try:
                  # Fallback without specific params
                  balance = exchange.fetch_balance()
                  available_quote = 'N/A'
                  if balance and QUOTE_CURRENCY in balance and balance[QUOTE_CURRENCY].get('free') is not None:
                       available_quote = f"{Decimal(str(balance[QUOTE_CURRENCY]['free'])):.4f}"
                  logger.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
             except Exception as fallback_err:
                  logger.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type.{RESET}")
        except Exception as balance_err:
             # Catch other potential issues like network errors during the test
             logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type and network.{RESET}")

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

            # Helper to safely convert to Decimal
            def safe_decimal(value):
                if value is None: return None
                try:
                    d = Decimal(str(value))
                    return d if d > 0 else None
                except (InvalidOperation, ValueError, TypeError):
                    lg.warning(f"Invalid price value encountered: {value}")
                    return None

            price = safe_decimal(last_price)
            if price is not None: lg.debug(f"Using 'last' price: {price}")

            if price is None:
                bid = safe_decimal(bid_price)
                ask = safe_decimal(ask_price)
                if bid is not None and ask is not None:
                    price = (bid + ask) / 2
                    lg.debug(f"Using bid/ask midpoint: {price}")
                elif ask is not None: # Fallback to ask
                    price = ask
                    lg.warning(f"Using 'ask' price fallback: {price}")
                elif bid is not None: # Fallback to bid
                    price = bid
                    lg.warning(f"Using 'bid' price fallback: {price}")

            if price is not None:
                return price
            else:
                lg.warning(f"Failed to get a valid price from ticker attempt {attempts + 1}.")
                # Continue to retry logic if price is None or invalid

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting longer...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 5) # Longer wait for rate limits
            attempts += 1 # Count this attempt
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Don't retry on most exchange errors (e.g., bad symbol) unless it's a known temporary issue
            if e.http_status in RETRY_ERROR_CODES:
                 lg.warning(f"{NEON_YELLOW}Retryable exchange error ({e.http_status}). Retrying...{RESET}")
            else:
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
                  # Add timeout parameter if supported by the exchange object's options
                  params = {}
                  if 'fetchOHLCVTimeout' in exchange.options:
                       params['requestTimeout'] = exchange.options['fetchOHLCVTimeout']

                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
                  if ohlcv is not None and len(ohlcv) > 0: # Basic check if data was returned
                    break # Success
                  else:
                    lg.warning(f"fetch_ohlcv returned empty or None for {symbol} (Attempt {attempt+1}). Retrying...")

             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                      raise e # Re-raise the last error
             except ccxt.RateLimitExceeded as e:
                 wait_time = max(RETRY_DELAY_SECONDS * 5, e.params.get('retry-after', RETRY_DELAY_SECONDS * 5)) # Use retry-after if available
                 lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                 time.sleep(wait_time) # Longer delay for rate limits
             except ccxt.ExchangeError as e:
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 if e.http_status in RETRY_ERROR_CODES and attempt < MAX_API_RETRIES:
                      lg.warning(f"{NEON_YELLOW}Retryable exchange error ({e.http_status}). Retrying...{RESET}")
                      time.sleep(RETRY_DELAY_SECONDS)
                 else:
                      raise e # Re-raise non-retryable or final exchange errors
             except Exception as e: # Catch broader exceptions during fetch
                  lg.error(f"{NEON_RED}Unexpected error during kline fetch for {symbol}: {e}{RESET}", exc_info=True)
                  if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                  else: raise e

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe}.{RESET}")
             return df

        # Data Cleaning and Type Conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric, coercing errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        # Drop rows with NaN in essential price columns or zero/negative close price
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0]
        # Optionally handle NaN volumes (e.g., fill with 0 or forward fill)
        df['volume'].fillna(0, inplace=True)

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price/timestamp data for {symbol}.")

        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{RESET}")
             return pd.DataFrame()

        df.sort_index(inplace=True) # Ensure chronological order
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching/processing klines for {symbol} after retries: {e}{RESET}")
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
            # Add timeout parameter if supported
            params = {}
            if 'fetchOrderBookTimeout' in exchange.options:
                 params['requestTimeout'] = exchange.options['fetchOrderBookTimeout']

            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            if not orderbook:
                lg.warning(f"fetch_order_book returned None/empty for {symbol} (Attempt {attempts+1}).")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                 lg.warning(f"{NEON_YELLOW}Invalid orderbook structure for {symbol}. Attempt {attempts + 1}. Response: {orderbook}{RESET}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 # This might be valid if the book is truly empty momentarily
                 lg.debug(f"Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).")
                 return orderbook # Return empty book
            else:
                 # Basic validation of price/amount format in first bid/ask
                 valid_structure = True
                 if orderbook['bids']:
                      if len(orderbook['bids'][0]) != 2 or not all(isinstance(x, (int, float)) for x in orderbook['bids'][0]):
                           valid_structure = False; lg.warning("Invalid bid structure")
                 if orderbook['asks'] and valid_structure:
                      if len(orderbook['asks'][0]) != 2 or not all(isinstance(x, (int, float)) for x in orderbook['asks'][0]):
                           valid_structure = False; lg.warning("Invalid ask structure")

                 if valid_structure:
                     lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                     return orderbook
                 else:
                      lg.warning(f"{NEON_YELLOW}Orderbook price/amount format seems invalid for {symbol}. Attempt {attempts + 1}.{RESET}")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = max(RETRY_DELAY_SECONDS * 5, e.params.get('retry-after', RETRY_DELAY_SECONDS * 5))
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempts += 1
            continue
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            if e.http_status in RETRY_ERROR_CODES:
                 lg.warning(f"{NEON_YELLOW}Retryable exchange error ({e.http_status}). Retrying...{RESET}")
            else:
                return None # Don't retry other exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry unexpected

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
        self.ta_column_names: Dict[str, Optional[str]] = {} # Stores actual column names

        if not self.weights:
             logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}.")
             # Use empty dict to prevent crashes, but log error
             self.weights = {}

        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Calculate Fib levels after indicators

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
        """Helper to find the actual column name generated by pandas_ta."""
        # More robust patterns based on common pandas_ta naming conventions
        expected_patterns = {
            "ATR": [f"ATRr_{self.config.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{self.config.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{self.config.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{self.config.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{self.config.get('cci_window', DEFAULT_CCI_WINDOW)}_"], # Often has constant suffix like _0.015
            "Williams_R": [f"WILLR_{self.config.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{self.config.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP_D"], # VWAP often has _D suffix if daily anchor
            "PSAR_long": [f"PSARl_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{self.config.get('psar_af', DEFAULT_PSAR_AF)}_{self.config.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{self.config.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}"], # Simpler pattern
            "StochRSI_D": [f"STOCHRSId_{self.config.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}"], # Simpler pattern
            "RSI": [f"RSI_{self.config.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [f"BBL_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_"], # Often has std suffix
            "BB_Middle": [f"BBM_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_"],
            "BB_Upper": [f"BBU_{self.config.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_"],
            "Volume_MA": [f"VOL_SMA_{self.config.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"] # Custom name
        }
        patterns = expected_patterns.get(base_name, [])
        if not patterns and base_name == "VWAP": patterns = ["VWAP"] # Fallback for VWAP if VWAP_D not found

        # Search dataframe columns for a match starting with any pattern
        df_cols = result_df.columns.tolist()
        for col in df_cols:
             for pattern in patterns:
                 if col.startswith(pattern):
                     self.logger.debug(f"Found column '{col}' for base '{base_name}' using pattern '{pattern}'.")
                     return col

        # Fallback: simple case-insensitive base name search (less reliable)
        base_lower = base_name.lower()
        for col in df_cols:
            if base_lower in col.lower():
                self.logger.debug(f"Found column '{col}' for base '{base_name}' using fallback search.")
                return col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {df_cols}")
        return None


    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check sufficient data length based on longest required period
        required_periods = []
        for key, default in {
            "atr_period": DEFAULT_ATR_PERIOD, "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
            "momentum_period": DEFAULT_MOMENTUM_PERIOD, "cci_window": DEFAULT_CCI_WINDOW,
            "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW, "mfi_window": DEFAULT_MFI_WINDOW,
            "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
            "rsi_period": DEFAULT_RSI_WINDOW, "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
            "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD, "sma_10_window": DEFAULT_SMA_10_WINDOW,
            "fibonacci_window": DEFAULT_FIB_WINDOW
        }.items():
             # Check if indicator is actually enabled before considering its period
             indicator_name = key.split('_')[0] # Simple heuristic (e.g., 'atr', 'ema', 'stoch') - might need refinement
             if indicator_name == 'stoch': indicator_name = 'stoch_rsi' # Specific case
             if indicator_name == 'sma': indicator_name = 'sma_10'
             if indicator_name == 'bollinger': indicator_name = 'bollinger_bands'
             if indicator_name == 'volume': indicator_name = 'volume_confirmation'

             if self.config.get("indicators", {}).get(indicator_name, True): # Assume enabled if not specified
                 required_periods.append(self.config.get(key, default))

        # StochRSI requires RSI period + Stoch period + K/D smoothing
        if self.config.get("indicators", {}).get("stoch_rsi"):
             stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
             rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
             k_len = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
             d_len = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
             # Effective length is roughly RSI_len + Stoch_len + D_len
             required_periods.append(rsi_len + stoch_rsi_len + d_len)


        min_required_data = max(required_periods) + 20 if required_periods else 50 # Add buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may contain NaNs.{RESET}")
             # Continue, but expect NaNs

        try:
            # Use a temporary DataFrame for calculations to avoid modifying the original if errors occur mid-way
            df_calc = self.df.copy()
            indicators_config = self.config.get("indicators", {})

            # Make sure necessary base columns exist and are numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df_calc.columns:
                    self.logger.error(f"Missing required column '{col}' in DataFrame for {self.symbol}. Cannot calculate indicators.")
                    return
                # Ensure numeric type, coercing errors
                df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            # Drop rows with NaN in OHLC after conversion, as TA functions need them
            df_calc.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if df_calc.empty:
                self.logger.error(f"DataFrame became empty after dropping NaN OHLC values for {self.symbol}. Cannot calculate indicators.")
                return


            # Always calculate ATR (needed for SL/TP sizing and BE)
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # --- Calculate Enabled Indicators ---
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
                # MFI requires typical price = (High + Low + Close) / 3
                if all(c in df_calc for c in ['high', 'low', 'close']):
                    df_calc.ta.mfi(length=mfi_period, append=True)
                    self.ta_column_names["MFI"] = self._get_ta_col_name("MFI", df_calc)
                else: self.logger.warning("MFI requires 'high', 'low', 'close'. Skipping.")


            if indicators_config.get("vwap", False):
                # VWAP typically needs daily anchor or specific implementation
                # pandas_ta VWAP often defaults to daily ('anchor="D"')
                try:
                    df_calc.ta.vwap(anchor="D", append=True) # Explicitly request daily anchor
                    self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)
                except Exception as vwap_err:
                     self.logger.warning(f"VWAP calculation failed (may need more data or specific anchor): {vwap_err}. Skipping.")


            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                # PSAR calculation can sometimes return None if data is insufficient/problematic
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                if psar_result is not None and not psar_result.empty:
                    df_calc = pd.concat([df_calc, psar_result], axis=1)
                    self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                    self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)
                else: self.logger.warning("PSAR calculation returned no result. Skipping.")


            if indicators_config.get("sma_10", False):
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                # Check if enough data for underlying RSI calculation first
                if len(df_calc) >= stoch_rsi_rsi_len + stoch_rsi_len + stoch_rsi_d: # Approx check
                    stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                    if stochrsi_result is not None and not stochrsi_result.empty:
                         # Concatenate safely, avoiding duplicate columns if names clash
                         new_cols = {col: f"{col}_stochrsi" for col in stochrsi_result.columns if col in df_calc.columns}
                         stochrsi_result.rename(columns=new_cols, inplace=True)
                         df_calc = pd.concat([df_calc, stochrsi_result], axis=1)
                         self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                         self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)
                    else: self.logger.warning("StochRSI calculation returned no result. Skipping.")
                else: self.logger.warning(f"Insufficient data for StochRSI ({len(df_calc)} rows). Skipping.")


            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = float(self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)) # Ensure float
                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std)
                if bbands_result is not None and not bbands_result.empty:
                    # Concatenate safely
                    new_cols = {col: f"{col}_bb" for col in bbands_result.columns if col in df_calc.columns}
                    bbands_result.rename(columns=new_cols, inplace=True)
                    df_calc = pd.concat([df_calc, bbands_result], axis=1)
                    self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                    self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                    self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)
                else: self.logger.warning("Bollinger Bands calculation returned no result. Skipping.")


            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                if 'volume' in df_calc.columns:
                    vol_ma_col_name = f"VOL_SMA_{vol_ma_period}" # Custom name
                    # Calculate SMA on volume, filling potential NaNs in volume with 0 first
                    df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)
                    self.ta_column_names["Volume_MA"] = vol_ma_col_name
                else: self.logger.warning("Volume column not found. Skipping Volume MA calculation.")

            # Calculation finished, assign back to self.df
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
             # This often happens if pandas_ta is not installed or df format is wrong
             self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} (check pandas_ta install & df structure): {e}{RESET}", exc_info=True)
             self.df = self.df # Keep original df if calc failed
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)
            self.df = self.df # Keep original df if calc failed


    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest values from self.df."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(f"Cannot update latest values: DataFrame empty or last row is all NaN for {self.symbol}.")
            # Initialize with NaNs if update fails
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low", "Open"]}
            return

        try:
            latest = self.df.iloc[-1]
            updated_values = {}

            # Process TA indicator columns found during calculation
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index:
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            # Store ATR as Decimal for precision in SL/TP calcs
                            if key == "ATR":
                                updated_values[key] = Decimal(str(value))
                            # Store most other indicators as float for scoring
                            else:
                                updated_values[key] = float(value)
                        except (ValueError, TypeError, InvalidOperation):
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) for {self.symbol}.")
                            updated_values[key] = np.nan
                    else:
                        updated_values[key] = np.nan # Store NaN if value is NaN
                else:
                    # Log only if the column was expected but missing from the last row
                    if key in self.ta_column_names:
                        self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found or NaN in latest data for {self.symbol}. Storing NaN.")
                    updated_values[key] = np.nan

            # Add essential price/volume data as Decimal for precision
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                 value = latest.get(base_col)
                 key_name = base_col.capitalize()
                 if pd.notna(value):
                      try:
                           updated_values[key_name] = Decimal(str(value))
                      except (ValueError, TypeError, InvalidOperation):
                           self.logger.warning(f"Could not convert base value for '{base_col}' ({value}) to Decimal for {self.symbol}.")
                           updated_values[key_name] = np.nan
                 else:
                      updated_values[key_name] = np.nan

            self.indicator_values = updated_values

            # Format for logging, handling NaNs gracefully
            valid_values_log = {}
            price_prec = self.get_price_precision() # Get precision once
            for k, v in self.indicator_values.items():
                 if pd.notna(v):
                     if isinstance(v, Decimal):
                          # Use specific precision for prices/ATR, default for others
                          prec = price_prec if k in ['Open','High','Low','Close','ATR'] else 6
                          try: valid_values_log[k] = f"{v:.{prec}f}"
                          except InvalidOperation: valid_values_log[k] = "NaN(Decimal)" # Handle potential Decimal NaNs
                     elif isinstance(v, float):
                          valid_values_log[k] = f"{v:.5f}"
                     else: # Handle other types if necessary
                          valid_values_log[k] = str(v)
                 # else: Keep implicit NaN (don't add to log dict)

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
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
             self.logger.warning(f"Missing 'high' or 'low' columns for Fibonacci on {self.symbol}.")
             self.fib_levels_data = {}
             return {}
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure high/low are numeric before max/min
            high_col = pd.to_numeric(df_slice["high"], errors='coerce')
            low_col = pd.to_numeric(df_slice["low"], errors='coerce')

            high_price_raw = high_col.dropna().max()
            low_price_raw = low_col.dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                 self.logger.warning(f"Could not find valid high/low in window for Fibonacci on {self.symbol}.")
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
                    # Calculate level price: For uptrend, level = High - Diff * Pct
                    # For downtrend, level = Low + Diff * Pct. Assume High > Low for simplicity here.
                    level_price = (high - (diff * Decimal(str(level_pct))))
                    # Quantize to the market's price precision (rounding down for support levels)
                    level_price_quantized = level_price.quantize(rounding_factor, rounding=ROUND_DOWN)
                    levels[level_name] = level_price_quantized
            else:
                 # If high equals low, all levels are the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol}.")
                 level_price_quantized = high.quantize(rounding_factor, rounding=ROUND_DOWN)
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            # Log levels formatted as strings
            log_levels = {k: f"{v:.{price_precision}f}" for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {log_levels}")
            return levels
        except (InvalidOperation, TypeError, ValueError) as conv_err:
             self.logger.error(f"{NEON_RED}Fibonacci calculation error (Decimal conversion) for {self.symbol}: {conv_err}{RESET}")
             self.fib_levels_data = {}
             return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info."""
        try:
            # 1. Try market['precision']['price'] (often number of decimals as int)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')

            if isinstance(price_precision_val, int) and price_precision_val >= 0:
                 return price_precision_val

            # 2. Try market['precision']['price'] as tick size (float/str)
            if isinstance(price_precision_val, (float, str)):
                 tick_size = Decimal(str(price_precision_val))
                 if tick_size > 0:
                     # Calculate decimals from tick size (e.g., 0.01 -> 2, 0.0005 -> 4)
                     return abs(tick_size.normalize().as_tuple().exponent)

            # 3. Try market['limits']['price']['min'] (sometimes the tick size)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                min_price_tick = Decimal(str(min_price_val))
                # Heuristic: If min price is small (e.g., < 0.1), assume it's the tick size
                if min_price_tick > 0 and min_price_tick < Decimal('0.1'):
                    self.logger.debug(f"Inferring price precision from min price limit ({min_price_tick}) for {self.symbol}.")
                    return abs(min_price_tick.normalize().as_tuple().exponent)

            # 4. Fallback: Infer from last close price
            last_close = self.indicator_values.get("Close") # Uses Decimal value
            if isinstance(last_close, Decimal) and last_close > 0:
                 precision = abs(last_close.normalize().as_tuple().exponent)
                 self.logger.debug(f"Inferring price precision from last close price ({last_close}) as {precision} for {self.symbol}.")
                 return max(0, precision) # Ensure non-negative

        except (KeyError, TypeError, ValueError, InvalidOperation, AttributeError) as e:
            self.logger.warning(f"Could not reliably determine price precision for {self.symbol} from market info: {e}. Falling back.")

        # Default fallback precision
        default_precision = 4
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        try:
            # 1. Try precision.price first (often the tick size as float/str)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if isinstance(price_precision_val, (float, str)):
                 tick_size = Decimal(str(price_precision_val))
                 if tick_size > 0: return tick_size

            # 2. Fallback: limits.price.min (sometimes represents tick size)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                min_tick_from_limit = Decimal(str(min_price_val))
                if min_tick_from_limit > 0:
                    # Heuristic check: assume it's a tick if small
                    if min_tick_from_limit < Decimal('0.1'):
                         self.logger.debug(f"Using tick size from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                         return min_tick_from_limit
                    else:
                         self.logger.debug(f"limits.price.min ({min_tick_from_limit}) seems too large for tick size, potentially min order price.")

            # 3. Fallback: calculate from decimal places if precision.price was an int
            if isinstance(price_precision_val, int) and price_precision_val >= 0:
                 tick_size = Decimal('1') / (Decimal('10') ** price_precision_val)
                 self.logger.debug(f"Calculated tick size from precision.price (decimals={price_precision_val}): {tick_size} for {self.symbol}")
                 return tick_size

        except (KeyError, TypeError, ValueError, InvalidOperation, AttributeError) as e:
             self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using precision fallback.")

        # Final fallback: Use get_price_precision (decimal places)
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.warning(f"Using fallback tick size based on precision places ({price_precision_places}): {fallback_tick} for {self.symbol}")
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
                     # Should not happen if calculate_fibonacci_levels works correctly
                     self.logger.warning(f"Non-decimal value found in fib_levels_data: {name}={level_price}")

            # Sort by distance, then by level price as a tie-breaker (arbitrary but consistent)
            level_distances.sort(key=lambda x: (x['distance'], x['level']))
            # Return list of (name, price) tuples
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score based on latest values. Returns float score or NaN."""
        ema_short = self.indicator_values.get("EMA_Short") # Float
        ema_long = self.indicator_values.get("EMA_Long") # Float
        current_price_val = self.indicator_values.get("Close") # Decimal

        # Convert Decimal price to float for comparison, handle NaN/None
        try:
            current_price_float = float(current_price_val) if isinstance(current_price_val, Decimal) else np.nan
        except (TypeError, ValueError):
            current_price_float = np.nan

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price_float):
            return np.nan

        # Simple alignment check
        if current_price_float > ema_short > ema_long: return 1.0 # Strong bullish alignment
        elif current_price_float < ema_short < ema_long: return -1.0 # Strong bearish alignment
        # Weaker alignment / crossover signals (optional, can be refined)
        elif ema_short > ema_long and current_price_float > ema_long: return 0.5 # Bullish crossover / above long EMA
        elif ema_short < ema_long and current_price_float < ema_long: return -0.5 # Bearish crossover / below long EMA
        else: return 0.0 # Neutral / conflicting

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
        debug_scores = {} # Store individual scores for logging

        # --- Pre-checks ---
        if not self.indicator_values:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values dictionary is empty.{RESET}")
             return "HOLD"
        # Check if any core indicators (not just OHLCV) have non-NaN values
        core_indicators_present = any(pd.notna(v) for k, v in self.indicator_values.items() if k not in ['Open','High','Low','Close','Volume', 'ATR']) # Exclude ATR too as it's always calculated
        if not core_indicators_present:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: All core indicator values are NaN.{RESET}")
            return "HOLD"
        if pd.isna(current_price) or not isinstance(current_price, Decimal) or current_price <= 0:
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
             return "HOLD"

        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
             self.logger.error(f"Active weight set '{self.active_weight_set_name}' missing or empty for {self.symbol}. Cannot generate signal.")
             return "HOLD"

        # --- Iterate Through Enabled Indicators ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue # Skip disabled indicators

            weight_str = active_weights.get(indicator_key)
            # Skip if no weight defined for this enabled indicator in the active set
            if weight_str is None: continue

            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue # Skip zero-weighted indicators
            except (InvalidOperation, ValueError, TypeError):
                self.logger.warning(f"Invalid weight format '{weight_str}' for indicator '{indicator_key}' in set '{self.active_weight_set_name}'. Skipping.")
                continue

            # Find the corresponding check method
            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method = getattr(self, check_method_name)
                indicator_score_float = np.nan # Expect float score from check methods
                try:
                    # Pass necessary data to the check method
                    if indicator_key == "orderbook":
                         # Only call if orderbook data was fetched successfully
                         indicator_score_float = method(orderbook_data, current_price) if orderbook_data else np.nan
                    else:
                         indicator_score_float = method() # Returns float score -1.0 to 1.0 or np.nan

                except Exception as e:
                    self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                    # Score remains NaN

                # Log the raw score before applying weight
                debug_scores[indicator_key] = f"{indicator_score_float:.2f}" if pd.notna(indicator_score_float) else "NaN"

                # Process valid scores
                if pd.notna(indicator_score_float):
                    try:
                        # Convert float score to Decimal for weighted sum
                        score_decimal = Decimal(str(indicator_score_float))
                        # Clamp score between -1 and 1 before applying weight
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_decimal))
                        score_contribution = clamped_score * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight # Sum weights of indicators that provided a score
                        active_indicator_count += 1
                    except (InvalidOperation, ValueError, TypeError) as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key} ({indicator_score_float}): {calc_err}")
                        nan_indicator_count += 1
                else:
                    # Count indicators that were enabled and weighted but returned NaN
                    nan_indicator_count += 1
            else:
                # Log warning if an enabled/weighted indicator doesn't have a check method
                self.logger.warning(f"Check method '{check_method_name}' not found for enabled/weighted indicator: {indicator_key} ({self.symbol})")

        # --- Determine Final Signal ---
        if total_weight_applied == 0:
             # This happens if all enabled/weighted indicators returned NaN or had errors
             self.logger.warning(f"No indicators contributed a valid score for {self.symbol}. Defaulting to HOLD.")
             final_signal = "HOLD"
        else:
            # Normalize score? Optional, depends on strategy. Current approach uses raw weighted sum.
            # normalized_score = final_signal_score / total_weight_applied # Example if normalization is desired

            # Use threshold from config (convert to Decimal)
            try: threshold = Decimal(str(self.config.get("signal_score_threshold", 1.5)))
            except InvalidOperation: threshold = Decimal("1.5"); self.logger.warning("Invalid signal_score_threshold, using default 1.5")

            # Determine signal based on threshold
            if final_signal_score >= threshold: final_signal = "BUY"
            elif final_signal_score <= -threshold: final_signal = "SELL"
            else: final_signal = "HOLD"

        # --- Log Summary ---
        price_prec = self.get_price_precision()
        log_msg = (
            f"Signal Summary ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set={self.active_weight_set_name}, Indis=[Actv:{active_indicator_count}/NaN:{nan_indicator_count}], WgtSum={total_weight_applied:.2f}, "
            f"Score={final_signal_score:.4f} (Thresh: +/-{threshold:.2f}) "
            f"==> {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        # Log individual scores only at debug level for less console noise
        self.logger.debug(f"  Indicator Scores: {debug_scores}")

        # Update internal signal state (less critical, but might be useful)
        if final_signal in self.signals: self.signals[final_signal] = 1
        return final_signal


    # --- Indicator Check Methods (returning float score -1.0 to 1.0 or np.nan) ---
    # Ensure these methods use self.indicator_values correctly and handle potential NaNs.

    def _check_ema_alignment(self) -> float:
        # Relies on calculate_ema_alignment_score which handles NaNs
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
             return np.nan
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum") # Float
        if pd.isna(momentum): return np.nan
        # Simple thresholding (adjust thresholds as needed)
        # Normalize momentum to roughly -1 to 1 range if possible, or use fixed thresholds
        # Example: Assume typical MOM range is -2 to 2 for this asset/timeframe
        # score = np.clip(momentum / 2.0, -1.0, 1.0)
        # Example using fixed thresholds:
        if momentum > 0.5: return 1.0  # Strong positive momentum
        if momentum < -0.5: return -1.0 # Strong negative momentum
        if momentum > 0.1: return 0.5 # Moderate positive
        if momentum < -0.1: return -0.5 # Moderate negative
        return 0.0 # Near zero

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume") # Decimal
        volume_ma_float = self.indicator_values.get("Volume_MA") # Float

        if pd.isna(current_volume) or pd.isna(volume_ma_float) or volume_ma_float <= 0: return np.nan
        try:
            volume_ma = Decimal(str(volume_ma_float)) # Convert MA to Decimal for comparison
            multiplier = Decimal(str(self.config.get("volume_confirmation_multiplier", 1.5)))

            if current_volume > volume_ma * multiplier:
                return 0.7 # High volume confirmation (positive score regardless of price direction)
            elif current_volume < volume_ma / multiplier:
                return -0.4 # Low volume suggests lack of conviction (negative score)
            else:
                return 0.0 # Neutral volume
        except (InvalidOperation, ValueError, TypeError):
            self.logger.warning("Error comparing volume to MA.")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K") # Float
        d = self.indicator_values.get("StochRSI_D") # Float
        if pd.isna(k) or pd.isna(d): return np.nan

        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
        score = 0.0

        # Oversold/Overbought conditions
        if k < oversold and d < oversold: score = 1.0 # Strong buy signal potential
        elif k > overbought and d > overbought: score = -1.0 # Strong sell signal potential

        # Crossover logic (K crossing D)
        # Use previous values if available, otherwise approximate with current diff
        # Note: Accessing previous values would require storing more state or looking back in self.df
        # Simple difference check:
        diff = k - d
        if k > d: # K above D (bullish)
            if score <= 0: score = 0.6 # If not already oversold, signal bullish cross
            else: score = max(score, 0.6) # Combine with oversold signal
        elif k < d: # K below D (bearish)
            if score >= 0: score = -0.6 # If not already overbought, signal bearish cross
            else: score = min(score, -0.6) # Combine with overbought signal

        # Dampen signal in neutral zone
        if oversold <= k <= overbought: score *= 0.5

        return np.clip(score, -1.0, 1.0)

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI") # Float
        if pd.isna(rsi): return np.nan
        # More granular scoring based on RSI levels
        if rsi <= 30: return 1.0    # Oversold
        if rsi >= 70: return -1.0   # Overbought
        if rsi < 40: return 0.6     # Approaching oversold / weak
        if rsi > 60: return -0.6    # Approaching overbought / weak
        if 45 < rsi < 55: return 0.0 # Neutral zone
        if 40 <= rsi <= 45: return 0.2 # Weak bullish momentum
        if 55 <= rsi <= 60: return -0.2 # Weak bearish momentum
        # Fallback for ranges not explicitly covered (shouldn't be needed with above)
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI") # Float
        if pd.isna(cci): return np.nan
        # Score based on typical CCI thresholds
        if cci <= -150: return 1.0   # Strong oversold / potential reversal up
        if cci >= 150: return -1.0  # Strong overbought / potential reversal down
        if cci < -80: return 0.6    # Moderately oversold
        if cci > 80: return -0.6   # Moderately overbought
        # Consider zero line cross or direction
        if cci > 0: return -0.1 # Slightly bearish bias above zero? (Or adjust logic)
        if cci < 0: return 0.1  # Slightly bullish bias below zero?
        return 0.0 # Around zero

    def _check_wr(self) -> float: # Williams %R
        wr = self.indicator_values.get("Williams_R") # Float (-100 to 0)
        if pd.isna(wr): return np.nan
        # WR scores are inverse of RSI/Stoch
        if wr <= -80: return 1.0    # Oversold (potential buy)
        if wr >= -20: return -1.0   # Overbought (potential sell)
        if wr < -50: return 0.4     # In lower half (more bullish)
        if wr > -50: return -0.4    # In upper half (more bearish)
        return 0.0 # Near -50 midpoint

    def _check_psar(self) -> float:
        psar_l = self.indicator_values.get("PSAR_long") # Float (price when trend is long)
        psar_s = self.indicator_values.get("PSAR_short") # Float (price when trend is short)
        # PSAR indicates trend direction. If PSARl has a value, trend is up. If PSARs has value, trend is down.
        # They are mutually exclusive in the pandas_ta output (one is NaN, the other has the SAR value).
        if pd.notna(psar_l) and pd.isna(psar_s): return 1.0 # Uptrend indicated
        elif pd.notna(psar_s) and pd.isna(psar_l): return -1.0 # Downtrend indicated
        else: return 0.0 # No clear trend / NaN values

    def _check_sma_10(self) -> float:
        sma_10 = self.indicator_values.get("SMA10") # Float
        last_close_val = self.indicator_values.get("Close") # Decimal
        try: last_close_float = float(last_close_val) if isinstance(last_close_val, Decimal) else np.nan
        except (TypeError, ValueError): last_close_float = np.nan

        if pd.isna(sma_10) or pd.isna(last_close_float): return np.nan

        # Simple price vs SMA comparison
        if last_close_float > sma_10: return 0.6 # Price above short-term MA (bullish)
        if last_close_float < sma_10: return -0.6 # Price below short-term MA (bearish)
        return 0.0 # Price is exactly on the SMA

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP") # Float
        last_close_val = self.indicator_values.get("Close") # Decimal
        try: last_close_float = float(last_close_val) if isinstance(last_close_val, Decimal) else np.nan
        except (TypeError, ValueError): last_close_float = np.nan

        if pd.isna(vwap) or pd.isna(last_close_float): return np.nan

        # Price relative to VWAP
        if last_close_float > vwap: return 0.7 # Price above VWAP (bullish, institutional support)
        if last_close_float < vwap: return -0.7 # Price below VWAP (bearish, institutional resistance)
        return 0.0

    def _check_mfi(self) -> float: # Money Flow Index
        mfi = self.indicator_values.get("MFI") # Float (0-100)
        if pd.isna(mfi): return np.nan
        # Similar interpretation to RSI but includes volume
        if mfi <= 20: return 1.0    # Oversold (potential buying pressure incoming)
        if mfi >= 80: return -1.0   # Overbought (potential selling pressure incoming)
        if mfi < 40: return 0.4     # Weakness, approaching oversold
        if mfi > 60: return -0.4    # Strength, approaching overbought
        # Neutral zone can be wider for MFI
        if 40 <= mfi <= 60: return 0.0
        return 0.0 # Fallback

    def _check_bollinger_bands(self) -> float:
        bb_lower = self.indicator_values.get("BB_Lower") # Float
        bb_upper = self.indicator_values.get("BB_Upper") # Float
        bb_middle = self.indicator_values.get("BB_Middle") # Float
        last_close_val = self.indicator_values.get("Close") # Decimal
        try: last_close_float = float(last_close_val) if isinstance(last_close_val, Decimal) else np.nan
        except (TypeError, ValueError): last_close_float = np.nan

        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(last_close_float):
            return np.nan

        # Check for touches/crosses of bands
        if last_close_float < bb_lower: return 1.0 # Below lower band (potential reversal buy)
        if last_close_float > bb_upper: return -1.0 # Above upper band (potential reversal sell)

        # Position within the bands
        band_width = bb_upper - bb_lower
        if band_width > 0:
             # Score based on proximity to upper/lower band relative to middle
             if last_close_float > bb_middle:
                 # Closer to upper band -> more bearish (within band context)
                 proximity_to_upper = (last_close_float - bb_middle) / (bb_upper - bb_middle) if (bb_upper - bb_middle) > 0 else 0
                 return -0.5 * proximity_to_upper # Score from 0 (at middle) to -0.5 (near upper)
             else: # Closer to lower band -> more bullish
                 proximity_to_lower = (bb_middle - last_close_float) / (bb_middle - bb_lower) if (bb_middle - bb_lower) > 0 else 0
                 return 0.5 * proximity_to_lower # Score from 0 (at middle) to 0.5 (near lower)
        else: # Bands are very tight or inverted (unlikely)
            return 0.0

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes order book depth imbalance. Returns float score or NaN."""
        if not orderbook_data: return np.nan
        try:
            bids = orderbook_data.get('bids', []) # [[price, amount], ...] sorted high to low
            asks = orderbook_data.get('asks', []) # [[price, amount], ...] sorted low to high
            if not bids or not asks:
                 self.logger.debug("Orderbook missing bids or asks.")
                 return np.nan

            # Analyze cumulative volume within a certain price range (e.g., +/- 0.5% from current price)
            price_range_pct = Decimal("0.005") # 0.5%
            price_low_limit = current_price * (Decimal(1) - price_range_pct)
            price_high_limit = current_price * (Decimal(1) + price_range_pct)

            bid_volume_in_range = sum(Decimal(str(bid[1])) for bid in bids if Decimal(str(bid[0])) >= price_low_limit)
            ask_volume_in_range = sum(Decimal(str(ask[1])) for ask in asks if Decimal(str(ask[0])) <= price_high_limit)

            # Alternative: Fixed number of levels
            # num_levels_to_check = 10
            # bid_volume_sum = sum(Decimal(str(bid[1])) for bid in bids[:num_levels_to_check])
            # ask_volume_sum = sum(Decimal(str(ask[1])) for ask in asks[:num_levels_to_check])

            bid_volume_sum = bid_volume_in_range
            ask_volume_sum = ask_volume_in_range

            total_volume = bid_volume_sum + ask_volume_sum
            if total_volume <= 0:
                 self.logger.debug("Zero total volume in analyzed orderbook range.")
                 return 0.0

            # Calculate Order Book Imbalance (OBI)
            obi = (bid_volume_sum - ask_volume_sum) / total_volume
            score = float(obi) # OBI is naturally scaled between -1 (all asks) and 1 (all bids)

            # Clamp score just in case of edge cases
            score = np.clip(score, -1.0, 1.0)

            self.logger.debug(f"Orderbook check ({self.symbol}): Range={price_range_pct:.2%}, "
                              f"BidVol={bid_volume_sum:.4f}, AskVol={ask_volume_sum:.4f}, OBI={obi:.4f}, Score={score:.4f}")
            return score

        except (TypeError, ValueError, InvalidOperation) as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol} (data format issue?): {e}{RESET}")
            return np.nan
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error during orderbook analysis for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential TP and initial SL levels based on entry estimate, ATR,
        and multipliers. Returns (entry_price_estimate, take_profit, stop_loss), all Decimal or None.
        Ensures SL/TP are valid relative to entry and tick size.
        """
        # Ensure entry price is valid Decimal
        if not isinstance(entry_price_estimate, Decimal) or pd.isna(entry_price_estimate) or entry_price_estimate <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid entry price estimate ({entry_price_estimate}).{RESET}")
            return entry_price_estimate, None, None

        if signal not in ["BUY", "SELL"]:
            # Return entry price even if no signal, but SL/TP are None
            return entry_price_estimate, None, None

        atr_val = self.indicator_values.get("ATR") # Should be Decimal
        if not isinstance(atr_val, Decimal) or pd.isna(atr_val) or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid ATR ({atr_val}). Using zero offset.{RESET}")
            # Allow calculation to proceed with zero ATR, resulting in SL/TP near entry (or potentially invalid)
            # This might be better than failing entirely if ATR calculation failed temporarily
            atr_val = Decimal('0')
            # return entry_price_estimate, None, None # Original behavior: fail if ATR invalid


        try:
            entry_price = entry_price_estimate # Use the provided estimate

            # Use Decimal for multipliers
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"Min tick size ({min_tick}) invalid for {self.symbol}. Cannot reliably calculate SL/TP.")
                 return entry_price, None, None

            take_profit = None
            stop_loss = None
            tp_offset = atr_val * tp_multiple
            sl_offset = atr_val * sl_multiple

            if signal == "BUY":
                tp_raw = entry_price + tp_offset
                sl_raw = entry_price - sl_offset
                # Quantize TP UP to nearest tick/precision, SL DOWN
                take_profit = (tp_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (sl_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
            elif signal == "SELL":
                tp_raw = entry_price - tp_offset
                sl_raw = entry_price + sl_offset
                # Quantize TP DOWN to nearest tick/precision, SL UP
                take_profit = (tp_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (sl_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

            # --- Validation and Adjustments ---
            # Ensure SL is strictly beyond entry by at least one tick
            if signal == "BUY" and stop_loss >= entry_price:
                 stop_loss = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                 self.logger.debug(f"Adjusted BUY SL below entry (min 1 tick): {stop_loss}")
            elif signal == "SELL" and stop_loss <= entry_price:
                 stop_loss = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                 self.logger.debug(f"Adjusted SELL SL above entry (min 1 tick): {stop_loss}")

            # Ensure TP is potentially profitable (strictly beyond entry by at least one tick)
            if signal == "BUY" and take_profit <= entry_price:
                 # Option 1: Set to minimum profitable target
                 take_profit = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                 self.logger.warning(f"{NEON_YELLOW}BUY TP calculation non-profitable. Adjusted to min profitable target: {take_profit}{RESET}")
                 # Option 2: Set TP to None
                 # self.logger.warning(f"{NEON_YELLOW}BUY TP calculation non-profitable (TP {take_profit} <= Entry {entry_price}). Setting TP to None.{RESET}")
                 # take_profit = None
            elif signal == "SELL" and take_profit >= entry_price:
                 # Option 1: Set to minimum profitable target
                 take_profit = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                 self.logger.warning(f"{NEON_YELLOW}SELL TP calculation non-profitable. Adjusted to min profitable target: {take_profit}{RESET}")
                 # Option 2: Set TP to None
                 # self.logger.warning(f"{NEON_YELLOW}SELL TP calculation non-profitable (TP {take_profit} >= Entry {entry_price}). Setting TP to None.{RESET}")
                 # take_profit = None

            # Ensure SL/TP are positive numbers
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in non-positive price ({stop_loss}). Setting SL to None.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.warning(f"{NEON_YELLOW}Take profit calculation resulted in non-positive price ({take_profit}). Setting TP to None.{RESET}")
                take_profit = None

            # Final log of calculated values
            tp_str = f"{take_profit:.{price_precision}f}" if take_profit else "None"
            sl_str = f"{stop_loss:.{price_precision}f}" if stop_loss else "None"
            atr_str = f"{atr_val:.{price_precision+1}f}" if isinstance(atr_val, Decimal) else str(atr_val)
            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_price:.{price_precision}f}, TP={tp_str}, SL={sl_str}, ATR={atr_str}")
            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError) as calc_err:
             self.logger.error(f"{NEON_RED}Error during TP/SL calculation ({self.symbol}): {calc_err}{RESET}", exc_info=True)
             return entry_price_estimate, None, None
        except Exception as e:
             self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
             return entry_price_estimate, None, None


# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency, handling various structures."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info = None
            account_types_to_try = ['CONTRACT', 'UNIFIED', None] # Try specific, unified, then default
            found_structure = False
            last_error = None

            for acc_type in account_types_to_try:
                 try:
                     params = {'type': acc_type} if acc_type else {}
                     lg.debug(f"Fetching balance (Params: {params}) for {currency}...")
                     balance_info = exchange.fetch_balance(params=params)

                     # --- Parse Balance Info ---
                     available_balance_str = None
                     # 1. Standard CCXT (Top level or currency key)
                     if balance_info and currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free']: {available_balance_str}")
                     elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                         available_balance_str = str(balance_info['free'][currency])
                         lg.debug(f"Found balance via top-level 'free' dict: {available_balance_str}")

                     # 2. Bybit V5 Nested Structure (Unified/Contract)
                     elif balance_info and 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             # Check accountType matches if possible (CONTRACT=Derivatives, UNIFIED=UTA, SPOT=Spot)
                             # acc_info_type = account.get('accountType') # TODO: Check if this exists and is useful

                             if isinstance(account.get('coin'), list):
                                 for coin_data in account['coin']:
                                      if coin_data.get('coin') == currency:
                                          # Prioritize available for withdrawal/trade, then wallet balance
                                          free = coin_data.get('availableToWithdraw') or \
                                                 coin_data.get('availableBalance') or \
                                                 coin_data.get('availableToBorrow') or \
                                                 coin_data.get('walletBalance')
                                          if free is not None:
                                               available_balance_str = str(free)
                                               lg.debug(f"Found balance via Bybit V5 nested ['coin']: {available_balance_str}")
                                               break
                                 if available_balance_str is not None: break
                         if available_balance_str is None:
                              lg.debug(f"Currency '{currency}' not found within V5 'info.result.list[].coin[]' structure for type '{acc_type}'.")

                     # 3. Fallback to 'total' balance if 'free' is missing/zero but total exists
                     if available_balance_str is None or Decimal(available_balance_str) == 0:
                         total_balance = None
                         if balance_info and currency in balance_info and balance_info[currency].get('total') is not None:
                              total_balance = balance_info[currency]['total']
                         elif 'total' in balance_info and currency in balance_info['total'] and balance_info['total'][currency] is not None:
                              total_balance = balance_info['total'][currency]

                         if total_balance is not None and Decimal(str(total_balance)) > 0:
                              lg.warning(f"{NEON_YELLOW}Using 'total' balance ({total_balance}) as fallback for {currency} (Free: {available_balance_str}).{RESET}")
                              available_balance_str = str(total_balance)

                     # --- Convert to Decimal ---
                     if available_balance_str is not None:
                         try:
                             final_balance = Decimal(available_balance_str)
                             if final_balance >= 0:
                                  lg.info(f"Available {currency} balance: {final_balance:.4f} (using type: {acc_type or 'default'})")
                                  return final_balance
                             else:
                                  lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Trying next type.")
                                  # Continue to next account type if balance is negative
                         except (InvalidOperation, ValueError, TypeError) as e:
                             lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}. Trying next type.")
                             # Continue to next account type
                     else:
                         # If balance string is None after checks, means currency wasn't found for this acc_type
                         lg.debug(f"Currency '{currency}' not found in balance response for type '{acc_type}'.")
                         # Continue to next account type

                 except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                     lg.debug(f"Error fetching balance for type '{acc_type or 'default'}': {e}. Trying next.")
                     last_error = e
                     continue # Try next account type
                 except Exception as e:
                     lg.warning(f"Unexpected error fetching balance type '{acc_type or 'default'}': {e}. Trying next.")
                     last_error = e
                     continue # Try next account type

            # If loop finishes without returning a balance
            lg.error(f"{NEON_RED}Could not determine balance for {currency} after trying account types: {account_types_to_try}. Last Error: {last_error}{RESET}")
            lg.debug(f"Last full balance_info structure received (if any): {balance_info}")
            return None # Failed after trying all types

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance for {currency}: {e}. Retrying...{RESET}")
            last_error = e
        except ccxt.RateLimitExceeded as e:
            wait_time = max(RETRY_DELAY_SECONDS * 5, e.params.get('retry-after', RETRY_DELAY_SECONDS * 5))
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            attempts += 1
            continue # Skip standard delay, retry immediately after wait
        except Exception as e: # Catch errors from the initial try/except blocks if fetch failed entirely
            lg.error(f"{NEON_RED}Critical error during balance fetch attempt {attempts + 1} for {currency}: {e}{RESET}", exc_info=True)
            last_error = e
            # Don't retry unexpected critical errors immediately
            return None

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. Last error: {last_error}{RESET}")
    return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision, limits, contract type."""
    lg = logger
    try:
        # Ensure markets are loaded
        if not exchange.markets or symbol not in exchange.markets:
             lg.info(f"Market info for {symbol} not loaded or missing, reloading markets...")
             try:
                 exchange.load_markets(reload=True)
             except Exception as load_err:
                 lg.error(f"{NEON_RED}Failed to reload markets: {load_err}{RESET}")
                 return None # Cannot proceed without markets

        if symbol not in exchange.markets:
             lg.error(f"{NEON_RED}Market {symbol} still not found after reloading.{RESET}")
             # Attempt common variations (e.g., PERP for futures) - less reliable
             variations = [f"{symbol}PERP"] if '/' in symbol else []
             for var_sym in variations:
                  if var_sym in exchange.markets:
                       lg.warning(f"Found variation '{var_sym}' instead of '{symbol}'. Using '{var_sym}'.")
                       symbol = var_sym # Use the found variation
                       break
             else: # If loop completes without finding variation
                 return None


        market = exchange.market(symbol)
        if market:
            # Standardize market type detection
            market_type = market.get('type', 'unknown') # spot, future, swap
            is_linear = market.get('linear', False)
            is_inverse = market.get('inverse', False)
            is_contract = market.get('contract', False) or market_type in ['swap', 'future'] or is_linear or is_inverse

            contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "Spot/Other"

            # Extract relevant info safely
            precision = market.get('precision', {})
            limits = market.get('limits', {})
            amount_limits = limits.get('amount', {})
            price_limits = limits.get('price', {})
            cost_limits = limits.get('cost', {})
            contract_size = market.get('contractSize', '1' if is_contract else None) # Default contract size to 1 if contract, else None

            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                f"Type={market_type}, Contract Type={contract_type}, Active={market.get('active', True)}, "
                f"Precision(Price/Amount): {precision.get('price')}/{precision.get('amount')}, "
                f"Limits(Amount Min/Max): {amount_limits.get('min')}/{amount_limits.get('max')}, "
                f"Limits(Price Min/Max): {price_limits.get('min')}/{price_limits.get('max')}, "
                f"Limits(Cost Min/Max): {cost_limits.get('min')}/{cost_limits.get('max')}, "
                f"Contract Size: {contract_size}"
            )
            # Add our standardized flags for easier use later
            market['is_contract'] = is_contract
            market['contract_type'] = contract_type # 'Linear', 'Inverse', 'Spot/Other'
            return market
        else:
             # Should not happen if symbol is in exchange.markets
             lg.error(f"{NEON_RED}Market dictionary is None for {symbol} despite being in market list.{RESET}")
             return None
    except ccxt.BadSymbol as e:
         # This might occur if the initially passed symbol was truly invalid
         lg.error(f"{NEON_RED}Symbol '{symbol}' not supported or invalid according to CCXT: {e}{RESET}")
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
    exchange: ccxt.Exchange, # Pass exchange for formatting helpers
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size based on risk, SL distance, balance, and market constraints.
    Uses Decimal for precision. Returns size in base currency or contracts.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    contract_type = market_info.get('contract_type', 'Spot/Other')
    # Use 'contracts' for size unit if it's a derivative, otherwise use base currency name
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or balance <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
         lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
         return None
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= 0 or \
       not isinstance(entry_price, Decimal) or entry_price <= 0:
        lg.error(f"Position sizing failed ({symbol}): Invalid entry ({entry_price}) or SL ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
         lg.error(f"Position sizing failed ({symbol}): SL price ({initial_stop_loss_price}) equals entry price ({entry_price}).")
         return None
    if 'limits' not in market_info or 'precision' not in market_info:
         lg.error(f"Position sizing failed ({symbol}): Market info missing limits/precision.")
         # Attempt to proceed with defaults? Risky. Better to fail.
         return None

    try:
        # --- Risk Amount ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))

        # --- SL Distance ---
        sl_distance_per_unit_price = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit_price <= 0:
             lg.error(f"Position sizing failed ({symbol}): SL distance is zero/negative ({sl_distance_per_unit_price}).")
             return None

        # --- Contract Size (Value per contract) ---
        contract_size_val = Decimal('1.0') # Default for Spot or if info missing
        if is_contract:
            contract_size_str = market_info.get('contractSize')
            if contract_size_str is not None:
                try: contract_size_val = Decimal(str(contract_size_str)); assert contract_size_val > 0
                except (InvalidOperation, ValueError, TypeError, AssertionError):
                    lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}, using 1.0.")
                    contract_size_val = Decimal('1.0')
            else: lg.warning(f"Contract size missing for {symbol}, assuming 1.0.")

        # --- Calculate Risk per Unit (in Quote Currency) ---
        # For Linear contracts/Spot: Risk per unit = SL distance * Contract Size (value)
        # For Inverse contracts: Risk per unit = SL distance * Contract Size (value) / SL Price (approx) - More complex, needs care
        risk_per_unit_quote = Decimal('0')
        if contract_type == 'Linear' or not is_contract:
            risk_per_unit_quote = sl_distance_per_unit_price * contract_size_val
        elif contract_type == 'Inverse':
            # Value of 1 contract = contract_size_val / price
            # Risk = Size * (Value@Entry - Value@SL)
            # Risk = Size * (CS/Entry - CS/SL) = Size*CS*(1/Entry - 1/SL)
            # Size = RiskAmt / (CS * |1/Entry - 1/SL|)
            # Alternatively, approx: Risk per contract = CS * SL_Dist / SL_Price
            if initial_stop_loss_price > 0:
                 risk_per_unit_quote = (contract_size_val * sl_distance_per_unit_price) / initial_stop_loss_price
                 lg.debug(f"Using Inverse contract risk approximation: {risk_per_unit_quote:.8f} quote per contract.")
            else:
                 lg.error(f"Cannot calculate risk for inverse contract {symbol} with zero/negative SL price.")
                 return None
        else:
            lg.error(f"Unknown contract type '{contract_type}' for sizing {symbol}.")
            return None

        if risk_per_unit_quote <= 0:
            lg.error(f"Calculated risk per unit is zero/negative ({risk_per_unit_quote}). Cannot size position.")
            return None

        # --- Calculate Initial Size ---
        calculated_size = risk_amount_quote / risk_per_unit_quote

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit_price}")
        lg.info(f"  ContractType={contract_type}, ContractSize={contract_size_val}, RiskPerUnit={risk_per_unit_quote:.8f} {quote_currency}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {}) # Cost is usually size * price (for linear/spot)
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount') # Number of decimal places or step size

        # Helper to get Decimal limit or default
        def get_limit(limit_dict, key, default_val_str):
            val_str = limit_dict.get(key)
            if val_str is not None:
                try: return Decimal(str(val_str))
                except (InvalidOperation, ValueError, TypeError): pass
            return Decimal(default_val_str)

        min_amount = get_limit(amount_limits, 'min', '0')
        max_amount = get_limit(amount_limits, 'max', 'inf')
        min_cost = get_limit(cost_limits, 'min', '0')
        max_cost = get_limit(cost_limits, 'max', 'inf')

        # 1. Clamp by MIN/MAX AMOUNT limits
        adjusted_size = max(min_amount, min(calculated_size, max_amount))
        if adjusted_size != calculated_size:
             lg.warning(f"{NEON_YELLOW}Size adjusted by amount limits: {calculated_size:.8f} -> {adjusted_size:.8f} {size_unit}{RESET}")
             if adjusted_size == min_amount and calculated_size < min_amount:
                 lg.warning(f"  Reason: Calculated size was below minimum ({min_amount}).")
             elif adjusted_size == max_amount and calculated_size > max_amount:
                 lg.warning(f"  Reason: Calculated size was above maximum ({max_amount}).")

        # 2. Check COST limits (approximate for linear/spot)
        # Cost calculation needs care for inverse contracts
        current_cost = Decimal('0')
        if contract_type == 'Linear' or not is_contract:
            current_cost = adjusted_size * entry_price * contract_size_val # contract_size_val is 1 for spot
        elif contract_type == 'Inverse':
            # Cost = Size * ContractSize / EntryPrice (in Quote currency)
            if entry_price > 0:
                 current_cost = adjusted_size * contract_size_val / entry_price
            else: lg.warning("Cannot calculate cost for inverse with zero entry price.")

        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Estimated Cost={current_cost:.4f} {quote_currency}")

        # Adjust size based on COST limits if necessary
        cost_adjusted = False
        if min_cost > 0 and current_cost > 0 and current_cost < min_cost :
             lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} below min cost {min_cost:.4f}. Attempting to increase size.{RESET}")
             required_size_for_min_cost = None
             if contract_type == 'Linear' or not is_contract:
                 if entry_price > 0 and contract_size_val > 0:
                      required_size_for_min_cost = min_cost / (entry_price * contract_size_val)
             elif contract_type == 'Inverse':
                  if contract_size_val > 0:
                       required_size_for_min_cost = min_cost * entry_price / contract_size_val # Rearranged

             if required_size_for_min_cost is None or required_size_for_min_cost <= 0:
                 lg.error("Cannot calculate required size for min cost.")
                 return None

             lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
             if required_size_for_min_cost > max_amount:
                  lg.error(f"{NEON_RED}Cannot meet min cost {min_cost:.4f} without exceeding max amount {max_amount:.8f}. Aborted.{RESET}")
                  return None
             # Ensure the size required for min cost also meets min amount
             if required_size_for_min_cost < min_amount:
                  lg.warning(f"{NEON_YELLOW}Size required for min cost ({required_size_for_min_cost:.8f}) is below min amount ({min_amount:.8f}). Adjusting to min amount instead.{RESET}")
                  # This might still not meet min cost, but it's the smallest valid amount.
                  # Re-evaluate if min_amount should take precedence or if trade should be aborted.
                  # For now, let's use min_amount.
                  if adjusted_size < min_amount: # Only adjust if current adjusted size is also too low
                       adjusted_size = min_amount
                       cost_adjusted = True
                       lg.info(f"  Adjusting size to min amount limit: {adjusted_size:.8f}")
                  # If adjusted_size was already >= min_amount (due to initial calc), don't reduce it.
             else:
                  lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f}")
                  adjusted_size = required_size_for_min_cost
                  cost_adjusted = True


        elif max_cost > 0 and current_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} exceeds max cost {max_cost:.4f}. Reducing size.{RESET}")
             adjusted_size_for_max_cost = None
             if contract_type == 'Linear' or not is_contract:
                 if entry_price > 0 and contract_size_val > 0:
                      adjusted_size_for_max_cost = max_cost / (entry_price * contract_size_val)
             elif contract_type == 'Inverse':
                  if contract_size_val > 0:
                       adjusted_size_for_max_cost = max_cost * entry_price / contract_size_val

             if adjusted_size_for_max_cost is None or adjusted_size_for_max_cost <= 0:
                  lg.error("Cannot calculate size reduction for max cost.")
                  return None

             lg.info(f"  Reduced size allowed by max cost: {adjusted_size_for_max_cost:.8f}")
             # Ensure the reduced size doesn't go below min amount
             if adjusted_size_for_max_cost < min_amount:
                  lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is below min amount {min_amount:.8f}. Aborted.{RESET}")
                  return None
             else:
                 adjusted_size = adjusted_size_for_max_cost
                 cost_adjusted = True

        # 3. Apply Amount Precision/Step Size
        try:
            # Use ccxt's decimal_to_precision for rounding based on market precision info
            # Determine padding mode (TRUNCATE is safer, avoids rounding up over limits)
            # The precision value itself can be int (decimals) or float/str (step size)
            amount_precision_mode = exchange.TRUNCATE # Or exchange.ROUND
            formatted_size_str = exchange.decimal_to_precision(
                adjusted_size,
                rounding_mode=amount_precision_mode,
                precision=amount_precision_val, # Pass the market precision value directly
                padding_mode=exchange.NO_PADDING # Avoid trailing zeros unless needed
            )
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Mode: {amount_precision_mode}, Precision: {amount_precision_val}): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as fmt_err:
            lg.error(f"{NEON_RED}Critical error applying amount precision using ccxt helpers: {fmt_err}. Aborting sizing.{RESET}", exc_info=True)
            # Fallback to manual rounding is risky without knowing if precision is step or decimals
            # It's safer to abort if the exchange formatting fails.
            return None

        # --- Final Validation ---
        if final_size <= 0:
             lg.error(f"{NEON_RED}Position size became zero/negative ({final_size}) after adjustments. Aborted.{RESET}")
             return None
        # Check against min/max amount AGAIN after precision formatting
        if final_size < min_amount:
             lg.error(f"{NEON_RED}Final size {final_size} after precision formatting is below minimum amount {min_amount}. Aborted.{RESET}")
             return None
        if final_size > max_amount:
             lg.error(f"{NEON_RED}Final size {final_size} after precision formatting is above maximum amount {max_amount}. Aborted.{RESET}")
             return None

        # Final cost check (recalculate with final_size)
        final_cost = Decimal('0')
        if contract_type == 'Linear' or not is_contract:
             final_cost = final_size * entry_price * contract_size_val
        elif contract_type == 'Inverse':
             if entry_price > 0: final_cost = final_size * contract_size_val / entry_price

        if min_cost > 0 and final_cost > 0 and final_cost < min_cost:
            # This might happen if rounding down pushed cost below minimum
            lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} which is below minimum cost {min_cost:.4f}. Aborted.{RESET}")
            return None
        if max_cost > 0 and final_cost > max_cost:
            # This might happen if rounding up pushed cost above maximum (less likely with TRUNCATE)
            lg.error(f"{NEON_RED}Final size {final_size} results in cost {final_cost:.4f} which is above maximum cost {max_cost:.4f}. Aborted.{RESET}")
            return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for symbol: {symbol} (Attempt {attempts + 1})")
            positions: List[Dict] = []
            fetch_all = False # Flag to determine if we need fetch_positions() without args

            # Attempt fetch single symbol (preferred for efficiency, esp. Bybit V5)
            try:
                # Add timeout parameter if supported
                params = {}
                if 'fetchPositionsTimeout' in exchange.options:
                     params['requestTimeout'] = exchange.options['fetchPositionsTimeout']
                # Bybit V5 might need category hint if not default
                # params['category'] = 'linear' # Or 'inverse' or 'spot' - depends on symbol type if ambiguous
                positions = exchange.fetch_positions([symbol], params=params)
                lg.debug(f"fetch_positions([symbol]) returned {len(positions)} entries.")
            except ccxt.ArgumentsRequired:
                # Exchange doesn't support fetching single symbol, need to fetch all
                lg.debug("fetch_positions requires no arguments, fetching all.")
                fetch_all = True
            except ccxt.ExchangeError as e:
                 # Handle common "position not found" errors gracefully
                 no_pos_codes_v5 = [110025] # Position idx not match / Position is closed (Bybit V5)
                 no_pos_msg_patterns = ["position not found", "position is closed", "no position found"]
                 err_str = str(e).lower()
                 if any(p in err_str for p in no_pos_msg_patterns) or (hasattr(e, 'code') and e.code in no_pos_codes_v5):
                      lg.info(f"No position found for {symbol} (Exchange confirmed: {e}).")
                      return None # Explicitly no position found
                 else:
                      # Re-raise other exchange errors for retry logic below
                      raise e
            # except Exception as e: # Catch other unexpected errors during single fetch
            #      lg.error(f"Unexpected error fetching single position for {symbol}: {e}", exc_info=True)
            #      # Re-raise for retry logic
            #      raise e

            if fetch_all:
                try:
                     params = {}
                     if 'fetchPositionsTimeout' in exchange.options:
                          params['requestTimeout'] = exchange.options['fetchPositionsTimeout']
                     all_positions = exchange.fetch_positions(params=params)
                     # Filter for the specific symbol
                     positions = [p for p in all_positions if p.get('symbol') == symbol]
                     lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
                except Exception as e:
                     # Re-raise errors during 'fetch all' for retry logic
                     raise e

            # --- Process fetched positions ---
            active_position = None
            # Use a small threshold to consider position non-zero due to potential float inaccuracies
            size_threshold = Decimal('1e-9')

            for pos in positions:
                pos_size_str = None
                # Try standard 'contracts', then Bybit V5 'info.size'
                if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
                elif pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size'])

                if pos_size_str is None: continue # Skip if size cannot be determined

                try:
                    position_size = Decimal(pos_size_str)
                    # Check if the absolute size is greater than the threshold
                    if abs(position_size) > size_threshold:
                        active_position = pos
                        lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                        # Assume only one active position per symbol in one-way mode
                        break
                except (InvalidOperation, ValueError, TypeError):
                    lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol}.")
                    continue # Skip this position entry

            # --- Post-Process the found active position ---
            if active_position:
                # Standardize and enhance the position dictionary
                info_dict = active_position.get('info', {})

                # Size (Decimal)
                size_decimal = Decimal('0')
                size_str = active_position.get('contracts') or info_dict.get('size')
                if size_str is not None:
                    try: size_decimal = Decimal(str(size_str))
                    except (InvalidOperation, ValueError, TypeError): pass # Keep zero if parsing fails

                # Side ('long' or 'short')
                side = active_position.get('side')
                if side not in ['long', 'short']: # Infer side if missing or invalid
                    if size_decimal > size_threshold: side = 'long'
                    elif size_decimal < -size_threshold: side = 'short'
                    else: side = 'unknown' # Should not happen if size > threshold check passed
                active_position['side'] = side # Store standardized/inferred side

                # Entry Price (Decimal)
                entry_price = None
                entry_str = active_position.get('entryPrice') or info_dict.get('avgPrice')
                if entry_str is not None:
                     try: entry_price = Decimal(str(entry_str))
                     except (InvalidOperation, ValueError, TypeError): pass
                active_position['entryPriceDecimal'] = entry_price # Store as Decimal

                # Stop Loss Price (Decimal)
                sl_price = None
                sl_str = active_position.get('stopLossPrice') or info_dict.get('stopLoss')
                if sl_str is not None and str(sl_str).strip() != '' and str(sl_str) != '0':
                     try: sl_price = Decimal(str(sl_str))
                     except (InvalidOperation, ValueError, TypeError): pass
                active_position['stopLossPriceDecimal'] = sl_price

                # Take Profit Price (Decimal)
                tp_price = None
                tp_str = active_position.get('takeProfitPrice') or info_dict.get('takeProfit')
                if tp_str is not None and str(tp_str).strip() != '' and str(tp_str) != '0':
                     try: tp_price = Decimal(str(tp_str))
                     except (InvalidOperation, ValueError, TypeError): pass
                active_position['takeProfitPriceDecimal'] = tp_price

                # Trailing Stop Distance (Decimal) - Bybit V5: 'trailingStop'
                tsl_distance = None
                tsl_dist_str = info_dict.get('trailingStop')
                if tsl_dist_str is not None and str(tsl_dist_str).strip() != '' and str(tsl_dist_str) != '0':
                     try: tsl_distance = Decimal(str(tsl_dist_str))
                     except (InvalidOperation, ValueError, TypeError): pass
                active_position['trailingStopLossDistanceDecimal'] = tsl_distance

                # TSL Activation Price (Decimal) - Bybit V5: 'activePrice'
                tsl_activation_price = None
                tsl_act_str = info_dict.get('activePrice')
                if tsl_act_str is not None and str(tsl_act_str).strip() != '' and str(tsl_act_str) != '0':
                     try: tsl_activation_price = Decimal(str(tsl_act_str))
                     except (InvalidOperation, ValueError, TypeError): pass
                active_position['tslActivationPriceDecimal'] = tsl_activation_price

                # --- Log Summary ---
                # Helper to format Decimal for logging or return 'N/A'
                def format_log_decimal(d_val: Optional[Decimal], precision: int):
                    if d_val is not None and isinstance(d_val, Decimal):
                        try: return f"{d_val:.{precision}f}"
                        except InvalidOperation: return "NaN"
                    return "N/A"

                # Get precisions for formatting
                analyzer_temp = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, get_market_info(exchange, symbol, lg) or {})
                price_prec = analyzer_temp.get_price_precision()
                amount_prec = 8 # Default amount precision if market info fails for size display
                try:
                     market_inf = get_market_info(exchange, symbol, lg)
                     if market_inf and market_inf.get('precision',{}).get('amount') is not None:
                          amount_prec_val = market_inf['precision']['amount']
                          if isinstance(amount_prec_val, int): amount_prec = amount_prec_val
                          elif isinstance(amount_prec_val, (float, str)): # Step size
                               amount_prec = abs(Decimal(str(amount_prec_val)).normalize().as_tuple().exponent)
                except Exception: pass

                contracts_abs = format_log_decimal(abs(size_decimal), amount_prec)
                entry_p_fmt = format_log_decimal(entry_price, price_prec)
                liq_price_fmt = format_log_decimal(active_position.get('liquidationPrice'), price_prec) # Liq price might be string directly
                leverage_str = active_position.get('leverage') or info_dict.get('leverage')
                leverage_fmt = f"{Decimal(leverage_str):.1f}x" if leverage_str is not None else 'N/A'
                pnl_fmt = format_log_decimal(active_position.get('unrealizedPnl'), 4) # More precision for PNL
                sl_fmt = format_log_decimal(sl_price, price_prec)
                tp_fmt = format_log_decimal(tp_price, price_prec)
                tsl_dist_fmt = format_log_decimal(tsl_distance, price_prec)
                tsl_act_fmt = format_log_decimal(tsl_activation_price, price_prec)

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={contracts_abs}, Entry={entry_p_fmt}, Liq={liq_price_fmt}, "
                            f"Lev={leverage_fmt}, PnL={pnl_fmt}, SL={sl_fmt}, TP={tp_fmt}, "
                            f"TSL(Dist/Act): {tsl_dist_fmt}/{tsl_act_fmt}")
                logger.debug(f"Full position details for {symbol}: {active_position}")
                return active_position # Return the enhanced dictionary
            else:
                # This case is reached if fetch succeeded but no position entries had size > threshold
                logger.info(f"No active open position found for {symbol}.")
                return None

        # --- Retry Logic for Fetch Errors ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching positions for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = max(RETRY_DELAY_SECONDS * 5, e.params.get('retry-after', RETRY_DELAY_SECONDS * 5))
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            attempts += 1
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
             # Only retry specific potentially temporary exchange errors
             if e.http_status in RETRY_ERROR_CODES:
                  lg.warning(f"{NEON_YELLOW}Retryable exchange error ({e.http_status}) fetching positions: {e}. Retrying...{RESET}")
             else:
                  lg.error(f"{NEON_RED}Non-retryable exchange error fetching positions for {symbol}: {e}{RESET}")
                  return None # Fail fast on non-retryable errors
        except Exception as e:
             lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
             return None # Fail fast on unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             time.sleep(RETRY_DELAY_SECONDS)

    # If retries exhausted
    lg.error(f"{NEON_RED}Failed to fetch positions for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract).")
        return True # Considered success as no action needed
    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be positive integer.")
        return False
    if not exchange.has.get('setLeverage'):
         lg.error(f"{NEON_RED}Exchange {exchange.id} does not support setLeverage via CCXT.{RESET}")
         return False

    # Check against market limits if available
    leverage_limits = market_info.get('limits', {}).get('leverage', {})
    min_leverage = leverage_limits.get('min')
    max_leverage = leverage_limits.get('max')
    if max_leverage is not None and leverage > max_leverage:
         lg.error(f"{NEON_RED}Desired leverage {leverage}x exceeds market maximum {max_leverage}x for {symbol}. Aborting.{RESET}")
         return False
    if min_leverage is not None and leverage < min_leverage:
         lg.warning(f"{NEON_YELLOW}Desired leverage {leverage}x is below market minimum {min_leverage}x for {symbol}. Using minimum.{RESET}")
         leverage = int(min_leverage) # Adjust to minimum allowed

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        # Prepare parameters, especially for Bybit V5 which requires buy/sell leverage
        params = {}
        if 'bybit' in exchange.id.lower():
             # Bybit V5 requires string format for leverage in params
             params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
             lg.debug(f"Using Bybit V5 params for setLeverage: {params}")

        # Add timeout parameter if supported
        if 'setLeverageTimeout' in exchange.options: # Check if specific timeout exists
             params['requestTimeout'] = exchange.options['setLeverageTimeout']
        elif 'createOrderTimeout' in exchange.options: # Fallback to general order timeout
             params['requestTimeout'] = exchange.options['createOrderTimeout']


        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # Verification: Some exchanges return the set leverage. Bybit might not directly.
        # We assume success if no exception is raised, but log the response.
        # A subsequent fetch_position call is the best verification.
        lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
        return True

    except ccxt.ExchangeError as e:
        # Handle common errors specifically
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")

        if bybit_code == 110045 or "leverage not modified" in err_str:
            lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True # Treat as success
        elif bybit_code in [110028, 110009, 110055, 30086] or "margin mode" in err_str or "isolated margin" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross). Leverage setting might require specific mode or fail if positions exist in wrong mode.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed the current Risk Limit tier for {symbol}. Check Bybit Risk Limits in UI.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid or out of range for {symbol}. Check allowed range ({min_leverage}-{max_leverage}).{RESET}")
        elif "set margin mode" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: May need to set margin mode (Isolated/Cross) before setting leverage.{RESET}")
        # Add more specific error code handling if encountered
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
         lg.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}. Consider retrying.{RESET}")
         # Decide whether to retry here or let the main loop handle retries
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False # Return False if any error occurred


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal,
    market_info: Dict,
    logger: Optional[logging.Logger] = None,
    reduce_only: bool = False # Flag for closing orders
) -> Optional[Dict]:
    """
    Places a market order using CCXT. Returns the order dictionary on success, None on failure.
    Uses reduce_only flag for closing positions. Handles Decimal conversion and basic validation.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', '')
    size_unit = "Contracts" if is_contract else base_currency
    action = "Close" if reduce_only else "Open/Increase"

    # --- Validate Size ---
    if not isinstance(position_size, Decimal) or position_size <= 0:
        lg.error(f"Trade aborted ({symbol} {side} {action}): Invalid position size ({position_size}). Must be positive Decimal.")
        return None
    # Convert Decimal size to float for ccxt amount parameter
    try:
        amount_float = float(position_size)
        if amount_float <= 0: raise ValueError("Float conversion resulted in non-positive size")
    except (ValueError, TypeError) as e:
        lg.error(f"Trade aborted ({symbol} {side} {action}): Failed to convert size {position_size} to valid float: {e}")
        return None

    # --- Prepare Parameters ---
    params = {}
    # Bybit V5 specific parameters for derivatives
    if is_contract and 'bybit' in exchange.id.lower():
        params = {
            'category': market_info.get('contract_type', 'linear').lower(), # linear, inverse, spot
            'positionIdx': 0,  # 0 for One-Way Mode. Needs adjustment for Hedge Mode (1=Buy, 2=Sell).
            'reduceOnly': reduce_only,
            # 'timeInForce': 'IOC' if reduce_only else 'GTC', # Market orders are usually FOK/IOC by default
        }
        # For closing orders, ImmediateOrCancel is often preferred
        if reduce_only:
             params['timeInForce'] = 'IOC'

    # Add timeout parameter if supported
    if 'createOrderTimeout' in exchange.options:
         params['requestTimeout'] = exchange.options['createOrderTimeout']


    lg.info(f"Attempting to place {action} {side.upper()} {order_type} order for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {size_unit} ({position_size})") # Log float and original Decimal
    lg.info(f"  Params: {params}")

    try:
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float, # CCXT generally expects float for amount
            price=None, # Market order doesn't need price
            params=params
        )
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A') # e.g., 'open', 'closed', 'canceled'
        filled_size = order.get('filled', 0.0)
        avg_price = order.get('average')

        lg.info(f"{NEON_GREEN}{action} Trade Placed! ID: {order_id}, Initial Status: {order_status}, Filled: {filled_size}, AvgPrice: {avg_price}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} {action}): {order}")

        # Basic check if market order filled immediately (status 'closed')
        if order_status == 'closed' and filled_size > 0:
             lg.info(f"{NEON_GREEN}Market order filled immediately.{RESET}")
        elif order_status == 'open':
             lg.warning(f"{NEON_YELLOW}Market order status is 'open'. May take time to fill or failed partially. Monitor position.{RESET}")
        # Add checks for other statuses like 'rejected', 'canceled' if applicable

        return order

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {action} {side} order ({symbol}): {e}{RESET}")
        # Log current balance for debugging
        try:
            balance = fetch_balance(exchange, market_info.get('quote', QUOTE_CURRENCY), lg)
            lg.info(f"Current available balance: {balance}")
        except Exception: lg.warning("Could not fetch balance for logging.")
        lg.error(f"{NEON_YELLOW} >> Hint: Check available margin/balance, leverage, and calculated order cost.{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters placing {action} {side} order ({symbol}): {e}{RESET}")
        lg.error(f"  Order Details: Size={amount_float}, Params={params}")
        bybit_code = getattr(e, 'code', None)
        if bybit_code == 110017 or "exceeds risk limit" in str(e).lower():
             lg.error(f"{NEON_YELLOW} >> Hint (110017): Order size likely exceeds Risk Limit for current leverage. Increase risk limit tier or reduce size/leverage.{RESET}")
        elif bybit_code == 110007 or "order qty" in str(e).lower():
             lg.error(f"{NEON_YELLOW} >> Hint (110007): Order quantity invalid. Check min/max amount limits and step size ({market_info.get('limits',{}).get('amount')}). Final size: {position_size}{RESET}")
        elif bybit_code == 110014 and reduce_only:
             lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only order failed. Position might be smaller than close size, already closed, or API issue? Check position first.{RESET}")
        elif "order cost" in str(e).lower():
             lg.error(f"{NEON_YELLOW} >> Hint: Order cost invalid. Check min/max cost limits ({market_info.get('limits',{}).get('cost')}).{RESET}")
        else: lg.error(f"{NEON_YELLOW} >> Hint: Review order size, price (N/A for market), and parameters against exchange rules.{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing {action} order ({symbol}): {e}. May need retry.{RESET}")
        # Let main loop handle retries usually, but could implement local retry here too
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing {action} order ({symbol}): {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110025 and reduce_only: # Position not found/zero on close attempt
            lg.warning(f"{NEON_YELLOW} >> Hint (110025): Position might have been closed already or size mismatch when placing reduce-only order.{RESET}")
        elif bybit_code == 10001: # Parameter error general
             lg.error(f"{NEON_YELLOW} >> Hint (10001): General parameter error. Double-check all params: {params}{RESET}")
        # Add more specific Bybit codes if needed
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
    Handles parameter formatting, API call, and error interpretation.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract).")
        return False # Cannot set protection on non-contracts
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information (needed for context like posIdx).")
        return False
    if 'bybit' not in exchange.id.lower():
        lg.error(f"Protection setting logic currently specific to Bybit V5. Exchange: {exchange.id}")
        return False

    pos_side = position_info.get('side') # Should be 'long' or 'short' from get_open_position
    if pos_side not in ['long', 'short']:
         lg.error(f"Cannot set protection for {symbol}: Invalid or missing position side ('{pos_side}') in position_info.")
         return False

    # Validate inputs are positive Decimals if provided
    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0)

    if not has_sl and not has_tp and not has_tsl:
         lg.info(f"No valid protection parameters provided for {symbol}. No protection set/updated.")
         # If the intent was to set nothing, return True. If inputs were invalid, maybe False? Let's assume True.
         return True

    # --- Prepare API Parameters (Bybit V5 /v5/position/set-trading-stop) ---
    category = market_info.get('contract_type', 'linear').lower() # 'linear' or 'inverse'
    # Determine positionIdx (crucial for Bybit Hedge Mode, usually 0 for One-Way)
    position_idx = 0 # Default for One-Way
    try:
        # Try to get from info dict first, as standard field might be missing
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
             position_idx = int(pos_idx_val)
        # Add logic here if Hedge Mode needs specific idx (1 for long, 2 for short)
        # hedge_mode = exchange.options.get('hedgeMode', False) # Check if hedge mode is enabled
        # if hedge_mode: position_idx = 1 if pos_side == 'long' else 2

    except (ValueError, TypeError) as idx_err:
         lg.warning(f"Could not parse positionIdx from position info ({idx_err}), using default {position_idx}.")

    params = {
        'category': category,
        'symbol': market_info['id'], # Use exchange-specific ID
        'tpslMode': 'Full', # Apply to whole position ('Full') or partial ('Partial')
        # Trigger price type: LastPrice, MarkPrice, IndexPrice
        'slTriggerBy': 'LastPrice', # Adjust if Mark Price preferred
        'tpTriggerBy': 'LastPrice',
        # Order type when triggered: Market or Limit
        'slOrderType': 'Market', # Market recommended for SL
        'tpOrderType': 'Market', # Market or Limit for TP
        'positionIdx': position_idx
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} PosIdx: {position_idx}):"]

    # --- Format and Add Protection Parameters ---
    try:
        # Use exchange formatting methods for price and potentially distance
        # Need market info for precision/tick size
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, CONFIG, market_info) # Temp instance for helpers
        price_precision_digits = analyzer.get_price_precision()
        min_tick = analyzer.get_min_tick_size()
        if min_tick <= 0: raise ValueError("Invalid min_tick_size")

        # Helper to format price string using exchange rules (handles precision/tick)
        def format_price_param(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try:
                 # Use price_to_precision which handles tick size rounding
                 formatted = exchange.price_to_precision(symbol, float(price_decimal))
                 # Ensure it's not zero after formatting if original was positive
                 if Decimal(formatted) <= 0:
                      lg.warning(f"Price {price_decimal} formatted to non-positive '{formatted}'. Skipping.")
                      return None
                 return formatted
            except Exception as e:
                 lg.error(f"Failed to format price {price_decimal} using exchange.price_to_precision: {e}")
                 # Fallback: Manual quantization (less reliable)
                 # quantized = price_decimal.quantize(min_tick, rounding=ROUND_DOWN if price_decimal<0 else ROUND_UP) # Example
                 # return str(quantized)
                 return None # Safer to return None if formatting fails

        # Handle TSL first, as Bybit prioritizes it over fixed SL if both are sent
        if has_tsl:
            # Format TSL distance (treated as a price difference, needs tick size)
            # Round distance UP to nearest tick increment to ensure it's at least the intended value
            try:
                 quantized_tsl_distance = (trailing_stop_distance / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                 if quantized_tsl_distance < min_tick: quantized_tsl_distance = min_tick # Enforce minimum distance
                 formatted_tsl_distance = exchange.decimal_to_precision(
                     quantized_tsl_distance, exchange.ROUND, precision=price_precision_digits # Use price precision for distance formatting
                 ).rstrip('0').rstrip('.') # Clean up trailing zeros
                 if Decimal(formatted_tsl_distance) <= 0: raise ValueError("Formatted TSL distance non-positive")
            except Exception as e:
                 lg.warning(f"Failed to format TSL distance {trailing_stop_distance} (MinTick: {min_tick}): {e}"); formatted_tsl_distance = None

            formatted_activation_price = format_price_param(tsl_activation_price)

            if formatted_tsl_distance and formatted_activation_price:
                params['trailingStop'] = formatted_tsl_distance
                params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                # If TSL is successfully set, Bybit ignores 'stopLoss'. No need to explicitly remove 'stopLoss' param here.
                has_sl = False # Mark fixed SL as effectively overridden/not set in this API call
            else:
                lg.error(f"Failed to format valid TSL parameters for {symbol}. TSL will not be set.")
                has_tsl = False # Mark TSL as failed

        # Fixed Stop Loss (only add if TSL wasn't successfully prepared)
        if has_sl:
            formatted_sl = format_price_param(stop_loss_price)
            if formatted_sl:
                params['stopLoss'] = formatted_sl
                log_parts.append(f"  Fixed SL: {formatted_sl}")
            else:
                lg.error(f"Failed to format valid SL price {stop_loss_price}. Fixed SL will not be set.")
                has_sl = False # Mark SL as failed

        # Fixed Take Profit (can be set alongside SL or TSL)
        if has_tp:
            formatted_tp = format_price_param(take_profit_price)
            if formatted_tp:
                params['takeProfit'] = formatted_tp
                log_parts.append(f"  Fixed TP: {formatted_tp}")
            else:
                lg.error(f"Failed to format valid TP price {take_profit_price}. Fixed TP will not be set.")
                has_tp = False # Mark TP as failed

    except Exception as fmt_err:
         lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    # Check if any protection is actually being sent after formatting and TSL override logic
    # Note: Sending empty strings might clear existing SL/TP on Bybit V5, be careful.
    # Only send parameters that have valid formatted values.
    final_params = {k: v for k, v in params.items() if v is not None and str(v).strip() != ''}

    # Check if essential protection keys are present in the final params
    protection_keys_present = any(k in final_params for k in ['stopLoss', 'takeProfit', 'trailingStop'])

    if not protection_keys_present:
        lg.warning(f"No valid protection parameters could be formatted or remained after adjustments for {symbol}. No API call made.")
        # Return True if the original intent was to set nothing (all inputs None)
        # Return False if formatting failed for potentially valid inputs. Let's return False to indicate failure.
        return False

    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={final_params})")

    # --- Call Bybit V5 API Endpoint ---
    try:
        # Add specific timeout for this call if available
        api_params = final_params.copy()
        if 'privatePostV5PositionSetTradingStopTimeout' in exchange.options:
             api_params['requestTimeout'] = exchange.options['privatePostV5PositionSetTradingStopTimeout']

        response = exchange.private_post('/v5/position/set-trading-stop', api_params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        # --- Parse Response ---
        # Bybit V5 response structure: { "retCode": 0, "retMsg": "OK", "result": {}, "retExtInfo": {}, "time": ... }
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {}) # Often contains more details on failure

        if ret_code == 0:
            # Even with retCode 0, retMsg might indicate partial success or no change needed
            if "not modified" in ret_msg.lower():
                 lg.info(f"{NEON_YELLOW}Position protection already set to target values or no change needed for {symbol}. Response: {ret_msg}{RESET}")
            elif "success" in ret_msg.lower() or "ok" in ret_msg.lower():
                 lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            else:
                # Success code but ambiguous message, log as info/warning
                lg.warning(f"{NEON_YELLOW}Protection API call successful (Code 0) but message unclear: '{ret_msg}' for {symbol}. Check position status.{RESET}")
            return True
        else:
            # --- Handle Specific Error Codes ---
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            if ret_code == 110013: # Parameter error
                 lg.error(f"{NEON_YELLOW} >> Hint (110013 Parameter Error): Check SL/TP prices vs Mark/Last/Entry, TSL dist/act prices, tick size compliance. Sent: {final_params}{RESET}")
            elif ret_code == 110043: # Position status error (e.g., closing)
                 lg.error(f"{NEON_YELLOW} >> Hint (110043 Position Status): Position might be closing or in liquidation, cannot modify TP/SL.{RESET}")
            elif ret_code == 110036: # TSL Active price invalid
                 lg.error(f"{NEON_YELLOW} >> Hint (110036 TSL Activation Price): Activation price {final_params.get('activePrice')} likely invalid (already passed, wrong side vs entry, too close?).{RESET}")
            elif ret_code == 110086: # SL equals TP
                 lg.error(f"{NEON_YELLOW} >> Hint (110086 SL=TP): Stop loss price cannot equal take profit price.{RESET}")
            elif ret_code == 110037: # TSL distance invalid
                 lg.error(f"{NEON_YELLOW} >> Hint (110037 TSL Distance): Trailing stop distance {final_params.get('trailingStop')} invalid (too small/large, violates tick size?). MinTick: {min_tick}{RESET}")
            elif ret_code == 110025: # Position not found / closed
                 lg.error(f"{NEON_YELLOW} >> Hint (110025 Position Not Found): Position might have closed unexpectedly before protection could be set.{RESET}")
            # Add more hints based on Bybit V5 API documentation for this endpoint
            return False

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error during protection API call for {symbol}: {e}. May need retry.{RESET}")
    except ccxt.ExchangeError as e: # Catch potential errors CCXT wraps
         lg.error(f"{NEON_RED}ExchangeError during protection API call for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during protection API call for {symbol}: {e}{RESET}", exc_info=True)
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Confirmed position dict with 'entryPriceDecimal', 'side'
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Optional TP to set alongside TSL
) -> bool:
    """
    Calculates TSL parameters based on config and position, then calls the internal helper
    to set the TSL (and optionally TP) using the exchange API.
    """
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss disabled in config for {symbol}. Skipping TSL setup.")
        # Return True because the *intent* was not to set TSL, not an error.
        # Or return False to indicate TSL wasn't actively set? Let's choose False.
        return False

    # --- Get TSL Config Parameters (use Decimal) ---
    try:
        callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format in config ({symbol}): {e}. Cannot calculate TSL.{RESET}")
        return False
    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid 'trailing_stop_callback_rate' ({callback_rate}) in config. Must be positive.{RESET}")
        return False
    if activation_percentage < 0:
         lg.error(f"{NEON_RED}Invalid 'trailing_stop_activation_percentage' ({activation_percentage}) in config. Must be non-negative.{RESET}")
         return False

    # --- Get Position Info ---
    try:
        entry_price = position_info.get('entryPriceDecimal') # Use pre-calculated Decimal
        side = position_info.get('side')
        if not isinstance(entry_price, Decimal) or entry_price <= 0 or side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing or invalid position info (entryPriceDecimal, side) for TSL calc ({symbol}). Entry: {entry_price}, Side: {side}.{RESET}")
            return False
    except KeyError:
         lg.error(f"{NEON_RED}Missing required keys ('entryPriceDecimal', 'side') in position_info dict for TSL calc ({symbol}).{RESET}")
         return False
    except Exception as e: # Catch other potential errors accessing dict
        lg.error(f"{NEON_RED}Error accessing position info for TSL calculation ({symbol}): {e}.{RESET}")
        return False

    # --- Calculate TSL Parameters ---
    try:
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance for helpers
        price_precision = analyzer.get_price_precision()
        min_tick_size = analyzer.get_min_tick_size()
        if min_tick_size <= 0: raise ValueError("Invalid min_tick_size")

        # 1. Calculate Activation Price
        activation_price = None
        activation_offset = entry_price * activation_percentage

        if side == 'long':
            raw_activation = entry_price + activation_offset
            # Quantize UP to nearest tick
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
            # Ensure activation is strictly > entry if percentage > 0
            if activation_percentage > 0 and activation_price <= entry_price:
                activation_price = entry_price + min_tick_size # Move one tick away
            # For immediate activation (0%), set slightly above entry (Bybit might require > 0 offset)
            # Let's set it one tick away for 0% as well, safer.
            elif activation_percentage == 0:
                 activation_price = entry_price + min_tick_size
        else: # short
            raw_activation = entry_price - activation_offset
            # Quantize DOWN to nearest tick
            activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
            # Ensure activation is strictly < entry if percentage > 0
            if activation_percentage > 0 and activation_price >= entry_price:
                 activation_price = entry_price - min_tick_size # Move one tick away
            # For immediate activation (0%), set slightly below entry
            elif activation_percentage == 0:
                 activation_price = entry_price - min_tick_size

        if activation_price is None or activation_price <= 0:
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) is invalid for {symbol}.{RESET}")
             return False

        # 2. Calculate Trailing Stop Distance (based on activation price, as per Bybit V5 logic)
        # Distance = Activation Price * Callback Rate
        trailing_distance_raw = activation_price * callback_rate
        # Quantize distance UP to nearest tick size increment
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        # Enforce minimum distance (e.g., at least one tick)
        if trailing_distance < min_tick_size:
             lg.debug(f"Calculated TSL distance {trailing_distance} below min tick {min_tick_size}, adjusting.")
             trailing_distance = min_tick_size
        if trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance zero/negative ({trailing_distance}) for {symbol}.{RESET}")
             return False

        # --- Log Calculated Params ---
        tp_str = f"{take_profit_price:.{price_precision}f}" if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else "None"
        lg.info(f"Calculated TSL Params for {symbol} ({side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price: {activation_price:.{price_precision}f}")
        lg.info(f"  => Trailing Distance: {trailing_distance:.{price_precision}f} (MinTick: {min_tick_size})")
        lg.info(f"  Take Profit Price: {tp_str} (Will be set alongside TSL if provided)")

        # 3. Call helper to set TSL (and TP) via API
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None, # Explicitly None when setting TSL
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except (InvalidOperation, ValueError, TypeError) as calc_err:
        lg.error(f"{NEON_RED}Error during TSL parameter calculation for {symbol}: {calc_err}{RESET}", exc_info=True)
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Market Info ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping cycle.{RESET}")
        return
    # Update symbol to the one validated by get_market_info (handles variations)
    symbol = market_info['symbol']

    # --- 2. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe.")
         return

    # Determine kline limit based on what indicators need (use TradingAnalyzer helper?)
    # Or just use a generous limit.
    kline_limit = 500 # Ensure enough for longest indicator period + buffer
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50: # Basic check for minimum data
        lg.error(f"{NEON_RED}Insufficient kline data fetched for {symbol} (fetched {len(klines_df)}). Skipping cycle.{RESET}")
        return

    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
         lg.warning(f"{NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close as fallback.{RESET}")
         try:
             # Use Decimal for last close as well
             last_close_val = klines_df['close'].iloc[-1] # Already numeric from fetch_klines
             if pd.notna(last_close_val) and last_close_val > 0:
                  current_price = Decimal(str(last_close_val))
                  lg.info(f"Using last close price: {current_price}")
             else:
                 lg.error(f"{NEON_RED}Last close price is invalid ({last_close_val}). Cannot proceed without current price.{RESET}")
                 return
         except (IndexError, ValueError, TypeError, InvalidOperation) as e:
             lg.error(f"{NEON_RED}Error getting last close price: {e}. Cannot proceed.{RESET}")
             return

    orderbook_data = None
    # Fetch orderbook only if the indicator is enabled AND has weight in the active set
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    orderbook_weight = Decimal('0')
    try: orderbook_weight = Decimal(str(active_weights.get("orderbook", 0)))
    except InvalidOperation: pass
    if config.get("indicators",{}).get("orderbook", False) and orderbook_weight != 0:
         orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)
         if orderbook_data is None: lg.warning("Orderbook fetch failed, proceeding without it.")

    # --- 3. Analyze Data ---
    analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    if not analyzer.indicator_values:
         # This indicates a failure in _calculate_all_indicators or _update_latest_indicator_values
         lg.error(f"{NEON_RED}Indicator calculation or update failed for {symbol}. Skipping signal generation.{RESET}")
         return

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)

    # Calculate potential SL/TP based on *current price* as entry estimate for initial sizing/logging
    _, tp_potential, sl_potential_sizing = analyzer.calculate_entry_tp_sl(current_price, signal)
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr = analyzer.indicator_values.get("ATR") # Decimal or NaN

    # --- Log Analysis Summary ---
    lg.info(f"Current Price: {current_price:.{price_precision}f}")
    lg.info(f"ATR ({config.get('atr_period')}): {current_atr:.{price_precision+1}f}" if isinstance(current_atr, Decimal) else 'N/A')
    lg.info(f"Potential Initial SL (for sizing): {sl_potential_sizing if sl_potential_sizing else 'N/A'}")
    lg.info(f"Potential Initial TP: {tp_potential if tp_potential else 'N/A'}")
    tsl_enabled = config.get('enable_trailing_stop', False)
    be_enabled = config.get('enable_break_even', False)
    lg.info(f"Configured Protections: TSL={'Enabled' if tsl_enabled else 'Disabled'} | BE={'Enabled' if be_enabled else 'Disabled'}")

    # --- 4. Trading Logic ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading is disabled in config. Analysis complete for {symbol}.")
        # Log cycle end time even in analysis mode
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return # Stop here if trading disabled

    # --- Check Existing Position ---
    # This is crucial: get position status *before* deciding to open/close/manage
    open_position = get_open_position(exchange, symbol, lg) # Returns enhanced dict or None

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***")

            # --- Pre-Trade Checks ---
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative ({balance}).{RESET}")
                return
            if sl_potential_sizing is None:
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation for sizing failed (ATR invalid?).{RESET}")
                 return

            # Set Leverage (only for contracts)
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1)) # Ensure integer
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage to {leverage}x.{RESET}")
                         # Optional: Check if current leverage is already correct? Might be complex.
                         return # Abort if leverage setting fails critically
                else: lg.info(f"Leverage setting skipped (leverage configured as {leverage}).")
            else: lg.info(f"Leverage setting skipped (Spot market).")

            # Calculate Position Size
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_potential_sizing, # Use the SL calculated for sizing
                entry_price=current_price, # Use current price as entry estimate for sizing
                market_info=market_info,
                exchange=exchange,
                logger=lg
            )
            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Position size calculation failed or resulted in zero/negative size ({position_size}).{RESET}")
                return

            # --- Place Trade ---
            lg.info(f"==> Placing {signal} market order | Size: {position_size} <==")
            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                trade_signal=signal,
                position_size=position_size,
                market_info=market_info,
                logger=lg,
                reduce_only=False
            )

            # --- Post-Trade Actions (Confirmation & Protection) ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                lg.info(f"Attempting position confirmation for {symbol} after order {order_id}...")
                # Re-fetch position after delay
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    lg.info(f"{NEON_GREEN}Position Confirmed! Details logged above.{RESET}")
                    try:
                        # Get actual entry price from confirmed position
                        entry_price_actual = confirmed_position.get('entryPriceDecimal')
                        if not isinstance(entry_price_actual, Decimal) or entry_price_actual <= 0:
                             lg.warning(f"Could not get valid actual entry price from confirmed position. Using estimate: {current_price}")
                             entry_price_actual = current_price # Fallback, less accurate

                        lg.info(f"Actual Entry Price (from position): ~{entry_price_actual:.{price_precision}f}")

                        # --- Set Protection based on ACTUAL entry ---
                        lg.info("Calculating protection levels based on actual entry price...")
                        # Recalculate SL/TP using the actual entry price
                        _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                        protection_set_success = False
                        if config.get("enable_trailing_stop", False):
                             lg.info(f"Setting Trailing Stop Loss (Initial TP target: {tp_final})...")
                             # Pass the confirmed position dict which contains side, entry, posIdx etc.
                             protection_set_success = set_trailing_stop_loss(
                                 exchange=exchange,
                                 symbol=symbol,
                                 market_info=market_info,
                                 position_info=confirmed_position,
                                 config=config,
                                 logger=lg,
                                 take_profit_price=tp_final # Pass recalculated TP to set alongside TSL
                             )
                        else: # Set Fixed SL/TP
                             lg.info(f"Setting Fixed Stop Loss ({sl_final}) and Take Profit ({tp_final})...")
                             if sl_final or tp_final: # Only call if at least one is valid
                                 protection_set_success = _set_position_protection(
                                     exchange=exchange,
                                     symbol=symbol,
                                     market_info=market_info,
                                     position_info=confirmed_position, # Pass confirmed position
                                     logger=lg,
                                     stop_loss_price=sl_final,
                                     take_profit_price=tp_final
                                 )
                             else:
                                 lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation failed based on actual entry. No fixed protection will be set via API.{RESET}")
                                 # Consider this case: should the trade proceed without protection? Depends on risk tolerance.

                        if protection_set_success:
                             lg.info(f"{NEON_GREEN}=== TRADE ENTRY & INITIAL PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                        else:
                             # Protection setting failed AFTER position confirmed open
                             lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET INITIAL PROTECTION ({symbol} {signal}) ===")
                             lg.warning(f"{NEON_YELLOW}!!! POSITION IS OPEN WITHOUT SL/TP/TSL - MANUAL MONITORING REQUIRED !!!{RESET}")

                    except Exception as post_trade_err:
                         lg.error(f"{NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                         lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")
                else:
                    # Position NOT confirmed after delay
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM active position after {confirm_delay}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have been rejected, failed to fill, or API/network delay. Manual investigation required! Check exchange UI.{RESET}")
                    # Should we try to cancel the order here? Risky if partially filled.
            else:
                # place_trade function returned None or order without ID
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). Order not placed successfully. See previous logs. ===")
        else: # signal == HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No action taken.")

    # --- Scenario 2: Existing Open Position ---
    else: # open_position is not None (already logged by get_open_position)
        pos_side = open_position.get('side', 'unknown') # Should be 'long' or 'short'
        if pos_side == 'unknown':
             lg.error(f"Position found for {symbol}, but side is unknown. Cannot manage.")
             return

        lg.info(f"Managing existing {pos_side.upper()} position for {symbol}.")

        # --- Check for Exit Signal ---
        # Close position if the new signal opposes the existing position direction
        exit_signal_triggered = (pos_side == 'long' and signal == "SELL") or \
                                (pos_side == 'short' and signal == "BUY")

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position. Closing position... ***{RESET}")
            try:
                # Determine close side and get size from position info
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close_str = open_position.get('contracts') or open_position.get('info',{}).get('size')
                if size_to_close_str is None:
                    raise ValueError("Cannot determine position size from position_info to close.")

                size_to_close = abs(Decimal(str(size_to_close_str)))
                if size_to_close <= 0:
                    raise ValueError(f"Position size '{size_to_close_str}' is zero or invalid.")

                lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                close_order = place_trade(
                    exchange=exchange,
                    symbol=symbol,
                    trade_signal=close_side_signal,
                    position_size=size_to_close,
                    market_info=market_info,
                    logger=lg,
                    reduce_only=True # IMPORTANT: Set reduceOnly to True
                )

                if close_order and close_order.get('id'):
                    lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                    # Optional: Wait and confirm position is actually closed by fetching again
                    # time.sleep(POSITION_CONFIRM_DELAY_SECONDS)
                    # closed_pos_check = get_open_position(exchange, symbol, lg)
                    # if closed_pos_check is None: lg.info("Position confirmed closed.")
                    # else: lg.warning("Position closure not yet confirmed.")
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check/intervention required!{RESET}")

            except (ValueError, InvalidOperation, TypeError) as close_size_err:
                 lg.error(f"{NEON_RED}Error determining size for closing position {symbol}: {close_size_err}{RESET}")
                 lg.warning(f"{NEON_YELLOW}Manual intervention likely needed to close the position!{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error closing position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")

        else: # Hold signal or signal matches position direction
            lg.info(f"Signal ({signal}) allows holding the existing {pos_side} position.")

            # --- Manage Existing Position (BE, TSL checks/updates) ---
            # Check if Trailing Stop is currently active on the exchange for this position
            is_tsl_active_on_exchange = False
            try:
                 # Check the Decimal value stored in the enhanced position dict
                 tsl_dist = open_position.get('trailingStopLossDistanceDecimal')
                 if isinstance(tsl_dist, Decimal) and tsl_dist > 0:
                      is_tsl_active_on_exchange = True
                      lg.debug(f"Trailing Stop Loss detected as active on exchange (Distance: {tsl_dist}).")
            except Exception as tsl_check_err:
                 lg.warning(f"Could not reliably check if TSL is active on exchange: {tsl_check_err}")

            # --- Break-Even Check (Only if BE enabled AND TSL is NOT active on exchange) ---
            if config.get("enable_break_even", False) and not is_tsl_active_on_exchange:
                lg.debug(f"Checking Break-Even conditions for {symbol} (TSL not active)...")
                try:
                    entry_price = open_position.get('entryPriceDecimal')
                    if not isinstance(entry_price, Decimal) or entry_price <= 0:
                        raise ValueError("Invalid entry price in position info for BE check.")
                    if not isinstance(current_atr, Decimal) or current_atr <= 0:
                        lg.warning("BE Check skipped: Invalid ATR for calculation.")
                    else:
                        profit_target_atr_mult = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        if offset_ticks < 0: offset_ticks = 0 # Ensure non-negative offset

                        # Calculate current profit/loss in price points
                        price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                        # Calculate profit in terms of ATR multiples
                        profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('inf') if price_diff > 0 else Decimal('-inf')

                        lg.debug(f"BE Check: Entry={entry_price:.{price_precision}f}, Current={current_price:.{price_precision}f}, "
                                 f"Price Diff={price_diff:.{price_precision}f}, ATR={current_atr:.{price_precision+1}f}, "
                                 f"Profit ATRs={profit_in_atr:.2f}, Target ATRs={profit_target_atr_mult}")

                        # Check if profit target is reached
                        if profit_in_atr >= profit_target_atr_mult:
                            lg.info(f"Break-Even profit target reached ({profit_in_atr:.2f} >= {profit_target_atr_mult} ATRs).")
                            # Calculate target BE Stop Loss price
                            tick_offset_value = min_tick_size * offset_ticks
                            if pos_side == 'long':
                                be_stop_price_target = (entry_price + tick_offset_value).quantize(min_tick_size, rounding=ROUND_UP)
                            else: # short
                                be_stop_price_target = (entry_price - tick_offset_value).quantize(min_tick_size, rounding=ROUND_DOWN)

                            # Get current SL price as Decimal (if it exists and is valid)
                            current_sl_price = open_position.get('stopLossPriceDecimal') # Already Decimal or None

                            # Determine if we need to update the SL
                            update_be_sl = False
                            if be_stop_price_target is not None and be_stop_price_target > 0:
                                if current_sl_price is None:
                                    update_be_sl = True # No SL exists, set BE SL
                                    lg.info("BE triggered: No current SL found. Setting BE SL.")
                                # For long, update if new BE SL is higher than current SL
                                elif pos_side == 'long' and be_stop_price_target > current_sl_price:
                                    update_be_sl = True
                                    lg.info(f"BE triggered: Target BE SL {be_stop_price_target} is higher than current SL {current_sl_price}.")
                                # For short, update if new BE SL is lower than current SL
                                elif pos_side == 'short' and be_stop_price_target < current_sl_price:
                                    update_be_sl = True
                                    lg.info(f"BE triggered: Target BE SL {be_stop_price_target} is lower than current SL {current_sl_price}.")
                                else:
                                    # BE triggered, but current SL is already better or equal
                                    lg.debug(f"BE Triggered, but current SL ({current_sl_price}) is already better than or equal to target ({be_stop_price_target}). No update needed.")
                            else:
                                lg.error("Calculated BE Stop Loss target price is invalid. Cannot update.")

                            # If update is needed, call the protection function
                            if update_be_sl:
                                lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price_target} ***{RESET}")
                                # Preserve existing TP if possible
                                current_tp_price = open_position.get('takeProfitPriceDecimal') # Already Decimal or None

                                # Call the protection helper ONLY setting the SL (and preserving TP if exists)
                                success = _set_position_protection(
                                    exchange=exchange,
                                    symbol=symbol,
                                    market_info=market_info,
                                    position_info=open_position, # Pass the full position info
                                    logger=lg,
                                    stop_loss_price=be_stop_price_target, # Set the new BE SL
                                    take_profit_price=current_tp_price, # Preserve existing TP
                                    # Ensure TSL params are None when setting fixed SL
                                    trailing_stop_distance=None,
                                    tsl_activation_price=None
                                )
                                if success:
                                     lg.info(f"{NEON_GREEN}Break-Even SL set successfully via API.{RESET}")
                                     # Optional: Update local position_info state? Or rely on next fetch.
                                else:
                                     lg.error(f"{NEON_RED}Failed to set Break-Even SL via API.{RESET}")
                        else:
                            lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {profit_target_atr_mult} ATRs).")
                except (ValueError, InvalidOperation, TypeError) as be_calc_err:
                    lg.error(f"{NEON_RED}Error during break-even calculation ({symbol}): {be_calc_err}{RESET}", exc_info=True)
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Unexpected error during break-even check ({symbol}): {be_err}{RESET}", exc_info=True)
            elif is_tsl_active_on_exchange:
                 lg.info(f"Break-even check skipped: Trailing Stop Loss is already active on the exchange.")
            else: # BE disabled in config
                 lg.debug(f"Break-even check skipped: Disabled in config.")

            # --- Placeholder for other potential management logic ---
            # E.g., Adjusting TP based on new analysis? (Use caution)
            # E.g., Re-trying failed TSL setup from previous cycle?


    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis & Trade Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY # Allow main to update globals if config reloaded

    # Use a generic logger for initialization phase
    setup_logger("init") # Ensure init logger is set up
    init_logger = logging.getLogger("init") # Get the init logger instance

    init_logger.info(f"--- Starting LiveXY Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # Load configuration first
    try:
        CONFIG = load_config(CONFIG_FILE)
        QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Update global QUOTE_CURRENCY
        init_logger.info(f"Configuration loaded successfully from {CONFIG_FILE}")
        init_logger.info(f"Using Quote Currency: {QUOTE_CURRENCY}")
        init_logger.debug(f"Full Config: {CONFIG}") # Log full config at debug level
    except Exception as cfg_err:
        init_logger.critical(f"{NEON_RED}Failed to load configuration: {cfg_err}. Exiting.{RESET}", exc_info=True)
        return

    # Log library versions
    try: pandas_ta_version = ta.version()
    except AttributeError: pandas_ta_version = "N/A" # Older versions might not have version()
    init_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={pandas_ta_version}, Numpy={np.__version__}")

    # --- Safety Confirmation for Live Trading ---
    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED in {CONFIG_FILE} !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet) environment.{RESET}")
         else:
              init_logger.warning(f"{NEON_RED}!!! CAUTION: Using REAL MONEY environment !!!{RESET}")
         # Display key risk parameters
         risk_pct = CONFIG.get('risk_per_trade', 0) * 100
         leverage = CONFIG.get('leverage', 0)
         tsl_status = 'ON' if CONFIG.get('enable_trailing_stop') else 'OFF'
         be_status = 'ON' if CONFIG.get('enable_break_even') else 'OFF'
         init_logger.warning(f"Key Settings: Risk={risk_pct:.2f}%, Leverage={leverage}x, TSL={tsl_status}, BE={be_status}")
         try:
             # Require user confirmation
             confirm = input(f">>> Review ALL settings above and in {CONFIG_FILE}. "
                             f"Type '{NEON_GREEN}confirm{RESET}' to proceed with live trading, or press Enter/Ctrl+C to abort: ")
             if confirm.lower() != 'confirm':
                  init_logger.info("User aborted live trading confirmation. Exiting.")
                  return
             init_logger.info("User confirmed live trading settings. Proceeding...")
         except KeyboardInterrupt:
              init_logger.info("User aborted via Ctrl+C. Exiting.")
              return
    else:
         init_logger.info(f"{NEON_YELLOW}Trading is disabled ('enable_trading': false). Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange ---
    init_logger.info("Initializing CCXT exchange...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize CCXT exchange. Please check API keys, permissions, network, and logs. Exiting.{RESET}")
        return
    init_logger.info(f"Exchange '{exchange.id}' initialized successfully.")

    # --- Get Target Symbol and Interval ---
    target_symbol = None
    while target_symbol is None:
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter trading symbol (e.g., BTC/USDT, ETH/USDT:USDT): {RESET}").strip().upper()
            if not symbol_input_raw: continue
            # Basic normalization (replace common separators)
            symbol_input = symbol_input_raw.replace('-', '/').replace(':', '/')
            if '/' not in symbol_input: # Add default quote if missing separator
                 symbol_input = f"{symbol_input}/{QUOTE_CURRENCY}"
                 init_logger.info(f"Assuming quote currency: {symbol_input}")

            init_logger.info(f"Validating symbol '{symbol_input}' on {exchange.id}...")
            market_info = get_market_info(exchange, symbol_input, init_logger)

            if market_info:
                target_symbol = market_info['symbol'] # Use the exact symbol from market info
                market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                break # Exit loop once symbol is valid
            else:
                # Try common variations if initial fails (e.g., adding :QUOTE for linear perps)
                variations_to_try = []
                if '/' in symbol_input and ':' not in symbol_input:
                    base, quote = symbol_input.split('/', 1)
                    variations_to_try.append(f"{base}/{quote}:{quote}") # e.g., BTC/USDT:USDT
                    # Add PERP variation?
                    # variations_to_try.append(f"{symbol_input}PERP") # e.g., BTC/USDTPERP

                found_variation = False
                if variations_to_try:
                     init_logger.info(f"Symbol '{symbol_input}' not found directly. Trying variations: {variations_to_try}")
                     for sym_var in variations_to_try:
                         market_info = get_market_info(exchange, sym_var, init_logger)
                         if market_info:
                             target_symbol = market_info['symbol']
                             market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                             init_logger.info(f"Validated Symbol using variation: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                             found_variation = True
                             break # Exit inner loop
                if found_variation: break # Exit outer loop
                else: init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' and variations not found or invalid on {exchange.id}. Please check the symbol.{RESET}")

        except Exception as e:
            init_logger.error(f"Error during symbol validation: {e}", exc_info=True)
            # Loop continues to ask for symbol

    # --- Get Interval ---
    selected_interval = None
    while selected_interval is None:
        default_interval = CONFIG.get('interval', '5') # Get default from loaded config
        interval_input = input(f"{NEON_YELLOW}Enter analysis interval [{'/'.join(VALID_INTERVALS)}] (default: {default_interval}): {RESET}").strip()
        if not interval_input: interval_input = default_interval

        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            selected_interval = interval_input
            # Update config in memory for this run (doesn't save back to file)
            CONFIG["interval"] = selected_interval
            ccxt_tf = CCXT_INTERVAL_MAP[selected_interval]
            init_logger.info(f"Using interval: {selected_interval} (Maps to CCXT: {ccxt_tf})")
            break # Exit loop
        else:
            init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Please choose from: {', '.join(VALID_INTERVALS)}.{RESET}")

    # --- Setup Symbol-Specific Logger ---
    symbol_logger = setup_logger(target_symbol) # Use the validated symbol
    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    # Log key parameters again to symbol logger
    risk_pct = CONFIG.get('risk_per_trade', 0) * 100
    leverage = CONFIG.get('leverage', 0)
    tsl_status = 'ON' if CONFIG.get('enable_trailing_stop') else 'OFF'
    be_status = 'ON' if CONFIG.get('enable_break_even') else 'OFF'
    trading_status = 'ENABLED' if CONFIG.get('enable_trading') else 'DISABLED (Analysis Only)'
    symbol_logger.info(f"Config: Risk={risk_pct:.2f}%, Lev={leverage}x, TSL={tsl_status}, BE={be_status}, Trading={trading_status}")
    symbol_logger.info(f"Using Weight Set: '{CONFIG.get('active_weight_set', 'default')}'")

    # --- Main Trading Loop ---
    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Start: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")

            try:
                # --- Optional: Reload Config Each Cycle ---
                # Consider the performance impact vs. need for dynamic updates
                # current_config = load_config(CONFIG_FILE)
                # QUOTE_CURRENCY = current_config.get("quote_currency", "USDT")
                # config_to_use = current_config
                # --- Use static config loaded at start ---
                config_to_use = CONFIG

                # --- Run Analysis and Trading Logic ---
                analyze_and_trade_symbol(exchange, target_symbol, config_to_use, symbol_logger)

            # --- Graceful Handling of Common CCXT/Network Errors ---
            except ccxt.RateLimitExceeded as e:
                wait_time = max(LOOP_DELAY_SECONDS, e.params.get('retry-after', 60)) # Wait at least loop delay or suggested time
                symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting {wait_time:.1f}s...{RESET}")
                time.sleep(wait_time)
            except (ccxt.NetworkError, requests.exceptions.RequestException, ccxt.RequestTimeout) as e:
                symbol_logger.error(f"{NEON_RED}Network error encountered: {e}. Waiting {RETRY_DELAY_SECONDS*3}s before next cycle...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                symbol_logger.critical(f"{NEON_RED}CRITICAL Authentication Error: {e}. API keys may be invalid or expired. Stopping bot.{RESET}")
                break # Stop the bot on auth errors
            except ccxt.ExchangeNotAvailable as e:
                symbol_logger.error(f"{NEON_RED}Exchange unavailable (e.g., temporary outage): {e}. Waiting 60s...{RESET}")
                time.sleep(60)
            except ccxt.OnMaintenance as e:
                symbol_logger.error(f"{NEON_RED}Exchange is under maintenance: {e}. Waiting 5 minutes...{RESET}")
                time.sleep(300)
            # Catch specific errors from analysis/trade logic if needed
            # except ValueError as ve: # Example
            #     symbol_logger.error(f"Data validation error: {ve}", exc_info=True)
            #     time.sleep(LOOP_DELAY_SECONDS) # Wait before next attempt
            except Exception as loop_error:
                # Catch any other unexpected errors within the loop
                symbol_logger.error(f"{NEON_RED}Unhandled error in main loop: {loop_error}{RESET}", exc_info=True)
                # Wait a bit longer after unexpected errors
                time.sleep(LOOP_DELAY_SECONDS * 2)

            # --- Calculate Sleep Time ---
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as critical_error:
        # Catch errors outside the main loop (e.g., during init after confirmation)
        init_logger.critical(f"{NEON_RED}Critical unhandled error in main execution: {critical_error}{RESET}", exc_info=True)
    finally:
        # --- Shutdown Sequence ---
        shutdown_msg = f"--- LiveXY Bot for {target_symbol or 'N/A'} Stopping ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---"
        # Log shutdown to both init and symbol loggers if they exist
        if 'init_logger' in locals() and isinstance(init_logger, logging.Logger): init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger): symbol_logger.info(shutdown_msg)

        # Close exchange connection if initialized
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Closing CCXT exchange connection...")
                exchange.close()
                init_logger.info("Exchange connection closed.")
            except Exception as close_err:
                init_logger.error(f"Error closing exchange connection: {close_err}")

        logging.shutdown() # Flush and close all handlers
        print(f"\n{NEON_YELLOW}Bot stopped.{RESET}")


if __name__ == "__main__":
    # This block now just runs the main function directly.
    # The thought process included writing the file, but the final code should just execute.
    main()
