# livexy.py
# Enhanced version focusing on stop-loss/take-profit mechanisms, including break-even logic.

import json
import logging
import math
import os
import re  # Imported for parsing rate limit messages
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
getcontext().prec = 28  # Increased precision for financial calculations
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
TIMEZONE = ZoneInfo("America/Chicago")  # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Delay between retries
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]  # Intervals supported by the bot's logic
CCXT_INTERVAL_MAP = {  # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]  # HTTP status codes considered retryable
# Default periods (can be overridden by config.json)
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
DEFAULT_FIB_WINDOW = 100
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 15  # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY = 10  # Seconds to wait after placing order before checking position status
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


def load_config(filepath: str) -> dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
    and ensuring all default keys are present.
    """
    default_config = {
        "interval": "5",  # Default to 5 minute interval (string format for our logic)
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
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True,  # Default to enabling TSL (exchange TSL)
        # Trail distance as a percentage of the activation/high-water-mark price (e.g., 0.5%)
        # Bybit API expects absolute distance, this percentage is used for calculation.
        "trailing_stop_callback_rate": 0.002,  # Example: 0.5% trail distance relative to entry/activation
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry.
        "trailing_stop_activation_percentage": 0,  # Example: Activate when 0.3% in profit
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
            # Return default config anyway if file creation fails
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added during the update
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
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Optional: Handle type mismatches if needed
        # elif type(default_value) != type(updated_config.get(key)):
        #     print(f"Warning: Type mismatch for key '{key}'. Default: {type(default_value)}, Loaded: {type(updated_config.get(key))}. Using default.")
        #     updated_config[key] = default_value
    return updated_config


CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")  # Get quote currency from config
console_log_level = logging.INFO  # Default console level, can be changed in main()


# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    # Clean symbol for filename (replace / and : which are invalid in filenames)
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"livexy_bot_{safe_symbol}"  # Use safe symbol in logger name
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        # Ensure existing handlers have the correct level (e.g., if changed dynamically)
        for handler in logger.handlers:
             if isinstance(handler, logging.StreamHandler):
                  handler.setLevel(console_log_level)  # Update console level
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
        file_handler.setLevel(logging.DEBUG)  # Log everything to the file
        logger.addHandler(file_handler)
    except Exception:
        pass

    # Stream Handler (console, writes INFO and above by default for cleaner output)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'  # Add timestamp format to console
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level based on global variable
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs)
    logger.propagate = False

    return logger


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with error handling."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'linear',  # Assume linear contracts (USDT margined) - adjust if needed
                'adjustForTimeDifference': True,  # Auto-sync time with server
                # Connection timeouts (milliseconds)
                'fetchTickerTimeout': 10000,  # 10 seconds
                'fetchBalanceTimeout': 15000,  # 15 seconds
                'createOrderTimeout': 20000,  # Timeout for placing orders
                'fetchOrderTimeout': 15000,  # Timeout for fetching orders
                'fetchPositionsTimeout': 15000,  # Timeout for fetching positions
                # Add any exchange-specific options if needed
                # 'recvWindow': 10000, # Example for Binance if needed
                'brokerId': 'livexyBot',  # Example: Add a broker ID for Bybit if desired
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
        account_type_to_test = 'CONTRACT'  # Or 'UNIFIED' based on your account
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            # Use our enhanced balance function
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, logger)
            if balance_decimal is not None:
                 logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 logger.warning(f"{NEON_YELLOW}Initial balance fetch returned None. Check API permissions/account type if trading fails.{RESET}")
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None  # Critical failure, cannot proceed
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
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
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

        # 1. Try 'last' price first, ensure it's positive
        if last_price is not None:
            try:
                last_decimal = Decimal(str(last_price))
                if last_decimal > 0:
                    price = last_decimal
                    lg.debug(f"Using 'last' price for {symbol}: {price}")
                else:
                    lg.warning(f"'Last' price ({last_decimal}) is not positive for {symbol}.")
            except Exception as e:
                lg.warning(f"Could not parse 'last' price ({last_price}) for {symbol}: {e}")

        # 2. If 'last' is invalid, try bid/ask midpoint
        if price is None and bid_price is not None and ask_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                ask_decimal = Decimal(str(ask_price))
                if bid_decimal > 0 and ask_decimal > 0:
                    # Ensure bid is not higher than ask (can happen in volatile/illiquid moments)
                    if bid_decimal <= ask_decimal:
                        price = (bid_decimal + ask_decimal) / 2
                        lg.debug(f"Using bid/ask midpoint for {symbol}: {price} (Bid: {bid_decimal}, Ask: {ask_decimal})")
                    else:
                        lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' as fallback.")
                        price = ask_decimal  # Use ask as a safer fallback in this case
                else:
                    lg.warning(f"Bid ({bid_decimal}) or Ask ({ask_decimal}) price is not positive for {symbol}.")
            except Exception as e:
                lg.warning(f"Could not parse bid/ask prices ({bid_price}, {ask_price}) for {symbol}: {e}")

        # 3. If midpoint fails or wasn't used, try ask price (potentially better for market buy estimate)
        if price is None and ask_price is not None:
            try:
                ask_decimal = Decimal(str(ask_price))
                if ask_decimal > 0:
                    price = ask_decimal
                    lg.warning(f"Using 'ask' price as fallback for {symbol}: {price}")
                else:
                    lg.warning(f"'Ask' price ({ask_decimal}) is not positive for {symbol}.")
            except Exception as e:
                lg.warning(f"Could not parse 'ask' price ({ask_price}) for {symbol}: {e}")

        # 4. If ask fails, try bid price (potentially better for market sell estimate)
        if price is None and bid_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                if bid_decimal > 0:
                    price = bid_decimal
                    lg.warning(f"Using 'bid' price as fallback for {symbol}: {price}")
                else:
                    lg.warning(f"'Bid' price ({bid_decimal}) is not positive for {symbol}.")
            except Exception as e:
                lg.warning(f"Could not parse 'bid' price ({bid_price}) for {symbol}: {e}")

        # --- Final Check ---
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
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if ohlcv is not None and len(ohlcv) > 0:  # Basic check if data was returned and not empty
                    break  # Success
                else:
                    lg.warning(f"fetch_ohlcv returned {type(ohlcv)} (length {len(ohlcv) if ohlcv is not None else 'N/A'}) for {symbol} (Attempt {attempt + 1}). Retrying...")
                    # Optional: Add a small delay even on None/empty return if it might be transient
                    time.sleep(1)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                    raise e  # Re-raise the last error
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 5  # Default wait time
                try:
                     # Try to parse recommended wait time from Bybit/other exchange messages
                     if 'try again in' in str(e).lower():
                         wait_time_ms_str = re.search(r'try again in (\d+)ms', str(e).lower())
                         if wait_time_ms_str:
                             wait_time = int(wait_time_ms_str.group(1)) / 1000
                             wait_time = max(1, int(wait_time + 1))  # Add a buffer and ensure minimum 1s
                     elif 'rate limit' in str(e).lower():  # Generic rate limit message
                         # Try to extract number before 'ms' or 's'
                         match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                         if match:
                             num = int(match.group(1))
                             unit = match.group(2)
                             wait_time = num / 1000 if unit == 'ms' else num
                             wait_time = max(1, int(wait_time + 1))
                except Exception:
                     pass  # Use default if parsing fails
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                # Depending on the error, might not be retryable
                raise e  # Re-raise non-network errors immediately

        if not ohlcv:  # Check if list is empty or still None after retries
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} immediately after creation.{RESET}")
            return df

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid timestamps
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame empty after timestamp conversion for {symbol} {timeframe}.{RESET}")
            return df
        df.set_index('timestamp', inplace=True)

        # Ensure numeric types, coerce errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with any NaN in critical price columns
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Drop rows with non-positive close price (invalid data)
        df = df[df['close'] > 0]
        # Optional: Drop rows with zero volume if it indicates bad data
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

    except ccxt.NetworkError as e:  # Catch error if retries fail
        lg.error(f"{NEON_RED}Network error fetching klines for {symbol} after retries: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error processing klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> dict | None:
    """Fetch orderbook data using ccxt with retries and basic validation."""
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
                 return orderbook  # Return the empty book, signal generation needs to handle this
            else:
                 # Looks valid
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5  # Default wait time
            try:
                 if 'try again in' in str(e).lower():
                     wait_time_ms_str = re.search(r'try again in (\d+)ms', str(e).lower())
                     if wait_time_ms_str:
                         wait_time = int(wait_time_ms_str.group(1)) / 1000
                         wait_time = max(1, int(wait_time + 1))
                 elif 'rate limit' in str(e).lower():
                     match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                     if match:
                         num = int(match.group(1))
                         unit = match.group(2)
                         wait_time = num / 1000 if unit == 'ms' else num
                         wait_time = max(1, int(wait_time + 1))
            except Exception: pass
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
        # Prioritize exact match or expected patterns
        cfg = self.config  # Shortcut
        # Ensure std dev for BB is float for formatting
        bb_std_dev = float(cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV))
        expected_patterns = {
            "ATR": [f"ATRr_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}", f"ATR_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}"],  # Added ATR_ variation
            "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_0.015"],  # Default const suffix
            "Williams_R": [f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP", "VWAP_D"],  # Handle potential suffix like _D for daily reset
            "PSAR_long": [f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "BB_Middle": [f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "BB_Upper": [f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "Volume_MA": [f"VOL_SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"]  # Custom name
        }
        patterns = expected_patterns.get(base_name, [])
        for pattern in patterns:
            if pattern in result_df.columns:
                return pattern
            # Check variation without const suffix (e.g., CCI)
            base_pattern_parts = pattern.split('_')
            if len(base_pattern_parts) > 2 and base_pattern_parts[-1].replace('.', '').isdigit():
                pattern_no_suffix = '_'.join(base_pattern_parts[:-1])
                if pattern_no_suffix in result_df.columns:
                    self.logger.debug(f"Found column '{pattern_no_suffix}' for base '{base_name}' (without const suffix).")
                    return pattern_no_suffix

        # Check for common parameter-less suffix variations if specific pattern not found
        for pattern in patterns:
             base_pattern = pattern.split('_')[0]  # e.g., "ATRr" -> "ATRr"
             if base_pattern in result_df.columns:
                  self.logger.debug(f"Found column '{base_pattern}' for base '{base_name}' (parameter-less variation).")
                  return base_pattern
             # Try base name itself (e.g., "CCI" if "CCI_20_0.015" not found)
             if base_name.upper() in result_df.columns:
                  self.logger.debug(f"Found column '{base_name.upper()}' for base '{base_name}'.")
                  return base_name.upper()

        # Fallback: search for base name (case-insensitive) if specific pattern not found
        # This is less reliable but might catch variations
        for col in result_df.columns:
            # More specific check: starts with base name + underscore (e.g., "CCI_")
            if col.lower().startswith(base_name.lower() + "_"):
                 self.logger.debug(f"Found column '{col}' for base '{base_name}' using prefix fallback search.")
                 return col
            # Less specific check: contains base name
            if base_name.lower() in col.lower():
                self.logger.debug(f"Found column '{col}' for base '{base_name}' using basic fallback search.")
                return col

        self.logger.warning(f"Could not find column name for indicator '{base_name}' in DataFrame columns: {result_df.columns.tolist()}")
        return None

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check for sufficient data length
        periods_needed = []
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        if True: periods_needed.append(cfg.get("atr_period", DEFAULT_ATR_PERIOD))  # Always calc ATR
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

        min_required_data = max(periods_needed) + 20 if periods_needed else 50  # Add buffer

        if len(self.df) < min_required_data:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate or NaN.{RESET}")
             # Continue calculation, but expect NaNs

        try:
            df_calc = self.df.copy()  # Work on a copy

            # --- Calculate indicators using pandas_ta ---
            indicators_config = self.config.get("indicators", {})

            # Always calculate ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # Calculate other indicators based on config flags
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
                bb_std_float = float(bb_std)
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
                df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)  # Use ta.sma directly
                self.ta_column_names["Volume_MA"] = vol_ma_col_name

            # Assign the df with calculated indicators back to self.df
            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e:
            self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} (check pandas_ta method name/version?): {e}{RESET}", exc_info=True)
             # self.df remains the original data without calculated indicators
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)
            # Decide how to handle - clear df or keep original? Keep original for now.

        # Note: _update_latest_indicator_values is called after this in __init__

    def _update_latest_indicator_values(self) -> None:
        """Updates the indicator_values dict with the latest float values from self.df."""
        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame is empty for {self.symbol}.")
            self.indicator_values = dict.fromkeys(list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"], np.nan)
            return
        # Check if the last row contains any non-NaN values before proceeding
        try:
            if self.df.iloc[-1].isnull().all():
                self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: Last row of DataFrame contains all NaNs for {self.symbol}.{RESET}")
                self.indicator_values = dict.fromkeys(list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"], np.nan)
                return
        except IndexError:
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty or too short.")
             self.indicator_values = dict.fromkeys(list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"], np.nan)  # Reset values
             return

        try:
            latest = self.df.iloc[-1]
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
                    # Log only if the indicator was supposed to be calculated (i.e., key exists)
                    if key in self.ta_column_names:  # Check if key was attempted
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

            self.indicator_values = updated_values
            # Filter out NaN for debug log brevity
            valid_values = {k: f"{v:.5f}" if isinstance(v, float) else v for k, v in self.indicator_values.items() if pd.notna(v)}
            self.logger.debug(f"Latest indicator float values updated for {self.symbol}: {valid_values}")

        except IndexError:  # Catch again just in case df became empty between checks
            self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty or too short.")
            self.indicator_values = dict.fromkeys(list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"], np.nan)  # Reset values
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            self.indicator_values = dict.fromkeys(list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"], np.nan)  # Reset values

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
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
                    min_tick = None  # Disable quantization if tick size is invalid

                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level: High - (Range * Pct) for downtrend assumption (standard)
                    # Or Low + (Range * Pct) for uptrend assumption
                    # Using High - diff * level assumes retracement from High towards Low
                    level_price = (high - (diff * Decimal(str(level_pct))))

                    # Quantize the result to the market's price precision (using tick size) if possible
                    if min_tick:
                        # Use ROUND_DOWN for fib levels below 50%, ROUND_UP for levels above 50%? Or consistent rounding?
                        # Let's use ROUND_DOWN consistently for now for simplicity, as it's often used for support/resistance.
                        level_price_quantized = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    else:
                        level_price_quantized = level_price  # Use raw value if quantization fails

                    levels[level_name] = level_price_quantized
            else:
                 # If high == low, all levels will be the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}. Setting levels to high/low.")
                 min_tick = self.get_min_tick_size()
                 if min_tick > 0:
                     level_price_quantized = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 else:
                     level_price_quantized = high  # Use raw if quantization fails
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { {k: str(v) for k, v in levels.items()} }")  # Log as strings
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
                precision = abs(min_tick.normalize().as_tuple().exponent)
                # self.logger.debug(f"Derived price precision {precision} from min tick size {min_tick} for {self.symbol}")
                return precision
            else:
                 self.logger.warning(f"Min tick size ({min_tick}) is invalid for {self.symbol}. Attempting fallback precision methods.")

        except Exception as e:
            self.logger.warning(f"Error getting/using min tick size for precision derivation ({self.symbol}): {e}. Attempting fallback methods.")

        # Fallback: Use ccxt market['precision']['price'] if it's an integer (decimal places)
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if isinstance(price_precision_val, int):
                 self.logger.debug(f"Using price precision from market_info['precision']['price']: {price_precision_val}")
                 return price_precision_val
            # If precision.price is float/string (might be tick size), try deriving from it
            elif isinstance(price_precision_val, (float, str)):
                try:
                    tick_from_prec = Decimal(str(price_precision_val))
                    if tick_from_prec > 0:
                        precision = abs(tick_from_prec.normalize().as_tuple().exponent)
                        self.logger.debug(f"Derived price precision {precision} from market_info['precision']['price'] (interpreted as tick size {tick_from_prec}) for {self.symbol}")
                        return precision
                except Exception:
                     self.logger.debug(f"Could not derive precision from market_info['precision']['price'] value: {price_precision_val}")
        except Exception as e:
            self.logger.warning(f"Error accessing market_info['precision']['price'] for precision: {e}")

        # Fallback: Infer from last close price format if tick size failed
        try:
            last_close = self.indicator_values.get("Close")  # Uses float value
            if last_close and pd.notna(last_close) and last_close > 0:
                try:
                    s_close = format(Decimal(str(last_close)), 'f')  # Format to avoid scientific notation
                    if '.' in s_close:
                        precision = len(s_close.split('.')[-1])
                        self.logger.debug(f"Inferring price precision from last close price ({s_close}) as {precision} for {self.symbol}.")
                        return precision
                    else:
                        return 0  # No decimal places
                except Exception as e_close:
                    self.logger.warning(f"Error inferring precision from close price {last_close}: {e_close}")
        except Exception as e_outer:
            self.logger.warning(f"Could not access/parse close price for precision fallback: {e_outer}")

        # Default fallback precision
        default_precision = 4  # Common default for USDT pairs, adjust if needed
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        try:
            # 1. Prioritize market['info']['tickSize'] (Bybit V5 specific)
            info_dict = self.market_info.get('info', {})
            bybit_tick_size = info_dict.get('tickSize')
            if bybit_tick_size is not None:
                 try:
                      tick_size_bybit = Decimal(str(bybit_tick_size))
                      if tick_size_bybit > 0:
                          # self.logger.debug(f"Using tick size from market_info['info']['tickSize']: {tick_size_bybit} for {self.symbol}")
                          return tick_size_bybit
                 except Exception:
                      self.logger.debug(f"Could not parse info['tickSize'] '{bybit_tick_size}' as Decimal for {self.symbol}.")

            # 2. Check CCXT standard market['precision']['price'] (can be tick size)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')  # This is often the tick size
            if price_precision_val is not None:
                try:
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0:
                        # self.logger.debug(f"Using tick size from precision.price: {tick_size} for {self.symbol}")
                        return tick_size
                except Exception:
                    self.logger.debug(f"Could not parse precision.price '{price_precision_val}' as Decimal for {self.symbol}. Trying other methods.")

            # 3. Fallback: Check limits.price.min (sometimes this represents tick size, less reliable)
            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                try:
                    min_tick_from_limit = Decimal(str(min_price_val))
                    # Check if it looks like a tick size (power of 10) - heuristic
                    if min_tick_from_limit > 0 and (min_tick_from_limit == 1 or math.log10(min_tick_from_limit) % 1 == 0):
                        self.logger.debug(f"Potentially using tick size from limits.price.min: {min_tick_from_limit} for {self.symbol}")
                        return min_tick_from_limit
                except Exception:
                    self.logger.debug(f"Could not parse limits.price.min '{min_price_val}' as Decimal for {self.symbol}.")

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using default fallback.")

        # Absolute fallback: A very small number if everything else fails
        fallback_tick = Decimal('0.00000001')  # Small default
        self.logger.warning(f"Using extremely small fallback tick size for {self.symbol}: {fallback_tick}. Price quantization may be inaccurate.")
        return fallback_tick

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[tuple[str, Decimal]]:
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
            # self.logger.debug(f"EMA alignment check skipped for {self.symbol}: Missing required values.")
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
        # Check if *any* core indicator value (excluding price/vol) is non-NaN
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
            self.logger.error(f"Active weight set '{self.active_weight_set_name}' is missing or empty in config for {self.symbol}. Cannot generate signal.")
            return "HOLD"

        # --- Iterate Through Enabled Indicators with Weights ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue  # Skip disabled indicators

            weight_str = active_weights.get(indicator_key)
            if weight_str is None: continue  # Skip if no weight defined for this enabled indicator

            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue  # Skip if weight is zero
            except Exception:
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
                        if orderbook_data:  # Only call if data exists
                            indicator_score = method(orderbook_data, current_price)
                        else:
                            self.logger.debug(f"Orderbook data not available for {self.symbol}, skipping orderbook check.")
                            indicator_score = np.nan  # Treat as NaN if data missing
                    else:
                        indicator_score = method()  # Returns float score or np.nan

                except Exception as e:
                    self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                    indicator_score = np.nan  # Treat as NaN on error

                # --- Process Score ---
                debug_scores[indicator_key] = f"{indicator_score:.2f}" if pd.notna(indicator_score) else "NaN"
                if pd.notna(indicator_score):
                    try:
                        score_decimal = Decimal(str(indicator_score))  # Convert float score to Decimal
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
            self.logger.warning(f"No indicators contributed to the signal score for {self.symbol} (Total Weight Applied = 0). Defaulting to HOLD.")
            final_signal = "HOLD"
        else:
            # Normalize score? Optional. threshold works on weighted sum.
            # normalized_score = final_signal_score / total_weight_applied
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
        price_precision = self.get_price_precision()  # Get precision for logging current price
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
        if console_log_level <= logging.DEBUG:  # Only log detailed scores if console level is DEBUG
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
        scale_factor = 5.0  # Scales 0.2 to 1.0
        score = momentum * scale_factor
        return max(-1.0, min(1.0, score))  # Clamp score between -1 and 1

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume supports potential move (relative to MA). Score is direction-neutral."""
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))  # Use float

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma <= 0:
            return np.nan

        try:
            if current_volume > volume_ma * multiplier:
                # High volume suggests stronger conviction or potential climax/exhaustion.
                # Positive score indicates significance, not direction.
                return 0.7  # Strong confirmation/significance
            elif current_volume < volume_ma / multiplier:
                # Low volume suggests lack of interest or consolidation.
                return -0.4  # Negative score indicates lack of confirmation
            else:
                return 0.0  # Neutral volume
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
            score = 1.0  # Both deep oversold -> Strong bullish
        elif k > overbought and d > overbought:
            score = -1.0  # Both deep overbought -> Strong bearish

        # 2. K vs D Relationship (momentum indication)
        # Give higher weight to K crossing D than just K being above/below D
        # Requires previous state, approximating: if K is significantly different from D
        diff = k - d
        if abs(diff) > 5:  # Threshold for significant difference/cross potential
            if diff > 0:  # K moved above D (or is significantly above)
                score = max(score, 0.6) if score >= 0 else 0.6  # Bullish momentum, higher score if not already bearish
            else:  # K moved below D (or is significantly below)
                score = min(score, -0.6) if score <= 0 else -0.6  # Bearish momentum, higher score if not already bullish
        else:  # K and D are close
            if k > d: score = max(score, 0.2)  # Weak bullish
            elif k < d: score = min(score, -0.2)  # Weak bearish

        # 3. Consider position within range (0-100) - less important than extremes/crosses
        if oversold <= k <= overbought:  # Inside normal range
            # Optionally scale based on proximity to mid-point (50)
            # Ensure divisor isn't zero if thresholds are equal
            range_width = overbought - oversold
            if range_width > 0:
                 mid_range_score = (k - (oversold + range_width / 2)) / (range_width / 2)  # Scales -1 to 1 within the range
                 # Combine with existing score (e.g., average or weighted average)
                 score = (score + mid_range_score * 0.3) / 1.3  # Give mid-range position less weight
            # else: score remains unchanged if range is zero

        return max(-1.0, min(1.0, score))  # Clamp final score

    def _check_rsi(self) -> float:
        """Checks RSI indicator."""
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan

        if rsi <= 30: return 1.0
        if rsi >= 70: return -1.0
        if rsi < 40: return 0.5  # Leaning oversold/bullish
        if rsi > 60: return -0.5  # Leaning overbought/bearish
        # Smoother transition in the middle
        if 40 <= rsi <= 60:
            # Linear scale from +0.5 (at 40) to -0.5 (at 60)
            return 0.5 - (rsi - 40) * (1.0 / 20.0)
        return 0.0  # Fallback

    def _check_cci(self) -> float:
        """Checks CCI indicator."""
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan
        # CCI extremes often signal reversal potential
        if cci <= -150: return 1.0  # Strong Oversold -> Bullish
        if cci >= 150: return -1.0  # Strong Overbought -> Bearish
        if cci < -80: return 0.6  # Moderately Oversold
        if cci > 80: return -0.6  # Moderately Overbought
        # Trend confirmation near zero line - scale based on value
        if -80 <= cci <= 80:
            # Linear scale from +0.6 (at -80) to -0.6 (at 80)
             return - (cci / 80.0) * 0.6
        return 0.0  # Fallback

    def _check_wr(self) -> float:  # Williams %R
        """Checks Williams %R indicator."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        # WR: -100 (most oversold) to 0 (most overbought)
        if wr <= -80: return 1.0  # Oversold -> Bullish
        if wr >= -20: return -1.0  # Overbought -> Bearish
        # Scale linearly in the middle range (-80 to -20)
        if -80 < wr < -20:
            # Scale from +1.0 (at -80) to -1.0 (at -20)
            return 1.0 - (wr - (-80.0)) * (2.0 / 60.0)
        # Handle edge cases (exactly -80 or -20) or values outside range
        elif wr == -80: return 1.0
        elif wr == -20: return -1.0
        elif wr < -80: return 1.0  # Treat below -80 as 1.0
        elif wr > -20: return -1.0  # Treat above -20 as -1.0
        return 0.0  # Fallback (shouldn't be reached with above logic)

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
            return 0.0  # Neutral or undetermined

    def _check_sma_10(self) -> float:  # Example using SMA10
        """Checks price relative to SMA10."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return np.nan
        # Simple crossover score
        if last_close > sma_10: return 0.6  # Price above SMA -> Bullish bias
        if last_close < sma_10: return -0.6  # Price below SMA -> Bearish bias
        return 0.0

    def _check_vwap(self) -> float:
        """Checks price relative to VWAP."""
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return np.nan
        # VWAP acts as a dynamic support/resistance or fair value indicator
        if last_close > vwap: return 0.7  # Price above VWAP -> Bullish intraday sentiment
        if last_close < vwap: return -0.7  # Price below VWAP -> Bearish intraday sentiment
        return 0.0

    def _check_mfi(self) -> float:
        """Checks Money Flow Index."""
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan
        # MFI combines price and volume for overbought/oversold signals
        if mfi <= 20: return 1.0  # Oversold -> Bullish potential
        if mfi >= 80: return -1.0  # Overbought -> Bearish potential
        if mfi < 40: return 0.4  # Leaning oversold/accumulation
        if mfi > 60: return -0.4  # Leaning overbought/distribution
        # Scale linearly in the middle range (40 to 60)
        if 40 <= mfi <= 60:
            # Scale from +0.4 (at 40) to -0.4 (at 60)
            return 0.4 - (mfi - 40) * (0.8 / 20.0)
        return 0.0  # Fallback

    def _check_bollinger_bands(self) -> float:
        """Checks price relative to Bollinger Bands."""
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        bb_middle = self.indicator_values.get("BB_Middle", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(last_close):
            return np.nan

        # 1. Price relative to outer bands (mean reversion signals)
        if last_close < bb_lower: return 1.0  # Below lower band -> Strong bullish potential
        if last_close > bb_upper: return -1.0  # Above upper band -> Strong bearish potential

        # 2. Price relative to middle band (trend confirmation)
        # Bandwidth calculation for volatility context could also be added
        band_width = (bb_upper - bb_lower)
        if band_width <= 0: return 0.0  # Avoid division by zero if bands are flat

        # Avoid division by zero if middle band = upper/lower band
        upper_range = bb_upper - bb_middle
        lower_range = bb_middle - bb_lower

        if last_close > bb_middle:
            # Price above middle band -> weaker bullish signal (already in upper half)
            # Scale score based on proximity to upper band: closer to upper band -> weaker bullish / stronger bearish potential
            # Example: Score from +0.5 (at middle) to -0.5 (at upper band)
            proximity_to_upper = (last_close - bb_middle) / upper_range if upper_range > 0 else 0
            score = 0.5 - proximity_to_upper  # Max +0.5, min -0.5 (approx)
            return max(-0.5, min(0.5, score))  # Clamp result

        if last_close < bb_middle:
            # Price below middle band -> weaker bearish signal
            # Scale score based on proximity to lower band: closer to lower band -> weaker bearish / stronger bullish potential
            # Example: Score from -0.5 (at middle) to +0.5 (at lower band)
            proximity_to_lower = (bb_middle - last_close) / lower_range if lower_range > 0 else 0
            score = -0.5 + proximity_to_lower  # Max +0.5, min -0.5 (approx)
            return max(-0.5, min(0.5, score))  # Clamp result

        return 0.0  # Exactly on middle band

    def _check_orderbook(self, orderbook_data: dict | None, current_price: Decimal) -> float:
        """Analyzes order book depth for immediate pressure. Returns float score or NaN."""
        if not orderbook_data:
            self.logger.debug(f"Orderbook check skipped for {self.symbol}: No data provided.")
            return np.nan

        try:
            bids = orderbook_data.get('bids', [])  # List of [price (str), volume (str)]
            asks = orderbook_data.get('asks', [])  # List of [price (str), volume (str)]

            if not bids or not asks:
                self.logger.debug(f"Orderbook check skipped for {self.symbol}: Bids or asks list is empty.")
                return np.nan  # Need both sides for imbalance calculation

            # --- Simple Order Book Imbalance (OBI) within N levels ---
            # Alternative: Use % range around current price
            num_levels_to_check = 10  # Check top N levels
            bid_volume_sum = sum(Decimal(str(bid[1])) for bid in bids[:num_levels_to_check])
            ask_volume_sum = sum(Decimal(str(ask[1])) for ask in asks[:num_levels_to_check])

            total_volume = bid_volume_sum + ask_volume_sum
            if total_volume == 0:
                self.logger.debug(f"Orderbook check: No volume within top {num_levels_to_check} levels for {self.symbol}.")
                return 0.0  # Neutral if no volume

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

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan  # Return NaN on error

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on the provided entry price, ATR, and configured multipliers. Uses Decimal precision.
        The initial SL calculated here is primarily used for position sizing and setting the initial protection.
        Returns (entry_price, take_profit, stop_loss), all as Decimal or None.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None  # No TP/SL needed for HOLD

        atr_val_float = self.indicator_values.get("ATR")  # Get float ATR value
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float))  # Convert valid float ATR to Decimal

            # Get multipliers from config, convert to Decimal
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision for logging
            price_precision = self.get_price_precision()
            # Get minimum price increment (tick size) for quantization
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).")
                 return entry_price, None, None

            take_profit = None
            stop_loss = None

            if signal == "BUY":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Quantize TP UP, SL DOWN to the nearest tick size
                # Rounding UP for TP to potentially capture more profit
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                # Rounding DOWN for SL to give slightly more room (away from entry)
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick

            elif signal == "SELL":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Quantize TP DOWN, SL UP to the nearest tick size
                # Rounding DOWN for TP to potentially capture more profit
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                # Rounding UP for SL to give slightly more room (away from entry)
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

            # --- Validation ---
            # Ensure SL is actually beyond entry by at least one tick
            min_sl_distance = min_tick  # Minimum distance from entry for SL
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_sl_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation ({stop_loss:.{price_precision}f}) is too close to or above entry ({entry_price:.{price_precision}f}). Adjusting SL down to {adjusted_sl:.{price_precision}f}.{RESET}")
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_sl_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation ({stop_loss:.{price_precision}f}) is too close to or below entry ({entry_price:.{price_precision}f}). Adjusting SL up to {adjusted_sl:.{price_precision}f}.{RESET}")
                stop_loss = adjusted_sl

            # Ensure TP is potentially profitable relative to entry (at least one tick away)
            min_tp_distance = min_tick  # Minimum distance from entry for TP
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
            atr_log = f"{atr:.{price_precision + 1}f}" if atr else 'N/A'
            entry_log = f"{entry_price:.{price_precision}f}" if entry_price else 'N/A'

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log}, ATR={atr_log}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency, handling Bybit V5 structures."""
    lg = logger
    try:
        balance_info = None
        # For Bybit V5, specify the account type relevant to the market (e.g., CONTRACT, UNIFIED)
        # Let's prioritize 'CONTRACT' for derivatives, then 'UNIFIED', then default.
        account_types_to_try = ['CONTRACT', 'UNIFIED']
        successful_acc_type_used = None  # Keep track of which type worked

        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency}...")
                # Bybit V5 uses 'accountType' in params for fetch_balance
                balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                # Check if the desired currency is present in this balance info structure
                # Standard ccxt check:
                if currency in balance_info:
                    lg.debug(f"Found currency '{currency}' in balance structure using type '{acc_type}'.")
                    # Need to check if 'free' or equivalent exists within this structure
                    if balance_info[currency].get('free') is not None:
                         successful_acc_type_used = acc_type
                         break  # Found balance info with free balance directly
                    # Fallback: Check if 'info' structure has the currency data (Bybit V5 case)
                    elif 'info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
                        balance_list = balance_info['info']['result']['list']
                        if isinstance(balance_list, list):
                            found_in_nested = False
                            for account in balance_list:
                                # Check accountType field within the list items
                                if account.get('accountType') == acc_type:
                                    coin_list = account.get('coin')
                                    if isinstance(coin_list, list):
                                        if any(coin_data.get('coin') == currency for coin_data in coin_list):
                                            lg.debug(f"Currency '{currency}' confirmed in nested V5 balance structure (type: {acc_type}).")
                                            found_in_nested = True
                                            break  # Found in this account's coin list
                            if found_in_nested:
                                successful_acc_type_used = acc_type
                                break  # Break outer loop too
                    # If neither standard 'free' nor nested V5 found, continue to next type
                    lg.debug(f"Currency '{currency}' found in balance structure (type '{acc_type}'), but missing 'free' or V5 nested data. Trying next type.")
                    balance_info = None  # Reset to try next type

                # Check nested Bybit V5 structure directly if currency not top-level key
                elif 'info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
                    balance_list = balance_info['info']['result']['list']
                    if isinstance(balance_list, list):
                        found_in_nested = False
                        for account in balance_list:
                            if account.get('accountType') == acc_type:
                                coin_list = account.get('coin')
                                if isinstance(coin_list, list):
                                    if any(coin_data.get('coin') == currency for coin_data in coin_list):
                                        lg.debug(f"Found currency '{currency}' in nested V5 balance structure using type '{acc_type}'.")
                                        found_in_nested = True
                                        break  # Found in this account's coin list
                        if found_in_nested:
                            lg.debug(f"Using balance_info from successful nested check (type: {acc_type})")
                            successful_acc_type_used = acc_type
                            break  # Break outer loop too

                lg.debug(f"Currency '{currency}' not found in balance structure using type '{acc_type}'. Trying next.")
                balance_info = None  # Reset if currency not found in this structure

            except ccxt.ExchangeError as e:
                # Ignore errors indicating the account type doesn't exist, try the next one
                # Bybit V5 error for invalid accountType might be different, adjust if needed (e.g., check error code)
                # Example Bybit error code: 10001 (parameter error)
                bybit_code = getattr(e, 'code', None)
                if "account type not support" in str(e).lower() or "invalid account type" in str(e).lower() or bybit_code == 10001:
                    lg.debug(f"Account type '{acc_type}' not supported or parameter error fetching balance: {e}. Trying next.")
                    continue
                else:
                    lg.warning(f"Exchange error fetching balance for account type {acc_type}: {e}. Trying next.")
                    continue  # Try next type on other exchange errors too? Maybe safer.
            except Exception as e:
                lg.warning(f"Unexpected error fetching balance for account type {acc_type}: {e}. Trying next.")
                continue

        # If specific account types failed, try default fetch_balance without params
        if not balance_info:
            lg.debug(f"Fetching balance using default parameters for {currency}...")
            try:
                balance_info = exchange.fetch_balance()
                successful_acc_type_used = "Default"  # Indicate default fetch was used
            except Exception as e:
                lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                return None

        # --- Parse the balance_info ---
        available_balance_str = None

        # 1. Standard CCXT structure: balance_info[currency]['free']
        if currency in balance_info and 'free' in balance_info[currency] and balance_info[currency]['free'] is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str} (Acc Type Used: {successful_acc_type_used})")

        # 2. Bybit V5 structure (often nested): Check 'info' -> 'result' -> 'list'
        elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
            balance_list = balance_info['info']['result']['list']

            found_balance = False
            # Iterate through the list and find the currency, prioritizing CONTRACT, then UNIFIED if type wasn't fixed
            account_type_priority = ['CONTRACT', 'UNIFIED'] if successful_acc_type_used in ['CONTRACT', 'UNIFIED', 'Default', None] else [successful_acc_type_used]

            for target_type in account_type_priority:
                for account in balance_list:
                    current_account_type = account.get('accountType')  # e.g., 'CONTRACT', 'UNIFIED', 'SPOT'
                    # Check if the account type matches the target type we are looking for in this priority pass
                    if current_account_type == target_type:
                        coin_list = account.get('coin')
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get('coin') == currency:
                                    # Bybit V5 'availableToWithdraw' or 'availableBalance' are preferred for free balance
                                    free = coin_data.get('availableToWithdraw')  # Older field? Less reliable
                                    if free is None or str(free) == '': free = coin_data.get('availableBalance')  # Preferred field
                                    # Fallback to walletBalance only if others are missing (might include unrealized PnL)
                                    if free is None or str(free) == '': free = coin_data.get('walletBalance')

                                    if free is not None and str(free) != '':
                                        available_balance_str = str(free)
                                        lg.debug(f"Found balance via Bybit V5 nested structure: {available_balance_str} {currency} (Account: {current_account_type or 'N/A'})")
                                        found_balance = True
                                        break  # Found the currency in this account type
                        if found_balance: break  # Found in this account type, move to next priority if needed
                if found_balance: break  # Found in this priority pass, stop searching

            if not found_balance:
                lg.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]' structure for relevant account type(s).")

        # 3. Fallback: Check top-level 'free' dictionary if present
        elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
            available_balance_str = str(balance_info['free'][currency])
            lg.debug(f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency} (Acc Type Used: {successful_acc_type_used})")

        # 4. Last Resort: Check 'total' balance if 'free' is still missing
        if available_balance_str is None:
            total_balance = None
            if currency in balance_info and 'total' in balance_info[currency] and balance_info[currency]['total'] is not None:
                total_balance = balance_info[currency]['total']
            # Add check for Bybit nested total if primary failed
            elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                 balance_list = balance_info['info']['result']['list']
                 found_total = False
                 # Iterate through priority again
                 account_type_priority = ['CONTRACT', 'UNIFIED'] if successful_acc_type_used in ['CONTRACT', 'UNIFIED', 'Default', None] else [successful_acc_type_used]
                 for target_type in account_type_priority:
                    for account in balance_list:
                        current_account_type = account.get('accountType')
                        if current_account_type == target_type:
                            coin_list = account.get('coin')
                            if isinstance(coin_list, list):
                                for coin_data in coin_list:
                                    if coin_data.get('coin') == currency:
                                        total_balance_val = coin_data.get('walletBalance')  # Use walletBalance as total proxy
                                        if total_balance_val is not None and str(total_balance_val) != '':
                                             total_balance = total_balance_val
                                             found_total = True
                                             break
                                if found_total: break
                        if found_total: break
                    if found_total: break

                 if total_balance is not None:
                      lg.debug(f"Using 'walletBalance' ({total_balance}) from nested structure as 'total' fallback.")

            if total_balance is not None:
                lg.warning(f"{NEON_YELLOW}Could not determine 'free'/'available' balance for {currency}. Using 'total' balance ({total_balance}) as fallback. This might include collateral/unrealized PnL.{RESET}")
                available_balance_str = str(total_balance)
            else:
                lg.error(f"{NEON_RED}Could not determine any balance for {currency}. Balance info structure not recognized or currency missing.{RESET}")
                lg.debug(f"Full balance_info structure: {balance_info}")  # Log structure for debugging
                return None

        # --- Convert to Decimal ---
        try:
            final_balance = Decimal(available_balance_str)
            if final_balance >= 0:  # Allow zero balance
                lg.info(f"Available {currency} balance: {final_balance:.4f} (Source Acc Type: {successful_acc_type_used})")
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


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Gets market information like precision, limits, contract type, ensuring markets are loaded."""
    lg = logger
    try:
        # Ensure markets are loaded; reload if symbol is missing
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded or symbol missing, reloading markets...")
            exchange.load_markets(reload=True)  # Force reload

        # Check again after reloading
        if symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Market {symbol} still not found after reloading markets. Check symbol spelling and availability on {exchange.id}.{RESET}")
            return None

        market = exchange.market(symbol)
        if market:
            # Log key details for confirmation and debugging
            market_type = market.get('type', 'unknown')  # spot, swap, future etc.
            is_linear = market.get('linear', False)
            is_inverse = market.get('inverse', False)
            contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "Spot/Other"
            # Add contract flag for easier checking later
            market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']

            # --- Add Derived Precision/Tick Info ---
            # Ensure these are present for later use, deriving if necessary
            # Create a dummy analyzer instance to access precision helpers without full analysis
            dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.to_datetime([]))
            analyzer_temp = TradingAnalyzer(dummy_df, lg, {}, market)  # Pass minimal data
            min_tick_dec = analyzer_temp.get_min_tick_size()
            price_precision_int = analyzer_temp.get_price_precision()

            if 'precision' not in market: market['precision'] = {}
            # Store derived values if not already present or invalid
            # Use Decimal for internal representation, float for ccxt compatibility if needed
            if market['precision'].get('price') is None:  # precision.price often holds tick size
                 market['precision']['price'] = float(min_tick_dec)  # Store tick size as float for ccxt consistency
                 lg.debug(f"Added derived min tick size {min_tick_dec} as precision.price to market info.")
            if market['precision'].get('tick') is None:  # Add explicit 'tick' field if missing
                 market['precision']['tick'] = float(min_tick_dec)
                 lg.debug(f"Added derived min tick size {min_tick_dec} as precision.tick to market info.")
            if market['precision'].get('price_decimals') is None:  # Add explicit price decimals field
                 market['precision']['price_decimals'] = price_precision_int
                 lg.debug(f"Added derived price decimals {price_precision_int} as precision.price_decimals to market info.")

            # Ensure amount precision info is usable
            amount_precision_val = market.get('precision', {}).get('amount')
            amount_decimals = None
            amount_step_size = None
            if isinstance(amount_precision_val, int):  # Usually means decimal places
                amount_decimals = amount_precision_val
                amount_step_size = Decimal('1e-' + str(amount_decimals))
            elif isinstance(amount_precision_val, (float, str)):  # Usually means step size
                try:
                    amount_step_size = Decimal(str(amount_precision_val))
                    if amount_step_size > 0:
                        amount_decimals = abs(amount_step_size.normalize().as_tuple().exponent)
                    else: amount_step_size = None  # Invalid step size
                except Exception: pass  # Ignore conversion error
            # Store derived amount info if missing
            if market['precision'].get('amount_decimals') is None and amount_decimals is not None:
                market['precision']['amount_decimals'] = amount_decimals
                lg.debug(f"Added derived amount decimals {amount_decimals} as precision.amount_decimals to market info.")
            if market['precision'].get('amount_step') is None and amount_step_size is not None:
                market['precision']['amount_step'] = float(amount_step_size)  # Store step as float
                lg.debug(f"Added derived amount step size {amount_step_size} as precision.amount_step to market info.")

            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract={contract_type}, "
                f"Precision(PriceTick/PriceDecimals/AmountStep): {market.get('precision', {}).get('tick', 'N/A')}/{market.get('precision', {}).get('price_decimals', 'N/A')}/{market.get('precision', {}).get('amount_step', 'N/A')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min', 'N/A')}/{market.get('limits', {}).get('amount', {}).get('max', 'N/A')}, "
                f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min', 'N/A')}/{market.get('limits', {}).get('cost', {}).get('max', 'N/A')}, "
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
    initial_stop_loss_price: Decimal,  # The calculated initial SL price (Decimal)
    entry_price: Decimal,  # Estimated entry price (e.g., current market price) (Decimal)
    market_info: dict,
    exchange: ccxt.Exchange,  # Pass exchange object for formatting helpers
    logger: logging.Logger | None = None
) -> Decimal | None:
    """Calculates position size based on risk percentage, initial SL distance, balance,
    and market constraints (min/max size, precision, contract size). Returns Decimal size or None.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('linear', bool(not is_contract))  # Assume spot is linear type
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
    # Allow zero or negative stop loss price if it resulted from calculation, but warn
    if initial_stop_loss_price <= 0:
        lg.warning(f"Position sizing for {symbol}: Calculated initial stop loss price ({initial_stop_loss_price}) is zero or negative. Ensure SL calculation is correct.")
        # Do not return None here, let the distance calculation handle it

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
            lg.warning(f"Could not parse contract size '{contract_size_str}' for {symbol}. Defaulting to 1.")
            contract_size = Decimal('1')

        # --- Calculate Size based on Market Type ---
        calculated_size = Decimal('0')
        if is_linear:  # Linear contracts or Spot
            # Size (in Base units or Contracts) = Risk Amount (Quote) / (SL Distance per Unit (Quote/Base) * Contract Size (Base/Contract))
            # Denominator represents risk per unit (base/contract) in quote currency
            risk_per_unit_quote = sl_distance_per_unit * contract_size
            if risk_per_unit_quote <= 0:
                lg.error(f"Position sizing failed for {symbol}: Risk per unit is zero or negative ({risk_per_unit_quote}). Check SL distance or contract size.")
                return None
            calculated_size = risk_amount_quote / risk_per_unit_quote
            lg.debug(f"  Linear/Spot Sizing: RiskAmt={risk_amount_quote} / (SLDist={sl_distance_per_unit} * ContSize={contract_size}) = {calculated_size}")
        else:  # Inverse contract logic (e.g., BTC/USD settled in BTC)
            # Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
            # Risk per Contract (Quote) = Contract Value (Base/Contract) * SL Distance (Quote/Base)
            # Contract Value (Base/Contract) = Contract Size (Quote/Contract) / Entry Price (Quote/Base)
            # Here, `contractSize` usually represents the Quote value per contract (e.g., 1 for 1 USD).
            contract_value_quote = contract_size
            if entry_price > 0:
                contract_value_base = contract_value_quote / entry_price
                risk_per_contract_quote = contract_value_base * sl_distance_per_unit

                if risk_per_contract_quote > 0:
                     calculated_size = risk_amount_quote / risk_per_contract_quote
                     lg.debug(f"  Inverse Sizing: ContValBase={contract_value_base}, RiskPerContQuote={risk_per_contract_quote}")
                     lg.debug(f"  Inverse Sizing: RiskAmt={risk_amount_quote} / RiskPerContQuote={risk_per_contract_quote} = {calculated_size}")
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
        cost_limits = limits.get('cost', {})  # Cost = size * price
        precision = market_info.get('precision', {})
        # Use derived amount precision info
        amount_step_size_float = precision.get('amount_step')
        amount_step_size = Decimal(str(amount_step_size_float)) if amount_step_size_float is not None else None

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
        current_cost = Decimal('0')
        if is_linear:
            current_cost = adjusted_size * entry_price * contract_size
        else:  # Inverse Cost = Size (Contracts) * Contract Value (Quote/Contract)
            contract_value_quote = contract_size  # Assumes contractSize holds the quote value
            current_cost = adjusted_size * contract_value_quote
            lg.debug(f"  Inverse Cost Calculation: Size={adjusted_size} * ContractValueQuote={contract_value_quote} = {current_cost}")

        lg.debug(f"  Cost Check: Amount-Adjusted Size={adjusted_size:.8f} {size_unit}, Estimated Cost={current_cost:.4f} {quote_currency}")
        lg.debug(f"  Cost Limits: Min={min_cost}, Max={max_cost}")

        if min_cost > 0 and current_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} {quote_currency} (Size: {adjusted_size:.8f}) is below minimum cost {min_cost:.4f} {quote_currency}. Attempting to increase size.{RESET}")
            # Calculate required size to meet min cost
            required_size_for_min_cost = Decimal('0')
            if is_linear:
                if entry_price > 0 and contract_size > 0:
                    required_size_for_min_cost = min_cost / (entry_price * contract_size)
                else:
                    lg.error("Cannot calculate size for min cost (linear) due to invalid entry price or contract size.")
                    return None
            else:  # Inverse
                contract_value_quote = contract_size  # Use contract_size as value
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
            elif required_size_for_min_cost < min_amount:
                 lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size {required_size_for_min_cost:.8f}, but min amount is {min_amount:.8f}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f} {size_unit}")
                 adjusted_size = required_size_for_min_cost
                 # Re-calculate cost for logging/final checks
                 if is_linear:
                    current_cost = adjusted_size * entry_price * contract_size
                 else:  # Inverse
                    contract_value_quote = contract_size  # Use contract_size as value
                    current_cost = adjusted_size * contract_value_quote

        elif max_cost > 0 and current_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {current_cost:.4f} {quote_currency} exceeds maximum cost {max_cost:.4f} {quote_currency}. Reducing size.{RESET}")
            adjusted_size_for_max_cost = Decimal('0')
            if is_linear:
                if entry_price > 0 and contract_size > 0:
                    adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                else:
                    lg.error("Cannot calculate size for max cost (linear) due to invalid entry price or contract size.")
                    return None
            else:  # Inverse
                contract_value_quote = contract_size  # Use contract_size as value
                if contract_value_quote > 0:
                    adjusted_size_for_max_cost = max_cost / contract_value_quote
                else:
                    lg.error("Cannot calculate size for max cost (inverse) due to invalid contract value.")
                    return None

            lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f} {size_unit}")

            # Check if this reduced size is now below min amount
            if adjusted_size_for_max_cost < min_amount:
                lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is now below minimum amount {min_amount:.8f}. Cannot meet conflicting limits. Trade aborted.{RESET}")
                return None
            else:
                adjusted_size = adjusted_size_for_max_cost

        # 3. Apply Amount Precision/Step Size (rounding DOWN - TRUNCATE)
        final_size = Decimal('0')
        try:
            # Use exchange.amount_to_precision with TRUNCATE (floor/ROUND_DOWN for positive numbers)
            # to ensure we don't exceed balance/risk due to rounding up size.
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit} using exchange helper.")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision for {symbol} ({fmt_err}). Using manual rounding (ROUND_DOWN) based on step size.{RESET}")
            # Fallback to manual rounding based on derived step size
            if amount_step_size is not None and amount_step_size > 0:
                # Round down to the nearest step size increment
                final_size = (adjusted_size // amount_step_size) * amount_step_size
                lg.info(f"Applied manual amount step size ({amount_step_size}), rounded down: {adjusted_size:.8f} -> {final_size} {size_unit}")
            else:
                lg.warning(f"{NEON_YELLOW}Amount step size invalid or not found for {symbol}. Using size adjusted only for limits: {adjusted_size:.8f}{RESET}")
                final_size = adjusted_size  # Use the size adjusted for limits only

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
                 lg.warning(f"Final formatted size {final_size} is extremely close to min amount {min_amount}. Proceeding cautiously.")

        # Final check against min cost AFTER formatting/rounding
        final_cost = Decimal('0')
        if is_linear:
            final_cost = final_size * entry_price * contract_size
        else:  # Inverse
            contract_value_quote = contract_size  # Use contract_size as value
            final_cost = final_size * contract_value_quote

        if min_cost > 0 and final_cost < min_cost:
             # Check if it's close enough due to rounding? Use tolerance.
            if not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6):
                 lg.error(f"{NEON_RED}Final formatted size {final_size} {size_unit} results in cost {final_cost:.4f} {quote_currency} which is below minimum cost {min_cost:.4f} {quote_currency} for {symbol}. Trade aborted.{RESET}")
                 return None
            else:
                 lg.warning(f"Final cost {final_cost} is very close to min cost {min_cost}. Proceeding cautiously.")

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(f"{NEON_RED}Position sizing error for {symbol}: Missing market info key {e}. Market: {market_info}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks for an open position for the given symbol using fetch_positions.
    Returns the enhanced position dictionary from CCXT if an active position exists, otherwise None.
    Handles variations in position reporting and enhances with SL/TP/TSL info from 'info' dict.
    """
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: list[dict] = []
        market = None  # Fetch market info for context

        # Bybit V5: fetch_positions usually requires the symbol and category
        try:
            market = exchange.market(symbol)
            if not market:
                 lg.error(f"Cannot fetch positions: Market info not found for '{symbol}' via exchange.market().")
                 return None
            # Determine category based on market info
            category = 'linear' if market.get('linear', True) else 'inverse'
            params = {'category': category, 'symbol': market['id']}  # Fetch specific symbol using market ID
            lg.debug(f"Using params for fetch_positions: {params}")
            # Fetch positions for the specific symbol
            # Note: Bybit V5 fetch_positions with symbol returns a list (usually 0 or 1 element)
            positions = exchange.fetch_positions(symbols=[market['symbol']], params=params)  # Use unified symbol here

        except ccxt.BadSymbol as e:
            lg.error(f"Cannot fetch positions: Invalid symbol '{symbol}' for {exchange.id}. {e}")
            return None
        except ccxt.ExchangeError as e:
            # Handle errors like "symbol not found" or Bybit "no position found" codes
            no_pos_codes_v5 = [110021, 110025]  # Position is closed or not exist / Position idx not match
            bybit_code = getattr(e, 'code', None)
            if "symbol not found" in str(e).lower() or "instrument not found" in str(e).lower() or bybit_code in no_pos_codes_v5:
                lg.info(f"No position found for {symbol} (Exchange error: {e}).")
                return None
            # Re-raise other exchange errors
            lg.error(f"Exchange error fetching position for {symbol}: {e}", exc_info=True)
            return None
        except Exception as e:
            lg.error(f"Error fetching position for {symbol}: {e}", exc_info=True)
            return None

        # --- Process the fetched positions ---
        active_position = None
        if positions:  # Check if the list is not empty
            # In One-Way mode, there should be only one position entry per symbol
            if len(positions) > 1:
                lg.warning(f"Found multiple position entries ({len(positions)}) for {symbol}. This might indicate Hedge Mode or an API issue. Using the first entry with non-zero size.")
                # Find the first position with a meaningful size
                for pos_entry in positions:
                     pos_size_str_chk = pos_entry.get('info', {}).get('size')
                     if pos_size_str_chk is not None:
                         try:
                             if abs(Decimal(str(pos_size_str_chk))) > Decimal('1e-9'):
                                 pos = pos_entry
                                 lg.debug(f"Using position entry with size {pos_size_str_chk}.")
                                 break
                         except Exception: pass
                else:  # If no position with size found in the loop
                    lg.warning("Multiple position entries found, but none with non-zero size. Assuming no active position.")
                    return None
            else:
                pos = positions[0]  # Use the first/only entry

            # --- Validate Position Size ---
            pos_size_str = None
            # Prefer 'info.size' (Bybit V5), then 'contracts' (standard), then others
            if pos.get('info', {}).get('size') is not None: pos_size_str = str(pos['info']['size'])
            elif pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif pos.get('contractSize') is not None: pos_size_str = str(pos['contractSize'])

            if pos_size_str is None:
                lg.warning(f"Could not find position size field in position data for {symbol}. Assuming no active position. Data: {pos}")
                return None  # Cannot proceed without size

            try:
                position_size = Decimal(pos_size_str)
                # Use a small threshold based on minimum contract size if available, else small number
                min_size_threshold = Decimal('1e-9')  # Default threshold
                if market:
                    try:
                        min_amt = market.get('limits', {}).get('amount', {}).get('min')
                        if min_amt is not None:
                            # Use a fraction of min size, but ensure it's not excessively large or zero
                            min_size_threshold = max(min_size_threshold, Decimal(str(min_amt)) * Decimal('0.01'))
                    except Exception: pass  # Ignore errors fetching min size

                # Check if absolute size is meaningfully non-zero
                if abs(position_size) > min_size_threshold:
                    active_position = pos
                    lg.debug(f"Found potential active position entry for {symbol} with size {position_size}.")
                else:
                    lg.info(f"Position found for {symbol}, but size ({position_size}) is below threshold ({min_size_threshold}). Treating as no active position.")
                    return None  # Treat effectively zero size as no position

            except Exception as e:
                lg.warning(f"Could not parse position size '{pos_size_str}' as Decimal for {symbol}. Assuming no active position. Error: {e}")
                return None  # Cannot proceed if size is invalid

        # --- Post-Process the found active position (if any) ---
        if active_position:
            # Ensure market info is attached for convenience
            if market:
                active_position['market'] = market

            # --- Determine Side ---
            side = active_position.get('side')  # Standard 'long' or 'short'
            # Bybit V5: 'info.side' can be 'Buy' (long), 'Sell' (short), or 'None' (no position)
            info_side = active_position.get('info', {}).get('side')
            size_decimal = Decimal(str(active_position.get('info', {}).get('size', '0')))  # Use info.size for sign check if needed

            if side not in ['long', 'short']:
                if info_side == 'Buy': side = 'long'
                elif info_side == 'Sell': side = 'short'
                else:
                    # Infer from size sign as a last resort (less reliable with Bybit V5 info.size)
                    lg.warning(f"Ambiguous side ('{side}') and info.side ('{info_side}') for {symbol}. Attempting inference from size sign (less reliable).")
                    size_threshold = Decimal('1e-9')  # Re-define here for clarity
                    if size_decimal > size_threshold: side = 'long'
                    elif size_decimal < -size_threshold: side = 'short'
                    else:
                        lg.error(f"Position size {size_decimal} is close to zero, could not determine side reliably for {symbol}. Treating as no position.")
                        return None

                active_position['side'] = side  # Update the dict
                lg.debug(f"Determined position side as '{side}' for {symbol}.")

            # --- Ensure 'contracts' field matches 'info.size' and is positive ---
            # CCXT standard 'contracts' should represent the absolute amount
            try:
                info_size_abs = abs(Decimal(str(active_position.get('info', {}).get('size', '0'))))
                active_position['contracts'] = float(info_size_abs)  # Store as float for ccxt standard
                lg.debug(f"Standardized 'contracts' field to positive value: {active_position['contracts']}")
            except Exception:
                lg.warning(f"Could not standardize 'contracts' field based on info.size for {symbol}.")

            # --- Enhance with SL/TP/TSL info from 'info' dict ---
            info_dict = active_position.get('info', {})

            # Helper to extract and validate price/distance from info dict, returning Decimal or None
            def get_valid_decimal_from_info(key: str) -> Decimal | None:
                val_str = info_dict.get(key)
                # Bybit uses '0', '0.0', '0.00' etc. for inactive/zero values. Treat empty string as None.
                if val_str is not None and str(val_str).strip() != '':
                    try:
                        price_dec = Decimal(str(val_str).strip())
                        # Allow zero for TSL distance/activation, require positive for SL/TP
                        if key in ['stopLoss', 'takeProfit']:
                            if price_dec > 0: return price_dec
                        elif key in ['trailingStop', 'activePrice']:
                             if price_dec >= 0: return price_dec  # Allow zero
                        else:  # Other fields like entryPrice, liqPrice
                             if price_dec > 0: return price_dec
                    except Exception: pass  # Ignore conversion errors
                return None

            # Populate standard fields if missing or invalid, using validated values from info
            # CCXT expects float or None for these standard fields
            sl_val_dec = get_valid_decimal_from_info('stopLoss')
            tp_val_dec = get_valid_decimal_from_info('takeProfit')
            active_position['stopLossPrice'] = float(sl_val_dec) if sl_val_dec else None
            active_position['takeProfitPrice'] = float(tp_val_dec) if tp_val_dec else None

            # Add TSL info if available - distance/value and activation price
            # Store parsed Decimal versions for easier checks later, keep original string too
            tsl_dist_dec = get_valid_decimal_from_info('trailingStop')
            tsl_act_dec = get_valid_decimal_from_info('activePrice')
            active_position['trailingStopLossDistanceRaw'] = info_dict.get('trailingStop', '')  # Store raw string
            active_position['tslActivationPriceRaw'] = info_dict.get('activePrice', '')  # Store raw string
            active_position['trailingStopLossDistanceDecimal'] = tsl_dist_dec if tsl_dist_dec is not None else Decimal('0')  # Default to 0 if None/invalid
            active_position['tslActivationPriceDecimal'] = tsl_act_dec if tsl_act_dec is not None else Decimal('0')  # Default to 0 if None/invalid

            # Helper function to format price/value for logging
            def format_log_value(value: str | float | Decimal | None, precision: int = 6) -> str:
                if value is None or str(value).strip() == '': return 'N/A'
                try:
                    d_value = Decimal(str(value))
                    # Format with specified precision
                    return f"{d_value:.{precision}f}"
                except Exception: return str(value)  # Return raw if conversion fails

            # Log details of the confirmed active position
            log_precision = market['precision']['price_decimals'] if market and market.get('precision', {}).get('price_decimals') is not None else 6
            amount_precision = market['precision']['amount_decimals'] if market and market.get('precision', {}).get('amount_decimals') is not None else 8

            entry_price_val = active_position.get('entryPrice', info_dict.get('avgPrice'))
            entry_price = format_log_value(entry_price_val, log_precision)
            contracts_val = active_position.get('contracts')  # Use standardized positive value
            contracts = format_log_value(contracts_val, amount_precision)
            liq_price = format_log_value(active_position.get('liquidationPrice'), log_precision)
            leverage_str = active_position.get('leverage', info_dict.get('leverage'))
            leverage = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str is not None and str(leverage_str).strip() != '' else 'N/A'
            pnl_val = active_position.get('unrealizedPnl', info_dict.get('unrealisedPnl'))  # Check standard and info field
            pnl = format_log_value(pnl_val, 4)
            sl_price = format_log_value(active_position.get('stopLossPrice'), log_precision)
            tp_price = format_log_value(active_position.get('takeProfitPrice'), log_precision)
            # Use the parsed Decimal for check, format the raw string for log
            is_tsl_active_log = active_position.get('trailingStopLossDistanceDecimal', Decimal(0)) > 0
            tsl_dist_log = format_log_value(active_position.get('trailingStopLossDistanceRaw'), log_precision)
            tsl_act_log = format_log_value(active_position.get('tslActivationPriceRaw'), log_precision)

            logger.info(f"{NEON_GREEN}Active {side.upper()} position found for {symbol}:{RESET} "
                        f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                        f"Leverage={leverage}, PnL={pnl}, SL={sl_price}, TP={tp_price}, "
                        f"TSL Active: {is_tsl_active_log} (Dist={tsl_dist_log}/Act={tsl_act_log})")
            logger.debug(f"Full position details for {symbol}: {json.dumps(active_position, default=str, indent=2)}")  # Dump full dict for debug
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
        no_pos_codes = [110021, 110025]  # Bybit V5: Position not found / is closed / idx mismatch
        err_str = str(e).lower()
        err_code = getattr(e, 'code', None)

        if any(msg in err_str for msg in no_pos_msgs) or (err_code in no_pos_codes):
            lg.info(f"No open position found for {symbol} (Confirmed by exchange error: {e}).")
            return None
        # Bybit V5 might return success (retCode 0) but an empty list if no position - handled by the logic above.
        lg.error(f"{NEON_RED}Unhandled Exchange error fetching positions for {symbol}: {e} (Code: {err_code}){RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)

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
        # TODO: Could implement direct API call here if needed as a fallback.
        return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")

        # --- Prepare Bybit V5 specific parameters (if applicable) ---
        params = {}
        if 'bybit' in exchange.id.lower():
            # Fetch current margin mode if possible to avoid errors (optional, needs position usually)
            # ... (margin mode detection logic removed for brevity, as setting both leverages is generally safe) ...

            # Bybit V5 set_leverage endpoint requires buy/sell leverage and category.
            # Use string representation for leverage in params for Bybit.
            category = 'linear' if market_info.get('linear', True) else 'inverse'
            params = {
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage),
                'category': category
            }
            lg.debug(f"Using Bybit V5 specific params for set_leverage: {params}")

        # --- Call set_leverage ---
        # The `leverage` argument is the primary value, `params` provides extra details
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)

        # Log response for debugging, but success isn't guaranteed by non-exception response
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # --- Verification ---
        # A successful response (no exception, or specific success code/message) implies success.
        verified = False
        if response is not None:
            if isinstance(response, dict):
                 # Check Bybit V5 specific response structure (might be nested in 'info')
                 ret_code = response.get('retCode', response.get('info', {}).get('retCode'))
                 if ret_code == 0:
                     lg.debug(f"Set leverage call for {symbol} confirmed success (retCode 0).")
                     verified = True
                 # Check for "leverage not modified" code (e.g., 110045)
                 elif ret_code == 110045:
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation: Code {ret_code}).{RESET}")
                     verified = True  # Treat as success
                 elif ret_code is not None:  # Got a non-zero retCode
                      ret_msg = response.get('retMsg', response.get('info', {}).get('retMsg', 'Unknown Error'))
                      lg.warning(f"Set leverage call for {symbol} returned non-zero retCode {ret_code} ({ret_msg}). Treating as failure.")
                      verified = False  # Treat as failure if non-zero code returned
                 else:
                      # No retCode found, rely on lack of exception
                      lg.debug(f"Set leverage call for {symbol} returned a response (no retCode found). Assuming success if no error.")
                      verified = True
            else:
                 # Response is not a dict, rely on lack of exception
                 lg.debug(f"Set leverage call for {symbol} returned non-dict response. Assuming success if no error.")
                 verified = True
        else:
            # If response is None, it *might* still be okay (Bybit V5 sometimes returns None via ccxt). Rely on no exception.
            lg.debug(f"Set leverage call for {symbol} returned None. Assuming success as no error was raised.")
            verified = True  # Tentative

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
        bybit_code = getattr(e, 'code', None)  # CCXT often maps Bybit retCode to e.code
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")

        # --- Handle Common Bybit V5 Leverage Errors ---
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
            lg.error(f"{NEON_YELLOW} >> Hint: May indicate insufficient available margin if using Isolated Margin and increasing leverage requires more margin allocation possibility, although less common for just setting leverage.{RESET}")

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,  # "BUY" or "SELL"
    position_size: Decimal,  # Size calculated previously (Decimal)
    market_info: dict,
    logger: logging.Logger | None = None,
    reduce_only: bool = False  # Added flag for closing trades
) -> dict | None:
    """Places a market order using CCXT. Can be used for opening or closing (with reduce_only=True).
    Returns the order dictionary on success, None on failure.
    SL/TP/TSL should be set *after* opening trades and verifying position.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    base_currency = market_info.get('base', '')
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else base_currency

    # Convert Decimal size to float for ccxt create_order amount parameter
    try:
        # Use absolute size for amount, side determines direction
        amount_float = float(abs(position_size))
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} reduce={reduce_only}): Invalid position size for order amount ({amount_float}). Must be positive.")
            return None
    except Exception as e:
        lg.error(f"Trade aborted ({symbol} {side} reduce={reduce_only}): Failed to convert position size {position_size} to float: {e}")
        return None

    # --- Prepare Order Parameters ---
    params = {
        # Bybit V5: Specify position index for One-Way vs Hedge mode
        # Assuming One-Way mode (positionIdx=0) is default/required
        'positionIdx': 0,  # 0 for One-Way mode, 1 for Buy in Hedge, 2 for Sell in Hedge
        'reduceOnly': reduce_only,  # Use the passed flag
        # 'timeInForce': 'IOC', # Optional: Market orders usually fill quickly anyway
        'closeOnTrigger': False,  # Default is False
    }
    # Add category for Bybit V5
    if 'bybit' in exchange.id.lower():
        category = 'linear' if market_info.get('linear', True) else 'inverse'
        params['category'] = category

    action = "Closing" if reduce_only else "Opening"
    lg.info(f"Attempting to place {side.upper()} {order_type} order ({action}) for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {size_unit}")
    lg.info(f"  Params: {params}")

    try:
        # --- Execute Market Order ---
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None,  # Market order doesn't need price
            params=params
        )

        # --- Log Success and Basic Order Details ---
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A')  # Market orders might be 'closed' quickly or 'open' initially
        lg.info(f"{NEON_GREEN}Trade Order Placed Successfully ({action})! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} reduce={reduce_only}): {json.dumps(order, default=str, indent=2)}")  # Log the full order response

        # IMPORTANT: Market orders might not fill instantly or exactly at the desired price.
        # The calling function MUST wait and verify the resulting position using get_open_position
        # especially after opening trades, to get the actual entry price before setting protection.
        # For closing trades, verification is also good practice.
        return order  # Return the order dictionary

    # --- Error Handling ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order ({action}) for {symbol}: {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        # Log balance info if possible
        try:
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            lg.error(f"  Available {QUOTE_CURRENCY} balance: {balance}")
            # Hint based on Bybit codes
            if bybit_code == 110007 or "insufficient available balance" in str(e).lower():
                lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Check available margin balance, leverage, and if order cost exceeds balance. Cost ~ Size * Price / Leverage.{RESET}")
        except Exception as bal_err:
            lg.error(f"  Could not fetch balance for context: {bal_err}")

    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {symbol} ({action}): {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        lg.error(f"  Size: {amount_float}, Type: {order_type}, Side: {side}, Params: {params}")
        lg.error(f"  Market Limits: Amount={market_info.get('limits', {}).get('amount')}, Cost={market_info.get('limits', {}).get('cost')}")
        lg.error(f"  Market Precision: Amount={market_info.get('precision', {}).get('amount_step')}, Price={market_info.get('precision', {}).get('tick')}")
        # Hint based on Bybit codes
        if bybit_code == 10001 and "parameter error" in str(e).lower():
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Check if size/price violates precision or limits.{RESET}")
        elif bybit_code == 110017:  # Order quantity exceeds limit
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Order size {amount_float} might violate exchange's min/max quantity per order.{RESET}")
        elif bybit_code == 110040:  # Order size is less than the minimum order size
             lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Order size {amount_float} is below the minimum allowed. Check `calculate_position_size` logic and market limits.{RESET}")
        elif bybit_code == 110014 and reduce_only:  # Reduce-only order failed
             lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Reduce-only close order failed. Size ({amount_float}) might exceed open position, position already closed, or API issue?{RESET}")

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing order ({action}) for {symbol}: {e}{RESET}")
        # Order placement is critical, might need specific retry/state handling here,
        # but for now, we assume failure if network error occurs during placement.

    except ccxt.ExchangeError as e:
        # Handle specific Bybit V5 error codes for better diagnostics
        bybit_code = getattr(e, 'code', None)  # CCXT often maps retCode to e.code
        err_str = str(e).lower()
        lg.error(f"{NEON_RED}Exchange error placing order ({action}) for {symbol}: {e} (Code: {bybit_code}){RESET}")

        # --- Bybit V5 Specific Error Hints ---
        if bybit_code == 110007:  # Insufficient margin/balance (already handled by InsufficientFunds, but double check)
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Insufficient balance/margin. Check available balance, leverage. Cost ~ Size * Price / Leverage.{RESET}")
        elif bybit_code == 110043:  # Order cost not available / exceeds limit
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Order cost likely exceeds available balance or risk limits. Check Cost ~ Size * Price / Leverage.{RESET}")
        elif bybit_code == 110044:  # Position size has exceeded the risk limit
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Opening this position (size {amount_float}) would exceed Bybit's risk limit tier for the current leverage. Check Bybit risk limit docs or reduce size/leverage.{RESET}")
        elif bybit_code == 110014 and not reduce_only:  # Reduce-only order failed (shouldn't happen here unless reduceOnly=True)
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Reduce-only flag might be incorrectly set? Ensure 'reduceOnly' is False when opening/increasing.{RESET}")
        elif bybit_code == 110055:  # Position idx not match position mode
            lg.error(f"{NEON_YELLOW} >> Hint (Code {bybit_code}): Mismatch between 'positionIdx' parameter ({params.get('positionIdx')}) and account's Position Mode (must be One-Way, not Hedge). Check Bybit account trade settings.{RESET}")
        elif bybit_code == 10005 or "order link id exists" in err_str:  # Duplicate Order ID
            lg.warning(f"{NEON_YELLOW}Duplicate order ID detected (Code {bybit_code}). May indicate a network issue causing retry, or order was already placed. Check position status manually!{RESET}")
            # Treat as failure for now, requires manual check
            return None
        elif "risk limit can't be place order" in err_str:  # Another risk limit message
            lg.error(f"{NEON_YELLOW} >> Hint: Order blocked by risk limits. Check Bybit risk limit tiers vs leverage/position size.{RESET}")
        elif bybit_code == 110021 and reduce_only:  # Position already closed when trying to close
             lg.warning(f"{NEON_YELLOW} >> Hint (Code 110021): Position not found or already closed when attempting to close (reduceOnly=True).{RESET}")
             # Treat this potentially as success (position is already closed)
             return {'id': 'N/A', 'status': 'closed', 'info': {'reason': 'Position already closed/not found on close attempt'}}  # Return dummy success

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing order ({action}) for {symbol}: {e}{RESET}", exc_info=True)

    return None  # Return None if order failed for any reason


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,  # Confirmed position dict from get_open_position
    logger: logging.Logger,
    stop_loss_price: Decimal | str | None = None,  # Allow '0' for cancel
    take_profit_price: Decimal | str | None = None,  # Allow '0' for cancel
    trailing_stop_distance: Decimal | str | None = None,  # Trailing distance in price points (Decimal or '0')
    tsl_activation_price: Decimal | str | None = None,  # Price to activate TSL (Decimal or '0')
) -> bool:
    """Internal helper to set SL, TP, or TSL for an existing position using Bybit's V5 API
    (`/v5/position/set-trading-stop`) via CCXT's `private_post` method.
    Returns True on success, False on failure.
    Note: Setting a fixed SL (`stop_loss_price`) will likely cancel an active TSL on Bybit,
          and vice-versa. This function prioritizes TSL if both valid TSL and SL are provided.
          To cancel SL/TP/TSL, pass '0' as a string for the respective parameter (Bybit API behavior).
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

    # Get side and size from position_info (needed for context/logging, not API params here)
    pos_side = position_info.get('side')
    pos_size_str = position_info.get('contracts', position_info.get('info', {}).get('size'))
    if pos_side not in ['long', 'short'] or pos_size_str is None:
        lg.error(f"Cannot set protection for {symbol}: Invalid or missing position side ('{pos_side}') or size in position_info.")
        return False

    # Check if values are valid Decimals and positive (or zero for activation price/distance if canceling)
    # Allow '0' string input for cancellation
    def is_valid_or_cancel(val: Decimal | str | None, allow_zero_decimal: bool = False) -> bool:
        if val is None: return False
        if isinstance(val, str) and val == '0': return True  # Explicit cancel
        if isinstance(val, Decimal):
             if allow_zero_decimal: return val >= 0  # Allow Decimal(0)
             else: return val > 0  # Require positive for SL/TP prices
        return False

    # Validate inputs based on their role
    has_sl_intent = stop_loss_price is not None
    has_tp_intent = take_profit_price is not None
    has_tsl_intent = trailing_stop_distance is not None  # Activation price is optional if distance is '0'

    is_sl_valid = has_sl_intent and is_valid_or_cancel(stop_loss_price, allow_zero_decimal=False)
    is_tp_valid = has_tp_intent and is_valid_or_cancel(take_profit_price, allow_zero_decimal=False)
    # TSL distance needs check, activation price needs check (can be 0)
    is_tsl_dist_valid = has_tsl_intent and is_valid_or_cancel(trailing_stop_distance, allow_zero_decimal=True)  # Distance can be '0'
    is_tsl_act_valid = is_valid_or_cancel(tsl_activation_price, allow_zero_decimal=True)  # Activation can be '0' or None initially? API expects it if tsl dist > 0
    is_tsl_valid = is_tsl_dist_valid and (tsl_activation_price is None or is_tsl_act_valid)  # Allow None activation initially? No, API requires it if distance > 0. Let's require it if dist > 0.
    if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0:
        is_tsl_valid = is_tsl_dist_valid and is_tsl_act_valid  # If distance > 0, activation must be valid (or '0')
    elif isinstance(trailing_stop_distance, str) and trailing_stop_distance == '0':
        is_tsl_valid = True  # Cancelling TSL is valid

    if not is_sl_valid and not is_tp_valid and not is_tsl_valid:
        lg.info(f"No valid protection parameters (SL, TP, TSL > 0 or '0' for cancel) provided for {symbol}. No protection set/modified.")
        # Consider this success as no API call was intended/needed
        return True

    # --- Prepare API Parameters for /v5/position/set-trading-stop ---
    # Determine category ('linear' or 'inverse')
    category = 'linear' if market_info.get('linear', True) else 'inverse'

    # Get position index from position_info, default to 0 (One-Way)
    position_idx = 0  # Default for One-Way mode
    try:
        # Attempt to get positionIdx from the 'info' dict if available
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
            position_idx = int(pos_idx_val)
    except (ValueError, TypeError):
        lg.warning(f"Could not parse positionIdx '{pos_idx_val}' from position info for {symbol}. Defaulting to {position_idx}.")

    params = {
        'category': category,
        'symbol': market_info['id'],  # Use exchange-specific ID (e.g., BTCUSDT)
        'tpslMode': 'Full',  # Apply to the whole position ('Partial' needs size param)
        # Trigger price type (LastPrice, MarkPrice, IndexPrice) - Use LastPrice for simplicity unless Mark is required
        'slTriggerBy': 'LastPrice',
        'tpTriggerBy': 'LastPrice',
        # Order type when triggered (Market is usually preferred for SL/TP)
        'slOrderType': 'Market',
        'tpOrderType': 'Market',
        'positionIdx': position_idx
    }

    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} position, Idx: {position_idx}):"]

    # --- Format and Add Parameters using exchange helpers ---
    tsl_added_to_params = False
    sl_added_to_params = False
    tp_added_to_params = False
    try:
        # Helper to format price using exchange's precision rules
        # Handles Decimal input and '0' string for cancellation
        def format_param_value(value: Decimal | str | None, is_price: bool = True) -> str | None:
            if value is None: return None
            if isinstance(value, str) and value == '0': return '0'  # Pass cancel signal directly
            if isinstance(value, Decimal):
                if value < 0: return None  # Invalid value
                if value == 0: return '0'  # Treat Decimal 0 as cancel signal for API
                try:
                    # Use price_to_precision for prices, amount_to_precision for distance? No, Bybit expects price format for distance too.
                    # Bybit API expects string numbers
                    formatted_val = exchange.price_to_precision(symbol, float(value))
                    # Ensure it's not zero after formatting if original wasn't zero
                    if Decimal(formatted_val) == 0 and value != 0:
                         lg.warning(f"Formatting value {value} resulted in '0.0' due to precision for {symbol}. Using raw value string '{str(value)}' if possible, else skipping.")
                         # Attempt raw string, might fail API validation but better than sending 0 incorrectly
                         # Check if raw string format is simple number
                         if re.fullmatch(r'\d+(\.\d+)?', str(value)):
                              return str(value)
                         else:
                              return None  # Skip if complex raw string
                    return formatted_val
                except Exception as fmt_err:
                    lg.warning(f"Could not format value {value} using exchange.price_to_precision: {fmt_err}. Skipping this parameter.")
                    return None
            return None  # Invalid type

        # Trailing Stop Loss (Set first, as it might override fixed SL)
        if is_tsl_valid:
            formatted_tsl_distance = format_param_value(trailing_stop_distance, is_price=False)  # Distance treated as price precision by Bybit API
            formatted_activation_price = format_param_value(tsl_activation_price, is_price=True)  # Activation is a price

            # Handle cancellation case explicitly
            if isinstance(trailing_stop_distance, str) and trailing_stop_distance == '0':
                params['trailingStop'] = '0'  # Cancel TSL
                # Bybit API might require activePrice='0' when cancelling TSL distance, or just ignore it. Set to 0 for safety.
                params['activePrice'] = '0'
                log_parts.append("  Trailing SL: Cancelling (Distance=0)")
                tsl_added_to_params = True  # Mark TSL action as handled
            # Handle setting TSL case
            elif formatted_tsl_distance is not None and formatted_tsl_distance != '0' and formatted_activation_price is not None:
                params['trailingStop'] = formatted_tsl_distance
                params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Distance={formatted_tsl_distance}, Activation={formatted_activation_price}")
                tsl_added_to_params = True
            else:
                lg.error(f"Failed to format valid TSL parameters for {symbol} (Dist: {trailing_stop_distance}, Act: {tsl_activation_price}). Cannot set TSL.")
                is_tsl_valid = False  # Mark TSL as failed

        # Fixed Stop Loss - Only add if TSL was NOT successfully set to an active value
        if is_sl_valid and (not tsl_added_to_params or params.get('trailingStop') == '0'):
            formatted_sl = format_param_value(stop_loss_price, is_price=True)
            if formatted_sl is not None:
                params['stopLoss'] = formatted_sl
                log_parts.append(f"  Fixed SL: {formatted_sl}")
                sl_added_to_params = True
            else:
                # If formatting failed but intent was there, mark as invalid
                if has_sl_intent: is_sl_valid = False
        elif has_sl_intent and tsl_added_to_params and params.get('trailingStop', '0') != '0':
             lg.warning(f"Both valid 'stopLoss' and active 'trailingStop' provided for {symbol}. Prioritizing TSL. Fixed 'stopLoss' parameter ignored.")
             is_sl_valid = False  # Mark fixed SL as not set

        # Fixed Take Profit
        if is_tp_valid:
            formatted_tp = format_param_value(take_profit_price, is_price=True)
            if formatted_tp is not None:
                params['takeProfit'] = formatted_tp
                log_parts.append(f"  Fixed TP: {formatted_tp}")
                tp_added_to_params = True
            else:
                # If formatting failed but intent was there, mark as invalid
                if has_tp_intent: is_tp_valid = False

    except Exception as fmt_err:
        lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
        return False

    # Check if any protection parameter is actually being set or cancelled
    protection_keys_in_params = [k for k in ['stopLoss', 'takeProfit', 'trailingStop', 'activePrice'] if k in params]

    if not protection_keys_in_params:
        lg.warning(f"No valid protection parameters could be formatted or remain after adjustments for {symbol}. No API call made.")
        # If nothing was intended, return True (success). If something was intended but failed formatting/prioritization, return False.
        if has_sl_intent or has_tp_intent or has_tsl_intent:
            if not is_sl_valid and not is_tp_valid and not is_tsl_valid:
                 return False  # Intent failed completely
            else:
                 # Intent partially met or overridden (e.g., SL ignored due to TSL, or formatting failed for one part)
                 # Decide if partial success is okay. Let's return True if *any* intended action made it to params.
                 if sl_added_to_params or tp_added_to_params or tsl_added_to_params:
                      return True
                 else:
                      return False  # No intended action made it to params
        else:
            return True  # Nothing was intended

    # Log the attempt
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={params})")

    # --- Call Bybit V5 API Endpoint ---
    try:
        # Use private_post as set_trading_stop might not be standard ccxt method
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        # --- Check Response ---
        # Bybit V5 standard response structure
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {})  # Contains extra details on failure

        if ret_code == 0:
            # Check for specific non-error messages that indicate no change
            no_change_msgs = [
                "stoplosstakeprofittrailingstopwerenotmodified",  # Exact match
                "stop loss take profit trailing stop were not modified",  # Space variation
                "not modified"  # General phrase
            ]
            processed_ret_msg = ret_msg.lower().replace(",", "").replace(".", "").replace("and", "").replace(" ", "")
            if any(msg in processed_ret_msg for msg in no_change_msgs):
                lg.info(f"{NEON_YELLOW}Position protection already set to target values or not modified for {symbol} (Exchange confirmation: {ret_msg}).{RESET}")
            else:
                lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            return True
        else:
            # Log specific error hints based on Bybit V5 documentation/codes
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            # --- Add hints based on common error codes for this endpoint ---
            if ret_code == 110043:  # Set tpsl failed
                lg.error(f"{NEON_YELLOW} >> Hint (110043): Ensure 'tpslMode' ('Full'/'Partial') is correct. Check trigger prices (e.g., SL below entry for long?). Check `retExtInfo`.{RESET}")
            elif ret_code == 110021:  # Position closed
                 lg.error(f"{NEON_YELLOW} >> Hint (110021): Position is already closed. Cannot set protection.{RESET}")
            elif ret_code == 110025:  # Position not found / size is zero / Idx mismatch
                lg.error(f"{NEON_YELLOW} >> Hint (110025): Position might have closed, changed size unexpectedly, or `positionIdx` ({params.get('positionIdx')}) mismatch?{RESET}")
            elif ret_code == 110044:  # Risk limit related
                lg.error(f"{NEON_YELLOW} >> Hint (110044): Protection settings might conflict with risk limits? Unlikely but possible.{RESET}")
            elif ret_code == 110055:  # Position idx not match position mode
                lg.error(f"{NEON_YELLOW} >> Hint (110055): Ensure 'positionIdx' ({params.get('positionIdx')}) matches account's Position Mode (One-Way vs Hedge).{RESET}")
            elif ret_code == 10001:  # Parameter Error (check retExtInfo for details)
                lg.error(f"{NEON_YELLOW} >> Hint (10001): General Parameter Error. Check ExtInfo: {ret_ext}. Is SL/TP/TSL value valid (non-zero, respects tick size)? Is `activePrice` valid for TSL? Are SL/TP on the wrong side of entry?{RESET}")
            elif ret_code == 110013:  # Parameter Error (less specific)
                lg.error(f"{NEON_YELLOW} >> Hint (110013): Check parameters: Is SL/TP/TSL value valid? Are SL/TP on the wrong side of entry?{RESET}")
            elif ret_code == 110036:  # Active price is invalid (for TSL)
                lg.error(f"{NEON_YELLOW} >> Hint (110036): TSL Activation price ({params.get('activePrice')}) might be invalid (e.g., too close, wrong side, already passed).{RESET}")
            elif ret_code == 110086:  # SL/TP price cannot be the same
                lg.error(f"{NEON_YELLOW} >> Hint (110086): Stop loss price cannot be the same as Take profit price.{RESET}")
            elif ret_code == 110087:  # TP must be greater than entry (for long) / less than entry (for short)
                 lg.error(f"{NEON_YELLOW} >> Hint (110087): Take Profit price ({params.get('takeProfit')}) is on the wrong side of the entry price for a {pos_side} position.{RESET}")
            elif ret_code == 110088:  # SL must be less than entry (for long) / greater than entry (for short)
                 lg.error(f"{NEON_YELLOW} >> Hint (110088): Stop Loss price ({params.get('stopLoss')}) is on the wrong side of the entry price for a {pos_side} position.{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower():
                lg.error(f"{NEON_YELLOW} >> Hint: TSL distance ({params.get('trailingStop')}) might be invalid (too small/large, tick size).{RESET}")
            elif "stop loss price is invalid" in ret_msg.lower() or "sl price invalid" in ret_msg.lower():
                lg.error(f"{NEON_YELLOW} >> Hint: Fixed SL price ({params.get('stopLoss')}) might be invalid (wrong side, too close/far, tick size).{RESET}")
            elif "take profit price is invalid" in ret_msg.lower() or "tp price invalid" in ret_msg.lower():
                lg.error(f"{NEON_YELLOW} >> Hint: Fixed TP price ({params.get('takeProfit')}) might be invalid (wrong side, too close/far, tick size).{RESET}")
            # Add more specific codes as encountered
            return False

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting protection for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:  # Catch potential errors from private_post call itself if not parsed above
        lg.error(f"{NEON_RED}Exchange error during protection API call for {symbol}: {e}{RESET}")
    except KeyError as e:
        lg.error(f"{NEON_RED}Error setting protection for {symbol}: Missing expected key {e} in market/position info.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

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
        # This function's purpose is only TSL, so return False indicating TSL wasn't set.
        return False

    # --- Get TSL parameters from config ---
    try:
        # Use Decimal for calculations
        callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
    except Exception as e:
        lg.error(f"{NEON_RED}Invalid TSL parameter format in config for {symbol}: {e}. Cannot calculate TSL.{RESET}")
        return False

    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_callback_rate ({callback_rate}) in config for {symbol}. Must be positive.{RESET}")
        return False
    # Activation percentage can be zero if TSL should be active immediately
    if activation_percentage < 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_activation_percentage ({activation_percentage}) in config for {symbol}. Must be non-negative.{RESET}")
        return False

    # --- Extract required position details ---
    try:
        # Use reliable fields from CCXT unified position structure or info dict
        entry_price_str = position_info.get('entryPrice', position_info.get('info', {}).get('avgPrice'))
        side = position_info.get('side')  # Should be 'long' or 'short'

        if entry_price_str is None or side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing required position info (entryPrice, side) to calculate TSL for {symbol}. Position: {position_info}{RESET}")
            return False

        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0:
            lg.error(f"{NEON_RED}Invalid entry price ({entry_price}) from position info for {symbol}. Cannot calculate TSL.{RESET}")
            return False

    except (TypeError, ValueError, KeyError) as e:
        lg.error(f"{NEON_RED}Error parsing position info for TSL calculation ({symbol}): {e}. Position: {position_info}{RESET}")
        return False

    # --- Calculate TSL parameters for Bybit API ---
    try:
        # Create a temporary analyzer instance just to use utility methods
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.to_datetime([]))
        if not market_info:
             lg.error("Cannot calculate TSL: Market info not available.")
             return False
        temp_analyzer = TradingAnalyzer(df=dummy_df, logger=lg, config=config, market_info=market_info)
        price_precision = temp_analyzer.get_price_precision()
        min_tick_size = temp_analyzer.get_min_tick_size()
        if min_tick_size <= 0:
            lg.error(f"Cannot calculate TSL: Invalid min tick size ({min_tick_size}) for {symbol}.")
            return False

        # 1. Calculate Activation Price
        activation_price = None
        if activation_percentage > 0:
            activation_offset = entry_price * activation_percentage
            if side == 'long':
                # Activate when price moves UP by the percentage
                raw_activation = entry_price + activation_offset
                # Round UP away from entry price for activation, respecting ticks
                activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                # Ensure activation is strictly above entry by at least one tick
                if activation_price <= entry_price:
                    activation_price = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
            else:  # side == 'short'
                # Activate when price moves DOWN by the percentage
                raw_activation = entry_price - activation_offset
                # Round DOWN away from entry price for activation, respecting ticks
                activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                # Ensure activation is strictly below entry by at least one tick
                if activation_price >= entry_price:
                    activation_price = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)
        else:
            # Activate immediately (use activation price '0')
            activation_price = Decimal('0')
            lg.info(f"TSL activation percentage is zero for {symbol}, setting immediate activation (API value '0').")

        # Ensure activation price is non-negative (can be zero)
        if activation_price is None or activation_price < 0:
            lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) is invalid for {symbol}. Cannot set TSL.{RESET}")
            return False

        # 2. Calculate Trailing Stop Distance (in price points)
        # Bybit API: trailingStop is the absolute price distance (e.g., 10 for $10 trail)
        # Calculate this based on the callback rate applied to the entry price (or activation price if preferred)
        # Using entry price provides a consistent distance initially.
        trailing_distance_raw = entry_price * callback_rate
        # Round distance UP to ensure it's at least the calculated value, respecting tick size
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size

        # Ensure minimum distance respects tick size and is positive
        if trailing_distance < min_tick_size:
            lg.warning(f"{NEON_YELLOW}Calculated TSL distance {trailing_distance} is smaller than min tick size {min_tick_size} for {symbol}. Adjusting to min tick size.{RESET}")
            trailing_distance = min_tick_size
        elif trailing_distance <= 0:
            lg.error(f"{NEON_RED}Calculated TSL distance is zero or negative ({trailing_distance}) for {symbol}. Check callback rate ({callback_rate}) and entry price ({entry_price}).{RESET}")
            return False

        # Format prices/distances for logging
        act_price_log = f"{activation_price:.{price_precision}f}" if activation_price > 0 else '0 (Immediate)'
        trail_dist_log = f"{trailing_distance:.{price_precision}f}"
        tp_log = f"{take_profit_price:.{price_precision}f}" if take_profit_price and take_profit_price > 0 else "N/A"

        lg.info(f"Calculated TSL Params for {symbol} ({side.upper()}):")
        lg.info(f"  Entry Price: {entry_price:.{price_precision}f}")
        lg.info(f"  Callback Rate: {callback_rate:.3%}")
        lg.info(f"  Activation Pct: {activation_percentage:.3%}")
        lg.info(f"  => Activation Price (API): {act_price_log}")
        lg.info(f"  => Trailing Distance (API): {trail_dist_log}")
        lg.info(f"  Take Profit (Optional): {tp_log}")

        # 3. Call the helper function to set TSL (and TP if provided)
        # Pass the VALIDATED & CALCULATED TSL parameters
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,  # Pass the full position info dict
            logger=lg,
            stop_loss_price='0',  # Explicitly cancel fixed SL when setting TSL by passing '0'
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,  # Pass calculated distance
            tsl_activation_price=activation_price  # Pass calculated activation price (can be 0)
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL parameters for {symbol}: {e}{RESET}", exc_info=True)
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

    # Determine required kline history (can be refined)
    kline_limit = 500  # Ensure enough for long lookbacks + indicator buffer

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50:  # Need a reasonable minimum history
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping analysis cycle.{RESET}")
        return

    # Fetch current price
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close from klines as fallback.{RESET}")
        try:
            # Ensure last close is valid Decimal
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0:
                current_price = Decimal(str(last_close_val))
                lg.info(f"Using last close price: {current_price}")
            else:
                lg.error(f"{NEON_RED}Last close price ({last_close_val}) is also invalid. Cannot proceed for {symbol}.{RESET}")
                return
        except (IndexError, ValueError, TypeError) as e:
            lg.error(f"{NEON_RED}Error getting last close price for {symbol}: {e}. Cannot proceed.{RESET}")
            return

    # Fetch order book if enabled and weighted
    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators", {}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)

    # --- 2. Analyze Data & Generate Signal ---
    analyzer = TradingAnalyzer(
        df=klines_df.copy(),  # Pass a copy to avoid modification issues
        logger=lg,
        config=config,
        market_info=market_info
    )

    # Check if analyzer initialized correctly (indicators calculated)
    if not analyzer.indicator_values:
        lg.error(f"{NEON_RED}Indicator calculation failed or produced no values for {symbol}. Skipping signal generation.{RESET}")
        return

    # Generate the trading signal
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)

    # Calculate potential initial SL/TP based on *current* price and ATR (used for sizing and potential fixed protection IF NO POSITION EXISTS)
    # Note: If a position exists, SL/TP should be based on the *entry* price.
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)
    # Get other analysis results if needed
    # fib_levels = analyzer.get_nearest_fibonacci_levels(current_price)
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr_float = analyzer.indicator_values.get("ATR")  # Get float ATR

    # --- 3. Log Analysis Summary ---
    # (Signal logging is handled within generate_trading_signal now)
    atr_log = f"{current_atr_float:.{price_precision + 1}f}" if current_atr_float and pd.notna(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision}f}" if tp_potential else 'N/A'

    lg.info(f"ATR: {atr_log}")
    lg.info(f"Potential Initial SL (for new trade sizing): {sl_pot_log}")
    lg.info(f"Potential Initial TP (for new trade): {tp_pot_log}")
    # lg.info(f"Nearest Fib Levels: " + ", ".join([f"{name}={level:.{price_precision}f}" for name, level in fib_levels[:3]]))
    tsl_enabled = config.get('enable_trailing_stop')
    be_enabled = config.get('enable_break_even')
    lg.info(f"Trailing Stop: {'Enabled' if tsl_enabled else 'Disabled'} | Break Even: {'Enabled' if be_enabled else 'Disabled'}")

    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
        # Log cycle time even if not trading
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # --- Get Current Position Status ---
    # This is a critical step, ensure it's robust and includes SL/TP/TSL info
    open_position = get_open_position(exchange, symbol, lg)  # Returns enhanced position dict or None

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Open Position: Initiating Trade Sequence for {symbol} ***")

            # --- Pre-Trade Checks & Setup ---
            # a) Check Balance
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None:  # fetch_balance returns None on critical errors
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot proceed, failed to fetch balance for {QUOTE_CURRENCY}.{RESET}")
                return
            if balance <= 0:  # Check if balance is actually positive
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Insufficient balance ({balance} {QUOTE_CURRENCY}).{RESET}")
                return

            # b) Check if potential SL calculation succeeded (needed for sizing)
            if sl_potential is None:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot proceed, Potential Initial Stop Loss calculation failed (ATR valid?). Risk cannot be determined.{RESET}")
                return

            # c) Set Leverage (only for contracts)
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        # Decide if failure to set leverage is critical
                        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set/confirm leverage to {leverage}x. Cannot proceed safely.{RESET}")
                        return  # Abort trade if leverage setting fails
                else:
                    lg.warning(f"Leverage setting skipped for {symbol}: Configured leverage is zero or negative ({leverage}).")
            else:
                lg.info(f"Leverage setting skipped for {symbol} (Spot market).")

            # d) Calculate Position Size using potential SL
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_potential,  # Use potential SL based on current price for sizing
                entry_price=current_price,  # Use current price as entry estimate for sizing
                market_info=market_info,
                exchange=exchange,  # Pass exchange for formatting helpers
                logger=lg
            )

            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size calculated ({position_size}). Check balance, risk, SL, market limits, and logs.{RESET}")
                return

            # --- Place Initial Market Order ---
            lg.info(f"==> Placing {signal} market order for {symbol} | Size: {position_size} <==")
            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                trade_signal=signal,
                position_size=position_size,  # Pass Decimal size
                market_info=market_info,
                logger=lg,
                reduce_only=False  # Opening trade
            )

            # --- Post-Order: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                lg.info(f"Order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for position confirmation...")

                # Wait for exchange to potentially process the order and update position state
                time.sleep(POSITION_CONFIRM_DELAY)

                # Attempt to confirm the position *after* the delay
                lg.info(f"Attempting to confirm position for {symbol} after order {order_id}...")
                confirmed_position = get_open_position(exchange, symbol, lg)  # Use enhanced function

                if confirmed_position:
                    # --- Position Confirmed ---
                    try:
                        # Use the reliable fields from the enhanced position dict
                        entry_price_actual_str = confirmed_position.get('entryPrice', confirmed_position.get('info', {}).get('avgPrice'))
                        pos_size_actual_str = confirmed_position.get('contracts')  # Use standardized 'contracts'
                        entry_price_actual = Decimal('0')
                        pos_size_actual = Decimal('0')
                        valid_entry = False

                        # Validate actual entry price and size
                        if entry_price_actual_str and pos_size_actual_str is not None:
                            try:
                                entry_price_actual = Decimal(str(entry_price_actual_str))
                                pos_size_actual = Decimal(str(pos_size_actual_str))  # Already absolute from get_open_position
                                # Use a small threshold for size validation
                                min_size_threshold = Decimal('1e-9')
                                if market_info:
                                    try:
                                        min_amt = market_info.get('limits', {}).get('amount', {}).get('min')
                                        if min_amt is not None:
                                            # Use a fraction of min size, but ensure it's not excessively large or zero
                                            min_size_threshold = max(min_size_threshold, Decimal(str(min_amt)) * Decimal('0.01'))
                                    except Exception: pass

                                if entry_price_actual > 0 and pos_size_actual >= min_size_threshold:
                                    valid_entry = True
                                else:
                                    lg.error(f"Confirmed position has invalid entry price ({entry_price_actual}) or size ({pos_size_actual} vs threshold {min_size_threshold}).")
                            except Exception as parse_err:
                                lg.error(f"Error parsing confirmed position entry/size: {parse_err}")
                        else:
                            lg.error("Confirmed position missing entryPrice or contracts/size information.")

                        if valid_entry:
                            lg.info(f"{NEON_GREEN}Position Confirmed for {symbol}! Actual Entry: ~{entry_price_actual:.{price_precision}f}, Actual Size: {pos_size_actual}{RESET}")

                            # --- Recalculate SL/TP based on ACTUAL entry price ---
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                            # --- Set Protection based on Config (TSL or Fixed SL/TP) ---
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting Trailing Stop Loss for {symbol} (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(
                                    exchange=exchange,
                                    symbol=symbol,
                                    market_info=market_info,
                                    position_info=confirmed_position,  # Pass the fetched position dict
                                    config=config,
                                    logger=lg,
                                    take_profit_price=tp_actual  # Pass optional TP based on actual entry
                                )
                            else:
                                # Set Fixed SL and TP using the helper function
                                lg.info(f"Setting Fixed Stop Loss ({sl_actual}) and Take Profit ({tp_actual}) for {symbol}...")
                                if sl_actual or tp_actual:  # Only call if at least one is valid
                                    protection_set_success = _set_position_protection(
                                        exchange=exchange,
                                        symbol=symbol,
                                        market_info=market_info,
                                        position_info=confirmed_position,
                                        logger=lg,
                                        stop_loss_price=sl_actual,  # Pass Decimal or None
                                        take_profit_price=tp_actual,  # Pass Decimal or None
                                        trailing_stop_distance='0',  # Ensure TSL is cancelled if somehow active
                                        tsl_activation_price='0'
                                    )
                                else:
                                    lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation based on actual entry failed or resulted in None for {symbol}. No fixed protection set.{RESET}")
                                    # Consider if no protection is acceptable or if trade should be closed
                                    protection_set_success = True  # Treat as 'success' in terms of not failing, but no protection set

                            # --- Final Status Log ---
                            if protection_set_success:
                                lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE for {symbol} ({signal}) ===")
                            else:
                                lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION (SL/TP/TSL) for {symbol} ({signal}) ===")
                                lg.warning(f"{NEON_YELLOW}Position is open without automated protection. Manual monitoring/intervention may be required!{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Trade placed, but confirmed position data is invalid. Cannot set protection.{RESET}")
                            lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")

                    except Exception as post_trade_err:
                        lg.error(f"{NEON_RED}Error during post-trade processing (protection setting) for {symbol}: {post_trade_err}{RESET}", exc_info=True)
                        lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")

                else:
                    # --- Position NOT Confirmed ---
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position for {symbol} after {POSITION_CONFIRM_DELAY}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have failed to fill, filled partially, rejected, or there's a significant delay/API issue.{RESET}")
                    lg.warning(f"{NEON_YELLOW}No protection will be set. Manual investigation of order {order_id} and position status required!{RESET}")
                    # Optional: Try fetching the order status itself for more info
                    try:
                        order_status = exchange.fetch_order(order_id, symbol)
                        lg.info(f"Status of order {order_id}: {order_status}")
                    except Exception as fetch_order_err:
                        lg.warning(f"Could not fetch status for order {order_id}: {fetch_order_err}")

            else:
                # place_trade function returned None or a dummy success dict (e.g., already closed)
                if trade_order and trade_order.get('info', {}).get('reason'):
                     lg.info(f"Trade placement for {symbol} ({signal}) resulted in status: {trade_order['info']['reason']}. No further action.")
                else:
                     lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ({signal}). See previous logs for details. ===")
        else:
            # No open position, and signal is HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No trade action taken.")

    # --- Scenario 2: Existing Open Position Found ---
    else:  # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size_str = open_position.get('contracts', 'N/A')  # Use standardized positive size
        entry_price_str = open_position.get('entryPrice', open_position.get('info', {}).get('avgPrice', 'N/A'))
        # Use the enhanced get_open_position which pulls SL/TP/TSL info
        current_sl_price_val = open_position.get('stopLossPrice')  # Can be None or float
        current_tp_price_val = open_position.get('takeProfitPrice')  # Can be None or float
        tsl_distance_dec = open_position.get('trailingStopLossDistanceDecimal', Decimal(0))  # Use parsed Decimal
        is_tsl_active = tsl_distance_dec > 0

        # Log current state including TSL status
        sl_log_str = f"{current_sl_price_val:.{price_precision}f}" if current_sl_price_val else 'N/A'
        tp_log_str = f"{current_tp_price_val:.{price_precision}f}" if current_tp_price_val else 'N/A'
        lg.info(f"Existing {pos_side.upper()} position found for {symbol}. Size: {pos_size_str}, Entry: {entry_price_str}, SL: {sl_log_str}, TP: {tp_log_str}, TSL Active: {is_tsl_active}")

        # Check if the new signal opposes the current position direction
        if (pos_side == 'long' and signal == "SELL") or \
           (pos_side == 'short' and signal == "BUY"):
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position for {symbol}. ***{RESET}")

            # --- Initiate Position Close ---
            lg.info(f"Attempting to close {pos_side} position for {symbol} with a market order...")
            close_order = None
            try:
                # Determine the side needed to close the position
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                # Ensure we have the correct size to close from the position info
                size_to_close_str = open_position.get('contracts')  # Use standardized positive size
                if size_to_close_str is None:
                    raise ValueError(f"Could not determine size of existing {pos_side} position for {symbol} to close.")

                # Use absolute value of the size for the closing order amount
                size_to_close = abs(Decimal(str(size_to_close_str)))
                if size_to_close <= 0:
                    raise ValueError(f"Existing position size {size_to_close_str} is zero or invalid. Cannot close.")

                # --- Place Closing Market Order (reduceOnly=True) ---
                close_order = place_trade(
                    exchange=exchange,
                    symbol=symbol,
                    trade_signal=close_side_signal,  # Side opposite to position
                    position_size=size_to_close,    # Absolute size
                    market_info=market_info,
                    logger=lg,
                    reduce_only=True  # CRITICAL for closing
                )

                if close_order:
                     order_id = close_order.get('id', 'N/A')
                     # If place_trade returned the dummy success for 'position not found/closed', log differently
                     if close_order.get('info', {}).get('reason'):
                          lg.info(f"{NEON_GREEN}Position CLOSE for {symbol} confirmed ({close_order['info']['reason']}).{RESET}")
                     else:
                          lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {order_id}{RESET}")
                          lg.info(f"{NEON_YELLOW}Position should be closed. Verify on exchange if necessary.{RESET}")
                     # We assume the reduceOnly market order will close the position.
                     # Future: Add a check after delay to confirm position is None.
                else:
                     # place_trade failed
                     lg.error(f"{NEON_RED}Failed to place position CLOSE order for {symbol}. Manual intervention required!{RESET}")

            except ValueError as ve:
                lg.error(f"{NEON_RED}Error preparing to close position for {symbol}: {ve}{RESET}")
            except Exception as e:  # Catch any other errors during close attempt
                lg.error(f"{NEON_RED}Unexpected error closing position {symbol}: {e}{RESET}", exc_info=True)
                lg.warning(f"{NEON_YELLOW}Manual check/closure of position {symbol} may be required!{RESET}")

        elif signal == "HOLD" or (signal == "BUY" and pos_side == 'long') or (signal == "SELL" and pos_side == 'short'):
            # --- Manage Existing Position (HOLD signal or signal matches position) ---
            lg.info(f"Signal ({signal}) allows holding existing {pos_side} position for {symbol}.")

            # --- ** Break-Even Stop Logic ** ---
            # Only run BE logic if enabled, if there's a position, and if TSL is NOT currently active
            if config.get("enable_break_even", False) and not is_tsl_active:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    # --- Gather necessary data for BE check ---
                    # Ensure entry price is valid Decimal
                    if not entry_price_str or str(entry_price_str).lower() == 'n/a':
                        raise ValueError("Missing or invalid entry price for BE check")
                    entry_price = Decimal(str(entry_price_str))
                    if entry_price <= 0: raise ValueError(f"Invalid entry price ({entry_price}) for BE check")

                    # Ensure current ATR is valid float/Decimal
                    if current_atr_float is None or pd.isna(current_atr_float) or current_atr_float <= 0:
                        lg.warning("Cannot check break-even: Invalid ATR value.")
                        raise ValueError("Invalid ATR for BE check")  # Skip BE check if ATR invalid
                    current_atr_decimal = Decimal(str(current_atr_float))

                    # Get BE config parameters
                    profit_target_atr_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    offset_ticks = int(config.get("break_even_offset_ticks", 2))

                    # Calculate profit/loss in price points
                    price_diff = Decimal('0')
                    if pos_side == 'long':
                        price_diff = current_price - entry_price
                    else:  # short
                        price_diff = entry_price - current_price

                    # Check if profit target (in price points) is reached
                    profit_target_price_diff = profit_target_atr_multiple * current_atr_decimal

                    lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision}f}, Target Diff={profit_target_price_diff:.{price_precision}f} ({profit_target_atr_multiple}*ATR)")

                    # Check if profit target is reached
                    if price_diff >= profit_target_price_diff:
                        # Calculate target break-even SL price
                        be_stop_price = None
                        if min_tick_size <= 0: raise ValueError(f"Invalid min_tick_size ({min_tick_size}) for BE calculation.")
                        tick_offset = min_tick_size * offset_ticks
                        if pos_side == 'long':
                            # BE SL is entry + offset, rounded UP to nearest tick
                            be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP)
                        else:  # short
                            # BE SL is entry - offset, rounded DOWN to nearest tick
                            be_stop_price = (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                        # Get current SL price as Decimal (handle None, 0, errors)
                        current_sl_price_dec = None
                        if current_sl_price_val is not None:  # Check standard field from get_open_pos
                             try:
                                 sl_dec_temp = Decimal(str(current_sl_price_val))
                                 if sl_dec_temp > 0: current_sl_price_dec = sl_dec_temp
                             except Exception:
                                 lg.warning(f"Could not parse current SL price '{current_sl_price_val}' for BE comparison.")

                        # Determine if update is needed:
                        # - No current SL is set OR
                        # - Target BE SL is better (higher for long, lower for short) than current SL
                        update_be = False
                        if be_stop_price is not None and be_stop_price > 0:  # Ensure target BE is valid
                            if current_sl_price_dec is None:  # No current SL, set BE SL
                                update_be = True
                                lg.info("Profit target hit. No current SL found.")
                            elif pos_side == 'long' and be_stop_price > current_sl_price_dec:  # Move SL up to BE
                                update_be = True
                                lg.info(f"Profit target hit. Current SL {current_sl_price_dec} < Target BE SL {be_stop_price}.")
                            elif pos_side == 'short' and be_stop_price < current_sl_price_dec:  # Move SL down to BE
                                update_be = True
                                lg.info(f"Profit target hit. Current SL {current_sl_price_dec} > Target BE SL {be_stop_price}.")
                            else:
                                lg.debug(f"BE Triggered, but current SL ({current_sl_price_dec}) is already at or better than target BE SL ({be_stop_price}). No update needed.")
                        else:
                            lg.warning(f"Calculated BE Stop Price ({be_stop_price}) is invalid. Cannot update.")

                        # Execute the update if needed
                        if update_be:
                            lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                            # Fetch current TP to preserve it when setting SL
                            current_tp_price_dec = None
                            if current_tp_price_val is not None:  # Use value from get_open_pos
                                try:
                                    tp_dec_temp = Decimal(str(current_tp_price_val))
                                    if tp_dec_temp > 0:
                                        current_tp_price_dec = tp_dec_temp
                                except Exception: pass

                            success = _set_position_protection(
                                exchange=exchange,
                                symbol=symbol,
                                market_info=market_info,
                                position_info=open_position,  # Pass the full position dict
                                logger=lg,
                                stop_loss_price=be_stop_price,  # Pass Decimal BE price
                                take_profit_price=current_tp_price_dec,  # Preserve existing TP (pass Decimal or None)
                                trailing_stop_distance='0',  # Ensure TSL is cancelled/overridden by passing '0'
                                tsl_activation_price='0'    # Ensure TSL activation is also cancelled
                            )
                            if success:
                                lg.info(f"{NEON_GREEN}Break-Even SL successfully set/updated for {symbol}.{RESET}")
                                # Future: Optionally update internal state or re-fetch position info
                            else:
                                lg.error(f"{NEON_RED}Failed to set Break-Even SL for {symbol}. Check logs.{RESET}")
                    else:
                        lg.debug(f"Profit target for break-even not yet reached ({price_diff:.{price_precision}f} < {profit_target_price_diff:.{price_precision}f}).")

                except ValueError as ve:  # Catch specific value errors from data prep
                    lg.warning(f"Skipping BE check due to invalid data: {ve}")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check/execution for {symbol}: {be_err}{RESET}", exc_info=True)
            elif is_tsl_active:
                lg.info(f"Break-even check skipped for {symbol}: Trailing Stop Loss is already active.")
            elif not config.get("enable_break_even", False):
                lg.debug(f"Break-even check skipped for {symbol}: Disabled in config.")

            # --- Other Management Logic (Placeholder) ---
            # TODO: Add logic here to:
            # 1. Check if TSL/Fixed SL was *supposed* to be active but isn't. Retry setting?
            # 2. Monitor proximity to Liq price and potentially reduce risk?
            # 3. Add pyramiding logic if signal matches position strongly? (High Risk, requires careful sizing)
            # 4. Update TP target based on new analysis? (e.g., if price moves significantly)
            lg.debug(f"No other position management actions taken for {symbol} this cycle.")

    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY, console_log_level  # Allow main loop to potentially reload config

    # Use a generic logger for initial setup, then switch to symbol-specific logger
    # Ensure the 'init' logger is set up using setup_logger
    setup_logger("init")  # Creates handlers for 'init' logger name
    init_logger = logging.getLogger("init")  # Get the logger instance
    # Access console handler to potentially change level later
    console_handler = None
    for handler in init_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            console_handler = handler
            break
    # Set console level based on global variable (initially INFO)
    # console_log_level is already defined globally
    if console_handler:
        console_handler.setLevel(console_log_level)

    init_logger.info(f"--- Starting LiveXY Trading Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    init_logger.info(f"Loading configuration from {CONFIG_FILE}...")
    # Reload config here in case defaults were created/updated
    CONFIG = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")  # Ensure quote currency is set globally
    init_logger.info(f"Configuration loaded. Quote Currency: {QUOTE_CURRENCY}")
    # Display key versions
    init_logger.info(f"Using CCXT Version: {ccxt.__version__}")
    init_logger.info(f"Using Pandas Version: {pd.__version__}")
    try:
        init_logger.info(f"Using Pandas TA Version: {ta.version() if callable(ta.version) else getattr(ta, '__version__', 'N/A')}")
    except Exception:
        init_logger.warning("Could not determine pandas_ta version.")

    # --- Safety Checks & User Confirmation ---
    if CONFIG.get("enable_trading"):
        init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED in {CONFIG_FILE} !!!{RESET}")
        if CONFIG.get("use_sandbox"):
            init_logger.warning(f"{NEON_YELLOW}Using SANDBOX environment (Testnet). No real funds at risk.{RESET}")
        else:
            init_logger.warning(f"{NEON_RED}!!! USING REAL MONEY ENVIRONMENT !!! Ensure configuration and risk settings are correct!{RESET}")
        init_logger.warning(f"Key Settings: Risk Per Trade: {CONFIG.get('risk_per_trade', 0) * 100:.2f}%, Leverage: {CONFIG.get('leverage', 0)}x, TSL: {'Enabled' if CONFIG.get('enable_trailing_stop') else 'Disabled'}, BE: {'Enabled' if CONFIG.get('enable_break_even') else 'Disabled'}")

        try:
            confirm = input(f">>> Review settings above. Press {NEON_GREEN}Enter{RESET} to continue, or {NEON_RED}Ctrl+C{RESET} to abort... ")
            init_logger.info("User acknowledged live trading settings. Proceeding...")
        except KeyboardInterrupt:
            init_logger.info("User aborted startup. Exiting.")
            return
    else:
        init_logger.info(f"{NEON_YELLOW}Trading is disabled ('enable_trading': false in config). Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange ---
    init_logger.info("Initializing exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}")
        return
    init_logger.info(f"Exchange {exchange.id} initialized successfully.")

    # --- Symbol and Interval Selection ---
    target_symbol = None
    market_info = None
    while True:
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter symbol to trade (e.g., BTC/USDT, ETH/USDT:USDT): {RESET}").strip()
            if not symbol_input_raw: continue

            # Normalize input: Convert to uppercase, replace common separators if needed
            symbol_input = symbol_input_raw.upper().replace('-', '/')

            # CCXT prefers BASE/QUOTE format, or BASE/QUOTE:QUOTE for linear swaps on some exchanges
            # Try fetching market info directly with the input
            init_logger.info(f"Validating symbol '{symbol_input}'...")
            market_info = get_market_info(exchange, symbol_input, init_logger)

            if market_info:
                target_symbol = market_info['symbol']  # Use the exact symbol confirmed by ccxt
                market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                break  # Symbol validated
            else:
                # If direct match fails, try common variations
                variations = []
                base_curr = ""
                quote_curr = QUOTE_CURRENCY  # Assume default quote if not specified
                if '/' in symbol_input:
                    parts = symbol_input.split('/')
                    base_curr = parts[0]
                    if len(parts) > 1: quote_curr = parts[1].split(':')[0]  # Handle optional :XXX part
                else:  # Assume symbol like BTCUSDT
                    if symbol_input.endswith(quote_curr):
                        base_curr = symbol_input[:-len(quote_curr)]
                    else:
                        base_curr = symbol_input  # Guess base

                if base_curr:
                    # Try BASE/QUOTE
                    variations.append(f"{base_curr}/{quote_curr}")
                    # Try BASE/QUOTE:QUOTE (common for linear perps)
                    variations.append(f"{base_curr}/{quote_curr}:{quote_curr}")

                found_variation = False
                if variations:
                    init_logger.info(f"Symbol '{symbol_input}' not found directly. Trying variations: {variations}")
                    for sym_var in variations:
                        init_logger.debug(f"Trying variation '{sym_var}'...")
                        market_info = get_market_info(exchange, sym_var, init_logger)
                        if market_info:
                            target_symbol = market_info['symbol']
                            market_type_desc = "Contract" if market_info.get('is_contract', False) else "Spot"
                            init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                            found_variation = True
                            break

                if found_variation:
                    break  # Variation validated
                else:
                    init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' and common variations could not be validated on {exchange.id}. Please check the symbol and try again.{RESET}")
                    # Optional: List some available derivative markets for guidance
                    try:
                        markets = exchange.load_markets()
                        deriv_symbols = [s for s, m in markets.items() if m.get('contract', False) and QUOTE_CURRENCY in s]
                        init_logger.info(f"Sample available {QUOTE_CURRENCY} derivative symbols: {deriv_symbols[:10]}")
                    except Exception as e:
                        init_logger.warning(f"Could not fetch available symbols: {e}")

        except Exception as e:
            init_logger.error(f"Error during symbol validation: {e}", exc_info=True)

    # --- Interval Selection ---
    selected_interval = None
    while True:
        interval_input = input(f"{NEON_YELLOW}Enter analysis interval [{'/'.join(VALID_INTERVALS)}] (current default: {CONFIG['interval']}): {RESET}").strip()
        if not interval_input:  # Use default if empty
            interval_input = CONFIG['interval']
            init_logger.info(f"Using default interval from config: {interval_input}")

        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            selected_interval = interval_input
            # Update config in memory for this run (doesn't save back to file)
            CONFIG["interval"] = selected_interval
            ccxt_tf = CCXT_INTERVAL_MAP[selected_interval]
            init_logger.info(f"Using analysis interval: {selected_interval} (CCXT: {ccxt_tf})")
            break
        else:
            init_logger.error(f"{NEON_RED}Invalid interval: '{interval_input}'. Please choose from {VALID_INTERVALS} or press Enter for default.{RESET}")

    # --- Setup Logger for the specific symbol ---
    # Use the validated target_symbol which might include ':'
    symbol_logger = setup_logger(target_symbol)  # Get logger instance for this symbol
    # Ensure symbol logger's console handler also uses the correct level
    if console_handler:
        for handler in symbol_logger.handlers:
             if isinstance(handler, logging.StreamHandler):
                  handler.setLevel(console_log_level)
                  break

    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    symbol_logger.info(f"Using Configuration: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if CONFIG['enable_trailing_stop'] else 'OFF'}, BE={'ON' if CONFIG['enable_break_even'] else 'OFF'}, Trading={'ENABLED' if CONFIG['enable_trading'] else 'DISABLED'}")

    # --- Main Execution Loop ---
    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Starting at {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # --- Optional: Reload config each loop? ---
                # Useful for dynamic adjustments without restarting the bot
                # CONFIG = load_config(CONFIG_FILE)
                # QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Reload quote currency too
                # symbol_logger.debug("Config reloaded.")
                # --- End Optional Reload ---

                # Perform analysis and potentially trade/manage position
                analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger)

            # --- Handle Specific Errors within the loop ---
            except ccxt.RateLimitExceeded as e:
                # Extract wait time from error message if possible
                wait_time = RETRY_DELAY_SECONDS * 5  # Default wait time
                try:
                     if 'try again in' in str(e).lower():
                         wait_time_ms_str = re.search(r'try again in (\d+)ms', str(e).lower())
                         if wait_time_ms_str:
                             wait_time = int(wait_time_ms_str.group(1)) / 1000
                             wait_time = max(1, int(wait_time + 1))
                     elif 'rate limit' in str(e).lower():
                         match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                         if match:
                             num = int(match.group(1))
                             unit = match.group(2)
                             wait_time = num / 1000 if unit == 'ms' else num
                             wait_time = max(1, int(wait_time + 1))
                except Exception: pass
                symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting {wait_time}s...{RESET}")
                time.sleep(wait_time)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                symbol_logger.error(f"{NEON_RED}Network error during main loop: {e}. Waiting {RETRY_DELAY_SECONDS * 3}s...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS * 3)  # Longer wait for network issues
            except ccxt.AuthenticationError as e:
                symbol_logger.critical(f"{NEON_RED}CRITICAL: Authentication Error in loop: {e}. API keys may be invalid, expired, permissions revoked, or IP changed. Stopping bot.{RESET}")
                break  # Stop the bot on authentication errors
            except ccxt.ExchangeNotAvailable as e:
                symbol_logger.error(f"{NEON_RED}Exchange not available (e.g., temporary outage): {e}. Waiting 60s...{RESET}")
                time.sleep(60)
            except ccxt.OnMaintenance as e:
                symbol_logger.error(f"{NEON_RED}Exchange is under maintenance: {e}. Waiting 5 minutes...{RESET}")
                time.sleep(300)
            # --- Handle Generic Errors ---
            except Exception as loop_error:
                symbol_logger.error(f"{NEON_RED}An uncaught error occurred in the main analysis loop: {loop_error}{RESET}", exc_info=True)
                symbol_logger.info("Attempting to continue after 15s delay...")
                time.sleep(15)

            # --- Loop Delay Calculation ---
            # Ensure the loop runs approximately every LOOP_DELAY_SECONDS
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle finished in {elapsed_time:.2f}s. Waiting {sleep_time:.2f}s for next cycle...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as critical_error:
        # Catch errors outside the main try/except loop (e.g., during setup before logger swap)
        init_logger.critical(f"{NEON_RED}A critical unhandled error occurred outside the main loop: {critical_error}{RESET}", exc_info=True)
        # Log to symbol logger too if it was initialized
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
            symbol_logger.critical(f"{NEON_RED}A critical unhandled error occurred outside the main loop: {critical_error}{RESET}", exc_info=True)
    finally:
        shutdown_msg = f"--- LiveXY Trading Bot for {target_symbol if target_symbol else 'N/A'} Stopping ---"
        # Use init logger as symbol logger might not be defined if error was early
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
            symbol_logger.info(shutdown_msg)

        # --- Close Exchange Connection ---
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Closing exchange connection...")
                exchange.close()
                init_logger.info("Exchange connection closed.")
            except Exception as close_err:
                init_logger.error(f"Error closing exchange connection: {close_err}")

        # --- Final Log Message ---
        final_msg = "Bot stopped."
        init_logger.info(final_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
            symbol_logger.info(final_msg)
        logging.shutdown()  # Ensure all handlers are flushed and closed


if __name__ == "__main__":
    # Note: The self-write block from the original is removed for clarity and
    # because it's generally not a standard practice.
    # If you need the self-write functionality, uncomment and adapt the block below.
    """
    output_filename = "livexy.py" # Ensure output filename matches script name
    try:
        current_script_path = os.path.abspath(__file__)
        output_script_path = os.path.abspath(output_filename)

        if not os.path.exists(output_script_path) and current_script_path != output_script_path:
             with open(current_script_path, 'r', encoding='utf-8') as current_file:
                 script_content = current_file.read()
             with open(output_script_path, 'w', encoding='utf-8') as output_file:
                 header = f"# {output_filename}\n# Enhanced version focusing on stop-loss/take-profit mechanisms, including break-even logic.\n\n"
                 script_content = re.sub(r'^# livex[xy]\\.py.*\n(# Enhanced version.*\n)*', '', script_content, flags=re.MULTILINE)
                 output_file.write(header + script_content)
             print(f"Enhanced script content written to {output_filename}")
             print(f"{NEON_YELLOW}Note: Running the *original* script file. The new '{output_filename}' was just created.{RESET}")
        elif current_script_path != output_script_path:
             print(f"Skipping self-write: {output_filename} already exists.")

        main()
    except NameError:
         print(f"{NEON_YELLOW}Warning: Could not determine current script path (__file__ undefined). Skipping self-write check.{RESET}")
         main() # Still attempt to run main
    except Exception as e:
        print(f"Error during self-write check or running main: {e}")
        try:
            main()
        except Exception as main_e:
             print(f"Error running main after file write/check failure: {main_e}")
    """
    # Directly run main()
    main()
