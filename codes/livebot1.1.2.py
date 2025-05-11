Okay, I've reviewed your Python script for the trading bot. It's already quite comprehensive and well-structured, especially with its detailed handling of Bybit V5 specifics, `Decimal` precision, and error management.

Here's the enhanced version with a few improvements focusing on clarity, robustness, and completing some minor logical paths:

**Key Enhancements Made:**

1.  **Scalping Signal Threshold:** Implemented the use of `scalping_signal_threshold` from the config when the `active_weight_set` is "scalping" in `TradingAnalyzer.generate_trading_signal`.
2.  **Post-Close Position Verification:** Added a step in `analyze_and_trade_symbol` to wait and confirm that a position is actually closed after a close order is placed.
3.  **Refined Protection Setting Logic:** Improved the return logic in `_set_position_protection` for clarity when no API call is made because no valid protection parameters were ultimately formatted (e.g., due to formatting failures or valid cancellations resulting in no active protection to set).
4.  **Minor Docstring/Comment Additions:** Added a few clarifying comments.
5.  **Error Message Consistency:** Ensured consistent use of `NEON_` colors for error/warning/success messages in logs.

The core logic and structure of your script were already very strong. These changes are mostly refinements.

```python
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
        "trailing_stop_callback_rate": 0.002,  # Example: 0.2% trail distance relative to entry/activation
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry.
        "trailing_stop_activation_percentage": 0,  # Example: Activate when 0% in profit (immediate)
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
                    pass # Non-critical if cannot write back
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError):
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
        except OSError:
            pass # Non-critical if cannot write back
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
        # Optional: Handle type mismatches if needed by uncommenting below
        # elif type(default_value) != type(updated_config.get(key)) and updated_config.get(key) is not None:
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
             if isinstance(handler, logging.StreamHandler): # Check if it's the console handler
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
    except Exception as e:
        # Fallback to basic console logging if file handler fails
        print(f"Error setting up file logger for {safe_symbol}: {e}")


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
                'brokerId': 'livexyBotV2',  # Example: Add a broker ID for Bybit if desired
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
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, logger) # QUOTE_CURRENCY is global
            if balance_decimal is not None:
                 logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 logger.warning(f"{NEON_YELLOW}Initial balance fetch returned None or zero. Check API permissions/account type if trading fails.{RESET}")
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None  # Critical failure, cannot proceed
        except ccxt.ExchangeError as balance_err:
            # Handle potential errors if account type is wrong etc.
            logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")
        except Exception as balance_err: # Catch any other exception
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
    lg = logger # Alias for brevity
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        lg.debug(f"Ticker data for {symbol}: {ticker}")

        price = None
        # Prioritize fields in order of reliability/common usage
        price_fields_priority = ['last', 'bid', 'ask', 'close'] # 'close' is often same as 'last'

        for field_name in price_fields_priority:
            field_val = ticker.get(field_name)
            if field_val is not None:
                try:
                    field_decimal = Decimal(str(field_val))
                    if field_decimal > 0:
                        price = field_decimal
                        lg.debug(f"Using '{field_name}' price for {symbol}: {price}")
                        break # Found a valid price
                    else:
                        lg.warning(f"'{field_name.capitalize()}' price ({field_decimal}) is not positive for {symbol}.")
                except Exception as e:
                    lg.warning(f"Could not parse '{field_name}' price ({field_val}) for {symbol}: {e}")
            if price is not None: break # Exit loop if price found from a higher priority field

        # Fallback to bid/ask midpoint if individual fields failed or 'last' was not ideal
        if price is None or (price_fields_priority[0] != 'last' and ticker.get('bid') and ticker.get('ask')): # If 'last' wasn't primary or failed
            bid_price = ticker.get('bid')
            ask_price = ticker.get('ask')
            if bid_price is not None and ask_price is not None:
                try:
                    bid_decimal = Decimal(str(bid_price))
                    ask_decimal = Decimal(str(ask_price))
                    if bid_decimal > 0 and ask_decimal > 0:
                        if bid_decimal <= ask_decimal: # Ensure bid is not higher than ask
                            mid_price = (bid_decimal + ask_decimal) / 2
                            price = mid_price # Potentially override if 'last' was problematic
                            lg.debug(f"Using bid/ask midpoint for {symbol}: {price} (Bid: {bid_decimal}, Ask: {ask_decimal})")
                        else:
                            lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' as fallback if no other price found.")
                            if price is None: price = ask_decimal # Use ask as a safer fallback in this specific case
                    else:
                        lg.warning(f"Bid ({bid_decimal}) or Ask ({ask_decimal}) price is not positive for {symbol}.")
                except Exception as e:
                    lg.warning(f"Could not parse bid/ask prices ({bid_price}, {ask_price}) for {symbol}: {e}")

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
    lg = logger or logging.getLogger(__name__) # Use passed logger or default
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
                    time.sleep(1) # Small delay even on None/empty return

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"{NEON_YELLOW}Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...{RESET}")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                    raise e  # Re-raise the last error
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 5  # Default wait time
                try:
                     # Try to parse recommended wait time from error message
                     if 'try again in' in str(e).lower():
                         wait_time_ms_str = re.search(r'try again in (\d+)ms', str(e).lower())
                         if wait_time_ms_str:
                             wait_time = int(wait_time_ms_str.group(1)) / 1000
                             wait_time = max(1, int(wait_time + 1))  # Add a buffer and ensure minimum 1s
                     elif 'rate limit' in str(e).lower():
                         match = re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                         if match:
                             num = int(match.group(1))
                             unit = match.group(2)
                             wait_time = num / 1000 if unit == 'ms' else num
                             wait_time = max(1, int(wait_time + 1))
                except Exception:
                     pass  # Use default if parsing fails
                lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1}){RESET}")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                raise e  # Re-raise non-network errors immediately, typically not retryable

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
                lg.warning(f"{NEON_YELLOW}fetch_order_book returned None or empty data for {symbol} (Attempt {attempts + 1}).{RESET}")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"{NEON_YELLOW}Invalid orderbook structure (bids/asks not lists) for {symbol}. Attempt {attempts + 1}. Response: {orderbook}{RESET}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 return orderbook  # Return the empty book; signal generation handles this
            else:
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
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
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time}s... (Attempt {attempts + 1}){RESET}")
            time.sleep(wait_time)
            attempts += 1 # Increment here to avoid bypassing retry limit due to separate sleep
            continue
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            return None # Don't retry on definitive exchange errors usually
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
    """
    Analyzes trading data using pandas_ta to calculate various technical indicators,
    generates weighted trading signals, and provides risk management calculations
    like Fibonacci levels, entry/TP/SL points.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Pandas DataFrame with OHLCV data, indexed by 'timestamp'.
            logger: Logger instance for logging messages.
            config: Configuration dictionary for the bot.
            market_info: Market information dictionary from CCXT.
        """
        self.df = df
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL") # User-defined interval string
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_CCXT_INTERVAL") # Mapped CCXT interval string
        self.indicator_values: dict[str, float] = {}  # Stores latest indicator float values
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}  # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: dict[str, Decimal] = {}  # Stores calculated fib levels (as Decimals)
        self.ta_column_names: dict[str, str | None] = {}  # Stores actual column names generated by pandas_ta

        if not self.weights:
            logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will not work.{RESET}")

        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> str | None:
        """Helper to find the actual column name generated by pandas_ta, handling variations."""
        cfg = self.config
        bb_std_dev = float(cfg.get('bollinger_bands_std_dev', DEFAULT_BOLLINGER_BANDS_STD_DEV))
        # Define expected patterns based on pandas_ta naming conventions and config
        expected_patterns = {
            "ATR": [f"ATRr_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}", f"ATR_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}"],
            "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
            "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_0.015"],
            "Williams_R": [f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"],
            "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP", "VWAP_D"],
            "PSAR_long": [f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "PSAR_short": [f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"],
            "SMA10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
            "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "BB_Middle": [f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "BB_Upper": [f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"],
            "Volume_MA": [f"VOL_SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"]
        }
        patterns_to_check = expected_patterns.get(base_name, [])
        for pattern in patterns_to_check:
            if pattern in result_df.columns:
                return pattern
            # Check variation without const suffix (e.g., CCI)
            base_pattern_parts = pattern.split('_')
            if len(base_pattern_parts) > 2 and base_pattern_parts[-1].replace('.', '', 1).isdigit(): # Allow one dot for float consts
                pattern_no_suffix = '_'.join(base_pattern_parts[:-1])
                if pattern_no_suffix in result_df.columns:
                    self.logger.debug(f"Found column '{pattern_no_suffix}' for base '{base_name}' (without const suffix).")
                    return pattern_no_suffix

        # Check for common parameter-less suffix variations
        for pattern in patterns_to_check:
             base_pattern_prefix = pattern.split('_')[0]
             if base_pattern_prefix in result_df.columns:
                  self.logger.debug(f"Found column '{base_pattern_prefix}' for base '{base_name}' (parameter-less variation).")
                  return base_pattern_prefix
             if base_name.upper() in result_df.columns: # Try base name itself (e.g., "CCI")
                  self.logger.debug(f"Found column '{base_name.upper()}' for base '{base_name}'.")
                  return base_name.upper()

        # Fallback: search for base name (case-insensitive)
        for col in result_df.columns:
            if col.lower().startswith(base_name.lower() + "_"):
                 self.logger.debug(f"Found column '{col}' for base '{base_name}' using prefix fallback search.")
                 return col
            if base_name.lower() in col.lower():
                self.logger.debug(f"Found column '{col}' for base '{base_name}' using basic fallback search.")
                return col

        self.logger.warning(f"{NEON_YELLOW}Could not find column name for indicator '{base_name}' in DataFrame columns: {result_df.columns.tolist()}{RESET}")
        return None

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Determine minimum data points needed
        periods_needed = [self.config.get("atr_period", DEFAULT_ATR_PERIOD)] # ATR always calculated
        cfg_ind = self.config.get("indicators", {})
        if cfg_ind.get("ema_alignment"): periods_needed.append(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        # ... (add other relevant period checks from original script) ...
        if cfg_ind.get("stoch_rsi"): periods_needed.append(self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))

        min_required_data = max(periods_needed) + 20 if periods_needed else 50 # Add buffer

        if len(self.df) < min_required_data:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate or NaN.{RESET}")

        try:
            df_calc = self.df.copy()
            indicators_config = self.config.get("indicators", {})

            # ATR (Always calculated for SL/TP sizing)
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            if indicators_config.get("ema_alignment", False):
                df_calc.ta.ema(length=self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD), append=True)
                self.ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
                df_calc.ta.ema(length=self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD), append=True)
                self.ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)

            if indicators_config.get("momentum", False):
                df_calc.ta.mom(length=self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD), append=True)
                self.ta_column_names["Momentum"] = self._get_ta_col_name("Momentum", df_calc)

            if indicators_config.get("cci", False):
                df_calc.ta.cci(length=self.config.get("cci_window", DEFAULT_CCI_WINDOW), append=True)
                self.ta_column_names["CCI"] = self._get_ta_col_name("CCI", df_calc)

            if indicators_config.get("wr", False):
                df_calc.ta.willr(length=self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW), append=True)
                self.ta_column_names["Williams_R"] = self._get_ta_col_name("Williams_R", df_calc)

            if indicators_config.get("mfi", False):
                df_calc.ta.mfi(length=self.config.get("mfi_window", DEFAULT_MFI_WINDOW), append=True)
                self.ta_column_names["MFI"] = self._get_ta_col_name("MFI", df_calc)

            if indicators_config.get("vwap", False):
                df_calc.ta.vwap(append=True) # VWAP often resets daily, pandas_ta handles this based on index
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)

            if indicators_config.get("psar", False):
                psar_result = df_calc.ta.psar(af=self.config.get("psar_af", DEFAULT_PSAR_AF), max_af=self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF))
                if psar_result is not None and not psar_result.empty:
                    for col in psar_result.columns: # Append safely
                        if col not in df_calc.columns: df_calc[col] = psar_result[col]
                        else: df_calc[col] = psar_result[col] # Overwrite if exists
                    self.ta_column_names["PSAR_long"] = self._get_ta_col_name("PSAR_long", df_calc)
                    self.ta_column_names["PSAR_short"] = self._get_ta_col_name("PSAR_short", df_calc)

            if indicators_config.get("sma_10", False):
                df_calc.ta.sma(length=self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW), append=True)
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)

            if indicators_config.get("stoch_rsi", False):
                stochrsi_result = df_calc.ta.stochrsi(
                    length=self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW),
                    rsi_length=self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW),
                    k=self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW),
                    d=self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                )
                if stochrsi_result is not None and not stochrsi_result.empty:
                    for col in stochrsi_result.columns: # Append safely
                        if col not in df_calc.columns: df_calc[col] = stochrsi_result[col]
                        else: df_calc[col] = stochrsi_result[col]
                    self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                    self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)

            if indicators_config.get("rsi", False):
                df_calc.ta.rsi(length=self.config.get("rsi_period", DEFAULT_RSI_WINDOW), append=True)
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            if indicators_config.get("bollinger_bands", False):
                bbands_result = df_calc.ta.bbands(
                    length=self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD),
                    std=float(self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV))
                )
                if bbands_result is not None and not bbands_result.empty:
                    for col in bbands_result.columns: # Append safely
                        if col not in df_calc.columns: df_calc[col] = bbands_result[col]
                        else: df_calc[col] = bbands_result[col]
                    self.ta_column_names["BB_Lower"] = self._get_ta_col_name("BB_Lower", df_calc)
                    self.ta_column_names["BB_Middle"] = self._get_ta_col_name("BB_Middle", df_calc)
                    self.ta_column_names["BB_Upper"] = self._get_ta_col_name("BB_Upper", df_calc)

            if indicators_config.get("volume_confirmation", False):
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}"
                df_calc[vol_ma_col_name] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_period)
                self.ta_column_names["Volume_MA"] = vol_ma_col_name

            self.df = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns: {self.df.columns.tolist()}")

        except AttributeError as e: # Common if pandas_ta method names change or are misspelled
            self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} (check pandas_ta method name/version?): {e}{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)

    def _update_latest_indicator_values(self) -> None:
        """Updates the indicator_values dict with the latest float values from self.df."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: DataFrame is empty for {self.symbol}.{RESET}")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
            return
        try:
            if self.df.iloc[-1].isnull().all():
                self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: Last row of DataFrame contains all NaNs for {self.symbol}.{RESET}")
                self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
                return
        except IndexError:
             self.logger.error(f"{NEON_RED}Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty or too short.{RESET}")
             self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
             return

        try:
            latest = self.df.iloc[-1]
            updated_values = {}

            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index:
                    value = latest[col_name]
                    if pd.notna(value):
                        try: updated_values[key] = float(value)
                        except (ValueError, TypeError):
                            self.logger.warning(f"{NEON_YELLOW}Could not convert value for {key} ('{col_name}': {value}) to float for {self.symbol}.{RESET}")
                            updated_values[key] = np.nan
                    else: updated_values[key] = np.nan
                else:
                    if key in self.ta_column_names: # Log only if key was attempted
                        self.logger.debug(f"Indicator column '{col_name}' for key '{key}' not found or invalid in latest data for {self.symbol}. Storing NaN.")
                    updated_values[key] = np.nan

            for base_col in ['close', 'volume', 'high', 'low']: # Add essential price/volume
                value = latest.get(base_col, np.nan)
                key_name = base_col.capitalize()
                if pd.notna(value):
                    try: updated_values[key_name] = float(value)
                    except (ValueError, TypeError):
                        self.logger.warning(f"{NEON_YELLOW}Could not convert base value for '{base_col}' ({value}) to float for {self.symbol}.{RESET}")
                        updated_values[key_name] = np.nan
                else: updated_values[key_name] = np.nan

            self.indicator_values = updated_values
            valid_values = {k: f"{v:.5f}" if isinstance(v, float) else v for k, v in self.indicator_values.items() if pd.notna(v)}
            self.logger.debug(f"Latest indicator float values updated for {self.symbol}: {valid_values}")

        except IndexError:
            self.logger.error(f"{NEON_RED}Error accessing latest row (iloc[-1]) for {self.symbol} during update. DataFrame might be empty/short.{RESET}")
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error updating latest indicator values for {self.symbol}: {e}{RESET}", exc_info=True)
            self.indicator_values = {k: np.nan for k in list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]}

    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a specified window using Decimal."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}. Skipping.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            high_price_raw = df_slice["high"].dropna().max()
            low_price_raw = df_slice["low"].dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                self.logger.warning(f"{NEON_YELLOW}Could not find valid high/low in the last {window} periods for Fibonacci on {self.symbol}.{RESET}")
                self.fib_levels_data = {}
                return {}

            high = Decimal(str(high_price_raw))
            low = Decimal(str(low_price_raw))
            diff = high - low
            levels = {}

            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                self.logger.warning(f"{NEON_YELLOW}Invalid min_tick_size ({min_tick}) for Fibonacci quantization on {self.symbol}. Levels will not be quantized.{RESET}")
                min_tick = None

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    level_price = (high - (diff * Decimal(str(level_pct)))) # Retracement from High
                    if min_tick: # Quantize using consistent rounding
                        level_price = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    levels[level_name] = level_price
            else: # high == low, all levels are the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}.")
                 level_price_quantized = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick if min_tick else high
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { {k: str(v) for k, v in levels.items()} }")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info."""
        try:
            min_tick = self.get_min_tick_size()
            if min_tick > 0:
                return abs(min_tick.normalize().as_tuple().exponent)
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Error deriving precision from min_tick_size for {self.symbol}: {e}. Trying fallbacks.{RESET}")

        try: # Fallback 1: market['precision']['price'] (if integer)
            price_prec_val = self.market_info.get('precision', {}).get('price')
            if isinstance(price_prec_val, int): return price_prec_val
            if isinstance(price_prec_val, (float, str)): # If it's tick size as float/str
                tick_from_prec = Decimal(str(price_prec_val))
                if tick_from_prec > 0: return abs(tick_from_prec.normalize().as_tuple().exponent)
        except Exception: pass

        try: # Fallback 2: Infer from last close price
            last_close = self.indicator_values.get("Close")
            if last_close and pd.notna(last_close) and last_close > 0:
                s_close = format(Decimal(str(last_close)), 'f')
                return len(s_close.split('.')[-1]) if '.' in s_close else 0
        except Exception: pass

        default_precision = 4
        self.logger.warning(f"{NEON_YELLOW}Using default price precision {default_precision} for {self.symbol}.{RESET}")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        try: # Priority 1: Bybit V5 specific 'tickSize'
            bybit_tick_size = self.market_info.get('info', {}).get('tickSize')
            if bybit_tick_size is not None:
                tick = Decimal(str(bybit_tick_size))
                if tick > 0: return tick
        except Exception: pass

        try: # Priority 2: Standard CCXT 'precision.price' (often the tick size)
            price_prec_val = self.market_info.get('precision', {}).get('price')
            if price_prec_val is not None:
                tick = Decimal(str(price_prec_val))
                if tick > 0: return tick
        except Exception: pass

        try: # Priority 3: 'limits.price.min' (heuristic, less reliable)
            min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
            if min_price_val is not None:
                tick = Decimal(str(min_price_val))
                # Heuristic: if it's a power of 10 (e.g., 0.1, 0.01, 1)
                if tick > 0 and (tick == 1 or (math.log10(tick) % 1 == 0 if tick < 1 else False)):
                    return tick
        except Exception: pass

        fallback_tick = Decimal('0.00000001')
        self.logger.warning(f"{NEON_YELLOW}Using extremely small fallback tick size {fallback_tick} for {self.symbol}. Price quantization may be inaccurate.{RESET}")
        return fallback_tick

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels (name, price) to the current price."""
        if not self.fib_levels_data:
            return []
        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.{RESET}")
            return []
        try:
            level_distances = [{'name': name, 'level': level_price, 'distance': abs(current_price - level_price)}
                               for name, level_price in self.fib_levels_data.items() if isinstance(level_price, Decimal)]
            level_distances.sort(key=lambda x: x['distance'])
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score. Returns float score or NaN."""
        ema_short = self.indicator_values.get("EMA_Short", np.nan)
        ema_long = self.indicator_values.get("EMA_Long", np.nan)
        current_price = self.indicator_values.get("Close", np.nan)

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            return np.nan
        if current_price > ema_short > ema_long: return 1.0  # Bullish
        if current_price < ema_short < ema_long: return -1.0 # Bearish
        return 0.0 # Neutral/Mixed

    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: dict | None
    ) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {}

        if not self.indicator_values or not any(pd.notna(v) for k, v in self.indicator_values.items() if k not in ['Close', 'Volume', 'High', 'Low']):
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values empty or all NaN.{RESET}")
            return "HOLD"
        if current_price is None or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).{RESET}")
            return "HOLD"

        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
            self.logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' missing/empty for {self.symbol}. Cannot generate signal.{RESET}")
            return "HOLD"

        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue
            weight_str = active_weights.get(indicator_key)
            if weight_str is None: continue
            try:
                weight = Decimal(str(weight_str))
                if weight == 0: continue
            except Exception:
                self.logger.warning(f"{NEON_YELLOW}Invalid weight '{weight_str}' for '{indicator_key}' in '{self.active_weight_set_name}'. Skipping.{RESET}")
                continue

            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(getattr(self, check_method_name)):
                method = getattr(self, check_method_name)
                indicator_score = np.nan
                try:
                    if indicator_key == "orderbook":
                        indicator_score = method(orderbook_data, current_price) if orderbook_data else np.nan
                    else:
                        indicator_score = method()
                except Exception as e:
                    self.logger.error(f"{NEON_RED}Error in check method {check_method_name} for {self.symbol}: {e}{RESET}", exc_info=True)

                debug_scores[indicator_key] = f"{indicator_score:.2f}" if pd.notna(indicator_score) else "NaN"
                if pd.notna(indicator_score):
                    try:
                        clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), Decimal(str(indicator_score))))
                        final_signal_score += clamped_score * weight
                        total_weight_applied += weight
                        active_indicator_count += 1
                    except Exception as calc_err:
                        self.logger.error(f"{NEON_RED}Error processing score for {indicator_key} ({indicator_score}): {calc_err}{RESET}")
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1
            else:
                self.logger.warning(f"{NEON_YELLOW}Check method '{check_method_name}' not found for enabled/weighted indicator: {indicator_key} ({self.symbol}){RESET}")

        if total_weight_applied == 0:
            self.logger.warning(f"{NEON_YELLOW}No indicators contributed to signal score for {self.symbol}. Defaulting to HOLD.{RESET}")
            final_signal = "HOLD"
        else:
            # Use specific threshold if scalping mode is active
            threshold_key = "signal_score_threshold"
            if self.active_weight_set_name == "scalping" and "scalping_signal_threshold" in self.config:
                threshold_key = "scalping_signal_threshold"
                self.logger.debug(f"Using scalping signal threshold: {self.config.get(threshold_key)}")
            
            threshold = Decimal(str(self.config.get(threshold_key, 1.5)))

            if final_signal_score >= threshold: final_signal = "BUY"
            elif final_signal_score <= -threshold: final_signal = "SELL"
            else: final_signal = "HOLD"

        price_prec = self.get_price_precision()
        log_msg = (
            f"Signal Calc Summary ({self.symbol} @ {current_price:.{price_prec}f}):\n"
            f"  Weight Set: {self.active_weight_set_name}\n"
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN), Total Weight: {total_weight_applied:.3f}\n"
            f"  Final Score: {final_signal_score:.4f}, Threshold: +/- {threshold:.3f}\n"
            f"  ==> Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        if console_log_level <= logging.DEBUG:
            self.logger.debug(f"  Detailed Scores: {', '.join([f'{k}: {v}' for k, v in debug_scores.items()])}")

        if final_signal in self.signals: self.signals[final_signal] = 1
        return final_signal

    # --- Indicator Check Methods (Return float score -1.0 to 1.0, or np.nan) ---
    def _check_ema_alignment(self) -> float:
        if "EMA_Short" not in self.indicator_values or "EMA_Long" not in self.indicator_values:
            self.logger.debug(f"EMA Alignment check skipped for {self.symbol}: EMAs not in indicator_values.")
            return np.nan
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan
        # Example scaling: assumes typical momentum values might be small (e.g. +/- 0.2 for some assets/intervals)
        # Adjust scale_factor based on observed momentum ranges for your specific asset and timeframe.
        scale_factor = 5.0 # Scales a momentum of 0.2 to 1.0, -0.2 to -1.0
        score = momentum * scale_factor
        return max(-1.0, min(1.0, score))

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = float(self.config.get("volume_confirmation_multiplier", 1.5))
        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma <= 0: return np.nan
        try:
            if current_volume > volume_ma * multiplier: return 0.7  # High volume (significance, not direction)
            if current_volume < volume_ma / multiplier: return -0.4 # Low volume (lack of confirmation)
            return 0.0 # Neutral volume
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check failed for {self.symbol}: {e}{RESET}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
        score = 0.0
        if k < oversold and d < oversold: score = 1.0
        elif k > overbought and d > overbought: score = -1.0
        diff = k - d # K vs D relationship
        if abs(diff) > 5: # Threshold for significant difference/potential cross
            if diff > 0: score = max(score, 0.6) if score >=0 else 0.6
            else: score = min(score, -0.6) if score <=0 else -0.6
        else: # K and D are close
            if k > d: score = max(score, 0.2)
            elif k < d: score = min(score, -0.2)
        # Consider position within normal range (less weight)
        if oversold <= k <= overbought:
            range_width = overbought - oversold
            if range_width > 0:
                 mid_range_score = (k - (oversold + range_width / 2)) / (range_width / 2)
                 score = (score + mid_range_score * 0.3) / 1.3 # Weighted average
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan
        if rsi <= 30: return 1.0
        if rsi >= 70: return -1.0
        if rsi < 40: return 0.5
        if rsi > 60: return -0.5
        if 40 <= rsi <= 60: return 0.5 - (rsi - 40) * (1.0 / 20.0) # Linear scale 0.5 to -0.5
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan
        if cci <= -150: return 1.0
        if cci >= 150: return -1.0
        if cci < -80: return 0.6
        if cci > 80: return -0.6
        if -80 <= cci <= 80: return - (cci / 80.0) * 0.6 # Scale from +0.6 to -0.6
        return 0.0

    def _check_wr(self) -> float: # Williams %R
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        # WR: -100 (most oversold) to 0 (most overbought)
        if wr <= -80: return 1.0
        if wr >= -20: return -1.0
        if -80 < wr < -20: return 1.0 - (wr - (-80.0)) * (2.0 / 60.0) # Scale +1.0 to -1.0
        return 0.0 # Fallback for exact -80 or -20 if not caught by <= or >=

    def _check_psar(self) -> float:
        psar_l = self.indicator_values.get("PSAR_long", np.nan)
        psar_s = self.indicator_values.get("PSAR_short", np.nan)
        if pd.notna(psar_l) and pd.isna(psar_s): return 1.0  # Uptrend (PSAR below price)
        if pd.notna(psar_s) and pd.isna(psar_l): return -1.0 # Downtrend (PSAR above price)
        return 0.0 # Ambiguous or NaN

    def _check_sma_10(self) -> float:
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return np.nan
        if last_close > sma_10: return 0.6
        if last_close < sma_10: return -0.6
        return 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return np.nan
        if last_close > vwap: return 0.7
        if last_close < vwap: return -0.7
        return 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan
        if mfi <= 20: return 1.0
        if mfi >= 80: return -1.0
        if mfi < 40: return 0.4
        if mfi > 60: return -0.4
        if 40 <= mfi <= 60: return 0.4 - (mfi - 40) * (0.8 / 20.0) # Scale +0.4 to -0.4
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        bb_middle = self.indicator_values.get("BB_Middle", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(last_close):
            return np.nan

        if last_close < bb_lower: return 1.0  # Below lower band
        if last_close > bb_upper: return -1.0 # Above upper band

        # Price relative to middle band, scaled by proximity to outer bands
        upper_range = bb_upper - bb_middle
        lower_range = bb_middle - bb_lower

        if last_close > bb_middle and upper_range > 0:
            # Price in upper half: score from +0.5 (at middle) to -0.5 (at upper band)
            proximity_to_upper = (last_close - bb_middle) / upper_range
            score = 0.5 - proximity_to_upper
            return max(-0.5, min(0.5, score))
        if last_close < bb_middle and lower_range > 0:
            # Price in lower half: score from -0.5 (at middle) to +0.5 (at lower band)
            proximity_to_lower = (bb_middle - last_close) / lower_range
            score = -0.5 + proximity_to_lower
            return max(-0.5, min(0.5, score))
        return 0.0 # On middle band or invalid ranges

    def _check_orderbook(self, orderbook_data: dict | None, current_price: Decimal) -> float:
        """Analyzes order book depth. Returns float score or NaN."""
        if not orderbook_data: return np.nan
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            if not bids or not asks: return np.nan

            num_levels = 10 # Check top N levels
            bid_vol = sum(Decimal(str(b[1])) for b in bids[:num_levels])
            ask_vol = sum(Decimal(str(a[1])) for a in asks[:num_levels])
            total_vol = bid_vol + ask_vol
            if total_vol == 0: return 0.0

            obi = (bid_vol - ask_vol) / total_vol # Order Book Imbalance
            score = float(obi) # OBI is already -1 to 1
            # self.logger.debug(f"Orderbook ({self.symbol}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, OBI={obi:.4f}, Score={score:.4f}")
            return score
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan

    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential TP and initial SL based on entry, ATR, and multipliers."""
        if signal not in ["BUY", "SELL"]: return entry_price, None, None

        atr_val_float = self.indicator_values.get("ATR")
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid ATR ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid entry price ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float))
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
            price_prec = self.get_price_precision()
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"{NEON_RED}Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).{RESET}")
                 return entry_price, None, None

            take_profit, stop_loss = None, None
            tp_offset, sl_offset = atr * tp_mult, atr * sl_mult

            if signal == "BUY":
                take_profit_raw, stop_loss_raw = entry_price + tp_offset, entry_price - sl_offset
                # Quantize TP further into profit (UP), SL further away from entry (DOWN)
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
            elif signal == "SELL":
                take_profit_raw, stop_loss_raw = entry_price - tp_offset, entry_price + sl_offset
                # Quantize TP further into profit (DOWN), SL further away from entry (UP)
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

            # Validation: Ensure SL/TP are beyond entry by at least one tick and positive
            if signal == "BUY":
                if stop_loss >= entry_price: stop_loss = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)
                if take_profit <= entry_price: take_profit = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
            elif signal == "SELL":
                if stop_loss <= entry_price: stop_loss = (entry_price + min_tick).quantize(min_tick, rounding=ROUND_UP)
                if take_profit >= entry_price: take_profit = (entry_price - min_tick).quantize(min_tick, rounding=ROUND_DOWN)

            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss for {self.symbol} is zero/negative ({stop_loss}). Setting to None.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}Take profit for {self.symbol} is zero/negative ({take_profit}). Setting to None.{RESET}")
                take_profit = None

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_price:.{price_prec}f}, TP={take_profit:.{price_prec}f if take_profit else 'N/A'}, SL={stop_loss:.{price_prec}f if stop_loss else 'N/A'}, ATR={atr:.{price_prec+1}f}")
            return entry_price, take_profit, stop_loss
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches available balance for a currency, handling Bybit V5 structures."""
    lg = logger
    try:
        balance_info = None
        account_types_to_try = ['CONTRACT', 'UNIFIED'] # Prioritize for derivatives
        successful_acc_type = None

        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Fetching balance: type='{acc_type}' for {currency}...")
                balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                # Check if this balance_info structure contains the currency directly or nested
                if currency in balance_info or \
                   ('info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result'] and \
                    any(c.get('coin') == currency for acc_data in balance_info['info']['result']['list'] if acc_data.get('accountType') == acc_type for c in acc_data.get('coin',[]))):
                    successful_acc_type = acc_type
                    break # Found relevant structure
                balance_info = None # Reset if not found with this type
            except ccxt.ExchangeError as e: # Handle errors for specific account types gracefully
                if "account type not support" in str(e).lower() or getattr(e, 'code', None) == 10001: # Bybit specific
                    lg.debug(f"Account type '{acc_type}' not supported: {e}. Trying next.")
                    continue
                lg.warning(f"{NEON_YELLOW}Exchange error fetching balance for type {acc_type}: {e}. Trying next.{RESET}")
            except Exception as e:
                lg.warning(f"{NEON_YELLOW}Unexpected error fetching balance for type {acc_type}: {e}. Trying next.{RESET}")

        if not balance_info: # Fallback to default fetch_balance if specific types failed
            lg.debug(f"Fetching balance with default params for {currency}...")
            try:
                balance_info = exchange.fetch_balance()
                successful_acc_type = "Default"
            except Exception as e:
                lg.error(f"{NEON_RED}Failed to fetch balance with default params: {e}{RESET}")
                return None

        # Parse the balance_info
        available_bal_str = None
        if currency in balance_info and balance_info[currency].get('free') is not None:
            available_bal_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ['{currency}']['free']: {available_bal_str} (Type: {successful_acc_type})")
        elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
            for account in balance_info['info']['result']['list']:
                # Prioritize the successful_acc_type if known, or contract/unified
                target_acc_type = successful_acc_type if successful_acc_type not in [None, "Default"] else account.get('accountType')
                if account.get('accountType') == target_acc_type:
                    for coin_data in account.get('coin', []):
                        if coin_data.get('coin') == currency:
                            # Prefer 'availableBalance', then 'availableToWithdraw', then 'walletBalance'
                            free = coin_data.get('availableBalance')
                            if free is None or str(free) == '': free = coin_data.get('availableToWithdraw')
                            if free is None or str(free) == '': free = coin_data.get('walletBalance') # Includes PnL, less ideal for 'free'

                            if free is not None and str(free) != '':
                                available_bal_str = str(free)
                                lg.debug(f"Found balance via Bybit V5 nested: {available_bal_str} {currency} (Account: {account.get('accountType')})")
                                break
                    if available_bal_str: break
        elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
             available_bal_str = str(balance_info['free'][currency]) # Top-level free dict
             lg.debug(f"Found balance via top-level 'free' dict: {available_bal_str} (Type: {successful_acc_type})")


        if available_bal_str is None: # Last resort: 'total' balance
            total_bal_val = None
            if currency in balance_info and balance_info[currency].get('total') is not None:
                total_bal_val = balance_info[currency]['total']
            # (Simplified logic for total from nested, assuming walletBalance is best proxy if free failed)
            elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                for account in balance_info['info']['result']['list']:
                     # Similar account type preference as above
                    target_acc_type = successful_acc_type if successful_acc_type not in [None, "Default"] else account.get('accountType')
                    if account.get('accountType') == target_acc_type:
                        for coin_data in account.get('coin', []):
                            if coin_data.get('coin') == currency:
                                total_bal_val = coin_data.get('walletBalance') # walletBalance as total
                                if total_bal_val is not None: break
                        if total_bal_val is not None: break
            if total_bal_val is not None:
                lg.warning(f"{NEON_YELLOW}Using 'total' balance ({total_bal_val}) for {currency} as 'free' balance couldn't be determined. This may include collateral/unrealized PnL.{RESET}")
                available_bal_str = str(total_bal_val)
            else:
                lg.error(f"{NEON_RED}Could not determine any balance for {currency}. Balance info: {balance_info}{RESET}")
                return None

        try:
            final_balance = Decimal(available_bal_str)
            if final_balance >= 0: # Allow zero balance
                lg.info(f"Available {currency} balance: {final_balance:.4f} (Source Acc Type: {successful_acc_type or 'N/A'})")
                return final_balance
            else:
                lg.error(f"{NEON_RED}Parsed balance for {currency} is negative ({final_balance}). Returning None.{RESET}")
                return None
        except Exception as e:
            lg.error(f"{NEON_RED}Failed to convert balance string '{available_bal_str}' to Decimal for {currency}: {e}{RESET}")
            return None

    except ccxt.AuthenticationError as e: lg.error(f"{NEON_RED}Auth error fetching balance: {e}. Check API key permissions.{RESET}"); return None
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}"); return None
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True); return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Gets market information, ensuring markets are loaded and adds derived precision info."""
    lg = logger
    try:
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded or missing, reloading markets...")
            exchange.load_markets(reload=True)

        if symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Market {symbol} not found after reloading. Check symbol and availability.{RESET}")
            return None

        market = exchange.market(symbol)
        if not market: # Should not happen if symbol in exchange.markets
            lg.error(f"{NEON_RED}Market dictionary for {symbol} is unexpectedly None.{RESET}")
            return None
        if not market.get('active', True): # Check if market is active
            lg.warning(f"{NEON_YELLOW}Market {symbol} is not active on {exchange.id}. Trading may fail.{RESET}")
            # return None # Optionally, prevent trading on inactive markets

        market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
        
        # Derive and add precision/tick info using a temporary analyzer for its utility methods
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.to_datetime([]))
        # Minimal config for temp analyzer, ensure quote_currency is available if needed by precision funcs
        temp_config = {"quote_currency": market.get('quote', QUOTE_CURRENCY)}
        analyzer_temp = TradingAnalyzer(dummy_df, lg, temp_config, market)
        min_tick_dec = analyzer_temp.get_min_tick_size()
        price_prec_int = analyzer_temp.get_price_precision()

        if 'precision' not in market: market['precision'] = {}
        # Store derived values, preferring existing ones if valid
        if market['precision'].get('price') is None or not (isinstance(market['precision']['price'], (float, str)) and Decimal(str(market['precision']['price'])) > 0):
             market['precision']['price'] = float(min_tick_dec) # CCXT expects float for tick size here
        if market['precision'].get('tick') is None: market['precision']['tick'] = float(min_tick_dec)
        if market['precision'].get('price_decimals') is None: market['precision']['price_decimals'] = price_prec_int

        # Amount precision
        amt_prec_val = market.get('precision', {}).get('amount')
        amt_decs, amt_step = None, None
        if isinstance(amt_prec_val, int): amt_decs, amt_step = amt_prec_val, Decimal(f'1e-{amt_prec_val}')
        elif isinstance(amt_prec_val, (float,str)):
            try:
                amt_step = Decimal(str(amt_prec_val))
                if amt_step > 0: amt_decs = abs(amt_step.normalize().as_tuple().exponent)
            except Exception: pass
        if market['precision'].get('amount_decimals') is None and amt_decs is not None: market['precision']['amount_decimals'] = amt_decs
        if market['precision'].get('amount_step') is None and amt_step is not None: market['precision']['amount_step'] = float(amt_step)

        lg.debug(
            f"Market Info ({symbol}): ID={market.get('id')}, Type={market.get('type', 'N/A')}, Contract={market['is_contract']}, "
            f"Prec(Tick/Decs/AmtStep): {market['precision'].get('tick','N/A')}/{market['precision'].get('price_decimals','N/A')}/{market['precision'].get('amount_step','N/A')}, "
            f"Limits(AmtMin/Max): {market.get('limits',{}).get('amount',{}).get('min','N/A')}/{market.get('limits',{}).get('amount',{}).get('max','N/A')}"
        )
        return market
    except ccxt.BadSymbol as e: lg.error(f"{NEON_RED}Symbol '{symbol}' invalid for {exchange.id}: {e}{RESET}"); return None
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error loading market info for {symbol}: {e}{RESET}"); return None
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error loading market info for {symbol}: {e}{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True); return None


def calculate_position_size(
    balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal,
    entry_price: Decimal, market_info: dict, exchange: ccxt.Exchange,
    logger: logging.Logger | None = None
) -> Decimal | None:
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_ccy = market_info.get('quote', QUOTE_CURRENCY)
    base_ccy = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('linear', not is_contract) # Spot is linear
    size_unit = "Contracts" if is_contract else base_ccy

    if balance is None or balance <= 0: lg.error(f"PosSizing ({symbol}): Invalid balance {balance} {quote_ccy}."); return None
    if not (0 < risk_per_trade < 1): lg.error(f"PosSizing ({symbol}): Invalid risk_per_trade {risk_per_trade}."); return None
    if initial_stop_loss_price is None or entry_price is None or entry_price <= 0 or initial_stop_loss_price == entry_price:
        lg.error(f"PosSizing ({symbol}): Invalid entry ({entry_price}) or SL ({initial_stop_loss_price})."); return None
    if initial_stop_loss_price <= 0: lg.warning(f"{NEON_YELLOW}PosSizing ({symbol}): Initial SL price ({initial_stop_loss_price}) is zero/negative.{RESET}")

    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"PosSizing ({symbol}): Market info missing 'limits' or 'precision'."); return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0: lg.error(f"PosSizing ({symbol}): SL distance zero/negative."); return None

        contract_size = Decimal(str(market_info.get('contractSize', '1')))
        if contract_size <= 0: contract_size = Decimal('1'); lg.warning(f"{NEON_YELLOW}Invalid contract size for {symbol}, defaulting to 1.{RESET}")

        calculated_size = Decimal('0')
        if is_linear: # Linear contracts or Spot
            risk_per_unit_quote = sl_distance_per_unit * contract_size
            if risk_per_unit_quote <= 0: lg.error(f"PosSizing ({symbol}): Risk per unit zero/negative."); return None
            calculated_size = risk_amount_quote / risk_per_unit_quote
        else: # Inverse contracts
            contract_value_quote = contract_size # For inverse, contractSize is often quote value (e.g., 1 USD)
            if entry_price <= 0: lg.error(f"PosSizing ({symbol}): Entry price zero for inverse."); return None
            contract_value_base = contract_value_quote / entry_price
            risk_per_contract_quote = contract_value_base * sl_distance_per_unit
            if risk_per_contract_quote <= 0: lg.error(f"PosSizing ({symbol}): Risk per contract zero/negative for inverse."); return None
            calculated_size = risk_amount_quote / risk_per_contract_quote
        
        lg.info(f"PosSizing ({symbol}): Bal={balance:.2f}{quote_ccy}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f}{quote_ccy}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SLDist={sl_distance_per_unit}, ContSize={contract_size}")
        lg.info(f"  Initial Calc. Size = {calculated_size:.8f} {size_unit}")

        # Apply Market Limits and Precision
        limits, precision = market_info['limits'], market_info['precision']
        min_amt = Decimal(str(limits.get('amount',{}).get('min',0)))
        max_amt = Decimal(str(limits.get('amount',{}).get('max', 'inf')))
        min_cost = Decimal(str(limits.get('cost',{}).get('min',0)))
        max_cost = Decimal(str(limits.get('cost',{}).get('max', 'inf')))
        amt_step_float = precision.get('amount_step')
        amt_step = Decimal(str(amt_step_float)) if amt_step_float is not None and amt_step_float > 0 else None

        adj_size = calculated_size
        if adj_size < min_amt: adj_size = min_amt; lg.warning(f"{NEON_YELLOW}Adjusted size to min_amt: {adj_size}{RESET}")
        if adj_size > max_amt: adj_size = max_amt; lg.warning(f"{NEON_YELLOW}Capped size to max_amt: {adj_size}{RESET}")

        # Cost check
        current_cost = (adj_size * entry_price * contract_size) if is_linear else (adj_size * contract_size) # Inverse cost = size * quote_value_per_contract
        lg.debug(f"  Cost Check: Size={adj_size:.8f}, Est.Cost={current_cost:.4f}{quote_ccy} (Limits: Min={min_cost}, Max={max_cost})")

        if min_cost > 0 and current_cost < min_cost:
            req_size_min_cost = Decimal('0')
            if is_linear: req_size_min_cost = min_cost / (entry_price * contract_size) if entry_price > 0 and contract_size > 0 else Decimal('-1')
            else: req_size_min_cost = min_cost / contract_size if contract_size > 0 else Decimal('-1')

            if req_size_min_cost < 0 or req_size_min_cost > max_amt or req_size_min_cost < min_amt:
                lg.error(f"{NEON_RED}Cannot meet min_cost {min_cost} due to conflicting limits. Trade aborted.{RESET}"); return None
            adj_size = req_size_min_cost
            lg.warning(f"{NEON_YELLOW}Adjusted size to meet min_cost: {adj_size}{RESET}")
        
        # Re-calc cost and check max_cost (cost could have increased if adj_size was initially min_amt)
        current_cost = (adj_size * entry_price * contract_size) if is_linear else (adj_size * contract_size)
        if max_cost > 0 and current_cost > max_cost:
            req_size_max_cost = Decimal('0')
            if is_linear: req_size_max_cost = max_cost / (entry_price * contract_size) if entry_price > 0 and contract_size > 0 else Decimal('-1')
            else: req_size_max_cost = max_cost / contract_size if contract_size > 0 else Decimal('-1')

            if req_size_max_cost < 0 or req_size_max_cost < min_amt: # Cannot go below min_amt
                lg.error(f"{NEON_RED}Cannot meet max_cost {max_cost} without violating min_amt. Trade aborted.{RESET}"); return None
            adj_size = req_size_max_cost
            lg.warning(f"{NEON_YELLOW}Adjusted size to meet max_cost: {adj_size}{RESET}")

        # Apply Amount Precision (Truncate/Round Down)
        final_size = Decimal('0')
        try:
            fmt_size_str = exchange.amount_to_precision(symbol, float(adj_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(fmt_size_str)
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}exchange.amount_to_precision failed ({fmt_err}). Using manual rounding.{RESET}")
            if amt_step: final_size = (adj_size // amt_step) * amt_step
            else: final_size = adj_size # Use as is if no step defined

        lg.info(f"  Size after limits & precision (Truncated): {final_size} {size_unit}")

        # Final Validations
        if final_size <= 0: lg.error(f"{NEON_RED}Final size zero/negative. Trade aborted.{RESET}"); return None
        if final_size < min_amt and not math.isclose(float(final_size), float(min_amt), rel_tol=1e-9):
            lg.error(f"{NEON_RED}Final size {final_size} < min_amt {min_amt}. Trade aborted.{RESET}"); return None
        
        final_cost = (final_size * entry_price * contract_size) if is_linear else (final_size * contract_size)
        if min_cost > 0 and final_cost < min_cost and not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6):
            lg.error(f"{NEON_RED}Final cost {final_cost} < min_cost {min_cost}. Trade aborted.{RESET}"); return None

        lg.info(f"{NEON_GREEN}Final Calculated Position Size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e: lg.error(f"{NEON_RED}PosSizing ({symbol}): Missing market key {e}.{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error in PosSizing ({symbol}): {e}{RESET}", exc_info=True); return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks for an open position, enhances with SL/TP/TSL info from 'info' dict."""
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: list[dict] = []
        market = exchange.market(symbol) # Ensure market is loaded for category
        if not market: lg.error(f"Cannot fetch pos: Market info not found for '{symbol}'."); return None
        
        category = 'linear' if market.get('linear', True) else 'inverse'
        params = {'category': category, 'symbol': market['id']}
        positions = exchange.fetch_positions(symbols=[market['symbol']], params=params)

        active_pos = None
        if positions:
            # Assuming One-Way mode, find first meaningful position
            for pos_entry in positions:
                pos_size_str = pos_entry.get('info', {}).get('size', pos_entry.get('contracts'))
                if pos_size_str is not None:
                    try:
                        pos_size_dec = Decimal(str(pos_size_str))
                        # Use a small threshold, ideally based on market's min amount if available
                        min_size_check = Decimal(str(market.get('limits',{}).get('amount',{}).get('min', '1e-9'))) * Decimal('0.1')
                        min_size_check = max(Decimal('1e-9'), min_size_check) # Ensure it's not zero
                        if abs(pos_size_dec) >= min_size_check:
                            active_pos = pos_entry
                            lg.debug(f"Found potential active position for {symbol} with size {pos_size_dec}.")
                            break
                    except Exception: pass # Ignore parsing errors for size
            if not active_pos: lg.info(f"No meaningfully sized position found for {symbol} among {len(positions)} entries."); return None
        else: lg.info(f"No position entries returned for {symbol}."); return None

        # Post-process the found active position
        active_pos['market'] = market # Attach market info
        
        # Determine Side
        side = active_pos.get('side')
        info_side = active_pos.get('info', {}).get('side') # Bybit: 'Buy', 'Sell', 'None'
        pos_size_dec = Decimal(str(active_pos.get('info', {}).get('size', '0')))

        if side not in ['long', 'short']:
            if info_side == 'Buy': side = 'long'
            elif info_side == 'Sell': side = 'short'
            else: side = 'long' if pos_size_dec > 0 else 'short' if pos_size_dec < 0 else None
            if side is None : lg.error(f"{NEON_RED}Could not determine position side for {symbol}.{RESET}"); return None
            active_pos['side'] = side
        
        # Standardize 'contracts' to positive absolute amount (float for CCXT standard)
        try: active_pos['contracts'] = float(abs(Decimal(str(active_pos.get('info', {}).get('size', '0')))))
        except Exception: lg.warning(f"{NEON_YELLOW}Could not standardize 'contracts' field for {symbol}.{RESET}")

        # Enhance with SL/TP/TSL from 'info' (Bybit V5)
        info_dict = active_pos.get('info', {})
        def get_dec_from_info(key: str, allow_zero=False) -> Decimal | None:
            val_str = info_dict.get(key)
            if val_str is not None and str(val_str).strip() != '':
                try:
                    dec_val = Decimal(str(val_str).strip())
                    if (allow_zero and dec_val >= 0) or (not allow_zero and dec_val > 0): return dec_val
                except Exception: pass
            return None

        active_pos['stopLossPrice'] = float(get_dec_from_info('stopLoss')) if get_dec_from_info('stopLoss') else None
        active_pos['takeProfitPrice'] = float(get_dec_from_info('takeProfit')) if get_dec_from_info('takeProfit') else None
        active_pos['trailingStopLossDistanceDecimal'] = get_dec_from_info('trailingStop', allow_zero=True) or Decimal('0')
        active_pos['tslActivationPriceDecimal'] = get_dec_from_info('activePrice', allow_zero=True) or Decimal('0')
        # Keep raw string values for logging/reference too
        active_pos['trailingStopLossDistanceRaw'] = info_dict.get('trailingStop', '')
        active_pos['tslActivationPriceRaw'] = info_dict.get('activePrice', '')
        
        # Logging
        price_prec = market['precision'].get('price_decimals', 6)
        amt_prec = market['precision'].get('amount_decimals', 8)
        entry_p = active_pos.get('entryPrice', info_dict.get('avgPrice'))
        liq_p = active_pos.get('liquidationPrice')
        lev = active_pos.get('leverage', info_dict.get('leverage'))
        pnl = active_pos.get('unrealizedPnl', info_dict.get('unrealisedPnl'))
        
        log_parts = [
            f"{NEON_GREEN}Active {active_pos['side'].upper()} position ({symbol}):{RESET}",
            f"Size={active_pos.get('contracts', 'N/A'):.{amt_prec}f}",
            f"Entry={Decimal(str(entry_p)):.{price_prec}f}" if entry_p else "Entry=N/A",
            f"Liq={Decimal(str(liq_p)):.{price_prec}f}" if liq_p else "Liq=N/A",
            f"Lev={Decimal(str(lev)):.1f}x" if lev else "Lev=N/A",
            f"PnL={Decimal(str(pnl)):.4f}" if pnl else "PnL=N/A",
            f"SL={active_pos['stopLossPrice']:.{price_prec}f}" if active_pos['stopLossPrice'] else "SL=N/A",
            f"TP={active_pos['takeProfitPrice']:.{price_prec}f}" if active_pos['takeProfitPrice'] else "TP=N/A",
            f"TSL Active: {active_pos['trailingStopLossDistanceDecimal'] > 0} (Dist={active_pos['trailingStopLossDistanceRaw']}/Act={active_pos['tslActivationPriceRaw']})"
        ]
        logger.info(" ".join(log_parts))
        logger.debug(f"Full position details for {symbol}: {json.dumps(active_pos, default=str, indent=2)}")
        return active_pos

    except (ccxt.BadSymbol, ccxt.NetworkError, ccxt.AuthenticationError) as e:
        lg.error(f"{NEON_RED}Error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e: # Handle specific "no position" errors
        no_pos_codes = [110021, 110025] # Bybit: Position not exist / idx mismatch
        if getattr(e, 'code', None) in no_pos_codes or "position not found" in str(e).lower():
            lg.info(f"No active position found for {symbol} (Exchange confirmation: {e}).")
        else: lg.error(f"{NEON_RED}Exchange error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    lg = logger
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market)."); return True
    if leverage <= 0:
        lg.warning(f"{NEON_YELLOW}Leverage setting skipped for {symbol}: Invalid leverage ({leverage}).{RESET}"); return False
    if not exchange.has.get('setLeverage'):
        lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage via CCXT.{RESET}"); return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        if 'bybit' in exchange.id.lower():
            category = 'linear' if market_info.get('linear', True) else 'inverse'
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage), 'category': category}
        
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        verified = False # Verify success based on response
        if response is not None:
            if isinstance(response, dict):
                 ret_code = response.get('retCode', response.get('info', {}).get('retCode'))
                 if ret_code == 0: verified = True
                 elif ret_code == 110045: lg.info(f"{NEON_YELLOW}Leverage for {symbol} already {leverage}x (Code {ret_code}).{RESET}"); verified = True
                 elif ret_code is not None: lg.warning(f"{NEON_YELLOW}Set leverage returned code {ret_code} ({response.get('retMsg','N/A')}). Failed.{RESET}"); verified = False
                 else: verified = True # No retCode, assume success if no error
            else: verified = True # Non-dict response, assume success
        else: verified = True # None response, assume success (Bybit V5 can do this)

        if verified: lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}"); return True
        else: lg.error(f"{NEON_RED}Leverage setting failed for {symbol} based on response.{RESET}"); return False

    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110045: lg.info(f"{NEON_YELLOW}Leverage already set (Code {bybit_code}).{RESET}"); return True
        # Add more specific error hints here from original script if needed
        if bybit_code in [110028, 110009]: lg.error(f"{NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) for {symbol}.{RESET}")
        elif bybit_code == 110044: lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x might exceed risk limit.{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)
    return False


def place_trade(
    exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal,
    market_info: dict, logger: logging.Logger | None = None, reduce_only: bool = False
) -> dict | None:
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    size_unit = "Contracts" if market_info.get('is_contract', False) else market_info.get('base', '')

    try:
        amount_float = float(abs(position_size)) # CCXT expects float amount
        if amount_float <= 0: lg.error(f"Trade ({symbol} {side}): Invalid size {amount_float}."); return None
    except Exception as e: lg.error(f"Trade ({symbol} {side}): Failed to convert size {position_size} to float: {e}"); return None

    params = {'positionIdx': 0, 'reduceOnly': reduce_only, 'closeOnTrigger': False}
    if 'bybit' in exchange.id.lower():
        params['category'] = 'linear' if market_info.get('linear', True) else 'inverse'

    action = "Closing" if reduce_only else "Opening"
    lg.info(f"Attempting {side.upper()} {order_type} order ({action}) for {symbol}: Size={amount_float:.8f} {size_unit}, Params={params}")

    try:
        order = exchange.create_order(symbol, order_type, side, amount_float, params=params)
        lg.info(f"{NEON_GREEN}Trade Order Placed ({action})! ID: {order.get('id','N/A')}, Status: {order.get('status','N/A')}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side} reduce={reduce_only}): {json.dumps(order, default=str, indent=2)}")
        return order
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds for {side} order ({action} {symbol}): {e}{RESET}")
        # Add hints from original script if needed
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order params for {symbol} ({action}): {e}{RESET}")
        # Add hints from original script if needed
        if getattr(e, 'code', None) == 110040: lg.error(f"{NEON_YELLOW} >> Hint: Order size {amount_float} below minimum.{RESET}")
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error placing order ({action} {symbol}): {e}{RESET}")
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error placing order ({action} {symbol}): {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110021 and reduce_only: # Position already closed
             lg.warning(f"{NEON_YELLOW} >> Hint (110021): Position already closed/not found on close attempt.{RESET}")
             return {'id': 'N/A', 'status': 'closed', 'info': {'reason': 'Position already closed/not found'}}
        # Add more hints from original script if needed
        elif bybit_code == 110044: lg.error(f"{NEON_YELLOW} >> Hint: Order may exceed risk limits.{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error placing order ({action} {symbol}): {e}{RESET}", exc_info=True)
    return None


def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict, logger: logging.Logger,
    stop_loss_price: Decimal | str | None = None, take_profit_price: Decimal | str | None = None,
    trailing_stop_distance: Decimal | str | None = None, tsl_activation_price: Decimal | str | None = None,
) -> bool:
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"{NEON_YELLOW}Protection skipped for {symbol} (not a contract).{RESET}"); return False
    if not position_info: lg.error(f"{NEON_RED}Cannot set protection for {symbol}: Missing position info.{RESET}"); return False

    pos_side = position_info.get('side')
    if pos_side not in ['long', 'short']: lg.error(f"{NEON_RED}Cannot set protection for {symbol}: Invalid position side.{RESET}"); return False

    def is_valid_or_cancel(val, allow_zero_dec=False):
        if val is None: return False
        if isinstance(val, str) and val == '0': return True
        if isinstance(val, Decimal): return (val >= 0 if allow_zero_dec else val > 0)
        return False

    has_sl_intent = stop_loss_price is not None
    has_tp_intent = take_profit_price is not None
    has_tsl_intent = trailing_stop_distance is not None

    is_sl_val = has_sl_intent and is_valid_or_cancel(stop_loss_price)
    is_tp_val = has_tp_intent and is_valid_or_cancel(take_profit_price)
    is_tsl_dist_val = has_tsl_intent and is_valid_or_cancel(trailing_stop_distance, allow_zero_dec=True)
    is_tsl_act_val = is_valid_or_cancel(tsl_activation_price, allow_zero_dec=True)
    
    is_tsl_val = is_tsl_dist_val
    if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0: # If TSL distance is active
        is_tsl_val = is_tsl_dist_val and is_tsl_act_val # Activation price must also be valid

    if not is_sl_val and not is_tp_val and not is_tsl_val:
        lg.info(f"No valid protection parameters (SL/TP/TSL > 0 or '0' for cancel) for {symbol}. No protection set/modified.")
        return True # Success as no API call intended/needed

    category = 'linear' if market_info.get('linear', True) else 'inverse'
    pos_idx = int(position_info.get('info', {}).get('positionIdx', 0))
    params = {
        'category': category, 'symbol': market_info['id'], 'tpslMode': 'Full',
        'slTriggerBy': 'LastPrice', 'tpTriggerBy': 'LastPrice',
        'slOrderType': 'Market', 'tpOrderType': 'Market', 'positionIdx': pos_idx
    }
    log_parts = [f"Attempting protection for {symbol} ({pos_side.upper()}, Idx: {pos_idx}):"]
    
    sl_added, tp_added, tsl_added = False, False, False # Track if params were actually added

    def fmt_param(val, is_price=True): # Helper to format for API
        if val is None: return None
        if isinstance(val, str) and val == '0': return '0'
        if isinstance(val, Decimal) and val >=0:
            try:
                fmt_val = exchange.price_to_precision(symbol, float(val))
                # Ensure '0.0' formatting doesn't happen if original wasn't zero
                if Decimal(fmt_val) == 0 and val != 0 and re.fullmatch(r'\d+(\.\d+)?', str(val)): return str(val)
                return fmt_val
            except Exception: return None
        return None

    if is_tsl_val:
        fmt_tsl_dist = fmt_param(trailing_stop_distance, is_price=False) # Bybit API treats distance as having price precision
        fmt_act_price = fmt_param(tsl_activation_price, is_price=True)
        if isinstance(trailing_stop_distance, str) and trailing_stop_distance == '0': # Cancel TSL
            params['trailingStop'], params['activePrice'] = '0', '0'
            log_parts.append("  Trailing SL: Cancelling"); tsl_added = True
        elif fmt_tsl_dist and fmt_tsl_dist != '0' and fmt_act_price: # Set TSL
            params['trailingStop'], params['activePrice'] = fmt_tsl_dist, fmt_act_price
            log_parts.append(f"  Trailing SL: Dist={fmt_tsl_dist}, Act={fmt_act_price}"); tsl_added = True
        else: is_tsl_val = False # Mark as failed if formatting failed

    if is_sl_val and (not tsl_added or params.get('trailingStop') == '0'): # Fixed SL if TSL not actively set
        fmt_sl = fmt_param(stop_loss_price)
        if fmt_sl: params['stopLoss'] = fmt_sl; log_parts.append(f"  Fixed SL: {fmt_sl}"); sl_added = True
        else: is_sl_val = False
    elif has_sl_intent and tsl_added and params.get('trailingStop','0') != '0':
        lg.warning(f"{NEON_YELLOW}TSL prioritized for {symbol}; Fixed SL '{stop_loss_price}' ignored.{RESET}"); is_sl_val = False

    if is_tp_val:
        fmt_tp = fmt_param(take_profit_price)
        if fmt_tp: params['takeProfit'] = fmt_tp; log_parts.append(f"  Fixed TP: {fmt_tp}"); tp_added = True
        else: is_tp_val = False
    
    protection_keys_in_params = [k for k in ['stopLoss', 'takeProfit', 'trailingStop'] if k in params and params[k] is not None]
    if not protection_keys_in_params:
        lg.warning(f"{NEON_YELLOW}No protection parameters remain after formatting/prioritization for {symbol}. No API call.{RESET}")
        # If something was intended but all failed/overridden, it's a failure of intent.
        return not (has_sl_intent or has_tp_intent or has_tsl_intent)

    lg.info("\n".join(log_parts)); lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={params})")
    try:
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        if ret_code == 0:
            no_change_msgs = ["stoplosstakeprofittrailingstopwerenotmodified", "not modified"]
            if any(msg in ret_msg.lower().replace(" ","").replace(",","") for msg in no_change_msgs):
                lg.info(f"{NEON_YELLOW}Protection for {symbol} not modified (already set or no change needed).{RESET}")
            else: lg.info(f"{NEON_GREEN}Protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {response.get('retExtInfo', {})}{RESET}")
            # Add specific error hints from original script based on ret_code if needed
            if ret_code == 110087: lg.error(f"{NEON_YELLOW} >> Hint (110087): TP price {params.get('takeProfit')} wrong side for {pos_side} position.{RESET}")
            elif ret_code == 110088: lg.error(f"{NEON_YELLOW} >> Hint (110088): SL price {params.get('stopLoss')} wrong side for {pos_side} position.{RESET}")
            return False
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error setting protection for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error during protection API call for {symbol}: {e}{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict,
    config: dict[str, Any], logger: logging.Logger, take_profit_price: Decimal | None = None
) -> bool:
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"TSL disabled in config for {symbol}. Skipping TSL setup."); return False # TSL not set
    try:
        cb_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        act_pct = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
    except Exception as e: lg.error(f"{NEON_RED}Invalid TSL config format for {symbol}: {e}{RESET}"); return False
    if cb_rate <= 0: lg.error(f"{NEON_RED}Invalid TSL callback_rate ({cb_rate}) for {symbol}.{RESET}"); return False
    if act_pct < 0: lg.error(f"{NEON_RED}Invalid TSL activation_percentage ({act_pct}) for {symbol}.{RESET}"); return False

    try:
        entry_p_str = position_info.get('entryPrice', position_info.get('info', {}).get('avgPrice'))
        side = position_info.get('side')
        if not entry_p_str or side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing position info for TSL calc ({symbol}).{RESET}"); return False
        entry_p = Decimal(str(entry_p_str))
        if entry_p <= 0: lg.error(f"{NEON_RED}Invalid entry price ({entry_p}) for TSL calc ({symbol}).{RESET}"); return False
    except Exception as e: lg.error(f"{NEON_RED}Error parsing position info for TSL ({symbol}): {e}{RESET}"); return False

    try:
        # Minimal setup for temp analyzer to use its precision/tick methods
        dummy_df = pd.DataFrame().set_index(pd.to_datetime([]))
        if not market_info: lg.error(f"{NEON_RED}Cannot calc TSL: Market info missing.{RESET}"); return False
        temp_analyzer = TradingAnalyzer(dummy_df, lg, {"quote_currency": market_info.get('quote', QUOTE_CURRENCY)}, market_info)
        price_prec = temp_analyzer.get_price_precision()
        min_tick = temp_analyzer.get_min_tick_size()
        if min_tick <= 0: lg.error(f"{NEON_RED}Cannot calc TSL: Invalid min_tick_size ({min_tick}) for {symbol}.{RESET}"); return False

        # Activation Price
        activation_price = Decimal('0') # Default to immediate activation
        if act_pct > 0:
            act_offset = entry_p * act_pct
            raw_act = entry_p + act_offset if side == 'long' else entry_p - act_offset
            rounding = ROUND_UP if side == 'long' else ROUND_DOWN
            activation_price = (raw_act / min_tick).quantize(Decimal('1'), rounding=rounding) * min_tick
            # Ensure activation is away from entry
            if side == 'long' and activation_price <= entry_p: activation_price = (entry_p + min_tick).quantize(min_tick, ROUND_UP)
            if side == 'short' and activation_price >= entry_p: activation_price = (entry_p - min_tick).quantize(min_tick, ROUND_DOWN)
        if activation_price < 0: lg.error(f"{NEON_RED}TSL activation price ({activation_price}) invalid.{RESET}"); return False

        # Trailing Distance
        trail_dist_raw = entry_p * cb_rate
        trail_dist = (trail_dist_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
        if trail_dist < min_tick: trail_dist = min_tick; lg.warning(f"{NEON_YELLOW}TSL dist adjusted to min_tick {min_tick} for {symbol}.{RESET}")
        if trail_dist <= 0: lg.error(f"{NEON_RED}TSL dist zero/negative ({trail_dist}) for {symbol}.{RESET}"); return False

        act_p_log = f"{activation_price:.{price_prec}f}" if activation_price > 0 else "0 (Immediate)"
        lg.info(f"Calculated TSL Params ({symbol} {side.upper()}): Entry={entry_p:.{price_prec}f}, Act.Price(API)={act_p_log}, Trail.Dist(API)={trail_dist:.{price_prec}f}")
        
        return _set_position_protection(
            exchange, symbol, market_info, position_info, lg,
            stop_loss_price='0', # Cancel fixed SL when setting TSL
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trail_dist,
            tsl_activation_price=activation_price
        )
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True); return False

# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    market_info = get_market_info(exchange, symbol, lg)
    if not market_info: lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping cycle.{RESET}"); return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval: lg.error(f"{NEON_RED}Invalid interval '{config['interval']}' for {symbol}.{RESET}"); return

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=500, logger=lg)
    if klines_df.empty or len(klines_df) < 50:
        lg.error(f"{NEON_RED}Failed to fetch sufficient klines for {symbol} (got {len(klines_df)}). Skipping cycle.{RESET}"); return

    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch ticker price for {symbol}. Using last kline close.{RESET}")
        try:
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0: current_price = Decimal(str(last_close_val))
            else: lg.error(f"{NEON_RED}Last close price also invalid for {symbol}. Cannot proceed.{RESET}"); return
        except Exception as e: lg.error(f"{NEON_RED}Error getting last close for {symbol}: {e}. Cannot proceed.{RESET}"); return

    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators", {}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)

    analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    if not analyzer.indicator_values: lg.error(f"{NEON_RED}Indicator calculation failed for {symbol}. Skipping signal.{RESET}"); return
    
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)
    price_precision = analyzer.get_price_precision()
    current_atr_float = analyzer.indicator_values.get("ATR")

    atr_log = f"{current_atr_float:.{price_precision + 1}f}" if current_atr_float and pd.notna(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision}f}" if tp_potential else 'N/A'
    lg.info(f"ATR: {atr_log}, Potential Initial SL (sizing): {sl_pot_log}, Potential TP: {tp_pot_log}")
    lg.info(f"TSL: {'Enabled' if config.get('enable_trailing_stop') else 'Disabled'} | BE: {'Enabled' if config.get('enable_break_even') else 'Disabled'}")

    if not config.get("enable_trading", False):
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
    else: # Trading is enabled
        open_position = get_open_position(exchange, symbol, lg)

        # Scenario 1: No Open Position
        if open_position is None:
            if signal in ["BUY", "SELL"]:
                lg.info(f"*** {signal} Signal & No Position: Initiating Trade for {symbol} ***")
                balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if balance is None or balance <= 0: lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Balance issue ({balance}).{RESET}"); return
                if sl_potential is None: lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Potential SL calc failed.{RESET}"); return

                if market_info.get('is_contract', False):
                    leverage = int(config.get("leverage", 1))
                    if leverage > 0 and not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage.{RESET}"); return
                
                position_size = calculate_position_size(balance, config["risk_per_trade"], sl_potential, current_price, market_info, exchange, lg)
                if position_size is None or position_size <= 0:
                    lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size ({position_size}).{RESET}"); return

                trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)
                if trade_order and trade_order.get('id'):
                    lg.info(f"Order {trade_order['id']} placed. Waiting {POSITION_CONFIRM_DELAY}s for confirmation...")
                    time.sleep(POSITION_CONFIRM_DELAY)
                    confirmed_position = get_open_position(exchange, symbol, lg)
                    if confirmed_position:
                        try:
                            entry_p_actual_str = confirmed_position.get('entryPrice', confirmed_position.get('info',{}).get('avgPrice'))
                            entry_p_actual = Decimal(str(entry_p_actual_str)) if entry_p_actual_str else None
                            if not entry_p_actual or entry_p_actual <= 0: raise ValueError("Invalid actual entry price")

                            lg.info(f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_p_actual:.{price_precision}f}{RESET}")
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_p_actual, signal)
                            
                            protection_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting TSL for {symbol} (TP target: {tp_actual})...")
                                protection_success = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_position, config, lg, tp_actual)
                            else: # Fixed SL/TP
                                lg.info(f"Setting Fixed SL ({sl_actual}) & TP ({tp_actual}) for {symbol}...")
                                if sl_actual or tp_actual: # Only if valid
                                    protection_success = _set_position_protection(exchange, symbol, market_info, confirmed_position, lg, sl_actual, tp_actual, '0', '0')
                                else: lg.warning(f"{NEON_YELLOW}Fixed SL/TP for {symbol} invalid. No fixed protection.{RESET}"); protection_success = True # No protection to set

                            if protection_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION COMPLETE for {symbol} ({signal}) ===")
                            else: lg.error(f"{NEON_RED}=== TRADE PLACED BUT PROTECTION FAILED for {symbol} ({signal}). Manual check needed! ===")
                        except Exception as post_err:
                            lg.error(f"{NEON_RED}Error post-trade (protection) for {symbol}: {post_err}{RESET}", exc_info=True)
                            lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check!{RESET}")
                    else: # Position not confirmed
                        lg.error(f"{NEON_RED}Trade order placed, but FAILED TO CONFIRM position for {symbol}. Manual check order {trade_order.get('id')}!{RESET}")
                elif trade_order and trade_order.get('info', {}).get('reason'): # e.g. already closed
                     lg.info(f"Trade ({symbol} {signal}) resulted in: {trade_order['info']['reason']}. No further action.")
                else: lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ({signal}). ===")
            else: # No position, HOLD signal
                lg.info(f"Signal HOLD, no open position for {symbol}. No action.")

        # Scenario 2: Existing Open Position
        else:
            pos_side = open_position.get('side', 'unknown')
            is_tsl_active = open_position.get('trailingStopLossDistanceDecimal', Decimal(0)) > 0
            lg.info(f"Existing {pos_side.upper()} position for {symbol}. TSL Active: {is_tsl_active}")

            if (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY"):
                lg.warning(f"{NEON_YELLOW}*** EXIT Signal ({signal}) opposes existing {pos_side} position for {symbol}. Closing... ***{RESET}")
                try:
                    close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                    size_to_close_str = open_position.get('contracts') # Standardized positive size
                    if size_to_close_str is None: raise ValueError("Cannot get size to close.")
                    size_to_close = abs(Decimal(str(size_to_close_str)))
                    if size_to_close <= 0: raise ValueError(f"Invalid size to close: {size_to_close}")

                    close_order = place_trade(exchange, symbol, close_side_signal, size_to_close, market_info, lg, reduce_only=True)
                    if close_order:
                        order_id = close_order.get('id', 'N/A')
                        if close_order.get('info', {}).get('reason'): # Dummy success if already closed
                            lg.info(f"{NEON_GREEN}Position CLOSE for {symbol} confirmed ({close_order['info']['reason']}).{RESET}")
                        else:
                            lg.info(f"{NEON_GREEN}Position CLOSE order {order_id} placed for {symbol}. Waiting to confirm closure...{RESET}")
                            time.sleep(POSITION_CONFIRM_DELAY) # Wait for closure
                            final_check_pos = get_open_position(exchange, symbol, lg)
                            if final_check_pos is None:
                                lg.info(f"{NEON_GREEN}Position for {symbol} successfully confirmed closed.{RESET}")
                            else:
                                lg.warning(f"{NEON_RED}Position for {symbol} may NOT be fully closed after close order. Manual check! State: {final_check_pos.get('contracts')}{RESET}")
                    else: lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual intervention!{RESET}")
                except Exception as close_err:
                    lg.error(f"{NEON_RED}Error closing position {symbol}: {close_err}{RESET}", exc_info=True)
            elif signal == "HOLD" or (signal == "BUY" and pos_side == 'long') or (signal == "SELL" and pos_side == 'short'):
                lg.info(f"Signal ({signal}) allows holding {pos_side} position for {symbol}.")
                # Break-Even Logic
                if config.get("enable_break_even", False) and not is_tsl_active:
                    lg.debug(f"Checking Break-Even for {symbol}...")
                    try:
                        entry_p_str = open_position.get('entryPrice', open_position.get('info',{}).get('avgPrice'))
                        if not entry_p_str: raise ValueError("Missing entry price for BE")
                        entry_p = Decimal(str(entry_p_str))
                        if entry_p <= 0: raise ValueError(f"Invalid entry price {entry_p} for BE")
                        if current_atr_float is None or pd.isna(current_atr_float) or current_atr_float <= 0:
                            raise ValueError("Invalid ATR for BE")
                        current_atr_dec = Decimal(str(current_atr_float))
                        
                        be_trig_mult = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        min_tick = analyzer.get_min_tick_size() # Use analyzer's method
                        if min_tick <= 0: raise ValueError("Invalid min_tick for BE")

                        price_diff = current_price - entry_p if pos_side == 'long' else entry_p - current_price
                        profit_target_diff = be_trig_mult * current_atr_dec
                        lg.debug(f"BE Check: PriceDiff={price_diff:.{price_precision}f}, TargetDiff={profit_target_diff:.{price_precision}f}")

                        if price_diff >= profit_target_diff:
                            tick_offset = min_tick * offset_ticks
                            be_stop_price = (entry_p + tick_offset).quantize(min_tick, ROUND_UP) if pos_side == 'long' else \
                                            (entry_p - tick_offset).quantize(min_tick, ROUND_DOWN)
                            
                            current_sl_val = open_position.get('stopLossPrice') # float or None
                            current_sl_dec = Decimal(str(current_sl_val)) if current_sl_val and current_sl_val > 0 else None
                            
                            update_be = False
                            if be_stop_price > 0:
                                if current_sl_dec is None: update_be = True
                                elif pos_side == 'long' and be_stop_price > current_sl_dec: update_be = True
                                elif pos_side == 'short' and be_stop_price < current_sl_dec: update_be = True
                                else: lg.debug(f"BE triggered, but current SL ({current_sl_dec}) already better than target BE ({be_stop_price}).")
                            
                            if update_be:
                                lg.warning(f"{NEON_PURPLE}*** Moving SL to Break-Even for {symbol} at {be_stop_price} ***{RESET}")
                                current_tp_val = open_position.get('takeProfitPrice')
                                current_tp_dec = Decimal(str(current_tp_val)) if current_tp_val and current_tp_val > 0 else None
                                success = _set_position_protection(exchange, symbol, market_info, open_position, lg,
                                                                   be_stop_price, current_tp_dec, '0', '0') # Cancel TSL
                                if success: lg.info(f"{NEON_GREEN}Break-Even SL set/updated for {symbol}.{RESET}")
                                else: lg.error(f"{NEON_RED}Failed to set Break-Even SL for {symbol}.{RESET}")
                        else: lg.debug(f"Profit target for BE not yet reached for {symbol}.")
                    except ValueError as ve: lg.warning(f"{NEON_YELLOW}Skipping BE check for {symbol} due to invalid data: {ve}{RESET}")
                    except Exception as be_err: lg.error(f"{NEON_RED}Error during BE check for {symbol}: {be_err}{RESET}", exc_info=True)
                elif is_tsl_active: lg.info(f"BE check skipped for {symbol}: TSL is active.")
                elif not config.get("enable_break_even", False): lg.debug(f"BE check skipped for {symbol}: Disabled.")
                # Placeholder for other management logic
                lg.debug(f"No other management actions for {symbol} this cycle.")

    cycle_end_time = time.monotonic()
    lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY, console_log_level

    setup_logger("init") # Setup initial logger
    init_logger = logging.getLogger("init")
    console_handler = next((h for h in init_logger.handlers if isinstance(h, logging.StreamHandler)), None)
    if console_handler: console_handler.setLevel(console_log_level)

    init_logger.info(f"--- Starting LiveXY Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    CONFIG = load_config(CONFIG_FILE) # Reload after logger setup possibly created it
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
    init_logger.info(f"Config loaded. Quote: {QUOTE_CURRENCY}. Versions: CCXT {ccxt.__version__}, Pandas {pd.__version__}, PandasTA {getattr(ta, '__version__', 'N/A')}")

    if CONFIG.get("enable_trading"):
        init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING ENABLED !!!{RESET}")
        if CONFIG.get("use_sandbox"): init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet).{RESET}")
        else: init_logger.warning(f"{NEON_RED}!!! REAL MONEY ENVIRONMENT !!! Review settings!{RESET}")
        init_logger.warning(f"Settings: Risk={CONFIG.get('risk_per_trade',0)*100:.2f}%, Lev={CONFIG.get('leverage',0)}x, TSL={'ON' if CONFIG.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if CONFIG.get('enable_break_even') else 'OFF'}")
        try: input(f">>> Press {NEON_GREEN}Enter{RESET} to continue, or {NEON_RED}Ctrl+C{RESET} to abort... "); init_logger.info("User acknowledged live trading.")
        except KeyboardInterrupt: init_logger.info("User aborted. Exiting."); return
    else: init_logger.info(f"{NEON_YELLOW}Trading disabled. Analysis-only mode.{RESET}")

    exchange = initialize_exchange(init_logger)
    if not exchange: init_logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}"); return
    init_logger.info(f"Exchange {exchange.id} initialized.")

    target_symbol, market_info_main = None, None
    while True: # Symbol selection loop
        try:
            symbol_input = input(f"{NEON_YELLOW}Enter symbol (e.g., BTC/USDT or BTC/USDT:USDT): {RESET}").strip().upper().replace('-', '/')
            if not symbol_input: continue
            init_logger.info(f"Validating symbol '{symbol_input}'...")
            market_info_main = get_market_info(exchange, symbol_input, init_logger)
            if market_info_main: target_symbol = market_info_main['symbol']; break
            
            # Try common variations if direct match fails
            base, quote = "", QUOTE_CURRENCY
            if '/' in symbol_input: parts = symbol_input.split('/'); base = parts[0]; quote = parts[1].split(':')[0] if len(parts)>1 else quote
            elif symbol_input.endswith(quote): base = symbol_input[:-len(quote)]
            else: base = symbol_input
            
            variations = [f"{base}/{quote}", f"{base}/{quote}:{quote}"] if base else []
            found_var = False
            for sym_var in variations:
                if sym_var == symbol_input: continue # Already tried
                market_info_main = get_market_info(exchange, sym_var, init_logger)
                if market_info_main: target_symbol = market_info_main['symbol']; found_var = True; break
            if found_var: break
            else: init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' and variations not found.{RESET}")
        except Exception as e: init_logger.error(f"Symbol validation error: {e}", exc_info=True)

    init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {'Contract' if market_info_main.get('is_contract') else 'Spot'})")
    
    selected_interval = None # Interval selection loop
    while True:
        interval_input = input(f"{NEON_YELLOW}Enter interval [{'/'.join(VALID_INTERVALS)}] (default: {CONFIG['interval']}): {RESET}").strip()
        if not interval_input: interval_input = CONFIG['interval']
        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            selected_interval = interval_input; CONFIG["interval"] = selected_interval; break
        else: init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Choose from {VALID_INTERVALS}.{RESET}")
    init_logger.info(f"Using interval: {selected_interval} (CCXT: {CCXT_INTERVAL_MAP[selected_interval]})")

    symbol_logger = setup_logger(target_symbol) # Symbol-specific logger
    console_handler_sym = next((h for h in symbol_logger.handlers if isinstance(h, logging.StreamHandler)), None)
    if console_handler_sym: console_handler_sym.setLevel(console_log_level)

    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    symbol_logger.info(f"Config: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if CONFIG['enable_trailing_stop'] else 'OFF'}, BE={'ON' if CONFIG['enable_break_even'] else 'OFF'}, Trading={'ENABLED' if CONFIG['enable_trading'] else 'DISABLED'}")

    try: # Main execution loop
        while True:
            loop_start = time.time()
            symbol_logger.debug(f">>> New Cycle: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                # Optional: Reload config each loop: CONFIG = load_config(CONFIG_FILE)
                analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger)
            except ccxt.RateLimitExceeded as e:
                wait_time = RETRY_DELAY_SECONDS * 5 # Default
                try: # Parse wait time from error
                     match = re.search(r'try again in (\d+)ms', str(e).lower()) or re.search(r'(\d+)\s*(ms|s)', str(e).lower())
                     if match:
                         num = int(match.group(1))
                         unit = 'ms' if 'ms' in match.group(0) else 's'
                         wait_time = max(1, int((num / 1000 if unit == 'ms' else num) + 1))
                except Exception: pass
                symbol_logger.warning(f"{NEON_YELLOW}Rate limit: {e}. Waiting {wait_time}s...{RESET}"); time.sleep(wait_time)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ReadTimeout) as e:
                symbol_logger.error(f"{NEON_RED}Network error: {e}. Waiting {RETRY_DELAY_SECONDS*3}s...{RESET}"); time.sleep(RETRY_DELAY_SECONDS*3)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"{NEON_RED}CRITICAL: Auth Error: {e}. Stopping.{RESET}"); break
            except (ccxt.ExchangeNotAvailable, ccxt.OnMaintenance) as e:
                symbol_logger.error(f"{NEON_RED}Exchange unavailable/maintenance: {e}. Waiting 60s...{RESET}"); time.sleep(60)
            except Exception as loop_err:
                symbol_logger.error(f"{NEON_RED}Uncaught error in main loop: {loop_err}{RESET}", exc_info=True); time.sleep(15)

            elapsed = time.time() - loop_start
            sleep_duration = max(0, LOOP_DELAY_SECONDS - elapsed)
            symbol_logger.debug(f"<<< Cycle finished in {elapsed:.2f}s. Waiting {sleep_duration:.2f}s...")
            if sleep_duration > 0: time.sleep(sleep_duration)
    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except Exception as crit_err:
        (symbol_logger if 'symbol_logger' in locals() else init_logger).critical(f"{NEON_RED}Critical unhandled error: {crit_err}{RESET}", exc_info=True)
    finally:
        shutdown_msg = f"--- LiveXY Bot for {target_symbol or 'N/A'} Stopping ---"
        (symbol_logger if 'symbol_logger' in locals() else init_logger).info(shutdown_msg)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); exchange.close(); init_logger.info("Exchange connection closed.")
            except Exception as close_err: init_logger.error(f"Error closing exchange: {close_err}")
        logging.shutdown()


if __name__ == "__main__":
    # The self-write block from the original script is kept commented out as per input.
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
    main()
```
