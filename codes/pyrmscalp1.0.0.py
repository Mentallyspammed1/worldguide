"""Enhanced Multi-Symbol Trading Bot for Bybit (V5 API)
Merges features, optimizations, and best practices from previous versions.
Includes: pandas_ta.Strategy, Decimal precision, robust CCXT interaction,
          multi-symbol support, state management, TSL/BE logic, MA cross exit.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any

try:
    from zoneinfo import ZoneInfo  # Preferred (Python 3.9+)
except ImportError:
    from pytz import timezone as ZoneInfo  # Fallback for older Python

# Third-party libraries - alphabetized
import ccxt
import numpy as np  # pandas_ta might require numpy
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialization ---
try:
    getcontext().prec = 36  # Increased precision for intermediate calculations
except Exception:
    pass
init(autoreset=True)    # Initialize colorama
load_dotenv()           # Load environment variables from .env file

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Environment Variable Loading and Validation ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit(1)  # Exit if keys are missing

# --- Configuration File and Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
STATE_FILE = "bot_state.json"  # File to persist state like break-even triggers

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Timezone for logging and display
try:
    TZ_NAME = os.getenv("BOT_TIMEZONE", "America/Chicago")  # Default timezone
    TIMEZONE = ZoneInfo(TZ_NAME)
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# --- API Interaction Constants ---
MAX_API_RETRIES = 4  # Increased max retries slightly
RETRY_DELAY_SECONDS = 5  # Base delay between retries
RATE_LIMIT_BUFFER_SECONDS = 0.5  # Add small buffer to rate limit waits
MARKET_RELOAD_INTERVAL_SECONDS = 3600  # Reload markets every hour
POSITION_CONFIRM_DELAY = 10  # Seconds to wait after placing order before checking position status

# Intervals supported by the bot's internal logic (ensure config matches)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
# Map bot intervals to ccxt's expected timeframe format
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}

# --- Default Indicator/Strategy Parameters (can be overridden by config.json) ---
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14  # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 14     # Window for underlying RSI in StochRSI
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0  # Ensure float for calculations
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels

# --- Bot Timing and Delays ---
DEFAULT_LOOP_DELAY_SECONDS = 15  # Default time between full cycles

# --- Global Variables ---
loggers: dict[str, logging.Logger] = {}  # Dictionary to hold logger instances
console_log_level = logging.INFO  # Default, can be changed by args
QUOTE_CURRENCY = "USDT"  # Default, updated from config
LOOP_DELAY_SECONDS = DEFAULT_LOOP_DELAY_SECONDS  # Default, updated from config


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information (API keys) from logs."""
    REDACTED_STR = "***REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        original_message = super().format(record)
        redacted_message = original_message
        # More robust redaction - less likely to partially match
        if API_KEY and len(API_KEY) > 4:
            redacted_message = redacted_message.replace(API_KEY, self.REDACTED_STR)
        if API_SECRET and len(API_SECRET) > 4:
            redacted_message = redacted_message.replace(API_SECRET, self.REDACTED_STR)
        return redacted_message


class LocalTimeFormatter(SensitiveFormatter):
    """Formatter that uses the configured local timezone for console output."""
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
        return dt.timetuple()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=TIMEZONE)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{s},{int(record.msecs):03d}"
        return s


def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up a logger with rotating file and timezone-aware console handlers."""
    global console_log_level  # Access global console level setting

    logger_instance_name = f"livebot_{name.replace('/', '_').replace(':', '-')}" if is_symbol_logger else f"livebot_{name}"

    if logger_instance_name in loggers:
        logger = loggers[logger_instance_name]
        # Update existing console handler level if needed
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        # No need to log reconfiguration unless debugging logger setup itself
        # logger.debug(f"Logger '{logger_instance_name}' already configured. Console level set to {logging.getLevelName(console_log_level)}.")
        return logger

    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)

    # Prevent setting level lower than WARNING for root/ccxt loggers if desired
    effective_level = logging.DEBUG  # Default: Capture everything at logger level
    logger.setLevel(effective_level)

    # --- File Handler (UTC Timestamps) ---
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use UTC for file logs
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Ensure UTC conversion for file handler
        file_formatter.converter = time.gmtime
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to the file
        logger.addHandler(file_handler)
    except Exception:
        pass

    # --- Stream Handler (Local Time Timestamps) ---
    stream_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    stream_formatter = LocalTimeFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} {TIMEZONE.tzname(datetime.now(TIMEZONE))} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S,%f'  # Include microseconds if needed
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level)  # Set level from global variable
    logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent duplicate logs in parent/root logger
    loggers[logger_instance_name] = logger
    # Initial info log goes through the handlers configured above
    logger.info(f"Logger '{logger_instance_name}' initialized. File: '{os.path.basename(log_filename)}', Console Level: {logging.getLevelName(console_log_level)}")
    return logger


def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance."""
    return setup_logger(name, is_symbol_logger)


# --- Configuration Management ---
def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    keys_added_or_type_mismatch = False
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
            keys_added_or_type_mismatch = True
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_updated_config, nested_keys_added = _ensure_config_keys(updated_config[key], default_value)
            if nested_keys_added:
                updated_config[key] = nested_updated_config
                keys_added_or_type_mismatch = True
        elif updated_config.get(key) is not None and type(default_value) != type(updated_config.get(key)):
            # Check type mismatch, allowing Int -> Float/Decimal promotion silently
            is_promoting_num = (isinstance(default_value, (float, Decimal)) and isinstance(updated_config.get(key), int))
            if not is_promoting_num:
                # Warn about other type mismatches but keep the loaded value
                pass
                # No need to set keys_added_or_type_mismatch = True here, as we are using the loaded value

    return updated_config, keys_added_or_type_mismatch


def load_config(filepath: str) -> dict[str, Any]:
    """Load config, create default if missing, ensure keys exist, save updates."""
    default_config = {
        "symbols": ["BTC/USDT:USDT"],  # List of CCXT symbols (e.g., BASE/QUOTE:SETTLE)
        "interval": "5",  # Default interval (string, ensure it's in VALID_INTERVALS)
        "loop_delay": DEFAULT_LOOP_DELAY_SECONDS,
        "quote_currency": "USDT",
        "enable_trading": False,  # SAFETY FIRST: Default to False
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet)
        "risk_per_trade": 0.01,  # Risk 1% (as float)
        "leverage": 20,          # Desired leverage (check exchange limits!)
        "max_concurrent_positions_total": 1,  # Global limit across all symbols
        "position_mode": "One-Way",  # "One-Way" or "Hedge" - MUST MATCH EXCHANGE SETTING
        # --- Indicator Periods & Settings ---
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
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        # --- Signal Generation & Thresholds ---
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5,  # Use float
        "stoch_rsi_oversold_threshold": 25.0,  # Use float
        "stoch_rsi_overbought_threshold": 75.0,  # Use float
        "volume_confirmation_multiplier": 1.5,  # Use float
        "scalping_signal_threshold": 2.5,  # Use float (for 'scalping' weight set)
        # --- Risk Management Multipliers (based on ATR) ---
        "stop_loss_multiple": 1.8,  # Use float (for initial SL sizing)
        "take_profit_multiple": 0.7,  # Use float
        # --- Exit Strategies ---
        "enable_ma_cross_exit": True,
        # --- Trailing Stop Loss Config (Exchange-based TSL) ---
        "enable_trailing_stop": True,
        # Trail distance as a percentage of entry price (e.g., 0.005 for 0.5%)
        # This percentage is used to calculate the absolute 'trailingStop' value for Bybit API
        "trailing_stop_callback_rate": 0.005,  # Use float
        # Activate TSL when price moves this percentage in profit (e.g., 0.003 for 0.3%)
        # Set to 0 for immediate activation (API value '0'). Calculated into 'activePrice' for Bybit API.
        "trailing_stop_activation_percentage": 0.003,  # Use float
        # --- Break-Even Stop Config ---
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,  # Use float (Trigger BE when profit = X * ATR)
        "break_even_offset_ticks": 2,  # Use integer (Place BE SL N ticks beyond entry)
        "break_even_force_fixed_sl": True,  # Cancel TSL and set fixed SL at BE price? (Safer for Bybit V5)
        # --- Indicator Enable/Disable Flags ---
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        # --- Indicator Weighting Sets ---
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": {
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default"  # Which weight set to use
    }

    # --- File Handling ---
    config_to_use = default_config
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            # Use the default config we just wrote
        except OSError:
            pass
            # config_to_use remains default_config
    else:
        # --- Load Existing Config and Merge Defaults ---
        try:
            with open(filepath, encoding="utf-8") as f:
                config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config
            updated_config_from_file, keys_added = _ensure_config_keys(config_from_file, default_config)
            config_to_use = updated_config_from_file  # Use the potentially updated loaded config
            if keys_added:
                # Save the updated config back to the file
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(config_to_use, f_write, indent=4, sort_keys=True)
                except OSError:
                    pass  # Failed to save, clear the flag
        except (FileNotFoundError, json.JSONDecodeError):
            config_to_use = default_config  # Revert to default
            try:  # Attempt to recreate default if loading failed
                with open(filepath, "w", encoding="utf-8") as f_recreate:
                    json.dump(default_config, f_recreate, indent=4, sort_keys=True)
            except OSError:
                pass
        except Exception:
            config_to_use = default_config  # Revert to default

    # Ensure numeric types are correct (e.g., convert floats/ints loaded from JSON to Decimal where needed)
    # This is better handled when accessing the config values, using Decimal(str(config.get(...)))
    # but we can do a basic pass here if necessary for critical values.

    return config_to_use


# --- State Management ---
def load_state(filepath: str, logger: logging.Logger) -> dict[str, Any]:
    """Loads the bot's state from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"Loaded previous state from {filepath}")
                return state
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading state file {filepath}: {e}. Starting with empty state.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}. Starting with empty state.", exc_info=True)
            return {}
    else:
        logger.info("No previous state file found. Starting with empty state.")
        return {}


def save_state(filepath: str, state: dict[str, Any], logger: logging.Logger) -> None:
    """Saves the bot's state to a JSON file."""
    try:
        # Create a copy to avoid modifying the original during serialization
        state_to_save = state.copy()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=4)
        logger.debug(f"Saved current state to {filepath}")
    except (OSError, TypeError) as e:
        logger.error(f"Error saving state file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state: {e}", exc_info=True)


# --- CCXT Exchange Setup ---
def initialize_exchange(config: dict[str, Any], logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with V5 settings, error handling, and tests."""
    lg = logger
    global QUOTE_CURRENCY  # Allow updating global quote currency

    try:
        # Update global quote currency from config
        QUOTE_CURRENCY = config.get("quote_currency", "USDT")
        lg.info(f"Using Quote Currency: {QUOTE_CURRENCY}")

        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let ccxt handle basic rate limiting
            'rateLimit': 120,  # Bybit V5 has complex limits, set a base ms delay (adjust as needed)
            'options': {
                'defaultType': 'linear',  # Assume linear contracts (USDT margined)
                'adjustForTimeDifference': True,  # Auto-sync time with server
                # Connection timeouts (milliseconds) - Use generous timeouts
                'recvWindow': 10000,         # Bybit default 5000, increased safety
                'fetchTickerTimeout': 15000,  # 15 seconds
                'fetchBalanceTimeout': 20000,  # 20 seconds
                'createOrderTimeout': 25000,  # 25 seconds
                'fetchOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'cancelOrderTimeout': 20000,
                'fetchOHLCVTimeout': 20000,
                'setLeverageTimeout': 20000,
                'fetchMarketsTimeout': 30000,  # Longer timeout for market loading

                'brokerId': 'EnhancedWhale71',  # Add a broker ID for Bybit tracking
                # Explicitly request V5 API for relevant endpoints using 'versions'
                'versions': {
                    'public': {
                        'GET': {
                            'market/tickers': 'v5',
                            'market/kline': 'v5',
                            'market/orderbook': 'v5',
                        },
                    },
                    'private': {
                        'GET': {
                            'position/list': 'v5',
                            'account/wallet-balance': 'v5',
                            'order/realtime': 'v5',  # For fetching open orders
                            'order/history': 'v5',  # For fetching historical orders
                        },
                        'POST': {
                            'order/create': 'v5',
                            'order/cancel': 'v5',
                            'position/set-leverage': 'v5',
                            'position/trading-stop': 'v5',  # For SL/TP/TSL
                        },
                    },
                },
                 # Fallback default options (can help if 'versions' mapping is incomplete in CCXT)
                 'default_options': {
                    'fetchPositions': 'v5',
                    'fetchBalance': 'v5',
                    'createOrder': 'v5',
                    'fetchOrder': 'v5',
                    'fetchTicker': 'v5',
                    'fetchOHLCV': 'v5',
                    'fetchOrderBook': 'v5',
                    'setLeverage': 'v5',
                    'private_post_v5_position_trading_stop': 'v5',  # More explicit target
                },
                # Map standard types to Bybit V5 account types
                'accountsByType': {
                    'spot': 'SPOT',
                    'future': 'CONTRACT',  # Covers linear/inverse implicitly via symbol
                    'swap': 'CONTRACT',   # Covers linear/inverse implicitly via symbol
                    'margin': 'UNIFIED',  # Unified handles margin
                    'option': 'OPTION',
                    'unified': 'UNIFIED',
                    'contract': 'CONTRACT',
                },
                 # Map Bybit V5 account types back if needed (less common use)
                'accountsById': {
                    'SPOT': 'spot',
                    'CONTRACT': 'future',  # Map back to a common type
                    'UNIFIED': 'unified',
                    'OPTION': 'option',
                },
                'bybit': {  # Bybit specific options
                     'defaultSettleCoin': QUOTE_CURRENCY,
                }
            }
        }

        exchange_id = 'bybit'
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        if config.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}--- USING LIVE TRADING MODE (Real Money) ---{RESET}")

        # --- Test Connection & Load Markets ---
        lg.info(f"Connecting to {exchange.id} (Sandbox: {config.get('use_sandbox', True)})...")
        lg.info(f"Loading markets for {exchange.id}... (CCXT Version: {ccxt.__version__})")
        try:
            # Let CCXT handle initial market loading, but ensure it happens
            exchange.load_markets()
            exchange.last_load_markets_timestamp = time.time()
            lg.info(f"Markets loaded successfully for {exchange.id}. Found {len(exchange.markets)} markets.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            lg.error(f"{NEON_RED}Error loading markets: {e}. Check connection/API settings.{RESET}", exc_info=True)
            # traceback.print_exc() # Print full traceback for market load errors
            return None

        # --- Test API Credentials & Permissions (Fetch Balance) ---
        lg.info(f"Attempting initial balance fetch for {QUOTE_CURRENCY}...")
        try:
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, lg)  # Uses the robust version

            if balance_decimal is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 # fetch_balance already logs errors/warnings
                 lg.warning(f"{NEON_YELLOW}Initial balance fetch returned None or failed. Check logs. Ensure API keys have 'Read' permissions and correct account type (CONTRACT/UNIFIED/SPOT) is accessible for {QUOTE_CURRENCY}.{RESET}")
                 if config.get("enable_trading"):
                     lg.error(f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}")
                     return None
                 else:
                     lg.warning("Continuing in non-trading mode despite balance fetch issue.")

        except ccxt.AuthenticationError as auth_err:
            lg.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            lg.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None
        except Exception as balance_err:
            lg.warning(f"{NEON_YELLOW}Warning during initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")
            # traceback.print_exc()
            if config.get("enable_trading"):
                 lg.error(f"{NEON_RED}Aborting initialization due to balance fetch error in trading mode.{RESET}")
                 return None

        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError, Exception) as e:
        lg.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None


# --- CCXT API Call Helper with Retries ---
def safe_ccxt_call(
    exchange: ccxt.Exchange,
    method_name: str,
    logger: logging.Logger,
    max_retries: int = MAX_API_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
    *args, **kwargs
) -> Any:
    """Safely calls a CCXT method with retry logic for network errors and rate limits."""
    lg = logger
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            method = getattr(exchange, method_name)
            # lg.debug(f"Calling {method_name} (Attempt {attempt + 1}/{max_retries + 1}) Args: {args}, Kwargs: {kwargs}")
            result = method(*args, **kwargs)
            # lg.debug(f"{method_name} call successful.")
            return result
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * (2 ** attempt)  # Default exponential backoff
            suggested_wait = None
            try:  # Try parsing suggested wait time from Bybit error message
                import re
                # Example Bybit msg: "Rate limit exceeded, please try again in 153 ms." or "Too many visits!"
                error_msg = str(e).lower()
                match_ms = re.search(r'(?:try again in|retry after)\s*(\d+)\s*ms', error_msg)
                match_s = re.search(r'(?:try again in|retry after)\s*(\d+)\s*s', error_msg)
                if match_ms: suggested_wait = max(1, math.ceil(int(match_ms.group(1)) / 1000) + RATE_LIMIT_BUFFER_SECONDS)
                elif match_s: suggested_wait = max(1, int(match_s.group(1)) + RATE_LIMIT_BUFFER_SECONDS)
                elif "too many visits" in error_msg or "limit" in error_msg:
                    # Generic rate limit message, use exponential backoff but maybe slightly longer
                    suggested_wait = wait_time + 1.0
            except Exception: pass  # Ignore parsing errors, use default backoff

            final_wait = suggested_wait if suggested_wait is not None else wait_time
            lg.warning(f"Rate limit hit calling {method_name}. Retrying in {final_wait:.2f}s... (Attempt {attempt + 1}) Error: {e}")
            time.sleep(final_wait)
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            lg.warning(f"Network/DDoS/Timeout error calling {method_name}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1})")
            time.sleep(wait_time)
        except ccxt.AuthenticationError as e:
            lg.error(f"{NEON_RED}Authentication Error calling {method_name}: {e}. Check API keys/permissions. Not retrying.{RESET}")
            raise e  # Re-raise critical auth errors
        except ccxt.ExchangeError as e:
            # Check for specific non-retryable exchange errors
            # Bybit V5 Error Codes: https://bybit-exchange.github.io/docs/v5/error_code
            bybit_code = getattr(e, 'code', None)  # CCXT sometimes puts code here
            if bybit_code is None and hasattr(e, 'args') and len(e.args) > 0:
                 # Sometimes the code is in the message, e.g., "bybit {"retCode":110007,...}"
                 try:
                     error_details = str(e.args[0])
                     if "retCode" in error_details:
                         # Basic parsing, might need refinement
                         details_dict = json.loads(error_details[error_details.find('{'):error_details.rfind('}') + 1])
                         bybit_code = details_dict.get('retCode')
                 except Exception: pass  # Ignore parsing errors

            # List of potentially non-retryable Bybit errors (add more as needed)
            # 110007: Insufficient margin / balance
            # 110013: Leverage exceeds risk limit tier
            # 110017: Order qty exceeds max limit
            # 110020: Position size would exceed risk limit
            # 110025: Invalid order price (too far from mark price, etc.)
            # 110043: Set leverage failed (same leverage, etc) - Can sometimes be ignored?
            # 110045: Position status is not normal (e.g., during liquidation)
            # 170007: Invalid symbol / instrument not found
            # 170131: Quantity too low / less than min order qty
            # 170132: Price precision invalid
            # 170133: Qty precision invalid
            # 170140: Order cost too low / less than min order value
            non_retryable_codes = [110007, 110013, 110017, 110020, 110025, 110045, 170007, 170131, 170132, 170133, 170140]

            if bybit_code in non_retryable_codes:
                lg.error(f"{NEON_RED}Non-retryable Exchange Error calling {method_name}: {e} (Code: {bybit_code}). Not retrying.{RESET}")
                raise e  # Re-raise immediately
            else:
                 lg.warning(f"{NEON_YELLOW}Retryable Exchange Error calling {method_name}: {e} (Code: {bybit_code}). Retrying... (Attempt {attempt + 1}){RESET}")
                 last_exception = e
                 wait_time = retry_delay * (2 ** attempt)
                 time.sleep(wait_time)  # Wait and retry for other exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected Error calling {method_name}: {e}. Not retrying.{RESET}", exc_info=True)
            # traceback.print_exc()
            raise e  # Re-raise unexpected errors

    lg.error(f"{NEON_RED}Max retries ({max_retries}) reached for {method_name}. Last error: {last_exception}{RESET}")
    raise last_exception  # Re-raise the last exception after exhausting retries


# --- Market Info Helper Functions ---
def _determine_category(market_info: dict, logger: logging.Logger) -> str | None:
    """Helper to determine Bybit V5 category from market info."""
    if not market_info: return None
    market_type = market_info.get('type')  # spot, swap, future, option
    is_linear = market_info.get('linear', False)
    is_inverse = market_info.get('inverse', False)
    is_contract = market_info.get('contract', False)
    is_spot = market_info.get('spot', False)
    is_option = market_info.get('option', False)
    symbol = market_info.get('symbol', 'N/A')

    # Prioritize standard CCXT types
    if market_type == 'spot': return 'spot'
    if market_type == 'option': return 'option'
    # For futures/swaps, determine linear/inverse
    if market_type == 'swap' or market_type == 'future' or is_contract:
        if is_linear: return 'linear'
        if is_inverse: return 'inverse'
        # Fallback guess for contracts based on settle currency if flags missing
        settle = market_info.get('settle', '').upper()
        quote = market_info.get('quote', '').upper()
        if settle == 'USDT' or settle == 'USDC' or settle == quote:
            # logger.debug(f"Assuming 'linear' for {symbol} based on settle/quote ({settle}/{quote}).")
            return 'linear'
        elif settle == market_info.get('base', '').upper():
            # logger.debug(f"Assuming 'inverse' for {symbol} based on settle/base ({settle}).")
            return 'inverse'
        else:
            logger.warning(f"Could not determine linear/inverse for contract {symbol} (Settle: {settle}). Defaulting to linear.")
            return 'linear'  # Default guess

    # Fallback based on flags if type is missing/unknown
    elif is_linear: return 'linear'
    elif is_inverse: return 'inverse'
    elif is_spot: return 'spot'
    elif is_option: return 'option'
    else:
        logger.warning(f"Could not determine category for market {symbol}. Market Type: {market_type}, Flags: L={is_linear},I={is_inverse},C={is_contract},S={is_spot},O={is_option}. Defaulting to None.")
        return None  # Cannot determine


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Fetches and returns enhanced market information for a symbol, handling reloads."""
    lg = logger
    try:
        # Check if reload is needed
        reload_needed = False
        now = time.time()
        last_load_time = getattr(exchange, 'last_load_markets_timestamp', 0)

        if now - last_load_time > MARKET_RELOAD_INTERVAL_SECONDS:
            reload_needed = True
            lg.info(f"Market info older than {MARKET_RELOAD_INTERVAL_SECONDS}s ({now - last_load_time:.0f}s). Reloading...")
        elif symbol not in exchange.markets:
             reload_needed = True
             lg.warning(f"Symbol {symbol} not found in cached markets. Reloading...")

        if reload_needed:
            try:
                lg.info("Attempting to reload markets...")
                # Use safe call for loading markets
                markets = safe_ccxt_call(exchange, 'load_markets', lg, reload=True)
                if markets:
                    exchange.last_load_markets_timestamp = time.time()
                    lg.info(f"Markets reloaded successfully. Found {len(markets)} markets.")
                else:
                    lg.error(f"{NEON_RED}Market reload call returned None/empty. Check API/connection.{RESET}")
                    # Attempt to use potentially stale data if available
                    if symbol not in exchange.markets:
                         return None  # Cannot proceed without market info
            except Exception as load_err:
                 lg.error(f"{NEON_RED}Failed to reload markets: {load_err}{RESET}")
                 # traceback.print_exc()
                 if symbol not in exchange.markets:
                     return None

        # Get market data using ccxt's safe method
        market = exchange.market(symbol)
        if not market:
             lg.error(f"{NEON_RED}Market data for {symbol} could not be retrieved even after load attempt.{RESET}")
             # Maybe the symbol format is wrong (e.g., missing :SETTLE)?
             if ':' not in symbol and QUOTE_CURRENCY:
                 test_symbol = f"{symbol}:{QUOTE_CURRENCY}"
                 lg.warning(f"Attempting fetch with settlement currency: {test_symbol}")
                 market = exchange.market(test_symbol)
                 if market:
                      lg.info(f"Successfully found market using {test_symbol}")
                      symbol = test_symbol  # Update symbol if found this way
                 else:
                      lg.error(f"Still failed to find market for {symbol} or {test_symbol}")
                      return None
             else:
                 return None

        # --- Enhance market dict with standardized/derived info ---
        # Use a copy to avoid modifying the global exchange.markets dict directly
        enhanced_market = market.copy()

        enhanced_market['category'] = _determine_category(enhanced_market, lg)
        enhanced_market['min_tick_size'] = get_min_tick_size_from_market(enhanced_market, lg)
        enhanced_market['price_precision_digits'] = get_price_precision_digits_from_market(enhanced_market, lg)
        enhanced_market['amount_precision_digits'] = get_amount_precision_digits_from_market(enhanced_market, lg)
        enhanced_market['min_order_amount'] = get_min_order_amount_from_market(enhanced_market, lg)
        enhanced_market['min_order_cost'] = get_min_order_cost_from_market(enhanced_market, lg)
        enhanced_market['contract_size'] = get_contract_size_from_market(enhanced_market, lg)
        enhanced_market['is_contract'] = enhanced_market.get('contract', False) or enhanced_market.get('type') in ['swap', 'future']

        # Log key derived values if debugging
        # lg.debug(f"Enhanced Market Info ({symbol}): Category={enhanced_market['category']}, Tick={enhanced_market['min_tick_size']}, "
        #          f"PricePrec={enhanced_market['price_precision_digits']}, AmtPrec={enhanced_market['amount_precision_digits']}, "
        #          f"MinAmt={enhanced_market['min_order_amount']}, MinCost={enhanced_market['min_order_cost']}, "
        #          f"ContractSize={enhanced_market['contract_size']}, IsContract={enhanced_market['is_contract']}")

        return enhanced_market

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Symbol Error: {e}. Is '{symbol}' correctly formatted and supported by {exchange.id}?{RESET}")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        lg.error(f"{NEON_RED}API error fetching/processing market info for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching market info for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
    return None


def get_min_tick_size_from_market(market: dict, logger: logging.Logger) -> Decimal | None:
    """Extracts minimum price increment (tick size) from CCXT market structure as Decimal."""
    symbol = market.get('symbol', 'N/A')
    tick_size = None
    try:
        # 1. Standard CCXT precision structure
        precision_price = market.get('precision', {}).get('price')
        if precision_price is not None:
            tick_size = Decimal(str(precision_price))
            # logger.debug(f"Tick size from precision.price for {symbol}: {tick_size}")
            if tick_size > 0: return tick_size

        # 2. Bybit V5 info structure (more reliable)
        info = market.get('info', {})
        tick_size_str = info.get('priceFilter', {}).get('tickSize')
        if tick_size_str is not None:
            tick_size = Decimal(str(tick_size_str))
            # logger.debug(f"Tick size from info.priceFilter.tickSize for {symbol}: {tick_size}")
            if tick_size > 0: return tick_size

        # 3. Fallback: Infer from price precision digits (less accurate)
        digits = get_price_precision_digits_from_market(market, logger)  # Call helper, handles its own fallbacks
        if digits is not None:
            inferred_tick = Decimal('1') / (Decimal('10') ** digits)
            # logger.debug(f"Tick size inferred from price digits ({digits}) for {symbol}: {inferred_tick}")
            return inferred_tick

        logger.warning(f"Could not determine price tick size for {symbol}. Checked precision.price, info.priceFilter.tickSize, inferred from digits.")
        return None
    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining min tick size for {symbol}: {e}")
        return None


def get_price_precision_digits_from_market(market: dict, logger: logging.Logger) -> int | None:
    """Extracts number of decimal places for price from CCXT market structure."""
    symbol = market.get('symbol', 'N/A')
    digits = None
    try:
        # 1. Standard CCXT precision.price (if it represents digits directly)
        precision_price = market.get('precision', {}).get('price')
        if precision_price is not None:
            try:
                price_prec_dec = Decimal(str(precision_price))
                # If it looks like integer digits (>= 0)
                if price_prec_dec >= 0 and price_prec_dec == price_prec_dec.to_integral_value():
                    digits = int(price_prec_dec)
                    # logger.debug(f"Price digits from precision.price for {symbol}: {digits}")
                    if digits is not None: return digits
            except (InvalidOperation, ValueError, TypeError):
                 pass  # If conversion fails or it's not integer-like, proceed

        # 2. Infer from tick size (most reliable if tick size is found)
        min_tick = get_min_tick_size_from_market(market, logger)  # Use helper, handles its own logic
        if min_tick is not None and min_tick > 0:
            # Use Decimal's exponent after normalization
            digits = abs(min_tick.normalize().as_tuple().exponent)
            # logger.debug(f"Price digits inferred from tick size ({min_tick}) for {symbol}: {digits}")
            if digits is not None: return digits

        # 3. Fallback: Bybit 'priceScale' (often represents digits)
        info = market.get('info', {})
        price_scale_str = info.get('priceScale')
        if price_scale_str is not None:
            try:
                 digits = int(price_scale_str)
                 # logger.debug(f"Price digits from info.priceScale for {symbol}: {digits}")
                 if digits is not None and digits >= 0: return digits
            except (ValueError, TypeError): pass

        logger.warning(f"Could not determine price precision digits for {symbol}. Checked precision.price, inferred from tick, info.priceScale. Defaulting to 8.")
        return 8  # Fallback default
    except Exception as e:
        logger.error(f"Error determining price precision digits for {symbol}: {e}")
        return 8  # Fallback default


def get_amount_precision_digits_from_market(market: dict, logger: logging.Logger) -> int | None:
    """Extracts number of decimal places for amount/quantity from CCXT market structure."""
    symbol = market.get('symbol', 'N/A')
    digits = None
    try:
        # 1. Infer from amount step size (most reliable: lotSizeFilter.qtyStep)
        info = market.get('info', {})
        step_size_str = info.get('lotSizeFilter', {}).get('qtyStep')
        if step_size_str is not None:
            try:
                step_size = Decimal(str(step_size_str))
                if step_size > 0:
                    if step_size == 1: digits = 0  # Handle step size of 1
                    else: digits = abs(step_size.normalize().as_tuple().exponent)
                    # logger.debug(f"Amount digits from info.lotSizeFilter.qtyStep ({step_size}) for {symbol}: {digits}")
                    if digits is not None: return digits
            except (InvalidOperation, ValueError, TypeError): pass

        # 2. Standard CCXT precision.amount (often digits, sometimes step size)
        precision_amount = market.get('precision', {}).get('amount')
        if precision_amount is not None:
            try:
                amount_prec_dec = Decimal(str(precision_amount))
                if amount_prec_dec > 0 and amount_prec_dec <= 1:  # Looks like step size
                     if amount_prec_dec == 1: digits = 0
                     else: digits = abs(amount_prec_dec.normalize().as_tuple().exponent)
                     # logger.debug(f"Amount digits inferred from precision.amount step ({amount_prec_dec}) for {symbol}: {digits}")
                elif amount_prec_dec >= 0 and amount_prec_dec == amount_prec_dec.to_integral_value():  # Looks like digits
                     digits = int(amount_prec_dec)
                     # logger.debug(f"Amount digits directly from precision.amount ({digits}) for {symbol}: {digits}")

                if digits is not None: return digits
            except (InvalidOperation, ValueError, TypeError): pass

        # 3. Fallback: Infer from min order quantity (if step size missing)
        min_qty = get_min_order_amount_from_market(market, logger)  # Uses helper
        if min_qty is not None and min_qty > 0:
             if min_qty == 1: digits = 0
             else: digits = abs(min_qty.normalize().as_tuple().exponent)
             # logger.debug(f"Amount digits inferred from min qty ({min_qty}) for {symbol}: {digits}")
             if digits is not None: return digits

        logger.warning(f"Could not determine amount precision digits for {symbol}. Checked qtyStep, precision.amount, inferred from minQty. Defaulting to 8.")
        return 8  # Fallback default
    except Exception as e:
        logger.error(f"Error determining amount precision digits for {symbol}: {e}")
        return 8  # Fallback default


def get_min_order_amount_from_market(market: dict, logger: logging.Logger) -> Decimal | None:
    """Extracts minimum order amount/quantity from CCXT market structure as Decimal."""
    symbol = market.get('symbol', 'N/A')
    min_amount = None
    try:
        # 1. Bybit V5 info structure (most reliable)
        info = market.get('info', {})
        min_amount_str = info.get('lotSizeFilter', {}).get('minOrderQty')
        if min_amount_str is not None:
            min_amount = Decimal(str(min_amount_str))
            # logger.debug(f"Min amount from info.lotSizeFilter.minOrderQty for {symbol}: {min_amount}")
            if min_amount > 0: return min_amount

        # 2. Standard CCXT limits structure
        limits_amount = market.get('limits', {}).get('amount', {})
        min_amount_lim = limits_amount.get('min')
        if min_amount_lim is not None:
            min_amount = Decimal(str(min_amount_lim))
            # logger.debug(f"Min amount from limits.amount.min for {symbol}: {min_amount}")
            if min_amount > 0: return min_amount

        logger.warning(f"Could not find minimum order amount for {symbol}. Checked info.lotSizeFilter.minOrderQty, limits.amount.min.")
        return None
    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining min order amount for {symbol}: {e}")
        return None


def get_min_order_cost_from_market(market: dict, logger: logging.Logger) -> Decimal | None:
    """Extracts minimum order cost (value) from CCXT market structure as Decimal."""
    symbol = market.get('symbol', 'N/A')
    min_cost = None
    try:
        # 1. Bybit V5 info structure (Spot often has minOrderAmt)
        info = market.get('info', {})
        min_cost_str = info.get('lotSizeFilter', {}).get('minOrderAmt')  # Note: Often for Spot only
        if min_cost_str is not None:
            min_cost = Decimal(str(min_cost_str))
            # logger.debug(f"Min cost from info.lotSizeFilter.minOrderAmt for {symbol}: {min_cost}")
            if min_cost > 0: return min_cost

        # 2. Standard CCXT limits structure
        limits_cost = market.get('limits', {}).get('cost', {})
        min_cost_lim = limits_cost.get('min')
        if min_cost_lim is not None:
            min_cost = Decimal(str(min_cost_lim))
            # logger.debug(f"Min cost from limits.cost.min for {symbol}: {min_cost}")
            if min_cost > 0: return min_cost

        # logger.debug(f"Minimum order cost not explicitly specified for {symbol}. Checked info.lotSizeFilter.minOrderAmt, limits.cost.min. May need calculation.")
        return None  # Indicate not directly available
    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining min order cost for {symbol}: {e}")
        return None


def get_contract_size_from_market(market: dict, logger: logging.Logger) -> Decimal:
    """Extracts contract size from market info as Decimal, defaulting to 1."""
    symbol = market.get('symbol', 'N/A')
    contract_size = Decimal('1')  # Default
    try:
        # 1. Standard CCXT field
        cs_raw_ccxt = market.get('contractSize')
        if cs_raw_ccxt is not None:
            cs_dec = Decimal(str(cs_raw_ccxt))
            if cs_dec > 0:
                # logger.debug(f"Contract size from market.contractSize for {symbol}: {cs_dec}")
                contract_size = cs_dec; return contract_size  # Return immediately if found valid

        # 2. Bybit specific field in 'info'
        cs_raw_info = market.get('info', {}).get('contractSize')
        if cs_raw_info is not None:
            cs_dec = Decimal(str(cs_raw_info))
            if cs_dec > 0:
                # logger.debug(f"Contract size from info.contractSize for {symbol}: {cs_dec}")
                contract_size = cs_dec; return contract_size  # Return immediately

        # If neither field yields a positive value, the default '1' remains
        # logger.debug(f"Contract size not specified or invalid for {symbol}. Defaulting to {contract_size}.")

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining contract size for {symbol}: {e}. Defaulting to 1.")
        contract_size = Decimal('1')  # Ensure default on error

    return contract_size


# --- Data Fetching Wrappers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict) -> Decimal | None:
    """Fetch current price using safe_ccxt_call and market info."""
    lg = logger
    try:
        params = {}
        category = market_info.get('category')
        if category and exchange.id == 'bybit':  # Only add category for Bybit
            params['category'] = category
            # lg.debug(f"Using params for fetch_ticker ({symbol}): {params}")

        ticker = safe_ccxt_call(exchange, 'fetch_ticker', lg, symbol=symbol, params=params)
        if not ticker:
            lg.warning(f"fetch_ticker returned None for {symbol}.")
            return None

        # Helper to safely convert to Decimal and check positivity
        def safe_decimal(key: str) -> Decimal | None:
            value = ticker.get(key)
            if value is None: return None
            try:
                d_val = Decimal(str(value))
                return d_val if d_val > 0 else None
            except (InvalidOperation, ValueError, TypeError):
                # lg.warning(f"Invalid numeric value for '{key}' in ticker: {value}")
                return None

        # Prioritize: last > mark (if available) > midpoint(bid/ask) > ask > bid
        price = safe_decimal('last')
        if price: lg.debug(f"Using 'last' price: {price}"); return price

        mark = safe_decimal('mark')  # Bybit often provides mark price for derivatives
        if mark: lg.debug(f"Using 'mark' price: {mark}"); return mark

        bid = safe_decimal('bid')
        ask = safe_decimal('ask')
        if bid and ask and bid < ask:  # Ensure bid < ask
            price = (bid + ask) / Decimal('2')
            lg.debug(f"Using bid/ask midpoint: {price} (Bid: {bid}, Ask: {ask})")
            return price

        # Use ask/bid as last resort if only one is valid
        if ask: lg.warning(f"Using 'ask' price as fallback: {ask}"); return ask
        if bid: lg.warning(f"Using 'bid' price as last resort: {bid}"); return bid

        lg.error(f"{NEON_RED}Failed to extract a valid positive price from ticker for {symbol}. Ticker: {ticker}{RESET}")
        return None

    except Exception as e:
        lg.error(f"{NEON_RED}Error in fetch_current_price_ccxt for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger, market_info: dict) -> pd.DataFrame:
    """Fetch OHLCV klines using safe_ccxt_call and market info, returns processed DataFrame."""
    lg = logger
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
    if not ccxt_timeframe:
        lg.error(f"Invalid timeframe '{timeframe}' provided for {symbol}. Valid: {list(CCXT_INTERVAL_MAP.keys())}")
        return pd.DataFrame()

    try:
        params = {}
        category = market_info.get('category')
        # Kline endpoint often requires category for Bybit V5
        if category and exchange.id == 'bybit':
            params['category'] = category
            # lg.debug(f"Using params for fetch_ohlcv ({symbol}): {params}")

        # Use safe_ccxt_call for the API request
        ohlcv = safe_ccxt_call(exchange, 'fetch_ohlcv', lg, symbol=symbol, timeframe=ccxt_timeframe, limit=limit, params=params)

        if ohlcv is None or len(ohlcv) == 0:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline DataFrame empty for {symbol} {timeframe} after creation.{RESET}")
            return df

        # Convert timestamp to datetime, set as index (UTC by default from CCXT)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline DataFrame empty after timestamp conversion for {symbol}.{RESET}")
            return df
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric (Decimal for precision)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            # df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert directly to Decimal for consistency
            df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))

        initial_len = len(df)
        # Drop rows with NaN in critical OHLC columns or non-positive close
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > Decimal('0')]
        # Optional: Fill NaN volume with 0 if needed by indicators (MFI might need volume)
        df['volume'] = df['volume'].apply(lambda x: x if x.is_finite() else Decimal('0'))

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid OHLC data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data empty after cleaning for {symbol}.{RESET}")
            return pd.DataFrame()

        df.sort_index(inplace=True)  # Ensure chronological order
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe} (Limit: {limit})")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error in fetch_klines_ccxt for {symbol} {timeframe}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: dict) -> dict | None:
    """Fetch orderbook using safe_ccxt_call and market info."""
    lg = logger
    try:
        params = {}
        category = market_info.get('category')
        # Orderbook endpoint often requires category for Bybit V5
        if category and exchange.id == 'bybit':
            params['category'] = category
            # lg.debug(f"Using params for fetch_order_book ({symbol}): {params}")

        # Use safe_ccxt_call for the API request
        orderbook = safe_ccxt_call(exchange, 'fetch_order_book', lg, symbol=symbol, limit=limit, params=params)

        # Basic validation
        if not orderbook:
            lg.warning(f"fetch_order_book returned None/empty for {symbol}.")
            return None
        if not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
            lg.warning(f"{NEON_YELLOW}Invalid orderbook structure for {symbol}. Missing bids/asks lists. Response: {orderbook}{RESET}")
            return None
        if not orderbook['bids'] and not orderbook['asks']:
             lg.warning(f"{NEON_YELLOW}Orderbook received but bids/asks lists are empty for {symbol}.{RESET}")
             # Return empty book, might be valid state

        # lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
        return orderbook

    except Exception as e:
        lg.error(f"{NEON_RED}Error in fetch_orderbook_ccxt for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches available balance for a currency, handling Bybit V5 structures/retries."""
    lg = logger
    last_exception = None
    # Relevant account types for Bybit V5 (CONTRACT covers swap/future, UNIFIED for UTA)
    account_types_to_try = ['CONTRACT', 'UNIFIED', 'SPOT']

    for attempt in range(MAX_API_RETRIES + 1):
        balance_info = None
        successful_acc_type = None

        # Try fetching for specific account types
        for acc_type in account_types_to_try:
            try:
                params = {}
                # Bybit V5 fetch_balance often needs accountType and optionally coin
                if exchange.id == 'bybit':
                    params['accountType'] = acc_type
                    params['coin'] = currency  # Specify the coin to get detailed balance

                lg.debug(f"Fetching balance using params={params} for {currency} (Attempt {attempt + 1}, Type: {acc_type})")

                # Use safe_ccxt_call - disable inner retries to let outer loop control retries across types
                balance_info = safe_ccxt_call(exchange, 'fetch_balance', lg, max_retries=0, params=params)

                # Check if the response seems valid for this account type/currency
                # Need to parse the specific structure returned by Bybit V5
                if balance_info and 'info' in balance_info and balance_info['info'].get('retCode') == 0:
                    # Check within info -> result -> list
                    if 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
                        bal_list = balance_info['info']['result']['list']
                        if isinstance(bal_list, list) and len(bal_list) > 0:
                             # Check if the returned list contains our currency or account info
                             # V5 structure: list[{accountType, coin:[{coin, availableBalance,...}]}]
                             found_currency_in_list = False
                             for account_data in bal_list:
                                 if isinstance(account_data.get('coin'), list):
                                     for coin_data in account_data['coin']:
                                         if coin_data.get('coin') == currency:
                                             found_currency_in_list = True; break
                                 if found_currency_in_list: break

                             if found_currency_in_list:
                                 lg.debug(f"Received valid balance structure for accountType '{acc_type}'.")
                                 successful_acc_type = acc_type
                                 break  # Found valid data for this type, proceed to parse
                             else:
                                 lg.debug(f"Balance list received for '{acc_type}', but currency '{currency}' not found within.")
                                 balance_info = None  # Reset to try next type
                        else:
                             lg.debug(f"Balance list empty for accountType '{acc_type}'. Trying next.")
                             balance_info = None
                    else:
                        lg.debug(f"Balance structure for accountType '{acc_type}' missing 'result' or 'list'. Trying next. Info: {balance_info.get('info')}")
                        balance_info = None
                else:
                    ret_code = balance_info.get('info', {}).get('retCode', 'N/A') if balance_info else 'N/A'
                    ret_msg = balance_info.get('info', {}).get('retMsg', 'N/A') if balance_info else 'N/A'
                    lg.debug(f"Balance structure for accountType '{acc_type}' seems empty or indicates error (Code: {ret_code}, Msg: {ret_msg}). Trying next.")
                    balance_info = None  # Reset to try next type

            except ccxt.ExchangeError as e:
                # Ignore "account type not support" errors and try the next type
                ignore_msgs = ["account type not support", "invalid account type", "accounttype invalid"]
                # Bybit specific codes for account issues
                bybit_code = None
                if hasattr(e, 'args') and len(e.args) > 0:
                    try:
                        error_details = str(e.args[0])
                        if "retCode" in error_details:
                            details_dict = json.loads(error_details[error_details.find('{'):error_details.rfind('}') + 1])
                            bybit_code = details_dict.get('retCode')
                    except Exception: pass
                # 10001: Parameter error (could be invalid accountType)
                # 30086: Unified trade Account Type Error / Not supported operation
                if any(msg in str(e).lower() for msg in ignore_msgs) or bybit_code in [10001, 30086]:
                    lg.debug(f"Account type '{acc_type}' not supported or parameter error fetching balance: {e}. Trying next.")
                    last_exception = e
                    continue
                else:
                    lg.warning(f"Non-ignorable ExchangeError fetching balance with type {acc_type}: {e}")
                    last_exception = e
                    # Don't break inner loop, let outer loop handle retries if applicable
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                 # Let the outer retry loop handle these - break inner loop
                 lg.warning(f"Network/RateLimit error during balance fetch attempt with type {acc_type}: {e}")
                 last_exception = e
                 break  # Break inner loop, let outer loop retry
            except Exception as e:
                 lg.error(f"Unexpected error during balance fetch attempt with type {acc_type}: {e}", exc_info=True)
                 # traceback.print_exc()
                 last_exception = e
                 break  # Break inner loop, let outer loop retry

        # If a valid structure was found or network error occurred, break the outer retry loop
        if balance_info or isinstance(last_exception, (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded)):
             break

        # If specific types failed without network errors, wait and retry outer loop
        if attempt < MAX_API_RETRIES:
             wait_time = retry_delay * (2 ** attempt)
             lg.warning(f"Balance fetch attempt {attempt + 1} failed for all types. Retrying in {wait_time}s...")
             time.sleep(wait_time)
        else:
             lg.error(f"{NEON_RED}Max retries reached fetching balance for {currency}. Last error: {last_exception}{RESET}")
             return None  # Exhausted retries

    # --- Parse the balance_info (if successful) ---
    if not balance_info:
         lg.error(f"{NEON_RED}Failed to fetch any valid balance information for {currency} after all attempts. Last error: {last_exception}{RESET}")
         return None

    available_balance_str = None
    try:
        # Parse Bybit V5 structure: info -> result -> list -> coin[]
        if 'info' in balance_info and balance_info['info'].get('retCode') == 0 and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
            balance_list = balance_info['info']['result']['list']
            for account in balance_list:
                # V5 often returns only the requested account type, but check just in case
                # if account.get('accountType') == successful_acc_type: # Match the type that worked
                coin_list = account.get('coin')
                if isinstance(coin_list, list):
                    for coin_data in coin_list:
                        if coin_data.get('coin') == currency:
                            # Prefer 'availableToWithdraw' or 'availableBalance' (more accurate for trading)
                            # Bybit V5 names: availableBalance, walletBalance
                            free = coin_data.get('availableBalance')  # This seems most relevant for placing new trades
                            if free is None: free = coin_data.get('availableToWithdraw')  # Fallback
                            if free is None: free = coin_data.get('walletBalance')  # Last resort total balance

                            if free is not None and str(free).strip() != "":
                                available_balance_str = str(free)
                                lg.debug(f"Found balance via Bybit V5 structure: {available_balance_str} {currency} (Account: {account.get('accountType', 'Unknown')}, Field: {'availableBalance' if coin_data.get('availableBalance') is not None else ('availableToWithdraw' if coin_data.get('availableToWithdraw') is not None else 'walletBalance')})")
                                break  # Found the currency
                    if available_balance_str is not None: break  # Found the currency in this account type
            if available_balance_str is None:
                lg.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]' for account type '{successful_acc_type}'. Structure: {balance_info.get('info')}")

        # Fallback: Standard CCXT structure (less likely for Bybit V5 specific fetch)
        elif currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('free') is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}")

        elif 'free' in balance_info and isinstance(balance_info.get('free'), dict) and currency in balance_info['free']:
            available_balance_str = str(balance_info['free'][currency])
            lg.debug(f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency}")

        # If still not found, log error
        if available_balance_str is None:
            lg.error(f"{NEON_RED}Could not extract balance for {currency} from structure. Response Info: {balance_info.get('info')}{RESET}")
            # lg.debug(f"Full balance response: {balance_info}")
            return None

        # --- Convert to Decimal ---
        final_balance = Decimal(available_balance_str)
        if final_balance.is_finite() and final_balance >= 0:
            lg.info(f"Available {currency} balance: {final_balance:.4f}")
            return final_balance
        else:
            lg.error(f"Parsed balance for {currency} is invalid (Negative or NaN/Inf): {final_balance}"); return None

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Failed to convert balance '{available_balance_str}' to Decimal for {currency}: {e}."); return None
    except Exception as e:
        lg.error(f"Unexpected error parsing balance structure for {currency}: {e}", exc_info=True)
        # traceback.print_exc()
        return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta.Strategy, generates weighted signals,
    and provides risk management helpers. Uses Decimal for precision.
    Manages state like break-even trigger status via a passed-in dictionary.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],
        symbol_state: dict[str, Any],  # Mutable state dict for this symbol
    ) -> None:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
             raise ValueError("TradingAnalyzer requires a non-empty pandas DataFrame.")
        if not market_info or not isinstance(market_info, dict):
            raise ValueError("TradingAnalyzer requires a valid market_info dictionary.")
        if symbol_state is None or not isinstance(symbol_state, dict):
             raise ValueError("TradingAnalyzer requires a valid symbol_state dictionary.")

        self.df = df.copy()  # Work on a copy to avoid modifying the original DataFrame passed in
        self.logger = logger
        self.config = config
        self.market_info = market_info  # Keep reference
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.symbol_state = symbol_state  # Store reference to the mutable state dict

        # Use Decimal for internal indicator storage where precision matters (e.g., price levels)
        self.indicator_values: dict[str, Decimal | None] = {}  # Stores latest indicator Decimal/float values
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: dict[str, Decimal] = {}  # Stores calculated fib levels (Decimal)
        self.ta_strategy: ta.Strategy | None = None  # Store the TA strategy object
        self.ta_column_map: dict[str, str] = {}  # Map generic name (e.g., "EMA_short") to actual column name

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Signal scores might be zero.{RESET}")

        # --- Convert DataFrame OHLCV to float for pandas_ta ---
        # pandas_ta generally works best with float64
        try:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 if col in self.df.columns and pd.api.types.is_decimal_dtype(self.df[col]):
                      self.df[col] = self.df[col].apply(lambda x: float(x) if x.is_finite() else np.nan)
                 elif col in self.df.columns:  # Ensure numeric otherwise
                      self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Verify conversion
            # self.logger.debug(f"DataFrame dtypes after float conversion: {self.df.dtypes}")

        except Exception as e:
             logger.error(f"Error converting DataFrame columns to float for {self.symbol}: {e}", exc_info=True)
             # Decide how to handle: raise error, or try to continue? For now, continue.

        # --- Initialize and Calculate ---
        self._define_ta_strategy()
        self._calculate_all_indicators()
        self._update_latest_indicator_values()  # Populates self.indicator_values with Decimals
        self.calculate_fibonacci_levels()  # Populates self.fib_levels_data with Decimals

    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even status from the shared symbol state."""
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool) -> None:
        """Sets the break-even status in the shared symbol state."""
        if self.symbol_state.get('break_even_triggered') != value:
            self.symbol_state['break_even_triggered'] = value
            self.logger.info(f"Break-even status for {self.symbol} set to: {value}")

    def _define_ta_strategy(self) -> None:
        """Defines the pandas_ta Strategy based on config."""
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})

        # Helper to safely get numeric config values or use defaults
        def get_num_param(key: str, default: int | float) -> int | float:
            val = cfg.get(key, default)
            try:
                if isinstance(default, int): return int(val)
                if isinstance(default, float): return float(val)
                return val  # Should not happen with int/float defaults
            except (ValueError, TypeError): return default

        # Get periods/windows/settings
        atr_p = get_num_param("atr_period", DEFAULT_ATR_PERIOD)
        ema_s = get_num_param("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
        ema_l = get_num_param("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
        rsi_p = get_num_param("rsi_period", DEFAULT_RSI_WINDOW)
        bb_p = get_num_param("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        bb_std = get_num_param("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        cci_w = get_num_param("cci_window", DEFAULT_CCI_WINDOW)
        wr_w = get_num_param("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        mfi_w = get_num_param("mfi_window", DEFAULT_MFI_WINDOW)
        stochrsi_w = get_num_param("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
        stochrsi_rsi_w = get_num_param("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
        stochrsi_k = get_num_param("stoch_rsi_k", DEFAULT_K_WINDOW)
        stochrsi_d = get_num_param("stoch_rsi_d", DEFAULT_D_WINDOW)
        psar_af = get_num_param("psar_af", DEFAULT_PSAR_AF)
        psar_max = get_num_param("psar_max_af", DEFAULT_PSAR_MAX_AF)
        sma10_w = get_num_param("sma_10_window", DEFAULT_SMA_10_WINDOW)
        mom_p = get_num_param("momentum_period", DEFAULT_MOMENTUM_PERIOD)
        vol_ma_p = get_num_param("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        ta_list = []
        self.ta_column_map = {}  # Reset map

        # Always add ATR (needed for risk management)
        ta_list.append({"kind": "atr", "length": atr_p})
        self.ta_column_map["ATR"] = f"ATRr_{atr_p}"  # pandas_ta uses 'r' suffix for RMA smoothed ATR

        # Add indicators based on config flags
        if indi_cfg.get("ema_alignment") or cfg.get("enable_ma_cross_exit"):
            if ema_s > 0:
                 ta_list.append({"kind": "ema", "length": ema_s, "col_names": (f"EMA_{ema_s}",)})
                 self.ta_column_map["EMA_Short"] = f"EMA_{ema_s}"
            if ema_l > 0:
                 ta_list.append({"kind": "ema", "length": ema_l, "col_names": (f"EMA_{ema_l}",)})
                 self.ta_column_map["EMA_Long"] = f"EMA_{ema_l}"
        if indi_cfg.get("momentum") and mom_p > 0:
            ta_list.append({"kind": "mom", "length": mom_p, "col_names": (f"MOM_{mom_p}",)})
            self.ta_column_map["Momentum"] = f"MOM_{mom_p}"
        if indi_cfg.get("volume_confirmation") and vol_ma_p > 0:
            # Calculate SMA of volume, use specific column name via col_names
            ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_p, "col_names": (f"VOL_SMA_{vol_ma_p}",)})
            self.ta_column_map["Volume_MA"] = f"VOL_SMA_{vol_ma_p}"
        if indi_cfg.get("stoch_rsi") and stochrsi_w > 0 and stochrsi_rsi_w > 0 and stochrsi_k > 0 and stochrsi_d > 0:
            k_col = f"STOCHRSIk_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            d_col = f"STOCHRSId_{stochrsi_w}_{stochrsi_rsi_w}_{stochrsi_k}_{stochrsi_d}"
            ta_list.append({"kind": "stochrsi", "length": stochrsi_w, "rsi_length": stochrsi_rsi_w, "k": stochrsi_k, "d": stochrsi_d, "col_names": (k_col, d_col)})
            self.ta_column_map["StochRSI_K"] = k_col
            self.ta_column_map["StochRSI_D"] = d_col
        if indi_cfg.get("rsi") and rsi_p > 0:
            ta_list.append({"kind": "rsi", "length": rsi_p, "col_names": (f"RSI_{rsi_p}",)})
            self.ta_column_map["RSI"] = f"RSI_{rsi_p}"
        if indi_cfg.get("bollinger_bands") and bb_p > 0:
            # Define column names explicitly
            bbl_col = f"BBL_{bb_p}_{bb_std:.1f}"
            bbm_col = f"BBM_{bb_p}_{bb_std:.1f}"
            bbu_col = f"BBU_{bb_p}_{bb_std:.1f}"
            # bba_col = f"BBA_{bb_p}_{bb_std:.1f}" # Bandwidth
            # bbp_col = f"BBP_{bb_p}_{bb_std:.1f}" # Percent
            ta_list.append({"kind": "bbands", "length": bb_p, "std": bb_std, "col_names": (bbl_col, bbm_col, bbu_col, f"BBB_{bb_p}_{bb_std:.1f}", f"BBP_{bb_p}_{bb_std:.1f}")})  # Include Bandwidth and Percent cols returned by bbands
            self.ta_column_map["BB_Lower"] = bbl_col
            self.ta_column_map["BB_Middle"] = bbm_col
            self.ta_column_map["BB_Upper"] = bbu_col
        if indi_cfg.get("vwap"):
             # VWAP calculation might need typ price (HLC/3), check ta docs
             # Add typical price column if needed by vwap implementation
             if 'typical' not in self.df.columns:
                 self.df['typical'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
             # Use 'typical' as the close price for VWAP? Or let ta handle HLC input?
             # Default ta.vwap uses HLC/3 implicitly if volume present.
             vwap_col = "VWAP_D"  # Default column name, may vary
             ta_list.append({"kind": "vwap", "col_names": (vwap_col,)})  # Assuming daily reset, adjust anchor if needed
             self.ta_column_map["VWAP"] = vwap_col
        if indi_cfg.get("cci") and cci_w > 0:
            cci_col = f"CCI_{cci_w}_0.015"  # Default suffix added by ta
            ta_list.append({"kind": "cci", "length": cci_w, "col_names": (cci_col,)})
            self.ta_column_map["CCI"] = cci_col
        if indi_cfg.get("wr") and wr_w > 0:
            wr_col = f"WILLR_{wr_w}"
            ta_list.append({"kind": "willr", "length": wr_w, "col_names": (wr_col,)})
            self.ta_column_map["WR"] = wr_col
        if indi_cfg.get("psar"):
            # Format AF/MAX consistently for column names
            psar_af_str = f"{psar_af}".rstrip('0').rstrip('.') if '.' in f"{psar_af}" else f"{psar_af}"
            psar_max_str = f"{psar_max}".rstrip('0').rstrip('.') if '.' in f"{psar_max}" else f"{psar_max}"
            l_col = f"PSARl_{psar_af_str}_{psar_max_str}"
            s_col = f"PSARs_{psar_af_str}_{psar_max_str}"
            af_col = f"PSARaf_{psar_af_str}_{psar_max_str}"
            r_col = f"PSARr_{psar_af_str}_{psar_max_str}"
            ta_list.append({"kind": "psar", "af": psar_af, "max_af": psar_max, "col_names": (l_col, s_col, af_col, r_col)})
            self.ta_column_map["PSAR_Long"] = l_col
            self.ta_column_map["PSAR_Short"] = s_col
            self.ta_column_map["PSAR_AF"] = af_col
            self.ta_column_map["PSAR_Reversal"] = r_col
        if indi_cfg.get("sma_10") and sma10_w > 0:
            ta_list.append({"kind": "sma", "length": sma10_w, "col_names": (f"SMA_{sma10_w}",)})
            self.ta_column_map["SMA10"] = f"SMA_{sma10_w}"
        if indi_cfg.get("mfi") and mfi_w > 0:
            # MFI needs typical price and volume
            if 'typical' not in self.df.columns:
                self.df['typical'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
            ta_list.append({"kind": "mfi", "length": mfi_w, "col_names": (f"MFI_{mfi_w}",)})
            self.ta_column_map["MFI"] = f"MFI_{mfi_w}"

        if not ta_list:
            self.logger.warning(f"No indicators enabled or configured correctly for {self.symbol}. Analysis will be limited.")
            return  # No strategy to define

        self.ta_strategy = ta.Strategy(
            name="EnhancedMultiIndicator",
            description="Calculates multiple TA indicators based on config",
            ta=ta_list
        )
        self.logger.debug(f"Defined TA Strategy for {self.symbol} with {len(ta_list)} indicators.")
        # self.logger.debug(f"TA Column Map: {self.ta_column_map}")
        # self.logger.debug(f"Strategy TA List: {self.ta_strategy.ta}")

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using the defined pandas_ta strategy."""
        if self.df.empty:
            self.logger.warning(f"DataFrame empty, cannot calculate indicators for {self.symbol}.")
            return
        if not self.ta_strategy:
            self.logger.warning(f"TA Strategy not defined or empty for {self.symbol}. Skipping indicator calculation.")
            return

        # Check for sufficient data length based on strategy's requirements
        min_required_data = self.ta_strategy.required if hasattr(self.ta_strategy, 'required') else 50  # Estimate if required not available
        buffer = 20  # Add buffer for stability
        if len(self.df) < min_required_data + buffer:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators reliably (min recommended: {min_required_data + buffer}). Results may be inaccurate.{RESET}")

        try:
            # Apply the strategy to the DataFrame (modifies it in place)
            self.logger.debug(f"Running pandas_ta strategy calculation for {self.symbol}...")
            self.df.ta.strategy(self.ta_strategy, timed=False)  # Set timed=True for performance debugging
            self.logger.debug(f"Finished indicator calculations for {self.symbol}.")
            # Log columns added by the strategy for verification
            # new_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'typical']]
            # self.logger.debug(f"Columns after TA calculation: {self.df.columns.tolist()}")

        except AttributeError as ae:
             if "'Decimal' object has no attribute" in str(ae):
                 self.logger.error(f"{NEON_RED}Pandas TA Error for {self.symbol}: Calculation failed, likely due to Decimal type input. Ensure DataFrame uses float64. Error: {ae}{RESET}", exc_info=False)
             else:
                 self.logger.error(f"{NEON_RED}Pandas TA attribute error for {self.symbol}: {ae}. Is pandas_ta installed correctly?{RESET}", exc_info=True)
                 # traceback.print_exc()
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta strategy for {self.symbol}: {e}{RESET}", exc_info=True)
            # traceback.print_exc()
            # Invalidate column map if calculation fails? Or just let _update_latest handle missing cols.

    def _update_latest_indicator_values(self) -> None:
        """Updates the indicator_values dict with the latest Decimal values from self.df."""
        self.indicator_values = {}  # Reset values
        if self.df.empty or len(self.df) == 0:
            self.logger.warning(f"Cannot update latest values: DataFrame empty/short for {self.symbol}.")
            return

        try:
            # Use .iloc[-1] to get the last row (Series)
            latest_series = self.df.iloc[-1]

            if latest_series.isnull().all():
                self.logger.warning(f"Cannot update latest values: Last row is all NaNs for {self.symbol}.")
                return

            # Helper to convert to Decimal, handling NaN/None/Inf
            def to_decimal(value: Any) -> Decimal | None:
                if pd.isna(value) or value is None: return None
                try:
                     dec_val = Decimal(str(value))
                     return dec_val if dec_val.is_finite() else None  # Return None for Inf/-Inf
                except (InvalidOperation, ValueError, TypeError): return None

            # Populate from calculated indicators using the map
            for generic_name, actual_col_name in self.ta_column_map.items():
                if actual_col_name and actual_col_name in latest_series.index:
                    value = latest_series[actual_col_name]
                    self.indicator_values[generic_name] = to_decimal(value)
                else:
                    # self.logger.debug(f"Column '{actual_col_name}' for indicator '{generic_name}' not found in DataFrame.")
                    self.indicator_values[generic_name] = None  # Store None if column missing

            # Add essential price/volume data as Decimal (use capitalized keys for consistency)
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                if base_col in latest_series.index:
                     value = latest_series[base_col]
                     self.indicator_values[base_col.capitalize()] = to_decimal(value)
                else:
                     self.indicator_values[base_col.capitalize()] = None

            # Log non-None values for debugging
            valid_values_str = {}
            for k, v in self.indicator_values.items():
                 if v is not None:
                     try: valid_values_str[k] = f"{v:.5f}"  # Format Decimal for logging
                     except: valid_values_str[k] = str(v)  # Fallback

            self.logger.debug(f"Latest indicator Decimal values updated for {self.symbol}: {valid_values_str}")

        except IndexError:
             self.logger.error(f"Error updating latest indicator values: DataFrame index out of bounds (empty?) for {self.symbol}.")
        except KeyError as ke:
            self.logger.error(f"Error updating latest indicator values: Column key error '{ke}' for {self.symbol}. TA calculation might have failed.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
            # traceback.print_exc()
            self.indicator_values = {}  # Reset on error

    # --- Precision and Market Info Helpers (using market_info) ---
    def get_min_tick_size(self) -> Decimal | None:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        tick = self.market_info.get('min_tick_size')
        if tick is None or tick <= 0:
             self.logger.warning(f"Invalid min_tick_size ({tick}) retrieved for {self.symbol}")
             return None
        return tick

    def get_price_precision_digits(self) -> int:
        """Gets price precision (number of decimal places) from market info."""
        digits = self.market_info.get('price_precision_digits')
        return digits if digits is not None and digits >= 0 else 8  # Default fallback 8

    def get_amount_precision_digits(self) -> int:
        """Gets amount precision (number of decimal places) from market info."""
        digits = self.market_info.get('amount_precision_digits')
        return digits if digits is not None and digits >= 0 else 8  # Default fallback 8

    def quantize_price(self, price: Decimal | float | str, rounding=ROUND_DOWN) -> Decimal | None:
        """Quantizes a price DOWN to the market's tick size."""
        min_tick = self.get_min_tick_size()
        if min_tick is None:
            self.logger.error(f"Cannot quantize price for {self.symbol}: Missing min_tick_size.")
            return None
        try:
            price_decimal = Decimal(str(price))
            if not price_decimal.is_finite(): return None  # Handle NaN/Inf input

            # Quantize to the nearest tick using the specified rounding mode
            quantized = (price_decimal / min_tick).quantize(Decimal('0'), rounding=rounding) * min_tick
            # Optional: Format to ensure correct number of decimal places as per tick size
            # digits = abs(min_tick.normalize().as_tuple().exponent)
            # return Decimal(f"{quantized:.{digits}f}")
            return quantized
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing price '{price}' for {self.symbol}: {e}")
            return None

    def quantize_amount(self, amount: Decimal | float | str, rounding=ROUND_DOWN) -> Decimal | None:
        """Quantizes an amount to the market's amount precision (step size) using specified rounding."""
        amount_digits = self.get_amount_precision_digits()
        if amount_digits is None:  # Should use the fallback 8 from the getter
            self.logger.error(f"Cannot quantize amount for {self.symbol}: Invalid amount_precision_digits.")
            return None
        try:
            amount_decimal = Decimal(str(amount))
            if not amount_decimal.is_finite(): return None  # Handle NaN/Inf input

            # Amount step size is 10^(-digits)
            step_size = Decimal('1') / (Decimal('10') ** amount_digits)
            quantized = (amount_decimal / step_size).quantize(Decimal('0'), rounding=rounding) * step_size
            # Format to the correct number of decimal places to avoid issues with trailing zeros beyond precision
            # This ensures the string representation matches the required precision
            return Decimal(f"{quantized:.{amount_digits}f}")
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error quantizing amount '{amount}' for {self.symbol}: {e}")
            return None

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a window using Decimal and quantization."""
        self.fib_levels_data = {}  # Reset
        window = window or int(self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW))
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fib window ({window}) on {self.symbol}.")
            return {}

        df_slice = self.df.tail(window)
        try:
            # Use Decimal directly if df columns are already Decimal, otherwise convert from float
            if pd.api.types.is_decimal_dtype(df_slice["high"]):
                high_raw = df_slice["high"].dropna().max(); low_raw = df_slice["low"].dropna().min()
            else:  # Assume float if not Decimal
                 high_raw = Decimal(str(df_slice["high"].dropna().max()))
                 low_raw = Decimal(str(df_slice["low"].dropna().min()))

            if not high_raw.is_finite() or not low_raw.is_finite():
                self.logger.warning(f"No valid high/low in last {window} periods for Fib on {self.symbol}. High={high_raw}, Low={low_raw}"); return {}

            high = high_raw; low = low_raw; diff = high - low
            levels = {}; min_tick = self.get_min_tick_size()

            if diff >= Decimal('0') and min_tick is not None:
                for level_pct_float in FIB_LEVELS:
                    level_pct = Decimal(str(level_pct_float))
                    level_price_raw = high - (diff * level_pct)
                    # Quantize DOWN to nearest tick
                    level_price = self.quantize_price(level_price_raw, rounding=ROUND_DOWN)  # Quantize Fib levels down
                    if level_price is not None:
                         levels[f"Fib_{level_pct * 100:.1f}%"] = level_price
                    else:
                         self.logger.warning(f"Failed to quantize Fib level {level_pct * 100:.1f}% ({level_price_raw}) for {self.symbol}")
            elif min_tick is None:
                 self.logger.warning(f"Invalid min_tick_size for Fib quantization on {self.symbol}. Using raw values.")
                 for level_pct_float in FIB_LEVELS:
                     level_pct = Decimal(str(level_pct_float))
                     level_price_raw = high - (diff * level_pct)
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_raw  # Store raw Decimal
            else:  # Should not happen if diff < 0, but handle defensively
                 self.logger.warning(f"Invalid range (high < low?) for Fib on {self.symbol}. High={high}, Low={low}")
                 return {}

            self.fib_levels_data = levels
            price_prec = self.get_price_precision_digits()
            {k: f"{v:.{price_prec}f}" for k, v in levels.items()}
            # self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {log_levels}")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            # traceback.print_exc()
            self.fib_levels_data = {}; return {}

    # --- Indicator Check Methods (Return float score -1.0 to 1.0, or None if value missing) ---
    # These methods now use self.indicator_values[GENERIC_NAME] which should contain Decimals
    def _get_indicator_float(self, name: str) -> float | None:
        """Safely gets indicator value as float."""
        val = self.indicator_values.get(name)
        if val is None or not val.is_finite(): return None
        try: return float(val)
        except: return None

    def _check_ema_alignment(self) -> float | None:
        ema_short = self._get_indicator_float("EMA_Short")
        ema_long = self._get_indicator_float("EMA_Long")
        if ema_short is None or ema_long is None: return None
        if ema_short > ema_long: return 1.0
        if ema_short < ema_long: return -1.0
        return 0.0

    def _check_momentum(self) -> float | None:
        mom = self._get_indicator_float("Momentum")
        if mom is None: return None
        # Simple scaling (adjust scale_factor based on typical range for asset/TF)
        # Example: if typical MOM is +/- 5, scale_factor = 0.2. If +/- 0.5, scale_factor = 2.0
        scale_factor = 0.1  # Example scale factor - NEEDS TUNING
        return max(-1.0, min(1.0, mom * scale_factor))

    def _check_volume_confirmation(self) -> float | None:
        vol = self._get_indicator_float("Volume")
        vol_ma = self._get_indicator_float("Volume_MA")
        if vol is None or vol_ma is None or vol_ma <= 0: return None
        try: mult = float(self.config.get("volume_confirmation_multiplier", 1.5))
        except: mult = 1.5
        if vol > vol_ma * mult: return 0.7  # High volume confirmation (positive signal only)
        # Optionally add negative signal for low volume:
        # if vol < vol_ma / mult: return -0.4 # Low volume lack of confirmation
        return 0.0  # Neutral

    def _check_stoch_rsi(self) -> float | None:
        k = self._get_indicator_float("StochRSI_K"); d = self._get_indicator_float("StochRSI_D")
        if k is None or d is None: return None
        try:
            oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
            overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        except: oversold, overbought = 25.0, 75.0
        score = 0.0
        # Prioritize crossing signals
        # Check previous values if available (more robust crossing check) - requires storing previous state or looking at df[-2]
        # Simple check based on current values:
        if k < oversold and d < oversold:  # Both oversold
             score = 0.8 if k > d else 0.6  # Stronger signal if K crossed D upwards recently
        elif k > overbought and d > overbought:  # Both overbought
             score = -0.8 if k < d else -0.6  # Stronger signal if K crossed D downwards recently
        # Zone signals
        elif k < oversold: score = 0.5  # In oversold zone
        elif k > overbought: score = -0.5  # In overbought zone
        # Trend bias based on K vs D
        elif k > d: score = 0.2  # K over D bullish bias
        elif k < d: score = -0.2  # K under D bearish bias
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float | None:
        rsi = self._get_indicator_float("RSI")
        if rsi is None: return None
        # Using more granular scoring based on zones
        if rsi <= 20: return 1.0   # Very oversold
        if rsi <= 30: return 0.7   # Oversold
        if rsi >= 80: return -1.0  # Very overbought
        if rsi >= 70: return -0.7  # Overbought
        # Linear scale between 30 and 70
        if 30 < rsi < 70: return 1.0 - (rsi - 30.0) * (2.0 / 40.0)
        return 0.0  # Between 30-70 but exactly on boundary (shouldn't happen often)

    def _check_cci(self) -> float | None:
        cci = self._get_indicator_float("CCI")
        if cci is None: return None
        # Using standard CCI levels
        if cci <= -200: return 1.0  # Extreme oversold
        if cci <= -100: return 0.7  # Oversold entry/confirmation
        if cci >= 200: return -1.0  # Extreme overbought
        if cci >= 100: return -0.7  # Overbought entry/confirmation
        # Scale between -100 and 100 (less signal strength)
        if -100 < cci < 100: return -(cci / 100.0) * 0.3  # Weaker signal towards zero crossing
        return 0.0

    def _check_wr(self) -> float | None:  # Williams %R
        wr = self._get_indicator_float("WR")  # Note: WR is typically negative, ranging -100 to 0
        if wr is None: return None
        if wr <= -90: return 1.0   # Very oversold
        if wr <= -80: return 0.7   # Oversold
        if wr >= -10: return -1.0  # Very overbought
        if wr >= -20: return -0.7  # Overbought
        # Linear scale between -80 and -20
        if -80 < wr < -20: return 1.0 - (wr - (-80.0)) * (2.0 / 60.0)
        return 0.0

    def _check_psar(self) -> float | None:
        psar_l_val = self.indicator_values.get("PSAR_Long")  # Decimal or None
        psar_s_val = self.indicator_values.get("PSAR_Short")  # Decimal or None
        # Check if PSAR value exists (is not None and is finite)
        is_long_active = psar_l_val is not None and psar_l_val.is_finite()
        is_short_active = psar_s_val is not None and psar_s_val.is_finite()

        # PSAR Long is active when PSAR dot is below price (uptrend)
        if is_long_active and not is_short_active: return 1.0
        # PSAR Short is active when PSAR dot is above price (downtrend)
        if not is_long_active and is_short_active: return -1.0

        # Check for reversal signal (PSARr column)
        # reversal = self._get_indicator_float("PSAR_Reversal")
        # if reversal is not None and reversal != 0:
        #     # logger.debug(f"PSAR reversal detected: {reversal}")
        #     return 1.0 if reversal > 0 else -1.0 # Use reversal signal directly

        return 0.0  # Neutral/Ambiguous (e.g., first few bars, or if both L/S somehow appear)

    def _check_sma10(self) -> float | None:
        sma = self._get_indicator_float("SMA10")
        close = self._get_indicator_float("Close")
        if sma is None or close is None: return None
        if close > sma: return 0.5  # Price above SMA10 -> Weak Buy
        if close < sma: return -0.5  # Price below SMA10 -> Weak Sell
        return 0.0

    def _check_vwap(self) -> float | None:
        vwap = self._get_indicator_float("VWAP")
        close = self._get_indicator_float("Close")
        if vwap is None or close is None: return None
        if close > vwap: return 0.6  # Price above VWAP -> Moderate Buy
        if close < vwap: return -0.6  # Price below VWAP -> Moderate Sell
        return 0.0

    def _check_mfi(self) -> float | None:
        mfi = self._get_indicator_float("MFI")
        if mfi is None: return None
        # Similar zones to RSI
        if mfi <= 15: return 1.0   # Very oversold
        if mfi <= 25: return 0.7   # Oversold
        if mfi >= 85: return -1.0  # Very overbought
        if mfi >= 75: return -0.7  # Overbought
        # Linear scale between 25 and 75
        if 25 < mfi < 75: return 1.0 - (mfi - 25.0) * (2.0 / 50.0)
        return 0.0

    def _check_bollinger_bands(self) -> float | None:
        bbl = self._get_indicator_float("BB_Lower")
        bbu = self._get_indicator_float("BB_Upper")
        close = self._get_indicator_float("Close")
        if bbl is None or bbu is None or close is None: return None

        band_range = bbu - bbl
        if band_range <= 0: return 0.0  # Avoid division by zero if bands invalid

        # Score based on proximity to bands (normalized)
        # Value close to 1 means close to lower band, close to -1 means close to upper band
        # Value around 0 means near middle band
        position_in_band = (close - bbl) / band_range  # Range 0 to 1 usually
        score = 1.0 - 2.0 * position_in_band  # Rescale to -1 to 1

        # Add stronger signal if price touches or crosses bands
        if close <= bbl: score = 1.0  # Strong buy signal (touch/cross lower)
        if close >= bbu: score = -1.0  # Strong sell signal (touch/cross upper)

        return max(-1.0, min(1.0, score))  # Clamp score

    def _check_orderbook(self, orderbook_data: dict | None) -> float | None:
        """Analyzes Order Book Imbalance (OBI)."""
        if not orderbook_data: return None
        try:
            bids = orderbook_data.get('bids', []); asks = orderbook_data.get('asks', [])
            if not bids or not asks: return 0.0  # Neutral if one side is empty

            # Consider top N levels or volume within X% of mid-price
            levels_to_check = min(len(bids), len(asks), int(self.config.get("orderbook_limit", 10)))
            if levels_to_check <= 0: return 0.0

            # Calculate Weighted Mid Price
            # best_bid_price = Decimal(str(bids[0][0])); best_ask_price = Decimal(str(asks[0][0]))
            # best_bid_vol = Decimal(str(bids[0][1])); best_ask_vol = Decimal(str(asks[0][1]))
            # mid_price = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol) if (best_bid_vol + best_ask_vol) > 0 else (best_bid_price + best_ask_price) / 2

            # Sum volume within N levels
            bid_vol = sum(Decimal(str(b[1])) for b in bids[:levels_to_check])
            ask_vol = sum(Decimal(str(a[1])) for a in asks[:levels_to_check])
            total_vol = bid_vol + ask_vol

            if total_vol <= 0: return 0.0  # Neutral if no volume in checked levels

            obi = (bid_vol - ask_vol) / total_vol  # OBI range: -1 to +1
            # self.logger.debug(f"OBI ({levels_to_check} levels): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, OBI={obi:.4f}")

            # Return OBI directly as score, clamped between -1 and 1
            return float(max(Decimal("-1.0"), min(Decimal("1.0"), obi)))

        except (InvalidOperation, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error calculating OBI for {self.symbol}: {e}"); return None
        except Exception as e: self.logger.error(f"Unexpected OBI error: {e}"); return None

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(self, current_price_dec: Decimal, orderbook_data: dict | None) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Reset signals, default HOLD
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0; nan_indicator_count = 0
        debug_scores = {}

        if not self.indicator_values or current_price_dec is None or not current_price_dec.is_finite() or current_price_dec <= 0:
            self.logger.warning(f"Cannot generate signal for {self.symbol}: Invalid inputs (indicators empty or price invalid: {current_price_dec}).")
            return "HOLD"

        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name)
        if not active_weights:
            self.logger.error(f"Weight set '{self.active_weight_set_name}' missing/empty for {self.symbol}. HOLDING.")
            return "HOLD"

        # Map indicator keys from config to their check methods
        indicator_check_methods = {
            "ema_alignment": self._check_ema_alignment,
            "momentum": self._check_momentum,
            "volume_confirmation": self._check_volume_confirmation,
            "stoch_rsi": self._check_stoch_rsi,
            "rsi": self._check_rsi,
            "bollinger_bands": self._check_bollinger_bands,
            "vwap": self._check_vwap,
            "cci": self._check_cci,
            "wr": self._check_wr,
            "psar": self._check_psar,
            "sma_10": self._check_sma10,
            "mfi": self._check_mfi,
            "orderbook": lambda: self._check_orderbook(orderbook_data),  # Lambda for extra arg
        }

        # --- Calculate Weighted Score ---
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled:
                debug_scores[indicator_key] = "Disabled"
                continue

            weight_str = active_weights.get(indicator_key)
            if weight_str is None:
                debug_scores[indicator_key] = "No Weight"
                continue  # Skip if no weight defined for this enabled indicator

            try:
                weight = Decimal(str(weight_str))
                if weight == 0:
                     debug_scores[indicator_key] = "Weight=0"
                     continue  # Skip zero weights silently
            except (InvalidOperation, ValueError, TypeError):
                self.logger.warning(f"Invalid weight '{weight_str}' for indicator '{indicator_key}' in set '{self.active_weight_set_name}'. Skipping.")
                debug_scores[indicator_key] = f"Invalid Wt({weight_str})"
                continue

            check_method = indicator_check_methods.get(indicator_key)
            if check_method and callable(check_method):
                indicator_score_float = None  # Use None to indicate missing value
                try:
                    indicator_score_float = check_method()
                except Exception as e:
                    self.logger.error(f"Error in check method for {indicator_key} on {self.symbol}: {e}", exc_info=True)
                    # traceback.print_exc()

                if indicator_score_float is not None and math.isfinite(indicator_score_float):
                    try:
                        # Clamp score between -1.0 and 1.0 before applying weight
                        clamped_score = max(-1.0, min(1.0, indicator_score_float))
                        indicator_score_decimal = Decimal(str(clamped_score))
                        weighted_score = indicator_score_decimal * weight
                        final_signal_score += weighted_score
                        total_weight_applied += abs(weight)  # Use absolute weight for normalization denominator
                        active_indicator_count += 1
                        debug_scores[indicator_key] = f"{indicator_score_float:.2f}(x{weight:.2f})={weighted_score:.3f}"
                    except (InvalidOperation, ValueError, TypeError) as calc_err:
                        self.logger.error(f"Error processing score for {indicator_key}: {calc_err}")
                        debug_scores[indicator_key] = "Calc Err"
                        nan_indicator_count += 1
                else:
                    debug_scores[indicator_key] = "NaN/None"
                    nan_indicator_count += 1
            elif indicator_key in active_weights:  # Only warn if weight exists but method doesn't
                self.logger.warning(f"Check method not found or implemented for enabled indicator with weight: {indicator_key}")
                debug_scores[indicator_key] = "No Method"

        # --- Determine Final Signal ---
        final_signal = "HOLD"
        normalized_score = Decimal("0.0")  # Score normalized by sum of absolute weights

        if total_weight_applied > 0:
            normalized_score = (final_signal_score / total_weight_applied).quantize(Decimal("0.0001"))
        elif active_indicator_count > 0:
             self.logger.warning(f"No non-zero weights applied for {active_indicator_count} active indicators on {self.symbol}. Final score is 0. HOLDING.")
        # else: # No active indicators case is handled by score being 0

        # Use the appropriate threshold based on the active weight set
        threshold_key = "scalping_signal_threshold" if self.active_weight_set_name == "scalping" else "signal_score_threshold"
        default_threshold = 2.5 if self.active_weight_set_name == "scalping" else 1.5
        try:
            # Threshold applies to the *non-normalized* score (sum of weighted scores)
            threshold = Decimal(str(self.config.get(threshold_key, default_threshold)))
        except (InvalidOperation, ValueError, TypeError):
            threshold = Decimal(str(default_threshold))
            self.logger.warning(f"Invalid threshold value for {threshold_key}. Using default: {threshold}")

        # Determine signal based on threshold
        if final_signal_score >= threshold:
            final_signal = "BUY"
        elif final_signal_score <= -threshold:
            final_signal = "SELL"

        # --- Log Summary ---
        price_prec = self.get_price_precision_digits()
        ", ".join([f"{k}: {v}" for k, v in debug_scores.items() if v not in ["Disabled", "No Weight", "Weight=0"]])  # Show only relevant scores
        log_msg = (f"Signal Calc ({self.symbol} @ {current_price_dec:.{price_prec}f}): "
                    f"Set='{self.active_weight_set_name}', "
                    f"Indis(Actv/NaN): {active_indicator_count}/{nan_indicator_count}, "
                    f"TotalAbsWt: {total_weight_applied:.3f}, "
                    f"RawScore: {final_signal_score:.4f}, NormScore: {normalized_score:.4f}, "
                    f"Threshold: +/-{threshold:.3f} -> "
                    f"Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}")
        self.logger.info(log_msg)
        if nan_indicator_count > 0 or active_indicator_count == 0:
             self.logger.debug(f"  Detailed Scores: {debug_scores}")  # Log all scores if issues

        # Update internal signal state
        if final_signal in self.signals:
            self.signals[final_signal] = 1
            if final_signal != "HOLD": self.signals["HOLD"] = 0

        return final_signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential TP and initial SL based on entry, ATR, config, quantized to tick size.
        Returns (quantized_entry, quantized_tp, quantized_sl). Entry price is also quantized.
        """
        quantized_entry, take_profit, stop_loss = None, None, None
        if signal not in ["BUY", "SELL"] or entry_price is None or not entry_price.is_finite() or entry_price <= 0:
            self.logger.error(f"Invalid input for TP/SL calc: Signal={signal}, Entry={entry_price}")
            return None, None, None

        # Quantize entry price first (use rounding appropriate for side?) - Use default rounding for entry itself
        quantized_entry = self.quantize_price(entry_price, rounding=ROUND_DOWN)  # Quantize entry down generally
        if quantized_entry is None:
             self.logger.error(f"Failed to quantize entry price {entry_price} for TP/SL calc.")
             return None, None, None

        atr_val = self.indicator_values.get("ATR")
        if atr_val is None or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"Cannot calculate TP/SL for {self.symbol}: Invalid ATR ({atr_val}).")
            return quantized_entry, None, None  # Return quantized entry, but no SL/TP

        try:
            atr = atr_val  # Already Decimal
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
            min_tick = self.get_min_tick_size()  # Already checked in quantize_price, but check again

            if min_tick is None:  # Should not happen if entry quantization worked
                self.logger.error(f"Cannot calculate TP/SL for {self.symbol}: Min tick size missing after entry quantization.")
                return quantized_entry, None, None

            tp_offset = atr * tp_mult; sl_offset = atr * sl_mult

            # Calculate raw TP/SL based on *quantized* entry
            if signal == "BUY":
                tp_raw = quantized_entry + tp_offset; sl_raw = quantized_entry - sl_offset
                # Quantize TP UP (away from entry), SL DOWN (away from entry)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_UP)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_DOWN)
            elif signal == "SELL":
                tp_raw = quantized_entry - tp_offset; sl_raw = quantized_entry + sl_offset
                # Quantize TP DOWN (away from entry), SL UP (away from entry)
                take_profit = self.quantize_price(tp_raw, rounding=ROUND_DOWN)
                stop_loss = self.quantize_price(sl_raw, rounding=ROUND_UP)

            # --- Validation ---
            if stop_loss is not None:
                 # Ensure SL is strictly beyond entry by >= 1 tick
                 if signal == "BUY" and stop_loss >= quantized_entry:
                     stop_loss = self.quantize_price(quantized_entry - min_tick, rounding=ROUND_DOWN)
                     self.logger.debug(f"Adjusted BUY SL below entry: {stop_loss}")
                 elif signal == "SELL" and stop_loss <= quantized_entry:
                     stop_loss = self.quantize_price(quantized_entry + min_tick, rounding=ROUND_UP)
                     self.logger.debug(f"Adjusted SELL SL above entry: {stop_loss}")
                 # Ensure SL is positive
                 if stop_loss <= 0:
                     self.logger.error(f"Calculated SL is zero/negative ({stop_loss}). Setting SL to None."); stop_loss = None

            if take_profit is not None:
                 # Ensure TP is strictly beyond entry by >= 1 tick
                 if signal == "BUY" and take_profit <= quantized_entry:
                     take_profit = self.quantize_price(quantized_entry + min_tick, rounding=ROUND_UP)
                     self.logger.debug(f"Adjusted BUY TP above entry: {take_profit}")
                 elif signal == "SELL" and take_profit >= quantized_entry:
                     take_profit = self.quantize_price(quantized_entry - min_tick, rounding=ROUND_DOWN)
                     self.logger.debug(f"Adjusted SELL TP below entry: {take_profit}")
                 # Ensure TP is positive
                 if take_profit <= 0:
                     self.logger.error(f"Calculated TP is zero/negative ({take_profit}). Setting TP to None."); take_profit = None

            # --- Log Results ---
            prec = self.get_price_precision_digits()
            tp_log = f"{take_profit:.{prec}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{prec}f}" if stop_loss else 'N/A'
            entry_log = f"{quantized_entry:.{prec}f}"
            atr_log = f"{atr:.{prec + 1}f}"  # Show ATR with slightly more precision

            self.logger.info(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log} (ATR={atr_log})")
            return quantized_entry, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            # traceback.print_exc()
            return quantized_entry, None, None  # Return quantized entry on error, but no TP/SL


# --- Position Sizing ---
def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,  # Expect float 0.01 for 1%
    entry_price: Decimal,
    stop_loss_price: Decimal,
    market_info: dict,
    leverage: int,  # Pass configured leverage
    logger: logging.Logger
) -> Decimal | None:
    """Calculates position size based on balance, risk %, SL distance, market limits, and leverage."""
    lg = logger
    symbol = market_info.get('symbol', 'N/A')
    contract_size = market_info.get('contract_size', Decimal('1'))
    min_order_amount = market_info.get('min_order_amount')  # Decimal or None
    min_order_cost = market_info.get('min_order_cost')  # Decimal or None
    amount_digits = market_info.get('amount_precision_digits')  # Int or None
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('inverse', False)

    # Validate inputs
    if balance <= 0:
        lg.error(f"Cannot calculate position size for {symbol}: Balance is zero or negative ({balance}).")
        return None
    if entry_price <= 0 or stop_loss_price <= 0:
        lg.error(f"Cannot calculate position size for {symbol}: Entry ({entry_price}) or SL ({stop_loss_price}) is invalid.")
        return None
    if entry_price == stop_loss_price:
        lg.error(f"Cannot calculate position size for {symbol}: Entry and SL prices are identical.")
        return None
    if amount_digits is None:
         lg.error(f"Cannot calculate position size for {symbol}: Amount precision digits not found.")
         return None
    if risk_per_trade <= 0 or risk_per_trade >= 1:
        lg.error(f"Cannot calculate position size: Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
        return None
    if leverage <= 0 and is_contract:
         lg.error(f"Cannot calculate position size for contract {symbol}: Leverage must be positive ({leverage}).")
         return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))  # Risk amount in Quote currency (e.g., USDT)
        sl_distance_points = abs(entry_price - stop_loss_price)

        # Calculate value per point in Quote currency
        # For linear contracts (BASE/USDT): value per point = contract_size (points are USDT)
        # For inverse contracts (BASE/USD settled in BASE): value per point = contract_size / entry_price (approx value of 1 point in BASE, need to convert to QUOTE)
        # For spot: Not applicable directly, size is calculated differently (Amount = RiskAmtQuote / SL_Distance_Quote)

        if is_contract:
            if is_inverse:
                # Risk per contract in BASE currency = sl_distance_points * contract_size / entry_price (approx)
                # More accurately: Risk = contract_size * abs(1/entry - 1/sl)
                risk_per_contract_base = contract_size * abs(Decimal('1') / entry_price - Decimal('1') / stop_loss_price)
                # Convert risk per contract to QUOTE currency
                risk_per_contract_quote = risk_per_contract_base * entry_price  # Approx conversion using entry price
            else:  # Linear contract
                # Risk per contract in QUOTE currency = sl_distance_points * contract_size
                risk_per_contract_quote = sl_distance_points * contract_size

            if risk_per_contract_quote <= 0:
                 lg.error(f"Cannot calculate position size for {symbol}: Risk per contract is zero or negative ({risk_per_contract_quote}).")
                 return None

            # Ideal size in contracts = Total risk amount (Quote) / Risk per contract (Quote)
            size_unquantized = risk_amount_quote / risk_per_contract_quote
        else:  # Spot
             # Ideal size in BASE currency = Risk amount (Quote) / SL distance (Quote)
             if sl_distance_points <= 0:
                 lg.error(f"Cannot calculate position size for spot {symbol}: SL distance is zero or negative.")
                 return None
             size_unquantized = risk_amount_quote / sl_distance_points

        lg.debug(f"Position Size Calc ({symbol}): Balance={balance:.2f}, RiskAmt={risk_amount_quote:.4f}, "
                 f"SLDist={sl_distance_points}, Entry={entry_price}, SL={stop_loss_price}, "
                 f"{'RiskPerContrQuote=' + str(risk_per_contract_quote) if is_contract else ''}, "
                 f"UnquantizedSize={size_unquantized:.8f}")

        # Quantize the size according to market rules (amount precision) - ROUND DOWN
        step_size = Decimal('1') / (Decimal('10') ** amount_digits)
        quantized_size = (size_unquantized / step_size).quantize(Decimal('0'), rounding=ROUND_DOWN) * step_size
        # Format to correct digits just in case quantization resulted in tiny residual
        quantized_size = Decimal(f"{quantized_size:.{amount_digits}f}")

        lg.debug(f"Quantized Size ({symbol}): {quantized_size} (Step: {step_size})")

        # --- Validation against market limits and margin ---
        if quantized_size <= 0:
            lg.warning(f"{NEON_YELLOW}Calculated position size ({quantized_size}) is zero after quantization for {symbol}. Cannot place trade.{RESET}")
            return None

        # Check against minimum order amount
        if min_order_amount is not None and quantized_size < min_order_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {quantized_size} is below minimum order amount {min_order_amount} for {symbol}. Adjust risk or balance.{RESET}")
            return None

        # Estimate order cost/margin required
        order_value = quantized_size * entry_price * (contract_size if is_inverse else Decimal('1'))  # Value in quote currency (approx for inverse)
        margin_required = order_value / Decimal(leverage) if is_contract else order_value  # Margin for contracts, full value for spot

        # Check against minimum order cost (if specified)
        if min_order_cost is not None and order_value < min_order_cost:
            lg.warning(f"{NEON_YELLOW}Estimated order value {order_value:.4f} is below minimum cost {min_order_cost} for {symbol}. Cannot place trade.{RESET}")
            return None

        # Check if required margin exceeds available balance (basic check)
        # Note: This doesn't account for existing positions or maintenance margin. Exchange check is final.
        if margin_required > balance:
             lg.warning(f"{NEON_YELLOW}Estimated margin required ({margin_required:.4f}) exceeds available balance ({balance:.4f}) for {symbol} at {leverage}x leverage. Cannot place trade.{RESET}")
             # Optionally, calculate max possible size based on balance and retry? For now, just fail.
             # max_size_based_on_margin = (balance * Decimal(leverage)) / (entry_price * (contract_size if is_inverse else Decimal('1')))
             # quantized_max_size = ... quantize max_size_based_on_margin ...
             # if quantized_max_size >= min_order_amount: lg.warning("Consider reducing risk % or leverage."); return quantized_max_size
             return None

        lg.info(f"Calculated position size for {symbol}: {quantized_size}")
        return quantized_size

    except (InvalidOperation, ValueError, TypeError, Exception) as e:
        lg.error(f"{NEON_RED}Error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None

# --- CCXT Trading Action Wrappers ---


def fetch_positions_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict) -> dict | None:
    """Fetches open position for a specific symbol using V5 API."""
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol)  # Use the exchange's specific ID

    if not category or category not in ['linear', 'inverse']:  # Positions only for derivatives
        lg.debug(f"Skipping position check for non-derivative symbol {symbol} (Category: {category})")
        return None
    if not exchange.has.get('fetchPositions'):
         lg.error(f"Exchange {exchange.id} does not support fetchPositions(). Cannot check position.")
         return None

    try:
        # Bybit V5 fetch_positions requires category and optionally symbol or settleCoin
        params = {'category': category, 'symbol': market_id}
        lg.debug(f"Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Use safe_ccxt_call
        all_positions = safe_ccxt_call(exchange, 'fetch_positions', lg, symbols=[symbol], params=params)  # Request specific symbol if possible

        if all_positions is None:  # Error handled by safe_ccxt_call
            lg.warning(f"fetch_positions call returned None for {symbol}.")
            return None

        # fetch_positions should return a list, potentially empty
        if not isinstance(all_positions, list):
            lg.error(f"fetch_positions did not return a list for {symbol}. Response: {all_positions}")
            return None

        # Filter for the specific symbol (double check, as API might return others)
        # And ensure the position size is non-zero
        for pos in all_positions:
            if pos.get('symbol') == symbol:
                try:
                     # Check 'contracts' (standard ccxt) or 'size' (Bybit V5 info)
                     pos_size_str = pos.get('contracts', pos.get('info', {}).get('size'))
                     if pos_size_str is None:
                         lg.warning(f"Position data for {symbol} missing size/contracts. Data: {pos}")
                         continue  # Check next entry in list if any

                     pos_size = Decimal(str(pos_size_str))
                     if pos_size != 0:  # Check for non-zero size (can be negative for short)
                          # Get side ('long'/'short')
                          pos_side = pos.get('side')
                          if pos_side is None:  # Infer side from size if missing
                              pos_side = 'long' if pos_size > 0 else 'short' if pos_size < 0 else None

                          if pos_side:
                               pos['side'] = pos_side  # Ensure side is set
                               pos['contracts'] = abs(pos_size)  # Ensure contracts is positive size
                               lg.info(f"Found active {pos_side} position for {symbol}: Size={abs(pos_size)}, Entry={pos.get('entryPrice')}")
                               # Add market info to position dict for convenience
                               pos['market_info'] = market_info
                               return pos
                          else:
                               lg.warning(f"Position found for {symbol} but could not determine side (Size: {pos_size}).")
                               continue

                     else:
                          # Position exists but size is 0 (recently closed or error)
                          lg.debug(f"Position found for {symbol} but size is 0. Treating as no active position.")
                          # return None # Should continue checking list in case of hedge mode oddities

                except (InvalidOperation, ValueError, TypeError, KeyError) as e:
                     lg.error(f"Could not parse position data for {symbol}: {e}. Data: {pos}")
                     # return None # Treat as error / no position

        lg.debug(f"No active non-zero position found for {symbol} after checking {len(all_positions)} results.")
        return None  # No active position found for this specific symbol

    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching or processing positions for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger, market_info: dict) -> bool | None:
    """Sets leverage for a symbol using V5 API."""
    lg = logger
    category = market_info.get('category')
    market_id = market_info.get('id', symbol)

    if not category or category not in ['linear', 'inverse']:
        lg.debug(f"Skipping leverage setting for non-derivative symbol {symbol} (Category: {category})")
        return True  # Not applicable, treat as success
    if not exchange.has.get('setLeverage'):
         lg.error(f"Exchange {exchange.id} does not support setLeverage().")
         return False

    # Check current leverage first (requires fetch_position) - Optional optimization
    # current_pos = fetch_positions_ccxt(exchange, symbol, logger, market_info)
    # current_leverage = int(current_pos.get('leverage', 0)) if current_pos else 0
    # if current_leverage == leverage:
    #     lg.debug(f"Leverage already set to {leverage}x for {symbol}.")
    #     return True

    try:
        # Bybit V5 requires buyLeverage and sellLeverage, set both to the same value
        params = {
            'category': category,
            'symbol': market_id,  # Pass symbol in params for V5
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        }
        lg.info(f"Setting leverage for {symbol} (MarketID: {market_id}) to {leverage}x...")
        lg.debug(f"Leverage Params: {params}")
        # CCXT's set_leverage should handle mapping to the correct V5 endpoint and params
        result = safe_ccxt_call(exchange, 'set_leverage', lg, leverage=leverage, symbol=symbol, params=params)

        # Check result if possible (CCXT might not return detailed info)
        # Bybit V5 POST /v5/position/set-leverage returns {} on success (retCode 0)
        # We rely on safe_ccxt_call not raising an exception
        lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (Result: {result}).{RESET}")
        return True
    except ccxt.ExchangeError as e:
         # Handle specific error code 110043: "Set leverage not modified"
         bybit_code = None
         if hasattr(e, 'args') and len(e.args) > 0:
              try:
                  error_details = str(e.args[0])
                  if "retCode" in error_details:
                       details_dict = json.loads(error_details[error_details.find('{'):error_details.rfind('}') + 1])
                       bybit_code = details_dict.get('retCode')
              except Exception: pass
         if bybit_code == 110043:
              lg.info(f"Leverage already set to {leverage}x for {symbol} (Exchange code 110043).")
              return True
         else:
              lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x: {e} (Code: {bybit_code}){RESET}", exc_info=True)
              # traceback.print_exc()
              return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return False


def create_order_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    order_type: str,  # 'market', 'limit'
    side: str,       # 'buy', 'sell'
    amount: Decimal,
    price: Decimal | None = None,  # Required for limit orders
    params: dict | None = None,  # For extra params like reduceOnly, positionIdx, tpsl
    logger: logging.Logger | None = None,
    market_info: dict | None = None
) -> dict | None:
    """Creates an order using safe_ccxt_call with V5 parameters and Decimal inputs."""
    lg = logger or get_logger('main')  # Use main logger if none provided
    if not market_info:
         lg.error(f"Market info required for create_order_ccxt ({symbol})")
         return None

    category = market_info.get('category')
    market_info.get('id', symbol)
    if not category:
         lg.error(f"Cannot determine category for {symbol}. Cannot place order.")
         return None

    # Validate amount & price
    if amount <= 0:
         lg.error(f"Order amount must be positive for {symbol}. Amount: {amount}")
         return None
    if order_type.lower() == 'limit':
        if price is None or price <= 0:
            lg.error(f"Valid positive price parameter is required for limit orders ({symbol}). Price: {price}")
            return None
        # Format price to string with correct precision (use quantize without rounding?)
        price_digits = market_info.get('price_precision_digits', 8)
        price_str = f"{price:.{price_digits}f}"
    else:
        price_str = None  # No price for market orders

    # Format amount to string with correct precision
    amount_digits = market_info.get('amount_precision_digits', 8)
    amount_str = f"{amount:.{amount_digits}f}"

    # Prepare base parameters for Bybit V5
    # CCXT handles symbol mapping, but 'category' is often needed in params
    order_params = {'category': category}

    # Add position mode if needed (usually set account-wide, but can be specified)
    # position_mode = config.get("position_mode", "One-Way") # Assume loaded elsewhere
    # if position_mode == "Hedge":
    #     order_params['positionIdx'] = 1 if side == 'buy' else 2
    # else: # One-Way
    #     order_params['positionIdx'] = 0

    # Add user-provided params, potentially overriding base ones
    if params:
        order_params.update(params)
        lg.debug(f"Using additional order params: {params}")

    try:
        lg.info(f"Attempting to create {side.upper()} {order_type.upper()} order: {amount_str} {symbol} "
                f"{'@ ' + price_str if price_str else 'at Market'}")
        lg.debug(f"Final Order Params: {order_params}")

        # Convert Decimal amount/price to float for CCXT, as it often expects floats
        amount_float = float(amount_str)
        price_float = float(price_str) if price_str else None

        order_result = safe_ccxt_call(
            exchange,
            'create_order',
            lg,
            symbol=symbol,  # Use the standard symbol format CCXT expects
            type=order_type,
            side=side,
            amount=amount_float,
            price=price_float,
            params=order_params
        )

        if order_result and order_result.get('id'):
            # Bybit V5 create order response is nested in 'info'
            ret_code = order_result.get('info', {}).get('retCode', -1)
            ret_msg = order_result.get('info', {}).get('retMsg', 'Unknown')
            order_id = order_result.get('id')  # CCXT usually extracts the ID

            if ret_code == 0:
                 lg.info(f"{NEON_GREEN}Successfully created {side} {order_type} order for {symbol}. Order ID: {order_id}{RESET}")
                 # lg.debug(f"Order Result: {order_result}")
                 return order_result  # Return the parsed CCXT order structure
            else:
                 # Order was likely rejected by exchange despite API call success
                 lg.error(f"{NEON_RED}Order placement rejected by exchange for {symbol}. Code={ret_code}, Msg='{ret_msg}'. Order ID (if assigned): {order_id}{RESET}")
                 lg.debug(f"Rejected Order Result: {order_result}")
                 return None  # Indicate failure
        elif order_result:
             # Call succeeded but no order ID returned - unusual
             lg.error(f"Order placement call for {symbol} returned a result but no Order ID. Response: {order_result}")
             return None
        else:
            # safe_ccxt_call would have raised error if it failed after retries
            lg.error(f"Order placement call for {symbol} returned None unexpectedly (after retries).")
            return None

    except Exception as e:
        # Errors raised by safe_ccxt_call or validation steps
        lg.error(f"{NEON_RED}Failed to create order for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None


def set_protection_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_price: Decimal | None = None,  # Bybit V5: Absolute distance value (price ticks)
    trailing_active_price: Decimal | None = None,  # Bybit V5: Activation price trigger
    position_idx: int = 0,  # 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge
    logger: logging.Logger | None = None,
    market_info: dict | None = None
) -> bool:
    """Sets Stop Loss, Take Profit, and/or Trailing Stop using Bybit V5 position/trading-stop."""
    lg = logger or get_logger('main')
    if not market_info:
         lg.error(f"Market info required for set_protection_ccxt ({symbol})")
         return False

    category = market_info.get('category')
    market_id = market_info.get('id', symbol)
    price_digits = market_info.get('price_precision_digits', 8)

    if not category or category not in ['linear', 'inverse']:
        lg.warning(f"Cannot set SL/TP/TSL for non-derivative symbol {symbol} (Category: {category})")
        return False  # Not applicable for Spot

    # Check if the exchange explicitly supports this method via V5 endpoint
    # This check might be complex. Assume 'private_post_position_trading_stop' is the target.
    # We will call it directly using safe_ccxt_call.

    params = {
        'category': category,
        'symbol': market_id,
        # 'positionIdx': position_idx, # Required for Hedge mode, optional for One-Way (defaults to 0)
        # V5 requires tpslMode or specific order types. 'Full' targets the whole position.
        'tpslMode': 'Full',
        # stopLoss, takeProfit, trailingStop, activePrice are the key V5 params
    }
    # Add positionIdx based on config/mode if necessary
    # position_mode = config.get("position_mode", "One-Way")
    # if position_mode == "Hedge": params['positionIdx'] = position_idx # Use passed idx

    log_parts = []
    # Format prices to strings with correct precision if provided

    def format_price(price: Decimal | None) -> str:
        if price is not None and price.is_finite() and price > 0:
            return f"{price:.{price_digits}f}"
        return "0"  # Send "0" or empty string to cancel/not set

    sl_str = format_price(stop_loss_price)
    tp_str = format_price(take_profit_price)
    tsl_dist_str = format_price(trailing_stop_price)  # Trail distance treated as price value
    tsl_act_str = format_price(trailing_active_price)

    params['stopLoss'] = sl_str
    params['takeProfit'] = tp_str
    params['trailingStop'] = tsl_dist_str
    if tsl_dist_str != "0":  # Only include activePrice if TSL is being set
        params['activePrice'] = tsl_act_str

    # Log the intended actions
    if sl_str != "0": log_parts.append(f"SL={sl_str}")
    if tp_str != "0": log_parts.append(f"TP={tp_str}")
    if tsl_dist_str != "0": log_parts.append(f"TSL_Dist={tsl_dist_str}" + (f", Act={tsl_act_str}" if tsl_act_str != "0" else ", Act=Immediate"))

    if not log_parts:
        lg.warning(f"No valid protection levels provided for set_protection_ccxt ({symbol}). Nothing to set.")
        # Return True as technically no setting failed, but nothing was done.
        # Or False if expectation is that *something* should be set. Let's return True.
        return True

    try:
        lg.info(f"Setting protection for {symbol} (MarketID: {market_id}): {', '.join(log_parts)}")
        lg.debug(f"Protection Params: {params}")

        # Call the specific V5 endpoint directly via safe_ccxt_call
        # Method name format might vary slightly in different ccxt versions, check exchange.has
        # Assume 'privatePostPositionTradingStop' or similar is available internally in ccxt
        method_to_call = 'private_post_position_trading_stop'  # Check ccxt source or exchange.has if unsure
        result = safe_ccxt_call(exchange, method_to_call, lg, params=params)

        # Bybit V5 returns {"retCode": 0, "retMsg": "OK", ...} on success
        if result and result.get('retCode') == 0:
            lg.info(f"{NEON_GREEN}Successfully set protection for {symbol}.{RESET}")
            # lg.debug(f"Protection Result: {result}")
            return True
        elif result:
            ret_code = result.get('retCode', -1)
            ret_msg = result.get('retMsg', 'Unknown')
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}. Exchange Response: Code={ret_code}, Msg='{ret_msg}'{RESET}")
            lg.debug(f"Full protection result: {result}")
            return False
        else:
            # safe_ccxt_call failed after retries
            lg.error(f"Set protection call ({method_to_call}) for {symbol} failed or returned None after retries.")
            return False

    except Exception as e:
        lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return False


def close_position_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    position_data: dict,  # Pass the fetched position data
    logger: logging.Logger | None = None,
    market_info: dict | None = None
) -> dict | None:
    """Closes an existing position by placing a market order in the opposite direction with reduceOnly."""
    lg = logger or get_logger('main')
    if not market_info:
         lg.error(f"Market info required for close_position_ccxt ({symbol})")
         return None
    if not position_data:
         lg.error(f"Position data required for close_position_ccxt ({symbol})")
         return None

    try:
        # Use Decimal for size
        pos_size_str = position_data.get('contracts', position_data.get('info', {}).get('size'))  # Prefer 'contracts' if available
        position_side = position_data.get('side')  # 'long' or 'short'

        if pos_size_str is None or position_side is None:
             lg.error(f"Missing size or side in position data for closing {symbol}. Data: {position_data}")
             return None

        position_size = Decimal(str(pos_size_str))
        # Ensure size is positive absolute value for order placement
        amount_to_close = abs(position_size)

        if amount_to_close <= 0:
            lg.warning(f"Attempted to close position for {symbol}, but parsed size is {position_size}. No action needed.")
            return None  # No position to close

        close_side = 'sell' if position_side == 'long' else 'buy'

        lg.info(f"Attempting to close {position_side} position for {symbol} (Size: {amount_to_close}) by placing a {close_side} MARKET order...")

        # Add reduceOnly flag to ensure it only closes the position
        # Bybit V5 uses 'reduceOnly': True
        params = {
            'reduceOnly': True,
            # Add positionIdx if using Hedge Mode based on the position being closed
            # 'positionIdx': position_data.get('info', {}).get('positionIdx', 0) # Get from position info if possible
        }
        # position_mode = config.get("position_mode", "One-Way")
        # if position_mode == "Hedge":
        #      idx = position_data.get('info',{}).get('positionIdx')
        #      if idx is not None: params['positionIdx'] = idx

        close_order = create_order_ccxt(
            exchange=exchange,
            symbol=symbol,
            order_type='market',
            side=close_side,
            amount=amount_to_close,  # Pass the positive Decimal amount
            params=params,
            logger=lg,
            market_info=market_info
        )

        if close_order and close_order.get('id'):
            lg.info(f"{NEON_GREEN}Successfully placed MARKET order to close position for {symbol}. Close Order ID: {close_order.get('id')}{RESET}")
            # Note: Market order execution isn't guaranteed instantly.
            # Further checks might be needed to confirm closure (e.g., re-fetch position).
            return close_order  # Return the order info dict
        else:
            lg.error(f"{NEON_RED}Failed to place market order to close position for {symbol}. Check logs above.{RESET}")
            return None

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
         lg.error(f"Error parsing position data for closing {symbol}: {e}. Position Data: {position_data}", exc_info=True)
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Error attempting to close position for {symbol}: {e}{RESET}", exc_info=True)
        # traceback.print_exc()
        return None


# --- Main Bot Logic ---
def run_bot(exchange: ccxt.Exchange, config: dict[str, Any], bot_state: dict[str, Any]) -> None:
    """Main bot execution loop."""
    main_logger = get_logger('main')
    main_logger.info(f"{NEON_CYAN}=== Starting Enhanced Trading Bot ==={RESET}")
    main_logger.info(f"Trading Enabled: {config.get('enable_trading')}")
    main_logger.info(f"Sandbox Mode: {config.get('use_sandbox')}")
    main_logger.info(f"Symbols: {config.get('symbols')}")
    main_logger.info(f"Interval: {config.get('interval')}")
    main_logger.info(f"Risk Per Trade: {config.get('risk_per_trade') * 100:.2f}%")
    main_logger.info(f"Leverage: {config.get('leverage')}x")
    main_logger.info(f"Max Concurrent Positions: {config.get('max_concurrent_positions_total')}")
    main_logger.info(f"Position Mode: {config.get('position_mode')}")
    main_logger.info(f"Active Weight Set: {config.get('active_weight_set')}")
    main_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")

    global LOOP_DELAY_SECONDS
    try:
        LOOP_DELAY_SECONDS = int(config.get("loop_delay", DEFAULT_LOOP_DELAY_SECONDS))
        if LOOP_DELAY_SECONDS < 5:  # Add minimum delay sanity check
             main_logger.warning(f"Loop delay {LOOP_DELAY_SECONDS}s is very short. Increasing to 5s.")
             LOOP_DELAY_SECONDS = 5
    except (ValueError, TypeError):
        main_logger.error(f"Invalid loop_delay value in config. Using default: {DEFAULT_LOOP_DELAY_SECONDS}s")
        LOOP_DELAY_SECONDS = DEFAULT_LOOP_DELAY_SECONDS

    symbols_to_trade = config.get("symbols", [])
    if not symbols_to_trade or not isinstance(symbols_to_trade, list):
        main_logger.error("No valid 'symbols' list configured in config.json. Exiting.")
        return

    # Initialize state for each symbol if not present
    for symbol in symbols_to_trade:
        if symbol not in bot_state:
            bot_state[symbol] = {
                "break_even_triggered": False,
                "last_signal": "HOLD",
                "last_entry_price": None,  # Store entry price for BE reference if needed
                # Add other state variables as needed (e.g., last order time, consecutive losses)
            }
        # Ensure necessary keys exist even if state was loaded
        if "break_even_triggered" not in bot_state[symbol]: bot_state[symbol]["break_even_triggered"] = False
        if "last_signal" not in bot_state[symbol]: bot_state[symbol]["last_signal"] = "HOLD"
        if "last_entry_price" not in bot_state[symbol]: bot_state[symbol]["last_entry_price"] = None

    cycle_count = 0
    while True:
        cycle_count += 1
        start_time = time.time()
        main_logger.info(f"{NEON_BLUE}--- Starting Bot Cycle {cycle_count} ---{RESET}")

        # --- Pre-Cycle Checks ---
        # Fetch balance once per cycle (if trading enabled)
        current_balance: Decimal | None = None
        if config.get("enable_trading"):
            try:
                current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
                if current_balance is None:
                    main_logger.error(f"{NEON_RED}Failed to fetch {QUOTE_CURRENCY} balance. Trading actions will be skipped this cycle.{RESET}")
                    # Continue loop, but trading actions will fail safe later
                # else: # Balance logged by fetch_balance on success
            except Exception as e:
                 main_logger.error(f"Error fetching balance at start of cycle: {e}", exc_info=True)
                 # traceback.print_exc()

        # Fetch currently open positions across all monitored symbols
        open_positions_count = 0
        active_positions: dict[str, dict] = {}  # Store fetched active positions {symbol: position_data}
        main_logger.debug("Fetching positions for all configured symbols...")
        for symbol in symbols_to_trade:
            # Need market info to fetch positions correctly
            temp_logger = get_logger(symbol, is_symbol_logger=True)  # Use symbol logger
            market_info = get_market_info(exchange, symbol, temp_logger)
            if not market_info:
                 temp_logger.error(f"Cannot fetch position for {symbol}: Failed to get market info.")
                 continue  # Skip symbol if market info fails

            # Only fetch if it's a contract market
            if market_info.get('is_contract'):
                position = fetch_positions_ccxt(exchange, symbol, temp_logger, market_info)
                if position and position.get('contracts', Decimal('0')) != Decimal('0'):
                    # Position size check already done in fetch_positions_ccxt
                    open_positions_count += 1
                    active_positions[symbol] = position  # Store for later use
                    # Update state with entry price if position exists and state doesn't have it
                    if bot_state[symbol].get("last_entry_price") is None:
                         entry_p_str = position.get('entryPrice')
                         if entry_p_str is not None:
                              try: bot_state[symbol]["last_entry_price"] = str(Decimal(entry_p_str))  # Store as string
                              except: pass
            else:
                 temp_logger.debug(f"Skipping position fetch for non-contract symbol: {symbol}")

        max_allowed_positions = int(config.get("max_concurrent_positions_total", 1))
        main_logger.info(f"Currently open positions: {open_positions_count} / {max_allowed_positions}")

        # --- Symbol Processing Loop ---
        for symbol in symbols_to_trade:
            symbol_logger = get_logger(symbol, is_symbol_logger=True)
            symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
            symbol_state = bot_state[symbol]  # Get mutable state for this symbol

            try:
                # 1. Get Market Info (already fetched for position check, but get again for consistency?)
                # Let's re-fetch to ensure freshness within the symbol loop
                market_info = get_market_info(exchange, symbol, symbol_logger)
                if not market_info:
                    symbol_logger.error(f"Failed to get market info for {symbol}. Skipping this cycle.")
                    continue

                # 2. Fetch Data (Klines, Price, Orderbook)
                timeframe = config.get("interval", "5")
                if timeframe not in CCXT_INTERVAL_MAP:
                    symbol_logger.error(f"Invalid interval '{timeframe}' in config for {symbol}. Skipping.")
                    continue

                # Determine required kline limit based on longest indicator + buffer
                periods = [  # Collect all periods used
                    int(config.get(p, d)) for p, d in [
                        ("atr_period", DEFAULT_ATR_PERIOD),
                        ("ema_long_period", DEFAULT_EMA_LONG_PERIOD),
                        ("rsi_period", DEFAULT_RSI_WINDOW),
                        ("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD),
                        ("cci_window", DEFAULT_CCI_WINDOW),
                        ("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW),
                        ("mfi_window", DEFAULT_MFI_WINDOW),
                        ("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW),
                        ("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW),
                        ("sma_10_window", DEFAULT_SMA_10_WINDOW),
                        ("momentum_period", DEFAULT_MOMENTUM_PERIOD),
                        ("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD),
                        ("fibonacci_window", DEFAULT_FIB_WINDOW),
                    ] if int(config.get(p, d)) > 0  # Only consider positive periods
                ]
                kline_limit = max(periods) + 50 if periods else 150  # Longest period + buffer, or default 150

                df = fetch_klines_ccxt(exchange, symbol, timeframe, kline_limit, symbol_logger, market_info)
                if df.empty or len(df) < kline_limit / 2:  # Check if we got significantly less than requested
                    symbol_logger.warning(f"Kline data unavailable or insufficient ({len(df)} rows) for {symbol}. Skipping analysis.")
                    continue

                current_price_dec = fetch_current_price_ccxt(exchange, symbol, symbol_logger, market_info)
                if current_price_dec is None:
                    symbol_logger.warning(f"Current price unavailable for {symbol}. Skipping analysis.")
                    continue

                orderbook = None
                if config.get("indicators", {}).get("orderbook"):
                    try:
                        orderbook = fetch_orderbook_ccxt(exchange, symbol, int(config.get("orderbook_limit", 25)), symbol_logger, market_info)
                    except Exception as ob_err:
                        symbol_logger.warning(f"Failed to fetch orderbook for {symbol}: {ob_err}")
                    # Don't skip if orderbook fails, just proceed without it

                # 3. Analyze Data
                analyzer = TradingAnalyzer(df, symbol_logger, config, market_info, symbol_state)

                # 4. Check Existing Position & Manage
                current_position = active_positions.get(symbol)  # Use position fetched at start of cycle

                if current_position:
                    pos_side = current_position.get('side')
                    pos_size_str = current_position.get('contracts', current_position.get('info', {}).get('size'))
                    symbol_logger.info(f"Managing existing {pos_side} position (Size: {pos_size_str}).")
                    # Pass necessary data to management function
                    manage_existing_position(exchange, config, symbol_logger, analyzer, current_position, current_price_dec)
                    # Note: manage_existing_position might close the position. We don't check for new entries immediately after.

                # 5. Check for New Entry Signal (if no position OR if within limits - contract markets only)
                elif market_info.get('is_contract') and open_positions_count < max_allowed_positions:
                    symbol_logger.info("No active position. Checking for entry signals...")
                    # Reset BE trigger state if no position exists
                    if analyzer.break_even_triggered:
                         symbol_logger.info("Resetting break-even triggered state as position no longer exists.")
                         analyzer.break_even_triggered = False  # Update via property setter
                    if symbol_state.get("last_entry_price") is not None:
                         symbol_state["last_entry_price"] = None  # Clear last entry price

                    # Generate signal
                    signal = analyzer.generate_trading_signal(current_price_dec, orderbook)
                    symbol_state["last_signal"] = signal  # Store last signal

                    if signal in ["BUY", "SELL"]:
                        if config.get("enable_trading"):
                            if current_balance is not None and current_balance > 0:
                                 # Attempt to open a new position
                                 opened_new = attempt_new_entry(exchange, config, symbol_logger, analyzer, signal, current_price_dec, current_balance)
                                 if opened_new:
                                     open_positions_count += 1  # Increment count immediately
                                     # State (like BE triggered) is reset inside attempt_new_entry or manage
                                     # Store entry price in state
                                     # TODO: Get actual entry price from order result if possible
                                     approx_entry = analyzer.quantize_price(current_price_dec)  # Approximate entry
                                     symbol_state["last_entry_price"] = str(approx_entry) if approx_entry else str(current_price_dec)
                                     # Break inner loop to re-evaluate counts/balance? Maybe not needed if count updated.
                            else:
                                 symbol_logger.warning(f"Trading enabled but balance is {current_balance}. Cannot enter {signal} trade.")
                        else:  # Trading disabled
                             symbol_logger.info(f"Entry signal '{signal}' generated but trading is disabled.")
                    # No else needed for HOLD signal

                elif current_position is None and market_info.get('is_contract'):  # No position, but max positions reached
                     symbol_logger.info(f"Max concurrent positions ({open_positions_count}) reached. Skipping new entry check for {symbol}.")
                elif not market_info.get('is_contract'):  # Spot market
                     symbol_logger.debug(f"Spot market {symbol}. Entry logic currently focused on contracts. Skipping entry check.")
                     # TODO: Add specific spot entry logic if needed

            except Exception as e:
                symbol_logger.error(f"{NEON_RED}!!! Unhandled error during cycle for {symbol}: {e} !!!{RESET}", exc_info=True)
                # traceback.print_exc()

            symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
            # Small delay between symbols to potentially ease API load
            time.sleep(0.2)  # Reduced delay

        # --- Post-Cycle ---
        end_time = time.time()
        cycle_duration = end_time - start_time
        main_logger.info(f"{NEON_BLUE}--- Bot Cycle {cycle_count} Finished (Duration: {cycle_duration:.2f}s) ---{RESET}")

        # Save state periodically (e.g., every cycle or less frequently)
        save_state(STATE_FILE, bot_state, main_logger)

        # Wait for the next cycle
        wait_time = max(0, LOOP_DELAY_SECONDS - cycle_duration)
        if wait_time > 0:
            main_logger.info(f"Waiting {wait_time:.2f}s for next cycle (Loop Delay: {LOOP_DELAY_SECONDS}s)...")
            time.sleep(wait_time)
        else:
            main_logger.warning(f"Cycle duration ({cycle_duration:.2f}s) exceeded loop delay ({LOOP_DELAY_SECONDS}s). Starting next cycle immediately.")


def manage_existing_position(
    exchange: ccxt.Exchange,
    config: dict[str, Any],
    logger: logging.Logger,
    analyzer: TradingAnalyzer,  # Provides access to indicators, market_info, state
    position_data: dict,
    current_price_dec: Decimal
) -> None:
    """Handles logic for managing an open position (BE, MA Cross Exit)."""
    symbol = position_data.get('symbol')
    position_side = position_data.get('side')  # 'long' or 'short'
    entry_price_str = position_data.get('entryPrice')
    # Use absolute size from position_data if available
    pos_size_str = position_data.get('contracts', position_data.get('info', {}).get('size'))
    market_info = analyzer.market_info  # Get from analyzer
    symbol_state = analyzer.symbol_state  # Get state dict

    if not all([symbol, position_side, entry_price_str, pos_size_str]):
        logger.error(f"Incomplete position data for management: {position_data}")
        return

    try:
        entry_price = Decimal(str(entry_price_str))
        position_size = Decimal(str(pos_size_str))  # This is absolute size
        if position_size <= 0:
             logger.warning(f"Position size is {position_size} for {symbol} in management. Skipping.")
             return  # Should not happen if called correctly

        # --- 1. Check MA Cross Exit ---
        if config.get("enable_ma_cross_exit"):
            ema_short_f = analyzer._get_indicator_float("EMA_Short")
            ema_long_f = analyzer._get_indicator_float("EMA_Long")

            if ema_short_f is not None and ema_long_f is not None:
                is_adverse_cross = False
                # Check cross condition (using a small tolerance/hysteresis might be better)
                tolerance = 0.0001  # Example tolerance (adjust based on price scale)
                if position_side == 'long' and ema_short_f < ema_long_f * (1 - tolerance):
                    is_adverse_cross = True
                    logger.warning(f"{NEON_YELLOW}MA Cross Exit Triggered (Long): Short EMA ({ema_short_f:.5f}) crossed below Long EMA ({ema_long_f:.5f}). Closing position.{RESET}")
                elif position_side == 'short' and ema_short_f > ema_long_f * (1 + tolerance):
                    is_adverse_cross = True
                    logger.warning(f"{NEON_YELLOW}MA Cross Exit Triggered (Short): Short EMA ({ema_short_f:.5f}) crossed above Long EMA ({ema_long_f:.5f}). Closing position.{RESET}")

                if is_adverse_cross and config.get("enable_trading"):
                    logger.info("Attempting to close position due to MA cross...")
                    close_result = close_position_ccxt(exchange, symbol, position_data, logger, market_info)
                    if close_result:
                        # Reset state after closing attempt (assuming market order will fill)
                        symbol_state["break_even_triggered"] = False
                        symbol_state["last_signal"] = "HOLD"  # Reset signal state
                        symbol_state["last_entry_price"] = None
                        logger.info(f"Position close order placed for {symbol} due to MA Cross.")
                        return  # Exit management logic as position is being closed
                    else:
                         logger.error(f"Failed to place position close order for {symbol} after MA Cross trigger.")
                         # Continue with other checks? Or retry close? For now, continue.

        # --- 2. Check Break-Even Trigger (Only if not already triggered) ---
        if config.get("enable_break_even") and not analyzer.break_even_triggered:
            atr_val = analyzer.indicator_values.get("ATR")  # Decimal
            if atr_val is not None and atr_val.is_finite() and atr_val > 0:
                try:
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    profit_target_points = atr_val * trigger_multiple

                    # Calculate current profit in price points
                    current_profit_points = Decimal('0')
                    if position_side == 'long':
                        current_profit_points = current_price_dec - entry_price
                    elif position_side == 'short':
                        current_profit_points = entry_price - current_price_dec

                    # Check if profit target is reached
                    if current_profit_points >= profit_target_points:
                        logger.info(f"{NEON_GREEN}Break-Even Triggered for {symbol}!{RESET} Profit ({current_profit_points:.5f}) >= Target ({profit_target_points:.5f})")

                        # Calculate BE price (entry + offset)
                        min_tick = analyzer.get_min_tick_size()
                        offset_ticks = int(config.get("break_even_offset_ticks", 2))

                        if min_tick is not None and min_tick > 0 and offset_ticks >= 0:
                            offset_value = min_tick * Decimal(offset_ticks)
                            be_stop_price = None
                            if position_side == 'long':
                                be_stop_price_raw = entry_price + offset_value
                                # Quantize UP to ensure it's at least offset_ticks away / above entry
                                be_stop_price = analyzer.quantize_price(be_stop_price_raw, rounding=ROUND_UP)
                                # Final check: ensure BE stop is strictly > entry
                                if be_stop_price is not None and be_stop_price <= entry_price:
                                     be_stop_price = analyzer.quantize_price(entry_price + min_tick, rounding=ROUND_UP)

                            elif position_side == 'short':
                                be_stop_price_raw = entry_price - offset_value
                                # Quantize DOWN to ensure it's at least offset_ticks away / below entry
                                be_stop_price = analyzer.quantize_price(be_stop_price_raw, rounding=ROUND_DOWN)
                                # Final check: ensure BE stop is strictly < entry
                                if be_stop_price is not None and be_stop_price >= entry_price:
                                     be_stop_price = analyzer.quantize_price(entry_price - min_tick, rounding=ROUND_DOWN)

                            if be_stop_price is not None and be_stop_price > 0:
                                logger.info(f"Calculated Break-Even Stop Price: {be_stop_price}")
                                # Set the new SL
                                if config.get("enable_trading"):
                                    # Get current TP/TSL from position info to potentially preserve them
                                    # Note: CCXT parsing might place these in 'info'
                                    pos_info = position_data.get('info', {})
                                    current_tp_str = pos_info.get('takeProfit')  # Check info dict first
                                    if not current_tp_str: current_tp_str = position_data.get('takeProfit')  # Check top level
                                    current_tsl_dist_str = pos_info.get('trailingStop')
                                    current_tsl_act_str = pos_info.get('activePrice')

                                    current_tp = Decimal(current_tp_str) if current_tp_str and current_tp_str != "0" else None
                                    current_tsl_dist = Decimal(current_tsl_dist_str) if current_tsl_dist_str and current_tsl_dist_str != "0" else None
                                    current_tsl_act = Decimal(current_tsl_act_str) if current_tsl_act_str and current_tsl_act_str != "0" else None

                                    # Decide whether to keep TSL or force fixed SL at BE
                                    use_tsl = config.get("enable_trailing_stop") and not config.get("break_even_force_fixed_sl")
                                    tsl_to_set = current_tsl_dist if use_tsl and current_tsl_dist else None
                                    act_to_set = current_tsl_act if use_tsl and tsl_to_set else None
                                    tp_to_set = current_tp  # Always keep existing TP unless explicitly changed

                                    logger.info(f"Attempting to set BE SL={be_stop_price}" + (f", preserving TP={tp_to_set}" if tp_to_set else "") + (f", preserving TSL={tsl_to_set}" if tsl_to_set else "") + (", removing TSL" if not tsl_to_set and current_tsl_dist else ""))

                                    # Get positionIdx if needed
                                    pos_idx = pos_info.get('positionIdx', 0)

                                    success = set_protection_ccxt(
                                        exchange=exchange,
                                        symbol=symbol,
                                        stop_loss_price=be_stop_price,
                                        take_profit_price=tp_to_set,
                                        trailing_stop_price=tsl_to_set,
                                        trailing_active_price=act_to_set,
                                        position_idx=pos_idx,
                                        logger=logger,
                                        market_info=market_info
                                    )
                                    if success:
                                        logger.info(f"{NEON_GREEN}Successfully set break-even stop loss for {symbol}.{RESET}")
                                        analyzer.break_even_triggered = True  # Update state via property setter
                                    else:
                                        logger.error(f"{NEON_RED}Failed to set break-even stop loss via API for {symbol}. State not updated.{RESET}")
                            else:
                                 logger.error(f"Calculated invalid break-even stop price ({be_stop_price}). Cannot set BE.")
                        else:
                             logger.error(f"Cannot calculate break-even offset for {symbol}: Invalid tick size ({min_tick}) or offset ticks ({offset_ticks}).")
                except (InvalidOperation, ValueError, TypeError) as e:
                    logger.error(f"Error during break-even calculation for {symbol}: {e}")
            else:
                 logger.warning(f"Cannot check break-even trigger for {symbol}: Invalid ATR ({atr_val}).")

        # --- 3. Add other management logic here ---
        # (e.g., partial closes based on Fib levels, TSL adjustments if done manually)
        # logger.debug(f"Finished managing position for {symbol}.")

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error parsing position data during management for {symbol}: {e}. Data: {position_data}", exc_info=True)
        # traceback.print_exc()
    except Exception as e:
        logger.error(f"Unexpected error managing position for {symbol}: {e}", exc_info=True)
        # traceback.print_exc()


def attempt_new_entry(
    exchange: ccxt.Exchange,
    config: dict[str, Any],
    logger: logging.Logger,
    analyzer: TradingAnalyzer,  # Provides access to indicators, market_info, state, quantization
    signal: str,  # "BUY" or "SELL"
    entry_price_signal: Decimal,  # Price at time of signal generation
    current_balance: Decimal
) -> bool:
    """Attempts to calculate size, set leverage, place order, and set protection."""
    symbol = analyzer.symbol
    market_info = analyzer.market_info
    symbol_state = analyzer.symbol_state

    logger.info(f"Attempting {signal} entry for {symbol} based on signal price {entry_price_signal:.{analyzer.get_price_precision_digits()}f}")

    # 1. Calculate Quantized Entry, TP, SL based on signal price and current ATR
    quantized_entry, take_profit_price, stop_loss_price = analyzer.calculate_entry_tp_sl(entry_price_signal, signal)

    if quantized_entry is None:
        logger.error(f"Cannot enter {signal} trade for {symbol}: Failed to quantize entry price.")
        return False
    if stop_loss_price is None:
        logger.error(f"Cannot enter {signal} trade for {symbol}: Failed to calculate a valid Stop Loss.")
        return False
    # TP is optional, can proceed without it if None

    # 2. Calculate Position Size
    risk_per_trade = float(config.get("risk_per_trade", 0.01))  # Ensure float
    leverage = int(config.get("leverage", 10))  # Ensure int
    position_size = calculate_position_size(
        balance=current_balance,
        risk_per_trade=risk_per_trade,
        entry_price=quantized_entry,  # Use quantized entry for sizing
        stop_loss_price=stop_loss_price,
        market_info=market_info,
        leverage=leverage,
        logger=logger
    )

    if position_size is None or position_size <= 0:
        logger.error(f"Cannot enter {signal} trade for {symbol}: Failed to calculate a valid position size (Result: {position_size}).")
        return False

    # 3. Set Leverage (do this before ordering, only for contracts)
    if market_info.get('is_contract'):
        leverage_set = set_leverage_ccxt(exchange, symbol, leverage, logger, market_info)
        if not leverage_set:
            logger.error(f"Failed to set leverage for {symbol}. Aborting trade entry.")
            return False
        # Add a small delay after setting leverage if needed by exchange
        # time.sleep(0.5) # Optional short delay

    # 4. Place Entry Order (Market Order for simplicity)
    side = 'buy' if signal == 'BUY' else 'sell'
    # Use positionIdx=0 for One-Way mode (default)
    # For Hedge mode, would need logic here to determine 1 (buy) or 2 (sell)
    entry_order_params = {
         # 'positionIdx': 0 # Default for One-Way
    }
    # position_mode = config.get("position_mode", "One-Way")
    # if position_mode == "Hedge": entry_order_params['positionIdx'] = 1 if side == 'buy' else 2

    entry_order = create_order_ccxt(
        exchange=exchange,
        symbol=symbol,
        order_type='market',  # Consider using Limit orders slightly adjusted for better entry
        side=side,
        amount=position_size,  # Pass Decimal size
        params=entry_order_params,
        logger=logger,
        market_info=market_info
    )

    if entry_order is None or not entry_order.get('id'):
        logger.error(f"Failed to place entry market order for {symbol}. Aborting trade.")
        return False

    # --- Entry Order Placed ---
    order_id = entry_order['id']
    logger.info(f"Entry order ({order_id}) placed successfully for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for confirmation/fill...")
    time.sleep(POSITION_CONFIRM_DELAY)  # Wait for order to likely fill and position to appear

    # 5. Fetch Actual Entry Price (More Robust) - Optional but Recommended
    # Attempt to fetch the filled order or the updated position info
    actual_entry_price = quantized_entry  # Default to calculated entry
    filled_size = position_size  # Assume full fill for now
    try:
        # Option A: Fetch Order
        # fetched_order = safe_ccxt_call(exchange, 'fetch_order', logger, id=order_id, symbol=symbol, params={'category': market_info.get('category')})
        # if fetched_order and fetched_order.get('status') == 'closed' and fetched_order.get('average'):
        #     actual_entry_price = Decimal(str(fetched_order['average']))
        #     filled_size = Decimal(str(fetched_order.get('filled', position_size)))
        #     logger.info(f"Confirmed entry order filled. Actual Entry: {actual_entry_price}, Filled Size: {filled_size}")
        # else:
        #     logger.warning(f"Could not confirm entry order fill details via fetch_order. Using calculated entry price. Order status: {fetched_order.get('status') if fetched_order else 'N/A'}")

        # Option B: Re-fetch Position (might be more reliable)
        logger.debug(f"Re-fetching position for {symbol} to confirm entry and get details...")
        time.sleep(1)  # Extra short delay before fetching position
        updated_position = fetch_positions_ccxt(exchange, symbol, logger, market_info)
        if updated_position and updated_position.get('entryPrice'):
             entry_p_str = updated_position.get('entryPrice')
             actual_entry_price = Decimal(str(entry_p_str))
             # Verify size matches expected (or is close)
             current_size_str = updated_position.get('contracts', updated_position.get('info', {}).get('size'))
             current_size = Decimal(str(current_size_str)) if current_size_str else Decimal('0')
             filled_size = abs(current_size)  # Update filled size
             logger.info(f"Confirmed position update. Actual Entry: {actual_entry_price}, Current Size: {filled_size} (Expected: {position_size})")
             if abs(filled_size - position_size) / position_size > Decimal('0.01'):  # Check if fill size deviates significantly
                   logger.warning(f"Filled size {filled_size} differs significantly from ordered size {position_size}.")
                   # Adjust SL/TP based on actual filled size? Complex. For now, proceed with original plan.
        else:
             logger.warning(f"Could not confirm entry details by re-fetching position. Using calculated entry price {actual_entry_price}.")

    except Exception as confirm_err:
         logger.error(f"Error confirming entry order/position details: {confirm_err}. Using calculated entry price {actual_entry_price}.")
         # traceback.print_exc()

    # 6. Set SL/TP/TSL Protection using the *actual* or best-known entry price
    tsl_distance = None
    tsl_activation_price = None
    if config.get("enable_trailing_stop"):
        try:
            callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
            activation_perc = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
            min_tick = analyzer.get_min_tick_size()

            if callback_rate > 0 and min_tick is not None and min_tick > 0:
                # Calculate absolute distance value based on actual entry price
                tsl_distance_raw = actual_entry_price * callback_rate
                # Quantize distance UP to nearest tick (safer trail)
                tsl_distance = (tsl_distance_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

                if tsl_distance < min_tick:  # Ensure trail is at least one tick
                     tsl_distance = min_tick
                     logger.warning(f"Calculated TSL distance {tsl_distance_raw} too small, adjusted to min tick: {tsl_distance}")

                # Calculate activation price if percentage > 0
                if activation_perc > 0:
                    activation_offset = actual_entry_price * activation_perc
                    tsl_activation_price_raw = None
                    if signal == "BUY":
                        tsl_activation_price_raw = actual_entry_price + activation_offset
                        # Quantize UP, away from entry
                        tsl_activation_price = analyzer.quantize_price(tsl_activation_price_raw, rounding=ROUND_UP)
                        # Ensure activation is > entry
                        if tsl_activation_price is not None and tsl_activation_price <= actual_entry_price:
                            tsl_activation_price = analyzer.quantize_price(actual_entry_price + min_tick, rounding=ROUND_UP)
                    else:  # SELL
                        tsl_activation_price_raw = actual_entry_price - activation_offset
                        # Quantize DOWN, away from entry
                        tsl_activation_price = analyzer.quantize_price(tsl_activation_price_raw, rounding=ROUND_DOWN)
                         # Ensure activation is < entry
                        if tsl_activation_price is not None and tsl_activation_price >= actual_entry_price:
                            tsl_activation_price = analyzer.quantize_price(actual_entry_price - min_tick, rounding=ROUND_DOWN)

                    # Ensure activation price is valid (positive)
                    if tsl_activation_price is not None and tsl_activation_price <= 0: tsl_activation_price = None

                logger.debug(f"Calculated TSL: Distance={tsl_distance}, ActivationPrice={tsl_activation_price} (based on entry {actual_entry_price})")
            else:
                 logger.warning("Could not calculate TSL distance for {symbol} due to invalid rate ({callback_rate}) or tick size ({min_tick}).")
        except Exception as tsl_calc_err:
            logger.error(f"Error calculating TSL parameters for {symbol}: {tsl_calc_err}", exc_info=True)
            # traceback.print_exc()

    # Get positionIdx if needed
    pos_idx = 0
    # if position_mode == "Hedge": pos_idx = 1 if side == 'buy' else 2

    protection_set = set_protection_ccxt(
        exchange=exchange,
        symbol=symbol,
        stop_loss_price=stop_loss_price,  # Use originally calculated SL price
        take_profit_price=take_profit_price,  # Use originally calculated TP price
        trailing_stop_price=tsl_distance,  # Use calculated TSL distance
        trailing_active_price=tsl_activation_price,  # Use calculated TSL activation
        position_idx=pos_idx,
        logger=logger,
        market_info=market_info
    )

    if not protection_set:
        logger.error(f"{NEON_RED}Failed to set initial SL/TP/TSL for {symbol} after entry! Position might be unprotected.{RESET}")
        # CRITICAL: Consider closing the position immediately if protection fails and trading is enabled
        if config.get("enable_trading"):
            logger.warning(f"Attempting emergency close of unprotected position {symbol}...")
            # Need position data again - re-fetch?
            pos_to_close = fetch_positions_ccxt(exchange, symbol, logger, market_info)
            if pos_to_close:
                 close_position_ccxt(exchange, symbol, pos_to_close, logger, market_info)
            else:
                 logger.error(f"Could not fetch position data to perform emergency close for {symbol}.")
        return False  # Indicate overall entry failure

    logger.info(f"{NEON_GREEN}Successfully entered {signal} trade for {symbol} with protection.{RESET}")
    # Reset BE state for the new trade
    symbol_state["break_even_triggered"] = False
    symbol_state["last_entry_price"] = str(actual_entry_price)  # Store actual entry price string
    return True  # Indicate success


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Bybit V5 Multi-Symbol Trading Bot")
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE, help=f"Path to configuration file (default: {CONFIG_FILE})"
    )
    parser.add_argument(
        "--state", type=str, default=STATE_FILE, help=f"Path to state file (default: {STATE_FILE})"
    )
    parser.add_argument(
        "--symbol", "--symbols", dest="symbols", type=str, help="Override symbols in config: trade only specified symbol(s), comma-separated (e.g., BTC/USDT:USDT,ETH/USDT:USDT)"
    )
    parser.add_argument(
        "--live", action="store_true", help="Enable live trading (overrides config 'enable_trading=false' and 'use_sandbox=true')"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG level logging to console"
    )
    args = parser.parse_args()

    # Set console log level based on args BEFORE setting up loggers
    console_log_level = logging.DEBUG if args.debug else logging.INFO

    # Setup main logger first using the determined level
    main_logger = get_logger('main')
    main_logger.info(" --- Bot Starting --- ")

    # Load configuration
    config = load_config(args.config)

    # Override symbols if specified via command line
    if args.symbols:
        override_symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        if override_symbols:
             main_logger.warning(f"{NEON_YELLOW}Overriding config symbols via command line. Trading ONLY: {override_symbols}{RESET}")
             config["symbols"] = override_symbols
        else:
             main_logger.error("Empty symbol list provided via command line override. Check --symbols argument.")
             sys.exit(1)

    # Override trading mode and sandbox if --live flag is set
    if args.live:
        main_logger.warning(f"{NEON_RED}--- LIVE TRADING ENABLED via command line override! ---{RESET}")
        if config.get("enable_trading", False) is False:
             main_logger.warning("Overriding config 'enable_trading=false'.")
             config["enable_trading"] = True
        if config.get("use_sandbox", True) is True:
             main_logger.warning("Overriding config 'use_sandbox=true'.")
             config["use_sandbox"] = False
    # Log final effective trading/sandbox mode
    elif config.get("enable_trading"):
         main_logger.warning(f"{NEON_RED}--- Live trading enabled via config file ---{RESET}")
    else:
         main_logger.info("Live trading disabled.")
    if config.get("use_sandbox"):
         main_logger.warning(f"{NEON_YELLOW}Sandbox mode (testnet) is ACTIVE.{RESET}")
    else:
         main_logger.warning(f"{NEON_RED}Sandbox mode is INACTIVE (using real exchange).{RESET}")

    # Load state
    bot_state = load_state(args.state, main_logger)

    # Initialize exchange
    exchange = initialize_exchange(config, main_logger)

    # --- Run Bot ---
    if exchange:
        main_logger.info(f"{NEON_GREEN}Exchange initialized successfully. Starting main loop...{RESET}")
        try:
            run_bot(exchange, config, bot_state)
        except KeyboardInterrupt:
            main_logger.info("Bot stopped by user (KeyboardInterrupt).")
        except Exception as e:
            main_logger.critical(f"{NEON_RED}!!! BOT CRASHED due to unhandled exception in main loop: {e} !!!{RESET}", exc_info=True)
            # traceback.print_exc() # Ensure traceback is printed
        finally:
            # Save state on exit, regardless of reason
            main_logger.info("Attempting to save final state...")
            save_state(args.state, bot_state, main_logger)
            main_logger.info("--- Bot Shutdown ---")
    else:
        main_logger.critical("Failed to initialize exchange. Bot cannot start.")

    logging.shutdown()  # Ensure all logs are flushed before exit
    sys.exit(0 if exchange else 1)  # Exit with error code if exchange failed init
