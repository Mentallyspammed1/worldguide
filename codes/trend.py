```python
# merged_bot.py
# Merged version combining features from livebot7.py and volbot5.py
# Supports both LiveXY (indicator scoring) and Volbot (trend/OB) strategies,
# configurable via config.json. Includes enhanced risk management (TSL, BE).

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta # Ensure pandas_ta is installed
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo # Use zoneinfo (Python 3.9+)

# Initialize colorama and set Decimal precision
init(autoreset=True)
getcontext().prec = 28  # Increased precision for financial calculations
load_dotenv()

# --- Constants ---
# Color Scheme (Consolidated)
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# Strategy-Specific Colors & Log Levels
COLOR_UP = Fore.CYAN + Style.BRIGHT
COLOR_DN = Fore.YELLOW + Style.BRIGHT
COLOR_BULL_BOX = Fore.GREEN
COLOR_BEAR_BOX = Fore.RED
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX
COLOR_INFO = Fore.MAGENTA
COLOR_HEADER = Fore.BLUE + Style.BRIGHT
COLOR_WARNING = NEON_YELLOW
COLOR_ERROR = NEON_RED
COLOR_SUCCESS = NEON_GREEN

# API Credentials (Loaded from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    print(f"{COLOR_ERROR}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    raise ValueError("API Key/Secret not found in environment variables.")

# File/Directory Configuration
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Time & Retry Configuration
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default timezone string
try:
    TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_STR)
except Exception as tz_err:
    print(f"{COLOR_ERROR}CRITICAL: Default timezone '{DEFAULT_TIMEZONE_STR}' invalid or system tzdata missing: {tz_err}. Exiting.{RESET}")
    exit(1)

MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
LOOP_DELAY_SECONDS = 15 # Min time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status

# Interval Configuration
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# API Error Codes for Retry Logic (HTTP status codes)
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]

# --- Default Indicator/Strategy Parameters (overridden by config.json) ---
# LiveXY Defaults
DEFAULT_ATR_PERIOD_LIVEXY = 14 # Renamed to avoid conflict with Risk ATR
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14
DEFAULT_STOCH_WINDOW = 12
DEFAULT_K_WINDOW = 3
DEFAULT_D_WINDOW = 3
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50 # Note: Fibonacci levels not used in signal generation logic yet
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # For potential future use

# Volbot Defaults
DEFAULT_VOLBOT_LENGTH = 40
DEFAULT_VOLBOT_ATR_LENGTH = 200
DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK = 1000
DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE = 100
DEFAULT_VOLBOT_OB_SOURCE = "Wicks" # "Wicks" or "Bodys"
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L = 25
DEFAULT_VOLBOT_MAX_BOXES = 50

# Default Risk Management Parameters (overridden by config.json)
DEFAULT_ATR_PERIOD_RISK = 14 # Default ATR period specifically for Risk Management (SL/TP/BE)

# Global QUOTE_CURRENCY placeholder, dynamically loaded from config
QUOTE_CURRENCY = "USDT" # Default fallback, updated by load_config

# Default console log level (updated by config)
console_log_level = logging.INFO

# --- Logger Setup (from volbot5) ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that redacts sensitive information like API keys/secrets from log messages."""
    REDACTION_STR = "***REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive info."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, self.REDACTION_STR)
        if API_SECRET:
            msg = msg.replace(API_SECRET, self.REDACTION_STR)
        return msg

def setup_logger(name_suffix: str) -> logging.Logger:
    """
    Sets up a logger instance with specified suffix, file rotation, and colored console output.
    Prevents adding duplicate handlers and updates console level based on global setting.
    """
    global console_log_level
    safe_suffix = re.sub(r'[^\w\-]+', '_', name_suffix)
    logger_name = f"merged_bot_{safe_suffix}" # Updated logger name prefix
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.level != console_log_level:
                    logger.debug(f"Updating console handler level for {logger_name} to {logging.getLevelName(console_log_level)}")
                    handler.setLevel(console_log_level)
        return logger

    logger.setLevel(logging.DEBUG)

    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{COLOR_ERROR}Error setting up file logger for {log_filename}: {e}{RESET}")

    stream_handler = logging.StreamHandler()
    level_colors = {
        logging.DEBUG: NEON_BLUE, logging.INFO: NEON_GREEN, logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED, logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    class ColorFormatter(SensitiveFormatter):
        def format(self, record):
            log_color = level_colors.get(record.levelno, RESET)
            record.levelname = f"{log_color}{record.levelname:<8}{RESET}"
            record.asctime = f"{NEON_BLUE}{self.formatTime(record, self.datefmt)}{RESET}"
            base_name = record.name.split('_', 2)[-1] if record.name.count('_') >= 2 else record.name # Adjusted split index
            record.name_part = f"{NEON_PURPLE}[{base_name}]{RESET}"
            record.msg = f"{log_color if record.levelno >= logging.WARNING else ''}{record.getMessage()}{RESET}"
            formatted_message = super(SensitiveFormatter, self).format(record)
            return f"{formatted_message}{RESET}"

    stream_formatter = ColorFormatter(
        "%(asctime)s - %(levelname)s - %(name_part)s - %(message)s",
        datefmt='%H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

# --- Configuration Loading (Merged) ---
def load_config(filepath: str) -> Dict[str, Any]:
    """Loads configuration, creates default, ensures keys, updates globals."""
    global QUOTE_CURRENCY, TIMEZONE, console_log_level

    default_config = {
        # --- General Bot Settings ---
        "timezone": DEFAULT_TIMEZONE_STR,
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
        "enable_trading": False,
        "use_sandbox": True,
        "risk_per_trade": 0.01,
        "leverage": 10, # Combined leverage setting
        "max_concurrent_positions": 1, # (Informational for now)
        "quote_currency": "USDT",
        "console_log_level": "INFO",
        "signal_mode": "both_align", # 'livexy', 'volbot', 'both_align'

        # --- Risk Management Settings ---
        "atr_period_risk": DEFAULT_ATR_PERIOD_RISK, # ATR period for SL/TP/BE
        "stop_loss_multiple": 1.8,
        "take_profit_multiple": 0.7,
        "enable_trailing_stop": True,
        # Use string for callback rate for Bybit flexibility (e.g., "0.005" or "50")
        "trailing_stop_callback_rate": "0.005",
        "trailing_stop_activation_percentage": 0.003,
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2,

        # --- LiveXY Strategy Settings ---
        "livexy_enabled": True, # Flag to enable LiveXY
        "livexy_atr_period": DEFAULT_ATR_PERIOD_LIVEXY, # Renamed config key
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
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5, # Score needed for LiveXY signal
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "volume_confirmation_multiplier": 1.5,
        "fibonacci_window": DEFAULT_FIB_WINDOW, # (Informational for now)
        "indicators": { # Control which LiveXY indicators are calculated/weighted
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
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
        "active_weight_set": "default", # Which LiveXY weight set to use

        # --- Volbot Strategy Settings ---
        "volbot_enabled": True, # Flag to enable Volbot
        "volbot_length": DEFAULT_VOLBOT_LENGTH,
        "volbot_atr_length": DEFAULT_VOLBOT_ATR_LENGTH,
        "volbot_volume_percentile_lookback": DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK,
        "volbot_volume_normalization_percentile": DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE,
        "volbot_ob_source": DEFAULT_VOLBOT_OB_SOURCE,
        "volbot_pivot_left_len_h": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H,
        "volbot_pivot_right_len_h": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H,
        "volbot_pivot_left_len_l": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L,
        "volbot_pivot_right_len_l": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L,
        "volbot_max_boxes": DEFAULT_VOLBOT_MAX_BOXES,
        "volbot_signal_on_trend_flip": True,
        "volbot_signal_on_ob_entry": True,
    }

    loaded_config = {}
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            print(f"{COLOR_WARNING}Created default config file: {filepath}{RESET}")
            loaded_config = default_config.copy()
        except IOError as e:
            print(f"{COLOR_ERROR}Error creating default config file {filepath}: {e}. Using built-in defaults.{RESET}")
            loaded_config = default_config.copy()
    else:
        try:
            with open(filepath, 'r', encoding="utf-8") as f:
                loaded_config_from_file = json.load(f)
            loaded_config, config_updated = _ensure_config_keys(loaded_config_from_file, default_config)
            if config_updated:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(loaded_config, f_write, indent=4, sort_keys=True)
                    print(f"{COLOR_WARNING}Updated config file '{filepath}' with missing default keys.{RESET}")
                except IOError as e:
                    print(f"{COLOR_ERROR}Error writing updated config file {filepath}: {e}{RESET}")
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            print(f"{COLOR_ERROR}Error loading config {filepath}: {e}. Using defaults & recreating file.{RESET}")
            loaded_config = default_config.copy()
            try:
                with open(filepath, "w", encoding="utf-8") as f: json.dump(default_config, f, indent=4, sort_keys=True)
                print(f"{COLOR_WARNING}Recreated default config file: {filepath}{RESET}")
            except IOError as e_create: print(f"{COLOR_ERROR}Error recreating config file: {e_create}{RESET}")

    # --- Update global settings ---
    new_quote_currency = loaded_config.get("quote_currency", default_config["quote_currency"]).upper()
    if new_quote_currency != QUOTE_CURRENCY:
        print(f"{COLOR_INFO}Setting QUOTE_CURRENCY to: {new_quote_currency}{RESET}")
        QUOTE_CURRENCY = new_quote_currency

    level_name = loaded_config.get("console_log_level", "INFO").upper()
    new_log_level = getattr(logging, level_name, logging.INFO)
    if new_log_level != console_log_level:
        print(f"{COLOR_INFO}Setting console log level to: {level_name}{RESET}")
        console_log_level = new_log_level

    config_tz_str = loaded_config.get("timezone", DEFAULT_TIMEZONE_STR)
    try:
        new_tz = ZoneInfo(config_tz_str)
        if TIMEZONE is None or new_tz.key != TIMEZONE.key:
            print(f"{COLOR_INFO}Setting timezone to: {config_tz_str}{RESET}")
            TIMEZONE = new_tz
    except Exception as tz_err:
        print(f"{COLOR_ERROR}Invalid timezone '{config_tz_str}' in config: {tz_err}. Using default '{TIMEZONE.key}'.{RESET}")
        loaded_config["timezone"] = TIMEZONE.key

    # --- Validate specific config values ---
    if loaded_config.get("interval") not in VALID_INTERVALS:
        print(f"{COLOR_ERROR}Invalid 'interval': '{loaded_config.get('interval')}'. Using default '5'.{RESET}")
        loaded_config["interval"] = default_config["interval"]
    if loaded_config.get("volbot_ob_source") not in ["Wicks", "Bodys"]:
        print(f"{COLOR_ERROR}Invalid 'volbot_ob_source': '{loaded_config.get('volbot_ob_source')}'. Using default 'Wicks'.{RESET}")
        loaded_config["volbot_ob_source"] = default_config["volbot_ob_source"]
    if loaded_config.get("signal_mode") not in ['livexy', 'volbot', 'both_align']:
        print(f"{COLOR_ERROR}Invalid 'signal_mode': '{loaded_config.get('signal_mode')}'. Using default 'both_align'.{RESET}")
        loaded_config["signal_mode"] = default_config["signal_mode"]
    try:
        risk_val = float(loaded_config.get("risk_per_trade", default_config["risk_per_trade"]))
        if not (0 < risk_val < 1): raise ValueError("Risk must be > 0 and < 1")
        loaded_config["risk_per_trade"] = risk_val
    except (ValueError, TypeError) as e:
        print(f"{COLOR_ERROR}Invalid 'risk_per_trade': '{loaded_config.get('risk_per_trade')}'. Using default {default_config['risk_per_trade']}.{RESET}")
        loaded_config["risk_per_trade"] = default_config["risk_per_trade"]

    return loaded_config

def _ensure_config_keys(loaded_config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Recursively ensures default keys exist in loaded config."""
    updated = False
    for key, default_value in default_config.items():
        if key not in loaded_config:
            loaded_config[key] = default_value
            updated = True
            print(f"{COLOR_INFO}Config: Added missing key '{key}' with default value: {repr(default_value)[:50]}{RESET}") # Log default value concisely
        elif isinstance(default_value, dict) and isinstance(loaded_config.get(key), dict):
            nested_updated = _ensure_config_keys(loaded_config[key], default_value)[1]
            if nested_updated: updated = True
    return loaded_config, updated

# Load configuration globally AFTER functions are defined
CONFIG = load_config(CONFIG_FILE)

# --- CCXT Exchange Setup (from volbot5) ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object."""
    lg = logger
    try:
        exchange_id = 'bybit'
        exchange_class = getattr(ccxt, exchange_id)
        exchange_options = {
            'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', 'adjustForTimeDifference': True,
                'recvWindow': 10000, 'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000, 'cancelOrderTimeout': 15000, 'fetchOHLCVTimeout': 20000,
                'fetchPositionTimeout': 15000, 'fetchPositionsTimeout': 20000,
            },
            'requests_session': requests.Session()
        }
        retries = Retry(total=MAX_API_RETRIES, backoff_factor=0.5, status_forcelist=RETRY_ERROR_CODES, allowed_methods=None)
        adapter = HTTPAdapter(max_retries=retries)
        exchange_options['requests_session'].mount('https://', adapter)
        exchange_options['requests_session'].mount('http://', adapter)
        exchange = exchange_class(exchange_options)
        use_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(use_sandbox)
        sandbox_status = f"{COLOR_WARNING}SANDBOX MODE{RESET}" if use_sandbox else f"{COLOR_ERROR}LIVE TRADING MODE{RESET}"
        lg.warning(f"Exchange {exchange.id} (v{exchange.version}) initialized. Status: {sandbox_status}")
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            exchange.load_markets(reload=True)
            lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols).")
        except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
            lg.critical(f"{COLOR_ERROR}CRITICAL: Failed to load markets: {e}. Cannot proceed.{RESET}")
            return None
        lg.info(f"Testing API keys and connection via balance fetch ({QUOTE_CURRENCY})...")
        test_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if test_balance is not None:
            lg.info(f"{COLOR_SUCCESS}API connection successful. Initial {QUOTE_CURRENCY} balance: {test_balance:.4f}{RESET}")
        else:
            lg.critical(f"{COLOR_ERROR}CRITICAL: Initial balance fetch failed. Check API keys, permissions, IP whitelist, network, and Testnet/Live matching.{RESET}")
            return None
        return exchange
    except Exception as e:
        lg.critical(f"{COLOR_ERROR}CRITICAL Error initializing exchange: {e}{RESET}", exc_info=True)
    return None

# --- CCXT Data Fetching (Consolidated & Refined) ---

def safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely convert a value to Decimal, handling Nones, strings, non-finite numbers."""
    if value is None: return default
    try:
        str_value = str(value).strip()
        if not str_value: return default
        d = Decimal(str_value)
        if not d.is_finite(): return default
        return d
    except (InvalidOperation, ValueError, TypeError):
        return default

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches current market price using ticker with fallbacks and retries."""
    lg = logger
    price: Optional[Decimal] = None
    attempt = 0
    max_attempts = MAX_API_RETRIES + 1

    while attempt < max_attempts:
        attempt += 1
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempt}/{max_attempts})...")
            params = {}
            if 'bybit' in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    if market: params['category'] = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                    else: params['category'] = 'linear' # Default guess
                except Exception: params['category'] = 'linear'
            ticker = exchange.fetch_ticker(symbol, params=params)
            last = safe_decimal(ticker.get('last'))
            bid = safe_decimal(ticker.get('bid'))
            ask = safe_decimal(ticker.get('ask'))
            if last and last > 0: price = last; lg.debug(f"Using 'last' price: {price}"); break
            if bid and ask and bid > 0 and ask > 0 and bid <= ask: price = (bid + ask) / 2; lg.debug(f"Using mid price: {price}"); break
            if ask and ask > 0: price = ask; lg.debug(f"Using 'ask' price: {price}"); break
            if bid and bid > 0: price = bid; lg.debug(f"Using 'bid' price: {price}"); break
            lg.warning(f"No valid ticker price found for {symbol} (Attempt {attempt}). Ticker: {ticker}")
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS)
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"Network error fetching price for {symbol} (Attempt {attempt}): {e}. Retrying...")
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS * attempt)
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * (attempt + 1)
            lg.warning(f"Rate limit fetching price for {symbol} (Attempt {attempt}). Retrying in {wait_time}s: {e}")
            if attempt < max_attempts: time.sleep(wait_time)
        except ccxt.BadSymbol as e: lg.error(f"{COLOR_ERROR}Invalid symbol {symbol} for ticker: {e}{RESET}"); return None
        except ccxt.ExchangeError as e:
            if "symbol not found" in str(e).lower(): lg.error(f"{COLOR_ERROR}Symbol {symbol} not found for ticker: {e}{RESET}"); return None
            lg.warning(f"Exchange error fetching price for {symbol} (Attempt {attempt}): {e}. Retrying...")
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e: lg.error(f"{COLOR_ERROR}Unexpected error fetching price {symbol}: {e}{RESET}", exc_info=True); return None

    if price and price > 0: return price
    lg.error(f"{COLOR_ERROR}Failed to fetch valid price for {symbol} after {max_attempts} attempts.{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Fetches, validates, and cleans OHLCV kline data."""
    lg = logger or logging.getLogger(__name__)
    empty_df = pd.DataFrame()
    try:
        if not exchange.has.get('fetchOHLCV'): lg.error(f"{exchange.id} no support fetchOHLCV."); return empty_df
        ohlcv: Optional[List[List[Any]]] = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1})...")
                params = {}
                if 'bybit' in exchange.id.lower():
                    try:
                         market = exchange.market(symbol)
                         if market: params['category'] = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                         else: params['category'] = 'linear'
                    except Exception: params['category'] = 'linear'
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
                if ohlcv and len(ohlcv) > 0: lg.debug(f"Received {len(ohlcv)} klines."); break
                else: lg.warning(f"fetch_ohlcv returned empty {symbol} {timeframe} (Attempt {attempt+1}). Retrying..."); time.sleep(RETRY_DELAY_SECONDS)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES: lg.warning(f"Network error klines {symbol} (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(RETRY_DELAY_SECONDS * (attempt+1))
                else: raise e
            except ccxt.RateLimitExceeded as e:
                 wait_match = re.search(r'(\d+)\s*(?:ms|s)', str(e).lower())
                 wait_time = int(wait_match.group(1)) / 1000 if wait_match and 'ms' in wait_match.group(0) else int(wait_match.group(1)) if wait_match else RETRY_DELAY_SECONDS * (attempt+2)
                 lg.warning(f"Rate limit klines {symbol}. Retrying in {wait_time}s (Attempt {attempt+1}).")
                 if attempt < MAX_API_RETRIES: time.sleep(wait_time + 1)
                 else: raise e
            except ccxt.BadSymbol as e: lg.error(f"{COLOR_ERROR}Invalid symbol {symbol} for klines: {e}{RESET}"); return empty_df
            except ccxt.ExchangeError as e:
                 if "symbol not found" in str(e).lower(): lg.error(f"{COLOR_ERROR}Symbol {symbol} not found for klines: {e}{RESET}"); return empty_df
                 lg.warning(f"Exchange error klines {symbol} (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e: lg.error(f"{COLOR_ERROR}Unexpected error fetching klines {symbol}: {e}{RESET}", exc_info=True); raise e

        if not ohlcv: lg.warning(f"No kline data for {symbol} {timeframe} after retries."); return empty_df
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty: return empty_df
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = df[col].apply(lambda x: safe_decimal(x, default=np.nan))
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0]
        df['volume'].fillna(Decimal(0), inplace=True) # Fill NaN volume with Decimal 0
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0: lg.debug(f"Dropped {rows_dropped} invalid rows from klines.")
        if df.empty: lg.warning(f"Kline data empty after cleaning for {symbol}."); return empty_df
        df.sort_index(inplace=True)
        if df.index.duplicated().any():
            lg.warning(f"Duplicate timestamps found, keeping last entry.")
            df = df[~df.index.duplicated(keep='last')]
        lg.info(f"Fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df
    except Exception as e: lg.error(f"{COLOR_ERROR}Error processing klines {symbol}: {e}{RESET}", exc_info=True); return empty_df


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and validation."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.has['fetchOrderBook']: lg.error(f"{exchange.id} no support fetchOrderBook."); return None
            lg.debug(f"Fetching order book {symbol}, limit={limit} (Attempt {attempts+1})...")
            params = {}
            if 'bybit' in exchange.id.lower():
                 try: market = exchange.market(symbol); params['category'] = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                 except Exception: params['category'] = 'linear'
            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)
            if not orderbook: lg.warning(f"fetch_order_book returned None for {symbol} (Attempt {attempts+1}).")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                 lg.warning(f"Invalid orderbook structure {symbol} (Attempt {attempts+1}). Response: {orderbook}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 lg.warning(f"Orderbook received but bids/asks empty {symbol} (Attempt {attempts+1}).")
                 return orderbook # Return empty book
            else:
                 lg.debug(f"Fetched orderbook {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"Orderbook net error {symbol}: {e}. Retrying... (Attempt {attempts+1})")
        except ccxt.RateLimitExceeded as e:
            wait_match = re.search(r'(\d+)\s*(?:ms|s)', str(e).lower())
            wait_time = int(wait_match.group(1))/1000 if wait_match and 'ms' in wait_match.group(0) else int(wait_match.group(1)) if wait_match else RETRY_DELAY_SECONDS*(attempts+2)
            lg.warning(f"Rate limit orderbook {symbol}. Retrying in {wait_time}s... (Attempt {attempts+1})")
            time.sleep(wait_time + 1)
            attempts += 1; continue # Skip standard delay
        except ccxt.ExchangeError as e: lg.error(f"{COLOR_ERROR}Exchange error orderbook {symbol}: {e}{RESET}"); return None
        except Exception as e: lg.error(f"{COLOR_ERROR}Unexpected error orderbook {symbol}: {e}{RESET}", exc_info=True); return None
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS)
    lg.error(f"{COLOR_ERROR}Max retries fetching orderbook {symbol}.{RESET}")
    return None

# --- Volbot Strategy Calculation Functions (from volbot5) ---
def ema_swma(series: pd.Series, length: int, logger: logging.Logger) -> pd.Series:
    """Calculates Smoothed Weighted Moving Average (EMA of weighted last 4 values)."""
    lg = logger; lg.debug(f"Calculating SWMA length: {length}...")
    required_periods = 4
    numeric_series = pd.to_numeric(series, errors='coerce')
    if len(numeric_series) < required_periods:
        lg.warning(f"Series len {len(numeric_series)} < {required_periods}. SWMA needs {required_periods}. Returning standard EMA.")
        ema_result = ta.ema(numeric_series, length=length, adjust=False)
        return ema_result if isinstance(ema_result, pd.Series) else pd.Series(ema_result, index=series.index)
    w0 = numeric_series.fillna(0) / 6; w1 = numeric_series.shift(1).fillna(0) * 2 / 6
    w2 = numeric_series.shift(2).fillna(0) * 2 / 6; w3 = numeric_series.shift(3).fillna(0) * 1 / 6
    weighted_series = w0 + w1 + w2 + w3
    weighted_series[numeric_series.isna()] = np.nan
    weighted_series.iloc[:required_periods-1] = np.nan
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length, adjust=False)
    result_series = smoothed_ema.reindex(series.index)
    lg.debug(f"SWMA calculation finished. Result NaNs: {result_series.isna().sum()}")
    return result_series

def calculate_volatility_levels(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Calculates Volumatic Trend indicators: EMAs, ATR, dynamic levels, volume metrics."""
    lg = logger; lg.info("Calculating Volumatic Trend Levels...")
    length = config.get("volbot_length", DEFAULT_VOLBOT_LENGTH)
    atr_length = config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH)
    volume_lookback = config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK)
    min_len = max(length + 3, atr_length, volume_lookback) + 10
    if len(df) < min_len:
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)}) for Volumatic calc (min ~{min_len}).{RESET}")
        # Add placeholder columns
        cols = ['ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat', 'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat', 'vol_dn_step_strat', 'vol_trend_up_level_strat', 'vol_trend_dn_level_strat', 'volume_delta_strat', 'volume_total_strat', 'cum_vol_delta_since_change_strat', 'cum_vol_total_since_change_strat', 'last_trend_change_idx']
        for col in cols: df[col] = np.nan
        return df
    df_calc = df.copy()
    try:
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols: df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
        df_calc['volume_float'] = pd.to_numeric(df_calc['volume'], errors='coerce').fillna(0)

        df_calc['ema1_strat'] = ema_swma(df_calc['close'], length, lg)
        df_calc['ema2_strat'] = ta.ema(df_calc['close'], length=length, adjust=False)
        df_calc['atr_strat'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=atr_length)
        df_calc.dropna(subset=['ema1_strat', 'ema2_strat', 'atr_strat'], inplace=True)
        if df_calc.empty: lg.warning("DF empty after dropping initial NaNs."); return df

        df_calc['trend_up_strat'] = (df_calc['ema1_strat'] > df_calc['ema2_strat']).astype('boolean')
        df_calc['trend_changed_strat'] = df_calc['trend_up_strat'].diff().fillna(False)
        level_cols = ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat']
        for col in level_cols: df_calc[col] = np.nan

        change_indices = df_calc.index[df_calc['trend_changed_strat']]
        if not change_indices.empty:
            ema1_prev = pd.to_numeric(df_calc['ema1_strat'].shift(1), errors='coerce').loc[change_indices]
            atr_prev = pd.to_numeric(df_calc['atr_strat'].shift(1), errors='coerce').loc[change_indices]
            valid_mask = pd.notna(ema1_prev) & pd.notna(atr_prev) & (atr_prev > 0)
            valid_indices = change_indices[valid_mask]
            if not valid_indices.empty:
                valid_ema1 = ema1_prev[valid_mask]; valid_atr = atr_prev[valid_mask]
                upper = valid_ema1 + valid_atr * 3.0; lower = valid_ema1 - valid_atr * 3.0
                lower_vol = lower + valid_atr * 4.0; upper_vol = upper - valid_atr * 4.0
                step_up = (lower_vol - lower).clip(lower=0.0) / 100.0; step_dn = (upper - upper_vol).clip(lower=0.0) / 100.0
                df_calc.loc[valid_indices, 'upper_strat'] = upper; df_calc.loc[valid_indices, 'lower_strat'] = lower
                df_calc.loc[valid_indices, 'lower_vol_strat'] = lower_vol; df_calc.loc[valid_indices, 'upper_vol_strat'] = upper_vol
                df_calc.loc[valid_indices, 'step_up_strat'] = step_up; df_calc.loc[valid_indices, 'step_dn_strat'] = step_dn
        for col in level_cols: df_calc[col] = df_calc[col].ffill()

        max_vol_lookback = df_calc['volume_float'].rolling(window=volume_lookback, min_periods=max(1, volume_lookback // 10)).max()
        df_calc['vol_norm_strat'] = np.where(pd.notna(max_vol_lookback) & (max_vol_lookback > 1e-9), (df_calc['volume_float'].fillna(0) / max_vol_lookback * 100.0), 0.0).clip(0.0, 100.0)
        df_calc['vol_up_step_strat'] = (df_calc['step_up_strat'].fillna(0.0) * df_calc['vol_norm_strat'].fillna(0.0))
        df_calc['vol_dn_step_strat'] = (df_calc['step_dn_strat'].fillna(0.0) * df_calc['vol_norm_strat'].fillna(0.0))
        df_calc['vol_trend_up_level_strat'] = df_calc['lower_strat'].fillna(0.0) + df_calc['vol_up_step_strat'].fillna(0.0)
        df_calc['vol_trend_dn_level_strat'] = df_calc['upper_strat'].fillna(0.0) - df_calc['vol_dn_step_strat'].fillna(0.0)

        df_calc['volume_delta_float'] = np.where(df_calc['close'] > df_calc['open'], df_calc['volume_float'], np.where(df_calc['close'] < df_calc['open'], -df_calc['volume_float'], 0.0)).fillna(0.0)
        df_calc['volume_total_float'] = df_calc['volume_float'].fillna(0.0)
        trend_block_group = df_calc['trend_changed_strat'].cumsum()
        df_calc['cum_vol_delta_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_delta_float'].cumsum()
        df_calc['cum_vol_total_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_total_float'].cumsum()
        last_change_ts = df_calc.index.to_series().where(df_calc['trend_changed_strat']).ffill()
        df_calc['last_trend_change_idx'] = last_change_ts

        df_calc.drop(columns=['volume_float', 'volume_delta_float', 'volume_total_float'], inplace=True, errors='ignore')
        lg.info("Volumatic Trend Levels calculation complete.")
        return df_calc
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Volumatic Trend calculation: {e}{RESET}", exc_info=True)
        return df

def calculate_pivot_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Identifies Pivot High (PH) and Pivot Low (PL) points for Order Block detection."""
    lg = logger
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    left_h = config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    left_l = config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    lg.info(f"Calculating Pivots (Source: {source}, L/R H: {left_h}/{right_h}, L: {left_l}/{right_l})...")
    min_len = max(left_h + right_h + 1, left_l + right_l + 1)
    if len(df) < min_len:
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)}) for Pivots (min ~{min_len}). Skipping.{RESET}")
        df['ph_strat'] = pd.NA; df['pl_strat'] = pd.NA; return df
    df_calc = df.copy()
    try:
        high_src = 'high' if source == "Wicks" else 'close'
        low_src = 'low' if source == "Wicks" else 'open'
        if high_src not in df_calc or low_src not in df_calc:
             lg.error(f"Missing pivot source columns {high_src}/{low_src}. Skipping."); df['ph_strat'] = pd.NA; df['pl_strat'] = pd.NA; return df
        df_calc[f'{high_src}_dec'] = df_calc[high_src].apply(lambda x: safe_decimal(x, default=pd.NA))
        df_calc[f'{low_src}_dec'] = df_calc[low_src].apply(lambda x: safe_decimal(x, default=pd.NA))
        high_col_dec = df_calc[f'{high_src}_dec']; low_col_dec = df_calc[f'{low_src}_dec']
        df_calc['ph_strat'] = pd.NA; df_calc['pl_strat'] = pd.NA

        # Using simple loop for clarity and Decimal compatibility
        for i in range(left_h, len(df_calc) - right_h):
             pivot_val = high_col_dec.iloc[i]; if pd.isna(pivot_val): continue
             left_vals = high_col_dec.iloc[i-left_h : i]; right_vals = high_col_dec.iloc[i+1 : i+right_h+1]
             if not left_vals.isna().any() and (pivot_val > left_vals).all() and \
                not right_vals.isna().any() and (pivot_val > right_vals).all():
                  df_calc.loc[df_calc.index[i], 'ph_strat'] = pivot_val
        for i in range(left_l, len(df_calc) - right_l):
             pivot_val = low_col_dec.iloc[i]; if pd.isna(pivot_val): continue
             left_vals = low_col_dec.iloc[i-left_l : i]; right_vals = low_col_dec.iloc[i+1 : i+right_l+1]
             if not left_vals.isna().any() and (pivot_val < left_vals).all() and \
                not right_vals.isna().any() and (pivot_val < right_vals).all():
                  df_calc.loc[df_calc.index[i], 'pl_strat'] = pivot_val

        df_calc.drop(columns=[f'{high_src}_dec', f'{low_src}_dec'], inplace=True, errors='ignore')
        df_calc['ph_strat'] = df_calc['ph_strat'].astype(object); df_calc['pl_strat'] = df_calc['pl_strat'].astype(object)
        lg.info(f"Pivots calculated. Found {df_calc['ph_strat'].notna().sum()} PH, {df_calc['pl_strat'].notna().sum()} PL.")
        return df_calc
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error calculating Pivots: {e}{RESET}", exc_info=True)
        df['ph_strat'] = pd.NA; df['pl_strat'] = pd.NA; return df

def manage_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """Identifies, creates, and manages the state of Order Blocks (OBs)."""
    lg = logger; lg.info("Managing Order Block Boxes...")
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    offset_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    offset_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    max_boxes = config.get("volbot_max_boxes", DEFAULT_VOLBOT_MAX_BOXES)
    df_calc = df.copy(); bull_boxes = []; bear_boxes = []
    active_bull_boxes = []; active_bear_boxes = []; box_counter = 0
    if 'ph_strat' not in df_calc or 'pl_strat' not in df_calc:
        lg.warning("Pivot columns missing. Skipping OBs."); df_calc['active_bull_ob_strat'] = None; df_calc['active_bear_ob_strat'] = None; return df_calc, [], []
    df_calc['active_bull_ob_strat'] = pd.Series(dtype='object'); df_calc['active_bear_ob_strat'] = pd.Series(dtype='object')
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_calc: df_calc[f'{col}_dec'] = df_calc[col].apply(lambda x: safe_decimal(x, default=pd.NA))
        else: lg.error(f"Missing column '{col}'. Skipping OBs."); return df_calc, [], []
    try:
        for i in range(len(df_calc)):
            idx = df_calc.index[i]; close_dec = df_calc.at[idx, 'close_dec']; if pd.isna(close_dec): continue
            next_active_bull = []; active_bull_ref = None
            for box in active_bull_boxes:
                if close_dec < box['bottom']: box['state'] = 'closed'; box['end_idx'] = idx; lg.debug(f"Closed Bull OB: {box['id']} at {idx}")
                else: next_active_bull.append(box);
                if box['state'] == 'active' and box['bottom'] <= close_dec <= box['top']: active_bull_ref = box
            active_bull_boxes = next_active_bull
            next_active_bear = []; active_bear_ref = None
            for box in active_bear_boxes:
                if close_dec > box['top']: box['state'] = 'closed'; box['end_idx'] = idx; lg.debug(f"Closed Bear OB: {box['id']} at {idx}")
                else: next_active_bear.append(box)
                if box['state'] == 'active' and box['bottom'] <= close_dec <= box['top']: active_bear_ref = box
            active_bear_boxes = next_active_bear
            df_calc.loc[idx, 'active_bull_ob_strat'] = active_bull_ref; df_calc.loc[idx, 'active_bear_ob_strat'] = active_bear_ref

            pivot_h = df_calc.at[idx, 'ph_strat']
            if pd.notna(pivot_h):
                ob_iloc = i - offset_h;
                if ob_iloc >= 0:
                    ob_idx = df_calc.index[ob_iloc]; top_p, bot_p = pd.NA, pd.NA
                    if source == "Bodys": top_p, bot_p = df_calc.at[ob_idx, 'open_dec'], df_calc.at[ob_idx, 'close_dec']
                    else:                 top_p, bot_p = df_calc.at[ob_idx, 'high_dec'], df_calc.at[ob_idx, 'close_dec']
                    if pd.notna(top_p) and pd.notna(bot_p):
                        top_price, bot_price = max(top_p, bot_p), min(top_p, bot_p)
                        if top_price > bot_price:
                            box_counter += 1; new_box = {'id':f'BearOB_{box_counter}','type':'bear','start_idx':ob_idx,'pivot_idx':idx,'end_idx':None,'top':top_price,'bottom':bot_price,'state':'active'}
                            bear_boxes.append(new_box); active_bear_boxes.append(new_box); lg.debug(f"Created Bear OB: {new_box['id']}")
            pivot_l = df_calc.at[idx, 'pl_strat']
            if pd.notna(pivot_l):
                ob_iloc = i - offset_l;
                if ob_iloc >= 0:
                    ob_idx = df_calc.index[ob_iloc]; top_p, bot_p = pd.NA, pd.NA
                    if source == "Bodys": top_p, bot_p = df_calc.at[ob_idx, 'close_dec'], df_calc.at[ob_idx, 'open_dec']
                    else:                 top_p, bot_p = df_calc.at[ob_idx, 'open_dec'], df_calc.at[ob_idx, 'low_dec']
                    if pd.notna(top_p) and pd.notna(bot_p):
                        top_price, bot_price = max(top_p, bot_p), min(top_p, bot_p)
                        if top_price > bot_price:
                             box_counter += 1; new_box = {'id':f'BullOB_{box_counter}','type':'bull','start_idx':ob_idx,'pivot_idx':idx,'end_idx':None,'top':top_price,'bottom':bot_price,'state':'active'}
                             bull_boxes.append(new_box); active_bull_boxes.append(new_box); lg.debug(f"Created Bull OB: {new_box['id']}")
            if len(active_bull_boxes) > max_boxes:
                active_bull_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True); removed = active_bull_boxes[max_boxes:]; active_bull_boxes = active_bull_boxes[:max_boxes]
                for box in removed: box['state'] = 'trimmed'; lg.debug(f"Trimmed Bull OB: {box['id']}")
            if len(active_bear_boxes) > max_boxes:
                active_bear_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True); removed = active_bear_boxes[max_boxes:]; active_bear_boxes = active_bear_boxes[:max_boxes]
                for box in removed: box['state'] = 'trimmed'; lg.debug(f"Trimmed Bear OB: {box['id']}")
        for col in price_cols: df_calc.drop(columns=[f'{col}_dec'], inplace=True, errors='ignore')
        num_active_bull = sum(1 for b in active_bull_boxes); num_active_bear = sum(1 for b in active_bear_boxes);
        lg.info(f"OB management complete. Total: {len(bull_boxes)} Bull, {len(bear_boxes)} Bear. Active: {num_active_bull} Bull, {num_active_bear} Bear.")
        return df_calc, bull_boxes, bear_boxes
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during OB management: {e}{RESET}", exc_info=True)
        df['active_bull_ob_strat'] = None; df['active_bear_ob_strat'] = None; return df, [], []

# --- Trading Analyzer Class (Merged) ---
class TradingAnalyzer:
    """Analyzes data using LiveXY and/or Volbot, generates signals, manages risk."""
    def __init__(
        self, df_raw: pd.DataFrame, logger: logging.Logger, config: Dict[str, Any],
        market_info: Dict[str, Any], orderbook_data: Optional[Dict] = None
    ) -> None:
        self.df_raw = df_raw
        self.df_processed = pd.DataFrame()
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.orderbook_data = orderbook_data # Store fetched orderbook data
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN")
        self.min_tick_size = self._determine_min_tick_size()
        self.price_precision = self._determine_price_precision()
        self.amount_precision = self._determine_amount_precision()
        self.amount_step_size = self._determine_amount_step_size()

        # LiveXY specific state
        self.livexy_enabled = config.get("livexy_enabled", True)
        self.livexy_indicator_values: Dict[str, float] = {} # Stores latest LiveXY float values
        self.livexy_ta_column_names: Dict[str, Optional[str]] = {} # Stores pandas_ta column names for LiveXY
        self.livexy_active_weight_set_name = config.get("active_weight_set", "default")
        self.livexy_weights = config.get("weight_sets",{}).get(self.livexy_active_weight_set_name, {})

        # Volbot specific state
        self.volbot_enabled = config.get("volbot_enabled", True)
        self.all_bull_boxes: List[Dict] = []
        self.all_bear_boxes: List[Dict] = []
        self.latest_active_bull_ob: Optional[Dict] = None
        self.latest_active_bear_ob: Optional[Dict] = None

        # Combined State
        self.strategy_state: Dict[str, Any] = {} # Holds latest values (Decimal where possible) from all enabled strategies

        # Initial setup
        self.logger.debug(f"Analyzer initialized for {self.symbol}: Tick={self.min_tick_size}, PricePrec={self.price_precision}, AmtPrec={self.amount_precision}, AmtStep={self.amount_step_size}")
        self._calculate_indicators()
        self._update_latest_strategy_state() # Update state after calculations

    # --- Precision/Rounding Helpers ---
    def _determine_min_tick_size(self) -> Decimal:
        try:
            price_prec_val = self.market_info.get('precision', {}).get('price'); tick = safe_decimal(price_prec_val)
            if tick and tick > 0: return tick
            min_price_limit = self.market_info.get('limits', {}).get('price', {}).get('min'); tick = safe_decimal(min_price_limit)
            if tick and tick > 0 and tick < 10: return tick # Heuristic: check if it looks like tick size
        except Exception as e: self.logger.warning(f"Could not determine tick size for {self.symbol}: {e}. Using fallback.")
        last_price = safe_decimal(self.df_raw['close'].iloc[-1]) if not self.df_raw.empty and 'close' in self.df_raw.columns else None
        if last_price:
            if last_price > 1000: default_tick = Decimal('0.1'); elif last_price > 10: default_tick = Decimal('0.01');
            elif last_price > 0.1: default_tick = Decimal('0.001'); else: default_tick = Decimal('0.00001');
        else: default_tick = Decimal('0.0001')
        self.logger.warning(f"Using default/fallback tick size {default_tick} for {self.symbol}.")
        return default_tick

    def _determine_price_precision(self) -> int:
        try:
            if self.min_tick_size > 0: return abs(self.min_tick_size.normalize().as_tuple().exponent)
        except Exception: pass; return 4 # Default

    def _determine_amount_step_size(self) -> Decimal:
        try:
            amount_prec_val = self.market_info.get('precision', {}).get('amount'); step_size = safe_decimal(amount_prec_val)
            if step_size and step_size > 0: return step_size
            if isinstance(amount_prec_val, int) and amount_prec_val >= 0: return Decimal('1') / (Decimal('10') ** amount_prec_val)
            min_amount_limit = self.market_info.get('limits', {}).get('amount', {}).get('min'); step_size = safe_decimal(min_amount_limit)
            if step_size and step_size > 0 and step_size <= 1: return step_size # Heuristic
        except Exception as e: self.logger.warning(f"Could not determine amount step size {self.symbol}: {e}.")
        return Decimal('0.00000001') # Default

    def _determine_amount_precision(self) -> int:
        try:
            if self.amount_step_size > 0: return abs(self.amount_step_size.normalize().as_tuple().exponent)
        except Exception: pass; return 8 # Default

    def get_price_precision(self) -> int: return self.price_precision
    def get_amount_precision(self) -> int: return self.amount_precision
    def get_min_tick_size(self) -> Decimal: return self.min_tick_size
    def get_amount_step_size(self) -> Decimal: return self.amount_step_size

    def round_price(self, price: Union[Decimal, float, str, None], rounding_mode=ROUND_HALF_UP) -> Optional[Decimal]:
        """Rounds price to tick size using specified mode."""
        price_dec = safe_decimal(price); min_tick = self.min_tick_size
        if price_dec is None or min_tick is None or min_tick <= 0: return None
        try: return (price_dec / min_tick).quantize(Decimal('1'), rounding=rounding_mode) * min_tick
        except Exception as e: self.logger.error(f"Error rounding price {price_dec}: {e}"); return None

    def round_amount(self, amount: Union[Decimal, float, str, None]) -> Optional[Decimal]:
        """Rounds (floors) amount DOWN to step size."""
        amount_dec = safe_decimal(amount); step_size = self.amount_step_size
        if amount_dec is None or step_size is None or step_size <= 0: return None
        try:
            if amount_dec < 0: self.logger.warning(f"Rounding negative amount: {amount_dec}."); # TODO: Define behavior for negative amounts if needed
            rounded_amount = (amount_dec // step_size) * step_size # Floor division
            return rounded_amount
        except Exception as e: self.logger.error(f"Error rounding amount {amount_dec}: {e}"); return None

    # --- LiveXY Indicator Calculation Helpers ---
    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> Optional[str]:
         """Helper to find pandas_ta column name."""
         cfg = self.config; patterns = []
         # Define expected patterns (add more as needed)
         epatterns = {
             "ATR": [f"ATRr_{cfg.get('livexy_atr_period', DEFAULT_ATR_PERIOD_LIVEXY)}"],
             "EMA_Short": [f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"],
             "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
             "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
             # Add other LiveXY indicators here...
             "StochRSI_K": [f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
             "StochRSI_D": [f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"],
             # ... other indicators ...
         }
         patterns = epatterns.get(base_name, [])
         for pattern in patterns:
             if pattern in result_df.columns: return pattern
             # Fallback search (less reliable)
             base_pattern = pattern.split('_')[0]
             if base_pattern in result_df.columns: return base_pattern
             if base_name.upper() in result_df.columns: return base_name.upper()
         for col in result_df.columns:
             if col.lower().startswith(base_name.lower() + "_"): return col
             if base_name.lower() in col.lower(): return col
         self.logger.warning(f"Could not find col name for indicator '{base_name}' in {result_df.columns.tolist()}")
         return None

    def _calculate_livexy_indicators(self, df_calc: pd.DataFrame) -> pd.DataFrame:
        """Calculates LiveXY indicators using pandas_ta."""
        if not self.livexy_enabled: return df_calc
        self.logger.info("Calculating LiveXY strategy indicators...")
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        self.livexy_ta_column_names = {} # Reset mappings

        try:
            # --- Calculate indicators based on config flags ---
            if True: # Example: Always calc ATR for LiveXY?
                atr_period = cfg.get("livexy_atr_period", DEFAULT_ATR_PERIOD_LIVEXY)
                df_calc.ta.atr(length=atr_period, append=True)
                self.livexy_ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            if indi_cfg.get("ema_alignment"):
                df_calc.ta.ema(length=cfg.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD), append=True)
                self.livexy_ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
                df_calc.ta.ema(length=cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD), append=True)
                self.livexy_ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)

            if indi_cfg.get("rsi"):
                df_calc.ta.rsi(length=cfg.get("rsi_period", DEFAULT_RSI_WINDOW), append=True)
                self.livexy_ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            # --- Calculate StochRSI ---
            if indi_cfg.get("stoch_rsi"):
                stochrsi_result = df_calc.ta.stochrsi(
                    length=cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW),
                    rsi_length=cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW),
                    k=cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW),
                    d=cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)
                )
                if stochrsi_result is not None and not stochrsi_result.empty:
                    for col in stochrsi_result.columns:
                        if col not in df_calc.columns: df_calc[col] = stochrsi_result[col]
                        else: df_calc[col] = stochrsi_result[col] # Overwrite if exists
                    self.livexy_ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
                    self.livexy_ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)
                else: self.logger.warning(f"StochRSI calculation empty for {self.symbol}.")

            # Add calculations for ALL other enabled indicators (momentum, cci, wr, mfi, vwap, psar, sma10, bbands, volume_ma)
            # ... (omitted for brevity, but follow the pattern above) ...

            self.logger.info("LiveXY indicator calculations complete.")

        except AttributeError as e: self.logger.error(f"AttributeError calculating LiveXY indicators for {self.symbol} (check pandas_ta method/version?): {e}", exc_info=True)
        except Exception as e: self.logger.error(f"Error calculating LiveXY indicators for {self.symbol}: {e}", exc_info=True)
        return df_calc

    # --- Combined Indicator Calculation ---
    def _calculate_indicators(self) -> None:
        """Calculates all enabled indicators (Risk ATR, LiveXY, Volbot)."""
        if self.df_raw.empty:
            self.logger.warning(f"{COLOR_WARNING}Raw DF empty, cannot calculate indicators.{RESET}")
            self.df_processed = pd.DataFrame()
            return

        # Minimum length check (simplified, individual functions also check)
        if len(self.df_raw) < 50:
             self.logger.warning(f"{COLOR_WARNING}Insufficient raw data ({len(self.df_raw)} points) for reliable analysis.{RESET}")
             # Continue, but expect NaNs

        try:
            df_calc = self.df_raw.copy()

            # --- Ensure Base Numeric Columns ---
            price_cols = ['open', 'high', 'low', 'close']; vol_col = 'volume'
            for col in price_cols + [vol_col]:
                if col not in df_calc.columns:
                    self.logger.error(f"Missing required column '{col}'. Cannot calculate indicators."); self.df_processed = pd.DataFrame(); return
                # Convert to numeric for calculations, but keep original Decimal if it exists
                if not isinstance(df_calc[col].dtype, (np.number, np.bool_)): # Check if not already numeric or boolean
                    # Check if original IS Decimal before converting
                    is_decimal = all(isinstance(v, Decimal) for v in df_calc[col].dropna())
                    if not is_decimal:
                         df_calc[f'{col}_float'] = pd.to_numeric(df_calc[col], errors='coerce')
                         self.logger.debug(f"Created float version for column '{col}' for calculations.")
                    else:
                         # Use original Decimal column if it's already Decimal (less common for raw data)
                         # TA functions might still require float, handled internally where needed
                         pass
            # Ensure essential price columns are present as float if needed
            for col in price_cols:
                if f'{col}_float' not in df_calc and not all(isinstance(v, Decimal) for v in df_calc[col].dropna()):
                     df_calc[f'{col}_float'] = pd.to_numeric(df_calc[col], errors='coerce')
                if col == 'volume' and f'{col}_float' not in df_calc and not all(isinstance(v, Decimal) for v in df_calc[col].dropna()):
                     df_calc[f'{col}_float'] = pd.to_numeric(df_calc[col], errors='coerce')

            # Prefer float columns for TA library functions
            high_col = 'high_float' if 'high_float' in df_calc else 'high'
            low_col = 'low_float' if 'low_float' in df_calc else 'low'
            close_col = 'close_float' if 'close_float' in df_calc else 'close'
            volume_col = 'volume_float' if 'volume_float' in df_calc else 'volume'


            # --- Calculate Risk Management ATR ---
            atr_period_risk = self.config.get("atr_period_risk", DEFAULT_ATR_PERIOD_RISK)
            df_calc['atr_risk'] = ta.atr(df_calc[high_col], df_calc[low_col], df_calc[close_col], length=atr_period_risk)
            df_calc['atr_risk_dec'] = df_calc['atr_risk'].apply(lambda x: safe_decimal(x, default=pd.NA)) # Store Decimal version
            self.logger.debug(f"Calculated Risk ATR (Length: {atr_period_risk})")

            # --- Calculate LiveXY Strategy Indicators ---
            if self.livexy_enabled:
                df_calc = self._calculate_livexy_indicators(df_calc)

            # --- Calculate Volbot Strategy Indicators ---
            if self.volbot_enabled:
                df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
                df_calc = calculate_pivot_order_blocks(df_calc, self.config, self.logger)
                df_calc, self.all_bull_boxes, self.all_bear_boxes = manage_order_blocks(df_calc, self.config, self.logger)

            # --- Cleanup Temporary Float Columns ---
            temp_float_cols = [f'{col}_float' for col in price_cols + [vol_col] if f'{col}_float' in df_calc]
            df_calc.drop(columns=temp_float_cols, inplace=True, errors='ignore')

            self.df_processed = df_calc
            self.logger.debug(f"All indicator calculations complete. Processed DF has {len(self.df_processed)} rows.")

        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Error during combined indicator calculation: {e}{RESET}", exc_info=True)
            self.df_processed = pd.DataFrame()

    # --- LiveXY Signal Generation Helpers ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score based on LiveXY indicators."""
        ema_short = self.livexy_indicator_values.get("EMA_Short", np.nan)
        ema_long = self.livexy_indicator_values.get("EMA_Long", np.nan)
        current_price = self.livexy_indicator_values.get("Close", np.nan) # Use Close from LiveXY context
        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price): return np.nan
        if current_price > ema_short > ema_long: return 1.0
        elif current_price < ema_short < ema_long: return -1.0
        else: return 0.0

    def _check_livexy_indicator(self, indicator_key: str) -> float:
         """Calls the specific check method for a LiveXY indicator."""
         # This function dispatches to individual _check_* methods (like in livebot7)
         # Each _check_* method needs access to self.livexy_indicator_values
         # Example for StochRSI:
         if indicator_key == "stoch_rsi":
             k = self.livexy_indicator_values.get("StochRSI_K", np.nan)
             d = self.livexy_indicator_values.get("StochRSI_D", np.nan)
             if pd.isna(k) or pd.isna(d): return np.nan
             oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
             overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
             score = 0.0
             if k < oversold and d < oversold: score = 1.0
             elif k > overbought and d > overbought: score = -1.0
             diff = k - d
             if abs(diff) > 5: score = max(score, 0.6) if diff > 0 else min(score, -0.6)
             else: score = max(score, 0.2) if k > d else min(score, -0.2) if k < d else score
             return max(-1.0, min(1.0, score)) # Clamp
         elif indicator_key == "ema_alignment":
             return self.calculate_ema_alignment_score()
         elif indicator_key == "rsi":
             rsi = self.livexy_indicator_values.get("RSI", np.nan)
             if pd.isna(rsi): return np.nan
             if rsi <= 30: return 1.0;
             if rsi >= 70: return -1.0;
             if rsi < 40: return 0.5;
             if rsi > 60: return -0.5;
             if 40 <= rsi <= 60: return 0.5 - (rsi - 40) * (1.0 / 20.0)
             return 0.0
         elif indicator_key == "orderbook":
            if not self.orderbook_data: return np.nan
            try:
                bids = self.orderbook_data.get('bids', []); asks = self.orderbook_data.get('asks', [])
                if not bids or not asks: return np.nan
                n = 10 # Check top N levels
                bid_vol = sum(safe_decimal(bid[1], 0) for bid in bids[:n])
                ask_vol = sum(safe_decimal(ask[1], 0) for ask in asks[:n])
                total_vol = bid_vol + ask_vol
                obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else Decimal(0)
                return float(obi) # Score is OBI
            except Exception as e: self.logger.warning(f"Orderbook analysis failed: {e}"); return np.nan

         # Add other _check_* methods for all indicators in config['indicators']...
         # ... (omitted for brevity) ...

         # Fallback if check method not implemented for an enabled indicator
         self.logger.warning(f"No specific check logic implemented for LiveXY indicator: {indicator_key}")
         return np.nan

    def _generate_livexy_signal(self) -> Tuple[str, float]:
        """Generates BUY/SELL/HOLD signal based on weighted LiveXY indicator scores."""
        if not self.livexy_enabled: return "HOLD", 0.0
        signal = "HOLD"; final_score = Decimal("0.0"); total_weight = Decimal("0.0")
        current_price = safe_decimal(self.strategy_state.get('close')) # Use latest close from combined state
        if current_price is None: self.logger.warning("LiveXY signal: HOLD (Missing current price)"); return "HOLD", 0.0

        # --- Ensure livexy_indicator_values are populated ---
        # This part is called before the combined _update... method, needs separate update
        if self.df_processed.empty: return "HOLD", 0.0
        try:
            latest = self.df_processed.iloc[-1]
            temp_livexy_vals = {}
            for key, col_name in self.livexy_ta_column_names.items():
                if col_name and col_name in latest.index and pd.notna(latest[col_name]):
                    try: temp_livexy_vals[key] = float(latest[col_name])
                    except (ValueError, TypeError): temp_livexy_vals[key] = np.nan
                else: temp_livexy_vals[key] = np.nan
            # Add core values needed by checks
            for k in ['Close', 'Volume', 'High', 'Low']: temp_livexy_vals[k] = safe_decimal(latest.get(k.lower()), np.nan)
            self.livexy_indicator_values = temp_livexy_vals # Update internal state for LiveXY checks
        except Exception as e: self.logger.error(f"Failed to update livexy_indicator_values: {e}"); return "HOLD", 0.0
        # --- End update ----

        if not self.livexy_weights: self.logger.error("LiveXY weights not found/empty."); return "HOLD", 0.0
        debug_scores = {}
        for key, enabled in self.config.get("indicators", {}).items():
            if not enabled: continue
            weight_str = self.livexy_weights.get(key)
            if weight_str is None: continue
            weight = safe_decimal(weight_str, 0)
            if weight == 0: continue
            score_float = self._check_livexy_indicator(key) # Calls _check_* methods
            debug_scores[key] = f"{score_float:.2f}" if pd.notna(score_float) else "NaN"
            if pd.notna(score_float):
                 clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), safe_decimal(score_float, 0)))
                 final_score += clamped_score * weight
                 total_weight += weight

        threshold = safe_decimal(self.config.get("signal_score_threshold", 1.5), Decimal(1.5))
        if total_weight > 0:
            # normalized_score = final_score / total_weight # Optional normalization
            if final_score >= threshold: signal = "BUY"
            elif final_score <= -threshold: signal = "SELL"
        else: self.logger.warning("LiveXY: No indicators contributed to score.")

        score_details = ", ".join([f"{k}:{v}" for k, v in debug_scores.items()])
        self.logger.debug(f"LiveXY Signal Calc: Score={final_score:.4f}, Wght={total_weight:.2f}, Thresh={threshold}, Signal={signal}")
        if console_log_level <= logging.DEBUG: self.logger.debug(f"  LiveXY Scores: {score_details}")
        return signal, float(final_score) # Return signal and raw score

    # --- Volbot Signal Generation Helper ---
    def _generate_volbot_signal(self) -> str:
        """Generates BUY/SELL/HOLD signal based on Volbot rules."""
        if not self.volbot_enabled: return "HOLD"
        signal = "HOLD"
        try:
            is_trend_up = self.strategy_state.get('trend_up_strat'); trend_changed = self.strategy_state.get('trend_changed_strat', False)
            is_in_bull_ob = self.strategy_state.get('is_in_active_bull_ob', False); is_in_bear_ob = self.strategy_state.get('is_in_active_bear_ob', False)
            signal_on_flip = self.config.get("volbot_signal_on_trend_flip", True); signal_on_ob = self.config.get("volbot_signal_on_ob_entry", True)
            if is_trend_up is None: self.logger.debug("Volbot signal: HOLD (Trend undetermined)"); return "HOLD"
            trend_str = f"{COLOR_UP}UP{RESET}" if is_trend_up else f"{COLOR_DN}DOWN{RESET}"
            ob_status = f"{' '+COLOR_BULL_BOX if is_in_bull_ob else ''}{'InBull' if is_in_bull_ob else ''}" \
                        f"{RESET if is_in_bull_ob else ''}{' '+COLOR_BEAR_BOX if is_in_bear_ob else ''}" \
                        f"{'InBear' if is_in_bear_ob else ''}{RESET if is_in_bear_ob else ''}" \
                        f"{' NoActiveOB' if not is_in_bull_ob and not is_in_bear_ob else ''}"
            if signal_on_flip and trend_changed:
                signal = "BUY" if is_trend_up else "SELL"; reason = f"Trend flipped to {trend_str}"
                color = COLOR_UP if is_trend_up else COLOR_DN
                self.logger.debug(f"{color}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                return signal
            if signal_on_ob:
                if is_trend_up and is_in_bull_ob:
                    signal = "BUY"; ob_id = self.latest_active_bull_ob.get('id','N/A') if self.latest_active_bull_ob else 'N/A'; reason = f"Price in Bull OB '{ob_id}' ({trend_str})"
                    self.logger.debug(f"{COLOR_BULL_BOX}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                    return signal
                elif not is_trend_up and is_in_bear_ob:
                    signal = "SELL"; ob_id = self.latest_active_bear_ob.get('id','N/A') if self.latest_active_bear_ob else 'N/A'; reason = f"Price in Bear OB '{ob_id}' ({trend_str})"
                    self.logger.debug(f"{COLOR_BEAR_BOX}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                    return signal
            self.logger.debug(f"Volbot Signal: HOLD (Conditions: Trend={trend_str},{ob_status})")
        except KeyError as e: self.logger.error(f"{COLOR_ERROR}Volbot Signal Error: Missing key '{e}'.{RESET}"); return "HOLD"
        except Exception as e: self.logger.error(f"{COLOR_ERROR}Volbot Signal Error: {e}{RESET}", exc_info=True); return "HOLD"
        return signal

    # --- Combined Signal Generation ---
    def generate_trading_signal(self) -> str:
        """Generates final signal based on enabled strategies and configured mode."""
        final_signal = "HOLD"
        livexy_signal, livexy_score = "HOLD", 0.0
        volbot_signal = "HOLD"

        if self.livexy_enabled:
            livexy_signal, livexy_score = self._generate_livexy_signal()
        if self.volbot_enabled:
            volbot_signal = self._generate_volbot_signal()

        mode = self.config.get("signal_mode", "both_align")
        price_fmt = f".{self.price_precision}f"
        current_price = self.strategy_state.get('close', 'N/A')
        price_str = f"{current_price:{price_fmt}}" if isinstance(current_price, Decimal) else str(current_price)

        log_prefix = f"Signal Generation ({self.symbol} @ {price_str}): "
        enabled_str = f"LiveXY={'ON' if self.livexy_enabled else 'OFF'}, Volbot={'ON' if self.volbot_enabled else 'OFF'}, Mode={mode}"

        if not self.livexy_enabled and not self.volbot_enabled:
            self.logger.warning(f"{log_prefix}HOLD (No strategies enabled)")
            return "HOLD"
        elif self.livexy_enabled and not self.volbot_enabled:
            final_signal = livexy_signal
            self.logger.info(f"{log_prefix}Final Signal: {signal_to_color(final_signal)} (Source: LiveXY Score={livexy_score:.2f}) | {enabled_str}")
        elif not self.livexy_enabled and self.volbot_enabled:
            final_signal = volbot_signal
            self.logger.info(f"{log_prefix}Final Signal: {signal_to_color(final_signal)} (Source: Volbot) | {enabled_str}")
        else: # Both enabled
            if mode == 'livexy':
                final_signal = livexy_signal
                self.logger.info(f"{log_prefix}Final Signal: {signal_to_color(final_signal)} (Priority: LiveXY Score={livexy_score:.2f}, Volbot={volbot_signal}) | {enabled_str}")
            elif mode == 'volbot':
                final_signal = volbot_signal
                self.logger.info(f"{log_prefix}Final Signal: {signal_to_color(final_signal)} (Priority: Volbot, LiveXY={livexy_signal}) | {enabled_str}")
            elif mode == 'both_align':
                if livexy_signal == volbot_signal and livexy_signal != "HOLD":
                    final_signal = livexy_signal
                    self.logger.info(f"{log_prefix}Final Signal: {signal_to_color(final_signal)} (Aligned: LiveXY={livexy_signal}/Volbot={volbot_signal}) | {enabled_str}")
                else:
                    final_signal = "HOLD" # Default to HOLD if not aligned
                    reason = "Signals Not Aligned" if livexy_signal != volbot_signal else "Both HOLD"
                    self.logger.info(f"{log_prefix}Final Signal: {signal_to_color(final_signal)} (Reason: {reason}, LiveXY={livexy_signal}, Volbot={volbot_signal}) | {enabled_str}")
            else: # Unknown mode
                self.logger.error(f"Unknown signal_mode '{mode}'. Defaulting to HOLD.")
                final_signal = "HOLD"

        return final_signal

    # --- Combined State Update ---
    def _update_latest_strategy_state(self) -> None:
        """Updates combined strategy_state dict with latest values."""
        self.strategy_state = {} ; self.latest_active_bull_ob = None; self.latest_active_bear_ob = None
        if self.df_processed.empty: self.logger.warning("Cannot update state: Processed DF empty."); return
        try:
            latest = self.df_processed.iloc[-1]
            if latest.isnull().all(): self.logger.warning("Last row all NaNs."); return

            # Define columns to extract and convert to Decimal where appropriate
            cols_to_extract = {
                'open': True, 'high': True, 'low': True, 'close': True, 'volume': True, # Core Data
                'atr_risk_dec': 'atr_risk', # Risk ATR (use pre-calculated Decimal column)
                # LiveXY Indicators (use float values from TA) - Conversion happens in _generate_livexy_signal if needed
                # Volbot Indicators
                'ema1_strat': True, 'ema2_strat': True, 'atr_strat': True, 'trend_up_strat': False, 'trend_changed_strat': False,
                'upper_strat': True, 'lower_strat': True, 'lower_vol_strat': True, 'upper_vol_strat': True,
                'step_up_strat': True, 'step_dn_strat': True, 'vol_norm_strat': False, # Keep norm vol as float
                'vol_up_step_strat': True, 'vol_dn_step_strat': True, 'vol_trend_up_level_strat': True, 'vol_trend_dn_level_strat': True,
                'cum_vol_delta_since_change_strat': True, 'cum_vol_total_since_change_strat': True,
                'last_trend_change_idx': False, # Keep as Timestamp
                'ph_strat': False, 'pl_strat': False, # Keep as Decimal/NA/Object
                'active_bull_ob_strat': False, 'active_bear_ob_strat': False # Keep as Dict/None/Object
            }

            for col, state_key in cols_to_extract.items():
                if col in latest.index and pd.notna(latest[col]):
                    value = latest[col]
                    key_name = state_key if isinstance(state_key, str) else col # Map key name if needed
                    if isinstance(state_key, bool) and state_key: # Convert to Decimal if True
                         self.strategy_state[key_name] = safe_decimal(value, default=None)
                    elif isinstance(state_key, bool) and not state_key: # Keep original type if False
                         self.strategy_state[key_name] = value
                    # Handle case where state_key is str (already mapped)
                    elif isinstance(state_key, str):
                         self.strategy_state[key_name] = value # Assume correct type (Decimal already handled)

            # Update latest OB refs
            self.latest_active_bull_ob = self.strategy_state.get('active_bull_ob_strat')
            self.latest_active_bear_ob = self.strategy_state.get('active_bear_ob_strat')
            self.strategy_state['is_in_active_bull_ob'] = self.latest_active_bull_ob is not None
            self.strategy_state['is_in_active_bear_ob'] = self.latest_active_bear_ob is not None

            # Log state compactly
            log_state = {}; price_fmt = f".{self.price_precision}f"; vol_fmt = ".2f"; atr_fmt = ".5f"
            for k, v in self.strategy_state.items():
                if isinstance(v, Decimal):
                     fmt = price_fmt if any(p in k for p in ['price','level','strat','open','high','low','close','tp','sl','upper','lower']) else vol_fmt if 'vol' in k else atr_fmt if 'atr' in k else ".8f"
                     log_state[k] = f"{v:{fmt}}"
                elif isinstance(v, (bool, pd._libs.missing.NAType, type(None))): log_state[k] = str(v) if pd.notna(v) else 'None'
                elif isinstance(v, pd.Timestamp): log_state[k] = v.strftime('%H:%M:%S')
                elif k not in ['active_bull_ob_strat', 'active_bear_ob_strat']: log_state[k] = v # Avoid logging full OB dicts
            self.logger.debug(f"Latest strategy state updated: {log_state}")
            if self.latest_active_bull_ob: self.logger.debug(f"  Latest Active Bull OB: ID={self.latest_active_bull_ob.get('id')}")
            if self.latest_active_bear_ob: self.logger.debug(f"  Latest Active Bear OB: ID={self.latest_active_bear_ob.get('id')}")
        except IndexError: self.logger.error("Error accessing latest row.")
        except Exception as e: self.logger.error(f"Error updating latest state: {e}", exc_info=True)


    # --- Risk Management Calculation ---
    def calculate_entry_tp_sl(self, entry_price: Decimal, signal: str) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates TP/SL based on entry, signal, Risk ATR, and multiples."""
        if signal not in ["BUY", "SELL"]: return entry_price, None, None
        atr_val = self.strategy_state.get("atr_risk") # Use Risk ATR (Decimal or None)
        tp_mult = safe_decimal(self.config.get("take_profit_multiple", 1.0)); sl_mult = safe_decimal(self.config.get("stop_loss_multiple", 1.5))
        min_tick = self.min_tick_size
        valid = True
        if not (isinstance(entry_price, Decimal) and entry_price > 0): self.logger.error(f"TP/SL Invalid entry: {entry_price}"); valid = False
        if not isinstance(atr_val, Decimal): self.logger.error(f"TP/SL Invalid Risk ATR: {atr_val}"); valid = False
        elif atr_val <= 0: self.logger.warning(f"TP/SL Risk ATR is zero/negative ({atr_val})."); atr_val = Decimal('0'); # Proceed w/ 0 offset
        if not (isinstance(tp_mult, Decimal) and tp_mult >= 0): self.logger.error(f"TP/SL Invalid TP mult: {tp_mult}"); valid = False
        if not (isinstance(sl_mult, Decimal) and sl_mult > 0): self.logger.error(f"TP/SL Invalid SL mult: {sl_mult}"); valid = False
        if not (isinstance(min_tick, Decimal) and min_tick > 0): self.logger.error(f"TP/SL Invalid min_tick: {min_tick}"); valid = False
        if not valid: return entry_price, None, None
        try:
            tp_offset = atr_val * tp_mult; sl_offset = atr_val * sl_mult
            tp_raw, sl_raw = None, None
            if signal == "BUY": tp_raw = entry_price + tp_offset; sl_raw = entry_price - sl_offset
            else: tp_raw = entry_price - tp_offset; sl_raw = entry_price + sl_offset
            tp, sl = None, None
            price_fmt = f'.{self.price_precision}f'
            if tp_raw is not None:
                 tp_round = ROUND_UP if signal == "BUY" else ROUND_DOWN # Conservative TP rounding
                 tp = (tp_raw / min_tick).quantize(Decimal('1'), rounding=tp_round) * min_tick
                 self.logger.debug(f"TP Raw={tp_raw}, Rounded={tp}")
            if sl_raw is not None:
                 sl_round = ROUND_DOWN if signal == "BUY" else ROUND_UP # Conservative SL rounding
                 sl = (sl_raw / min_tick).quantize(Decimal('1'), rounding=sl_round) * min_tick
                 self.logger.debug(f"SL Raw={sl_raw}, Rounded={sl}")

            # Validation checks
            if sl is not None:
                 if signal == "BUY" and sl >= entry_price: sl -= min_tick; self.logger.warning(f"Adj BUY SL down: {sl:{price_fmt}}")
                 elif signal == "SELL" and sl <= entry_price: sl += min_tick; self.logger.warning(f"Adj SELL SL up: {sl:{price_fmt}}")
            if tp is not None and tp_mult > 0:
                 if signal == "BUY" and tp <= entry_price: tp += min_tick; self.logger.warning(f"Adj BUY TP up: {tp:{price_fmt}}")
                 elif signal == "SELL" and tp >= entry_price: tp -= min_tick; self.logger.warning(f"Adj SELL TP down: {tp:{price_fmt}}")
            if sl is not None and sl <= 0: self.logger.error(f"SL zero/negative ({sl})."); sl = None
            if tp is not None and tp <= 0 and tp_mult > 0: self.logger.error(f"TP zero/negative ({tp})."); tp = None
            elif tp is not None and tp <=0 and tp_mult == 0: tp = None # Treat 0 TP mult as no TP

            tp_str = f"{tp:{price_fmt}}" if tp else "None"; sl_str = f"{sl:{price_fmt}}" if sl else "None"
            self.logger.info(f"Calculated TP/SL for {self.symbol} {signal} (RiskATR={atr_val}): Entry={entry_price:{price_fmt}}, TP={tp_str}, SL={sl_str}")
            return entry_price, tp, sl
        except Exception as e: self.logger.error(f"Error calculating TP/SL: {e}", exc_info=True); return entry_price, None, None

# --- Helper Functions (Leveraging volbot5 versions) ---
fetch_balance = fetch_balance # Use the more robust one from volbot5
get_market_info = get_market_info # Use the one with validation and enhancements from volbot5
get_open_position = get_open_position # Use the enhanced one from volbot5
calculate_position_size = calculate_position_size # Use the one integrated with Analyzer rounding
place_market_order = place_market_order # Use the robust one with Analyzer rounding
set_position_protection = set_position_protection # Use the detailed one with change detection

# Helper to color signal logs
def signal_to_color(signal):
    if signal == "BUY": return f"{NEON_GREEN}{signal}{RESET}"
    if signal == "SELL": return f"{NEON_RED}{signal}{RESET}"
    return f"{NEON_YELLOW}{signal}{RESET}"

# --- Main Trading Loop Function (Merged) ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Performs one cycle of analysis and trading for a symbol."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ==---")
    cycle_start_time = time.monotonic()
    try:
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info: lg.error(f"Skipping cycle {symbol}: Invalid market info."); return
        interval_str = config.get("interval", "5"); ccxt_timeframe = CCXT_INTERVAL_MAP.get(interval_str)
        if not ccxt_timeframe: lg.error(f"Invalid interval {interval_str} for {symbol}."); return

        # Klines fetch limit calculation (moved here for clarity)
        min_required_data = 250 # Fallback
        try:
             buffer = 100
             min_len_livexy = 0
             if config.get("livexy_enabled"): min_len_livexy = max([config.get(k[0], k[1]) for k in [('livexy_atr_period',DEFAULT_ATR_PERIOD_LIVEXY),('ema_long_period',DEFAULT_EMA_LONG_PERIOD),('rsi_period',DEFAULT_RSI_WINDOW),('bollinger_bands_period',DEFAULT_BOLLINGER_BANDS_PERIOD),('cci_window',DEFAULT_CCI_WINDOW),('williams_r_window',DEFAULT_WILLIAMS_R_WINDOW),('mfi_window',DEFAULT_MFI_WINDOW),('stoch_rsi_window',DEFAULT_STOCH_RSI_WINDOW),('sma_10_window',DEFAULT_SMA_10_WINDOW),('momentum_period',DEFAULT_MOMENTUM_PERIOD),('volume_ma_period',DEFAULT_VOLUME_MA_PERIOD)]] + [50]) # Approx min for LiveXY
             min_len_volbot = 0
             if config.get("volbot_enabled"): min_len_volbot = max([config.get(k[0],k[1]) for k in [('volbot_length', DEFAULT_VOLBOT_LENGTH),('volbot_atr_length', DEFAULT_VOLBOT_ATR_LENGTH),('volbot_volume_percentile_lookback', DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK)]] + [config.get('volbot_pivot_left_len_h',0)+config.get('volbot_pivot_right_len_h',0)+1, config.get('volbot_pivot_left_len_l',0)+config.get('volbot_pivot_right_len_l',0)+1])
             min_len_risk = config.get("atr_period_risk", DEFAULT_ATR_PERIOD_RISK)
             min_required_data = max(min_len_livexy, min_len_volbot, min_len_risk, 0) + buffer
             lg.debug(f"Min klines needed: {min_required_data} (LiveXY: {min_len_livexy}, Volbot: {min_len_volbot}, Risk: {min_len_risk}, Buffer: {buffer})")
        except Exception as e: lg.error(f"Error calculating min klines: {e}. Using fallback {min_required_data}.")

        kline_limit = min_required_data
        df_klines = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, limit=kline_limit, logger=lg)
        min_acceptable_klines = 50
        if df_klines.empty or len(df_klines) < min_acceptable_klines:
            lg.warning(f"Insufficient kline data {symbol} (got {len(df_klines)}, needed ~{min_required_data}). Skipping."); return

        # Fetch order book if LiveXY needs it
        orderbook_data = None
        active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
        if config.get("livexy_enabled") and config.get("indicators",{}).get("orderbook", False) and safe_decimal(active_weights.get("orderbook", 0)) != 0:
            orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)

        analyzer = TradingAnalyzer(df_raw=df_klines, logger=lg, config=config, market_info=market_info, orderbook_data=orderbook_data)
        if analyzer.df_processed.empty or not analyzer.strategy_state:
            lg.error(f"Failed to calculate indicators/state for {symbol}. Skipping."); return

        current_position = get_open_position(exchange, symbol, lg)
        has_open_position = current_position is not None
        position_side = current_position.get('side') if has_open_position else None
        position_entry_price = current_position.get('entryPrice') if has_open_position else None
        position_size = current_position.get('contracts', Decimal('0')) if has_open_position else Decimal('0')

        current_price = fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price is None:
            last_close = analyzer.strategy_state.get('close');
            if isinstance(last_close, Decimal) and last_close > 0: current_price = last_close; lg.warning(f"Using last close ({current_price}) as current price.")
            else: lg.error(f"Failed to get current price for {symbol}. Cannot proceed."); return
        price_fmt = f".{analyzer.get_price_precision()}f"

        signal = analyzer.generate_trading_signal()
        trading_enabled = config.get("enable_trading", False)

        # --- Position Management Logic ---
        if has_open_position and isinstance(position_size, Decimal) and isinstance(position_entry_price, Decimal) and abs(position_size) > Decimal('1e-9'):
            # --- IN POSITION ---
            exit_signal = (position_side == 'long' and signal == "SELL") or (position_side == 'short' and signal == "BUY")
            if exit_signal:
                reason = f"Opposing signal ({signal})"
                lg.info(f"{signal_to_color(signal)} Exit Signal Triggered: {reason}. Closing {position_side.upper()} position.")
                if trading_enabled:
                    close_side = 'sell' if position_side == 'long' else 'buy'
                    close_order = place_market_order(exchange, symbol, close_side, abs(position_size), market_info, analyzer, lg, params={'reduceOnly': True})
                    if close_order:
                        lg.info(f"{COLOR_SUCCESS}Market order placed to close position.")
                        time.sleep(1); # Pause before cancelling stops
                        try:
                            cancel_success = set_position_protection(exchange, symbol, market_info, analyzer, Decimal(0), Decimal(0), {'trailingStop': '0'}, logger=lg)
                            if cancel_success: lg.info("Cancellation request for stops sent.")
                            else: lg.warning("Could not confirm cancellation of stops.")
                        except Exception as cancel_err: lg.warning(f"Error cancelling stops: {cancel_err}")
                    else: lg.error(f"{COLOR_ERROR}Failed to place close order. MANUAL INTERVENTION REQUIRED!")
                return # End cycle after attempting closure
            else:
                # --- Risk Management (BE, TSL Activation) ---
                lg.info(f"No exit signal. Performing risk management for {position_side} position...")
                new_sl, new_tsl_params, needs_update = None, None, False
                current_sl = current_position.get('stopLossPrice'); current_tp = current_position.get('takeProfitPrice')
                current_tsl_active = current_position.get('trailingStopLossActive', False)
                # --- Break-Even ---
                enable_be = config.get("enable_break_even", True); disable_be_if_tsl = True
                run_be = enable_be and not (disable_be_if_tsl and current_tsl_active)
                if run_be:
                    risk_atr = analyzer.strategy_state.get("atr_risk"); min_tick = analyzer.get_min_tick_size()
                    if isinstance(risk_atr, Decimal) and risk_atr > 0 and min_tick > 0:
                        be_trigger_mult = safe_decimal(config.get("break_even_trigger_atr_multiple", 1.0), 1.0)
                        be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        profit_target = risk_atr * be_trigger_mult
                        current_profit = (current_price - position_entry_price) if position_side == 'long' else (position_entry_price - current_price)
                        lg.debug(f"BE Check: Profit={current_profit:{price_fmt}}, Target={profit_target:{price_fmt}}")
                        if current_profit >= profit_target:
                            offset = min_tick * be_offset_ticks
                            be_raw = position_entry_price + offset if position_side == 'long' else position_entry_price - offset
                            be_round = ROUND_UP if position_side == 'long' else ROUND_DOWN
                            be_price = (be_raw / min_tick).quantize(Decimal('1'), rounding=be_round) * min_tick
                            is_better = current_sl is None or (position_side == 'long' and be_price > current_sl) or (position_side == 'short' and be_price < current_sl)
                            if is_better:
                                lg.info(f"{COLOR_SUCCESS}Break-Even Triggered! Moving SL to {be_price:{price_fmt}}"); new_sl = be_price; needs_update = True
                            else: lg.debug(f"BE triggered, but proposed SL {be_price} not better than current {current_sl}.")
                    elif enable_be: lg.warning(f"Cannot calc BE: Invalid Risk ATR({risk_atr}) or Min Tick({min_tick}).")
                # --- TSL Activation ---
                enable_tsl = config.get("enable_trailing_stop", True)
                if enable_tsl and not current_tsl_active:
                    tsl_rate_str = str(config.get("trailing_stop_callback_rate", "0.005"))
                    tsl_act_perc = safe_decimal(config.get("trailing_stop_activation_percentage", "0.003"), 0)
                    if re.match(r"^\d+(\.\d+)?%?$", tsl_rate_str) and safe_decimal(tsl_rate_str.replace('%','')) > 0:
                        activate = False
                        if tsl_act_perc <= 0: activate = True; lg.warning("TSL active threshold <=0, attempting immediate activation.")
                        else:
                            profit_perc = 0;
                            if position_entry_price > 0: profit_perc = (current_price / position_entry_price) - 1 if position_side == 'long' else 1 - (current_price / position_entry_price)
                            lg.debug(f"TSL Act Check: Profit%={profit_perc:.4%}, Threshold%={tsl_act_perc:.4%}")
                            if profit_perc >= tsl_act_perc: activate = True; lg.info(f"{COLOR_SUCCESS}TSL activation threshold reached.")
                        if activate:
                            new_tsl_params = {'trailingStop': tsl_rate_str}; needs_update = True
                            if tsl_act_perc > 0: # Calculate activation price if needed
                                act_raw = position_entry_price * (1 + tsl_act_perc) if position_side == 'long' else position_entry_price * (1 - tsl_act_perc)
                                act_round = ROUND_UP if position_side == 'long' else ROUND_DOWN
                                act_price = (act_raw / analyzer.get_min_tick_size()).quantize(Decimal('1'), rounding=act_round) * analyzer.get_min_tick_size()
                                if act_price > 0: new_tsl_params['activePrice'] = f"{act_price:{price_fmt}}"; lg.info(f" Calculated TSL Act Price: {new_tsl_params['activePrice']}")
                    else: lg.error(f"Invalid TSL callback rate '{tsl_rate_str}' in config.")
                # --- Update Protection API Call ---
                if needs_update and trading_enabled:
                    lg.info("Attempting to update position protection...")
                    final_sl = new_sl if new_sl is not None else current_sl # Prioritize BE SL if set
                    final_tp = current_tp # Keep existing TP
                    final_tsl = new_tsl_params # Apply new TSL params if activation triggered
                    success = set_position_protection(exchange, symbol, market_info, analyzer, final_sl, final_tp, final_tsl, current_position, lg)
                    if success: lg.info(f"{COLOR_SUCCESS}Protection update request sent.")
                    else: lg.error(f"{COLOR_ERROR}Failed to update protection.")
                elif needs_update and not trading_enabled: lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have updated protection.")
                else: lg.info("No risk management actions triggered requiring protection update.")
        else:
            # --- OUT OF POSITION ---
            lg.info(f"No open position for {symbol}. Checking for entry signals ({signal}).")
            if signal in ["BUY", "SELL"]:
                lg.info(f"{signal_to_color(signal)} Entry Signal Detected @ {current_price:{price_fmt}}")
                quote_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if quote_balance is None: lg.warning(f"Could not fetch {QUOTE_CURRENCY} balance."); quote_balance = Decimal('0')
                _, potential_tp, potential_sl = analyzer.calculate_entry_tp_sl(current_price, signal)
                if potential_sl is None: lg.error(f"{COLOR_ERROR}Cannot enter {signal}: Failed to calculate initial SL."); return
                risk_float = config.get("risk_per_trade", 0.01)
                pos_size = calculate_position_size(quote_balance, risk_float, potential_sl, current_price, market_info, analyzer, lg)
                if pos_size is None or pos_size <= 0: lg.error(f"{COLOR_ERROR}Cannot enter {signal}: Position size calculation failed ({pos_size})."); return
                if trading_enabled:
                    lg.info(f"Attempting to enter {signal} | Size={pos_size} | SL={potential_sl:{price_fmt}} | TP={potential_tp:{price_fmt} if potential_tp else 'None'}")
                    # Leverage Setting
                    if market_info.get('is_contract'):
                        lev = int(config.get("leverage", 10))
                        try:
                            lg.info(f"Setting leverage to {lev}x"); params={'category': market_info.get('contract_type','linear').lower(), 'buyLeverage': str(float(lev)), 'sellLeverage': str(float(lev))} if 'bybit' in exchange.id.lower() else {}; exchange.set_leverage(lev, symbol, params=params)
                        except Exception as e: lg.warning(f"{COLOR_WARNING}Leverage not modified to {lev}x ({e}). Proceeding.")
                    # Place Entry Order
                    entry_order = place_market_order(exchange, symbol, signal.lower(), pos_size, market_info, analyzer, lg)
                    # Set Protection After Entry
                    if entry_order and entry_order.get('status','').lower() in ['closed','filled']:
                        lg.info(f"{COLOR_SUCCESS}Entry order likely filled (ID:{entry_order.get('id')}). Waiting {POSITION_CONFIRM_DELAY}s to set protection...")
                        time.sleep(POSITION_CONFIRM_DELAY)
                        tsl_entry_params = None
                        if config.get("enable_trailing_stop", True):
                            tsl_rate_str = str(config.get("trailing_stop_callback_rate","0.005"))
                            if re.match(r"^\d+(\.\d+)?%?$", tsl_rate_str) and safe_decimal(tsl_rate_str.replace('%','')) > 0:
                                tsl_entry_params={'trailingStop':tsl_rate_str}; tsl_act_perc=safe_decimal(config.get("trailing_stop_activation_percentage","0.003"),0)
                                if tsl_act_perc > 0: # Calculate initial activation price
                                    filled_price = safe_decimal(entry_order.get('average', entry_order.get('price', current_price))) or current_price
                                    act_raw = filled_price * (1+tsl_act_perc) if signal=="BUY" else filled_price * (1-tsl_act_perc); act_round = ROUND_UP if signal=="BUY" else ROUND_DOWN
                                    act_price = (act_raw / analyzer.get_min_tick_size()).quantize(Decimal('1'),rounding=act_round) * analyzer.get_min_tick_size()
                                    if act_price > 0: tsl_entry_params['activePrice'] = f"{act_price:{price_fmt}}"
                            else: lg.error(f"Invalid TSL callback rate '{tsl_rate_str}'. Cannot set initial TSL.")
                        protection_success = set_position_protection(exchange, symbol, market_info, analyzer, potential_sl, potential_tp, tsl_entry_params, logger=lg)
                        if protection_success: lg.info(f"{COLOR_SUCCESS}Initial protection set for new {signal} position.")
                        else: lg.error(f"{COLOR_ERROR}Failed to set initial protection for {symbol}. MANUAL CHECK REQUIRED!")
                    elif entry_order: lg.error(f"Entry order status uncertain ('{entry_order.get('status')}'). Cannot set protection. MANUAL CHECK REQUIRED!")
                    else: lg.error(f"Entry order failed for {signal} {symbol}.")
                else: lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have entered {signal} | Size={pos_size} | SL={potential_sl:{price_fmt}} | TP={potential_tp:{price_fmt} if potential_tp else 'None'}")
            elif signal == "HOLD": lg.info("Signal is HOLD, no position. No action.")

    except ccxt.AuthenticationError as e: lg.critical(f"{COLOR_ERROR}CRITICAL AUTH ERROR: {e}. Stopping.", exc_info=True); raise
    except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: lg.error(f"Network Error cycle {symbol}: {e}.") # Allow retry
    except ccxt.ExchangeError as e: lg.error(f"Exchange Error cycle {symbol}: {e}.", exc_info=True) # Allow retry unless critical (handled higher up)
    except Exception as e: lg.error(f"!!! UNHANDLED EXCEPTION cycle {symbol} !!!: {e}", exc_info=True) # Log unexpected errors
    finally: lg.info(f"---== Finished {symbol} ({time.monotonic() - cycle_start_time:.2f}s) ==---")

# --- Main Execution ---
def main() -> None:
    """Initializes and runs the main trading loop."""
    init_logger = setup_logger("init")
    init_logger.info(f"{COLOR_HEADER}--- Merged Bot Initializing ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    init_logger.info(f"Config: {os.path.abspath(CONFIG_FILE)}, Logs: {os.path.abspath(LOG_DIRECTORY)}")
    init_logger.info(f"Quote: {QUOTE_CURRENCY}, Trading: {CONFIG.get('enable_trading')}, Sandbox: {CONFIG.get('use_sandbox')}")
    init_logger.info(f"Interval: {CONFIG.get('interval')}, Timezone: {TIMEZONE.key}, LogLevel: {logging.getLevelName(console_log_level)}")
    init_logger.info(f"Signal Mode: {CONFIG.get('signal_mode')}")
    init_logger.info(f"LiveXY_ON={CONFIG.get('livexy_enabled')}, Volbot_ON={CONFIG.get('volbot_enabled')}")

    exchange = initialize_exchange(init_logger)
    if not exchange: init_logger.critical("Exchange init failed. Exiting."); return

    # --- Symbol Selection ---
    symbols_to_trade: List[str] = []
    init_logger.info(f"{COLOR_HEADER}--- Symbol Selection ---{RESET}")
    while True:
        list_str = f"Current: {', '.join(symbols_to_trade)}" if symbols_to_trade else "Current: empty"
        print(f"\n{list_str}")
        prompt = f"Enter symbol (e.g. BTC/USDT), '{COLOR_CYAN}all{RESET}' active linear {QUOTE_CURRENCY} perps, '{COLOR_YELLOW}clear{RESET}', or {COLOR_GREEN}Enter{RESET} to start ({len(symbols_to_trade)}): "
        try: symbol_input = input(prompt).strip().upper()
        except EOFError: symbol_input = "" # Handle non-interactive
        if not symbol_input and symbols_to_trade: init_logger.info(f"Starting with {len(symbols_to_trade)} symbols."); break
        if not symbol_input and not symbols_to_trade: print(f"{COLOR_WARNING}List empty. Add symbols or 'all'.{RESET}"); continue
        if symbol_input == 'CLEAR': symbols_to_trade = []; print("List cleared."); continue
        if symbol_input == 'ALL':
            init_logger.info(f"Fetching active linear {QUOTE_CURRENCY} perpetual swaps...")
            try:
                all_mkts = exchange.load_markets(True)
                linear_swaps = [m['symbol'] for m in all_mkts.values() if m.get('active') and m.get('linear') and m.get('swap') and m.get('quote','').upper()==QUOTE_CURRENCY]
                init_logger.info(f"Found {len(linear_swaps)} symbols. Adding to list.")
                # Add only unique symbols
                new_symbols = [s for s in linear_swaps if s not in symbols_to_trade]
                symbols_to_trade.extend(new_symbols)
                init_logger.info(f"Added {len(new_symbols)} new symbols.")
            except Exception as e: init_logger.error(f"Error fetching 'all' symbols: {e}")
            continue
        # Validate single symbol input
        init_logger.info(f"Validating symbol '{symbol_input}'...")
        market_info = get_market_info(exchange, symbol_input, init_logger)
        if market_info:
            validated_symbol = market_info['symbol']
            if validated_symbol not in symbols_to_trade:
                symbols_to_trade.append(validated_symbol); print(f"{COLOR_SUCCESS}Added {validated_symbol}{RESET}")
            else: print(f"{COLOR_YELLOW}{validated_symbol} already in list.{RESET}")
        else: print(f"{COLOR_ERROR}Symbol '{symbol_input}' invalid or not found.{RESET}")

    if not symbols_to_trade: init_logger.critical("No symbols selected to trade. Exiting."); return

    # --- Safety Check ---
    if CONFIG.get("enable_trading"):
        init_logger.warning(f"{COLOR_RED}!!! LIVE TRADING ENABLED !!!{RESET}")
        if not CONFIG.get("use_sandbox"): init_logger.warning(f"{COLOR_RED}!!! USING REAL MONEY !!!{RESET}")
        else: init_logger.warning(f"{COLOR_YELLOW}Using SANDBOX (Testnet).{RESET}")
        init_logger.warning(f"Review Config: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={CONFIG['enable_trailing_stop']}, BE={CONFIG['enable_break_even']}")
        try: input(f"Press {COLOR_GREEN}Enter{RESET} to confirm and continue, or {COLOR_RED}Ctrl+C{RESET} to abort... "); init_logger.info("User confirmed.")
        except KeyboardInterrupt: init_logger.info("User abort."); return
    else: init_logger.info(f"{COLOR_YELLOW}TRADING DISABLED. Analysis-only mode.{RESET}")

    # --- Setup Loggers for Each Symbol ---
    symbol_loggers = {symbol: setup_logger(symbol.replace('/', '_').replace(':', '-')) for symbol in symbols_to_trade}

    # --- Main Loop ---
    init_logger.info(f"{COLOR_HEADER}--- Starting Main Trading Loop for {len(symbols_to_trade)} Symbols ---{RESET}")
    try:
        while True:
            loop_start = time.time()
            for symbol in symbols_to_trade:
                symbol_logger = symbol_loggers[symbol]
                try:
                    # --- Optional: Reload Config ---
                    # CONFIG = load_config(CONFIG_FILE) # Uncomment to reload config each cycle
                    # ... update dependent variables (QUOTE_CURRENCY, log levels) if reloading ...
                    analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except ccxt.AuthenticationError: raise # Critical, stop immediately
                except Exception as symbol_err:
                    symbol_logger.error(f"Unhandled error during cycle for {symbol}: {symbol_err}", exc_info=True)
                    symbol_logger.info("Attempting to continue with next symbol/cycle...")
            # --- Loop Delay ---
            elapsed = time.time() - loop_start
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed)
            init_logger.debug(f"Main loop cycle finished in {elapsed:.2f}s. Waiting {sleep_time:.2f}s...")
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt: init_logger.info("Keyboard Interrupt. Shutting down...")
    except ccxt.AuthenticationError: init_logger.critical("AUTH ERROR DETECTED. BOT STOPPED.") # Already logged in analyze_and_trade
    except Exception as e: init_logger.critical(f"Critical unhandled error in main loop: {e}", exc_info=True)
    finally:
        init_logger.info(f"{COLOR_HEADER}--- Merged Bot Stopping ---{RESET}")
        if exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); exchange.close()
            except Exception as ce: init_logger.error(f"Error closing connection: {ce}")
        logging.shutdown()
        init_logger.info("Bot stopped.")

if __name__ == "__main__":
    main()
```