Application starting...
2025-05-01 03:59:18 [ERROR   ] INIT - Unexpected error loading markets (Attempt 1/3): string indices must be integers, not 'str'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/neonta_v3.py", line 396, in initialize_markets
    self.markets = await self.exchange.load_markets()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 293, in load_markets
    raise e
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 285, in load_markets
    result = await self.markets_loading
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 274, in load_markets_helper
    currencies = await self.fetch_currencies()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/bybit.py", line 1600, in fetch_currencies
    response = await self.privateGetV5AssetCoinQueryInfo(params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 901, in request
    return await self.fetch2(path, api, method, params, headers, body, config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 876, in fetch2
    request = self.sign(path, api, method, params, headers, body)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/bybit.py", line 8713, in sign
    url = self.implode_hostname(self.urls['api'][api]) + '/' + path
                                ~~~~~~~~~~~~~~~~^^^^^
TypeError: string indices must be integers, not 'str'
2025-05-01 03:59:23 [ERROR   ] INIT - Unexpected error loading markets (Attempt 2/3): string indices must be integers, not 'str'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/neonta_v3.py", line 396, in initialize_markets
    self.markets = await self.exchange.load_markets()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 293, in load_markets
    raise e
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 285, in load_markets
    result = await self.markets_loading
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 274, in load_markets_helper
    currencies = await self.fetch_currencies()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/bybit.py", line 1600, in fetch_currencies
    response = await self.privateGetV5AssetCoinQueryInfo(params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 901, in request
    return await self.fetch2(path, api, method, params, headers, body, config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/base/exchange.py", line 876, in fetch2
    request = self.sign(path, api, method, params, headers, body)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/async_support/bybit.py", line 8713, in sign
    url = self.implode_hostname(self.urls['api'][api]) + '/' + path
                                ~~~~~~~~~~~~~~~~^^^^^
TypeError: string indices must be integers, not 'str'
2025-05-01 03:59:32 [ERROR   ] INIT - Unexpected error loading markets (Attempt 3/3): string indices must be integers, not 'str'
Traceback (most recent call last):3

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, getcontext
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.exceptions import RequestException

# --- Initialization ---
init(autoreset=True)
load_dotenv()
getcontext().prec = 18  # Set Decimal precision

# --- Constants ---
CONFIG_FILE_NAME = "config.json"
LOG_DIRECTORY_NAME = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago"
MAX_API_RETRIES = 5
INITIAL_RETRY_DELAY_SECONDS = 5
MAX_RETRY_DELAY_SECONDS = 60
CCXT_TIMEOUT_MS = 20000

# Bybit API Configuration
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

API_ENV = os.getenv("API_ENVIRONMENT", "prod").lower()
CCXT_URLS = {
    'test': 'https://api-testnet.bybit.com',
    'prod': 'https://api.bybit.com',
}
CCXT_BASE_URL = CCXT_URLS.get(API_ENV, CCXT_URLS['prod'])

# Timezone Configuration
try:
    APP_TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR))
except ZoneInfoNotFoundError:
    print(f"{Fore.YELLOW}Warning: Timezone '{os.getenv('TIMEZONE', DEFAULT_TIMEZONE_STR)}' not found. Using UTC.{Style.RESET_ALL}")
    APP_TIMEZONE = ZoneInfo("UTC")

# Paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = BASE_DIR / CONFIG_FILE_NAME
LOG_DIRECTORY = BASE_DIR / LOG_DIRECTORY_NAME
LOG_DIRECTORY.mkdir(exist_ok=True)

# Timeframes
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}
REVERSE_CCXT_INTERVAL_MAP = {v: k for k, v in CCXT_INTERVAL_MAP.items()}

# Color Constants
class Color(Enum):
    GREEN = Fore.LIGHTGREEN_EX
    BLUE = Fore.CYAN
    PURPLE = Fore.MAGENTA
    YELLOW = Fore.YELLOW
    RED = Fore.LIGHTRED_EX
    RESET = Style.RESET_ALL

    @staticmethod
    def format(text: str, color: 'Color') -> str:
        return f"{color.value}{text}{Color.RESET.value}"

# Signal States Enum
class SignalState(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    STRONG_BULLISH = "Strong Bullish"
    STRONG_BEARISH = "Strong Bearish"
    RANGING = "Ranging"
    OVERBOUGHT = "Overbought"
    OVERSOLD = "Oversold"
    ABOVE_SIGNAL = "Above Signal"
    BELOW_SIGNAL = "Below Signal"
    BREAKOUT_UPPER = "Breakout Upper"
    BREAKDOWN_LOWER = "Breakdown Lower"
    WITHIN_BANDS = "Within Bands"
    HIGH_VOLUME = "High"
    LOW_VOLUME = "Low"
    AVERAGE_VOLUME = "Average"
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    FLAT = "Flat"
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"
    FLIP_BULLISH = "Flip Bullish"
    FLIP_BEARISH = "Flip Bearish"
    NONE = "None"
    NA = "N/A"


# --- Configuration Loading ---
@dataclass
class IndicatorSettings:
    default_interval: str = "15"
    momentum_period: int = 10
    volume_ma_period: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    stoch_rsi_period: int = 14
    stoch_k_period: int = 3
    stoch_d_period: int = 3
    cci_period: int = 20
    williams_r_period: int = 14
    mfi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_bands_period: int = 20
    bollinger_bands_std_dev: float = 2.0
    ema_short_period: int = 12
    ema_long_period: int = 26
    sma_short_period: int = 10
    sma_long_period: int = 50
    adx_period: int = 14
    psar_step: float = 0.02
    psar_max_step: float = 0.2

@dataclass
class AnalysisFlags:
    ema_alignment: bool = True
    momentum_crossover: bool = False # Requires more logic
    volume_confirmation: bool = True
    rsi_divergence: bool = False # Basic check implemented
    macd_divergence: bool = True # Basic check implemented
    stoch_rsi_cross: bool = True
    rsi_threshold: bool = True
    mfi_threshold: bool = True
    cci_threshold: bool = True
    williams_r_threshold: bool = True
    macd_cross: bool = True
    bollinger_bands_break: bool = True
    adx_trend_strength: bool = True
    obv_trend: bool = True
    adi_trend: bool = True
    psar_flip: bool = True

@dataclass
class Thresholds:
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    mfi_overbought: int = 80
    mfi_oversold: int = 20
    cci_overbought: int = 100
    cci_oversold: int = -100
    williams_r_overbought: int = -20
    williams_r_oversold: int = -80
    adx_trending: int = 25

@dataclass
class OrderbookSettings:
    limit: int = 50
    cluster_threshold_usd: int = 10000
    cluster_proximity_pct: float = 0.1

@dataclass
class LoggingSettings:
    level: str = "INFO"
    rotation_max_bytes: int = 10 * 1024 * 1024
    rotation_backup_count: int = 5

@dataclass
class AppConfig:
    analysis_interval_seconds: int = 30
    kline_limit: int = 200
    indicator_settings: IndicatorSettings = field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = field(default_factory=AnalysisFlags)
    thresholds: Thresholds = field(default_factory=Thresholds)
    orderbook_settings: OrderbookSettings = field(default_factory=OrderbookSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def _merge_dicts(cls, default: dict, user: dict) -> dict:
        """Recursively merges user dict into default dict."""
        merged = default.copy()
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._merge_dicts(merged[key], value)
            else:
                # Only update if key exists in default structure (prevents arbitrary additions)
                # This requires default_config to be generated from the dataclass structure
                if key in merged:
                     merged[key] = value
                # else: print warning about unknown key?
        return merged

    @classmethod
    def _dict_to_dataclass(cls, data_class, data_dict):
        """Converts a dictionary to a nested dataclass structure."""
        field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
        init_args = {}
        for name, type_hint in field_types.items():
            if name in data_dict:
                value = data_dict[name]
                # Check if the field type is itself a dataclass
                if hasattr(type_hint, '__dataclass_fields__'):
                    init_args[name] = cls._dict_to_dataclass(type_hint, value)
                else:
                    init_args[name] = value
            # else: rely on default_factory or default value defined in dataclass
        return data_class(**init_args)

    @classmethod
    def load(cls, filepath: Path) -> 'AppConfig':
        default_config_obj = cls()
        # Convert default dataclass to dict for merging comparison
        # This is a bit complex; simpler if not supporting deep merge of user config
        # For now, we'll stick to the original merge logic but use the dataclass for defaults
        default_config_dict = {
            "analysis_interval_seconds": default_config_obj.analysis_interval_seconds,
            "kline_limit": default_config_obj.kline_limit,
            "indicator_settings": default_config_obj.indicator_settings.__dict__,
            "analysis_flags": default_config_obj.analysis_flags.__dict__,
            "thresholds": default_config_obj.thresholds.__dict__,
            "orderbook_settings": default_config_obj.orderbook_settings.__dict__,
            "logging": default_config_obj.logging.__dict__,
        }

        if not filepath.exists():
            try:
                with filepath.open('w', encoding="utf-8") as f:
                    json.dump(default_config_dict, f, indent=2)
                print(Color.format(f"Created new config file '{filepath}' with defaults.", Color.YELLOW))
                return default_config_obj # Return the default dataclass object
            except IOError as e:
                print(Color.format(f"Error creating default config file: {e}", Color.RED))
                print(Color.format("Loading internal defaults.", Color.YELLOW))
                return default_config_obj

        try:
            with filepath.open("r", encoding="utf-8") as f:
                user_config = json.load(f)

            merged_config_dict = cls._merge_dicts(default_config_dict, user_config)
            # Convert the final merged dict back into the nested dataclass structure
            return cls._dict_to_dataclass(cls, merged_config_dict)

        except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError) as e:
            print(Color.format(f"Error loading/parsing config file '{filepath}': {e}", Color.RED))
            print(Color.format("Loading internal defaults.", Color.YELLOW))
            return default_config_obj

CONFIG = AppConfig.load(CONFIG_FILE_PATH)

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that masks API key/secret."""
    def format(self, record):
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        # Basic regex to remove color codes for file logging
        msg_no_color = re.sub(r'\x1b\[[0-9;]*m', '', msg)
        return msg_no_color

class ColorStreamFormatter(logging.Formatter):
    """Formatter that adds colors for stream output."""
    def __init__(self, fmt: str, datefmt: Optional[str] = None, symbol: str = "GENERAL"):
        super().__init__(fmt, datefmt)
        self.symbol = symbol

    def format(self, record):
        asctime_color = Color.BLUE.value
        level_color_map = {
            logging.DEBUG: Color.PURPLE.value,
            logging.INFO: Color.GREEN.value,
            logging.WARNING: Color.YELLOW.value,
            logging.ERROR: Color.RED.value,
            logging.CRITICAL: Color.RED.value + Style.BRIGHT,
        }
        level_color = level_color_map.get(record.levelno, Color.RESET.value)
        symbol_color = Color.YELLOW.value

        log_fmt = (
            f"{asctime_color}%(asctime)s{Color.RESET.value} "
            f"[{level_color}%(levelname)-8s{Color.RESET.value}] "
            f"{symbol_color}{self.symbol}{Color.RESET.value} - "
            f"%(message)s"
        )
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)

def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger instance for a specific symbol."""
    log_filename = LOG_DIRECTORY / f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
    logger = logging.getLogger(symbol)
    log_level_str = CONFIG.logging.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (with rotation and sensitive data masking)
    try:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=CONFIG.logging.rotation_max_bytes,
            backupCount=CONFIG.logging.rotation_backup_count,
            encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(Color.format(f"Failed to set up file logger for {symbol}: {e}", Color.RED))

    # Stream Handler (with colors)
    stream_handler = logging.StreamHandler()
    stream_formatter = ColorStreamFormatter(
        "", # Format string is defined within the formatter class
        datefmt='%Y-%m-%d %H:%M:%S',
        symbol=symbol
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(log_level) # Stream handler respects the logger's level
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger if it has handlers
    logger.propagate = False

    return logger

# --- Utility Functions ---
async def async_sleep_with_jitter(seconds: float, max_jitter: float = 0.1) -> None:
    """Async sleep with random jitter."""
    jitter = np.random.uniform(0, seconds * max_jitter)
    await asyncio.sleep(seconds + jitter)

def format_decimal(value: Optional[Union[Decimal, float, int, str]], precision: int = 2) -> str:
    """Safely formats a value as a Decimal string with specified precision."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        decimal_value = Decimal(str(value))
        # Use quantize for proper Decimal rounding
        quantizer = Decimal('1e-' + str(precision))
        return str(decimal_value.quantize(quantizer))
    except (InvalidOperation, ValueError, TypeError):
        return str(value) # Fallback to simple string conversion

# --- CCXT Client ---
class BybitCCXTClient:
    """Asynchronous CCXT client for Bybit with retry logic."""
    def __init__(self, api_key: str, api_secret: str, base_url: str, logger_instance: logging.Logger):
        self.logger = logger_instance
        self.exchange = ccxt_async.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Default, can be overridden in params
                'adjustForTimeDifference': True,
            },
            'urls': {'api': base_url},
            'timeout': CCXT_TIMEOUT_MS,
        })
        self.markets: Optional[Dict[str, Any]] = None
        self.market_categories: Dict[str, str] = {} # Cache for market categories

    async def initialize_markets(self, retries: int = 3) -> bool:
        """Loads markets from the exchange with retries."""
        current_delay = INITIAL_RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                self.markets = await self.exchange.load_markets()
                if not self.markets:
                     raise ccxt.ExchangeError("load_markets returned None or empty dict")
                self.logger.info(f"Successfully loaded {len(self.markets)} markets from {self.exchange.name}.")
                self._cache_market_categories()
                return True
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, RequestException) as e:
                self.logger.warning(f"Failed to load markets (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s...")
            except ccxt.AuthenticationError as e:
                self.logger.error(Color.format(f"Authentication Error loading markets: {e}. Check API credentials.", Color.RED))
                return False # No point retrying auth errors
            except Exception as e:
                self.logger.exception(f"Unexpected error loading markets (Attempt {attempt + 1}/{retries}): {e}")

            if attempt < retries - 1:
                await async_sleep_with_jitter(current_delay)
                current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY_SECONDS)
            else:
                 self.logger.error(Color.format("Max retries reached for loading markets.", Color.RED))
        return False

    def _cache_market_categories(self):
        """Pre-calculates and caches market categories after loading markets."""
        if not self.markets:
            return
        self.market_categories = {}
        for symbol, details in self.markets.items():
            category = 'spot' # Default
            if details.get('linear'): category = 'linear'
            elif details.get('inverse'): category = 'inverse'
            elif details.get('spot'): category = 'spot'
            else: # Fallback guess based on quote currency
                if symbol.endswith('/USDT'): category = 'linear'
                elif symbol.endswith('/USD'): category = 'inverse'
                self.logger.debug(f"Guessed category '{category}' for symbol {symbol} based on quote currency.")
            self.market_categories[symbol] = category
        self.logger.debug(f"Cached categories for {len(self.market_categories)} markets.")


    def is_valid_symbol(self, symbol: str) -> bool:
        """Checks if the symbol exists in the loaded markets."""
        if self.markets is None:
            self.logger.warning("Markets not loaded, cannot validate symbol.")
            return False # Be strict if markets aren't loaded
        return symbol in self.markets

    def get_symbol_details(self, symbol: str) -> Optional[dict]:
        """Gets market details for a specific symbol."""
        if not self.is_valid_symbol(symbol):
             self.logger.warning(f"Attempted to get details for invalid or unloaded symbol: {symbol}")
             return None
        return self.markets.get(symbol)

    def get_market_category(self, symbol: str) -> str:
        """Gets the market category (linear, inverse, spot) for a symbol."""
        if symbol in self.market_categories:
            return self.market_categories[symbol]

        # Fallback if called before cache is populated or for an unknown symbol
        self.logger.warning(f"Category for symbol {symbol} not found in cache. Attempting to determine dynamically.")
        details = self.get_symbol_details(symbol) # Checks validity again
        if details:
            if details.get('linear'): return 'linear'
            if details.get('inverse'): return 'inverse'
            if details.get('spot'): return 'spot'

        # Final guess if details are missing
        if symbol.endswith('/USDT'): return 'linear'
        if symbol.endswith('/USD'): return 'inverse'
        self.logger.warning(f"Could not determine category for {symbol}, defaulting to 'spot'.")
        return 'spot' # Default fallback

    async def close(self):
        """Closes the underlying ccxt exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Closed ccxt exchange connection.")
            except Exception as e:
                 self.logger.error(f"Error closing ccxt connection: {e}")

    async def fetch_with_retry(self, method_name: str, *args, **kwargs) -> Optional[Any]:
        """Generic fetch method with retry logic for common transient errors."""
        retries = MAX_API_RETRIES
        current_delay = INITIAL_RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                method = getattr(self.exchange, method_name)
                result = await method(*args, **kwargs)
                return result
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection, RequestException) as e:
                log_msg = f"Network/Timeout/Rate Limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                self.logger.warning(Color.format(log_msg, Color.YELLOW))
            except ccxt.RateLimitExceeded as e:
                 log_msg = f"Explicit Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                 self.logger.warning(Color.format(log_msg, Color.YELLOW))
            except ccxt.ExchangeError as e:
                # Bybit specific error strings for retries
                error_str = str(e).lower()
                retryable_errors = ['too many visits', 'system busy', 'service unavailable', 'ip rate limit']
                if any(err in error_str for err in retryable_errors):
                     log_msg = f"Server busy/rate limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                     self.logger.warning(Color.format(log_msg, Color.YELLOW))
                else:
                    self.logger.error(Color.format(f"Non-retryable ccxt ExchangeError calling {method_name}: {e}", Color.RED))
                    # Log details for debugging non-retryable errors
                    self.logger.debug(f"Args: {args}, Kwargs: {kwargs}")
                    return None # Don't retry non-specified exchange errors
            except Exception as e:
                 # Catch unexpected errors
                 self.logger.exception(Color.format(f"Unexpected error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}", Color.RED))
                 self.logger.debug(f"Args: {args}, Kwargs: {kwargs}")
                 # Optionally retry unexpected errors, or return None immediately
                 # return None # If we don't want to retry unknown errors

            if attempt < retries - 1:
                await async_sleep_with_jitter(current_delay)
                current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY_SECONDS)
            else:
                self.logger.error(Color.format(f"Max retries reached for {method_name}.", Color.RED))
                return None
        return None # Should not be reached if loop completes, but added for safety

    async def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetches ticker information for a symbol."""
        self.logger.debug(f"Fetching ticker for {symbol}")
        category = self.get_market_category(symbol)
        params = {'category': category}
        # Use fetch_tickers which often works better than fetch_ticker for single symbols too
        tickers = await self.fetch_with_retry('fetch_tickers', symbols=[symbol], params=params)
        if tickers and symbol in tickers:
            return tickers[symbol]
        self.logger.error(f"Could not fetch ticker for {symbol} (category: {category})")
        return None

    async def fetch_current_price(self, symbol: str) -> Optional[Decimal]:
        """Fetches the last traded price for a symbol."""
        ticker = await self.fetch_ticker(symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            try:
                return Decimal(str(ticker['last']))
            except (InvalidOperation, TypeError) as e:
                self.logger.error(f"Error converting last price '{ticker['last']}' to Decimal for {symbol}: {e}")
                return None
        self.logger.warning(f"Last price not found or is null in ticker data for {symbol}")
        return None

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetches OHLCV data (klines) for a symbol and timeframe."""
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            self.logger.error(f"Invalid timeframe '{timeframe}' provided. Valid: {VALID_INTERVALS}")
            return pd.DataFrame()

        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching {limit} klines for {symbol} with interval {ccxt_timeframe} (category: {category})")
        params = {'category': category}
        klines = await self.fetch_with_retry('fetch_ohlcv', symbol, timeframe=ccxt_timeframe, limit=limit, params=params)

        if klines is None or len(klines) == 0:
            self.logger.warning(f"No kline data returned for {symbol} interval {ccxt_timeframe}")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                 self.logger.warning(f"Kline data for {symbol} resulted in an empty DataFrame initially.")
                 return pd.DataFrame()

            # Convert timestamp and set timezone
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(APP_TIMEZONE)

            # Convert OHLCV columns to numeric (Decimal for precision)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                # Use Decimal for price/volume if possible, fallback to float if needed
                try:
                    df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else None)
                except (InvalidOperation, TypeError):
                    self.logger.warning(f"Could not convert column '{col}' to Decimal, falling back to numeric float.")
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

            initial_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) # Drop rows with NaN in essential columns
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                 self.logger.warning(f"Dropped {dropped_rows} rows with NaN values from kline data for {symbol}.")

            if df.empty:
                 self.logger.warning(f"Kline data for {symbol} is empty after cleaning NaN values.")
                 return pd.DataFrame()

            # Sort by timestamp just in case API returns them out of order
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            self.logger.debug(f"Successfully fetched and processed {len(df)} klines for {symbol}.")
            return df

        except (ValueError, TypeError, KeyError, Exception) as e:
            self.logger.exception(f"Error processing kline data for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_orderbook(self, symbol: str, limit: int) -> Optional[dict]:
        """Fetches the order book for a symbol."""
        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching order book for {symbol} with limit {limit} (category: {category})")
        params = {'category': category}
        orderbook = await self.fetch_with_retry('fetch_order_book', symbol, limit=limit, params=params)

        if orderbook:
            # Basic validation
            if 'bids' in orderbook and 'asks' in orderbook and isinstance(orderbook['bids'], list) and isinstance(orderbook['asks'], list):
                self.logger.debug(f"Fetched order book for {symbol} with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks.")
                # Further validation could check if bids/asks contain [price, size] pairs
                return orderbook
            else:
                self.logger.warning(f"Fetched order book data for {symbol} is missing bids/asks or has incorrect format.")
                return None
        else:
            self.logger.warning(f"Failed to fetch order book for {symbol} after retries.")
            return None


# --- Trading Analyzer ---
class TradingAnalyzer:
    """Performs technical analysis and interprets results."""
    def __init__(self, config: AppConfig, logger_instance: logging.Logger, symbol: str):
        self.config = config
        self.logger = logger_instance
        self.symbol = symbol
        self.indicator_settings = config.indicator_settings
        self.analysis_flags = config.analysis_flags
        self.thresholds = config.thresholds
        self.orderbook_settings = config.orderbook_settings

        # Dynamically generate expected column names based on config
        self._generate_column_names()

    def _generate_column_names(self):
        """Generates expected indicator column names based on config."""
        self.col_names = {}
        is_ = self.indicator_settings # Alias for brevity
        # Helper to create names, converting std dev to float string for bbands
        def fmt_bb_std(std): return f"{float(std):.1f}" if isinstance(std, (int, float, Decimal)) else str(std)

        self.col_names['sma_short'] = f"SMA_{is_.sma_short_period}"
        self.col_names['sma_long'] = f"SMA_{is_.sma_long_period}"
        self.col_names['ema_short'] = f"EMA_{is_.ema_short_period}"
        self.col_names['ema_long'] = f"EMA_{is_.ema_long_period}"
        self.col_names['rsi'] = f"RSI_{is_.rsi_period}"
        self.col_names['stochrsi_k'] = f"STOCHRSIk_{is_.stoch_rsi_period}_{is_.rsi_period}_{is_.stoch_k_period}_{is_.stoch_d_period}"
        self.col_names['stochrsi_d'] = f"STOCHRSId_{is_.stoch_rsi_period}_{is_.rsi_period}_{is_.stoch_k_period}_{is_.stoch_d_period}"
        self.col_names['macd_line'] = f"MACD_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        self.col_names['macd_signal'] = f"MACDs_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        self.col_names['macd_hist'] = f"MACDh_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        self.col_names['bb_upper'] = f"BBU_{is_.bollinger_bands_period}_{fmt_bb_std(is_.bollinger_bands_std_dev)}"
        self.col_names['bb_lower'] = f"BBL_{is_.bollinger_bands_period}_{fmt_bb_std(is_.bollinger_bands_std_dev)}"
        self.col_names['bb_mid'] = f"BBM_{is_.bollinger_bands_period}_{fmt_bb_std(is_.bollinger_bands_std_dev)}"
        self.col_names['atr'] = f"ATRr_{is_.atr_period}" # pandas_ta uses ATRr for True Range average
        self.col_names['cci'] = f"CCI_{is_.cci_period}_0.015" # Default constant c=0.015 in pandas_ta CCI
        self.col_names['willr'] = f"WILLR_{is_.williams_r_period}"
        self.col_names['mfi'] = f"MFI_{is_.mfi_period}"
        self.col_names['adx'] = f"ADX_{is_.adx_period}"
        self.col_names['dmp'] = f"DMP_{is_.adx_period}" # +DI
        self.col_names['dmn'] = f"DMN_{is_.adx_period}" # -DI
        self.col_names['obv'] = "OBV"
        self.col_names['adosc'] = "ADOSC" # Accumulation/Distribution Oscillator (using this instead of ADI)
        psar_step = is_.psar_step
        psar_max = is_.psar_max_step
        self.col_names['psar_long'] = f"PSARl_{psar_step}_{psar_max}"
        self.col_names['psar_short'] = f"PSARs_{psar_step}_{psar_max}"
        self.col_names['psar_af'] = f"PSARaf_{psar_step}_{psar_max}" # Acceleration Factor
        self.col_names['psar_rev'] = f"PSARr_{psar_step}_{psar_max}" # Reversal signal

        # Custom calculated indicators (if needed, though pandas_ta covers most)
        self.col_names['mom'] = f"MOM_{is_.momentum_period}"
        self.col_names['vol_ma'] = f"VOL_MA_{is_.volume_ma_period}"

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates technical indicators using pandas_ta."""
        if df.empty:
            self.logger.warning("Cannot calculate indicators on empty DataFrame.")
            return df
        # Check if enough data for the longest period required by standard indicators
        min_data_needed = max(
            self.indicator_settings.sma_long_period,
            self.indicator_settings.ema_long_period,
            self.indicator_settings.macd_slow + self.indicator_settings.macd_signal, # MACD needs slow+signal
            self.indicator_settings.bollinger_bands_period,
            self.indicator_settings.adx_period * 2 # ADX often needs more data
        )
        if len(df) < min_data_needed:
             self.logger.warning(f"Insufficient data points ({len(df)} < {min_data_needed}) for some indicators. Results may be inaccurate or contain NaNs.")

        df_out = df.copy()
        is_ = self.indicator_settings # Alias

        # Define the strategy for pandas_ta
        strategy_ta = [
            {"kind": "sma", "length": is_.sma_short_period},
            {"kind": "sma", "length": is_.sma_long_period},
            {"kind": "ema", "length": is_.ema_short_period},
            {"kind": "ema", "length": is_.ema_long_period},
            {"kind": "rsi", "length": is_.rsi_period},
            {"kind": "stochrsi", "length": is_.stoch_rsi_period, "rsi_length": is_.rsi_period, "k": is_.stoch_k_period, "d": is_.stoch_d_period},
            {"kind": "macd", "fast": is_.macd_fast, "slow": is_.macd_slow, "signal": is_.macd_signal},
            {"kind": "bbands", "length": is_.bollinger_bands_period, "std": float(is_.bollinger_bands_std_dev)}, # Ensure std is float for ta-lib compatibility
            {"kind": "atr", "length": is_.atr_period},
            {"kind": "cci", "length": is_.cci_period},
            {"kind": "willr", "length": is_.williams_r_period},
            {"kind": "mfi", "length": is_.mfi_period},
            {"kind": "adx", "length": is_.adx_period},
            {"kind": "obv"},
            {"kind": "adosc"}, # Use AD Oscillator
            {"kind": "psar", "step": is_.psar_step, "max_step": is_.psar_max_step},
            # Custom calculations added separately if needed
            {"kind": "mom", "length": is_.momentum_period},
            {"kind": "sma", "close": "volume", "length": is_.volume_ma_period, "prefix": "VOL_MA"}, # Calculate volume MA directly
        ]

        # Filter out indicators with period 0 or less, as they are invalid
        valid_strategy_ta = []
        for item in strategy_ta:
            is_valid = True
            if "length" in item and item["length"] <= 0: is_valid = False
            if "period" in item and item["period"] <= 0: is_valid = False
            if "fast" in item and item["fast"] <= 0: is_valid = False
            if "slow" in item and item["slow"] <= 0: is_valid = False
            if "signal" in item and item["signal"] <= 0: is_valid = False
            if is_valid:
                valid_strategy_ta.append(item)
            else:
                self.logger.warning(f"Skipping indicator calculation due to invalid period <= 0: {item}")


        strategy = ta.Strategy(
            name="TradingAnalysis",
            description="Comprehensive TA using pandas_ta",
            ta=valid_strategy_ta
        )

        try:
           # Apply the strategy
           df_out.ta.strategy(strategy, timed=False)

           # Rename the volume MA column to match expected name
           vol_ma_generated_name = f"VOL_MA_SMA_{is_.volume_ma_period}" # Default name generated by prefix
           if vol_ma_generated_name in df_out.columns:
               df_out.rename(columns={vol_ma_generated_name: self.col_names['vol_ma']}, inplace=True)

           self.logger.debug(f"Calculated pandas_ta indicators for {self.symbol}.")

           # Fill potentially generated PSAR long/short columns if one is all NaN
           # pandas_ta psar returns PSARl and PSARs columns directly
           psar_l = self.col_names['psar_long']
           psar_s = self.col_names['psar_short']
           if psar_l in df_out.columns and psar_s in df_out.columns:
               if df_out[psar_l].isnull().all() and not df_out[psar_s].isnull().all():
                   df_out[psar_l] = df_out[psar_s]
                   self.logger.debug(f"Filled NaN PSAR long column from short column for {self.symbol}")
               elif df_out[psar_s].isnull().all() and not df_out[psar_l].isnull().all():
                   df_out[psar_s] = df_out[psar_l]
                   self.logger.debug(f"Filled NaN PSAR short column from long column for {self.symbol}")

        except Exception as e:
            self.logger.exception(f"Error calculating indicators using pandas_ta strategy for {self.symbol}: {e}")
            # Return the original DataFrame or an empty one depending on desired behavior
            return df # Return original df without indicators

        # Convert indicator columns to Decimal where appropriate (optional, adds overhead)
        # for col in self.col_names.values():
        #     if col in df_out.columns and df_out[col].dtype != 'object': # Avoid converting timestamp etc.
        #         try:
        #             # Only convert if not already Decimal
        #             if not isinstance(df_out[col].iloc[0], Decimal):
        #                  df_out[col] = df_out[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
        #         except (InvalidOperation, TypeError, IndexError):
        #              self.logger.warning(f"Could not convert indicator column '{col}' to Decimal.", exc_info=False)
        #              # Keep as float/numeric

        return df_out

    def _calculate_levels(self, df: pd.DataFrame, current_price: Decimal) -> dict:
        """Calculates support, resistance, and pivot levels."""
        levels = {"support": {}, "resistance": {}, "pivot": None}
        if df.empty or len(df) < 2:
            self.logger.warning("Insufficient data for level calculation.")
            return levels

        try:
            # Use the full period available in the DataFrame for robust levels
            high = df["high"].max()
            low = df["low"].min()
            # Use the most recent close for pivot calculation
            close = df["close"].iloc[-1]

            if pd.isna(high) or pd.isna(low) or pd.isna(close) or not all(isinstance(v, Decimal) for v in [high, low, close]):
                 self.logger.warning("NaN or non-Decimal values found in OHLC data, cannot calculate levels accurately.")
                 return levels

            # Fibonacci Retracement Levels
            diff = high - low
            if diff > Decimal("1e-9"): # Avoid division by zero or tiny differences
                fib_levels = {
                    "Fib 23.6%": high - diff * Decimal("0.236"),
                    "Fib 38.2%": high - diff * Decimal("0.382"),
                    "Fib 50.0%": high - diff * Decimal("0.5"),
                    "Fib 61.8%": high - diff * Decimal("0.618"),
                    "Fib 78.6%": high - diff * Decimal("0.786"),
                }
                for label, value in fib_levels.items():
                    if value < current_price: levels["support"][label] = value
                    else: levels["resistance"][label] = value

            # Pivot Points (Classical)
            try:
                pivot = (high + low + close) / Decimal(3)
                levels["pivot"] = pivot
                pivot_levels = {
                    "R1": (Decimal(2) * pivot) - low, "S1": (Decimal(2) * pivot) - high,
                    "R2": pivot + (high - low), "S2": pivot - (high - low),
                    "R3": high + Decimal(2) * (pivot - low), "S3": low - Decimal(2) * (high - pivot),
                }
                for label, value in pivot_levels.items():
                     if value < current_price: levels["support"][label] = value
                     else: levels["resistance"][label] = value
            except InvalidOperation as e: # Handles potential issues within pivot calcs
                self.logger.error(f"Invalid operation during pivot calculation: {e}")

        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.error(f"Error calculating levels for {self.symbol}: {e}")
        except Exception as e:
             self.logger.exception(f"Unexpected error calculating levels for {self.symbol}: {e}")

        return levels

    def _analyze_orderbook(self, orderbook: Optional[dict], current_price: Decimal, levels: dict) -> dict:
        """Analyzes order book for pressure and clusters near calculated levels."""
        analysis = {"clusters": [], "pressure": SignalState.NEUTRAL.value, "total_bid_usd": Decimal(0), "total_ask_usd": Decimal(0)}
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook or not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
            self.logger.debug("Orderbook data incomplete or unavailable for analysis.")
            return analysis

        try:
            # Convert to DataFrame, handling potential errors and ensuring Decimal type
            def to_decimal_df(data: List[List[Union[str, float]]], columns: List[str]) -> pd.DataFrame:
                df = pd.DataFrame(data, columns=columns)
                for col in columns:
                    # Convert to string first, then Decimal, coerce errors
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(lambda x: Decimal(x) if x not in ('', 'None') else pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce again if Decimal failed
                    df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal(0)) # Final conversion, fill NA with 0 Decimal
                return df

            bids_df = to_decimal_df(orderbook.get('bids', []), ['price', 'size'])
            asks_df = to_decimal_df(orderbook.get('asks', []), ['price', 'size'])

            # Filter out zero prices/sizes that might have resulted from conversion errors
            bids_df = bids_df[(bids_df['price'] > 0) & (bids_df['size'] > 0)]
            asks_df = asks_df[(asks_df['price'] > 0) & (asks_df['size'] > 0)]

            if bids_df.empty and asks_df.empty:
                self.logger.debug("Orderbook is empty after cleaning.")
                return analysis

            # Calculate USD value and total pressure
            bids_df['value_usd'] = bids_df['price'] * bids_df['size']
            asks_df['value_usd'] = asks_df['price'] * asks_df['size']

            total_bid_value = bids_df['value_usd'].sum()
            total_ask_value = asks_df['value_usd'].sum()
            analysis["total_bid_usd"] = total_bid_value
            analysis["total_ask_usd"] = total_ask_value

            total_value = total_bid_value + total_ask_value
            if total_value > 0:
                bid_ask_ratio = total_bid_value / total_value
                if bid_ask_ratio > Decimal("0.6"): analysis["pressure"] = Color.format("High Buy Pressure", Color.GREEN)
                elif bid_ask_ratio < Decimal("0.4"): analysis["pressure"] = Color.format("High Sell Pressure", Color.RED)
                else: analysis["pressure"] = Color.format("Neutral Pressure", Color.YELLOW)

            # Cluster Analysis
            cluster_threshold = Decimal(str(self.orderbook_settings.cluster_threshold_usd))
            proximity_pct = Decimal(str(self.orderbook_settings.cluster_proximity_pct)) / Decimal(100)

            # Combine support, resistance, and pivot for cluster checking
            all_levels = {**levels.get("support", {}), **levels.get("resistance", {})}
            if levels.get("pivot") is not None: all_levels["Pivot"] = levels["pivot"]

            processed_clusters = set() # Avoid duplicate clusters if levels overlap

            for name, level_price in all_levels.items():
                if not isinstance(level_price, Decimal) or level_price <= 0: continue

                min_price = level_price * (Decimal(1) - proximity_pct)
                max_price = level_price * (Decimal(1) + proximity_pct)

                # Bid clusters (potential support confirmation)
                bids_near = bids_df[(bids_df['price'] >= min_price) & (bids_df['price'] <= max_price)]
                bid_cluster_value = bids_near['value_usd'].sum()
                if bid_cluster_value >= cluster_threshold:
                    cluster_id = f"B_{name}_{level_price:.4f}" # Unique ID for the cluster
                    if cluster_id not in processed_clusters:
                        analysis["clusters"].append({
                            "type": "Support", "level_name": name, "level_price": level_price,
                            "cluster_value_usd": bid_cluster_value, "price_range": (min_price, max_price)
                        })
                        processed_clusters.add(cluster_id)

                # Ask clusters (potential resistance confirmation)
                asks_near = asks_df[(asks_df['price'] >= min_price) & (asks_df['price'] <= max_price)]
                ask_cluster_value = asks_near['value_usd'].sum()
                if ask_cluster_value >= cluster_threshold:
                     cluster_id = f"A_{name}_{level_price:.4f}"
                     if cluster_id not in processed_clusters:
                        analysis["clusters"].append({
                            "type": "Resistance", "level_name": name, "level_price": level_price,
                            "cluster_value_usd": ask_cluster_value, "price_range": (min_price, max_price)
                        })
                        processed_clusters.add(cluster_id)

        except (KeyError, ValueError, TypeError, InvalidOperation, AttributeError) as e:
            self.logger.error(f"Error analyzing orderbook for {self.symbol}: {e}")
            self.logger.debug(traceback.format_exc()) # Log stack trace for debugging
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing orderbook for {self.symbol}: {e}")

        # Sort clusters by value descending before returning
        analysis["clusters"] = sorted(analysis.get("clusters", []), key=lambda x: x['cluster_value_usd'], reverse=True)
        return analysis

    # --- Interpretation Helpers ---
    def _get_val(self, row: pd.Series, key: str, default: Any = None) -> Any:
        """Safely gets a value from a Pandas Series (row), handling missing keys."""
        val = row.get(key, default)
        # Return default if value is NaN or None
        return default if pd.isna(val) else val

    def _format_signal(self, label: str, value: Any, signal: SignalState, precision: int = 2, details: str = "") -> str:
        """Formats a signal line with color based on state."""
        value_str = format_decimal(value, precision)
        color_map = {
            SignalState.BULLISH: Color.GREEN, SignalState.STRONG_BULLISH: Color.GREEN, SignalState.OVERSOLD: Color.GREEN, SignalState.INCREASING: Color.GREEN, SignalState.ACCUMULATION: Color.GREEN, SignalState.FLIP_BULLISH: Color.GREEN, SignalState.BREAKDOWN_LOWER: Color.GREEN, SignalState.ABOVE_SIGNAL: Color.GREEN,
            SignalState.BEARISH: Color.RED, SignalState.STRONG_BEARISH: Color.RED, SignalState.OVERBOUGHT: Color.RED, SignalState.DECREASING: Color.RED, SignalState.DISTRIBUTION: Color.RED, SignalState.FLIP_BEARISH: Color.RED, SignalState.BREAKOUT_UPPER: Color.RED, SignalState.BELOW_SIGNAL: Color.RED,
            SignalState.NEUTRAL: Color.YELLOW, SignalState.RANGING: Color.YELLOW, SignalState.WITHIN_BANDS: Color.YELLOW, SignalState.AVERAGE_VOLUME: Color.YELLOW, SignalState.FLAT: Color.YELLOW, SignalState.NONE: Color.YELLOW,
            SignalState.HIGH_VOLUME: Color.PURPLE, SignalState.LOW_VOLUME: Color.PURPLE,
        }
        color = color_map.get(signal, Color.YELLOW) # Default to yellow
        signal_text = signal.value if signal != SignalState.NA else SignalState.NA.value
        details_text = f" ({details})" if details else ""
        return f"{label} ({value_str}): {Color.format(signal_text, color)}{details_text}"

    def _interpret_trend(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        # EMA Alignment
        if flags.ema_alignment and cols['ema_short'] in last_row and cols['ema_long'] in last_row:
            ema_short = self._get_val(last_row, cols['ema_short'])
            ema_long = self._get_val(last_row, cols['ema_long'])
            signal = SignalState.NA
            details = ""
            if ema_short is not None and ema_long is not None:
                signal = SignalState.NEUTRAL
                if ema_short > ema_long: signal = SignalState.BULLISH
                elif ema_short < ema_long: signal = SignalState.BEARISH
                details = f"S:{format_decimal(ema_short)} {'><'[ema_short > ema_long]} L:{format_decimal(ema_long)}"
            signals["ema_trend"] = signal
            summary.append(self._format_signal("EMA Align", "", signal, details=details))

        # ADX Trend Strength
        if flags.adx_trend_strength and cols['adx'] in last_row:
            adx = self._get_val(last_row, cols['adx'])
            dmp = self._get_val(last_row, cols.get('dmp'), 0) # +DI
            dmn = self._get_val(last_row, cols.get('dmn'), 0) # -DI
            signal = SignalState.NA
            details = ""
            if adx is not None and dmp is not None and dmn is not None:
                trend_threshold = Decimal(str(self.thresholds.adx_trending))
                if adx >= trend_threshold:
                    if dmp > dmn:
                        signal = SignalState.STRONG_BULLISH
                        details = "+DI > -DI"
                    else:
                        signal = SignalState.STRONG_BEARISH
                        details = "-DI > +DI"
                else:
                    signal = SignalState.RANGING
                    details = f"ADX < {trend_threshold}"
            signals["adx_trend"] = signal
            summary.append(self._format_signal("ADX Trend", adx, signal, details=details))

        # PSAR Flip
        if flags.psar_flip:
            psar_val = None
            psar_l = self._get_val(last_row, cols['psar_long'])
            psar_s = self._get_val(last_row, cols['psar_short'])
            close_val = self._get_val(last_row, 'close')
            psar_rev = self._get_val(last_row, cols.get('psar_rev')) # Reversal signal

            signal = SignalState.NA
            trend_signal = SignalState.NA
            details = ""

            if close_val is not None:
                # Determine current PSAR value and trend based on price position
                if psar_l is not None and close_val > psar_l:
                    psar_val = psar_l
                    trend_signal = SignalState.BULLISH
                elif psar_s is not None and close_val < psar_s:
                    psar_val = psar_s
                    trend_signal = SignalState.BEARISH
                elif psar_l is not None: # Price between PSARl and PSARs? Use the one closer? Or last known trend?
                    psar_val = psar_l # Default to long if ambiguous
                    trend_signal = SignalState.BULLISH if close_val > psar_l else SignalState.BEARISH
                elif psar_s is not None:
                     psar_val = psar_s
                     trend_signal = SignalState.BEARISH

            if psar_val is not None:
                 signal = trend_signal # Base signal is the current trend
                 # Check for flip using the reversal column 'PSARr'
                 flipped = psar_rev is not None and psar_rev == 1 # Check if reversal happened on this candle
                 if flipped:
                     signal = SignalState.FLIP_BULLISH if trend_signal == SignalState.BULLISH else SignalState.FLIP_BEARISH
                     details = "Just Flipped!"

            signals["psar_trend"] = trend_signal # Store the underlying trend
            signals["psar_signal"] = signal # Store the flip signal or trend
            summary.append(self._format_signal("PSAR", psar_val, signal, precision=4, details=details))

        return summary, signals

    def _interpret_oscillators(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        thresh = self.thresholds
        cols = self.col_names

        # RSI
        if flags.rsi_threshold and cols['rsi'] in last_row:
            rsi = self._get_val(last_row, cols['rsi'])
            signal = SignalState.NA
            if rsi is not None:
                ob = Decimal(str(thresh.rsi_overbought))
                os = Decimal(str(thresh.rsi_oversold))
                signal = SignalState.NEUTRAL
                if rsi >= ob: signal = SignalState.OVERBOUGHT
                elif rsi <= os: signal = SignalState.OVERSOLD
            signals["rsi_level"] = signal
            summary.append(self._format_signal("RSI", rsi, signal))

        # MFI
        if flags.mfi_threshold and cols['mfi'] in last_row:
            mfi = self._get_val(last_row, cols['mfi'])
            signal = SignalState.NA
            if mfi is not None:
                ob = Decimal(str(thresh.mfi_overbought))
                os = Decimal(str(thresh.mfi_oversold))
                signal = SignalState.NEUTRAL
                if mfi >= ob: signal = SignalState.OVERBOUGHT
                elif mfi <= os: signal = SignalState.OVERSOLD
            signals["mfi_level"] = signal
            summary.append(self._format_signal("MFI", mfi, signal))

        # CCI
        if flags.cci_threshold and cols['cci'] in last_row:
             cci = self._get_val(last_row, cols['cci'])
             signal = SignalState.NA
             if cci is not None:
                 ob = Decimal(str(thresh.cci_overbought))
                 os = Decimal(str(thresh.cci_oversold))
                 signal = SignalState.NEUTRAL
                 if cci >= ob: signal = SignalState.OVERBOUGHT
                 elif cci <= os: signal = SignalState.OVERSOLD
             signals["cci_level"] = signal
             summary.append(self._format_signal("CCI", cci, signal))

        # Williams %R
        if flags.williams_r_threshold and cols['willr'] in last_row:
             wr = self._get_val(last_row, cols['willr'])
             signal = SignalState.NA
             if wr is not None:
                 ob = Decimal(str(thresh.williams_r_overbought)) # Note: OB is higher value (e.g., -20)
                 os = Decimal(str(thresh.williams_r_oversold)) # Note: OS is lower value (e.g., -80)
                 signal = SignalState.NEUTRAL
                 if wr >= ob: signal = SignalState.OVERBOUGHT
                 elif wr <= os: signal = SignalState.OVERSOLD
             signals["wr_level"] = signal
             summary.append(self._format_signal("Williams %R", wr, signal))

        # StochRSI Cross
        if flags.stoch_rsi_cross and cols['stochrsi_k'] in last_row and cols['stochrsi_d'] in last_row:
            k_now = self._get_val(last_row, cols['stochrsi_k'])
            d_now = self._get_val(last_row, cols['stochrsi_d'])
            k_prev = self._get_val(prev_row, cols['stochrsi_k'])
            d_prev = self._get_val(prev_row, cols['stochrsi_d'])
            signal = SignalState.NA
            details = ""
            if k_now is not None and d_now is not None and k_prev is not None and d_prev is not None:
                crossed_bullish = k_now > d_now and k_prev <= d_prev
                crossed_bearish = k_now < d_now and k_prev >= d_prev
                if crossed_bullish:
                    signal = SignalState.BULLISH
                    details = "Crossed Up"
                elif crossed_bearish:
                    signal = SignalState.BEARISH
                    details = "Crossed Down"
                elif k_now > d_now: # Indicate current state if no cross
                    signal = SignalState.BULLISH
                    details = "K > D"
                elif k_now < d_now:
                    signal = SignalState.BEARISH
                    details = "K < D"
                else:
                    signal = SignalState.NEUTRAL
                    details = "K == D"

            signals["stochrsi_cross"] = signal
            summary.append(self._format_signal(f"StochRSI", f"K:{format_decimal(k_now)} D:{format_decimal(d_now)}", signal, details=details))

        return summary, signals

    def _interpret_macd(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        # MACD Cross
        if flags.macd_cross and cols['macd_line'] in last_row and cols['macd_signal'] in last_row:
            line_now = self._get_val(last_row, cols['macd_line'])
            sig_now = self._get_val(last_row, cols['macd_signal'])
            line_prev = self._get_val(prev_row, cols['macd_line'])
            sig_prev = self._get_val(prev_row, cols['macd_signal'])
            signal = SignalState.NA
            details = ""
            if line_now is not None and sig_now is not None and line_prev is not None and sig_prev is not None:
                crossed_bullish = line_now > sig_now and line_prev <= sig_prev
                crossed_bearish = line_now < sig_now and line_prev >= sig_prev
                if crossed_bullish:
                    signal = SignalState.BULLISH
                    details = "Crossed Up"
                elif crossed_bearish:
                    signal = SignalState.BEARISH
                    details = "Crossed Down"
                elif line_now > sig_now:
                    signal = SignalState.ABOVE_SIGNAL # Current state
                    details = "Line > Signal"
                elif line_now < sig_now:
                    signal = SignalState.BELOW_SIGNAL # Current state
                    details = "Line < Signal"
                else:
                    signal = SignalState.NEUTRAL
                    details = "Line == Signal"

            signals["macd_cross"] = signal
            summary.append(self._format_signal(f"MACD", f"L:{format_decimal(line_now, 4)} S:{format_decimal(sig_now, 4)}", signal, details=details))

        # MACD Divergence (Basic 2-point check)
        # WARNING: Simple check, prone to false signals. Real divergence needs pattern recognition.
        if flags.macd_divergence and cols['macd_hist'] in last_row and 'close' in last_row:
             hist_now = self._get_val(last_row, cols['macd_hist'])
             hist_prev = self._get_val(prev_row, cols['macd_hist'])
             price_now = self._get_val(last_row, 'close')
             price_prev = self._get_val(prev_row, 'close')
             signal = SignalState.NA
             if hist_now is not None and hist_prev is not None and price_now is not None and price_prev is not None:
                 signal = SignalState.NONE
                 # Bullish: Lower low in price, higher low in histogram (Must cross zero or be below zero)
                 if price_now < price_prev and hist_now > hist_prev and (hist_prev < 0 or hist_now < 0):
                     signal = SignalState.BULLISH
                     summary.append(Color.format("Potential Bullish MACD Divergence", Color.GREEN))
                 # Bearish: Higher high in price, lower high in histogram (Must cross zero or be above zero)
                 elif price_now > price_prev and hist_now < hist_prev and (hist_prev > 0 or hist_now > 0):
                     signal = SignalState.BEARISH
                     summary.append(Color.format("Potential Bearish MACD Divergence", Color.RED))
             signals["macd_divergence"] = signal
             # No separate line added if no divergence found

        return summary, signals

    def _interpret_bbands(self, last_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        if flags.bollinger_bands_break and cols['bb_upper'] in last_row and cols['bb_lower'] in last_row:
             upper = self._get_val(last_row, cols['bb_upper'])
             lower = self._get_val(last_row, cols['bb_lower'])
             middle = self._get_val(last_row, cols['bb_mid'])
             close_val = self._get_val(last_row, 'close')
             signal = SignalState.NA
             details = ""
             if upper is not None and lower is not None and middle is not None and close_val is not None:
                signal = SignalState.WITHIN_BANDS
                details = f"L:{format_decimal(lower)} M:{format_decimal(middle)} U:{format_decimal(upper)}"
                if close_val > upper:
                    signal = SignalState.BREAKOUT_UPPER
                    details += " (Price > Upper)"
                elif close_val < lower:
                     signal = SignalState.BREAKDOWN_LOWER
                     details += " (Price < Lower)"
             signals["bbands_signal"] = signal
             summary.append(self._format_signal("BBands", close_val, signal, details=details))

        return summary, signals

    def _interpret_volume(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        # Volume MA Comparison
        if flags.volume_confirmation and cols['vol_ma'] in last_row and 'volume' in last_row:
            volume = self._get_val(last_row, 'volume')
            vol_ma = self._get_val(last_row, cols['vol_ma'])
            signal = SignalState.NA
            details = ""
            if volume is not None and vol_ma is not None and vol_ma > 0:
                signal = SignalState.AVERAGE_VOLUME
                # Define thresholds relative to MA
                high_thresh = Decimal("1.5")
                low_thresh = Decimal("0.7")
                if volume > vol_ma * high_thresh: signal = SignalState.HIGH_VOLUME
                elif volume < vol_ma * low_thresh: signal = SignalState.LOW_VOLUME
                details = f"Vol:{format_decimal(volume,0)} MA:{format_decimal(vol_ma,0)}"
            signals["volume_level"] = signal
            summary.append(self._format_signal("Volume", "", signal, details=details))

        # OBV Trend
        if flags.obv_trend and cols['obv'] in last_row:
             obv_now = self._get_val(last_row, cols['obv'])
             obv_prev = self._get_val(prev_row, cols['obv'])
             signal = SignalState.NA
             if obv_now is not None and obv_prev is not None:
                 signal = SignalState.FLAT
                 if obv_now > obv_prev: signal = SignalState.INCREASING
                 elif obv_now < obv_prev: signal = SignalState.DECREASING
             signals["obv_trend"] = signal
             summary.append(self._format_signal("OBV Trend", obv_now, signal, precision=0))

        # ADOSC Trend (A/D Oscillator)
        if flags.adi_trend and cols['adosc'] in last_row:
             adosc_now = self._get_val(last_row, cols['adosc'])
             adosc_prev = self._get_val(prev_row, cols['adosc'])
             signal = SignalState.NA
             if adosc_now is not None and adosc_prev is not None:
                 trend = SignalState.FLAT
                 if adosc_now > 0 and adosc_now > adosc_prev: trend = SignalState.ACCUMULATION # Accumulation: Positive and rising
                 elif adosc_now < 0 and adosc_now < adosc_prev: trend = SignalState.DISTRIBUTION # Distribution: Negative and falling
                 elif adosc_now > adosc_prev: trend = SignalState.INCREASING # Generic rise if not clear A/D
                 elif adosc_now < adosc_prev: trend = SignalState.DECREASING # Generic fall
                 signal = trend
             signals["adi_trend"] = signal
             summary.append(self._format_signal("A/D Osc Trend", adosc_now, signal, precision=0))

        return summary, signals

    def _interpret_levels_orderbook(self, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> List[str]:
        """Formats the levels and orderbook analysis summary."""
        summary = []
        summary.append(Color.format("--- Levels & Orderbook ---", Color.BLUE))

        # Levels Summary
        pivot = levels.get("pivot")
        if pivot: summary.append(f"Pivot Point: ${format_decimal(pivot, 4)}")

        nearest_supports = sorted(levels.get("support", {}).items(), key=lambda item: abs(item[1] - current_price))[:3]
        nearest_resistances = sorted(levels.get("resistance", {}).items(), key=lambda item: abs(item[1] - current_price))[:3]

        if nearest_supports:
            summary.append("Nearest Support:")
            for name, price in nearest_supports: summary.append(f"  > {name}: ${format_decimal(price, 4)}")
        if nearest_resistances:
            summary.append("Nearest Resistance:")
            for name, price in nearest_resistances: summary.append(f"  > {name}: ${format_decimal(price, 4)}")
        if not nearest_supports and not nearest_resistances and not pivot:
             summary.append(Color.format("No significant levels calculated.", Color.YELLOW))

        # Orderbook Summary
        if orderbook_analysis and orderbook_analysis.get("total_bid_usd", 0) + orderbook_analysis.get("total_ask_usd", 0) > 0:
             pressure = orderbook_analysis.get('pressure', SignalState.NA.value) # Already formatted with color
             total_bid = orderbook_analysis.get('total_bid_usd', Decimal(0))
             total_ask = orderbook_analysis.get('total_ask_usd', Decimal(0))
             limit = self.orderbook_settings.limit

             summary.append(f"OB Pressure (Top {limit}): {pressure}")
             summary.append(f"OB Value (Bids): ${format_decimal(total_bid, 0)}")
             summary.append(f"OB Value (Asks): ${format_decimal(total_ask, 0)}")

             clusters = orderbook_analysis.get("clusters", [])
             if clusters:
                 summary.append(Color.format(f"Significant OB Clusters (Top 5):", Color.PURPLE))
                 for cluster in clusters[:5]:
                      color = Color.GREEN if cluster["type"] == "Support" else Color.RED
                      level_price_f = format_decimal(cluster['level_price'], 4)
                      cluster_val_f = format_decimal(cluster['cluster_value_usd'], 0)
                      summary.append(Color.format(f"  {cluster['type']} near {cluster['level_name']} (${level_price_f}) - Value: ${cluster_val_f}", color))
             else:
                summary.append(Color.format("No significant OB clusters found.", Color.YELLOW))
        else:
             summary.append(Color.format("Orderbook analysis unavailable or empty.", Color.YELLOW))

        return summary

    def _interpret_analysis(self, df: pd.DataFrame, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> dict:
        """Combines all interpretation steps."""
        interpretation = {"summary": [], "signals": {}}
        if df.empty or len(df) < 2:
            self.logger.warning(f"Insufficient data ({len(df)} rows) for interpretation on {self.symbol}.")
            interpretation["summary"].append(Color.format("Insufficient data for analysis.", Color.RED))
            # Add NA signals for all expected keys
            all_signal_keys = ["ema_trend", "adx_trend", "psar_trend", "psar_signal", "rsi_level", "mfi_level", "cci_level", "wr_level", "stochrsi_cross", "macd_cross", "macd_divergence", "bbands_signal", "volume_level", "obv_trend", "adi_trend"]
            for key in all_signal_keys:
                interpretation["signals"][key] = SignalState.NA
            return interpretation

        try:
             # Ensure index is sequential for iloc[-1] and [-2]
             df_indexed = df.reset_index(drop=True)
             last_row = df_indexed.iloc[-1]
             # Use last_row as prev_row if only one row exists after indicator calculation (though unlikely with checks)
             prev_row = df_indexed.iloc[-2] if len(df_indexed) >= 2 else last_row

             trend_summary, trend_signals = self._interpret_trend(last_row, prev_row)
             osc_summary, osc_signals = self._interpret_oscillators(last_row, prev_row)
             macd_summary, macd_signals = self._interpret_macd(last_row, prev_row)
             bb_summary, bb_signals = self._interpret_bbands(last_row)
             vol_summary, vol_signals = self._interpret_volume(last_row, prev_row)
             level_summary = self._interpret_levels_orderbook(current_price, levels, orderbook_analysis)

             interpretation["summary"].extend(trend_summary)
             interpretation["summary"].extend(osc_summary)
             interpretation["summary"].extend(macd_summary)
             interpretation["summary"].extend(bb_summary)
             interpretation["summary"].extend(vol_summary)
             interpretation["summary"].extend(level_summary)

             # Combine all signals, converting Enum members to their string values for JSON serialization
             all_signals = {**trend_signals, **osc_signals, **macd_signals, **bb_signals, **vol_signals}
             interpretation["signals"] = {k: v.value for k, v in all_signals.items()}


        except IndexError as e:
             self.logger.error(f"IndexError during interpretation for {self.symbol}. DataFrame length: {len(df)}. Error: {e}")
             interpretation["summary"].append(Color.format("Error accessing data for interpretation.", Color.RED))
             interpretation["signals"] = {k: SignalState.NA.value for k in interpretation["signals"]} # Set all signals to NA on error
        except Exception as e:
             self.logger.exception(f"Unexpected error during analysis interpretation for {self.symbol}: {e}")
             interpretation["summary"].append(Color.format(f"Error during interpretation: {e}", Color.RED))
             interpretation["signals"] = {k: SignalState.NA.value for k in interpretation["signals"]}

        return interpretation

    def analyze(self, df_klines: pd.DataFrame, current_price: Decimal, orderbook: Optional[dict]) -> dict:
        """Main analysis function orchestrating calculation and interpretation."""
        analysis_result = {
            "symbol": self.symbol,
            "timestamp": datetime.now(APP_TIMEZONE).isoformat(),
            "current_price": format_decimal(current_price, 4),
            "kline_interval": "N/A",
            "levels": {},
            "orderbook_analysis": {},
            "interpretation": {"summary": [Color.format("Analysis could not be performed.", Color.RED)], "signals": {}},
            "raw_indicators": {}
        }

        if df_klines.empty:
            self.logger.error(f"Kline data is empty for {self.symbol}, cannot perform analysis.")
            return analysis_result
        # Warning if less than 2 rows, as interpretation relies on prev_row
        if len(df_klines) < 2:
             self.logger.warning(f"Kline data has only {len(df_klines)} row(s) for {self.symbol}, analysis may be incomplete.")
             analysis_result["interpretation"]["summary"] = [Color.format("Warning: Insufficient kline data for full analysis.", Color.YELLOW)]
             # Allow analysis to proceed, but interpretation might yield N/A for some signals

        try:
             # Infer Interval from kline data timestamps
             if len(df_klines) >= 2:
                time_diff = df_klines['timestamp'].diff().min() # Use min difference to handle potential gaps
                if pd.notna(time_diff):
                    # Convert timedelta to a readable string (e.g., '0 days 01:00:00')
                    analysis_result["kline_interval"] = str(time_diff)
                else:
                     analysis_result["kline_interval"] = "N/A (single candle?)"


             # 1. Calculate Indicators
             df_with_indicators = self._calculate_indicators(df_klines)
             if df_with_indicators.empty:
                 self.logger.error(f"Indicator calculation failed or resulted in empty DataFrame for {self.symbol}.")
                 # Keep the initial error message in summary
                 return analysis_result

             # Store raw indicator values from the last row
             if not df_with_indicators.empty:
                last_row_indicators = df_with_indicators.iloc[-1].to_dict()
                analysis_result["raw_indicators"] = {
                    k: format_decimal(v, 4) if isinstance(v, (Decimal, float, np.floating)) else
                       (v.isoformat() if isinstance(v, pd.Timestamp) else str(v) if pd.notna(v) else None)
                    for k, v in last_row_indicators.items()
                }

             # 2. Calculate Levels
             levels = self._calculate_levels(df_with_indicators, current_price)
             analysis_result["levels"] = {
                 k: {name: format_decimal(price, 4) for name, price in v.items()} if isinstance(v, dict)
                   else (format_decimal(v, 4) if isinstance(v, Decimal) else v)
                 for k, v in levels.items()
             }

             # 3. Analyze Orderbook
             orderbook_analysis_raw = self._analyze_orderbook(orderbook, current_price, levels)
             # Format orderbook analysis results for output
             analysis_result["orderbook_analysis"] = {
                "pressure": orderbook_analysis_raw.get("pressure", SignalState.NA.value), # Pressure is already formatted
                "total_bid_usd": format_decimal(orderbook_analysis_raw.get("total_bid_usd", Decimal(0)), 0),
                "total_ask_usd": format_decimal(orderbook_analysis_raw.get("total_ask_usd", Decimal(0)), 0),
                "clusters": [
                    {
                        "type": c.get("type"), "level_name": c.get("level_name"),
                        "level_price": format_decimal(c.get("level_price"), 4),
                        "cluster_value_usd": format_decimal(c.get("cluster_value_usd"), 0),
                        "price_range": (format_decimal(c.get("price_range", (None, None))[0], 4), format_decimal(c.get("price_range", (None, None))[1], 4))
                    } for c in orderbook_analysis_raw.get("clusters", [])
                ]
             }

             # 4. Interpret Results
             interpretation = self._interpret_analysis(df_with_indicators, current_price, levels, orderbook_analysis_raw)
             analysis_result["interpretation"] = interpretation

        except Exception as e:
            self.logger.exception(f"Critical error during analysis pipeline for {self.symbol}: {e}")
            analysis_result["interpretation"]["summary"] = [Color.format(f"Critical Analysis Error: {e}", Color.RED)]
            # Ensure signals are marked as NA in case of pipeline error
            analysis_result["interpretation"]["signals"] = {k: SignalState.NA.value for k in analysis_result["interpretation"].get("signals", {})}


        return analysis_result

    def format_analysis_output(self, analysis_result: dict) -> str:
        """Formats the analysis result dictionary into a readable string."""
        symbol = analysis_result.get('symbol', 'N/A')
        timestamp_str = analysis_result.get('timestamp', 'N/A')
        try:
            # Attempt to parse and format timestamp nicely
            dt_obj = datetime.fromisoformat(timestamp_str).astimezone(APP_TIMEZONE)
            ts_formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, TypeError):
            ts_formatted = timestamp_str # Fallback to raw string

        price = analysis_result.get('current_price', 'N/A')
        interval = analysis_result.get('kline_interval', 'N/A') # This is inferred interval

        output = f"\n--- Analysis Report for {symbol} --- ({ts_formatted})\n"
        output += f"{Color.format('Interval:', Color.BLUE)} {interval}    {Color.format('Current Price:', Color.BLUE)} ${price}\n"

        interpretation = analysis_result.get("interpretation", {})
        summary_lines = interpretation.get("summary", [])

        if summary_lines:
             output += "\n" + "\n".join(summary_lines) + "\n"
        else:
             output += Color.format("No interpretation summary available.", Color.YELLOW) + "\n"

        return output


# --- Main Application Logic ---
async def run_analysis_loop(symbol: str, interval_config: str, client: BybitCCXTClient, analyzer: TradingAnalyzer, logger_instance: logging.Logger):
    """The main loop performing periodic analysis."""
    analysis_interval_sec = CONFIG.analysis_interval_seconds
    kline_limit = CONFIG.kline_limit
    orderbook_limit = analyzer.orderbook_settings.limit

    if analysis_interval_sec < 10:
        logger_instance.warning(f"Analysis interval ({analysis_interval_sec}s) is very short. Ensure system and API can handle the load.")

    while True:
        start_time = time.monotonic()
        logger_instance.debug(f"--- Starting Analysis Cycle for {symbol} ---")

        analysis_result = None # Define outside try block

        try:
            # Fetch data concurrently
            price_task = asyncio.create_task(client.fetch_current_price(symbol))
            klines_task = asyncio.create_task(client.fetch_klines(symbol, interval_config, kline_limit))
            orderbook_task = asyncio.create_task(client.fetch_orderbook(symbol, orderbook_limit))

            # Wait for critical data (price and klines) first
            done, pending = await asyncio.wait([price_task, klines_task], return_when=asyncio.ALL_COMPLETED)

            # Check results of critical tasks
            current_price = None
            df_klines = pd.DataFrame()
            for task in done:
                if task == price_task:
                    current_price = task.result()
                elif task == klines_task:
                    df_klines = task.result()

            if current_price is None or df_klines is None or df_klines.empty: # Check df_klines is not None before checking empty
                logger_instance.error(f"Failed to fetch critical data (price or klines) for {symbol}. Skipping analysis cycle.")
                # Cancel pending orderbook task if critical data failed
                if orderbook_task not in done: orderbook_task.cancel()
                await async_sleep_with_jitter(INITIAL_RETRY_DELAY_SECONDS) # Wait before retrying fetch
                continue

            # Fetch orderbook (already started), wait with a timeout
            orderbook = None
            try:
                # Wait for the already running task, don't create a new one
                orderbook = await asyncio.wait_for(orderbook_task, timeout=15.0)
            except asyncio.TimeoutError:
                 logger_instance.warning(f"Fetching orderbook for {symbol} timed out. Proceeding without it.")
                 # Task is automatically cancelled by wait_for on timeout
            except Exception as ob_err:
                logger_instance.error(f"Error fetching or processing orderbook task result for {symbol}: {ob_err}")
                # Ensure task is cancelled if it errored but didn't finish
                if not orderbook_task.done(): orderbook_task.cancel()


            # Perform analysis
            analysis_result = analyzer.analyze(df_klines, current_price, orderbook)

            # Output and Logging
            output_string = analyzer.format_analysis_output(analysis_result)
            # Log the formatted output (which includes colors for the console)
            # The file logger's formatter will strip colors.
            logger_instance.info(output_string)

            # Log structured JSON if debug level is enabled
            if logger_instance.isEnabledFor(logging.DEBUG):
                try:
                    # Create a deep copy to avoid modifying the original result
                    log_data = json.loads(json.dumps(analysis_result, default=str)) # Basic serialization first
                    # Remove color codes from the summary in the copy
                    if 'interpretation' in log_data and 'summary' in log_data['interpretation']:
                         log_data['interpretation']['summary'] = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in log_data['interpretation']['summary']]
                    # Remove color codes from OB pressure
                    if 'orderbook_analysis' in log_data and 'pressure' in log_data['orderbook_analysis']:
                         log_data['orderbook_analysis']['pressure'] = re.sub(r'\x1b\[[0-9;]*m', '', log_data['orderbook_analysis']['pressure'])

                    logger_instance.debug(json.dumps(log_data, indent=2, default=str)) # Use default=str for any remaining non-serializable types
                except Exception as json_err:
                     logger_instance.error(f"Error preparing analysis result for JSON debug logging: {json_err}")


        except ccxt.AuthenticationError as e:
            logger_instance.critical(Color.format(f"Authentication Error: {e}. Check API Key/Secret. Stopping.", Color.RED))
            break # Stop the loop on auth error
        except ccxt.InvalidNonce as e:
             logger_instance.critical(Color.format(f"Invalid Nonce Error: {e}. Check system time sync. Stopping.", Color.RED))
             break # Stop the loop on nonce error
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
             # These might occur if trading logic were added, log as error but continue analysis
             logger_instance.error(Color.format(f"Order-related ccxt error: {e}. Check parameters or funds.", Color.RED))
        except asyncio.CancelledError:
             logger_instance.info("Analysis task cancelled.")
             break
        except Exception as e:
            logger_instance.exception(Color.format(f"An unexpected error occurred in the main analysis loop for {symbol}: {e}", Color.RED))
            # Consider adding a longer sleep here to avoid rapid error loops
            await async_sleep_with_jitter(10) # Sleep 10s after unexpected error

        # Sleep management
        elapsed_time = time.monotonic() - start_time
        sleep_duration = max(0, analysis_interval_sec - elapsed_time)
        logger_instance.debug(f"Cycle took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s.")
        if elapsed_time > analysis_interval_sec:
             logger_instance.warning(f"Analysis cycle took longer ({elapsed_time:.2f}s) than configured interval ({analysis_interval_sec}s). Consider increasing interval or optimizing.")

        await async_sleep_with_jitter(sleep_duration) # Use sleep with jitter

    logger_instance.info(f"Analysis loop stopped for {symbol}.")


async def main():
    """Main entry point for the application."""
    # Setup a temporary logger for initialization
    init_logger = setup_logger("INIT")
    init_logger.info("Application starting...")

    # Initialize client to load markets for validation
    init_client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, init_logger)
    markets_loaded = await init_client.initialize_markets()

    if not markets_loaded or not init_client.markets:
        init_logger.critical(Color.format("Failed to load markets. Cannot proceed. Exiting.", Color.RED))
        await init_client.close()
        logging.shutdown()
        return

    valid_symbols = list(init_client.markets.keys())
    # Keep the loaded markets data to pass to the main client instance
    loaded_markets_data = init_client.markets
    loaded_market_categories = init_client.market_categories
    await init_client.close() # Close the temporary client

    # --- User Input for Symbol and Interval ---
    symbol = ""
    while True:
        try:
            symbol_input = input(Color.format("Enter trading symbol (e.g., BTC/USDT): ", Color.BLUE)).upper().strip()
            if not symbol_input: continue

            # Standardize potential variations (e.g., BTCUSDT to BTC/USDT)
            potential_symbol = symbol_input
            if "/" not in symbol_input and len(symbol_input) > 4:
                # Basic guess based on common quote currencies
                quotes = ['USDT', 'USD', 'BTC', 'ETH', 'EUR', 'GBP']
                for quote in quotes:
                    if symbol_input.endswith(quote):
                        base = symbol_input[:-len(quote)]
                        formatted = f"{base}/{quote}"
                        if formatted in valid_symbols:
                            print(Color.format(f"Assuming you mean: {formatted}", Color.YELLOW))
                            potential_symbol = formatted
                            break # Found a match

            if potential_symbol in valid_symbols:
                symbol = potential_symbol
                break
            else:
                print(Color.format(f"Invalid or unknown symbol: '{symbol_input}'.", Color.RED))
                # Suggest similar symbols (simple substring match)
                find_similar = [s for s in valid_symbols if symbol_input in s or symbol_input.replace('/','') in s.replace('/','')]
                if find_similar:
                     print(Color.format(f"Did you mean one of these? {', '.join(find_similar[:5])}", Color.YELLOW))
        except EOFError:
             print(Color.format("\nInput stream closed. Exiting.", Color.YELLOW))
             return

    interval_key = ""
    default_interval = CONFIG.indicator_settings.default_interval
    while True:
        try:
            interval_input = input(Color.format(f"Enter timeframe [{', '.join(VALID_INTERVALS)}] (default: {default_interval}): ", Color.BLUE)).strip()
            if not interval_input:
                 interval_key = default_interval
                 print(Color.format(f"Using default interval: {interval_key}", Color.YELLOW))
                 break
            if interval_input in VALID_INTERVALS:
                interval_key = interval_input
                break
            # Allow CCXT standard intervals like '1h', '1d'
            if interval_input in REVERSE_CCXT_INTERVAL_MAP:
                interval_key = REVERSE_CCXT_INTERVAL_MAP[interval_input]
                print(Color.format(f"Using interval {interval_key} (mapped from {interval_input})", Color.YELLOW))
                break
            print(Color.format(f"Invalid interval. Choose from: {', '.join(VALID_INTERVALS)} or compatible (e.g., 1h, 4h, 1d)", Color.RED))
        except EOFError:
             print(Color.format("\nInput stream closed. Exiting.", Color.YELLOW))
             return

    # --- Setup Main Components ---
    main_logger = setup_logger(symbol) # Setup logger specific to the chosen symbol
    main_logger.info("Logger initialized.")

    client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, main_logger)
    # Pass the already loaded market data to avoid fetching again
    client.markets = loaded_markets_data
    client.market_categories = loaded_market_categories # Pass cached categories
    main_logger.info(f"Client initialized for {symbol}. Markets loaded.")

    analyzer = TradingAnalyzer(CONFIG, main_logger, symbol)
    main_logger.info("Analyzer initialized.")

    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_key, "N/A")
    market_type = client.get_market_category(symbol)

    main_logger.info(f"Starting analysis for {Color.format(symbol, Color.PURPLE)} ({market_type}) on interval {Color.format(interval_key, Color.PURPLE)} ({ccxt_interval}).")
    main_logger.info(f"Using API Environment: {Color.format(API_ENV.upper(), Color.YELLOW)} ({client.exchange.urls['api']})")
    main_logger.info(f"Analysis loop interval: {CONFIG.analysis_interval_seconds} seconds.")
    main_logger.info(f"Kline fetch limit: {CONFIG.kline_limit} candles.")
    main_logger.info(f"Using Timezone: {APP_TIMEZONE}")


    # --- Run Analysis Loop ---
    main_task = None
    try:
        main_task = asyncio.create_task(run_analysis_loop(symbol, interval_key, client, analyzer, main_logger))
        await main_task
    except KeyboardInterrupt:
        main_logger.info(Color.format("Analysis stopped by user (KeyboardInterrupt).", Color.YELLOW))
        if main_task: main_task.cancel()
    except asyncio.CancelledError:
         main_logger.info(Color.format("Main task cancelled.", Color.YELLOW))
    except Exception as e:
         main_logger.critical(Color.format(f"Critical error during main execution: {e}", Color.RED), exc_info=True)
         if main_task: main_task.cancel()
    finally:
        main_logger.info("Shutting down...")
        await client.close() # Ensure client connection is closed
        main_logger.info("Application finished.")
        # Flush handlers and shutdown logging system
        # Get all handlers associated with this logger instance
        handlers = main_logger.handlers[:]
        for handler in handlers:
            try:
                handler.flush()
                handler.close()
                main_logger.removeHandler(handler)
            except Exception as e:
                 print(f"Error closing handler {handler}: {e}") # Use print as logger might be closing
        logging.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles Ctrl+C before the main async loop starts or after it exits
        print(f"\n{Color.YELLOW.value}Process interrupted by user. Exiting gracefully.{Color.RESET.value}")
    except Exception as e:
        # Catch any other unexpected top-level errors
        print(f"\n{Color.RED.value}A critical top-level error occurred: {e}{Color.RESET.value}")
        traceback.print_exc()
