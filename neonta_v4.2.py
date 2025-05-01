# -*- coding: utf-8 -*-
"""
Neonta v4.1: Cryptocurrency Technical Analysis Bot (Enhanced)

This script performs technical analysis on cryptocurrency pairs using data
fetched from the Bybit exchange via the ccxt library. It calculates various
technical indicators, identifies potential support/resistance levels, analyzes
order book data, and provides an interpretation of the market state.
"""

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
from decimal import Decimal, InvalidOperation, getcontext, ROUND_HALF_UP
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
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
init(autoreset=True)  # Initialize colorama
load_dotenv()         # Load environment variables from .env file
getcontext().prec = 28  # Set global Decimal precision (increased from 18 for safety)

# --- Constants ---
CONFIG_FILE_NAME = "config.json"
LOG_DIRECTORY_NAME = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default if not set or invalid
MAX_API_RETRIES = 5
INITIAL_RETRY_DELAY_SECONDS = 5.0
MAX_RETRY_DELAY_SECONDS = 60.0
CCXT_TIMEOUT_MS = 20000 # Milliseconds for CCXT requests
MAX_JITTER_FACTOR = 0.2 # Max jitter for sleep intervals (e.g., 0.2 = 20%)
DECIMAL_COMPARISON_THRESHOLD = Decimal("1e-12") # For near-zero checks
DECIMAL_ZERO = Decimal(0) # Constant for zero Decimal

# Bybit API Configuration (Ensure these are set in your .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print here as logger might not be fully set up
    print(f"{Fore.RED}Error: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file{Style.RESET_ALL}")
    raise ValueError("Missing Bybit API credentials in .env file")

API_ENV = os.getenv("API_ENVIRONMENT", "prod").lower() # 'prod' or 'test'
IS_TESTNET = API_ENV == 'test'

# Timezone Configuration
_user_tz_str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    APP_TIMEZONE = ZoneInfo(_user_tz_str)
except ZoneInfoNotFoundError:
    print(f"{Fore.YELLOW}Warning: Timezone '{_user_tz_str}' not found. Using UTC.{Style.RESET_ALL}")
    APP_TIMEZONE = ZoneInfo("UTC")

# Paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = BASE_DIR / CONFIG_FILE_NAME
LOG_DIRECTORY = BASE_DIR / LOG_DIRECTORY_NAME
try:
    LOG_DIRECTORY.mkdir(exist_ok=True) # Ensure log directory exists
except OSError as e:
    print(f"{Fore.RED}Error creating log directory '{LOG_DIRECTORY}': {e}{Style.RESET_ALL}")
    # Decide if this is fatal or if logging can be disabled/redirected
    raise

# Timeframes (Mapping user input to CCXT intervals)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}
REVERSE_CCXT_INTERVAL_MAP = {v: k for k, v in CCXT_INTERVAL_MAP.items()}

# Color Constants Enum
class Color(Enum):
    """Enum for storing colorama color codes."""
    GREEN = Fore.LIGHTGREEN_EX
    BLUE = Fore.CYAN
    PURPLE = Fore.MAGENTA
    YELLOW = Fore.YELLOW
    RED = Fore.LIGHTRED_EX
    RESET = Style.RESET_ALL

    @staticmethod
    def format(text: str, color: 'Color') -> str:
        """Formats text with the specified color."""
        # Ensure text is a string
        text_str = str(text)
        return f"{color.value}{text_str}{Color.RESET.value}"

# Signal States Enum
class SignalState(Enum):
    """Enum representing various analysis signal states."""
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
    HIGH_VOLUME = "High Volume"
    LOW_VOLUME = "Low Volume"
    AVERAGE_VOLUME = "Average Volume"
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    FLAT = "Flat"
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"
    FLIP_BULLISH = "Flip Bullish"
    FLIP_BEARISH = "Flip Bearish"
    NONE = "None" # Explicitly no signal detected
    NA = "N/A"   # Data or calculation unavailable

# --- Configuration Loading ---
@dataclass
class IndicatorSettings:
    """Settings for technical indicators."""
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
    """Flags to enable/disable specific analysis checks."""
    ema_alignment: bool = True
    momentum_crossover: bool = False # Requires more complex logic, disabled by default
    volume_confirmation: bool = True
    rsi_divergence: bool = False # Basic check implemented, disabled by default (prone to false signals)
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
    adi_trend: bool = True # Uses ADOSC (A/D Oscillator)
    psar_flip: bool = True

@dataclass
class Thresholds:
    """Threshold values for oscillator indicators."""
    rsi_overbought: float = 70.0 # Use float for direct comparison with pandas_ta results
    rsi_oversold: float = 30.0
    mfi_overbought: float = 80.0
    mfi_oversold: float = 20.0
    cci_overbought: float = 100.0
    cci_oversold: float = -100.0
    williams_r_overbought: float = -20.0 # Note: Higher value is Overbought for Williams %R
    williams_r_oversold: float = -80.0  # Note: Lower value is Oversold for Williams %R
    adx_trending: float = 25.0
    # Volume analysis thresholds (relative to MA)
    volume_high_factor: float = 1.5
    volume_low_factor: float = 0.7

@dataclass
class OrderbookSettings:
    """Settings for order book analysis."""
    limit: int = 50 # Number of bids/asks levels to fetch
    cluster_threshold_usd: float = 10000.0 # Minimum USD value to consider a cluster significant (use float)
    cluster_proximity_pct: float = 0.1 # Proximity percentage around levels to check for clusters (e.g., 0.1 = +/- 0.1%)

@dataclass
class LoggingSettings:
    """Configuration for logging."""
    level: str = "INFO"
    rotation_max_bytes: int = 10 * 1024 * 1024 # 10 MB
    rotation_backup_count: int = 5

@dataclass
class AppConfig:
    """Main application configuration structure."""
    analysis_interval_seconds: int = 30
    kline_limit: int = 200 # Number of candles to fetch for analysis
    indicator_settings: IndicatorSettings = field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = field(default_factory=AnalysisFlags)
    thresholds: Thresholds = field(default_factory=Thresholds)
    orderbook_settings: OrderbookSettings = field(default_factory=OrderbookSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def _dataclass_to_dict(cls, dc_instance) -> dict:
        """Recursively converts a dataclass instance to a dictionary."""
        if not hasattr(dc_instance, "__dataclass_fields__"):
            return dc_instance
        result = {}
        for f_info in dc_instance.__dataclass_fields__.values():
            value = getattr(dc_instance, f_info.name)
            if hasattr(value, "__dataclass_fields__"):
                result[f_info.name] = cls._dataclass_to_dict(value)
            elif isinstance(value, list):
                 result[f_info.name] = [cls._dataclass_to_dict(i) for i in value]
            else:
                result[f_info.name] = value
        return result

    @classmethod
    def _merge_dicts(cls, default: dict, user: dict) -> dict:
        """Recursively merges user dict into default dict, only updating existing keys."""
        merged = default.copy()
        for key, user_value in user.items():
            if key in merged:
                default_value = merged[key]
                if isinstance(user_value, dict) and isinstance(default_value, dict):
                    merged[key] = cls._merge_dicts(default_value, user_value)
                # Allow merging if types match, or if user provides a number for a default float/int
                elif isinstance(user_value, type(default_value)) or \
                     (isinstance(default_value, (float, int)) and isinstance(user_value, (float, int))):
                    merged[key] = user_value
                elif default_value is None: # Allow setting value if default is None
                     merged[key] = user_value
                else:
                    # Type mismatch, log warning and keep default
                    print(Color.format(f"Warning: Config type mismatch for key '{key}'. Expected {type(default_value)}, got {type(user_value)}. Using default value.", Color.YELLOW))
            # else: Ignore keys from user config that are not in the default structure
        return merged

    @classmethod
    def _dict_to_dataclass(cls, data_class, data_dict):
        """Converts a dictionary to a nested dataclass structure, respecting defaults and types."""
        field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
        init_args = {}
        default_instance = data_class() # Get default values

        for name, type_hint in field_types.items():
            if name in data_dict:
                value = data_dict[name]
                # If the field type is a dataclass, recursively convert the sub-dict
                if hasattr(type_hint, '__dataclass_fields__') and isinstance(value, dict):
                    init_args[name] = cls._dict_to_dataclass(type_hint, value)
                else:
                    # Attempt type conversion based on the type hint
                    try:
                        # Get the expected type (handle Union, Optional)
                        origin_type = getattr(type_hint, '__origin__', None)
                        if origin_type is Union:
                            # Try converting to the first non-None type in Union
                            # This is a simplification, might need refinement for complex Unions
                            actual_type = next(t for t in type_hint.__args__ if t is not type(None))
                        elif origin_type is Optional:
                             actual_type = type_hint.__args__[0]
                        else:
                            actual_type = type_hint

                        # Convert if value is not already the correct type
                        if not isinstance(value, actual_type):
                            converted_value = actual_type(value)
                            init_args[name] = converted_value
                        else:
                            init_args[name] = value # Already correct type
                    except (ValueError, TypeError) as e:
                         default_val = getattr(default_instance, name, None)
                         print(Color.format(f"Warning: Could not convert config value '{value}' for key '{name}' to type {type_hint}. Using default: '{default_val}'. Error: {e}", Color.YELLOW))
                         # Use default value from dataclass if conversion fails
                         init_args[name] = default_val
            # else: Rely on default_factory or default value defined in dataclass
        try:
            return data_class(**init_args)
        except TypeError as e:
             print(Color.format(f"Error creating dataclass {data_class.__name__} from config: {e}", Color.RED))
             print(Color.format("Using default values for this section.", Color.YELLOW))
             return data_class() # Return default instance on error

    @classmethod
    def load(cls, filepath: Path) -> 'AppConfig':
        """Loads configuration from a JSON file, merging with defaults."""
        default_config_obj = cls()
        default_config_dict = cls._dataclass_to_dict(default_config_obj)

        if not filepath.exists():
            print(Color.format(f"Config file '{filepath}' not found.", Color.YELLOW))
            try:
                with filepath.open('w', encoding="utf-8") as f:
                    json.dump(default_config_dict, f, indent=2, ensure_ascii=False)
                print(Color.format(f"Created new config file '{filepath}' with default settings.", Color.GREEN))
                return default_config_obj # Return the default dataclass object
            except IOError as e:
                print(Color.format(f"Error creating default config file '{filepath}': {e}", Color.RED))
                print(Color.format("Loading internal defaults.", Color.YELLOW))
                return default_config_obj

        try:
            with filepath.open("r", encoding="utf-8") as f:
                user_config = json.load(f)

            if not isinstance(user_config, dict):
                raise TypeError("Config file does not contain a valid JSON object.")

            merged_config_dict = cls._merge_dicts(default_config_dict, user_config)
            # Convert the final merged dict back into the nested dataclass structure
            loaded_config = cls._dict_to_dataclass(cls, merged_config_dict)
            print(Color.format(f"Successfully loaded configuration from '{filepath}'.", Color.GREEN))
            return loaded_config

        except (FileNotFoundError, json.JSONDecodeError, TypeError, IOError) as e:
            print(Color.format(f"Error loading/parsing config file '{filepath}': {e}", Color.RED))
            print(Color.format("Loading internal defaults.", Color.YELLOW))
            return default_config_obj
        except Exception as e:
            print(Color.format(f"Unexpected error loading config: {e}", Color.RED))
            traceback.print_exc()
            print(Color.format("Loading internal defaults.", Color.YELLOW))
            return default_config_obj

CONFIG = AppConfig.load(CONFIG_FILE_PATH)

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that masks API key/secret and removes color codes for file logging."""
    _color_code_regex = re.compile(r'\x1b\[[0-9;]*m')

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, masking secrets and removing color."""
        # Use the base class's format method first
        formatted_message = super().format(record)
        # Mask secrets
        msg_no_secrets = formatted_message
        if API_KEY:
            msg_no_secrets = msg_no_secrets.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg_no_secrets = msg_no_secrets.replace(API_SECRET, "***API_SECRET***")
        # Remove color codes
        msg_no_color = self._color_code_regex.sub('', msg_no_secrets)
        return msg_no_color

class ColorStreamFormatter(logging.Formatter):
    """Formatter that adds colors for stream (console) output."""
    _level_color_map = {
        logging.DEBUG: Color.PURPLE.value,
        logging.INFO: Color.GREEN.value,
        logging.WARNING: Color.YELLOW.value,
        logging.ERROR: Color.RED.value,
        logging.CRITICAL: Color.RED.value + Style.BRIGHT,
    }
    _asctime_color = Color.BLUE.value
    _symbol_color = Color.YELLOW.value
    _reset_color = Color.RESET.value

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style='%', symbol: str = "GENERAL"):
        # Initialize the base class correctly. The base class handles fmt, datefmt, style.
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.symbol = symbol
        # No need to store style separately, base class handles it via self._style

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with appropriate colors."""
        # Get the base formatted message first
        # This uses the fmt, datefmt, and style provided during __init__
        # We need to temporarily set the format string for the base class to use
        original_fmt = self._fmt
        original_style = self._style

        level_color = self._level_color_map.get(record.levelno, self._reset_color)

        # Define the colored format string components
        asctime_part = f"{self._asctime_color}%(asctime)s{self._reset_color}"
        level_part = f"[{level_color}%(levelname)-8s{self._reset_color}]"
        symbol_part = f"{self._symbol_color}{self.symbol}{self._reset_color}"
        message_part = "%(message)s" # The actual message will be colored later if needed

        # Construct the format string based on the style
        # Note: This assumes the default '%' style for simplicity here.
        # If '{' or '$' style is used, the construction needs adjustment.
        # For robustness, it might be better to format the parts separately and combine.
        if self._style == '%':
            self._fmt = f"{asctime_part} {level_part} {symbol_part} - {message_part}"
        else:
            # Fallback or adapt for other styles if necessary
            # For simplicity, we'll stick to the % style assumption for color formatting structure
            # Or, format the message first, then prepend colored parts.
            pass # Keep original fmt if style is not '%' for now

        # Let the base class format the record using the modified format string
        formatted_message = super().format(record)

        # Restore original format string and style for subsequent calls
        self._fmt = original_fmt
        self._style = original_style

        # The message part itself might contain color codes from Color.format() used elsewhere.
        # The base formatter doesn't add color to the message itself.
        return formatted_message


def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger instance for a specific symbol with file and stream handlers."""
    log_filename = LOG_DIRECTORY / f"{symbol.replace('/', '_')}_{datetime.now(APP_TIMEZONE).strftime('%Y%m%d')}.log"
    logger = logging.getLogger(symbol)

    # Determine log level from config, default to INFO if invalid
    log_level_str = CONFIG.logging.level.upper()
    log_level = getattr(logging, log_level_str, None)
    if log_level is None:
        print(Color.format(f"Warning: Invalid log level '{CONFIG.logging.level}' in config. Defaulting to INFO.", Color.YELLOW))
        log_level = logging.INFO
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times if logger already exists and is configured
    if logger.hasHandlers():
        # Optionally clear existing handlers if reconfiguration is desired
        # logger.handlers.clear()
        # Or just return the existing logger if setup is idempotent
        return logger

    # File Handler (with rotation and sensitive data masking)
    try:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=CONFIG.logging.rotation_max_bytes,
            backupCount=CONFIG.logging.rotation_backup_count,
            encoding='utf-8'
        )
        # Use ISO 8601 format for file logs, ensure timezone info is included
        # Use standard % style formatting for file logs
        file_formatter = SensitiveFormatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S%z', # ISO 8601 format with timezone
            style='%'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file, logger level controls overall
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        print(Color.format(f"Error setting up file logger for {symbol}: {e}. File logging disabled.", Color.RED))
    except Exception as e:
        print(Color.format(f"Unexpected error setting up file logger for {symbol}: {e}", Color.RED))
        traceback.print_exc()

    # Stream Handler (with colors)
    stream_handler = logging.StreamHandler()
    # Use standard % style formatting for the ColorStreamFormatter as well
    stream_formatter = ColorStreamFormatter(
        # fmt is handled dynamically within the formatter's format method
        datefmt='%Y-%m-%d %H:%M:%S', # Use a more readable date format for console
        symbol=symbol,
        style='%' # Explicitly use '%' style
    )
    stream_handler.setFormatter(stream_formatter)
    # Stream handler level should respect the overall logger level set from config
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger if it has handlers (avoids duplicate console logs)
    logger.propagate = False

    return logger

# --- Utility Functions ---
async def async_sleep_with_jitter(seconds: float, max_jitter_factor: float = MAX_JITTER_FACTOR) -> None:
    """Asynchronous sleep with random jitter to avoid thundering herd."""
    if seconds <= 0:
        return
    # Ensure jitter is non-negative
    jitter = abs(np.random.uniform(0, seconds * max_jitter_factor))
    await asyncio.sleep(seconds + jitter)

def format_decimal(value: Optional[Union[Decimal, float, int, str]], precision: int = 2, default_na: str = "N/A") -> str:
    """
    Safely formats a numeric value (or string representation) as a Decimal string
    with specified precision. Handles None, NaN, Inf, and conversion errors gracefully.
    """
    if value is None or pd.isna(value):
        return default_na
    try:
        # Convert to string first to handle floats accurately and avoid precision issues
        decimal_value = Decimal(str(value))

        # Check for Inf/NaN after conversion (Decimal can represent them)
        if not decimal_value.is_finite():
            return default_na # Or return 'Inf'/'NaN' if preferred

        # Use quantize for proper Decimal rounding based on precision
        # Create the quantizer string like '1E-2' for precision 2
        quantizer = Decimal('1e-' + str(precision))
        # ROUND_HALF_UP is a common rounding method
        return str(decimal_value.quantize(quantizer, rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, TypeError):
        # Fallback for values that cannot be converted to Decimal initially
        try:
            # Attempt to format as float if Decimal fails
            float_value = float(value)
            if not np.isfinite(float_value):
                return default_na
            return f"{float_value:.{precision}f}"
        except (ValueError, TypeError):
            # Last resort: simple string conversion if float formatting also fails
            # Avoid returning potentially misleading representations
            return default_na if isinstance(value, (float, np.floating)) else str(value)


# --- CCXT Client ---
class BybitCCXTClient:
    """
    Asynchronous CCXT client specifically for Bybit V5 API with robust error
    handling, retry logic, and market loading/management.
    """
    def __init__(self, api_key: str, api_secret: str, is_testnet: bool, logger_instance: logging.Logger):
        """
        Initializes the Bybit CCXT client.

        Args:
            api_key: The Bybit API key.
            api_secret: The Bybit API secret.
            is_testnet: Boolean indicating whether to use the testnet environment.
            logger_instance: The logger instance to use for logging.
        """
        self.logger = logger_instance
        self.is_testnet = is_testnet
        self._exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True, # Enable built-in rate limiter
            'options': {
                'defaultType': 'linear', # Default to linear contracts (USDT margined)
                'adjustForTimeDifference': True, # Auto-sync time with server
                'brokerId': 'NEONTA_V4.1', # Optional: Set a broker ID
                'recvWindow': 10000, # Increase recvWindow slightly (default 5000ms)
                # Explicitly set testnet mode using standard ccxt option
                'testnet': self.is_testnet,
                # V5 API specific options if needed (though defaultType often handles it)
                # 'defaultNetwork': 'spot' / 'linear' / 'inverse' / 'option'
                # 'versions': {'public': 'v5', 'private': 'v5'} # CCXT usually handles this
            },
            'timeout': CCXT_TIMEOUT_MS,
        }

        try:
            # Explicitly select bybit class
            self.exchange = ccxt_async.bybit(self._exchange_config)
            self.logger.info(f"CCXT client initialized for Bybit {'Testnet' if self.is_testnet else 'Mainnet'}.")
            # Log the actual API endpoint being used by CCXT
            api_url = self.exchange.urls.get('api', 'URL not available')
            if isinstance(api_url, dict): # URLs can be nested dicts
                api_url = api_url.get('public', api_url.get('private', 'Nested URL not found'))
            self.logger.debug(f"Using Base API URL: {api_url}")
        except ccxt.AuthenticationError as e:
             self.logger.critical(Color.format(f"CCXT Authentication Error during initialization: {e}. Check API credentials.", Color.RED))
             raise # Fatal error
        except Exception as e:
            self.logger.exception(f"Failed to initialize CCXT Bybit instance: {e}")
            raise  # Re-raise the exception to prevent starting with a broken client

        self.markets: Optional[Dict[str, Any]] = None
        self.market_categories: Dict[str, str] = {} # Cache for market categories (linear, inverse, spot)

    async def initialize_markets(self, retries: int = MAX_API_RETRIES) -> bool:
        """
        Loads market data from the exchange with retry logic.

        Args:
            retries: Maximum number of attempts to load markets.

        Returns:
            True if markets were loaded successfully, False otherwise.
        """
        self.logger.info("Attempting to load markets...")
        current_delay = INITIAL_RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                # Use reload=True if markets might already be partially loaded or stale
                self.markets = await self.exchange.load_markets(reload=True)
                if not self.markets:
                    # Raise an error if load_markets returns None or empty dict
                    raise ccxt.ExchangeError("load_markets returned None or an empty dictionary.")

                self.logger.info(f"Successfully loaded {len(self.markets)} markets from {self.exchange.name}.")
                self._cache_market_categories() # Populate category cache
                return True
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, RequestException) as e:
                self.logger.warning(f"Network/Timeout error loading markets (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s...")
            except ccxt.AuthenticationError as e:
                self.logger.error(Color.format(f"Authentication Error loading markets: {e}. Please check API credentials.", Color.RED))
                return False # Authentication errors are fatal, no point retrying
            except ccxt.ExchangeError as e:
                 # Catch other specific ccxt errors if needed
                 self.logger.error(f"CCXT ExchangeError loading markets (Attempt {attempt + 1}/{retries}): {e}")
            except Exception as e:
                # Catch any other unexpected errors during market loading
                # This includes the AttributeError if _cache_market_categories fails internally
                self.logger.exception(f"Unexpected error loading markets (Attempt {attempt + 1}/{retries}): {e}")

            if attempt < retries - 1:
                await async_sleep_with_jitter(current_delay)
                current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY_SECONDS) # Exponential backoff with cap
            else:
                self.logger.error(Color.format(f"Failed to load markets after {retries} attempts.", Color.RED))

        return False

    def _cache_market_categories(self) -> None:
        """Pre-calculates and caches market categories (linear, inverse, spot) after loading markets."""
        if not self.markets:
            self.logger.warning("Cannot cache market categories: Markets not loaded.")
            return

        self.market_categories = {}
        count = 0
        skipped_none = 0
        for symbol, details in self.markets.items():
            # FIX: Check if details dictionary is None before proceeding
            if details is None:
                self.logger.warning(f"Market details for symbol '{symbol}' are None. Skipping category caching for this symbol.")
                skipped_none += 1
                continue

            # Determine category based on CCXT market properties (type, contractType, settle)
            category = 'spot' # Default assumption
            market_type = details.get('type', 'spot').lower() # spot, linear, inverse, future, option
            contract_type = details.get('contractType', '').lower() # linear, inverse

            # FIX: Safely handle potential None value from .get('settle') before calling .upper()
            settle_value = details.get('settle')
            settle_currency = str(settle_value).upper() if settle_value is not None else '' # USDT, USD, BTC etc.

            if market_type == 'spot':
                category = 'spot'
            elif market_type in ['future', 'swap']:
                # Prioritize linear/inverse based on settle currency (USDT/USD) as it's often more reliable in V5
                if settle_currency == 'USDT':
                    category = 'linear'
                elif settle_currency == 'USD': # Bybit V5 uses USD for inverse settle
                    category = 'inverse'
                # Fallback to contractType if settle is ambiguous or missing
                elif contract_type == 'linear':
                    category = 'linear'
                elif contract_type == 'inverse':
                    category = 'inverse'
                else:
                    # Fallback guess if contractType and settle are missing/unexpected
                    quote = details.get('quote', '').upper()
                    if quote == 'USDT': category = 'linear'
                    elif quote == 'USD': category = 'inverse'
                    else: category = 'spot' # Less likely for derivatives, but a fallback
                    self.logger.debug(f"Guessed category '{category}' for derivative {symbol} based on quote '{quote}'. Market type: '{market_type}', Contract type: '{contract_type}', Settle: '{settle_currency}'.")
            elif market_type == 'option':
                 category = 'option' # Handle options if needed
                 self.logger.debug(f"Market {symbol} identified as Option. Treating as 'option' category.")
            else:
                # Fallback for unknown market types
                quote = details.get('quote', '').upper()
                if quote == 'USDT': category = 'linear'
                elif quote == 'USD': category = 'inverse'
                else: category = 'spot'
                self.logger.warning(f"Unknown market type '{market_type}' for {symbol}. Guessed category '{category}' based on quote '{quote}'.")


            self.market_categories[symbol] = category
            count += 1

        log_message = f"Cached categories for {count} markets (Spot/Linear/Inverse/Option)."
        if skipped_none > 0:
            log_message += f" Skipped {skipped_none} markets with None details."
        self.logger.info(log_message)


    def is_valid_symbol(self, symbol: str) -> bool:
        """Checks if the symbol exists in the loaded markets."""
        if self.markets is None:
            self.logger.warning("Markets not loaded, cannot validate symbol.")
            # Return False to be strict, prevents operations on potentially invalid symbols
            return False
        is_valid = symbol in self.markets and self.markets[symbol] is not None
        if not is_valid:
             self.logger.debug(f"Symbol '{symbol}' not found or has None details in loaded markets.")
        return is_valid

    def get_symbol_details(self, symbol: str) -> Optional[dict]:
        """
        Gets market details for a specific symbol from the loaded markets.

        Args:
            symbol: The market symbol (e.g., 'BTC/USDT').

        Returns:
            A dictionary containing market details, or None if the symbol is invalid,
            markets are not loaded, or details are None.
        """
        if not self.is_valid_symbol(symbol):
            # is_valid_symbol already logs if markets aren't loaded or symbol not found/None
            return None
        # self.markets is confirmed not None and details for symbol exist by is_valid_symbol
        return cast(Dict[str, Any], self.markets).get(symbol)

    def get_market_category(self, symbol: str) -> str:
        """
        Gets the market category ('linear', 'inverse', 'spot', 'option') for a symbol,
        using the cache if available.

        Args:
            symbol: The market symbol.

        Returns:
            The category string. Defaults to 'spot' if unable to determine.
        """
        if symbol in self.market_categories:
            return self.market_categories[symbol]

        # Fallback if called before cache is populated or for an unknown symbol
        self.logger.warning(f"Category for symbol {symbol} not found in cache. Attempting dynamic check.")
        details = self.get_symbol_details(symbol) # Checks validity again
        if details:
            # Re-run the logic from _cache_market_categories for this single symbol
            market_type = details.get('type', 'spot').lower()
            contract_type = details.get('contractType', '').lower()
            settle_value = details.get('settle')
            settle_currency = str(settle_value).upper() if settle_value is not None else ''

            if market_type == 'spot': return 'spot'
            if market_type in ['future', 'swap']:
                if settle_currency == 'USDT': return 'linear'
                if settle_currency == 'USD': return 'inverse'
                if contract_type == 'linear': return 'linear'
                if contract_type == 'inverse': return 'inverse'
            if market_type == 'option': return 'option'

            # Fallback guess based on quote if type/contract info was insufficient
            quote = details.get('quote', '').upper()
            if quote == 'USDT': return 'linear'
            if quote == 'USD': return 'inverse'
            if quote: return 'spot' # If quote exists but isn't USDT/USD, assume spot-like

        # Final fallback guess if details are missing or uninformative
        self.logger.warning(f"Could not reliably determine category for {symbol}, defaulting to 'spot'.")
        return 'spot'

    async def close(self) -> None:
        """Closes the underlying ccxt exchange connection gracefully."""
        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Closed CCXT exchange connection.")
            except Exception as e:
                self.logger.error(f"Error closing CCXT connection: {e}")

    async def fetch_with_retry(self, method_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        """
        Generic fetch method with retry logic for common transient API errors.

        Args:
            method_name: The name of the ccxt exchange method to call (e.g., 'fetch_ticker').
            *args: Positional arguments for the ccxt method.
            **kwargs: Keyword arguments for the ccxt method.

        Returns:
            The result from the ccxt method call, or None if it fails after retries.
        """
        retries = MAX_API_RETRIES
        current_delay = INITIAL_RETRY_DELAY_SECONDS
        last_exception: Optional[Exception] = None

        method = getattr(self.exchange, method_name, None)
        if not method or not callable(method):
            self.logger.error(f"Invalid CCXT method name: '{method_name}'")
            return None

        for attempt in range(retries):
            try:
                result = await method(*args, **kwargs)
                # Optional: Add basic validation here if needed (e.g., check if result is None)
                # if result is None:
                #     self.logger.warning(f"CCXT method {method_name} returned None.")
                return result
            # Specific, common transient errors first
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, RequestException) as e:
                log_msg = f"Network/Timeout/DDoS error calling {method_name} (Attempt {attempt + 1}/{retries}): {type(e).__name__}. Retrying in {current_delay:.1f}s..."
                self.logger.warning(Color.format(log_msg, Color.YELLOW))
                last_exception = e
            except ccxt.RateLimitExceeded as e:
                # Extract retry-after header if available (CCXT might parse it)
                retry_after_ms = getattr(e, 'retry_after', None) # CCXT sometimes adds this
                wait_time = current_delay # Default wait time
                if retry_after_ms:
                    # Use max of suggested or current backoff, ensure it's within bounds
                    suggested_wait = max(float(retry_after_ms) / 1000.0, 0.1) # Min 0.1s
                    wait_time = max(suggested_wait, current_delay)
                    log_msg = f"Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying after suggested {wait_time:.1f}s..."
                else:
                    log_msg = f"Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                self.logger.warning(Color.format(log_msg, Color.YELLOW))
                current_delay = wait_time # Update delay based on suggestion or backoff
                last_exception = e
            # Broader ExchangeError, check for potentially retryable Bybit messages
            except ccxt.ExchangeError as e:
                error_str = str(e).lower()
                # Bybit specific error codes/messages that might indicate temporary issues
                # Ref: https://bybit-exchange.github.io/docs/v5/error_code
                # Expanded list based on common transient issues
                retryable_bybit_errors = [
                    'too many visits', 'system busy', 'service unavailable', 'ip rate limit',
                    'server error', 'internal server error', 'gateway timeout',
                    '10001', # Parameter error (sometimes transient if related to timing/sync) - retry cautiously
                    '10002', # Request expired (recvWindow issue, maybe transient)
                    '10004', # Sign check error (can be transient if clock skew adjusts)
                    '10006', # recv_window error (definitely retry)
                    '10010', # request timeout
                    '10016', # service unavailable
                    '10017', # request failed process (generic, worth a retry)
                    '10018', # system busy
                    '30034', # Order quantity error (might be transient if market conditions change fast) - retry cautiously
                    '33004', # Risk limit error (might be transient) - retry cautiously
                    '130021', # Order quantity error (linear)
                    '130101', # Internal error
                    '130144', # System busy
                    '130150', # Service unavailable
                    '131001', # Internal error (spot)
                    '131200', # Service unavailable (spot)
                    '131204', # Request timeout (spot)
                ]
                # Check if the error string contains any of the retryable codes/messages
                if any(err_code in error_str for err_code in retryable_bybit_errors):
                    log_msg = f"Retryable server/rate limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                    self.logger.warning(Color.format(log_msg, Color.YELLOW))
                    last_exception = e
                else:
                    # Non-retryable ExchangeError (e.g., invalid symbol, insufficient funds)
                    self.logger.error(Color.format(f"Non-retryable CCXT ExchangeError calling {method_name}: {e}", Color.RED))
                    self.logger.debug(f"Args: {args}, Kwargs: {kwargs}")
                    return None # Do not retry these errors
            # Catch AuthenticationError separately as it's fatal
            except ccxt.AuthenticationError as e:
                 self.logger.error(Color.format(f"Authentication Error calling {method_name}: {e}. Check API credentials.", Color.RED))
                 return None # Fatal error
            # Catch all other unexpected exceptions
            except Exception as e:
                self.logger.exception(Color.format(f"Unexpected error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}", Color.RED))
                self.logger.debug(f"Args: {args}, Kwargs: {kwargs}")
                last_exception = e
                # Decide whether to retry unexpected errors or not. Retrying is safer for potentially transient issues.

            # If an exception occurred and we haven't returned yet, wait and retry
            if attempt < retries - 1:
                await async_sleep_with_jitter(current_delay)
                # Apply exponential backoff, capped at max delay
                current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY_SECONDS)
            else:
                self.logger.error(Color.format(f"Max retries ({retries}) reached for {method_name}. Last error: {last_exception}", Color.RED))
                return None # Max retries exceeded

        return None # Should technically not be reached, but ensures a return path

    async def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetches ticker information for a symbol using the retry mechanism."""
        self.logger.debug(f"Fetching ticker for {symbol}")
        if not self.is_valid_symbol(symbol):
            self.logger.error(f"Cannot fetch ticker: Invalid symbol '{symbol}'.")
            return None

        category = self.get_market_category(symbol)
        params = {'category': category}
        # Use fetch_tickers, often more reliable even for single symbols
        # Note: fetch_tickers requires a list/tuple of symbols
        tickers = await self.fetch_with_retry('fetch_tickers', symbols=[symbol], params=params)

        if tickers and isinstance(tickers, dict) and symbol in tickers:
            # Ensure the ticker data itself is a dictionary
            ticker_data = tickers[symbol]
            if isinstance(ticker_data, dict):
                return ticker_data
            else:
                self.logger.error(f"Unexpected ticker data format for {symbol} in fetch_tickers response: {ticker_data}")
                return None
        else:
            self.logger.error(f"Could not fetch ticker for {symbol} (category: {category}). Response: {tickers}")
            return None

    async def fetch_current_price(self, symbol: str) -> Optional[Decimal]:
        """Fetches the last traded price for a symbol and returns it as a Decimal."""
        ticker = await self.fetch_ticker(symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            try:
                # Convert to string first for accurate Decimal conversion
                price_str = str(ticker['last'])
                return Decimal(price_str)
            except (InvalidOperation, TypeError, ValueError) as e:
                self.logger.error(f"Error converting last price '{ticker['last']}' to Decimal for {symbol}: {e}")
                return None
        else:
            self.logger.warning(f"Last price not found or is null in ticker data for {symbol}.")
            return None

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetches OHLCV (kline) data for a symbol and timeframe, returning a processed DataFrame.

        Args:
            symbol: The market symbol.
            timeframe: The user-friendly timeframe string (e.g., "15", "1h").
            limit: The maximum number of klines to fetch.

        Returns:
            A pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            where 'timestamp' is timezone-aware (APP_TIMEZONE) and OHLCV are Decimals.
            Returns an empty DataFrame on failure or if no data is returned.
        """
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            self.logger.error(f"Invalid timeframe '{timeframe}' provided. Valid options: {VALID_INTERVALS}")
            return pd.DataFrame()
        if not self.is_valid_symbol(symbol):
             self.logger.error(f"Cannot fetch klines: Invalid symbol '{symbol}'.")
             return pd.DataFrame()

        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching {limit} klines for {symbol} | Interval: {ccxt_timeframe} | Category: {category}")
        params = {'category': category}
        # fetch_ohlcv(symbol, timeframe, since, limit, params)
        klines = await self.fetch_with_retry('fetch_ohlcv', symbol, timeframe=ccxt_timeframe, limit=limit, params=params)

        if klines is None or not isinstance(klines, list) or len(klines) == 0:
            self.logger.warning(f"No kline data returned for {symbol} interval {ccxt_timeframe}.")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                self.logger.warning(f"Kline data for {symbol} resulted in an empty DataFrame initially.")
                return pd.DataFrame()

            # Convert timestamp to datetime and set timezone
            # Errors='coerce' will turn unparseable timestamps into NaT
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            # Drop rows where timestamp conversion failed
            initial_rows_ts = len(df)
            df.dropna(subset=['timestamp'], inplace=True)
            dropped_ts_rows = initial_rows_ts - len(df)
            if dropped_ts_rows > 0:
                 self.logger.warning(f"Dropped {dropped_ts_rows} rows with invalid timestamps from klines for {symbol}.")

            if df.empty:
                 self.logger.warning(f"Kline data for {symbol} had no valid timestamps after conversion.")
                 return pd.DataFrame()

            # Convert timezone after ensuring timestamps are valid
            df['timestamp'] = df['timestamp'].dt.tz_convert(APP_TIMEZONE)

            # Convert OHLCV columns to Decimal for precision, coercing errors
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in essential_cols:
                # Convert to string first, then Decimal. Use pd.NA for unconvertible values.
                # Apply this conversion carefully
                converted_col = []
                for x in df[col]:
                    if pd.isna(x):
                        converted_col.append(pd.NA)
                    else:
                        try:
                            # Convert via string to preserve precision from source
                            converted_col.append(Decimal(str(x)))
                        except (InvalidOperation, TypeError, ValueError):
                            converted_col.append(pd.NA) # Mark unconvertible as NA
                # Use object dtype for Decimal, allows pd.NA
                df[col] = pd.Series(converted_col, index=df.index, dtype=object)

            initial_rows_ohlcv = len(df)
            # Drop rows with NA/NaN in essential OHLCV columns after conversion attempt
            df.dropna(subset=essential_cols, inplace=True)
            dropped_ohlcv_rows = initial_rows_ohlcv - len(df)
            if dropped_ohlcv_rows > 0:
                self.logger.warning(f"Dropped {dropped_ohlcv_rows} rows with missing/invalid essential OHLCV data from klines for {symbol}.")

            if df.empty:
                self.logger.warning(f"Kline data for {symbol} is empty after cleaning missing values.")
                return pd.DataFrame()

            # Ensure data types are correct (Decimal for OHLCV, datetime for timestamp)
            # This check might be computationally expensive for large dataframes
            # Consider enabling only for debugging if performance is an issue
            # for col in essential_cols:
            #      if not all(isinstance(x, Decimal) for x in df[col]):
            #           self.logger.warning(f"Column '{col}' contains non-Decimal values after processing.")
            #           # Attempt one more conversion or handle appropriately
            #           # Re-applying Decimal conversion might be redundant or hide underlying issues
            #           # df[col] = pd.to_numeric(df[col], errors='coerce') # Fallback to numeric if Decimal failed broadly

            # Sort by timestamp just in case API returns them out of order
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            self.logger.debug(f"Successfully fetched and processed {len(df)} klines for {symbol}.")
            return df

        except (ValueError, TypeError, KeyError, InvalidOperation) as e:
            self.logger.exception(f"Error processing kline data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
             self.logger.exception(f"Unexpected error processing kline data for {symbol}: {e}")
             return pd.DataFrame()


    async def fetch_orderbook(self, symbol: str, limit: int) -> Optional[dict]:
        """
        Fetches the order book for a symbol using the retry mechanism.

        Args:
            symbol: The market symbol.
            limit: The maximum number of price levels to fetch for bids and asks. Bybit V5 limits: [1, 200] for linear/inverse, [1, 50] for spot.

        Returns:
            A dictionary representing the order book (with 'bids', 'asks' keys),
            or None if fetching fails or the response is invalid.
        """
        if not self.is_valid_symbol(symbol):
             self.logger.error(f"Cannot fetch orderbook: Invalid symbol '{symbol}'.")
             return None

        category = self.get_market_category(symbol)

        # Adjust limit based on Bybit V5 category limits if necessary
        # Note: CCXT might handle this adjustment internally, but being explicit can prevent errors.
        bybit_limit = limit
        if category == 'spot' and limit > 50:
            self.logger.warning(f"Requested orderbook limit {limit} exceeds Bybit V5 spot limit (50). Adjusting to 50.")
            bybit_limit = 50
        elif category in ['linear', 'inverse'] and limit > 200:
             self.logger.warning(f"Requested orderbook limit {limit} exceeds Bybit V5 linear/inverse limit (200). Adjusting to 200.")
             bybit_limit = 200
        elif limit <= 0:
             self.logger.error(f"Invalid orderbook limit requested ({limit}). Must be positive.")
             return None


        self.logger.debug(f"Fetching order book for {symbol} | Limit: {bybit_limit} | Category: {category}")
        params = {'category': category}
        # fetch_order_book(symbol, limit, params)
        orderbook = await self.fetch_with_retry('fetch_order_book', symbol, limit=bybit_limit, params=params)

        if orderbook and isinstance(orderbook, dict):
            # Basic validation: Check for presence and list type of bids/asks
            bids = orderbook.get('bids')
            asks = orderbook.get('asks')
            if isinstance(bids, list) and isinstance(asks, list):
                # Optional: Deeper validation - check if bids/asks contain [price, size] pairs
                # And check if the number of levels matches the requested limit (or is close)
                valid_levels = lambda levels: all(isinstance(level, list) and len(level) == 2 for level in levels)
                if valid_levels(bids) and valid_levels(asks):
                    self.logger.debug(f"Fetched order book for {symbol} with {len(bids)} bids and {len(asks)} asks.")
                    return orderbook
                else:
                     self.logger.warning(f"Fetched order book data for {symbol} has invalid level format: {orderbook}")
                     return None
            else:
                self.logger.warning(f"Fetched order book data for {symbol} is missing bids/asks or has an unexpected format: {orderbook}")
                return None
        else:
            self.logger.warning(f"Failed to fetch order book for {symbol} or received invalid data after retries.")
            return None


# --- Trading Analyzer ---
class TradingAnalyzer:
    """
    Performs technical analysis using kline data and interprets the results.
    Optionally incorporates order book analysis.
    """
    def __init__(self, config: AppConfig, logger_instance: logging.Logger, symbol: str):
        """
        Initializes the TradingAnalyzer.

        Args:
            config: The application configuration object.
            logger_instance: The logger instance to use.
            symbol: The trading symbol being analyzed.
        """
        self.config = config
        self.logger = logger_instance
        self.symbol = symbol
        # Convenience accessors for config sections
        self.indicator_settings = config.indicator_settings
        self.analysis_flags = config.analysis_flags
        self.thresholds = config.thresholds
        self.orderbook_settings = config.orderbook_settings

        # Dynamically generate expected indicator column names based on config
        self._generate_column_names()

    def _generate_column_names(self) -> None:
        """Generates expected pandas_ta indicator column names based on config settings."""
        self.col_names: Dict[str, str] = {}
        is_ = self.indicator_settings # Alias for brevity

        # Helper to format Bollinger Bands std dev for column name consistency
        def fmt_bb_std(std: Union[float, int, Decimal]) -> str:
            try:
                # Format to one decimal place, consistent with pandas_ta default naming
                # Convert to float first as pandas_ta uses float internally
                return f"{float(std):.1f}"
            except (ValueError, TypeError):
                return str(std) # Fallback

        # Standard Indicators
        if is_.sma_short_period > 0: self.col_names['sma_short'] = f"SMA_{is_.sma_short_period}"
        if is_.sma_long_period > 0: self.col_names['sma_long'] = f"SMA_{is_.sma_long_period}"
        if is_.ema_short_period > 0: self.col_names['ema_short'] = f"EMA_{is_.ema_short_period}"
        if is_.ema_long_period > 0: self.col_names['ema_long'] = f"EMA_{is_.ema_long_period}"
        if is_.rsi_period > 0: self.col_names['rsi'] = f"RSI_{is_.rsi_period}"
        if all(p > 0 for p in [is_.stoch_rsi_period, is_.rsi_period, is_.stoch_k_period, is_.stoch_d_period]):
            self.col_names['stochrsi_k'] = f"STOCHRSIk_{is_.stoch_rsi_period}_{is_.rsi_period}_{is_.stoch_k_period}_{is_.stoch_d_period}"
            self.col_names['stochrsi_d'] = f"STOCHRSId_{is_.stoch_rsi_period}_{is_.rsi_period}_{is_.stoch_k_period}_{is_.stoch_d_period}"
        if all(p > 0 for p in [is_.macd_fast, is_.macd_slow, is_.macd_signal]):
            self.col_names['macd_line'] = f"MACD_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
            self.col_names['macd_signal'] = f"MACDs_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
            self.col_names['macd_hist'] = f"MACDh_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        if is_.bollinger_bands_period > 0 and is_.bollinger_bands_std_dev > 0:
            bb_std_str = fmt_bb_std(is_.bollinger_bands_std_dev)
            self.col_names['bb_upper'] = f"BBU_{is_.bollinger_bands_period}_{bb_std_str}"
            self.col_names['bb_lower'] = f"BBL_{is_.bollinger_bands_period}_{bb_std_str}"
            self.col_names['bb_mid'] = f"BBM_{is_.bollinger_bands_period}_{bb_std_str}"
        if is_.atr_period > 0: self.col_names['atr'] = f"ATRr_{is_.atr_period}" # pandas_ta uses ATRr for the 'True Range average' variant
        if is_.cci_period > 0: self.col_names['cci'] = f"CCI_{is_.cci_period}_0.015" # pandas_ta CCI includes the constant
        if is_.williams_r_period > 0: self.col_names['willr'] = f"WILLR_{is_.williams_r_period}"
        if is_.mfi_period > 0: self.col_names['mfi'] = f"MFI_{is_.mfi_period}"
        if is_.adx_period > 0:
            self.col_names['adx'] = f"ADX_{is_.adx_period}"
            self.col_names['dmp'] = f"DMP_{is_.adx_period}" # +DI component of ADX
            self.col_names['dmn'] = f"DMN_{is_.adx_period}" # -DI component of ADX
        self.col_names['obv'] = "OBV" # OBV has no standard parameters in name
        self.col_names['adosc'] = "ADOSC" # Accumulation/Distribution Oscillator (no params in name)
        if is_.psar_step > 0 and is_.psar_max_step > 0:
            # Format step/max_step consistently (e.g., handle 0.02 vs 0.2)
            psar_step_str = f"{is_.psar_step:.2f}".rstrip('0').rstrip('.')
            psar_max_str = f"{is_.psar_max_step:.2f}".rstrip('0').rstrip('.')
            self.col_names['psar_long'] = f"PSARl_{psar_step_str}_{psar_max_str}"
            self.col_names['psar_short'] = f"PSARs_{psar_step_str}_{psar_max_str}"
            self.col_names['psar_af'] = f"PSARaf_{psar_step_str}_{psar_max_str}" # Acceleration Factor
            self.col_names['psar_rev'] = f"PSARr_{psar_step_str}_{psar_max_str}" # Reversal signal (1 for reversal)
        if is_.momentum_period > 0: self.col_names['mom'] = f"MOM_{is_.momentum_period}"
        if is_.volume_ma_period > 0: self.col_names['vol_ma'] = f"VOL_MA_{is_.volume_ma_period}" # Custom name

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators using pandas_ta strategy.

        Args:
            df: Input DataFrame with OHLCV data (expects Decimal types).

        Returns:
            DataFrame with calculated indicator columns appended (as floats),
            or the original DataFrame if calculation fails or input is unsuitable.
        """
        if df.empty:
            self.logger.warning("Cannot calculate indicators: Input DataFrame is empty.")
            return df
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
             self.logger.error(f"Cannot calculate indicators: Input DataFrame missing required columns ({required_cols}). Found: {list(df.columns)}")
             return df

        # Create a copy to avoid modifying the original DataFrame passed in
        df_calc = df.copy()

        # Ensure OHLCV columns are numeric (float64) for pandas_ta compatibility
        # pandas_ta works best with floats. Convert Decimal to float for calculation.
        for col in required_cols:
            # Coerce to numeric (float64), turning errors (like non-numeric strings or objects) into NaN
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

        # Drop rows if essential numeric columns became NaN after conversion
        initial_rows = len(df_calc)
        df_calc.dropna(subset=required_cols, inplace=True)
        dropped_rows = initial_rows - len(df_calc)
        if dropped_rows > 0:
             self.logger.warning(f"Dropped {dropped_rows} rows due to non-numeric OHLCV data during indicator calculation prep.")

        if df_calc.empty:
             self.logger.error("DataFrame became empty after converting OHLCV to numeric. Cannot calculate indicators.")
             return df # Return original unmodified df

        # Check if enough data points exist for the longest required period
        is_ = self.indicator_settings
        # Estimate max period needed (consider MACD slow+signal, ADX needs more data)
        # Ensure all configured periods are positive before taking max
        valid_periods = [p for p in [
            is_.sma_long_period, is_.ema_long_period,
            is_.macd_slow + is_.macd_signal, # MACD needs signal periods beyond slow EMA
            is_.bollinger_bands_period,
            is_.adx_period * 2, # ADX often needs 2x period for smoothing
            is_.stoch_rsi_period + is_.rsi_period, # StochRSI needs underlying RSI data
            is_.volume_ma_period, is_.atr_period, is_.cci_period, is_.williams_r_period, is_.mfi_period,
            is_.momentum_period
        ] if isinstance(p, (int, float)) and p > 0]

        min_data_needed = max(valid_periods) if valid_periods else 1 # Need at least 1 row

        if len(df_calc) < min_data_needed:
            self.logger.warning(f"Insufficient data points ({len(df_calc)} < {min_data_needed} estimated needed) for some indicators. Results may be inaccurate or contain NaNs.")
        elif len(df_calc) < 2:
             self.logger.warning("Only one data point available. Most indicators cannot be calculated accurately.")
             # Allow calculation, but expect many NaNs

        # Define the strategy using pandas_ta structure
        # Ensure all periods are positive integers/floats and handle potential None values
        strategy_ta = [
            {"kind": "sma", "length": is_.sma_short_period} if is_.sma_short_period > 0 else None,
            {"kind": "sma", "length": is_.sma_long_period} if is_.sma_long_period > 0 else None,
            {"kind": "ema", "length": is_.ema_short_period} if is_.ema_short_period > 0 else None,
            {"kind": "ema", "length": is_.ema_long_period} if is_.ema_long_period > 0 else None,
            {"kind": "rsi", "length": is_.rsi_period} if is_.rsi_period > 0 else None,
            {"kind": "stochrsi", "length": is_.stoch_rsi_period, "rsi_length": is_.rsi_period, "k": is_.stoch_k_period, "d": is_.stoch_d_period} if all(p > 0 for p in [is_.stoch_rsi_period, is_.rsi_period, is_.stoch_k_period, is_.stoch_d_period]) else None,
            {"kind": "macd", "fast": is_.macd_fast, "slow": is_.macd_slow, "signal": is_.macd_signal} if all(p > 0 for p in [is_.macd_fast, is_.macd_slow, is_.macd_signal]) else None,
            {"kind": "bbands", "length": is_.bollinger_bands_period, "std": float(is_.bollinger_bands_std_dev)} if is_.bollinger_bands_period > 0 and is_.bollinger_bands_std_dev > 0 else None,
            {"kind": "atr", "length": is_.atr_period} if is_.atr_period > 0 else None,
            {"kind": "cci", "length": is_.cci_period} if is_.cci_period > 0 else None,
            {"kind": "willr", "length": is_.williams_r_period} if is_.williams_r_period > 0 else None,
            {"kind": "mfi", "length": is_.mfi_period} if is_.mfi_period > 0 else None,
            {"kind": "adx", "length": is_.adx_period} if is_.adx_period > 0 else None,
            {"kind": "obv"}, # OBV doesn't have a length parameter in the same way
            {"kind": "adosc"}, # AD Oscillator
            {"kind": "psar", "step": is_.psar_step, "max_step": is_.psar_max_step} if is_.psar_step > 0 and is_.psar_max_step > 0 else None,
            {"kind": "mom", "length": is_.momentum_period} if is_.momentum_period > 0 else None,
            # Calculate volume MA using 'sma' kind on 'volume' column, apply custom prefix
            {"kind": "sma", "close": "volume", "length": is_.volume_ma_period, "prefix": "VOL_MA"} if is_.volume_ma_period > 0 else None,
        ]

        # Filter out None entries (from invalid periods or disabled indicators)
        valid_strategy_ta = [item for item in strategy_ta if item is not None]

        if not valid_strategy_ta:
             self.logger.error("No valid indicators configured or all periods are invalid. Skipping calculation.")
             return df # Return original df

        # Create the pandas_ta strategy object
        strategy = ta.Strategy(
            name="NeontaAnalysis",
            description="Comprehensive TA using pandas_ta",
            ta=valid_strategy_ta
        )

        try:
            # Apply the strategy to the DataFrame (modifies df_calc in place)
            # Use append=True to add columns to df_calc instead of returning a new df
            df_calc.ta.strategy(strategy, timed=False, append=True)

            # Rename the volume MA column generated by pandas_ta to our expected name
            # Default name generated by prefix is like "VOL_MA_SMA_20"
            vol_ma_generated_name = f"VOL_MA_SMA_{is_.volume_ma_period}"
            vol_ma_target_name = self.col_names.get('vol_ma')

            if vol_ma_target_name and vol_ma_generated_name in df_calc.columns:
                if vol_ma_target_name != vol_ma_generated_name:
                    df_calc.rename(columns={vol_ma_generated_name: vol_ma_target_name}, inplace=True)
                    self.logger.debug(f"Renamed volume MA column from '{vol_ma_generated_name}' to '{vol_ma_target_name}'.")
            elif vol_ma_target_name and vol_ma_generated_name not in df_calc.columns and is_.volume_ma_period > 0:
                 self.logger.warning(f"Expected volume MA column '{vol_ma_generated_name}' or '{vol_ma_target_name}' not found after calculation.")


            self.logger.debug(f"Calculated pandas_ta indicators for {self.symbol}.")

            # Optional: Fill potentially generated PSAR long/short columns if one is all NaN
            # This handles cases where PSAR might only generate one column initially
            psar_l_col = self.col_names.get('psar_long')
            psar_s_col = self.col_names.get('psar_short')
            if psar_l_col and psar_s_col and psar_l_col in df_calc.columns and psar_s_col in df_calc.columns:
                # Check if one column exists and is entirely NaN while the other is not
                l_is_nan = df_calc[psar_l_col].isnull().all()
                s_is_nan = df_calc[psar_s_col].isnull().all()
                if l_is_nan and not s_is_nan:
                    df_calc[psar_l_col] = df_calc[psar_s_col] # Fill long from short
                    self.logger.debug(f"Filled NaN PSAR long column from short column for {self.symbol}")
                elif s_is_nan and not l_is_nan:
                    df_calc[psar_s_col] = df_calc[psar_l_col] # Fill short from long
                    self.logger.debug(f"Filled NaN PSAR short column from long column for {self.symbol}")

            # Result df_calc now contains original OHLCV (as float) + indicators (as float)
            # Return this DataFrame. Interpretation logic will use floats.
            return df_calc

        except Exception as e:
            # Log the specific FutureWarning related to dtype incompatibility if it occurs here
            if isinstance(e, FutureWarning) and 'incompatible dtype' in str(e):
                 self.logger.warning(f"Caught FutureWarning during pandas_ta calculation: {e}. This might relate to internal assignments within the library.")
            else:
                 self.logger.exception(f"Error calculating indicators using pandas_ta strategy for {self.symbol}: {e}")
            # Return the original DataFrame without indicators on failure
            return df

    def _calculate_levels(self, df: pd.DataFrame, current_price: Decimal) -> dict:
        """
        Calculates potential support, resistance, and pivot levels based on historical data.

        Args:
            df: DataFrame containing OHLCV data (expects numeric types, preferably float for consistency with indicators).
            current_price: The current market price as a Decimal.

        Returns:
            A dictionary containing 'support', 'resistance', and 'pivot' levels.
            Levels are stored as {level_name: Decimal(price)}.
        """
        levels: Dict[str, Any] = {"support": {}, "resistance": {}, "pivot": None}
        if df.empty or len(df) < 1: # Need at least 1 row for High/Low/Close
            self.logger.warning("Insufficient data for level calculation (need at least 1 row).")
            return levels
        if not isinstance(current_price, Decimal) or not current_price.is_finite():
             self.logger.error("Invalid current_price type or value for level calculation. Expected finite Decimal.")
             return levels

        try:
            # Use the full period available in the DataFrame for more robust levels
            # Ensure values are finite numbers before calculation
            # Use .loc to avoid potential SettingWithCopyWarning if df is a slice
            high_val = df.loc[:, "high"].max()
            low_val = df.loc[:, "low"].min()
            # Use the most recent close for pivot calculation
            close_val = df.loc[:, "close"].iloc[-1]

            # Validate that we have finite numeric values
            if not all(pd.notna(v) and np.isfinite(v) for v in [high_val, low_val, close_val]):
                self.logger.warning("NaN or non-finite values found in OHLC data. Cannot calculate levels accurately.")
                return levels

            # Convert valid OHLC values to Decimal for precise level calculations
            high = Decimal(str(high_val))
            low = Decimal(str(low_val))
            close = Decimal(str(close_val))

            # --- Fibonacci Retracement Levels ---
            diff = high - low
            # Check for zero or negligible difference to avoid errors/meaningless levels
            if diff > DECIMAL_COMPARISON_THRESHOLD:
                # Standard Fibonacci levels as Decimals
                fib_ratios = [Decimal("0.236"), Decimal("0.382"), Decimal("0.5"), Decimal("0.618"), Decimal("0.786")]
                fib_levels = {}
                # Calculate levels relative to the high/low range
                for ratio in fib_ratios:
                    # Level based on retracement from high
                    level_down = high - diff * ratio
                    fib_levels[f"Fib {ratio*100:.1f}%"] = level_down # Simplified name

                # Add High and Low as natural S/R
                fib_levels["Period High"] = high
                fib_levels["Period Low"] = low

                # Classify Fibonacci levels as support or resistance based on current price
                for label, value in fib_levels.items():
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value
                    # else: Level is exactly the current price, could be either? Ignore for now.
            else:
                 self.logger.debug("Price range (High - Low) is too small for Fibonacci calculation.")


            # --- Pivot Points (Classical Method) ---
            try:
                pivot = (high + low + close) / Decimal(3)
                levels["pivot"] = pivot

                # Calculate classical support and resistance levels based on the pivot
                r1 = (Decimal(2) * pivot) - low
                s1 = (Decimal(2) * pivot) - high
                r2 = pivot + diff # diff = high - low
                s2 = pivot - diff
                r3 = high + Decimal(2) * (pivot - low)
                s3 = low - Decimal(2) * (high - pivot)

                pivot_levels = {"R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

                # Classify pivot levels as support or resistance
                for label, value in pivot_levels.items():
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value

            except (InvalidOperation, ArithmeticError) as e:
                self.logger.error(f"Error during pivot point calculation: {e}")

        except (TypeError, ValueError, InvalidOperation, IndexError, KeyError) as e:
            self.logger.error(f"Error calculating levels for {self.symbol}: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error calculating levels for {self.symbol}: {e}")

        # Sort levels by price for cleaner output later (optional)
        # Ensure values are Decimals before sorting
        levels["support"] = dict(sorted(
            [(k, v) for k, v in levels["support"].items() if isinstance(v, Decimal)],
            key=lambda item: item[1], reverse=True
        )) # Highest support first
        levels["resistance"] = dict(sorted(
             [(k, v) for k, v in levels["resistance"].items() if isinstance(v, Decimal)],
            key=lambda item: item[1]
        )) # Lowest resistance first

        return levels

    def _analyze_orderbook(self, orderbook: Optional[dict], current_price: Decimal, levels: dict) -> dict:
        """
        Analyzes order book data for buy/sell pressure and identifies significant
        clusters of orders near calculated support/resistance levels.

        Args:
            orderbook: The fetched order book dictionary (containing 'bids' and 'asks').
            current_price: The current market price as a Decimal.
            levels: Dictionary containing calculated 'support' and 'resistance' levels (values are Decimals).

        Returns:
            A dictionary containing analysis results:
            - 'pressure': String indicating overall buy/sell pressure (with color formatting).
            - 'total_bid_usd': Total USD value of bids in the fetched book (Decimal).
            - 'total_ask_usd': Total USD value of asks in the fetched book (Decimal).
            - 'clusters': A list of dictionaries, each describing a significant order cluster found.
        """
        analysis: Dict[str, Any] = {
            "clusters": [],
            "pressure": SignalState.NA.value, # Default to N/A string value
            "total_bid_usd": DECIMAL_ZERO,
            "total_ask_usd": DECIMAL_ZERO
        }
        if not orderbook or not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
            self.logger.debug("Orderbook data incomplete or unavailable for analysis.")
            return analysis
        if not isinstance(current_price, Decimal) or not current_price.is_finite():
             self.logger.error("Invalid current_price type or value for orderbook analysis. Expected finite Decimal.")
             return analysis

        try:
            # Convert bids and asks to DataFrames for easier processing
            # Ensure data is converted to Decimal, handle errors gracefully
            def to_decimal_df(data: List[List[Union[str, float, int]]], columns: List[str]) -> pd.DataFrame:
                if not data: return pd.DataFrame(columns=columns)
                try:
                    # Create DataFrame with object dtype initially to hold Decimals
                    df = pd.DataFrame(data, columns=columns, dtype=object)
                    for col in columns:
                        # Convert via string for precision, coerce errors to pd.NA
                        df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)
                    # Drop rows where essential price/size conversion failed
                    df.dropna(subset=columns, inplace=True)
                    # Filter out zero prices/sizes after conversion
                    if 'price' in columns: df = df[df['price'] > DECIMAL_COMPARISON_THRESHOLD]
                    if 'size' in columns: df = df[df['size'] > DECIMAL_COMPARISON_THRESHOLD]
                    return df
                except (ValueError, TypeError, InvalidOperation, KeyError) as e:
                     self.logger.error(f"Error converting orderbook data to Decimal DataFrame: {e}")
                     return pd.DataFrame(columns=columns) # Return empty df on error

            bids_df = to_decimal_df(orderbook.get('bids', []), ['price', 'size'])
            asks_df = to_decimal_df(orderbook.get('asks', []), ['price', 'size'])

            if bids_df.empty and asks_df.empty:
                self.logger.debug("Orderbook is empty after cleaning zero/invalid values.")
                analysis["pressure"] = Color.format("Neutral Pressure", Color.YELLOW) # Indicate neutral if empty
                return analysis

            # Calculate USD value for each level and total pressure
            if not bids_df.empty:
                # Ensure calculation uses Decimals
                bids_df['value_usd'] = bids_df['price'].astype(object) * bids_df['size'].astype(object)
                analysis["total_bid_usd"] = bids_df['value_usd'].sum()
            if not asks_df.empty:
                asks_df['value_usd'] = asks_df['price'].astype(object) * asks_df['size'].astype(object)
                analysis["total_ask_usd"] = asks_df['value_usd'].sum()

            total_bid_usd = analysis["total_bid_usd"]
            total_ask_usd = analysis["total_ask_usd"]
            total_value = total_bid_usd + total_ask_usd

            if total_value > DECIMAL_COMPARISON_THRESHOLD:
                # Simple pressure calculation based on total value ratio
                bid_ask_ratio = total_bid_usd / total_value
                # Define thresholds for high pressure (e.g., > 60% or < 40%)
                high_pressure_threshold = Decimal("0.6")
                low_pressure_threshold = Decimal("0.4")
                if bid_ask_ratio > high_pressure_threshold:
                    analysis["pressure"] = Color.format("High Buy Pressure", Color.GREEN)
                elif bid_ask_ratio < low_pressure_threshold:
                    analysis["pressure"] = Color.format("High Sell Pressure", Color.RED)
                else:
                    analysis["pressure"] = Color.format("Neutral Pressure", Color.YELLOW)
            else:
                 analysis["pressure"] = Color.format("Neutral Pressure", Color.YELLOW) # Neutral if total value is zero

            # --- Cluster Analysis ---
            # Use float threshold from config, convert to Decimal
            cluster_threshold_usd = Decimal(str(self.orderbook_settings.cluster_threshold_usd))
            # Convert proximity percentage to a Decimal factor
            proximity_factor = Decimal(str(self.orderbook_settings.cluster_proximity_pct / 100.0))

            # Combine all calculated support, resistance, and pivot levels for checking
            all_levels_to_check: Dict[str, Decimal] = {}
            # Ensure only valid Decimal levels are included
            all_levels_to_check.update({k: v for k, v in levels.get("support", {}).items() if isinstance(v, Decimal)})
            all_levels_to_check.update({k: v for k, v in levels.get("resistance", {}).items() if isinstance(v, Decimal)})
            pivot_level = levels.get("pivot")
            if pivot_level is not None and isinstance(pivot_level, Decimal):
                all_levels_to_check["Pivot"] = pivot_level

            processed_clusters = set() # Track processed levels to avoid duplicates if S/R overlap

            for name, level_price in all_levels_to_check.items():
                if not isinstance(level_price, Decimal) or level_price <= DECIMAL_COMPARISON_THRESHOLD: continue

                # Define the price range around the level to check for clusters
                price_delta = level_price * proximity_factor
                min_price = level_price - price_delta
                max_price = level_price + price_delta

                # Check for significant bid clusters (potential support confirmation) near the level
                if not bids_df.empty:
                    # Ensure comparison is between Decimals
                    bids_near_level = bids_df[(bids_df['price'] >= min_price) & (bids_df['price'] <= max_price)]
                    bid_cluster_value_usd = bids_near_level['value_usd'].sum()

                    if bid_cluster_value_usd >= cluster_threshold_usd:
                        # Use level price for uniqueness check, format consistently
                        cluster_id = f"BID_{level_price:.8f}" # Use sufficient precision for ID
                        if cluster_id not in processed_clusters:
                            analysis["clusters"].append({
                                "type": "Support", # Bid cluster implies potential support
                                "level_name": name,
                                "level_price": level_price,
                                "cluster_value_usd": bid_cluster_value_usd,
                                "price_range": (min_price, max_price) # Store the range checked
                            })
                            processed_clusters.add(cluster_id)

                # Check for significant ask clusters (potential resistance confirmation) near the level
                if not asks_df.empty:
                    asks_near_level = asks_df[(asks_df['price'] >= min_price) & (asks_df['price'] <= max_price)]
                    ask_cluster_value_usd = asks_near_level['value_usd'].sum()

                    if ask_cluster_value_usd >= cluster_threshold_usd:
                        cluster_id = f"ASK_{level_price:.8f}"
                        if cluster_id not in processed_clusters:
                            analysis["clusters"].append({
                                "type": "Resistance", # Ask cluster implies potential resistance
                                "level_name": name,
                                "level_price": level_price,
                                "cluster_value_usd": ask_cluster_value_usd,
                                "price_range": (min_price, max_price)
                            })
                            processed_clusters.add(cluster_id)

        except (KeyError, ValueError, TypeError, InvalidOperation, AttributeError) as e:
            self.logger.error(f"Error analyzing orderbook for {self.symbol}: {e}")
            self.logger.debug(traceback.format_exc()) # Log stack trace for debugging OB issues
        except Exception as e:
            # Catch any unexpected errors during order book analysis
            self.logger.exception(f"Unexpected error analyzing orderbook for {self.symbol}: {e}")

        # Sort identified clusters by their USD value (descending) for prominence
        analysis["clusters"] = sorted(
            analysis.get("clusters", []),
            # Ensure 'cluster_value_usd' is Decimal for sorting
            key=lambda x: x.get('cluster_value_usd', DECIMAL_ZERO),
            reverse=True
        )

        return analysis

    # --- Interpretation Helpers ---

    def _get_val(self, row: pd.Series, key: Optional[str], default: Any = None) -> Any:
        """
        Safely gets a value from a Pandas Series (representing a DataFrame row),
        handling missing keys and NaN/None/NA values.

        Args:
            row: The pandas Series (row).
            key: The key (column name) to retrieve. Can be None.
            default: The value to return if the key is missing or the value is NaN/None/NA.

        Returns:
            The value from the Series, or the default value.
        """
        if key is None or key not in row.index: # Check row.index for validity
            # self.logger.debug(f"Indicator key '{key}' not found in DataFrame row.")
            return default
        val = row[key]
        # Check for pandas NA, numpy NaN, or Python None
        # pd.isna() handles all these cases
        return default if pd.isna(val) else val

    def _format_signal(self, label: str, value: Any, signal: SignalState, precision: int = 2, details: str = "") -> str:
        """
        Formats a single line of the analysis summary, applying color based on the signal state.

        Args:
            label: The label for the indicator/signal (e.g., "RSI", "MACD Cross").
            value: The numeric value of the indicator (or relevant info). Can be None or non-numeric.
            signal: The SignalState enum member representing the interpretation.
            precision: The decimal precision for formatting the value (if numeric).
            details: Additional context or information string to append.

        Returns:
            A formatted string ready for printing, including color codes.
        """
        # Format the value part carefully
        value_str = ""
        if value is not None:
            # Try formatting as Decimal/float if possible
            try:
                 # Attempt numeric formatting only if it's likely a number
                 if isinstance(value, (Decimal, float, int, np.number)):
                     value_str = format_decimal(value, precision)
                 else: # Otherwise, just convert to string
                     value_str = str(value)
            except (ValueError, TypeError):
                 value_str = str(value) # Fallback to simple string conversion

        # Map SignalState to Color
        color_map = {
            SignalState.BULLISH: Color.GREEN, SignalState.STRONG_BULLISH: Color.GREEN,
            SignalState.OVERSOLD: Color.GREEN, SignalState.INCREASING: Color.GREEN,
            SignalState.ACCUMULATION: Color.GREEN, SignalState.FLIP_BULLISH: Color.GREEN,
            SignalState.ABOVE_SIGNAL: Color.GREEN,

            SignalState.BEARISH: Color.RED, SignalState.STRONG_BEARISH: Color.RED,
            SignalState.OVERBOUGHT: Color.RED, SignalState.DECREASING: Color.RED,
            SignalState.DISTRIBUTION: Color.RED, SignalState.FLIP_BEARISH: Color.RED,
            SignalState.BELOW_SIGNAL: Color.RED,

            SignalState.NEUTRAL: Color.YELLOW, SignalState.RANGING: Color.YELLOW,
            SignalState.WITHIN_BANDS: Color.YELLOW, SignalState.AVERAGE_VOLUME: Color.YELLOW,
            SignalState.FLAT: Color.YELLOW, SignalState.NONE: Color.YELLOW,
            SignalState.BREAKDOWN_LOWER: Color.YELLOW, # Color debatable, Yellow seems neutral warning
            SignalState.BREAKOUT_UPPER: Color.YELLOW, # Color debatable, Yellow seems neutral warning

            SignalState.HIGH_VOLUME: Color.PURPLE, SignalState.LOW_VOLUME: Color.PURPLE,

            SignalState.NA: Color.YELLOW, # Default color for N/A or unmapped states
        }
        color = color_map.get(signal, Color.YELLOW) # Default to yellow if signal not in map

        # Get the string value from the SignalState enum
        signal_text = signal.value if isinstance(signal, SignalState) else str(signal)

        # Construct the output string
        label_part = f"{label}"
        # Only show value part if value_str is meaningful (not empty or N/A)
        value_part = f" ({value_str})" if value_str and value_str != "N/A" else ""
        signal_part = f": {Color.format(signal_text, color)}"
        details_part = f" ({details})" if details else ""

        return f"{label_part}{value_part}{signal_part}{details_part}"

    def _interpret_trend(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets trend indicators (EMAs, ADX, PSAR)."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- EMA Alignment ---
        signal_key = "ema_trend"
        signals[signal_key] = SignalState.NA # Default
        if flags.ema_alignment:
            # Get values as floats (since indicators are calculated as floats)
            ema_short = self._get_val(last_row, cols.get('ema_short'), default=np.nan)
            ema_long = self._get_val(last_row, cols.get('ema_long'), default=np.nan)
            label = f"EMA Align ({self.indicator_settings.ema_short_period}/{self.indicator_settings.ema_long_period})"
            signal = SignalState.NA
            details = ""
            value = None # No single value for alignment

            if np.isfinite(ema_short) and np.isfinite(ema_long):
                signal = SignalState.NEUTRAL
                comparison_char = '>' if ema_short > ema_long else '<' if ema_short < ema_long else '='
                if ema_short > ema_long: signal = SignalState.BULLISH
                elif ema_short < ema_long: signal = SignalState.BEARISH
                # Format floats for display
                details = f"S:{format_decimal(ema_short)} {comparison_char} L:{format_decimal(ema_long)}"
                signals[signal_key] = signal
            else:
                 details = "N/A" # Indicate missing data

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        # --- ADX Trend Strength ---
        signal_key = "adx_trend"
        signals[signal_key] = SignalState.NA # Default
        if flags.adx_trend_strength:
            adx = self._get_val(last_row, cols.get('adx'), default=np.nan)
            dmp = self._get_val(last_row, cols.get('dmp'), default=np.nan) # +DI
            dmn = self._get_val(last_row, cols.get('dmn'), default=np.nan) # -DI
            label = f"ADX Trend ({self.indicator_settings.adx_period})"
            signal = SignalState.NA
            details = ""
            value = adx # Display ADX value

            if np.isfinite(adx) and np.isfinite(dmp) and np.isfinite(dmn):
                trend_threshold = self.thresholds.adx_trending # Already float

                if adx >= trend_threshold:
                    if dmp > dmn:
                        signal = SignalState.STRONG_BULLISH
                        details = f"+DI ({format_decimal(dmp)}) > -DI ({format_decimal(dmn)})"
                    elif dmn > dmp:
                        signal = SignalState.STRONG_BEARISH
                        details = f"-DI ({format_decimal(dmn)}) > +DI ({format_decimal(dmp)})"
                    else: # dmp == dmn (unlikely but possible)
                         signal = SignalState.RANGING # Treat as ranging if DI lines are equal
                         details = "+DI == -DI"
                else:
                    signal = SignalState.RANGING
                    details = f"ADX < {trend_threshold:.2f}"
                signals[signal_key] = signal
            else:
                 details = "N/A"

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        # --- PSAR Flip ---
        signal_key_trend = "psar_trend"
        signal_key_signal = "psar_signal"
        signals[signal_key_trend] = SignalState.NA # Default trend
        signals[signal_key_signal] = SignalState.NA # Default signal (flip or trend)
        if flags.psar_flip:
            # PSAR values from pandas_ta can appear in either long or short column depending on trend
            psar_l = self._get_val(last_row, cols.get('psar_long'), default=np.nan)
            psar_s = self._get_val(last_row, cols.get('psar_short'), default=np.nan)
            close_val = self._get_val(last_row, 'close', default=np.nan)

            label = "PSAR"
            signal = SignalState.NA
            trend_signal = SignalState.NA
            details = ""
            psar_display_val = np.nan # The PSAR value to display

            # Determine the active PSAR value and current trend
            if np.isfinite(close_val):
                # If psar_long is valid, it means trend is currently UP (price > psar_long)
                if np.isfinite(psar_l):
                    psar_display_val = psar_l
                    trend_signal = SignalState.BULLISH
                # If psar_short is valid, it means trend is currently DOWN (price < psar_short)
                elif np.isfinite(psar_s):
                    psar_display_val = psar_s
                    trend_signal = SignalState.BEARISH
                # else: Both psar_l and psar_s are NaN, cannot determine trend

            if trend_signal != SignalState.NA:
                signals[signal_key_trend] = trend_signal # Store the determined trend
                signal = trend_signal # Base signal is the current trend

                # Check if a reversal occurred on *this* candle by comparing current trend with previous.
                prev_psar_l = self._get_val(prev_row, cols.get('psar_long'), default=np.nan)
                prev_psar_s = self._get_val(prev_row, cols.get('psar_short'), default=np.nan)
                prev_trend_signal = SignalState.NA
                if np.isfinite(prev_psar_l): prev_trend_signal = SignalState.BULLISH
                elif np.isfinite(prev_psar_s): prev_trend_signal = SignalState.BEARISH

                if prev_trend_signal != SignalState.NA and trend_signal != prev_trend_signal:
                    # Trend flipped between previous and current candle
                    signal = SignalState.FLIP_BULLISH if trend_signal == SignalState.BULLISH else SignalState.FLIP_BEARISH
                    details = "Just Flipped!"

                signals[signal_key_signal] = signal # Store the final signal (flip or trend)
            else:
                 details = "N/A"

            summary_lines.append(self._format_signal(label, psar_display_val, signal, precision=4, details=details))

        return summary_lines, signals

    def _interpret_oscillators(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets oscillator indicators (RSI, MFI, CCI, Williams %R, StochRSI)."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        thresh = self.thresholds # Contains floats
        cols = self.col_names

        # --- RSI Level ---
        signal_key = "rsi_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.rsi_threshold:
            rsi = self._get_val(last_row, cols.get('rsi'), default=np.nan)
            label = f"RSI ({self.indicator_settings.rsi_period})"
            signal = SignalState.NA
            if np.isfinite(rsi):
                signal = SignalState.NEUTRAL
                if rsi >= thresh.rsi_overbought: signal = SignalState.OVERBOUGHT
                elif rsi <= thresh.rsi_oversold: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, rsi, signal))

        # --- MFI Level ---
        signal_key = "mfi_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.mfi_threshold:
            mfi = self._get_val(last_row, cols.get('mfi'), default=np.nan)
            label = f"MFI ({self.indicator_settings.mfi_period})"
            signal = SignalState.NA
            if np.isfinite(mfi):
                signal = SignalState.NEUTRAL
                if mfi >= thresh.mfi_overbought: signal = SignalState.OVERBOUGHT
                elif mfi <= thresh.mfi_oversold: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, mfi, signal))

        # --- CCI Level ---
        signal_key = "cci_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.cci_threshold:
            cci = self._get_val(last_row, cols.get('cci'), default=np.nan)
            label = f"CCI ({self.indicator_settings.cci_period})"
            signal = SignalState.NA
            if np.isfinite(cci):
                signal = SignalState.NEUTRAL
                if cci >= thresh.cci_overbought: signal = SignalState.OVERBOUGHT
                elif cci <= thresh.cci_oversold: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, cci, signal))

        # --- Williams %R Level ---
        signal_key = "wr_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.williams_r_threshold:
            wr = self._get_val(last_row, cols.get('willr'), default=np.nan)
            label = f"Williams %R ({self.indicator_settings.williams_r_period})"
            signal = SignalState.NA
            if np.isfinite(wr):
                signal = SignalState.NEUTRAL
                # Overbought is when value is *above* the OB threshold (closer to 0)
                if wr >= thresh.williams_r_overbought: signal = SignalState.OVERBOUGHT
                # Oversold is when value is *below* the OS threshold (closer to -100)
                elif wr <= thresh.williams_r_oversold: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, wr, signal))

        # --- StochRSI Cross ---
        signal_key = "stochrsi_cross"
        signals[signal_key] = SignalState.NA # Default
        if flags.stoch_rsi_cross:
            k_now = self._get_val(last_row, cols.get('stochrsi_k'), default=np.nan)
            d_now = self._get_val(last_row, cols.get('stochrsi_d'), default=np.nan)
            k_prev = self._get_val(prev_row, cols.get('stochrsi_k'), default=np.nan)
            d_prev = self._get_val(prev_row, cols.get('stochrsi_d'), default=np.nan)
            label = f"StochRSI ({self.indicator_settings.stoch_k_period}/{self.indicator_settings.stoch_d_period})"
            signal = SignalState.NA
            details = ""
            # Format value string carefully, handling potential NaNs
            k_str = format_decimal(k_now)
            d_str = format_decimal(d_now)
            value = f"K:{k_str} D:{d_str}" if k_str != "N/A" or d_str != "N/A" else None

            if np.isfinite(k_now) and np.isfinite(d_now) and np.isfinite(k_prev) and np.isfinite(d_prev):
                # Check for bullish crossover: K crosses above D
                crossed_bullish = k_now > d_now and k_prev <= d_prev
                # Check for bearish crossover: K crosses below D
                crossed_bearish = k_now < d_now and k_prev >= d_prev

                if crossed_bullish:
                    signal = SignalState.BULLISH
                    details = "Crossed Up"
                elif crossed_bearish:
                    signal = SignalState.BEARISH
                    details = "Crossed Down"
                # If no cross, indicate the current state (K above/below D)
                elif k_now > d_now:
                    signal = SignalState.ABOVE_SIGNAL # K is above D
                    details = "K > D"
                elif k_now < d_now:
                    signal = SignalState.BELOW_SIGNAL # K is below D
                    details = "K < D"
                else:
                    signal = SignalState.NEUTRAL # K == D
                    details = "K == D"
                signals[signal_key] = signal
            else:
                 details = "N/A" # Indicate missing data for cross check

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        return summary_lines, signals

    def _interpret_macd(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets MACD line/signal cross and basic divergence."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- MACD Cross ---
        signal_key = "macd_cross"
        signals[signal_key] = SignalState.NA # Default
        if flags.macd_cross:
            line_now = self._get_val(last_row, cols.get('macd_line'), default=np.nan)
            sig_now = self._get_val(last_row, cols.get('macd_signal'), default=np.nan)
            line_prev = self._get_val(prev_row, cols.get('macd_line'), default=np.nan)
            sig_prev = self._get_val(prev_row, cols.get('macd_signal'), default=np.nan)
            label = f"MACD ({self.indicator_settings.macd_fast}/{self.indicator_settings.macd_slow}/{self.indicator_settings.macd_signal})"
            signal = SignalState.NA
            details = ""
            line_str = format_decimal(line_now, 4)
            sig_str = format_decimal(sig_now, 4)
            value = f"L:{line_str} S:{sig_str}" if line_str != "N/A" or sig_str != "N/A" else None

            if np.isfinite(line_now) and np.isfinite(sig_now) and np.isfinite(line_prev) and np.isfinite(sig_prev):
                crossed_bullish = line_now > sig_now and line_prev <= sig_prev
                crossed_bearish = line_now < sig_now and line_prev >= sig_prev

                if crossed_bullish:
                    signal = SignalState.BULLISH
                    details = "Crossed Up"
                elif crossed_bearish:
                    signal = SignalState.BEARISH
                    details = "Crossed Down"
                elif line_now > sig_now:
                    signal = SignalState.ABOVE_SIGNAL
                    details = "Line > Signal"
                elif line_now < sig_now:
                    signal = SignalState.BELOW_SIGNAL
                    details = "Line < Signal"
                else:
                    signal = SignalState.NEUTRAL
                    details = "Line == Signal"
                signals[signal_key] = signal
            else:
                 details = "N/A"

            summary_lines.append(self._format_signal(label, value, signal, precision=4, details=details)) # Use higher precision for MACD

        # --- MACD Divergence (Basic 2-point check) ---
        # WARNING: This is a very simplified check and prone to false signals.
        # Real divergence analysis requires pattern recognition over multiple pivots.
        signal_key = "macd_divergence"
        signals[signal_key] = SignalState.NA # Default
        if flags.macd_divergence:
            hist_now = self._get_val(last_row, cols.get('macd_hist'), default=np.nan)
            hist_prev = self._get_val(prev_row, cols.get('macd_hist'), default=np.nan)
            price_now = self._get_val(last_row, 'close', default=np.nan)
            price_prev = self._get_val(prev_row, 'close', default=np.nan)
            signal = SignalState.NA # Default to NA if data missing

            if np.isfinite(hist_now) and np.isfinite(hist_prev) and np.isfinite(price_now) and np.isfinite(price_prev):
                signal = SignalState.NONE # Default to None (no divergence detected)

                # Basic Bullish Divergence: Lower low in price, higher low in histogram (near/below zero)
                if price_now < price_prev and hist_now > hist_prev and (hist_prev < 0 or hist_now < 0):
                    signal = SignalState.BULLISH
                    summary_lines.append(Color.format("Potential Bullish MACD Divergence", Color.GREEN))

                # Basic Bearish Divergence: Higher high in price, lower high in histogram (near/above zero)
                elif price_now > price_prev and hist_now < hist_prev and (hist_prev > 0 or hist_now > 0):
                    signal = SignalState.BEARISH
                    summary_lines.append(Color.format("Potential Bearish MACD Divergence", Color.RED))

                signals[signal_key] = signal
            # No separate summary line added unless divergence is detected

        return summary_lines, signals

    def _interpret_bbands(self, last_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets Bollinger Bands breakouts/position."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- Bollinger Bands Break/Position ---
        signal_key = "bbands_signal"
        signals[signal_key] = SignalState.NA # Default
        if flags.bollinger_bands_break:
            upper = self._get_val(last_row, cols.get('bb_upper'), default=np.nan)
            lower = self._get_val(last_row, cols.get('bb_lower'), default=np.nan)
            middle = self._get_val(last_row, cols.get('bb_mid'), default=np.nan)
            close_val = self._get_val(last_row, 'close', default=np.nan)
            label = f"BBands ({self.indicator_settings.bollinger_bands_period}/{self.indicator_settings.bollinger_bands_std_dev})"
            signal = SignalState.NA
            details = ""
            value = close_val # Display current close price relative to bands

            if np.isfinite(upper) and np.isfinite(lower) and np.isfinite(middle) and np.isfinite(close_val):
                signal = SignalState.WITHIN_BANDS # Default assumption
                details = f"L:{format_decimal(lower)} M:{format_decimal(middle)} U:{format_decimal(upper)}"

                if close_val > upper:
                    signal = SignalState.BREAKOUT_UPPER # Price above upper band
                    details += " (Price > Upper)"
                elif close_val < lower:
                    signal = SignalState.BREAKDOWN_LOWER # Price below lower band
                    details += " (Price < Lower)"
                # Optional: Check position relative to middle band if within bands
                elif close_val > middle: details += " (Above Middle)"
                elif close_val < middle: details += " (Below Middle)"
                else: details += " (On Middle)"

                signals[signal_key] = signal
            else:
                 details = "N/A"

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        return summary_lines, signals

    def _interpret_volume(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets volume levels and volume-based indicators (OBV, ADOSC)."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names
        thresh = self.thresholds # Access volume thresholds

        # --- Volume Level vs MA ---
        signal_key = "volume_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.volume_confirmation:
            volume = self._get_val(last_row, 'volume', default=np.nan)
            vol_ma = self._get_val(last_row, cols.get('vol_ma'), default=np.nan)
            label = f"Volume vs MA({self.indicator_settings.volume_ma_period})"
            signal = SignalState.NA
            details = ""
            value = volume # Display current volume

            if np.isfinite(volume) and np.isfinite(vol_ma):
                if vol_ma > 1e-9: # Avoid division by zero or near-zero MA
                    signal = SignalState.AVERAGE_VOLUME # Default
                    # Use thresholds from config
                    high_thresh_factor = thresh.volume_high_factor
                    low_thresh_factor = thresh.volume_low_factor

                    if volume > vol_ma * high_thresh_factor:
                        signal = SignalState.HIGH_VOLUME
                    elif volume < vol_ma * low_thresh_factor:
                        signal = SignalState.LOW_VOLUME

                    details = f"Vol:{format_decimal(volume, 0)} MA:{format_decimal(vol_ma, 0)}"
                    signals[signal_key] = signal
                else: # Handle zero or very low MA
                    signal = SignalState.LOW_VOLUME if volume < 1 else SignalState.AVERAGE_VOLUME # Treat as low unless volume itself is substantial
                    details = f"Vol:{format_decimal(volume, 0)} MA: ~0"
                    signals[signal_key] = signal
            else:
                 details = "N/A"

            summary_lines.append(self._format_signal(label, value, signal, precision=0, details=details))

        # --- OBV Trend ---
        signal_key = "obv_trend"
        signals[signal_key] = SignalState.NA # Default
        if flags.obv_trend:
            obv_now = self._get_val(last_row, cols.get('obv'), default=np.nan)
            obv_prev = self._get_val(prev_row, cols.get('obv'), default=np.nan)
            label = "OBV Trend"
            signal = SignalState.NA
            value = obv_now

            if np.isfinite(obv_now) and np.isfinite(obv_prev):
                # Check for significant change (optional, avoids noise)
                change_threshold = 0.0001 # Example: 0.01% change relative to previous value
                diff = obv_now - obv_prev
                relative_diff = abs(diff / obv_prev) if abs(obv_prev) > 1e-9 else abs(diff)

                if relative_diff < change_threshold:
                     signal = SignalState.FLAT
                elif obv_now > obv_prev:
                     signal = SignalState.INCREASING
                elif obv_now < obv_prev:
                     signal = SignalState.DECREASING
                else: # Should be covered by FLAT but as fallback
                     signal = SignalState.FLAT

                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, value, signal, precision=0))

        # --- A/D Oscillator Trend (ADOSC) ---
        signal_key = "adi_trend" # Keep key name generic for Accum/Dist
        signals[signal_key] = SignalState.NA # Default
        if flags.adi_trend:
            adosc_now = self._get_val(last_row, cols.get('adosc'), default=np.nan)
            adosc_prev = self._get_val(prev_row, cols.get('adosc'), default=np.nan)
            label = "A/D Osc Trend"
            signal = SignalState.NA
            value = adosc_now

            if np.isfinite(adosc_now) and np.isfinite(adosc_prev):
                signal = SignalState.FLAT # Default trend

                # Accumulation: Generally positive and rising ADOSC
                if adosc_now > 0 and adosc_now > adosc_prev:
                    signal = SignalState.ACCUMULATION
                # Distribution: Generally negative and falling ADOSC
                elif adosc_now < 0 and adosc_now < adosc_prev:
                    signal = SignalState.DISTRIBUTION
                # If not clearly A/D, indicate simple direction
                elif adosc_now > adosc_prev:
                    signal = SignalState.INCREASING
                elif adosc_now < adosc_prev:
                    signal = SignalState.DECREASING

                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, precision=0))

        return summary_lines, signals

    def _interpret_levels_orderbook(self, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> List[str]:
        """Formats the calculated levels and orderbook analysis into summary lines."""
        summary_lines = []
        summary_lines.append(Color.format("\n--- Levels & Orderbook ---", Color.BLUE))

        # --- Levels Summary ---
        pivot = levels.get("pivot") # Should be Decimal or None
        support_levels = levels.get("support", {}) # Dict[str, Decimal]
        resistance_levels = levels.get("resistance", {}) # Dict[str, Decimal]

        if pivot and isinstance(pivot, Decimal):
            summary_lines.append(f"Pivot Point: ${format_decimal(pivot, 4)}")
        else:
             summary_lines.append("Pivot Point: N/A")

        # Show nearest levels (e.g., top 3 closest)
        def get_nearest(level_dict: Dict[str, Decimal], price: Decimal, count: int) -> List[Tuple[str, Decimal]]:
            if not level_dict: return []
            # Calculate absolute difference using Decimals
            try:
                # Filter out non-Decimal values just in case
                valid_levels = [(k, v) for k, v in level_dict.items() if isinstance(v, Decimal)]
                return sorted(valid_levels, key=lambda item: abs(item[1] - price))[:count]
            except TypeError as e:
                 self.logger.error(f"TypeError sorting levels: {e}. Check level data types.")
                 return [] # Return empty list on error

        nearest_supports = get_nearest(support_levels, current_price, 3)
        nearest_resistances = get_nearest(resistance_levels, current_price, 3)

        if nearest_supports:
            summary_lines.append("Nearest Support:")
            for name, price in nearest_supports:
                summary_lines.append(f"  > {name}: ${format_decimal(price, 4)}")
        else:
             summary_lines.append("Nearest Support: None Calculated")

        if nearest_resistances:
            summary_lines.append("Nearest Resistance:")
            for name, price in nearest_resistances:
                summary_lines.append(f"  > {name}: ${format_decimal(price, 4)}")
        else:
            summary_lines.append("Nearest Resistance: None Calculated")

        if not nearest_supports and not nearest_resistances and (not pivot or not isinstance(pivot, Decimal)):
            summary_lines.append(Color.format("No significant levels calculated.", Color.YELLOW))

        # --- Orderbook Summary ---
        summary_lines.append("") # Add a blank line
        ob_limit = self.orderbook_settings.limit
        # Values from _analyze_orderbook are already Decimals or formatted strings
        total_bid_usd = orderbook_analysis.get('total_bid_usd', DECIMAL_ZERO)
        total_ask_usd = orderbook_analysis.get('total_ask_usd', DECIMAL_ZERO)

        if total_bid_usd + total_ask_usd > DECIMAL_COMPARISON_THRESHOLD:
            # Pressure is already formatted with color in _analyze_orderbook
            pressure_str = orderbook_analysis.get('pressure', SignalState.NA.value)
            summary_lines.append(f"OB Pressure (Top {ob_limit}): {pressure_str}")
            summary_lines.append(f"OB Value (Bids): ${format_decimal(total_bid_usd, 0)}")
            summary_lines.append(f"OB Value (Asks): ${format_decimal(total_ask_usd, 0)}")

            clusters = orderbook_analysis.get("clusters", []) # List of dicts
            if clusters:
                # Display top N clusters (e.g., 5)
                max_clusters_to_show = 5
                summary_lines.append(Color.format(f"Significant OB Clusters (Top {max_clusters_to_show}):", Color.PURPLE))
                for cluster in clusters[:max_clusters_to_show]:
                    cluster_type = cluster.get('type', 'N/A')
                    color = Color.GREEN if cluster_type == "Support" else Color.RED if cluster_type == "Resistance" else Color.YELLOW
                    level_name = cluster.get('level_name', 'N/A')
                    level_price = cluster.get('level_price') # Should be Decimal
                    cluster_val = cluster.get('cluster_value_usd') # Should be Decimal

                    level_price_f = format_decimal(level_price, 4)
                    cluster_val_f = format_decimal(cluster_val, 0)

                    summary_lines.append(Color.format(f"  {cluster_type} near {level_name} (${level_price_f}) - Value: ${cluster_val_f}", color))
            else:
                summary_lines.append(Color.format("No significant OB clusters found near levels.", Color.YELLOW))
        else:
            summary_lines.append(Color.format(f"Orderbook analysis unavailable or empty (Top {ob_limit}).", Color.YELLOW))

        return summary_lines

    def _interpret_analysis(self, df: pd.DataFrame, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> dict:
        """
        Combines all interpretation steps into a final summary and signal dictionary.

        Args:
            df: DataFrame with OHLCV and calculated indicator data (indicators as floats).
            current_price: Current market price (Decimal).
            levels: Calculated support/resistance/pivot levels (Decimals).
            orderbook_analysis: Results from order book analysis (Decimals).

        Returns:
            A dictionary containing:
            - 'summary': A list of formatted strings for the analysis report.
            - 'signals': A dictionary mapping signal keys (e.g., 'ema_trend') to
                         their corresponding SignalState enum *values* (strings).
        """
        interpretation: Dict[str, Any] = {"summary": [], "signals": {}}

        # Define all expected signal keys to ensure they exist in the output, even if NA
        all_signal_keys = [
            "ema_trend", "adx_trend", "psar_trend", "psar_signal",
            "rsi_level", "mfi_level", "cci_level", "wr_level", "stochrsi_cross",
            "macd_cross", "macd_divergence", "bbands_signal",
            "volume_level", "obv_trend", "adi_trend"
        ]
        # Initialize all signals to NA enum value
        interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}


        if df.empty or len(df) < 2:
            self.logger.warning(f"Insufficient data ({len(df)} rows) for full interpretation on {self.symbol}.")
            interpretation["summary"].append(Color.format("Insufficient data for full analysis interpretation.", Color.YELLOW))
            # Keep signals as NA initialized above
            return interpretation
        if not isinstance(current_price, Decimal) or not current_price.is_finite():
             # This check should ideally happen before calling this function
             self.logger.error("Invalid current_price type or value for interpretation. Expected finite Decimal.")
             interpretation["summary"].append(Color.format("Invalid current price for interpretation.", Color.RED))
             return interpretation


        try:
            # Ensure index is sequential for iloc[-1] and [-2] access
            # df from _calculate_indicators should already have a clean index if reset_index was used,
            # but resetting again is safe. Use .copy() if df might be a slice.
            # Use .iloc directly as index should be clean after _calculate_indicators
            last_row = df.iloc[-1]
            # Use last_row as prev_row if only one row exists after indicator calculation (handles edge case)
            prev_row = df.iloc[-2] if len(df) >= 2 else last_row

            # --- Run Interpretation Sections ---
            trend_summary, trend_signals = self._interpret_trend(last_row, prev_row)
            osc_summary, osc_signals = self._interpret_oscillators(last_row, prev_row)
            macd_summary, macd_signals = self._interpret_macd(last_row, prev_row)
            bb_summary, bb_signals = self._interpret_bbands(last_row)
            vol_summary, vol_signals = self._interpret_volume(last_row, prev_row)
            level_ob_summary = self._interpret_levels_orderbook(current_price, levels, orderbook_analysis)

            # --- Combine Results ---
            interpretation["summary"].extend(trend_summary)
            interpretation["summary"].extend(osc_summary)
            interpretation["summary"].extend(macd_summary)
            interpretation["summary"].extend(bb_summary)
            interpretation["summary"].extend(vol_summary)
            interpretation["summary"].extend(level_ob_summary)

            # Combine all signal dictionaries, overwriting defaults
            all_signals_enums = {
                **trend_signals, **osc_signals, **macd_signals,
                **bb_signals, **vol_signals
            }

            # Update the interpretation dict, converting Enum members to their string values
            for key, signal_enum in all_signals_enums.items():
                 if key in interpretation["signals"]: # Ensure we only update expected keys
                     interpretation["signals"][key] = signal_enum.value if isinstance(signal_enum, SignalState) else SignalState.NA.value
                 else:
                      self.logger.warning(f"Generated signal key '{key}' not in expected list {all_signal_keys}. Ignoring.")


        except IndexError as e:
            self.logger.error(f"IndexError during interpretation for {self.symbol}. DataFrame length: {len(df)}. Error: {e}")
            interpretation["summary"].append(Color.format("Error accessing data rows for interpretation.", Color.RED))
            # Reset signals to NA on error
            interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}
        except KeyError as e:
             self.logger.error(f"KeyError during interpretation for {self.symbol}. Missing indicator column? Error: {e}")
             interpretation["summary"].append(Color.format(f"Error accessing indicator data ({e}) for interpretation.", Color.RED))
             interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}
        except Exception as e:
            self.logger.exception(f"Unexpected error during analysis interpretation for {self.symbol}: {e}")
            interpretation["summary"].append(Color.format(f"Unexpected error during interpretation: {e}", Color.RED))
            interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}

        return interpretation

    def analyze(self, df_klines_decimal: pd.DataFrame, current_price: Optional[Decimal], orderbook: Optional[dict]) -> dict:
        """
        Main analysis function orchestrating data processing, indicator calculation,
        level calculation, order book analysis, and interpretation.

        Args:
            df_klines_decimal: DataFrame containing OHLCV kline data (expects Decimals).
            current_price: The current market price as a Decimal, or None if unavailable.
            orderbook: The fetched order book dictionary, or None if unavailable.

        Returns:
            A dictionary containing the comprehensive analysis result, including raw
            indicator values, levels, order book insights, and interpretation summary/signals.
        """
        # Initialize the result structure
        analysis_result: Dict[str, Any] = {
            "symbol": self.symbol,
            "timestamp": datetime.now(APP_TIMEZONE).isoformat(),
            "current_price": "N/A",
            "kline_interval": "N/A", # Will be inferred from data
            "levels": {"support": {}, "resistance": {}, "pivot": "N/A"},
            "orderbook_analysis": {"pressure": SignalState.NA.value, "total_bid_usd": "N/A", "total_ask_usd": "N/A", "clusters": []},
            "interpretation": {"summary": [Color.format("Analysis could not be performed.", Color.RED)], "signals": {}},
            "raw_indicators": {} # Store last row's indicator values (as floats)
        }
        # Initialize signals sub-dictionary
        all_signal_keys = [
            "ema_trend", "adx_trend", "psar_trend", "psar_signal",
            "rsi_level", "mfi_level", "cci_level", "wr_level", "stochrsi_cross",
            "macd_cross", "macd_divergence", "bbands_signal",
            "volume_level", "obv_trend", "adi_trend"
        ]
        analysis_result["interpretation"]["signals"] = {key: SignalState.NA.value for key in all_signal_keys}


        # --- Pre-checks ---
        if current_price is None:
             self.logger.warning(f"Current price is unavailable for {self.symbol}. Analysis will be incomplete.")
             analysis_result["interpretation"]["summary"] = [Color.format("Current price unavailable, analysis incomplete.", Color.YELLOW)]
             # Allow analysis to proceed, but interpretation might be limited
        elif not isinstance(current_price, Decimal) or not current_price.is_finite():
             self.logger.error(f"Invalid current_price type or value ({current_price}) received for {self.symbol}. Cannot proceed.")
             analysis_result["interpretation"]["summary"] = [Color.format("Invalid current price type/value, analysis failed.", Color.RED)]
             return analysis_result
        else:
             analysis_result["current_price"] = format_decimal(current_price, 4) # Format price early

        if df_klines_decimal.empty:
            self.logger.error(f"Kline data is empty for {self.symbol}. Cannot perform analysis.")
            # Keep the initial error message in summary
            return analysis_result
        if len(df_klines_decimal) < 2:
            # Allow analysis, but log warning. Interpretation handles < 2 rows.
            self.logger.warning(f"Kline data has only {len(df_klines_decimal)} row(s) for {self.symbol}. Analysis requires >= 2 rows for comparisons; results may be incomplete.")
            # Update summary if it's still the default error
            if "Analysis could not be performed" in analysis_result["interpretation"]["summary"][0]:
                 analysis_result["interpretation"]["summary"] = [Color.format("Warning: Insufficient kline data (< 2 rows) for full analysis.", Color.YELLOW)]


        try:
            # --- Infer Kline Interval ---
            if len(df_klines_decimal) >= 2 and 'timestamp' in df_klines_decimal.columns:
                # Calculate time difference between consecutive timestamps
                time_diffs = df_klines_decimal['timestamp'].diff()
                # Find the most common time difference (mode) as the interval
                # Use dropna() in case the first diff is NaT
                mode_diff = time_diffs.dropna().mode()
                if not mode_diff.empty:
                    # Format timedelta nicely (e.g., '0 days 00:15:00')
                    interval_timedelta = mode_diff[0]
                    # Try to map back to user-friendly interval key
                    seconds = interval_timedelta.total_seconds()
                    minutes = seconds / 60
                    hours = minutes / 60
                    days = hours / 24
                    weeks = days / 7
                    # Approximate mapping back (might not be perfect for all intervals)
                    if minutes in [1, 3, 5, 15, 30]: interval_key = str(int(minutes))
                    elif hours in [1, 2, 4, 6, 12]: interval_key = str(int(hours * 60))
                    elif days == 1: interval_key = "D"
                    elif weeks == 1: interval_key = "W"
                    # Monthly ('M') is harder to infer accurately from timedelta
                    else: interval_key = f"{interval_timedelta}" # Fallback to timedelta string

                    analysis_result["kline_interval"] = interval_key
                else:
                    # Handle cases with only one diff or variable intervals
                    median_diff = time_diffs.median()
                    if pd.notna(median_diff):
                         analysis_result["kline_interval"] = f"~{str(median_diff)} (Median)"
                    else:
                         analysis_result["kline_interval"] = "Variable/Unknown"
            elif len(df_klines_decimal) == 1:
                 analysis_result["kline_interval"] = "Single Candle"


            # --- 1. Calculate Indicators ---
            # Pass the kline df (with Decimals), _calculate_indicators converts to float internally
            df_with_indicators = self._calculate_indicators(df_klines_decimal)

            if df_with_indicators.empty or len(df_with_indicators) == 0:
                self.logger.error(f"Indicator calculation failed or resulted in empty DataFrame for {self.symbol}.")
                if "Analysis could not be performed" in analysis_result["interpretation"]["summary"][0]:
                     analysis_result["interpretation"]["summary"] = [Color.format("Indicator calculation failed.", Color.RED)]
                return analysis_result # Stop analysis if indicators fail critically

            # Store raw indicator values (as floats) from the *last* row for inspection/debugging
            if not df_with_indicators.empty:
                last_row_indicators = df_with_indicators.iloc[-1].to_dict()
                analysis_result["raw_indicators"] = {
                    k: (format_decimal(v, 4) if isinstance(v, (float, np.floating, int, Decimal)) else # Format numerics
                       (v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else # Format dates
                        (str(v) if pd.notna(v) and not isinstance(v, (dict, list, tuple)) else None))) # Stringify others, skip complex/None
                    for k, v in last_row_indicators.items()
                    # Include only expected indicator columns and core OHLCV + timestamp
                    if k in self.col_names.values() or k in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                }


            # --- 2. Calculate Levels ---
            # Levels calculation needs Decimal price, and uses the float OHLCV from df_with_indicators
            levels_decimal = {}
            if isinstance(current_price, Decimal) and current_price.is_finite():
                levels_decimal = self._calculate_levels(df_with_indicators, current_price)
                # Format level prices (which are Decimals) for the result dictionary
                analysis_result["levels"] = {
                    "support": {name: format_decimal(price, 4) for name, price in levels_decimal.get("support", {}).items()},
                    "resistance": {name: format_decimal(price, 4) for name, price in levels_decimal.get("resistance", {}).items()},
                    "pivot": format_decimal(levels_decimal.get("pivot"), 4) if levels_decimal.get("pivot") is not None else "N/A"
                }
            else:
                 # Keep default "N/A" values set during initialization
                 self.logger.warning("Skipping level calculation due to missing or invalid current price.")


            # --- 3. Analyze Orderbook ---
            # Orderbook analysis needs Decimal price and Decimal levels
            orderbook_analysis_raw = {}
            if orderbook and isinstance(current_price, Decimal) and current_price.is_finite():
                 # Pass the levels dictionary containing Decimals
                 orderbook_analysis_raw = self._analyze_orderbook(orderbook, current_price, levels_decimal)
                 # Format orderbook analysis results for output dictionary
                 analysis_result["orderbook_analysis"] = {
                    "pressure": orderbook_analysis_raw.get("pressure", SignalState.NA.value), # Pressure is already formatted
                    "total_bid_usd": format_decimal(orderbook_analysis_raw.get("total_bid_usd", DECIMAL_ZERO), 0),
                    "total_ask_usd": format_decimal(orderbook_analysis_raw.get("total_ask_usd", DECIMAL_ZERO), 0),
                    "clusters": [
                        {
                            "type": c.get("type", "N/A"),
                            "level_name": c.get("level_name", "N/A"),
                            "level_price": format_decimal(c.get("level_price"), 4), # Format Decimal
                            "cluster_value_usd": format_decimal(c.get("cluster_value_usd"), 0), # Format Decimal
                            # Format the tuple elements (Decimals) for price range
                            "price_range": (
                                format_decimal(c.get("price_range", (None, None))[0], 4),
                                format_decimal(c.get("price_range", (None, None))[1], 4)
                            )
                        } for c in orderbook_analysis_raw.get("clusters", [])
                    ]
                 }
            else:
                 self.logger.debug("Skipping orderbook analysis due to missing orderbook data or invalid/missing current price.")
                 # Keep default NA values in analysis_result["orderbook_analysis"]


            # --- 4. Interpret Results ---
            # Interpretation uses the DataFrame with float indicators and Decimal price/levels/OB analysis
            if isinstance(current_price, Decimal) and current_price.is_finite():
                 interpretation = self._interpret_analysis(df_with_indicators, current_price, levels_decimal, orderbook_analysis_raw)
                 analysis_result["interpretation"] = interpretation
            else:
                 # Interpretation without price is limited, provide placeholder message
                 self.logger.warning("Skipping full interpretation due to missing or invalid current price.")
                 analysis_result["interpretation"]["summary"] = [Color.format("Interpretation skipped due to missing/invalid current price.", Color.YELLOW)]
                 # Keep signals as NA

        except Exception as e:
            self.logger.exception(f"Critical error during analysis pipeline for {self.symbol}: {e}")
            analysis_result["interpretation"]["summary"] = [Color.format(f"Critical Analysis Pipeline Error: {e}", Color.RED)]
            # Ensure signals are marked as NA in case of pipeline error
            analysis_result["interpretation"]["signals"] = {
                k: SignalState.NA.value for k in analysis_result["interpretation"].get("signals", {})
            }

        return analysis_result

    def format_analysis_output(self, analysis_result: dict) -> str:
        """
        Formats the analysis result dictionary into a human-readable string for logging/display.

        Args:
            analysis_result: The dictionary returned by the analyze() method.

        Returns:
            A formatted string summarizing the analysis.
        """
        symbol = analysis_result.get('symbol', 'N/A')
        timestamp_str = analysis_result.get('timestamp', 'N/A')
        ts_formatted = timestamp_str # Default to ISO string
        try:
            # Attempt to parse and format timestamp nicely in the application's timezone
            if timestamp_str != 'N/A':
                dt_obj = datetime.fromisoformat(timestamp_str).astimezone(APP_TIMEZONE)
                ts_formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, TypeError):
            pass # Fallback to raw ISO string if parsing fails

        price = analysis_result.get('current_price', 'N/A') # Already formatted
        # Use the inferred interval from the analysis result
        interval_display = analysis_result.get('kline_interval', 'N/A')

        # --- Header ---
        header = f"\n--- Analysis Report for {Color.format(symbol, Color.PURPLE)} --- ({ts_formatted})\n"
        info_line = f"{Color.format('Interval:', Color.BLUE)} {interval_display}    {Color.format('Current Price:', Color.BLUE)} ${price}\n"

        # --- Interpretation Summary ---
        interpretation = analysis_result.get("interpretation", {})
        summary_lines = interpretation.get("summary", []) # These lines already contain color formatting

        if summary_lines:
            # Join the formatted lines from the interpretation steps
            interpretation_block = "\n".join(summary_lines) + "\n"
        else:
            interpretation_block = Color.format("No interpretation summary available.", Color.YELLOW) + "\n"

        # --- Combine Parts ---
        output = header + info_line + interpretation_block
        return output


# --- Main Application Logic ---
async def run_analysis_loop(symbol: str, interval_config: str, client: BybitCCXTClient, analyzer: TradingAnalyzer, logger_instance: logging.Logger) -> None:
    """
    The main asynchronous loop that periodically fetches data, runs analysis,
    and logs the results for a specific symbol.

    Args:
        symbol: The trading symbol to analyze.
        interval_config: The user-friendly timeframe string (e.g., "15", "1h").
        client: The initialized BybitCCXTClient instance.
        analyzer: The initialized TradingAnalyzer instance.
        logger_instance: The logger instance for this symbol.
    """
    analysis_interval_sec = float(CONFIG.analysis_interval_seconds)
    kline_limit = CONFIG.kline_limit
    orderbook_limit = analyzer.orderbook_settings.limit

    if analysis_interval_sec < 10:
        logger_instance.warning(f"Analysis interval ({analysis_interval_sec}s) is very short. Ensure system and API rate limits can handle the load.")
    if not interval_config or interval_config not in VALID_INTERVALS:
         logger_instance.critical(f"Invalid interval '{interval_config}' passed to analysis loop. Stopping.")
         return

    logger_instance.info(f"Starting analysis loop for {symbol} with interval {interval_config}...")

    while True:
        cycle_start_time = time.monotonic()
        logger_instance.debug(f"--- Starting Analysis Cycle for {symbol} ---")

        analysis_result: Optional[dict] = None # Initialize for the cycle
        current_price: Optional[Decimal] = None
        df_klines: Optional[pd.DataFrame] = None
        orderbook: Optional[dict] = None

        try:
            # --- Fetch Data Concurrently ---
            # Create tasks for fetching data
            price_task = asyncio.create_task(client.fetch_current_price(symbol))
            klines_task = asyncio.create_task(client.fetch_klines(symbol, interval_config, kline_limit))
            # Only create orderbook task if limit is positive
            orderbook_task = None
            if orderbook_limit > 0:
                orderbook_task = asyncio.create_task(client.fetch_orderbook(symbol, orderbook_limit))
            else:
                 logger_instance.debug("Orderbook fetching disabled (limit <= 0).")

            # --- Wait for Data ---
            # Use gather to wait for all, handling potential errors
            tasks_to_gather = [price_task, klines_task]
            if orderbook_task:
                tasks_to_gather.append(orderbook_task)

            # Wait for all tasks to complete, return_exceptions=True prevents gather from stopping on first error
            results = await asyncio.gather(*tasks_to_gather, return_exceptions=True)

            # --- Process Results and Handle Errors ---
            price_result = results[0]
            klines_result = results[1]
            orderbook_result = results[2] if orderbook_task else None

            # Check price result
            if isinstance(price_result, Exception):
                logger_instance.error(f"Error fetching current price for {symbol}: {price_result}")
                current_price = None # Mark as unavailable
            elif price_result is None:
                 logger_instance.warning(f"Failed to fetch current price for {symbol} (returned None).")
                 current_price = None
            elif isinstance(price_result, Decimal) and price_result.is_finite():
                current_price = price_result
            else:
                 logger_instance.error(f"Fetched current price for {symbol} is invalid: {price_result}")
                 current_price = None


            # Check klines result
            if isinstance(klines_result, Exception):
                logger_instance.error(f"Error fetching klines for {symbol}: {klines_result}")
                df_klines = pd.DataFrame() # Use empty DataFrame to signal failure
            elif klines_result is None or not isinstance(klines_result, pd.DataFrame) or klines_result.empty:
                 logger_instance.error(f"Failed to fetch valid kline data for {symbol}. Skipping analysis cycle.")
                 df_klines = pd.DataFrame()
            else:
                df_klines = klines_result

            # Check orderbook result (only if task was created)
            if orderbook_task:
                if isinstance(orderbook_result, Exception):
                    logger_instance.error(f"Error fetching orderbook for {symbol}: {orderbook_result}")
                    orderbook = None
                elif orderbook_result is None:
                     logger_instance.warning(f"Failed to fetch orderbook for {symbol} (returned None).")
                     orderbook = None
                else:
                    orderbook = orderbook_result

            # --- Perform Analysis (only if klines are valid) ---
            if df_klines is not None and not df_klines.empty:
                # Pass the original klines DataFrame with Decimals
                analysis_result = analyzer.analyze(df_klines, current_price, orderbook)
            else:
                 logger_instance.error(f"Skipping analysis for {symbol} due to missing or invalid kline data.")
                 # Optionally sleep longer if klines consistently fail
                 await async_sleep_with_jitter(INITIAL_RETRY_DELAY_SECONDS)
                 continue # Skip to next cycle iteration


            # --- Output and Logging ---
            if analysis_result:
                output_string = analyzer.format_analysis_output(analysis_result)
                # Log the formatted output (includes colors for console via ColorStreamFormatter)
                # The file logger's SensitiveFormatter will strip colors.
                logger_instance.info(output_string)

                # Log structured JSON at DEBUG level (without colors)
                if logger_instance.isEnabledFor(logging.DEBUG):
                    try:
                        # Create a deep copy for logging to avoid modifying the original result
                        # Use json cycle detection indirectly via standard dumps/loads
                        # Convert Decimals to strings for JSON compatibility
                        def decimal_default(obj):
                            if isinstance(obj, Decimal):
                                return str(obj)
                            # Let the base default method raise the TypeError
                            return json.JSONEncoder().default(obj)

                        log_data_str = json.dumps(analysis_result, default=decimal_default, indent=2)
                        log_data = json.loads(log_data_str) # Convert back to dict structure

                        # Remove color codes from summary and pressure for clean JSON log
                        if 'interpretation' in log_data and 'summary' in log_data['interpretation']:
                            log_data['interpretation']['summary'] = [
                                SensitiveFormatter._color_code_regex.sub('', line)
                                for line in log_data['interpretation']['summary'] if isinstance(line, str)
                            ]
                        if 'orderbook_analysis' in log_data and 'pressure' in log_data['orderbook_analysis']:
                             pressure_val = log_data['orderbook_analysis']['pressure']
                             if isinstance(pressure_val, str):
                                 log_data['orderbook_analysis']['pressure'] = SensitiveFormatter._color_code_regex.sub('', pressure_val)

                        logger_instance.debug(f"Analysis Result JSON:\n{json.dumps(log_data, indent=2)}")

                    except (TypeError, ValueError) as json_err:
                        logger_instance.error(f"Error serializing analysis result for JSON debug logging: {json_err}")
                        # Log the problematic structure if possible (careful with large data)
                        # logger_instance.debug(f"Problematic analysis_result structure: {analysis_result}")
                    except Exception as log_json_err:
                         logger_instance.error(f"Unexpected error preparing analysis result for JSON debug logging: {log_json_err}")
            else:
                 # This case should be less likely now analysis runs even with missing price/OB
                 logger_instance.error("Analysis result was None or empty after execution. Skipping output.")


        # --- Specific Exception Handling for the Loop ---
        except ccxt.AuthenticationError as e:
            logger_instance.critical(Color.format(f"Authentication Error during analysis cycle: {e}. Check API Key/Secret. Stopping analysis for {symbol}.", Color.RED))
            break # Stop the loop for this symbol on auth error
        except ccxt.InvalidNonce as e:
            logger_instance.critical(Color.format(f"Invalid Nonce Error: {e}. Check system time sync with server. Stopping analysis for {symbol}.", Color.RED))
            break # Stop the loop on nonce error (usually requires intervention)
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            # These might occur if trading logic were added. Log as error but continue analysis.
            logger_instance.error(Color.format(f"Order-related CCXT error encountered: {e}. Check parameters or funds if trading.", Color.RED))
            # Continue the loop for analysis purposes
        except asyncio.CancelledError:
            logger_instance.info(f"Analysis task for {symbol} was cancelled.")
            break # Exit the loop cleanly if cancelled
        except Exception as e:
            logger_instance.exception(Color.format(f"An unexpected error occurred in the main analysis loop for {symbol}: {e}", Color.RED))
            # Add a longer sleep after unexpected errors to prevent rapid failure loops
            await async_sleep_with_jitter(10.0)

        # --- Sleep Management ---
        cycle_end_time = time.monotonic()
        elapsed_time = cycle_end_time - cycle_start_time
        sleep_duration = max(0, analysis_interval_sec - elapsed_time)

        logger_instance.debug(f"Analysis cycle for {symbol} took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s (+ jitter).")

        if elapsed_time > analysis_interval_sec:
            logger_instance.warning(
                f"Analysis cycle duration ({elapsed_time:.2f}s) exceeded configured interval ({analysis_interval_sec}s). "
                f"Consider increasing the interval or optimizing analysis."
            )
            # Avoid negative sleep, sleep for a minimal duration with jitter if overrun
            sleep_duration = 0.1

        await async_sleep_with_jitter(sleep_duration) # Use sleep with jitter

    logger_instance.info(f"Analysis loop stopped for {symbol}.")


async def main() -> None:
    """
    Main entry point for the application. Sets up logging, initializes the CCXT client,
    prompts user for symbol/interval, and starts the analysis loop. Handles graceful shutdown.
    """
    # Setup a temporary logger for initial setup steps until symbol is chosen
    init_logger = setup_logger("INIT")
    init_logger.info("Application starting...")
    init_logger.info(f"Using configuration from: {CONFIG_FILE_PATH}")
    init_logger.info(f"Log directory: {LOG_DIRECTORY}")
    init_logger.info(f"Timezone: {APP_TIMEZONE}")
    init_logger.info(f"API Environment: {'Testnet' if IS_TESTNET else 'Mainnet'}")

    # --- Initialize Client and Load Markets ---
    # Use a temporary client instance just for loading markets initially
    temp_client: Optional[BybitCCXTClient] = None
    markets_loaded = False
    loaded_markets_data: Optional[Dict[str, Any]] = None
    loaded_market_categories: Dict[str, str] = {}
    valid_symbols: List[str] = []

    try:
        temp_client = BybitCCXTClient(API_KEY, API_SECRET, IS_TESTNET, init_logger)
        markets_loaded = await temp_client.initialize_markets()

        if not markets_loaded or not temp_client.markets:
            init_logger.critical(Color.format("Failed to load markets during initialization. Cannot proceed. Exiting.", Color.RED))
            return # Exit early

        # Get valid symbols (where details are not None) and keep market data
        valid_symbols = [s for s, d in temp_client.markets.items() if d is not None]
        loaded_markets_data = temp_client.markets
        loaded_market_categories = temp_client.market_categories # Pass cached categories too
        init_logger.info(f"Market data loaded successfully ({len(valid_symbols)} valid markets found).")

    except Exception as client_init_err:
         init_logger.critical(Color.format(f"Failed during client initialization or market loading: {client_init_err}", Color.RED), exc_info=True)
         return # Exit early
    finally:
        if temp_client:
            await temp_client.close() # Close the temporary client connection


    # --- User Input for Symbol and Interval ---
    selected_symbol = ""
    while True:
        try:
            print(Color.format("\nPlease enter the symbol to analyze.", Color.BLUE))
            symbol_input = input(Color.format("Enter trading symbol (e.g., BTC/USDT): ", Color.YELLOW)).strip().upper()
            if not symbol_input: continue # Ask again if empty input

            # Attempt to standardize format (e.g., BTCUSDT -> BTC/USDT)
            potential_symbol = symbol_input
            if "/" not in symbol_input and len(symbol_input) > 3: # Basic check
                # Common quote currencies
                quotes = ['USDT', 'USD', 'USDC', 'BTC', 'ETH', 'EUR', 'GBP', 'DAI', 'BUSD'] # Add more if needed
                for quote in quotes:
                    if symbol_input.endswith(quote) and len(symbol_input) > len(quote): # Ensure base exists
                        base = symbol_input[:-len(quote)]
                        formatted = f"{base}/{quote}"
                        if formatted in valid_symbols:
                            print(Color.format(f"Assuming symbol: {formatted}", Color.CYAN))
                            potential_symbol = formatted
                            break # Found a potential match

            # Validate the (potentially formatted) symbol against loaded markets
            if potential_symbol in valid_symbols:
                selected_symbol = potential_symbol
                print(Color.format(f"Selected symbol: {selected_symbol}", Color.GREEN))
                break
            else:
                print(Color.format(f"Invalid or unsupported symbol: '{symbol_input}'.", Color.RED))
                # Suggest similar symbols based on simple substring matching (case-insensitive)
                input_lower = symbol_input.lower().replace('/', '')
                find_similar = [
                    s for s in valid_symbols
                    if input_lower in s.lower().replace('/', '')
                ][:10] # Limit suggestions
                if find_similar:
                    print(Color.format(f"Did you mean one of these? {', '.join(find_similar)}", Color.YELLOW))

        except EOFError:
            print(Color.format("\nInput stream closed. Exiting.", Color.YELLOW))
            logging.shutdown()
            return
        except KeyboardInterrupt:
             print(Color.format("\nOperation cancelled by user during input.", Color.YELLOW))
             logging.shutdown()
             return


    selected_interval_key = ""
    default_interval = CONFIG.indicator_settings.default_interval
    # Validate default interval from config
    if default_interval not in VALID_INTERVALS:
        print(Color.format(f"Warning: Default interval '{default_interval}' from config is invalid. Using '15'.", Color.YELLOW))
        default_interval = "15" # Fallback to a known valid interval

    while True:
        try:
            print(Color.format(f"\nEnter analysis timeframe.", Color.BLUE))
            interval_prompt = (f"Available: [{', '.join(VALID_INTERVALS)}]\n"
                               f"Enter timeframe (default: {default_interval}): ")
            interval_input = input(Color.format(interval_prompt, Color.YELLOW)).strip()

            if not interval_input:
                selected_interval_key = default_interval
                print(Color.format(f"Using default interval: {selected_interval_key}", Color.CYAN))
                break

            # Check if input is directly in our valid keys ('1', '15', 'D', etc.)
            if interval_input in VALID_INTERVALS:
                selected_interval_key = interval_input
                print(Color.format(f"Selected interval: {selected_interval_key}", Color.GREEN))
                break

            # Check if input is a standard CCXT interval ('1m', '1h', '1d', etc.)
            # Allow case-insensitivity for CCXT intervals (e.g., '1d' or '1D')
            interval_input_lower = interval_input.lower()
            # Handle '1M' specifically as it's uppercase in CCXT map
            ccxt_interval_to_check = '1M' if interval_input == '1M' else interval_input_lower

            if ccxt_interval_to_check in REVERSE_CCXT_INTERVAL_MAP:
                selected_interval_key = REVERSE_CCXT_INTERVAL_MAP[ccxt_interval_to_check]
                print(Color.format(f"Using interval {selected_interval_key} (mapped from {interval_input})", Color.CYAN))
                break

            print(Color.format(f"Invalid interval '{interval_input}'. Please choose from the list or use standard CCXT format (e.g., 1h, 4h, 1d).", Color.RED))

        except EOFError:
            print(Color.format("\nInput stream closed. Exiting.", Color.YELLOW))
            logging.shutdown()
            return
        except KeyboardInterrupt:
             print(Color.format("\nOperation cancelled by user during input.", Color.YELLOW))
             logging.shutdown()
             return

    # --- Setup Main Components for the Selected Symbol ---
    main_logger = setup_logger(selected_symbol) # Setup logger specific to the chosen symbol
    main_logger.info(f"Logger initialized for symbol {selected_symbol}.")

    # Create the main client instance, passing the already loaded market data
    client = BybitCCXTClient(API_KEY, API_SECRET, IS_TESTNET, main_logger)
    client.markets = loaded_markets_data # Assign pre-loaded markets
    client.market_categories = loaded_market_categories # Assign pre-cached categories
    main_logger.info(f"CCXT Client initialized for {selected_symbol}. Markets assigned.")

    analyzer = TradingAnalyzer(CONFIG, main_logger, selected_symbol)
    main_logger.info("Trading Analyzer initialized.")

    # Log final setup details
    ccxt_interval = CCXT_INTERVAL_MAP.get(selected_interval_key, "N/A")
    market_type = client.get_market_category(selected_symbol)
    api_url = client.exchange.urls.get('api', 'URL not available')
    if isinstance(api_url, dict): api_url = api_url.get('public', api_url.get('private', 'Nested URL not found'))

    main_logger.info(f"--- Starting Analysis ---")
    main_logger.info(f"Symbol: {Color.format(selected_symbol, Color.PURPLE)} ({market_type.capitalize()})")
    main_logger.info(f"Interval: {Color.format(selected_interval_key, Color.PURPLE)} (CCXT: {ccxt_interval})")
    main_logger.info(f"API Env: {Color.format(API_ENV.upper(), Color.YELLOW)} (URL: {api_url})")
    main_logger.info(f"Loop Interval: {CONFIG.analysis_interval_seconds} seconds")
    main_logger.info(f"Kline Limit: {CONFIG.kline_limit} candles")
    main_logger.info(f"Orderbook Limit: {CONFIG.orderbook_settings.limit} levels")
    main_logger.info(f"Timezone: {APP_TIMEZONE}")


    # --- Run Analysis Loop and Handle Shutdown ---
    main_task = None
    try:
        # Create and run the main analysis task
        main_task = asyncio.create_task(
            run_analysis_loop(selected_symbol, selected_interval_key, client, analyzer, main_logger)
        )
        await main_task # Wait for the loop to complete or be cancelled

    except KeyboardInterrupt:
        main_logger.info(Color.format("\nCtrl+C detected. Stopping analysis loop...", Color.YELLOW))
        if main_task and not main_task.done():
            main_task.cancel()
            # Wait briefly for cancellation to propagate
            try:
                await asyncio.wait_for(main_task, timeout=2.0) # Wait max 2s for task to finish cancelling
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass # Expected exceptions during cancellation
    except asyncio.CancelledError:
        main_logger.info(Color.format("Main analysis task was cancelled externally.", Color.YELLOW))
        # Expected during shutdown if cancelled from outside
    except Exception as e:
        main_logger.critical(Color.format(f"A critical error occurred during main execution: {e}", Color.RED), exc_info=True)
        if main_task and not main_task.done():
            main_task.cancel()
            try:
                await asyncio.wait_for(main_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
    finally:
        main_logger.info("Initiating shutdown sequence...")
        # Ensure the client connection is closed
        if 'client' in locals() and client and hasattr(client, 'exchange') and client.exchange: # Check client and exchange exist
            await client.close()
        main_logger.info("Application finished.")

        # Flush and close logger handlers gracefully
        # Get handlers associated with this specific logger instance
        if 'main_logger' in locals() and main_logger:
            handlers = main_logger.handlers[:]
            for handler in handlers:
                try:
                    handler.flush()
                    handler.close()
                    main_logger.removeHandler(handler)
                except Exception as e:
                    # Use print as logger might be unreliable during shutdown
                    print(f"Error closing handler {handler}: {e}")

        # Attempt to shutdown the entire logging system
        logging.shutdown()


if __name__ == "__main__":
    # Set numpy print options to suppress scientific notation for easier reading if needed
    # np.set_printoptions(suppress=True, precision=8)

    # Set pandas display options if desired
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)

    try:
        # Run the main asynchronous function
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C if it occurs *before* the main async loop starts
        # or *after* it exits but before the script terminates.
        print(f"\n{Color.YELLOW.value}Process interrupted by user. Exiting gracefully.{Color.RESET.value}")
    except Exception as e:
        # Catch any other unexpected top-level errors during script execution
        print(f"\n{Color.RED.value}A critical top-level error occurred: {e}{Color.RESET.value}")
        traceback.print_exc() # Print detailed traceback for top-level errors
    finally:
        # Ensure colorama reset is called on exit
        print(Style.RESET_ALL)

