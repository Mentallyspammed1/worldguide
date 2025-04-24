```python
# -*- coding: utf-8 -*-
"""
Pyrmethus Volumatic Bot - Automated Trading Bot using Volumatic Trend and Order Blocks Strategy.
Version: 1.4.1+enhancements
"""

# --- Standard Library Imports ---
import hashlib
import hmac
import json
import logging
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# --- Timezone Handling ---
# Attempt to import the standard library's zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # Fallback for older Python versions or if tzdata is not installed
    print("Warning: 'zoneinfo' module not found. Falling back to basic UTC implementation. "
          "For accurate local time logging, ensure Python 3.9+ and install 'tzdata' (`pip install tzdata`).")

    # Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface
    class ZoneInfo: # type: ignore [no-redef]
        """Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface."""
        _key = "UTC" # Class attribute, always UTC

        def __init__(self, key: str):
            """Initializes the fallback ZoneInfo. Always uses UTC."""
            if key.upper() != "UTC":
                print(f"Warning: Fallback ZoneInfo initialized with key '{key}', but will always use UTC.")
            # Store the requested key, though internally we always use UTC
            self._requested_key = key

        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            """Attaches UTC timezone info to a datetime object. Returns None if input is None."""
            return dt.replace(tzinfo=timezone.utc) if dt else None

        def fromutc(self, dt: datetime) -> datetime:
            """Converts a UTC datetime to this timezone (which is UTC)."""
            if dt.tzinfo is None:
                # Assume naive datetime is UTC for conversion, though standard ZoneInfo might raise error
                print("Warning: Calling fromutc on naive datetime in UTC fallback.")
                return dt.replace(tzinfo=timezone.utc)
            # If already timezone-aware, ensure it's UTC
            return dt.astimezone(timezone.utc)

        def utcoffset(self, dt: Optional[datetime]) -> timedelta:
            """Returns the UTC offset (always zero for UTC)."""
            return timedelta(0)

        def dst(self, dt: Optional[datetime]) -> timedelta:
            """Returns the DST offset (always zero for UTC)."""
            return timedelta(0)

        def tzname(self, dt: Optional[datetime]) -> str:
            """Returns the timezone name (always 'UTC')."""
            return "UTC"

        def __repr__(self) -> str:
            return f"ZoneInfo(key='{self._requested_key}') [Fallback: Always UTC]"

        def __str__(self) -> str:
            return self._key

    class ZoneInfoNotFoundError(Exception): # type: ignore [no-redef]
        """Exception raised when a timezone is not found (fallback definition)."""
        pass

# --- Third-Party Library Imports ---
# Grouped by general purpose
# Data Handling & Numerics
import numpy as np
import pandas as pd
import pandas_ta as ta  # Technical Analysis library

# API & Networking
import requests         # For HTTP requests (often used by ccxt)
import ccxt             # Crypto Exchange Trading Library

# Utilities
from colorama import Fore, Style, init as colorama_init # Colored console output
from dotenv import load_dotenv                        # Load environment variables

# --- Initial Setup ---
# Set Decimal precision globally for accurate financial calculations
getcontext().prec = 28
# Initialize Colorama for cross-platform colored output (reset colors after each print)
colorama_init(autoreset=True)
# Load environment variables from a .env file if it exists
load_dotenv()

# --- Constants ---
BOT_VERSION = "1.4.1+enhancements"

# --- API Credentials (Loaded from Environment) ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print directly here as logger might not be fully set up
    print(f"{Fore.RED}{Style.BRIGHT}FATAL ERROR: BYBIT_API_KEY and/or BYBIT_API_SECRET environment variables are missing.{Style.RESET_ALL}")
    print(f"{Fore.RED}Please ensure they are set in your system environment or in a '.env' file in the bot's directory.{Style.RESET_ALL}")
    print(f"{Fore.RED}Exiting due to missing credentials.{Style.RESET_ALL}")
    sys.exit(1)

# --- Configuration File & Logging ---
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"

# --- Timezone Configuration ---
DEFAULT_TIMEZONE_STR: str = "America/Chicago" # Default timezone if not set in .env or config
# Prioritize TIMEZONE from .env, fallback to default
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    print(f"{Fore.RED}{Style.BRIGHT}Warning: Timezone '{TIMEZONE_STR}' not found using 'zoneinfo'. Falling back to UTC.{Style.RESET_ALL}")
    print(f"{Fore.RED}Ensure 'tzdata' is installed (`pip install tzdata`) for non-UTC timezones.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update the string representation
except Exception as tz_err:
    print(f"{Fore.RED}{Style.BRIGHT}Warning: An error occurred initializing timezone '{TIMEZONE_STR}': {tz_err}. Falling back to UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update the string representation

# --- API & Timing Constants ---
# These can be overridden by config.json values where applicable
MAX_API_RETRIES: int = 3           # Max number of retries for most failed API calls
RETRY_DELAY_SECONDS: int = 5       # Initial delay between retries (often increased exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Delay after placing order to confirm position status (allows exchange processing)
LOOP_DELAY_SECONDS: int = 15       # Base delay between main loop cycles (per symbol)
BYBIT_API_KLINE_LIMIT: int = 1000  # Max klines per Bybit V5 API request (important for fetch_klines_ccxt)

# --- Data & Strategy Constants ---
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Supported intervals in config.json
CCXT_INTERVAL_MAP: Dict[str, str] = { # Map config intervals to CCXT standard timeframes
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT: int = 750     # Default number of klines to fetch if not specified in config
MAX_DF_LEN: int = 2000             # Maximum length of DataFrame to keep in memory (prevents memory bloat)

# Default Volumatic Trend (VT) parameters (can be overridden by config.json)
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Note: This param wasn't used in the original placeholder logic - keep or remove?

# Default Order Block (OB) parameters (can be overridden by config.json)
DEFAULT_OB_SOURCE: str = "Wicks"    # "Wicks" or "Body"
DEFAULT_PH_LEFT: int = 10           # Pivot High lookback/forward periods
DEFAULT_PH_RIGHT: int = 10
DEFAULT_PL_LEFT: int = 10           # Pivot Low lookback/forward periods
DEFAULT_PL_RIGHT: int = 10
DEFAULT_OB_EXTEND: bool = True      # Extend OB boxes until violated
DEFAULT_OB_MAX_BOXES: int = 50      # Max number of active OBs to track per side

# --- Trading Constants ---
QUOTE_CURRENCY: str = "USDT"       # Default quote currency (will be updated by loaded config)

# --- UI Constants (Colorama Foregrounds and Styles) ---
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN         # Duplicate of NEON_BLUE, kept for compatibility
RESET: str = Style.RESET_ALL       # Resets all styles and colors
BRIGHT: str = Style.BRIGHT         # Makes text brighter
DIM: str = Style.DIM               # Makes text dimmer

# --- Create Log Directory ---
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e:
    print(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create log directory '{LOG_DIRECTORY}': {e}{RESET}")
    print(f"{NEON_RED}Please check permissions and ensure the path is valid.{RESET}")
    sys.exit(1)

# --- Global State ---
_shutdown_requested: bool = False # Flag for graceful shutdown triggered by signal handler

# --- Type Definitions (Enhanced with Docstrings) ---
class OrderBlock(TypedDict):
    """Represents an identified Order Block."""
    id: str                 # Unique identifier (e.g., "BULL_1678886400000")
    type: str               # "BULL" or "BEAR"
    timestamp: pd.Timestamp # Timestamp of the candle defining the block (UTC)
    top: Decimal            # Top price level of the block
    bottom: Decimal         # Bottom price level of the block
    active: bool            # Is the block currently considered active (not violated, not expired)?
    violated: bool          # Has the block been violated by price action?
    violation_ts: Optional[pd.Timestamp] # Timestamp when the violation occurred (UTC)
    extended_to_ts: Optional[pd.Timestamp] # Timestamp the box is currently visually extended to (UTC)

class StrategyAnalysisResults(TypedDict):
    """Results from the strategy analysis performed on a DataFrame."""
    dataframe: pd.DataFrame # The analyzed DataFrame including indicator columns
    last_close: Decimal     # The closing price of the most recent candle
    current_trend_up: Optional[bool] # Current trend direction (True=Up, False=Down, None=Undetermined/Sideways)
    trend_just_changed: bool # Did the trend direction change on the last candle?
    active_bull_boxes: List[OrderBlock] # List of currently active bullish Order Blocks
    active_bear_boxes: List[OrderBlock] # List of currently active bearish Order Blocks
    vol_norm_int: Optional[int] # Normalized volume indicator value (0-100), if used by strategy
    atr: Optional[Decimal]    # Current Average True Range value
    upper_band: Optional[Decimal] # Upper band value (e.g., from Volumatic Trend or other indicator)
    lower_band: Optional[Decimal] # Lower band value (e.g., from Volumatic Trend or other indicator)

class MarketInfo(TypedDict):
    """
    Standardized market information derived from ccxt `market` structure,
    enhanced with convenience fields and Decimal types for precision.
    """
    # --- Standard CCXT Fields (subset, may vary slightly by exchange) ---
    id: str                 # Exchange-specific market ID (e.g., 'BTCUSDT')
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    base: str               # Base currency (e.g., 'BTC')
    quote: str              # Quote currency (e.g., 'USDT')
    settle: Optional[str]   # Settle currency (usually for futures/swaps, e.g., 'USDT' or 'BTC')
    baseId: str             # Exchange-specific base currency ID
    quoteId: str            # Exchange-specific quote currency ID
    settleId: Optional[str] # Exchange-specific settle currency ID
    type: str               # Market type ('spot', 'swap', 'future', 'option', etc.)
    spot: bool              # Is it a spot market?
    margin: bool            # Is margin trading allowed? (May overlap with futures/swaps)
    swap: bool              # Is it a perpetual swap?
    future: bool            # Is it a dated future?
    option: bool            # Is it an options market?
    active: Optional[bool]  # Is the market currently active/tradeable? (Can be None if not provided)
    contract: bool          # Is it a contract (swap, future, option)? (Convenience flag)
    linear: Optional[bool]  # Linear contract? (Quote currency settlement)
    inverse: Optional[bool] # Inverse contract? (Base currency settlement)
    quanto: Optional[bool]  # Quanto contract? (Settled in a third currency)
    taker: float            # Taker fee rate (as a fraction, e.g., 0.00075)
    maker: float            # Maker fee rate (as a fraction, e.g., 0.0002)
    contractSize: Optional[Any] # Size of one contract (often 1 for linear, value in USD for inverse)
    expiry: Optional[int]   # Timestamp (ms) of future/option expiry
    expiryDatetime: Optional[str] # ISO 8601 datetime string of expiry
    strike: Optional[float] # Strike price for options
    optionType: Optional[str] # 'call' or 'put' for options
    precision: Dict[str, Any] # Price and amount precision rules (e.g., {'price': 0.01, 'amount': 0.001})
    limits: Dict[str, Any]    # Order size and cost limits (e.g., {'amount': {'min': 0.001, 'max': 100}})
    info: Dict[str, Any]      # Raw market data dictionary directly from the exchange API response
    # --- Added/Derived Fields for Convenience and Precision ---
    is_contract: bool         # Enhanced convenience flag: True if swap, future, or option
    is_linear: bool           # Enhanced convenience flag: True if linear contract (and is a contract)
    is_inverse: bool          # Enhanced convenience flag: True if inverse contract (and is a contract)
    contract_type_str: str    # User-friendly string: "Spot", "Linear", "Inverse", "Option", or "Unknown"
    min_amount_decimal: Optional[Decimal] # Minimum order size (in base currency/contracts) as Decimal
    max_amount_decimal: Optional[Decimal] # Maximum order size (in base currency/contracts) as Decimal
    min_cost_decimal: Optional[Decimal]   # Minimum order cost (in quote currency) as Decimal
    max_cost_decimal: Optional[Decimal]   # Maximum order cost (in quote currency) as Decimal
    amount_precision_step_decimal: Optional[Decimal] # Smallest increment for order amount as Decimal
    price_precision_step_decimal: Optional[Decimal]  # Smallest increment for order price as Decimal
    contract_size_decimal: Decimal # Contract size as Decimal (defaults to 1 if not applicable/found)

class PositionInfo(TypedDict):
    """
    Standardized position information derived from ccxt `position` structure,
    enhanced with Decimal types, protection order fields, and bot-specific state tracking.
    """
    # --- Standard CCXT Fields (subset, may vary slightly by exchange/method) ---
    id: Optional[str]       # Position ID (exchange-specific, may not always be present)
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    timestamp: Optional[int] # Position creation/update timestamp (ms)
    datetime: Optional[str]  # ISO 8601 datetime string of timestamp
    contracts: Optional[float] # Number of contracts (use size_decimal derived field instead for precision)
    contractSize: Optional[Any] # Size of one contract for this position
    side: Optional[str]      # 'long' or 'short'
    notional: Optional[Any]  # Position value in quote currency (use Decimal derived field)
    leverage: Optional[Any]  # Position leverage (use Decimal derived field)
    unrealizedPnl: Optional[Any] # Unrealized profit/loss (use Decimal derived field)
    realizedPnl: Optional[Any]   # Realized profit/loss (use Decimal derived field)
    collateral: Optional[Any]    # Margin used for the position (use Decimal derived field)
    entryPrice: Optional[Any]    # Average entry price (use Decimal derived field)
    markPrice: Optional[Any]     # Current mark price (use Decimal derived field)
    liquidationPrice: Optional[Any] # Estimated liquidation price (use Decimal derived field)
    marginMode: Optional[str]    # 'isolated' or 'cross'
    hedged: Optional[bool]       # Is hedging enabled for this position? (Less common now)
    maintenanceMargin: Optional[Any] # Maintenance margin required (use Decimal derived field)
    maintenanceMarginPercentage: Optional[float] # Maintenance margin rate
    initialMargin: Optional[Any] # Initial margin used (use Decimal derived field)
    initialMarginPercentage: Optional[float] # Initial margin rate
    marginRatio: Optional[float] # Margin ratio (health indicator)
    lastUpdateTimestamp: Optional[int] # Timestamp of last position update from exchange (ms)
    info: Dict[str, Any]         # Raw position data dictionary directly from the exchange API response
    # --- Added/Derived Fields for Convenience and Precision ---
    size_decimal: Decimal        # Position size as Decimal (positive for long, negative for short)
    entryPrice_decimal: Optional[Decimal] # Entry price as Decimal
    markPrice_decimal: Optional[Decimal] # Mark price as Decimal
    liquidationPrice_decimal: Optional[Decimal] # Liquidation price as Decimal
    leverage_decimal: Optional[Decimal] # Leverage as Decimal
    unrealizedPnl_decimal: Optional[Decimal] # Unrealized PnL as Decimal
    notional_decimal: Optional[Decimal] # Notional value as Decimal
    collateral_decimal: Optional[Decimal] # Collateral as Decimal
    initialMargin_decimal: Optional[Decimal] # Initial margin as Decimal
    maintenanceMargin_decimal: Optional[Decimal] # Maintenance margin as Decimal
    stopLossPrice: Optional[str] # Current stop loss price (as string from exchange, may be '0' if not set)
    takeProfitPrice: Optional[str] # Current take profit price (as string from exchange, may be '0' if not set)
    trailingStopLoss: Optional[str] # Current trailing stop distance/price (as string, interpretation depends on exchange)
    tslActivationPrice: Optional[str] # Trailing stop activation price (as string, if available/set)
    # --- Bot State Tracking (Managed internally by the bot, reflects *bot's* actions/knowledge) ---
    be_activated: bool           # Has the break-even logic been triggered *by the bot* for this position?
    tsl_activated: bool          # Has the trailing stop loss been activated *by the bot* or detected as active on the exchange?

class SignalResult(TypedDict):
    """Result of the signal generation process."""
    signal: str              # "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
    reason: str              # Explanation for the generated signal
    initial_sl: Optional[Decimal] # Calculated initial stop loss price for a new entry signal
    initial_tp: Optional[Decimal] # Calculated initial take profit price for a new entry signal

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom log formatter to redact sensitive API keys and secrets."""
    # Placeholders must be distinct and unlikely to occur naturally in logs
    _api_key_placeholder = "***BYBIT_API_KEY_REDACTED***"
    _api_secret_placeholder = "***BYBIT_API_SECRET_REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting configured API key and secret."""
        # Ensure message is formatted first by the base class
        formatted_msg = super().format(record)
        redacted_msg = formatted_msg

        # Only perform redaction if keys are actually set and are non-empty strings
        key = API_KEY
        secret = API_SECRET
        try:
            if key and isinstance(key, str) and len(key) > 4: # Check length to avoid redacting short common strings
                # Use regex for safer replacement (avoid partial matches if keys are substrings of other words)
                # Word boundaries (\b) help, but keys might not have them. Simple replace is often okay here.
                redacted_msg = redacted_msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and len(secret) > 4:
                redacted_msg = redacted_msg.replace(secret, self._api_secret_placeholder)
        except Exception as e:
            # Avoid crashing the application if redaction fails unexpectedly
            print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
            # Return the original formatted message in case of error
            return formatted_msg
        return redacted_msg

class NeonConsoleFormatter(SensitiveFormatter):
    """
    Custom log formatter for console output. Displays messages with level-specific
    colors, includes the logger name, and uses the globally configured local timezone
    for timestamps. Inherits redaction logic from SensitiveFormatter.
    """
    _level_colors = {
        logging.DEBUG: DIM + NEON_CYAN,    # Dim Cyan for Debug
        logging.INFO: NEON_BLUE,           # Bright Cyan for Info
        logging.WARNING: NEON_YELLOW,      # Bright Yellow for Warning
        logging.ERROR: NEON_RED,           # Bright Red for Error
        logging.CRITICAL: BRIGHT + NEON_RED # Bright Red and Bold for Critical
    }
    _default_color = NEON_BLUE
    _log_format = (
        f"{DIM}%(asctime)s{RESET} {NEON_PURPLE}[%(name)s]{RESET} "
        f"%(levelcolor)s%(levelname)-8s{RESET} %(message)s"
    )
    _date_format = '%Y-%m-%d %H:%M:%S' # Include date for clarity on console

    def __init__(self, **kwargs):
        """Initializes the formatter, setting the format string and date format."""
        # Note: We don't pass fmt here, as we build it dynamically in format()
        super().__init__(fmt=self._log_format, datefmt=self._date_format, **kwargs)
        # Ensure the global TIMEZONE is used for timestamp conversion
        self.converter = lambda timestamp, _: datetime.fromtimestamp(timestamp, tz=TIMEZONE).timetuple()

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record with colors, local timestamp, and applies redaction."""
        # Assign the color based on level before formatting
        record.levelcolor = self._level_colors.get(record.levelno, self._default_color)

        # Format the message using the base class (which applies redaction via SensitiveFormatter)
        # This now uses the dynamically set `converter` for local time.
        formatted_and_redacted_msg = super().format(record)

        # No need for manual redaction here, SensitiveFormatter handles it.
        return formatted_and_redacted_msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger instance with standardized handlers:
    1. Rotating File Handler: Logs all DEBUG level messages and above to a file, using UTC timestamps.
    2. Console Handler: Logs INFO level messages and above (or level from CONSOLE_LOG_LEVEL env var)
       to the console, using local timestamps and colors.

    Args:
        name: The name for the logger (e.g., 'init', 'main', 'BTC/USDT'). Used in log messages and filename.

    Returns:
        A configured logging.Logger instance.
    """
    # Sanitize the logger name for use in filenames (replace slashes, colons, etc.)
    safe_filename_part = re.sub(r'[^\w\-.]', '_', name) # Allow word chars, hyphen, dot; replace others with underscore
    logger_name = f"pyrmethus.{safe_filename_part}" # Use dot notation for potential hierarchical logging
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if the logger was already configured
    if logger.hasHandlers():
        logger.debug(f"Logger '{logger_name}' already configured. Skipping setup.")
        return logger

    logger.setLevel(logging.DEBUG) # Set the lowest level to capture all messages; handlers will filter.

    # --- File Handler (UTC Timestamps) ---
    try:
        # Rotate log file when it reaches 10MB, keep up to 5 backup files.
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.DEBUG) # Log everything to the file

        # Use UTC time for file logs for consistency across different environments.
        # Include milliseconds and line number for detailed debugging.
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC [%(name)s:%(lineno)d] %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter.converter = time.gmtime # Explicitly use UTC for timestamps in file logs
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    except Exception as e:
        # Log setup errors should be visible on the console, even if console handler fails later
        print(f"{NEON_RED}{BRIGHT}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler (Local Timestamps & Colors) ---
    try:
        sh = logging.StreamHandler(sys.stdout)

        # Determine console log level from environment variable or default to INFO
        console_log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, console_log_level_str, logging.INFO)
        if not isinstance(log_level, int): # Fallback if getattr fails or returns non-level
            print(f"{NEON_YELLOW}Warning: Invalid CONSOLE_LOG_LEVEL '{console_log_level_str}'. Defaulting to INFO.{RESET}")
            log_level = logging.INFO
        sh.setLevel(log_level)

        # Use the custom NeonConsoleFormatter for colors and local time formatting.
        # No format string needed here as the formatter class defines it.
        console_formatter = NeonConsoleFormatter()
        sh.setFormatter(console_formatter)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}{BRIGHT}Error setting up console logger: {e}{RESET}")

    # Prevent messages from propagating to the root logger if handlers are added here
    logger.propagate = False

    logger.debug(f"Logger '{logger_name}' initialized. File: '{log_filename}', Console Level: {logging.getLevelName(sh.level)}")
    return logger

# --- Initial Logger Setup ---
# Used for messages during the initialization phase before symbol-specific loggers are created.
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}===== Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing ====={Style.RESET_ALL}")
init_logger.info(f"Using Timezone for Console Logs: {TIMEZONE_STR} ({TIMEZONE})")
init_logger.debug(f"Decimal Precision Set To: {getcontext().prec}")
# Remind user about dependencies
init_logger.debug("Ensure required packages are installed: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (recommended)")

# --- Configuration Loading & Validation ---

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures all keys from default_config exist in config.
    Adds missing keys with their default values and logs these additions.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.
        parent_key: Internal tracking string for nested key paths (used for logging).

    Returns:
        A tuple containing:
        - The updated configuration dictionary (potentially modified).
        - A boolean indicating if any changes (key additions) were made.
    """
    updated_config = config.copy() # Work on a copy to avoid modifying the original dict directly
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing entirely
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' with default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Key exists and both default and loaded values are dictionaries, so recurse
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config # Update the nested dictionary
                changed = True
        # Optional: Add basic type checking here if needed, but robust validation is done later.
        # Example: Check if the type of the loaded value matches the type of the default value.
        # elif type(default_value) is not type(updated_config.get(key)) and default_value is not None:
        #     init_logger.debug(f"Config Note: Type mismatch for '{full_key_path}'. Expected {type(default_value).__name__}, got {type(updated_config.get(key)).__name__}. Validation will handle.")
        #     pass # Let validation handle corrections or report warnings

    return updated_config, changed

def _validate_and_correct_numeric(
    cfg_level: Dict[str, Any],
    default_level: Dict[str, Any],
    leaf_key: str,
    key_path: str,
    min_val: Union[Decimal, int, float],
    max_val: Union[Decimal, int, float],
    is_strict_min: bool = False,
    is_int: bool = False,
    allow_zero: bool = False
) -> bool:
    """
    Validates a numeric value within a configuration dictionary level.
    Corrects type (e.g., str -> int/float, float -> int if required) and clamps
    the value to the specified [min_val, max_val] range if necessary.
    Uses the corresponding default value as a fallback for invalid types or out-of-range values.

    Args:
        cfg_level: The current dictionary level of the loaded config being validated.
        default_level: The corresponding dictionary level in the default config structure.
        leaf_key: The specific key (string) to validate within the current level.
        key_path: The full dot-notation path to the key (e.g., 'protection.max_loss') for logging.
        min_val: The minimum allowed value (inclusive unless is_strict_min is True).
        max_val: The maximum allowed value (inclusive).
        is_strict_min: If True, the value must be strictly greater than min_val (>).
        is_int: If True, the value should be validated and potentially corrected to an integer.
        allow_zero: If True, zero is considered a valid value even if outside the min/max range (but still type-validated).

    Returns:
        True if the value was corrected or replaced with the default, False otherwise.
    """
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key) # Get default for fallback
    corrected = False
    final_val = original_val # Assume no change initially

    try:
        # 1. Initial Type Check & Rejection of Boolean
        if isinstance(original_val, bool):
             raise TypeError("Boolean type is not valid for numeric configuration.")

        # 2. Attempt Conversion to Decimal for Robust Validation
        # Convert to string first to handle potential float inaccuracies and numeric strings "1.0"
        try:
            num_val = Decimal(str(original_val))
        except (InvalidOperation, TypeError, ValueError):
             # Handle cases where conversion to string or Decimal fails (e.g., None, empty string, non-numeric string)
             raise TypeError(f"Value '{repr(original_val)}' cannot be converted to a number.")

        # 3. Check for Non-Finite Values (NaN, Infinity)
        if not num_val.is_finite():
            raise ValueError("Non-finite value (NaN or Infinity) is not allowed.")

        # Convert range limits to Decimal for comparison
        min_dec = Decimal(str(min_val))
        max_dec = Decimal(str(max_val))

        # 4. Range Check
        is_zero = num_val.is_zero()
        min_check_passed = (num_val > min_dec) if is_strict_min else (num_val >= min_dec)
        range_check_passed = min_check_passed and (num_val <= max_dec)

        if not range_check_passed and not (allow_zero and is_zero):
            # Value is outside the allowed range and is not an allowed zero
            range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
            allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
            raise ValueError(f"Value {num_val.normalize()} is outside the allowed range {allowed_str}.")

        # 5. Type Check and Correction (Integer or Float)
        needs_type_correction = False
        target_type = int if is_int else float # Expected final type

        if is_int:
            # Check if the Decimal value has a fractional part or if original type wasn't int
            if num_val % 1 != 0: # Has fractional part
                needs_type_correction = True
                # Truncate towards zero for integer conversion
                final_val = int(num_val.to_integral_value(rounding=ROUND_DOWN))
                init_logger.info(f"{NEON_YELLOW}Config Update: Truncated fractional part for integer key '{key_path}' from {repr(original_val)} to {repr(final_val)}.{RESET}")
                # Re-check range after truncation
                final_dec = Decimal(final_val)
                min_check_passed = (final_dec > min_dec) if is_strict_min else (final_dec >= min_dec)
                range_check_passed = min_check_passed and (final_dec <= max_dec)
                if not range_check_passed and not (allow_zero and final_dec.is_zero()):
                    # If truncation pushes it out of range, it's an error -> use default
                    range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                    allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
                    raise ValueError(f"Value truncated to {final_val}, which is outside the allowed range {allowed_str}.")
            elif not isinstance(original_val, int): # Value is whole number, but not stored as int (e.g., 10.0 or "10")
                 needs_type_correction = True
                 final_val = int(num_val)
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for integer key '{key_path}' from {type(original_val).__name__} to int (value: {repr(final_val)}).{RESET}")
            else:
                 final_val = int(num_val) # Already conceptually an integer and stored as int

        else: # Expecting float
            # Check if original type wasn't float or int (int is acceptable for float fields)
            if not isinstance(original_val, (float, int)):
                 needs_type_correction = True
                 final_val = float(num_val) # Convert validated Decimal to float
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for float key '{key_path}' from {type(original_val).__name__} to float (value: {repr(final_val)}).{RESET}")
            # Check if float representation needs update due to precision or Decimal conversion
            # (e.g., original was "0.1", stored as float(Decimal("0.1")))
            elif isinstance(original_val, float):
                 converted_float = float(num_val)
                 # Use a small tolerance for float comparison
                 if abs(original_val - converted_float) > 1e-9:
                      needs_type_correction = True
                      final_val = converted_float
                      init_logger.info(f"{NEON_YELLOW}Config Update: Adjusted float value for '{key_path}' due to precision from {repr(original_val)} to {repr(final_val)}.{RESET}")
                 else:
                      final_val = converted_float # Keep as float
            elif isinstance(original_val, int):
                 # Convert int to float if the field expects float
                 final_val = float(original_val)
                 # Optionally log this as a type correction
                 # needs_type_correction = True
                 # init_logger.info(f"Config Update: Converted integer value for float key '{key_path}' to float ({repr(final_val)}).")
            else: # Already a float and close enough
                 final_val = float(num_val)

        # Mark as corrected if type was changed
        if needs_type_correction:
            corrected = True

    except (ValueError, InvalidOperation, TypeError, AssertionError) as e:
        # Handle validation errors (range, type conversion, non-finite, etc.)
        range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
        if allow_zero: range_str += " or 0"
        expected_type = 'integer' if is_int else 'float'
        init_logger.warning(
            f"{NEON_YELLOW}Config Validation Warning: Invalid value for '{key_path}'.\n"
            f"  Provided: {repr(original_val)} (Type: {type(original_val).__name__})\n"
            f"  Problem: {e}\n"
            f"  Expected: {expected_type} in range {range_str}\n"
            f"  Using default value: {repr(default_val)}{RESET}"
        )
        final_val = default_val # Use the default value on any validation error
        corrected = True

    # Update the configuration dictionary if a correction was made
    if corrected:
        cfg_level[leaf_key] = final_val

    return corrected


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    - Creates a default config file if it doesn't exist.
    - Ensures all necessary keys are present, adding defaults for missing keys.
    - Validates parameter types and ranges, correcting or using defaults if invalid.
    - Saves the updated (corrected) configuration back to the file if changes were made.
    - Updates the global QUOTE_CURRENCY based on the loaded/default config.

    Args:
        filepath: The path to the configuration JSON file (e.g., "config.json").

    Returns:
        The loaded, validated, and potentially corrected configuration dictionary.
        Returns the default configuration if the file cannot be loaded or created.
    """
    global QUOTE_CURRENCY # Allow updating the global QUOTE_CURRENCY constant

    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    # --- Define Default Configuration Structure ---
    # This structure serves as the template and provides default values.
    default_config = {
        # Trading Core
        "trading_pairs": ["BTC/USDT"],          # List of symbols to trade (e.g., ["BTC/USDT", "ETH/USDT"])
        "interval": "5",                        # Kline timeframe (must be in VALID_INTERVALS)
        "enable_trading": False,                # Master switch: MUST BE true for live orders. Safety default: false.
        "use_sandbox": True,                    # Use exchange's sandbox/testnet environment? Safety default: true.
        "quote_currency": "USDT",               # Primary currency for balance, PnL, risk (e.g., USDT, BUSD)
        "max_concurrent_positions": 1,          # Maximum number of positions allowed open simultaneously across all pairs.

        # Risk & Sizing
        "risk_per_trade": 0.01,                 # Fraction of available balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                         # Desired leverage for contract trading (0 or 1 for spot/no leverage). Exchange limits apply.

        # API & Timing
        "retry_delay": RETRY_DELAY_SECONDS,             # Base delay in seconds between API retry attempts
        "loop_delay_seconds": LOOP_DELAY_SECONDS,       # Delay in seconds between processing cycles for each symbol
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order placement before confirming position status

        # Data Fetching
        "fetch_limit": DEFAULT_FETCH_LIMIT,             # Default number of klines to fetch historically
        "orderbook_limit": 25,                          # Limit for order book depth fetching (if feature added later)

        # Strategy Parameters (Volumatic Trend + Order Blocks)
        "strategy_params": {
            # Volumatic Trend (VT)
            "vt_length": DEFAULT_VT_LENGTH,             # Lookback period for VT calculation (integer > 0)
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,     # Lookback period for ATR calculation (integer > 0)
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, # Lookback for Volume EMA/SWMA (integer > 0) - Placeholder Usage
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER), # ATR multiplier for VT bands (float > 0) - Placeholder Usage

            # Order Blocks (OB)
            "ob_source": DEFAULT_OB_SOURCE,             # Candle part to define OB: "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT,                 # Lookback periods for Pivot High detection (integer > 0)
            "ph_right": DEFAULT_PH_RIGHT,               # Lookforward periods for Pivot High detection (integer > 0)
            "pl_left": DEFAULT_PL_LEFT,                 # Lookback periods for Pivot Low detection (integer > 0)
            "pl_right": DEFAULT_PL_RIGHT,               # Lookforward periods for Pivot Low detection (integer > 0)
            "ob_extend": DEFAULT_OB_EXTEND,             # Extend OB visualization until violated? (boolean)
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,       # Max number of active OBs to track per side (integer > 0)
            "ob_entry_proximity_factor": 1.005,         # Proximity factor for entry signal near OB (float >= 1.0) - Placeholder Usage
            "ob_exit_proximity_factor": 1.001,          # Proximity factor for exit signal near opposite OB (float >= 1.0) - Placeholder Usage
        },

        # Protection Parameters (Stop Loss, Take Profit, Trailing, Break Even)
        "protection": {
            # Initial SL/TP (ATR-based)
            "initial_stop_loss_atr_multiple": 1.8,      # Initial SL distance = ATR * this multiple (float > 0)
            "initial_take_profit_atr_multiple": 0.7,    # Initial TP distance = ATR * this multiple (float >= 0, 0 means no initial TP)

            # Break Even (BE)
            "enable_break_even": True,                  # Enable moving SL to break-even? (boolean)
            "break_even_trigger_atr_multiple": 1.0,     # Move SL to BE when price moves ATR * multiple in profit (float > 0)
            "break_even_offset_ticks": 2,               # Offset SL from entry by this many price ticks for BE (integer >= 0)

            # Trailing Stop Loss (TSL)
            "enable_trailing_stop": True,               # Enable Trailing Stop Loss? (boolean) - Placeholder Usage
            "trailing_stop_callback_rate": 0.005,       # TSL distance/offset (interpretation depends on exchange/implementation, e.g., 0.005 = 0.5% or 0.5 price points) (float > 0) - Placeholder Usage
            "trailing_stop_activation_percentage": 0.003, # Activate TSL when price moves this % from entry (float >= 0) - Placeholder Usage
        }
    }

    config_needs_saving: bool = False
    loaded_config: Dict[str, Any] = {} # Initialize empty

    # --- Step 1: File Existence Check & Creation ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Configuration file '{filepath}' not found. Creating a new one with default settings.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Dump the default config structure to the new file with indentation
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully created default config file: {filepath}{RESET}")
            # Use the defaults since the file was just created
            loaded_config = default_config
            config_needs_saving = False # No need to save again immediately
            # Update global QUOTE_CURRENCY from the default we just wrote
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return defaults directly

        except IOError as e:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create config file '{filepath}': {e}.{RESET}")
            init_logger.critical(f"{NEON_RED}Please check directory permissions. Using internal defaults as fallback.{RESET}")
            # Fallback to using internal defaults in memory
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config

    # --- Step 2: File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
            raise TypeError("Configuration file content is not a valid JSON object (must be a dictionary).")
        init_logger.info(f"Successfully loaded configuration from '{filepath}'.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from config file '{filepath}': {e}{RESET}")
        init_logger.error(f"{NEON_RED}The file might be corrupted. Attempting to recreate it with default settings.{RESET}")
        try:
            # Optional: Backup corrupted file before overwriting
            backup_path = f"{filepath}.corrupted_{int(time.time())}.bak"
            os.replace(filepath, backup_path)
            init_logger.info(f"Backed up corrupted config to: {backup_path}")
        except Exception as backup_err:
             init_logger.warning(f"Could not back up corrupted config file: {backup_err}")

        # Recreate the default file
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return the defaults
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Error recreating config file: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Fallback to internal defaults
    except Exception as e:
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config # Fallback to internal defaults

    # --- Step 3: Ensure Keys and Validate ---
    try:
        # Ensure all default keys exist, add missing ones with default values
        updated_config, keys_added = _ensure_config_keys(loaded_config, default_config)
        if keys_added:
            config_needs_saving = True # Mark for saving later if keys were added

        # --- Validation Logic ---
        init_logger.debug("Starting configuration parameter validation...")

        # Helper function to navigate nested dicts safely for validation
        def get_nested_levels(cfg: Dict, path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
            """Gets the dict level and leaf key for validation, handling errors."""
            keys = path.split('.')
            current_cfg_level = cfg
            current_def_level = default_config
            try:
                for key in keys[:-1]: # Iterate through parent keys
                    if key not in current_cfg_level or not isinstance(current_cfg_level[key], dict):
                        raise KeyError(f"Path segment '{key}' not found or not a dictionary in loaded config.")
                    if key not in current_def_level or not isinstance(current_def_level[key], dict):
                         raise KeyError(f"Path segment '{key}' not found or not a dictionary in default config (structure mismatch).")
                    current_cfg_level = current_cfg_level[key]
                    current_def_level = current_def_level[key]
                leaf_key = keys[-1]
                # Ensure the leaf key exists in the default structure (should always if default_config is correct)
                if leaf_key not in current_def_level:
                    raise KeyError(f"Leaf key '{leaf_key}' not found in default config structure for path '{path}'.")
                return current_cfg_level, current_def_level, leaf_key
            except (KeyError, TypeError) as e:
                init_logger.error(f"Config validation structure error: Cannot access path '{path}'. Reason: {e}. Ensure config structure matches default.")
                return None, None, None

        # Define validation function that uses the helper and marks config for saving on changes
        def validate_numeric(cfg: Dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False):
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None:
                # Error already logged by helper, cannot validate this key.
                # Attempt to recover by setting the default value at the highest possible level if path was partially valid? Risky.
                # For safety, we might just proceed, relying on later code to handle potential missing keys,
                # or stop the bot if config is fundamentally broken. Let's assume _ensure_config_keys handles missing keys.
                init_logger.error(f"Skipping validation for '{key_path}' due to structure error.")
                return

            # Check if key exists at the leaf level (should exist due to _ensure_config_keys)
            if leaf_key not in cfg_level:
                 init_logger.warning(f"Config validation: Key '{key_path}' unexpectedly missing after ensure_keys. Using default value: {repr(def_level[leaf_key])}.")
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 return

            # Perform the validation and correction using the helper function
            corrected = _validate_and_correct_numeric(
                cfg_level, def_level, leaf_key, key_path,
                min_val, max_val, is_strict_min, is_int, allow_zero
            )
            if corrected:
                config_needs_saving = True # Mark for saving if _validate_and_correct_numeric made changes

        def validate_boolean(cfg: Dict, key_path: str):
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None: return # Error handled

            if leaf_key not in cfg_level: # Should not happen after _ensure_config_keys
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 init_logger.warning(f"Config validation: Boolean key '{key_path}' missing. Using default: {repr(def_level[leaf_key])}.")
                 return

            current_value = cfg_level[leaf_key]
            if not isinstance(current_value, bool):
                # Attempt to interpret common string representations
                corrected_val = None
                if isinstance(current_value, str):
                    val_lower = current_value.lower().strip()
                    if val_lower in ['true', 'yes', '1', 'on']: corrected_val = True
                    elif val_lower in ['false', 'no', '0', 'off']: corrected_val = False

                if corrected_val is not None:
                     init_logger.info(f"{NEON_YELLOW}Config Update: Corrected boolean-like value for '{key_path}' from {repr(current_value)} to {repr(corrected_val)}.{RESET}")
                     cfg_level[leaf_key] = corrected_val
                     config_needs_saving = True
                else:
                     # Cannot interpret, use default
                     init_logger.warning(f"Config Validation Warning: Invalid value for boolean key '{key_path}': {repr(current_value)}. Expected true/false. Using default: {repr(def_level[leaf_key])}.")
                     cfg_level[leaf_key] = def_level[leaf_key]
                     config_needs_saving = True

        def validate_string_choice(cfg: Dict, key_path: str, choices: List[str]):
             nonlocal config_needs_saving
             cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
             if cfg_level is None or def_level is None or leaf_key is None: return # Error handled

             if leaf_key not in cfg_level: # Should not happen
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 init_logger.warning(f"Config validation: Choice key '{key_path}' missing. Using default: {repr(def_level[leaf_key])}.")
                 return

             current_value = cfg_level[leaf_key]
             # Case-insensitive check for convenience, but store the canonical version from `choices`
             corrected_value = None
             if isinstance(current_value, str):
                 for choice in choices:
                      if current_value.lower() == choice.lower():
                          corrected_value = choice # Use the canonical casing
                          break

             if corrected_value is None: # Not a valid choice (or wrong type)
                 init_logger.warning(f"Config Validation Warning: Invalid value for '{key_path}': {repr(current_value)}. Must be one of {choices} (case-insensitive). Using default: {repr(def_level[leaf_key])}.")
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
             elif corrected_value != current_value: # Valid choice, but wrong case
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected case for '{key_path}' from '{current_value}' to '{corrected_value}'.{RESET}")
                 cfg_level[leaf_key] = corrected_value
                 config_needs_saving = True


        # --- Apply Validations to `updated_config` ---
        # Trading Core
        pairs = updated_config.get("trading_pairs", [])
        if not isinstance(pairs, list) or not pairs or not all(isinstance(s, str) and s and '/' in s for s in pairs):
            init_logger.warning(f"{NEON_YELLOW}Config Validation Warning: Invalid 'trading_pairs'. Must be a non-empty list of strings like 'BTC/USDT'. Using default: {default_config['trading_pairs']}.{RESET}")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True

        validate_string_choice(updated_config, "interval", VALID_INTERVALS)
        validate_boolean(updated_config, "enable_trading")
        validate_boolean(updated_config, "use_sandbox")

        # Validate quote_currency (must be non-empty string)
        qc = updated_config.get("quote_currency")
        if not isinstance(qc, str) or not qc.strip():
            init_logger.warning(f"{NEON_YELLOW}Config Validation Warning: Invalid 'quote_currency': {repr(qc)}. Must be a non-empty string. Using default: '{default_config['quote_currency']}'.{RESET}")
            updated_config["quote_currency"] = default_config["quote_currency"]
            config_needs_saving = True
        # Update the global QUOTE_CURRENCY immediately after validation
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT") # Use updated value or default
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")

        validate_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True) # Range 1 to 100

        # Risk & Sizing
        validate_numeric(updated_config, "risk_per_trade", 0.0, 1.0, is_strict_min=True) # Risk must be > 0% and <= 100%
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Allow 0 or 1 for spot, up to 200 for contracts (example limit)

        # API & Timing
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True) # 1-60 seconds
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True) # 1 second to 1 hour
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 120, is_int=True) # 1-120 seconds

        # Data Fetching
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True) # Min 50, max MAX_DF_LEN
        validate_numeric(updated_config, "orderbook_limit", 1, 100, is_int=True) # 1-100 depth

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # Range 1.0 to 1.1 (adjust as needed)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # Range 1.0 to 1.1 (adjust as needed)
        validate_string_choice(updated_config, "strategy_params.ob_source", ["Wicks", "Body"])
        validate_boolean(updated_config, "strategy_params.ob_extend")

        # Protection Params
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.0, 20.0, is_strict_min=True) # SL distance must be > 0
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0.0, 50.0, allow_zero=True) # TP distance can be 0 (disabled)
        validate_boolean(updated_config, "protection.enable_break_even")
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.0, 10.0, is_strict_min=True) # Trigger must be > 0 ATRs
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # Offset can be 0
        validate_boolean(updated_config, "protection.enable_trailing_stop")
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0, 0.2, is_strict_min=True) # Callback must be > 0 (e.g., 0-20%)
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0.0, 0.2, allow_zero=True) # Activation can be 0% (immediate)

        init_logger.debug("Configuration parameter validation complete.")

        # --- Step 4: Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info(f"{NEON_YELLOW}Configuration updated with defaults or corrections. Saving changes to '{filepath}'...{RESET}")
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Config file '{filepath}' updated successfully.{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with the updated configuration in memory, but changes are not saved to file.")

        init_logger.info(f"{Fore.CYAN}# Configuration loading and validation complete.{Style.RESET_ALL}")
        return updated_config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: An unexpected error occurred during config processing: {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        # Ensure QUOTE_CURRENCY is set from internal default even in this fatal case
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config

# --- Load Configuration ---
CONFIG = load_config(CONFIG_FILE)

# --- Exchange Initialization ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT exchange instance (Bybit) using API keys from environment
    and settings from the global CONFIG. Loads markets and performs an initial balance check.

    Args:
        logger: The logger instance to use for initialization messages.

    Returns:
        A configured ccxt.bybit instance if successful, otherwise None.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Initializing Bybit Exchange Connection...{Style.RESET_ALL}")

    try:
        # CCXT Exchange Options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Use ccxt's built-in rate limiter
            'options': {
                'defaultType': 'linear',      # Prefer linear contracts (USDT margined)
                'adjustForTimeDifference': True, # Auto-sync client time with server time
                # Set reasonable timeouts for common operations (in milliseconds)
                'fetchTickerTimeout': 15000,    # 15 seconds for fetching ticker
                'fetchBalanceTimeout': 25000,   # 25 seconds for fetching balance
                'createOrderTimeout': 35000,    # 35 seconds for creating orders
                'cancelOrderTimeout': 25000,    # 25 seconds for cancelling orders
                'fetchPositionsTimeout': 30000, # 30 seconds for fetching positions
                'fetchOHLCVTimeout': 60000,     # 60 seconds for fetching klines (can be large)
                # Bybit specific options (example, adjust if needed)
                # 'recvWindow': 10000, # Optional: Increase receive window if timestamp errors occur
            }
        }
        # Explicitly create Bybit instance
        exchange = ccxt.bybit(exchange_options)

        # Set Sandbox Mode based on config
        is_sandbox = CONFIG.get('use_sandbox', True) # Default to sandbox for safety
        exchange.set_sandbox_mode(is_sandbox)
        env_type = "Sandbox/Testnet" if is_sandbox else "LIVE Trading"
        env_color = NEON_YELLOW if is_sandbox else NEON_RED

        lg.warning(f"{env_color}{BRIGHT}!!! <<< {env_type} Environment ACTIVE >>> Exchange: {exchange.id} !!!{RESET}")
        if not is_sandbox and not CONFIG.get('enable_trading'):
            lg.warning(f"{NEON_YELLOW}Warning: LIVE environment selected, but 'enable_trading' is FALSE in config. No live orders will be placed.{RESET}")
        elif is_sandbox and CONFIG.get('enable_trading'):
             lg.info(f"Note: 'enable_trading' is TRUE, but operating in SANDBOX mode. Orders will be placed on the testnet.")


        # --- Load Market Data ---
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error: Optional[Exception] = None
        # Use a loop with retries for loading markets
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
                # Force reload on subsequent attempts to potentially fix temporary issues
                exchange.load_markets(reload=(attempt > 0))

                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market data loaded successfully. Found {len(exchange.markets)} symbols.{RESET}")
                    markets_loaded = True
                    break # Exit retry loop on success
                else:
                    # This case indicates an issue even if no exception was raised (e.g., empty response)
                    last_market_error = ValueError("Market data structure received from exchange is empty.")
                    lg.warning(f"Market data appears empty (Attempt {attempt + 1}). Retrying...")

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"Network error loading markets (Attempt {attempt + 1}): {e}. Retrying...")
            except ccxt.RateLimitExceeded as e:
                 last_market_error = e
                 # Use a longer delay specifically for rate limit errors
                 wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2) # Exponential backoff + extra base
                 lg.warning(f"{NEON_YELLOW}Rate limit exceeded loading markets: {e}. Waiting {wait_time}s before next attempt...{RESET}")
                 time.sleep(wait_time)
                 continue # Don't count as standard attempt, just wait and retry
            except ccxt.AuthenticationError as e:
                # Authentication errors are fatal and non-retryable
                last_market_error = e
                lg.critical(f"{NEON_RED}{BRIGHT}Authentication Error loading markets: {e}{RESET}")
                lg.critical(f"{NEON_RED}Please check your API Key, Secret, and ensure IP whitelist (if used) is correct. Exiting.{RESET}")
                return None
            except ccxt.ExchangeNotAvailable as e:
                # Exchange maintenance or temporary unavailability
                last_market_error = e
                lg.warning(f"Exchange not available loading markets (Attempt {attempt + 1}): {e}. Retrying...")
            except Exception as e:
                # Catch any other unexpected critical errors during market loading
                last_market_error = e
                lg.critical(f"{NEON_RED}{BRIGHT}Unexpected critical error loading markets: {e}{RESET}", exc_info=True)
                lg.critical(f"{NEON_RED}Exiting due to unexpected market loading error.{RESET}")
                return None # Non-retryable for unexpected errors

            # Wait before the next retry attempt (if not successful yet)
            if not markets_loaded and attempt < MAX_API_RETRIES:
                # Use exponential backoff for retry delay
                delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.info(f"Waiting {delay}s before retrying market load...")
                time.sleep(delay)

        # Check if markets were loaded successfully after all retries
        if not markets_loaded:
            lg.critical(f"{NEON_RED}{BRIGHT}Failed to load market data after {MAX_API_RETRIES + 1} attempts.{RESET}")
            lg.critical(f"{NEON_RED}Last error encountered: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"Exchange initialized: {exchange.id} | Version: {ccxt.__version__} | Sandbox: {is_sandbox}")

        # --- Initial Balance Check ---
        # Use the globally configured quote currency for the primary balance check
        balance_currency = QUOTE_CURRENCY
        lg.info(f"Performing initial balance check for {balance_currency}...")
        initial_balance: Optional[Decimal] = None
        try:
            # fetch_balance can raise AuthenticationError, handle it specifically
            initial_balance = fetch_balance(exchange, balance_currency, lg)
        except ccxt.AuthenticationError as auth_err:
            # Catch auth error here again as fetch_balance might re-raise it
            lg.critical(f"{NEON_RED}{BRIGHT}Authentication Error during initial balance check: {auth_err}{RESET}")
            lg.critical(f"{NEON_RED}Cannot verify balance. Exiting.{RESET}")
            return None
        except Exception as balance_err:
            # Log other balance fetch errors as warnings, especially if trading is disabled
            lg.warning(f"{NEON_YELLOW}Initial balance check for {balance_currency} failed: {balance_err}{RESET}", exc_info=False)

        # Evaluate outcome of balance check
        if initial_balance is not None:
            # Successfully fetched balance
            lg.info(f"{NEON_GREEN}Initial balance check successful: {initial_balance.normalize()} {balance_currency}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange initialization complete.{Style.RESET_ALL}")
            return exchange
        else:
            # Balance check failed (returned None)
            lg.error(f"{NEON_RED}Initial balance check FAILED for {balance_currency}. Could not retrieve balance.{RESET}")
            # Decide whether to proceed based on 'enable_trading' flag
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}{BRIGHT}Trading is ENABLED, but the initial balance check failed.{RESET}")
                lg.critical(f"{NEON_RED}Cannot proceed safely without confirming balance. Exiting.{RESET}")
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is DISABLED. Proceeding cautiously without initial balance confirmation.{RESET}")
                lg.info(f"{Fore.CYAN}# Exchange initialization complete (Balance check failed, Trading Disabled).{Style.RESET_ALL}")
                return exchange # Allow proceeding only if trading is off

    except ccxt.AuthenticationError as e:
         # Catch auth errors during the initial setup phase (before market load/balance check)
         lg.critical(f"{NEON_RED}{BRIGHT}Authentication error during exchange setup: {e}. Exiting.{RESET}")
         return None
    except Exception as e:
        # Catch any other unexpected critical errors during initialization
        lg.critical(f"{NEON_RED}{BRIGHT}A critical error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Helper Functions (Enhanced & Refined) ---

def _safe_market_decimal(value: Optional[Any], field_name: str,
                         allow_zero: bool = True, allow_negative: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value (often from market or position data) into a Decimal object.
    Handles None, empty strings, non-finite numbers (NaN, Inf), and applies configurable
    checks for zero and negative values.

    Args:
        value: The value to convert (can be string, int, float, Decimal, None, etc.).
        field_name: Name of the field being converted (used for logging context on failure).
        allow_zero: If True, Decimal('0') is considered a valid result.
        allow_negative: If True, negative Decimal values are considered valid.

    Returns:
        The converted Decimal value if valid according to the rules, otherwise None.
    """
    if value is None:
        # init_logger.debug(f"SafeDecimal: Input value for '{field_name}' is None.")
        return None

    try:
        # Convert to string first for consistent handling of floats and potential leading/trailing spaces
        s_val = str(value).strip()
        if not s_val: # Handle empty strings explicitly after stripping
            # init_logger.debug(f"SafeDecimal: Empty string rejected for '{field_name}'.")
            return None

        d_val = Decimal(s_val)

        # Check for non-finite values (NaN, Infinity) which are invalid for market data
        if not d_val.is_finite():
             # init_logger.debug(f"SafeDecimal: Non-finite value ({value}) rejected for '{field_name}'.")
             return None

        # Validate based on flags
        if not allow_zero and d_val.is_zero():
            # init_logger.debug(f"SafeDecimal: Zero value rejected for '{field_name}' (Value: {value}).")
            return None
        if not allow_negative and d_val < Decimal('0'):
            # init_logger.debug(f"SafeDecimal: Negative value ({d_val.normalize()}) rejected for '{field_name}' (Value: {value}).")
            return None

        return d_val # Return the valid Decimal object
    except (InvalidOperation, TypeError, ValueError) as e:
        # Log conversion errors at debug level as they can be frequent with optional fields
        # init_logger.debug(f"SafeDecimal: Failed to convert '{field_name}' to Decimal. Input: {repr(value)}, Error: {e}")
        return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price value according to the market's price precision rules using ccxt.
    Ensures the input price is a valid positive Decimal before attempting formatting.

    Args:
        exchange: The ccxt exchange instance (with loaded markets).
        symbol: The market symbol (e.g., 'BTC/USDT').
        price: The price value to format (Decimal, float, or string representation).

    Returns:
        The formatted price as a string if successful and valid, otherwise None.
    """
    # 1. Validate and convert input price to a positive Decimal
    price_decimal = _safe_market_decimal(price, f"format_price_input({symbol})", allow_zero=False, allow_negative=False)

    if price_decimal is None:
        init_logger.warning(f"Price formatting skipped ({symbol}): Input price '{price}' is invalid, zero, or negative.")
        return None

    # 2. Format using ccxt's price_to_precision
    try:
        # ccxt's method typically requires a float argument
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))

        # 3. Post-Validation: Ensure formatted string is still a valid positive number
        # This catches cases where precision might round down to zero or near-zero, or invalid chars.
        formatted_decimal = _safe_market_decimal(formatted_str, f"format_price_output({symbol})", allow_zero=False, allow_negative=False)
        if formatted_decimal is None:
             init_logger.warning(f"Price formatting warning ({symbol}): Input '{price}' formatted to non-positive or invalid value '{formatted_str}'. Returning None.")
             return None

        return formatted_str # Return the successfully formatted and validated price string
    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
        init_logger.error(f"Price formatting failed ({symbol}): Error accessing market precision: {e}. Ensure markets are loaded.")
        return None
    except (InvalidOperation, ValueError, TypeError) as e:
        # Should be caught by _safe_market_decimal mostly, but keep as fallback for float conversion or ccxt method errors
        init_logger.warning(f"Price formatting failed ({symbol}): Error during formatting process for price '{price}': {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during formatting
        init_logger.warning(f"Price formatting failed ({symbol}): Unexpected error for price '{price}': {e}")
        return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using ccxt's `fetch_ticker`.
    Attempts to use price sources in order of preference: 'last', mid-price ('bid'/'ask'), 'ask', 'bid'.
    Includes retry logic for network/exchange errors.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        logger: The logger instance for messages specific to this operation.

    Returns:
        The current price as a Decimal object if successfully fetched and valid, otherwise None.
    """
    lg = logger
    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for current price ({symbol}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            ticker = exchange.fetch_ticker(symbol)

            # --- Extract and Validate Prices from Ticker ---
            price: Optional[Decimal] = None
            source: str = "N/A" # Source of the price (last, mid, ask, bid)

            # Helper to safely get a positive Decimal from ticker data
            def safe_positive_decimal_from_ticker(key: str) -> Optional[Decimal]:
                return _safe_market_decimal(ticker.get(key), f"ticker.{key} ({symbol})", allow_zero=False, allow_negative=False)

            # 1. Try 'last' price
            price = safe_positive_decimal_from_ticker('last')
            if price:
                source = "'last' price"
            else:
                # 2. Try mid-price from 'bid' and 'ask'
                bid = safe_positive_decimal_from_ticker('bid')
                ask = safe_positive_decimal_from_ticker('ask')
                if bid and ask:
                    # Sanity check: ensure bid < ask before calculating mid-price
                    if bid < ask:
                        price = (bid + ask) / Decimal('2')
                        # Quantize mid-price to price tick precision for consistency? Optional.
                        # price = price.quantize(price_tick, ROUND_HALF_UP)
                        source = f"mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()})"
                    else:
                        # If bid >= ask (crossed or equal book), something is wrong.
                        # Prefer 'ask' as a safer estimate in this scenario.
                        price = ask
                        source = f"'ask' price (used due to crossed/equal book: Bid={bid}, Ask={ask})"
                        lg.warning(f"Crossed/equal order book detected for {symbol} (Bid >= Ask). Using Ask price.")
                elif ask:
                    # 3. Fallback to 'ask' price
                    price = ask
                    source = f"'ask' price ({ask.normalize()})"
                elif bid:
                    # 4. Fallback to 'bid' price
                    price = bid
                    source = f"'bid' price ({bid.normalize()})"

            # --- Return Valid Price or Log Warning ---
            if price:
                normalized_price = price.normalize() # Remove trailing zeros for logging
                lg.debug(f"Current price ({symbol}) obtained from {source}: {normalized_price}")
                return normalized_price
            else:
                # Ticker fetched, but no usable price source found
                last_exception = ValueError(f"No valid price source (last/bid/ask) found in ticker response for {symbol}.")
                lg.warning(f"Could not find a valid price in ticker data ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}")
                # Continue to retry logic below

        # --- Error Handling for API Call ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Use a longer, potentially exponential delay for rate limit errors
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2)
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Rate limit doesn't count as a standard attempt here, just wait and loop again
            continue
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching price: {e}. Cannot continue.{RESET}")
            return None
        except ccxt.BadSymbol as e:
             # Fatal error for this symbol
             last_exception = e
             lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching price on {exchange.id}: {e}.{RESET}")
             return None
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., maintenance, temporary issues) - potentially retryable
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors during price fetching
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price ({symbol}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as potentially fatal for safety in this context
            return None

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff for retries

    # If loop finishes without successfully fetching a price
    lg.error(f"{NEON_RED}Failed to fetch current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches historical kline/OHLCV data for a symbol using ccxt's fetch_ohlcv.
    Handles exchange-specific pagination (e.g., Bybit V5 'until' parameter),
    data validation, lag checks, deduplication, and length trimming. Includes robust retry logic.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        timeframe: The CCXT timeframe string (e.g., '5m', '1h', '1d').
        limit: The total number of candles desired (approximate, due to pagination).
        logger: The logger instance for messages specific to this operation.

    Returns:
        A pandas DataFrame containing the OHLCV data, indexed by UTC timestamp,
        with columns ['open', 'high', 'low', 'close', 'volume'] as Decimals.
        Returns an empty DataFrame if fetching fails or data is invalid.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Fetching klines for {symbol} | Timeframe: {timeframe} | Target Limit: {limit}...{Style.RESET_ALL}")

    # --- Pre-checks ---
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetching OHLCV data via ccxt.")
        return pd.DataFrame()

    # Estimate minimum candles needed based on strategy params (best effort)
    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        # Find the max lookback period needed by indicators used
        min_required = max(
            sp.get('vt_length', 0) * 2,             # EMA/SWMA needs buffer
            sp.get('vt_atr_period', 0) + 1,         # ATR needs N+1
            sp.get('vt_vol_ema_length', 0) * 2,     # EMA needs buffer
            sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1, # Pivots need full window
            sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1
        ) + 50 # Add a generous safety buffer for calculations & stability
        lg.debug(f"Estimated minimum candles required by strategy: ~{min_required}")
    except Exception as e:
        lg.warning(f"Could not estimate minimum candle requirement for {symbol}: {e}")

    if limit < min_required:
        lg.warning(f"{NEON_YELLOW}Requested kline limit ({limit}) is less than the estimated strategy requirement ({min_required}) for {symbol}. "
                   f"Indicator accuracy may be affected, especially on initial runs.{RESET}")

    # Determine category and market ID for Bybit V5 API (required params)
    category = 'spot' # Default assumption
    market_id = symbol # Default to symbol if market info lookup fails
    is_bybit = 'bybit' in exchange.id.lower()
    try:
        # Fetch market info to determine type and specific ID
        market = exchange.market(symbol)
        market_id = market['id'] # Use exchange-specific ID
        if market.get('linear'): category = 'linear'
        elif market.get('inverse'): category = 'inverse'
        elif market.get('spot'): category = 'spot'
        # else category remains 'spot' or determined by defaultType
        lg.debug(f"Using API parameters for {symbol}: category='{category}', market ID='{market_id}' (for Bybit V5).")
    except (ccxt.BadSymbol, KeyError, TypeError) as e:
        lg.warning(f"Could not reliably determine market category/ID for {symbol}: {e}. "
                   f"Proceeding with defaults (category='{category}', market_id='{market_id}'). May fail if incorrect for Bybit V5.")

    # --- Fetching Loop ---
    all_ohlcv_data: List[List] = [] # Stores raw candle lists [[ts, o, h, l, c, v], ...]
    remaining_limit = limit
    end_timestamp_ms: Optional[int] = None # For pagination: fetch candles *before* this timestamp (exclusive)

    # Determine the maximum number of candles the API returns per request
    api_limit_per_req = getattr(exchange, 'limits', {}).get('fetchOHLCV', {}).get('limit', BYBIT_API_KLINE_LIMIT)
    if api_limit_per_req is None or api_limit_per_req <= 0:
        lg.warning(f"Could not determine API limit per request for OHLCV on {exchange.id}. Using default: {BYBIT_API_KLINE_LIMIT}")
        api_limit_per_req = BYBIT_API_KLINE_LIMIT

    # Calculate max chunks generously to avoid infinite loops if API behaves unexpectedly
    max_chunks = math.ceil(limit / api_limit_per_req) + 5 # Add a buffer
    chunk_num = 0
    total_fetched_raw = 0

    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        # Request size for this chunk (up to API limit or remaining needed)
        fetch_size = min(remaining_limit, api_limit_per_req)
        lg.debug(f"Fetching kline chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. "
                 f"Ending before TS: {datetime.fromtimestamp(end_timestamp_ms / 1000, tz=timezone.utc).isoformat() if end_timestamp_ms else 'Latest'}")

        attempts = 0
        last_exception: Optional[Exception] = None
        chunk_data: Optional[List[List]] = None

        while attempts <= MAX_API_RETRIES:
            try:
                # --- Prepare API Call Arguments ---
                params = {'category': category} if is_bybit else {} # Pass category for Bybit V5
                fetch_args: Dict[str, Any] = {
                    'symbol': symbol,       # Use standard symbol for ccxt call
                    'timeframe': timeframe,
                    'limit': fetch_size,
                    'params': params
                }
                # Add 'until' parameter for pagination (fetches candles ending *before* this timestamp)
                # CCXT handles mapping 'until' to exchange-specific params like 'end' for Bybit V5
                if end_timestamp_ms:
                    fetch_args['until'] = end_timestamp_ms

                # --- Execute API Call ---
                lg.debug(f"Calling exchange.fetch_ohlcv with args: {fetch_args}")
                fetched_chunk = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(fetched_chunk) if fetched_chunk else 0
                lg.debug(f"API returned {fetched_count_chunk} candles for chunk {chunk_num}.")

                # --- Basic Validation of Fetched Chunk ---
                if fetched_chunk:
                    # Check if data looks valid (e.g., expected number of columns: timestamp, O, H, L, C, V)
                    if not all(isinstance(candle, list) and len(candle) >= 6 for candle in fetched_chunk):
                        raise ValueError(f"Invalid candle format received in chunk {chunk_num} for {symbol}. Expected list with >= 6 values.")

                    chunk_data = fetched_chunk # Assign valid data

                    # --- Data Lag Check (only on the first chunk when fetching latest data) ---
                    if chunk_num == 1 and end_timestamp_ms is None:
                        try:
                            last_candle_ts_ms = chunk_data[-1][0] # Timestamp of the most recent candle in the chunk
                            last_ts = pd.to_datetime(last_candle_ts_ms, unit='ms', utc=True, errors='raise')
                            interval_seconds = exchange.parse_timeframe(timeframe)

                            if interval_seconds:
                                # Allow up to ~2.5 intervals of lag before warning/retrying
                                max_lag_seconds = interval_seconds * 2.5
                                current_utc_time = pd.Timestamp.utcnow()
                                actual_lag_seconds = (current_utc_time - last_ts).total_seconds()

                                if actual_lag_seconds > max_lag_seconds:
                                    lag_error_msg = (f"Potential data lag detected! Last candle time ({last_ts}) is "
                                                     f"{actual_lag_seconds:.1f}s old. Max allowed lag for {timeframe} is ~{max_lag_seconds:.1f}s.")
                                    last_exception = ValueError(lag_error_msg)
                                    lg.warning(f"{NEON_YELLOW}Lag Check ({symbol}): {lag_error_msg} Retrying fetch...{RESET}")
                                    chunk_data = None # Discard potentially stale data and force retry
                                    # No break here, let the retry logic handle waiting below
                                else:
                                    lg.debug(f"Lag check passed ({symbol}): Last candle {actual_lag_seconds:.1f}s old (within limit).")
                                    break # Valid chunk received, exit retry loop for this chunk
                            else:
                                lg.warning(f"Could not parse timeframe '{timeframe}' to seconds for lag check.")
                                break # Proceed without lag check if timeframe parsing fails
                        except (IndexError, TypeError, ValueError) as ts_err:
                             lg.warning(f"Could not perform lag check: Error processing timestamp in first chunk ({symbol}): {ts_err}")
                             break # Proceed if lag check itself fails
                    else: # Not the first chunk or not fetching latest, no lag check needed
                        break # Valid chunk received, exit retry loop

                else: # API returned empty list []
                    lg.debug(f"API returned no data (empty list) for chunk {chunk_num} ({symbol}).")
                    # If it's the *first* chunk, it might be a temporary issue, so retry.
                    # If it's a *later* chunk, it likely means the end of historical data.
                    if chunk_num > 1:
                        lg.info(f"Assuming end of historical data for {symbol} after chunk {chunk_num-1} returned empty.")
                        remaining_limit = 0 # Stop fetching more chunks
                        break # Exit retry loop for this chunk (as it's expected end of history)
                    else:
                        # First chunk was empty, treat as potential error and let retry logic handle it
                         last_exception = ValueError("API returned empty list for the first kline chunk.")
                         # No break here, continue to retry logic


            # --- Error Handling for fetch_ohlcv Call ---
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_exception = e
                lg.warning(f"{NEON_YELLOW}Network error fetching klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2)
                lg.warning(f"{NEON_YELLOW}Rate limit fetching klines chunk {chunk_num} ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
                time.sleep(wait_time)
                continue # Don't increment standard attempts, just wait
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Authentication error fetching klines: {e}. Cannot continue.{RESET}")
                return pd.DataFrame() # Fatal
            except ccxt.BadSymbol as e:
                 last_exception = e
                 lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching klines on {exchange.id}: {e}.{RESET}")
                 return pd.DataFrame() # Fatal for this symbol
            except ccxt.ExchangeError as e:
                last_exception = e
                # Check for specific non-retryable errors (e.g., invalid timeframe)
                err_str = str(e).lower()
                non_retryable_msgs = ["invalid timeframe", "interval not supported", "symbol invalid", "instrument not found", "invalid category", "market is closed"]
                if any(msg in err_str for msg in non_retryable_msgs):
                    lg.critical(f"{NEON_RED}Non-retryable exchange error fetching klines for {symbol}: {e}. Stopping kline fetch.{RESET}")
                    return pd.DataFrame() # Fatal for this symbol
                else:
                    # Treat other exchange errors as potentially retryable
                    lg.warning(f"{NEON_YELLOW}Exchange error fetching klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ValueError as e: # Catch our validation errors (e.g., candle format, lag)
                 last_exception = e
                 lg.error(f"{NEON_RED}Data validation error fetching klines chunk {chunk_num} ({symbol}): {e}. Retrying...{RESET}")
            except Exception as e:
                last_exception = e
                lg.error(f"{NEON_RED}Unexpected error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True)
                # Treat unexpected errors cautiously - stop fetching for this symbol for safety
                return pd.DataFrame()

            # --- Retry Logic ---
            attempts += 1
            # Only sleep if we need to retry (chunk_data is None or lag detected and more attempts left)
            if chunk_data is None and attempts <= MAX_API_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

        # --- Process Successful Chunk or Handle Failure for this Chunk ---
        if chunk_data:
            # Prepend the new chunk to maintain chronological order (oldest fetched first appears first in list)
            all_ohlcv_data = chunk_data + all_ohlcv_data
            chunk_len = len(chunk_data)
            remaining_limit -= chunk_len
            total_fetched_raw += chunk_len

            # Set timestamp for the next older chunk request
            # Use the timestamp of the *first* candle in the *current* chunk
            try:
                next_until_ts = chunk_data[0][0] # Timestamp of the oldest candle in this chunk
                if not isinstance(next_until_ts, (int, float)) or next_until_ts <= 0:
                    raise ValueError(f"Invalid timestamp found in first candle of chunk: {next_until_ts}")
                # 'until' is exclusive, so we want candles *before* this timestamp.
                # No need to subtract 1ms if the timestamp itself is the boundary.
                end_timestamp_ms = int(next_until_ts)
            except (IndexError, TypeError, ValueError) as ts_err:
                lg.error(f"Error determining next 'until' timestamp from chunk data ({symbol}): {ts_err}. Stopping pagination.")
                remaining_limit = 0 # Stop fetching if we can't paginate correctly

            # Check if the exchange returned fewer candles than requested (might indicate end of history)
            if chunk_len < fetch_size:
                lg.debug(f"Received fewer candles ({chunk_len}) than requested ({fetch_size}) for chunk {chunk_num}. Assuming end of historical data.")
                remaining_limit = 0 # Stop fetching more chunks

        else: # Failed to fetch this chunk after all retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            lg.error(f"  Last error for chunk {chunk_num}: {last_exception}")
            if not all_ohlcv_data:
                # Failed on the very first chunk, cannot proceed
                lg.error(f"Failed to fetch the initial chunk for {symbol}. Cannot construct DataFrame.")
                return pd.DataFrame()
            else:
                # Failed on a subsequent chunk, proceed with the data collected so far
                lg.warning(f"Proceeding with {total_fetched_raw} raw candles fetched before the error occurred.")
                break # Exit the main fetching loop

        # Small delay between chunk requests to be polite to the API, especially if looping
        if remaining_limit > 0 and chunk_num < max_chunks:
            time.sleep(0.3) # 300ms delay

    # --- Post-Fetching Checks ---
    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines for {symbol} because maximum chunk limit ({max_chunks}) was reached. "
                   f"Fetched {total_fetched_raw} raw candles.")

    if not all_ohlcv_data:
        lg.error(f"No kline data could be successfully fetched for {symbol} {timeframe}.")
        return pd.DataFrame()

    lg.info(f"Total raw klines fetched across all chunks: {len(all_ohlcv_data)}")

    # --- Data Deduplication and Sorting ---
    # Use a dictionary keyed by timestamp to automatically handle duplicates; keep the last seen entry.
    unique_candles_dict: Dict[int, List] = {}
    invalid_candle_count = 0
    for candle in all_ohlcv_data:
        try:
            # Validate candle structure and timestamp
            if not isinstance(candle, list) or len(candle) < 6:
                 invalid_candle_count += 1
                 continue
            timestamp = int(candle[0])
            if timestamp <= 0:
                 invalid_candle_count += 1
                 continue
            # Store/overwrite candle in dict using timestamp as key
            unique_candles_dict[timestamp] = candle
        except (IndexError, TypeError, ValueError):
            invalid_candle_count += 1
            continue # Skip candles with invalid format or timestamp

    if invalid_candle_count > 0:
         lg.warning(f"Skipped {invalid_candle_count} invalid raw candle entries during deduplication for {symbol}.")

    # Extract unique candles and sort them chronologically by timestamp
    unique_data = sorted(list(unique_candles_dict.values()), key=lambda x: x[0])
    final_unique_count = len(unique_data)

    duplicates_removed = len(all_ohlcv_data) - invalid_candle_count - final_unique_count
    if duplicates_removed > 0:
        lg.info(f"Removed {duplicates_removed} duplicate candle(s) based on timestamp for {symbol}.")
    elif duplicates_removed < 0: # Should not happen with dict method
         lg.warning(f"Data count mismatch during deduplication ({symbol}). Raw: {len(all_ohlcv_data)}, Invalid: {invalid_candle_count}, Final Unique: {final_unique_count}")

    if not unique_data:
        lg.error(f"No valid, unique kline data remaining after processing for {symbol}.")
        return pd.DataFrame()

    # Trim excess data if more than the originally requested limit was fetched due to chunking overlaps
    # Keep the most recent 'limit' candles
    if final_unique_count > limit:
        lg.debug(f"Fetched {final_unique_count} unique candles, trimming to the target limit of {limit} (keeping most recent).")
        unique_data = unique_data[-limit:]
        final_unique_count = len(unique_data) # Update count after trimming

    # --- DataFrame Creation and Cleaning ---
    try:
        lg.debug(f"Processing {final_unique_count} final unique candles into DataFrame for {symbol}...")
        # Standard OHLCV columns expected by most strategies
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Use only the columns available in the data (up to 6)
        df = pd.DataFrame(unique_data, columns=cols[:len(unique_data[0])])

        # Convert timestamp column to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        initial_len_ts = len(df)
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        if len(df) < initial_len_ts:
            lg.warning(f"Dropped {initial_len_ts - len(df)} rows with invalid timestamps during conversion for {symbol}.")
        if df.empty:
            lg.error(f"DataFrame became empty after timestamp conversion and NaN drop ({symbol}).")
            return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision, handling potential errors
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Apply the robust _safe_market_decimal conversion to each element
                # Allow zero only for volume, disallow negative for all OHLCV
                df[col] = df[col].apply(
                    lambda x: _safe_market_decimal(x, f"df.{col}", allow_zero=(col=='volume'), allow_negative=False)
                )
                # Check if conversion resulted in all NaNs (indicates bad input data type)
                if df[col].isnull().all():
                     lg.warning(f"Column '{col}' for {symbol} became all NaN after Decimal conversion. Original data might be incompatible (e.g., all strings).")
            elif col != 'volume': # Volume might legitimately be missing from some exchanges/endpoints
                lg.error(f"Essential OHLC column '{col}' not found in fetched data for {symbol}. Cannot proceed.")
                return pd.DataFrame() # Fail if essential OHLC columns are missing

        # --- Data Cleaning: Drop rows with NaN in essential columns ---
        initial_len_clean = len(df)
        essential_cols_present = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        df.dropna(subset=essential_cols_present, inplace=True)
        # Also ensure volume is present and non-NaN if the column exists
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)

        rows_dropped = initial_len_clean - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid OHLCV data during cleaning for {symbol}.")

        if df.empty:
            lg.warning(f"DataFrame became empty after cleaning NaN/invalid values ({symbol}).")
            return pd.DataFrame()

        # Verify index is sorted chronologically (monotonic increasing)
        if not df.index.is_monotonic_increasing:
            lg.warning(f"DataFrame index for {symbol} is not monotonically increasing. Sorting index...")
            df.sort_index(inplace=True)

        # Optional: Limit final DataFrame length to prevent excessive memory usage over time
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds max ({MAX_DF_LEN}). Trimming oldest data ({symbol}).")
            df = df.iloc[-MAX_DF_LEN:] # Keep only the most recent MAX_DF_LEN rows

        lg.info(f"{NEON_GREEN}Successfully processed {len(df)} klines into DataFrame for {symbol} {timeframe}.{RESET}")
        # lg.debug(f"DataFrame Head:\n{df.head().to_string(max_colwidth=15)}")
        # lg.debug(f"DataFrame Tail:\n{df.tail().to_string(max_colwidth=15)}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing fetched klines into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Retrieves and standardizes market information for a given symbol from the exchange.
    Uses the cached `exchange.markets` data, attempting a refresh if the symbol is not found initially.
    Parses critical information like precision and limits into Decimal types and derives convenience flags.
    Includes retry logic for potential temporary issues during market refresh.

    Args:
        exchange: The initialized ccxt exchange instance (markets should ideally be loaded).
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        logger: The logger instance for messages specific to this operation.

    Returns:
        A MarketInfo TypedDict containing standardized and enhanced market data if found and valid,
        otherwise None. Returns None immediately if the symbol is definitively not supported.
    """
    lg = logger
    lg.debug(f"Retrieving market details for symbol: {symbol}...")
    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            market_dict: Optional[Dict] = None
            market_found_in_cache = False

            # 1. Try to get market from cache first
            if exchange.markets and symbol in exchange.markets:
                market_dict = exchange.markets[symbol]
                market_found_in_cache = True
                lg.debug(f"Market info for '{symbol}' found in cache.")
            else:
                # 2. If not in cache, attempt to refresh market map (once per retry cycle)
                if attempts == 0: # Only try reloading once initially if not found
                    lg.info(f"Market details for '{symbol}' not found in cache. Attempting to refresh market map...")
                    try:
                        exchange.load_markets(reload=True)
                        lg.info(f"Market map refreshed. Found {len(exchange.markets)} markets.")
                        # Try accessing from cache again after reload
                        if exchange.markets and symbol in exchange.markets:
                            market_dict = exchange.markets[symbol]
                            lg.debug(f"Market info for '{symbol}' found after refresh.")
                        else:
                             # Symbol not found even after refresh - likely invalid/unsupported
                             raise ccxt.BadSymbol(f"Symbol '{symbol}' not found on {exchange.id} after market refresh.")
                    except ccxt.BadSymbol as e:
                        # This is definitive: symbol doesn't exist on the exchange
                        lg.error(f"{NEON_RED}{e}{RESET}")
                        return None # Non-retryable
                    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError) as reload_err:
                        # Log refresh error but continue to retry logic below
                        last_exception = reload_err
                        lg.warning(f"Failed to refresh market map while looking for '{symbol}': {reload_err}. Will retry.")
                        # market_dict remains None, loop will retry
                    except Exception as reload_err:
                         # Unexpected error during refresh
                         last_exception = reload_err
                         lg.error(f"Unexpected error refreshing markets for '{symbol}': {reload_err}", exc_info=True)
                         return None # Treat as fatal
                else:
                     # If still not found on subsequent attempts, rely on previous error or BadSymbol
                     lg.debug(f"Market '{symbol}' still not found after previous attempts/errors.")
                     # Allow loop to finish and report failure based on last_exception

            # 3. Process Market Dictionary if Found
            if market_dict:
                lg.debug(f"Raw market data found for {symbol}. Parsing and standardizing...")
                # --- Standardize and Enhance Market Data ---
                std_market = market_dict.copy() # Work on a copy

                # Basic type flags from ccxt structure
                is_spot = std_market.get('spot', False)
                is_swap = std_market.get('swap', False)
                is_future = std_market.get('future', False)
                is_option = std_market.get('option', False)
                is_contract_base = std_market.get('contract', False) # Base 'contract' flag

                # Determine if it's any kind of contract
                std_market['is_contract'] = is_swap or is_future or is_option or is_contract_base
                is_linear = std_market.get('linear') # Can be True/False/None
                is_inverse = std_market.get('inverse') # Can be True/False/None

                # Ensure linear/inverse flags are boolean and only True if it's actually a contract
                std_market['is_linear'] = bool(is_linear) and std_market['is_contract']
                std_market['is_inverse'] = bool(is_inverse) and std_market['is_contract']

                # Determine contract type string for logging/logic
                if std_market['is_linear']:
                    std_market['contract_type_str'] = "Linear"
                elif std_market['is_inverse']:
                    std_market['contract_type_str'] = "Inverse"
                elif is_spot:
                     std_market['contract_type_str'] = "Spot"
                elif is_option:
                     std_market['contract_type_str'] = "Option"
                elif std_market['is_contract']: # Catch-all for other contract types
                     std_market['contract_type_str'] = "Contract (Other)"
                else:
                     std_market['contract_type_str'] = "Unknown"

                # --- Extract Precision and Limits Safely using Helper ---
                precision = std_market.get('precision', {})
                limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                # Convert precision steps to Decimal (must be positive)
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), f"{symbol} precision.amount", allow_zero=False, allow_negative=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), f"{symbol} precision.price", allow_zero=False, allow_negative=False)

                # Convert limits to Decimal
                # Amounts must be non-negative
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), f"{symbol} limits.amount.min", allow_zero=True, allow_negative=False)
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), f"{symbol} limits.amount.max", allow_zero=False, allow_negative=False) # Max > 0 generally
                # Costs must be non-negative
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), f"{symbol} limits.cost.min", allow_zero=True, allow_negative=False)
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), f"{symbol} limits.cost.max", allow_zero=False, allow_negative=False) # Max > 0 generally

                # Convert contract size to Decimal (default to 1 if missing/invalid/spot)
                # Contract size must be positive
                contract_size_val = std_market.get('contractSize') if std_market['is_contract'] else '1'
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, f"{symbol} contractSize", allow_zero=False, allow_negative=False) or Decimal('1')

                # --- Validation of Critical Data ---
                # Precision steps are essential for placing orders correctly. Fail if missing.
                if std_market['amount_precision_step_decimal'] is None:
                    raise ValueError(f"CRITICAL VALIDATION FAILED ({symbol}): Missing essential 'precision.amount' data! Cannot determine order size step.")
                if std_market['price_precision_step_decimal'] is None:
                    raise ValueError(f"CRITICAL VALIDATION FAILED ({symbol}): Missing essential 'precision.price' data! Cannot determine price step.")
                # Min amount is often needed for order placement validation. Warn if missing.
                if std_market['min_amount_decimal'] is None:
                     lg.warning(f"{NEON_YELLOW}Market Validation Warning ({symbol}): Missing 'limits.amount.min' data. Order sizing/placement might fail if size is too small.{RESET}")

                # --- Log Parsed Details ---
                # Helper for formatting optional Decimals for logging, avoids None errors
                def fmt_dec_log(d: Optional[Decimal]) -> str:
                    return str(d.normalize()) if d is not None and d.is_finite() else 'N/A'

                amt_step_str = fmt_dec_log(std_market['amount_precision_step_decimal'])
                price_step_str = fmt_dec_log(std_market['price_precision_step_decimal'])
                min_amt_str = fmt_dec_log(std_market['min_amount_decimal'])
                max_amt_str = fmt_dec_log(std_market['max_amount_decimal'])
                min_cost_str = fmt_dec_log(std_market['min_cost_decimal'])
                max_cost_str = fmt_dec_log(std_market['max_cost_decimal'])
                contr_size_str = fmt_dec_log(std_market['contract_size_decimal'])
                active_status = std_market.get('active', 'Unknown') # Handle potentially missing 'active' key

                log_msg = (
                    f"Market Details Parsed ({symbol}): Type={std_market['contract_type_str']}, Active={active_status}\n"
                    f"  Precision: Amount Step={amt_step_str}, Price Step={price_step_str}\n"
                    f"  Limits Amt (Min/Max): {min_amt_str} / {max_amt_str}\n"
                    f"  Limits Cost(Min/Max): {min_cost_str} / {max_cost_str}"
                )
                if std_market['is_contract']:
                     log_msg += f"\n  Contract Size: {contr_size_str}"
                lg.info(log_msg) # Log at INFO level as it's important setup info

                # --- Cast to TypedDict and Return ---
                try:
                    # Attempt to cast the enhanced dictionary to the MarketInfo type
                    # This primarily serves static analysis; runtime check is implicit
                    final_market_info: MarketInfo = std_market # type: ignore [assignment]
                    return final_market_info
                except Exception as cast_err:
                    # Should not happen if MarketInfo matches the dict structure, but catch just in case
                    lg.error(f"Internal error casting market dictionary to MarketInfo type ({symbol}): {cast_err}. Returning raw dict cautiously.")
                    return std_market # type: ignore [return-value] # Return the dict anyway

            # else: Market dictionary was not found or retrieved, loop will retry or fail.

        # --- Error Handling for the Loop Iteration ---
        except ccxt.BadSymbol as e:
            # This should have been caught during the refresh attempt, but handle here for robustness
            lg.error(f"Symbol '{symbol}' confirmed invalid on {exchange.id}: {e}")
            return None # Non-retryable
        except ValueError as e: # Catch our critical validation errors
             lg.error(f"Market info validation failed for {symbol}: {e}", exc_info=True)
             return None # Treat validation errors as fatal for this symbol
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError) as e:
            # Handle errors that might occur during market refresh attempt
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network/Exchange error retrieving market info/refreshing ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error retrieving market info: {e}. Cannot continue.{RESET}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error retrieving market info ({symbol}): {e}{RESET}", exc_info=True)
            return None # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to get market info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency from the exchange.
    Attempts to handle different account structures (e.g., Bybit V5 Unified/Contract/Spot).
    Includes retry logic and robust parsing to Decimal.

    Args:
        exchange: The initialized ccxt exchange instance.
        currency: The currency code (e.g., 'USDT', 'BTC'). Case-sensitive, usually uppercase.
        logger: The logger instance for messages specific to this operation.

    Returns:
        The available balance as a Decimal object if found and valid, otherwise None.

    Raises:
        ccxt.AuthenticationError: If authentication fails during the balance check (re-raised).
    """
    lg = logger
    lg.debug(f"Fetching available balance for currency: {currency}...")
    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None
            balance_source: str = "N/A" # Where the balance was found (e.g., account type, field name)
            found: bool = False
            balance_info: Optional[Dict] = None # Store the last fetched balance structure for debugging
            is_bybit = 'bybit' in exchange.id.lower()

            # For Bybit, try specific account types relevant for trading, then default.
            # Order matters: UNIFIED (if applicable), CONTRACT, SPOT, then default ''.
            # Check exchange specs if UNIFIED covers CONTRACT/SPOT balances automatically.
            # Assuming fetch_balance without params might give combined or default (e.g., SPOT or Unified).
            # Bybit V5 types: UNIFIED, CONTRACT, SPOT, FUND, OPTION
            types_to_check = ['UNIFIED', 'CONTRACT', 'SPOT', ''] if is_bybit else [''] # Default '' works for most exchanges

            for acc_type in types_to_check:
                if not is_bybit and acc_type: continue # Skip specific types for non-bybit

                try:
                    params = {'accountType': acc_type} if acc_type and is_bybit else {} # Only add type for Bybit if specified
                    type_desc = f"Account Type: '{acc_type}'" if acc_type else "Default Account"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")

                    # Use fetch_balance (more common and standardized)
                    balance_info = exchange.fetch_balance(params=params)
                    lg.debug(f"Raw balance response ({type_desc}): {balance_info}") # Log raw structure for debugging

                    # --- Try Standard CCXT Structure ('free' field) ---
                    # Structure: { 'CUR': {'free': X, 'used': Y, 'total': Z}, ... } or { 'free': {'CUR': X}, ... }
                    if currency in balance_info and isinstance(balance_info[currency], dict) and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        balance_source = f"{type_desc} (ccxt '{currency}.free' field)"
                        found = True
                        break # Found balance, exit account type loop
                    elif 'free' in balance_info and isinstance(balance_info['free'], dict) and balance_info['free'].get(currency) is not None:
                         balance_str = str(balance_info['free'][currency])
                         balance_source = f"{type_desc} (ccxt 'free.{currency}' field)"
                         found = True
                         break # Found balance

                    # --- Try Bybit V5 Specific Structure (nested within 'info') ---
                    # Often: info -> result -> list -> [ { accountType, coin: [ { coin, availableToWithdraw/availableBalance } ] } ]
                    elif (is_bybit and 'info' in balance_info and
                          isinstance(balance_info.get('info'), dict) and
                          isinstance(balance_info['info'].get('result'), dict) and
                          isinstance(balance_info['info']['result'].get('list'), list)):

                        lg.debug("Parsing Bybit V5 specific 'info' structure for balance...")
                        for account_details in balance_info['info']['result']['list']:
                            # Check if this entry matches the account type we queried (or if query was default '')
                            # And ensure 'coin' list exists and is a list
                            fetched_acc_type = account_details.get('accountType')
                            # Match if type is specified and matches, OR if default type ('') was queried (accept any type found)
                            type_match = (acc_type and fetched_acc_type == acc_type) or (not acc_type)

                            if type_match and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if isinstance(coin_data, dict) and coin_data.get('coin') == currency:
                                        # Try different fields for available balance in preferred order
                                        val = coin_data.get('availableToWithdraw') # Most preferred (liquid balance)
                                        src = 'availableToWithdraw'
                                        if val is None:
                                            val = coin_data.get('availableBalance') # Next best (might include unrealized PnL?)
                                            src = 'availableBalance'
                                        # WalletBalance often includes frozen/used margin, less useful for placing new orders
                                        # if val is None:
                                        #      val = coin_data.get('walletBalance')
                                        #      src = 'walletBalance'

                                        if val is not None:
                                            balance_str = str(val)
                                            actual_source_type = fetched_acc_type or 'Default/Unknown'
                                            balance_source = f"Bybit V5 info ({actual_source_type}, field: '{src}')"
                                            found = True
                                            break # Found coin data, exit coin loop
                                if found: break # Exit account details loop
                        if found: break # Exit account type loop

                    # If not found in standard or Bybit V5 structure for this account type, continue to next type
                    if not found:
                         lg.debug(f"Balance for '{currency}' not found in expected structures for {type_desc}.")

                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # Bybit might throw specific errors for invalid account types; treat these as non-fatal for the loop
                    if acc_type and ("account type does not exist" in err_str or "invalid account type" in err_str or "3400000" in str(e)): # 3400000 generic param error
                        lg.debug(f"Account type '{acc_type}' not found or invalid for balance check on Bybit. Trying next...")
                        last_exception = e # Keep track of the error
                        continue # Try the next account type
                    else:
                        # Re-raise other exchange errors to be handled by the main handler below
                        raise e
                except Exception as e:
                    # Unexpected error during a specific account type check
                    lg.warning(f"Unexpected error fetching/parsing balance for {type_desc}: {e}. Trying next...", exc_info=True)
                    last_exception = e
                    continue # Try the next account type

            # --- Process Result After Checking All Account Types ---
            if found and balance_str is not None:
                # Use safe decimal conversion, allowing zero but not negative
                bal_dec = _safe_market_decimal(balance_str, f"balance_str({currency}) from {balance_source}", allow_zero=True, allow_negative=False)

                if bal_dec is not None:
                    lg.info(f"Successfully parsed balance for {currency} from {balance_source}: {bal_dec.normalize()}")
                    return bal_dec
                else:
                    # If conversion fails despite finding a string, it indicates bad data from API
                    raise ccxt.ExchangeError(f"Failed to convert seemingly valid balance string '{balance_str}' from {balance_source} to non-negative Decimal for {currency}.")
            elif not found:
                # Currency not found in any checked structure after trying all account types
                lg.debug(f"Balance information for currency '{currency}' not found in any checked response structure(s).")
                # Set a specific error if not already set
                if last_exception is None:
                    last_exception = ccxt.ExchangeError(f"Balance for '{currency}' not found in response structures after checking all account types.")
                # Continue to retry logic below

        # --- Error Handling for fetch_balance call (outer loop) ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2)
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance ({currency}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # This is critical and non-retryable. Re-raise to be caught by the caller.
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching balance: {e}. Cannot continue.{RESET}")
            raise e
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., temporary issues, maintenance)
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance ({currency}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as potentially fatal for balance check stability
            return None

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> Optional[PositionInfo]:
    """
    Fetches and standardizes the currently open position for a specific contract symbol.
    Returns None if no position exists, if the symbol is not a contract, or if an error occurs.
    Parses key values to Decimal and includes retry logic.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        market_info: The corresponding MarketInfo dictionary for the symbol (must be valid).
        logger: The logger instance for messages specific to this operation.

    Returns:
        A PositionInfo TypedDict if an active position (size != 0) exists, otherwise None.
    """
    lg = logger

    # --- Pre-checks ---
    if not market_info.get('is_contract'):
        lg.debug(f"Position check skipped for {symbol}: Market type is '{market_info.get('contract_type_str', 'Unknown')}', not a contract.")
        return None

    market_id = market_info.get('id') # Exchange-specific ID
    is_bybit = 'bybit' in exchange.id.lower()

    # Determine category for Bybit V5 based on standardized info
    category = 'linear' # Default assumption for contracts
    if market_info.get('is_linear'): category = 'linear'
    elif market_info.get('is_inverse'): category = 'inverse'
    # else: category remains default ('linear') - assumes bot focuses on linear/inverse

    if not market_id:
        lg.error(f"Cannot check position for {symbol}: Invalid or missing market ID in market_info.")
        return None
    if is_bybit and category not in ['linear', 'inverse']:
         # Bybit V5 position endpoints usually require linear or inverse category
         lg.error(f"Cannot check position for Bybit symbol {symbol}: Determined category '{category}' is not 'linear' or 'inverse'. Check market info/config.")
         return None

    lg.debug(f"Checking for open position for {symbol} (Market ID: '{market_id}', Category: '{category if is_bybit else 'N/A'}')...")

    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions ({symbol}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            positions_list: List[Dict] = [] # Initialize as empty list

            # --- Fetch Positions from Exchange ---
            try:
                params = {}
                # Bybit V5 requires category for contract positions
                if is_bybit:
                     params['category'] = category
                     # Filtering by symbol/market_id might be possible/helpful
                     # Bybit V5 fetchPositions can filter by symbol (market_id)
                     params['symbol'] = market_id

                lg.debug(f"Fetching positions with parameters: {params}")

                # Use fetch_positions if available and reliable
                if exchange.has.get('fetchPositions'):
                    # Fetch positions using specified params (which might include symbol filter for Bybit)
                    # Note: Some exchanges might ignore the symbol filter in params and return all positions
                    all_fetched_positions = exchange.fetch_positions(params=params)

                    # Filter results explicitly for the target symbol/market_id, as API filter might not work
                    positions_list = [
                        p for p in all_fetched_positions
                        if p and (p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id)
                    ]
                    lg.debug(f"Fetched {len(all_fetched_positions)} total position(s) via fetch_positions, "
                             f"filtered down to {len(positions_list)} matching symbol '{symbol}' or market ID '{market_id}'.")

                elif exchange.has.get('fetchPosition'):
                     # Fallback to fetchPosition if fetchPositions is not available/supported
                     lg.warning(f"Exchange {exchange.id} lacks fetchPositions. Using fallback fetchPosition for {symbol} (may be less efficient or have different behavior).")
                     # fetchPosition usually requires the symbol argument directly
                     pos_data = exchange.fetch_position(symbol, params=params)
                     # fetch_position typically returns a single dict or raises error if no position
                     # Wrap in list for consistent processing, only if it seems valid
                     if pos_data and isinstance(pos_data, dict):
                          positions_list = [pos_data]
                     else:
                          positions_list = [] # No position found or invalid response
                     lg.debug(f"fetchPosition returned: {'Position data found' if positions_list else 'No position data found'}")
                else:
                    # If neither method is supported, we cannot get position info
                    raise ccxt.NotSupported(f"{exchange.id} does not support fetchPositions or fetchPosition via ccxt.")

            except ccxt.ExchangeError as e:
                 # Specific handling for "position not found" or similar errors which indicate no open position
                 # Bybit V5 retCode: 110025 = position not found / Position is closed
                 common_no_pos_msgs = ["position not found", "no position", "position does not exist", "position is closed", "no active position"]
                 bybit_no_pos_codes = ['110025'] # Bybit V5 specific code

                 err_str = str(e).lower()
                 # Try to extract Bybit retCode if present in the error message/args
                 code_str = ""
                 match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
                 if match: code_str = match.group(2)
                 else: # Fallback to checking attributes on the exception object
                      code_attr = getattr(e, 'code', None) or getattr(e, 'retCode', None)
                      if code_attr is not None: code_str = str(code_attr)

                 is_bybit_no_pos = is_bybit and code_str and any(code_str == code for code in bybit_no_pos_codes)
                 is_common_no_pos = any(msg in err_str for msg in common_no_pos_msgs)

                 if is_bybit_no_pos or is_common_no_pos:
                     lg.info(f"No open position found for {symbol} (API indicated no position: Code='{code_str}', Msg='{err_str[:80]}...').")
                     return None # This is the expected outcome when no position exists, not an error state
                 else:
                     # Re-raise other exchange errors to be handled below
                     raise e

            # --- Process Filtered Positions to Find the Active One ---
            active_raw_position: Optional[Dict] = None

            # Define a small threshold for position size based on market amount precision
            # This helps ignore dust positions left over from previous trades.
            size_threshold = Decimal('1e-9') # Default tiny threshold if precision unavailable
            amt_step = market_info.get('amount_precision_step_decimal')
            if amt_step and amt_step > 0:
                # Use a fraction of the step size (e.g., half a step) as the threshold
                size_threshold = amt_step / Decimal('2')
            lg.debug(f"Using position size threshold > {size_threshold.normalize()} for {symbol}.")

            # Iterate through the list of positions matching the symbol
            for pos_data in positions_list:
                # Try to get size from 'info' (often more reliable/direct, e.g., Bybit 'size')
                # Fallback to standard 'contracts' field if 'info.size' not found
                size_raw = pos_data.get('info', {}).get('size')
                if size_raw is None: size_raw = pos_data.get('contracts')

                # Safely convert size to Decimal, allowing zero and negative values
                size_decimal = _safe_market_decimal(size_raw, f"{symbol} pos.size/contracts", allow_zero=True, allow_negative=True)

                if size_decimal is None:
                    lg.debug(f"Skipping position data entry with missing or invalid size field ({symbol}). Raw size: {repr(size_raw)}")
                    continue

                # Check if the *absolute* size exceeds the threshold (effectively non-zero)
                if abs(size_decimal) > size_threshold:
                    if active_raw_position is not None:
                         # This indicates multiple active positions for the same symbol, possibly hedge mode or API error
                         lg.warning(f"Multiple active positions found for {symbol} (One-Way mode assumed)! Using the first one found. Position 1 Size: {active_raw_position.get('size_decimal', 'N/A')}, Position 2 Size: {size_decimal}. Check exchange/position mode.")
                         # Stick with the first one found for now
                    else:
                         active_raw_position = pos_data
                         # Store the parsed Decimal size directly in the dict for standardization
                         active_raw_position['size_decimal'] = size_decimal
                         lg.debug(f"Found active position candidate for {symbol} with size: {size_decimal.normalize()}")
                         # Don't break here if multiple positions might exist (e.g., hedge mode),
                         # but the warning above will fire. If strict one-way expected, could break.
                else:
                     lg.debug(f"Ignoring position data entry with size near zero ({symbol}, Size: {size_decimal.normalize()}).")

            # --- Standardize and Return Active Position Info ---
            if active_raw_position:
                std_pos = active_raw_position.copy()
                info_dict = std_pos.get('info', {}) # Raw exchange-specific data for fallbacks

                # Ensure critical size_decimal is present
                parsed_size = std_pos.get('size_decimal')
                if parsed_size is None: # Should not happen if active_raw_position was set
                    lg.error(f"Internal error: Active position found for {symbol} but parsed size is missing.")
                    return None

                # Determine Side (long/short) - crucial and sometimes inconsistent across exchanges/endpoints
                side = std_pos.get('side') # Standard ccxt field

                # Infer side if standard field is missing, ambiguous, or potentially incorrect
                if side not in ['long', 'short']:
                    inferred_side = None
                    # Try inferring from Bybit V5 'info.side' ('Buy'/'Sell')
                    side_v5 = str(info_dict.get('side', '')).strip().lower()
                    if side_v5 == 'buy': inferred_side = 'long'
                    elif side_v5 == 'sell': inferred_side = 'short'

                    # If still no side, infer from the sign of the size
                    if inferred_side is None:
                        if parsed_size > size_threshold: inferred_side = 'long'
                        elif parsed_size < -size_threshold: inferred_side = 'short'

                    if inferred_side:
                         if side and side != inferred_side: # Log if standard field contradicts inferred
                              lg.warning(f"Inconsistent side info for {symbol}: Standard field='{side}', Inferred='{inferred_side}'. Using inferred side.")
                         side = inferred_side
                    else:
                         # Cannot determine side even from size
                         lg.error(f"Could not determine position side for {symbol}. Standard field='{side}', Size='{parsed_size}'. Raw Info: {info_dict}")
                         return None # Cannot proceed without knowing the side

                std_pos['side'] = side # Store the determined side

                # Safely parse other relevant fields to Decimal where applicable
                # Prefer standard ccxt fields, fallback to common 'info' dict fields if needed
                std_pos['entryPrice_decimal'] = _safe_market_decimal(
                    std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice'), # Bybit V5 uses avgPrice in info
                    f"{symbol} pos.entry", allow_zero=False, allow_negative=False) # Entry price must be positive
                std_pos['leverage_decimal'] = _safe_market_decimal(
                    std_pos.get('leverage') or info_dict.get('leverage'),
                    f"{symbol} pos.leverage", allow_zero=False, allow_negative=False) # Leverage > 0
                std_pos['liquidationPrice_decimal'] = _safe_market_decimal(
                    std_pos.get('liquidationPrice') or info_dict.get('liqPrice'), # Bybit V5 uses liqPrice
                    f"{symbol} pos.liq", allow_zero=False, allow_negative=False) # Liq price > 0 (usually)
                std_pos['markPrice_decimal'] = _safe_market_decimal(
                    std_pos.get('markPrice') or info_dict.get('markPrice'),
                    f"{symbol} pos.mark", allow_zero=False, allow_negative=False) # Mark price > 0
                std_pos['unrealizedPnl_decimal'] = _safe_market_decimal(
                    std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl'), # Bybit V5 spelling
                    f"{symbol} pos.pnl", allow_zero=True, allow_negative=True) # PnL can be zero or negative
                std_pos['notional_decimal'] = _safe_market_decimal(
                    std_pos.get('notional') or info_dict.get('positionValue'), # Bybit V5 uses positionValue
                    f"{symbol} pos.notional", allow_zero=True, allow_negative=False) # Notional value >= 0
                std_pos['collateral_decimal'] = _safe_market_decimal(
                    std_pos.get('collateral') or info_dict.get('positionIM') or info_dict.get('collateral'), # Bybit V5 uses positionIM (Initial Margin) sometimes
                    f"{symbol} pos.collateral", allow_zero=True, allow_negative=False) # Collateral >= 0
                std_pos['initialMargin_decimal'] = _safe_market_decimal(
                    std_pos.get('initialMargin') or info_dict.get('positionIM'),
                    f"{symbol} pos.initialMargin", allow_zero=True, allow_negative=False)
                std_pos['maintenanceMargin_decimal'] = _safe_market_decimal(
                    std_pos.get('maintenanceMargin') or info_dict.get('positionMM'), # Bybit V5 uses positionMM
                    f"{symbol} pos.maintMargin", allow_zero=True, allow_negative=False)


                # Extract protection orders (SL, TP, TSL) - these are often strings in 'info' or root
                # Check both root level and info dict. Value '0' or '0.0' means not set.
                def get_protection_value(field_names: List[str]) -> Optional[str]:
                    """Safely gets a protection order value from root or info, returns None if zero/empty/invalid."""
                    value_str: Optional[str] = None
                    raw_value: Optional[Any] = None
                    for name in field_names:
                        raw_value = std_pos.get(name) # Check root level first
                        if raw_value is None: raw_value = info_dict.get(name) # Check info dict
                        if raw_value is not None: break # Found a value

                    if raw_value is None: return None
                    value_str = str(raw_value).strip()

                    # Check if it represents a valid, non-zero price/value using safe decimal conversion
                    dec_val = _safe_market_decimal(value_str, f"{symbol} prot.{'/'.join(field_names)}", allow_zero=False, allow_negative=False)
                    if dec_val is not None:
                         return value_str # Return the original string if it represents a valid, non-zero value
                    else:
                         return None # Treat '0', '0.0', empty string, or invalid as no active order set

                std_pos['stopLossPrice'] = get_protection_value(['stopLoss', 'stopLossPrice'])
                std_pos['takeProfitPrice'] = get_protection_value(['takeProfit', 'takeProfitPrice'])
                # Bybit V5 TSL fields: trailingStop (distance/offset string), activePrice (activation price string)
                std_pos['trailingStopLoss'] = get_protection_value(['trailingStop']) # Check only 'trailingStop'
                std_pos['tslActivationPrice'] = get_protection_value(['activePrice']) # Check only 'activePrice'

                # Initialize bot state tracking fields (these will be updated by bot logic if needed)
                # Default to False, but sync with exchange state if TSL detected active
                std_pos['be_activated'] = False # Bot has not activated BE for this instance yet
                exchange_tsl_active = bool(std_pos['trailingStopLoss']) and bool(std_pos['tslActivationPrice'])
                std_pos['tsl_activated'] = exchange_tsl_active # Reflect if TSL seems active on exchange

                # --- Log Found Position Details ---
                # Helper for logging optional Decimal values safely
                def fmt_log(val: Optional[Decimal]) -> str:
                    return val.normalize() if val is not None else 'N/A'

                ep = fmt_log(std_pos['entryPrice_decimal'])
                sz = std_pos['size_decimal'].normalize()
                sl = std_pos.get('stopLossPrice') or 'N/A'
                tp = std_pos.get('takeProfitPrice') or 'N/A'
                tsl_dist = std_pos.get('trailingStopLoss') or 'N/A'
                tsl_act = std_pos.get('tslActivationPrice') or 'N/A'
                tsl_str = "Inactive"
                if exchange_tsl_active:
                     tsl_str = f"ACTIVE (Dist/Offset={tsl_dist} | ActPrice={tsl_act})"
                elif std_pos.get('trailingStopLoss') or std_pos.get('tslActivationPrice'): # Partially set?
                     tsl_str = f"PARTIAL? (Dist/Offset={tsl_dist} | ActPrice={tsl_act})"

                pnl = fmt_log(std_pos['unrealizedPnl_decimal'])
                liq = fmt_log(std_pos['liquidationPrice_decimal'])
                lev = fmt_log(std_pos['leverage_decimal'])
                notional = fmt_log(std_pos['notional_decimal'])

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET}\n"
                        f"  Size={sz}, Entry={ep}, Mark={fmt_log(std_pos['markPrice_decimal'])}, Notional={notional}\n"
                        f"  Liq={liq}, Leverage={lev}x, MarginMode={std_pos.get('marginMode', 'N/A')}\n"
                        f"  Unrealized PnL: {pnl}\n"
                        f"  Protections (from exchange): SL={sl}, TP={tp}, TSL={tsl_str}")

                # --- Cast to TypedDict and Return ---
                try:
                    final_position_info: PositionInfo = std_pos # type: ignore [assignment]
                    return final_position_info
                except Exception as cast_err:
                    lg.error(f"Internal error casting position dictionary to PositionInfo type ({symbol}): {cast_err}. Returning raw dict cautiously.")
                    return std_pos # type: ignore [return-value] # Return the dict anyway

            else:
                # No position found with size > threshold after checking all filtered entries
                lg.info(f"No active position found for {symbol} (checked {len(positions_list)} potential entries).")
                return None

        # --- Error Handling for the Loop Iteration ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2)
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching positions: {e}. Cannot continue.{RESET}")
            return None
        except ccxt.NotSupported as e:
             # Fatal error for this function
             last_exception = e
             lg.error(f"{NEON_RED}Position fetching method not supported by {exchange.id}: {e}. Cannot get position info.{RESET}")
             return None
        except ccxt.ExchangeError as e:
            # Handled specific "no position" cases above, this catches other exchange errors
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors during position fetching/processing
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching/processing positions ({symbol}): {e}{RESET}", exc_info=True)
            return None # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a given contract symbol using ccxt's `set_leverage` method.
    Handles specific requirements for exchanges like Bybit V5 (category, buy/sell leverage).
    Includes retry logic and checks for success, no change needed, or fatal errors.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        leverage: The desired integer leverage level (e.g., 10 for 10x). Must be positive.
        market_info: The MarketInfo dictionary for the symbol (must be valid).
        logger: The logger instance for messages specific to this operation.

    Returns:
        True if leverage was set successfully or already set to the desired value, False otherwise.
    """
    lg = logger

    # --- Pre-checks ---
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True # No action needed for non-contracts, considered success

    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Leverage setting failed ({symbol}): Invalid leverage value '{leverage}'. Must be a positive integer.")
        return False

    # Check if the exchange supports setting leverage via the ccxt method
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.warning(f"Leverage setting might not be supported via ccxt's setLeverage method for {exchange.id}. Attempting anyway, but may fail or require manual configuration.")
        # Proceed cautiously, but warn the user.

    market_id = market_info.get('id') # Exchange-specific ID
    is_bybit = 'bybit' in exchange.id.lower()

    # Determine category for Bybit V5
    category = 'linear' # Default guess
    if market_info.get('is_linear'): category = 'linear'
    elif market_info.get('is_inverse'): category = 'inverse'

    if not market_id:
         lg.error(f"Leverage setting failed ({symbol}): Market ID missing in market_info.")
         return False
    if is_bybit and category not in ['linear', 'inverse']:
         lg.error(f"Leverage setting failed for Bybit symbol {symbol}: Invalid category '{category}'. Must be 'linear' or 'inverse'.")
         return False

    lg.info(f"Attempting to set leverage for {symbol} (Market ID: {market_id}, Category: {category if is_bybit else 'N/A'}) to {leverage}x...")

    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"set_leverage call attempt {attempts + 1}/{MAX_API_RETRIES + 1} for {symbol} to {leverage}x...")
            params = {}

            # --- Exchange-Specific Parameter Handling ---
            if is_bybit:
                 # Bybit V5 requires category and separate buy/sell leverage as strings in params
                 params = {
                     'category': category,
                     'buyLeverage': str(leverage), # Must be strings
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 specific leverage parameters: {params}")

            # --- Execute set_leverage Call ---
            # Note: ccxt `set_leverage` takes leverage as float/int, symbol, and optional params dict
            response = exchange.set_leverage(leverage=float(leverage), symbol=symbol, params=params) # Use float leverage here
            lg.debug(f"Raw response from set_leverage ({symbol}): {response}")

            # --- Response Validation (especially for Bybit V5) ---
            # Success is often indicated by the absence of an exception.
            # However, some exchanges return codes/messages we can check.
            ret_code_str: Optional[str] = None
            ret_msg: str = "N/A"

            if isinstance(response, dict):
                # Try extracting Bybit V5 style response codes/messages from 'info' first
                info_dict = response.get('info', {})
                raw_code = info_dict.get('retCode') # Primary location in V5 response
                if raw_code is None: raw_code = response.get('retCode') # Fallback to root level
                ret_code_str = str(raw_code) if raw_code is not None else None
                ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown message'))

            # Bybit V5 Success Code: 0
            # Bybit V5 "Leverage not modified" Code: 110045 (often accompanied by "Parameter error" message)
            bybit_success_codes = ['0']
            bybit_no_change_codes = ['110045']

            if ret_code_str in bybit_success_codes:
                lg.info(f"{NEON_GREEN}Leverage successfully set for {symbol} to {leverage}x (Code: {ret_code_str}).{RESET}")
                return True
            elif ret_code_str in bybit_no_change_codes and ("leverage not modified" in ret_msg.lower() or "same leverage" in ret_msg.lower()):
                lg.info(f"{NEON_YELLOW}Leverage for {symbol} is already {leverage}x (Code: {ret_code_str} - Not Modified). Success.{RESET}")
                return True
            elif response is not None:
                 # If no specific error code checked or found, assume success if no exception was raised
                 lg.info(f"{NEON_GREEN}Leverage set/confirmed for {symbol} to {leverage}x (No specific code checked/found, assumed success).{RESET}")
                 return True
            else:
                 # Response was None or empty, which is unexpected if no exception occurred
                 raise ccxt.ExchangeError(f"Received unexpected empty response after setting leverage for {symbol}.")

        # --- Error Handling for set_leverage call ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str_lower = str(e).lower()
            # Try to extract error code again for detailed logging/decision making
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            else: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

            lg.error(f"{NEON_RED}Exchange error setting leverage ({symbol} to {leverage}x): {e} (Code: {err_code_str}){RESET}")

            # Check if the error indicates leverage was already set (redundant check, but safe)
            if err_code_str in bybit_no_change_codes and ("leverage not modified" in err_str_lower or "same leverage" in err_str_lower()):
                lg.info(f"{NEON_YELLOW}Leverage already set to {leverage}x (confirmed via error response {err_code_str}). Success.{RESET}")
                return True

            # Check for known fatal/non-retryable error codes or messages
            # (Adjust based on exchange - using Bybit V5 examples)
            fatal_codes = [
                '10001', # Parameter error (e.g., leverage value invalid for symbol)
                '10004', # Sign check error (API keys)
                '110013',# Risk limit error
                '110028',# Cross margin mode cannot modify leverage individually
                '110043',# Cannot set leverage when position exists (Isolated margin)
                '110044',# Cannot set leverage when order exists (Isolated margin)
                '110055',# Cannot set leverage under Isolated for cross margin position
                '3400045',# Leverage outside min/max limits
                '110066', # Cannot set leverage under Portfolio Margin
                '110076', # Leverage reduction is restricted when position exists
            ]
            fatal_messages = [
                "margin mode", "position exists", "order exists", "risk limit", "parameter error",
                "insufficient available balance", "invalid leverage", "leverage exceed",
                "isolated margin", "portfolio margin", "api key", "authentication failed"
            ]
            is_fatal_code = err_code_str in fatal_codes
            is_fatal_message = any(msg in err_str_lower for msg in fatal_messages)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE leverage error for {symbol}. Aborting leverage setting.{RESET}")
                # Provide more specific advice based on common errors
                if any(code in err_code_str for code in ['110043', '110044', '110076']) or any(s in err_str_lower for s in ["position exists", "order exists"]):
                    lg.error(" >> Cannot change leverage while a position or active orders exist (especially in Isolated Margin). Close position/orders first.")
                elif any(s in err_str_lower for s in ["margin mode", "cross margin", "isolated margin", "portfolio margin"]) or any(code in err_code_str for code in ['110028', '110055', '110066']):
                     lg.error(" >> Leverage change might conflict with current margin mode (Cross/Isolated/Portfolio) or account settings.")
                elif any(s in err_str_lower for s in ["parameter error", "invalid leverage", "leverage exceed"]) or any(code in err_code_str for code in ['10001', '3400045']):
                     lg.error(f" >> Leverage value {leverage}x might be invalid or outside allowed limits for {symbol}. Check exchange rules.")

                return False # Non-retryable failure

            # If not identified as fatal, proceed to retry logic below

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting leverage ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error setting leverage ({symbol}): {e}. Cannot continue.{RESET}")
            return False
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting leverage ({symbol}): {e}{RESET}", exc_info=True)
            return False # Treat unexpected errors as fatal

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return False

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: MarketInfo,
    exchange: ccxt.Exchange, # Keep exchange instance for potential future use (e.g., fetching quote price for complex calcs)
    logger: logging.Logger
) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on available balance, risk percentage,
    entry price, and stop loss price. Adheres to market constraints (min/max size, step size,
    min/max cost) and accounts for contract type (linear/inverse). Rounds the final size
    down to the nearest valid amount step.

    Args:
        balance: Available trading balance in the quote currency (Decimal).
        risk_per_trade: The fraction of the balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: The calculated initial stop loss price (Decimal, must be positive).
        entry_price: The estimated or actual entry price (Decimal, must be positive).
        market_info: The MarketInfo dictionary for the symbol (must contain valid precision/limits).
        exchange: The ccxt exchange instance (currently unused but kept for signature consistency).
        logger: The logger instance for messages specific to this calculation.

    Returns:
        The calculated and adjusted position size as a Decimal (in base currency for spot,
        or number of contracts for futures), rounded DOWN to the correct precision step.
        Returns None if calculation fails due to invalid inputs, constraints, or mathematical errors.
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE') # Fallback if missing
    base_currency = market_info.get('base', 'BASE')   # Fallback if missing
    is_inverse = market_info.get('is_inverse', False)
    is_spot = market_info.get('spot', False)
    # Determine the unit of the calculated size for logging/clarity
    size_unit = base_currency if is_spot else "Contracts"

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or balance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Balance is zero, negative, or invalid ({balance} {quote_currency}).")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade)) # Convert float risk to Decimal
        # Risk must be strictly positive and less than or equal to 1 (100%)
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
             raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive).")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None

    # Ensure prices are valid positive Decimals
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive Stop Loss price ({initial_stop_loss_price}).")
        return None
    if not isinstance(entry_price, Decimal) or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive Entry price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): Entry price ({entry_price.normalize()}) and Stop Loss price ({initial_stop_loss_price.normalize()}) cannot be the same.")
        return None
    # Ensure SL is on the correct side of entry
    if (entry_price > initial_stop_loss_price and entry_price < initial_stop_loss_price) or \
       (entry_price < initial_stop_loss_price and entry_price > initial_stop_loss_price):
        # This check seems redundant if SL != entry is already checked, but kept for clarity
        # It should actually check if SL is below entry for long, above for short. Assumes side later.
        pass # Check will be implicit in stop_loss_distance calculation

    # --- Extract and Validate Market Constraints ---
    try:
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal'] # Used for logging/adjustments
        min_amount = market_info['min_amount_decimal'] # Can be None if not specified by exchange
        max_amount = market_info['max_amount_decimal'] # Can be None
        min_cost = market_info['min_cost_decimal']     # Can be None
        max_cost = market_info['max_cost_decimal']     # Can be None
        contract_size = market_info['contract_size_decimal'] # Should default to 1 if missing

        # Validate critical constraints needed for calculation and adjustment
        if not (amount_step and amount_step > 0): raise ValueError("Amount precision step (amount_step) is missing or invalid.")
        if not (price_step and price_step > 0): raise ValueError("Price precision step (price_step) is missing or invalid.")
        if not (contract_size and contract_size > 0): raise ValueError("Contract size (contract_size) is missing or invalid.")

        # Use effective limits for calculations, treating None as non-restrictive (0 or infinity)
        min_amount_eff = min_amount if min_amount is not None and min_amount >= 0 else Decimal('0')
        max_amount_eff = max_amount if max_amount is not None and max_amount > 0 else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None and min_cost >= 0 else Decimal('0')
        max_cost_eff = max_cost if max_cost is not None and max_cost > 0 else Decimal('inf')

        # Helper for logging optional Decimals
        def fmt_dec_log_size(d: Optional[Decimal]) -> str: return str(d.normalize()) if d is not None else 'N/A'
        lg.debug(f"  Market Constraints ({symbol}):")
        lg.debug(f"    Amount Step: {fmt_dec_log_size(amount_step)}, Min Amount: {fmt_dec_log_size(min_amount)}, Max Amount: {fmt_dec_log_size(max_amount)}")
        lg.debug(f"    Price Step : {fmt_dec_log_size(price_step)}")
        lg.debug(f"    Cost Min   : {fmt_dec_log_size(min_cost)}, Cost Max: {fmt_dec_log_size(max_cost)}")
        lg.debug(f"    Contract Size: {fmt_dec_log_size(contract_size)}, Type: {market_info['contract_type_str']}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error accessing or validating required market details from market_info: {e}")
        lg.debug(f"  Problematic MarketInfo: {market_info}")
        return None

    # --- Core Size Calculation ---
    # Calculate risk amount in quote currency, quantize early to avoid minor precision diffs
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN) # Round down risk amount slightly for safety
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance <= Decimal('0'): # Should be caught earlier, but safeguard
        lg.error(f"Sizing failed ({symbol}): Stop loss distance is zero or negative ({stop_loss_distance}).")
        return None

    lg.info(f"  Inputs:")
    lg.info(f"    Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"    Risk % : {risk_decimal:.2%}")
    lg.info(f"    Risk Amt: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"    Entry Price: {entry_price.normalize()}")
    lg.info(f"    Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"    SL Distance (Price Points): {stop_loss_distance.normalize()}")

    calculated_size = Decimal('NaN') # Initialize as NaN
    try:
        if not is_inverse:
            # --- Linear Contract or Spot ---
            # Risk per unit = SL Distance (Quote) * ContractSize (Base/Quote or 1 for spot)
            # Size (Base/Contracts) = Risk Amount (Quote) / Risk per unit (Quote)
            risk_per_unit = stop_loss_distance * contract_size
            if risk_per_unit <= Decimal('1e-18'): # Avoid division by zero or near-zero
                raise ZeroDivisionError(f"Calculated risk per unit ({risk_per_unit}) is near zero. Check prices/contract size.")
            calculated_size = risk_amount_quote / risk_per_unit
            lg.debug(f"  Linear/Spot Calculation: Size = {risk_amount_quote.normalize()} / ({stop_loss_distance.normalize()} * {contract_size.normalize()}) = {calculated_size}")
        else:
            # --- Inverse Contract ---
            # Risk per contract = ContractSize (USD/Contract) * |(1/Entry) - (1/SL)| (Value change in Base per Contract)
            # Size (Contracts) = Risk Amount (Quote) / (Risk per contract (Base) * EntryPrice (Quote/Base)) <- Converts risk per contract to quote terms
            # Simplified: Size (Contracts) = (Risk Amount (Quote) * Entry Price (Quote/Base)) / (ContractSize (USD/Contract) * SL Distance (Quote))
            # Let's use the formula derived from value change: Size = Risk Amount / (ContractSize * |(1/Entry) - (1/SL)|)
            # This calculates size directly in contracts.
            inverse_factor = abs((Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price))
            if inverse_factor <= Decimal('1e-18'):
                raise ZeroDivisionError(f"Calculated inverse factor |(1/E)-(1/SL)| ({inverse_factor}) is near zero. Check prices.")
            risk_per_contract_base_terms = contract_size * inverse_factor # This is risk in BASE currency terms per contract
            if risk_per_contract_base_terms <= Decimal('1e-18'):
                 raise ZeroDivisionError(f"Calculated risk per contract in base terms ({risk_per_contract_base_terms}) is near zero.")
            # To get size in contracts using quote risk amount, convert risk per contract to quote terms:
            # risk_per_contract_quote_terms = risk_per_contract_base_terms * entry_price # Approx value change in quote per contract
            # calculated_size = risk_amount_quote / risk_per_contract_quote_terms
            # OR use the direct formula: Size = Risk Amount / (ContractSize * |1/Entry - 1/SL|) <- This seems wrong dimensionally.
            # Let's re-derive: Risk = Size * ContractSize * |1/SL - 1/Entry|. Units: Quote = Contracts * (Base*Quote/Contract) * |1/Quote - 1/Quote| ?? No.
            # Let's use: AmountToRisk(Quote) = PosSize(Contracts) * ContractValue(Quote) * | Entry - SL | / Entry
            # For Inverse: ContractValue(Quote) = ContractSize(Base) * EntryPrice(Quote/Base)
            # Risk(Quote) = Size(Contracts) * ContractSize(Base) * EntryPrice(Quote/Base) * SL_Dist(Quote) / EntryPrice(Quote/Base)
            # Risk(Quote) = Size(Contracts) * ContractSize(Base) * SL_Dist(Quote)
            # Size (Contracts) = Risk Amount (Quote) / (Contract Size (Base) * SL Distance (Quote))
            # This looks like the Linear formula but ContractSize has different units (Base vs Base/Quote).
            # Let's stick to the value change per contract approach:
            # Value change per contract in quote = ContractSize * | Entry/Entry - Entry/SL | = ContractSize * Entry * |1/Entry - 1/SL| -- Seems wrong.
            # Value change per contract in BASE = ContractSize * |1/Entry - 1/SL|
            # Value change per contract in QUOTE = Value change per contract in BASE * EntryPrice (approx)
            # Size = Risk Amount (Quote) / (Value change per contract in QUOTE)
            risk_per_contract_quote_approx = risk_per_contract_base_terms * entry_price
            if risk_per_contract_quote_approx <= Decimal('1e-18'):
                 raise ZeroDivisionError(f"Calculated approximate risk per contract in quote terms ({risk_per_contract_quote_approx}) is near zero.")
            calculated_size = risk_amount_quote / risk_per_contract_quote_approx

            lg.debug(f"  Inverse Calculation (Approx): Size = RiskAmt / (ContractSize * |1/E - 1/SL| * EntryPrice)")
            lg.debug(f"  = {risk_amount_quote.normalize()} / ({contract_size.normalize()} * {inverse_factor} * {entry_price.normalize()})")
            lg.debug(f"  = {risk_amount_quote.normalize()} / ({risk_per_contract_quote_approx}) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as e:
        lg.error(f"Sizing failed ({symbol}): Mathematical error during core calculation: {e}.")
        return None

    # Ensure calculated size is a valid positive number before proceeding
    if not calculated_size.is_finite() or calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial calculated size is zero, negative, or invalid ({calculated_size}).")
        lg.debug(f"  Check inputs: RiskAmt={risk_amount_quote}, SLDist={stop_loss_distance}, CtrSize={contract_size}, Inverse={is_inverse}")
        return None

    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Adjust Size Based on Constraints ---
    adjusted_size = calculated_size
    adjustment_reason = [] # Track reasons for adjustments

    # Helper to estimate cost accurately based on contract type
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        """Estimates the order cost in quote currency."""
        if not isinstance(size, Decimal) or not isinstance(price, Decimal) or not size.is_finite() or not price.is_finite() or price <= 0 or size <= 0:
            lg.warning(f"Cost estimation skipped: Invalid size ({size}) or price ({price}).")
            return None
        try:
            cost: Decimal
            if not is_inverse: # Linear / Spot
                # Cost (Quote) = Size (Contracts/Base) * ContractSize (Base/Quote or 1) * EntryPrice (Quote/Base)
                cost = size * contract_size * price
            else: # Inverse
                # Cost (Margin required in Quote) = Position Value (Base) / Leverage
                # Position Value (Base) = Size (Contracts) * ContractSize (Base/Contract)
                # Let's estimate the notional value in Quote instead, as 'cost' limits usually apply to notional
                # Notional Value (Quote) = Size (Contracts) * ContractSize (Base/Contract) * Price (Quote/Base)
                # This looks the same as Linear. Let's re-verify Bybit's definition.
                # Bybit: Order Cost = Order Quantity * Order Price (Spot)
                # Bybit: Order Cost (Margin) = OrderValue / Leverage = (Qty * Price * Multiplier) / Leverage (Linear)
                # Bybit: Order Cost (Margin) = OrderValue / Leverage = (Qty * Multiplier / Price) / Leverage (Inverse)
                # Min/Max Cost limits likely apply to OrderValue (Notional) before leverage.
                # Let's assume Cost Limit applies to Notional Value in Quote.
                if price <= 0: raise ZeroDivisionError("Entry price must be positive for cost estimation.")
                cost = (size * contract_size * price) if not is_inverse else (size * contract_size / price) # Revisit this inverse cost calc if needed based on exchange definition
                # Simpler: Assume cost limit applies to Notional in Quote for both types
                cost = size * contract_size * price

            # Quantize cost to a reasonable precision (e.g., 8 decimal places) for checks
            return cost.quantize(Decimal('1e-8'), ROUND_UP) # Round up cost estimate slightly for safety
        except (InvalidOperation, OverflowError, ZeroDivisionError) as cost_err:
            lg.error(f"Cost estimation failed: {cost_err} (Size: {size}, Price: {price}, CtrSize: {contract_size}, Inverse: {is_inverse})")
            return None

    # 1. Apply Min/Max Amount Limits
    # Ensure comparison uses effective limits (handling None/infinity)
    if adjusted_size < min_amount_eff:
        adjustment_reason.append(f"Adjusted UP to Min Amount {fmt_dec_log_size(min_amount)}")
        adjusted_size = min_amount_eff
    if adjusted_size > max_amount_eff:
        adjustment_reason.append(f"Adjusted DOWN to Max Amount {fmt_dec_log_size(max_amount)}")
        adjusted_size = max_amount_eff

    # Ensure adjusted size is still positive after min/max amount adjustments
    if not adjusted_size.is_finite() or adjusted_size <= Decimal('0'):
         lg.error(f"Sizing failed ({symbol}): Size became zero, negative or invalid ({adjusted_size}) after applying Amount limits {fmt_dec_log_size(min_amount)}/{fmt_dec_log_size(max_amount)}.")
         return None
    if adjustment_reason: lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit} ({'; '.join(adjustment_reason)})")
    else: lg.debug(f"  Size conforms to Amount Limits.")

    # 2. Apply Min/Max Cost Limits (Requires estimating cost based on size *after* amount limits)
    cost_adjustment_reason = []
    estimated_cost = estimate_cost(adjusted_size, entry_price)

    if estimated_cost is not None:
        lg.debug(f"  Estimated Cost (based on size after amount limits, {symbol}): {estimated_cost.normalize()} {quote_currency}")

        # Check Min Cost
        if estimated_cost < min_cost_eff:
            cost_adjustment_reason.append(f"Estimated cost {estimated_cost.normalize()} < Min Cost {fmt_dec_log_size(min_cost)}")
            # We cannot simply increase size to meet min cost without recalculating risk.
            # If the initial risk-based size results in cost < min_cost, it means the risk % or balance is too low
            # relative to the minimum order value allowed by the exchange.
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Calculated size {adjusted_size.normalize()} results in estimated cost {estimated_cost.normalize()} "
                     f"which is below the minimum required cost {fmt_dec_log_size(min_cost)}. "
                     f"Consider increasing risk % or ensuring sufficient balance.{RESET}")
            return None # Fail because we cannot meet min cost without violating risk parameters

        # Check Max Cost
        if estimated_cost > max_cost_eff:
            cost_adjustment_reason.append(f"Estimated cost {estimated_cost.normalize()} > Max Cost {fmt_dec_log_size(max_cost)}")
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Estimated cost {estimated_cost.normalize()} > Max Cost {fmt_dec_log_size(max_cost)}. Attempting to reduce size to meet Max Cost.{RESET}")
            try:
                # Calculate the theoretical maximum size allowed by max cost
                max_size_for_max_cost: Decimal
                if not is_inverse:
                     denominator = entry_price * contract_size
                     if denominator <= 0: raise ZeroDivisionError("Entry price * contract size is zero or negative.")
                     max_size_for_max_cost = max_cost_eff / denominator
                else:
                     # Assuming Cost = Size * ContractSize * Price (Notional in Quote)
                     denominator = entry_price * contract_size
                     if denominator <= 0: raise ZeroDivisionError("Entry price * contract size is zero or negative.")
                     max_size_for_max_cost = max_cost_eff / denominator
                     # If using Cost = Size*ContractSize/Price:
                     # numerator = max_cost_eff * entry_price
                     # if contract_size <= 0: raise ZeroDivisionError("Contract size is zero or negative.")
                     # max_size_for_max_cost = numerator / contract_size

                if not max_size_for_max_cost.is_finite() or max_size_for_max_cost <= 0:
                     raise ValueError(f"Calculated max size allowed by max cost is non-positive or invalid ({max_size_for_max_cost}).")

                lg.info(f"  Theoretical max size allowed by Max Cost ({symbol}): {max_size_for_max_cost.normalize()} {size_unit}")

                # Adjust size down to the max allowed by cost, but ensure it doesn't go below min_amount
                new_adjusted_size = min(adjusted_size, max_size_for_max_cost) # Take the smaller of current or max allowed by cost
                # Ensure it's still >= min_amount AFTER potentially reducing for max cost
                if new_adjusted_size < min_amount_eff:
                     lg.error(f"{NEON_RED}Sizing failed ({symbol}): Reducing size to meet Max Cost ({fmt_dec_log_size(max_cost)}) would result in size {new_adjusted_size.normalize()} "
                              f"which is below Min Amount ({fmt_dec_log_size(min_amount)}). Cannot satisfy both constraints.{RESET}")
                     return None
                else:
                     adjusted_size = new_adjusted_size
                     cost_adjustment_reason.append(f"Adjusted DOWN to {adjusted_size.normalize()} {size_unit} to meet Max Cost")

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating or applying max size allowed by Max Cost: {e}.{RESET}")
                return None

    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        # Cost limits exist, but we couldn't estimate cost initially (shouldn't happen if inputs are valid)
        lg.warning(f"Could not estimate position cost accurately for {symbol}. Cost limit checks (Min: {fmt_dec_log_size(min_cost)}, Max: {fmt_dec_log_size(max_cost)}) were skipped.")

    if cost_adjustment_reason: lg.debug(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit} ({'; '.join(cost_adjustment_reason)})")
    else: lg.debug(f"  Size conforms to Cost Limits.")


    # 3. Apply Amount Precision (Step Size) - FINAL step, ROUND DOWN
    final_size = adjusted_size
    precision_adjustment_reason = ""
    try:
        if amount_step <= 0: raise ValueError("Amount step size is not positive.")
        # Use quantize with ROUND_DOWN for the final size adjustment
        # final_size = adjusted_size.quantize(amount_step, rounding=ROUND_DOWN) # Doesn't work if step isn't power of 10
        # Correct way: Divide by step, floor the result (get number of steps), multiply back
        num_steps = (adjusted_size / amount_step).to_integral_value(rounding=ROUND_DOWN)
        final_size = num_steps * amount_step

        if final_size != adjusted_size:
            precision_adjustment_reason = f"Rounded DOWN from {adjusted_size.normalize()} to nearest Amount Step {amount_step.normalize()}"
            lg.info(f"Applied amount precision ({symbol}): {precision_adjustment_reason}")
        else:
            lg.debug(f"Size already conforms to amount precision ({symbol}, Step: {amount_step.normalize()}).")

    except (InvalidOperation, ValueError, ZeroDivisionError) as e:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error applying amount precision (step size={amount_step}): {e}.{RESET}")
        return None

    # --- Final Validation after Precision ---
    # Check if final size is positive and finite
    if not final_size.is_finite() or final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size after precision adjustment is zero, negative, or invalid ({final_size}). "
                 f"Original calculated: {calculated_size.normalize()}{RESET}")
        return None

    # Re-check Min Amount (rounding down might violate it if min_amount itself wasn't multiple of step)
    if final_size < min_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is less than Min Amount {fmt_dec_log_size(min_amount)} after applying precision rounding.{RESET}")
        lg.error(f"  This usually means Min Amount is not a multiple of Amount Step, or the calculated size was extremely close to Min Amount.")
        # It's generally unsafe to bump up to min_amount as it changes risk profile. Fail here.
        return None

    # Re-check Max Amount (should be impossible if rounding down, but check for safety)
    if final_size > max_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is greater than Max Amount {fmt_dec_log_size(max_amount)} after precision rounding (unexpected!).{RESET}")
        return None

    # Re-check Cost Limits with the final precise size
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}): {final_cost.normalize()} {quote_currency}")
        # Check Min Cost again (rounding down amount might violate it)
        if final_cost < min_cost_eff:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} < Min Cost {fmt_dec_log_size(min_cost)} after precision adjustment.{RESET}")
            # Fail for safety, as increasing size would violate risk.
            return None
        # Check Max Cost again (unlikely to be violated by rounding down amount, but check)
        elif final_cost > max_cost_eff:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} > Max Cost {fmt_dec_log_size(max_cost)} after precision adjustment (unexpected!).{RESET}")
            return None
    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        # Cost limits exist, but we couldn't estimate final cost (shouldn't happen here)
        lg.warning(f"Could not perform final cost check for {symbol} after precision adjustment. Order might fail if cost limits are violated.")

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    if final_cost:
         lg.info(f"    Estimated Final Cost: {final_cost.normalize()} {quote_currency}")
    if adjustment_reason or cost_adjustment_reason or precision_adjustment_reason:
        lg.info(f"    Adjustments applied: {'; '.join(filter(None, [', '.join(adjustment_reason), ', '.join(cost_adjustment_reason), precision_adjustment_reason]))}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size

def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """
    Cancels a specific open order by its ID using ccxt.cancel_order.
    Includes retry logic and handles common errors like OrderNotFound or InvalidOrder gracefully.
    Passes necessary parameters for exchanges like Bybit V5.

    Args:
        exchange: The initialized ccxt exchange instance.
        order_id: The ID string of the order to cancel.
        symbol: The market symbol associated with the order (required by some exchanges like Bybit).
        logger: The logger instance for messages specific to this operation.

    Returns:
        True if the order was successfully cancelled or confirmed already closed/not found, False otherwise.
    """
    lg = logger
    attempts = 0
    last_exception: Optional[Exception] = None
    lg.info(f"Attempting to cancel order ID '{order_id}' for symbol {symbol}...")

    # Prepare parameters (e.g., category for Bybit V5) - requires market info
    params = {}
    is_bybit = 'bybit' in exchange.id.lower()
    if is_bybit:
        try:
            # Attempt to get market info to determine category and market_id
            market = exchange.market(symbol)
            market_id = market['id']
            category = 'spot' # Default
            if market.get('linear'): category = 'linear'
            elif market.get('inverse'): category = 'inverse'
            elif market.get('spot'): category = 'spot'
            else: category = 'linear' # Fallback guess

            params['category'] = category
            # Bybit V5 cancelOrder might require symbol (market_id) in params, not just as argument
            params['symbol'] = market_id
            lg.debug(f"Using Bybit V5 params for cancelOrder: {params}")
        except Exception as e:
            lg.warning(f"Could not get market details to determine category/market_id for cancelOrder ({symbol}): {e}. Proceeding without specific params.")
            # If market lookup fails, ccxt might still handle it with just the symbol argument

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1}/{MAX_API_RETRIES + 1} for ID {order_id} ({symbol})...")
            # Use standard symbol in the main call, pass specifics in params if needed
            # Ensure order_id is a string
            exchange.cancel_order(str(order_id), symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully requested cancellation for order {order_id} ({symbol}).{RESET}")
            # Note: Success here means the API call succeeded, not necessarily that the order *was* cancelled (it might have filled just before).
            return True

        except ccxt.OrderNotFound:
            # Order doesn't exist - could be already filled, cancelled manually, or wrong ID passed
            lg.warning(f"{NEON_YELLOW}Order ID '{order_id}' ({symbol}) not found on the exchange. Assuming cancellation is effectively complete or unnecessary.{RESET}")
            return True # Treat as success for workflow purposes (the order is not open)
        except ccxt.InvalidOrder as e:
             # E.g., order already filled/cancelled and API gives a specific error for trying to cancel again
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Cannot cancel order '{order_id}' ({symbol}) due to its current state (e.g., already filled/cancelled): {e}. Assuming cancellation complete.{RESET}")
             return True # Treat as success if it cannot be cancelled due to its state
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 1) # Exponential backoff for rate limit
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded cancelling order {order_id} ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt
        except ccxt.ExchangeError as e:
            # Other exchange errors during cancellation
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}. Retrying...{RESET}")
            # Check for potentially non-retryable cancel errors if needed (e.g., permissions)
            # err_str = str(e).lower()
            # if "permission denied" in err_str: return False
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error cancelling order {order_id} ({symbol}): {e}. Cannot continue.{RESET}")
            return False
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as failure for safety
            return False

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to cancel order {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return False

def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"
    position_size: Decimal, # The calculated size for the order (always positive Decimal)
    market_info: MarketInfo,
    logger: logging.Logger,
    reduce_only: bool = False,
    params: Optional[Dict[str, Any]] = None # Allow passing extra exchange-specific params
) -> Optional[Dict]:
    """
    Places a market order based on the trade signal and calculated size using ccxt.create_order.
    Handles specifics for exchanges like Bybit V5 (category, reduceOnly, positionIdx).
    Includes retry logic and detailed error handling with hints.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        trade_signal: The action type ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The calculated size for the order (must be a positive Decimal).
        market_info: The MarketInfo dictionary for the symbol (must be valid).
        logger: The logger instance for messages specific to this operation.
        reduce_only: If True, attempts to set the reduceOnly flag (for closing/reducing positions).
        params: Optional dictionary of extra parameters for ccxt's create_order (overrides defaults).

    Returns:
        The order result dictionary from ccxt if the API call for placement was successful,
        otherwise None. Success indicates the order was accepted by the exchange, not necessarily filled.
    """
    lg = logger

    # --- Determine Order Side from Signal ---
    side_map = {
        "BUY": "buy",         # Opening a long position or increasing long
        "SELL": "sell",       # Opening a short position or increasing short
        "EXIT_SHORT": "buy",  # Closing/reducing a short position (buy back)
        "EXIT_LONG": "sell"   # Closing/reducing a long position (sell off)
    }
    side = side_map.get(trade_signal.upper())
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' received for {symbol}. Cannot determine order side.")
        return None

    # --- Validate Position Size ---
    if not isinstance(position_size, Decimal) or not position_size.is_finite() or position_size <= Decimal('0'):
        lg.error(f"Invalid position size '{position_size}' provided for {symbol}. Must be a positive, finite Decimal.")
        return None

    # --- Prepare Order Details ---
    order_type = 'market' # Strategy currently uses market orders
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base_currency
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info.get('id') # Use exchange-specific ID
    is_bybit = 'bybit' in exchange.id.lower()

    if not market_id:
         lg.error(f"Cannot place trade for {symbol}: Market ID missing in market_info.")
         return None

    # --- Apply Amount Precision and Convert Size to Float for CCXT ---
    # It's crucial to apply precision *before* converting to float to avoid issues.
    final_size_decimal = position_size # Start with the input size
    try:
        amount_step = market_info['amount_precision_step_decimal']
        if amount_step is None or amount_step <= 0:
             raise ValueError("Amount precision step is missing or invalid in market info.")

        # Round the size DOWN to the nearest valid step
        num_steps = (final_size_decimal / amount_step).to_integral_value(rounding=ROUND_DOWN)
        rounded_size_decimal = num_steps * amount_step

        if rounded_size_decimal <= 0:
            raise ValueError(f"Position size {position_size.normalize()} rounded down to zero or negative based on step {amount_step.normalize()}. Cannot place order.")

        if rounded_size_decimal != final_size_decimal:
             lg.warning(f"Adjusting order size {final_size_decimal.normalize()} to {rounded_size_decimal.normalize()} due to precision step {amount_step.normalize()} before placing order.")
             final_size_decimal = rounded_size_decimal # Use the rounded size

        # Convert the final, rounded Decimal size to float for ccxt
        amount_float = float(final_size_decimal)
        # Final check for effective zero after float conversion (paranoia)
        if abs(amount_float) < 1e-15: # Use a very small number comparison
            raise ValueError(f"Final position size {final_size_decimal.normalize()} converts to near-zero float ({amount_float}).")

    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Failed to apply precision or convert size {position_size.normalize()} for order placement ({symbol}): {e}")
        return None

    # --- Base Order Arguments for ccxt.create_order ---
    order_args: Dict[str, Any] = {
        'symbol': symbol,     # Use standard symbol for ccxt call
        'type': order_type,
        'side': side,
        'amount': amount_float, # Pass the float amount
        # 'price': None, # Not needed for market orders
    }

    # --- Prepare Exchange-Specific Parameters ---
    order_params: Dict[str, Any] = {}
    if is_bybit and is_contract:
        try:
            # Determine category from market_info
            category = 'linear' # Default assumption
            if market_info.get('is_linear'): category = 'linear'
            elif market_info.get('is_inverse'): category = 'inverse'
            else: raise ValueError(f"Invalid Bybit contract category derived from market_info: {market_info.get('contract_type_str')}")

            order_params = {
                'category': category,
                'positionIdx': 0 # Assume one-way mode (index 0). Hedge mode would need 1 or 2. Check Bybit docs.
            }

            if reduce_only:
                order_params['reduceOnly'] = True
                # Bybit V5 often requires/prefers IOC or FOK for reduceOnly market orders to avoid accidental increases
                # IOC (Immediate Or Cancel) is generally safer for market reduceOnly.
                order_params['timeInForce'] = 'IOC'
                lg.debug(f"Setting Bybit V5 specific params: reduceOnly=True, timeInForce='IOC' for {symbol}.")
            else:
                 # For opening orders, default TIF is usually GTC, which is fine for market orders
                 pass

        except Exception as e:
            lg.error(f"Failed to set Bybit V5 specific parameters for {symbol} order: {e}. Proceeding with base params, order might fail.")
            order_params = {} # Reset params if setup failed

    # Merge any externally provided params (allowing override of defaults/calculated params)
    if params and isinstance(params, dict):
        lg.debug(f"Merging external parameters into order: {params}")
        order_params.update(params)

    # Add params dict to order_args if it's not empty
    if order_params:
        order_args['params'] = order_params

    # --- Log Order Intent Clearly ---
    lg.warning(f"{BRIGHT}===> Placing Trade Order ({action_desc}) <==={RESET}")
    lg.warning(f"  Symbol : {symbol} (Market ID: {market_id})")
    lg.warning(f"  Type   : {order_type.upper()}")
    lg.warning(f"  Side   : {side.upper()} (Derived from: {trade_signal})")
    lg.warning(f"  Size   : {final_size_decimal.normalize()} {size_unit} (Float: {amount_float})") # Log both Decimal and float used
    if order_args.get('params'):
        lg.warning(f"  Params : {order_args['params']}")

    # --- Execute Order Placement with Retry ---
    attempts = 0
    last_exception: Optional[Exception] = None
    order_result: Optional[Dict] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")

            # +++++ Place the Order via CCXT +++++
            order_result = exchange.create_order(**order_args)
            # ++++++++++++++++++++++++++++++++++++

            # --- Log Success ---
            order_id = order_result.get('id', 'N/A')
            status = order_result.get('status', 'unknown') # e.g., 'open', 'closed', 'canceled'
            # Market orders might fill immediately ('closed') or remain 'open' briefly.
            # Check fill details if available.
            avg_price_raw = order_result.get('average')
            filled_raw = order_result.get('filled')

            # Use safe conversion allowing zero (e.g., if order is 'open' but not filled yet)
            avg_price_dec = _safe_market_decimal(avg_price_raw, 'order.average', allow_zero=True, allow_negative=False)
            filled_dec = _safe_market_decimal(filled_raw, 'order.filled', allow_zero=True, allow_negative=False)

            log_msg_parts = [
                f"{NEON_GREEN}{action_desc} Order Placement API Call Succeeded!{RESET}",
                f"ID: {order_id}",
                f"Status: {status}"
            ]
            if avg_price_dec is not None and avg_price_dec > 0:
                log_msg_parts.append(f"Avg Fill Price: ~{avg_price_dec.normalize()}")
            if filled_dec is not None:
                log_msg_parts.append(f"Filled Amount: {filled_dec.normalize()} {size_unit}")

            lg.info(" ".join(log_msg_parts))
            lg.debug(f"Full order result ({symbol}): {json.dumps(order_result, indent=2)}") # Pretty print result

            # Exit retry loop on success
            break

        # --- Error Handling for create_order ---
        except ccxt.InsufficientFunds as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Insufficient Funds.{RESET}")
            lg.error(f"  Check available balance ({CONFIG.get('quote_currency', 'N/A')}) and margin requirements for size {final_size_decimal.normalize()} {size_unit} with leverage {CONFIG.get('leverage', 'N/A')}x.")
            lg.error(f"  Error details: {e}")
            return None # Non-retryable
        except ccxt.InvalidOrder as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Invalid Order Parameters.{RESET}")
            lg.error(f"  Error details: {e}")
            lg.error(f"  Order Arguments Sent: {order_args}")
            # Provide hints based on error message and market info
            err_lower = str(e).lower()
            min_a_str = fmt_dec_log(market_info.get('min_amount_decimal'))
            min_c_str = fmt_dec_log(market_info.get('min_cost_decimal'))
            amt_s_str = fmt_dec_log(market_info.get('amount_precision_step_decimal'))
            max_a_str = fmt_dec_log(market_info.get('max_amount_decimal'))
            max_c_str = fmt_dec_log(market_info.get('max_cost_decimal'))

            hint = ""
            if any(s in err_lower for s in ["minimum order", "too small", "less than minimum", "min notional", "min value", "order value is too small"]):
                hint = f"Check order size ({final_size_decimal.normalize()}) vs Min Amount ({min_a_str}) and estimated order cost vs Min Cost ({min_c_str})."
            elif any(s in err_lower for s in ["precision", "lot size", "step size", "size precision", "quantity precision", "order qty invalid"]):
                hint = f"Check order size ({final_size_decimal.normalize()}) precision against Amount Step ({amt_s_str}). Ensure size is a multiple of step."
            elif any(s in err_lower for s in ["exceed", "too large", "greater than maximum", "max value", "max order qty", "position size exceed max limit"]):
                hint = f"Check order size ({final_size_decimal.normalize()}) vs Max Amount ({max_a_str}) and estimated order cost vs Max Cost ({max_c_str}). Also check position/risk limits."
            elif "reduce only" in err_lower or "reduceonly" in err_lower or "position is closed" in err_lower:
                hint = f"Reduce-only order failed. Ensure an open position exists in the correct direction and the size ({final_size_decimal.normalize()}) does not increase the position."
            elif "position size" in err_lower or "position idx" in err_lower or "position side does not match" in err_lower:
                 hint = f"Order conflicts with existing position, leverage limits, or position mode (One-Way vs Hedge). Check positionIdx param if using Bybit."
            elif "risk limit" in err_lower:
                 hint = f"Order may exceed account risk limits set on the exchange. Check exchange settings."

            if hint: lg.error(f"  >> Hint: {hint}")
            return None # Non-retryable
        except ccxt.ExchangeError as e:
            last_exception = e
            err_code = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code = match.group(2)
            else: err_code = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

            lg.warning(f"{NEON_YELLOW}Order Placement Exchange Error ({symbol} {action_desc}): {e} (Code: {err_code}). Retry {attempts + 1}...{RESET}")

            # Check for known fatal/non-retryable codes or messages (can customize per exchange)
            fatal_codes = ['10001', '10004', '110017', '110040', '30086', '3303001', '3400088', '110043'] # Add more as needed
            fatal_msgs = ["invalid parameter", "precision", "exceed limit", "risk limit", "invalid symbol", "api key", "authentication failed", "leverage exceed", "account mode", "position mode"]
            is_fatal_code = err_code in fatal_codes
            is_fatal_message = any(msg in str(e).lower() for msg in fatal_msgs)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE order placement error ({err_code}). Check arguments, config, and account status.{RESET}")
                return None # Non-retryable failure

            # If not identified as fatal, proceed to retry logic below

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2)
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error placing order ({symbol}): {e}. Cannot continue.{RESET}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order ({symbol}): {e}{RESET}", exc_info=True)
            return None # Treat unexpected errors as fatal

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES and order_result is None: # Only retry if order_result is still None
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempts - 1))) # Exponential backoff

    # --- Handle Failure After Retries ---
    if order_result is None:
        lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        lg.error(f"  Last error encountered: {last_exception}")
        return None

    return order_result # Return the successful order dictionary

# --- Placeholder Functions (Require Full Implementation) ---

def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # TSL distance/offset (interpretation depends on exchange)
    tsl_activation_price: Optional[Decimal] = None   # Price at which TSL should activate (Bybit V5)
) -> bool:
    """
    Sets or modifies Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL)
    for an existing position using exchange-specific API calls.

    ** NOTE: This function uses Bybit V5's `private_post_position_set_trading_stop` endpoint structure
       as an example. Implementation details WILL vary for other exchanges. Requires thorough testing. **

    Args:
        exchange: Initialized ccxt exchange instance.
        symbol: Standard symbol (e.g., 'BTC/USDT').
        market_info: Market information (contains market_id, category, precision).
        position_info: Current position details (contains side, entryPrice, etc.).
        logger: Logger instance.
        stop_loss_price: Target SL price (Decimal). Set to '0' or None to remove existing SL.
        take_profit_price: Target TP price (Decimal). Set to '0' or None to remove existing TP.
        trailing_stop_distance: TSL distance/offset (Decimal, positive). Exchange-specific interpretation. Set to '0' or None to remove TSL.
        tsl_activation_price: Price to activate the TSL (Decimal, Bybit V5). Usually required if setting TSL distance. Set to '0' or None to remove.

    Returns:
        True if the protection setting API call was successfully sent and the exchange
             responded with a success code (e.g., 0 for Bybit V5).
        False otherwise (invalid input, formatting error, API error, non-zero return code).
    """
    lg = logger
    lg.debug(f"Attempting to set/modify position protection for {symbol}...")

    log_parts = []
    if stop_loss_price is not None: log_parts.append(f"SL={stop_loss_price.normalize() if stop_loss_price > 0 else 'REMOVE'}")
    if take_profit_price is not None: log_parts.append(f"TP={take_profit_price.normalize() if take_profit_price > 0 else 'REMOVE'}")
    if trailing_stop_distance is not None: log_parts.append(f"TSL Dist={trailing_stop_distance.normalize() if trailing_stop_distance > 0 else 'REMOVE'}")
    if tsl_activation_price is not None: log_parts.append(f"TSL Act={tsl_activation_price.normalize() if tsl_activation_price > 0 else 'REMOVE'}")

    if not log_parts:
        lg.debug("No protection parameters provided to set/modify.")
        return True # Nothing to do, considered success

    lg.info(f"  Target protections: {', '.join(log_parts)}")

    # --- Exchange Specific Logic ---
    is_bybit = 'bybit' in exchange.id.lower()

    if is_bybit:
        # --- Bybit V5 Example using implicit private POST call ---
        try:
            market_id = market_info['id']
            category = 'linear' # Default assumption
            if market_info.get('is_linear'): category = 'linear'
            elif market_info.get('is_inverse'): category = 'inverse'
            else: raise ValueError(f"Invalid or unsupported Bybit category for protection setting ({market_info.get('contract_type_str')})")

            # --- Prepare Parameters for Bybit V5 API ---
            # Endpoint: POST /v5/position/set-trading-stop
            params: Dict[str, Any] = {
                'category': category,
                'symbol': market_id,
                'positionIdx': 0, # Assuming one-way mode (0). Hedge mode uses 1 (Buy) or 2 (Sell). Needs config if supporting hedge.
                'tpslMode': 'Full', # Apply to entire position ('Partial' also possible)
                # Default trigger to Mark Price, make configurable if needed
                'slTriggerBy': 'MarkPrice',
                'tpTriggerBy': 'MarkPrice',
                # Default order type to Market, make configurable if needed
                'slOrderType': 'Market',
                'tpOrderType': 'Market',
            }

            # Format and add parameters if provided and valid
            # Use '0' string to remove existing SL/TP/TSL on Bybit V5
            param_added = False

            if stop_loss_price is not None:
                if stop_loss_price <= 0: # Request to remove SL
                    params['stopLoss'] = '0'
                    param_added = True
                else:
                    sl_str = _format_price(exchange, symbol, stop_loss_price)
                    if sl_str: params['stopLoss'] = sl_str; param_added = True
                    else: lg.error(f"Invalid SL price format for {symbol}: {stop_loss_price}"); return False

            if take_profit_price is not None:
                 if take_profit_price <= 0: # Request to remove TP
                     params['takeProfit'] = '0'
                     param_added = True
                 else:
                     tp_str = _format_price(exchange, symbol, take_profit_price)
                     if tp_str: params['takeProfit'] = tp_str; param_added = True
                     else: lg.error(f"Invalid TP price format for {symbol}: {take_profit_price}"); return False

            # Trailing Stop logic for Bybit V5
            if trailing_stop_distance is not None:
                 if trailing_stop_distance <= 0: # Request to remove TSL
                     params['trailingStop'] = '0'
                     # Also remove activation price if removing TSL distance
                     if 'activePrice' not in params: params['activePrice'] = '0'
                     param_added = True
                 else:
                     # Bybit 'trailingStop' expects a string distance/offset in price points.
                     # Formatting needs care - use price precision? Or amount precision? Assume price.
                     # Needs testing based on Bybit's exact requirement for TSL distance format.
                     # Using price_to_precision might format it like a price, not a distance.
                     # Let's try formatting it simply based on number of decimal places of price tick.
                     price_tick = market_info.get('price_precision_step_decimal')
                     if not price_tick: raise ValueError("Price tick precision missing for TSL distance formatting.")
                     # Determine number of decimal places needed for the distance string
                     decimal_places = abs(price_tick.as_tuple().exponent)
                     ts_dist_str = f"{trailing_stop_distance:.{decimal_places}f}" # Format distance to required decimal places

                     # Basic validation: ensure formatted string is positive number
                     if _safe_market_decimal(ts_dist_str, "tsl_dist_str", False, False):
                          params['trailingStop'] = ts_dist_str
                          param_added = True
                     else:
                          lg.error(f"Invalid TSL distance format for {symbol}: {trailing_stop_distance} -> Formatted: '{ts_dist_str}'"); return False

            if tsl_activation_price is not None:
                 if tsl_activation_price <= 0: # Request to remove activation price
                     params['activePrice'] = '0'
                     # Note: Removing activation price might implicitly disable TSL even if distance remains? Test needed.
                     param_added = True # Mark as added even if removing
                 else:
                     act_str = _format_price(exchange, symbol, tsl_activation_price)
                     if act_str: params['activePrice'] = act_str; param_added = True
                     else: lg.error(f"Invalid TSL activation price format for {symbol}: {tsl_activation_price}"); return False

            # --- Call API only if parameters were actually added/modified ---
            if param_added:
                 lg.info(f"Calling Bybit V5 set_trading_stop API for {symbol} with params: {params}")
                 response: Optional[Dict] = None
                 last_api_error: Optional[Exception] = None
                 # Use retry logic for the API call
                 for attempt in range(MAX_API_RETRIES + 1):
                     try:
                         # Use ccxt's implicit method mapping if available (check ccxt source/docs)
                         # E.g., if ccxt maps it to a function like `exchange.set_trading_stop(params)`
                         # Or use the explicit private method call structure:
                         response = exchange.private_post_position_set_trading_stop(params)
                         lg.debug(f"API Response (set_trading_stop, attempt {attempt+1}): {response}")

                         # Check Bybit V5 return code (0 indicates success)
                         ret_code = response.get('retCode')
                         ret_msg = response.get('retMsg', 'No message')

                         if ret_code == 0:
                             lg.info(f"{NEON_GREEN}Protection set/modified successfully via API for {symbol} (Code: 0).{RESET}")
                             return True # Success
                         else:
                             # API call succeeded but returned non-zero code (error)
                             last_api_error = ccxt.ExchangeError(f"Bybit API error setting protection: {ret_msg} (Code: {ret_code})")
                             lg.error(f"{last_api_error}")
                             # Check if retryable based on code? Some errors might be permanent.
                             # Example: 110001 (param error), 110025 (position closed), 110043 (cannot set on isolated with pos)
                             non_retryable_codes = [110001, 110025, 110043, 110044]
                             if ret_code in non_retryable_codes:
                                  lg.error(f"Non-retryable error code {ret_code} received. Aborting protection setting.")
                                  return False
                             # Otherwise, allow retry loop to continue

                     except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError) as e:
                         last_api_error = e
                         lg.warning(f"{NEON_YELLOW}API Error attempt {attempt+1} setting protection ({symbol}): {e}. Retrying...{RESET}")
                     except ccxt.AuthenticationError as e:
                          last_api_error = e
                          lg.critical(f"{NEON_RED}Authentication error setting protection: {e}. Cannot continue.{RESET}")
                          return False # Non-retryable
                     except Exception as e:
                          last_api_error = e
                          lg.error(f"{NEON_RED}Unexpected error calling protection API ({symbol}): {e}{RESET}", exc_info=True)
                          return False # Treat unexpected as non-retryable

                     # Wait before retrying
                     if attempt < MAX_API_RETRIES:
                          time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))

                 # If loop finishes without success
                 lg.error(f"Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_api_error}")
                 return False

            else:
                 lg.debug("No valid protection parameters were formatted to send to the API.")
                 return True # Considered success as no action was needed/possible based on inputs

        except (ccxt.ExchangeError, ccxt.NetworkError, ValueError, TypeError) as e:
            lg.error(f"Error preparing or calling protection API for Bybit {symbol}: {e}", exc_info=True)
            return False
        except Exception as e:
             lg.error(f"Unexpected error setting protection for Bybit {symbol}: {e}", exc_info=True)
             return False

    else:
        # --- Fallback / Other Exchanges ---
        lg.error(f"Protection setting logic (_set_position_protection) not implemented for exchange {exchange.id}.")
        # Implementation would involve finding the correct ccxt method or parameters for that specific exchange.
        # Possibilities:
        # - `exchange.edit_order(id, symbol, type, side, amount, price, params={'stopLossPrice': ..., 'takeProfitPrice': ...})` if SL/TP are attached to orders.
        # - `exchange.create_order(symbol, 'STOP_MARKET', side, amount, price=trigger_price, params={'stopLossPrice': ..., 'reduceOnly': True})` for placing separate SL orders.
        # - Exchange-specific methods like `exchange.private_post_some_endpoint(...)`.
        return False # Assume failure if not implemented

def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    config: Dict[str, Any], # Pass global config for TSL parameters
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Can optionally set TP at the same time
) -> bool:
    """
    Calculates Trailing Stop Loss (TSL) parameters based on configuration and current position,
    then attempts to set the TSL on the exchange via `_set_position_protection`.

    ** NOTE: This function relies on the `_set_position_protection` implementation.
       The interpretation of 'callback_rate' (percentage vs price points) and the
       calculation of 'distance' based on it are critical and potentially exchange-specific.
       The current implementation assumes Bybit V5 style distance and activation price. Requires testing. **

    Args:
        exchange: Initialized ccxt exchange instance.
        symbol: Standard symbol.
        market_info: Market information.
        position_info: Current position details.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional TP price (Decimal) to set simultaneously.

    Returns:
        True if TSL setup API call was initiated successfully via `_set_position_protection`, False otherwise.
    """
    lg = logger
    lg.debug(f"Calculating and setting initial Trailing Stop Loss for {symbol}...")

    prot_cfg = config.get('protection', {})
    # Get TSL parameters from config, convert to Decimal
    try:
        # Callback rate: The distance the price needs to move against the trail before SL triggers.
        # Interpretation: Assume this is a percentage of entry price to determine a fixed price distance? Or percentage of current price?
        # Let's use percentage of ENTRY price for consistency with initial risk.
        callback_rate = Decimal(str(prot_cfg.get('trailing_stop_callback_rate', 0.005))) # e.g., 0.5%
        # Activation percentage: How far price must move in profit from entry before TSL activates.
        activation_perc = Decimal(str(prot_cfg.get('trailing_stop_activation_percentage', 0.003))) # e.g., 0.3%

        if not (callback_rate.is_finite() and callback_rate > 0):
             lg.error(f"Invalid TSL callback rate ({callback_rate}) in config. Must be positive finite."); return False
        if not (activation_perc.is_finite() and activation_perc >= 0):
             lg.error(f"Invalid TSL activation percentage ({activation_perc}) in config. Must be non-negative finite."); return False
    except (ValueError, InvalidOperation, TypeError) as e:
         lg.error(f"Invalid TSL parameter format in config for {symbol}: {e}"); return False

    # Get required position/market data
    entry_price = position_info.get('entryPrice_decimal') # Use the parsed Decimal field
    side = position_info.get('side')
    price_tick = market_info.get('price_precision_step_decimal')

    if not entry_price or entry_price <= 0:
        lg.error(f"Cannot calculate TSL for {symbol}: Invalid or missing entry price ({entry_price})."); return False
    if side not in ['long', 'short']:
         lg.error(f"Cannot calculate TSL for {symbol}: Invalid position side ({side})."); return False
    if not price_tick or price_tick <= 0:
        lg.error(f"Cannot calculate TSL for {symbol}: Invalid price tick ({price_tick})."); return False

    # --- TSL Calculation Logic ---
    # 1. Calculate Activation Price:
    #    - Long: entry_price * (1 + activation_perc)
    #    - Short: entry_price * (1 - activation_perc)
    #    Round to price tick precision (round *away* from entry for safety margin)
    activation_price_calc: Decimal
    try:
        if side == 'long':
            act_raw = entry_price * (Decimal('1') + activation_perc)
            # Round UP, away from entry
            activation_price_calc = (act_raw / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick
            # Ensure activation price is strictly above entry after rounding
            if activation_price_calc <= entry_price: activation_price_calc = entry_price + price_tick
        else: # short
            act_raw = entry_price * (Decimal('1') - activation_perc)
            # Round DOWN, away from entry
            activation_price_calc = (act_raw / price_tick).quantize(Decimal('1'), ROUND_DOWN) * price_tick
            # Ensure activation price is strictly below entry after rounding
            if activation_price_calc >= entry_price: activation_price_calc = entry_price - price_tick

        # Ensure activation price is positive
        if activation_price_calc <= 0:
            raise ValueError(f"Calculated TSL activation price ({activation_price_calc}) is non-positive.")

    except (InvalidOperation, ValueError) as e:
        lg.error(f"Error calculating TSL activation price for {symbol}: {e}"); return False


    # 2. Calculate TSL Distance (Callback):
    #    Interpretation is crucial. Assuming Bybit V5 style: distance in price points.
    #    Calculate this distance based on callback_rate % of ENTRY price.
    tsl_distance_calc: Decimal
    try:
        dist_raw = entry_price * callback_rate
        # Round distance UP to the nearest price tick to ensure minimum distance
        tsl_distance_calc = (dist_raw / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick

        # Ensure distance is at least one price tick
        if tsl_distance_calc < price_tick:
             tsl_distance_calc = price_tick
             lg.debug(f"TSL calculated distance was less than price tick for {symbol}. Using minimum tick distance: {price_tick.normalize()}")

        if tsl_distance_calc <= 0:
             raise ValueError(f"Calculated TSL distance ({tsl_distance_calc}) is non-positive.")

    except (InvalidOperation, ValueError) as e:
         lg.error(f"Error calculating TSL distance for {symbol}: {e}"); return False


    lg.info(f"Calculated TSL parameters for {symbol}: ActivationPrice={activation_price_calc.normalize()}, Distance={tsl_distance_calc.normalize()} (based on {callback_rate:.2%} of entry)")

    # --- Call the actual protection setting function ---
    lg.info(f"Calling _set_position_protection to apply TSL (and optional TP) for {symbol}")
    # Pass the calculated Decimal values. _set_position_protection handles formatting.
    success = _set_position_protection(
        exchange, symbol, market_info, position_info, lg,
        stop_loss_price=None, # Let TSL manage the stop loss; don't set a fixed SL here.
        take_profit_price=take_profit_price, # Pass through optional TP if provided
        trailing_stop_distance=tsl_distance_calc, # Pass calculated distance
        tsl_activation_price=activation_price_calc # Pass calculated activation price
    )

    if success:
        lg.info(f"Trailing stop loss setup/update API call initiated successfully for {symbol}.")
        # IMPORTANT: The calling function (`manage_existing_position`) should update
        # the internal `position_state['tsl_activated'] = True` AFTER this returns True.
    else:
        lg.error(f"Failed to set up/update trailing stop loss for {symbol} via _set_position_protection.")

    return success # Return result of the underlying API call attempt

class VolumaticOBStrategy:
    """
    Encapsulates the Volumatic Trend + Order Block strategy logic.
    This class is responsible for calculating technical indicators, identifying the
    prevailing trend, finding order blocks based on pivot points, and managing the
    state of these order blocks (active, violated, extended).

    ** WARNING: This is currently a PLACEHOLDER implementation. **
    The core algorithms for Volumatic Trend (VT) calculation and Order Block (OB)
    detection/management need to be fully implemented based on their specific definitions.
    The current methods provide basic structure but use simplified logic or standard
    library functions (like simple EMA instead of Volume-Weighted calculations)
    and may not accurately reflect the intended strategy.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        """
        Initializes the strategy engine with parameters from the config.

        Args:
            config: The bot's configuration dictionary.
            market_info: Market information for the symbol this engine instance handles.
            logger: Logger instance for strategy-specific messages.

        Raises:
            ValueError: If essential strategy parameters are missing or invalid.
        """
        self.lg = logger
        self.symbol = market_info['symbol']
        self.market_info = market_info
        self.params = config.get('strategy_params', {})
        self.protection_params = config.get('protection', {}) # Needed? Maybe not directly here.
        self.price_tick = market_info.get('price_precision_step_decimal')
        if self.price_tick is None or self.price_tick <= 0:
            self.lg.warning(f"Invalid price tick ({self.price_tick}) in market_info for {self.symbol}. Using small default for internal calcs.")
            self.price_tick = Decimal('0.00000001')

        # --- Extract and Validate Strategy Parameters ---
        try:
            # VT Params
            self.vt_len = int(self.params.get('vt_length', DEFAULT_VT_LENGTH))
            self.vt_atr_period = int(self.params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD))
            self.vt_vol_ema_len = int(self.params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH)) # Placeholder usage
            self.vt_atr_mult = Decimal(str(self.params.get('vt_atr_multiplier', DEFAULT_VT_ATR_MULTIPLIER))) # Placeholder usage
            # OB Params
            self.ob_source = str(self.params.get('ob_source', DEFAULT_OB_SOURCE))
            self.ph_left = int(self.params.get('ph_left', DEFAULT_PH_LEFT))
            self.ph_right = int(self.params.get('ph_right', DEFAULT_PH_RIGHT))
            self.pl_left = int(self.params.get('pl_left', DEFAULT_PL_LEFT))
            self.pl_right = int(self.params.get('pl_right', DEFAULT_PL_RIGHT))
            self.ob_extend = bool(self.params.get('ob_extend', DEFAULT_OB_EXTEND))
            self.ob_max_boxes = int(self.params.get('ob_max_boxes', DEFAULT_OB_MAX_BOXES))

            # Basic validation of parameter ranges
            if not (self.vt_len > 0 and self.vt_atr_period > 0 and self.vt_vol_ema_len > 0):
                raise ValueError("VT parameters (vt_length, vt_atr_period, vt_vol_ema_length) must be positive integers.")
            if not (self.vt_atr_mult.is_finite() and self.vt_atr_mult > 0):
                 raise ValueError("VT parameter vt_atr_multiplier must be a positive number.")
            if not (self.ph_left > 0 and self.ph_right > 0 and self.pl_left > 0 and self.pl_right > 0):
                 raise ValueError("Pivot lookback/forward periods (ph_left/right, pl_left/right) must be positive integers.")
            if not (self.ob_max_boxes > 0):
                 raise ValueError("OB parameter ob_max_boxes must be a positive integer.")
            if self.ob_source not in ["Wicks", "Body"]:
                 raise ValueError(f"Invalid OB parameter ob_source: '{self.ob_source}'. Must be 'Wicks' or 'Body'.")

        except (ValueError, TypeError, KeyError) as e:
             self.lg.error(f"Invalid or missing strategy parameter for {self.symbol}: {e}")
             raise ValueError(f"Strategy Initialization Failed for {self.symbol}: Invalid parameters.") from e

        # Estimate minimum data length needed for calculations
        self.min_data_len = max(
            self.vt_len * 2,             # EMA/SWMA needs buffer
            self.vt_atr_period + 1,      # ATR needs N+1
            self.vt_vol_ema_len * 2,     # Vol EMA needs buffer
            self.ph_left + self.ph_right + 1, # Pivots need full window
            self.pl_left + self.pl_right + 1
        ) + 50 # Add a generous safety buffer

        self.lg.info(f"Strategy Engine initialized for {self.symbol} (Min data length ~{self.min_data_len})")
        self.lg.debug(f"  Params: VT Len={self.vt_len}, ATR Period={self.vt_atr_period}, Vol EMA={self.vt_vol_ema_len}, ATR Mult={self.vt_atr_mult}, "
                      f"OB Src={self.ob_source}, Pivots L/R=({self.ph_left}/{self.ph_right}, {self.pl_left}/{self.pl_right}), Extend={self.ob_extend}, Max Boxes={self.ob_max_boxes}")

        # State for tracking order blocks across updates (persistent between `update` calls)
        # These lists store OrderBlock dictionaries.
        self._active_bull_boxes: List[OrderBlock] = []
        self._active_bear_boxes: List[OrderBlock] = []


    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Placeholder: Calculates Smoothed Weighted Moving Average (SWMA) or EMA.
        ** This needs to be replaced with the exact calculation required by the
           Volumatic Trend definition, potentially involving volume weighting. **
        Uses `pandas_ta.swma` if available, otherwise simple EMA as a fallback placeholder.
        """
        self.lg.debug(f"Placeholder _ema_swma called for length {length}. Requires actual VT implementation.")
        if series.empty or length <= 0 or series.isnull().all():
             return pd.Series(dtype=np.float64, index=series.index) # Return empty/NaN series matching index

        try:
            # Ensure input is float64 for pandas_ta
            series_float = series.astype(np.float64)
            result = None
            # Try using pandas_ta implementation if it matches the desired SWMA definition
            if hasattr(ta, 'swma'):
                 result = ta.swma(series_float, length=length)
            else:
                 # Fallback to simple EMA if ta.swma doesn't exist or isn't the correct SWMA
                 self.lg.warning("pandas_ta.swma not found or specific VT SWMA needed. Using simple EMA as placeholder for _ema_swma.")
                 result = ta.ema(series_float, length=length)

            return result.astype(np.float64) if result is not None else pd.Series(np.nan, index=series.index, dtype=np.float64)
        except Exception as e:
             self.lg.error(f"Error calculating SWMA/EMA (placeholder) for length {length}: {e}", exc_info=False) # Less verbose traceback for common calc errors
             return pd.Series(np.nan, index=series.index, dtype=np.float64) # Return NaN series on error

    def _find_pivots(self, series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
        """
        Placeholder: Finds pivot high or low points.
        ** This is a highly simplified and likely inaccurate placeholder. **
        A correct implementation should efficiently find points that are strictly higher (for PH)
        or lower (for PL) than all points within `left` bars before and `right` bars after.
        Returns a boolean Series indicating pivot points (True where pivot occurs).
        """
        self.lg.debug(f"Placeholder _find_pivots called (Left:{left}, Right:{right}, High:{is_high}). Requires accurate implementation.")
        if series.empty or left < 0 or right < 0 or series.isnull().all():
            return pd.Series(False, index=series.index)

        pivots = pd.Series(False, index=series.index)
        # This basic rolling window comparison is inefficient and likely incorrect for true pivots.
        # A proper implementation might use comparisons with shifted series or dedicated libraries.
        # Example using rolling (still potentially flawed interpretation of pivots):
        window_size = left + right + 1
        if window_size > len(series): return pivots # Not enough data

        # Rolling apply is one way, but defining the pivot condition precisely is key
        # This example checks if value is max/min in the centered window
        # rolling_window = series.rolling(window=window_size, center=True) # Center=True might align better
        # if is_high:
        #     pivots = series == rolling_window.max()
        # else:
        #     pivots = series == rolling_window.min()
        # # Need to handle edges and ensure strict inequality / uniqueness if required by definition

        self.lg.warning("Pivot detection logic (_find_pivots) is a basic placeholder and likely inaccurate.")
        # Returning all False for now until implemented
        return pivots


    def update(self, df: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes the input OHLCV DataFrame to calculate indicators, determine trend,
        and identify order blocks according to the strategy's (placeholder) logic.

        ** WARNING: This function requires the full implementation of the Volumatic Trend
           and Order Block detection algorithms. The current calculations are placeholders. **

        Args:
            df: The input OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume'
                columns as Decimal type, indexed by UTC timestamp.

        Returns:
            A StrategyAnalysisResults dictionary containing the processed DataFrame
            and key strategy outputs (trend, ATR, bands, active OBs). Returns results
            with default/empty values if calculations fail or data is insufficient.
        """
        self.lg.info(f"Running strategy update for {self.symbol} on {len(df)} candles ending {df.index[-1] if not df.empty else 'N/A'}")

        # --- Default Result Structure ---
        # Initialize with current state of OBs, update if analysis runs successfully
        default_result = StrategyAnalysisResults(
            dataframe=df.copy(), # Start with a copy of the input
            last_close=df['close'].iloc[-1] if not df.empty else Decimal('0'),
            current_trend_up=None,
            trend_just_changed=False,
            active_bull_boxes=[box for box in self._active_bull_boxes if box['active'] and not box['violated']], # Filter current state
            active_bear_boxes=[box for box in self._active_bear_boxes if box['active'] and not box['violated']],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if len(df) < self.min_data_len:
            self.lg.warning(f"DataFrame length ({len(df)}) for {self.symbol} is less than minimum required ({self.min_data_len}). Strategy results may be inaccurate or incomplete.")
            # Return default results but ensure last_close is updated if possible
            if not df.empty: default_result['last_close'] = df['close'].iloc[-1]
            return default_result

        # --- Perform Calculations (Placeholders & Actual Implementation Needed) ---
        df_analysis = df.copy() # Work on a copy
        try:
            # Ensure necessary columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_analysis.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_analysis.columns]
                raise ValueError(f"Missing required columns in DataFrame: {missing}")

            # Convert Decimal columns to float for pandas_ta compatibility
            # Handle potential NaNs introduced during data loading/conversion
            high_f = df_analysis['high'].apply(lambda x: float(x) if pd.notna(x) and isinstance(x, Decimal) and x.is_finite() else np.nan)
            low_f = df_analysis['low'].apply(lambda x: float(x) if pd.notna(x) and isinstance(x, Decimal) and x.is_finite() else np.nan)
            close_f = df_analysis['close'].apply(lambda x: float(x) if pd.notna(x) and isinstance(x, Decimal) and x.is_finite() else np.nan)
            volume_f = df_analysis['volume'].apply(lambda x: float(x) if pd.notna(x) and isinstance(x, Decimal) and x.is_finite() else np.nan)

            # --- === Volumatic Trend (VT) Calculation (Placeholder) === ---
            # 1. Calculate Base Indicators (ATR is usually needed)
            atr_series_f = ta.atr(high_f, low_f, close_f, length=self.vt_atr_period)
            if atr_series_f is None or atr_series_f.isnull().all():
                 self.lg.warning(f"ATR calculation failed or resulted in all NaNs for {self.symbol}.")
                 atr_series_f = pd.Series(np.nan, index=df_analysis.index) # Ensure series exists
            df_analysis['atr'] = atr_series_f.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            current_atr = df_analysis['atr'].iloc[-1] if pd.notna(df_analysis['atr'].iloc[-1]) else None

            # 2. Calculate Volume-Weighted Price Component (Placeholder: Using simple close EMA)
            #    *** Replace with actual calculation: e.g., EMA/SWMA of (Volume * TypicalPrice) / Volume ***
            self.lg.warning("VT calculation uses PLACEHOLDER logic (EMA of close, simple ATR bands). Needs full implementation.")
            vt_center_line_f = ta.ema(close_f, length=self.vt_len) # Placeholder: EMA of close
            df_analysis['vt_center'] = vt_center_line_f.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

            # 3. Calculate VT Bands (Placeholder: Simple +/- ATR from center line)
            df_analysis['vt_upper'] = vt_center_line_f + atr_series_f * float(self.vt_atr_mult)
            df_analysis['vt_lower'] = vt_center_line_f - atr_series_f * float(self.vt_atr_mult)
            # Convert bands back to Decimal
            df_analysis['vt_upper'] = df_analysis['vt_upper'].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            df_analysis['vt_lower'] = df_analysis['vt_lower'].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

            upper_band = df_analysis['vt_upper'].iloc[-1] if pd.notna(df_analysis['vt_upper'].iloc[-1]) else None
            lower_band = df_analysis['vt_lower'].iloc[-1] if pd.notna(df_analysis['vt_lower'].iloc[-1]) else None

            # 4. Determine VT Trend Direction (Placeholder: Based on close vs center line)
            #    *** Replace with actual VT trend rule (e.g., slope, price vs bands) ***
            df_analysis['trend_up'] = close_f > vt_center_line_f # Example: Trend up if close > center
            current_trend_up = bool(df_analysis['trend_up'].iloc[-1]) if pd.notna(df_analysis['trend_up'].iloc[-1]) else None

            # Check if trend changed on the last candle
            trend_just_changed = False
            if len(df_analysis) > 1 and pd.notna(df_analysis['trend_up'].iloc[-1]) and pd.notna(df_analysis['trend_up'].iloc[-2]):
                 trend_just_changed = (df_analysis['trend_up'].iloc[-1] != df_analysis['trend_up'].iloc[-2])

            # 5. Calculate Volume Normalization (Placeholder: simple min-max scaling)
            #    *** Replace if Volumatic strategy uses a specific normalization method ***
            vol_min = volume_f.min()
            vol_max = volume_f.max()
            if not volume_f.empty and pd.notna(vol_min) and pd.notna(vol_max) and vol_max > vol_min:
                 df_analysis['vol_norm'] = (volume_f - vol_min) / (vol_max - vol_min) * 100
            else:
                 df_analysis['vol_norm'] = 0.0 # Assign 0 if calculation not possible
            vol_norm_int = int(df_analysis['vol_norm'].iloc[-1]) if pd.notna(df_analysis['vol_norm'].iloc[-1]) else None
            # --- === End VT Placeholder === ---


            # --- === Order Block (OB) Detection & Management (Placeholder) === ---
            # 1. Find Pivots (Using placeholder _find_pivots)
            #    *** Replace with accurate pivot detection algorithm ***
            self.lg.warning("Order Block detection uses PLACEHOLDER logic (inaccurate pivots, no OB creation/violation/extension). Needs full implementation.")
            df_analysis['ph'] = self._find_pivots(df_analysis['high'], self.ph_left, self.ph_right, is_high=True)
            df_analysis['pl'] = self._find_pivots(df_analysis['low'], self.pl_left, self.pl_right, is_high=False)

            # 2. Identify New Order Blocks based on Pivots
            #    - Iterate backwards or use signals from df_analysis['ph']/['pl'].
            #    - Define OB top/bottom based on pivot candle's High/Low (Wicks) or Open/Close (Body) -> self.ob_source.
            #    - Create new OrderBlock dictionary with unique ID, type, timestamp, prices, active=True, violated=False.
            #    - Prepend new OBs to self._active_bull_boxes / self._active_bear_boxes.
            #    - *** Placeholder: No new OBs are created ***
            new_bull_boxes = [] # Placeholder
            new_bear_boxes = [] # Placeholder
            self._active_bull_boxes = new_bull_boxes + self._active_bull_boxes
            self._active_bear_boxes = new_bear_boxes + self._active_bear_boxes

            # 3. Update Existing Order Blocks (Violation & Extension)
            #    - Iterate through self._active_bull_boxes and self._active_bear_boxes.
            #    - Check Violation: Has the current candle's close crossed the boundary of an active OB?
            #        - Bull OB violated if close < bottom.
            #        - Bear OB violated if close > top.
            #        - If violated, set active=False, violated=True, violation_ts=current_timestamp.
            #    - Check Extension: If self.ob_extend is True and box is not violated, update 'extended_to_ts'.
            #    - *** Placeholder: No violation or extension logic implemented ***

            # 4. Manage Max Number of Boxes
            #    - If len(self._active_bull_boxes) > self.ob_max_boxes, remove the oldest ones.
            #    - If len(self._active_bear_boxes) > self.ob_max_boxes, remove the oldest ones.
            self._active_bull_boxes = self._active_bull_boxes[:self.ob_max_boxes]
            self._active_bear_boxes = self._active_bear_boxes[:self.ob_max_boxes]

            # 5. Filter Final Active Boxes for Result
            #    - Select only boxes that are currently active and not violated.
            active_bull_boxes = [box for box in self._active_bull_boxes if box.get('active', False) and not box.get('violated', False)]
            active_bear_boxes = [box for box in self._active_bear_boxes if box.get('active', False) and not box.get('violated', False)]
            # --- === End OB Placeholder === ---

        except Exception as e:
            self.lg.error(f"Error during strategy calculation for {self.symbol}: {e}", exc_info=True)
            # Return default results on error, ensuring last_close is updated if possible
            if not df.empty: default_result['last_close'] = df['close'].iloc[-1]
            # Keep existing OBs in the default result (as analysis failed)
            default_result['active_bull_boxes'] = [box for box in self._active_bull_boxes if box.get('active', False) and not box.get('violated', False)]
            default_result['active_bear_boxes'] = [box for box in self._active_bear_boxes if box.get('active', False) and not box.get('violated', False)]
            return default_result

        # --- Prepare and Return Analysis Results ---
        last_close = df_analysis['close'].iloc[-1] if not df_analysis.empty else Decimal('0')
        self.lg.info(f"Strategy update complete for {self.symbol}.")
        self.lg.info(f"  Last Close: {last_close.normalize()}")
        self.lg.info(f"  VT Trend Up (Placeholder): {current_trend_up}, Trend Changed: {trend_just_changed}")
        self.lg.info(f"  ATR: {current_atr.normalize() if current_atr else 'N/A'}")
        self.lg.info(f"  VT
