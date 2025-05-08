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
    # Ensure tzdata is installed for non-UTC timezones with zoneinfo
    try:
        ZoneInfo("America/Chicago") # Attempt to load a non-UTC zone
    except ZoneInfoNotFoundError:
        print("Warning: 'zoneinfo' is available, but 'tzdata' package seems missing or corrupt.")
        print("         `pip install tzdata` is recommended for reliable timezone support.")
        # Continue with zoneinfo, but it might fail for non-UTC zones later
    except Exception as tz_init_err:
         print(f"Warning: Error initializing test timezone with 'zoneinfo': {tz_init_err}")
         # Continue cautiously
except ImportError:
    # Fallback for older Python versions or if zoneinfo itself is not installed
    print("Warning: 'zoneinfo' module not found (requires Python 3.9+). Falling back to basic UTC implementation.")
    print("         For accurate local time logging, upgrade Python or use a backport library.")

    # Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface
    class ZoneInfo: # type: ignore [no-redef]
        """Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface."""
        _key = "UTC" # Class attribute, always UTC

        def __init__(self, key: str):
            """
            Initializes the fallback ZoneInfo. Always uses UTC, regardless of the key provided.
            Logs a warning if a non-UTC key is requested.
            """
            if key.upper() != "UTC":
                # Use print as logger might not be ready during early import
                print(f"Warning: Fallback ZoneInfo initialized with key '{key}', but will always use UTC.")
            # Store the requested key for representation, though internally we always use UTC
            self._requested_key = key

        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            """Attaches UTC timezone info to a datetime object. Returns None if input is None."""
            if dt is None:
                return None
            # Only add tzinfo if the datetime object is naive
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            # If already timezone-aware, return as-is (or convert to UTC if that's desired behavior?)
            # Standard ZoneInfo usually replaces tzinfo. Mimic that.
            return dt.replace(tzinfo=timezone.utc)


        def fromutc(self, dt: datetime) -> datetime:
            """Converts a UTC datetime to this timezone (which is always UTC in the fallback)."""
            if not isinstance(dt, datetime):
                raise TypeError("fromutc() requires a datetime argument")
            if dt.tzinfo is None:
                # Standard library raises ValueError for naive datetime in fromutc
                # print("Warning: Calling fromutc on naive datetime in UTC fallback. Assuming UTC.")
                # return dt.replace(tzinfo=timezone.utc)
                 raise ValueError("fromutc: naive datetime has no timezone to observe")

            # If already timezone-aware, ensure it's converted to UTC
            return dt.astimezone(timezone.utc)

        def utcoffset(self, dt: Optional[datetime]) -> timedelta:
            """Returns the UTC offset (always zero for UTC)."""
            return timedelta(0)

        def dst(self, dt: Optional[datetime]) -> timedelta:
            """Returns the Daylight Saving Time offset (always zero for UTC)."""
            return timedelta(0)

        def tzname(self, dt: Optional[datetime]) -> str:
            """Returns the timezone name (always 'UTC')."""
            return "UTC"

        def __repr__(self) -> str:
            """Provides a clear representation indicating it's the fallback."""
            return f"ZoneInfo(key='{self._requested_key}') [Fallback: Always UTC]"

        def __str__(self) -> str:
            """Returns the effective timezone key (always UTC)."""
            return self._key

    class ZoneInfoNotFoundError(Exception): # type: ignore [no-redef]
        """Exception raised when a timezone is not found (fallback definition)."""
        pass

# --- Third-Party Library Imports ---
# Grouped by general purpose for better organization
# Data Handling & Numerics
import numpy as np
import pandas as pd
import pandas_ta as ta  # Technical Analysis library

# API & Networking
import requests         # For HTTP requests (often used implicitly by ccxt)
import ccxt             # Crypto Exchange Trading Library

# Utilities
from colorama import Fore, Style, init as colorama_init # Colored console output
from dotenv import load_dotenv                        # Load environment variables from .env file

# --- Initial Setup ---
# Set Decimal precision globally for accurate financial calculations
# Consider if higher precision is needed, 28 is often sufficient but depends on assets traded.
getcontext().prec = 28
# Initialize Colorama for cross-platform colored output (autoreset ensures color ends after each print)
colorama_init(autoreset=True)
# Load environment variables from a .env file if it exists in the current directory or parent directories.
load_dotenv()

# --- Constants ---
BOT_VERSION = "1.4.1+enhancements"

# --- API Credentials (Loaded from Environment Variables) ---
# Critical check: Ensure API keys are loaded before proceeding.
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print directly here as logger might not be fully initialized yet.
    print(f"{Fore.RED}{Style.BRIGHT}FATAL ERROR: BYBIT_API_KEY and/or BYBIT_API_SECRET environment variables are missing.{Style.RESET_ALL}")
    print(f"{Fore.RED}Please ensure they are set in your system environment or in a '.env' file in the bot's directory.{Style.RESET_ALL}")
    print(f"{Fore.RED}The bot cannot authenticate with the exchange and will now exit.{Style.RESET_ALL}")
    sys.exit(1) # Exit immediately if credentials are missing

# --- Configuration File & Logging ---
CONFIG_FILE: str = "config.json"    # Name of the configuration file
LOG_DIRECTORY: str = "bot_logs"     # Directory to store log files

# --- Timezone Configuration ---
DEFAULT_TIMEZONE_STR: str = "America/Chicago" # Default timezone if not specified elsewhere
# Prioritize TIMEZONE from .env, fallback to the default defined above.
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    # Use print as logger might not be ready
    print(f"{Fore.YELLOW}{Style.BRIGHT}Warning: Timezone '{TIMEZONE_STR}' not found using 'zoneinfo'. Falling back to UTC.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Ensure 'tzdata' is installed (`pip install tzdata`) for non-UTC timezones.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update the string representation to reflect fallback
except Exception as tz_err:
    # Catch any other unexpected errors during ZoneInfo initialization
    print(f"{Fore.RED}{Style.BRIGHT}Warning: An unexpected error occurred initializing timezone '{TIMEZONE_STR}': {tz_err}. Falling back to UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update the string representation

# --- API & Timing Constants ---
# These serve as defaults and can often be overridden by values in config.json.
MAX_API_RETRIES: int = 3           # Max number of retries for most failed API calls (excluding rate limits)
RETRY_DELAY_SECONDS: int = 5       # Initial delay in seconds between retries (often increased exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Delay after placing order to confirm position status (allows exchange processing time)
LOOP_DELAY_SECONDS: int = 15       # Base delay in seconds between main loop cycles (per symbol)
BYBIT_API_KLINE_LIMIT: int = 1000  # Max klines per Bybit V5 API request (important constraint for fetch_klines_ccxt)

# --- Data & Strategy Constants ---
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Supported intervals in config.json (string format)
CCXT_INTERVAL_MAP: Dict[str, str] = { # Mapping from config interval strings to CCXT standard timeframe strings
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT: int = 750     # Default number of klines to fetch historically if not specified in config
MAX_DF_LEN: int = 2000             # Maximum length of DataFrame to keep in memory (helps prevent memory bloat over time)

# Default Volumatic Trend (VT) parameters (can be overridden by config.json)
# These are placeholders until the actual VT logic is implemented.
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950 # Placeholder: Lookback for Volume EMA/SWMA
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0 # Placeholder: ATR multiplier for VT bands
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Placeholder: Parameter mentioned in original code, usage unclear in placeholder logic. Keep or remove?

# Default Order Block (OB) parameters (can be overridden by config.json)
DEFAULT_OB_SOURCE: str = "Wicks"    # Source for defining OB price levels: "Wicks" (High/Low) or "Body" (Open/Close)
DEFAULT_PH_LEFT: int = 10           # Pivot High lookback periods
DEFAULT_PH_RIGHT: int = 10          # Pivot High lookforward periods
DEFAULT_PL_LEFT: int = 10           # Pivot Low lookback periods
DEFAULT_PL_RIGHT: int = 10          # Pivot Low lookforward periods
DEFAULT_OB_EXTEND: bool = True      # Whether to visually extend OB boxes until violated
DEFAULT_OB_MAX_BOXES: int = 50      # Maximum number of active Order Blocks to track per side (bull/bear)

# --- Trading Constants ---
# This will be updated by the loaded configuration. Defaulting to USDT.
QUOTE_CURRENCY: str = "USDT"

# --- UI Constants (Colorama Foregrounds and Styles) ---
# Define constants for frequently used colors and styles for console output.
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN         # Duplicate of NEON_BLUE, kept for potential compatibility if used elsewhere
RESET: str = Style.RESET_ALL       # Resets all styles and colors to default
BRIGHT: str = Style.BRIGHT         # Makes text brighter/bolder
DIM: str = Style.DIM               # Makes text dimmer

# --- Create Log Directory ---
# Ensure the directory for log files exists before setting up logging handlers.
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e:
    print(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create log directory '{LOG_DIRECTORY}': {e}{RESET}")
    print(f"{NEON_RED}Please check permissions and ensure the path is valid. Exiting.{RESET}")
    sys.exit(1)

# --- Global State ---
_shutdown_requested: bool = False # Flag used for graceful shutdown triggered by signal handler

# --- Type Definitions (Enhanced with Docstrings and Detail) ---
class OrderBlock(TypedDict):
    """Represents an identified Order Block (OB) zone."""
    id: str                 # Unique identifier (e.g., "BULL_1678886400000_PH")
    type: str               # "BULL" (demand zone) or "BEAR" (supply zone)
    timestamp: pd.Timestamp # Timestamp of the candle defining the block (UTC, pandas Timestamp for convenience)
    top: Decimal            # Top price level (highest point) of the block
    bottom: Decimal         # Bottom price level (lowest point) of the block
    active: bool            # Is the block currently considered active (not violated, not expired/invalidated)?
    violated: bool          # Has the block been violated by subsequent price action?
    violation_ts: Optional[pd.Timestamp] # Timestamp when the violation occurred (UTC, pandas Timestamp)
    extended_to_ts: Optional[pd.Timestamp] # Timestamp the box is currently visually extended to (UTC, pandas Timestamp), if extension enabled

class StrategyAnalysisResults(TypedDict):
    """
    Contains the results from the strategy analysis performed on historical data (DataFrame).
    Includes indicators, trend status, and identified active order blocks.
    """
    dataframe: pd.DataFrame # The analyzed DataFrame including all calculated indicator columns
    last_close: Decimal     # The closing price of the most recent candle in the DataFrame
    current_trend_up: Optional[bool] # Current trend direction (True=Up, False=Down, None=Undetermined/Sideways) based on strategy rules
    trend_just_changed: bool # Did the trend direction change on the very last candle compared to the previous one?
    active_bull_boxes: List[OrderBlock] # List of currently active bullish Order Blocks identified by the strategy
    active_bear_boxes: List[OrderBlock] # List of currently active bearish Order Blocks identified by the strategy
    vol_norm_int: Optional[int] # Normalized volume indicator value (e.g., 0-100), if used by strategy (Placeholder)
    atr: Optional[Decimal]    # Current Average True Range value from the last candle
    upper_band: Optional[Decimal] # Upper band value from the last candle (e.g., from Volumatic Trend or other channel indicator)
    lower_band: Optional[Decimal] # Lower band value from the last candle (e.g., from Volumatic Trend or other channel indicator)

class MarketInfo(TypedDict):
    """
    Standardized and enhanced market information derived from the ccxt `market` structure.
    Provides convenient access to critical details like precision, limits, and contract type,
    using Decimal types for financial accuracy where appropriate.
    """
    # --- Standard CCXT Fields (subset, presence may vary by exchange) ---
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
    active: Optional[bool]  # Is the market currently active/tradeable? (Can be None if not provided by exchange)
    contract: bool          # Is it a contract (swap, future, option)? (Base ccxt flag)
    linear: Optional[bool]  # Linear contract? (Quote currency settlement, e.g., USDT-margined)
    inverse: Optional[bool] # Inverse contract? (Base currency settlement, e.g., BTC-margined)
    quanto: Optional[bool]  # Quanto contract? (Settled in a third currency)
    taker: float            # Taker fee rate (as a fraction, e.g., 0.00075 for 0.075%)
    maker: float            # Maker fee rate (as a fraction, e.g., 0.0002 for 0.02%)
    contractSize: Optional[Any] # Size of one contract (often 1 for linear, value in USD for inverse). Use decimal version for calcs.
    expiry: Optional[int]   # Timestamp (milliseconds UTC) of future/option expiry
    expiryDatetime: Optional[str] # ISO 8601 datetime string of expiry
    strike: Optional[float] # Strike price for options
    optionType: Optional[str] # 'call' or 'put' for options
    precision: Dict[str, Any] # Price and amount precision rules (e.g., {'price': 0.01, 'amount': 0.001}). Use decimal versions for calcs.
    limits: Dict[str, Any]    # Order size and cost limits (e.g., {'amount': {'min': 0.001, 'max': 100}}). Use decimal versions for calcs.
    info: Dict[str, Any]      # Raw market data dictionary directly from the exchange API response (useful for debugging or accessing non-standard fields)
    # --- Added/Derived Fields for Convenience and Precision ---
    is_contract: bool         # Enhanced convenience flag: True if swap, future, or option based on ccxt flags
    is_linear: bool           # Enhanced convenience flag: True if linear contract (and is_contract is True)
    is_inverse: bool          # Enhanced convenience flag: True if inverse contract (and is_contract is True)
    contract_type_str: str    # User-friendly string describing the market type: "Spot", "Linear", "Inverse", "Option", or "Unknown"
    min_amount_decimal: Optional[Decimal] # Minimum order size (in base currency for spot, contracts for futures) as Decimal, or None if not specified
    max_amount_decimal: Optional[Decimal] # Maximum order size (in base/contracts) as Decimal, or None
    min_cost_decimal: Optional[Decimal]   # Minimum order cost (value in quote currency, usually price * amount * contractSize) as Decimal, or None
    max_cost_decimal: Optional[Decimal]   # Maximum order cost (value in quote currency) as Decimal, or None
    amount_precision_step_decimal: Optional[Decimal] # Smallest increment allowed for order amount (step size) as Decimal. Crucial for placing orders.
    price_precision_step_decimal: Optional[Decimal]  # Smallest increment allowed for order price (tick size) as Decimal. Crucial for placing orders.
    contract_size_decimal: Decimal # Contract size as Decimal (defaults to 1 if not applicable/found, e.g., for spot or if info missing)

class PositionInfo(TypedDict):
    """
    Standardized and enhanced position information derived from the ccxt `position` structure.
    Provides convenient access to key position details like size, entry price, PnL, and protection orders,
    using Decimal types for financial accuracy. Includes bot-specific state tracking.
    """
    # --- Standard CCXT Fields (subset, presence may vary by exchange/method) ---
    id: Optional[str]       # Position ID (exchange-specific, may not always be present or unique)
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    timestamp: Optional[int] # Position creation/update timestamp (milliseconds UTC)
    datetime: Optional[str]  # ISO 8601 datetime string representation of the timestamp
    contracts: Optional[float] # Number of contracts (legacy/float). Use size_decimal derived field instead for precision.
    contractSize: Optional[Any] # Size of one contract for this position. Use market_info.contract_size_decimal for calculations.
    side: Optional[str]      # Position side: 'long' or 'short'
    notional: Optional[Any]  # Position value in quote currency (approx). Use notional_decimal derived field for precision.
    leverage: Optional[Any]  # Position leverage. Use leverage_decimal derived field for precision.
    unrealizedPnl: Optional[Any] # Unrealized profit/loss. Use unrealizedPnl_decimal derived field for precision.
    realizedPnl: Optional[Any]   # Realized profit/loss (may not always be populated). Use Decimal if needed.
    collateral: Optional[Any]    # Margin used for the position (in margin currency). Use collateral_decimal derived field for precision.
    entryPrice: Optional[Any]    # Average entry price. Use entryPrice_decimal derived field for precision.
    markPrice: Optional[Any]     # Current mark price used for PnL calculation. Use markPrice_decimal derived field for precision.
    liquidationPrice: Optional[Any] # Estimated liquidation price. Use liquidationPrice_decimal derived field for precision.
    marginMode: Optional[str]    # Margin mode: 'isolated' or 'cross'
    hedged: Optional[bool]       # Is hedging enabled for this position? (Less common now, usually account-level setting)
    maintenanceMargin: Optional[Any] # Maintenance margin required for the position. Use maintenanceMargin_decimal derived field.
    maintenanceMarginPercentage: Optional[float] # Maintenance margin rate (as fraction, e.g., 0.005 for 0.5%)
    initialMargin: Optional[Any] # Initial margin used to open the position. Use initialMargin_decimal derived field.
    initialMarginPercentage: Optional[float] # Initial margin rate (as fraction)
    marginRatio: Optional[float] # Margin ratio (health indicator, lower is better, triggers liquidation). Formula varies by exchange.
    lastUpdateTimestamp: Optional[int] # Timestamp of last position update from exchange (milliseconds UTC)
    info: Dict[str, Any]         # Raw position data dictionary directly from the exchange API response (essential for accessing non-standard fields like protection orders)
    # --- Added/Derived Fields for Convenience and Precision ---
    size_decimal: Decimal        # Position size as Decimal (positive for long, negative for short, derived from 'contracts' or 'info')
    entryPrice_decimal: Optional[Decimal] # Entry price as Decimal
    markPrice_decimal: Optional[Decimal] # Mark price as Decimal
    liquidationPrice_decimal: Optional[Decimal] # Liquidation price as Decimal
    leverage_decimal: Optional[Decimal] # Leverage as Decimal
    unrealizedPnl_decimal: Optional[Decimal] # Unrealized PnL as Decimal
    notional_decimal: Optional[Decimal] # Notional value (position value in quote currency) as Decimal
    collateral_decimal: Optional[Decimal] # Collateral (margin used) as Decimal
    initialMargin_decimal: Optional[Decimal] # Initial margin as Decimal
    maintenanceMargin_decimal: Optional[Decimal] # Maintenance margin as Decimal
    # --- Protection Order Status (Extracted from 'info' or root, strings often returned) ---
    stopLossPrice: Optional[str] # Current stop loss price set on the exchange (as string, often '0' or '0.0' if not set)
    takeProfitPrice: Optional[str] # Current take profit price set on the exchange (as string, often '0' or '0.0' if not set)
    trailingStopLoss: Optional[str] # Current trailing stop distance/offset (as string, interpretation depends on exchange/endpoint, e.g., Bybit V5 'trailingStop')
    tslActivationPrice: Optional[str] # Trailing stop activation price (as string, if available/set, e.g., Bybit V5 'activePrice')
    # --- Bot State Tracking (Managed internally by the bot logic, reflects *bot's* knowledge/actions on this position instance) ---
    be_activated: bool           # Has the break-even logic been triggered and successfully executed *by the bot* for this position instance?
    tsl_activated: bool          # Has the trailing stop loss been activated *by the bot* or detected as active on the exchange?

class SignalResult(TypedDict):
    """Represents the outcome of the signal generation process based on strategy analysis."""
    signal: str              # The generated signal: "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
    reason: str              # A human-readable explanation for why the signal was generated
    initial_sl: Optional[Decimal] # Calculated initial stop loss price if the signal is for a new entry ("BUY" or "SELL")
    initial_tp: Optional[Decimal] # Calculated initial take profit price if the signal is for a new entry ("BUY" or "SELL")

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom log formatter that redacts sensitive API keys and secrets from log messages.
    Inherits from `logging.Formatter` and overrides the `format` method.
    """
    # Use highly distinct placeholders unlikely to appear naturally in logs.
    _api_key_placeholder = "***BYBIT_API_KEY_REDACTED***"
    _api_secret_placeholder = "***BYBIT_API_SECRET_REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record using the parent class, then redacts sensitive strings.

        Args:
            record: The logging.LogRecord to format.

        Returns:
            The formatted log message string with sensitive data redacted.
        """
        # First, format the message using the standard formatter logic
        formatted_msg = super().format(record)
        redacted_msg = formatted_msg

        # Perform redaction only if keys are actually set, non-empty strings, and reasonably long
        key = API_KEY
        secret = API_SECRET
        try:
            # Check type and length > 4 to avoid redacting short common words if keys are short/invalid
            if key and isinstance(key, str) and len(key) > 4:
                # Simple string replacement is usually sufficient and faster than regex here.
                # If keys could be substrings of other words, regex with boundaries `\b` might be safer,
                # but API keys rarely have word boundaries.
                redacted_msg = redacted_msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and len(secret) > 4:
                redacted_msg = redacted_msg.replace(secret, self._api_secret_placeholder)
        except Exception as e:
            # Prevent crashing the application if redaction fails unexpectedly.
            # Log error directly to stderr as the logger itself might be involved.
            print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
            # Return the original formatted message in case of error to ensure log continuity.
            return formatted_msg
        return redacted_msg

class NeonConsoleFormatter(SensitiveFormatter):
    """
    Custom log formatter for visually appealing console output.
    - Displays messages with level-specific colors using Colorama.
    - Includes the logger name for context.
    - Uses the globally configured local timezone (TIMEZONE) for timestamps.
    - Inherits redaction logic from SensitiveFormatter.
    """
    # Define colors for different logging levels
    _level_colors = {
        logging.DEBUG: DIM + NEON_CYAN,       # Dim Cyan for Debug
        logging.INFO: NEON_BLUE,              # Bright Cyan for Info
        logging.WARNING: NEON_YELLOW,         # Bright Yellow for Warning
        logging.ERROR: NEON_RED,              # Bright Red for Error
        logging.CRITICAL: BRIGHT + NEON_RED   # Bright Red and Bold for Critical
    }
    _default_color = NEON_BLUE # Default color if level not in map
    # Define the log format string using Colorama constants
    _log_format = (
        f"{DIM}%(asctime)s{RESET} {NEON_PURPLE}[%(name)-15s]{RESET} " # Timestamp (Dim), Logger Name (Purple, padded)
        f"%(levelcolor)s%(levelname)-8s{RESET} %(message)s"          # Level (Colored, padded), Message (Default color)
    )
    # Define the timestamp format (Year-Month-Day Hour:Minute:Second)
    _date_format = '%Y-%m-%d %H:%M:%S' # Include date for clarity on console

    def __init__(self, **kwargs):
        """Initializes the formatter, setting the format string, date format, and timestamp converter."""
        # Initialize the parent class (SensitiveFormatter -> logging.Formatter)
        # We don't pass `fmt` directly here as we use dynamic `levelcolor`.
        super().__init__(fmt=self._log_format, datefmt=self._date_format, **kwargs)
        # Set the time converter to use the globally configured TIMEZONE for local time display.
        # This lambda converts the timestamp (usually Unix epoch float) to a timetuple in the local zone.
        self.converter = lambda timestamp, _: datetime.fromtimestamp(timestamp, tz=TIMEZONE).timetuple()

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record with level-specific colors, local timestamps, and redaction.

        Args:
            record: The logging.LogRecord to format.

        Returns:
            The fully formatted, colored, and redacted log message string.
        """
        # Dynamically assign the color based on the log level before formatting.
        record.levelcolor = self._level_colors.get(record.levelno, self._default_color)

        # Format the message using the base class's format method.
        # SensitiveFormatter.format() is called, which performs redaction.
        # The timestamp conversion uses the `self.converter` set in __init__.
        formatted_and_redacted_msg = super().format(record)

        # No need for manual redaction here; it's handled by the parent SensitiveFormatter.
        return formatted_and_redacted_msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger instance with standardized handlers and formatters.

    Configures two handlers:
    1. Rotating File Handler: Logs all DEBUG level messages and above to a file
       (e.g., 'bot_logs/pyrmethus.init.log'). Uses UTC timestamps for consistency.
       Redacts sensitive API keys/secrets.
    2. Console Handler: Logs INFO level messages and above (or level from CONSOLE_LOG_LEVEL
       environment variable) to the console (stdout). Uses local timestamps (based on global
       TIMEZONE) and colored output. Redacts sensitive API keys/secrets.

    Args:
        name: The name for the logger (e.g., 'init', 'main', 'BTC/USDT').
              Used in log messages and to generate the log filename.

    Returns:
        A configured logging.Logger instance ready for use.
    """
    # Sanitize the logger name for safe use in filenames
    # Replace common problematic characters (slashes, colons, spaces) with underscores.
    safe_filename_part = re.sub(r'[^\w\-.]', '_', name)
    # Use dot notation for logger names to support potential hierarchical logging features.
    logger_name = f"pyrmethus.{safe_filename_part}"
    # Construct the full path for the log file.
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times if the logger was already configured
    # (e.g., if this function is called again with the same name).
    if logger.hasHandlers():
        logger.debug(f"Logger '{logger_name}' already configured. Skipping setup.")
        return logger

    # Set the logger's base level to DEBUG. Handlers will filter messages based on their own levels.
    logger.setLevel(logging.DEBUG)

    # --- File Handler (UTC Timestamps, Redaction) ---
    try:
        # Rotate log file when it reaches 10MB, keep up to 5 backup files. Use UTF-8 encoding.
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.DEBUG) # Log everything (DEBUG and above) to the file.

        # Use the SensitiveFormatter for redaction.
        # Include milliseconds and line number for detailed debugging in the file.
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC [%(name)s:%(lineno)d] %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Explicitly set the converter to use UTC time (gmtime) for file logs.
        file_formatter.converter = time.gmtime
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    except Exception as e:
        # Log setup errors should be visible immediately, even if console handler fails later.
        print(f"{NEON_RED}{BRIGHT}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler (Local Timestamps, Colors, Redaction) ---
    try:
        sh = logging.StreamHandler(sys.stdout) # Log to standard output

        # Determine console log level from environment variable, default to INFO.
        console_log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, console_log_level_str, None) # Get level integer from string
        # Fallback to INFO if the environment variable is invalid or doesn't correspond to a logging level.
        if not isinstance(log_level, int):
            print(f"{NEON_YELLOW}Warning: Invalid CONSOLE_LOG_LEVEL '{console_log_level_str}'. Defaulting to INFO.{RESET}")
            log_level = logging.INFO
        sh.setLevel(log_level) # Set the minimum level for console output.

        # Use the custom NeonConsoleFormatter for colors, local time, and redaction.
        # The formatter class defines its own format string and date format.
        console_formatter = NeonConsoleFormatter()
        sh.setFormatter(console_formatter)
        logger.addHandler(sh)
    except Exception as e:
        # Log setup errors should be visible immediately.
        print(f"{NEON_RED}{BRIGHT}Error setting up console logger: {e}{RESET}")

    # Prevent messages logged by this logger from propagating up to the root logger,
    # which might have its own handlers (e.g., default StreamHandler) causing duplicate output.
    logger.propagate = False

    logger.debug(f"Logger '{logger_name}' initialized. File: '{log_filename}', Console Level: {logging.getLevelName(sh.level)}")
    return logger

# --- Initial Logger Setup ---
# Create a logger instance specifically for the initialization phase.
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}===== Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing ====={Style.RESET_ALL}")
init_logger.info(f"Using Timezone for Console Logs: {TIMEZONE_STR} ({TIMEZONE})")
init_logger.debug(f"Decimal Precision Set To: {getcontext().prec}")
# Remind user about dependencies, helpful for troubleshooting setup issues.
init_logger.debug("Ensure required packages are installed: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (recommended)")

# --- Configuration Loading & Validation ---

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures that all keys present in the default_config structure
    also exist in the loaded config dictionary. Adds missing keys with their
    default values and logs these additions.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing the expected default keys and values.
        parent_key: Internal tracking string for nested key paths (used for logging).

    Returns:
        A tuple containing:
        - The updated configuration dictionary (potentially modified with added keys).
        - A boolean indicating if any changes (key additions) were made.
    """
    updated_config = config.copy() # Work on a copy to avoid modifying the original dict in place
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing entirely, add it with the default value
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' with default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Key exists and both default and loaded values are dictionaries -> recurse into nested dict
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config # Update the nested dictionary if changes were made within it
                changed = True
        # Optional: Add basic type checking here if desired, but robust validation is done later.
        # Example: Compare type(default_value) vs type(updated_config.get(key))
        # elif type(default_value) is not type(updated_config.get(key)) and default_value is not None:
        #     init_logger.debug(f"Config Note: Type mismatch for '{full_key_path}'. Expected {type(default_value).__name__}, got {type(updated_config.get(key)).__name__}. Validation will handle.")
        #     pass # Let the more specific validation function handle type corrections or warnings

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
    Validates a numeric configuration value within a dictionary level.
    - Checks type (rejects bools, attempts conversion from str/int/float to Decimal).
    - Checks for non-finite values (NaN, Infinity).
    - Checks if the value is within the specified range [min_val, max_val].
    - Handles strict minimum (> min_val) and optional allowance for zero.
    - If `is_int` is True, ensures the value is a whole number and converts it to `int`.
    - If `is_int` is False, ensures the value is numeric and converts it to `float`.
    - If validation fails at any step, logs a warning and replaces the value with the corresponding default.
    - Logs informational messages for type corrections or truncations.

    Args:
        cfg_level: The current dictionary level of the loaded config being validated.
        default_level: The corresponding dictionary level in the default config structure.
        leaf_key: The specific key (string) to validate within the current level.
        key_path: The full dot-notation path to the key (e.g., 'protection.max_loss') for logging.
        min_val: The minimum allowed value (inclusive unless is_strict_min is True).
        max_val: The maximum allowed value (inclusive).
        is_strict_min: If True, the value must be strictly greater than min_val (>).
        is_int: If True, the value should be validated and stored as an integer.
        allow_zero: If True, zero is considered a valid value even if outside the main min/max range.

    Returns:
        True if the value was corrected or replaced with the default, False otherwise.
    """
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key) # Get default for fallback on error
    corrected = False
    final_val = original_val # Assume no change initially

    try:
        # 1. Reject Boolean Type Explicitly
        if isinstance(original_val, bool):
             raise TypeError("Boolean type is not valid for numeric configuration.")

        # 2. Attempt Conversion to Decimal for Robust Validation
        # Convert to string first to handle floats accurately and numeric strings like "1.0" or " 5 ".
        try:
            num_val = Decimal(str(original_val).strip())
        except (InvalidOperation, TypeError, ValueError):
             # Handle cases where conversion to string or Decimal fails (e.g., None, empty string, non-numeric string like "abc")
             raise TypeError(f"Value '{repr(original_val)}' cannot be converted to a number.")

        # 3. Check for Non-Finite Values (NaN, Infinity)
        if not num_val.is_finite():
            raise ValueError("Non-finite value (NaN or Infinity) is not allowed.")

        # Convert range limits to Decimal for precise comparison
        min_dec = Decimal(str(min_val))
        max_dec = Decimal(str(max_val))

        # 4. Range Check
        is_zero = num_val.is_zero()
        # Check minimum boundary condition
        min_check_passed = (num_val > min_dec) if is_strict_min else (num_val >= min_dec)
        # Check maximum boundary condition
        max_check_passed = (num_val <= max_dec)
        # Combine range checks
        range_check_passed = min_check_passed and max_check_passed

        # If range check fails AND it's not an allowed zero, raise error
        if not range_check_passed and not (allow_zero and is_zero):
            range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
            allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
            # Use normalize() for cleaner display of the out-of-range number
            raise ValueError(f"Value {num_val.normalize()} is outside the allowed range {allowed_str}.")

        # 5. Type Check and Correction (Integer or Float/Decimal)
        needs_type_correction = False
        target_type_str = 'integer' if is_int else 'float'

        if is_int:
            # Check if the Decimal value has a fractional part
            if num_val % 1 != 0:
                needs_type_correction = True
                # Truncate towards zero (standard int() behavior)
                final_val = int(num_val.to_integral_value(rounding=ROUND_DOWN))
                init_logger.info(f"{NEON_YELLOW}Config Update: Truncated fractional part for integer key '{key_path}' from {repr(original_val)} to {repr(final_val)}.{RESET}")
                # Re-check range after truncation, as truncation could push it out of bounds
                final_dec_trunc = Decimal(final_val)
                min_check_passed_trunc = (final_dec_trunc > min_dec) if is_strict_min else (final_dec_trunc >= min_dec)
                range_check_passed_trunc = min_check_passed_trunc and (final_dec_trunc <= max_dec)
                if not range_check_passed_trunc and not (allow_zero and final_dec_trunc.is_zero()):
                    range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                    allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
                    raise ValueError(f"Value truncated to {final_val}, which is outside the allowed range {allowed_str}.")
            # Check if the original type wasn't already int (e.g., it was 10.0 or "10")
            elif not isinstance(original_val, int):
                 needs_type_correction = True
                 final_val = int(num_val) # Convert the whole number Decimal to int
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for integer key '{key_path}' from {type(original_val).__name__} to int (value: {repr(final_val)}).{RESET}")
            else:
                 # Already an integer conceptually and stored as int
                 final_val = int(num_val) # Ensure it's definitely int type

        else: # Expecting float
            # Check if original type wasn't float or int (int is acceptable for float fields)
            if not isinstance(original_val, (float, int)):
                 needs_type_correction = True
                 final_val = float(num_val) # Convert validated Decimal to float
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for float key '{key_path}' from {type(original_val).__name__} to float (value: {repr(final_val)}).{RESET}")
            # Check if float representation needs update due to precision differences after Decimal conversion
            # (e.g., original was "0.1", stored as float(Decimal("0.1")))
            elif isinstance(original_val, float):
                 converted_float = float(num_val)
                 # Use a small tolerance for float comparison to avoid flagging tiny representation differences
                 if abs(original_val - converted_float) > 1e-9:
                      needs_type_correction = True
                      final_val = converted_float
                      init_logger.info(f"{NEON_YELLOW}Config Update: Adjusted float value for '{key_path}' due to precision from {repr(original_val)} to {repr(final_val)}.{RESET}")
                 else:
                      final_val = converted_float # Keep as float
            elif isinstance(original_val, int):
                 # Convert int to float if the field expects float - technically a type correction
                 final_val = float(original_val)
                 # Optionally log this:
                 # needs_type_correction = True
                 # init_logger.info(f"Config Update: Converted integer value for float key '{key_path}' to float ({repr(final_val)}).")
            else: # Already a float and numerically close enough after Decimal conversion
                 final_val = float(num_val)

        # Mark as corrected if type was changed
        if needs_type_correction:
            corrected = True

    except (ValueError, InvalidOperation, TypeError, AssertionError) as e:
        # Handle all validation errors caught above (range, type conversion, non-finite, etc.)
        range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
        if allow_zero: range_str += " or 0"
        init_logger.warning(
            f"{NEON_YELLOW}Config Validation Warning: Invalid value for '{key_path}'.\n"
            f"  Provided: {repr(original_val)} (Type: {type(original_val).__name__})\n"
            f"  Problem: {e}\n"
            f"  Expected: {target_type_str} in range {range_str}\n"
            f"  >>> Using default value: {repr(default_val)}{RESET}"
        )
        final_val = default_val # Use the default value on any validation error
        corrected = True

    # Update the configuration dictionary only if a correction or replacement was made
    if corrected:
        cfg_level[leaf_key] = final_val

    return corrected


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads, validates, and potentially corrects the bot's configuration from a JSON file.

    Steps:
    1. Defines the default configuration structure and values.
    2. Checks if the specified config file exists. If not, creates it with default values.
    3. Loads the configuration from the JSON file. Handles JSON decoding errors by attempting
       to back up the corrupted file and recreating a default one.
    4. Ensures all necessary keys from the default structure are present in the loaded config,
       adding missing keys with their default values.
    5. Performs detailed validation and type/range correction for each parameter using helper
       functions (_validate_and_correct_numeric, validate_boolean, validate_string_choice).
       Uses default values as fallbacks for invalid entries.
    6. If any keys were added or values corrected during the process, saves the updated
       configuration back to the file.
    7. Updates the global QUOTE_CURRENCY based on the final loaded/default value.

    Args:
        filepath: The path to the configuration JSON file (e.g., "config.json").

    Returns:
        The loaded, validated, and potentially corrected configuration dictionary.
        Returns the default configuration dictionary if the file cannot be loaded or created
        after multiple attempts, or if critical errors occur during processing.
    """
    global QUOTE_CURRENCY # Allow updating the global QUOTE_CURRENCY setting

    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    # --- Define Default Configuration Structure ---
    # This dictionary serves as the template and provides default values for missing keys
    # and fallback values during validation.
    default_config = {
        # == Trading Core ==
        "trading_pairs": ["BTC/USDT"],          # List of market symbols to trade (e.g., ["BTC/USDT", "ETH/USDT"])
        "interval": "5",                        # Kline timeframe (must be one of VALID_INTERVALS strings)
        "enable_trading": False,                # Master switch: MUST BE true for live order placement. Safety default: false.
        "use_sandbox": True,                    # Use exchange's sandbox/testnet environment? Safety default: true.
        "quote_currency": "USDT",               # Primary currency for balance, PnL, risk calculations (e.g., USDT, BUSD). Case-sensitive.
        "max_concurrent_positions": 1,          # Maximum number of positions allowed open simultaneously across all pairs.

        # == Risk & Sizing ==
        "risk_per_trade": 0.01,                 # Fraction of available balance to risk per trade (e.g., 0.01 = 1%). Must be > 0.0 and <= 1.0.
        "leverage": 20,                         # Desired leverage for contract trading (integer). 0 or 1 typically means spot/no leverage. Exchange limits apply.

        # == API & Timing ==
        "retry_delay": RETRY_DELAY_SECONDS,             # Base delay in seconds between API retry attempts (integer)
        "loop_delay_seconds": LOOP_DELAY_SECONDS,       # Delay in seconds between processing cycles for each symbol (integer)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order placement before checking position status (integer)

        # == Data Fetching ==
        "fetch_limit": DEFAULT_FETCH_LIMIT,             # Default number of historical klines to fetch (integer)
        "orderbook_limit": 25,                          # Limit for order book depth fetching (integer, if feature implemented later)

        # == Strategy Parameters (Volumatic Trend + Order Blocks) ==
        "strategy_params": {
            # -- Volumatic Trend (VT) - Placeholders --
            "vt_length": DEFAULT_VT_LENGTH,             # Lookback period for VT calculation (integer > 0)
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,     # Lookback period for ATR calculation within VT (integer > 0)
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, # Placeholder: Lookback for Volume EMA/SWMA (integer > 0)
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER), # Placeholder: ATR multiplier for VT bands (float > 0)

            # -- Order Blocks (OB) - Placeholders --
            "ob_source": DEFAULT_OB_SOURCE,             # Candle part to define OB price range: "Wicks" or "Body" (string)
            "ph_left": DEFAULT_PH_LEFT,                 # Lookback periods for Pivot High detection (integer > 0)
            "ph_right": DEFAULT_PH_RIGHT,               # Lookforward periods for Pivot High detection (integer > 0)
            "pl_left": DEFAULT_PL_LEFT,                 # Lookback periods for Pivot Low detection (integer > 0)
            "pl_right": DEFAULT_PL_RIGHT,               # Lookforward periods for Pivot Low detection (integer > 0)
            "ob_extend": DEFAULT_OB_EXTEND,             # Extend OB visualization until violated? (boolean)
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,       # Max number of active OBs to track per side (integer > 0)
            "ob_entry_proximity_factor": 1.005,         # Placeholder: Proximity factor for entry signal near OB (float >= 1.0)
            "ob_exit_proximity_factor": 1.001,          # Placeholder: Proximity factor for exit signal near opposite OB (float >= 1.0)
        },

        # == Protection Parameters (Stop Loss, Take Profit, Trailing, Break Even) ==
        "protection": {
            # -- Initial SL/TP (often ATR-based) --
            "initial_stop_loss_atr_multiple": 1.8,      # Initial SL distance = ATR * this multiple (float > 0)
            "initial_take_profit_atr_multiple": 0.7,    # Initial TP distance = ATR * this multiple (float >= 0, 0 means no initial TP)

            # -- Break Even (BE) --
            "enable_break_even": True,                  # Enable moving SL to break-even? (boolean)
            "break_even_trigger_atr_multiple": 1.0,     # Move SL to BE when price moves ATR * multiple in profit (float > 0)
            "break_even_offset_ticks": 2,               # Offset SL from entry by this many price ticks for BE (integer >= 0)

            # -- Trailing Stop Loss (TSL) - Placeholders --
            "enable_trailing_stop": True,               # Enable Trailing Stop Loss? (boolean) - Requires implementation
            "trailing_stop_callback_rate": 0.005,       # Placeholder: TSL callback/distance (float > 0). Interpretation depends on exchange/implementation (e.g., 0.005 = 0.5% or 0.5 price points).
            "trailing_stop_activation_percentage": 0.003, # Placeholder: Activate TSL when price moves this % from entry (float >= 0).
        }
    }

    config_needs_saving: bool = False # Flag to track if the loaded config was modified
    loaded_config: Dict[str, Any] = {} # Initialize as empty dict

    # --- Step 1: File Existence Check & Creation ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Configuration file '{filepath}' not found. Creating a new one with default settings.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Dump the default config structure to the new file with pretty indentation
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully created default config file: {filepath}{RESET}")
            # Use the defaults since the file was just created
            loaded_config = default_config
            config_needs_saving = False # No need to save again immediately
            # Update global QUOTE_CURRENCY from the default we just wrote
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return defaults directly as the file is now correct

        except IOError as e:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create config file '{filepath}': {e}.{RESET}")
            init_logger.critical(f"{NEON_RED}Please check directory permissions. Using internal defaults as fallback.{RESET}")
            # Fallback to using internal defaults in memory if file creation fails
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config

    # --- Step 2: File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        # Basic check: Ensure the loaded data is a dictionary (JSON object)
        if not isinstance(loaded_config, dict):
            raise TypeError("Configuration file content is not a valid JSON object (must be a dictionary at the top level).")
        init_logger.info(f"Successfully loaded configuration from '{filepath}'.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from config file '{filepath}': {e}{RESET}")
        init_logger.error(f"{NEON_RED}The file might be corrupted. Attempting to back up corrupted file and recreate with defaults.{RESET}")
        try:
            # Attempt to back up the corrupted file before overwriting
            backup_path = f"{filepath}.corrupted_{int(time.time())}.bak"
            # Use os.replace for atomic rename where possible
            os.replace(filepath, backup_path)
            init_logger.info(f"Backed up corrupted config to: {backup_path}")
        except Exception as backup_err:
             init_logger.warning(f"Could not back up corrupted config file '{filepath}': {backup_err}")

        # Attempt to recreate the default file
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return the defaults
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Error recreating config file after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Fallback to internal defaults
    except Exception as e:
        # Catch any other unexpected errors during file loading or initial parsing
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config # Fallback to internal defaults

    # --- Step 3: Ensure Keys and Validate Parameters ---
    try:
        # Ensure all default keys exist in the loaded config, add missing ones
        updated_config, keys_added = _ensure_config_keys(loaded_config, default_config)
        if keys_added:
            config_needs_saving = True # Mark for saving later if keys were added

        # --- Validation Logic ---
        init_logger.debug("Starting configuration parameter validation...")

        # Helper function to safely navigate nested dictionaries for validation
        def get_nested_levels(cfg: Dict, path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
            """Gets the dictionary level and leaf key for validation, handling potential path errors."""
            keys = path.split('.')
            current_cfg_level = cfg
            current_def_level = default_config
            try:
                # Iterate through parent keys to reach the level containing the target key
                for key in keys[:-1]:
                    if key not in current_cfg_level or not isinstance(current_cfg_level[key], dict):
                        # This should ideally not happen if _ensure_config_keys worked correctly,
                        # but check defensively.
                        raise KeyError(f"Path segment '{key}' not found or not a dictionary in loaded config during validation.")
                    if key not in current_def_level or not isinstance(current_def_level[key], dict):
                         raise KeyError(f"Path segment '{key}' not found or not a dictionary in default config structure (mismatch).")
                    current_cfg_level = current_cfg_level[key]
                    current_def_level = current_def_level[key]
                leaf_key = keys[-1]
                # Ensure the final key exists in the default structure (should always if default_config is correct)
                if leaf_key not in current_def_level:
                    raise KeyError(f"Leaf key '{leaf_key}' not found in default config structure for path '{path}'. Check default_config definition.")
                return current_cfg_level, current_def_level, leaf_key
            except (KeyError, TypeError) as e:
                init_logger.error(f"Config validation structure error: Cannot access path '{path}'. Reason: {e}. Ensure config structure matches default.")
                return None, None, None # Indicate failure to access the level

        # Wrapper validation function for numeric values
        def validate_numeric(cfg: Dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False):
            """Applies numeric validation using helpers and marks config for saving on changes."""
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None:
                init_logger.error(f"Skipping validation for '{key_path}' due to config structure error.")
                # Decide how to handle: stop bot? Use default? Assume _ensure_keys handled it?
                # For now, log error and continue, relying on later code to handle potential issues.
                # If validation is critical, consider raising an exception here.
                return

            # Check if key exists at the leaf level (should exist due to _ensure_config_keys)
            if leaf_key not in cfg_level:
                 init_logger.warning(f"Config validation: Key '{key_path}' unexpectedly missing after ensure_keys. Using default value: {repr(def_level[leaf_key])}.")
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 return

            # Perform the actual validation and correction using the helper function
            corrected = _validate_and_correct_numeric(
                cfg_level, def_level, leaf_key, key_path,
                min_val, max_val, is_strict_min, is_int, allow_zero
            )
            if corrected:
                config_needs_saving = True # Mark for saving if the helper made changes

        # Wrapper validation function for boolean values
        def validate_boolean(cfg: Dict, key_path: str):
            """Validates and corrects boolean config values, marking config for saving on changes."""
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None: return # Error already logged

            if leaf_key not in cfg_level: # Should not happen after _ensure_config_keys
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 init_logger.warning(f"Config validation: Boolean key '{key_path}' missing. Using default: {repr(def_level[leaf_key])}.")
                 return

            current_value = cfg_level[leaf_key]
            if not isinstance(current_value, bool):
                # Attempt to interpret common string representations ("true", "false", "1", "0", etc.)
                corrected_val = None
                if isinstance(current_value, str):
                    val_lower = current_value.lower().strip()
                    if val_lower in ['true', 'yes', '1', 'on']: corrected_val = True
                    elif val_lower in ['false', 'no', '0', 'off']: corrected_val = False
                # Handle numeric 1/0 as well?
                elif isinstance(current_value, int) and current_value in [0, 1]:
                    corrected_val = bool(current_value)

                if corrected_val is not None:
                     init_logger.info(f"{NEON_YELLOW}Config Update: Corrected boolean-like value for '{key_path}' from {repr(current_value)} to {repr(corrected_val)}.{RESET}")
                     cfg_level[leaf_key] = corrected_val
                     config_needs_saving = True
                else:
                     # Cannot interpret the value as boolean, use the default
                     init_logger.warning(f"Config Validation Warning: Invalid value for boolean key '{key_path}': {repr(current_value)}. Expected true/false or equivalent. Using default: {repr(def_level[leaf_key])}.")
                     cfg_level[leaf_key] = def_level[leaf_key]
                     config_needs_saving = True

        # Wrapper validation function for string choices
        def validate_string_choice(cfg: Dict, key_path: str, choices: List[str], case_sensitive: bool = False):
             """Validates string value against a list of allowed choices, marking config for saving on changes."""
             nonlocal config_needs_saving
             cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
             if cfg_level is None or def_level is None or leaf_key is None: return # Error handled

             if leaf_key not in cfg_level: # Should not happen
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 init_logger.warning(f"Config validation: Choice key '{key_path}' missing. Using default: {repr(def_level[leaf_key])}.")
                 return

             current_value = cfg_level[leaf_key]
             # Check if the value is a string and is in the allowed choices (case-insensitive by default)
             corrected_value = None
             found_match = False
             if isinstance(current_value, str):
                 for choice in choices:
                      if case_sensitive:
                          if current_value == choice:
                              corrected_value = choice # Use the exact match
                              found_match = True
                              break
                      else:
                          if current_value.lower() == choice.lower():
                              corrected_value = choice # Use the canonical casing from `choices` list
                              found_match = True
                              break

             if not found_match: # Not a valid choice (or wrong type)
                 init_logger.warning(f"Config Validation Warning: Invalid value for '{key_path}': {repr(current_value)}. Must be one of {choices} ({'case-sensitive' if case_sensitive else 'case-insensitive'}). Using default: {repr(def_level[leaf_key])}.")
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
             elif corrected_value != current_value: # Valid choice, but potentially wrong case (if case-insensitive)
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected case/value for '{key_path}' from '{current_value}' to '{corrected_value}'.{RESET}")
                 cfg_level[leaf_key] = corrected_value
                 config_needs_saving = True


        # --- Apply Validations to `updated_config` ---
        # == Trading Core ==
        pairs = updated_config.get("trading_pairs", [])
        if not isinstance(pairs, list) or not pairs or not all(isinstance(s, str) and s and '/' in s for s in pairs):
            init_logger.warning(f"{NEON_YELLOW}Config Validation Warning: Invalid 'trading_pairs'. Must be a non-empty list of strings in 'BASE/QUOTE' format (e.g., ['BTC/USDT']). Using default: {default_config['trading_pairs']}.{RESET}")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True

        validate_string_choice(updated_config, "interval", VALID_INTERVALS)
        validate_boolean(updated_config, "enable_trading")
        validate_boolean(updated_config, "use_sandbox")

        # Validate quote_currency (must be non-empty string, ideally uppercase)
        qc = updated_config.get("quote_currency")
        if not isinstance(qc, str) or not qc.strip():
            init_logger.warning(f"{NEON_YELLOW}Config Validation Warning: Invalid 'quote_currency': {repr(qc)}. Must be a non-empty string. Using default: '{default_config['quote_currency']}'.{RESET}")
            updated_config["quote_currency"] = default_config["quote_currency"]
            config_needs_saving = True
        else:
            # Normalize to uppercase for consistency
            qc_upper = qc.strip().upper()
            if qc_upper != updated_config["quote_currency"]:
                 init_logger.info(f"{NEON_YELLOW}Config Update: Normalized 'quote_currency' from '{updated_config['quote_currency']}' to '{qc_upper}'.{RESET}")
                 updated_config["quote_currency"] = qc_upper
                 config_needs_saving = True

        # Update the global QUOTE_CURRENCY immediately after validation/correction
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT") # Use updated value or default fallback
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")

        validate_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True) # Range 1 to 100 concurrent positions

        # == Risk & Sizing ==
        validate_numeric(updated_config, "risk_per_trade", 0.0, 1.0, is_strict_min=True) # Risk must be > 0% and <= 100%
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Allow 0 or 1 for spot, up to e.g. 200x for contracts (adjust max as needed)

        # == API & Timing ==
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True) # 1-60 seconds base retry delay
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True) # 1 second to 1 hour loop delay
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 120, is_int=True) # 1-120 seconds confirm delay

        # == Data Fetching ==
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True) # Min 50 klines, max MAX_DF_LEN
        validate_numeric(updated_config, "orderbook_limit", 1, 100, is_int=True) # 1-100 order book depth levels

        # == Strategy Params ==
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True) # ATR period <= max data length
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True) # Vol EMA period <= max data length
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0) # ATR multiplier range 0.1 to 20.0
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True) # Pivot lookback 1-100
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True) # Pivot lookforward 1-100
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True) # Pivot lookback 1-100
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True) # Pivot lookforward 1-100
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True) # Max OBs 1-500
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # Proximity factor 1.0 to 1.1 (adjust range as needed)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # Proximity factor 1.0 to 1.1 (adjust range as needed)
        validate_string_choice(updated_config, "strategy_params.ob_source", ["Wicks", "Body"], case_sensitive=False) # Case-insensitive check
        validate_boolean(updated_config, "strategy_params.ob_extend")

        # == Protection Params ==
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.0, 20.0, is_strict_min=True) # Initial SL distance must be > 0 ATRs
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0.0, 50.0, allow_zero=True) # Initial TP distance can be 0 (disabled) or positive
        validate_boolean(updated_config, "protection.enable_break_even")
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.0, 10.0, is_strict_min=True) # BE trigger must be > 0 ATRs in profit
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # BE offset ticks can be 0 or positive
        validate_boolean(updated_config, "protection.enable_trailing_stop")
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0, 0.2, is_strict_min=True) # TSL callback must be > 0 (e.g., 0-20%)
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0.0, 0.2, allow_zero=True) # TSL activation can be 0% (immediate once triggered) or positive

        init_logger.debug("Configuration parameter validation complete.")

        # --- Step 4: Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info(f"{NEON_YELLOW}Configuration updated with defaults or corrections. Saving changes to '{filepath}'...{RESET}")
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Config file '{filepath}' updated successfully.{RESET}")
             except Exception as save_err:
                 # Log error but continue with the corrected config in memory
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with the updated configuration in memory, but changes are NOT saved to the file.")

        init_logger.info(f"{Fore.CYAN}# Configuration loading and validation complete.{Style.RESET_ALL}")
        return updated_config # Return the fully processed config

    except Exception as e:
        # Catch any unexpected errors during the key ensuring or validation phase
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: An unexpected error occurred during config processing: {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        # Ensure QUOTE_CURRENCY is set from internal default even in this fatal case
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config # Fallback to internal defaults

# --- Load Configuration ---
# Call the function to load/validate/create the config file and store the result.
CONFIG = load_config(CONFIG_FILE)

# --- Exchange Initialization ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT exchange instance (Bybit) using API keys from environment
    and settings from the global CONFIG dictionary. Loads markets with retry logic
    and performs an initial balance check.

    Args:
        logger: The logger instance to use for initialization messages.

    Returns:
        A configured and verified `ccxt.bybit` instance if successful, otherwise None.
        Returns None on critical errors like authentication failure or inability to load markets.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Initializing Bybit Exchange Connection...{Style.RESET_ALL}")

    try:
        # --- Configure CCXT Exchange Options ---
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Use ccxt's built-in rate limiter
            'options': {
                # Prefer linear contracts (USDT margined) as default for Bybit V5 if type not specified
                'defaultType': 'linear',
                # Automatically adjust request timestamps for clock drift between client and server
                'adjustForTimeDifference': True,
                # Set reasonable timeouts for common operations (in milliseconds) to prevent hangs
                'fetchTickerTimeout': 15000,    # 15 seconds for fetching ticker
                'fetchBalanceTimeout': 25000,   # 25 seconds for fetching balance
                'createOrderTimeout': 35000,    # 35 seconds for creating orders
                'cancelOrderTimeout': 25000,    # 25 seconds for cancelling orders
                'fetchPositionsTimeout': 30000, # 30 seconds for fetching positions
                'fetchOHLCVTimeout': 60000,     # 60 seconds for fetching klines (can take longer for large history)
                # Bybit specific options (example, adjust if needed based on Bybit docs or errors)
                # 'recvWindow': 10000, # Optional: Increase receive window if timestamp-related errors occur
            }
        }
        # Explicitly create the Bybit instance using ccxt factory function
        exchange = ccxt.bybit(exchange_options)

        # Set Sandbox Mode based on config file setting
        is_sandbox = CONFIG.get('use_sandbox', True) # Default to sandbox for safety if key missing
        exchange.set_sandbox_mode(is_sandbox)
        env_type = "Sandbox/Testnet" if is_sandbox else "LIVE Trading"
        env_color = NEON_YELLOW if is_sandbox else NEON_RED + BRIGHT

        # Log the active environment prominently
        lg.warning(f"{env_color}!!! <<< {env_type} Environment ACTIVE >>> Exchange: {exchange.id} !!!{RESET}")
        # Add extra warnings for potentially dangerous configurations
        if not is_sandbox and not CONFIG.get('enable_trading'):
            lg.warning(f"{NEON_YELLOW}Warning: LIVE environment selected, but 'enable_trading' is FALSE in config. No live orders will be placed.{RESET}")
        elif is_sandbox and CONFIG.get('enable_trading'):
             lg.info(f"Note: 'enable_trading' is TRUE, but operating in SANDBOX mode. Orders will be placed on the testnet only.")
        elif not is_sandbox and CONFIG.get('enable_trading'):
              lg.critical(f"{NEON_RED}{BRIGHT}CRITICAL WARNING: LIVE TRADING IS ENABLED! REAL FUNDS ARE AT RISK!{RESET}")
              lg.critical(f"{NEON_RED}{BRIGHT}Ensure configuration and strategy are thoroughly tested before proceeding.{RESET}")


        # --- Load Market Data with Retry Logic ---
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error: Optional[Exception] = None
        # Use a loop with retries for loading markets, as it's crucial and can fail temporarily
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
                # Force reload on subsequent attempts (attempt > 0) to potentially fix temporary issues
                exchange.load_markets(reload=(attempt > 0))

                # Check if markets were loaded successfully and contain data
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market data loaded successfully. Found {len(exchange.markets)} symbols.{RESET}")
                    markets_loaded = True
                    break # Exit retry loop on success
                else:
                    # This case indicates an issue even if no exception was raised (e.g., empty response from API)
                    last_market_error = ValueError("Market data structure received from exchange is empty or invalid.")
                    lg.warning(f"Market data appears empty or invalid (Attempt {attempt + 1}). Retrying...")

            # Handle specific, potentially retryable errors
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"Network error loading markets (Attempt {attempt + 1}): {e}. Retrying...")
            except ccxt.RateLimitExceeded as e:
                 last_market_error = e
                 # Use a longer, potentially exponential delay specifically for rate limit errors
                 wait_time = RETRY_DELAY_SECONDS * (2 ** attempt + 2) # Exponential backoff + extra base time
                 lg.warning(f"{NEON_YELLOW}Rate limit exceeded loading markets: {e}. Waiting {wait_time}s before next attempt...{RESET}")
                 time.sleep(wait_time)
                 # Rate limit doesn't count as a standard attempt here, just wait and loop again
                 continue
            except ccxt.AuthenticationError as e:
                # Authentication errors are fatal and non-retryable
                last_market_error = e
                lg.critical(f"{NEON_RED}{BRIGHT}Authentication Error loading markets: {e}{RESET}")
                lg.critical(f"{NEON_RED}Please check your API Key, Secret, and ensure IP whitelist (if used) is correct. Exiting.{RESET}")
                return None # Exit initialization
            except ccxt.ExchangeNotAvailable as e:
                # Exchange maintenance or temporary unavailability - potentially retryable
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
                # Use exponential backoff for retry delay: delay = base * 2^attempt
                delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                lg.info(f"Waiting {delay}s before retrying market load...")
                time.sleep(delay)

        # Check if markets were loaded successfully after all retries
        if not markets_loaded:
            lg.critical(f"{NEON_RED}{BRIGHT}Failed to load market data after {MAX_API_RETRIES + 1} attempts.{RESET}")
            lg.critical(f"{NEON_RED}Last error encountered: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"Exchange initialized: {exchange.id} | CCXT Version: {ccxt.__version__} | Sandbox: {is_sandbox}")

        # --- Initial Balance Check ---
        # Use the globally configured quote currency for the primary balance check
        balance_currency = QUOTE_CURRENCY
        lg.info(f"Performing initial balance check for {balance_currency}...")
        initial_balance: Optional[Decimal] = None
        try:
            # Use the dedicated helper function for fetching balance
            # fetch_balance handles retries and parsing internally, but can raise AuthenticationError
            initial_balance = fetch_balance(exchange, balance_currency, lg)
        except ccxt.AuthenticationError as auth_err:
            # Catch auth error here again as fetch_balance might re-raise it
            lg.critical(f"{NEON_RED}{BRIGHT}Authentication Error during initial balance check: {auth_err}{RESET}")
            lg.critical(f"{NEON_RED}Cannot verify balance. Exiting.{RESET}")
            return None
        except Exception as balance_err:
            # Log other balance fetch errors as warnings, especially if trading is disabled
            # The fetch_balance function should have logged details already.
            lg.warning(f"{NEON_YELLOW}Initial balance check for {balance_currency} failed: {balance_err}{RESET}", exc_info=False)

        # Evaluate outcome of balance check
        if initial_balance is not None:
            # Successfully fetched balance
            lg.info(f"{NEON_GREEN}Initial balance check successful: {initial_balance.normalize()} {balance_currency}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange initialization complete.{Style.RESET_ALL}")
            return exchange
        else:
            # Balance check failed (fetch_balance returned None after retries)
            lg.error(f"{NEON_RED}Initial balance check FAILED for {balance_currency}. Could not retrieve balance after retries.{RESET}")
            # Decide whether to proceed based on 'enable_trading' flag
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}{BRIGHT}Trading is ENABLED, but the initial balance check failed.{RESET}")
                lg.critical(f"{NEON_RED}Cannot proceed safely without confirming available balance. Exiting.{RESET}")
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is DISABLED. Proceeding cautiously without initial balance confirmation.{RESET}")
                lg.info(f"{Fore.CYAN}# Exchange initialization complete (Warning: Balance check failed, Trading Disabled).{Style.RESET_ALL}")
                return exchange # Allow proceeding only if trading is off

    except ccxt.AuthenticationError as e:
         # Catch auth errors during the initial setup phase (e.g., ccxt.bybit() call)
         lg.critical(f"{NEON_RED}{BRIGHT}Authentication error during exchange setup: {e}. Exiting.{RESET}")
         return None
    except Exception as e:
        # Catch any other unexpected critical errors during the initialization process
        lg.critical(f"{NEON_RED}{BRIGHT}A critical error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Helper Functions (Enhanced & Refined) ---

def _safe_market_decimal(value: Optional[Any], field_name: str,
                         allow_zero: bool = True, allow_negative: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value (often from market, position, or order data) into a Decimal object.
    Handles None, empty strings, non-finite numbers (NaN, Inf), and applies configurable
    checks for zero and negative values based on the context.

    Args:
        value: The input value to convert (can be string, int, float, Decimal, None, etc.).
        field_name: Name of the field being converted (used for logging context on failure).
        allow_zero: If True, Decimal('0') is considered a valid result.
        allow_negative: If True, negative Decimal values are considered valid.

    Returns:
        The converted Decimal value if valid according to the rules, otherwise None.
        Logs failures at DEBUG level to avoid excessive noise for optional fields.
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

        # Attempt conversion to Decimal
        d_val = Decimal(s_val)

        # Check for non-finite values (NaN, Infinity) which are generally invalid for market data
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

        # If all checks pass, return the valid Decimal object
        return d_val
    except (InvalidOperation, TypeError, ValueError) as e:
        # Log conversion errors at debug level as they can be frequent with optional or unexpected fields from APIs
        # init_logger.debug(f"SafeDecimal: Failed to convert '{field_name}' to Decimal. Input: {repr(value)}, Error: {e}")
        return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price value according to the market's price precision rules using ccxt.
    - Ensures the input price is a valid positive Decimal before formatting.
    - Uses `exchange.price_to_precision()` for formatting.
    - Validates that the formatted string still represents a positive number.

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
        # ccxt's method typically requires a float argument, convert the validated Decimal.
        # Note: This float conversion might lose precision for very large/small numbers, but is often required by ccxt methods.
        # Ensure the Decimal is finite before converting.
        if not price_decimal.is_finite():
             raise ValueError("Input Decimal price is not finite.")
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))

        # 3. Post-Validation: Ensure formatted string is still a valid positive number string
        # This catches cases where precision might round down to zero or near-zero, or create invalid formats.
        # Use allow_zero=False here as prices are expected to be positive.
        formatted_decimal = _safe_market_decimal(formatted_str, f"format_price_output({symbol})", allow_zero=False, allow_negative=False)
        if formatted_decimal is None:
             init_logger.warning(f"Price formatting warning ({symbol}): Input '{price}' formatted to non-positive or invalid value '{formatted_str}' using market precision. Returning None.")
             return None

        # Return the successfully formatted and validated price string
        return formatted_str
    except ccxt.BadSymbol:
        init_logger.error(f"Price formatting failed ({symbol}): Symbol not found on exchange '{exchange.id}'. Ensure markets are loaded and symbol is correct.")
        return None
    except ccxt.ExchangeError as e:
        init_logger.error(f"Price formatting failed ({symbol}): Exchange error accessing market precision: {e}.")
        return None
    except (InvalidOperation, ValueError, TypeError) as e:
        # Catches errors during Decimal validation, float conversion, or potential issues within ccxt method
        init_logger.warning(f"Price formatting failed ({symbol}): Error during formatting process for price '{price}': {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during formatting
        init_logger.warning(f"Price formatting failed ({symbol}): Unexpected error for price '{price}': {e}", exc_info=True)
        return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using ccxt's `fetch_ticker`.
    Attempts to use price sources in order of preference: 'last', mid-price ('bid'/'ask'), 'ask', 'bid'.
    Includes retry logic with exponential backoff for network/exchange errors.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        logger: The logger instance for messages specific to this operation.

    Returns:
        The current price as a Decimal object if successfully fetched and valid (positive), otherwise None.
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

            # Helper to safely get a positive Decimal from ticker data using our robust function
            def safe_positive_decimal_from_ticker(key: str) -> Optional[Decimal]:
                return _safe_market_decimal(ticker.get(key), f"ticker.{key} ({symbol})", allow_zero=False, allow_negative=False)

            # 1. Try 'last' price (most recent trade price)
            price = safe_positive_decimal_from_ticker('last')
            if price:
                source = "'last' price"
            else:
                # 2. Try mid-price derived from 'bid' and 'ask'
                bid = safe_positive_decimal_from_ticker('bid')
                ask = safe_positive_decimal_from_ticker('ask')
                if bid and ask:
                    # Sanity check: ensure bid < ask before calculating mid-price
                    if bid < ask:
                        price = (bid + ask) / Decimal('2')
                        # Optional: Quantize mid-price to price tick precision for consistency?
                        # price_tick = _safe_market_decimal(exchange.markets[symbol]['precision']['price'], 'price_tick')
                        # if price_tick: price = price.quantize(price_tick, ROUND_HALF_UP)
                        source = f"mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()})"
                    else:
                        # If bid >= ask (crossed or equal book), something is unusual.
                        # Prefer 'ask' price as a generally safer estimate for buying/covering short.
                        price = ask
                        source = f"'ask' price (used due to crossed/equal book: Bid={bid}, Ask={ask})"
                        lg.warning(f"Crossed/equal order book detected for {symbol} (Bid >= Ask). Using Ask price as fallback.")
                elif ask:
                    # 3. Fallback to 'ask' price if only ask is available/valid
                    price = ask
                    source = f"'ask' price ({ask.normalize()})"
                elif bid:
                    # 4. Fallback to 'bid' price if only bid is available/valid
                    price = bid
                    source = f"'bid' price ({bid.normalize()})"

            # --- Return Valid Price or Log Warning and Continue Retry ---
            if price:
                normalized_price = price.normalize() # Remove trailing zeros for cleaner logging
                lg.debug(f"Current price ({symbol}) obtained from {source}: {normalized_price}")
                return normalized_price # Success
            else:
                # Ticker was fetched, but no usable price source (last/bid/ask) found or validated.
                last_exception = ValueError(f"No valid, positive price source (last/bid/ask) found in ticker response for {symbol}.")
                lg.warning(f"Could not find a valid price in ticker data ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}")
                # Let the loop continue to the retry logic below

        # --- Error Handling for the fetch_ticker API Call ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Use a longer, potentially exponential delay for rate limit errors
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 2) # Exponential backoff + extra base
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Rate limit doesn't count as a standard attempt, just wait and loop again
            continue
        except ccxt.AuthenticationError as e:
            # Fatal error, non-retryable
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching price: {e}. Cannot continue.{RESET}")
            return None
        except ccxt.BadSymbol as e:
             # Fatal error for this specific symbol
             last_exception = e
             lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching price on {exchange.id}: {e}.{RESET}")
             return None
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., maintenance, temporary issues) - potentially retryable
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ValueError as e: # Catch our internal validation error
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Data validation error after fetching ticker ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors during price fetching
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price ({symbol}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as potentially fatal for safety in this context
            return None

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            # Use exponential backoff for retry delay
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1))
            lg.info(f"Waiting {delay}s before retrying price fetch for {symbol}...")
            time.sleep(delay)

    # If loop finishes without successfully fetching a price
    lg.error(f"{NEON_RED}Failed to fetch current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches historical kline/OHLCV data for a symbol using ccxt's `fetch_ohlcv`.
    Handles exchange-specific pagination requirements (e.g., Bybit V5 'until' parameter),
    data validation, lag checks on recent data, deduplication, sorting, and length trimming.
    Includes robust retry logic for network and temporary exchange errors.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        timeframe: The CCXT timeframe string (e.g., '5m', '1h', '1d').
        limit: The total number of candles desired (target count, actual may vary slightly).
        logger: The logger instance for messages specific to this operation.

    Returns:
        A pandas DataFrame containing the OHLCV data, indexed by UTC timestamp (pandas Timestamp).
        Columns: ['open', 'high', 'low', 'close', 'volume'] as Decimal type.
        Returns an empty DataFrame if fetching fails critically or data is invalid/insufficient.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Fetching klines for {symbol} | Timeframe: {timeframe} | Target Limit: {limit}...{Style.RESET_ALL}")

    # --- Pre-checks ---
    # Check if the exchange instance supports fetching OHLCV data via ccxt
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetching OHLCV data via ccxt's standard method.")
        return pd.DataFrame()

    # Estimate minimum candles needed based on strategy params (best effort for warning)
    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        # Find the max lookback period needed by indicators used in the strategy
        # Multiply EMA/SWMA lengths by 2 for stability buffer. Add 1 for ATR/pivots.
        min_required = max(
            sp.get('vt_length', 0) * 2,
            sp.get('vt_atr_period', 0) + 1,
            sp.get('vt_vol_ema_length', 0) * 2,
            sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1,
            sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1
        ) + 50 # Add a generous safety buffer for calculations & stability
        lg.debug(f"Estimated minimum candles required by strategy logic: ~{min_required}")
    except Exception as e:
        lg.warning(f"Could not estimate minimum candle requirement for {symbol}: {e}")

    if limit < min_required:
        lg.warning(f"{NEON_YELLOW}Requested kline limit ({limit}) is less than the estimated strategy requirement ({min_required}) for {symbol}. "
                   f"Indicator accuracy may be affected, especially on initial data load.{RESET}")

    # Determine category and market ID for Bybit V5 API (required params)
    category = 'spot' # Default assumption
    market_id = symbol # Default to symbol if market info lookup fails
    is_bybit = 'bybit' in exchange.id.lower()
    try:
        # Fetch market info to determine type (linear/inverse/spot) and specific ID
        market = exchange.market(symbol) # Raises BadSymbol if not found
        market_id = market['id'] # Use exchange-specific ID for API calls
        # Determine category based on market type flags
        if market.get('linear'): category = 'linear'
        elif market.get('inverse'): category = 'inverse'
        elif market.get('spot'): category = 'spot'
        # else category remains default (e.g., 'spot' or 'linear' based on exchange.options['defaultType'])
        lg.debug(f"Using API parameters for {symbol} kline fetch: category='{category}', market ID='{market_id}' (relevant for Bybit V5).")
    except (ccxt.BadSymbol, KeyError, TypeError) as e:
        lg.warning(f"Could not reliably determine market category/ID for {symbol} kline fetch: {e}. "
                   f"Proceeding with defaults (category='{category}', market_id='{market_id}'). May fail if incorrect for Bybit V5 API.")

    # --- Fetching Loop for Pagination ---
    all_ohlcv_data: List[List] = [] # Stores raw candle lists [[ts, o, h, l, c, v], ...] from oldest to newest
    remaining_limit = limit
    end_timestamp_ms: Optional[int] = None # For pagination: fetch candles ending *before* this timestamp (exclusive)

    # Determine the maximum number of candles the API returns per request from exchange limits
    api_limit_per_req = getattr(exchange, 'limits', {}).get('fetchOHLCV', {}).get('limit', BYBIT_API_KLINE_LIMIT)
    # Fallback if limit info is missing or invalid
    if api_limit_per_req is None or not isinstance(api_limit_per_req, int) or api_limit_per_req <= 0:
        lg.warning(f"Could not determine API limit per request for OHLCV on {exchange.id}. Using default: {BYBIT_API_KLINE_LIMIT}")
        api_limit_per_req = BYBIT_API_KLINE_LIMIT
    lg.debug(f"API kline fetch limit per request: {api_limit_per_req}")

    # Calculate max chunks generously to prevent potential infinite loops if API behaves unexpectedly
    max_chunks = math.ceil(limit / api_limit_per_req) + 5 # Add a buffer of 5 extra chunks
    chunk_num = 0
    total_fetched_raw = 0

    # Loop until we have fetched the desired limit or hit max chunks or end of history
    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        # Determine the number of candles to request in this chunk
        fetch_size = min(remaining_limit, api_limit_per_req)
        end_ts_str = (f"ending before TS: {datetime.fromtimestamp(end_timestamp_ms / 1000, tz=timezone.utc).isoformat()}"
                      if end_timestamp_ms else "requesting latest")
        lg.debug(f"Fetching kline chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. ({end_ts_str})")

        attempts = 0
        last_exception: Optional[Exception] = None
        chunk_data: Optional[List[List]] = None

        # Retry loop for fetching a single chunk
        while attempts <= MAX_API_RETRIES:
            try:
                # --- Prepare API Call Arguments ---
                params = {'category': category} if is_bybit else {} # Pass category for Bybit V5 if applicable
                fetch_args: Dict[str, Any] = {
                    'symbol': symbol,       # Use standard symbol for ccxt call
                    'timeframe': timeframe,
                    'limit': fetch_size,    # Number of candles for this chunk
                    'params': params        # Include category etc. here
                }
                # Add 'until' parameter for pagination (fetches candles ending *before* this timestamp)
                # CCXT typically handles mapping 'until' to exchange-specific params like 'end' for Bybit V5
                if end_timestamp_ms:
                    fetch_args['until'] = end_timestamp_ms

                # --- Execute API Call ---
                lg.debug(f"Calling exchange.fetch_ohlcv with args: {fetch_args}")
                fetched_chunk = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(fetched_chunk) if fetched_chunk else 0
                lg.debug(f"API returned {fetched_count_chunk} raw candles for chunk {chunk_num}.")

                # --- Basic Validation of Fetched Chunk ---
                if fetched_chunk:
                    # Check if data format looks valid (list of lists with at least 6 elements: ts,o,h,l,c,v)
                    if not all(isinstance(candle, list) and len(candle) >= 6 for candle in fetched_chunk):
                        # Raise error to trigger retry, as format is wrong
                        raise ValueError(f"Invalid candle format received in chunk {chunk_num} for {symbol}. Expected list with >= 6 values per candle.")

                    chunk_data = fetched_chunk # Assign valid data

                    # --- Data Lag Check (only on the first chunk when fetching latest data) ---
                    # This helps catch cases where the exchange API is serving stale data.
                    if chunk_num == 1 and end_timestamp_ms is None:
                        try:
                            # Timestamp of the most recent candle in the chunk (should be the last element)
                            last_candle_ts_ms = chunk_data[-1][0]
                            last_ts = pd.to_datetime(last_candle_ts_ms, unit='ms', utc=True, errors='raise')
                            interval_seconds = exchange.parse_timeframe(timeframe) # Convert timeframe string to seconds

                            if interval_seconds:
                                # Allow up to ~2.5 intervals of lag before warning/retrying
                                # Adjust multiplier as needed based on exchange reliability and timeframe sensitivity
                                max_lag_seconds = interval_seconds * 2.5
                                current_utc_time = pd.Timestamp.utcnow() # Use pandas for consistency
                                actual_lag_seconds = (current_utc_time - last_ts).total_seconds()

                                if actual_lag_seconds > max_lag_seconds:
                                    lag_error_msg = (f"Potential data lag detected! Last candle time ({last_ts}) is "
                                                     f"{actual_lag_seconds:.1f}s old. Max allowed lag for {timeframe} is ~{max_lag_seconds:.1f}s.")
                                    # Treat lag as a potentially retryable error
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
                        except (IndexError, TypeError, ValueError, pd.errors.OutOfBoundsDatetime) as ts_err:
                             lg.warning(f"Could not perform lag check: Error processing timestamp in first chunk ({symbol}): {ts_err}")
                             break # Proceed if lag check itself fails, but use the data received
                    else: # Not the first chunk or not fetching latest, no lag check needed
                        break # Valid chunk received, exit retry loop

                else: # API returned empty list []
                    lg.debug(f"API returned no data (empty list) for chunk {chunk_num} ({symbol}).")
                    # If it's the *first* chunk, it might be a temporary issue or invalid symbol/params -> retry.
                    # If it's a *later* chunk during pagination, it likely means we've reached the end of available historical data.
                    if chunk_num > 1:
                        lg.info(f"Assuming end of historical data for {symbol} after API returned empty chunk {chunk_num}.")
                        remaining_limit = 0 # Stop fetching more chunks
                        chunk_data = [] # Ensure chunk_data is empty list, not None
                        break # Exit retry loop for this (empty) chunk, proceed to next stage
                    else:
                        # First chunk was empty, treat as potential error and let retry logic handle it
                         last_exception = ValueError("API returned empty list for the first kline chunk request.")
                         # No break here, continue to retry logic below

            # --- Error Handling for fetch_ohlcv Call ---
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_exception = e
                lg.warning(f"{NEON_YELLOW}Network error fetching klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 2) # Exponential backoff + extra base
                lg.warning(f"{NEON_YELLOW}Rate limit fetching klines chunk {chunk_num} ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
                time.sleep(wait_time)
                continue # Don't increment standard attempts, just wait
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Authentication error fetching klines: {e}. Cannot continue.{RESET}")
                return pd.DataFrame() # Fatal error
            except ccxt.BadSymbol as e:
                 last_exception = e
                 lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching klines on {exchange.id}: {e}.{RESET}")
                 return pd.DataFrame() # Fatal for this symbol
            except ccxt.ExchangeError as e:
                last_exception = e
                # Check for specific non-retryable errors (e.g., invalid timeframe, invalid category)
                err_str = str(e).lower()
                # List common messages indicating parameters are wrong and retrying won't help
                non_retryable_msgs = [
                    "invalid timeframe", "interval not supported", "symbol invalid", "instrument not found",
                    "invalid category", "market is closed", "parameter error", "invalid argument",
                    # Bybit V5 specific error codes/substrings might be added here if identified
                    "3400000", # Bybit V5 generic parameter error example
                ]
                if any(msg in err_str for msg in non_retryable_msgs):
                    lg.critical(f"{NEON_RED}Non-retryable exchange error fetching klines for {symbol}: {e}. Stopping kline fetch.{RESET}")
                    return pd.DataFrame() # Fatal for this symbol/timeframe combination
                else:
                    # Treat other exchange errors as potentially temporary and retryable
                    lg.warning(f"{NEON_YELLOW}Exchange error fetching klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ValueError as e: # Catch our internal validation errors (e.g., candle format, lag)
                 last_exception = e
                 # These are often retryable if caused by temporary API glitches
                 lg.error(f"{NEON_RED}Data validation error fetching klines chunk {chunk_num} ({symbol}): {e}. Retrying...{RESET}")
            except Exception as e:
                last_exception = e
                lg.error(f"{NEON_RED}Unexpected error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True)
                # Treat unexpected errors cautiously - stop fetching for this symbol for safety
                return pd.DataFrame()

            # --- Retry Logic ---
            attempts += 1
            # Only sleep if we need to retry (chunk_data is None or lag detected) and more attempts left
            if chunk_data is None and attempts <= MAX_API_RETRIES:
                delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
                lg.info(f"Waiting {delay}s before retrying kline chunk {chunk_num}...")
                time.sleep(delay)

        # --- Process Successful Chunk or Handle Failure for this Chunk ---
        if chunk_data is not None: # Check includes case where chunk_data became [] after empty fetch on chunk > 1
            if chunk_data: # Only process if the chunk actually contains data
                # Prepend the new chunk to maintain chronological order in the final list
                # (Oldest fetched chunk appears first in `all_ohlcv_data`)
                all_ohlcv_data = chunk_data + all_ohlcv_data
                chunk_len = len(chunk_data)
                remaining_limit -= chunk_len
                total_fetched_raw += chunk_len

                # Set timestamp for the next older chunk request based on the *oldest* candle in this chunk
                # Use the timestamp of the *first* candle in the *current* chunk (index 0)
                try:
                    next_until_ts = chunk_data[0][0] # Timestamp of the oldest candle in this chunk
                    if not isinstance(next_until_ts, (int, float)) or next_until_ts <= 0:
                        raise ValueError(f"Invalid timestamp found in first candle of chunk: {next_until_ts}")
                    # 'until' is exclusive, so we want candles ending *before* this timestamp.
                    # The timestamp itself serves as the exclusive boundary.
                    end_timestamp_ms = int(next_until_ts)
                except (IndexError, TypeError, ValueError) as ts_err:
                    lg.error(f"Error determining next 'until' timestamp from chunk data ({symbol}): {ts_err}. Stopping pagination.")
                    remaining_limit = 0 # Stop fetching if we can't paginate correctly

                # Check if the exchange returned fewer candles than requested (might indicate end of history)
                if chunk_len < fetch_size:
                    lg.debug(f"Received fewer candles ({chunk_len}) than requested ({fetch_size}) for chunk {chunk_num}. Assuming end of historical data.")
                    remaining_limit = 0 # Stop fetching more chunks
            else:
                 # Chunk data was explicitly set to [] (e.g., empty response on chunk > 1), do nothing here, loop condition will handle exit.
                 lg.debug("Empty chunk processed (likely end of history).")

        else: # Failed to fetch this chunk after all retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            lg.error(f"  Last error for chunk {chunk_num}: {last_exception}")
            if not all_ohlcv_data:
                # Failed on the very first chunk, cannot proceed
                lg.error(f"Failed to fetch the initial kline chunk for {symbol}. Cannot construct DataFrame.")
                return pd.DataFrame()
            else:
                # Failed on a subsequent chunk, proceed with the data collected so far
                lg.warning(f"Proceeding with {total_fetched_raw} raw candles fetched before the error occurred on chunk {chunk_num}.")
                break # Exit the main fetching loop

        # Small delay between chunk requests to be polite to the API, especially if looping many times
        if remaining_limit > 0 and chunk_num < max_chunks:
            time.sleep(0.3) # 300ms delay seems reasonable

    # --- Post-Fetching Checks ---
    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines for {symbol} because maximum chunk limit ({max_chunks}) was reached. "
                   f"Fetched {total_fetched_raw} raw candles. Requested limit was {limit}.")

    if not all_ohlcv_data:
        lg.error(f"No kline data could be successfully fetched for {symbol} {timeframe}.")
        return pd.DataFrame()

    lg.info(f"Total raw klines fetched across all chunks: {len(all_ohlcv_data)}")

    # --- Data Deduplication and Sorting ---
    # Use a dictionary keyed by timestamp to automatically handle duplicates; keeps the last seen entry for a given timestamp.
    unique_candles_dict: Dict[int, List] = {}
    invalid_candle_count = 0
    for candle in all_ohlcv_data:
        try:
            # Validate candle structure and timestamp before adding to dict
            if not isinstance(candle, list) or len(candle) < 6:
                 invalid_candle_count += 1
                 continue # Skip malformed candle
            timestamp = int(candle[0])
            if timestamp <= 0:
                 invalid_candle_count += 1
                 continue # Skip candle with invalid timestamp
            # Store/overwrite candle in dict using timestamp as key
            unique_candles_dict[timestamp] = candle
        except (IndexError, TypeError, ValueError):
            # Catch errors during extraction or int conversion
            invalid_candle_count += 1
            continue # Skip candles with invalid format or timestamp

    if invalid_candle_count > 0:
         lg.warning(f"Skipped {invalid_candle_count} invalid raw candle entries during deduplication for {symbol}.")

    # Extract unique candles from the dictionary values and sort them chronologically by timestamp
    unique_data = sorted(list(unique_candles_dict.values()), key=lambda x: x[0])
    final_unique_count = len(unique_data)

    duplicates_removed = len(all_ohlcv_data) - invalid_candle_count - final_unique_count
    if duplicates_removed > 0:
        lg.info(f"Removed {duplicates_removed} duplicate candle(s) based on timestamp for {symbol}.")
    elif duplicates_removed < 0: # Should not happen with the dictionary deduplication method
         lg.warning(f"Data count mismatch during deduplication ({symbol}). Raw: {len(all_ohlcv_data)}, Invalid: {invalid_candle_count}, Final Unique: {final_unique_count}")

    if not unique_data:
        lg.error(f"No valid, unique kline data remaining after processing for {symbol}.")
        return pd.DataFrame()

    # Trim excess data if more than the originally requested limit was fetched due to chunking overlaps
    # Keep the most recent 'limit' candles by slicing from the end.
    if final_unique_count > limit:
        lg.debug(f"Fetched {final_unique_count} unique candles, trimming to the target limit of {limit} (keeping most recent).")
        unique_data = unique_data[-limit:]
        final_unique_count = len(unique_data) # Update count after trimming

    # --- DataFrame Creation and Cleaning ---
    try:
        lg.debug(f"Processing {final_unique_count} final unique candles into DataFrame for {symbol}...")
        # Standard OHLCV columns expected by most strategies
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Use only the columns available in the raw data (up to the first 6)
        df = pd.DataFrame(unique_data, columns=cols[:len(unique_data[0])])

        # Convert timestamp column to pandas Datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        initial_len_ts = len(df)
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        rows_dropped_ts = initial_len_ts - len(df)
        if rows_dropped_ts > 0:
            lg.warning(f"Dropped {rows_dropped_ts} rows with invalid timestamps during conversion for {symbol}.")
        if df.empty:
            lg.error(f"DataFrame became empty after timestamp conversion and NaN drop ({symbol}).")
            return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision, handling potential errors during conversion
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Apply the robust _safe_market_decimal conversion to each element
                # Allow zero only for volume, disallow negative for all OHLCV
                # Store result as object type initially to hold Decimals
                df[col] = df[col].apply(
                    lambda x: _safe_market_decimal(x, f"df.{col}", allow_zero=(col=='volume'), allow_negative=False)
                ).astype('object') # Keep as object dtype to hold Decimals
                # Check if conversion resulted in all NaNs (indicates bad input data type incompatible with Decimal)
                if df[col].isnull().all():
                     lg.warning(f"Column '{col}' for {symbol} became all NaN after Decimal conversion. Original data might be incompatible (e.g., all strings).")
            elif col != 'volume': # Volume might legitimately be missing from some exchanges/endpoints, but OHLC are essential
                lg.error(f"Essential OHLC column '{col}' not found in fetched data for {symbol}. Cannot proceed.")
                return pd.DataFrame() # Fail if essential OHLC columns are missing

        # --- Data Cleaning: Drop rows with NaN/None in essential Decimal columns ---
        initial_len_clean = len(df)
        # Define essential columns that must not be NaN/None after Decimal conversion
        essential_cols_present = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        # Also ensure volume is present and non-NaN/None if the column exists
        if 'volume' in df.columns:
            essential_cols_present.append('volume')

        df.dropna(subset=essential_cols_present, inplace=True)

        rows_dropped_clean = initial_len_clean - len(df)
        if rows_dropped_clean > 0:
            lg.debug(f"Dropped {rows_dropped_clean} rows with NaN/invalid OHLCV data during cleaning for {symbol}.")

        if df.empty:
            lg.warning(f"DataFrame became empty after cleaning NaN/invalid values ({symbol}).")
            return pd.DataFrame()

        # Verify index is sorted chronologically (should be due to sorting earlier, but double-check)
        if not df.index.is_monotonic_increasing:
            lg.warning(f"DataFrame index for {symbol} is not monotonically increasing after processing. Sorting index...")
            df.sort_index(inplace=True)

        # Optional: Limit final DataFrame length to prevent excessive memory usage over long runs
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds max configured length ({MAX_DF_LEN}). Trimming oldest data ({symbol}).")
            df = df.iloc[-MAX_DF_LEN:] # Keep only the most recent MAX_DF_LEN rows

        lg.info(f"{NEON_GREEN}Successfully processed {len(df)} klines into DataFrame for {symbol} {timeframe}.{RESET}")
        # Log head/tail at DEBUG level for verification if needed
        # lg.debug(f"DataFrame Head ({symbol}):\n{df.head().to_string(max_colwidth=15)}")
        # lg.debug(f"DataFrame Tail ({symbol}):\n{df.tail().to_string(max_colwidth=15)}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing fetched klines into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Retrieves, standardizes, and enhances market information for a given symbol from the exchange.
    - Uses the cached `exchange.markets` data primarily.
    - Attempts a market refresh if the symbol is not found initially.
    - Parses critical information like precision and limits into Decimal types.
    - Derives convenience flags (e.g., `is_linear`, `contract_type_str`).
    - Includes retry logic for potential temporary issues during market refresh.
    - Validates essential fields like precision steps.

    Args:
        exchange: The initialized ccxt exchange instance (markets should ideally be loaded).
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        logger: The logger instance for messages specific to this operation.

    Returns:
        A MarketInfo TypedDict containing standardized and enhanced market data if found and valid,
        otherwise None. Returns None immediately if the symbol is definitively not supported
        or if critical validation fails.
    """
    lg = logger
    lg.debug(f"Retrieving market details for symbol: {symbol}...")
    attempts = 0
    max_refresh_attempts = 1 # Only attempt market refresh once if symbol not found initially
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        market_dict: Optional[Dict] = None
        market_found_in_cache = False
        try:
            # 1. Try to get market from cache first
            if exchange.markets and symbol in exchange.markets:
                market_dict = exchange.markets[symbol]
                market_found_in_cache = True
                lg.debug(f"Market info for '{symbol}' found in cache.")
            elif attempts < max_refresh_attempts: # Only try reloading once if not found in cache
                # 2. If not in cache, attempt to refresh market map
                lg.info(f"Market details for '{symbol}' not found in cache. Attempting to refresh market map (Attempt {attempts + 1})...")
                try:
                    exchange.load_markets(reload=True) # Force reload
                    lg.info(f"Market map refreshed. Found {len(exchange.markets)} markets.")
                    # Try accessing from cache again after reload
                    if exchange.markets and symbol in exchange.markets:
                        market_dict = exchange.markets[symbol]
                        lg.debug(f"Market info for '{symbol}' found after refresh.")
                    else:
                         # Symbol not found even after refresh - likely invalid/unsupported
                         raise ccxt.BadSymbol(f"Symbol '{symbol}' not found on {exchange.id} after explicit market refresh.")
                except ccxt.BadSymbol as e:
                    # This is definitive: symbol doesn't exist on the exchange
                    lg.error(f"{NEON_RED}{e}{RESET}")
                    return None # Non-retryable failure for this symbol
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError) as reload_err:
                    # Log refresh error but continue to main retry logic below
                    last_exception = reload_err
                    lg.warning(f"Failed to refresh market map while looking for '{symbol}': {reload_err}. Will retry entire process.")
                    # market_dict remains None, main loop will retry
                except Exception as reload_err:
                     # Unexpected error during refresh
                     last_exception = reload_err
                     lg.error(f"Unexpected error refreshing markets for '{symbol}': {reload_err}", exc_info=True)
                     return None # Treat unexpected refresh errors as fatal
            else:
                 # Not found in cache, and refresh was already attempted or not applicable
                 lg.debug(f"Market '{symbol}' still not found after previous attempts/refresh.")
                 # Allow loop to finish and report failure based on last_exception or BadSymbol if raised
                 # If loop finishes without market_dict, it will fail below.
                 if not last_exception: # Ensure an error exists if we reach the end without finding the market
                      last_exception = ccxt.BadSymbol(f"Symbol '{symbol}' not found on {exchange.id} after all checks.")


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
                is_contract_base = std_market.get('contract', False) # Base 'contract' flag from ccxt

                # Determine if it's any kind of contract using combined flags
                std_market['is_contract'] = is_swap or is_future or is_option or is_contract_base
                is_linear = std_market.get('linear') # Can be True/False/None
                is_inverse = std_market.get('inverse') # Can be True/False/None

                # Ensure derived linear/inverse flags are boolean and only True if it's actually a contract market
                std_market['is_linear'] = bool(is_linear) and std_market['is_contract']
                std_market['is_inverse'] = bool(is_inverse) and std_market['is_contract']

                # Determine a user-friendly contract type string for logging/logic
                if std_market['is_linear']:
                    std_market['contract_type_str'] = "Linear"
                elif std_market['is_inverse']:
                    std_market['contract_type_str'] = "Inverse"
                elif is_spot:
                     std_market['contract_type_str'] = "Spot"
                elif is_option:
                     std_market['contract_type_str'] = "Option"
                elif std_market['is_contract']: # Catch-all for other contract types marked by base flag
                     std_market['contract_type_str'] = "Contract (Other)"
                else:
                     std_market['contract_type_str'] = "Unknown"

                # --- Extract Precision and Limits Safely using Helper ---
                precision = std_market.get('precision', {})
                limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                # Convert precision steps (tick sizes) to Decimal (must be positive)
                # These are CRITICAL for placing orders correctly.
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), f"{symbol} precision.amount", allow_zero=False, allow_negative=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), f"{symbol} precision.price", allow_zero=False, allow_negative=False)

                # Convert order limits to Decimal
                # Amounts must be non-negative (min can be 0)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), f"{symbol} limits.amount.min", allow_zero=True, allow_negative=False)
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), f"{symbol} limits.amount.max", allow_zero=False, allow_negative=False) # Max generally must be > 0 if specified
                # Costs must be non-negative (min can be 0)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), f"{symbol} limits.cost.min", allow_zero=True, allow_negative=False)
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), f"{symbol} limits.cost.max", allow_zero=False, allow_negative=False) # Max generally must be > 0 if specified

                # Convert contract size to Decimal (default to 1 if missing/invalid/spot)
                # Contract size must be positive if specified for contracts.
                contract_size_val = std_market.get('contractSize') if std_market['is_contract'] else '1'
                # Use Decimal('1') as fallback if conversion fails or it's not a contract
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, f"{symbol} contractSize", allow_zero=False, allow_negative=False) or Decimal('1')

                # --- Validation of Critical Extracted Data ---
                # Precision steps are absolutely essential for placing orders correctly. Fail if missing/invalid.
                if std_market['amount_precision_step_decimal'] is None:
                    raise ValueError(f"CRITICAL VALIDATION FAILED ({symbol}): Missing or invalid 'precision.amount' step data! Cannot determine order size precision.")
                if std_market['price_precision_step_decimal'] is None:
                    raise ValueError(f"CRITICAL VALIDATION FAILED ({symbol}): Missing or invalid 'precision.price' step data! Cannot determine price precision.")
                # Min amount is often needed for order placement validation. Warn if missing, but don't fail outright.
                if std_market['min_amount_decimal'] is None:
                     lg.warning(f"{NEON_YELLOW}Market Validation Warning ({symbol}): Missing 'limits.amount.min' data. Order sizing/placement might fail later if calculated size is below exchange minimum.{RESET}")

                # --- Log Parsed Details for Confirmation ---
                # Helper for formatting optional Decimals for logging, avoids errors with None
                def fmt_dec_log(d: Optional[Decimal]) -> str:
                    return str(d.normalize()) if d is not None and d.is_finite() else 'N/A'

                amt_step_str = fmt_dec_log(std_market['amount_precision_step_decimal'])
                price_step_str = fmt_dec_log(std_market['price_precision_step_decimal'])
                min_amt_str = fmt_dec_log(std_market['min_amount_decimal'])
                max_amt_str = fmt_dec_log(std_market['max_amount_decimal'])
                min_cost_str = fmt_dec_log(std_market['min_cost_decimal'])
                max_cost_str = fmt_dec_log(std_market['max_cost_decimal'])
                contr_size_str = fmt_dec_log(std_market['contract_size_decimal'])
                active_status = std_market.get('active') # Handle potentially missing 'active' key
                active_str = str(active_status) if active_status is not None else 'Unknown'

                log_msg = (
                    f"Market Details Parsed ({symbol}): Type={std_market['contract_type_str']}, Active={active_str}\n"
                    f"  Precision: Amount Step={amt_step_str}, Price Step={price_step_str}\n"
                    f"  Limits Amt (Min/Max): {min_amt_str} / {max_amt_str} {std_market.get('base', '')}\n"
                    f"  Limits Cost(Min/Max): {min_cost_str} / {max_cost_str} {std_market.get('quote', '')}"
                )
                if std_market['is_contract']:
                     log_msg += f"\n  Contract Size: {contr_size_str}"
                lg.info(log_msg) # Log at INFO level as it's important setup information for the symbol

                # --- Cast to TypedDict and Return ---
                try:
                    # Attempt to cast the enhanced dictionary to the MarketInfo type
                    # This primarily serves static analysis; runtime validation was done above.
                    final_market_info: MarketInfo = std_market # type: ignore [assignment]
                    return final_market_info # Success
                except Exception as cast_err:
                    # Should not happen if MarketInfo definition matches the dict structure, but catch defensively
                    lg.error(f"Internal error casting market dictionary to MarketInfo type ({symbol}): {cast_err}. Returning raw dict cautiously.")
                    # Return the dictionary anyway, but type checkers will complain.
                    return std_market # type: ignore [return-value]

            # else: Market dictionary was not found or retrieved yet, loop will retry or fail below.

        # --- Error Handling for the Loop Iteration ---
        except ccxt.BadSymbol as e:
            # This should have been caught during the refresh attempt, but handle here for robustness
            lg.error(f"Symbol '{symbol}' confirmed invalid on {exchange.id}: {e}")
            return None # Non-retryable
        except ValueError as e: # Catch our critical validation errors (e.g., missing precision)
             lg.error(f"Market info validation failed critically for {symbol}: {e}", exc_info=True)
             return None # Treat validation errors as fatal for this symbol
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError) as e:
            # Handle errors that might occur during the initial market access or the refresh attempt
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network/Exchange error retrieving/refreshing market info ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error retrieving market info: {e}. Cannot continue.{RESET}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during processing
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error retrieving market info ({symbol}): {e}{RESET}", exc_info=True)
            return None # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
            lg.info(f"Waiting {delay}s before retrying market info retrieval for {symbol}...")
            time.sleep(delay)

    # If loop finishes without success (market_dict remained None)
    lg.error(f"{NEON_RED}Failed to get market info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    # Ensure last_exception is set if it wasn't caught previously
    if last_exception is None: last_exception = ccxt.ExchangeError(f"Unknown error or market '{symbol}' not found after retries.")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency from the exchange using `fetch_balance`.
    - Handles different account structures (e.g., Bybit V5 Unified/Contract/Spot) by trying relevant types.
    - Includes retry logic with exponential backoff.
    - Parses the balance robustly to a non-negative Decimal.

    Args:
        exchange: The initialized ccxt exchange instance.
        currency: The currency code (e.g., 'USDT', 'BTC'). Case-sensitive, usually uppercase.
        logger: The logger instance for messages specific to this operation.

    Returns:
        The available balance as a Decimal object if found and valid (non-negative), otherwise None.

    Raises:
        ccxt.AuthenticationError: If authentication fails during the balance check (re-raised for critical handling).
    """
    lg = logger
    lg.debug(f"Fetching available balance for currency: {currency}...")
    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None
            balance_source: str = "N/A" # Describes where the balance value was found
            found: bool = False
            balance_info: Optional[Dict] = None # Store the raw balance response for debugging if needed
            is_bybit = 'bybit' in exchange.id.lower()

            # Define account types to check, relevant primarily for Bybit V5.
            # Order might matter: UNIFIED often holds CONTRACT/SPOT balances. Check specific exchange docs.
            # Bybit V5 types: UNIFIED, CONTRACT, SPOT, FUND, OPTION. We mostly care about trading balances.
            # '' (empty string) usually fetches the default account type (e.g., SPOT or UNIFIED).
            types_to_check = ['UNIFIED', 'CONTRACT', 'SPOT', ''] if is_bybit else [''] # For non-Bybit, just check default

            # Iterate through relevant account types
            for acc_type in types_to_check:
                # Skip specific types if not Bybit
                if not is_bybit and acc_type: continue

                try:
                    params = {}
                    # Add accountType parameter only for Bybit and if a specific type is being checked
                    if acc_type and is_bybit:
                        params['accountType'] = acc_type
                    type_desc = f"Account Type: '{acc_type}'" if acc_type else "Default Account"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")

                    # Fetch balance using ccxt's standard method
                    balance_info = exchange.fetch_balance(params=params)
                    # lg.debug(f"Raw balance response ({type_desc}): {json.dumps(balance_info, indent=2)}") # Verbose logging

                    # --- Try Standard CCXT Structure ---
                    # Expected structure: { 'CUR': {'free': X, 'used': Y, 'total': Z}, ... }
                    # Or sometimes: { 'free': {'CUR': X}, ... }
                    if currency in balance_info and isinstance(balance_info[currency], dict) and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        balance_source = f"{type_desc} (ccxt standard '{currency}.free' field)"
                        found = True; break # Found balance, exit account type loop
                    elif 'free' in balance_info and isinstance(balance_info['free'], dict) and balance_info['free'].get(currency) is not None:
                         balance_str = str(balance_info['free'][currency])
                         balance_source = f"{type_desc} (ccxt standard 'free.{currency}' field)"
                         found = True; break # Found balance

                    # --- Try Bybit V5 Specific Structure (nested within 'info') ---
                    # Structure: info -> result -> list -> [ { accountType, coin: [ { coin, availableToWithdraw/availableBalance } ] } ]
                    elif (is_bybit and 'info' in balance_info and
                          isinstance(balance_info.get('info'), dict) and
                          isinstance(balance_info['info'].get('result'), dict) and
                          isinstance(balance_info['info']['result'].get('list'), list)):

                        lg.debug("Parsing Bybit V5 specific 'info' structure for balance...")
                        for account_details in balance_info['info']['result']['list']:
                            # Check if this entry matches the account type we queried (or if query was default '')
                            fetched_acc_type = account_details.get('accountType')
                            # Type matches if explicit type was requested and matches, OR if default type ('') was requested (accept any type found)
                            type_match = (acc_type and fetched_acc_type == acc_type) or (not acc_type)

                            # Ensure 'coin' list exists and is actually a list
                            if type_match and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if isinstance(coin_data, dict) and coin_data.get('coin') == currency:
                                        # Try different fields for available balance in preferred order
                                        # 'availableToWithdraw' is usually the most liquid balance.
                                        # 'availableBalance' might include unrealized PnL, check Bybit docs.
                                        val = coin_data.get('availableToWithdraw')
                                        src = 'availableToWithdraw'
                                        if val is None:
                                            val = coin_data.get('availableBalance')
                                            src = 'availableBalance'
                                        # 'walletBalance' often includes frozen/used margin, less useful for placing new orders.
                                        # if val is None: val = coin_data.get('walletBalance'); src = 'walletBalance'

                                        if val is not None:
                                            balance_str = str(val)
                                            # Use the actual account type found in the response for logging source
                                            actual_source_type = fetched_acc_type or 'Default/Unknown'
                                            balance_source = f"Bybit V5 info ({actual_source_type}, field: '{src}')"
                                            found = True; break # Found coin data, exit coin loop
                                if found: break # Exit account details loop
                        if found: break # Exit account type loop (outer loop)

                    # If not found in standard or Bybit V5 structure for this account type, log and continue to next type
                    if not found:
                         lg.debug(f"Balance for '{currency}' not found in expected structures for {type_desc}.")

                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # Bybit might throw specific errors for invalid account types; treat these as non-fatal for the *inner* loop
                    # Check for error messages or codes indicating the account type doesn't exist or is invalid.
                    bybit_invalid_type_codes = ['3400000'] # Example Bybit code for param error
                    bybit_invalid_type_msgs = ["account type does not exist", "invalid account type"]
                    is_invalid_type_error = acc_type and is_bybit and (
                        any(code in str(e) for code in bybit_invalid_type_codes) or
                        any(msg in err_str for msg in bybit_invalid_type_msgs)
                    )
                    if is_invalid_type_error:
                        lg.debug(f"Account type '{acc_type}' not found or invalid for balance check on Bybit. Trying next type...")
                        last_exception = e # Keep track of the error, but continue loop
                        continue # Try the next account type
                    else:
                        # Re-raise other exchange errors to be handled by the main handler below
                        raise e
                except Exception as e:
                    # Catch unexpected errors during a specific account type check
                    lg.warning(f"Unexpected error fetching/parsing balance for {type_desc}: {e}. Trying next...", exc_info=True)
                    last_exception = e # Keep track of the error
                    continue # Try the next account type

            # --- Process Result After Checking All Account Types ---
            if found and balance_str is not None:
                # Use safe decimal conversion, allowing zero but not negative balance
                bal_dec = _safe_market_decimal(balance_str, f"balance_str({currency}) from {balance_source}", allow_zero=True, allow_negative=False)

                if bal_dec is not None:
                    lg.info(f"Successfully parsed balance for {currency} from {balance_source}: {bal_dec.normalize()}")
                    return bal_dec # Success
                else:
                    # If conversion fails despite finding a string, it indicates bad data from API - raise error
                    raise ccxt.ExchangeError(f"Failed to convert seemingly valid balance string '{balance_str}' from {balance_source} to non-negative Decimal for {currency}.")
            elif not found:
                # Currency balance was not found in any checked structure after trying all relevant account types
                lg.debug(f"Balance information for currency '{currency}' not found in any checked response structure(s).")
                # Set a specific error message if no other exception was caught during the process
                if last_exception is None:
                    last_exception = ccxt.ExchangeError(f"Balance for '{currency}' not found in response structures after checking all relevant account types.")
                # Let the loop continue to the main retry logic below

        # --- Error Handling for fetch_balance call (outer loop) ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 2) # Exponential backoff + extra base
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance ({currency}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # This is critical and non-retryable. Re-raise to be caught by the caller (e.g., initialize_exchange).
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching balance: {e}. Cannot continue.{RESET}")
            raise e # Re-raise for critical handling
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., temporary issues, maintenance)
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors during the process
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance ({currency}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as potentially fatal for balance check stability
            return None

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
            lg.info(f"Waiting {delay}s before retrying balance fetch for {currency}...")
            time.sleep(delay)

    # If loop finishes without success after all retries
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> Optional[PositionInfo]:
    """
    Fetches and standardizes the currently open position for a specific contract symbol.
    - Returns None if no position exists, if the symbol is not a contract, or if an error occurs.
    - Uses `fetchPositions` or fallback `fetchPosition`.
    - Parses key values to Decimal, infers side if necessary, extracts protection orders from 'info'.
    - Includes retry logic and ignores near-zero "dust" positions.
    - Initializes bot-specific state fields (`be_activated`, `tsl_activated`).

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        market_info: The corresponding MarketInfo dictionary for the symbol (must be valid and contain contract details).
        logger: The logger instance for messages specific to this operation.

    Returns:
        A PositionInfo TypedDict if an active position (size significantly different from zero)
        exists for the symbol, otherwise None.
    """
    lg = logger

    # --- Pre-checks ---
    # Ensure this is a contract market, as positions usually only apply to contracts.
    if not market_info.get('is_contract'):
        lg.debug(f"Position check skipped for {symbol}: Market type is '{market_info.get('contract_type_str', 'Unknown')}', not a contract.")
        return None

    market_id = market_info.get('id') # Exchange-specific ID needed for some API calls
    is_bybit = 'bybit' in exchange.id.lower()

    # Determine category for Bybit V5 based on standardized market info
    category = 'linear' # Default assumption for contracts if not specified
    if market_info.get('is_linear'): category = 'linear'
    elif market_info.get('is_inverse'): category = 'inverse'
    # Add handling for options if needed: elif market_info.get('is_option'): category = 'option'

    # Validate necessary inputs
    if not market_id:
        lg.error(f"Cannot check position for {symbol}: Invalid or missing market ID in market_info.")
        return None
    # Bybit V5 position endpoints usually require 'linear' or 'inverse' category for futures/swaps.
    if is_bybit and category not in ['linear', 'inverse']:
         # Spot positions aren't typically fetched this way. Option positions might use 'option' category.
         lg.error(f"Cannot check position for Bybit symbol {symbol}: Determined category '{category}' is not 'linear' or 'inverse'. Check market info/logic.")
         return None

    lg.debug(f"Checking for open position for {symbol} (Market ID: '{market_id}', Category: '{category if is_bybit else 'N/A'}')...")

    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions ({symbol}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            positions_list: List[Dict] = [] # Initialize as empty list to store raw position data matching the symbol

            # --- Fetch Positions from Exchange ---
            try:
                params = {}
                # Prepare parameters, especially for Bybit V5 which requires category
                if is_bybit:
                     params['category'] = category
                     # Bybit V5 fetchPositions can often filter by symbol (market_id) directly in params
                     params['symbol'] = market_id
                     # For hedge mode, might need settlement coin, check docs: params['settleCoin'] = market_info.get('settleId')

                lg.debug(f"Fetching positions with parameters: {params}")

                # Prefer fetch_positions if available, as it usually returns all positions (or filtered if supported)
                if exchange.has.get('fetchPositions'):
                    # Fetch positions using specified params (which might include symbol filter for Bybit)
                    # Note: Some exchanges might ignore the symbol filter in params and return all positions anyway.
                    all_fetched_positions = exchange.fetch_positions(params=params)

                    # Explicitly filter results for the target symbol/market_id, as API filter might not work or return extras.
                    # Check both standard 'symbol' and exchange-specific 'info.symbol' (which often holds market_id)
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
                     # Pass category in params if needed (e.g., Bybit)
                     pos_data = exchange.fetch_position(symbol, params=params)
                     # fetch_position typically returns a single dict for the open position, or raises error if no position.
                     # Wrap in list for consistent processing below, only if it seems like valid position data.
                     if pos_data and isinstance(pos_data, dict) and pos_data.get('symbol') == symbol:
                          positions_list = [pos_data]
                     else:
                          # If it returns None, empty dict, or wrong symbol, treat as no position found
                          positions_list = []
                     lg.debug(f"fetchPosition returned: {'Position data found' if positions_list else 'No position data found'}")
                else:
                    # If neither method is supported, we cannot get position info reliably.
                    raise ccxt.NotSupported(f"{exchange.id} does not support fetchPositions or fetchPosition via ccxt. Cannot get position info.")

            except ccxt.ExchangeError as e:
                 # Specific handling for errors that explicitly indicate "no position found"
                 # These are not errors in the bot's logic, but expected outcomes.
                 # Bybit V5 retCode: 110025 = position not found / Position is closed
                 common_no_pos_msgs = ["position not found", "no position", "position does not exist", "position is closed", "no active position"]
                 bybit_no_pos_codes = ['110025'] # Bybit V5 specific code

                 err_str = str(e).lower()
                 # Try to extract Bybit retCode from the error message or attributes
                 code_str = ""
                 match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
                 if match: code_str = match.group(2)
                 else: # Fallback to checking attributes on the exception object
                      code_attr = getattr(e, 'code', None) or getattr(e, 'retCode', None)
                      if code_attr is not None: code_str = str(code_attr)

                 # Check if the error matches known "no position" indicators
                 is_bybit_no_pos = is_bybit and code_str and any(code_str == code for code in bybit_no_pos_codes)
                 is_common_no_pos = any(msg in err_str for msg in common_no_pos_msgs)

                 if is_bybit_no_pos or is_common_no_pos:
                     lg.info(f"No open position found for {symbol} (API indicated no position: Code='{code_str}', Msg='{err_str[:80]}...').")
                     return None # This is the expected outcome when no position exists, not an error state for the bot.
                 else:
                     # Re-raise other exchange errors to be handled by the main retry logic
                     raise e

            # --- Process Filtered Positions to Find the Active One ---
            active_raw_position: Optional[Dict] = None

            # Define a small threshold for position size based on market amount precision step.
            # This helps ignore negligible "dust" positions left over from previous trades.
            size_threshold = Decimal('1e-9') # Default tiny threshold if precision unavailable
            amt_step = market_info.get('amount_precision_step_decimal')
            if amt_step and amt_step > 0:
                # Use a fraction of the step size (e.g., half a step) as the threshold.
                # Ensures we only consider positions larger than half the smallest tradeable unit.
                size_threshold = amt_step / Decimal('2')
            lg.debug(f"Using position size threshold > {size_threshold.normalize()} (abs value) for {symbol}.")

            # Iterate through the list of positions matching the symbol (usually 0 or 1 in One-Way mode)
            for pos_data in positions_list:
                # Try to get size: Prefer 'info' dict fields (often more reliable, e.g., Bybit 'size')
                # Fallback to standard ccxt 'contracts' field if primary source not found.
                size_raw = pos_data.get('info', {}).get('size') # Bybit V5 uses 'size' in info
                if size_raw is None: size_raw = pos_data.get('contracts') # Standard ccxt field

                # Safely convert size to Decimal, allowing zero and negative values (for shorts)
                size_decimal = _safe_market_decimal(size_raw, f"{symbol} pos.size/contracts", allow_zero=True, allow_negative=True)

                if size_decimal is None:
                    lg.debug(f"Skipping position data entry with missing or invalid size field ({symbol}). Raw size: {repr(size_raw)}")
                    continue # Skip this entry

                # Check if the *absolute* size exceeds the threshold (effectively non-zero)
                if abs(size_decimal) > size_threshold:
                    if active_raw_position is not None:
                         # This indicates multiple active positions for the same symbol found.
                         # Possible in Hedge mode, or could be an API inconsistency/error in One-Way mode.
                         lg.warning(f"{NEON_YELLOW}Multiple active positions found for {symbol} (One-Way mode assumed)! Using the first one found. "
                                    f"Position 1 Size: {active_raw_position.get('size_decimal', 'N/A')}, Position 2 Size: {size_decimal}. "
                                    f"Check exchange position mode setting.{RESET}")
                         # Stick with the first one found for now in One-Way assumption.
                         # If Hedge mode support is added, logic needs to handle this differently.
                    else:
                         # This is the first (or only) active position found matching criteria
                         active_raw_position = pos_data
                         # Store the parsed Decimal size directly in the dict for standardization and later use
                         active_raw_position['size_decimal'] = size_decimal
                         lg.debug(f"Found active position candidate for {symbol} with size: {size_decimal.normalize()}")
                         # Don't break here immediately if multiple positions might exist (e.g., hedge mode),
                         # but the warning above will fire on subsequent finds.
                         # If strict one-way mode is guaranteed, could break here.
                else:
                     lg.debug(f"Ignoring position data entry with size near zero ({symbol}, Size: {size_decimal.normalize()}, Threshold: {size_threshold.normalize()}).")

            # --- Standardize and Return Active Position Info ---
            if active_raw_position:
                # Create a standardized dictionary based on PositionInfo TypedDict
                std_pos = active_raw_position.copy() # Work on a copy
                info_dict = std_pos.get('info', {}) # Keep reference to raw exchange data for fallbacks

                # Ensure critical size_decimal is present (should be if active_raw_position was set)
                parsed_size = std_pos.get('size_decimal')
                if parsed_size is None or not isinstance(parsed_size, Decimal):
                    lg.error(f"Internal error: Active position found for {symbol} but parsed size_decimal is missing or invalid.")
                    return None # Cannot proceed without valid size

                # Determine Side ('long' or 'short') - crucial and sometimes inconsistent across exchanges/endpoints
                side = std_pos.get('side') # Standard ccxt field

                # Infer side if standard field is missing, ambiguous, or potentially incorrect ('both'/'none'?)
                if side not in ['long', 'short']:
                    inferred_side = None
                    # Try inferring from Bybit V5 'info.side' ('Buy'/'Sell' maps to long/short)
                    side_v5 = str(info_dict.get('side', '')).strip().lower()
                    if side_v5 == 'buy': inferred_side = 'long'
                    elif side_v5 == 'sell': inferred_side = 'short'

                    # If still no side, infer from the sign of the parsed size (most reliable fallback)
                    if inferred_side is None:
                        if parsed_size > size_threshold: inferred_side = 'long'
                        elif parsed_size < -size_threshold: inferred_side = 'short'

                    if inferred_side:
                         if side and side != inferred_side: # Log if standard field existed but contradicted inferred
                              lg.warning(f"Inconsistent side info for {symbol}: Standard field='{side}', Inferred from size/info='{inferred_side}'. Using inferred side.")
                         side = inferred_side
                    else:
                         # Cannot determine side even from size sign (e.g., size is exactly zero or threshold issue)
                         lg.error(f"Could not determine position side for {symbol}. Standard field='{side}', Size='{parsed_size}'. Raw Info: {info_dict}")
                         return None # Cannot proceed without knowing the side

                std_pos['side'] = side # Store the determined/validated side

                # Safely parse other relevant fields to Decimal where applicable using helper
                # Prefer standard ccxt fields, fallback to common 'info' dict fields if standard is None
                # Ensure prices are positive, leverage is positive, PnL/margin can be zero/negative.
                std_pos['entryPrice_decimal'] = _safe_market_decimal(
                    std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice'), # Bybit V5 often uses 'avgPrice' in info
                    f"{symbol} pos.entry", allow_zero=False, allow_negative=False)
                std_pos['leverage_decimal'] = _safe_market_decimal(
                    std_pos.get('leverage') or info_dict.get('leverage'),
                    f"{symbol} pos.leverage", allow_zero=False, allow_negative=False)
                std_pos['liquidationPrice_decimal'] = _safe_market_decimal(
                    std_pos.get('liquidationPrice') or info_dict.get('liqPrice'), # Bybit V5 uses 'liqPrice'
                    f"{symbol} pos.liq", allow_zero=True, allow_negative=False) # Liq price can sometimes be 0/None if far away or cross margin
                std_pos['markPrice_decimal'] = _safe_market_decimal(
                    std_pos.get('markPrice') or info_dict.get('markPrice'),
                    f"{symbol} pos.mark", allow_zero=False, allow_negative=False)
                std_pos['unrealizedPnl_decimal'] = _safe_market_decimal(
                    std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl'), # Bybit V5 spelling difference
                    f"{symbol} pos.pnl", allow_zero=True, allow_negative=True) # PnL can be zero or negative
                std_pos['notional_decimal'] = _safe_market_decimal(
                    std_pos.get('notional') or info_dict.get('positionValue'), # Bybit V5 uses 'positionValue'
                    f"{symbol} pos.notional", allow_zero=True, allow_negative=False) # Notional value >= 0
                std_pos['collateral_decimal'] = _safe_market_decimal(
                    std_pos.get('collateral') or info_dict.get('positionIM') or info_dict.get('collateral'), # Bybit V5 uses positionIM (Initial Margin) as collateral measure sometimes
                    f"{symbol} pos.collateral", allow_zero=True, allow_negative=False) # Collateral >= 0
                std_pos['initialMargin_decimal'] = _safe_market_decimal(
                    std_pos.get('initialMargin') or info_dict.get('positionIM'), # Bybit V5 'positionIM'
                    f"{symbol} pos.initialMargin", allow_zero=True, allow_negative=False)
                std_pos['maintenanceMargin_decimal'] = _safe_market_decimal(
                    std_pos.get('maintenanceMargin') or info_dict.get('positionMM'), # Bybit V5 'positionMM'
                    f"{symbol} pos.maintMargin", allow_zero=True, allow_negative=False)

                # Extract protection orders (SL, TP, TSL) - these are often strings in 'info' or sometimes root level
                # Check both root level and info dict. Value '0' or '0.0' or empty string often means not set.
                def get_protection_value(field_names: List[str]) -> Optional[str]:
                    """Safely gets a protection order value (string) from root or info dict.
                       Returns the string value if it represents a valid, non-zero price/value.
                       Returns None if the value is missing, '0', '0.0', empty, or invalid."""
                    value_str: Optional[str] = None
                    raw_value: Optional[Any] = None
                    # Check multiple potential field names
                    for name in field_names:
                        raw_value = std_pos.get(name) # Check root level first
                        if raw_value is None: raw_value = info_dict.get(name) # Check info dict
                        if raw_value is not None: break # Found a potential value

                    if raw_value is None: return None # Not found
                    value_str = str(raw_value).strip()
                    if not value_str: return None # Empty string means not set

                    # Check if the string represents a valid, non-zero number using safe decimal conversion
                    # Allow zero=False ensures '0' and '0.0' are treated as None (not set)
                    dec_val = _safe_market_decimal(value_str, f"{symbol} prot.{'/'.join(field_names)}", allow_zero=False, allow_negative=False)
                    if dec_val is not None:
                         return value_str # Return the original, valid, non-zero string value
                    else:
                         return None # Treat '0', '0.0', or other invalid strings as no active order set

                # Extract standard SL/TP prices
                std_pos['stopLossPrice'] = get_protection_value(['stopLoss', 'stopLossPrice', 'slPrice']) # Add common variations
                std_pos['takeProfitPrice'] = get_protection_value(['takeProfit', 'takeProfitPrice', 'tpPrice']) # Add common variations

                # Extract Trailing Stop Loss related fields (highly exchange specific)
                # Bybit V5 uses 'trailingStop' (distance/offset string) and 'activePrice' (activation price string) in position info
                std_pos['trailingStopLoss'] = get_protection_value(['trailingStop']) # Check only standard Bybit V5 field name
                std_pos['tslActivationPrice'] = get_protection_value(['activePrice']) # Check only standard Bybit V5 field name

                # Initialize bot state tracking fields based on initial detection
                # These will be updated by bot logic later if the bot takes action (e.g., moves SL to BE).
                std_pos['be_activated'] = False # Bot has not activated Break Even for this position instance yet
                # Check if TSL seems active based on exchange data (both distance and activation price are set and non-zero)
                exchange_tsl_active = bool(std_pos['trailingStopLoss']) and bool(std_pos['tslActivationPrice'])
                std_pos['tsl_activated'] = exchange_tsl_active # Reflect initial state detected from exchange

                # --- Log Found Position Details ---
                # Helper for logging optional Decimal values safely and concisely
                def fmt_log(val: Optional[Decimal]) -> str:
                    # Use normalize() to remove trailing zeros for cleaner display
                    return val.normalize() if val is not None and val.is_finite() else 'N/A'

                ep = fmt_log(std_pos['entryPrice_decimal'])
                sz = std_pos['size_decimal'].normalize()
                sl = std_pos.get('stopLossPrice') or 'N/A'
                tp = std_pos.get('takeProfitPrice') or 'N/A'
                tsl_dist = std_pos.get('trailingStopLoss') or 'N/A'
                tsl_act = std_pos.get('tslActivationPrice') or 'N/A'
                # Determine TSL status string for logging
                tsl_str = "Inactive"
                if exchange_tsl_active:
                     tsl_str = f"ACTIVE (Dist/Offset={tsl_dist} | ActPrice={tsl_act})"
                elif std_pos.get('trailingStopLoss') or std_pos.get('tslActivationPrice'): # Partially set? Indicates potential issue or intermediate state.
                     tsl_str = f"PARTIAL? (Dist/Offset={tsl_dist} | ActPrice={tsl_act})"

                pnl = fmt_log(std_pos['unrealizedPnl_decimal'])
                liq = fmt_log(std_pos['liquidationPrice_decimal'])
                lev = fmt_log(std_pos['leverage_decimal'])
                notional = fmt_log(std_pos['notional_decimal'])
                margin_mode = std_pos.get('marginMode', 'N/A') # Get margin mode if available

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET}\n"
                        f"  Size={sz}, Entry={ep}, Mark={fmt_log(std_pos['markPrice_decimal'])}, Notional={notional}\n"
                        f"  Liq={liq}, Leverage={lev}x, MarginMode={margin_mode}\n"
                        f"  Unrealized PnL: {pnl}\n"
                        f"  Protections (from exchange): SL={sl}, TP={tp}, TSL={tsl_str}")

                # --- Cast to TypedDict and Return ---
                try:
                    # Cast the standardized dictionary to the PositionInfo type for static analysis benefits
                    final_position_info: PositionInfo = std_pos # type: ignore [assignment]
                    return final_position_info # Success
                except Exception as cast_err:
                    # Should not happen if PositionInfo matches the dict structure, but catch defensively
                    lg.error(f"Internal error casting position dictionary to PositionInfo type ({symbol}): {cast_err}. Returning raw dict cautiously.")
                    # Return the dictionary anyway, but type checkers might complain.
                    return std_pos # type: ignore [return-value]

            else:
                # No position found with size > threshold after checking all filtered entries
                lg.info(f"No active position found for {symbol} (checked {len(positions_list)} potential entries from API).")
                return None # No active position exists

        # --- Error Handling for the Loop Iteration ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 2) # Exponential backoff + extra base
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # Fatal error
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching positions: {e}. Cannot continue.{RESET}")
            return None
        except ccxt.NotSupported as e:
             # Fatal error for this function if exchange doesn't support required methods
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
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
            lg.info(f"Waiting {delay}s before retrying position fetch for {symbol}...")
            time.sleep(delay)

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    lg.error(f"  Last error encountered: {last_exception}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a given contract symbol using ccxt's `set_leverage` method.
    - Handles specific requirements for exchanges like Bybit V5 (category, buy/sell leverage as strings).
    - Includes retry logic and checks response codes for success, no change needed, or fatal errors.
    - Validates input leverage and checks for contract market type.

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        leverage: The desired integer leverage level (e.g., 10 for 10x). Must be positive.
        market_info: The MarketInfo dictionary for the symbol (must be valid and contain contract details).
        logger: The logger instance for messages specific to this operation.

    Returns:
        True if leverage was set successfully or confirmed to be already set to the desired value, False otherwise.
    """
    lg = logger

    # --- Pre-checks ---
    # Ensure it's a contract market where leverage setting is applicable
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True # No action needed for non-contracts, considered success in this context

    # Validate the input leverage value
    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Leverage setting failed ({symbol}): Invalid leverage value '{leverage}'. Must be a positive integer.")
        return False

    # Check if the exchange instance has the setLeverage capability advertised by ccxt
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.warning(f"Leverage setting might not be directly supported via ccxt's standard `setLeverage` method for {exchange.id}. "
                   f"Attempting anyway, but it may fail or require manual exchange configuration.")
        # Proceed cautiously, but warn the user. The call might still work via implicit methods.

    market_id = market_info.get('id') # Exchange-specific ID often needed in params
    is_bybit = 'bybit' in exchange.id.lower()

    # Determine category for Bybit V5 based on market info
    category = 'linear' # Default assumption
    if market_info.get('is_linear'): category = 'linear'
    elif market_info.get('is_inverse'): category = 'inverse'
    # Add option category if needed: elif market_info.get('is_option'): category = 'option'

    # Validate necessary info for the call
    if not market_id:
         lg.error(f"Leverage setting failed ({symbol}): Market ID missing in market_info.")
         return False
    # Bybit V5 leverage setting usually requires linear or inverse category for futures/swaps
    if is_bybit and category not in ['linear', 'inverse']:
         lg.error(f"Leverage setting failed for Bybit symbol {symbol}: Invalid or unsupported category '{category}'. Must be 'linear' or 'inverse'.")
         return False

    lg.info(f"Attempting to set leverage for {symbol} (Market ID: {market_id}, Category: {category if is_bybit else 'N/A'}) to {leverage}x...")

    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"set_leverage call attempt {attempts + 1}/{MAX_API_RETRIES + 1} for {symbol} to {leverage}x...")
            params = {}

            # --- Exchange-Specific Parameter Handling (Bybit V5 Example) ---
            if is_bybit:
                 # Bybit V5 requires category and separate buy/sell leverage specified as strings in the params dictionary.
                 # Ensure leverage is passed as a string for both buy and sell sides for One-Way mode.
                 params = {
                     'category': category,
                     'buyLeverage': str(leverage), # Must be string representation
                     'sellLeverage': str(leverage) # Must be string representation
                 }
                 lg.debug(f"Using Bybit V5 specific leverage parameters: {params}")

            # --- Execute set_leverage Call ---
            # Note: ccxt's standard `set_leverage` method takes leverage as float/int, symbol, and optional params dict.
            # Pass the desired leverage (can be int or float), standard symbol, and prepared params.
            response = exchange.set_leverage(leverage=float(leverage), symbol=symbol, params=params)
            lg.debug(f"Raw response from set_leverage ({symbol}): {response}")

            # --- Response Validation (Crucial for confirming success) ---
            # Success is often indicated by the absence of an exception.
            # However, checking response codes/messages provides stronger confirmation, especially for exchanges like Bybit.
            ret_code_str: Optional[str] = None
            ret_msg: str = "N/A"

            # Attempt to extract standard response codes/messages from the result dictionary
            if isinstance(response, dict):
                # Try extracting Bybit V5 style response codes/messages from 'info' dict first, then root level.
                info_dict = response.get('info', {})
                raw_code = info_dict.get('retCode') # Primary location in Bybit V5 response
                if raw_code is None: raw_code = response.get('retCode') # Fallback to root level if not in info
                ret_code_str = str(raw_code) if raw_code is not None else None
                ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown message'))

            # --- Check Specific Exchange Response Codes (Bybit V5 Example) ---
            bybit_success_codes = ['0'] # Bybit V5 Success Code
            # Bybit V5 Code indicating leverage was already set / not modified
            # Message often contains "leverage not modified" or "same leverage"
            bybit_no_change_codes = ['110045']

            if ret_code_str in bybit_success_codes:
                lg.info(f"{NEON_GREEN}Leverage successfully set for {symbol} to {leverage}x (Exchange Code: {ret_code_str}).{RESET}")
                return True # Definite success
            elif ret_code_str in bybit_no_change_codes and ("leverage not modified" in ret_msg.lower() or "same leverage" in ret_msg.lower()):
                lg.info(f"{NEON_YELLOW}Leverage for {symbol} is already {leverage}x (Exchange Code: {ret_code_str} - Not Modified). Success.{RESET}")
                return True # Already set, considered success
            elif response is not None and ret_code_str is None:
                 # If no specific error code was found but the call didn't raise an exception and returned something,
                 # assume success for exchanges that don't provide clear codes in the success response.
                 lg.info(f"{NEON_GREEN}Leverage set/confirmed for {symbol} to {leverage}x (No specific success code checked/found, assumed success based on API response).{RESET}")
                 return True
            elif response is None:
                 # Response was None or empty, which is unexpected if no exception occurred
                 raise ccxt.ExchangeError(f"Received unexpected empty response after setting leverage for {symbol}. Cannot confirm success.")
            else:
                # Received a non-zero, non-"no-change" code, indicating an error reported by the exchange API itself
                raise ccxt.ExchangeError(f"Leverage setting failed. Exchange returned error: {ret_msg} (Code: {ret_code_str})")


        # --- Error Handling for set_leverage call (Catching Exceptions) ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str_lower = str(e).lower()
            # Try to extract error code again for detailed logging/decision making
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            else: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

            lg.error(f"{NEON_RED}Exchange error setting leverage ({symbol} to {leverage}x): {e} (Code: {err_code_str}){RESET}")

            # Check again if the error indicates leverage was already set (some exchanges might throw error instead of code)
            if err_code_str in bybit_no_change_codes and ("leverage not modified" in err_str_lower or "same leverage" in err_str_lower):
                lg.info(f"{NEON_YELLOW}Leverage already set to {leverage}x (confirmed via error response {err_code_str}). Success.{RESET}")
                return True

            # Check for known fatal/non-retryable error codes or messages
            # (Customize based on target exchange - using Bybit V5 examples)
            fatal_codes = [
                '10001', # Parameter error (e.g., leverage value invalid/outside limits for symbol)
                '10004', # Sign check error (API keys invalid)
                '110013',# Risk limit error (exceeds account/symbol risk limit)
                '110028',# Cross margin mode active, cannot modify leverage individually (usually needs account-level setting)
                '110043',# Cannot set leverage when position exists (specific to Isolated Margin mode)
                '110044',# Cannot set leverage when active order exists (specific to Isolated Margin mode)
                '110055',# Cannot set leverage under Isolated mode for a Cross margin position (mode mismatch)
                '3400045',# Leverage value outside min/max limits defined by exchange for this symbol
                '110066', # Cannot set leverage under Portfolio Margin mode
                '110076', # Leverage reduction is restricted when position exists (even in Cross mode sometimes)
                '30086', # Order value exceeds risk limit (Bybit Spot margin related?)
                '3303001', # Invalid leverage (Binance example)
            ]
            fatal_messages = [
                "margin mode", "position exists", "order exists", "risk limit", "parameter error",
                "insufficient available balance", "invalid leverage", "leverage exceed",
                "isolated margin", "portfolio margin", "api key", "authentication failed"
            ]
            is_fatal_code = err_code_str in fatal_codes
            is_fatal_message = any(msg in err_str_lower for msg in fatal_messages)

            # If error is identified as fatal, log hint and return False (non-retryable)
            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE leverage setting error for {symbol}. Aborting leverage setting.{RESET}")
                # Provide more specific advice based on common fatal errors
                if any(code in err_code_str for code in ['110043', '110044', '110076']) or any(s in err_str_lower for s in ["position exists", "order exists"]):
                    lg.error(" >> Advice: Cannot change leverage while a position or active orders exist for this symbol (especially in Isolated Margin mode). Close position/orders first or check margin mode.")
                elif any(s in err_str_lower for s in ["margin mode", "cross margin", "isolated margin", "portfolio margin"]) or any(code in err_code_str for code in ['110028', '110055', '110066']):
                     lg.error(" >> Advice: Leverage change might conflict with the current account margin mode (Cross/Isolated/Portfolio). Check exchange settings or use appropriate API parameters if available.")
                elif any(s in err_str_lower for s in ["parameter error", "invalid leverage", "leverage exceed"]) or any(code in err_code_str for code in ['10001', '3400045', '3303001']):
                     lg.error(f" >> Advice: Leverage value {leverage}x might be invalid or outside the allowed limits for {symbol}. Check exchange rules and symbol specifications.")

                return False # Non-retryable failure

            # If not identified as fatal, allow the loop to proceed to retry logic below

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
            return False # Treat unexpected errors as fatal for safety

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
            lg.info(f"Waiting {delay}s before retrying leverage setting for {symbol}...")
            time.sleep(delay)

    # If loop finishes without success after all retries
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
    entry price, and stop loss price, while adhering to market constraints.

    - Uses Decimal for all financial calculations.
    - Validates inputs (balance, risk, prices).
    - Extracts and validates market constraints (min/max size, step size, min/max cost, contract size) from MarketInfo.
    - Applies correct sizing formula for Linear/Spot vs. Inverse contracts.
    - Adjusts calculated size sequentially based on:
        1. Min/Max Amount limits.
        2. Min/Max Cost limits (estimates cost and may reduce size for Max Cost, fails if Min Cost cannot be met).
        3. Amount Precision step size (rounds DOWN to the nearest valid step).
    - Performs final validation checks after all adjustments.
    - Logs calculation steps and adjustments clearly.

    Args:
        balance: Available trading balance in the quote currency (Decimal, must be positive).
        risk_per_trade: The fraction of the balance to risk (e.g., 0.01 for 1%). Must be > 0.0 and <= 1.0.
        initial_stop_loss_price: The calculated initial stop loss price (Decimal, must be positive and different from entry).
        entry_price: The estimated or actual entry price (Decimal, must be positive).
        market_info: The MarketInfo dictionary for the symbol (must contain valid precision/limits/type).
        exchange: The ccxt exchange instance (currently unused but kept for signature consistency).
        logger: The logger instance for messages specific to this calculation.

    Returns:
        The calculated and adjusted position size as a Decimal (in base currency for spot,
        or number of contracts for futures), rounded DOWN to the correct precision step.
        Returns None if calculation fails due to invalid inputs, constraints violation,
        or mathematical errors (e.g., division by zero).
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
    if not isinstance(balance, Decimal) or not balance.is_finite() or balance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Balance is zero, negative, or invalid ({balance} {quote_currency}).")
        return None
    try:
        # Convert float risk_per_trade to Decimal for calculations
        risk_decimal = Decimal(str(risk_per_trade))
        # Risk must be strictly positive (> 0%) and less than or equal to 1 (100%)
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
             raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive).")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None

    # Ensure prices are valid positive Decimals and SL != Entry
    if not (isinstance(initial_stop_loss_price, Decimal) and initial_stop_loss_price.is_finite() and initial_stop_loss_price > Decimal('0')):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive Stop Loss price ({initial_stop_loss_price}).")
        return None
    if not (isinstance(entry_price, Decimal) and entry_price.is_finite() and entry_price > Decimal('0')):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive Entry price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): Entry price ({entry_price.normalize()}) and Stop Loss price ({initial_stop_loss_price.normalize()}) cannot be the same.")
        return None
    # Implicit check: Ensure SL is on the losing side of entry (e.g., SL < Entry for long, SL > Entry for short)
    # This is handled by `abs(entry_price - initial_stop_loss_price)` below.

    # --- Extract and Validate Market Constraints ---
    try:
        # Get pre-parsed Decimal values from MarketInfo
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal'] # Used for logging/adjustments
        min_amount = market_info['min_amount_decimal'] # Can be None
        max_amount = market_info['max_amount_decimal'] # Can be None
        min_cost = market_info['min_cost_decimal']     # Can be None
        max_cost = market_info['max_cost_decimal']     # Can be None
        contract_size = market_info['contract_size_decimal'] # Should default to Decimal('1') if missing

        # Validate critical constraints needed for calculation and adjustment
        if not (amount_step and amount_step.is_finite() and amount_step > 0):
            raise ValueError("Amount precision step (amount_step) is missing, invalid, or non-positive.")
        if not (price_step and price_step.is_finite() and price_step > 0):
            raise ValueError("Price precision step (price_step) is missing, invalid, or non-positive.")
        if not (contract_size and contract_size.is_finite() and contract_size > 0):
            raise ValueError("Contract size (contract_size) is missing, invalid, or non-positive.")

        # Define effective limits for calculations, treating None as non-restrictive (0 or infinity)
        # Use Decimal('0') for min limits if None, Decimal('inf') for max limits if None.
        min_amount_eff = min_amount if min_amount is not None and min_amount >= 0 else Decimal('0')
        max_amount_eff = max_amount if max_amount is not None and max_amount > 0 else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None and min_cost >= 0 else Decimal('0')
        max_cost_eff = max_cost if max_cost is not None and max_cost > 0 else Decimal('inf')

        # Log the constraints being used
        def fmt_dec_log_size(d: Optional[Decimal]) -> str: return str(d.normalize()) if d is not None and d.is_finite() else 'N/A'
        lg.debug(f"  Market Constraints ({symbol}):")
        lg.debug(f"    Amount Step: {fmt_dec_log_size(amount_step)}, Min Amt: {fmt_dec_log_size(min_amount)}, Max Amt: {fmt_dec_log_size(max_amount)}")
        lg.debug(f"    Price Step : {fmt_dec_log_size(price_step)}")
        lg.debug(f"    Cost Min   : {fmt_dec_log_size(min_cost)}, Cost Max: {fmt_dec_log_size(max_cost)}")
        lg.debug(f"    Contract Size: {fmt_dec_log_size(contract_size)}, Type: {market_info['contract_type_str']}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error accessing or validating required market details from market_info: {e}")
        lg.debug(f"  Problematic MarketInfo structure provided: {market_info}")
        return None

    # --- Core Size Calculation ---
    # Calculate risk amount in quote currency. Quantize early to avoid minor precision differences later.
    # Round down risk amount slightly for safety margin. Use reasonable precision (e.g., 8 decimal places).
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN)
    # Calculate stop loss distance in price points (always positive)
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    # Ensure stop loss distance is valid (should be > 0)
    if stop_loss_distance <= Decimal('0'): # Should be caught by SL != Entry check, but safeguard
        lg.error(f"Sizing failed ({symbol}): Stop loss distance is zero or negative ({stop_loss_distance}).")
        return None

    # Log input values for calculation
    lg.info(f"  Inputs:")
    lg.info(f"    Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"    Risk % : {risk_decimal:.2%}") # Format risk percentage
    lg.info(f"    Risk Amt: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"    Entry Price: {entry_price.normalize()}")
    lg.info(f"    Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"    SL Distance (Price Points): {stop_loss_distance.normalize()}")

    calculated_size = Decimal('NaN') # Initialize as NaN
    try:
        if not is_inverse:
            # --- Linear Contract or Spot ---
            # Formula: Size = Risk Amount (Quote) / (SL Distance (Quote) * Contract Size (Base/Quote or 1 for spot))
            # Risk per unit represents the loss in quote currency per one unit (contract or base currency) if SL is hit.
            risk_per_unit = stop_loss_distance * contract_size
            # Avoid division by zero or near-zero (can happen with very small price/SL distance or contract size)
            if risk_per_unit <= Decimal('1e-18'): # Use a small threshold
                raise ZeroDivisionError(f"Calculated risk per unit ({risk_per_unit}) is near zero. Check prices/SL distance/contract size.")
            calculated_size = risk_amount_quote / risk_per_unit
            lg.debug(f"  Linear/Spot Calculation: Size = RiskAmt / (SL_Dist * ContractSize)")
            lg.debug(f"  = {risk_amount_quote.normalize()} / ({stop_loss_distance.normalize()} * {contract_size.normalize()}) = {calculated_size}")
        else:
            # --- Inverse Contract ---
            # Formula: Size (Contracts) = Risk Amount (Quote) / (Contract Size (Base/Contract) * SL Distance (Quote))
            # Note: Contract Size for inverse is often in Base currency (e.g., 1 BTC) or USD value (e.g., 100 USD).
            # If ContractSize is value (e.g., 100 USD), the formula needs adjustment. Assuming ContractSize is in Base units here.
            # Risk per contract in Quote terms = ContractSize (Base) * SL_Distance (Quote)
            # Check Bybit docs: For Inverse, Position Value (Base) = Qty * ContractSize. Margin (Quote) = PosVal(Base) / EntryPrice / Lev.
            # Loss per contract (Quote) = ContractSize(Base) * | 1/SL - 1/Entry | * SL_Price (approx) -- complicated
            # Let's use the simpler formula assuming ContractSize is in Base units:
            # Risk(Quote) = Size(Contracts) * ContractSize(Base) * SL_Dist(Quote)
            risk_per_contract_quote = contract_size * stop_loss_distance
            if risk_per_contract_quote <= Decimal('1e-18'):
                raise ZeroDivisionError(f"Calculated risk per contract in quote terms ({risk_per_contract_quote}) is near zero for inverse contract. Check prices/SL distance/contract size.")
            calculated_size = risk_amount_quote / risk_per_contract_quote
            # If ContractSize was value (e.g., 100 USD), formula might be: Size = RiskAmt * Entry / (ContractSize * SL_Dist) -- Needs verification.
            lg.debug(f"  Inverse Calculation: Size = RiskAmt / (ContractSize_Base * SL_Dist)")
            lg.debug(f"  = {risk_amount_quote.normalize()} / ({contract_size.normalize()} * {stop_loss_distance.normalize()}) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as e:
        lg.error(f"Sizing failed ({symbol}): Mathematical error during core calculation: {e}.")
        return None

    # Ensure calculated size is a valid positive finite number before proceeding
    if not calculated_size.is_finite() or calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial calculated size is zero, negative, or invalid ({calculated_size}).")
        lg.debug(f"  Check inputs: RiskAmt={risk_amount_quote}, SLDist={stop_loss_distance}, CtrSize={contract_size}, Inverse={is_inverse}")
        return None

    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Adjust Size Based on Constraints (Sequential Application) ---
    adjusted_size = calculated_size
    adjustment_reason = [] # Track reasons for adjustments for logging

    # Helper function to estimate order cost accurately based on contract type
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        """Estimates the order cost (notional value) in quote currency."""
        if not (isinstance(size, Decimal) and size.is_finite() and size > 0): return None
        if not (isinstance(price, Decimal) and price.is_finite() and price > 0): return None
        try:
            cost: Decimal
            # Cost Limit usually applies to Notional Value (Order Value) in Quote Currency
            # Linear/Spot: Notional (Quote) = Size (Contracts/Base) * ContractSize (Base/Quote or 1) * Price (Quote/Base)
            # Inverse: Notional (Quote) = Size (Contracts) * ContractSize (Base/Contract) * Price (Quote/Base)
            # Note: For Inverse, the value in BASE is Size * ContractSize. Value in QUOTE depends on price.
            # Assume cost limits apply to quote notional for both types for simplicity unless exchange docs specify otherwise.
            cost = size * contract_size * price

            # Quantize cost estimate to a reasonable precision (e.g., 8 decimal places) and round UP slightly for safety margin in checks.
            return cost.quantize(Decimal('1e-8'), ROUND_UP)
        except (InvalidOperation, OverflowError, ZeroDivisionError) as cost_err:
            lg.error(f"Cost estimation failed during sizing: {cost_err} (Size: {size}, Price: {price}, CtrSize: {contract_size}, Inverse: {is_inverse})")
            return None

    # --- Constraint 1: Apply Min/Max Amount Limits ---
    size_before_amount_limits = adjusted_size
    # Check Minimum Amount
    if adjusted_size < min_amount_eff:
        adjustment_reason.append(f"Adjusted UP to Min Amount {fmt_dec_log_size(min_amount)}")
        adjusted_size = min_amount_eff
    # Check Maximum Amount
    if adjusted_size > max_amount_eff:
        adjustment_reason.append(f"Adjusted DOWN to Max Amount {fmt_dec_log_size(max_amount)}")
        adjusted_size = max_amount_eff

    # Log if adjusted by amount limits
    if adjusted_size != size_before_amount_limits:
        lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit} ({'; '.join(adjustment_reason)})")
    else:
        lg.debug(f"  Size conforms to Amount Limits.")

    # Ensure adjusted size is still positive and finite after amount limit adjustments
    if not adjusted_size.is_finite() or adjusted_size <= Decimal('0'):
         lg.error(f"Sizing failed ({symbol}): Size became zero, negative or invalid ({adjusted_size}) after applying Amount limits {fmt_dec_log_size(min_amount)}/{fmt_dec_log_size(max_amount)}.")
         return None

    # --- Constraint 2: Apply Min/Max Cost Limits ---
    # Estimate cost based on size *after* amount limits have been applied.
    cost_adjustment_reason = []
    size_before_cost_limits = adjusted_size
    estimated_cost = estimate_cost(adjusted_size, entry_price)

    if estimated_cost is not None:
        lg.debug(f"  Estimated Cost (based on size after amount limits, {symbol}): {estimated_cost.normalize()} {quote_currency}")

        # Check Minimum Cost
        if estimated_cost < min_cost_eff:
            cost_adjustment_reason.append(f"Estimated cost {estimated_cost.normalize()} < Min Cost {fmt_dec_log_size(min_cost)}")
            # If the risk-based size (already potentially adjusted by min amount) results in cost < min_cost,
            # it means the risk % or balance is too low relative to the minimum order value allowed.
            # It's generally unsafe to increase size just to meet min cost, as it violates the risk parameter.
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Calculated size {adjusted_size.normalize()} results in estimated cost {estimated_cost.normalize()} "
                     f"which is below the minimum required cost {fmt_dec_log_size(min_cost)}. "
                     f"Cannot meet Min Cost without exceeding risk %. Consider increasing risk % or ensuring sufficient balance.{RESET}")
            return None # Fail because we cannot meet min cost safely

        # Check Maximum Cost
        if estimated_cost > max_cost_eff:
            cost_adjustment_reason.append(f"Estimated cost {estimated_cost.normalize()} > Max Cost {fmt_dec_log_size(max_cost)}")
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Estimated cost {estimated_cost.normalize()} > Max Cost {fmt_dec_log_size(max_cost)}. Attempting to reduce size to meet Max Cost.{RESET}")
            try:
                # Calculate the theoretical maximum size allowed by max cost
                # Max Size = Max Cost (Quote) / (Contract Size * Entry Price (Quote)) (Assuming cost is notional)
                denominator = entry_price * contract_size
                if denominator <= 0: raise ZeroDivisionError("Entry price * contract size is zero or negative for max cost calculation.")
                max_size_for_max_cost = max_cost_eff / denominator

                if not max_size_for_max_cost.is_finite() or max_size_for_max_cost <= 0:
                     raise ValueError(f"Calculated max size allowed by max cost is non-positive or invalid ({max_size_for_max_cost}).")

                lg.info(f"  Theoretical max size allowed by Max Cost ({symbol}): {max_size_for_max_cost.normalize()} {size_unit}")

                # Adjust size down to the maximum allowed by cost, but ensure it doesn't go below min_amount
                new_adjusted_size = min(adjusted_size, max_size_for_max_cost) # Take the smaller of current or max allowed by cost

                # Ensure the size reduced for max cost doesn't violate the min amount limit
                if new_adjusted_size < min_amount_eff:
                     lg.error(f"{NEON_RED}Sizing failed ({symbol}): Reducing size to meet Max Cost ({fmt_dec_log_size(max_cost)}) would result in size {new_adjusted_size.normalize()} "
                              f"which is below Min Amount ({fmt_dec_log_size(min_amount)}). Cannot satisfy both constraints simultaneously.{RESET}")
                     return None
                else:
                     # If the new size is valid, apply it
                     adjusted_size = new_adjusted_size
                     cost_adjustment_reason.append(f"Adjusted DOWN to {adjusted_size.normalize()} {size_unit} to meet Max Cost")

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating or applying max size allowed by Max Cost: {e}.{RESET}")
                return None

    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        # Cost limits exist according to market info, but we couldn't estimate cost (shouldn't happen if inputs valid)
        lg.warning(f"Could not estimate position cost accurately for {symbol}. Cost limit checks (Min: {fmt_dec_log_size(min_cost)}, Max: {fmt_dec_log_size(max_cost)}) were skipped.")

    # Log if adjusted by cost limits
    if adjusted_size != size_before_cost_limits:
        lg.debug(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit} ({'; '.join(cost_adjustment_reason)})")
    else:
        lg.debug(f"  Size conforms to Cost Limits.")

    # Ensure adjusted size is still positive and finite after cost limit adjustments
    if not adjusted_size.is_finite() or adjusted_size <= Decimal('0'):
         lg.error(f"Sizing failed ({symbol}): Size became zero, negative or invalid ({adjusted_size}) after applying Cost limits.")
         return None

    # --- Constraint 3: Apply Amount Precision (Step Size) ---
    # This is the FINAL adjustment. ROUND DOWN to the nearest valid step size.
    final_size = adjusted_size
    precision_adjustment_reason = ""
    try:
        if amount_step <= 0: raise ValueError("Amount step size is not positive.") # Should be caught earlier

        # Use quantize with ROUND_DOWN for final size adjustment.
        # Correct way for arbitrary step sizes (not just powers of 10):
        # Divide by step, floor the result (get number of full steps), multiply back by step.
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

    # --- Final Validation Checks ---
    # Check 1: Ensure final size is positive and finite after precision rounding
    if not final_size.is_finite() or final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size after precision adjustment is zero, negative, or invalid ({final_size}). "
                 f"Original calculated: {calculated_size.normalize()}. Adjusted before precision: {adjusted_size.normalize()}{RESET}")
        return None

    # Check 2: Re-check Min Amount (rounding down might violate it if min_amount itself wasn't a multiple of step)
    if final_size < min_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is less than Min Amount {fmt_dec_log_size(min_amount)} after applying precision rounding.{RESET}")
        lg.error(f"  This usually means Min Amount ({min_amount_eff}) is not a multiple of Amount Step ({amount_step}), or the calculated size was extremely close to Min Amount.")
        # It's generally unsafe to bump size up to min_amount as it changes risk profile. Fail here.
        return None

    # Check 3: Re-check Max Amount (should be impossible if rounding down, but check for safety)
    if final_size > max_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is greater than Max Amount {fmt_dec_log_size(max_amount)} after precision rounding (unexpected!).{RESET}")
        return None

    # Check 4: Re-check Cost Limits with the final precise size
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
        # Cost limits exist, but we couldn't estimate final cost (shouldn't happen here if inputs valid)
        lg.warning(f"Could not perform final cost check for {symbol} after precision adjustment. Order might fail if cost limits are violated.")

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    if final_cost:
         lg.info(f"    Estimated Final Cost: {final_cost.normalize()} {quote_currency}")
    all_adjustments = '; '.join(filter(None, [', '.join(adjustment_reason), ', '.join(cost_adjustment_reason), precision_adjustment_reason]))
    if all_adjustments:
        lg.info(f"    Adjustments applied: {all_adjustments}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size

def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """
    Cancels a specific open order by its ID using ccxt.cancel_order.
    - Includes retry logic with exponential backoff.
    - Handles common errors like OrderNotFound or InvalidOrder (e.g., already filled/cancelled) gracefully by returning True.
    - Passes necessary parameters for exchanges like Bybit V5 (category, symbol in params).

    Args:
        exchange: The initialized ccxt exchange instance.
        order_id: The ID string of the order to cancel.
        symbol: The market symbol associated with the order (required by ccxt method and potentially in params).
        logger: The logger instance for messages specific to this operation.

    Returns:
        True if the order was successfully cancelled via API, or if it was confirmed already closed/not found
             (meaning it's no longer open and requires no further cancel action).
        False if cancellation failed after retries due to persistent errors (e.g., network, auth, unexpected).
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
            # Attempt to get market info to determine category and market_id for params
            market = exchange.market(symbol)
            market_id = market['id']
            category = 'spot' # Default assumption
            if market.get('linear'): category = 'linear'
            elif market.get('inverse'): category = 'inverse'
            elif market.get('spot'): category = 'spot'
            elif market.get('option'): category = 'option'
            else: category = 'linear' # Fallback guess for contracts

            # Bybit V5 cancelOrder requires category and symbol (market_id) in params
            params['category'] = category
            params['symbol'] = market_id
            lg.debug(f"Using Bybit V5 params for cancelOrder: {params}")
        except Exception as e:
            lg.warning(f"Could not get market details to determine category/market_id for cancelOrder ({symbol}): {e}. Proceeding without specific Bybit params.")
            # If market lookup fails, ccxt might still handle it with just the symbol argument, or fail if params are mandatory.

    # Ensure order_id is a string for the API call
    order_id_str = str(order_id)

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1}/{MAX_API_RETRIES + 1} for ID {order_id_str} ({symbol})...")
            # Use standard symbol in the main call arguments, pass specifics in params dict
            exchange.cancel_order(order_id_str, symbol, params=params)
            # If the call succeeds without exception, assume cancellation request was accepted by the exchange.
            lg.info(f"{NEON_GREEN}Successfully requested cancellation for order {order_id_str} ({symbol}).{RESET}")
            # Note: Success here means the API call succeeded. The order might have already filled just before
            # the cancel request arrived, which is handled by OrderNotFound/InvalidOrder exceptions.
            return True # Exit loop and return success

        except ccxt.OrderNotFound:
            # Order doesn't exist on the exchange. This is common if it was already filled,
            # cancelled manually, or the wrong ID was passed. Treat as success for workflow.
            lg.warning(f"{NEON_YELLOW}Order ID '{order_id_str}' ({symbol}) not found on the exchange. Assuming cancellation is effectively complete or unnecessary.{RESET}")
            return True # Treat as success (order is not open)
        except ccxt.InvalidOrder as e:
             # E.g., order already filled/cancelled/rejected, and API gives a specific error for trying to cancel again.
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Cannot cancel order '{order_id_str}' ({symbol}) due to its current state (e.g., already filled/cancelled): {e}. Assuming cancellation complete.{RESET}")
             return True # Treat as success (order is not open or cancellable)

        # Handle retryable errors
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id_str} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 1) # Exponential backoff for rate limit
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded cancelling order {order_id_str} ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count rate limit wait as standard attempt, just wait and loop
        except ccxt.ExchangeError as e:
            # Other potentially temporary exchange errors during cancellation
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error cancelling order {order_id_str} ({symbol}): {e}. Retrying...{RESET}")
            # Check for potentially non-retryable cancel errors if needed (e.g., permissions)
            # err_str = str(e).lower()
            # if "permission denied" in err_str or "invalid api key" in err_str: return False # Example non-retryable

        # Handle fatal errors
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error cancelling order {order_id_str} ({symbol}): {e}. Cannot continue.{RESET}")
            return False
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id_str} ({symbol}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as failure for safety
            return False

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
            lg.info(f"Waiting {delay}s before retrying order cancellation for {order_id_str}...")
            time.sleep(delay)

    # If loop finishes without success after all retries
    lg.error(f"{NEON_RED}Failed to cancel order {order_id_str} ({symbol}) after {MAX_API_RETRIES + 1} attempts.{RESET}")
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
    params: Optional[Dict[str, Any]] = None # Allow passing extra exchange-specific params externally
) -> Optional[Dict]:
    """
    Places a market order based on the trade signal and calculated size using ccxt.create_order.
    - Determines correct 'side' ('buy' or 'sell') based on the trade signal.
    - Applies amount precision (rounding down) to the position size before placing.
    - Handles specifics for exchanges like Bybit V5 (category, reduceOnly, positionIdx, timeInForce).
    - Includes retry logic with exponential backoff for temporary errors.
    - Provides detailed error handling and hints for common failure reasons (e.g., insufficient funds, invalid order).

    Args:
        exchange: The initialized ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        trade_signal: The action type ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The calculated size for the order (must be a positive, finite Decimal).
        market_info: The MarketInfo dictionary for the symbol (must be valid and contain precision).
        logger: The logger instance for messages specific to this operation.
        reduce_only: If True, sets the reduceOnly flag (for closing/reducing positions).
        params: Optional dictionary of extra parameters for ccxt's create_order (overrides defaults calculated here).

    Returns:
        The order result dictionary from ccxt if the API call for placement was successful
        (indicating the order was accepted by the exchange, not necessarily filled yet),
        otherwise None.
    """
    lg = logger

    # --- Determine Order Side from Signal ---
    # Map the abstract signal to the required 'buy' or 'sell' side for the order.
    side_map = {
        "BUY": "buy",         # Opening a long position or increasing an existing long
        "SELL": "sell",       # Opening a short position or increasing an existing short
        "EXIT_SHORT": "buy",  # Closing/reducing a short position requires buying back
        "EXIT_LONG": "sell"   # Closing/reducing a long position requires selling off
    }
    side = side_map.get(trade_signal.upper())
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' received for {symbol}. Cannot determine order side.")
        return None

    # --- Validate Position Size ---
    if not isinstance(position_size, Decimal) or not position_size.is_finite() or position_size <= Decimal('0'):
        lg.error(f"Invalid position size '{position_size}' provided for order placement ({symbol}). Must be a positive, finite Decimal.")
        return None

    # --- Prepare Order Details ---
    order_type = 'market' # This strategy uses market orders for entry/exit
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base_currency # Unit for logging
    # Description for logging clarity
    action_desc = f"{trade_signal} ({'Reduce-Only' if reduce_only else 'Open/Increase'})"
    market_id = market_info.get('id') # Exchange-specific ID often needed in params
    is_bybit = 'bybit' in exchange.id.lower()

    if not market_id:
         lg.error(f"Cannot place trade for {symbol}: Market ID missing in market_info.")
         return None

    # --- Apply Amount Precision and Convert Size to Float for CCXT ---
    # It's CRUCIAL to apply precision using Decimal *before* converting to float for the API call.
    final_size_decimal = position_size # Start with the input size
    amount_float: float = 0.0 # Initialize float amount
    try:
        amount_step = market_info['amount_precision_step_decimal']
        if amount_step is None or amount_step <= 0:
             raise ValueError("Amount precision step is missing or invalid in market info. Cannot format order size.")

        # Round the size DOWN to the nearest valid step size using Decimal division and floor
        num_steps = (final_size_decimal / amount_step).to_integral_value(rounding=ROUND_DOWN)
        rounded_size_decimal = num_steps * amount_step

        # Check if rounding resulted in zero or negative size
        if rounded_size_decimal <= 0:
            raise ValueError(f"Position size {position_size.normalize()} rounded down to zero or negative based on amount step {amount_step.normalize()}. Cannot place order.")

        # Log if rounding occurred
        if rounded_size_decimal != final_size_decimal:
             lg.warning(f"Adjusting order size {final_size_decimal.normalize()} to {rounded_size_decimal.normalize()} {size_unit} "
                        f"due to precision step {amount_step.normalize()} before placing order.")
             final_size_decimal = rounded_size_decimal # Use the rounded size for logging and conversion

        # Convert the final, rounded Decimal size to float for the ccxt API call
        amount_float = float(final_size_decimal)
        # Final sanity check: ensure float conversion didn't result in effective zero (due to float limitations)
        if abs(amount_float) < 1e-15: # Use a very small number comparison threshold
            raise ValueError(f"Final position size {final_size_decimal.normalize()} converts to near-zero float ({amount_float}). Potential precision issue.")

    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Failed to apply precision or convert size {position_size.normalize()} for order placement ({symbol}): {e}")
        return None

    # --- Base Order Arguments for ccxt.create_order ---
    order_args: Dict[str, Any] = {
        'symbol': symbol,     # Use standard symbol (e.g., 'BTC/USDT') for ccxt call
        'type': order_type,
        'side': side,
        'amount': amount_float, # Pass the precision-adjusted float amount
        # 'price': None, # Not needed for market orders
    }

    # --- Prepare Exchange-Specific Parameters ---
    # These are passed in the 'params' dictionary of create_order
    order_params: Dict[str, Any] = {}
    if is_bybit and is_contract:
        try:
            # Determine category from market_info
            category = 'linear' # Default assumption
            if market_info.get('is_linear'): category = 'linear'
            elif market_info.get('is_inverse'): category = 'inverse'
            elif market_info.get('is_option'): category = 'option' # Add if options are supported
            else: raise ValueError(f"Invalid Bybit contract category derived from market_info: {market_info.get('contract_type_str')}")

            order_params = {
                'category': category,
                # positionIdx: 0 for One-Way mode. Hedge mode uses 1 (Buy side) or 2 (Sell side).
                # Needs config setting if hedge mode is supported by bot. Assume One-Way (0).
                'positionIdx': 0
            }

            # Handle reduceOnly flag and associated TimeInForce for Bybit V5
            if reduce_only:
                order_params['reduceOnly'] = True
                # Bybit V5 often requires/prefers IOC or FOK for reduceOnly market orders
                # to prevent the order resting and potentially increasing position if price moves.
                # IOC (Immediate Or Cancel) is generally safer for market reduceOnly.
                order_params['timeInForce'] = 'IOC'
                lg.debug(f"Setting Bybit V5 specific params: reduceOnly=True, timeInForce='IOC' for {symbol}.")
            else:
                 # For opening/increasing orders, default TIF is usually GTC (Good Til Cancelled),
                 # which is fine for market orders as they execute immediately anyway. No need to set explicitly.
                 pass

        except Exception as e:
            lg.error(f"Failed to set Bybit V5 specific parameters for {symbol} order: {e}. Proceeding with base params, but order might fail if params are mandatory.")
            order_params = {} # Reset params to empty if setup failed

    # Merge any externally provided params (e.g., from signal generation)
    # This allows overriding defaults or adding custom flags.
    if params and isinstance(params, dict):
        lg.debug(f"Merging external parameters into order: {params}")
        order_params.update(params)

    # Add the final params dict to order_args if it's not empty
    if order_params:
        order_args['params'] = order_params

    # --- Log Order Intent Clearly Before Execution ---
    lg.warning(f"{BRIGHT}===> Placing Trade Order ({action_desc}) <==={RESET}")
    lg.warning(f"  Symbol : {symbol} (Market ID: {market_id})")
    lg.warning(f"  Type   : {order_type.upper()}")
    lg.warning(f"  Side   : {side.upper()} (Derived from: {trade_signal})")
    lg.warning(f"  Size   : {final_size_decimal.normalize()} {size_unit} (Float sent: {amount_float})") # Log both Decimal and float used
    if order_args.get('params'):
        lg.warning(f"  Params : {order_args['params']}")

    # --- Execute Order Placement with Retry Logic ---
    attempts = 0
    last_exception: Optional[Exception] = None
    order_result: Optional[Dict] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")

            # +++++ Place the Order via CCXT +++++
            order_result = exchange.create_order(**order_args)
            # ++++++++++++++++++++++++++++++++++++

            # --- Log Success and Basic Order Details ---
            order_id = order_result.get('id', 'N/A') if order_result else 'N/A'
            status = order_result.get('status', 'unknown') if order_result else 'unknown'
            # Market orders might fill immediately ('closed') or remain 'open' briefly until fully filled.
            # Check fill details if available in the response.
            avg_price_raw = order_result.get('average') if order_result else None
            filled_raw = order_result.get('filled') if order_result else None

            # Safely convert fill details to Decimal, allowing zero (e.g., if order is 'open' but not filled yet)
            avg_price_dec = _safe_market_decimal(avg_price_raw, 'order.average', allow_zero=True, allow_negative=False)
            filled_dec = _safe_market_decimal(filled_raw, 'order.filled', allow_zero=True, allow_negative=False)

            log_msg_parts = [
                f"{NEON_GREEN}{action_desc} Order Placement API Call Succeeded!{RESET}",
                f"ID: {order_id}",
                f"Status: {status}"
            ]
            # Add fill details if available and valid
            if avg_price_dec is not None and avg_price_dec > 0:
                log_msg_parts.append(f"Avg Fill Price: ~{avg_price_dec.normalize()}")
            if filled_dec is not None: # Can be 0 if status is 'open'
                log_msg_parts.append(f"Filled Amount: {filled_dec.normalize()} {size_unit}")

            lg.info(" ".join(log_msg_parts))
            # Log the full raw result at debug level for detailed analysis if needed
            lg.debug(f"Full order result ({symbol}): {json.dumps(order_result, indent=2)}")

            # Exit retry loop on successful API call
            break

        # --- Error Handling for create_order ---
        except ccxt.InsufficientFunds as e:
            # This is a common and critical error, usually non-retryable without balance change.
            last_exception = e
            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Insufficient Funds.{RESET}")
            lg.error(f"  Check available balance ({CONFIG.get('quote_currency', 'N/A')}) and margin requirements for size {final_size_decimal.normalize()} {size_unit} "
                     f"with leverage {CONFIG.get('leverage', 'N/A')}x.")
            lg.error(f"  Error details from exchange: {e}")
            return None # Non-retryable failure
        except ccxt.InvalidOrder as e:
            # Order rejected by exchange due to invalid parameters (size, price, flags, etc.). Usually non-retryable.
            last_exception = e
            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Invalid Order Parameters.{RESET}")
            lg.error(f"  Error details from exchange: {e}")
            # Log the arguments sent to help diagnose
            lg.error(f"  Order Arguments Sent to CCXT: {order_args}")
            # Provide hints based on common causes related to market constraints
            err_lower = str(e).lower()
            # Helper for formatting optional Decimals for hints
            def fmt_dec_log(d: Optional[Decimal]) -> str: return str(d.normalize()) if d is not None and d.is_finite() else 'N/A'
            min_a_str = fmt_dec_log(market_info.get('min_amount_decimal'))
            min_c_str = fmt_dec_log(market_info.get('min_cost_decimal'))
            amt_s_str = fmt_dec_log(market_info.get('amount_precision_step_decimal'))
            max_a_str = fmt_dec_log(market_info.get('max_amount_decimal'))
            max_c_str = fmt_dec_log(market_info.get('max_cost_decimal'))

            hint = ""
            # Check error message for keywords indicating specific limit violations
            if any(s in err_lower for s in ["minimum order", "too small", "less than minimum", "min notional", "min value", "order value is too small", "30036"]): # 30036: Binance min notional
                hint = f"Check order size ({final_size_decimal.normalize()}) vs Min Amount ({min_a_str}) and estimated order cost vs Min Cost ({min_c_str})."
            elif any(s in err_lower for s in ["precision", "lot size", "step size", "size precision", "quantity precision", "order qty invalid", "filter failure: lot_size"]):
                hint = f"Check order size ({final_size_decimal.normalize()}) precision against Amount Step ({amt_s_str}). Ensure size is a multiple of the step size."
            elif any(s in err_lower for s in ["exceed", "too large", "greater than maximum", "max value", "max order qty", "position size exceed max limit", "filter failure: market_lot_size"]):
                hint = f"Check order size ({final_size_decimal.normalize()}) vs Max Amount ({max_a_str}) and estimated order cost vs Max Cost ({max_c_str}). Also check position/risk limits."
            elif any(s in err_lower for s in ["reduce only", "reduceonly", "position is closed", "no position found", "available quantity is insufficient"]):
                hint = f"Reduce-only order failed. Ensure an open position exists in the correct direction and the order size ({final_size_decimal.normalize()}) does not exceed the position size."
            elif any(s in err_lower for s in ["position size", "position idx", "position side does not match", "risk limit", "exceed risk limit"]):
                 hint = f"Order conflicts with existing position, leverage limits, risk limits, or position mode (One-Way vs Hedge). Check positionIdx param if using Bybit."
            elif any(s in err_lower for s in ["timeinforce", "time_in_force", "tif"]):
                 hint = f"Invalid TimeInForce parameter used. Check TIF compatibility with order type (Market/Limit) and flags (reduceOnly)."

            if hint: lg.error(f"  >> Hint: {hint}")
            return None # Treat InvalidOrder as non-retryable
        except ccxt.ExchangeError as e:
            # Other exchange errors that might be temporary or specific codes
            last_exception = e
            # Extract error code if possible
            err_code = ""
            match = re.search(r'(retCode|ret_code|code)\s*[:=]\s*"?(-?\d+)"?', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code = match.group(2)
            else: err_code = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

            lg.warning(f"{NEON_YELLOW}Order Placement Exchange Error ({symbol} {action_desc}): {e} (Code: {err_code}). Retry {attempts + 1}...{RESET}")

            # Check for known fatal/non-retryable codes or messages (customize per exchange)
            # Examples (combine codes/messages from various potential errors)
            fatal_codes = ['10001', '10004', '110017', '110040', '30086', '3303001', '3400088', '110043', '-2010', '-1111'] # Add more as needed (e.g., Binance -2010 Insufficient balance, -1111 Invalid price)
            fatal_msgs = ["invalid parameter", "precision", "exceed limit", "risk limit", "invalid symbol", "api key", "authentication failed", "leverage exceed", "account mode", "position mode", "margin is insufficient"]
            is_fatal_code = err_code in fatal_codes
            is_fatal_message = any(msg in str(e).lower() for msg in fatal_msgs)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE order placement error (Code: {err_code}). Check arguments, config, account status, and exchange rules.{RESET}")
                return None # Non-retryable failure

            # If not identified as fatal, allow the loop to proceed to retry logic below

        # Handle retryable network/rate limit errors
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 2) # Exponential backoff + extra base
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't count as standard attempt

        # Handle fatal errors
        except ccxt.AuthenticationError as e:
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
        # Only retry if the order placement hasn't succeeded yet (order_result is still None)
        if attempts <= MAX_API_RETRIES and order_result is None:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1)) # Exponential backoff
            lg.info(f"Waiting {delay}s before retrying order placement for {symbol}...")
            time.sleep(delay)

    # --- Handle Failure After Retries ---
    if order_result is None:
        lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        lg.error(f"  Last error encountered: {last_exception}")
        return None

    # Return the successful order dictionary from the exchange
    return order_result

# --- Placeholder Functions (Require Full Strategy-Specific Implementation) ---

def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # TSL distance/offset (interpretation depends on exchange)
    tsl_activation_price: Optional[Decimal] = None   # Price at which TSL should activate (e.g., Bybit V5)
) -> bool:
    """
    Sets or modifies Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL)
    for an existing position using exchange-specific API calls.

    ** WARNING: This function is a PLACEHOLDER and uses Bybit V5's specific endpoint
       (`private_post_position_set_trading_stop`) structure as an example. **
       Implementation details (endpoint, parameters, formatting) WILL VARY SIGNIFICANTLY
       for other exchanges or even different Bybit API versions/account types.
       Requires thorough testing against the target exchange API documentation.

    Args:
        exchange: Initialized ccxt exchange instance.
        symbol: Standard symbol (e.g., 'BTC/USDT').
        market_info: Market information (contains market_id, category, precision).
        position_info: Current position details (contains side, entryPrice, etc.).
        logger: Logger instance.
        stop_loss_price: Target SL price (Decimal). Provide Decimal('0') or None to remove existing SL.
        take_profit_price: Target TP price (Decimal). Provide Decimal('0') or None to remove existing TP.
        trailing_stop_distance: TSL distance/offset (Decimal, positive). Interpretation is exchange-specific. Provide Decimal('0') or None to remove TSL.
        tsl_activation_price: Price to activate the TSL (Decimal, e.g., for Bybit V5). Usually required if setting TSL distance. Provide Decimal('0') or None to remove.

    Returns:
        True if the protection setting API call was successfully sent and the exchange
             responded with an indication of success (e.g., return code 0 for Bybit V5).
        False otherwise (invalid input, formatting error, API error, non-zero return code).
    """
    lg = logger
    lg.debug(f"Attempting to set/modify position protection for {symbol}...")

    # Log the requested changes
    log_parts = []
    if stop_loss_price is not None: log_parts.append(f"SL={stop_loss_price.normalize() if stop_loss_price > 0 else 'REMOVE'}")
    if take_profit_price is not None: log_parts.append(f"TP={take_profit_price.normalize() if take_profit_price > 0 else 'REMOVE'}")
    if trailing_stop_distance is not None: log_parts.append(f"TSL Dist={trailing_stop_distance.normalize() if trailing_stop_distance > 0 else 'REMOVE'}")
    if tsl_activation_price is not None: log_parts.append(f"TSL Act={tsl_activation_price.normalize() if tsl_activation_price > 0 else 'REMOVE'}")

    if not log_parts:
        lg.debug("No protection parameters provided to set/modify. Skipping API call.")
        return True # Nothing to do, considered successful completion of this task

    lg.info(f"  Target protections: {', '.join(log_parts)}")

    # --- Exchange Specific Logic ---
    # This section needs to be adapted based on the target exchange's API for setting SL/TP/TSL.
    is_bybit = 'bybit' in exchange.id.lower()

    if is_bybit:
        # --- Bybit V5 Example using implicit private POST call ---
        # This assumes the ccxt library has an implicit method mapping for this endpoint,
        # or requires using `exchange.private_post_position_set_trading_stop(params)`.
        # Check ccxt documentation/source for the specific exchange implementation.
        lg.debug("Using Bybit V5 specific logic for setting trading stop...")
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
                # positionIdx: 0 for One-Way. Hedge Mode needs 1 (Buy) or 2 (Sell). Requires config if supporting hedge.
                'positionIdx': 0,
                # tpslMode: 'Full' applies to entire position. 'Partial' applies to a portion (needs size param). Defaulting to 'Full'.
                'tpslMode': 'Full',
                # Trigger price type: Defaulting to MarkPrice. Could be 'LastPrice' or 'IndexPrice'. Make configurable if needed.
                'slTriggerBy': 'MarkPrice',
                'tpTriggerBy': 'MarkPrice',
                # Order type when triggered: Defaulting to Market. Could be 'Limit' (requires orderPrice param).
                'slOrderType': 'Market',
                'tpOrderType': 'Market',
            }

            # Format and add protection parameters if provided and valid
            # Bybit V5 uses string '0' to remove existing SL/TP/TSL.
            param_added = False # Flag to track if any protection parameter was actually set/modified

            # Stop Loss
            if stop_loss_price is not None:
                if stop_loss_price <= 0: # Request to remove SL
                    params['stopLoss'] = '0'
                    param_added = True
                else:
                    # Format the SL price according to market precision
                    sl_str = _format_price(exchange, symbol, stop_loss_price)
                    if sl_str:
                        params['stopLoss'] = sl_str
                        param_added = True
                    else:
                        lg.error(f"Invalid SL price format for {symbol}: {stop_loss_price}. Cannot set protection."); return False

            # Take Profit
            if take_profit_price is not None:
                 if take_profit_price <= 0: # Request to remove TP
                     params['takeProfit'] = '0'
                     param_added = True
                 else:
                     # Format the TP price according to market precision
                     tp_str = _format_price(exchange, symbol, take_profit_price)
                     if tp_str:
                         params['takeProfit'] = tp_str
                         param_added = True
                     else:
                         lg.error(f"Invalid TP price format for {symbol}: {take_profit_price}. Cannot set protection."); return False

            # Trailing Stop Loss Distance
            if trailing_stop_distance is not None:
                 if trailing_stop_distance <= 0: # Request to remove TSL distance
                     params['trailingStop'] = '0'
                     param_added = True
                     # Also ensure activation price is removed if removing TSL distance
                     if 'activePrice' not in params: params['activePrice'] = '0'
                 else:
                     # Bybit 'trailingStop' expects a string distance/offset in *price points*.
                     # Formatting needs care - it's a distance, not a price level.
                     # Format based on the number of decimal places of the price tick size.
                     price_tick = market_info.get('price_precision_step_decimal')
                     if not price_tick or price_tick <= 0:
                         raise ValueError("Price tick precision missing or invalid, cannot format TSL distance.")
                     # Determine number of decimal places needed from the price tick
                     decimal_places = abs(price_tick.as_tuple().exponent)
                     # Format the positive distance value to the required number of decimal places
                     ts_dist_str = f"{trailing_stop_distance:.{decimal_places}f}"

                     # Basic validation: ensure formatted string is still a positive number string
                     if _safe_market_decimal(ts_dist_str, "tsl_dist_str", allow_zero=False, allow_negative=False):
                          params['trailingStop'] = ts_dist_str
                          param_added = True
                     else:
                          lg.error(f"Invalid TSL distance format for {symbol}: {trailing_stop_distance} -> Formatted: '{ts_dist_str}'. Cannot set protection."); return False

            # Trailing Stop Loss Activation Price
            if tsl_activation_price is not None:
                 if tsl_activation_price <= 0: # Request to remove activation price
                     params['activePrice'] = '0'
                     param_added = True # Mark as added even if removing, to ensure API call is made if needed
                 else:
                     # Format the activation price according to market precision
                     act_str = _format_price(exchange, symbol, tsl_activation_price)
                     if act_str:
                         params['activePrice'] = act_str
                         param_added = True
                     else:
                         lg.error(f"Invalid TSL activation price format for {symbol}: {tsl_activation_price}. Cannot set protection."); return False

            # --- Call API only if parameters were actually added/modified ---
            if param_added:
                 lg.info(f"Calling Bybit V5 set_trading_stop API
