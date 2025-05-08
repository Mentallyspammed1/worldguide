import json
import logging
import math
import os
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, TypedDict

# --- Timezone Handling ---
try:
    # Use the standard library's zoneinfo if available (Python 3.9+)
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # Fallback for older Python versions or if tzdata is not installed
    # Basic UTC fallback implementation mimicking ZoneInfo interface
    class ZoneInfo:  # type: ignore [no-redef]
        """Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface."""
        def __init__(self, key: str) -> None:
            self._key = "UTC"  # Store the key, though we always use UTC

        def __call__(self, dt: datetime | None = None) -> datetime | None:
            """Attach UTC timezone info to a datetime object."""
            return dt.replace(tzinfo=UTC) if dt else None

        def fromutc(self, dt: datetime) -> datetime:
            """Convert a UTC datetime to this timezone (which is UTC)."""
            return dt.replace(tzinfo=UTC)

        def utcoffset(self, dt: datetime | None) -> timedelta:
            """Return the UTC offset (always zero for UTC)."""
            return timedelta(0)

        def dst(self, dt: datetime | None) -> timedelta:
            """Return the DST offset (always zero for UTC)."""
            return timedelta(0)

        def tzname(self, dt: datetime | None) -> str:
            """Return the timezone name (always 'UTC')."""
            return "UTC"
    class ZoneInfoNotFoundError(Exception):  # type: ignore [no-redef]
        """Exception raised when a timezone is not found (fallback definition)."""
        pass

# --- Third-Party Library Imports ---
import ccxt  # Crypto Exchange Trading Library
import numpy as np
import pandas as pd
import pandas_ta as ta  # Technical Analysis library
import requests  # For HTTP requests (used by ccxt)

# Colorama for colored console output
from colorama import Fore, Style
from colorama import init as colorama_init

# Dotenv for loading environment variables from a .env file
from dotenv import load_dotenv

# --- Initial Setup ---
# Set Decimal precision for accurate calculations
getcontext().prec = 28
# Initialize Colorama for cross-platform colored output
colorama_init(autoreset=True)
# Load environment variables from .env file
load_dotenv()

# --- Constants ---
BOT_VERSION = "1.4.1"

# --- API Credentials ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit(1)

# --- Configuration & Logging ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago"  # Default if not set in .env
TIMEZONE_STR = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"
except Exception:
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

# --- API & Timing Constants ---
MAX_API_RETRIES: int = 3           # Max number of retries for failed API calls
RETRY_DELAY_SECONDS: int = 5       # Initial delay between retries
POSITION_CONFIRM_DELAY_SECONDS: int = 8  # Delay after placing order to confirm position status
LOOP_DELAY_SECONDS: int = 15       # Base delay between main loop cycles
BYBIT_API_KLINE_LIMIT: int = 1000  # Max klines per Bybit API request (V5)

# --- Data & Strategy Constants ---
VALID_INTERVALS: list[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]  # Supported intervals in config
CCXT_INTERVAL_MAP: dict[str, str] = {  # Map config intervals to CCXT timeframes
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT: int = 750     # Default number of klines to fetch if not in config
MAX_DF_LEN: int = 2000             # Maximum length of DataFrame to keep in memory (avoids memory bloat)

# Default Volumatic Trend (VT) parameters
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0  # Note: This param wasn't used in the original placeholder logic

# Default Order Block (OB) parameters
DEFAULT_OB_SOURCE: str = "Wicks"    # "Wicks" or "Body"
DEFAULT_PH_LEFT: int = 10          # Pivot High lookback/forward periods
DEFAULT_PH_RIGHT: int = 10
DEFAULT_PL_LEFT: int = 10          # Pivot Low lookback/forward periods
DEFAULT_PL_RIGHT: int = 10
DEFAULT_OB_EXTEND: bool = True     # Extend OB boxes until violated
DEFAULT_OB_MAX_BOXES: int = 50     # Max number of active OBs to track

# --- Trading Constants ---
QUOTE_CURRENCY: str = "USDT"       # Default quote currency (updated by config)

# --- UI Constants (Colorama) ---
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN
RESET: str = Style.RESET_ALL
BRIGHT: str = Style.BRIGHT
DIM: str = Style.DIM

# --- Create Log Directory ---
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError:
    sys.exit(1)

# --- Global State ---
_shutdown_requested: bool = False  # Flag for graceful shutdown


# --- Type Definitions ---
class OrderBlock(TypedDict):
    """Represents an identified Order Block."""
    id: str                 # Unique identifier (e.g., "BULL_1678886400000")
    type: str               # "BULL" or "BEAR"
    timestamp: pd.Timestamp  # Timestamp of the candle defining the block
    top: Decimal            # Top price of the block
    bottom: Decimal         # Bottom price of the block
    active: bool            # Is the block currently considered active?
    violated: bool          # Has the block been violated?
    violation_ts: pd.Timestamp | None  # Timestamp of violation
    extended_to_ts: pd.Timestamp | None  # Timestamp the box is currently extended to


class StrategyAnalysisResults(TypedDict):
    """Results from the strategy analysis on a DataFrame."""
    dataframe: pd.DataFrame  # The analyzed DataFrame with indicators
    last_close: Decimal     # Last closing price
    current_trend_up: bool | None  # Current trend direction (True=Up, False=Down, None=Undetermined)
    trend_just_changed: bool  # Did the trend change on the last candle?
    active_bull_boxes: list[OrderBlock]  # List of currently active bullish OBs
    active_bear_boxes: list[OrderBlock]  # List of currently active bearish OBs
    vol_norm_int: int | None  # Normalized volume indicator value (if used)
    atr: Decimal | None    # Current ATR value
    upper_band: Decimal | None  # Upper band value (from VT or other indicator)
    lower_band: Decimal | None  # Lower band value (from VT or other indicator)


class MarketInfo(TypedDict):
    """Standardized market information from ccxt, enhanced with derived fields."""
    # --- Standard CCXT Fields ---
    id: str                 # Exchange-specific market ID (e.g., 'BTCUSDT')
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    base: str               # Base currency (e.g., 'BTC')
    quote: str              # Quote currency (e.g., 'USDT')
    settle: str | None   # Settle currency (usually for futures)
    baseId: str             # Exchange-specific base ID
    quoteId: str            # Exchange-specific quote ID
    settleId: str | None  # Exchange-specific settle ID
    type: str               # Market type ('spot', 'swap', 'future', etc.)
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool            # Is the market currently active/tradeable?
    contract: bool          # Is it a contract (swap, future)?
    linear: bool | None  # Linear contract?
    inverse: bool | None  # Inverse contract?
    quanto: bool | None  # Quanto contract?
    taker: float            # Taker fee rate
    maker: float            # Maker fee rate
    contractSize: Any | None  # Size of one contract
    expiry: int | None
    expiryDatetime: str | None
    strike: float | None
    optionType: str | None
    precision: dict[str, Any]  # Price and amount precision rules
    limits: dict[str, Any]    # Order size and cost limits
    info: dict[str, Any]      # Raw market data from the exchange
    # --- Added/Derived Fields ---
    is_contract: bool         # Convenience flag: True if swap, future, or option
    is_linear: bool           # Convenience flag: True if linear contract
    is_inverse: bool          # Convenience flag: True if inverse contract
    contract_type_str: str    # "Spot", "Linear", "Inverse", "Option", or "Unknown"
    min_amount_decimal: Decimal | None  # Minimum order size as Decimal
    max_amount_decimal: Decimal | None  # Maximum order size as Decimal
    min_cost_decimal: Decimal | None   # Minimum order cost as Decimal
    max_cost_decimal: Decimal | None   # Maximum order cost as Decimal
    amount_precision_step_decimal: Decimal | None  # Smallest amount increment as Decimal
    price_precision_step_decimal: Decimal | None  # Smallest price increment as Decimal
    contract_size_decimal: Decimal  # Contract size as Decimal (defaults to 1 if not applicable/found)


class PositionInfo(TypedDict):
    """Standardized position information from ccxt, enhanced with state tracking."""
    # --- Standard CCXT Fields (subset, may vary slightly by exchange) ---
    id: str | None       # Position ID (exchange-specific)
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    timestamp: int | None  # Position creation/update timestamp (ms)
    datetime: str | None  # ISO 8601 datetime string
    contracts: float | None  # Number of contracts (use size_decimal instead)
    contractSize: Any | None  # Size of one contract for this position
    side: str | None      # 'long' or 'short'
    notional: Any | None  # Position value in quote currency (Decimal preferred)
    leverage: Any | None  # Position leverage (Decimal preferred)
    unrealizedPnl: Any | None  # Unrealized profit/loss (Decimal preferred)
    realizedPnl: Any | None   # Realized profit/loss (Decimal preferred)
    collateral: Any | None    # Margin used for the position (Decimal preferred)
    entryPrice: Any | None    # Average entry price (Decimal preferred)
    markPrice: Any | None     # Current mark price (Decimal preferred)
    liquidationPrice: Any | None  # Estimated liquidation price (Decimal preferred)
    marginMode: str | None    # 'isolated' or 'cross'
    hedged: bool | None       # Is hedging enabled for this position? (Less common now)
    maintenanceMargin: Any | None  # Decimal preferred
    maintenanceMarginPercentage: float | None
    initialMargin: Any | None  # Decimal preferred
    initialMarginPercentage: float | None
    marginRatio: float | None
    lastUpdateTimestamp: int | None
    info: dict[str, Any]         # Raw position data from the exchange
    # --- Added/Derived Fields ---
    size_decimal: Decimal        # Position size as Decimal (positive for long, negative for short)
    stopLossPrice: str | None  # Current stop loss price (as string from exchange, needs parsing)
    takeProfitPrice: str | None  # Current take profit price (as string from exchange, needs parsing)
    trailingStopLoss: str | None  # Current trailing stop distance/price (as string, interpretation depends on exchange)
    tslActivationPrice: str | None  # Trailing stop activation price (as string, if available)
    # --- Bot State Tracking (Managed internally by the bot) ---
    be_activated: bool           # Has the break-even logic been triggered for this position by the bot?
    tsl_activated: bool          # Has the trailing stop loss been activated (either by bot or detected on exchange)?


class SignalResult(TypedDict):
    """Result of the signal generation process."""
    signal: str              # "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
    reason: str              # Explanation for the signal
    initial_sl: Decimal | None  # Calculated initial stop loss price for a new entry
    initial_tp: Decimal | None  # Calculated initial take profit price for a new entry


# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom log formatter to redact sensitive API keys and secrets."""
    _api_key_placeholder = "***BYBIT_API_KEY***"
    _api_secret_placeholder = "***BYBIT_API_SECRET***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive information."""
        msg = super().format(record)
        key = API_KEY
        secret = API_SECRET
        try:
            # Only redact if keys are actually set and are strings
            if key and isinstance(key, str) and key in msg:
                msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg:
                msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception:
            # Avoid crashing the application if redaction fails
            pass
        return msg


class NeonConsoleFormatter(SensitiveFormatter):
    """Formats log messages for the console with timestamps (local time) and colors."""
    _level_colors = {
        logging.DEBUG: NEON_CYAN + DIM,
        logging.INFO: NEON_BLUE,
        logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED,
        logging.CRITICAL: NEON_RED + BRIGHT
    }
    _tz = TIMEZONE  # Use the globally configured timezone

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record with level-specific colors and local timestamp."""
        level_color = self._level_colors.get(record.levelno, NEON_BLUE)  # Default to blue
        log_fmt = (
            f"{NEON_BLUE}%(asctime)s{RESET} - "
            f"{level_color}%(levelname)-8s{RESET} - "
            f"{NEON_PURPLE}[%(name)s]{RESET} - "
            f"%(message)s"
        )
        # Create a formatter for *this record* to use the local timezone
        # Note: Using a lambda for converter makes it dynamic per record
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        # Dynamically set the converter to use the local timezone for this record
        formatter.converter = lambda *args: datetime.now(self._tz).timetuple()  # type: ignore[assignment]

        # Use the parent SensitiveFormatter's format method for redaction,
        # applying it to the message *after* it has been formatted with local time and colors.
        original_message = record.getMessage()  # Get original message before formatting
        record.message = formatter.formatMessage(record)  # Temporarily set the already formatted message
        super().format(record)  # Apply redaction to the formatted message
        record.message = original_message  # Restore original message for other handlers

        # The super().format() might re-apply formatting, let's directly return the redacted version
        # of the already formatted string. Need to adjust how SensitiveFormatter is used.

        # Let's rethink: format the message string first, then apply redaction to it.
        log_entry = formatter.format(record)  # Get the fully formatted string (local time, colors)
        # Now apply redaction to this final string
        redacted_log_entry = log_entry  # Start with the formatted string
        key = API_KEY
        secret = API_SECRET
        try:
            if key and isinstance(key, str) and key in redacted_log_entry:
                redacted_log_entry = redacted_log_entry.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in redacted_log_entry:
                redacted_log_entry = redacted_log_entry.replace(secret, self._api_secret_placeholder)
        except Exception:
            pass

        return redacted_log_entry


def setup_logger(name: str) -> logging.Logger:
    """Sets up and returns a logger instance with file (UTC) and console (local) handlers.

    Args:
        name: The name for the logger (often the symbol or 'main'/'init').

    Returns:
        Configured Logger instance.
    """
    # Sanitize name for filename: replace slashes and colons
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists and has handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers filter later

    # --- File Handler (UTC Time) ---
    try:
        # Rotate log files (10MB each, keep 5 backups)
        fh = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        # Use UTC time for file logs for consistency across servers/timezones
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter.converter = time.gmtime  # Use UTC time tuple
        fh.setFormatter(file_formatter)
        fh.setLevel(logging.DEBUG)  # Log everything to the file
        logger.addHandler(fh)
    except Exception:
        # Log setup errors should be visible on the console
        pass

    # --- Console Handler (Local Time) ---
    try:
        sh = logging.StreamHandler(sys.stdout)
        # The NeonConsoleFormatter handles local time and colors internally
        sh.setFormatter(NeonConsoleFormatter("%(message)s"))  # Basic format string, formatter class does the work

        # Set console log level from environment variable or default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception:
        pass

    logger.propagate = False  # Prevent messages going to the root logger
    return logger


# --- Initial Logger ---
# Used for setup messages before symbol-specific loggers are created
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR} ({TIMEZONE}) for console logs.")
# Add a note about requirements
init_logger.debug("Ensure required packages are installed: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (optional but recommended)")


# --- Configuration Loading & Validation ---
def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from default_config exist in config.
    Adds missing keys with default values and logs changes.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.
        parent_key: Internal tracking for nested key paths (for logging).

    Returns:
        A tuple containing the updated configuration dictionary and a boolean
        indicating if any changes were made.
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing entirely
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' with default value = {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Key exists and is a dictionary, recurse into it
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config
                changed = True
        # Optional: Add type checking here if needed, e.g., if loaded key type mismatch default type
        elif type(updated_config.get(key)) is not type(default_value) and default_value is not None:
             # Basic type mismatch check (excluding None defaults)
             # More robust type checking happens during validation
             # init_logger.debug(f"Config Note: Type mismatch for '{full_key_path}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Validation will handle.")
             pass  # Let validation handle corrections

    return updated_config, changed


def _validate_and_correct_numeric(
    cfg_level: dict[str, Any],
    default_level: dict[str, Any],
    leaf_key: str,
    key_path: str,
    min_val: Decimal | int | float,
    max_val: Decimal | int | float,
    is_strict_min: bool = False,
    is_int: bool = False,
    allow_zero: bool = False
) -> bool:
    """Validates a numeric value within a config dictionary level.
    Corrects type (e.g., str -> int/float, float -> int) and clamps to range if necessary,
    using the default value as a fallback for invalid types or out-of-range values.

    Args:
        cfg_level: The current dictionary level of the loaded config.
        default_level: The corresponding dictionary level in the default config.
        leaf_key: The specific key to validate within the current level.
        key_path: The full dot-notation path to the key (for logging).
        min_val: The minimum allowed value.
        max_val: The maximum allowed value.
        is_strict_min: If True, value must be strictly greater than min_val.
        is_int: If True, the value should be an integer.
        allow_zero: If True, zero is allowed even if outside the min/max range (but still validated).

    Returns:
        True if the value was corrected or replaced with the default, False otherwise.
    """
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key)
    corrected = False
    final_val = original_val  # Assume no change initially

    try:
        # Explicitly disallow boolean types for numeric fields
        if isinstance(original_val, bool):
             raise TypeError("Boolean type found where numeric expected.")

        # Attempt conversion to Decimal for robust comparison and validation
        # Convert to string first to handle potential float inaccuracies and numeric strings
        num_val = Decimal(str(original_val))

        # Check for non-finite values (NaN, Infinity)
        if not num_val.is_finite():
            raise ValueError("Non-finite value (NaN or Infinity) found.")

        min_dec = Decimal(str(min_val))
        max_dec = Decimal(str(max_val))

        # Range Check
        is_zero = num_val.is_zero()
        min_check_passed = num_val > min_dec if is_strict_min else num_val >= min_dec
        range_check_passed = min_check_passed and num_val <= max_dec

        if not range_check_passed and not (allow_zero and is_zero):
            raise ValueError(f"Value {num_val} outside allowed range {'(' if is_strict_min else '['}{min_val}, {max_val}{']'}{' or 0' if allow_zero else ''}.")

        # Type Check and Correction (if needed)
        needs_type_correction = False

        if is_int:
            # Check if the Decimal value has fractional part or if original type wasn't int
            if num_val % 1 != 0 or not isinstance(original_val, int):
                needs_type_correction = True
                final_val = int(num_val.to_integral_value(rounding=ROUND_DOWN))  # Truncate towards zero for safety
                # Re-check range after potential truncation
                final_dec = Decimal(final_val)
                min_check_passed = final_dec > min_dec if is_strict_min else final_dec >= min_dec
                range_check_passed = min_check_passed and final_dec <= max_dec
                if not range_check_passed and not (allow_zero and final_dec.is_zero()):
                    raise ValueError(f"Value truncated to {final_val}, which is outside allowed range.")
            else:
                final_val = int(num_val)  # Already an integer conceptually

        else:  # Expecting float
            # Check if original type wasn't float (or int which is acceptable)
            if not isinstance(original_val, (float, int)):
                 needs_type_correction = True
                 final_val = float(num_val)
            # Check if float representation is significantly different (handles precision issues)
            # Convert validated Decimal to float for storage
            converted_float = float(num_val)
            if isinstance(original_val, float) and abs(original_val - converted_float) > 1e-9:
                 needs_type_correction = True
                 final_val = converted_float
            elif isinstance(original_val, int):  # Allow int if it converts cleanly to float
                 final_val = float(original_val)
                 # Mark as correction only if original wasn't float/int
                 # needs_type_correction = True # Optional: Mark int->float as correction
            else:  # Already a float and close enough
                 final_val = converted_float

        if needs_type_correction:
            init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(final_val)}.{RESET}")
            corrected = True

    except (ValueError, InvalidOperation, TypeError, AssertionError) as e:
        range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
        if allow_zero: range_str += " or 0"
        expected_type = 'integer' if is_int else 'float'
        init_logger.warning(
            f"{NEON_YELLOW}Config Validation: Invalid value for '{key_path}' = {repr(original_val)}. "
            f"Using default: {repr(default_val)}. Error: {e}. "
            f"Expected: {expected_type}, Range: {range_str}{RESET}"
        )
        final_val = default_val  # Use default value on error
        corrected = True

    # Update the config dictionary if a correction was made
    if corrected:
        cfg_level[leaf_key] = final_val

    return corrected


def load_config(filepath: str) -> dict[str, Any]:
    """Loads configuration from a JSON file, creates a default one if missing,
    validates parameters, ensures all necessary keys are present, and saves corrections.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        The loaded and validated configuration dictionary.
    """
    global QUOTE_CURRENCY  # Allow updating the global QUOTE_CURRENCY constant
    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    default_config = {
        "trading_pairs": ["BTC/USDT"],
        "interval": "5",  # Default timeframe (must be in VALID_INTERVALS)
        "retry_delay": RETRY_DELAY_SECONDS,  # Use constant default
        "fetch_limit": DEFAULT_FETCH_LIMIT,  # Use constant default
        "orderbook_limit": 25,  # Limit for order book fetching (if used later)
        "enable_trading": False,  # Safety default: trading disabled
        "use_sandbox": True,    # Safety default: use sandbox environment
        "risk_per_trade": 0.01,  # Risk 1% of capital per trade (as float 0.0 to 1.0)
        "leverage": 20,         # Default leverage (integer, 0 for spot/no leverage)
        "max_concurrent_positions": 1,  # Max simultaneous positions (integer >= 1)
        "quote_currency": "USDT",  # Default quote currency (string)
        "loop_delay_seconds": LOOP_DELAY_SECONDS,  # Use constant default (integer > 0)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,  # Use constant default (integer > 0)

        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,  # int > 0
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,  # int > 0
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,  # int > 0
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER),  # float > 0
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER),  # float > 0
            "ob_source": DEFAULT_OB_SOURCE,  # "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT,  # int > 0
            "ph_right": DEFAULT_PH_RIGHT,  # int > 0
            "pl_left": DEFAULT_PL_LEFT,  # int > 0
            "pl_right": DEFAULT_PL_RIGHT,  # int > 0
            "ob_extend": DEFAULT_OB_EXTEND,  # boolean
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,  # int > 0
            "ob_entry_proximity_factor": 1.005,  # float >= 1.0 (multiplier for OB range)
            "ob_exit_proximity_factor": 1.001,  # float >= 1.0 (multiplier for opposite OB range)
        },

        "protection": {
            "enable_trailing_stop": True,  # boolean
            "trailing_stop_callback_rate": 0.005,  # float > 0 (e.g., 0.005 for 0.5%)
            "trailing_stop_activation_percentage": 0.003,  # float >= 0 (e.g., 0.003 for 0.3% move)
            "enable_break_even": True,  # boolean
            "break_even_trigger_atr_multiple": 1.0,  # float > 0 (multiple of ATR)
            "break_even_offset_ticks": 2,  # int >= 0 (number of price ticks)
            "initial_stop_loss_atr_multiple": 1.8,  # float > 0 (multiple of ATR)
            "initial_take_profit_atr_multiple": 0.7,  # float >= 0 (multiple of ATR, 0 means no TP)
        }
    }

    config_needs_saving: bool = False
    loaded_config: dict[str, Any] = {}

    # --- File Existence Check ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default configuration.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Convert Decimals in default to float for JSON compatibility if any were added
                # json.dump(default_config, f, indent=4, ensure_ascii=False, default=float) # If needed
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully created default config file: {filepath}{RESET}")
            # Update global QUOTE_CURRENCY from the default we just wrote
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config
        except OSError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error creating config file '{filepath}': {e}. Using internal defaults.{RESET}")
            # Still update global QUOTE_CURRENCY from internal default
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config  # Return default in-memory config

    # --- File Loading ---
    try:
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
            raise TypeError("Configuration file content is not a valid JSON object.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Attempting to recreate default file.{RESET}")
        try:
            # Backup corrupted file? Maybe later. For now, overwrite.
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config
        except OSError as e_create:
            init_logger.critical(f"{NEON_RED}FATAL: Error recreating config file: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config
    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected error loading config file '{filepath}': {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config

    # --- Ensure Keys and Validate ---
    try:
        # Ensure all default keys exist, add missing ones
        updated_config, keys_added = _ensure_config_keys(loaded_config, default_config)
        if keys_added:
            config_needs_saving = True  # Mark for saving later

        # --- Validation Logic ---
        init_logger.debug("# Validating configuration parameters...")

        # Helper function to navigate nested dicts for validation
        def get_nested_levels(cfg: dict, path: str) -> tuple[dict | None, dict | None, str | None]:
            """Gets the dict level and leaf key for validation."""
            keys = path.split('.')
            current_cfg_level = cfg
            current_def_level = default_config
            try:
                for key in keys[:-1]:
                    current_cfg_level = current_cfg_level[key]
                    current_def_level = current_def_level[key]
                leaf_key = keys[-1]
                # Ensure both levels are dicts before proceeding
                if not isinstance(current_cfg_level, dict) or not isinstance(current_def_level, dict):
                     init_logger.error(f"Config validation error: Invalid structure at path '{path}'.")
                     return None, None, None
                return current_cfg_level, current_def_level, leaf_key
            except (KeyError, TypeError):
                # This might happen if ensure_keys failed or structure is wrong
                init_logger.error(f"Config validation error: Cannot access path '{path}'. Ensure structure matches default.")
                return None, None, None

        # Define validation function that uses the helper
        def validate_numeric(cfg: dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False) -> None:
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None:
                # Error already logged by helper, cannot validate
                # Attempt to recover by setting the default value at the highest level possible if path was partially valid
                keys = key_path.split('.')
                current_cfg = cfg
                try:
                     for key in keys[:-1]: current_cfg = current_cfg[key]
                     if isinstance(current_cfg, dict):
                          current_cfg[keys[-1]] = default_config  # Set default value
                          config_needs_saving = True
                          init_logger.warning(f"Config validation: Resetting '{key_path}' to default due to access error.")
                except: pass  # Ignore errors during recovery attempt
                return

            # Check if key exists at the leaf level (should exist due to _ensure_config_keys)
            if leaf_key not in cfg_level:
                 init_logger.warning(f"Config validation: Key '{key_path}' unexpectedly missing after ensure_keys. Using default.")
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 return

            # Perform the validation and correction
            corrected = _validate_and_correct_numeric(
                cfg_level, def_level, leaf_key, key_path,
                min_val, max_val, is_strict_min, is_int, allow_zero
            )
            if corrected:
                config_needs_saving = True

        def validate_boolean(cfg: dict, key_path: str) -> None:
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None: return  # Error handled

            if leaf_key not in cfg_level:  # Should not happen
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 return

            if not isinstance(cfg_level[leaf_key], bool):
                init_logger.warning(f"Config Validation: Invalid value for boolean '{key_path}' = {repr(cfg_level[leaf_key])}. Expected true/false. Using default: {repr(def_level[leaf_key])}.")
                cfg_level[leaf_key] = def_level[leaf_key]
                config_needs_saving = True

        def validate_string_choice(cfg: dict, key_path: str, choices: list[str]) -> None:
             nonlocal config_needs_saving
             cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
             if cfg_level is None or def_level is None or leaf_key is None: return  # Error handled

             if leaf_key not in cfg_level:  # Should not happen
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True
                 return

             current_value = cfg_level[leaf_key]
             if not isinstance(current_value, str) or current_value not in choices:
                 init_logger.warning(f"Config Validation: Invalid value for '{key_path}' = {repr(current_value)}. Must be one of {choices}. Using default: {repr(def_level[leaf_key])}.")
                 cfg_level[leaf_key] = def_level[leaf_key]
                 config_needs_saving = True

        # --- Apply Validations ---
        # Top Level
        pairs = updated_config.get("trading_pairs", [])
        if not isinstance(pairs, list) or not all(isinstance(s, str) and s and '/' in s for s in pairs):
            init_logger.warning(f"{NEON_YELLOW}Config Validation: Invalid 'trading_pairs' format. Must be a list of non-empty strings like 'BTC/USDT'. Using default: {default_config['trading_pairs']}.{RESET}")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True

        validate_string_choice(updated_config, "interval", VALID_INTERVALS)
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", 0.0, 1.0, is_strict_min=True)  # Risk must be > 0 and <= 1.0 (100%)
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True)  # Allow 0 for spot/no leverage
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        validate_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True)  # Example range, must be >= 1

        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
            init_logger.warning(f"{NEON_YELLOW}Config Validation: Invalid 'quote_currency'. Must be a non-empty string. Using default '{default_config['quote_currency']}'.{RESET}")
            updated_config["quote_currency"] = default_config["quote_currency"]
            config_needs_saving = True
        # Update the global QUOTE_CURRENCY immediately after validation
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")

        validate_boolean(updated_config, "enable_trading")
        validate_boolean(updated_config, "use_sandbox")

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.vt_step_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1)  # e.g., 1.0 to 1.1 range (must be >= 1.0)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)  # e.g., 1.0 to 1.1 range (must be >= 1.0)
        validate_string_choice(updated_config, "strategy_params.ob_source", ["Wicks", "Body"])
        validate_boolean(updated_config, "strategy_params.ob_extend")

        # Protection Params
        validate_boolean(updated_config, "protection.enable_trailing_stop")
        validate_boolean(updated_config, "protection.enable_break_even")
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0, 0.1, is_strict_min=True)  # Must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0.0, 0.1, allow_zero=True)  # Can be 0
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.0, 10.0, is_strict_min=True)  # Must be > 0
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)  # Can be 0
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.0, 20.0, is_strict_min=True)  # SL > 0
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0.0, 20.0, allow_zero=True)  # TP can be 0 (disabled)

        # --- Save Updated Config if Needed ---
        if config_needs_saving:
             init_logger.info(f"{NEON_YELLOW}Configuration requires updates. Saving changes to '{filepath}'...{RESET}")
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Config file '{filepath}' updated successfully.{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 # Continue with the updated config in memory, but warn the user

        init_logger.info(f"{Fore.CYAN}# Configuration loading and validation complete.{Style.RESET_ALL}")
        return updated_config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: An unexpected error occurred during config processing: {e}. Using internal defaults.{RESET}", exc_info=True)
        # Ensure QUOTE_CURRENCY is set even in this fatal case
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config


# --- Load Configuration ---
CONFIG = load_config(CONFIG_FILE)


# --- Exchange Initialization ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT exchange instance with API keys and settings from config.
    Loads markets and performs an initial balance check.

    Args:
        logger: The logger instance to use for messages.

    Returns:
        A configured ccxt.Exchange instance or None if initialization fails.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Initializing Bybit exchange connection...{Style.RESET_ALL}")
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable built-in rate limiting
            'options': {
                'defaultType': 'linear',      # Prefer linear contracts if available
                'adjustForTimeDifference': True,  # Auto-sync time with server
                # Set reasonable timeouts for common operations (in milliseconds)
                'fetchTickerTimeout': 15000,    # 15 seconds
                'fetchBalanceTimeout': 20000,   # 20 seconds
                'createOrderTimeout': 30000,    # 30 seconds
                'cancelOrderTimeout': 20000,    # 20 seconds
                'fetchPositionsTimeout': 20000,  # 20 seconds
                'fetchOHLCVTimeout': 60000,     # 60 seconds (increased for potentially large history)
                # Bybit specific options (example, may vary)
                # 'recvWindow': 10000, # Optional: Increase receive window if needed
            }
        }
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode based on config
        is_sandbox = CONFIG.get('use_sandbox', True)  # Default to sandbox for safety
        exchange.set_sandbox_mode(is_sandbox)

        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}<<< SANDBOX MODE ACTIVE >>> Exchange: {exchange.id} {RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< LIVE TRADING ACTIVE >>> Exchange: {exchange.id} !!!{RESET}")

        # Load market data (crucial for symbol info, precision, limits)
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}...")
                # Force reload on retries to potentially fix temporary issues
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market data loaded successfully ({len(exchange.markets)} symbols found).{RESET}")
                    markets_loaded = True
                    break
                else:
                    # This case might indicate an issue even if no exception was raised
                    last_market_error = ValueError("Market data structure is empty after loading.")
                    lg.warning(f"Market data appears empty (Attempt {attempt + 1}). Retrying...")

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"Network error loading markets (Attempt {attempt + 1}): {e}. Retrying...")
            except ccxt.RateLimitExceeded as e:
                 last_market_error = e
                 wait = RETRY_DELAY_SECONDS * 3
                 lg.warning(f"{NEON_YELLOW}Rate limit exceeded loading markets: {e}. Waiting {wait}s...{RESET}")
                 time.sleep(wait)
                 continue  # Don't count as standard attempt, just wait
            except ccxt.AuthenticationError as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. "
                            f"Check API Key/Secret and permissions. Exiting.{RESET}")
                return None  # Non-retryable
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected critical error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None  # Non-retryable

            # Wait before retrying if not loaded and more attempts remain
            if not markets_loaded and attempt < MAX_API_RETRIES:
                delay = RETRY_DELAY_SECONDS * (attempt + 1)  # Basic exponential backoff
                lg.warning(f"Retrying market load in {delay}s...")
                time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load market data after {MAX_API_RETRIES + 1} attempts. "
                        f"Last error: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"Exchange initialized: {exchange.id} | Sandbox: {is_sandbox}")

        # Perform an initial balance check (optional but recommended)
        balance_currency = CONFIG.get("quote_currency", QUOTE_CURRENCY)  # Use configured quote currency
        lg.info(f"Performing initial balance check for {balance_currency}...")
        initial_balance: Decimal | None = None
        try:
            initial_balance = fetch_balance(exchange, balance_currency, lg)
        except ccxt.AuthenticationError as auth_err:
            # Catch auth error specifically here as fetch_balance might re-raise it
            lg.critical(f"{NEON_RED}Authentication error during initial balance check: {auth_err}. Exiting.{RESET}")
            return None
        except Exception as balance_err:
            # Log other balance errors as warnings, especially if trading is disabled
            lg.warning(f"{NEON_YELLOW}Initial balance check failed: {balance_err}.{RESET}", exc_info=False)  # exc_info=False to avoid noisy tracebacks for common issues

        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial balance check successful: {initial_balance.normalize()} {balance_currency}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange initialization complete.{Style.RESET_ALL}")
            return exchange
        else:
            # Balance check failed, decide whether to proceed
            lg.error(f"{NEON_RED}Initial balance check FAILED for {balance_currency}.{RESET}")
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading is enabled, but the initial balance check failed. "
                            f"Cannot proceed safely. Exiting.{RESET}")
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without balance confirmation.{RESET}")
                lg.info(f"{Fore.CYAN}# Exchange initialization complete (balance check failed, trading disabled).{Style.RESET_ALL}")
                return exchange  # Allow proceeding if trading is off

    except ccxt.AuthenticationError as e:
         lg.critical(f"{NEON_RED}Authentication error during exchange setup: {e}. Exiting.{RESET}")
         return None
    except Exception as e:
        lg.critical(f"{NEON_RED}A critical error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Helper Functions ---


def _safe_market_decimal(value: Any | None, field_name: str,
                         allow_zero: bool = True, allow_negative: bool = False) -> Decimal | None:
    """Safely converts a value (often from market or position data) to a Decimal.
    Handles None, empty strings, non-finite numbers, and applies zero/negative checks.

    Args:
        value: The value to convert (can be string, number, None).
        field_name: Name of the field being converted (for logging context).
        allow_zero: Allow zero as a valid value.
        allow_negative: Allow negative values.

    Returns:
        The Decimal value, or None if conversion fails or value is invalid according to checks.
    """
    if value is None:
        return None
    try:
        # Convert to string first to handle potential floats accurately and empty strings
        s_val = str(value).strip()
        if not s_val:  # Handle empty strings explicitly
            # init_logger.debug(f"Empty string rejected for '{field_name}'")
            return None
        d_val = Decimal(s_val)

        # Check for NaN or Infinity
        if not d_val.is_finite():
             # init_logger.debug(f"Non-finite value rejected for '{field_name}': {value}")
             return None

        # Validate based on flags
        if not allow_zero and d_val.is_zero():
            # init_logger.debug(f"Zero value rejected for '{field_name}': {value}")
            return None
        if not allow_negative and d_val < Decimal('0'):
            # init_logger.debug(f"Negative value rejected for '{field_name}': {value}")
            return None

        return d_val
    except (InvalidOperation, TypeError, ValueError):
        # init_logger.debug(f"Failed to convert '{field_name}' to Decimal: {repr(value)}")
        return None


def _format_price(exchange: ccxt.Exchange, symbol: str, price: Decimal | float | str) -> str | None:
    """Formats a price according to the market's precision rules using ccxt.
    Ensures the price is positive before formatting and the result is still positive.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        price: The price to format.

    Returns:
        The formatted price as a string, or None if formatting fails or price is invalid.
    """
    try:
        # Use safe conversion first
        price_decimal = _safe_market_decimal(price, f"format_price_input({symbol})", allow_zero=False, allow_negative=False)

        if price_decimal is None:
            init_logger.warning(f"Attempted to format invalid or non-positive price '{price}' for {symbol}. Returning None.")
            return None

        # Use ccxt's built-in method for price formatting
        # It typically requires a float argument
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))

        # Double-check: Ensure formatted string is still a positive value
        # (Handles cases where precision might round down to zero or near-zero)
        formatted_decimal = _safe_market_decimal(formatted_str, f"format_price_output({symbol})", allow_zero=False, allow_negative=False)
        if formatted_decimal is None:
             init_logger.warning(f"Price '{price}' for {symbol} formatted to non-positive or invalid value '{formatted_str}'. Returning None.")
             return None

        return formatted_str
    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
        init_logger.error(f"Error accessing market precision for {symbol}: {e}. Cannot format price.")
        return None
    except (InvalidOperation, ValueError, TypeError) as e:
        # Should be caught by _safe_market_decimal mostly, but keep as fallback
        init_logger.warning(f"Error converting price '{price}' for formatting ({symbol}): {e}")
        return None
    except Exception as e:
        # Catch unexpected errors during formatting
        init_logger.warning(f"Unexpected error formatting price '{price}' for {symbol}: {e}")
        return None


def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the current market price for a symbol using ccxt's fetch_ticker.
    Attempts to use 'last', then mid-price ('bid'/'ask'), then 'ask', then 'bid'.
    Includes retry logic.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        logger: The logger instance for messages.

    Returns:
        The current price as a Decimal, or None if fetching fails.
    """
    lg = logger
    attempts = 0
    last_exception: Exception | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for price ({symbol}, Attempt {attempts + 1})...")
            ticker = exchange.fetch_ticker(symbol)
            price: Decimal | None = None
            source = "N/A"  # Source of the price (last, mid, ask, bid)

            # Helper to safely get Decimal from ticker data
            def safe_decimal_from_ticker(val: Any | None, name: str) -> Decimal | None:
                # Prices must be positive
                return _safe_market_decimal(val, f"ticker.{name} ({symbol})", allow_zero=False, allow_negative=False)

            # Try sources in order of preference
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')
            if price:
                source = "'last' price"
            else:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask:
                    # Ensure bid < ask before calculating mid-price
                    if bid < ask:
                        price = (bid + ask) / Decimal('2')  # Mid-price
                        source = f"mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()})"
                    else:
                        # If bid >= ask, something is wrong, prefer ask as safer estimate
                        price = ask
                        source = f"'ask' price (used due to crossed/equal book: Bid={bid}, Ask={ask})"
                elif ask:
                    price = ask  # Fallback to ask
                    source = f"'ask' price ({ask.normalize()})"
                elif bid:
                    price = bid  # Fallback to bid
                    source = f"'bid' price ({bid.normalize()})"

            if price:
                normalized_price = price.normalize()
                lg.debug(f"Current price ({symbol}) obtained from {source}: {normalized_price}")
                return normalized_price
            else:
                # Ticker fetched, but no usable price found
                last_exception = ValueError(f"No valid price source (last/bid/ask) found in ticker response for {symbol}.")
                lg.warning(f"Could not find a valid price in ticker data ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}. Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Use a longer delay for rate limit errors
            wait = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price ({symbol}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            # Rate limit doesn't count as a standard attempt here, just wait and loop again
            continue
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching price: {e}. Cannot continue.{RESET}")
            return None  # Non-retryable
        except ccxt.BadSymbol as e:
             last_exception = e
             lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching price on {exchange.id}.{RESET}")
             return None  # Non-retryable
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., maintenance, temporary issues)
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching price ({symbol}): {e}. Retrying...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price ({symbol}): {e}{RESET}", exc_info=True)
            # Consider if this should be retryable or fatal
            return None  # Treat unexpected errors as potentially fatal for safety

        # Increment attempt count and wait before retrying (if applicable)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches historical kline/OHLCV data for a symbol using ccxt.
    Handles pagination/chunking required by exchanges like Bybit (using 'until').
    Includes retry logic, data validation, lag check, deduplication, and length trimming.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        timeframe: The timeframe string (e.g., '5m', '1h', '1d').
        limit: The total number of candles desired.
        logger: The logger instance for messages.

    Returns:
        A pandas DataFrame containing the OHLCV data, indexed by timestamp (UTC),
        or an empty DataFrame if fetching fails or data is invalid.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Fetching klines for {symbol} | Timeframe: {timeframe} | Target Limit: {limit}...{Style.RESET_ALL}")

    # --- Pre-checks ---
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetching OHLCV data.")
        return pd.DataFrame()

    # Estimate minimum candles needed based on strategy params (best effort)
    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        # Find the max lookback period needed by indicators
        min_required = max(
            sp.get('vt_length', 0) * 2,  # Example: Need more for initial EMA calculations
            sp.get('vt_atr_period', 0),
            sp.get('vt_vol_ema_length', 0),
            sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1,  # Pivots need left+right+current
            sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1
        ) + 50  # Add a generous buffer for calculations
        lg.debug(f"Estimated minimum candles required by strategy: ~{min_required}")
    except Exception as e:
        lg.warning(f"Could not estimate minimum candle requirement: {e}")

    if limit < min_required:
        lg.warning(f"{NEON_YELLOW}Requested kline limit ({limit}) is less than the estimated strategy requirement ({min_required}). "
                   f"Indicator accuracy may be affected, especially on initial runs.{RESET}")

    # Determine category and market ID for Bybit V5 API
    category = 'spot'  # Default assumption
    market_id = symbol  # Default to symbol if market info fails
    is_bybit = 'bybit' in exchange.id.lower()
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        if market.get('linear'): category = 'linear'
        elif market.get('inverse'): category = 'inverse'
        # else category remains 'spot'
        lg.debug(f"Using API parameters: category='{category}', market ID='{market_id}' (for Bybit).")
    except (ccxt.BadSymbol, KeyError, TypeError) as e:
        lg.warning(f"Could not reliably determine market category/ID for {symbol}: {e}. "
                   f"Proceeding with defaults (category='{category}', market_id='{market_id}'). May fail if incorrect for Bybit.")

    # --- Fetching Loop ---
    all_ohlcv_data: list[list] = []
    remaining_limit = limit
    end_timestamp_ms: int | None = None  # For pagination: fetch candles *before* this timestamp
    # Calculate max chunks generously to avoid infinite loops if API behaves unexpectedly
    # Use the exchange's specific limit if available, otherwise the Bybit default
    api_limit_per_req = getattr(exchange, 'limits', {}).get('fetchOHLCV', {}).get('limit', BYBIT_API_KLINE_LIMIT)
    max_chunks = math.ceil(limit / api_limit_per_req) + 3  # Add a buffer
    chunk_num = 0
    total_fetched = 0

    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        fetch_size = min(remaining_limit, api_limit_per_req)
        lg.debug(f"Fetching kline chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. "
                 f"Ending before TS: {datetime.fromtimestamp(end_timestamp_ms / 1000, tz=UTC) if end_timestamp_ms else 'Latest'}")

        attempts = 0
        last_exception: Exception | None = None
        chunk_data: list[list] | None = None

        while attempts <= MAX_API_RETRIES:
            try:
                # --- Prepare API Call ---
                params = {'category': category} if is_bybit else {}
                fetch_args: dict[str, Any] = {
                    'symbol': symbol,       # Use standard symbol for ccxt call
                    'timeframe': timeframe,
                    'limit': fetch_size,
                    'params': params        # Pass category for Bybit if applicable
                }
                # Add 'until' parameter for pagination (fetches candles ending *before* this timestamp)
                # CCXT handles 'until' internally for many exchanges, but double check if needed
                # Bybit V5 uses 'end' in params, but ccxt might map 'until' to it. Let's rely on ccxt 'until'.
                if end_timestamp_ms:
                    fetch_args['until'] = end_timestamp_ms

                # --- Execute API Call ---
                lg.debug(f"Calling fetch_ohlcv with args: {fetch_args}")
                fetched_chunk = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(fetched_chunk) if fetched_chunk else 0
                lg.debug(f"API returned {fetched_count_chunk} candles for chunk {chunk_num}.")

                # --- Basic Validation & Lag Check (for the first chunk primarily) ---
                if fetched_chunk:
                    # Check if data looks valid (e.g., expected number of columns)
                    if not all(len(candle) >= 6 for candle in fetched_chunk):  # Timestamp, O, H, L, C, V
                        raise ValueError(f"Invalid candle format received in chunk {chunk_num} for {symbol}. Candles have < 6 values.")

                    chunk_data = fetched_chunk  # Assign valid data

                    # Data Lag Check (only on the first chunk from the "latest" end)
                    if chunk_num == 1 and end_timestamp_ms is None:
                        try:
                            last_candle_ts_ms = chunk_data[-1][0]
                            last_ts = pd.to_datetime(last_candle_ts_ms, unit='ms', utc=True)
                            interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                # Allow up to 2.5 intervals of lag before warning/retrying
                                max_lag_seconds = interval_seconds * 2.5
                                current_utc_time = pd.Timestamp.utcnow()
                                actual_lag_seconds = (current_utc_time - last_ts).total_seconds()

                                if actual_lag_seconds > max_lag_seconds:
                                    lag_error_msg = (f"Potential data lag detected! Last candle time ({last_ts}) is "
                                                     f"{actual_lag_seconds:.1f}s behind current time ({current_utc_time}). "
                                                     f"Max allowed lag for {timeframe} is ~{max_lag_seconds:.1f}s.")
                                    last_exception = ValueError(lag_error_msg)
                                    lg.warning(f"{NEON_YELLOW}Lag Check ({symbol}): {lag_error_msg} Retrying fetch...{RESET}")
                                    chunk_data = None  # Discard potentially stale data and force retry
                                    # No break here, let the retry logic handle it below
                                else:
                                    lg.debug(f"Lag check passed ({symbol}): Last candle {actual_lag_seconds:.1f}s old (within limit).")
                                    break  # Valid chunk received, exit retry loop
                            else:
                                lg.warning(f"Could not parse timeframe '{timeframe}' for lag check.")
                                break  # Proceed without lag check if timeframe parsing fails
                        except IndexError:
                             lg.warning("Could not perform lag check: No data in first chunk?")
                             break  # Should not happen if chunk_data is non-empty, but handle defensively
                        except Exception as ts_err:
                            lg.warning(f"Error during lag check ({symbol}): {ts_err}. Proceeding cautiously.")
                            break  # Proceed if lag check itself fails
                    else:  # Not the first chunk or not fetching latest, no lag check needed
                        break  # Valid chunk received, exit retry loop

                else:  # API returned empty list
                    lg.debug(f"API returned no data for chunk {chunk_num}. Assuming end of history or temporary issue.")
                    # If it's the *first* chunk, we should retry. If later chunks, maybe end of history.
                    if chunk_num > 1:
                        lg.info(f"No more historical data found for {symbol} after chunk {chunk_num - 1}.")
                        remaining_limit = 0  # Assume end of history if not the first chunk
                    # No 'break' here, let retry logic handle potential temporary issue for first chunk
                    # Unless we assume end of history, then break outer loop
                    if remaining_limit == 0:
                        break

            # --- Error Handling for fetch_ohlcv call ---
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_exception = e
                lg.warning(f"{NEON_YELLOW}Network error fetching klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait = RETRY_DELAY_SECONDS * 3  # Longer wait for rate limits
                lg.warning(f"{NEON_YELLOW}Rate limit fetching klines chunk {chunk_num} ({symbol}): {e}. Waiting {wait}s...{RESET}")
                time.sleep(wait)
                continue  # Don't increment standard attempts, just wait
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Authentication error fetching klines: {e}. Cannot continue.{RESET}")
                return pd.DataFrame()  # Fatal
            except ccxt.BadSymbol as e:
                 last_exception = e
                 lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching klines on {exchange.id}.{RESET}")
                 return pd.DataFrame()  # Fatal
            except ccxt.ExchangeError as e:
                last_exception = e
                lg.error(f"{NEON_RED}Exchange error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}")
                # Check for specific non-retryable errors (e.g., invalid timeframe)
                err_str = str(e).lower()
                non_retryable_msgs = ["invalid timeframe", "interval not supported", "symbol invalid", "instrument not found", "invalid category"]
                if any(msg in err_str for msg in non_retryable_msgs):
                    lg.critical(f"{NEON_RED}Non-retryable exchange error encountered: {e}. Stopping kline fetch for {symbol}.{RESET}")
                    return pd.DataFrame()  # Fatal for this symbol
                # Otherwise, treat as potentially retryable
            except ValueError as e:  # Catch our validation errors (e.g., candle format, lag)
                 last_exception = e
                 lg.error(f"{NEON_RED}Data validation error fetching klines chunk {chunk_num} ({symbol}): {e}. Retrying...{RESET}")
            except Exception as e:
                last_exception = e
                lg.error(f"{NEON_RED}Unexpected error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True)
                # Treat unexpected errors cautiously - potentially stop fetching for this symbol
                return pd.DataFrame()

            # --- Retry Logic ---
            attempts += 1
            # Only sleep if we need to retry (chunk_data is None and more attempts left)
            if chunk_data is None and attempts <= MAX_API_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempts)

        # --- Process Successful Chunk or Handle Failure ---
        if chunk_data:
            # Prepend the new chunk to maintain chronological order (oldest fetched first)
            all_ohlcv_data = chunk_data + all_ohlcv_data
            chunk_len = len(chunk_data)
            remaining_limit -= chunk_len
            total_fetched += chunk_len

            # Set timestamp for the next older chunk request
            # Use the timestamp of the *first* candle in the *current* chunk minus 1 millisecond
            # Ensure timestamp is valid before using
            try:
                next_until_ts = chunk_data[0][0]
                if not isinstance(next_until_ts, (int, float)) or next_until_ts <= 0:
                    raise ValueError(f"Invalid timestamp found in first candle of chunk: {next_until_ts}")
                end_timestamp_ms = int(next_until_ts) - 1
            except (IndexError, TypeError, ValueError) as ts_err:
                lg.error(f"Error determining next 'until' timestamp from chunk data ({symbol}): {ts_err}. Stopping pagination.")
                remaining_limit = 0  # Stop fetching if we can't paginate

            # Check if the exchange returned fewer candles than requested (might be end of history)
            if chunk_len < fetch_size:
                lg.debug(f"Received fewer candles ({chunk_len}) than requested ({fetch_size}). Assuming end of historical data.")
                remaining_limit = 0  # Stop fetching more chunks

        else:  # Failed to fetch chunk after retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after {MAX_API_RETRIES + 1} attempts. "
                     f"Last error: {last_exception}{RESET}")
            if not all_ohlcv_data:
                # Failed on the very first chunk, cannot proceed
                lg.error(f"Failed to fetch the initial chunk for {symbol}. Cannot construct DataFrame.")
                return pd.DataFrame()
            else:
                # Failed on a subsequent chunk, proceed with what we have
                lg.warning(f"Proceeding with {total_fetched} candles fetched before the error occurred.")
                break  # Exit the fetching loop

        # Small delay between chunk requests to be polite to the API
        if remaining_limit > 0:
            time.sleep(0.5)  # 500ms delay

    # --- Post-Fetching Checks ---
    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines for {symbol} because maximum chunk limit ({max_chunks}) was reached. "
                   f"Fetched {total_fetched} candles.")

    if not all_ohlcv_data:
        lg.error(f"No kline data could be fetched for {symbol} {timeframe}.")
        return pd.DataFrame()

    lg.info(f"Total raw klines fetched: {len(all_ohlcv_data)}")

    # --- Data Deduplication and Sorting ---
    # Use a dictionary to store the latest candle for each timestamp
    unique_candles_dict = {}
    for candle in all_ohlcv_data:
        try:
            timestamp = int(candle[0])
            if timestamp <= 0: continue  # Skip invalid timestamps
            # Keep the candle that appeared later in the raw list (usually more recent update)
            unique_candles_dict[timestamp] = candle
        except (IndexError, TypeError, ValueError):
            lg.warning(f"Skipping invalid candle format during deduplication: {candle}")
            continue

    # Extract unique candles and sort by timestamp
    unique_data = sorted(unique_candles_dict.values(), key=lambda x: x[0])

    duplicates_removed = len(all_ohlcv_data) - len(unique_data)
    if duplicates_removed > 0:
        lg.warning(f"Removed {duplicates_removed} duplicate candle(s) based on timestamp for {symbol}.")

    # Trim excess data if more than requested limit was fetched due to chunking/overlaps
    if len(unique_data) > limit:
        lg.debug(f"Fetched {len(unique_data)} unique candles, trimming to the requested limit of {limit}.")
        unique_data = unique_data[-limit:]

    # --- DataFrame Creation and Cleaning ---
    try:
        lg.debug(f"Processing {len(unique_data)} final unique candles into DataFrame for {symbol}...")
        # Standard OHLCV columns
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Adjust columns based on actual data received (some exchanges might omit volume)
        df = pd.DataFrame(unique_data, columns=cols[:len(unique_data[0])])

        # Convert timestamp to datetime objects (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        # Drop rows where timestamp conversion failed
        initial_len_ts = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        if len(df) < initial_len_ts:
            lg.warning(f"Dropped {initial_len_ts - len(df)} rows with invalid timestamps for {symbol}.")
        if df.empty:
            lg.error(f"DataFrame became empty after timestamp conversion ({symbol}).")
            return pd.DataFrame()

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Use pd.to_numeric first for broad conversion, then apply Decimal
                # Errors='coerce' turns unconvertible values (like non-numeric strings) into NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')

                # Apply Decimal conversion, explicitly handling NaN/Inf introduced by to_numeric
                df[col] = numeric_series.apply(
                    lambda x: _safe_market_decimal(x, f"df.{col}", allow_zero=(col == 'volume'), allow_negative=False)
                              if pd.notna(x) else Decimal('NaN')  # Use helper for safety
                )
                # Check if conversion resulted in all NaNs (indicates bad data type)
                if df[col].isnull().all():
                     lg.warning(f"Column '{col}' for {symbol} became all NaN after Decimal conversion. Original data type might be incompatible.")

            elif col != 'volume':  # Volume might legitimately be missing
                lg.warning(f"Expected OHLC column '{col}' not found in fetched data for {symbol}.")
                # Return empty if essential columns are missing? Or fill with NaN? Filling is risky.
                return pd.DataFrame()  # Fail if essential OHLC missing

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in essential OHLC columns (use columns present in df)
        essential_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        df.dropna(subset=essential_cols, inplace=True)
        # Ensure close price is positive
        if 'close' in df.columns:
            df = df[df['close'] > Decimal('0')]
        # Handle volume column if it exists
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)  # Drop rows with NaN volume
            df = df[df['volume'] >= Decimal('0')]  # Ensure volume is non-negative

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with invalid/NaN OHLCV data for {symbol}.")

        if df.empty:
            lg.warning(f"DataFrame became empty after cleaning NaN/invalid values ({symbol}).")
            return pd.DataFrame()

        # Verify index is monotonic increasing (sorted by time)
        if not df.index.is_monotonic_increasing:
            lg.warning(f"DataFrame index for {symbol} is not monotonic increasing. Sorting...")
            df.sort_index(inplace=True)

        # Optional: Limit DataFrame length to prevent excessive memory usage
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds max ({MAX_DF_LEN}). Trimming oldest data ({symbol}).")
            df = df.iloc[-MAX_DF_LEN:].copy()  # Keep the most recent MAX_DF_LEN rows

        lg.info(f"{NEON_GREEN}Successfully processed {len(df)} klines for {symbol} {timeframe}.{RESET}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing fetched klines into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> MarketInfo | None:
    """Retrieves and standardizes market information for a symbol from the exchange.
    Includes derived fields for convenience (e.g., is_linear, decimal precision/limits).
    Includes retry logic and validation of critical precision data.

    Args:
        exchange: The ccxt exchange instance (markets should be loaded).
        symbol: The market symbol (e.g., 'BTC/USDT').
        logger: The logger instance for messages.

    Returns:
        A MarketInfo TypedDict containing standardized market data, or None if not found/error.
    """
    lg = logger
    lg.debug(f"Retrieving market details for symbol: {symbol}...")
    attempts = 0
    last_exception: Exception | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            market: dict | None = None
            # Check if markets are loaded and contain the symbol
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' not found in cached data. Attempting to refresh market map...")
                try:
                    exchange.load_markets(reload=True)
                    lg.info(f"Market map refreshed. Found {len(exchange.markets)} markets.")
                except Exception as reload_err:
                    # Log refresh error but continue to try fetching the specific market below
                    last_exception = reload_err
                    lg.warning(f"Failed to refresh market map while looking for '{symbol}': {reload_err}")

            # Attempt to get the specific market dictionary using ccxt's safe method
            try:
                # exchange.market() throws BadSymbol if not found after load_markets
                market = exchange.market(symbol)
            except ccxt.BadSymbol:
                # This is definitive: the symbol doesn't exist on the exchange according to ccxt
                lg.error(f"{NEON_RED}Symbol '{symbol}' is invalid or not supported by {exchange.id} according to loaded markets.{RESET}")
                return None  # Non-retryable
            except Exception as fetch_err:
                # Other errors during market dict retrieval (might be temporary network issue?)
                last_exception = fetch_err
                lg.warning(f"Error retrieving market dictionary for '{symbol}': {fetch_err}. Retry {attempts + 1}...")
                market = None  # Ensure market is None to trigger retry logic

            if market:
                lg.debug(f"Raw market data found for {symbol}. Parsing and standardizing...")
                # --- Standardize and Enhance Market Data ---
                std_market = market.copy()  # Work on a copy

                # Basic type flags from ccxt structure
                is_spot = std_market.get('spot', False)
                is_swap = std_market.get('swap', False)
                is_future = std_market.get('future', False)
                is_option = std_market.get('option', False)  # Added option check
                is_contract_base = std_market.get('contract', False)  # Base 'contract' flag

                # Determine if it's any kind of contract
                std_market['is_contract'] = is_swap or is_future or is_option or is_contract_base
                is_linear = std_market.get('linear')  # Can be True/False/None
                is_inverse = std_market.get('inverse')  # Can be True/False/None

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
                     std_market['contract_type_str'] = "Option"  # Handle options if needed
                elif std_market['is_contract']:  # Catch-all for contract types ccxt doesn't label linear/inverse
                     std_market['contract_type_str'] = "Contract (Other)"
                else:
                     std_market['contract_type_str'] = "Unknown"

                # --- Extract Precision and Limits Safely using Helper ---
                precision = std_market.get('precision', {})
                limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                # Convert precision steps to Decimal (must be positive)
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), f"{symbol} prec.amount", allow_zero=False, allow_negative=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), f"{symbol} prec.price", allow_zero=False, allow_negative=False)

                # Convert limits to Decimal (must be non-negative, except maybe cost?)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), f"{symbol} lim.amt.min", allow_zero=True, allow_negative=False)
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), f"{symbol} lim.amt.max", allow_zero=False, allow_negative=False)  # Max > 0
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), f"{symbol} lim.cost.min", allow_zero=True, allow_negative=False)
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), f"{symbol} lim.cost.max", allow_zero=False, allow_negative=False)  # Max > 0

                # Convert contract size to Decimal (default to 1 if missing/invalid/spot)
                contract_size_val = std_market.get('contractSize') if std_market['is_contract'] else '1'
                # Contract size must be positive
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, f"{symbol} contractSize", allow_zero=False, allow_negative=False) or Decimal('1')

                # --- Validation of Critical Data ---
                # Precision steps are essential for placing orders correctly
                if std_market['amount_precision_step_decimal'] is None:
                    lg.critical(f"{NEON_RED}CRITICAL VALIDATION FAILED ({symbol}): Missing essential 'precision.amount' data! Cannot proceed safely with this symbol.{RESET}")
                    return None
                if std_market['price_precision_step_decimal'] is None:
                    lg.critical(f"{NEON_RED}CRITICAL VALIDATION FAILED ({symbol}): Missing essential 'precision.price' data! Cannot proceed safely with this symbol.{RESET}")
                    return None
                # Min amount is often needed for order placement
                if std_market['min_amount_decimal'] is None:
                     lg.warning(f"{NEON_YELLOW}Market Validation Warning ({symbol}): Missing 'limits.amount.min' data. Sizing/ordering might fail.{RESET}")
                     # Decide whether to proceed or return None. Proceeding cautiously for now.

                # --- Log Parsed Details ---
                # Helper for formatting optional Decimals for logging
                def fmt_dec_log(d: Decimal | None) -> str:
                    return d.normalize() if d is not None else 'N/A'

                amt_s = fmt_dec_log(std_market['amount_precision_step_decimal'])
                price_s = fmt_dec_log(std_market['price_precision_step_decimal'])
                min_a = fmt_dec_log(std_market['min_amount_decimal'])
                max_a = fmt_dec_log(std_market['max_amount_decimal'])
                min_c = fmt_dec_log(std_market['min_cost_decimal'])
                max_c = fmt_dec_log(std_market['max_cost_decimal'])
                contr_s = fmt_dec_log(std_market['contract_size_decimal'])
                active_status = std_market.get('active', 'Unknown')

                log_msg = (
                    f"Market Details Parsed ({symbol}): Type={std_market['contract_type_str']}, Active={active_status}\n"
                    f"  Precision (Amount/Price Step): {amt_s} / {price_s}\n"
                    f"  Limits    (Amount Min/Max) : {min_a} / {max_a}\n"
                    f"  Limits    (Cost Min/Max)   : {min_c} / {max_c}\n"
                )
                if std_market['is_contract']:
                     log_msg += f"  Contract Size: {contr_s}"
                lg.debug(log_msg)

                # --- Cast to TypedDict and Return ---
                try:
                    # Attempt to cast the enhanced dictionary to the MarketInfo type
                    # This primarily serves static analysis; runtime check is implicit
                    final_market_info: MarketInfo = std_market  # type: ignore [assignment]
                    return final_market_info
                except Exception as cast_err:
                    # Should not happen if MarketInfo matches the dict structure, but catch just in case
                    lg.error(f"Internal error casting market dictionary to MarketInfo type ({symbol}): {cast_err}. Returning raw dict cautiously.")
                    return std_market  # type: ignore [return-value] # Return the dict anyway

            else:
                # Market object was None after attempting fetch/lookup
                if attempts < MAX_API_RETRIES:
                    lg.warning(f"Market '{symbol}' not found or fetch failed (Attempt {attempts + 1}). Retrying...")
                # else: # Error message handled in the next block after loop finishes

        # --- Error Handling for the Loop Iteration ---
        except ccxt.BadSymbol as e:
            # This might be caught inside, but also handle here for robustness
            lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}")
            return None  # Non-retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error retrieving market info ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to NetworkError fetching market info ({symbol}).{RESET}")
                return None
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error retrieving market info: {e}. Cannot continue.{RESET}")
            return None  # Non-retryable
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error retrieving market info ({symbol}): {e}. Retrying...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to ExchangeError fetching market info ({symbol}).{RESET}")
                return None
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error retrieving market info ({symbol}): {e}{RESET}", exc_info=True)
            return None  # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to get market info for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency from the exchange.
    Handles different account types for exchanges like Bybit (Unified/Contract/Spot).
    Includes retry logic and robust parsing.

    Args:
        exchange: The ccxt exchange instance.
        currency: The currency code (e.g., 'USDT', 'BTC').
        logger: The logger instance for messages.

    Returns:
        The available balance as a Decimal, or None if fetching fails or currency not found.

    Raises:
        ccxt.AuthenticationError: If authentication fails during the balance check.
    """
    lg = logger
    lg.debug(f"Fetching balance for currency: {currency}...")
    attempts = 0
    last_exception: Exception | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: str | None = None
            balance_source: str = "N/A"  # Where the balance was found (e.g., account type, field name)
            found: bool = False
            balance_info: dict | None = None  # Store the last fetched balance structure for debugging
            is_bybit = 'bybit' in exchange.id.lower()

            # For Bybit, try specific account types, then default
            # For other exchanges, the default '' usually works
            # Bybit V5 account types: UNIFIED, CONTRACT, SPOT (and others like FUND)
            # Let's prioritize UNIFIED/CONTRACT for trading, fallback to SPOT/Default
            types_to_check = ['UNIFIED', 'CONTRACT', 'SPOT', ''] if is_bybit else ['']

            for acc_type in types_to_check:
                if not is_bybit and acc_type: continue  # Skip specific types for non-bybit

                try:
                    params = {'accountType': acc_type} if acc_type else {}
                    type_desc = f"Account Type: '{acc_type}'" if acc_type else "Default Account"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")

                    # Use fetch_total_balance for potentially broader compatibility
                    # balance_info = exchange.fetch_total_balance(params=params) # Check if fetch_total_balance is better
                    balance_info = exchange.fetch_balance(params=params)

                    # --- Try standard ccxt structure first ('free' field) ---
                    # Balance structure: { 'CUR': {'free': X, 'used': Y, 'total': Z}, ... }
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        balance_source = f"{type_desc} (ccxt 'free' field)"
                        found = True
                        break  # Found balance, exit account type loop

                    # --- Try Bybit V5 specific structure (nested within 'info') ---
                    # Structure: info -> result -> list -> [ { accountType, coin: [ { coin, availableToWithdraw/availableBalance } ] } ]
                    elif (is_bybit and 'info' in balance_info and
                          isinstance(balance_info.get('info'), dict) and
                          isinstance(balance_info['info'].get('result'), dict) and
                          isinstance(balance_info['info']['result'].get('list'), list)):

                        for account_details in balance_info['info']['result']['list']:
                            # Check if this entry matches the account type we queried (or if query was default '')
                            # And ensure 'coin' list exists
                            fetched_acc_type = account_details.get('accountType')
                            # Match specific type or if default type was queried, accept any type found
                            type_match = (acc_type and fetched_acc_type == acc_type) or (not acc_type)

                            if type_match and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Try different fields for available balance in preferred order
                                        val = coin_data.get('availableToWithdraw')  # Most preferred
                                        src = 'availableToWithdraw'
                                        if val is None:
                                            val = coin_data.get('availableBalance')  # Next best (might include borrowed?)
                                            src = 'availableBalance'
                                        # WalletBalance is less useful as it includes frozen/used margin
                                        # if val is None:
                                        #      val = coin_data.get('walletBalance')
                                        #      src = 'walletBalance'

                                        if val is not None:
                                            balance_str = str(val)
                                            balance_source = f"Bybit V5 ({fetched_acc_type or 'UnknownType'}, field: '{src}')"
                                            found = True
                                            break  # Found coin data, exit coin loop
                                if found: break  # Exit account details loop
                        if found: break  # Exit account type loop

                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # Bybit might throw specific errors for invalid account types
                    if acc_type and ("account type does not exist" in err_str or "invalid account type" in err_str):
                        lg.debug(f"Account type '{acc_type}' not found or invalid for balance check. Trying next...")
                        last_exception = e  # Keep track of the last error
                        continue  # Try the next account type
                    elif acc_type:
                         # Other exchange error for a specific type, log and try next
                         lg.debug(f"Exchange error fetching balance for {acc_type}: {e}. Trying next...")
                         last_exception = e
                         continue
                    else:
                        # Exchange error on the default account type - raise it to main handler
                        raise e
                except Exception as e:
                    # Unexpected error during a specific account type check
                    lg.warning(f"Unexpected error fetching balance for {acc_type or 'Default'}: {e}. Trying next...")
                    last_exception = e
                    continue  # Try the next account type

            # --- Process Result After Checking All Account Types ---
            if found and balance_str is not None:
                # Use safe decimal conversion, allowing zero but not negative
                bal_dec = _safe_market_decimal(balance_str, f"balance_str({currency})", allow_zero=True, allow_negative=False)

                if bal_dec is not None:
                    lg.debug(f"Successfully parsed balance ({currency}) from {balance_source}: {bal_dec.normalize()}")
                    return bal_dec
                else:
                    # If conversion fails despite finding a string, treat as an exchange error
                    raise ccxt.ExchangeError(f"Failed to convert valid balance string '{balance_str}' to non-negative Decimal for {currency}.")
            elif not found and balance_info is not None:
                # Currency not found in any checked structure
                lg.debug(f"Balance information for currency '{currency}' not found in the response structure(s).")
                # Don't raise error here, just continue loop if retries remain
                last_exception = ccxt.ExchangeError(f"Balance for '{currency}' not found in response.")
                # Continue to retry logic below
            elif balance_info is None and not found:
                 # If balance_info itself was None (e.g., from API error)
                 lg.debug(f"Balance info was not retrieved for {currency} in this attempt.")
                 # Rely on exception handling below for retries

        # --- Error Handling for fetch_balance call (outer loop) ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance ({currency}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue  # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # This is critical and non-retryable
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching balance: {e}. Cannot continue.{RESET}")
            raise e  # Re-raise to be caught by the caller (e.g., initialize_exchange)
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., temporary issues)
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance ({currency}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as potentially fatal for balance check
            return None

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> PositionInfo | None:
    """Fetches the currently open position for a specific contract symbol.
    Returns None if no position exists or if the symbol is not a contract.
    Standardizes the position information, parses key values to Decimal, and includes retry logic.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        market_info: The corresponding MarketInfo dictionary for the symbol.
        logger: The logger instance for messages.

    Returns:
        A PositionInfo TypedDict if an active position exists (size != 0), otherwise None.
    """
    lg = logger

    # --- Pre-checks ---
    if not market_info.get('is_contract'):
        lg.debug(f"Position check skipped for {symbol}: It is a '{market_info.get('contract_type_str', 'Unknown')}' market, not a contract.")
        return None

    market_id = market_info.get('id')
    # Determine category for Bybit V5 based on standardized info
    category = 'linear'  # Default guess for contracts
    if market_info.get('is_linear'): category = 'linear'
    elif market_info.get('is_inverse'): category = 'inverse'
    is_bybit = 'bybit' in exchange.id.lower()

    if not market_id:
        lg.error(f"Cannot check position for {symbol}: Invalid market ID ('{market_id}') in market_info.")
        return None
    if is_bybit and category not in ['linear', 'inverse']:
         lg.error(f"Cannot check position for Bybit symbol {symbol}: Invalid category '{category}'. Must be 'linear' or 'inverse'.")
         return None

    lg.debug(f"Checking for open position for {symbol} (Market ID: '{market_id}', Category: '{category if is_bybit else 'N/A'}')...")

    attempts = 0
    last_exception: Exception | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions ({symbol}, Attempt {attempts + 1})...")
            positions: list[dict] = []  # Initialize as empty list

            # --- Fetch Positions from Exchange ---
            try:
                params = {}
                # Bybit V5 requires category
                if is_bybit:
                     params['category'] = category
                     # Optionally filter by symbol or market_id if API supports it well
                     # params['symbol'] = market_id # Sometimes helps, sometimes hinders if API expects base currency for position fetch

                lg.debug(f"Fetching positions with parameters: {params}")

                # Use fetch_positions if available and reliable
                # Some exchanges require fetching all positions and filtering locally
                if exchange.has.get('fetchPositions'):
                    # Fetch for the specific symbol if possible, otherwise fetch all and filter
                    # Note: fetch_positions([symbol]) might not work reliably on all exchanges
                    # Fetching all might be safer but less efficient. Let's try specific first.
                    try:
                        all_fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
                    except ccxt.NotSupported:
                         lg.debug(f"fetch_positions with specific symbol not supported by {exchange.id}. Fetching all...")
                         all_fetched_positions = exchange.fetch_positions(params=params)  # Fetch all for the category/default
                    except Exception as fetch_all_err:
                         # Handle case where even fetch_positions (all) fails
                         lg.warning(f"Error using fetch_positions: {fetch_all_err}. Falling back to fetch_position if available.")
                         all_fetched_positions = []  # Ensure it's empty for fallback

                    # Filter results for the specific symbol
                    positions = [
                        p for p in all_fetched_positions
                        if p and (p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id)
                    ]
                    lg.debug(f"Fetched {len(all_fetched_positions)} position(s) via fetch_positions, "
                             f"filtered to {len(positions)} matching {symbol}/{market_id}.")

                elif exchange.has.get('fetchPosition'):
                     # Fallback to fetchPosition if fetchPositions is not available/supported
                     lg.debug(f"Using fallback fetchPosition for {symbol}...")
                     pos = exchange.fetch_position(symbol, params=params)
                     # fetch_position usually returns a single dict or raises error if no position
                     # Ensure size exists and is non-zero before adding
                     pos_size_str = str(pos.get('info', {}).get('size', pos.get('contracts', ''))).strip()
                     if pos and pos_size_str and _safe_market_decimal(pos_size_str, "pos_size_check", allow_zero=False):
                          positions = [pos]  # Wrap in list for consistency
                     else:
                          positions = []  # No active position found
                     lg.debug(f"fetchPosition returned: {'Position found' if positions else 'No active position found'}")
                else:
                    raise ccxt.NotSupported(f"{exchange.id} does not support fetchPositions or fetchPosition.")

            except ccxt.ExchangeError as e:
                 # Specific handling for "position not found" errors which are not real errors
                 # Bybit V5 retCode: 110025 = position not found / Position is closed
                 common_no_pos_msgs = ["position not found", "no position", "position does not exist", "position is closed"]
                 bybit_no_pos_codes = [110025]

                 err_str = str(e).lower()
                 # Try to extract Bybit retCode if present
                 code_str = ""
                 match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
                 if match: code_str = match.group(2)
                 else: code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))  # Fallback

                 is_bybit_no_pos = is_bybit and code_str and any(str(c) == code_str for c in bybit_no_pos_codes)
                 is_common_no_pos = any(msg in err_str for msg in common_no_pos_msgs)

                 if is_bybit_no_pos or is_common_no_pos:
                     lg.info(f"No open position found for {symbol} (API indicated no position: Code='{code_str}', Msg='{err_str[:60]}...').")
                     return None  # This is the expected outcome when no position exists
                 else:
                     # Re-raise other exchange errors
                     raise e

            # --- Process Fetched Positions ---
            active_raw_position: dict | None = None

            # Define a small threshold for position size based on market precision
            # Use amount step if available, otherwise a very small Decimal fraction
            size_threshold = Decimal('1e-9')  # Default tiny threshold
            amt_step = market_info.get('amount_precision_step_decimal')
            if amt_step and amt_step > 0:
                # Use a fraction of the step size (e.g., 1%) if step is reasonably large,
                # otherwise stick to a small absolute value.
                if amt_step > Decimal('1e-8'):
                     size_threshold = amt_step * Decimal('0.01')
                # else: keep the 1e-9 default
            lg.debug(f"Using position size threshold > {size_threshold.normalize()} for {symbol}.")

            # Iterate through filtered positions to find one with non-negligible size
            for pos_data in positions:
                # Try to get size from 'info' (often more reliable) or standard 'contracts' field
                # Prefer 'size' from Bybit V5 'info' if available
                size_raw = pos_data.get('info', {}).get('size')
                if size_raw is None:  # Fallback to 'contracts' if 'size' not in info
                    size_raw = pos_data.get('contracts')

                # Safely convert size to Decimal
                size_decimal = _safe_market_decimal(size_raw, f"{symbol} pos.size/contracts", allow_zero=True, allow_negative=True)

                if size_decimal is None:
                    lg.debug(f"Skipping position data with missing or invalid size field ({symbol}). Raw size: {repr(size_raw)}")
                    continue

                # Check if absolute size exceeds the threshold (effectively non-zero)
                if abs(size_decimal) > size_threshold:
                    active_raw_position = pos_data
                    # Store the parsed Decimal size directly in the dict for later use
                    active_raw_position['size_decimal'] = size_decimal
                    lg.debug(f"Found active position candidate for {symbol} with size: {size_decimal.normalize()}")
                    break  # Found the first active position, stop searching
                else:
                     lg.debug(f"Skipping position data with size near zero ({symbol}, Size: {size_decimal.normalize()}).")

            # --- Standardize and Return Active Position ---
            if active_raw_position:
                std_pos = active_raw_position.copy()
                info_dict = std_pos.get('info', {})  # Raw exchange-specific data

                # Determine Side (long/short) - crucial and sometimes inconsistent
                parsed_size = std_pos['size_decimal']  # Use the Decimal size we stored
                side = std_pos.get('side')  # Standard ccxt field

                # Infer side if standard field is missing or ambiguous
                if side not in ['long', 'short']:
                    # Try inferring from Bybit V5 'info.side' (Buy/Sell) or from the sign of the size
                    side_v5 = str(info_dict.get('side', '')).strip().lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif parsed_size > size_threshold: side = 'long'  # Positive size implies long
                    elif parsed_size < -size_threshold: side = 'short'  # Negative size implies short
                    else: side = None  # Cannot determine side

                if not side:
                    lg.error(f"Could not determine position side for {symbol}. Size: {parsed_size}. Raw Info: {info_dict}")
                    return None  # Cannot proceed without knowing the side

                std_pos['side'] = side

                # Safely parse other relevant fields to Decimal where applicable
                # Prefer standard ccxt fields, fallback to 'info' dict fields if needed
                std_pos['entryPrice'] = _safe_market_decimal(
                    std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice'),  # Bybit V5 uses avgPrice in info
                    f"{symbol} pos.entry", allow_zero=False, allow_negative=False)
                std_pos['leverage'] = _safe_market_decimal(
                    std_pos.get('leverage') or info_dict.get('leverage'),
                    f"{symbol} pos.leverage", allow_zero=False, allow_negative=False)  # Leverage > 0
                std_pos['liquidationPrice'] = _safe_market_decimal(
                    std_pos.get('liquidationPrice') or info_dict.get('liqPrice'),  # Bybit V5 uses liqPrice in info
                    f"{symbol} pos.liq", allow_zero=False, allow_negative=False)  # Liq price > 0
                std_pos['unrealizedPnl'] = _safe_market_decimal(
                    std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl'),  # Bybit V5 uses unrealisedPnl
                    f"{symbol} pos.pnl", allow_zero=True, allow_negative=True)  # PnL can be zero or negative
                std_pos['notional'] = _safe_market_decimal(
                    std_pos.get('notional') or info_dict.get('positionValue'),  # Bybit V5 uses positionValue
                    f"{symbol} pos.notional", allow_zero=True, allow_negative=False)  # Notional >= 0

                # Extract protection orders (SL, TP, TSL) - these are often strings in 'info'
                # We need to check if the value represents an *active* order (e.g., not '0' or '0.0')
                def get_protection_value(field_name: str) -> str | None:
                    """Safely gets a protection order value from info, returns None if zero/empty/invalid."""
                    value = info_dict.get(field_name)
                    if value is None: return None
                    s_value = str(value).strip()
                    # Check if it's a non-zero numeric value using safe decimal conversion
                    dec_val = _safe_market_decimal(s_value, f"{symbol} prot.{field_name}", allow_zero=False, allow_negative=False)
                    if dec_val is not None:
                         return s_value  # Return the original string if it represents a valid, non-zero price/value
                    else:
                         return None  # Treat '0', '0.0', empty string, or invalid as no order set

                std_pos['stopLossPrice'] = get_protection_value('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_value('takeProfit')
                # Bybit V5 TSL fields: trailingStop (distance/price), activePrice (activation price)
                std_pos['trailingStopLoss'] = get_protection_value('trailingStop')  # This is the trailing distance/offset
                std_pos['tslActivationPrice'] = get_protection_value('activePrice')  # This is the activation price

                # Initialize bot state tracking fields (these will be updated by bot logic)
                std_pos['be_activated'] = False  # Break-even not yet activated by bot logic for *this instance*
                # TSL considered active *by the exchange* if the exchange reports a non-zero trailingStop value AND activation price
                # Note: Internal bot state `position_state['tsl_activated']` tracks if the *bot* has activated it.
                exchange_tsl_active = bool(std_pos['trailingStopLoss']) and bool(std_pos['tslActivationPrice'])
                std_pos['tsl_activated'] = exchange_tsl_active  # Reflect current exchange status

                # --- Log Found Position ---
                # Helper for logging optional Decimal values safely
                def fmt_log(val: Any | None) -> str:
                    # Use safe conversion allowing zero/negative for display where appropriate
                    dec = _safe_market_decimal(val, 'log_fmt', True, True)
                    return dec.normalize() if dec is not None else 'N/A'

                ep = fmt_log(std_pos.get('entryPrice'))
                sz = std_pos['size_decimal'].normalize()
                sl = fmt_log(std_pos.get('stopLossPrice'))
                tp = fmt_log(std_pos.get('takeProfitPrice'))
                tsl_dist = fmt_log(std_pos.get('trailingStopLoss'))
                tsl_act = fmt_log(std_pos.get('tslActivationPrice'))
                tsl_str = "Inactive"
                if exchange_tsl_active:
                     tsl_str = f"ACTIVE (Dist/Px={tsl_dist} | ActPx={tsl_act})"
                elif std_pos.get('trailingStopLoss') or std_pos.get('tslActivationPrice'):  # Partially set?
                     tsl_str = f"PARTIAL? (Dist/Px={tsl_dist} | ActPx={tsl_act})"

                pnl = fmt_log(std_pos.get('unrealizedPnl'))
                liq = fmt_log(std_pos.get('liquidationPrice'))
                lev = fmt_log(std_pos.get('leverage'))
                notional = fmt_log(std_pos.get('notional'))

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} "
                        f"Size={sz}, Entry={ep}, Notional={notional}, Liq={liq}, Lev={lev}x, PnL={pnl}\n"
                        f"  Protections: SL={sl}, TP={tp}, TSL={tsl_str}")

                # --- Cast to TypedDict and Return ---
                try:
                    final_position_info: PositionInfo = std_pos  # type: ignore [assignment]
                    return final_position_info
                except Exception as cast_err:
                    lg.error(f"Internal error casting position dictionary to PositionInfo type ({symbol}): {cast_err}. Returning raw dict cautiously.")
                    return std_pos  # type: ignore [return-value]

            else:
                # No position found with size > threshold after checking all returned data
                lg.info(f"No active position found for {symbol} (checked {len(positions)} entries).")
                return None

        # --- Error Handling for the Loop Iteration ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions ({symbol}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue  # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching positions: {e}. Cannot continue.{RESET}")
            return None  # Non-retryable
        except ccxt.NotSupported as e:
             last_exception = e
             lg.error(f"{NEON_RED}Position fetching method not supported by {exchange.id}: {e}. Cannot get position info.{RESET}")
             return None  # Non-retryable
        except ccxt.ExchangeError as e:
            # Handled specific "no position" cases above, this catches others
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching positions ({symbol}): {e}{RESET}", exc_info=True)
            return None  # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """Sets the leverage for a given contract symbol using ccxt's set_leverage.
    Handles specific requirements for exchanges like Bybit V5 (category, buy/sell leverage as strings).
    Includes retry logic and checks for success, no change needed, or fatal errors.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        leverage: The desired integer leverage level (e.g., 10 for 10x).
        market_info: The MarketInfo dictionary for the symbol.
        logger: The logger instance for messages.

    Returns:
        True if leverage was set successfully or already set to the desired value, False otherwise.
    """
    lg = logger

    # --- Pre-checks ---
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True  # No action needed for non-contracts

    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Leverage setting failed for {symbol}: Invalid leverage value '{leverage}'. Must be a positive integer.")
        return False

    # Check if the exchange supports setting leverage via ccxt method
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        # Check if leverage can be set via market creation/modification (less common with ccxt)
        if exchange.has.get('createMarket'):
             lg.warning(f"Exchange {exchange.id} might require setting leverage via market creation/modification, not directly via set_leverage. Skipping.")
             # Assume success for now if direct method not available, but warn user.
             # A more robust solution would check market capabilities or use exchange-specific methods.
             return True  # Cautiously assume ok if method missing, user must configure manually
        else:
             lg.error(f"Leverage setting failed: Exchange {exchange.id} does not support setLeverage method or market modification via ccxt.")
             return False

    market_id = market_info.get('id')
    # Determine category for Bybit V5
    category = 'linear'  # Default guess
    if market_info.get('is_linear'): category = 'linear'
    elif market_info.get('is_inverse'): category = 'inverse'
    is_bybit = 'bybit' in exchange.id.lower()

    if not market_id:
         lg.error(f"Leverage setting failed for {symbol}: Market ID missing in market_info.")
         return False
    if is_bybit and category not in ['linear', 'inverse']:
         lg.error(f"Leverage setting failed for Bybit symbol {symbol}: Invalid category '{category}'. Must be 'linear' or 'inverse'.")
         return False

    lg.info(f"Attempting to set leverage for {symbol} (Market ID: {market_id}, Category: {category if is_bybit else 'N/A'}) to {leverage}x...")

    attempts = 0
    last_exception: Exception | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"set_leverage call attempt {attempts + 1} for {symbol} to {leverage}x...")
            params = {}

            # --- Exchange-Specific Parameter Handling (Bybit V5 example) ---
            if is_bybit:
                 # Bybit V5 requires category and separate buy/sell leverage as strings
                 params = {
                     'category': category,
                     'buyLeverage': str(leverage),  # Must be strings for Bybit V5 API
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 specific leverage parameters: {params}")

            # --- Execute set_leverage Call ---
            # Note: ccxt `set_leverage` takes leverage as float/int, symbol, and params
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            lg.debug(f"Raw response from set_leverage ({symbol}): {response}")

            # --- Response Validation (especially for Bybit V5) ---
            ret_code_str: str | None = None
            ret_msg: str = "Response format not recognized or empty."

            if isinstance(response, dict):
                # Try extracting Bybit V5 style response codes/messages from 'info' first
                info_dict = response.get('info', {})
                raw_code = info_dict.get('retCode')  # Primary location in V5
                if raw_code is None: raw_code = response.get('retCode')  # Fallback to root level
                ret_code_str = str(raw_code) if raw_code is not None else None
                ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown message'))  # Prefer info.retMsg

            # Check Bybit success code (0) or "leverage not modified" code (e.g., 110045)
            # Note: Code 110045 might mean "leverage not modified" or other param errors. Check message too.
            bybit_success_codes = ['0']
            bybit_no_change_codes = ['110045']  # "Parameter Error" - often means leverage not modified

            if ret_code_str in bybit_success_codes:
                lg.info(f"{NEON_GREEN}Leverage successfully set for {symbol} to {leverage}x (Code: {ret_code_str}).{RESET}")
                return True
            elif ret_code_str in bybit_no_change_codes and ("leverage not modified" in ret_msg.lower() or "same leverage" in ret_msg.lower()):
                lg.info(f"{NEON_YELLOW}Leverage for {symbol} is already {leverage}x (Code: {ret_code_str} - Not Modified). Success.{RESET}")
                return True
            elif ret_code_str is not None and ret_code_str not in bybit_success_codes:
                # Specific Bybit error code received that isn't success or known no-change
                raise ccxt.ExchangeError(f"Bybit API error setting leverage for {symbol}: {ret_msg} (Code: {ret_code_str})")
            elif response is not None and not is_bybit:
                 # Non-Bybit exchange or unrecognized Bybit success response - assume success if no exception
                 lg.info(f"{NEON_GREEN}Leverage set/confirmed for {symbol} to {leverage}x (No specific code checked/found in response, assumed success).{RESET}")
                 return True
            elif response is None:
                 # Response was None or empty, which is unexpected
                 raise ccxt.ExchangeError(f"Received unexpected empty response after setting leverage for {symbol}.")
            # else: If Bybit response but code is None or not recognized, let retry logic handle

        # --- Error Handling for set_leverage call ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str_lower = str(e).lower()
            # Try to extract error code again, specifically for logging/decision making
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            else: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))  # Fallback

            lg.error(f"{NEON_RED}Exchange error setting leverage ({symbol} to {leverage}x): {e} (Code: {err_code_str}){RESET}")

            # Check if the error indicates leverage was already set (redundant but safe)
            if err_code_str in bybit_no_change_codes and ("leverage not modified" in err_str_lower or "same leverage" in err_str_lower):
                lg.info(f"{NEON_YELLOW}Leverage already set to {leverage}x (confirmed via error response {err_code_str}). Success.{RESET}")
                return True

            # Check for known fatal/non-retryable error codes or messages
            # These may need adjustment based on the specific exchange (using Bybit V5 examples)
            fatal_codes = [
                '10001',  # Parameter error (e.g., invalid leverage value for symbol)
                '10004',  # Sign check error (API keys)
                '110009',  # Symbol expired
                '110013',  # Risk limit error (might prevent leverage change)
                '110028',  # Cross margin mode cannot modify leverage
                '110043',  # Set leverage error when position exists (isolated margin)
                '110044',  # Set leverage error when order exists (isolated margin)
                '110055',  # Cannot set leverage under Isolated margin mode for cross margin position
                '3400045',  # Leverage less than min limit / greater than max limit
                '110066',  # Cannot set leverage under Portfolio Margin account mode
            ]
            fatal_messages = [
                "margin mode", "position exists", "order exists", "risk limit", "parameter error",
                "insufficient available balance", "invalid leverage value",
                "isolated margin mode", "portfolio margin"
            ]
            is_fatal_code = err_code_str in fatal_codes
            is_fatal_message = any(msg in err_str_lower for msg in fatal_messages)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE leverage error for {symbol}. Aborting leverage setting.{RESET}")
                # Potentially add more specific advice based on the error
                if "position exists" in err_str_lower or "order exists" in err_str_lower or err_code_str in ['110043', '110044']:
                    lg.error(" >> Cannot change leverage while a position or active orders exist (especially in Isolated Margin).")
                elif "margin mode" in err_str_lower or err_code_str in ['110028', '110055', '110066']:
                     lg.error(" >> Leverage change might conflict with current margin mode (Cross/Isolated/Portfolio) or account settings.")
                elif "parameter error" in err_str_lower or "invalid leverage" in err_str_lower or err_code_str in ['10001', '3400045']:
                     lg.error(f" >> Leverage value {leverage}x might be invalid for {symbol}. Check exchange limits.")

                return False  # Non-retryable failure

            # If not fatal, proceed to retry logic
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to ExchangeError setting leverage ({symbol}).{RESET}")
                return False

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting leverage ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to NetworkError setting leverage ({symbol}).{RESET}")
                return False
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error setting leverage ({symbol}): {e}. Cannot continue.{RESET}")
            return False  # Non-retryable
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting leverage ({symbol}): {e}{RESET}", exc_info=True)
            return False  # Treat unexpected errors as fatal

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return False


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: MarketInfo,
    exchange: ccxt.Exchange,  # Keep exchange for potential future use (e.g., fetching quote price)
    logger: logging.Logger
) -> Decimal | None:
    """Calculates the position size based on available balance, risk percentage,
    entry price, and stop loss price. Considers market constraints (min/max size,
    step size, min/max cost) and contract type (linear/inverse). Applies precision rounding.

    Args:
        balance: Available trading balance in the quote currency.
        risk_per_trade: The fraction of the balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: The calculated initial stop loss price (must be positive).
        entry_price: The estimated entry price (e.g., current market price, must be positive).
        market_info: The MarketInfo dictionary for the symbol (must contain valid precision/limits).
        exchange: The ccxt exchange instance (currently unused but kept for signature consistency).
        logger: The logger instance for messages.

    Returns:
        The calculated and adjusted position size as a Decimal (in base currency for spot,
        or number of contracts for futures), rounded to the correct precision step.
        Returns None if calculation fails due to invalid inputs or constraints.
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE')  # Fallback if missing
    base_currency = market_info.get('base', 'BASE')   # Fallback if missing
    is_inverse = market_info.get('is_inverse', False)
    is_spot = market_info.get('spot', False)
    # Determine the unit of the calculated size for logging
    size_unit = base_currency if is_spot else "Contracts"

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    # --- Input Validation ---
    if balance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Balance is zero or negative ({balance.normalize()} {quote_currency}).")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        # Risk must be strictly positive and less than or equal to 1 (100%)
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
             raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive).")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None
    # Ensure prices are valid Decimals and positive
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive Stop Loss price ({initial_stop_loss_price}).")
        return None
    if not isinstance(entry_price, Decimal) or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive Entry price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): Entry price and Stop Loss price cannot be the same ({entry_price.normalize()}).")
        return None

    # --- Extract Market Constraints ---
    try:
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal']  # Used for logging/potential adjustments
        min_amount = market_info['min_amount_decimal']  # Can be None
        max_amount = market_info['max_amount_decimal']  # Can be None
        min_cost = market_info['min_cost_decimal']     # Can be None
        max_cost = market_info['max_cost_decimal']     # Can be None
        contract_size = market_info['contract_size_decimal']  # Should default to 1 if missing

        # Validate critical constraints needed for calculation/adjustment
        if not (amount_step and amount_step > 0): raise ValueError("Amount precision step (amount_step) is missing or invalid.")
        if not (price_step and price_step > 0): raise ValueError("Price precision step (price_step) is missing or invalid.")
        if not (contract_size and contract_size > 0): raise ValueError("Contract size (contract_size) is missing or invalid.")
        # Treat None limits as non-restrictive for calculations
        min_amount_eff = min_amount if min_amount is not None else Decimal('0')
        max_amount_eff = max_amount if max_amount is not None else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None else Decimal('0')
        max_cost_eff = max_cost if max_cost is not None else Decimal('inf')

        lg.debug(f"  Market Constraints ({symbol}):")
        lg.debug(f"    Amount Step: {amount_step.normalize()}, Min Amount: {fmt_dec_log(min_amount)}, Max Amount: {fmt_dec_log(max_amount)}")
        lg.debug(f"    Price Step : {price_step.normalize()}")
        lg.debug(f"    Cost Min   : {fmt_dec_log(min_cost)}, Cost Max: {fmt_dec_log(max_cost)}")
        lg.debug(f"    Contract Size: {contract_size.normalize()}, Type: {market_info['contract_type_str']}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error accessing or validating required market details: {e}")
        lg.debug(f"  Problematic MarketInfo: {market_info}")
        return None

    # --- Core Size Calculation ---
    # Quantize risk amount early to avoid precision issues down the line
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN)  # Risk amount in quote currency
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance <= Decimal('0'):
        # Should be caught by earlier check, but safeguard here
        lg.error(f"Sizing failed ({symbol}): Stop loss distance is zero or negative.")
        return None

    lg.info("  Inputs:")
    lg.info(f"    Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"    Risk % : {risk_decimal:.2%}")
    lg.info(f"    Risk Amt: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"    Entry Price: {entry_price.normalize()}")
    lg.info(f"    Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"    SL Distance: {stop_loss_distance.normalize()}")

    calculated_size = Decimal('0')
    try:
        if not is_inverse:
            # --- Linear Contract or Spot ---
            # Formula: Size = Risk Amount / (ContractSize * SL Distance)
            # Value change per contract/base unit at the stop loss level
            value_change_per_unit = stop_loss_distance * contract_size
            if value_change_per_unit <= Decimal('1e-18'):  # Avoid division by near-zero
                lg.error(f"Sizing failed ({symbol}, Linear/Spot): Calculated value change per unit ({value_change_per_unit}) is near zero. Check prices/contract size.")
                return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Calculation: Size = {risk_amount_quote.normalize()} / ({stop_loss_distance.normalize()} * {contract_size.normalize()}) = {calculated_size}")
        else:
            # --- Inverse Contract ---
            # Formula: Size = Risk Amount / (ContractSize * |(1/Entry) - (1/SL)|)
            # Value change per contract at the stop loss level (in quote currency terms)
            inverse_factor = abs((Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price))
            if inverse_factor <= Decimal('1e-18'):  # Avoid division by near-zero
                lg.error(f"Sizing failed ({symbol}, Inverse): Calculated inverse factor ({inverse_factor}) is near zero. Check prices.")
                return None
            risk_per_contract_unit = contract_size * inverse_factor
            if risk_per_contract_unit <= Decimal('1e-18'):  # Avoid division by near-zero
                 lg.error(f"Sizing failed ({symbol}, Inverse): Calculated risk per contract unit ({risk_per_contract_unit}) is near zero.")
                 return None
            calculated_size = risk_amount_quote / risk_per_contract_unit
            lg.debug(f"  Inverse Calculation: Size = {risk_amount_quote.normalize()} / ({contract_size.normalize()} * {inverse_factor}) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as e:
        lg.error(f"Sizing failed ({symbol}): Mathematical error during core calculation: {e}.")
        return None

    # Ensure calculated size is positive before proceeding
    if calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial calculated size is zero or negative ({calculated_size.normalize()}). "
                 f"Check risk amount ({risk_amount_quote.normalize()}), SL distance ({stop_loss_distance.normalize()}), and contract type/size.")
        return None

    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Adjust Size Based on Constraints ---
    adjusted_size = calculated_size

    # Helper to estimate cost accurately based on contract type
    def estimate_cost(size: Decimal, price: Decimal) -> Decimal | None:
        """Estimates the cost of a position in quote currency."""
        if not isinstance(size, Decimal) or not isinstance(price, Decimal) or price <= 0 or size <= 0:
            lg.warning(f"Cost estimation skipped: Invalid size ({size}) or price ({price}).")
            return None
        try:
            cost: Decimal
            if not is_inverse:  # Linear / Spot
                # Cost = Size (Contracts/Base) * ContractSize * EntryPrice
                cost = size * contract_size * price
            else:  # Inverse
                # Cost = Size (Contracts) * ContractSize / EntryPrice (Cost is in Base currency value, but expressed in Quote)
                cost = (size * contract_size) / price
            # Quantize cost to a reasonable precision (e.g., 8 decimal places) for checks
            return cost.quantize(Decimal('1e-8'), ROUND_UP)  # Round up cost estimate slightly for safety
        except (InvalidOperation, OverflowError, ZeroDivisionError) as cost_err:
            lg.error(f"Cost estimation failed: {cost_err} (Size: {size}, Price: {price}, ContractSize: {contract_size}, Inverse: {is_inverse})")
            return None

    # 1. Apply Min/Max Amount Limits
    # Ensure comparison uses effective limits (handling None)
    if adjusted_size < min_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Initial size {adjusted_size.normalize()} < Min Amount {fmt_dec_log(min_amount)}. Adjusting UP to Min Amount.{RESET}")
        adjusted_size = min_amount_eff  # Use the Decimal value
    if adjusted_size > max_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Initial size {adjusted_size.normalize()} > Max Amount {fmt_dec_log(max_amount)}. Adjusting DOWN to Max Amount.{RESET}")
        adjusted_size = max_amount_eff  # Use the Decimal value

    # Ensure adjusted size is still positive after min/max amount adjustments
    if adjusted_size <= Decimal('0'):
         lg.error(f"Sizing failed ({symbol}): Size became zero or negative ({adjusted_size.normalize()}) after applying Min/Max Amount limits. Min: {fmt_dec_log(min_amount)}, Max: {fmt_dec_log(max_amount)}")
         return None

    lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 2. Apply Min/Max Cost Limits (Requires estimating cost based on size after amount limits)
    cost_adjusted = False
    estimated_cost = estimate_cost(adjusted_size, entry_price)

    if estimated_cost is not None:
        lg.debug(f"  Estimated Cost (after amount limits, {symbol}): {estimated_cost.normalize()} {quote_currency}")

        # Check Min Cost
        if estimated_cost < min_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Estimated cost {estimated_cost.normalize()} < Min Cost {fmt_dec_log(min_cost)}. Attempting to increase size.{RESET}")
            cost_adjusted = True
            try:
                # Calculate the theoretical size needed to meet min cost
                required_size_for_min_cost: Decimal
                if not is_inverse:
                    denominator = entry_price * contract_size
                    if denominator <= 0: raise ZeroDivisionError("Entry price * contract size is zero or negative.")
                    required_size_for_min_cost = min_cost_eff / denominator
                else:
                    numerator = min_cost_eff * entry_price
                    if contract_size <= 0: raise ZeroDivisionError("Contract size is zero or negative.")
                    required_size_for_min_cost = numerator / contract_size

                if required_size_for_min_cost <= 0: raise ValueError("Calculated required size for min cost is non-positive.")

                lg.info(f"  Theoretical size required for Min Cost ({symbol}): {required_size_for_min_cost.normalize()} {size_unit}")

                # Adjust size up to the required size, but respect min_amount and max_amount
                # Start with the larger of the current min_amount or the required size
                target_size = max(min_amount_eff, required_size_for_min_cost)

                # Ensure the target size doesn't exceed max_amount
                if target_size > max_amount_eff:
                    lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet Min Cost ({fmt_dec_log(min_cost)}). "
                             f"Required size ({target_size.normalize()}) exceeds Max Amount ({fmt_dec_log(max_amount)}).{RESET}")
                    return None
                else:
                    adjusted_size = target_size
                    lg.info(f"  Adjusted size UP to meet Min Cost (respecting limits): {adjusted_size.normalize()} {size_unit}")
                    # Re-estimate cost after adjustment for Max Cost check below (if needed)
                    estimated_cost = estimate_cost(adjusted_size, entry_price)
                    if estimated_cost: lg.debug(f"  Re-estimated Cost: {estimated_cost.normalize()} {quote_currency}")

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating size required for Min Cost: {e}.{RESET}")
                return None

        # Check Max Cost (only if cost was successfully estimated)
        if estimated_cost is not None and estimated_cost > max_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Estimated cost {estimated_cost.normalize()} > Max Cost {fmt_dec_log(max_cost)}. Attempting to reduce size.{RESET}")
            cost_adjusted = True
            try:
                # Calculate the theoretical maximum size allowed by max cost
                max_size_for_max_cost: Decimal
                if not is_inverse:
                     denominator = entry_price * contract_size
                     if denominator <= 0: raise ZeroDivisionError("Entry price * contract size is zero or negative.")
                     max_size_for_max_cost = max_cost_eff / denominator
                else:
                     numerator = max_cost_eff * entry_price
                     if contract_size <= 0: raise ZeroDivisionError("Contract size is zero or negative.")
                     max_size_for_max_cost = numerator / contract_size

                if max_size_for_max_cost <= 0: raise ValueError("Calculated max size allowed by max cost is non-positive.")

                lg.info(f"  Theoretical max size allowed by Max Cost ({symbol}): {max_size_for_max_cost.normalize()} {size_unit}")

                # Adjust size down to the max allowed by cost, but ensure it doesn't go below min_amount
                target_size = min(adjusted_size, max_size_for_max_cost)  # Take the smaller of current or max allowed by cost
                final_target_size = max(min_amount_eff, target_size)  # Ensure it's still >= min_amount

                # Log the adjustment clearly
                if final_target_size < adjusted_size:  # Check if adjustment actually happened
                     adjusted_size = final_target_size
                     lg.info(f"  Adjusted size DOWN to meet Max Cost (respecting Min Amount): {adjusted_size.normalize()} {size_unit}")
                     if adjusted_size < target_size:  # Log if Min Amount capped the reduction
                          lg.debug(f"    (Note: Size reduction for Max Cost was limited by Min Amount {min_amount_eff.normalize()})")
                else:
                     lg.debug("  Size already within Max Cost limit. No reduction needed.")

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating max size allowed by Max Cost: {e}.{RESET}")
                return None

    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        # Cost limits exist, but we couldn't estimate cost initially
        lg.warning(f"Could not estimate position cost accurately for {symbol}. Cost limit checks (Min: {fmt_dec_log(min_cost)}, Max: {fmt_dec_log(max_cost)}) were skipped.")

    if cost_adjusted:
        lg.debug(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 3. Apply Amount Precision (Step Size) - Crucial final step, round DOWN
    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step size is not positive.")
        # Divide by step, quantize to integer (rounding down), then multiply back
        final_size = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN) * amount_step

        if final_size != adjusted_size:
            lg.info(f"Applied amount precision ({symbol}, Step: {amount_step.normalize()}, Rounded DOWN): "
                    f"{adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
        else:
            lg.debug(f"Size already conforms to amount precision ({symbol}, Step: {amount_step.normalize()}).")

    except (InvalidOperation, ValueError, ZeroDivisionError) as e:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error applying amount precision (step size): {e}.{RESET}")
        return None

    # --- Final Validation after Precision ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size after precision adjustment is zero or negative ({final_size.normalize()}).{RESET}")
        return None

    # Re-check Min Amount (rounding down might violate it)
    if final_size < min_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is less than Min Amount {fmt_dec_log(min_amount)} after applying precision.{RESET}")
        # Attempt to bump up to the *exact* min amount IF it aligns with step size, or the next step otherwise.
        # Simpler: Just fail if rounding down violates min amount. Bumping up changes risk profile.
        # If min_amount itself is not a multiple of amount_step, the market definition is problematic.
        # Let's stick to failing here for safety.
        # Alternative: Could try setting size to min_amount_eff and re-applying step rounding *up*? Complex.
        return None

    # Re-check Max Amount (shouldn't be possible if rounding down, but check anyway)
    if final_size > max_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is greater than Max Amount {fmt_dec_log(max_amount)} after applying precision (unexpected!).{RESET}")
        return None

    # Re-check Cost Limits with the final precise size
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}): {final_cost.normalize()} {quote_currency}")
        # Check Min Cost again (rounding down amount might violate it)
        if final_cost < min_cost_eff:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} < Min Cost {fmt_dec_log(min_cost)} after precision adjustment.{RESET}")
            # Again, bumping size changes risk. Fail for safety.
            return None

        # Check Max Cost again (unlikely to be violated by rounding down amount, but check)
        elif final_cost > max_cost_eff:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} > Max Cost {fmt_dec_log(max_cost)} after precision adjustment (unexpected!).{RESET}")
            return None
    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        # Cost limits exist, but we couldn't estimate final cost
        lg.warning(f"Could not perform final cost check for {symbol} after precision adjustment. Order might fail if cost limits are violated.")

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    if final_cost:
         lg.info(f"    Estimated Final Cost: {final_cost.normalize()} {quote_currency}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size


def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """Cancels a specific order by its ID using ccxt.cancel_order.
    Includes retry logic and handles common errors like OrderNotFound and InvalidOrder gracefully.
    Passes necessary parameters for exchanges like Bybit V5.

    Args:
        exchange: The ccxt exchange instance.
        order_id: The ID of the order to cancel.
        symbol: The market symbol associated with the order (required by some exchanges).
        logger: The logger instance for messages.

    Returns:
        True if the order was successfully cancelled or confirmed not found/already closed, False otherwise.
    """
    lg = logger
    attempts = 0
    last_exception: Exception | None = None
    lg.info(f"Attempting to cancel order ID {order_id} for symbol {symbol}...")

    # Prepare parameters (e.g., category for Bybit V5)
    market_id = symbol  # Default
    params = {}
    is_bybit = 'bybit' in exchange.id.lower()
    if is_bybit:
        try:
            # We need market info to determine category, but don't have it passed in.
            # Attempt to fetch it here, or rely on defaults/config if needed.
            # For simplicity here, let's assume linear/spot based on symbol (crude).
            # A better approach would pass MarketInfo or fetch it.
            market = exchange.market(symbol)  # Fetch market info
            market_id = market['id']  # Use market_id if available
            if market.get('linear'): category = 'linear'
            elif market.get('inverse'): category = 'inverse'
            elif market.get('spot'): category = 'spot'
            else: category = 'linear'  # Default guess if type unclear
            params['category'] = category
            # Bybit cancelOrder might need symbol (market_id) in params too
            params['symbol'] = market_id
            lg.debug(f"Using Bybit params for cancelOrder: {params}")
        except Exception as e:
            lg.warning(f"Could not get market details to determine category/market_id for cancel ({symbol}): {e}. Proceeding without category/specific symbol param.")
            # Use original symbol if market lookup fails
            market_id = symbol

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1} for ID {order_id} ({symbol})...")
            # Use standard symbol in the main call, pass specifics in params
            exchange.cancel_order(order_id, symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully cancelled order {order_id} ({symbol}).{RESET}")
            return True

        except ccxt.OrderNotFound:
            # Order doesn't exist - could be already filled, cancelled, or wrong ID
            lg.warning(f"{NEON_YELLOW}Order ID {order_id} ({symbol}) not found on the exchange. Assuming cancellation is effectively complete.{RESET}")
            return True  # Treat as success for workflow purposes
        except ccxt.InvalidOrder as e:
             # E.g., order already filled/cancelled and API gives specific error
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Invalid order state for cancellation ({symbol}, ID: {order_id}): {e}. Assuming cancellation unnecessary/complete.{RESET}")
             return True  # Treat as success if it cannot be cancelled due to state
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 2  # Shorter wait for cancel might be okay
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded cancelling order {order_id} ({symbol}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue  # Don't count as standard attempt
        except ccxt.ExchangeError as e:
            # Other exchange errors during cancellation
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}. Retrying...{RESET}")
            # Check for potentially non-retryable cancel errors if needed
            # err_str = str(e).lower()
            # if "some specific non-retryable cancel message" in err_str:
            #     return False
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error cancelling order {order_id} ({symbol}): {e}. Cannot continue.{RESET}")
            return False  # Non-retryable
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as failure for safety
            return False

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to cancel order {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,  # "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"
    position_size: Decimal,
    market_info: MarketInfo,
    logger: logging.Logger,
    reduce_only: bool = False,
    params: dict | None = None  # Allow passing extra params for flexibility
) -> dict | None:
    """Places a market order based on the trade signal and calculated size using ccxt.create_order.
    Handles specifics for exchanges like Bybit V5 (category, reduceOnly, positionIdx).
    Includes retry logic and detailed error handling with hints.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        trade_signal: The action to take ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The calculated size for the order (always positive Decimal).
        market_info: The MarketInfo dictionary for the symbol.
        logger: The logger instance for messages.
        reduce_only: If True, set the reduceOnly flag (for closing/reducing positions).
        params: Optional dictionary of extra parameters for create_order (overrides defaults).

    Returns:
        The order result dictionary from ccxt if successful, otherwise None.
    """
    lg = logger

    # --- Determine Order Side ---
    side_map = {
        "BUY": "buy",         # Opening a long position
        "SELL": "sell",       # Opening a short position
        "EXIT_SHORT": "buy",  # Closing a short position (buy back base/contracts)
        "EXIT_LONG": "sell"   # Closing a long position (sell off base/contracts)
    }
    side = side_map.get(trade_signal.upper())
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided for {symbol}. Cannot determine order side.")
        return None

    # --- Validate Position Size ---
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'):
        lg.error(f"Invalid position size '{position_size}' provided for {symbol}. Must be a positive Decimal.")
        return None

    # --- Prepare Order Details ---
    order_type = 'market'  # Strategy currently uses market orders
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base_currency
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info.get('id')  # Use exchange-specific ID
    is_bybit = 'bybit' in exchange.id.lower()

    if not market_id:
         lg.error(f"Cannot place trade for {symbol}: Market ID missing in market_info.")
         return None

    # Convert Decimal size to float for ccxt (ensure it's not near zero float)
    try:
        # First, ensure size respects amount precision step before converting to float
        amount_step = market_info['amount_precision_step_decimal']
        if amount_step is None:
             raise ValueError("Amount precision step is missing in market info.")
        # Round the size according to step size before float conversion
        rounded_size = (position_size / amount_step).quantize(Decimal('1'), ROUND_DOWN) * amount_step
        if rounded_size <= 0:
            raise ValueError(f"Position size {position_size} rounded down to zero or negative based on step {amount_step}.")
        if rounded_size != position_size:
             lg.warning(f"Adjusting order size {position_size.normalize()} to {rounded_size.normalize()} due to precision step {amount_step.normalize()} before placing order.")
             position_size = rounded_size  # Use the rounded size

        amount_float = float(position_size)
        # Check for effective zero after float conversion ( paranoia check)
        if abs(amount_float) < 1e-15:
            raise ValueError("Position size converts to near-zero float.")
    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Failed to convert/validate position size {position_size.normalize()} for order placement ({symbol}): {e}")
        return None

    # Base order arguments for ccxt create_order
    order_args: dict[str, Any] = {
        'symbol': symbol,     # Use standard symbol for ccxt call
        'type': order_type,
        'side': side,
        'amount': amount_float,
        # 'price': None, # Not needed for market orders
    }

    # --- Exchange-Specific Parameters ---
    order_params: dict[str, Any] = {}
    if is_bybit and is_contract:
        try:
            # Determine category from market_info
            category = 'linear'  # Default
            if market_info.get('is_linear'): category = 'linear'
            elif market_info.get('is_inverse'): category = 'inverse'
            else: raise ValueError(f"Invalid contract category derived from market_info: {market_info.get('contract_type_str')}")

            order_params = {
                'category': category,
                'positionIdx': 0  # Assume one-way mode (0 index). Hedge mode would need 1 or 2.
                # Other potential Bybit params: timeInForce, postOnly, etc.
            }

            if reduce_only:
                order_params['reduceOnly'] = True
                # Bybit V5 often requires IOC/FOK for reduceOnly market orders to prevent accidental increase
                # IOC (Immediate Or Cancel) is generally safer for market reduceOnly
                order_params['timeInForce'] = 'IOC'
                lg.debug(f"Setting Bybit V5 reduceOnly=True and timeInForce='IOC' for {symbol}.")

        except Exception as e:
            lg.error(f"Failed to set Bybit V5 specific parameters for {symbol}: {e}. Proceeding with base params, order might fail.")
            order_params = {}  # Reset params if setup failed

    # Merge any externally provided params (allowing override of defaults)
    if params:
        lg.debug(f"Merging external parameters into order: {params}")
        order_params.update(params)

    # Add params to order_args if any exist
    if order_params:
        order_args['params'] = order_params

    # --- Log Order Intent ---
    lg.warning(f"{BRIGHT}===> Placing Trade Order ({action_desc}) <==={RESET}")
    lg.warning(f"  Symbol : {symbol} ({market_id})")
    lg.warning(f"  Type   : {order_type.upper()}")
    lg.warning(f"  Side   : {side.upper()} ({trade_signal})")
    lg.warning(f"  Size   : {position_size.normalize()} {size_unit} (Float: {amount_float})")  # Log both Dec and float
    if order_params:
        lg.warning(f"  Params : {order_params}")

    # --- Execute Order Placement with Retry ---
    attempts = 0
    last_exception: Exception | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1})...")
            # --- Place the Order ---
            order_result = exchange.create_order(**order_args)
            # --- --- --- --- --- ---

            # --- Log Success ---
            order_id = order_result.get('id', 'N/A')
            status = order_result.get('status', 'N/A')  # e.g., 'open', 'closed', 'canceled'
            # Average fill price and filled amount might not be immediately available for market orders
            avg_price_raw = order_result.get('average')
            filled_raw = order_result.get('filled')

            # Use safe conversion allowing zero (e.g., if order is open but not filled yet)
            avg_price_dec = _safe_market_decimal(avg_price_raw, 'order.average', allow_zero=True, allow_negative=False)
            filled_dec = _safe_market_decimal(filled_raw, 'order.filled', allow_zero=True, allow_negative=False)

            log_msg = (
                f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET}\n"
                f"  ID: {order_id}, Status: {status}"
            )
            # Only add price/fill info if available and makes sense
            if avg_price_dec is not None and avg_price_dec > 0:
                log_msg += f", Avg Fill Price: ~{avg_price_dec.normalize()}"
            if filled_dec is not None:
                log_msg += f", Filled Amount: {filled_dec.normalize()} {size_unit}"

            lg.info(log_msg)
            lg.debug(f"Full order result ({symbol}): {order_result}")
            return order_result  # Return the successful order details

        # --- Error Handling for create_order ---
        except ccxt.InsufficientFunds as e:
            # Non-retryable error
            last_exception = e
            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Insufficient Funds. Check balance and margin.{RESET}")
            lg.error(f"  Error details: {e}")
            return None
        except ccxt.InvalidOrder as e:
            # Order parameters are wrong (size, price, limits, etc.) - Non-retryable
            last_exception = e
            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Invalid Order Parameters.{RESET}")
            lg.error(f"  Error details: {e}")
            lg.error(f"  Order Arguments Sent: {order_args}")
            # Provide hints based on error message and market info
            err_lower = str(e).lower()
            # Use fmt_dec_log for cleaner logging of potentially None values
            min_a = fmt_dec_log(market_info.get('min_amount_decimal'))
            min_c = fmt_dec_log(market_info.get('min_cost_decimal'))
            amt_s = fmt_dec_log(market_info.get('amount_precision_step_decimal'))
            max_a = fmt_dec_log(market_info.get('max_amount_decimal'))
            max_c = fmt_dec_log(market_info.get('max_cost_decimal'))

            if any(s in err_lower for s in ["minimum order", "too small", "less than minimum", "min notional", "min value"]):
                lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) against Min Amount ({min_a}) and order cost against Min Cost ({min_c}).")
            elif any(s in err_lower for s in ["precision", "lot size", "step size", "size precision", "quantity precision"]):
                lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) precision against Amount Step ({amt_s}).")
            elif any(s in err_lower for s in ["exceed", "too large", "greater than maximum", "max value", "max order qty"]):
                lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) against Max Amount ({max_a}) and order cost against Max Cost ({max_c}).")
            elif "reduce only" in err_lower or "reduceonly" in err_lower:
                # Bybit: 110025 (Position is closed), 110031 (Reduce-only rule violated)
                lg.error(f"  >> Hint: Reduce-only order failed. Ensure there's an open position to reduce and the size ({position_size.normalize()}) doesn't increase the position.")
            elif "position size" in err_lower or "position idx" in err_lower:
                 lg.error("  >> Hint: Check if order size conflicts with existing position, leverage limits, or position mode (One-Way/Hedge).")

            return None
        except ccxt.ExchangeError as e:
            # General exchange errors - potentially retryable depending on the specific error
            last_exception = e
            # Try to extract error code for better logging/decision
            err_code = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code = match.group(2)
            else: err_code = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))  # Fallback

            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Exchange Error. (Code: {err_code}){RESET}")
            lg.error(f"  Error details: {e}")

            # Check for known fatal/non-retryable error codes or messages
            # (Examples, may need adjustment per exchange - focusing on Bybit V5)
            fatal_codes = [
                '10001',  # Bybit: Parameter error (often non-retryable if args are wrong)
                '10004',  # Bybit: Sign error (API key issue)
                '110007',  # Bybit: Batch orders count exceeds limit (if using batch)
                '110013',  # Bybit: Price precision issue (InvalidOrder should catch, but also here)
                '110014',  # Bybit: Size precision issue (InvalidOrder should catch)
                '110017',  # Bybit: Position idx not match position mode (config error)
                '110025',  # Bybit: Position not found/closed (relevant for reduceOnly, InvalidOrder might catch)
                '110031',  # Bybit: Reduce-only rule violated (InvalidOrder should catch)
                '110040',  # Bybit: Order qty exceeds risk limit (config/leverage issue)
                '30086',  # Bybit: Order cost exceeds risk limit (config/leverage issue)
                '3303001',  # Bybit SPOT: Invalid symbol
                '3303005',  # Bybit SPOT: Price/Qty precision issue (InvalidOrder should catch)
                '3400060',  # Bybit SPOT: Order amount exceeds balance (InsufficientFunds should catch)
                '3400088',  # Bybit: Leverage exceed max limit (config error)
                '110043',  # Bybit: Cannot set leverage with open position (shouldn't happen here, but listed)
            ]
            # Add messages that often indicate non-retryable issues
            fatal_msgs = [
                "invalid parameter", "precision", "exceed limit", "risk limit",
                "invalid symbol", "lot size", "api key", "authentication failed",
                "insufficient balance", "leverage exceed", "trigger liquidation",
                "account not unified", "unified account function", "position mode"
            ]
            is_fatal_code = err_code in fatal_codes
            is_fatal_message = any(msg in str(e).lower() for msg in fatal_msgs)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE order placement error for {symbol}. Check arguments and config.{RESET}")
                return None  # Non-retryable failure

            # If not identified as fatal, proceed to retry logic
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to ExchangeError placing order ({symbol}).{RESET}")
                return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to NetworkError placing order ({symbol}).{RESET}")
                return None
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 3  # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order ({symbol}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue  # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error placing order ({symbol}): {e}. Cannot continue.{RESET}")
            return None  # Non-retryable
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order ({symbol}): {e}{RESET}", exc_info=True)
            return None  # Treat unexpected errors as fatal

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


# --- Placeholder Functions (Require Full Implementation for Strategy) ---

def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    logger: logging.Logger,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_distance: Decimal | None = None,  # TSL distance/offset (interpretation depends on exchange)
    tsl_activation_price: Decimal | None = None   # Price at which TSL should activate (Bybit V5)
) -> bool:
    """Placeholder: Sets Stop Loss (SL), Take Profit (TP), and potentially Trailing Stop Loss (TSL)
    for an existing position using exchange-specific methods or parameters.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **

    Requires careful implementation based on the target exchange's API for modifying
    position attributes or placing conditional SL/TP/TSL orders.

    Bybit V5 Example (`POST /v5/position/set-trading-stop`):
    - Requires `category`, `symbol`, `positionIdx`.
    - Accepts `stopLoss`, `takeProfit`, `tpslMode` ('Full' or 'Partial').
    - Accepts `slTriggerBy`, `tpTriggerBy` ('MarkPrice', 'LastPrice', 'IndexPrice').
    - Accepts `trailingStop` (string: distance like "50" or price like "25000").
    - Accepts `activePrice` (string: activation price for TSL).
    - Accepts `slOrderType`, `tpOrderType` ('Market' or 'Limit').
    - If using Limit SL/TP, requires `slLimitPrice`, `tpLimitPrice`.

    Other exchanges might use `edit_order`, `create_order` with specific params, or dedicated methods.

    Args:
        exchange: ccxt exchange instance.
        symbol: Standard symbol (e.g., 'BTC/USDT').
        market_info: Market information (contains market_id, category, precision).
        position_info: Current position details (contains side, entryPrice, etc.).
        logger: Logger instance.
        stop_loss_price: Target SL price (Decimal). Will be formatted.
        take_profit_price: Target TP price (Decimal). Will be formatted.
        trailing_stop_distance: TSL distance/offset (Decimal). Interpretation varies; needs formatting.
        tsl_activation_price: Price to activate the TSL (Decimal, Bybit V5). Needs formatting.

    Returns:
        True if protection setting was attempted successfully (API call succeeded, check response code),
        False otherwise (invalid input, formatting error, API error).
    """
    lg = logger
    lg.warning(f"{NEON_YELLOW}Placeholder Function Called: _set_position_protection for {symbol}{RESET}")
    log_parts = []
    if stop_loss_price: log_parts.append(f"SL={stop_loss_price.normalize()}")
    if take_profit_price: log_parts.append(f"TP={take_profit_price.normalize()}")
    if trailing_stop_distance: log_parts.append(f"TSL Dist={trailing_stop_distance.normalize()}")
    if tsl_activation_price: log_parts.append(f"TSL Act={tsl_activation_price.normalize()}")
    if not log_parts:
        lg.debug("  No protection parameters provided to set.")
        return True  # Nothing to do, considered success

    lg.info(f"  Attempting to set: {', '.join(log_parts)}")

    # --- Exchange Specific Logic ---
    is_bybit = 'bybit' in exchange.id.lower()

    if is_bybit:
        # --- Bybit V5 Example ---
        try:
            market_id = market_info['id']
            category = 'linear'  # Default
            if market_info.get('is_linear'): category = 'linear'
            elif market_info.get('is_inverse'): category = 'inverse'
            else: raise ValueError("Invalid Bybit category")

            params: dict[str, Any] = {
                'category': category,
                'symbol': market_id,
                'positionIdx': 0,  # Assuming one-way mode
                'tpslMode': 'Full',  # Set SL/TP for the entire position
                # Default trigger to Mark Price, make configurable if needed
                'slTriggerBy': 'MarkPrice',
                'tpTriggerBy': 'MarkPrice',
                # Default order type to Market, make configurable if needed
                'slOrderType': 'Market',
                'tpOrderType': 'Market',
            }

            # Format and add parameters if provided
            if stop_loss_price:
                sl_str = _format_price(exchange, symbol, stop_loss_price)
                if sl_str: params['stopLoss'] = sl_str
                else: lg.error(f"Invalid SL price format for {symbol}: {stop_loss_price}"); return False
            if take_profit_price:
                 tp_str = _format_price(exchange, symbol, take_profit_price)
                 if tp_str: params['takeProfit'] = tp_str
                 else: lg.error(f"Invalid TP price format for {symbol}: {take_profit_price}"); return False

            # Trailing Stop logic needs careful handling
            if trailing_stop_distance:
                 # Bybit 'trailingStop' expects a string representing distance/offset in price points
                 # Ensure the distance is positive
                 if trailing_stop_distance <= 0:
                      lg.error(f"Invalid TSL distance for {symbol}: {trailing_stop_distance}. Must be positive."); return False
                 # Format the distance like a price step (crude assumption, might need different precision)
                 ts_dist_str = exchange.price_to_precision(symbol, float(trailing_stop_distance))
                 if ts_dist_str and float(ts_dist_str) > 0:  # Double check formatting didn't yield zero/negative
                      params['trailingStop'] = ts_dist_str
                 else:
                      lg.error(f"Invalid TSL distance format for {symbol}: {trailing_stop_distance} -> {ts_dist_str}"); return False

            if tsl_activation_price:
                 # Bybit 'activePrice' expects a string activation price
                 act_str = _format_price(exchange, symbol, tsl_activation_price)
                 if act_str: params['activePrice'] = act_str
                 else: lg.error(f"Invalid TSL activation price format for {symbol}: {tsl_activation_price}"); return False

            # Only call API if there's something to set
            if any(k in params for k in ['stopLoss', 'takeProfit', 'trailingStop']):
                 lg.info(f"Calling Bybit V5 set_trading_stop (placeholder) with params: {params}")

                 # --- !!! Actual API Call Needed Here !!! ---
                 # Example using ccxt's implicit methods (check if available/correct)
                 # response = exchange.set_trading_stop(params) # Hypothetical method
                 # Or using explicit private call:
                 # response = exchange.private_post_position_set_trading_stop(params)

                 # --- Placeholder Response Check ---
                 # Assume success for now
                 api_success = True  # Placeholder
                 api_ret_code = 0  # Placeholder
                 api_ret_msg = "OK (Placeholder)"  # Placeholder

                 # if response.get('retCode') == 0:
                 #     lg.info(f"Protection set successfully via API for {symbol}.")
                 #     return True
                 # else:
                 #     lg.error(f"API Error setting protection for {symbol}: {response.get('retMsg')} (Code: {response.get('retCode')})")
                 #     return False

                 if api_success:
                     lg.info(f"{NEON_GREEN}Protection set successfully via API (Placeholder) for {symbol}. Code={api_ret_code}, Msg={api_ret_msg}{RESET}")
                     return True
                 else:
                     lg.error(f"API Error setting protection (Placeholder) for {symbol}: {api_ret_msg} (Code: {api_ret_code})")
                     return False

            else:
                 lg.debug("No valid protection parameters formatted to send.")
                 return True  # Nothing to set, considered success

        except (ccxt.ExchangeError, ccxt.NetworkError, ValueError, TypeError) as e:
            lg.error(f"Error preparing or calling protection API for Bybit {symbol}: {e}", exc_info=True)
            return False
        except Exception as e:
             lg.error(f"Unexpected error setting protection for Bybit {symbol}: {e}", exc_info=True)
             return False

    else:
        # --- Fallback / Other Exchanges ---
        lg.error(f"Protection setting logic not implemented for exchange {exchange.id} in placeholder.")
        # Implementation would involve finding the correct ccxt method or parameters
        # E.g., using params in create_order for SL/TP if supported, or edit_position methods.
        return False  # Assume failure if not implemented


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    config: dict[str, Any],  # Pass config for TSL parameters
    logger: logging.Logger,
    take_profit_price: Decimal | None = None  # Can optionally set TP at the same time
) -> bool:
    """Placeholder: Sets up an initial Trailing Stop Loss (TSL) based on config parameters
    by calculating the required distance and activation price.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **

    This function calculates the necessary parameters and then calls `_set_position_protection`
    to interact with the exchange API. The interpretation of 'callback_rate' (percentage vs price points)
    and how the exchange API handles TSL distance ('trailingStop' param for Bybit) is critical.

    Args:
        exchange: ccxt exchange instance.
        symbol: Standard symbol.
        market_info: Market information.
        position_info: Current position details.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional TP price (Decimal) to set simultaneously.

    Returns:
        True if TSL setup was attempted successfully via `_set_position_protection`, False otherwise.
    """
    lg = logger
    lg.warning(f"{NEON_YELLOW}Placeholder Function Called: set_trailing_stop_loss for {symbol}{RESET}")

    prot_cfg = config.get('protection', {})
    # Get TSL parameters from config, convert to Decimal
    try:
        callback_rate = Decimal(str(prot_cfg.get('trailing_stop_callback_rate', 0.005)))  # e.g., 0.5%
        activation_perc = Decimal(str(prot_cfg.get('trailing_stop_activation_percentage', 0.003)))  # e.g., 0.3%
        if callback_rate <= 0:
             lg.error(f"Invalid TSL callback rate ({callback_rate}) in config. Must be > 0."); return False
        if activation_perc < 0:
             lg.error(f"Invalid TSL activation percentage ({activation_perc}) in config. Must be >= 0."); return False
    except (ValueError, InvalidOperation, TypeError) as e:
         lg.error(f"Invalid TSL parameter format in config: {e}"); return False

    # Get required position/market data
    entry_price = _safe_market_decimal(position_info.get('entryPrice'), f"{symbol} tsl.entry", False, False)
    # Use mark price if available for current price, fallback to ticker price
    current_price_raw = position_info.get('markPrice')
    source = "position mark price"
    if current_price_raw is None:
         current_price_raw = fetch_current_price_ccxt(exchange, symbol, lg)  # Fetch ticker if mark price missing
         source = "ticker price"

    current_price = _safe_market_decimal(current_price_raw, f"{symbol} tsl.current ({source})", False, False)
    side = position_info.get('side')
    price_tick = market_info.get('price_precision_step_decimal')

    if not entry_price or not current_price or side not in ['long', 'short'] or not price_tick:
        lg.error(f"Cannot set TSL for {symbol}: Missing required data (Entry: {entry_price}, Current: {current_price}, Side: {side}, Tick: {price_tick}).")
        return False

    # --- TSL Calculation Logic ---
    # 1. Calculate Activation Price:
    #    - Long: entry_price * (1 + activation_perc)
    #    - Short: entry_price * (1 - activation_perc)
    #    Round to price tick precision (towards safety - further from entry)
    activation_price_calc: Decimal
    if side == 'long':
        act_raw = entry_price * (Decimal('1') + activation_perc)
        activation_price_calc = (act_raw / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick
    else:  # short
        act_raw = entry_price * (Decimal('1') - activation_perc)
        activation_price_calc = (act_raw / price_tick).quantize(Decimal('1'), ROUND_DOWN) * price_tick

    # Ensure activation price didn't round to or past entry
    if side == 'long' and activation_price_calc <= entry_price: activation_price_calc = entry_price + price_tick
    if side == 'short' and activation_price_calc >= entry_price: activation_price_calc = entry_price - price_tick
    # Ensure positive
    if activation_price_calc <= 0:
        lg.error(f"Calculated TSL activation price ({activation_price_calc}) is non-positive for {symbol}. Cannot set TSL."); return False

    # 2. Calculate TSL Distance (Callback):
    #    Interpretation depends heavily on the exchange API.
    #    Bybit V5 'trailingStop' expects the distance in price points (e.g., "50" for $50 distance).
    #    Let's assume callback_rate is a percentage of the *entry price* to determine the fixed price distance.
    #    Alternative: could be % of current price, or absolute price points from config.
    #    Using entry price provides a consistent distance based on initial risk setup.
    tsl_distance_calc = (entry_price * callback_rate).quantize(price_tick, ROUND_UP)  # Round distance up slightly

    # Ensure distance is at least one price tick
    if tsl_distance_calc < price_tick:
         tsl_distance_calc = price_tick
         lg.warning(f"Calculated TSL distance was less than price tick for {symbol}. Using minimum tick distance: {price_tick.normalize()}")

    lg.info(f"Calculated TSL params for {symbol}: ActivationPrice={activation_price_calc.normalize()}, Distance={tsl_distance_calc.normalize()} (based on entry price %)")

    # --- Call the actual protection setting function ---
    lg.info(f"Calling _set_position_protection to apply TSL (and optional TP) for {symbol}")
    # Pass the calculated Decimal values. _set_position_protection will handle formatting.
    success = _set_position_protection(
        exchange, symbol, market_info, position_info, lg,
        stop_loss_price=None,  # Not setting regular SL here, TSL replaces/manages it
        take_profit_price=take_profit_price,  # Pass through optional TP
        trailing_stop_distance=tsl_distance_calc,  # Pass calculated distance
        tsl_activation_price=activation_price_calc  # Pass calculated activation price
    )

    if success:
        lg.info(f"Trailing stop loss setup/update initiated successfully for {symbol}.")
        # Note: Internal state marker `position_state['tsl_activated']` should be set
        # in the calling function (`manage_existing_position`) after this returns True.
    else:
        lg.error(f"Failed to set up/update trailing stop loss for {symbol} via _set_position_protection.")

    return success  # Return result of the underlying call


class VolumaticOBStrategy:
    """Placeholder: Encapsulates the Volumatic Trend + Order Block strategy logic.
    Calculates indicators, identifies trends, finds order blocks, and stores analysis results.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **

    The core logic for calculating the Volumatic Trend (VT) based on the described
    EMA/SWMA of volume-weighted price, ATR bands, and trend determination needs
    to be implemented. Similarly, the Pivot High/Low detection and Order Block
    identification (based on pivots and source 'Wicks'/'Body'), violation checks,
    and extension logic need to be fully coded.
    """
    def __init__(self, config: dict[str, Any], market_info: MarketInfo, logger: logging.Logger) -> None:
        self.lg = logger
        self.symbol = market_info['symbol']
        self.market_info = market_info
        self.params = config.get('strategy_params', {})
        self.protection_params = config.get('protection', {})
        self.price_tick = market_info.get('price_precision_step_decimal') or Decimal('0.00000001')  # Miniscule default

        # --- Extract Params (with defaults and type safety) ---
        try:
            self.vt_len = int(self.params.get('vt_length', DEFAULT_VT_LENGTH))
            self.vt_atr_period = int(self.params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD))
            self.vt_vol_ema_len = int(self.params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH))
            self.vt_atr_mult = Decimal(str(self.params.get('vt_atr_multiplier', DEFAULT_VT_ATR_MULTIPLIER)))
            # self.vt_step_atr_mult = Decimal(str(self.params.get('vt_step_atr_multiplier', DEFAULT_VT_STEP_ATR_MULTIPLIER))) # If needed
            self.ob_source = str(self.params.get('ob_source', DEFAULT_OB_SOURCE))  # "Wicks" or "Body"
            self.ph_left = int(self.params.get('ph_left', DEFAULT_PH_LEFT))
            self.ph_right = int(self.params.get('ph_right', DEFAULT_PH_RIGHT))
            self.pl_left = int(self.params.get('pl_left', DEFAULT_PL_LEFT))
            self.pl_right = int(self.params.get('pl_right', DEFAULT_PL_RIGHT))
            self.ob_extend = bool(self.params.get('ob_extend', DEFAULT_OB_EXTEND))
            self.ob_max_boxes = int(self.params.get('ob_max_boxes', DEFAULT_OB_MAX_BOXES))
            # Validate basic parameter ranges
            if not (self.vt_len > 0 and self.vt_atr_period > 0 and self.vt_vol_ema_len > 0 and
                    self.vt_atr_mult > 0 and self.ph_left > 0 and self.ph_right > 0 and
                    self.pl_left > 0 and self.pl_right > 0 and self.ob_max_boxes > 0):
                raise ValueError("One or more strategy parameters are out of valid range (e.g., <= 0).")
            if self.ob_source not in ["Wicks", "Body"]:
                 raise ValueError(f"Invalid ob_source parameter: {self.ob_source}")

        except (ValueError, TypeError) as e:
             self.lg.error(f"Invalid strategy parameter format or value for {self.symbol}: {e}")
             # Optionally re-raise or set defaults to stop initialization
             raise ValueError(f"Strategy Initialization Failed for {self.symbol}: Invalid parameters.") from e

        # Estimate minimum data length needed (ensure sufficient history for longest lookback)
        self.min_data_len = max(
            self.vt_len * 2,  # Need more for initial EMA/MA stability
            self.vt_atr_period + 1,  # ATR needs N+1 periods
            self.vt_vol_ema_len * 2,  # EMA needs buffer
            self.ph_left + self.ph_right + 1,  # Pivots need lookback+lookforward+current
            self.pl_left + self.pl_right + 1
        ) + 50  # Add a generous buffer

        self.lg.info(f"Strategy Engine initialized for {self.symbol} with min data length ~{self.min_data_len}")
        self.lg.debug(f"  Params: VT Len={self.vt_len}, ATR Period={self.vt_atr_period}, "
                      f"Vol EMA={self.vt_vol_ema_len}, ATR Mult={self.vt_atr_mult}, "
                      f"OB Src={self.ob_source}, Pivots L/R=({self.ph_left}/{self.ph_right}, {self.pl_left}/{self.pl_right}), "
                      f"Extend={self.ob_extend}, Max Boxes={self.ob_max_boxes}")

        # State for tracking order blocks across updates (persistent between `update` calls)
        self._active_bull_boxes: list[OrderBlock] = []
        self._active_bear_boxes: list[OrderBlock] = []

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Placeholder: Calculates Smoothed Weighted Moving Average (SWMA) via EMA.
        Actual implementation depends on the precise definition of SWMA used.
        Using pandas_ta.swma if available, otherwise simple EMA as fallback placeholder.
        """
        self.lg.debug(f"Placeholder _ema_swma called for length {length}")
        if series.empty or length <= 0 or series.isnull().all():
             return pd.Series(dtype=np.float64, index=series.index)  # Return empty/NaN series matching index
        try:
            # Try using pandas_ta implementation if it matches the desired SWMA
            if hasattr(ta, 'swma'):
                 # Ensure input is float64 for pandas_ta
                 result = ta.swma(series.astype(np.float64), length=length)
                 return result.astype(np.float64)  # Ensure output is float64
            else:
                 # Fallback to simple EMA if ta.swma doesn't exist
                 self.lg.warning("pandas_ta.swma not found, using simple EMA as placeholder for _ema_swma.")
                 result = ta.ema(series.astype(np.float64), length=length)
                 return result.astype(np.float64)
        except Exception as e:
             self.lg.error(f"Error calculating SWMA/EMA (placeholder) for length {length}: {e}", exc_info=True)
             return pd.Series(np.nan, index=series.index, dtype=np.float64)  # Return NaN series on error

    def _find_pivots(self, series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
        """Placeholder: Finds pivot high or low points.
        A pivot high is a point higher than `left` bars before and `right` bars after.
        A pivot low is a point lower than `left` bars before and `right` bars after.
        Returns a boolean Series indicating pivot points.
        """
        self.lg.debug(f"Placeholder _find_pivots called (Left:{left}, Right:{right}, High:{is_high})")
        if series.empty or left < 0 or right < 0 or series.isnull().all():
            return pd.Series(False, index=series.index)

        pivots = pd.Series(False, index=series.index)
        # Basic rolling window comparison (inefficient placeholder)
        # A proper implementation would use more optimized methods (e.g., comparing shifted series)
        # This placeholder likely won't work correctly for identifying true pivots.
        for i in range(left, len(series) - right):
            window = series.iloc[i - left : i + right + 1]
            if window.isnull().any(): continue  # Skip windows with NaN

            is_pivot = False
            current_val = series.iloc[i]
            if is_high:
                 # Check if current is strictly the highest in the window
                 if current_val >= window.max() and (window == current_val).sum() == 1:
                      is_pivot = True
            else:  # is_low
                 # Check if current is strictly the lowest in the window
                 if current_val <= window.min() and (window == current_val).sum() == 1:
                     is_pivot = True

            pivots.iloc[i] = is_pivot

        self.lg.warning("Pivot detection logic is a basic placeholder and may not be accurate.")
        return pivots

    def update(self, df: pd.DataFrame) -> StrategyAnalysisResults:
        """Placeholder: Processes the input DataFrame to calculate indicators and identify strategy elements.
        ** This function needs the full implementation of VT and OB logic. **.

        Args:
            df: The OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume' as Decimal, indexed by UTC timestamp.

        Returns:
            A StrategyAnalysisResults dictionary containing the processed DataFrame and key strategy outputs.
            Returns results with None/empty values if calculations fail or data is insufficient.
        """
        self.lg.debug(f"Running strategy update for {self.symbol} with DataFrame length {len(df)}")

        # --- Default Result Structure ---
        default_result = StrategyAnalysisResults(
            dataframe=df, last_close=df['close'].iloc[-1] if not df.empty else Decimal('0'),
            current_trend_up=None, trend_just_changed=False,
            active_bull_boxes=self._active_bull_boxes,  # Use current state
            active_bear_boxes=self._active_bear_boxes,
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if len(df) < self.min_data_len:
            self.lg.warning(f"DataFrame length ({len(df)}) is less than minimum required ({self.min_data_len}) for {self.symbol}. Strategy results may be inaccurate.")
            # Return default results but update last_close if possible
            if not df.empty: default_result['last_close'] = df['close'].iloc[-1]
            return default_result

        # --- Perform Calculations (Placeholders) ---
        df_analysis = df.copy()
        try:
            # Convert Decimal columns to float for pandas_ta compatibility
            # Handle potential NaNs before conversion
            high_f = df_analysis['high'].apply(lambda x: float(x) if pd.notna(x) else np.nan)
            low_f = df_analysis['low'].apply(lambda x: float(x) if pd.notna(x) else np.nan)
            close_f = df_analysis['close'].apply(lambda x: float(x) if pd.notna(x) else np.nan)
            volume_f = df_analysis['volume'].apply(lambda x: float(x) if pd.notna(x) else np.nan)

            # 1. Calculate Indicators (ATR, Vol EMA, VT Bands)
            # ATR using pandas_ta
            atr_series_f = ta.atr(high_f, low_f, close_f, length=self.vt_atr_period)
            df_analysis['atr'] = atr_series_f.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            current_atr = df_analysis['atr'].iloc[-1] if not df_analysis.empty and pd.notna(df_analysis['atr'].iloc[-1]) else None

            # Volume Weighted Price (Placeholder: using typical price)
            typical_price = (df_analysis['high'] + df_analysis['low'] + df_analysis['close']) / 3
            # Vol EMA (Placeholder - using EMA of typical price, NOT volume-weighted)
            # *** Needs actual Vol-Weighted Price & EMA/SWMA implementation ***
            vol_ema_placeholder = self._ema_swma(typical_price, self.vt_vol_ema_len)
            df_analysis['vol_ema'] = vol_ema_placeholder.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

            # VT Bands (Placeholder - using simple +/- ATR from close)
            # *** Needs actual VT center line (e.g., EMA/SWMA of vol-weighted price) and band calculation ***
            mid_band_placeholder = ta.ema(close_f, length=self.vt_len)  # Simple EMA of close
            df_analysis['vt_upper'] = mid_band_placeholder + atr_series_f * float(self.vt_atr_mult)
            df_analysis['vt_lower'] = mid_band_placeholder - atr_series_f * float(self.vt_atr_mult)
            # Convert back to Decimal
            df_analysis['vt_upper'] = df_analysis['vt_upper'].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            df_analysis['vt_lower'] = df_analysis['vt_lower'].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

            upper_band = df_analysis['vt_upper'].iloc[-1] if not df_analysis.empty and pd.notna(df_analysis['vt_upper'].iloc[-1]) else None
            lower_band = df_analysis['vt_lower'].iloc[-1] if not df_analysis.empty and pd.notna(df_analysis['vt_lower'].iloc[-1]) else None

            # Trend (Placeholder - based on close vs previous close)
            # *** Needs actual VT trend logic (e.g., price vs bands, slope) ***
            df_analysis['trend_up'] = df_analysis['close'] > df_analysis['close'].shift(1)
            current_trend_up = bool(df_analysis['trend_up'].iloc[-1]) if not df_analysis.empty and pd.notna(df_analysis['trend_up'].iloc[-1]) else None
            trend_just_changed = False
            if len(df_analysis) > 1 and pd.notna(df_analysis['trend_up'].iloc[-1]) and pd.notna(df_analysis['trend_up'].iloc[-2]):
                 trend_just_changed = df_analysis['trend_up'].iloc[-1] != df_analysis['trend_up'].iloc[-2]

            # Volume Normalization (Placeholder - simple min-max scaling)
            # *** Needs actual Volumatic calculation if different ***
            vol_min = volume_f.min()
            vol_max = volume_f.max()
            if not volume_f.empty and vol_max > vol_min and pd.notna(vol_min) and pd.notna(vol_max):
                 df_analysis['vol_norm'] = (volume_f - vol_min) / (vol_max - vol_min) * 100
            else:
                 df_analysis['vol_norm'] = 0.0
            vol_norm_int = int(df_analysis['vol_norm'].iloc[-1]) if not df_analysis.empty and pd.notna(df_analysis['vol_norm'].iloc[-1]) else None

            # 2. Find Pivots (Placeholder)
            # *** Needs actual Pivot implementation ***
            df_analysis['ph'] = self._find_pivots(df_analysis['high'], self.ph_left, self.ph_right, is_high=True)
            df_analysis['pl'] = self._find_pivots(df_analysis['low'], self.pl_left, self.pl_right, is_high=False)

            # 3. Identify Order Blocks (Placeholder)
            # *** Needs actual OB identification, violation, extension logic ***
            # This logic should:
            # - Iterate backwards from recent data or use pivot signals.
            # - Define OB based on pivot candle's wick/body (self.ob_source) and price levels.
            # - Create new OrderBlock dicts.
            # - Add new OBs to self._active_bull_boxes / self._active_bear_boxes (prepend).
            # - Check for violation of existing boxes by current price action (close crossing boundary).
            # - Update violated boxes (active=False, violated=True, violation_ts).
            # - Extend non-violated boxes if self.ob_extend is True (update extended_to_ts).
            # - Remove oldest boxes if len > self.ob_max_boxes.
            # - Filter out violated boxes before returning active lists.
            self.lg.debug("Order Block identification, violation, and extension logic is a placeholder.")
            # For placeholder, just return the current state (no updates)
            active_bull_boxes = [box for box in self._active_bull_boxes if box['active'] and not box['violated']]
            active_bear_boxes = [box for box in self._active_bear_boxes if box['active'] and not box['violated']]

        except Exception as e:
            self.lg.error(f"Error during strategy calculation for {self.symbol}: {e}", exc_info=True)
            # Return default results on error, ensuring last_close is updated if possible
            if not df.empty: default_result['last_close'] = df['close'].iloc[-1]
            # Keep existing OBs in the default result
            default_result['active_bull_boxes'] = [box for box in self._active_bull_boxes if box['active'] and not box['violated']]
            default_result['active_bear_boxes'] = [box for box in self._active_bear_boxes if box['active'] and not box['violated']]
            return default_result

        # --- Prepare and Return Results ---
        last_close = df_analysis['close'].iloc[-1] if not df_analysis.empty else Decimal('0')
        self.lg.info(f"Strategy update complete for {self.symbol}. Last Close: {last_close.normalize()}, TrendUp (Placeholder): {current_trend_up}, ATR: {current_atr.normalize() if current_atr else 'N/A'}")
        self.lg.debug(f"  Active Bull OBs (Placeholder): {len(active_bull_boxes)}, Active Bear OBs (Placeholder): {len(active_bear_boxes)}")

        return StrategyAnalysisResults(
            dataframe=df_analysis,  # Return the DataFrame with added indicator columns
            last_close=last_close,
            current_trend_up=current_trend_up,
            trend_just_changed=trend_just_changed,
            active_bull_boxes=active_bull_boxes,  # Use the potentially updated lists
            active_bear_boxes=active_bear_boxes,
            vol_norm_int=vol_norm_int,
            atr=current_atr,
            upper_band=upper_band,  # Placeholder value
            lower_band=lower_band  # Placeholder value
        )


class SignalGenerator:
    """Placeholder: Generates trading signals ("BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT")
    based on the results from the VolumaticOBStrategy analysis and current position state.

     ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **

    The core logic needs to implement the entry and exit rules based on the strategy:
    - Entry Conditions:
        - Trend alignment (e.g., VT trend is up for BUY).
        - Price interaction with an Order Block (e.g., price enters a Bullish OB).
        - Proximity check using `ob_entry_proximity_factor`.
        - Volume confirmation (optional, using `vol_norm_int`?).
    - Exit Conditions:
        - Price hitting an opposite Order Block (using `ob_exit_proximity_factor`).
        - Trend reversal signal from VT.
        - (Stop Loss / Take Profit are handled by exchange orders, not direct exit signals here).
    - Calculates initial SL/TP based on entry price, ATR, and potentially OB levels.
    """
    def __init__(self, config: dict[str, Any], market_info: MarketInfo, logger: logging.Logger) -> None:
        self.lg = logger
        self.symbol = market_info['symbol']
        self.market_info = market_info
        self.strategy_params = config.get('strategy_params', {})
        self.protection_params = config.get('protection', {})
        # Ensure price tick is valid Decimal
        price_tick_raw = market_info.get('price_precision_step_decimal')
        if price_tick_raw is None or price_tick_raw <= 0:
            self.lg.warning(f"Invalid price tick ({price_tick_raw}) in market_info for {self.symbol}. Using small default.")
            self.price_tick = Decimal('0.00000001')
        else:
             self.price_tick = price_tick_raw

        # --- Extract Params with Type Safety ---
        try:
            self.entry_prox_factor = Decimal(str(self.strategy_params.get('ob_entry_proximity_factor', 1.005)))
            self.exit_prox_factor = Decimal(str(self.strategy_params.get('ob_exit_proximity_factor', 1.001)))
            self.sl_atr_mult = Decimal(str(self.protection_params.get('initial_stop_loss_atr_multiple', 1.8)))
            self.tp_atr_mult = Decimal(str(self.protection_params.get('initial_take_profit_atr_multiple', 0.7)))
            # Validate basic ranges
            if not (self.entry_prox_factor >= 1 and self.exit_prox_factor >= 1 and
                    self.sl_atr_mult > 0 and self.tp_atr_mult >= 0):
                raise ValueError("One or more signal generator parameters are out of valid range.")
        except (ValueError, TypeError, InvalidOperation) as e:
             self.lg.error(f"Invalid signal generator parameter format or value for {self.symbol}: {e}")
             raise ValueError(f"Signal Generator Initialization Failed for {self.symbol}: Invalid parameters.") from e

        self.lg.info(f"Signal Generator initialized for {self.symbol}")
        self.lg.debug(f"  Params: Entry Prox Factor={self.entry_prox_factor}, Exit Prox Factor={self.exit_prox_factor}, "
                      f"SL ATR Mult={self.sl_atr_mult}, TP ATR Mult={self.tp_atr_mult}")

    def _calculate_initial_sl_tp(self, entry_price: Decimal, side: str, atr: Decimal) -> tuple[Decimal | None, Decimal | None]:
        """Placeholder: Calculates initial SL and TP based on entry price, side, and ATR multiplier.
        Applies price tick precision rounding away from the entry price.

        Args:
            entry_price: The intended entry price (Decimal).
            side: 'long' or 'short'.
            atr: The current ATR value (Decimal).

        Returns:
            Tuple (initial_sl, initial_tp), both Optional[Decimal]. Returns None if inputs are invalid.
        """
        if not isinstance(entry_price, Decimal) or entry_price <= 0:
             self.lg.warning(f"Cannot calculate SL/TP for {self.symbol}: Invalid entry price ({entry_price}).")
             return None, None
        if not isinstance(atr, Decimal) or atr <= 0:
             self.lg.warning(f"Cannot calculate SL/TP for {self.symbol}: Invalid ATR ({atr}).")
             return None, None
        if side not in ['long', 'short']:
             self.lg.warning(f"Cannot calculate SL/TP for {self.symbol}: Invalid side ({side}).")
             return None, None

        sl_distance = atr * self.sl_atr_mult
        tp_distance = atr * self.tp_atr_mult if self.tp_atr_mult > 0 else None

        initial_sl_raw: Decimal | None = None
        initial_tp_raw: Decimal | None = None

        if side == 'long':
            initial_sl_raw = entry_price - sl_distance
            if tp_distance:
                initial_tp_raw = entry_price + tp_distance
        else:  # short
            initial_sl_raw = entry_price + sl_distance
            if tp_distance:
                initial_tp_raw = entry_price - tp_distance

        # --- Apply Price Precision and Safety Checks ---
        final_sl: Decimal | None = None
        final_tp: Decimal | None = None

        # Stop Loss: Round *away* from entry, ensure it didn't cross entry, ensure positive
        if initial_sl_raw is not None:
             rounding = ROUND_DOWN if side == 'long' else ROUND_UP  # Round further away
             sl_rounded = (initial_sl_raw / self.price_tick).quantize(Decimal('1'), rounding) * self.price_tick

             # Check if rounding made SL equal or cross entry price
             if side == 'long' and sl_rounded >= entry_price:
                 sl_rounded = entry_price - self.price_tick  # Move one tick away if crossed
             elif side == 'short' and sl_rounded <= entry_price:
                 sl_rounded = entry_price + self.price_tick  # Move one tick away if crossed

             # Final check: ensure SL is positive
             if sl_rounded > 0:
                 final_sl = sl_rounded
             else:
                 self.lg.warning(f"Calculated initial SL ({sl_rounded.normalize()}) is non-positive after rounding for {self.symbol}. Setting SL to None.")

        # Take Profit: Round *away* from entry, ensure it didn't cross entry, ensure positive
        if initial_tp_raw is not None:
             rounding = ROUND_UP if side == 'long' else ROUND_DOWN  # Round further away
             tp_rounded = (initial_tp_raw / self.price_tick).quantize(Decimal('1'), rounding) * self.price_tick

             # Check if rounding made TP equal or cross entry price
             if side == 'long' and tp_rounded <= entry_price:
                 tp_rounded = entry_price + self.price_tick  # Move one tick away if crossed
             elif side == 'short' and tp_rounded >= entry_price:
                 tp_rounded = entry_price - self.price_tick  # Move one tick away if crossed

             # Final check: ensure TP is positive
             if tp_rounded > 0:
                 final_tp = tp_rounded
             else:
                 self.lg.warning(f"Calculated initial TP ({tp_rounded.normalize()}) is non-positive after rounding for {self.symbol}. Setting TP to None.")

        self.lg.debug(f"Calculated SL/TP for {self.symbol} ({side}): Entry={entry_price.normalize()}, ATR={atr.normalize()} -> "
                      f"SL={final_sl.normalize() if final_sl else 'None'}, TP={final_tp.normalize() if final_tp else 'None'}")
        return final_sl, final_tp

    def generate_signal(
        self,
        analysis: StrategyAnalysisResults,
        current_position: PositionInfo | None,
        symbol: str  # Explicit symbol for clarity
    ) -> SignalResult:
        """Placeholder: Analyzes strategy results and current position to generate a trading signal.
        ** This function needs the full implementation of the strategy's entry/exit rules. **.

        Args:
            analysis: The results from the VolumaticOBStrategy update.
            current_position: The current open position details (or None).
            symbol: The symbol being analyzed.

        Returns:
            A SignalResult dictionary containing the signal ("BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"),
            reasoning, and potentially calculated initial SL/TP for new entries.
        """
        self.lg.debug(f"Generating signal for {symbol}...")

        # --- Extract Key Analysis Results ---
        trend_up = analysis.get('current_trend_up')
        analysis.get('trend_just_changed', False)
        atr = analysis.get('atr')
        last_close = analysis.get('last_close')
        bull_boxes = analysis.get('active_bull_boxes', [])
        bear_boxes = analysis.get('active_bear_boxes', [])

        # --- Default Signal ---
        signal = "HOLD"
        reason = "No conditions met."
        initial_sl = None
        initial_tp = None

        # --- Strategy Logic (Placeholder Examples) ---
        # ** Replace with actual Volumatic Trend + Order Block logic **

        # --- Exit Logic First (if position exists) ---
        if current_position:
            pos_side = current_position.get('side')
            _safe_market_decimal(current_position.get('entryPrice'), "pos.entry", False, False)

            # Example Exit 1: Trend Reversal
            if pos_side == 'long' and trend_up is False:
                 signal = "EXIT_LONG"
                 reason = "Placeholder: Trend changed to down."
            elif pos_side == 'short' and trend_up is True:
                 signal = "EXIT_SHORT"
                 reason = "Placeholder: Trend changed to up."

            # Example Exit 2: Price hits opposite OB (Placeholder - needs proximity check)
            elif pos_side == 'long' and bear_boxes and last_close:
                 # Check if close is near/inside the nearest bear box
                 nearest_bear_box = min(bear_boxes, key=lambda b: b['bottom'])  # Example: nearest bottom
                 # Placeholder proximity check
                 if last_close >= nearest_bear_box['bottom'] * (Decimal('1') - (self.exit_prox_factor - 1)):  # Example check
                      signal = "EXIT_LONG"
                      reason = f"Placeholder: Price near Bear OB {nearest_bear_box['id']} @ {nearest_bear_box['bottom']}-{nearest_bear_box['top']}."
            elif pos_side == 'short' and bull_boxes and last_close:
                 # Check if close is near/inside the nearest bull box
                 nearest_bull_box = max(bull_boxes, key=lambda b: b['top'])  # Example: nearest top
                 # Placeholder proximity check
                 if last_close <= nearest_bull_box['top'] * self.exit_prox_factor:  # Example check
                      signal = "EXIT_SHORT"
                      reason = f"Placeholder: Price near Bull OB {nearest_bull_box['id']} @ {nearest_bull_box['bottom']}-{nearest_bull_box['top']}."

            # If exit condition met, clear SL/TP calculation
            if signal.startswith("EXIT"):
                initial_sl = None
                initial_tp = None
            else:
                 reason = f"Holding {pos_side} position. Trend aligned or no exit condition met."

        # --- Entry Logic (if no position exists) ---
        elif current_position is None:
            # Example Entry: Trend aligned and price touches relevant OB (Placeholder)
            if trend_up is True and bull_boxes and last_close and atr:
                # Check if close is near/inside the highest bull box
                highest_bull_box = max(bull_boxes, key=lambda b: b['top'])
                # Placeholder proximity check
                if last_close <= highest_bull_box['top'] * self.entry_prox_factor:  # Example check
                    signal = "BUY"
                    reason = f"Placeholder: Trend UP, Price near Bull OB {highest_bull_box['id']} @ {highest_bull_box['bottom']}-{highest_bull_box['top']}."
                    # Calculate SL/TP based on entry (last_close approx) and ATR
                    initial_sl, initial_tp = self._calculate_initial_sl_tp(last_close, 'long', atr)
                    if initial_sl is None:  # Cannot enter without SL
                         signal = "HOLD"; reason += " (Failed to calculate valid SL)."
            elif trend_up is False and bear_boxes and last_close and atr:
                 # Check if close is near/inside the lowest bear box
                 lowest_bear_box = min(bear_boxes, key=lambda b: b['bottom'])
                 # Placeholder proximity check
                 if last_close >= lowest_bear_box['bottom'] * (Decimal('1') - (self.entry_prox_factor - 1)):  # Example check
                     signal = "SELL"
                     reason = f"Placeholder: Trend DOWN, Price near Bear OB {lowest_bear_box['id']} @ {lowest_bear_box['bottom']}-{lowest_bear_box['top']}."
                     # Calculate SL/TP based on entry (last_close approx) and ATR
                     initial_sl, initial_tp = self._calculate_initial_sl_tp(last_close, 'short', atr)
                     if initial_sl is None:  # Cannot enter without SL
                          signal = "HOLD"; reason += " (Failed to calculate valid SL)."
            else:
                 reason = "Trend unclear or no suitable OB interaction for entry."

        # --- Final Logging ---
        self.lg.info(f"Signal for {symbol}: {BRIGHT}{signal}{RESET} ({reason})")
        if initial_sl or initial_tp:
            self.lg.info(f"  Calculated Entry Protections: SL={initial_sl.normalize() if initial_sl else 'N/A'}, TP={initial_tp.normalize() if initial_tp else 'N/A'}")

        return SignalResult(
            signal=signal,
            reason=reason,
            initial_sl=initial_sl,
            initial_tp=initial_tp
        )

# --- Trading Workflow Functions ---


def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: dict[str, Any],
    logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy,
    signal_generator: SignalGenerator,
    market_info: MarketInfo,
    position_states: dict[str, dict[str, bool]]  # Shared state for BE/TSL activation per symbol
) -> None:
    """Orchestrates the analysis and trading logic for a single symbol within a loop cycle.
    Fetches data, runs analysis, gets position, manages existing position (BE/TSL),
    generates signal, and executes trade actions.

    Args:
        exchange: ccxt exchange instance.
        symbol: Symbol to process.
        config: Bot configuration.
        logger: Logger for this symbol.
        strategy_engine: Initialized strategy engine for this symbol.
        signal_generator: Initialized signal generator for this symbol.
        market_info: Market information for this symbol.
        position_states: Dictionary holding persistent state (BE/TSL flags) for symbols.
    """
    lg = logger
    lg.info(f"--- Starting Analysis & Trading Cycle for: {symbol} ---")

    try:
        # 1. Fetch Data
        timeframe_cfg = config.get("interval", "5")
        ccxt_tf = CCXT_INTERVAL_MAP.get(timeframe_cfg)
        fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
        if not ccxt_tf:
             lg.error(f"Invalid interval '{timeframe_cfg}' in config. Skipping {symbol}.")
             return
        df = fetch_klines_ccxt(exchange, symbol, ccxt_tf, fetch_limit, lg)
        if df.empty:
             lg.warning(f"No kline data fetched for {symbol}. Skipping analysis.")
             return
        if len(df) < strategy_engine.min_data_len:
            lg.warning(f"Insufficient kline data for {symbol} (got {len(df)}, need ~{strategy_engine.min_data_len}). Strategy results may be inaccurate.")
            # Allow proceeding but strategy should handle insufficient data internally

        # 2. Run Strategy Analysis
        analysis_results = strategy_engine.update(df)
        # Strategy update should return default/empty results if it fails internally
        if analysis_results is None:  # Should not happen if strategy returns default
             lg.error(f"Strategy analysis returned None for {symbol}. Skipping trade logic.")
             return
        if analysis_results['last_close'] <= 0:  # Basic sanity check on analysis output
             lg.error(f"Strategy analysis resulted in invalid last_close ({analysis_results['last_close']}) for {symbol}. Skipping.")
             return

        # 3. Check Current Position & Sync State
        current_position = get_open_position(exchange, symbol, market_info, lg)

        # Get or initialize symbol-specific state dictionary
        position_state = position_states.setdefault(symbol, {'be_activated': False, 'tsl_activated': False})

        # Sync internal state if position exists and protections are detected on exchange
        if current_position:
             # If exchange reports TSL active (non-zero distance/activation), ensure our state reflects that
             exchange_tsl_active = bool(current_position.get('trailingStopLoss')) and bool(current_position.get('tslActivationPrice'))
             if exchange_tsl_active and not position_state['tsl_activated']:
                  lg.info(f"Detected active TSL on exchange for {symbol} (TSL Dist/Act Price set). Syncing internal state.")
                  position_state['tsl_activated'] = True
             elif not exchange_tsl_active and position_state['tsl_activated']:
                  # If bot thinks TSL is active, but exchange doesn't show it (e.g., manual cancel)
                  lg.warning(f"Internal state indicates TSL active for {symbol}, but not detected on exchange. Resetting internal TSL flag.")
                  position_state['tsl_activated'] = False

             # BE state is harder to detect externally, rely on internal flag mostly.
             # If SL is at/near entry, could infer BE, but complex. Keep internal flag primary.
             # Reset BE flag if position side changes (should happen on close/re-entry)
        else:
             # If no position, ensure state flags are reset
             if position_state['be_activated'] or position_state['tsl_activated']:
                  lg.info(f"No position found for {symbol}. Resetting internal BE/TSL state flags.")
                  position_state['be_activated'] = False
                  position_state['tsl_activated'] = False

        # 4. Manage Existing Position (SL updates, BE, TSL activation)
        if current_position:
            # Pass the potentially updated position_state to the management function
            manage_existing_position(exchange, symbol, market_info, current_position, analysis_results, position_state, lg)
            # Note: We might want to re-fetch position info *after* management if SL/TP/TSL were potentially modified
            # to get the absolute latest state before signal generation. For now, assume management updates `position_state`
            # and signal generation can work with the slightly older `current_position` info + updated `position_state`.
            # current_position = get_open_position(exchange, symbol, market_info, lg) # Optional re-fetch

        # 5. Generate Signal
        signal_info = signal_generator.generate_signal(analysis_results, current_position, symbol)

        # 6. Execute Trade Action
        # Pass the potentially updated position_state to the execution function
        execute_trade_action(exchange, symbol, market_info, current_position, signal_info, analysis_results, position_state, lg)

    except ccxt.AuthenticationError as e:
        # Propagate auth errors immediately to stop the bot
        lg.critical(f"{NEON_RED}Authentication Error during {symbol} processing: {e}. Stopping bot.{RESET}")
        raise e  # Re-raise to be caught by main loop
    except Exception as e:
        lg.error(f"{NEON_RED}!! Unhandled error during analysis/trading cycle for {symbol}: {e} !!{RESET}", exc_info=True)
        # Continue to the next symbol in the main loop

    finally:
        lg.info(f"--- Finished Analysis & Trading Cycle for: {symbol} ---")


def manage_existing_position(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    analysis_results: StrategyAnalysisResults,
    position_state: dict[str, bool],  # Internal state for BE/TSL (mutable)
    logger: logging.Logger
) -> None:
    """Placeholder: Manages an existing open position based on configuration and analysis results.
    Handles logic for activating Break-Even (BE) and Trailing Stop Loss (TSL).
    Updates the `position_state` dictionary directly if activation is successful.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION REQUIRED **

    - Requires full implementation of BE and TSL trigger condition checks.
    - Calls `_set_position_protection` or `set_trailing_stop_loss` for API interaction.
    - Updates `position_state` flags upon successful activation via API.
    """
    lg = logger
    lg.debug(f"Managing existing {position_info.get('side', 'N/A')} position for {symbol}...")

    config = CONFIG  # Access global config
    prot_cfg = config.get('protection', {})
    enable_be = prot_cfg.get('enable_break_even', False)
    enable_tsl = prot_cfg.get('enable_trailing_stop', False)
    price_tick = market_info.get('price_precision_step_decimal')

    # Check if essential data is present
    if not price_tick or price_tick <= 0:
         lg.error(f"Cannot manage position for {symbol}: Invalid price tick ({price_tick}).")
         return

    # --- Break-Even Logic ---
    # Only attempt BE if enabled, not already activated internally, and TSL is not already active (BE usually takes precedence)
    if enable_be and not position_state.get('be_activated', False) and not position_state.get('tsl_activated', False):
        try:
            be_trigger_atr_mult = Decimal(str(prot_cfg.get('break_even_trigger_atr_multiple', 1.0)))
            be_offset_ticks = int(prot_cfg.get('break_even_offset_ticks', 2))

            atr = analysis_results.get('atr')
            entry_price = _safe_market_decimal(position_info.get('entryPrice'), f"{symbol} be.entry", False, False)
            last_close = analysis_results.get('last_close')
            side = position_info.get('side')

            # Check if all required data is valid
            if atr and entry_price and last_close and side and be_trigger_atr_mult > 0 and be_offset_ticks >= 0:
                trigger_distance = atr * be_trigger_atr_mult
                be_target_price: Decimal | None = None
                be_trigger_price: Decimal | None = None
                should_trigger_be = False

                # Calculate BE offset amount
                offset_amount = price_tick * be_offset_ticks

                if side == 'long':
                     be_trigger_price = entry_price + trigger_distance
                     if last_close >= be_trigger_price:
                          should_trigger_be = True
                          # Calculate BE target price (entry + offset), rounded UP to nearest tick
                          be_target_raw = entry_price + offset_amount
                          be_target_price = (be_target_raw / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick
                          # Ensure BE target is still above entry after rounding
                          if be_target_price <= entry_price: be_target_price = entry_price + price_tick

                elif side == 'short':
                     be_trigger_price = entry_price - trigger_distance
                     if last_close <= be_trigger_price:
                          should_trigger_be = True
                          # Calculate BE target price (entry - offset), rounded DOWN to nearest tick
                          be_target_raw = entry_price - offset_amount
                          be_target_price = (be_target_raw / price_tick).quantize(Decimal('1'), ROUND_DOWN) * price_tick
                          # Ensure BE target is still below entry after rounding
                          if be_target_price >= entry_price: be_target_price = entry_price - price_tick

                # Trigger BE if conditions met and target price is valid
                if should_trigger_be and be_target_price and be_target_price > 0:
                    lg.info(f"{NEON_YELLOW}Break-Even Triggered for {symbol} ({side.upper()})! Price {last_close.normalize()} vs Trigger {be_trigger_price.normalize()}{RESET}")
                    lg.info(f"  Moving SL to BE Price: {be_target_price.normalize()} (Entry {entry_price.normalize()}, Offset {be_offset_ticks} ticks)")
                    # --- Call API to move SL ---
                    success = _set_position_protection(exchange, symbol, market_info, position_info, lg, stop_loss_price=be_target_price)
                    if success:
                           position_state['be_activated'] = True  # Update shared state directly
                           lg.info(f"BE Stop Loss set successfully for {symbol}.")
                    else:
                           lg.error(f"Failed to set BE Stop Loss for {symbol}.")
                # else: # Log if trigger not met
                    # lg.debug(f"BE condition not met for {symbol}: Price {last_close.normalize()} vs Trigger {be_trigger_price.normalize() if be_trigger_price else 'N/A'}")

            else:
                 lg.debug(f"BE Check Skipped ({symbol}): Missing data (ATR={atr}, Entry={entry_price}, Close={last_close}, Side={side}) or invalid config (Mult={be_trigger_atr_mult}, Offset={be_offset_ticks}).")

        except (ValueError, TypeError, InvalidOperation) as e:
             lg.error(f"Error during Break-Even calculation for {symbol}: {e}")

    # --- Trailing Stop Loss Activation Logic ---
    # Only activate TSL if enabled, not already active internally, and BE hasn't been activated
    # (If BE activates, it usually overrides TSL activation until TSL potentially takes over later)
    if enable_tsl and not position_state.get('tsl_activated', False) and not position_state.get('be_activated', False):
        try:
            activation_perc = Decimal(str(prot_cfg.get('trailing_stop_activation_percentage', 0.003)))
            entry_price = _safe_market_decimal(position_info.get('entryPrice'), f"{symbol} tsl_act.entry", False, False)
            last_close = analysis_results.get('last_close')
            side = position_info.get('side')

            # Check required data
            if entry_price and last_close and side and activation_perc >= 0:  # Allow 0% activation (activate immediately)
                 activation_threshold_met = False
                 activation_trigger_price: Decimal | None = None

                 if side == 'long':
                     activation_trigger_price = entry_price * (Decimal('1') + activation_perc)
                     if last_close >= activation_trigger_price:
                         activation_threshold_met = True
                 elif side == 'short':
                     activation_trigger_price = entry_price * (Decimal('1') - activation_perc)
                     if last_close <= activation_trigger_price:
                         activation_threshold_met = True

                 if activation_threshold_met:
                     lg.info(f"{NEON_YELLOW}Trailing Stop Activation Threshold Met for {symbol} ({side.upper()})! Price {last_close.normalize()} vs Trigger {activation_trigger_price.normalize()}. Activating TSL...{RESET}")
                     # Call the function to set up the TSL on the exchange
                     # This calculates distance/activation and calls _set_position_protection
                     success = set_trailing_stop_loss(exchange, symbol, market_info, position_info, config, lg)
                     if success:
                         position_state['tsl_activated'] = True  # Update shared state directly
                         lg.info(f"TSL activation successful for {symbol}.")
                     else:
                         lg.error(f"Failed to activate TSL for {symbol}.")
                 # else: # Log if trigger not met
                     # lg.debug(f"TSL Activation condition not met for {symbol}: Price {last_close.normalize()} vs Trigger {activation_trigger_price.normalize() if activation_trigger_price else 'N/A'}")

            else:
                 lg.debug(f"TSL Activation Check Skipped ({symbol}): Missing data (Entry={entry_price}, Close={last_close}, Side={side}) or invalid config (Activation%={activation_perc}).")

        except (ValueError, TypeError, InvalidOperation) as e:
             lg.error(f"Error during Trailing Stop activation check for {symbol}: {e}")

    # --- Potential Future Logic: Dynamic SL/TSL Adjustment ---
    # Could add logic here to trail the SL based on indicators (e.g., VT lower/upper band)
    # or adjust TSL parameters if the exchange allows modification.
    # lg.debug(f"Dynamic SL/TSL adjustment logic not implemented in placeholder.")

    lg.debug(f"Finished managing position for {symbol}. Current State: BE Active={position_state.get('be_activated', False)}, TSL Active={position_state.get('tsl_activated', False)}")


def execute_trade_action(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    current_position: PositionInfo | None,
    signal_info: SignalResult,
    analysis_results: StrategyAnalysisResults,
    position_state: dict[str, bool],  # Pass state to reset on close (mutable)
    logger: logging.Logger
) -> None:
    """Placeholder: Executes the trade action based on the generated signal.
    Handles opening new positions (calculating size, setting leverage, placing order, setting initial SL/TP)
    and closing existing positions. Manages position state flags.

    ** THIS IS A PLACEHOLDER - Requires review of max position logic and error handling **

    Args:
        exchange: ccxt exchange instance.
        symbol: Symbol to trade.
        market_info: Market information.
        current_position: Current position details (or None).
        signal_info: Generated signal result.
        analysis_results: Strategy analysis results (for entry price approx).
        position_state: Dictionary holding persistent state (BE/TSL flags) for this symbol.
        logger: Logger instance.
    """
    lg = logger
    signal = signal_info['signal']
    reason = signal_info['reason']
    config = CONFIG  # Access global config
    enable_trading = config.get('enable_trading', False)
    max_positions_global = config.get('max_concurrent_positions', 1)  # Global limit

    lg.debug(f"Executing trade action '{signal}' for {symbol}. Reason: {reason}")

    # --- Exit Logic ---
    if signal == "EXIT_LONG" and current_position and current_position.get('side') == 'long':
        lg.warning(f"{NEON_YELLOW}>>> Closing LONG position for {symbol} due to: {reason}{RESET}")
        if not enable_trading:
             lg.warning(f"Trading disabled. Skipping close order for {symbol}.")
             return

        # Get size from position info (should be positive for long)
        close_size = current_position.get('size_decimal')
        if close_size is None or close_size <= 0:
             lg.error(f"Cannot close {symbol} LONG: Invalid or non-positive position size ({close_size}) found.")
             return

        # Place market sell order to close
        order_result = place_trade(exchange, symbol, signal, close_size, market_info, lg, reduce_only=True)
        if order_result:
             # Reset internal state for this symbol after successful close attempt
             # (API success doesn't guarantee fill, but we reset state optimistically)
             lg.info(f"Resetting internal state for {symbol} after placing close order.")
             position_state['be_activated'] = False
             position_state['tsl_activated'] = False
        else:
             lg.error(f"Failed to place closing order for {symbol} LONG. State not reset.")

    elif signal == "EXIT_SHORT" and current_position and current_position.get('side') == 'short':
        lg.warning(f"{NEON_YELLOW}>>> Closing SHORT position for {symbol} due to: {reason}{RESET}")
        if not enable_trading:
             lg.warning(f"Trading disabled. Skipping close order for {symbol}.")
             return

        # Get absolute size from position info (size_decimal is negative for short)
        close_size = abs(current_position.get('size_decimal', Decimal(0)))
        if close_size <= 0:
             lg.error(f"Cannot close {symbol} SHORT: Invalid or zero position size ({current_position.get('size_decimal')}) found.")
             return

        # Place market buy order to close
        order_result = place_trade(exchange, symbol, signal, close_size, market_info, lg, reduce_only=True)
        if order_result:
             # Reset internal state
             lg.info(f"Resetting internal state for {symbol} after placing close order.")
             position_state['be_activated'] = False
             position_state['tsl_activated'] = False
        else:
             lg.error(f"Failed to place closing order for {symbol} SHORT. State not reset.")

    # --- Entry Logic ---
    elif signal in ["BUY", "SELL"] and current_position is None:
        lg.warning(f"{NEON_GREEN}>>> Received {signal} signal for {symbol}. Attempting to open position... ({reason}){RESET}")
        if not enable_trading:
             lg.warning(f"Trading disabled. Skipping entry order for {symbol}.")
             return

        # --- Check Max Concurrent Positions ---
        # This requires tracking positions across *all* symbols processed by the bot.
        # This state needs to be managed externally or passed into this function.
        # Placeholder: Assume we need a function `get_current_active_positions_count()`
        # current_active_positions = get_current_active_positions_count(exchange, valid_symbols, lg) # Needs implementation
        current_active_positions = 0  # !!! Placeholder - Replace with actual count !!!
        lg.warning("Max concurrent position check is using a PLACEHOLDER count of 0.")  # Warn about placeholder

        if current_active_positions >= max_positions_global:
             lg.warning(f"Skipping entry for {symbol}: Max concurrent positions ({max_positions_global}) reached globally.")
             return

        # --- Calculate Position Size ---
        balance_currency = config.get("quote_currency", QUOTE_CURRENCY)
        # Fetch balance *just before* sizing
        balance = fetch_balance(exchange, balance_currency, lg)
        risk_per_trade = config.get("risk_per_trade", 0.01)
        initial_sl = signal_info.get('initial_sl')
        # Use last close as approximate entry price for sizing calculation
        entry_price_approx = analysis_results.get('last_close')

        if balance is None:
             lg.error(f"Cannot calculate position size for {symbol}: Failed to fetch balance for {balance_currency}.")
             return
        if initial_sl is None:
             lg.error(f"Cannot calculate position size for {symbol}: Initial SL was not provided by signal generator.")
             return
        if entry_price_approx is None or entry_price_approx <= 0:
             lg.error(f"Cannot calculate position size for {symbol}: Invalid approximate entry price ({entry_price_approx}) from analysis.")
             return

        pos_size = calculate_position_size(balance, risk_per_trade, initial_sl, entry_price_approx, market_info, exchange, lg)

        if pos_size is None or pos_size <= 0:
            lg.error(f"Position size calculation failed or resulted in zero/negative size for {symbol}. Cannot place entry order.")
            return

        # --- Set Leverage (only if needed and applicable) ---
        config.get("leverage", 0)
