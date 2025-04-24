# pyrmethus_volumatic_bot.py - Merged & Minified v1.4.1 (Fixed global scope error)
# Enhanced Version: Improved readability, structure, comments, and robustness.

# --- Standard Library Imports ---
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
import signal
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# --- Timezone Handling ---
try:
    # Use the standard library's zoneinfo if available (Python 3.9+)
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # Fallback for older Python versions or if tzdata is not installed
    print("Warning: 'zoneinfo' module not found. Falling back to UTC. "
          "Ensure Python 3.9+ and install 'tzdata' (`pip install tzdata`).")
    # Basic UTC fallback implementation mimicking ZoneInfo interface
    class ZoneInfo: # type: ignore [no-redef]
        """Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface."""
        def __init__(self, key: str):
            self._key = "UTC" # Store the key, though we always use UTC
        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            """Attach UTC timezone info to a datetime object."""
            return dt.replace(tzinfo=timezone.utc) if dt else None
        def fromutc(self, dt: datetime) -> datetime:
            """Convert a UTC datetime to this timezone (which is UTC)."""
            return dt.replace(tzinfo=timezone.utc)
        def utcoffset(self, dt: Optional[datetime]) -> timedelta:
            """Return the UTC offset (always zero for UTC)."""
            return timedelta(0)
        def dst(self, dt: Optional[datetime]) -> timedelta:
            """Return the DST offset (always zero for UTC)."""
            return timedelta(0)
        def tzname(self, dt: Optional[datetime]) -> str:
            """Return the timezone name (always 'UTC')."""
            return "UTC"
    class ZoneInfoNotFoundError(Exception): # type: ignore [no-redef]
        """Exception raised when a timezone is not found (fallback definition)."""
        pass

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
import pandas_ta as ta  # Technical Analysis library
import requests         # For HTTP requests (used by ccxt)
import ccxt             # Crypto Exchange Trading Library

# Colorama for colored console output
from colorama import Fore, Style, init as colorama_init

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
    print(f"{Fore.RED}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET environment variables are missing. "
          f"Ensure they are set in your system or in a .env file. Exiting.{Style.RESET_ALL}")
    sys.exit(1)

# --- Configuration & Logging ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default if not set in .env
TIMEZONE_STR = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    print(f"{Fore.RED}Timezone '{TIMEZONE_STR}' not found using 'zoneinfo'. Falling back to UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"
except Exception as tz_err:
    print(f"{Fore.RED}An error occurred initializing timezone '{TIMEZONE_STR}': {tz_err}. Falling back to UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

# --- API & Timing Constants ---
MAX_API_RETRIES: int = 3           # Max number of retries for failed API calls
RETRY_DELAY_SECONDS: int = 5       # Initial delay between retries
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Delay after placing order to confirm position status
LOOP_DELAY_SECONDS: int = 15       # Base delay between main loop cycles
BYBIT_API_KLINE_LIMIT: int = 1000  # Max klines per Bybit API request

# --- Data & Strategy Constants ---
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Supported intervals in config
CCXT_INTERVAL_MAP: Dict[str, str] = { # Map config intervals to CCXT timeframes
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT: int = 750     # Default number of klines to fetch if not in config
MAX_DF_LEN: int = 2000             # Maximum length of DataFrame to keep in memory

# Default Volumatic Trend (VT) parameters
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0

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
except OSError as e:
    print(f"{NEON_RED}{BRIGHT}FATAL: Could not create log directory '{LOG_DIRECTORY}': {e}. Exiting.{RESET}")
    sys.exit(1)

# --- Global State ---
_shutdown_requested: bool = False # Flag for graceful shutdown

# --- Type Definitions ---
class OrderBlock(TypedDict):
    """Represents an identified Order Block."""
    id: str                 # Unique identifier (e.g., "BULL_1678886400000")
    type: str               # "BULL" or "BEAR"
    timestamp: pd.Timestamp # Timestamp of the candle defining the block
    top: Decimal            # Top price of the block
    bottom: Decimal         # Bottom price of the block
    active: bool            # Is the block currently considered active?
    violated: bool          # Has the block been violated?
    violation_ts: Optional[pd.Timestamp] # Timestamp of violation
    extended_to_ts: Optional[pd.Timestamp] # Timestamp the box is currently extended to

class StrategyAnalysisResults(TypedDict):
    """Results from the strategy analysis on a DataFrame."""
    dataframe: pd.DataFrame # The analyzed DataFrame with indicators
    last_close: Decimal     # Last closing price
    current_trend_up: Optional[bool] # Current trend direction (True=Up, False=Down, None=Undetermined)
    trend_just_changed: bool # Did the trend change on the last candle?
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    vol_norm_int: Optional[int] # Normalized volume indicator value (if used)
    atr: Optional[Decimal]    # Current ATR value
    upper_band: Optional[Decimal] # Upper band value (from VT or other indicator)
    lower_band: Optional[Decimal] # Lower band value (from VT or other indicator)

class MarketInfo(TypedDict):
    """Standardized market information from ccxt."""
    id: str                 # Exchange-specific market ID (e.g., 'BTCUSDT')
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    base: str               # Base currency (e.g., 'BTC')
    quote: str              # Quote currency (e.g., 'USDT')
    settle: Optional[str]   # Settle currency (usually for futures)
    baseId: str             # Exchange-specific base ID
    quoteId: str            # Exchange-specific quote ID
    settleId: Optional[str] # Exchange-specific settle ID
    type: str               # Market type ('spot', 'swap', 'future', etc.)
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool            # Is the market currently active/tradeable?
    contract: bool          # Is it a contract (swap, future)?
    linear: Optional[bool]  # Linear contract?
    inverse: Optional[bool] # Inverse contract?
    quanto: Optional[bool]  # Quanto contract?
    taker: float            # Taker fee rate
    maker: float            # Maker fee rate
    contractSize: Optional[Any] # Size of one contract
    expiry: Optional[int]
    expiryDatetime: Optional[str]
    strike: Optional[float]
    optionType: Optional[str]
    precision: Dict[str, Any] # Price and amount precision rules
    limits: Dict[str, Any]    # Order size and cost limits
    info: Dict[str, Any]      # Raw market data from the exchange
    # --- Added/Derived Fields ---
    is_contract: bool         # Convenience flag: True if swap or future
    is_linear: bool           # Convenience flag: True if linear contract
    is_inverse: bool          # Convenience flag: True if inverse contract
    contract_type_str: str    # "Spot", "Linear", "Inverse", or "Unknown"
    min_amount_decimal: Optional[Decimal] # Minimum order size as Decimal
    max_amount_decimal: Optional[Decimal] # Maximum order size as Decimal
    min_cost_decimal: Optional[Decimal]   # Minimum order cost as Decimal
    max_cost_decimal: Optional[Decimal]   # Maximum order cost as Decimal
    amount_precision_step_decimal: Optional[Decimal] # Smallest amount increment as Decimal
    price_precision_step_decimal: Optional[Decimal]  # Smallest price increment as Decimal
    contract_size_decimal: Decimal # Contract size as Decimal (defaults to 1)

class PositionInfo(TypedDict):
    """Standardized position information from ccxt."""
    id: Optional[str]       # Position ID (exchange-specific)
    symbol: str             # Standardized symbol (e.g., 'BTC/USDT')
    timestamp: Optional[int] # Position creation/update timestamp (ms)
    datetime: Optional[str]  # ISO 8601 datetime string
    contracts: Optional[float] # Number of contracts (may be deprecated, use size_decimal)
    contractSize: Optional[Any] # Size of one contract for this position
    side: Optional[str]      # 'long' or 'short'
    notional: Optional[Any]  # Position value in quote currency
    leverage: Optional[Any]  # Position leverage
    unrealizedPnl: Optional[Any] # Unrealized profit/loss
    realizedPnl: Optional[Any]   # Realized profit/loss
    collateral: Optional[Any]    # Margin used for the position
    entryPrice: Optional[Any]    # Average entry price
    markPrice: Optional[Any]     # Current mark price
    liquidationPrice: Optional[Any] # Estimated liquidation price
    marginMode: Optional[str]    # 'isolated' or 'cross'
    hedged: Optional[bool]       # Is hedging enabled for this position?
    maintenanceMargin: Optional[Any]
    maintenanceMarginPercentage: Optional[float]
    initialMargin: Optional[Any]
    initialMarginPercentage: Optional[float]
    marginRatio: Optional[float]
    lastUpdateTimestamp: Optional[int]
    info: Dict[str, Any]         # Raw position data from the exchange
    # --- Added/Derived Fields ---
    size_decimal: Decimal        # Position size as Decimal (positive for long, negative for short)
    stopLossPrice: Optional[str] # Current stop loss price (as string from exchange)
    takeProfitPrice: Optional[str] # Current take profit price (as string from exchange)
    trailingStopLoss: Optional[str] # Current trailing stop distance/price (as string)
    tslActivationPrice: Optional[str] # Trailing stop activation price (as string)
    # --- Bot State Tracking ---
    be_activated: bool           # Has the break-even logic been triggered for this position?
    tsl_activated: bool          # Has the trailing stop loss been activated for this position?

class SignalResult(TypedDict):
    """Result of the signal generation process."""
    signal: str              # "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
    reason: str              # Explanation for the signal
    initial_sl: Optional[Decimal] # Calculated initial stop loss price for a new entry
    initial_tp: Optional[Decimal] # Calculated initial take profit price for a new entry

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
        except Exception as e:
            # Avoid crashing the application if redaction fails
            print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
        return msg

class NeonConsoleFormatter(SensitiveFormatter):
    """Formats log messages for the console with timestamps and colors."""
    _level_colors = {
        logging.DEBUG: NEON_CYAN + DIM,
        logging.INFO: NEON_BLUE,
        logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED,
        logging.CRITICAL: NEON_RED + BRIGHT
    }
    _tz = TIMEZONE # Use the globally configured timezone

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record with level-specific colors and local timestamp."""
        level_color = self._level_colors.get(record.levelno, NEON_BLUE) # Default to blue
        log_fmt = (
            f"{NEON_BLUE}%(asctime)s{RESET} - "
            f"{level_color}%(levelname)-8s{RESET} - "
            f"{NEON_PURPLE}[%(name)s]{RESET} - "
            f"%(message)s"
        )
        # Create a formatter for *this record* to use the local timezone
        # Note: Using a lambda for converter makes it dynamic per record
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore[assignment]

        # Use the parent SensitiveFormatter's format method for redaction,
        # but with the locally-formatted message string.
        original_message = record.getMessage()
        record.message = formatter.formatMessage(record) # Temporarily set formatted msg
        formatted_redacted_message = super().format(record)
        record.message = original_message # Restore original message for other handlers
        return formatted_redacted_message

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger instance with file and console handlers.

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

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels; handlers filter later

    # --- File Handler ---
    try:
        # Rotate log files (10MB each, keep 5 backups)
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Use UTC time for file logs for consistency across servers/timezones
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter.converter = time.gmtime # type: ignore[attr-defined] # Use UTC
        fh.setFormatter(file_formatter)
        fh.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(fh)
    except Exception as e:
        # Log setup errors should be visible on the console
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler ---
    try:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(NeonConsoleFormatter("%(message)s")) # Format defined within the class

        # Set console log level from environment variable or default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False # Prevent messages going to the root logger
    return logger

# --- Initial Logger ---
# Used for setup messages before symbol-specific loggers are created
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR} ({TIMEZONE})")
# Add a note about requirements
init_logger.debug("Ensure required packages are installed: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (optional but recommended)")

# --- Configuration Loading & Validation ---
def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures all keys from default_config exist in config.
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
        # elif type(updated_config.get(key)) is not type(default_value):
        #     init_logger.warning(f"Config Warning: Type mismatch for '{full_key_path}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Using default.")
        #     updated_config[key] = default_value
        #     changed = True
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
    Validates a numeric value within a config dictionary level.
    Corrects type and clamps to range if necessary, using the default value as a fallback.

    Args:
        cfg_level: The current dictionary level of the loaded config.
        default_level: The corresponding dictionary level in the default config.
        leaf_key: The specific key to validate within the current level.
        key_path: The full dot-notation path to the key (for logging).
        min_val: The minimum allowed value.
        max_val: The maximum allowed value.
        is_strict_min: If True, value must be strictly greater than min_val.
        is_int: If True, the value should be an integer.
        allow_zero: If True, zero is allowed even if outside the min/max range.

    Returns:
        True if the value was corrected or replaced with the default, False otherwise.
    """
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key)
    corrected = False
    final_val = original_val

    try:
        # Attempt conversion to Decimal for robust comparison
        num_val = Decimal(str(original_val))
        min_dec = Decimal(str(min_val))
        max_dec = Decimal(str(max_val))

        # Range Check
        min_check_passed = num_val > min_dec if is_strict_min else num_val >= min_dec
        range_check_passed = min_check_passed and num_val <= max_dec
        zero_allowed = allow_zero and num_val.is_zero()

        if not range_check_passed and not zero_allowed:
            raise ValueError("Value outside allowed range.")

        # Type Check and Correction
        target_type = int if is_int else float
        converted_val = target_type(num_val) # Convert validated Decimal to target type

        needs_type_correction = False
        if isinstance(original_val, bool): # Explicitly disallow bools for numeric fields
             raise TypeError("Boolean found where numeric expected.")
        elif is_int and not isinstance(original_val, int):
            needs_type_correction = True
        elif not is_int: # Expecting float
            if isinstance(original_val, int):
                # Allow int if it converts cleanly to float
                converted_val = float(original_val)
                needs_type_correction = True # Technically corrected type
            elif not isinstance(original_val, float):
                 needs_type_correction = True
            # Check if float representation is significantly different (handles precision issues)
            elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9:
                 needs_type_correction = True

        if needs_type_correction:
            init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
            final_val = converted_val
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
        final_val = default_val
        corrected = True

    # Update the config dictionary if a correction was made
    if corrected:
        cfg_level[leaf_key] = final_val

    return corrected

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file, creates a default one if missing,
    validates parameters, and ensures all necessary keys are present.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        The loaded and validated configuration dictionary.
    """
    global QUOTE_CURRENCY # Allow updating the global QUOTE_CURRENCY constant
    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    default_config = {
        "trading_pairs": ["BTC/USDT"],
        "interval": "5", # Default timeframe
        "retry_delay": RETRY_DELAY_SECONDS, # Use constant default
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Use constant default
        "orderbook_limit": 25, # Limit for order book fetching (if used later)
        "enable_trading": False, # Safety default: trading disabled
        "use_sandbox": True,    # Safety default: use sandbox environment
        "risk_per_trade": 0.01, # Risk 1% of capital per trade
        "leverage": 20,         # Default leverage (if applicable)
        "max_concurrent_positions": 1, # Max simultaneous positions
        "quote_currency": "USDT", # Default quote currency
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Use constant default
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Use constant default

        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER),
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER),
            "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005, # How close price needs to be to OB for entry (e.g., 1.005 = 0.5% range)
            "ob_exit_proximity_factor": 1.001, # How close price needs to be to opposite OB for exit
        },

        "protection": {
            "enable_trailing_stop": True,
            "trailing_stop_callback_rate": 0.005, # 0.5% trailing stop distance (example)
            "trailing_stop_activation_percentage": 0.003, # Activate TSL when price moves 0.3% in profit (example)
            "enable_break_even": True,
            "break_even_trigger_atr_multiple": 1.0, # Move SL to BE when price moves 1 * ATR
            "break_even_offset_ticks": 2, # Offset BE stop by 2 price ticks
            "initial_stop_loss_atr_multiple": 1.8, # Initial SL distance based on ATR
            "initial_take_profit_atr_multiple": 0.7, # Initial TP distance based on ATR (0 means no TP)
        }
    }

    config_needs_saving: bool = False
    loaded_config: Dict[str, Any] = {}

    # --- File Existence Check ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default configuration.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully created default config file: {filepath}{RESET}")
            # Update global QUOTE_CURRENCY from the default we just wrote
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config
        except IOError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error creating config file '{filepath}': {e}. Using internal defaults.{RESET}")
            # Still update global QUOTE_CURRENCY from internal default
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return default in-memory config

    # --- File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
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
        except IOError as e_create:
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
            config_needs_saving = True # Mark for saving later

        # --- Validation Logic ---
        init_logger.debug("# Validating configuration parameters...")

        # Helper function to navigate nested dicts for validation
        def get_nested_levels(cfg: Dict, path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
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
                init_logger.error(f"Config validation error: Cannot access path '{path}'.")
                return None, None, None

        # Define validation function that uses the helper
        def validate_numeric(cfg: Dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False):
            nonlocal config_needs_saving
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None:
                # Error already logged by helper
                return # Cannot validate

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


        # --- Apply Validations ---
        # Top Level
        pairs = updated_config.get("trading_pairs", [])
        if not isinstance(pairs, list) or not all(isinstance(s, str) and s and '/' in s for s in pairs):
            init_logger.warning(f"Invalid 'trading_pairs' format. Using default: {default_config['trading_pairs']}.")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True

        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"Invalid 'interval' value '{updated_config.get('interval')}'. Must be one of {VALID_INTERVALS}. Using default '{default_config['interval']}'.")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True

        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('0.5'), is_strict_min=True) # Risk must be > 0
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Allow 0 for spot/no leverage
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        validate_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True) # Example range

        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
            init_logger.warning(f"Invalid 'quote_currency'. Using default '{default_config['quote_currency']}'.")
            updated_config["quote_currency"] = default_config["quote_currency"]
            config_needs_saving = True
        # Update the global QUOTE_CURRENCY immediately after validation, as logger might use it
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")


        if not isinstance(updated_config.get("enable_trading"), bool):
            init_logger.warning(f"Invalid 'enable_trading' value. Must be true or false. Using default '{default_config['enable_trading']}'.")
            updated_config["enable_trading"] = default_config["enable_trading"]
            config_needs_saving = True

        if not isinstance(updated_config.get("use_sandbox"), bool):
            init_logger.warning(f"Invalid 'use_sandbox' value. Must be true or false. Using default '{default_config['use_sandbox']}'.")
            updated_config["use_sandbox"] = default_config["use_sandbox"]
            config_needs_saving = True

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.vt_step_atr_multiplier", 0.1, 20.0) # Added validation
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # e.g., 1.0 to 1.1 range
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # e.g., 1.0 to 1.1 range

        if updated_config.get("strategy_params", {}).get("ob_source") not in ["Wicks", "Body"]:
            init_logger.warning(f"Invalid strategy_params.ob_source. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'.")
            if "strategy_params" not in updated_config: updated_config["strategy_params"] = {} # Ensure dict exists
            updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
            config_needs_saving = True

        if not isinstance(updated_config.get("strategy_params", {}).get("ob_extend"), bool):
            init_logger.warning(f"Invalid strategy_params.ob_extend. Must be true or false. Using default '{DEFAULT_OB_EXTEND}'.")
            if "strategy_params" not in updated_config: updated_config["strategy_params"] = {}
            updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND
            config_needs_saving = True

        # Protection Params
        if not isinstance(updated_config.get("protection", {}).get("enable_trailing_stop"), bool):
            init_logger.warning(f"Invalid protection.enable_trailing_stop. Using default.")
            if "protection" not in updated_config: updated_config["protection"] = {}
            updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]
            config_needs_saving = True

        if not isinstance(updated_config.get("protection", {}).get("enable_break_even"), bool):
            init_logger.warning(f"Invalid protection.enable_break_even. Using default.")
            if "protection" not in updated_config: updated_config["protection"] = {}
            updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]
            config_needs_saving = True

        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.1'), is_strict_min=True) # Must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.1'), allow_zero=True)
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0'))
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('20.0'), is_strict_min=True) # SL > 0
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('20.0'), allow_zero=True) # TP can be 0 (disabled)

        # --- Save Updated Config if Needed ---
        if config_needs_saving:
             init_logger.info(f"Configuration requires updates. Saving changes to '{filepath}'...")
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
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT exchange instance with API keys and settings from config.

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
            'enableRateLimit': True, # Enable built-in rate limiting
            'options': {
                'defaultType': 'linear',      # Prefer linear contracts if available
                'adjustForTimeDifference': True, # Auto-sync time with server
                # Set reasonable timeouts for common operations (in milliseconds)
                'fetchTickerTimeout': 15000,    # 15 seconds
                'fetchBalanceTimeout': 20000,   # 20 seconds
                'createOrderTimeout': 30000,    # 30 seconds
                'cancelOrderTimeout': 20000,    # 20 seconds
                'fetchPositionsTimeout': 20000, # 20 seconds
                'fetchOHLCVTimeout': 60000,     # 60 seconds
            }
        }
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode based on config
        is_sandbox = CONFIG.get('use_sandbox', True) # Default to sandbox for safety
        exchange.set_sandbox_mode(is_sandbox)

        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}<<< SANDBOX MODE ACTIVE >>> Exchange: {exchange.id} {RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< LIVE TRADING ACTIVE >>> Exchange: {exchange.id} !!!{RESET}")

        # Load market data (crucial for symbol info, precision, limits)
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error: Optional[Exception] = None
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
            except ccxt.AuthenticationError as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. "
                            f"Check API Key/Secret and permissions. Exiting.{RESET}")
                return None # Non-retryable
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected critical error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None # Non-retryable

            # Wait before retrying if not loaded and more attempts remain
            if not markets_loaded and attempt < MAX_API_RETRIES:
                delay = RETRY_DELAY_SECONDS * (attempt + 1) # Exponential backoff
                lg.warning(f"Retrying market load in {delay}s...")
                time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load market data after {MAX_API_RETRIES + 1} attempts. "
                        f"Last error: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"Exchange initialized: {exchange.id} | Sandbox: {is_sandbox}")

        # Perform an initial balance check (optional but recommended)
        balance_currency = CONFIG.get("quote_currency", QUOTE_CURRENCY) # Use configured quote currency
        lg.info(f"Performing initial balance check for {balance_currency}...")
        initial_balance: Optional[Decimal] = None
        try:
            initial_balance = fetch_balance(exchange, balance_currency, lg)
        except ccxt.AuthenticationError as auth_err:
            # Catch auth error specifically here as fetch_balance might re-raise it
            lg.critical(f"{NEON_RED}Authentication error during initial balance check: {auth_err}. Exiting.{RESET}")
            return None
        except Exception as balance_err:
            # Log other balance errors as warnings, especially if trading is disabled
            lg.warning(f"{NEON_YELLOW}Initial balance check failed: {balance_err}.{RESET}", exc_info=False) # exc_info=False to avoid noisy tracebacks for common issues

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
                return exchange # Allow proceeding if trading is off

    except ccxt.AuthenticationError as e:
         lg.critical(f"{NEON_RED}Authentication error during exchange setup: {e}. Exiting.{RESET}")
         return None
    except Exception as e:
        lg.critical(f"{NEON_RED}A critical error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Helper Functions ---

def _safe_market_decimal(value: Optional[Any], field_name: str,
                         allow_zero: bool = True, allow_negative: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value (often from market data) to a Decimal.

    Args:
        value: The value to convert (can be string, number, None).
        field_name: Name of the field being converted (for logging).
        allow_zero: Allow zero as a valid value.
        allow_negative: Allow negative values.

    Returns:
        The Decimal value, or None if conversion fails or value is invalid.
    """
    if value is None:
        return None
    try:
        # Convert to string first to handle potential floats accurately
        s_val = str(value).strip()
        if not s_val: # Handle empty strings
            return None
        d_val = Decimal(s_val)

        # Validate based on flags
        if not allow_zero and d_val.is_zero():
            # init_logger.debug(f"Zero value rejected for '{field_name}': {value}")
            return None
        if not allow_negative and d_val < Decimal('0'):
            # init_logger.debug(f"Negative value rejected for '{field_name}': {value}")
            return None

        # Check for NaN or Infinity (though Decimal usually raises InvalidOperation)
        if not d_val.is_finite():
             # init_logger.debug(f"Non-finite value rejected for '{field_name}': {value}")
             return None

        return d_val
    except (InvalidOperation, TypeError, ValueError):
        # init_logger.debug(f"Failed to convert '{field_name}' to Decimal: {value}")
        return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price according to the market's precision rules using ccxt.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        price: The price to format.

    Returns:
        The formatted price as a string, or None if formatting fails or price is invalid.
    """
    try:
        price_decimal = Decimal(str(price))
        # Ensure price is positive before formatting
        if price_decimal <= Decimal('0'):
            init_logger.warning(f"Attempted to format non-positive price '{price}' for {symbol}. Returning None.")
            return None

        # Use ccxt's built-in method for price formatting
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))

        # Double-check: Ensure formatted string is still a positive value
        # (Handles cases where precision might round down to zero)
        if Decimal(formatted_str) <= Decimal('0'):
             init_logger.warning(f"Price '{price}' for {symbol} formatted to non-positive value '{formatted_str}'. Returning None.")
             return None

        return formatted_str
    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
        init_logger.error(f"Error accessing market precision for {symbol}: {e}. Cannot format price.")
        return None
    except (InvalidOperation, ValueError, TypeError) as e:
        init_logger.warning(f"Error converting price '{price}' to Decimal/float for formatting ({symbol}): {e}")
        return None
    except Exception as e:
        # Catch unexpected errors during formatting
        init_logger.warning(f"Unexpected error formatting price '{price}' for {symbol}: {e}")
        return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using ccxt's fetch_ticker.
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
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for price ({symbol}, Attempt {attempts + 1})...")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None
            source = "N/A" # Source of the price (last, mid, ask, bid)

            # Helper to safely get Decimal from ticker data
            def safe_decimal_from_ticker(val: Optional[Any], name: str) -> Optional[Decimal]:
                return _safe_market_decimal(val, f"ticker.{name}", allow_zero=False, allow_negative=False)

            # Try sources in order of preference
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')
            if price:
                source = "'last' price"
            else:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask:
                    price = (bid + ask) / Decimal('2') # Mid-price
                    source = f"mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()})"
                elif ask:
                    price = ask # Fallback to ask
                    source = f"'ask' price ({ask.normalize()})"
                elif bid:
                    price = bid # Fallback to bid
                    source = f"'bid' price ({bid.normalize()})"

            if price:
                normalized_price = price.normalize()
                lg.debug(f"Current price ({symbol}) obtained from {source}: {normalized_price}")
                return normalized_price
            else:
                # Ticker fetched, but no usable price found
                last_exception = ValueError("No valid price source (last/bid/ask) found in ticker response.")
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
            return None # Non-retryable
        except ccxt.BadSymbol as e:
             last_exception = e
             lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching price on {exchange.id}.{RESET}")
             return None # Non-retryable
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., maintenance, temporary issues)
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching price ({symbol}): {e}. Retrying...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price ({symbol}): {e}{RESET}", exc_info=True)
            # Consider if this should be retryable or fatal
            return None # Treat unexpected errors as potentially fatal for safety

        # Increment attempt count and wait before retrying (if applicable)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches historical kline/OHLCV data for a symbol using ccxt.
    Handles pagination/chunking required by exchanges like Bybit.
    Includes retry logic and data validation.

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
            sp.get('vt_length', 0) * 2, # Example: Need more for initial EMA calculations
            sp.get('vt_atr_period', 0),
            sp.get('vt_vol_ema_length', 0),
            sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1, # Pivots need left+right+current
            sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1
        ) + 50 # Add a buffer
        lg.debug(f"Estimated minimum candles required by strategy: ~{min_required}")
    except Exception as e:
        lg.warning(f"Could not estimate minimum candle requirement: {e}")

    if limit < min_required:
        lg.warning(f"{NEON_YELLOW}Requested kline limit ({limit}) is less than the estimated strategy requirement ({min_required}). "
                   f"Indicator accuracy may be affected, especially on initial runs.{RESET}")

    # Determine category and market ID for Bybit V5 API
    category = 'spot' # Default assumption
    market_id = symbol # Default to symbol if market info fails
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        if market.get('linear'): category = 'linear'
        elif market.get('inverse'): category = 'inverse'
        # else category remains 'spot'
        lg.debug(f"Using API parameters: category='{category}', market ID='{market_id}'.")
    except (ccxt.BadSymbol, KeyError, TypeError) as e:
        lg.warning(f"Could not reliably determine market category/ID for {symbol}: {e}. "
                   f"Proceeding with defaults (category='{category}', market_id='{market_id}'). May fail if incorrect.")

    # --- Fetching Loop ---
    all_ohlcv_data: List[List] = []
    remaining_limit = limit
    end_timestamp_ms: Optional[int] = None # For pagination: fetch candles *before* this timestamp
    # Calculate max chunks generously to avoid infinite loops if API behaves unexpectedly
    max_chunks = math.ceil(limit / BYBIT_API_KLINE_LIMIT) + 2 # Add a buffer
    chunk_num = 0
    total_fetched = 0

    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT)
        lg.debug(f"Fetching kline chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. "
                 f"Ending before TS: {datetime.fromtimestamp(end_timestamp_ms / 1000, tz=timezone.utc) if end_timestamp_ms else 'Latest'}")

        attempts = 0
        last_exception: Optional[Exception] = None
        chunk_data: Optional[List[List]] = None

        while attempts <= MAX_API_RETRIES:
            try:
                # --- Prepare API Call ---
                params = {'category': category} if 'bybit' in exchange.id.lower() else {}
                fetch_args: Dict[str, Any] = {
                    'symbol': symbol,       # Use standard symbol for ccxt call
                    'timeframe': timeframe,
                    'limit': fetch_size,
                    'params': params        # Pass category for Bybit
                }
                # Add 'until' parameter for pagination (fetches candles ending *before* this timestamp)
                if end_timestamp_ms:
                    fetch_args['until'] = end_timestamp_ms

                # --- Execute API Call ---
                lg.debug(f"Calling fetch_ohlcv with args: {fetch_args}")
                chunk_data = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(chunk_data) if chunk_data else 0
                lg.debug(f"API returned {fetched_count_chunk} candles for chunk {chunk_num}.")

                if chunk_data:
                    # --- Data Lag Check (for the first chunk only) ---
                    if chunk_num == 1:
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
                                    chunk_data = None # Discard potentially stale data and force retry
                                    # No break here, let the retry logic handle it
                                else:
                                    lg.debug(f"Lag check passed ({symbol}): Last candle {actual_lag_seconds:.1f}s old (within limit).")
                                    break # Valid chunk received, exit retry loop
                            else:
                                lg.warning(f"Could not parse timeframe '{timeframe}' for lag check.")
                                break # Proceed without lag check if timeframe parsing fails
                        except IndexError:
                             lg.warning(f"Could not perform lag check: No data in first chunk?")
                             break # Should not happen if chunk_data is non-empty, but handle defensively
                        except Exception as ts_err:
                            lg.warning(f"Error during lag check ({symbol}): {ts_err}. Proceeding cautiously.")
                            break # Proceed if lag check itself fails

                    else: # Not the first chunk, no lag check needed
                        break # Valid chunk received, exit retry loop

                else:
                    # API returned empty list - could be end of history or temporary issue
                    lg.debug(f"API returned no data for chunk {chunk_num}. Assuming end of history or temporary issue.")
                    # If it's the *first* chunk, we should retry. If later chunks, maybe end of history.
                    if chunk_num > 1:
                        remaining_limit = 0 # Assume end of history if not the first chunk
                    # No 'break' here, let retry logic handle potential temporary issue
                    # Unless we assume end of history, then break outer loop
                    if remaining_limit == 0:
                        break


            # --- Error Handling for fetch_ohlcv call ---
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_exception = e
                lg.warning(f"{NEON_YELLOW}Network error fetching klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait = RETRY_DELAY_SECONDS * 3 # Longer wait for rate limits
                lg.warning(f"{NEON_YELLOW}Rate limit fetching klines chunk {chunk_num} ({symbol}): {e}. Waiting {wait}s...{RESET}")
                time.sleep(wait)
                continue # Don't increment standard attempts, just wait
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Authentication error fetching klines: {e}. Cannot continue.{RESET}")
                return pd.DataFrame() # Fatal
            except ccxt.BadSymbol as e:
                 last_exception = e
                 lg.error(f"{NEON_RED}Invalid symbol '{symbol}' for fetching klines on {exchange.id}.{RESET}")
                 return pd.DataFrame() # Fatal
            except ccxt.ExchangeError as e:
                last_exception = e
                lg.error(f"{NEON_RED}Exchange error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}")
                # Check for specific non-retryable errors (e.g., invalid timeframe)
                err_str = str(e).lower()
                non_retryable_msgs = ["invalid timeframe", "interval not supported", "symbol invalid", "instrument not found"]
                if any(msg in err_str for msg in non_retryable_msgs):
                    lg.critical(f"{NEON_RED}Non-retryable exchange error encountered: {e}. Stopping kline fetch for {symbol}.{RESET}")
                    return pd.DataFrame() # Fatal for this symbol
                # Otherwise, treat as potentially retryable
            except Exception as e:
                last_exception = e
                lg.error(f"{NEON_RED}Unexpected error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True)
                # Treat unexpected errors cautiously - potentially stop fetching for this symbol
                return pd.DataFrame()

            # --- Retry Logic ---
            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None: # Only sleep if we need to retry
                time.sleep(RETRY_DELAY_SECONDS * attempts)

        # --- Process Successful Chunk or Handle Failure ---
        if chunk_data:
            # Prepend the new chunk to maintain chronological order
            all_ohlcv_data = chunk_data + all_ohlcv_data
            chunk_len = len(chunk_data)
            remaining_limit -= chunk_len
            total_fetched += chunk_len

            # Set timestamp for the next older chunk request
            # Use the timestamp of the *first* candle in the *current* chunk minus 1 millisecond
            end_timestamp_ms = chunk_data[0][0] - 1

            # Check if the exchange returned fewer candles than requested (might be end of history)
            if chunk_len < fetch_size:
                lg.debug(f"Received fewer candles ({chunk_len}) than requested ({fetch_size}). Assuming end of historical data.")
                remaining_limit = 0 # Stop fetching more chunks

        else: # Failed to fetch chunk after retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after {MAX_API_RETRIES + 1} attempts. "
                     f"Last error: {last_exception}{RESET}")
            if not all_ohlcv_data:
                # Failed on the very first chunk, cannot proceed
                lg.error(f"Failed to fetch the initial chunk for {symbol}. Cannot construct DataFrame.")
                return pd.DataFrame()
            else:
                # Failed on a subsequent chunk, proceed with what we have
                lg.warning(f"Proceeding with {total_fetched} candles fetched before the error occurred.")
                break # Exit the fetching loop

        # Small delay between chunk requests to be polite to the API
        if remaining_limit > 0:
            time.sleep(0.5) # 500ms delay

    # --- Post-Fetching Checks ---
    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines for {symbol} because maximum chunk limit ({max_chunks}) was reached. "
                   f"Fetched {total_fetched} candles.")

    if not all_ohlcv_data:
        lg.error(f"No kline data could be fetched for {symbol} {timeframe}.")
        return pd.DataFrame()

    lg.info(f"Total raw klines fetched: {len(all_ohlcv_data)}")

    # --- Data Deduplication and Sorting ---
    # Exchanges might occasionally return overlapping data, especially near real-time
    seen_timestamps = set()
    unique_data = []
    # Iterate in reverse chronological order (most recent first) to keep the latest if duplicates exist
    for candle in reversed(all_ohlcv_data):
        timestamp = candle[0]
        if timestamp not in seen_timestamps:
            # Insert at the beginning to rebuild correct chronological order
            unique_data.insert(0, candle)
            seen_timestamps.add(timestamp)

    duplicates_removed = len(all_ohlcv_data) - len(unique_data)
    if duplicates_removed > 0:
        lg.warning(f"Removed {duplicates_removed} duplicate candle(s) based on timestamp for {symbol}.")

    # Ensure data is sorted chronologically (should be, but double-check)
    # unique_data.sort(key=lambda x: x[0]) # Not needed if inserted correctly above

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
        df.dropna(subset=['timestamp'], inplace=True)
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
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # Apply Decimal conversion, handling potential NaN/infs introduced by to_numeric
                df[col] = numeric_series.apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                )
            else:
                lg.warning(f"Expected column '{col}' not found in fetched data for {symbol}.")

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in essential OHLC columns
        essential_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_cols, inplace=True)
        # Ensure close price is positive
        df = df[df['close'] > Decimal('0')]
        # Handle volume column if it exists
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True) # Drop rows with NaN volume
            df = df[df['volume'] >= Decimal('0')] # Ensure volume is non-negative

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with invalid OHLCV data for {symbol}.")

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
            df = df.iloc[-MAX_DF_LEN:].copy() # Keep the most recent MAX_DF_LEN rows

        lg.info(f"{NEON_GREEN}Successfully processed {len(df)} klines for {symbol} {timeframe}.{RESET}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing fetched klines into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Retrieves and standardizes market information for a symbol from the exchange.
    Includes derived fields for convenience (e.g., is_linear, decimal precision/limits).
    Includes retry logic.

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
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            market: Optional[Dict] = None
            # Check if markets are loaded and contain the symbol
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' not found in cached data. Attempting to refresh market map...")
                try:
                    exchange.load_markets(reload=True)
                    lg.info(f"Market map refreshed. Found {len(exchange.markets)} markets.")
                except Exception as reload_err:
                    # Log refresh error but continue to try fetching the specific market below
                    last_exception = reload_err
                    lg.error(f"Failed to refresh market map while looking for '{symbol}': {reload_err}")

            # Attempt to get the specific market dictionary
            try:
                market = exchange.market(symbol)
            except ccxt.BadSymbol:
                # This is definitive: the symbol doesn't exist on the exchange
                lg.error(f"{NEON_RED}Symbol '{symbol}' is invalid or not supported by {exchange.id}.{RESET}")
                return None # Non-retryable
            except Exception as fetch_err:
                # Other errors during market dict retrieval (might be temporary)
                last_exception = fetch_err
                lg.warning(f"Error retrieving market dictionary for '{symbol}': {fetch_err}. Retry {attempts + 1}...")
                market = None # Ensure market is None to trigger retry logic

            if market:
                lg.debug(f"Raw market data found for {symbol}. Parsing and standardizing...")
                # --- Standardize and Enhance Market Data ---
                std_market = market.copy() # Work on a copy

                # Basic type flags
                is_spot = std_market.get('spot', False)
                is_swap = std_market.get('swap', False)
                is_future = std_market.get('future', False)
                is_option = std_market.get('option', False) # Added option check
                is_contract_base = std_market.get('contract', False) # Base 'contract' flag

                # Determine contract type
                std_market['is_contract'] = is_swap or is_future or is_option or is_contract_base
                is_linear = std_market.get('linear') # Can be True/False/None
                is_inverse = std_market.get('inverse') # Can be True/False/None

                # Ensure linear/inverse flags are only True if it's actually a contract
                std_market['is_linear'] = bool(is_linear) and std_market['is_contract']
                std_market['is_inverse'] = bool(is_inverse) and std_market['is_contract']

                # Determine contract type string
                if std_market['is_linear']:
                    std_market['contract_type_str'] = "Linear"
                elif std_market['is_inverse']:
                    std_market['contract_type_str'] = "Inverse"
                elif is_spot:
                     std_market['contract_type_str'] = "Spot"
                elif is_option:
                     std_market['contract_type_str'] = "Option" # Handle options if needed
                else:
                     std_market['contract_type_str'] = "Unknown"

                # --- Extract Precision and Limits Safely ---
                precision = std_market.get('precision', {})
                limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                # Convert precision steps to Decimal
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), f"{symbol} prec.amount", allow_zero=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), f"{symbol} prec.price", allow_zero=False)

                # Convert limits to Decimal
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), f"{symbol} lim.amt.min")
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), f"{symbol} lim.amt.max", allow_negative=False) # Max amount shouldn't be negative
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), f"{symbol} lim.cost.min")
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), f"{symbol} lim.cost.max", allow_negative=False) # Max cost shouldn't be negative

                # Convert contract size to Decimal (default to 1 if missing/invalid)
                contract_size_val = std_market.get('contractSize', '1')
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, f"{symbol} contractSize", allow_zero=False) or Decimal('1')

                # --- Validation of Critical Data ---
                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None:
                    lg.critical(f"{NEON_RED}CRITICAL VALIDATION FAILED ({symbol}): Missing essential precision data! "
                                f"Amount Step: {std_market['amount_precision_step_decimal']}, "
                                f"Price Step: {std_market['price_precision_step_decimal']}. Cannot proceed safely with this symbol.{RESET}")
                    # Depending on strictness, you might return None here or let it continue with potential issues later
                    # Returning None is safer if precision is absolutely required for trading.
                    return None

                # --- Log Parsed Details ---
                amt_s = std_market['amount_precision_step_decimal'].normalize()
                price_s = std_market['price_precision_step_decimal'].normalize()
                min_a = std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] else 'N/A'
                max_a = std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] else 'N/A'
                min_c = std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] else 'N/A'
                max_c = std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] else 'N/A'
                contr_s = std_market['contract_size_decimal'].normalize()
                active_status = std_market.get('active', 'Unknown')

                log_msg = (
                    f"Market Details Parsed ({symbol}): Type={std_market['contract_type_str']}, Active={active_status}\n"
                    f"  Precision (Amount/Price Step): {amt_s} / {price_s}\n"
                    f"  Limits    (Amount Min/Max) : {min_a} / {max_a}\n"
                    f"  Limits    (Cost Min/Max)   : {min_c} / {max_c}\n"
                    f"  Contract Size: {contr_s}"
                )
                lg.debug(log_msg)

                # --- Cast to TypedDict and Return ---
                try:
                    # Attempt to cast the enhanced dictionary to the MarketInfo type
                    final_market_info: MarketInfo = std_market # type: ignore [assignment]
                    return final_market_info
                except Exception as cast_err:
                    # Should not happen if MarketInfo matches the dict structure, but catch just in case
                    lg.error(f"Internal error casting market dictionary to MarketInfo type ({symbol}): {cast_err}. Returning raw dict.")
                    return std_market # type: ignore [return-value] # Return the dict anyway

            else:
                # Market object was None after attempting fetch/lookup
                if attempts < MAX_API_RETRIES:
                    lg.warning(f"Market '{symbol}' not found or fetch failed (Attempt {attempts + 1}). Retrying...")
                else:
                    # Failed after all retries
                    lg.error(f"{NEON_RED}Failed to retrieve market information for '{symbol}' after {MAX_API_RETRIES + 1} attempts. "
                             f"Last error: {last_exception}{RESET}")
                    return None

        # --- Error Handling for the Loop Iteration ---
        except ccxt.BadSymbol as e:
            # This might be caught inside, but also handle here for robustness
            lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}")
            return None # Non-retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error retrieving market info ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to NetworkError fetching market info ({symbol}).{RESET}")
                return None
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error retrieving market info: {e}. Cannot continue.{RESET}")
            return None # Non-retryable
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
            return None # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # Should not be reached if logic is correct, but as a fallback:
    lg.error(f"{NEON_RED}Failed to get market info for {symbol} after exhausting attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency from the exchange.
    Handles different account types for exchanges like Bybit (Unified/Contract).
    Includes retry logic.

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
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None
            balance_source: str = "N/A" # Where the balance was found (e.g., account type, field name)
            found: bool = False
            balance_info: Optional[Dict] = None # Store the last fetched balance structure for debugging

            # For Bybit, try specific account types, then default
            # For other exchanges, the default '' usually works
            types_to_check = ['UNIFIED', 'CONTRACT', ''] if 'bybit' in exchange.id.lower() else ['']

            for acc_type in types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}
                    type_desc = f"Account Type: {acc_type}" if acc_type else "Default Account"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")

                    balance_info = exchange.fetch_balance(params=params)

                    # --- Try standard ccxt structure first ('free' field) ---
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        balance_source = f"{type_desc} ('free' field)"
                        found = True
                        break # Found balance, exit account type loop

                    # --- Try Bybit V5 specific structure (nested within 'info') ---
                    # Structure: info -> result -> list -> [ { accountType, coin: [ { coin, availableToWithdraw/availableBalance } ] } ]
                    elif ('bybit' in exchange.id.lower() and 'info' in balance_info and
                          isinstance(balance_info.get('info'), dict) and
                          isinstance(balance_info['info'].get('result'), dict) and
                          isinstance(balance_info['info']['result'].get('list'), list)):

                        for account_details in balance_info['info']['result']['list']:
                            # Check if this entry matches the account type we queried (or if query was default)
                            # And ensure 'coin' list exists
                            if (not acc_type or account_details.get('accountType') == acc_type) and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Try different fields for available balance in preferred order
                                        val = coin_data.get('availableToWithdraw') # Most preferred
                                        src = 'availableToWithdraw'
                                        if val is None:
                                            val = coin_data.get('availableBalance') # Next best
                                            src = 'availableBalance'
                                        if val is None:
                                             val = coin_data.get('walletBalance') # Less ideal, might include frozen
                                             src = 'walletBalance'

                                        if val is not None:
                                            balance_str = str(val)
                                            balance_source = f"Bybit V5 ({account_details.get('accountType', 'UnknownType')}, field: '{src}')"
                                            found = True
                                            break # Found coin data, exit coin loop
                                if found: break # Exit account details loop
                        if found: break # Exit account type loop

                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # Bybit might throw specific errors for invalid account types
                    if acc_type and ("account type does not exist" in err_str or "invalid account type" in err_str):
                        lg.debug(f"Account type '{acc_type}' not found or invalid for balance check. Trying next...")
                        last_exception = e # Keep track of the last error
                        continue # Try the next account type
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
                    continue # Try the next account type

            # --- Process Result After Checking All Account Types ---
            if found and balance_str is not None:
                try:
                    bal_dec = Decimal(balance_str)
                    # Ensure balance is not negative (can happen with margin debt reporting)
                    final_bal = max(bal_dec, Decimal('0'))
                    lg.debug(f"Successfully parsed balance ({currency}) from {balance_source}: {final_bal.normalize()}")
                    return final_bal
                except (ValueError, InvalidOperation, TypeError) as e:
                    # If conversion fails despite finding a string, treat as an exchange error
                    raise ccxt.ExchangeError(f"Failed to convert balance string '{balance_str}' to Decimal for {currency}: {e}")
            elif not found:
                # Currency not found in any checked structure
                raise ccxt.ExchangeError(f"Balance information for currency '{currency}' not found in the response structure. "
                                         f"Last raw response structure (keys): {list(balance_info.keys()) if balance_info else 'None'}")

        # --- Error Handling for fetch_balance call (outer loop) ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance ({currency}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            # This is critical and non-retryable
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching balance: {e}. Cannot continue.{RESET}")
            raise e # Re-raise to be caught by the caller (e.g., initialize_exchange)
        except ccxt.ExchangeError as e:
            # General exchange errors (e.g., currency not found, temporary issues)
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
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> Optional[PositionInfo]:
    """
    Fetches the currently open position for a specific contract symbol.
    Returns None if no position exists or if the symbol is not a contract.
    Standardizes the position information and includes retry logic.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        market_info: The corresponding MarketInfo dictionary for the symbol.
        logger: The logger instance for messages.

    Returns:
        A PositionInfo TypedDict if an active position exists, otherwise None.
    """
    lg = logger

    # --- Pre-checks ---
    if not market_info.get('is_contract'):
        lg.debug(f"Position check skipped for {symbol}: It is a '{market_info.get('contract_type_str', 'Unknown')}' market, not a contract.")
        return None

    market_id = market_info.get('id')
    category = market_info.get('contract_type_str', 'Unknown').lower() # 'linear', 'inverse', or 'spot'/'unknown'

    if not market_id or category not in ['linear', 'inverse']:
        lg.error(f"Cannot check position for {symbol}: Invalid market ID ('{market_id}') or category ('{category}') in market_info.")
        return None

    lg.debug(f"Checking for open position for {symbol} (Market ID: '{market_id}', Category: '{category}')...")

    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions ({symbol}, Attempt {attempts + 1})...")
            positions: List[Dict] = [] # Initialize as empty list

            # --- Fetch Positions from Exchange ---
            try:
                # Bybit V5 requires category and optionally symbol
                # Other exchanges might just need symbol or fetch all positions
                params = {'category': category}
                # Some exchanges might benefit from specifying the symbol if fetch_positions supports it
                if exchange.has.get('fetchPositions') and exchange.has['fetchPositions'] != 'emulated':
                     params['symbol'] = market_id # Pass market_id if supported

                lg.debug(f"Fetching positions with parameters: {params}")

                if exchange.has.get('fetchPositions'):
                    # Use fetch_positions if available
                    all_fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params) # Request specific symbol if possible
                    # Filter results just in case the exchange returns more than requested
                    positions = [
                        p for p in all_fetched_positions
                        if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id
                    ]
                    lg.debug(f"Fetched {len(all_fetched_positions)} position(s) via fetch_positions, "
                             f"filtered to {len(positions)} matching {symbol}/{market_id}.")
                elif exchange.has.get('fetchPosition'):
                     # Fallback to fetchPosition if fetchPositions is not available
                     lg.debug(f"Using fallback fetchPosition for {symbol}...")
                     pos = exchange.fetch_position(symbol, params=params)
                     # fetch_position usually returns a single dict or raises error if no position
                     positions = [pos] if pos else [] # Wrap in list for consistency
                     lg.debug(f"fetchPosition returned: {'Position found' if positions else 'No position found'}")
                else:
                    raise ccxt.NotSupported(f"{exchange.id} does not support fetchPositions or fetchPosition.")

            except ccxt.ExchangeError as e:
                 # Specific handling for "position not found" errors which are not real errors
                 # Bybit V5 retCode: 110025 = position not found
                 # Common error messages:
                 common_no_pos_msgs = ["position not found", "no position", "position does not exist", "order not found"] # Some APIs use order context
                 bybit_no_pos_codes = [110025]

                 err_str = str(e).lower()
                 # Try to extract Bybit retCode if present
                 code_str = ""
                 match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
                 if match:
                     code_str = match.group(2)
                 else: # Fallback to general code attribute if regex fails
                      code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

                 is_bybit_no_pos = code_str and any(str(c) == code_str for c in bybit_no_pos_codes)
                 is_common_no_pos = any(msg in err_str for msg in common_no_pos_msgs)

                 if is_bybit_no_pos or is_common_no_pos:
                     lg.info(f"No open position found for {symbol} (API indicated no position: Code='{code_str}', Msg='{err_str[:50]}...').")
                     return None # This is the expected outcome when no position exists
                 else:
                     # Re-raise other exchange errors
                     raise e

            # --- Process Fetched Positions ---
            active_raw_position: Optional[Dict] = None

            # Define a small threshold for position size based on market precision
            # Use amount step if available, otherwise a very small Decimal
            size_threshold = Decimal('1e-9') # Default tiny threshold
            try:
                amt_step = market_info.get('amount_precision_step_decimal')
                if amt_step and amt_step > 0:
                    # Use a fraction of the step size as the threshold (e.g., 1%)
                    size_threshold = amt_step * Decimal('0.01')
            except Exception as e:
                 lg.warning(f"Could not determine precise size threshold for {symbol} from market info: {e}. Using default {size_threshold.normalize()}")
            lg.debug(f"Using position size threshold > {size_threshold.normalize()} for {symbol}.")

            # Iterate through filtered positions to find one with non-negligible size
            for pos_data in positions:
                # Try to get size from 'info' (often more reliable) or standard 'contracts' field
                size_str = str(pos_data.get('info', {}).get('size', pos_data.get('contracts', ''))).strip()
                if not size_str:
                    lg.debug(f"Skipping position data with missing size field ({symbol}). Data: {pos_data.get('info', {})}")
                    continue

                try:
                    size_decimal = Decimal(size_str)
                    # Check if absolute size exceeds the threshold
                    if abs(size_decimal) > size_threshold:
                        active_raw_position = pos_data
                        # Store the parsed Decimal size directly in the dict for later use
                        active_raw_position['size_decimal'] = size_decimal
                        lg.debug(f"Found active position candidate for {symbol} with size: {size_decimal.normalize()}")
                        break # Found the first active position, stop searching
                    else:
                         lg.debug(f"Skipping position data with size near zero ({symbol}, Size: {size_decimal.normalize()}).")
                except (ValueError, InvalidOperation, TypeError) as e:
                    lg.warning(f"Could not parse position size string '{size_str}' for {symbol}: {e}. Skipping this entry.")
                    continue

            # --- Standardize and Return Active Position ---
            if active_raw_position:
                std_pos = active_raw_position.copy()
                info_dict = std_pos.get('info', {}) # Raw exchange-specific data

                # Determine Side (long/short) - crucial and sometimes inconsistent
                parsed_size = std_pos['size_decimal'] # Use the Decimal size we stored
                side = std_pos.get('side') # Standard ccxt field

                if side not in ['long', 'short']:
                    # Try inferring from Bybit V5 'side' (Buy/Sell) or from the sign of the size
                    side_v5 = str(info_dict.get('side', '')).strip().lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif parsed_size > size_threshold: side = 'long' # Positive size implies long
                    elif parsed_size < -size_threshold: side = 'short' # Negative size implies short
                    else: side = None # Cannot determine side

                if not side:
                    lg.error(f"Could not determine position side for {symbol}. Size: {parsed_size}. Raw Info: {info_dict}")
                    # Cannot proceed without knowing the side
                    return None

                std_pos['side'] = side

                # Safely parse other relevant fields to Decimal where applicable
                # Prefer standard ccxt fields, fallback to 'info' dict fields if needed
                std_pos['entryPrice'] = _safe_market_decimal(
                    std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice'), # Bybit uses avgPrice in info
                    f"{symbol} pos.entry", allow_zero=False)
                std_pos['leverage'] = _safe_market_decimal(
                    std_pos.get('leverage') or info_dict.get('leverage'),
                    f"{symbol} pos.leverage", allow_zero=False)
                std_pos['liquidationPrice'] = _safe_market_decimal(
                    std_pos.get('liquidationPrice') or info_dict.get('liqPrice'), # Bybit uses liqPrice in info
                    f"{symbol} pos.liq", allow_zero=False)
                std_pos['unrealizedPnl'] = _safe_market_decimal(
                    std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl'), # Bybit uses unrealisedPnl
                    f"{symbol} pos.pnl", allow_zero=True, allow_negative=True) # PnL can be zero or negative

                # Extract protection orders (SL, TP, TSL) - these are often strings in 'info'
                def get_protection_value(field_name: str) -> Optional[str]:
                    """Safely gets a protection order value from info, returns None if zero or invalid."""
                    value = info_dict.get(field_name)
                    if value is None: return None
                    s_value = str(value).strip()
                    try:
                        # Check if it's a numeric value > epsilon, return as string
                        if s_value and abs(Decimal(s_value)) > Decimal('1e-12'):
                             return s_value
                        else:
                             return None # Treat '0', '0.0', etc., as no order set
                    except (InvalidOperation, ValueError, TypeError):
                        # Handle non-numeric strings if they appear
                        # lg.debug(f"Non-numeric value found for protection field '{field_name}': {s_value}")
                        return None # Treat non-numeric as no order set

                std_pos['stopLossPrice'] = get_protection_value('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_value('takeProfit')
                # Bybit V5 TSL fields: trailingStop (distance/price), activePrice (activation price)
                std_pos['trailingStopLoss'] = get_protection_value('trailingStop')
                std_pos['tslActivationPrice'] = get_protection_value('activePrice')

                # Initialize bot state tracking fields
                std_pos['be_activated'] = False # Break-even not yet activated by bot logic
                # TSL considered active if the exchange reports a non-zero trailingStop value
                std_pos['tsl_activated'] = bool(std_pos['trailingStopLoss'])

                # --- Log Found Position ---
                # Helper for logging optional Decimal values
                def fmt_log(val: Optional[Any]) -> str:
                    dec = _safe_market_decimal(val, 'log_fmt', True, True)
                    return dec.normalize() if dec is not None else 'N/A'

                ep = fmt_log(std_pos.get('entryPrice'))
                sz = std_pos['size_decimal'].normalize()
                sl = fmt_log(std_pos.get('stopLossPrice'))
                tp = fmt_log(std_pos.get('takeProfitPrice'))
                tsl_dist = fmt_log(std_pos.get('trailingStopLoss'))
                tsl_act = fmt_log(std_pos.get('tslActivationPrice'))
                tsl_str = "N/A"
                if tsl_dist != 'N/A' or tsl_act != 'N/A':
                     tsl_str = f"Dist/Px={tsl_dist} | ActPx={tsl_act}"

                pnl = fmt_log(std_pos.get('unrealizedPnl'))
                liq = fmt_log(std_pos.get('liquidationPrice'))
                lev = fmt_log(std_pos.get('leverage'))

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} "
                        f"Size={sz}, Entry={ep}, Liq={liq}, Lev={lev}x, PnL={pnl}, "
                        f"SL={sl}, TP={tp}, TSL=({tsl_str})")

                # --- Cast to TypedDict and Return ---
                try:
                    final_position_info: PositionInfo = std_pos # type: ignore [assignment]
                    return final_position_info
                except Exception as cast_err:
                    lg.error(f"Internal error casting position dictionary to PositionInfo type ({symbol}): {cast_err}. Returning raw dict.")
                    return std_pos # type: ignore [return-value]

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
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching positions: {e}. Cannot continue.{RESET}")
            return None # Non-retryable
        except ccxt.ExchangeError as e:
            # Handled specific "no position" cases above, this catches others
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching positions ({symbol}): {e}{RESET}", exc_info=True)
            return None # Treat unexpected errors as fatal for this function

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a given contract symbol.
    Handles specific requirements for exchanges like Bybit (category, buy/sell leverage).
    Includes retry logic and checks for success/no change.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        leverage: The desired integer leverage level.
        market_info: The MarketInfo dictionary for the symbol.
        logger: The logger instance for messages.

    Returns:
        True if leverage was set successfully or already set to the desired value, False otherwise.
    """
    lg = logger

    # --- Pre-checks ---
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True # No action needed for non-contracts

    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Leverage setting failed for {symbol}: Invalid leverage value '{leverage}'. Must be a positive integer.")
        return False

    # Check if the exchange supports setting leverage
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.error(f"Leverage setting failed: Exchange {exchange.id} does not support setLeverage method.")
        return False

    market_id = market_info.get('id')
    category = market_info.get('contract_type_str', 'Unknown').lower()

    if not market_id:
         lg.error(f"Leverage setting failed for {symbol}: Market ID missing in market_info.")
         return False

    lg.info(f"Attempting to set leverage for {symbol} (Market ID: {market_id}, Category: {category}) to {leverage}x...")

    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"set_leverage call attempt {attempts + 1} for {market_id} to {leverage}x...")
            params = {}

            # --- Exchange-Specific Parameter Handling (Bybit V5 example) ---
            if 'bybit' in exchange.id.lower():
                 if category not in ['linear', 'inverse']:
                      lg.error(f"Leverage setting failed for Bybit symbol {symbol}: Invalid category '{category}'. Must be 'linear' or 'inverse'.")
                      return False # Cannot proceed without valid category for Bybit
                 # Bybit V5 requires category and separate buy/sell leverage
                 params = {
                     'category': category,
                     'buyLeverage': str(leverage), # Must be strings for Bybit API
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 specific leverage parameters: {params}")

            # --- Execute set_leverage Call ---
            # Note: ccxt `set_leverage` takes leverage as float/int, symbol, and params
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            lg.debug(f"Raw response from set_leverage ({symbol}): {response}")

            # --- Response Validation (especially for Bybit) ---
            ret_code_str: Optional[str] = None
            ret_msg: str = "Response format not recognized or empty."

            if isinstance(response, dict):
                # Try extracting Bybit V5 style response codes/messages from 'info'
                info_dict = response.get('info', {})
                raw_code = info_dict.get('retCode') # Primary location in V5
                # Fallback to root level if 'info' structure isn't as expected
                if raw_code is None: raw_code = response.get('retCode')
                ret_code_str = str(raw_code) if raw_code is not None else None

                # Get message
                ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown message')) # Prefer info.retMsg

            # Check Bybit success code (0) or "leverage not modified" code (110045)
            if ret_code_str == '0':
                lg.info(f"{NEON_GREEN}Leverage successfully set for {market_id} to {leverage}x (Code: 0).{RESET}")
                return True
            elif ret_code_str == '110045':
                lg.info(f"{NEON_YELLOW}Leverage for {market_id} is already {leverage}x (Code: 110045 - Not Modified). Success.{RESET}")
                return True
            elif ret_code_str is not None and ret_code_str not in ['None', '0']:
                # Specific Bybit error code received
                raise ccxt.ExchangeError(f"Bybit API error setting leverage for {symbol}: {ret_msg} (Code: {ret_code_str})")
            elif response is not None:
                 # Non-Bybit exchange or unrecognized Bybit success response - assume success if no exception
                 lg.info(f"{NEON_GREEN}Leverage set/confirmed for {market_id} to {leverage}x (No specific code in response, assumed success).{RESET}")
                 return True
            else:
                 # Response was None or empty, which is unexpected
                 raise ccxt.ExchangeError(f"Received unexpected empty response after setting leverage for {symbol}.")


        # --- Error Handling for set_leverage call ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str_lower = str(e).lower()
            # Try to extract error code again, specifically for logging/decision making
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            else: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

            lg.error(f"{NEON_RED}Exchange error setting leverage ({market_id} to {leverage}x): {e} (Code: {err_code_str}){RESET}")

            # Check if the error indicates leverage was already set (redundant but safe)
            if err_code_str == '110045' or "leverage not modified" in err_str_lower:
                lg.info(f"{NEON_YELLOW}Leverage already set to {leverage}x (confirmed via error response). Success.{RESET}")
                return True

            # Check for known fatal/non-retryable error codes or messages
            fatal_codes = ['10001','10004','110009','110013','110028','110043','110044','110055','3400045'] # Example Bybit codes
            fatal_messages = [
                "margin mode", "position exists", "risk limit", "parameter error",
                "insufficient available balance", "invalid leverage value",
                "isolated margin mode" # Can prevent leverage change
            ]
            is_fatal_code = err_code_str in fatal_codes
            is_fatal_message = any(msg in err_str_lower for msg in fatal_messages)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE leverage error for {symbol}. Aborting leverage setting.{RESET}")
                # Potentially add more specific advice based on the error
                if "position exists" in err_str_lower:
                    lg.error(" >> Cannot change leverage while a position is open.")
                elif "margin mode" in err_str_lower:
                     lg.error(" >> Leverage change might conflict with current margin mode (cross/isolated).")
                return False # Non-retryable failure

            # If not fatal, proceed to retry logic
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to ExchangeError setting leverage ({symbol}).{RESET}")
                return False

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting leverage ({market_id}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached due to NetworkError setting leverage ({symbol}).{RESET}")
                return False
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error setting leverage ({symbol}): {e}. Cannot continue.{RESET}")
            return False # Non-retryable
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting leverage ({market_id}): {e}{RESET}", exc_info=True)
            return False # Treat unexpected errors as fatal

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to set leverage for {market_id} to {leverage}x after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return False

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: MarketInfo,
    exchange: ccxt.Exchange, # Added exchange for potential future use (e.g., fetching quote price)
    logger: logging.Logger
) -> Optional[Decimal]:
    """
    Calculates the position size based on available balance, risk percentage,
    entry price, and stop loss price. Considers market constraints (min/max size,
    step size, min/max cost) and contract type (linear/inverse).

    Args:
        balance: Available trading balance in the quote currency.
        risk_per_trade: The fraction of the balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: The calculated initial stop loss price.
        entry_price: The estimated entry price (e.g., current market price).
        market_info: The MarketInfo dictionary for the symbol.
        exchange: The ccxt exchange instance.
        logger: The logger instance for messages.

    Returns:
        The calculated and adjusted position size as a Decimal, or None if calculation fails.
        The size represents the quantity in base currency for spot, or number of contracts for futures.
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE') # Fallback if missing
    base_currency = market_info.get('base', 'BASE')   # Fallback if missing
    is_inverse = market_info.get('is_inverse', False)
    is_spot = market_info.get('spot', False)
    # Determine the unit of the calculated size
    size_unit = base_currency if is_spot else "Contracts"

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    # --- Input Validation ---
    if balance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Balance is zero or negative ({balance.normalize()} {quote_currency}).")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
             raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive).")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Entry price ({entry_price.normalize()}) or Stop Loss price ({initial_stop_loss_price.normalize()}) is non-positive.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): Entry price and Stop Loss price cannot be the same ({entry_price.normalize()}).")
        return None

    # --- Extract Market Constraints ---
    try:
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal'] # Used for logging/potential adjustments
        min_amount = market_info['min_amount_decimal'] or Decimal('0') # Treat None as 0
        max_amount = market_info['max_amount_decimal'] or Decimal('inf') # Treat None as infinity
        min_cost = market_info['min_cost_decimal'] or Decimal('0')
        max_cost = market_info['max_cost_decimal'] or Decimal('inf')
        contract_size = market_info['contract_size_decimal'] # Should default to 1 if missing

        # Validate critical constraints
        if not (amount_step and amount_step > 0): raise ValueError("Amount precision step is missing or invalid.")
        if not (price_step and price_step > 0): raise ValueError("Price precision step is missing or invalid.")
        if not (contract_size and contract_size > 0): raise ValueError("Contract size is missing or invalid.")

        lg.debug(f"  Market Constraints ({symbol}):")
        lg.debug(f"    Amount Step: {amount_step.normalize()}, Min Amount: {min_amount.normalize()}, Max Amount: {max_amount.normalize()}")
        lg.debug(f"    Price Step : {price_step.normalize()}")
        lg.debug(f"    Cost Min   : {min_cost.normalize()}, Cost Max: {max_cost.normalize()}")
        lg.debug(f"    Contract Size: {contract_size.normalize()}, Type: {market_info['contract_type_str']}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error accessing or validating required market details: {e}")
        lg.debug(f"  Problematic MarketInfo: {market_info}")
        return None

    # --- Core Size Calculation ---
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN) # Risk amount in quote currency
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance <= Decimal('0'):
        # Should be caught by earlier check, but safeguard here
        lg.error(f"Sizing failed ({symbol}): Stop loss distance is zero or negative.")
        return None

    lg.info(f"  Inputs:")
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
            # Risk Amount = Size * ContractSize * Price Change per Contract
            # Size = Risk Amount / (ContractSize * SL Distance)
            value_change_per_unit = stop_loss_distance * contract_size
            if value_change_per_unit <= Decimal('1e-18'): # Avoid division by near-zero
                lg.error(f"Sizing failed ({symbol}, Linear/Spot): Calculated value change per unit is near zero. Check prices/contract size.")
                return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Calculation: Size = {risk_amount_quote} / ({stop_loss_distance} * {contract_size}) = {calculated_size}")
        else:
            # --- Inverse Contract ---
            # Risk Amount = Size * ContractSize * |(1/Entry) - (1/SL)|
            # Size = Risk Amount / (ContractSize * |(1/Entry) - (1/SL)|)
            if entry_price <= 0 or initial_stop_loss_price <= 0:
                 lg.error(f"Sizing failed ({symbol}, Inverse): Entry or SL price is non-positive, cannot calculate inverse factor.")
                 return None
            inverse_factor = abs((Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price))
            if inverse_factor <= Decimal('1e-18'): # Avoid division by near-zero
                lg.error(f"Sizing failed ({symbol}, Inverse): Calculated inverse factor is near zero. Check prices.")
                return None
            risk_per_contract_unit = contract_size * inverse_factor
            if risk_per_contract_unit <= Decimal('1e-18'): # Avoid division by near-zero
                 lg.error(f"Sizing failed ({symbol}, Inverse): Calculated risk per contract unit is near zero.")
                 return None
            calculated_size = risk_amount_quote / risk_per_contract_unit
            lg.debug(f"  Inverse Calculation: Size = {risk_amount_quote} / ({contract_size} * {inverse_factor}) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as e:
        lg.error(f"Sizing failed ({symbol}): Mathematical error during core calculation: {e}.")
        return None

    if calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial calculated size is zero or negative ({calculated_size.normalize()}). "
                 f"Check risk amount, SL distance, and contract type.")
        return None

    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Adjust Size Based on Constraints ---
    adjusted_size = calculated_size

    # Helper to estimate cost
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        """Estimates the cost of a position in quote currency."""
        if not isinstance(size, Decimal) or not isinstance(price, Decimal) or price <= 0 or size <= 0:
            return None
        try:
            if not is_inverse: # Linear / Spot
                cost = size * price * contract_size
            else: # Inverse
                cost = (size * contract_size) / price
            # Quantize cost to a reasonable precision (e.g., 8 decimal places) for checks
            return cost.quantize(Decimal('1e-8'), ROUND_UP) # Round up cost estimate slightly
        except (InvalidOperation, OverflowError, ZeroDivisionError):
            return None

    # 1. Apply Min/Max Amount Limits
    if min_amount > 0 and adjusted_size < min_amount:
        lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Initial size {adjusted_size.normalize()} < Min Amount {min_amount.normalize()}. Adjusting UP to Min Amount.{RESET}")
        adjusted_size = min_amount
    if max_amount < Decimal('inf') and adjusted_size > max_amount:
        lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Initial size {adjusted_size.normalize()} > Max Amount {max_amount.normalize()}. Adjusting DOWN to Max Amount.{RESET}")
        adjusted_size = max_amount

    lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 2. Apply Min/Max Cost Limits (Requires estimating cost)
    cost_adjusted = False
    estimated_cost = estimate_cost(adjusted_size, entry_price)

    if estimated_cost is not None:
        lg.debug(f"  Estimated Cost (after amount limits, {symbol}): {estimated_cost.normalize()} {quote_currency}")

        # Check Min Cost
        if min_cost > 0 and estimated_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Estimated cost {estimated_cost.normalize()} < Min Cost {min_cost.normalize()}. Attempting to increase size.{RESET}")
            cost_adjusted = True
            try:
                # Calculate the theoretical size needed to meet min cost
                required_size_for_min_cost: Decimal
                if not is_inverse:
                    if entry_price * contract_size <= 0: raise ZeroDivisionError("Entry price * contract size is zero.")
                    required_size_for_min_cost = min_cost / (entry_price * contract_size)
                else:
                    if contract_size <= 0: raise ZeroDivisionError("Contract size is zero.")
                    required_size_for_min_cost = (min_cost * entry_price) / contract_size

                if required_size_for_min_cost <= 0: raise ValueError("Calculated required size is non-positive.")

                lg.info(f"  Theoretical size required for Min Cost ({symbol}): {required_size_for_min_cost.normalize()} {size_unit}")

                # Adjust size up to the required size, but respect min_amount and max_amount
                target_size = max(min_amount, required_size_for_min_cost)

                if target_size > max_amount:
                    lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet Min Cost ({min_cost.normalize()}). "
                             f"Required size ({target_size.normalize()}) exceeds Max Amount ({max_amount.normalize()}).{RESET}")
                    return None
                else:
                    adjusted_size = target_size
                    lg.info(f"  Adjusted size UP to meet Min Cost (respecting limits): {adjusted_size.normalize()} {size_unit}")

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating size required for Min Cost: {e}.{RESET}")
                return None

        # Check Max Cost
        elif max_cost < Decimal('inf') and estimated_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Estimated cost {estimated_cost.normalize()} > Max Cost {max_cost.normalize()}. Attempting to reduce size.{RESET}")
            cost_adjusted = True
            try:
                # Calculate the theoretical maximum size allowed by max cost
                max_size_for_max_cost: Decimal
                if not is_inverse:
                     if entry_price * contract_size <= 0: raise ZeroDivisionError("Entry price * contract size is zero.")
                     max_size_for_max_cost = max_cost / (entry_price * contract_size)
                else:
                     if contract_size <= 0: raise ZeroDivisionError("Contract size is zero.")
                     max_size_for_max_cost = (max_cost * entry_price) / contract_size

                if max_size_for_max_cost <= 0: raise ValueError("Calculated max size is non-positive.")

                lg.info(f"  Theoretical max size allowed by Max Cost ({symbol}): {max_size_for_max_cost.normalize()} {size_unit}")

                # Adjust size down, ensuring it doesn't go below min_amount
                target_size = min(adjusted_size, max_size_for_max_cost) # Take the smaller of current or max allowed
                adjusted_size = max(min_amount, target_size) # Ensure it's still >= min_amount

                # Check if adjustment actually happened and if it fell below min_amount implicitly
                if adjusted_size < target_size:
                     lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Size reduction for Max Cost resulted in {adjusted_size.normalize()}, which is capped by Min Amount {min_amount.normalize()}.{RESET}")
                elif adjusted_size == target_size and target_size < calculated_size: # Check if we actually reduced it
                    lg.info(f"  Adjusted size DOWN to meet Max Cost (respecting limits): {adjusted_size.normalize()} {size_unit}")
                # If adjusted_size didn't change, no log needed here

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating max size allowed by Max Cost: {e}.{RESET}")
                return None

    elif min_cost > 0 or max_cost < Decimal('inf'):
        # Cost limits exist, but we couldn't estimate cost
        lg.warning(f"Could not estimate position cost accurately for {symbol}. Cost limit checks (Min: {min_cost.normalize()}, Max: {max_cost.normalize()}) will be skipped.")

    if cost_adjusted:
        lg.debug(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")


    # 3. Apply Amount Precision (Step Size) - Crucial final step
    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step size is not positive.")
        # Divide by step, round down to nearest integer multiple, then multiply back
        final_size = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN) * amount_step

        if final_size != adjusted_size:
            lg.info(f"Applied amount precision ({symbol}, Step: {amount_step.normalize()}, Rounded DOWN): "
                    f"{adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
        else:
            lg.debug(f"Size already conforms to amount precision ({symbol}, Step: {amount_step.normalize()}).")

    except (InvalidOperation, ValueError, ZeroDivisionError) as e:
        lg.error(f"{NEON_RED}Error applying amount precision (step size) for {symbol}: {e}. Using unrounded size {adjusted_size.normalize()} cautiously.{RESET}")
        # Decide whether to proceed with unrounded size or fail
        # Proceeding is risky as order might fail. Failing is safer.
        # Let's choose to fail for now.
        return None


    # --- Final Validation after Precision ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size after precision adjustment is zero or negative ({final_size.normalize()}).{RESET}")
        return None

    # Re-check Min Amount (rounding down might violate it)
    if min_amount > 0 and final_size < min_amount:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is less than Min Amount {min_amount.normalize()} after applying precision.{RESET}")
        # Attempt to bump up to the next step if possible? Or just fail? Failing is safer.
        # Let's try bumping ONE step.
        next_step_size = final_size + amount_step
        if next_step_size >= min_amount and next_step_size <= max_amount:
             final_cost_next_step = estimate_cost(next_step_size, entry_price)
             if final_cost_next_step is not None and final_cost_next_step >= min_cost and final_cost_next_step <= max_cost:
                  lg.warning(f"{NEON_YELLOW}Final size was below min amount. Bumping UP one step to {next_step_size.normalize()} {size_unit} as it meets all limits.{RESET}")
                  final_size = next_step_size
             else:
                  lg.error(f"{NEON_RED}Cannot bump size to meet Min Amount as next step ({next_step_size.normalize()}) violates other limits (MaxAmt/MinCost/MaxCost). Failing sizing.{RESET}")
                  return None
        else:
             lg.error(f"{NEON_RED}Cannot bump size to meet Min Amount as next step ({next_step_size.normalize()}) violates Max Amount limit ({max_amount.normalize()}). Failing sizing.{RESET}")
             return None


    # Re-check Max Amount (shouldn't be possible if rounding down, but check anyway)
    if max_amount < Decimal('inf') and final_size > max_amount:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is greater than Max Amount {max_amount.normalize()} after applying precision (unexpected!).{RESET}")
        return None

    # Re-check Cost Limits with the final precise size
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}): {final_cost.normalize()} {quote_currency}")
        # Check Min Cost again (rounding down amount might violate it)
        if min_cost > 0 and final_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Final cost {final_cost.normalize()} < Min Cost {min_cost.normalize()} after precision adjustment.{RESET}")
            # Try bumping size by one step IF it doesn't violate other limits
            try:
                next_size = final_size + amount_step
                next_cost = estimate_cost(next_size, entry_price)

                can_bump = (
                    next_cost is not None and
                    next_size <= max_amount and
                    next_cost >= min_cost and # Check if bump meets min cost
                    next_cost <= max_cost
                )

                if can_bump:
                    lg.info(f"{NEON_YELLOW}Bumping final size by one step ({symbol}) to {next_size.normalize()} to meet Min Cost limit.{RESET}")
                    final_size = next_size
                    final_cost = estimate_cost(final_size, entry_price) # Recalculate final cost
                    lg.debug(f"  Final Cost after bump to meet Min Cost: {final_cost.normalize() if final_cost else 'N/A'}")
                else:
                    lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet Min Cost. Final size {final_size.normalize()} has cost {final_cost.normalize()}, "
                             f"and bumping size to {next_size.normalize()} would violate other limits (MaxAmt={max_amount.normalize()}, MaxCost={max_cost.normalize()}) "
                             f"or failed cost estimation ({next_cost}).{RESET}")
                    return None
            except Exception as e:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error occurred while attempting to bump size for Min Cost: {e}.{RESET}")
                return None

        # Check Max Cost again (unlikely to be violated by rounding down, but check)
        elif max_cost < Decimal('inf') and final_cost > max_cost:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} > Max Cost {max_cost.normalize()} after precision adjustment (unexpected!).{RESET}")
            return None
    elif min_cost > 0 or max_cost < Decimal('inf'):
        # Cost limits exist, but we couldn't estimate final cost
        lg.warning(f"Could not perform final cost check for {symbol} after precision adjustment. Order might fail if cost limits are violated.")

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    if final_cost:
         lg.info(f"    Estimated Final Cost: {final_cost.normalize()} {quote_currency}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size

def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """
    Cancels a specific order by its ID.
    Includes retry logic and handles common errors like OrderNotFound.

    Args:
        exchange: The ccxt exchange instance.
        order_id: The ID of the order to cancel.
        symbol: The market symbol associated with the order (required by some exchanges).
        logger: The logger instance for messages.

    Returns:
        True if the order was successfully cancelled or confirmed not found, False otherwise.
    """
    lg = logger
    attempts = 0
    last_exception: Optional[Exception] = None
    lg.info(f"Attempting to cancel order ID {order_id} for symbol {symbol}...")

    # Prepare parameters (e.g., category for Bybit V5)
    market_id = symbol
    params = {}
    if 'bybit' in exchange.id.lower():
        try:
            market = exchange.market(symbol)
            market_id = market['id'] # Use market_id if available
            if market.get('linear'): category = 'linear'
            elif market.get('inverse'): category = 'inverse'
            elif market.get('spot'): category = 'spot'
            else: category = 'linear' # Default guess
            params['category'] = category
            # Bybit cancelOrder might need symbol in params too
            params['symbol'] = market_id
            lg.debug(f"Using Bybit params for cancelOrder: {params}")
        except Exception as e:
            lg.warning(f"Could not get market details to determine category/market_id for cancel ({symbol}): {e}. Proceeding without category.")
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
            return True # Treat as success for workflow purposes
        except ccxt.InvalidOrder as e:
             # E.g., order already filled/cancelled and API gives specific error
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Invalid order state for cancellation ({symbol}, ID: {order_id}): {e}. Assuming cancellation unnecessary/complete.{RESET}")
             return True # Treat as success if it cannot be cancelled due to state
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 2 # Shorter wait for cancel might be okay
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded cancelling order {order_id} ({symbol}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue # Don't count as standard attempt
        except ccxt.ExchangeError as e:
            # Other exchange errors during cancellation
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}. Retrying...{RESET}")
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error cancelling order {order_id} ({symbol}): {e}. Cannot continue.{RESET}")
            return False # Non-retryable
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True)
            # Treat unexpected errors as failure for safety
            return False

        # --- Retry Logic ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to cancel order {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return False

def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"
    position_size: Decimal,
    market_info: MarketInfo,
    logger: logging.Logger,
    reduce_only: bool = False,
    params: Optional[Dict] = None # Allow passing extra params
) -> Optional[Dict]:
    """
    Places a market order based on the trade signal and calculated size.
    Handles specifics for exchanges like Bybit (category, reduceOnly).
    Includes retry logic and error handling.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The standardized market symbol (e.g., 'BTC/USDT').
        trade_signal: The action to take ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The calculated size for the order (always positive).
        market_info: The MarketInfo dictionary for the symbol.
        logger: The logger instance for messages.
        reduce_only: If True, set the reduceOnly flag (for closing positions).
        params: Optional dictionary of extra parameters for create_order.

    Returns:
        The order result dictionary from ccxt if successful, otherwise None.
    """
    lg = logger

    # --- Determine Order Side ---
    side_map = {
        "BUY": "buy",         # Opening a long position
        "SELL": "sell",       # Opening a short position
        "EXIT_SHORT": "buy",  # Closing a short position (buy back)
        "EXIT_LONG": "sell"   # Closing a long position (sell off)
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
    order_type = 'market' # Strategy currently uses market orders
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base_currency
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info.get('id')

    if not market_id:
         lg.error(f"Cannot place trade for {symbol}: Market ID missing in market_info.")
         return None

    # Convert Decimal size to float for ccxt (ensure it's not near zero float)
    try:
        amount_float = float(position_size)
        if amount_float < 1e-15: # Check for effective zero after float conversion
            raise ValueError("Position size converts to near-zero float.")
    except (ValueError, TypeError) as e:
        lg.error(f"Failed to convert position size {position_size.normalize()} to valid float for order placement ({symbol}): {e}")
        return None

    # Base order arguments for ccxt create_order
    order_args: Dict[str, Any] = {
        'symbol': symbol,     # Use standard symbol for ccxt call
        'type': order_type,
        'side': side,
        'amount': amount_float,
        # 'price': None, # Not needed for market orders
    }

    # --- Exchange-Specific Parameters ---
    order_params: Dict[str, Any] = {}
    if 'bybit' in exchange.id.lower() and is_contract:
        try:
            category = market_info.get('contract_type_str', 'Linear').lower()
            if category not in ['linear', 'inverse']:
                raise ValueError(f"Invalid category '{category}' for Bybit contract.")

            order_params = {
                'category': category,
                'positionIdx': 0 # Assume one-way mode (0 index)
                # Other potential Bybit params: timeInForce, postOnly, etc.
            }

            if reduce_only:
                order_params['reduceOnly'] = True
                # Bybit often requires IOC/FOK for reduceOnly market orders
                order_params['timeInForce'] = 'IOC' # Immediate Or Cancel
                lg.debug(f"Setting Bybit V5 reduceOnly and IOC flags for {symbol}.")

        except Exception as e:
            lg.error(f"Failed to set Bybit V5 specific parameters for {symbol}: {e}. Proceeding with base params.")
            order_params = {} # Reset params if setup failed

    # Merge any externally provided params
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
    lg.warning(f"  Size   : {position_size.normalize()} {size_unit}")
    if order_params:
        lg.warning(f"  Params : {order_params}")


    # --- Execute Order Placement with Retry ---
    attempts = 0
    last_exception: Optional[Exception] = None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1})...")
            # --- Place the Order ---
            order_result = exchange.create_order(**order_args)
            # --- --- --- --- --- ---

            # --- Log Success ---
            order_id = order_result.get('id', 'N/A')
            status = order_result.get('status', 'N/A')
            avg_price_raw = order_result.get('average') # Price at which the order was filled (avg)
            filled_raw = order_result.get('filled')   # Amount filled

            avg_price_dec = _safe_market_decimal(avg_price_raw, 'order.average', allow_zero=True) # Avg price can be 0 if not filled?
            filled_dec = _safe_market_decimal(filled_raw, 'order.filled', allow_zero=True) # Filled can be 0 initially

            log_msg = (
                f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET}\n"
                f"  ID: {order_id}, Status: {status}"
            )
            if avg_price_dec is not None:
                log_msg += f", Avg Fill Price: ~{avg_price_dec.normalize()}"
            if filled_dec is not None:
                log_msg += f", Filled Amount: {filled_dec.normalize()} {size_unit}"

            lg.info(log_msg)
            lg.debug(f"Full order result ({symbol}): {order_result}")
            return order_result # Return the successful order details

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
            min_a = market_info.get('min_amount_decimal', 'N/A')
            min_c = market_info.get('min_cost_decimal', 'N/A')
            amt_s = market_info.get('amount_precision_step_decimal', 'N/A')
            max_a = market_info.get('max_amount_decimal', 'N/A')
            max_c = market_info.get('max_cost_decimal', 'N/A')

            if any(s in err_lower for s in ["minimum", "too small", "less than minimum"]):
                lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) against Min Amount ({min_a}) and potentially Min Cost ({min_c}).")
            elif any(s in err_lower for s in ["precision", "lot size", "step size", "size precision"]):
                lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) precision against Amount Step ({amt_s}).")
            elif any(s in err_lower for s in ["exceed", "too large", "greater than maximum"]):
                lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) against Max Amount ({max_a}) and potentially Max Cost ({max_c}).")
            elif "reduce only" in err_lower or "reduceonly" in err_lower:
                lg.error(f"  >> Hint: Reduce-only order failed. Ensure there's an open position to reduce and the size is appropriate.")
            elif "position size" in err_lower:
                 lg.error(f"  >> Hint: Check if order size conflicts with existing position or leverage limits.")

            return None
        except ccxt.ExchangeError as e:
            # General exchange errors - potentially retryable
            last_exception = e
            # Try to extract error code for better logging/decision
            err_code = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code = match.group(2)
            else: err_code = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

            lg.error(f"{NEON_RED}Order Placement Failed ({symbol} {action_desc}): Exchange Error. (Code: {err_code}){RESET}")
            lg.error(f"  Error details: {e}")

            # Check for known fatal/non-retryable error codes or messages
            # (Examples, may need adjustment per exchange)
            fatal_codes = [
                '10001', # Bybit: Parameter error
                '10004', # Bybit: Sign error (API key issue)
                '110007',# Bybit: Batch orders count exceeds limit (if using batch)
                '110013',# Bybit: Price precision issue
                '110014',# Bybit: Size precision issue
                '110017',# Bybit: Position idx not match position mode
                '110025',# Bybit: Position not found (might occur on reduceOnly if pos closed race condition)
                '110040',# Bybit: Order qty exceeds risk limit
                '30086', # Bybit: Order cost exceeds risk limit
                '3303001',# Bybit SPOT: Invalid symbol
                '3303005',# Bybit SPOT: Price/Qty precision issue
                '3400060',# Bybit SPOT: Order amount exceeds balance
                '3400088',# Bybit: Leverage exceed max limit
            ]
            fatal_msgs = [
                "invalid parameter", "precision", "exceed limit", "risk limit",
                "invalid symbol", "reduce only", "lot size",
                "insufficient balance", "leverage exceed", "trigger liquidation",
                "account not unified", "unified account function" # Errors related to account type mismatch
            ]
            is_fatal_code = err_code in fatal_codes
            is_fatal_message = any(msg in str(e).lower() for msg in fatal_msgs)

            if is_fatal_code or is_fatal_message:
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE order placement error for {symbol}.{RESET}")
                return None # Non-retryable failure

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
            wait = RETRY_DELAY_SECONDS * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order ({symbol}): {e}. Waiting {wait}s...{RESET}")
            time.sleep(wait)
            continue # Don't count as standard attempt
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error placing order ({symbol}): {e}. Cannot continue.{RESET}")
            return None # Non-retryable
        except Exception as e:
            # Catch any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order ({symbol}): {e}{RESET}", exc_info=True)
            return None # Treat unexpected errors as fatal

        # --- Wait Before Next Retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts. "
             f"Last error: {last_exception}{RESET}")
    return None


# --- Placeholder Functions (Require Full Implementation) ---

def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # Distance from activation/current price
    tsl_activation_price: Optional[Decimal] = None   # Price at which TSL should activate
) -> bool:
    """
    Placeholder: Sets Stop Loss (SL), Take Profit (TP), and potentially Trailing Stop Loss (TSL)
    for an existing position using exchange-specific methods or parameters.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs to handle formatting prices/distances according to market precision.
    - Needs to use the correct ccxt method or parameters for the exchange (e.g., modify_position, create_order with sl/tp params, specific set_sl_tp methods).
    - Needs robust error handling for API calls.
    - Needs to handle partial setting (e.g., only SL, or SL+TP).
    - Bybit V5 uses `positionIdx`, `category`, `stopLoss`, `takeProfit`, `tpslMode` ('Full' or 'Partial'), `slTriggerBy`, `tpTriggerBy`, potentially `trailingStop`, `activePrice`.

    Args:
        exchange: ccxt exchange instance.
        symbol: Standard symbol.
        market_info: Market information.
        position_info: Current position details.
        logger: Logger instance.
        stop_loss_price: Target SL price.
        take_profit_price: Target TP price.
        trailing_stop_distance: TSL distance (in price points or percentage - needs clarification).
        tsl_activation_price: Price to activate the TSL.

    Returns:
        True if protection was set successfully, False otherwise.
    """
    lg = logger
    lg.warning(f"{NEON_YELLOW}Placeholder Function Called: _set_position_protection for {symbol}{RESET}")
    lg.debug(f"  Attempting to set: SL={stop_loss_price}, TP={take_profit_price}, TSL Dist={trailing_stop_distance}, TSL Act={tsl_activation_price}")

    # --- !!! Actual implementation needed here !!! ---
    # Example structure for Bybit V5 (HIGHLY SIMPLIFIED):
    if 'bybit' in exchange.id.lower():
        params = {
            'category': market_info.get('contract_type_str', 'linear').lower(),
            'symbol': market_info['id'],
            'positionIdx': 0, # Assuming one-way mode
            # 'tpslMode': 'Full', # Or 'Partial'
            # 'slTriggerBy': 'MarkPrice', # Or 'LastPrice', 'IndexPrice'
            # 'tpTriggerBy': 'MarkPrice',
        }
        if stop_loss_price:
            sl_str = _format_price(exchange, symbol, stop_loss_price)
            if sl_str: params['stopLoss'] = sl_str
            else: lg.error(f"Invalid SL price format for {symbol}: {stop_loss_price}"); return False
        if take_profit_price:
             tp_str = _format_price(exchange, symbol, take_profit_price)
             if tp_str: params['takeProfit'] = tp_str
             else: lg.error(f"Invalid TP price format for {symbol}: {take_profit_price}"); return False
        # Trailing Stop logic is more complex - needs distance/activation price formatting
        if trailing_stop_distance:
             # This needs careful handling based on whether distance is price points or %
             # ts_str = exchange.price_to_precision(symbol, float(trailing_stop_distance)) # Example
             # params['trailingStop'] = ts_str
             lg.warning("Trailing stop distance setting not fully implemented in placeholder.")
             pass
        if tsl_activation_price:
             # act_str = _format_price(exchange, symbol, tsl_activation_price)
             # if act_str: params['activePrice'] = act_str
             lg.warning("Trailing stop activation price setting not fully implemented in placeholder.")
             pass

        if 'stopLoss' in params or 'takeProfit' in params or 'trailingStop' in params:
             lg.info(f"Calling set_trading_stop (placeholder) with params: {params}")
             # try:
             #     response = exchange.private_post_position_set_trading_stop(params) # Example V5 endpoint
             #     # Check response code (0 for success)
             #     if response.get('retCode') == 0:
             #         lg.info(f"Protection set successfully via API for {symbol}.")
             #         return True
             #     else:
             #         lg.error(f"API Error setting protection for {symbol}: {response.get('retMsg')} (Code: {response.get('retCode')})")
             #         return False
             # except Exception as e:
             #     lg.error(f"Exception setting protection for {symbol}: {e}", exc_info=True)
             #     return False
             return True # Assume success for placeholder
        else:
             lg.debug("No protection parameters provided to set.")
             return True # Nothing to set, considered success

    # Fallback / Other exchanges
    lg.error(f"Protection setting not implemented for exchange {exchange.id} in placeholder.")
    return False # Assume failure if not implemented

def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    config: Dict[str, Any], # Pass config for TSL parameters
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Can optionally set TP at the same time
) -> bool:
    """
    Placeholder: Sets up an initial Trailing Stop Loss (TSL) based on config parameters.
    This is typically called when enabling TSL for the first time on a position.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs to calculate the TSL distance and activation price based on config
      (e.g., callback_rate, activation_percentage) and current price/entry price.
    - Needs to call _set_position_protection or the relevant API method.

    Args:
        exchange: ccxt exchange instance.
        symbol: Standard symbol.
        market_info: Market information.
        position_info: Current position details.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional TP price to set simultaneously.

    Returns:
        True if TSL setup was successful, False otherwise.
    """
    lg = logger
    lg.warning(f"{NEON_YELLOW}Placeholder Function Called: set_trailing_stop_loss for {symbol}{RESET}")

    prot_cfg = config.get('protection', {})
    callback_rate = prot_cfg.get('trailing_stop_callback_rate', 0.005) # e.g., 0.5%
    activation_perc = prot_cfg.get('trailing_stop_activation_percentage', 0.003) # e.g., 0.3%

    entry_price = position_info.get('entryPrice')
    current_price = fetch_current_price_ccxt(exchange, symbol, lg) # Need current price
    side = position_info.get('side')

    if not entry_price or not current_price or not side:
        lg.error(f"Cannot set TSL for {symbol}: Missing entry price, current price, or side.")
        return False

    entry_price_dec = Decimal(str(entry_price))

    # --- !!! TSL Calculation Logic Needed Here !!! ---
    # 1. Calculate Activation Price:
    #    - Long: entry_price * (1 + activation_perc)
    #    - Short: entry_price * (1 - activation_perc)
    # 2. Calculate TSL Distance (Callback):
    #    - This might be an absolute price distance or percentage depending on API.
    #    - Bybit V5 'trailingStop' can be distance (e.g., "100") or price ("25000").
    #    - Let's assume callback_rate is price distance for now (needs clarification).
    #    - Distance = current_price * callback_rate (or entry_price * callback_rate?) - Needs strategy decision.
    #    - Let's use current price for distance calculation as an example.
    tsl_distance_calc = current_price * Decimal(str(callback_rate))
    activation_price_calc: Decimal
    if side == 'long':
        activation_price_calc = entry_price_dec * (Decimal('1') + Decimal(str(activation_perc)))
    else: # short
        activation_price_calc = entry_price_dec * (Decimal('1') - Decimal(str(activation_perc)))

    lg.info(f"Calculated TSL params for {symbol}: ActivationPrice={activation_price_calc.normalize()}, Distance={tsl_distance_calc.normalize()} (based on current price)")

    # Format for API (placeholder - assumes distance is absolute price points)
    tsl_distance_str = exchange.price_to_precision(symbol, float(tsl_distance_calc)) # Example formatting
    activation_price_str = _format_price(exchange, symbol, activation_price_calc)

    if not tsl_distance_str or not activation_price_str:
         lg.error(f"Failed to format TSL distance or activation price for {symbol}.")
         return False

    # --- Call the actual protection setting function ---
    lg.info(f"Calling _set_position_protection to apply TSL (and optional TP) for {symbol}")
    # Note: Passing distance and activation price based on calculation above.
    # The actual _set_position_protection needs to handle these correctly for the exchange API.
    # This placeholder passes them as separate args, but Bybit might take them differently.
    success = _set_position_protection(
        exchange, symbol, market_info, position_info, lg,
        stop_loss_price=None, # Not setting regular SL here
        take_profit_price=take_profit_price, # Pass through optional TP
        trailing_stop_distance=Decimal(tsl_distance_str), # Pass calculated distance
        tsl_activation_price=Decimal(activation_price_str) # Pass calculated activation price
    )

    if success:
        lg.info(f"Trailing stop loss setup initiated successfully for {symbol}.")
        # Update internal state marker (this should ideally be done in the main loop after confirmation)
        # position_info['tsl_activated'] = True # Mark as activated (logically)
    else:
        lg.error(f"Failed to set up trailing stop loss for {symbol} via _set_position_protection.")

    return success # Return result of the underlying call


class VolumaticOBStrategy:
    """
    Placeholder: Encapsulates the Volumatic Trend + Order Block strategy logic.
    Calculates indicators, identifies trends, finds order blocks, and stores analysis results.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs implementation of Volumatic Trend calculation (_ema_swma, trend logic).
    - Needs implementation of Pivot High/Low detection (_find_pivots).
    - Needs implementation of Order Block identification and management (creation, violation, extension).
    - Needs robust handling of DataFrame lengths and potential NaN values.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        self.lg = logger
        self.symbol = market_info['symbol']
        self.market_info = market_info
        self.params = config.get('strategy_params', {})
        self.protection_params = config.get('protection', {})
        self.price_tick = market_info['price_precision_step_decimal'] or Decimal('0.00000001') # Miniscule default

        # --- Extract Params (with defaults) ---
        self.vt_len = self.params.get('vt_length', DEFAULT_VT_LENGTH)
        self.vt_atr_period = self.params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD)
        self.vt_vol_ema_len = self.params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH)
        self.vt_atr_mult = Decimal(str(self.params.get('vt_atr_multiplier', DEFAULT_VT_ATR_MULTIPLIER)))
        # self.vt_step_atr_mult = Decimal(str(self.params.get('vt_step_atr_multiplier', DEFAULT_VT_STEP_ATR_MULTIPLIER))) # If needed
        self.ob_source = self.params.get('ob_source', DEFAULT_OB_SOURCE) # "Wicks" or "Body"
        self.ph_left = self.params.get('ph_left', DEFAULT_PH_LEFT)
        self.ph_right = self.params.get('ph_right', DEFAULT_PH_RIGHT)
        self.pl_left = self.params.get('pl_left', DEFAULT_PL_LEFT)
        self.pl_right = self.params.get('pl_right', DEFAULT_PL_RIGHT)
        self.ob_extend = self.params.get('ob_extend', DEFAULT_OB_EXTEND)
        self.ob_max_boxes = self.params.get('ob_max_boxes', DEFAULT_OB_MAX_BOXES)

        # Estimate minimum data length needed
        self.min_data_len = max(self.vt_len * 2, self.vt_atr_period, self.vt_vol_ema_len,
                                self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1) + 50

        self.lg.info(f"Strategy Engine initialized for {self.symbol} with min data length ~{self.min_data_len}")
        self.lg.debug(f"  Params: VT Len={self.vt_len}, ATR Period={self.vt_atr_period}, "
                      f"Vol EMA={self.vt_vol_ema_len}, ATR Mult={self.vt_atr_mult}, "
                      f"OB Src={self.ob_source}, Pivots L/R=({self.ph_left}/{self.ph_right}, {self.pl_left}/{self.pl_right}), "
                      f"Extend={self.ob_extend}, Max Boxes={self.ob_max_boxes}")

        # State for tracking order blocks across updates
        self._active_bull_boxes: List[OrderBlock] = []
        self._active_bear_boxes: List[OrderBlock] = []


    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Placeholder: Calculates Smoothed Weighted Moving Average (SWMA) via EMA."""
        # Actual SWMA might be different, this is just a placeholder calculation
        self.lg.debug(f"Placeholder _ema_swma called for length {length}")
        if series.empty or length <= 0: return pd.Series(dtype=float)
        # Simple EMA as placeholder
        # return ta.ema(series, length=length)
        # Or a simple rolling mean as an even simpler placeholder
        return series.rolling(window=min(length, len(series))).mean() # Basic placeholder

    def _find_pivots(self, series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
        """Placeholder: Finds pivot high or low points."""
        self.lg.debug(f"Placeholder _find_pivots called (Left:{left}, Right:{right}, High:{is_high})")
        if series.empty or left < 0 or right < 0: return pd.Series(False, index=series.index)
        # Basic placeholder logic: returns False everywhere
        # Real implementation needs to compare points within the left/right window
        return pd.Series(False, index=series.index)

    def update(self, df: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Placeholder: Processes the input DataFrame to calculate indicators and identify strategy elements.

        Args:
            df: The OHLCV DataFrame with data up to the present.

        Returns:
            A StrategyAnalysisResults dictionary containing the processed DataFrame and key strategy outputs.
        """
        self.lg.debug(f"Running strategy update for {self.symbol} with DataFrame length {len(df)}")
        if len(df) < self.min_data_len:
            self.lg.warning(f"DataFrame length ({len(df)}) is less than minimum required ({self.min_data_len}) for {self.symbol}. Strategy results may be inaccurate.")
            # Return default/empty results if not enough data
            return StrategyAnalysisResults(
                dataframe=df, last_close=df['close'].iloc[-1] if not df.empty else Decimal('0'),
                current_trend_up=None, trend_just_changed=False,
                active_bull_boxes=[], active_bear_boxes=[],
                vol_norm_int=None, atr=None, upper_band=None, lower_band=None
            )

        # --- !!! Placeholder Calculations !!! ---
        # 1. Calculate Indicators (ATR, Vol EMA, VT Bands)
        df_analysis = df.copy()
        try:
            # ATR
            atr_series = ta.atr(df_analysis['high'].astype(float), df_analysis['low'].astype(float), df_analysis['close'].astype(float), length=self.vt_atr_period)
            df_analysis['atr'] = atr_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
            current_atr = df_analysis['atr'].iloc[-1] if pd.notna(df_analysis['atr'].iloc[-1]) else None

            # Vol EMA (Placeholder - using close price EMA)
            vol_ema = self._ema_swma(df_analysis['close'], self.vt_vol_ema_len) # Placeholder using close
            df_analysis['vol_ema'] = vol_ema

            # VT Bands (Placeholder - using simple +/- ATR)
            mid_band = df_analysis['close'].rolling(self.vt_len).mean() # Simple MA placeholder
            df_analysis['vt_upper'] = mid_band + df_analysis['atr'] * self.vt_atr_mult
            df_analysis['vt_lower'] = mid_band - df_analysis['atr'] * self.vt_atr_mult
            upper_band = df_analysis['vt_upper'].iloc[-1] if pd.notna(df_analysis['vt_upper'].iloc[-1]) else None
            lower_band = df_analysis['vt_lower'].iloc[-1] if pd.notna(df_analysis['vt_lower'].iloc[-1]) else None

            # Trend (Placeholder - based on close vs previous close)
            df_analysis['trend_up'] = df_analysis['close'] > df_analysis['close'].shift(1)
            current_trend_up = bool(df_analysis['trend_up'].iloc[-1]) if pd.notna(df_analysis['trend_up'].iloc[-1]) else None
            trend_just_changed = df_analysis['trend_up'].iloc[-1] != df_analysis['trend_up'].iloc[-2] if len(df_analysis) > 1 and pd.notna(df_analysis['trend_up'].iloc[-1]) and pd.notna(df_analysis['trend_up'].iloc[-2]) else False

            # Volume Normalization (Placeholder - simple scaling)
            df_analysis['vol_norm'] = (df_analysis['volume'] - df_analysis['volume'].min()) / (df_analysis['volume'].max() - df_analysis['volume'].min()) * 100 if not df_analysis['volume'].empty and df_analysis['volume'].max() > df_analysis['volume'].min() else 0
            vol_norm_int = int(df_analysis['vol_norm'].iloc[-1]) if pd.notna(df_analysis['vol_norm'].iloc[-1]) else None

            # 2. Find Pivots (Placeholder)
            df_analysis['ph'] = self._find_pivots(df_analysis['high'], self.ph_left, self.ph_right, is_high=True)
            df_analysis['pl'] = self._find_pivots(df_analysis['low'], self.pl_left, self.pl_right, is_high=False)

            # 3. Identify Order Blocks (Placeholder)
            # This logic needs to:
            # - Look for pivots.
            # - Define OB based on pivot candle's wick/body (self.ob_source).
            # - Add new OBs to self._active_bull_boxes / self._active_bear_boxes.
            # - Limit the number of boxes (self.ob_max_boxes).
            # - Check for violation of existing boxes by current price action.
            # - Extend boxes if self.ob_extend is True.
            self.lg.debug("Order Block identification logic is a placeholder.")
            # Keep existing placeholder boxes for now
            active_bull_boxes = self._active_bull_boxes
            active_bear_boxes = self._active_bear_boxes

        except Exception as e:
            self.lg.error(f"Error during strategy calculation for {self.symbol}: {e}", exc_info=True)
            # Return potentially incomplete results or defaults
            return StrategyAnalysisResults(
                dataframe=df, last_close=df['close'].iloc[-1] if not df.empty else Decimal('0'),
                current_trend_up=None, trend_just_changed=False,
                active_bull_boxes=self._active_bull_boxes, active_bear_boxes=self._active_bear_boxes,
                vol_norm_int=None, atr=None, upper_band=None, lower_band=None
            )

        last_close = df_analysis['close'].iloc[-1] if not df_analysis.empty else Decimal('0')
        self.lg.debug(f"Strategy update complete for {self.symbol}. Last Close: {last_close}, TrendUp: {current_trend_up}, ATR: {current_atr}")

        return StrategyAnalysisResults(
            dataframe=df_analysis,
            last_close=last_close,
            current_trend_up=current_trend_up,
            trend_just_changed=trend_just_changed,
            active_bull_boxes=active_bull_boxes, # Use the (placeholder) updated lists
            active_bear_boxes=active_bear_boxes,
            vol_norm_int=vol_norm_int,
            atr=current_atr,
            upper_band=upper_band,
            lower_band=lower_band
        )


class SignalGenerator:
    """
    Placeholder: Generates trading signals ("BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT")
    based on the results from the VolumaticOBStrategy analysis and current position state.

     ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs logic to check trend alignment.
    - Needs logic to check price proximity to active Order Blocks.
    - Needs logic to determine entry signals based on OB interaction.
    - Needs logic to determine exit signals (e.g., price hitting opposite OB, trend reversal).
    - Needs to calculate initial SL/TP based on entry price and ATR/OB levels.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        self.lg = logger
        self.symbol = market_info['symbol']
        self.market_info = market_info
        self.strategy_params = config.get('strategy_params', {})
        self.protection_params = config.get('protection', {})
        self.price_tick = market_info['price_precision_step_decimal'] or Decimal('0.00000001') # Miniscule default

        # --- Extract Params ---
        self.entry_prox_factor = Decimal(str(self.strategy_params.get('ob_entry_proximity_factor', 1.005)))
        self.exit_prox_factor = Decimal(str(self.strategy_params.get('ob_exit_proximity_factor', 1.001)))
        self.sl_atr_mult = Decimal(str(self.protection_params.get('initial_stop_loss_atr_multiple', 1.8)))
        self.tp_atr_mult = Decimal(str(self.protection_params.get('initial_take_profit_atr_multiple', 0.7)))

        self.lg.info(f"Signal Generator initialized for {self.symbol}")
        self.lg.debug(f"  Params: Entry Prox={self.entry_prox_factor}, Exit Prox={self.exit_prox_factor}, "
                      f"SL Mult={self.sl_atr_mult}, TP Mult={self.tp_atr_mult}")

    def _calculate_initial_sl_tp(self, entry_price: Decimal, side: str, atr: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Placeholder: Calculates initial SL and TP based on entry, side, and ATR."""
        if atr <= 0:
             self.lg.warning(f"Cannot calculate SL/TP for {self.symbol}: ATR is zero or negative ({atr}).")
             return None, None

        sl_distance = atr * self.sl_atr_mult
        tp_distance = atr * self.tp_atr_mult if self.tp_atr_mult > 0 else None

        initial_sl = None
        initial_tp = None

        if side == 'long':
            initial_sl = entry_price - sl_distance
            if tp_distance:
                initial_tp = entry_price + tp_distance
        elif side == 'short':
            initial_sl = entry_price + sl_distance
            if tp_distance:
                initial_tp = entry_price - tp_distance

        # Ensure SL/TP are positive
        if initial_sl is not None and initial_sl <= 0:
            self.lg.warning(f"Calculated initial SL ({initial_sl.normalize()}) is non-positive for {self.symbol}. Setting SL to None.")
            initial_sl = None
        if initial_tp is not None and initial_tp <= 0:
            self.lg.warning(f"Calculated initial TP ({initial_tp.normalize()}) is non-positive for {self.symbol}. Setting TP to None.")
            initial_tp = None

        # Apply price precision (rounding away from entry) - CRUCIAL
        if initial_sl is not None:
             rounding = ROUND_DOWN if side == 'long' else ROUND_UP
             initial_sl = (initial_sl / self.price_tick).quantize(Decimal('1'), rounding) * self.price_tick
             # Ensure SL didn't cross entry after rounding
             if side == 'long' and initial_sl >= entry_price: initial_sl = entry_price - self.price_tick
             if side == 'short' and initial_sl <= entry_price: initial_sl = entry_price + self.price_tick
             if initial_sl <= 0: initial_sl = None # Final check after rounding

        if initial_tp is not None:
             rounding = ROUND_UP if side == 'long' else ROUND_DOWN
             initial_tp = (initial_tp / self.price_tick).quantize(Decimal('1'), rounding) * self.price_tick
             # Ensure TP didn't cross entry after rounding
             if side == 'long' and initial_tp <= entry_price: initial_tp = entry_price + self.price_tick
             if side == 'short' and initial_tp >= entry_price: initial_tp = entry_price - self.price_tick
             if initial_tp <= 0: initial_tp = None # Final check after rounding


        self.lg.debug(f"Calculated SL/TP for {self.symbol} ({side}): Entry={entry_price.normalize()}, ATR={atr.normalize()} -> SL={initial_sl.normalize() if initial_sl else 'None'}, TP={initial_tp.normalize() if initial_tp else 'None'}")
        return initial_sl, initial_tp

    def generate_signal(
        self,
        analysis: StrategyAnalysisResults,
        current_position: Optional[PositionInfo],
        symbol: str # Redundant with self.symbol but explicit
    ) -> SignalResult:
        """
        Placeholder: Analyzes strategy results and current position to generate a trading signal.

        Args:
            analysis: The results from the VolumaticOBStrategy update.
            current_position: The current open position details (or None).
            symbol: The symbol being analyzed.

        Returns:
            A SignalResult dictionary containing the signal and reasoning.
        """
        self.lg.debug(f"Generating signal for {symbol}...")

        # --- !!! Placeholder Signal Logic !!! ---
        # This needs to implement the actual entry/exit rules based on:
        # - analysis.current_trend_up
        # - analysis.last_close proximity to analysis.active_bull_boxes / active_bear_boxes
        # - analysis.trend_just_changed
        # - Whether a position is already open (current_position)

        signal = "HOLD"
        reason = "Placeholder: No entry/exit conditions met."
        initial_sl = None
        initial_tp = None

        # Example: Basic Trend Following (No OBs) - VERY Basic Placeholder
        trend_up = analysis.get('current_trend_up')
        atr = analysis.get('atr')
        last_close = analysis.get('last_close')

        if current_position is None: # No position open, look for entry
            if trend_up is True and atr and last_close:
                 signal = "BUY"
                 reason = "Placeholder: Trend is up."
                 initial_sl, initial_tp = self._calculate_initial_sl_tp(last_close, 'long', atr)
                 if initial_sl is None: # Cannot enter without SL
                      signal = "HOLD"; reason = "Placeholder: Trend up, but failed to calculate SL."
            elif trend_up is False and atr and last_close:
                 signal = "SELL"
                 reason = "Placeholder: Trend is down."
                 initial_sl, initial_tp = self._calculate_initial_sl_tp(last_close, 'short', atr)
                 if initial_sl is None: # Cannot enter without SL
                      signal = "HOLD"; reason = "Placeholder: Trend down, but failed to calculate SL."
            else:
                 reason = "Placeholder: Trend unclear or missing data for entry."

        else: # Position is open, look for exit
            pos_side = current_position.get('side')
            if pos_side == 'long' and trend_up is False:
                 signal = "EXIT_LONG"
                 reason = "Placeholder: Trend changed to down while long."
            elif pos_side == 'short' and trend_up is True:
                 signal = "EXIT_SHORT"
                 reason = "Placeholder: Trend changed to up while short."
            else:
                 reason = f"Placeholder: Holding {pos_side} position, trend aligned or unclear."

        self.lg.info(f"Signal for {symbol}: {BRIGHT}{signal}{RESET} ({reason})")
        if initial_sl or initial_tp:
            self.lg.info(f"  Calculated Entry Protections: SL={initial_sl.normalize() if initial_sl else 'N/A'}, TP={initial_tp.normalize() if initial_tp else 'N/A'}")

        return SignalResult(
            signal=signal,
            reason=reason,
            initial_sl=initial_sl,
            initial_tp=initial_tp
        )

def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy,
    signal_generator: SignalGenerator,
    market_info: MarketInfo,
    position_states: Dict[str, Dict[str, bool]] # Shared state for BE/TSL activation per symbol
):
    """
    Placeholder: Orchestrates the analysis and trading logic for a single symbol.
    Fetches data, runs analysis, generates signal, manages position, executes trades.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs error handling for each step.
    - Needs logic to manage position state transitions (e.g., setting BE/TSL flags).
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
        if df.empty or len(df) < strategy_engine.min_data_len:
            lg.warning(f"Insufficient kline data for {symbol} (got {len(df)}, need ~{strategy_engine.min_data_len}). Skipping analysis.")
            return

        # 2. Run Strategy Analysis
        analysis_results = strategy_engine.update(df)
        if analysis_results is None: # Check if update failed internally
             lg.error(f"Strategy analysis failed for {symbol}. Skipping trade logic.")
             return

        # 3. Check Current Position
        current_position = get_open_position(exchange, symbol, market_info, lg)

        # Get or initialize symbol-specific state
        position_state = position_states.setdefault(symbol, {'be_activated': False, 'tsl_activated': False})
        # Sync state if position exists and protections are set externally/persistently
        if current_position:
             # If exchange reports TSL active, ensure our state reflects that
             if current_position.get('tsl_activated'): # Use the flag we added in get_open_position
                  if not position_state['tsl_activated']:
                       lg.info(f"Detected active TSL on exchange for {symbol}. Syncing internal state.")
                       position_state['tsl_activated'] = True
             # BE state is harder to detect externally, rely on internal flag mostly

        # 4. Manage Existing Position (SL updates, BE, TSL activation)
        if current_position:
            manage_existing_position(exchange, symbol, market_info, current_position, analysis_results, position_state, lg)
            # Re-fetch position info in case management changed SL/TP/TSL state
            # current_position = get_open_position(exchange, symbol, market_info, lg)
            # We might skip re-fetch for now and assume manage_existing_position updates state internally if needed

        # 5. Generate Signal
        signal_info = signal_generator.generate_signal(analysis_results, current_position, symbol)

        # 6. Execute Trade Action
        execute_trade_action(exchange, symbol, market_info, current_position, signal_info, analysis_results, position_state, lg)

    except ccxt.AuthenticationError as e:
        # Propagate auth errors immediately to stop the bot
        lg.critical(f"{NEON_RED}Authentication Error during {symbol} processing: {e}. Stopping bot.{RESET}")
        raise e # Re-raise to be caught by main loop
    except Exception as e:
        lg.error(f"{NEON_RED}!! Unhandled error during analysis/trading cycle for {symbol}: {e} !!{RESET}", exc_info=True)
        # Continue to the next symbol

    finally:
        lg.info(f"--- Finished Analysis & Trading Cycle for: {symbol} ---")


def manage_existing_position(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    analysis_results: StrategyAnalysisResults,
    position_state: Dict[str, bool], # Internal state for BE/TSL
    logger: logging.Logger
):
    """
    Placeholder: Manages an existing open position.
    Handles logic for activating Break-Even (BE) and Trailing Stop Loss (TSL).
    (Potentially updates normal SL based on strategy rules, though not implemented here).

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs logic to check BE trigger conditions (e.g., price > entry + N * ATR).
    - Needs logic to check TSL activation conditions (e.g., price > entry + activation_%).
    - Needs to call _set_position_protection to move SL for BE or activate TSL via API.
    - Needs to update position_state flags upon successful activation.
    """
    lg = logger
    lg.debug(f"Managing existing {position_info['side']} position for {symbol}...")

    config = CONFIG # Access global config
    prot_cfg = config.get('protection', {})
    enable_be = prot_cfg.get('enable_break_even', False)
    enable_tsl = prot_cfg.get('enable_trailing_stop', False)

    # --- Break-Even Logic ---
    if enable_be and not position_state.get('be_activated', False):
        be_trigger_atr_mult = Decimal(str(prot_cfg.get('break_even_trigger_atr_multiple', 1.0)))
        be_offset_ticks = int(prot_cfg.get('break_even_offset_ticks', 2))
        atr = analysis_results.get('atr')
        entry_price_str = position_info.get('entryPrice')
        last_close = analysis_results.get('last_close')
        price_tick = market_info['price_precision_step_decimal']

        if atr and entry_price_str and last_close and price_tick and be_trigger_atr_mult > 0:
            entry_price = Decimal(str(entry_price_str))
            side = position_info['side']
            trigger_distance = atr * be_trigger_atr_mult
            be_target_price = entry_price # Default BE target is entry

            # Calculate BE offset
            offset_amount = price_tick * be_offset_ticks
            if side == 'long':
                 be_target_price += offset_amount # Move SL slightly above entry
                 be_trigger_price = entry_price + trigger_distance
                 if last_close >= be_trigger_price:
                      lg.info(f"{NEON_YELLOW}Break-Even Triggered for {symbol} (Long)! Price {last_close} >= Trigger {be_trigger_price.normalize()}{RESET}")
                      lg.info(f"  Moving SL to BE Price: {be_target_price.normalize()} (Entry {entry_price} + {be_offset_ticks} ticks)")
                      # --- Call API to move SL ---
                      success = _set_position_protection(exchange, symbol, market_info, position_info, lg, stop_loss_price=be_target_price)
                      if success:
                           position_state['be_activated'] = True # Mark BE as done for this position
                           lg.info(f"BE Stop Loss set successfully for {symbol}.")
                      else:
                           lg.error(f"Failed to set BE Stop Loss for {symbol}.")

            elif side == 'short':
                 be_target_price -= offset_amount # Move SL slightly below entry
                 be_trigger_price = entry_price - trigger_distance
                 if last_close <= be_trigger_price:
                      lg.info(f"{NEON_YELLOW}Break-Even Triggered for {symbol} (Short)! Price {last_close} <= Trigger {be_trigger_price.normalize()}{RESET}")
                      lg.info(f"  Moving SL to BE Price: {be_target_price.normalize()} (Entry {entry_price} - {be_offset_ticks} ticks)")
                      # --- Call API to move SL ---
                      success = _set_position_protection(exchange, symbol, market_info, position_info, lg, stop_loss_price=be_target_price)
                      if success:
                           position_state['be_activated'] = True
                           lg.info(f"BE Stop Loss set successfully for {symbol}.")
                      else:
                           lg.error(f"Failed to set BE Stop Loss for {symbol}.")
        # else: lg.debug(f"BE Check Skipped ({symbol}): Missing data (ATR, Entry, Close, Tick) or trigger <= 0.")


    # --- Trailing Stop Loss Activation Logic ---
    # Only activate TSL if it's enabled, not already active internally, and BE hasn't moved SL yet (optional rule)
    if enable_tsl and not position_state.get('tsl_activated', False) and not position_state.get('be_activated', False):
        # TSL activation is often based on % move from entry, or reaching a certain profit level
        # The set_trailing_stop_loss placeholder already contains some logic, but activation check might be here.
        # Let's assume set_trailing_stop_loss handles activation check and API call.
        # We just need to call it once when conditions are met.

        activation_perc = Decimal(str(prot_cfg.get('trailing_stop_activation_percentage', 0.003)))
        entry_price_str = position_info.get('entryPrice')
        last_close = analysis_results.get('last_close')

        if entry_price_str and last_close and activation_perc > 0:
             entry_price = Decimal(str(entry_price_str))
             side = position_info['side']
             activation_threshold_met = False

             if side == 'long' and last_close >= entry_price * (Decimal('1') + activation_perc):
                 activation_threshold_met = True
             elif side == 'short' and last_close <= entry_price * (Decimal('1') - activation_perc):
                 activation_threshold_met = True

             if activation_threshold_met:
                 lg.info(f"{NEON_YELLOW}Trailing Stop Activation Threshold Met for {symbol}! Activating TSL...{RESET}")
                 # Call the function to set up the TSL on the exchange
                 success = set_trailing_stop_loss(exchange, symbol, market_info, position_info, config, lg)
                 if success:
                     position_state['tsl_activated'] = True # Mark TSL as activated internally
                     lg.info(f"TSL activation successful for {symbol}.")
                 else:
                     lg.error(f"Failed to activate TSL for {symbol}.")
        # else: lg.debug(f"TSL Activation Check Skipped ({symbol}): Missing data or activation % <= 0.")

    # --- Potential Future Logic: Dynamic SL Adjustment ---
    # Could add logic here to trail the SL based on indicators (e.g., VT lower/upper band)
    # lg.debug(f"Dynamic SL adjustment logic not implemented in placeholder.")

    lg.debug(f"Finished managing position for {symbol}.")


def execute_trade_action(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    current_position: Optional[PositionInfo],
    signal_info: SignalResult,
    analysis_results: StrategyAnalysisResults,
    position_state: Dict[str, bool], # Pass state to reset on close
    logger: logging.Logger
):
    """
    Placeholder: Executes the trade action based on the generated signal.
    Handles opening new positions and closing existing ones.

    ** THIS IS A PLACEHOLDER - FULL IMPLEMENTATION IS REQUIRED **
    - Needs to check max concurrent position limits.
    - Needs to calculate position size for new entries.
    - Needs to call place_trade for market orders.
    - Needs to call _set_position_protection immediately after entry.
    - Needs to reset position_state flags when closing a position.
    """
    lg = logger
    signal = signal_info['signal']
    reason = signal_info['reason']
    config = CONFIG # Access global config
    enable_trading = config.get('enable_trading', False)
    max_positions = config.get('max_concurrent_positions', 1)

    lg.debug(f"Executing trade action '{signal}' for {symbol}. Reason: {reason}")

    # --- Exit Logic ---
    if signal == "EXIT_LONG" and current_position and current_position['side'] == 'long':
        lg.warning(f"{NEON_YELLOW}>>> Closing LONG position for {symbol} due to: {reason}{RESET}")
        if not enable_trading:
             lg.warning(f"Trading disabled. Skipping close order for {symbol}.")
             return
        close_size = current_position['size_decimal'] # Close the full position size
        if close_size > 0:
             order_result = place_trade(exchange, symbol, signal, close_size, market_info, lg, reduce_only=True)
             if order_result:
                 # Reset internal state for this symbol after successful close
                 position_state['be_activated'] = False
                 position_state['tsl_activated'] = False
                 lg.info(f"Internal state reset for {symbol} after closing position.")
             else:
                 lg.error(f"Failed to place closing order for {symbol}. State not reset.")
        else:
             lg.error(f"Cannot close {symbol}: Invalid position size ({close_size}) in position info.")

    elif signal == "EXIT_SHORT" and current_position and current_position['side'] == 'short':
        lg.warning(f"{NEON_YELLOW}>>> Closing SHORT position for {symbol} due to: {reason}{RESET}")
        if not enable_trading:
             lg.warning(f"Trading disabled. Skipping close order for {symbol}.")
             return
        close_size = abs(current_position['size_decimal']) # Use positive size for closing order amount
        if close_size > 0:
             order_result = place_trade(exchange, symbol, signal, close_size, market_info, lg, reduce_only=True)
             if order_result:
                 # Reset internal state
                 position_state['be_activated'] = False
                 position_state['tsl_activated'] = False
                 lg.info(f"Internal state reset for {symbol} after closing position.")
             else:
                 lg.error(f"Failed to place closing order for {symbol}. State not reset.")
        else:
             lg.error(f"Cannot close {symbol}: Invalid position size ({current_position['size_decimal']}) in position info.")

    # --- Entry Logic ---
    elif signal in ["BUY", "SELL"] and current_position is None:
        lg.warning(f"{NEON_GREEN}>>> Opening {signal} position for {symbol} due to: {reason}{RESET}")
        if not enable_trading:
             lg.warning(f"Trading disabled. Skipping entry order for {symbol}.")
             return

        # --- Check Max Concurrent Positions ---
        # This requires tracking positions across *all* symbols, which isn't fully implemented here.
        # Placeholder: Assume we always allow entry if no position for *this* symbol.
        # Real implementation needs a global position counter.
        current_active_positions = 0 # Placeholder - needs global tracking
        if current_active_positions >= max_positions:
             lg.warning(f"Skipping entry for {symbol}: Max concurrent positions ({max_positions}) reached.")
             return

        # --- Calculate Position Size ---
        balance_currency = config.get("quote_currency", QUOTE_CURRENCY)
        balance = fetch_balance(exchange, balance_currency, lg)
        risk_per_trade = config.get("risk_per_trade", 0.01)
        initial_sl = signal_info.get('initial_sl')
        # Use last close as approximate entry price for calculation
        entry_price_approx = analysis_results.get('last_close')

        if balance is None or initial_sl is None or entry_price_approx is None:
            lg.error(f"Cannot calculate position size for {symbol}: Missing balance, initial SL, or entry price.")
            return

        pos_size = calculate_position_size(balance, risk_per_trade, initial_sl, entry_price_approx, market_info, exchange, lg)

        if pos_size is None or pos_size <= 0:
            lg.error(f"Position size calculation failed or resulted in zero/negative size for {symbol}. Cannot place order.")
            return

        # --- Set Leverage ---
        # Leverage should ideally be set once per symbol, maybe at startup or before first trade.
        # Setting it here might be redundant or cause issues if a position exists (though we check for None).
        leverage = config.get("leverage", 0)
        if leverage > 0 and market_info.get('is_contract'):
             # Check current leverage first? Might be complex. Just try setting.
             set_leverage_success = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg)
             if not set_leverage_success:
                  lg.error(f"Failed to set leverage {leverage}x for {symbol}. Aborting entry.")
                  return
        else:
             lg.debug(f"Leverage setting skipped for {symbol} (leverage={leverage}, is_contract={market_info.get('is_contract')})")


        # --- Place Entry Order ---
        entry_order_result = place_trade(exchange, symbol, signal, pos_size, market_info, lg, reduce_only=False)

        if entry_order_result and entry_order_result.get('id'):
            # --- Set Initial SL/TP Immediately After Entry ---
            # Wait briefly for position to be potentially recognized by exchange
            confirm_delay = config.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS)
            lg.info(f"Waiting {confirm_delay}s after entry order for position confirmation...")
            time.sleep(confirm_delay)

            # Fetch the newly opened position details
            new_position = get_open_position(exchange, symbol, market_info, lg)
            if new_position:
                initial_tp = signal_info.get('initial_tp')
                lg.info(f"Setting initial protection for new {symbol} position: SL={initial_sl.normalize()}, TP={initial_tp.normalize() if initial_tp else 'None'}")
                protection_success = _set_position_protection(exchange, symbol, market_info, new_position, lg,
                                                              stop_loss_price=initial_sl,
                                                              take_profit_price=initial_tp)
                if protection_success:
                    lg.info(f"Initial SL/TP set successfully for {symbol}.")
                    # Reset internal state for the new position
                    position_state['be_activated'] = False
                    position_state['tsl_activated'] = False
                else:
                    lg.error(f"Failed to set initial SL/TP for {symbol} after entry! Manual intervention may be required.")
                    # Consider closing the position immediately if SL setup fails? Risky.
            else:
                lg.error(f"Could not confirm new position for {symbol} after entry order to set SL/TP.")
        else:
            lg.error(f"Entry order placement failed for {symbol}. No protection set.")

    elif signal == "HOLD":
        lg.info(f"Holding position or waiting for signal for {symbol}. Reason: {reason}")
    else:
        lg.debug(f"Signal '{signal}' not resulting in action for {symbol} (e.g., trying to enter while position exists).")


# --- Signal Handling & Main Loop ---

def _handle_shutdown_signal(signum, frame):
    """Sets the shutdown flag when SIGINT or SIGTERM is received."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    # Use init_logger as it's always available
    init_logger.warning(f"\n{NEON_RED}{BRIGHT}Shutdown signal ({signal_name}) received! Initiating graceful shutdown...{RESET}")
    _shutdown_requested = True

# Helper to get logger without recreating handlers
_loggers: Dict[str, logging.Logger] = {}
def get_logger_for_symbol(symbol: str) -> logging.Logger:
    """Gets or creates a logger for a specific symbol."""
    if symbol not in _loggers:
        _loggers[symbol] = setup_logger(symbol)
    return _loggers[symbol]

def main():
    """Main execution function of the bot."""
    global CONFIG, _shutdown_requested # Allow modification of shutdown flag

    main_logger = setup_logger("main")
    main_logger.info(f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Volumatic Bot v{BOT_VERSION} Starting ---{Style.RESET_ALL}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_shutdown_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown_signal) # Termination signal (e.g., systemctl stop)

    # --- Initialize Exchange ---
    exchange = initialize_exchange(main_logger)
    if not exchange:
        main_logger.critical("Exchange initialization failed. Bot cannot start. Shutting down.")
        sys.exit(1)

    # --- Validate Trading Pairs & Initialize Strategy Objects ---
    trading_pairs_config = CONFIG.get("trading_pairs", [])
    if not trading_pairs_config:
         main_logger.critical("No trading pairs configured in 'config.json'. Exiting.")
         sys.exit(1)

    valid_pairs: List[str] = []
    market_infos: Dict[str, MarketInfo] = {}
    strategy_engines: Dict[str, VolumaticOBStrategy] = {}
    signal_generators: Dict[str, SignalGenerator] = {}
    all_pairs_valid = True

    main_logger.info(f"Validating configured trading pairs: {trading_pairs_config}")
    for pair in trading_pairs_config:
        pair_logger = get_logger_for_symbol(pair) # Get logger early for validation messages
        pair_logger.info(f"Validating pair: {pair}...")
        market_info = get_market_info(exchange, pair, pair_logger)

        if market_info and market_info.get('active'):
            pair_logger.info(f" -> {NEON_GREEN}{pair} is valid and active on {exchange.id}.{RESET}")
            valid_pairs.append(pair)
            market_infos[pair] = market_info

            # Initialize strategy and signal generator for the valid pair
            try:
                strategy_engines[pair] = VolumaticOBStrategy(CONFIG, market_info, pair_logger)
                signal_generators[pair] = SignalGenerator(CONFIG, market_info, pair_logger)
                pair_logger.info(f" -> Strategy and Signal Generator initialized for {pair}.")
            except ValueError as init_err:
                # Catch specific initialization errors if raised by constructors
                pair_logger.error(f" -> {NEON_RED}Initialization error for {pair}: {init_err}. Skipping this pair.{RESET}")
                all_pairs_valid = False
                valid_pairs.remove(pair)
                market_infos.pop(pair, None)
            except Exception as init_err:
                # Catch unexpected errors during init
                pair_logger.error(f" -> {NEON_RED}Unexpected error initializing strategy/signal generator for {pair}: {init_err}. Skipping.{RESET}", exc_info=True)
                all_pairs_valid = False
                valid_pairs.remove(pair)
                market_infos.pop(pair, None)
        else:
            pair_logger.error(f" -> {NEON_RED}{pair} is invalid, inactive, or details could not be fetched. Skipping.{RESET}")
            all_pairs_valid = False

    if not valid_pairs:
        main_logger.critical("No valid trading pairs found after validation. Cannot start trading loop. Exiting.")
        sys.exit(1)

    if not all_pairs_valid:
        main_logger.warning(f"Some configured pairs were invalid. Proceeding with valid pairs: {valid_pairs}")
    else:
         main_logger.info(f"All configured pairs validated successfully: {valid_pairs}")

    if not CONFIG.get('enable_trading', False):
        main_logger.warning(f"{NEON_YELLOW}{BRIGHT}--- TRADING IS DISABLED --- (Set 'enable_trading': true in config.json to enable){RESET}")
    else:
        main_logger.warning(f"{NEON_RED}{BRIGHT}--- LIVE TRADING IS ENABLED --- Verify configuration carefully!{RESET}")


    # --- Main Trading Loop ---
    main_logger.info(f"{Fore.CYAN}### Starting Main Trading Loop ###{Style.RESET_ALL}")
    loop_count = 0
    # Initialize position state tracking dictionary (shared across iterations)
    position_states: Dict[str, Dict[str, bool]] = {
        sym: {'be_activated': False, 'tsl_activated': False} for sym in valid_pairs
    }

    while not _shutdown_requested:
        loop_count += 1
        main_logger.debug(f"--- Main Loop Cycle #{loop_count} Started ---")
        start_time = time.monotonic()

        # Process each valid symbol
        for symbol in valid_pairs:
            if _shutdown_requested: break # Check flag before processing each symbol

            symbol_logger = get_logger_for_symbol(symbol)
            # symbol_logger.info(f"--- Processing: {symbol} (Cycle #{loop_count}) ---") # Reduced verbosity

            try:
                 # Ensure we have market info (should exist from validation)
                 market_info = market_infos[symbol]

                 # Call the main analysis and trading function for the symbol
                 analyze_and_trade_symbol(
                     exchange, symbol, CONFIG, symbol_logger,
                     strategy_engines[symbol], signal_generators[symbol],
                     market_info, position_states
                 )

            except ccxt.AuthenticationError as e:
                # Critical error, stop the bot immediately
                symbol_logger.critical(f"{NEON_RED}Authentication Error during main loop for {symbol}: {e}. Stopping bot.{RESET}")
                _shutdown_requested = True # Set flag to exit loop cleanly
                break # Exit symbol loop
            except Exception as symbol_err:
                # Log errors for a specific symbol but continue the loop for others
                symbol_logger.error(f"{NEON_RED}!! Unhandled Exception in main loop for {symbol}: {symbol_err} !!{RESET}", exc_info=True)
            # finally:
                # symbol_logger.info(f"--- Finished Processing: {symbol} ---") # Reduced verbosity

            if _shutdown_requested: break # Check flag again after processing symbol

            # Small delay between symbols to avoid hitting rate limits too quickly if many pairs
            time.sleep(0.2)

        if _shutdown_requested:
            main_logger.info("Shutdown requested. Exiting main loop...")
            break # Exit main while loop

        # --- Loop Delay ---
        end_time = time.monotonic()
        cycle_duration = end_time - start_time
        loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
        wait_time = max(0, loop_delay - cycle_duration)

        main_logger.info(f"Cycle {loop_count} completed in {cycle_duration:.2f}s. Waiting {wait_time:.2f}s for next cycle...")

        # Sleep incrementally to allow faster shutdown response
        for _ in range(int(wait_time)):
             if _shutdown_requested: break
             time.sleep(1)
        if not _shutdown_requested and wait_time % 1 > 0: # Sleep remaining fraction
             time.sleep(wait_time % 1)

    # --- Shutdown Sequence ---
    main_logger.info(f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot Shutting Down Gracefully ---{Style.RESET_ALL}")
    # Add any cleanup tasks here (e.g., cancelling open orders if desired)
    # print("Cancelling open orders...") # Example cleanup task

    logging.shutdown() # Flush and close all logging handlers
    print("Shutdown complete.")
    sys.exit(0)


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Catch Ctrl+C if signal handler didn't catch it first (shouldn't happen often)
        init_logger.info("KeyboardInterrupt caught in __main__. Exiting.")
        _shutdown_requested = True # Ensure flag is set
        sys.exit(0)
    except Exception as global_err:
        # Catch any truly unexpected errors that weren't handled in main()
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL UNHANDLED EXCEPTION in __main__:{RESET} {global_err}", exc_info=True)
        logging.shutdown() # Attempt to flush logs
        sys.exit(1) # Exit with error code
