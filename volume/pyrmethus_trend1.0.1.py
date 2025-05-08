```python
# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.3.1: Comprehensive enhancements based on v1.3.0 review.
#               Improved kline fetching (multi-request), docstrings, type hinting (TypedDict),
#               error handling (retryable vs non-retryable), Decimal usage, API interaction robustness,
#               config validation, logging clarity, refactored strategy/signal logic into classes,
#               and overall code structure improvements.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5 (v1.3.1)

This bot implements a trading strategy based on the combination of:
1.  **Volumatic Trend:** A trend-following indicator using EMA/SWMA logic,
    ATR-based volatility bands calculated at trend changes, and normalized volume analysis.
2.  **Pivot Order Blocks (OBs):** Identifying potential support/resistance zones
    based on pivot highs and lows derived from candle wicks or bodies.

Key Features:
-   Connects to Bybit V5 API (Linear/Inverse contracts).
-   Supports Sandbox (testnet) and Live trading environments via `.env` config.
-   Fetches OHLCV data robustly, handling API limits (> 1000 candles) via multiple requests.
-   Calculates strategy indicators using pandas and pandas-ta, with Decimal precision.
-   Identifies Volumatic Trend direction and changes.
-   Detects Pivot Highs/Lows and creates/manages Order Blocks (Active, Violated, Extend).
-   Generates BUY/SELL/EXIT_LONG/EXIT_SHORT/HOLD signals based on trend alignment and OB proximity.
-   Calculates position size based on risk percentage, stop-loss distance, and market constraints.
-   Sets leverage for contract markets.
-   Places market orders to enter and exit positions.
-   Advanced Position Management:
    -   Sets initial Stop Loss (SL) and Take Profit (TP) based on ATR multiples.
    -   Implements Trailing Stop Loss (TSL) activation via API based on profit percentage and callback rate.
    -   Implements Break-Even (BE) stop adjustment based on ATR profit targets.
    -   *Note:* BE/TSL activation state is currently managed in memory per cycle and not persistent across bot restarts.
-   Robust API interaction with configurable retries, detailed error handling (Network, Rate Limit, Auth, Exchange-specific codes), and validation.
-   Secure handling of API credentials via `.env` file.
-   Flexible configuration via `config.json` with validation, defaults, and auto-update of missing/invalid fields.
-   Detailed logging with Neon color scheme for console output and rotating file logs (UTC timestamps).
-   Sensitive data (API keys/secrets) redaction in logs.
-   Graceful shutdown handling (Ctrl+C, SIGTERM).
-   Sequential multi-symbol trading capability.
-   Structured code using classes for Strategy and Signal Generation logic.
"""

# --- Core Libraries ---
import hashlib
import hmac
import json
import logging
import math
import os
import re # Needed for error code parsing
import signal # For SIGTERM handling
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# Use zoneinfo for modern timezone handling (requires tzdata package)
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    print(f"Warning: 'zoneinfo' module not found. Falling back to UTC. "
          f"For timezone support, ensure Python 3.9+ and install 'tzdata' (`pip install tzdata`).")
    # Simple UTC fallback if zoneinfo is not available
    class ZoneInfo: # type: ignore [no-redef]
        def __init__(self, key: str):
            if key != "UTC":
                 print(f"Requested timezone '{key}' unavailable, using UTC.")
            self._key = "UTC"
        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            if dt: return dt.replace(tzinfo=timezone.utc)
            return None
        def fromutc(self, dt: datetime) -> datetime:
            return dt.replace(tzinfo=timezone.utc)
        def utcoffset(self, dt: Optional[datetime]) -> Optional[timedelta]:
            return timedelta(0)
        def dst(self, dt: Optional[datetime]) -> Optional[timedelta]:
            return timedelta(0)
        def tzname(self, dt: Optional[datetime]) -> Optional[str]:
            return "UTC"
    class ZoneInfoNotFoundError(Exception): pass # type: ignore [no-redef]


# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy (pip install numpy)
import pandas as pd # Requires pandas (pip install pandas)
import pandas_ta as ta # Requires pandas_ta (pip install pandas_ta)
import requests # Requires requests (pip install requests)
import ccxt # Requires ccxt (pip install ccxt)
from colorama import Fore, Style, init as colorama_init # Requires colorama (pip install colorama)
from dotenv import load_dotenv # Requires python-dotenv (pip install python-dotenv)

# --- Initialize Environment and Settings ---
getcontext().prec = 28 # Set Decimal precision globally for high-accuracy calculations
colorama_init(autoreset=True) # Initialize Colorama for console colors, resetting after each print
load_dotenv() # Load environment variables from a .env file in the project root

# --- Constants ---
BOT_VERSION = "1.3.1" # <<<< Version Updated >>>>

# API Credentials (Loaded securely from .env file)
API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Critical error if API keys are missing - Use print as logger might not be ready
    print(f"{Fore.RED}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file. The arcane seals are incomplete! Exiting.{Style.RESET_ALL}")
    sys.exit(1)

# Configuration and Logging Files/Directories
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
DEFAULT_TIMEZONE_STR: str = "America/Chicago" # Default timezone if not set in .env
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    # Attempt to load user-specified timezone
    # Example: "America/Chicago", "Europe/London", "Asia/Tokyo", "UTC"
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    print(f"{Fore.RED}Timezone '{TIMEZONE_STR}' not found. Install 'tzdata' (`pip install tzdata`) or check name. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update string to reflect fallback
except Exception as tz_err:
    print(f"{Fore.RED}Failed to initialize timezone '{TIMEZONE_STR}'. Error: {tz_err}. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

# API Interaction Settings
MAX_API_RETRIES: int = 3        # Maximum number of consecutive retries for failed API calls
RETRY_DELAY_SECONDS: int = 5    # Base delay (in seconds) between API retries (increases per retry)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Wait time after placing an entry order before fetching position details to confirm
LOOP_DELAY_SECONDS: int = 15    # Default delay between trading cycles (can be overridden in config.json)
BYBIT_API_KLINE_LIMIT: int = 1000 # Maximum number of Klines Bybit V5 API returns per request

# Timeframes Mapping (Config Key to CCXT String)
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750 # Default number of klines to fetch if not specified or less than strategy needs
MAX_DF_LEN: int = 2000         # Internal limit to prevent excessive memory usage by the Pandas DataFrame

# Strategy Defaults (Used if values are missing, invalid, or out of range in config.json)
DEFAULT_VT_LENGTH: int = 40             # Volumatic Trend EMA/SWMA length
DEFAULT_VT_ATR_PERIOD: int = 200        # ATR period for Volumatic Trend bands
DEFAULT_VT_VOL_EMA_LENGTH: int = 950    # Volume Normalization EMA length (Adjusted: 1000 often > API limit)
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0  # ATR multiplier for Volumatic Trend bands
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Currently unused step ATR multiplier
DEFAULT_OB_SOURCE: str = "Wicks"        # Order Block source ("Wicks" or "Body")
DEFAULT_PH_LEFT: int = 10               # Pivot High lookback periods (left)
DEFAULT_PH_RIGHT: int = 10              # Pivot High lookback periods (right)
DEFAULT_PL_LEFT: int = 10               # Pivot Low lookback periods (left)
DEFAULT_PL_RIGHT: int = 10              # Pivot Low lookback periods (right)
DEFAULT_OB_EXTEND: bool = True          # Extend Order Block visuals to the latest candle
DEFAULT_OB_MAX_BOXES: int = 50          # Max number of *active* Order Blocks to track per type (Bull/Bear)

# Dynamically loaded from config: QUOTE_CURRENCY (e.g., "USDT")
QUOTE_CURRENCY: str = "USDT" # Placeholder, will be updated by load_config()

# Logging Colors (Neon Theme for Console Output)
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN
RESET: str = Style.RESET_ALL
BRIGHT: str = Style.BRIGHT
DIM: str = Style.DIM

# Ensure log directory exists before setting up loggers
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e:
     print(f"{NEON_RED}{BRIGHT}FATAL: Could not create log directory '{LOG_DIRECTORY}': {e}. Ensure permissions are correct.{RESET}")
     sys.exit(1)

# Global flag for shutdown signal (used by signal handler and main loop)
_shutdown_requested = False

# --- Type Definitions for Structured Data ---
class OrderBlock(TypedDict):
    """Represents a bullish or bearish Order Block identified on the chart."""
    id: str                 # Unique identifier (e.g., "B_231026143000")
    type: str               # 'bull' or 'bear'
    timestamp: pd.Timestamp # Timestamp of the candle that formed the OB (pivot candle)
    top: Decimal            # Top price level of the OB
    bottom: Decimal         # Bottom price level of the OB
    active: bool            # True if the OB is currently considered valid (not violated)
    violated: bool          # True if the price has closed beyond the OB boundary
    violation_ts: Optional[pd.Timestamp] # Timestamp when violation occurred
    extended_to_ts: Optional[pd.Timestamp] # Timestamp the OB box currently extends to (if extend=True)

class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis process."""
    dataframe: pd.DataFrame             # The DataFrame with all calculated indicators (Decimal values)
    last_close: Decimal                 # The closing price of the most recent candle (use Decimal('NaN') if invalid)
    current_trend_up: Optional[bool]    # True if Volumatic Trend is up, False if down, None if undetermined
    trend_just_changed: bool            # True if the trend flipped on the last candle
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    vol_norm_int: Optional[int]         # Normalized volume indicator (0-~200 int) for the last candle
    atr: Optional[Decimal]              # ATR value for the last candle (must be positive Decimal if valid)
    upper_band: Optional[Decimal]       # Volumatic Trend upper band value for the last candle
    lower_band: Optional[Decimal]       # Volumatic Trend lower band value for the last candle

class MarketInfo(TypedDict):
    """Standardized market information dictionary derived from ccxt.market."""
    # Standard CCXT fields (may vary slightly by exchange)
    id: str                     # Exchange-specific market ID (e.g., 'BTCUSDT')
    symbol: str                 # Standardized symbol (e.g., 'BTC/USDT')
    base: str                   # Base currency code (e.g., 'BTC')
    quote: str                  # Quote currency code (e.g., 'USDT')
    settle: Optional[str]       # Settlement currency (usually quote for linear, base for inverse)
    baseId: str                 # Exchange-specific base ID
    quoteId: str                # Exchange-specific quote ID
    settleId: Optional[str]     # Exchange-specific settle ID
    type: str                   # 'spot', 'swap', 'future', etc.
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool                # Whether the market is currently active/tradable
    contract: bool              # True if it's a derivative contract (swap, future)
    linear: Optional[bool]      # True if linear contract (presence varies)
    inverse: Optional[bool]     # True if inverse contract (presence varies)
    quanto: Optional[bool]      # True if quanto contract (presence varies)
    taker: float                # Taker fee rate
    maker: float                # Maker fee rate
    contractSize: Optional[Any] # Size of one contract (often float or int, convert to Decimal)
    expiry: Optional[int]       # Unix timestamp of expiry (milliseconds)
    expiryDatetime: Optional[str]# ISO8601 datetime string of expiry
    strike: Optional[float]     # Option strike price
    optionType: Optional[str]   # 'call' or 'put'
    precision: Dict[str, Any]   # {'amount': float/str, 'price': float/str, 'cost': float/str, 'base': float, 'quote': float} - Source for Decimal steps
    limits: Dict[str, Any]      # {'leverage': {'min': float, 'max': float}, 'amount': {'min': float/str, 'max': float/str}, 'price': {'min': float, 'max': float}, 'cost': {'min': float/str, 'max': float/str}} - Source for Decimal limits
    info: Dict[str, Any]        # Exchange-specific raw market info
    # Custom added fields for convenience and robustness
    is_contract: bool           # Reliable check for derivatives (True if swap, future, or contract=True)
    is_linear: bool             # True only if linear contract AND is_contract=True
    is_inverse: bool            # True only if inverse contract AND is_contract=True
    contract_type_str: str      # "Linear", "Inverse", "Spot", or "Unknown" for logging/logic
    min_amount_decimal: Optional[Decimal] # Parsed minimum order size (in base units or contracts), non-negative
    max_amount_decimal: Optional[Decimal] # Parsed maximum order size, positive if set
    min_cost_decimal: Optional[Decimal]   # Parsed minimum order cost (in quote currency), non-negative
    max_cost_decimal: Optional[Decimal]   # Parsed maximum order cost, positive if set
    amount_precision_step_decimal: Optional[Decimal] # Parsed step size for amount (e.g., 0.001), must be positive
    price_precision_step_decimal: Optional[Decimal]  # Parsed step size for price (e.g., 0.01), must be positive
    contract_size_decimal: Decimal  # Parsed contract size as Decimal (must be positive, defaults to 1 if not applicable/found)

class PositionInfo(TypedDict):
    """Standardized position information dictionary derived from ccxt.position."""
    # Standard CCXT fields (availability varies by exchange)
    id: Optional[str]           # Position ID (often None or same as symbol)
    symbol: str                 # Standardized symbol (e.g., 'BTC/USDT')
    timestamp: Optional[int]    # Creation timestamp (milliseconds)
    datetime: Optional[str]     # ISO8601 creation datetime string
    contracts: Optional[float]  # Deprecated/inconsistent, use size_decimal instead
    contractSize: Optional[Any] # Size of one contract for this position (convert to Decimal)
    side: Optional[str]         # 'long' or 'short' (parsed/validated)
    notional: Optional[Any]     # Position value in quote currency (convert to Decimal)
    leverage: Optional[Any]     # Leverage used for this position (convert to Decimal)
    unrealizedPnl: Optional[Any]# Unrealized Profit/Loss (convert to Decimal)
    realizedPnl: Optional[Any]  # Realized Profit/Loss (convert to Decimal)
    collateral: Optional[Any]   # Collateral used (convert to Decimal)
    entryPrice: Optional[Any]   # Average entry price (convert to Decimal)
    markPrice: Optional[Any]    # Current mark price (convert to Decimal)
    liquidationPrice: Optional[Any] # Estimated liquidation price (convert to Decimal)
    marginMode: Optional[str]   # 'cross' or 'isolated'
    hedged: Optional[bool]      # Whether the position is part of a hedge
    maintenanceMargin: Optional[Any] # Maintenance margin required (convert to Decimal)
    maintenanceMarginPercentage: Optional[float] # Maintenance margin rate
    initialMargin: Optional[Any]# Initial margin used (convert to Decimal)
    initialMarginPercentage: Optional[float] # Initial margin rate
    marginRatio: Optional[float]# Margin ratio (maintenance / collateral)
    lastUpdateTimestamp: Optional[int] # Timestamp of last update (milliseconds)
    info: Dict[str, Any]        # Exchange-specific raw position info (crucial for Bybit V5 details like SL/TP/TSL strings)
    # Custom added/parsed fields
    size_decimal: Decimal       # Parsed position size as Decimal (positive for long, negative for short, non-zero for active position)
    stopLossPrice: Optional[str]# Parsed SL price from info (string format from Bybit, non-zero if set)
    takeProfitPrice: Optional[str]# Parsed TP price from info (string format from Bybit, non-zero if set)
    trailingStopLoss: Optional[str]# Parsed TSL distance from info (string format from Bybit, non-zero if set)
    tslActivationPrice: Optional[str]# Parsed TSL activation price from info (string format from Bybit, non-zero if set)
    # Custom flags for bot state tracking (IN-MEMORY ONLY, NOT PERSISTENT)
    be_activated: bool          # True if Break-Even has been set for this position instance by the bot
    tsl_activated: bool         # True if Trailing Stop Loss has been set for this position instance by the bot


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive API keys/secrets
    from log messages to prevent accidental exposure in log files or console.
    """
    _api_key_placeholder = "***API_KEY***"
    _api_secret_placeholder = "***API_SECRET***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing API keys/secrets with placeholders."""
        msg = super().format(record)
        # Ensure API keys exist and are strings before attempting replacement
        key = API_KEY; secret = API_SECRET
        try:
            if key and isinstance(key, str) and key in msg:
                msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg:
                msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception:
            # Ignore potential errors during redaction (e.g., if msg is not string-like)
            pass
        return msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a dedicated logger instance for a specific context (e.g., 'init', 'BTC/USDT').

    Configures both a console handler (with Neon colors, level filtering based on
    CONSOLE_LOG_LEVEL environment variable, and timezone-aware timestamps)
    and a rotating file handler (capturing DEBUG level and above, using UTC timestamps,
    with sensitive data redaction).

    Args:
        name (str): The name for the logger (e.g., "init", "BTC/USDT"). Used for filtering
                    and naming the log file.

    Returns:
        logging.Logger: The configured logging.Logger instance. Returns existing instance if already configured.
    """
    safe_name = name.replace('/', '_').replace(':', '-') # Sanitize name for filenames/logger keys
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger instance already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels; handlers below will filter output

    # --- File Handler (DEBUG level, Rotating, Redaction, UTC Timestamps) ---
    try:
        # Rotate log file when it reaches 10MB, keep 5 backup files
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Use SensitiveFormatter for detailed file log, redacting API keys/secrets
        # Include milliseconds in file log timestamps
        ff = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S' # Standard date format for files
        )
        ff.converter = time.gmtime # type: ignore # Use UTC time for file logs
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG) # Log everything from DEBUG upwards to the file
        logger.addHandler(fh)
    except Exception as e:
        # Use print for errors during logger setup itself, as logger might not be functional
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler (Configurable Level, Neon Colors, Local Timezone Timestamps) ---
    try:
        sh = logging.StreamHandler(sys.stdout) # Explicitly use stdout
        # Define color mapping for different log levels
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,      # Dim Cyan for Debug
            logging.INFO: NEON_BLUE,             # Bright Cyan for Info
            logging.WARNING: NEON_YELLOW,        # Bright Yellow for Warning
            logging.ERROR: NEON_RED,             # Bright Red for Error
            logging.CRITICAL: NEON_RED + BRIGHT, # Bright Red + Bold for Critical
        }

        # Custom formatter for console output with colors and timezone-aware timestamps
        class NeonConsoleFormatter(SensitiveFormatter):
            """Applies Neon color scheme and configured timezone to console log messages."""
            _level_colors = level_colors
            _tz = TIMEZONE # Use the globally configured timezone object

            def format(self, record: logging.LogRecord) -> str:
                level_color = self._level_colors.get(record.levelno, NEON_BLUE) # Default to Info color
                # Format: Time(Local) - Level - [LoggerName] - Message
                log_fmt = (
                    f"{NEON_BLUE}%(asctime)s{RESET} - "
                    f"{level_color}%(levelname)-8s{RESET} - "
                    f"{NEON_PURPLE}[%(name)s]{RESET} - "
                    f"%(message)s"
                )
                # Create a formatter instance with the defined format and date style
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S') # Use Time only for console clarity
                # Ensure timestamps reflect the configured TIMEZONE
                formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore
                # Apply sensitive data redaction before returning the final message
                # We inherit from SensitiveFormatter, so super().format handles redaction
                # Need explicit super() call for nested classes
                return super(NeonConsoleFormatter, self).format(record)

        sh.setFormatter(NeonConsoleFormatter())
        # Get desired console log level from environment variable (e.g., DEBUG, INFO, WARNING), default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO) # Fallback to INFO if invalid level provided
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False # Prevent log messages from bubbling up to the root logger
    return logger

# Initialize the 'init' logger early for messages during startup and configuration loading
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}Pyrmethus Volumatic Bot v{BOT_VERSION} awakening...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR}")

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively checks if all keys from `default_config` exist in `config`.
    If a key is missing, it's added to `config` with the default value from `default_config`.
    Logs any keys that were added using the `init_logger`.

    Args:
        config (Dict[str, Any]): The configuration dictionary loaded from the file.
        default_config (Dict[str, Any]): The dictionary containing default structure and values.
        parent_key (str): Used internally for tracking nested key paths for logging (e.g., "strategy_params.vt_length").

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing:
            - The potentially updated configuration dictionary.
            - A boolean indicating whether any changes were made (True if keys were added).
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Spell: Added missing parameter '{full_key_path}' with default enchantment: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # If both default and loaded values are dicts, recurse into nested dict
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                # If nested dict was changed, update the parent dict and mark as changed
                updated_config[key] = nested_config
                changed = True
        # Optional: Could add type mismatch check here, but validation below handles it more robustly.
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads, validates, and potentially updates configuration from a JSON file.

    Steps:
    1. Checks if the config file exists. If not, creates a default one.
    2. Loads the JSON data from the file. Handles decoding errors by recreating the default file.
    3. Ensures all keys from the default structure exist in the loaded config, adding missing ones.
    4. Performs detailed type and range validation on critical numeric and string parameters.
       - Uses default values and logs warnings/corrections if validation fails.
       - Leverages Decimal for robust numeric comparisons.
    5. Validates the `trading_pairs` list.
    6. If any keys were added or values corrected, saves the updated config back to the file.
    7. Updates the global `QUOTE_CURRENCY` based on the validated config.
    8. Returns the validated (and potentially updated) configuration dictionary.

    Args:
        filepath (str): The path to the configuration JSON file (e.g., "config.json").

    Returns:
        Dict[str, Any]: The loaded and validated configuration dictionary. Returns default configuration
                        if the file cannot be read, created, or parsed, or if validation encounters
                        unexpected errors. Returns the internal default if file recreation fails.
    """
    init_logger.info(f"{Fore.CYAN}# Conjuring configuration from '{filepath}'...{Style.RESET_ALL}")
    # Define the default configuration structure and values
    default_config = {
        # General Settings
        "trading_pairs": ["BTC/USDT"],  # List of symbols to trade (e.g., ["BTC/USDT", "ETH/USDT"])
        "interval": "5",                # Default timeframe (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS, # Base delay for API retries
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Default klines to fetch per cycle
        "orderbook_limit": 25,          # (Currently Unused) Limit for order book fetching if implemented
        "enable_trading": False,        # Master switch for placing actual trades
        "use_sandbox": True,            # Use Bybit's testnet environment
        "risk_per_trade": 0.01,         # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                 # Default leverage for contract trading (0 to disable setting)
        "max_concurrent_positions": 1,  # (Currently Unused) Max open positions allowed simultaneously
        "quote_currency": "USDT",       # The currency to calculate balance and risk against
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay between trading cycles
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait after order before checking position

        # Strategy Parameters (Volumatic Trend + OB)
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER), # Store as float in JSON
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER), # Unused, store as float
            "ob_source": DEFAULT_OB_SOURCE, # "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT, # Pivot High lookbacks
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT, # Pivot Low lookbacks
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005, # Price must be <= OB top * factor (long) or >= OB bottom / factor (short)
            "ob_exit_proximity_factor": 1.001   # Exit if price >= Bear OB top / factor or <= Bull OB bottom * factor
        },
        # Position Protection Parameters
        "protection": {
             "enable_trailing_stop": True,      # Use trailing stop loss
             "trailing_stop_callback_rate": 0.005, # TSL distance as % of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Activate TSL when price moves this % from entry (e.g., 0.003 = 0.3%)
             "enable_break_even": True,         # Move SL to entry + offset when profit target hit
             "break_even_trigger_atr_multiple": 1.0, # Profit needed (in ATR multiples) to trigger BE
             "break_even_offset_ticks": 2,       # Move SL this many ticks beyond entry for BE
             "initial_stop_loss_atr_multiple": 1.8, # Initial SL distance in ATR multiples
             "initial_take_profit_atr_multiple": 0.7 # Initial TP distance in ATR multiples (0 to disable)
        }
    }
    config_needs_saving: bool = False
    loaded_config: Dict[str, Any] = {}

    # --- File Existence Check & Default Creation ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config scroll '{filepath}' not found. Crafting a default scroll.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Crafted default config scroll: {filepath}{RESET}")
            # Update global QUOTE_CURRENCY immediately after creating default
            global QUOTE_CURRENCY
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Return defaults immediately
        except IOError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error crafting default config scroll '{filepath}': {e}. The weave is broken!{RESET}")
            init_logger.warning("Using internal default configuration runes. Bot may falter.")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Use internal defaults if file creation fails

    # --- File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
             raise TypeError("Configuration scroll does not contain a valid arcane map (JSON object).")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error deciphering JSON from '{filepath}': {e}. Recrafting default scroll.{RESET}")
        try: # Attempt to recreate the file with defaults if corrupted
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recrafted default config scroll due to corruption: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}FATAL: Error recrafting default config scroll after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected rift loading config scroll '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.warning("Using internal default configuration runes. Bot may falter.")
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config

    # --- Validation and Merging ---
    try:
        # Ensure all keys from default_config exist in loaded_config
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True # Mark for saving if keys were added

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            """
            Validates a numeric config value at `key_path` (e.g., "protection.leverage").

            Checks type (int/float), range [min_val, max_val] or (min_val, max_val] if strict.
            Uses the default value from `default_config` and logs a warning/info if correction needed.
            Updates the `cfg` dictionary in place if correction is made.
            Uses Decimal for robust comparisons.

            Args:
                cfg (Dict): The config dictionary being validated (updated in place).
                key_path (str): The dot-separated path to the key (e.g., "protection.leverage").
                min_val (Union[int, float, Decimal]): Minimum allowed value (inclusive unless is_strict_min).
                max_val (Union[int, float, Decimal]): Maximum allowed value (inclusive).
                is_strict_min (bool): If True, value must be strictly greater than min_val.
                is_int (bool): If True, value must be an integer.
                allow_zero (bool): If True, zero is allowed even if outside min/max range.

            Returns:
                bool: True if a correction was made, False otherwise.
            """
            nonlocal config_needs_saving # Allow modification of the outer scope variable
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                # Traverse nested dictionaries to reach the target key
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key)
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid path '{key_path}'. Cannot validate.")
                return False # Path itself is wrong

            if original_val is None:
                # This case should be rare due to _ensure_config_keys, but handle defensively
                init_logger.warning(f"Config validation: Rune missing at '{key_path}'. Using default: {repr(default_val)}")
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val # Start with the original value

            try:
                # Convert to Decimal for robust comparison, handle potential strings
                num_val = Decimal(str(original_val))
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))

                # Check range
                min_check = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check = min_check and num_val <= max_dec
                # Check if zero is allowed and value is zero, bypassing range check if so
                zero_ok = allow_zero and num_val == Decimal(0)

                if not range_check and not zero_ok:
                    raise ValueError("Value outside allowed arcane boundaries.")

                # Check type and convert if necessary
                target_type = int if is_int else float
                # Attempt conversion to target type
                converted_val = target_type(num_val)

                # Check if type or value changed significantly after conversion
                # This ensures int remains int, float remains float (within tolerance)
                needs_correction = False
                if isinstance(original_val, bool): # Don't try to convert bools here
                     raise TypeError("Boolean value found where numeric essence expected.")
                elif is_int and not isinstance(original_val, int):
                    needs_correction = True
                elif not is_int and not isinstance(original_val, float):
                     # If float expected, allow int input but convert it
                    if isinstance(original_val, int):
                        converted_val = float(original_val) # Explicitly convert int to float
                        needs_correction = True
                    else: # Input is neither float nor int (e.g., string)
                        needs_correction = True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9:
                    # Check if float value changed significantly after potential Decimal conversion
                    needs_correction = True
                elif isinstance(original_val, int) and original_val != converted_val:
                     # Should not happen if is_int=True, but check defensively
                     needs_correction = True

                if needs_correction:
                    init_logger.info(f"{NEON_YELLOW}Config Spell: Corrected essence/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
                    final_val = converted_val
                    corrected = True

            except (ValueError, InvalidOperation, TypeError) as e:
                # Handle cases where value is non-numeric, out of range, or conversion fails
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                if allow_zero: range_str += " or 0"
                init_logger.warning(f"{NEON_YELLOW}Config rune '{key_path}': Invalid value '{repr(original_val)}'. Using default: {repr(default_val)}. Error: {e}. Expected: {'integer' if is_int else 'float'}, Boundaries: {range_str}{RESET}")
                final_val = default_val # Use the default value
                corrected = True

            # If a correction occurred, update the config dictionary and mark for saving
            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        init_logger.debug("# Scrutinizing configuration runes...")
        # --- Apply Validations to Specific Config Keys ---
        # General
        if not isinstance(updated_config.get("trading_pairs"), list) or \
           not all(isinstance(s, str) and s for s in updated_config.get("trading_pairs", [])):
            init_logger.warning(f"{NEON_YELLOW}Invalid 'trading_pairs'. Must be list of non-empty strings. Using default {default_config['trading_pairs']}.{RESET}")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"{NEON_YELLOW}Invalid 'interval' '{updated_config.get('interval')}'. Valid: {VALID_INTERVALS}. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True) # Ensure minimum useful fetch limit
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('1'), is_strict_min=True) # Risk must be > 0 and <= 1
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Leverage 0 means no setting attempt
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
             init_logger.warning(f"Invalid 'quote_currency'. Must be non-empty string. Using default '{default_config['quote_currency']}'.")
             updated_config["quote_currency"] = default_config["quote_currency"]
             config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool):
             init_logger.warning(f"Invalid 'enable_trading'. Must be true/false. Using default '{default_config['enable_trading']}'.")
             updated_config["enable_trading"] = default_config["enable_trading"]
             config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
             init_logger.warning(f"Invalid 'use_sandbox'. Must be true/false. Using default '{default_config['use_sandbox']}'.")
             updated_config["use_sandbox"] = default_config["use_sandbox"]
             config_needs_saving = True

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True) # Allow long ATR period
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True) # Allow long Vol EMA
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # e.g., 1.005 = 0.5% proximity
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # e.g., 1.001 = 0.1% proximity
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
             config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
             init_logger.warning(f"Invalid strategy_params.ob_extend. Must be true/false. Using default '{DEFAULT_OB_EXTEND}'.")
             updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND
             config_needs_saving = True

        # Protection Params
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool):
             init_logger.warning(f"Invalid protection.enable_trailing_stop. Must be true/false. Using default '{default_config['protection']['enable_trailing_stop']}'.")
             updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]
             config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
             init_logger.warning(f"Invalid protection.enable_break_even. Must be true/false. Using default '{default_config['protection']['enable_break_even']}'.")
             updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]
             config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.5'), is_strict_min=True) # Must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True) # 0 means activate immediately
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0'))
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # 0 means move SL exactly to entry
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('100.0'), is_strict_min=True) # Must be > 0
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('100.0'), allow_zero=True) # 0 disables initial TP

        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             try:
                 # json.dumps handles basic types (int, float, str, bool, list, dict)
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Inscribed updated configuration runes to scroll: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error inscribing updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with corrected runes in memory, but scroll update failed.")

        # Update the global QUOTE_CURRENCY from the validated config
        global QUOTE_CURRENCY
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency focus set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(f"{Fore.CYAN}# Configuration conjuration complete.{Style.RESET_ALL}")

        return updated_config # Return the validated and potentially corrected config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected vortex during configuration processing: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config # Fallback to defaults on unexpected error

# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)
# QUOTE_CURRENCY is updated inside load_config()

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and validates the CCXT Bybit exchange object.

    Steps:
    1. Sets API keys, rate limiting, default type (linear), timeouts.
    2. Configures sandbox mode based on `config.json`.
    3. Loads exchange markets with retries, ensuring markets are actually populated.
    4. Performs an initial balance check for the configured `QUOTE_CURRENCY`.
       - If trading is enabled, a failed balance check is treated as a fatal error.
       - If trading is disabled, logs a warning but allows proceeding.

    Args:
        logger (logging.Logger): The logger instance to use for status messages.

    Returns:
        Optional[ccxt.Exchange]: The initialized ccxt.Exchange object if successful, otherwise None.
    """
    lg = logger # Alias for convenience
    lg.info(f"{Fore.CYAN}# Binding the arcane energies to the Bybit exchange...{Style.RESET_ALL}")
    try:
        # Common CCXT exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',         # Assume linear contracts by default
                'adjustForTimeDifference': True, # Auto-adjust for clock skew
                # Timeouts for various operations (in milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 30000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'fetchOHLCVTimeout': 60000,      # Longer timeout for potentially large kline fetches
            }
        }
        # Instantiate the Bybit exchange object
        exchange = ccxt.bybit(exchange_options)

        # Configure Sandbox Mode
        is_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< OPERATING IN SANDBOX REALM (Testnet Environment) >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< OPERATING IN LIVE REALM - REAL ASSETS AT STAKE >>> !!!{RESET}")

        # Load Markets with Retries
        lg.info(f"Summoning market knowledge for {exchange.id}...")
        markets_loaded = False
        last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market summon attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
                # Force reload on retries to ensure fresh market data
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market knowledge summoned successfully ({len(exchange.markets)} symbols charted).{RESET}")
                    markets_loaded = True
                    break # Exit retry loop on success
                else:
                    last_market_error = ValueError("Market summoning returned an empty void")
                    lg.warning(f"Market summoning returned empty void (Attempt {attempt + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"Aetheric disturbance (Network Error) summoning markets (Attempt {attempt + 1}): {e}.")
                if attempt >= MAX_API_RETRIES:
                    lg.critical(f"{NEON_RED}Max retries exceeded summoning markets. Last echo: {last_market_error}. Binding failed.{RESET}")
                    return None
            except ccxt.AuthenticationError as e:
                 last_market_error = e
                 lg.critical(f"{NEON_RED}Authentication ritual failed: {e}. Check API seals. Binding failed.{RESET}")
                 return None
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected rift summoning markets: {e}. Binding failed.{RESET}", exc_info=True)
                return None

            # Apply delay before retrying
            if not markets_loaded and attempt < MAX_API_RETRIES:
                 delay = RETRY_DELAY_SECONDS * (attempt + 1) # Increase delay per attempt
                 lg.warning(f"Retrying market summon in {delay}s...")
                 time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to summon markets after all attempts. Last echo: {last_market_error}. Binding failed.{RESET}")
            return None

        lg.info(f"Exchange binding established: {exchange.id} | Sandbox Realm: {is_sandbox}")

        # Initial Balance Check
        lg.info(f"Scrying initial balance for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try:
            initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err:
            # Handle auth errors specifically here as they are critical during balance check
            lg.critical(f"{NEON_RED}Authentication Ritual Failed during balance scrying: {auth_err}. Binding failed.{RESET}")
            return None
        except Exception as balance_err:
             # Catch other potential errors during the initial balance check
             lg.warning(f"{NEON_YELLOW}Initial balance scrying encountered a flicker: {balance_err}.{RESET}", exc_info=False)
             # Let the logic below decide based on trading enabled status

        # Evaluate balance check result based on trading mode
        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial available essence: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange binding complete and validated.{Style.RESET_ALL}")
            return exchange # Success!
        else:
            # Balance fetch failed (fetch_balance logs the failure reason)
            lg.error(f"{NEON_RED}Initial balance scrying FAILED for {QUOTE_CURRENCY}.{RESET}")
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading rituals enabled, but balance scrying failed. Cannot proceed safely. Binding failed.{RESET}")
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading rituals disabled. Proceeding without confirmed balance, but spells may falter.{RESET}")
                lg.info(f"{Fore.CYAN}# Exchange binding complete (balance unconfirmed).{Style.RESET_ALL}")
                return exchange # Allow proceeding in non-trading mode

    except Exception as e:
        # Catch-all for errors during the initialization process itself
        lg.critical(f"{NEON_RED}Failed to bind to CCXT exchange: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers ---
def _safe_market_decimal(value: Optional[Any], field_name: str, allow_zero: bool = True) -> Optional[Decimal]:
    """
    Safely converts a market info value (potentially str, float, int) to Decimal.

    Handles None, empty strings, and invalid numeric formats.
    Logs debug messages for conversion issues.

    Args:
        value (Optional[Any]): The value to convert.
        field_name (str): The name of the field being converted (for logging).
        allow_zero (bool): Whether a value of zero is considered valid.

    Returns:
        Optional[Decimal]: The converted Decimal value, or None if conversion fails
                           or the value is invalid according to `allow_zero`.
    """
    if value is None:
        return None
    try:
        s_val = str(value).strip()
        if not s_val:
            return None
        d_val = Decimal(s_val)
        # Check positivity/zero constraints
        if not allow_zero and d_val <= Decimal('0'):
             return None # Must be strictly positive
        if d_val < Decimal('0'):
             return None # Must be non-negative if zero is allowed
        return d_val
    except (InvalidOperation, TypeError, ValueError):
        # init_logger.debug(f"Could not convert market info field '{field_name}' value '{value}' to Decimal.") # Too verbose
        return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price to the exchange's required precision string using `price_to_precision`.

    Handles Decimal, float, or string input. Ensures the input price is positive.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The market symbol (e.g., 'BTC/USDT').
        price (Union[Decimal, float, str]): The price value to format.

    Returns:
        Optional[str]: The formatted price string suitable for API calls, or None if formatting fails
                       or the input price is invalid (non-positive).
    """
    try:
        price_decimal = Decimal(str(price))
        if price_decimal <= 0:
            # init_logger.debug(f"Price formatting skipped: Input price {price_decimal} is not positive.") # Potentially verbose
            return None # Price must be positive

        # Use CCXT's helper for correct rounding/truncating based on market precision rules
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))

        # Final check: ensure formatted price is still positive after formatting (e.g., didn't round down to 0)
        if Decimal(formatted_str) > 0:
            return formatted_str
        else:
            # init_logger.debug(f"Formatted price '{formatted_str}' resulted in zero or negative value. Ignoring.")
            return None
    except (InvalidOperation, ValueError, TypeError, KeyError, AttributeError) as e:
        init_logger.warning(f"Error formatting price {price} for {symbol}: {e}")
        return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using `fetch_ticker` with fallbacks.

    Prioritizes 'last' price. Falls back progressively:
    1. Mid-price ((bid + ask) / 2) if both bid and ask are valid.
    2. 'ask' price if only ask is valid.
    3. 'bid' price if only bid is valid.

    Includes retry logic for network errors and rate limits. Handles AuthenticationError critically.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The current price as a Decimal, or None if fetching fails after retries
                           or a non-retryable error occurs.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching current price pulse for {symbol} (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None
            source = "N/A" # Track which price source was used

            # Helper to safely convert ticker values to positive Decimal
            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                """Safely converts ticker field to positive Decimal."""
                if value is None: return None
                try:
                    s_val = str(value).strip()
                    if not s_val: return None
                    dec_val = Decimal(s_val)
                    return dec_val if dec_val > Decimal('0') else None
                except (ValueError, InvalidOperation, TypeError):
                    lg.debug(f"Could not parse ticker field '{field_name}' value '{value}' to Decimal.")
                    return None

            # 1. Try 'last' price
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')
            if price: source = "'last' price"

            # 2. Fallback to mid-price if 'last' is invalid
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask:
                    price = (bid + ask) / Decimal('2')
                    source = f"mid-price (B:{bid.normalize()}, A:{ask.normalize()})"
                # 3. Fallback to 'ask' if only ask is valid
                elif ask:
                    price = ask
                    source = f"'ask' price ({ask.normalize()})"
                # 4. Fallback to 'bid' if only bid is valid
                elif bid:
                    price = bid
                    source = f"'bid' price ({bid.normalize()})"

            # Check if a valid price was obtained
            if price:
                lg.debug(f"Price pulse captured ({symbol}) via {source}: {price.normalize()}")
                return price.normalize() # Ensure normalization
            else:
                last_exception = ValueError(f"No valid price found in ticker (last, mid, ask, bid). Ticker: {ticker}")
                lg.warning(f"No valid price pulse ({symbol}, Attempt {attempts + 1}). Retrying...")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit fetching price ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempts for rate limit, just wait and retry
            continue
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Auth ritual failed fetching price: {e}. Stopping.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange rift fetching price ({symbol}): {e}{RESET}")
            # Could add checks for specific non-retryable error codes here if needed
            # For now, assume potentially retryable unless it's an auth error
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex fetching price ({symbol}): {e}{RESET}", exc_info=True)
            return None # Exit on unexpected errors

        # Increment attempt counter and apply delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to capture price pulse ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV (kline) data using CCXT's `fetch_ohlcv` method, handling pagination.

    - Fetches historical data in chunks going backward in time until the `limit` is reached
      or no more data is available.
    - Handles Bybit V5 'category' parameter automatically based on market info.
    - Implements robust retry logic per chunk for network errors and rate limits.
    - Validates timestamp lag of the most recent chunk to detect potential staleness.
    - Processes combined data into a Pandas DataFrame with Decimal types for precision.
    - Cleans data (drops rows with NaNs in key columns, zero prices/volumes).
    - Trims DataFrame to `MAX_DF_LEN` to manage memory usage.
    - Ensures DataFrame is sorted by timestamp.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        timeframe (str): The CCXT timeframe string (e.g., "5m", "1h", "1d").
        limit (int): The desired total number of klines to fetch.
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the OHLCV data, indexed by timestamp (UTC),
                      with columns ['open', 'high', 'low', 'close', 'volume'] as Decimals.
                      Returns an empty DataFrame if fetching or processing fails critically.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Gathering historical echoes (Klines) for {symbol} | TF: {timeframe} | Target Limit: {limit}...{Style.RESET_ALL}")
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    # Calculate minimum required candles for strategy (rough estimate for logging)
    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        min_required = max(sp.get('vt_length', 0)*2, sp.get('vt_atr_period', 0), sp.get('vt_vol_ema_length', 0),
                           sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1,
                           sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1) + 50 # Add buffer
        lg.debug(f"Estimated minimum candles required by strategy: {min_required}")
        if limit < min_required:
            lg.warning(f"{NEON_YELLOW}Requested limit ({limit}) is less than estimated strategy requirement ({min_required}). Indicator accuracy may be affected.{RESET}")
    except Exception as e:
        lg.warning(f"Could not estimate minimum required candles: {e}")

    # Determine category and market ID for Bybit V5
    category = 'spot' # Default
    market_id = symbol # Default
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        lg.debug(f"Using Bybit category: {category} and Market ID: {market_id} for kline fetch.")
    except KeyError:
        lg.warning(f"Market '{symbol}' not found in loaded markets during kline fetch setup. Using defaults.")
    except Exception as e:
        lg.warning(f"Could not determine market category/ID for {symbol} kline fetch: {e}. Using defaults.")

    all_ohlcv_data: List[List[Union[int, float, str]]] = []
    remaining_limit = limit
    end_timestamp_ms: Optional[int] = None # Fetch going backwards from current time (most recent first)
    max_chunks = math.ceil(limit / BYBIT_API_KLINE_LIMIT) + 2 # Allow a couple extra chunks for safety
    chunk_num = 0

    # --- Fetching Loop (Handles Pagination) ---
    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT)
        lg.debug(f"Fetching chunk {chunk_num}/{max_chunks} ({fetch_size} klines) for {symbol}. Target remaining: {remaining_limit}. End TS: {end_timestamp_ms}")

        attempts = 0
        last_exception = None
        chunk_data: Optional[List[List[Union[int, float, str]]]] = None

        # --- Retry Loop (Per Chunk) ---
        while attempts <= MAX_API_RETRIES:
            try:
                params = {'category': category} if 'bybit' in exchange.id.lower() else {}
                # CCXT handles the 'until' parameter based on end_timestamp_ms if supported
                # If 'until' isn't directly supported, it might fetch most recent - check CCXT docs per exchange if needed
                fetch_args: Dict[str, Any] = {
                    'symbol': symbol, # Use standard symbol for fetch_ohlcv
                    'timeframe': timeframe,
                    'limit': fetch_size,
                    'params': params
                }
                if end_timestamp_ms:
                    fetch_args['until'] = end_timestamp_ms # Fetch candles *before* this timestamp

                chunk_data = exchange.fetch_ohlcv(**fetch_args)
                fetched_count = len(chunk_data) if chunk_data else 0
                lg.debug(f"API returned {fetched_count} candles for chunk {chunk_num} (requested {fetch_size}).")

                if chunk_data:
                    # --- Basic Validation (Timestamp Lag Check on First Chunk) ---
                    if chunk_num == 1: # Only check lag on the most recent chunk
                        try:
                            last_candle_timestamp_ms = chunk_data[-1][0]
                            last_ts = pd.to_datetime(last_candle_timestamp_ms, unit='ms', utc=True)
                            now_utc = pd.Timestamp.utcnow()
                            interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                max_allowed_lag = interval_seconds * 2.5
                                actual_lag = (now_utc - last_ts).total_seconds()
                                if actual_lag > max_allowed_lag:
                                    last_exception = ValueError(f"Kline data potentially stale (Lag: {actual_lag:.1f}s > Max: {max_allowed_lag:.1f}s).")
                                    lg.warning(f"{NEON_YELLOW}Timestamp lag detected ({symbol}, Chunk 1): {last_exception}. Retrying fetch...{RESET}")
                                    chunk_data = None # Discard and retry the first chunk
                                    # No break here, let retry logic handle it
                                else:
                                    lg.debug(f"Timestamp lag check passed for first chunk ({symbol}).")
                                    break # Chunk fetched and validated, exit retry loop
                            else:
                                lg.warning(f"Could not parse timeframe '{timeframe}' for lag check. Skipping validation.")
                                break # Proceed without lag check
                        except Exception as ts_err:
                            lg.warning(f"Could not validate timestamp lag ({symbol}, Chunk 1): {ts_err}. Proceeding cautiously.")
                            break # Proceed if validation fails
                    else:
                         # For subsequent chunks, just break retry loop on success
                         break
                else:
                    # If API returns empty list, it might mean no more data available going back
                    lg.debug(f"API returned no data for chunk {chunk_num} (End TS: {end_timestamp_ms}). Assuming end of history.")
                    remaining_limit = 0 # Stop fetching further chunks
                    break # Exit retry loop for this chunk

            # --- Error Handling (Per Chunk) ---
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_exception = e
                lg.warning(f"{NEON_YELLOW}Network error fetching kline chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait_time = RETRY_DELAY_SECONDS * 3
                lg.warning(f"{NEON_YELLOW}Rate limit fetching kline chunk {chunk_num} ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
                time.sleep(wait_time)
                continue # Continue retry loop without incrementing attempts
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Auth ritual failed fetching klines: {e}. Stopping.{RESET}")
                return pd.DataFrame() # Fatal error
            except ccxt.ExchangeError as e:
                last_exception = e
                lg.error(f"{NEON_RED}Exchange rift fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}")
                # Check for non-retryable errors specific to klines if known
                if "invalid timeframe" in str(e).lower() or "Interval is not supported" in str(e):
                     lg.critical(f"{NEON_RED}Invalid timeframe '{timeframe}' for {exchange.id}. Stopping fetch.{RESET}")
                     return pd.DataFrame()
                # Assume potentially retryable otherwise
            except Exception as e:
                last_exception = e
                lg.error(f"{NEON_RED}Unexpected vortex fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True)
                return pd.DataFrame() # Stop on unexpected errors

            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None: # Only sleep if retry is needed
                 time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt
        # --- End Retry Loop (Per Chunk) ---

        # --- Process Successful Chunk ---
        if chunk_data:
            # Prepend older data to the main list (since we fetch backwards)
            all_ohlcv_data = chunk_data + all_ohlcv_data
            remaining_limit -= len(chunk_data)

            # Set the end timestamp for the *next* fetch request to be the timestamp
            # of the *oldest* candle received in this chunk, minus 1ms, to avoid overlap.
            end_timestamp_ms = chunk_data[0][0] - 1

            # Check if we received fewer candles than requested, implies end of history
            if len(chunk_data) < fetch_size:
                 lg.debug(f"Received fewer candles ({len(chunk_data)}) than requested ({fetch_size}) for chunk {chunk_num}. Assuming end of available history.")
                 remaining_limit = 0 # Stop fetching
        else:
            # Fetching the chunk failed after all retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
            # Decide whether to proceed with partial data or fail entirely
            if not all_ohlcv_data: # Failed on the very first chunk
                 lg.error(f"Failed on first chunk ({symbol}). Returning empty DataFrame.")
                 return pd.DataFrame()
            else:
                 lg.warning(f"Proceeding with {len(all_ohlcv_data)} candles fetched before error occurred in chunk {chunk_num}.")
                 break # Exit the main fetching loop

        # Small delay between chunk fetches to be kind to the API
        if remaining_limit > 0: time.sleep(0.5)
    # --- End Fetching Loop ---

    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines ({symbol}) after {max_chunks} chunks, but {remaining_limit} candles still targeted. Increase max_chunks if needed.")

    # --- Process Combined Data ---
    if not all_ohlcv_data:
        lg.error(f"No kline data could be successfully fetched for {symbol} {timeframe}.")
        return pd.DataFrame()

    lg.info(f"Total klines fetched across all requests: {len(all_ohlcv_data)}")

    # Deduplicate based on timestamp (just in case of overlap, keep first occurrence - oldest)
    seen_timestamps = set()
    unique_data = []
    # Iterate in reverse to keep the latest instance in case of exact duplicates (unlikely but possible)
    for candle in reversed(all_ohlcv_data):
        ts = candle[0]
        if ts not in seen_timestamps:
            unique_data.append(candle)
            seen_timestamps.add(ts)
    unique_data.reverse() # Put back in ascending time order

    if len(unique_data) != len(all_ohlcv_data):
        lg.warning(f"Removed {len(all_ohlcv_data) - len(unique_data)} duplicate candle timestamps.")
    all_ohlcv_data = unique_data

    # Sort by timestamp just to be absolutely sure (should be sorted from fetch + prepend)
    all_ohlcv_data.sort(key=lambda x: x[0])

    # Trim to the originally requested number of candles (most recent ones) if more were fetched
    if len(all_ohlcv_data) > limit:
        lg.debug(f"Fetched {len(all_ohlcv_data)} candles, trimming to originally requested limit {limit}.")
        all_ohlcv_data = all_ohlcv_data[-limit:]

    # --- Process into DataFrame ---
    try:
        lg.debug(f"Processing {len(all_ohlcv_data)} final candles into DataFrame ({symbol})...")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_ohlcv_data, columns=cols[:len(all_ohlcv_data[0])])

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        if df.empty:
            lg.error(f"DataFrame empty after timestamp conversion ({symbol}).")
            return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential errors robustly
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Apply pd.to_numeric first, coercing errors to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # Convert valid finite numbers to Decimal, others become Decimal('NaN')
                df[col] = numeric_series.apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                )
            else:
                 lg.warning(f"Expected column '{col}' not found ({symbol}).")

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in essential price columns or non-positive close price
        essential_price_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_price_cols, inplace=True)
        df = df[df['close'] > Decimal('0')]
        # Drop rows with NaN volume or negative volume (if volume column exists)
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            df = df[df['volume'] >= Decimal('0')] # Allow zero volume

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Purged {rows_dropped} rows ({symbol}) during cleaning (NaNs, zero/neg prices/vol).")

        if df.empty:
            lg.warning(f"Kline DataFrame empty after cleaning ({symbol}).")
            return pd.DataFrame()

        # Ensure DataFrame is sorted by timestamp index (final check)
        if not df.index.is_monotonic_increasing:
            lg.warning(f"Kline index not monotonic ({symbol}), sorting..."); df.sort_index(inplace=True)

        # --- Memory Management ---
        # Trim DataFrame if it exceeds the maximum allowed length
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) > max ({MAX_DF_LEN}). Trimming."); df = df.iloc[-MAX_DF_LEN:].copy()

        lg.info(f"{NEON_GREEN}Successfully gathered and processed {len(df)} kline echoes for {symbol} {timeframe}{RESET}")
        return df
    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline echoes ({symbol}): {e}{RESET}", exc_info=True)
        return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Retrieves, validates, and standardizes market information for a symbol.

    - Reloads markets if the symbol is initially not found.
    - Extracts precision (price, amount), limits (min/max amount, cost),
      contract type (linear/inverse/spot), and contract size.
    - Adds convenience flags (`is_contract`, `is_linear`, etc.) and parsed
      Decimal values for precision/limits to the returned dictionary.
    - Includes retry logic for network errors.
    - Logs critical errors and returns None if essential precision data (amount/price steps) is missing or invalid.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object (must have markets loaded).
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[MarketInfo]: A MarketInfo TypedDict containing standardized market details, including
                              parsed Decimal values for limits/precision, or None if the market is not found,
                              essential data is missing, or a critical error occurs.
    """
    lg = logger
    lg.debug(f"Seeking market details for symbol: {symbol}...")
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            market: Optional[Dict] = None
            # Check if markets are loaded and the symbol exists
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' not found in memory. Refreshing market map...")
                try:
                    exchange.load_markets(reload=True) # Force reload
                    lg.info("Market map refreshed.")
                except Exception as reload_err:
                     last_exception = reload_err
                     lg.error(f"Failed to refresh market map while searching for {symbol}: {reload_err}")
                     # Continue to the retry logic below

            # Try fetching the market dictionary again after potential reload
            try:
                market = exchange.market(symbol)
            except ccxt.BadSymbol:
                 market = None # Handled below
            except Exception as fetch_err:
                 last_exception = fetch_err
                 lg.warning(f"Error fetching market dict for '{symbol}' after reload: {fetch_err}. Retrying...")
                 market = None

            if market is None:
                # Symbol not found or error fetching market dict
                if attempts < MAX_API_RETRIES:
                    lg.warning(f"Symbol '{symbol}' not found or market fetch failed (Attempt {attempts + 1}). Retrying check...")
                    # Fall through to retry delay
                else:
                    lg.error(f"{NEON_RED}Market '{symbol}' not found on {exchange.id} after reload and retries. Last echo: {last_exception}{RESET}")
                    return None # Symbol definitively not found or fetch failed
            else:
                # --- Market Found - Extract and Standardize Details ---
                lg.debug(f"Market found for '{symbol}'. Standardizing details...")
                std_market = market.copy() # Work on a copy

                # Add custom flags for easier logic later
                std_market['is_contract'] = std_market.get('contract', False) or std_market.get('type') in ['swap', 'future']
                std_market['is_linear'] = bool(std_market.get('linear')) and std_market['is_contract']
                std_market['is_inverse'] = bool(std_market.get('inverse')) and std_market['is_contract']
                std_market['contract_type_str'] = "Linear" if std_market['is_linear'] else \
                                                  "Inverse" if std_market['is_inverse'] else \
                                                  "Spot" if std_market.get('spot') else "Unknown"

                # Safely parse precision and limits into Decimal
                precision = std_market.get('precision', {})
                limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                # Parse precision steps (must be positive)
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), 'precision.amount', allow_zero=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), 'precision.price', allow_zero=False)

                # Parse limits (allow zero for min, must be positive if set)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), 'limits.amount.min', allow_zero=True)
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), 'limits.amount.max', allow_zero=False)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), 'limits.cost.min', allow_zero=True)
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), 'limits.cost.max', allow_zero=False)

                # Parse contract size (must be positive, default to 1)
                contract_size_val = std_market.get('contractSize', '1')
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, 'contractSize', allow_zero=False) or Decimal('1')

                # --- Critical Validation: Essential Precision ---
                # These are crucial for calculations and order placement
                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None:
                    lg.error(f"{NEON_RED}CRITICAL VALIDATION FAILED:{RESET} Market '{symbol}' missing essential positive precision runes.")
                    lg.error(f"  Amount Step: {std_market['amount_precision_step_decimal']}, Price Step: {std_market['price_precision_step_decimal']}")
                    lg.error(f"  Raw Precision Dict: {precision}")
                    lg.error("Cannot proceed safely without valid amount and price precision steps.")
                    return None # Returning None forces the calling function to handle the failure

                # Log extracted details for verification
                amt_step_str = std_market['amount_precision_step_decimal'].normalize()
                price_step_str = std_market['price_precision_step_decimal'].normalize()
                min_amt_str = std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] is not None else 'N/A'
                max_amt_str = std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] is not None else 'N/A'
                min_cost_str = std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] is not None else 'N/A'
                max_cost_str = std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] is not None else 'N/A'
                contract_size_str = std_market['contract_size_decimal'].normalize()

                log_msg = (
                    f"Market Details ({symbol}): Type={std_market['contract_type_str']}, Active={std_market.get('active')}\n"
                    f"  Precision (Amt/Price): {amt_step_str} / {price_step_str}\n"
                    f"  Limits (Amt Min/Max): {min_amt_str} / {max_amt_str}\n"
                    f"  Limits (Cost Min/Max): {min_cost_str} / {max_cost_str}\n"
                    f"  Contract Size: {contract_size_str}"
                )
                lg.debug(log_msg)

                # Cast to MarketInfo TypedDict before returning
                try:
                    # Directly cast assuming the structure matches after parsing.
                    final_market_info: MarketInfo = std_market # type: ignore
                    return final_market_info
                except Exception as cast_err:
                     lg.error(f"Error casting market dict to TypedDict: {cast_err}")
                     # Fallback: Return the dictionary anyway if casting fails but data seems okay
                     return std_market # type: ignore

        # --- Error Handling with Retries ---
        except ccxt.BadSymbol as e:
            # Symbol is definitively invalid according to the exchange
            lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}")
            return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance retrieving market info ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries for NetworkError market info ({symbol}).{RESET}")
                return None
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Auth ritual failed getting market info: {e}. Stopping.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange rift retrieving market info ({symbol}): {e}{RESET}")
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries for ExchangeError market info ({symbol}).{RESET}")
                 return None
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex retrieving market info ({symbol}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to retrieve market info ({symbol}) after all attempts. Last echo: {last_exception}{RESET}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available trading balance for a specific currency (e.g., USDT).

    - Handles Bybit V5 account types (UNIFIED, CONTRACT) to find the correct balance.
    - Parses various potential balance fields ('free', 'availableToWithdraw', 'availableBalance').
    - Includes retry logic for network errors and rate limits.
    - Handles authentication errors critically by re-raising them.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        currency (str): The currency code to fetch the balance for (e.g., "USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The available balance as a Decimal (non-negative), or None if fetching fails
                           after retries or a non-auth critical error occurs.

    Raises:
        ccxt.AuthenticationError: If authentication fails during the balance fetch.
    """
    lg = logger
    lg.debug(f"Scrying balance for currency: {currency}...")
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None
            found: bool = False
            balance_info: Optional[Dict] = None

            # Bybit V5 often requires specifying account type (Unified or Contract)
            # Check specific types first, then fallback to default request
            account_types_to_check = []
            if 'bybit' in exchange.id.lower():
                 # UNIFIED often holds assets for Linear USDT/USDC contracts
                 # CONTRACT is for Inverse contracts
                 # Check both if unsure, let the API return the relevant one
                 account_types_to_check = ['UNIFIED', 'CONTRACT']
            account_types_to_check.append('') # Always check default/unspecified type

            for acc_type in account_types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}
                    type_desc = f"Type: {acc_type}" if acc_type else "Default"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")
                    balance_info = exchange.fetch_balance(params=params)

                    # --- Try different ways to extract balance ---
                    # 1. Standard CCXT structure ('free' field)
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        lg.debug(f"Found balance in 'free' field ({type_desc}): {balance_str}")
                        found = True; break

                    # 2. Bybit V5 structure (often nested in 'info') - Check specific fields
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                        for account_details in balance_info['info']['result']['list']:
                             # Check if accountType matches the one requested (or if no type was requested)
                             # AND if the account details contain coin information
                             if (not acc_type or account_details.get('accountType') == acc_type) and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Prioritize fields representing available funds for trading
                                        balance_val = coin_data.get('availableToWithdraw') # Often best for Unified
                                        source_field = 'availableToWithdraw'
                                        if balance_val is None:
                                             balance_val = coin_data.get('availableBalance') # Alternative
                                             source_field = 'availableBalance'
                                        if balance_val is None:
                                             balance_val = coin_data.get('walletBalance') # Less preferred (may include non-tradable)
                                             source_field = 'walletBalance'

                                        if balance_val is not None:
                                            balance_str = str(balance_val)
                                            lg.debug(f"Found balance in Bybit V5 (Acc: {account_details.get('accountType')}, Field: {source_field}): {balance_str}")
                                            found = True; break # Found in coin list
                                if found: break # Found in account details list
                        if found: break # Found across account types

                except ccxt.ExchangeError as e:
                    # Errors like "account type does not exist" are expected when checking multiple types
                    if acc_type and ("account type does not exist" in str(e).lower() or "invalid account type" in str(e).lower()):
                        lg.debug(f"Account type '{acc_type}' not found. Trying next...")
                    elif acc_type: # Other errors for specific types
                        lg.debug(f"Minor exchange rift fetching balance ({type_desc}): {e}. Trying next...")
                    else: # Raise error if default fetch fails
                        raise e
                    continue # Try the next account type
                except Exception as e:
                    # Catch other unexpected errors during a specific account type check
                    lg.warning(f"Unexpected flicker fetching balance ({type_desc or 'Default'}): {e}. Trying next...")
                    last_exception = e # Store for potential final error message
                    continue # Try the next account type

            # --- Process the result ---
            if found and balance_str is not None:
                try:
                    balance_decimal = Decimal(balance_str)
                    # Ensure balance is not negative
                    final_balance = max(balance_decimal, Decimal('0'))
                    lg.debug(f"Parsed balance ({currency}): {final_balance.normalize()}")
                    return final_balance # Success
                except (ValueError, InvalidOperation, TypeError) as e:
                    # Raise an error if the found balance string cannot be converted
                    raise ccxt.ExchangeError(f"Failed to convert balance string '{balance_str}' ({currency}): {e}")
            elif not found and balance_info is not None:
                # If not found after checking all types, but we got some response
                raise ccxt.ExchangeError(f"Could not find balance for '{currency}'. Last response info: {balance_info.get('info')}")
            elif not found and balance_info is None:
                # If fetch_balance itself failed to return anything meaningful
                raise ccxt.ExchangeError(f"Could not find balance for '{currency}'. Fetch failed.")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit fetching balance ({currency}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Auth ritual failed fetching balance: {e}. Stopping.{RESET}")
            raise e # Re-raise AuthenticationError to be handled by the caller
        except ccxt.ExchangeError as e:
            last_exception = e
            # Log exchange errors (like currency not found, conversion errors) and retry
            lg.warning(f"{NEON_YELLOW}Exchange rift fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex fetching balance ({currency}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to scry balance ({currency}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return None

# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[PositionInfo]:
    """
    Checks for an existing open position for the given symbol using `fetch_positions`.

    - Handles Bybit V5 specifics (category, symbol filtering, parsing `info` field).
    - Determines position side ('long'/'short') and size accurately using Decimal.
    - Parses key position details (entry price, leverage, SL/TP, TSL) into a standardized format.
    - Includes retry logic for network errors and rate limits.
    - Returns a standardized `PositionInfo` dictionary if an active position is found (size > threshold),
      otherwise returns None. Returns None on failure.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[PositionInfo]: A PositionInfo TypedDict containing details of the open position if found,
                                otherwise None.
    """
    lg = logger
    attempts = 0
    last_exception = None
    market_id: Optional[str] = None
    category: Optional[str] = None

    # --- Determine Market ID and Category ---
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        if category == 'spot':
            lg.info(f"Position check skipped for {symbol}: Spot market.")
            return None # Positions are not applicable to spot in the same way
        lg.debug(f"Using Market ID: {market_id}, Category: {category} for position check.")
    except KeyError:
        lg.error(f"Market '{symbol}' not found in loaded markets. Cannot check position.")
        return None
    except Exception as e:
        lg.error(f"Error determining market details for position check ({symbol}): {e}")
        return None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts + 1})...")
            positions: List[Dict] = []

            # --- Fetch Positions (Handling Bybit V5 Specifics) ---
            try:
                # Attempt to fetch positions for the specific symbol and category
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching positions with params: {params}")

                # Use fetch_positions and filter, as fetchPositionsForSymbol isn't standard/reliable across CCXT versions/exchanges
                if exchange.has.get('fetchPositions'):
                     all_positions = exchange.fetch_positions(params=params)
                     # Filter the results manually by symbol or market ID
                     positions = [
                         p for p in all_positions
                         if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id
                     ]
                     lg.debug(f"Fetched {len(all_positions)} total positions ({category}), filtered to {len(positions)} for {symbol}.")
                else:
                     raise ccxt.NotSupported("Exchange does not support fetchPositions.")

            except ccxt.ExchangeError as e:
                 # Bybit often returns specific codes for "no position" or related issues
                 no_pos_codes = [110025] # e.g., "position idx not match position mode" can indicate no pos in one-way
                 no_pos_messages = ["position not found", "no position", "position does not exist"]
                 err_str = str(e).lower()
                 # Try to extract retCode reliably
                 code_str = ""
                 match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
                 if match:
                     code_str = match.group(2)
                 if not code_str: # Fallback check on exception attributes
                      code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

                 code_match = any(str(c) in code_str for c in no_pos_codes) if code_str else False

                 if code_match or any(msg in err_str for msg in no_pos_messages):
                     lg.info(f"No open position found for {symbol} (Exchange message: {e}).")
                     return None # No position exists
                 else:
                     # Re-raise other exchange errors to be handled by the main retry loop
                     raise e

            # --- Process Fetched Positions ---
            active_position_raw: Optional[Dict] = None
            # Define a small threshold based on amount precision to consider a position "open"
            size_threshold = Decimal('1e-9') # Default small value
            try:
                # Use a fraction of the minimum amount step as the threshold (e.g., 1% of step)
                amt_step = Decimal(str(exchange.market(symbol)['precision']['amount']))
                if amt_step > 0:
                    size_threshold = amt_step * Decimal('0.01')
            except Exception as prec_err:
                lg.warning(f"Could not get amount precision for {symbol} to set size threshold: {prec_err}. Using default: {size_threshold}")
            lg.debug(f"Using position size threshold (absolute): {size_threshold.normalize()}")

            # Iterate through the filtered positions to find an active one
            for pos in positions:
                # Extract size from 'info' (Bybit V5 'size') or standard 'contracts' field
                size_str_info = str(pos.get('info', {}).get('size', '')).strip()
                size_str_std = str(pos.get('contracts', '')).strip() # Standard field (often float)
                size_str = size_str_info if size_str_info else size_str_std # Prioritize info['size']

                if not size_str:
                    lg.debug(f"Skipping position entry with missing size data: {pos.get('info', {})}")
                    continue

                try:
                    # Convert size to Decimal and check against threshold (absolute value)
                    size_decimal = Decimal(size_str)
                    if abs(size_decimal) > size_threshold:
                        # Found an active position with significant size
                        active_position_raw = pos
                        active_position_raw['size_decimal'] = size_decimal # Store the parsed Decimal size
                        lg.debug(f"Found active position entry ({symbol}): Size={size_decimal.normalize()}")
                        break # Stop searching once an active position is found
                    else:
                        lg.debug(f"Skipping position entry near-zero size ({size_decimal.normalize()}): {pos.get('info', {})}")
                except (ValueError, InvalidOperation, TypeError) as parse_err:
                     # Log error if size string cannot be parsed, skip this entry
                     lg.warning(f"Could not parse position size '{size_str}' ({symbol}): {parse_err}. Skipping.")
                     continue # Move to the next position entry

            # --- Format and Return Active Position ---
            if active_position_raw:
                # Standardize the position dictionary using PositionInfo structure
                std_pos = active_position_raw.copy() # Work on a copy
                info = std_pos.get('info', {}) # Exchange-specific details

                # Determine Side (long/short) reliably
                side = std_pos.get('side') # Standard CCXT field
                size = std_pos['size_decimal'] # Use the parsed Decimal size

                if side not in ['long', 'short']:
                    # Fallback using Bybit V5 'side' field ('Buy'/'Sell') or inferred from size
                    side_v5 = str(info.get('side', '')).lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif size > size_threshold: side = 'long' # Infer from positive size
                    elif size < -size_threshold: side = 'short' # Infer from negative size
                    else: side = None # Cannot determine side

                if not side:
                    lg.error(f"Could not determine side for active position {symbol}. Size: {size}. Data: {info}")
                    return None # Cannot proceed without side
                std_pos['side'] = side # Update the standardized dict

                # Standardize other key fields (prefer standard CCXT, fallback to info) using safe Decimal conversion
                std_pos['entryPrice'] = _safe_market_decimal(std_pos.get('entryPrice') or info.get('avgPrice') or info.get('entryPrice'), 'entryPrice', allow_zero=False)
                std_pos['leverage'] = _safe_market_decimal(std_pos.get('leverage') or info.get('leverage'), 'leverage', allow_zero=False)
                std_pos['liquidationPrice'] = _safe_market_decimal(std_pos.get('liquidationPrice') or info.get('liqPrice'), 'liquidationPrice', allow_zero=False)
                std_pos['unrealizedPnl'] = _safe_market_decimal(std_pos.get('unrealizedPnl') or info.get('unrealisedPnl') or info.get('unrealizedPnl'), 'unrealizedPnl', allow_zero=True) # Pnl can be zero

                # Parse protection levels from 'info' (Bybit V5 specific fields)
                # Ensure they are non-empty strings and represent non-zero values before storing
                def get_protection_field(field_name: str) -> Optional[str]:
                    """Extracts protection field if valid non-zero number string."""
                    value = info.get(field_name)
                    s_value = str(value).strip() if value is not None else None
                    if not s_value: return None
                    try:
                         # Check if it's a valid number and not effectively zero
                         if abs(Decimal(s_value)) > Decimal('1e-12'): # Use tolerance for zero check
                             return s_value
                    except (InvalidOperation, ValueError, TypeError):
                         return None # Ignore if not a valid non-zero number string
                    return None

                std_pos['stopLossPrice'] = get_protection_field('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_field('takeProfit')
                std_pos['trailingStopLoss'] = get_protection_field('trailingStop') # TSL distance
                std_pos['tslActivationPrice'] = get_protection_field('activePrice') # TSL activation price

                # Initialize bot state flags (these are in-memory and not persistent)
                std_pos['be_activated'] = False # Will be set by management logic if BE applied during this run
                std_pos['tsl_activated'] = bool(std_pos['trailingStopLoss']) # True if TSL distance is already set via API

                # Helper for formatting Decimal values for logging
                def format_decimal_log(value: Optional[Any]) -> str:
                    dec_val = _safe_market_decimal(value, 'log', True)
                    return dec_val.normalize() if dec_val is not None else 'N/A'

                # Log summary of the found position
                ep_str = format_decimal_log(std_pos.get('entryPrice'))
                size_str = std_pos['size_decimal'].normalize()
                sl_str = format_decimal_log(std_pos.get('stopLossPrice'))
                tp_str = format_decimal_log(std_pos.get('takeProfitPrice'))
                tsl_dist_str = format_decimal_log(std_pos.get('trailingStopLoss'))
                tsl_act_str = format_decimal_log(std_pos.get('tslActivationPrice'))
                tsl_log = f"Dist={tsl_dist_str}/Act={tsl_act_str}" if tsl_dist_str != 'N/A' or tsl_act_str != 'N/A' else "N/A"
                pnl_str = format_decimal_log(std_pos.get('unrealizedPnl'))
                liq_str = format_decimal_log(std_pos.get('liquidationPrice'))

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} Size={size_str}, Entry={ep_str}, Liq={liq_str}, PnL={pnl_str}, SL={sl_str}, TP={tp_str}, TSL={tsl_log}")

                # Cast to PositionInfo TypedDict before returning
                try:
                    # Assume structure matches after parsing
                    final_position_info: PositionInfo = std_pos # type: ignore
                    return final_position_info
                except Exception as cast_err:
                     lg.error(f"Error casting position to TypedDict ({symbol}): {cast_err}")
                     return std_pos # type: ignore # Return raw dict if cast fails
            else:
                # No position with size > threshold was found after filtering
                lg.info(f"No active position found for {symbol}.")
                return None

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit fetching positions ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Auth ritual failed fetching positions: {e}. Stopping.{RESET}")
            return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange rift fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            # Could add checks for specific non-retryable errors here
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex fetching positions ({symbol}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to get position info ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a derivatives symbol using `set_leverage`.

    - Skips if the market is not a contract (spot) or leverage is invalid (<= 0).
    - Handles Bybit V5 specific parameters (category, buy/sell leverage).
    - Includes retry logic for network/exchange errors.
    - Checks for specific Bybit codes indicating success (`0`) or leverage already set (`110045`).
    - Identifies and handles known non-retryable leverage errors.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        leverage (int): The desired integer leverage level.
        market_info (MarketInfo): The standardized MarketInfo dictionary.
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        bool: True if leverage was set successfully or was already set correctly, False otherwise.
    """
    lg = logger
    # Validate input and market type
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped ({symbol}): Not a contract market.")
        return True # Consider success as no action needed for spot
    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped ({symbol}): Invalid leverage value ({leverage}). Must be a positive integer.")
        return False
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage method.")
        return False

    market_id = market_info['id'] # Use the exchange-specific market ID

    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting leverage set ({market_id} to {leverage}x, Attempt {attempts + 1})...")
            params = {}
            # --- Bybit V5 Specific Parameters ---
            if 'bybit' in exchange.id.lower():
                 # Determine category and set buy/sell leverage explicitly for Bybit V5
                 category = market_info.get('contract_type_str', 'Linear').lower() # 'linear' or 'inverse'
                 if category not in ['linear', 'inverse']:
                      lg.warning(f"Leverage skipped: Invalid category '{category}' ({symbol}). Must be 'linear' or 'inverse'.")
                      return False
                 params = {
                     'category': category,
                     #'symbol': market_id, # Symbol passed separately to set_leverage
                     'buyLeverage': str(leverage), # Bybit expects strings
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 setLeverage params: {params}")

            # --- Execute set_leverage call ---
            # Pass market_id as the symbol argument
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"set_leverage raw response ({symbol}): {response}")

            # --- Check Response (Bybit V5 specific codes) ---
            ret_code_str = None
            ret_msg = "N/A"
            if isinstance(response, dict):
                 info_dict = response.get('info', {}) # CCXT often puts raw response here
                 # Check retCode in info first, then top level
                 ret_code_info = info_dict.get('retCode')
                 ret_code_top = response.get('retCode')
                 # Use the first non-zero code found, prioritize info
                 if ret_code_info is not None and ret_code_info != 0:
                     ret_code_str = str(ret_code_info)
                 elif ret_code_top is not None:
                     ret_code_str = str(ret_code_top)
                 else: # Both are None or 0
                     ret_code_str = str(ret_code_info) if ret_code_info is not None else str(ret_code_top)

                 ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown Bybit msg'))

            if ret_code_str == '0':
                 lg.info(f"{NEON_GREEN}Leverage set ({market_id} to {leverage}x, Code: 0).{RESET}")
                 return True
            elif ret_code_str == '110045': # "Leverage not modified"
                 lg.info(f"{NEON_YELLOW}Leverage already {leverage}x ({market_id}, Code: 110045).{RESET}")
                 return True
            elif ret_code_str is not None and ret_code_str not in ['None', '0']: # Check if a non-zero code was returned
                 # Raise an error for other non-zero Bybit return codes
                 raise ccxt.ExchangeError(f"Bybit API error setting leverage ({symbol}): {ret_msg} (Code: {ret_code_str})")
            else:
                # Assume success if no specific error code structure is found and no exception was raised by CCXT
                lg.info(f"{NEON_GREEN}Leverage set/confirmed ({market_id} to {leverage}x, No specific error code).{RESET}")
                return True

        # --- Error Handling with Retries ---
        except ccxt.ExchangeError as e:
            last_exception = e
            # Try to extract error code more reliably
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            err_str = str(e).lower()
            lg.error(f"{NEON_RED}Exchange rift setting leverage ({market_id}): {e} (Code: {err_code_str}){RESET}")

            # Check for non-retryable conditions based on code or message
            if err_code_str == '110045' or "leverage not modified" in err_str:
                lg.info(f"{NEON_YELLOW}Leverage already set (via error). Success.{RESET}")
                return True # Already set is considered success

            # List of known fatal Bybit error codes for leverage setting
            fatal_codes = [
                '10001', '10004', '110009', '110013', '110028', '110043',
                '110044', '110055', '3400045'
            ]
            fatal_messages = ["margin mode", "position exists", "risk limit", "parameter error", "insufficient balance", "invalid leverage"]

            if err_code_str in fatal_codes or any(msg in err_str for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE leverage error ({symbol}). Aborting.{RESET}")
                return False # Fatal error

            # If error is potentially retryable and retries remain
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries for ExchangeError setting leverage ({symbol}).{RESET}")
                return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance setting leverage ({market_id}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries for NetworkError setting leverage ({symbol}).{RESET}")
                return False

        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Auth ritual failed setting leverage ({symbol}): {e}. Stopping.{RESET}")
             return False # Fatal error for this operation

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex setting leverage ({market_id}): {e}{RESET}", exc_info=True)
            return False # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed leverage set ({market_id} to {leverage}x) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return False

def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on risk parameters and market constraints.

    Uses Decimal for all financial calculations to ensure precision. Handles both
    linear and inverse contracts. Applies market precision and limits (amount, cost)
    to the calculated size, rounding the final size DOWN to the nearest valid step.
    Includes a final check to bump size up one step if needed to meet minimum cost.

    Args:
        balance (Decimal): Available trading balance (in quote currency, e.g., USDT). Must be positive.
        risk_per_trade (float): Fraction of balance to risk (e.g., 0.01 for 1%). Must be > 0 and <= 1.
        initial_stop_loss_price (Decimal): The calculated initial stop loss price. Must be positive and different from entry.
        entry_price (Decimal): The intended entry price (or current price). Must be positive.
        market_info (MarketInfo): The standardized MarketInfo dictionary containing precision, limits, contract type, etc.
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object (used for precision formatting).
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The calculated position size as a Decimal, adjusted for market rules (positive value),
                           or None if sizing is not possible (e.g., invalid inputs, insufficient balance for min size,
                           cannot meet limits, calculation errors).
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE')
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('is_inverse', False)
    # Determine the unit of the size (Contracts for derivatives, Base currency for Spot)
    size_unit = base_currency if market_info.get('spot', False) else "Contracts"

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    # --- Input Validation ---
    if balance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Invalid balance {balance.normalize()}.")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
            raise ValueError("Risk per trade must be > 0 and <= 1.")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Invalid risk '{risk_per_trade}': {e}")
        return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Entry ({entry_price.normalize()}) / SL ({initial_stop_loss_price.normalize()}) must be positive.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): SL price equals Entry price.")
        return None

    # --- Extract Market Constraints (using pre-parsed Decimal values from MarketInfo) ---
    try:
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal'] # Needed for cost estimation/validation
        min_amount = market_info['min_amount_decimal'] # Can be None or 0
        max_amount = market_info['max_amount_decimal'] # Can be None
        min_cost = market_info['min_cost_decimal']     # Can be None or 0
        max_cost = market_info['max_cost_decimal']     # Can be None
        contract_size = market_info['contract_size_decimal']

        # Check if essential constraints are valid (already checked in get_market_info, but re-check here)
        if amount_step is None or amount_step <= 0: raise ValueError(f"Invalid amount precision step: {amount_step}")
        if price_step is None or price_step <= 0: raise ValueError(f"Invalid price precision step: {price_step}")
        if contract_size <= Decimal('0'): raise ValueError(f"Invalid contract size: {contract_size}")

        # Set defaults for optional limits if they are None (treat as unconstrained)
        min_amount_eff = min_amount if min_amount is not None else Decimal('0')
        max_amount_eff = max_amount if max_amount is not None else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None else Decimal('0')
        max_cost_eff = max_cost if max_cost is not None else Decimal('inf')

        lg.debug(f"  Market Constraints ({symbol}): AmtStep={amount_step.normalize()}, Min/Max Amt={min_amount_eff.normalize()}/{max_amount_eff.normalize()}, "
                 f"Min/Max Cost={min_cost_eff.normalize()}/{max_cost_eff.normalize()}, ContrSize={contract_size.normalize()}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error validating market details: {e}")
        lg.debug(f" MarketInfo used: {market_info}")
        return None

    # --- Calculate Risk Amount and Stop Loss Distance ---
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN) # Quantize risk amount early
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance <= Decimal('0'): # Should be caught earlier, but double-check
        lg.error(f"Sizing failed ({symbol}): SL distance zero.")
        return None

    lg.info(f"  Balance: {balance.normalize()} {quote_currency}, Risk: {risk_decimal:.2%} ({risk_amount_quote.normalize()} {quote_currency})")
    lg.info(f"  Entry: {entry_price.normalize()}, SL: {initial_stop_loss_price.normalize()}, SL Dist: {stop_loss_distance.normalize()}")
    lg.info(f"  Contract Type: {market_info['contract_type_str']}")

    # --- Calculate Initial Position Size (based on risk) ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear Contracts or Spot
            # Value change per unit (contract or base currency unit) = Price distance * Contract Size
            value_change_per_unit = stop_loss_distance * contract_size
            if value_change_per_unit <= Decimal('1e-18'): # Use tolerance for zero check
                 lg.error(f"Sizing failed ({symbol}, Lin/Spot): Value change per unit near zero.")
                 return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Calc: {risk_amount_quote} / {value_change_per_unit} = {calculated_size}")

        else: # Inverse Contracts
            # Risk per contract (in quote) = Contract Size * |(1 / Entry) - (1 / SL)|
            if entry_price <= 0 or initial_stop_loss_price <= 0:
                 lg.error(f"Sizing failed ({symbol}, Inv): Entry/SL zero/negative.")
                 return None
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor <= Decimal('1e-18'): # Tolerance for zero check
                 lg.error(f"Sizing failed ({symbol}, Inv): Inverse factor near zero.")
                 return None
            risk_per_contract = contract_size * inverse_factor
            if risk_per_contract <= Decimal('1e-18'): # Tolerance for zero check
                 lg.error(f"Sizing failed ({symbol}, Inv): Risk per contract near zero.")
                 return None
            calculated_size = risk_amount_quote / risk_per_contract
            lg.debug(f"  Inverse Calc: {risk_amount_quote} / {risk_per_contract} = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as calc_err:
        lg.error(f"Sizing failed ({symbol}): Calc error: {calc_err}.")
        return None

    if calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial size zero/negative ({calculated_size.normalize()}). Check risk/balance/SL distance.")
        return None
    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Apply Market Limits and Precision ---
    adjusted_size = calculated_size

    # Helper function to estimate cost
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        """Estimates order cost based on size, price, contract type."""
        if price <= 0 or size <= 0: return None
        try:
             # Cost = Size * Price * ContractSize (Linear/Spot) or Size * ContractSize / Price (Inverse)
             return (size * price * contract_size) if not is_inverse else (size * contract_size / price)
        except (InvalidOperation, OverflowError, ZeroDivisionError):
            return None

    # 1. Apply Amount Limits (Min/Max Size)
    if min_amount_eff > 0 and adjusted_size < min_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Calc size {adjusted_size.normalize()} < min {min_amount_eff.normalize()}. Adjusting UP.{RESET}")
        adjusted_size = min_amount_eff
    if max_amount_eff < Decimal('inf') and adjusted_size > max_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Calc size {adjusted_size.normalize()} > max {max_amount_eff.normalize()}. Adjusting DOWN.{RESET}")
        adjusted_size = max_amount_eff
    lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 2. Apply Cost Limits (Min/Max Order Value)
    cost_adj_applied = False
    est_cost = estimate_cost(adjusted_size, entry_price)
    if est_cost is not None:
        lg.debug(f"  Estimated Cost (after amount limits, {symbol}): {est_cost.normalize()} {quote_currency}")
        # Check Minimum Cost
        if min_cost_eff > 0 and est_cost < min_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Est cost {est_cost.normalize()} < min cost {min_cost_eff.normalize()}. Increasing size.{RESET}")
            try:
                # Calculate size needed to meet min cost
                req_size = (min_cost_eff / (entry_price * contract_size)) if not is_inverse else (min_cost_eff * entry_price / contract_size)
                if req_size <= 0: raise ValueError("Invalid required size for min cost")
                lg.info(f"  Size required for min cost ({symbol}): {req_size.normalize()} {size_unit}")
                # Check if this required size exceeds max amount limit
                if max_amount_eff < Decimal('inf') and req_size > max_amount_eff:
                    lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet min cost ({min_cost_eff.normalize()}) without exceeding max amount ({max_amount_eff.normalize()}).{RESET}")
                    return None
                # Adjust size up, ensuring it's still >= original min_amount
                adjusted_size = max(min_amount_eff, req_size)
                cost_adj_applied = True
            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Failed min cost size calc: {cost_calc_err}.{RESET}")
                return None
        # Check Maximum Cost
        elif max_cost_eff < Decimal('inf') and est_cost > max_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Est cost {est_cost.normalize()} > max cost {max_cost_eff.normalize()}. Reducing size.{RESET}")
            try:
                # Calculate max size allowed by max cost
                max_size = (max_cost_eff / (entry_price * contract_size)) if not is_inverse else (max_cost_eff * entry_price / contract_size)
                if max_size <= 0: raise ValueError("Invalid max size for max cost")
                lg.info(f"  Max size allowed by max cost ({symbol}): {max_size.normalize()} {size_unit}")
                # Adjust size down: take min(current size, max size from cost), ensure >= min_amount
                adjusted_size = max(min_amount_eff, min(adjusted_size, max_size))
                cost_adj_applied = True
            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Failed max cost size calc: {cost_calc_err}.{RESET}")
                return None
    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        lg.warning(f"Could not estimate cost ({symbol}) for limit check.")

    if cost_adj_applied:
        lg.info(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 3. Apply Amount Precision (Rounding DOWN to step size for safety)
    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step zero/negative.")
        # Manual Rounding Down: size = floor(value / step) * step
        final_size = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN) * amount_step
        if final_size != adjusted_size:
            lg.info(f"Applied amount precision ({symbol}, Rounded DOWN to {amount_step.normalize()}): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
    except (InvalidOperation, ValueError, TypeError) as fmt_err:
        lg.error(f"{NEON_RED}Error applying amount precision ({symbol}): {fmt_err}. Using unrounded: {final_size.normalize()}{RESET}")
        # Continue with unrounded size, but subsequent checks might fail

    # --- Final Validation after Precision ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size zero/negative ({final_size.normalize()}).{RESET}")
        return None
    # Check Min Amount again after rounding down
    if min_amount_eff > 0 and final_size < min_amount_eff:
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} < min amount {min_amount_eff.normalize()} after precision.{RESET}")
        return None
    # Check Max Amount again (should be fine if rounded down, but check)
    if max_amount_eff < Decimal('inf') and final_size > max_amount_eff:
         lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} > max amount {max_amount_eff.normalize()} after precision.{RESET}")
         return None

    # Final check on cost after precision (especially if min cost is required)
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}): {final_cost.normalize()} {quote_currency}")
        # Check if final cost is below min cost after rounding down
        if min_cost_eff > 0 and final_cost < min_cost_eff:
             lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Final cost {final_cost.normalize()} < min cost {min_cost_eff.normalize()} after rounding.{RESET}")
             # Attempt to bump size up by one step if possible
             try:
                 next_size = final_size + amount_step
                 next_cost = estimate_cost(next_size, entry_price)
                 if next_cost is not None:
                     # Check if bumping up is valid (meets min cost, doesn't exceed max amount/cost)
                     can_bump = (next_cost >= min_cost_eff) and \
                                (max_amount_eff == Decimal('inf') or next_size <= max_amount_eff) and \
                                (max_cost_eff == Decimal('inf') or next_cost <= max_cost_eff)

                     if can_bump:
                         lg.info(f"{NEON_YELLOW}Bumping final size ({symbol}) up one step to {next_size.normalize()} for min cost.{RESET}")
                         final_size = next_size
                         final_cost = estimate_cost(final_size, entry_price) # Recalculate cost for logging
                         lg.debug(f"  Final Cost after bump: {final_cost.normalize() if final_cost else 'N/A'}")
                     else:
                         lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet min cost even bumping size due to other limits (Max Amount/Cost).{RESET}")
                         return None
                 else:
                      lg.error(f"{NEON_RED}Sizing failed ({symbol}): Could not estimate cost for bumped size.{RESET}")
                      return None
             except Exception as bump_err:
                 lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error bumping size: {bump_err}.{RESET}")
                 return None
        # Check if final cost exceeds max cost (should be rare if rounded down, but check)
        elif max_cost_eff < Decimal('inf') and final_cost > max_cost_eff:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} > max cost {max_cost_eff.normalize()} after precision.{RESET}")
            return None
    elif min_cost_eff > 0:
         lg.warning(f"Could not perform final cost check ({symbol}) after precision.")

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size

def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """
    Cancels an order by ID with retries.

    - Handles Bybit V5 specifics (category, symbol needed even for ID cancel).
    - Treats `OrderNotFound` error as success (already cancelled/filled).
    - Includes retry logic for network/exchange errors and rate limits.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        order_id (str): The ID of the order to cancel.
        symbol (str): The symbol the order belongs to (required by some exchanges/APIs).
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        bool: True if the order was cancelled successfully or was already not found, False otherwise.
    """
    lg = logger
    attempts = 0
    last_exception = None
    lg.info(f"Attempting to cancel order ID: {order_id} for {symbol}...")
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1} for ID {order_id} ({symbol})...")
            params = {}
            market_id = symbol # Default
            # --- Bybit V5 Specific Params ---
            if 'bybit' in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    market_id = market['id'] # Use exchange-specific ID
                    params['category'] = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                    params['symbol'] = market_id
                    lg.debug(f"Using Bybit V5 cancel params: {params}")
                except Exception as e:
                    lg.warning(f"Could not determine category/market_id for cancel order {order_id} ({symbol}): {e}. Using defaults.")

            # --- Execute cancel_order call ---
            # Pass standard symbol to CCXT method, specific params handle exchange needs
            exchange.cancel_order(order_id, symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully cancelled order ID: {order_id} for {symbol}.{RESET}")
            return True

        # --- Error Handling ---
        except ccxt.OrderNotFound:
            # If the order doesn't exist, it's effectively cancelled.
            lg.warning(f"{NEON_YELLOW}Order ID {order_id} ({symbol}) not found. Already cancelled or filled? Treating as success.{RESET}")
            return True
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 2 # Shorter wait for cancel
            lg.warning(f"{NEON_YELLOW}Rate limit cancelling order {order_id} ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}{RESET}")
            # Assume most exchange errors are retryable for cancel unless specific codes indicate otherwise
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True)
            return False # Non-retryable

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to cancel order ID {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return False

def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: MarketInfo,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Places a market order (buy or sell) using `create_order`.

    - Maps trade signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT") to order sides ("buy", "sell").
    - Handles Bybit V5 specific parameters (category, positionIdx, reduceOnly, timeInForce).
    - Includes retry logic for network/exchange errors and rate limits.
    - Identifies and handles non-retryable order errors (e.g., insufficient funds, invalid parameters) with hints.
    - Logs order details clearly.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        trade_signal (str): The signal driving the trade ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size (Decimal): The calculated position size (must be positive Decimal).
        market_info (MarketInfo): The standardized MarketInfo dictionary.
        logger (logging.Logger): The logger instance for status messages.
        reduce_only (bool): Set to True for closing orders to ensure they only reduce/close a position.
        params (Optional[Dict]): Optional additional parameters to pass to create_order's `params` argument.

    Returns:
        Optional[Dict]: The order dictionary returned by CCXT upon successful placement, or None if the
                        order fails after retries or due to fatal errors.
    """
    lg = logger
    # Map signal to CCXT side ('buy' or 'sell')
    side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}
    side = side_map.get(trade_signal.upper())

    # --- Input Validation ---
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided to place_trade.")
        return None
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'):
        lg.error(f"Invalid position size provided to place_trade: {position_size}. Must be positive Decimal.")
        return None

    order_type = 'market' # This bot currently only uses market orders
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    size_unit = base_currency if market_info.get('spot', False) else "Contracts"
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info['id'] # Use the exchange-specific market ID

    # --- Prepare Order Arguments ---
    # CCXT typically expects float amount, convert Decimal carefully
    try:
         amount_float = float(position_size)
         if amount_float <= 1e-15: raise ValueError("Size negligible after float conversion.")
    except (ValueError, TypeError) as float_err:
         lg.error(f"Failed to convert size {position_size.normalize()} ({symbol}) to valid positive float: {float_err}")
         return None

    order_args: Dict[str, Any] = {
        'symbol': market_id, # Use market_id here
        'type': order_type,
        'side': side,
        'amount': amount_float,
    }
    order_params: Dict[str, Any] = {} # For exchange-specific parameters

    # --- Bybit V5 Specific Parameters ---
    if 'bybit' in exchange.id.lower() and is_contract:
        try:
            category = market_info.get('contract_type_str', 'Linear').lower()
            if category not in ['linear', 'inverse']:
                 raise ValueError(f"Invalid category '{category}' for Bybit order.")
            order_params = {
                'category': category,
                'positionIdx': 0  # Use 0 for one-way mode (required by Bybit V5 for non-hedge mode)
            }
            if reduce_only:
                order_params['reduceOnly'] = True
                # Use IOC for reduceOnly market orders to prevent resting if market moves away quickly
                order_params['timeInForce'] = 'IOC' # Immediate Or Cancel
            lg.debug(f"Using Bybit V5 order params ({symbol}): {order_params}")
        except Exception as e:
            lg.error(f"Failed to set Bybit V5 params ({symbol}): {e}. Order might fail.")
            # Proceed cautiously without params if setting failed

    # Merge any additional custom parameters provided by the caller
    if params:
        order_params.update(params)

    if order_params:
        order_args['params'] = order_params # Add exchange-specific params to the main args

    # Log the trade attempt
    lg.info(f"{BRIGHT}===> Attempting {action_desc} | {side.upper()} {order_type.upper()} Order | {symbol} | Size: {position_size.normalize()} {size_unit} <==={RESET}")
    if order_params: lg.debug(f"  with Params ({symbol}): {order_params}")

    # --- Execute Order with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1})...")
            order_result = exchange.create_order(**order_args)

            # Log Success
            order_id = order_result.get('id', 'N/A')
            status = order_result.get('status', 'N/A')
            # Safely format potential Decimal/float/str values from result
            avg_price_dec = _safe_market_decimal(order_result.get('average'), 'order.avg', True)
            filled_dec = _safe_market_decimal(order_result.get('filled'), 'order.filled', True) # Allow zero filled initially
            log_msg = (
                f"{NEON_GREEN}{action_desc} Order Placed!{RESET} ID: {order_id}, Status: {status}"
            )
            if avg_price_dec: log_msg += f", AvgFill: ~{avg_price_dec.normalize()}"
            if filled_dec: log_msg += f", Filled: {filled_dec.normalize()}"
            lg.info(log_msg)
            lg.debug(f"Full order result ({symbol}): {order_result}")
            return order_result # Return the successful order details

        # --- Error Handling with Retries ---
        except ccxt.InsufficientFunds as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Insufficient funds. {e}{RESET}")
            return None # Non-retryable
        except ccxt.InvalidOrder as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Invalid order params. {e}{RESET}")
            lg.error(f"  Args: {order_args}")
            # Add hints based on common causes
            err_lower = str(e).lower()
            min_amt_str = market_info.get('min_amount_decimal', 'N/A')
            min_cost_str = market_info.get('min_cost_decimal', 'N/A')
            amt_step_str = market_info.get('amount_precision_step_decimal', 'N/A')
            max_amt_str = market_info.get('max_amount_decimal', 'N/A')
            max_cost_str = market_info.get('max_cost_decimal', 'N/A')

            if "minimum" in err_lower or "too small" in err_lower or "lower than limit" in err_lower:
                 lg.error(f"  >> Hint: Check size/cost ({position_size.normalize()}) vs market mins (MinAmt: {min_amt_str}, MinCost: {min_cost_str}).")
            elif "precision" in err_lower or "lot size" in err_lower or "step size" in err_lower:
                 lg.error(f"  >> Hint: Check size ({position_size.normalize()}) against amount step ({amt_step_str}).")
            elif "exceed" in err_lower or "too large" in err_lower or "greater than limit" in err_lower:
                 lg.error(f"  >> Hint: Check size/cost ({position_size.normalize()}) vs market maxs (MaxAmt: {max_amt_str}, MaxCost: {max_cost_str}).")
            elif "reduce only" in err_lower:
                 lg.error(f"  >> Hint: Reduce-only failed. Check position size/direction or if order would increase position.")
            return None # Non-retryable
        except ccxt.ExchangeError as e:
            last_exception = e
            # Try to extract error code
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Exchange rift. {e} (Code: {err_code_str}){RESET}")

            # Check for known fatal Bybit error codes related to orders
            fatal_order_codes = [
                '10001', '10004', '110007', '110013', '110014', '110017', '110025',
                '110040', '30086', '3303001', '3303005', '3400060', '3400088'
            ]
            fatal_messages = ["invalid parameter", "precision", "exceed limit", "risk limit", "invalid symbol", "reduce only check failed", "lot size"]

            if err_code_str in fatal_order_codes or any(msg in str(e).lower() for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE order error ({symbol}).{RESET}")
                return None # Non-retryable

            # Assume other exchange errors might be temporary and retry if attempts remain
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries for ExchangeError placing order ({symbol}).{RESET}")
                 return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries for NetworkError placing order ({symbol}).{RESET}")
                return None

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit placing order ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Auth ritual failed placing order ({symbol}): {e}. Stopping.{RESET}")
             return None # Fatal error for this operation

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex placing order ({symbol}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter (only if not a rate limit wait) and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to place {action_desc} order ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return None

def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """
    Internal helper: Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL)
    for an existing position using Bybit's V5 private API endpoint `/v5/position/set-trading-stop`.

    **Important:** This uses a direct API call (`private_post`) and relies on Bybit's specific
    V5 endpoint and parameters. TSL settings take precedence over fixed SL. Use "0" to clear levels.

    Handles parameter validation (e.g., SL/TP/Activation relative to entry), price formatting
    to market precision, and API response checking.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        market_info (MarketInfo): The standardized MarketInfo dictionary.
        position_info (PositionInfo): The standardized PositionInfo dictionary for the open position.
        logger (logging.Logger): The logger instance for status messages.
        stop_loss_price (Optional[Decimal]): Desired fixed SL price. Set to 0 or None to clear/ignore.
        take_profit_price (Optional[Decimal]): Desired fixed TP price. Set to 0 or None to clear/ignore.
        trailing_stop_distance (Optional[Decimal]): Desired TSL distance (in price units). Set to 0 or None to clear/ignore. Must be positive if setting TSL.
        tsl_activation_price (Optional[Decimal]): Price at which TSL should activate. Required if distance > 0.

    Returns:
        bool: True if the protection was set/updated successfully via API or if no change was needed.
              False if validation fails, API call fails after retries, or a critical error occurs.
    """
    lg = logger
    endpoint = '/v5/position/set-trading-stop' # Bybit V5 endpoint

    # --- Input and State Validation ---
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        return False # Cannot set SL/TP/TSL on spot
    if not position_info:
        lg.error(f"Protection setting failed for {symbol}: Missing position information.")
        return False
    pos_side = position_info.get('side')
    entry_price_any = position_info.get('entryPrice') # Can be Decimal or None
    if pos_side not in ['long', 'short']:
        lg.error(f"Protection setting failed for {symbol}: Invalid position side ('{pos_side}').")
        return False
    try:
        if entry_price_any is None: raise ValueError("Missing entry price")
        entry_price = Decimal(str(entry_price_any)) # Ensure Decimal
        if entry_price <= 0: raise ValueError("Entry price must be positive")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Protection setting failed for {symbol}: Invalid or missing entry price ('{entry_price_any}'): {e}")
        return False
    try:
        price_tick = market_info['price_precision_step_decimal']
        if price_tick is None or price_tick <= 0: raise ValueError("Invalid price tick size")
    except (KeyError, ValueError, TypeError) as e:
         lg.error(f"Protection setting failed for {symbol}: Could not get valid price precision: {e}")
         return False

    params_to_set: Dict[str, Any] = {} # Parameters to send in the API request
    log_parts: List[str] = [f"{BRIGHT}Attempting protection set ({symbol} {pos_side.upper()} @ {entry_price.normalize()}):{RESET}"]
    any_protection_requested = False # Flag to check if any valid protection was requested
    set_tsl_active = False # Flag if TSL distance > 0 is being set

    # --- Format and Validate Protection Parameters ---
    try:
        # Helper to format price to exchange precision string using the global helper
        def format_param(price_decimal: Optional[Decimal], param_name: str) -> Optional[str]:
            """Formats price to string, returns None if invalid/negative."""
            if price_decimal is None: return None
            # Allow "0" to clear protection
            if price_decimal == 0: return "0"
            # Use the global formatter which handles precision and positivity check
            return _format_price(exchange, market_info['symbol'], price_decimal)

        # --- Trailing Stop Loss (TSL) ---
        # Bybit requires TSL distance (trailingStop) and activation price (activePrice)
        if isinstance(trailing_stop_distance, Decimal):
            any_protection_requested = True
            if trailing_stop_distance > 0: # Setting an active TSL
                # Ensure distance is at least one tick
                min_valid_distance = max(trailing_stop_distance, price_tick)

                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0:
                    lg.error(f"TSL failed ({symbol}): Valid positive activation price required for TSL distance > 0.")
                else:
                    # Validate activation price makes sense relative to entry (must be beyond entry)
                    is_valid_activation = (pos_side == 'long' and tsl_activation_price > entry_price) or \
                                          (pos_side == 'short' and tsl_activation_price < entry_price)
                    if not is_valid_activation:
                        lg.error(f"TSL failed ({symbol}): Activation {tsl_activation_price.normalize()} invalid vs entry {entry_price.normalize()} for {pos_side}.")
                    else:
                        fmt_distance = format_param(min_valid_distance, "TSL Distance")
                        fmt_activation = format_param(tsl_activation_price, "TSL Activation")
                        if fmt_distance and fmt_activation:
                            params_to_set['trailingStop'] = fmt_distance
                            params_to_set['activePrice'] = fmt_activation
                            log_parts.append(f"  - Setting TSL: Dist={fmt_distance}, Act={fmt_activation}")
                            set_tsl_active = True # Mark TSL as being actively set
                        else:
                            lg.error(f"TSL failed ({symbol}): Could not format params (Dist: {fmt_distance}, Act: {fmt_activation}).")
            elif trailing_stop_distance == 0: # Clearing TSL
                params_to_set['trailingStop'] = "0"
                # Also clear activation price when clearing TSL distance for Bybit
                params_to_set['activePrice'] = "0"
                log_parts.append("  - Clearing TSL (Dist & Act Price set to 0)")
                # set_tsl_active remains False
            else: # Negative distance
                 lg.warning(f"Invalid negative TSL distance ({trailing_stop_distance.normalize()}) for {symbol}. Ignoring TSL.")

        # --- Fixed Stop Loss (SL) ---
        # Can only set fixed SL if TSL is *not* being actively set (Bybit limitation)
        if not set_tsl_active:
            if isinstance(stop_loss_price, Decimal):
                any_protection_requested = True
                if stop_loss_price > 0: # Setting an active SL
                    # Validate SL price makes sense relative to entry (must be beyond entry)
                    is_valid_sl = (pos_side == 'long' and stop_loss_price < entry_price) or \
                                  (pos_side == 'short' and stop_loss_price > entry_price)
                    if not is_valid_sl:
                        lg.error(f"SL failed ({symbol}): SL price {stop_loss_price.normalize()} invalid vs entry {entry_price.normalize()} for {pos_side}.")
                    else:
                        fmt_sl = format_param(stop_loss_price, "Stop Loss")
                        if fmt_sl:
                            params_to_set['stopLoss'] = fmt_sl
                            log_parts.append(f"  - Setting Fixed SL: {fmt_sl}")
                        else:
                            lg.error(f"SL failed ({symbol}): Could not format SL price {stop_loss_price.normalize()}.")
                elif stop_loss_price == 0: # Clearing SL
                    # Only send "0" if SL field wasn't already populated by TSL logic
                    if 'stopLoss' not in params_to_set:
                         params_to_set['stopLoss'] = "0"
                         log_parts.append("  - Clearing Fixed SL (set to 0)")
                # Negative SL price already handled by format_param
        elif isinstance(stop_loss_price, Decimal) and stop_loss_price > 0:
             # TSL is active, cannot set fixed SL
             lg.warning(f"Ignoring fixed SL request ({stop_loss_price.normalize()}) because active TSL is being set.")

        # --- Fixed Take Profit (TP) ---
        # TP can usually be set alongside SL or TSL
        if isinstance(take_profit_price, Decimal):
            any_protection_requested = True
            if take_profit_price > 0: # Setting an active TP
                # Validate TP price makes sense relative to entry (must be beyond entry)
                is_valid_tp = (pos_side == 'long' and take_profit_price > entry_price) or \
                              (pos_side == 'short' and take_profit_price < entry_price)
                if not is_valid_tp:
                    lg.error(f"TP failed ({symbol}): TP price {take_profit_price.normalize()} invalid vs entry {entry_price.normalize()} for {pos_side}.")
                else:
                    fmt_tp = format_param(take_profit_price, "Take Profit")
                    if fmt_tp:
                        params_to_set['takeProfit'] = fmt_tp
                        log_parts.append(f"  - Setting Fixed TP: {fmt_tp}")
                    else:
                        lg.error(f"TP failed ({symbol}): Could not format TP price {take_profit_price.normalize()}.")
            elif take_profit_price == 0: # Clearing TP
                 if 'takeProfit' not in params_to_set:
                     params_to_set['takeProfit'] = "0"
                     log_parts.append("  - Clearing Fixed TP (set to 0)")
            # Negative TP price already handled by format_param

    except Exception as validation_err:
        lg.error(f"Unexpected error during protection validation ({symbol}): {validation_err}", exc_info=True)
        return False

    # --- Check if any valid parameters were actually prepared for the API call ---
    if not params_to_set:
        if any_protection_requested:
            lg.warning(f"{NEON_YELLOW}Protection skipped ({symbol}): No valid parameters after validation. No API call made.{RESET}")
            return False # Return False because the requested action couldn't be fulfilled
        else:
            lg.debug(f"No protection changes requested ({symbol}). Skipping API.")
            return True # Success, as no action was needed

    # --- Prepare Final API Parameters ---
    category = market_info.get('contract_type_str', 'Linear').lower()
    market_id = market_info['id']
    # Get position index (should be 0 for one-way mode)
    position_idx = 0 # Default to one-way mode
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
        if position_idx != 0: lg.warning(f"Detected positionIdx={position_idx}. Ensure this matches Bybit mode (One-Way vs Hedge).")
    except (ValueError, TypeError): pass # Ignore parsing errors, use default 0

    # Construct the final parameters dictionary for the API call
    final_api_params: Dict[str, Any] = {
        'category': category,
        'symbol': market_id,
        'positionIdx': position_idx # Specify position index (0 for one-way)
    }
    final_api_params.update(params_to_set) # Add the specific SL/TP/TSL values

    # Add trigger/order type parameters (can be customized later if needed)
    final_api_params.update({
        'tpslMode': 'Full',         # Apply to entire position
        'slTriggerBy': 'LastPrice', # Trigger SL based on Last Price
        'tpTriggerBy': 'LastPrice', # Trigger TP based on Last Price
        'slOrderType': 'Market',    # Use Market order when SL is triggered
        'tpOrderType': 'Market',    # Use Market order when TP is triggered
    })

    # Log the attempt only if there are parameters to set
    lg.info("\n".join(log_parts)) # Log what is being attempted
    lg.debug(f"  Final API params for {endpoint} ({symbol}): {final_api_params}")

    # --- Execute API Call with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing private_post {endpoint} ({symbol}, Attempt {attempts + 1})...")
            # Use exchange.private_post for endpoints not directly mapped by CCXT standard methods
            response = exchange.private_post(endpoint, params=final_api_params)
            lg.debug(f"Raw response from {endpoint} ({symbol}): {response}")

            # --- Check Bybit V5 Response ---
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown msg')

            if ret_code == 0:
                 # Check message for "not modified" cases which are still success
                 no_change_msgs = ["not modified", "no need to modify", "parameter not change", "order is not modified"]
                 if any(m in ret_msg.lower() for m in no_change_msgs):
                     lg.info(f"{NEON_YELLOW}Protection parameters already set or no change needed ({symbol}). (Message: {ret_msg}){RESET}")
                 else:
                     lg.info(f"{NEON_GREEN}Protection set/updated successfully ({symbol}, Code: 0).{RESET}")
                 return True # Success

            else:
                 # Log the specific Bybit error and raise ExchangeError for retry or handling
                 error_message = f"Bybit API error setting protection ({symbol}): {ret_msg} (Code: {ret_code})"
                 lg.error(f"{NEON_RED}{error_message}{RESET}")
                 # Attach code to exception if possible
                 exc = ccxt.ExchangeError(error_message)
                 setattr(exc, 'code', ret_code) # Set attribute for easier checking
                 raise exc

        # --- Standard CCXT Error Handling with Retries ---
        except ccxt.ExchangeError as e:
            last_exception = e
            # Try to extract error code more reliably
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            err_str = str(e).lower()
            lg.warning(f"{NEON_YELLOW}Exchange error setting protection ({symbol}): {e} (Code: {err_code_str}). Retry {attempts + 1}...{RESET}")

            # Check for known fatal/non-retryable error codes/messages for set-trading-stop
            fatal_protect_codes = [
                '10001', '10002', '110013', '110036', '110043', '110084', '110085',
                '110086', '110103', '110104', '110110', '3400045',
                # Add more specific codes if encountered
                '3400048', '3400051', '3400052', '3400070', '3400071', '3400072', '3400073'
            ]
            fatal_messages = ["invalid parameter", "invalid price", "cannot be higher", "cannot be lower", "position status", "precision error", "activation price", "distance invalid", "cannot be the same"]

            if err_code_str in fatal_protect_codes or any(msg in err_str for msg in fatal_messages):
                 lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE protection error ({symbol}). Aborting protection set.{RESET}")
                 return False # Fatal error for this operation

            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries for ExchangeError setting protection ({symbol}).{RESET}")
                 return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Aetheric disturbance setting protection ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries for NetworkError setting protection ({symbol}).{RESET}")
                return False

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit setting protection ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Auth ritual failed setting protection ({symbol}): {e}. Stopping.{RESET}")
            return False # Fatal error for this operation

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected vortex setting protection ({symbol}): {e}{RESET}", exc_info=True)
            return False # Stop on unexpected errors

        # Increment attempt counter and delay before retrying (only for retryable errors)
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Increase delay per attempt

    lg.error(f"{NEON_RED}Failed to set protection ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
    return False

# --- Volumatic Trend + OB Strategy Implementation (Refactored into Class) ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic for the Volumatic Trend and Pivot Order Block strategy calculations.

    Responsibilities:
    - Calculates Volumatic Trend indicators (EMAs, ATR Bands based on trend change, Volume Normalization).
    - Identifies Pivot Highs/Lows based on configuration.
    - Creates Order Blocks (OBs) from pivots using configured source (Wicks/Body).
    - Manages the state of OBs (active, violated, extends).
    - Prunes the list of active OBs to a maximum number per type.
    - Returns structured analysis results including the processed DataFrame.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        """
        Initializes the strategy engine with parameters from the config.

        Args:
            config (Dict[str, Any]): The main configuration dictionary.
            market_info (MarketInfo): The standardized MarketInfo dictionary.
            logger (logging.Logger): The logger instance for this strategy instance.

        Raises:
            ValueError: If critical configuration parameters are invalid or missing.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger # Alias for convenience
        self.symbol = market_info.get('symbol', 'UnknownSymbol')

        strategy_cfg = config.get("strategy_params", {})

        # Load strategy parameters (already validated by load_config)
        try:
            self.vt_length = int(strategy_cfg["vt_length"])
            self.vt_atr_period = int(strategy_cfg["vt_atr_period"])
            self.vt_vol_ema_length = int(strategy_cfg["vt_vol_ema_length"])
            self.vt_atr_multiplier = Decimal(str(strategy_cfg["vt_atr_multiplier"]))

            self.ob_source = str(strategy_cfg["ob_source"]) # "Wicks" or "Body"
            self.ph_left = int(strategy_cfg["ph_left"])
            self.ph_right = int(strategy_cfg["ph_right"])
            self.pl_left = int(strategy_cfg["pl_left"])
            self.pl_right = int(strategy_cfg["pl_right"])
            self.ob_extend = bool(strategy_cfg["ob_extend"])
            self.ob_max_boxes = int(strategy_cfg["ob_max_boxes"])

            # Basic sanity checks on loaded values
            if not (self.vt_length > 0 and self.vt_atr_period > 0 and self.vt_vol_ema_length > 0 and \
                    self.vt_atr_multiplier > 0 and self.ph_left > 0 and self.ph_right > 0 and \
                    self.pl_left > 0 and self.pl_right > 0 and self.ob_max_boxes > 0):
                raise ValueError("One or more strategy parameters are invalid (must be positive).")
            if self.ob_source not in ["Wicks", "Body"]:
                 raise ValueError(f"Invalid ob_source '{self.ob_source}'. Must be 'Wicks' or 'Body'.")

        except (KeyError, ValueError, TypeError) as e:
            self.lg.error(f"FATAL: Failed to initialize VolumaticOBStrategy ({self.symbol}) due to invalid config: {e}")
            self.lg.debug(f"Strategy Config received: {strategy_cfg}")
            raise ValueError(f"Strategy initialization failed ({self.symbol}): {e}") from e

        # Initialize Order Block storage (maintained within the instance)
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Calculate minimum data length required based on longest lookback period
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length) # Use *2 for EMA stability buffer
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1)
        stabilization_buffer = 50 # General buffer for indicator stabilization
        self.min_data_len = max(required_for_vt, required_for_pivots) + stabilization_buffer

        # Log initialized parameters
        self.lg.info(f"{NEON_CYAN}--- Initializing VolumaticOB Strategy Engine ({self.symbol}) ---{RESET}")
        self.lg.info(f"  VT Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Vol EMA Length={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  OB Params: Source='{self.ob_source}', PH Lookback={self.ph_left}/{self.ph_right}, PL Lookback={self.pl_left}/{self.pl_right}, Extend OBs={self.ob_extend}, Max Active OBs={self.ob_max_boxes}")
        self.lg.info(f"  Minimum Historical Data Required: ~{self.min_data_len} candles")

        # Warning if required data exceeds typical API limits significantly
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 10: # Add small buffer to limit check
            self.lg.warning(f"{NEON_YELLOW}CONFIGURATION NOTE ({self.symbol}):{RESET} Strategy requires {self.min_data_len} candles, which might exceed the API fetch limit ({BYBIT_API_KLINE_LIMIT}) in a single request. "
                          f"Ensure 'fetch_limit' in config.json is sufficient or consider reducing long lookback periods (vt_atr_period, vt_vol_ema_length).")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates EMA(SWMA(series, 4), length) using float for performance."""
        if not isinstance(series, pd.Series) or len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index, dtype=float)
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isnull().all(): return pd.Series(np.nan, index=series.index, dtype=float)
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        swma = numeric_series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan)
        return ema_of_swma

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, is_high: bool) -> pd.Series:
        """Identifies Pivot Highs/Lows using strict inequality and float comparison."""
        if not isinstance(series, pd.Series) or series.empty or left_bars < 1 or right_bars < 1:
            return pd.Series(False, index=series.index, dtype=bool)
        num_series = pd.to_numeric(series, errors='coerce')
        if num_series.isnull().all(): return pd.Series(False, index=series.index, dtype=bool)

        pivot_conditions = num_series.notna()
        # Check left bars
        for i in range(1, left_bars + 1):
            shifted = num_series.shift(i)
            condition = (num_series > shifted) if is_high else (num_series < shifted)
            pivot_conditions &= condition.fillna(False) # Exclude NaNs from comparison
        # Check right bars
        for i in range(1, right_bars + 1):
            shifted = num_series.shift(-i)
            condition = (num_series > shifted) if is_high else (num_series < shifted)
            pivot_conditions &= condition.fillna(False) # Exclude NaNs from comparison

        return pivot_conditions.fillna(False)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes historical OHLCV data to calculate indicators and manage Order Blocks.

        Args:
            df_input (pd.DataFrame): Input DataFrame with OHLCV data (Decimals).

        Returns:
            StrategyAnalysisResults: Structured results including processed DataFrame, trend, OBs, etc.
        """
        # Prepare a default/empty result structure for failure cases
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('NaN'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if df_input.empty:
            self.lg.error(f"Strategy update failed ({self.symbol}): Input DataFrame is empty.")
            return empty_results

        df = df_input.copy() # Work on a copy

        # --- Input Data Validation ---
        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
            self.lg.error(f"Strategy update failed ({self.symbol}): DataFrame index must be a monotonic DatetimeIndex.")
            df.sort_index(inplace=True) # Attempt sort
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.lg.error(f"Strategy update failed ({self.symbol}): Missing required columns: {required_cols}.")
            return empty_results
        if len(df) < self.min_data_len:
            self.lg.warning(f"Strategy update ({self.symbol}): Insufficient data ({len(df)} < ~{self.min_data_len} required). Results may be inaccurate.")
            # Proceed but warn

        self.lg.debug(f"Starting strategy analysis ({self.symbol}) on {len(df)} candles.")

        # --- Convert to Float for TA Libraries ---
        try:
            df_float = pd.DataFrame(index=df.index)
            for col in required_cols:
                df_float[col] = pd.to_numeric(df[col], errors='coerce')
            initial_float_len = len(df_float)
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if len(df_float) < initial_float_len:
                 self.lg.debug(f"Dropped {initial_float_len - len(df_float)} rows ({self.symbol}) due to NaN OHLC after float conversion.")
            if df_float.empty:
                self.lg.error(f"Strategy update failed ({self.symbol}): DataFrame empty after float conversion.")
                return empty_results
        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error converting to float: {e}", exc_info=True)
            return empty_results

        # --- Indicator Calculations (using df_float) ---
        try:
            self.lg.debug(f"Calculating indicators ({self.symbol})...")
            # ATR
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)

            # Volumatic Trend EMAs
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length) # EMA(SWMA(close))
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan) # Standard EMA(close)

            # Determine Trend Direction (ema2 crosses previous ema1)
            valid_comparison = df_float['ema2'].notna() & df_float['ema1'].shift(1).notna()
            trend_up_series = pd.Series(np.nan, index=df_float.index, dtype=object)
            trend_up_series[valid_comparison] = df_float['ema2'] > df_float['ema1'].shift(1)
            trend_up_series.ffill(inplace=True) # Fill initial NaNs

            # Identify Trend Changes
            trend_changed_series = (trend_up_series != trend_up_series.shift(1)) & \
                                   trend_up_series.notna() & trend_up_series.shift(1).notna()
            df_float['trend_changed'] = trend_changed_series.fillna(False).astype(bool)
            df_float['trend_up'] = trend_up_series.astype(bool) # Final boolean trend

            # Capture EMA1 and ATR at the point of trend change for band calculation
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Volumatic Trend Bands
            atr_multiplier_float = float(self.vt_atr_multiplier)
            valid_band_calc = df_float['ema1_for_bands'].notna() & df_float['atr_for_bands'].notna()
            df_float['upper_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_multiplier_float), np.nan)
            df_float['lower_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_multiplier_float), np.nan)

            # Volume Normalization (using EMA of volume)
            vol_ema = ta.ema(df_float['volume'].fillna(0.0), length=self.vt_vol_ema_length, fillna=np.nan)
            # Avoid division by zero or near-zero EMA
            vol_ema_safe = vol_ema.replace(0, np.nan).fillna(method='bfill').fillna(1e-9) # Replace 0, backfill, then fill remaining with small number
            df_float['vol_norm'] = (df_float['volume'].fillna(0.0) / vol_ema_safe) * 100.0
            # Clip volume normalization (e.g., 0-200%) and convert to integer
            df_float['vol_norm_int'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0).astype(int)

            # Pivot High/Low Calculation
            high_series = df_float['high'] if self.ob_source == "Wicks" else df_float[['open', 'close']].max(axis=1)
            low_series = df_float['low'] if self.ob_source == "Wicks" else df_float[['open', 'close']].min(axis=1)
            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, is_high=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, is_high=False)

            self.lg.debug(f"Indicator calculations complete ({self.symbol}) (float).")

        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error during indicator calculation: {e}", exc_info=True)
            return empty_results

        # --- Copy Calculated Float Results back to Original Decimal DataFrame ---
        try:
            self.lg.debug(f"Converting calculated indicators back to Decimal format ({self.symbol})...")
            indicator_cols_numeric = ['atr', 'ema1', 'ema2', 'upper_band', 'lower_band', 'vol_norm'] # Keep vol_norm as Decimal for now
            indicator_cols_int = ['vol_norm_int']
            indicator_cols_bool = ['trend_up', 'trend_changed', 'is_ph', 'is_pl']

            for col in indicator_cols_numeric:
                if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)
                    df[col] = source_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            for col in indicator_cols_int:
                if col in df_float.columns:
                     source_series = df_float[col].reindex(df.index)
                     df[col] = source_series.fillna(0).astype(int) # Fill NaNs with 0 for int conversion
            for col in indicator_cols_bool:
                 if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)
                    if col == 'trend_up': # Keep None/NaN for trend_up if calculation failed
                         df[col] = source_series.apply(lambda x: bool(x) if pd.notna(x) else None).astype('boolean') # Use nullable boolean type
                    else:
                         df[col] = source_series.fillna(False).astype(bool)

        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error converting indicators to Decimal: {e}", exc_info=True)
            # Continue if possible, OB management might still work partially

        # --- Clean Final Decimal DataFrame ---
        initial_len_final = len(df)
        # Require valid close, ATR, pivots, and a determined trend (not None)
        essential_cols = ['close', 'atr', 'is_ph', 'is_pl', 'trend_up']
        df.dropna(subset=essential_cols, inplace=True)
        # Additional check for positive ATR
        df = df[df['atr'] > Decimal('0')]

        rows_dropped_final = initial_len_final - len(df)
        if rows_dropped_final > 0:
            self.lg.debug(f"Dropped {rows_dropped_final} rows ({self.symbol}) from final DataFrame due to missing indicators (likely at start).")

        if df.empty:
            self.lg.warning(f"Strategy update ({self.symbol}): DataFrame empty after final indicator cleaning.")
            empty_results['dataframe'] = df # Return the empty df
            return empty_results

        self.lg.debug(f"Indicators finalized in Decimal DataFrame ({self.symbol}). Processing Order Blocks...")

        # --- Order Block Management ---
        try:
            new_ob_count = 0
            violated_ob_count = 0
            last_candle_idx = df.index[-1]

            # --- Identify New Order Blocks ---
            new_bull_candidates: List[OrderBlock] = []
            new_bear_candidates: List[OrderBlock] = []
            existing_ob_ids = {ob['id'] for ob in self.bull_boxes + self.bear_boxes}

            for timestamp, candle in df.iterrows():
                # Create Bearish OB from Pivot High
                if candle.get('is_ph'):
                    ob_id = f"B_{timestamp.strftime('%y%m%d%H%M%S')}"
                    if ob_id not in existing_ob_ids: # Check if this pivot already created an OB
                        ob_top = candle['high'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                        ob_bottom = candle['open'] if self.ob_source == "Wicks" else min(candle['open'], candle['close']) # Corrected: Use open for Bearish wick source
                        if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom:
                            new_bear_candidates.append(OrderBlock(
                                id=ob_id, type='bear', timestamp=timestamp, top=ob_top, bottom=ob_bottom,
                                active=True, violated=False, violation_ts=None, extended_to_ts=timestamp
                            ))
                            new_ob_count += 1
                            existing_ob_ids.add(ob_id) # Add immediately to prevent duplicates within this run

                # Create Bullish OB from Pivot Low
                if candle.get('is_pl'):
                    ob_id = f"L_{timestamp.strftime('%y%m%d%H%M%S')}"
                    if ob_id not in existing_ob_ids:
                        ob_top = candle['open'] if self.ob_source == "Wicks" else max(candle['open'], candle['close']) # Corrected: Use open for Bullish wick source
                        ob_bottom = candle['low'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                        if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom:
                            new_bull_candidates.append(OrderBlock(
                                id=ob_id, type='bull', timestamp=timestamp, top=ob_top, bottom=ob_bottom,
                                active=True, violated=False, violation_ts=None, extended_to_ts=timestamp
                            ))
                            new_ob_count += 1
                            existing_ob_ids.add(ob_id)

            # Add new candidates to the main lists
            self.bull_boxes.extend(new_bull_candidates)
            self.bear_boxes.extend(new_bear_candidates)
            if new_ob_count > 0:
                self.lg.debug(f"Identified {new_ob_count} new potential Order Blocks ({self.symbol}).")

            # --- Manage Existing and New Order Blocks (Violations, Extend) ---
            # Iterate through all current boxes and check against relevant history
            all_boxes = self.bull_boxes + self.bear_boxes
            for box in all_boxes:
                 if not box['active']: continue # Skip already inactive boxes

                 # Find candles after the box was formed
                 relevant_candles = df[df.index > box['timestamp']]
                 for ts, candle in relevant_candles.iterrows():
                      close_price = candle.get('close')
                      if isinstance(close_price, Decimal) and close_price.is_finite():
                           # Check for violation
                           violated = False
                           if box['type'] == 'bull' and close_price < box['bottom']: violated = True
                           elif box['type'] == 'bear' and close_price > box['top']: violated = True

                           if violated:
                                box['active'] = False
                                box['violated'] = True
                                box['violation_ts'] = ts
                                violated_ob_count += 1
                                self.lg.debug(f"{box['type'].capitalize()} OB {box['id']} VIOLATED ({self.symbol}) at {ts.strftime('%H:%M')} by close {close_price.normalize()}")
                                break # Stop checking this box once violated
                           elif self.ob_extend:
                                # Extend active box to this timestamp
                                box['extended_to_ts'] = ts
                      # else: lg.warning(f"Invalid close price at {ts} for OB check.")

            if violated_ob_count > 0:
                 self.lg.debug(f"Processed violations for {violated_ob_count} Order Blocks ({self.symbol}).")

            # --- Prune Order Blocks ---
            # Keep only the 'ob_max_boxes' most recent *active* ones per type
            self.bull_boxes = sorted([b for b in self.bull_boxes if b['active']], key=lambda b: b['timestamp'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted([b for b in self.bear_boxes if b['active']], key=lambda b: b['timestamp'], reverse=True)[:self.ob_max_boxes]
            self.lg.debug(f"Pruned Order Blocks ({self.symbol}). Kept Active: Bulls={len(self.bull_boxes)}, Bears={len(self.bear_boxes)} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error during Order Block processing: {e}", exc_info=True)
            # Continue, but OBs might be inaccurate

        # --- Prepare Final StrategyAnalysisResults ---
        last_candle_final = df.iloc[-1] if not df.empty else None

        # Helper functions to safely extract values from the last candle
        def safe_decimal_from_candle(col_name: str, positive_only: bool = False) -> Optional[Decimal]:
            if last_candle_final is None: return None
            value = last_candle_final.get(col_name)
            if isinstance(value, Decimal) and value.is_finite():
                 return value if not positive_only or value > Decimal('0') else None
            return None

        def safe_bool_from_candle(col_name: str) -> Optional[bool]:
            if last_candle_final is None: return None
            value = last_candle_final.get(col_name)
            return bool(value) if pd.notna(value) else None # pd.notna handles None, NaN, NA

        def safe_int_from_candle(col_name: str) -> Optional[int]:
             if last_candle_final is None: return None
             value = last_candle_final.get(col_name)
             try: return int(value) if pd.notna(value) else None
             except (ValueError, TypeError): return None

        # Construct the results dictionary
        final_dataframe = df # Return the fully processed DataFrame with Decimals
        last_close_val = safe_decimal_from_candle('close') or Decimal('NaN') # Use NaN if invalid
        current_trend_val = safe_bool_from_candle('trend_up') # Can be None
        trend_changed_val = bool(safe_bool_from_candle('trend_changed')) # Default False if None/NaN
        vol_norm_int_val = safe_int_from_candle('vol_norm_int')
        atr_val = safe_decimal_from_candle('atr', positive_only=True) # ATR must be positive
        upper_band_val = safe_decimal_from_candle('upper_band')
        lower_band_val = safe_decimal_from_candle('lower_band')

        analysis_results = StrategyAnalysisResults(
            dataframe=final_dataframe,
            last_close=last_close_val,
            current_trend_up=current_trend_val,
            trend_just_changed=trend_changed_val,
            active_bull_boxes=self.bull_boxes, # Return the pruned list of active OBs
            active_bear_boxes=self.bear_boxes,
            vol_norm_int=vol_norm_int_val,
            atr=atr_val,
            upper_band=upper_band_val,
            lower_band=lower_band_val
        )

        # Log summary of the final results for the *last* candle
        trend_str = f"{NEON_GREEN}UP{RESET}" if analysis_results['current_trend_up'] is True else \
                    f"{NEON_RED}DOWN{RESET}" if analysis_results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}Undetermined{RESET}"
        atr_str = f"{analysis_results['atr'].normalize()}" if analysis_results['atr'] else "N/A"
        time_str = last_candle_final.name.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_final is not None else "N/A"

        self.lg.debug(f"--- Strategy Analysis Results ({self.symbol} @ {time_str}) ---")
        self.lg.debug(f"  Last Close: {analysis_results['last_close'].normalize() if analysis_results['last_close'].is_finite() else 'NaN'}")
        self.lg.debug(f"  Trend: {trend_str} (Changed on this candle: {analysis_results['trend_just_changed']})")
        self.lg.debug(f"  ATR: {atr_str}")
        self.lg.debug(f"  Volume Norm (%): {analysis_results['vol_norm_int']}")
        self.lg.debug(f"  Bands (Upper/Lower): {analysis_results['upper_band'].normalize() if analysis_results['upper_band'] else 'N/A'} / {analysis_results['lower_band'].normalize() if analysis_results['lower_band'] else 'N/A'}")
        self.lg.debug(f"  Active OBs (Bull/Bear): {len(analysis_results['active_bull_boxes'])} / {len(analysis_results['active_bear_boxes'])}")
        # Optionally log the details of the active OBs at DEBUG level
        # for ob in analysis_results['active_bull_boxes']: self.lg.debug(f"    Bull OB: {ob['id']} [{ob['bottom'].normalize()} - {ob['top'].normalize()}]")
        # for ob in analysis_results['active_bear_boxes']: self.lg.debug(f"    Bear OB: {ob['id']} [{ob['bottom'].normalize()} - {ob['top'].normalize()}]")
        self.lg.debug(f"---------------------------------------------")

        return analysis_results

# --- Signal Generation based on Strategy Results (Refactored into Class) ---
class SignalGenerator:
    """
    Generates trading signals based on strategy analysis and position state.

    Responsibilities:
    - Evaluates `StrategyAnalysisResults` against entry/exit rules.
    - Considers the current open position (if any).
    - Generates signals: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD".
    - Calculates initial Stop Loss (SL) and Take Profit (TP) levels for new entries.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the Signal Generator with parameters from the config.

        Args:
            config (Dict[str, Any]): The main configuration dictionary.
            logger (logging.Logger): The logger instance.

        Raises:
            ValueError: If critical configuration parameters are invalid or missing.
        """
        self.config = config
        self.logger = logger
        self.lg = logger # Alias
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})

        try:
            # Load parameters used for signal generation and SL/TP calculation
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))

            # Basic validation
            if not (self.ob_entry_proximity_factor >= 1): raise ValueError("ob_entry_proximity_factor must be >= 1.0")
            if not (self.ob_exit_proximity_factor >= 1): raise ValueError("ob_exit_proximity_factor must be >= 1.0")
            if not (self.initial_tp_atr_multiple >= 0): raise ValueError("initial_take_profit_atr_multiple must be >= 0")
            if not (self.initial_sl_atr_multiple > 0): raise ValueError("initial_stop_loss_atr_multiple must be > 0")

            self.lg.info(f"{NEON_CYAN}--- Initializing Signal Generator ---{RESET}")
            self.lg.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor.normalize()}")
            self.lg.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor.normalize()}")
            self.lg.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()}")
            self.lg.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")
            self.lg.info(f"-----------------------------------")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.lg.error(f"{NEON_RED}FATAL: Error initializing SignalGenerator parameters from config: {e}.{RESET}", exc_info=True)
             raise ValueError(f"SignalGenerator initialization failed: {e}") from e

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[PositionInfo], symbol: str) -> str:
        """
        Determines the trading signal based on strategy analysis and current position.

        Logic Flow:
        1. Validate inputs from analysis_results.
        2. If position exists, check for Exit conditions (trend flip, OB proximity).
        3. If no position exists, check for Entry conditions (trend alignment, OB proximity).
        4. Default to HOLD if no entry/exit conditions met.

        Args:
            analysis_results (StrategyAnalysisResults): The results from `VolumaticOBStrategy.update()`.
            open_position (Optional[PositionInfo]): Standardized `PositionInfo` dict, or None if no position.
            symbol (str): The trading symbol (for logging).

        Returns:
            str: The generated signal string: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or "HOLD".
        """
        lg = self.logger

        # --- Validate Input ---
        # Check for essential valid results from analysis
        if not analysis_results or \
           analysis_results['current_trend_up'] is None or \
           not analysis_results['last_close'].is_finite() or \
           analysis_results['last_close'] <= 0 or \
           analysis_results['atr'] is None or not analysis_results['atr'].is_finite() or analysis_results['atr'] <= 0:
            lg.warning(f"{NEON_YELLOW}Signal Generation ({symbol}): Invalid or incomplete strategy analysis results. Defaulting to HOLD.{RESET}")
            lg.debug(f"  Problematic Analysis: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
            return "HOLD"

        # Extract key values for easier access
        last_close = analysis_results['last_close']
        trend_is_up = analysis_results['current_trend_up'] # This is boolean (cannot be None after checks)
        trend_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        position_side = open_position['side'] if open_position else None

        signal: str = "HOLD" # Default signal

        lg.debug(f"--- Signal Generation Check ({symbol}) ---")
        trend_log = 'UP' if trend_is_up else 'DOWN'
        lg.debug(f"  Input: Close={last_close.normalize()}, Trend={trend_log}, TrendChanged={trend_changed}, Position={position_side or 'None'}")
        lg.debug(f"  Active OBs: Bull={len(active_bull_obs)}, Bear={len(active_bear_obs)}")

        # --- 1. Check Exit Conditions (if position exists) ---
        if position_side == 'long':
            # Exit Long if trend flips down *on the last candle*
            if trend_is_up is False and trend_changed:
                signal = "EXIT_LONG"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal ({symbol}): Trend flipped to DOWN on last candle.{RESET}")
            # Exit Long if price violates (gets near/below) an active Bullish OB (use exit proximity factor)
            elif active_bull_obs and signal == "HOLD": # Check only if not already exiting
                try:
                    for ob in active_bull_obs:
                         # Exit threshold: Price <= OB Bottom * Exit Proximity Factor
                         exit_threshold = ob['bottom'] * self.ob_exit_proximity_factor
                         if last_close <= exit_threshold:
                              signal = "EXIT_LONG"
                              lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal ({symbol}): Price {last_close.normalize()} <= Bull OB exit threshold {exit_threshold.normalize()} (OB ID: {ob['id']}, Bottom: {ob['bottom'].normalize()}){RESET}")
                              break # Exit on first violated OB
                except (InvalidOperation, Exception) as e:
                    lg.warning(f"Error during Bullish OB exit check ({symbol}, long): {e}")

        elif position_side == 'short':
            # Exit Short if trend flips up *on the last candle*
            if trend_is_up is True and trend_changed:
                signal = "EXIT_SHORT"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal ({symbol}): Trend flipped to UP on last candle.{RESET}")
            # Exit Short if price violates (gets near/above) an active Bearish OB (use exit proximity factor)
            elif active_bear_obs and signal == "HOLD": # Check only if not already exiting
                try:
                    for ob in active_bear_obs:
                         # Exit threshold: Price >= OB Top / Exit Proximity Factor
                         # Ensure factor is > 0 before division
                         if self.ob_exit_proximity_factor <= 0: continue # Skip if factor invalid
                         exit_threshold = ob['top'] / self.ob_exit_proximity_factor
                         if last_close >= exit_threshold:
                              signal = "EXIT_SHORT"
                              lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal ({symbol}): Price {last_close.normalize()} >= Bear OB exit threshold {exit_threshold.normalize()} (OB ID: {ob['id']}, Top: {ob['top'].normalize()}){RESET}")
                              break # Exit on first violated OB
                except (ZeroDivisionError, InvalidOperation, Exception) as e:
                    lg.warning(f"Error during Bearish OB exit check ({symbol}, short): {e}")

        # If an exit signal was generated, return it immediately
        if signal != "HOLD":
            lg.debug(f"--- Signal Result ({symbol}): {signal} (Exit Condition Met) ---")
            return signal

        # --- 2. Check Entry Conditions (if NO position exists) ---
        if position_side is None:
            # Check for BUY signal: Trend is UP and price is within a Bullish OB's proximity
            if trend_is_up is True and active_bull_obs:
                for ob in active_bull_obs:
                    try:
                        # Entry zone: OB Bottom <= Price <= OB Top * Entry Proximity Factor
                        entry_zone_bottom = ob['bottom']
                        entry_zone_top = ob['top'] * self.ob_entry_proximity_factor
                        if entry_zone_bottom <= last_close <= entry_zone_top:
                            signal = "BUY"
                            lg.info(f"{NEON_GREEN}{BRIGHT}BUY Signal ({symbol}): Trend UP & Price {last_close.normalize()} within Bull OB entry zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}] (OB ID: {ob['id']}){RESET}")
                            break # Take the first valid entry signal found
                    except (InvalidOperation, Exception) as e:
                         lg.warning(f"Error checking Bull OB {ob.get('id')} ({symbol}) for entry: {e}")

            # Check for SELL signal: Trend is DOWN and price is within a Bearish OB's proximity
            elif trend_is_up is False and active_bear_obs:
                for ob in active_bear_obs:
                    try:
                        # Entry zone: OB Bottom / Entry Proximity Factor <= Price <= OB Top
                        if self.ob_entry_proximity_factor <= 0: continue # Skip if factor invalid
                        entry_zone_bottom = ob['bottom'] / self.ob_entry_proximity_factor
                        entry_zone_top = ob['top']
                        if entry_zone_bottom <= last_close <= entry_zone_top:
                            signal = "SELL"
                            lg.info(f"{NEON_RED}{BRIGHT}SELL Signal ({symbol}): Trend DOWN & Price {last_close.normalize()} within Bear OB entry zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}] (OB ID: {ob['id']}){RESET}")
                            break # Take the first valid entry signal found
                    except (ZeroDivisionError, InvalidOperation, Exception) as e:
                         lg.warning(f"Error checking Bear OB {ob.get('id')} ({symbol}) for entry: {e}")

        # --- 3. Default to HOLD ---
        if signal == "HOLD":
            lg.debug(f"Signal ({symbol}): HOLD - No valid entry or exit conditions met.")

        lg.debug(f"--- Signal Result ({symbol}): {signal} ---")
        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: MarketInfo, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial Take Profit (TP) and Stop Loss (SL) levels for a new entry.

        Uses entry price, current ATR, configured multipliers, and market price precision.
        Ensures SL/TP levels are strictly beyond the entry price after formatting.

        Args:
            entry_price (Decimal): The estimated or actual entry price (positive Decimal).
            signal (str): The entry signal ("BUY" or "SELL").
            atr (Decimal): The current Average True Range value (positive Decimal).
            market_info (MarketInfo): The standardized MarketInfo dictionary.
            exchange (ccxt.Exchange): The initialized ccxt.Exchange object (for price formatting).

        Returns:
            Tuple[Optional[Decimal], Optional[Decimal]]: A tuple containing:
                - Calculated Take Profit price (Decimal), or None if disabled or calculation fails.
                - Calculated Stop Loss price (Decimal), or None if calculation fails critically.
            Returns (None, None) if inputs are invalid or critical errors occur.
        """
        lg = self.logger
        symbol = market_info['symbol']
        lg.debug(f"Calculating Initial TP/SL ({symbol}) for {signal} signal at entry {entry_price.normalize()} with ATR {atr.normalize()}")

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            lg.error(f"TP/SL Calc Failed ({symbol}): Invalid signal '{signal}'.")
            return None, None
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
             lg.error(f"TP/SL Calc Failed ({symbol}): Entry price ({entry_price}) must be a valid positive Decimal.")
             return None, None
        if not isinstance(atr, Decimal) or not atr.is_finite() or atr <= 0:
            lg.error(f"TP/SL Calc Failed ({symbol}): ATR ({atr}) must be a valid positive Decimal.")
            return None, None
        try:
            price_tick = market_info['price_precision_step_decimal']
            if price_tick is None or price_tick <= 0: raise ValueError("Invalid price tick size")
        except (KeyError, ValueError, TypeError) as e:
            lg.error(f"TP/SL Calc Failed ({symbol}): Could not get valid price precision: {e}")
            return None, None

        # --- Calculate Raw TP/SL ---
        try:
            tp_atr_multiple = self.initial_tp_atr_multiple # Already Decimal from init
            sl_atr_multiple = self.initial_sl_atr_multiple # Already Decimal from init

            # Calculate offsets
            tp_offset = atr * tp_atr_multiple
            sl_offset = atr * sl_atr_multiple

            # Calculate raw levels
            take_profit_raw: Optional[Decimal] = None
            if tp_atr_multiple > 0: # Only calculate TP if multiplier is positive
                 take_profit_raw = (entry_price + tp_offset) if signal == "BUY" else (entry_price - tp_offset)

            stop_loss_raw = (entry_price - sl_offset) if signal == "BUY" else (entry_price + sl_offset)

            lg.debug(f"  Raw Levels ({symbol}): TP={take_profit_raw.normalize() if take_profit_raw else 'N/A'}, SL={stop_loss_raw.normalize()}")

            # --- Format Levels to Market Precision ---
            # Helper function to format and validate price level
            def format_level(price_decimal: Optional[Decimal], level_name: str) -> Optional[Decimal]:
                """Formats price to exchange precision, returns positive Decimal or None."""
                if price_decimal is None or not price_decimal.is_finite() or price_decimal <= 0:
                    lg.debug(f"Calculated {level_name} ({symbol}) is invalid or zero/negative ({price_decimal}).")
                    return None
                try:
                    # Use the global helper which handles precision and returns string
                    formatted_str = _format_price(exchange, symbol, price_decimal)
                    if formatted_str:
                         return Decimal(formatted_str) # Convert back to Decimal
                    else:
                         lg.warning(f"Formatted {level_name} ({symbol}) failed. Input: {price_decimal.normalize()}.")
                         return None
                except Exception as e:
                    lg.error(f"Error formatting {level_name} ({symbol}) value {price_decimal.normalize()}: {e}.")
                    return None

            # Format TP and SL
            take_profit_final = format_level(take_profit_raw, "Take Profit")
            stop_loss_final = format_level(stop_loss_raw, "Stop Loss")

            # --- Final Adjustments and Validation ---
            # Ensure SL is strictly beyond entry after formatting
            if stop_loss_final is not None:
                sl_invalid = (signal == "BUY" and stop_loss_final >= entry_price) or \
                             (signal == "SELL" and stop_loss_final <= entry_price)
                if sl_invalid:
                    lg.warning(f"Formatted {signal} Stop Loss ({symbol}) {stop_loss_final.normalize()} is not strictly beyond entry {entry_price.normalize()}. Adjusting by one tick.")
                    # Adjust SL by one tick further away from entry
                    adjusted_sl_raw = (stop_loss_final - price_tick) if signal == "BUY" else (stop_loss_final + price_tick)
                    stop_loss_final = format_level(adjusted_sl_raw, "Adjusted Stop Loss") # Reformat
                    if stop_loss_final is None or stop_loss_final <= 0:
                         lg.error(f"{NEON_RED}CRITICAL ({symbol}): Failed to calculate valid adjusted SL after initial SL was invalid.{RESET}")
                         return take_profit_final, None # Return None for SL is critical failure

            # Ensure TP is strictly beyond entry (if TP is enabled and calculated)
            if take_profit_final is not None:
                tp_invalid = (signal == "BUY" and take_profit_final <= entry_price) or \
                             (signal == "SELL" and take_profit_final >= entry_price)
                if tp_invalid:
                    lg.warning(f"Formatted {signal} Take Profit ({symbol}) {take_profit_final.normalize()} is not strictly beyond entry {entry_price.normalize()}. Disabling TP for this entry.")
                    take_profit_final = None # Disable TP if it ends up on the wrong side

            # Log final calculated levels
            tp_log = take_profit_final.normalize() if take_profit_final else "None (Disabled or Calc Failed)"
            sl_log = stop_loss_final.normalize() if stop_loss_final else "None (Calc Failed!)"
            lg.info(f"  >>> Calculated Initial Levels ({symbol}): TP={tp_log}, SL={sl_log}")

            # Critical check: Ensure SL calculation was successful
            if stop_loss_final is None:
                lg.error(f"{NEON_RED}Stop Loss calculation failed critically ({symbol}). Cannot determine position size or place trade safely.{RESET}")
                return take_profit_final, None # Return None for SL

            return take_profit_final, stop_loss_final

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL ({symbol}): {e}{RESET}", exc_info=True)
            return None, None


# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: MarketInfo) -> None:
    """
    Performs one full cycle of analysis and trading logic for a single symbol.

    Args:
        exchange (ccxt.Exchange): Initialized ccxt.Exchange object.
        symbol (str): Trading symbol to analyze and trade.
        config (Dict[str, Any]): Main configuration dictionary.
        logger (logging.Logger): Logger instance for this symbol's activity.
        strategy_engine (VolumaticOBStrategy): Initialized strategy engine instance.
        signal_generator (SignalGenerator): Initialized signal generator instance.
        market_info (MarketInfo): Standardized MarketInfo dictionary for the symbol.
    """
    lg = logger
    lg.info(f"\n{BRIGHT}---=== Cycle Start: Analyzing {symbol} ({config['interval']} TF) ===---{RESET}")
    cycle_start_time = time.monotonic()

    # Log key config settings for this cycle
    prot_cfg = config.get("protection", {})
    strat_cfg = config.get("strategy_params", {})
    lg.debug(f"Cycle Config ({symbol}): Trading={'ENABLED' if config.get('enable_trading') else 'DISABLED'}, Sandbox={config.get('use_sandbox')}, "
             f"Risk={config.get('risk_per_trade'):.2%}, Lev={config.get('leverage')}x, "
             f"TSL={'ON' if prot_cfg.get('enable_trailing_stop') else 'OFF'} (Act%={prot_cfg.get('trailing_stop_activation_percentage'):.3%}, CB%={prot_cfg.get('trailing_stop_callback_rate'):.3%}), "
             f"BE={'ON' if prot_cfg.get('enable_break_even') else 'OFF'} (TrigATR={prot_cfg.get('break_even_trigger_atr_multiple')}, Offset={prot_cfg.get('break_even_offset_ticks')} ticks), "
             f"InitSL Mult={prot_cfg.get('initial_stop_loss_atr_multiple')}, InitTP Mult={prot_cfg.get('initial_take_profit_atr_multiple')}, "
             f"OB Source={strat_cfg.get('ob_source')}")

    # --- 1. Fetch Kline Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
        lg.critical(f"Invalid interval '{config['interval']}' ({symbol}). Cannot map to CCXT timeframe. Skipping.")
        return

    min_required_data = strategy_engine.min_data_len
    fetch_limit_from_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    fetch_limit_needed = max(min_required_data, fetch_limit_from_config)

    lg.info(f"Requesting {fetch_limit_needed} klines for {symbol} ({ccxt_interval}). (Strategy requires min: {min_required_data})")
    # Use the robust multi-request fetch function
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit_needed, logger=lg)
    fetched_count = len(klines_df)

    # --- 2. Validate Fetched Data ---
    if klines_df.empty or fetched_count < min_required_data:
        lg.error(f"Fetched only {fetched_count} klines for {symbol}, but strategy requires {min_required_data}. "
                 f"Analysis may be inaccurate or fail. Skipping cycle.")
        return

    # --- 3. Run Strategy Analysis ---
    lg.debug(f"Running strategy analysis engine ({symbol})...")
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
        lg.error(f"{NEON_RED}Strategy analysis update failed unexpectedly ({symbol}): {analysis_err}{RESET}", exc_info=True)
        return

    # Validate essential analysis results
    if not analysis_results or \
       analysis_results['current_trend_up'] is None or \
       not analysis_results['last_close'].is_finite() or \
       analysis_results['last_close'] <= 0 or \
       analysis_results['atr'] is None or not analysis_results['atr'].is_finite() or analysis_results['atr'] <= 0:
        lg.error(f"{NEON_RED}Strategy analysis ({symbol}) did not produce valid essential results. Skipping cycle.{RESET}")
        lg.debug(f"Problematic Analysis Results ({symbol}): Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
        return
    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr'] # Guaranteed valid positive Decimal here
    lg.info(f"Strategy Analysis Complete ({symbol}): Trend={'UP' if analysis_results['current_trend_up'] else 'DOWN'}, "
            f"Last Close={latest_close.normalize()}, ATR={current_atr.normalize()}")

    # --- 4. Get Current Market State (Price & Position) ---
    lg.debug(f"Fetching current market price and checking for open positions ({symbol})...")
    current_market_price = fetch_current_price_ccxt(exchange, symbol, lg)
    open_position: Optional[PositionInfo] = get_open_position(exchange, symbol, lg) # Returns standardized dict or None

    # Determine price to use for real-time checks (prefer live price, fallback to last close)
    price_for_checks: Optional[Decimal] = None
    if current_market_price and current_market_price > 0:
         price_for_checks = current_market_price
    elif latest_close > 0:
         price_for_checks = latest_close
         lg.debug(f"Using last kline close price ({symbol}, {latest_close.normalize()}) for checks as live price is unavailable/invalid.")
    else:
         lg.error(f"{NEON_RED}Cannot determine a valid current price ({symbol}, Live={current_market_price}, LastClose={latest_close}). Skipping position management/entry checks.{RESET}")
         # Allow cycle to finish (might still log HOLD) but don't proceed with trading actions
         price_for_checks = None


    # --- 5. Generate Trading Signal ---
    lg.debug(f"Generating trading signal ({symbol})...")
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position, symbol)
    except Exception as signal_err:
        lg.error(f"{NEON_RED}Signal generation failed unexpectedly ({symbol}): {signal_err}{RESET}", exc_info=True)
        return

    lg.info(f"Generated Signal ({symbol}): {BRIGHT}{signal}{RESET}")

    # --- 6. Trading Logic Execution ---
    trading_enabled = config.get("enable_trading", False)

    # --- Scenario: Trading Disabled ---
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading is DISABLED ({symbol}).{RESET} Analysis complete. Signal was: {signal}")
        if open_position is None and signal in ["BUY", "SELL"]: lg.info(f"  (Action if enabled: Would attempt to {signal} {symbol})")
        elif open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]: lg.info(f"  (Action if enabled: Would attempt to {signal} current {open_position['side']} position)")
        elif open_position: lg.info(f"  (Action if enabled: Would manage existing {open_position['side']} position)")
        # Log HOLD implicitly
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Analysis-Only Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    # ======================================
    # --- Trading IS Enabled Below Here ---
    # ======================================
    lg.info(f"{BRIGHT}Trading is ENABLED ({symbol}). Processing signal '{signal}'...{RESET}")

    # --- Scenario 1: No Position -> Consider Entry ---
    if open_position is None and signal in ["BUY", "SELL"]:
        lg.info(f"{BRIGHT}*** {signal} Signal & No Position ({symbol}): Initiating Entry Sequence... ***{RESET}")

        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= 0:
            lg.error(f"{NEON_RED}Entry Aborted ({symbol} {signal}): Cannot fetch valid balance ({balance}).{RESET}")
            return

        initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(latest_close, signal, current_atr, market_info, exchange)
        if initial_sl_calc is None:
            lg.error(f"{NEON_RED}Entry Aborted ({symbol} {signal}): Failed to calculate valid initial SL.{RESET}")
            return
        if initial_tp_calc is None: lg.info(f"Initial TP calculation failed or disabled ({symbol}). Proceeding without initial TP.")

        if market_info['is_contract']:
            leverage_to_set = int(config.get('leverage', 0))
            if leverage_to_set > 0:
                if not set_leverage_ccxt(exchange, symbol, leverage_to_set, market_info, lg):
                    lg.error(f"{NEON_RED}Entry Aborted ({symbol} {signal}): Failed to set leverage.{RESET}")
                    return
            else: lg.info(f"Leverage setting skipped ({symbol}): Config leverage is 0.")

        position_size = calculate_position_size(balance, config["risk_per_trade"], initial_sl_calc, latest_close, market_info, exchange, lg)
        if position_size is None or position_size <= 0:
            lg.error(f"{NEON_RED}Entry Aborted ({symbol} {signal}): Position sizing failed ({position_size}).{RESET}")
            return

        lg.warning(f"{BRIGHT}===> PLACING {signal} MARKET ORDER ({symbol}) | Size: {position_size.normalize()} <==={RESET}")
        trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)

        if trade_order and trade_order.get('id'):
            confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
            lg.info(f"Order {trade_order['id']} placed ({symbol}). Waiting {confirm_delay}s for position confirmation...")
            time.sleep(confirm_delay)

            confirmed_position: Optional[PositionInfo] = None
            for confirm_attempt in range(2):
                 confirmed_position = get_open_position(exchange, symbol, lg)
                 if confirmed_position: break
                 if confirm_attempt == 0: lg.warning(f"Position confirmation ({symbol}) attempt 1 failed, retrying in 3s..."); time.sleep(3)

            if confirmed_position:
                try:
                    entry_price_actual_str = confirmed_position.get('entryPrice')
                    entry_price_actual = _safe_market_decimal(entry_price_actual_str, 'pos.entryPrice', False) or latest_close # Fallback
                    if entry_price_actual == latest_close: lg.warning(f"Using latest close price ({symbol}) for protection calc.")

                    lg.info(f"{NEON_GREEN}Position Confirmed ({symbol})! Entry: ~{entry_price_actual.normalize()}{RESET}")

                    prot_tp_calc, prot_sl_calc = signal_generator.calculate_initial_tp_sl(entry_price_actual, signal, current_atr, market_info, exchange)

                    if prot_sl_calc is None:
                        lg.error(f"{NEON_RED}{BRIGHT}CRITICAL ERROR ({symbol}): Position entered, but failed to recalculate SL! POSITION UNPROTECTED!{RESET}")
                    else:
                        lg.info(f"Setting initial protection (SL={prot_sl_calc.normalize()}, TP={prot_tp_calc.normalize() if prot_tp_calc else 'None'}) for {symbol}...")
                        # Use the internal helper directly for initial SL/TP
                        protection_set_success = _set_position_protection(
                            exchange, symbol, market_info, confirmed_position, lg,
                            stop_loss_price=prot_sl_calc,
                            take_profit_price=prot_tp_calc # Pass TP (or None if disabled/failed)
                        )
                        if protection_set_success:
                            lg.info(f"{NEON_GREEN}{BRIGHT}=== ENTRY & INITIAL PROTECTION SETUP COMPLETE ({symbol} {signal}) ==={RESET}")
                        else:
                            lg.error(f"{NEON_RED}{BRIGHT}=== TRADE PLACED ({symbol} {signal}), BUT FAILED TO SET PROTECTION! MANUAL MONITORING REQUIRED! ==={RESET}")
                except Exception as post_trade_err:
                    lg.error(f"{NEON_RED}Error during post-trade protection setup ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                    lg.warning(f"{NEON_YELLOW}Position confirmed open ({symbol}), but protection setup failed! Manual check recommended!{RESET}")
            else:
                lg.error(f"{NEON_RED}Order {trade_order['id']} placed ({symbol}), but FAILED TO CONFIRM open position! Manual check required!{RESET}")
        else:
            lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). ===")

    # --- Scenario 2: Existing Position -> Consider Exit or Manage ---
    elif open_position:
        pos_side = open_position['side']
        pos_size_decimal = open_position.get('size_decimal', Decimal('0'))
        entry_price = _safe_market_decimal(open_position.get('entryPrice'), 'pos.entryPrice', False)

        # Basic validation of existing position data
        if pos_size_decimal == Decimal('0') or entry_price is None:
             lg.warning(f"Existing position data ({symbol}) seems invalid (Size={pos_size_decimal}, Entry={entry_price}). Skipping management.")
        else:
            lg.info(f"Existing {pos_side.upper()} position found ({symbol}, Size: {pos_size_decimal.normalize()}, Entry: {entry_price.normalize()}). Signal: {signal}")

            # Check for Exit Signal
            exit_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                             (signal == "EXIT_SHORT" and pos_side == 'short')

            if exit_triggered:
                lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal Received ({symbol})! Closing {pos_side} position... ***{RESET}")
                try:
                    size_to_close = abs(pos_size_decimal)
                    lg.info(f"===> Placing {signal} MARKET Order (Reduce Only) ({symbol}) | Size: {size_to_close.normalize()} <===")
                    close_order = place_trade(exchange, symbol, signal, size_to_close, market_info, lg, reduce_only=True)
                    if close_order and close_order.get('id'):
                        lg.info(f"{NEON_GREEN}Position CLOSE order ({close_order['id']}) placed successfully for {symbol}.{RESET}")
                    else:
                        lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual intervention may be required!{RESET}")
                except Exception as close_err:
                    lg.error(f"{NEON_RED}Error trying to close {pos_side} position ({symbol}): {close_err}{RESET}", exc_info=True)
            elif signal == "HOLD" and price_for_checks: # Only manage if HOLD signal and live price available
                # --- Position Management (BE, TSL Activation) ---
                lg.debug(f"Signal is HOLD ({symbol}). Performing position management checks...")
                # Get current state (BE/TSL flags are IN-MEMORY ONLY for this cycle)
                be_activated = open_position.get('be_activated', False)
                tsl_activated_api = bool(open_position.get('trailingStopLoss')) # Check if TSL is set via API
                tsl_activated_bot = open_position.get('tsl_activated', False) # Check bot's memory flag

                # --- Break-Even Logic ---
                enable_be = prot_cfg.get('enable_break_even', True)
                if enable_be and not be_activated and not tsl_activated_api: # Only trigger BE if not already active and TSL not active
                    be_trigger_mult = Decimal(str(prot_cfg.get('break_even_trigger_atr_multiple', 1.0)))
                    be_offset_ticks = int(prot_cfg.get('break_even_offset_ticks', 2))
                    price_tick = market_info['price_precision_step_decimal'] # Already validated earlier
                    be_stop_price: Optional[Decimal] = None

                    profit_target_price = (entry_price + (current_atr * be_trigger_mult)) if pos_side == 'long' else (entry_price - (current_atr * be_trigger_mult))
                    lg.debug(f"BE Check ({symbol}): Current={price_for_checks.normalize()}, Target={profit_target_price.normalize()}")

                    be_triggered = (pos_side == 'long' and price_for_checks >= profit_target_price) or \
                                   (pos_side == 'short' and price_for_checks <= profit_target_price)

                    if be_triggered:
                        lg.info(f"BE Triggered ({symbol} {pos_side})!")
                        be_offset_price = price_tick * be_offset_ticks
                        raw_be_stop_price = (entry_price + be_offset_price) if pos_side == 'long' else (entry_price - be_offset_price)
                        # Format BE stop price
                        be_stop_price_str = _format_price(exchange, symbol, raw_be_stop_price)
                        if be_stop_price_str:
                            be_stop_price = Decimal(be_stop_price_str)
                            # Check if current SL is already better than BE target
                            current_sl_str = open_position.get('stopLossPrice')
                            current_sl = _safe_market_decimal(current_sl_str, 'pos.SL', False) if current_sl_str else None
                            needs_update = True
                            if current_sl:
                                if (pos_side == 'long' and current_sl >= be_stop_price) or \
                                   (pos_side == 'short' and current_sl <= be_stop_price):
                                    needs_update = False
                                    lg.info(f"BE ({symbol}): Current SL ({current_sl.normalize()}) already at/better than calculated BE ({be_stop_price.normalize()}).")

                            if needs_update:
                                lg.warning(f"{BRIGHT}>>> Moving SL to Break-Even ({symbol}) at ~{be_stop_price.normalize()} <<<")
                                protect_success = _set_position_protection(exchange, symbol, market_info, open_position, lg, stop_loss_price=be_stop_price)
                                if protect_success:
                                    open_position['be_activated'] = True # Update in-memory flag
                                else: lg.error(f"{NEON_RED}Failed to set Break-Even SL for {symbol}!{RESET}")
                        else: lg.error(f"Failed to format BE stop price ({symbol}): {raw_be_stop_price}")

                # --- Trailing Stop Loss Activation Logic ---
                enable_tsl = prot_cfg.get('enable_trailing_stop', True)
                # Only activate TSL if enabled, not already activated (by bot or API), and BE is not active
                if enable_tsl and not tsl_activated_bot and not tsl_activated_api and not open_position.get('be_activated'):
                    tsl_activation_perc = Decimal(str(prot_cfg.get('trailing_stop_activation_percentage', 0.003)))
                    tsl_callback_rate = Decimal(str(prot_cfg.get('trailing_stop_callback_rate', 0.005)))
                    price_tick = market_info['price_precision_step_decimal'] # Already validated
                    tsl_distance_calc: Optional[Decimal] = None
                    tsl_activation_price_calc: Optional[Decimal] = None

                    # Calculate activation price
                    activation_target_price = (entry_price * (Decimal('1') + tsl_activation_perc)) if pos_side == 'long' else (entry_price * (Decimal('1') - tsl_activation_perc))
                    lg.debug(f"TSL Activation Check ({symbol}): Current={price_for_checks.normalize()}, Target={activation_target_price.normalize()}")

                    tsl_triggered = (pos_side == 'long' and price_for_checks >= activation_target_price) or \
                                    (pos_side == 'short' and price_for_checks <= activation_target_price)

                    if tsl_triggered:
                         lg.info(f"TSL Activation Triggered ({symbol} {pos_side})!")
                         # Calculate distance based on *activation price* and callback rate
                         raw_distance = abs(activation_target_price) * tsl_callback_rate
                         # Ensure distance is at least one tick and quantize UP
                         tsl_distance_calc = max((raw_distance / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick, price_tick)
                         # Activation price needs formatting too
                         activation_price_fmt = _format_price(exchange, symbol, activation_target_price)
                         if activation_price_fmt:
                             tsl_activation_price_calc = Decimal(activation_price_fmt)
                         else:
                              lg.error(f"Failed to format TSL activation price ({symbol}): {activation_target_price}")
                              tsl_distance_calc = None # Invalidate if activation price bad

                         if tsl_distance_calc and tsl_activation_price_calc:
                             lg.warning(f"{BRIGHT}>>> Activating Trailing Stop Loss ({symbol}) | Dist: {tsl_distance_calc.normalize()}, ActPrice: {tsl_activation_price_calc.normalize()} <<<")
                             # Call protection function with TSL parameters
                             protect_success = _set_position_protection(
                                 exchange, symbol, market_info, open_position, lg,
                                 trailing_stop_distance=tsl_distance_calc,
                                 tsl_activation_price=tsl_activation_price_calc
                             )
                             if protect_success:
                                 open_position['tsl_activated'] = True # Update in-memory flag
                             else:
                                 lg.error(f"{NEON_RED}Failed to activate Trailing Stop Loss for {symbol}!{RESET}")
                         else: lg.error(f"TSL Activation failed ({symbol}): Invalid calculated distance or activation price.")

            # Log end of management checks
            lg.debug(f"Position management checks complete ({symbol}).")

    # --- Scenario 3: No Position, HOLD signal ---
    elif open_position is None and signal == "HOLD":
        lg.info(f"No open position and HOLD signal
