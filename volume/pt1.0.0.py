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
    based on pivot highs and lows derived from candle wicks or bodies (configurable).

Key Features:
-   Connects to Bybit V5 API (supporting Linear/Inverse contracts).
-   Supports Sandbox (testnet) and Live trading environments via `.env` config.
-   Fetches OHLCV data robustly, handling API limits (> 1000 candles) via multiple requests.
-   Calculates strategy indicators using pandas and pandas-ta, leveraging high-precision Decimal types internally where appropriate.
-   Identifies Volumatic Trend direction and changes.
-   Detects Pivot Highs/Lows and creates/manages Order Blocks (Active, Violated, Extended).
-   Generates BUY/SELL/EXIT_LONG/EXIT_SHORT/HOLD signals based on trend alignment and Order Block proximity rules.
-   Calculates position size based on risk percentage, stop-loss distance, and market constraints (precision, limits).
-   Sets leverage for contract markets (optional).
-   Places market orders to enter and exit positions.
-   Advanced Position Management:
    -   Sets initial Stop Loss (SL) and Take Profit (TP) based on ATR multiples.
    -   Implements Trailing Stop Loss (TSL) activation via API based on profit percentage and callback rate.
    -   Implements Break-Even (BE) stop adjustment based on ATR profit targets.
    -   **NOTE:** Break-Even (BE) and Trailing Stop Loss (TSL) activation states are currently managed
        **in-memory per cycle** and are **not persistent** across bot restarts. If the bot restarts,
        it relies on the exchange's reported SL/TP/TSL values.
-   Robust API interaction with configurable retries, detailed error handling (Network, Rate Limit, Auth, Exchange-specific codes), and validation.
-   Secure handling of API credentials via `.env` file.
-   Flexible configuration via `config.json` with validation, default values, and auto-update of missing/invalid fields.
-   Detailed logging with a Neon color scheme for console output and rotating file logs (UTC timestamps).
-   Sensitive data (API keys/secrets) redaction in logs.
-   Graceful shutdown handling (Ctrl+C, SIGTERM).
-   Sequential multi-symbol trading capability.
-   Structured code using classes for Strategy Calculation (`VolumaticOBStrategy`) and Signal Generation (`SignalGenerator`) logic.
"""

# --- Core Libraries ---
import hashlib
import hmac
import json
import logging
import math
import os
import re # Needed for parsing error codes/messages
import signal # For SIGTERM handling
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation # High precision math
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# Use zoneinfo for modern timezone handling (requires tzdata package)
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # Fallback for environments without zoneinfo (e.g., older Python or missing tzdata)
    print(f"Warning: 'zoneinfo' module not found. Falling back to UTC. "
          f"For timezone support, ensure Python 3.9+ and install 'tzdata' (`pip install tzdata`).")
    # Define a simple UTC fallback class mimicking zoneinfo's basic behavior
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
import numpy as np # Requires numpy (pip install numpy) - Used by pandas-ta and some calculations
import pandas as pd # Requires pandas (pip install pandas) - Core data manipulation
import pandas_ta as ta # Requires pandas_ta (pip install pandas_ta) - Technical analysis indicators
import requests # Requires requests (pip install requests) - Used by ccxt for HTTP communication
import ccxt # Requires ccxt (pip install ccxt) - Core library for interacting with crypto exchanges
from colorama import Fore, Style, init as colorama_init # Requires colorama (pip install colorama) - Console text coloring
from dotenv import load_dotenv # Requires python-dotenv (pip install python-dotenv) - Loading environment variables

# --- Initialize Environment and Settings ---
getcontext().prec = 28 # Set Decimal precision globally for high-accuracy financial calculations
colorama_init(autoreset=True) # Initialize Colorama for console colors, resetting style after each print
load_dotenv() # Load environment variables from a `.env` file in the project root

# --- Constants ---
BOT_VERSION = "1.3.1" # Current bot version

# API Credentials (Loaded securely from .env file)
API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Critical error if API keys are missing - Use print as logger might not be initialized yet
    print(f"{Fore.RED}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file. Cannot authenticate with the exchange. Exiting.{Style.RESET_ALL}")
    sys.exit(1)

# Configuration and Logging Files/Directories
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
DEFAULT_TIMEZONE_STR: str = "America/Chicago" # Default timezone if not specified in .env
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    # Attempt to load user-specified timezone using zoneinfo
    # Examples: "America/Chicago", "Europe/London", "Asia/Tokyo", "UTC"
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    print(f"{Fore.RED}Timezone '{TIMEZONE_STR}' not found. Install 'tzdata' (`pip install tzdata`) or check the name. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update string to reflect fallback
except Exception as tz_err:
    print(f"{Fore.RED}Failed to initialize timezone '{TIMEZONE_STR}'. Error: {tz_err}. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

# API Interaction Settings
MAX_API_RETRIES: int = 3        # Max consecutive retries for failed API calls (NetworkError, RateLimit, etc.)
RETRY_DELAY_SECONDS: int = 5    # Base delay (seconds) between API retries (increases exponentially: delay * attempt#)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Wait time (seconds) after placing an entry order before fetching position details to confirm entry
LOOP_DELAY_SECONDS: int = 15    # Default delay (seconds) between trading cycles (can be overridden in config.json)
BYBIT_API_KLINE_LIMIT: int = 1000 # Maximum number of Klines Bybit V5 API returns per request (used for pagination)

# Timeframes Mapping (Config Key to CCXT String)
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750 # Default klines to fetch if not specified or less than strategy needs (balances API load and strategy needs)
MAX_DF_LEN: int = 2000         # Internal limit to prevent excessive memory usage by the Pandas DataFrame (trims oldest data)

# Strategy Defaults (Used if values are missing, invalid, or out of range in config.json)
DEFAULT_VT_LENGTH: int = 40             # Volumatic Trend EMA/SWMA length
DEFAULT_VT_ATR_PERIOD: int = 200        # ATR period for Volumatic Trend bands calculation
DEFAULT_VT_VOL_EMA_LENGTH: int = 950    # Volume Normalization EMA length (Adjusted: 1000 often > API limit, use slightly less)
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0  # ATR multiplier for Volumatic Trend bands
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Currently unused step ATR multiplier (can be removed if definitely not needed)
DEFAULT_OB_SOURCE: str = "Wicks"        # Order Block source: "Wicks" (high/low) or "Body" (max/min of open/close)
DEFAULT_PH_LEFT: int = 10               # Pivot High lookback periods (left of pivot candle)
DEFAULT_PH_RIGHT: int = 10              # Pivot High lookback periods (right of pivot candle)
DEFAULT_PL_LEFT: int = 10               # Pivot Low lookback periods (left of pivot candle)
DEFAULT_PL_RIGHT: int = 10              # Pivot Low lookback periods (right of pivot candle)
DEFAULT_OB_EXTEND: bool = True          # Extend Order Block visuals (box right edge) to the latest candle
DEFAULT_OB_MAX_BOXES: int = 50          # Max number of *active* Order Blocks to track per type (Bull/Bear) to manage memory/performance

# Dynamically loaded from config: QUOTE_CURRENCY (e.g., "USDT")
QUOTE_CURRENCY: str = "USDT" # Placeholder, will be updated by load_config() based on config file value

# Logging Colors (Neon Theme for Console Output)
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN # Used for Debug
RESET: str = Style.RESET_ALL
BRIGHT: str = Style.BRIGHT
DIM: str = Style.DIM

# Ensure log directory exists before setting up loggers
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e:
     print(f"{NEON_RED}{BRIGHT}FATAL: Could not create log directory '{LOG_DIRECTORY}': {e}. Ensure permissions are correct. Exiting.{RESET}")
     sys.exit(1)

# Global flag for shutdown signal (used by signal handler and main loop)
_shutdown_requested = False

# --- Type Definitions for Structured Data ---
class OrderBlock(TypedDict):
    """Represents a bullish or bearish Order Block identified on the chart."""
    id: str                 # Unique identifier (e.g., "B_231026143000" for Bearish, "L_..." for Bullish)
    type: str               # 'bull' (from Pivot Low) or 'bear' (from Pivot High)
    timestamp: pd.Timestamp # Timestamp of the candle that formed the OB (the pivot candle)
    top: Decimal            # Top price level of the OB (max of source for pivot candle)
    bottom: Decimal         # Bottom price level of the OB (min of source for pivot candle)
    active: bool            # True if the OB is currently considered valid (price hasn't closed beyond it)
    violated: bool          # True if the price has closed beyond the OB boundary
    violation_ts: Optional[pd.Timestamp] # Timestamp when violation occurred (close price confirmed violation)
    extended_to_ts: Optional[pd.Timestamp] # Timestamp the OB box visually extends to (if ob_extend=True in config)

class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis process for a single symbol/timeframe."""
    dataframe: pd.DataFrame             # The Pandas DataFrame with all calculated indicators (Decimal values where appropriate)
    last_close: Decimal                 # The closing price of the most recent candle (use Decimal('NaN') if invalid)
    current_trend_up: Optional[bool]    # True if Volumatic Trend is up, False if down, None if undetermined (e.g., insufficient data)
    trend_just_changed: bool            # True if the trend direction flipped on the very last candle
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs, sorted newest first
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs, sorted newest first
    vol_norm_int: Optional[int]         # Normalized volume indicator (0-200 int) for the last candle, relative to its EMA
    atr: Optional[Decimal]              # ATR (Average True Range) value for the last candle (must be positive Decimal if valid)
    upper_band: Optional[Decimal]       # Volumatic Trend upper band value for the last candle
    lower_band: Optional[Decimal]       # Volumatic Trend lower band value for the last candle

class MarketInfo(TypedDict):
    """Standardized market information dictionary derived from ccxt.market object."""
    # Standard CCXT fields (may vary slightly by exchange, presence not guaranteed for all)
    id: str                     # Exchange-specific market ID (e.g., 'BTCUSDT')
    symbol: str                 # Standardized symbol (e.g., 'BTC/USDT')
    base: str                   # Base currency code (e.g., 'BTC')
    quote: str                  # Quote currency code (e.g., 'USDT')
    settle: Optional[str]       # Settlement currency (usually quote for linear, base for inverse)
    baseId: str                 # Exchange-specific base ID
    quoteId: str                # Exchange-specific quote ID
    settleId: Optional[str]     # Exchange-specific settle ID
    type: str                   # Market type: 'spot', 'swap', 'future', etc.
    spot: bool                  # True if spot market
    margin: bool                # True if margin trading allowed (distinct from derivatives)
    swap: bool                  # True if perpetual swap contract
    future: bool                # True if futures contract (non-perpetual)
    option: bool                # True if options contract
    active: bool                # Whether the market is currently active/tradable on the exchange
    contract: bool              # True if it's a derivative contract (swap, future, option)
    linear: Optional[bool]      # True if linear contract (settled in quote currency)
    inverse: Optional[bool]     # True if inverse contract (settled in base currency)
    quanto: Optional[bool]      # True if quanto contract (settled in a third currency)
    taker: float                # Taker fee rate (as a fraction, e.g., 0.0006)
    maker: float                # Maker fee rate (as a fraction, e.g., 0.0001)
    contractSize: Optional[Any] # Size of one contract (often float or int, convert to Decimal for calculations)
    expiry: Optional[int]       # Unix timestamp of expiry (milliseconds), for futures/options
    expiryDatetime: Optional[str]# ISO8601 datetime string of expiry
    strike: Optional[float]     # Option strike price
    optionType: Optional[str]   # Option type: 'call' or 'put'
    precision: Dict[str, Any]   # Precision rules: {'amount': float/str, 'price': float/str, 'cost': float/str, 'base': float, 'quote': float} - Source for Decimal steps
    limits: Dict[str, Any]      # Trading limits: {'leverage': {'min': float, 'max': float}, 'amount': {'min': float/str, 'max': float/str}, 'price': {'min': float, 'max': float}, 'cost': {'min': float/str, 'max': float/str}} - Source for Decimal limits
    info: Dict[str, Any]        # Raw, exchange-specific market info dictionary provided by CCXT
    # Custom added fields for convenience and robustness
    is_contract: bool           # Reliable check for derivatives (True if swap, future, or contract=True)
    is_linear: bool             # True only if linear contract AND is_contract=True
    is_inverse: bool            # True only if inverse contract AND is_contract=True
    contract_type_str: str      # User-friendly contract type: "Linear", "Inverse", "Spot", or "Unknown"
    min_amount_decimal: Optional[Decimal] # Parsed minimum order size (in base units for spot, or contracts for derivatives). Must be non-negative.
    max_amount_decimal: Optional[Decimal] # Parsed maximum order size. Must be positive if set.
    min_cost_decimal: Optional[Decimal]   # Parsed minimum order cost (in quote currency). Must be non-negative.
    max_cost_decimal: Optional[Decimal]   # Parsed maximum order cost. Must be positive if set.
    amount_precision_step_decimal: Optional[Decimal] # Parsed step size for amount (e.g., 0.001 BTC). Must be positive. CRITICAL for order placement.
    price_precision_step_decimal: Optional[Decimal]  # Parsed step size for price (e.g., 0.01 USDT). Must be positive. CRITICAL for order placement and SL/TP calculation.
    contract_size_decimal: Decimal  # Parsed contract size as Decimal (e.g., 1 for BTC/USDT perp, 100 for BTC/USD inverse). Must be positive, defaults to 1 if not applicable/found.

class PositionInfo(TypedDict):
    """Standardized position information dictionary derived from ccxt.position object."""
    # Standard CCXT fields (availability and naming can vary by exchange)
    id: Optional[str]           # Position ID (often None or same as symbol)
    symbol: str                 # Standardized symbol (e.g., 'BTC/USDT')
    timestamp: Optional[int]    # Creation timestamp (milliseconds)
    datetime: Optional[str]     # ISO8601 creation datetime string
    contracts: Optional[float]  # DEPRECATED/inconsistent, use size_decimal instead. Represents size in contracts.
    contractSize: Optional[Any] # Size of one contract for this position (convert to Decimal if needed, usually matches MarketInfo)
    side: Optional[str]         # Position side: 'long' or 'short' (parsed/validated)
    notional: Optional[Any]     # Position value in quote currency (e.g., USDT value for BTC/USDT). Convert to Decimal.
    leverage: Optional[Any]     # Leverage used for this position. Convert to Decimal.
    unrealizedPnl: Optional[Any]# Unrealized Profit/Loss. Convert to Decimal.
    realizedPnl: Optional[Any]  # Realized Profit/Loss. Convert to Decimal.
    collateral: Optional[Any]   # Collateral allocated/used for this position (depends on margin mode). Convert to Decimal.
    entryPrice: Optional[Any]   # Average entry price of the position. Convert to Decimal. CRITICAL field.
    markPrice: Optional[Any]    # Current mark price used for PnL calculation and liquidation checks. Convert to Decimal.
    liquidationPrice: Optional[Any] # Estimated liquidation price. Convert to Decimal.
    marginMode: Optional[str]   # Margin mode: 'cross' or 'isolated'
    hedged: Optional[bool]      # Whether the position is part of a hedge (relevant in Hedge Mode)
    maintenanceMargin: Optional[Any] # Maintenance margin required to keep the position open. Convert to Decimal.
    maintenanceMarginPercentage: Optional[float] # Maintenance margin rate (as fraction, e.g., 0.005)
    initialMargin: Optional[Any]# Initial margin used to open the position. Convert to Decimal.
    initialMarginPercentage: Optional[float] # Initial margin rate (based on leverage)
    marginRatio: Optional[float]# Margin ratio (e.g., maintenance margin / collateral)
    lastUpdateTimestamp: Optional[int] # Timestamp of last position update (milliseconds)
    info: Dict[str, Any]        # Raw, exchange-specific position info dictionary provided by CCXT. CRUCIAL for Bybit V5 details like SL/TP/TSL strings, positionIdx, etc.
    # Custom added/parsed fields for easier use
    size_decimal: Decimal       # Parsed position size as Decimal. Positive for long, negative for short. Non-zero indicates an active position.
    stopLossPrice: Optional[str]# Parsed SL price from `info` (string format from Bybit, e.g., "29000.5"). Non-zero string if set.
    takeProfitPrice: Optional[str]# Parsed TP price from `info` (string format from Bybit). Non-zero string if set.
    trailingStopLoss: Optional[str]# Parsed TSL *distance* from `info` (string format from Bybit). Non-zero string if active.
    tslActivationPrice: Optional[str]# Parsed TSL *activation price* from `info` (string format from Bybit). Non-zero string if TSL active.
    # Custom flags for bot state tracking (**IN-MEMORY ONLY**, NOT PERSISTENT across restarts)
    be_activated: bool          # True if Break-Even has been set for this position instance by the bot during the current run.
    tsl_activated: bool         # True if Trailing Stop Loss has been set for this position instance by the bot during the current run.


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive API keys/secrets
    from log messages to prevent accidental exposure in log files or console output.
    """
    _api_key_placeholder = "***BYBIT_API_KEY***" # More specific placeholder
    _api_secret_placeholder = "***BYBIT_API_SECRET***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing API keys/secrets with placeholders."""
        # Ensure API keys exist and are strings before attempting replacement
        # Use local copies to avoid potential race conditions if keys were reloaded (unlikely here)
        key = API_KEY
        secret = API_SECRET
        # Format the message using the parent class's logic first
        msg = super().format(record)
        try:
            # Perform redaction using the formatted message string
            if key and isinstance(key, str) and key in msg:
                msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg:
                msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception as e:
            # Avoid crashing the logger if redaction fails unexpectedly
            # Log this internal error cautiously (might cause infinite loop if logger itself is broken)
            print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
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

    # Avoid adding multiple handlers if the logger was already configured (e.g., in interactive sessions)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Set the base level to capture all messages; handlers filter output

    # --- File Handler (DEBUG level, Rotating, Redaction, UTC Timestamps) ---
    try:
        # Rotate log file when it reaches 10MB, keep 5 backup files
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Use SensitiveFormatter for detailed file log, redacting API keys/secrets
        # Include milliseconds in file log timestamps for precise timing
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S' # Standard ISO-like date format for files
        )
        # Explicitly set UTC time for file logs for consistency across environments
        file_formatter.converter = time.gmtime # type: ignore
        fh.setFormatter(file_formatter)
        fh.setLevel(logging.DEBUG) # Log everything from DEBUG upwards to the file
        logger.addHandler(fh)
    except Exception as e:
        # Use print for errors during logger setup itself, as logger might not be functional
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler (Configurable Level, Neon Colors, Local Timezone Timestamps) ---
    try:
        sh = logging.StreamHandler(sys.stdout) # Explicitly use stdout for console output
        # Define color mapping for different log levels
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,      # Dim Cyan for Debug
            logging.INFO: NEON_BLUE,             # Bright Cyan (or Blue) for Info
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
                level_color = self._level_colors.get(record.levelno, NEON_BLUE) # Default to Info color if level unknown
                # Format: Time(Local) - Level - [LoggerName] - Message
                log_fmt = (
                    f"{NEON_BLUE}%(asctime)s{RESET} - " # Timestamp color
                    f"{level_color}%(levelname)-8s{RESET} - " # Level color
                    f"{NEON_PURPLE}[%(name)s]{RESET} - " # Logger name color
                    f"%(message)s" # Message (will be colored by context)
                )
                # Create a formatter instance with the defined format and date style
                # Use Time only for console clarity, Date is less relevant for live monitoring
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
                # Ensure timestamps reflect the configured TIMEZONE by setting the converter
                formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore
                # Apply sensitive data redaction before returning the final message
                # We inherit from SensitiveFormatter, so super().format handles redaction
                # Must call super() explicitly for nested class inheritance
                return super(NeonConsoleFormatter, self).format(record)

        sh.setFormatter(NeonConsoleFormatter())
        # Get desired console log level from environment variable (e.g., DEBUG, INFO, WARNING), default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        try:
             log_level = getattr(logging, log_level_str)
        except AttributeError:
             print(f"{NEON_YELLOW}Warning: Invalid CONSOLE_LOG_LEVEL '{log_level_str}'. Defaulting to INFO.{RESET}")
             log_level = logging.INFO # Fallback to INFO if invalid level provided
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    # Prevent log messages from bubbling up to the root logger, avoiding duplicate outputs
    # if the root logger is configured elsewhere.
    logger.propagate = False
    return logger

# Initialize the 'init' logger early for messages during startup and configuration loading
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR}")

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively checks if all keys from `default_config` exist in `config`.
    If a key is missing, it's added to `config` with the default value from `default_config`.
    Logs any keys that were added using the `init_logger`. This helps maintain config structure across updates.

    Args:
        config (Dict[str, Any]): The configuration dictionary loaded from the file.
        default_config (Dict[str, Any]): The dictionary containing the expected default structure and values.
        parent_key (str): Used internally for tracking nested key paths for logging (e.g., "strategy_params.vt_length").

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing:
            - The potentially updated configuration dictionary (`config` potentially modified).
            - A boolean indicating whether any changes were made (True if keys were added).
    """
    updated_config = config.copy() # Work on a copy to avoid modifying the original dict directly in loop
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing parameter '{full_key_path}' with default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # If both default and loaded values are dictionaries, recurse into the nested dictionary
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                # If the nested dictionary was changed, update the parent dictionary and mark as changed
                updated_config[key] = nested_config
                changed = True
        # Optional: Could add a type mismatch check here, but the main validation function handles it more robustly.
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads, validates, and potentially updates the bot's configuration from a JSON file.

    Steps:
    1. Checks if the config file exists. If not, creates a default `config.json` file.
    2. Loads the JSON data from the file. Handles decoding errors by attempting to recreate the default file.
    3. Ensures all expected keys (from the default structure) exist in the loaded config, adding missing ones with default values.
    4. Performs detailed type and range validation on critical numeric and string parameters using `validate_numeric`.
       - Uses default values and logs warnings/corrections if validation fails or values are out of bounds.
       - Leverages Decimal for robust numeric comparisons where appropriate.
    5. Validates the `trading_pairs` list structure and content.
    6. If any keys were added or values corrected during validation, saves the updated configuration back to the file.
    7. Updates the global `QUOTE_CURRENCY` variable based on the validated configuration.
    8. Returns the validated (and potentially updated) configuration dictionary.

    Args:
        filepath (str): The path to the configuration JSON file (e.g., "config.json").

    Returns:
        Dict[str, Any]: The loaded and validated configuration dictionary. Returns the internal default configuration
                        if the file cannot be read, created, or parsed, or if validation encounters unexpected errors.
    """
    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")
    # Define the default configuration structure and values (used for validation and creation)
    default_config = {
        # General Settings
        "trading_pairs": ["BTC/USDT"],  # List of symbols to trade (e.g., ["BTC/USDT", "ETH/USDT"])
        "interval": "5",                # Default timeframe (must be in VALID_INTERVALS, e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS, # Base delay (seconds) for API retries
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Default klines to fetch per cycle (balances API load and strategy needs)
        "orderbook_limit": 25,          # (Currently Unused) Limit for order book fetching if feature is added
        "enable_trading": False,        # Master switch: Set to true to enable placing actual trades
        "use_sandbox": True,            # Use Bybit's testnet environment (True) or live environment (False)
        "risk_per_trade": 0.01,         # Fraction of available balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                 # Default leverage for contract trading (0 to disable explicit setting)
        "max_concurrent_positions": 1,  # (Currently Unused) Max open positions allowed simultaneously across all symbols
        "quote_currency": "USDT",       # The currency to calculate balance and risk against (e.g., USDT, USDC)
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay (seconds) between trading cycles for each symbol
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait (seconds) after order before confirming position

        # Strategy Parameters (Volumatic Trend + OB)
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,             # Volumatic Trend EMA/SWMA length
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,     # ATR period for Volumatic Trend bands calculation
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, # Volume Normalization EMA length
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER), # ATR multiplier for Volumatic Trend bands (store as float in JSON)
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER), # Unused, store as float
            "ob_source": DEFAULT_OB_SOURCE,             # Order Block source: "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT, # Pivot High lookback periods
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT, # Pivot Low lookback periods
            "ob_extend": DEFAULT_OB_EXTEND,             # Extend Order Block visuals to latest candle
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,       # Max active OBs to track per type
            "ob_entry_proximity_factor": 1.005, # Price must be <= OB top * factor (long) or >= OB bottom / factor (short) for entry signal
            "ob_exit_proximity_factor": 1.001   # Exit signal if price >= Bear OB top / factor or <= Bull OB bottom * factor
        },
        # Position Protection Parameters
        "protection": {
             "enable_trailing_stop": True,      # Master switch for Trailing Stop Loss feature
             "trailing_stop_callback_rate": 0.005, # TSL distance as % of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Activate TSL when price moves this % in profit from entry (e.g., 0.003 = 0.3%)
             "enable_break_even": True,         # Master switch for Break-Even feature
             "break_even_trigger_atr_multiple": 1.0, # Profit needed (in ATR multiples from entry) to trigger BE
             "break_even_offset_ticks": 2,       # Move SL this many price ticks beyond entry price for BE (uses market's price precision)
             "initial_stop_loss_atr_multiple": 1.8, # Initial SL distance calculated as ATR * this multiple
             "initial_take_profit_atr_multiple": 0.7 # Initial TP distance calculated as ATR * this multiple (Set to 0 to disable initial TP)
        }
    }
    config_needs_saving: bool = False # Flag to track if the config file should be overwritten
    loaded_config: Dict[str, Any] = {}

    # --- File Existence Check & Default Creation ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating a default config file.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Write the default configuration with nice formatting
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default configuration file: {filepath}{RESET}")
            # Update global QUOTE_CURRENCY immediately after creating default
            global QUOTE_CURRENCY
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Return the defaults immediately, no further processing needed
        except IOError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error creating default config file '{filepath}': {e}. Cannot proceed without configuration.{RESET}")
            init_logger.warning("Using internal default configuration values. Bot functionality may be limited.")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Use internal defaults if file creation fails

    # --- File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
             # Ensure the loaded JSON is actually a dictionary (object)
             raise TypeError("Configuration file does not contain a valid JSON object (dictionary).")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Attempting to recreate with defaults.{RESET}")
        try: # Attempt to overwrite the corrupted file with defaults
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recreated default config file due to corruption: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Return defaults after recreating
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}FATAL: Error recreating default config file after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Use internal defaults if recreation fails
    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.warning("Using internal default configuration values. Bot functionality may be limited.")
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config # Fallback to internal defaults on unexpected errors

    # --- Validation and Merging ---
    try:
        # Ensure all keys from default_config exist in loaded_config, add missing keys with defaults
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True # Mark for saving if keys were added

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            """
            Validates a numeric configuration value at a given `key_path` (e.g., "protection.leverage").

            Checks type (int/float), range [min_val, max_val] or (min_val, max_val] if strict.
            Uses the default value from `default_config` and logs a warning/info message if correction is needed.
            Updates the `cfg` dictionary in place if a correction is made.
            Uses Decimal for robust comparisons, especially for financial values.

            Args:
                cfg (Dict): The configuration dictionary being validated (modified in place).
                key_path (str): The dot-separated path to the key (e.g., "protection.leverage").
                min_val (Union[int, float, Decimal]): Minimum allowed value (inclusive unless is_strict_min).
                max_val (Union[int, float, Decimal]): Maximum allowed value (inclusive).
                is_strict_min (bool): If True, value must be strictly greater than min_val.
                is_int (bool): If True, value must be an integer.
                allow_zero (bool): If True, zero is allowed even if min_val > 0 or max_val < 0.

            Returns:
                bool: True if a correction was made (value changed or type corrected), False otherwise.
            """
            nonlocal config_needs_saving # Allow modification of the outer scope variable
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                # Traverse nested dictionaries to reach the target key and its default
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key) # Get value from loaded config
                default_val = default_level.get(leaf_key) # Get corresponding default value
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid key path '{key_path}'. Cannot validate.")
                return False # Path itself is wrong, cannot proceed

            if original_val is None:
                # This case should be rare due to _ensure_config_keys, but handle defensively
                init_logger.warning(f"Config validation: Value missing at '{key_path}'. Using default: {repr(default_val)}")
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val # Start with the original value, assume it's okay

            try:
                # Convert to Decimal for robust comparison, handle potential strings in JSON
                num_val = Decimal(str(original_val))
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))

                # Check range constraints
                min_check_passed = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check_passed = min_check_passed and num_val <= max_dec
                # Check if zero is allowed and value is zero, bypassing range check if so
                zero_is_valid = allow_zero and num_val.is_zero()

                if not range_check_passed and not zero_is_valid:
                    # Value is outside the allowed range (and not a permitted zero)
                    raise ValueError("Value outside allowed range.")

                # Check type and convert if necessary
                target_type = int if is_int else float
                # Attempt conversion to the target numeric type (int or float)
                converted_val = target_type(num_val)

                # Check if type or value changed significantly after conversion
                # This ensures int remains int, float remains float (within tolerance), and handles string inputs
                needs_correction = False
                if isinstance(original_val, bool): # Don't try to convert bools here; should be caught by type check
                     raise TypeError("Boolean value found where numeric value expected.")
                elif is_int and not isinstance(original_val, int):
                    # If int expected but input wasn't int (e.g., float 10.0 or string "10"), correct it
                    needs_correction = True
                elif not is_int and not isinstance(original_val, float):
                     # If float expected, allow int input but convert it to float
                    if isinstance(original_val, int):
                        converted_val = float(original_val) # Explicitly convert int to float
                        needs_correction = True
                    else: # Input is neither float nor int (e.g., string "0.01"), correct type
                        needs_correction = True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9:
                    # Check if float value changed significantly after Decimal conversion and back (handles precision issues)
                    needs_correction = True
                elif isinstance(original_val, int) and original_val != converted_val:
                     # Should not happen if is_int=True, but check defensively
                     needs_correction = True

                if needs_correction:
                    init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
                    final_val = converted_val # Use the correctly typed/converted value
                    corrected = True

            except (ValueError, InvalidOperation, TypeError) as e:
                # Handle cases where value is non-numeric, out of range, or conversion fails
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                if allow_zero: range_str += " or 0"
                init_logger.warning(f"{NEON_YELLOW}Config Validation: Invalid value '{repr(original_val)}' for '{key_path}'. Using default: {repr(default_val)}. Error: {e}. Expected: {'integer' if is_int else 'float'}, Range: {range_str}{RESET}")
                final_val = default_val # Use the default value as a fallback
                corrected = True

            # If a correction occurred, update the config dictionary and mark for saving
            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        init_logger.debug("# Validating configuration parameters...")
        # --- Apply Validations to Specific Config Keys ---
        # General Settings
        if not isinstance(updated_config.get("trading_pairs"), list) or \
           not all(isinstance(s, str) and s and '/' in s for s in updated_config.get("trading_pairs", [])): # Basic check for symbol format
            init_logger.warning(f"{NEON_YELLOW}Invalid 'trading_pairs'. Must be a list of non-empty strings containing '/'. Using default {default_config['trading_pairs']}.{RESET}")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"{NEON_YELLOW}Invalid 'interval' '{updated_config.get('interval')}'. Valid options: {VALID_INTERVALS}. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True) # Retry delay 1-60 seconds
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True) # Ensure minimum useful fetch limit, max internal limit
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('0.5'), is_strict_min=True) # Risk must be > 0% and <= 50% (sanity cap)
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Leverage 0 means no setting attempt, reasonable max
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True) # Loop delay 1 second to 1 hour
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True) # Confirm delay 1-60 seconds
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
             init_logger.warning(f"Invalid 'quote_currency'. Must be non-empty string. Using default '{default_config['quote_currency']}'.")
             updated_config["quote_currency"] = default_config["quote_currency"]
             config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool):
             init_logger.warning(f"Invalid 'enable_trading'. Must be true or false. Using default '{default_config['enable_trading']}'.")
             updated_config["enable_trading"] = default_config["enable_trading"]
             config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
             init_logger.warning(f"Invalid 'use_sandbox'. Must be true or false. Using default '{default_config['use_sandbox']}'.")
             updated_config["use_sandbox"] = default_config["use_sandbox"]
             config_needs_saving = True

        # Strategy Parameters
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True) # VT length reasonable range
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True) # Allow long ATR period up to max DF length
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True) # Allow long Vol EMA up to max DF length
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0) # ATR multiplier reasonable range
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True) # Pivot lookback reasonable range
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True) # Max OBs reasonable range
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # Entry proximity factor (e.g., 1.005 = 0.5% proximity)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # Exit proximity factor (e.g., 1.001 = 0.1% proximity)
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
             config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
             init_logger.warning(f"Invalid strategy_params.ob_extend. Must be true or false. Using default '{DEFAULT_OB_EXTEND}'.")
             updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND
             config_needs_saving = True

        # Protection Parameters
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool):
             init_logger.warning(f"Invalid protection.enable_trailing_stop. Must be true or false. Using default '{default_config['protection']['enable_trailing_stop']}'.")
             updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]
             config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
             init_logger.warning(f"Invalid protection.enable_break_even. Must be true or false. Using default '{default_config['protection']['enable_break_even']}'.")
             updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]
             config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.1'), is_strict_min=True) # TSL Callback Rate (0.01% to 10%), must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.1'), allow_zero=True) # TSL Activation % (0% to 10%), 0 means activate immediately if profitable
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0')) # BE Trigger ATR multiple range
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # BE Offset ticks (0 means exactly at entry)
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('20.0'), is_strict_min=True) # Initial SL ATR multiple range, must be > 0
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('20.0'), allow_zero=True) # Initial TP ATR multiple range, 0 disables initial TP

        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             try:
                 # json.dumps handles basic types (int, float, str, bool, list, dict) automatically
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Configuration file '{filepath}' updated with missing/corrected values.{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with corrected configuration in memory, but file update failed.")

        # Update the global QUOTE_CURRENCY from the validated config
        global QUOTE_CURRENCY
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(f"{Fore.CYAN}# Configuration loading and validation complete.{Style.RESET_ALL}")

        return updated_config # Return the validated and potentially corrected config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected error during configuration processing: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Ensure quote currency is set even on failure
        return default_config # Fallback to internal defaults on unexpected validation error

# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)
# QUOTE_CURRENCY is updated inside load_config()

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and validates the CCXT Bybit exchange object.

    Steps:
    1. Sets API keys, enables rate limiting, sets default contract type (linear), and configures timeouts.
    2. Configures sandbox mode based on `use_sandbox` setting in `config.json`.
    3. Loads exchange markets with retries, ensuring markets are actually populated.
    4. Performs an initial balance check for the configured `QUOTE_CURRENCY`.
       - If trading is enabled (`enable_trading`=True), a failed balance check is treated as a fatal error.
       - If trading is disabled, logs a warning but allows proceeding without a confirmed balance.

    Args:
        logger (logging.Logger): The logger instance to use for status messages.

    Returns:
        Optional[ccxt.Exchange]: The initialized `ccxt.Exchange` object if successful, otherwise None.
    """
    lg = logger # Alias for convenience
    lg.info(f"{Fore.CYAN}# Initializing connection to Bybit exchange...{Style.RESET_ALL}")
    try:
        # Common CCXT exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in request rate limiter
            'options': {
                'defaultType': 'linear',         # Assume linear contracts by default (e.g., USDT margined)
                'adjustForTimeDifference': True, # Auto-adjust client time to match server time
                # Timeouts for various API operations (in milliseconds)
                'fetchTickerTimeout': 15000,    # Timeout for getting current price ticker
                'fetchBalanceTimeout': 20000,   # Timeout for fetching account balance
                'createOrderTimeout': 30000,    # Timeout for placing an order
                'cancelOrderTimeout': 20000,    # Timeout for cancelling an order
                'fetchPositionsTimeout': 20000, # Timeout for fetching open positions
                'fetchOHLCVTimeout': 60000,     # Longer timeout for potentially large kline history fetches
            }
        }
        # Instantiate the Bybit exchange object using CCXT
        exchange = ccxt.bybit(exchange_options)

        # Configure Sandbox Mode based on config
        is_sandbox = CONFIG.get('use_sandbox', True) # Default to sandbox if missing
        exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< OPERATING IN SANDBOX MODE (Testnet Environment) >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< OPERATING IN LIVE MODE - REAL ASSETS AT RISK >>> !!!{RESET}")

        # Load Markets with Retries - Crucial for getting symbol info, precision, limits
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
                # Force reload on retries to ensure fresh market data, especially if first attempt failed
                exchange.load_markets(reload=(attempt > 0))
                # Check if markets were actually loaded
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market data loaded successfully ({len(exchange.markets)} symbols found).{RESET}")
                    markets_loaded = True
                    break # Exit retry loop on success
                else:
                    # This case indicates an issue even if no exception was raised (e.g., empty response)
                    last_market_error = ValueError("Exchange returned empty market data.")
                    lg.warning(f"Market data load returned empty results (Attempt {attempt + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                # Handle network-related errors during market loading
                last_market_error = e
                lg.warning(f"Network Error loading markets (Attempt {attempt + 1}): {e}.")
                if attempt >= MAX_API_RETRIES:
                    lg.critical(f"{NEON_RED}Max retries exceeded loading markets. Last error: {last_market_error}. Initialization failed.{RESET}")
                    return None
            except ccxt.AuthenticationError as e:
                 # Handle authentication errors (invalid API keys)
                 last_market_error = e
                 lg.critical(f"{NEON_RED}Authentication failed: {e}. Check API Key/Secret. Initialization failed.{RESET}")
                 return None
            except Exception as e:
                # Handle any other unexpected errors during market loading
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Initialization failed.{RESET}", exc_info=True)
                return None

            # Apply increasing delay before retrying market load
            if not markets_loaded and attempt < MAX_API_RETRIES:
                 delay = RETRY_DELAY_SECONDS * (attempt + 1) # Increase delay per attempt
                 lg.warning(f"Retrying market load in {delay}s...")
                 time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load market data after all attempts. Last error: {last_market_error}. Initialization failed.{RESET}")
            return None

        lg.info(f"Exchange connection established: {exchange.id} | Sandbox Mode: {is_sandbox}")

        # Initial Balance Check - Verify API keys work and get starting balance
        lg.info(f"Performing initial balance check for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try:
            # Use the dedicated balance fetching function with retries
            initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err:
            # Handle authentication errors specifically here as they are critical during balance check
            lg.critical(f"{NEON_RED}Authentication Failed during initial balance check: {auth_err}. Initialization failed.{RESET}")
            return None
        except Exception as balance_err:
             # Catch other potential errors during the initial balance check (Network, etc.)
             lg.warning(f"{NEON_YELLOW}Initial balance check encountered an error: {balance_err}.{RESET}", exc_info=False)
             # Let the logic below decide based on trading enabled status

        # Evaluate balance check result based on trading mode
        if initial_balance is not None:
            # Balance fetched successfully
            lg.info(f"{NEON_GREEN}Initial available balance: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange initialization complete and validated.{Style.RESET_ALL}")
            return exchange # Success! Return the initialized exchange object
        else:
            # Balance fetch failed (fetch_balance logs the failure reason)
            lg.error(f"{NEON_RED}Initial balance check FAILED for {QUOTE_CURRENCY}.{RESET}")
            # If trading is enabled, this is a critical failure
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading is enabled, but initial balance check failed. Cannot proceed safely. Initialization failed.{RESET}")
                return None
            else:
                # If trading is disabled, allow proceeding but warn the user
                lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without confirmed balance, but risk calculations may be inaccurate.{RESET}")
                lg.info(f"{Fore.CYAN}# Exchange initialization complete (balance check failed, trading disabled).{Style.RESET_ALL}")
                return exchange # Allow proceeding in non-trading mode

    except Exception as e:
        # Catch-all for errors during the CCXT instantiation process itself
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange object: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers ---
def _safe_market_decimal(value: Optional[Any], field_name: str, allow_zero: bool = True, allow_negative: bool = False) -> Optional[Decimal]:
    """
    Internal helper: Safely converts a market info value (potentially str, float, int, None) to Decimal.

    Handles None, empty strings, non-numeric strings, and validates against zero/negative constraints.
    Logs debug messages for conversion issues (can be verbose).

    Args:
        value (Optional[Any]): The value to convert (e.g., from market['limits']['amount']['min']).
        field_name (str): The name of the field being converted (for logging purposes).
        allow_zero (bool): Whether a value of zero is considered valid.
        allow_negative (bool): Whether negative values are considered valid (rarely needed for market info).

    Returns:
        Optional[Decimal]: The converted Decimal value, or None if conversion fails, value is None,
                           or the value is invalid according to `allow_zero`/`allow_negative` constraints.
    """
    if value is None:
        return None
    try:
        # Convert to string first to handle various input types robustly
        s_val = str(value).strip()
        if not s_val: # Handle empty strings
            return None
        d_val = Decimal(s_val) # Attempt conversion to Decimal

        # Check constraints
        if not allow_zero and d_val.is_zero():
            # init_logger.debug(f"Decimal conversion skipped for '{field_name}': Value '{value}' is zero, but zero not allowed.")
            return None
        if not allow_negative and d_val < Decimal('0'):
            # init_logger.debug(f"Decimal conversion skipped for '{field_name}': Value '{value}' is negative, but negative not allowed.")
            return None

        return d_val # Return the valid Decimal
    except (InvalidOperation, TypeError, ValueError) as e:
        # Log conversion failures at debug level as they can be common for missing/invalid fields
        # init_logger.debug(f"Could not convert market info field '{field_name}' value '{repr(value)}' to Decimal: {e}")
        return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price value to the exchange's required precision string using `exchange.price_to_precision`.

    Handles Decimal, float, or string input. Ensures the input price is positive before formatting.
    Uses CCXT's built-in method to handle exchange-specific rounding/truncation rules correctly.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The market symbol (e.g., 'BTC/USDT').
        price (Union[Decimal, float, str]): The price value to format.

    Returns:
        Optional[str]: The formatted price string suitable for API calls (e.g., "29000.50"),
                       or None if formatting fails or the input price is invalid (non-positive).
    """
    try:
        # Convert input to Decimal for validation
        price_decimal = Decimal(str(price))
        # Price must be strictly positive for formatting (API usually rejects zero/negative prices)
        if price_decimal <= Decimal('0'):
            # init_logger.debug(f"Price formatting skipped for {symbol}: Input price {price_decimal} is not positive.") # Potentially verbose
            return None

        # Use CCXT's helper function for exchange-specific formatting
        # It requires a float input, so convert the validated Decimal to float
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))

        # Final sanity check: ensure formatted price string can be converted back to a positive Decimal
        # This catches cases where formatting might round down to zero unexpectedly.
        if Decimal(formatted_str) > Decimal('0'):
            return formatted_str
        else:
            # init_logger.debug(f"Formatted price '{formatted_str}' for {symbol} resulted in zero or negative value. Ignoring.")
            return None
    except (InvalidOperation, ValueError, TypeError, KeyError, AttributeError) as e:
        # Catch potential errors during conversion, formatting, or if market/precision info is missing
        init_logger.warning(f"Error formatting price '{price}' for {symbol}: {e}")
        return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using `fetch_ticker` with robust fallbacks.

    Prioritizes 'last' price. Falls back progressively if 'last' is unavailable or invalid:
    1. Mid-price ((bid + ask) / 2) if both bid and ask are valid positive numbers.
    2. 'ask' price if only ask is valid positive.
    3. 'bid' price if only bid is valid positive.

    Includes retry logic for network errors and rate limits. Handles AuthenticationError critically.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The current market price as a Decimal, or None if fetching fails after retries,
                           no valid price source is found, or a non-retryable error occurs.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching current price for {symbol} (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None
            source = "N/A" # Track which price source was used

            # Helper to safely convert ticker values to positive Decimal using the internal helper
            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                """Safely converts ticker field to positive Decimal, returns None otherwise."""
                return _safe_market_decimal(value, f"ticker.{field_name}", allow_zero=False, allow_negative=False)

            # 1. Try 'last' price first
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')
            if price:
                source = "'last' price"
            else:
                # 2. Fallback to mid-price if 'last' is invalid or missing
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask:
                    # Calculate mid-price only if both bid and ask are valid positive Decimals
                    price = (bid + ask) / Decimal('2')
                    source = f"mid-price (Bid:{bid.normalize()}, Ask:{ask.normalize()})"
                # 3. Fallback to 'ask' if only ask is valid
                elif ask:
                    price = ask
                    source = f"'ask' price ({ask.normalize()})"
                # 4. Fallback to 'bid' if only bid is valid
                elif bid:
                    price = bid
                    source = f"'bid' price ({bid.normalize()})"

            # Check if a valid price was obtained from any source
            if price:
                # Normalize to remove trailing zeros for consistent logging/comparison
                normalized_price = price.normalize()
                lg.debug(f"Current price ({symbol}) obtained via {source}: {normalized_price}")
                return normalized_price
            else:
                # If no valid price found after checking all sources
                last_exception = ValueError(f"No valid positive price found in ticker (last, mid, ask, bid). Ticker data: {ticker}")
                lg.warning(f"Could not find valid current price ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}. Retrying...")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Use a slightly longer delay for rate limit errors
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price ({symbol}): {e}. Pausing for {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempts for rate limit, just wait and retry immediately
            continue
        except ccxt.AuthenticationError as e:
             # Authentication errors are critical and non-retryable
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication error fetching price: {e}. Stopping price fetch.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            # Handle general exchange errors (e.g., symbol not found, temporary issues)
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching price ({symbol}): {e}{RESET}")
            # Could add checks for specific non-retryable error codes here if known
            # For now, assume potentially retryable unless it's an auth error or max retries hit
        except Exception as e:
            # Handle unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price ({symbol}): {e}{RESET}", exc_info=True)
            return None # Exit on unexpected errors, likely non-retryable

        # Increment attempt counter and apply increasing delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to fetch current price ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV (kline) data using CCXT's `fetch_ohlcv` method, robustly handling pagination and errors.

    - Fetches historical data in chunks going backward in time until the `limit` is reached
      or no more data is available from the exchange for the specified period.
    - Automatically determines and uses the correct 'category' parameter for Bybit V5 API calls based on market info.
    - Implements robust retry logic per chunk for network errors and rate limits.
    - Validates the timestamp lag of the most recent chunk to detect potentially stale data.
    - Combines fetched chunks and processes them into a Pandas DataFrame.
    - Converts OHLCV columns to Decimal type for high precision.
    - Cleans the data by dropping rows with NaNs in essential columns (OHLC) or non-positive prices/volumes.
    - Trims the final DataFrame to `MAX_DF_LEN` if it exceeds this internal memory limit.
    - Ensures the final DataFrame is sorted by timestamp (ascending).

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
    lg.info(f"{Fore.CYAN}# Fetching historical klines for {symbol} | TF: {timeframe} | Target Limit: {limit}...{Style.RESET_ALL}")

    # Check if the exchange supports fetching OHLCV data
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support the fetchOHLCV method. Cannot fetch klines.")
        return pd.DataFrame()

    # Estimate minimum required candles for strategy based on config (for logging/warning)
    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        # Find the longest lookback period required by any indicator
        min_required = max(
            sp.get('vt_length', 0) * 2, # Use *2 buffer for EMA stabilization
            sp.get('vt_atr_period', 0),
            sp.get('vt_vol_ema_length', 0),
            sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1,
            sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1
        ) + 50 # Add a general buffer (e.g., 50 candles)
        lg.debug(f"Estimated minimum candles required by strategy configuration: {min_required}")
        if limit < min_required:
            lg.warning(f"{NEON_YELLOW}Requested kline limit ({limit}) is less than estimated strategy requirement ({min_required}). Indicator accuracy may be affected, especially on initial runs.{RESET}")
    except Exception as e:
        lg.warning(f"Could not estimate minimum required candles due to config error: {e}")

    # Determine category and market ID for Bybit V5 specific API calls
    category = 'spot' # Default category
    market_id = symbol # Default market ID (use standard symbol if market info fails)
    try:
        # Retrieve market details from the loaded exchange markets
        market = exchange.market(symbol)
        market_id = market['id'] # Use the exchange-specific ID for API calls
        # Determine category based on market type flags
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        lg.debug(f"Using Bybit category: '{category}' and Market ID: '{market_id}' for kline fetch.")
    except KeyError:
        lg.warning(f"Market '{symbol}' not found in loaded markets during kline fetch setup. Using default category '{category}' and market ID '{market_id}'.")
    except Exception as e:
        lg.warning(f"Could not determine market category/ID for {symbol} kline fetch: {e}. Using defaults.")

    all_ohlcv_data: List[List[Union[int, float, str]]] = [] # List to store combined data from all chunks
    remaining_limit = limit # Number of candles still needed
    end_timestamp_ms: Optional[int] = None # Timestamp to fetch *until* (exclusive), used for pagination backwards
    # Calculate max chunks needed, add buffer for safety (e.g., if API returns slightly fewer than requested)
    max_chunks = math.ceil(limit / BYBIT_API_KLINE_LIMIT) + 2
    chunk_num = 0
    total_fetched_count = 0

    # --- Fetching Loop (Handles Pagination by fetching backwards in time) ---
    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        # Determine how many candles to request in this chunk
        fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT)
        lg.debug(f"Fetching kline chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. Target remaining: {remaining_limit}. Ending before TS: {end_timestamp_ms}")

        attempts = 0
        last_exception = None
        chunk_data: Optional[List[List[Union[int, float, str]]]] = None

        # --- Retry Loop (Per Chunk) ---
        while attempts <= MAX_API_RETRIES:
            try:
                # Prepare parameters, including Bybit's category if applicable
                params = {'category': category} if 'bybit' in exchange.id.lower() else {}
                # Prepare arguments for fetch_ohlcv
                fetch_args: Dict[str, Any] = {
                    'symbol': symbol, # Use standard symbol for CCXT's fetch_ohlcv
                    'timeframe': timeframe,
                    'limit': fetch_size,
                    'params': params
                }
                # If end_timestamp_ms is set (i.e., not the first chunk), fetch candles *before* this time
                if end_timestamp_ms:
                    fetch_args['until'] = end_timestamp_ms # CCXT handles 'until' parameter for pagination

                # Execute the API call to fetch the chunk
                chunk_data = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(chunk_data) if chunk_data else 0
                lg.debug(f"API returned {fetched_count_chunk} candles for chunk {chunk_num} (requested {fetch_size}).")

                if chunk_data:
                    # --- Basic Validation (Timestamp Lag Check on First Chunk) ---
                    if chunk_num == 1: # Only check lag on the *most recent* chunk (first one fetched)
                        try:
                            # Get timestamp of the latest candle received in this chunk
                            last_candle_timestamp_ms = chunk_data[-1][0]
                            last_ts = pd.to_datetime(last_candle_timestamp_ms, unit='ms', utc=True)
                            now_utc = pd.Timestamp.utcnow()
                            # Get interval duration in seconds
                            interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                # Allow lag up to ~2.5 intervals (adjust multiplier if needed)
                                max_allowed_lag = interval_seconds * 2.5
                                actual_lag = (now_utc - last_ts).total_seconds()
                                if actual_lag > max_allowed_lag:
                                    # Data might be stale, warn and trigger a retry for this chunk
                                    last_exception = ValueError(f"Kline data potentially stale (Lag: {actual_lag:.1f}s > Max Allowed: {max_allowed_lag:.1f}s).")
                                    lg.warning(f"{NEON_YELLOW}Timestamp lag detected ({symbol}, Chunk 1): {last_exception}. Retrying fetch...{RESET}")
                                    chunk_data = None # Discard stale data and let retry logic handle it
                                    # No 'break' here, fall through to retry delay
                                else:
                                    lg.debug(f"Timestamp lag check passed for first chunk ({symbol}).")
                                    break # Chunk fetched and validated, exit retry loop
                            else:
                                lg.warning(f"Could not parse timeframe '{timeframe}' to seconds for lag check. Skipping validation.")
                                break # Proceed without lag check if parsing fails
                        except Exception as ts_err:
                            lg.warning(f"Could not validate timestamp lag ({symbol}, Chunk 1): {ts_err}. Proceeding cautiously.")
                            break # Proceed if validation itself fails
                    else:
                         # For subsequent (older) chunks, just break retry loop on successful fetch
                         break
                else:
                    # If API returns an empty list, it likely means no more historical data is available
                    lg.debug(f"API returned no data for chunk {chunk_num} (End TS: {end_timestamp_ms}). Assuming end of available history for this period.")
                    remaining_limit = 0 # Stop fetching further chunks
                    break # Exit retry loop for this chunk

            # --- Error Handling (Per Chunk) ---
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_exception = e
                lg.warning(f"{NEON_YELLOW}Network error fetching kline chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait_time = RETRY_DELAY_SECONDS * 3 # Longer wait for rate limits
                lg.warning(f"{NEON_YELLOW}Rate limit fetching kline chunk {chunk_num} ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
                time.sleep(wait_time)
                continue # Continue retry loop without incrementing attempts
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Authentication error fetching klines: {e}. Stopping fetch.{RESET}")
                return pd.DataFrame() # Fatal error, cannot proceed
            except ccxt.ExchangeError as e:
                last_exception = e
                lg.error(f"{NEON_RED}Exchange error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}")
                # Check for specific non-retryable errors related to klines if known
                err_str = str(e).lower()
                if "invalid timeframe" in err_str or "interval is not supported" in err_str or "symbol invalid" in err_str:
                     lg.critical(f"{NEON_RED}Non-retryable error fetching klines ({symbol}): {e}. Stopping fetch.{RESET}")
                     return pd.DataFrame()
                # Assume potentially retryable otherwise
            except Exception as e:
                last_exception = e
                lg.error(f"{NEON_RED}Unexpected error fetching klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True)
                return pd.DataFrame() # Stop on unexpected errors

            # Increment attempt counter and apply delay (only if retry is needed and not rate limited)
            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None: # Only sleep if retry is needed
                 time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff
        # --- End Retry Loop (Per Chunk) ---

        # --- Process Successful Chunk ---
        if chunk_data:
            # Prepend the newly fetched (older) data to the main list
            all_ohlcv_data = chunk_data + all_ohlcv_data
            chunk_len = len(chunk_data)
            remaining_limit -= chunk_len
            total_fetched_count += chunk_len

            # Set the end timestamp for the *next* fetch request to be the timestamp
            # of the *oldest* candle received in this chunk, minus 1ms, to avoid overlap and fetch older data.
            end_timestamp_ms = chunk_data[0][0] - 1

            # Check if the API returned fewer candles than requested, implies end of available history
            if chunk_len < fetch_size:
                 lg.debug(f"Received fewer candles ({chunk_len}) than requested ({fetch_size}) for chunk {chunk_num}. Assuming end of available history.")
                 remaining_limit = 0 # Stop fetching further chunks
        else:
            # Fetching the chunk failed after all retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
            # Decide whether to proceed with partial data or fail entirely
            if not all_ohlcv_data: # Failed on the very first chunk
                 lg.error(f"Failed on first chunk ({symbol}). Cannot proceed. Returning empty DataFrame.")
                 return pd.DataFrame()
            else:
                 lg.warning(f"Proceeding with {total_fetched_count} candles fetched before error occurred in chunk {chunk_num}.")
                 break # Exit the main fetching loop, use the data gathered so far

        # Small delay between chunk fetches to be polite to the API
        if remaining_limit > 0: time.sleep(0.5)
    # --- End Fetching Loop ---

    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines ({symbol}) after reaching max chunks ({max_chunks}), but {remaining_limit} candles still targeted. Consider increasing 'fetch_limit' or checking API.")

    # --- Process Combined Data ---
    if not all_ohlcv_data:
        lg.error(f"No kline data could be successfully fetched for {symbol} {timeframe}. Returning empty DataFrame.")
        return pd.DataFrame()

    lg.info(f"Total klines fetched across all requests: {total_fetched_count}")

    # --- Deduplicate and Sort ---
    # Although fetching backwards should prevent duplicates, add a check just in case of API overlap issues.
    seen_timestamps = set()
    unique_data = []
    # Iterate in reverse to keep the latest instance in case of exact duplicates (unlikely but safe)
    for candle in reversed(all_ohlcv_data):
        ts = candle[0] # Timestamp is the first element
        if ts not in seen_timestamps:
            unique_data.append(candle)
            seen_timestamps.add(ts)
    unique_data.reverse() # Put back in ascending time order (oldest first)

    duplicates_removed = len(all_ohlcv_data) - len(unique_data)
    if duplicates_removed > 0:
        lg.warning(f"Removed {duplicates_removed} duplicate candle timestamps during processing ({symbol}).")
    all_ohlcv_data = unique_data

    # Sort by timestamp just to be absolutely sure (should already be sorted)
    all_ohlcv_data.sort(key=lambda x: x[0])

    # Trim to the originally requested number of candles (most recent ones) if more were fetched due to chunking logic
    if len(all_ohlcv_data) > limit:
        lg.debug(f"Fetched {len(all_ohlcv_data)} unique candles, trimming to originally requested limit {limit} (most recent).")
        all_ohlcv_data = all_ohlcv_data[-limit:]

    # --- Convert to Pandas DataFrame with Decimal Types ---
    try:
        lg.debug(f"Processing {len(all_ohlcv_data)} final candles into DataFrame ({symbol})...")
        # Standard CCXT OHLCV format: [timestamp, open, high, low, close, volume]
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_ohlcv_data, columns=cols[:len(all_ohlcv_data[0])]) # Handle if fewer columns returned

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        if df.empty:
            lg.error(f"DataFrame empty after timestamp conversion ({symbol}). Cannot proceed.")
            return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision, handling potential errors robustly
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Apply pd.to_numeric first, coercing errors (non-numeric strings) to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # Convert valid finite numbers to Decimal, map NaN/inf/-inf to Decimal('NaN')
                df[col] = numeric_series.apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                )
            else:
                 lg.warning(f"Expected column '{col}' not found in fetched kline data ({symbol}).")

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in essential price columns (O, H, L, C)
        essential_price_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_price_cols, inplace=True)
        # Drop rows with non-positive close price (often indicates bad data)
        df = df[df['close'] > Decimal('0')]
        # Drop rows with NaN volume or negative volume (if volume column exists)
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            # Allow zero volume candles (can happen during low activity periods)
            df = df[df['volume'] >= Decimal('0')]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows ({symbol}) during cleaning (NaNs, zero/negative prices/volumes).")

        if df.empty:
            lg.warning(f"Kline DataFrame empty after cleaning ({symbol}). Cannot proceed.")
            return pd.DataFrame()

        # Ensure DataFrame is sorted by timestamp index (final check after potential drops)
        if not df.index.is_monotonic_increasing:
            lg.warning(f"Kline index not monotonic after cleaning ({symbol}), sorting again..."); df.sort_index(inplace=True)

        # --- Memory Management ---
        # Trim DataFrame if it exceeds the maximum allowed length (remove oldest data)
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds max ({MAX_DF_LEN}). Trimming oldest data ({symbol})."); df = df.iloc[-MAX_DF_LEN:].copy()

        lg.info(f"{NEON_GREEN}Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}{RESET}")
        return df
    except Exception as e:
        lg.error(f"{NEON_RED}Error processing fetched kline data into DataFrame ({symbol}): {e}{RESET}", exc_info=True)
        return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Retrieves, validates, and standardizes market information for a given symbol using `exchange.market()`.

    - Reloads markets using `exchange.load_markets(reload=True)` if the symbol is initially not found.
    - Extracts key details like precision (price step, amount step), limits (min/max amount, cost),
      contract type (linear/inverse/spot), and contract size from the CCXT market structure.
    - Parses these details into Decimal types for reliable calculations.
    - Adds convenience flags (`is_contract`, `is_linear`, `is_inverse`, `contract_type_str`) for easier logic.
    - Includes retry logic for network errors during the potential market reload.
    - Logs critical errors and returns None if the symbol is not found after reload, or if essential precision data
      (amount step, price step) is missing, invalid, or non-positive, as these are crucial for trading operations.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object (must have markets loaded or be able to load them).
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[MarketInfo]: A `MarketInfo` TypedDict containing standardized market details, including
                              parsed Decimal values for limits/precision. Returns None if the market is not found,
                              essential data (amount/price precision steps) is missing or invalid, or a critical error occurs.
    """
    lg = logger
    lg.debug(f"Retrieving market information for symbol: {symbol}...")
    attempts = 0
    last_exception = None
    market_dict: Optional[Dict] = None

    while attempts <= MAX_API_RETRIES:
        try:
            # Check if markets are loaded and the symbol exists in the current market list
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' not found in memory. Attempting to refresh market data...")
                try:
                    # Force reload market data from the exchange
                    exchange.load_markets(reload=True)
                    lg.info("Market data refreshed successfully.")
                except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as reload_err:
                    last_exception = reload_err
                    lg.warning(f"Network error refreshing market data for {symbol}: {reload_err}. Retry {attempts + 1}...")
                    # Fall through to the retry delay logic
                except ccxt.AuthenticationError as reload_err:
                    last_exception = reload_err
                    lg.critical(f"{NEON_RED}Authentication error refreshing market data: {reload_err}. Cannot proceed.{RESET}")
                    return None # Fatal error
                except Exception as reload_err:
                     last_exception = reload_err
                     lg.error(f"Failed to refresh market data while searching for {symbol}: {reload_err}")
                     # Fall through to the retry delay logic

            # Try fetching the market dictionary again after potential reload
            try:
                # Retrieve the specific market dictionary using the standard symbol
                market_dict = exchange.market(symbol)
            except ccxt.BadSymbol:
                 # CCXT raises BadSymbol if the symbol is definitively not available on the exchange
                 lg.error(f"{NEON_RED}Symbol '{symbol}' is not listed or is invalid on {exchange.id}.{RESET}")
                 return None # Non-retryable error
            except Exception as fetch_err:
                 # Catch other errors during market dictionary retrieval
                 last_exception = fetch_err
                 lg.warning(f"Error retrieving market dictionary for '{symbol}' after potential reload: {fetch_err}. Retry {attempts + 1}...")
                 market_dict = None # Ensure market_dict is None to trigger retry

            if market_dict:
                # --- Market Found - Extract and Standardize Details ---
                lg.debug(f"Market found for '{symbol}'. Parsing and standardizing details...")
                # Work on a copy to avoid modifying the cached market object
                std_market = market_dict.copy()

                # Add custom flags for easier logic later based on standard CCXT fields
                std_market['is_contract'] = std_market.get('contract', False) or std_market.get('type') in ['swap', 'future', 'option']
                std_market['is_linear'] = bool(std_market.get('linear')) and std_market['is_contract']
                std_market['is_inverse'] = bool(std_market.get('inverse')) and std_market['is_contract']
                # Determine user-friendly contract type string
                std_market['contract_type_str'] = "Linear" if std_market['is_linear'] else \
                                                  "Inverse" if std_market['is_inverse'] else \
                                                  "Spot" if std_market.get('spot') else "Unknown"

                # --- Safely parse precision and limits into Decimal ---
                precision_info = std_market.get('precision', {})
                limits_info = std_market.get('limits', {})
                amount_limits_info = limits_info.get('amount', {})
                cost_limits_info = limits_info.get('cost', {})

                # Parse precision steps (CRITICAL - must be positive)
                # Use the safe helper which handles None, strings, and positivity checks
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision_info.get('amount'), 'precision.amount', allow_zero=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision_info.get('price'), 'precision.price', allow_zero=False)

                # Parse limits (allow zero for min, must be positive if set for max)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits_info.get('min'), 'limits.amount.min', allow_zero=True)
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits_info.get('max'), 'limits.amount.max', allow_zero=False)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits_info.get('min'), 'limits.cost.min', allow_zero=True)
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits_info.get('max'), 'limits.cost.max', allow_zero=False)

                # Parse contract size (must be positive, default to 1 if not applicable/found)
                contract_size_val = std_market.get('contractSize', '1') # Default to '1' if missing
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, 'contractSize', allow_zero=False) or Decimal('1') # Ensure positive, fallback to 1

                # --- Critical Validation: Essential Precision Steps ---
                # These are absolutely required for placing orders and calculating sizes correctly.
                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None:
                    lg.critical(f"{NEON_RED}CRITICAL VALIDATION FAILED:{RESET} Market '{symbol}' missing essential positive precision data.")
                    lg.error(f"  Parsed Amount Step: {std_market['amount_precision_step_decimal']}, Parsed Price Step: {std_market['price_precision_step_decimal']}")
                    lg.error(f"  Raw Precision Dict from CCXT: {precision_info}")
                    lg.error("Cannot proceed safely without valid amount and price precision steps. Check exchange API or CCXT market data.")
                    return None # Returning None forces the calling function to handle the failure gracefully

                # Log extracted details for verification at debug level
                amt_step_str = std_market['amount_precision_step_decimal'].normalize()
                price_step_str = std_market['price_precision_step_decimal'].normalize()
                min_amt_str = std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] is not None else 'None'
                max_amt_str = std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] is not None else 'None'
                min_cost_str = std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] is not None else 'None'
                max_cost_str = std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] is not None else 'None'
                contract_size_str = std_market['contract_size_decimal'].normalize()

                log_msg = (
                    f"Market Details Parsed ({symbol}): Type={std_market['contract_type_str']}, Active={std_market.get('active', 'N/A')}\n"
                    f"  Precision Steps (Amount/Price): {amt_step_str} / {price_step_str}\n"
                    f"  Limits - Amount (Min/Max): {min_amt_str} / {max_amt_str}\n"
                    f"  Limits - Cost (Min/Max): {min_cost_str} / {max_cost_str}\n"
                    f"  Contract Size: {contract_size_str}"
                )
                lg.debug(log_msg)

                # Cast the processed dictionary to MarketInfo TypedDict for type safety downstream
                try:
                    # Direct casting assumes the structure matches after parsing. Type checking tools can verify this.
                    final_market_info: MarketInfo = std_market # type: ignore
                    return final_market_info
                except Exception as cast_err:
                     # This should ideally not happen if parsing logic matches TypedDict structure
                     lg.error(f"Internal error casting processed market dict to MarketInfo TypedDict ({symbol}): {cast_err}")
                     # Fallback: Return the dictionary anyway if casting fails but data seems okay
                     return std_market # type: ignore
            else:
                # Market dictionary was None, meaning fetch failed or symbol not found after reload
                if attempts < MAX_API_RETRIES:
                    lg.warning(f"Symbol '{symbol}' not found or market fetch failed (Attempt {attempts + 1}). Retrying check...")
                    # Fall through to retry delay logic
                else:
                    # Max retries reached, symbol definitively not found or fetch failed
                    lg.error(f"{NEON_RED}Market '{symbol}' not found on {exchange.id} after reload and retries. Last error: {last_exception}{RESET}")
                    return None

        # --- Error Handling (Outside Market Fetch/Reload) ---
        # These catch errors related to the CCXT market() call itself or unexpected issues
        except ccxt.BadSymbol as e:
            # Symbol is definitively invalid according to the exchange (caught again here for safety)
            lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}")
            return None
        except ccxt.AuthenticationError as e:
             # Should have been caught during reload, but handle defensively
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication error retrieving market info: {e}. Stopping.{RESET}")
             return None # Fatal error
        except ccxt.ExchangeError as e:
            # Handle general exchange errors during market info retrieval
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error retrieving market info ({symbol}): {e}{RESET}")
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries for ExchangeError retrieving market info ({symbol}).{RESET}")
                 return None
        except Exception as e:
            # Handle any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error retrieving market info ({symbol}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying (only if market_dict is None)
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without finding the market
    lg.error(f"{NEON_RED}Failed to retrieve market info for '{symbol}' after all attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available trading balance for a specific currency (e.g., USDT).

    - Handles Bybit V5 account types (UNIFIED, CONTRACT) automatically to find the relevant balance,
      as assets for linear contracts are often in the UNIFIED account.
    - Parses various potential balance fields ('free', 'availableToWithdraw', 'availableBalance')
      to robustly find the usable balance amount.
    - Includes retry logic for network errors and rate limits.
    - Handles authentication errors critically by re-raising them, as they indicate a fundamental issue.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        currency (str): The currency code to fetch the balance for (e.g., "USDT"). Case-sensitive.
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The available balance as a non-negative Decimal, or None if fetching fails
                           after retries, the currency is not found, or a non-auth critical error occurs.

    Raises:
        ccxt.AuthenticationError: If authentication fails during the balance fetch attempt.
    """
    lg = logger
    lg.debug(f"Fetching balance for currency: {currency}...")
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None # Store the raw balance string found
            balance_source_desc: str = "N/A" # Description of where balance was found
            found: bool = False
            balance_info: Optional[Dict] = None # Store the raw response from fetch_balance

            # Bybit V5 often requires specifying account type.
            # Check relevant types, then fallback to default request if specific types fail.
            account_types_to_check = []
            if 'bybit' in exchange.id.lower():
                 # UNIFIED often holds assets for Linear USDT/USDC contracts and Spot
                 # CONTRACT is primarily for Inverse contracts
                 # Check both relevant types first, then default
                 account_types_to_check = ['UNIFIED', 'CONTRACT']
            account_types_to_check.append('') # Always check default/unspecified type as a fallback

            for acc_type in account_types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {} # Only add param if type specified
                    type_desc = f"Account Type: {acc_type}" if acc_type else "Default Account"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")
                    balance_info = exchange.fetch_balance(params=params)

                    # --- Try different ways to extract balance from the response ---

                    # 1. Standard CCXT structure: Look for currency key and 'free' sub-key
                    if currency in balance_info and isinstance(balance_info[currency], dict) and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        balance_source_desc = f"{type_desc} ('free' field)"
                        lg.debug(f"Found balance in standard structure ({balance_source_desc}): {balance_str}")
                        found = True; break # Exit account type loop

                    # 2. Bybit V5 structure: Check nested 'info' -> 'result' -> 'list' -> 'coin' array
                    elif 'info' in balance_info and isinstance(balance_info.get('info'), dict) and \
                         'result' in balance_info['info'] and isinstance(balance_info['info'].get('result'), dict) and \
                         isinstance(balance_info['info']['result'].get('list'), list):

                        for account_details in balance_info['info']['result']['list']:
                             # Check if accountType matches the one requested (or if no type was requested)
                             account_type_match = (not acc_type or account_details.get('accountType') == acc_type)
                             # Check if this account detail contains coin information
                             if account_type_match and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Found the correct currency within this account type
                                        # Prioritize fields representing available funds for trading:
                                        # 'availableToWithdraw' or 'availableBalance' are usually best for Unified/Linear
                                        # 'walletBalance' might include locked funds, less preferred
                                        balance_val = coin_data.get('availableToWithdraw')
                                        source_field = 'availableToWithdraw'
                                        if balance_val is None:
                                             balance_val = coin_data.get('availableBalance')
                                             source_field = 'availableBalance'
                                        # Fallback to walletBalance if others missing
                                        if balance_val is None:
                                             balance_val = coin_data.get('walletBalance')
                                             source_field = 'walletBalance'

                                        if balance_val is not None:
                                            balance_str = str(balance_val)
                                            balance_source_desc = f"Bybit V5 ({account_details.get('accountType')} Account, Field: '{source_field}')"
                                            lg.debug(f"Found balance in Bybit V5 structure ({balance_source_desc}): {balance_str}")
                                            found = True; break # Found in coin list, exit inner loop
                                if found: break # Exit account details loop
                        if found: break # Exit account type loop

                except ccxt.ExchangeError as e:
                    # Errors like "account type does not exist" are expected when checking multiple types, ignore them
                    err_str = str(e).lower()
                    if acc_type and ("account type does not exist" in err_str or "invalid account type" in err_str):
                        lg.debug(f"Account type '{acc_type}' not found or invalid for balance fetch. Trying next...")
                    elif acc_type: # Log other exchange errors for specific types but continue trying others
                        lg.debug(f"Exchange error fetching balance ({type_desc}): {e}. Trying next account type...")
                        last_exception = e # Store last error encountered
                    else: # Re-raise error only if the default fetch (acc_type='') fails
                        raise e
                    continue # Try the next account type
                except Exception as e:
                    # Catch other unexpected errors during a specific account type check
                    lg.warning(f"Unexpected error fetching balance ({type_desc}): {e}. Trying next account type...")
                    last_exception = e # Store last error encountered
                    continue # Try the next account type

            # --- Process the result after checking all account types ---
            if found and balance_str is not None:
                try:
                    # Convert the found balance string to Decimal
                    balance_decimal = Decimal(balance_str)
                    # Ensure balance is not negative (clamp to 0 if API returns small negative dust)
                    final_balance = max(balance_decimal, Decimal('0'))
                    lg.debug(f"Parsed balance ({currency}) from {balance_source_desc}: {final_balance.normalize()}")
                    return final_balance # Success
                except (ValueError, InvalidOperation, TypeError) as e:
                    # Raise an error if the found balance string cannot be converted to Decimal
                    raise ccxt.ExchangeError(f"Failed to convert balance string '{balance_str}' for {currency} (from {balance_source_desc}) to Decimal: {e}")
            elif not found and balance_info is not None:
                # If fetch_balance returned data, but the currency wasn't found in expected structures
                lg.warning(f"Could not find balance for '{currency}' in the response structure. Last raw response info: {balance_info.get('info', balance_info)}")
                raise ccxt.ExchangeError(f"Balance for currency '{currency}' not found in response.")
            elif not found and balance_info is None:
                # If fetch_balance itself failed to return anything meaningful after trying all types
                raise ccxt.ExchangeError(f"Balance fetch for '{currency}' failed to return any data.")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit fetching balance ({currency}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            # Authentication errors are critical, re-raise immediately
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching balance: {e}. Stopping balance fetch.{RESET}")
            raise e # Re-raise AuthenticationError to be handled by the caller (e.g., initialize_exchange)
        except ccxt.ExchangeError as e:
            # Log other exchange errors (like currency not found, conversion errors) and allow retry
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e:
            # Handle unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance ({currency}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[PositionInfo]:
    """
    Checks for an existing open position for the given symbol using `fetch_positions`.

    - Handles Bybit V5 specifics: uses 'category' parameter, filters results by symbol/market ID,
      and parses crucial details (size, SL/TP, TSL) from the nested `info` field.
    - Determines position side ('long'/'short') and size accurately using Decimal. Size is positive for long, negative for short.
    - Parses key position details (entry price, leverage, PnL, SL/TP, TSL distance/activation) into a standardized `PositionInfo` format.
    - Includes retry logic for network errors and rate limits.
    - Returns a standardized `PositionInfo` dictionary if an active position is found (size significantly different from zero),
      otherwise returns None. Returns None on failure.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[PositionInfo]: A `PositionInfo` TypedDict containing details of the open position if found,
                                otherwise None (no position or fetch failed).
    """
    lg = logger
    attempts = 0
    last_exception = None
    market_id: Optional[str] = None
    category: Optional[str] = None

    # --- Determine Market ID and Category (required for Bybit V5 position fetching) ---
    try:
        market = exchange.market(symbol)
        market_id = market['id'] # Use exchange-specific ID
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        # Positions are typically only relevant for contracts (Linear/Inverse)
        if category == 'spot':
            lg.debug(f"Position check skipped for {symbol}: Spot market.")
            return None # No position concept in the same way for spot
        lg.debug(f"Using Market ID: '{market_id}', Category: '{category}' for position check.")
    except KeyError:
        lg.error(f"Market '{symbol}' not found in loaded markets. Cannot check position.")
        return None
    except Exception as e:
        lg.error(f"Error determining market details for position check ({symbol}): {e}")
        return None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts + 1})...")
            positions: List[Dict] = [] # List to store raw position data from CCXT

            # --- Fetch Positions (Handling Bybit V5 Specifics) ---
            try:
                # Use fetch_positions with specific symbol and category params for Bybit V5
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching positions with params: {params}")

                # Check if exchange supports fetchPositions (most do)
                if exchange.has.get('fetchPositions'):
                     # Fetch positions - this might return positions for other symbols if filtering isn't perfect on exchange side
                     all_positions = exchange.fetch_positions(params=params)
                     # Filter the results manually by symbol (standard) or market ID (from info) just in case
                     positions = [
                         p for p in all_positions
                         if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id
                     ]
                     lg.debug(f"Fetched {len(all_positions)} total positions ({category}), filtered to {len(positions)} matching {symbol}/{market_id}.")
                else:
                     # Should not happen for Bybit, but handle defensively
                     raise ccxt.NotSupported(f"{exchange.id} does not support fetchPositions.")

            except ccxt.ExchangeError as e:
                 # Bybit often returns specific error codes/messages for "no position found" scenarios,
                 # which are not actual errors in this context. Treat them as "no position".
                 no_pos_codes = [
                     110025, # "position idx not match position mode" (can indicate no pos in one-way mode)
                     # Add other known "no position" codes if discovered
                 ]
                 no_pos_messages = [
                     "position not found", "no position", "position does not exist",
                     "order not found or too late to cancel" # Sometimes used ambiguously
                 ]
                 err_str = str(e).lower()
                 # Try to extract Bybit's retCode reliably from the error arguments
                 code_str = ""
                 # Check common patterns for retCode in error messages
                 match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
                 if match:
                     code_str = match.group(2)
                 if not code_str: # Fallback check on exception attributes if regex fails
                      code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))

                 # Check if the extracted code or message indicates "no position"
                 code_match = any(str(c) == code_str for c in no_pos_codes) if code_str else False
                 msg_match = any(msg in err_str for msg in no_pos_messages)

                 if code_match or msg_match:
                     lg.info(f"No open position found for {symbol} (Detected via exchange message: {e}).")
                     return None # Successfully determined no position exists
                 else:
                     # Re-raise other exchange errors to be handled by the main retry loop below
                     raise e

            # --- Process Fetched Positions to Find the Active One ---
            active_position_raw: Optional[Dict] = None
            # Define a small threshold based on amount precision to consider a position "open"
            # Avoids treating dust positions as active. Use a fraction of the minimum amount step.
            size_threshold = Decimal('1e-9') # Default small value if precision unavailable
            try:
                amt_step = market_info.get('amount_precision_step_decimal')
                if amt_step and amt_step > 0:
                    # Use a small fraction (e.g., 1%) of the step size as the threshold
                    size_threshold = amt_step * Decimal('0.01')
                else:
                    lg.warning(f"Could not get valid amount precision for {symbol}. Using default size threshold: {size_threshold}")
            except Exception as prec_err:
                lg.warning(f"Error getting amount precision for {symbol} size threshold: {prec_err}. Using default: {size_threshold}")
            lg.debug(f"Using position size threshold (absolute value > {size_threshold.normalize()}) to determine active position ({symbol}).")

            # Iterate through the filtered positions (usually 0 or 1 in one-way mode)
            for pos in positions:
                # Extract size: Prioritize Bybit V5's info['size'], fallback to standard 'contracts'
                # Bybit V5 `info['size']` is usually a string representing contracts (positive or negative)
                size_str_info = str(pos.get('info', {}).get('size', '')).strip()
                # Standard CCXT `contracts` field (often float, less reliable for Bybit V5)
                size_str_std = str(pos.get('contracts', '')).strip()
                # Use info['size'] if available and non-empty, otherwise fallback
                size_str = size_str_info if size_str_info else size_str_std

                if not size_str:
                    lg.debug(f"Skipping position entry with missing size data ({symbol}): {pos.get('info', {})}")
                    continue # Skip if no size information found

                try:
                    # Convert size to Decimal and check if its absolute value exceeds the threshold
                    size_decimal = Decimal(size_str)
                    if abs(size_decimal) > size_threshold:
                        # Found an active position with significant size
                        active_position_raw = pos
                        # Store the parsed Decimal size directly in the dictionary for standardization
                        active_position_raw['size_decimal'] = size_decimal
                        lg.debug(f"Found active position entry ({symbol}): Size={size_decimal.normalize()}")
                        break # Stop searching once an active position is found (assume one-way mode)
                    else:
                        # Log positions with near-zero size for debugging, but don't treat as active
                        lg.debug(f"Skipping position entry with near-zero size ({symbol}, Size={size_decimal.normalize()}): {pos.get('info', {})}")
                except (ValueError, InvalidOperation, TypeError) as parse_err:
                     # Log error if size string cannot be parsed, skip this entry
                     lg.warning(f"Could not parse position size '{size_str}' for {symbol}: {parse_err}. Skipping this position entry.")
                     continue # Move to the next position entry in the list

            # --- Format and Return Active Position Info ---
            if active_position_raw:
                # Standardize the position dictionary using PositionInfo structure
                std_pos = active_position_raw.copy() # Work on a copy
                info_field = std_pos.get('info', {}) # Get the raw exchange-specific details

                # Determine Side ('long'/'short') reliably
                side = std_pos.get('side') # Standard CCXT field ('long' or 'short')
                parsed_size = std_pos['size_decimal'] # Use the Decimal size parsed earlier

                # If standard 'side' field is missing or invalid, infer from Bybit V5 'side' or size sign
                if side not in ['long', 'short']:
                    side_v5 = str(info_field.get('side', '')).strip().lower() # Bybit uses 'Buy'/'Sell'
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif parsed_size > size_threshold: side = 'long' # Infer from positive size
                    elif parsed_size < -size_threshold: side = 'short' # Infer from negative size
                    else: side = None # Cannot determine side if size is near zero

                if not side:
                    lg.error(f"Could not determine side for active position ({symbol}). Size: {parsed_size}. Data: {info_field}")
                    return None # Cannot proceed without a clear side

                std_pos['side'] = side # Update the standardized dict with the determined side

                # Standardize other key fields using safe Decimal conversion helper
                # Prioritize standard CCXT fields, fallback to Bybit V5 `info` fields
                std_pos['entryPrice'] = _safe_market_decimal(std_pos.get('entryPrice') or info_field.get('avgPrice') or info_field.get('entryPrice'), 'pos.entryPrice', allow_zero=False)
                std_pos['leverage'] = _safe_market_decimal(std_pos.get('leverage') or info_field.get('leverage'), 'pos.leverage', allow_zero=False)
                std_pos['liquidationPrice'] = _safe_market_decimal(std_pos.get('liquidationPrice') or info_field.get('liqPrice'), 'pos.liqPrice', allow_zero=False)
                std_pos['unrealizedPnl'] = _safe_market_decimal(std_pos.get('unrealizedPnl') or info_field.get('unrealisedPnl') or info_field.get('unrealizedPnl'), 'pos.pnl', allow_zero=True, allow_negative=True) # Pnl can be zero or negative

                # --- Parse protection levels from Bybit V5 `info` field ---
                # These fields often return "0" as a string if not set. We want None if not set, or the string value if set.
                def get_protection_field(field_name: str) -> Optional[str]:
                    """Extracts protection field from info if it's a valid non-zero number string."""
                    value = info_field.get(field_name)
                    s_value = str(value).strip() if value is not None else None
                    if not s_value: return None # Treat None or empty string as not set
                    try:
                         # Check if it represents a non-zero value
                         if abs(Decimal(s_value)) > Decimal('1e-12'): # Use tolerance for floating point "0" strings like "0.000000"
                             return s_value # Return the string value if it's non-zero
                    except (InvalidOperation, ValueError, TypeError):
                         # If it's not a valid number string (e.g., "None", ""), treat as not set
                         return None
                    # If it's a zero value string ("0", "0.0"), treat as not set
                    return None

                std_pos['stopLossPrice'] = get_protection_field('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_field('takeProfit')
                std_pos['trailingStopLoss'] = get_protection_field('trailingStop') # TSL distance string
                std_pos['tslActivationPrice'] = get_protection_field('activePrice') # TSL activation price string

                # Initialize bot state flags (these are IN-MEMORY ONLY and reset each run)
                std_pos['be_activated'] = False # Will be set by management logic if BE applied *during this run*
                # Infer initial bot TSL state based on whether TSL distance is already set via API
                std_pos['tsl_activated'] = bool(std_pos['trailingStopLoss'])

                # Helper for formatting Decimal values for logging, handling None
                def format_decimal_log(value: Optional[Any]) -> str:
                    """Formats Decimal/None for logging."""
                    dec_val = _safe_market_decimal(value, 'log_format', allow_zero=True, allow_negative=True)
                    return dec_val.normalize() if dec_val is not None else 'N/A'

                # Log summary of the found active position
                ep_str = format_decimal_log(std_pos.get('entryPrice'))
                size_str = std_pos['size_decimal'].normalize() # Already Decimal
                sl_str = std_pos.get('stopLossPrice') or 'N/A' # Use string directly or N/A
                tp_str = std_pos.get('takeProfitPrice') or 'N/A'
                tsl_dist_str = std_pos.get('trailingStopLoss') or 'N/A'
                tsl_act_str = std_pos.get('tslActivationPrice') or 'N/A'
                # Combine TSL log parts only if at least one is set
                tsl_log = f"Dist={tsl_dist_str}/Act={tsl_act_str}" if tsl_dist_str != 'N/A' or tsl_act_str != 'N/A' else "N/A"
                pnl_str = format_decimal_log(std_pos.get('unrealizedPnl'))
                liq_str = format_decimal_log(std_pos.get('liquidationPrice'))

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} Size={size_str}, Entry={ep_str}, Liq={liq_str}, PnL={pnl_str}, SL={sl_str}, TP={tp_str}, TSL={tsl_log}")

                # Cast the standardized dictionary to PositionInfo TypedDict before returning
                try:
                    # Assume structure matches after parsing
                    final_position_info: PositionInfo = std_pos # type: ignore
                    return final_position_info
                except Exception as cast_err:
                     # This indicates a mismatch between parsing and TypedDict definition
                     lg.error(f"Internal error casting position dict to PositionInfo TypedDict ({symbol}): {cast_err}")
                     return std_pos # type: ignore # Return raw dict if cast fails, but log error

            else:
                # No position with size > threshold was found after filtering
                lg.info(f"No active position found for {symbol} (or size below threshold).")
                return None

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit fetching positions ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error fetching positions: {e}. Stopping position check.{RESET}")
            return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            # Log other exchange errors and allow retry unless max attempts reached
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            # Could add checks for specific non-retryable errors here if needed
        except Exception as e:
            # Handle unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching positions ({symbol}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to get position info ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a derivatives symbol using CCXT's `set_leverage` method.

    - Skips the operation if the market is not a contract (spot) or if the requested leverage is invalid (<= 0).
    - Handles Bybit V5 specific parameters: uses 'category' and sets both 'buyLeverage' and 'sellLeverage'.
    - Includes retry logic for network/exchange errors.
    - Checks Bybit V5 response codes: Treats `0` (success) and `110045` (leverage not modified, already set) as success.
    - Identifies and handles known non-retryable leverage setting errors based on Bybit codes or messages.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        leverage (int): The desired integer leverage level (e.g., 10 for 10x).
        market_info (MarketInfo): The standardized MarketInfo dictionary for the symbol.
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        bool: True if leverage was set successfully or was already set to the desired level, False otherwise.
    """
    lg = logger
    # --- Input and Market Type Validation ---
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        # Consider it success as no action is needed for spot markets
        return True
    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value '{leverage}'. Must be a positive integer.")
        return False
    # Check if the exchange instance supports setting leverage
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support the setLeverage method.")
        return False

    market_id = market_info['id'] # Use the exchange-specific market ID for the API call

    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {market_id} to {leverage}x (Attempt {attempts + 1})...")
            params = {}
            # --- Prepare Bybit V5 Specific Parameters ---
            if 'bybit' in exchange.id.lower():
                 # Determine category (linear/inverse) from market info
                 category = market_info.get('contract_type_str', 'Linear').lower() # Default to linear if unknown
                 if category not in ['linear', 'inverse']:
                      lg.error(f"Leverage setting failed ({symbol}): Invalid contract category '{category}'. Must be 'linear' or 'inverse'.")
                      return False
                 # Bybit V5 requires category and setting both buy/sell leverage as strings
                 params = {
                     'category': category,
                     'buyLeverage': str(leverage),
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 setLeverage params: {params}")
            # Other exchanges might require different params, add logic here if needed

            # --- Execute set_leverage call ---
            # Pass the market_id as the symbol argument for Bybit, leverage as int, and specific params
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"set_leverage raw response ({symbol}): {response}")

            # --- Check Response (Especially for Bybit V5 specific codes) ---
            ret_code_str: Optional[str] = None
            ret_msg: str = "N/A"
            # Bybit V5 response structure often has retCode/retMsg in the 'info' field
            if isinstance(response, dict):
                 info_dict = response.get('info', {}) # CCXT usually puts raw response here
                 # Check for retCode in info first, then top level if info is missing/empty
                 ret_code_info = info_dict.get('retCode')
                 ret_code_top = response.get('retCode')
                 # Prefer retCode from info if available
                 raw_code = ret_code_info if ret_code_info is not None else ret_code_top
                 if raw_code is not None:
                     ret_code_str = str(raw_code)
                 # Get message similarly
                 ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown Bybit message'))

            # Check Bybit return codes
            if ret_code_str == '0':
                 lg.info(f"{NEON_GREEN}Leverage set successfully for {market_id} to {leverage}x (Code: 0).{RESET}")
                 return True
            elif ret_code_str == '110045': # Bybit code for "Leverage not modified"
                 lg.info(f"{NEON_YELLOW}Leverage for {market_id} already set to {leverage}x (Code: 110045).{RESET}")
                 return True
            elif ret_code_str is not None and ret_code_str not in ['None', '0']: # Check if a non-zero, non-success code was returned
                 # Raise an ExchangeError for other non-zero Bybit return codes to allow retry or specific handling
                 error_message = f"Bybit API error setting leverage ({symbol}): {ret_msg} (Code: {ret_code_str})"
                 exc = ccxt.ExchangeError(error_message)
                 setattr(exc, 'code', ret_code_str) # Attach code for easier checking in except block
                 raise exc
            else:
                # If no specific error code structure is found (e.g., for other exchanges or older APIs)
                # and no exception was raised by CCXT, assume success.
                lg.info(f"{NEON_GREEN}Leverage set/confirmed for {market_id} to {leverage}x (No specific error code in response).{RESET}")
                return True

        # --- Error Handling with Retries ---
        except ccxt.ExchangeError as e:
            last_exception = e
            # Try to extract error code more reliably from the exception itself
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', '')) # Fallback check
            err_str_lower = str(e).lower()
            lg.error(f"{NEON_RED}Exchange error setting leverage ({market_id}): {e} (Code: {err_code_str}){RESET}")

            # Check for non-retryable conditions based on code or message content
            if err_code_str == '110045' or "leverage not modified" in err_str_lower:
                lg.info(f"{NEON_YELLOW}Leverage already set (confirmed via error message). Treating as success.{RESET}")
                return True # Already set is considered success

            # List of known fatal/non-retryable Bybit error codes for leverage setting
            fatal_codes = [
                '10001', # Parameter error
                '10004', # Sign check error (API key issue)
                '110009', # Cannot set leverage under Isolated margin mode with position or active order
                '110013', # Parameter '{0}' is invalid
                '110028', # The leverage cannot be greater than the maximum leverage of the risk limit tier
                '110043', # Set leverage not modified
                '110044', # Leverage can not be greater than specification leverage
                '110055', # Cannot set leverage when position mode is Portfolio Margin
                '3400045', # Position exists, cannot modify leverage under Isolated Margin mode
                # Add more codes as identified...
            ]
            fatal_messages = [
                "margin mode", "position exists", "risk limit", "parameter error",
                "insufficient available balance", "invalid leverage value"
            ]

            # Check if the error code or message indicates a non-retryable issue
            if err_code_str in fatal_codes or any(msg in err_str_lower for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE leverage error ({symbol}). Aborting leverage set.{RESET}")
                return False # Fatal error, do not retry

            # If error is potentially retryable and retries remain
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached for ExchangeError setting leverage ({symbol}).{RESET}")
                return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting leverage ({market_id}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached for NetworkError setting leverage ({symbol}).{RESET}")
                return False

        except ccxt.AuthenticationError as e:
             # Authentication errors are critical and non-retryable
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication error setting leverage ({symbol}): {e}. Stopping.{RESET}")
             return False # Fatal error
        except Exception as e:
            # Handle any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting leverage ({market_id}): {e}{RESET}", exc_info=True)
            return False # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to set leverage for {market_id} to {leverage}x after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False

def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on risk parameters, balance, and market constraints.

    Uses high-precision Decimal for all financial calculations. Handles differences between
    linear and inverse contracts. Applies market precision (amount step) and limits (min/max amount, min/max cost)
    to the calculated size. Rounds the final size DOWN to the nearest valid step size for safety, then performs
    a final check to ensure minimum cost is met (potentially bumping size up by one step if needed and possible).

    Args:
        balance (Decimal): Available trading balance (in quote currency, e.g., USDT). Must be positive.
        risk_per_trade (float): Fraction of balance to risk (e.g., 0.01 for 1%). Must be > 0 and <= 1.
        initial_stop_loss_price (Decimal): The calculated initial stop loss price. Must be positive and different from entry price.
        entry_price (Decimal): The intended entry price (or current price if using market order). Must be positive.
        market_info (MarketInfo): The standardized `MarketInfo` dictionary containing precision, limits, contract type, contract size, etc.
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object (used only for logging market details if needed).
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The calculated position size as a positive Decimal, adjusted for market rules (units are contracts for derivatives, base currency for spot),
                           or None if sizing is not possible due to invalid inputs, insufficient balance for minimum size,
                           inability to meet market limits, or calculation errors.
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE') # e.g., USDT
    base_currency = market_info.get('base', 'BASE')   # e.g., BTC
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('is_inverse', False)
    # Determine the unit of the position size (Contracts for derivatives, Base currency units for Spot)
    size_unit = "Contracts" if is_contract else base_currency

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or balance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive balance provided: {balance}.")
        return None
    try:
        # Convert risk percentage to Decimal for calculation
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
            # Risk must be strictly positive and less than or equal to 100%
            raise ValueError("Risk per trade must be > 0.0 and <= 1.0.")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial stop loss price ({initial_stop_loss_price}) must be a positive Decimal.")
        return None
    if not isinstance(entry_price, Decimal) or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Entry price ({entry_price}) must be a positive Decimal.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): Stop loss price cannot be equal to entry price.")
        return None

    # --- Extract Market Constraints (using pre-parsed Decimal values from MarketInfo) ---
    try:
        # These should have been validated by get_market_info, but check again for robustness
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal'] # Needed for cost estimation/validation
        min_amount = market_info['min_amount_decimal'] # Minimum order size (can be None or 0)
        max_amount = market_info['max_amount_decimal'] # Maximum order size (can be None)
        min_cost = market_info['min_cost_decimal']     # Minimum order cost (can be None or 0)
        max_cost = market_info['max_cost_decimal']     # Maximum order cost (can be None)
        contract_size = market_info['contract_size_decimal'] # Size of 1 contract (Decimal, defaults to 1)

        # Ensure essential constraints are valid positive Decimals
        if amount_step is None or amount_step <= 0: raise ValueError(f"Invalid amount precision step: {amount_step}")
        if price_step is None or price_step <= 0: raise ValueError(f"Invalid price precision step: {price_step}")
        if contract_size <= Decimal('0'): raise ValueError(f"Invalid contract size: {contract_size}")

        # Set effective limits for easier comparison (use 0 for min, infinity for max if None)
        min_amount_eff = min_amount if min_amount is not None else Decimal('0')
        max_amount_eff = max_amount if max_amount is not None else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None else Decimal('0')
        max_cost_eff = max_cost if max_cost is not None else Decimal('inf')

        lg.debug(f"  Market Constraints ({symbol}): AmtStep={amount_step.normalize()}, PriceStep={price_step.normalize()}, "
                 f"Min/Max Amt={min_amount_eff.normalize()}/{max_amount_eff.normalize()}, "
                 f"Min/Max Cost={min_cost_eff.normalize()}/{max_cost_eff.normalize()}, "
                 f"Contract Size={contract_size.normalize()}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error accessing or validating market details from MarketInfo: {e}")
        lg.debug(f"  MarketInfo used for sizing: {market_info}")
        return None

    # --- Calculate Risk Amount and Stop Loss Distance ---
    # Amount of quote currency to risk on this trade
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN) # Quantize risk amount early for precision
    # Distance between entry and stop loss in price units
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance <= Decimal('0'): # Should be caught by earlier check, but verify
        lg.error(f"Sizing failed ({symbol}): Calculated stop loss distance is zero or negative.")
        return None

    lg.info(f"  Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"  Risk Percentage: {risk_decimal:.2%}")
    lg.info(f"  Risk Amount: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"  Entry Price: {entry_price.normalize()}")
    lg.info(f"  Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"  Stop Loss Distance: {stop_loss_distance.normalize()}")
    lg.info(f"  Contract Type: {market_info['contract_type_str']}")

    # --- Calculate Initial Position Size (based on risk, SL distance, and contract type) ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear Contracts (e.g., BTC/USDT) or Spot markets
            # For Linear/Spot: Size = Risk Amount / (SL Distance * Contract Size)
            # Value change per unit (contract or base currency unit) = Price distance * Contract Size
            value_change_per_unit = stop_loss_distance * contract_size
            if value_change_per_unit <= Decimal('1e-18'): # Avoid division by zero/negligible value
                 lg.error(f"Sizing failed ({symbol}, Linear/Spot): Calculated value change per unit is near zero. Check SL distance and contract size.")
                 return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Size Calc: Risk {risk_amount_quote} / (SL Dist {stop_loss_distance} * ContrSize {contract_size}) = {calculated_size}")

        else: # Inverse Contracts (e.g., BTC/USD settled in BTC)
            # For Inverse: Size = Risk Amount / (Contract Size * |(1 / Entry) - (1 / SL)|)
            # Risk per contract (in quote currency) = Contract Size * abs((1 / Entry Price) - (1 / SL Price))
            if entry_price <= 0 or initial_stop_loss_price <= 0: # Should be caught earlier, but double check
                 lg.error(f"Sizing failed ({symbol}, Inverse): Entry or SL price is zero/negative.")
                 return None
            # Calculate the inverse factor carefully
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor <= Decimal('1e-18'): # Avoid division by negligible value
                 lg.error(f"Sizing failed ({symbol}, Inverse): Calculated inverse factor is near zero. Check entry/SL prices.")
                 return None
            risk_per_contract = contract_size * inverse_factor
            if risk_per_contract <= Decimal('1e-18'): # Avoid division by zero/negligible value
                 lg.error(f"Sizing failed ({symbol}, Inverse): Calculated risk per contract is near zero.")
                 return None
            calculated_size = risk_amount_quote / risk_per_contract
            lg.debug(f"  Inverse Size Calc: Risk {risk_amount_quote} / (ContrSize {contract_size} * InvFactor {inverse_factor}) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as calc_err:
        lg.error(f"Sizing failed ({symbol}): Calculation error occurred: {calc_err}.")
        return None

    if calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Initial calculated size is zero or negative ({calculated_size.normalize()}). Check risk/balance/SL distance.")
        return None
    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Apply Market Limits and Precision to the calculated size ---
    adjusted_size = calculated_size # Start with the risk-based size

    # Helper function to estimate order cost based on size, price, and contract type
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        """Estimates order cost in quote currency."""
        if not isinstance(size, Decimal) or not isinstance(price, Decimal) or price <= 0 or size <= 0:
            return None
        try:
             # Cost = Size * Price * ContractSize (for Linear/Spot)
             # Cost = Size * ContractSize / Price (for Inverse)
             if not is_inverse:
                 cost = size * price * contract_size
             else:
                 cost = (size * contract_size) / price
             # Quantize cost to a reasonable precision (e.g., 8 decimal places) for comparison
             return cost.quantize(Decimal('1e-8'), ROUND_UP)
        except (InvalidOperation, OverflowError, ZeroDivisionError):
            return None

    # 1. Apply Amount Limits (Min/Max Size)
    if min_amount_eff > 0 and adjusted_size < min_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Calculated size {adjusted_size.normalize()} is below minimum amount {min_amount_eff.normalize()}. Adjusting UP to minimum.{RESET}")
        adjusted_size = min_amount_eff
    if adjusted_size > max_amount_eff: # max_amount_eff is Decimal('inf') if no max limit
        lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Calculated size {adjusted_size.normalize()} exceeds maximum amount {max_amount_eff.normalize()}. Adjusting DOWN to maximum.{RESET}")
        adjusted_size = max_amount_eff
    lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 2. Apply Cost Limits (Min/Max Order Value in Quote Currency)
    cost_adjustment_made = False
    estimated_current_cost = estimate_cost(adjusted_size, entry_price)
    if estimated_current_cost is not None:
        lg.debug(f"  Estimated Cost (after amount limits adjustment, {symbol}): {estimated_current_cost.normalize()} {quote_currency}")

        # Check Minimum Cost
        if estimated_current_cost < min_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Estimated cost {estimated_current_cost.normalize()} is below minimum cost {min_cost_eff.normalize()}. Attempting to increase size.{RESET}")
            cost_adjustment_made = True
            try:
                # Calculate the size required to meet the minimum cost
                if not is_inverse:
                    # Req Size = Min Cost / (Entry Price * Contract Size)
                    if entry_price <= 0 or contract_size <= 0: raise ValueError("Invalid price/contract size for linear cost calc")
                    required_size_for_min_cost = min_cost_eff / (entry_price * contract_size)
                else:
                    # Req Size = (Min Cost * Entry Price) / Contract Size
                    if contract_size <= 0: raise ValueError("Invalid contract size for inverse cost calc")
                    required_size_for_min_cost = (min_cost_eff * entry_price) / contract_size

                if required_size_for_min_cost <= 0: raise ValueError("Calculated required size for min cost is zero/negative.")
                lg.info(f"  Size required to meet min cost ({symbol}): {required_size_for_min_cost.normalize()} {size_unit}")

                # Check if this required size exceeds the maximum amount limit
                if required_size_for_min_cost > max_amount_eff:
                    lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet minimum cost ({min_cost_eff.normalize()}) without exceeding maximum amount limit ({max_amount_eff.normalize()}).{RESET}")
                    return None

                # Adjust size up to the required size (or keep original min amount if that was larger)
                adjusted_size = max(min_amount_eff, required_size_for_min_cost)

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating size required for minimum cost: {cost_calc_err}.{RESET}")
                return None

        # Check Maximum Cost
        elif estimated_current_cost > max_cost_eff: # max_cost_eff is Decimal('inf') if no max limit
            lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Estimated cost {estimated_current_cost.normalize()} exceeds maximum cost {max_cost_eff.normalize()}. Reducing size.{RESET}")
            cost_adjustment_made = True
            try:
                # Calculate the maximum size allowed by the maximum cost
                if not is_inverse:
                    # Max Size = Max Cost / (Entry Price * Contract Size)
                    if entry_price <= 0 or contract_size <= 0: raise ValueError("Invalid price/contract size for linear cost calc")
                    max_size_for_max_cost = max_cost_eff / (entry_price * contract_size)
                else:
                    # Max Size = (Max Cost * Entry Price) / Contract Size
                    if contract_size <= 0: raise ValueError("Invalid contract size for inverse cost calc")
                    max_size_for_max_cost = (max_cost_eff * entry_price) / contract_size

                if max_size_for_max_cost <= 0: raise ValueError("Calculated max size for max cost is zero/negative.")
                lg.info(f"  Maximum size allowed by max cost ({symbol}): {max_size_for_max_cost.normalize()} {size_unit}")

                # Adjust size down: take the minimum of the current adjusted size and the max size allowed by cost.
                # Also ensure it doesn't go below the minimum amount limit.
                adjusted_size = max(min_amount_eff, min(adjusted_size, max_size_for_max_cost))

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
                lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calculating max size allowed by maximum cost: {cost_calc_err}.{RESET}")
                return None
    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'):
        # If cost couldn't be estimated but limits exist, warn the user.
        lg.warning(f"Could not estimate order cost ({symbol}) to check against cost limits [{min_cost_eff.normalize()} - {max_cost_eff.normalize()}]. Proceeding without cost limit check.")

    if cost_adjustment_made:
        lg.info(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    # 3. Apply Amount Precision (Rounding DOWN to the nearest valid step size)
    # Rounding down is generally safer to avoid exceeding balance or limits slightly due to precision.
    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step size must be positive.")
        # Manual Rounding Down using Decimal quantize: floor(value / step) * step
        # Divide by step, round down to integer, then multiply back by step.
        quantized_steps = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN)
        final_size = quantized_steps * amount_step

        if final_size != adjusted_size:
            lg.info(f"Applied amount precision ({symbol}, Step: {amount_step.normalize()}, Rounded DOWN): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
        else:
             lg.debug(f"Size {adjusted_size.normalize()} already matches amount precision step {amount_step.normalize()} ({symbol}).")

    except (InvalidOperation, ValueError, TypeError) as fmt_err:
        # If formatting fails, log error but potentially continue with unrounded size (might fail order placement)
        lg.error(f"{NEON_RED}Error applying amount precision ({symbol}): {fmt_err}. Using unrounded size: {final_size.normalize()}{RESET}")
        # Keep final_size as it was before this step

    # --- Final Validation Checks after Precision Application ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final calculated size is zero or negative ({final_size.normalize()}) after applying precision/limits.{RESET}")
        return None

    # Check Min Amount again after rounding down (crucial)
    if final_size < min_amount_eff:
        # This can happen if the only valid size after rounding down is below the minimum.
        lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} is below minimum amount {min_amount_eff.normalize()} after applying precision.{RESET}")
        return None

    # Check Max Amount again (should generally be okay if rounded down, but check defensively)
    if final_size > max_amount_eff:
         lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} exceeds maximum amount {max_amount_eff.normalize()} after applying precision (unexpected).{RESET}")
         return None

    # Final check on cost after precision (especially important if min cost is required)
    final_estimated_cost = estimate_cost(final_size, entry_price)
    if final_estimated_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}, after precision): {final_estimated_cost.normalize()} {quote_currency}")

        # Check if final cost is below min cost after rounding down size
        if final_estimated_cost < min_cost_eff:
             lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Final estimated cost {final_estimated_cost.normalize()} is below minimum cost {min_cost_eff.normalize()} after rounding size down.{RESET}")
             # Attempt to bump size up by exactly one step if possible and if it meets min cost without exceeding other limits
             try:
                 next_step_size = final_size + amount_step
                 next_step_cost = estimate_cost(next_step_size, entry_price)

                 if next_step_cost is not None:
                     # Check if bumping up is valid:
                     # - Meets min cost
                     # - Does not exceed max amount
                     # - Does not exceed max cost
                     can_bump_up = (next_step_cost >= min_cost_eff) and \
                                   (next_step_size <= max_amount_eff) and \
                                   (next_step_cost <= max_cost_eff)

                     if can_bump_up:
                         lg.info(f"{NEON_YELLOW}Bumping final size ({symbol}) up by one step ({amount_step.normalize()}) to {next_step_size.normalize()} to meet minimum cost requirement.{RESET}")
                         final_size = next_step_size # Update final size
                         # Log the new estimated cost after bumping up
                         final_cost_after_bump = estimate_cost(final_size, entry_price)
                         lg.debug(f"  Final Estimated Cost after bump ({symbol}): {final_cost_after_bump.normalize() if final_cost_after_bump else 'N/A'}")
                     else:
                         # Cannot bump up due to other limits
                         lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet minimum cost. Bumping size by one step would violate other limits (Max Amount: {max_amount_eff.normalize()}, Max Cost: {max_cost_eff.normalize()}).{RESET}")
                         return None
                 else:
                      # Should not happen if estimate_cost worked before
                      lg.error(f"{NEON_RED}Sizing failed ({symbol}): Could not estimate cost for bumped size check.{RESET}")
                      return None
             except Exception as bump_err:
                 lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error occurred while trying to bump size for minimum cost: {bump_err}.{RESET}")
                 return None

        # Check if final cost exceeds max cost (should be rare if rounded down, but check)
        elif final_estimated_cost > max_cost_eff:
            lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final estimated cost {final_estimated_cost.normalize()} exceeds maximum cost {max_cost_eff.normalize()} after applying precision (unexpected).{RESET}")
            return None
    elif min_cost_eff > 0:
         # Warn if min cost exists but couldn't be checked
         lg.warning(f"Could not perform final cost check ({symbol}) against minimum cost ({min_cost_eff.normalize()}) after precision adjustment.")

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    if final_estimated_cost: lg.info(f"    Estimated Cost: ~{final_estimated_cost.normalize()} {quote_currency}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size

def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """
    Cancels an open order by its ID using `exchange.cancel_order`.

    - Handles Bybit V5 specifics: May require 'category' and 'symbol' in params even when cancelling by ID.
    - Treats `ccxt.OrderNotFound` error as success, assuming the order was already cancelled or filled.
    - Includes retry logic for network/exchange errors and rate limits.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        order_id (str): The unique ID of the order to cancel.
        symbol (str): The symbol the order belongs to (required by some exchanges/APIs, including Bybit V5).
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        bool: True if the order was cancelled successfully or was confirmed as not found, False otherwise.
    """
    lg = logger
    attempts = 0
    last_exception = None
    lg.info(f"Attempting to cancel order ID: {order_id} for symbol {symbol}...")

    # --- Determine Market ID and Category (potentially needed for Bybit V5 cancel) ---
    market_id = symbol # Default
    category = 'spot'  # Default
    params = {}
    if 'bybit' in exchange.id.lower():
        try:
            market = exchange.market(symbol)
            market_id = market['id'] # Use exchange-specific ID
            category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
            # Bybit V5 cancelOrder often requires category and symbol even with orderId
            params = {'category': category, 'symbol': market_id}
            lg.debug(f"Using Bybit V5 cancelOrder params: {params}")
        except Exception as e:
            lg.warning(f"Could not determine category/market_id for cancel order {order_id} ({symbol}): {e}. Using defaults, cancellation might fail.")
            # Use defaults, but API call might fail if params are strictly required

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1} for ID {order_id} ({symbol})...")
            # --- Execute cancel_order call ---
            # Pass standard symbol to CCXT method, exchange-specific needs handled via params
            exchange.cancel_order(order_id, symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully cancelled order ID: {order_id} for {symbol}.{RESET}")
            return True

        # --- Error Handling ---
        except ccxt.OrderNotFound:
            # If the order doesn't exist on the exchange, it's effectively cancelled or already filled.
            lg.warning(f"{NEON_YELLOW}Order ID {order_id} ({symbol}) not found on exchange. Already cancelled or filled? Treating as success.{RESET}")
            return True # Consider "not found" as a successful outcome for cancellation intent
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            # Retry on network issues
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            # Wait and retry on rate limits
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 2 # Shorter wait for cancel might be okay
            lg.warning(f"{NEON_YELLOW}Rate limit cancelling order {order_id} ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.ExchangeError as e:
            # Log other exchange errors and allow retry
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}{RESET}")
            # Assume most exchange errors might be temporary for cancel, unless specific codes indicate otherwise
            # Add checks for known non-retryable cancel errors if necessary
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error cancelling order ({symbol}): {e}. Stopping.{RESET}")
            return False # Fatal error
        except Exception as e:
            # Handle unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True)
            return False # Non-retryable

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to cancel order ID {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False

def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: MarketInfo,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Places a market order (buy or sell) using `exchange.create_order`.

    - Maps trade signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT") to appropriate order sides ("buy", "sell").
    - Handles Bybit V5 specific parameters: 'category', 'positionIdx' (for one-way mode), 'reduceOnly',
      and sets 'timeInForce' to 'IOC' for reduce-only market orders.
    - Includes retry logic for network/exchange errors and rate limits.
    - Identifies and handles common non-retryable order errors (e.g., insufficient funds, invalid parameters,
      risk limit issues) with informative log messages and hints.
    - Logs order placement attempts and results clearly.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        trade_signal (str): The signal driving the trade ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size (Decimal): The calculated position size (must be a positive Decimal). Units depend on market type.
        market_info (MarketInfo): The standardized `MarketInfo` dictionary for the symbol.
        logger (logging.Logger): The logger instance for status messages.
        reduce_only (bool): Set to True for closing orders to ensure they only reduce or close an existing position.
        params (Optional[Dict]): Optional dictionary of additional parameters to pass to the underlying
                                 `create_order` call's `params` argument (for exchange-specific features).

    Returns:
        Optional[Dict]: The raw order dictionary returned by CCXT upon successful placement, or None if the
                        order fails after retries or encounters a fatal, non-retryable error.
    """
    lg = logger
    # Map the strategy signal to the required CCXT order side ('buy' or 'sell')
    side_map = {
        "BUY": "buy",      # Entering long
        "SELL": "sell",     # Entering short
        "EXIT_SHORT": "buy", # Closing short requires a buy order
        "EXIT_LONG": "sell"  # Closing long requires a sell order
    }
    side = side_map.get(trade_signal.upper())

    # --- Input Validation ---
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided to place_trade function for {symbol}.")
        return None
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'):
        lg.error(f"Invalid position size provided to place_trade for {symbol}: {position_size}. Must be a positive Decimal.")
        return None

    order_type = 'market' # This bot currently only uses market orders for simplicity
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base_currency # Determine unit for logging
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase" # For logging clarity
    market_id = market_info['id'] # Use the exchange-specific market ID for the API call

    # --- Prepare Order Arguments for CCXT ---
    # CCXT's create_order typically expects the amount as a float. Convert Decimal carefully.
    try:
         amount_float = float(position_size)
         # Add a check for extremely small values after float conversion
         if amount_float <= 1e-15: # Adjust threshold if necessary
              raise ValueError("Position size is negligible after conversion to float.")
    except (ValueError, TypeError, OverflowError) as float_err:
         lg.error(f"Failed to convert valid Decimal size {position_size.normalize()} ({symbol}) to float for API call: {float_err}")
         return None

    # Base arguments for create_order
    order_args: Dict[str, Any] = {
        'symbol': symbol, # Use standard symbol for CCXT method signature
        'type': order_type,
        'side': side,
        'amount': amount_float,
        # 'price': None, # Not needed for market orders
    }
    # Dictionary for exchange-specific parameters nested under 'params'
    order_params: Dict[str, Any] = {}

    # --- Add Bybit V5 Specific Parameters ---
    if 'bybit' in exchange.id.lower() and is_contract:
        try:
            category = market_info.get('contract_type_str', 'Linear').lower() # linear or inverse
            if category not in ['linear', 'inverse']:
                 raise ValueError(f"Invalid contract category '{category}' for Bybit order placement.")
            # Common parameters for Bybit V5 contract orders
            order_params = {
                'category': category,
                # Use positionIdx=0 for One-Way Mode. For Hedge Mode, different values (1 for long, 2 for short) would be needed.
                # This bot assumes One-Way Mode. Ensure Bybit account is set accordingly.
                'positionIdx': 0
            }
            # Add reduceOnly flag if requested
            if reduce_only:
                order_params['reduceOnly'] = True
                # For reduceOnly market orders, using IOC (Immediate Or Cancel) is often recommended
                # to prevent the order from resting if the market moves away quickly, potentially
                # failing the reduce-only check later. FOK (Fill Or Kill) could also be used.
                order_params['timeInForce'] = 'IOC'
            lg.debug(f"Using Bybit V5 order params ({symbol}): {order_params}")
        except Exception as e:
            lg.error(f"Failed to set Bybit V5 specific parameters for order ({symbol}): {e}. Order might fail.")
            # Proceed cautiously without params if setting failed, but expect potential errors

    # Merge any additional custom parameters provided by the caller into order_params
    if params and isinstance(params, dict):
        order_params.update(params)
        lg.debug(f"Added custom params to order ({symbol}): {params}")

    # Add the collected exchange-specific parameters to the main order arguments if any exist
    if order_params:
        order_args['params'] = order_params

    # Log the trade attempt clearly
    lg.warning(f"{BRIGHT}===> PLACING {action_desc} | {side.upper()} {order_type.upper()} Order | {symbol} | Size: {position_size.normalize()} {size_unit} <==={RESET}")
    if order_params: lg.debug(f"  with Params ({symbol}): {order_params}")

    # --- Execute Order with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1})...")
            # Place the order using CCXT's unified method
            order_result = exchange.create_order(**order_args)

            # --- Log Success ---
            order_id = order_result.get('id', 'N/A')
            status = order_result.get('status', 'N/A') # e.g., 'open', 'closed', 'canceled'
            # Safely format potential Decimal/float/str values from result for logging
            avg_price_dec = _safe_market_decimal(order_result.get('average'), 'order.average', allow_zero=True, allow_negative=False)
            filled_dec = _safe_market_decimal(order_result.get('filled'), 'order.filled', allow_zero=True, allow_negative=False) # Allow zero filled initially for market orders

            log_msg = (
                f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET} ID: {order_id}, Status: {status}"
            )
            # Add details if available in the response
            if avg_price_dec: log_msg += f", AvgFillPrice: ~{avg_price_dec.normalize()}"
            if filled_dec is not None: log_msg += f", FilledAmount: {filled_dec.normalize()}"
            lg.info(log_msg)
            lg.debug(f"Full order result ({symbol}): {order_result}")
            return order_result # Return the successful order details dictionary

        # --- Error Handling with Retries ---
        except ccxt.InsufficientFunds as e:
            # Non-retryable error: Not enough balance for the order
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Insufficient Funds. {e}{RESET}")
            # Log current balance for context
            try:
                current_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                lg.error(f"  Current {QUOTE_CURRENCY} Balance: {current_balance.normalize() if current_balance else 'Fetch Failed'}")
            except: pass # Avoid errors in error handling
            return None # Non-retryable
        except ccxt.InvalidOrder as e:
            # Non-retryable error: Order parameters rejected by the exchange (size, price, limits, etc.)
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Invalid Order Parameters. {e}{RESET}")
            lg.error(f"  Order Arguments Sent: {order_args}")
            # Provide hints based on common causes by checking error message and market info
            err_lower = str(e).lower()
            min_amt_str = market_info.get('min_amount_decimal', 'N/A')
            min_cost_str = market_info.get('min_cost_decimal', 'N/A')
            amt_step_str = market_info.get('amount_precision_step_decimal', 'N/A')
            max_amt_str = market_info.get('max_amount_decimal', 'N/A')
            max_cost_str = market_info.get('max_cost_decimal', 'N/A')

            if "minimum order" in err_lower or "min order" in err_lower or "too small" in err_lower or "lower than limit" in err_lower:
                 lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) or estimated cost against market minimums (MinAmt: {min_amt_str}, MinCost: {min_cost_str}).")
            elif "precision" in err_lower or "lot size" in err_lower or "step size" in err_lower or "multiple of" in err_lower:
                 lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) against the required amount step/precision ({amt_step_str}).")
            elif "exceed maximum" in err_lower or "max order" in err_lower or "too large" in err_lower or "greater than limit" in err_lower:
                 lg.error(f"  >> Hint: Check order size ({position_size.normalize()}) or estimated cost against market maximums (MaxAmt: {max_amt_str}, MaxCost: {max_cost_str}).")
            elif "reduce only" in err_lower or "reduce-only" in err_lower:
                 lg.error(f"  >> Hint: Reduce-only order failed. Ensure there's an open position in the correct direction and the order size doesn't increase the position.")
            elif "position idx not match" in err_lower:
                 lg.error(f"  >> Hint: Check Bybit position mode (One-Way vs Hedge) and ensure it matches bot's assumption (currently One-Way, positionIdx=0).")
            return None # Non-retryable
        except ccxt.ExchangeError as e:
            # Handle general exchange errors, potentially retryable
            last_exception = e
            # Try to extract error code for better logging/debugging
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', '')) # Fallback
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Exchange Error. {e} (Code: {err_code_str}){RESET}")

            # Check for known fatal/non-retryable Bybit error codes related to order placement
            # These codes often indicate configuration issues, risk limit problems, or state mismatches.
            fatal_order_codes = [
                '10001', # Parameter error
                '10004', # Sign check error (API key issue)
                '110007', # Abnormal trading activity detected (risk control)
                '110013', # Parameter '{0}' is invalid
                '110014', # Order quantity exceeds the lower limit
                '110017', # Order quantity exceeds the upper limit
                '110025', # Position idx not match position mode (check One-Way/Hedge mode)
                '110040', # Order amount is lower than the minimum order amount (similar to InvalidOrder)
                '30086',  # Order cost exceeds risk limit (Bybit specific risk control)
                '3303001',# Insufficient available balance (redundant with InsufficientFunds, but check code)
                '3303005',# Position size exceeds maximum limit allowed
                '3400060',# Order would trigger immediate liquidation
                '3400088',# Reduce-only order condition not met
                # Add more known fatal codes here...
            ]
            fatal_messages = [
                "invalid parameter", "precision error", "exceed limit", "risk limit",
                "invalid symbol", "reduce only check failed", "lot size error",
                "insufficient available balance", "leverage exceed limit", "trigger liquidation"
            ]

            # Check if the error code or message indicates a non-retryable issue
            err_str_lower = str(e).lower()
            if err_code_str in fatal_order_codes or any(msg in err_str_lower for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE order placement error detected ({symbol}). Check parameters, balance, risk limits, or position state.{RESET}")
                return None # Non-retryable

            # If error seems potentially temporary and retries remain
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries reached for ExchangeError placing order ({symbol}).{RESET}")
                 return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            # Retry on network issues
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached for NetworkError placing order ({symbol}).{RESET}")
                return None

        except ccxt.RateLimitExceeded as e:
            # Wait and retry on rate limits
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit placing order ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
             # Authentication errors are critical and non-retryable
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication error placing order ({symbol}): {e}. Stopping.{RESET}")
             return None # Fatal error
        except Exception as e:
            # Handle any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order ({symbol}): {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter (only if not a rate limit wait) and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to place {action_desc} order ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """
    Internal helper: Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL)
    for an existing position using Bybit's V5 private API endpoint `/v5/position/set-trading-stop`.

    **WARNING:** This function uses a direct, non-standard CCXT private API call (`private_post`).
    It is specific to Bybit V5 and relies on the exact endpoint and parameter names documented by Bybit.
    Changes in the Bybit API may break this function.

    - Handles parameter validation: ensures SL/TP/Activation prices are valid relative to entry price and side.
    - Formats price parameters to the market's required precision using `_format_price`.
    - Understands Bybit's logic: Setting an active TSL (`trailingStop` > 0) overrides any fixed `stopLoss` value sent in the same request.
    - Allows clearing protection levels by passing Decimal('0') or 0.
    - Includes retry logic for common API errors.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object with valid credentials.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        market_info (MarketInfo): The standardized MarketInfo dictionary for the symbol.
        position_info (PositionInfo): The standardized PositionInfo dictionary for the open position.
        logger (logging.Logger): The logger instance for status messages.
        stop_loss_price (Optional[Decimal]): Desired fixed SL price. Pass Decimal('0') or 0 to clear/remove existing SL (if TSL is not being set).
        take_profit_price (Optional[Decimal]): Desired fixed TP price. Pass Decimal('0') or 0 to clear/remove existing TP.
        trailing_stop_distance (Optional[Decimal]): Desired TSL distance (in price units, e.g., 100 for 100 USDT distance). Pass Decimal('0') or 0 to clear/remove TSL. Must be positive if setting TSL.
        tsl_activation_price (Optional[Decimal]): Price at which the TSL should activate. Required if `trailing_stop_distance` > 0.

    Returns:
        bool: True if the protection was set/updated successfully via API (including cases where no change was needed according to the API response).
              False if validation fails, the API call fails after retries, or a critical non-retryable error occurs.
    """
    lg = logger
    # Bybit V5 specific endpoint for setting SL/TP/TSL
    endpoint = '/v5/position/set-trading-stop'
    lg.debug(f"Preparing to call Bybit endpoint: {endpoint} for {symbol}")

    # --- Input and State Validation ---
    if not market_info.get('is_contract', False):
        lg.error(f"Protection setting failed ({symbol}): Cannot set SL/TP/TSL on non-contract markets.")
        return False
    if not position_info:
        lg.error(f"Protection setting failed ({symbol}): Missing current position information.")
        return False

    # Extract necessary info from position and market data, ensuring valid types
    pos_side = position_info.get('side')
    entry_price_any = position_info.get('entryPrice') # Could be Decimal or None initially

    if pos_side not in ['long', 'short']:
        lg.error(f"Protection setting failed ({symbol}): Invalid or missing position side ('{pos_side}').")
        return False
    try:
        if entry_price_any is None: raise ValueError("Missing entry price in position info.")
        entry_price = Decimal(str(entry_price_any)) # Ensure entry price is Decimal
        if not entry_price.is_finite() or entry_price <= 0: raise ValueError("Entry price must be a positive finite number.")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Protection setting failed ({symbol}): Invalid or missing entry price ('{entry_price_any}'): {e}")
        return False
    try:
        # Get price precision step (tick size) for formatting and validation
        price_tick = market_info['price_precision_step_decimal']
        if price_tick is None or not price_tick.is_finite() or price_tick <= 0:
             raise ValueError("Invalid price tick size in market info.")
    except (KeyError, ValueError, TypeError) as e:
         lg.error(f"Protection setting failed ({symbol}): Could not get valid price precision step: {e}")
         return False

    # --- Prepare Parameters for the API Call ---
    params_to_set: Dict[str, Any] = {} # Dictionary to hold parameters for the API request
    log_parts: List[str] = [f"{BRIGHT}Preparing protection update ({symbol} {pos_side.upper()} @ Entry: {entry_price.normalize()}):{RESET}"]
    any_protection_requested = False # Flag to track if any valid protection level was requested
    set_tsl_active = False # Flag to indicate if an active TSL (distance > 0) is being set

    try:
        # Helper to format price parameter to exchange precision string using the global helper
        # Returns the formatted string, or None if formatting fails or input is invalid
        def format_param(price_decimal: Optional[Union[Decimal, int, float]], param_name: str) -> Optional[str]:
            """Formats price to string for API, handles None/0, returns None on failure."""
            if price_decimal is None: return None
            try:
                d_price = Decimal(str(price_decimal))
                # Allow "0" string to be sent to clear protection levels
                if d_price.is_zero(): return "0"
                # Use the global formatter which handles precision and positivity check
                formatted = _format_price(exchange, market_info['symbol'], d_price)
                if formatted:
                    return formatted
                else:
                    lg.error(f"Failed to format {param_name} ({symbol}): Input {d_price.normalize()}")
                    return None
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.error(f"Error converting {param_name} ({symbol}) value '{price_decimal}' for formatting: {e}")
                return None

        # --- Trailing Stop Loss (TSL) Parameters ---
        # Bybit requires both `trailingStop` (distance) and `activePrice` if TSL distance > 0
        if trailing_stop_distance is not None:
            any_protection_requested = True
            try:
                tsl_dist_dec = Decimal(str(trailing_stop_distance))
                if tsl_dist_dec > 0: # Setting an active TSL
                    # Ensure distance is at least one price tick
                    min_valid_distance = max(tsl_dist_dec, price_tick)

                    # Activation price is mandatory for active TSL
                    if tsl_activation_price is None:
                        raise ValueError("TSL activation price is required when setting TSL distance > 0.")
                    tsl_act_dec = Decimal(str(tsl_activation_price))
                    if tsl_act_dec <= 0:
                        raise ValueError("TSL activation price must be positive.")

                    # Validate activation price makes sense relative to entry (must be beyond entry in profit direction)
                    is_valid_activation = (pos_side == 'long' and tsl_act_dec > entry_price) or \
                                          (pos_side == 'short' and tsl_act_dec < entry_price)
                    if not is_valid_activation:
                        raise ValueError(f"TSL activation price {tsl_act_dec.normalize()} is not valid relative to entry {entry_price.normalize()} for {pos_side} position.")

                    # Format parameters for API
                    fmt_distance = format_param(min_valid_distance, "TSL Distance")
                    fmt_activation = format_param(tsl_act_dec, "TSL Activation")

                    if fmt_distance and fmt_activation:
                        params_to_set['trailingStop'] = fmt_distance
                        params_to_set['activePrice'] = fmt_activation
                        log_parts.append(f"  - Setting TSL: Distance={fmt_distance}, ActivationPrice={fmt_activation}")
                        set_tsl_active = True # Mark TSL as being actively set
                    else:
                        # Formatting failed, cannot set TSL
                        lg.error(f"TSL setting failed ({symbol}): Could not format TSL parameters (Dist: {fmt_distance}, Act: {fmt_activation}).")

                elif tsl_dist_dec.is_zero(): # Clearing TSL
                    params_to_set['trailingStop'] = "0"
                    # Bybit documentation suggests clearing activation price might also be needed when clearing distance. Send "0" for safety.
                    params_to_set['activePrice'] = "0"
                    log_parts.append("  - Clearing TSL (Distance and Activation Price set to '0')")
                    set_tsl_active = False # Ensure flag is false if clearing
                else: # Negative distance is invalid
                     raise ValueError(f"Invalid negative TSL distance provided: {tsl_dist_dec.normalize()}")
            except (ValueError, InvalidOperation, TypeError) as tsl_err:
                 lg.error(f"TSL parameter validation failed ({symbol}): {tsl_err}")

        # --- Fixed Stop Loss (SL) Parameter ---
        # IMPORTANT: According to Bybit V5 docs, if `trailingStop` > 0 is sent, the `stopLoss` value is ignored by the API.
        # Therefore, only attempt to set a fixed SL if an active TSL is *not* being set in this same request.
        if not set_tsl_active:
            if stop_loss_price is not None:
                any_protection_requested = True
                try:
                    sl_dec = Decimal(str(stop_loss_price))
                    if sl_dec > 0: # Setting an active SL
                        # Validate SL price is on the correct side of the entry price
                        is_valid_sl = (pos_side == 'long' and sl_dec < entry_price) or \
                                      (pos_side == 'short' and sl_dec > entry_price)
                        if not is_valid_sl:
                            raise ValueError(f"Stop Loss price {sl_dec.normalize()} is invalid relative to entry {entry_price.normalize()} for {pos_side} position.")

                        fmt_sl = format_param(sl_dec, "Stop Loss")
                        if fmt_sl:
                            params_to_set['stopLoss'] = fmt_sl
                            log_parts.append(f"  - Setting Fixed SL: {fmt_sl}")
                        else:
                            lg.error(f"SL setting failed ({symbol}): Could not format SL price {sl_dec.normalize()}.")
                    elif sl_dec.is_zero(): # Clearing SL
                        # Only send "0" if SL field wasn't already populated (shouldn't be if set_tsl_active is False)
                        if 'stopLoss' not in params_to_set:
                             params_to_set['stopLoss'] = "0"
                             log_parts.append("  - Clearing Fixed SL (set to '0')")
                    # Negative SL price already handled/rejected by format_param
                except (ValueError, InvalidOperation, TypeError) as sl_err:
                    lg.error(f"SL parameter validation failed ({symbol}): {sl_err}")
        elif stop_loss_price is not None and Decimal(str(stop_loss_price)) > 0:
             # If user tried to set a fixed SL while also setting an active TSL, log a warning.
             lg.warning(f"Ignoring fixed SL request ({stop_loss_price}) because an active TSL is being set simultaneously ({symbol}). Bybit API prioritizes TSL.")

        # --- Fixed Take Profit (TP) Parameter ---
        # TP can usually be set independently or alongside SL/TSL.
        if take_profit_price is not None:
            any_protection_requested = True
            try:
                tp_dec = Decimal(str(take_profit_price))
                if tp_dec > 0: # Setting an active TP
                    # Validate TP price is on the correct side of the entry price (profit side)
                    is_valid_tp = (pos_side == 'long' and tp_dec > entry_price) or \
                                  (pos_side == 'short' and tp_dec < entry_price)
                    if not is_valid_tp:
                        raise ValueError(f"Take Profit price {tp_dec.normalize()} is invalid relative to entry {entry_price.normalize()} for {pos_side} position.")

                    fmt_tp = format_param(tp_dec, "Take Profit")
                    if fmt_tp:
                        params_to_set['takeProfit'] = fmt_tp
                        log_parts.append(f"  - Setting Fixed TP: {fmt_tp}")
                    else:
                        lg.error(f"TP setting failed ({symbol}): Could not format TP price {tp_dec.normalize()}.")
                elif tp_dec.is_zero(): # Clearing TP
                     if 'takeProfit' not in params_to_set: # Avoid overwriting if already set (unlikely here)
                         params_to_set['takeProfit'] = "0"
                         log_parts.append("  - Clearing Fixed TP (set to '0')")
                # Negative TP price already handled/rejected by format_param
            except (ValueError, InvalidOperation, TypeError) as tp_err:
                 lg.error(f"TP parameter validation failed ({symbol}): {tp_err}")

    except Exception as validation_err:
        lg.error(f"Unexpected error during protection parameter validation ({symbol}): {validation_err}", exc_info=True)
        return False

    # --- Check if any valid parameters were actually prepared for the API call ---
    if not params_to_set:
        if any_protection_requested:
            # If protection was requested but validation failed for all items
            lg.warning(f"{NEON_YELLOW}Protection update skipped ({symbol}): No valid parameters generated after validation/formatting. No API call made.{RESET}")
            return False # Return False because the requested action couldn't be fulfilled
        else:
            # If no protection levels were provided in the function call initially
            lg.debug(f"No protection changes requested or parameters provided ({symbol}). Skipping API call.")
            return True # Success, as no action was needed

    # --- Prepare Final API Parameters Dictionary ---
    category = market_info.get('contract_type_str', 'Linear').lower() # linear or inverse
    market_id = market_info['id'] # Exchange-specific symbol ID
    # Get position index (should typically be 0 for one-way mode assumed by this bot)
    position_idx = 0 # Default to one-way mode
    try:
        # Extract positionIdx from the raw position info if available
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
            position_idx = int(pos_idx_val)
        # Warn if using hedge mode index, as bot logic assumes one-way
        if position_idx != 0:
            lg.warning(f"Detected positionIdx={position_idx} for {symbol}. Ensure Bybit account is in One-Way mode or adjust logic if using Hedge Mode.")
    except (ValueError, TypeError):
        lg.warning(f"Could not parse positionIdx for {symbol}. Using default {position_idx}.")

    # Construct the final parameters dictionary for the API call
    final_api_params: Dict[str, Any] = {
        'category': category,
        'symbol': market_id,
        'positionIdx': position_idx # Specify position index (0 for one-way, 1/2 for hedge)
    }
    # Add the validated and formatted SL/TP/TSL parameters
    final_api_params.update(params_to_set)

    # Add default trigger/order type parameters (can be customized later if needed via function args)
    # These defaults match common usage for market SL/TP.
    final_api_params.update({
        'tpslMode': 'Full',         # Apply protection to the entire position ('Full') or partial ('Partial')
        'slTriggerBy': 'LastPrice', # Trigger SL based on Last Price ('MarkPrice', 'IndexPrice' also possible)
        'tpTriggerBy': 'LastPrice', # Trigger TP based on Last Price
        'slOrderType': 'Market',    # Use a Market order when SL is triggered ('Limit' also possible)
        'tpOrderType': 'Market',    # Use a Market order when TP is triggered
    })

    # Log the parameters being sent only if there are parameters to set
    lg.info("\n".join(log_parts)) # Log multi-line summary of what is being attempted
    lg.debug(f"  Final API params for {endpoint} ({symbol}): {final_api_params}")

    # --- Execute API Call with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing private_post {endpoint} ({symbol}, Attempt {attempts + 1})...")
            # Use exchange.private_post for endpoints not directly mapped by standard CCXT methods
            # This requires the endpoint path and the parameters dictionary.
            response = exchange.private_post(endpoint, params=final_api_params)
            lg.debug(f"Raw response from {endpoint} ({symbol}): {response}")

            # --- Check Bybit V5 Response Code ---
            # Successful response usually has retCode = 0
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown message')

            if ret_code == 0:
                 # Check message for "not modified" cases, which indicate success but no change made
                 no_change_msgs = ["not modified", "no need to modify", "parameter not change", "order is not modified", "same as the current"]
                 if any(m in ret_msg.lower() for m in no_change_msgs):
                     lg.info(f"{NEON_YELLOW}Protection parameters already set or no change needed for {symbol} (API Msg: '{ret_msg}').{RESET}")
                 else:
                     lg.info(f"{NEON_GREEN}Protection set/updated successfully for {symbol} (Code: 0).{RESET}")
                 return True # Success

            else:
                 # Log the specific Bybit error and raise ExchangeError for retry or handling
                 error_message = f"Bybit API error setting protection ({symbol}): {ret_msg} (Code: {ret_code})"
                 lg.error(f"{NEON_RED}{error_message}{RESET}")
                 # Attach code to exception if possible for easier checking in except block
                 exc = ccxt.ExchangeError(error_message)
                 setattr(exc, 'code', ret_code) # Set attribute dynamically
                 raise exc # Raise the error to trigger retry or specific handling

        # --- Standard CCXT Error Handling with Retries ---
        except ccxt.ExchangeError as e:
            last_exception = e
            # Try to extract error code more reliably from the exception
            err_code_str = ""
            match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE)
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', '')) # Fallback
            err_str_lower = str(e).lower()
            lg.warning(f"{NEON_YELLOW}Exchange error setting protection ({symbol}): {e} (Code: {err_code_str}). Retry {attempts + 1}...{RESET}")

            # Check for known fatal/non-retryable error codes/messages for set-trading-stop
            # These often relate to invalid parameters, prices, or position state.
            fatal_protect_codes = [
                '10001', # Parameter error
                '10002', # Request parameter error (often invalid format/value)
                '110013',# Parameter '{0}' is invalid
                '110036',# tp/sl price cannot be higher/lower than trigger price
                '110043',# Set tp/sl not modified (already handled as success, but check code)
                '110084',# The stop loss price cannot be greater than the trigger price {0}
                '110085',# The stop loss price cannot be less than the trigger price {0}
                '110086',# The take profit price cannot be less than the trigger price {0}
                '110103',# The take profit price cannot be greater than the trigger price {0}
                '110104',# The trigger price cannot be less than the current last price {0}
                '110110',# The trigger price cannot be greater than the current last price {0}
                '3400045',# Position exists, cannot modify... (less relevant here, more for leverage)
                '3400048',# The take profit price must be greater than the stop loss price
                '3400051',# The stop loss price is invalid
                '3400052',# The take profit price is invalid
                '3400070',# The trailing stop distance is invalid
                '3400071',# The activation price is invalid
                '3400072',# The activation price cannot be the same as the current price
                '3400073',# The trailing stop distance must be greater than 0
                # Add more specific codes if encountered during testing...
            ]
            fatal_messages = [
                "invalid parameter", "invalid price", "cannot be higher than", "cannot be lower than",
                "position status not normal", "precision error", "activation price invalid",
                "distance invalid", "cannot be the same", "price is out of range", "less than mark price"
            ]

            # Check if the error code or message indicates a non-retryable issue
            if err_code_str in fatal_protect_codes or any(msg in err_str_lower for msg in fatal_messages):
                 lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE protection setting error ({symbol}). Check parameters relative to current price/position. Aborting protection set.{RESET}")
                 return False # Fatal error for this operation

            # If error seems potentially temporary and retries remain
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries reached for ExchangeError setting protection ({symbol}).{RESET}")
                 return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            # Retry on network issues
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting protection ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached for NetworkError setting protection ({symbol}).{RESET}")
                return False

        except ccxt.RateLimitExceeded as e:
            # Wait and retry on rate limits
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit setting protection ({symbol}): {e}. Pausing {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
            # Authentication errors are critical and non-retryable
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error setting protection ({symbol}): {e}. Stopping.{RESET}")
            return False # Fatal error
        except Exception as e:
            # Handle any other unexpected errors
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting protection ({symbol}): {e}{RESET}", exc_info=True)
            return False # Stop on unexpected errors

        # Increment attempt counter and delay before retrying (only for retryable errors)
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # If loop completes without success
    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False

# --- Volumatic Trend + OB Strategy Implementation (Refactored into Class) ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic for calculating the Volumatic Trend and Pivot Order Block strategy indicators.

    Responsibilities:
    - Takes historical OHLCV data as input.
    - Calculates Volumatic Trend indicators:
        - EMA/SWMA based trend direction.
        - ATR (Average True Range).
        - Volatility Bands based on ATR calculated at trend changes.
        - Normalized Volume relative to its EMA.
    - Identifies Pivot Highs and Pivot Lows based on configured lookback periods.
    - Creates Order Blocks (OBs) from these pivots using the configured source (Wicks or Body).
    - Manages the state of Order Blocks: identifies new OBs, marks violated OBs, and optionally extends active OBs visually.
    - Prunes the list of active OBs to a configured maximum number per type (Bull/Bear) to manage performance/memory.
    - Returns structured analysis results (`StrategyAnalysisResults`) including the processed DataFrame with all indicators.

    Note: Indicator calculations are performed using float types via pandas-ta for performance,
          then results are converted back to Decimal where appropriate for the final DataFrame.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        """
        Initializes the strategy engine with parameters from the configuration.

        Args:
            config (Dict[str, Any]): The main configuration dictionary containing `strategy_params`.
            market_info (MarketInfo): The standardized `MarketInfo` dictionary for the symbol (used for symbol name).
            logger (logging.Logger): The logger instance dedicated to this strategy instance/symbol.

        Raises:
            ValueError: If critical configuration parameters under `strategy_params` are missing, invalid, or out of range.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger # Alias for convenience
        self.symbol = market_info.get('symbol', 'UnknownSymbol') # Get symbol for logging

        strategy_cfg = config.get("strategy_params", {}) # Get the strategy sub-dictionary

        # --- Load and Validate Strategy Parameters ---
        try:
            # These parameters should have been pre-validated by load_config, but re-check here
            self.vt_length = int(strategy_cfg["vt_length"])
            self.vt_atr_period = int(strategy_cfg["vt_atr_period"])
            self.vt_vol_ema_length = int(strategy_cfg["vt_vol_ema_length"])
            # Convert multiplier to Decimal for internal use if needed, though float is fine for ATR calc
            self.vt_atr_multiplier = Decimal(str(strategy_cfg["vt_atr_multiplier"]))

            self.ob_source = str(strategy_cfg["ob_source"]) # "Wicks" or "Body"
            self.ph_left = int(strategy_cfg["ph_left"])
            self.ph_right = int(strategy_cfg["ph_right"])
            self.pl_left = int(strategy_cfg["pl_left"])
            self.pl_right = int(strategy_cfg["pl_right"])
            self.ob_extend = bool(strategy_cfg["ob_extend"])
            self.ob_max_boxes = int(strategy_cfg["ob_max_boxes"])

            # Basic sanity checks on critical parameters
            if not (self.vt_length > 0 and self.vt_atr_period > 0 and self.vt_vol_ema_length > 0 and \
                    self.vt_atr_multiplier > 0 and self.ph_left > 0 and self.ph_right > 0 and \
                    self.pl_left > 0 and self.pl_right > 0 and self.ob_max_boxes > 0):
                raise ValueError("One or more strategy parameters are invalid (must be positive integers/decimals where applicable).")
            if self.ob_source not in ["Wicks", "Body"]:
                 raise ValueError(f"Invalid ob_source '{self.ob_source}'. Must be 'Wicks' or 'Body'.")

        except (KeyError, ValueError, TypeError, InvalidOperation) as e:
            # If parameters are missing or invalid despite prior validation (should not happen), raise critical error
            self.lg.critical(f"{NEON_RED}FATAL: Failed to initialize VolumaticOBStrategy ({self.symbol}) due to invalid config parameters: {e}{RESET}")
            self.lg.debug(f"Strategy Config received by class: {strategy_cfg}")
            raise ValueError(f"Strategy initialization failed ({self.symbol}): {e}") from e

        # Initialize Order Block storage (maintained within the instance across updates)
        # These lists store the history of identified OBs for the symbol.
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Calculate minimum historical data length required based on the longest lookback period used
        required_for_vt = max(
            self.vt_length * 2, # Use a buffer (e.g., 2x length) for EMA stabilization
            self.vt_atr_period,
            self.vt_vol_ema_length
        )
        required_for_pivots = max(
            self.ph_left + self.ph_right + 1, # Total candles needed to confirm a pivot
            self.pl_left + self.pl_right + 1
        )
        stabilization_buffer = 50 # General buffer for indicator calculations to settle
        self.min_data_len = max(required_for_vt, required_for_pivots) + stabilization_buffer

        # Log initialized parameters for this strategy instance
        self.lg.info(f"{NEON_CYAN}--- Initializing VolumaticOB Strategy Engine ({self.symbol}) ---{RESET}")
        self.lg.info(f"  VT Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Vol EMA Length={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  OB Params: Source='{self.ob_source}', PH Lookback (L/R)={self.ph_left}/{self.ph_right}, PL Lookback (L/R)={self.pl_left}/{self.pl_right}, Extend OBs={self.ob_extend}, Max Active OBs={self.ob_max_boxes}")
        self.lg.info(f"  Minimum Historical Data Recommended: ~{self.min_data_len} candles")

        # Warning if required data significantly exceeds typical API single request limits
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 50: # Check against limit + buffer
            self.lg.warning(
                f"{NEON_YELLOW}CONFIGURATION NOTE ({self.symbol}):{RESET} Strategy requires {self.min_data_len} candles, "
                f"which may exceed the API fetch limit ({BYBIT_API_KLINE_LIMIT}) per request. "
                f"Ensure 'fetch_limit' in config.json is sufficient (currently {self.config.get('fetch_limit', 'Default')}) "
                f"or consider reducing long lookback periods (e.g., vt_atr_period, vt_vol_ema_length)."
            )

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates EMA(SWMA(series, window=4), length) using float for performance.
        SWMA is a Smoothed Moving Average with weights [1, 2, 2, 1] / 6.
        """
        if not isinstance(series, pd.Series) or len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index, dtype=float) # Return NaNs if input invalid

        # Convert series to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # If all values are NaN after conversion, return NaNs
        if numeric_series.isnull().all():
            return pd.Series(np.nan, index=series.index, dtype=float)

        # Calculate SWMA(4) using rolling apply with specific weights
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        swma = numeric_series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate EMA of the SWMA result
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan) # Use pandas_ta EMA
        return ema_of_swma

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, is_high: bool) -> pd.Series:
        """
        Identifies Pivot Highs or Pivot Lows based on strict inequality comparison over lookback periods.
        Uses float comparison for performance.

        Args:
            series (pd.Series): The price series (e.g., high for PH, low for PL) as floats.
            left_bars (int): Number of bars to the left that must be lower (for PH) or higher (for PL).
            right_bars (int): Number of bars to the right that must be lower (for PH) or higher (for PL).
            is_high (bool): True to find Pivot Highs, False to find Pivot Lows.

        Returns:
            pd.Series: A boolean Series, True where a pivot is identified.
        """
        if not isinstance(series, pd.Series) or series.empty or left_bars < 1 or right_bars < 1:
            return pd.Series(False, index=series.index, dtype=bool) # Return all False if input invalid

        # Convert to numeric, coercing errors
        num_series = pd.to_numeric(series, errors='coerce')
        if num_series.isnull().all():
            return pd.Series(False, index=series.index, dtype=bool) # Return all False if only NaNs

        # Initialize pivot conditions with non-NaN check
        pivot_conditions = num_series.notna()

        # Check left bars: current bar must be strictly greater (PH) or smaller (PL) than N preceding bars
        for i in range(1, left_bars + 1):
            shifted = num_series.shift(i)
            # Define comparison based on is_high flag
            condition = (num_series > shifted) if is_high else (num_series < shifted)
            # Combine with existing conditions, treating NaNs in comparison as False
            pivot_conditions &= condition.fillna(False)

        # Check right bars: current bar must be strictly greater (PH) or smaller (PL) than N succeeding bars
        for i in range(1, right_bars + 1):
            shifted = num_series.shift(-i)
            # Define comparison based on is_high flag
            condition = (num_series > shifted) if is_high else (num_series < shifted)
            # Combine with existing conditions, treating NaNs in comparison as False
            pivot_conditions &= condition.fillna(False)

        # Return the final boolean series, filling any remaining NaNs with False
        return pivot_conditions.fillna(False)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes historical OHLCV data to calculate all strategy indicators and manage Order Blocks.

        Takes a DataFrame with Decimal OHLCV values, performs calculations (often using floats for speed),
        updates the internal state of Order Blocks, and returns a structured result.

        Args:
            df_input (pd.DataFrame): Input DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                                     containing Decimal values, and a DatetimeIndex (UTC).

        Returns:
            StrategyAnalysisResults: A TypedDict containing the processed DataFrame (with indicators added),
                                     current trend status, lists of active order blocks, and key indicator
                                     values for the last candle. Returns a default/empty result on critical failure.
        """
        # Prepare a default/empty result structure for early returns on failure
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('NaN'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if df_input.empty:
            self.lg.error(f"Strategy update failed ({self.symbol}): Input DataFrame is empty.")
            return empty_results

        # Work on a copy to avoid modifying the original DataFrame passed to the function
        df = df_input.copy()

        # --- Input Data Validation ---
        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
            self.lg.warning(f"Strategy update ({self.symbol}): Input DataFrame index is not a monotonic DatetimeIndex. Attempting to sort...")
            try:
                df.sort_index(inplace=True)
                if not df.index.is_monotonic_increasing: # Check again after sorting
                    raise ValueError("Index still not monotonic after sorting.")
            except Exception as sort_err:
                self.lg.error(f"Strategy update failed ({self.symbol}): Could not sort DataFrame index: {sort_err}")
                return empty_results

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            self.lg.error(f"Strategy update failed ({self.symbol}): Input DataFrame missing required columns: {missing_cols}.")
            return empty_results

        # Check if sufficient data length is available based on calculated minimum
        if len(df) < self.min_data_len:
            self.lg.warning(f"Strategy update ({self.symbol}): Insufficient data length provided ({len(df)} candles) "
                          f"compared to recommended minimum ({self.min_data_len}). Indicator results may be inaccurate.")
            # Proceed with calculation, but results might be unreliable, especially initial values

        self.lg.debug(f"Starting strategy analysis ({self.symbol}) on DataFrame with {len(df)} candles.")

        # --- Convert Input Decimal Data to Float for TA-Lib/Pandas-TA Performance ---
        # Most TA libraries work faster with floats. Perform calculations with floats, then convert back where needed.
        try:
            df_float = pd.DataFrame(index=df.index) # Create a new DataFrame for float calculations
            for col in required_cols:
                # Convert Decimal column to numeric (float), coercing errors to NaN
                df_float[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where essential OHLC float values became NaN (due to bad input data)
            initial_float_len = len(df_float)
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            rows_dropped_float = initial_float_len - len(df_float)
            if rows_dropped_float > 0:
                 self.lg.debug(f"Dropped {rows_dropped_float} rows ({self.symbol}) during float conversion due to NaN OHLC values.")

            # If DataFrame becomes empty after dropping NaNs, cannot proceed
            if df_float.empty:
                self.lg.error(f"Strategy update failed ({self.symbol}): DataFrame became empty after converting OHLC to float and dropping NaNs.")
                return empty_results
        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error during conversion to float: {e}", exc_info=True)
            return empty_results

        # --- Indicator Calculations (using df_float) ---
        try:
            self.lg.debug(f"Calculating technical indicators ({self.symbol}) using float data...")
            # 1. ATR (Average True Range)
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)

            # 2. Volumatic Trend EMAs
            # ema1: Smoothed EMA (EMA of SWMA)
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)
            # ema2: Standard EMA
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)

            # 3. Determine Trend Direction
            # Trend is UP if current ema2 crosses *above* the *previous* ema1
            # Need valid values for both current ema2 and previous ema1
            valid_trend_comparison = df_float['ema2'].notna() & df_float['ema1'].shift(1).notna()
            # Initialize trend series with NaN
            trend_up_series = pd.Series(np.nan, index=df_float.index, dtype=object) # Use object to allow NaN/True/False
            # Calculate trend where comparison is valid
            trend_up_series.loc[valid_trend_comparison] = df_float['ema2'] > df_float['ema1'].shift(1)
            # Forward fill the trend to propagate the state until the next valid calculation
            trend_up_series.ffill(inplace=True)
            # Final boolean trend series (NaNs remain where trend couldn't be determined initially)
            df_float['trend_up'] = trend_up_series.astype('boolean') # Use nullable boolean type

            # 4. Identify Trend Changes
            # A change occurs if the current trend is different from the previous trend, and both are valid (not NaN)
            trend_changed_series = (df_float['trend_up'] != df_float['trend_up'].shift(1)) & \
                                   df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'] = trend_changed_series.fillna(False).astype(bool)

            # 5. Calculate Volumatic Trend Bands
            # Capture EMA1 and ATR values *at the exact candle where the trend changed*
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)
            # Forward fill these captured values to use them for band calculation until the next trend change
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()
            # Calculate upper and lower bands using the forward-filled values and the ATR multiplier
            atr_multiplier_float = float(self.vt_atr_multiplier) # Use float multiplier with float indicators
            valid_band_calc = df_float['ema1_for_bands'].notna() & df_float['atr_for_bands'].notna() & (df_float['atr_for_bands'] > 0)
            df_float['upper_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_multiplier_float), np.nan)
            df_float['lower_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_multiplier_float), np.nan)

            # 6. Volume Normalization
            # Calculate EMA of volume
            vol_ema = ta.ema(df_float['volume'].fillna(0.0), length=self.vt_vol_ema_length, fillna=np.nan)
            # Avoid division by zero or near-zero EMA: replace 0 with NaN, backfill, then fill remaining NaNs with a tiny number
            vol_ema_safe = vol_ema.replace(0, np.nan).fillna(method='bfill').fillna(1e-9)
            # Calculate normalized volume = (Current Volume / Volume EMA) * 100
            df_float['vol_norm'] = (df_float['volume'].fillna(0.0) / vol_ema_safe) * 100.0
            # Clip normalized volume (e.g., between 0% and 200%) and convert to integer for simpler interpretation/use
            df_float['vol_norm_int'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0).astype(int)

            # 7. Pivot High/Low Identification
            # Select the source series based on config ('Wicks' or 'Body')
            high_series = df_float['high'] if self.ob_source == "Wicks" else df_float[['open', 'close']].max(axis=1)
            low_series = df_float['low'] if self.ob_source == "Wicks" else df_float[['open', 'close']].min(axis=1)
            # Find pivots using the helper function
            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, is_high=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, is_high=False)

            self.lg.debug(f"Indicator calculations complete ({self.symbol}) using float data.")

        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error during indicator calculation: {e}", exc_info=True)
            # Attempt to return partial results if possible, otherwise empty
            empty_results['dataframe'] = df # Return original df if calc failed
            return empty_results

        # --- Copy Calculated Float Results back to Original Decimal DataFrame ---
        # Reindex calculated series to match the original Decimal DataFrame's index (handles rows dropped during float conversion)
        # Convert float indicators back to Decimal where precision is desired (e.g., prices, ATR)
        try:
            self.lg.debug(f"Converting calculated indicators back to Decimal format ({self.symbol})...")
            # Columns to convert back to Decimal (handle potential NaNs)
            indicator_cols_decimal = ['atr', 'ema1', 'ema2', 'upper_band', 'lower_band', 'vol_norm']
            # Columns to keep as integer
            indicator_cols_int = ['vol_norm_int']
            # Columns to keep as boolean (or nullable boolean for trend_up)
            indicator_cols_bool = ['trend_up', 'trend_changed', 'is_ph', 'is_pl']

            for col in indicator_cols_decimal:
                if col in df_float.columns:
                    # Reindex to match original df index, then convert valid floats to Decimal
                    source_series = df_float[col].reindex(df.index)
                    df[col] = source_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
                else: self.lg.warning(f"Indicator column '{col}' not found in float df ({symbol}).")

            for col in indicator_cols_int:
                if col in df_float.columns:
                     source_series = df_float[col].reindex(df.index)
                     # Fill NaN with 0 before converting to int
                     df[col] = source_series.fillna(0).astype(int)
                else: self.lg.warning(f"Indicator column '{col}' not found in float df ({symbol}).")

            for col in indicator_cols_bool:
                 if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)
                    # Use nullable boolean for trend_up to preserve None state
                    if col == 'trend_up':
                        df[col] = source_series.astype('boolean')
                    else: # Other bools can default NaN to False
                        df[col] = source_series.fillna(False).astype(bool)
                 else: self.lg.warning(f"Indicator column '{col}' not found in float df ({symbol}).")

        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error converting calculated indicators back to Decimal/target types: {e}", exc_info=True)
            # Return the DataFrame as is, but results might be inconsistent
            empty_results['dataframe'] = df
            return empty_results

        # --- Clean Final Decimal DataFrame ---
        # Drop rows at the beginning where indicators couldn't be calculated (contain NaNs)
        initial_len_final = len(df)
        # Define essential columns that must have valid values for the strategy to function
        essential_cols = ['close', 'atr', 'trend_up', 'is_ph', 'is_pl'] # Trend must be determined (not None)
        df.dropna(subset=essential_cols, inplace=True)
        # Also ensure ATR is positive (it should be, but check defensively)
        if 'atr' in df.columns:
            df = df[df['atr'] > Decimal('0')]

        rows_dropped_final = initial_len_final - len(df)
        if rows_dropped_final > 0:
            self.lg.debug(f"Dropped {rows_dropped_final} initial rows ({self.symbol}) from final DataFrame due to missing essential indicators (NaNs).")

        if df.empty:
            self.lg.warning(f"Strategy update ({self.symbol}): DataFrame became empty after final indicator cleaning (dropping initial NaNs).")
            empty_results['dataframe'] = df # Return the empty df
            return empty_results

        self.lg.debug(f"Indicators finalized in Decimal DataFrame ({self.symbol}). Processing Order Blocks...")

        # --- Order Block (OB) Management ---
        # This section updates the persistent lists self.bull_boxes and self.bear_boxes
        try:
            new_ob_count = 0
            violated_ob_count = 0
            extended_ob_count = 0
            # Get the timestamp of the last candle in the current DataFrame
            last_candle_ts = df.index[-1]

            # --- Identify New Order Blocks from Pivots in the DataFrame ---
            new_bull_candidates: List[OrderBlock] = []
            new_bear_candidates: List[OrderBlock] = []
            # Create a set of existing OB IDs to avoid recreating them if pivots persist
            existing_ob_ids = {ob['id'] for ob in self.bull_boxes + self.bear_boxes}

            # Iterate through the DataFrame to find pivot points and create OBs
            for timestamp, candle in df.iterrows():
                # Create Bearish OB from Pivot High (PH)
                if candle.get('is_ph'):
                    # Generate a unique ID based on type and timestamp
                    ob_id = f"B_{timestamp.strftime('%y%m%d%H%M%S')}" # 'B' for Bearish
                    # Check if this OB already exists in our persistent lists
                    if ob_id not in existing_ob_ids:
                        # Determine OB boundaries based on config ('Wicks' or 'Body')
                        ob_top = candle['high'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                        # For Bearish OB from Wick source, bottom is typically the candle's open price
                        ob_bottom = candle['open'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                        # Ensure boundaries are valid Decimals and top > bottom
                        if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom:
                            new_bear_candidates.append(OrderBlock(
                                id=ob_id, type='bear', timestamp=timestamp, top=ob_top, bottom=ob_bottom,
                                active=True, violated=False, violation_ts=None, extended_to_ts=timestamp # Initially extends only to its own candle
                            ))
                            new_ob_count += 1
                            existing_ob_ids.add(ob_id) # Add immediately to prevent duplicates within this run

                # Create Bullish OB from Pivot Low (PL)
                if candle.get('is_pl'):
                    ob_id = f"L_{timestamp.strftime('%y%m%d%H%M%S')}" # 'L' for Bullish (Low pivot)
                    if ob_id not in existing_ob_ids:
                        # For Bullish OB from Wick source, top is typically the candle's open price
                        ob_top = candle['open'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                        ob_bottom = candle['low'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                        if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom:
                            new_bull_candidates.append(OrderBlock(
                                id=ob_id, type='bull', timestamp=timestamp, top=ob_top, bottom=ob_bottom,
                                active=True, violated=False, violation_ts=None, extended_to_ts=timestamp
                            ))
                            new_ob_count += 1
                            existing_ob_ids.add(ob_id)

            # Add the newly identified OB candidates to the persistent lists
            self.bull_boxes.extend(new_bull_candidates)
            self.bear_boxes.extend(new_bear_candidates)
            if new_ob_count > 0:
                self.lg.debug(f"Identified {new_ob_count} new potential Order Blocks ({self.symbol}).")

            # --- Check Existing and New OBs for Violations and Extend Active Ones ---
            # Iterate through all currently tracked boxes (existing + new)
            all_boxes = self.bull_boxes + self.bear_boxes
            active_boxes_after_update: List[OrderBlock] = [] # List to store boxes still active after this update

            for box in all_boxes:
                 # Skip boxes that were already marked inactive in previous runs
                 if not box['active']: continue

                 # Find candles in the current DataFrame that occurred *after* the box was formed
                 relevant_candles = df[df.index > box['timestamp']]
                 box_violated_in_this_run = False

                 for ts, candle in relevant_candles.iterrows():
                      close_price = candle.get('close')
                      # Ensure close price is a valid Decimal
                      if isinstance(close_price, Decimal) and close_price.is_finite():
                           # Check for violation: close price goes beyond the OB boundary
                           violation_condition = False
                           if box['type'] == 'bull' and close_price < box['bottom']: violation_condition = True
                           elif box['type'] == 'bear' and close_price > box['top']: violation_condition = True

                           if violation_condition:
                                # Mark the box as inactive and record violation time
                                box['active'] = False
                                box['violated'] = True
                                box['violation_ts'] = ts
                                violated_ob_count += 1
                                self.lg.debug(f"{box['type'].capitalize()} OB {box['id']} ({self.symbol}) VIOLATED at {ts.strftime('%Y-%m-%d %H:%M')} by close {close_price.normalize()}")
                                box_violated_in_this_run = True
                                break # Stop checking this box once violated
                           # If not violated and extend is enabled, update the extended timestamp
                           elif self.ob_extend:
                                box['extended_to_ts'] = ts
                                extended_ob_count += 1 # Count extensions (can be many per box)
                      # else: self.lg.warning(f"Invalid close price at {ts} for OB check ({symbol}).")

                 # If the box survived the violation checks in this run, add it to the list of active boxes
                 if not box_violated_in_this_run and box['active']:
                      # If extending, ensure the final extension timestamp is the last candle's timestamp
                      if self.ob_extend: box['extended_to_ts'] = last_candle_ts
                      active_boxes_after_update.append(box)

            if violated_ob_count > 0:
                 self.lg.debug(f"Marked {violated_ob_count} Order Blocks as violated ({self.symbol}).")
            # if extended_ob_count > 0 and self.ob_extend:
            #      self.lg.debug(f"Extended active Order Blocks to {last_candle_ts.strftime('%Y-%m-%d %H:%M')} ({symbol}).") # Can be verbose

            # Update the main lists to contain only boxes that are still active after this update
            self.bull_boxes = [b for b in active_boxes_after_update if b['type'] == 'bull']
            self.bear_boxes = [b for b in active_boxes_after_update if b['type'] == 'bear']

            # --- Prune Order Block Lists ---
            # Keep only the 'ob_max_boxes' most *recent* active ones per type to limit memory usage
            # Sort active boxes by timestamp (descending) and take the top N
            self.bull_boxes = sorted(self.bull_boxes, key=lambda b: b['timestamp'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted(self.bear_boxes, key=lambda b: b['timestamp'], reverse=True)[:self.ob_max_boxes]
            self.lg.debug(f"Pruned active Order Blocks ({self.symbol}). Kept newest: Bulls={len(self.bull_boxes)}, Bears={len(self.bear_boxes)} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.lg.error(f"Strategy update failed ({self.symbol}): Error during Order Block processing: {e}", exc_info=True)
            # Continue, but OBs might be inaccurate or missing

        # --- Prepare Final StrategyAnalysisResults ---
        # Get the last row of the processed DataFrame (should exist after cleaning checks)
        last_candle_final = df.iloc[-1] if not df.empty else None

        # Helper functions to safely extract values from the last candle row
        def safe_decimal_from_candle(col_name: str, positive_only: bool = False) -> Optional[Decimal]:
            """Safely extracts a Decimal value, returns None if missing, NaN, or invalid."""
            if last_candle_final is None: return None
            value = last_candle_final.get(col_name)
            if isinstance(value, Decimal) and value.is_finite():
                 # Return value if check passes, otherwise None
                 return value if not positive_only or value > Decimal('0') else None
            return None

        def safe_bool_from_candle(col_name: str) -> Optional[bool]:
            """Safely extracts a boolean value, handles pandas nullable boolean, returns None if missing/NaN."""
            if last_candle_final is None: return None
            value = last_candle_final.get(col_name)
            # Check for pandas nullable boolean NA or standard None/NaN
            return bool(value) if pd.notna(value) else None

        def safe_int_from_candle(col_name: str) -> Optional[int]:
             """Safely extracts an integer value, returns None if missing or conversion fails."""
             if last_candle_final is None: return None
             value = last_candle_final.get(col_name)
             try:
                 # Convert to int only if value is not NaN/None
                 return int(value) if pd.notna(value) else None
             except (ValueError, TypeError):
                 return None

        # Construct the final results dictionary using safe extraction
        final_dataframe = df # Return the fully processed DataFrame
        last_close_val = safe_decimal_from_candle('close') or Decimal('NaN') # Use NaN if invalid/missing
        current_trend_val = safe_bool_from_candle('trend_up') # Can be True, False, or None
        trend_changed_val = bool(safe_bool_from_candle('trend_changed')) # Default to False if missing/NaN
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

        # Log summary of the final results for the *last* candle for quick review
        trend_str = f"{NEON_GREEN}UP{RESET}" if analysis_results['current_trend_up'] is True else \
                    f"{NEON_RED}DOWN{RESET}" if analysis_results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}Undetermined{RESET}"
        atr_str = f"{analysis_results['atr'].normalize()}" if analysis_results['atr'] else "N/A"
        time_str = last_candle_final.name.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_final is not None else "N/A"

        self.lg.debug(f"--- Strategy Analysis Results ({self.symbol} @ {time_str}) ---")
        self.lg.debug(f"  Last Close: {analysis_results['last_close'].normalize() if analysis_results['last_close'].is_finite() else 'NaN'}")
        self.lg.debug(f"  Trend: {trend_str} (Trend Changed on Last Candle: {analysis_results['trend_just_changed']})")
        self.lg.debug(f"  ATR: {atr_str}")
        self.lg.debug(f"  Volume Norm (% EMA): {analysis_results['vol_norm_int']}")
        self.lg.debug(f"  VT Bands (Upper/Lower): {analysis_results['upper_band'].normalize() if analysis_results['upper_band'] else 'N/A'} / {analysis_results['lower_band'].normalize() if analysis_results['lower_band'] else 'N/A'}")
        self.lg.debug(f"  Active OBs (Bull/Bear): {len(analysis_results['active_bull_boxes'])} / {len(analysis_results['active_bear_boxes'])}")
        # Optionally log the details of the active OBs at DEBUG level if needed for deep debugging
        # for ob in analysis_results['active_bull_boxes']: self.lg.debug(f"    Bull OB: {ob['id']} [{ob['bottom'].normalize()} - {ob['top'].normalize()}] Active: {ob['active']}")
        # for ob in analysis_results['active_bear_boxes']: self.lg.debug(f"    Bear OB: {ob['id']} [{ob['bottom'].normalize()} - {ob['top'].normalize()}] Active: {ob['active']}")
        self.lg.debug(f"---------------------------------------------")

        return analysis_results

# --- Signal Generation based on Strategy Results (Refactored into Class) ---
class SignalGenerator:
    """
    Generates trading signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD")
    based on the results from the `VolumaticOBStrategy` and the current position state.

    Responsibilities:
    - Evaluates `StrategyAnalysisResults` against configured entry and exit rules.
    - Considers the current open position (if any) provided as input.
    - Entry Rules: Trend alignment + Price proximity to a relevant active Order Block.
    - Exit Rules: Trend reversal (on the last candle) OR Price proximity violation of an opposite Order Block.
    - Calculates initial Stop Loss (SL) and Take Profit (TP) levels for potential new entries based on ATR multiples.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the Signal Generator with parameters from the configuration.

        Args:
            config (Dict[str, Any]): The main configuration dictionary containing `strategy_params` and `protection` sections.
            logger (logging.Logger): The logger instance for signal generation messages.

        Raises:
            ValueError: If critical configuration parameters related to signal generation or SL/TP calculation
                        are missing, invalid, or out of range.
        """
        self.config = config
        self.logger = logger
        self.lg = logger # Alias for convenience
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})

        try:
            # Load parameters used for signal generation rules and SL/TP calculation
            # These should have been validated by load_config, but check type/range again
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))

            # Basic validation of loaded parameters
            if not (self.ob_entry_proximity_factor >= 1): raise ValueError("ob_entry_proximity_factor must be >= 1.0")
            if not (self.ob_exit_proximity_factor >= 1): raise ValueError("ob_exit_proximity_factor must be >= 1.0")
            if not (self.initial_tp_atr_multiple >= 0): raise ValueError("initial_take_profit_atr_multiple must be >= 0 (0 disables TP)")
            if not (self.initial_sl_atr_multiple > 0): raise ValueError("initial_stop_loss_atr_multiple must be strictly > 0")

            self.lg.info(f"{NEON_CYAN}--- Initializing Signal Generator ---{RESET}")
            self.lg.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor.normalize()}")
            self.lg.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor.normalize()}")
            self.lg.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()}")
            self.lg.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")
            self.lg.info(f"-----------------------------------")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             # If parameters are missing or invalid despite prior validation (should not happen)
             self.lg.critical(f"{NEON_RED}FATAL: Error initializing SignalGenerator parameters from config: {e}.{RESET}", exc_info=True)
             raise ValueError(f"SignalGenerator initialization failed: {e}") from e

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[PositionInfo], symbol: str) -> str:
        """
        Determines the trading signal based on strategy analysis results and the current position state.

        Logic Flow:
        1. Validate essential inputs from `analysis_results` (trend, close price, ATR).
        2. If a position exists: Check for Exit conditions (trend flip on last candle, or price violating opposing OB).
        3. If no position exists: Check for Entry conditions (trend aligned with signal, price within relevant OB proximity).
        4. If neither entry nor exit conditions are met, default to "HOLD".

        Args:
            analysis_results (StrategyAnalysisResults): The results dictionary from `VolumaticOBStrategy.update()`.
            open_position (Optional[PositionInfo]): Standardized `PositionInfo` dict if a position is open, otherwise None.
            symbol (str): The trading symbol (used for logging).

        Returns:
            
